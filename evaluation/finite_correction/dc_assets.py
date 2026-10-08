#!/usr/bin/env python3
"""Cryptographic asset resolution for the frozen DC protocol.

Logical locators in ``expected_assets.json`` are deliberately not paths.  A
caller must explicitly bind an asset ID to a local file.  This prevents the
validation/lock phase from accidentally expanding a server path or opening
test arrays.  Assets whose identity has not been independently established
remain unresolved and cannot be promoted by this module.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from dc_schemas import (
    MANIFEST_SCHEMA,
    SchemaError,
    atomic_write_json,
    json_sha256,
    load_json,
    validate_manifest,
)


PathLike = Union[str, os.PathLike]
DEFAULT_VALIDATION_SPLITS: Tuple[str, ...] = ("shared", "validation")


class AssetError(RuntimeError):
    """Base class for asset-resolution failures."""


class UnresolvedAssetError(AssetError):
    """Raised when an asset has no independently verified identity."""


class AssetHashMismatch(AssetError):
    """Raised when bytes do not match the expected SHA-256."""


class SplitFirewallViolation(AssetError):
    """Raised when validation/lock code is asked to resolve a test asset."""


def load_expected_assets(path: PathLike) -> Dict[str, Any]:
    document = load_json(path)
    validate_manifest(document)
    return document


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return json_sha256(manifest)


def asset_index(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    validate_manifest(manifest)
    return {asset["asset_id"]: asset for asset in manifest["assets"]}


def unresolved_assets(
    manifest: Mapping[str, Any], splits: Optional[Iterable[str]] = None
) -> Tuple[str, ...]:
    index = asset_index(manifest)
    allowed = None if splits is None else set(splits)
    return tuple(
        asset_id
        for asset_id, asset in index.items()
        if not asset["resolved"] and (allowed is None or asset["split"] in allowed)
    )


def sha256_file(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    candidate = Path(path)
    if candidate.is_symlink():
        raise AssetError("symlink asset paths are forbidden: %s" % candidate)
    if not candidate.is_file():
        raise AssetError("asset is not a regular file: %s" % candidate)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_asset(asset: Mapping[str, Any], path: PathLike) -> Dict[str, Any]:
    """Verify one already-resolved manifest asset against explicit bytes."""

    if not isinstance(asset, Mapping):
        raise AssetError("asset entry must be an object")
    asset_id = asset.get("asset_id", "<unknown>")
    if asset.get("resolved") is not True:
        raise UnresolvedAssetError(
            "asset %s is unresolved; update the audited manifest, do not invent a hash"
            % asset_id
        )
    expected_hash = asset.get("sha256")
    expected_size = asset.get("byte_size")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise AssetError("asset %s has no valid expected SHA-256" % asset_id)
    if type(expected_size) is not int or expected_size <= 0:
        raise AssetError("asset %s has no valid expected byte size" % asset_id)
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise AssetError("asset %s is not a regular non-symlink file" % asset_id)
    observed_size = candidate.stat().st_size
    if observed_size != expected_size:
        raise AssetHashMismatch(
            "asset %s size mismatch: expected %d, observed %d"
            % (asset_id, expected_size, observed_size)
        )
    observed_hash = sha256_file(candidate)
    if observed_hash != expected_hash:
        raise AssetHashMismatch(
            "asset %s SHA-256 mismatch: expected %s, observed %s"
            % (asset_id, expected_hash, observed_hash)
        )
    return {
        "asset_id": asset_id,
        "status": "VERIFIED",
        "path": str(candidate.resolve()),
        "byte_size": observed_size,
        "sha256": observed_hash,
    }


def verify_assets(
    manifest: Mapping[str, Any],
    bindings: Mapping[str, PathLike],
    required_ids: Optional[Sequence[str]] = None,
    allowed_splits: Iterable[str] = DEFAULT_VALIDATION_SPLITS,
) -> Dict[str, Any]:
    """Verify explicitly bound inputs without crossing the split firewall.

    The default phase is validation/lock construction.  It accepts only
    ``shared`` and ``validation`` assets and never touches a binding for a test
    asset.  A future locked-test runner may call this function with an explicit
    test-only allow-list after validating an immutable lock; that policy is
    intentionally outside this validation/lock module.
    """

    validate_manifest(manifest)
    index = asset_index(manifest)
    allowed = set(allowed_splits)
    if not allowed or not allowed.issubset({"shared", "validation", "test", "server"}):
        raise AssetError("allowed_splits is empty or invalid")
    ids = list(required_ids) if required_ids is not None else [
        asset_id for asset_id, asset in index.items() if asset["split"] in allowed
    ]
    if len(ids) != len(set(ids)):
        raise AssetError("required_ids contains duplicates")

    reports = []
    for asset_id in ids:
        if asset_id not in index:
            raise AssetError("unknown required asset_id: %s" % asset_id)
        asset = index[asset_id]
        if asset["split"] not in allowed:
            raise SplitFirewallViolation(
                "asset %s has split %s, outside allowed splits %s"
                % (asset_id, asset["split"], sorted(allowed))
            )
        if asset["resolved"] is not True:
            raise UnresolvedAssetError(
                "required asset %s is unresolved: %s"
                % (asset_id, asset.get("unresolved_reason", "no reason recorded"))
            )
        if asset_id not in bindings:
            raise AssetError("no explicit path binding for asset %s" % asset_id)
        reports.append(verify_asset(asset, bindings[asset_id]))

    return {
        "schema": MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256(manifest),
        "allowed_splits": sorted(allowed),
        "verified_assets": reports,
        "verified_count": len(reports),
        "test_assets_read": any(index[item["asset_id"]]["split"] == "test" for item in reports),
    }


def write_manifest(path: PathLike, manifest: Mapping[str, Any], overwrite: bool = False) -> Path:
    validate_manifest(manifest)
    return atomic_write_json(path, manifest, overwrite=overwrite)


__all__ = [
    "AssetError",
    "AssetHashMismatch",
    "DEFAULT_VALIDATION_SPLITS",
    "SplitFirewallViolation",
    "UnresolvedAssetError",
    "asset_index",
    "load_expected_assets",
    "manifest_sha256",
    "sha256_file",
    "unresolved_assets",
    "verify_asset",
    "verify_assets",
    "write_manifest",
]
