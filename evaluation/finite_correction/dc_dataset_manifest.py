#!/usr/bin/env python3
"""Build or verify a byte-level manifest for one LoDoPaB evaluation split.

Each manifest contains validation *or* test, never both.  The finite-DC study
reads only the official validation and test observations and ground truths;
patient maps are authenticated separately by ``expected_assets.json``.

The dataset root itself may be a site-specific path or a symlink.  Identity is
defined only by the 56 expected regular HDF5 files for the requested split,
their lexical basenames, byte sizes, and SHA-256 digests.  No HDF5 array is
opened by this utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "dc.lodopab_file_manifest.v1"
SPLITS = ("validation", "test")
KINDS = ("ground_truth", "observation")
FILES_PER_KIND = 28


class DatasetManifestError(RuntimeError):
    """Raised when LoDoPaB file identity or manifest structure changes."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if not path.is_file() or path.is_symlink():
        raise DatasetManifestError(f"not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_basenames(split: str) -> tuple[str, ...]:
    if split not in SPLITS:
        raise DatasetManifestError(f"unsupported split: {split}")
    return tuple(
        f"{kind}_{split}_{index:03d}.hdf5"
        for kind in KINDS
        for index in range(FILES_PER_KIND)
    )


def _root_sha256(entries: Iterable[dict[str, Any]]) -> str:
    """Hash canonical ``basename\0size\0sha256\n`` leaves in lexical order."""

    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["basename"])):
        leaf = (
            f"{entry['basename']}\0{int(entry['byte_size'])}\0"
            f"{entry['sha256']}\n"
        ).encode("utf-8")
        digest.update(leaf)
    return digest.hexdigest()


def build_split(root: Path, split: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for basename in expected_basenames(split):
        path = root / basename
        entries.append(
            {
                "basename": basename,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "file_count": len(entries),
        "byte_size": sum(int(item["byte_size"]) for item in entries),
        "root_sha256": _root_sha256(entries),
        "entries": sorted(entries, key=lambda item: str(item["basename"])),
    }


def build_manifest(root: Path, split: str) -> dict[str, Any]:
    if not root.is_dir():
        raise DatasetManifestError(f"dataset root is not a directory: {root}")
    if split not in SPLITS:
        raise DatasetManifestError(f"unsupported split: {split}")
    return {
        "schema": SCHEMA,
        "identity_scope": (
            f"LoDoPaB-CT v1.0.0 official {split} ground-truth and observation "
            "HDF5 files"
        ),
        "hash_algorithm": "sha256",
        "root_algorithm": "SHA256 over lexical basename\\0byte_size\\0file_sha256\\n leaves",
        "expected_files_per_split": FILES_PER_KIND * len(KINDS),
        "split": split,
        "files": build_split(root, split),
    }


def _require_sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise DatasetManifestError(f"invalid SHA-256 at {where}")
    return value


def validate_manifest(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise DatasetManifestError("dataset manifest must be an object")
    expected_top = {
        "schema",
        "identity_scope",
        "hash_algorithm",
        "root_algorithm",
        "expected_files_per_split",
        "split",
        "files",
    }
    if set(document) != expected_top:
        raise DatasetManifestError("dataset manifest top-level fields changed")
    if document["schema"] != SCHEMA or document["hash_algorithm"] != "sha256":
        raise DatasetManifestError("unsupported dataset manifest schema/hash")
    if document["expected_files_per_split"] != FILES_PER_KIND * len(KINDS):
        raise DatasetManifestError("expected file count changed")
    split = document.get("split")
    if split not in SPLITS:
        raise DatasetManifestError("manifest split must be validation or test")
    value = document.get("files")
    if not isinstance(value, dict) or set(value) != {
        "file_count",
        "byte_size",
        "root_sha256",
        "entries",
    }:
        raise DatasetManifestError(f"invalid split manifest: {split}")
    entries = value["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected_basenames(split)):
        raise DatasetManifestError(f"wrong entry count for {split}")
    if value["file_count"] != len(entries):
        raise DatasetManifestError(f"file_count mismatch for {split}")
    observed_names: list[str] = []
    observed_bytes = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "basename",
            "byte_size",
            "sha256",
        }:
            raise DatasetManifestError(f"invalid {split} entry {index}")
        basename = entry["basename"]
        if not isinstance(basename, str):
            raise DatasetManifestError(f"invalid basename at {split}/{index}")
        byte_size = entry["byte_size"]
        if type(byte_size) is not int or byte_size <= 0:
            raise DatasetManifestError(f"invalid byte size at {split}/{basename}")
        _require_sha256(entry["sha256"], f"{split}/{basename}")
        observed_names.append(basename)
        observed_bytes += byte_size
    if observed_names != list(expected_basenames(split)):
        raise DatasetManifestError(f"file order/names changed for {split}")
    if value["byte_size"] != observed_bytes:
        raise DatasetManifestError(f"aggregate byte size mismatch for {split}")
    if value["root_sha256"] != _root_sha256(entries):
        raise DatasetManifestError(f"root hash mismatch for {split}")
    return document


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetManifestError(f"cannot read dataset manifest: {path}") from exc
    return validate_manifest(document)


def verify_split(root: Path, manifest: dict[str, Any], split: str) -> dict[str, Any]:
    validate_manifest(manifest)
    if manifest["split"] != split:
        raise DatasetManifestError(
            f"manifest split {manifest['split']} cannot authorize {split}"
        )
    observed = build_split(root, split)
    expected = manifest["files"]
    if observed != expected:
        raise DatasetManifestError(f"LoDoPaB {split} file identity mismatch")
    return {
        "schema": "dc.lodopab_split_verification.v1",
        "status": "VERIFIED",
        "split": split,
        "file_count": observed["file_count"],
        "byte_size": observed["byte_size"],
        "root_sha256": observed["root_sha256"],
    }


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DatasetManifestError(f"refusing to overwrite: {path}")
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--data", type=Path, required=True)
    build.add_argument("--split", choices=SPLITS, required=True)
    build.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--data", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--split", choices=SPLITS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        document = build_manifest(args.data, args.split)
        validate_manifest(document)
        atomic_write_json(args.out, document)
        print(args.out)
        return 0
    document = load_manifest(args.manifest)
    print(json.dumps(verify_split(args.data, document, args.split), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
