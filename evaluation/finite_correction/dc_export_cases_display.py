#!/usr/bin/env python3
"""AMENDMENT 5 — display-only, case-locked 125-view qualitative export.

Purpose
-------
Figure 4 of the manuscript needs the 125-view reconstructions (reference,
FBP, network output, network + finite DC) for the same three outcome-locked
test cases that the frozen 50-view ``export-cases`` command produced.  The
frozen runner refuses any view other than 50 for that command, and every
frozen chain file is byte-locked by the COMPLETE 125-view test manifest, so
the constraint cannot be lifted inside the runner without invalidating the
manifest authentication chain.

This script therefore lifts exactly one constraint — "qualitative export is
frozen to the 50-view test analysis" — in a separate, additive program, and
only for a **display-only, case-locked** 125-view export:

* zero bytes of the frozen chain (``dc_sparseview_runner.py``,
  ``dc_qualitative.py``, protocol, schemas, ...) are modified; the chain is
  re-hashed and must equal the pinned digests that produced both the 125-view
  COMPLETE test manifest and the 50-view FINAL export;
* the three case IDs are **inherited** from the authenticated 50-view case
  lock and its FINAL export manifest — they are never re-selected on the
  125-view outcomes, and the IDs are additionally pinned in this file;
* the 125-view checkpoints, validation-selected lock (k = 4), full test grid,
  k = 0 parity canary, prediction memmaps, FBP cache, and operator identity
  are authenticated exactly as the frozen 50-view export authenticates them
  (same functions, view parameter 125);
* no training, no checkpoint change, no metric recomputation for reporting:
  the 18 panel metrics are re-derived only as a parity canary against the
  frozen 125-view test CSV (same tolerances as the frozen export);
* the output manifest is ``run_kind = "DISPLAY_ONLY"`` and carries the
  amendment and provenance record, so the frozen Figure C renderer and every
  FINAL-only consumer reject it by construction.

Frozen-module reuse: every view-independent routine is imported from the
frozen runner.  The four view-dependent routines (prediction-asset binding,
operator identity, output writer, and the ``export-cases`` body itself) are
re-implemented here with an explicit ``views`` parameter; their frozen
originals hard-code 50.  ``dc_qualitative.PRIMARY_VIEW`` is overridden at
runtime, inside a guarded context, for the duration of the 125-view test-grid
authentication — the same attribute-injection technique as AMENDMENT 4, with
the frozen bytes untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import dc_qualitative  # noqa: E402
import dc_sparseview_runner as runner  # noqa: E402
from dc_assets import sha256_file  # noqa: E402
from dc_core import (  # noqa: E402
    BORDER,
    CROP_SIZE,
    FULL_SIZE,
    crop_canvas,
    dc_definition,
    embed_crop,
    finite_dc_trajectory,
    projection_residual,
)
from dc_metrics import metric_definition  # noqa: E402
from dc_schemas import ITERATIONS, json_sha256, load_json  # noqa: E402


AMENDMENT_ID = "AMENDMENT 5"
AMENDMENT_DOCUMENT = "AMENDMENT_5_display_only_125view_export.md"
DISPLAY_VIEWS = 125
PRIMARY_VIEWS = 50
DISPLAY_RUN_KIND = "DISPLAY_ONLY"
EXPECTED_SELECTED_K = 4
SINOGRAM_BINS = 513

# The three outcome-locked cases of the 50-view FINAL export, in locked order.
LOCKED_CASES: tuple[tuple[str, float, int], ...] = (
    ("low_or_failure", 0.10, 1564),
    ("median", 0.50, 2340),
    ("high_benefit", 0.90, 3162),
)
LOCKED_CASE_IDS = tuple(item[2] for item in LOCKED_CASES)
LOCKED_CASE_LABELS = tuple(item[0] for item in LOCKED_CASES)

# Byte identity of the frozen chain that produced the 125-view COMPLETE test
# manifest (run adbacb5b…) and the 50-view FINAL export (run 0f61a38d…).
FROZEN_CHAIN_SHA256 = {
    "dc_assets.py": "5cf18d0b6243b38fc1a5497d807fa19c0af7da65d603c3d95d2adab4c7a9c782",
    "dc_core.py": "d2f84a9fec416d71968221d63a9d320e747a6ddcaf304f023670abc4dce7a461",
    "dc_dataset_manifest.py": "f1a288406e50a2fb9ca8ac044268a6aef7818fb35552e151abc9aacbad83e528",
    "dc_metrics.py": "7d7efc69932a96f30406234a18829892252129146532f9c12eca78aadf043876",
    "dc_model_factory.py": "091c8edce41afcba6bc09ed60fa479ee1eb65202891bdd906da7f3f42ea8ff79",
    "dc_pareto_stats.py": "edc410ac266e569c7aaab7a0facf07393f3967ae3400e787177fbe23d668459e",
    "dc_qualitative.py": "c9b41a78c4d03e5f51130b1c86dc9724e6f9b2ef24c705ca6de375faf892dfd6",
    "dc_schemas.py": "a03c1353cf0cbc2cd41e27bad0b371b17ecacd1d8fa35d21feb695a75d12319d",
    "dc_selection.py": "295fac5a356ddf48d2a97d7eff0bace01caec44de6bd220e91a47bc06fcb189e",
    "dc_sparseview_runner.py": "c59bb43a61ccf98c039629b2357a5491cdf1d72aebb1a3f70db1efa5791a73e1",
    "expected_assets.json": "160369bf2e4a1040df1e30a48aed5d567fdc59a731679f1827f8f48a2daea370",
    "frozen_dc_protocol.json": "c84bdd8f1750f42f1f63b6daffc941ce58c486888523455df7782db266ca0e6f",
}

# Locked 125-view checkpoints (lock payload of the 125-view FINAL test run).
FROZEN_CHECKPOINT_SHA256_125 = {
    "UNet": "bf6069bf7a952e39a726ebe1d2d510c4c6d96466b3e318e013a76824f2d30e00",
    "TransUNet": "17c59dbf2ea9a6ec8a2e36206266e91a427f260f780878c3591d44075b808105",
    "WavTransUNet": "aa1fe4f3ca2f7c00a876a835a66c4a0234ed0e46cefce1a757e622cd27cd2d0a",
}

PRIMARY_SELECTION_RULE = "deterministic q10/q50/q90 case lock; no override"
DISPLAY_SELECTION_RULE = (
    "inherited 50-view q10/q50/q90 case lock (AMENDMENT 5 display-only); "
    "no reselection on 125-view outcomes; no override"
)
INTENDED_USE = (
    "Figure 4 display panels only. Every reported 125-view metric remains the "
    "frozen test_locked/test_case_metrics.csv value; the 18 panel metrics "
    "here are re-derived solely as a parity canary against that CSV."
)


class DisplayExportError(runner.RunnerError):
    pass


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DisplayExportError(f"{where} must be an object")
    return value


# ---------------------------------------------------------------------------
# Frozen-chain and provenance authentication
# ---------------------------------------------------------------------------


def assert_frozen_chain() -> dict[str, str]:
    """The frozen chain files next to this script must be the pinned bytes."""

    observed = runner._package_hashes()
    if observed != FROZEN_CHAIN_SHA256:
        drift = sorted(
            name
            for name in set(observed) | set(FROZEN_CHAIN_SHA256)
            if observed.get(name) != FROZEN_CHAIN_SHA256.get(name)
        )
        raise DisplayExportError(f"frozen chain bytes differ from pinned digests: {drift}")
    if dc_qualitative.PRIMARY_VIEW != PRIMARY_VIEWS:
        raise DisplayExportError("frozen dc_qualitative.PRIMARY_VIEW is not 50")
    return dict(observed)


@contextlib.contextmanager
def primary_view_override(views: int) -> Iterator[None]:
    """Guarded runtime override of the frozen 50-view selector constant.

    Only the 125-view test-grid authentication runs inside this context; the
    frozen module bytes are untouched and the constant is restored afterwards.
    """

    if dc_qualitative.PRIMARY_VIEW != PRIMARY_VIEWS:
        raise DisplayExportError("nested or foreign PRIMARY_VIEW override detected")
    dc_qualitative.PRIMARY_VIEW = views
    try:
        yield
    finally:
        dc_qualitative.PRIMARY_VIEW = PRIMARY_VIEWS


def authenticate_primary_provenance(
    case_lock_path: Path,
    export_manifest_path: Path,
    protocol_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Bind the case IDs to the authenticated 50-view lock and FINAL export."""

    for path, name in ((case_lock_path, "primary case lock"), (export_manifest_path, "primary export manifest")):
        if path.is_symlink() or not path.is_file():
            raise DisplayExportError(f"{name} must be a regular non-symlink file")
    case_lock = load_json(case_lock_path)
    export_manifest = load_json(export_manifest_path)

    expected_manifest = {
        "schema": "dc.run_manifest.v1",
        "status": "COMPLETE",
        "command": "export-cases",
        "run_kind": "FINAL",
        "views": PRIMARY_VIEWS,
        "split": "test",
        "protocol_sha256": protocol_hash,
    }
    for field, value in expected_manifest.items():
        if export_manifest.get(field) != value:
            raise DisplayExportError(
                f"primary export manifest {field}={export_manifest.get(field)!r}, expected {value!r}"
            )
    if export_manifest.get("package_sha256") != FROZEN_CHAIN_SHA256:
        raise DisplayExportError("primary export manifest was not produced by the frozen chain")
    run_id = export_manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise DisplayExportError("primary export manifest run_id is missing")
    inputs = _mapping(export_manifest.get("inputs"), "primary export inputs")
    lock_input = _mapping(inputs.get("case_lock"), "primary export inputs.case_lock")
    if lock_input.get("sha256") != sha256_file(case_lock_path):
        raise DisplayExportError("primary case lock bytes differ from the FINAL export input")

    expected_lock = {
        "schema": "dc.case_lock.v1",
        "status": "LOCKED",
        "view": PRIMARY_VIEWS,
        "protocol_sha256": protocol_hash,
        "selection_rule": (
            "nearest ranks to q10/q50/q90 of W+DC PSNR minus "
            "max(U+DC,T+DC) PSNR; case-ID tie-break"
        ),
        "interpretation": "descriptive, outcome-locked before image inspection",
        "image_exported": False,
    }
    for field, value in expected_lock.items():
        if case_lock.get(field) != value:
            raise DisplayExportError(
                f"primary case lock {field}={case_lock.get(field)!r}, expected {value!r}"
            )
    cases = case_lock.get("cases")
    if not isinstance(cases, list) or len(cases) != len(LOCKED_CASES):
        raise DisplayExportError("primary case lock does not hold exactly three cases")
    for case, (label, quantile, case_id) in zip(cases, LOCKED_CASES):
        entry = _mapping(case, "primary case lock case")
        if (
            entry.get("label") != label
            or entry.get("case_id") != case_id
            or type(entry.get("case_id")) is not int
            or not np.isclose(float(entry.get("quantile", np.nan)), quantile, rtol=0.0, atol=1.0e-12)
            or type(entry.get("rank")) is not int
        ):
            raise DisplayExportError(
                f"primary case lock case differs from the pinned lock: {entry} vs {(label, quantile, case_id)}"
            )
    coverage = _mapping(export_manifest.get("coverage"), "primary export coverage")
    if coverage.get("case_ids") != list(LOCKED_CASE_IDS) or coverage.get("case_labels") != list(
        LOCKED_CASE_LABELS
    ):
        raise DisplayExportError("primary export coverage differs from the pinned case lock")
    config = _mapping(export_manifest.get("scientific_config"), "primary export scientific_config")
    if config.get("selection") != PRIMARY_SELECTION_RULE:
        raise DisplayExportError("primary export selection rule differs")
    if not runner._canonical_equal(config.get("selected_cases"), cases):
        raise DisplayExportError("primary export selected_cases differ from the case lock")
    if len({item["case_id"] for item in cases}) != 3:
        raise DisplayExportError("primary case IDs are not unique")
    return [dict(item) for item in cases], case_lock, export_manifest


def authenticate_display_test_evidence(
    *,
    test_manifest_path: Path,
    test_csv: Path,
    lock_path: Path,
    protocol_path: Path,
    views: int,
    backend: str,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], dict[tuple[int, str, int], dict[str, Any]]]:
    """Authenticate the 125-view lock, COMPLETE test manifest, and full grid."""

    if views != DISPLAY_VIEWS:
        raise DisplayExportError(f"display export is prespecified for {DISPLAY_VIEWS} views only")
    test_manifest = runner._require_complete_manifest(test_manifest_path, "evaluate-test")
    if test_manifest.get("views") != views or test_manifest.get("split") != "test":
        raise DisplayExportError(
            f"display export requires the COMPLETE {views}-view test analysis"
        )
    with primary_view_override(views):
        lock, protocol, selected = dc_qualitative._load_lock_and_protocol(lock_path, protocol_path)
        authenticated, run_id, expected_rows = dc_qualitative._authenticate_test_manifest(
            test_manifest_path, test_csv, lock_path, lock, protocol
        )
        payload = _mapping(lock.get("payload"), "lock.payload")
        grid = dc_qualitative._read_authenticated_test_rows(
            test_csv,
            run_id=run_id,
            expected_rows=expected_rows,
            lock_payload=payload,
        )
    if not runner._canonical_equal(test_manifest, authenticated):
        raise DisplayExportError("independent test-manifest authentication disagrees")
    if payload.get("views") != views:
        raise DisplayExportError("lock views differ from the display view")
    if int(selected["k"]) != EXPECTED_SELECTED_K:
        raise DisplayExportError(
            f"lock selected k={selected['k']} differs from the pinned display K={EXPECTED_SELECTED_K}"
        )
    if payload.get("checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256_125:
        raise DisplayExportError("locked checkpoint identities differ from the pinned 125-view checkpoints")
    canary = _mapping(test_manifest.get("canaries"), "test manifest canaries")
    k0_canary = _mapping(canary.get("k0_parity"), "test manifest k0 parity")
    if (
        k0_canary.get("status") != "PASS"
        or k0_canary.get("cases") != runner.EXPECTED_COUNTS["test"]
        or k0_canary.get("models") != list(runner.CANONICAL_MODELS)
    ):
        raise DisplayExportError("COMPLETE test manifest lacks the full k=0 parity canary")
    environment = _mapping(test_manifest.get("environment"), "test environment")
    if environment.get("backend") != backend:
        raise DisplayExportError("export backend differs from the authenticated test backend")
    if {str(row["backend"]) for row in grid.values()} != {backend}:
        raise DisplayExportError("authenticated test metric grid backend is not canonical")
    return lock, test_manifest, selected, grid


# ---------------------------------------------------------------------------
# View-parameterised re-implementations of frozen 50-view routines
# ---------------------------------------------------------------------------


def authenticated_test_prediction_assets(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    views: int,
) -> tuple[Path, dict[str, Path], dict[str, dict[str, Any]]]:
    """Frozen ``_authenticated_test_prediction_assets`` with ``views`` asset IDs."""

    inputs = _mapping(manifest.get("inputs"), "test manifest inputs")
    outputs = _mapping(manifest.get("outputs"), "test manifest outputs")
    lock_payload = _mapping(lock.get("payload"), "lock.payload")
    checkpoint_map = _mapping(lock_payload.get("checkpoint_sha256"), "lock.payload.checkpoint_sha256")
    if set(checkpoint_map) != set(runner.CANONICAL_MODELS):
        raise DisplayExportError("locked checkpoint map is not canonical")

    verification = _mapping(inputs.get("asset_verification"), "test manifest asset_verification")
    verified = verification.get("verified_assets")
    if not isinstance(verified, list):
        raise DisplayExportError("test manifest verified-assets list is missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in verified:
        entry = _mapping(item, "test manifest verified asset")
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or asset_id in by_id:
            raise DisplayExportError("test manifest verified-assets IDs are invalid")
        by_id[asset_id] = entry

    fbp_entry = by_id.get(f"fbp_test_{views}")
    if fbp_entry is None or fbp_entry.get("status") != "VERIFIED":
        raise DisplayExportError(f"authenticated {views}-view test FBP asset is missing")
    fbp_path_value = fbp_entry.get("path")
    if not isinstance(fbp_path_value, str):
        raise DisplayExportError("authenticated test FBP path is missing")
    fbp_path = Path(fbp_path_value)
    if sha256_file(fbp_path) != fbp_entry.get("sha256"):
        raise DisplayExportError("test FBP cache differs from authenticated manifest")

    root_entry = _mapping(outputs.get("prediction_root"), "test prediction root")
    root_value = root_entry.get("path")
    if not isinstance(root_value, str):
        raise DisplayExportError("test prediction-root path is missing")
    root_path = Path(root_value)
    if root_path.is_symlink() or not root_path.is_dir():
        raise DisplayExportError("test prediction root must be a regular directory")
    prediction_root = root_path.resolve()
    model_evidence = _mapping(inputs.get("model_evidence"), "test model evidence")
    paths: dict[str, Path] = {}
    evidence: dict[str, dict[str, Any]] = {}
    asset_prefix = {
        "UNet": f"checkpoint_unet_{views}",
        "TransUNet": f"checkpoint_transunet_{views}",
        "WavTransUNet": f"checkpoint_wavtransunet_{views}",
    }
    for model in runner.CANONICAL_MODELS:
        expected_checkpoint = checkpoint_map[model]
        if not runner._valid_sha256(expected_checkpoint):
            raise DisplayExportError(f"invalid locked checkpoint SHA-256: {model}")
        output = _mapping(outputs.get(f"predictions_{model}"), f"test prediction output {model}")
        path_value = output.get("path")
        if not isinstance(path_value, str):
            raise DisplayExportError(f"test prediction path is missing: {model}")
        path = Path(path_value)
        if path.resolve().parent != prediction_root:
            raise DisplayExportError(f"test prediction escapes authenticated root: {model}")
        if output.get("shape") != [
            runner.EXPECTED_COUNTS["test"],
            1,
            CROP_SIZE,
            CROP_SIZE,
        ] or output.get("dtype") != "float32":
            raise DisplayExportError(f"test prediction schema differs: {model}")
        if sha256_file(path) != output.get("sha256"):
            raise DisplayExportError(f"test prediction bytes differ from manifest: {model}")

        inference = _mapping(output.get("inference_report"), f"test inference report {model}")
        expected_inference = {
            "cases": runner.EXPECTED_COUNTS["test"],
            "input_dtype": "float32",
            "output_dtype": "float32",
            "normalized": False,
            "clipped": False,
            "case_order_preserved": True,
            "exact_coverage": True,
        }
        for field, expected in expected_inference.items():
            if inference.get(field) != expected:
                raise DisplayExportError(f"test inference report {field} differs: {model}")

        model_record = _mapping(model_evidence.get(model), f"model evidence {model}")
        build = _mapping(model_record.get("build"), f"model build evidence {model}")
        load = _mapping(model_record.get("load"), f"model load evidence {model}")
        if build.get("architecture") != model:
            raise DisplayExportError(f"model build architecture differs: {model}")
        post_state = load.get("post_load_state_sha256")
        if not runner._valid_sha256(post_state):
            raise DisplayExportError(f"post-load state hash is invalid: {model}")
        if (
            load.get("checkpoint_sha256") != expected_checkpoint
            or output.get("checkpoint_sha256") != expected_checkpoint
            or output.get("post_load_state_sha256") != post_state
            or load.get("strict") is not True
            or load.get("missing_keys") not in ([], ())
            or load.get("unexpected_keys") not in ([], ())
        ):
            raise DisplayExportError(f"checkpoint/state evidence differs: {model}")
        checkpoint_path_value = load.get("checkpoint_path")
        if not isinstance(checkpoint_path_value, str):
            raise DisplayExportError(f"checkpoint evidence path is missing: {model}")
        checkpoint_path = Path(checkpoint_path_value)
        if sha256_file(checkpoint_path) != expected_checkpoint:
            raise DisplayExportError(f"checkpoint bytes differ from locked identity: {model}")
        asset = by_id.get(asset_prefix[model])
        if asset is None or asset.get("status") != "VERIFIED":
            raise DisplayExportError(f"verified checkpoint asset is missing: {model}")
        if (
            asset.get("sha256") != expected_checkpoint
            or not isinstance(asset.get("path"), str)
            or Path(str(asset["path"])).resolve() != checkpoint_path.resolve()
        ):
            raise DisplayExportError(f"verified checkpoint asset differs: {model}")

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (
            runner.EXPECTED_COUNTS["test"],
            1,
            CROP_SIZE,
            CROP_SIZE,
        ) or array.dtype != np.float32:
            raise DisplayExportError(f"test prediction array differs from manifest: {model}")
        paths[model] = path
        evidence[model] = {
            "prediction_path": str(path.resolve()),
            "prediction_sha256": output["sha256"],
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": expected_checkpoint,
            "post_load_state_sha256": post_state,
            "strict_checkpoint_load": True,
            "raw_float32_inference": True,
            "normalized": False,
            "clipped": False,
        }
    return fbp_path, paths, evidence


def assert_export_operator_identity(
    test_manifest: Mapping[str, Any],
    observed_environment: Mapping[str, Any],
    observed_operator: Mapping[str, Any],
    views: int,
) -> None:
    """Frozen ``_assert_export_operator_identity`` with a ``views`` range shape."""

    expected_environment = _mapping(test_manifest.get("environment"), "test environment")
    for field in ("dival", "odl", "astra", "backend"):
        if observed_environment.get(field) != expected_environment.get(field):
            raise DisplayExportError(f"qualitative export {field} differs from formal test")
    test_canaries = _mapping(test_manifest.get("canaries"), "test canaries")
    expected_operator = _mapping(test_canaries.get("operator"), "test operator canary")
    for field, expected in (
        ("domain_shape", [FULL_SIZE, FULL_SIZE]),
        ("range_shape", [views, SINOGRAM_BINS]),
    ):
        if expected_operator.get(field) != expected or observed_operator.get(field) != expected:
            raise DisplayExportError(f"qualitative export operator {field} differs")
    try:
        expected_power = float(expected_operator.get("lambda_max_ata"))
        observed_power = float(observed_operator.get("lambda_max_ata"))
    except (TypeError, ValueError) as exc:
        raise DisplayExportError("qualitative export operator power canary is invalid") from exc
    if not np.isfinite(expected_power) or not np.isfinite(observed_power):
        raise DisplayExportError("qualitative export operator power canary is nonfinite")
    if not np.isclose(expected_power, observed_power, rtol=1.0e-6, atol=0.0):
        raise DisplayExportError("qualitative export operator power canary differs from formal test")


def expected_array_shapes(views: int) -> dict[str, tuple[int, ...]]:
    return {
        "ground_truth": (CROP_SIZE, CROP_SIZE),
        "fbp_canvas": (FULL_SIZE, FULL_SIZE),
        "measured_sinogram": (views, SINOGRAM_BINS),
        **{key: (CROP_SIZE, CROP_SIZE) for key in runner.QUALITATIVE_PANEL_KEYS},
        **{f"error_{key}": (CROP_SIZE, CROP_SIZE) for key in runner.QUALITATIVE_PANEL_KEYS},
    }


def write_qualitative_outputs(
    out: Path,
    case_payloads: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    views: int,
) -> dict[str, Any]:
    """Frozen ``_write_qualitative_outputs`` with a ``views``-row sinogram.

    File names, NPZ keys, metric CSV header, and SHA-256 inventory are
    identical to the frozen 50-view export.
    """

    if len(case_payloads) != 3 or len(metric_rows) != 18:
        raise DisplayExportError("qualitative export requires exactly 3 cases and 18 metric rows")
    expected_labels = [item[0] for item in runner.QUALITATIVE_CASE_LABELS]
    if [item.get("label") for item in case_payloads] != expected_labels:
        raise DisplayExportError("qualitative case payload order differs from q10/q50/q90 lock")
    shapes = expected_array_shapes(views)
    expected_metric_keys = {
        (int(payload["case_id"]), panel_key)
        for payload in case_payloads
        for panel_key in runner.QUALITATIVE_PANEL_KEYS
    }
    observed_metric_keys: set[tuple[int, str]] = set()
    observed_gate_labels: set[tuple[str, bool, bool]] = set()
    for row in metric_rows:
        key = (int(row["case_id"]), str(row["panel_key"]))
        if key in observed_metric_keys:
            raise DisplayExportError(f"duplicate qualitative metric row: {key}")
        observed_metric_keys.add(key)
        if row.get("clipped") is not False or row.get("normalized") is not False:
            raise DisplayExportError("qualitative metrics must remain raw and unclipped")
        status = row.get("operating_point_status")
        claim_allowed = row.get("operating_point_claim_allowed")
        diagnostic_fallback = row.get("diagnostic_fallback")
        if (
            status not in ("PASS", "FAIL")
            or type(claim_allowed) is not bool
            or type(diagnostic_fallback) is not bool
            or (status == "PASS") != claim_allowed
            or diagnostic_fallback == claim_allowed
        ):
            raise DisplayExportError("qualitative operating-point gate label is inconsistent")
        observed_gate_labels.add((status, claim_allowed, diagnostic_fallback))
        for field in ("psnr", "ssim", "rmse", "projection_residual"):
            try:
                finite = np.isfinite(float(row[field]))
            except (KeyError, TypeError, ValueError) as exc:
                raise DisplayExportError(f"invalid qualitative metric {field}: {key}") from exc
            if not finite:
                raise DisplayExportError(f"nonfinite qualitative metric {field}: {key}")
    if observed_metric_keys != expected_metric_keys:
        raise DisplayExportError("qualitative metric rows do not cover all six locked panels")
    if len(observed_gate_labels) != 1:
        raise DisplayExportError("qualitative metric rows mix operating-point gate labels")
    files: dict[str, Any] = {}
    for payload in case_payloads:
        label = str(payload["label"])
        case_id = int(payload["case_id"])
        arrays = _mapping(payload.get("arrays"), f"qualitative arrays {label}")
        if set(arrays) != set(shapes):
            raise DisplayExportError(f"qualitative array schema differs: {label}")
        canonical_arrays: dict[str, np.ndarray] = {}
        for key, expected_shape in shapes.items():
            value = np.asarray(arrays[key])
            if value.shape != expected_shape or value.dtype != np.float32:
                raise DisplayExportError(f"qualitative array shape/dtype differs: {label}/{key}")
            if not np.all(np.isfinite(value)):
                raise DisplayExportError(f"qualitative array is nonfinite: {label}/{key}")
            canonical_arrays[key] = value
        path = out / f"case_{label}_{case_id:04d}.npz"
        runner._atomic_write_npz(path, canonical_arrays)
        with np.load(path, allow_pickle=False) as stored:
            inventory = {
                key: {"shape": list(stored[key].shape), "dtype": str(stored[key].dtype)}
                for key in sorted(stored.files)
            }
        files[label] = {
            "case_id": case_id,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "arrays": inventory,
        }

    metrics_partial = out / "qualitative_case_metrics.csv.partial"
    with metrics_partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.QUALITATIVE_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(metric_rows)
        stream.flush()
        os.fsync(stream.fileno())
    metrics_path = out / "qualitative_case_metrics.csv"
    os.replace(metrics_partial, metrics_path)

    hash_partial = out / "SHA256SUMS.csv.partial"
    hash_columns = ("artifact", "sha256")
    with hash_partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=hash_columns)
        writer.writeheader()
        for item in files.values():
            writer.writerow({"artifact": Path(item["path"]).name, "sha256": item["sha256"]})
        writer.writerow({"artifact": metrics_path.name, "sha256": sha256_file(metrics_path)})
        stream.flush()
        os.fsync(stream.fileno())
    hash_path = out / "SHA256SUMS.csv"
    os.replace(hash_partial, hash_path)
    return {
        "case_arrays": files,
        "case_metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
            "rows": len(metric_rows),
            "header": list(runner.QUALITATIVE_METRIC_COLUMNS),
        },
        "hash_inventory": {
            "path": str(hash_path.resolve()),
            "sha256": sha256_file(hash_path),
            "rows": len(files) + 1,
        },
    }


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def command_export_cases_display(args: argparse.Namespace) -> int:
    views = int(args.views)
    chain = assert_frozen_chain()
    protocol, protocol_hash = runner._protocol()
    del protocol
    # Authenticate every pre-existing artifact — inherited case IDs, 125-view
    # lock/test grid, and prediction/checkpoint identity — before image access
    # or output-directory creation, exactly as the frozen export does.
    primary_cases, primary_case_lock, primary_export_manifest = authenticate_primary_provenance(
        args.primary_case_lock, args.primary_export_manifest, protocol_hash
    )
    lock, test_manifest, selected, grid = authenticate_display_test_evidence(
        test_manifest_path=args.test_manifest,
        test_csv=args.test_csv,
        lock_path=args.lock,
        protocol_path=args.protocol,
        views=views,
        backend=args.backend,
    )
    for case in primary_cases:
        if (int(case["case_id"]), "WavTransUNet", EXPECTED_SELECTED_K) not in grid:
            raise DisplayExportError(f"inherited case is absent from the {views}-view test grid: {case}")
    fbp_path, prediction_paths, checkpoint_evidence = authenticated_test_prediction_assets(
        test_manifest, lock, views
    )
    operating_gate = _mapping(lock["payload"].get("operating_point_gate"), "lock.payload.operating_point_gate")
    diagnostic_fallback = operating_gate.get("claim_allowed") is False
    script_path = Path(__file__).resolve()

    out = runner._new_output_directory(args.out)
    manifest_path = out / "qualitative_export_manifest.json"
    manifest = runner._started_manifest("export-cases", views, "test", run_kind=DISPLAY_RUN_KIND)
    manifest["amendment"] = {
        "id": AMENDMENT_ID,
        "document": AMENDMENT_DOCUMENT,
        "scope": (
            "lifts only the 'qualitative export is frozen to the 50-view test analysis' "
            "constraint, for a display-only, case-locked 125-view export"
        ),
        "frozen_chain_modified": False,
        "frozen_chain_sha256_verified": chain,
        "display_export_script": {"path": str(script_path), "sha256": sha256_file(script_path)},
    }
    manifest["display_only"] = True
    manifest["intended_use"] = INTENDED_USE
    manifest["case_provenance"] = {
        "rule": DISPLAY_SELECTION_RULE,
        "locked_case_ids": list(LOCKED_CASE_IDS),
        "locked_case_labels": list(LOCKED_CASE_LABELS),
        "primary_views": PRIMARY_VIEWS,
        "primary_case_lock": {
            "path": str(args.primary_case_lock.resolve()),
            "sha256": sha256_file(args.primary_case_lock),
            "test_run_id": primary_case_lock.get("test_run_id"),
            "selected_k": primary_case_lock.get("selected_k"),
        },
        "primary_export_manifest": {
            "path": str(args.primary_export_manifest.resolve()),
            "sha256": sha256_file(args.primary_export_manifest),
            "run_id": primary_export_manifest.get("run_id"),
        },
    }
    runner._write_manifest(manifest_path, manifest)
    try:
        manifest["inputs"] = {
            "test_manifest": {
                "path": str(args.test_manifest.resolve()),
                "sha256": sha256_file(args.test_manifest),
                "run_id": test_manifest["run_id"],
            },
            "test_case_metrics": {
                "path": str(args.test_csv.resolve()),
                "sha256": sha256_file(args.test_csv),
            },
            "dc_lock": {
                "path": str(args.lock.resolve()),
                "sha256": sha256_file(args.lock),
                "payload_sha256": lock["payload_sha256"],
            },
            "case_lock": {
                "path": str(args.primary_case_lock.resolve()),
                "sha256": sha256_file(args.primary_case_lock),
                "inherited_from_views": PRIMARY_VIEWS,
            },
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
                "canonical_json_sha256": test_manifest["protocol_sha256"],
            },
            "fbp_cache": {
                "path": str(fbp_path.resolve()),
                "sha256": sha256_file(fbp_path),
            },
            "fresh_prediction_and_checkpoint_evidence": checkpoint_evidence,
        }
        torch, dival, dataset, operator = runner._import_physics(
            args.data, args.dival_root, views, backend=args.backend
        )
        manifest["environment"] = runner._environment(torch, dival, args.backend)
        manifest["canaries"]["operator"] = runner._operator_canaries(operator, views)
        assert_export_operator_identity(
            test_manifest,
            manifest["environment"],
            manifest["canaries"]["operator"],
            views,
        )
        fbp = np.load(fbp_path, mmap_mode="r", allow_pickle=False)
        if fbp.shape != (
            runner.EXPECTED_COUNTS["test"],
            FULL_SIZE,
            FULL_SIZE,
        ) or fbp.dtype != np.float32:
            raise DisplayExportError("authenticated test FBP array shape/dtype changed")
        predictions = {
            model: np.load(path, mmap_mode="r", allow_pickle=False)
            for model, path in prediction_paths.items()
        }
        selected_k = int(selected["k"])
        lam = float(selected["lambda"])
        multiplier = float(selected["lambda_multiplier"])
        eta = float(selected["eta"])
        checkpoint_map = _mapping(lock["payload"].get("checkpoint_sha256"), "lock checkpoint map")

        case_payloads: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        parity_maxima = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "projection_residual": 0.0}
        dc_effect: dict[str, dict[str, Any]] = {}
        panel_keys = {
            "UNet": ("unet", "unet_finite_dc"),
            "TransUNet": ("transunet", "transunet_finite_dc"),
            "WavTransUNet": ("wtransunet", "wtransunet_finite_dc"),
        }
        for case in primary_cases:
            case_id = int(case["case_id"])
            y_obs, gt = dataset.get_sample(case_id, part="test")
            y = np.asarray(y_obs, dtype=np.float32)
            gt362 = np.asarray(gt, dtype=np.float32)
            if y.shape != (views, SINOGRAM_BINS) or gt362.shape != (FULL_SIZE, FULL_SIZE):
                raise DisplayExportError(f"selected test sample has unexpected shape: {case_id}")
            arrays: dict[str, np.ndarray] = {
                "ground_truth": np.asarray(crop_canvas(gt362), dtype=np.float32),
                "fbp_canvas": np.asarray(fbp[case_id], dtype=np.float32),
                "measured_sinogram": y,
            }
            for model in runner.CANONICAL_MODELS:
                k0_canvas = embed_crop(runner._prediction_crop(predictions[model], case_id), fbp[case_id])
                trajectory = finite_dc_trajectory(
                    operator,
                    k0_canvas,
                    fbp[case_id],
                    y,
                    regularization=lam,
                    step_size=eta,
                    iterations=(0, selected_k),
                )
                if trajectory.diverged or selected_k not in trajectory.snapshots:
                    raise DisplayExportError(
                        f"selected qualitative trajectory diverged: {model}, case={case_id}"
                    )
                for phase, k, canvas, panel_key in (
                    ("k0", 0, k0_canvas, panel_keys[model][0]),
                    ("finite_dc", selected_k, trajectory.snapshots[selected_k], panel_keys[model][1]),
                ):
                    metrics = runner._metrics_one(canvas, gt362, "cuda")
                    residual = (
                        projection_residual(operator, canvas, y)
                        if k == 0
                        else float(trajectory.residuals[selected_k])
                    )
                    observed = dict(metrics)
                    observed["projection_residual"] = residual
                    differences = runner._assert_qualitative_metric_parity(
                        observed,
                        grid[(case_id, model, k)],
                        model=model,
                        case_id=case_id,
                        k=k,
                    )
                    for name, difference in differences.items():
                        parity_maxima[name] = max(parity_maxima[name], difference)
                    crop = np.asarray(crop_canvas(canvas), dtype=np.float32)
                    arrays[panel_key] = crop
                    arrays[f"error_{panel_key}"] = np.asarray(crop - arrays["ground_truth"], dtype=np.float32)
                    metric_rows.append(
                        {
                            "schema": "dc.qualitative_case_metric.v1",
                            "case_label": case["label"],
                            "quantile": case["quantile"],
                            "rank": case["rank"],
                            "case_id": case_id,
                            "patient_id": grid[(case_id, model, k)]["patient_id"],
                            "panel_key": panel_key,
                            "architecture": model,
                            "phase": phase,
                            "dc_iteration": k,
                            "checkpoint_sha256": checkpoint_map[model],
                            "post_load_state_sha256": checkpoint_evidence[model]["post_load_state_sha256"],
                            "lambda": lam if k else 0.0,
                            "lambda_multiplier": multiplier if k else 0.0,
                            "eta": eta if k else 0.0,
                            "psnr": metrics["psnr_db"],
                            "ssim": metrics["ssim"],
                            "rmse": metrics["rmse"],
                            "projection_residual": residual,
                            "clipped": False,
                            "normalized": False,
                            "backend": args.backend,
                            "source_test_run_id": test_manifest["run_id"],
                            "operating_point_status": operating_gate["status"],
                            "operating_point_claim_allowed": operating_gate["claim_allowed"],
                            "diagnostic_fallback": diagnostic_fallback,
                        }
                    )
                # Display-only canary: the finite-DC panel must be a genuine
                # k=selected_k update of the k=0 panel, never the k=0 bytes.
                k0_key, dc_key = panel_keys[model]
                max_abs_update = float(np.max(np.abs(arrays[dc_key] - arrays[k0_key])))
                if not (max_abs_update > 0.0):
                    raise DisplayExportError(
                        f"finite-DC panel is identical to k=0: {model}, case={case_id}"
                    )
                dc_effect[f"{case_id}:{model}"] = {
                    "k": selected_k,
                    "max_abs_update_vs_k0": max_abs_update,
                    "projection_residual_k0": float(trajectory.residuals[0]),
                    "projection_residual_k": float(trajectory.residuals[selected_k]),
                }
            case_payloads.append({"label": case["label"], "case_id": case_id, "arrays": arrays})

        manifest["outputs"] = write_qualitative_outputs(out, case_payloads, metric_rows, views)
        manifest["coverage"] = {
            "cases": 3,
            "reconstruction_panels_per_case": 6,
            "metric_rows": 18,
            "case_labels": [item["label"] for item in primary_cases],
            "case_ids": [item["case_id"] for item in primary_cases],
        }
        manifest["scientific_config"] = {
            "selection": DISPLAY_SELECTION_RULE,
            "operating_point_gate": dict(operating_gate),
            "operating_point_label": (
                "validation-selected claim-eligible operating point"
                if not diagnostic_fallback
                else "diagnostic fallback; claim-ineligible"
            ),
            "selected_cases": primary_cases,
            "selected_cases_quantile_rank_refer_to_views": PRIMARY_VIEWS,
            "panel_order": [
                "ground_truth",
                "unet",
                "unet_finite_dc",
                "transunet",
                "transunet_finite_dc",
                "wtransunet",
                "wtransunet_finite_dc",
            ],
            "selected_finite_dc": {
                "k": selected_k,
                "lambda": lam,
                "lambda_multiplier": multiplier,
                "eta": eta,
            },
            "array_domain": "raw float32 LoDoPaB normalized image domain",
            "image_crop": [BORDER, BORDER + CROP_SIZE, BORDER, BORDER + CROP_SIZE],
            "metric_definition": metric_definition(),
            "data_consistency_definition": dc_definition(),
            "reconstruction_clipping": False,
            "reconstruction_normalization": False,
            "display_windowing_applied": False,
            "error_map_definition": "reconstruction minus ground_truth in raw crop domain",
            "interactive_case_selection": False,
        }
        manifest["canaries"]["selected_row_parity"] = {
            "status": "PASS",
            "rows": 18,
            "metric_tolerance": 1.0e-5,
            "projection_residual_tolerance": 1.0e-8,
            "max_abs_difference": parity_maxima,
            "reference": "frozen 125-view test_locked/test_case_metrics.csv rows (case, model, k in {0, 4})",
        }
        manifest["canaries"]["finite_dc_is_not_k0"] = {
            "status": "PASS",
            "k": selected_k,
            "pairs": dc_effect,
        }
        runner._complete_manifest(manifest_path, manifest)
        return 0
    except BaseException as exc:
        runner._fail_manifest(manifest_path, manifest, exc)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", type=int, required=True, choices=(DISPLAY_VIEWS,))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-case-lock", type=Path, required=True)
    parser.add_argument("--primary-export-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dival-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("astra_cuda",), default="astra_cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(command_export_cases_display(args))


if __name__ == "__main__":
    raise SystemExit(main())
