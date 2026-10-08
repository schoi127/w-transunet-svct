#!/usr/bin/env python3
"""Render authenticated qualitative Figure C from a COMPLETE export bundle.

This renderer accepts only the ``export-cases`` manifest.  The three locked
NPZ bundles, raw metric CSV, and SHA-256 inventory are resolved from that
manifest and independently authenticated before rendering.  It never opens
the dataset, checkpoints, predictions, validation/test tables, or case-lock
inputs and exposes no case, model, window, or operating-point override.

Display limits cannot adapt to a model outcome: the grayscale window is the
fixed 0.5th--99.5th percentile rule applied only to the three ground-truth
arrays, and the common absolute-error ceiling is exactly 10% of that GT-only
display range.  These are display transforms only; exported arrays and
reported metrics remain raw and unclipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dc_core import dc_definition
from dc_metrics import metric_definition, numpy_psnr_rmse
from dc_schemas import ITERATIONS, json_sha256, load_json


EXPORT_SCHEMA = "dc.run_manifest.v1"
FIGURE_SCHEMA = "dc.qualitative_figure_manifest.v1"
VIEW = 50
IMAGE_SIZE = 352
FULL_SIZE = 362
DISPLAY_PERCENTILES = (0.5, 99.5)
ERROR_RANGE_FRACTION = 0.10
PNG_DPI = 200

CASE_SPECS = (
    ("low_or_failure", 0.10, "q10 / low-or-failure"),
    ("median", 0.50, "q50 / median"),
    ("high_benefit", 0.90, "q90 / high-benefit"),
)
PANEL_SPECS = (
    ("unet", "UNet", "k0", "U-Net"),
    ("unet_finite_dc", "UNet", "finite_dc", "U-Net + DC"),
    ("transunet", "TransUNet", "k0", "TransUNet"),
    ("transunet_finite_dc", "TransUNet", "finite_dc", "TransUNet + DC"),
    ("wtransunet", "WavTransUNet", "k0", "W-TransUNet"),
    (
        "wtransunet_finite_dc",
        "WavTransUNet",
        "finite_dc",
        "W-TransUNet + DC",
    ),
)
PANEL_KEYS = tuple(item[0] for item in PANEL_SPECS)
EXPECTED_ARRAYS = {
    "ground_truth": (IMAGE_SIZE, IMAGE_SIZE),
    "fbp_canvas": (FULL_SIZE, FULL_SIZE),
    "measured_sinogram": (VIEW, 513),
    **{key: (IMAGE_SIZE, IMAGE_SIZE) for key in PANEL_KEYS},
    **{f"error_{key}": (IMAGE_SIZE, IMAGE_SIZE) for key in PANEL_KEYS},
}
METRIC_FIELDS = (
    "schema",
    "case_label",
    "quantile",
    "rank",
    "case_id",
    "patient_id",
    "panel_key",
    "architecture",
    "phase",
    "dc_iteration",
    "checkpoint_sha256",
    "post_load_state_sha256",
    "lambda",
    "lambda_multiplier",
    "eta",
    "psnr",
    "ssim",
    "rmse",
    "projection_residual",
    "clipped",
    "normalized",
    "backend",
    "source_test_run_id",
    "operating_point_status",
    "operating_point_claim_allowed",
    "diagnostic_fallback",
)
EXPORT_PACKAGE_FILENAMES = (
    "dc_sparseview_runner.py",
    "dc_assets.py",
    "dc_schemas.py",
    "dc_model_factory.py",
    "dc_metrics.py",
    "dc_core.py",
    "dc_selection.py",
    "dc_pareto_stats.py",
    "dc_qualitative.py",
    "dc_dataset_manifest.py",
    "frozen_dc_protocol.json",
    "expected_assets.json",
)


class QualitativeFigureError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualitativeFigureError(f"{where} must be an object")
    return value


def _regular_file(path: Path, where: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise QualitativeFigureError(f"{where} must be a regular non-symlink file")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QualitativeFigureError(f"{where} is not numeric") from exc
    if not math.isfinite(result):
        raise QualitativeFigureError(f"{where} is nonfinite")
    return result


def _canonical_int(value: Any, where: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise QualitativeFigureError(f"{where} is not an integer") from exc
    if str(result) != str(value):
        raise QualitativeFigureError(f"{where} is not a canonical integer")
    return result


def _csv_bool(value: Any, where: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise QualitativeFigureError(f"{where} must be exactly True or False")


def _package_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for name in EXPORT_PACKAGE_FILENAMES:
        path = root / name
        _regular_file(path, f"package file {name}")
        result[name] = sha256(path)
    return result


def _new_output_directory(path: Path) -> Path:
    if path.is_symlink():
        raise QualitativeFigureError("output directory cannot be a symlink")
    resolved = path.resolve(strict=False)
    package_root = Path(__file__).resolve().parent
    if resolved == package_root:
        raise QualitativeFigureError("output cannot overwrite the renderer package")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise QualitativeFigureError("output directory must be new and empty")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _authenticate_source_manifest(path: Path) -> dict[str, Any]:
    _regular_file(path, "export manifest")
    try:
        manifest = load_json(path)
    except Exception as exc:
        raise QualitativeFigureError(f"invalid export manifest: {exc}") from exc
    expected = {
        "schema": EXPORT_SCHEMA,
        "status": "COMPLETE",
        "command": "export-cases",
        "run_kind": "FINAL",
        "views": VIEW,
        "split": "test",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise QualitativeFigureError(
                f"export manifest {field}={manifest.get(field)!r}, expected {value!r}"
            )
    if manifest.get("package_sha256") != _package_hashes():
        raise QualitativeFigureError("export manifest package hash mismatch")
    protocol_path = Path(__file__).resolve().parent / "frozen_dc_protocol.json"
    if manifest.get("protocol_sha256") != json_sha256(load_json(protocol_path)):
        raise QualitativeFigureError("export manifest protocol hash mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise QualitativeFigureError("export manifest run_id is missing")

    coverage = _mapping(manifest.get("coverage"), "export coverage")
    expected_labels = [item[0] for item in CASE_SPECS]
    if (
        coverage.get("cases") != 3
        or coverage.get("reconstruction_panels_per_case") != 6
        or coverage.get("metric_rows") != 18
        or coverage.get("case_labels") != expected_labels
        or not isinstance(coverage.get("case_ids"), list)
        or len(coverage["case_ids"]) != 3
        or len(set(coverage["case_ids"])) != 3
    ):
        raise QualitativeFigureError("export coverage is not the locked 3-case grid")

    config = _mapping(manifest.get("scientific_config"), "export scientific_config")
    if (
        config.get("selection")
        != "deterministic q10/q50/q90 case lock; no override"
        or config.get("panel_order")
        != ["ground_truth", *PANEL_KEYS]
        or config.get("array_domain")
        != "raw float32 LoDoPaB normalized image domain"
        or config.get("image_crop") != [5, 357, 5, 357]
        or config.get("metric_definition") != metric_definition()
        or config.get("data_consistency_definition") != dc_definition()
        or config.get("reconstruction_clipping") is not False
        or config.get("reconstruction_normalization") is not False
        or config.get("display_windowing_applied") is not False
        or config.get("error_map_definition")
        != "reconstruction minus ground_truth in raw crop domain"
        or config.get("interactive_case_selection") is not False
    ):
        raise QualitativeFigureError("export scientific configuration differs")
    selected_cases = config.get("selected_cases")
    if not isinstance(selected_cases, list) or len(selected_cases) != 3:
        raise QualitativeFigureError("export selected_cases is incomplete")
    for case, (label, quantile, _title), coverage_id in zip(
        selected_cases, CASE_SPECS, coverage["case_ids"]
    ):
        item = _mapping(case, f"selected case {label}")
        if (
            item.get("label") != label
            or not math.isclose(
                _finite(item.get("quantile"), f"selected case {label} quantile"),
                quantile,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or type(item.get("rank")) is not int
            or item["rank"] < 0
            or type(item.get("case_id")) is not int
            or item["case_id"] != coverage_id
            or not math.isfinite(
                _finite(
                    item.get("delta_psnr_vs_stronger_dc_baseline"),
                    f"selected case {label} effect",
                )
            )
        ):
            raise QualitativeFigureError(f"selected case differs: {label}")

    selected_dc = _mapping(
        config.get("selected_finite_dc"), "export selected_finite_dc"
    )
    if (
        type(selected_dc.get("k")) is not int
        or selected_dc["k"] not in ITERATIONS
        or selected_dc["k"] == 0
        or _finite(selected_dc.get("lambda"), "selected lambda") <= 0.0
        or _finite(selected_dc.get("lambda_multiplier"), "selected multiplier") <= 0.0
        or _finite(selected_dc.get("eta"), "selected eta") <= 0.0
    ):
        raise QualitativeFigureError("selected finite-DC point is invalid")

    gate = _mapping(config.get("operating_point_gate"), "operating-point gate")
    status = gate.get("status")
    claim_allowed = gate.get("claim_allowed")
    fallback = gate.get("fallback_used")
    reason = gate.get("failure_reason")
    if (
        status not in ("PASS", "FAIL")
        or type(claim_allowed) is not bool
        or type(fallback) is not bool
        or (status == "PASS") != claim_allowed
        or fallback == claim_allowed
        or (status == "PASS" and reason is not None)
        or (
            status == "FAIL"
            and (not isinstance(reason, str) or not reason.strip())
        )
    ):
        raise QualitativeFigureError("operating-point gate is inconsistent")
    expected_gate_label = (
        "validation-selected claim-eligible operating point"
        if claim_allowed
        else "diagnostic fallback; claim-ineligible"
    )
    if config.get("operating_point_label") != expected_gate_label:
        raise QualitativeFigureError("operating-point label differs from gate")

    canaries = _mapping(manifest.get("canaries"), "export canaries")
    parity = _mapping(canaries.get("selected_row_parity"), "selected-row parity")
    if parity.get("status") != "PASS" or parity.get("rows") != 18:
        raise QualitativeFigureError("export lacks a passing 18-row parity canary")
    return manifest


def _resolve_bundle_path(
    manifest_path: Path, record: Mapping[str, Any], where: str
) -> Path:
    value = record.get("path")
    if not isinstance(value, str):
        raise QualitativeFigureError(f"{where} path is missing")
    path = Path(value)
    _regular_file(path, where)
    if path.resolve().parent != manifest_path.resolve().parent:
        raise QualitativeFigureError(f"{where} is outside the authenticated export bundle")
    digest = record.get("sha256")
    if not _valid_sha256(digest) or sha256(path) != digest:
        raise QualitativeFigureError(f"{where} hash mismatch")
    return path


def _read_inventory(
    manifest_path: Path,
    inventory_record: Mapping[str, Any],
    expected: Mapping[str, str],
) -> Path:
    if set(inventory_record) != {"path", "sha256", "rows"}:
        raise QualitativeFigureError("hash inventory output record differs")
    path = _resolve_bundle_path(manifest_path, inventory_record, "hash inventory")
    if inventory_record.get("rows") != 4:
        raise QualitativeFigureError("hash inventory must contain exactly four rows")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("artifact", "sha256"):
            raise QualitativeFigureError("hash inventory header differs")
        rows = list(reader)
    if len(rows) != 4:
        raise QualitativeFigureError("hash inventory row count differs")
    observed: dict[str, str] = {}
    for row in rows:
        name = row.get("artifact")
        digest = row.get("sha256")
        if not isinstance(name, str) or name in observed or not _valid_sha256(digest):
            raise QualitativeFigureError("hash inventory contains an invalid row")
        observed[name] = digest
    if observed != dict(expected):
        raise QualitativeFigureError("hash inventory differs from manifest outputs")
    return path


def _load_arrays(
    manifest_path: Path,
    case_outputs: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Path]]:
    expected_labels = {item[0] for item in CASE_SPECS}
    if set(case_outputs) != expected_labels:
        raise QualitativeFigureError("case-array output labels differ")
    arrays_by_label: dict[str, dict[str, np.ndarray]] = {}
    paths: dict[str, Path] = {}
    for selected, (label, _quantile, _title) in zip(selected_cases, CASE_SPECS):
        record = _mapping(case_outputs.get(label), f"case-array output {label}")
        if set(record) != {"case_id", "path", "sha256", "arrays"}:
            raise QualitativeFigureError(f"case-array output schema differs: {label}")
        if record.get("case_id") != selected.get("case_id"):
            raise QualitativeFigureError(f"case-array ID differs: {label}")
        inventory = _mapping(record.get("arrays"), f"case-array inventory {label}")
        if set(inventory) != set(EXPECTED_ARRAYS):
            raise QualitativeFigureError(f"case-array inventory keys differ: {label}")
        for key, shape in EXPECTED_ARRAYS.items():
            spec = _mapping(inventory.get(key), f"case-array inventory {label}/{key}")
            if spec.get("shape") != list(shape) or spec.get("dtype") != "float32":
                raise QualitativeFigureError(f"case-array inventory differs: {label}/{key}")
        path = _resolve_bundle_path(manifest_path, record, f"case-array bundle {label}")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(EXPECTED_ARRAYS):
                raise QualitativeFigureError(f"NPZ array keys differ: {label}")
            arrays: dict[str, np.ndarray] = {}
            for key, shape in EXPECTED_ARRAYS.items():
                value = np.asarray(stored[key])
                if value.shape != shape or value.dtype != np.float32:
                    raise QualitativeFigureError(f"NPZ array schema differs: {label}/{key}")
                if not np.all(np.isfinite(value)):
                    raise QualitativeFigureError(f"NPZ array is nonfinite: {label}/{key}")
                arrays[key] = value.copy()
        gt = arrays["ground_truth"]
        for key in PANEL_KEYS:
            expected_error = np.asarray(arrays[key] - gt, dtype=np.float32)
            if not np.array_equal(arrays[f"error_{key}"], expected_error):
                raise QualitativeFigureError(
                    f"stored error array differs from reconstruction minus GT: {label}/{key}"
                )
        arrays_by_label[label] = arrays
        paths[label] = path
    return arrays_by_label, paths


def _load_metrics(
    manifest_path: Path,
    record: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    selected_dc: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], Path]:
    path = _resolve_bundle_path(manifest_path, record, "qualitative metric CSV")
    if set(record) != {"path", "sha256", "rows", "header"}:
        raise QualitativeFigureError("qualitative metric output record differs")
    if record.get("rows") != 18 or record.get("header") != list(METRIC_FIELDS):
        raise QualitativeFigureError("qualitative metric output schema differs")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise QualitativeFigureError("qualitative metric CSV header differs")
        raw_rows = list(reader)
    if len(raw_rows) != 18:
        raise QualitativeFigureError("qualitative metric CSV row count differs")
    cases = {str(case["label"]): case for case in selected_cases}
    panel_specs = {item[0]: item for item in PANEL_SPECS}
    expected_keys = {(label, key) for label in cases for key in PANEL_KEYS}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    patient_by_case: dict[str, int] = {}
    source_run_ids: set[str] = set()
    backends: set[str] = set()
    checkpoint_by_model: dict[str, str] = {}
    postload_by_model: dict[str, str] = {}
    for line, raw in enumerate(raw_rows, start=2):
        if raw.get("schema") != "dc.qualitative_case_metric.v1":
            raise QualitativeFigureError(f"line {line}: metric schema differs")
        label = str(raw.get("case_label"))
        panel_key = str(raw.get("panel_key"))
        key = (label, panel_key)
        if key not in expected_keys or key in rows:
            raise QualitativeFigureError(f"line {line}: metric grid key differs")
        case = cases[label]
        spec = panel_specs[panel_key]
        case_id = _canonical_int(raw.get("case_id"), f"line {line} case_id")
        patient_id = _canonical_int(raw.get("patient_id"), f"line {line} patient_id")
        rank = _canonical_int(raw.get("rank"), f"line {line} rank")
        k = _canonical_int(raw.get("dc_iteration"), f"line {line} dc_iteration")
        quantile = _finite(raw.get("quantile"), f"line {line} quantile")
        if (
            case_id != case["case_id"]
            or rank != case["rank"]
            or quantile != float(case["quantile"])
            or raw.get("architecture") != spec[1]
            or raw.get("phase") != spec[2]
            or k != (0 if spec[2] == "k0" else int(selected_dc["k"]))
        ):
            raise QualitativeFigureError(f"line {line}: metric identity differs")
        if label in patient_by_case and patient_by_case[label] != patient_id:
            raise QualitativeFigureError(f"line {line}: patient ID changes within case")
        patient_by_case[label] = patient_id
        checkpoint_hash = raw.get("checkpoint_sha256")
        postload_hash = raw.get("post_load_state_sha256")
        if not _valid_sha256(checkpoint_hash) or not _valid_sha256(postload_hash):
            raise QualitativeFigureError(f"line {line}: model-state hash is invalid")
        model = str(raw["architecture"])
        if model in checkpoint_by_model and checkpoint_by_model[model] != checkpoint_hash:
            raise QualitativeFigureError(f"line {line}: checkpoint hash changes within model")
        if model in postload_by_model and postload_by_model[model] != postload_hash:
            raise QualitativeFigureError(f"line {line}: post-load hash changes within model")
        checkpoint_by_model[model] = str(checkpoint_hash)
        postload_by_model[model] = str(postload_hash)
        psnr = _finite(raw.get("psnr"), f"line {line} psnr")
        ssim = _finite(raw.get("ssim"), f"line {line} ssim")
        rmse = _finite(raw.get("rmse"), f"line {line} rmse")
        residual = _finite(
            raw.get("projection_residual"), f"line {line} projection_residual"
        )
        if not -1.0 <= ssim <= 1.0 or rmse < 0.0 or residual < 0.0:
            raise QualitativeFigureError(f"line {line}: metric range is invalid")
        if (
            _csv_bool(raw.get("clipped"), f"line {line} clipped")
            or _csv_bool(raw.get("normalized"), f"line {line} normalized")
        ):
            raise QualitativeFigureError("qualitative metrics must be raw and unclipped")
        status = raw.get("operating_point_status")
        claim = _csv_bool(
            raw.get("operating_point_claim_allowed"), f"line {line} claim gate"
        )
        fallback = _csv_bool(
            raw.get("diagnostic_fallback"), f"line {line} fallback gate"
        )
        if (
            status != gate["status"]
            or claim != gate["claim_allowed"]
            or fallback != gate["fallback_used"]
        ):
            raise QualitativeFigureError(f"line {line}: metric gate differs from manifest")
        lam = _finite(raw.get("lambda"), f"line {line} lambda")
        multiplier = _finite(raw.get("lambda_multiplier"), f"line {line} multiplier")
        eta = _finite(raw.get("eta"), f"line {line} eta")
        if spec[2] == "k0":
            if (lam, multiplier, eta) != (0.0, 0.0, 0.0):
                raise QualitativeFigureError(f"line {line}: k=0 DC parameters are nonzero")
        elif not all(
            math.isclose(observed, float(selected_dc[field]), rel_tol=1e-12, abs_tol=1e-15)
            for observed, field in (
                (lam, "lambda"),
                (multiplier, "lambda_multiplier"),
                (eta, "eta"),
            )
        ):
            raise QualitativeFigureError(f"line {line}: finite-DC parameters differ")
        source_run_id = raw.get("source_test_run_id")
        backend = raw.get("backend")
        if not isinstance(source_run_id, str) or not source_run_id:
            raise QualitativeFigureError(f"line {line}: test run ID is missing")
        if not isinstance(backend, str) or not backend:
            raise QualitativeFigureError(f"line {line}: backend is missing")
        source_run_ids.add(source_run_id)
        backends.add(backend)
        rows[key] = dict(raw)
    if set(rows) != expected_keys or len(patient_by_case) != 3:
        raise QualitativeFigureError("qualitative metric grid is incomplete")
    if len(source_run_ids) != 1 or len(backends) != 1:
        raise QualitativeFigureError("metric provenance differs across panels")
    if set(checkpoint_by_model) != {item[1] for item in PANEL_SPECS}:
        raise QualitativeFigureError("metric checkpoint evidence is incomplete")
    return rows, path


def authenticate_bundle(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, Path],
]:
    """Authenticate every byte consumed by the renderer before plotting."""

    manifest = _authenticate_source_manifest(manifest_path)
    outputs = _mapping(manifest.get("outputs"), "export outputs")
    if set(outputs) != {"case_arrays", "case_metrics", "hash_inventory"}:
        raise QualitativeFigureError("export output set differs")
    config = _mapping(manifest["scientific_config"], "export scientific_config")
    selected_cases = list(config["selected_cases"])
    selected_dc = _mapping(config["selected_finite_dc"], "selected finite DC")
    gate = _mapping(config["operating_point_gate"], "operating-point gate")
    case_outputs = _mapping(outputs["case_arrays"], "case-array outputs")
    arrays, array_paths = _load_arrays(
        manifest_path, case_outputs, selected_cases
    )
    metrics, metrics_path = _load_metrics(
        manifest_path,
        _mapping(outputs["case_metrics"], "metric output"),
        selected_cases,
        selected_dc,
        gate,
    )
    expected_inventory = {
        path.name: str(_mapping(case_outputs[label], f"case output {label}")["sha256"])
        for label, path in array_paths.items()
    }
    expected_inventory[metrics_path.name] = str(
        _mapping(outputs["case_metrics"], "metric output")["sha256"]
    )
    inventory_path = _read_inventory(
        manifest_path,
        _mapping(outputs["hash_inventory"], "hash inventory output"),
        expected_inventory,
    )
    # Portable image/metric parity: unlike projection residual and exact SSIM,
    # PSNR/RMSE can be recomputed without the physics stack or PyTorch.  This
    # catches a fully rehashed but internally mismatched image/annotation pair.
    for label, _quantile, _title in CASE_SPECS:
        gt = arrays[label]["ground_truth"]
        for panel_key in PANEL_KEYS:
            expected_psnr, expected_rmse = numpy_psnr_rmse(
                arrays[label][panel_key], gt
            )
            row = metrics[(label, panel_key)]
            if (
                abs(float(row["psnr"]) - float(expected_psnr[0])) > 1.0e-5
                or abs(float(row["rmse"]) - float(expected_rmse[0])) > 1.0e-5
            ):
                raise QualitativeFigureError(
                    f"stored image differs from PSNR/RMSE annotation: {label}/{panel_key}"
                )
    paths = {**array_paths, "metrics": metrics_path, "inventory": inventory_path}
    return manifest, arrays, metrics, paths


def _display_scales(
    arrays: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    gt_values = np.concatenate(
        [
            np.asarray(arrays[label]["ground_truth"], dtype=np.float64).ravel()
            for label, _quantile, _title in CASE_SPECS
        ]
    )
    low, high = np.percentile(
        gt_values, DISPLAY_PERCENTILES, method="linear"
    ).tolist()
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise QualitativeFigureError("GT-only display percentile range is degenerate")
    error_high = ERROR_RANGE_FRACTION * (high - low)
    if not math.isfinite(error_high) or error_high <= 0.0:
        raise QualitativeFigureError("GT-only absolute-error display range is invalid")
    return {
        "image": {"vmin": low, "vmax": high, "cmap": "gray"},
        "absolute_error": {"vmin": 0.0, "vmax": error_high, "cmap": "magma"},
        "rule": {
            "source": "ground_truth arrays only; reconstruction arrays and metrics excluded",
            "ground_truth_percentiles": list(DISPLAY_PERCENTILES),
            "percentile_method": "numpy linear",
            "absolute_error_ceiling": (
                "0.10 * (GT p99.5 - GT p0.5)"
            ),
            "absolute_error_fraction": ERROR_RANGE_FRACTION,
            "shared_across_cases_and_models": True,
            "display_only_clipping": True,
            "raw_arrays_modified": False,
        },
    }


def _atomic_savefig(fig: Any, path: Path, *, fmt: str, dpi: int | None = None) -> None:
    partial = path.with_name(path.name + ".partial")
    metadata = (
        {"Date": None, "Creator": "dc_make_qualitative_figure.py"}
        if fmt == "svg"
        else {"Software": "dc_make_qualitative_figure.py"}
    )
    try:
        fig.savefig(
            partial,
            format=fmt,
            dpi=dpi,
            metadata=metadata,
            facecolor="white",
            edgecolor="white",
        )
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _render(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    gate_label: str,
    claim_allowed: bool,
    selected_k: int,
    scales: Mapping[str, Any],
    svg_path: Path,
    png_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "dc.qualitative.figure.v1",
        }
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 7, figsize=(16.0, 13.0), squeeze=False)
    fig.subplots_adjust(
        left=0.055, right=0.915, bottom=0.035, top=0.905, wspace=0.035, hspace=0.34
    )
    image_mappable = None
    error_mappable = None
    for case_index, (selected, (label, _quantile, case_title)) in enumerate(
        zip(selected_cases, CASE_SPECS)
    ):
        case_arrays = arrays[label]
        image_row = 2 * case_index
        error_row = image_row + 1
        gt_axis = axes[image_row, 0]
        image_mappable = gt_axis.imshow(
            case_arrays["ground_truth"],
            cmap=scales["image"]["cmap"],
            vmin=scales["image"]["vmin"],
            vmax=scales["image"]["vmax"],
        )
        image_mappable.set_gid(f"image_{label}_ground_truth")
        gt_axis.set_ylabel(
            f"{case_title}\ncase {selected['case_id']}", fontsize=8, fontweight="bold"
        )
        gt_axis.set_xlabel("reference", fontsize=5, labelpad=1)
        if case_index == 0:
            gt_axis.set_title("Ground truth", fontsize=8, fontweight="bold")
        axes[error_row, 0].axis("off")
        axes[error_row, 0].text(
            0.5,
            0.5,
            "absolute error\n(common scale)",
            ha="center",
            va="center",
            fontsize=7,
            transform=axes[error_row, 0].transAxes,
        ).set_gid(f"absolute_error_scale_label_{label}")
        for column, (panel_key, _model, _phase, panel_title) in enumerate(
            PANEL_SPECS, start=1
        ):
            axis = axes[image_row, column]
            image = axis.imshow(
                case_arrays[panel_key],
                cmap=scales["image"]["cmap"],
                vmin=scales["image"]["vmin"],
                vmax=scales["image"]["vmax"],
            )
            image.set_gid(f"image_{label}_{panel_key}")
            row = metrics[(label, panel_key)]
            axis.set_xlabel(
                "P %.2f dB | S %.3f\nr %.4g"
                % (
                    float(row["psnr"]),
                    float(row["ssim"]),
                    float(row["projection_residual"]),
                ),
                fontsize=5,
                labelpad=1,
            )
            if case_index == 0:
                title = panel_title.replace(" + DC", f" + DC (k={selected_k})")
                axis.set_title(title, fontsize=8, fontweight="bold")
            error_axis = axes[error_row, column]
            error_mappable = error_axis.imshow(
                np.abs(case_arrays[f"error_{panel_key}"]),
                cmap=scales["absolute_error"]["cmap"],
                vmin=scales["absolute_error"]["vmin"],
                vmax=scales["absolute_error"]["vmax"],
            )
            error_mappable.set_gid(f"absolute_error_{label}_{panel_key}")
        for axis in axes[image_row].tolist() + axes[error_row].tolist():
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)

    if image_mappable is None or error_mappable is None:
        raise QualitativeFigureError("no qualitative panels were rendered")
    image_cax = fig.add_axes([0.928, 0.585, 0.012, 0.24])
    error_cax = fig.add_axes([0.928, 0.185, 0.012, 0.24])
    image_colorbar = fig.colorbar(image_mappable, cax=image_cax)
    image_colorbar.set_label("raw normalized value", fontsize=7)
    image_colorbar.ax.tick_params(labelsize=6)
    error_colorbar = fig.colorbar(error_mappable, cax=error_cax)
    error_colorbar.set_label("absolute error", fontsize=7)
    error_colorbar.ax.tick_params(labelsize=6)
    image_cax.set_gid("common_image_colorbar")
    error_cax.set_gid("common_absolute_error_colorbar")

    title_color = "#B2182B" if not claim_allowed else "#111111"
    title = fig.suptitle(
        "Figure C. Locked 50-view qualitative reconstructions and absolute errors\n"
        + gate_label,
        fontsize=12,
        fontweight="bold",
        color=title_color,
        y=0.972,
    )
    title.set_gid(
        "operating_point_claim_eligible"
        if claim_allowed
        else "operating_point_diagnostic_fallback_claim_ineligible"
    )
    if not claim_allowed:
        warning = fig.text(
            0.5,
            0.925,
            "DIAGNOSTIC FALLBACK — CLAIM-INELIGIBLE",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#B2182B",
            bbox={"facecolor": "#FDE0DD", "edgecolor": "#B2182B", "pad": 3.0},
        )
        warning.set_gid("diagnostic_fallback_claim_ineligible_banner")
    _atomic_savefig(fig, svg_path, fmt="svg")
    _atomic_savefig(fig, png_path, fmt="png", dpi=PNG_DPI)
    plt.close(fig)


def render_qualitative_figure(
    export_manifest: Path, output_dir: Path
) -> dict[str, Path]:
    manifest, arrays, metrics, source_paths = authenticate_bundle(export_manifest)
    config = _mapping(manifest["scientific_config"], "export scientific_config")
    selected_cases = list(config["selected_cases"])
    gate = _mapping(config["operating_point_gate"], "operating-point gate")
    selected_dc = _mapping(config["selected_finite_dc"], "selected finite DC")
    scales = _display_scales(arrays)

    out = _new_output_directory(output_dir)
    svg_path = out / "figure_c_qualitative_reconstructions.svg"
    png_path = out / "figure_c_qualitative_reconstructions.png"
    _render(
        arrays,
        metrics,
        selected_cases,
        str(config["operating_point_label"]),
        bool(gate["claim_allowed"]),
        int(selected_dc["k"]),
        scales,
        svg_path,
        png_path,
    )
    renderer_path = Path(__file__).resolve()
    result_manifest: dict[str, Any] = {
        "schema": FIGURE_SCHEMA,
        "status": "COMPLETE",
        "source": {
            "export_manifest": {
                "path": str(export_manifest.resolve()),
                "sha256": sha256(export_manifest),
                "run_id": manifest["run_id"],
                "schema": manifest["schema"],
                "command": manifest["command"],
                "status": manifest["status"],
            },
            "artifacts": {
                label: {"path": str(path.resolve()), "sha256": sha256(path)}
                for label, path in source_paths.items()
            },
        },
        "renderer": {
            "path": str(renderer_path),
            "sha256": sha256(renderer_path),
            "backend": "matplotlib Agg",
            "png_dpi": PNG_DPI,
        },
        "operating_point": {
            "gate": dict(gate),
            "label": config["operating_point_label"],
            "diagnostic_fallback": not bool(gate["claim_allowed"]),
            "claim_eligible": bool(gate["claim_allowed"]),
            "selected_finite_dc": dict(selected_dc),
        },
        "case_selection": {
            "rule": config["selection"],
            "cases": selected_cases,
            "interactive_selection": False,
        },
        "display_scales": scales,
        "layout": {
            "kind": "combined 3-case by image/absolute-error grid",
            "case_order": [item[0] for item in CASE_SPECS],
            "panel_order": ["ground_truth", *PANEL_KEYS],
            "rows_per_case": ["reconstruction", "absolute_error"],
            "metrics_shown": ["psnr", "ssim", "projection_residual"],
            "absolute_error_definition": "abs(reconstruction - ground_truth)",
        },
        "outputs": {
            "svg": {"path": str(svg_path.resolve()), "sha256": sha256(svg_path)},
            "png": {"path": str(png_path.resolve()), "sha256": sha256(png_path)},
        },
    }
    manifest_path = out / "figure_c_manifest.json"
    payload = json.dumps(result_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    try:
        with partial.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, manifest_path)
    finally:
        if partial.exists():
            partial.unlink()
    return {"svg": svg_path, "png": png_path, "manifest": manifest_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for label, path in render_qualitative_figure(
        args.export_manifest, args.out
    ).items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
