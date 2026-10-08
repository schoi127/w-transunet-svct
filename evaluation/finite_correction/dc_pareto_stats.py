#!/usr/bin/env python3
"""Patient-aware statistics for architecture-neutral finite-DC trajectories.

The script deliberately separates two analyses:

1. ``locked`` operating points are selected on validation data and then held
   fixed for test evaluation.  These are the deployable comparisons.
2. residual-matched curves are validation-target-locked aggregate frontier
   functionals interpolated from test-set Pareto knots.  Patient-cluster
   bootstrap inference is allowed only through an explicit coverage/claim
   gate.  These functionals estimate the trade-off curve but are not observed
   fractional-iteration or deployable reconstructions.

The formal CLI accepts only the runner's exact canonical validation/test CSV
schema and requires their COMPLETE manifests, validation lock, patient maps,
and current protocol.  All files are authenticated before the lock-selected
validation branch is materialized.  The internal ``Observation`` reader is
retained only as a pure testing/library utility; it is not reachable through
the formal CLI.  No missing cases, duplicate keys, non-finite metrics,
multiple trajectories, or mismatched architecture/iteration grids are
accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from dc_selection import aggregate_validation_candidates, select_all_targets
from dc_schemas import (
    CASE_METRICS_SCHEMA,
    ITERATIONS as PROTOCOL_ITERATIONS,
    LAMBDA_MULTIPLIERS as PROTOCOL_LAMBDA_MULTIPLIERS,
    MODELS as PROTOCOL_MODELS,
    SPLIT_FIREWALL_DISCLOSURE,
    json_sha256,
    load_json,
    validate_case_metric_row,
    validate_lock,
    validate_protocol,
)


ARCHITECTURES = ("UNet", "TransUNet", "WavTransUNet")
BASELINES = ("UNet", "TransUNet")
SPLITS = ("validation", "test")
METRICS = ("projection_residual", "psnr", "ssim", "rmse")
METRIC_DEFINITIONS = {
    "projection_residual": "||A*x-y||_2/||y||_2",
    "psnr": "peak signal-to-noise ratio",
    "ssim": "structural similarity index",
    "rmse": "root mean squared error",
}
METRIC_UNITS = {
    "projection_residual": "relative ratio",
    "psnr": "dB",
    "ssim": "unitless",
    "rmse": "normalized image intensity",
}
INTERNAL_REQUIRED_COLUMNS = {
    "split",
    "view",
    "case_id",
    "patient_id",
    "architecture",
    "trajectory_id",
    "dc_iteration",
    *METRICS,
}
RUNNER_REQUIRED_COLUMNS = {
    "split",
    "views",
    "case_id",
    "patient_id",
    "model",
    "candidate_id",
    "k",
    "projection_residual",
    "psnr_db",
    "ssim",
    "rmse",
}
# These optional runner fields are explicit, frozen provenance fields.  Any
# other extra column is rejected, preventing a misspelling from being ignored.
RUNNER_OPTIONAL_COLUMNS = {
    "schema",
    "checkpoint_sha256",
    "lambda",
    "lambda_multiplier",
    "eta",
    "diverged",
    "validation_selected",
    "backend",
    "source_run_id",
}
RUNNER_ALLOWED_COLUMNS = RUNNER_REQUIRED_COLUMNS | RUNNER_OPTIONAL_COLUMNS
RUNNER_TO_INTERNAL = {
    "split": "split",
    "views": "view",
    "case_id": "case_id",
    "patient_id": "patient_id",
    "model": "architecture",
    "candidate_id": "trajectory_id",
    "k": "dc_iteration",
    "projection_residual": "projection_residual",
    "psnr_db": "psnr",
    "ssim": "ssim",
    "rmse": "rmse",
}
TARGET_QUANTILES = (("q25", 0.25), ("q50", 0.50), ("q75", 0.75))
PRIMARY_TARGET = "q50"
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 0
MIN_VALID_BOOTSTRAP_FRACTION = 0.95
RAW_CASE_COLUMNS = (
    "schema",
    "split",
    "case_id",
    "patient_id",
    "views",
    "architecture",
    "checkpoint_sha256",
    "dc_iteration",
    "lambda",
    "lambda_multiplier",
    "eta",
    "psnr",
    "ssim",
    "rmse",
    "projection_residual",
    "diverged",
    "validation_selected",
    "backend",
    "source_run_id",
)
PACKAGE_FILENAMES = (
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
MODEL_SLUGS = {
    "UNet": "unet",
    "TransUNet": "transunet",
    "WavTransUNet": "wavtransunet",
}
RESIDUAL_MATCHED_WIDE_FIELDS = (
    "view",
    "target_id",
    "primary",
    "target_residual",
    "estimand",
    "inference_eligibility",
    "residual_matching_claim_allowed",
    "superiority_inference_eligible",
    "inferential_curve_eligible",
    "failure_code",
    "failure_reason",
    "uncertainty_method",
    "n_bootstrap",
    *(
        f"{slug}_{metric}{suffix}"
        for slug in MODEL_SLUGS.values()
        for metric in ("psnr", "ssim", "rmse")
        for suffix in ("", "_ci95_lo", "_ci95_hi", "_coverage_ok")
    ),
)
EMPTY_OUTPUT_FIELDS = {
    "targets": (
        "view",
        "target_id",
        "quantile_log_common_support",
        "target_residual",
        "common_validation_support_lo",
        "common_validation_support_hi",
        "primary",
        "source",
        "claim_allowed",
        "failure_reason",
    ),
    "dominance": (
        "view",
        "estimand",
        "objective",
        "region",
        "candidate",
        "baseline",
        "candidate_strictly_set_dominates",
        "claim_allowed",
        "failure_reason",
    ),
    "descriptive_points": (
        "analysis",
        "view",
        "target_id",
        "target_residual",
        "estimand",
        "architecture",
        "covered",
        "claim_allowed",
        "failure_reason",
    ),
    "descriptive_stats": (
        "analysis",
        "view",
        "target_id",
        "estimand",
        "candidate",
        "baseline",
        "metric",
        "mean_diff",
        "ci95_lo",
        "ci95_hi",
        "coverage_ok",
        "claim_allowed",
        "inference_eligibility",
        "inferential_curve_eligible",
        "superiority_inference_eligible",
        "deployable_reconstruction",
        "failure_reason",
    ),
    "area_points": (
        "view",
        "estimand",
        "architecture",
        "support_lo_q25",
        "support_hi_q75",
        "normalized_psnr_frontier_area_db",
        "coverage_ok",
        "claim_allowed",
        "failure_reason",
    ),
    "area_stats": (
        "view",
        "estimand",
        "candidate",
        "baseline",
        "normalized_area_diff_db",
        "ci95_lo",
        "ci95_hi",
        "coverage_ok",
        "claim_allowed",
        "failure_reason",
    ),
    "residual_matched_wide": RESIDUAL_MATCHED_WIDE_FIELDS,
}
if tuple(PROTOCOL_MODELS) != ARCHITECTURES:  # pragma: no cover - import-time contract
    raise RuntimeError("stats/model schema disagreement")


class AnalysisError(ValueError):
    """Raised when a scientific-integrity check fails closed."""


@dataclass(frozen=True)
class Observation:
    split: str
    view: int
    case_id: int
    patient_id: int
    architecture: str
    trajectory_id: str
    dc_iteration: int
    projection_residual: float
    psnr: float
    ssim: float
    rmse: float


@dataclass
class Cube:
    """Validated aligned arrays indexed by split/view/architecture/iteration."""

    views: tuple[int, ...]
    trajectories: dict[int, str]
    iterations: dict[int, tuple[int, ...]]
    case_ids: dict[tuple[str, int], np.ndarray]
    patient_ids: dict[tuple[str, int], np.ndarray]
    values: dict[tuple[str, int, str, int, str], np.ndarray]

    def get(
        self, split: str, view: int, architecture: str, iteration: int, metric: str
    ) -> np.ndarray:
        return self.values[(split, view, architecture, iteration, metric)]


@dataclass(frozen=True)
class FrozenAnalysisSpec:
    """All scientific choices authenticated from one validation lock."""

    view: int
    trajectory_id: str
    selected_lambda_multiplier: float
    selected_lambda: float
    selected_eta: float
    selected_k: int
    k_grid: tuple[int, ...]
    target_rows: tuple[dict[str, object], ...]
    common_support_lo: float | None
    common_support_hi: float | None
    claim_allowed: bool
    operating_point_claim_allowed: bool
    residual_matching_claim_allowed: bool
    residual_matching_failure_code: str
    failure_reason: str
    checkpoint_sha256: dict[str, str]
    lock_payload_sha256: str
    protocol_sha256: str


@dataclass(frozen=True)
class AuthenticatedViewBundle:
    cube: Cube
    spec: FrozenAnalysisSpec
    evidence: dict[str, object]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _current_package_hashes() -> dict[str, str]:
    """Hash the exact analysis/execution package named by runner manifests."""

    root = Path(__file__).resolve().parent
    output: dict[str, str] = {}
    for name in PACKAGE_FILENAMES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"formal package member is missing or a symlink: {path}")
        output[name] = sha256(path)
    return output


def _parse_int(value: str, field: str, line: int) -> int:
    try:
        parsed = int(value)
    except Exception as exc:  # pragma: no cover - exact parser exception varies
        raise AnalysisError(f"line {line}: invalid integer {field}={value!r}") from exc
    return parsed


def _parse_float(value: str, field: str, line: int) -> float:
    try:
        parsed = float(value)
    except Exception as exc:  # pragma: no cover
        raise AnalysisError(f"line {line}: invalid float {field}={value!r}") from exc
    if not math.isfinite(parsed):
        raise AnalysisError(f"line {line}: non-finite {field}={value!r}")
    return parsed


def _normalize_row(
    row: Mapping[str, str], schema: str, line: int
) -> dict[str, str]:
    if schema == "internal_alias_v1":
        return {name: row[name] for name in INTERNAL_REQUIRED_COLUMNS}
    if schema == "frozen_runner_v1":
        if row.get("diverged", "").strip().lower() in {"true", "1"}:
            raise AnalysisError(
                f"line {line}: diverged runner rows cannot enter Pareto statistics"
            )
        return {
            internal: row[runner]
            for runner, internal in RUNNER_TO_INTERNAL.items()
        }
    raise AssertionError(schema)  # pragma: no cover


def read_observations(
    path: Path, *, expected_split: str | None = None
) -> list[Observation]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if fields == INTERNAL_REQUIRED_COLUMNS:
            schema = "internal_alias_v1"
        elif RUNNER_REQUIRED_COLUMNS <= fields <= RUNNER_ALLOWED_COLUMNS:
            schema = "frozen_runner_v1"
        else:
            internal_missing = INTERNAL_REQUIRED_COLUMNS - fields
            runner_missing = RUNNER_REQUIRED_COLUMNS - fields
            runner_unknown = fields - RUNNER_ALLOWED_COLUMNS
            raise AnalysisError(
                "unrecognized input schema; accepted schemas are exact "
                "internal_alias_v1 or frozen_runner_v1 with explicit optional "
                f"fields. internal_missing={sorted(internal_missing)}, "
                f"runner_missing={sorted(runner_missing)}, "
                f"runner_unknown={sorted(runner_unknown)}"
            )
        out: list[Observation] = []
        for line, row in enumerate(reader, start=2):
            row = _normalize_row(row, schema, line)
            split = row["split"].strip()
            if expected_split is not None and split != expected_split:
                raise AnalysisError(
                    f"line {line}: {path} is the {expected_split} input but "
                    f"contains split={split!r}"
                )
            architecture = row["architecture"].strip()
            trajectory_id = row["trajectory_id"].strip()
            if not trajectory_id:
                raise AnalysisError(f"line {line}: empty trajectory_id")
            obs = Observation(
                split=split,
                view=_parse_int(row["view"], "view", line),
                case_id=_parse_int(row["case_id"], "case_id", line),
                patient_id=_parse_int(row["patient_id"], "patient_id", line),
                architecture=architecture,
                trajectory_id=trajectory_id,
                dc_iteration=_parse_int(
                    row["dc_iteration"], "dc_iteration", line
                ),
                projection_residual=_parse_float(
                    row["projection_residual"], "projection_residual", line
                ),
                psnr=_parse_float(row["psnr"], "psnr", line),
                ssim=_parse_float(row["ssim"], "ssim", line),
                rmse=_parse_float(row["rmse"], "rmse", line),
            )
            if obs.projection_residual <= 0:
                raise AnalysisError(
                    f"line {line}: projection_residual must be positive"
                )
            if obs.rmse < 0:
                raise AnalysisError(f"line {line}: rmse must be nonnegative")
            if obs.dc_iteration < 0:
                raise AnalysisError(
                    f"line {line}: dc_iteration must be nonnegative"
                )
            out.append(obs)
    if not out:
        raise AnalysisError("input contains no data rows")
    return out


def _csv_bool(value: str, field: str, line: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AnalysisError(f"line {line}: {field} must be exactly True or False")


def read_raw_case_rows(
    path: Path, *, expected_split: str, expected_view: int
) -> list[dict[str, object]]:
    """Read the runner's exact canonical CSV and validate every row."""

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_CASE_COLUMNS:
            raise AnalysisError(
                f"{path}: canonical header mismatch; expected "
                f"{RAW_CASE_COLUMNS}, found {tuple(reader.fieldnames or ())}"
            )
        for line, raw in enumerate(reader, start=2):
            item: dict[str, object] = dict(raw)
            try:
                for field in ("case_id", "patient_id", "views", "dc_iteration"):
                    item[field] = _parse_int(str(item[field]), field, line)
                for field in ("lambda", "lambda_multiplier", "eta"):
                    item[field] = _parse_float(str(item[field]), field, line)
                item["diverged"] = _csv_bool(str(item["diverged"]), "diverged", line)
                item["validation_selected"] = _csv_bool(
                    str(item["validation_selected"]), "validation_selected", line
                )
                for field in METRICS:
                    raw_value = str(item[field])
                    item[field] = (
                        None
                        if raw_value in ("", "None")
                        else _parse_float(raw_value, field, line)
                    )
                validate_case_metric_row(item)
            except AnalysisError:
                raise
            except Exception as exc:
                raise AnalysisError(
                    f"{path}, line {line}: canonical row validation failed: {exc}"
                ) from exc
            if item["schema"] != CASE_METRICS_SCHEMA:
                raise AnalysisError(f"{path}, line {line}: wrong row schema")
            if item["split"] != expected_split:
                raise AnalysisError(
                    f"{path}, line {line}: expected split={expected_split}, "
                    f"found {item['split']}"
                )
            if item["views"] != expected_view:
                raise AnalysisError(
                    f"{path}, line {line}: expected view={expected_view}, "
                    f"found {item['views']}"
                )
            rows.append(item)
    if not rows:
        raise AnalysisError(f"{path}: no case rows")
    return rows


def _require_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{where} must be an object")
    return value


def _require_complete_run_manifest(
    path: Path,
    *,
    command: str,
    split: str,
    view: int,
    protocol_sha256: str,
) -> dict[str, object]:
    try:
        document = load_json(path)
    except Exception as exc:
        raise AnalysisError(f"invalid run manifest {path}: {exc}") from exc
    expected = {
        "schema": "dc.run_manifest.v1",
        "status": "COMPLETE",
        "command": command,
        "run_kind": "FINAL",
        "views": view,
        "split": split,
        "protocol_sha256": protocol_sha256,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise AnalysisError(
                f"{path}: manifest {field}={document.get(field)!r}, expected {value!r}"
            )
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AnalysisError(f"{path}: manifest run_id is missing")
    _require_mapping(document.get("inputs"), f"{path}.inputs")
    _require_mapping(document.get("outputs"), f"{path}.outputs")
    if document.get("split_firewall") != SPLIT_FIREWALL_DISCLOSURE:
        raise AnalysisError(f"{path}: split-firewall disclosure changed")
    package_hashes = _require_mapping(
        document.get("package_sha256"), f"{path}.package_sha256"
    )
    current_hashes = _current_package_hashes()
    if dict(package_hashes) != current_hashes:
        raise AnalysisError(
            f"{path}: execution package hashes differ from the current package"
        )
    environment = _require_mapping(
        document.get("environment"), f"{path}.environment"
    )
    if environment.get("backend") != "astra_cuda":
        raise AnalysisError(f"{path}: formal manifest backend is not astra_cuda")
    if environment.get("astra_cuda") is not True:
        raise AnalysisError(f"{path}: ASTRA CUDA availability was not confirmed")
    return document


def _manifest_output(
    manifest: Mapping[str, object], name: str, where: str
) -> Mapping[str, object]:
    outputs = _require_mapping(manifest.get("outputs"), f"{where}.outputs")
    output = _require_mapping(outputs.get(name), f"{where}.outputs.{name}")
    digest = output.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AnalysisError(f"{where}.outputs.{name}.sha256 is invalid")
    return output


def _verified_asset_sha(
    manifest: Mapping[str, object], asset_id: str, where: str
) -> str:
    inputs = _require_mapping(manifest.get("inputs"), f"{where}.inputs")
    verification = _require_mapping(
        inputs.get("asset_verification"), f"{where}.inputs.asset_verification"
    )
    reports = verification.get("verified_assets")
    if not isinstance(reports, list):
        raise AnalysisError(f"{where}: verified_assets is absent")
    matching = [
        item
        for item in reports
        if isinstance(item, Mapping) and item.get("asset_id") == asset_id
    ]
    if len(matching) != 1 or matching[0].get("status") != "VERIFIED":
        raise AnalysisError(f"{where}: asset {asset_id} was not uniquely VERIFIED")
    digest = matching[0].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AnalysisError(f"{where}: asset {asset_id} SHA-256 is invalid")
    return digest


def _load_patient_map(
    path: Path, *, expected_cases: int, expected_patients: int, expected_sha256: str
) -> np.ndarray:
    if sha256(path) != expected_sha256:
        raise AnalysisError(f"patient-map SHA-256 mismatch: {path}")
    try:
        values = np.loadtxt(path, dtype=np.int64)
    except Exception as exc:
        raise AnalysisError(f"cannot read patient map {path}: {exc}") from exc
    if values.shape != (expected_cases,):
        raise AnalysisError(
            f"{path}: expected {expected_cases} patient IDs, found {values.shape}"
        )
    if len(np.unique(values)) != expected_patients:
        raise AnalysisError(
            f"{path}: expected {expected_patients} patients, "
            f"found {len(np.unique(values))}"
        )
    return values


def _same_float(a: object, b: object) -> bool:
    try:
        return float(a) == float(b)
    except Exception:
        return False


def _logical_observations_from_locked_rows(
    validation_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    view: int,
    selected_multiplier: float,
    selected_lambda: float,
    selected_eta: float,
    selected_k: int,
    k_grid: tuple[int, ...],
    common_scale: float,
    lambda_max_ata: float,
    checkpoint_sha256: Mapping[str, str],
    validation_patients: np.ndarray,
    test_patients: np.ndarray,
    validation_run_id: str,
    test_run_id: str,
    trajectory_id: str,
) -> list[Observation]:
    """Filter validation to the locked branch and enforce every row invariant."""

    def common_checks(
        rows: Sequence[Mapping[str, object]], split: str, patient_map: np.ndarray
    ) -> None:
        run_ids = {str(row["source_run_id"]) for row in rows}
        expected_run = validation_run_id if split == "validation" else test_run_id
        if run_ids != {expected_run}:
            raise AnalysisError(
                f"{split}: source_run_id mismatch; found={sorted(run_ids)}, "
                f"expected={expected_run}"
            )
        seen_raw: set[tuple[object, ...]] = set()
        for row in rows:
            case_id = int(row["case_id"])
            if not 0 <= case_id < len(patient_map):
                raise AnalysisError(f"{split}: case_id outside patient map: {case_id}")
            if int(row["patient_id"]) != int(patient_map[case_id]):
                raise AnalysisError(
                    f"{split}: patient mismatch at case {case_id}: "
                    f"CSV={row['patient_id']}, map={patient_map[case_id]}"
                )
            architecture = str(row["architecture"])
            if architecture not in ARCHITECTURES:
                raise AnalysisError(f"{split}: unknown architecture {architecture}")
            if row["checkpoint_sha256"] != checkpoint_sha256[architecture]:
                raise AnalysisError(
                    f"{split}: checkpoint hash mismatch for {architecture}"
                )
            if row["backend"] != "astra_cuda":
                raise AnalysisError(f"{split}: formal backend is not astra_cuda")
            if split == "test" and bool(row["diverged"]):
                raise AnalysisError(f"{split}: divergent row entered locked analysis")
            k = int(row["dc_iteration"])
            multiplier_key = 0.0 if k == 0 else float(row["lambda_multiplier"])
            raw_key = (case_id, architecture, k, multiplier_key)
            if raw_key in seen_raw:
                raise AnalysisError(f"{split}: duplicate raw row {raw_key}")
            seen_raw.add(raw_key)
            if k == 0:
                if not all(_same_float(row[field], 0.0) for field in ("lambda", "lambda_multiplier", "eta")):
                    raise AnalysisError(f"{split}: k=0 DC parameters must be zero")

    common_checks(validation_rows, "validation", validation_patients)
    common_checks(test_rows, "test", test_patients)

    expected_validation_raw = (
        len(validation_patients)
        * len(ARCHITECTURES)
        * (
            1
            + (len(k_grid) - 1) * len(PROTOCOL_LAMBDA_MULTIPLIERS)
        )
    )
    if len(validation_rows) != expected_validation_raw:
        raise AnalysisError(
            "raw validation sweep is incomplete: "
            f"{len(validation_rows)} rows != {expected_validation_raw}"
        )
    validation_coverage: dict[tuple[int, str], set[tuple[float, int]]] = {}
    for row in validation_rows:
        k = int(row["dc_iteration"])
        multiplier = float(row["lambda_multiplier"])
        key = (int(row["case_id"]), str(row["architecture"]))
        validation_coverage.setdefault(key, set()).add((multiplier, k))
        if k > 0:
            if multiplier not in PROTOCOL_LAMBDA_MULTIPLIERS:
                raise AnalysisError("validation row uses an unfrozen lambda multiplier")
            expected_lambda = multiplier * common_scale
            expected_eta = 1.0 / (lambda_max_ata + expected_lambda)
            if not (
                _same_float(row["lambda"], expected_lambda)
                and _same_float(row["eta"], expected_eta)
            ):
                raise AnalysisError(
                    "validation row violates lambda/eta formula for its branch"
                )
    expected_pairs = {(0.0, 0)} | {
        (float(multiplier), k)
        for multiplier in PROTOCOL_LAMBDA_MULTIPLIERS
        for k in k_grid
        if k > 0
    }
    if (
        len(validation_coverage)
        != len(validation_patients) * len(ARCHITECTURES)
        or any(pairs != expected_pairs for pairs in validation_coverage.values())
    ):
        raise AnalysisError("raw validation candidate grid is incomplete")

    if any(bool(row["validation_selected"]) for row in validation_rows):
        raise AnalysisError("validation sweep rows must not mark a test operating point")

    selected_validation = [
        row
        for row in validation_rows
        if int(row["dc_iteration"]) == 0
        or _same_float(row["lambda_multiplier"], selected_multiplier)
    ]
    expected_validation = len(validation_patients) * len(ARCHITECTURES) * len(k_grid)
    if len(selected_validation) != expected_validation:
        raise AnalysisError(
            "locked validation branch is incomplete: "
            f"{len(selected_validation)} rows != {expected_validation}"
        )
    expected_test = len(test_patients) * len(ARCHITECTURES) * len(k_grid)
    if len(test_rows) != expected_test:
        raise AnalysisError(
            f"locked test trajectory is incomplete: {len(test_rows)} != {expected_test}"
        )

    def locked_checks(rows: Sequence[Mapping[str, object]], split: str) -> None:
        coverage: dict[tuple[int, str], set[int]] = {}
        for row in rows:
            k = int(row["dc_iteration"])
            if k not in k_grid:
                raise AnalysisError(f"{split}: k={k} outside lock k_grid")
            if k > 0 and not (
                _same_float(row["lambda_multiplier"], selected_multiplier)
                and _same_float(row["lambda"], selected_lambda)
                and _same_float(row["eta"], selected_eta)
            ):
                raise AnalysisError(
                    f"{split}: row does not use lock-selected lambda/eta"
                )
            expected_flag = split == "test" and k == selected_k
            if bool(row["validation_selected"]) != expected_flag:
                raise AnalysisError(
                    f"{split}: validation_selected flag mismatch at "
                    f"case={row['case_id']}, model={row['architecture']}, k={k}"
                )
            key = (int(row["case_id"]), str(row["architecture"]))
            coverage.setdefault(key, set()).add(k)
            if any(row[metric] is None for metric in METRICS):
                raise AnalysisError(f"{split}: selected trajectory has null metrics")
        expected_grid = set(k_grid)
        if any(found != expected_grid for found in coverage.values()):
            raise AnalysisError(f"{split}: one or more case/model k grids are incomplete")
        expected_keys = (
            len(validation_patients) if split == "validation" else len(test_patients)
        ) * len(ARCHITECTURES)
        if len(coverage) != expected_keys:
            raise AnalysisError(f"{split}: case/model coverage is incomplete")

    locked_checks(selected_validation, "validation")
    locked_checks(test_rows, "test")

    observations: list[Observation] = []
    for row in list(selected_validation) + list(test_rows):
        observations.append(
            Observation(
                split=str(row["split"]),
                view=view,
                case_id=int(row["case_id"]),
                patient_id=int(row["patient_id"]),
                architecture=str(row["architecture"]),
                trajectory_id=trajectory_id,
                dc_iteration=int(row["dc_iteration"]),
                projection_residual=float(row["projection_residual"]),
                psnr=float(row["psnr"]),
                ssim=float(row["ssim"]),
                rmse=float(row["rmse"]),
            )
        )
    return observations


def build_cube(
    observations: Sequence[Observation],
    *,
    expected_views: Sequence[int] | None = None,
    expected_validation_cases: int | None = 3522,
    expected_test_cases: int | None = 3553,
    expected_validation_patients: int | None = 60,
    expected_test_patients: int | None = 60,
) -> Cube:
    """Validate complete pairing and convert rows to aligned arrays."""

    splits = {o.split for o in observations}
    if splits != set(SPLITS):
        raise AnalysisError(f"expected splits {SPLITS}, found {sorted(splits)}")
    architectures = {o.architecture for o in observations}
    if architectures != set(ARCHITECTURES):
        raise AnalysisError(
            f"expected architectures {ARCHITECTURES}, found {sorted(architectures)}"
        )
    views = tuple(sorted({o.view for o in observations}, reverse=True))
    if expected_views is not None and set(views) != set(expected_views):
        raise AnalysisError(
            f"expected views {sorted(expected_views)}, found {sorted(views)}"
        )

    # A case must map to exactly one patient within a split/view, independent
    # of architecture and iteration.
    case_patient: dict[tuple[str, int, int], int] = {}
    unique_keys: set[tuple[str, int, int, str, str, int]] = set()
    grouped: dict[
        tuple[str, int, str, str, int], dict[int, Observation]
    ] = {}
    for o in observations:
        cp_key = (o.split, o.view, o.case_id)
        previous = case_patient.setdefault(cp_key, o.patient_id)
        if previous != o.patient_id:
            raise AnalysisError(
                f"patient mismatch for split/view/case {cp_key}: "
                f"{previous} vs {o.patient_id}"
            )
        unique_key = (
            o.split,
            o.view,
            o.case_id,
            o.architecture,
            o.trajectory_id,
            o.dc_iteration,
        )
        if unique_key in unique_keys:
            raise AnalysisError(f"duplicate row key: {unique_key}")
        unique_keys.add(unique_key)
        grouped.setdefault(
            (o.split, o.view, o.architecture, o.trajectory_id, o.dc_iteration), {}
        )[o.case_id] = o

    trajectories: dict[int, str] = {}
    iterations: dict[int, tuple[int, ...]] = {}
    case_ids: dict[tuple[str, int], np.ndarray] = {}
    patient_ids: dict[tuple[str, int], np.ndarray] = {}
    values: dict[tuple[str, int, str, int, str], np.ndarray] = {}

    expected_counts = {
        "validation": expected_validation_cases,
        "test": expected_test_cases,
    }
    expected_patients = {
        "validation": expected_validation_patients,
        "test": expected_test_patients,
    }

    for view in views:
        view_rows = [o for o in observations if o.view == view]
        traj_by_arm_split: dict[tuple[str, str], set[str]] = {}
        k_by_arm_split: dict[tuple[str, str], set[int]] = {}
        for o in view_rows:
            traj_by_arm_split.setdefault((o.split, o.architecture), set()).add(
                o.trajectory_id
            )
            k_by_arm_split.setdefault((o.split, o.architecture), set()).add(
                o.dc_iteration
            )
        for key, found in traj_by_arm_split.items():
            if len(found) != 1:
                raise AnalysisError(
                    f"multiple trajectory_id values for view={view}, {key}: "
                    f"{sorted(found)}"
                )
        all_trajectories = {
            next(iter(found)) for found in traj_by_arm_split.values()
        }
        if len(all_trajectories) != 1:
            raise AnalysisError(
                f"trajectory_id differs across split/architecture for view={view}: "
                f"{sorted(all_trajectories)}"
            )
        trajectories[view] = next(iter(all_trajectories))

        all_k_grids = {tuple(sorted(v)) for v in k_by_arm_split.values()}
        if len(all_k_grids) != 1:
            raise AnalysisError(
                f"iteration grid differs across split/architecture for view={view}: "
                f"{sorted(all_k_grids)}"
            )
        k_grid = next(iter(all_k_grids))
        if not k_grid or k_grid[0] != 0:
            raise AnalysisError(f"view={view}: iteration grid must include k=0")
        iterations[view] = k_grid

        for split in SPLITS:
            ids = sorted(
                {o.case_id for o in view_rows if o.split == split}
            )
            if ids != list(range(len(ids))):
                raise AnalysisError(
                    f"{split}, view={view}: case_id must be contiguous 0..N-1"
                )
            expected_n = expected_counts[split]
            if expected_n is not None and len(ids) != expected_n:
                raise AnalysisError(
                    f"{split}, view={view}: expected {expected_n} cases, "
                    f"found {len(ids)}"
                )
            cids = np.asarray(ids, dtype=np.int64)
            pids = np.asarray(
                [case_patient[(split, view, i)] for i in ids], dtype=np.int64
            )
            expected_p = expected_patients[split]
            if expected_p is not None and len(np.unique(pids)) != expected_p:
                raise AnalysisError(
                    f"{split}, view={view}: expected {expected_p} patients, "
                    f"found {len(np.unique(pids))}"
                )
            case_ids[(split, view)] = cids
            patient_ids[(split, view)] = pids

            for architecture in ARCHITECTURES:
                for k in k_grid:
                    key = (
                        split,
                        view,
                        architecture,
                        trajectories[view],
                        k,
                    )
                    rows = grouped.get(key)
                    if rows is None:
                        raise AnalysisError(f"missing group {key}")
                    if set(rows) != set(ids):
                        missing = sorted(set(ids) - set(rows))[:5]
                        extra = sorted(set(rows) - set(ids))[:5]
                        raise AnalysisError(
                            f"case-set mismatch for {key}; missing={missing}, extra={extra}"
                        )
                    ordered = [rows[i] for i in ids]
                    for metric in METRICS:
                        arr = np.asarray(
                            [getattr(o, metric) for o in ordered], dtype=np.float64
                        )
                        if not np.all(np.isfinite(arr)):
                            raise AnalysisError(f"non-finite values after alignment: {key}")
                        values[(split, view, architecture, k, metric)] = arr

    # The patient map is an experimental property and must not change with view.
    for split in SPLITS:
        reference = patient_ids[(split, views[0])]
        for view in views[1:]:
            if not np.array_equal(reference, patient_ids[(split, view)]):
                raise AnalysisError(
                    f"{split}: case-to-patient mapping differs across views"
                )

    return Cube(
        views=views,
        trajectories=trajectories,
        iterations=iterations,
        case_ids=case_ids,
        patient_ids=patient_ids,
        values=values,
    )


def _recompute_protocol_support(
    cube: Cube, view: int
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Recompute every model range and their (possibly empty) intersection."""

    ranges: dict[str, tuple[float, float]] = {}
    for architecture in ARCHITECTURES:
        means = [
            float(
                np.mean(
                    cube.get(
                        "validation",
                        view,
                        architecture,
                        k,
                        "projection_residual",
                    )
                )
            )
            for k in cube.iterations[view]
        ]
        ranges[architecture] = (min(means), max(means))
    lower = max(item[0] for item in ranges.values())
    upper = min(item[1] for item in ranges.values())
    if not (math.isfinite(lower) and math.isfinite(upper) and lower > 0):
        raise AnalysisError(
            f"view={view}: recomputed selected-lambda residual ranges are invalid"
        )
    return lower, upper, ranges


def _recompute_protocol_targets(
    cube: Cube, view: int
) -> tuple[float, float, dict[str, float]]:
    """Independent canary for a positive-width common residual support."""

    lower, upper, _ = _recompute_protocol_support(cube, view)
    if not upper > lower:
        raise AnalysisError(
            f"view={view}: recomputed supports have no positive-width common interval"
        )
    log_lower = math.log(lower)
    log_width = math.log(upper) - log_lower
    targets = {
        label: math.exp(log_lower + quantile * log_width)
        for label, quantile in TARGET_QUANTILES
    }
    return lower, upper, targets


def _assert_close_locked(observed: float, expected: float, where: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise AnalysisError(
            f"{where} mismatch: recomputed={observed:.17g}, "
            f"locked={expected:.17g}"
        )


def load_authenticated_view_bundle(
    *,
    view: int,
    validation_csv: Path,
    sweep_manifest_path: Path,
    lock_path: Path,
    test_csv: Path,
    test_manifest_path: Path,
    validation_patient_map_path: Path,
    test_patient_map_path: Path,
    protocol_path: Path,
    expected_validation_cases: int = 3522,
    expected_test_cases: int = 3553,
    expected_patients: int = 60,
) -> AuthenticatedViewBundle:
    """Authenticate and materialize one formal, lock-selected view bundle."""

    try:
        protocol = load_json(protocol_path)
        validate_protocol(protocol)
    except Exception as exc:
        raise AnalysisError(f"invalid current protocol {protocol_path}: {exc}") from exc
    protocol_hash = json_sha256(protocol)

    try:
        lock = load_json(lock_path)
        validate_lock(lock)
    except Exception as exc:
        raise AnalysisError(f"invalid DC lock {lock_path}: {exc}") from exc
    payload = _require_mapping(lock.get("payload"), "lock.payload")
    if payload.get("protocol_sha256") != protocol_hash:
        raise AnalysisError("lock is not bound to the supplied current protocol")
    if payload.get("selection_split") != "validation":
        raise AnalysisError("lock selection split is not validation")
    if payload.get("views") != view:
        raise AnalysisError(
            f"lock view={payload.get('views')!r}, requested view={view}"
        )
    payload_hash = str(lock["payload_sha256"])
    lock_file_hash = sha256(lock_path)

    sweep = _require_complete_run_manifest(
        sweep_manifest_path,
        command="sweep-validation",
        split="validation",
        view=view,
        protocol_sha256=protocol_hash,
    )
    test_manifest = _require_complete_run_manifest(
        test_manifest_path,
        command="evaluate-test",
        split="test",
        view=view,
        protocol_sha256=protocol_hash,
    )
    if sweep["run_id"] == test_manifest["run_id"]:
        raise AnalysisError("validation and test manifests reuse the same run_id")
    if payload.get("validation_manifest_sha256") != sha256(sweep_manifest_path):
        raise AnalysisError("lock does not bind the supplied validation sweep manifest")

    sweep_coverage = _require_mapping(sweep.get("coverage"), "sweep.coverage")
    if (
        sweep_coverage.get("cases") != expected_validation_cases
        or sweep_coverage.get("models") != len(ARCHITECTURES)
        or sweep_coverage.get("baseline_rows")
        != expected_validation_cases * len(ARCHITECTURES)
    ):
        raise AnalysisError("validation sweep manifest coverage is incomplete")
    test_coverage = _require_mapping(
        test_manifest.get("coverage"), "test_manifest.coverage"
    )
    expected_test_rows = (
        expected_test_cases * len(ARCHITECTURES) * len(PROTOCOL_ITERATIONS)
    )
    if (
        test_coverage.get("cases") != expected_test_cases
        or test_coverage.get("models") != len(ARCHITECTURES)
        or test_coverage.get("rows") != expected_test_rows
    ):
        raise AnalysisError("test manifest coverage is incomplete")

    validation_hash = sha256(validation_csv)
    test_hash = sha256(test_csv)
    validation_output = _manifest_output(
        sweep, "case_metrics", "sweep_manifest"
    )
    test_output = _manifest_output(test_manifest, "case_metrics", "test_manifest")
    if validation_output.get("sha256") != validation_hash:
        raise AnalysisError("validation CSV does not match sweep-manifest output hash")
    if payload.get("validation_case_metrics_sha256") != validation_hash:
        raise AnalysisError("validation CSV does not match lock hash")
    if test_output.get("sha256") != test_hash:
        raise AnalysisError("test CSV does not match test-manifest output hash")

    test_inputs = _require_mapping(test_manifest.get("inputs"), "test_manifest.inputs")
    test_lock = _require_mapping(test_inputs.get("lock"), "test_manifest.inputs.lock")
    if test_lock.get("sha256") != lock_file_hash:
        raise AnalysisError("test manifest used a different lock file")
    if test_lock.get("payload_sha256") != payload_hash:
        raise AnalysisError("test manifest used a different lock payload")

    selected = _require_mapping(payload.get("selected"), "lock.payload.selected")
    try:
        selected_multiplier = float(selected["lambda_multiplier"])
        selected_lambda = float(selected["lambda"])
        selected_eta = float(selected["eta"])
        selected_k = int(selected["k"])
    except Exception as exc:
        raise AnalysisError("lock selected parameters are incomplete") from exc
    if not all(
        math.isfinite(value) and value > 0
        for value in (selected_multiplier, selected_lambda, selected_eta)
    ):
        raise AnalysisError("lock selected lambda parameters must be finite and positive")
    if selected_multiplier not in PROTOCOL_LAMBDA_MULTIPLIERS:
        raise AnalysisError("lock selected multiplier is outside the current protocol")
    raw_grid = payload.get("k_grid")
    if not isinstance(raw_grid, list) or any(type(k) is not int for k in raw_grid):
        raise AnalysisError("lock k_grid is invalid")
    k_grid = tuple(raw_grid)
    if k_grid != tuple(PROTOCOL_ITERATIONS):
        raise AnalysisError(
            f"lock k_grid={k_grid} differs from current protocol={tuple(PROTOCOL_ITERATIONS)}"
        )
    if selected_k not in k_grid or selected_k == 0:
        raise AnalysisError("lock selected k is outside the finite-DC grid")

    scientific_config = _require_mapping(
        sweep.get("scientific_config"), "sweep.scientific_config"
    )
    try:
        common_scale = float(scientific_config["common_scale"])
        lambda_max_ata = float(scientific_config["lambda_max_ata"])
    except Exception as exc:
        raise AnalysisError("validation scientific configuration is incomplete") from exc
    if not (
        math.isfinite(common_scale)
        and common_scale > 0
        and math.isfinite(lambda_max_ata)
        and lambda_max_ata > 0
    ):
        raise AnalysisError("validation scale or operator norm is invalid")
    if tuple(scientific_config.get("k_grid", ())) != tuple(PROTOCOL_ITERATIONS):
        raise AnalysisError("validation-manifest k grid differs from the protocol")
    if tuple(float(x) for x in scientific_config.get("lambda_multipliers", ())) != tuple(
        PROTOCOL_LAMBDA_MULTIPLIERS
    ):
        raise AnalysisError("validation-manifest lambda grid differs from the protocol")
    if not (
        _same_float(payload.get("common_scale"), common_scale)
        and _same_float(payload.get("lambda_max_ata"), lambda_max_ata)
    ):
        raise AnalysisError("lock and validation manifest disagree on DC scaling")
    if not _same_float(selected_lambda, selected_multiplier * common_scale):
        raise AnalysisError("lock selected lambda violates the common-scale formula")
    if not _same_float(selected_eta, 1.0 / (lambda_max_ata + selected_lambda)):
        raise AnalysisError("lock selected eta violates the frozen step-size formula")

    checkpoint_map_raw = _require_mapping(
        payload.get("checkpoint_sha256"), "lock.payload.checkpoint_sha256"
    )
    if set(checkpoint_map_raw) != set(ARCHITECTURES):
        raise AnalysisError("lock checkpoint map is incomplete")
    checkpoint_map = {name: str(checkpoint_map_raw[name]) for name in ARCHITECTURES}
    if any(len(digest) != 64 for digest in checkpoint_map.values()):
        raise AnalysisError("lock contains an invalid checkpoint SHA-256")
    checkpoint_assets = {
        "UNet": f"checkpoint_unet_{view}",
        "TransUNet": f"checkpoint_transunet_{view}",
        "WavTransUNet": f"checkpoint_wavtransunet_{view}",
    }
    for architecture, asset_id in checkpoint_assets.items():
        sweep_digest = _verified_asset_sha(sweep, asset_id, "sweep_manifest")
        test_digest = _verified_asset_sha(test_manifest, asset_id, "test_manifest")
        if sweep_digest != checkpoint_map[architecture] or test_digest != checkpoint_map[architecture]:
            raise AnalysisError(
                f"verified checkpoint hash chain failed for {architecture}"
            )

    sweep_patient_hash = _verified_asset_sha(
        sweep, "patient_map_validation", "sweep_manifest"
    )
    if payload.get("validation_patient_map_sha256") != sweep_patient_hash:
        raise AnalysisError("lock and sweep disagree on validation patient map")
    test_patient_hash = _verified_asset_sha(
        test_manifest, "patient_map_test", "test_manifest"
    )
    validation_patients = _load_patient_map(
        validation_patient_map_path,
        expected_cases=expected_validation_cases,
        expected_patients=expected_patients,
        expected_sha256=sweep_patient_hash,
    )
    test_patients = _load_patient_map(
        test_patient_map_path,
        expected_cases=expected_test_cases,
        expected_patients=expected_patients,
        expected_sha256=test_patient_hash,
    )

    validation_rows = read_raw_case_rows(
        validation_csv, expected_split="validation", expected_view=view
    )
    test_rows = read_raw_case_rows(
        test_csv, expected_split="test", expected_view=view
    )
    manifest_test_rows = test_output.get("rows")
    if type(manifest_test_rows) is not int or manifest_test_rows != len(test_rows):
        raise AnalysisError(
            f"test-manifest row count {manifest_test_rows!r} != CSV {len(test_rows)}"
        )

    # The lock schema intentionally permits extension fields, so a self-hash
    # alone does not authenticate the scientific choice.  Re-run the frozen
    # validation selector on the complete raw sweep and require byte-canonical
    # equality of the selected candidate and target definition.
    try:
        summaries = aggregate_validation_candidates(
            validation_rows, expected_cases=expected_validation_cases
        )
        recomputed_selection = select_all_targets(
            summaries, require_complete_grid=True
        )
    except Exception as exc:
        raise AnalysisError(
            f"independent validation selection recomputation failed: {exc}"
        ) from exc
    recomputed_operating = _require_mapping(
        recomputed_selection.get("selected_operating_point"),
        "recomputed.selected_operating_point",
    )
    if json_sha256(selected) != json_sha256(recomputed_operating.get("selected")):
        raise AnalysisError("lock selected candidate differs from frozen recomputation")
    operating_gate = _require_mapping(
        payload.get("operating_point_gate"), "lock.payload.operating_point_gate"
    )
    expected_gate = {
        field: recomputed_operating.get(field)
        for field in (
            "status",
            "claim_allowed",
            "psnr_floor_db",
            "fallback_used",
            "failure_reason",
        )
    }
    if dict(operating_gate) != expected_gate:
        raise AnalysisError("lock operating-point gate differs from frozen recomputation")
    matching = _require_mapping(
        payload.get("residual_matching"), "lock.payload.residual_matching"
    )
    recomputed_matching = _require_mapping(
        recomputed_selection.get("target_definition"),
        "recomputed.target_definition",
    )
    if json_sha256(matching) != json_sha256(recomputed_matching):
        raise AnalysisError("lock residual targets differ from frozen recomputation")

    trajectory_id = f"lock:{payload_hash}"
    observations = _logical_observations_from_locked_rows(
        validation_rows,
        test_rows,
        view=view,
        selected_multiplier=selected_multiplier,
        selected_lambda=selected_lambda,
        selected_eta=selected_eta,
        selected_k=selected_k,
        k_grid=k_grid,
        common_scale=common_scale,
        lambda_max_ata=lambda_max_ata,
        checkpoint_sha256=checkpoint_map,
        validation_patients=validation_patients,
        test_patients=test_patients,
        validation_run_id=str(sweep["run_id"]),
        test_run_id=str(test_manifest["run_id"]),
        trajectory_id=trajectory_id,
    )
    cube = build_cube(
        observations,
        expected_views=[view],
        expected_validation_cases=expected_validation_cases,
        expected_test_cases=expected_test_cases,
        expected_validation_patients=expected_patients,
        expected_test_patients=expected_patients,
    )

    interval = matching.get("interval")
    targets_raw = matching.get("targets")
    common_contract = (
        matching.get("primary") == PRIMARY_TARGET
        and matching.get("spacing") == "log"
        and matching.get("extrapolation") == "forbidden"
        and isinstance(targets_raw, Mapping)
    )
    if not common_contract:
        raise AnalysisError("lock residual-matching definition is incomplete or changed")

    matching_failure_code = ""
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and set(targets_raw) == {label for label, _ in TARGET_QUANTILES}
    ):
        if matching.get("status") not in ("PASS", "FAIL"):
            raise AnalysisError("target-bearing residual matching has invalid status")
        support_lo, support_hi = float(interval[0]), float(interval[1])
        locked_targets = {
            label: float(targets_raw[label]) for label, _ in TARGET_QUANTILES
        }
        if not (
            support_lo > 0
            and support_hi > support_lo
            and all(math.isfinite(value) for value in locked_targets.values())
        ):
            raise AnalysisError(
                "lock residual support/targets are non-finite or lack positive width"
            )
        recomputed_lo, recomputed_hi, recomputed_targets = (
            _recompute_protocol_targets(cube, view)
        )
        _assert_close_locked(
            recomputed_lo, support_lo, "common residual support lower"
        )
        _assert_close_locked(
            recomputed_hi, support_hi, "common residual support upper"
        )
        for label, value in locked_targets.items():
            _assert_close_locked(recomputed_targets[label], value, f"target {label}")
    elif matching.get("status") == "FAIL":
        matching_failure_code = str(matching.get("failure_code", ""))
        expected_fail_keys = {
            "status",
            "claim_allowed",
            "fallback_used",
            "failure_code",
            "failure_reason",
            "views",
            "interval",
            "spacing",
            "targets",
            "primary",
            "extrapolation",
            "selected_lambda_multiplier",
            "selected_lambda",
            "trajectory_k",
            "model_ranges",
        }
        if (
            set(matching) != expected_fail_keys
            or matching_failure_code != "NO_COMMON_INTERVAL"
            or matching.get("claim_allowed") is not False
            or matching.get("fallback_used") is not True
            or not isinstance(matching.get("failure_reason"), str)
            or not matching.get("failure_reason")
            or matching.get("views") != view
            or interval is not None
            or dict(targets_raw) != {}
            or not _same_float(
                matching.get("selected_lambda_multiplier"), selected_multiplier
            )
            or not _same_float(matching.get("selected_lambda"), selected_lambda)
            or tuple(matching.get("trajectory_k", ())) != k_grid
        ):
            raise AnalysisError("structured NO_COMMON_INTERVAL lock is invalid")
        model_ranges = _require_mapping(
            matching.get("model_ranges"), "lock.residual_matching.model_ranges"
        )
        if set(model_ranges) != set(ARCHITECTURES):
            raise AnalysisError("NO_COMMON_INTERVAL model ranges are incomplete")
        recomputed_lo, recomputed_hi, recomputed_ranges = (
            _recompute_protocol_support(cube, view)
        )
        if recomputed_hi > recomputed_lo:
            raise AnalysisError(
                "NO_COMMON_INTERVAL lock contradicts a positive-width recomputed interval"
            )
        for architecture in ARCHITECTURES:
            bounds = model_ranges[architecture]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise AnalysisError(
                    f"NO_COMMON_INTERVAL range is invalid for {architecture}"
                )
            for observed, expected in zip(
                recomputed_ranges[architecture], (float(bounds[0]), float(bounds[1]))
            ):
                _assert_close_locked(
                    observed, expected, f"{architecture} residual-range canary"
                )
        support_lo = support_hi = None
        locked_targets: dict[str, float] = {}
    else:
        raise AnalysisError("residual-matching status must be PASS or FAIL")

    op_allowed = operating_gate.get("claim_allowed") is True
    match_allowed = matching.get("claim_allowed") is True
    claim_allowed = op_allowed and match_allowed
    reasons = [
        str(reason)
        for reason in (
            operating_gate.get("failure_reason"), matching.get("failure_reason")
        )
        if reason not in (None, "")
    ]
    failure_reason = "; ".join(reasons)
    if not claim_allowed and not failure_reason:
        failure_reason = "one_or_more_validation_gates_disallowed_claims"
    target_rows = (
        tuple(
            {
                "view": view,
                "target_id": label,
                "quantile_log_common_support": quantile,
                "target_residual": locked_targets[label],
                "common_validation_support_lo": support_lo,
                "common_validation_support_hi": support_hi,
                "primary": label == PRIMARY_TARGET,
                "source": "authenticated_dc_lock",
                "claim_allowed": claim_allowed,
                "failure_reason": failure_reason,
            }
            for label, quantile in TARGET_QUANTILES
        )
        if locked_targets
        else ()
    )
    spec = FrozenAnalysisSpec(
        view=view,
        trajectory_id=trajectory_id,
        selected_lambda_multiplier=selected_multiplier,
        selected_lambda=selected_lambda,
        selected_eta=selected_eta,
        selected_k=selected_k,
        k_grid=k_grid,
        target_rows=target_rows,
        common_support_lo=support_lo,
        common_support_hi=support_hi,
        claim_allowed=claim_allowed,
        operating_point_claim_allowed=op_allowed,
        residual_matching_claim_allowed=match_allowed,
        residual_matching_failure_code=matching_failure_code,
        failure_reason=failure_reason,
        checkpoint_sha256=checkpoint_map,
        lock_payload_sha256=payload_hash,
        protocol_sha256=protocol_hash,
    )
    evidence: dict[str, object] = {
        "view": view,
        "protocol": str(protocol_path),
        "protocol_canonical_sha256": protocol_hash,
        "protocol_file_sha256": sha256(protocol_path),
        "lock": str(lock_path),
        "lock_file_sha256": lock_file_hash,
        "lock_payload_sha256": payload_hash,
        "validation_csv": str(validation_csv),
        "validation_csv_sha256": validation_hash,
        "sweep_manifest": str(sweep_manifest_path),
        "sweep_manifest_sha256": sha256(sweep_manifest_path),
        "test_csv": str(test_csv),
        "test_csv_sha256": test_hash,
        "test_manifest": str(test_manifest_path),
        "test_manifest_sha256": sha256(test_manifest_path),
        "validation_patient_map": str(validation_patient_map_path),
        "validation_patient_map_sha256": sweep_patient_hash,
        "test_patient_map": str(test_patient_map_path),
        "test_patient_map_sha256": test_patient_hash,
        "validation_run_id": sweep["run_id"],
        "test_run_id": test_manifest["run_id"],
        "authenticated": True,
    }
    return AuthenticatedViewBundle(cube=cube, spec=spec, evidence=evidence)


def dominates(point_a: tuple[float, float], point_b: tuple[float, float]) -> bool:
    """Exact minimization(R)/maximization(P) dominance predicate."""

    ra, pa = point_a
    rb, pb = point_b
    return ra <= rb and pa >= pb and (ra < rb or pa > pb)


def pareto_front_indices(points: Sequence[tuple[float, float]]) -> list[int]:
    """Return deterministic nondominated indices, deduplicating exact ties."""

    keep: list[int] = []
    seen: set[tuple[float, float]] = set()
    for i, point in enumerate(points):
        if point in seen:
            continue
        seen.add(point)
        if not any(
            j != i and dominates(other, point)
            for j, other in enumerate(points)
        ):
            keep.append(i)
    return keep


def set_strictly_dominates(
    candidate_points: Sequence[tuple[float, float]],
    reference_points: Sequence[tuple[float, float]],
) -> bool:
    """True only if every reference Pareto point is strictly dominated."""

    if not candidate_points or not reference_points:
        return False
    candidate_front = [
        candidate_points[i] for i in pareto_front_indices(candidate_points)
    ]
    reference_front = [
        reference_points[i] for i in pareto_front_indices(reference_points)
    ]
    return all(
        any(dominates(candidate, reference) for candidate in candidate_front)
        for reference in reference_front
    )


def weakly_covers_set(
    candidate_points: Sequence[tuple[float, float]],
    reference_points: Sequence[tuple[float, float]],
) -> bool:
    """Weak set coverage, retained separately from strict dominance."""

    if not candidate_points or not reference_points:
        return False
    candidate_front = [
        candidate_points[i] for i in pareto_front_indices(candidate_points)
    ]
    reference_front = [
        reference_points[i] for i in pareto_front_indices(reference_points)
    ]
    return all(
        any(c[0] <= r[0] and c[1] >= r[1] for c in candidate_front)
        for r in reference_front
    )


def patient_components(
    values: np.ndarray, patient_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, codes = np.unique(patient_ids, return_inverse=True)
    counts = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    sums = np.bincount(codes, weights=values, minlength=len(unique))
    return sums, counts, sums / counts


def aggregate_value(
    values: np.ndarray, patient_ids: np.ndarray, estimand: str
) -> tuple[float, float, float]:
    if estimand == "slice_weighted":
        return (
            float(np.mean(values)),
            float(np.median(values)),
            float(np.std(values, ddof=1)),
        )
    if estimand == "equal_patient":
        _, _, means = patient_components(values, patient_ids)
        return (
            float(np.mean(means)),
            float(np.median(means)),
            float(np.std(means, ddof=1)),
        )
    raise AnalysisError(f"unknown estimand {estimand!r}")


def aggregate_points(cube: Cube) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split in SPLITS:
        for view in cube.views:
            pids = cube.patient_ids[(split, view)]
            for estimand in ("slice_weighted", "equal_patient"):
                for architecture in ARCHITECTURES:
                    for k in cube.iterations[view]:
                        rec: dict[str, object] = {
                            "split": split,
                            "view": view,
                            "estimand": estimand,
                            "architecture": architecture,
                            "trajectory_id": cube.trajectories[view],
                            "dc_iteration": k,
                            "n_cases": len(pids),
                            "n_patients": len(np.unique(pids)),
                        }
                        for metric in METRICS:
                            vals = cube.get(split, view, architecture, k, metric)
                            mean, median, sd = aggregate_value(vals, pids, estimand)
                            rec[metric] = mean
                            rec[f"{metric}_median"] = median
                            rec[f"{metric}_sd"] = sd
                        records.append(rec)

    # Pareto membership uses the PSNR objective.  A separate SSIM flag is
    # reported, but SSIM/RMSE at PSNR-matched points always reuse PSNR knots.
    for split in SPLITS:
        for view in cube.views:
            for estimand in ("slice_weighted", "equal_patient"):
                for architecture in ARCHITECTURES:
                    subset = [
                        r
                        for r in records
                        if r["split"] == split
                        and r["view"] == view
                        and r["estimand"] == estimand
                        and r["architecture"] == architecture
                    ]
                    psnr_points = [
                        (float(r["projection_residual"]), float(r["psnr"]))
                        for r in subset
                    ]
                    ssim_points = [
                        (float(r["projection_residual"]), float(r["ssim"]))
                        for r in subset
                    ]
                    pset = set(pareto_front_indices(psnr_points))
                    sset = set(pareto_front_indices(ssim_points))
                    for i, rec in enumerate(subset):
                        rec["pareto_psnr"] = i in pset
                        rec["pareto_ssim"] = i in sset
                    ordered = sorted(
                        subset, key=lambda r: int(r["dc_iteration"])
                    )
                    residuals = np.asarray(
                        [float(r["projection_residual"]) for r in ordered]
                    )
                    violation = bool(np.any(np.diff(residuals) > 0))
                    for rec in subset:
                        rec["residual_nonmonotone_in_k"] = violation

                # Also identify the nondominated geometric points in the
                # union of all three architecture trajectories.  These flags
                # remain a descriptive property of the complete empirical
                # trajectory and do not imply residual-matched superiority.
                union_subset = [
                    r
                    for r in records
                    if r["split"] == split
                    and r["view"] == view
                    and r["estimand"] == estimand
                ]
                for objective in ("psnr", "ssim"):
                    union_points = [
                        (
                            float(r["projection_residual"]),
                            float(r[objective]),
                        )
                        for r in union_subset
                    ]
                    union_front = set(pareto_front_indices(union_points))
                    for index, rec in enumerate(union_subset):
                        rec[f"pareto_union_{objective}"] = index in union_front
    return records


def _subset_points(
    aggregate: Sequence[Mapping[str, object]],
    split: str,
    view: int,
    estimand: str,
    architecture: str,
) -> list[Mapping[str, object]]:
    return sorted(
        [
            r
            for r in aggregate
            if r["split"] == split
            and int(r["view"]) == view
            and r["estimand"] == estimand
            and r["architecture"] == architecture
        ],
        key=lambda r: int(r["dc_iteration"]),
    )


def psnr_frontier_knots(
    points: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    objective = [
        (float(r["projection_residual"]), float(r["psnr"])) for r in points
    ]
    knots = [points[i] for i in pareto_front_indices(objective)]
    return sorted(
        knots,
        key=lambda r: (
            float(r["projection_residual"]),
            float(r["psnr"]),
            int(r["dc_iteration"]),
        ),
    )


def interpolate_psnr_frontier(
    points: Sequence[Mapping[str, object]], target_residual: float
) -> dict[str, object]:
    """Interpolate PSNR and reuse the exact same bracket for SSIM/RMSE."""

    knots = psnr_frontier_knots(points)
    if not knots:
        return {"covered": False, "reason": "empty_frontier"}
    residuals = np.asarray(
        [float(r["projection_residual"]) for r in knots], dtype=np.float64
    )
    if target_residual < residuals[0] or target_residual > residuals[-1]:
        return {
            "covered": False,
            "reason": "target_outside_frontier_support",
            "support_lo": float(residuals[0]),
            "support_hi": float(residuals[-1]),
        }
    exact = np.flatnonzero(residuals == target_residual)
    if len(exact):
        left = right = int(exact[0])
        weight = 0.0
    else:
        right = int(np.searchsorted(residuals, target_residual, side="right"))
        left = right - 1
        denominator = residuals[right] - residuals[left]
        if denominator <= 0:
            raise AnalysisError("non-increasing residual knots after deduplication")
        weight = float((target_residual - residuals[left]) / denominator)
    lo, hi = knots[left], knots[right]

    def blend(metric: str) -> float:
        return float(
            (1.0 - weight) * float(lo[metric]) + weight * float(hi[metric])
        )

    return {
        "covered": True,
        "reason": "",
        "support_lo": float(residuals[0]),
        "support_hi": float(residuals[-1]),
        "bracket_k_lo": int(lo["dc_iteration"]),
        "bracket_k_hi": int(hi["dc_iteration"]),
        "weight_hi": weight,
        "projection_residual": float(target_residual),
        "psnr": blend("psnr"),
        "ssim": blend("ssim"),
        "rmse": blend("rmse"),
    }


def normalized_frontier_area(
    points: Sequence[Mapping[str, object]], support_lo: float, support_hi: float
) -> float | None:
    """Mean PSNR over a fixed residual interval; no extrapolation."""

    if not support_hi > support_lo:
        raise AnalysisError("frontier-area interval must have positive width")
    knots = psnr_frontier_knots(points)
    if not knots:
        return None
    residuals = [float(r["projection_residual"]) for r in knots]
    if support_lo < min(residuals) or support_hi > max(residuals):
        return None
    xs = [support_lo]
    xs.extend(x for x in residuals if support_lo < x < support_hi)
    xs.append(support_hi)
    ys = []
    for x in xs:
        result = interpolate_psnr_frontier(points, x)
        if not result["covered"]:
            return None
        ys.append(float(result["psnr"]))
    x_array = np.asarray(xs, dtype=np.float64)
    y_array = np.asarray(ys, dtype=np.float64)
    # Spell out the trapezoidal rule for compatibility with both NumPy 1.26
    # (ETRI/DIVal) and NumPy >=2.4, where ``np.trapz`` has been removed.
    integral = np.sum(
        0.5 * (y_array[:-1] + y_array[1:]) * np.diff(x_array)
    )
    return float(integral / (support_hi - support_lo))


def validation_targets(
    aggregate: Sequence[Mapping[str, object]], view: int
) -> tuple[float, float, list[dict[str, object]]]:
    """Lock q25/q50/q75 on the log common validation residual range."""

    supports = []
    for architecture in ARCHITECTURES:
        points = _subset_points(
            aggregate, "validation", view, "slice_weighted", architecture
        )
        knots = psnr_frontier_knots(points)
        if not knots:
            raise AnalysisError(
                f"view={view}, architecture={architecture}: empty validation frontier"
            )
        residuals = [float(r["projection_residual"]) for r in knots]
        supports.append((min(residuals), max(residuals)))
    common_lo = max(x[0] for x in supports)
    common_hi = min(x[1] for x in supports)
    if not (common_lo > 0 and common_hi > common_lo):
        raise AnalysisError(
            f"view={view}: no positive common validation residual support; "
            f"supports={supports}"
        )
    targets = []
    for label, quantile in TARGET_QUANTILES:
        residual = math.exp(
            math.log(common_lo)
            + quantile * (math.log(common_hi) - math.log(common_lo))
        )
        targets.append(
            {
                "view": view,
                "target_id": label,
                "quantile_log_common_support": quantile,
                "target_residual": residual,
                "common_validation_support_lo": common_lo,
                "common_validation_support_hi": common_hi,
                "primary": label == PRIMARY_TARGET,
            }
        )
    return common_lo, common_hi, targets


def select_locked_iteration(
    validation_points: Sequence[Mapping[str, object]], target_residual: float
) -> Mapping[str, object] | None:
    """Max validation PSNR subject to mean residual <= target; tie: lower k."""

    eligible = [
        p
        for p in validation_points
        if float(p["projection_residual"]) <= target_residual
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda p: (-float(p["psnr"]), int(p["dc_iteration"])),
    )[0]


def make_bootstrap_weights(
    n_patients: int, n_bootstrap: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_patients, size=(n_bootstrap, n_patients))
    weights = np.zeros((n_bootstrap, n_patients), dtype=np.int16)
    for b in range(n_bootstrap):
        weights[b] = np.bincount(draws[b], minlength=n_patients)
    return weights


def bootstrap_metric_means(
    values: np.ndarray, patient_ids: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sums, counts, patient_means = patient_components(values, patient_ids)
    pooled = (weights @ sums) / (weights @ counts)
    equal_patient = (weights @ patient_means) / weights.shape[1]
    return pooled, equal_patient


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _wilcoxon_patient_means(
    difference: np.ndarray, patient_ids: np.ndarray
) -> tuple[float, float]:
    _, _, patient_means = patient_components(difference, patient_ids)
    if np.all(patient_means == 0):
        return 0.0, 1.0
    result = wilcoxon(patient_means)
    return float(result.statistic), float(result.pvalue)


def fixed_point_contrast_stats(
    difference: np.ndarray,
    patient_ids: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, object]]:
    pooled_boot, equal_boot = bootstrap_metric_means(
        difference, patient_ids, weights
    )
    wilcoxon_stat, wilcoxon_p = _wilcoxon_patient_means(
        difference, patient_ids
    )
    output = []
    for estimand, boot in (
        ("slice_weighted", pooled_boot),
        ("equal_patient", equal_boot),
    ):
        point, median, sd = aggregate_value(difference, patient_ids, estimand)
        lo, hi = _percentile_ci(boot)
        if estimand == "slice_weighted":
            units = difference
        else:
            _, _, units = patient_components(difference, patient_ids)
        output.append(
            {
                "estimand": estimand,
                "mean_diff": point,
                "median_diff": median,
                "sd_diff": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_positive": int(np.sum(units > 0)),
                "n_negative": int(np.sum(units < 0)),
                "n_ties": int(np.sum(units == 0)),
                "proportion_positive": float(np.mean(units > 0)),
                "patient_wilcoxon_stat": wilcoxon_stat,
                "patient_wilcoxon_p": wilcoxon_p,
                "n_bootstrap": int(weights.shape[0]),
                "valid_bootstrap": int(weights.shape[0]),
                "bootstrap_coverage_fraction": 1.0,
                "coverage_ok": True,
            }
        )
    return output


def _with_favoring_direction(
    row: Mapping[str, object], *, favorable_direction: str
) -> dict[str, object]:
    """Add endpoint-aware favoring counts without changing the estimand.

    ``fixed_point_contrast_stats`` always records signed differences and
    positive/negative counts.  This adapter makes the scientific direction
    explicit because lower RMSE/projection residual and higher PSNR/SSIM are
    favorable.  Ties are never assigned to either arm.
    """

    if favorable_direction not in ("positive", "negative"):
        raise AnalysisError(
            f"favorable_direction must be positive or negative, got {favorable_direction!r}"
        )
    result = dict(row)
    if favorable_direction == "positive":
        favoring = int(row["n_positive"])
        opposing = int(row["n_negative"])
    else:
        favoring = int(row["n_negative"])
        opposing = int(row["n_positive"])
    ties = int(row["n_ties"])
    total = favoring + opposing + ties
    if total <= 0:
        raise AnalysisError("favoring summary has no paired units")
    result.update(
        {
            "favorable_direction": favorable_direction,
            "favoring_unit": (
                "slice" if row["estimand"] == "slice_weighted" else "patient"
            ),
            "n_favoring": favoring,
            "n_opposing": opposing,
            "n_ties": ties,
            "proportion_favoring": favoring / total,
            "practical_or_clinical_threshold_prespecified": False,
        }
    )
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values for one explicitly defined family."""

    if not p_values:
        raise AnalysisError("Holm family must not be empty")
    p = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise AnalysisError(f"invalid p-values for Holm adjustment: {p_values}")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    m = len(p)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, value in enumerate(sorted_p):
        candidate = min(1.0, (m - rank) * float(value))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def holm_adjust_two(p_values: Sequence[float]) -> list[float]:
    if len(p_values) != 2:
        raise AnalysisError("Holm family must contain exactly two p-values")
    return holm_adjust(p_values)


def _write_csv(
    path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    empty_fields: Sequence[str] | None = None,
) -> None:
    if not records and not empty_fields:
        raise AnalysisError(
            f"refusing to write an empty table without a frozen schema: {path}"
        )
    fields: list[str] = []
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    if not records:
        fields = list(empty_fields or ())
    if not fields or len(fields) != len(set(fields)):
        raise AnalysisError(f"invalid CSV field schema for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _bootstrap_aggregate_lookup(
    cube: Cube,
    view: int,
    weights: np.ndarray,
    *,
    split: str = "test",
) -> dict[tuple[str, str, int, str], np.ndarray]:
    """Cluster-bootstrap means for both estimands and every trajectory metric."""

    if split not in SPLITS:
        raise AnalysisError(f"unknown split {split!r}")
    pids = cube.patient_ids[(split, view)]
    out: dict[tuple[str, str, int, str], np.ndarray] = {}
    for architecture in ARCHITECTURES:
        for k in cube.iterations[view]:
            for metric in METRICS:
                values = cube.get(split, view, architecture, k, metric)
                pooled, equal = bootstrap_metric_means(values, pids, weights)
                out[("slice_weighted", architecture, k, metric)] = pooled
                out[("equal_patient", architecture, k, metric)] = equal
    return out


def _add_single_arm_cluster_intervals(
    cube: Cube,
    aggregate: Sequence[dict[str, object]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[tuple[str, int], dict[tuple[str, str, int, str], np.ndarray]]:
    """Attach patient-cluster CIs to every aggregate single-arm endpoint.

    The slice-weighted and equal-patient point estimands differ, but both use
    paired resampling of whole patients.  One deterministic bootstrap draw
    matrix is shared across models, iterations, and endpoints within a
    split/view so downstream paired contrasts retain their pairing.
    """

    lookups: dict[
        tuple[str, int], dict[tuple[str, str, int, str], np.ndarray]
    ] = {}
    for split in SPLITS:
        for view in cube.views:
            pids = cube.patient_ids[(split, view)]
            n_patients = len(np.unique(pids))
            weights = make_bootstrap_weights(n_patients, n_bootstrap, seed)
            lookups[(split, view)] = _bootstrap_aggregate_lookup(
                cube, view, weights, split=split
            )

    for record in aggregate:
        split = str(record["split"])
        view = int(record["view"])
        estimand = str(record["estimand"])
        architecture = str(record["architecture"])
        k = int(record["dc_iteration"])
        lookup = lookups[(split, view)]
        for metric in METRICS:
            boot = lookup[(estimand, architecture, k, metric)]
            lo, hi = _percentile_ci(boot)
            record[f"{metric}_ci95_lo"] = lo
            record[f"{metric}_ci95_hi"] = hi
        record["ci_method"] = "patient-cluster percentile bootstrap"
        record["resampling_unit"] = "patient"
        record["n_bootstrap"] = n_bootstrap
        record["bootstrap_coverage_fraction"] = 1.0
        record["coverage_ok"] = True
    return lookups


def _replicate_points(
    lookup: Mapping[tuple[str, str, int, str], np.ndarray],
    estimand: str,
    architecture: str,
    iterations: Sequence[int],
    replicate: int,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for k in iterations:
        point: dict[str, object] = {"dc_iteration": k}
        for metric in METRICS:
            point[metric] = float(
                lookup[(estimand, architecture, k, metric)][replicate]
            )
        points.append(point)
    return points


def _case_values_at_interpolation(
    cube: Cube,
    view: int,
    architecture: str,
    metric: str,
    interpolation: Mapping[str, object],
) -> np.ndarray:
    k_lo = int(interpolation["bracket_k_lo"])
    k_hi = int(interpolation["bracket_k_hi"])
    weight = float(interpolation["weight_hi"])
    lo = cube.get("test", view, architecture, k_lo, metric)
    hi = cube.get("test", view, architecture, k_hi, metric)
    return (1.0 - weight) * lo + weight * hi


def run_analysis(
    cube: Cube,
    *,
    frozen_spec: FrozenAnalysisSpec,
    evidence: Mapping[str, object],
    output_dir: Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    if cube.views != (frozen_spec.view,):
        raise AnalysisError(
            f"formal stats require exactly locked view {frozen_spec.view}; "
            f"cube contains {cube.views}"
        )
    if cube.iterations[frozen_spec.view] != frozen_spec.k_grid:
        raise AnalysisError("cube k grid differs from authenticated lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_points(cube)
    targets = [dict(row) for row in frozen_spec.target_rows]
    target_by_view = {frozen_spec.view: targets}
    bootstrap_lookups = _add_single_arm_cluster_intervals(
        cube, aggregate, n_bootstrap=n_bootstrap, seed=seed
    )

    iteration_matched_points: list[dict[str, object]] = []
    locked_points: list[dict[str, object]] = []
    locked_stats: list[dict[str, object]] = []
    h1_image_prior_stats: list[dict[str, object]] = []
    h2_within_model_dc_stats: list[dict[str, object]] = []
    manuscript_summary: list[dict[str, object]] = []
    descriptive_points: list[dict[str, object]] = []
    descriptive_stats: list[dict[str, object]] = []
    residual_matched_wide: list[dict[str, object]] = []
    area_points: list[dict[str, object]] = []
    area_stats: list[dict[str, object]] = []
    dominance_rows: list[dict[str, object]] = []

    for view in cube.views:
        pids = cube.patient_ids[("test", view)]
        n_patients = len(np.unique(pids))
        weights = make_bootstrap_weights(n_patients, n_bootstrap, seed)
        boot_lookup = bootstrap_lookups[("test", view)]
        point_lookup = {
            (str(r["split"]), str(r["estimand"]), str(r["architecture"]), int(r["dc_iteration"])): r
            for r in aggregate
            if int(r["view"]) == view
        }

        # Iteration-matched points preserve every common computational-budget
        # comparison regardless of whether residual matching is geometrically
        # possible.  Pareto-superiority eligibility is reported separately.
        for record in aggregate:
            if record["split"] != "test" or int(record["view"]) != view:
                continue
            iteration_matched_points.append(
                {
                    "analysis": "iteration_matched_common_finite_dc_budget",
                    **record,
                    "selected_operating_point": (
                        int(record["dc_iteration"]) == frozen_spec.selected_k
                    ),
                    "claim_allowed": evidence.get("authenticated") is True,
                    "pareto_membership_descriptive_only": True,
                    "pareto_superiority_claim_allowed": (
                        frozen_spec.residual_matching_claim_allowed
                    ),
                    "residual_matching_failure_code": (
                        frozen_spec.residual_matching_failure_code
                    ),
                    "failure_reason": (
                        ""
                        if frozen_spec.residual_matching_claim_allowed
                        else frozen_spec.failure_reason
                    ),
                }
            )

        # H1: the learned image-prior contrast is evaluated before any DC.
        # It is independent of the validation DC operating-point and residual-
        # matching gates, but still requires an authenticated formal bundle.
        h1_claim_allowed = evidence.get("authenticated") is True
        h1_failure_reason = (
            "" if h1_claim_allowed else "formal_bundle_was_not_authenticated"
        )
        for baseline in BASELINES:
            for metric in ("psnr", "ssim", "rmse"):
                difference = cube.get(
                    "test", view, "WavTransUNet", 0, metric
                ) - cube.get("test", view, baseline, 0, metric)
                favorable_direction = "negative" if metric == "rmse" else "positive"
                for summary in fixed_point_contrast_stats(
                    difference, pids, weights
                ):
                    directed = _with_favoring_direction(
                        summary, favorable_direction=favorable_direction
                    )
                    h1_image_prior_stats.append(
                        {
                            "hypothesis": "H1_image_prior_advantage_before_DC",
                            "analysis": "paired_fixed_checkpoint_test_k0",
                            "endpoint_role": (
                                "primary" if metric == "psnr" else "secondary"
                            ),
                            "view": view,
                            "candidate": "WavTransUNet",
                            "comparator": baseline,
                            "candidate_dc_iteration": 0,
                            "comparator_dc_iteration": 0,
                            "metric": metric,
                            "metric_definition": METRIC_DEFINITIONS[metric],
                            "difference_unit": METRIC_UNITS[metric],
                            "difference_definition": (
                                f"WavTransUNet(k=0) - {baseline}(k=0)"
                            ),
                            "favors_candidate_when": (
                                "difference < 0"
                                if favorable_direction == "negative"
                                else "difference > 0"
                            ),
                            "zero_difference_interpretation": "tie",
                            **directed,
                            "n_favoring_candidate": directed["n_favoring"],
                            "proportion_favoring_candidate": directed[
                                "proportion_favoring"
                            ],
                            "paired": True,
                            "uncertainty_method": (
                                "paired patient-cluster percentile bootstrap"
                            ),
                            "resampling_unit": "patient",
                            "slice_level_summary_is_descriptive": (
                                directed["estimand"] == "slice_weighted"
                            ),
                            "claim_gate_basis": (
                                "authenticated fixed-checkpoint k=0 comparison; "
                                "independent of DC selection and residual-matching gates"
                            ),
                            "claim_allowed": h1_claim_allowed,
                            "failure_reason": h1_failure_reason,
                            "holm_family": (
                                f"view{view}_H1_k0_psnr_two_comparators"
                                if metric == "psnr"
                                else ""
                            ),
                            "holm_p": "",
                        }
                    )

        # H2: within each architecture, compare the one validation-locked
        # shared finite-DC operating point with that architecture's own k=0.
        # Negative residual/RMSE and positive PSNR/SSIM differences favor DC.
        shared_k = frozen_spec.selected_k
        h2_claim_allowed = frozen_spec.operating_point_claim_allowed
        h2_failure_reason = (
            ""
            if h2_claim_allowed
            else "validation_operating_point_gate_disallowed_claims"
        )
        for architecture in ARCHITECTURES:
            for metric in METRICS:
                difference = cube.get(
                    "test", view, architecture, shared_k, metric
                ) - cube.get("test", view, architecture, 0, metric)
                favorable_direction = (
                    "negative"
                    if metric in ("projection_residual", "rmse")
                    else "positive"
                )
                for summary in fixed_point_contrast_stats(
                    difference, pids, weights
                ):
                    directed = _with_favoring_direction(
                        summary, favorable_direction=favorable_direction
                    )
                    h2_within_model_dc_stats.append(
                        {
                            "hypothesis": "H2_recovery_of_measurement_consistency",
                            "analysis": "paired_within_model_locked_finite_DC_vs_k0",
                            "endpoint_role": (
                                "primary"
                                if metric == "projection_residual"
                                else "secondary"
                            ),
                            "view": view,
                            "architecture": architecture,
                            "finite_dc_iteration": shared_k,
                            "reference_dc_iteration": 0,
                            "selected_lambda_multiplier": frozen_spec.selected_lambda_multiplier,
                            "selected_lambda": frozen_spec.selected_lambda,
                            "selected_eta": frozen_spec.selected_eta,
                            "metric": metric,
                            "metric_definition": METRIC_DEFINITIONS[metric],
                            "difference_unit": METRIC_UNITS[metric],
                            "difference_definition": (
                                f"{architecture}(k={shared_k}) - {architecture}(k=0)"
                            ),
                            "favors_finite_dc_when": (
                                "difference < 0"
                                if favorable_direction == "negative"
                                else "difference > 0"
                            ),
                            "zero_difference_interpretation": "tie",
                            **directed,
                            "n_favoring_finite_dc": directed["n_favoring"],
                            "proportion_favoring_finite_dc": directed[
                                "proportion_favoring"
                            ],
                            "paired": True,
                            "uncertainty_method": (
                                "paired patient-cluster percentile bootstrap"
                            ),
                            "resampling_unit": "patient",
                            "slice_level_summary_is_descriptive": (
                                directed["estimand"] == "slice_weighted"
                            ),
                            "claim_gate_basis": (
                                "authenticated shared validation-selected finite-DC "
                                "operating point; independent of residual-matching interpolation"
                            ),
                            "claim_allowed": h2_claim_allowed,
                            "failure_reason": h2_failure_reason,
                            "holm_family": (
                                f"view{view}_H2_projection_residual_three_models"
                                if metric == "projection_residual"
                                else ""
                            ),
                            "holm_p": "",
                        }
                    )

        # Exact global set dominance is a descriptive geometric property of
        # the complete locked test trajectories and remains computable even
        # without a common residual interval.  The bounded central-region
        # diagnostic is added only when validation supplied q25--q75 targets.
        # Neither geometry substitutes for residual-matched inference.
        if target_by_view[view]:
            central_lo = float(target_by_view[view][0]["target_residual"])
            central_hi = float(target_by_view[view][-1]["target_residual"])
            regions = (
                ("global", -math.inf, math.inf),
                ("validation_central_q25_q75", central_lo, central_hi),
            )
        else:
            central_lo = central_hi = None
            regions = (("global", -math.inf, math.inf),)
        for estimand in ("slice_weighted", "equal_patient"):
            for objective in ("psnr", "ssim"):
                all_points = {
                    architecture: _subset_points(
                        aggregate, "test", view, estimand, architecture
                    )
                    for architecture in ARCHITECTURES
                }
                for baseline in BASELINES:
                    for region, lo, hi in regions:
                        wp = [
                            (
                                float(r["projection_residual"]),
                                float(r[objective]),
                            )
                            for r in all_points["WavTransUNet"]
                            if lo <= float(r["projection_residual"]) <= hi
                        ]
                        bp = [
                            (
                                float(r["projection_residual"]),
                                float(r[objective]),
                            )
                            for r in all_points[baseline]
                            if lo <= float(r["projection_residual"]) <= hi
                        ]
                        dominance_rows.append(
                            {
                                "view": view,
                                "estimand": estimand,
                                "objective": objective,
                                "region": region,
                                "candidate": "WavTransUNet",
                                "baseline": baseline,
                                "candidate_points": len(wp),
                                "baseline_points": len(bp),
                                "candidate_frontier_points": len(
                                    pareto_front_indices(wp)
                                ),
                                "baseline_frontier_points": len(
                                    pareto_front_indices(bp)
                                ),
                                "candidate_strictly_set_dominates": (
                                    set_strictly_dominates(wp, bp)
                                ),
                                "candidate_weakly_covers": weakly_covers_set(
                                    wp, bp
                                ),
                                "baseline_strictly_set_dominates": (
                                    set_strictly_dominates(bp, wp)
                                ),
                                "coverage_ok": bool(wp and bp),
                                "aggregate_mean_frontier_descriptive_only": True,
                                "global_geometry_available": region == "global",
                                "residual_matched_superiority_inference_eligible": (
                                    False
                                ),
                                "claim_allowed": frozen_spec.claim_allowed,
                                "failure_code": (
                                    ""
                                    if frozen_spec.residual_matching_claim_allowed
                                    else frozen_spec.residual_matching_failure_code
                                ),
                                "failure_reason": frozen_spec.failure_reason,
                            }
                        )

        # The sole deployable operating point is the shared validation-selected
        # k from the authenticated lock.  Residual targets below define
        # aggregate curve functionals and never trigger a second model-specific
        # selection or create a deployable fractional-k reconstruction.
        for architecture in ARCHITECTURES:
            for estimand in ("slice_weighted", "equal_patient"):
                validation_point = point_lookup[
                    ("validation", estimand, architecture, shared_k)
                ]
                test_point = point_lookup[("test", estimand, architecture, shared_k)]
                locked_points.append(
                    {
                        "analysis": "validation_locked_shared_iteration",
                        "view": view,
                        "architecture": architecture,
                        "estimand": estimand,
                        "selected_dc_iteration": shared_k,
                        "selected_lambda_multiplier": frozen_spec.selected_lambda_multiplier,
                        "selected_lambda": frozen_spec.selected_lambda,
                        "selected_eta": frozen_spec.selected_eta,
                        "selection_rule": "shared_validation_minimax_operating_point_from_dc_lock",
                        "validation_projection_residual": float(
                            validation_point["projection_residual"]
                        ),
                        "validation_psnr": float(validation_point["psnr"]),
                        "test_projection_residual": float(
                            test_point["projection_residual"]
                        ),
                        "test_psnr": float(test_point["psnr"]),
                        "test_ssim": float(test_point["ssim"]),
                        "test_rmse": float(test_point["rmse"]),
                        "claim_allowed": frozen_spec.operating_point_claim_allowed,
                        "failure_reason": (
                            ""
                            if frozen_spec.operating_point_claim_allowed
                            else "validation_operating_point_gate_disallowed_claims"
                        ),
                    }
                )

        # Six manuscript-ready method rows per view: k=0 and the one common
        # validation-selected finite-DC k for each architecture.  Both
        # estimands and their patient-cluster intervals are retained in wide
        # columns so the six-row layout never hides the patient sensitivity.
        for architecture in ARCHITECTURES:
            for stage, k in (("without_DC", 0), ("finite_DC", shared_k)):
                summary_row: dict[str, object] = {
                    "view": view,
                    "method": (
                        architecture
                        if stage == "without_DC"
                        else f"{architecture} + finite DC"
                    ),
                    "architecture": architecture,
                    "stage": stage,
                    "dc_iteration": k,
                    "selected_lambda_multiplier": (
                        frozen_spec.selected_lambda_multiplier
                        if stage == "finite_DC"
                        else 0.0
                    ),
                    "selected_lambda": (
                        frozen_spec.selected_lambda if stage == "finite_DC" else 0.0
                    ),
                    "selected_eta": (
                        frozen_spec.selected_eta if stage == "finite_DC" else 0.0
                    ),
                    "primary_estimand": "slice_weighted",
                    "sensitivity_estimand": "equal_patient",
                    "uncertainty_method": (
                        "patient-cluster percentile bootstrap"
                    ),
                    "n_bootstrap": n_bootstrap,
                    "claim_allowed": (
                        evidence.get("authenticated") is True
                        if stage == "without_DC"
                        else frozen_spec.operating_point_claim_allowed
                    ),
                    "failure_reason": (
                        ""
                        if stage == "without_DC"
                        or frozen_spec.operating_point_claim_allowed
                        else "validation_operating_point_gate_disallowed_claims"
                    ),
                }
                for estimand in ("slice_weighted", "equal_patient"):
                    point = point_lookup[("test", estimand, architecture, k)]
                    for metric in METRICS:
                        prefix = f"{estimand}_{metric}"
                        summary_row[f"{prefix}_mean"] = point[metric]
                        summary_row[f"{prefix}_median"] = point[
                            f"{metric}_median"
                        ]
                        summary_row[f"{prefix}_sample_sd"] = point[
                            f"{metric}_sd"
                        ]
                        summary_row[f"{prefix}_ci95_lo"] = point[
                            f"{metric}_ci95_lo"
                        ]
                        summary_row[f"{prefix}_ci95_hi"] = point[
                            f"{metric}_ci95_hi"
                        ]
                manuscript_summary.append(summary_row)

        for baseline in BASELINES:
            for metric in METRICS:
                diff = cube.get(
                    "test", view, "WavTransUNet", shared_k, metric
                ) - cube.get("test", view, baseline, shared_k, metric)
                for stats in fixed_point_contrast_stats(diff, pids, weights):
                    locked_stats.append(
                        {
                            "analysis": "validation_locked_shared_iteration",
                            "view": view,
                            "candidate": "WavTransUNet",
                            "baseline": baseline,
                            "candidate_dc_iteration": shared_k,
                            "baseline_dc_iteration": shared_k,
                            "metric": metric,
                            **stats,
                            "claim_allowed": frozen_spec.operating_point_claim_allowed,
                            "failure_reason": (
                                ""
                                if frozen_spec.operating_point_claim_allowed
                                else "validation_operating_point_gate_disallowed_claims"
                            ),
                            "holm_family": (
                                f"view{view}_locked_shared_k_psnr_two_baselines"
                                if metric == "psnr"
                                else ""
                            ),
                            "holm_p": "",
                        }
                    )

        # Aggregate curve-functional interpolation at validation-locked
        # residual targets.  This is inferential only when the explicit
        # residual-matching and bootstrap-coverage gates pass; it is never an
        # observed fractional-k or deployable reconstruction.
        for target in target_by_view[view]:
            target_id = str(target["target_id"])
            target_r = float(target["target_residual"])
            point_interpolations: dict[
                tuple[str, str], dict[str, object]
            ] = {}
            point_records: dict[tuple[str, str], dict[str, object]] = {}
            bootstrap_interpolations: dict[
                tuple[str, str, str], np.ndarray
            ] = {}
            for estimand in ("slice_weighted", "equal_patient"):
                for architecture in ARCHITECTURES:
                    test_points = _subset_points(
                        aggregate, "test", view, estimand, architecture
                    )
                    interp = interpolate_psnr_frontier(test_points, target_r)
                    point_interpolations[(estimand, architecture)] = interp
                    record = {
                        "analysis": "validation_locked_test_curve_functional",
                        "view": view,
                        "target_id": target_id,
                        "primary": target_id == PRIMARY_TARGET,
                        "target_residual": target_r,
                        "estimand": estimand,
                        "architecture": architecture,
                        "claim_allowed": frozen_spec.claim_allowed,
                        "interpolated_curve_functional": True,
                        "deployable_reconstruction": False,
                        "inferential_curve_eligible": False,
                        "superiority_inference_eligible": False,
                        "failure_reason": frozen_spec.failure_reason,
                        **interp,
                    }
                    point_records[(estimand, architecture)] = record
                    descriptive_points.append(record)
                    for metric in ("psnr", "ssim", "rmse"):
                        bootstrap_interpolations[(estimand, architecture, metric)] = np.full(
                            n_bootstrap, np.nan, dtype=np.float64
                        )

                for b in range(n_bootstrap):
                    for architecture in ARCHITECTURES:
                        rep_points = _replicate_points(
                            boot_lookup,
                            estimand,
                            architecture,
                            cube.iterations[view],
                            b,
                        )
                        rep_interp = interpolate_psnr_frontier(
                            rep_points, target_r
                        )
                        if rep_interp["covered"]:
                            for metric in ("psnr", "ssim", "rmse"):
                                bootstrap_interpolations[
                                    (estimand, architecture, metric)
                                ][b] = float(rep_interp[metric])

                # Single-arm uncertainty at the matched residual.  The same
                # PSNR-frontier brackets/weights produced all three endpoints.
                wide_row: dict[str, object] = {
                    "view": view,
                    "target_id": target_id,
                    "primary": target_id == PRIMARY_TARGET,
                    "target_residual": target_r,
                    "estimand": estimand,
                    "inference_eligibility": (
                        "INFERENTIAL_CURVE_FUNCTIONAL"
                        if frozen_spec.residual_matching_claim_allowed
                        else "NOT_AVAILABLE"
                    ),
                    "residual_matching_claim_allowed": (
                        frozen_spec.residual_matching_claim_allowed
                    ),
                    "superiority_inference_eligible": (
                        frozen_spec.residual_matching_claim_allowed
                    ),
                    "inferential_curve_eligible": (
                        frozen_spec.residual_matching_claim_allowed
                    ),
                    "failure_code": frozen_spec.residual_matching_failure_code,
                    "failure_reason": frozen_spec.failure_reason,
                    "uncertainty_method": (
                        "patient-cluster percentile bootstrap"
                    ),
                    "n_bootstrap": n_bootstrap,
                }
                for architecture in ARCHITECTURES:
                    slug = MODEL_SLUGS[architecture]
                    point_record = point_records[(estimand, architecture)]
                    for metric in ("psnr", "ssim", "rmse"):
                        boot = bootstrap_interpolations[
                            (estimand, architecture, metric)
                        ]
                        valid = np.isfinite(boot)
                        coverage_fraction = float(np.mean(valid))
                        coverage_ok = bool(
                            point_record.get("covered", False)
                            and coverage_fraction
                            >= MIN_VALID_BOOTSTRAP_FRACTION
                        )
                        lo, hi = _percentile_ci(boot)
                        point_record[f"{metric}_ci95_lo"] = (
                            lo if coverage_ok else ""
                        )
                        point_record[f"{metric}_ci95_hi"] = (
                            hi if coverage_ok else ""
                        )
                        point_record[f"{metric}_valid_bootstrap"] = int(
                            np.sum(valid)
                        )
                        point_record[
                            f"{metric}_bootstrap_coverage_fraction"
                        ] = coverage_fraction
                        point_record[f"{metric}_coverage_ok"] = coverage_ok
                        wide_row[f"{slug}_{metric}"] = (
                            point_record.get(metric, "") if coverage_ok else ""
                        )
                        wide_row[f"{slug}_{metric}_ci95_lo"] = (
                            lo if coverage_ok else ""
                        )
                        wide_row[f"{slug}_{metric}_ci95_hi"] = (
                            hi if coverage_ok else ""
                        )
                        wide_row[f"{slug}_{metric}_coverage_ok"] = coverage_ok
                    point_record["inferential_curve_eligible"] = bool(
                        frozen_spec.residual_matching_claim_allowed
                        and point_record.get("covered", False)
                        and point_record.get("psnr_coverage_ok", False)
                    )
                    point_record["superiority_inference_eligible"] = bool(
                        point_record["inferential_curve_eligible"]
                    )
                wide_row["inferential_curve_eligible"] = bool(
                    frozen_spec.residual_matching_claim_allowed
                    and all(
                        bool(wide_row[f"{slug}_psnr_coverage_ok"])
                        for slug in MODEL_SLUGS.values()
                    )
                )
                wide_row["superiority_inference_eligible"] = bool(
                    wide_row["inferential_curve_eligible"]
                )
                residual_matched_wide.append(wide_row)

                for baseline in BASELINES:
                    for metric in ("psnr", "ssim", "rmse"):
                        w_boot = bootstrap_interpolations[
                            (estimand, "WavTransUNet", metric)
                        ]
                        b_boot = bootstrap_interpolations[
                            (estimand, baseline, metric)
                        ]
                        delta_boot = w_boot - b_boot
                        valid = np.isfinite(delta_boot)
                        coverage_fraction = float(np.mean(valid))
                        coverage_ok = (
                            coverage_fraction >= MIN_VALID_BOOTSTRAP_FRACTION
                        )
                        w_interp = point_interpolations[
                            (estimand, "WavTransUNet")
                        ]
                        b_interp = point_interpolations[(estimand, baseline)]
                        point_covered = bool(
                            w_interp.get("covered", False)
                            and b_interp.get("covered", False)
                        )
                        if point_covered:
                            w_cases = _case_values_at_interpolation(
                                cube,
                                view,
                                "WavTransUNet",
                                metric,
                                w_interp,
                            )
                            b_cases = _case_values_at_interpolation(
                                cube, view, baseline, metric, b_interp
                            )
                            diff = w_cases - b_cases
                            mean_diff, median_diff, sd_diff = aggregate_value(
                                diff, pids, estimand
                            )
                            wstat, pvalue = _wilcoxon_patient_means(diff, pids)
                        else:
                            mean_diff = median_diff = sd_diff = float("nan")
                            wstat = pvalue = float("nan")
                        lo, hi = _percentile_ci(delta_boot)
                        descriptive_stats.append(
                            {
                                "analysis": "validation_locked_test_curve_functional",
                                "view": view,
                                "target_id": target_id,
                                "primary": target_id == PRIMARY_TARGET,
                                "target_residual": target_r,
                                "estimand": estimand,
                                "candidate": "WavTransUNet",
                                "baseline": baseline,
                                "metric": metric,
                                "mean_diff": mean_diff,
                                "median_diff": median_diff,
                                "sd_diff": sd_diff,
                                "ci95_lo": lo if coverage_ok else "",
                                "ci95_hi": hi if coverage_ok else "",
                                "patient_wilcoxon_stat": wstat,
                                "patient_wilcoxon_p": pvalue,
                                "n_bootstrap": n_bootstrap,
                                "valid_bootstrap": int(np.sum(valid)),
                                "bootstrap_coverage_fraction": coverage_fraction,
                                "coverage_ok": bool(coverage_ok and point_covered),
                                "claim_allowed": (
                                    frozen_spec.residual_matching_claim_allowed
                                    and bool(coverage_ok and point_covered)
                                ),
                                "inference_eligibility": (
                                    "INFERENTIAL_CURVE_FUNCTIONAL"
                                    if frozen_spec.residual_matching_claim_allowed
                                    and bool(coverage_ok and point_covered)
                                    else "NOT_AVAILABLE"
                                ),
                                "superiority_inference_eligible": (
                                    frozen_spec.residual_matching_claim_allowed
                                    and bool(coverage_ok and point_covered)
                                ),
                                "inferential_curve_eligible": (
                                    frozen_spec.residual_matching_claim_allowed
                                    and bool(coverage_ok and point_covered)
                                ),
                                "deployable_reconstruction": False,
                                "estimand_interpretation": (
                                    "validation-locked aggregate frontier functional; "
                                    "not an observed fractional-iteration reconstruction"
                                ),
                                "failure_reason": frozen_spec.failure_reason,
                                "holm_family": (
                                    f"view{view}_descriptive_q50_psnr_two_baselines"
                                    if target_id == PRIMARY_TARGET
                                    and metric == "psnr"
                                    else ""
                                ),
                                "holm_p": "",
                            }
                        )

        if not target_by_view[view]:
            # A structured negative validation result must not manufacture a
            # central interval, residual targets, or a frontier-area domain.
            continue

        # Normalized PSNR-frontier area over validation-defined q25--q75.
        for estimand in ("slice_weighted", "equal_patient"):
            point_areas: dict[str, float | None] = {}
            bootstrap_areas: dict[str, np.ndarray] = {}
            for architecture in ARCHITECTURES:
                points = _subset_points(
                    aggregate, "test", view, estimand, architecture
                )
                area = normalized_frontier_area(points, central_lo, central_hi)
                point_areas[architecture] = area
                area_points.append(
                    {
                        "view": view,
                        "estimand": estimand,
                        "architecture": architecture,
                        "support_lo_q25": central_lo,
                        "support_hi_q75": central_hi,
                        "normalized_psnr_frontier_area_db": (
                            area if area is not None else ""
                        ),
                        "coverage_ok": area is not None,
                        "claim_allowed": frozen_spec.claim_allowed,
                        "failure_reason": frozen_spec.failure_reason,
                    }
                )
                reps = np.full(n_bootstrap, np.nan, dtype=np.float64)
                for b in range(n_bootstrap):
                    rep_points = _replicate_points(
                        boot_lookup,
                        estimand,
                        architecture,
                        cube.iterations[view],
                        b,
                    )
                    rep_area = normalized_frontier_area(
                        rep_points, central_lo, central_hi
                    )
                    if rep_area is not None:
                        reps[b] = rep_area
                bootstrap_areas[architecture] = reps
            for baseline in BASELINES:
                delta = (
                    bootstrap_areas["WavTransUNet"]
                    - bootstrap_areas[baseline]
                )
                valid = np.isfinite(delta)
                coverage_fraction = float(np.mean(valid))
                coverage_ok = (
                    coverage_fraction >= MIN_VALID_BOOTSTRAP_FRACTION
                    and point_areas["WavTransUNet"] is not None
                    and point_areas[baseline] is not None
                )
                lo, hi = _percentile_ci(delta)
                point_diff = (
                    float(point_areas["WavTransUNet"])
                    - float(point_areas[baseline])
                    if point_areas["WavTransUNet"] is not None
                    and point_areas[baseline] is not None
                    else float("nan")
                )
                area_stats.append(
                    {
                        "view": view,
                        "estimand": estimand,
                        "candidate": "WavTransUNet",
                        "baseline": baseline,
                        "support_lo_q25": central_lo,
                        "support_hi_q75": central_hi,
                        "normalized_area_diff_db": point_diff,
                        "ci95_lo": lo if coverage_ok else "",
                        "ci95_hi": hi if coverage_ok else "",
                        "n_bootstrap": n_bootstrap,
                        "valid_bootstrap": int(np.sum(valid)),
                        "bootstrap_coverage_fraction": coverage_fraction,
                        "coverage_ok": coverage_ok,
                        "claim_allowed": frozen_spec.claim_allowed,
                        "failure_reason": frozen_spec.failure_reason,
                    }
                )

    # Two-baseline Holm families are separate for the deployable shared-k
    # comparison and the descriptive primary (q50) residual match.
    for view in cube.views:
        for estimand in ("slice_weighted", "equal_patient"):
            h1_family = [
                row
                for row in h1_image_prior_stats
                if int(row["view"]) == view
                and row["estimand"] == estimand
                and row["metric"] == "psnr"
            ]
            if len(h1_family) != 2:
                raise AnalysisError(
                    f"Holm H1 PSNR family expected 2 rows, found "
                    f"{len(h1_family)} for view={view}, estimand={estimand}"
                )
            for row, adjusted_p in zip(
                h1_family,
                holm_adjust(
                    [float(item["patient_wilcoxon_p"]) for item in h1_family]
                ),
            ):
                row["holm_p"] = adjusted_p

            h2_family = [
                row
                for row in h2_within_model_dc_stats
                if int(row["view"]) == view
                and row["estimand"] == estimand
                and row["metric"] == "projection_residual"
            ]
            if len(h2_family) != len(ARCHITECTURES):
                raise AnalysisError(
                    f"Holm H2 projection-residual family expected "
                    f"{len(ARCHITECTURES)} rows, found {len(h2_family)} for "
                    f"view={view}, estimand={estimand}"
                )
            for row, adjusted_p in zip(
                h2_family,
                holm_adjust(
                    [float(item["patient_wilcoxon_p"]) for item in h2_family]
                ),
            ):
                row["holm_p"] = adjusted_p

            locked_family = [
                r
                for r in locked_stats
                if int(r["view"]) == view
                and r["estimand"] == estimand
                and r["metric"] == "psnr"
            ]
            descriptive_family = [
                r
                for r in descriptive_stats
                if int(r["view"]) == view
                and r["estimand"] == estimand
                and r["target_id"] == PRIMARY_TARGET
                and r["metric"] == "psnr"
            ]
            families = [("locked shared-k", locked_family)]
            if target_by_view[view]:
                families.append(("q50 curve functional", descriptive_family))
            elif descriptive_family:
                raise AnalysisError(
                    "q50 curve-functional rows exist without authenticated residual targets"
                )
            for label, family in families:
                if len(family) != 2:
                    raise AnalysisError(
                        f"Holm {label} PSNR family expected 2 rows, "
                        f"found {len(family)} for view={view}, estimand={estimand}"
                    )
                adjusted = holm_adjust_two(
                    [float(r["patient_wilcoxon_p"]) for r in family]
                )
                for row, adjusted_p in zip(family, adjusted):
                    row["holm_p"] = adjusted_p

    for record in aggregate:
        record["claim_allowed"] = evidence.get("authenticated") is True
        record["pareto_membership_descriptive_only"] = True
        record["pareto_superiority_claim_allowed"] = (
            frozen_spec.residual_matching_claim_allowed
        )
        record["residual_matching_failure_code"] = (
            frozen_spec.residual_matching_failure_code
        )
        record["failure_reason"] = (
            ""
            if frozen_spec.residual_matching_claim_allowed
            else frozen_spec.failure_reason
        )

    paths = {
        "aggregate": output_dir / "aggregate_operating_points.csv",
        "iteration_matched": output_dir / "iteration_matched_operating_points.csv",
        "targets": output_dir / "validation_locked_residual_targets.csv",
        "dominance": output_dir / "empirical_set_dominance.csv",
        "locked_points": output_dir / "locked_operating_points.csv",
        "locked_stats": output_dir / "locked_pairwise_stats.csv",
        "h1_image_prior_stats": output_dir / "h1_image_prior_pairwise_stats.csv",
        "h2_within_model_dc_stats": output_dir
        / "h2_within_model_finite_dc_stats.csv",
        "manuscript_summary": output_dir
        / "manuscript_six_method_summary.csv",
        "descriptive_points": output_dir
        / "descriptive_residual_matched_points.csv",
        "descriptive_stats": output_dir
        / "descriptive_residual_matched_stats.csv",
        "residual_matched_wide": output_dir
        / "manuscript_residual_matched_wide.csv",
        "area_points": output_dir / "normalized_frontier_area.csv",
        "area_stats": output_dir / "normalized_frontier_area_contrasts.csv",
        "meta": output_dir / "dc_pareto_stats_meta.json",
    }
    _write_csv(paths["aggregate"], aggregate)
    _write_csv(paths["iteration_matched"], iteration_matched_points)
    _write_csv(
        paths["targets"], targets, empty_fields=EMPTY_OUTPUT_FIELDS["targets"]
    )
    _write_csv(
        paths["dominance"],
        dominance_rows,
        empty_fields=EMPTY_OUTPUT_FIELDS["dominance"],
    )
    _write_csv(paths["locked_points"], locked_points)
    _write_csv(paths["locked_stats"], locked_stats)
    _write_csv(paths["h1_image_prior_stats"], h1_image_prior_stats)
    _write_csv(paths["h2_within_model_dc_stats"], h2_within_model_dc_stats)
    _write_csv(paths["manuscript_summary"], manuscript_summary)
    _write_csv(
        paths["descriptive_points"],
        descriptive_points,
        empty_fields=EMPTY_OUTPUT_FIELDS["descriptive_points"],
    )
    _write_csv(
        paths["descriptive_stats"],
        descriptive_stats,
        empty_fields=EMPTY_OUTPUT_FIELDS["descriptive_stats"],
    )
    _write_csv(
        paths["residual_matched_wide"],
        residual_matched_wide,
        empty_fields=EMPTY_OUTPUT_FIELDS["residual_matched_wide"],
    )
    _write_csv(
        paths["area_points"],
        area_points,
        empty_fields=EMPTY_OUTPUT_FIELDS["area_points"],
    )
    _write_csv(
        paths["area_stats"],
        area_stats,
        empty_fields=EMPTY_OUTPUT_FIELDS["area_stats"],
    )

    table_records: dict[str, Sequence[Mapping[str, object]]] = {
        "aggregate": aggregate,
        "iteration_matched": iteration_matched_points,
        "targets": targets,
        "dominance": dominance_rows,
        "locked_points": locked_points,
        "locked_stats": locked_stats,
        "h1_image_prior_stats": h1_image_prior_stats,
        "h2_within_model_dc_stats": h2_within_model_dc_stats,
        "manuscript_summary": manuscript_summary,
        "descriptive_points": descriptive_points,
        "descriptive_stats": descriptive_stats,
        "residual_matched_wide": residual_matched_wide,
        "area_points": area_points,
        "area_stats": area_stats,
    }
    residual_dependent_labels = {
        "targets",
        "descriptive_points",
        "descriptive_stats",
        "residual_matched_wide",
        "area_points",
        "area_stats",
    }
    output_manifest: dict[str, dict[str, object]] = {}
    for label, records in table_records.items():
        residual_dependent = label in residual_dependent_labels
        available = not residual_dependent or bool(targets)
        if label == "h1_image_prior_stats":
            output_claim_allowed = evidence.get("authenticated") is True
        elif label == "dominance":
            # The table is available as exact descriptive geometry even when
            # no common residual interval exists, but that negative gate
            # still prohibits a matched-residual superiority claim.
            output_claim_allowed = frozen_spec.claim_allowed
        elif label in {
            "locked_points",
            "locked_stats",
            "h2_within_model_dc_stats",
            "manuscript_summary",
        }:
            output_claim_allowed = frozen_spec.operating_point_claim_allowed
        elif residual_dependent:
            output_claim_allowed = frozen_spec.residual_matching_claim_allowed
        else:
            output_claim_allowed = evidence.get("authenticated") is True
        output_manifest[label] = {
            "path": str(paths[label]),
            "sha256": sha256(paths[label]),
            "rows": len(records),
            "status": "AVAILABLE" if available else "NOT_AVAILABLE",
            "claim_allowed": bool(output_claim_allowed and available),
            "failure_code": (
                frozen_spec.residual_matching_failure_code
                if (
                    (residual_dependent and not available)
                    or (
                        label == "dominance"
                        and not frozen_spec.residual_matching_claim_allowed
                    )
                )
                else ""
            ),
            "failure_reason": (
                frozen_spec.failure_reason
                if (
                    (residual_dependent and not available)
                    or (
                        label == "dominance"
                        and not frozen_spec.residual_matching_claim_allowed
                    )
                )
                else ""
            ),
        }

    meta = {
        "schema": "dc.pareto_stats_manifest.v1",
        "status": "COMPLETE",
        "evidence": dict(evidence),
        "outputs": output_manifest,
        "accepted_formal_input_schema": list(RAW_CASE_COLUMNS),
        "architectures": list(ARCHITECTURES),
        "views": list(cube.views),
        "trajectories": cube.trajectories,
        "iteration_grids": {str(k): list(v) for k, v in cube.iterations.items()},
        "target_rule": (
            "authenticated q25/q50/q75 from dc_lock; independent all-k "
            "selected-lambda validation recomputation passed"
            if targets
            else "no targets: authenticated validation lock reported "
            "NO_COMMON_INTERVAL and independent support recomputation agreed"
        ),
        "prespecified_primary_target_id": PRIMARY_TARGET,
        "primary_target": PRIMARY_TARGET if targets else None,
        "locked_selection_rule": "one shared validation-selected lambda/eta/k from authenticated dc_lock",
        "descriptive_interpolation": (
            "inferential validation-locked aggregate curve functional: linear "
            "between test aggregate PSNR Pareto knots, paired patient-cluster "
            "bootstrap uncertainty, no extrapolation, and the same brackets/"
            "weights for SSIM and RMSE; it is not a deployable or observed "
            "fractional-iteration reconstruction"
            if targets
            else "NOT_AVAILABLE: validation had no positive-width common "
            "residual interval; no targets or values were constructed"
        ),
        "analysis_availability": {
            "H1_image_prior": True,
            "H2_within_model_finite_DC": True,
            "iteration_matched": True,
            "shared_validation_selected_k": True,
            "residual_matched": bool(targets),
            "inferential_curve_eligible": bool(
                targets and frozen_spec.residual_matching_claim_allowed
            ),
            "empirical_global_pareto_geometry": True,
            "central_region_pareto_geometry": bool(targets),
            "pareto_superiority": bool(targets),
            "normalized_frontier_area": bool(targets),
            "residual_matching_failure_code": (
                frozen_spec.residual_matching_failure_code
            ),
            "failure_reason": (
                "" if targets else frozen_spec.failure_reason
            ),
        },
        "hypothesis_inference": {
            "H1": {
                "table": "h1_image_prior_stats",
                "contrast": "WavTransUNet(k=0) minus each baseline(k=0)",
                "primary_endpoint": "PSNR",
                "secondary_endpoints": ["SSIM", "RMSE"],
                "claim_gate": "authenticated fixed-checkpoint test comparison; independent of DC gates",
                "multiplicity": "Holm across two primary PSNR comparator contrasts, separately by estimand",
            },
            "H2": {
                "table": "h2_within_model_dc_stats",
                "contrast": "within architecture, selected finite-DC k minus k=0",
                "primary_endpoint": "projection_residual",
                "secondary_endpoints": ["PSNR", "SSIM", "RMSE"],
                "claim_gate": "authenticated validation-selected operating-point gate; independent of residual interpolation gate",
                "multiplicity": "Holm across three primary projection-residual model contrasts, separately by estimand",
            },
            "H3": {
                "table": "descriptive_stats",
                "contrast": "WavTransUNet minus each baseline at the validation-locked q50 aggregate residual target",
                "primary_endpoint": "PSNR",
                "secondary_endpoints": ["SSIM", "RMSE"],
                "estimand": "validation-locked aggregate frontier functional",
                "not_deployable": True,
                "claim_gate": "authenticated residual target, no extrapolation, at least 95% paired patient-bootstrap coverage, and Holm-adjusted two-baseline q50 PSNR family; cannot alone establish framework superiority",
                "multiplicity": "Holm across two q50 PSNR comparator contrasts, separately by estimand",
            },
            "uncertainty": "paired patient-cluster percentile bootstrap for slice-weighted and equal-patient means",
            "favoring": "reported at the estimand unit: slices for slice-weighted rows and patients for equal-patient rows",
            "practical_or_clinical_threshold": "none prespecified; no clinical interpretation is inferred",
        },
        "locked_spec": {
            "view": frozen_spec.view,
            "trajectory_id": frozen_spec.trajectory_id,
            "selected_lambda_multiplier": frozen_spec.selected_lambda_multiplier,
            "selected_lambda": frozen_spec.selected_lambda,
            "selected_eta": frozen_spec.selected_eta,
            "selected_k": frozen_spec.selected_k,
            "k_grid": list(frozen_spec.k_grid),
            "common_support": (
                [
                    frozen_spec.common_support_lo,
                    frozen_spec.common_support_hi,
                ]
                if targets
                else None
            ),
            "targets": {
                str(row["target_id"]): float(row["target_residual"])
                for row in frozen_spec.target_rows
            },
            "claim_allowed": frozen_spec.claim_allowed,
            "operating_point_claim_allowed": frozen_spec.operating_point_claim_allowed,
            "residual_matching_claim_allowed": frozen_spec.residual_matching_claim_allowed,
            "residual_matching_failure_code": (
                frozen_spec.residual_matching_failure_code
            ),
            "failure_reason": frozen_spec.failure_reason,
            "checkpoint_sha256": frozen_spec.checkpoint_sha256,
            "lock_payload_sha256": frozen_spec.lock_payload_sha256,
            "protocol_sha256": frozen_spec.protocol_sha256,
        },
        "primary_estimand": "slice-weighted mean",
        "co_primary_sensitivity": "equal-patient mean",
        "inference": "paired patient-cluster percentile bootstrap",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "minimum_valid_bootstrap_fraction": MIN_VALID_BOOTSTRAP_FRACTION,
        "holm_families": [
            "H1: two k=0 primary PSNR contrasts",
            "H2: three within-model primary projection-residual contrasts",
            "locked shared-k: two primary PSNR contrasts",
            *(
                ["q50 curve functional: two primary PSNR contrasts"]
                if targets
                else []
            ),
        ],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "python": platform.python_version(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, type=int, choices=(125, 50))
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--sweep-manifest", required=True, type=Path)
    parser.add_argument("--dc-lock", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--validation-patient-map", required=True, type=Path)
    parser.add_argument("--test-patient-map", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir.is_symlink():
        raise AnalysisError("--out-dir may not be a symlink")
    if args.out_dir.exists() and (
        not args.out_dir.is_dir() or any(args.out_dir.iterdir())
    ):
        raise AnalysisError("--out-dir must be new or empty")
    bundle = load_authenticated_view_bundle(
        view=args.view,
        validation_csv=args.validation_csv,
        sweep_manifest_path=args.sweep_manifest,
        lock_path=args.dc_lock,
        test_csv=args.test_csv,
        test_manifest_path=args.test_manifest,
        validation_patient_map_path=args.validation_patient_map,
        test_patient_map_path=args.test_patient_map,
        protocol_path=args.protocol,
    )
    protocol = load_json(args.protocol)
    statistics = _require_mapping(protocol.get("statistics"), "protocol.statistics")
    n_bootstrap = int(statistics["bootstrap_replicates"])
    seed = int(statistics["seed"])
    paths = run_analysis(
        bundle.cube,
        frozen_spec=bundle.spec,
        evidence=bundle.evidence,
        output_dir=args.out_dir,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
