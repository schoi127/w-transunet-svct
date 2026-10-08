#!/usr/bin/env python3
"""Authenticated cross-view claim decision for the finite-DC study.

This CPU-only program consumes the two independently completed per-view
statistics bundles (125 and 50 views).  It authenticates the protocol,
execution package, run manifests, and every declared statistics output before
reading scientific results.  It then applies one conservative, fixed decision
rule across both view conditions.

The residual-matched q50 quantity is an inferential *curve functional* only
when the statistics bundle opts in through the explicit
``inferential_curve_eligible`` gate and at least 95% of bootstrap replicates
cover the target.  It is never treated as an observed fractional-iteration or
deployable reconstruction.  The validation-selected shared integer-k result
is evaluated separately as the deployable operating point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


VIEWS = (125, 50)
ARCHITECTURES = ("UNet", "TransUNet", "WavTransUNet")
BASELINES = ("UNet", "TransUNet")
META_SCHEMA = "dc.pareto_stats_manifest.v1"
DECISION_SCHEMA = "dc.cross_view_claim_decision.v1"
ALPHA = 0.05
MIN_CURVE_BOOTSTRAP_COVERAGE = 0.95
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# This is the exact execution/analysis package authenticated by the runner and
# by dc_pareto_stats.py.  The decision script is deliberately downstream and
# is not part of the GPU execution package, avoiding a circular hash.
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

REQUIRED_OUTPUTS = {
    "aggregate",
    "iteration_matched",
    "targets",
    "dominance",
    "locked_points",
    "locked_stats",
    "h1_image_prior_stats",
    "h2_within_model_dc_stats",
    "manuscript_summary",
    "descriptive_points",
    "descriptive_stats",
    "residual_matched_wide",
    "area_points",
    "area_stats",
}


class DecisionError(ValueError):
    """Raised on authentication, schema, or decision-contract failure."""


@dataclass(frozen=True)
class VerifiedTable:
    label: str
    path: Path
    sha256: str
    rows: tuple[dict[str, str], ...]
    status: str
    claim_allowed: bool


@dataclass(frozen=True)
class VerifiedBundle:
    view: int
    meta_path: Path
    meta_sha256: str
    meta: Mapping[str, Any]
    protocol_sha256: str
    package_sha256: Mapping[str, str]
    tables: Mapping[str, VerifiedTable]
    evidence_hashes: Mapping[str, Any]


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DecisionError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "JSON input")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except DecisionError:
        raise
    except Exception as exc:
        raise DecisionError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionError(f"{path}: top-level JSON must be an object")
    return value


def canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecisionError(f"value is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_regular_file(path: Path, where: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DecisionError(f"{where} is missing, non-regular, or a symlink: {path}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionError(f"{where} must be an object")
    return value


def _bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise DecisionError(f"{where} must be a boolean")
    return value


def _int(value: Any, where: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DecisionError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise DecisionError(f"{where} must be >= {minimum}")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise DecisionError(f"{where} must be a lowercase SHA-256")
    return value


def _absolute_path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DecisionError(f"{where} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise DecisionError(f"{where} must be absolute for reproducible authentication")
    return path


def _current_package_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for name in PACKAGE_FILENAMES:
        path = root / name
        _require_regular_file(path, f"current package member {name}")
        result[name] = sha256(path)
    return result


def _read_csv_exact(path: Path) -> tuple[dict[str, str], ...]:
    _require_regular_file(path, "statistics CSV")
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DecisionError(f"empty CSV: {path}") from exc
            if not header or any(not field for field in header):
                raise DecisionError(f"{path}: CSV header contains an empty field")
            if len(header) != len(set(header)):
                raise DecisionError(f"{path}: duplicate CSV header")
            rows: list[dict[str, str]] = []
            for line, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise DecisionError(
                        f"{path}, line {line}: expected {len(header)} columns, "
                        f"found {len(values)}"
                    )
                rows.append(dict(zip(header, values)))
    except DecisionError:
        raise
    except Exception as exc:
        raise DecisionError(f"cannot read CSV {path}: {exc}") from exc
    return tuple(rows)


def _csv_bool(value: str, where: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DecisionError(f"{where} must be exactly True or False, found {value!r}")


def _csv_float(value: str, where: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise DecisionError(f"{where} must be numeric, found {value!r}") from exc
    if not math.isfinite(result):
        raise DecisionError(f"{where} must be finite")
    return result


def _verify_run_manifest(
    path: Path,
    *,
    expected_sha256: str,
    view: int,
    protocol_sha256: str,
    current_package: Mapping[str, str],
    expected_command: str,
    expected_split: str,
    where: str,
) -> Mapping[str, Any]:
    _require_regular_file(path, where)
    if sha256(path) != expected_sha256:
        raise DecisionError(f"{where}: run-manifest SHA-256 mismatch")
    document = load_json(path)
    expected = {
        "schema": "dc.run_manifest.v1",
        "status": "COMPLETE",
        "command": expected_command,
        "run_kind": "FINAL",
        "views": view,
        "split": expected_split,
        "protocol_sha256": protocol_sha256,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise DecisionError(
                f"{where}.{field}={document.get(field)!r}; expected {value!r}"
            )
    package = _mapping(document.get("package_sha256"), f"{where}.package_sha256")
    normalized = {str(k): _sha(v, f"{where}.package_sha256.{k}") for k, v in package.items()}
    if normalized != dict(current_package):
        raise DecisionError(f"{where}: execution package differs from current package")
    environment = _mapping(document.get("environment"), f"{where}.environment")
    if environment.get("backend") != "astra_cuda" or environment.get("astra_cuda") is not True:
        raise DecisionError(f"{where}: formal execution was not confirmed on ASTRA CUDA")
    return document


def verify_bundle(meta_path: Path, expected_view: int) -> VerifiedBundle:
    _require_regular_file(meta_path, "statistics meta")
    meta_path = meta_path.resolve()
    meta = load_json(meta_path)
    if meta.get("schema") != META_SCHEMA or meta.get("status") != "COMPLETE":
        raise DecisionError(f"{meta_path}: statistics meta is not COMPLETE {META_SCHEMA}")
    if meta.get("views") != [expected_view]:
        raise DecisionError(
            f"{meta_path}: views must be exactly [{expected_view}], found {meta.get('views')!r}"
        )
    if meta.get("architectures") != list(ARCHITECTURES):
        raise DecisionError(f"{meta_path}: architecture contract changed")
    expected_statistical_contract = {
        "primary_estimand": "slice-weighted mean",
        "co_primary_sensitivity": "equal-patient mean",
        "inference": "paired patient-cluster percentile bootstrap",
    }
    for field, expected_value in expected_statistical_contract.items():
        if meta.get(field) != expected_value:
            raise DecisionError(
                f"{meta_path}: statistical contract {field} changed"
            )
    if _int(meta.get("n_bootstrap"), f"{meta_path}.n_bootstrap", minimum=1) < 1000:
        raise DecisionError(f"{meta_path}: fewer than 1000 patient-cluster bootstrap replicates")
    minimum_coverage = meta.get("minimum_valid_bootstrap_fraction")
    if type(minimum_coverage) not in (int, float) or float(minimum_coverage) < MIN_CURVE_BOOTSTRAP_COVERAGE:
        raise DecisionError(f"{meta_path}: bootstrap coverage contract is below 0.95")

    evidence = _mapping(meta.get("evidence"), f"{meta_path}.evidence")
    if evidence.get("authenticated") is not True or evidence.get("view") != expected_view:
        raise DecisionError(f"{meta_path}: evidence is not authenticated for view {expected_view}")
    locked = _mapping(meta.get("locked_spec"), f"{meta_path}.locked_spec")
    if locked.get("view") != expected_view:
        raise DecisionError(f"{meta_path}: locked view mismatch")
    protocol_sha = _sha(locked.get("protocol_sha256"), f"{meta_path}.locked_spec.protocol_sha256")
    if evidence.get("protocol_canonical_sha256") != protocol_sha:
        raise DecisionError(f"{meta_path}: protocol hashes disagree")

    protocol_path = _absolute_path(evidence.get("protocol"), f"{meta_path}.evidence.protocol")
    protocol_file_sha = _sha(
        evidence.get("protocol_file_sha256"),
        f"{meta_path}.evidence.protocol_file_sha256",
    )
    _require_regular_file(protocol_path, "protocol")
    if sha256(protocol_path) != protocol_file_sha:
        raise DecisionError(f"{meta_path}: protocol file SHA-256 mismatch")
    protocol = load_json(protocol_path)
    if protocol.get("schema") != "dc.protocol.v1" or protocol.get("immutable") is not True:
        raise DecisionError(f"{meta_path}: protocol is not immutable dc.protocol.v1")
    if protocol.get("views") != [125, 50]:
        raise DecisionError(f"{meta_path}: protocol view grid changed")
    if canonical_json_sha256(protocol) != protocol_sha:
        raise DecisionError(f"{meta_path}: canonical protocol SHA-256 mismatch")

    current_package = _current_package_hashes()
    run_documents: dict[str, Mapping[str, Any]] = {}
    run_evidence: dict[str, Any] = {}
    run_contracts = {
        "sweep_manifest": ("sweep-validation", "validation"),
        "test_manifest": ("evaluate-test", "test"),
    }
    for prefix, (expected_command, expected_split) in run_contracts.items():
        run_path = _absolute_path(evidence.get(prefix), f"{meta_path}.evidence.{prefix}")
        run_sha = _sha(
            evidence.get(f"{prefix}_sha256"),
            f"{meta_path}.evidence.{prefix}_sha256",
        )
        run_documents[prefix] = _verify_run_manifest(
            run_path,
            expected_sha256=run_sha,
            view=expected_view,
            protocol_sha256=protocol_sha,
            current_package=current_package,
            expected_command=expected_command,
            expected_split=expected_split,
            where=f"{meta_path}.{prefix}",
        )
        run_evidence[prefix] = {"path": str(run_path), "sha256": run_sha}
    if run_documents["sweep_manifest"].get("run_id") == run_documents["test_manifest"].get("run_id"):
        raise DecisionError(f"{meta_path}: validation and test run IDs are identical")

    # Re-authenticate every upstream file that the statistics manifest names,
    # not only the downstream tables.  This makes the decision record an
    # independently checkable evidence root rather than trust in a prior read.
    upstream_specs = (
        ("lock", "lock_file_sha256"),
        ("validation_csv", "validation_csv_sha256"),
        ("test_csv", "test_csv_sha256"),
        ("validation_patient_map", "validation_patient_map_sha256"),
        ("test_patient_map", "test_patient_map_sha256"),
    )
    upstream_evidence: dict[str, Any] = {}
    for path_field, sha_field in upstream_specs:
        upstream_path = _absolute_path(
            evidence.get(path_field), f"{meta_path}.evidence.{path_field}"
        )
        expected_upstream_sha = _sha(
            evidence.get(sha_field), f"{meta_path}.evidence.{sha_field}"
        )
        _require_regular_file(upstream_path, f"upstream evidence {path_field}")
        if sha256(upstream_path) != expected_upstream_sha:
            raise DecisionError(f"{meta_path}: upstream evidence {path_field} SHA-256 mismatch")
        upstream_evidence[path_field] = {
            "path": str(upstream_path.resolve()),
            "sha256": expected_upstream_sha,
        }

    # The CSVs consumed by the statistics layer must also be exactly the
    # case-metric outputs declared by their originating runner manifests.
    for run_prefix, evidence_field in (
        ("sweep_manifest", "validation_csv"),
        ("test_manifest", "test_csv"),
    ):
        run_outputs = _mapping(
            run_documents[run_prefix].get("outputs"),
            f"{meta_path}.{run_prefix}.outputs",
        )
        case_metrics = _mapping(
            run_outputs.get("case_metrics"),
            f"{meta_path}.{run_prefix}.outputs.case_metrics",
        )
        if _sha(
            case_metrics.get("sha256"),
            f"{meta_path}.{run_prefix}.outputs.case_metrics.sha256",
        ) != upstream_evidence[evidence_field]["sha256"]:
            raise DecisionError(
                f"{meta_path}: {run_prefix} does not bind {evidence_field}"
            )
        if _absolute_path(
            case_metrics.get("path"),
            f"{meta_path}.{run_prefix}.outputs.case_metrics.path",
        ).resolve() != Path(upstream_evidence[evidence_field]["path"]):
            raise DecisionError(
                f"{meta_path}: {run_prefix} case-metric path differs from {evidence_field}"
            )
        rows_value = case_metrics.get("rows")
        if rows_value is None and run_prefix == "sweep_manifest":
            # AMENDMENT 3 (2026-08-18, schema defect; see AMENDMENT_3_sweep_rows.md):
            # the runner's COMPLETE sweep manifest writes case_metrics as
            # {path, sha256} only — the "rows" field exists solely in the
            # in-progress writer — so this frozen integer requirement can never
            # be satisfied by a real bundle (test manifests do carry rows and
            # keep the original requirement). The CSV named here was verified
            # byte-identical to the statistics evidence a few lines above, so
            # count its data rows directly: strictly stronger evidence than
            # the absent manifest integer. No threshold, gate, selection, or
            # statistic is altered.
            with open(
                Path(upstream_evidence[evidence_field]["path"]),
                "r",
                encoding="utf-8",
            ) as stream:
                rows_value = sum(1 for _ in stream) - 1
        _int(
            rows_value,
            f"{meta_path}.{run_prefix}.outputs.case_metrics.rows",
            minimum=1,
        )

    outputs = _mapping(meta.get("outputs"), f"{meta_path}.outputs")
    if set(outputs) != REQUIRED_OUTPUTS:
        raise DecisionError(
            f"{meta_path}: output set changed; missing={sorted(REQUIRED_OUTPUTS-set(outputs))}, "
            f"extra={sorted(set(outputs)-REQUIRED_OUTPUTS)}"
        )
    tables: dict[str, VerifiedTable] = {}
    output_evidence: dict[str, Any] = {}
    for label in sorted(REQUIRED_OUTPUTS):
        descriptor = _mapping(outputs[label], f"{meta_path}.outputs.{label}")
        output_path = _absolute_path(
            descriptor.get("path"), f"{meta_path}.outputs.{label}.path"
        ).resolve()
        if output_path.parent != meta_path.parent:
            raise DecisionError(
                f"{meta_path}: output {label} is outside the statistics bundle directory"
            )
        output_sha = _sha(
            descriptor.get("sha256"), f"{meta_path}.outputs.{label}.sha256"
        )
        expected_rows = _int(
            descriptor.get("rows"), f"{meta_path}.outputs.{label}.rows", minimum=0
        )
        status = descriptor.get("status")
        if status not in ("AVAILABLE", "NOT_AVAILABLE"):
            raise DecisionError(f"{meta_path}: invalid output status for {label}")
        claim_allowed = _bool(
            descriptor.get("claim_allowed"),
            f"{meta_path}.outputs.{label}.claim_allowed",
        )
        if status == "NOT_AVAILABLE" and expected_rows != 0:
            raise DecisionError(f"{meta_path}: unavailable output {label} has rows")
        if sha256(output_path) != output_sha:
            raise DecisionError(f"{meta_path}: output {label} SHA-256 mismatch")
        rows = _read_csv_exact(output_path)
        if len(rows) != expected_rows:
            raise DecisionError(
                f"{meta_path}: output {label} row count {len(rows)} != {expected_rows}"
            )
        for line, row in enumerate(rows, start=2):
            if "view" in row and row["view"] not in ("", str(expected_view)):
                raise DecisionError(
                    f"{output_path}, line {line}: cross-view row contamination"
                )
        tables[label] = VerifiedTable(
            label=label,
            path=output_path,
            sha256=output_sha,
            rows=rows,
            status=str(status),
            claim_allowed=claim_allowed,
        )
        output_evidence[label] = {
            "path": str(output_path),
            "sha256": output_sha,
            "rows": expected_rows,
            "status": status,
        }

    return VerifiedBundle(
        view=expected_view,
        meta_path=meta_path,
        meta_sha256=sha256(meta_path),
        meta=meta,
        protocol_sha256=protocol_sha,
        package_sha256=current_package,
        tables=tables,
        evidence_hashes={
            "meta": {"path": str(meta_path), "sha256": sha256(meta_path)},
            "protocol": {"path": str(protocol_path), "sha256": protocol_file_sha, "canonical_sha256": protocol_sha},
            "run_manifests": run_evidence,
            "upstream_inputs": upstream_evidence,
            "outputs": output_evidence,
        },
    )


def _select_one(
    table: VerifiedTable,
    *,
    where: str,
    **criteria: str,
) -> Mapping[str, str]:
    rows = [
        row for row in table.rows
        if all(row.get(field) == value for field, value in criteria.items())
    ]
    if len(rows) != 1:
        raise DecisionError(
            f"{where}: expected exactly one row for {criteria}, found {len(rows)}"
        )
    return rows[0]


def _favorable_contrast(
    primary: Mapping[str, str],
    sensitivity: Mapping[str, str],
    *,
    direction: str,
    where: str,
    require_claim: bool = True,
) -> dict[str, Any]:
    if direction not in ("positive", "negative"):
        raise AssertionError(direction)
    if require_claim and not _csv_bool(primary.get("claim_allowed", ""), f"{where}.claim_allowed"):
        return {"pass": False, "reason": "claim_gate_closed"}
    mean = _csv_float(primary.get("mean_diff", ""), f"{where}.mean_diff")
    lo = _csv_float(primary.get("ci95_lo", ""), f"{where}.ci95_lo")
    hi = _csv_float(primary.get("ci95_hi", ""), f"{where}.ci95_hi")
    holm = _csv_float(primary.get("holm_p", ""), f"{where}.holm_p")
    sensitivity_mean = _csv_float(
        sensitivity.get("mean_diff", ""), f"{where}.equal_patient.mean_diff"
    )
    if direction == "positive":
        effect_ok = mean > 0
        ci_ok = lo > 0
        sensitivity_ok = sensitivity_mean > 0
    else:
        effect_ok = mean < 0
        ci_ok = hi < 0
        sensitivity_ok = sensitivity_mean < 0
    result = {
        "pass": bool(effect_ok and ci_ok and holm < ALPHA and sensitivity_ok),
        "direction": direction,
        "slice_weighted_mean_diff": mean,
        "slice_weighted_ci95": [lo, hi],
        "slice_weighted_holm_p": holm,
        "equal_patient_mean_diff": sensitivity_mean,
        "effect_direction_consistent": sensitivity_ok,
        "criteria": {
            "slice_weighted_effect_favorable": effect_ok,
            "slice_weighted_ci_favorable": ci_ok,
            "holm_adjusted_p_below_0_05": holm < ALPHA,
            "equal_patient_effect_direction_favorable": sensitivity_ok,
        },
    }
    return result


def _two_baseline_gate(
    table: VerifiedTable,
    *,
    view: int,
    baseline_field: str,
    extra: Mapping[str, str],
    direction: str,
    gate_name: str,
) -> dict[str, Any]:
    contrasts: dict[str, Any] = {}
    for baseline in BASELINES:
        common = {"view": str(view), baseline_field: baseline, **dict(extra)}
        primary = _select_one(
            table,
            where=f"{gate_name}.{view}.{baseline}.slice_weighted",
            estimand="slice_weighted",
            **common,
        )
        sensitivity = _select_one(
            table,
            where=f"{gate_name}.{view}.{baseline}.equal_patient",
            estimand="equal_patient",
            **common,
        )
        contrasts[baseline] = _favorable_contrast(
            primary,
            sensitivity,
            direction=direction,
            where=f"{gate_name}.{view}.{baseline}",
        )
    return {
        "state": "PASS" if all(item["pass"] for item in contrasts.values()) else "FAIL",
        "view": view,
        "contrasts": contrasts,
    }


def _h1_gate(bundle: VerifiedBundle) -> dict[str, Any]:
    table = bundle.tables["h1_image_prior_stats"]
    if table.status != "AVAILABLE" or not table.claim_allowed:
        return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "H1_table_not_claim_eligible"}
    return _two_baseline_gate(
        table,
        view=bundle.view,
        baseline_field="comparator",
        extra={"metric": "psnr", "endpoint_role": "primary"},
        direction="positive",
        gate_name="H1_k0_PSNR",
    )


def _h2_w_gate(bundle: VerifiedBundle) -> dict[str, Any]:
    table = bundle.tables["h2_within_model_dc_stats"]
    locked = _mapping(bundle.meta.get("locked_spec"), "locked_spec")
    selected_k = _int(locked.get("selected_k"), "locked_spec.selected_k", minimum=0)
    if selected_k <= 0:
        return {"state": "FAIL", "view": bundle.view, "reason": "validation_selected_k_is_not_finite_DC"}
    if locked.get("operating_point_claim_allowed") is not True:
        return {"state": "FAIL", "view": bundle.view, "reason": "validation_operating_gate_failed"}
    if table.status != "AVAILABLE" or not table.claim_allowed:
        return {"state": "FAIL", "view": bundle.view, "reason": "H2_operating_point_table_gate_failed"}
    common = {
        "view": str(bundle.view),
        "architecture": "WavTransUNet",
        "metric": "projection_residual",
        "endpoint_role": "primary",
        "finite_dc_iteration": str(selected_k),
    }
    primary = _select_one(table, where="H2.W.slice_weighted", estimand="slice_weighted", **common)
    sensitivity = _select_one(table, where="H2.W.equal_patient", estimand="equal_patient", **common)
    result = _favorable_contrast(
        primary,
        sensitivity,
        direction="negative",
        where=f"H2_W_projection_residual.{bundle.view}",
    )
    return {
        "state": "PASS" if result["pass"] else "FAIL",
        "view": bundle.view,
        "selected_k": selected_k,
        "contrast": result,
    }


def _locked_shared_k_gate(bundle: VerifiedBundle) -> dict[str, Any]:
    table = bundle.tables["locked_stats"]
    locked = _mapping(bundle.meta.get("locked_spec"), "locked_spec")
    selected_k = _int(locked.get("selected_k"), "locked_spec.selected_k", minimum=0)
    if selected_k <= 0 or locked.get("operating_point_claim_allowed") is not True:
        return {"state": "FAIL", "view": bundle.view, "reason": "shared_integer_k_operating_gate_failed"}
    if table.status != "AVAILABLE" or not table.claim_allowed:
        return {"state": "FAIL", "view": bundle.view, "reason": "locked_stats_not_claim_eligible"}
    return _two_baseline_gate(
        table,
        view=bundle.view,
        baseline_field="baseline",
        extra={
            "metric": "psnr",
            "candidate_dc_iteration": str(selected_k),
            "baseline_dc_iteration": str(selected_k),
        },
        direction="positive",
        gate_name="shared_integer_k_PSNR",
    )


def _curve_q50_gate(bundle: VerifiedBundle) -> dict[str, Any]:
    availability = _mapping(bundle.meta.get("analysis_availability"), "analysis_availability")
    table = bundle.tables["descriptive_stats"]
    targets = bundle.tables["targets"]
    if "inferential_curve_eligible" not in availability:
        return {
            "state": "UNAVAILABLE",
            "view": bundle.view,
            "reason": "explicit_inferential_curve_gate_absent",
        }
    explicit_meta_gate = availability.get("inferential_curve_eligible") is True
    if not explicit_meta_gate:
        return {
            "state": "FAIL",
            "view": bundle.view,
            "reason": "completed_validation_curve_gate_failed",
        }
    if table.status != "AVAILABLE" or targets.status != "AVAILABLE":
        return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "q50_target_or_curve_table_unavailable"}
    if not table.claim_allowed or not targets.claim_allowed:
        return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "q50_output_claim_gate_closed"}
    target = _select_one(
        targets,
        where=f"q50_target.{bundle.view}",
        view=str(bundle.view),
        target_id="q50",
        primary="True",
    )
    if not _csv_bool(target.get("claim_allowed", ""), "q50_target.claim_allowed"):
        return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "q50_target_claim_gate_closed"}

    contrasts: dict[str, Any] = {}
    for baseline in BASELINES:
        common = {
            "view": str(bundle.view),
            "baseline": baseline,
            "target_id": "q50",
            "primary": "True",
            "metric": "psnr",
        }
        primary = _select_one(
            table,
            where=f"q50_curve.{bundle.view}.{baseline}.slice_weighted",
            estimand="slice_weighted",
            **common,
        )
        sensitivity = _select_one(
            table,
            where=f"q50_curve.{bundle.view}.{baseline}.equal_patient",
            estimand="equal_patient",
            **common,
        )
        for estimand, row in (("slice_weighted", primary), ("equal_patient", sensitivity)):
            if row.get("inferential_curve_eligible") != "True":
                return {
                    "state": "UNAVAILABLE",
                    "view": bundle.view,
                    "reason": "row_level_inferential_curve_gate_closed",
                }
            if not _csv_bool(row.get("coverage_ok", ""), f"q50_curve.{baseline}.{estimand}.coverage_ok"):
                return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "q50_curve_point_not_covered"}
            coverage = _csv_float(
                row.get("bootstrap_coverage_fraction", ""),
                f"q50_curve.{baseline}.{estimand}.bootstrap_coverage_fraction",
            )
            if coverage < MIN_CURVE_BOOTSTRAP_COVERAGE:
                return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "q50_bootstrap_coverage_below_0_95"}
        contrasts[baseline] = _favorable_contrast(
            primary,
            sensitivity,
            direction="positive",
            where=f"q50_curve_PSNR.{bundle.view}.{baseline}",
        )
    return {
        "state": "PASS" if all(item["pass"] for item in contrasts.values()) else "FAIL",
        "view": bundle.view,
        "estimand": "validation-defined q50 aggregate PSNR frontier curve functional",
        "deployable_reconstruction": False,
        "contrasts": contrasts,
    }


def _global_dominance_gate(bundle: VerifiedBundle) -> dict[str, Any]:
    table = bundle.tables["dominance"]
    if table.status != "AVAILABLE":
        return {"state": "UNAVAILABLE", "view": bundle.view, "reason": "dominance_table_unavailable"}
    results: dict[str, Any] = {}
    all_pass = True
    for baseline in BASELINES:
        estimands: dict[str, bool] = {}
        for estimand in ("slice_weighted", "equal_patient"):
            row = _select_one(
                table,
                where=f"global_dominance.{bundle.view}.{baseline}.{estimand}",
                view=str(bundle.view),
                estimand=estimand,
                objective="psnr",
                region="global",
                candidate="WavTransUNet",
                baseline=baseline,
            )
            coverage = _csv_bool(row.get("coverage_ok", ""), "dominance.coverage_ok")
            global_available = _csv_bool(
                row.get("global_geometry_available", ""),
                "dominance.global_geometry_available",
            )
            strict = _csv_bool(
                row.get("candidate_strictly_set_dominates", ""),
                "dominance.candidate_strictly_set_dominates",
            )
            estimands[estimand] = bool(coverage and global_available and strict)
        results[baseline] = estimands
        all_pass = all_pass and all(estimands.values())
    return {
        "state": "PASS" if all_pass else "FAIL",
        "view": bundle.view,
        "definition": "every baseline Pareto-front point is strictly dominated by at least one WavTransUNet point",
        "aggregate_geometry_only": True,
        "results": results,
    }


def _question_status(per_view: Mapping[int, Mapping[str, Any]]) -> str:
    states = [per_view[view]["state"] for view in VIEWS]
    if states == ["PASS", "PASS"]:
        return "YES"
    if "PASS" in states:
        return "PARTIALLY"
    return "NO"


def _view_summary(per_view: Mapping[int, Mapping[str, Any]]) -> str:
    return "; ".join(f"{view}-view={per_view[view]['state']}" for view in VIEWS)


def make_decision(bundle_125: VerifiedBundle, bundle_50: VerifiedBundle) -> dict[str, Any]:
    bundles = {125: bundle_125, 50: bundle_50}
    if bundle_125.view != 125 or bundle_50.view != 50:
        raise DecisionError("exactly one authenticated 125-view and one 50-view bundle are required")
    if bundle_125.protocol_sha256 != bundle_50.protocol_sha256:
        raise DecisionError("cross-view protocol SHA-256 mismatch")
    if dict(bundle_125.package_sha256) != dict(bundle_50.package_sha256):
        raise DecisionError("cross-view execution-package SHA-256 mismatch")

    gates: dict[str, dict[int, dict[str, Any]]] = {
        "H1_k0_image_fidelity": {view: _h1_gate(bundle) for view, bundle in bundles.items()},
        "H2_W_projection_residual": {view: _h2_w_gate(bundle) for view, bundle in bundles.items()},
        "shared_integer_k_PSNR": {view: _locked_shared_k_gate(bundle) for view, bundle in bundles.items()},
        "q50_curve_PSNR": {view: _curve_q50_gate(bundle) for view, bundle in bundles.items()},
        "global_exact_set_dominance": {view: _global_dominance_gate(bundle) for view, bundle in bundles.items()},
    }

    framework_view_pass: dict[int, bool] = {}
    framework_view_available: dict[int, bool] = {}
    for view in VIEWS:
        components = (
            gates["H1_k0_image_fidelity"][view],
            gates["H2_W_projection_residual"][view],
            gates["shared_integer_k_PSNR"][view],
            gates["q50_curve_PSNR"][view],
        )
        framework_view_pass[view] = all(item["state"] == "PASS" for item in components)
        framework_view_available[view] = all(item["state"] != "UNAVAILABLE" for item in components)

    passing_views = [view for view in VIEWS if framework_view_pass[view]]
    if len(passing_views) == 2:
        tier_code: str | None = "A"
        tier_label = "TIER_A_CROSS_VIEW_FIXED_CHECKPOINT_FRAMEWORK"
        tier_reason = "All prespecified framework gates passed in both severe-view conditions."
    elif len(passing_views) == 1 and all(framework_view_available.values()):
        tier_code = "B"
        tier_label = f"TIER_B_VIEW_SPECIFIC_{passing_views[0]}_VIEW_ONLY"
        tier_reason = (
            f"All framework gates passed only in the {passing_views[0]}-view condition; "
            "the claim must name that condition."
        )
    elif all(framework_view_available.values()):
        tier_code = "C"
        tier_label = "TIER_C_COMPLETED_SUPERIORITY_NOT_SUPPORTED"
        tier_reason = "The completed cross-view evidence failed one or more required superiority gates."
    else:
        tier_code = None
        tier_label = "NOT_YET_CLASSIFIABLE"
        unavailable = [
            f"{name}:{view}"
            for name, values in gates.items()
            if name != "global_exact_set_dominance"
            for view, result in values.items()
            if result["state"] == "UNAVAILABLE"
        ]
        tier_reason = "Required evidence is unavailable: " + ", ".join(unavailable)

    q1_status = _question_status(gates["H1_k0_image_fidelity"])
    q2_status = _question_status(gates["H2_W_projection_residual"])
    q3_status = _question_status(gates["q50_curve_PSNR"])
    q4_status = _question_status(gates["global_exact_set_dominance"])
    q5_status = "YES" if tier_code == "A" else "PARTIALLY" if tier_code == "B" else "NO"

    if tier_code == "A":
        wording = (
            "Across the 50- and 125-view LoDoPaB-CT conditions, the fixed "
            "W-TransUNet checkpoint followed by the shared validation-selected "
            "finite-DC procedure retained higher PSNR than the fixed U-Net and "
            "TransUNet checkpoints at the deployable shared integer-k operating "
            "point and on the validation-defined q50 residual-matched curve "
            "functional, while finite DC reduced its projection residual. This "
            "supports the evaluated fixed-checkpoint pipeline as a unified "
            "learned-prior and measurement-anchoring framework."
        )
    elif tier_code == "B":
        view = passing_views[0]
        wording = (
            f"In the {view}-view LoDoPaB-CT condition, the fixed W-TransUNet "
            "checkpoint with the shared validation-selected finite-DC procedure "
            "retained higher PSNR than the fixed U-Net and TransUNet checkpoints "
            "at the deployable shared integer-k point and on the q50 residual-"
            "matched curve functional, while reducing its projection residual. "
            "This conclusion is condition-specific."
        )
    elif tier_code == "C":
        wording = (
            "The completed fixed-checkpoint evaluation does not support an "
            "unrestricted claim that W-TransUNet with finite DC outperforms both "
            "U-Net and TransUNet across the evaluated view conditions."
        )
    else:
        wording = (
            "The physics-guided framework superiority claim is not yet "
            "classifiable because at least one required authenticated operating-"
            "point or residual-matched curve result is unavailable."
        )

    common_qualification = (
        "All conclusions concern six fixed archived checkpoints under their "
        "recorded, non-equivalent training protocols; this is a predictive, "
        "fixed-prior comparison and does not identify an architectural cause. "
        "No minimum practical or clinical effect threshold was prespecified."
    )
    questions = {
        "Q1": {
            "status": q1_status,
            "answer": "Fixed-checkpoint k=0 PSNR comparison: " + _view_summary(gates["H1_k0_image_fidelity"]),
        },
        "Q2": {
            "status": q2_status,
            "answer": "W-TransUNet finite-DC projection-residual comparison: " + _view_summary(gates["H2_W_projection_residual"]),
        },
        "Q3": {
            "status": q3_status,
            "answer": "Validation-defined q50 PSNR curve-functional comparison: " + _view_summary(gates["q50_curve_PSNR"]),
        },
        "Q4": {
            "status": q4_status,
            "answer": "Exact global aggregate PSNR set-dominance result: " + _view_summary(gates["global_exact_set_dominance"]),
        },
        "Q5": {
            "status": q5_status,
            "answer": tier_reason,
        },
        "Q6": {
            "status": q5_status,
            "answer": wording + " " + common_qualification,
        },
    }

    return {
        "schema": DECISION_SCHEMA,
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision_scope": [125, 50],
        "primary_estimand": "slice-weighted mean with patient-cluster uncertainty",
        "sensitivity_estimand": "equal-patient mean effect direction",
        "alpha": ALPHA,
        "minimum_curve_bootstrap_coverage": MIN_CURVE_BOOTSTRAP_COVERAGE,
        "practical_or_clinical_effect_threshold_prespecified": False,
        "architecture_causal_claim_allowed": False,
        "curve_functional_is_deployable_fractional_k_reconstruction": False,
        "tier": {
            "code": tier_code,
            "label": tier_label,
            "view_specific_passes": passing_views,
            "reason": tier_reason,
        },
        "questions": questions,
        "gates": {name: {str(view): value for view, value in results.items()} for name, results in gates.items()},
        "recommended_wording": wording,
        "mandatory_qualification": common_qualification,
        "evidence": {
            "decision_program": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "protocol_canonical_sha256": bundle_125.protocol_sha256,
            "execution_package_sha256": dict(bundle_125.package_sha256),
            "views": {
                "125": dict(bundle_125.evidence_hashes),
                "50": dict(bundle_50.evidence_hashes),
            },
        },
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _new_output_directory(path: Path) -> Path:
    """Create or accept only an empty, non-symlink decision directory."""

    if path.is_symlink():
        raise DecisionError(f"output directory cannot be a symlink: {path}")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise DecisionError(f"output directory must be new and empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def write_decision(decision: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = _new_output_directory(output_dir)
    json_path = output_dir / "dc_cross_view_claim_decision.json"
    csv_path = output_dir / "dc_cross_view_claim_decision.csv"
    md_path = output_dir / "dc_cross_view_claim_decision.md"

    json_text = json.dumps(decision, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    rows = []
    questions = _mapping(decision["questions"], "questions")
    for question_id in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        item = _mapping(questions[question_id], question_id)
        rows.append((question_id, str(item["status"]), str(item["answer"])))
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("question", "decision", "evidence_summary"))
    writer.writerows(rows)
    csv_text = buffer.getvalue()

    tier = _mapping(decision["tier"], "tier")
    md_lines = [
        "# Cross-view finite-DC claim decision",
        "",
        f"**Classification:** {tier['label']}",
        "",
    ]
    for question_id, status, answer in rows:
        md_lines.extend((f"## {question_id} — {status}", "", answer, ""))
    md_lines.extend(
        (
            "## Claim wording",
            "",
            str(decision["recommended_wording"]),
            "",
            str(decision["mandatory_qualification"]),
            "",
            "The residual-matched q50 result is an aggregate curve functional; "
            "the shared integer-k result is the deployable operating-point comparison.",
            "",
        )
    )
    _atomic_write(json_path, json_text)
    _atomic_write(csv_path, csv_text)
    _atomic_write(md_path, "\n".join(md_lines))
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-meta-125", required=True, type=Path)
    parser.add_argument("--stats-meta-50", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_125 = verify_bundle(args.stats_meta_125, 125)
    bundle_50 = verify_bundle(args.stats_meta_50, 50)
    decision = make_decision(bundle_125, bundle_50)
    paths = write_decision(decision, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
