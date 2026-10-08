#!/usr/bin/env python3
"""Authenticate formal test evidence and lock qualitative case IDs.

No reconstruction array is opened or exported here.  Cases are selected only
after the lock, current protocol, COMPLETE test manifest, and the exact full
test metric grid have passed authentication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dc_schemas import (
    CASE_METRIC_FIELDS,
    ITERATIONS,
    MODELS,
    PROTOCOL_SCHEMA,
    SPLIT_COUNTS,
    SPLIT_FIREWALL_DISCLOSURE,
    SPLIT_PATIENTS,
    atomic_write_json,
    json_sha256,
    load_json,
    validate_case_metric_row,
    validate_lock,
    validate_protocol,
)


PRIMARY_VIEW = 50
TEST_MANIFEST_SCHEMA = "dc.run_manifest.v1"
CASE_LOCK_SCHEMA = "dc.case_lock.v1"


class QualitativeError(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualitativeError(f"{where} must be an object")
    return value


def _parse_int(value: str, field: str, line: int) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise QualitativeError(f"line {line}: invalid {field}={value!r}") from exc
    if str(parsed) != value:
        raise QualitativeError(f"line {line}: noncanonical integer {field}={value!r}")
    return parsed


def _parse_float(value: str, field: str, line: int) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise QualitativeError(f"line {line}: invalid {field}={value!r}") from exc
    if not math.isfinite(parsed):
        raise QualitativeError(f"line {line}: nonfinite {field}")
    return parsed


def _parse_bool(value: str, field: str, line: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise QualitativeError(f"line {line}: {field} must be exactly True or False")


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)
    except Exception:
        return False


def _load_lock_and_protocol(
    lock_path: Path, protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    try:
        protocol = load_json(protocol_path)
        validate_protocol(protocol)
    except Exception as exc:
        raise QualitativeError(f"invalid current protocol: {exc}") from exc
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise QualitativeError("current dc.protocol.v1 required")
    try:
        lock = load_json(lock_path)
        validate_lock(lock)
    except Exception as exc:
        raise QualitativeError(f"invalid authenticated lock: {exc}") from exc
    payload = _mapping(lock.get("payload"), "lock.payload")
    if payload.get("protocol_sha256") != json_sha256(protocol):
        raise QualitativeError("lock is not bound to the supplied current protocol")
    if payload.get("selection_split") != "validation":
        raise QualitativeError("lock is not validation-selected")
    if payload.get("views") != PRIMARY_VIEW:
        raise QualitativeError("primary qualitative selector is prespecified for 50 views")
    if payload.get("k_grid") != list(ITERATIONS):
        raise QualitativeError("lock iteration grid differs from current protocol")
    selected = _mapping(payload.get("selected"), "lock.payload.selected")
    selected_k = selected.get("k")
    if type(selected_k) is not int or selected_k not in ITERATIONS or selected_k == 0:
        raise QualitativeError("lock has no valid validation-selected finite iteration")
    checkpoint_map = _mapping(payload.get("checkpoint_sha256"), "lock.payload.checkpoint_sha256")
    if set(checkpoint_map) != set(MODELS):
        raise QualitativeError("lock checkpoint map is incomplete")
    for model in MODELS:
        digest = checkpoint_map[model]
        if not isinstance(digest, str) or len(digest) != 64:
            raise QualitativeError(f"invalid locked checkpoint hash for {model}")
    gate = _mapping(payload.get("operating_point_gate"), "lock.payload.operating_point_gate")
    status = gate.get("status")
    claim_allowed = gate.get("claim_allowed")
    fallback_used = gate.get("fallback_used")
    if status not in ("PASS", "FAIL") or type(claim_allowed) is not bool or type(
        fallback_used
    ) is not bool:
        raise QualitativeError("lock operating-point gate is invalid")
    if (status == "PASS") != claim_allowed or fallback_used == claim_allowed:
        raise QualitativeError("lock operating-point gate status is inconsistent")
    failure_reason = gate.get("failure_reason")
    if status == "FAIL" and (
        not isinstance(failure_reason, str) or not failure_reason.strip()
    ):
        raise QualitativeError("diagnostic fallback requires a failure reason")
    if status == "PASS" and failure_reason is not None:
        raise QualitativeError("claim-eligible operating point cannot have a failure reason")
    return lock, protocol, selected


def _authenticate_test_manifest(
    test_manifest_path: Path,
    test_csv: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], str, int]:
    try:
        manifest = load_json(test_manifest_path)
    except Exception as exc:
        raise QualitativeError(f"invalid test manifest: {exc}") from exc
    expected = {
        "schema": TEST_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "command": "evaluate-test",
        "run_kind": "FINAL",
        "split": "test",
        "views": PRIMARY_VIEW,
        "protocol_sha256": json_sha256(protocol),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise QualitativeError(
                f"test manifest {field}={manifest.get(field)!r}, expected {value!r}"
            )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise QualitativeError("test manifest run_id is missing")
    if manifest.get("split_firewall") != SPLIT_FIREWALL_DISCLOSURE:
        raise QualitativeError("test manifest split-firewall disclosure changed")

    inputs = _mapping(manifest.get("inputs"), "test manifest inputs")
    lock_input = _mapping(inputs.get("lock"), "test manifest inputs.lock")
    recorded_lock_path = lock_input.get("path")
    if (
        not isinstance(recorded_lock_path, str)
        or Path(recorded_lock_path).resolve() != lock_path.resolve()
    ):
        raise QualitativeError("supplied lock is not the test manifest input path")
    if lock_input.get("sha256") != sha256(lock_path):
        raise QualitativeError("test manifest used a different lock file")
    if lock_input.get("payload_sha256") != lock.get("payload_sha256"):
        raise QualitativeError("test manifest used a different lock payload")

    outputs = _mapping(manifest.get("outputs"), "test manifest outputs")
    case_output = _mapping(outputs.get("case_metrics"), "test manifest outputs.case_metrics")
    recorded_path = case_output.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != test_csv.resolve():
        raise QualitativeError("supplied test CSV is not the manifest output path")
    observed_hash = sha256(test_csv)
    if case_output.get("sha256") != observed_hash:
        raise QualitativeError("test CSV hash differs from COMPLETE manifest")
    expected_rows = SPLIT_COUNTS["test"] * len(MODELS) * len(ITERATIONS)
    if case_output.get("rows") != expected_rows:
        raise QualitativeError("test manifest output row count is incomplete")
    coverage = _mapping(manifest.get("coverage"), "test manifest coverage")
    if coverage.get("cases") != SPLIT_COUNTS["test"]:
        raise QualitativeError("test manifest case coverage is incomplete")
    if coverage.get("models") != len(MODELS) or coverage.get("rows") != expected_rows:
        raise QualitativeError("test manifest model/grid coverage is incomplete")
    return manifest, run_id, expected_rows


def _read_authenticated_test_rows(
    test_csv: Path,
    *,
    run_id: str,
    expected_rows: int,
    lock_payload: Mapping[str, Any],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    selected = _mapping(lock_payload.get("selected"), "lock.payload.selected")
    selected_k = int(selected["k"])
    checkpoint_map = _mapping(lock_payload.get("checkpoint_sha256"), "lock.payload.checkpoint_sha256")
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    patients_by_case: dict[int, int] = {}
    if test_csv.is_symlink() or not test_csv.is_file():
        raise QualitativeError("test CSV must be a regular file")
    with test_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CASE_METRIC_FIELDS:
            raise QualitativeError(
                f"canonical test CSV header mismatch: {tuple(reader.fieldnames or ())}"
            )
        for line, raw in enumerate(reader, start=2):
            item: dict[str, Any] = dict(raw)
            try:
                for field in ("case_id", "patient_id", "views", "dc_iteration"):
                    item[field] = _parse_int(str(item[field]), field, line)
                for field in ("lambda", "lambda_multiplier", "eta"):
                    item[field] = _parse_float(str(item[field]), field, line)
                item["diverged"] = _parse_bool(str(item["diverged"]), "diverged", line)
                item["validation_selected"] = _parse_bool(
                    str(item["validation_selected"]), "validation_selected", line
                )
                for field in ("psnr", "ssim", "rmse", "projection_residual"):
                    item[field] = _parse_float(str(item[field]), field, line)
                validate_case_metric_row(item)
            except QualitativeError:
                raise
            except Exception as exc:
                raise QualitativeError(f"line {line}: invalid canonical test row: {exc}") from exc
            if item["split"] != "test" or item["views"] != PRIMARY_VIEW:
                raise QualitativeError(f"line {line}: test split/view mismatch")
            if item["source_run_id"] != run_id:
                raise QualitativeError(f"line {line}: source_run_id differs from test manifest")
            if item["diverged"]:
                raise QualitativeError(f"line {line}: formal test trajectory is divergent")
            model = str(item["architecture"])
            if item["checkpoint_sha256"] != checkpoint_map[model]:
                raise QualitativeError(f"line {line}: checkpoint differs from lock for {model}")
            k = int(item["dc_iteration"])
            if item["validation_selected"] != (k == selected_k):
                raise QualitativeError(f"line {line}: validation_selected flag is inconsistent")
            if k > 0 and not all(
                _same_float(item[field], selected[field])
                for field in ("lambda", "lambda_multiplier", "eta")
            ):
                raise QualitativeError(f"line {line}: finite-DC parameters differ from lock")
            case_id = int(item["case_id"])
            patient_id = int(item["patient_id"])
            if case_id in patients_by_case and patients_by_case[case_id] != patient_id:
                raise QualitativeError(f"line {line}: patient_id changes within case")
            patients_by_case[case_id] = patient_id
            key = (case_id, model, k)
            if key in rows:
                raise QualitativeError(f"duplicate test grid row: {key}")
            rows[key] = item

    if len(rows) != expected_rows:
        raise QualitativeError(f"test CSV rows {len(rows)} != authenticated {expected_rows}")
    expected_keys = {
        (case_id, model, k)
        for case_id in range(SPLIT_COUNTS["test"])
        for model in MODELS
        for k in ITERATIONS
    }
    if set(rows) != expected_keys:
        missing = sorted(expected_keys - set(rows))[:5]
        extra = sorted(set(rows) - expected_keys)[:5]
        raise QualitativeError(f"incomplete canonical test grid; missing={missing} extra={extra}")
    if len(set(patients_by_case.values())) != SPLIT_PATIENTS["test"]:
        raise QualitativeError("test patient coverage differs from frozen protocol")
    return rows


def select_cases(
    test_csv: Path,
    lock_path: Path,
    test_manifest_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, object]:
    lock, protocol, selected = _load_lock_and_protocol(lock_path, protocol_path)
    manifest, run_id, expected_rows = _authenticate_test_manifest(
        test_manifest_path, test_csv, lock_path, lock, protocol
    )
    payload = _mapping(lock["payload"], "lock.payload")
    gate = _mapping(
        payload.get("operating_point_gate"), "lock.payload.operating_point_gate"
    )
    grid = _read_authenticated_test_rows(
        test_csv,
        run_id=run_id,
        expected_rows=expected_rows,
        lock_payload=payload,
    )
    selected_k = int(selected["k"])
    effects: list[tuple[float, int]] = []
    for case_id in range(SPLIT_COUNTS["test"]):
        values = {
            model: float(grid[(case_id, model, selected_k)]["psnr"])
            for model in MODELS
        }
        effects.append(
            (
                values["WavTransUNet"]
                - max(values["UNet"], values["TransUNet"]),
                case_id,
            )
        )
    effects.sort(key=lambda item: (item[0], item[1]))
    selections = []
    for label, quantile in (
        ("low_or_failure", 0.10),
        ("median", 0.50),
        ("high_benefit", 0.90),
    ):
        rank = int(round(quantile * (len(effects) - 1)))
        effect, case_id = effects[rank]
        selections.append(
            {
                "label": label,
                "quantile": quantile,
                "rank": rank,
                "case_id": case_id,
                "delta_psnr_vs_stronger_dc_baseline": effect,
            }
        )
    document: dict[str, object] = {
        "schema": CASE_LOCK_SCHEMA,
        "status": "LOCKED",
        "view": PRIMARY_VIEW,
        "selected_k": selected_k,
        "models": list(MODELS),
        "k_grid": list(ITERATIONS),
        "authenticated_test_rows": expected_rows,
        "test_csv_sha256": sha256(test_csv),
        "test_manifest_sha256": sha256(test_manifest_path),
        "test_run_id": manifest["run_id"],
        "dc_lock_sha256": sha256(lock_path),
        "dc_lock_payload_sha256": lock["payload_sha256"],
        "protocol_sha256": json_sha256(protocol),
        "selection_rule": "nearest ranks to q10/q50/q90 of W+DC PSNR minus max(U+DC,T+DC) PSNR; case-ID tie-break",
        "operating_point_gate": dict(gate),
        "operating_point_label": (
            "validation-selected claim-eligible operating point"
            if gate["claim_allowed"]
            else "diagnostic fallback; claim-ineligible"
        ),
        "cases": selections,
        "interpretation": "descriptive, outcome-locked before image inspection",
        "image_exported": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(output_path, document)
    except FileExistsError:
        raise
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = select_cases(
        args.test_csv,
        args.lock,
        args.test_manifest,
        args.protocol,
        args.out,
    )
    for case in result["cases"]:
        print(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
