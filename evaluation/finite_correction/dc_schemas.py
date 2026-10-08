#!/usr/bin/env python3
"""Versioned, fail-closed schemas for the architecture-neutral DC study.

The module intentionally uses only the Python standard library so that the
same validation code can run on the Mac plumbing environment and the ETRI
execution environment.  Unknown schema versions, duplicate JSON keys,
non-finite JSON numbers, extra fields, and mutable lock replacement are all
rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


PROTOCOL_SCHEMA = "dc.protocol.v1"
LOCK_SCHEMA = "dc.lock.v1"
MANIFEST_SCHEMA = "dc.manifest.v1"
CASE_METRICS_SCHEMA = "dc.case_metrics.v1"

MODELS: Tuple[str, ...] = ("UNet", "TransUNet", "WavTransUNet")
VIEWS: Tuple[int, ...] = (125, 50)
ITERATIONS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
LAMBDA_MULTIPLIERS: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

SPLIT_COUNTS = {"validation": 3522, "test": 3553}
SPLIT_PATIENTS = {"validation": 60, "test": 60}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CASE_METRIC_FIELDS: Tuple[str, ...] = (
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

SPLIT_FIREWALL_DISCLOSURE = {
    "test_file_existence_and_patient_map_metadata_may_be_read_by_dataset_constructor": True,
    "test_observation_or_ground_truth_arrays_read": False,
}


class SchemaError(ValueError):
    """Raised when a versioned document violates its declared schema."""


class ImmutableOutputError(FileExistsError):
    """Raised when an immutable output already exists."""


PathLike = Union[str, os.PathLike]


def _reject_json_constant(value: str) -> None:
    raise SchemaError("non-finite JSON constant is forbidden: %s" % value)


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity."""

    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(value, dict):
        raise SchemaError("top-level JSON value must be an object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used for all hashes."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError("value is not canonical finite JSON: %s" % exc) from exc
    return text.encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("%s must be an object" % where)
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str] = (),
    where: str = "object",
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    keys = set(value)
    missing = sorted(required_set - keys)
    extra = sorted(keys - allowed)
    if missing or extra:
        raise SchemaError(
            "%s keys invalid; missing=%s extra=%s" % (where, missing, extra)
        )


def _require_bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise SchemaError("%s must be a boolean" % where)
    return value


def _require_int(value: Any, where: str, minimum: Optional[int] = None) -> int:
    if type(value) is not int:
        raise SchemaError("%s must be an integer" % where)
    if minimum is not None and value < minimum:
        raise SchemaError("%s must be >= %d" % (where, minimum))
    return value


def _require_number(
    value: Any, where: str, minimum: Optional[float] = None
) -> float:
    if type(value) not in (int, float):
        raise SchemaError("%s must be numeric" % where)
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError("%s must be finite" % where)
    if minimum is not None and number < minimum:
        raise SchemaError("%s must be >= %s" % (where, minimum))
    return number


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % where)
    return value


def _require_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SchemaError("%s must be a lowercase 64-hex SHA-256" % where)
    return value


def validate_split_firewall(value: Any, where: str = "split_firewall") -> Mapping[str, Any]:
    value = _require_mapping(value, where)
    _require_exact_keys(value, SPLIT_FIREWALL_DISCLOSURE, where=where)
    for key, expected in SPLIT_FIREWALL_DISCLOSURE.items():
        actual = _require_bool(value[key], "%s.%s" % (where, key))
        if actual is not expected:
            raise SchemaError(
                "%s.%s must be %r" % (where, key, expected)
            )
    return value


def validate_protocol(document: Any) -> Mapping[str, Any]:
    """Validate the frozen protocol with exact v1 semantics."""

    doc = _require_mapping(document, "protocol")
    _require_exact_keys(
        doc,
        (
            "schema",
            "protocol_id",
            "immutable",
            "models",
            "views",
            "dataset",
            "geometry",
            "projection_endpoint",
            "dc",
            "validation_selection",
            "residual_matching",
            "statistics",
            "split_firewall",
            "output_schemas",
        ),
        where="protocol",
    )
    if doc["schema"] != PROTOCOL_SCHEMA:
        raise SchemaError("unsupported protocol schema: %r" % doc["schema"])
    _require_string(doc["protocol_id"], "protocol.protocol_id")
    if _require_bool(doc["immutable"], "protocol.immutable") is not True:
        raise SchemaError("protocol.immutable must be true")
    if list(doc["models"]) != list(MODELS):
        raise SchemaError("protocol.models must equal %r" % (list(MODELS),))
    if list(doc["views"]) != list(VIEWS):
        raise SchemaError("protocol.views must equal %r" % (list(VIEWS),))

    dataset = _require_mapping(doc["dataset"], "protocol.dataset")
    _require_exact_keys(dataset, ("name", "version", "splits"), where="protocol.dataset")
    if dataset["name"] != "LoDoPaB-CT" or dataset["version"] != "1.0.0":
        raise SchemaError("dataset must be LoDoPaB-CT v1.0.0")
    splits = _require_mapping(dataset["splits"], "protocol.dataset.splits")
    _require_exact_keys(splits, ("validation", "test"), where="protocol.dataset.splits")
    for split in ("validation", "test"):
        item = _require_mapping(splits[split], "protocol.dataset.splits.%s" % split)
        _require_exact_keys(item, ("slices", "patients"), where="protocol.dataset.splits.%s" % split)
        if _require_int(item["slices"], "%s.slices" % split) != SPLIT_COUNTS[split]:
            raise SchemaError("wrong %s slice count" % split)
        if _require_int(item["patients"], "%s.patients" % split) != SPLIT_PATIENTS[split]:
            raise SchemaError("wrong %s patient count" % split)

    geometry = _require_mapping(doc["geometry"], "protocol.geometry")
    _require_exact_keys(
        geometry,
        ("operator_shape", "network_shape", "embed_slice", "fbp_border_pixels", "frame_source"),
        where="protocol.geometry",
    )
    if geometry["operator_shape"] != [362, 362] or geometry["network_shape"] != [352, 352]:
        raise SchemaError("geometry shapes must be 362x362 and 352x352")
    if geometry["embed_slice"] != "[5:357,5:357]":
        raise SchemaError("geometry.embed_slice must be [5:357,5:357]")
    if geometry["fbp_border_pixels"] != 5 or geometry["frame_source"] != "condition_matched_fbp":
        raise SchemaError("the five-pixel condition-matched FBP frame is mandatory")

    endpoint = _require_mapping(doc["projection_endpoint"], "protocol.projection_endpoint")
    _require_exact_keys(endpoint, ("formula", "sinogram", "interpretation"), where="protocol.projection_endpoint")
    if endpoint["formula"] != "||A*x-y||_2/||y||_2" or endpoint["sinogram"] != "measured_post_log":
        raise SchemaError("projection endpoint definition changed")

    dc = _require_mapping(doc["dc"], "protocol.dc")
    _require_exact_keys(
        dc,
        (
            "objective",
            "update",
            "inner_products",
            "initial_state",
            "anchor",
            "restore_fbp_border_after_every_step",
            "clipping",
            "positivity_projection",
            "lambda_scale",
            "lambda_multipliers",
            "power_iteration",
            "step_size",
            "iterations",
            "divergence",
        ),
        where="protocol.dc",
    )
    if dc["objective"] != "0.5*||A*x-y||_Y^2 + 0.5*lambda*||x-x_net||_X^2":
        raise SchemaError("DC objective changed")
    if dc["update"] != "x_next=restore_fbp_border(x-eta*(A_star*(A*x-y)+lambda*(x-x_net)))":
        raise SchemaError("DC update changed")
    inner_products = _require_mapping(dc["inner_products"], "protocol.dc.inner_products")
    expected_inner_products = {
        "domain": "X is the ODL reconstruction-space discretized inner product",
        "range": "Y is the ODL projection-space discretized inner product",
        "adjoint": "A_star satisfies <A*x,z>_Y=<x,A_star*z>_X",
        "toy_fallback": "Euclidean inner products for operators without ODL domain/range inner products",
    }
    if inner_products != expected_inner_products:
        raise SchemaError("DC inner-product contract changed")
    if dc["initial_state"] != "x_net" or dc["anchor"] != "x_net":
        raise SchemaError("DC must start from and anchor to x_net")
    if _require_bool(dc["restore_fbp_border_after_every_step"], "dc.restore_fbp_border_after_every_step") is not True:
        raise SchemaError("FBP frame restoration is mandatory")
    if _require_bool(dc["clipping"], "dc.clipping") is not False:
        raise SchemaError("clipping must remain disabled")
    if _require_bool(dc["positivity_projection"], "dc.positivity_projection") is not False:
        raise SchemaError("positivity projection must remain disabled")
    scale = _require_mapping(dc["lambda_scale"], "protocol.dc.lambda_scale")
    _require_exact_keys(scale, ("formula", "scope", "per_view"), where="protocol.dc.lambda_scale")
    if scale["formula"] != "mean_model_case(||A_star*(A*x-y)||_X/||x||_X)":
        raise SchemaError("lambda scale formula changed")
    if scale["scope"] != "all_models_all_validation_cases" or scale["per_view"] is not True:
        raise SchemaError("lambda scale must be common across all models and validation cases, per view")
    if tuple(float(v) for v in dc["lambda_multipliers"]) != LAMBDA_MULTIPLIERS:
        raise SchemaError("lambda multiplier grid changed")
    power = _require_mapping(dc["power_iteration"], "protocol.dc.power_iteration")
    _require_exact_keys(power, ("iterations", "seed", "quantity"), where="protocol.dc.power_iteration")
    if power != {"iterations": 20, "seed": 0, "quantity": "lambda_max(A_star*A)"}:
        raise SchemaError("power-iteration contract changed")
    step = _require_mapping(dc["step_size"], "protocol.dc.step_size")
    _require_exact_keys(step, ("formula", "shared_across_models"), where="protocol.dc.step_size")
    if step["formula"] != "1/(L+lambda)" or step["shared_across_models"] is not True:
        raise SchemaError("step-size contract changed")
    if tuple(dc["iterations"]) != ITERATIONS:
        raise SchemaError("iteration grid changed")
    divergence = _require_mapping(dc["divergence"], "protocol.dc.divergence")
    _require_exact_keys(divergence, ("nonfinite", "residual_ratio_above"), where="protocol.dc.divergence")
    if divergence["nonfinite"] is not True or float(divergence["residual_ratio_above"]) != 2.0:
        raise SchemaError("divergence rule changed")

    selection = _require_mapping(doc["validation_selection"], "protocol.validation_selection")
    _require_exact_keys(
        selection,
        (
            "split",
            "shared_candidate_across_models",
            "psnr_floor_db",
            "median_residual_ratio_upper_bound_exclusive",
            "gate_scope",
            "score",
            "tie_break_order",
            "failure_status",
            "fallback_is_claim_eligible",
        ),
        where="protocol.validation_selection",
    )
    if selection["split"] != "validation" or selection["shared_candidate_across_models"] is not True:
        raise SchemaError("selection must be validation-only and architecture-neutral")
    if (
        float(selection["psnr_floor_db"]) != -0.10
        or float(selection["median_residual_ratio_upper_bound_exclusive"]) != 1.0
        or selection["gate_scope"] != "every_model"
    ):
        raise SchemaError("common operating-point gate changed")
    if selection["score"] != "minimize_worst_model_median_residual_ratio":
        raise SchemaError("minimax selection score changed")
    if selection["tie_break_order"] != ["mean_model_ratio", "smaller_k", "larger_lambda"]:
        raise SchemaError("selection tie breaks changed")
    if selection["failure_status"] != "FAIL" or selection["fallback_is_claim_eligible"] is not False:
        raise SchemaError("fail-closed fallback contract changed")

    matching = _require_mapping(doc["residual_matching"], "protocol.residual_matching")
    _require_exact_keys(
        matching,
        ("interval", "spacing", "quantiles", "primary", "extrapolation"),
        where="protocol.residual_matching",
    )
    if matching["interval"] != "intersection_of_model_mean_absolute_residual_ranges_across_k_at_validation_selected_lambda":
        raise SchemaError("common validation absolute-residual interval changed")
    if matching["spacing"] != "log" or matching["quantiles"] != {"q25": 0.25, "q50": 0.5, "q75": 0.75}:
        raise SchemaError("residual target grid changed")
    if matching["primary"] != "q50" or matching["extrapolation"] != "forbidden":
        raise SchemaError("q50 must be primary and extrapolation forbidden")

    stats = _require_mapping(doc["statistics"], "protocol.statistics")
    _require_exact_keys(stats, ("bootstrap_replicates", "seed", "resampling_unit"), where="protocol.statistics")
    if stats != {"bootstrap_replicates": 10000, "seed": 0, "resampling_unit": "patient_cluster"}:
        raise SchemaError("bootstrap contract changed")

    validate_split_firewall(doc["split_firewall"], "protocol.split_firewall")
    schemas = _require_mapping(doc["output_schemas"], "protocol.output_schemas")
    _require_exact_keys(schemas, ("manifest", "case_metrics", "lock"), where="protocol.output_schemas")
    if schemas != {"manifest": MANIFEST_SCHEMA, "case_metrics": CASE_METRICS_SCHEMA, "lock": LOCK_SCHEMA}:
        raise SchemaError("output schema versions changed")
    return doc


def validate_manifest(document: Any) -> Mapping[str, Any]:
    doc = _require_mapping(document, "manifest")
    _require_exact_keys(
        doc,
        ("schema", "manifest_id", "hash_algorithm", "assets", "split_firewall"),
        where="manifest",
    )
    if doc["schema"] != MANIFEST_SCHEMA:
        raise SchemaError("unsupported manifest schema: %r" % doc["schema"])
    _require_string(doc["manifest_id"], "manifest.manifest_id")
    if doc["hash_algorithm"] != "sha256":
        raise SchemaError("manifest.hash_algorithm must be sha256")
    validate_split_firewall(doc["split_firewall"], "manifest.split_firewall")
    if not isinstance(doc["assets"], list) or not doc["assets"]:
        raise SchemaError("manifest.assets must be a non-empty array")
    seen = set()
    for index, raw in enumerate(doc["assets"]):
        where = "manifest.assets[%d]" % index
        asset = _require_mapping(raw, where)
        _require_exact_keys(
            asset,
            (
                "asset_id",
                "role",
                "split",
                "locator",
                "resolved",
                "sha256",
                "byte_size",
                "evidence",
                "metadata",
            ),
            optional=("unresolved_reason",),
            where=where,
        )
        asset_id = _require_string(asset["asset_id"], where + ".asset_id")
        if asset_id in seen:
            raise SchemaError("duplicate asset_id: %s" % asset_id)
        seen.add(asset_id)
        _require_string(asset["role"], where + ".role")
        if asset["split"] not in ("shared", "validation", "test", "server"):
            raise SchemaError("%s.split is invalid" % where)
        _require_string(asset["locator"], where + ".locator")
        resolved = _require_bool(asset["resolved"], where + ".resolved")
        if not isinstance(asset["evidence"], list) or not asset["evidence"]:
            raise SchemaError("%s.evidence must be a non-empty array" % where)
        for evidence in asset["evidence"]:
            _require_string(evidence, where + ".evidence[]")
        _require_mapping(asset["metadata"], where + ".metadata")
        if resolved:
            _require_sha256(asset["sha256"], where + ".sha256")
            _require_int(asset["byte_size"], where + ".byte_size", minimum=1)
            if "unresolved_reason" in asset:
                raise SchemaError("resolved asset %s cannot have unresolved_reason" % asset_id)
        else:
            if asset["sha256"] is not None or asset["byte_size"] is not None:
                raise SchemaError("unresolved asset %s cannot invent hash or size" % asset_id)
            _require_string(asset.get("unresolved_reason"), where + ".unresolved_reason")
    return doc


def validate_case_metric_row(row: Any) -> Mapping[str, Any]:
    item = _require_mapping(row, "case metric row")
    _require_exact_keys(item, CASE_METRIC_FIELDS, where="case metric row")
    if item["schema"] != CASE_METRICS_SCHEMA:
        raise SchemaError("unsupported case-metrics schema: %r" % item["schema"])
    split = item["split"]
    if split not in SPLIT_COUNTS:
        raise SchemaError("case metric split must be validation or test")
    case_id = _require_int(item["case_id"], "case_id", minimum=0)
    if case_id >= SPLIT_COUNTS[split]:
        raise SchemaError("case_id is outside the official %s split" % split)
    _require_int(item["patient_id"], "patient_id", minimum=0)
    if item["views"] not in VIEWS:
        raise SchemaError("views must be 125 or 50")
    if item["architecture"] not in MODELS:
        raise SchemaError("unknown architecture")
    _require_sha256(item["checkpoint_sha256"], "checkpoint_sha256")
    iteration = _require_int(item["dc_iteration"], "dc_iteration", minimum=0)
    if iteration not in ITERATIONS:
        raise SchemaError("dc_iteration is outside the frozen grid")
    diverged = _require_bool(item["diverged"], "diverged")
    _require_bool(item["validation_selected"], "validation_selected")
    _require_string(item["backend"], "backend")
    _require_string(item["source_run_id"], "source_run_id")
    if iteration == 0:
        for field in ("lambda", "lambda_multiplier", "eta"):
            if float(_require_number(item[field], field)) != 0.0:
                raise SchemaError("k=0 requires %s=0" % field)
    else:
        _require_number(item["lambda"], "lambda", minimum=0.0)
        if float(item["lambda"]) <= 0.0:
            raise SchemaError("lambda must be positive for k>0")
        multiplier = _require_number(item["lambda_multiplier"], "lambda_multiplier", minimum=0.0)
        if multiplier not in LAMBDA_MULTIPLIERS:
            raise SchemaError("lambda_multiplier is outside the frozen grid")
        if _require_number(item["eta"], "eta", minimum=0.0) <= 0.0:
            raise SchemaError("eta must be positive for k>0")
    metric_fields = ("psnr", "ssim", "rmse", "projection_residual")
    if diverged:
        if any(item[field] is not None for field in metric_fields):
            raise SchemaError("diverged rows must use JSON null for all metrics")
    else:
        _require_number(item["psnr"], "psnr")
        ssim = _require_number(item["ssim"], "ssim")
        if not -1.0 <= ssim <= 1.0:
            raise SchemaError("ssim must be in [-1,1]")
        _require_number(item["rmse"], "rmse", minimum=0.0)
        _require_number(item["projection_residual"], "projection_residual", minimum=0.0)
    return item


def _validate_lock_payload(value: Any) -> Mapping[str, Any]:
    payload = _require_mapping(value, "lock payload")
    required = ("protocol_sha256", "selection_split", "split_firewall")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise SchemaError("lock payload is missing required fields: %s" % missing)
    _require_sha256(payload["protocol_sha256"], "lock payload.protocol_sha256")
    validate_split_firewall(
        payload["split_firewall"], "lock payload.split_firewall"
    )
    if payload["selection_split"] != "validation":
        raise SchemaError("lock payload selection_split must be validation")
    if "views" in payload and payload["views"] not in VIEWS:
        raise SchemaError("lock payload views must be 125 or 50")
    if "protocol_schema" in payload and payload["protocol_schema"] != PROTOCOL_SCHEMA:
        raise SchemaError("lock payload protocol schema changed")
    if "manifest_schema" in payload and payload["manifest_schema"] != MANIFEST_SCHEMA:
        raise SchemaError("lock payload manifest schema changed")
    if "manifest_sha256" in payload:
        _require_sha256(payload["manifest_sha256"], "lock payload.manifest_sha256")
    return payload


def make_lock(payload: Any) -> Dict[str, Any]:
    payload = dict(_validate_lock_payload(payload))
    return {
        "schema": LOCK_SCHEMA,
        "locked": True,
        "payload": payload,
        "payload_sha256": json_sha256(payload),
    }


def validate_lock(document: Any) -> Mapping[str, Any]:
    doc = _require_mapping(document, "lock")
    _require_exact_keys(doc, ("schema", "locked", "payload", "payload_sha256"), where="lock")
    if doc["schema"] != LOCK_SCHEMA:
        raise SchemaError("unsupported lock schema: %r" % doc["schema"])
    if _require_bool(doc["locked"], "lock.locked") is not True:
        raise SchemaError("lock.locked must be true")
    payload = _validate_lock_payload(doc["payload"])
    _require_sha256(doc["payload_sha256"], "lock.payload_sha256")
    actual = json_sha256(payload)
    if doc["payload_sha256"] != actual:
        raise SchemaError("lock payload self-hash mismatch")
    return doc


def validate_document(document: Any) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise SchemaError("document must be an object")
    schema = document.get("schema")
    if schema == PROTOCOL_SCHEMA:
        return validate_protocol(document)
    if schema == MANIFEST_SCHEMA:
        return validate_manifest(document)
    if schema == LOCK_SCHEMA:
        return validate_lock(document)
    if schema == CASE_METRICS_SCHEMA:
        return validate_case_metric_row(document)
    raise SchemaError("unknown schema: %r" % schema)


def atomic_write_json(path: PathLike, value: Any, overwrite: bool = False) -> Path:
    """Atomically write canonical JSON; immutable by default.

    With ``overwrite=False`` a hard-link publish is used, so an existing lock
    can never be replaced between a check and a rename.  With ``overwrite=True``
    ``os.replace`` provides atomic publication for ordinary derived outputs.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(str(temporary), str(destination))
        else:
            try:
                os.link(str(temporary), str(destination))
            except FileExistsError as exc:
                raise ImmutableOutputError(
                    "immutable output already exists: %s" % destination
                ) from exc
            temporary.unlink()
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync.  The atomic
            # publish itself has already succeeded.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_lock(path: PathLike, lock: Any) -> Path:
    validate_lock(lock)
    return atomic_write_json(path, lock, overwrite=False)


__all__ = [
    "CASE_METRICS_SCHEMA",
    "CASE_METRIC_FIELDS",
    "ITERATIONS",
    "LAMBDA_MULTIPLIERS",
    "LOCK_SCHEMA",
    "MANIFEST_SCHEMA",
    "MODELS",
    "PROTOCOL_SCHEMA",
    "SPLIT_COUNTS",
    "SPLIT_FIREWALL_DISCLOSURE",
    "SchemaError",
    "ImmutableOutputError",
    "VIEWS",
    "atomic_write_json",
    "canonical_json_bytes",
    "json_sha256",
    "load_json",
    "make_lock",
    "validate_case_metric_row",
    "validate_document",
    "validate_lock",
    "validate_manifest",
    "validate_protocol",
    "validate_split_firewall",
    "write_lock",
]
