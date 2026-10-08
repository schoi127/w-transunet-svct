#!/usr/bin/env python3
"""Architecture-neutral validation selection and immutable DC lock creation."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dc_schemas import (
    ITERATIONS,
    LAMBDA_MULTIPLIERS,
    MANIFEST_SCHEMA,
    MODELS,
    PROTOCOL_SCHEMA,
    SPLIT_FIREWALL_DISCLOSURE,
    SchemaError,
    VIEWS,
    json_sha256,
    make_lock,
    validate_case_metric_row,
    validate_lock,
    validate_manifest,
    validate_protocol,
    write_lock,
)


PSNR_FLOOR_DB = -0.10
RESIDUAL_RATIO_UPPER_BOUND_EXCLUSIVE = 1.0
TARGET_QUANTILES = {"q25": 0.25, "q50": 0.5, "q75": 0.75}
PRIMARY_TARGET = "q50"


class SelectionError(RuntimeError):
    """Raised when validation evidence is incomplete or inconsistent."""


def _finite(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SelectionError("%s must be finite" % where)
    return float(value)


def compute_common_validation_scale(
    rows: Iterable[Mapping[str, Any]],
    expected_models: Sequence[str] = MODELS,
    expected_cases: Optional[int] = None,
) -> float:
    """Compute mean ||A^T(Ax-y)||/||x|| over all models and cases.

    Rows must belong to one view and contain ``architecture``, ``case_id``,
    ``adjoint_data_gradient_norm``, and ``image_norm``.  Model-specific case
    sets must be identical; omissions and duplicates fail closed.
    """

    expected_models = tuple(expected_models)
    if tuple(expected_models) != MODELS:
        raise SelectionError("the frozen model set or order changed")
    seen: Dict[str, set] = {model: set() for model in expected_models}
    values: List[float] = []
    views = set()
    for index, row in enumerate(rows):
        if row.get("split", "validation") != "validation":
            raise SelectionError("scale row %d is not validation data" % index)
        model = row.get("architecture")
        if model not in expected_models:
            raise SelectionError("scale row %d has unknown architecture" % index)
        case_id = row.get("case_id")
        if type(case_id) is not int or case_id < 0:
            raise SelectionError("scale row %d has invalid case_id" % index)
        if case_id in seen[model]:
            raise SelectionError("duplicate scale row for %s case %d" % (model, case_id))
        seen[model].add(case_id)
        numerator = _finite(row.get("adjoint_data_gradient_norm"), "adjoint_data_gradient_norm")
        denominator = _finite(row.get("image_norm"), "image_norm")
        if numerator < 0.0 or denominator <= 0.0:
            raise SelectionError("scale norms must satisfy numerator>=0 and denominator>0")
        values.append(numerator / denominator)
        if "views" in row:
            if row["views"] not in VIEWS:
                raise SelectionError("scale row has unsupported view count")
            views.add(row["views"])
    if not values:
        raise SelectionError("no validation scale rows")
    if len(views) > 1:
        raise SelectionError("lambda scale is computed separately for each view")
    reference_cases = seen[expected_models[0]]
    if not reference_cases:
        raise SelectionError("no cases for %s" % expected_models[0])
    for model in expected_models[1:]:
        if seen[model] != reference_cases:
            raise SelectionError("model validation case sets are not identical")
    if expected_cases is not None and len(reference_cases) != expected_cases:
        raise SelectionError(
            "expected %d validation cases, observed %d"
            % (expected_cases, len(reference_cases))
        )
    scale = sum(values) / len(values)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SelectionError("common validation scale is not finite and positive")
    return scale


def aggregate_validation_candidates(
    rows: Iterable[Mapping[str, Any]],
    expected_models: Sequence[str] = MODELS,
    expected_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create per-model candidate medians from aligned case-metric rows.

    Residual ratio is computed per case relative to that architecture's k=0
    residual, then median-aggregated.  Delta PSNR is paired to the same k=0
    case.  Any divergent case marks the complete shared candidate divergent.
    """

    expected_models = tuple(expected_models)
    if expected_models != MODELS:
        raise SelectionError("the frozen model set or order changed")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise SelectionError("no case-metric rows")
    for row in materialized:
        validate_case_metric_row(row)
        if row["split"] != "validation":
            raise SelectionError("selection may consume validation rows only")
    views = {row["views"] for row in materialized}
    if len(views) != 1:
        raise SelectionError("candidate aggregation requires exactly one view")
    view = next(iter(views))

    baseline: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    candidates: Dict[Tuple[str, float, float, float, int], Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in materialized:
        model = row["architecture"]
        case_id = row["case_id"]
        if row["dc_iteration"] == 0:
            key = (model, case_id)
            if key in baseline:
                raise SelectionError("duplicate k=0 row for %s case %d" % key)
            if row["diverged"]:
                raise SelectionError("k=0 baseline cannot be divergent")
            if row["projection_residual"] <= 0.0:
                raise SelectionError("k=0 residual must be positive")
            baseline[key] = row
        else:
            key = (
                model,
                float(row["lambda_multiplier"]),
                float(row["lambda"]),
                float(row["eta"]),
                int(row["dc_iteration"]),
            )
            if case_id in candidates[key]:
                raise SelectionError("duplicate candidate case row: %r case %d" % (key, case_id))
            candidates[key][case_id] = row

    model_cases = {
        model: {case_id for architecture, case_id in baseline if architecture == model}
        for model in expected_models
    }
    reference_cases = model_cases[expected_models[0]]
    if not reference_cases:
        raise SelectionError("missing k=0 baseline rows")
    for model in expected_models[1:]:
        if model_cases[model] != reference_cases:
            raise SelectionError("k=0 case sets are not identical across models")
    if expected_cases is not None and len(reference_cases) != expected_cases:
        raise SelectionError(
            "expected %d validation cases, observed %d"
            % (expected_cases, len(reference_cases))
        )

    summaries: List[Dict[str, Any]] = []
    for key in sorted(candidates, key=lambda item: (item[4], item[1], item[0])):
        model, multiplier, lam, eta, k = key
        if multiplier not in LAMBDA_MULTIPLIERS or k not in ITERATIONS or k == 0:
            raise SelectionError("candidate is outside the frozen grid")
        rows_by_case = candidates[key]
        if set(rows_by_case) != reference_cases:
            raise SelectionError("candidate %r does not have complete aligned coverage" % (key,))
        any_diverged = any(row["diverged"] for row in rows_by_case.values())
        baseline_mean = sum(
            float(baseline[(model, case_id)]["projection_residual"])
            for case_id in reference_cases
        ) / len(reference_cases)
        if any_diverged:
            residual_median = None
            delta_psnr_median = None
            mean_residual = None
        else:
            ratios = []
            deltas = []
            residuals = []
            for case_id in sorted(reference_cases):
                current = rows_by_case[case_id]
                base = baseline[(model, case_id)]
                ratio = current["projection_residual"] / base["projection_residual"]
                if not math.isfinite(ratio) or ratio <= 0.0:
                    raise SelectionError("non-positive residual ratio")
                ratios.append(ratio)
                deltas.append(current["psnr"] - base["psnr"])
                residuals.append(float(current["projection_residual"]))
            residual_median = float(median(ratios))
            delta_psnr_median = float(median(deltas))
            mean_residual = sum(residuals) / len(residuals)
        summaries.append(
            {
                "views": view,
                "architecture": model,
                "lambda_multiplier": multiplier,
                "lambda": lam,
                "eta": eta,
                "k": k,
                "median_residual_ratio": residual_median,
                "median_delta_psnr_db": delta_psnr_median,
                "mean_projection_residual": mean_residual,
                "mean_baseline_projection_residual": baseline_mean,
                "diverged": any_diverged,
                "n_cases": len(reference_cases),
            }
        )
    if not summaries:
        raise SelectionError("no k>0 candidates")
    return summaries


SUMMARY_FIELDS = {
    "views",
    "architecture",
    "lambda_multiplier",
    "lambda",
    "eta",
    "k",
    "median_residual_ratio",
    "median_delta_psnr_db",
    "mean_projection_residual",
    "mean_baseline_projection_residual",
    "diverged",
    "n_cases",
}


def _group_summaries(
    summaries: Iterable[Mapping[str, Any]],
    expected_models: Sequence[str] = MODELS,
    require_complete_grid: bool = True,
) -> Tuple[int, Dict[Tuple[float, int], Dict[str, Mapping[str, Any]]]]:
    expected_models = tuple(expected_models)
    if expected_models != MODELS:
        raise SelectionError("the frozen model set or order changed")
    groups: Dict[Tuple[float, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    views = set()
    for index, raw in enumerate(summaries):
        row = dict(raw)
        missing = SUMMARY_FIELDS - set(row)
        extra = set(row) - SUMMARY_FIELDS
        if missing or extra:
            raise SelectionError("summary %d keys invalid; missing=%s extra=%s" % (index, sorted(missing), sorted(extra)))
        if row["architecture"] not in expected_models:
            raise SelectionError("unknown summary architecture")
        if row["views"] not in VIEWS:
            raise SelectionError("summary view count is outside protocol")
        views.add(row["views"])
        multiplier = _finite(row["lambda_multiplier"], "lambda_multiplier")
        lam = _finite(row["lambda"], "lambda")
        eta = _finite(row["eta"], "eta")
        k = row["k"]
        if multiplier not in LAMBDA_MULTIPLIERS or lam <= 0.0 or eta <= 0.0:
            raise SelectionError("summary candidate has invalid lambda or eta")
        if type(k) is not int or k not in ITERATIONS or k == 0:
            raise SelectionError("summary candidate has invalid k")
        if type(row["diverged"]) is not bool:
            raise SelectionError("summary diverged must be boolean")
        if type(row["n_cases"]) is not int or row["n_cases"] <= 0:
            raise SelectionError("summary n_cases must be positive")
        if row["diverged"]:
            if any(
                row[field] is not None
                for field in (
                    "median_residual_ratio",
                    "median_delta_psnr_db",
                    "mean_projection_residual",
                )
            ):
                raise SelectionError("divergent summary endpoints must be null")
        else:
            ratio = _finite(row["median_residual_ratio"], "median_residual_ratio")
            _finite(row["median_delta_psnr_db"], "median_delta_psnr_db")
            mean_residual = _finite(row["mean_projection_residual"], "mean_projection_residual")
            if ratio <= 0.0:
                raise SelectionError("median residual ratio must be positive")
            if mean_residual <= 0.0:
                raise SelectionError("mean projection residual must be positive")
        if _finite(
            row["mean_baseline_projection_residual"],
            "mean_baseline_projection_residual",
        ) <= 0.0:
            raise SelectionError("mean baseline projection residual must be positive")
        key = (multiplier, k)
        if row["architecture"] in groups[key]:
            raise SelectionError("duplicate architecture in candidate %r" % (key,))
        groups[key][row["architecture"]] = row
    if len(views) != 1:
        raise SelectionError("selection requires summaries from exactly one view")
    if require_complete_grid:
        expected_keys = {
            (multiplier, k)
            for multiplier in LAMBDA_MULTIPLIERS
            for k in ITERATIONS
            if k != 0
        }
        missing = sorted(expected_keys - set(groups))
        extra = sorted(set(groups) - expected_keys)
        if missing or extra:
            raise SelectionError(
                "validation candidate grid is incomplete; missing=%s extra=%s"
                % (missing, extra)
            )
    for key, group in groups.items():
        if set(group) != set(expected_models):
            raise SelectionError("candidate %r is incomplete across models" % (key,))
        lambdas = {float(item["lambda"]) for item in group.values()}
        etas = {float(item["eta"]) for item in group.values()}
        counts = {item["n_cases"] for item in group.values()}
        if len(lambdas) != 1 or len(etas) != 1 or len(counts) != 1:
            raise SelectionError("candidate %r is not shared identically across models" % (key,))
    for model in expected_models:
        baselines = {
            float(group[model]["mean_baseline_projection_residual"])
            for group in groups.values()
        }
        if len(baselines) != 1:
            raise SelectionError("baseline mean residual changed across candidates for %s" % model)
    return next(iter(views)), groups


def _candidate_is_finite(group: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(not row["diverged"] for row in group.values())


def _candidate_passes_gate(
    group: Mapping[str, Mapping[str, Any]], psnr_floor_db: float
) -> bool:
    return _candidate_is_finite(group) and all(
        float(row["median_delta_psnr_db"]) >= psnr_floor_db
        and float(row["median_residual_ratio"])
        < RESIDUAL_RATIO_UPPER_BOUND_EXCLUSIVE
        for row in group.values()
    )


def common_validation_residual_interval(
    summaries: Iterable[Mapping[str, Any]],
    selected_operating_point: Mapping[str, Any],
    expected_models: Sequence[str] = MODELS,
    require_complete_grid: bool = True,
) -> Dict[str, Any]:
    """Intersect model mean absolute-residual ranges at the selected lambda.

    The support for each model consists of its k=0 mean residual plus every
    nondivergent k on the validation-selected lambda trajectory.  The
    intersection is the only range in which later residual matching is
    allowed; no alternate lambda or extrapolation fallback is permitted.
    """

    view, groups = _group_summaries(
        summaries, expected_models, require_complete_grid=require_complete_grid
    )
    if selected_operating_point.get("selection_split") != "validation":
        raise SelectionError("operating point was not selected on validation")
    if selected_operating_point.get("views") != view:
        raise SelectionError("operating-point view does not match summaries")
    selected = selected_operating_point.get("selected")
    if not isinstance(selected, Mapping):
        return {
            "status": "FAIL",
            "claim_allowed": False,
            "fallback_used": True,
            "failure_reason": "no_validation_selected_lambda",
            "views": view,
            "selected_lambda_multiplier": None,
            "selected_lambda": None,
            "lower": None,
            "upper": None,
            "trajectory_k": [],
            "model_ranges": {},
        }
    multiplier = _finite(selected.get("lambda_multiplier"), "selected lambda_multiplier")
    lam = _finite(selected.get("lambda"), "selected lambda")
    selected_k = selected.get("k")
    if type(selected_k) is not int or selected_k not in ITERATIONS or selected_k == 0:
        raise SelectionError("selected k is outside the frozen grid")
    lambda_groups = {key: group for key, group in groups.items() if key[0] == multiplier}
    expected_k = {k for k in ITERATIONS if k != 0}
    if {key[1] for key in lambda_groups} != expected_k and require_complete_grid:
        raise SelectionError("selected lambda trajectory is incomplete")
    if (multiplier, selected_k) not in lambda_groups:
        raise SelectionError("selected lambda/k is absent from validation summaries")
    trajectory = {
        key: group
        for key, group in lambda_groups.items()
        if _candidate_is_finite(group)
    }
    if not trajectory:
        raise SelectionError("selected lambda has no finite validation trajectory")
    trajectory_lambdas = {
        float(next(iter(group.values()))["lambda"])
        for group in trajectory.values()
    }
    if trajectory_lambdas != {lam}:
        raise SelectionError("selected lambda value does not match its trajectory")
    model_ranges = {}
    for model in expected_models:
        baseline_values = {
            float(group[model]["mean_baseline_projection_residual"])
            for group in trajectory.values()
        }
        if len(baseline_values) != 1:
            raise SelectionError("baseline residual is not stable for %s" % model)
        values = list(baseline_values) + [
            float(group[model]["mean_projection_residual"])
            for group in trajectory.values()
        ]
        model_ranges[model] = (min(values), max(values))
    lower = max(bounds[0] for bounds in model_ranges.values())
    upper = min(bounds[1] for bounds in model_ranges.values())
    # A singleton intersection cannot define interpolation or a normalized
    # frontier-area interval.  Treat it as unavailable rather than inventing
    # three coincident residual targets.
    interval_exists = lower > 0.0 and upper > 0.0 and lower < upper
    parent_claim_allowed = selected_operating_point.get("claim_allowed") is True
    status = "PASS" if interval_exists and parent_claim_allowed else "FAIL"
    if not interval_exists:
        reason = (
            "selected_lambda_model_mean_absolute_residual_ranges_have_no_"
            "positive_width_common_interval"
        )
    elif not parent_claim_allowed:
        reason = "selected_operating_point_is_diagnostic_fallback"
    else:
        reason = None
    return {
        "status": status,
        "claim_allowed": status == "PASS",
        "fallback_used": status != "PASS",
        "failure_reason": reason,
        "views": view,
        "selected_lambda_multiplier": multiplier,
        "selected_lambda": lam,
        "lower": lower if interval_exists else None,
        "upper": upper if interval_exists else None,
        "trajectory_k": [0] + sorted(key[1] for key in trajectory),
        "model_ranges": {
            model: list(model_ranges[model]) for model in expected_models
        },
    }


def log_spaced_targets(interval: Mapping[str, Any]) -> Dict[str, Any]:
    lower = _finite(interval.get("lower"), "interval.lower")
    upper = _finite(interval.get("upper"), "interval.upper")
    if lower <= 0.0 or upper <= 0.0 or lower >= upper:
        raise SelectionError(
            "target interval must be positive and have strictly positive width"
        )
    log_lower = math.log(lower)
    log_width = math.log(upper) - log_lower
    targets = {
        label: math.exp(log_lower + quantile * log_width)
        for label, quantile in TARGET_QUANTILES.items()
    }
    return {
        "status": interval.get("status", "PASS"),
        "claim_allowed": bool(interval.get("claim_allowed", True)),
        "fallback_used": bool(interval.get("fallback_used", False)),
        "failure_reason": interval.get("failure_reason"),
        "views": interval.get("views"),
        "interval": [lower, upper],
        "spacing": "log",
        "targets": targets,
        "primary": PRIMARY_TARGET,
        "extrapolation": "forbidden",
    }


def select_operating_point(
    summaries: Iterable[Mapping[str, Any]],
    psnr_floor_db: float = PSNR_FLOOR_DB,
    expected_models: Sequence[str] = MODELS,
    require_complete_grid: bool = True,
) -> Dict[str, Any]:
    """Select the one shared lambda/k before defining residual targets."""

    view, groups = _group_summaries(
        summaries, expected_models, require_complete_grid=require_complete_grid
    )
    finite_groups = {key: group for key, group in groups.items() if _candidate_is_finite(group)}
    eligible = {key: group for key, group in finite_groups.items() if _candidate_passes_gate(group, psnr_floor_db)}
    fallback_used = not eligible
    pool = eligible if eligible else finite_groups
    if not pool:
        return {
            "status": "FAIL",
            "claim_allowed": False,
            "fallback_used": True,
            "failure_reason": "all_shared_candidates_diverged_or_were_nonfinite",
            "selection_split": "validation",
            "views": view,
            "psnr_floor_db": psnr_floor_db,
            "median_residual_ratio_upper_bound_exclusive": RESIDUAL_RATIO_UPPER_BOUND_EXCLUSIVE,
            "selected": None,
        }

    ranked = []
    for (multiplier, k), group in pool.items():
        model_ratios = {
            model: float(group[model]["median_residual_ratio"])
            for model in expected_models
        }
        worst = max(model_ratios.values())
        mean_score = sum(model_ratios.values()) / len(model_ratios)
        lam = float(next(iter(group.values()))["lambda"])
        ranked.append(((worst, mean_score, k, -lam), multiplier, k, lam, group, model_ratios))
    ranked.sort(key=lambda item: item[0])
    score, multiplier, k, lam, group, model_ratios = ranked[0]
    selected_row = next(iter(group.values()))
    status = "PASS" if eligible else "FAIL"
    reason = "no_shared_candidate_met_common_operating_point_gate" if fallback_used else None
    return {
        "status": status,
        "claim_allowed": status == "PASS",
        "fallback_used": fallback_used,
        "failure_reason": reason,
        "selection_split": "validation",
        "views": view,
        "psnr_floor_db": psnr_floor_db,
        "median_residual_ratio_upper_bound_exclusive": RESIDUAL_RATIO_UPPER_BOUND_EXCLUSIVE,
        "selected": {
            "lambda_multiplier": multiplier,
            "lambda": lam,
            "eta": float(selected_row["eta"]),
            "k": k,
            "worst_model_median_residual_ratio": score[0],
            "mean_model_median_residual_ratio": score[1],
            "models": {
                model: {
                    "median_residual_ratio": float(group[model]["median_residual_ratio"]),
                    "median_delta_psnr_db": float(group[model]["median_delta_psnr_db"]),
                    "mean_projection_residual": float(group[model]["mean_projection_residual"]),
                }
                for model in expected_models
            },
        },
    }


def select_all_targets(
    summaries: Iterable[Mapping[str, Any]],
    psnr_floor_db: float = PSNR_FLOOR_DB,
    require_complete_grid: bool = True,
) -> Dict[str, Any]:
    materialized = [dict(row) for row in summaries]
    operating_point = select_operating_point(
        materialized,
        psnr_floor_db=psnr_floor_db,
        require_complete_grid=require_complete_grid,
    )
    interval = common_validation_residual_interval(
        materialized,
        operating_point,
        require_complete_grid=require_complete_grid,
    )
    if interval["lower"] is None:
        target_definition = {
            "status": "FAIL",
            "claim_allowed": False,
            "fallback_used": True,
            "failure_code": "NO_COMMON_INTERVAL",
            "failure_reason": interval["failure_reason"],
            "views": interval["views"],
            "interval": None,
            "spacing": "log",
            "targets": {},
            "primary": PRIMARY_TARGET,
            "extrapolation": "forbidden",
            "selected_lambda_multiplier": interval[
                "selected_lambda_multiplier"
            ],
            "selected_lambda": interval["selected_lambda"],
            "trajectory_k": interval["trajectory_k"],
            "model_ranges": interval["model_ranges"],
        }
        return {
            "status": "FAIL",
            "claim_allowed": False,
            "fallback_used": True,
            "failure_reason": interval["failure_reason"],
            "views": interval["views"],
            "selected_operating_point": operating_point,
            "target_definition": target_definition,
        }
    target_definition = log_spaced_targets(interval)
    claim_allowed = operating_point["claim_allowed"] and target_definition["claim_allowed"]
    return {
        "status": "PASS" if claim_allowed else "FAIL",
        "claim_allowed": claim_allowed,
        "fallback_used": not claim_allowed,
        "failure_reason": None if claim_allowed else target_definition.get("failure_reason") or "one_or_more_operating_points_failed",
        "views": interval["views"],
        "selected_operating_point": operating_point,
        "target_definition": target_definition,
    }


def build_frozen_lock(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selections_by_view: Mapping[str, Any],
    require_all_assets_resolved: bool = False,
    require_selection_assets_resolved: bool = True,
) -> Dict[str, Any]:
    """Bind protocol, asset manifest, and validation selections by self-hash."""

    validate_protocol(protocol)
    validate_manifest(manifest)
    unresolved_all = [
        asset["asset_id"] for asset in manifest["assets"] if not asset["resolved"]
    ]
    unresolved_selection = [
        asset["asset_id"]
        for asset in manifest["assets"]
        if not asset["resolved"] and asset["split"] in ("shared", "validation")
    ]
    if require_all_assets_resolved and unresolved_all:
        raise SelectionError(
            "cannot freeze a formal lock with unresolved assets: %s"
            % ", ".join(unresolved_all)
        )
    if require_selection_assets_resolved and unresolved_selection:
        raise SelectionError(
            "cannot freeze a validation lock with unresolved selection assets: %s"
            % ", ".join(unresolved_selection)
        )
    expected_view_keys = {str(view) for view in VIEWS}
    if set(selections_by_view) != expected_view_keys:
        raise SelectionError("lock requires validation selections for views 125 and 50")
    for view_key, selection in selections_by_view.items():
        if not isinstance(selection, Mapping):
            raise SelectionError("selection %s must be an object" % view_key)
        if selection.get("views") != int(view_key):
            raise SelectionError("selection view key/value mismatch")
        point = selection.get("selected_operating_point")
        if not isinstance(point, Mapping) or point.get("selection_split") != "validation":
            raise SelectionError("selection %s operating point is not validation-only" % view_key)
        target_definition = selection.get("target_definition")
        if not isinstance(target_definition, Mapping):
            raise SelectionError(
                "selection %s lacks a residual-matching definition" % view_key
            )
        target_keys = set(target_definition.get("targets", {}))
        positive_targets = target_keys == set(TARGET_QUANTILES)
        negative_keys = {
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
        negative_targets = (
            set(target_definition) == negative_keys
            and
            target_definition.get("status") == "FAIL"
            and target_definition.get("claim_allowed") is False
            and target_definition.get("fallback_used") is True
            and target_definition.get("failure_code") == "NO_COMMON_INTERVAL"
            and target_definition.get("interval") is None
            and target_keys == set()
            and target_definition.get("views") == int(view_key)
            and target_definition.get("spacing") == "log"
            and target_definition.get("extrapolation") == "forbidden"
            and isinstance(target_definition.get("failure_reason"), str)
            and bool(target_definition.get("failure_reason"))
        )
        if not positive_targets and not negative_targets:
            raise SelectionError(
                "selection %s has neither q25/q50/q75 nor an authenticated "
                "NO_COMMON_INTERVAL result" % view_key
            )
        if target_definition.get("primary") != PRIMARY_TARGET:
            raise SelectionError("selection %s does not retain q50 as primary" % view_key)
    payload = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": json_sha256(protocol),
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_sha256": json_sha256(manifest),
        "selection_split": "validation",
        "split_firewall": dict(SPLIT_FIREWALL_DISCLOSURE),
        "selections_by_view": dict(selections_by_view),
    }
    lock = make_lock(payload)
    validate_lock(lock)
    return lock


def assert_lock_matches(
    lock: Mapping[str, Any], protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    validate_lock(lock)
    validate_protocol(protocol)
    validate_manifest(manifest)
    payload = lock["payload"]
    if payload.get("protocol_sha256") != json_sha256(protocol):
        raise SelectionError("lock does not match the protocol")
    if payload.get("manifest_sha256") != json_sha256(manifest):
        raise SelectionError("lock does not match the asset manifest")


def write_frozen_lock(path: str, lock: Mapping[str, Any]) -> None:
    validate_lock(lock)
    write_lock(path, lock)


__all__ = [
    "PRIMARY_TARGET",
    "PSNR_FLOOR_DB",
    "RESIDUAL_RATIO_UPPER_BOUND_EXCLUSIVE",
    "SelectionError",
    "TARGET_QUANTILES",
    "aggregate_validation_candidates",
    "assert_lock_matches",
    "build_frozen_lock",
    "common_validation_residual_interval",
    "compute_common_validation_scale",
    "log_spaced_targets",
    "select_all_targets",
    "select_operating_point",
    "write_frozen_lock",
]
