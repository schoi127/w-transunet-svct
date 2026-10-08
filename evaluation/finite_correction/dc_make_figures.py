#!/usr/bin/env python3
"""Render authenticated empirical aggregate finite-DC trajectories.

Only COMPLETE, hash-bound tables published by ``dc_pareto_stats.py`` are
accepted.  Pareto flags are descriptive properties of empirical aggregate
points; the plots do not perform selection, interpolation, or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dc_schemas import ITERATIONS, MODELS, VIEWS, load_json


COLORS = {"UNet": "#0072B2", "TransUNet": "#D55E00", "WavTransUNet": "#009E73"}
MARKERS = {"UNet": "o", "TransUNet": "s", "WavTransUNet": "^"}
ANALYSIS_LABEL = (
    "Empirical aggregate trajectories; descriptive union-Pareto geometry"
)
MATCHED_RESIDUAL_UNAVAILABLE_LABEL = (
    "Matched-residual inference unavailable: no positive-width common "
    "validation residual interval"
)
STATS_META_SCHEMA = "dc.pareto_stats_manifest.v1"
FIGURE_MANIFEST_SCHEMA = "dc.figure_manifest.v1"
REQUIRED_TABLES = {
    "aggregate": "aggregate_operating_points.csv",
    "targets": "validation_locked_residual_targets.csv",
}


class FigureError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FigureError(f"figure input is not a regular file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise FigureError(f"missing CSV header: {path}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise FigureError(f"empty figure input: {path}")
    return rows


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureError(f"{where} must be an object")
    return value


def _finite(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except Exception as exc:
        raise FigureError(f"invalid {key}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise FigureError(f"nonfinite {key}")
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    raw = row.get(key)
    try:
        value = int(str(raw))
    except Exception as exc:
        raise FigureError(f"invalid integer {key}: {raw!r}") from exc
    if str(value) != str(raw):
        raise FigureError(f"noncanonical integer {key}: {raw!r}")
    return value


def _boolean(row: Mapping[str, str], key: str) -> bool:
    value = row.get(key)
    if value == "True":
        return True
    if value == "False":
        return False
    raise FigureError(f"{key} must be exactly True or False")


def _optional_interval(
    row: Mapping[str, str], key: str, point: float
) -> tuple[float, float] | None:
    """Read an optional authenticated pointwise 95% interval.

    Current formal statistics provide patient-cluster bootstrap intervals for
    all aggregate endpoints.  This parser identifies a missing pair, and the
    formal-grid validator rejects it rather than silently emitting a figure
    without uncertainty.  A percentile interval is not required to contain
    the original point estimate.
    """

    lo_key = f"{key}_ci95_lo"
    hi_key = f"{key}_ci95_hi"
    has_lo = lo_key in row
    has_hi = hi_key in row
    if not has_lo and not has_hi:
        return None
    if has_lo != has_hi or row.get(lo_key) in (None, "") or row.get(hi_key) in (
        None,
        "",
    ):
        raise FigureError(f"incomplete aggregate 95% interval for {key}")
    lo = _finite(row, lo_key)
    hi = _finite(row, hi_key)
    if lo > hi:
        raise FigureError(f"invalid aggregate 95% interval for {key}")
    return lo, hi


def _authenticate_stats_tables(
    stats_dir: Path, view: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], dict[str, dict[str, Any]], Path]:
    if view not in VIEWS:
        raise FigureError(f"unsupported view count: {view}")
    meta_path = stats_dir / "dc_pareto_stats_meta.json"
    try:
        meta = load_json(meta_path)
    except Exception as exc:
        raise FigureError(f"invalid statistics manifest {meta_path}: {exc}") from exc
    if meta.get("schema") != STATS_META_SCHEMA or meta.get("status") != "COMPLETE":
        raise FigureError("COMPLETE dc.pareto_stats_manifest.v1 required")
    if meta.get("architectures") != list(MODELS):
        raise FigureError("statistics manifest model set/order changed")
    if meta.get("views") != [view]:
        raise FigureError("statistics manifest must contain exactly the requested locked view")
    locked = _mapping(meta.get("locked_spec"), "statistics manifest locked_spec")
    if locked.get("view") != view:
        raise FigureError("locked statistics view mismatch")
    if locked.get("k_grid") != list(ITERATIONS):
        raise FigureError("locked iteration grid differs from protocol")
    selected_k = locked.get("selected_k")
    if type(selected_k) is not int or selected_k not in ITERATIONS or selected_k == 0:
        raise FigureError("invalid validation-selected finite iteration")
    overall_claim_allowed = locked.get("claim_allowed")
    operating_point_claim_allowed = locked.get("operating_point_claim_allowed")
    residual_matching_claim_allowed = locked.get("residual_matching_claim_allowed")
    for field, value in (
        ("claim_allowed", overall_claim_allowed),
        ("operating_point_claim_allowed", operating_point_claim_allowed),
        ("residual_matching_claim_allowed", residual_matching_claim_allowed),
    ):
        if type(value) is not bool:
            raise FigureError(f"statistics locked_spec.{field} must be Boolean")
    if overall_claim_allowed is not (
        operating_point_claim_allowed and residual_matching_claim_allowed
    ):
        raise FigureError("statistics locked claim gates are internally inconsistent")
    if meta.get("inference") != "paired patient-cluster percentile bootstrap":
        raise FigureError("patient-cluster bootstrap inference is required for figures")
    grids = _mapping(meta.get("iteration_grids"), "statistics manifest iteration_grids")
    if grids.get(str(view)) != list(ITERATIONS):
        raise FigureError("statistics aggregate iteration grid differs from lock")
    outputs = _mapping(meta.get("outputs"), "statistics manifest outputs")
    tables: dict[str, list[dict[str, str]]] = {}
    authenticated: dict[str, dict[str, Any]] = {}
    for label, filename in REQUIRED_TABLES.items():
        entry = _mapping(outputs.get(label), f"statistics manifest outputs.{label}")
        digest = entry.get("sha256")
        row_count = entry.get("rows")
        status = entry.get("status", "AVAILABLE")
        if not isinstance(digest, str) or len(digest) != 64:
            raise FigureError(f"invalid statistics hash for {label}")
        if status not in ("AVAILABLE", "NOT_AVAILABLE"):
            raise FigureError(f"invalid statistics availability status for {label}")
        if type(row_count) is not int or row_count < 0:
            raise FigureError(f"invalid statistics row count for {label}")
        if label == "aggregate" and (status != "AVAILABLE" or row_count <= 0):
            raise FigureError("aggregate trajectories must be AVAILABLE and nonempty")
        if label == "targets":
            if status == "AVAILABLE" and row_count <= 0:
                raise FigureError("AVAILABLE residual-target table must be nonempty")
            if status == "NOT_AVAILABLE":
                if row_count != 0:
                    raise FigureError("NOT_AVAILABLE residual-target table must have zero rows")
                if entry.get("claim_allowed") is not False:
                    raise FigureError("NOT_AVAILABLE residual targets cannot allow claims")
                if entry.get("failure_code") != "NO_COMMON_INTERVAL":
                    raise FigureError("NOT_AVAILABLE residual targets require NO_COMMON_INTERVAL")
                if not isinstance(entry.get("failure_reason"), str) or not entry.get(
                    "failure_reason"
                ):
                    raise FigureError("NOT_AVAILABLE residual targets require a failure reason")
        path = stats_dir / filename
        recorded_path = entry.get("path")
        if (
            not isinstance(recorded_path, str)
            or Path(recorded_path).resolve() != path.resolve()
        ):
            raise FigureError(f"statistics output path mismatch for {label}")
        rows = read_csv(path, allow_empty=status == "NOT_AVAILABLE")
        observed_hash = sha256(path)
        if observed_hash != digest:
            raise FigureError(f"statistics input hash mismatch for {label}")
        if len(rows) != row_count:
            raise FigureError(
                f"statistics input row count mismatch for {label}: {len(rows)} != {row_count}"
            )
        tables[label] = rows
        authenticated[label] = {
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "rows": len(rows),
            "status": status,
            "claim_allowed": entry.get("claim_allowed"),
            "failure_code": entry.get("failure_code", ""),
            "failure_reason": entry.get("failure_reason", ""),
            "meta_output_label": label,
        }
    return meta, tables, authenticated, meta_path


def _validated_plot_rows(
    aggregate: Sequence[Mapping[str, str]],
    targets: Sequence[Mapping[str, str]],
    view: int,
    *,
    expected_n_bootstrap: int,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    expected_all = {
        (split, estimand, model, k)
        for split in ("validation", "test")
        for estimand in ("slice_weighted", "equal_patient")
        for model in MODELS
        for k in ITERATIONS
    }
    observed_all: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for raw in aggregate:
        row = dict(raw)
        if _integer(row, "view") != view:
            raise FigureError("aggregate table contains a view outside the locked analysis")
        split = row.get("split")
        estimand = row.get("estimand")
        model = row.get("architecture")
        k = _integer(row, "dc_iteration")
        key = (str(split), str(estimand), str(model), k)
        if key not in expected_all:
            raise FigureError(f"aggregate contains an unlocked analysis point: {key}")
        if key in observed_all:
            raise FigureError(f"duplicate aggregate analysis point: {key}")
        for metric in ("projection_residual", "psnr", "ssim"):
            point = _finite(row, metric)
            if _optional_interval(row, metric, point) is None:
                raise FigureError(
                    f"authenticated patient-cluster 95% interval required for {metric}"
                )
        if row.get("ci_method") != "patient-cluster percentile bootstrap":
            raise FigureError("aggregate CI method is not patient-cluster bootstrap")
        if row.get("resampling_unit") != "patient":
            raise FigureError("aggregate CI resampling unit must be patient")
        if _integer(row, "n_bootstrap") != expected_n_bootstrap:
            raise FigureError("aggregate n_bootstrap differs from statistics manifest")
        coverage = _finite(row, "bootstrap_coverage_fraction")
        if not 0.95 <= coverage <= 1.0:
            raise FigureError("aggregate bootstrap coverage fraction is unacceptable")
        if row.get("coverage_ok") != "True":
            raise FigureError("aggregate bootstrap coverage is not acceptable")
        _boolean(row, "pareto_psnr")
        _boolean(row, "pareto_ssim")
        union_psnr = _boolean(row, "pareto_union_psnr")
        union_ssim = _boolean(row, "pareto_union_ssim")
        if union_psnr and not _boolean(row, "pareto_psnr"):
            raise FigureError("union PSNR frontier point cannot be within-model dominated")
        if union_ssim and not _boolean(row, "pareto_ssim"):
            raise FigureError("union SSIM frontier point cannot be within-model dominated")
        observed_all[key] = row
    if set(observed_all) != expected_all:
        missing = sorted(expected_all - set(observed_all))
        extra = sorted(set(observed_all) - expected_all)
        raise FigureError(f"incomplete locked aggregate grid; missing={missing} extra={extra}")
    by_model = {
        model: [
            observed_all[("test", "slice_weighted", model, k)]
            for k in ITERATIONS
        ]
        for model in MODELS
    }

    matching = [dict(row) for row in targets]
    if not matching:
        return by_model, []
    if any(_integer(row, "view") != view for row in matching):
        raise FigureError("target table contains a view outside the locked analysis")
    if len(matching) != 3 or {row.get("target_id") for row in matching} != {"q25", "q50", "q75"}:
        raise FigureError("locked residual targets must contain exactly q25/q50/q75")
    for row in matching:
        _finite(row, "target_residual")
        primary = _boolean(row, "primary")
        if primary != (row["target_id"] == "q50"):
            raise FigureError("q50 must be the sole primary residual target")
    matching.sort(key=lambda row: _finite(row, "target_residual"))
    return by_model, matching


def _style_axes(ax: Any) -> None:
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_model_trajectory(
    ax: Any,
    rows: Sequence[Mapping[str, str]],
    model: str,
    *,
    x_key: str,
    y_key: str,
    pareto_key: str | None,
    union_pareto_key: str | None,
    selected_k: int,
    figure_label: str,
    annotate_iterations: bool,
    operating_point_claim_allowed: bool,
) -> dict[str, Any]:
    x = [_finite(row, x_key) if x_key != "dc_iteration" else _integer(row, x_key) for row in rows]
    y = [_finite(row, y_key) for row in rows]
    color = COLORS[model]
    ax.plot(x, y, color=color, linewidth=1.5, label=model.replace("WavTransUNet", "W-TransUNet"))
    uncertainty_drawn = False
    x_ci_k: list[int] = []
    y_ci_k: list[int] = []
    annotations: list[dict[str, Any]] = []
    annotation_offsets = {
        "UNet": (-10, -13),
        "TransUNet": (10, -1),
        "WavTransUNet": (-10, 11),
    }
    for row, x_value, y_value in zip(rows, x, y):
        k = _integer(row, "dc_iteration")
        x_interval = (
            None
            if x_key == "dc_iteration"
            else _optional_interval(row, x_key, float(x_value))
        )
        y_interval = _optional_interval(row, y_key, float(y_value))
        # To preserve readability, uncertainty is shown at the two operating
        # points that anchor the scientific comparison.  All point estimates
        # remain visible.  The authenticated intervals are still validated at
        # every k above, so missing all-point uncertainty cannot be hidden.
        if k in (0, selected_k) and x_interval is not None:
            x_artist = ax.hlines(
                y_value,
                x_interval[0],
                x_interval[1],
                color=color,
                linewidth=0.75,
                alpha=0.38,
                zorder=2,
            )
            x_artist.set_gid(f"ci_x_{figure_label}_{model}_k{k}")
            uncertainty_drawn = True
            x_ci_k.append(k)
        if k in (0, selected_k) and y_interval is not None:
            y_artist = ax.vlines(
                x_value,
                y_interval[0],
                y_interval[1],
                color=color,
                linewidth=0.75,
                alpha=0.38,
                zorder=2,
            )
            y_artist.set_gid(f"ci_y_{figure_label}_{model}_k{k}")
            uncertainty_drawn = True
            y_ci_k.append(k)

        within_pareto = True if pareto_key is None else _boolean(row, pareto_key)
        union_pareto = (
            True if union_pareto_key is None else _boolean(row, union_pareto_key)
        )
        if union_pareto_key is None:
            pareto_class = "not_applicable"
            facecolor = color
            edgecolor = color
            marker_alpha = 1.0
            marker_size = 34
            marker_width = 1.1
        elif union_pareto:
            pareto_class = "union_frontier"
            facecolor = color
            edgecolor = color
            marker_alpha = 1.0
            marker_size = 38
            marker_width = 1.1
        elif within_pareto:
            pareto_class = "within_model_only"
            facecolor = "white"
            edgecolor = color
            marker_alpha = 1.0
            marker_size = 36
            marker_width = 1.1
        else:
            pareto_class = "dominated_within_model"
            facecolor = "white"
            edgecolor = color
            marker_alpha = 0.22
            marker_size = 24
            marker_width = 0.7
        marker_artist = ax.scatter(
            [x_value],
            [y_value],
            marker=MARKERS[model],
            s=marker_size,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=marker_width,
            alpha=marker_alpha,
            zorder=3,
        )
        marker_artist.set_gid(
            f"point_{figure_label}_{model}_k{k}_{pareto_class}"
        )
        if k == 0:
            k0_artist = ax.scatter([x_value], [y_value], marker="X", s=72, facecolors="white", edgecolors="black", linewidths=1.0, zorder=4)
            k0_artist.set_gid(f"network_output_{figure_label}_{model}_k0")
        if k == selected_k:
            op_color = "#F0E442" if operating_point_claim_allowed else "#B3B3B3"
            op_artist = ax.scatter([x_value], [y_value], marker="*", s=125, facecolors=op_color, edgecolors="black", linewidths=0.8, zorder=5)
            op_role = "validation_selected" if operating_point_claim_allowed else "diagnostic_fallback"
            op_artist.set_gid(
                f"operating_point_{op_role}_{figure_label}_{model}_k{k}"
            )
        if annotate_iterations:
            dx, dy = annotation_offsets[model]
            role = (
                "validation_selected"
                if k == selected_k and operating_point_claim_allowed
                else "diagnostic_fallback"
                if k == selected_k
                else "network_output"
                if k == 0
                else "trajectory"
            )
            label = f"k={k}"
            annotation = ax.annotate(
                label,
                (x_value, y_value),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="center",
                fontsize=6.3,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.45,
                },
                zorder=6,
            )
            annotation.set_gid(f"annotation_{figure_label}_{model}_k{k}")
            annotations.append(
                {
                    "architecture": model,
                    "k": k,
                    "text": label,
                    "role": role,
                    "offset_points": [dx, dy],
                }
            )
    return {
        "uncertainty_drawn": uncertainty_drawn,
        "x_ci_k": sorted(set(x_ci_k)),
        "y_ci_k": sorted(set(y_ci_k)),
        "annotations": annotations,
    }


def _add_marker_legend(
    ax: Any,
    selected_k: int,
    *,
    include_pareto: bool,
    include_uncertainty: bool,
    operating_point_claim_allowed: bool,
) -> None:
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    extra = [
        Line2D([], [], marker="X", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=7, label="k=0 (network output)"),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markerfacecolor=(
                "#F0E442" if operating_point_claim_allowed else "#B3B3B3"
            ),
            markeredgecolor="black",
            markersize=10,
            label=(
                f"k={selected_k} (validation-selected; claim-eligible)"
                if operating_point_claim_allowed
                else f"k={selected_k} (diagnostic fallback; claim-ineligible)"
            ),
        ),
    ]
    if include_pareto:
        extra.extend(
            [
                Line2D([], [], marker="o", linestyle="None", markerfacecolor="#666666", markeredgecolor="#666666", markersize=5, label="filled: union-frontier point"),
                Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#666666", markersize=5, label="open: within-model frontier only"),
                Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#999999", alpha=0.25, markersize=5, label="faint: dominated within model"),
            ]
        )
    if include_uncertainty:
        extra.append(
            Line2D(
                [],
                [],
                color="#666666",
                linewidth=0.8,
                alpha=0.55,
                label="95% patient-cluster bootstrap CI",
            )
        )
    ax.legend(handles + extra, labels + [item.get_label() for item in extra], frameon=False, fontsize=7)


def _save_svg(fig: Any, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    fig.savefig(partial, format="svg")
    partial.replace(path)


def make_figures(stats_dir: Path, output_dir: Path, *, view: int) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Preserve semantic text and artist IDs in SVG so manuscript assembly and
    # automated review can audit every k annotation and uncertainty segment.
    matplotlib.rcParams["svg.fonttype"] = "none"

    meta, tables, authenticated, meta_path = _authenticate_stats_tables(stats_dir, view)
    expected_n_bootstrap = meta.get("n_bootstrap")
    if type(expected_n_bootstrap) is not int or expected_n_bootstrap <= 0:
        raise FigureError("statistics manifest n_bootstrap must be a positive integer")
    by_model, targets = _validated_plot_rows(
        tables["aggregate"],
        tables["targets"],
        view,
        expected_n_bootstrap=expected_n_bootstrap,
    )
    locked = _mapping(meta["locked_spec"], "statistics manifest locked_spec")
    selected_k = int(locked["selected_k"])
    overall_claim_allowed = bool(locked["claim_allowed"])
    operating_point_claim_allowed = bool(locked["operating_point_claim_allowed"])
    residual_matching_claim_allowed = bool(locked["residual_matching_claim_allowed"])
    target_status = str(authenticated["targets"]["status"])
    matched_residual_available = target_status == "AVAILABLE"
    if matched_residual_available != bool(targets):
        raise FigureError("residual-target status and authenticated table disagree")
    if matched_residual_available is not residual_matching_claim_allowed:
        raise FigureError(
            "residual-target availability and locked residual-matching gate disagree"
        )
    if authenticated["targets"].get("claim_allowed") is not residual_matching_claim_allowed:
        raise FigureError(
            "residual-target output claim gate and locked residual-matching gate disagree"
        )
    availability = meta.get("analysis_availability")
    if availability is not None:
        availability_map = _mapping(availability, "statistics analysis_availability")
        if availability_map.get("residual_matched") is not matched_residual_available:
            raise FigureError("statistics residual-matched availability disagrees with targets")
        if availability_map.get("pareto_superiority") is not matched_residual_available:
            raise FigureError("statistics Pareto-superiority availability disagrees with targets")

    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FigureError(f"figure output directory must be new and empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    outputs: dict[str, Path] = {}
    uncertainty_by_figure: dict[str, bool] = {}
    uncertainty_geometry: dict[str, dict[str, Any]] = {}
    iteration_annotations: dict[str, dict[str, Any]] = {}
    plot_specs = (
        ("psnr_vs_residual", "projection_residual", "psnr", "pareto_psnr", "pareto_union_psnr", "Normalized projection residual", "PSNR (dB)"),
        ("psnr_vs_k", "dc_iteration", "psnr", None, None, "Finite DC iteration k", "PSNR (dB)"),
        ("residual_vs_k", "dc_iteration", "projection_residual", None, None, "Finite DC iteration k", "Normalized projection residual"),
        ("ssim_vs_residual", "projection_residual", "ssim", "pareto_ssim", "pareto_union_ssim", "Normalized projection residual", "SSIM"),
    )
    for label, x_key, y_key, pareto_key, union_pareto_key, xlabel, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        uncertainty_drawn = False
        figure_ci: dict[str, dict[str, list[int]]] = {}
        figure_annotations: list[dict[str, Any]] = []
        for model in MODELS:
            rendering = _plot_model_trajectory(
                ax,
                by_model[model],
                model,
                x_key=x_key,
                y_key=y_key,
                pareto_key=pareto_key,
                union_pareto_key=union_pareto_key,
                selected_k=selected_k,
                figure_label=label,
                annotate_iterations=x_key == "projection_residual",
                operating_point_claim_allowed=operating_point_claim_allowed,
            )
            uncertainty_drawn = bool(rendering["uncertainty_drawn"]) or uncertainty_drawn
            figure_ci[model] = {
                "x_error_k": list(rendering["x_ci_k"]),
                "y_error_k": list(rendering["y_ci_k"]),
            }
            figure_annotations.extend(rendering["annotations"])
        if x_key == "projection_residual":
            for target in targets:
                ax.axvline(
                    _finite(target, "target_residual"),
                    color="#777777",
                    alpha=0.28,
                    linewidth=0.8,
                    linestyle="--",
                )
        else:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks(list(ITERATIONS))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x_key == "projection_residual":
            # Leave deterministic breathing room for the complete k-label
            # set; annotations do not otherwise participate in autoscaling.
            ax.margins(x=0.10, y=0.12)
        availability_label = (
            "Validation-locked residual targets shown"
            if matched_residual_available
            else MATCHED_RESIDUAL_UNAVAILABLE_LABEL
        )
        ax.set_title(
            f"{ANALYSIS_LABEL}\n{view}-view {ylabel} versus {xlabel.lower()}\n"
            + (
                f"Operating point: validation-selected k={selected_k} (claim-eligible)"
                if operating_point_claim_allowed
                else f"Operating point: diagnostic fallback k={selected_k} (claim-ineligible)"
            )
        )
        ax.text(
            0.01,
            0.01,
            availability_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        _style_axes(ax)
        _add_marker_legend(
            ax,
            selected_k,
            include_pareto=pareto_key is not None,
            include_uncertainty=uncertainty_drawn,
            operating_point_claim_allowed=operating_point_claim_allowed,
        )
        fig.tight_layout()
        path = output_dir / f"{label}_{view}view.svg"
        _save_svg(fig, path)
        plt.close(fig)
        outputs[label] = path
        uncertainty_by_figure[label] = uncertainty_drawn
        uncertainty_geometry[label] = {
            "architectures": figure_ci,
            "x_error_required": x_key == "projection_residual",
            "y_error_required": True,
            "rendered_at_k": [0, selected_k],
            "svg_gid_prefixes": [f"ci_x_{label}_", f"ci_y_{label}_"],
        }
        iteration_annotations[label] = {
            "drawn": x_key == "projection_residual",
            "count": len(figure_annotations),
            "expected_count": len(MODELS) * len(ITERATIONS) if x_key == "projection_residual" else 0,
            "k_grid": list(ITERATIONS) if x_key == "projection_residual" else [],
            "records": figure_annotations,
            "collision_avoidance": (
                "fixed architecture-specific point offsets with translucent white label backgrounds"
                if x_key == "projection_residual"
                else "not applicable"
            ),
        }

    operating_point_status = (
        "VALIDATION_SELECTED" if operating_point_claim_allowed else "DIAGNOSTIC_FALLBACK"
    )
    operating_point_label = (
        f"k={selected_k} (validation-selected; claim-eligible)"
        if operating_point_claim_allowed
        else f"k={selected_k} (diagnostic fallback; claim-ineligible)"
    )

    manifest: dict[str, Any] = {
        "schema": FIGURE_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "view": view,
        "models": list(MODELS),
        "k_grid": list(ITERATIONS),
        "validation_selected_k": selected_k if operating_point_claim_allowed else None,
        "diagnostic_fallback_k": (
            None if operating_point_claim_allowed else selected_k
        ),
        "operating_point": {
            "k": selected_k,
            "status": operating_point_status,
            "label": operating_point_label,
            "operating_point_claim_allowed": operating_point_claim_allowed,
            "overall_claim_allowed": overall_claim_allowed,
            "residual_matching_claim_allowed": residual_matching_claim_allowed,
            "source": "statistics manifest locked_spec",
            "marker": (
                "yellow black-edged star"
                if operating_point_claim_allowed
                else "gray black-edged star"
            ),
        },
        "analysis_label": ANALYSIS_LABEL,
        "interpretation": (
            "Empirical aggregate trajectories only; Pareto membership is descriptive and no model-specific operating point is selected. "
            + (
                "Validation-locked target guide lines are displayed; residual-matched inference remains governed by the authenticated statistics output."
                if matched_residual_available
                else MATCHED_RESIDUAL_UNAVAILABLE_LABEL
                + "; no residual-target guide lines or matched-residual values are displayed."
            )
        ),
        "residual_targets": {
            "status": target_status,
            "rows": len(targets),
            "matched_residual_inference_available": matched_residual_available,
            "guide_lines_drawn": matched_residual_available,
            "failure_code": authenticated["targets"].get("failure_code", ""),
            "failure_reason": authenticated["targets"].get("failure_reason", ""),
        },
        "uncertainty": {
            "display": "authenticated pointwise intervals at k=0 and the locked finite operating point; all point estimates retained",
            "method": "95% patient-cluster percentile bootstrap",
            "drawn_by_figure": uncertainty_by_figure,
            "geometry": uncertainty_geometry,
        },
        "iteration_annotations": iteration_annotations,
        "statistics_manifest": {
            "path": str(meta_path.resolve()),
            "sha256": sha256(meta_path),
            "schema": meta["schema"],
            "status": meta["status"],
            "lock_payload_sha256": locked.get("lock_payload_sha256"),
            "protocol_sha256": locked.get("protocol_sha256"),
        },
        "renderer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "inputs": authenticated,
        "marker_key": {
            "k0": "black-edged X",
            "operating_point": operating_point_label,
            "operating_point_marker": (
                "yellow black-edged star"
                if operating_point_claim_allowed
                else "gray black-edged star"
            ),
            "union_frontier": "filled model marker",
            "within_model_frontier_only": "open model marker",
            "dominated_within_model": "very light model marker",
        },
        "pareto_encoding": {
            "scope": "union of all three architecture trajectories at the test slice-weighted estimand",
            "filled": "pareto_union_psnr/pareto_union_ssim is True",
            "open": "within-model Pareto is True and union Pareto is False",
            "faint": "within-model Pareto is False",
            "claim": "descriptive geometry only; marker membership does not establish model superiority",
        },
        "outputs": {},
    }
    for label, path in outputs.items():
        manifest["outputs"][label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    manifest_path = output_dir / f"figure_manifest_{view}view.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", type=int, choices=VIEWS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for label, path in make_figures(args.stats_dir, args.out, view=args.view).items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
