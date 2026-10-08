#!/usr/bin/env python3
"""Independent R3/G1 paired-statistics and cluster-sensitivity audit.

This script intentionally does not import the experiment's paired_stats.py.
It accepts either the combined long-format G1 table produced by
``build_g1_input.py`` or the two inference tables directly.  The prespecified
gate is evaluated only from the test-slice PSNR bootstrap.  Patient-cluster
analyses are reported as sensitivity analyses and never replace that gate.

Examples
--------
python3 r3_g1_independent_stats.py \
  --combined /path/to/E-G1/stats/g1_per_image.csv \
  --patient-ids /path/to/patient_ids_rand_test.csv \
  --out-json /tmp/r3_g1_independent_stats.json \
  --out-csv /tmp/r3_g1_independent_stats.csv

python3 r3_g1_independent_stats.py \
  --full /path/to/E-G1-full/infer/per_image_metrics.csv \
  --null /path/to/E-G1-null/infer/per_image_metrics.csv \
  --patient-ids /path/to/patient_ids_rand_test.csv \
  --out-json /tmp/r3_g1_independent_stats.json \
  --out-csv /tmp/r3_g1_independent_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


METRICS = ("psnr", "ssim", "rmse")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "rmse": False}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    required = {"angle", "index", "model", *METRICS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return rows


def arm_from_rows(rows: Iterable[dict], label: str, source_model: str) -> Dict[int, np.ndarray]:
    selected: Dict[int, np.ndarray] = {}
    angles = set()
    for row in rows:
        if row["model"] not in {label, source_model}:
            continue
        idx = int(row["index"])
        angle = int(float(row["angle"]))
        angles.add(angle)
        if idx in selected:
            raise ValueError(f"{label}: duplicate index {idx}")
        values = np.asarray([float(row[m]) for m in METRICS], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: non-finite metric at index {idx}: {values}")
        selected[idx] = values
    if not selected:
        raise ValueError(f"no rows found for arm {label!r} (source model {source_model!r})")
    if angles != {125}:
        raise ValueError(f"{label}: expected angle 125 only, got {sorted(angles)}")
    return selected


def load_arms(args: argparse.Namespace) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], dict]:
    provenance = {}
    if args.combined:
        path = Path(args.combined)
        rows = read_csv(path)
        full = arm_from_rows(rows, "G1_full", "__never__")
        null = arm_from_rows(rows, "G1_null", "__never__")
        provenance[str(path)] = sha256(path)
    else:
        full_path, null_path = Path(args.full), Path(args.null)
        full = arm_from_rows(read_csv(full_path), "G1_full", "WavResTransUNet")
        null = arm_from_rows(read_csv(null_path), "G1_null", "WavResTransUNet")
        provenance[str(full_path)] = sha256(full_path)
        provenance[str(null_path)] = sha256(null_path)
    return full, null, provenance


def percentile_ci_cases(d: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    """Reference-compatible case bootstrap, generated in bounded chunks."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    n = d.size
    chunk = 128
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        draw = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = d[draw].mean(axis=1)
    return tuple(float(x) for x in np.percentile(out, [2.5, 97.5]))


def cluster_summaries(d: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    counts = np.empty(uniq.size, dtype=np.int64)
    sums = np.empty(uniq.size, dtype=np.float64)
    means = np.empty(uniq.size, dtype=np.float64)
    for j, group in enumerate(uniq):
        x = d[groups == group]
        counts[j] = x.size
        sums[j] = x.sum(dtype=np.float64)
        means[j] = x.mean(dtype=np.float64)
    return counts, sums, means


def percentile_ci_clusters(
    d: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Return slice-weighted cluster CI, patient-equal CI, patient-equal mean.

    The first interval matches the experiment implementation: sample patients
    with replacement, pool all slices of sampled patients, then take the pooled
    slice mean.  The second interval samples the 60 patient means and gives each
    patient equal weight; it is an explicitly labelled alternative estimand.
    """
    counts, sums, patient_means = cluster_summaries(d, groups)
    rng = np.random.default_rng(seed)
    weighted = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, counts.size, counts.size)
        weighted[b] = sums[draw].sum() / counts[draw].sum()
    weighted_ci = tuple(float(x) for x in np.percentile(weighted, [2.5, 97.5]))

    rng = np.random.default_rng(seed)
    equal = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, patient_means.size, patient_means.size)
        equal[b] = patient_means[draw].mean()
    equal_ci = tuple(float(x) for x in np.percentile(equal, [2.5, 97.5]))
    return weighted_ci, equal_ci, float(patient_means.mean())


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Equivalent to scipy.stats.rankdata(x, method='average') for finite x."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def rank_biserial(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size == 0:
        return float("nan")
    ranks = average_ranks(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    return float((r_plus - r_minus) / (r_plus + r_minus))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def support_class(mean: float, ci: Sequence[float]) -> str:
    if ci[0] > 0:
        return "SUPPORT"
    if mean > 0 and ci[0] <= 0 <= ci[1]:
        return "INCONCLUSIVE"
    return "NO_SUPPORT"


def metric_result(
    metric: str,
    null: np.ndarray,
    full: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    d = full - null
    case_ci = percentile_ci_cases(d, n_boot=n_boot, seed=seed)
    cluster_ci, patient_equal_ci, patient_equal_mean = percentile_ci_clusters(
        d, groups, n_boot=n_boot, seed=seed
    )
    sd = float(d.std(ddof=1))
    dz = float(d.mean() / sd) if sd > 0 else float("nan")
    rb = rank_biserial(d)
    orient = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
    favorable = orient * d > 0
    adverse = orient * d < 0
    n_favorable, n_adverse = int(favorable.sum()), int(adverse.sum())
    n_ties = int((d == 0).sum())
    favor_ci = wilson_interval(n_favorable, d.size)
    result = {
        "metric": metric,
        "difference_definition": "full_minus_null",
        "higher_is_better": HIGHER_IS_BETTER[metric],
        "n": int(d.size),
        "null_mean": float(null.mean()),
        "full_mean": float(full.mean()),
        "mean_difference": float(d.mean()),
        "median_difference": float(np.median(d)),
        "sd_difference_ddof1": sd,
        "case_bootstrap_ci95": list(case_ci),
        "cluster_bootstrap_slice_weighted_ci95": list(cluster_ci),
        "patient_equal_mean_difference": patient_equal_mean,
        "patient_equal_bootstrap_ci95": list(patient_equal_ci),
        "cohen_dz_raw_full_minus_null": dz,
        "cohen_dz_oriented_favor_full": orient * dz,
        "rank_biserial_raw_full_minus_null": rb,
        "rank_biserial_oriented_favor_full": orient * rb,
        "n_favor_full": n_favorable,
        "n_favor_null": n_adverse,
        "n_ties": n_ties,
        "favor_full_proportion_all_slices": n_favorable / d.size,
        "favor_full_proportion_wilson_ci95": list(favor_ci),
        "favor_full_proportion_excluding_ties": (
            n_favorable / (n_favorable + n_adverse)
            if n_favorable + n_adverse
            else float("nan")
        ),
    }
    if metric == "psnr":
        result["prespecified_gate_from_case_bootstrap"] = support_class(float(d.mean()), case_ci)
        result["cluster_sensitivity_support_class"] = support_class(float(d.mean()), cluster_ci)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--combined", help="combined long CSV with G1_full/G1_null model labels")
    inputs.add_argument("--full", help="full-arm inference per_image_metrics.csv")
    parser.add_argument("--null", help="null-arm inference per_image_metrics.csv")
    parser.add_argument("--patient-ids", required=True,
                        help="LoDoPaB patient_ids_rand_test.csv in test-index order")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-n", type=int, default=3_553)
    args = parser.parse_args()
    if args.full and not args.null:
        parser.error("--full requires --null")
    if args.null and not args.full:
        parser.error("--null requires --full")
    if args.boot < 1:
        parser.error("--boot must be positive")

    full_map, null_map, provenance = load_arms(args)
    full_idx, null_idx = set(full_map), set(null_map)
    if full_idx != null_idx:
        raise ValueError(
            f"arm index sets differ: full-only={sorted(full_idx-null_idx)[:10]}, "
            f"null-only={sorted(null_idx-full_idx)[:10]}"
        )
    indexes = np.asarray(sorted(full_idx), dtype=np.int64)
    expected = np.arange(args.expected_n, dtype=np.int64)
    if not np.array_equal(indexes, expected):
        raise ValueError(
            f"expected exact indexes 0..{args.expected_n-1}; got n={indexes.size}, "
            f"first={indexes[:5].tolist()}, last={indexes[-5:].tolist()}"
        )
    full = np.stack([full_map[int(i)] for i in indexes])
    null = np.stack([null_map[int(i)] for i in indexes])

    patient_path = Path(args.patient_ids)
    patient_ids = np.loadtxt(patient_path, dtype=np.int64)
    if patient_ids.ndim != 1 or patient_ids.size != args.expected_n:
        raise ValueError(
            f"patient ID map must have {args.expected_n} scalar rows; got shape {patient_ids.shape}"
        )
    if np.unique(patient_ids).size != 60:
        raise ValueError(f"expected 60 test patients; got {np.unique(patient_ids).size}")
    provenance[str(patient_path)] = sha256(patient_path)

    results = [
        metric_result(metric, null[:, j], full[:, j], patient_ids, args.boot, args.seed)
        for j, metric in enumerate(METRICS)
    ]
    psnr = next(x for x in results if x["metric"] == "psnr")
    report = {
        "analysis": "R3/G1 independent paired audit",
        "primary_gate": "paired test-slice PSNR, full-null, 10000 percentile bootstrap resamples, seed 0",
        "gate_result": psnr["prespecified_gate_from_case_bootstrap"],
        "n_slices": int(indexes.size),
        "n_patients": int(np.unique(patient_ids).size),
        "bootstrap": int(args.boot),
        "seed": int(args.seed),
        "input_sha256": provenance,
        "metrics": results,
        "interpretation_note": (
            "The prespecified gate uses the case/slice bootstrap. Cluster-resampled "
            "intervals are sensitivity analyses and do not retroactively redefine the gate."
        ),
    }

    out_json, out_csv = Path(args.out_json), Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat_rows = []
    for row in results:
        flat = {k: v for k, v in row.items() if not isinstance(v, list)}
        for k, value in row.items():
            if isinstance(value, list) and len(value) == 2:
                flat[f"{k}_lo"], flat[f"{k}_hi"] = value
        flat_rows.append(flat)
    columns = list(flat_rows[0])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps({
        "gate_result": report["gate_result"],
        "n_slices": report["n_slices"],
        "n_patients": report["n_patients"],
        "psnr": psnr,
        "out_json": str(out_json),
        "out_csv": str(out_csv),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
