#!/usr/bin/env python3
"""Patient-cluster sensitivity analysis for projection fidelity and DC.

This script is read-only with respect to the T9 evidence tree.  It computes
paired mean differences with (i) the legacy slice bootstrap, (ii) a patient
cluster bootstrap that preserves the slice-weighted estimand, and (iii) an
equal-patient estimand based on patient-specific mean differences.  The latter
two use the verified LoDoPaB test patient map (60 patients).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


BOOT = 10_000
SEED = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def slice_bootstrap(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    out = np.empty(BOOT)
    for b in range(BOOT):
        out[b] = d[rng.integers(0, len(d), len(d))].mean()
    return percentile_ci(out)


def cluster_bootstrap_pooled(d: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Resample patients, pool all slices, and retain the slice-weighted mean."""
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(SEED)
    out = np.empty(BOOT)
    for b in range(BOOT):
        draw = rng.integers(0, len(uniq), len(uniq))
        out[b] = d[np.concatenate([members[k] for k in draw])].mean()
    return percentile_ci(out)


def cluster_bootstrap_equal_patient(
    d: np.ndarray, groups: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Average within patient first, then bootstrap patients with equal weight."""
    uniq = np.unique(groups)
    patient_means = np.array([d[groups == g].mean() for g in uniq])
    rng = np.random.default_rng(SEED)
    out = np.empty(BOOT)
    for b in range(BOOT):
        out[b] = patient_means[rng.integers(0, len(uniq), len(uniq))].mean()
    lo, hi = percentile_ci(out)
    return float(patient_means.mean()), lo, hi, patient_means


def effects(d: np.ndarray) -> dict[str, float | int]:
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz)) if len(nz) else np.array([])
    rp = float(ranks[nz > 0].sum()) if len(nz) else float("nan")
    rn = float(ranks[nz < 0].sum()) if len(nz) else float("nan")
    rb = (rp - rn) / (rp + rn) if len(nz) else float("nan")
    w = wilcoxon(d)
    return {
        "median_diff": float(np.median(d)),
        "sd_diff": float(d.std(ddof=1)),
        "cohen_dz": float(d.mean() / d.std(ddof=1)),
        "rank_biserial": float(rb),
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "n_positive": int((d > 0).sum()),
        "n_negative": int((d < 0).sum()),
        "n_ties": int((d == 0).sum()),
        "proportion_positive": float((d > 0).mean()),
        "proportion_negative": float((d < 0).mean()),
    }


def patient_effects(patient_means: np.ndarray) -> dict[str, float | int]:
    """Paired inference with the patient, rather than the slice, as the unit."""
    nz = patient_means[patient_means != 0]
    ranks = rankdata(np.abs(nz)) if len(nz) else np.array([])
    rp = float(ranks[nz > 0].sum()) if len(nz) else float("nan")
    rn = float(ranks[nz < 0].sum()) if len(nz) else float("nan")
    rb = (rp - rn) / (rp + rn) if len(nz) else float("nan")
    w = wilcoxon(patient_means)
    return {
        "patient_median_diff": float(np.median(patient_means)),
        "patient_sd_diff": float(patient_means.std(ddof=1)),
        "patient_cohen_dz": float(
            patient_means.mean() / patient_means.std(ddof=1)
        ),
        "patient_rank_biserial": float(rb),
        "patient_wilcoxon_stat": float(w.statistic),
        "patient_wilcoxon_p": float(w.pvalue),
        "patient_proportion_positive": float((patient_means > 0).mean()),
        "patient_proportion_negative": float((patient_means < 0).mean()),
    }


def paired_from_long(
    path: Path, index: str, arm: str, value: str, arm_a: str, arm_b: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path)
    a = {int(r[index]): float(r[value]) for r in rows if r[arm] == arm_a}
    b = {int(r[index]): float(r[value]) for r in rows if r[arm] == arm_b}
    if set(a) != set(b):
        raise ValueError(f"index mismatch in {path}: {arm_a} vs {arm_b}")
    idx = np.array(sorted(a), dtype=int)
    return idx, np.array([b[i] - a[i] for i in idx], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t9", default="/Volumes/T9/wtransunet_project")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t9 = Path(args.t9)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pids_path = t9 / "lodopab/datasets/lodopab/patient_ids_rand_test.csv"
    abs_pids = np.loadtxt(pids_path, dtype=int)
    groups_all = abs_pids - abs_pids.min()
    if len(groups_all) != 3553 or len(np.unique(groups_all)) != 60:
        raise ValueError("unexpected LoDoPaB test patient mapping")

    root = t9 / "_returns/phase3_etri/results"
    specs = []
    for angle in (125, 50):
        proj = root / f"R2b_projection_fidelity/projection_residual_{angle}angle.csv"
        specs.append((f"projection_W_minus_Trans_{angle}", proj, "index", "model", "r", "TransUNet", "WavTransUNet"))
        specs.append((f"projection_W_minus_UNet_{angle}", proj, "index", "model", "r", "UNet", "WavTransUNet"))
        dc = root / f"R2b_dc_prototype/dc_test_{angle}angle.csv"
        for metric in ("r", "psnr"):
            specs.append((f"DC_after_minus_before_{metric}_{angle}", dc, "index", "model", metric, "WavTransUNet", "WavTransUNet_DC"))

    records = []
    fingerprints: dict[str, str] = {str(pids_path): sha256(pids_path)}
    for label, path, idx_col, arm_col, metric, arm_a, arm_b in specs:
        idx, d = paired_from_long(path, idx_col, arm_col, metric, arm_a, arm_b)
        if len(idx) != 3553 or not np.array_equal(idx, np.arange(3553)):
            raise ValueError(f"non-complete index in {path}")
        groups = groups_all[idx]
        fingerprints[str(path)] = sha256(path)
        slo, shi = slice_bootstrap(d)
        plo, phi = cluster_bootstrap_pooled(d, groups)
        ep_mean, elo, ehi, patient_means = cluster_bootstrap_equal_patient(d, groups)
        rec = {
            "contrast": label,
            "angle": int(label.rsplit("_", 1)[-1]),
            "metric": metric,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "n_slices": len(d),
            "n_patients": len(np.unique(groups)),
            "mean_diff_slice_weighted": float(d.mean()),
            "ci95_slice_lo": slo,
            "ci95_slice_hi": shi,
            "ci95_patient_cluster_pooled_lo": plo,
            "ci95_patient_cluster_pooled_hi": phi,
            "mean_diff_equal_patient": ep_mean,
            "ci95_equal_patient_lo": elo,
            "ci95_equal_patient_hi": ehi,
            "patient_mean_min": float(patient_means.min()),
            "patient_mean_max": float(patient_means.max()),
            "patients_positive": int((patient_means > 0).sum()),
            "patients_negative": int((patient_means < 0).sum()),
            **patient_effects(patient_means),
            **effects(d),
            "bootstrap": BOOT,
            "seed": SEED,
            "input_path": str(path),
        }
        records.append(rec)

    csv_path = out / "cluster_paired_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "description": "post-hoc patient-cluster sensitivity analysis",
        "estimands": {
            "patient_cluster_pooled": "resample patients and pool all selected slices; preserves slice-weighted mean",
            "equal_patient": "mean within patient, then average patients equally",
        },
        "bootstrap": BOOT,
        "seed": SEED,
        "patient_map": str(pids_path),
        "patient_id_offset": int(abs_pids.min()),
        "n_patients": int(len(np.unique(groups_all))),
        "patient_slice_count_min": int(np.bincount(groups_all).min()),
        "patient_slice_count_max": int(np.bincount(groups_all).max()),
        "input_sha256": fingerprints,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }
    (out / "cluster_paired_stats_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(csv_path)
    for rec in records:
        print(
            rec["contrast"],
            f'mean={rec["mean_diff_slice_weighted"]:.9g}',
            f'patient-cluster CI=[{rec["ci95_patient_cluster_pooled_lo"]:.9g}, {rec["ci95_patient_cluster_pooled_hi"]:.9g}]',
            f'equal-patient={rec["mean_diff_equal_patient"]:.9g}',
            f'CI=[{rec["ci95_equal_patient_lo"]:.9g}, {rec["ci95_equal_patient_hi"]:.9g}]',
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
