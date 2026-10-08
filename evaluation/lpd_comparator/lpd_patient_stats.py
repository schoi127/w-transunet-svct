#!/usr/bin/env python3
"""Frozen patient-aware comparison of returned sparse-view LPD outputs.

The script never trains or infers.  It joins complete per-slice CSVs by index,
computes W-TransUNet-minus-LPD effects, and writes slice-weighted, patient-
cluster, and equal-patient uncertainty summaries.  PSNR is primary; SSIM,
RMSE, and the matched-frame projection contrast are secondary.  Native LPD
projection residual is summarized separately and is not contrasted with the
composite post-processing outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


N = 3_553
N_PATIENTS = 60
BOOT = 10_000
SEED = 0
OFFICIAL_PATIENT_MAP_SHA256 = (
    "76a6db69773d78f6578a520d509c23f572541cdda6a4009317379b79132180a4"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def indexed(
    path: Path,
    metric: str,
    model: str,
    views: int,
    convention: str | None = None,
):
    selected = []
    for r in rows(path):
        if r["model"] != model:
            continue
        if convention is not None and r.get("convention") != convention:
            continue
        selected.append(r)
    if any(int(float(r["angle"])) != views for r in selected):
        raise ValueError(f"{path}: row angle does not match --views {views}")
    out = {int(r["index"]): float(r[metric]) for r in selected}
    if set(out) != set(range(N)) or len(selected) != N:
        raise ValueError(
            f"{path}: expected exactly indices 0..{N-1} once for {model}/{convention}"
        )
    a = np.array([out[i] for i in range(N)], dtype=np.float64)
    if not np.isfinite(a).all():
        raise FloatingPointError(f"nonfinite value in {path}")
    return a


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))


def summarize(diff: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == p) for p in uniq]
    patient_means = np.array([diff[i].mean() for i in members])

    rng = np.random.default_rng(SEED)
    slice_reps = np.empty(BOOT)
    for b in range(BOOT):
        slice_reps[b] = diff[rng.integers(0, N, N)].mean()

    rng = np.random.default_rng(SEED)
    cluster_reps = np.empty(BOOT)
    for b in range(BOOT):
        draw = rng.integers(0, N_PATIENTS, N_PATIENTS)
        cluster_reps[b] = diff[np.concatenate([members[i] for i in draw])].mean()

    rng = np.random.default_rng(SEED)
    equal_reps = np.empty(BOOT)
    for b in range(BOOT):
        equal_reps[b] = patient_means[
            rng.integers(0, N_PATIENTS, N_PATIENTS)
        ].mean()

    sd = diff.std(ddof=1)
    patient_sd = patient_means.std(ddof=1)
    slo, shi = ci(slice_reps)
    clo, chi = ci(cluster_reps)
    elo, ehi = ci(equal_reps)
    return {
        "n_slices": N,
        "n_patients": N_PATIENTS,
        "mean_diff_slice_weighted": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "sd_diff": float(sd),
        "cohen_dz_slice": float(diff.mean() / sd) if sd else float("nan"),
        "proportion_positive_slices": float((diff > 0).mean()),
        "proportion_negative_slices": float((diff < 0).mean()),
        "ci95_slice_lo": slo,
        "ci95_slice_hi": shi,
        "ci95_patient_cluster_lo": clo,
        "ci95_patient_cluster_hi": chi,
        "mean_diff_equal_patient": float(patient_means.mean()),
        "ci95_equal_patient_lo": elo,
        "ci95_equal_patient_hi": ehi,
        "patient_median_diff": float(np.median(patient_means)),
        "patient_cohen_dz": float(patient_means.mean() / patient_sd)
        if patient_sd
        else float("nan"),
        "patients_positive": int((patient_means > 0).sum()),
        "patients_negative": int((patient_means < 0).sum()),
        "patients_tied": int((patient_means == 0).sum()),
        "bootstrap": BOOT,
        "seed": SEED,
    }


def summarize_absolute(values: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    """Descriptive patient-aware summary for one method, not a contrast."""
    s = summarize(values, groups)
    return {
        "n_slices": s["n_slices"],
        "n_patients": s["n_patients"],
        "mean_slice_weighted": s["mean_diff_slice_weighted"],
        "median": s["median_diff"],
        "sd": s["sd_diff"],
        "ci95_slice_lo": s["ci95_slice_lo"],
        "ci95_slice_hi": s["ci95_slice_hi"],
        "ci95_patient_cluster_lo": s["ci95_patient_cluster_lo"],
        "ci95_patient_cluster_hi": s["ci95_patient_cluster_hi"],
        "mean_equal_patient": s["mean_diff_equal_patient"],
        "ci95_equal_patient_lo": s["ci95_equal_patient_lo"],
        "ci95_equal_patient_hi": s["ci95_equal_patient_hi"],
        "bootstrap": s["bootstrap"],
        "seed": s["seed"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, choices=(125, 50), required=True)
    ap.add_argument("--lpd-metrics", type=Path, required=True)
    ap.add_argument("--lpd-projection", type=Path, required=True)
    ap.add_argument("--lpd-manifest", type=Path, required=True)
    ap.add_argument("--w-metrics", type=Path, required=True)
    ap.add_argument("--w-projection", type=Path, required=True)
    ap.add_argument("--patient-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    inputs = [
        a.lpd_metrics,
        a.lpd_projection,
        a.lpd_manifest,
        a.w_metrics,
        a.w_projection,
        a.patient_map,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    lpd_manifest = json.loads(a.lpd_manifest.read_text(encoding="utf-8"))
    if lpd_manifest.get("status") != "COMPLETE" or lpd_manifest.get("views") != a.views:
        raise ValueError("LPD inference manifest is not COMPLETE for the requested view")
    if lpd_manifest.get("per_image_metrics_sha256") != sha256(a.lpd_metrics):
        raise ValueError("LPD metric CSV hash does not match its inference manifest")
    if lpd_manifest.get("projection_residual_sha256") != sha256(a.lpd_projection):
        raise ValueError("LPD projection CSV hash does not match its inference manifest")
    if sha256(a.patient_map) != OFFICIAL_PATIENT_MAP_SHA256:
        raise ValueError("patient map is not the byte-locked official LoDoPaB test map")
    pids = np.loadtxt(a.patient_map, dtype=int)
    if pids.shape != (N,) or np.unique(pids).size != N_PATIENTS:
        raise ValueError("patient map must contain 3,553 slices from 60 patients")
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise FileExistsError("statistics output directory must be empty")

    records = []
    for metric in ("psnr", "ssim", "rmse"):
        w = indexed(a.w_metrics, metric, "WavResTransUNet", a.views)
        lpd = indexed(a.lpd_metrics, metric, "LearnedPD", a.views)
        rec = {
            "views": a.views,
            "endpoint": metric,
            "contrast": "WavResTransUNet_minus_LearnedPD",
            "direction": "positive favors W" if metric in ("psnr", "ssim") else "negative favors W",
            **summarize(w - lpd, pids),
        }
        records.append(rec)

    w_r = indexed(a.w_projection, "r", "WavTransUNet", a.views, "fbp")
    lpd_composite_r = indexed(
        a.lpd_projection,
        "r",
        "LearnedPD",
        a.views,
        "fbp_frame_primary_cross_method",
    )
    records.append(
        {
            "views": a.views,
            "endpoint": "projection_r_fbp_frame_primary_cross_method",
            "contrast": "WavTransUNet_minus_LearnedPD",
            "direction": "negative favors W",
            **summarize(w_r - lpd_composite_r, pids),
        }
    )
    lpd_native_r = indexed(
        a.lpd_projection,
        "r",
        "LearnedPD",
        a.views,
        "native362_intrinsic_secondary",
    )
    native_summary = {
        "status": "COMPLETE",
        "views": a.views,
        "endpoint": "LearnedPD native-362 normalized projection residual",
        "interpretation": "method-intrinsic descriptive endpoint; not directly ranked against FBP-frame composite post-processors",
        **summarize_absolute(lpd_native_r, pids),
    }

    out_csv = a.out / f"lpd_patient_stats_{a.views}view.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    native_path = a.out / f"lpd_native_projection_{a.views}view.json"
    with native_path.open("w", encoding="utf-8") as f:
        json.dump(native_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    meta = {
        "status": "COMPLETE",
        "views": a.views,
        "primary_endpoint": "PSNR, patient-cluster 95% CI",
        "secondary_endpoints": [
            "SSIM",
            "RMSE",
            "projection residual fbp_frame_primary_cross_method",
            "LearnedPD native362_intrinsic_secondary (descriptive only)",
        ],
        "estimand": "slice-weighted mean paired difference",
        "bootstrap": BOOT,
        "seed": SEED,
        "input_sha256": {str(p.resolve()): sha256(p) for p in inputs},
        "output": str(out_csv.resolve()),
        "output_sha256": sha256(out_csv),
        "native_projection_output": str(native_path.resolve()),
        "native_projection_output_sha256": sha256(native_path),
        "numpy": np.__version__,
    }
    with (a.out / f"lpd_patient_stats_{a.views}view_meta.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
