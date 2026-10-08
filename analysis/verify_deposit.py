#!/usr/bin/env python3
"""Self-contained verification of the W-TransUNet reproducibility deposit.

This script re-derives the paper's headline numbers **from the deposited per-case
and per-slice records only**.  It needs no GPU, no LoDoPaB-CT data, no trained
checkpoint, no external drive and no network.  Everything it reads lives inside
this bundle.

What it recomputes
------------------
1.  Operating points of current Table 3 and Section 3.3 (read from the
    deposited legacy Table 2 file).  Slice-weighted mean PSNR and mean projection
    residual for U-Net, TransUNet and W-TransUNet at ``dc_iteration = 0`` and at
    the validation-locked K (K = 4 at 125 views, K = 1 at 50 views), directly
    from ``derived_results/dc_test_per_case/test_case_metrics_{125,50}view.csv``
    (74,613 rows each).  The Learned Primal-Dual row is recomputed from the
    independent A2 per-slice deposit.

2.  Paired W-TransUNet-versus-LPD contrasts of current Table 4 and Fig. 6
    (deposited legacy Table 5 file), from
    ``derived_results/secondary_analyses/A2_wdc_vs_lpd_{125,50}/per_slice_*.csv``.
    Differences are re-formed from the component PSNR columns rather than read
    from the stored delta column (the stored column is checked for consistency
    afterwards).  The interval is the patient-cluster percentile bootstrap:
    10,000 resamples, ``numpy.random.default_rng(0)``, each resample drawing 60
    patients with replacement and pooling *every* slice of each drawn patient,
    then taking the pooled slice mean; the 2.5th and 97.5th percentiles of the
    10,000 replicate means are the interval.  This is the procedure of
    ``analysis/_lib/a_common.py::summarize`` (itself copied verbatim from the
    remote ``lpd_patient_stats.py``), reimplemented here independently.

3.  Directional-input result of current Section 3.7 and Fig. 7 (deposited
    legacy Table 6 file), from ``derived_results/g1_ablation/
    g1_per_image.csv`` joined by index to ``configs/patient_ids_rand_test.csv``.
    The prespecified gate is the **case-level (slice) bootstrap**: 10,000
    resamples of 3,553 slices with replacement, ``default_rng(0)``, drawn in
    chunks of 128 replicates exactly as the remote audit program did, because
    the chunking fixes the random stream.  The patient-cluster interval is the
    declared sensitivity analysis and never replaces the gate.
4.  Table 1 of the 2026-08-30 backbone-rewrite manuscript.  The mean and
    population standard deviation (ddof = 0, the convention of the evaluation
    harness's own summary files, asserted from the deposited table's ``sd_ddof``
    column rather than assumed) of PSNR, SSIM and RMSE for the sparse-view FBP
    input and the three networks at 1000, 500, 250, 125 and 50 views, and the two paired
    W-TransUNet-minus-baseline PSNR contrasts per view count with their
    patient-cluster percentile intervals, Wilcoxon signed-rank p values on the 60
    patient means and win counts, all recomputed from
    ``derived_results/table1_per_image/<views>/metrics_per_test.csv`` (3,553 rows
    each) and checked against ``derived_results/table1_recomputed_means.csv`` and
    ``derived_results/table1_paired_contrasts.csv``.  The interval procedure is the
    same patient-cluster percentile bootstrap as in 2, evaluated as sum over count
    (``cluster_bootstrap_ci_pooled``), 10,000 replicates, ``default_rng(0)``.

Paths
-----
Every path is resolved from a single root.  By default the root is inferred from
this file's own location (``<root>/code/local_postprocessing/verify_deposit.py``
=> root is two directories up).  Set ``WTRANSUNET_RELEASE_ROOT`` to override.
No absolute path appears anywhere in this file.

Exit status
-----------
0 if every check PASSes, 1 if any check FAILs, 2 if an input is missing.

Usage
-----
    python3 code/local_postprocessing/verify_deposit.py
    WTRANSUNET_RELEASE_ROOT=/path/to/bundle python3 .../verify_deposit.py
    python3 .../verify_deposit.py --quick     # skip the three bootstraps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# constants -- all of them either protocol facts or values reported in the
# submitted manuscript.  Nothing here is a computation result.
# --------------------------------------------------------------------------- #
N_SLICES = 3553
N_PATIENTS = 60
N_SWEEP_ROWS = 74613          # 3553 slices x 3 architectures x 7 k values
BOOT = 10000
SEED = 0
CHUNK = 128                   # replicate-chunk size of the remote G1 case bootstrap
LOCK_K = {125: 4, 50: 1}
ARCH_LABEL = {"UNet": "U-Net", "TransUNet": "TransUNet", "WavTransUNet": "W-TransUNet"}
OFFICIAL_PATIENT_MAP_SHA256 = (
    "76a6db69773d78f6578a520d509c23f572541cdda6a4009317379b79132180a4"
)

# Values as printed in the submitted manuscript body (3 decimal places).
MANUSCRIPT_CONTRASTS = {
    (125, "bareW"): (-0.139, -0.190, -0.103),
    (125, "WDC"):   (-0.194, -0.242, -0.154),
    (50, "bareW"):  (-0.300, -0.404, -0.224),
    (50, "WDC"):    (-0.370, -0.466, -0.298),
}
# G1, as printed in the manuscript and in Section 3.7 of the current manuscript.
MANUSCRIPT_G1 = {
    "mean": -0.0049,
    "case_ci": (-0.0079, -0.0018),
    "cluster_ci": (-0.0098, -0.0004),
    "wins_full": 1812,
    "wins_null": 1741,
}

CANNOT_VERIFY = [
    ("The sparse-view FBP input residuals of Section 3.3 -- 125 views: PSNR 14.9372, "
     "SSIM 0.11843, RMSE 0.12550, projection residual 0.044290; 50 views: 10.8896, "
     "0.05568, 0.19935, 0.114542",
     "their per-image records belong to an intermediate archive tree that is not "
     "redistributed (INPUT_EVIDENCE/{125,50}view/per_image_metrics.csv and "
     "INPUT_EVIDENCE/{125,50}view/projection_residual_{views}angle.csv), and nothing "
     "inside this deposit carries them: dc_test_per_case/ holds only the UNet, "
     "TransUNet and WavTransUNet architectures, and secondary_analyses/A4_seven_method_* "
     "holds only the seven compared methods with no FBP-input row. The two rows appear "
     "in the deposited operating-point table only as a transcription of the reported "
     "table and are NOT independently checked here"),
    ("The image and sinogram panels of Fig. 1(b), Fig. 2 and Fig. 4, and the "
     "model-forward profile of Table 2",
     "these require the raw reconstruction arrays and the trained checkpoints. The "
     "~1.27 GB checkpoints are not bundled and LoDoPaB-CT is not redistributed, so no "
     "image-domain or sinogram-domain visual claim can be re-derived here, and neither "
     "can any parameter count, FLOP count, peak-memory or latency figure. The Fig. 1(a) "
     "schematic is drawn rather than measured and falls under the same exclusion"),
    ("The validation sweep that produced the locked constants (K, lambda, eta)",
     "its raw per-case table, validation_case_metrics.csv, was not retained and survives "
     "only as its SHA-256 digest in the run manifests and as its aggregated derivatives, "
     "so the selection cannot be re-executed from this deposit; this script can only "
     "confirm that the deposited test rows carry validation_selected=True at the locked K"),
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class Report:
    """Collects PASS/FAIL rows and prints them as one table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, str, bool]] = []

    def check(self, section: str, name: str, got: float, want: float, tol: float) -> bool:
        diff = abs(float(got) - float(want))
        ok = diff <= tol
        self.rows.append(
            (section, name, f"{got:.10g}", f"{want:.10g}", f"{diff:.3e}", ok)
        )
        return ok

    def note(self, section: str, name: str, text: str, ok: bool) -> bool:
        self.rows.append((section, name, text, text if ok else "expected", "-", ok))
        return ok

    @property
    def failures(self) -> int:
        return sum(1 for r in self.rows if not r[5])

    def render(self) -> None:
        w = [max(len(str(r[i])) for r in self.rows + [("SECTION", "CHECK", "RECOMPUTED",
                                                       "REPORTED", "ABS DIFF", True)])
             for i in range(5)]
        head = ("SECTION", "CHECK", "RECOMPUTED", "REPORTED", "ABS DIFF", "RESULT")
        line = "  ".join(head[i].ljust(w[i]) for i in range(5)) + "  " + head[5]
        print(line)
        print("-" * len(line))
        last = None
        for sec, name, got, want, diff, ok in self.rows:
            shown = sec if sec != last else ""
            last = sec
            print("  ".join(
                [shown.ljust(w[0]), name.ljust(w[1]), got.ljust(w[2]),
                 want.ljust(w[3]), diff.ljust(w[4])]
            ) + "  " + ("PASS" if ok else "FAIL"))


def tol_for(reported: str | float, floor: float = 1e-12) -> float:
    """Half a unit in the last reported decimal place."""
    d = Decimal(str(reported))
    exp = -d.as_tuple().exponent
    return max(0.5 * 10.0 ** (-exp), floor)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(path: Path) -> Path:
    if not path.exists():
        print(f"MISSING INPUT: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path


# --------------------------------------------------------------------------- #
# bootstrap procedures (reimplemented, not imported)
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(diff: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Patient-cluster percentile bootstrap, slice-weighted.

    Draw 60 patients with replacement, pool every slice of each drawn patient,
    take the pooled slice mean.  10,000 replicates, default_rng(0).
    """
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == p) for p in uniq]
    rng = np.random.default_rng(SEED)
    reps = np.empty(BOOT)
    for b in range(BOOT):
        draw = rng.integers(0, uniq.size, uniq.size)
        reps[b] = diff[np.concatenate([members[i] for i in draw])].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def case_bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    """Case/slice percentile bootstrap in 128-replicate chunks, default_rng(0)."""
    rng = np.random.default_rng(SEED)
    out = np.empty(BOOT)
    n = diff.size
    for start in range(0, BOOT, CHUNK):
        stop = min(start + CHUNK, BOOT)
        draw = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = diff[draw].mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap_ci_pooled(diff: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Same estimand as cluster_bootstrap_ci, computed as sum/count as the remote
    G1 audit program did.  Kept separate so each reported interval is reproduced
    by the routine that produced it."""
    uniq = np.unique(groups)
    counts = np.array([(groups == p).sum() for p in uniq], dtype=np.int64)
    sums = np.array([diff[groups == p].sum(dtype=np.float64) for p in uniq])
    rng = np.random.default_rng(SEED)
    reps = np.empty(BOOT)
    for b in range(BOOT):
        draw = rng.integers(0, counts.size, counts.size)
        reps[b] = sums[draw].sum() / counts[draw].sum()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# section 1 -- Table 2 operating points
# --------------------------------------------------------------------------- #
def load_sweep(path: Path, views: int) -> dict:
    """Return {(arch, k): {'psnr': array, 'residual': array}} ordered by case_id."""
    acc: dict[tuple[str, int], dict[int, tuple[float, float, int]]] = {}
    n_rows = 0
    selected_k: set[int] = set()
    diverged = 0
    patients: dict[int, int] = {}
    for r in read_rows(path):
        n_rows += 1
        if int(r["views"]) != views or r["split"] != "test":
            raise ValueError(f"unexpected views/split row in {path.name}")
        if r["diverged"] != "False":
            diverged += 1
        key = (r["architecture"], int(r["dc_iteration"]))
        if r["validation_selected"] == "True":
            selected_k.add(int(r["dc_iteration"]))
        cid = int(r["case_id"])
        patients[cid] = int(r["patient_id"])
        acc.setdefault(key, {})[cid] = (
            float(r["psnr"]), float(r["projection_residual"]), int(r["patient_id"])
        )
    out = {"n_rows": n_rows, "selected_k": sorted(selected_k), "diverged": diverged,
           "n_cases": len(patients), "n_patients": len(set(patients.values())),
           "arms": {}}
    for key, d in acc.items():
        if set(d) != set(range(N_SLICES)):
            raise ValueError(f"{key}: case ids are not exactly 0..{N_SLICES - 1}")
        arr = np.array([d[i] for i in range(N_SLICES)], dtype=np.float64)
        out["arms"][key] = {"psnr": arr[:, 0], "residual": arr[:, 1]}
    return out


def parse_table2(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    return {(int(r["Views"]), r["Method"]): r for r in read_rows(path)}


def section_table2(root: Path, rep: Report, coverage: dict) -> dict:
    print("[1/4] Operating points of current Table 3 and Section 3.3 -- "
          "recomputing from the 74,613-row sweep CSVs ...")
    t2_path = require(root / "derived_results" / "figure_data"
                      / "Table2_test_operating_points__table2.csv")
    t2 = parse_table2(t2_path)
    coverage["table2_rows_total"] = len(t2)
    coverage["table2_rows_unverifiable"] = sorted(
        f"{views}v {method}" for (views, method) in t2 if method == "Sparse-view FBP input"
    )
    per_slice: dict[int, dict[str, np.ndarray]] = {}
    for views in (125, 50):
        p = require(root / "derived_results" / "dc_test_per_case"
                    / f"test_case_metrics_{views}view.csv")
        sw = load_sweep(p, views)
        sec = f"T3A/{views}v"
        rep.note(sec, "sweep row count", f"{sw['n_rows']}", sw["n_rows"] == N_SWEEP_ROWS)
        rep.note(sec, "distinct case_id", f"{sw['n_cases']}", sw["n_cases"] == N_SLICES)
        rep.note(sec, "distinct patient_id", f"{sw['n_patients']}",
                 sw["n_patients"] == N_PATIENTS)
        rep.note(sec, "rows flagged diverged", f"{sw['diverged']}", sw["diverged"] == 0)
        rep.note(sec, "validation_selected k", f"{sw['selected_k']}",
                 sw["selected_k"] == [LOCK_K[views]])

        for arch, label in ARCH_LABEL.items():
            for k, suffix in ((0, ""), (LOCK_K[views], "+DC")):
                arm = sw["arms"][(arch, k)]
                row = t2[(views, label + suffix)]
                for field, col in (("psnr", "PSNR (dB)"),
                                   ("residual", "Projection residual")):
                    got = float(arm[field].mean())
                    want = row[col]
                    rep.check(sec, f"{label}{suffix} k={k} mean {field}",
                              got, float(want), tol_for(want))
                coverage.setdefault("table2_rows_verified", set()).add(
                    (views, label + suffix)
                )
    return per_slice


# --------------------------------------------------------------------------- #
# section 2 -- paired W vs LPD contrasts
# --------------------------------------------------------------------------- #
def parse_table5(path: Path) -> dict[int, dict[str, str]]:
    return {int(r["Views"]): r for r in read_rows(path)}


def parse_ci_cell(cell: str) -> tuple[float, float, float]:
    """'-0.1389 [-0.1902, -0.1029]' -> (mean, lo, hi) plus the raw strings."""
    point, rest = cell.split("[", 1)
    lo, hi = rest.rstrip("]").split(",")
    return float(point.strip()), float(lo.strip()), float(hi.strip())


def raw_ci_cell(cell: str) -> tuple[str, str, str]:
    point, rest = cell.split("[", 1)
    lo, hi = rest.rstrip("]").split(",")
    return point.strip(), lo.strip(), hi.strip()


def section_contrasts(root: Path, rep: Report, quick: bool, coverage: dict) -> None:
    print("[2/4] Paired W-TransUNet vs LPD contrasts of current Table 4 "
          "and Fig. 6 -- "
          "patient-cluster bootstrap, 10,000 resamples, seed 0 ...")
    t2_path = root / "derived_results" / "figure_data" / \
        "Table2_test_operating_points__table2.csv"
    t2 = parse_table2(t2_path)
    t5_path = require(root / "derived_results" / "figure_data"
                      / "Table5_lpd_cross_class_comparison__table5.csv")
    t5 = parse_table5(t5_path)

    for views in (125, 50):
        k = LOCK_K[views]
        p = require(root / "derived_results" / "secondary_analyses"
                    / f"A2_wdc_vs_lpd_{views}" / f"per_slice_{views}.csv")
        rows = read_rows(p)
        sec = f"T3C/{views}v"
        if len(rows) != N_SLICES:
            rep.note(sec, "per-slice row count", f"{len(rows)}", False)
            continue
        rep.note(sec, "per-slice row count", f"{len(rows)}", True)
        cid = np.array([int(r["case_id"]) for r in rows])
        order = np.argsort(cid, kind="mergesort")
        rows = [rows[i] for i in order]
        pid = np.array([int(r["patient_id"]) for r in rows])
        rep.note(sec, "distinct patient_id", f"{np.unique(pid).size}",
                 np.unique(pid).size == N_PATIENTS)

        psnr_w0 = np.array([float(r["psnr_W_k0"]) for r in rows])
        psnr_wk = np.array([float(r[f"psnr_W_k{k}"]) for r in rows])
        psnr_lpd = np.array([float(r["psnr_LPD"]) for r in rows])
        res_lpd = np.array([float(r["residual_LPD"]) for r in rows])

        # LPD row of the deposited legacy Table 2 file, recomputed from this
        # independent deposit
        lpd_row = t2[(views, "Learned Primal–Dual")]
        rep.check(f"T3A/{views}v", "LPD mean psnr (from A2 per-slice)",
                  float(psnr_lpd.mean()), float(lpd_row["PSNR (dB)"]),
                  tol_for(lpd_row["PSNR (dB)"]))
        rep.check(f"T3A/{views}v", "LPD mean residual (from A2 per-slice)",
                  float(res_lpd.mean()), float(lpd_row["Projection residual"]),
                  tol_for(lpd_row["Projection residual"]))
        coverage.setdefault("table2_rows_verified", set()).add(
            (views, "Learned Primal\u2013Dual")
        )

        # differences re-formed from components, then cross-checked against the
        # stored delta columns
        d_bare = psnr_w0 - psnr_lpd
        d_wdc = psnr_wk - psnr_lpd
        stored_bare = np.array([float(r["dpsnr_bareW"]) for r in rows])
        stored_wdc = np.array([float(r["dpsnr_WDC"]) for r in rows])
        rep.check(sec, "max|recomputed-stored| dpsnr_bareW",
                  float(np.abs(d_bare - stored_bare).max()), 0.0, 1e-12)
        rep.check(sec, "max|recomputed-stored| dpsnr_WDC",
                  float(np.abs(d_wdc - stored_wdc).max()), 0.0, 1e-12)

        for tag, diff in (("bareW", d_bare), ("WDC", d_wdc)):
            mean = float(diff.mean())
            if quick:
                lo = hi = float("nan")
            else:
                lo, hi = cluster_bootstrap_ci(diff, pid)

            # (a) against the manuscript body values
            m_mean, m_lo, m_hi = MANUSCRIPT_CONTRASTS[(views, tag)]
            rep.check(sec, f"{tag}-LPD mean vs manuscript", mean, m_mean, tol_for(m_mean))
            if not quick:
                rep.check(sec, f"{tag}-LPD CI lo vs manuscript", lo, m_lo, tol_for(m_lo))
                rep.check(sec, f"{tag}-LPD CI hi vs manuscript", hi, m_hi, tol_for(m_hi))

            # (b) against the deposited legacy Table 5 cell, at its own precision
            col = ("W − LPD PSNR (95% CI) [frozen endpoint]" if tag == "bareW"
                   else "W+DC − LPD PSNR (95% CI) [post hoc]")
            s_mean, s_lo, s_hi = raw_ci_cell(t5[views][col])
            rep.check(sec, f"{tag}-LPD mean vs deposited legacy Table 5 (current Table 4)", mean, float(s_mean), tol_for(s_mean))
            if not quick:
                rep.check(sec, f"{tag}-LPD CI lo vs deposited legacy Table 5 (current Table 4)", lo, float(s_lo), tol_for(s_lo))
                rep.check(sec, f"{tag}-LPD CI hi vs deposited legacy Table 5 (current Table 4)", hi, float(s_hi), tol_for(s_hi))

            # (c) patient sign count reported by the deposited legacy Table 5
            pm = np.array([diff[pid == q].mean() for q in np.unique(pid)])
            npos = int((pm > 0).sum())
            ccol = ("Patients with positive mean W − LPD (of 60)" if tag == "bareW"
                    else "Patients with positive mean W+DC − LPD (of 60)")
            want = int(t5[views][ccol].split("/")[0])
            rep.check(sec, f"{tag} patients with positive mean", npos, want, 0)


# --------------------------------------------------------------------------- #
# section 3 -- G1 directional input
# --------------------------------------------------------------------------- #
def section_g1(root: Path, rep: Report, quick: bool) -> None:
    print("[3/4] Directional-input substitution of current Supplementary Table S3 and "
          "Fig. 7 -- "
          "case bootstrap (prespecified gate) and patient-cluster sensitivity ...")
    sec = "T3D/125v"
    per = require(root / "derived_results" / "g1_ablation" / "g1_per_image.csv")
    pmap = require(root / "configs" / "patient_ids_rand_test.csv")

    got_sha = sha256_file(pmap)
    rep.note(sec, "patient map sha256", got_sha[:16] + "...",
             got_sha == OFFICIAL_PATIENT_MAP_SHA256)
    pid = np.loadtxt(pmap, dtype=int)
    rep.note(sec, "patient map shape", f"{pid.shape[0]} slices / "
             f"{np.unique(pid).size} patients",
             pid.shape == (N_SLICES,) and np.unique(pid).size == N_PATIENTS)

    arms: dict[str, dict[int, float]] = {"G1_full": {}, "G1_null": {}}
    for r in read_rows(per):
        if r["model"] not in arms:
            raise ValueError(f"unexpected model {r['model']!r}")
        if int(float(r["angle"])) != 125:
            raise ValueError("G1 deposit must be 125-view only")
        arms[r["model"]][int(r["index"])] = float(r["psnr"])
    ok = all(set(d) == set(range(N_SLICES)) for d in arms.values())
    rep.note(sec, "both arms cover indices 0..3552", "yes" if ok else "no", ok)
    if not ok:
        return

    full = np.array([arms["G1_full"][i] for i in range(N_SLICES)])
    null = np.array([arms["G1_null"][i] for i in range(N_SLICES)])
    diff = full - null

    rep.check(sec, "mean paired dPSNR (full-null)", float(diff.mean()),
              MANUSCRIPT_G1["mean"], tol_for(MANUSCRIPT_G1["mean"]))
    rep.check(sec, "wins, full arm", int((diff > 0).sum()),
              MANUSCRIPT_G1["wins_full"], 0)
    rep.check(sec, "wins, null arm", int((diff < 0).sum()),
              MANUSCRIPT_G1["wins_null"], 0)
    rep.check(sec, "wins sum to n", int((diff > 0).sum() + (diff < 0).sum()),
              N_SLICES, 0)

    if quick:
        return
    lo, hi = case_bootstrap_ci(diff)
    rep.check(sec, "case (slice) bootstrap CI lo [prespecified gate]",
              lo, MANUSCRIPT_G1["case_ci"][0], tol_for(MANUSCRIPT_G1["case_ci"][0]))
    rep.check(sec, "case (slice) bootstrap CI hi [prespecified gate]",
              hi, MANUSCRIPT_G1["case_ci"][1], tol_for(MANUSCRIPT_G1["case_ci"][1]))
    clo, chi = cluster_bootstrap_ci_pooled(diff, pid)
    rep.check(sec, "patient-cluster CI lo [sensitivity]",
              clo, MANUSCRIPT_G1["cluster_ci"][0], tol_for(MANUSCRIPT_G1["cluster_ci"][0]))
    rep.check(sec, "patient-cluster CI hi [sensitivity]",
              chi, MANUSCRIPT_G1["cluster_ci"][1], tol_for(MANUSCRIPT_G1["cluster_ci"][1]))
    rep.note(sec, "gate verdict from the case interval",
             "NO_SUPPORT" if not (lo > 0) else "SUPPORT", not (lo > 0))


# --------------------------------------------------------------------------- #
# section 4 -- Table 1, the five-view accuracy table of the backbone rewrite
#
# Added 2026-08-30.  Recomputes, from derived_results/table1_per_image/ alone,
# every mean and standard deviation printed in Table 1 and every paired PSNR
# contrast printed in its two right-hand columns, and checks them against the two
# aggregate tables deposited beside them.  The aggregate tables are the target of
# the comparison, never the source of the recomputation.
# --------------------------------------------------------------------------- #
T1_VIEWS = (1000, 500, 250, 125, 50)
T1_METHOD_COLUMN = {"FBP": "fbp", "U-Net": "unet",
                    "TransUNet": "transunet", "W-TransUNet": "wavtransunet"}
T1_CONTRAST_BASE = {"W-TransUNet minus TransUNet": "transunet",
                    "W-TransUNet minus U-Net": "unet"}


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks of ``a`` with ties given their average rank (scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="quicksort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilcoxon_two_sided(d: np.ndarray) -> tuple[float, str]:
    """Two-sided Wilcoxon signed-rank p value on the 60 patient means.

    Uses scipy when it is importable, because the deposited value was produced by
    ``scipy.stats.wilcoxon`` with its defaults (zero_method='wilcox', two-sided,
    asymptotic for n = 60, no continuity correction) and then agrees bit for bit.
    Falls back to the same asymptotic formula implemented on numpy alone, which
    reproduces the deposited values to a relative 1e-14, so the deposit still
    verifies on a machine that has no scipy.
    """
    try:
        from scipy.stats import wilcoxon as _w          # noqa: PLC0415
        return float(_w(d).pvalue), "scipy"
    except Exception:                                    # pragma: no cover
        pass
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = d.size
    r = _rankdata_average(np.abs(d))
    r_plus = float(np.sum(r[d > 0]))
    mean = n * (n + 1.0) * 0.25
    var = n * (n + 1.0) * (2.0 * n + 1.0)
    _, counts = np.unique(r, return_counts=True)
    var -= 0.5 * float((counts ** 3 - counts).sum())
    se = math.sqrt(var / 24.0)
    z = (r_plus - mean) / se
    return 2.0 * (0.5 * math.erfc(abs(z) / math.sqrt(2.0))), "numpy"


def section_table1(root: Path, rep: Report, quick: bool) -> None:
    print("[4/4] Table 1 five-view accuracy -- recomputing means, SDs and paired "
          "contrasts from the per-image tables ...")
    base = require(root / "derived_results" / "table1_per_image")
    means_path = require(root / "derived_results" / "table1_recomputed_means.csv")
    contrasts_path = require(root / "derived_results" / "table1_paired_contrasts.csv")
    pmap = require(root / "configs" / "patient_ids_rand_test.csv")

    means = {(int(r["views"]), r["method"]): r for r in read_rows(means_path)}
    contrasts = {(int(r["views"]), r["contrast"], r["metric"]): r
                 for r in read_rows(contrasts_path)}
    rep.note("T1", "aggregate rows in table1_recomputed_means.csv",
             f"{len(means)}", len(means) == 20)
    rep.note("T1", "rows in table1_paired_contrasts.csv",
             f"{len(contrasts)}", len(contrasts) == 30)
    # The SD convention is read from the file, never assumed.  Table 1 prints the
    # population standard deviation, which is what the evaluation harness's own
    # metrics_report.csv/.txt print; if the deposited table ever carried a different
    # ddof this check fails rather than silently comparing two different estimators.
    declared = {r.get("sd_ddof") for r in means.values()}
    rep.note("T1", "declared sd_ddof in table1_recomputed_means.csv",
             ",".join(sorted(str(d) for d in declared)), declared == {"0"})

    pid = np.loadtxt(pmap, dtype=int)
    order = np.unique(pid)
    groups = [np.flatnonzero(pid == p) for p in order]
    counts = np.array([g.size for g in groups], dtype=float)

    for views in T1_VIEWS:
        sec = f"T1/{views}v"
        per = require(base / str(views) / "metrics_per_test.csv")
        rows = read_rows(per)
        rep.note(sec, "per-image row count", f"{len(rows)}", len(rows) == N_SLICES)
        if len(rows) != N_SLICES:
            continue
        col = {name: np.array([float(r[name]) for r in rows])
               for name in rows[0] if name != "test_idx"}

        # (a) the 4 x 3 mean +/- SD cells of this view block
        for method, prefix in T1_METHOD_COLUMN.items():
            want = means[(views, method)]
            for metric in ("psnr", "ssim", "rmse"):
                v = col[f"{prefix}_{metric}"]
                rep.check(sec, f"{method} {metric} mean", float(v.mean()),
                          float(want[f"{metric}_mean"]),
                          tol_for(want[f"{metric}_mean"]))
                # population standard deviation, ddof = 0, as asserted above
                rep.check(sec, f"{method} {metric} SD", float(v.std(ddof=0)),
                          float(want[f"{metric}_sd"]), tol_for(want[f"{metric}_sd"]))

        # (b) the two paired PSNR contrasts of this view block
        for contrast, prefix in T1_CONTRAST_BASE.items():
            want = contrasts[(views, contrast, "PSNR")]
            diff = col["wavtransunet_psnr"] - col[f"{prefix}_psnr"]
            short = contrast.replace("W-TransUNet minus ", "W - ")
            rep.check(sec, f"{short} mean", float(diff.mean()),
                      float(want["mean_diff"]), tol_for(want["mean_diff"]))
            rep.check(sec, f"{short} slices favouring W",
                      int((diff > 0).sum()), int(want["slices_favoring_W"]), 0)
            patient_means = np.array([diff[g].mean() for g in groups])
            rep.check(sec, f"{short} patients favouring W",
                      int((patient_means > 0).sum()),
                      int(want["patients_favoring_W"]), 0)
            p_value, engine = wilcoxon_two_sided(patient_means)
            reported_p = float(want["wilcoxon_p_patient_means"])
            # relative tolerance: an absolute one is meaningless at p ~ 1e-11
            rep.check(sec, f"{short} Wilcoxon p ({engine})", p_value, reported_p,
                      max(abs(reported_p) * 1e-12, 1e-300))
            if quick:
                continue
            sums = np.array([diff[g].sum(dtype=np.float64) for g in groups])
            rng = np.random.default_rng(SEED)
            draws = rng.integers(0, order.size, size=(BOOT, order.size))
            reps = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            rep.check(sec, f"{short} CI lo", float(lo),
                      float(want["ci95_lo"]), tol_for(want["ci95_lo"]))
            rep.check(sec, f"{short} CI hi", float(hi),
                      float(want["ci95_hi"]), tol_for(want["ci95_hi"]))


# --------------------------------------------------------------------------- #
# optional section -- parity with the submitted artifacts, when they are present
# --------------------------------------------------------------------------- #
def section_submission_parity(root: Path, rep: Report) -> bool:
    """If the submission package sits next to the bundle, assert that every
    deposited table CSV is byte-identical to the submitted one.  Skipped (not
    failed) when the submission package is absent, so the bundle still verifies
    standalone."""
    sub = root.parent / "SUBMISSION" / "tables"
    # Only the submission package this deposit was built for can be compared
    # here: derived_results/figure_data/ carries that package's table
    # numbering.  A sibling SUBMISSION/ from a different manuscript is not a
    # parity target, and reporting its tables as missing would be a false
    # failure, so the section is skipped unless the marker directory is there.
    if not sub.is_dir() or not (sub / "Table5_lpd_cross_class_comparison").is_dir():
        return False
    print("[+]   Submitted tables found next to the bundle -- checking byte parity ...")
    fd = root / "derived_results" / "figure_data"
    for d in sorted(sub.iterdir()):
        if not d.is_dir():
            continue
        n = d.name.split("_", 1)[0].replace("Table", "")
        submitted = d / f"table{n}.csv"
        deposited = fd / f"{d.name}__table{n}.csv"
        if not submitted.exists() or not deposited.exists():
            rep.note("PARITY", f"Table {n} present in both", "no", False)
            continue
        a, b = sha256_file(deposited), sha256_file(submitted)
        rep.note("PARITY", f"Table {n} deposited == submitted", a[:16] + "...", a == b)
    return True


# --------------------------------------------------------------------------- #
def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).resolve()
    env = os.environ.get("WTRANSUNET_RELEASE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="bundle root (default: inferred from this file, or "
                         "$WTRANSUNET_RELEASE_ROOT)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the three bootstraps (means and counts only)")
    args = ap.parse_args()
    root = resolve_root(args.root)

    print("=" * 78)
    print("W-TransUNet reproducibility deposit -- independent verification")
    print("=" * 78)
    print(f"bundle root : {root}")
    print(f"python      : {sys.version.split()[0]}   numpy {np.__version__}")
    print("inputs      : deposited CSVs only -- no GPU, no checkpoints, "
          "no LoDoPaB-CT, no external drive")
    if args.quick:
        print("mode        : --quick (bootstrap intervals SKIPPED)")
    print()
    if not (root / "derived_results").is_dir():
        print(f"MISSING INPUT: {root/'derived_results'} is not a directory", file=sys.stderr)
        return 2

    rep = Report()
    coverage: dict = {}
    section_table2(root, rep, coverage)
    section_contrasts(root, rep, args.quick, coverage)
    section_g1(root, rep, args.quick)
    section_table1(root, rep, args.quick)
    parity = section_submission_parity(root, rep)

    print()
    rep.render()
    print()

    if not parity:
        print("SKIPPED: byte parity against the submitted tables -- the SUBMISSION "
              "package is not\n         beside this bundle. The deposited copies "
              "under derived_results/figure_data/\n         were verified byte-identical "
              "to the submitted tables when the bundle was built.")
        print()

    total = coverage.get("table2_rows_total")
    done = len(coverage.get("table2_rows_verified", ()))
    missing = coverage.get("table2_rows_unverifiable", [])
    if total:
        print("OPERATING-POINT COVERAGE (current Table S4 and Table 3; "
              "deposited legacy Table 2 file)")
        print("-" * 78)
        print(f"  {done} of {total} data rows of the deposited operating-point "
              f"table were independently recomputed above.")
        if missing:
            print(f"  {len(missing)} row(s) could NOT be recomputed from this deposit: "
                  + ", ".join(missing))
            print("  Those are the sparse-view FBP input rows. Their values exist in the "
                  "deposited table\n  only as a transcription of the submitted table; nothing "
                  "here checks them.\n  A PASS below does NOT cover them. See the next section "
                  "for why.")
        print()

    print("NOT VERIFIABLE FROM THIS DEPOSIT")
    print("-" * 78)
    for what, why in CANNOT_VERIFY:
        print(f"  * {what}")
        print(f"      reason: {why}")
    print()


    print("DEPOSITED VERBATIM RATHER THAN RECOMPUTED")
    print("-" * 78)
    print("  The operator-qualification diagnostics behind Supplementary Fig. S1(d) and")
    print("  Table S2 -- the relative adjoint-identity errors of the matched ASTRA-CPU and")
    print("  the executed ASTRA-CUDA pair, the s20 power estimate and the lock chronology")
    print("  -- are run-manifest measurements deposited verbatim under")
    print("  derived_results/figure_data/. They can be read straight from the deposit and")
    print("  the supplementary figure redrawn from them, but they are transcriptions, not")
    print("  statistics over per-case records, so this program recomputes no value of")
    print("  theirs and none of the checks above covers them.")
    print()
    total = len(rep.rows)
    bad = rep.failures
    print(f"{total - bad}/{total} checks PASS, {bad} FAIL")
    if bad:
        print("RESULT: FAIL -- a recomputed value disagrees with the reported value.")
        print("        Do not adjust this script to make it pass; report the "
              "discrepancy to the authors.")
        return 1
    print("RESULT: PASS -- every value this script recomputed agrees with the reported value "
          "to\n        within half a unit in its last reported decimal place. This is a "
          "statement about\n        the checks listed above only, not about the quantities "
          "listed as not verifiable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
