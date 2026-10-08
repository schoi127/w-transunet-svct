#!/usr/bin/env python3
"""Train or evaluate a frozen sparse-view DIVal Learned Primal-Dual comparator.

This runner is deliberately separate from the W-TransUNet code.  It uses the
official LoDoPaB split and DIVal angle-subset operator, records immutable input
fingerprints, keeps test inference unavailable from the ``train`` command, and
writes one row per official test slice during ``infer``.

The package is a pre-analysis execution aid, not evidence that the experiment
has been run.  A CUDA-capable ASTRA environment is required for train/infer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPECTED_LENGTHS = {"train": 35_820, "validation": 3_522, "test": 3_553}
VIEWS_ALLOWED = (125, 50)
IMAGE_CROP = slice(5, 357)
MODEL_LABEL = "LearnedPD"
OFFICIAL_HP_SHA256 = (
    "768c4e0ab1357d4704fb7e1ef267b7ca399be9ce5ab01ea9a24a64465ce1aaa8"
)
EXPECTED_FBP_CACHE_SHA256 = {
    125: "2ccc383a473d974d5c11a5e5370a96bc830401c7bcadee9c7329a5514a257a0a",
    50: "dbd2bdb8a0e0fe6c52ff7c70170430cb3b80a3c46095d52f6dbc3585d21bdf89",
}
FBP_CANARY_RTOL = 1e-5
FBP_CANARY_ATOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_state_sha256(model) -> str:
    """Hash tensor names, shapes, dtypes and values in a model state."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(a.shape).encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def require_empty_or_resume(out: Path, resume: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    occupied = [p for p in out.iterdir() if not p.name.startswith(".")]
    if occupied and not resume:
        raise FileExistsError(
            f"output directory is not empty: {out}; pass --resume only after audit"
        )


def add_common(ap: argparse.ArgumentParser) -> None:
    here = Path(__file__).resolve().parent
    ap.add_argument("--views", type=int, choices=VIEWS_ALLOWED, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dival-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--hyper-params",
        type=Path,
        default=here / "lodopab_learnedpd_reference_hyper_params.json",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="read-only assets/config/operator checks")
    add_common(p)
    p.add_argument("--impl", default="astra_cuda")

    p = sub.add_parser("train", help="train and validation-select; never touches test")
    add_common(p)
    p.add_argument("--impl", default="astra_cuda", choices=("astra_cuda",))
    p.add_argument("--epochs", type=int, help="pilot override; 10 for final run")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("infer", help="evaluate a frozen checkpoint on all test slices")
    add_common(p)
    p.add_argument("--impl", default="astra_cuda", choices=("astra_cuda",))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--fbp-cache", type=Path, required=True)
    p.add_argument("--save-reconstructions", action="store_true")
    p.add_argument("--resume", action="store_true")
    return ap.parse_args()


def validate_assets(args: argparse.Namespace) -> dict[str, object]:
    data = args.data.resolve()
    dival_root = args.dival_root.resolve()
    hp_path = args.hyper_params.resolve()
    if not data.is_dir() or not dival_root.is_dir() or not hp_path.is_file():
        raise FileNotFoundError("data, dival-root, or hyper-parameter file is missing")
    if sha256(hp_path) != OFFICIAL_HP_SHA256:
        raise ValueError("official hyper-parameter JSON hash mismatch")
    hp = json.loads(hp_path.read_text(encoding="utf-8"))
    required_hp = {
        "batch_size": 1,
        "epochs": 10,
        "niter": 10,
        "internal_ch": 64,
        "lr": 0.0001,
        "lr_min": 0.0001,
        "init_fbp": True,
        "init_frequency_scaling": 0.7,
    }
    if hp != required_hp:
        raise ValueError(f"unexpected reference hyper-parameters: {hp}")

    last_ids = {"train": 279, "validation": 27, "test": 27}
    evidence: dict[str, str] = {}
    # No command reads test observations or ground truth before infer() has
    # authenticated the final training manifest and checkpoint.  The infer
    # command fingerprints its test assets only after that gate.
    # LoDoPaBDataset itself checks test observation filenames and reads
    # split-level patient-map metadata at construction, including the test map;
    # that metadata access is disclosed separately and is not an
    # observation/ground-truth array access.
    parts_to_hash = ("train", "validation")
    for part in parts_to_hash:
        last = last_ids[part]
        for i in (0, last):
            for kind in ("observation", "ground_truth"):
                path = data / f"{kind}_{part}_{i:03d}.hdf5"
                if not path.is_file():
                    raise FileNotFoundError(path)
                evidence[str(path)] = sha256(path)
    angle_indices = list(range(0, 1000, 1000 // args.views))
    if len(angle_indices) != args.views:
        raise ValueError("view count must divide the 1,000-view acquisition")

    tracked_code = [
        Path(__file__).resolve(),
        dival_root / "dival/reconstructors/learnedpd_reconstructor.py",
        dival_root / "dival/reconstructors/standard_learned_reconstructor.py",
        dival_root / "dival/reconstructors/reconstructor.py",
        dival_root / "dival/reconstructors/networks/iterative.py",
        dival_root / "dival/datasets/angle_subset_dataset.py",
        dival_root / "dival/datasets/lodopab_dataset.py",
        dival_root / "dival/datasets/standard.py",
        dival_root / "dival/measure.py",
        dival_root / "dival/version.py",
    ]
    for path in tracked_code:
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence[str(path)] = sha256(path)

    return {
        "command": args.command,
        "views": args.views,
        "angle_indices": angle_indices,
        "data": str(data),
        "dival_root": str(dival_root),
        "hyper_params": hp,
        "hyper_params_path": str(hp_path),
        "hyper_params_sha256": sha256(hp_path),
        "test_patient_count": "not validated before authenticated inference",
        "test_file_existence_and_patient_map_metadata_may_be_read_by_dataset_constructor": True,
        "test_observations_or_ground_truth_read_by_runner_during_training": False,
        "expected_lengths": EXPECTED_LENGTHS,
        "input_sha256": evidence,
    }


def fingerprint_authorized_test_assets(data: Path) -> dict[str, object]:
    """Fingerprint test files only after the final-checkpoint gate passes."""
    evidence: dict[str, str] = {}
    for i in (0, 27):
        for kind in ("observation", "ground_truth"):
            path = data / f"{kind}_test_{i:03d}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(path)
            evidence[str(path)] = sha256(path)
    pid = data / "patient_ids_rand_test.csv"
    patient_ids = np.loadtxt(pid, dtype=int)
    if patient_ids.shape != (EXPECTED_LENGTHS["test"],):
        raise ValueError("test patient map does not contain exactly 3,553 rows")
    if np.unique(patient_ids).size != 60:
        raise ValueError("test patient map does not contain exactly 60 patients")
    evidence[str(pid)] = sha256(pid)
    return {"input_sha256": evidence, "test_patient_count": 60}


def import_runtime(args: argparse.Namespace):
    sys.path.insert(0, str(args.dival_root.resolve()))
    import torch
    import torch.nn.functional as functional
    import dival
    from dival import get_standard_dataset
    from dival.config import CONFIG
    from dival.reconstructors.learnedpd_reconstructor import LearnedPDReconstructor

    # Avoid DIVal set_config: it mutates ~/.dival/config.json.  LoDoPaBDataset
    # captures DATA_PATH at module import, so update both values explicitly.
    CONFIG["lodopab_dataset"]["data_path"] = str(args.data.resolve())
    import dival.datasets.lodopab_dataset as lodopab_module

    lodopab_module.DATA_PATH = str(args.data.resolve())
    return torch, functional, dival, get_standard_dataset, LearnedPDReconstructor


def environment_record(torch, dival) -> dict[str, object]:
    try:
        import astra
        astra_version = getattr(astra, "__version__", "unknown")
        astra_cuda = bool(astra.use_cuda())
    except Exception as exc:  # pragma: no cover - depends on remote runtime
        astra_version = f"ERROR: {exc}"
        astra_cuda = False
    try:
        import odl
        odl_version = odl.__version__
    except Exception as exc:  # pragma: no cover
        odl_version = f"ERROR: {exc}"
    try:
        import tensorboard
        tensorboard_version = tensorboard.__version__
    except Exception as exc:  # pragma: no cover
        tensorboard_version = f"ERROR: {exc}"
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(Path(dival.__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        git_commit = "UNAVAILABLE"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dival": getattr(dival, "__version__", "unknown"),
        "dival_file": str(Path(dival.__file__).resolve()),
        "dival_git_commit": git_commit,
        "odl": odl_version,
        "tensorboard": tensorboard_version,
        "astra": astra_version,
        "astra_cuda": astra_cuda,
    }


def construct(
    args: argparse.Namespace, hp: dict[str, object], log_dir: Path | None = None
):
    torch, functional, dival, get_standard_dataset, reconstructor_cls = import_runtime(args)
    dataset = get_standard_dataset(
        "lodopab", impl=args.impl, num_angles=args.views, sorted_by_patient=False
    )
    lengths = {part: dataset.get_len(part) for part in EXPECTED_LENGTHS}
    if lengths != EXPECTED_LENGTHS:
        raise ValueError(f"unexpected split lengths: {lengths}")
    ray = dataset.get_ray_trafo(impl=args.impl)
    checkpoint_base = args.out.resolve() / "best_model"
    reconstructor = reconstructor_cls(
        ray,
        hyper_params=hp,
        num_data_loader_workers=args.workers,
        use_cuda=True,
        show_pbar=True,
        log_dir=str(log_dir) if log_dir is not None else None,
        save_best_learned_params_path=str(checkpoint_base),
        torch_manual_seed=args.seed,
        shuffle=True,
    )
    return torch, functional, dival, dataset, ray, reconstructor


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ASTRA/PyTorch may remain nondeterministic.  Do not claim determinism.
    torch.backends.cudnn.benchmark = False


class ValidationAuditWriter:
    """Minimal DIVal SummaryWriter replacement that records only audit scalars.

    DIVal otherwise writes one TensorBoard event per training batch (358,200
    calls for the frozen run).  This replacement changes logging only: it
    records the two validation scalars per epoch and any nonfinite scalar from
    any phase in a flushed JSON-lines file.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)

    @staticmethod
    def _float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        value = self._float(scalar_value)
        keep = tag in ("loss/validation", "psnr/validation") or not np.isfinite(value)
        if keep:
            record = {"tag": str(tag), "step": int(global_step), "value": value}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()

    def add_images(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def read_validation_trace(trace_path: Path, expected_epochs: int) -> dict[str, object]:
    """Read the runner-owned epoch validation trace emitted inside DIVal."""
    if not trace_path.is_file():
        raise RuntimeError("DIVal did not write the runner validation trace")
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nonfinite = [r for r in records if not np.isfinite(float(r["value"]))]
    if nonfinite:
        raise FloatingPointError(f"nonfinite DIVal training/validation scalar: {nonfinite[0]}")
    psnr_records = [r for r in records if r["tag"] == "psnr/validation"]
    loss_records = [r for r in records if r["tag"] == "loss/validation"]
    if len(psnr_records) != expected_epochs or len(loss_records) != expected_epochs:
        raise RuntimeError(
            f"expected {expected_epochs} validation epochs; found "
            f"{len(psnr_records)} PSNR and {len(loss_records)} loss records"
        )
    values = np.array([r["value"] for r in psnr_records], dtype=np.float64)
    best_index = int(np.argmax(values))
    return {
        "tag": "psnr/validation",
        "epochs": [
            {
                "epoch": i + 1,
                "logged_step": int(psnr_record["step"]),
                "psnr_db": float(psnr_record["value"]),
                "mse_loss": float(loss_record["value"]),
            }
            for i, (psnr_record, loss_record) in enumerate(
                zip(psnr_records, loss_records)
            )
        ],
        "best_epoch_from_log": best_index + 1,
        "best_psnr_db_from_log": float(values[best_index]),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha256(trace_path),
        "logging_instrument": "runner ValidationAuditWriter; training-batch finite canary plus epoch validation loss/PSNR",
    }


def evaluate_restored_validation(reconstructor, dataset) -> dict[str, object]:
    """Re-evaluate the restored best weights on the complete validation set."""
    values = np.empty(EXPECTED_LENGTHS["validation"], dtype=np.float64)
    start = time.time()
    count = 0
    for idx, (obs, gt) in enumerate(dataset.generator(part="validation")):
        if idx >= len(values):
            raise RuntimeError("validation generator yielded too many samples")
        rec = np.asarray(reconstructor.reconstruct(obs))
        target = np.asarray(gt)
        if rec.shape != (362, 362) or target.shape != (362, 362):
            raise ValueError("unexpected restored-best validation image shape")
        mse = np.mean((rec - target) ** 2)
        data_range = np.max(target) - np.min(target)
        value = 20 * np.log10(data_range) - 10 * np.log10(mse)
        if not np.isfinite(value):
            raise FloatingPointError(
                f"nonfinite restored-best validation PSNR at index {idx}"
            )
        values[idx] = value
        count += 1
    if count != len(values):
        raise RuntimeError(
            f"validation generator yielded {count}, expected {len(values)}"
        )
    return {
        "n": int(len(values)),
        "mean_psnr_db": float(values.mean()),
        "min_psnr_db": float(values.min()),
        "max_psnr_db": float(values.max()),
        "elapsed_seconds": time.time() - start,
        "definition": "DIVal full-362 per-image ground-truth-range PSNR",
    }


def preflight(args: argparse.Namespace, base: dict[str, object]) -> int:
    hp = dict(base["hyper_params"])
    torch, _, dival, dataset, ray, reconstructor = construct(args, hp)
    env = environment_record(torch, dival)
    if args.impl == "astra_cuda" and not (env["cuda_available"] and env["astra_cuda"]):
        raise RuntimeError("astra_cuda preflight requires CUDA-enabled PyTorch and ASTRA")
    op_shape = {"domain": list(ray.domain.shape), "range": list(ray.range.shape)}
    if op_shape != {"domain": [362, 362], "range": [args.views, 513]}:
        raise ValueError(f"operator shape mismatch: {op_shape}")
    reconstructor.init_model()
    n_params = sum(p.numel() for p in reconstructor.model.parameters())
    record = {
        **base,
        "status": "PREFLIGHT_PASS",
        "environment": env,
        "operator_shape": op_shape,
        "learned_parameters": n_params,
        "resolved_hyper_params": reconstructor.hyper_params,
        "opnorm": float(reconstructor.opnorm),
        "timestamp_unix": time.time(),
    }
    write_json(args.out.resolve() / "preflight.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def train(args: argparse.Namespace, base: dict[str, object]) -> int:
    out = args.out.resolve()
    require_empty_or_resume(out, args.resume)
    if args.resume:
        raise NotImplementedError(
            "DIVal stores weights only; exact optimizer/scheduler resume is not supported"
        )
    hp = dict(base["hyper_params"])
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        hp["epochs"] = args.epochs
    is_final = hp["epochs"] == 10
    trace_path = out / "validation_trace.jsonl"
    audit_log_dir = out / "audit_logging_enabled"
    torch, _, dival, dataset, ray, reconstructor = construct(
        args, hp, log_dir=audit_log_dir
    )
    seed_everything(torch, args.seed)
    env = environment_record(torch, dival)
    if not (env["cuda_available"] and env["astra_cuda"]):
        raise RuntimeError("training requires CUDA-enabled PyTorch and ASTRA")
    start = time.time()
    record = {
        **base,
        "hyper_params_executed": hp,
        "resolved_hyper_params": reconstructor.hyper_params,
        "run_kind": "final" if is_final else "timing_pilot",
        "environment": env,
        "operator_shape": {"domain": list(ray.domain.shape), "range": list(ray.range.shape)},
        "seed": args.seed,
        "known_nondeterminism": "single-seed CUDA/ASTRA run; bitwise determinism not claimed",
        "test_observations_or_ground_truth_accessed": False,
        "test_file_existence_and_patient_map_metadata_may_be_read_by_dataset_constructor": True,
        "checkpoint_rule": "first integer epoch attaining the highest mean validation PSNR (strict > update in DIVal)",
        "audit_logging": "DIVal SummaryWriter sink replaced by runner ValidationAuditWriter; no training computation changed",
        "status": "STARTED",
        "start_unix": start,
    }
    write_json(out / "train_manifest.json", record)
    initial_state: dict[str, str] = {}
    original_init_model = reconstructor.init_model

    def audited_init_model() -> None:
        original_init_model()
        if not initial_state:
            initial_state["sha256"] = model_state_sha256(reconstructor.model)

    # Capture the exact initialization performed inside DIVal's own train()
    # rather than attempting to reproduce it in a separate dry initialization.
    reconstructor.init_model = audited_init_model
    import dival.reconstructors.standard_learned_reconstructor as standard_module

    had_summary_writer = hasattr(standard_module, "SummaryWriter")
    original_summary_writer = getattr(standard_module, "SummaryWriter", None)
    original_tensorboard_available = standard_module.TENSORBOARD_AVAILABLE

    def audit_writer_factory(*args, **kwargs):
        return ValidationAuditWriter(trace_path)

    # Replace DIVal's logging sink, not its optimizer/model/data path.  This
    # avoids 358,200 per-batch TensorBoard writes while preserving a flushed
    # epoch validation trace and a nonfinite scalar canary.
    standard_module.SummaryWriter = audit_writer_factory
    standard_module.TENSORBOARD_AVAILABLE = True
    torch.cuda.reset_peak_memory_stats()
    try:
        reconstructor.train(dataset)
    finally:
        reconstructor.init_model = original_init_model
        if had_summary_writer:
            standard_module.SummaryWriter = original_summary_writer
        else:
            delattr(standard_module, "SummaryWriter")
        standard_module.TENSORBOARD_AVAILABLE = original_tensorboard_available
    dival_train_end = time.time()
    if "sha256" not in initial_state:
        raise RuntimeError("failed to fingerprint the actual initialized model")
    selected_state_sha256 = model_state_sha256(reconstructor.model)
    if selected_state_sha256 == initial_state["sha256"]:
        raise RuntimeError(
            "restored state is identical to initialization; no validated training update was selected"
        )
    validation_trace = read_validation_trace(trace_path, int(hp["epochs"]))
    restored_validation = evaluate_restored_validation(reconstructor, dataset)
    logged_best = float(validation_trace["best_psnr_db_from_log"])
    if abs(restored_validation["mean_psnr_db"] - logged_best) > 1e-3:
        raise RuntimeError(
            "restored-best validation PSNR does not match the DIVal epoch trace "
            f"within 0.001 dB: {restored_validation['mean_psnr_db']} vs {logged_best}"
        )
    # DIVal's validation-improvement hook writes before best_model_wts is
    # restored, so save once more here after train() restores the best state.
    reconstructor.save_params(str(out / "best_model"))
    checkpoint = out / "best_model.pt"
    record.update(
        {
            "status": "COMPLETE",
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - start,
            "dival_train_elapsed_seconds": dival_train_end - start,
            "post_training_audit_elapsed_seconds": time.time() - dival_train_end,
            "initial_model_state_sha256": initial_state["sha256"],
            "selected_model_state_sha256": selected_state_sha256,
            "validation_trace": validation_trace,
            "restored_best_validation": restored_validation,
            "selected_epoch_evidence": "best_epoch_from_log; independently restored full-validation mean agrees within 0.001 dB",
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "saved_hyper_params": str(out / "best_model_hyper_params.json"),
            "saved_hyper_params_sha256": sha256(out / "best_model_hyper_params.json"),
        }
    )
    write_json(out / "train_manifest.json", record)
    return 0


def gaussian_ssim(torch, functional, pred, gt, data_range):
    coords = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, 11, 11).contiguous()
    pred_pad = functional.pad(pred, (5, 5, 5, 5), mode="reflect")
    gt_pad = functional.pad(gt, (5, 5, 5, 5), mode="reflect")
    mu_x = functional.conv2d(pred_pad, kernel)
    mu_y = functional.conv2d(gt_pad, kernel)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = functional.conv2d(functional.pad(pred * pred, (5, 5, 5, 5), mode="reflect"), kernel) - mu_x2
    sigma_y2 = functional.conv2d(functional.pad(gt * gt, (5, 5, 5, 5), mode="reflect"), kernel) - mu_y2
    sigma_xy = functional.conv2d(functional.pad(pred * gt, (5, 5, 5, 5), mode="reflect"), kernel) - mu_xy
    sigma_x2, sigma_y2 = sigma_x2.clamp_min(0), sigma_y2.clamp_min(0)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    out = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return float(out.mean())


def image_metrics(torch, functional, pred: np.ndarray, gt: np.ndarray):
    p = torch.from_numpy(np.asarray(pred, dtype=np.float32))[None, None]
    g = torch.from_numpy(np.asarray(gt, dtype=np.float32))[None, None]
    dr = (g.max() - g.min()).clamp_min(1e-8)
    mse = (p - g).square().mean().clamp_min(1e-12)
    psnr = float(20 * torch.log10(dr) - 10 * torch.log10(mse))
    # Match W-TransUNet's audited implementation exactly: PSNR clamps the MSE,
    # whereas RMSE adds 1e-12 before the square root.
    rmse = float(torch.sqrt((p - g).square().mean() + 1e-12))
    return psnr, gaussian_ssim(torch, functional, p, g, dr), rmse


def check_fbp_cache_canary(fbp_operator, observation, cached: np.ndarray) -> dict[str, object]:
    """Verify fixed test index 0 against the declared Ram-Lak 1.0 FBP."""
    regenerated = np.asarray(fbp_operator(observation), dtype=np.float32)
    cached = np.asarray(cached, dtype=np.float32)
    if regenerated.shape != (362, 362) or cached.shape != (362, 362):
        raise ValueError("FBP canary shape mismatch")
    delta = regenerated - cached
    rel_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(cached), 1e-12))
    max_abs = float(np.max(np.abs(delta)))
    passed = bool(
        np.allclose(
            regenerated,
            cached,
            rtol=FBP_CANARY_RTOL,
            atol=FBP_CANARY_ATOL,
            equal_nan=False,
        )
    )
    if not passed:
        raise ValueError(
            "FBP cache canary failed for test index 0 under Ram-Lak 1.0: "
            f"relative L2={rel_l2}, max abs={max_abs}"
        )
    return {
        "status": "PASS",
        "test_index": 0,
        "filter_type": "Ram-Lak",
        "frequency_scaling": 1.0,
        "rtol": FBP_CANARY_RTOL,
        "atol": FBP_CANARY_ATOL,
        "relative_l2": rel_l2,
        "max_abs": max_abs,
    }


def infer(args: argparse.Namespace, base: dict[str, object]) -> int:
    out = args.out.resolve()
    require_empty_or_resume(out, args.resume)
    if args.resume:
        raise NotImplementedError(
            "partial test inference cannot be resumed or promoted to complete evidence"
        )
    checkpoint = args.checkpoint.resolve()
    train_manifest_path = args.train_manifest.resolve()
    cache_path = args.fbp_cache.resolve()
    if not checkpoint.is_file() or not train_manifest_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("checkpoint, train manifest, or condition-matched 362 FBP cache missing")
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    if train_manifest.get("status") != "COMPLETE" or train_manifest.get("run_kind") != "final":
        raise ValueError("test inference is allowed only for a completed 10-epoch final run")
    if train_manifest.get("views") != args.views or train_manifest.get("seed") != args.seed:
        raise ValueError("checkpoint manifest view/seed mismatch")
    if train_manifest.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("checkpoint hash does not match its completed training manifest")
    if f"{args.views}angle" not in str(cache_path):
        raise ValueError("FBP cache path does not encode the requested view count")
    train_code = {
        Path(path).name: digest
        for path, digest in train_manifest.get("input_sha256", {}).items()
        if path.endswith(".py")
    }
    infer_code = {
        Path(path).name: digest
        for path, digest in base.get("input_sha256", {}).items()
        if path.endswith(".py")
    }
    if train_code != infer_code:
        raise ValueError("runner/DIVal source hashes changed between training and inference")
    hp = dict(train_manifest["hyper_params_executed"])
    if hp != base["hyper_params"]:
        raise ValueError("final checkpoint did not use the frozen reference configuration")
    # This is the first observation/ground-truth access to the test split in
    # the workflow, and it occurs only after all final-checkpoint gates above.
    test_assets = fingerprint_authorized_test_assets(Path(base["data"]))
    base["input_sha256"].update(test_assets["input_sha256"])
    base["test_patient_count"] = test_assets["test_patient_count"]
    torch, functional, dival, dataset, ray, reconstructor = construct(args, hp)
    seed_everything(torch, args.seed)
    env = environment_record(torch, dival)
    if not (env["cuda_available"] and env["astra_cuda"]):
        raise RuntimeError("inference requires CUDA-enabled PyTorch and ASTRA")
    reconstructor.load_learned_params(str(checkpoint.with_suffix("")))
    reconstructor.model.eval()
    if reconstructor.hyper_params != train_manifest.get("resolved_hyper_params"):
        raise ValueError("resolved hyper-parameters changed between training and inference")
    n_params = sum(p.numel() for p in reconstructor.model.parameters())

    fbp = np.load(cache_path, mmap_mode="r")
    if fbp.shape != (EXPECTED_LENGTHS["test"], 362, 362):
        raise ValueError(f"FBP cache shape mismatch: {fbp.shape}")
    cache_sha256 = sha256(cache_path)
    if cache_sha256 != EXPECTED_FBP_CACHE_SHA256[args.views]:
        raise ValueError(
            "FBP cache is not the byte-locked cache used for W-TransUNet evaluation"
        )
    from odl.tomo import fbp_op

    cache_canary_operator = fbp_op(
        ray, filter_type="Ram-Lak", frequency_scaling=1.0
    )
    rec_path = out / "reconstructions_float32.npy"
    recs = None
    if args.save_reconstructions:
        recs = np.lib.format.open_memmap(
            rec_path, mode="w+", dtype=np.float32, shape=(EXPECTED_LENGTHS["test"], 362, 362)
        )

    metrics_path = out / "per_image_metrics.csv"
    proj_path = out / "projection_residual.csv"
    start = time.time()
    manifest = {
        **base,
        "status": "STARTED",
        "environment": env,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "train_manifest": str(train_manifest_path),
        "train_manifest_sha256": sha256(train_manifest_path),
        "resolved_hyper_params": reconstructor.hyper_params,
        "learned_parameters": n_params,
        "fbp_cache": str(cache_path),
        "fbp_cache_sha256": cache_sha256,
        "image_endpoint": "central native output crop [5:357,5:357], raw/unclipped",
        "projection_primary_cross_method": "central 352 LearnedPD crop inserted into the same byte-locked 362 FBP frame used for W-TransUNet",
        "projection_secondary_intrinsic": "native 362x362 LearnedPD output under matched A; not directly ranked against composite post-processors",
        "start_unix": start,
    }
    write_json(out / "infer_manifest.json", manifest)

    torch.cuda.synchronize()
    cuda_baseline_allocated = torch.cuda.memory_allocated()
    cuda_baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    inference_ms: list[float] = []
    cache_canary = None
    with metrics_path.open("w", newline="", encoding="utf-8") as fm, proj_path.open(
        "w", newline="", encoding="utf-8"
    ) as fp:
        wm = csv.writer(fm)
        wp = csv.writer(fp)
        wm.writerow(["angle", "index", "model", "psnr", "ssim", "rmse", "inference_ms"])
        wp.writerow(["angle", "index", "model", "r", "convention"])
        for idx in range(EXPECTED_LENGTHS["test"]):
            obs, gt = dataset.get_sample(idx, part="test")
            y = np.asarray(obs, dtype=np.float32)
            gt362 = np.asarray(gt, dtype=np.float32)
            if idx == 0:
                cache_canary = check_fbp_cache_canary(
                    cache_canary_operator, obs, fbp[idx]
                )
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            x362 = np.asarray(reconstructor.reconstruct(obs), dtype=np.float32)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - infer_start) * 1000
            inference_ms.append(elapsed_ms)
            if x362.shape != (362, 362) or not np.isfinite(x362).all():
                raise FloatingPointError(f"invalid reconstruction at index {idx}")
            psnr, ssim, rmse = image_metrics(
                torch, functional, x362[IMAGE_CROP, IMAGE_CROP], gt362[IMAGE_CROP, IMAGE_CROP]
            )
            if not np.isfinite([psnr, ssim, rmse]).all():
                raise FloatingPointError(f"invalid image metric at index {idx}")
            wm.writerow([args.views, idx, MODEL_LABEL, psnr, ssim, rmse, elapsed_ms])
            denom = float(np.linalg.norm(y))
            if not np.isfinite(denom) or denom <= 0:
                raise FloatingPointError(f"invalid observation norm at index {idx}")
            native_r = float(np.linalg.norm(np.asarray(ray(ray.domain.element(x362))) - y) / denom)
            composite = np.asarray(fbp[idx], dtype=np.float32).copy()
            if not np.isfinite(composite).all():
                raise FloatingPointError(f"invalid FBP cache value at index {idx}")
            composite[IMAGE_CROP, IMAGE_CROP] = x362[IMAGE_CROP, IMAGE_CROP]
            composite_r = float(np.linalg.norm(np.asarray(ray(ray.domain.element(composite))) - y) / denom)
            if not np.isfinite([native_r, composite_r]).all():
                raise FloatingPointError(f"invalid projection residual at index {idx}")
            wp.writerow(
                [
                    args.views,
                    idx,
                    MODEL_LABEL,
                    composite_r,
                    "fbp_frame_primary_cross_method",
                ]
            )
            wp.writerow(
                [
                    args.views,
                    idx,
                    MODEL_LABEL,
                    native_r,
                    "native362_intrinsic_secondary",
                ]
            )
            if recs is not None:
                recs[idx] = x362
            if idx % 100 == 0:
                print(f"test index {idx}/{EXPECTED_LENGTHS['test']}", flush=True)
    if recs is not None:
        recs.flush()

    with metrics_path.open(newline="", encoding="utf-8") as f:
        if sum(1 for _ in csv.DictReader(f)) != EXPECTED_LENGTHS["test"]:
            raise RuntimeError("metric coverage is not exactly 3,553")
    with proj_path.open(newline="", encoding="utf-8") as f:
        if sum(1 for _ in csv.DictReader(f)) != 2 * EXPECTED_LENGTHS["test"]:
            raise RuntimeError("projection coverage is not exactly 2 x 3,553")
    if cache_canary is None or len(inference_ms) != EXPECTED_LENGTHS["test"]:
        raise RuntimeError("inference timing or FBP canary coverage failure")
    timing = np.asarray(inference_ms, dtype=np.float64)
    manifest.update(
        {
            "status": "COMPLETE",
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - start,
            "per_image_metrics_sha256": sha256(metrics_path),
            "projection_residual_sha256": sha256(proj_path),
            "reconstructions_sha256": sha256(rec_path) if recs is not None else None,
            "n_test": EXPECTED_LENGTHS["test"],
            "fbp_cache_canary": cache_canary,
            "model_reconstruct_timing_ms": {
                "definition": "per-slice end-to-end DIVal reconstruct call with CUDA synchronization; excludes metric and residual computation",
                "n": int(timing.size),
                "mean": float(timing.mean()),
                "median": float(np.median(timing)),
                "p95": float(np.percentile(timing, 95)),
                "minimum": float(timing.min()),
                "maximum": float(timing.max()),
            },
            "cuda_memory_mib": {
                "definition": "whole inference loop, including projection residual operations; resident model is included in baseline",
                "baseline_allocated": cuda_baseline_allocated / 2**20,
                "baseline_reserved": cuda_baseline_reserved / 2**20,
                "peak_allocated": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved": torch.cuda.max_memory_reserved() / 2**20,
                "peak_allocated_increment": (
                    torch.cuda.max_memory_allocated() - cuda_baseline_allocated
                )
                / 2**20,
            },
        }
    )
    write_json(out / "infer_manifest.json", manifest)
    return 0


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    require_empty_or_resume(args.out, bool(getattr(args, "resume", False)))
    base = validate_assets(args)
    if args.command == "preflight":
        return preflight(args, base)
    if args.command == "train":
        return train(args, base)
    if args.command == "infer":
        return infer(args, base)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
