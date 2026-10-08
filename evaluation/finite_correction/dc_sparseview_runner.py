#!/usr/bin/env python3
"""Fail-closed architecture-neutral finite-DC execution runner.

The scientific workflow is intentionally split into commands that cannot tune
on test data:

``preflight-validation`` -> ``prepare-validation`` -> ``sweep-validation``
-> ``lock-protocol`` -> ``evaluate-test``.

``smoke-validation`` -> ``pilot-dc`` is a separate, non-authorizing execution
diagnostic and cannot replace or feed the formal lock/test chain.

Validation commands have no test-array arguments.  The test command verifies a
cryptographic validation lock before it resolves a test path.  CUDA/ASTRA and
DIVal imports are lazy, so schema and synthetic tests run on the Mac while the
formal physics execution remains an ETRI ASTRA-CUDA task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dc_assets import load_expected_assets, sha256_file, verify_assets
from dc_core import (
    BORDER,
    CROP_SIZE,
    FULL_SIZE,
    adjoint_relative_error,
    crop_canvas,
    dc_definition,
    embed_crop,
    finite_dc_trajectory,
    gradient_scale_term,
    power_iteration_ata,
    projection_residual,
)
from dc_metrics import metric_definition, torch_image_metrics
from dc_dataset_manifest import load_manifest as load_dataset_manifest, verify_split
from dc_schemas import (
    CASE_METRICS_SCHEMA,
    ITERATIONS,
    LAMBDA_MULTIPLIERS,
    MODELS,
    SPLIT_FIREWALL_DISCLOSURE,
    VIEWS,
    atomic_write_json,
    json_sha256,
    load_json,
    make_lock,
    validate_case_metric_row,
    validate_lock,
    validate_protocol,
    write_lock,
)
from dc_selection import (
    aggregate_validation_candidates,
    compute_common_validation_scale,
    select_all_targets,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PACKAGE_ROOT / "frozen_dc_protocol.json"
EXPECTED_ASSETS_PATH = PACKAGE_ROOT / "expected_assets.json"
EXPECTED_COUNTS = {"validation": 3522, "test": 3553}
EXPECTED_PATIENTS = {"validation": 60, "test": 60}
FORMAL_TEST_INFERENCE_BATCH = 16
RUNTIME_POWER_RTOL = 1.0e-6
RUNTIME_POWER_ATOL = 0.0
CANONICAL_MODELS = tuple(MODELS)
CASE_COLUMNS = (
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

QUALITATIVE_METRIC_COLUMNS = (
    "schema",
    "case_label",
    "quantile",
    "rank",
    "case_id",
    "patient_id",
    "panel_key",
    "architecture",
    "phase",
    "dc_iteration",
    "checkpoint_sha256",
    "post_load_state_sha256",
    "lambda",
    "lambda_multiplier",
    "eta",
    "psnr",
    "ssim",
    "rmse",
    "projection_residual",
    "clipped",
    "normalized",
    "backend",
    "source_test_run_id",
    "operating_point_status",
    "operating_point_claim_allowed",
    "diagnostic_fallback",
)

QUALITATIVE_CASE_LABELS = (
    ("low_or_failure", 0.10),
    ("median", 0.50),
    ("high_benefit", 0.90),
)

QUALITATIVE_PANEL_KEYS = (
    "unet",
    "unet_finite_dc",
    "transunet",
    "transunet_finite_dc",
    "wtransunet",
    "wtransunet_finite_dc",
)


def _stats_row(row: Mapping[str, Any], trajectory_id: str) -> dict[str, Any]:
    """Convert the execution row to the exact CPU statistics schema."""

    return {
        "split": row["split"],
        "views": row["views"],
        "case_id": row["case_id"],
        "patient_id": row["patient_id"],
        "model": row["architecture"],
        "candidate_id": trajectory_id,
        "k": row["dc_iteration"],
        "projection_residual": row["projection_residual"],
        "psnr_db": row["psnr"],
        "ssim": row["ssim"],
        "rmse": row["rmse"],
        "schema": row["schema"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "lambda": row["lambda"],
        "lambda_multiplier": row["lambda_multiplier"],
        "eta": row["eta"],
        "diverged": row["diverged"],
        "validation_selected": row["validation_selected"],
        "backend": row["backend"],
        "source_run_id": row["source_run_id"],
    }


def _write_stats_row(
    writer: csv.DictWriter,
    row: Mapping[str, Any],
    trajectory_id: str,
) -> None:
    writer.writerow(_stats_row(row, trajectory_id))


class RunnerError(RuntimeError):
    pass


def _protocol() -> tuple[dict[str, Any], str]:
    value = load_json(PROTOCOL_PATH)
    validate_protocol(value)
    return value, json_sha256(value)


def _package_hashes() -> dict[str, str]:
    names = (
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
    result: dict[str, str] = {}
    for name in names:
        path = PACKAGE_ROOT / name
        if not path.is_file():
            raise RunnerError(f"package file missing: {path}")
        result[name] = sha256_file(path)
    return result


def _new_output_directory(path: Path) -> Path:
    if path.is_symlink():
        raise RunnerError(f"output path may not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    protected = [PACKAGE_ROOT.resolve(), Path("<T9_DRIVE>").resolve()]
    if any(resolved == item or item in resolved.parents for item in protected):
        if PACKAGE_ROOT.resolve() in resolved.parents:
            # Package result directories are allowed, but never a source file's
            # parent or the package root itself.
            if resolved == PACKAGE_ROOT.resolve():
                raise RunnerError("output may not be the package source directory")
        elif Path("<T9_DRIVE>").resolve() in resolved.parents:
            raise RunnerError("formal outputs may not be written into the T9 evidence tree")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise RunnerError(f"output directory must be new and empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _started_manifest(command: str, views: int, split: str, run_kind: str = "FINAL") -> dict[str, Any]:
    protocol, protocol_hash = _protocol()
    del protocol
    return {
        "schema": "dc.run_manifest.v1",
        "status": "STARTED",
        "command": command,
        "run_kind": run_kind,
        "run_id": str(uuid.uuid4()),
        "views": views,
        "split": split,
        "protocol_sha256": protocol_hash,
        "package_sha256": _package_hashes(),
        "split_firewall": dict(SPLIT_FIREWALL_DISCLOSURE),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_utc": None,
        "inputs": {},
        "outputs": {},
        "coverage": {},
        "canaries": {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "failure": None,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, manifest, overwrite=True)


def _complete_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["status"] = "COMPLETE"
    manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(path, manifest)


def _fail_manifest(path: Path, manifest: dict[str, Any], exc: BaseException) -> None:
    manifest["status"] = "FAILED"
    manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    _write_manifest(path, manifest)


def _patient_ids(path: Path, expected_n: int) -> np.ndarray:
    ids = np.loadtxt(path, dtype=np.int64)
    if ids.shape != (expected_n,):
        raise RunnerError(f"patient map expected {expected_n} rows, got {ids.shape}")
    if np.unique(ids).size != 60:
        raise RunnerError("patient map must contain exactly 60 unique patients")
    return ids


def _import_physics(data: Path, dival_root: Path, views: int, *, backend: str = "astra_cuda"):
    if str(dival_root.resolve()) not in sys.path:
        sys.path.insert(0, str(dival_root.resolve()))
    import torch
    import dival
    from dival import get_standard_dataset
    from dival.config import CONFIG

    CONFIG["lodopab_dataset"]["data_path"] = str(data.resolve())
    import dival.datasets.lodopab_dataset as lodopab_module

    lodopab_module.DATA_PATH = str(data.resolve())
    dataset = get_standard_dataset("lodopab", impl=backend, num_angles=views)
    operator = dataset.get_ray_trafo(impl=backend)
    return torch, dival, dataset, operator


def _environment(torch: Any, dival: Any, backend: str) -> dict[str, Any]:
    try:
        import astra
        astra_version = astra.__version__
        astra_cuda = bool(astra.use_cuda())
    except Exception as exc:  # pragma: no cover - runtime dependent
        astra_version = f"ERROR: {exc}"
        astra_cuda = False
    try:
        import odl
        odl_version = odl.__version__
    except Exception as exc:  # pragma: no cover
        odl_version = f"ERROR: {exc}"
    if backend == "astra_cuda" and (not torch.cuda.is_available() or not astra_cuda):
        raise RunnerError("formal execution requires both torch CUDA and ASTRA CUDA")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dival": getattr(dival, "__version__", "unknown"),
        "odl": odl_version,
        "astra": astra_version,
        "astra_cuda": astra_cuda,
        "backend": backend,
    }


def _python_major_minor(version: Any) -> tuple[int, int]:
    if not isinstance(version, str) or not version:
        raise RunnerError("runtime Python version is missing")
    try:
        parts = version.split()[0].split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise RunnerError(f"invalid runtime Python version: {version!r}") from exc


def _assert_test_runtime_parity(
    sweep: Mapping[str, Any],
    lock: Mapping[str, Any],
    observed_environment: Mapping[str, Any],
    observed_operator: Mapping[str, Any],
) -> dict[str, Any]:
    """Require formal test software/operator parity with validation.

    GPU model names are recorded but are not part of the mathematical parity
    gate.  Binary/container identity is also outside this version-level gate.
    """

    expected_environment = _mapping(sweep.get("environment"), "validation environment")
    if _python_major_minor(observed_environment.get("python")) != _python_major_minor(
        expected_environment.get("python")
    ):
        raise RunnerError("formal test Python major/minor differs from validation")
    exact_fields = (
        "numpy",
        "torch",
        "torch_cuda",
        "dival",
        "odl",
        "astra",
        "backend",
    )
    for field in exact_fields:
        if observed_environment.get(field) != expected_environment.get(field):
            raise RunnerError(f"formal test runtime {field} differs from validation")
    if observed_environment.get("cuda_available") is not True or observed_environment.get(
        "astra_cuda"
    ) is not True:
        raise RunnerError("formal test runtime lacks required CUDA backends")

    canaries = _mapping(sweep.get("canaries"), "validation canaries")
    expected_operator = _mapping(canaries.get("operator"), "validation operator canary")
    for field in ("domain_shape", "range_shape"):
        if observed_operator.get(field) != expected_operator.get(field):
            raise RunnerError(f"formal test operator {field} differs from validation")
    try:
        observed_power = float(observed_operator["lambda_max_ata"])
        sweep_power = float(expected_operator["lambda_max_ata"])
        sweep_config_power = float(
            _mapping(sweep.get("scientific_config"), "validation scientific_config")[
                "lambda_max_ata"
            ]
        )
        locked_power = float(_mapping(lock.get("payload"), "lock.payload")["lambda_max_ata"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("formal test power-canary evidence is invalid") from exc
    if not all(np.isfinite(value) and value > 0.0 for value in (
        observed_power,
        sweep_power,
        sweep_config_power,
        locked_power,
    )):
        raise RunnerError("formal test power-canary evidence is nonfinite")
    for label, expected in (
        ("validation operator canary", sweep_power),
        ("validation scientific config", sweep_config_power),
        ("validation lock", locked_power),
    ):
        if not np.isclose(
            observed_power,
            expected,
            rtol=RUNTIME_POWER_RTOL,
            atol=RUNTIME_POWER_ATOL,
        ):
            raise RunnerError(f"formal test power estimate differs from {label}")
    return {
        "status": "PASS",
        "python_match": "major.minor",
        "exact_version_fields": list(exact_fields),
        "validation_gpu": expected_environment.get("gpu"),
        "test_gpu": observed_environment.get("gpu"),
        "gpu_name_match_required": False,
        "binary_container_identity_locked": False,
        "operator_domain_shape": observed_operator.get("domain_shape"),
        "operator_range_shape": observed_operator.get("range_shape"),
        "observed_lambda_max_ata": observed_power,
        "validation_lambda_max_ata": sweep_power,
        "validation_config_lambda_max_ata": sweep_config_power,
        "locked_lambda_max_ata": locked_power,
        "power_rtol": RUNTIME_POWER_RTOL,
        "power_atol": RUNTIME_POWER_ATOL,
    }


def _operator_canaries(operator: Any, views: int) -> dict[str, Any]:
    if tuple(operator.domain.shape) != (FULL_SIZE, FULL_SIZE):
        raise RunnerError(f"operator domain mismatch: {operator.domain.shape}")
    if tuple(operator.range.shape) != (views, 513):
        raise RunnerError(f"operator range mismatch: {operator.range.shape}")
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(3):
        x = rng.standard_normal((FULL_SIZE, FULL_SIZE)).astype(np.float32)
        z = rng.standard_normal((views, 513)).astype(np.float32)
        pairs.append((x, z))
    # AMENDMENT 1 (2026-08-17, user-authorized; see AMENDMENT_1_cuda_adjoint_canary.md):
    # ASTRA's CUDA forward/backprojector kernels are an unmatched pair, so the
    # weighted-space adjoint identity cannot reach the frozen 1.0e-4 bound there
    # (measured max 5.64e-01 with this exact seed/sequence). The identity gate —
    # a geometry/implementation check per STATIC_AUDIT.md — therefore runs on the
    # matched ASTRA-CPU twin operator, exactly as it was validated at freeze time
    # (measured max 1.10e-05). The CUDA errors are recorded descriptively, and the
    # shape checks and the power iteration that feeds eta stay on the execution
    # (CUDA) operator, which is also the operator the frozen DC update itself uses.
    cuda_errors = [adjoint_relative_error(operator, x, z) for x, z in pairs]
    from dival import get_standard_dataset as _amend1_gsd
    cpu_twin = _amend1_gsd("lodopab", impl="astra_cpu", num_angles=views)
    cpu_operator = cpu_twin.get_ray_trafo(impl="astra_cpu")
    if tuple(cpu_operator.domain.shape) != (FULL_SIZE, FULL_SIZE):
        raise RunnerError(f"cpu twin domain mismatch: {cpu_operator.domain.shape}")
    if tuple(cpu_operator.range.shape) != (views, 513):
        raise RunnerError(f"cpu twin range mismatch: {cpu_operator.range.shape}")
    errors = [adjoint_relative_error(cpu_operator, x, z) for x, z in pairs]
    if max(errors) > 1.0e-4:
        raise RunnerError(f"adjoint canary failed: {errors}")
    estimate, trace = power_iteration_ata(operator, seed=0, iterations=20)
    return {
        "domain_shape": list(operator.domain.shape),
        "range_shape": list(operator.range.shape),
        "adjoint_relative_errors": errors,
        "adjoint_identity_backend": "astra_cpu_matched_pair (AMENDMENT 1)",
        "cuda_adjoint_relative_errors_descriptive": cuda_errors,
        "lambda_max_ata": estimate,
        "power_trace": trace,
    }


def _asset_bindings(args: argparse.Namespace, split: str) -> dict[str, Path]:
    view = args.views
    result = {
        f"checkpoint_unet_{view}": args.checkpoint_unet,
        f"checkpoint_transunet_{view}": args.checkpoint_transunet,
        f"checkpoint_wavtransunet_{view}": args.checkpoint_wavtransunet,
        "pretrained_r50_vit_b16_imagenet21k": args.pretrained_npz,
        f"patient_map_{split}": args.patient_map,
        f"fbp_{split}_{view}": args.fbp_cache,
        f"lodopab_{split}_file_manifest": args.dataset_manifest,
    }
    if split == "test":
        result[f"archived_metrics_{view}"] = args.archived_metrics
        result[f"archived_projection_{view}"] = args.archived_projection
    return result


def _verify_phase_assets(args: argparse.Namespace, split: str) -> dict[str, Any]:
    manifest = load_expected_assets(EXPECTED_ASSETS_PATH)
    bindings = _asset_bindings(args, split)
    required = tuple(bindings)
    allowed = ("shared", split)
    report = verify_assets(
        manifest,
        bindings,
        required_ids=required,
        allowed_splits=allowed,
    )
    # The small package manifest is an authenticated asset; this second gate
    # re-hashes every official observation/ground-truth HDF5 byte for this
    # split. Validation phases never open the separate test manifest.
    dataset_manifest = load_dataset_manifest(args.dataset_manifest)
    report["lodopab_file_verification"] = verify_split(
        args.data,
        dataset_manifest,
        split,
    )
    return report


def _require_complete_manifest(path: Path, command: str, *, run_kind: str = "FINAL") -> dict[str, Any]:
    document = load_json(path)
    if document.get("schema") != "dc.run_manifest.v1":
        raise RunnerError("unsupported run manifest schema")
    if document.get("status") != "COMPLETE" or document.get("command") != command:
        raise RunnerError(f"manifest is not COMPLETE {command}")
    if document.get("run_kind") != run_kind:
        raise RunnerError(f"manifest run_kind must be {run_kind}")
    _, expected_protocol = _protocol()
    if document.get("protocol_sha256") != expected_protocol:
        raise RunnerError("manifest protocol hash mismatch")
    if document.get("package_sha256") != _package_hashes():
        raise RunnerError("manifest package hash mismatch")
    return document


def _prediction_paths(prepared_manifest: Mapping[str, Any]) -> dict[str, Path]:
    outputs = _mapping(prepared_manifest.get("outputs"), "prediction manifest outputs")
    root_record = _mapping(outputs.get("prediction_root"), "prediction root output")
    root_value = root_record.get("path")
    if not isinstance(root_value, str):
        raise RunnerError("prediction root path is missing")
    root = Path(root_value)
    if root.is_symlink() or not root.is_dir():
        raise RunnerError("prediction root must be an existing regular directory")
    paths = {model: root / f"{model}.npy" for model in CANONICAL_MODELS}
    for model, path in paths.items():
        output = _mapping(outputs.get(f"predictions_{model}"), f"prediction output {model}")
        recorded_path = output.get("path")
        if (
            not isinstance(recorded_path, str)
            or Path(recorded_path).resolve() != path.resolve()
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != output.get("sha256")
        ):
            raise RunnerError(f"prepared prediction hash mismatch: {model}")
    return paths


def _prediction_crop(array: np.ndarray, index: int) -> np.ndarray:
    """Read one 352 crop from either ``(N,352,352)`` or ``(N,1,352,352)``."""

    if array.ndim == 4 and array.shape[1:] == (1, CROP_SIZE, CROP_SIZE):
        value = array[index, 0]
    elif array.ndim == 3 and array.shape[1:] == (CROP_SIZE, CROP_SIZE):
        value = array[index]
    else:
        raise RunnerError(f"prediction array shape is not canonical: {array.shape}")
    return np.asarray(value, dtype=np.float32)


def command_preflight_validation(args: argparse.Namespace) -> int:
    out = _new_output_directory(args.out)
    path = out / "preflight_manifest.json"
    manifest = _started_manifest("preflight-validation", args.views, "validation")
    _write_manifest(path, manifest)
    try:
        manifest["inputs"]["asset_verification"] = _verify_phase_assets(args, "validation")
        torch, dival, dataset, operator = _import_physics(
            args.data, args.dival_root, args.views, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        manifest["canaries"]["operator"] = _operator_canaries(operator, args.views)
        patient_ids = _patient_ids(args.patient_map, EXPECTED_COUNTS["validation"])
        fbp = np.load(args.fbp_cache, mmap_mode="r")
        if fbp.shape != (EXPECTED_COUNTS["validation"], FULL_SIZE, FULL_SIZE) or fbp.dtype != np.float32:
            raise RunnerError(f"validation FBP cache mismatch: {fbp.shape}, {fbp.dtype}")
        crop = crop_canvas(np.asarray(fbp[0], dtype=np.float32))
        if not np.array_equal(embed_crop(crop, np.asarray(fbp[0])), np.asarray(fbp[0])):
            raise RunnerError("crop/embed FBP round-trip failed")
        expected_angles = list(range(0, 1000, 1000 // args.views))
        if len(expected_angles) != args.views:
            raise RunnerError("sparse-view index construction failed")
        manifest["canaries"].update(
            {
                "angle_indices": expected_angles,
                "patient_count": int(np.unique(patient_ids).size),
                "fbp_shape": list(fbp.shape),
                "metric_definition": metric_definition(),
            }
        )
        manifest["status"] = "PREFLIGHT_PASS"
        manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_manifest(path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(path, manifest, exc)
        raise


def command_smoke_validation(args: argparse.Namespace) -> int:
    if not 1 <= args.n <= 64:
        raise RunnerError("smoke n must be between 1 and 64")
    # The smoke command reuses the final preparation path but its manifest can
    # never authorize selection or test access.
    args.run_kind = "SMOKE"
    return _prepare_validation(args, limit=args.n)


def command_prepare_validation(args: argparse.Namespace) -> int:
    return _prepare_validation(args, limit=None)


def _authenticate_smoke_predictions(
    smoke_manifest_path: Path,
    *,
    views: int,
    n: int,
) -> tuple[dict[str, Any], dict[str, Path]]:
    smoke = _require_complete_manifest(
        smoke_manifest_path, "smoke-validation", run_kind="SMOKE"
    )
    if smoke.get("split") != "validation" or smoke.get("views") != views:
        raise RunnerError("pilot requires matching validation-only smoke predictions")
    coverage = _mapping(smoke.get("coverage"), "smoke prediction coverage")
    if (
        coverage.get("expected_cases") != n
        or coverage.get("processed_cases") != n
        or coverage.get("models") != list(CANONICAL_MODELS)
    ):
        raise RunnerError("smoke prediction coverage differs from pilot n/models")
    paths = _prediction_paths(smoke)
    for model, path in paths.items():
        output = _mapping(
            smoke.get("outputs", {}).get(f"predictions_{model}"),
            f"smoke prediction output {model}",
        )
        if output.get("shape") != [n, 1, CROP_SIZE, CROP_SIZE] or output.get(
            "dtype"
        ) != "float32":
            raise RunnerError(f"smoke prediction schema differs for {model}")
        report = _mapping(output.get("inference_report"), f"smoke inference report {model}")
        for field, expected in (
            ("cases", n),
            ("input_dtype", "float32"),
            ("output_dtype", "float32"),
            ("normalized", False),
            ("clipped", False),
            ("case_order_preserved", True),
            ("exact_coverage", True),
        ):
            if report.get(field) != expected:
                raise RunnerError(f"smoke inference report differs: {model}/{field}")
    return smoke, paths


def _authenticate_smoke_checkpoint_evidence(
    smoke: Mapping[str, Any],
    expected_checkpoint_hashes: Mapping[str, str],
    *,
    expected_checkpoint_paths: Mapping[str, Path] | None = None,
    expected_code_root: Path | None = None,
) -> dict[str, Any]:
    """Bind pilot inputs to the exact checkpoint states used by smoke inference."""

    from dc_model_factory import AUDITED_MODEL_SOURCE_SHA256

    if set(expected_checkpoint_hashes) != set(CANONICAL_MODELS):
        raise RunnerError("pilot checkpoint map is incomplete")
    inputs = _mapping(smoke.get("inputs"), "smoke inputs")
    evidence = _mapping(inputs.get("model_evidence"), "smoke model evidence")
    if set(evidence) != set(CANONICAL_MODELS):
        raise RunnerError("smoke model evidence is incomplete")
    authenticated: dict[str, Any] = {}
    topology = {
        "UNet": {
            "unet_scales": 5,
            "unet_skip_channels": 4,
            "unet_use_sigmoid": False,
            "unet_use_norm": True,
        },
        "TransUNet": {
            "transunet_variant": "R50-ViT-B_16",
            "transunet_n_classes": 1,
            "transunet_n_skip": 3,
            "transunet_grid": [22, 22],
        },
        "WavTransUNet": {
            "transunet_variant": "R50-ViT-B_16",
            "transunet_n_classes": 1,
            "transunet_n_skip": 3,
            "transunet_grid": [22, 22],
            "wav_base_channels": 64,
            "wav_blocks": 8,
            "wav_norm": "gn",
            "wav_upsample": "bilinear",
            "residual_output": True,
        },
    }
    expected_sources = {
        "UNet": ("unet.py",),
        "TransUNet": (
            "vit_seg_modeling.py",
            "vit_seg_configs.py",
            "vit_seg_modeling_resnet_skip.py",
        ),
        "WavTransUNet": (
            "vit_seg_modeling.py",
            "vit_seg_configs.py",
            "vit_seg_modeling_resnet_skip.py",
            "wavelet_ops.py",
        ),
    }
    for model in CANONICAL_MODELS:
        record = _mapping(evidence.get(model), f"smoke model evidence {model}")
        build = _mapping(record.get("build"), f"smoke model build {model}")
        load = _mapping(record.get("load"), f"smoke checkpoint load {model}")
        expected_hash = expected_checkpoint_hashes[model]
        post_load_hash = load.get("post_load_state_sha256")
        source_hashes = _mapping(
            build.get("source_sha256"), f"smoke source hashes {model}"
        )
        if (
            build.get("architecture") != model
            or build.get("image_size") != CROP_SIZE
            or build.get("source_hashes_enforced") is not True
            or dict(source_hashes)
            != {
                name: AUDITED_MODEL_SOURCE_SHA256[name]
                for name in expected_sources[model]
            }
            or any(build.get(field) != value for field, value in topology[model].items())
        ):
            raise RunnerError(f"smoke model topology/source evidence differs: {model}")
        if expected_code_root is not None and (
            not isinstance(build.get("code_root"), str)
            or Path(str(build["code_root"])).resolve() != expected_code_root.resolve()
        ):
            raise RunnerError(f"smoke model code root differs: {model}")
        if expected_checkpoint_paths is not None and (
            model not in expected_checkpoint_paths
            or not isinstance(load.get("checkpoint_path"), str)
            or Path(str(load["checkpoint_path"])).resolve()
            != expected_checkpoint_paths[model].resolve()
        ):
            raise RunnerError(f"smoke checkpoint path differs: {model}")
        if (
            load.get("checkpoint_sha256") != expected_hash
            or load.get("strict") is not True
            or load.get("missing_keys") not in ([], ())
            or load.get("unexpected_keys") not in ([], ())
            or not _valid_sha256(post_load_hash)
        ):
            raise RunnerError(f"smoke checkpoint/state evidence differs: {model}")
        authenticated[model] = {
            "checkpoint_sha256": expected_hash,
            "post_load_state_sha256": post_load_hash,
            "strict": True,
        }
    return authenticated


PILOT_STATUS_COLUMNS = (
    "views",
    "candidate_id",
    "lambda_multiplier",
    "lambda",
    "eta",
    "status",
    "processed_cases",
    "expected_cases",
    "models_completed",
    "expected_models",
    "metric_rows",
    "expected_metric_rows",
    "elapsed_seconds",
    "case_model_trajectories_per_second",
    "peak_gpu_memory_bytes",
    "reason",
)


def _validate_pilot_result_grid(
    rows: Sequence[Mapping[str, Any]],
    candidate_status: Sequence[Mapping[str, Any]],
    *,
    n: int,
    views: int,
    run_id: str,
) -> dict[str, int | bool]:
    """Fail closed on the complete non-authorizing pilot result grid."""

    expected_keys = {
        (case_id, model, 0, 0.0)
        for case_id in range(n)
        for model in CANONICAL_MODELS
    }
    expected_keys.update(
        (case_id, model, k, float(multiplier))
        for multiplier in LAMBDA_MULTIPLIERS
        for case_id in range(n)
        for model in CANONICAL_MODELS
        for k in ITERATIONS[1:]
    )
    observed_keys: set[tuple[int, str, int, float]] = set()
    for row in rows:
        try:
            validate_case_metric_row(row)
        except Exception as exc:
            raise RunnerError(f"pilot metric row is noncanonical: {exc}") from exc
        if (
            row.get("split") != "validation"
            or row.get("views") != views
            or row.get("source_run_id") != run_id
        ):
            raise RunnerError("pilot metric row split/view/run identity differs")
        key = (
            int(row["case_id"]),
            str(row["architecture"]),
            int(row["dc_iteration"]),
            float(row["lambda_multiplier"]),
        )
        if key in observed_keys:
            raise RunnerError(f"duplicate pilot metric row: {key}")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)[:3]
        extra = sorted(observed_keys - expected_keys)[:3]
        raise RunnerError(f"pilot metric grid differs; missing={missing} extra={extra}")

    if len(candidate_status) != len(LAMBDA_MULTIPLIERS):
        raise RunnerError("pilot candidate-status coverage is incomplete")
    observed_multipliers: list[float] = []
    status_by_multiplier: dict[float, Mapping[str, Any]] = {}
    expected_nonzero_rows = n * len(CANONICAL_MODELS) * (len(ITERATIONS) - 1)
    for record in candidate_status:
        try:
            multiplier = float(record.get("lambda_multiplier"))
            expected_cases = int(record.get("expected_cases"))
            expected_models = int(record.get("expected_models"))
            metric_count = int(record.get("metric_rows"))
            expected_metric_count = int(record.get("expected_metric_rows"))
            processed_cases = int(record.get("processed_cases"))
            completed_models = int(record.get("models_completed"))
            lam = float(record.get("lambda"))
            eta = float(record.get("eta"))
            elapsed = float(record.get("elapsed_seconds"))
            throughput = float(record.get("case_model_trajectories_per_second"))
        except (TypeError, ValueError) as exc:
            raise RunnerError("pilot candidate status contains invalid counts") from exc
        observed_multipliers.append(multiplier)
        status = record.get("status")
        if (
            status not in ("COMPLETE", "DIVERGED")
            or record.get("views") != views
            or record.get("candidate_id") != f"m_{multiplier:g}"
            or expected_cases != n
            or expected_models != n * len(CANONICAL_MODELS)
            or metric_count != expected_nonzero_rows
            or expected_metric_count != expected_nonzero_rows
            or not np.isfinite(lam)
            or lam <= 0.0
            or not np.isfinite(eta)
            or eta <= 0.0
            or not np.isfinite(elapsed)
            or elapsed < 0.0
            or not np.isfinite(throughput)
            or throughput < 0.0
        ):
            raise RunnerError("pilot candidate status is noncanonical")
        peak_memory = record.get("peak_gpu_memory_bytes")
        if peak_memory != "" and (
            type(peak_memory) is not int or peak_memory < 0
        ):
            raise RunnerError("pilot peak-GPU-memory evidence is invalid")
        if status == "COMPLETE" and (
            processed_cases != n
            or completed_models != n * len(CANONICAL_MODELS)
            or record.get("reason") != ""
        ):
            raise RunnerError("complete pilot candidate has incomplete coverage")
        if status == "DIVERGED" and not str(record.get("reason", "")).strip():
            raise RunnerError("divergent pilot candidate lacks a failure reason")
        status_by_multiplier[multiplier] = record
    if observed_multipliers != list(LAMBDA_MULTIPLIERS):
        raise RunnerError("pilot candidate order/grid differs from frozen protocol")
    for row in rows:
        if int(row["dc_iteration"]) == 0:
            if row.get("diverged") is not False:
                raise RunnerError("pilot baseline row cannot be divergent")
            continue
        multiplier = float(row["lambda_multiplier"])
        record = status_by_multiplier[multiplier]
        if (
            not np.isclose(
                float(row["lambda"]), float(record["lambda"]), rtol=0.0, atol=0.0
            )
            or not np.isclose(
                float(row["eta"]), float(record["eta"]), rtol=0.0, atol=0.0
            )
            or bool(row["diverged"]) != (record["status"] == "DIVERGED")
        ):
            raise RunnerError("pilot metric/status trajectory evidence differs")
    return {
        "rows": len(rows),
        "candidate_rows": len(candidate_status),
        "exact_metric_grid": True,
        "exact_candidate_grid": True,
    }


def command_pilot_dc(args: argparse.Namespace) -> int:
    """Run a validation-only, non-authorizing finite-DC execution pilot."""

    if not 1 <= args.n <= 64:
        raise RunnerError("pilot n must be between 1 and 64")
    smoke, predictions = _authenticate_smoke_predictions(
        args.smoke_manifest,
        views=args.views,
        n=args.n,
    )
    out = _new_output_directory(args.out)
    manifest_path = out / "pilot_dc_manifest.json"
    manifest = _started_manifest(
        "pilot-dc", args.views, "validation", run_kind="SMOKE"
    )
    _write_manifest(manifest_path, manifest)
    try:
        manifest["authorization"] = {
            "lock_authorization": False,
            "test_authorization": False,
            "scientific_selection_authorization": False,
            "purpose": "execution stability and cost diagnostics only",
        }
        manifest["inputs"]["smoke_prediction_manifest"] = {
            "path": str(args.smoke_manifest.resolve()),
            "sha256": sha256_file(args.smoke_manifest),
            "run_id": smoke["run_id"],
            "run_kind": smoke["run_kind"],
        }
        current_asset_verification = _verify_phase_assets(args, "validation")
        manifest["inputs"]["asset_verification"] = current_asset_verification
        smoke_inputs = _mapping(smoke.get("inputs"), "smoke inputs")
        smoke_asset_verification = _mapping(
            smoke_inputs.get("asset_verification"), "smoke asset verification"
        )
        if not _canonical_equal(
            smoke_asset_verification, current_asset_verification
        ):
            raise RunnerError("pilot assets differ from authenticated smoke inputs")
        checkpoint_hashes = {
            model: sha256_file(path)
            for model, path in (
                ("UNet", args.checkpoint_unet),
                ("TransUNet", args.checkpoint_transunet),
                ("WavTransUNet", args.checkpoint_wavtransunet),
            )
        }
        manifest["inputs"]["smoke_model_evidence"] = (
            _authenticate_smoke_checkpoint_evidence(
                smoke,
                checkpoint_hashes,
                expected_checkpoint_paths={
                    "UNet": args.checkpoint_unet,
                    "TransUNet": args.checkpoint_transunet,
                    "WavTransUNet": args.checkpoint_wavtransunet,
                },
                expected_code_root=args.wtu_code,
            )
        )
        torch, dival, dataset, operator = _import_physics(
            args.data, args.dival_root, args.views, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        operator_info = _operator_canaries(operator, args.views)
        manifest["canaries"]["operator"] = operator_info
        fbp = np.load(args.fbp_cache, mmap_mode="r", allow_pickle=False)
        if fbp.shape != (
            EXPECTED_COUNTS["validation"],
            FULL_SIZE,
            FULL_SIZE,
        ) or fbp.dtype != np.float32:
            raise RunnerError("pilot validation FBP cache shape/dtype differs")
        if int(dataset.get_len("validation")) != EXPECTED_COUNTS["validation"]:
            raise RunnerError("official validation split length changed")
        patients = _patient_ids(
            args.patient_map, EXPECTED_COUNTS["validation"]
        )
        arrays = {
            model: np.load(path, mmap_mode="r", allow_pickle=False)
            for model, path in predictions.items()
        }

        data_cache: list[tuple[np.ndarray, np.ndarray]] = []
        scale_rows: list[dict[str, Any]] = []
        for index in range(args.n):
            y_obs, gt = dataset.get_sample(index, part="validation")
            y = np.asarray(y_obs, dtype=np.float32)
            gt362 = np.asarray(gt, dtype=np.float32)
            if y.shape != (args.views, 513) or gt362.shape != (
                FULL_SIZE,
                FULL_SIZE,
            ):
                raise RunnerError(f"pilot validation sample shape differs: {index}")
            data_cache.append((y, gt362))
            for model in CANONICAL_MODELS:
                canvas = embed_crop(
                    _prediction_crop(arrays[model], index), fbp[index]
                )
                image_norm = float(np.linalg.norm(canvas.astype(np.float64)))
                scale_rows.append(
                    {
                        "split": "validation",
                        "views": args.views,
                        "architecture": model,
                        "case_id": index,
                        "adjoint_data_gradient_norm": gradient_scale_term(
                            operator, canvas, y
                        )
                        * image_norm,
                        "image_norm": image_norm,
                    }
                )
        common_scale = compute_common_validation_scale(
            scale_rows, expected_cases=args.n
        )
        lam_max = float(operator_info["lambda_max_ata"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        total_start = time.perf_counter()
        metric_rows: list[dict[str, Any]] = []
        # k=0 is invariant across lambda candidates and is published once.
        for index, (y, gt362) in enumerate(data_cache):
            for model in CANONICAL_MODELS:
                canvas = embed_crop(
                    _prediction_crop(arrays[model], index), fbp[index]
                )
                metric_rows.append(
                    _row(
                        split="validation",
                        case_id=index,
                        patient_id=int(patients[index]),
                        views=args.views,
                        architecture=model,
                        checkpoint_sha256=checkpoint_hashes[model],
                        k=0,
                        lam=0.0,
                        multiplier=0.0,
                        eta=0.0,
                        metrics=_metrics_one(canvas, gt362, "cuda"),
                        residual=projection_residual(operator, canvas, y),
                        diverged=False,
                        selected=False,
                        backend=args.backend,
                        run_id=manifest["run_id"],
                    )
                )

        candidate_status: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            observed_peak_gpu_memory = int(torch.cuda.max_memory_allocated())
        else:
            observed_peak_gpu_memory = 0
        for multiplier in LAMBDA_MULTIPLIERS:
            lam = float(multiplier * common_scale)
            eta = float(1.0 / (lam_max + lam))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            candidate_start = time.perf_counter()
            candidate_rows: list[dict[str, Any]] = []
            processed_cases = 0
            models_completed = 0
            diverged = False
            reason = ""
            for index, (y, gt362) in enumerate(data_cache):
                case_models = 0
                for model in CANONICAL_MODELS:
                    canvas = embed_crop(
                        _prediction_crop(arrays[model], index), fbp[index]
                    )
                    trajectory = finite_dc_trajectory(
                        operator,
                        canvas,
                        fbp[index],
                        y,
                        regularization=lam,
                        step_size=eta,
                        iterations=ITERATIONS,
                    )
                    if trajectory.diverged:
                        diverged = True
                        reason = (
                            f"{model} case {index}: "
                            f"{trajectory.divergence_reason}"
                        )
                        break
                    for k in ITERATIONS[1:]:
                        candidate_rows.append(
                            _row(
                                split="validation",
                                case_id=index,
                                patient_id=int(patients[index]),
                                views=args.views,
                                architecture=model,
                                checkpoint_sha256=checkpoint_hashes[model],
                                k=k,
                                lam=lam,
                                multiplier=float(multiplier),
                                eta=eta,
                                metrics=_metrics_one(
                                    trajectory.snapshots[k], gt362, "cuda"
                                ),
                                residual=trajectory.residuals[k],
                                diverged=False,
                                selected=False,
                                backend=args.backend,
                                run_id=manifest["run_id"],
                            )
                        )
                    case_models += 1
                    models_completed += 1
                if diverged:
                    break
                if case_models == len(CANONICAL_MODELS):
                    processed_cases += 1
            expected_metric_rows = (
                args.n * len(CANONICAL_MODELS) * (len(ITERATIONS) - 1)
            )
            if diverged:
                candidate_rows = list(
                    _complete_divergent_candidate_rows(
                        patients=patients[: args.n],
                        views=args.views,
                        checkpoint_hash=checkpoint_hashes,
                        lam=lam,
                        multiplier=float(multiplier),
                        eta=eta,
                        backend=args.backend,
                        run_id=manifest["run_id"],
                    )
                )
            metric_rows.extend(candidate_rows)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                candidate_peak_gpu_memory: int | str = int(
                    torch.cuda.max_memory_allocated()
                )
                observed_peak_gpu_memory = max(
                    observed_peak_gpu_memory, int(candidate_peak_gpu_memory)
                )
            else:
                candidate_peak_gpu_memory = ""
            elapsed = time.perf_counter() - candidate_start
            candidate_status.append(
                {
                    "views": args.views,
                    "candidate_id": f"m_{multiplier:g}",
                    "lambda_multiplier": float(multiplier),
                    "lambda": lam,
                    "eta": eta,
                    "status": "DIVERGED" if diverged else "COMPLETE",
                    "processed_cases": processed_cases,
                    "expected_cases": args.n,
                    "models_completed": models_completed,
                    "expected_models": args.n * len(CANONICAL_MODELS),
                    "metric_rows": len(candidate_rows),
                    "expected_metric_rows": expected_metric_rows,
                    "elapsed_seconds": elapsed,
                    "case_model_trajectories_per_second": (
                        models_completed / elapsed if elapsed > 0.0 else 0.0
                    ),
                    "peak_gpu_memory_bytes": candidate_peak_gpu_memory,
                    "reason": reason,
                }
            )

        rows_partial = out / "pilot_case_metrics.csv.partial"
        with rows_partial.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CASE_COLUMNS)
            writer.writeheader()
            writer.writerows(metric_rows)
            stream.flush()
            os.fsync(stream.fileno())
        rows_path = out / "pilot_case_metrics.csv"
        os.replace(rows_partial, rows_path)
        status_partial = out / "pilot_candidate_status.csv.partial"
        with status_partial.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PILOT_STATUS_COLUMNS)
            writer.writeheader()
            writer.writerows(candidate_status)
            stream.flush()
            os.fsync(stream.fileno())
        status_path = out / "pilot_candidate_status.csv"
        os.replace(status_partial, status_path)

        total_elapsed = time.perf_counter() - total_start
        complete_candidates = sum(
            row["status"] == "COMPLETE" for row in candidate_status
        )
        diverged_candidates = len(candidate_status) - complete_candidates
        expected_total_rows = args.n * len(CANONICAL_MODELS) * (
            1 + len(LAMBDA_MULTIPLIERS) * (len(ITERATIONS) - 1)
        )
        if len(metric_rows) != expected_total_rows:
            raise RunnerError(
                f"pilot published rows {len(metric_rows)} != {expected_total_rows}"
            )
        coverage_canary = _validate_pilot_result_grid(
            metric_rows,
            candidate_status,
            n=args.n,
            views=args.views,
            run_id=manifest["run_id"],
        )
        manifest["outputs"] = {
            "case_metrics": {
                "path": str(rows_path.resolve()),
                "sha256": sha256_file(rows_path),
                "rows": len(metric_rows),
            },
            "candidate_status": {
                "path": str(status_path.resolve()),
                "sha256": sha256_file(status_path),
                "rows": len(candidate_status),
            },
        }
        manifest["scientific_config"] = {
            "source": "unchanged frozen protocol",
            "pilot_subset_rule": "first n canonical validation indices",
            "pilot_case_ids": list(range(args.n)),
            "common_scale": common_scale,
            "lambda_max_ata": lam_max,
            "lambda_multipliers": list(LAMBDA_MULTIPLIERS),
            "k_grid": list(ITERATIONS),
            "models": list(CANONICAL_MODELS),
            "metric_definition": metric_definition(),
            "data_consistency_definition": dc_definition(),
        }
        manifest["coverage"] = {
            "cases": args.n,
            "processed_cases": args.n,
            "case_ids": list(range(args.n)),
            "models": len(CANONICAL_MODELS),
            "model_names": list(CANONICAL_MODELS),
            "lambda_candidates": len(LAMBDA_MULTIPLIERS),
            "k_grid": list(ITERATIONS),
            "baseline_rows": args.n * len(CANONICAL_MODELS),
            "published_metric_rows": len(metric_rows),
            "expected_metric_rows": expected_total_rows,
            "complete_candidates": complete_candidates,
            "diverged_candidates": diverged_candidates,
            "exact_published_grid": len(metric_rows) == expected_total_rows,
            "all_trajectories_stable": diverged_candidates == 0,
            "grid_canary": coverage_canary,
        }
        manifest["timing"] = {
            "total_seconds": total_elapsed,
            "case_model_trajectories": sum(
                int(row["models_completed"]) for row in candidate_status
            ),
            "case_model_trajectories_per_second": (
                sum(int(row["models_completed"]) for row in candidate_status)
                / total_elapsed
                if total_elapsed > 0.0
                else 0.0
            ),
            "peak_gpu_memory_bytes": (
                observed_peak_gpu_memory
                if torch.cuda.is_available()
                else None
            ),
            "peak_gpu_memory_source": (
                "torch.cuda.max_memory_allocated; excludes memory owned by ASTRA"
                if torch.cuda.is_available()
                else "not available"
            ),
        }
        manifest["canaries"]["stability"] = {
            "status": "PASS" if diverged_candidates == 0 else "DIAGNOSTIC_DIVERGENCE",
            "all_candidates_complete": diverged_candidates == 0,
            "complete_candidates": complete_candidates,
            "diverged_candidates": diverged_candidates,
            "candidate_status_sha256": sha256_file(status_path),
        }
        _complete_manifest(manifest_path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _prepare_validation(args: argparse.Namespace, limit: int | None) -> int:
    command = "smoke-validation" if limit is not None else "prepare-validation"
    run_kind = "SMOKE" if limit is not None else "FINAL"
    out = _new_output_directory(args.out)
    path = out / "prepare_manifest.json"
    manifest = _started_manifest(command, args.views, "validation", run_kind=run_kind)
    _write_manifest(path, manifest)
    try:
        preflight = load_json(args.preflight_manifest)
        if preflight.get("status") != "PREFLIGHT_PASS" or preflight.get("views") != args.views:
            raise RunnerError("a matching PREFLIGHT_PASS manifest is required")
        _, protocol_hash = _protocol()
        if preflight.get("protocol_sha256") != protocol_hash:
            raise RunnerError("preflight protocol mismatch")
        manifest["inputs"]["preflight_manifest"] = {
            "path": str(args.preflight_manifest.resolve()),
            "sha256": sha256_file(args.preflight_manifest),
        }
        manifest["inputs"]["asset_verification"] = _verify_phase_assets(args, "validation")
        torch, dival, dataset, _ = _import_physics(
            args.data, args.dival_root, args.views, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        from dc_model_factory import build_and_load_archived_model, infer_raw_float32

        device = torch.device("cuda")
        checkpoint_paths = {
            "UNet": args.checkpoint_unet,
            "TransUNet": args.checkpoint_transunet,
            "WavTransUNet": args.checkpoint_wavtransunet,
        }
        asset_ids = {
            "UNet": f"checkpoint_unet_{args.views}",
            "TransUNet": f"checkpoint_transunet_{args.views}",
            "WavTransUNet": f"checkpoint_wavtransunet_{args.views}",
        }
        models: dict[str, Any] = {}
        model_evidence: dict[str, Any] = {}
        for model in CANONICAL_MODELS:
            built, build_report, load_report = build_and_load_archived_model(
                model,
                args.wtu_code,
                checkpoint_paths[model],
                expected_assets_path=EXPECTED_ASSETS_PATH,
                checkpoint_asset_id=asset_ids[model],
                views=args.views,
                device=device,
            )
            models[model] = built
            model_evidence[model] = {
                "build": build_report.to_dict(),
                "load": load_report.to_dict(),
            }
        manifest["inputs"]["model_evidence"] = model_evidence
        total = limit or EXPECTED_COUNTS["validation"]
        fbp = np.load(args.fbp_cache, mmap_mode="r")
        inputs = np.empty((total, 1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        for index in range(total):
            inputs[index, 0] = crop_canvas(fbp[index])
        prediction_root = out / "predictions"
        prediction_root.mkdir()
        for model in CANONICAL_MODELS:
            destination = prediction_root / f"{model}.npy"
            memory = np.lib.format.open_memmap(
                destination,
                mode="w+",
                dtype=np.float32,
                shape=(total, 1, CROP_SIZE, CROP_SIZE),
            )
            _, inference_report = infer_raw_float32(
                models[model],
                inputs,
                output=memory,
                batch_size=args.batch,
                device=device,
            )
            memory.flush()
            del memory
            manifest["outputs"][f"predictions_{model}"] = {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "shape": [total, 1, CROP_SIZE, CROP_SIZE],
                "dtype": "float32",
                "inference_report": inference_report.to_dict(),
            }
        manifest["outputs"]["prediction_root"] = {"path": str(prediction_root.resolve())}
        manifest["coverage"] = {
            "expected_cases": total,
            "processed_cases": total,
            "models": list(CANONICAL_MODELS),
        }
        _complete_manifest(path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(path, manifest, exc)
        raise


def _iter_split(dataset: Any, split: str, n: int):
    """Stream exactly ``n`` random-access samples without materializing a split.

    ``Dataset.get_data_pairs`` builds Python lists containing every requested
    sinogram and ground-truth image.  At LoDoPaB scale that needlessly retains
    several gigabytes and makes a formal sweep fragile.  The official
    LoDoPaB dataset implements deterministic random access, so every pass is
    streamed by index and the split length is checked before any sample read.
    """

    observed_n = int(dataset.get_len(split))
    if observed_n != int(n):
        raise RunnerError(
            f"official {split} split length changed: expected {n}, observed {observed_n}"
        )
    for index in range(n):
        yield index, dataset.get_sample(index, part=split)


def _row(
    *,
    split: str,
    case_id: int,
    patient_id: int,
    views: int,
    architecture: str,
    checkpoint_sha256: str,
    k: int,
    lam: float,
    multiplier: float,
    eta: float,
    metrics: Mapping[str, float] | None,
    residual: float | None,
    diverged: bool,
    selected: bool,
    backend: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema": CASE_METRICS_SCHEMA,
        "split": split,
        "case_id": case_id,
        "patient_id": patient_id,
        "views": views,
        "architecture": architecture,
        "checkpoint_sha256": checkpoint_sha256,
        "dc_iteration": k,
        "lambda": lam if k else 0.0,
        "lambda_multiplier": multiplier if k else 0.0,
        "eta": eta if k else 0.0,
        "psnr": None if metrics is None else float(metrics["psnr_db"]),
        "ssim": None if metrics is None else float(metrics["ssim"]),
        "rmse": None if metrics is None else float(metrics["rmse"]),
        "projection_residual": None if residual is None else float(residual),
        "diverged": bool(diverged),
        "validation_selected": bool(selected),
        "backend": backend,
        "source_run_id": run_id,
    }


def _trajectory_id(multiplier: float, *, baseline: bool = False) -> str:
    return "baseline" if baseline else f"m_{multiplier:g}"


def _complete_divergent_candidate_rows(
    *,
    patients: Sequence[int],
    views: int,
    checkpoint_hash: Mapping[str, str],
    lam: float,
    multiplier: float,
    eta: float,
    backend: str,
    run_id: str,
) -> Iterable[dict[str, Any]]:
    """Emit complete null coverage when one shared lambda trajectory diverges.

    Selection is architecture-neutral, so a divergence in any model/case
    invalidates that lambda for every model and every nonzero k.  Publishing
    the full versioned row grid preserves the prespecified candidate instead
    of silently dropping it from aggregation.
    """

    for case_id, patient_id in enumerate(patients):
        for model in CANONICAL_MODELS:
            for k in ITERATIONS[1:]:
                yield _row(
                    split="validation",
                    case_id=case_id,
                    patient_id=int(patient_id),
                    views=views,
                    architecture=model,
                    checkpoint_sha256=checkpoint_hash[model],
                    k=k,
                    lam=lam,
                    multiplier=multiplier,
                    eta=eta,
                    metrics=None,
                    residual=None,
                    diverged=True,
                    selected=False,
                    backend=backend,
                    run_id=run_id,
                )


def _metrics_one(image362: np.ndarray, gt362: np.ndarray, device: str) -> dict[str, float]:
    values = torch_image_metrics(crop_canvas(image362), crop_canvas(gt362), device=device)
    return {name: float(array[0]) for name, array in values.items()}


def command_sweep_validation(args: argparse.Namespace) -> int:
    out = _new_output_directory(args.out)
    path = out / "sweep_manifest.json"
    manifest = _started_manifest("sweep-validation", args.views, "validation")
    _write_manifest(path, manifest)
    try:
        prepared = _require_complete_manifest(args.prepared_manifest, "prepare-validation")
        if prepared["views"] != args.views or prepared["coverage"]["processed_cases"] != 3522:
            raise RunnerError("prepared validation coverage mismatch")
        predictions = _prediction_paths(prepared)
        manifest["inputs"]["prepared_manifest"] = {
            "path": str(args.prepared_manifest.resolve()),
            "sha256": sha256_file(args.prepared_manifest),
        }
        manifest["inputs"]["asset_verification"] = _verify_phase_assets(args, "validation")
        torch, dival, dataset, operator = _import_physics(
            args.data, args.dival_root, args.views, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        operator_info = _operator_canaries(operator, args.views)
        manifest["canaries"]["operator"] = operator_info
        patients = _patient_ids(args.patient_map, 3522)
        fbp = np.load(args.fbp_cache, mmap_mode="r")
        arrays = {model: np.load(file, mmap_mode="r") for model, file in predictions.items()}
        # Pass 1: common scale over all models and all validation cases.
        scale_rows = []
        for index, (y_obs, _gt) in _iter_split(dataset, "validation", 3522):
            y = np.asarray(y_obs, dtype=np.float32)
            for model in CANONICAL_MODELS:
                canvas = embed_crop(_prediction_crop(arrays[model], index), fbp[index])
                scale_rows.append(
                    {
                        "split": "validation",
                        "views": args.views,
                        "architecture": model,
                        "case_id": index,
                        "adjoint_data_gradient_norm": gradient_scale_term(operator, canvas, y)
                        * float(np.linalg.norm(canvas.astype(np.float64))),
                        "image_norm": float(np.linalg.norm(canvas.astype(np.float64))),
                    }
                )
        common_scale = compute_common_validation_scale(scale_rows, expected_cases=3522)
        lam_max = float(operator_info["lambda_max_ata"])
        rows_path = out / "validation_case_metrics.csv.partial"
        status_path = out / "validation_candidate_status.csv.partial"
        checkpoint_hash = {
            "UNet": sha256_file(args.checkpoint_unet),
            "TransUNet": sha256_file(args.checkpoint_transunet),
            "WavTransUNet": sha256_file(args.checkpoint_wavtransunet),
        }
        with rows_path.open("w", newline="", encoding="utf-8") as stream, status_path.open(
            "w", newline="", encoding="utf-8"
        ) as status_stream:
            writer = csv.DictWriter(stream, fieldnames=CASE_COLUMNS)
            writer.writeheader()
            status_writer = csv.DictWriter(
                status_stream,
                fieldnames=(
                    "views",
                    "candidate_id",
                    "lambda_multiplier",
                    "lambda",
                    "eta",
                    "status",
                    "processed_cases",
                    "reason",
                ),
            )
            status_writer.writeheader()
            # k=0 appears once per architecture/case.
            data_cache: list[tuple[np.ndarray, np.ndarray]] = []
            for index, (y_obs, gt) in _iter_split(dataset, "validation", 3522):
                y = np.asarray(y_obs, dtype=np.float32)
                gt362 = np.asarray(gt, dtype=np.float32)
                data_cache.append((y, gt362))
                for model in CANONICAL_MODELS:
                    canvas = embed_crop(_prediction_crop(arrays[model], index), fbp[index])
                    metrics = _metrics_one(canvas, gt362, "cuda")
                    writer.writerow(
                        _row(
                            split="validation",
                            case_id=index,
                            patient_id=int(patients[index]),
                            views=args.views,
                            architecture=model,
                            checkpoint_sha256=checkpoint_hash[model],
                            k=0,
                            lam=0.0,
                            multiplier=0.0,
                            eta=0.0,
                        metrics=metrics,
                            residual=projection_residual(operator, canvas, y),
                            diverged=False,
                            selected=False,
                        backend=args.backend,
                            run_id=manifest["run_id"],
                        )
                    )
            for multiplier in LAMBDA_MULTIPLIERS:
                lam = float(multiplier * common_scale)
                eta = float(1.0 / (lam_max + lam))
                candidate_rows: list[dict[str, Any]] = []
                diverged = False
                reason = ""
                processed = 0
                for index, (y, gt362) in enumerate(data_cache):
                    for model in CANONICAL_MODELS:
                        canvas = embed_crop(_prediction_crop(arrays[model], index), fbp[index])
                        trajectory = finite_dc_trajectory(
                            operator,
                            canvas,
                            fbp[index],
                            y,
                            regularization=lam,
                            step_size=eta,
                            iterations=ITERATIONS,
                        )
                        if trajectory.diverged:
                            diverged = True
                            reason = f"{model} case {index}: {trajectory.divergence_reason}"
                            break
                        for k in ITERATIONS[1:]:
                            metrics = _metrics_one(trajectory.snapshots[k], gt362, "cuda")
                            candidate_rows.append(
                                _row(
                                    split="validation",
                                    case_id=index,
                                    patient_id=int(patients[index]),
                                    views=args.views,
                                    architecture=model,
                                    checkpoint_sha256=checkpoint_hash[model],
                                    k=k,
                                    lam=lam,
                                    multiplier=float(multiplier),
                                    eta=eta,
                                    metrics=metrics,
                                    residual=trajectory.residuals[k],
                                    diverged=False,
                                    selected=False,
                                    backend=args.backend,
                                    run_id=manifest["run_id"],
                                )
                            )
                    if diverged:
                        break
                    processed += 1
                candidate_id = f"m_{multiplier:g}"
                if diverged:
                    writer.writerows(
                        _complete_divergent_candidate_rows(
                            patients=patients,
                            views=args.views,
                            checkpoint_hash=checkpoint_hash,
                            lam=lam,
                            multiplier=float(multiplier),
                            eta=eta,
                            backend=args.backend,
                            run_id=manifest["run_id"],
                        )
                    )
                else:
                    writer.writerows(candidate_rows)
                status_writer.writerow(
                    {
                        "views": args.views,
                        "candidate_id": candidate_id,
                        "lambda_multiplier": multiplier,
                        "lambda": lam,
                        "eta": eta,
                        "status": "DIVERGED" if diverged else "COMPLETE",
                        "processed_cases": processed,
                        "reason": reason,
                    }
                )
                stream.flush()
                status_stream.flush()
        final_rows = out / "validation_case_metrics.csv"
        final_status = out / "validation_candidate_status.csv"
        os.replace(rows_path, final_rows)
        os.replace(status_path, final_status)
        manifest["outputs"] = {
            "case_metrics": {
                "path": str(final_rows.resolve()),
                "sha256": sha256_file(final_rows),
            },
            "candidate_status": {
                "path": str(final_status.resolve()),
                "sha256": sha256_file(final_status),
            },
        }
        manifest["scientific_config"] = {
            "common_scale": common_scale,
            "lambda_max_ata": lam_max,
            "lambda_multipliers": list(LAMBDA_MULTIPLIERS),
            "k_grid": list(ITERATIONS),
        }
        manifest["coverage"] = {"baseline_rows": 3522 * 3, "cases": 3522, "models": 3}
        _complete_manifest(path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(path, manifest, exc)
        raise


def _read_case_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CASE_COLUMNS:
            raise RunnerError(
                f"case-metrics CSV header changed: {reader.fieldnames!r}"
            )
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in ("case_id", "patient_id", "views", "dc_iteration"):
                row[field] = int(row[field])
            for field in ("lambda", "lambda_multiplier", "eta"):
                row[field] = float(row[field])
            for field in ("diverged", "validation_selected"):
                if row[field] not in ("True", "False"):
                    raise RunnerError(f"case-metrics {field} must be True or False")
                row[field] = row[field] == "True"
            for field in ("psnr", "ssim", "rmse", "projection_residual"):
                row[field] = None if row[field] in ("", "None") else float(row[field])
            rows.append(row)
    return rows


def command_lock_protocol(args: argparse.Namespace) -> int:
    out = _new_output_directory(args.out)
    sweep = _require_complete_manifest(args.sweep_manifest, "sweep-validation")
    if sweep["coverage"]["cases"] != 3522:
        raise RunnerError("lock refuses incomplete validation coverage")
    metrics_path = Path(sweep["outputs"]["case_metrics"]["path"])
    if sha256_file(metrics_path) != sweep["outputs"]["case_metrics"]["sha256"]:
        raise RunnerError("validation metrics hash mismatch")
    rows = _read_case_rows(metrics_path)
    summaries = aggregate_validation_candidates(rows, expected_cases=3522)
    protocol, protocol_hash = _protocol()
    common_scale = float(sweep["scientific_config"]["common_scale"])
    lam_max = float(sweep["scientific_config"]["lambda_max_ata"])
    checkpoint_hashes = {
        model: next(
            row["checkpoint_sha256"] for row in rows if row["architecture"] == model
        )
        for model in CANONICAL_MODELS
    }
    selection = select_all_targets(summaries, require_complete_grid=True)
    operating_point = selection["selected_operating_point"]
    selected = operating_point["selected"]
    payload = {
        "protocol_sha256": protocol_hash,
        "selection_split": "validation",
        "split_firewall": dict(SPLIT_FIREWALL_DISCLOSURE),
        "views": int(args.views if hasattr(args, "views") else sweep["views"]),
        "validation_manifest_sha256": sha256_file(args.sweep_manifest),
        "validation_case_metrics_sha256": sha256_file(metrics_path),
        "validation_patient_map_sha256": sha256_file(args.patient_map),
        "checkpoint_sha256": checkpoint_hashes,
        "common_scale": common_scale,
        "lambda_max_ata": lam_max,
        "selected": selected,
        "operating_point_gate": {
            "status": operating_point["status"],
            "claim_allowed": operating_point["claim_allowed"],
            "psnr_floor_db": operating_point["psnr_floor_db"],
            "fallback_used": operating_point["fallback_used"],
            "failure_reason": operating_point["failure_reason"],
        },
        "residual_matching": selection["target_definition"],
        "k_grid": list(ITERATIONS),
    }
    lock = make_lock(payload)
    destination = out / "dc_lock.json"
    write_lock(destination, lock)
    # Produce the one selected validation trajectory consumed by the statistics
    # program.  The complete 7-lambda sweep remains immutable in the sweep
    # directory, while no downstream curve may connect different branches.
    selected_multiplier = float(selected["lambda_multiplier"])
    selected_rows = [
        row
        for row in rows
        if row["dc_iteration"] == 0
        or float(row["lambda_multiplier"]) == selected_multiplier
    ]
    expected_selected = 3522 * len(CANONICAL_MODELS) * len(ITERATIONS)
    if len(selected_rows) != expected_selected:
        raise RunnerError(
            f"selected validation trajectory rows {len(selected_rows)} != {expected_selected}"
        )
    # Statistics authenticates and filters the immutable raw validation sweep
    # using this lock.  No second derived schema is published here.
    return 0


def _canonical_equal(left: Any, right: Any) -> bool:
    """Compare scientific selections using the canonical JSON representation."""

    return json_sha256(left) == json_sha256(right)


def _authenticate_lock_before_test(
    lock_path: Path,
    sweep_manifest_path: Path,
    validation_csv_path: Path,
    *,
    expected_views: int,
    expected_validation_cases: int = EXPECTED_COUNTS["validation"],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the complete validation evidence before any test binding.

    The lock self-hash is necessary but insufficient: this function also
    authenticates the COMPLETE sweep manifest and full validation CSV, then
    independently re-runs the frozen selector and requires canonical equality
    of the selected point, gate, and residual targets.  Callers must invoke it
    before resolving or hashing any test asset.
    """

    lock = load_json(lock_path)
    validate_lock(lock)
    _, protocol_hash = _protocol()
    payload = lock["payload"]
    if payload.get("protocol_sha256") != protocol_hash:
        raise RunnerError("lock protocol mismatch")
    if payload.get("selection_split") != "validation":
        raise RunnerError("lock was not created from validation")
    if payload.get("views") != expected_views:
        raise RunnerError("lock view mismatch")
    required_payload = {
        "validation_manifest_sha256",
        "validation_case_metrics_sha256",
        "validation_patient_map_sha256",
        "checkpoint_sha256",
        "common_scale",
        "lambda_max_ata",
        "selected",
        "operating_point_gate",
        "residual_matching",
        "k_grid",
    }
    missing = sorted(required_payload - set(payload))
    if missing:
        raise RunnerError(f"test lock lacks frozen validation evidence: {missing}")
    checkpoint_payload = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_payload, Mapping) or set(checkpoint_payload) != set(
        CANONICAL_MODELS
    ):
        raise RunnerError("test lock checkpoint map is not canonical")
    if not isinstance(payload.get("selected"), Mapping):
        raise RunnerError("test lock has no selected operating point")
    if not isinstance(payload.get("operating_point_gate"), Mapping):
        raise RunnerError("test lock has no operating-point gate")
    if not isinstance(payload.get("residual_matching"), Mapping):
        raise RunnerError("test lock has no residual-matching definition")

    sweep = _require_complete_manifest(sweep_manifest_path, "sweep-validation")
    if sweep.get("split") != "validation" or sweep.get("views") != expected_views:
        raise RunnerError("validation sweep split/view mismatch")
    if sweep.get("coverage", {}).get("cases") != expected_validation_cases:
        raise RunnerError("validation sweep coverage mismatch")
    if payload.get("validation_manifest_sha256") != sha256_file(sweep_manifest_path):
        raise RunnerError("lock validation-manifest hash mismatch")

    output = sweep.get("outputs", {}).get("case_metrics")
    if not isinstance(output, Mapping):
        raise RunnerError("validation sweep lacks case-metrics evidence")
    if Path(output.get("path", "")).resolve() != validation_csv_path.resolve():
        raise RunnerError("supplied validation CSV is not the sweep output")
    validation_hash = sha256_file(validation_csv_path)
    if output.get("sha256") != validation_hash:
        raise RunnerError("validation CSV hash differs from sweep manifest")
    if payload.get("validation_case_metrics_sha256") != validation_hash:
        raise RunnerError("validation CSV hash differs from lock")
    candidate_status = sweep.get("outputs", {}).get("candidate_status")
    if not isinstance(candidate_status, Mapping):
        raise RunnerError("validation sweep lacks candidate-status evidence")
    candidate_status_path = Path(candidate_status.get("path", ""))
    if sha256_file(candidate_status_path) != candidate_status.get("sha256"):
        raise RunnerError("validation candidate-status hash mismatch")

    asset_verification = sweep.get("inputs", {}).get("asset_verification")
    if not isinstance(asset_verification, Mapping):
        raise RunnerError("validation sweep lacks asset-verification evidence")
    verified_assets = asset_verification.get("verified_assets")
    if not isinstance(verified_assets, list):
        raise RunnerError("validation asset-verification list is invalid")
    verified_by_id: dict[str, Mapping[str, Any]] = {}
    for item in verified_assets:
        if not isinstance(item, Mapping) or not isinstance(item.get("asset_id"), str):
            raise RunnerError("validation asset-verification entry is invalid")
        asset_id = item["asset_id"]
        if asset_id in verified_by_id:
            raise RunnerError(f"duplicate validation asset verification: {asset_id}")
        verified_by_id[asset_id] = item
    patient_asset = verified_by_id.get("patient_map_validation")
    if patient_asset is None or payload.get("validation_patient_map_sha256") != patient_asset.get(
        "sha256"
    ):
        raise RunnerError("lock validation-patient-map hash mismatch")

    rows = _read_case_rows(validation_csv_path)
    if sha256_file(validation_csv_path) != validation_hash:
        raise RunnerError("validation CSV changed during authentication")
    summaries = aggregate_validation_candidates(
        rows, expected_cases=expected_validation_cases
    )
    recomputed = select_all_targets(summaries, require_complete_grid=True)
    operating = recomputed["selected_operating_point"]
    if not _canonical_equal(payload.get("selected"), operating.get("selected")):
        raise RunnerError("lock selected point differs from frozen validation recomputation")
    expected_gate = {
        field: operating.get(field)
        for field in (
            "status",
            "claim_allowed",
            "psnr_floor_db",
            "fallback_used",
            "failure_reason",
        )
    }
    if not _canonical_equal(payload.get("operating_point_gate"), expected_gate):
        raise RunnerError("lock operating-point gate differs from validation recomputation")
    if not _canonical_equal(payload.get("residual_matching"), recomputed.get("target_definition")):
        raise RunnerError("lock residual targets differ from validation recomputation")

    scientific = sweep.get("scientific_config", {})
    if float(payload.get("common_scale")) != float(scientific.get("common_scale")):
        raise RunnerError("lock common scale differs from validation sweep")
    if float(payload.get("lambda_max_ata")) != float(scientific.get("lambda_max_ata")):
        raise RunnerError("lock power estimate differs from validation sweep")
    if tuple(payload.get("k_grid", ())) != tuple(ITERATIONS):
        raise RunnerError("lock iteration grid changed")

    checkpoint_map: dict[str, str] = {}
    for model in CANONICAL_MODELS:
        values = {
            row["checkpoint_sha256"]
            for row in rows
            if row["architecture"] == model
        }
        if len(values) != 1:
            raise RunnerError(f"validation checkpoint identity is not unique: {model}")
        checkpoint_map[model] = next(iter(values))
    if payload.get("checkpoint_sha256") != checkpoint_map:
        raise RunnerError("lock checkpoint map differs from validation sweep")
    for model, prefix in (
        ("UNet", "checkpoint_unet"),
        ("TransUNet", "checkpoint_transunet"),
        ("WavTransUNet", "checkpoint_wavtransunet"),
    ):
        asset = verified_by_id.get(f"{prefix}_{expected_views}")
        if asset is None or asset.get("sha256") != checkpoint_map[model]:
            raise RunnerError(
                f"validation checkpoint row differs from verified asset: {model}"
            )
    return lock, sweep


ARCHIVED_METRIC_COLUMNS = ("angle", "index", "model", "psnr", "ssim", "rmse")
ARCHIVED_PROJECTION_COLUMNS = ("angle", "index", "model", "r", "convention")
ARCHIVED_METRIC_MODELS = {
    "UNet": "UNet",
    "TransUNet": "TransUNet",
    "WavResTransUNet": "WavTransUNet",
}
ARCHIVED_METRIC_ALL_MODELS = tuple(ARCHIVED_METRIC_MODELS) + ("FBP_INPUT",)
ARCHIVED_PROJECTION_ALL_MODELS = CANONICAL_MODELS + ("FBP",)


def _strict_csv_rows(path: Path, expected_header: Sequence[str]) -> Iterable[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"archived canary input is not a regular file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(expected_header):
            raise RunnerError(
                f"archived CSV header changed for {path}: {reader.fieldnames!r}"
            )
        yield from reader


def _finite_archive_float(value: str, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"invalid archived {where}: {value!r}") from exc
    if not np.isfinite(result):
        raise RunnerError(f"nonfinite archived {where}")
    return result


def _load_archived_k0_references(
    metrics_csv: Path,
    projection_csv: Path,
    *,
    views: int,
    expected_cases: int = EXPECTED_COUNTS["test"],
) -> tuple[dict[tuple[int, str], dict[str, float]], dict[tuple[int, str], float], dict[str, Any]]:
    """Validate complete immutable legacy k=0 reference tables before DC."""

    metrics_sha256 = sha256_file(metrics_csv)
    projection_sha256 = sha256_file(projection_csv)
    metrics: dict[tuple[int, str], dict[str, float]] = {}
    metrics_rows = 0
    metrics_all_keys: set[tuple[int, str]] = set()
    for row in _strict_csv_rows(metrics_csv, ARCHIVED_METRIC_COLUMNS):
        try:
            angle = int(row["angle"])
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise RunnerError("archived metric angle/index is invalid") from exc
        if angle != views or not 0 <= index < expected_cases:
            raise RunnerError("archived metric view/index is outside the locked test set")
        raw_model = row["model"]
        if raw_model not in ARCHIVED_METRIC_ALL_MODELS:
            raise RunnerError(f"unknown archived metric model: {raw_model!r}")
        raw_key = (index, raw_model)
        if raw_key in metrics_all_keys:
            raise RunnerError(f"duplicate archived metric row: {raw_key}")
        metrics_all_keys.add(raw_key)
        model = ARCHIVED_METRIC_MODELS.get(raw_model)
        if model is None:
            for metric in ("psnr", "ssim", "rmse"):
                _finite_archive_float(row[metric], f"{metric} {raw_key}")
            continue
        key = (index, model)
        if key in metrics:
            raise RunnerError(f"duplicate archived metric row: {key}")
        values = {
            metric: _finite_archive_float(row[metric], f"{metric} {key}")
            for metric in ("psnr", "ssim", "rmse")
        }
        if not -1.0 <= values["ssim"] <= 1.0 or values["rmse"] < 0.0:
            raise RunnerError(f"archived metric values are outside their domains: {key}")
        metrics[key] = values
        metrics_rows += 1

    expected_keys = {
        (index, model)
        for index in range(expected_cases)
        for model in CANONICAL_MODELS
    }
    expected_metric_all_keys = {
        (index, model)
        for index in range(expected_cases)
        for model in ARCHIVED_METRIC_ALL_MODELS
    }
    if metrics_all_keys != expected_metric_all_keys:
        raise RunnerError(
            "archived metric table does not contain exactly four models per case"
        )
    if set(metrics) != expected_keys or metrics_rows != len(expected_keys):
        raise RunnerError(
            f"archived metric coverage mismatch: {metrics_rows} != {len(expected_keys)}"
        )

    residuals: dict[tuple[int, str], float] = {}
    projection_rows = 0
    projection_all_keys: set[tuple[int, str]] = set()
    for row in _strict_csv_rows(projection_csv, ARCHIVED_PROJECTION_COLUMNS):
        try:
            angle = int(row["angle"])
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise RunnerError("archived projection angle/index is invalid") from exc
        if angle != views or not 0 <= index < expected_cases:
            raise RunnerError("archived projection view/index is outside the locked test set")
        if row["convention"] != "fbp":
            raise RunnerError("archived projection convention must be fbp")
        model = row["model"]
        if model not in ARCHIVED_PROJECTION_ALL_MODELS:
            raise RunnerError(f"unknown archived projection model: {model!r}")
        raw_key = (index, model)
        if raw_key in projection_all_keys:
            raise RunnerError(f"duplicate archived projection row: {raw_key}")
        projection_all_keys.add(raw_key)
        value = _finite_archive_float(row["r"], f"projection residual {raw_key}")
        if value < 0.0:
            raise RunnerError(f"negative archived projection residual: {raw_key}")
        if model == "FBP":
            continue
        key = (index, model)
        residuals[key] = value
        projection_rows += 1
    expected_projection_all_keys = {
        (index, model)
        for index in range(expected_cases)
        for model in ARCHIVED_PROJECTION_ALL_MODELS
    }
    if projection_all_keys != expected_projection_all_keys:
        raise RunnerError(
            "archived projection table does not contain exactly four models per case"
        )
    if set(residuals) != expected_keys or projection_rows != len(expected_keys):
        raise RunnerError(
            f"archived projection coverage mismatch: {projection_rows} != {len(expected_keys)}"
        )
    if sha256_file(metrics_csv) != metrics_sha256:
        raise RunnerError("archived metric table changed during validation")
    if sha256_file(projection_csv) != projection_sha256:
        raise RunnerError("archived projection table changed during validation")
    evidence = {
        "metrics": {
            "path": str(metrics_csv.resolve()),
            "sha256": metrics_sha256,
            "header": list(ARCHIVED_METRIC_COLUMNS),
            "total_rows": len(metrics_all_keys),
            "model_case_rows": metrics_rows,
            "expected_model_case_rows": len(expected_keys),
        },
        "projection": {
            "path": str(projection_csv.resolve()),
            "sha256": projection_sha256,
            "header": list(ARCHIVED_PROJECTION_COLUMNS),
            "total_rows": len(projection_all_keys),
            "model_case_rows": projection_rows,
            "expected_model_case_rows": len(expected_keys),
            "convention": "fbp",
        },
    }
    return metrics, residuals, evidence


def _assert_k0_parity(
    observed_metrics: Mapping[tuple[int, str], Mapping[str, float]],
    observed_residuals: Mapping[tuple[int, str], float],
    archived_metrics: Mapping[tuple[int, str], Mapping[str, float]],
    archived_residuals: Mapping[tuple[int, str], float],
    *,
    # AMENDMENT 2 (2026-08-17, user-authorized; see AMENDMENT_2_k0_parity_tolerance.md):
    # the frozen bounds (1.0e-5 / 1.0e-8) were calibrated against same-bytes
    # reprojection reproducibility (observed 7.629e-6 / 1.86e-9). The frozen test
    # command instead re-infers reconstructions freshly, so cuDNN run-to-run
    # nondeterminism enters the k=0 values; the first formal run exceeded the
    # residual bound by 0.3% on one of 10,659 model-cases (abs_diff 1.00302e-8).
    # Both bounds are scaled x10 (1.0e-4 / 1.0e-7); a genuine checkpoint/data/
    # operator mismatch produces differences 4+ orders of magnitude larger, so
    # the gate's discriminative purpose is unchanged.
    metric_tolerance: float = 1.0e-4,
    residual_tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Require complete fresh-inference parity before any finite-DC update."""

    if set(observed_metrics) != set(archived_metrics):
        raise RunnerError("fresh/archived metric k=0 key sets differ")
    if set(observed_residuals) != set(archived_residuals):
        raise RunnerError("fresh/archived residual k=0 key sets differ")
    maxima = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "projection_residual": 0.0}
    for key in sorted(archived_metrics):
        for metric in ("psnr", "ssim", "rmse"):
            difference = abs(
                float(observed_metrics[key][metric])
                - float(archived_metrics[key][metric])
            )
            maxima[metric] = max(maxima[metric], difference)
            if difference > metric_tolerance:
                raise RunnerError(
                    f"k=0 {metric} parity failed: {key}, abs_diff={difference:.9g}"
                )
        difference = abs(
            float(observed_residuals[key]) - float(archived_residuals[key])
        )
        maxima["projection_residual"] = max(
            maxima["projection_residual"], difference
        )
        if difference > residual_tolerance:
            raise RunnerError(
                f"k=0 projection parity failed: {key}, abs_diff={difference:.9g}"
            )
    return {
        "status": "PASS",
        "cases": len(archived_metrics) // len(CANONICAL_MODELS),
        "models": list(CANONICAL_MODELS),
        "metric_tolerance": metric_tolerance,
        "projection_residual_tolerance": residual_tolerance,
        "max_abs_difference": maxima,
    }


def command_evaluate_test(args: argparse.Namespace) -> int:
    # This must remain the first file access in the test command.  It consumes
    # validation evidence only and independently authenticates the frozen
    # choice before any test input is resolved, hashed, or read.
    lock, sweep = _authenticate_lock_before_test(
        args.lock,
        args.sweep_manifest,
        args.validation_csv,
        expected_views=args.views,
    )
    out = _new_output_directory(args.out)
    path = out / "test_manifest.json"
    manifest = _started_manifest("evaluate-test", args.views, "test")
    _write_manifest(path, manifest)
    try:
        manifest["inputs"]["lock"] = {
            "path": str(args.lock.resolve()),
            "sha256": sha256_file(args.lock),
            "payload_sha256": lock["payload_sha256"],
        }
        manifest["inputs"]["authenticated_validation"] = {
            "sweep_manifest": {
                "path": str(args.sweep_manifest.resolve()),
                "sha256": sha256_file(args.sweep_manifest),
                "run_id": sweep["run_id"],
            },
            "case_metrics": {
                "path": str(args.validation_csv.resolve()),
                "sha256": sha256_file(args.validation_csv),
            },
            "selection_recomputed": True,
        }
        # Authenticate every test byte source (including the two historical
        # parity-oracle CSVs) before parsing those references.
        manifest["inputs"]["asset_verification"] = _verify_phase_assets(args, "test")
        archived_metrics, archived_residuals, archived_evidence = (
            _load_archived_k0_references(
                args.archived_metrics,
                args.archived_projection,
                views=args.views,
            )
        )
        manifest["inputs"]["archived_k0_references"] = archived_evidence
        torch, dival, dataset, operator = _import_physics(
            args.data, args.dival_root, args.views, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        manifest["canaries"]["operator"] = _operator_canaries(operator, args.views)
        manifest["canaries"]["validation_test_runtime_parity"] = (
            _assert_test_runtime_parity(
                sweep,
                lock,
                manifest["environment"],
                manifest["canaries"]["operator"],
            )
        )
        patients = _patient_ids(args.patient_map, EXPECTED_COUNTS["test"])
        fbp = np.load(args.fbp_cache, mmap_mode="r")
        if fbp.shape != (
            EXPECTED_COUNTS["test"],
            FULL_SIZE,
            FULL_SIZE,
        ) or fbp.dtype != np.float32:
            raise RunnerError(f"test FBP cache mismatch: {fbp.shape}, {fbp.dtype}")

        from dc_model_factory import build_and_load_archived_model, infer_raw_float32

        device = torch.device("cuda")
        checkpoint_paths = {
            "UNet": args.checkpoint_unet,
            "TransUNet": args.checkpoint_transunet,
            "WavTransUNet": args.checkpoint_wavtransunet,
        }
        checkpoint_asset_ids = {
            "UNet": f"checkpoint_unet_{args.views}",
            "TransUNet": f"checkpoint_transunet_{args.views}",
            "WavTransUNet": f"checkpoint_wavtransunet_{args.views}",
        }
        locked_checkpoint_hash = dict(lock["payload"]["checkpoint_sha256"])
        prediction_root = out / "predictions"
        prediction_root.mkdir()
        model_evidence: dict[str, Any] = {}
        manifest["inputs"]["model_evidence"] = model_evidence
        inputs = fbp[:, None, BORDER:-BORDER, BORDER:-BORDER]
        for model in CANONICAL_MODELS:
            built, build_report, load_report = build_and_load_archived_model(
                model,
                args.wtu_code,
                checkpoint_paths[model],
                expected_assets_path=EXPECTED_ASSETS_PATH,
                checkpoint_asset_id=checkpoint_asset_ids[model],
                views=args.views,
                device=device,
            )
            if load_report.checkpoint_sha256 != locked_checkpoint_hash.get(model):
                raise RunnerError(
                    f"freshly loaded checkpoint differs from validation lock: {model}"
                )
            destination = prediction_root / f"{model}.npy"
            memory = np.lib.format.open_memmap(
                destination,
                mode="w+",
                dtype=np.float32,
                shape=(EXPECTED_COUNTS["test"], 1, CROP_SIZE, CROP_SIZE),
            )
            _, inference_report = infer_raw_float32(
                built,
                inputs,
                output=memory,
                batch_size=FORMAL_TEST_INFERENCE_BATCH,
                device=device,
            )
            memory.flush()
            del memory
            model_evidence[model] = {
                "build": build_report.to_dict(),
                "load": load_report.to_dict(),
            }
            manifest["outputs"][f"predictions_{model}"] = {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "shape": [EXPECTED_COUNTS["test"], 1, CROP_SIZE, CROP_SIZE],
                "dtype": "float32",
                "inference_report": inference_report.to_dict(),
                "checkpoint_sha256": load_report.checkpoint_sha256,
                "post_load_state_sha256": load_report.post_load_state_sha256,
            }
            del built
            torch.cuda.empty_cache()
        manifest["outputs"]["prediction_root"] = {
            "path": str(prediction_root.resolve())
        }
        arrays = {
            model: np.load(prediction_root / f"{model}.npy", mmap_mode="r")
            for model in CANONICAL_MODELS
        }

        # Complete fresh-inference parity must pass before the first finite-DC
        # trajectory.  This second test pass is deliberate and cannot tune any
        # operating parameter.
        observed_metrics: dict[tuple[int, str], dict[str, float]] = {}
        observed_residuals: dict[tuple[int, str], float] = {}
        for index, (y_obs, gt) in _iter_split(
            dataset, "test", EXPECTED_COUNTS["test"]
        ):
            y = np.asarray(y_obs, dtype=np.float32)
            gt362 = np.asarray(gt, dtype=np.float32)
            for model in CANONICAL_MODELS:
                key = (index, model)
                canvas = embed_crop(
                    _prediction_crop(arrays[model], index), fbp[index]
                )
                current = _metrics_one(canvas, gt362, "cuda")
                observed_metrics[key] = {
                    "psnr": current["psnr_db"],
                    "ssim": current["ssim"],
                    "rmse": current["rmse"],
                }
                observed_residuals[key] = projection_residual(operator, canvas, y)
        manifest["canaries"]["k0_parity"] = _assert_k0_parity(
            observed_metrics,
            observed_residuals,
            archived_metrics,
            archived_residuals,
        )
        del archived_metrics, archived_residuals

        selected = lock["payload"]["selected"]
        multiplier = float(selected["lambda_multiplier"])
        lam = float(selected["lambda"])
        eta = float(selected["eta"])
        checkpoint_hash = locked_checkpoint_hash
        output_partial = out / "test_case_metrics.csv.partial"
        with output_partial.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CASE_COLUMNS)
            writer.writeheader()
            processed = 0
            for index, (y_obs, gt) in _iter_split(
                dataset, "test", EXPECTED_COUNTS["test"]
            ):
                y = np.asarray(y_obs, dtype=np.float32)
                gt362 = np.asarray(gt, dtype=np.float32)
                for model in CANONICAL_MODELS:
                    canvas = embed_crop(_prediction_crop(arrays[model], index), fbp[index])
                    trajectory = finite_dc_trajectory(
                        operator,
                        canvas,
                        fbp[index],
                        y,
                        regularization=lam,
                        step_size=eta,
                        iterations=ITERATIONS,
                    )
                    if trajectory.diverged:
                        raise RunnerError(
                            f"locked test trajectory diverged: {model}, case={index}, "
                            f"{trajectory.divergence_reason}"
                        )
                    for k in ITERATIONS:
                        if k == 0:
                            reference = observed_metrics[(index, model)]
                            metrics = {
                                "psnr_db": reference["psnr"],
                                "ssim": reference["ssim"],
                                "rmse": reference["rmse"],
                            }
                            residual = observed_residuals[(index, model)]
                        else:
                            metrics = _metrics_one(
                                trajectory.snapshots[k], gt362, "cuda"
                            )
                            residual = trajectory.residuals[k]
                        case_row = _row(
                                split="test",
                                case_id=index,
                                patient_id=int(patients[index]),
                                views=args.views,
                                architecture=model,
                                checkpoint_sha256=checkpoint_hash[model],
                                k=k,
                                lam=lam,
                                multiplier=multiplier,
                                eta=eta,
                                metrics=metrics,
                                residual=residual,
                                diverged=False,
                                selected=(k == int(selected["k"])),
                                backend=args.backend,
                                run_id=manifest["run_id"],
                            )
                        writer.writerow(case_row)
                processed += 1
                if processed % 100 == 0:
                    stream.flush()
        output = out / "test_case_metrics.csv"
        os.replace(output_partial, output)
        expected_rows = (
            EXPECTED_COUNTS["test"]
            * len(CANONICAL_MODELS)
            * len(ITERATIONS)
        )
        with output.open(encoding="utf-8") as stream:
            observed_rows = sum(1 for _ in stream) - 1
        if observed_rows != expected_rows:
            raise RunnerError(f"test output rows {observed_rows} != {expected_rows}")
        manifest["outputs"]["case_metrics"] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "rows": observed_rows,
        }
        manifest["coverage"] = {"cases": processed, "rows": observed_rows, "models": 3}
        _complete_manifest(path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(path, manifest, exc)
        raise

def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{where} must be an object")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_qualitative_cases(
    grid: Mapping[tuple[int, str, int], Mapping[str, Any]],
    selected_k: int,
) -> list[dict[str, Any]]:
    """Recompute the prespecified q10/q50/q90 cases without image access."""

    effects: list[tuple[float, int]] = []
    for case_id in range(EXPECTED_COUNTS["test"]):
        values = {
            model: float(grid[(case_id, model, selected_k)]["psnr"])
            for model in CANONICAL_MODELS
        }
        effects.append(
            (
                values["WavTransUNet"]
                - max(values["UNet"], values["TransUNet"]),
                case_id,
            )
        )
    effects.sort(key=lambda item: (item[0], item[1]))
    result: list[dict[str, Any]] = []
    for label, quantile in QUALITATIVE_CASE_LABELS:
        rank = int(round(quantile * (len(effects) - 1)))
        effect, case_id = effects[rank]
        result.append(
            {
                "label": label,
                "quantile": quantile,
                "rank": rank,
                "case_id": case_id,
                "delta_psnr_vs_stronger_dc_baseline": effect,
            }
        )
    return result


def _authenticate_case_lock_for_export(
    *,
    test_manifest_path: Path,
    test_csv: Path,
    lock_path: Path,
    case_lock_path: Path,
    protocol_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Any],
    dict[tuple[int, str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Authenticate and independently reproduce the outcome-locked cases."""

    from dc_qualitative import (
        _authenticate_test_manifest,
        _load_lock_and_protocol,
        _read_authenticated_test_rows,
    )

    lock, protocol, selected = _load_lock_and_protocol(lock_path, protocol_path)
    manifest, run_id, expected_rows = _authenticate_test_manifest(
        test_manifest_path, test_csv, lock_path, lock, protocol
    )
    payload = _mapping(lock.get("payload"), "lock.payload")
    operating_gate = _mapping(
        payload.get("operating_point_gate"), "lock.payload.operating_point_gate"
    )
    gate_label = (
        "validation-selected claim-eligible operating point"
        if operating_gate.get("claim_allowed") is True
        else "diagnostic fallback; claim-ineligible"
    )
    grid = _read_authenticated_test_rows(
        test_csv,
        run_id=run_id,
        expected_rows=expected_rows,
        lock_payload=payload,
    )
    if case_lock_path.is_symlink() or not case_lock_path.is_file():
        raise RunnerError("case lock must be a regular non-symlink file")
    case_lock = load_json(case_lock_path)
    required = {
        "schema": "dc.case_lock.v1",
        "status": "LOCKED",
        "view": 50,
        "selected_k": int(selected["k"]),
        "models": list(CANONICAL_MODELS),
        "k_grid": list(ITERATIONS),
        "authenticated_test_rows": expected_rows,
        "test_csv_sha256": sha256_file(test_csv),
        "test_manifest_sha256": sha256_file(test_manifest_path),
        "test_run_id": run_id,
        "dc_lock_sha256": sha256_file(lock_path),
        "dc_lock_payload_sha256": lock["payload_sha256"],
        "protocol_sha256": json_sha256(protocol),
        "selection_rule": (
            "nearest ranks to q10/q50/q90 of W+DC PSNR minus "
            "max(U+DC,T+DC) PSNR; case-ID tie-break"
        ),
        "operating_point_gate": dict(operating_gate),
        "operating_point_label": gate_label,
        "interpretation": "descriptive, outcome-locked before image inspection",
        "image_exported": False,
    }
    for field, expected in required.items():
        if not _canonical_equal(case_lock.get(field), expected):
            raise RunnerError(f"case lock {field} differs from authenticated evidence")
    expected_cases = _expected_qualitative_cases(grid, int(selected["k"]))
    if not _canonical_equal(case_lock.get("cases"), expected_cases):
        raise RunnerError("case lock IDs differ from deterministic q10/q50/q90 recomputation")
    if len({item["case_id"] for item in expected_cases}) != 3:
        raise RunnerError("deterministic q10/q50/q90 case IDs are not unique")
    return lock, manifest, selected, grid, expected_cases


def _authenticated_test_prediction_assets(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[Path, dict[str, Path], dict[str, dict[str, Any]]]:
    """Bind export inputs only to fresh predictions/checkpoint state evidence."""

    inputs = _mapping(manifest.get("inputs"), "test manifest inputs")
    outputs = _mapping(manifest.get("outputs"), "test manifest outputs")
    lock_payload = _mapping(lock.get("payload"), "lock.payload")
    checkpoint_map = _mapping(
        lock_payload.get("checkpoint_sha256"), "lock.payload.checkpoint_sha256"
    )
    if set(checkpoint_map) != set(CANONICAL_MODELS):
        raise RunnerError("locked checkpoint map is not canonical")

    verification = _mapping(
        inputs.get("asset_verification"), "test manifest asset_verification"
    )
    verified = verification.get("verified_assets")
    if not isinstance(verified, list):
        raise RunnerError("test manifest verified-assets list is missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in verified:
        entry = _mapping(item, "test manifest verified asset")
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or asset_id in by_id:
            raise RunnerError("test manifest verified-assets IDs are invalid")
        by_id[asset_id] = entry

    fbp_entry = by_id.get("fbp_test_50")
    if fbp_entry is None or fbp_entry.get("status") != "VERIFIED":
        raise RunnerError("authenticated 50-view test FBP asset is missing")
    fbp_path_value = fbp_entry.get("path")
    if not isinstance(fbp_path_value, str):
        raise RunnerError("authenticated test FBP path is missing")
    fbp_path = Path(fbp_path_value)
    if sha256_file(fbp_path) != fbp_entry.get("sha256"):
        raise RunnerError("test FBP cache differs from authenticated manifest")

    root_entry = _mapping(outputs.get("prediction_root"), "test prediction root")
    root_value = root_entry.get("path")
    if not isinstance(root_value, str):
        raise RunnerError("test prediction-root path is missing")
    root_path = Path(root_value)
    if root_path.is_symlink() or not root_path.is_dir():
        raise RunnerError("test prediction root must be a regular directory")
    prediction_root = root_path.resolve()
    model_evidence = _mapping(inputs.get("model_evidence"), "test model evidence")
    paths: dict[str, Path] = {}
    evidence: dict[str, dict[str, Any]] = {}
    asset_prefix = {
        "UNet": "checkpoint_unet_50",
        "TransUNet": "checkpoint_transunet_50",
        "WavTransUNet": "checkpoint_wavtransunet_50",
    }
    for model in CANONICAL_MODELS:
        expected_checkpoint = checkpoint_map[model]
        if not _valid_sha256(expected_checkpoint):
            raise RunnerError(f"invalid locked checkpoint SHA-256: {model}")
        output = _mapping(
            outputs.get(f"predictions_{model}"), f"test prediction output {model}"
        )
        path_value = output.get("path")
        if not isinstance(path_value, str):
            raise RunnerError(f"test prediction path is missing: {model}")
        path = Path(path_value)
        if path.resolve().parent != prediction_root:
            raise RunnerError(f"test prediction escapes authenticated root: {model}")
        if output.get("shape") != [
            EXPECTED_COUNTS["test"],
            1,
            CROP_SIZE,
            CROP_SIZE,
        ] or output.get("dtype") != "float32":
            raise RunnerError(f"test prediction schema differs: {model}")
        if sha256_file(path) != output.get("sha256"):
            raise RunnerError(f"test prediction bytes differ from manifest: {model}")

        inference = _mapping(
            output.get("inference_report"), f"test inference report {model}"
        )
        expected_inference = {
            "cases": EXPECTED_COUNTS["test"],
            "input_dtype": "float32",
            "output_dtype": "float32",
            "normalized": False,
            "clipped": False,
            "case_order_preserved": True,
            "exact_coverage": True,
        }
        for field, expected in expected_inference.items():
            if inference.get(field) != expected:
                raise RunnerError(f"test inference report {field} differs: {model}")

        model_record = _mapping(model_evidence.get(model), f"model evidence {model}")
        build = _mapping(model_record.get("build"), f"model build evidence {model}")
        load = _mapping(model_record.get("load"), f"model load evidence {model}")
        if build.get("architecture") != model:
            raise RunnerError(f"model build architecture differs: {model}")
        post_state = load.get("post_load_state_sha256")
        if not _valid_sha256(post_state):
            raise RunnerError(f"post-load state hash is invalid: {model}")
        if (
            load.get("checkpoint_sha256") != expected_checkpoint
            or output.get("checkpoint_sha256") != expected_checkpoint
            or output.get("post_load_state_sha256") != post_state
            or load.get("strict") is not True
            or load.get("missing_keys") not in ([], ())
            or load.get("unexpected_keys") not in ([], ())
        ):
            raise RunnerError(f"checkpoint/state evidence differs: {model}")
        checkpoint_path_value = load.get("checkpoint_path")
        if not isinstance(checkpoint_path_value, str):
            raise RunnerError(f"checkpoint evidence path is missing: {model}")
        checkpoint_path = Path(checkpoint_path_value)
        if sha256_file(checkpoint_path) != expected_checkpoint:
            raise RunnerError(f"checkpoint bytes differ from locked identity: {model}")
        asset = by_id.get(asset_prefix[model])
        if asset is None or asset.get("status") != "VERIFIED":
            raise RunnerError(f"verified checkpoint asset is missing: {model}")
        if (
            asset.get("sha256") != expected_checkpoint
            or not isinstance(asset.get("path"), str)
            or Path(str(asset["path"])).resolve() != checkpoint_path.resolve()
        ):
            raise RunnerError(f"verified checkpoint asset differs: {model}")

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (
            EXPECTED_COUNTS["test"],
            1,
            CROP_SIZE,
            CROP_SIZE,
        ) or array.dtype != np.float32:
            raise RunnerError(f"test prediction array differs from manifest: {model}")
        paths[model] = path
        evidence[model] = {
            "prediction_path": str(path.resolve()),
            "prediction_sha256": output["sha256"],
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": expected_checkpoint,
            "post_load_state_sha256": post_state,
            "strict_checkpoint_load": True,
            "raw_float32_inference": True,
            "normalized": False,
            "clipped": False,
        }
    return fbp_path, paths, evidence


def _assert_qualitative_metric_parity(
    observed: Mapping[str, float],
    archived: Mapping[str, Any],
    *,
    model: str,
    case_id: int,
    k: int,
    metric_tolerance: float = 1.0e-5,
    residual_tolerance: float = 1.0e-8,
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for observed_name, archived_name, tolerance in (
        ("psnr_db", "psnr", metric_tolerance),
        ("ssim", "ssim", metric_tolerance),
        ("rmse", "rmse", metric_tolerance),
        ("projection_residual", "projection_residual", residual_tolerance),
    ):
        difference = abs(float(observed[observed_name]) - float(archived[archived_name]))
        maxima[archived_name] = difference
        if difference > tolerance:
            raise RunnerError(
                f"qualitative export parity failed: case={case_id}, model={model}, "
                f"k={k}, metric={archived_name}, abs_diff={difference:.9g}"
            )
    return maxima


def _assert_export_operator_identity(
    test_manifest: Mapping[str, Any],
    observed_environment: Mapping[str, Any],
    observed_operator: Mapping[str, Any],
) -> None:
    """Require the same formal software/backend/operator contract as test."""

    expected_environment = _mapping(
        test_manifest.get("environment"), "test environment"
    )
    for field in ("dival", "odl", "astra", "backend"):
        if observed_environment.get(field) != expected_environment.get(field):
            raise RunnerError(f"qualitative export {field} differs from formal test")
    test_canaries = _mapping(test_manifest.get("canaries"), "test canaries")
    expected_operator = _mapping(test_canaries.get("operator"), "test operator canary")
    for field, expected in (
        ("domain_shape", [FULL_SIZE, FULL_SIZE]),
        ("range_shape", [50, 513]),
    ):
        if expected_operator.get(field) != expected or observed_operator.get(field) != expected:
            raise RunnerError(f"qualitative export operator {field} differs")
    try:
        expected_power = float(expected_operator.get("lambda_max_ata"))
        observed_power = float(observed_operator.get("lambda_max_ata"))
    except (TypeError, ValueError) as exc:
        raise RunnerError("qualitative export operator power canary is invalid") from exc
    if not np.isfinite(expected_power) or not np.isfinite(observed_power):
        raise RunnerError("qualitative export operator power canary is nonfinite")
    if not np.isclose(expected_power, observed_power, rtol=1.0e-6, atol=0.0):
        raise RunnerError("qualitative export operator power canary differs from formal test")


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _write_qualitative_outputs(
    out: Path,
    case_payloads: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically publish three array bundles, metrics, and a hash inventory."""

    if len(case_payloads) != 3 or len(metric_rows) != 18:
        raise RunnerError("qualitative export requires exactly 3 cases and 18 metric rows")
    expected_labels = [item[0] for item in QUALITATIVE_CASE_LABELS]
    if [item.get("label") for item in case_payloads] != expected_labels:
        raise RunnerError("qualitative case payload order differs from q10/q50/q90 lock")
    expected_array_shapes = {
        "ground_truth": (CROP_SIZE, CROP_SIZE),
        "fbp_canvas": (FULL_SIZE, FULL_SIZE),
        "measured_sinogram": (50, 513),
        **{key: (CROP_SIZE, CROP_SIZE) for key in QUALITATIVE_PANEL_KEYS},
        **{
            f"error_{key}": (CROP_SIZE, CROP_SIZE)
            for key in QUALITATIVE_PANEL_KEYS
        },
    }
    expected_metric_keys = {
        (int(payload["case_id"]), panel_key)
        for payload in case_payloads
        for panel_key in QUALITATIVE_PANEL_KEYS
    }
    observed_metric_keys: set[tuple[int, str]] = set()
    observed_gate_labels: set[tuple[str, bool, bool]] = set()
    for row in metric_rows:
        key = (int(row["case_id"]), str(row["panel_key"]))
        if key in observed_metric_keys:
            raise RunnerError(f"duplicate qualitative metric row: {key}")
        observed_metric_keys.add(key)
        if row.get("clipped") is not False or row.get("normalized") is not False:
            raise RunnerError("qualitative metrics must remain raw and unclipped")
        status = row.get("operating_point_status")
        claim_allowed = row.get("operating_point_claim_allowed")
        diagnostic_fallback = row.get("diagnostic_fallback")
        if (
            status not in ("PASS", "FAIL")
            or type(claim_allowed) is not bool
            or type(diagnostic_fallback) is not bool
            or (status == "PASS") != claim_allowed
            or diagnostic_fallback == claim_allowed
        ):
            raise RunnerError("qualitative operating-point gate label is inconsistent")
        observed_gate_labels.add((status, claim_allowed, diagnostic_fallback))
        for field in ("psnr", "ssim", "rmse", "projection_residual"):
            try:
                finite = np.isfinite(float(row[field]))
            except (KeyError, TypeError, ValueError) as exc:
                raise RunnerError(f"invalid qualitative metric {field}: {key}") from exc
            if not finite:
                raise RunnerError(f"nonfinite qualitative metric {field}: {key}")
    if observed_metric_keys != expected_metric_keys:
        raise RunnerError("qualitative metric rows do not cover all six locked panels")
    if len(observed_gate_labels) != 1:
        raise RunnerError("qualitative metric rows mix operating-point gate labels")
    files: dict[str, Any] = {}
    for payload in case_payloads:
        label = str(payload["label"])
        case_id = int(payload["case_id"])
        arrays = _mapping(payload.get("arrays"), f"qualitative arrays {label}")
        if set(arrays) != set(expected_array_shapes):
            raise RunnerError(f"qualitative array schema differs: {label}")
        canonical_arrays: dict[str, np.ndarray] = {}
        for key, expected_shape in expected_array_shapes.items():
            value = np.asarray(arrays[key])
            if value.shape != expected_shape or value.dtype != np.float32:
                raise RunnerError(
                    f"qualitative array shape/dtype differs: {label}/{key}"
                )
            if not np.all(np.isfinite(value)):
                raise RunnerError(f"qualitative array is nonfinite: {label}/{key}")
            canonical_arrays[key] = value
        path = out / f"case_{label}_{case_id:04d}.npz"
        _atomic_write_npz(path, canonical_arrays)
        with np.load(path, allow_pickle=False) as stored:
            inventory = {
                key: {"shape": list(stored[key].shape), "dtype": str(stored[key].dtype)}
                for key in sorted(stored.files)
            }
        files[label] = {
            "case_id": case_id,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "arrays": inventory,
        }

    metrics_partial = out / "qualitative_case_metrics.csv.partial"
    with metrics_partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALITATIVE_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(metric_rows)
        stream.flush()
        os.fsync(stream.fileno())
    metrics_path = out / "qualitative_case_metrics.csv"
    os.replace(metrics_partial, metrics_path)

    hash_partial = out / "SHA256SUMS.csv.partial"
    hash_columns = ("artifact", "sha256")
    with hash_partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=hash_columns)
        writer.writeheader()
        for item in files.values():
            writer.writerow({"artifact": Path(item["path"]).name, "sha256": item["sha256"]})
        writer.writerow(
            {"artifact": metrics_path.name, "sha256": sha256_file(metrics_path)}
        )
        stream.flush()
        os.fsync(stream.fileno())
    hash_path = out / "SHA256SUMS.csv"
    os.replace(hash_partial, hash_path)
    return {
        "case_arrays": files,
        "case_metrics": {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
            "rows": len(metric_rows),
            "header": list(QUALITATIVE_METRIC_COLUMNS),
        },
        "hash_inventory": {
            "path": str(hash_path.resolve()),
            "sha256": sha256_file(hash_path),
            "rows": len(files) + 1,
        },
    }


def command_export_cases(args: argparse.Namespace) -> int:
    # Authenticate every pre-existing artifact, including deterministic case
    # selection and fresh checkpoint-derived predictions, before image access
    # or output-directory creation.
    test_manifest = _require_complete_manifest(args.test_manifest, "evaluate-test")
    if test_manifest.get("views") != 50 or test_manifest.get("split") != "test":
        raise RunnerError("qualitative export is frozen to the 50-view test analysis")
    lock, authenticated_manifest, selected, grid, selected_cases = (
        _authenticate_case_lock_for_export(
            test_manifest_path=args.test_manifest,
            test_csv=args.test_csv,
            lock_path=args.lock,
            case_lock_path=args.case_lock,
            protocol_path=args.protocol,
        )
    )
    if not _canonical_equal(test_manifest, authenticated_manifest):
        raise RunnerError("independent test-manifest authentication disagrees")
    fbp_path, prediction_paths, checkpoint_evidence = (
        _authenticated_test_prediction_assets(test_manifest, lock)
    )
    canary = _mapping(test_manifest.get("canaries"), "test manifest canaries")
    k0_canary = _mapping(canary.get("k0_parity"), "test manifest k0 parity")
    if (
        k0_canary.get("status") != "PASS"
        or k0_canary.get("cases") != EXPECTED_COUNTS["test"]
        or k0_canary.get("models") != list(CANONICAL_MODELS)
    ):
        raise RunnerError("COMPLETE test manifest lacks the full k=0 parity canary")
    environment = _mapping(test_manifest.get("environment"), "test environment")
    if environment.get("backend") != args.backend:
        raise RunnerError("export backend differs from the authenticated test backend")
    if {str(row["backend"]) for row in grid.values()} != {args.backend}:
        raise RunnerError("authenticated test metric grid backend is not canonical")
    operating_gate = _mapping(
        lock["payload"].get("operating_point_gate"),
        "lock.payload.operating_point_gate",
    )
    diagnostic_fallback = operating_gate.get("claim_allowed") is False

    out = _new_output_directory(args.out)
    manifest_path = out / "qualitative_export_manifest.json"
    manifest = _started_manifest("export-cases", 50, "test")
    _write_manifest(manifest_path, manifest)
    try:
        manifest["inputs"] = {
            "test_manifest": {
                "path": str(args.test_manifest.resolve()),
                "sha256": sha256_file(args.test_manifest),
                "run_id": test_manifest["run_id"],
            },
            "test_case_metrics": {
                "path": str(args.test_csv.resolve()),
                "sha256": sha256_file(args.test_csv),
            },
            "dc_lock": {
                "path": str(args.lock.resolve()),
                "sha256": sha256_file(args.lock),
                "payload_sha256": lock["payload_sha256"],
            },
            "case_lock": {
                "path": str(args.case_lock.resolve()),
                "sha256": sha256_file(args.case_lock),
            },
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
                "canonical_json_sha256": test_manifest["protocol_sha256"],
            },
            "fbp_cache": {
                "path": str(fbp_path.resolve()),
                "sha256": sha256_file(fbp_path),
            },
            "fresh_prediction_and_checkpoint_evidence": checkpoint_evidence,
        }
        torch, dival, dataset, operator = _import_physics(
            args.data, args.dival_root, 50, backend=args.backend
        )
        manifest["environment"] = _environment(torch, dival, args.backend)
        manifest["canaries"]["operator"] = _operator_canaries(operator, 50)
        _assert_export_operator_identity(
            test_manifest,
            manifest["environment"],
            manifest["canaries"]["operator"],
        )
        fbp = np.load(fbp_path, mmap_mode="r", allow_pickle=False)
        if fbp.shape != (
            EXPECTED_COUNTS["test"],
            FULL_SIZE,
            FULL_SIZE,
        ) or fbp.dtype != np.float32:
            raise RunnerError("authenticated test FBP array shape/dtype changed")
        predictions = {
            model: np.load(path, mmap_mode="r", allow_pickle=False)
            for model, path in prediction_paths.items()
        }
        selected_k = int(selected["k"])
        lam = float(selected["lambda"])
        multiplier = float(selected["lambda_multiplier"])
        eta = float(selected["eta"])
        checkpoint_map = _mapping(
            lock["payload"].get("checkpoint_sha256"), "lock checkpoint map"
        )

        case_payloads: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        parity_maxima = {
            "psnr": 0.0,
            "ssim": 0.0,
            "rmse": 0.0,
            "projection_residual": 0.0,
        }
        for case in selected_cases:
            case_id = int(case["case_id"])
            y_obs, gt = dataset.get_sample(case_id, part="test")
            y = np.asarray(y_obs, dtype=np.float32)
            gt362 = np.asarray(gt, dtype=np.float32)
            if y.shape != (50, 513) or gt362.shape != (FULL_SIZE, FULL_SIZE):
                raise RunnerError(f"selected test sample has unexpected shape: {case_id}")
            arrays: dict[str, np.ndarray] = {
                "ground_truth": np.asarray(crop_canvas(gt362), dtype=np.float32),
                "fbp_canvas": np.asarray(fbp[case_id], dtype=np.float32),
                "measured_sinogram": y,
            }
            panel_keys = {
                "UNet": ("unet", "unet_finite_dc"),
                "TransUNet": ("transunet", "transunet_finite_dc"),
                "WavTransUNet": ("wtransunet", "wtransunet_finite_dc"),
            }
            for model in CANONICAL_MODELS:
                k0_canvas = embed_crop(
                    _prediction_crop(predictions[model], case_id), fbp[case_id]
                )
                trajectory = finite_dc_trajectory(
                    operator,
                    k0_canvas,
                    fbp[case_id],
                    y,
                    regularization=lam,
                    step_size=eta,
                    iterations=(0, selected_k),
                )
                if trajectory.diverged or selected_k not in trajectory.snapshots:
                    raise RunnerError(
                        f"selected qualitative trajectory diverged: {model}, case={case_id}"
                    )
                for phase, k, canvas, panel_key in (
                    ("k0", 0, k0_canvas, panel_keys[model][0]),
                    (
                        "finite_dc",
                        selected_k,
                        trajectory.snapshots[selected_k],
                        panel_keys[model][1],
                    ),
                ):
                    metrics = _metrics_one(canvas, gt362, "cuda")
                    residual = (
                        projection_residual(operator, canvas, y)
                        if k == 0
                        else float(trajectory.residuals[selected_k])
                    )
                    observed = dict(metrics)
                    observed["projection_residual"] = residual
                    differences = _assert_qualitative_metric_parity(
                        observed,
                        grid[(case_id, model, k)],
                        model=model,
                        case_id=case_id,
                        k=k,
                    )
                    for name, difference in differences.items():
                        parity_maxima[name] = max(parity_maxima[name], difference)
                    crop = np.asarray(crop_canvas(canvas), dtype=np.float32)
                    arrays[panel_key] = crop
                    arrays[f"error_{panel_key}"] = np.asarray(
                        crop - arrays["ground_truth"], dtype=np.float32
                    )
                    metric_rows.append(
                        {
                            "schema": "dc.qualitative_case_metric.v1",
                            "case_label": case["label"],
                            "quantile": case["quantile"],
                            "rank": case["rank"],
                            "case_id": case_id,
                            "patient_id": grid[(case_id, model, k)]["patient_id"],
                            "panel_key": panel_key,
                            "architecture": model,
                            "phase": phase,
                            "dc_iteration": k,
                            "checkpoint_sha256": checkpoint_map[model],
                            "post_load_state_sha256": checkpoint_evidence[model][
                                "post_load_state_sha256"
                            ],
                            "lambda": lam if k else 0.0,
                            "lambda_multiplier": multiplier if k else 0.0,
                            "eta": eta if k else 0.0,
                            "psnr": metrics["psnr_db"],
                            "ssim": metrics["ssim"],
                            "rmse": metrics["rmse"],
                            "projection_residual": residual,
                            "clipped": False,
                            "normalized": False,
                            "backend": args.backend,
                            "source_test_run_id": test_manifest["run_id"],
                            "operating_point_status": operating_gate["status"],
                            "operating_point_claim_allowed": operating_gate[
                                "claim_allowed"
                            ],
                            "diagnostic_fallback": diagnostic_fallback,
                        }
                    )
            case_payloads.append(
                {"label": case["label"], "case_id": case_id, "arrays": arrays}
            )

        manifest["outputs"] = _write_qualitative_outputs(
            out, case_payloads, metric_rows
        )
        manifest["coverage"] = {
            "cases": 3,
            "reconstruction_panels_per_case": 6,
            "metric_rows": 18,
            "case_labels": [item["label"] for item in selected_cases],
            "case_ids": [item["case_id"] for item in selected_cases],
        }
        manifest["scientific_config"] = {
            "selection": "deterministic q10/q50/q90 case lock; no override",
            "operating_point_gate": dict(operating_gate),
            "operating_point_label": (
                "validation-selected claim-eligible operating point"
                if not diagnostic_fallback
                else "diagnostic fallback; claim-ineligible"
            ),
            "selected_cases": selected_cases,
            "panel_order": [
                "ground_truth",
                "unet",
                "unet_finite_dc",
                "transunet",
                "transunet_finite_dc",
                "wtransunet",
                "wtransunet_finite_dc",
            ],
            "selected_finite_dc": {
                "k": selected_k,
                "lambda": lam,
                "lambda_multiplier": multiplier,
                "eta": eta,
            },
            "array_domain": "raw float32 LoDoPaB normalized image domain",
            "image_crop": [BORDER, BORDER + CROP_SIZE, BORDER, BORDER + CROP_SIZE],
            "metric_definition": metric_definition(),
            "data_consistency_definition": dc_definition(),
            "reconstruction_clipping": False,
            "reconstruction_normalization": False,
            "display_windowing_applied": False,
            "error_map_definition": "reconstruction minus ground_truth in raw crop domain",
            "interactive_case_selection": False,
        }
        manifest["canaries"]["selected_row_parity"] = {
            "status": "PASS",
            "rows": 18,
            "metric_tolerance": 1.0e-5,
            "projection_residual_tolerance": 1.0e-8,
            "max_abs_difference": parity_maxima,
        }
        _complete_manifest(manifest_path, manifest)
        return 0
    except BaseException as exc:
        _fail_manifest(manifest_path, manifest, exc)
        raise


def _add_common_assets(parser: argparse.ArgumentParser, *, split: str) -> None:
    parser.add_argument("--views", type=int, required=True, choices=VIEWS)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dival-root", type=Path, required=True)
    parser.add_argument("--wtu-code", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--fbp-cache", type=Path, required=True)
    parser.add_argument("--patient-map", type=Path, required=True)
    parser.add_argument("--checkpoint-unet", type=Path, required=True)
    parser.add_argument("--checkpoint-transunet", type=Path, required=True)
    parser.add_argument("--checkpoint-wavtransunet", type=Path, required=True)
    parser.add_argument("--pretrained-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("astra_cuda",), default="astra_cuda")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight-validation")
    _add_common_assets(preflight, split="validation")
    preflight.set_defaults(function=command_preflight_validation)

    smoke = commands.add_parser("smoke-validation")
    _add_common_assets(smoke, split="validation")
    smoke.add_argument("--preflight-manifest", type=Path, required=True)
    smoke.add_argument("--batch", type=int, default=8)
    smoke.add_argument("--n", type=int, default=32)
    smoke.set_defaults(function=command_smoke_validation)

    pilot = commands.add_parser("pilot-dc")
    _add_common_assets(pilot, split="validation")
    pilot.add_argument("--smoke-manifest", type=Path, required=True)
    pilot.add_argument("--n", type=int, default=32)
    pilot.set_defaults(function=command_pilot_dc)

    prepare = commands.add_parser("prepare-validation")
    _add_common_assets(prepare, split="validation")
    prepare.add_argument("--preflight-manifest", type=Path, required=True)
    prepare.add_argument("--batch", type=int, default=32)
    prepare.set_defaults(function=command_prepare_validation)

    sweep = commands.add_parser("sweep-validation")
    _add_common_assets(sweep, split="validation")
    sweep.add_argument("--prepared-manifest", type=Path, required=True)
    sweep.set_defaults(function=command_sweep_validation)

    lock = commands.add_parser("lock-protocol")
    lock.add_argument("--sweep-manifest", type=Path, required=True)
    lock.add_argument("--patient-map", type=Path, required=True)
    lock.add_argument("--out", type=Path, required=True)
    lock.set_defaults(function=command_lock_protocol)

    test = commands.add_parser("evaluate-test")
    # Lock is syntactically first and authenticated before any test binding.
    test.add_argument("--lock", type=Path, required=True)
    test.add_argument("--sweep-manifest", type=Path, required=True)
    test.add_argument("--validation-csv", type=Path, required=True)
    _add_common_assets(test, split="test")
    test.add_argument("--archived-metrics", type=Path, required=True)
    test.add_argument("--archived-projection", type=Path, required=True)
    test.set_defaults(
        function=command_evaluate_test,
        batch=FORMAL_TEST_INFERENCE_BATCH,
    )

    export = commands.add_parser("export-cases")
    export.add_argument("--test-manifest", type=Path, required=True)
    export.add_argument("--test-csv", type=Path, required=True)
    export.add_argument("--lock", type=Path, required=True)
    export.add_argument("--case-lock", type=Path, required=True)
    export.add_argument("--protocol", type=Path, required=True)
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--dival-root", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--backend", choices=("astra_cuda",), default="astra_cuda")
    export.set_defaults(function=command_export_cases)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
