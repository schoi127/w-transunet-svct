#!/usr/bin/env python3
"""Exact fixed-checkpoint model construction for the DC/Pareto study.

This module intentionally has no top-level PyTorch import.  Importing it is safe
for manifest and lock-building code; PyTorch and the audited W-TransUNet source
modules are imported only when a model or tensor operation is requested.

Models are constructed from an *explicit* W-TransUNet code root containing the
archived ``src/unet.py``, ``src/vit_seg_modeling.py`` and
``src/wavelet_ops.py`` implementations.  The side-effectful ``inference.py`` is
never imported.  TransUNet ImageNet weights are not loaded before the archived
checkpoint: an exact strict state-dict match requires the checkpoint to contain
every parameter and persistent buffer and then overwrites all of them.  Loading
the NPZ first would therefore change neither the final state nor the topology,
while adding a large unrelated I/O dependency.  The build report makes this
choice explicit.

The post-load state hash is independent of ``torch.save`` container metadata.
For every state-dict item in sorted key order it hashes length-prefixed key,
dtype and shape metadata followed by the contiguous CPU tensor bytes.  This
provides a complete, deterministic fingerprint of all parameters and persistent
buffers after strict loading.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, os.PathLike]

IMAGE_SIZE = 352
TRANSUNET_GRID = 22

# These identify the model-definition files independently audited on T9.  A
# caller may pass ``expected_source_sha256=None`` to record rather than enforce
# them only when a different source archive has itself been audited and locked.
AUDITED_MODEL_SOURCE_SHA256: Mapping[str, str] = {
    "unet.py": "9900fc7e50c29e88c6f55b293da3548096d50cc6c6020fe7f2ffa13cb5af3866",
    "vit_seg_modeling.py": "de252895b30d186da903535bf014f7bd744e33d5ad55c2bf5d862d662dea9620",
    "vit_seg_configs.py": "661639dec10545b2038d95a2414b76ea9552607613889cab2c25ef3ec1bc453c",
    "vit_seg_modeling_resnet_skip.py": "57baf976b3776c509dd2e0059bebc609723eebbb0bbce105838d95fda0adc90c",
    "wavelet_ops.py": "866f722ba767532f821cfa9c163e9c4d36e551f7ea0a1e7ebd4b0a4229b6fde2",
}

_ARCHITECTURE_ALIASES: Mapping[str, str] = {
    "unet": "UNet",
    "u-net": "UNet",
    "transunet": "TransUNet",
    "wavtransunet": "WavTransUNet",
    "wavrestransunet": "WavTransUNet",
    "wtransunet": "WavTransUNet",
    "w-transunet": "WavTransUNet",
}

_REQUIRED_SOURCES: Mapping[str, Tuple[str, ...]] = {
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

_ALLOWED_WRAPPER_PREFIXES: Tuple[str, ...] = (
    "model.",
    "net.",
    "unet.",
    "transunet.",
    "generator.",
    "reconstructor.",
)


class ModelFactoryError(RuntimeError):
    """Base class for model-factory failures."""


class SourceAuditError(ModelFactoryError):
    """Raised when the explicit model source root fails its audit."""


class CheckpointAssetError(ModelFactoryError):
    """Raised when a checkpoint does not match its frozen expected asset."""


class CheckpointTopologyError(ModelFactoryError):
    """Raised when no permitted mapping strictly matches the model topology."""


class RawInferenceError(ModelFactoryError):
    """Raised when raw batch inference violates its locked contract."""


@dataclass(frozen=True)
class ModelBuildReport:
    architecture: str
    code_root: str
    image_size: int
    source_sha256: Mapping[str, str]
    source_hashes_enforced: bool
    pretrained_npz_loaded: bool
    pretrained_skip_reason: Optional[str]
    unet_scales: Optional[int] = None
    unet_skip_channels: Optional[int] = None
    unet_use_sigmoid: Optional[bool] = None
    unet_use_norm: Optional[bool] = None
    transunet_variant: Optional[str] = None
    transunet_n_classes: Optional[int] = None
    transunet_n_skip: Optional[int] = None
    transunet_grid: Optional[Tuple[int, int]] = None
    wav_base_channels: Optional[int] = None
    wav_blocks: Optional[int] = None
    wav_norm: Optional[str] = None
    wav_upsample: Optional[str] = None
    residual_output: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: str
    checkpoint_asset_id: Optional[str]
    checkpoint_byte_size: int
    checkpoint_sha256: str
    mapping: str
    tensor_count: int
    post_load_state_sha256: str
    strict: bool
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawInferenceReport:
    cases: int
    batches: int
    requested_batch_size: int
    tail_batch_size: int
    input_dtype: str
    output_dtype: str
    normalized: bool
    clipped: bool
    case_order_preserved: bool
    exact_coverage: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise ModelFactoryError("PyTorch is required only for model/tensor operations") from exc
    return torch


def _sha256_file(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ModelFactoryError("symlink paths are forbidden for audited files: %s" % candidate)
    if not candidate.is_file():
        raise ModelFactoryError("not a regular file: %s" % candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_architecture(name: str) -> str:
    key = str(name).strip().lower().replace("_", "").replace(" ", "")
    try:
        return _ARCHITECTURE_ALIASES[key]
    except KeyError as exc:
        raise ModelFactoryError("unsupported architecture: %r" % name) from exc


def _source_dir(code_root: PathLike) -> Tuple[Path, Path]:
    root_input = Path(code_root)
    if not root_input.exists():
        raise SourceAuditError("explicit WTU code root does not exist: %s" % root_input)
    root = root_input.resolve()
    src = root / "src"
    if not src.is_dir():
        raise SourceAuditError("WTU code root must contain a src directory: %s" % root)
    return root, src


def verify_model_sources(
    code_root: PathLike,
    architecture: str,
    expected_source_sha256: Optional[Mapping[str, str]] = AUDITED_MODEL_SOURCE_SHA256,
) -> Tuple[Path, Dict[str, str]]:
    """Verify and fingerprint the exact source files needed by one model."""

    canonical = canonical_architecture(architecture)
    root, src = _source_dir(code_root)
    observed: Dict[str, str] = {}
    for filename in _REQUIRED_SOURCES[canonical]:
        path = src / filename
        observed_hash = _sha256_file(path)
        observed[filename] = observed_hash
        if expected_source_sha256 is not None:
            expected = expected_source_sha256.get(filename)
            if expected is None:
                raise SourceAuditError("no expected source hash for %s" % filename)
            if observed_hash != expected:
                raise SourceAuditError(
                    "source SHA-256 mismatch for %s: expected %s, observed %s"
                    % (path, expected, observed_hash)
                )
    return root, observed


def _audited_package(src: Path) -> str:
    """Register an isolated synthetic package rooted at the explicit src path."""

    package_name = "_dc_wtu_%s" % hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__file__ = str(src / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(src)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return package_name


def _load_source_module(src: Path, leaf_name: str) -> Any:
    if leaf_name == "inference":
        raise SourceAuditError("side-effectful inference.py is forbidden in the model factory")
    package_name = _audited_package(src)
    full_name = "%s.%s" % (package_name, leaf_name)
    if full_name in sys.modules:
        return sys.modules[full_name]
    path = src / (leaf_name + ".py")
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise SourceAuditError("could not create import specification for %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    return module


def _build_transunet(vit_module: Any, image_size: int) -> Any:
    if image_size != IMAGE_SIZE:
        raise ModelFactoryError("archived TransUNet topology requires image_size=352")
    config = copy.deepcopy(vit_module.CONFIGS["R50-ViT-B_16"])
    config.n_classes = 1
    config.n_skip = 3
    grid = image_size // int(config.patch_size)
    if grid != TRANSUNET_GRID:
        raise ModelFactoryError("expected R50-ViT-B/16 grid 22, observed %d" % grid)
    config.patches.grid = (grid, grid)
    return vit_module.VisionTransformer(config, img_size=image_size, num_classes=1)


def _build_wavtransunet(torch: Any, vit_module: Any, wavelet_module: Any, image_size: int) -> Any:
    transunet = _build_transunet(vit_module, image_size)
    mix = wavelet_module.WavMixResNet(
        in_ch=4,
        out_ch=1,
        base_ch=64,
        num_blocks=8,
        norm="gn",
    )
    haar_dwt_hvd = wavelet_module.haar_dwt_hvd
    upsample_like = wavelet_module._upsample_like

    class WavResTransUNet(torch.nn.Module):
        """Thin topology-identical wrapper around audited source components."""

        def __init__(self) -> None:
            super().__init__()
            self.wav_upsample = "bilinear"
            self.mix = mix
            self.transunet = transunet
            self.residual_out = True

        def forward(self, x: Any) -> Any:
            _, h, v, d = haar_dwt_hvd(x)
            h = upsample_like(h, x, mode=self.wav_upsample)
            v = upsample_like(v, x, mode=self.wav_upsample)
            d = upsample_like(d, x, mode=self.wav_upsample)
            x4 = torch.cat([x, h, v, d], dim=1)
            x_mix = self.mix(x4)
            out = self.transunet(x_mix)
            if self.residual_out:
                out = x_mix + out
            return out

    WavResTransUNet.__name__ = "WavResTransUNet"
    return WavResTransUNet()


def build_archived_model(
    architecture: str,
    code_root: PathLike,
    *,
    image_size: int = IMAGE_SIZE,
    expected_source_sha256: Optional[Mapping[str, str]] = AUDITED_MODEL_SOURCE_SHA256,
) -> Tuple[Any, ModelBuildReport]:
    """Construct an exact archived topology without loading ``inference.py``.

    Construction occurs on CPU.  The complete archived checkpoint should be
    strictly loaded and hashed before moving the model to an accelerator.
    """

    canonical = canonical_architecture(architecture)
    if image_size != IMAGE_SIZE:
        raise ModelFactoryError("archived fixed-checkpoint topology requires image_size=352")
    root, source_hashes = verify_model_sources(
        code_root, canonical, expected_source_sha256=expected_source_sha256
    )
    src = root / "src"
    torch = _torch()

    if canonical == "UNet":
        unet_module = _load_source_module(src, "unet")
        model = unet_module.get_unet_model(
            in_ch=1,
            out_ch=1,
            scales=5,
            skip=4,
            use_sigmoid=False,
            use_norm=True,
        )
        report = ModelBuildReport(
            architecture=canonical,
            code_root=str(root),
            image_size=image_size,
            source_sha256=source_hashes,
            source_hashes_enforced=expected_source_sha256 is not None,
            pretrained_npz_loaded=False,
            pretrained_skip_reason=None,
            unet_scales=5,
            unet_skip_channels=4,
            unet_use_sigmoid=False,
            unet_use_norm=True,
        )
        return model, report

    vit_module = _load_source_module(src, "vit_seg_modeling")
    pretrained_reason = (
        "Skipped because the subsequent exact strict checkpoint load requires and "
        "overwrites every parameter and persistent buffer; the NPZ cannot affect "
        "the final post-load state."
    )
    if canonical == "TransUNet":
        model = _build_transunet(vit_module, image_size)
        report = ModelBuildReport(
            architecture=canonical,
            code_root=str(root),
            image_size=image_size,
            source_sha256=source_hashes,
            source_hashes_enforced=expected_source_sha256 is not None,
            pretrained_npz_loaded=False,
            pretrained_skip_reason=pretrained_reason,
            transunet_variant="R50-ViT-B_16",
            transunet_n_classes=1,
            transunet_n_skip=3,
            transunet_grid=(22, 22),
        )
        return model, report

    wavelet_module = _load_source_module(src, "wavelet_ops")
    model = _build_wavtransunet(torch, vit_module, wavelet_module, image_size)
    report = ModelBuildReport(
        architecture=canonical,
        code_root=str(root),
        image_size=image_size,
        source_sha256=source_hashes,
        source_hashes_enforced=expected_source_sha256 is not None,
        pretrained_npz_loaded=False,
        pretrained_skip_reason=pretrained_reason,
        transunet_variant="R50-ViT-B_16",
        transunet_n_classes=1,
        transunet_n_skip=3,
        transunet_grid=(22, 22),
        wav_base_channels=64,
        wav_blocks=8,
        wav_norm="gn",
        wav_upsample="bilinear",
        residual_output=True,
    )
    return model, report


def read_expected_checkpoint_asset(
    expected_assets_path: PathLike,
    asset_id: str,
    *,
    architecture: Optional[str] = None,
    views: Optional[int] = None,
) -> Dict[str, Any]:
    """Read one resolved fixed-checkpoint identity from expected_assets.json."""

    path = Path(expected_assets_path)
    if path.is_symlink() or not path.is_file():
        raise CheckpointAssetError("expected-assets document is not a regular file: %s" % path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointAssetError("could not parse expected-assets document: %s" % path) from exc
    if document.get("hash_algorithm") != "sha256" or not isinstance(document.get("assets"), list):
        raise CheckpointAssetError("invalid expected-assets document")
    matches = [item for item in document["assets"] if item.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise CheckpointAssetError("expected exactly one asset %r, found %d" % (asset_id, len(matches)))
    asset = matches[0]
    if asset.get("role") != "fixed_checkpoint" or asset.get("resolved") is not True:
        raise CheckpointAssetError("asset %s is not a resolved fixed checkpoint" % asset_id)
    expected_hash = asset.get("sha256")
    expected_size = asset.get("byte_size")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise CheckpointAssetError("asset %s has no valid SHA-256" % asset_id)
    if type(expected_size) is not int or expected_size <= 0:
        raise CheckpointAssetError("asset %s has no valid byte size" % asset_id)
    metadata = asset.get("metadata", {})
    if architecture is not None:
        expected_architecture = canonical_architecture(architecture)
        recorded = canonical_architecture(metadata.get("architecture", ""))
        if recorded != expected_architecture:
            raise CheckpointAssetError(
                "asset architecture mismatch: expected %s, recorded %s"
                % (expected_architecture, recorded)
            )
    if views is not None and metadata.get("views") != int(views):
        raise CheckpointAssetError(
            "asset view mismatch: expected %d, recorded %r" % (int(views), metadata.get("views"))
        )
    return dict(asset)


def verify_checkpoint_file(
    checkpoint_path: PathLike,
    *,
    expected_sha256: str,
    expected_byte_size: int,
) -> Tuple[int, str]:
    """Verify checkpoint bytes before any deserialization."""

    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise CheckpointAssetError("checkpoint is not a regular non-symlink file: %s" % path)
    observed_size = path.stat().st_size
    if observed_size != expected_byte_size:
        raise CheckpointAssetError(
            "checkpoint size mismatch: expected %d, observed %d"
            % (expected_byte_size, observed_size)
        )
    observed_hash = _sha256_file(path)
    if observed_hash != expected_sha256:
        raise CheckpointAssetError(
            "checkpoint SHA-256 mismatch: expected %s, observed %s"
            % (expected_sha256, observed_hash)
        )
    return observed_size, observed_hash


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise CheckpointTopologyError("checkpoint does not contain a recognizable state dictionary")


def _strip_prefix(state: Mapping[str, Any], prefix: str) -> Optional[Dict[str, Any]]:
    if not state or not all(key.startswith(prefix) for key in state):
        return None
    return {key[len(prefix):]: value for key, value in state.items()}


def _state_candidates(state: Mapping[str, Any]) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """Yield only the fixed, audited prefix mappings; never guess a wrapper."""

    seen: set[Tuple[str, ...]] = set()

    def emit(label: str, candidate: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        if candidate is None:
            return
        signature = tuple(candidate.keys())
        if signature in seen:
            return
        seen.add(signature)
        yield label, candidate

    yield from emit("raw", state)
    module_stripped = _strip_prefix(state, "module.")
    yield from emit("strip_module.", module_stripped)
    for prefix in _ALLOWED_WRAPPER_PREFIXES:
        yield from emit("strip_%s" % prefix, _strip_prefix(state, prefix))
        if module_stripped is not None:
            yield from emit(
                "strip_module.+%s" % prefix,
                _strip_prefix(module_stripped, prefix),
            )


def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _topology_difference(
    model_state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    model_keys = set(model_state)
    candidate_keys = set(candidate)
    missing = tuple(sorted(model_keys - candidate_keys))
    unexpected = tuple(sorted(candidate_keys - model_keys))
    shape_mismatch = tuple(
        sorted(
            key
            for key in model_keys & candidate_keys
            if _shape_tuple(model_state[key]) != _shape_tuple(candidate[key])
        )
    )
    return missing, unexpected, shape_mismatch


def canonical_state_sha256(state_or_model: Any) -> str:
    """Hash every parameter and persistent buffer in a deterministic format."""

    torch = _torch()
    state = state_or_model.state_dict() if hasattr(state_or_model, "state_dict") else state_or_model
    if not isinstance(state, Mapping) or not state:
        raise CheckpointTopologyError("cannot hash an empty or non-mapping state")
    digest = hashlib.sha256()

    def update_field(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little", signed=False))
        digest.update(payload)

    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise CheckpointTopologyError("state entry %s is not a tensor" % key)
        cpu = tensor.detach().cpu().contiguous()
        update_field(str(key).encode("utf-8"))
        update_field(str(cpu.dtype).encode("ascii"))
        update_field(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
        if cpu.dtype == torch.bfloat16:
            payload = cpu.view(torch.uint8).numpy().tobytes(order="C")
        else:
            payload = cpu.numpy().tobytes(order="C")
        update_field(payload)
    return digest.hexdigest()


def _torch_load_verified(path: Path, map_location: str = "cpu") -> Any:
    torch = _torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        return torch.load(path, map_location=map_location)


def load_strict_checkpoint(
    model: Any,
    checkpoint_path: PathLike,
    *,
    expected_sha256: str,
    expected_byte_size: int,
    checkpoint_asset_id: Optional[str] = None,
) -> CheckpointLoadReport:
    """Hash-verify, topology-check, strictly load, then hash the full state."""

    path = Path(checkpoint_path)
    observed_size, observed_hash = verify_checkpoint_file(
        path,
        expected_sha256=expected_sha256,
        expected_byte_size=expected_byte_size,
    )
    checkpoint = _torch_load_verified(path, map_location="cpu")
    raw_state = _extract_state_dict(checkpoint)
    if not raw_state or not all(isinstance(key, str) for key in raw_state):
        raise CheckpointTopologyError("state dictionary keys must be non-empty strings")
    model_state = model.state_dict()
    failures = []
    for mapping_name, candidate in _state_candidates(raw_state):
        missing, unexpected, shape_mismatch = _topology_difference(model_state, candidate)
        if missing or unexpected or shape_mismatch:
            failures.append(
                "%s(missing=%d, unexpected=%d, shape=%d)"
                % (mapping_name, len(missing), len(unexpected), len(shape_mismatch))
            )
            continue
        incompatible = model.load_state_dict(candidate, strict=True)
        missing_after = tuple(incompatible.missing_keys)
        unexpected_after = tuple(incompatible.unexpected_keys)
        if missing_after or unexpected_after:  # defensive; strict=True should raise
            raise CheckpointTopologyError(
                "strict load returned incompatible keys for %s" % mapping_name
            )
        post_hash = canonical_state_sha256(model)
        return CheckpointLoadReport(
            checkpoint_path=str(path.resolve()),
            checkpoint_asset_id=checkpoint_asset_id,
            checkpoint_byte_size=observed_size,
            checkpoint_sha256=observed_hash,
            mapping=mapping_name,
            tensor_count=len(model_state),
            post_load_state_sha256=post_hash,
            strict=True,
            missing_keys=(),
            unexpected_keys=(),
        )
    raise CheckpointTopologyError(
        "no permitted prefix mapping exactly matches model topology: %s"
        % ("; ".join(failures) if failures else "no candidates")
    )


def build_and_load_archived_model(
    architecture: str,
    code_root: PathLike,
    checkpoint_path: PathLike,
    *,
    expected_assets_path: PathLike,
    checkpoint_asset_id: str,
    views: int,
    device: Union[str, Any] = "cpu",
    expected_source_sha256: Optional[Mapping[str, str]] = AUDITED_MODEL_SOURCE_SHA256,
) -> Tuple[Any, ModelBuildReport, CheckpointLoadReport]:
    """Build the exact topology and load only its hash-verified expected asset."""

    canonical = canonical_architecture(architecture)
    asset = read_expected_checkpoint_asset(
        expected_assets_path,
        checkpoint_asset_id,
        architecture=canonical,
        views=views,
    )
    model, build_report = build_archived_model(
        canonical,
        code_root,
        image_size=IMAGE_SIZE,
        expected_source_sha256=expected_source_sha256,
    )
    load_report = load_strict_checkpoint(
        model,
        checkpoint_path,
        expected_sha256=asset["sha256"],
        expected_byte_size=asset["byte_size"],
        checkpoint_asset_id=checkpoint_asset_id,
    )
    model.to(device)
    model.eval()
    return model, build_report, load_report


def infer_raw_float32(
    model: Any,
    inputs: Any,
    *,
    batch_size: int,
    device: Union[str, Any] = "cpu",
    output: Optional[Any] = None,
    expected_spatial: Optional[Tuple[int, int]] = (IMAGE_SIZE, IMAGE_SIZE),
) -> Tuple[Any, RawInferenceReport]:
    """Run raw float32 inference with exact, ordered tail-batch coverage.

    ``inputs`` must be ``(N,1,H,W)``.  Values are converted to float32 but are
    never normalized, clipped, masked, or passed through autocast.  If ``output``
    is supplied it may be an ndarray/memmap with the exact same shape and
    float32 dtype; otherwise a float32 NumPy array is allocated.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise RawInferenceError("batch_size must be a positive integer")
    torch = _torch()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - scientific runtime dependency
        raise RawInferenceError("NumPy is required for raw batch inference") from exc

    shape = tuple(int(item) for item in getattr(inputs, "shape", ()))
    if len(shape) != 4 or shape[1] != 1:
        raise RawInferenceError("inputs must have shape (N,1,H,W), observed %r" % (shape,))
    if shape[0] <= 0:
        raise RawInferenceError("inputs must contain at least one case")
    if expected_spatial is not None and shape[2:] != tuple(expected_spatial):
        raise RawInferenceError(
            "input spatial shape mismatch: expected %r, observed %r"
            % (tuple(expected_spatial), shape[2:])
        )
    n_cases = shape[0]
    expected_output_shape = shape
    if output is None:
        destination = np.empty(expected_output_shape, dtype=np.float32)
    else:
        destination = output
        if tuple(getattr(destination, "shape", ())) != expected_output_shape:
            raise RawInferenceError("output shape must equal input shape")
        if getattr(destination, "dtype", None) != np.dtype(np.float32):
            raise RawInferenceError("output dtype must be float32")

    covered = np.zeros(n_cases, dtype=np.uint8)
    batches = 0
    tail_size = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n_cases, batch_size):
            stop = min(start + batch_size, n_cases)
            source_batch = inputs[start:stop]
            if torch.is_tensor(source_batch):
                batch = source_batch.detach().to(device=device, dtype=torch.float32)
            else:
                batch_np = np.asarray(source_batch, dtype=np.float32)
                batch = torch.from_numpy(np.ascontiguousarray(batch_np)).to(device)
            prediction = model(batch)
            if not torch.is_tensor(prediction):
                raise RawInferenceError("model output must be one tensor")
            if tuple(int(item) for item in prediction.shape) != (stop - start,) + shape[1:]:
                raise RawInferenceError(
                    "model output shape mismatch for cases [%d:%d]: observed %r"
                    % (start, stop, tuple(prediction.shape))
                )
            raw = prediction.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            destination[start:stop] = raw
            covered[start:stop] += 1
            batches += 1
            tail_size = stop - start

    exact_coverage = bool(np.all(covered == 1))
    if not exact_coverage:
        raise RawInferenceError("case coverage was not exactly once per input")
    if getattr(destination, "flush", None) is not None:
        destination.flush()
    report = RawInferenceReport(
        cases=n_cases,
        batches=batches,
        requested_batch_size=batch_size,
        tail_batch_size=tail_size,
        input_dtype=str(getattr(inputs, "dtype", "unknown")),
        output_dtype=str(destination.dtype),
        normalized=False,
        clipped=False,
        case_order_preserved=True,
        exact_coverage=True,
    )
    return destination, report


__all__ = [
    "AUDITED_MODEL_SOURCE_SHA256",
    "CheckpointAssetError",
    "CheckpointLoadReport",
    "CheckpointTopologyError",
    "IMAGE_SIZE",
    "ModelBuildReport",
    "ModelFactoryError",
    "RawInferenceError",
    "RawInferenceReport",
    "SourceAuditError",
    "build_and_load_archived_model",
    "build_archived_model",
    "canonical_architecture",
    "canonical_state_sha256",
    "infer_raw_float32",
    "load_strict_checkpoint",
    "read_expected_checkpoint_asset",
    "verify_checkpoint_file",
    "verify_model_sources",
]
