#!/usr/bin/env python3
"""Architecture-neutral finite data-consistency numerical core.

This module contains no network logic and never selects hyperparameters.  A
single immutable network reconstruction ``x0`` is refined on the LoDoPaB 362
pixel operator domain while its outer five-pixel frame is reset to the matched
FBP after every update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


FULL_SIZE = 362
CROP_SIZE = 352
BORDER = 5


class ArrayOperator(Protocol):
    """Optional light-weight interface used by unit tests."""

    def forward_np(self, image: np.ndarray) -> np.ndarray: ...
    def adjoint_np(self, measurement: np.ndarray) -> np.ndarray: ...


def embed_crop(crop352: np.ndarray, fbp362: np.ndarray) -> np.ndarray:
    crop = np.asarray(crop352, dtype=np.float32)
    frame = np.asarray(fbp362, dtype=np.float32)
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"expected 352x352 network output, got {crop.shape}")
    if frame.shape != (FULL_SIZE, FULL_SIZE):
        raise ValueError(f"expected 362x362 FBP, got {frame.shape}")
    if not np.all(np.isfinite(crop)) or not np.all(np.isfinite(frame)):
        raise ValueError("nonfinite reconstruction or FBP")
    canvas = frame.copy()
    canvas[BORDER : BORDER + CROP_SIZE, BORDER : BORDER + CROP_SIZE] = crop
    return canvas


def crop_canvas(canvas362: np.ndarray) -> np.ndarray:
    canvas = np.asarray(canvas362)
    if canvas.shape != (FULL_SIZE, FULL_SIZE):
        raise ValueError(f"expected 362x362 canvas, got {canvas.shape}")
    return np.ascontiguousarray(
        canvas[BORDER : BORDER + CROP_SIZE, BORDER : BORDER + CROP_SIZE], dtype=np.float32
    )


def restore_fbp_frame(image: np.ndarray, fbp362: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32).copy()
    fbp = np.asarray(fbp362, dtype=np.float32)
    if value.shape != (FULL_SIZE, FULL_SIZE) or fbp.shape != value.shape:
        raise ValueError("frame restoration requires two 362x362 arrays")
    value[:BORDER, :] = fbp[:BORDER, :]
    value[-BORDER:, :] = fbp[-BORDER:, :]
    value[:, :BORDER] = fbp[:, :BORDER]
    value[:, -BORDER:] = fbp[:, -BORDER:]
    return value


def _forward(operator: object, image: np.ndarray) -> np.ndarray:
    if hasattr(operator, "forward_np"):
        result = operator.forward_np(np.asarray(image, dtype=np.float32))
    else:
        element = operator.domain.element(np.asarray(image, dtype=np.float32))
        result = operator(element)
    return np.asarray(result, dtype=np.float32)


def _adjoint(operator: object, measurement: np.ndarray) -> np.ndarray:
    if hasattr(operator, "adjoint_np"):
        result = operator.adjoint_np(np.asarray(measurement, dtype=np.float32))
    else:
        element = operator.range.element(np.asarray(measurement, dtype=np.float32))
        result = operator.adjoint(element)
    return np.asarray(result, dtype=np.float32)


def _odl_inner_product_spaces(operator: object) -> tuple[object, object] | None:
    """Return ODL-like domain/range spaces when both expose exact inner products.

    ``odl.tomo.RayTransform.adjoint`` is the adjoint with respect to the
    discretized-space inner products, which include quadrature weights.  Raw
    ``np.vdot`` therefore does not test its adjoint identity.  Lightweight toy
    operators used by unit tests deliberately omit these spaces and retain the
    ordinary Euclidean fallback.
    """

    domain = getattr(operator, "domain", None)
    range_space = getattr(operator, "range", None)
    for space in (domain, range_space):
        if space is None or not callable(getattr(space, "element", None)) or not callable(
            getattr(space, "inner", None)
        ):
            return None
    return domain, range_space


def _space_inner(space: object, left: np.ndarray, right: np.ndarray) -> float:
    """Evaluate one real discretized-space inner product as a finite scalar."""

    left_element = space.element(np.asarray(left, dtype=np.float32))
    right_element = space.element(np.asarray(right, dtype=np.float32))
    value = float(np.real(space.inner(left_element, right_element)))
    if not np.isfinite(value):
        raise FloatingPointError("space inner product is not finite")
    return value


def projection_residual(operator: object, image: np.ndarray, measurement: np.ndarray) -> float:
    y = np.asarray(measurement, dtype=np.float32)
    norm = float(np.linalg.norm(y.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("measurement norm must be positive and finite")
    difference = _forward(operator, image).astype(np.float64) - y.astype(np.float64)
    value = float(np.linalg.norm(difference) / norm)
    if not np.isfinite(value):
        raise FloatingPointError("projection residual is not finite")
    return value


def gradient_scale_term(operator: object, image: np.ndarray, measurement: np.ndarray) -> float:
    x = np.asarray(image, dtype=np.float32)
    norm = float(np.linalg.norm(x.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("network reconstruction norm must be positive and finite")
    gradient = _adjoint(operator, _forward(operator, x) - np.asarray(measurement, dtype=np.float32))
    value = float(np.linalg.norm(gradient.astype(np.float64)) / norm)
    if not np.isfinite(value):
        raise FloatingPointError("gradient scale is not finite")
    return value


def quadratic_objective(
    operator: object,
    image: np.ndarray,
    measurement: np.ndarray,
    anchor: np.ndarray,
    regularization: float,
) -> float:
    residual = _forward(operator, image).astype(np.float64) - np.asarray(measurement, dtype=np.float64)
    displacement = np.asarray(image, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    spaces = _odl_inner_product_spaces(operator)
    if spaces is None:
        residual_squared = float(np.sum(residual * residual))
        displacement_squared = float(np.sum(displacement * displacement))
    else:
        domain, range_space = spaces
        residual_squared = _space_inner(range_space, residual, residual)
        displacement_squared = _space_inner(domain, displacement, displacement)
    return 0.5 * residual_squared + 0.5 * regularization * displacement_squared


@dataclass(frozen=True)
class DCTrajectory:
    snapshots: dict[int, np.ndarray]
    residuals: dict[int, float]
    objectives: dict[int, float]
    diverged: bool
    divergence_reason: str | None


def finite_dc_trajectory(
    operator: object,
    network_canvas: np.ndarray,
    fbp362: np.ndarray,
    measurement: np.ndarray,
    *,
    regularization: float,
    step_size: float,
    iterations: Iterable[int],
    divergence_ratio: float = 2.0,
) -> DCTrajectory:
    """Run one finite projected-gradient trajectory and retain locked snapshots.

    No adaptive step-size reduction, clipping, positivity projection, or
    outcome-dependent stopping is performed.  The input anchor is immutable.
    """

    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be positive and finite")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step size must be positive and finite")
    k_grid = sorted({int(value) for value in iterations})
    if not k_grid or k_grid[0] != 0 or any(value < 0 for value in k_grid):
        raise ValueError("iteration grid must be nonnegative and include zero")
    anchor = np.asarray(network_canvas, dtype=np.float32).copy()
    fbp = np.asarray(fbp362, dtype=np.float32)
    y = np.asarray(measurement, dtype=np.float32)
    if anchor.shape != (FULL_SIZE, FULL_SIZE) or fbp.shape != anchor.shape:
        raise ValueError("DC inputs must use the 362x362 operator domain")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(y)):
        raise ValueError("DC inputs contain NaN or infinity")

    baseline_residual = projection_residual(operator, anchor, y)
    snapshots = {0: anchor.copy()}
    residuals = {0: baseline_residual}
    objectives = {0: quadratic_objective(operator, anchor, y, anchor, regularization)}
    x = anchor.copy()
    diverged = False
    reason: str | None = None

    for iteration in range(1, k_grid[-1] + 1):
        data_gradient = _adjoint(operator, _forward(operator, x) - y)
        gradient = data_gradient + np.float32(regularization) * (x - anchor)
        x = x - np.float32(step_size) * gradient
        x = restore_fbp_frame(x, fbp)
        if not np.all(np.isfinite(x)):
            diverged = True
            reason = f"nonfinite reconstruction at k={iteration}"
            break
        if iteration in k_grid:
            residual = projection_residual(operator, x, y)
            if residual > divergence_ratio * baseline_residual:
                diverged = True
                reason = (
                    f"residual {residual:.9g} exceeded {divergence_ratio:g} times "
                    f"baseline {baseline_residual:.9g} at k={iteration}"
                )
                break
            snapshots[iteration] = x.copy()
            residuals[iteration] = residual
            objectives[iteration] = quadratic_objective(
                operator, x, y, anchor, regularization
            )

    return DCTrajectory(snapshots, residuals, objectives, diverged, reason)


def power_iteration_ata(
    operator: object,
    shape: tuple[int, int] = (FULL_SIZE, FULL_SIZE),
    *,
    seed: int = 0,
    iterations: int = 20,
) -> tuple[float, list[dict[str, float]]]:
    if iterations <= 0:
        raise ValueError("power iterations must be positive")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(shape).astype(np.float32)
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm <= 0.0:
        raise FloatingPointError("zero initial power vector")
    vector /= np.float32(norm)
    trace: list[dict[str, float]] = []
    estimate = 0.0
    for index in range(iterations):
        transformed = _adjoint(operator, _forward(operator, vector))
        transformed_norm = float(np.linalg.norm(transformed.astype(np.float64)))
        vector_norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(transformed_norm) or transformed_norm <= 0.0:
            raise FloatingPointError("invalid power-iteration norm")
        estimate = transformed_norm / vector_norm
        rayleigh = float(
            np.vdot(vector.astype(np.float64), transformed.astype(np.float64)).real
            / np.vdot(vector.astype(np.float64), vector.astype(np.float64)).real
        )
        trace.append(
            {
                "iteration": float(index + 1),
                "norm_ratio": estimate,
                "rayleigh": rayleigh,
            }
        )
        vector = transformed / np.float32(transformed_norm)
    return float(estimate), trace


def adjoint_relative_error(
    operator: object,
    image: np.ndarray,
    measurement: np.ndarray,
) -> float:
    x = np.asarray(image, dtype=np.float32)
    z = np.asarray(measurement, dtype=np.float32)
    forward = _forward(operator, x)
    adjoint = _adjoint(operator, z)
    spaces = _odl_inner_product_spaces(operator)
    if spaces is None:
        left = float(np.vdot(forward.astype(np.float64), z.astype(np.float64)).real)
        right = float(np.vdot(x.astype(np.float64), adjoint.astype(np.float64)).real)
    else:
        domain, range_space = spaces
        left = _space_inner(range_space, forward, z)
        right = _space_inner(domain, x, adjoint)
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def dc_definition() -> dict[str, object]:
    return {
        "objective": "0.5*||A x-y||_Y^2 + 0.5*lambda*||x-x0||_X^2",
        "update": "x <- x-eta*(A_star(Ax-y)+lambda*(x-x0))",
        "inner_products": (
            "X and Y are the ODL reconstruction/projection discretized-space inner products; "
            "A_star is their adjoint. Operators without such spaces use Euclidean inner products."
        ),
        "domain_shape": [FULL_SIZE, FULL_SIZE],
        "network_crop": [CROP_SIZE, CROP_SIZE],
        "crop_offset": [BORDER, BORDER],
        "frame_rule": "restore condition-matched 362x362 FBP outer 5-pixel frame after every step",
        "projection_residual": "||A x-y||_2/||y||_2 over every acquired sinogram bin",
        "clipping": False,
        "positivity": False,
        "anatomical_mask": False,
        "adaptive_step_size": False,
    }
