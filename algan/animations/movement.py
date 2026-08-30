"""High-value geometric animations that are not simple context composition.

Algan's
:class:`~algan.animation_timeline.animation_contexts.AnimationContext` system
already replaces most of Manim's small Animation subclasses: fades, rotations,
scales, method calls, and grouped/succession animations are ordinary Mob
operations nested in ``Sync``, ``Seq``, and ``Lag``.  This module is reserved
for animations which need reusable geometric algorithms rather than merely a
convenience class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Sync, animation_manager_for
from algan.constants import easings
from algan.constants.spatial import ORIGIN
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.surfaces.surface import Surface
from algan.utils.tensor_utils import cast_to_tensor, unsquish


def _resolve_mobject_and_callable(first, second, *, function_name: str):
    """Accept both Algan's ``(mobject, function)`` and Manim's reverse order."""
    if isinstance(first, Mob) and callable(second):
        return first, second
    if callable(first) and isinstance(second, Mob):
        return second, first
    raise TypeError(
        f"{function_name} expects a Mob and a callable; received "
        f"{type(first).__name__} and {type(second).__name__}."
    )


def _geometry_point_owners(mobject: Mob) -> list[Mob]:
    """Return fixed-size point buffers which define ``mobject``'s geometry.

    Bezier circuits render from their control-point component; surfaces render
    from their sampled grid.  Modifying those child Mobs gives pointwise
    animations a common implementation for 2-D paths, text, meshes, imported
    models, and the sphere batches used by point clouds.
    """
    owners: list[Mob] = []
    seen: set[int] = set()
    for descendant in mobject.get_descendants():
        owner = None
        if isinstance(descendant, BezierCircuitCubic):
            owner = descendant.control_points
        elif isinstance(descendant, Surface):
            owner = descendant.grid
        if owner is not None and id(owner) not in seen:
            seen.add(id(owner))
            owners.append(owner)
    if not owners:
        raise TypeError(
            f"{type(mobject).__name__} has no deformable Bezier or surface geometry."
        )
    return owners


def _as_tensor_like(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Convert ``value`` without imposing Algan's animatable row dimensions."""
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _coerce_mapping_result(result: Any, reference: torch.Tensor) -> torch.Tensor:
    result = _as_tensor_like(result, reference)
    if result.shape != reference.shape:
        try:
            result = result.expand_as(reference)
        except RuntimeError as exc:
            raise ValueError(
                "Point mapping functions must return the same shape as their input; "
                f"got {tuple(result.shape)} for {tuple(reference.shape)}."
            ) from exc
    return result


def _call_point_function(function: Callable, points: torch.Tensor) -> torch.Tensor:
    """Call a vectorized point function, with a scalar NumPy fallback.

    The fast path is intentionally PyTorch-native because it is also used while
    materializing ``PhaseFlow``.  The fallback keeps ordinary Manim-style
    ``lambda p: ...`` callables convenient for one-off pointwise transforms.
    """
    try:
        return _coerce_mapping_result(function(points), points)
    except Exception:  # user callbacks can fail in many legitimate ways
        pass

    flat = points.detach().cpu().reshape(-1, 3).numpy()
    try:
        mapped = np.asarray([function(point) for point in flat], dtype=np.float64)
        return torch.as_tensor(
            mapped,
            device=points.device,
            dtype=points.dtype,
        ).reshape_as(points)
    except Exception as scalar_error:
        raise TypeError(
            "The point function failed for both a vectorized torch.Tensor and "
            "individual NumPy points."
        ) from scalar_error


def _call_homotopy_function(
    function: Callable,
    points: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Call either ``f(points, t)`` or Manim's ``f(x, y, z, t)`` API."""
    try:
        return _coerce_mapping_result(function(points, t), points)
    except Exception:
        pass

    t_values = t.reshape(points.shape[0], -1)[:, :1].expand(-1, points.shape[1])
    x, y, z = points.unbind(-1)
    try:
        result = function(x, y, z, t_values)
        if isinstance(result, (tuple, list)) and len(result) == 3:
            result = torch.stack(
                [_as_tensor_like(value, points) for value in result],
                dim=-1,
            )
        return _coerce_mapping_result(result, points)
    except Exception:
        pass

    points_np = points.detach().cpu().numpy()
    times_np = t_values.detach().cpu().numpy()
    try:
        result = function(
            points_np[..., 0],
            points_np[..., 1],
            points_np[..., 2],
            times_np,
        )
        if isinstance(result, (tuple, list)) and len(result) == 3:
            result = np.stack(result, axis=-1)
        return torch.as_tensor(
            result, device=points.device, dtype=points.dtype
        ).reshape_as(points)
    except Exception:
        pass

    try:
        mapped = [
            function(float(x), float(y), float(z), float(time))
            for frame_points, frame_times in zip(points_np, times_np)
            for (x, y, z), time in zip(frame_points, frame_times)
        ]
        return torch.as_tensor(
            np.asarray(mapped), device=points.device, dtype=points.dtype
        ).reshape_as(points)
    except Exception as exc:
        raise TypeError(
            "The homotopy failed for the vectorized f(points, t) API and the "
            "Manim-compatible f(x, y, z, t) API."
        ) from exc


@animated_function(
    animated_args={"t": 0.0},
    unique_args=["initial_points", "target_points"],
)
def _pointwise_interpolation_step(
    point_owner: Mob,
    t,
    initial_points,
    target_points,
):
    t = cast_to_tensor(t).to(initial_points)
    frame_count = t.shape[0]
    start = initial_points.expand(frame_count, -1, -1)
    target = target_points.expand(frame_count, -1, -1)
    point_owner.location = torch.lerp(start, target, t.reshape(frame_count, 1, 1))
    return point_owner


def ApplyPointwiseFunction(
    mobject,
    function=None,
    *,
    about_point=ORIGIN,
    duration: float = 1.0,
    easing=None,
):
    """Animate an arbitrary point mapping over all renderable geometry.

    Both ``ApplyPointwiseFunction(mob, function)`` and Manim's argument order
    ``ApplyPointwiseFunction(function, mob)`` are accepted.  The function may
    operate on a whole ``[..., 3]`` torch tensor or on one NumPy point at a time.
    """
    mobject, function = _resolve_mobject_and_callable(
        mobject, function, function_name="ApplyPointwiseFunction"
    )
    about_point = cast_to_tensor(about_point)
    with Sync(
        duration=duration,
        easing=easing,
        animation_manager=animation_manager_for(mobject),
    ):
        for owner in _geometry_point_owners(mobject):
            initial = owner.location.clone()
            centered = initial - about_point.to(initial)
            target = _call_point_function(function, centered) + about_point.to(initial)
            owner.animate_function(
                _pointwise_interpolation_step,
                initial_points=initial,
                target_points=target,
            )
    return mobject


def ApplyMatrix(
    mobject,
    matrix=None,
    *,
    about_point=ORIGIN,
    duration: float = 1.0,
    easing=None,
):
    """Animate a 2×2 or 3×3 linear transform about ``about_point``.

    Accepts both ``ApplyMatrix(mob, matrix)`` and Manim's
    ``ApplyMatrix(matrix, mob)`` argument order.
    """
    if not isinstance(mobject, Mob) and isinstance(matrix, Mob):
        mobject, matrix = matrix, mobject
    if not isinstance(mobject, Mob):
        raise TypeError("ApplyMatrix expects a Mob and a 2x2 or 3x3 matrix.")
    matrix = torch.as_tensor(matrix, dtype=torch.get_default_dtype())
    if matrix.shape == (2, 2):
        full_matrix = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        full_matrix[:2, :2] = matrix
        matrix = full_matrix
    elif matrix.shape != (3, 3):
        raise ValueError("matrix must have shape (2, 2) or (3, 3)")

    return ApplyPointwiseFunction(
        mobject,
        lambda points: points @ matrix.to(points).transpose(-1, -2),
        about_point=about_point,
        duration=duration,
        easing=easing,
    )


def _apply_complex_function(function: Callable, points: torch.Tensor) -> torch.Tensor:
    z = torch.complex(points[..., 0], points[..., 1])
    try:
        result = function(z)
        result = _as_tensor_like(result, z)
    except Exception:
        z_np = z.detach().cpu().numpy()
        try:
            result = function(z_np)
        except Exception:
            result = np.vectorize(function)(z_np)
        result = torch.as_tensor(result, device=z.device, dtype=z.dtype)
    if result.shape != z.shape:
        result = result.expand_as(z)
    return torch.stack((result.real, result.imag, points[..., 2]), dim=-1).to(points)


def ApplyComplexFunction(
    mobject,
    function=None,
    *,
    about_point=ORIGIN,
    duration: float = 1.0,
    easing=None,
):
    """Animate a complex map on the x-y plane while preserving z."""
    mobject, function = _resolve_mobject_and_callable(
        mobject, function, function_name="ApplyComplexFunction"
    )
    return ApplyPointwiseFunction(
        mobject,
        lambda points: _apply_complex_function(function, points),
        about_point=about_point,
        duration=duration,
        easing=easing,
    )


@animated_function(
    animated_args={"t": 0.0},
    unique_args=["homotopy_func", "initial_points"],
)
def _homotopy_step(point_owner, t, homotopy_func, initial_points):
    t = cast_to_tensor(t).to(initial_points)
    frame_count = t.shape[0]
    points = initial_points.expand(frame_count, -1, -1)
    point_owner.location = _call_homotopy_function(
        homotopy_func,
        points,
        t.reshape(frame_count, 1, 1),
    )
    return point_owner


def Homotopy(
    mobject,
    homotopy_func=None,
    *,
    duration: float = 2.0,
    easing=None,
):
    """Animate a continuous point deformation.

    The existing Algan vectorized API ``f(points, t)`` remains supported, and
    Manim's scalar ``f(x, y, z, t)`` callback form is accepted as well.  Unlike
    the former implementation, this works for surfaces and triangle meshes in
    addition to cubic Bezier paths.
    """
    mobject, homotopy_func = _resolve_mobject_and_callable(
        mobject, homotopy_func, function_name="Homotopy"
    )
    with Sync(
        duration=duration,
        easing=easing,
        animation_manager=animation_manager_for(mobject),
    ):
        for owner in _geometry_point_owners(mobject):
            owner.animate_function(
                _homotopy_step,
                homotopy_func=homotopy_func,
                initial_points=owner.location.clone(),
            )
    return mobject


def ComplexHomotopy(
    mobject,
    complex_homotopy=None,
    *,
    duration: float = 2.0,
    easing=None,
):
    """Animate ``f(z, t)`` on the x-y plane while preserving z."""
    mobject, complex_homotopy = _resolve_mobject_and_callable(
        mobject, complex_homotopy, function_name="ComplexHomotopy"
    )

    def homotopy(points, t):
        z = torch.complex(points[..., 0], points[..., 1])
        time = t.reshape(points.shape[0], -1)[:, :1].expand_as(z.real)
        try:
            result = complex_homotopy(z, time)
            result = _as_tensor_like(result, z)
        except Exception:
            z_np = z.detach().cpu().numpy()
            t_np = time.detach().cpu().numpy()
            try:
                result = complex_homotopy(z_np, t_np)
            except Exception:
                result = np.vectorize(complex_homotopy)(z_np, t_np)
            result = torch.as_tensor(result, device=z.device, dtype=z.dtype)
        return torch.stack((result.real, result.imag, points[..., 2]), dim=-1).to(
            points
        )

    return Homotopy(
        mobject,
        homotopy,
        duration=duration,
        easing=easing,
    )


@animated_function(
    animated_args={"t": 0.0},
    unique_args=["vector_field", "initial_points", "virtual_time", "integration_steps"],
)
def _phase_flow_step(
    point_owner,
    t,
    vector_field,
    initial_points,
    virtual_time,
    integration_steps,
):
    """Deterministic RK4 integration from the initial geometry to each frame."""
    t = cast_to_tensor(t).to(initial_points)
    frame_count = t.shape[0]
    points = initial_points.expand(frame_count, -1, -1).clone()
    dt = t.reshape(frame_count, 1, 1) * float(virtual_time) / int(integration_steps)
    for _ in range(int(integration_steps)):
        k1 = _call_point_function(vector_field, points)
        k2 = _call_point_function(vector_field, points + 0.5 * dt * k1)
        k3 = _call_point_function(vector_field, points + 0.5 * dt * k2)
        k4 = _call_point_function(vector_field, points + dt * k3)
        points = points + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    point_owner.location = points
    return point_owner


def PhaseFlow(
    mobject,
    function=None,
    *,
    virtual_time: float = 1.0,
    integration_steps: int = 32,
    duration: float = 1.0,
    easing=easings.identity,
):
    """Flow geometry through a vector field using deterministic RK4 integration.

    The result is independent of render frame rate and batch boundaries, unlike
    an updater that repeatedly applies a small Euler step.
    """
    mobject, function = _resolve_mobject_and_callable(
        mobject, function, function_name="PhaseFlow"
    )
    if integration_steps < 1:
        raise ValueError("integration_steps must be at least 1")
    with Sync(
        duration=duration,
        easing=easing,
        animation_manager=animation_manager_for(mobject),
    ):
        for owner in _geometry_point_owners(mobject):
            owner.animate_function(
                _phase_flow_step,
                vector_field=function,
                initial_points=owner.location.clone(),
                virtual_time=float(virtual_time),
                integration_steps=int(integration_steps),
            )
    return mobject


def _path_control_points(path: Mob) -> torch.Tensor:
    paths = [
        descendant.control_points.location
        for descendant in path.get_descendants()
        if isinstance(descendant, BezierCircuitCubic) and not descendant.empty
    ]
    if not paths:
        raise TypeError(
            "MoveAlongPath requires a path containing cubic Bezier geometry."
        )
    frame_count = max(points.shape[0] for points in paths)
    expanded = [
        points if points.shape[0] == frame_count else points.expand(frame_count, -1, -1)
        for points in paths
    ]
    result = torch.cat(expanded, dim=-2)
    if result.shape[-2] % 4 != 0:
        raise ValueError("Bezier path control-point count must be divisible by four.")
    return result


def _bezier_path_point(
    control_points: torch.Tensor,
    alpha: torch.Tensor,
    samples_per_curve: int,
) -> torch.Tensor:
    """Arc-length-parameterized cubic path sampling, vectorized over frames."""
    curves = unsquish(control_points, -2, 4)
    u = torch.linspace(
        0.0,
        1.0,
        samples_per_curve + 1,
        device=control_points.device,
        dtype=control_points.dtype,
    ).view(1, 1, -1, 1)
    one_minus_u = 1.0 - u
    sampled = (
        one_minus_u**3 * curves[..., 0:1, :]
        + 3.0 * one_minus_u**2 * u * curves[..., 1:2, :]
        + 3.0 * one_minus_u * u**2 * curves[..., 2:3, :]
        + u**3 * curves[..., 3:4, :]
    )
    interval_starts = sampled[..., :-1, :].reshape(control_points.shape[0], -1, 3)
    interval_ends = sampled[..., 1:, :].reshape(control_points.shape[0], -1, 3)
    lengths = (interval_ends - interval_starts).norm(dim=-1)
    cumulative = torch.cat(
        (
            torch.zeros_like(lengths[..., :1]),
            lengths.cumsum(dim=-1),
        ),
        dim=-1,
    )
    total = cumulative[..., -1].clamp_min(1e-12)
    target = alpha.clamp(0.0, 1.0) * total
    interval_index = torch.searchsorted(
        cumulative[..., 1:].contiguous(),
        target.unsqueeze(-1).contiguous(),
        right=False,
    ).squeeze(-1)
    interval_index = interval_index.clamp(max=lengths.shape[-1] - 1)
    gather_index = interval_index.view(-1, 1, 1).expand(-1, 1, 3)
    start = interval_starts.gather(1, gather_index).squeeze(1)
    end = interval_ends.gather(1, gather_index).squeeze(1)
    previous_length = cumulative.gather(1, interval_index.unsqueeze(-1)).squeeze(-1)
    interval_length = lengths.gather(1, interval_index.unsqueeze(-1)).squeeze(-1)
    local_alpha = torch.where(
        interval_length > 1e-12,
        (target - previous_length) / interval_length,
        torch.zeros_like(target),
    )
    return torch.lerp(start, end, local_alpha.unsqueeze(-1))


@animated_function(
    animated_args={"t": 0.0},
    unique_args=["path", "initial_location", "initial_center", "samples_per_curve"],
)
def _move_along_path_step(
    mobject,
    t,
    path,
    initial_location,
    initial_center,
    samples_per_curve,
):
    t = cast_to_tensor(t).to(initial_location)
    frame_count = t.shape[0]
    control_points = _path_control_points(path)
    if control_points.shape[0] != frame_count:
        control_points = control_points.expand(frame_count, -1, -1)
    point = _bezier_path_point(
        control_points,
        t.reshape(frame_count, -1)[:, 0],
        int(samples_per_curve),
    )
    displacement = point.view(frame_count, 1, 3) - initial_center.to(point).view(
        1, 1, 3
    )
    mobject.location = initial_location.expand(frame_count, -1, -1) + displacement
    return mobject


def MoveAlongPath(
    mobject: Mob,
    path: Mob,
    *,
    duration: float = 1.0,
    easing=None,
    samples_per_curve: int = 24,
):
    """Move ``mobject`` along the arc length of a cubic Bezier path.

    The path may itself animate in the same context; its materialized control
    points are sampled for every frame.  ``samples_per_curve`` controls only the
    arc-length approximation, not the rendered geometry.
    """
    if not isinstance(mobject, Mob) or not isinstance(path, Mob):
        raise TypeError("MoveAlongPath expects two Mobs.")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")
    # Validate eagerly so a malformed path fails while defining the scene.
    _path_control_points(path)
    with Sync(
        duration=duration,
        easing=easing,
        animation_manager=animation_manager_for(mobject, path),
    ):
        mobject.animate_function(
            _move_along_path_step,
            path=path,
            initial_location=mobject.location.clone(),
            initial_center=mobject.get_center().detach().reshape(-1, 3)[0],
            samples_per_curve=int(samples_per_curve),
        )
    return mobject


__all__ = [
    "ApplyPointwiseFunction",
    "ApplyMatrix",
    "ApplyComplexFunction",
    "Homotopy",
    "ComplexHomotopy",
    "PhaseFlow",
    "MoveAlongPath",
]
