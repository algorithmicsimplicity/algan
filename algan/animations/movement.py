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

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Sync, animation_manager_for
from algan.constants import easings
from algan.constants.spatial import ORIGIN
from algan.errors import AlganConfigurationError
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.surfaces.surface import Surface
from algan.utils.api_renames import _renamed_keywords
from algan.utils.memory_utils import InsufficientMemoryException, is_out_of_memory
from algan.utils.tensor_utils import cast_to_tensor, unsquish


def _resolve_mob_and_callable(first, second, *, function_name: str):
    """Accept both Algan's ``(mob, function)`` and Manim's reverse order."""
    if isinstance(first, Mob) and callable(second):
        return first, second
    if callable(first) and isinstance(second, Mob):
        return second, first
    raise TypeError(
        f"{function_name} expects a Mob and a callable; received "
        f"{type(first).__name__} and {type(second).__name__}."
    )


def _geometry_point_owners(mob: Mob) -> list[Mob]:
    """Return fixed-size point buffers which define ``mob``'s geometry.

    Bezier circuits render from their control-point component; surfaces render
    from their sampled grid.  Modifying those child Mobs gives pointwise
    animations a common implementation for 2-D paths, text, meshes, imported
    models, and the sphere batches used by point clouds.
    """
    owners: list[Mob] = []
    seen: set[int] = set()
    for descendant in mob.get_descendants():
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
            f"{type(mob).__name__} has no deformable Bezier or surface geometry."
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
            raise AlganConfigurationError(
                "Point mapping functions must return the same shape as their input; "
                f"got {tuple(result.shape)} for {tuple(reference.shape)}."
            ) from exc
    return result


def _coerce_complex_result(result: Any, z: torch.Tensor) -> torch.Tensor:
    result = _as_tensor_like(result, z)
    if result.shape != z.shape:
        try:
            result = result.expand_as(z)
        except RuntimeError as exc:
            raise AlganConfigurationError(
                "Complex functions must return one value for each input value; "
                f"got shape {tuple(result.shape)} for {tuple(z.shape)}."
            ) from exc
    return result


# Calling a user's function
# -------------------------
# These animations accept Algan's vectorized callbacks, which take every point
# at once as a tensor, and Manim's, which take one point -- or one coordinate
# per argument -- at a time. Nothing about a function says which it was written
# for, so it is called in each accepted form in turn until one works.


#: Errors a function raises however its input is passed, so they never mean it
#: expects its input in another form: an undefined name or a failed import is a
#: bug in the function, and exhausted memory has to reach the render loop, which
#: retries the batch with fewer frames.
_ERRORS_NO_FORM_EXPLAINS = (
    NameError,
    ImportError,
    MemoryError,
    RecursionError,
    InsufficientMemoryException,
)


def _could_mean_another_form(exc: Exception) -> bool:
    """Whether ``exc`` might only mean the function takes its input another way.

    A function written for one point at a time typically fails on a whole
    tensor where it first indexes or converts its input -- a TypeError,
    ValueError, IndexError or RuntimeError -- so any error but those in
    ``_ERRORS_NO_FORM_EXPLAINS`` and out-of-memory errors might.
    """
    return not isinstance(exc, _ERRORS_NO_FORM_EXPLAINS) and not is_out_of_memory(exc)


def _callable_name(function: Callable) -> str:
    return getattr(function, "__name__", None) or type(function).__name__


def _describe_callable(function: Callable) -> str:
    """``function``'s name and signature, as an error message shows it."""
    try:
        # Not following __wrapped__: a decorator may take its input another
        # way than the function it wraps, and the call binds to the decorator.
        signature = inspect.signature(function, follow_wrapped=False)
    except (TypeError, ValueError):
        return _callable_name(function)
    return f"{_callable_name(function)}{signature}"


def _accepts_positional(function: Callable, count: int) -> bool | None:
    """Whether ``function`` can be called with ``count`` positional arguments.

    ``None`` when Python cannot tell, as for some builtins; every form is then
    tried.
    """
    try:
        signature = inspect.signature(function, follow_wrapped=False)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind(*range(count))
    except TypeError:
        return False
    return True


def _summarize_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message.splitlines()[0]}"


class _UserFunction:
    """A user's function, called in the first form it accepts.

    ``forms`` are ``(description, call)`` pairs, tried in order, where
    ``call(function, *inputs)`` calls the function in that form and returns
    its result as a tensor, and ``description`` says how it calls it, with
    ``{name}`` standing for the function's name.

    An error that might only mean the function takes its input another way
    (see :func:`_could_mean_another_form`) moves on to the next form. Any other
    error, or the error from the last form, is raised as the function raised
    it -- unless several forms were tried and every one failed: then a
    :class:`TypeError` says how each was called and what it raised, and the
    traceback above it shows each call's own error, first call first.

    The first form that works is kept, and later calls use only it: the
    function has shown it takes its input that way, so a later error is its
    own and is raised as it raised it. So one animation does not switch forms
    from one batch of frames to the next.
    """

    def __init__(
        self,
        animation: str,
        function: Callable,
        forms: Sequence[tuple[str, Callable]],
    ):
        self.animation = animation
        self.function = function
        name = _callable_name(function)
        self.forms = tuple(
            (description.format(name=name), call) for description, call in forms
        )
        self.working_form: Callable | None = None

    def __call__(self, *inputs):
        if self.working_form is not None:
            return self.working_form(self.function, *inputs)
        failures: list[str] = []
        try:
            return self._call_from(0, inputs, failures)
        except Exception as exc:
            if len(failures) < 2 or not _could_mean_another_form(exc):
                raise
            attempts = "\n".join(f"  {failure}" for failure in failures)
            raise TypeError(
                f"{self.animation} could not call {_callable_name(self.function)}; "
                f"every way it tried failed:\n{attempts}\n"
                "Each call's own error is shown above, first call first."
            ) from exc

    def _call_from(self, index, inputs, failures):
        description, call = self.forms[index]
        try:
            result = call(self.function, *inputs)
        except Exception as exc:
            # Only the summary is kept: the exception itself, held here, would
            # tie its traceback's frames -- and their tensors -- in a cycle.
            failures.append(f"{description} raised {_summarize_error(exc)}")
            if index + 1 == len(self.forms) or not _could_mean_another_form(exc):
                raise
            # Inside this handler, so the next form's error is chained to this
            # one and the traceback shows both.
            return self._call_from(index + 1, inputs, failures)
        self.working_form = call
        return result


def _require_positional(
    animation: str, function: Callable, count: int, form: str
) -> None:
    if _accepts_positional(function, count) is False:
        raise TypeError(
            f"{animation} calls its function as {form}, which "
            f"{_describe_callable(function)} does not accept."
        )


def _on_points(function, points):
    return _coerce_mapping_result(function(points), points)


def _on_each_numpy_point(function, points):
    flat = points.detach().cpu().reshape(-1, 3).numpy()
    mapped = np.asarray([function(point) for point in flat], dtype=np.float64)
    return torch.as_tensor(
        mapped,
        device=points.device,
        dtype=points.dtype,
    ).reshape_as(points)


_POINT_FUNCTION_FORMS = (
    ("{name}(points) with every point in one [..., 3] torch.Tensor", _on_points),
    ("{name}(point) with one NumPy point at a time", _on_each_numpy_point),
)


def _point_function(animation: str, function: Callable) -> _UserFunction:
    """Wrap a point mapping, or a vector field, as ``f(points) -> points``.

    The fast form is PyTorch-native because ``PhaseFlow`` also calls it while
    materializing frames; the NumPy form keeps Manim-style ``lambda p: ...``
    callables working for one-off pointwise transforms.
    """
    _require_positional(animation, function, 1, "f(points)")
    return _UserFunction(animation, function, _POINT_FUNCTION_FORMS)


def _per_point_times(points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return t.reshape(points.shape[0], -1)[:, :1].expand(-1, points.shape[1])


def _homotopy_on_points(function, points, t):
    return _coerce_mapping_result(function(points, t), points)


def _homotopy_on_coordinate_tensors(function, points, t):
    x, y, z = points.unbind(-1)
    result = function(x, y, z, _per_point_times(points, t))
    if isinstance(result, (tuple, list)) and len(result) == 3:
        result = torch.stack(
            [_as_tensor_like(value, points) for value in result],
            dim=-1,
        )
    return _coerce_mapping_result(result, points)


def _homotopy_on_coordinate_arrays(function, points, t):
    points_np = points.detach().cpu().numpy()
    times_np = _per_point_times(points, t).detach().cpu().numpy()
    result = function(
        points_np[..., 0],
        points_np[..., 1],
        points_np[..., 2],
        times_np,
    )
    if isinstance(result, (tuple, list)) and len(result) == 3:
        result = np.stack(result, axis=-1)
    return torch.as_tensor(result, device=points.device, dtype=points.dtype).reshape_as(
        points
    )


def _homotopy_on_each_point(function, points, t):
    points_np = points.detach().cpu().numpy()
    times_np = _per_point_times(points, t).detach().cpu().numpy()
    mapped = [
        function(float(x), float(y), float(z), float(time))
        for frame_points, frame_times in zip(points_np, times_np)
        for (x, y, z), time in zip(frame_points, frame_times)
    ]
    return torch.as_tensor(
        np.asarray(mapped), device=points.device, dtype=points.dtype
    ).reshape_as(points)


_COORDINATE_HOMOTOPY_FORMS = (
    ("{name}(x, y, z, t) with torch.Tensors", _homotopy_on_coordinate_tensors),
    ("{name}(x, y, z, t) with NumPy arrays", _homotopy_on_coordinate_arrays),
    ("{name}(x, y, z, t) with one point's floats at a time", _homotopy_on_each_point),
)


def _homotopy_function(animation: str, function: Callable) -> _UserFunction:
    """Wrap Algan's ``f(points, t)`` or Manim's ``f(x, y, z, t)`` homotopy.

    The function's signature picks between the two, so a function that takes
    only one of them is only called that way: an error inside it is raised
    as it raised it, instead of being retried with the wrong arguments.
    """
    forms = []
    if _accepts_positional(function, 2) is not False:
        forms.append(("{name}(points, t) with torch.Tensors", _homotopy_on_points))
    if _accepts_positional(function, 4) is not False:
        forms.extend(_COORDINATE_HOMOTOPY_FORMS)
    if not forms:
        raise TypeError(
            f"{animation} calls its function as f(points, t) or f(x, y, z, t), "
            f"and {_describe_callable(function)} accepts neither."
        )
    return _UserFunction(animation, function, forms)


def _complex_on_tensors(function, z, *rest):
    return _coerce_complex_result(function(z, *rest), z)


def _complex_on_arrays(function, z, *rest):
    arrays = [value.detach().cpu().numpy() for value in (z, *rest)]
    return _coerce_complex_result(function(*arrays), z)


def _complex_on_each_value(function, z, *rest):
    arrays = [value.detach().cpu().numpy() for value in (z, *rest)]
    return _coerce_complex_result(np.vectorize(function)(*arrays), z)


def _complex_function(animation: str, function: Callable) -> _UserFunction:
    """Wrap a complex map ``f(z)``."""
    _require_positional(animation, function, 1, "f(z)")
    return _UserFunction(
        animation,
        function,
        (
            ("{name}(z) with one complex torch.Tensor", _complex_on_tensors),
            ("{name}(z) with one complex NumPy array", _complex_on_arrays),
            ("{name}(z) with one complex number at a time", _complex_on_each_value),
        ),
    )


def _complex_homotopy_function(animation: str, function: Callable) -> _UserFunction:
    """Wrap a complex homotopy ``f(z, t)``."""
    _require_positional(animation, function, 2, "f(z, t)")
    return _UserFunction(
        animation,
        function,
        (
            ("{name}(z, t) with torch.Tensors", _complex_on_tensors),
            ("{name}(z, t) with NumPy arrays", _complex_on_arrays),
            ("{name}(z, t) with one value of each at a time", _complex_on_each_value),
        ),
    )


def _map_complex_plane(complex_function: Callable, points: torch.Tensor):
    """Apply ``complex_function`` to the x-y plane, keeping each point's z."""
    z = torch.complex(points[..., 0], points[..., 1])
    result = complex_function(z)
    return torch.stack((result.real, result.imag, points[..., 2]), dim=-1).to(points)


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


def _record_point_mapping(
    mob: Mob,
    mapping: Callable,
    *,
    about_point,
    runtime: float,
    easing,
) -> Mob:
    """Animate every point of ``mob``'s geometry to ``mapping`` of it.

    ``mapping`` takes and returns a ``[..., 3]`` tensor of points relative to
    ``about_point``.
    """
    about_point = cast_to_tensor(about_point)
    with Sync(
        runtime=runtime,
        easing=easing,
        animation_manager=animation_manager_for(mob),
    ):
        for owner in _geometry_point_owners(mob):
            initial = owner.location.clone()
            centered = initial - about_point.to(initial)
            target = mapping(centered) + about_point.to(initial)
            owner.animate_function(
                _pointwise_interpolation_step,
                initial_points=initial,
                target_points=target,
            )
    return mob


@_renamed_keywords(mobject="mob")
def ApplyPointwiseFunction(
    mob: Mob | Callable,
    function: Callable | Mob | None = None,
    *,
    about_point: torch.Tensor = ORIGIN,
    runtime: float = 1.0,
    easing: Callable | None = None,
) -> Mob:
    """Move every point of a Mob's geometry to where a function sends it.

    Each point travels in a straight line from where it is to ``function`` of
    it, so any reshaping you can write as a formula -- bending a line into an
    arc, rounding a square off into a circle -- becomes an animation. Paths,
    text, surfaces, meshes and imported models are all reshaped.

    Animation
    ---------
    Recorded as an animation of ``runtime`` seconds; an enclosing context's
    ``runtime`` rescales it like any other animation, and inside ``Off()`` the
    points move at once. ``function`` runs when the animation is recorded, for
    each part of the Mob, and never while rendering. Applies to the Mob and all
    of its descendants.

    Parameters
    ----------
    mob
        The Mob to reshape. Manim's order, ``ApplyPointwiseFunction(function,
        mob)``, is accepted too.
    function
        The point mapping, relative to ``about_point``. It is called first with
        every point at once, as a ``torch.Tensor`` of shape ``(*, 3)``, and must
        then return a tensor of the same shape: this is the fast form. If that
        call raises, it is called with one point at a time instead, as a NumPy
        array of shape ``(3,)``, the way Manim calls it. An undefined name, a
        failed import or running out of memory is raised at once rather than
        retried. It has no usable default: the ``None`` in the signature only
        lets the two argument orders share it.
    about_point
        Point the mapping is centered on, shape ``(3,)``: ``function`` receives
        each point relative to it, and its result is placed relative to it.
        Defaults to ``ORIGIN``.
    runtime
        How long the points take to arrive, in seconds. Defaults to ``1.0``.
    easing
        Easing applied to the points' progress. Defaults to ``None``, meaning
        the enclosing context's (``easings.smooth`` at the top level).

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    TypeError
        If the arguments are not a Mob and a callable, if ``function`` cannot
        take a single argument, if the Mob has no deformable geometry, or if
        ``function`` fails both ways it is called, where returning points of a
        different shape from those it was given counts as failing. That last
        error lists what each call raised, and the traceback above it shows
        each call's own error.

    Examples
    --------
    Pull a square's points onto the unit circle, rounding it off:

    .. algan:: Example1ApplyPointwiseFunction

        from algan import *

        square = Square().spawn()
        ApplyPointwiseFunction(
            square, lambda points: points / points.norm(dim=-1, keepdim=True)
        )

        Scene.save_video()
    """
    mob, function = _resolve_mob_and_callable(
        mob, function, function_name="ApplyPointwiseFunction"
    )
    return _record_point_mapping(
        mob,
        _point_function("ApplyPointwiseFunction", function),
        about_point=about_point,
        runtime=runtime,
        easing=easing,
    )


@_renamed_keywords(mobject="mob")
def ApplyMatrix(
    mob,
    matrix=None,
    *,
    about_point=ORIGIN,
    runtime: float = 1.0,
    easing=None,
):
    """Animate a 2×2 or 3×3 linear transform about ``about_point``.

    Accepts both ``ApplyMatrix(mob, matrix)`` and Manim's
    ``ApplyMatrix(matrix, mob)`` argument order.
    """
    if not isinstance(mob, Mob) and isinstance(matrix, Mob):
        mob, matrix = matrix, mob
    if not isinstance(mob, Mob):
        raise TypeError("ApplyMatrix expects a Mob and a 2x2 or 3x3 matrix.")
    matrix = torch.as_tensor(matrix, dtype=torch.get_default_dtype())
    if matrix.shape == (2, 2):
        full_matrix = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        full_matrix[:2, :2] = matrix
        matrix = full_matrix
    elif matrix.shape != (3, 3):
        raise AlganConfigurationError("matrix must have shape (2, 2) or (3, 3)")

    return ApplyPointwiseFunction(
        mob,
        lambda points: points @ matrix.to(points).transpose(-1, -2),
        about_point=about_point,
        runtime=runtime,
        easing=easing,
    )


@_renamed_keywords(mobject="mob")
def ApplyComplexFunction(
    mob: Mob | Callable,
    function: Callable | Mob | None = None,
    *,
    about_point: torch.Tensor = ORIGIN,
    runtime: float = 1.0,
    easing: Callable | None = None,
) -> Mob:
    """Move a Mob's points to where a complex function sends them.

    The x-y plane is read as the complex plane, ``z = x + iy``, and each point
    travels in a straight line to ``function(z)``, keeping its own depth (its z
    coordinate). Conformal maps such as ``z ** 2`` or ``torch.exp(z)`` are the
    usual subject.

    Animation
    ---------
    Recorded as an animation of ``runtime`` seconds; an enclosing context's
    ``runtime`` rescales it like any other animation, and inside ``Off()`` the
    points move at once. ``function`` runs when the animation is recorded, for
    each part of the Mob, and never while rendering. Applies to the Mob and all
    of its descendants.

    Parameters
    ----------
    mob
        The Mob to transform. Manim's order, ``ApplyComplexFunction(function,
        mob)``, is accepted too.
    function
        The complex map, relative to ``about_point``. It is called with every
        point at once as a complex ``torch.Tensor``; if that raises, as a complex
        NumPy array; and if that raises too, with one complex number at a time.
        It must return one complex value for each value it is given. An
        undefined name, a failed import or running out of memory is raised at
        once rather than retried. It has no usable default: the ``None`` in the
        signature only lets the two argument orders share it.
    about_point
        Point treated as ``z = 0``, shape ``(3,)``. Defaults to ``ORIGIN``.
    runtime
        How long the points take to arrive, in seconds. Defaults to ``1.0``.
    easing
        Easing applied to the points' progress. Defaults to ``None``, meaning
        the enclosing context's (``easings.smooth`` at the top level).

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    TypeError
        If the arguments are not a Mob and a callable, if ``function`` cannot
        take a single argument, if the Mob has no deformable geometry, or if
        ``function`` fails every way it is called, where returning a different
        number of values from those it was given counts as failing. That last
        error lists what each call raised, and the traceback above it shows
        each call's own error.

    Examples
    --------
    Wrap a square into a ring sector with the complex exponential:

    .. algan:: Example1ApplyComplexFunction

        from algan import *
        import torch

        square = Square().spawn()
        ApplyComplexFunction(square, lambda z: torch.exp(z))

        Scene.save_video()
    """
    mob, function = _resolve_mob_and_callable(
        mob, function, function_name="ApplyComplexFunction"
    )
    complex_function = _complex_function("ApplyComplexFunction", function)
    return _record_point_mapping(
        mob,
        lambda points: _map_complex_plane(complex_function, points),
        about_point=about_point,
        runtime=runtime,
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
    point_owner.location = homotopy_func(points, t.reshape(frame_count, 1, 1))
    return point_owner


@_renamed_keywords(mobject="mob")
def Homotopy(
    mob: Mob | Callable,
    homotopy_func: Callable | Mob | None = None,
    *,
    runtime: float = 2.0,
    easing: Callable | None = None,
) -> Mob:
    """Deform a Mob's geometry continuously, with a function of position and time.

    A homotopy says where each point is at every moment of the animation, not
    only where it ends up, so a shape can bend, ripple or swirl on its way. At
    progress ``t`` each point is at ``homotopy_func`` of its starting position
    and ``t``; for the animation to start where the Mob is, the function should
    return every point unchanged at ``t = 0``. Paths, text, surfaces, meshes
    and imported models are all deformed.

    Animation
    ---------
    Recorded as an animation of ``runtime`` seconds, over which ``t`` sweeps
    from ``0`` to ``1``; an enclosing context's ``runtime`` rescales it like any
    other animation, and inside ``Off()`` the deformation at ``t = 1`` is
    applied at once. ``homotopy_func`` is called again for every batch of
    frames while rendering. Applies to the Mob and all of its descendants.

    Parameters
    ----------
    mob
        The Mob to deform. Manim's order, ``Homotopy(homotopy_func, mob)``, is
        accepted too.
    homotopy_func
        The deformation, in one of two forms, told apart by how many arguments
        it takes:

        * ``f(points, t)``, Algan's form. ``points`` holds every starting point
          at once, as a ``torch.Tensor`` of shape ``(*, 3)``; ``t`` is a tensor
          that broadcasts against it; the result has the shape of ``points``.
        * ``f(x, y, z, t)``, Manim's form, returning ``(x, y, z)``. It is
          called with tensors of coordinates first, then with NumPy arrays, then
          one point at a time with floats, until one works, and then keeps
          being called that way.

        A function that takes only ``(points, t)`` is only ever called that
        way, so an error inside it is raised as it was raised. An undefined
        name, a failed import or running out of memory is raised at once
        rather than retried. It has no usable default: the ``None`` in the
        signature only lets the two argument orders share it.
    runtime
        How long the deformation lasts, in seconds. Defaults to ``2.0``.
    easing
        Easing applied to ``t``. Defaults to ``None``, meaning the enclosing
        context's (``easings.smooth`` at the top level).

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    TypeError
        If the arguments are not a Mob and a callable, if ``homotopy_func``
        takes neither form, if the Mob has no deformable geometry, or if every
        way of calling ``homotopy_func`` fails, where returning points of a
        different shape from those it was given counts as failing. That last
        error lists what each call raised, and the traceback above it shows
        each call's own error.
    :class:`.AlganConfigurationError`
        If a ``homotopy_func`` that takes only ``(points, t)`` returns points of
        a different shape from those it was given.

    Examples
    --------
    Ripple a square, and let it settle flat again:

    .. algan:: Example1Homotopy

        from algan import *
        import torch

        square = Square().spawn()


        def ripple(x, y, z, t):
            return x, y + 0.3 * torch.sin(3 * x) * torch.sin(torch.pi * t), z


        Homotopy(square, ripple)

        Scene.save_video()
    """
    mob, homotopy_func = _resolve_mob_and_callable(
        mob, homotopy_func, function_name="Homotopy"
    )
    homotopy = _homotopy_function("Homotopy", homotopy_func)
    with Sync(
        runtime=runtime,
        easing=easing,
        animation_manager=animation_manager_for(mob),
    ):
        for owner in _geometry_point_owners(mob):
            owner.animate_function(
                _homotopy_step,
                homotopy_func=homotopy,
                initial_points=owner.location.clone(),
            )
    return mob


@_renamed_keywords(mobject="mob")
def ComplexHomotopy(
    mob: Mob | Callable,
    complex_homotopy: Callable | Mob | None = None,
    *,
    runtime: float = 2.0,
    easing: Callable | None = None,
) -> Mob:
    """Deform a Mob continuously in the complex plane, with a function of time.

    The x-y plane is read as the complex plane, ``z = x + iy``: at progress
    ``t`` each point is at ``complex_homotopy(z, t)`` of its starting position,
    and keeps its own depth (its z coordinate). It is the complex counterpart
    of :func:`Homotopy`; for the animation to start where the Mob is, the
    function should return ``z`` unchanged at ``t = 0``.

    Animation
    ---------
    Recorded as an animation of ``runtime`` seconds, over which ``t`` sweeps
    from ``0`` to ``1``; an enclosing context's ``runtime`` rescales it like any
    other animation, and inside ``Off()`` the deformation at ``t = 1`` is
    applied at once. ``complex_homotopy`` is called again for every batch of
    frames while rendering. Applies to the Mob and all of its descendants.

    Parameters
    ----------
    mob
        The Mob to deform. Manim's order, ``ComplexHomotopy(complex_homotopy,
        mob)``, is accepted too.
    complex_homotopy
        The deformation ``f(z, t)``. It is called with every point at once, as a
        complex ``torch.Tensor`` ``z`` and a real one ``t`` of the same shape; if
        that raises, with NumPy arrays; and if that raises too, with one complex
        number and time at a time. Whichever works first is kept. It must return
        one complex value for each value of ``z``. An undefined name, a failed
        import or running out of memory is raised at once rather than retried.
        It has no usable default: the ``None`` in the signature only lets the
        two argument orders share it.
    runtime
        How long the deformation lasts, in seconds. Defaults to ``2.0``.
    easing
        Easing applied to ``t``. Defaults to ``None``, meaning the enclosing
        context's (``easings.smooth`` at the top level).

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    TypeError
        If the arguments are not a Mob and a callable, if ``complex_homotopy``
        cannot take two arguments, if the Mob has no deformable geometry, or if
        every way of calling ``complex_homotopy`` fails, where returning a
        different number of values from those it was given counts as failing.
        That last error lists what each call raised, and the traceback above
        it shows each call's own error.

    Examples
    --------
    Twist a square, its corners further than its middle:

    .. algan:: Example1ComplexHomotopy

        from algan import *
        import torch

        square = Square().spawn()


        def twist(z, t):
            return z * torch.exp(1j * torch.pi / 2 * t * z.abs())


        ComplexHomotopy(square, twist)

        Scene.save_video()
    """
    mob, complex_homotopy = _resolve_mob_and_callable(
        mob, complex_homotopy, function_name="ComplexHomotopy"
    )
    complex_function = _complex_homotopy_function("ComplexHomotopy", complex_homotopy)

    def homotopy(points, t):
        z = torch.complex(points[..., 0], points[..., 1])
        time = t.reshape(points.shape[0], -1)[:, :1].expand_as(z.real)
        result = complex_function(z, time)
        return torch.stack((result.real, result.imag, points[..., 2]), dim=-1).to(
            points
        )

    return Homotopy(
        mob,
        homotopy,
        runtime=runtime,
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
        k1 = vector_field(points)
        k2 = vector_field(points + 0.5 * dt * k1)
        k3 = vector_field(points + 0.5 * dt * k2)
        k4 = vector_field(points + dt * k3)
        points = points + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    point_owner.location = points
    return point_owner


@_renamed_keywords(mobject="mob")
def PhaseFlow(
    mob: Mob | Callable,
    function: Callable | Mob | None = None,
    *,
    virtual_time: float = 1.0,
    integration_steps: int = 32,
    runtime: float = 1.0,
    easing: Callable | None = easings.identity,
) -> Mob:
    """Carry a Mob's geometry along a vector field, like a leaf on a current.

    Every point follows the field's flow: at each moment it moves with the
    velocity the field gives where it is. The flow is integrated from the
    starting geometry with fourth-order Runge-Kutta, so the result does not
    depend on the frame rate or on how frames are batched, unlike an updater
    that adds a small step every frame.

    Animation
    ---------
    Recorded as an animation of ``runtime`` seconds; an enclosing context's
    ``runtime`` rescales it like any other animation, and inside ``Off()`` the
    whole flow is applied at once. ``function`` is called again while
    rendering, four times per integration step for every batch of frames.
    Applies to the Mob and all of its descendants.

    Parameters
    ----------
    mob
        The Mob to carry. Manim's order, ``PhaseFlow(function, mob)``, is
        accepted too.
    function
        The vector field: the velocity at each point, in world units per unit
        of ``virtual_time``. It is called first with every point at once, as a
        ``torch.Tensor`` of shape ``(*, 3)``, and must then return velocities of
        the same shape: this is the fast form. If that call raises, it is
        called with one point at a time instead, as a NumPy array of shape
        ``(3,)``, the way Manim calls it. Whichever works first is kept for the
        rest of the animation. An undefined name, a failed import or running
        out of memory is raised at once rather than retried. It has no usable
        default: the ``None`` in the signature only lets the two argument
        orders share it.
    virtual_time
        How long to follow the field, in the field's own time: in a field of
        constant velocity ``v`` every point moves ``v * virtual_time``.
        Independent of ``runtime``. Defaults to ``1.0``.
    integration_steps
        Runge-Kutta steps from the start of the flow to each frame; more steps
        follow curving flow lines more closely. Must be at least ``1``.
        Defaults to ``32``.
    runtime
        How long the animation lasts, in seconds. Defaults to ``1.0``.
    easing
        Easing applied to the flow's progress. Defaults to
        ``easings.identity``: the flow advances at a constant rate.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    TypeError
        If the arguments are not a Mob and a callable, if ``function`` cannot
        take a single argument, if the Mob has no deformable geometry, or if
        ``function`` fails both ways it is called, where returning velocities
        of a different shape from the points it was given counts as failing.
        That last error lists what each call raised, and the traceback above
        it shows each call's own error.
    :class:`.AlganConfigurationError`
        If ``integration_steps`` is less than ``1``.

    Examples
    --------
    Turn four dots a quarter of the way around the origin:

    .. algan:: Example1PhaseFlow

        from algan import *
        import torch

        dots = Group([Dot(location=RIGHT * r, radius=0.15) for r in (0.5, 1, 1.5, 2)])
        dots.spawn()


        def rotation(points):
            x, y, z = points.unbind(-1)
            return torch.stack((-y, x, torch.zeros_like(z)), -1)


        PhaseFlow(dots, rotation, virtual_time=torch.pi / 2)

        Scene.save_video()
    """
    mob, function = _resolve_mob_and_callable(mob, function, function_name="PhaseFlow")
    if integration_steps < 1:
        raise AlganConfigurationError("integration_steps must be at least 1")
    vector_field = _point_function("PhaseFlow", function)
    with Sync(
        runtime=runtime,
        easing=easing,
        animation_manager=animation_manager_for(mob),
    ):
        for owner in _geometry_point_owners(mob):
            owner.animate_function(
                _phase_flow_step,
                vector_field=vector_field,
                initial_points=owner.location.clone(),
                virtual_time=float(virtual_time),
                integration_steps=int(integration_steps),
            )
    return mob


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
        raise AlganConfigurationError(
            "Bezier path control-point count must be divisible by four."
        )
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
    mob,
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
    mob.location = initial_location.expand(frame_count, -1, -1) + displacement
    return mob


@_renamed_keywords(mobject="mob")
def MoveAlongPath(
    mob: Mob,
    path: Mob,
    *,
    runtime: float = 1.0,
    easing=None,
    samples_per_curve: int = 24,
):
    """Move ``mob`` along the arc length of a cubic Bezier path.

    The path may itself animate in the same context; its materialized control
    points are sampled for every frame.  ``samples_per_curve`` controls only the
    arc-length approximation, not the rendered geometry.
    """
    if not isinstance(mob, Mob) or not isinstance(path, Mob):
        raise TypeError("MoveAlongPath expects two Mobs.")
    if samples_per_curve < 2:
        raise AlganConfigurationError("samples_per_curve must be at least 2")
    # Validate eagerly so a malformed path fails while defining the scene.
    _path_control_points(path)
    with Sync(
        runtime=runtime,
        easing=easing,
        animation_manager=animation_manager_for(mob, path),
    ):
        mob.animate_function(
            _move_along_path_step,
            path=path,
            initial_location=mob.location.clone(),
            initial_center=mob.get_center().detach().reshape(-1, 3)[0],
            samples_per_curve=int(samples_per_curve),
        )
    return mob


__all__ = [
    "ApplyPointwiseFunction",
    "ApplyMatrix",
    "ApplyComplexFunction",
    "Homotopy",
    "ComplexHomotopy",
    "PhaseFlow",
    "MoveAlongPath",
]
