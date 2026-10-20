"""Errors raised inside the callbacks of Homotopy, PhaseFlow and their kin.

These animations accept Algan's vectorized callbacks and Manim's
one-point-at-a-time ones, and tell them apart by calling the function. A
mistake inside the user's function has to reach the user as the function
raised it, not as an error about calling it the other way.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

import algan
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def materialize(*times: float):
    SceneManager.instance().current_scene.timeline_manager.set_state_to_times(
        torch.tensor(times, dtype=torch.get_default_dtype())
    )


def line():
    return algan.Line([0, 0, 0], [1, 0, 0], add_to_scene=False).spawn(False)


def error_chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and every exception it was raised from or during, newest first."""
    chain = []
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return chain


def test_bug_in_a_points_and_time_homotopy_is_raised_as_the_function_raised_it():
    """It used to be retried as f(x, y, z, t) and reported as an argument-count
    TypeError, with the NameError nowhere in the traceback.
    """

    def drift(points, t):
        return points + undefined_offset * t  # noqa: F821

    with pytest.raises(NameError, match="undefined_offset"):
        algan.Homotopy(line(), drift)

    def bad_shape(points, t):
        return points @ torch.eye(2)

    with pytest.raises(RuntimeError) as caught:
        algan.Homotopy(line(), bad_shape)
    assert error_chain(caught.value) == [caught.value]


def test_homotopy_is_only_called_with_the_arguments_its_signature_takes():
    calls = []

    def coordinates(x, y, z, t):
        calls.append(type(x))
        return x, y + t, z

    algan.Homotopy(line(), coordinates)
    assert calls == [torch.Tensor]

    with pytest.raises(TypeError, match="accepts neither"):
        algan.Homotopy(line(), lambda points: points)


def test_every_failed_form_is_reported_and_chained():
    raised = []

    def domain_error(x, y, z, t):
        try:
            return x, y + math.sqrt(-1 - t), z
        except ValueError as error:
            raised.append(error)
            raise

    with pytest.raises(TypeError) as caught:
        algan.Homotopy(line(), domain_error)
    message = str(caught.value)
    assert "Homotopy could not call domain_error" in message
    assert "with torch.Tensors raised ValueError" in message
    assert "with NumPy arrays raised TypeError" in message
    # Read from the error itself: Python 3.14 changed math's wording.
    assert f"at a time raised ValueError: {raised[-1]}" in message
    # Each attempt's own error, with its traceback, is in the chain, the last
    # attempt's first.
    chain = error_chain(caught.value)
    assert [type(error) for error in chain] == [
        TypeError,
        ValueError,
        TypeError,
        ValueError,
    ]
    assert chain[1] is raised[-1]
    assert chain[3] is raised[0]


def test_a_vectorized_point_function_keeps_its_own_error_when_both_forms_fail():
    """The per-point retry's error used to be the only one left to read."""

    def bad_shape(points):
        return points @ torch.eye(2)

    with pytest.raises(TypeError) as caught:
        algan.ApplyPointwiseFunction(line(), bad_shape)
    assert "with every point in one [..., 3] torch.Tensor raised RuntimeError" in str(
        caught.value
    )
    chain = error_chain(caught.value)
    assert isinstance(chain[-1], RuntimeError)
    assert "shapes cannot be multiplied" in str(chain[-1])


@pytest.mark.parametrize(
    "error",
    [
        NameError("name 'scale' is not defined"),
        ModuleNotFoundError("No module named 'scipy_typo'"),
        MemoryError(),
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory"),
        RuntimeError(
            "[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 8589934592 bytes."
        ),
    ],
    ids=["NameError", "ImportError", "MemoryError", "CUDA", "driver", "CPU"],
)
def test_errors_no_calling_form_explains_are_raised_without_a_retry(error):
    """An out-of-memory error in particular has to reach the render loop, which
    retries the batch with fewer frames; it used to become a TypeError after a
    retry on the CPU, one point at a time.
    """
    calls = []

    def field(points):
        calls.append(points.shape)
        raise error

    with pytest.raises(type(error)) as caught:
        algan.PhaseFlow(line(), field, integration_steps=1)
    assert caught.value is error
    assert len(calls) == 1


def test_phase_flow_keeps_the_form_that_worked_across_batches():
    """A Manim-style field also runs on a whole batch of exactly three frames,
    where ``p[0..2]`` index frames instead of coordinates. It used to be called
    that way in such a batch, and flowed the wrong points.
    """
    vectorized_calls = []

    def rotation(p):
        if isinstance(p, torch.Tensor):
            vectorized_calls.append(tuple(p.shape))
        return np.array([-p[1], p[0], p[2] * 0])

    mob = line()
    algan.PhaseFlow(mob, rotation, virtual_time=1.0, integration_steps=4)
    assert len(vectorized_calls) == 1  # once, when recorded

    materialize(0.25, 0.5, 0.75)
    batched = mob.control_points.location.clone()
    singles = []
    for time in (0.25, 0.5, 0.75):
        materialize(time)
        singles.append(mob.control_points.location.clone())
    assert torch.allclose(batched, torch.cat(singles), atol=1e-6)
    assert len(vectorized_calls) == 1


def test_an_error_after_a_form_has_worked_is_the_functions_own():
    """A field that has run on whole tensors is not retried one point at a
    time when it later fails.
    """
    failing = []
    calls = []

    def field(points):
        calls.append(type(points))
        if failing:
            raise ValueError("velocity is undefined here")
        return torch.zeros_like(points)

    algan.PhaseFlow(line(), field, integration_steps=1)
    failing.append(True)
    with pytest.raises(ValueError, match="velocity is undefined here") as caught:
        materialize(0.5)
    assert error_chain(caught.value) == [caught.value]
    assert set(calls) == {torch.Tensor}


def test_complex_function_errors_name_the_animation_and_function_the_user_wrote():
    """They used to surface as ApplyPointwiseFunction failing to call an
    internal lambda on a NumPy point.
    """

    def bad_shape(z):
        return z.conj() @ torch.eye(5)

    with pytest.raises(TypeError) as caught:
        algan.ApplyComplexFunction(line(), bad_shape)
    message = str(caught.value)
    assert message.startswith("ApplyComplexFunction could not call bad_shape;")
    assert "ApplyPointwiseFunction" not in message
    assert "shapes cannot be multiplied" in str(error_chain(caught.value)[-1])


def test_bug_in_a_complex_homotopy_is_raised_as_the_function_raised_it():
    def spin(z, t):
        return z * torch.exp(1j * rate * t)  # noqa: F821

    with pytest.raises(NameError, match="rate"):
        algan.ComplexHomotopy(line(), spin)

    with pytest.raises(TypeError, match="as f\\(z, t\\)"):
        algan.ComplexHomotopy(line(), lambda z: z)


def test_callables_without_a_signature_are_still_accepted():
    mob = line()
    initial = mob.control_points.location.clone()
    algan.ApplyComplexFunction(mob, torch.exp, easing=algan.easings.identity)
    materialize(0.999999)
    z = torch.complex(initial[..., 0], initial[..., 1])
    assert torch.allclose(
        mob.control_points.location[..., 0], torch.exp(z).real, atol=1e-4
    )

    SceneManager.reset()
    mob = line()
    algan.ApplyComplexFunction(mob, np.exp, easing=algan.easings.identity)
    materialize(0.999999)
    assert torch.allclose(
        mob.control_points.location[..., 0], torch.exp(z).real, atol=1e-4
    )


def test_a_decorator_is_called_the_way_it_takes_its_input():
    """``functools.wraps`` gives an adapter the name, and the signature as
    ``inspect`` reads it, of the function it wraps, though it takes its input
    another way. Read that way, PhaseFlow refused an f(points) adapter of an
    f(x, y, z) field, and Homotopy tried an f(points, t) adapter only as
    f(x, y, z, t).
    """

    def taking_points(function):
        @functools.wraps(function)
        def adapter(points, *time):
            x, y, z = points.unbind(-1)
            values = function(x, y, z, *(t[..., 0] for t in time))
            return torch.stack(torch.broadcast_tensors(*values), -1)

        return adapter

    @taking_points
    def swirl(x, y, z):
        return -y, x, z * 0

    @taking_points
    def lift(x, y, z, t):
        return x, y + t, z

    mob = line()
    initial = mob.control_points.location.clone()
    algan.PhaseFlow(mob, swirl)
    materialize(0.999999)
    # The flow of (-y, x, 0) for a unit of time turns a radian about OUT.
    x, y = initial[..., 0], initial[..., 1]
    turned = mob.control_points.location
    assert torch.allclose(turned[..., 0], x * math.cos(1) - y * math.sin(1), atol=1e-4)
    assert torch.allclose(turned[..., 1], x * math.sin(1) + y * math.cos(1), atol=1e-4)

    SceneManager.reset()
    mob = line()
    initial = mob.control_points.location.clone()
    algan.Homotopy(mob, lift, runtime=1, easing=algan.easings.identity)
    materialize(0.999999)
    lifted = mob.control_points.location
    assert torch.allclose(lifted[..., 1], initial[..., 1] + 1, atol=1e-4)


def test_wrong_argument_count_is_reported_when_the_animation_is_recorded():
    with pytest.raises(
        TypeError, match="PhaseFlow calls its function as f\\(points\\)"
    ):
        algan.PhaseFlow(line(), lambda points, t: points)
    with pytest.raises(TypeError, match="as f\\(z\\)"):
        algan.ApplyComplexFunction(line(), lambda z, t: z)
