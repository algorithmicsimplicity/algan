"""What an ``@animated_function`` body receives, recorded or not.

A recorded call casts each animated argument to a tensor before calling the
body, and replay passes a batch of frames, so a body is written against
tensors. Inside ``Off()``, or before the Mob spawned, nothing is recorded, and
the body used to get the raw value instead: ``t.reshape(...)`` that worked in
every animation raised ``AttributeError`` on a plain number there.

A recorded call is kept by parameter name, so its positional arguments have
to be named, and any past the named parameters handed back by position.
"""

from __future__ import annotations

import pytest
import torch

from algan import RIGHT, Off, Square, animated_function
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _recording_function(seen):
    @animated_function(animated_args={"t": 0})
    def slide(mob, t=1, label="slide"):
        seen.append((t, label))
        # What a body written for replay does: treat ``t`` as a frame batch.
        mob.location = mob.location + RIGHT * t.reshape(-1, 1, 1)

    return slide


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda slide, mob: slide(mob), id="default"),
        pytest.param(lambda slide, mob: slide(mob, 0.5), id="positional"),
        pytest.param(lambda slide, mob: slide(mob, t=0.5), id="keyword"),
        pytest.param(lambda slide, mob: slide(mob, torch.tensor(0.5)), id="tensor"),
    ],
)
@pytest.mark.parametrize("where", ["recorded", "off", "unspawned"])
def test_the_body_gets_the_same_tensor_however_it_is_called(call, where):
    seen = []
    slide = _recording_function(seen)
    square = Square()
    if where != "unspawned":
        square.spawn()
    if where == "off":
        with Off():
            call(slide, square)
    else:
        call(slide, square)

    t, label = seen[-1]
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1, 1, 1)
    # Only the animated argument is cast; the rest arrive as passed.
    assert label == "slide"


def test_off_lands_where_the_recorded_animation_ends():
    seen = []
    slide = _recording_function(seen)
    animated = Square().spawn()
    instant = Square().spawn()

    slide(animated, 0.75)
    with Off():
        slide(instant, 0.75)

    assert torch.allclose(animated.location, instant.location)


def test_a_function_without_animated_arguments_is_called_as_given():
    seen = []

    @animated_function
    def tag(mob, value=3):
        seen.append(value)

    square = Square().spawn()
    with Off():
        tag(square)
        tag(square, 2.5)

    assert seen == [3, 2.5]


def _call_where(where, function, *args, **kwargs):
    square = Square()
    if where != "unspawned":
        square.spawn()
    if where == "off":
        with Off():
            return function(square, *args, **kwargs)
    return function(square, *args, **kwargs)


@pytest.mark.parametrize("where", ["recorded", "off", "unspawned"])
def test_arguments_past_the_named_ones_reach_the_body_in_order(where):
    """REGRESSION. A recorded call named its positional arguments by counting
    through every parameter, so the ones meant for ``*steps`` came back as
    keywords -- ``steps=2.0`` and ``scale=3.0``, clobbering the real
    ``scale`` -- and the body raised ``TypeError``.
    """
    seen = []

    @animated_function(animated_args={"t": 0})
    def slide(mob, t, *steps, scale=1):
        seen.append((steps, scale))
        mob.location = mob.location + RIGHT * t * sum(steps) * scale

    _call_where(where, slide, 1, 2.0, 3.0, scale=2)
    assert seen[-1] == ((2.0, 3.0), 2)


@pytest.mark.parametrize("where", ["recorded", "off", "unspawned"])
def test_one_positional_argument_too_many_raises_as_a_direct_call_does(where):
    """REGRESSION. Recorded, it ran the count past the parameter list and
    raised ``IndexError``; unrecorded, Python raises ``TypeError``.
    """

    @animated_function(animated_args={"amount": 0})
    def nudge(mob, amount):
        mob.location = mob.location + RIGHT * amount

    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were"):
        _call_where(where, nudge, 1, 2)
