"""Batch preparation's host-side shortcuts reproduce the tensor paths bit for bit.

Each shortcut here replaces small tensor ops that every render batch runs
hundreds of times; none is allowed to change a materialized value:

* replay locates each recorded function's frames by bisecting the batch's
  times on the host (``_SortedFrameTimes``) instead of a mask, ``nonzero`` and
  a gather per function (``ALGAN_OPT_DISABLE=framewindows`` restores those);
* triangle, mesh and PN-mesh primitives take their colors as plain tensors,
  not the ``Color`` subclass, which routed every op of the primitive build and
  the collection merge through ``__torch_function__``.
"""

import math

import numpy as np
import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan import BLUE, RED, RIGHT, UP, Cube, Square
from algan.animation_timeline.animation_contexts import Lag, Off, Seq, Sync
from algan.animation_timeline.timeline import _SortedFrameTimes
from algan.constants import easings
from algan.constants.color import Color
from algan.mobs.pn_mesh import PNMesh
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()


def _mask_window(times, start, end):
    """The tensor path's frames, as ``(first, stop)`` (the run is contiguous)."""
    inds = ((start <= times) & (times < end)).nonzero().view(-1).tolist()
    return (inds[0], inds[-1] + 1) if inds else None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_host_frame_windows_match_the_tensor_masks(dtype):
    g = torch.Generator().manual_seed(0)
    times = (torch.arange(90, dtype=torch.float64) / 30 + 1 / 7).to(dtype)
    host = _SortedFrameTimes.for_times(times)
    assert host is not None
    samples = times.tolist()
    bounds = [float(v) for v in torch.rand(400, generator=g, dtype=torch.float64) * 4]
    # On a sample, and one double-precision step either side of it: a bound a
    # hair above a float32 sample rounds onto it, so the tensor comparison
    # selects that sample although the double comparison would not.
    for value in samples[::7]:
        bounds += [value, math.nextafter(value, -math.inf), value + 2e-9, value - 2e-9]
    bounds += [0, 1, -1.0, 1e12, math.inf]
    for start in bounds[::3]:
        for end in bounds:
            window = host.window(start, end)
            expected = _mask_window(times, start, end)
            assert (window if window[1] > window[0] else None) == expected, (
                start,
                end,
            )


def test_host_frame_windows_decline_what_they_cannot_reproduce():
    assert _SortedFrameTimes.for_times(torch.tensor([0.0, 0.2, 0.1])) is None
    assert (
        _SortedFrameTimes.for_times(torch.tensor([0.0, 0.1], dtype=torch.float16))
        is None
    )
    assert _SortedFrameTimes.for_times(torch.tensor([math.nan, 0.1])) is None
    assert _SortedFrameTimes.for_times(torch.tensor([0.0, math.nan])) is None
    host = _SortedFrameTimes.for_times(torch.tensor([0.0, 0.1, 0.2]))
    # A bound that is not a Python number is left to the tensor path.
    assert host.window(torch.tensor(0.05), 0.15) is None
    assert host.window(np.float32(0.05), 0.15) is None
    # A NaN start selects nothing in the mask but would bisect to frame 0.
    assert host.window(math.nan, 0.15) is None
    # A NaN end selects nothing either way: an empty window.
    first, stop = host.window(0.05, math.nan)
    assert first == stop


def _author():
    a = Square().spawn(animate=False)
    b = Square().move(RIGHT * 2).spawn(animate=False)
    c = Cube().move(UP * 2).spawn(animate=False)
    with Seq(easing=easings.smooth):
        a.move(RIGHT)
        with Sync():
            b.color = BLUE
            c.rotate(40, UP)
        with Lag(0.5):
            a.color = RED
            b.move(UP)
            c.move(RIGHT * 0.5)
    return [a, b, c]


def _materialized_state(times):
    """Every attribute's materialized window at ``times``, as the render loop
    replays it.
    """
    scene = SceneManager.instance().current_scene
    tm = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        tm.set_state_to_times(times, active_mobs=list(scene.actors))
        state = {
            name: timeline.active_state.clone()
            for name, timeline in tm.attr_to_timeline.items()
        }
    tm.clear_buffers()
    return state


@pytest.mark.parametrize(
    "times",
    [
        torch.arange(0, 140) / 30.0,  # one batch over the whole scene
        torch.arange(37, 61) / 30.0,  # a window starting inside an animation
        torch.tensor([0.5, 1.25, 1.26, 3.9]),  # sparse frames (stills)
    ],
)
def test_replay_through_host_windows_is_bit_identical(monkeypatch, times):
    _author()
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({"framewindows"}))
    reference = _materialized_state(times)
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset())
    fast = _materialized_state(times)
    assert reference.keys() == fast.keys()
    for name in reference:
        assert torch.equal(reference[name], fast[name]), name


def test_triangle_and_pn_primitives_carry_plain_tensor_colors():
    cube = Cube(color=BLUE).spawn(animate=False)
    face = next(
        d for d in cube.faces.get_descendants() if hasattr(d, "get_render_primitives")
    )
    assert isinstance(face.color, Color)
    primitive = face.get_render_primitives()
    assert type(primitive.colors) is torch.Tensor
    pn = PNMesh(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]] * 3),
    ).spawn(animate=False)
    assert type(pn.get_render_primitives().colors) is torch.Tensor
