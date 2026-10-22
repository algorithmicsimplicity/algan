"""The collection merge's and the replayed bodies' shortcuts change no bits.

* ``broadcast_all`` reads every shape once instead of once per dimension;
  ``_broadcast_all_reference`` is the form it replaced
  (``ALGAN_OPT_DISABLE=broadcastall``).
* The triangle merge works each member's surface-param broadcast target out
  once instead of re-broadcasting its corners, colors and normals for every
  param (``mergeshapes``).
* ``_circuit_scale`` takes a circuit's control-point box from one fused
  ``aminmax`` pass at sizes where that wins (``circuitscale``).
* Replayed ``_apply_change`` / ``_apply_absolute_change_two`` bodies read
  uncopied and compute a recorded ``Color`` change as a plain tensor
  (``replaybodies``).
"""

import math
from types import SimpleNamespace

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan import BLUE, RED, RIGHT, UP, Circle, Square
from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants.color import Color
from algan.rendering.raytracing.primitives import (
    _gather_member_surface_params,
    _member_surface_param_reference,
)
from algan.rendering.raytracing.utils import _circuit_scale, _control_point_box
from algan.scene_manager import SceneManager
from algan.utils.tensor_utils import _broadcast_all_reference, broadcast_all


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()


def _same_view(a, b):
    if not isinstance(a, torch.Tensor):
        return a is b
    return (
        type(a) is type(b)
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.storage_offset() == b.storage_offset()
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    ("shapes", "ignored"),
    [
        ([(5, 3), (1, 3), (3,)], [-1]),
        ([(2, 4, 3, 5), (4, 3, 1), (1, 1, 1, 2), (3, 7)], [-1]),
        ([(1, 6, 3), (8, 6, 3), (8, 1, 1)], [-1]),
        ([(2, 3), (2, 3)], []),
        ([(4, 1, 2), (3, 2)], [0, -1, 9, -9]),
    ],
)
def test_broadcast_all_returns_the_reference_views(shapes, ignored):
    xs = [torch.randn(s) for s in shapes]
    xs.insert(1, None)
    xs.append(2.5)
    xs.append(torch.randn(4, 3).t())  # a non-contiguous input
    try:
        expected = _broadcast_all_reference(xs, ignored)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            broadcast_all(xs, ignored)
        return
    actual = broadcast_all(xs, ignored)
    assert len(actual) == len(expected)
    for a, e, x in zip(actual, expected, xs):
        assert _same_view(a, e) if isinstance(e, torch.Tensor) else a is x


def _member(T, n, *, params, normals_rank=3, colors_frames=None):
    member = SimpleNamespace(
        corners=torch.randn(T, n, 3),
        colors=torch.randn(colors_frames or T, n, 5),
        normals=torch.randn(*((T,) if normals_rank == 3 else ()), n, 3),
    )
    for name, value in params.items():
        setattr(member, name, value)
    return member


def test_member_surface_params_are_the_reference_views():
    names = ("reflectivity", "roughness", "refractive_index", "one_sided", "extra")
    members = [
        _member(6, 9, params={"roughness": torch.rand(6, 9, 1)}),
        _member(6, 9, params={"roughness": torch.rand(1, 9, 1), "one_sided": None}),
        _member(1, 12, params={"reflectivity": torch.rand(9).view(1, 9, 1)[:, :1]}),
        _member(4, 3, params={"extra": torch.rand(3, 1)}, normals_rank=2),
        _member(4, 3, params={"extra": torch.rand(4, 3, 1)}, colors_frames=1),
        # A param of higher rank than the references takes the reference path.
        _member(2, 3, params={"extra": torch.rand(2, 2, 3, 1)}),
    ]
    gathered = _gather_member_surface_params(members, names)
    for name in names:
        for member, view in zip(members, gathered[name]):
            reference = _member_surface_param_reference(member, name)
            if getattr(member, name, None) is None:
                # A fresh default fill each time: same values and layout.
                assert torch.equal(view, reference)
                assert view.stride() == reference.stride()
            else:
                assert _same_view(view, reference), (name, view.shape)


@pytest.mark.parametrize("shape", [(1, 10, 4, 3), (3, 40, 4, 3), (12, 300, 4, 3)])
def test_circuit_scale_is_bit_identical_through_the_fused_box(monkeypatch, shape):
    g = torch.Generator().manual_seed(1)
    corners = torch.randn(shape, generator=g) * 3
    corners[..., 0, 0, :] = 0.0
    corners[..., 1, 1, 0] = -0.0
    segments = shape[1]
    circuit_of_segment = torch.repeat_interleave(
        torch.arange(4),
        torch.tensor([segments // 4] * 3 + [segments - 3 * (segments // 4)]),
    )
    lo, hi = _control_point_box(corners)
    assert torch.equal(lo, corners.amin(-2))
    assert torch.equal(hi, corners.amax(-2))
    fused = [_circuit_scale(corners), _circuit_scale(corners, circuit_of_segment)]
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({"circuitscale"}))
    separate = [_circuit_scale(corners), _circuit_scale(corners, circuit_of_segment)]
    for a, b in zip(fused, separate):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))


def _author():
    square = Square().spawn(animate=False)
    circle = Circle().move(RIGHT * 2).spawn(animate=False)
    with Seq():
        square.pulse_color(RED)
        with Sync():
            circle.pulse_color(BLUE, opacity=0.5)
            square.color = BLUE
        square.move(UP)
        circle.color = Color(RED, opacity=0.7)


def _materialized(times):
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


def test_replayed_bodies_are_bit_identical(monkeypatch):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({"replaybodies"}))
    _author()
    times = torch.arange(0, 130) / 30.0
    reference = _materialized(times)
    SceneManager.reset()
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset())
    _author()
    fast = _materialized(times)
    assert reference.keys() == fast.keys()
    for name in reference:
        a, b = reference[name], fast[name]
        assert a.shape == b.shape, name
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), name
