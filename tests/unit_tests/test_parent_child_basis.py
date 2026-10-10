from __future__ import annotations

import math

import pytest
import torch

import algan
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_bases,
)
from algan.scene_manager import SceneManager

# In the fast suite: attribute propagation from a parent to its children is the
# rule the whole mob hierarchy is built on, and every composite Mob relies on it.
pytestmark = pytest.mark.fast


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture
def scene():
    SceneManager.reset()
    current = algan.Scene(scene_initializer=_empty_scene)
    yield current
    current._terminate()
    SceneManager.reset()


def _materialize(scene, *times):
    with algan.Off(
        record_attr_modifications=False,
        record_funcs=False,
        priority_level=math.inf,
    ):
        scene.timeline_manager.set_state_to_times(
            torch.tensor(times, dtype=torch.get_default_dtype())
        )


def test_synchronized_parent_child_rotations_preserve_descendant_bases(scene):
    group = algan.Group([algan.Square()]).arrange_in_grid().spawn(animate=False)
    square = group[0]
    descendants = square.get_descendants(include_self=True)
    initial_bases = {mob.id: mob.basis.clone().reshape(-1, 3, 3) for mob in descendants}

    with algan.Sync(runtime=1, easing=algan.easings.identity):
        group.rotate(180, algan.UP)
        square.rotate(180, algan.RIGHT)

    _materialize(scene, 0.25, 0.5, 0.75)

    for mob in descendants:
        actual = mob.basis.reshape(3, -1, 3, 3)
        expected = torch.stack(
            [
                initial_bases[mob.id]
                @ get_rotation_around_axis(180 * time, algan.UP, dim=-1)
                @ get_rotation_around_axis(180 * time, algan.RIGHT, dim=-1)
                for time in (0.25, 0.5, 0.75)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=0)


def _sheared_basis():
    """A basis whose rows are neither unit length nor mutually orthogonal."""
    return torch.tensor(
        [
            [0.02, 0.0, 0.0],
            [0.3, 1.1, 0.0],
            [0.0, 0.4, 0.05],
        ]
    )


def _flat(basis):
    """A 3x3 basis in the flat 9-channel layout ``Mob.basis`` stores."""
    return basis.reshape(9)


def test_rotation_between_bases_maps_a_sheared_basis_onto_the_target():
    source = _sheared_basis()
    target = torch.tensor(
        [
            [0.0, 0.7, 0.1],
            [-1.2, 0.2, 0.0],
            [0.05, 0.0, 0.3],
        ]
    )

    change = get_rotation_between_bases(source, target)

    # 5e-6 matches the tolerance used above: the solve behind
    # get_rotation_between_bases rounds differently across BLAS backends, and at
    # 1e-6 this shear overshot by 1.01e-6 on Linux/x86 while passing on Windows.
    torch.testing.assert_close(source @ change, target, atol=5e-6, rtol=0)


def test_assigning_a_basis_to_itself_does_not_drift(scene):
    # Mob.basis's setter records an absolute basis as a change relative to the
    # current one, so an inexact change is re-applied to the value it was
    # measured from. That used to amplify the float-noise shear of an
    # orthogonal basis roughly threefold per assignment -- and every
    # detach_history clone performs one, so wave_color's resolution refinement
    # collapsed a Cylinder's basis after a couple of dozen waves.
    square = algan.Square().spawn(animate=False)
    square.basis = _flat(_sheared_basis())
    original = square.basis.clone()

    for _ in range(40):
        square.basis = square.basis

    torch.testing.assert_close(square.basis, original, atol=1e-6, rtol=0)


def test_repeated_history_detachment_preserves_a_sheared_basis(scene):
    square = algan.Square().spawn(animate=False)
    square.basis = _flat(_sheared_basis())
    original = square.basis.clone()

    for _ in range(20):
        square.detach_history()

    torch.testing.assert_close(square.basis, original, atol=1e-6, rtol=0)


@pytest.fixture
def camera_scene():
    SceneManager.reset()
    current = algan.Scene()
    yield current
    current._terminate()
    SceneManager.reset()


def _camera_rider(scene):
    """A bar one unit in front of a camera 80 units out, riding it, turned 35 degrees.

    The setup the backprop video's probability bars use: brought forward along
    the camera's rays (so scaled by about 1/128) and parented to the camera,
    which then turns -- carrying the bar to world coordinates near 100, where
    float32 rounds positions to about 1e-5.
    """
    camera = scene.camera
    with algan.Off():
        camera.set_near_orthographic(distance=80)
    bar = algan.Rectangle(width=4, height=0.4, stroke_width=0, fill_opacity=0.9)
    eye, forward = camera.location.reshape(3), camera.forward.reshape(3)
    k = 1.0 / float(((bar.get_center().reshape(3) - eye) * forward).sum())
    bar.scale(k)
    bar.move_to(eye + (bar.location.reshape(3) - eye) * k)
    with algan.Off():
        bar.spawn(False)
        camera.add_children(bar)
    with algan.Sync(runtime=1):
        camera.rotate(35, algan.UP, about=algan.ORIGIN)
    return bar


def _shape_in_frame(mob):
    """Every location row of ``mob``'s subtree in its own frame, in float64.

    Scaling a Mob along its own axes leaves these unchanged, so they are what a
    drain and refill must give back.
    """
    from algan.geometry.geometry import map_global_to_local_coords

    rows = mob.get_animated_attribute("location", include_descendants=True)
    return map_global_to_local_coords(
        mob.location.reshape(1, 3).double(),
        mob.basis.reshape(1, 9).double(),
        rows.reshape(-1, 3).double(),
    )


@pytest.mark.parametrize("drain", [1e-4, 1e-7])
def test_a_camera_rider_drained_to_a_sliver_refills_to_its_shape(camera_scene, drain):
    # Drained, the bar is narrower than float32 can resolve at its world
    # coordinates, so its points collapse onto one line. Refilling used to
    # scale that line -- the bar stayed invisible -- and the basis came back
    # skewed by the float32 round trip through the drained basis's inverse.
    bar = _camera_rider(camera_scene)
    shape = _shape_in_frame(bar)
    full = bar.scale_coefficient.reshape(-1)[:3].clone()
    target = full * torch.tensor([0.8, 1.0, 1.0])
    with algan.Sync(runtime=0.5, easing=algan.easings.identity):
        bar.scale_coefficient = full * torch.tensor([drain, 1.0, 1.0])
    with algan.Sync(runtime=0.5, easing=algan.easings.identity):
        bar.scale_coefficient = target
    algan.Scene.wait(0.5)

    # 2e-3 of the bar's own extent: float32 places the full-size bar's points
    # to about 3e-4 of it out here, and a collapsed refill misses by all of it.
    # The scale is held to 3e-4: a float32 change may miss by up to 1e-4
    # before the setter switches to float64 (_BASIS_CHANGE_TOLERANCE), and
    # without that switch the 1e-7 drain misses by about a tenth.
    torch.testing.assert_close(_shape_in_frame(bar), shape, atol=2e-3, rtol=0)
    torch.testing.assert_close(
        bar.scale_coefficient.reshape(-1)[:3], target, rtol=3e-4, atol=0
    )
    # Mid-refill and after it, as the renderer would materialize them.
    for time, fill in ((1.75, 0.5 * (drain + 0.8)), (2.25, 0.8)):
        _materialize(camera_scene, time)
        torch.testing.assert_close(_shape_in_frame(bar), shape, atol=2e-3, rtol=0)
        torch.testing.assert_close(
            bar.scale_coefficient.reshape(-1)[:3],
            full * torch.tensor([fill, 1.0, 1.0]),
            rtol=3e-4,
            atol=0,
        )


def test_a_camera_rider_refilled_inside_an_animated_function_keeps_its_shape(
    camera_scene,
):
    # Replay re-runs a recorded function's body, so the write inside it is
    # decided again against each frame's (collapsed) state rather than replayed
    # from a record of its own.
    bar = _camera_rider(camera_scene)
    shape = _shape_in_frame(bar)
    full = bar.scale_coefficient.reshape(-1)[:3].clone()
    drained = full * torch.tensor([1e-4, 1.0, 1.0])
    target = full * torch.tensor([0.8, 1.0, 1.0])

    @algan.animated_function(animated_args={"t": 0.0})
    def refill(mob, t=1.0):
        mob.scale_coefficient = drained + (target - drained) * t

    with algan.Sync(runtime=0.5):
        bar.scale_coefficient = drained
    with algan.Sync(runtime=0.5, easing=algan.easings.identity):
        refill(bar)

    _materialize(camera_scene, 1.75)
    torch.testing.assert_close(_shape_in_frame(bar), shape, atol=2e-3, rtol=0)


def test_an_ordinary_mob_remembers_no_shape(scene):
    # The remembered shape is for Mobs whose rows have lost theirs. A Mob of
    # ordinary size near the origin must not carry one -- or the float64 basis
    # change -- since both would change what it computes.
    square = algan.Square().move(algan.RIGHT * 3).spawn(animate=False)
    line = algan.Line(algan.LEFT * 5, algan.RIGHT * 5).spawn(animate=False)
    for mob in (square, line):
        mob.rotate(30, algan.OUT)
        mob.scale(torch.tensor([0.01, 1.0, 1.0]))
        mob.scale(torch.tensor([100.0, 1.0, 1.0]))
        assert "_rest_shape" not in mob.__dict__
    changes = [
        event.kwargs["change"]
        for event in scene.timeline_manager.function_timeline.function_applications
        if "change" in event.kwargs and "rest_local" in event.kwargs
    ]
    assert changes
    assert all(change.dtype == torch.float32 for change in changes)
