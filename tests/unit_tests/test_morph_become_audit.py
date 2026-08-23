"""Defects found auditing ``Mob.become`` against what the target alone renders.

``test_morph_become.py`` covers the contracts ``become`` was written to keep.
These are the ones it was not: each test here was a reproduction first and a
guard second, and each names the external standard it is measured against
rather than an internal invariant.  The standard throughout is *the target*:
when a morph ends, the Scene has to hold what spawning the target alone would
have held -- the same geometry, the same fill, and no Mob the target would not
have registered.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE,
    LEFT,
    RIGHT,
    UP,
    Cross,
    Cube,
    Dot3D,
    Group,
    Line,
    Off,
    Scene,
    Sphere,
    Square,
    Surface,
    Sync,
    VGroup,
)


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _rendering_actors(scene):
    """The actors the render loop will ask for geometry."""
    return [actor for actor in scene.actors if hasattr(actor, "get_render_primitives")]


def _visible_points(scene, index=0):
    """Every point of every visible row at one materialized time index."""
    points = []
    for node in scene.actors:
        location = getattr(node, "location", None)
        opacity = getattr(node, "opacity", None)
        if location is None or opacity is None or location.shape[0] <= index:
            continue
        rows = location[index].reshape(-1, 3)
        alpha = opacity[index].reshape(-1)
        if alpha.numel() == 1:
            alpha = alpha.expand(rows.shape[0])
        if rows.numel() == 0 or alpha.numel() != rows.shape[0]:
            continue
        keep = alpha > 1e-3
        if bool(keep.any()):
            points.append(rows[keep])
    return torch.cat(points, 0) if points else None


def _chamfer(a, b):
    distances = torch.cdist(a.float(), b.float())
    return max(float(distances.amin(1).amax()), float(distances.amin(0).amax()))


def _static_points(mob):
    points = [
        node.location.reshape(-1, 3)
        for node in [mob, *mob.get_descendants()]
        if getattr(node, "location", None) is not None and node.location.numel()
    ]
    return torch.cat(points, 0)


# ---------------------------------------------------------------------------
# A Mob whose geometry an ancestor draws is not a Mob of its own
# ---------------------------------------------------------------------------


def test_morphing_into_a_polyhedron_does_not_publish_its_vertex_dots(scene):
    """A Polyhedron draws its faces and nothing else.

    ``Polyhedron.get_render_primitives`` returns ``_face_primitive_mobs()``:
    the vertex ``Dot3D``s and the edge Mobs under ``self.graph`` are children
    of the Polyhedron but are never drawn, and constructing one registers only
    the Polyhedron itself as an actor.  ``become`` used to walk that graph,
    treat each dot as a morph unit of its own, and register the result for
    rendering -- so a morphed Cube grew eight vertex beads and a wireframe that
    a spawned Cube does not have.
    """
    with Off():
        sphere = Sphere(radius=0.8).spawn()
        cube = Cube(side_length=1.0)
    with Sync(run_time=1.0):
        sphere.become(cube)

    assert not [actor for actor in scene.actors if isinstance(actor, Dot3D)]


def test_a_morphed_polyhedron_draws_each_face_once(scene):
    """The faces must not be published alongside the Polyhedron that draws them.

    A face registered in its own right is drawn twice: once by the Polyhedron
    and once by itself. The two copies are coplanar, so the double draw shows
    up along every silhouette edge rather than as an obviously wrong picture.
    """
    with Off():
        reference = Cube(side_length=1.0).spawn()
    expected = len(_rendering_actors(scene))

    with Scene() as fresh:
        with Off():
            sphere = Sphere(radius=0.8).spawn()
            cube = Cube(side_length=1.0)
        with Sync(run_time=1.0):
            result = sphere.become(cube)
        live = [
            actor
            for actor in _rendering_actors(fresh)
            if actor.is_spawned() and not actor.is_despawned()
        ]
        assert len(live) == expected, (
            f"a morphed Cube publishes {len(live)} renderable actors where a "
            f"spawned one publishes {expected}"
        )
        assert result is not None


# ---------------------------------------------------------------------------
# A stroke-only circuit is not a triangulation failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: VGroup(Line(LEFT, RIGHT), Line(UP, UP * -1)), id="crossed_lines"
        ),
        pytest.param(lambda: Cross(), id="cross"),
    ],
)
def test_an_unfilled_circuit_can_cross_primitive_families(scene, build):
    """Crossing to a mesh converts the source's fill; an unfilled one has none.

    ``_bezier_to_pn_soup`` triangulates each sub-path's interior. A stroke-only
    compound path -- two crossed lines, a Cross, a DashedLine, an Axes, a
    MathTex -- encloses no area at all, the triangulator returns no tiles, and
    ``torch.cat`` of an empty list raised inside the constructor. The morph is
    supposed to be possible: the fill is simply empty.
    """
    with Off():
        source = build().spawn()
        target = Sphere(radius=0.6).move(RIGHT * 1.5)
    with Sync(run_time=1.0):
        result = source.become(target)
    assert result is not None


# ---------------------------------------------------------------------------
# Fill is part of what a shape looks like
# ---------------------------------------------------------------------------


def test_become_takes_the_targets_sidedness(scene):
    """``two_sided`` and ``closed_shell`` describe the geometry, so they move with it.

    A full ``Sphere`` is a closed one-sided shell; a swept-partial one is open
    and two-sided, and the renderer reads both to decide whether a back-facing
    hit is shaded as an inside and whether ``opacity`` attenuates once or twice.
    Neither is animatable, so a morph between the two ended with one Sphere's
    geometry wearing the other's declaration.
    """
    with Off():
        source = Sphere(radius=0.7, u_range=(0.0, 3.14159 / 2)).spawn()
        target = Sphere(radius=0.7)
    assert (source.two_sided, source.closed_shell) != (
        target.two_sided,
        target.closed_shell,
    ), "the fixture no longer contrasts the two declarations"

    with Sync(run_time=1.0):
        result = source.become(target)
    assert result.two_sided is target.two_sided
    assert result.closed_shell is target.closed_shell


@pytest.mark.parametrize("source_filled", [True, False])
def test_become_takes_the_targets_fill(scene, source_filled):
    """``filled`` decides whether a circuit is a disc or a ring.

    It is not an animatable attribute, so the same-kind path never carried it:
    a filled Square becoming an unfilled one kept its fill and rendered as a
    solid where the target is an outline -- a full-range difference over 3.6%
    of the frame, not a subtlety.
    """
    with Off():
        source = Square(color=BLUE, filled=source_filled, border_width=0.05).spawn()
        target = Square(color=BLUE, filled=not source_filled, border_width=0.05)
    with Sync(run_time=1.0):
        result = source.become(target)
    assert result.filled is (not source_filled)


# ---------------------------------------------------------------------------
# A morph occupies its context whether or not it has anything to do
# ---------------------------------------------------------------------------


def test_an_empty_morph_still_spends_its_run_time(scene):
    """Two empty Groups record nothing, so the context cursor never moved.

    Every other become spends exactly one ``run_time``. A no-op that spends
    zero silently pulls everything after it in a ``Seq`` a second early.
    """
    with Off():
        source = Group().spawn()
        target = Group()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(run_time=1.0):
        source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)
    assert end - start == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A morph ends on the target, not on an approximation of it
# ---------------------------------------------------------------------------


def _wave(grid_width, grid_height):
    return Surface(
        lambda u, v: torch.stack(
            (u - 0.5, v - 0.5, 0.25 * torch.sin(6 * u) * torch.cos(6 * v)), -1
        ),
        grid_width=grid_width,
        grid_height=grid_height,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, unfixed. _reconcile_grid_pair resamples both sides to the "
        "per-axis maximum grid, so a target that is coarser along an axis is "
        "resampled UPWARD -- and F.interpolate over a coarse grid does not "
        "reproduce the surface those samples came from. The morph therefore "
        "ends on an interpolated near-miss of the target: 0.161 on a surface "
        "one unit across, 178 channel values over 0.70% of an LD frame. The "
        "fix is to re-evaluate the surface's own function on the finer grid "
        "instead of interpolating its samples -- Surface keeps it as `_func` "
        "and exposes `_current_surface_function()` -- but the grid holds "
        "transformed coordinates, so doing that correctly means going through "
        "the base-grid cache rather than calling the function directly."
    ),
)
def test_morphing_into_a_coarser_surface_lands_on_that_surface(scene):
    """Grid reconciliation resamples both sides to the per-axis maximum."""
    with Off():
        source = Surface(
            lambda u, v: torch.stack((u - 0.5, v - 0.5, torch.zeros_like(u)), -1),
            grid_width=6,
            grid_height=6,
        ).spawn()
        target = _wave(4, 9)
    reference = _static_points(target)

    with Sync(run_time=1.0):
        source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)
    scene.timeline_manager.set_state_to_times(torch.tensor([end]))
    points = _visible_points(scene)
    error = _chamfer(points, reference)
    scene.timeline_manager.clear_buffers()

    assert error < 0.02, f"the morph ended {error:.4f} away from the target surface"
