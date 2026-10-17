"""What ``become`` shows when the two Mobs are built differently.

Before a morph can interpolate anything it makes the two sides alike: surfaces
are resampled to one resolution, the side with fewer members, children or
parts is padded with collapsed copies, ``minimize_movement`` re-pairs a pack's
members by position, and a pair from different primitive families is turned
into two triangle soups. Each of those steps decides what is on screen at the
first and last frame of the morph, so every expectation here is read off the
materialized timeline -- vertex positions, colours and opacity at the start,
middle and end -- and measured against the source as it stood before the morph
and the target as it stands alone, never against the alignment's own
bookkeeping.

A feature test, so it sits outside the fast suite: it only breaks when
``become``'s structural alignment itself changes.
"""

from __future__ import annotations

import pytest
import torch

import algan.manim as mn
from algan import (
    BLUE,
    GREEN,
    LEFT,
    PURPLE,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    YELLOW,
    ApplyPointwiseFunction,
    Circle,
    Dot3D,
    Group,
    Off,
    Scene,
    Sphere,
    Square,
    Surface,
    Sync,
    Text,
    TriangleMesh,
    easings,
)
from algan.mobs.pn_mesh import PNMesh

ATOL = 1e-4


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _now(scene) -> float:
    return float(scene.animation_manager.context.timespan.current_time)


def _morph(scene, source, target, **kwargs):
    """Record ``source.become(target)`` over one linear second.

    Returns ``(result, start, end)``: linear, so the middle of the morph is
    exactly halfway between its two ends.
    """
    start = _now(scene)
    with Sync(runtime=1.0, easing=easings.identity):
        result = source.become(target, **kwargs)
    return result, start, _now(scene)


def _read(scene, times, read):
    """``read()`` with the Scene materialized at ``times``, buffers freed after."""
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    try:
        return read()
    finally:
        scene.timeline_manager.clear_buffers()


def _rgb(*colors) -> torch.Tensor:
    return torch.stack([torch.as_tensor(color).reshape(-1)[:3] for color in colors])


def _span(points: torch.Tensor) -> float:
    """The diagonal of ``points``' bounding box: 0 for a collapsed part."""
    return float((points.amax(0) - points.amin(0)).norm())


def _box_centre(points: torch.Tensor) -> torch.Tensor:
    return (points.amin(0) + points.amax(0)) / 2


# ---------------------------------------------------------------------------
# Surfaces of different resolution
# ---------------------------------------------------------------------------


def _plane(width, height, *, z=0.0, color=BLUE):
    """The unit square ``[-0.5, 0.5]^2`` at height ``z``, as a parametric surface."""
    return Surface(
        lambda uv: torch.cat((uv - 0.5, torch.full_like(uv[..., :1], z)), -1),
        grid_width=width,
        grid_height=height,
        color=color,
    )


def _bump(points):
    """Raise the unit square 0.4 at its centre, and not at all along its rim."""
    x, y = points[..., :1], points[..., 1:2]
    lift = 0.4 * (1 - 4 * x * x) * (1 - 4 * y * y)
    return torch.cat((x, y, points[..., 2:] + lift), -1)


def _grid(surface, index):
    return surface.grid.location[index].reshape(-1, 3).clone()


def test_a_surface_morphs_into_a_finer_one_through_the_halfway_shape(scene):
    """A 5x5 plane becomes a 9x9 parabolic sheet, point for point.

    The coarser side is resampled to the finer grid through its own function,
    so the first frame is still exactly the plane, the last exactly the sheet,
    and every point travels straight up at a steady rate in between, while the
    colour cross-fades and the surface stays opaque throughout.
    """
    with Off():
        source = _plane(5, 5).spawn()
        target = Surface(
            lambda uv: torch.cat((uv - 0.5, 0.5 * (uv[..., :1] - 0.5) ** 2), -1),
            grid_width=9,
            grid_height=9,
            color=RED,
        )
    result, start, end = _morph(scene, source, target)

    def read():
        return (
            result.grid.location.reshape(3, -1, 3).clone(),
            result.grid.color.reshape(3, -1, 5)[..., :3].clone(),
            result.opacity.reshape(3, -1).clone(),
        )

    points, colors, opacity = _read(scene, [start, (start + end) / 2, end], read)

    assert result is source
    assert (result.grid_width, result.grid_height) == (9, 9)
    u = torch.linspace(-0.5, 0.5, 9)
    lattice = torch.cartesian_prod(u, u)
    for index, share in enumerate((0.0, 0.5, 1.0)):
        torch.testing.assert_close(points[index, :, :2], lattice, atol=ATOL, rtol=0)
        expected_z = share * 0.5 * lattice[:, 0] ** 2
        torch.testing.assert_close(points[index, :, 2], expected_z, atol=ATOL, rtol=0)
        expected_color = (1 - share) * _rgb(BLUE) + share * _rgb(RED)
        torch.testing.assert_close(
            colors[index], expected_color.expand(81, 3), atol=1e-3, rtol=0
        )
    assert float(opacity.min()) == pytest.approx(1.0)


def test_a_reshaped_surface_keeps_its_shape_into_a_finer_one(scene):
    """REGRESSION. A surface reshaped point by point went flat as its morph began.

    Reconciling two resolutions resamples the coarser side through its own
    parametric function, under the transform ``_current_surface_function``
    fits from that function to the grid. No transform holds a pointwise
    reshape, so the 5x5 plane ``ApplyPointwiseFunction`` had raised 0.4 at its
    centre was resampled as a flat plane at the bump's mean height, 0.1, and
    that plane -- not the bump -- was the first frame of its morph into a
    finer surface. The first frame has to be the surface as it stood: every
    one of its points still there, and the bump still 0.4 high.
    """
    with Off():
        source = _plane(5, 5).spawn()
        ApplyPointwiseFunction(source, _bump)
        reshaped = _grid(source, 0)
        target = _plane(9, 9, z=-0.3, color=RED)
    result, start, end = _morph(scene, source, target)

    first, middle = _read(
        scene,
        [start, (start + end) / 2],
        lambda: (_grid(result, 0), _grid(result, 1)),
    )

    assert float(reshaped[:, 2].max()) == pytest.approx(0.4, abs=ATOL)
    missing = torch.cdist(reshaped, first).amin(1)
    assert float(missing.max()) < ATOL, (
        f"the first frame of the morph lost the reshaped surface: one of its "
        f"points is {float(missing.max()):.3f} from anything drawn"
    )
    assert float(first[:, 2].max()) == pytest.approx(0.4, abs=ATOL)
    assert float(first[:, 2].min()) == pytest.approx(0.0, abs=ATOL)
    # Halfway, the peak is halfway between the bump and the flat target.
    assert float(middle[:, 2].max()) == pytest.approx((0.4 - 0.3) / 2, abs=ATOL)


def test_a_morph_ends_on_a_reshaped_coarser_target(scene):
    """REGRESSION. A coarser target reshaped point by point was never reached.

    The same resampling, on the target side: a 5x5 plane at ``z = -0.3``
    bumped 0.4 by ``ApplyPointwiseFunction`` was resampled to the source's 9x9
    grid as a flat plane at ``z = -0.2``, and the morph ended on that plane
    rather than on the bump. It has to end on every point of the target.
    """
    with Off():
        source = _plane(9, 9).spawn()
        target = _plane(5, 5, z=-0.3, color=RED)
        ApplyPointwiseFunction(target, _bump)
        reshaped = _grid(target, 0)
    result, _, end = _morph(scene, source, target)

    last = _read(scene, [end], lambda: _grid(result, 0))

    assert float(reshaped[:, 2].max()) == pytest.approx(0.1, abs=ATOL)
    missed = torch.cdist(reshaped, last).amin(1)
    assert float(missed.max()) < ATOL, (
        f"the morph ended {float(missed.max()):.3f} away from a point of its "
        f"target surface"
    )
    assert float(last[:, 2].max()) == pytest.approx(0.1, abs=ATOL)
    assert float(last[:, 2].min()) == pytest.approx(-0.3, abs=ATOL)


def test_packed_surfaces_of_different_resolution_keep_each_member_whole(scene):
    """Two coarse spheres of a pack become two finer, larger ones.

    A pack is resampled member by member, by interpolating each member's own
    grid, so on the first frame each sphere is still round its own centre and
    no larger than it was -- interpolation between a coarse grid's samples
    cuts inside the sphere, never outside it. Each moves halfway by the
    middle and ends exactly on its target.
    """
    centres = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    with Off():
        source = Sphere.from_batches(
            centres, radius=0.5, grid_width=6, grid_height=5
        ).spawn()
        target = Sphere.from_batches(
            centres + UP, radius=0.7, grid_width=10, grid_height=8
        )
        target_grid = _grid(target, 0)
    result, start, end = _morph(scene, source, target)

    grids = _read(
        scene,
        [start, (start + end) / 2, end],
        lambda: result.grid.location.reshape(3, 2, 80, 3).clone(),
    )

    assert (result.grid_width, result.grid_height) == (10, 8)
    # Measured from each member's own centre: a point that strayed into the
    # other member's grid would be three units out.
    radii = (grids[0] - centres.unsqueeze(1)).norm(dim=-1)
    assert float(radii.max()) == pytest.approx(0.5, abs=ATOL)
    torch.testing.assert_close(grids[1], (grids[0] + grids[2]) / 2, atol=ATOL, rtol=0)
    torch.testing.assert_close(grids[2].reshape(-1, 3), target_grid, atol=ATOL, rtol=0)


# ---------------------------------------------------------------------------
# Packs that gain or lose members
# ---------------------------------------------------------------------------


def _spheres(count, height):
    """``count`` spheres in a row at ``height``: a pack, or one lone Sphere."""
    if count == 1:
        return Sphere(radius=0.3).move(UP * height)
    xs = torch.linspace(-2, 2, count)
    centres = torch.stack((xs, torch.full_like(xs, height), torch.zeros_like(xs)), -1)
    return Sphere.from_batches(centres, radius=0.3)


def _members(surface, index=0):
    """``surface``'s grid at time ``index``, one ``[W * H, 3]`` block per member."""
    rows = surface.grid.location[index].reshape(-1, 3)
    return rows.reshape(-1, surface.grid_width * surface.grid_height, 3).clone()


def _assert_whole_members(blocks, expected, *, when):
    """The blocks that are not collapsed points are ``expected``, in order.

    Where the padding puts the collapsed members among the others is the
    alignment's business. What has to hold is that every member drawn is a
    whole grid of one of the Mobs, and that nothing else is drawn.
    """
    whole = [block for block in blocks if _span(block) > 1e-6]
    assert len(whole) == len(expected), (
        f"{when}, {len(whole)} of the {len(blocks)} member grids are drawn "
        f"where that frame should show {len(expected)}"
    )
    for position, (block, member) in enumerate(zip(whole, expected)):
        worst = float((block - member).norm(dim=-1).max())
        assert worst < ATOL, (
            f"{when}, drawn member {position} is not that Mob's member "
            f"{position}: a point is {worst:.3f} from where it belongs"
        )


@pytest.mark.parametrize(
    ("source_count", "target_count"),
    [(3, 5), (5, 3), (1, 3), (3, 1)],
    ids=["pack-grows", "pack-shrinks", "lone-to-pack", "pack-to-lone"],
)
def test_a_pack_of_surfaces_changes_its_member_count_a_whole_grid_at_a_time(
    scene, source_count, target_count
):
    """REGRESSION. A pack's grids were padded one ROW at a time.

    A surface's grid holds one block of ``grid_width * grid_height`` rows per
    member, and the generic padding treated each row as an object of its own,
    spreading the new rows through every member's block. Three spheres
    becoming five had blocks of 294, 293 and 293 rows where a sphere has 176:
    the morph drew scrambled grids, and rendering it raised. Five becoming
    three ended scrambled. A lone sphere becoming a pack started with each of
    its rows drawn three times over, and ended as one surface the renderer
    stitched across all three spheres; three becoming one ended with each
    member folded onto a third of the sphere.

    Padding a pack adds whole members collapsed to a point. So the first
    frame draws exactly the source's spheres and the last exactly the
    target's, and what the morph hands back indexes like a pack of the larger
    count.
    """
    with Off():
        source = _spheres(source_count, height=0.0).spawn()
        target = _spheres(target_count, height=1.0)
        source_members = list(_members(source))
        target_members = list(_members(target))
    result, start, end = _morph(scene, source, target)

    first, last = _read(
        scene, [start, end], lambda: (_members(result, 0), _members(result, 1))
    )

    count = max(source_count, target_count)
    assert len(first) == len(last) == count
    _assert_whole_members(first, source_members, when="on the first frame")
    _assert_whole_members(last, target_members, when="on the last frame")
    assert len(result) == count


def test_a_pack_that_gains_members_renders(scene, tmp_path):
    """REGRESSION. The renderer refused a pack whose grids had been padded by row.

    It draws a pack one ``grid_width * grid_height`` block at a time and
    raised ``ValueError: packed surface grid entries must each contain exactly
    grid_width * grid_height = 176 points, got [294, 293, 293]`` for three
    spheres becoming five. The membership is read only by the renderer, so a
    frame is rendered to see it.
    """
    with Off():
        source = _spheres(3, height=0.0).spawn()
        target = _spheres(5, height=1.0)
    _, start, end = _morph(scene, source, target)

    rendered = scene.save_frame(
        tmp_path / "middle.png", SMOKE_TEST, at=(start + end) / 2
    )

    assert rendered.status == "rendered"
    assert (tmp_path / "middle.png").is_file()


def _dots(mob):
    """The pack of dots a point cloud draws."""
    return next(
        node for node in [mob, *mob.get_descendants()] if isinstance(node, Dot3D)
    )


def _cloud(count, height, color):
    xs = torch.linspace(-2, 2, count)
    points = torch.stack((xs, torch.full_like(xs, height), torch.zeros_like(xs)), -1)
    return mn.PMobject(points=points, point_radius=0.1, color=color)


@pytest.mark.parametrize(
    ("source_count", "target_count"), [(5, 8), (8, 5)], ids=["gains", "loses"]
)
def test_a_point_cloud_gains_and_loses_whole_dots(scene, source_count, target_count):
    """REGRESSION. A point cloud changing its number of points scrambled its dots.

    A cloud's dots are one packed sphere grid per point, so it took the same
    row-by-row padding as any pack: five dots becoming eight drew five dots'
    rows smeared over eight, and eight becoming five ended on five dots'
    rows smeared over eight. The first frame has to be the source's dots and
    the last the target's, each one a whole dot.
    """
    with Off():
        source = _cloud(source_count, 0.0, BLUE).spawn()
        target = _cloud(target_count, 1.0, RED)
        source_dots = list(_members(_dots(source)))
        target_dots = list(_members(_dots(target)))
    result, start, end = _morph(scene, source, target)
    dots = _dots(result)

    first, last = _read(
        scene, [start, end], lambda: (_members(dots, 0), _members(dots, 1))
    )

    _assert_whole_members(first, source_dots, when="on the first frame")
    _assert_whole_members(last, target_dots, when="on the last frame")


def test_minimize_movement_keeps_a_packs_members_where_they_are(scene):
    """REGRESSION. ``minimize_movement`` re-paired a pack's rows but not its grids.

    The target lists the same three places in the opposite order. Re-pairing
    by position moved the target's per-member rows -- each member's anchor,
    the point it turns and scales about -- but left its sphere grids in list
    order, so the outer spheres still crossed the screen to swap places, and
    each ended with its anchor on the other's sphere: scaling a member after
    the morph flung it four units away. Nothing has to move here, each place
    has to end in the colour the target gives it, and a member scaled after
    the morph has to grow where it is.
    """
    places = torch.tensor([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with Off():
        source = Sphere.from_batches(
            places, radius=0.3, colors=_rgb(RED, GREEN, BLUE)
        ).spawn()
        target = Sphere.from_batches(
            places.flip(0), radius=0.3, colors=_rgb(YELLOW, WHITE, PURPLE)
        )
    result, start, end = _morph(scene, source, target, minimize_movement=True)
    with Sync(runtime=1.0, easing=easings.identity):
        result[0].scale(2)
    scaled = _now(scene)

    def read():
        grids = result.grid.location.reshape(4, 3, -1, 3)
        return (
            ((grids.amin(2) + grids.amax(2)) / 2).clone(),
            (grids.amax(2) - grids.amin(2))[..., 0].clone(),
            result.grid.color.reshape(4, 3, -1, 5)[:, :, 0, :3].clone(),
        )

    centres, widths, colors = _read(
        scene, [start, (start + end) / 2, end, scaled], read
    )

    for index, when in enumerate(("first", "middle", "last")):
        travelled = float((centres[index] - places).norm(dim=-1).max())
        assert travelled < 1e-2, (
            f"on the {when} frame a sphere is {travelled:.2f} from where it "
            f"started, though the target has one at every place"
        )
    torch.testing.assert_close(
        colors[2], _rgb(PURPLE, WHITE, YELLOW), atol=1e-3, rtol=0
    )
    moved = float((centres[3, 0] - places[0]).norm())
    assert moved < 0.05, (
        f"scaling the first member after the morph put it {moved:.2f} from its "
        f"place: its anchor is not on its own sphere"
    )
    assert float(widths[3, 0]) == pytest.approx(2 * float(widths[2, 0]), rel=1e-3)


# ---------------------------------------------------------------------------
# Children that only one side has
# ---------------------------------------------------------------------------


def _subtree_points(mob, index):
    points = [
        node.location[index].reshape(-1, 3)
        for node in [mob, *mob.get_descendants()]
        if getattr(node, "location", None) is not None and node.location.numel()
    ]
    return torch.cat(points, 0)


def _brightest(mob, index):
    return max(
        float(node.opacity[index].max())
        for node in [mob, *mob.get_descendants()]
        if getattr(node, "opacity", None) is not None and node.opacity.numel()
    )


def _square_with(*children):
    square = Square(color=BLUE)
    square.add_children(*children)
    return square


def _child_states(scene, child, times):
    """The child's points and brightest opacity at each of ``times``."""

    def read():
        return [
            (_subtree_points(child, index).clone(), _brightest(child, index))
            for index in range(len(times))
        ]

    return _read(scene, times, read)


def test_a_primitive_gaining_a_child_grows_it_and_fades_it_in(scene):
    """A square with one child becomes a square with two.

    The child both sides have morphs as usual, colour and all. The new one
    starts collapsed to a point and invisible where its siblings leave room
    for it -- its own centre -- and grows at a steady rate while it fades in,
    ending as exactly the target's child.
    """
    with Off():
        source = _square_with(Circle(radius=0.2, color=RED).move(UP * 1.5)).spawn()
        target = _square_with(
            Circle(radius=0.2, color=GREEN).move(UP * 1.5),
            Circle(radius=0.3, color=YELLOW).move(RIGHT * 2),
        )
        target_child = _subtree_points(target.get_non_component_children()[1], 0)
    result, start, end = _morph(scene, source, target)
    kept, grown = result.get_non_component_children()

    times = [start, (start + end) / 2, end]
    states = _child_states(scene, grown, times)
    kept_colors = _read(
        scene, [start, end], lambda: kept.grid.color.reshape(2, -1, 5)[:, 0, :3].clone()
    )

    assert grown.is_spawned()
    assert grown in scene.actors
    (first, first_alpha), (middle, middle_alpha), (last, last_alpha) = states
    assert _span(first) < 1e-6
    torch.testing.assert_close(
        first[0], torch.tensor([2.0, 0.0, 0.0]), atol=ATOL, rtol=0
    )
    assert first_alpha == pytest.approx(0.0, abs=1e-3)
    assert _span(middle) == pytest.approx(_span(target_child) / 2, rel=1e-3)
    assert middle_alpha == pytest.approx(0.5, abs=1e-3)
    torch.testing.assert_close(last, target_child, atol=ATOL, rtol=0)
    assert last_alpha == pytest.approx(1.0, abs=1e-3)
    torch.testing.assert_close(kept_colors, _rgb(RED, GREEN), atol=1e-3, rtol=0)


def test_a_primitive_losing_a_child_shrinks_it_away_and_fades_it_out(scene):
    """A square with two children becomes a square with one.

    The surplus child shrinks to a point at its own centre while it fades
    out, and is gone -- collapsed and transparent -- on the last frame.
    """
    with Off():
        surplus = Circle(radius=0.3, color=YELLOW).move(RIGHT * 2)
        source = _square_with(
            Circle(radius=0.2, color=GREEN).move(UP * 1.5), surplus
        ).spawn()
        surplus_points = _subtree_points(surplus, 0)
        target = _square_with(Circle(radius=0.2, color=RED).move(UP * 1.5))
    result, start, end = _morph(scene, source, target)
    shrinking = result.get_non_component_children()[1]

    states = _child_states(scene, shrinking, [start, (start + end) / 2, end])

    (first, first_alpha), (middle, middle_alpha), (last, last_alpha) = states
    torch.testing.assert_close(first, surplus_points, atol=ATOL, rtol=0)
    assert first_alpha == pytest.approx(1.0, abs=1e-3)
    assert _span(middle) == pytest.approx(_span(surplus_points) / 2, rel=1e-3)
    assert middle_alpha == pytest.approx(0.5, abs=1e-3)
    assert _span(last) < 1e-6
    torch.testing.assert_close(
        last[0], torch.tensor([2.0, 0.0, 0.0]), atol=ATOL, rtol=0
    )
    assert last_alpha == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("minimize_movement", [False, True])
def test_minimize_movement_pairs_a_primitives_children_by_position(
    scene, minimize_movement
):
    """The target lists the same two children in the opposite order.

    Paired in list order, the circle on the left becomes the square on the
    right and the two cross; paired by position, each child stays where it
    is.
    """
    with Off():
        source = _square_with(
            Circle(radius=0.2, color=RED).move(LEFT * 2),
            Square(size=0.4, color=GREEN).move(RIGHT * 2),
        ).spawn()
        target = _square_with(
            Square(size=0.4, color=GREEN).move(RIGHT * 2),
            Circle(radius=0.2, color=RED).move(LEFT * 2),
        )
    result, start, end = _morph(
        scene, source, target, minimize_movement=minimize_movement
    )
    children = result.get_non_component_children()

    def read():
        return [
            [float(_subtree_points(child, index)[:, 0].mean()) for index in range(3)]
            for child in children
        ]

    paths = _read(scene, [start, (start + end) / 2, end], read)

    expected_ends = (-2.0, 2.0) if minimize_movement else (2.0, -2.0)
    for path, begin, finish in zip(paths, (-2.0, 2.0), expected_ends):
        assert path[0] == pytest.approx(begin, abs=1e-3)
        assert path[2] == pytest.approx(finish, abs=1e-3)


# ---------------------------------------------------------------------------
# Different primitive families meet as triangle soups
# ---------------------------------------------------------------------------


def _soup(scene):
    return next(actor for actor in scene.actors if isinstance(actor, PNMesh))


def _soup_state(soup):
    return (
        soup.location[0].reshape(-1, 3).clone(),
        soup.color[0].reshape(-1, 5)[:, :3].clone(),
        soup.opacity[0].reshape(-1).clone(),
    )


def _gradient(uv):
    """Red at ``u = 0``, green at ``u = 1``."""
    u = uv[..., :1]
    return torch.cat((1 - u, u, torch.zeros_like(u)), -1)


def _painted_square():
    square = Square(size=3, grid_width=8, grid_height=8)
    square.set_color_by_function(_gradient)
    return square


def _assert_painted_ramp(corners, colors, *, when):
    """``colors`` are the painted red-to-green ramp, running left to right.

    What the square itself draws there: the renderer blends its texels
    across it, so every point is a colour of the ramp, redder towards its
    left edge and greener towards its right.
    """
    off_ramp = float(
        torch.maximum((colors[:, 0] + colors[:, 1] - 1).abs(), colors[:, 2].abs()).max()
    )
    assert off_ramp < 1e-3, (
        f"{when}, the soup is not the painted red-to-green ramp: a corner is "
        f"{off_ramp:.3f} off it"
    )
    assert float(colors[:, 1].max() - colors[:, 1].min()) > 0.5, (
        f"{when}, the soup is one flat colour where the square is a gradient"
    )
    left = corners[:, 0] < corners[:, 0].min() + 1e-3
    right = corners[:, 0] > corners[:, 0].max() - 1e-3
    assert bool((colors[left, 0] > colors[left, 1]).all())
    assert bool((colors[right, 1] > colors[right, 0]).all())
    follows_x = torch.corrcoef(torch.stack((corners[:, 0], colors[:, 1])))[0, 1]
    assert float(follows_x) > 0.99


def test_a_painted_shape_becomes_a_solid_from_its_painted_colours(scene):
    """REGRESSION. A circuit painted per texel turned one flat colour mid-morph.

    A morph from a bezier shape into a sphere hands over to a triangle soup
    of the shape's fill. A fill painted by ``set_color_by_function`` is drawn
    from its texels, but the soup took the shape's own ``color`` -- only the
    colour it was built with -- so a red-to-green square turned flat salmon on
    the frame the soup took over. The soup's first frame has to carry the
    gradient the square was showing.
    """
    with Off():
        source = _painted_square().spawn()
        target = Sphere(radius=1.5)
    _morph(scene, source, target)
    soup = _soup(scene)

    corners, colors, opacity = _read(
        scene, [float(soup.lifespan.start())], lambda: _soup_state(soup)
    )

    assert float(corners[:, 2].abs().max()) < ATOL
    assert float(opacity.min()) == pytest.approx(1.0)
    _assert_painted_ramp(corners, colors, when="on the soup's first frame")


def test_a_solid_becomes_a_painted_shape_through_its_painted_colours(scene):
    """REGRESSION. A solid becoming a painted shape ran flat until the swap.

    The reverse: the soup morphs the sphere into the painted square's fill,
    then the square itself takes over. The soup was heading for the square's
    own ``color``, so it ended one flat colour and the gradient popped in when
    the square replaced it. The soup's last frame has to be the gradient the
    square is about to show.
    """
    with Off():
        source = Sphere(radius=1.5, color=BLUE).spawn()
        target = _painted_square()
    result, _, _ = _morph(scene, source, target)
    soup = _soup(scene)

    corners, colors, opacity = _read(
        scene, [float(soup.lifespan.end()) - 1e-4], lambda: _soup_state(soup)
    )

    assert isinstance(result, Square)
    assert float(corners[:, 2].abs().max()) < 1e-3
    assert float(opacity.min()) == pytest.approx(1.0)
    _assert_painted_ramp(corners, colors, when="on the soup's last frame")


def test_text_becomes_a_solid_starting_from_each_letters_fill_and_colour(scene):
    """A two-letter word in two colours becomes a sphere.

    The soup starts as the word's fill, flat where the word is and as wide as
    it, with each letter's triangles in that letter's colour.
    """
    with Off():
        source = Text("ab", font_size=60)
        source[0].color = RED
        source[1].color = GREEN
        source.spawn()
        width = float(source.get_width().reshape(-1)[0])
        target = Sphere(radius=0.5, color=BLUE)
    _morph(scene, source, target)
    soup = _soup(scene)

    corners, colors, opacity = _read(
        scene, [float(soup.lifespan.start())], lambda: _soup_state(soup)
    )

    red = (colors - _rgb(RED)).abs().amax(-1) < 1e-4
    green = (colors - _rgb(GREEN)).abs().amax(-1) < 1e-4
    assert bool((red | green).all()), "a soup corner is neither letter's colour"
    assert bool(red.any())
    assert bool(green.any())
    assert float(corners[red, 0].max()) < float(corners[green, 0].min()), (
        "the red letter's triangles are not all left of the green letter's"
    )
    assert float(corners[:, 2].abs().max()) < ATOL
    assert float(corners[:, 0].max() - corners[:, 0].min()) == pytest.approx(
        width, rel=0.02
    )
    assert float(opacity.min()) == pytest.approx(1.0)


def test_a_triangle_mesh_becomes_a_solid_from_its_own_triangles(scene):
    """Two triangles making a unit square become a sphere two units away.

    The soup starts as the mesh, in its colour, travels and cross-fades,
    arrives on the sphere in the sphere's colour, and hands over to the
    sphere itself on the last frame.
    """
    corners = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    centre = torch.tensor([2.0, 0.0, 0.0])
    with Off():
        source = TriangleMesh(corners, faces, color=RED).spawn()
        target = Sphere(radius=0.5, color=BLUE).move(RIGHT * 2)
        target_grid = _grid(target, 0)
    result, _, end = _morph(scene, source, target)
    soup = _soup(scene)
    start, finish = float(soup.lifespan.start()), float(soup.lifespan.end())
    times = [start, (start + finish) / 2, finish - 1e-4, finish]

    def read():
        points = soup.location.reshape(4, -1, 3).clone()
        colors = soup.color.reshape(4, -1, 5)[..., :3].clone()
        alphas = soup.opacity.reshape(4, -1).amax(-1).clone()
        return points, colors, alphas, _grid(result, 3), float(result.opacity[3].min())

    points, colors, alphas, final_grid, final_alpha = _read(scene, times, read)

    def solid(color):
        return _rgb(color).expand_as(colors[0])

    assert finish == pytest.approx(end)
    # Starts as the mesh: flat on the unit square, all of it, in red.
    torch.testing.assert_close(points[0].amin(0), corners.amin(0), atol=ATOL, rtol=0)
    torch.testing.assert_close(points[0].amax(0), corners.amax(0), atol=ATOL, rtol=0)
    torch.testing.assert_close(colors[0], solid(RED), atol=1e-3, rtol=0)
    # Halfway in colour, opaque throughout.
    halfway = (_rgb(RED) + _rgb(BLUE)) / 2
    torch.testing.assert_close(
        colors[1], halfway.expand_as(colors[1]), atol=1e-3, rtol=0
    )
    assert alphas[:3].tolist() == pytest.approx([1.0, 1.0, 1.0])
    # Arrives on the sphere, in blue.
    radii = (points[2] - centre).norm(dim=-1)
    assert float((radii - 0.5).abs().max()) < 1e-3
    torch.testing.assert_close(colors[2], solid(BLUE), atol=1e-3, rtol=0)
    # Then the sphere itself, exactly the target, takes over from the soup.
    assert float(alphas[3]) == pytest.approx(0.0)
    torch.testing.assert_close(final_grid, target_grid, atol=ATOL, rtol=0)
    assert final_alpha == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Words of different lengths, and a piecewise dissolve
# ---------------------------------------------------------------------------


def test_a_word_losing_letters_ends_exactly_as_wide_as_the_shorter_word(scene):
    """``abcde`` becomes ``ab``.

    The three surplus letters have to shrink away inside the word that is
    left, not stay drawn where they were: on the last frame nothing of the
    morph lies outside the target's width.
    """
    with Off():
        source = Text("abcde", font_size=40).spawn()
        target = Text("ab", font_size=40)
        target_left = float(target.get_bounding_box_min().reshape(-1, 3)[0, 0])
        target_width = float(target.get_width().reshape(-1)[0])
    result, _, end = _morph(scene, source, target)

    points = _read(scene, [end], lambda: _subtree_points(result, 0).clone())

    assert float(points[:, 0].min()) == pytest.approx(target_left, abs=1e-3)
    assert float(points[:, 0].max() - points[:, 0].min()) == pytest.approx(
        target_width, abs=1e-3
    )


def test_a_dissolve_crosses_between_groups_of_different_sizes(scene):
    """Two squares dissolve into three circles.

    ``strategy="dissolve"`` fades the source out as the target fades in, part
    by part: on the first frame only the squares are drawn, where they are;
    halfway both are at half strength; on the last frame only the circles,
    each where the target puts it.
    """
    places = torch.stack([UP + LEFT * 3, UP, UP + RIGHT * 3])
    with Off():
        squares = [
            Square(size=0.6, color=BLUE).move(LEFT * 2),
            Square(size=0.6, color=BLUE).move(RIGHT * 2),
        ]
        source = Group(*squares).spawn()
        target = Group(
            Circle(radius=0.4, color=RED).move(places[0]),
            Circle(radius=0.4, color=GREEN).move(places[1]),
            Circle(radius=0.4, color=YELLOW).move(places[2]),
        )
    result, start, end = _morph(scene, source, target, strategy="dissolve")
    circles = list(result.children)

    def read():
        parts = squares + circles
        alphas = [[_brightest(part, index) for index in range(3)] for part in parts]
        starts = torch.stack([_box_centre(_subtree_points(s, 0)) for s in squares])
        ends = torch.stack([_box_centre(_subtree_points(c, 2)) for c in circles])
        return alphas, starts, ends

    alphas, starts, ends = _read(scene, [start, (start + end) / 2, end], read)

    assert len(circles) == 3
    for alpha in alphas[:2]:
        assert alpha == pytest.approx([1.0, 0.5, 0.0], abs=1e-3)
    for alpha in alphas[2:]:
        assert alpha == pytest.approx([0.0, 0.5, 1.0], abs=1e-3)
    torch.testing.assert_close(
        starts, torch.stack([LEFT * 2, RIGHT * 2]).to(starts), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(ends, places.to(ends), atol=1e-3, rtol=0)
