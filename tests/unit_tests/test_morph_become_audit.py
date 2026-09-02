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
    Arrow3D,
    Circle,
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
    Tetrahedron,
    TriangleVertices,
)
from algan.animatable_base.mob_morph import MobMorphMixin
from algan.manim import Cross, VGroup


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


def _max_alpha(mob, index):
    """The brightest opacity anywhere in ``mob``'s subtree at one time index."""
    alphas = []
    for node in [mob, *mob.get_descendants()]:
        opacity = getattr(node, "opacity", None)
        if opacity is None or opacity.shape[0] <= index or opacity.numel() == 0:
            continue
        alphas.append(float(opacity[index].reshape(-1).max()))
    return max(alphas) if alphas else 0.0


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
        cube = Cube(size=1.0)
    with Sync(runtime=1.0):
        sphere.become(cube)

    assert not [actor for actor in scene.actors if isinstance(actor, Dot3D)]


def test_a_polyhedron_speaks_only_for_the_geometry_it_built(scene):
    """An aggregator's claim covers its own construction, not a user's child.

    ``Polyhedron`` draws its faces and owns a vertex-and-edge graph it never
    draws, so neither may be published separately. A Mob a *user* hangs on the
    Polyhedron is neither: the Polyhedron will not draw it, so withholding it
    from the Scene makes it disappear from the morph result even though
    spawning the same hierarchy directly shows it.
    """
    with Off():
        cube = Cube(size=1.0)
        extra = Sphere(radius=0.25).move(UP * 1.2)
        cube.add_children(extra)
        cube.spawn()
    spawned = {id(actor) for actor in _rendering_actors(scene) if actor.is_spawned()}
    assert id(extra) in spawned, "the fixture no longer registers the user's child"

    with Scene() as fresh:
        with Off():
            source = Sphere(radius=0.8).spawn()
            target = Cube(size=1.0)
            target.add_children(Sphere(radius=0.25).move(UP * 1.2))
        with Sync(runtime=1.0):
            result = source.become(target)
        drawn = [
            type(actor).__name__
            for actor in _rendering_actors(fresh)
            if actor.is_spawned() and not actor.is_despawned()
        ]
        assert result is not None
        assert "Sphere" in drawn, (
            f"the user's child was withheld from the Scene; drawn: {drawn}"
        )


def test_a_morphed_polyhedron_draws_each_face_once(scene):
    """The faces must not be published alongside the Polyhedron that draws them.

    A face registered in its own right is drawn twice: once by the Polyhedron
    and once by itself. The two copies are coplanar, so the double draw shows
    up along every silhouette edge rather than as an obviously wrong picture.
    """
    with Off():
        reference = Cube(size=1.0).spawn()
    expected = len(_rendering_actors(scene))

    with Scene() as fresh:
        with Off():
            sphere = Sphere(radius=0.8).spawn()
            cube = Cube(size=1.0)
        with Sync(runtime=1.0):
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
    with Sync(runtime=1.0):
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
        source = Sphere(radius=0.7, u_range=(0.0, 90.0)).spawn()
        target = Sphere(radius=0.7)
    assert (source.two_sided, source.closed_shell) != (
        target.two_sided,
        target.closed_shell,
    ), "the fixture no longer contrasts the two declarations"

    with Sync(runtime=1.0):
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
        source = Square(color=BLUE, filled=source_filled, stroke_width=0.05).spawn()
        target = Square(color=BLUE, filled=not source_filled, stroke_width=0.05)
    with Sync(runtime=1.0):
        result = source.become(target)
    assert result.filled is (not source_filled)


# ---------------------------------------------------------------------------
# A morph occupies its context whether or not it has anything to do
# ---------------------------------------------------------------------------


def test_an_empty_morph_still_spends_its_duration(scene):
    """Two empty Groups record nothing, so the context cursor never moved.

    Every other become spends exactly one ``runtime``. A no-op that spends
    zero silently pulls everything after it in a ``Seq`` a second early.
    """
    with Off():
        source = Group().spawn()
        target = Group()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)
    assert end - start == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A morph ends on the target, not on an approximation of it
# ---------------------------------------------------------------------------


def _wave(grid_width, grid_height):
    return Surface(
        lambda uv: torch.cat(
            (
                uv - 0.5,
                0.25 * torch.sin(6 * uv[..., :1]) * torch.cos(6 * uv[..., 1:]),
            ),
            -1,
        ),
        grid_width=grid_width,
        grid_height=grid_height,
    )


def test_morphing_into_a_coarser_surface_lands_on_that_surface(scene):
    """The morph must end on the target SURFACE, not on a resampling of it.

    Grid reconciliation moves both sides to the finer grid per axis, so a
    target coarser along an axis is resampled upward -- and interpolating a
    coarse grid does not reproduce the surface those samples came from. The
    morph used to end on a bilinear re-sampling of four columns of a wave
    rather than on the wave: 0.0258 mean deviation from the analytic surface,
    0.108 at worst. Re-evaluating the Surface's own parametric function on the
    reconciled grid ends on the wave itself.

    Measured against the analytic surface rather than the target's sample
    POINTS on purpose. The result is sampled at 6x9 where the target is 4x9, so
    the two point sets do not coincide however right the shape is -- a nearest-
    point comparison would report half a grid cell and call a correct morph
    wrong. What has to be true is that every point lies on the surface.
    """
    with Off():
        source = Surface(
            lambda uv: torch.cat((uv - 0.5, torch.zeros_like(uv[..., :1])), -1),
            grid_width=6,
            grid_height=6,
        ).spawn()
        target = _wave(4, 9)

    with Sync(runtime=1.0):
        source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)
    scene.timeline_manager.set_state_to_times(torch.tensor([end]))
    points = _visible_points(scene)
    scene.timeline_manager.clear_buffers()

    inside = (points[:, 0].abs() <= 0.5001) & (points[:, 1].abs() <= 0.5001)
    surface_points = points[inside]
    assert surface_points.shape[0] >= 40, "too few points to say anything"
    u = surface_points[:, 0] + 0.5
    v = surface_points[:, 1] + 0.5
    expected_z = 0.25 * torch.sin(6 * u) * torch.cos(6 * v)
    deviation = (surface_points[:, 2] - expected_z).abs()
    assert float(deviation.mean()) < 0.005, (
        f"the morph ended {float(deviation.mean()):.4f} off the target surface "
        f"on average (worst {float(deviation.max()):.4f})"
    )


# ---------------------------------------------------------------------------
# The assignment: geometry decides, not just the order children were listed in
# ---------------------------------------------------------------------------


def _part_centers(mob, index):
    centers = []
    for node in [mob, *mob.get_descendants()]:
        location = getattr(node, "location", None)
        if location is None or location.shape[0] <= index or not location.numel():
            continue
        if not hasattr(node, "get_render_primitives"):
            continue
        rows = location[index].reshape(-1, 3)
        centers.append((rows.amin(0) + rows.amax(0)) / 2)
    return centers


def test_parts_that_need_not_move_do_not_travel(scene):
    """Four squares in the same four places, listed in a different order.

    Nothing has to move. The old cost added a distance capped at 1e-3 to an
    order gap spanning [0, 1], so traversal order decided the assignment
    outright: all four converged on the centre of the screen, overlapped, and
    flew back out. The rebalanced cost normalizes the two against each other,
    and geometry wins where it is this lopsided.
    """
    places = [LEFT * 2 + UP, RIGHT * 2 + UP, RIGHT * 2 - UP, LEFT * 2 - UP]

    def build(order):
        return Group(*[Square(size=0.7).move(places[i]) for i in order])

    with Off():
        source = build([0, 1, 2, 3]).spawn()
        target = build([2, 0, 3, 1])
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        morphed = source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)

    times = [start, (start + end) / 2, end]
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    start_centers = _part_centers(morphed, 0)
    middle_centers = _part_centers(morphed, 1)
    scene.timeline_manager.clear_buffers()

    assert len(start_centers) == len(middle_centers) == 4
    travelled = max(
        float((a - b).norm()) for a, b in zip(start_centers, middle_centers)
    )
    assert travelled < 0.25, (
        f"a square moved {travelled:.2f} units mid-morph when the two "
        f"hierarchies differ only in the order their children are listed"
    )


# ---------------------------------------------------------------------------
# A morph has to show something the whole way through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_target",
    [
        pytest.param(
            lambda: Square(size=1.6, filled=False, stroke_width=0.06),
            id="unfilled_square",
        ),
        pytest.param(
            lambda: Group(Line(LEFT, RIGHT), Line(UP, -UP)), id="crossed_lines"
        ),
    ],
)
def test_a_morph_into_a_stroke_only_shape_is_never_blank(scene, build_target):
    """The PN medium carries fills, and a stroke-only circuit has none.

    ``_bezier_to_pn_soup`` zeroes an unfilled circuit's opacity, correctly --
    there is no interior to convert. But that makes the whole target soup
    transparent, so a geometric morph had nothing to show: the solid faded to
    nothing, roughly a third of the frames were empty, and the outline appeared
    at the end. Such a pair cross-fades instead.
    """
    with Off():
        source = Sphere(radius=0.8).spawn()
        target = build_target()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)

    times = [start + (end - start) * f for f in (0.2, 0.4, 0.5, 0.6, 0.8)]
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    counts = [
        0
        if _visible_points(scene, index) is None
        else _visible_points(scene, index).shape[0]
        for index in range(len(times))
    ]
    scene.timeline_manager.clear_buffers()

    assert all(count > 0 for count in counts), (
        f"the morph showed nothing at some point in its middle: {counts}"
    )


# ---------------------------------------------------------------------------
# An aggregate morphs and draws as one thing
# ---------------------------------------------------------------------------


def test_an_arrow3d_can_morph_in_both_directions(scene):
    """``Arrow3D`` draws its shaft, tip and end discs itself.

    It had no ``_morph_family``, so ``become`` decomposed it into those parts
    and then had to publish them separately: ``Sphere -> Arrow3D`` ended with
    the parts drawn twice, and ``Arrow3D -> Sphere`` raised outright. It is now
    one morph unit converted through the "aggregate" adapter.
    """
    with Off():
        source = Arrow3D().spawn()
        target = Sphere(radius=0.6).move(RIGHT * 1.5)
    with Sync(runtime=1.0):
        result = source.become(target)
    assert result is not None

    with Scene() as fresh:
        with Off():
            reference = Arrow3D().spawn()
        expected = len([a for a in _rendering_actors(fresh) if a.is_spawned()])
        assert reference is not None
    with Scene() as other:
        with Off():
            sphere = Sphere(radius=0.6).spawn()
            arrow = Arrow3D()
        with Sync(runtime=1.0):
            sphere.become(arrow)
        live = [
            a
            for a in _rendering_actors(other)
            if a.is_spawned() and not a.is_despawned()
        ]
        assert len(live) == expected, (
            f"a morphed Arrow3D publishes {len(live)} renderable actors where a "
            f"spawned one publishes {expected}"
        )


# ---------------------------------------------------------------------------
# A surface's image is part of what it looks like
# ---------------------------------------------------------------------------


def test_become_takes_the_targets_colour_texture(scene):
    """A texture is stored under a name encoding its own ``W * H``.

    Two surfaces with differently-sized textures therefore share no attribute
    for the same-kind morph's ``animatable_attrs`` intersection to copy, and
    the result kept the source's picture: a 4x4 red texture becoming an 8x4
    blue one ended red.
    """

    def textured(texture_width, texture_height, tint):
        surface = Surface(
            lambda uv: torch.cat((uv - 0.5, torch.zeros_like(uv[..., :1])), -1),
            grid_width=6,
            grid_height=6,
        )
        texture = torch.zeros(texture_width, texture_height, 5)
        texture[..., :3] = torch.tensor(tint, dtype=texture.dtype)
        texture[..., 3] = 1.0
        surface.color_texture = texture
        return surface

    with Off():
        source = textured(4, 4, (1.0, 0.0, 0.0)).spawn()
        target = textured(8, 4, (0.0, 0.0, 1.0))
    wanted = target.color_texture
    with Sync(runtime=1.0):
        result = source.become(target)
    got = result.color_texture

    assert got is not None
    assert wanted is not None
    assert tuple(got.shape) == tuple(wanted.shape)
    assert torch.allclose(got[..., :3], wanted[..., :3], atol=1e-3)


# ---------------------------------------------------------------------------
# A clone registers what its source registered, not everything it can reach
# ---------------------------------------------------------------------------


def test_cloning_a_polyhedron_publishes_no_vertex_beads(scene):
    """A clone is registered by the caller's policy; its parts are not.

    ``clone(add_to_scene=True)`` put the root's policy in the deepcopy memo and
    every descendant read it from there, so the copy registered geometry the
    original deliberately does not: a ``Polyhedron`` builds its faces and its
    vertex-and-edge graph with ``add_to_scene=False`` because it hands the
    faces to the renderer itself and never draws the graph at all.  A cloned
    Tetrahedron therefore grew four vertex beads and drew each of its four
    faces twice, beside an original that did neither.
    """
    with Off():
        original = Tetrahedron(edge_length=1.0).spawn()
    per_solid = len(_rendering_actors(scene))
    assert per_solid == 1, "the fixture no longer registers one actor per solid"

    with Off():
        original.clone()

    assert len(_rendering_actors(scene)) == 2 * per_solid
    assert not [actor for actor in scene.actors if isinstance(actor, Dot3D)]


def test_a_morphed_polyhedrons_history_publishes_no_vertex_beads(scene):
    """The frames *before* a morph belong to a clone, and it must look the same.

    ``become`` calls ``detach_history``, which clones the source so the clone
    can carry the recorded animation while the original starts fresh -- so
    everything the viewer sees up to the morph is the clone's rendering.  With
    the clone registering parts the original withholds, a Tetrahedron wore four
    vertex beads for its whole pre-morph life and lost them on the first frame
    of the morph, when the picture was handed back to the original.
    """
    with Off():
        source = Tetrahedron(edge_length=1.0).spawn()
        target = Sphere(radius=0.5)

    with Sync(runtime=1.0):
        source.become(target)

    beads = [actor for actor in scene.actors if isinstance(actor, Dot3D)]
    assert not beads, f"the historical clone published {len(beads)} vertex beads"
    doubled = [actor for actor in scene.actors if isinstance(actor, TriangleVertices)]
    assert not doubled, (
        f"the historical clone published {len(doubled)} face triangles the "
        "Tetrahedron draws itself, so every face was drawn twice"
    )


# ---------------------------------------------------------------------------
# Geometry with no counterpart arrives rather than popping
# ---------------------------------------------------------------------------


def test_a_surplus_target_fades_in_as_it_grows(scene):
    """A collapsed seed at full opacity is a bright speck that came from nowhere.

    A target primitive with no source is grown from a clone of itself collapsed
    onto the nearest existing source point.  At zero size that clone still
    carried the target's colour and material, so it rendered as a hard dot
    sitting at an unrelated vertex for a third of the morph before inflating
    into a solid.  It has to arrive, not appear.
    """
    with Off():
        source = Sphere(radius=0.4).move(LEFT).spawn()
        target = Group(
            Sphere(radius=0.4).move(LEFT),
            Sphere(radius=0.4).move(RIGHT),
        )
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        result = source.become(target)
    end = float(scene.animation_manager.context.timespan.current_time)

    scene.timeline_manager.set_state_to_times(torch.tensor([start, end]))
    grown = result[1]
    first = _max_alpha(grown, 0)
    last = _max_alpha(grown, 1)
    scene.timeline_manager.clear_buffers()

    assert first <= 1e-3, f"the surplus target started visible (alpha {first:.3f})"
    assert last >= 0.99, f"the surplus target never became solid (alpha {last:.3f})"


# ---------------------------------------------------------------------------
# A flag the renderer reads once cannot be animated, so it is not travelled
# ---------------------------------------------------------------------------


def test_a_fill_crossing_pair_is_ranked_below_a_like_filled_one(scene):
    """Crossing ``filled`` costs half a compatibility band, not a whole one.

    Type identity still leads -- a filled Square would rather become an
    unfilled Square than a filled Circle, which is the same rule that sends
    ``Square@left + Circle@right -> Circle@left + Square@right`` across the
    screen instead of changing shape in place.  But among counterparts of equal
    type and family, one that does not force the crossing wins.
    """
    rank = MobMorphMixin._primitive_compatibility_rank
    with Off():
        filled = Square(size=0.6, filled=True)
        also_filled = Square(size=0.6, filled=True)
        unfilled = Square(size=0.6, filled=False)
        filled_circle = Circle(radius=0.3, filled=True)

    assert rank(filled, also_filled) < rank(filled, unfilled)
    assert rank(filled, unfilled) < rank(filled, filled_circle)


@pytest.mark.parametrize("source_filled", [True, False])
def test_a_fill_crossing_morph_does_not_play_in_the_endpoints_fill(
    scene, source_filled
):
    """``filled`` is read once per render, so adopting it is not an ending.

    ``_adopt_structural_attrs`` runs after the recorded morph, but the timeline
    is fully recorded before anything renders -- so the renderer reads the
    adopted value on *every* frame of that mob's life, and a filled Circle
    became an outline on the morph's first frame and stayed one.  The flag also
    decides where the stroke goes, not merely whether the interior shows, so
    nothing animatable interpolates between the two: such a pair cross-fades,
    which leaves the source holding its own fill for as long as it is visible.
    """
    with Off():
        source = Square(color=BLUE, filled=source_filled, stroke_width=0.05).spawn()
        target = Square(color=BLUE, filled=not source_filled, stroke_width=0.05)

    with Sync(runtime=1.0):
        result = source.become(target)

    assert result.filled is (not source_filled)
    assert result is not source
    assert source.filled is source_filled, (
        "the source was made to render in the endpoint's fill for the whole "
        "morph rather than fading out in its own"
    )
