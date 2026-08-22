"""Circuits that do not lie in one plane are given 3-D geometry, not flattened.

A bezier circuit is normally resolved by intersecting a camera ray with the
circuit's own plane, which costs an orthogonal projection of the control points
onto that plane -- the identity for a shape that is genuinely planar, and a
different shape for one that is not.  Manim's 3-D objects are the case that is
not: a ``Sphere`` is a grid of curved quad tiles and a 3-D ``ParametricFunction``
is a stroked path that leaves its plane entirely.

``algan.mobs.nonplanar_circuit`` classifies every circuit once, at construction,
and these are the guards on that classification and on the geometry each branch
produces.  What they are protecting, in order of how expensive it was to learn:

* **Planar shapes must not be reclassified.**  Everything 2-D in Algan goes
  through the same constructor, so a threshold that is even slightly too tight
  would silently move every glyph of every ``Text`` onto a different renderer
  path.  The separation is not marginal -- planar geometry measures at float
  noise and curved geometry two to five orders of magnitude above it -- and
  ``test_planar_shapes_keep_the_analytic_path`` is what keeps it that way.
* **Tiles must stay welded.**  Per-tile flattening moves a control point shared
  by two neighbouring tiles 0.017 world units apart, which renders as an open
  seam.  Patch mode has to leave shared corners bit-identical.
* **A path must keep its depth.**  Flattening a helix discards its entire
  radius; the stroke split has to preserve every authored control point.

These are tensor assertions on the classification and the primitive build --
no render and no Taichi.  They are feature tests of this one module rather than
of anything the timeline or the Scene can break, so they stay out of the fast
suite (``tests/README.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import algan.external_libraries.manim as manim
from algan import (
    RIGHT,
    UP,
    Circle,
    Group,
    Line,
    ManimMob,
    RegularPolygon,
    Square,
    Text,
)
from algan.mobs.nonplanar_circuit import (
    PLANARITY_TOLERANCE,
    classify_circuit,
    plane_residual_ratio,
    straightness_ratio,
    subpath_bounds,
)
from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive


def _sphere_tiles(resolution=(8, 4)):
    """The individual quad tiles of a stock Manim sphere, in grid order."""
    sphere = manim.Sphere(resolution=resolution)
    return list(sphere.family_members_with_points())


def _helix(turns=3, stroke_width=4):
    return manim.ParametricFunction(
        lambda t: np.array([np.cos(t), np.sin(t), t / 3.0]),
        t_range=[0, turns * 2 * np.pi],
        stroke_width=stroke_width,
    )


def _primitives(mob):
    got = mob.get_render_primitives()
    if got is None:
        return []
    return got if isinstance(got, list) else [got]


def _control_points(mob):
    return mob.control_points.location.reshape(-1, 3)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: Square(), id="square"),
        pytest.param(lambda: Circle(), id="circle"),
        pytest.param(lambda: RegularPolygon(7), id="heptagon"),
        pytest.param(lambda: Line(-RIGHT, RIGHT), id="line"),
        pytest.param(lambda: Square().rotate(37, RIGHT + UP), id="tilted-square"),
        pytest.param(lambda: Text("Ag"), id="text"),
    ],
)
def test_planar_shapes_keep_the_analytic_path(make):
    """Nothing 2-D is reclassified -- not even tilted or curved.

    Every one of these is planar by construction, so the classifier must return
    ``None`` and leave the circuit on the analytic bezier path.  A ``Text``
    carries one circuit per glyph and each of them has to pass.
    """
    mob = make()
    for circuit in [mob, *Group(mob).get_descendants()]:
        plan = getattr(circuit, "_nonplanar_plan", "missing")
        if plan == "missing":
            continue
        assert plan is None, f"{type(circuit).__name__} was reclassified"


def test_planar_and_curved_geometry_are_orders_of_magnitude_apart():
    """The threshold sits in an empty gap, not on a boundary.

    The whole design rests on there being no ambiguous middle: if planar shapes
    landed near :data:`PLANARITY_TOLERANCE` then normal 2-D authoring would flip
    onto the 3-D path on rounding alone.  Measured here rather than asserted in
    a comment.
    """
    planar = torch.as_tensor(np.asarray(manim.Square().rotate(0.6).points)).float()
    tile = torch.as_tensor(np.asarray(_sphere_tiles()[6].points)).float()
    helix = torch.as_tensor(np.asarray(_helix(turns=1).points)).float()

    assert plane_residual_ratio(planar) < PLANARITY_TOLERANCE / 100
    assert plane_residual_ratio(tile) > PLANARITY_TOLERANCE * 10
    assert plane_residual_ratio(helix) > PLANARITY_TOLERANCE * 100


def test_the_batched_and_single_window_measurements_agree():
    """One covariance routine, two paths through it, and they must not drift.

    Sub-path planarity is measured for every window at once (a page of ``Text``
    is one circuit with a thousand of them, and looping cost 0.35s), with a
    short-cut for the single-window case every constructed shape hits.  The
    short-cut centres its window directly while the batched path differences
    prefix sums, so this pins the two together on geometry whose answer is a
    float-noise residual -- the regime the whole threshold lives in.
    """
    from algan.mobs.nonplanar_circuit import _eigen_ratios, _window_covariances

    tiles = _sphere_tiles(resolution=(6, 3))
    points = torch.as_tensor(np.concatenate([tile.points for tile in tiles])).float()
    lengths = [len(tile.points) for tile in tiles]
    starts = torch.tensor([sum(lengths[:i]) for i in range(len(lengths))])
    ends = starts + torch.tensor(lengths)

    batched = _eigen_ratios(_window_covariances(points, starts, ends), 0)
    for index in range(len(tiles)):
        single = _eigen_ratios(
            _window_covariances(
                points, starts[index : index + 1], ends[index : index + 1]
            ),
            0,
        )
        assert float(single[0]) == pytest.approx(float(batched[index]), rel=1e-6)


def test_filled_manim_tile_becomes_patches_and_open_path_becomes_a_stroke():
    tile = ManimMob(_sphere_tiles()[6], add_to_scene=False)
    assert tile._nonplanar_plan is not None
    assert tile._nonplanar_plan.mode == "patch"

    helix = ManimMob(_helix(), add_to_scene=False)
    assert helix._nonplanar_plan is not None
    assert helix._nonplanar_plan.mode == "stroke"


def test_a_packed_circuit_is_judged_on_its_members_not_their_union():
    """Two planar squares facing different ways are still two planar squares.

    A packed circuit concatenates its members' control points, and that union is
    non-planar the moment two members are not coplanar.  Classification is
    per sub-path precisely so a page of tilted shapes does not turn itself into
    3-D geometry.
    """
    first = manim.Square()
    second = manim.Square().rotate(np.pi / 3, axis=np.array([1.0, 0.0, 0.0]))
    points = torch.as_tensor(np.concatenate([first.points, second.points])).float()
    assert plane_residual_ratio(points) > PLANARITY_TOLERANCE
    assert classify_circuit(points, filled=True) is None


def test_a_pack_is_classified_against_all_of_its_members():
    """``batch_mobs`` clones its first member, so the plan has to be redone.

    A pack is built by cloning member zero and then writing every member's rows
    into the clone, which leaves a construction-time decision describing one
    tile of what is now a whole sphere -- and it did: ``batch=True`` rendered
    the whole sphere as its first tile until ``_after_repack`` was added.
    """
    tiles = _sphere_tiles(resolution=(6, 3))
    packed = ManimMob(manim.Sphere(resolution=(6, 3)), batch=True)
    (circuit,) = [
        child
        for child in packed.get_descendants()
        if getattr(child, "_nonplanar_plan", None) is not None
    ]
    circuit.spawn(animate=False)
    plan = circuit._nonplanar_plan
    assert plan.mode == "patch"
    assert plan.num_subpaths == len(tiles)

    (patches,) = [
        p for p in _primitives(circuit) if isinstance(p, LogicalPNTrianglePrimitive)
    ]
    # Every tile contributes at least one triangle; a sphere's pole rows
    # collapse to one each and its middle rows to two.
    assert patches.corners.shape[-2] // 3 >= len(tiles)
    assert patches.corners.shape[-2] == patches.colors.shape[-2]


def test_the_flag_restores_flattening(monkeypatch):
    monkeypatch.setenv("ALGAN_NONPLANAR_CIRCUITS", "0")
    assert ManimMob(_sphere_tiles()[6], add_to_scene=False)._nonplanar_plan is None
    assert ManimMob(_helix(), add_to_scene=False)._nonplanar_plan is None


def test_a_degenerate_circuit_is_left_alone():
    """An empty Mob's synthesized control points are a single repeated point."""
    empty = ManimMob(manim.VMobject(), add_to_scene=False)
    assert empty._nonplanar_plan is None


# --------------------------------------------------------------------------
# Patch mode
# --------------------------------------------------------------------------


def test_patch_corners_are_the_authored_control_points():
    """No flattening: every PN corner is a control point of the tile itself.

    The flattened path projects control points onto the tile's plane, which
    moves all of them.  Patch mode has to leave them exactly where the author
    put them, or the tile is a different shape however smoothly it is diced.
    """
    tile = ManimMob(_sphere_tiles()[6], add_to_scene=False).spawn(animate=False)
    (primitive,) = [
        p for p in _primitives(tile) if isinstance(p, LogicalPNTrianglePrimitive)
    ]
    corners = primitive.corners.reshape(-1, 3)
    authored = _control_points(tile)
    distance = (corners.unsqueeze(1) - authored.unsqueeze(0)).norm(dim=-1)
    assert float(distance.amin(1).max()) == pytest.approx(0.0, abs=1e-6)


def test_neighbouring_tiles_stay_welded():
    """Shared corners come out bit-identical, so the mesh has no seams.

    This is the defect that motivates patch mode: projecting each tile onto its
    own plane pulls a shared control point ~0.017 world units apart on a stock
    unit sphere.  Both halves are asserted -- that the new path welds, and that
    the old one did not -- because a weld test that would also pass on the
    broken path is not a test.
    """
    tiles = _sphere_tiles(resolution=(8, 4))
    a, b = tiles[0], tiles[1]
    shared = [
        (i, j)
        for i, p in enumerate(np.asarray(a.points))
        for j, q in enumerate(np.asarray(b.points))
        if np.linalg.norm(p - q) < 1e-6
    ]
    assert shared, "expected the two tiles to share control points"

    def welded(mob):
        circuit = ManimMob(mob, add_to_scene=False).spawn(animate=False)
        (primitive,) = [
            p for p in _primitives(circuit) if isinstance(p, LogicalPNTrianglePrimitive)
        ]
        return primitive.corners.reshape(-1, 3)

    corners_a, corners_b = welded(a), welded(b)
    points_a = torch.as_tensor(np.asarray(a.points)).float()
    points_b = torch.as_tensor(np.asarray(b.points)).float()

    gap = 0.0
    for i, j in shared:
        # Only corners reach the patch; a shared handle has no PN vertex.
        near_a = (corners_a - points_a[i]).norm(dim=-1).amin()
        near_b = (corners_b - points_b[j]).norm(dim=-1).amin()
        if float(near_a) > 1e-6 or float(near_b) > 1e-6:
            continue
        gap = max(gap, float((points_a[i] - points_b[j]).norm()))
    assert gap == pytest.approx(0.0, abs=1e-6)


def test_patch_normals_follow_the_sphere():
    """Corner normals are the surface's, not the tile's flat facet normal.

    A tile's corners lie on the sphere, so their normals must be radial -- which
    is what makes the PN patch bulge out to the sphere instead of staying the
    flat quad its corners span.
    """
    tile = ManimMob(_sphere_tiles(resolution=(12, 6))[9], add_to_scene=False).spawn(
        animate=False
    )
    (primitive,) = [
        p for p in _primitives(tile) if isinstance(p, LogicalPNTrianglePrimitive)
    ]
    corners = primitive.corners.reshape(-1, 3)
    normals = primitive.normals.reshape(-1, 3)
    radial = F.normalize(corners, p=2, dim=-1)
    alignment = (radial * F.normalize(normals, p=2, dim=-1)).sum(-1).abs()
    assert float(alignment.min()) > 0.99


def test_patches_follow_the_mob_under_animation():
    """The plan is topology; the geometry is rebuilt from the live rows.

    Classification happens once at construction, so a transform applied
    afterwards has to reach the patches through the timeline rather than
    through the plan.
    """
    tile = ManimMob(_sphere_tiles()[6], add_to_scene=False).spawn(animate=False)

    def corners():
        (primitive,) = [
            p for p in _primitives(tile) if isinstance(p, LogicalPNTrianglePrimitive)
        ]
        return primitive.corners.reshape(-1, 3).clone()

    before = corners()
    tile.rotate(90, RIGHT)
    after = corners()
    assert not torch.allclose(before, after, atol=1e-4)

    # A rotation is rigid: every pairwise distance survives it.
    def gram(points):
        return (points.unsqueeze(0) - points.unsqueeze(1)).norm(dim=-1)

    assert torch.allclose(gram(before), gram(after), atol=1e-4)


def test_a_filled_tile_with_a_stroke_draws_both():
    """Manim's ``Surface`` faces carry a stroke, and it survives the conversion.

    Two primitives: the PN patches, and the boundary as stroke runs biased
    toward the camera so they land in front of the fill they outline -- which is
    the order Manim draws a filled stroked tile in.
    """
    source = _sphere_tiles()[6]
    source.set_stroke(width=4, opacity=1)
    tile = ManimMob(source, add_to_scene=False).spawn(animate=False)
    kinds = [type(p).__name__ for p in _primitives(tile)]
    assert any("LogicalPN" in kind for kind in kinds)
    assert any("Bezier" in kind for kind in kinds)


# --------------------------------------------------------------------------
# Stroke mode
# --------------------------------------------------------------------------


def test_stroke_runs_partition_every_segment_in_order():
    """Runs tile the path: no segment drawn twice, none dropped."""
    helix = ManimMob(_helix(), add_to_scene=False)
    plan = helix._nonplanar_plan
    starts, counts = plan.run_starts, plan.run_counts
    assert int(starts[0]) == 0
    assert torch.equal(starts[1:], (starts + counts)[:-1])
    assert int((starts + counts)[-1]) == helix.control_points.location.shape[-2] // 4


def test_stroke_runs_break_only_where_the_path_curves():
    """A straight 3-D path is one run however long; a helix is cut by its turn.

    The split exists to bound how far turning a run's plane toward the camera
    can displace it, so it has to key off curvature rather than off length.
    """
    straight = manim.ParametricFunction(
        lambda t: np.array([t, 2 * t, 3 * t]), t_range=[0, 4]
    )
    assert straightness_ratio(
        torch.as_tensor(np.asarray(straight.points)).float()
    ) == pytest.approx(0.0, abs=1e-5)

    plan = ManimMob(_helix(), add_to_scene=False)._nonplanar_plan
    assert plan.num_runs > 1
    # Cut by curvature, not chopped to pieces: a helix has far more segments
    # than runs.
    assert plan.num_runs < int(plan.run_counts.sum()) / 4


def test_a_stroke_keeps_the_paths_full_depth():
    """The helix's radius survives, which is exactly what flattening destroyed.

    Flattening projects the whole path onto one plane; measured on a stock
    helix that discards a full unit of out-of-plane extent and renders it as a
    flat sinusoid.
    """
    source = _helix()
    helix = ManimMob(source, add_to_scene=False).spawn(animate=False)
    (primitive,) = _primitives(helix)
    corners = primitive.corners.reshape(-1, 3)
    authored = torch.as_tensor(np.asarray(source.points)).float()
    spread = corners.amax(0) - corners.amin(0)
    expected = authored.amax(0) - authored.amin(0)
    assert torch.allclose(spread, expected, atol=1e-4)
    # Every authored control point is still present, unmoved.
    distance = (corners.unsqueeze(1) - authored.unsqueeze(0)).norm(dim=-1)
    assert float(distance.amin(1).max()) == pytest.approx(0.0, abs=1e-5)


def test_a_stroke_run_faces_the_camera():
    """The band direction is perpendicular to the view, so no run draws thin.

    A circuit's stroke is a band lying in its plane, so a run left on its own
    osculating plane vanishes wherever a 3-D path curves toward the viewer.  The
    plane is turned about the run's axis until the band direction -- the second
    basis row -- is square to the view.
    """
    from algan.mobs.nonplanar_circuit import camera_eye, run_planes
    from algan.utils.tensor_utils import unsquish

    helix = ManimMob(_helix(), add_to_scene=False).spawn(animate=False)
    x = unsquish(helix.control_points.location, -2, 4)
    eye = camera_eye(helix)
    assert eye is not None
    centre, first, second, normal = run_planes(x, helix._nonplanar_plan, eye)

    view = F.normalize(centre - eye.to(centre), p=2, dim=-1)
    band = F.normalize(second, p=2, dim=-1)
    assert float((band * view).sum(-1).abs().max()) < 1e-4
    # The run's own axis stays in the plane, so the path keeps its length and
    # its depth along itself.
    assert float((F.normalize(first, p=2, dim=-1) * normal).sum(-1).abs().max()) < 1e-4


def test_a_stroke_run_falls_back_to_its_own_plane_without_a_camera():
    from algan.mobs.nonplanar_circuit import run_planes
    from algan.utils.tensor_utils import unsquish

    helix = ManimMob(_helix(), add_to_scene=False).spawn(animate=False)
    x = unsquish(helix.control_points.location, -2, 4)
    _, first, second, normal = run_planes(x, helix._nonplanar_plan, None)
    for row in (first, second, normal):
        assert torch.isfinite(row).all()
    assert float((F.normalize(first, p=2, dim=-1) * normal).sum(-1).abs().max()) < 1e-4


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_subpath_bounds_splits_a_glyph_from_its_holes():
    """Sub-paths are what classification and the stroke split are keyed on.

    A glyph with a counter is two closed loops in one circuit, which is the
    same structure a packed circuit's members have -- and telling them apart is
    what stops a page of tilted shapes reading as one non-planar blob.
    """
    (glyph,) = [
        child
        for child in Text("o").children
        if getattr(child, "control_points", None) is not None
    ]
    corners = glyph.control_points.location.reshape(-1, 4, 3)
    assert len(subpath_bounds(corners)) >= 2


def test_a_nonplanar_circuit_is_kept_out_of_the_batched_circuit_pack():
    """The vectorized pack builds flattened circuits and knows nothing else."""
    from algan.scene_manager import SceneManager

    scene = SceneManager.instance().current_scene
    flat = Square().spawn(animate=False)
    curved = ManimMob(_sphere_tiles()[6]).spawn(animate=False)
    assert scene._is_batchable_bezier(flat)
    assert not scene._is_batchable_bezier(curved)
