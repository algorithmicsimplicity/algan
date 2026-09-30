"""Tracing rendered geometry back to Mobs, for the object-id compositing pass.

While ``scene._aux_id_registry`` is a dict, the render loop stamps every
primitive with the ``Mob.id`` of the actor that built it, the collections and
the scene merge carry those stamps, and the merged scene gains two host tables:
``tri_obj_source_ids`` (global surface id -> Mob id) and ``circuit_source_ids``
(merged circuit -> Mob id). :mod:`algan.rendering.pass_identity` resolves those
Mobs to the ids the pass writes. These tests pin:

* the tables are right -- every visible surface and circuit names the actor
  that built it, checked against the geometry itself, across every producer
  family (flat meshes, diced logical PN, packed and batched circuits, stroke
  styles, nonplanar paths, the deferred batched builders);
* nothing changes when the pass is not armed -- no keys, identical tensors;
* the tables ride both the prefetch-worker and the synchronous batch paths;
* ``Mob.pass_index`` validation and inheritance, the owner rule, and the
  colour bijection.

Most build and merge a batch without rendering it (no render kernel runs), so
they cost projection and merge only. They are feature tests of this pass and
stay out of the fast suite.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
import torch

import algan
import algan.external_libraries.manim as manim
import algan.manim as mn
from algan import (
    DOWN,
    LEFT,
    ORIGIN,
    PREVIEW,
    RIGHT,
    SETTINGS,
    UP,
    AlganConfigurationError,
    Arrow,
    Arrow3D,
    Circle,
    Cube,
    Cylinder,
    Group,
    ManimMob,
    Mob,
    NumberLine,
    Off,
    Scene,
    Sphere,
    Square,
    Text,
)
from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit
from algan.rendering import pass_identity as pi
from algan.utils.mob_utils import batch_mobs

TINY = PREVIEW.set(resolution=(64, 36), frames_per_second=4)


# ---------------------------------------------------------------------------
# Mob.pass_index
# ---------------------------------------------------------------------------


def test_pass_index_defaults_to_none_and_accepts_whole_numbers():
    with Scene(), Off():
        square = Square()
        assert square.pass_index is None
        for value in (1, 7, 65535, np.int64(12), torch.tensor(3)):
            square.pass_index = value
            assert square.pass_index == int(value)
            assert type(square.pass_index) is int
        square.pass_index = None
        assert square.pass_index is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.True_,
        torch.tensor(True),
        0,
        -1,
        65536,
        1.0,
        2.5,
        "3",
        [1],
        float("nan"),
    ],
)
def test_pass_index_rejects_everything_else(value):
    with Scene(), Off():
        square = Square()
        square.pass_index = 4
        with pytest.raises(AlganConfigurationError):
            square.pass_index = value
        assert square.pass_index == 4, "a rejected value must leave the old one"


def test_pass_index_is_inherited_from_the_nearest_ancestor():
    with Scene(), Off():
        a, b, c = Square(), Circle(), Square()
        inner = Group(c)
        outer = Group(a, b, inner)
        assert a._resolved_pass_index() == (None, None)

        outer.pass_index = 3
        assert a._resolved_pass_index() == (3, outer)
        assert c._resolved_pass_index() == (3, outer)

        inner.pass_index = 9
        assert c._resolved_pass_index() == (9, inner), "nearest ancestor wins"
        assert b._resolved_pass_index() == (3, outer)

        b.pass_index = 5
        assert b._resolved_pass_index() == (5, b), "a Mob's own value wins"


def test_pass_index_breadth_first_over_several_parents():
    """Two parents are both nearer than any grandparent, in ``parents`` order."""
    with Scene(), Off():
        shared = Square()
        first = Group(shared)
        second = Group(shared)
        top = Group(first)
        top.pass_index = 1
        second.pass_index = 2
        assert shared.parents[:2] == [first, second]
        assert shared._resolved_pass_index() == (2, second)
        first.pass_index = 4
        assert shared._resolved_pass_index() == (4, first)


def test_pass_index_survives_clone_and_become():
    with Scene(), Off():
        square = Square().spawn()
        square.pass_index = 6
        assert square.clone().pass_index == 6
        square.become(Circle())
        assert square.pass_index == 6, "become must not adopt the target's None"


# ---------------------------------------------------------------------------
# The owner rule
# ---------------------------------------------------------------------------


class _Composite(Mob):
    """A user composite: a Mob subclass that only holds other Mobs."""


def test_auto_owners_name_the_object_not_its_internals():
    with Scene() as scene, Off():
        text = Text("ab")
        cube = Cube()
        cylinder = Cylinder(closed=True)
        circle, square = Circle(), Square()
        Group(circle, square)
        points = Square().control_points.location.reshape(-1, 4, 3).transpose(0, 1)
        triangulated = TriangulatedBezierCircuit(points, use_cache=False)
        composite = _Composite()
        inner_a, inner_b = Square(), Circle()
        composite.add_children(Group(inner_a), inner_b)
        line = NumberLine()
        arrow = Arrow(ORIGIN, RIGHT)
        vgroup = mn.VGroup(mn.Circle(), mn.Square())
        lonely = Mob()
        Group(lonely)

        def owner(mob):
            pass_id, found = pi.resolve_pass_owner(mob)
            assert pass_id == pi.AUTO_ID_BASE + found.id
            return found

        packs = [
            actor
            for actor in scene.actors
            if actor in text.get_descendants()
            and hasattr(actor, "get_render_primitives")
        ]
        assert packs, "a Text should register its glyph pack as an actor"
        assert all(owner(pack) is text for pack in packs)
        assert owner(cube) is cube
        assert owner(cylinder.bottom_cap) is cylinder
        assert owner(cylinder.top_cap) is cylinder
        vertices = [
            d
            for d in triangulated.get_descendants()
            if type(d).__name__ == "TriangleVertices"
        ]
        assert vertices, "a TriangulatedBezierCircuit draws through TriangleVertices"
        assert all(owner(v) is triangulated for v in vertices)
        assert owner(circle) is circle, "Group(a, b) reports a and b separately"
        assert owner(square) is square
        assert owner(inner_a) is composite, "a Mob subclass holding parts owns them"
        assert owner(inner_b) is composite
        line_parts = [
            d for d in line.get_descendants() if hasattr(d, "get_render_primitives")
        ]
        assert line_parts
        assert all(owner(p) is line for p in line_parts)
        arrow_parts = [
            d for d in arrow.get_descendants() if hasattr(d, "get_render_primitives")
        ]
        assert arrow_parts
        assert all(owner(p) is arrow for p in arrow_parts)
        vgroup_parts = [
            d
            for d in vgroup.get_descendants()
            if d is not vgroup and type(d) is ManimMob
        ]
        assert len(vgroup_parts) == 2
        assert all(owner(p) is p for p in vgroup_parts), "a VGroup only groups"
        assert owner(lonely) is lonely, "an all-container chain names the Mob"


def test_an_explicit_pass_index_names_its_setter():
    with Scene(), Off():
        text = Text("ab")
        group = Group(text)
        pack = next(d for d in text.get_descendants() if d is not text)
        group.pass_index = 17
        assert pi.resolve_pass_owner(pack) == (17, group)
        text.pass_index = 18
        assert pi.resolve_pass_owner(pack) == (18, text)


def test_describe_owner_is_json_and_matches_the_viewer_label():
    from algan.viewer.pixels import mob_label

    with Scene(), Off():
        cube = Cube()
        named = Square(name="title")
        named.pass_index = 3
        for mob in (cube, named):
            description = pi.describe_owner(mob)
            assert json.loads(json.dumps(description)) == description
            assert description["label"] == mob_label(mob)
            assert description["mob_id"] == mob.id
            assert description["class"] == type(mob).__name__
        assert pi.describe_owner(cube)["name"] is None
        assert pi.describe_owner(named)["name"] == "title"
        assert pi.describe_owner(named)["pass_index"] == 3


def test_pass_id_table_composes_with_a_source_table():
    with Scene(), Off():
        a, b = Square(), Circle()
        group = Group(a)
        group.pass_index = 5
        lut, owners = pi.pass_id_table({a.id: a, b.id: b})
        assert lut[0] == 0, "source -1 must land on the background"
        assert lut[a.id + 1] == 5
        assert lut[b.id + 1] == pi.AUTO_ID_BASE + b.id
        assert owners == {5: group, pi.AUTO_ID_BASE + b.id: b}
        sources = np.array([-1, b.id, a.id], dtype=np.int32)
        assert lut[sources + 1].tolist() == [0, pi.AUTO_ID_BASE + b.id, 5]


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


def test_pass_id_colours_are_a_bijection_with_black_as_background():
    assert pi.pass_id_color(0) == (0, 0, 0)
    assert pi.pass_id_from_color((0, 0, 0)) == 0
    generator = np.random.default_rng(0)
    sample = [1, 2, 3, 255, 256, 65535, 65536, 65537, (1 << 24) - 1]
    sample += generator.integers(0, 1 << 24, size=2000).tolist()
    for pass_id in sample:
        rgb = pi.pass_id_color(pass_id)
        assert all(0 <= channel <= 255 for channel in rgb)
        assert pi.pass_id_from_color(rgb) == pass_id
    # Consecutive ids must not be near-identical colours.
    colours = np.array([pi.pass_id_color(i) for i in range(1, 257)], dtype=np.int64)
    assert len({tuple(c) for c in colours}) == len(colours)
    steps = np.abs(np.diff(colours, axis=0)).max(axis=1)
    assert steps.min() >= 32
    with pytest.raises(ValueError):
        pi.pass_id_color(1 << 24)
    with pytest.raises(ValueError):
        pi.pass_id_color(-1)


def test_pass_id_colours_vectorised_match_the_scalar_version():
    ids = torch.tensor([[0, 1, 2, 65535], [65536, 70000, (1 << 24) - 1, -1]])
    colours = pi.pass_id_colors(ids)
    assert colours.dtype == torch.uint8
    assert colours.shape == (2, 4, 3)
    for index, pass_id in np.ndenumerate(ids.numpy()):
        expected = pi.pass_id_color(pass_id) if pass_id >= 0 else (0, 0, 0)
        assert tuple(colours[index].tolist()) == expected
    decoded = pi.pass_ids_from_colors(colours)
    assert decoded.dtype == torch.int64
    assert decoded[ids >= 0].tolist() == ids[ids >= 0].tolist()
    assert decoded[1, 3] == 0, "a missing source decodes as background"
    ids32 = ids.clamp_min(0).to(torch.int32)
    assert torch.equal(pi.pass_id_colors(ids32), pi.pass_id_colors(ids.clamp_min(0)))


def test_pass_identity_is_not_star_exported():
    exported = set(algan.__all__)
    for name in dir(pi):
        if not name.startswith("_"):
            value = getattr(pi, name)
            if getattr(value, "__module__", None) == pi.__name__ or name.isupper():
                assert name not in exported, name


# ---------------------------------------------------------------------------
# Tables through the real batch build and merge
# ---------------------------------------------------------------------------


def _sphere_tile():
    tiles = list(manim.Sphere(resolution=(8, 4)).family_members_with_points())
    return tiles[6]


def _helix():
    return manim.ParametricFunction(
        lambda t: np.array([0.4 * np.cos(t), 0.4 * np.sin(t), t / 8.0]),
        t_range=[0, 2 * np.pi],
        stroke_width=4,
    )


def _grid_positions():
    xs = (-4.5, -1.5, 1.5, 4.5)
    ys = (2.4, 0.0, -2.4)
    return [RIGHT * x + UP * y for y in ys for x in xs]


def _build_zoo():
    """One of every producer family, each owner alone in its own grid cell.

    Returns ``{owner: centre}``; the owner is what ``resolve_pass_owner``
    must report for the geometry drawn in that cell.
    """
    cells = iter(_grid_positions())
    owners = {}

    def place(mob):
        owners[mob] = next(cells)
        mob.move_to(owners[mob])
        return mob

    text = place(Text("ab").scale(0.8))  # packed circuits
    cube = place(Cube().scale(0.4))  # flat, one member per triangle, mesh_key
    sphere = place(Sphere(radius=0.6))  # diced logical PN, batched surface build
    cylinder = place(Cylinder(radius=0.4, height=1.0, closed=True))  # PN + caps
    circle = place(Circle(radius=0.5))  # deferred batched circuit build
    square = place(Square(size=1.0))  # (both inside a Group, below)
    group = Group(circle, square)
    # Triangulated fills: TriangleVertices under a plain Mob under the root.
    outline = Square(size=1.2, add_to_scene=False).move_to(next(cells))
    points = outline.control_points.location.reshape(-1, 4, 3).transpose(0, 1)
    triangulated = TriangulatedBezierCircuit(points, use_cache=False)
    owners[triangulated] = outline.location.reshape(3)
    pack = place(  # a pack: two circuits, one actor
        batch_mobs(
            [Square(size=0.5).move(LEFT * 0.4), Square(size=0.5).move(RIGHT * 0.4)]
        )
    )
    styled = place(Square(size=1.0, joint_type="miter"))  # parts=2 expansion
    helix = place(ManimMob(_helix()))  # nonplanar stroke runs
    tile = place(ManimMob(_sphere_tile()).scale(2.0))  # nonplanar PN patches
    arrow = place(Arrow3D(LEFT * 0.5, RIGHT * 0.5))  # aggregate of parts
    # Each family must really take the path it is here for.
    assert pack.control_points.parent_batch_sizes is not None
    assert styled._stroke_style_key() is not None
    assert helix._nonplanar_plan.mode == "stroke"
    assert tile._nonplanar_plan.mode == "patch"
    for mob in (text, cube, sphere, cylinder, group, triangulated, pack, styled):
        mob.spawn()
    for mob in (helix, tile, arrow):
        mob.spawn()
    return owners


def _merge_one_frame(scene, *, armed):
    """Build, project and merge frame 0 the way the render loop does."""
    if armed:
        scene._aux_id_registry = {}
    try:
        actors = [
            scene.camera,
            scene.camera.screen,
            *scene.light_sources,
            *scene.actors,
        ]
        with scene._batch_prep_context():
            collections, _end, state = scene._get_batch_of_primitives(
                0, 1, actors, 10**10
            )
        scene._prewarm_render_batch(collections, state)
        merged, _env = scene._prepare_merged_host_scene(collections, render_state=state)
        registry = getattr(scene, "_aux_id_registry", None)
    finally:
        scene.__dict__.pop("_aux_id_registry", None)
    return merged, registry


def _prepare(scene):
    fps = scene.frames_per_second
    scene.scene_times.append(
        [0, max(1, round(scene._recorded_end_time_for_render() * fps))]
    )
    scene._initialize_frames()


def _nearest_owner(point, owners):
    items = list(owners.items())
    centres = torch.stack([torch.as_tensor(c, dtype=torch.float32) for _, c in items])
    return items[int((centres - point).norm(dim=-1).argmin())][0]


def _assert_tables_name_the_geometry(merged, registry, owners):
    tri_obj = merged["tri_obj"].cpu()
    tri_pos = merged["tri_pos"].cpu().float()
    tri_sources = merged["tri_obj_source_ids"]
    assert isinstance(tri_sources, np.ndarray), "host numpy, never a tensor"
    assert tri_sources.dtype == np.int32
    assert tri_sources.shape[0] >= int(tri_obj.max()) + 1
    frame = 0
    ids = tri_obj[frame % tri_obj.shape[0]].long()
    corners = tri_pos[frame % tri_pos.shape[0]].reshape(-1, 3, 3)
    area = torch.linalg.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    ).norm(dim=-1)
    real = area > 1e-9
    seen_owners = set()
    for surface in ids[real].unique().tolist():
        source = int(tri_sources[surface])
        assert source >= 0, f"surface {surface} has no source"
        assert source in registry, f"source {source} was not registered"
        _pass_id, owner = pi.resolve_pass_owner(registry[source])
        members = real & (ids == surface)
        centre = corners[members].mean(1).median(0).values
        assert owner is _nearest_owner(centre, owners), (
            f"surface {surface} (source {type(registry[source]).__name__}"
            f"#{source}) resolved to {type(owner).__name__}#{owner.id}, "
            f"but its geometry sits in another object's cell"
        )
        seen_owners.add(owner)

    circuit_sources = merged["circuit_source_ids"]
    assert isinstance(circuit_sources, np.ndarray)
    assert circuit_sources.dtype == np.int32
    assert circuit_sources.shape == (merged["num_circuits"],)
    centres = merged["circuit_meta"].cpu()[0, :, 0:3].float()
    for circuit, source in enumerate(circuit_sources.tolist()):
        assert source >= 0, f"circuit {circuit} has no source"
        _pass_id, owner = pi.resolve_pass_owner(registry[source])
        assert owner is _nearest_owner(centres[circuit], owners), (
            f"circuit {circuit} (source {type(registry[source]).__name__}"
            f"#{source}) resolved to {type(owner).__name__}#{owner.id}"
        )
        seen_owners.add(owner)
    return seen_owners


def test_every_surface_and_circuit_names_the_mob_that_drew_it():
    with Scene(video_settings=TINY) as scene:
        with Off():
            owners = _build_zoo()
        _prepare(scene)
        merged, registry = _merge_one_frame(scene, armed=True)

        seen = _assert_tables_name_the_geometry(merged, registry, owners)
        assert seen == set(owners), (
            "every object must reach the tables: missing "
            f"{[type(o).__name__ for o in set(owners) - seen]}"
        )
        # Per-owner circuit counts: the pack's two members, the styled square's
        # interior + border, the glyph pack's two glyphs.
        counts = {}
        for source in merged["circuit_source_ids"].tolist():
            owner = pi.resolve_pass_owner(registry[source])[1]
            counts[type(owner).__name__] = counts.get(type(owner).__name__, 0) + 1
        assert counts["Text"] == 2
        assert counts["Circle"] == 1
        assert counts["Square"] == 1 + 2 + 2  # plain, pack of two, styled
        assert counts["ManimMob"] >= 1  # the helix's stroke runs

        # Registered ids are actors (or an aggregate's parts), keyed by Mob.id.
        assert all(mob.id == mob_id for mob_id, mob in registry.items())


def test_nothing_changes_when_the_pass_is_not_armed():
    with Scene(video_settings=TINY) as scene:
        with Off():
            _build_zoo()
        _prepare(scene)
        plain, registry = _merge_one_frame(scene, armed=False)
        armed, _ = _merge_one_frame(scene, armed=True)

    assert registry is None
    assert "tri_obj_source_ids" not in plain
    assert "circuit_source_ids" not in plain
    assert set(armed) - set(plain) == {"tri_obj_source_ids", "circuit_source_ids"}
    for key, value in plain.items():
        other = armed[key]
        if torch.is_tensor(value):
            assert torch.equal(value, other), f"{key} changed with the pass armed"
        elif isinstance(value, (bool, int, float, str, tuple)):
            assert value == other, f"{key} changed with the pass armed"
    assert plain["tri_obj_sources"] == armed["tri_obj_sources"]


@pytest.mark.parametrize(
    ("env", "mesh_id"),
    [
        ({"ALGAN_BEZIER_GROUP_RUNS": "0"}, True),
        ({"ALGAN_BATCH_BEZIER_PREP": "0", "ALGAN_BATCH_SURFACE_PREP": "0"}, True),
        ({}, False),
    ],
    ids=["no-group-runs", "no-batched-prep", "mesh-id-off"],
)
def test_the_fallback_builders_carry_sources_too(monkeypatch, env, mesh_id):
    """The per-actor fallbacks of both deferred builders name the same Mobs.

    So does the count-based surface numbering (``mesh_id`` off).
    """
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    previous = SETTINGS.raytracing.mesh_id
    SETTINGS.raytracing.experimental.set(mesh_id=mesh_id)
    try:
        with Scene(video_settings=TINY) as scene:
            cells = iter(_grid_positions())
            owners = {}
            with Off():
                for mob in (
                    Circle(radius=0.5),
                    Square(size=1.0),
                    batch_mobs([Square(size=0.4), Square(size=0.4).move(UP * 0.5)]),
                    Sphere(radius=0.5),
                    Cylinder(radius=0.4, height=1.0, closed=True),
                    Cube().scale(0.4),
                ):
                    owners[mob] = next(cells)
                    mob.move_to(owners[mob]).spawn()
            _prepare(scene)
            merged, registry = _merge_one_frame(scene, armed=True)
            seen = _assert_tables_name_the_geometry(merged, registry, owners)
            assert seen == set(owners)
    finally:
        SETTINGS.raytracing.experimental.set(mesh_id=previous)


@pytest.mark.parametrize("prefetch", ["1", "0"], ids=["prefetch", "serial"])
def test_the_tables_reach_the_render_on_both_batch_paths(monkeypatch, prefetch):
    """A real ``get_frames`` loop with the render kernel replaced by a probe.

    One frame per batch, so with prefetch on every batch after the first is
    prepared (and, on the CPU, merged) on the worker thread.
    """
    from algan.rendering.raytracing import scene_builder

    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", prefetch)
    merged_on = []
    original_merge = scene_builder._merge_scene

    def spying_merge(*args, **kwargs):
        merged = original_merge(*args, **kwargs)
        merged_on.append(threading.current_thread().name)
        return merged

    monkeypatch.setattr(scene_builder, "_merge_scene", spying_merge)
    rendered = []

    def probe_kernel(
        primitives, scene, width, height, t0, t1, background, transparent, *a, **k
    ):
        merged = getattr(primitives[0], "_rt_device_scene", None)
        if merged is None:
            merged = scene_builder._merge_scene(
                primitives, light_sources=k.get("light_sources", ())
            )
        rendered.append(
            (
                merged.get("tri_obj_source_ids"),
                merged.get("circuit_source_ids"),
            )
        )
        return torch.zeros((t1 - t0, height, width, 3), dtype=torch.uint8)

    monkeypatch.setattr(algan.KERNEL_REGISTRY, "render_kernel", probe_kernel)
    with (
        SETTINGS.computing.override(max_animation_batch_size=1),
        Scene(video_settings=TINY) as scene,
    ):
        with Off():
            square = Square().move(LEFT * 2).spawn()
            cube = Cube().scale(0.5).move(RIGHT * 2).spawn()
        square.move(DOWN)
        scene._aux_id_registry = {}
        try:
            frames = sum(batch.shape[0] for batch in scene.get_frames(0, 4))
        finally:
            registry = scene.__dict__.pop("_aux_id_registry")

    assert frames == 4
    assert rendered, "the probe kernel never ran"
    for tri_sources, circuit_sources in rendered:
        assert tri_sources is not None
        assert circuit_sources is not None
        assert set(tri_sources.tolist()) - {-1} == {cube.id}
        assert circuit_sources.tolist() == [square.id]
    assert registry == {square.id: square, cube.id: cube}
    if prefetch == "1":
        assert any(name.startswith("algan-batch-prep") for name in merged_on), (
            "no batch was merged on the prefetch worker"
        )
    else:
        assert not any(name.startswith("algan-batch-prep") for name in merged_on)


# ---------------------------------------------------------------------------
# Unit-level carry rules
# ---------------------------------------------------------------------------


def test_collections_carry_sources_through_a_sliced_window():
    """``slice_time_window`` drops ``_rt_*`` state; the sources must survive."""
    from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive

    with Scene(), Off():
        cube, other = Cube(), Cube()
        members = []
        for mob in (cube, other):
            for primitive in mob.get_render_primitives():
                primitive._source_mob_id = mob.id
                members.append(primitive)
        collection = RayTracedTrianglePrimitive(triangle_collection=members)
    assert collection._obj_ids_sources == [cube.id, other.id]
    assert collection._obj_count_sources == [cube.id] * 12 + [other.id] * 12
    frames = collection.corners.shape[0]
    sliced = collection.slice_time_window(0, frames, frames)
    assert sliced._obj_ids_sources == [cube.id, other.id]


def test_an_unstamped_collection_carries_nothing():
    from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive

    with Scene(), Off():
        members = list(Cube().get_render_primitives())
        collection = RayTracedTrianglePrimitive(triangle_collection=members)
    assert collection._obj_ids_sources is None
    assert collection._obj_count_sources is None


def test_emitter_quads_extend_the_surface_table_with_no_source():
    from algan.rendering.raytracing.area_light_quads import (
        _extend_source_ids_for_quads,
    )

    table = np.array([4, 4, 9], dtype=np.int32)
    merged = {"tri_obj_source_ids": table}
    # ``tri_obj.max() + 1`` can fall short of the table (id 2 unused here).
    _extend_source_ids_for_quads(merged, 2, 4)
    assert merged["tri_obj_source_ids"].tolist() == [4, 4, -1, -1, -1, -1]
    assert table.tolist() == [4, 4, 9], "the original array must not be written"
    untouched = {}
    _extend_source_ids_for_quads(untouched, 0, 2)
    assert untouched == {}
