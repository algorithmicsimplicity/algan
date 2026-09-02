"""Contracts of ``Mob.become``.

``become`` is the only public operation that changes a Mob's *structure* while
it is on screen: it re-batches the Mob onto fresh timeline rows, pads whichever
side has fewer parts, and then animates every attribute across.  All three of
those steps fail quietly rather than loudly -- a mispaired part flies across the
screen, a dropped target leaves geometry behind, and a morph recorded against
the wrong rows corrupts an unrelated Mob.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE,
    LEFT,
    RIGHT,
    YELLOW,
    Circle,
    Cylinder,
    Group,
    ImageMob,
    Off,
    RegularPolygon,
    Scene,
    Sphere,
    Square,
    Surface,
    Sync,
    Tetrahedron,
    Text,
    TriangleMesh,
    TriangleVertices,
)
from algan.animatable_base.mob_morph import MobMorphMixin


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _at(scene, mob, times):
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    return mob


def _geometry(mob):
    """Every point the Mob and its descendants own, at the current state."""
    points = [mob.location.reshape(-1, 3)]
    for child in mob.get_descendants():
        points.append(child.location.reshape(-1, 3))
    return torch.cat(points, 0)


def test_become_hands_back_a_spawned_registered_mob_to_keep_animating(scene):
    with Off():
        square = Square(color=BLUE).spawn()
    with Sync(runtime=1.0):
        morphed = square.become(Circle(radius=0.6, color=YELLOW))

    # ``detach_history=True`` may re-batch onto fresh rows and hand back a
    # different object, so the caller must use the *returned* Mob afterwards.
    # Whatever comes back has to be on screen and reachable by the renderer.
    assert morphed is not None
    assert morphed in scene.actors
    assert morphed.is_spawned()


def test_become_reaches_the_targets_appearance_and_travels_to_get_there(scene):
    with Off():
        square = Square(color=BLUE).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        morphed = square.become(Circle(radius=0.6, color=YELLOW))
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [start, (start + end) / 2, end])
    colors = morphed.color.reshape(3, -1, 5)
    # Ends at the target colour, and is genuinely in between halfway through.
    assert torch.allclose(
        colors[2, :, :3], colors[2, :1, :3].expand_as(colors[2, :, :3])
    )
    assert not torch.allclose(colors[0], colors[2], atol=1e-3)
    assert not torch.allclose(colors[1], colors[0], atol=1e-3)
    assert not torch.allclose(colors[1], colors[2], atol=1e-3)


def test_become_morphs_position_as_well_as_shape(scene):
    """Documented Transform semantics.

    This is why scenes build their morph targets where the Mob already is.
    """
    with Off():
        square = Square().move(LEFT * 2).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        morphed = square.become(Circle(radius=0.6).move(RIGHT * 2))
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [start, end])
    centers = _center_per_time(morphed, 2)
    assert centers[0][0] < -1.0
    assert centers[1][0] > 1.0


def _center_per_time(mob, count):
    points = [
        torch.cat(
            [mob.location[index].reshape(-1, 3)]
            + [child.location[index].reshape(-1, 3) for child in mob.get_descendants()],
            0,
        )
        for index in range(count)
    ]
    return [(p.amin(0) + p.amax(0)) / 2 for p in points]


@pytest.mark.parametrize("minimize_movement", [False, True])
def test_become_pads_whichever_side_has_fewer_parts(scene, minimize_movement):
    """A three-glyph word has to morph into a five-glyph one."""
    with Off():
        short = Text("ab", font_size=40).spawn()
        target = Text("abcde", font_size=40)
        # Measured before the timeline is materialized: authoring and
        # materialized state cannot be interleaved.
        target_width = float(target.get_width().reshape(-1)[0])
    with Sync(runtime=1.0):
        morphed = short.become(target, minimize_movement=minimize_movement)
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [end])
    points = _geometry(morphed)
    width = float(points[:, 0].amax() - points[:, 0].amin())
    assert width == pytest.approx(target_width, rel=0.15)


def test_minimize_movement_keeps_parts_closer_to_where_they_started(scene):
    """The whole point of the flag: pairing by proximity, not by index."""

    def total_travel(minimize):
        with Scene() as isolated:
            with Off():
                source = Group(
                    *[Square(size=0.4).move(RIGHT * x) for x in (-2, 0, 2)]
                ).spawn()
            start = float(isolated.animation_manager.context.timespan.current_time)
            with Sync(runtime=1.0):
                # Same three squares, listed in reverse order.
                morphed = source.become(
                    Group(*[Square(size=0.4).move(RIGHT * x) for x in (2, 0, -2)]),
                    minimize_movement=minimize,
                )
            end = float(isolated.animation_manager.context.timespan.current_time)
            isolated.timeline_manager.set_state_to_times(torch.tensor([start, end]))
            first = torch.cat(
                [c.location[0].reshape(-1, 3) for c in morphed.get_descendants()], 0
            )
            last = torch.cat(
                [c.location[1].reshape(-1, 3) for c in morphed.get_descendants()], 0
            )
            return float((last - first).norm(dim=-1).sum())

    assert total_travel(minimize=True) <= total_travel(minimize=False)


def test_become_crosses_primitive_types_and_returns_the_real_target(scene):
    with Off():
        square = Square().spawn()
    result = square.become(Sphere(radius=0.5))

    assert isinstance(result, Sphere)
    assert result in scene.actors
    assert result.is_spawned()
    assert not result.is_despawned()
    assert square.is_despawned()


def test_become_does_not_spawn_or_mutate_the_target(scene):
    with Off():
        square = Square().spawn()
        target = Circle(radius=0.6)
    target_center_before = target.get_center().clone()

    with Sync(runtime=1.0):
        square.become(target)

    assert not target.is_spawned()
    assert torch.allclose(target.get_center(), target_center_before, atol=1e-5)


def test_chained_becomes_keep_animating(scene):
    """Each morph must hand back a Mob the next one can morph again."""
    with Off():
        mob = Square(color=BLUE).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        mob = mob.become(Circle(radius=0.6))
    with Sync(runtime=1.0):
        mob = mob.become(RegularPolygon(6, radius=0.6, color=YELLOW))
    end = float(scene.animation_manager.context.timespan.current_time)

    assert end - start == pytest.approx(2.0, abs=1e-6)
    _at(scene, mob, [start, (start + end) / 2, end])
    assert mob.color.shape[0] == 3


def test_empty_container_grows_every_target_child_instead_of_dropping_them(scene):
    with Off():
        source = Group(scene=scene).spawn()
    target = Group(
        Square(scene=scene, add_to_scene=False),
        Circle(scene=scene, add_to_scene=False),
        scene=scene,
        add_to_scene=False,
    )

    result = source.become(target)

    assert [type(child) for child in result.get_non_component_children()] == [
        Square,
        Circle,
    ]
    assert all(child in scene.actors for child in result.children)


def test_cross_kind_replacement_keeps_the_sources_parent_slot(scene):
    with Off():
        group = Group(Square(scene=scene), scene=scene).spawn()
    source = group[0]

    result = source.become(Sphere(radius=0.5, scene=scene, add_to_scene=False))

    assert group[0] is result
    assert isinstance(group[0], Sphere)
    assert source not in group.children


def test_cross_kind_without_detach_history_keeps_identity_and_dissolves(scene):
    with Off():
        source = Square(scene=scene).spawn()

    result = source.become(
        Sphere(radius=0.5, scene=scene, add_to_scene=False),
        detach_history=False,
    )

    assert result is source
    assert source.is_despawned()
    assert any(
        isinstance(actor, Sphere) and actor.is_spawned() and not actor.is_despawned()
        for actor in scene.actors
    )


def test_hierarchy_pairing_morphs_across_container_boundaries(scene, monkeypatch):
    def reject_dissolve(*args, **kwargs):
        pytest.fail("non-image hierarchy primitives must not dissolve")

    monkeypatch.setattr(MobMorphMixin, "_record_dissolve", reject_dissolve)
    with Off():
        source = Group(Group(Square()), Circle()).spawn()
        target = Group(Sphere(radius=0.45), Group(Tetrahedron(edge_length=0.8)))

    result = source.become(target, strategy="morph")

    assert isinstance(result, Group)
    assert isinstance(result[0], Sphere)
    assert isinstance(result[1], Group)
    assert isinstance(result[1][0], Tetrahedron)
    assert any(parent is result for parent in result[0].parents)

    before = result[1][0].get_center().clone()
    with Off():
        result.move(RIGHT)
    assert torch.allclose(result[1][0].get_center(), before + RIGHT, atol=1e-5)


def test_hierarchy_pairing_grows_collapsed_extra_targets(scene, monkeypatch):
    def reject_dissolve(*args, **kwargs):
        pytest.fail("surplus non-image targets must grow through a morph")

    pn_pairs = []
    record_pn_morph = MobMorphMixin._record_pn_morph

    def track_pn_morph(self, source, target, **kwargs):
        pn_pairs.append((type(source), type(target)))
        return record_pn_morph(self, source, target, **kwargs)

    monkeypatch.setattr(MobMorphMixin, "_record_dissolve", reject_dissolve)
    monkeypatch.setattr(MobMorphMixin, "_record_pn_morph", track_pn_morph)
    with Off():
        source = Square().spawn()
        target = Group(
            Sphere(radius=0.4).move(LEFT),
            Tetrahedron(edge_length=0.75).move(RIGHT),
        )

    result = source.become(target)

    assert isinstance(result, Group)
    assert [type(child) for child in result.children] == [Sphere, Tetrahedron]
    assert all(child.is_spawned() and not child.is_despawned() for child in result)
    assert len(pn_pairs) == 1


def test_extra_surface_target_does_not_duplicate_a_matched_sphere(scene, monkeypatch):
    def reject_dissolve(*args, **kwargs):
        pytest.fail("surplus non-image targets must grow through a morph")

    pn_pairs = []
    record_pn_morph = MobMorphMixin._record_pn_morph

    def track_pn_morph(self, source, target, **kwargs):
        pn_pairs.append((type(source), type(target)))
        return record_pn_morph(self, source, target, **kwargs)

    monkeypatch.setattr(MobMorphMixin, "_record_dissolve", reject_dissolve)
    monkeypatch.setattr(MobMorphMixin, "_record_pn_morph", track_pn_morph)
    with Off():
        source = Sphere(radius=0.5).spawn()
        target = Group(
            Sphere(radius=0.4).move(LEFT),
            Cylinder(radius=0.3, height=0.8).move(RIGHT),
        )

    result = source.become(target, minimize_movement=True)

    assert isinstance(result, Group)
    assert [type(child) for child in result.children] == [Sphere, Cylinder]
    assert pn_pairs == []
    assert torch.allclose(result[1].grid.location, target[1].grid.location, atol=1e-6)


def test_hierarchy_pairing_collapses_surplus_sources(scene, monkeypatch):
    def reject_dissolve(*args, **kwargs):
        pytest.fail("surplus non-image sources must collapse through a morph")

    monkeypatch.setattr(MobMorphMixin, "_record_dissolve", reject_dissolve)
    with Off():
        source_parts = [Square().move(LEFT), Circle().move(RIGHT)]
        source = Group(*source_parts).spawn()

    result = source.become(Sphere(radius=0.5))

    assert isinstance(result, Sphere)
    assert result.is_spawned()
    assert not result.is_despawned()
    assert all(part.is_despawned() for part in source_parts)


def test_hierarchy_pairing_only_dissolves_images(scene, monkeypatch):
    calls = []
    record_dissolve = MobMorphMixin._record_dissolve

    def track_dissolve(self, source, target, **kwargs):
        calls.append((source._morph_family, target._morph_family))
        return record_dissolve(self, source, target, **kwargs)

    monkeypatch.setattr(MobMorphMixin, "_record_dissolve", track_dissolve)
    pixels = torch.ones((2, 2, 4))
    with Off():
        source = Group(ImageMob(pixels), Square()).spawn()
        target = Group(Sphere(radius=0.4), Tetrahedron(edge_length=0.75))

    source.become(target)

    assert len(calls) == 1
    assert "image" in calls[0]


def test_forced_morph_rejects_image_without_a_converter(scene):
    with Off():
        source = ImageMob(torch.ones((2, 2, 4))).spawn()

    with pytest.raises(NotImplementedError, match="ImageMob"):
        source.become(Sphere(radius=0.5), strategy="morph")
    assert source.is_spawned()
    assert not source.is_despawned()


def test_forced_cross_kind_morph_rejects_identity_preservation(scene):
    with Off():
        source = Square(scene=scene).spawn()

    with pytest.raises(NotImplementedError, match="detach_history=True"):
        source.become(
            Sphere(radius=0.5, scene=scene, add_to_scene=False),
            detach_history=False,
            strategy="morph",
        )
    assert source.is_spawned()
    assert not source.is_despawned()


def test_grid_morph_reconciles_resolution_and_plain_grid_metadata(scene):
    with Off():
        source = Surface(
            grid_width=3, grid_height=4, scene=scene, add_to_scene=True
        ).spawn()
    target = Surface(
        grid_width=6,
        grid_height=5,
        scene=scene,
        add_to_scene=False,
    )

    result = source.become(target)

    assert (result.grid_width, result.grid_height) == (6, 5)
    assert result.resolution == (5, 4)
    assert result.grid.location.shape[-2] == 30


def test_different_surface_types_use_pn_and_keep_target_topology(scene):
    with Off():
        source = Sphere(radius=0.5).spawn()
        target = Cylinder(radius=0.3, height=0.8, grid_width=12, grid_height=4)

    result = source.become(target)

    assert isinstance(result, Cylinder)
    assert (result.grid_width, result.grid_height) == (12, 4)
    assert torch.allclose(result.grid.location, target.grid.location, atol=1e-6)


def test_triangle_vertices_rebatch_and_adopt_plain_normals(scene):
    one_triangle = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    two_triangles = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    source_normals = torch.tensor(((0.0, 0.0, 1.0),)).expand(3, -1)
    target_normals = torch.tensor(((0.0, 1.0, 0.0),)).expand(6, -1)
    with Off():
        source = TriangleVertices(one_triangle, source_normals, scene=scene).spawn()
    target = TriangleVertices(
        two_triangles,
        target_normals,
        scene=scene,
        add_to_scene=False,
    )

    result = source.become(target)

    assert result.location.shape[-2] == 6
    assert torch.equal(result.normals, target_normals)


def test_cubic_segment_padding_keeps_new_segments_on_the_source_contour(scene):
    source = torch.tensor(
        (
            (
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.6, 0.0, 0.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (1.2, 0.0, 0.0),
                (1.4, 0.0, 0.0),
                (1.6, 0.0, 0.0),
            ),
        )
    )
    carrier = Square(scene=scene, add_to_scene=False)

    expanded = carrier._expand_n_tensor(source, 3)

    # Two source segments expand to five slots: [0, 0, 0, 1, 1]. Every
    # inserted segment stays collapsed at the preceding source endpoint rather
    # than jumping to another globally-nearest contour point.
    assert torch.equal(expanded[0], source[0])
    assert torch.equal(expanded[1], source[0, -1].expand_as(expanded[1]))
    assert torch.equal(expanded[2], source[0, -1].expand_as(expanded[2]))
    assert torch.equal(expanded[3], source[1])
    assert torch.equal(expanded[4], source[1, -1].expand_as(expanded[4]))


def test_triangle_mesh_rebatches_corner_metadata_with_triangle_geometry(scene):
    source_vertices = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    target_vertices = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )
    with Off():
        source = TriangleMesh(
            source_vertices,
            torch.tensor(((0, 1, 2),)),
            scene=scene,
        ).spawn()
    target = TriangleMesh(
        target_vertices,
        torch.tensor(((0, 1, 2), (1, 3, 2))),
        scene=scene,
        add_to_scene=False,
    )

    result = source.become(target)

    assert result.grid.num_points_per_object == 3
    assert result.grid.location.shape[-2] == 6
    assert result.corner_index.shape == (6,)
    assert result.num_triangles == 2


def test_pn_swap_uses_half_open_lifespans_with_no_gap_or_double_draw(scene):
    from algan.mobs.pn_mesh import PNMesh

    with Off():
        source = Square(scene=scene).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(runtime=1.0):
        result = source.become(Sphere(radius=0.5, scene=scene, add_to_scene=False))
    end = float(scene.animation_manager.context.timespan.current_time)
    soup = next(actor for actor in scene.actors if isinstance(actor, PNMesh))

    assert soup.lifespan.start() == pytest.approx(start + 0.3)
    assert soup.lifespan.end() == pytest.approx(end)
    epsilon = 1e-4
    _at(scene, result, [end - epsilon, end, end + epsilon])
    soup_visible = soup.opacity.reshape(3, -1).amax(-1) > 0
    result_visible = result.opacity.reshape(3, -1).amax(-1) > 0
    assert soup_visible.tolist() == [True, False, False]
    assert result_visible.tolist() == [False, True, True]


def test_mesh_to_circuit_swaps_borderless_then_grows_the_border(scene):
    from algan.mobs.pn_mesh import PNMesh

    corners = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with Off():
        source = TriangleVertices(corners, scene=scene).spawn()
    target = Circle(
        radius=0.5,
        stroke_width=4,
        scene=scene,
        add_to_scene=False,
    )
    with Sync(runtime=1.0):
        result = source.become(target)
    soup = next(actor for actor in scene.actors if isinstance(actor, PNMesh))

    assert soup.lifespan.start() == pytest.approx(0.0)
    assert soup.lifespan.end() == pytest.approx(0.7)
    _at(scene, result, [0.7, 0.85, 1.0])
    widths = result.stroke_width.reshape(3, -1)[:, 0]
    assert widths[0].item() == pytest.approx(0.0)
    assert 0 < widths[1].item() < 4
    assert widths[2].item() == pytest.approx(4.0)


def test_chained_become_crosses_a_primitive_family_hop(scene):
    with Off():
        mob = Square(scene=scene).spawn()
    with Sync(runtime=1.0):
        mob = mob.become(Sphere(radius=0.5, scene=scene, add_to_scene=False))
    with Sync(runtime=1.0):
        mob = mob.become(Circle(radius=0.5, scene=scene, add_to_scene=False))

    assert isinstance(mob, Circle)
    assert mob.is_spawned()
    assert not mob.is_despawned()
