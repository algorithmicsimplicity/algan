"""Packed surfaces: one Mob standing for many spheres.

``Surface.from_batches`` is the construction-time counterpart of ``batch_mobs``
-- it builds the packed grid directly instead of packing per-member Mobs after
the fact. The two must agree exactly, so ``batch_mobs`` over separately
constructed members is the reference throughout, the same role it plays for
``BezierCircuitCubic.from_batches`` in ``test_batched_bezier_mobs.py``.
"""

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, GREEN, RED, YELLOW
from algan.constants.spatial import OUT, UP
from algan.mobs.shapes_3d import Dot3D, Sphere
from algan.scene import Scene
from algan.scene_manager import SceneManager
from algan.utils.mob_utils import BatchedMobViewSequence, batch_mobs

CENTERS = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
)
KWARGS = {"radius": 0.3, "resolution": (8, 4)}


def _members(scene, colors=None):
    return [
        Sphere(
            location=center.view(1, 1, 3),
            add_to_scene=False,
            scene=scene,
            **({} if colors is None else {"color": colors[i]}),
            **KWARGS,
        )
        for i, center in enumerate(CENTERS)
    ]


def _assert_packs_match(actual, expected):
    for attr in ("location", "basis", "color", "opacity", "glow"):
        assert torch.equal(getattr(actual, attr), getattr(expected, attr)), attr
    for attr in ("location", "color", "opacity", "glow"):
        assert torch.equal(getattr(actual.grid, attr), getattr(expected.grid, attr)), (
            f"grid.{attr}"
        )
    assert torch.equal(actual.grid.parent_batch_sizes, expected.grid.parent_batch_sizes)
    assert len(actual) == len(expected) == len(CENTERS)

    actual_primitive = actual.get_render_primitives()
    expected_primitive = expected.get_render_primitives()
    for field in ("corners", "normals", "colors", "mesh_ids"):
        a = getattr(actual_primitive, field, None)
        e = getattr(expected_primitive, field, None)
        assert (a is None) == (e is None), field
        if a is not None:
            assert torch.equal(a, e), field


def test_direct_surface_batch_matches_object_batch():
    SceneManager.reset()
    with Scene() as scene, Off(record_funcs=False, record_attr_modifications=False):
        expected = batch_mobs(_members(scene), add_to_scene=True).spawn()
        actual = Sphere.from_batches(
            CENTERS, add_to_scene=True, scene=scene, **KWARGS
        ).spawn()
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0]))
        _assert_packs_match(actual, expected)


def test_per_member_colors_match_object_batch():
    SceneManager.reset()
    colors = [RED, GREEN, BLUE, YELLOW]
    with Scene() as scene, Off(record_funcs=False, record_attr_modifications=False):
        expected = batch_mobs(_members(scene, colors), add_to_scene=True).spawn()
        actual = Sphere.from_batches(
            CENTERS,
            colors=torch.stack([torch.as_tensor(c) for c in colors]),
            add_to_scene=True,
            scene=scene,
            **KWARGS,
        ).spawn()
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0]))
        _assert_packs_match(actual, expected)


def test_rgba_colors_match_a_dot3d_batch():
    """Dot3D repositions itself with ``move_to`` after construction, so the pack
    has to survive a constructor that writes its own location -- and a point
    cloud hands in RGBA, which carries opacity but no glow channel.
    """
    from algan.mobs.point_cloud import _rgba_to_color

    SceneManager.reset()
    rgbas = torch.tensor(
        [[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0], [0, 0, 1.0, 1.0], [1.0, 1.0, 0, 0.5]]
    )
    with Scene() as scene, Off(record_funcs=False, record_attr_modifications=False):
        expected = batch_mobs(
            [
                Dot3D(
                    point=center,
                    radius=0.1,
                    resolution=None,
                    color=_rgba_to_color(rgba),
                    add_to_scene=False,
                    scene=scene,
                )
                for center, rgba in zip(CENTERS, rgbas)
            ],
            add_to_scene=True,
        ).spawn()
        actual = Dot3D.from_batches(
            CENTERS,
            radius=0.1,
            resolution=None,
            colors=rgbas,
            add_to_scene=True,
            scene=scene,
        ).spawn()
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0]))
        _assert_packs_match(actual, expected)


def test_from_batches_rejects_colors_it_cannot_honour():
    SceneManager.reset()
    with Scene() as scene, Off():
        with pytest.raises(ValueError, match="checkered_color"):
            Sphere.from_batches(
                CENTERS,
                colors=torch.zeros((4, 3)),
                checkered_color=RED,
                add_to_scene=False,
                scene=scene,
                **KWARGS,
            )
        with pytest.raises(ValueError, match="colors to match"):
            Sphere.from_batches(
                CENTERS,
                colors=torch.zeros((3, 3)),
                add_to_scene=False,
                scene=scene,
                **KWARGS,
            )
        with pytest.raises(ValueError, match="at least one centre"):
            Sphere.from_batches(
                torch.zeros((0, 3)), add_to_scene=False, scene=scene, **KWARGS
            )


def test_point_cloud_builds_one_packed_mob():
    from algan.mobs.point_cloud import PMobject

    SceneManager.reset()
    with Scene() as scene, Off():
        cloud = PMobject(points=torch.rand((16, 3)), color=BLUE, scene=scene).spawn()
        geometry = [c for c in cloud.children if isinstance(c, Sphere)]
    assert len(geometry) == 1, "a point cloud's dots must reach the scene packed"
    assert len(geometry[0]) == 16


# Indexing a pack runs through Mob.__getitem__, _set_data_sub_inds and __len__,
# which any change to the Mob base or the timeline can break -- so these are in
# the fast suite while the equivalence tests above, which only fail when
# surface.py itself changes, are not.


def _packs(scene):
    """The same four spheres packed both ways, for tests that must cover both."""
    return {
        "from_batches": Sphere.from_batches(
            CENTERS, add_to_scene=True, scene=scene, **KWARGS
        ),
        "batch_mobs": batch_mobs(_members(scene), add_to_scene=True),
    }


@pytest.mark.fast
@pytest.mark.parametrize("how", ["from_batches", "batch_mobs"])
def test_indexing_a_pack_moves_only_that_member(how):
    """REGRESSION. ``batch_mobs`` summed its members into a single
    ``parent_batch_sizes`` entry and never set ``singleton_batch_indexing``, so
    ``len(pack)`` was 1 and ``pack[1]`` raised IndexError -- while every other
    packer indexed fine.
    """
    SceneManager.reset()
    with Scene() as scene, Off():
        pack = _packs(scene)[how].spawn()
        assert len(pack) == len(CENTERS)
        pack[2].move(UP * 3)

        centers = pack.location.view(-1, 3)
        assert centers[2, 1] == pytest.approx(3.0)
        for other in (0, 1, 3):
            assert centers[other, 1] == pytest.approx(0.0)

        # The move has to reach the member's grid rows too, not just its centre.
        grids = pack.grid.location.view(len(CENTERS), -1, 3)
        assert bool((grids[2][:, 1] > 2.5).all())
        assert not bool((grids[1][:, 1] > 2.5).any())


@pytest.mark.fast
@pytest.mark.parametrize("how", ["from_batches", "batch_mobs"])
def test_a_pack_slices_and_iterates_as_views(how):
    SceneManager.reset()
    with Scene() as scene, Off():
        pack = _packs(scene)[how].spawn()
        assert pack[1:3].location.shape[-2] == 2
        assert pack[-1].location.shape[-2] == 1

        views = BatchedMobViewSequence(pack, len(CENTERS))
        assert len(views) == len(CENTERS)
        # Views are cached, so indexing twice gives one object rather than
        # rebuilding the Python graph packing exists to avoid.
        assert views[1] is views[1]
        # A view shares the pack's id, and so its timeline rows and lifespan.
        assert views[1].id == pack.id


@pytest.mark.fast
@pytest.mark.parametrize("how", ["from_batches", "batch_mobs"])
def test_transforming_a_whole_pack_reaches_every_component(how):
    """A pack's change carries one row per member while the recursive write
    covers every row of every component, so it has to be spread over them
    first -- what ``parent_batch_sizes`` documents itself as being for, and
    what was never wired up: this raised a shape error for every pack,
    ``Text``'s included.
    """
    SceneManager.reset()
    with Scene() as scene, Off():
        pack = _packs(scene)[how].spawn()
        pack.move(UP)

        assert torch.allclose(pack.location.view(-1, 3)[:, 1], torch.ones(len(CENTERS)))
        grids = pack.grid.location.view(len(CENTERS), -1, 3)
        assert bool((grids[:, :, 1] > 0.5).all()), "the vertex grids must move too"

        # The rest of the transform API takes different paths into the same
        # subtree write; none of them should raise on a pack.
        pack.rotate(45, OUT)
        pack.scale(2)
        pack.color = RED
        pack.opacity = 0.5


@pytest.mark.fast
@pytest.mark.parametrize("how", ["from_batches", "batch_mobs"])
def test_a_per_member_write_lands_on_the_right_member(how):
    """REGRESSION, and a silent one. A subtree is addressed in buffer order,
    which is not descendant order, so distributing a per-member value by
    concatenating in descendant order lines up in count and hands every member
    a neighbour's value. Only distinct per-member values catch it -- a uniform
    move looks perfect either way.

    Both packers are covered because only one of them sees it: ``batch_mobs``
    allocates the pack's rows before its components', so its two orders agree
    by luck, while ``from_batches`` builds the grid first and they do not.
    """
    SceneManager.reset()
    heights = torch.tensor([5.0, 6.0, 7.0, 8.0])
    with Scene() as scene, Off():
        pack = _packs(scene)[how].spawn()
        target = torch.stack((CENTERS[:, 0], heights, CENTERS[:, 2]), dim=-1).unsqueeze(
            0
        )
        pack.set_location(target)

        assert torch.allclose(pack.location.view(-1, 3)[:, 1], heights)
        # And the grids follow their own member, not a neighbour's: each sits
        # within its radius of the height that member was sent to.
        grids = pack.grid.location.view(len(CENTERS), -1, 3)
        for member, height in enumerate(heights):
            assert grids[member][:, 1].min() > height - 2 * KWARGS["radius"]
            assert grids[member][:, 1].max() < height + 2 * KWARGS["radius"]


@pytest.mark.fast
def test_a_packed_text_batch_transforms_too():
    """The same fix, on the pack the engine has always shipped: a Text's glyph
    batch. It raised the same shape error, which is why a Text is moved through
    its unbatched container.
    """
    from algan.mobs.text import Text

    SceneManager.reset()
    with Scene() as scene, Off():
        text = Text("Hi", scene=scene).spawn()
        batch = text._character_batch
        before = batch.control_points.location.clone()
        batch.move(UP)
        assert torch.allclose(
            batch.control_points.location - before,
            torch.tensor([0.0, 1.0, 0.0]),
        )
