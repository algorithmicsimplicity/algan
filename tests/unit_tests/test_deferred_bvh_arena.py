"""A BVH built after the scene upload has to end up in the scene's arena.

``settings.bvh_defer`` lets a batch skip its STBVH build at merge time and
build the trees mid-render, once something actually needs one
(``scene_builder.build_deferred_bvhs``). By then the merged scene has been
copied into one ``ManualMemory`` allocation per dtype, and the widest kernels
bind their scene-indexed tables -- the BVH walk tables among them -- as offsets
into that single buffer (`arena_args_taichi`). A tree left in the ordinary
torch allocation the builder returns therefore makes every one of those
launches raise ``ArenaBindingError``, naming whichever table it reached first.
"""

import pytest
import torch

from algan.rendering.raytracing.arena_args_taichi import ArenaBindingError, pack
from algan.rendering.raytracing.scene_builder import (
    build_deferred_bvhs,
    copy_merged_scene_to_arena,
    rehome_deferred_bvhs_to_arena,
)
from algan.rendering.raytracing.stbvh import STBVH, bvh_arity
from algan.utils.memory_utils import ManualMemory

#: The i32 half of what ``wavefront_traverse_events`` binds: the triangle
#: tree's three walk tables beside a table the merged scene owns. Enough to
#: reproduce the launch-side check exactly -- ``edge_accel`` fixes the
#: allocation every other i32 argument has to be a view of.
_I32_SPEC = (
    ("t_node_miss", "i32", 1),
    ("t_leaf_prim", "i32", 1),
    ("t_leaf_tspan", "i32", 1),
    ("edge_accel", "i32", 1),
)


def _bvh(offset=0):
    num_leaves = bvh_arity
    num_nodes = (bvh_arity * num_leaves - 1) // (bvh_arity - 1)
    first_leaf = num_nodes - num_leaves
    nodes = torch.arange(num_nodes * 8, dtype=torch.float32).reshape(num_nodes, 8)
    blocks = torch.arange(first_leaf * 8 * bvh_arity, dtype=torch.float16).reshape(
        first_leaf, 8, bvh_arity
    )
    node_miss = torch.arange(num_nodes, dtype=torch.int32) + offset
    leaf_prim = torch.arange(num_leaves, dtype=torch.int32) + offset
    leaf_tspan = (torch.arange(num_leaves, dtype=torch.int32) + offset) << 16
    return STBVH.from_prebuilt(nodes, node_miss, leaf_prim, leaf_tspan, blocks)


def _uploaded_scene():
    """A scene in a CPU arena, holding placeholder trees as a deferral does."""
    # CPU availability is a fixed 2 GiB budget in ManualMemory; this reserves
    # roughly 2 MiB without relying on host free-memory telemetry.
    memory = ManualMemory(0.001, device=torch.device("cpu"), managed=True)
    placeholder = _bvh()
    scene = {
        "edge_accel": torch.arange(8, dtype=torch.int32),
        "tri_bvh": placeholder,
        "tri_opaque_bvh": placeholder,
    }
    return memory, copy_merged_scene_to_arena(scene, memory, persist=True)


def _pack_i32(scene):
    tree = scene["tri_bvh"]
    return pack(
        _I32_SPEC,
        [tree.node_miss, tree.leaf_prim, tree.leaf_tspan, scene["edge_accel"]],
    )


def test_rehome_puts_an_on_demand_tree_in_the_uploaded_scenes_allocation():
    memory, scene = _uploaded_scene()
    _pack_i32(scene)  # the uploaded placeholders already satisfy the convention

    # What build_deferred_bvhs produces: real trees in ordinary torch
    # allocations, made long after the scene upload.
    built = _bvh(offset=1000)
    scene["tri_bvh"] = built
    scene["tri_opaque_bvh"] = built
    with pytest.raises(ArenaBindingError, match="t_leaf_prim"):
        _pack_i32(scene)

    rehome_deferred_bvhs_to_arena(scene, memory)

    _pack_i32(scene)
    arena = memory.data.untyped_storage()._cdata
    for field in ("nodes", "blocks", "node_miss", "leaf_prim", "leaf_tspan"):
        moved = getattr(scene["tri_bvh"], field)
        assert moved.untyped_storage()._cdata == arena
        torch.testing.assert_close(moved, getattr(built, field))
    # An opaque tree aliased to its main tree must stay one tree, not become a
    # second copy: the kernels compare the two by identity nowhere, but the
    # arena would otherwise pay for it twice.
    assert scene["tri_opaque_bvh"] is scene["tri_bvh"]
    assert type(scene["tri_bvh"]) is STBVH
    assert scene["tri_bvh"].first_leaf == built.first_leaf


def test_rehome_is_idempotent():
    memory, scene = _uploaded_scene()
    scene["tri_bvh"] = scene["tri_opaque_bvh"] = _bvh(offset=7)
    rehome_deferred_bvhs_to_arena(scene, memory)
    settled = memory.get_pointers()
    tree = scene["tri_bvh"]

    rehome_deferred_bvhs_to_arena(scene, memory)

    assert memory.get_pointers() == settled, "a second re-home copied again"
    assert scene["tri_bvh"] is tree


def test_build_deferred_bvhs_rehomes_even_when_the_build_already_ran():
    """The out-of-memory retry path re-enters with ``bvh_deferred`` cleared.

    The build succeeded, the flag went down, and only the arena copy after it
    ran out of room. The retry has to finish the move rather than return early
    and hand the tracer a tree the kernels cannot bind.
    """
    memory, scene = _uploaded_scene()
    scene["bvh_deferred"] = False
    scene["tri_bvh"] = scene["tri_opaque_bvh"] = _bvh(offset=3)
    with pytest.raises(ArenaBindingError):
        _pack_i32(scene)

    build_deferred_bvhs(scene, memory)

    _pack_i32(scene)


def test_build_deferred_bvhs_without_an_arena_leaves_the_trees_alone():
    """``memory=None`` means no arena was involved, so there is nothing to move."""
    scene = {"bvh_deferred": False, "tri_bvh": _bvh()}
    tree = scene["tri_bvh"]

    build_deferred_bvhs(scene)

    assert scene["tri_bvh"] is tree
