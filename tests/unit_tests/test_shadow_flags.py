"""Per-mob shadow flags reach the renderer, and reach it as nothing by default.

``Mob.casts_shadows`` / ``Mob.receives_shadows`` are plain geometry
declarations in the style of ``two_sided`` and ``closed_shell``: set before
spawn, read once when the render primitive is built, carried to the renderer in
two words it already loads -- a bit in the BVH leaf word for casting, a material
block slot for receiving.

These are tensor-level assertions on the primitive build and the tree build; the
pixel-level proof that the flags change the right pixels and nothing else lives
in ``benchmarks/_shadow_flags_check.py``, which renders each flag against an
independently-rendered oracle (the same scene with the mob deleted, and the same
scene with shadows globally off).

Feature tests of the mob/renderer boundary, not of anything the timeline or the
Scene can break, so they stay out of the fast suite.
"""

from __future__ import annotations

import pytest
import torch

from algan import Cube, Group, Prism, Sphere, Square, Text
from algan.rendering.raytracing.primitives import shadow_cast_flag
from algan.rendering.raytracing.refit_bvh import (
    LINK_INVALID,
    LINK_NOCAST_BIT,
    LINK_OPAQUE_BIT,
    LINK_PRIM_MASK,
    build_refit_bvh,
)
from algan.rendering.raytracing.settings import _MAT_DEFAULTS
from algan.rendering.raytracing.shading_taichi import (
    _MAT_NO_SHADOW_RECEIVE,
    MAT_W,
)
from algan.rendering.raytracing.stbvh import LEAF_NOCAST_BIT, build_stbvh

# --------------------------------------------------------------------------
# The declaration reaches the primitive, through the hierarchy.
# --------------------------------------------------------------------------


def _primitives(mob):
    """Every render primitive a spawned mob's subtree contributes."""
    out = []
    for node in mob.get_descendants(include_self=True):
        if hasattr(node, "get_render_primitives"):
            got = node.get_render_primitives()
            out.extend(got if isinstance(got, (list, tuple)) else [got])
    return [p for p in out if p is not None]


def test_defaults_declare_casting_and_receiving(fresh_scene):
    """An untouched mob casts and receives, and says so in the packed value."""
    cube = Cube(side_length=1.0, fill_opacity=1).spawn(animate=False)
    for p in _primitives(cube):
        assert float(p.no_shadow_cast.max()) == 0.0
        assert float(p.no_shadow_receive.max()) == 0.0


@pytest.mark.parametrize("flag", ["casts_shadows", "receives_shadows"])
def test_flag_set_on_an_aggregate_reaches_the_geometry(fresh_scene, flag):
    """The Mob a user sets the flag on is not the Mob that holds the triangles.

    A ``Cube`` is a ``Polyhedron`` whose FACES build the primitives, so reading
    the flag off the mob that builds them would silently ignore
    ``cube.casts_shadows = False``. It did, before ``resolved_shadow_flags``.
    """
    cube = Cube(side_length=1.0, fill_opacity=1)
    setattr(cube, flag, False)
    cube.spawn(animate=False)
    attr = "no_shadow_cast" if flag == "casts_shadows" else "no_shadow_receive"
    prims = _primitives(cube)
    assert prims, "the cube built no render primitives"
    for p in prims:
        assert float(getattr(p, attr).min()) == 1.0, (
            f"{flag} did not reach {type(p).__name__}"
        )


def test_flag_on_a_group_applies_to_the_whole_subtree(fresh_scene):
    """``group.casts_shadows = False`` means the group, not just the node."""
    a = Cube(side_length=1.0, fill_opacity=1)
    b = Prism(dimensions=(2, 0.5, 2), fill_opacity=1)
    group = Group(a, b)
    group.casts_shadows = False
    group.spawn(animate=False)
    for member in (a, b):
        prims = _primitives(member)
        assert prims
        for p in prims:
            assert float(p.no_shadow_cast.min()) == 1.0


def test_a_sibling_is_unaffected(fresh_scene):
    """Resolution walks ANCESTORS, not the whole scene."""
    quiet = Cube(side_length=1.0, fill_opacity=1)
    quiet.casts_shadows = False
    loud = Cube(side_length=1.0, fill_opacity=1)
    quiet.spawn(animate=False)
    loud.spawn(animate=False)
    assert all(float(p.no_shadow_cast.min()) == 1.0 for p in _primitives(quiet))
    assert all(float(p.no_shadow_cast.max()) == 0.0 for p in _primitives(loud))


def test_a_circuit_declares_casting_and_ignores_receiving(fresh_scene):
    """2-D geometry casts but cannot receive -- the renderer draws it unlit.

    The circuit primitive accepts ``receives`` so a mob can declare both without
    knowing which primitive kind it built, and drops it on the floor. This test
    pins that asymmetry so it cannot be "fixed" into a slot that nothing reads.
    """
    square = Square()
    square.casts_shadows = False
    square.receives_shadows = False
    square.spawn(animate=False)
    prims = _primitives(square)
    assert prims
    for p in prims:
        assert float(p.no_shadow_cast.min()) == 1.0
        assert not hasattr(p, "no_shadow_receive")


def test_text_glyphs_inherit_from_the_text_mob(fresh_scene):
    """A packed mob's members are built from its own rows; the flag still lands."""
    text = Text("ab", font_size=40)
    text.casts_shadows = False
    text.spawn(animate=False)
    prims = _primitives(text)
    assert prims
    assert all(float(p.no_shadow_cast.min()) == 1.0 for p in prims)


def test_diced_surfaces_with_different_flags_do_not_merge(fresh_scene):
    """The invariant the leaf word depends on: one merged column, one mob's flag.

    A logical-PN collection dices adaptively per frame, so a column can host a
    patch of one mob in one frame and another mob's in the next. The leaf word
    carries ONE bit per column for the whole batch, so merging a non-caster with
    a caster made the caster's shadow partly disappear -- measured as a bite out
    of a sphere's shadow ellipse on every frame
    (``benchmarks/_shadow_flags_mixed_dice_check.py``). Keeping them in separate
    merge groups is what restores the assumption.
    """
    quiet = Sphere(radius=1.0)
    quiet.casts_shadows = False
    loud = Sphere(radius=1.0)
    quiet.spawn(animate=False)
    loud.spawn(animate=False)
    quiet_ids = {p.get_batch_identifier() for p in _primitives(quiet)}
    loud_ids = {p.get_batch_identifier() for p in _primitives(loud)}
    assert quiet_ids
    assert loud_ids
    assert not (quiet_ids & loud_ids), (
        "a non-casting surface shares a merge group with a casting one"
    )


def test_same_flag_surfaces_still_merge(fresh_scene):
    """...and the split costs nothing when the flags agree, which is the norm."""
    a = Sphere(radius=1.0)
    b = Sphere(radius=1.0)
    a.spawn(animate=False)
    b.spawn(animate=False)
    assert {p.get_batch_identifier() for p in _primitives(a)} == {
        p.get_batch_identifier() for p in _primitives(b)
    }


def test_a_morph_endpoint_adopts_the_flags(fresh_scene):
    """``become()`` takes plain geometry metadata from its target.

    Neither flag lives on the timeline, so the same-kind morph path -- which
    copies the intersection of the two Mobs' animatable attrs -- would carry
    neither, and a morph would end with the target's geometry wearing the
    source's shadow behaviour. ``_MORPH_ADOPTED_ATTRS`` is what stops that,
    beside ``two_sided`` and ``closed_shell``.
    """
    source = Cube(side_length=1.0, fill_opacity=1).spawn(animate=False)
    target = Cube(side_length=1.5, fill_opacity=1)
    target.casts_shadows = False
    target.receives_shadows = False
    assert (source.casts_shadows, source.receives_shadows) == (True, True)
    source._adopt_structural_attrs(target)
    assert (source.casts_shadows, source.receives_shadows) == (False, False)


def test_the_kill_switch_stops_the_flag_reaching_the_primitive(
    fresh_scene, monkeypatch
):
    """``ALGAN_PER_MOB_SHADOW_FLAGS=0`` leaves the packed value at its default.

    The switch is host-side by design -- nothing is stamped into the leaf word
    and the material slot keeps its 0.0 -- so no kernel variant changes and
    there is no ti.static gate to go stale mid-process. Rendered proof that the
    frame is then byte-identical lives in the acceptance harness; this pins the
    mechanism.
    """
    from algan.rendering.raytracing import settings as rt_settings

    monkeypatch.setattr(rt_settings, "PER_MOB_SHADOW_FLAGS", False)
    cube = Cube(side_length=1.0, fill_opacity=1)
    cube.casts_shadows = False
    cube.spawn(animate=False)
    for p in _primitives(cube):
        # The declaration is still made -- the switch acts where it is CONSUMED.
        assert float(p.no_shadow_cast.min()) == 1.0
        assert bool(shadow_cast_flag(p.no_shadow_cast, 1, torch.device("cpu")).all())


# --------------------------------------------------------------------------
# The declaration reaches the two words the traversal already loads.
# --------------------------------------------------------------------------


def test_receive_slot_is_inside_the_material_block():
    """Slot 33 has to be addressable, and the defaults row has to reach it."""
    assert _MAT_NO_SHADOW_RECEIVE < MAT_W
    assert len(_MAT_DEFAULTS) == MAT_W


def test_receive_slot_defaults_to_receiving():
    """THE PADDING RULE for the material block: a 0.0 in any slot must mean the
    behaviour that existed before it. A custom fragment pipeline's block is a
    different layout zero-padded to this width, so a zero read out of slot 33
    has to be "this surface is darkened by shadows" -- which is why the flag is
    stored negated. Get this backwards and every custom-pipeline mob in a mixed
    scene silently stops receiving shadows.
    """
    assert _MAT_DEFAULTS[_MAT_NO_SHADOW_RECEIVE] == 0.0


def _link_words(bvh):
    """The refit tree's per-(frame, node, child) link words, unpacked.

    They ride lane 6 of each block as a bit-cast int32, or split across lanes
    6/7 as int16 halves when the blocks are float16 (``BVH_BLOCK_F16``, the
    default). Reading them back is the only way to assert on the tree the
    kernel actually walks.
    """
    blocks = bvh.blocks
    if blocks.dtype == torch.float32:
        return blocks[:, 6].contiguous().view(torch.int32).flatten()
    halves = blocks.view(torch.int16).to(torch.int64)
    lo = halves[:, 6] & 0xFFFF
    hi = halves[:, 7] & 0xFFFF
    words = (lo | (hi << 16)).flatten()
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _bits(bvh):
    """Blocks as raw bit patterns -- an int view, because a float compare of
    two identical trees fails on the NaN payloads ``LINK_INVALID`` bit-casts to.
    """
    return bvh.blocks.view(
        torch.int16 if bvh.blocks.dtype == torch.float16 else torch.int32
    )


def _bounds(n):
    lo = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(1, n, 3)
    return lo.contiguous(), (lo + 0.5).contiguous()


@pytest.mark.parametrize("builder", ["stbvh", "refit"])
def test_leaf_word_is_untouched_when_everything_casts(builder):
    """The whole byte-identity argument: no non-caster, no changed bit."""
    lo, hi = _bounds(8)
    casts = torch.ones((1, 8), dtype=torch.bool)
    if builder == "stbvh":
        a = build_stbvh(lo, hi, num_frames=1, tightness=1.0)
        b = build_stbvh(lo, hi, num_frames=1, tightness=1.0, casts=casts)
        assert torch.equal(a.leaf_tspan, b.leaf_tspan)
        assert int((b.leaf_tspan & LEAF_NOCAST_BIT).sum()) == 0
    else:
        a = build_refit_bvh(lo, hi, num_frames=1)
        b = build_refit_bvh(lo, hi, num_frames=1, casts=casts)
        assert torch.equal(_bits(a), _bits(b))


def test_stbvh_stamps_only_the_non_casting_leaves():
    lo, hi = _bounds(8)
    casts = torch.ones((1, 8), dtype=torch.bool)
    casts[0, 3] = False
    casts[0, 5] = False
    bvh = build_stbvh(lo, hi, num_frames=1, tightness=1.0, casts=casts)
    stamped = bvh.leaf_prim[(bvh.leaf_tspan & LEAF_NOCAST_BIT) != 0]
    assert sorted(int(x) for x in stamped) == [3, 5]
    # The frame interval the flag shares its word with still reads back intact.
    live = bvh.leaf_prim >= 0
    assert int((bvh.leaf_tspan[live] & 0x7FFF).max()) == 0
    assert int(((bvh.leaf_tspan[live] >> 16) & 0x7FFF).max()) == 0


def test_refit_link_word_keeps_prim_opacity_and_flag_separable():
    """Narrowing the prim mask to 29 bits must not disturb its neighbours."""
    lo, hi = _bounds(8)
    casts = torch.ones((1, 8), dtype=torch.bool)
    casts[0, 2] = False
    opaque = torch.ones((1, 8), dtype=torch.bool)
    bvh = build_refit_bvh(lo, hi, num_frames=1, opaque=opaque, casts=casts)
    words = _link_words(bvh)
    # LINK_LEAF_BIT is the sign bit -- but so is every bit of LINK_INVALID,
    # which a dead slot carries and which would otherwise read as a flagged
    # leaf holding primitive LINK_PRIM_MASK.
    leaves = words[(words < 0) & (words != LINK_INVALID)]
    assert leaves.numel()
    prims = leaves & LINK_PRIM_MASK
    flagged = prims[(leaves & LINK_NOCAST_BIT) != 0]
    assert {int(x) for x in flagged} == {2}
    # Every leaf was declared opaque, and the flag did not eat that bit.
    assert int((leaves & LINK_OPAQUE_BIT != 0).sum()) == leaves.numel()
    assert int(prims.max()) <= 7


def test_prim_index_range_is_enforced_at_the_narrower_bound():
    assert LINK_PRIM_MASK == (1 << 29) - 1
    assert LINK_NOCAST_BIT == 1 << 29
    assert LINK_OPAQUE_BIT == 1 << 30
    # The three flags and the index may not overlap.
    assert LINK_PRIM_MASK & LINK_NOCAST_BIT == 0
    assert LINK_NOCAST_BIT & LINK_OPAQUE_BIT == 0


def test_leaf_nocast_bit_sits_above_the_frame_interval():
    """t0 occupies bits 0-14 and t1 bits 16-30, so 15 is the only free one."""
    assert LEAF_NOCAST_BIT == 1 << 15
    assert LEAF_NOCAST_BIT & 0x7FFF == 0
    assert LEAF_NOCAST_BIT & (0x7FFF << 16) == 0


def test_shadow_cast_flag_is_conservative_over_frames():
    """Declining on any frame declines on all of them (the flag is fixed)."""
    per_corner = torch.zeros((3, 4, 3, 1))
    per_corner[1, 2] = 1.0  # primitive 2 declines, on frame 1 only
    out = shadow_cast_flag(per_corner, 4, torch.device("cpu"))
    assert out.shape == (1, 4)
    assert [bool(x) for x in out[0]] == [True, True, False, True]


def test_shadow_cast_flag_defaults_to_casting():
    out = shadow_cast_flag(None, 5, torch.device("cpu"))
    assert out.shape == (1, 5)
    assert bool(out.all())
