from __future__ import annotations

import warnings
from typing import NamedTuple

import torch
import torch.nn.functional as F

from algan.constants.color import Color
from algan.environment import env_float
from algan.rendering.logical_pn import (
    OPPOSITE_EDGE,
    dice_pattern,
    dice_triangle_count,
    evaluate_cubic_curve,
    evaluate_logical_pn,
    evaluate_logical_pn_normals,
    interpolate_patch_vertex_attribute,
    logical_pn_control_points,
    logical_pn_edge_control_points,
    logical_pn_normal_control_points,
    mean_patch_edge_length,
    normalize_pixel_tolerance,
    snap_boundary_values,
)
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
    batch_arange,
)
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.rendering.raytracing.logical_pn_taichi import (
    bezier_chord_hull_error,
    pn_edge_chord_error,
    pn_patch_flatness_error,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
    MIN_ALPHA,
)
from algan.rendering.raytracing.settings import (
    _MAT_DEFAULTS,
    _MAT_SLOTS,
    _shader_is_core,
    _shader_material_id,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_NO_SHADOW_RECEIVE,
    _MAT_ONE_SIDED,
    MAT_W,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _unify_time
from algan.rendering.shaders.material_shaders import SHADER_FIXED_PARAM_COUNT
from algan.settings import SETTINGS
from algan.utils.memory_utils import empty_cache
from algan.utils.tensor_utils import broadcast_all, cast_to_tensor, unsquish

# rt_settings values are mutable module globals (set_samples_per_pixel etc.);
# read them live as rt_settings.X -- importing them by value freezes them at
# import time, before user code runs.
rt_settings = SETTINGS.raytracing
from algan.rendering.raytracing.settings import *  # noqa: F403 -- re-export for callers of this module
from algan.settings.kernel_settings import KERNEL_REGISTRY

_SAMPLE_TENSOR_CACHE = {}


def _sample_tensor(values, device, dtype):
    """Cached device tensor for a constant tuple of sample weights.

    The level searches ask for these once per level per chunk; rebuilding them
    with ``torch.tensor`` each time is a host-to-device copy (and a sync) per
    call, which on a mesh that resolves immediately is most of the search.
    """
    key = (values, device.type, device.index, dtype)
    cached = _SAMPLE_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, device=device, dtype=dtype)
        _SAMPLE_TENSOR_CACHE[key] = cached
    return cached


class _PatchChunk(NamedTuple):
    """One chunk of the dice's ``(frame, patch)`` work list, deduped by patch.

    The dice walks its selected pairs in PATCH-major order, so the frames that
    dice a given patch at a given level are consecutive.  ``unique_patches``
    then names each distinct patch once and ``inverse`` maps every row of the
    chunk back to its entry, which is all a frame-invariant source needs to be
    evaluated once and fanned out.  Both are ``None`` when nothing in this dice
    is frame invariant, and every method falls back to the per-row path.
    """

    patches: torch.Tensor  # [K] long
    frames: torch.Tensor  # [K] long
    unique_patches: torch.Tensor | None  # [U] long, ascending
    inverse: torch.Tensor | None  # [K] long into unique_patches

    @classmethod
    def of(cls, patches, frames, dedup):
        if not dedup:
            return cls(patches, frames, None, None)
        unique_patches, inverse = torch.unique_consecutive(patches, return_inverse=True)
        return cls(patches, frames, unique_patches, inverse)

    def rows_of(self, source, static):
        """This chunk's rows of ``source``, and whether they were deduped."""
        if static and self.unique_patches is not None:
            return source[0].index_select(0, self.unique_patches), True
        return source[self.frames, self.patches], False

    def fan_out(self, values, deduped):
        """Expand per-distinct-patch ``values`` back to one row per pair."""
        return values.index_select(0, self.inverse) if deduped else values

    def diced_attribute(self, source, vertex_uv, triangle_indices):
        """Interpolate a per-corner attribute onto this chunk's triangle soup.

        Interpolating on the shared subdivision vertices and gathering through
        ``triangle_indices`` is the same arithmetic on the same values as
        interpolating at every microtriangle corner -- the corners *are* those
        vertices -- over a sixth as many of them.

        Deliberately NOT deduped, even where the attribute is frame invariant
        and the rows are there for the taking: a barycentric blend of three
        corner values is so cheap that fanning the result back out costs more
        than evaluating it per row (measured 0.86x on a deforming mesh whose
        colours were static). Only the patch evaluation is expensive enough for
        the trade to pay.
        """
        values = source[self.frames, self.patches]
        values = interpolate_patch_vertex_attribute(values, vertex_uv)
        return values[:, triangle_indices]


class _PNCriterionInputs(NamedTuple):
    """Kernel-ready inputs shared by both logical PN level searches.

    Built once per dice (the searches call their criteria once per level) and
    passed down; ``None`` in its place means the searches stay on torch.
    """

    control_points: torch.Tensor  # [Tc, P, 10, 3]
    control_stride: int  # 0 when one control net serves every frame
    edge_controls: torch.Tensor  # [Te, P, 3, 4, 3]
    edge_stride: int
    cam_origin: torch.Tensor  # [T, 3]
    screen_point: torch.Tensor  # [T, 3]
    screen_basis: torch.Tensor  # [T, 3, 3]
    front_sign: torch.Tensor  # [T]
    slack: torch.Tensor  # [T], world-space accuracy of the surface; zeros = off


def _frame_broadcast_base(value):
    """Return ``(contiguous base, frame stride)`` of a per-frame tensor.

    ``_expand_frames`` hands out stride-0 views for geometry every frame of the
    batch shares -- a static mesh's control net is one copy however long the
    batch is. A Taichi ndarray needs real memory, and materializing that
    expansion would allocate the whole batch's worth of control points to say
    the same thing many times over, so pass the one real frame down and let the
    kernel multiply its frame index by the stride instead.
    """
    if value.shape[0] > 1 and value.stride(0) != 0:
        return value.contiguous(), 1
    return value[:1].contiguous(), 0


def _scatter_diced_rows(output, values, targets):
    """Write diced microtriangle rows into a ``[T, M, ...]`` output.

    ``targets`` indexes the flattened ``(frame, column)`` pair, so the whole
    write is one ``index_copy_`` over ``[T * M, ...]``. Every target row is
    written exactly once -- the patches of a frame occupy disjoint column spans
    -- so the copy needs no accumulation and its order does not matter.
    """
    trailing = output.shape[2:]
    output.view(-1, *trailing).index_copy_(0, targets, values.reshape(-1, *trailing))


def _bezier_criterion_inputs(corners, cam_o, sp, sb):
    """Kernel inputs for the bezier chord-count search, or ``None`` for torch.

    Same gate as :func:`_pn_criterion_inputs` -- the two searches are the same
    computation shape and share one toggle.
    """
    if not rt_settings.pn_criterion_kernel_active():
        return None
    tensors = (corners, cam_o, sp, sb)
    if any(
        value.device.type != "cuda" or value.dtype != torch.float32 for value in tensors
    ):
        return None
    base, stride = _frame_broadcast_base(corners)
    return (base, stride, cam_o.contiguous(), sp.contiguous(), sb.contiguous())


def _pn_criterion_inputs(
    control_points, edge_controls, cam_o, sp, sb, front_sign, slack=None
):
    """Kernel inputs for the level searches, or ``None`` to stay on torch.

    The kernels only run where projection already runs on the render thread
    against CUDA tensors (see ``settings.pn_criterion_kernel_active``); against
    CPU tensors Taichi would stage every argument through VRAM, which is a
    regression, not an optimization.
    """
    if not rt_settings.pn_criterion_kernel_active():
        return None
    tensors = (control_points, edge_controls, cam_o, sp, sb, front_sign)
    if any(
        value.device.type != "cuda" or value.dtype != torch.float32 for value in tensors
    ):
        return None
    control_base, control_stride = _frame_broadcast_base(control_points)
    edge_base, edge_stride = _frame_broadcast_base(edge_controls)
    # The kernels take the slack unconditionally; zeros are the "measure against
    # the PN patch exactly" case, which keeps one kernel signature instead of a
    # second compiled variant per template gate.
    if slack is None:
        slack = torch.zeros_like(front_sign)
    return _PNCriterionInputs(
        control_base,
        control_stride,
        edge_base,
        edge_stride,
        cam_o.contiguous(),
        sp.contiguous(),
        sb.contiguous(),
        front_sign.contiguous(),
        slack.to(front_sign.dtype).contiguous(),
    )


# Inverse of ``OPPOSITE_EDGE``: the corner a dice fans from when its rows are
# to run parallel to a given edge.
_APEX_OF_EDGE = torch.tensor([OPPOSITE_EDGE.index(edge) for edge in range(3)])


def _mesh_ids_from_collection(members, counts):
    """Resolve a triangle collection's per-triangle SURFACE ids.

    Returns ``(ids, n)`` where ``ids`` is an int32 ``[Ntri]`` tensor of
    collection-local surface indices and ``n`` the number of distinct ones, or
    ``(None, None)`` when no member declares identity -- in which case the
    caller's per-member ``counts`` already say it and nothing changes.

    Two declarations, both optional attributes a mob stamps on the primitive it
    builds:

    ``mesh_key``
        An opaque hashable. Consecutive members carrying the same key are ONE
        surface. This is what makes a ``Polyhedron`` a single mesh: it hands the
        batcher one member per triangle, and without a key a Cube's twelve
        coplanar triangles are twelve surfaces that can never share a run.
    ``mesh_ids``
        Per-triangle local shell indices for a member that is SEVERAL surfaces --
        every sphere of a packed grid, every disconnected part of an imported
        mesh. Needs no contiguity: the ids are threaded per triangle, not
        summarized as counts.

    ``mesh_ids`` wins where a member declares both. Keys are matched only
    against the immediately preceding member, so identity never depends on how
    far apart two members drifted in the concatenation.
    """
    if not any(
        getattr(m, "mesh_key", None) is not None
        or getattr(m, "mesh_ids", None) is not None
        for m in members
    ):
        return None, None

    device = members[0].corners.device
    blocks = []
    next_id = 0
    prev_key = None
    for i, member in enumerate(members):
        n_tri = int(member.corners.shape[1]) // 3 if counts is None else counts[i]
        local = getattr(member, "mesh_ids", None)
        if local is not None:
            local = torch.as_tensor(local, device=device).reshape(-1).to(torch.int32)
            if local.shape[0] != n_tri:
                raise ValueError(
                    f"mesh_ids has {local.shape[0]} entries for a member of "
                    f"{n_tri} triangles"
                )
            # Renumber into the collection's namespace, preserving distinctness.
            uniq, inverse = torch.unique(local, return_inverse=True)
            blocks.append(inverse.to(torch.int32) + next_id)
            next_id += int(uniq.shape[0])
            prev_key = None
            continue
        key = getattr(member, "mesh_key", None)
        merges = key is not None and prev_key is not None and key == prev_key
        if merges:
            surface = next_id - 1
        else:
            surface = next_id
            next_id += 1
        blocks.append(torch.full((n_tri,), surface, dtype=torch.int32, device=device))
        prev_key = key
    return torch.cat(blocks).contiguous(), next_id


def _declares_no_shadow_cast(primitive):
    """Whether this primitive declined to cast, as one bool for the whole mob.

    Only for keying a merge group (``get_batch_identifier``), never for what the
    renderer reads -- that is per primitive and goes through
    :func:`shadow_cast_flag`. A primitive's declaration is one constant, so
    reducing it to a bool here loses nothing.
    """
    value = getattr(primitive, "no_shadow_cast", None)
    return bool(value is not None and bool((value > 0.5).any()))


def shadow_cast_flag(no_shadow_cast, num_prims, device):
    """Per-primitive "this geometry blocks light" flag, ``[1, N]`` bool.

    Reduces the packed per-corner ``no_shadow_cast`` declaration
    (:meth:`RayTracedTrianglePrimitive.declare_shadow_flags`) to one bool per
    primitive, which is the granularity the BVH leaf word carries: a leaf holds
    a whole primitive, so a flag that varied across a triangle's corners could
    not be represented there. Reads CORNER 0 and reduces amax over what is left
    and over frames, so a primitive that declines to cast at ANY authored moment
    declines everywhere -- the flag is documented as fixed for the render, and
    reducing this way makes that true of the packed value rather than merely
    asserted of the API. Corner 0 stands for the whole triangle because every
    producer writes a corner-uniform constant (``declare_shadow_flags`` fills
    with ``full_like``, and the PN dice interpolates a constant to itself); a
    producer that ever wrote a per-corner value would want this to reduce over
    corners too, which it does not do today.

    Reducing over FRAMES is exact only because a merged primitive column means
    one mob for the whole batch. That is an enforced invariant rather than an
    accident: a diced collection would otherwise let a column change hands
    between frames, which is why
    ``LogicalPNTrianglePrimitive.get_batch_identifier`` splits merge groups by
    this flag.

    ``None`` (nothing declared -- a primitive built before the flags, or one
    whose collection merge filled the column with the 0.0 default) is every
    primitive casting, which is what they all did before, and so is the whole
    feature switched off (``PER_MOB_SHADOW_FLAGS``): with nothing to stamp, the
    leaf words are what they were before the flags existed.
    """
    if no_shadow_cast is None or not rt_settings.PER_MOB_SHADOW_FLAGS:
        return torch.ones((1, num_prims), dtype=torch.bool, device=device)
    v = no_shadow_cast.float()
    if v.dim() >= 4:  # [T?, N, 3, 1] -> per-primitive
        v = v[:, :, 0, :]
    blocked = v.reshape(v.shape[0], v.shape[1], -1).amax(-1).amax(0) > 0.5
    return (~blocked).reshape(1, -1).contiguous()


def closed_shell_ceiling_flag(closed_shell, transmission):
    """Fold a closed-shell declaration with its transmission exemption.

    Returns a per-triangle float32 tensor: 1.0 where the surface's coverage may
    be ceilinged as a closed shell (``SOLID_SHELL_ALPHA``), 0.0 anywhere the
    declaration is absent or the material transmits. Transmission exempts
    because refraction visits both shells as physical transport -- capping the
    second crossing would eat the refracted path -- and is taken amax over
    corners and frames, so a material that transmits at ANY authored moment
    stays exempt everywhere (conservative: it errs toward today's behaviour,
    never toward a wrongly-capped shell).

    ``closed_shell`` / ``transmission`` are the packed per-corner values,
    ``[T?, N, 3, 1]``-shaped or ``None`` (undeclared / non-PBR material).
    """
    if closed_shell is None:
        return None
    closed_v = closed_shell.float()
    if closed_v.dim() >= 4:  # [T?, N, 3, 1] -> per-triangle corner 0
        closed_v = closed_v[:, :, 0, :]
    closed_any = closed_v.reshape(closed_v.shape[0], -1).amax(0) > 0.5
    if transmission is None:
        transmits = torch.zeros_like(closed_any)
    else:
        trans_v = transmission.float()
        if trans_v.dim() >= 4:
            trans_v = trans_v[..., 0, :]
        transmits = trans_v.reshape(trans_v.shape[0], -1).amax(0) > 1e-6
    return (
        (closed_any & ~transmits)
        .to(torch.float32)
        .reshape(1, -1)
        .to(closed_shell.device)
    )


class RayTracedTrianglePrimitive(TrianglePrimitive):
    """Triangle batch rendered by ray tracing a spatio-temporal BVH."""

    frame_dependent_source_attrs = (
        "corners",
        "colors",
        "normals",
        "uvs",
        "texture_map",
        "material_texture_map",
        "normal_texture_map",
        "reflectivity",
        "roughness",
        "refractive_index",
        "transmission",
        "shader_param_values",
    )

    stbvh_tightness = env_float("ALGAN_STBVH_TIGHTNESS", 1.0)

    # Renderer-internal transport channels, shared with
    # ``RayTracedBezierCircuitPrimitive``. ``reflectivity`` stores material
    # metalness for historical packed-layout compatibility; a negative value
    # marks a non-PBR material. ``refractive_index`` is an unsigned magnitude
    # (0 = non-PBR) feeding dielectric F0 and Snell; ``transmission`` alone says
    # whether -- and how much -- the surface transmits. All are derived from the
    # material alone (see ``_derive_material_surface_params``) -- there is no
    # user-facing renderer control, matching the Three.js material interface.
    #
    # ``one_sided`` is the odd one out: the MOB declares it
    # (:meth:`declare_one_sided`) rather than the material, and it is packed
    # into the material block rather than into ``tri_extra``. It rides this
    # tuple for the machinery around it -- the per-member gather below, whose
    # default fill of 0.0 is exactly "two-sided, as before", and the logical-PN
    # dice, which has to carry it to every diced triangle.
    #
    # ``closed_shell`` rides beside it for the same reasons: the MOB declares
    # it (:meth:`declare_closed_shell`), members that say nothing are open
    # (the 0.0 fill), and the dice carries it to every diced triangle. Unlike
    # ``one_sided`` it is consumed on the HOST -- the sheet compaction's
    # closed-shell coverage ceiling reads it out of the merged scene as
    # ``tri_closed``, folded there with the transmission exemption (a closed
    # shell that transmits refracts through both shells and must keep them).
    # ``no_shadow_cast`` / ``no_shadow_receive`` ride here for the same reasons
    # again: the MOB declares them (:meth:`declare_shadow_flags`), a member that
    # says nothing takes the 0.0 fill, and the dice carries them to every diced
    # triangle. Both are spelled NEGATIVELY -- 0.0 is "casts" and "receives",
    # what every mob did before the flags existed -- because that 0.0 fill and
    # the material block's padding rule both require a zero to mean the old
    # behaviour. ``no_shadow_receive`` is packed into the material block beside
    # ``one_sided``; ``no_shadow_cast`` is consumed on the HOST, where it
    # becomes the BVH leaf word's caster bit (``_rt_frame_casts``).
    _surface_params = (
        "reflectivity",
        "roughness",
        "refractive_index",
        "transmission",
        "one_sided",
        "closed_shell",
        "no_shadow_cast",
        "no_shadow_receive",
    )

    def __init__(
        self,
        corners=None,
        colors=None,
        opacity=1,
        normals=None,
        perimeter_points=None,
        reverse_perimeter=False,
        triangle_collection=None,
        glow=0,
        shader=None,
        uvs=None,
        texture_map=None,
        material_texture_map=None,
        material_texture_flags=0,
        normal_texture_map=None,
        **shader_kwargs,
    ):
        if triangle_collection is not None:
            super().__init__(
                corners,
                colors,
                opacity,
                normals,
                perimeter_points,
                reverse_perimeter,
                triangle_collection,
                glow,
                shader,
                uvs=uvs,
                texture_map=texture_map,
                material_texture_map=material_texture_map,
                material_texture_flags=material_texture_flags,
                normal_texture_map=normal_texture_map,
                **shader_kwargs,
            )
            # Per-member primitive counts, in concatenation order. This is what
            # lets ``tri_obj`` (the resolve's per-triangle SOURCE-SURFACE id,
            # DESIGN_analytic_aa_v2.md ss4.2) tell the collection's mobs apart:
            # the batcher merges every same-identifier mob into one collection,
            # so "one part = one surface" is false the moment two spheres share
            # a batch. For a flat collection these are triangle counts; for a
            # logical-PN collection, PATCH counts (the dice maps them to
            # per-frame triangle ids).
            self._rt_obj_counts = [
                int(t.corners.shape[1]) // 3 for t in triangle_collection
            ]
            # A member's count is the right surface granularity only when one
            # member IS one surface, which is false at both ends: a
            # ``Polyhedron`` hands the batcher one member per TRIANGLE (a Cube
            # is twelve, so no run can ever span a face), and a packed-grid
            # ``Surface`` hands it one member for EVERY packed sphere at once.
            # Members may therefore declare identity themselves -- ``mesh_key``
            # to merge with the neighbours sharing it, ``mesh_ids`` to subdivide
            # into per-triangle shells -- which ``_mesh_ids_from_collection``
            # resolves into explicit per-triangle ids. ``None`` keeps the
            # per-member counts, so a mob that declares nothing is unchanged.
            self._rt_obj_ids, self._rt_obj_ids_n = _mesh_ids_from_collection(
                triangle_collection, self._rt_obj_counts
            )
            # Gather per-mob surface params with the same broadcast/cat
            # recipe the base class applies to corners/colors, so shapes
            # line up -- except along time: the references are sliced to a
            # single frame so a static parameter (the usual case) stays
            # single-frame instead of being expanded to the batch length.
            for name in self._surface_params:
                values = []
                for triangle in triangle_collection:
                    v = getattr(triangle, name, None)
                    if v is None:
                        fill = -1.0 if name == "reflectivity" else 0.0
                        v = torch.full_like(triangle.colors[:1, ..., :1], fill)
                    v = broadcast_all(
                        [
                            triangle.corners[:1],
                            triangle.colors[:1],
                            triangle.normals[:1],
                            v,
                        ],
                        ignored_dims=[-1],
                    )[-1][..., :1]
                    values.append(v)
                # A registered (animatable) surface param on an *animated* mob
                # materializes per batch timestep ([T, ...]) while static
                # mobs' params stay single-frame; unify the time dims before
                # the cat (the kernels index time as ``f % T`` either way).
                values, _ = _unify_time(values, "surface param merge")
                setattr(
                    self,
                    name,
                    unsquish(torch.cat(values, 1), -2, 3).to(self.corners.device),
                )
        else:
            super().__init__(
                corners,
                colors,
                opacity,
                normals,
                perimeter_points,
                reverse_perimeter,
                triangle_collection,
                glow,
                shader=shader,
                uvs=uvs,
                texture_map=texture_map,
                material_texture_map=material_texture_map,
                material_texture_flags=material_texture_flags,
                normal_texture_map=normal_texture_map,
                **shader_kwargs,
            )
            self._rt_obj_counts = None
            # A lone primitive is one surface unless it declares its own shells
            # (a packed-grid Surface, a multi-part glTF mesh).
            self._rt_obj_ids, self._rt_obj_ids_n = _mesh_ids_from_collection(
                [self], None
            )
            self._derive_material_surface_params()
            # Two-sided until the mob says otherwise, which it does after
            # construction (``declare_one_sided``) -- the same point at which
            # it declares ``mesh_key``.
            self.declare_one_sided(False)
            self.declare_closed_shell(False)
            self.declare_shadow_flags(True, True)

    def declare_one_sided(self, one_sided=True):
        """Declare whether this primitive's hits are shaded from one side only.

        Called by the mob that built the primitive, from its geometry rather
        than from its material: ``one_sided`` says the normals face out of a
        solid, so a back-facing hit is its inside and must be shaded as such
        instead of borrowing the viewer's side
        (:attr:`~algan.animatable_base.mob.Mob.two_sided`).

        Stored per corner, matching the other ``_surface_params``, so the
        collection merge and the logical-PN dice carry it with everything else.
        """
        self.one_sided = torch.full_like(
            self.colors[:1, ..., :1], 1.0 if one_sided else 0.0
        )
        return self

    def declare_closed_shell(self, closed=True):
        """Declare whether this primitive's triangles are a closed shell.

        Called by the mob that built the primitive, beside
        :meth:`declare_one_sided`: ``closed`` says the triangles tile a shell
        that encloses its interior -- every camera ray that enters crosses a
        second time on its way out -- so ``Mob.opacity`` can mean what it says
        (one attenuation of what is behind) instead of compositing once per
        crossing (:attr:`~algan.animatable_base.mob.Mob.closed_shell`).

        The declaration is consumed host-side by the sheet compaction's
        coverage ceiling; it never reaches a kernel. A surface whose material
        transmits is folded back to open at pack time (``_rt_tri_closed``):
        refraction visits both shells as physical transport, and the ceiling
        would eat the second one.
        """
        self.closed_shell = torch.full_like(
            self.colors[:1, ..., :1], 1.0 if closed else 0.0
        )
        return self

    def declare_shadow_flags(self, casts=True, receives=True):
        """Declare whether this primitive casts and receives shadows.

        Called by the mob that built the primitive, beside
        :meth:`declare_one_sided`, from
        :attr:`~algan.animatable_base.mob.Mob.casts_shadows` and
        :attr:`~algan.animatable_base.mob.Mob.receives_shadows`.

        Stored NEGATED and per corner, matching the other ``_surface_params``:
        the collection merge and the logical-PN dice fill an absent member with
        0.0, and 0.0 has to mean "casts" / "receives", the behaviour that
        existed before the flags. The two land in different places downstream --
        ``no_shadow_receive`` in the material block that ``_run_frag_pipeline``
        already holds at the hit, ``no_shadow_cast`` in the BVH leaf word the
        shadow traversal already loads -- but both are declared here, once,
        because a mob declares them together.
        """
        self.no_shadow_cast = torch.full_like(
            self.colors[:1, ..., :1], 0.0 if casts else 1.0
        )
        self.no_shadow_receive = torch.full_like(
            self.colors[:1, ..., :1], 0.0 if receives else 1.0
        )
        return self

    def _derive_material_surface_params(self):
        """Derive ray transport directly from material shader parameters.

        This intentionally does not copy values onto separate mob attributes:
        the tensors here are the materialised ``metalness``, ``roughness``,
        ``ior`` and ``transmission`` shader parameters, so animating those
        public material properties automatically updates ray transport.
        """
        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        by_name = dict(zip(names, values))
        template = self.colors[:1, ..., :1]

        metalness = by_name.get("metalness")
        if metalness is None:
            self.reflectivity = torch.full_like(template, -1.0)
            self.roughness = torch.zeros_like(template)
            self.refractive_index = torch.zeros_like(template)
            self.transmission = torch.zeros_like(template)
            return

        def surface_value(value, default):
            if value is None:
                value = torch.full_like(template, default)
            else:
                value = cast_to_tensor(value).to(self.colors.device)
            return broadcast_all(
                [self.corners[:1], self.colors[:1], value],
                ignored_dims=[-1],
            )[-1][..., :1]

        self.reflectivity = surface_value(metalness, 0.0)
        self.roughness = surface_value(by_name.get("roughness"), 1.0)

        ior = by_name.get("ior")
        if ior is None:
            # MeshStandardMaterial uses Three.js's fixed dielectric F0=0.04,
            # corresponding to IOR 1.5, and does not transmit.
            self.refractive_index = torch.full_like(self.reflectivity, 1.5)
            self.transmission = torch.zeros_like(self.reflectivity)
            return

        # ``transmission`` is a channel of its own, never folded into alpha:
        # alpha stays pure coverage (is the surface there / how faded), and
        # transmission is how much light passes through the part that IS there.
        # The kernel splits a hit into alpha*R reflected, alpha*(1-R)*T
        # transmitted, alpha*(1-R)*(1-T) shaded and (1-alpha) missed. Folding
        # the two together made an object at transmission=1 indistinguishable
        # from an absent one, and made a glass mob's spawn fade invisible.
        self.refractive_index = surface_value(ior, 1.5).abs()
        self.transmission = surface_value(by_name.get("transmission"), 0.0).clamp(
            0.0, 1.0
        )

    def _shaded_per_fragment(self):
        """True when this primitive's hits are shaded per fragment in-kernel
        (deterministic renderer, fragment shading on, core lit material or a
        custom fragment pipeline) rather than baked per vertex -- in which case
        ``colors`` stays raw albedo.
        """
        shader = getattr(self, "shader", None)
        if getattr(shader, "_frag_pipeline_id", None) is not None:
            # A custom pipeline always shades in-kernel on the deterministic
            # renderer (fragment shading is forced on for such a scene).
            return rt_settings.SAMPLES_PER_PIXEL <= 1
        return (
            rt_settings.FRAGMENT_SHADING
            and rt_settings.SAMPLES_PER_PIXEL <= 1
            and _shader_is_core(shader)
        )

    def _ordered_shader_param_values(self):
        """The shader's extra (material) parameters as a positional list in the
        shader's own signature order.

        Rebuild the argument list from the shader's signature so custom shaders
        remain robust to missing optional parameters. A parameter the mob does
        not carry itself falls back to
        ``SETTINGS.style.default_material``'s value for it (this primitive was
        built by the no-material fallback, so its mob has nothing registered),
        and only then to the shader signature's default.
        """
        import inspect

        sig = inspect.signature(self.shader).parameters
        num_fixed = SHADER_FIXED_PARAM_COUNT
        extra_names = list(sig.keys())[num_fixed:]

        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        by_name = dict(zip(names, values))
        if not by_name:
            # Exact no-op when the fallback did not run (empty class-level
            # mapping): nothing is added and signature defaults apply as before.
            by_name.update(self.default_material_params)

        args = []
        for name in extra_names:
            if name in by_name:
                args.append(by_name[name])
                continue
            default = sig[name].default
            v = default if default is not inspect._empty else 0
            args.append(v)
        return args

    def _shade_vertex_colors(self, camera, light_sources):
        """Vertex shading, identical to the rasterized pipeline. Skipped in
        physical mode (raw albedo, the pathtracer lights the scene) and when
        this primitive is shaded per fragment instead (see
        :meth:`_shaded_per_fragment`).
        """
        if self._shaded_per_fragment():
            return
        d = -1
        if getattr(self, "shader", None) is not None:
            param_values = self._ordered_shader_param_values()
            for light_source in light_sources:
                if getattr(light_source, "_render_aux", None) is not None:
                    # Extended light types (directional / ambient / spot /
                    # area / ...) are evaluated by the per-fragment lighting
                    # path, which their presence forces on; the per-vertex
                    # shader convention only knows point lights.
                    continue
                # A zero-colour frame row is a light outside its lifespan (or
                # genuinely black) and must contribute nothing -- exactly as
                # if the light were not in the list. The legacy default shader
                # lerps toward the light colour with a colour-independent
                # weight, so without this gate a not-yet-spawned light would
                # darken vertex-shaded mobs, and the output would depend on
                # whether a batch boundary happened to include the light.
                light_color = light_source.light_color
                live = (light_color != 0).any(dim=-1, keepdim=True)
                if not bool(live.any()):
                    continue
                with self.memory.temp():
                    shaded = self.shader(
                        self.memory,
                        self.corners,
                        self.normals,
                        self.colors[..., :d],
                        camera.ray_origin,
                        light_source.origin,
                        light_color,
                        1,
                        1,
                        *param_values,
                    )
                    if bool(live.all()):
                        # Every frame is live: plain assignment, bit-identical
                        # to the ungated path.
                        self.colors[..., :d] = shaded
                    else:
                        self.colors[..., :d] = torch.where(
                            live, shaded, self.colors[..., :d]
                        )

    def _pack_material(self):
        """Per-primitive material id ``[1, N]`` and the canonical material
        parameter block ``[Tm, N, MAT_W]`` consumed by the in-kernel fragment
        shader. Material properties are per-mob constants broadcast to
        vertices, so each triangle's value is taken from its first corner.
        Non-core (or absent) shaders get id 1 (passthrough) and default params.
        """
        colors = self.colors
        N = colors.shape[1]
        device = colors.device

        def per_triangle(value):
            v = value.float().to(device)
            if v.dim() >= 4:  # [T, N, 3, w] -> per-triangle corner 0
                v = v[:, :, 0, :]
            return v

        # Custom fragment pipeline (Mob.set_fragment_shader): the pipeline
        # metadata rides on the marker shader object (so it flows to the
        # primitive via the ordinary ``shader=`` handoff). A per-primitive
        # pipeline id (>= _USER_PIPELINE_BASE) and a variable-width param block
        # laid out by the pipeline's stages.
        shader = getattr(self, "shader", None)
        if getattr(shader, "_frag_pipeline_id", None) is not None:
            return self._pack_frag_pipeline(shader, N, device, per_triangle)

        mat_id = torch.full(
            (1, N), _shader_material_id(shader), dtype=torch.int32, device=device
        )
        # Later writes win. The block is sized from the primitive's own
        # per-frame parameter rows only, then filled in three passes:
        # SETTINGS.style.default_material's constant values first (a
        # no-material fallback primitive; they broadcast over every row and
        # contribute nothing to Tm), then the primitive's own registered
        # parameters overwriting them by name -- an explicit per-mob value
        # always beats the process-wide default.
        seeds = []
        if _shader_is_core(shader):
            # A configured default material such as
            # ``SETTINGS.style.set(default_material=MeshStandardMaterial(
            # roughness=0.3))`` must reach the packed block, not silently
            # render at ``_MAT_DEFAULTS``.
            for name, value in self.default_material_params.items():
                if name in _MAT_SLOTS:
                    seeds.append((name, torch.as_tensor(value, dtype=torch.float32)))
        pairs = []
        if _shader_is_core(shader):
            # The material's shader params, addressed by their real names.
            names = list(getattr(self, "shader_param_names", None) or [])
            values = list(getattr(self, "shader_param_values", None) or [])
            for name, value in zip(names, values):
                if name in _MAT_SLOTS and value is not None:
                    pairs.append((name, per_triangle(value)))
        # The mob's own declaration, not the material's: whether this geometry
        # has an outside for the shading side to be read off (see
        # ``declare_one_sided``). Packed here because this block is what
        # ``_run_frag_pipeline`` already has in hand at the hit.
        one_sided = getattr(self, "one_sided", None)
        if one_sided is not None:
            pairs.append((_MAT_ONE_SIDED, per_triangle(one_sided)))
        # Beside it, and for the same reason: whether shadows cast onto this
        # geometry darken it (``declare_shadow_flags``). The CASTING half of
        # that declaration is not here -- it never reaches a shading stage, only
        # the BVH leaf word (``_rt_frame_casts``).
        no_shadow_receive = getattr(self, "no_shadow_receive", None)
        if no_shadow_receive is not None and rt_settings.PER_MOB_SHADOW_FLAGS:
            pairs.append((_MAT_NO_SHADOW_RECEIVE, per_triangle(no_shadow_receive)))
        Tm = max([1] + [v.shape[0] for _n, v in pairs])
        mat = (
            torch.tensor(_MAT_DEFAULTS, device=device)
            .view(1, 1, MAT_W)
            .expand(Tm, N, MAT_W)
            .contiguous()
        )
        for name, v in seeds:
            start, width = _MAT_SLOTS[name]
            # A constant seed broadcasts over every time row and triangle.
            if v.numel() != width:
                v = v.reshape(-1).expand(width)
            mat[:, :, start : start + width] = v.to(device)
        for name, v in pairs:
            # A geometry-declared entry addresses its slot by index (it has no
            # material-property name to look up); a material's addresses it by
            # name. Both are single-slot or vector writes into the same block.
            start, width = (name, 1) if isinstance(name, int) else _MAT_SLOTS[name]
            if v.shape[-1] != width:  # broadcast a scalar into a vector slot
                v = v.expand(*v.shape[:-1], width)
            mat[:, :, start : start + width] = v
        return mat_id.contiguous(), mat.contiguous()

    def _pack_frag_pipeline(self, shader, N, device, per_triangle):
        """Per-primitive pipeline id ``[1, N]`` and the custom-pipeline parameter
        block ``[Tm, N, W]`` for a mob with a fragment pipeline
        (:meth:`~algan.mobs.mob.Mob.set_fragment_shader`). Each stage's
        parameters occupy a contiguous slot range (the marker shader's
        ``_frag_param_layout`` maps attr name -> absolute slot); values are the
        materialised animated ``shader_param_values``, with defaults filling any
        slot whose attr is absent.
        """
        pid = int(shader._frag_pipeline_id)
        W = int(shader._frag_total_width)
        layout = shader._frag_param_layout  # list of (name, slot, width, default)
        mat_id = torch.full((1, N), pid, dtype=torch.int32, device=device)

        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        val_by_name = dict(zip(names, values))

        # Default row (every slot is covered by exactly one layout entry).
        default_row = torch.zeros(W, dtype=torch.float32, device=device)
        for _name, slot, width, default in layout:
            dv = torch.as_tensor(default, dtype=torch.float32, device=device).flatten()
            if dv.numel() == 1 and width > 1:
                dv = dv.expand(width)
            default_row[slot : slot + width] = dv[:width]

        pairs = []
        for name, slot, width, _default in layout:
            v = val_by_name.get(name)
            if v is not None:
                pairs.append((slot, width, per_triangle(v)))
        Tm = max([1] + [v.shape[0] for _s, _w, v in pairs])
        mat = default_row.view(1, 1, W).expand(Tm, N, W).contiguous()
        for slot, width, v in pairs:
            if v.shape[-1] != width:  # broadcast a scalar into a vector slot
                v = v.expand(*v.shape[:-1], width)
            mat[:, :, slot : slot + width] = v
        return mat_id.contiguous(), mat.contiguous()

    def _pack_surface_extra(self, error_context):
        """Per-primitive surface params [Te, N, 15]: the interleaved per-corner
        (reflectivity, roughness) pairs in columns 0-5 (consumed by
        ``_triangle_extra`` in every kernel), followed by the per-corner
        refractive index in columns 6-8 (unsigned magnitude, 0 = non-PBR; read
        by the wavefront's ``_corner_ior``), followed by the per-corner
        transmission in columns 9-11 (0 = opaque to light passing through; read
        by ``_corner_transmission``), followed by the per-PRIMITIVE Beer-Lambert
        absorption coefficient in columns 12-14 (``_EXTRA_SIGMA``; read by the
        shadow march over a solid's interior chord).

        Sigma is per-primitive rather than per-corner -- one primitive is one
        material, so there is nothing to interpolate across the face -- and it
        does NOT ride ``_surface_params``: that tuple feeds the collection
        gather and the logical-PN dice, both of which slice values to one
        scalar channel per corner, which would truncate an RGB coefficient to
        its red channel. It travels as the ``attenuation_sigma`` shader
        parameter instead (computed by
        ``materials._attenuation_sigma``, packed into ``shader_param_values``
        exactly like ``metalness`` and ``ior``), which the collection merge and
        the PN dice both already carry through untouched.
        """
        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        if "attenuation_sigma" in names:
            sigma_raw = values[names.index("attenuation_sigma")]
        else:
            sigma_raw = None
        if sigma_raw is not None:
            sigma_e = sigma_raw.float()
        else:
            # No attenuation parameter on this material (every non-PBR and
            # legacy shader): zero is no absorption. Shaped like the other
            # per-corner params so the time unification below sees a
            # compatible tensor.
            sigma_e = torch.zeros_like(self.reflectivity.float()).expand(
                *self.reflectivity.shape[:-1], 3
            )
        (
            (
                reflectivity_e,
                roughness_e,
                ior_e,
                transmission_e,
                sigma_e,
            ),
            _,
        ) = _unify_time(
            [
                self.reflectivity.float(),
                self.roughness.float(),
                self.refractive_index.float(),
                self.transmission.float(),
                sigma_e,
            ],
            error_context,
        )
        n_t, n_p = reflectivity_e.shape[0], reflectivity_e.shape[1]
        refl_rough = torch.cat((reflectivity_e, roughness_e), -1).reshape(n_t, n_p, 6)
        ior = ior_e.reshape(n_t, n_p, 3)
        transmission = transmission_e.reshape(n_t, n_p, 3)
        # Collapse the per-corner fan of the RGB coefficient to one triple per
        # primitive. The parameter arrives shaped like every other surface
        # param -- [T, N, 3 corners, 3 channels] -- and a material is uniform
        # across a face, so every corner carries the same value and taking
        # corner 0 is lossless. The reshape names the corner axis explicitly
        # rather than assuming its position.
        sigma = sigma_e.reshape(sigma_e.shape[0], sigma_e.shape[1], -1, 3)[:, :, 0, :]
        return torch.cat((refl_rough, ior, transmission, sigma), -1).contiguous()

    def _pack_frame_visibility(self, lo, hi, colors, error_context):
        """Per-frame bounds; frames where a primitive is fully transparent
        and not glowing are marked empty so they never enter the BVH. Fully opaque frames
        are flagged so the trace kernel can prune hits behind them while
        gathering.
        """
        # Last channel is opacity. Indexing (rather than the Color.opacity
        # property) so this also works for textured surfaces (ImageMob), whose
        # per-vertex colors are plain tensors, not Color instances.
        alpha = colors[..., -1]

        # Alpha is pure coverage, so it alone decides presence: a mob that is
        # un-spawned or faded out is absent, while clear glass keeps its
        # coverage and stays visible (see _derive_material_surface_params).
        visible = alpha.amax(-1) > MIN_ALPHA
        # ...except where a colour texture, not the corner colours, is what
        # supplies coverage. Every cut-out image (a PNG sticker, an ImageMob)
        # has transparent corner texels, and a textured quad's corners ARE its
        # triangles' corners, so this test culled whole triangles out of an
        # otherwise perfectly visible picture -- chopping it along the quad
        # diagonal. The texture's own alpha decides instead.
        texture = getattr(self, "_rt_texture_map", None)
        if texture is not None:
            texture_visible = (
                texture.reshape(texture.shape[0], -1, texture.shape[-1])[..., -1].amax(
                    -1, keepdim=True
                )
                > MIN_ALPHA
            )
            (visible, texture_visible), _ = _unify_time(
                [visible, texture_visible], error_context
            )
            visible = visible | texture_visible
        # ...but full coverage is not enough to prune hits behind: a
        # transmissive surface still lets light through at alpha 1.
        opaque = alpha.amin(-1) >= 1.0 - 1e-6
        transmission = getattr(self, "transmission", None)
        if transmission is not None:
            opaque = opaque & (transmission[..., 0] <= 1e-6).all(-1)
        # ...and a colour texture that cannot be proven alpha-opaque makes
        # every hit's alpha texture-dependent, so the primitive must not
        # carry the interval-opaque BVH leaf flag: the traversal gather
        # prunes hits behind an opaque-flagged hit, which for a cut-out
        # texture (an ImageMob sticker) deleted the scene visible through
        # its transparent texels on the classic/secondary-ray path, and the
        # shadow any-hit early-out would turn the same texels into false
        # full occlusion. Mirrors scene_builder._texture_alpha_is_opaque.
        if (
            texture is not None
            and texture.shape[-1] >= 4
            and not bool((texture[..., 3] >= 1.0 - 1e-6).all())
        ):
            opaque = torch.zeros_like(opaque)

        (lo, hi, visible, opaque), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), opaque.unsqueeze(-1)], error_context
        )
        visible = visible.squeeze(-1)
        self._rt_frame_opaque = opaque.squeeze(-1).contiguous()
        # Rides beside the opacity flag because it has the same destination:
        # the BVH leaf word, where a bit the shadow traversal already loads
        # says whether this primitive may block a shadow ray.
        self._rt_frame_casts = shadow_cast_flag(
            getattr(self, "no_shadow_cast", None), lo.shape[1], lo.device
        )
        self._rt_frame_lo = torch.where(
            visible.unsqueeze(-1), lo, torch.tensor(EMPTY_LO, device=lo.device)
        ).contiguous()
        self._rt_frame_hi = torch.where(
            visible.unsqueeze(-1), hi, torch.tensor(EMPTY_HI, device=hi.device)
        ).contiguous()

    def _stash_texture_maps(self):
        """Stash the raw texture maps (color / material / normal) for merge
        time and return the packed ``[T, N, 6]`` per-triangle uv tensor, or
        None when the batch is untextured.
        """
        if self.uvs is None:
            self._rt_texture_map = None
            self._rt_material_texture = None
            self._rt_material_flags = 0
            self._rt_normal_texture = None
            return None
        uvs = (
            self.uvs.float()
            .reshape(self.uvs.shape[0], self.uvs.shape[1], 6)
            .contiguous()
        )
        self._rt_texture_map = (
            self.texture_map.float().contiguous()
            if self.texture_map is not None
            else None
        )
        mtex = getattr(self, "material_texture_map", None)
        self._rt_material_texture = (
            mtex.float().contiguous() if mtex is not None else None
        )
        self._rt_material_flags = int(getattr(self, "material_texture_flags", 0) or 0)
        ntex = getattr(self, "normal_texture_map", None)
        self._rt_normal_texture = (
            ntex.float().contiguous() if ntex is not None else None
        )
        return uvs

    def _release_unpacked_geometry(self):
        """Everything the renderer needs now lives in the packed arrays;
        release the unpacked geometry to halve resident GPU memory.
        """
        self.corners = self.normals = None
        self.reflectivity = self.roughness = self.refractive_index = None
        self.colors = self.shader_param_values = None
        self.uvs = self.texture_map = None
        self.material_texture_map = self.normal_texture_map = None

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)

    def project_to_screen(self, camera, light_sources):
        self._shade_vertex_colors(camera, light_sources)
        return self._pack_projected_flat_geometry(camera)

    def _pack_projected_flat_geometry(self, camera):
        corners = self.corners.float()
        normals = self.normals.float()
        # Hot/cold split, each array with its own (independent) time
        # dimension: positions are touched by every candidate
        # intersection, normals only by hits that bounce or scatter, and
        # reflectivity/roughness (usually static) only by confirmed hits.
        self._rt_tri_pos = corners.reshape(
            corners.shape[0], corners.shape[1], 9
        ).contiguous()
        self._rt_tri_norm = normals.reshape(
            normals.shape[0], normals.shape[1], 9
        ).contiguous()
        self._rt_tri_extra = self._pack_surface_extra("triangle surface params")
        self._rt_tri_colors = self.colors.float().contiguous()
        self._rt_tri_mat_id, self._rt_tri_mat = self._pack_material()
        # The closed-shell declaration folded with its one exemption, as one
        # per-triangle flag the merge carries and the sheet compaction reads
        # (``tri_closed``); see ``closed_shell_ceiling_flag`` for the folding.
        self._rt_tri_closed = closed_shell_ceiling_flag(
            getattr(self, "closed_shell", None),
            getattr(self, "transmission", None),
        )
        if self._rt_tri_closed is None:
            self._rt_tri_closed = torch.zeros(
                (1, corners.shape[1]), dtype=torch.float32, device=corners.device
            )
        self._rt_tri_closed = self._rt_tri_closed.contiguous()
        self._rt_num_frames = camera.ray_origin.shape[0]

        # Per-triangle SOURCE-SURFACE id, [1, N] (or [T, N] for diced logical
        # PN, whose row->patch mapping moves per frame with the adaptive
        # levels). Local member indices; the scene merge offsets them per
        # primitive so ids are globally unique. ``_rt_tri_obj_n`` is the id
        # count, kept beside it so the merge needs no device sync to offset.
        #
        # Three sources, most specific first: the diced logical-PN map; explicit
        # per-triangle ids resolved from the members' own ``mesh_key`` /
        # ``mesh_ids`` declarations (``_mesh_ids_from_collection``); then the
        # per-member counts, which are what a collection declaring nothing gets.
        pn_obj = getattr(self, "_logical_pn_tri_obj", None)
        obj_ids = getattr(self, "_rt_obj_ids", None)
        counts = getattr(self, "_rt_obj_counts", None)
        if not rt_settings.MESH_ID:
            obj_ids = None
        if pn_obj is not None:
            self._rt_tri_obj = pn_obj
            # The dice records its own id count: with MESH_ID on it resolves the
            # members' declaration rather than their counts, and the merge needs
            # the real number to offset this primitive's ids without colliding.
            self._rt_tri_obj_n = getattr(
                self, "_logical_pn_tri_obj_n", len(counts) if counts else 1
            )
        elif obj_ids is not None:
            self._rt_tri_obj = obj_ids.view(1, -1).to(corners.device).contiguous()
            self._rt_tri_obj_n = int(self._rt_obj_ids_n)
        elif counts:
            self._rt_tri_obj = (
                torch.repeat_interleave(
                    torch.arange(len(counts), dtype=torch.int32, device=corners.device),
                    torch.tensor(counts, dtype=torch.int64, device=corners.device),
                )
                .view(1, -1)
                .contiguous()
            )
            self._rt_tri_obj_n = len(counts)
        else:
            self._rt_tri_obj = torch.zeros(
                (1, corners.shape[1]), dtype=torch.int32, device=corners.device
            )
            self._rt_tri_obj_n = 1

        uvs = self._stash_texture_maps()
        self._rt_tri_uvs = uvs.to(corners.device) if uvs is not None else None

        self._pack_frame_visibility(
            corners.amin(-2),
            corners.amax(-2),
            self._rt_tri_colors,
            "triangle bounds/colors",
        )

        self._release_unpacked_geometry()
        return self

    def render(
        self,
        primitives,
        scene,
        save_image,
        screen_width,
        screen_height,
        time_start,
        time_end,
        background_color,
        transparent_background=False,
        *args,
        **kwargs,
    ):
        return KERNEL_REGISTRY.render_kernel(
            primitives,
            scene,
            screen_width,
            screen_height,
            time_start,
            time_end,
            background_color,
            transparent_background,
            *args,
            **kwargs,
        )


class LogicalPNTrianglePrimitive(RayTracedTrianglePrimitive):
    """Adaptively diced logical PN patches rendered as ordinary flat triangles.

    Logical PN patches use their fixed construction-time topology as source
    geometry and dice into flat triangles for each materialized camera frame.
    The packed result follows the normal flat-triangle/STBVH path -- no curved
    patch primitive reaches the ray tracer or the STBVH.

    **Every patch picks its own dice, in every frame.**  A patch that fills the
    screen costs what it needs and nothing else pays for it -- neither the other
    patches of the same mesh in that frame, nor the same patch in the frames
    where it is small or off screen.  Only the padded width of the output tensor
    is shared: each frame's patches are packed back to back, and the batch is
    padded to the largest per-frame total (surplus rows are marked invisible).

    A dice is ``(level, across level, apex)``, not one level: ``2 ** level`` rows
    fanning from the apex corner, each cut into at most ``2 ** across`` columns
    (:func:`~algan.rendering.logical_pn.dice_pattern`).  Equal levels *are* the
    uniform barycentric grid, so this only ever removes microtriangles from a
    patch whose two directions want different detail -- anything developable,
    where the flat direction would otherwise pay whatever the curved one costs.

    Independent per-patch dices would crack the mesh open along its seams, so
    the level of a patch's three boundary curves is decided separately from its
    interior:

    * A boundary curve's level is a function of that curve alone -- its two
      endpoints and their normals, which the two patches sharing it hold in
      common -- evaluated on canonically ordered controls
      (:func:`~algan.rendering.logical_pn.logical_pn_edge_control_points`) so
      both neighbours reach a bit-identical answer without any adjacency
      information.
    * A patch's own level is at least the largest of its three boundary levels,
      and is then raised until its interior is flat enough.  Its rows are then
      laid parallel to its *coarsest* boundary curve and the dice is coarsened
      across them for as long as the same criterion still passes, so an
      anisotropic patch is one the criterion has measured, never one inferred
      from its boundary.
    * Where the dice's level on an edge exceeds that edge's own, the boundary
      vertices are snapped back onto the coarser boundary polyline
      (:func:`~algan.rendering.logical_pn.snap_boundary_values`).  Levels are
      powers of two and the dice's knots along any one edge are evenly spaced,
      so the coarse polyline's knots are always vertices of the dice and the
      snapped boundary reproduces it exactly: the seam is watertight whatever
      the two neighbours chose.

    The tolerance guarantee is therefore stated per component: the diced
    boundary lands within the frame's error budget of the true boundary curve
    and the diced interior within it of the true patch, both in output pixels.
    In the band of microtriangles touching a snapped boundary the two
    displacements can add, for a worst case of twice the budget.  The budget
    itself is the finer of the two tolerances at this frame's resolution --
    ``render_tolerance`` as a fraction of the frame height, and
    ``render_tolerance_pixels`` as an absolute pixel count -- see
    :meth:`_pixel_threshold`.

    Both criteria measure against the *logical PN patch*, which is itself only
    an approximation of the surface the author asked for.  Where the mesh
    declares how good an approximation (``geometry_slack_ratio``, which a
    :class:`~algan.mobs.surfaces.surface.Surface` sets from its
    ``geometry_tolerance``), that much deviation is subtracted from what the
    searches measure: detail finer than the reference surface's own accuracy is
    the reference's error, not the surface's, and resolving it buys nothing.
    The guarantee then carries the accuracy of the logical surface as a second
    term, which is the bound the render already inherits from construction.
    """

    max_subdivision_level = 8
    # Hard ceiling on a single frame's diced triangle count, ``sum over patches
    # of 4 ** level``. Without a budget one pathological frame can ask for a
    # tessellation that cannot be allocated at all, and the render dies inside
    # the level search instead of degrading. Shrinking the frame window -- the
    # render loop's usual response to running out of memory -- cannot save it,
    # so the ceiling has to hold at a single frame.
    #
    # It is enforced *during* both level searches: a level is only promoted
    # while the frame it belongs to still fits, which bounds the searches as
    # well as their result. Deliberately independent of the frame window (a
    # level that moved with how many frames a render batch happened to cover
    # would make the mesh pop at batch boundaries) -- each frame is judged on
    # its own contents alone. With the screen guard in
    # ``_required_patch_levels`` it only binds on meshes that are already
    # enormous, where it trades tessellation quality (with a warning) for
    # finishing the render.
    max_diced_triangles = 2_000_000
    # Peak microtriangles evaluated in one go. The level searches and the dice
    # itself both stream through their work in chunks of this size, so scratch
    # stays bounded no matter how much geometry a frame ends up asking for.
    max_scratch_triangles = 1 << 18
    # Half-extent, in units of the output frame height, of the guard box that
    # projected samples are clamped into before their flatness error is
    # measured.  Comfortably contains the frame at any usual aspect ratio, plus
    # a margin of near-frame geometry.
    screen_guard_factor = 1.5
    _flatness_sample_weights = (
        (0.75, 0.25, 0.0),
        (0.5, 0.5, 0.0),
        (0.25, 0.75, 0.0),
        (0.0, 0.75, 0.25),
        (0.0, 0.5, 0.5),
        (0.0, 0.25, 0.75),
        (0.25, 0.0, 0.75),
        (0.5, 0.0, 0.5),
        (0.75, 0.0, 0.25),
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
    )
    # Parameters, within each chord of a boundary curve, at which that chord's
    # deviation from the curve is measured. A cubic's deviation from its chord
    # is ``3t(1-t)`` times a linear blend of two fixed vectors, so it has at
    # most two humps and these three samples land within 3% of its true peak
    # even in the worst (equal and opposite) case -- comfortably inside the
    # safety factor below. Sampling more finely measurably slowed the search
    # without moving a single level.
    _edge_sample_parameters = (0.25, 0.5, 0.75)
    _flatness_safety_factor = 1.25

    def __init__(
        self,
        *args,
        render_tolerance=0.5,
        render_tolerance_pixels=None,
        geometry_slack_ratio=0.0,
        **kwargs,
    ):
        collection = kwargs.get("triangle_collection")
        if collection is not None:
            tolerances = [
                float(getattr(p, "render_tolerance", render_tolerance))
                for p in collection
            ]
            render_tolerance = min(tolerances)
            # Both tolerances merge the same way and for the same reason: the
            # merged primitive is judged by one criterion, so the finest value
            # any member declares is the only one that cannot over-relax
            # another.
            render_tolerance_pixels = min(
                normalize_pixel_tolerance(
                    getattr(p, "render_tolerance_pixels", render_tolerance_pixels)
                )
                for p in collection
            )
            # A member that declares no slack (a hand-built patch soup is its
            # own surface, exactly) therefore pins the batch to zero.
            geometry_slack_ratio = min(
                float(getattr(p, "geometry_slack_ratio", 0.0)) for p in collection
            )
        super().__init__(*args, **kwargs)
        self.render_tolerance = float(render_tolerance)
        self.render_tolerance_pixels = normalize_pixel_tolerance(
            render_tolerance_pixels
        )
        self.geometry_slack_ratio = float(geometry_slack_ratio)
        if not torch.isfinite(torch.tensor(self.render_tolerance)):
            raise ValueError("render_tolerance must be finite")
        if self.render_tolerance <= 0:
            raise ValueError("render_tolerance must be greater than zero")

    def get_batch_identifier(self):
        # The shadow-casting declaration joins the key, and ONLY here: the BVH
        # leaf word carries one bit per merged primitive column for the whole
        # batch, which assumes a column means one mob's flag on every frame. A
        # diced collection breaks that assumption and nothing else does -- each
        # frame dices adaptively, so a column hosting a patch of mob A in one
        # frame can host a patch of mob B in the next, and the reduction that
        # turns the per-corner declaration into that one bit
        # (``shadow_cast_flag``) has to be conservative over frames. Merging a
        # non-caster with a caster therefore ate part of the CASTER's shadow:
        # measured as a bite out of a sphere's shadow ellipse on every frame of
        # ``benchmarks/_shadow_flags_mixed_dice_check.py``, which is the guard.
        # Splitting the merge group restores the assumption instead of weakening
        # the bit. Flat triangles and circuits need no such split -- neither is
        # diced, so their column-to-primitive mapping is fixed for the batch --
        # and ``receives_shadows`` needs none either, since the material block it
        # rides is itself per frame.
        return (
            f"{super().get_batch_identifier()}"
            f"_logical_pn_render_tolerance={self.render_tolerance}"
            f"_logical_pn_render_tolerance_pixels={self.render_tolerance_pixels}"
            f"_casts_shadows={_declares_no_shadow_cast(self)}"
        )

    def _pixel_threshold(self, screen_height):
        """This frame's error budget for the dice criteria, in output pixels.

        Both tolerances bound the same quantity and the finer one wins.
        ``render_tolerance`` is a fraction of the frame height, so it holds the
        error to a constant share of the picture however large the picture is;
        ``render_tolerance_pixels`` is an absolute count, so it holds the error
        to a constant number of pixels however small the picture is. A
        low-resolution render is therefore still diced well below a pixel --
        which the analytic-coverage antialiasing needs, since a microtriangle
        wider than a pixel is what its coverage is computed from -- while a
        high-resolution one no longer inherits triangles several pixels across
        from a tolerance that only ever scaled with the frame.
        """
        return min(
            self.render_tolerance * float(screen_height),
            self.render_tolerance_pixels,
        )

    @staticmethod
    def _project_to_output_pixels(points, cam_o, sp, sb, screen_height):
        """Perspective-project ``[T, ... ,3]`` points into output pixels.

        Returns the pixels, each point's depth, and the pixels-per-world-unit
        scale there -- the last so a caller holding a world-space length (the
        surface's own accuracy) can say what it is worth on screen at that
        point.
        """
        extra = points.ndim - 2
        camera_shape = (-1,) + (1,) * extra + (3,)
        camera_origin = cam_o.view(camera_shape)
        screen_point = sp.view(camera_shape)
        screen_normal = sb[:, 2].view(camera_shape)
        rays = points - camera_origin
        depth = (rays * screen_normal).sum(-1, keepdim=True)
        screen_distance = ((screen_point - camera_origin) * screen_normal).sum(
            -1, keepdim=True
        )
        projected = camera_origin + (screen_distance / depth) * rays
        relative = projected - screen_point
        screen_x = sb[:, 0].view(camera_shape)
        screen_y = sb[:, 1].view(camera_shape)
        pixels = torch.stack(
            (
                (relative * screen_x).sum(-1),
                (relative * screen_y).sum(-1),
            ),
            dim=-1,
        )
        half_height = float(screen_height) / 2.0
        depth = depth.squeeze(-1)
        scale = (screen_distance.squeeze(-1) / depth).abs() * half_height
        return pixels * half_height, depth, scale

    def _guarded_pixel_error(
        self, exact, approximated, cam, front_sign, screen_height, slack=None
    ):
        """Guarded projected pixel deviation between matching point sets.

        ``exact`` and ``approximated`` are ``[K, ..., 3]``; ``cam``,
        ``front_sign`` and ``slack`` carry one row per leading element.

        ``slack`` is the world-space distance the *reference* surface is itself
        uncertain by -- the logical PN patch's own accuracy, which is
        ``geometry_tolerance`` carried forward to render time. It is projected
        at each sample's own depth and subtracted from the deviation measured
        there, so neither search resolves detail the reference does not have
        (see ``rt_settings.PN_GEOMETRY_SLACK``). ``None`` measures against the PN
        patch exactly.

        The stopping criterion these errors feed is a *primary visibility* one:
        keep subdividing until the flat stand-in lands within
        ``render_tolerance`` of the true surface, measured in output pixels.
        Projected pixel coordinates are unbounded, though -- geometry off to the
        side of the view axis, or approaching the camera plane, projects
        arbitrarily far outside the frame -- so the raw error is not usable as a
        stopping criterion on its own.  A sample pair is therefore ignored
        unless at least one of its two projections lands inside a guard box
        around the frame (see ``screen_guard_factor``), and the pair is clamped
        into that box before being compared.  Deviation that happens entirely
        off frame costs nothing; anything in or near frame keeps its exact
        error, so on-screen tessellation is unaffected.

        Without that guard, ``camera.orbit`` -- which swings the scene sideways
        without turning the camera -- drove levels up frame after frame to
        resolve geometry that had long since left the frame, until the trial
        tessellations alone exhausted render memory.

        A sample at or behind the camera plane has no finite screen position, so
        it cannot steer subdivision at all; it is dropped and the in-front
        samples decide.  Geometry straddling the plane still refines on its
        front half, whose near-plane projection is genuinely large.
        """
        guard = self.screen_guard_factor * float(screen_height)
        exact_pixels, exact_depth, exact_scale = self._project_to_output_pixels(
            exact, *cam, screen_height
        )
        approximated_pixels, approximated_depth, _ = self._project_to_output_pixels(
            approximated, *cam, screen_height
        )
        error = (
            exact_pixels.clamp(-guard, guard) - approximated_pixels.clamp(-guard, guard)
        ).norm(dim=-1)
        if slack is not None:
            allowance = slack.view(-1, *((1,) * (error.ndim - 1))) * exact_scale
            error = (error - allowance).clamp_min(0)
        sign = front_sign.view(-1, *((1,) * (error.ndim - 1)))
        usable = (
            torch.isfinite(error)
            & (exact_depth * sign > 1e-7)
            & (approximated_depth * sign > 1e-7)
            & (
                (exact_pixels.abs() <= guard).all(-1)
                | (approximated_pixels.abs() <= guard).all(-1)
            )
        )
        return torch.where(usable, error, torch.zeros_like(error))

    @staticmethod
    def _triangle_counts(levels):
        """``4 ** levels``, a patch's diced triangle count at a uniform dice.

        The frame budget is enforced *during* the level searches, before either
        knows whether the patch will be allowed a coarser across level, so this
        is what they bound themselves with: an upper bound on the count they
        will actually emit (``dice_triangle_count``, which equals this when the
        two levels agree and is smaller otherwise). Bounding by the larger
        figure can only refuse a promotion the frame could have afforded, never
        admit one it could not.
        """
        return torch.bitwise_left_shift(torch.ones_like(levels), 2 * levels)

    def _required_subdivision_levels(
        self,
        control_points,
        edge_controls,
        cam_o,
        sp,
        sb,
        screen_height,
        geometry_static=False,
        slack=None,
    ):
        """Choose the crack-free logical PN levels of every patch and edge.

        Returns per-patch interior levels ``[T, P]`` and per-edge boundary
        levels ``[T, P, 3]``, both of which vary freely from patch to patch and
        from frame to frame.

        ``slack`` is the per-frame world-space accuracy of the logical surface
        itself; see :meth:`_guarded_pixel_error`. It is a property of the frame,
        not of the patch or the edge, so the two copies of a shared boundary
        curve are handed the same value and still reach the same level.
        """
        # Which side of the camera plane is in front: the screen plane's own
        # side, exactly as the renderer's front test decides it.
        front_sign = torch.sign(((sp - cam_o) * sb[:, 2]).sum(-1))
        cam = (cam_o, sp, sb)
        kernel = _pn_criterion_inputs(
            control_points, edge_controls, cam_o, sp, sb, front_sign, slack
        )
        edge_levels, edge_capped = self._required_edge_levels(
            edge_controls, cam, front_sign, screen_height, kernel, slack
        )
        levels, patch_capped = self._required_patch_levels(
            control_points,
            edge_levels.amax(-1),
            cam,
            front_sign,
            screen_height,
            kernel,
            # The torch fallback re-evaluates a patch once per frame that is
            # still searching; the kernel keeps its samples in registers and
            # has nothing to share, so only the fallback wants the work list
            # grouped for dedup.
            geometry_static and kernel is None,
            slack,
        )
        apex, across = self._coarsest_across_levels(
            control_points,
            levels,
            edge_levels,
            cam,
            front_sign,
            screen_height,
            kernel,
            slack,
        )
        if edge_capped or patch_capped:
            warnings.warn(
                "Logical PN render tessellation reached its safety cap before "
                "meeting render_tolerance for every patch.",
                RuntimeWarning,
                stacklevel=3,
            )
        return levels, edge_levels, apex, across

    def _coarsest_across_levels(
        self,
        control_points,
        levels,
        edge_levels,
        cam,
        front_sign,
        screen_height,
        kernel=None,
        slack=None,
    ):
        """Choose each patch's dice apex and across level, ``[T, P]`` each.

        The interior level from :meth:`_required_patch_levels` says how finely
        the patch has to be cut in its *worst* direction. Cutting the other
        direction that finely is what a uniform grid does and what this undoes:
        the rows run parallel to the patch's coarsest boundary curve, and the
        dice is cut across them only as far as its own measured error demands.

        The starting point is the coarsest dice the seam allows -- that boundary
        curve's own level, since the dice must reproduce its polyline -- and the
        answer is always one the criterion has actually passed at, never one
        inferred from the boundary. A patch that gains nothing keeps the uniform
        grid it already resolved at, at no extra cost: the first probe's error
        predicts how far it would have to be refined, and a prediction that
        reaches the interior level ends the search there.
        """
        floor = edge_levels.gather(-1, edge_levels.argmin(-1, keepdim=True)).squeeze(-1)
        apex = _APEX_OF_EDGE.to(edge_levels.device)[edge_levels.argmin(-1)]
        across = levels.clone()
        if not rt_settings.PN_ANISOTROPIC_DICE or not levels.numel():
            return apex, across

        max_level = int(self.max_subdivision_level)
        threshold = self._pixel_threshold(screen_height)
        candidate = floor.clone()
        # A patch whose coarsest boundary is already as fine as its interior has
        # nothing to coarsen; everything else starts at that boundary's level.
        searching = candidate < levels

        for _ in range(max_level + 1):
            if not bool(searching.any()):
                break
            error = self._grouped_pattern_error(
                control_points,
                levels,
                candidate,
                apex,
                searching,
                cam,
                front_sign,
                screen_height,
                kernel,
                slack,
            )
            passed = (error * self._flatness_safety_factor) <= threshold
            across = torch.where(searching & passed, candidate, across)
            # Error falls as the square of the column count, so one measurement
            # says how many doublings are still missing. Overshooting to the
            # interior level costs nothing to check: that dice is the uniform
            # one, which has already passed.
            deficit = (
                error * self._flatness_safety_factor / max(threshold, 1e-30)
            ).clamp_min(1.0)
            steps = torch.ceil(torch.log2(deficit.sqrt())).to(candidate.dtype)
            candidate = torch.where(passed, candidate, candidate + steps.clamp_min(1))
            searching = searching & ~passed & (candidate < levels)
        return apex, across

    def _grouped_pattern_error(
        self,
        control_points,
        levels,
        across,
        apex,
        selected_mask,
        cam,
        front_sign,
        screen_height,
        kernel=None,
        slack=None,
    ):
        """Flatness error of one candidate dice per patch, ``[T, P]``.

        Patches sharing a dice shape are measured together -- there are at most
        a few dozen shapes in a frame -- so this is the same batched criterion
        the level ladder runs, once per distinct ``(along, across, apex)``.
        """
        device = control_points.device
        dtype = control_points.dtype
        error = torch.zeros_like(levels, dtype=dtype)
        keys = (levels * (self.max_subdivision_level + 1) + across) * 3 + apex
        for key in torch.unique(keys[selected_mask]).tolist():
            group = selected_mask & (keys == key)
            selected = group.nonzero()
            if not selected.shape[0]:
                continue
            apex_of_key = int(key % 3)
            across_of_key = int((key // 3) % (self.max_subdivision_level + 1))
            along_of_key = int(key // (3 * (self.max_subdivision_level + 1)))
            group_error = self._patch_flatness_error(
                control_points,
                selected,
                dice_pattern(
                    along_of_key,
                    across_of_key,
                    apex_of_key,
                    device=device,
                    dtype=dtype,
                ),
                cam,
                front_sign,
                screen_height,
                kernel,
                False,
                slack,
            )
            error[selected[:, 0], selected[:, 1]] = group_error
        return error

    def _required_edge_levels(
        self, edge_controls, cam, front_sign, screen_height, kernel=None, slack=None
    ):
        """Per-boundary-curve subdivision levels, shape ``[T, P, 3]``.

        Each curve is judged on its canonically oriented cubic and nothing else
        (see
        :func:`~algan.rendering.logical_pn.logical_pn_edge_control_points`), so
        the two patches sharing a curve reach the same answer by identical
        arithmetic -- which is what lets them dice independently and still meet
        along the seam.

        A promotion is refused once it would break ``max_diced_triangles`` for
        the frame it belongs to, using ``4 ** max(edge levels)`` per patch as
        the lower bound on that frame's diced triangle count.  The refusal is
        taken per frame, which keeps the two copies of a shared curve in step.
        """
        device = edge_controls.device
        dtype = edge_controls.dtype
        num_frames, num_patches = edge_controls.shape[0], edge_controls.shape[1]
        max_level = int(self.max_subdivision_level)
        budget = max(1, int(self.max_diced_triangles))
        threshold = self._pixel_threshold(screen_height)

        levels = torch.zeros(
            (num_frames, num_patches, 3), dtype=torch.long, device=device
        )
        samples = _sample_tensor(self._edge_sample_parameters, device, dtype)
        active = torch.arange(levels.numel(), device=device)
        capped = False

        for level in range(max_level + 1):
            if active.numel() == 0:
                break
            error = self._edge_chord_error(
                edge_controls,
                active,
                level,
                cam,
                front_sign,
                samples,
                screen_height,
                kernel,
                slack,
            )
            candidates = active[(error * self._flatness_safety_factor) > threshold]
            if candidates.numel() == 0:
                break
            if level == max_level:
                capped = True
                break
            proposed = levels.clone()
            proposed.view(-1)[candidates] = level + 1
            blocked = self._triangle_counts(proposed.amax(-1)).sum(1) > budget
            frames, _patches, _edges = self._unravel_edges(candidates, num_patches)
            promoted = candidates[~blocked[frames]]
            capped = capped or bool(promoted.numel() != candidates.numel())
            levels.view(-1)[promoted] = level + 1
            active = promoted
        return levels, capped

    @staticmethod
    def _unravel_edges(flat_indices, num_patches):
        """Split flat ``[T, P, 3]`` edge indices into frame/patch/edge."""
        frames = torch.div(flat_indices, num_patches * 3, rounding_mode="floor")
        within = flat_indices - frames * (num_patches * 3)
        patches = torch.div(within, 3, rounding_mode="floor")
        return frames, patches, within - patches * 3

    def _edge_chord_error(
        self,
        edge_controls,
        active,
        level,
        cam,
        front_sign,
        samples,
        screen_height,
        kernel=None,
        slack=None,
    ):
        """Peak pixel deviation of each active curve from its chord polyline.

        The polyline has ``2 ** level`` chords; every chord is compared against
        the curve at ``_edge_sample_parameters``.  Work is streamed in chunks so
        scratch stays inside ``max_scratch_triangles`` however many curves are
        still looking for a level -- the fused kernel keeps its intermediates in
        registers and needs no chunking at all.
        """
        device = edge_controls.device
        dtype = edge_controls.dtype
        num_patches = edge_controls.shape[1]
        segments = 1 << level
        num_samples = samples.numel()
        if kernel is not None:
            error = torch.zeros(active.numel(), device=device, dtype=dtype)
            if active.numel():
                pn_edge_chord_error(
                    kernel.edge_controls,
                    kernel.edge_stride,
                    active.to(torch.int32).contiguous(),
                    samples,
                    kernel.cam_origin,
                    kernel.screen_point,
                    kernel.screen_basis,
                    kernel.front_sign,
                    kernel.slack,
                    error,
                    num_patches,
                    segments,
                    float(screen_height) / 2.0,
                    self.screen_guard_factor * float(screen_height),
                )
            return error
        chunk = max(
            1,
            int(self.max_scratch_triangles) // max(1, segments * num_samples),
        )
        # Knot and sample parameters are evaluated in one pass: at the low
        # levels almost every mesh settles on, the launch overhead of a second
        # pass over a few points per curve is the whole cost.
        steps = torch.arange(segments, device=device, dtype=dtype).unsqueeze(-1)
        parameters = torch.cat(
            (
                torch.arange(segments + 1, device=device, dtype=dtype) / segments,
                ((steps + samples.unsqueeze(0)) / segments).reshape(-1),
            )
        )
        blend = samples.view(1, 1, num_samples, 1)

        error = torch.empty(active.numel(), device=device, dtype=dtype)
        for start in range(0, active.numel(), chunk):
            selected = active[start : start + chunk]
            frames, patches, edges = self._unravel_edges(selected, num_patches)
            curve = evaluate_cubic_curve(
                edge_controls[frames, patches, edges], parameters
            )
            knots = curve[:, : segments + 1]
            exact = curve[:, segments + 1 :].reshape(-1, segments, num_samples, 3)
            chords = (
                knots[:, :-1].unsqueeze(2) * (1.0 - blend)
                + knots[:, 1:].unsqueeze(2) * blend
            )
            error[start : start + chunk] = self._guarded_pixel_error(
                exact,
                chords,
                tuple(value.index_select(0, frames) for value in cam),
                front_sign.index_select(0, frames),
                screen_height,
                None if slack is None else slack.index_select(0, frames),
            ).amax(dim=(1, 2))
        return error

    def _required_patch_levels(
        self,
        control_points,
        start,
        cam,
        front_sign,
        screen_height,
        kernel=None,
        share_patches=False,
        slack=None,
    ):
        """Per-patch interior subdivision levels, shape ``[T, P]``.

        Every patch starts at the largest of its three boundary levels -- the
        floor imposed by the snap in
        :func:`~algan.rendering.logical_pn.snap_boundary_values` -- and climbs
        only while its *own* dice misses the frame's budget
        (:meth:`_pixel_threshold`).  Because the
        active set shrinks as patches resolve, the whole search costs about a
        third more than the tessellation it settles on, rather than one full
        trial tessellation of the entire mesh per level tried.

        The criterion measures the *unsnapped* dice.  Folding the boundary snap
        in instead would be measuring against a floor the interior cannot get
        under -- the snap displacement is fixed by the boundary level, and is
        itself allowed to reach the budget -- so patches whose boundary
        resolved just inside the budget would climb to the safety cap
        without ever passing.  The two approximations are held to the tolerance
        separately (see the class docstring).
        """
        max_level = int(self.max_subdivision_level)
        budget = max(1, int(self.max_diced_triangles))
        threshold = self._pixel_threshold(screen_height)
        dtype = control_points.dtype
        levels = start.clone()
        if levels.numel() == 0:
            return levels, False
        unresolved = torch.ones_like(levels, dtype=torch.bool)
        # Accumulated on the device and read back once: a per-iteration
        # ``.any()`` would stall the queue at every level for a flag that only
        # decides whether to warn.
        capped = torch.zeros((), dtype=torch.bool, device=levels.device)

        for level in range(int(levels.amin().item()), max_level + 1):
            trying = unresolved & (levels == level)
            # PATCH-major when the criterion can share a patch's evaluation
            # between the frames still trying it (see ``_patch_flatness_error``);
            # the rows are independent, so this only decides who lands in a
            # chunk together.
            selected = (
                trying.t().contiguous().nonzero().flip(-1)
                if share_patches
                else trying.nonzero()
            )
            if not selected.shape[0]:
                if not bool(unresolved.any()):
                    break
                continue
            frames, patches = selected[:, 0], selected[:, 1]
            error = self._patch_flatness_error(
                control_points,
                selected,
                dice_pattern(level, level, 0, device=levels.device, dtype=dtype),
                cam,
                front_sign,
                screen_height,
                kernel,
                share_patches,
                slack,
            )
            failed = (error * self._flatness_safety_factor) > threshold
            if level == max_level:
                capped = capped | failed.any()
                break
            # Promote only where the frame still fits its triangle budget. The
            # whole step is written with masks rather than by splitting
            # ``selected`` into resolved/failed/frozen subsets: each such split
            # is a device synchronisation, and on a mesh that resolves at the
            # first level they cost more than the criterion itself.
            proposed = levels.clone()
            proposed[frames, patches] = torch.where(
                failed, level + 1, levels[frames, patches]
            )
            blocked = self._triangle_counts(proposed).sum(1) > budget
            promoted = failed & ~blocked[frames]
            capped = capped | (failed & ~promoted).any()
            levels[frames, patches] = torch.where(
                promoted, level + 1, levels[frames, patches]
            )
            unresolved[frames, patches] = promoted
            if not bool(unresolved.any()):
                break
        return levels, bool(capped)

    def _patch_flatness_error(
        self,
        control_points,
        selected,
        pattern,
        cam,
        front_sign,
        screen_height,
        kernel=None,
        share_patches=False,
        slack=None,
    ):
        """Peak pixel deviation of each selected patch's dice under ``pattern``,
        sampled at ``_flatness_sample_weights`` within every microtriangle.

        The criterion knows nothing about the shape of the dice beyond the list
        of microtriangles ``pattern`` hands it, which is what lets an
        anisotropic candidate be judged by exactly the arithmetic that judged
        the uniform one.
        """
        device = control_points.device
        dtype = control_points.dtype
        vertex_uv = pattern.vertex_uv
        triangle_indices = pattern.triangle_indices
        corner_uv = vertex_uv[triangle_indices]
        weights = _sample_tensor(self._flatness_sample_weights, device, dtype)
        if kernel is not None:
            # The kernel evaluates the patch at each microtriangle's own corners
            # rather than at the shared subdivision vertices: a vertex is
            # revisited by up to six threads, which is cheaper than the scratch
            # buffer sharing it would need. ``subdivision_triangle_uvs`` is
            # ``vertex_uv`` gathered through ``triangle_indices``, so the
            # parameters -- and hence the points -- are the same either way.
            error = torch.zeros(selected.shape[0], device=device, dtype=dtype)
            if selected.shape[0]:
                pn_patch_flatness_error(
                    kernel.control_points,
                    kernel.control_stride,
                    # ``nonzero`` hands back a transposed view on CUDA, which a
                    # Taichi ndarray cannot take.
                    selected.to(torch.int32).contiguous(),
                    corner_uv.contiguous(),
                    weights,
                    kernel.cam_origin,
                    kernel.screen_point,
                    kernel.screen_basis,
                    kernel.front_sign,
                    kernel.slack,
                    error,
                    float(screen_height) / 2.0,
                    self.screen_guard_factor * float(screen_height),
                )
            return error
        sample_uv = torch.einsum("sk,mka->msa", weights, corner_uv)
        # The dice's own vertices and the interior sample points are evaluated
        # in ONE pass over the concatenated parameters, as the edge criterion
        # already does. The patch expression is elementwise in uv -- the same
        # ten-term polynomial per parameter, whatever shape it arrives in -- so
        # this is the identical arithmetic on the identical values, and at the
        # low levels most meshes settle on, a second launch over a handful of
        # points per patch was half the cost of the whole level search.
        flat_sample_uv = sample_uv.reshape(-1, 2)
        num_vertices = vertex_uv.shape[0]
        combined_uv = torch.cat((vertex_uv, flat_sample_uv))
        chunk = max(
            1,
            int(self.max_scratch_triangles) // max(1, combined_uv.shape[0]),
        )

        error = torch.empty(selected.shape[0], device=device, dtype=dtype)
        for start in range(0, selected.shape[0], chunk):
            rows = selected[start : start + chunk]
            frames, patches = rows[:, 0], rows[:, 1]
            # What the criterion asks of the patch -- where its surface and its
            # level-``level`` dice sit in space -- has nothing to do with the
            # camera; only the projection that follows does. So a patch several
            # frames are still trying at this level is evaluated once and its
            # points fanned out, which is why ``_required_patch_levels`` hands
            # the rows over patch-major (see ``share_patches``).
            shared = _PatchChunk.of(patches, frames, share_patches)
            controls, deduped = shared.rows_of(control_points, share_patches)
            evaluated = evaluate_logical_pn(controls.unsqueeze(0), combined_uv)[0]
            vertices = evaluated[:, :num_vertices]
            # The sample points stay in their flat (microtriangle, sample)
            # order and the approximation is flattened to match, rather than
            # the exact points being reshaped back to [.., m, s, ..] -- that
            # reshape is a copy of a strided slice, and the comparison below is
            # elementwise with a max over both axes either way.
            approximated = torch.einsum(
                "sk,pmkc->pmsc", weights, vertices[:, triangle_indices]
            )
            error[start : start + chunk] = self._guarded_pixel_error(
                shared.fan_out(evaluated[:, num_vertices:], deduped),
                shared.fan_out(approximated.flatten(1, 2), deduped),
                tuple(value.index_select(0, frames) for value in cam),
                front_sign.index_select(0, frames),
                screen_height,
                None if slack is None else slack.index_select(0, frames),
            ).amax(dim=1)
        return error

    @staticmethod
    def _expanded_frames(value, num_frames, name):
        if value is None:
            return None
        if value.shape[0] not in (1, num_frames):
            raise ValueError(
                f"{name} has {value.shape[0]} frames, expected 1 or {num_frames}"
            )
        return _expand_frames(value, num_frames)

    @staticmethod
    def _collapse_redundant_frames(value):
        """Drop a source array's frame axis when every frame holds the same
        values, returning ``(array, frame_invariant)``.

        Materialization hands a mob's attributes back one row per frame even
        when the mob does not move, so a *static* mesh arrives here as ``T``
        byte-identical copies of one geometry.  Nothing downstream of this
        method needs the copies: the control nets are a function of the source
        alone, so building them ``T`` times produces ``T`` identical answers,
        the criterion kernels then upload ``T`` copies of a net they index by
        one frame, and the dice evaluates each patch once per frame to the same
        point.  Collapsing here is what lets all three notice.

        The test is an equality reduction over the source (cheap next to the
        dice, which is quadratically larger), and it is deliberately
        conservative: NaN never compares equal, so a mesh carrying one falls
        back to the per-frame path rather than being silently unified.
        """
        if value is None:
            return None, False
        if value.shape[0] == 1 or value.stride(0) == 0:
            return value[:1], True
        # Frame 1 first: a mesh that really is deforming differs there, and
        # that one comparison rejects it for a (T-1)th of what comparing the
        # whole batch costs.
        if not bool((value[1] == value[0]).all()):
            return value, False
        if value.shape[0] > 2 and not bool((value[2:] == value[:1]).all()):
            return value, False
        return value[:1], True

    def _dice_logical_pn(self, camera):
        num_frames = int(camera.ray_origin.shape[0])
        source_corners = self.corners.float()
        source_normals = self.normals.float()
        # Corners and normals collapse together or not at all: a control net is
        # built from both at once, so collapsing one of a pair would only
        # broadcast straight back out (and the stacks inside
        # ``logical_pn_normal_control_points`` mix collapsed and per-frame
        # terms, which do not stack). Both invariant is also exactly the
        # condition under which the dice may reuse a patch across frames.
        collapsed_corners, geometry_static = self._collapse_redundant_frames(
            source_corners
        )
        if geometry_static:
            collapsed_normals, geometry_static = self._collapse_redundant_frames(
                source_normals
            )
            if geometry_static:
                source_corners, source_normals = collapsed_corners, collapsed_normals
        device = source_corners.device
        dtype = source_corners.dtype
        cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)), num_frames).to(
            device
        )
        sp = _expand_frames(_flat_frames(camera.screen_point, (3,)), num_frames).to(
            device
        )
        sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)), num_frames).to(
            device
        )

        # Control nets are built on the source frames and only broadcast
        # afterwards, so a static mesh keeps one copy however many frames the
        # batch covers; the per-frame views below are indexed, never
        # materialized.
        control_points = self._expanded_frames(
            logical_pn_control_points(source_corners, source_normals),
            num_frames,
            "logical PN corners",
        )
        normal_control_points = self._expanded_frames(
            logical_pn_normal_control_points(source_corners, source_normals),
            num_frames,
            "logical PN normals",
        )
        edge_controls = self._expanded_frames(
            logical_pn_edge_control_points(source_corners, source_normals),
            num_frames,
            "logical PN edges",
        )
        output_height = getattr(camera, "output_screen_height", camera.screen_height)
        # How far this mesh's logical surface may itself be from the surface it
        # approximates, in world units, at the size it is being rendered at.
        # The level searches subtract its projection from what they measure.
        slack = None
        if rt_settings.PN_GEOMETRY_SLACK and self.geometry_slack_ratio > 0:
            slack = _expand_frames(
                mean_patch_edge_length(source_corners) * self.geometry_slack_ratio,
                num_frames,
            )
        levels, edge_levels, apex_levels, across_levels = (
            self._required_subdivision_levels(
                control_points,
                edge_controls,
                cam_o,
                sp,
                sb,
                output_height,
                geometry_static,
                slack,
            )
        )

        # Each frame packs its patches back to back at their own diced sizes;
        # only the batch's widest frame sets the padded width. A frame that
        # needs a fraction of the detail no longer pays for the frame that
        # needs the most, and neither does a patch for its neighbours.
        counts = dice_triangle_count(levels, across_levels)
        offsets = counts.cumsum(1) - counts
        max_triangles = int(counts.sum(1).amax().item()) if counts.numel() else 0

        # Per-frame triangle -> source-surface ids. A diced row's patch changes
        # from frame to frame with the adaptive levels, so unlike a flat
        # primitive's [1, N] ids this must be per frame: row c of frame f
        # belongs to the patch whose [offset, end) span contains c. Padding
        # tail rows (c >= the frame's total) clamp to the last patch; they are
        # alpha-zero and never emit a fragment.
        num_patches = counts.shape[1] if counts.ndim > 1 else 0
        counts_src = getattr(self, "_rt_obj_counts", None)
        # Same three sources in the same order as the flat path in
        # ``_pack_projected_flat_geometry``, most specific first: the members'
        # own ``mesh_key``/``mesh_ids`` declaration, then the per-member counts.
        # Consulting the declaration here is what makes it reach a DICED
        # surface at all -- ``_pack_projected_flat_geometry`` gives ``pn_obj``
        # priority over ``_rt_obj_ids``, so whatever this builds is final. A
        # packed-grid ``Surface`` is one member covering every packed sphere, so
        # without it the whole pack dices to a single surface id and the
        # per-grid ``mesh_ids`` that ``Surface.get_render_primitives`` stamps
        # are read by nothing. A logical-PN member's ``mesh_ids`` are per PATCH
        # (its ``corners`` are patch corners), which is the granularity wanted
        # here; the searchsorted below carries them to the diced rows.
        obj_ids = getattr(self, "_rt_obj_ids", None) if rt_settings.MESH_ID else None
        if obj_ids is not None:
            patch_source = obj_ids.reshape(-1).to(device=device, dtype=torch.int32)
            self._logical_pn_tri_obj_n = int(self._rt_obj_ids_n)
        elif counts_src:
            patch_source = torch.repeat_interleave(
                torch.arange(len(counts_src), dtype=torch.int32, device=device),
                torch.tensor(counts_src, dtype=torch.int64, device=device),
            )
            self._logical_pn_tri_obj_n = len(counts_src)
        else:
            patch_source = torch.zeros((num_patches,), dtype=torch.int32, device=device)
            self._logical_pn_tri_obj_n = 1
        if patch_source.shape[0] != num_patches:
            raise RuntimeError(
                "logical PN patch/source mismatch: "
                f"{patch_source.shape[0]} vs {num_patches} patches"
            )
        # Every patch of a single-surface mesh carries id 0, so the row -> patch
        # -> surface resolution below has one possible answer. Short-circuit it:
        # the searchsorted runs over the whole padded [T, max_triangles] grid,
        # which on a dense mesh is the largest single allocation the dice makes
        # outside the diced arrays themselves.
        if self._logical_pn_tri_obj_n == 1:
            self._logical_pn_tri_obj = torch.zeros(
                (num_frames, max_triangles), dtype=torch.int32, device=device
            )
        elif num_patches and max_triangles:
            ends = counts.cumsum(1)
            cols = torch.arange(max_triangles, device=device)
            patch_of_col = torch.searchsorted(
                ends.contiguous(),
                cols.unsqueeze(0).expand(num_frames, -1).contiguous(),
                right=True,
            ).clamp_max(num_patches - 1)
            self._logical_pn_tri_obj = (
                patch_source[patch_of_col].to(torch.int32).contiguous()
            )
        else:
            self._logical_pn_tri_obj = torch.zeros(
                (num_frames, max_triangles), dtype=torch.int32, device=device
            )

        colors = self._expanded_frames(
            self.colors.float(), num_frames, "logical PN colors"
        )
        surface_sources = {
            name: self._expanded_frames(
                getattr(self, name), num_frames, f"logical PN {name}"
            )
            for name in self._surface_params
        }
        shader_sources = [
            self._expanded_frames(value, num_frames, "logical PN shader parameter")
            for value in self.shader_param_values
        ]
        uv_source = self._expanded_frames(self.uvs, num_frames, "logical PN UVs")

        def allocate(values):
            return torch.zeros(
                (
                    num_frames,
                    max_triangles,
                    3,
                    values.shape[-1],
                ),
                device=values.device,
                dtype=values.dtype,
            )

        diced_corners = allocate(source_corners)
        diced_normals = allocate(source_normals)
        diced_colors = allocate(colors)
        diced_surface_params = {
            name: allocate(source) for name, source in surface_sources.items()
        }
        diced_shader_params = [allocate(v) for v in shader_sources]
        diced_uvs = allocate(uv_source) if uv_source is not None else None
        # Every attribute the dice writes, paired with the source it reads.
        # Attributes differ from one another only in width, so the loop below
        # treats them as one list.
        attribute_outputs = [(diced_colors, colors)]
        attribute_outputs += [
            (diced_surface_params[name], source)
            for name, source in surface_sources.items()
        ]
        attribute_outputs += list(zip(diced_shader_params, shader_sources))
        if diced_uvs is not None:
            attribute_outputs.append((diced_uvs, uv_source))
        # The rows a frame never writes are its tail: the frame's patches are
        # packed back to back from column 0, so column c is padding exactly
        # when it is past that frame's diced total. Stating it that way costs
        # one comparison over the grid, where marking the written rows costs a
        # fill plus an index_fill_ per chunk.
        frame_totals = counts.sum(1)
        batch_columns = torch.arange(max_triangles, device=device)
        padding = batch_columns >= frame_totals.unsqueeze(1)

        # One group per distinct dice shape, not per level: two patches at the
        # same interior level dice differently when one of them was allowed to
        # stay coarse across (see ``_coarsest_across_levels``).
        shape_keys = (
            levels * (self.max_subdivision_level + 1) + across_levels
        ) * 3 + apex_levels
        for key in shape_keys.unique(sorted=True).tolist():
            key = int(key)
            pattern = dice_pattern(
                key // (3 * (self.max_subdivision_level + 1)),
                (key // 3) % (self.max_subdivision_level + 1),
                key % 3,
                device=device,
                dtype=dtype,
            )
            # PATCH-major *only where the dedup can use it*: ``nonzero`` on the
            # transpose lists every frame of a patch consecutively, which is
            # what lets the dedup below see its duplicates inside one chunk (a
            # frame-major list puts a patch's frames a whole frame apart, so on
            # any mesh wider than a chunk no two would ever share one). It is
            # not free -- consecutive rows then write a frame apart in the
            # output instead of in one run, and read their control points the
            # same way -- so a deforming mesh, which has nothing to dedup,
            # stays frame-major and keeps its contiguous writes. Measured: 1.03x
            # against 0.97x on the deforming half of a real scene's dice calls.
            # The writes are disjoint and ``index_copy_`` does not care about
            # order, so the diced output is the same either way.
            trying = shape_keys == key
            selected = (
                trying.t().contiguous().nonzero().flip(-1)
                if geometry_static
                else trying.nonzero()
            )
            vertex_uv = pattern.vertex_uv
            triangle_indices = pattern.triangle_indices
            boundary = pattern.boundary
            num_triangles = triangle_indices.shape[0]
            columns = torch.arange(num_triangles, device=device)
            chunk = max(1, int(self.max_scratch_triangles) // num_triangles)

            for start in range(0, selected.shape[0], chunk):
                rows = selected[start : start + chunk]
                frames, patches = rows[:, 0], rows[:, 1]
                edges = edge_levels[frames, patches]

                # A patch's diced geometry depends on the patch and its level,
                # never on the camera -- the camera only picks the level. So
                # every frame that dices a patch at this level asks for exactly
                # the same points, and where the source is frame invariant (a
                # mesh that does not deform during the batch, which is most of
                # them) the evaluation is one answer repeated once per frame.
                # Evaluate it once per distinct patch and fan the rows out with
                # a gather: one pass, against the twenty-odd a patch evaluation
                # costs.
                chunk_rows = _PatchChunk.of(patches, frames, geometry_static)

                # The patch is evaluated once per shared subdivision vertex
                # (each is a corner of up to six microtriangles), snapped onto
                # its boundary polylines, and only then expanded to the
                # triangle-soup layout the packed geometry wants. The snap
                # happens after the fan-out: it is a function of the boundary
                # levels, which two frames dicing the same patch need not
                # share.
                controls, deduped = chunk_rows.rows_of(control_points, geometry_static)
                positions = snap_boundary_values(
                    chunk_rows.fan_out(
                        evaluate_logical_pn(controls.unsqueeze(0), vertex_uv)[0],
                        deduped,
                    ),
                    pattern.edge_levels,
                    edges,
                    boundary,
                )
                normal_controls, deduped = chunk_rows.rows_of(
                    normal_control_points, geometry_static
                )
                vertex_normals = F.normalize(
                    snap_boundary_values(
                        chunk_rows.fan_out(
                            evaluate_logical_pn_normals(
                                normal_controls.unsqueeze(0), vertex_uv
                            )[0],
                            deduped,
                        ),
                        pattern.edge_levels,
                        edges,
                        boundary,
                    ),
                    p=2,
                    dim=-1,
                )
                # Each selected (frame, patch) writes a *contiguous run* of
                # columns, which a two-index advanced-index scatter cannot
                # exploit -- it lowers to an ``index_put_`` over a
                # ``[chunk, num_triangles]`` destination. Folding (frame,
                # column) into one row index makes every write a single
                # ``index_copy_`` over the flattened ``[T * M, ...]`` output.
                targets = (
                    frames.unsqueeze(1) * max_triangles
                    + offsets[frames, patches].unsqueeze(1)
                    + columns
                ).reshape(-1)

                _scatter_diced_rows(
                    diced_corners, positions[:, triangle_indices], targets
                )
                _scatter_diced_rows(
                    diced_normals, vertex_normals[:, triangle_indices], targets
                )
                for output, source in attribute_outputs:
                    _scatter_diced_rows(
                        output,
                        chunk_rows.diced_attribute(source, vertex_uv, triangle_indices),
                        targets,
                    )

        self.corners = diced_corners
        self.normals = diced_normals
        self.colors = diced_colors
        for name, values in diced_surface_params.items():
            setattr(self, name, values)
        self.shader_param_values = diced_shader_params
        self.uvs = diced_uvs
        self._logical_pn_padding = padding
        self._logical_pn_subdivision_levels = levels
        self._logical_pn_edge_levels = edge_levels
        self._logical_pn_across_levels = across_levels
        self._logical_pn_apex = apex_levels
        self._logical_pn_triangle_counts = counts

    def project_to_screen(self, camera, light_sources):
        self._dice_logical_pn(camera)
        self._shade_vertex_colors(camera, light_sources)
        padding = self._logical_pn_padding
        if bool(padding.any()):
            self.colors[..., -1] = torch.where(
                padding.unsqueeze(-1),
                torch.zeros_like(self.colors[..., -1]),
                self.colors[..., -1],
            )
        return self._pack_projected_flat_geometry(camera)


def _evaluate_cubic_bezier_batch(p, t):
    """p: [..., 4, 3] control points, t: broadcastable parameter in [0, 1)."""
    mt = 1.0 - t
    return (
        (mt * mt * mt) * p[..., 0, :]
        + (3.0 * mt * mt * t) * p[..., 1, :]
        + (3.0 * mt * t * t) * p[..., 2, :]
        + (t * t * t) * p[..., 3, :]
    )


def _evaluate_cubic_bezier_derivative_batch(p, t):
    """Evaluate the derivative of cubic control points ``p`` at ``t``."""
    mt = 1.0 - t
    return 3.0 * (
        (mt * mt) * (p[..., 1, :] - p[..., 0, :])
        + (2.0 * mt * t) * (p[..., 2, :] - p[..., 1, :])
        + (t * t) * (p[..., 3, :] - p[..., 2, :])
    )


def _uniform_cubic_subcurves(corners, num_subdivisions):
    """Return the exact world-space controls of uniform cubic subcurves.

    ``corners`` is ``[T, S, 4, 3]`` and the result is
    ``[T, S, num_subdivisions, 4, 3]``.  Endpoint positions and derivatives
    determine the four controls of each restricted cubic exactly.
    """
    p = corners.unsqueeze(-3)
    t0 = (
        torch.arange(num_subdivisions, device=corners.device, dtype=corners.dtype)
        / num_subdivisions
    )
    t0 = t0.view(1, 1, -1, 1)
    t1 = t0 + 1.0 / num_subdivisions
    q0 = _evaluate_cubic_bezier_batch(p, t0)
    q3 = _evaluate_cubic_bezier_batch(p, t1)
    derivative_scale = 1.0 / (3.0 * num_subdivisions)
    q1 = q0 + derivative_scale * _evaluate_cubic_bezier_derivative_batch(p, t0)
    q2 = q3 - derivative_scale * _evaluate_cubic_bezier_derivative_batch(p, t1)
    return torch.stack((q0, q1, q2, q3), dim=-2)


def _packed_uniform_cubic_parameters(chord_counts, dtype, vertex_counts=None):
    """The exact ``k / n`` parameters used for packed polyline vertices.

    ``vertex_counts`` defaults to ``chord_counts``, which samples ``k < n`` only
    -- the cubic's final endpoint is supplied by the next segment's first
    vertex.  A segment that closes an open subpath does not share its endpoint
    with anything and asks for one extra vertex, giving it ``k == n`` (``t = 1``)
    as well.
    """
    if vertex_counts is None:
        vertex_counts = chord_counts
    repeated_counts = torch.repeat_interleave(chord_counts, vertex_counts)
    return batch_arange(vertex_counts).to(dtype) / repeated_counts.to(dtype)


def _point_to_segment_distance_squared(point, start, delta, length_squared):
    """Squared distance from ``point`` to the finite segment ``start+delta``."""
    along = ((point - start) * delta).sum(-1, keepdim=True)
    along = along / length_squared.clamp_min(1e-20)
    closest = start + along.clamp_(0.0, 1.0) * delta
    return (point - closest).square().sum(-1)


def _bezier_connection_visibility(corners, next_segment_inds):
    """Whether each selected segment connection is authored geometry.

    Discontinuous connections are synthesized only to close a fill contour and
    therefore must not contribute to the visible border.
    """
    (corners, next_segment_inds), _ = _unify_time(
        [corners, next_segment_inds.unsqueeze(-1)], "bezier connections"
    )
    next_segment_inds = next_segment_inds.squeeze(-1)
    segment_ends = corners[..., 3, :]
    segment_starts = corners[..., 0, :]
    gather_inds = next_segment_inds.unsqueeze(-1).expand(-1, -1, 3)
    next_starts = torch.gather(segment_starts, 1, gather_inds)
    return (segment_ends - next_starts).norm(p=2, dim=-1) <= 1e-5


def _circuit_parity_gathered(qx, qy, ex0, ey0, ex1, ey1, valid):
    """Even-odd crossing parity of each query against its gathered edge set.

    ``qx``/``qy`` are ``[T, Q]`` query points; the edge tensors are the
    query's own circuit's edges gathered to ``[T, Q, K]`` with a ``[Q, K]``
    slot-validity mask (padding slots duplicate a real edge of the circuit
    and are masked out of the count). The predicate is exactly the kernel's
    (``_bezier_point_metrics``): a +x ray, ``(y0 > v) != (y1 > v)`` and
    ``x_cross > u`` -- degenerate 1e9 edges can never satisfy the y-straddle.
    Returns ``[T, Q]`` bool (odd = inside). The parity is an exact integer
    count over per-pair-identical arithmetic, so it is bit-identical to a
    masked probe of all page edges over the same (query, edge) pairs.
    """
    v = qy.unsqueeze(-1)  # [T, Q, 1]
    straddle = (ey0 > v) != (ey1 > v)
    denom = ey1 - ey0
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    x_cross = ex0 + (v - ey0) * (ex1 - ex0) / denom
    hit = straddle & (x_cross > qx.unsqueeze(-1)) & valid.unsqueeze(0)
    return hit.sum(-1) % 2 == 1


def _circuit_edge_inward_signs(edges, vert_circuit):
    """Per-edge inward sign sigma in {-1, 0, +1} for ``edges`` [T, V, >=4].

    Probes the crossing parity just off each edge midpoint on both sides
    (``mid +/- eps * |e| * leftward_normal``). The definitional invariant --
    the two sides of an edge have opposite parity -- doubles as the validity
    check: where it fails (the probe crossed another feature, reachable at
    sub-pixel stems) the eps is halved and retried, and a still-inconsistent
    edge gets sigma 0, which the kernel reads as "fall back to the single
    half-plane". Even-odd holes come out right by construction: parity IS the
    fill rule, so a hole's contour gets signs pointing out of the hole
    regardless of its winding.

    Each query is probed against its own circuit's edges only, gathered per
    circuit (CSR over the sorted edge ids): O(T * sum_c Vc^2) instead of the
    all-page-edges O(T * V^2) probe -- orders of magnitude on a text page of
    many small glyph contours. Queries are bucketed by circuit-size class so
    a page's one big contour does not set the gather width for every glyph,
    and chunked to bound the [T, Q, K] scratch. Bit-identical to the masked
    full probe: the same (query, edge) pairs are evaluated with the same
    arithmetic, and the crossing parity is an exact integer count. Lands in
    animate/prep; only run when the wedge is live.
    """
    T, V = edges.shape[0], edges.shape[1]
    device = edges.device
    if V == 0:
        return torch.zeros((T, 0), device=device)
    ex0, ey0 = edges[..., 0], edges[..., 1]
    ex1, ey1 = edges[..., 2], edges[..., 3]
    mx = 0.5 * (ex0 + ex1)
    my = 0.5 * (ey0 + ey1)
    dx = ex1 - ex0
    dy = ey1 - ey0
    length = torch.sqrt(dx * dx + dy * dy)
    degen = (length < 1e-12) | (edges[..., :4].abs() >= 1e8).any(-1)
    inv_len = 1.0 / torch.clamp(length, min=1e-12)
    # Leftward perpendicular of the edge direction, unit length.
    lnx = -dy * inv_len
    lny = dx * inv_len
    circ = vert_circuit.to(device)

    # Edge ids grouped by circuit (circuits need not be contiguous in V).
    order = torch.argsort(circ, stable=True)
    counts_all = torch.bincount(circ, minlength=int(circ.max()) + 1)
    starts_all = torch.cumsum(counts_all, 0) - counts_all
    edge_start = starts_all[circ]  # [V] own circuit's start in sorted order
    edge_count = counts_all[circ]  # [V] own circuit's edge count
    # Power-of-two size class per edge: within a bucket the gather width is
    # at most 2x any member's circuit size, keeping total work within 2x of
    # sum_c Vc^2.
    size_class = torch.ceil(torch.log2(edge_count.to(torch.float64))).to(torch.long)

    sigma = torch.zeros((T, V), device=device)
    unresolved = ~degen
    # Six halvings reach eps ~1.6e-3 of the edge length: a stem's two long
    # walls sit a fraction of the EDGE LENGTH apart (a 3-unit wall on a
    # 0.02-unit stem needs eps*|e| < 0.01), and an unresolved wall falls back
    # to the single half-plane exactly where the wedge was meant to help.
    eps = 0.05
    budget = 4_000_000
    for _attempt in range(6):
        idx = unresolved.any(0).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        for cls in torch.unique(size_class[idx]):
            idx_b = idx[size_class[idx] == cls]
            K = int(edge_count[idx_b].max())
            chunk = max(1, budget // max(T * K, 1))
            for start in range(0, idx_b.numel(), chunk):
                sel = idx_b[start : start + chunk]
                cnt = edge_count[sel]  # [Q]
                slots = torch.arange(K, device=device)  # [K]
                slot_valid = slots.view(1, -1) < cnt.view(-1, 1)  # [Q, K]
                # Padding slots re-read the circuit's last edge and are
                # masked out of the parity count by slot_valid.
                gidx = edge_start[sel].view(-1, 1) + torch.minimum(
                    slots.view(1, -1), (cnt - 1).view(-1, 1)
                )
                flat = order[gidx].view(-1)  # [Q * K] gathered edge ids
                Q = sel.numel()
                gex0 = ex0[:, flat].view(T, Q, K)
                gey0 = ey0[:, flat].view(T, Q, K)
                gex1 = ex1[:, flat].view(T, Q, K)
                gey1 = ey1[:, flat].view(T, Q, K)
                off_x = (eps * length * lnx)[:, sel]
                off_y = (eps * length * lny)[:, sel]
                qx, qy = mx[:, sel], my[:, sel]
                left = _circuit_parity_gathered(
                    qx + off_x, qy + off_y, gex0, gey0, gex1, gey1, slot_valid
                )
                right = _circuit_parity_gathered(
                    qx - off_x, qy - off_y, gex0, gey0, gex1, gey1, slot_valid
                )
                settled = (left != right) & unresolved[:, sel]
                s = torch.where(left, 1.0, -1.0)
                sigma[:, sel] = torch.where(settled, s, sigma[:, sel])
                unresolved[:, sel] &= ~settled
        eps *= 0.5
    return sigma


class RayTracedBezierCircuitPrimitive(BezierCircuitPrimitive):
    """Planar bezier circuits rendered by ray tracing a spatio-temporal BVH.

    Circuits are sampled into polylines with a per-cubic screen-space error
    bound, then expressed in each circuit's own plane coordinates.  The trace
    kernel intersects rays with the plane and classifies hits by an even-odd
    crossing test (fill) plus a min distance to the polyline (border).
    Texture-mapped circuits (``ImageMob`` etc.) are sampled bilinearly
    in-kernel from their texture grid.
    """

    frame_dependent_source_attrs = (
        "corners",
        "colors",
        "normals",
        "border_width",
        "border_color",
        "mob_center",
        "grid_width",
        "grid_height",
        "basis1",
        "basis2",
        "next_segment_inds",
        "reflectivity",
        "roughness",
        "refractive_index",
        "transmission",
    )

    # Same renderer-internal transport channels as the triangle primitive, with
    # the same conventions: ``reflectivity`` is material metalness (negative =
    # non-PBR), ``refractive_index`` is an unsigned magnitude feeding dielectric
    # F0, and ``transmission`` says how much light passes through. A circuit
    # transmits as a thin pane rather than refracting (see ``circuit_scatter``).
    # ``no_shadow_cast`` rides here so the collection merge carries it: a
    # circuit is never a shadow RECEIVER (the renderer draws 2-D geometry
    # unlit, and an unlit hit builds no shadow event), but it is very much a
    # shadow CASTER -- the bezier tree is walked by every shadow ray -- so only
    # the casting half of the declaration is meaningful here. A member that
    # says nothing takes the 0.0 fill below, which is "casts", as before.
    _surface_params = (
        "reflectivity",
        "roughness",
        "refractive_index",
        "transmission",
        "no_shadow_cast",
    )

    # Non-PBR sentinel for metalness; the other channels are inert at 0.
    _surface_param_fill = {"reflectivity": -1.0}

    def declare_shadow_flags(self, casts=True, receives=True):
        """Declare whether this circuit casts a shadow.

        The circuit counterpart of
        :meth:`RayTracedTrianglePrimitive.declare_shadow_flags`, stored the same
        way (negated, so the merge's 0.0 fill means "casts"). ``receives`` is
        accepted so a mob can declare both without knowing which primitive kind
        it built, and deliberately ignored: 2-D geometry is drawn unlit and
        builds no shadow event, so there is no shadow for it to decline.
        """
        self.no_shadow_cast = torch.full_like(
            self.mob_center[..., :1], 0.0 if casts else 1.0
        )
        return self

    def __init__(
        self,
        *args,
        reflectivity=None,
        roughness=None,
        refractive_index=None,
        transmission=None,
        **kwargs,
    ):
        collection = kwargs.get("triangle_collection")
        super().__init__(*args, **kwargs)
        if collection is not None:
            for name in self._surface_params:
                values = []
                for primitive in collection:
                    value = getattr(primitive, name, None)
                    if value is None:
                        value = torch.full_like(
                            primitive.mob_center[..., :1],
                            self._surface_param_fill.get(name, 0.0),
                        )
                    values.append(value)
                values, _ = _unify_time(values, f"bezier {name} merge")
                setattr(self, name, torch.cat(values, 1).to(self.mob_center.device))
        else:
            template = self.mob_center[..., :1]
            for name, value in (
                ("reflectivity", reflectivity),
                ("roughness", roughness),
                ("refractive_index", refractive_index),
                ("transmission", transmission),
            ):
                if value is None:
                    value = torch.full_like(
                        template, self._surface_param_fill.get(name, 0.0)
                    )
                else:
                    value = cast_to_tensor(value).to(template.device)
                    value = broadcast_all([template, value], ignored_dims=[-1])[-1][
                        ..., :1
                    ]
                setattr(self, name, value)

    stbvh_tightness = env_float("ALGAN_STBVH_TIGHTNESS", 1.0)
    max_samples_per_segment = 512
    _rt_projection_aa = 1.0

    def project_to_screen(self, camera, light_sources):
        corners = self.corners.float().contiguous()  # [Tc, S, 4, 3]
        num_frames = camera.ray_origin.shape[0]
        self._rt_num_frames = num_frames

        device = corners.device
        cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)), num_frames).to(
            device
        )
        sp = _expand_frames(_flat_frames(camera.screen_point, (3,)), num_frames).to(
            device
        )
        sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)), num_frames).to(
            device
        )

        corners = self._apply_z_index_bias(corners, cam_o, sp)

        # Ratio of the internal render resolution to the output resolution: the
        # supersampling factor actually in force for this batch, which is 1 on
        # the analytic-AA route regardless of the requested anti_alias_level.
        self._rt_projection_aa = float(camera.screen_height) / float(
            getattr(camera, "output_screen_height", camera.screen_height)
        )

        num_samples = self._compute_samples_per_segment(
            corners,
            cam_o,
            sp,
            sb,
            camera.screen_height,
            bool(getattr(camera, "analytic_raster", False)),
        )
        self._build_circuit_geometry(corners, num_samples)
        self._build_frame_bounds(corners, cam_o, sp, sb, camera.screen_height)

        # The polylines/metadata now carry everything the renderer needs;
        # release the control points to reduce resident GPU memory.
        self.corners = None

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
        return self

    def _apply_z_index_bias(self, corners, cam_o, sp):
        """Nudge each circuit toward the camera by ``z_index`` tie-bins.

        Exactly coplanar circuits produce the same hit distance, so the resolve
        ranks them by an internal index that follows neither creation order nor
        hierarchy (see :attr:`~.BezierCircuitCubic.z_index`). Moving a circuit
        ``z_index * DEPTH_TIE_EPSILON`` along the view axis puts it that many
        depth bins nearer, which is the smallest displacement the ordering can
        see and far below anything the frame can show: at the default camera
        distance one bin is a relative depth change of ~1.4e-5.

        The shift is applied to the control points *and* the plane origins
        together, so the polylines, the plane metadata, the frame AABBs and the
        BVH are all built from the same displaced geometry -- the circuit's own
        ``(u, v)`` parametrization is unchanged, since the plane only slides
        along its view ray.

        Displacing along the *view axis* rather than the plane normal keeps the
        shape where it is on screen (a normal-space offset would slide a tilted
        circuit sideways). The cost is that a circuit seen nearly edge-on gets a
        proportionally smaller bias -- ``|dt| = bias * |f.n| / |rd.n|`` -- but
        such a circuit covers almost no pixels for the ordering to matter in.
        """
        if not getattr(self, "_has_z_index", False):
            return corners
        # Unit view axis per frame: the screen centre as seen from the eye.
        forward = F.normalize((sp - cam_o).float(), p=2, dim=-1)  # [T, 3]
        bias = self.z_index.to(corners.device).float() * DEPTH_TIE_EPSILON  # [1, C, 1]
        num_segments = self.num_segments_per_object.to(corners.device).view(-1).long()
        circuit_of_segment = torch.repeat_interleave(
            torch.arange(num_segments.shape[0], device=corners.device), num_segments
        )
        # [T, 1, 3] * [1, S, 1] -> [T, S, 3], one displacement per segment.
        offset = forward.unsqueeze(-2) * bias[..., circuit_of_segment, :]
        self.mob_center = self.mob_center - forward.unsqueeze(-2) * bias
        return corners - offset.unsqueeze(-2)

    def _compute_samples_per_segment(
        self, corners, cam_o, sp, sb, screen_h, analytic_raster=False
    ):
        """Choose uniform chord counts independently for every cubic segment.

        At each power-of-two subdivision level, the four exact world-space
        controls of every uniform subcurve are projected to the screen.  A
        perspective-projected Bezier with control points on the same side of
        the camera plane is a rational Bezier with positive weights, so it is
        contained by the projected control hull.  The greatest distance of
        that hull from the endpoint chord therefore bounds the curve-to-chord
        error.  We retain the first level whose bound is no larger than
        ``num_pixels_per_sample`` for every frame in the render batch.

        The returned value is the number of chords, despite the legacy
        ``num_samples`` name used by the packed geometry.  One chord evaluates
        two geometric endpoints; its final endpoint is shared with the next
        cubic in the packed representation -- except at the end of an open
        subpath, where ``_build_circuit_geometry`` emits it explicitly.
        """
        device = corners.device
        T = cam_o.shape[0]
        Tc = corners.shape[0]
        S = corners.shape[1]
        if S == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        if Tc not in (1, T):
            raise ValueError(
                f"Bezier controls have {Tc} frames, but the camera has {T}"
            )

        tolerance = float(self.num_pixels_per_sample)
        if tolerance <= 0:
            raise ValueError("num_pixels_per_sample must be greater than zero")
        if analytic_raster and rt_settings.analytic_aa_bez_active():
            # Analytic coverage resolves the outline continuously, so it also
            # exposes the flattening facets that the supersample box filter
            # hides. The classic 0.5 is measured against the SUPERSAMPLED
            # height, i.e. 0.25 output pixels at the AA=2 reference; analytic AA
            # runs at AA=1, where the same number would relax to 0.5. Tighten
            # (never loosen) to keep the reference smoothness.
            tolerance = min(tolerance, float(rt_settings.ANALYTIC_AA_CHORD_TOLERANCE))
        tolerance_squared = tolerance * tolerance

        chord_counts = torch.full(
            (S,), self.max_samples_per_segment, dtype=torch.long, device=device
        )
        active = torch.arange(S, device=device)
        num_subdivisions = 1
        kernel = _bezier_criterion_inputs(corners, cam_o, sp, sb)

        while active.numel() > 0:
            num_active = active.shape[0]
            max_error_squared = torch.zeros(
                (num_active,), dtype=corners.dtype, device=device
            )
            if kernel is not None:
                # Fused: the subcurve split, both projections and the hull
                # measurement never leave registers, so there is nothing to
                # chunk and no per-frame scratch to size.
                bezier_chord_hull_error(
                    kernel[0],
                    kernel[1],
                    active.to(torch.int32).contiguous(),
                    kernel[2],
                    kernel[3],
                    kernel[4],
                    max_error_squared,
                    T,
                    num_subdivisions,
                    screen_h / 2,
                )
                if num_subdivisions == self.max_samples_per_segment:
                    break
                resolved = max_error_squared <= tolerance_squared
                chord_counts[active[resolved]] = num_subdivisions
                active = active[~resolved]
                num_subdivisions = min(
                    num_subdivisions * 2, self.max_samples_per_segment
                )
                continue

            # Bound the largest temporary by projected control-point count.
            # The subcurve construction and projection use several arrays of
            # this shape, so a lower budget than the old single-pass sampler is
            # intentionally used here. The per-chunk reduction is a pure
            # ``torch.maximum`` over frames, so the chunk size cannot change
            # the result -- the budget only trades transient prep memory
            # (~16 B per control point across the temporaries, ~30 MB at 2e6)
            # against per-chunk dispatch overhead, which at the old 5e5 came
            # to thousands of tiny launches per text-heavy batch.
            chunk = max(1, int(2e6 // max(num_active * num_subdivisions * 4, 1)))
            for frame_start in range(0, T, chunk):
                frame_end = min(frame_start + chunk, T)
                if Tc == 1:
                    active_corners = corners[:, active]
                else:
                    active_corners = corners[frame_start:frame_end, active]
                controls = _uniform_cubic_subcurves(active_corners, num_subdivisions)

                frame_shape = (-1,) + (1,) * (controls.ndim - 2) + (3,)
                camera_origin = cam_o[frame_start:frame_end].view(frame_shape)
                screen_point = sp[frame_start:frame_end].view(frame_shape)
                screen_normal = sb[frame_start:frame_end, 2].view(frame_shape)
                rays = controls - camera_origin
                depth = (rays * screen_normal).sum(-1, keepdim=True)
                screen_distance = ((screen_point - camera_origin) * screen_normal).sum(
                    -1, keepdim=True
                )
                projected = camera_origin + (screen_distance / depth) * rays
                relative = projected - screen_point
                basis_shape = (-1,) + (1,) * (controls.ndim - 2) + (3,)
                screen_x = sb[frame_start:frame_end, 0].view(basis_shape)
                screen_y = sb[frame_start:frame_end, 1].view(basis_shape)
                points = torch.stack(
                    ((relative * screen_x).sum(-1), (relative * screen_y).sum(-1)),
                    dim=-1,
                ) * (screen_h / 2)

                chord_start = points[..., 0, :]
                chord_end = points[..., 3, :]
                chord = chord_end - chord_start
                chord_length_squared = chord.square().sum(-1, keepdim=True)

                error_squared = torch.maximum(
                    _point_to_segment_distance_squared(
                        points[..., 1, :], chord_start, chord, chord_length_squared
                    ),
                    _point_to_segment_distance_squared(
                        points[..., 2, :], chord_start, chord, chord_length_squared
                    ),
                )

                # Positive rational weights are required for the projected
                # control hull to be a bound.  A subcurve touching/crossing the
                # camera plane remains active and falls back to the hard cap.
                depth = depth.squeeze(-1)
                same_depth_side = (depth.amin(-1) > 1e-8) | (depth.amax(-1) < -1e-8)
                finite = torch.isfinite(points).all(-1).all(-1)
                valid_bound = same_depth_side & finite
                error_squared = torch.where(
                    valid_bound,
                    error_squared,
                    torch.full_like(error_squared, torch.inf),
                )
                frame_error_squared = error_squared.amax(dim=(0, 2))
                max_error_squared = torch.maximum(
                    max_error_squared, frame_error_squared
                )

            if num_subdivisions == self.max_samples_per_segment:
                break

            resolved = max_error_squared <= tolerance_squared
            chord_counts[active[resolved]] = num_subdivisions
            active = active[~resolved]
            num_subdivisions = min(num_subdivisions * 2, self.max_samples_per_segment)

        return chord_counts

    def _build_circuit_geometry(self, corners, num_samples):
        """Sample world-space polylines into per-circuit plane coordinates and
        pack the per-circuit metadata the trace kernel consumes.
        """
        device = corners.device
        S = corners.shape[1]
        num_segments = self.num_segments_per_object.to(device).view(-1).long()
        C = num_segments.shape[0]

        circuit_of_segment = torch.repeat_interleave(
            torch.arange(C, device=device), num_segments
        )

        nsi = (
            self.next_segment_inds.to(device)
            .reshape(self.next_segment_inds.shape[0], S)
            .long()
        )
        # A redirected edge is an invisible fill closure only when the cubic's
        # true endpoint and the selected next cubic's start are discontinuous.
        # Index wraparound alone is not sufficient: an ordinary closed circuit
        # (Circle, glyph outline, ...) also wraps to an earlier segment and its
        # final border edge must remain visible.
        connection_visible = _bezier_connection_visibility(corners, nsi)

        # The packed polyline samples t = k/n for k < n, taking each cubic's
        # endpoint from the first vertex of the segment it connects to.  That
        # holds only where the connection is continuous; a segment that CLOSES
        # AN OPEN SUBPATH links back to a start point somewhere else, so its
        # endpoint is nobody else's vertex and its final chord would simply be
        # missing.  Those segments get an explicit t = 1 vertex.  A straight
        # ``Line`` is the extreme case -- it resolves to a single chord, so
        # without the endpoint its whole outline collapses to one point and it
        # renders nothing at all.  Whether a connection is continuous can in
        # principle vary over the batch while the vertex count cannot, so a
        # segment discontinuous in ANY frame keeps the extra vertex; where it is
        # continuous the vertex merely duplicates the one it links to, which
        # contributes a zero-length edge to neither metric.
        needs_endpoint = (~connection_visible).any(0).long()
        verts_per_segment = num_samples + needs_endpoint
        vert_circuit = torch.repeat_interleave(circuit_of_segment, verts_per_segment)
        V = int(verts_per_segment.sum())

        t_params = _packed_uniform_cubic_parameters(
            num_samples, corners.dtype, verts_per_segment
        )
        ctrl = torch.repeat_interleave(corners, verts_per_segment, dim=1)
        verts = _evaluate_cubic_bezier_batch(ctrl, t_params.view(1, -1, 1))

        # Plane frame per circuit: normal + an arbitrary orthonormal basis.
        normals = F.normalize(self.normals.float(), p=2, dim=-1)
        centers = self.mob_center.float()
        (normals, centers), _ = _unify_time([normals, centers], "bezier planes")
        axis = torch.zeros_like(normals)
        axis[..., 0] = 1
        alt_axis = torch.zeros_like(normals)
        alt_axis[..., 1] = 1
        helper = torch.where(normals[..., :1].abs() < 0.9, axis, alt_axis)
        basis_u = F.normalize(torch.cross(normals, helper, dim=-1), p=2, dim=-1)
        basis_v = torch.cross(normals, basis_u, dim=-1)

        segment_lengths = (
            (corners[..., 1:, :] - corners[..., :-1, :]).square().sum(-1).sum(-1)
        )
        is_degenerate = segment_lengths < 1e-9
        edge_degenerate = torch.repeat_interleave(
            is_degenerate, verts_per_segment, dim=1
        )

        # Absolute polyline index of the first sample of each segment, and of
        # the sample each segment's last sample connects to (closing each
        # subpath through next_segment_inds, exactly like the rasterizer).
        seg_starts = verts_per_segment.cumsum(0) - verts_per_segment
        seg_ends = seg_starts - 1
        seg_ends[0] = V - 1
        seg_ends = torch.roll(seg_ends, -1, 0)
        next_start = seg_starts[nsi]  # [Tn, S]

        Tn = connection_visible.shape[0]
        border_visible = torch.ones((Tn, V), device=device, dtype=torch.float32)
        seg_ends_expanded = seg_ends.view(1, -1).expand(Tn, -1)
        border_visible.scatter_(1, seg_ends_expanded, connection_visible.float())

        (
            (
                verts_e,
                centers_e,
                basis_u_e,
                basis_v_e,
                next_start_e,
                edge_degenerate_e,
                border_visible_e,
            ),
            T_geo,
        ) = _unify_time(
            [
                verts,
                centers,
                basis_u,
                basis_v,
                next_start.unsqueeze(-1),
                edge_degenerate.unsqueeze(-1),
                border_visible.unsqueeze(-1),
            ],
            "bezier geometry",
        )
        next_start_e = next_start_e.squeeze(-1)
        edge_degenerate_e = edge_degenerate_e.squeeze(-1)
        border_visible_e = border_visible_e.squeeze(-1)

        rel = verts_e - centers_e[:, vert_circuit]
        u = (rel * basis_u_e[:, vert_circuit]).sum(-1)
        v = (rel * basis_v_e[:, vert_circuit]).sum(-1)
        locals_uv = torch.stack((u, v), -1)  # [T_geo, V, 2]
        next_uv = locals_uv.roll(-1, dims=1)
        gather_inds = next_start_e.unsqueeze(-1).expand(T_geo, -1, 2)
        next_uv[:, seg_ends] = torch.gather(locals_uv, 1, gather_inds)
        edges5 = (
            torch.cat((locals_uv, next_uv, border_visible_e.unsqueeze(-1)), -1)
            .float()
            .contiguous()
        )
        edges5 = torch.where(
            edge_degenerate_e.unsqueeze(-1),
            torch.tensor([1e9, 1e9, 1e9, 1e9, 0.0], device=device),
            edges5,
        )
        # Column 5: the edge's INWARD SIGN sigma (DESIGN_analytic_aa_v2.md
        # ss5.2) -- +1 when the drawn (odd-parity) side of the edge's line is
        # the side its leftward perpendicular points to, -1 the other way, 0
        # unknown (degenerate, or the probe could not settle it). Computed at
        # flatten time, where the contour is known, because the crossing
        # parity is a property of the QUERY and can orient only the nearest
        # wall -- recovering a second wall's side from handedness at a corner
        # was the ss21.6 wedge failure. Per frame: a morph can flip a
        # contour's winding mid-animation. Zeros unless the wedge is live
        # (the only reader), so the probe costs nothing otherwise.
        if rt_settings.analytic_aa_bez_mode() == 3:
            sigma = _circuit_edge_inward_signs(edges5, vert_circuit)
        else:
            sigma = torch.zeros(edges5.shape[:2], device=device)
        self._rt_edges = torch.cat((edges5, sigma.unsqueeze(-1)), -1).contiguous()

        samples_per_circuit = torch.zeros((C,), dtype=torch.long, device=device)
        samples_per_circuit.index_add_(0, circuit_of_segment, verts_per_segment)
        edge_offsets = torch.zeros((C + 1,), dtype=torch.long, device=device)
        edge_offsets[1:] = samples_per_circuit.cumsum(0)
        self._rt_edge_offsets = edge_offsets.to(torch.int32).contiguous()
        self._rt_circuit_of_segment = circuit_of_segment

        # Texture-grid transform: maps plane (u, v) displacements to the
        # mob-basis coordinates used by the texture lookup.
        def scaled(basis):
            basis = basis.float()
            return basis / basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(
                1e-12
            )

        basis1, basis2 = scaled(self.basis1), scaled(self.basis2)
        # ``border_width`` is authored in OUTPUT pixels, but every consumer
        # scales it by ``pixel_world_scale``, which is world-per-INTERNAL-pixel
        # (built from ``camera.screen_height``).  Convert here, so a supersampled
        # render draws the same apparent border as an analytic one instead of a
        # 1/aa-thin sliver.
        border_width = (
            self.border_width.float().reshape(self.border_width.shape[0], C)
            * self._rt_projection_aa
        )
        grid_w = self.grid_width.float().reshape(self.grid_width.shape[0], C)
        grid_h = self.grid_height.float().reshape(self.grid_height.shape[0], C)
        reflectivity = self.reflectivity.float()
        roughness = self.roughness.float()
        refractive_index = self.refractive_index.float()
        transmission = self.transmission.float()
        (
            (
                centers_m,
                normals_m,
                bu_m,
                bv_m,
                b1_m,
                b2_m,
                bw_m,
                gw_m,
                gh_m,
                reflectivity_m,
                roughness_m,
                ior_m,
                transmission_m,
            ),
            Tm,
        ) = _unify_time(
            [
                centers,
                normals,
                basis_u,
                basis_v,
                basis1,
                basis2,
                border_width.unsqueeze(-1),
                grid_w.unsqueeze(-1),
                grid_h.unsqueeze(-1),
                reflectivity,
                roughness,
                refractive_index,
                transmission,
            ],
            "bezier metadata",
        )
        filled = torch.full((Tm, C, 1), 1.0 if self.filled else 0.0, device=device)
        tex = torch.stack(
            (
                (b1_m * bu_m).sum(-1),
                (b1_m * bv_m).sum(-1),
                (b2_m * bu_m).sum(-1),
                (b2_m * bv_m).sum(-1),
            ),
            -1,
        ).nan_to_num_()
        self._rt_circuit_meta = torch.cat(
            (
                centers_m,
                normals_m,
                bu_m,
                bv_m,
                bw_m,
                filled,
                gw_m,
                gh_m,
                tex,
                reflectivity_m,
                roughness_m,
                ior_m,
                transmission_m,
            ),
            -1,
        ).contiguous()

        colors = self.colors.float()
        if colors.dim() == 3:  # plain fills: a 1x1 "texture" grid
            colors = colors.unsqueeze(-2)
        self._rt_circuit_colors = colors.contiguous().as_subclass(Color)
        border_colors = self.border_color.float()
        if border_colors.dim() == 3:
            border_colors = border_colors.unsqueeze(-2)
        self._rt_circuit_border_colors = border_colors.contiguous().as_subclass(Color)
        self._rt_border_width = border_width

    def _build_frame_bounds(self, corners, cam_o, sp, sb, screen_h):
        """Per-frame circuit AABBs (from control-point hulls, inflated by the
        screen-space border width and glow radius), with invisible frames marked empty.
        """
        device = corners.device
        C = self._rt_edge_offsets.shape[0] - 1
        circuit_of_segment = self._rt_circuit_of_segment

        seg_lo = corners.amin(-2)
        seg_hi = corners.amax(-2)
        Tb = seg_lo.shape[0]
        idx = circuit_of_segment.view(1, -1, 1).expand(Tb, -1, 3)
        lo = torch.full((Tb, C, 3), EMPTY_LO, device=device).scatter_reduce_(
            1, idx, seg_lo, "amin", include_self=True
        )
        hi = torch.full((Tb, C, 3), EMPTY_HI, device=device).scatter_reduce_(
            1, idx, seg_hi, "amax", include_self=True
        )

        fill_alpha = self._rt_circuit_colors.opacity.squeeze(-1).amax(
            -1
        )  # over texture
        fill_min = self._rt_circuit_colors.opacity.squeeze(-1).amin(-1)
        if not self.filled:
            fill_alpha = torch.zeros_like(fill_alpha)
        border_alpha_grid = self._rt_circuit_border_colors.opacity.squeeze(-1)
        border_alpha = border_alpha_grid.amax(-1)
        border_min = border_alpha_grid.amin(-1)
        border_on = self._rt_border_width > 1e-3
        glow_alpha = self._rt_circuit_colors[..., 3].amax(-1)
        visible = (
            (fill_alpha > MIN_ALPHA)
            | ((border_alpha > MIN_ALPHA) & border_on)
            | (glow_alpha > 0.0)
        )
        # Alpha is pure coverage, so it alone decides presence (see
        # ``_pack_frame_visibility``); transmission only bears on opacity.
        transmissive = self.transmission[..., 0] > 1e-6
        (
            (
                lo,
                hi,
                visible,
                fill_min,
                border_alpha,
                border_min,
                border_on,
                transmissive,
            ),
            _,
        ) = _unify_time(
            [
                lo,
                hi,
                visible.unsqueeze(-1),
                fill_min.unsqueeze(-1),
                border_alpha.unsqueeze(-1),
                border_min.unsqueeze(-1),
                border_on.unsqueeze(-1),
                transmissive.unsqueeze(-1),
            ],
            "bezier bounds/colors",
        )
        visible = visible.squeeze(-1)
        # A circuit is opaque (prunes hits behind it while gathering) only if
        # every region a hit can land in -- the fill/texture and, when shown,
        # the border -- is fully opaque.
        opaque = (fill_min.squeeze(-1) >= 1.0 - 1e-6) & (
            (~border_on.squeeze(-1)) | (border_min.squeeze(-1) >= 1.0 - 1e-6)
        )
        # A transmissive circuit lets light through even at full coverage, so
        # it can never prune hits behind it.
        opaque = opaque & ~transmissive.squeeze(-1)
        if not self.filled:
            opaque = torch.zeros_like(opaque)
        self._rt_frame_opaque = opaque.contiguous()
        # See the triangle primitive's assignment: same flag, same destination.
        self._rt_frame_casts = shadow_cast_flag(
            getattr(self, "no_shadow_cast", None), lo.shape[1], device
        )
        lo = torch.where(
            visible.unsqueeze(-1), lo, torch.tensor(EMPTY_LO, device=device)
        )
        hi = torch.where(
            visible.unsqueeze(-1), hi, torch.tensor(EMPTY_HI, device=device)
        )

        # Inflate by however far outside the control-point hull the circuit can
        # still draw, converted to world units at its distance from the camera.
        # A filled circuit's border runs INWARD, so the only outward reach is the
        # anti-crack outline dilation plus the analytic-coverage filter radius
        # (0.3 + 0.707 = 1.008 px at worst); an unfilled circuit's stroke is
        # centred on the path, so half its width reaches out as well.
        b1_norm = sb[:, 1].norm(p=2, dim=-1)
        screen_dist = (sp - cam_o).norm(p=2, dim=-1)
        pixel_world_scale = 2.0 / (screen_h * b1_norm * screen_dist).clamp_min(1e-12)
        centers = self._rt_circuit_meta[..., :3]
        dist = (centers - cam_o.view(-1, 1, 3)).norm(p=2, dim=-1)
        world_per_px = (pixel_world_scale.view(-1, 1) * dist).amax(0)

        inflate = (0.5 * self._rt_border_width.amax(0) + 1.5) * world_per_px
        self._rt_frame_lo = (lo - inflate.view(1, -1, 1)).contiguous()
        self._rt_frame_hi = (hi + inflate.view(1, -1, 1)).contiguous()

    def render(
        self,
        primitives,
        scene,
        save_image,
        screen_width,
        screen_height,
        time_start,
        time_end,
        background_color,
        transparent_background=False,
        *args,
        **kwargs,
    ):
        return KERNEL_REGISTRY.render_kernel(
            primitives,
            scene,
            screen_width,
            screen_height,
            time_start,
            time_end,
            background_color,
            transparent_background,
            *args,
            **kwargs,
        )
