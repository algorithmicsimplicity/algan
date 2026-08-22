"""Collection of helper functions used to combine collections of primitives
into contiguous tensor data-structures, ready to be shipped to ray tracing kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.bezier_acceleration import (
    build_bezier_edge_acceleration,
)
from algan.rendering.raytracing.primitives import (
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_REFLECTIVITY,
    _M_TRANSMISSION,
    _M_WIDTH,
)
from algan.rendering.raytracing.refit_bvh import build_refit_bvh
from algan.rendering.raytracing.settings import (
    _USER_PIPELINE_BASE,
    _constant_promotion_active,
)
from algan.rendering.raytracing.shading_taichi import _MID_UNLIT, MAT_W
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO, STBVH, build_stbvh
from algan.rendering.raytracing.utils import (
    _cat_collections,
    _cat_mat_blocks,
    _expand_frames,
    _flat_frames,
)
from algan.settings import SETTINGS
from algan.settings._startup import _RENDER_DEVICE
from algan.utils.color_space import srgb_to_linear
from algan.utils.memory_utils import (
    InsufficientMemoryException,
    begin_cuda_peak,
    empty_cache,
    end_cuda_peak,
)

_STBVH_TENSOR_FIELDS = ("nodes", "blocks", "node_miss", "leaf_prim", "leaf_tspan")


def _cat_circuit_color_grids(grids, error_context="bezier color-grid merge"):
    """Merge ``[T, C, P, 5]`` circuit grids with different resolutions.

    A scene can contain separately batched circuit collections whose texture
    grids have different point counts.  The Taichi kernels need one rectangular
    scene tensor, so pad the unused tail of each collection before concatenating
    its circuit axis.  Circuit metadata keeps the actual grid width/height, and
    the sampler therefore never reads these padding texels.
    """
    max_points = max(grid.shape[2] for grid in grids)
    padded = []
    for grid in grids:
        if grid.shape[2] < max_points:
            pad = torch.zeros(
                (*grid.shape[:2], max_points - grid.shape[2], grid.shape[3]),
                dtype=grid.dtype,
                device=grid.device,
            )
            grid = torch.cat((grid, pad), 2)
        padded.append(grid)
    return _cat_collections(padded, 1, error_context)


class _DeferredBackground:
    """Callback plus the metadata needed to fill the render output."""

    __slots__ = (
        "callback",
        "width",
        "height",
        "anti_alias_level",
        "first_frame",
        "frames_per_second",
        "device",
        "is_taichi_func",
    )

    def __init__(
        self,
        callback,
        width,
        height,
        anti_alias_level,
        first_frame,
        frames_per_second,
        device,
    ):
        self.callback = callback
        self.width = int(width)
        self.height = int(height)
        self.anti_alias_level = int(anti_alias_level)
        self.first_frame = int(first_frame)
        self.frames_per_second = float(frames_per_second)
        self.device = torch.device(device)
        self.is_taichi_func = bool(getattr(callback, "_is_taichi_function", False))


def _projected_scene_device(primitives):
    """Device carrying a projected primitive batch's ray-tracing tensors."""
    preferred = (
        "_rt_tri_pos",
        "_rt_edges",
        "_rt_circuit_meta",
        "_rt_frame_lo",
    )
    for primitive in primitives:
        for name in preferred:
            value = getattr(primitive, name, None)
            if torch.is_tensor(value):
                return value.device
        # Keep this tolerant of new ray-traced primitive types: any projected
        # ``_rt_*`` tensor is a better source of truth than a global device.
        for name, value in vars(primitive).items():
            if name.startswith("_rt_") and torch.is_tensor(value):
                return value.device
    raise ValueError("projected primitive batch contains no ray-tracing tensors")


# Non-geometry ``_rt_*`` attributes that must never be relocated with the
# packed inputs (the merged-scene cache, the arena-backed scene, and the
# host/env prep handles).
_MERGE_SKIP_ATTRS = frozenset(
    {"_rt_merged_scene", "_rt_device_scene", "_rt_prepared_host_scene", "_rt_env_meta"}
)


def _iter_primitive_input_tensors(primitive):
    """Yield ``(name, tensor)`` for each packed ``_rt_*`` geometry tensor of a
    projected primitive (skipping the merged-scene cache / handle attrs).
    """
    for name, value in list(vars(primitive).items()):
        if (
            name.startswith("_rt_")
            and name not in _MERGE_SKIP_ATTRS
            and torch.is_tensor(value)
        ):
            yield name, value


def _upload_primitive_inputs(primitives, device):
    """Move every primitive's packed ``_rt_*`` geometry onto ``device`` in
    place, so the subsequent merge + STBVH build run there. Each CPU source
    tensor is dropped as it is replaced; the merge nulls the device copies once
    the contiguous per-type arrays are built.
    """
    for primitive in primitives:
        for name, value in _iter_primitive_input_tensors(primitive):
            if value.device != device:
                setattr(primitive, name, value.to(device))


def gpu_merge_input_bytes(primitives):
    """Total bytes of a batch's packed ``_rt_*`` inputs (before the merge).

    Feeds the GPU merge's transient-peak estimate used by the render-arena
    preflight to keep the build inside the pool's headroom (see
    ``settings.MERGE_GPU_PEAK_FACTOR`` and ``RenderLoopMixin``).
    """
    total = 0
    for primitive in primitives:
        for _name, value in _iter_primitive_input_tensors(primitive):
            total += value.numel() * value.element_size()
    return total


def _iter_primitive_source_tensors(primitive, include_shader_params=True):
    """Yield ``(name, tensor)`` for each pre-projection source-geometry tensor
    of a primitive (corners/normals/colors/material/texture rows, ...). These
    are every torch tensor that is not a packed ``_rt_*`` output or a
    cache/handle attribute; project_to_screen releases them once packed.
    """
    for name, value in list(vars(primitive).items()):
        if (
            torch.is_tensor(value)
            and not name.startswith("_rt_")
            and name not in _MERGE_SKIP_ATTRS
        ):
            yield name, value
    if not include_shader_params or not hasattr(primitive, "shader_param_values"):
        return
    for name, value in zip(primitive.shader_param_names, primitive.shader_param_values):
        yield name, value


def upload_primitive_source(primitive, device):
    """Move a primitive's pre-projection source geometry onto ``device`` so
    ``project_to_screen`` (and its vertex shader) run there (project-on-gpu).
    """
    for name, value in _iter_primitive_source_tensors(
        primitive, include_shader_params=False
    ):
        if value.device != device:
            setattr(primitive, name, value.to(device))
    if not hasattr(primitive, "shader_param_values"):
        return
    for i in range(len(primitive.shader_param_values)):
        value = primitive.shader_param_values[i]
        if value.device != device:
            primitive.shader_param_values[i] = value.to(device)


def gpu_project_input_bytes(primitives):
    """Total bytes of a batch's pre-projection source geometry.

    Feeds the projection's transient-peak estimate used by the render-arena
    preflight (see ``settings.PROJECT_GPU_PEAK_FACTOR`` and
    ``RenderLoopMixin``). Already-projected primitives (source released) count
    zero.
    """
    total = 0
    for primitive in primitives:
        for _name, value in _iter_primitive_source_tensors(primitive):
            total += value.numel() * value.element_size()
    return total


def _collect_scene_tensors(scene):
    """Return each tensor object reachable from a merged scene exactly once."""
    tensors = []
    seen_tensors = set()
    seen_containers = set()

    def visit(value):
        if torch.is_tensor(value):
            key = id(value)
            if key not in seen_tensors:
                seen_tensors.add(key)
                tensors.append(value)
            return
        if isinstance(value, STBVH):
            key = id(value)
            if key in seen_containers:
                return
            seen_containers.add(key)
            for field in _STBVH_TENSOR_FIELDS:
                visit(getattr(value, field))
            return
        if isinstance(value, dict):
            key = id(value)
            if key in seen_containers:
                return
            seen_containers.add(key)
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            key = id(value)
            if key in seen_containers:
                return
            seen_containers.add(key)
            for item in value:
                visit(item)

    visit(scene)
    return tensors


def _scene_storage_groups(scene):
    """Group scene tensor views by their underlying untyped storage."""
    groups = {}
    for tensor in _collect_scene_tensors(scene):
        storage = tensor.untyped_storage()
        # ``untyped_storage()`` may return a fresh Python wrapper; ``_cdata``
        # identifies the actual TensorImpl storage and also distinguishes
        # separate zero-byte storages (whose data_ptr values are all zero).
        key = storage._cdata
        if key not in groups:
            groups[key] = {"storage": storage, "tensors": []}
        groups[key]["tensors"].append(tensor)
    return list(groups.values())


def get_merged_scene_tensor_nbytes(scene):
    """Bytes owned by the unique tensor storages reachable from ``scene``.

    Aliased tensors and aliased :class:`STBVH` fields are charged once.  This
    is the physical source-storage size, rather than a sum of tensor ``numel``
    values, so sliced/strided views are accounted and copied without breaking
    their alias relationship.
    """
    return sum(group["storage"].nbytes() for group in _scene_storage_groups(scene))


def _storage_is_arena(storage, memory):
    return storage._cdata == memory.data.untyped_storage()._cdata


def _storage_alignment(group):
    return max((tensor.element_size() for tensor in group["tensors"]), default=1)


def _group_needs_arena_copy(group, memory):
    tensors = group["tensors"]
    target = memory.data.device
    if any(tensor.device != target for tensor in tensors):
        return True
    if group["storage"].nbytes() == 0:
        return False
    return not _storage_is_arena(group["storage"], memory)


def get_merged_scene_arena_nbytes(scene, memory, *, persist=True):
    """Exact arena pointer delta for :func:`copy_merged_scene_to_arena`.

    The result includes byte-alignment padding at the arena's *current*
    forward or reverse pointer and skips tensors already backed by that arena.
    Consequently it can be compared directly with the corresponding pointer
    delta after an upload.  An unmanaged ``ManualMemory`` has no arena-backed
    allocation path and therefore consumes zero pointer bytes.
    """
    if not getattr(memory, "managed", False):
        return 0
    pointer = memory.current_reverse_pointer if persist else memory.current_pointer
    initial = pointer
    for group in _scene_storage_groups(scene):
        if not _group_needs_arena_copy(group, memory):
            continue
        nbytes = group["storage"].nbytes()
        if nbytes == 0:
            continue
        alignment = _storage_alignment(group)
        if persist:
            pointer -= pointer % alignment
            pointer -= nbytes
        else:
            pointer += (-pointer) % alignment
            pointer += nbytes
    return initial - pointer if persist else pointer - initial


def _arena_storage_copy(group, memory, persist):
    """Copy one source storage into aligned raw bytes owned by ``memory``."""
    storage = group["storage"]
    nbytes = storage.nbytes()
    target = memory.data.device
    if nbytes == 0:
        # Empty tensors own no bytes; use a zero-length view of the destination
        # arena solely to give reconstructed empty tensors the target device.
        return memory.data[:0]

    alignment = _storage_alignment(group)
    pointer = memory.current_reverse_pointer if persist else memory.current_pointer
    padding = pointer % alignment if persist else (-pointer) % alignment
    if padding:
        memory.get_tensor((padding,), dtype=torch.uint8, persist=persist)
    destination = memory.get_tensor((nbytes,), dtype=torch.uint8, persist=persist)

    # Wrapping an UntypedStorage produces a byte view without allocating a
    # second source tensor. ``copy_`` performs any host/device transfer directly
    # into the already-reserved destination bytes.
    source = torch.as_tensor(
        storage, dtype=torch.uint8, device=group["tensors"][0].device
    )
    destination.copy_(source)
    if destination.device != target:  # defensive: ManualMemory owns the target
        raise RuntimeError("ManualMemory returned a tensor on the wrong device")
    return destination


def _view_in_arena(source, destination_bytes):
    """Recreate ``source``'s shape/stride/offset over copied arena bytes."""
    itemsize = source.element_size()
    byte_offset = destination_bytes.storage_offset()
    if byte_offset % itemsize:
        raise RuntimeError("arena tensor storage is not dtype-aligned")
    # A zero-length typed view plus as_strided changes metadata only; it never
    # allocates destination storage. Storage offsets are absolute within the
    # ManualMemory uint8 backing tensor.
    typed = destination_bytes[:0].view(source.dtype)
    view = torch.as_strided(
        typed,
        source.size(),
        source.stride(),
        storage_offset=byte_offset // itemsize + source.storage_offset(),
    )
    if type(source) is not torch.Tensor:
        view = view.as_subclass(type(source))
    return view


def _rebuild_scene_with_tensors(value, tensor_map, memo):
    if torch.is_tensor(value):
        return tensor_map[id(value)]
    key = id(value)
    if key in memo:
        return memo[key]
    if isinstance(value, STBVH):
        # Type-aware: a RefitBVH rebuilds as a RefitBVH, carrying over the
        # scalar layout fields the tensor shapes alone cannot recover.
        rebuilt = type(value).from_prebuilt(
            tensor_map[id(value.nodes)],
            tensor_map[id(value.node_miss)],
            tensor_map[id(value.leaf_prim)],
            tensor_map[id(value.leaf_tspan)],
            tensor_map[id(value.blocks)],
            like=value,
        )
        memo[key] = rebuilt
        return rebuilt
    if isinstance(value, dict):
        rebuilt = {}
        memo[key] = rebuilt
        for item_key, item in value.items():
            rebuilt[item_key] = _rebuild_scene_with_tensors(item, tensor_map, memo)
        return rebuilt
    if isinstance(value, list):
        rebuilt = []
        memo[key] = rebuilt
        rebuilt.extend(
            _rebuild_scene_with_tensors(item, tensor_map, memo) for item in value
        )
        return rebuilt
    if isinstance(value, tuple):
        rebuilt = tuple(
            _rebuild_scene_with_tensors(item, tensor_map, memo) for item in value
        )
        memo[key] = rebuilt
        return rebuilt
    return value


def copy_merged_scene_to_arena(scene, memory, *, persist=True):
    """Copy the merged scene into the arena (see :func:`_copy_merged_scene_to_arena`).

    Wrapped as its own memory-calibration scope. Its size is already known
    exactly from ``get_merged_scene_arena_nbytes``, so this is recorded for
    stale-table detection rather than to be modelled.
    """
    with memory.scope("scene_upload"):
        return _copy_merged_scene_to_arena(scene, memory, persist=persist)


def _copy_merged_scene_to_arena(scene, memory, *, persist=True):
    """Copy every merged-scene tensor into ``memory`` without target allocs.

    One aligned arena range is reserved per unique source storage, then every
    tensor view is recreated with its original shape, stride and relative
    storage offset. Exact tensor aliases, view aliases and repeated STBVH
    objects are therefore preserved. STBVHs are cloned with their already-built
    ``blocks``; no BVH operation runs on the destination device.

    A scene already backed by the supplied arena is returned by identity. An
    unmanaged memory object explicitly disables automatic arena ownership, so
    it falls back to one ordinary destination allocation per unique source
    storage while preserving tensor/STBVH aliases.
    """
    groups = _scene_storage_groups(scene)
    if not getattr(memory, "managed", False):
        target = memory.data.device
        if all(
            tensor.device == target for group in groups for tensor in group["tensors"]
        ):
            return scene
        tensor_map = {}
        for group in groups:
            tensors = group["tensors"]
            if all(tensor.device == target for tensor in tensors):
                for tensor in tensors:
                    tensor_map[id(tensor)] = tensor
                continue
            storage = group["storage"]
            destination = torch.empty(
                (storage.nbytes(),), dtype=torch.uint8, device=target
            )
            source = torch.as_tensor(
                storage, dtype=torch.uint8, device=tensors[0].device
            )
            destination.copy_(source)
            for tensor in tensors:
                tensor_map[id(tensor)] = _view_in_arena(tensor, destination)
        return _rebuild_scene_with_tensors(scene, tensor_map, {})

    if not any(_group_needs_arena_copy(group, memory) for group in groups):
        return scene

    required = get_merged_scene_arena_nbytes(scene, memory, persist=persist)
    if required > memory.get_num_bytes_remaining():
        raise InsufficientMemoryException

    initial_pointers = memory.get_pointers()
    tensor_map = {}
    try:
        for group in groups:
            if not _group_needs_arena_copy(group, memory):
                for tensor in group["tensors"]:
                    tensor_map[id(tensor)] = tensor
                continue
            destination = _arena_storage_copy(group, memory, persist)
            for tensor in group["tensors"]:
                tensor_map[id(tensor)] = _view_in_arena(tensor, destination)
        return _rebuild_scene_with_tensors(scene, tensor_map, {})
    except Exception:
        # Keep uploads transactional: callers can retry with a smaller frame
        # window without retaining a partially consumed scene allocation.
        memory.set_pointers(initial_pointers)
        raise


def _dedup_time(x):
    """Collapse a leading (time) dimension that is constant across frames to
    length 1, so a temporally-constant map/colour is stored once instead of T
    times. The kernels index the time axis as ``f % shape[0]``, so a length-1
    axis is read by every frame.
    """
    if x.shape[0] > 1 and bool((x == x[:1]).all()):
        return x[:1].contiguous()
    return x


def _split_promotable(p, _append_texture, device, scene):
    """Partition a non-textured triangle primitive into the triangles that must
    stay per-vertex and the triangles whose colour + material are constant
    across their three corners and every frame (and are non-glowing). The
    constant triangles are grouped by value -- so a uniform mob is one group even
    when it was batched into a primitive alongside differently-coloured mobs --
    and each group is promoted to one shared 1x1 colour map + 1x1 material map
    (appended here to the shared texel buffer).

    Returns ``(keep_idx, promo_idx, promo_meta)``: ascending ``keep_idx`` selects
    the per-vertex triangles; ``promo_idx`` selects the promoted triangles
    grouped by value; ``promo_meta`` is the ``[len(promo_idx), 10]`` tex-meta
    (colour map cols 0-2, material map 3-5, no normal map 6-8 = -1, bitmask 9 =
    refl|rough|ior) aligned to ``promo_idx``. The kernel reads all three material
    properties from the material map, so promoted triangles need no per-vertex
    ``tri_colors``/``tri_extra`` row.
    """
    colors = p._rt_tri_colors  # [Tc, N, 3, 5]
    extra = p._rt_tri_extra  # [Te, N, 15]
    N = colors.shape[1]
    all_idx = torch.arange(N, device=device)
    if N == 0:
        empty = torch.zeros((0, 10), dtype=torch.int32, device=device)
        return all_idx, all_idx, empty
    # Per-triangle promotable: the three corners share one colour (all channels,
    # all frames) and one material (reflectivity 0/2/4, roughness 1/3/5, index of
    # refraction 6/7/8), and the triangle is non-glowing (glow magnitude cols
    # 9-11 zero; a nonzero default glow_radius in 12-14 is irrelevant once glow
    # is 0). Only such a triangle is fully described by a single 1x1 texel.
    color_eq = (colors == colors[:, :, :1, :]).all(-1).all(-1).all(0)  # [N]
    e = extra
    mat_eq = (
        (e[..., 0] == e[..., 2])
        & (e[..., 0] == e[..., 4])
        & (e[..., 1] == e[..., 3])
        & (e[..., 1] == e[..., 5])
        & (e[..., 6] == e[..., 7])
        & (e[..., 6] == e[..., 8])
    ).all(0)  # [N]
    nonglow = (e[..., 9:12] == 0).all(-1).all(0)  # [N]
    promotable = color_eq & mat_eq & nonglow
    keep_idx = all_idx[~promotable]
    promo_all = all_idx[promotable]
    if promo_all.numel() == 0:
        empty = torch.zeros((0, 10), dtype=torch.int32, device=device)
        return keep_idx, promo_all, empty

    # Group promoted triangles by their (per-frame) constant colour + material
    # value, so identical mobs share one pair of maps. The key is the corner-0
    # colour [T,5] plus material (refl, rough, ior) [T,3] over all frames.
    Tc, Te = colors.shape[0], extra.shape[0]
    T = max(Tc, Te)
    col0 = _expand_frames(colors[:, :, 0, :], T)[:, promo_all, :]  # [T,P,5]
    mat3 = _expand_frames(
        torch.stack([extra[..., 0], extra[..., 1], extra[..., 6]], -1), T
    )[:, promo_all, :]  # [T,P,3]
    key = (
        torch.cat([col0, mat3], -1).permute(1, 0, 2).reshape(promo_all.numel(), -1)
    )  # [P, 8T]
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)  # inv [P]
    order = torch.argsort(inv, stable=True)  # group identical values contiguously
    promo_idx = promo_all[order]
    inv_sorted = inv[order]

    # One colour + material map per distinct value; each promoted triangle's meta
    # row points at its group's maps.
    group_meta = []
    for gid in range(uniq.shape[0]):
        rep = int(promo_all[int((inv == gid).nonzero()[0])])
        cmap = _dedup_time(colors[:, rep : rep + 1, 0, :].contiguous())  # [T',1,5]
        color_meta = _append_texture(
            cmap.reshape(cmap.shape[0], 1, 1, 5).float().contiguous()
        )
        e0 = extra[:, rep : rep + 1, :]
        z = torch.zeros_like(e0[..., 0])
        mmap = _dedup_time(
            torch.stack(
                [e0[..., 0], e0[..., 1], e0[..., 6], e0[..., 9], z], -1
            ).contiguous()
        )
        material_meta = _append_texture(
            mmap.reshape(mmap.shape[0], 1, 1, 5).float().contiguous()
        )
        if bool((mmap[..., 3] > 1e-6).any()):
            scene["tex_has_refractive"] = True
        # Promoted metalness/IOR that can produce a nonzero Fresnel lobe
        # (mirrors _material_reflectance: metalness < 0 is the non-PBR
        # sentinel with R = 0; metalness 0 still reflects through the
        # dielectric lobe when IOR > 1).
        if bool(
            (
                (mmap[..., 0] > 0.0)
                | ((mmap[..., 0] >= 0.0) & (mmap[..., 2].abs() > 1.0 + 1e-4))
            ).any()
        ):
            scene["tex_has_reflective"] = True
        group_meta.append([*color_meta, *material_meta, -1, 0, 0, 1 | 2 | 4 | 8])
    group_meta = torch.tensor(group_meta, dtype=torch.int32, device=device)
    promo_meta = group_meta[inv_sorted]  # [P,10]
    return keep_idx, promo_idx, promo_meta


def _build_accel(
    lo, hi, num_frames, tightness, opaque=None, builder="morton", refit=None
):
    """Build one geometry type's acceleration structure: the classic
    spatio-temporal instance tree, or -- under ``settings.BVH_REFIT`` -- the
    shared-topology binned-SAH refit tree (refit_bvh.py; ``tightness`` /
    ``builder`` do not apply there). All trees of a batch dispatch through
    this one gate so every launch passes a single consistent ``refit``
    template to the kernels. ``refit`` overrides the live toggle so a
    deferred build (see ``build_deferred_bvhs``) reproduces the tree kind the
    batch's placeholder trees were merged with, even if the user flipped the
    toggle in between.
    """
    _rts = SETTINGS.raytracing
    if _rts.refit_bvh_active() if refit is None else refit:
        return build_refit_bvh(lo, hi, num_frames=num_frames, opaque=opaque)
    return build_stbvh(
        lo,
        hi,
        num_frames=num_frames,
        tightness=tightness,
        opaque=opaque,
        builder=builder,
    )


def _empty_scene_part(device, refit=None):
    """Placeholder BVH + arrays for an absent geometry type (same tree kind
    as the batch's real trees, so one compile-time flag covers all six).
    """
    lo = torch.full((1, 1, 3), EMPTY_LO, device=device)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=device)
    return _build_accel(lo, hi, num_frames=1, tightness=2.0, refit=refit)


def _build_opaque_bvh(
    lo, hi, opaque, num_frames, tightness, builder="morton", refit=None
):
    """Build a BVH containing only primitives proven opaque when visible.

    The primitive index space is intentionally unchanged; transparent and
    invisible slots become empty bounds so the prepass hit records can be used
    with the normal geometry arrays.
    """
    visible = (hi >= lo).all(-1)
    visible_opaque = visible & opaque
    opaque_lo = torch.where(
        visible_opaque.unsqueeze(-1), lo, torch.full_like(lo, EMPTY_LO)
    )
    opaque_hi = torch.where(
        visible_opaque.unsqueeze(-1), hi, torch.full_like(hi, EMPTY_HI)
    )
    return _build_accel(
        opaque_lo.contiguous(),
        opaque_hi.contiguous(),
        num_frames=num_frames,
        tightness=tightness,
        opaque=visible_opaque.contiguous(),
        builder=builder,
        refit=refit,
    )


def _bvh_deferral_eligible(scene):
    """Conservatively true when this batch provably never traverses a BVH.

    The hybrid raster front-end resolves and shades every primary ray without
    the trees; the trees are only walked by (a) primary traversal when a batch
    routes to the classic wavefront, (b) the sparse shadow queue, (c) bounced
    reflection/refraction/scatter continuations, and (d) the Monte Carlo
    megakernel. Every merge-time-knowable trigger for those is excluded here;
    runtime-only triggers (camera near clip, toggles flipped after the merge,
    an actually spawned continuation) are caught by the tracer, which calls
    ``build_deferred_bvhs`` before any traversal could happen -- so a false
    positive here costs a late build, never a wrong image.
    """
    _rts = SETTINGS.raytracing
    if not (_rts.BVH_DEFER and _rts.HYBRID_RASTER):
        return False
    if int(_rts.SAMPLES_PER_PIXEL) > 1 or _rts.SHADOWS or _rts.INPLACE_AA:
        return False
    if _rts.WF_TEXTURED or scene.get("textured_active"):
        return False
    if scene.get("mem_trim_active"):
        return False
    # Custom fragment pipelines may override scattering; conservatively keep
    # the trees for any scene that carries one.
    if scene.get("has_user_pipeline"):
        return False
    if (
        scene.get("has_refractive")
        or scene.get("has_refl_transparent")
        or scene.get("tri_has_reflective")
        or scene.get("bez_has_reflective")
    ):
        return False
    return scene["num_triangles"] > 0 or scene["num_circuits"] > 0


def _finalize_bvhs(scene, tri_inputs, bez_inputs, num_frames, device):
    """Build each geometry type's STBVHs from the captured merge inputs, or
    record a deferral (placeholder trees + retained build inputs) for batches
    that provably never traverse one (see ``_bvh_deferral_eligible``).
    """
    # The dedicated opaque-only trees are consumed solely under the
    # WF_OPAQUE_CLOSEST / WF_OPAQUE_PREPASS rollouts (both default OFF): the
    # tracer's opaque_closest/opaque_prepass templates compile every read out
    # otherwise. With neither live (read at merge time), alias the main tree
    # instead of building a second one -- ~40% of the per-batch BVH build.
    # ``opaque_bvh_skipped`` lets the tracer keep those features off for a
    # batch merged without real opaque trees if a toggle flips mid-render.
    opq_live = (
        not SETTINGS.raytracing.OPAQUE_BVH_SKIP_DEAD
        or SETTINGS.raytracing.WF_OPAQUE_CLOSEST
        or SETTINGS.raytracing.WF_OPAQUE_PREPASS
    )
    scene["opaque_bvh_skipped"] = not opq_live
    if _bvh_deferral_eligible(scene) and (
        tri_inputs is not None or bez_inputs is not None
    ):
        _rts = SETTINGS.raytracing
        placeholder = _empty_scene_part(device)
        if tri_inputs is not None:
            lo, hi, opaque = tri_inputs
            # Retained for the on-demand build (bez lo/hi are already stored
            # for the raster frontend; tri pair generation uses tri_screen).
            scene["tri_frame_lo"] = lo.contiguous()
            scene["tri_frame_hi"] = hi.contiguous()
            scene["tri_bvh"] = placeholder
            scene["tri_opaque_bvh"] = placeholder
        if bez_inputs is not None:
            scene["bez_bvh"] = placeholder
            scene["bez_opaque_bvh"] = placeholder
        scene["bvh_deferred"] = True
        scene["bvh_deferred_refit"] = bool(_rts.refit_bvh_active())
        return

    scene["bvh_deferred"] = False
    if tri_inputs is not None:
        lo, hi, opaque = tri_inputs
        # Median-split ordering: ~25% faster traversal than Morton at ~0.2s
        # extra build per batch; byte-identical for triangles (the depth-peel
        # is arrangement-invariant). PN/bezier BVHs stay Morton -- their
        # seam de-dup is discovery-order sensitive (see stbvh._BVH_BUILD).
        scene["tri_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=opaque,
            builder="split",
        )
        if not scene["tri_has_opaque"]:
            scene["tri_opaque_bvh"] = _empty_scene_part(device)
        elif not scene["tri_has_translucent"] or not opq_live:
            scene["tri_opaque_bvh"] = scene["tri_bvh"]
        else:
            scene["tri_opaque_bvh"] = _build_opaque_bvh(
                lo,
                hi,
                opaque,
                num_frames,
                RayTracedTrianglePrimitive.stbvh_tightness,
                builder="split",
            )
    if bez_inputs is not None:
        lo, hi, opaque = bez_inputs
        # ss3.4: bezier was the last type still pinned to Morton (PN, the other,
        # was deleted). Split ordering is a pure reorder, but a circuit's seam
        # de-dup is discovery-order sensitive, so it moves output at the epsilon
        # level -- hence the gate rather than a straight flip.
        bez_builder = "split" if SETTINGS.raytracing.BEZ_BVH_SPLIT else "morton"
        scene["bez_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque,
            builder=bez_builder,
        )
        if not scene["bez_has_opaque"]:
            scene["bez_opaque_bvh"] = _empty_scene_part(device)
        elif not scene["bez_has_translucent"] or not opq_live:
            scene["bez_opaque_bvh"] = scene["bez_bvh"]
        else:
            scene["bez_opaque_bvh"] = _build_opaque_bvh(
                lo,
                hi,
                opaque,
                num_frames,
                RayTracedBezierCircuitPrimitive.stbvh_tightness,
                builder=bez_builder,
            )


def build_deferred_bvhs(merged):
    """Build the STBVHs a deferred batch skipped (see ``_finalize_bvhs``).

    Called by the tracer the moment anything actually needs a tree: shadows,
    classic-wavefront routing, an actually spawned continuation ray, or the
    Monte Carlo path. Idempotent, and forces the tree kind recorded at merge
    time so the batch's placeholder and real trees always agree on the
    ``refit`` kernel template.
    """
    if not merged.get("bvh_deferred"):
        return
    refit = bool(merged.get("bvh_deferred_refit"))
    num_frames = int(merged["num_frames"])
    # Same opaque-tree skip as _finalize_bvhs (read live at this build).
    opq_live = (
        not SETTINGS.raytracing.OPAQUE_BVH_SKIP_DEAD
        or SETTINGS.raytracing.WF_OPAQUE_CLOSEST
        or SETTINGS.raytracing.WF_OPAQUE_PREPASS
    )
    merged["opaque_bvh_skipped"] = not opq_live
    if merged.get("num_triangles", 0) > 0 and merged.get("tri_frame_lo") is not None:
        lo = merged["tri_frame_lo"]
        hi = merged["tri_frame_hi"]
        opaque = merged["tri_frame_opaque"]
        merged["tri_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=opaque,
            builder="split",
            refit=refit,
        )
        if not merged["tri_has_opaque"]:
            merged["tri_opaque_bvh"] = _empty_scene_part(lo.device, refit=refit)
        elif not merged["tri_has_translucent"] or not opq_live:
            merged["tri_opaque_bvh"] = merged["tri_bvh"]
        else:
            merged["tri_opaque_bvh"] = _build_opaque_bvh(
                lo,
                hi,
                opaque,
                num_frames,
                RayTracedTrianglePrimitive.stbvh_tightness,
                builder="split",
                refit=refit,
            )
        merged["tri_frame_lo"] = None
        merged["tri_frame_hi"] = None
    if merged.get("num_circuits", 0) > 0:
        lo = merged["bez_frame_lo"]
        hi = merged["bez_frame_hi"]
        opaque = merged["bez_frame_opaque"]
        merged["bez_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque,
            refit=refit,
        )
        if not merged["bez_has_opaque"]:
            merged["bez_opaque_bvh"] = _empty_scene_part(lo.device, refit=refit)
        elif not merged["bez_has_translucent"] or not opq_live:
            merged["bez_opaque_bvh"] = merged["bez_bvh"]
        else:
            merged["bez_opaque_bvh"] = _build_opaque_bvh(
                lo,
                hi,
                opaque,
                num_frames,
                RayTracedBezierCircuitPrimitive.stbvh_tightness,
                refit=refit,
            )
    merged["bvh_deferred"] = False


def _build_mem_trim(scene, lo, hi, opaque, num_frames, device):
    """Build the 'Family A+B' memory-trim triangle arrays (see
    settings.WF_MEM_TRIM). Reorders prims into material-class bands -- band 0
    ``needs_mat`` (lit), band 1 ``needs_norm`` only (reflective / normal-mapped /
    promoted), band 2 bare (unlit matte) -- so that ``tri_norm`` and ``tri_mat``
    become compacted PREFIXES (needs_mat subset needs_norm, so both nest under a
    single permutation). ``tri_colors``/``tri_extra`` stay in their original
    (promotion-compacted) order, addressed by a per-prim remap ``col_row`` (-1 =
    promoted, colour/material from its 1x1 maps); ``tex_meta``/``uvs`` are widened
    to full band-order arrays indexed directly by prim. Byte-identical to the
    untrimmed path (only indexing/layout changes). Stores ``*_t`` variants +
    ``col_row`` + a band-reordered BVH; the wavefront picks them when engaged.
    """
    tri_pos = scene["tri_pos"].to(device)
    N = tri_pos.shape[1]
    if N == 0:
        scene["mem_trim_active"] = False
        return
    tri_norm = scene["tri_norm"].to(device)
    tri_mat = scene["tri_mat"].to(device)
    tri_mat_id = scene["tri_mat_id"].to(device)
    tri_extra = scene["tri_extra"].to(device)
    tri_uvs = scene["tri_uvs"].to(device)
    tri_tex_meta = scene["tri_tex_meta"].to(device)
    num_colored = int(scene["num_colored_triangles"])
    _UNLIT = 1
    Nc = tri_extra.shape[1]  # prims with a per-vertex colour/extra row

    lit = (tri_mat_id != _UNLIT).any(0)  # [N]
    refl = torch.zeros(N, dtype=torch.bool, device=device)
    if Nc > 0:
        e = tri_extra
        refl[:Nc] = ((e[..., 0] > 0) | (e[..., 2] > 0) | (e[..., 4] > 0)).any(0)
    promoted = torch.zeros(N, dtype=torch.bool, device=device)
    if Nc < N:
        promoted[Nc:] = True  # constant-material prims: value in 1x1 map
    normalmapped = torch.zeros(N, dtype=torch.bool, device=device)
    if tri_tex_meta.shape[0] > 0 and num_colored < N:
        nm = tri_tex_meta[:, 6] >= 0
        k = min(nm.shape[0], N - num_colored)
        normalmapped[num_colored : num_colored + k] = nm[:k]

    needs_mat = lit
    needs_norm = needs_mat | refl | promoted | normalmapped
    n_lit = int(needs_mat.sum().item())
    n_norm = int(needs_norm.sum().item())

    zeros = torch.zeros(N, dtype=torch.long, device=device)
    band = torch.where(needs_mat, zeros, torch.where(needs_norm, zeros + 1, zeros + 2))
    perm = torch.argsort(band, stable=True)  # band 0 first
    orig = perm  # orig idx of prim p

    tri_pos_t = tri_pos.index_select(1, perm).contiguous()
    tri_norm_t = tri_norm.index_select(1, perm)[:, : max(n_norm, 1)].contiguous()
    tri_mat_t = tri_mat.index_select(1, perm)[:, : max(n_lit, 1)].contiguous()
    tri_mat_id_t = tri_mat_id.index_select(1, perm).contiguous()
    col_row = (
        torch.where(orig < Nc, orig, torch.full_like(orig, -1))
        .to(torch.int32)
        .contiguous()
    )

    tex_meta_t = torch.zeros((N, 10), dtype=torch.int32, device=device)
    tex_meta_t[:, 0] = -1
    tex_meta_t[:, 3] = -1
    tex_meta_t[:, 6] = -1
    Tuv = tri_uvs.shape[0]
    tri_uvs_t = torch.zeros((Tuv, N, 6), dtype=tri_uvs.dtype, device=device)
    if tri_tex_meta.shape[0] > 0:
        has_meta = orig >= num_colored
        meta_src = (orig - num_colored).clamp(0, tri_tex_meta.shape[0] - 1)
        tex_meta_t = torch.where(
            has_meta.unsqueeze(1),
            tri_tex_meta.index_select(0, meta_src).int(),
            tex_meta_t,
        )
        uv_src = (orig - num_colored).clamp(0, tri_uvs.shape[1] - 1)
        tri_uvs_t = tri_uvs.index_select(1, uv_src) * has_meta.view(1, N, 1).to(
            tri_uvs.dtype
        )

    tri_bvh_t = _build_accel(
        lo.index_select(1, perm).contiguous(),
        hi.index_select(1, perm).contiguous(),
        num_frames=num_frames,
        tightness=RayTracedTrianglePrimitive.stbvh_tightness,
        opaque=opaque.index_select(1, perm).contiguous(),
        builder="split",
    )

    scene["tri_pos_t"] = tri_pos_t
    scene["tri_norm_t"] = tri_norm_t
    scene["tri_mat_t"] = tri_mat_t
    scene["tri_mat_id_t"] = tri_mat_id_t
    scene["tri_uvs_t"] = tri_uvs_t
    scene["tri_tex_meta_t"] = tex_meta_t
    scene["tri_col_row"] = col_row
    scene["tri_bvh_t"] = tri_bvh_t
    scene["mem_trim_active"] = True


def _promote_property_group(cv, present, num_frames, device):
    """Promote one per-corner property group of a flat-triangle batch to a
    texture bank, for the UNSUPPORTED legacy textured wavefront (see
    settings.WF_TEXTURED; kept for reference).

    ``cv`` is the per-corner value tensor ``[T, N, 3, C]`` (T frames, N
    triangles, 3 corners, C channels) and ``present`` a ``[N]`` bool mask of the
    triangles that actually carry this group (others get index -1 and sample
    nothing). A triangle whose three corners are equal in every frame is
    *constant across the surface* and is promoted to a shared 1x1 texture
    (grouped by value, so identical surfaces share one texel); one that varies
    per vertex gets its own 2x2 texture laid out ``[[v0, v1], [v2, v0]]`` so a
    bilinear lookup at the canonical corner UVs ``(0,0)/(1,0)/(0,1)``
    reproduces the corner values exactly and blends between them in the
    interior (an approximation of true barycentric interpolation).

    Returns ``(bank, meta, idx)``: ``bank`` is the flat texel buffer
    ``[Tb, num_texels, C]``, ``meta`` the ``[num_textures, 3]`` int32
    ``(offset, width, height)`` per texture and ``idx`` the ``[N]`` int32
    per-triangle texture index (-1 = absent).
    """
    T = num_frames
    cv = _expand_frames(cv, T).contiguous()
    N, C = cv.shape[1], cv.shape[3]
    idx = torch.full((N,), -1, dtype=torch.int32, device=device)
    tri_ids = torch.arange(N, device=device)
    # Constant across the surface: all three corners equal in every frame.
    const_mask = (cv == cv[:, :, :1, :]).all(3).all(2).all(0)  # [N]

    banks, metas = [], []
    texel_off = 0
    meta_base = 0

    def _emit(sel, texels_flat, per_tex_texels, w, h, inv):
        # texels_flat: [T, G * per_tex_texels, C]; inv: [len(sel)] group id.
        nonlocal texel_off, meta_base
        G = texels_flat.shape[1] // per_tex_texels
        banks.append(texels_flat)
        offs = (
            texel_off
            + torch.arange(G, device=device, dtype=torch.int32) * per_tex_texels
        )
        wv = torch.full((G,), w, dtype=torch.int32, device=device)
        hv = torch.full((G,), h, dtype=torch.int32, device=device)
        metas.append(torch.stack([offs, wv, hv], -1))
        idx[sel] = inv.to(torch.int32) + meta_base
        texel_off += G * per_tex_texels
        meta_base += G

    # Constant group -> one 1x1 texel per distinct value-over-time.
    cc = present & const_mask
    if bool(cc.any()):
        sel = tri_ids[cc]
        vals = cv[:, sel, 0, :]  # [T, nc, C]
        key = vals.permute(1, 0, 2).reshape(sel.numel(), T * C)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        G = uniq.shape[0]
        texels = uniq.reshape(G, T, C).permute(1, 0, 2).contiguous()  # [T,G,C]
        _emit(sel, texels, 1, 1, 1, inv)

    # Per-vertex group -> one 2x2 texture per distinct (v0, v1, v2)-over-time.
    cvary = present & ~const_mask
    if bool(cvary.any()):
        sel = tri_ids[cvary]
        vals = cv[:, sel, :, :]  # [T, nv, 3, C]
        key = vals.permute(1, 0, 2, 3).reshape(sel.numel(), T * 3 * C)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        G = uniq.shape[0]
        u = uniq.reshape(G, T, 3, C)
        v0, v1, v2 = u[:, :, 0, :], u[:, :, 1, :], u[:, :, 2, :]  # [G,T,C]
        # Column-major texel order (offset + cx*h + cy, h=2): texel(0,0)=v0,
        # texel(0,1)=v2, texel(1,0)=v1, texel(1,1)=v0 -> [[v0,v1],[v2,v0]].
        texs = torch.stack([v0, v2, v1, v0], 2)  # [G,T,4,C]
        texels = texs.permute(1, 0, 2, 3).reshape(T, G * 4, C).contiguous()
        _emit(sel, texels, 4, 2, 2, inv)

    if banks:
        bank = _dedup_time(torch.cat(banks, 1).contiguous())
        meta = torch.cat(metas, 0).contiguous()
    else:  # nothing in this group carries a texture
        bank = torch.zeros((1, 1, C), device=device)
        meta = torch.zeros((1, 3), dtype=torch.int32, device=device)
    return bank, meta, idx


def _build_textured_scene(scene, num_frames, device):
    """Build the three per-triangle texture banks the UNSUPPORTED legacy
    textured wavefront shades from (see settings.WF_TEXTURED; kept for
    reference), from the full per-vertex merged arrays (constant-promotion is
    disabled for this path so they span every triangle).

    Groups, each promoted independently by :func:`_promote_property_group`:

    * **colour** -- RGBA + glow (``tri_colors`` 5 channels, per vertex).
    * **surface** -- reflectivity / roughness / index-of-refraction (from
      ``tri_extra``, per vertex) used for scatter; index -1 for a matte surface
      (no reflectivity, no refraction) so the kernel skips the lookup.
    * **material** -- the shading parameter block prefixed with the pipeline id
      (``tri_mat_id`` + ``tri_mat``, per primitive, hence always 1x1); index -1
      for an unlit surface (no shading, colour passes through).

    Every triangle is assigned the canonical corner UVs ``(0,0)/(1,0)/(0,1)``.
    """
    T = num_frames
    tc = _expand_frames(scene["tri_colors"].to(device), T)  # [T,N,3,5]
    te = _expand_frames(scene["tri_extra"].to(device), T)  # [T,N,15]
    tm = _expand_frames(scene["tri_mat"].to(device), T)[..., :MAT_W]  # [T,N,MAT_W]
    tmi = _expand_frames(scene["tri_mat_id"].to(device), T)  # [T,N]
    N = tc.shape[1]

    # Colour: every triangle carries a colour.
    present = torch.ones(N, dtype=torch.bool, device=device)
    col_bank, col_meta, col_idx = _promote_property_group(tc, present, T, device)

    # Surface (scatter): per-corner (metalness, roughness, IOR, transmission)
    # gathered from tri_extra cols {0,2,4}/{1,3,5}/{6,7,8}/{9,10,11}.
    c0 = torch.stack([te[..., 0], te[..., 1], te[..., 6], te[..., 9]], -1)
    c1 = torch.stack([te[..., 2], te[..., 3], te[..., 7], te[..., 10]], -1)
    c2 = torch.stack([te[..., 4], te[..., 5], te[..., 8], te[..., 11]], -1)
    surf_corner = torch.stack([c0, c1, c2], 2)  # [T,N,3,4]
    refl = surf_corner[..., 0]
    transmission = surf_corner[..., 3]
    surf_present = (refl >= 0.0).any(0).any(-1) | (transmission > 1e-6).any(0).any(
        -1
    )  # [N]
    surf_bank, surf_meta, surf_idx = _promote_property_group(
        surf_corner, surf_present, T, device
    )

    # Material (shading): [pipeline id | 12-slot param block], per primitive so
    # always constant across the corners -> promotes to 1x1. Fed as a degenerate
    # per-corner tensor (all three corners equal) so it shares the promoter.
    mat_vec = torch.cat([tmi.unsqueeze(-1).float(), tm], -1)  # [T,N,13]
    lit = (tmi != _MID_UNLIT).any(0)  # [N]
    mat_corner = mat_vec.unsqueeze(2).expand(T, N, 3, 13)
    mat_bank, mat_meta, mat_idx = _promote_property_group(mat_corner, lit, T, device)

    scene["tx_color_bank"] = col_bank
    scene["tx_color_meta"] = col_meta
    scene["tx_color_idx"] = col_idx
    scene["tx_surf_bank"] = surf_bank
    scene["tx_surf_meta"] = surf_meta
    scene["tx_surf_idx"] = surf_idx
    scene["tx_mat_bank"] = mat_bank
    scene["tx_mat_meta"] = mat_meta
    scene["tx_mat_idx"] = mat_idx
    # Normal-map bank (feature): placeholder / index -1 for every triangle until
    # a Surface carries a normal map (the normal-map feature measures the
    # compiled-in cost; real maps would be promoted here like the colour bank).
    scene["tx_nmap_bank"] = torch.zeros((1, 1, 3), device=device)
    scene["tx_nmap_meta"] = torch.zeros((1, 3), dtype=torch.int32, device=device)
    scene["tx_nmap_idx"] = torch.full((N,), -1, dtype=torch.int32, device=device)
    # Canonical per-triangle corner UVs (shared, constant across frames).
    scene["tx_uv"] = (
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], device=device)
        .view(1, 1, 6)
        .expand(1, N, 6)
        .contiguous()
    )


def _merge_scene(primitives):
    """Merge the batch's collections into one set per geometry type --
    triangles and bezier circuits, each with a single STBVH
    over all frames -- cached for the batch.
    """
    first = primitives[0]
    cached = getattr(first, "_rt_merged_scene", None)
    if cached is not None:
        return cached

    _rts = SETTINGS.raytracing

    # By default the merge + STBVH build run on the render device (much faster
    # than the CPU build) rather than on the projected primitives' source
    # (CPU) device. The transient out-of-place peak of this build -- inputs
    # relocated to the device plus all cat / sort / BVH-pyramid scratch plus
    # the merged output -- lives in the render pool's non-arena headroom and is
    # bounded by the render-arena preflight's ``MERGE_GPU_PEAK_FACTOR``
    # estimate. ``MERGE_TRACK_PEAK`` optionally measures the exact peak here to
    # calibrate that factor (it resets the process peak counter, so it stays
    # opt-in and off during profiling runs).
    gpu_merge = _rts.merge_on_gpu_active()
    track_peak = gpu_merge and _rts.MERGE_TRACK_PEAK
    peak_token = None
    if gpu_merge:
        device = _RENDER_DEVICE
        if track_peak:
            peak_token = begin_cuda_peak(device)
        _upload_primitive_inputs(primitives, device)
        empty_cache(force_gc=False)
    else:
        device = _projected_scene_device(primitives)
        if device.type != "cpu":
            empty_cache(force_gc=False)
    triangles = [p for p in primitives if isinstance(p, RayTracedTrianglePrimitive)]
    beziers = [p for p in primitives if isinstance(p, RayTracedBezierCircuitPrimitive)]
    unknown = [p for p in primitives if p not in triangles and p not in beziers]
    if unknown:
        raise TypeError(
            "The ray traced renderer can only draw ray traced primitives; "
            f"got {[type(p).__name__ for p in unknown]}."
        )
    num_frames = max(p._rt_num_frames for p in primitives)

    scene = {}

    def _texture_alpha_is_opaque(tex):
        """Conservatively prove that a color texture cannot cut a surface."""
        if tex is None or tex.shape[-1] < 4:
            return True
        return bool((tex[..., 3] >= 1.0 - 1e-6).all())

    def _record_visibility(prefix, lo, hi, opaque, uncertain_alpha=False):
        visible = (hi >= lo).all(-1)
        has_visible = bool(visible.any())
        # A point-degenerate primitive cannot cover a pixel in the current
        # triangle/circuit intersection kernels.  Preserve a conservative
        # batch-wide bit so the sparse raster path can skip exact COUNT
        # discovery when every materialized primitive is collapsed to a point.
        has_extent = bool((visible & ((hi - lo) > 0.0).any(-1)).any())
        has_opaque = bool((visible & opaque).any())
        has_translucent = bool((visible & ~opaque).any())
        if uncertain_alpha and has_visible:
            has_translucent = True
        scene[f"{prefix}_has_visible"] = has_visible
        scene[f"{prefix}_has_extent"] = has_extent
        scene[f"{prefix}_has_opaque"] = has_opaque
        scene[f"{prefix}_has_translucent"] = has_translucent
        scene["has_uncertain_texture_alpha"] = scene.get(
            "has_uncertain_texture_alpha", False
        ) or (uncertain_alpha and has_visible)
        return has_visible, has_opaque, has_translucent

    # Shared flat texel buffer for *all* texture maps (color / material /
    # normal). Each map is appended once, padded to 5 channels and flattened to
    # [T, W*H, 5]; its placement is a (offset, w, h) triplet recorded in the
    # consuming geometry's metadata (offset -1 = no map), keyed by tri_tex_meta.
    # Assembled into scene["textures"] once the geometry blocks below have
    # appended.
    _texture_tensors = []
    _texel_offset = [0]

    def _append_texture(tex):
        if tex is None:
            return (-1, 0, 0)
        if tex.dim() == 3:  # [W, H, C]
            tex = tex.unsqueeze(0)  # [1, W, H, C]
        w, h, c = tex.shape[-3], tex.shape[-2], tex.shape[-1]
        if c < 5:
            tex = torch.cat((tex, tex.new_zeros((*tex.shape[:-1], 5 - c))), -1)
        # Flatten W and H (dimensions 1 and 2).
        _texture_tensors.append(tex.reshape(tex.shape[0], -1, 5))
        o = _texel_offset[0]
        _texel_offset[0] += w * h
        return (o, w, h)

    scene["tex_has_refractive"] = False
    scene["tex_has_reflective"] = False
    # Per-geometry BVH build inputs, captured by the merge sections below and
    # consumed at the end by ``_finalize_bvhs`` (which either builds the trees
    # or, for batches that provably never traverse one, defers them).
    tri_bvh_inputs = bez_bvh_inputs = None
    if triangles:
        # Constant-property promotion: triangles whose colour + material params
        # are constant across their corners (and frames) are rendered from a
        # shared 1x1 colour + material map instead of per-vertex tri_colors /
        # tri_extra rows (see _split_promotable). Detection is per triangle and
        # grouped by value, so a uniform mob is promoted even when it was batched
        # into one primitive alongside differently-coloured mobs. Promoted
        # triangles are ordered LAST (their prims sit past the shrunk arrays,
        # which the guarded kernel reads never index). With promotion inactive
        # every triangle is kept and this reduces byte-identically to the plain
        # per-vertex merge (see _sel: an all-keep selection returns the original
        # tensor, uncopied).
        _rts = SETTINGS.raytracing
        # The textured wavefront does its own (three-group) constant/per-vertex
        # promotion from the full per-vertex arrays, so the built-in single-map
        # promotion is turned off for it (it would shrink tri_colors/tri_extra
        # out from under the texture builder).
        promote = _constant_promotion_active() and not _rts.WF_TEXTURED
        plain_triangles = [
            p for p in triangles if getattr(p, "_rt_tri_uvs", None) is None
        ]
        textured_triangles = [
            p for p in triangles if getattr(p, "_rt_tri_uvs", None) is not None
        ]
        keep_idx, promo_idx, promo_meta = {}, {}, {}
        # See _sel below: identity-ness memoized per index tensor. Seeded
        # here for the promotion-inactive arange (identity by construction,
        # no device sync needed at all).
        _sel_identity = {}
        for p in plain_triangles:
            if promote:
                k, pr, meta = _split_promotable(p, _append_texture, device, scene)
            else:
                Np = p._rt_tri_pos.shape[1]
                k = torch.arange(Np, device=device)
                pr = torch.zeros((0,), dtype=torch.long, device=device)
                meta = torch.zeros((0, 10), dtype=torch.int32, device=device)
                _sel_identity[id(k)] = True
            keep_idx[id(p)] = k
            promo_idx[id(p)] = pr
            promo_meta[id(p)] = meta

        # Whether an index tensor is the identity selection is a property of
        # the tensor alone, and every per-primitive index is reused for ~18
        # different arrays below -- memoize the (synchronizing) device->host
        # equality test per index object (``_sel_identity``, seeded above) so
        # each primitive drains the queue at most once, not once per array.
        # The key tensors are held alive by keep_idx/promo_idx for the whole
        # merge, so id() keys are stable.
        def _sel(arr, idx):
            # Index the primitive axis (dim 1) by ``idx``. Only an *identity*
            # selection (every prim, in order) may return the original tensor
            # uncopied -- that keeps the promotion-inactive path byte-identical.
            # ``promo_idx`` covers every prim too (when a whole primitive is
            # promoted) but is a *permutation* (grouped by value, see
            # _split_promotable), so it must still be applied: skipping it would
            # leave the geometry in source order while ``promo_meta`` is in
            # group order, pairing each triangle with another group's maps.
            if idx.numel() == arr.shape[1]:
                identity = _sel_identity.get(id(idx))
                if identity is None:
                    identity = bool(
                        (idx == torch.arange(idx.numel(), device=idx.device)).all()
                    )
                    _sel_identity[id(idx)] = identity
                if identity:
                    return arr
            return arr.index_select(1, idx.to(arr.device))

        def _geom(name):
            # Global order: kept triangles of the plain primitives, then the
            # whole textured primitives, then the promoted triangles. Empty
            # selections are dropped so the promotion-inactive path passes each
            # original tensor through _cat_collections uncopied.
            keep = [
                _sel(getattr(p, name), keep_idx[id(p)])
                for p in plain_triangles
                if keep_idx[id(p)].numel()
            ]
            tex = [getattr(p, name) for p in textured_triangles]
            promo = [
                _sel(getattr(p, name), promo_idx[id(p)])
                for p in plain_triangles
                if promo_idx[id(p)].numel()
            ]
            return keep + tex + promo

        num_colored = sum(int(keep_idx[id(p)].numel()) for p in plain_triangles)
        scene["num_colored_triangles"] = num_colored
        _tri_parts = _geom("_rt_tri_pos")
        # Per-triangle SOURCE-SURFACE id, [T?, N] (DESIGN_analytic_aa_v2.md
        # ss4.2): the run rule sums exact clipped areas within one surface and
        # composites between surfaces. Built per primitive at pack time --
        # per-MEMBER within a batched collection (one part is NOT one surface:
        # the batcher merges every same-identifier mob into one), and per FRAME
        # for diced logical PN, whose row->patch mapping moves with the
        # adaptive levels. Offset per primitive here so ids are globally
        # unique; a primitive's kept and promoted slices share its offset, so
        # promotion cannot split a surface in two.
        _obj_base = 0
        for p in plain_triangles + textured_triangles:
            p._rt_tri_obj_global = p._rt_tri_obj + _obj_base
            _obj_base += int(getattr(p, "_rt_tri_obj_n", 1))
        scene["tri_obj"] = (
            _cat_collections(_geom("_rt_tri_obj_global"), 1, "triangle merge")
            .to(torch.int32)
            .contiguous()
        )
        scene["tri_pos"] = _cat_collections(_tri_parts, 1, "triangle merge")
        scene["tri_norm"] = _cat_collections(_geom("_rt_tri_norm"), 1, "triangle merge")
        scene["tri_mat_id"] = _cat_collections(
            _geom("_rt_tri_mat_id"), 1, "triangle merge"
        )
        scene["tri_mat"] = _cat_mat_blocks(_geom("_rt_tri_mat"), "triangle merge")
        lo = _cat_collections(_geom("_rt_frame_lo"), 1, "triangle merge")
        hi = _cat_collections(_geom("_rt_frame_hi"), 1, "triangle merge")
        opaque = _cat_collections(_geom("_rt_frame_opaque"), 1, "triangle merge")
        # Per-primitive color-texture alpha certainty for the hybrid raster
        # frontend.  The old global ``has_uncertain_texture_alpha`` flag forced
        # every triangle through the transparent path when a single cutout
        # texture was present, disabling otherwise-useful opaque z culling.
        # Keep the aggregate flag for the classic opaque-prepass gate, but also
        # preserve the exact primitive mask in merged triangle order.
        alpha_uncertain_parts = []
        for p in plain_triangles:
            nk = int(keep_idx[id(p)].numel())
            if nk:
                alpha_uncertain_parts.append(
                    torch.zeros((1, nk), dtype=torch.bool, device=device)
                )
        for p in textured_triangles:
            uncertain = getattr(
                p, "_rt_texture_map", None
            ) is not None and not _texture_alpha_is_opaque(p._rt_texture_map)
            alpha_uncertain_parts.append(
                torch.full(
                    (1, p._rt_tri_pos.shape[1]),
                    bool(uncertain),
                    dtype=torch.bool,
                    device=device,
                )
            )
        for p in plain_triangles:
            npromo = int(promo_idx[id(p)].numel())
            if npromo:
                alpha_uncertain_parts.append(
                    torch.zeros((1, npromo), dtype=torch.bool, device=device)
                )
        tri_alpha_uncertain = (
            torch.cat(alpha_uncertain_parts, 1)
            if alpha_uncertain_parts
            else torch.zeros(
                (1, scene["tri_pos"].shape[1]), dtype=torch.bool, device=device
            )
        )
        scene["tri_alpha_uncertain"] = tri_alpha_uncertain.contiguous()
        tri_uncertain_alpha = bool(tri_alpha_uncertain.any())
        _record_visibility("tri", lo, hi, opaque, tri_uncertain_alpha)

        # tri_colors / tri_extra span only the kept per-vertex triangles + the
        # textured primitives (a textured primitive may carry only material /
        # normal maps and fall back to per-vertex colour, color-map offset -1).
        # Promoted triangles have no row here; guarded kernel reads keep their
        # (past-the-end) prims from ever indexing these.
        vcolors = [
            _sel(p._rt_tri_colors, keep_idx[id(p)])
            for p in plain_triangles
            if keep_idx[id(p)].numel()
        ] + [p._rt_tri_colors for p in textured_triangles]
        vextra = [
            _sel(p._rt_tri_extra, keep_idx[id(p)])
            for p in plain_triangles
            if keep_idx[id(p)].numel()
        ] + [p._rt_tri_extra for p in textured_triangles]
        if any(t.shape[1] for t in vcolors):
            scene["tri_colors"] = _cat_collections(vcolors, 1, "triangle merge")
            scene["tri_extra"] = _cat_collections(vextra, 1, "triangle merge")
        else:  # every triangle promoted -> minimal placeholder rows
            scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
            scene["tri_extra"] = torch.zeros((1, 1, 15), device=device)

        # Any promoted group synthesises material maps. Promotion only runs
        # for deterministic fragment-shading renders (see
        # _constant_promotion_active), whose wavefront shade kernel guards
        # every per-vertex read, so the shrunk tri_colors/tri_extra arrays are
        # never mis-indexed.
        has_promoted = any(promo_idx[id(p)].numel() for p in plain_triangles)
        scene["has_material_textures"] = bool(has_promoted) or any(
            getattr(p, "_rt_material_texture", None) is not None
            or getattr(p, "_rt_normal_texture", None) is not None
            for p in textured_triangles
        )

        # UVs + tex-meta cover the [textured ++ promoted] tiers, indexed by
        # ``prim - num_colored_triangles``. Meta layout: cols 0-2 color map, 3-5
        # material map (reflectivity, roughness, index of refraction), 6-8 normal
        # map, 9 bitmask of texture-driven material properties (offset -1 = no
        # map -> per-vertex fallback).
        meta_parts, uvs_parts = [], []
        for p in textured_triangles:
            color_meta = _append_texture(p._rt_texture_map)
            mtex = getattr(p, "_rt_material_texture", None)
            material_meta = _append_texture(mtex)
            normal_meta = _append_texture(getattr(p, "_rt_normal_texture", None))
            flags = int(getattr(p, "_rt_material_flags", 0) or 0)
            if mtex is not None and (flags & 8) and bool((mtex[..., 3] > 1e-6).any()):
                scene["tex_has_refractive"] = True
            # A metalness-driving material map may produce a Fresnel lobe;
            # deliberately coarse (any such map counts) -- a false positive
            # only keeps the BVHs eagerly built.
            if mtex is not None and (flags & 1):
                scene["tex_has_reflective"] = True
            meta_parts.append(
                torch.tensor(
                    [*color_meta, *material_meta, *normal_meta, flags],
                    dtype=torch.int32,
                    device=device,
                )
                .view(1, 10)
                .expand(p._rt_tri_pos.shape[1], 10)
            )
            uvs_parts.append(p._rt_tri_uvs)
        for p in plain_triangles:
            n = int(promo_idx[id(p)].numel())
            if n:
                # A 1x1 map ignores UVs (both texels clamp to index 0), so a
                # single-frame zero UV row per promoted triangle suffices.
                meta_parts.append(promo_meta[id(p)])
                uvs_parts.append(torch.zeros((1, n, 6), device=device))
        if meta_parts:
            scene["tri_tex_meta"] = torch.cat(meta_parts, 0).contiguous()
            scene["tri_uvs"] = _cat_collections(uvs_parts, 1, "triangle merge")
        else:
            scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
            scene["tri_tex_meta"] = torch.full(
                (1, 10), -1, dtype=torch.int32, device=device
            )

        # Collapse temporally-constant triangle tables to one frame. Every
        # consumer reads their time axis as ``f % shape[0]`` (kernels) or
        # ``_expand_frames`` (raster host tables), and _build_mem_trim below
        # is T-agnostic, so a batch whose materials/normals/colours do not
        # animate stores one row instead of T -- tri_mat alone is [T, N, 26],
        # tens of MB of identical frames on ordinary scenes (rigid motion
        # lives in tri_pos, which is deliberately not collapsed).
        if SETTINGS.raytracing.MERGE_DEDUP_TIME:
            for _k in (
                "tri_norm",
                "tri_mat_id",
                "tri_mat",
                "tri_colors",
                "tri_extra",
                "tri_uvs",
            ):
                scene[_k] = _dedup_time(scene[_k])

        # Per-(frame, prim) visibility/opacity masks for the hybrid raster
        # front-end (settings.HYBRID_RASTER): candidate emission skips
        # invisible triangles and routes proven-opaque ones to the z-prepass.
        # Derived from the same bounds/opacity arrays the STBVH build uses.
        scene["tri_frame_valid"] = (hi >= lo).all(-1).contiguous()
        scene["tri_frame_opaque"] = opaque.contiguous()
        # Triangle STBVHs are built (or deferred) in _finalize_bvhs once every
        # routing flag this batch needs is known.
        tri_bvh_inputs = (lo, hi, opaque)
        if _rts.WF_MEM_TRIM:
            _build_mem_trim(scene, lo, hi, opaque, num_frames, device)
    else:
        scene["tri_pos"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_extra"] = torch.zeros((1, 1, 15), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
        scene["tri_tex_meta"] = torch.full(
            (1, 10), -1, dtype=torch.int32, device=device
        )
        scene["num_colored_triangles"] = 0
        scene["has_material_textures"] = False
        scene["tri_mat_id"] = torch.zeros((1, 1), dtype=torch.int32, device=device)
        scene["tri_obj"] = torch.zeros((1, 1), dtype=torch.int32, device=device)
        scene["tri_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)
        scene["tri_opaque_bvh"] = scene["tri_bvh"]
        scene["tri_frame_valid"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["tri_frame_opaque"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["tri_alpha_uncertain"] = torch.zeros(
            (1, 1), dtype=torch.bool, device=device
        )
        _record_visibility(
            "tri",
            torch.empty((0, 0, 3), device=device),
            torch.empty((0, 0, 3), device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
        )
    scene["num_triangles"] = scene["tri_pos"].shape[1] if triangles else 0

    # Any triangle whose material can produce a nonzero Fresnel lobe (mirrors
    # _material_reflectance: per-corner metalness in tri_extra cols 0/2/4, -1
    # = non-PBR sentinel with R = 0 exactly; metalness 0 still reflects when
    # the paired per-corner IOR in cols 6-8 exceeds 1). Promoted / material-
    # texture-driven metalness is tracked by ``tex_has_reflective``. Consumed
    # by the BVH deferral predicate: a reflective surface can spawn a
    # continuation ray, which needs the trees.
    if triangles:
        e = scene["tri_extra"]
        refl = e[..., 0:6:2]
        ior = e[..., 6:9].abs()
        scene["tri_has_reflective"] = bool(
            ((refl > 0.0) | ((refl >= 0.0) & (ior > 1.0 + 1e-4))).any()
            or scene.get("tex_has_reflective")
        )
        # A STRONG reflector -- metallic, not merely a dielectric's ~4% Fresnel
        # sheen. Consumed by the tracer to decide whether a batch is worth
        # over-allocating the split ray pool for: every PBR dielectric is
        # "reflective" by the test above, and treating those as splitting
        # quarters the tile size for a lobe nobody can see.
        scene["has_strong_reflective"] = bool(
            (refl > 0.2).any() or scene.get("tex_has_reflective")
        )
    else:
        scene["tri_has_reflective"] = bool(scene.get("tex_has_reflective"))
        scene["has_strong_reflective"] = bool(scene.get("tex_has_reflective"))

    # Assemble the shared texel buffer now that the flat-triangle block above
    # has appended its maps (offsets recorded in tri_tex_meta).
    if _texture_tensors:
        scene["textures"] = _cat_collections(_texture_tensors, 1, "texture merge")
    else:
        scene["textures"] = torch.zeros((1, 1, 5), device=device)

    # Refraction is active iff some triangle surface transmits (extra
    # columns 9-11, per-corner). Transmission -- not the IOR, which every PBR
    # material carries -- is what says light passes through; it gates the
    # wavefront's refraction template (and with it the split pool).
    def _extra_has_refractive(extra):
        return bool((extra[..., 9:12] > 1e-6).any())

    scene["has_refractive"] = _extra_has_refractive(scene["tri_extra"]) or bool(
        scene.get("tex_has_refractive")
    )

    # A surface that is both PBR-reflective and semi-transparent must trace its
    # reflection *and* its pass-through (see ``default_scatter``), which costs a
    # split pool slot -- so such a batch turns the refraction template on too.
    # Every PBR material has a non-zero Fresnel reflectance, so the test is
    # simply "metalness is set" (>= 0; -1 is the non-PBR sentinel) on a visible
    # non-opaque primitive. Deliberately conservative: a false positive only
    # costs a split slot, while a false negative drops a continuation.
    def _pbr_from_extra(attr):
        def mask(p):
            extra = getattr(p, attr, None)
            if extra is None or extra.shape[1] == 0:
                return None
            # Interleaved per-corner (metalness, roughness) in cols 0-5, so
            # metalness is 0/2/4 (matches ``_triangle_extra``).
            return (extra[..., 0:6:2] >= 0.0).any(0).any(-1)

        return mask

    def _pbr_from_circuit_meta(p):
        meta = getattr(p, "_rt_circuit_meta", None)
        if meta is None or meta.shape[1] == 0:
            return None
        return (meta[..., _M_REFLECTIVITY] >= 0.0).any(0)

    def _has_refl_transparent(prims, pbr_mask):
        for p in prims:
            lo = getattr(p, "_rt_frame_lo", None)
            has_pbr = pbr_mask(p)
            if lo is None or has_pbr is None:
                continue
            mtex = getattr(p, "_rt_material_texture", None)
            # Material-map bit 0 drives metalness from channel 0 (see
            # ``_tri_hit_material``).
            if (
                mtex is not None
                and (int(getattr(p, "_rt_material_flags", 0) or 0) & 1)
                and bool((mtex[..., 0] >= 0.0).any())
            ):
                has_pbr = torch.ones_like(has_pbr)
            visible = (p._rt_frame_hi >= lo).all(-1)
            translucent = (visible & ~p._rt_frame_opaque).any(0)
            if not _texture_alpha_is_opaque(getattr(p, "_rt_texture_map", None)):
                translucent = translucent | visible.any(0)
            if bool((has_pbr.to(translucent.device) & translucent).any()):
                return True
        return False

    scene["has_refl_transparent"] = _has_refl_transparent(
        triangles, _pbr_from_extra("_rt_tri_extra")
    ) or _has_refl_transparent(beziers, _pbr_from_circuit_meta)

    # Does anything in the batch pass light through itself? Shadow rays need
    # to know: a transmissive surface attenuates rather than blocks (see
    # ``raytrace_kernels_taichi._shadow_pass_through``), which is exactly the
    # thing the opaque any-hit shadow modes assume cannot happen -- they answer
    # "is there any hit" and treat a hit as full occlusion. ``tracer`` keeps
    # such a batch on the ordered march. Deliberately conservative in the same
    # way as ``has_refl_transparent``: a false positive costs a slower shadow
    # query, a false negative renders a black shadow under a pane of glass.
    def _any_transmissive(prims, attr, cols):
        for p in prims:
            extra = getattr(p, attr, None)
            if extra is None or extra.shape[1] == 0:
                continue
            if bool((extra[..., cols] > 0.0).any()):
                return True
        return False

    scene["has_transmissive"] = _any_transmissive(
        triangles, "_rt_tri_extra", slice(9, 12)
    ) or _any_transmissive(beziers, "_rt_circuit_meta", _M_TRANSMISSION)

    if beziers:
        scene["circuit_meta"] = _cat_collections(
            [p._rt_circuit_meta for p in beziers], 1, "bezier merge"
        )
        scene["circuit_border_colors"] = _cat_circuit_color_grids(
            [p._rt_circuit_border_colors for p in beziers]
        )
        scene["circuit_colors"] = _cat_circuit_color_grids(
            [p._rt_circuit_colors for p in beziers]
        )
        scene["edges_2d"] = _cat_collections(
            [p._rt_edges for p in beziers], 1, "bezier merge"
        )
        # Same time-band collapse as the triangle tables (consumers all read
        # ``f % shape[0]``). edges_2d holds LOCAL-plane coordinates, so text
        # that only moves/rotates/fades keeps it constant; collapsing it
        # before build_bezier_edge_acceleration below also builds the accel
        # tables for one frame instead of T (its header stride is
        # ``f % edges_2d.shape[0]``, so the two stay consistent by
        # construction).
        if SETTINGS.raytracing.MERGE_DEDUP_TIME:
            for _k in (
                "circuit_meta",
                "circuit_colors",
                "circuit_border_colors",
                "edges_2d",
            ):
                scene[_k] = _dedup_time(scene[_k])
        # Degenerate sampled edges use the exact sentinel row installed by
        # BezierCircuitPrimitives._build_circuit_geometry.  A batch containing
        # no other edge cannot pass the circuit intersection/winding test even
        # when border/glow inflation gave its point bounds nonzero extent.
        scene["bez_has_nondegenerate_edges"] = bool(
            (~(scene["edges_2d"][..., :4] == 1e9).all(-1)).any()
        )
        offsets, shift = [torch.zeros((1,), dtype=torch.int32, device=device)], 0
        for p in beziers:
            offsets.append(p._rt_edge_offsets[1:].long() + shift)
            shift = shift + p._rt_edges.shape[1]
        edge_offsets = torch.cat([o.to(torch.int32) for o in offsets]).contiguous()
        scene["edge_accel"] = build_bezier_edge_acceleration(
            scene["edges_2d"], edge_offsets
        )
        lo = _cat_collections([p._rt_frame_lo for p in beziers], 1, "bezier merge")
        hi = _cat_collections([p._rt_frame_hi for p in beziers], 1, "bezier merge")
        opaque = _cat_collections(
            [p._rt_frame_opaque for p in beziers], 1, "bezier merge"
        )
        _record_visibility("bez", lo, hi, opaque)
        # Per-(frame, circuit) visibility, opacity, and AABBs for the hybrid
        # raster frontend.  Proven-opaque circuits now participate in the typed
        # visibility buffer and cull geometry behind large filled 2D shapes;
        # translucent / reflective panes remain in the ordered fragment stream.
        scene["bez_frame_valid"] = (hi >= lo).all(-1).contiguous()
        scene["bez_frame_opaque"] = opaque.contiguous()
        scene["bez_frame_lo"] = lo.contiguous()
        scene["bez_frame_hi"] = hi.contiguous()
        # Bezier STBVHs are built (or deferred) in _finalize_bvhs.
        bez_bvh_inputs = (lo, hi, opaque)
        scene["num_circuits"] = scene["circuit_meta"].shape[1]
        # Any PBR circuit, not just a metallic one: metalness >= 0 is the whole
        # test, because a dielectric (metalness 0) still reflects its Fresnel
        # F0 (>= 4%). The legacy sorted pipeline (unsupported) composites
        # circuits with no reflectance term at all, so it may only be routed
        # when every circuit is non-PBR (metalness -1, reflectance exactly 0).
        scene["bez_has_reflective"] = bool(
            (scene["circuit_meta"][..., _M_REFLECTIVITY] >= 0.0).any()
        )
    else:
        scene["circuit_meta"] = torch.zeros((1, 1, _M_WIDTH), device=device)
        scene["circuit_colors"] = torch.zeros((1, 1, 1, 5), device=device)
        scene["circuit_border_colors"] = torch.zeros((1, 1, 1, 5), device=device)
        # Width 6: [x0, y0, x1, y1, border_visible, inward_sign] -- the sixth
        # column is the wedge's flatten-time sigma (DESIGN_analytic_aa_v2.md
        # ss5.2), present on every build so the kernels may read it
        # unconditionally.
        scene["edges_2d"] = torch.zeros((1, 1, 6), device=device)
        scene["edge_accel"] = torch.zeros((1,), dtype=torch.int32, device=device)
        scene["bez_bvh"] = _empty_scene_part(device)
        scene["bez_opaque_bvh"] = scene["bez_bvh"]
        scene["bez_frame_valid"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["bez_frame_opaque"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["bez_frame_lo"] = torch.full((1, 1, 3), EMPTY_LO, device=device)
        scene["bez_frame_hi"] = torch.full((1, 1, 3), EMPTY_HI, device=device)
        scene["num_circuits"] = 0
        scene["bez_has_reflective"] = False
        scene["bez_has_nondegenerate_edges"] = False
        _record_visibility(
            "bez",
            torch.empty((0, 0, 3), device=device),
            torch.empty((0, 0, 3), device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
        )

    scene["num_frames"] = num_frames
    # Host-side render classification.  The uploaded material-id tensors are
    # kernel data; renderer dispatch/bucketing must not run CUDA reductions or
    # uniqueness passes after the arena has consumed the device allowance.
    ids = scene["tri_mat_id"].detach().cpu()
    scene["tri_material_ids"] = tuple(
        int(value) for value in torch.unique(ids).tolist()
    )
    scene["has_user_pipeline"] = any(
        material_id >= _USER_PIPELINE_BASE for material_id in scene["tri_material_ids"]
    )
    scene["has_any_visible"] = any(
        scene[f"{prefix}_has_visible"] for prefix in ("tri", "bez")
    )
    scene["has_any_opaque"] = any(
        scene[f"{prefix}_has_opaque"] for prefix in ("tri", "bez")
    )
    scene["has_any_translucent"] = any(
        scene[f"{prefix}_has_translucent"] for prefix in ("tri", "bez")
    )
    scene["all_visible_opaque"] = (
        scene["has_any_visible"] and not scene["has_any_translucent"]
    )

    # UNSUPPORTED legacy texture-lookup shading (Surface / flat-triangle
    # scenes only: no bezier circuits; opt-in via WF_TEXTURED,
    # kept for reference). Builds the three per-triangle texture banks +
    # indexes the textured wavefront kernel consumes.
    scene["textured_active"] = False
    _rts = SETTINGS.raytracing
    if _rts.WF_TEXTURED and scene["num_triangles"] > 0 and scene["num_circuits"] == 0:
        _build_textured_scene(scene, num_frames, device)
        scene["textured_active"] = True

    # Build the per-geometry STBVHs -- or, for batches that provably never
    # traverse one (hybrid-raster primaries, no shadows, no reflective /
    # refractive / scatter materials), defer them entirely; the tracer builds
    # them on demand via ``build_deferred_bvhs`` if anything changes its mind.
    _finalize_bvhs(scene, tri_bvh_inputs, bez_bvh_inputs, num_frames, device)

    # The merged tensors replace the per-collection ones; release the
    # originals so peak GPU memory stays close to one copy of the scene.
    for p in triangles:
        p._rt_tri_pos = p._rt_tri_norm = None
        p._rt_tri_extra = p._rt_tri_colors = None
        p._rt_tri_mat_id = p._rt_tri_mat = None
        p._rt_tri_uvs = p._rt_texture_map = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None

    if device.type != "cpu":
        empty_cache(force_gc=False)
    # Measured transient device bytes the build allocated above the pre-merge
    # baseline, when opt-in peak tracking is on (see settings.MERGE_TRACK_PEAK);
    # -1 marks "not measured". Purely diagnostic -- the arena preflight bounds
    # the build with the MERGE_GPU_PEAK_FACTOR estimate, not this value.
    if track_peak:
        scene["_gpu_merge_peak_bytes"] = int(end_cuda_peak(peak_token))
    else:
        scene["_gpu_merge_peak_bytes"] = -1
    _decode_merged_colors(scene)
    first._rt_merged_scene = scene
    return scene


#: The merged arrays holding authored colour. Each is ``[..., 5]`` --
#: ``[r, g, b, glow, alpha]`` -- so only channels 0:3 are colour. Glow is an
#: additive emissive strength and alpha is coverage; neither is a colour and
#: neither is decoded.
_MERGED_COLOR_KEYS = ("tri_colors", "circuit_colors", "circuit_border_colors")


def _decode_merged_colors(scene):
    """Decode the batch's authored colour into the linear working space.

    This is the geometry half of the render boundary -- the single point where
    every primitive's colour crosses from display-referred (what the author
    typed, and what ``Mob.color`` still reads back) into the linear light the
    shading and compositing arithmetic needs. It runs once per batch, on the
    merged arrays, just before they are cached.

    Deliberately *not* done in :class:`~algan.constants.color.Color`: that is a
    ``torch.Tensor`` subclass which flows through the animation timeline, so
    decoding there would change what ``mob.color`` reads back and would make
    colour tweens interpolate in linear light. three.js does decode at its
    ``Color``; Algan does not, and a red-to-blue tween staying perceptually
    even is the reason.
    """
    if not rt_settings.LINEAR_COLOR_SPACE:
        return
    for key in _MERGED_COLOR_KEYS:
        arr = scene.get(key)
        if arr is None or arr.shape[-1] < 3:
            continue
        arr[..., :3] = srgb_to_linear(arr[..., :3])


def prewarm_merge_cache(primitives):
    """Build (and cache) a batch's merged scene + STBVHs ahead of the render.

    Idempotent -- ``_merge_scene`` caches on ``primitives[0]`` -- and a no-op
    for non-ray-traced primitives. Called from the batch-prep *worker* thread
    (see ``RenderLoopMixin.get_frames``): the merge + STBVH builds are
    torch-only (no Taichi) and read only the ``_rt_*`` arrays packed by the
    same prep task, so running them on the worker while the previous batch
    renders hides seconds of otherwise-serial main-thread time per render
    (~6.5s of STBVH builds alone on the UHD bezier benchmark). The render
    thread's own ``_merge_scene`` then returns the cache instantly.
    """
    if not primitives:
        return
    if not isinstance(
        primitives[0], (RayTracedTrianglePrimitive, RayTracedBezierCircuitPrimitive)
    ):
        return
    _merge_scene(primitives)


def _pack_lights(light_sources, num_frames, device):
    """Per-frame packed light rows for the deterministic tracer's fragment
    lighting: positions ``[T, L, 3]`` and color rows ``[T, L, C]``.

    ``C == 3`` (the legacy compact packing: RGB radiance only) whenever every
    light is a plain point light -- keeping such scenes on the kernels'
    original point-light arithmetic. Any *extended* light (a non-point type,
    or falloff / soft-shadow parameters; see :mod:`algan.rendering.lights`)
    widens every row to ``C == 16``::

        0:3  RGB radiance (intensity premultiplied)   9  cos outer (spot)
        3    light type id                            10 cos inner (spot)
        4    decay exponent                           11 shadow softness
        5    range (0 = infinite)                     12:15 ground RGB / SH
        6:9  direction                                15 power fraction (1/K)

    Area lights arrive pre-expanded into K emitter sample rows (see
    ``Scene._materialize_render_state``), each occupying its own light slot.
    """
    any_ext = any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    )
    if not any_ext:
        positions, colors = [], []
        for light in light_sources or ():
            origin = light.origin.detach().to(device)
            color = light.light_color.detach().to(device)
            positions.append(_expand_frames(_flat_frames(origin, (3,)), num_frames))
            col = color.reshape(color.shape[0], -1)
            colors.append(_expand_frames(col[:, :3].float(), num_frames))
        if not positions:
            return (
                torch.zeros((1, 1, 3), device=device),
                torch.zeros((1, 1, 3), device=device),
                0,
            )
        light_pos = torch.stack(positions, 1).to(device).contiguous()
        light_col = torch.stack(colors, 1).to(device).contiguous()
        return light_pos, light_col, light_pos.shape[1]

    positions, rows = [], []
    for light in light_sources or ():
        pos = light.origin.detach().to(device)  # [T, K, 3]
        col = light.light_color.detach().to(device)  # [T, K, >=3]
        aux = getattr(light, "_render_aux", None)  # [T, K, 13] or None
        if aux is not None:
            aux = aux.detach().to(device)
        num_samples = pos.shape[-2]
        pos = pos.reshape(pos.shape[0], num_samples, -1)[..., :3].float()
        col = col.reshape(col.shape[0], col.shape[-2], -1)[..., :3].float()
        for k in range(num_samples):
            positions.append(_expand_frames(pos[:, k], num_frames))
            c = _expand_frames(col[:, min(k, col.shape[1] - 1)], num_frames)
            if aux is None:
                # Plain point light sharing a pack with extended lights:
                # type 0 with a whole-light power fraction (col 12 -> packed
                # col 15).
                a = torch.zeros((c.shape[0], 13), dtype=torch.float32, device=device)
                a[:, 12] = 1.0
            else:
                a = _expand_frames(aux[:, k].float(), num_frames)
            rows.append(torch.cat((c, a), -1))
    light_pos = torch.stack(positions, 1).to(device).contiguous()
    light_col = torch.stack(rows, 1).to(device).contiguous()
    return light_pos, light_col, light_pos.shape[1]


def _prefill_deferred_background(out, background, frame_offset):
    """Evaluate a callback directly into its target-device output.

    The render arena already owns ``out``. Python callback frames are quantized
    and copied one at a time, bounding their temporary memory to one frame. A
    Taichi callback instead fills the complete batch in one kernel launch and
    has no callback result tensor to retain.
    """
    requested_device = background.device
    device = out.device
    if requested_device.type != device.type or (
        requested_device.index is not None
        and device.index is not None
        and requested_device.index != device.index
    ):
        raise RuntimeError("deferred background and render output devices differ")

    width = background.width
    height = background.height
    output_pixels = out.shape[1]
    full_pixels = width * height
    aa = background.anti_alias_level
    base_pixels = full_pixels // (aa * aa)

    if output_pixels not in (full_pixels, base_pixels):
        raise RuntimeError(
            "deferred background resolution does not match render output"
        )

    if background.is_taichi_func:
        from algan.rendering.raytracing.background_taichi import (
            fill_background_from_func,
        )

        # A Taichi func is inlined into this one batch-wide kernel. It writes
        # directly into the arena-backed output, so no per-frame loop or
        # supersampled intermediate allocation is needed.
        fill_background_from_func(
            out,
            background.callback,
            background.width,
            background.height,
            background.anti_alias_level,
            background.first_frame,
            frame_offset,
            background.frames_per_second,
        )
        return

    x = (torch.arange(width, device=device, dtype=torch.float32) / width).view(1, -1, 1)
    y = (torch.arange(height, device=device, dtype=torch.float32) / height).view(
        -1, 1, 1
    )

    k_ = 1  # out.shape[0]
    for local_frame in range(0, out.shape[0], k_):
        k = min(k_, out.shape[0] - local_frame)
        time = (
            torch.arange(
                background.first_frame + frame_offset + local_frame,
                background.first_frame + frame_offset + local_frame + k,
                device=device,
                dtype=torch.float32,
            ).view(k, 1, 1, 1)
            / background.frames_per_second
        )
        frame = background.callback(x, y, time)
        if not torch.is_tensor(frame):
            frame = torch.as_tensor(frame, device=device)
        frame = frame.to(device)

        if frame.dim() <= 1:
            values = (frame.float().flatten()[:5] * 255).round_().clamp_(0, 255)
            channels = min(values.shape[0], out.shape[-1])
            out[local_frame : local_frame + k, :, :channels].copy_(values[:channels])
            if out.shape[-1] > channels:
                out[local_frame : local_frame + k, :, channels:].copy_(values[-1])
            del frame, values, time
            continue

        channels = frame.shape[-1]
        rows = frame.reshape(frame.shape[0], -1, channels)
        if rows.shape[1] != (full_pixels):
            raise RuntimeError(
                "callable background must produce one value per supersampled "
                "pixel or a resolution-free color"
            )
        rows = torch.add(0.5, rows, alpha=255).clamp_(0, 255).to(torch.uint8)

        if output_pixels == base_pixels and aa > 1:
            image = rows.view(k, height, width, channels).float().permute(0, 3, 1, 2)
            rows = F.avg_pool2d(image, aa).permute(0, 2, 3, 1).reshape(-1, channels)
            rows = (rows + 0.5).clamp_(0, 255).to(torch.uint8)

        copied_channels = min(rows.shape[-1], out.shape[-1])
        out[local_frame : local_frame + k, :, :copied_channels].copy_(
            rows[..., :copied_channels]
        )
        if out.shape[-1] > copied_channels:
            out[local_frame : local_frame + k, :, copied_channels:].copy_(
                rows[..., -1:]
            )
        del frame, rows, time
        if output_pixels == base_pixels and aa > 1:
            del image

    del x, y
    # Return callback intermediates held by PyTorch's caching allocator to the
    # device before Taichi begins using the arena-backed frame.
    empty_cache(force_gc=False)


def _prefill_background(
    out, background_color, frame_offset, device, background_frames=None
):
    """Fill the output buffer with the background. Solid colors arrive as a
    float [channels] tensor in [0, 1]; animated/image backgrounds arrive as a
    uint8 row tensor [1 + frames * pixels, channels] (leading padding row).

    ``background_frames`` is how many frames that row tensor covers. Callers
    that know it should pass it: it is the only way to tell an image
    background's own resolution apart from the output's, and a mismatch there
    scrolls a different slice of the background into every frame rather than
    failing (the deferred path already raises for the same reason).
    """
    if isinstance(background_color, _DeferredBackground):
        _prefill_deferred_background(out, background_color, frame_offset)
        return

    num_frames, num_pixels, C_out = out.shape
    # Keep the background on its source device. ``Tensor.to(device)`` used to
    # materialize a full, untracked peer tensor on the rendering device before
    # writing the arena-backed output. ``copy_`` below performs device and dtype
    # conversion directly into the reserved destination instead.
    bg = background_color
    linear = rt_settings.LINEAR_COLOR_SPACE
    if bg.dim() <= 1 or bg.shape[0] == 1:  # solid color (in [0, 1] floats)
        vals = bg.float().flatten()[:5]
        if linear:
            # The background is the second colour ingest, and it composites
            # against linear geometry (``rs_acc * 255 + weight * bg``), so it
            # has to be linear too. Only 0:3 -- channel 3 is glow and the last
            # is alpha. Not rounded to integers under the linear space: this
            # buffer is float32 here (the linear space requires the float HDR
            # buffer, see the guard in tracer.py) and rounding a linear value
            # to a byte grid would crush the darks, which is exactly why 8-bit
            # buffers hold *encoded* values in the first place.
            vals = torch.cat((srgb_to_linear(vals[:3]), vals[3:]), 0) * 255
        else:
            vals = (vals * 255).round_().clamp_(0, 255)
        k = min(vals.shape[0], C_out)
        out[..., :k].copy_(vals[:k])
        if C_out > k:
            # Alpha (and any missing channel) defaults to the background's
            # last channel, matching opaque-by-default behavior.
            out[..., k:].copy_(vals[-1])
    else:
        rows = bg.reshape(-1, bg.shape[-1])[1:]
        if background_frames:
            source_pixels = rows.shape[0] // int(background_frames)
            if source_pixels != num_pixels:
                raise RuntimeError(
                    "background resolution does not match render output "
                    f"({source_pixels} pixels per frame vs {num_pixels}); "
                    "a super-sampled background must be averaged down with "
                    "_downsample_background first"
                )
        rows = rows[
            frame_offset * num_pixels : (frame_offset + num_frames) * num_pixels
        ]
        rows = rows.view(num_frames, num_pixels, -1)
        k = min(rows.shape[-1], C_out)
        if linear:
            # An image background is 8-bit sRGB like any other texture, so it
            # decodes the same way the solid colour above does. Done in float
            # at 0-255 scale to match what the composite expects.
            head = rows[..., : min(3, k)].float() * (1.0 / 255.0)
            out[..., : min(3, k)].copy_(srgb_to_linear(head) * 255.0)
            if k > 3:
                out[..., 3:k].copy_(rows[..., 3:k])
        else:
            out[..., :k].copy_(rows[..., :k])
        if C_out > k:
            out[..., k:].copy_(rows[..., -1:])


def _downsample_background(
    background_color, aa, num_frames, screen_height, screen_width
):
    """Average a super-sampled animated/image background down to the output
    resolution (box filter, matching ``post_process_frames``), so the in-place
    anti-aliased renderer -- which samples the background once per output pixel
    -- gets a background at the right resolution.

    Solid colors (resolution-free) and backgrounds that are not super-sampled
    (row count not ``num_frames * (screen_height*aa) * (screen_width*aa)``) are
    returned unchanged.
    """
    bg = background_color
    if not torch.is_tensor(bg) or bg.dim() <= 1 or bg.shape[0] == 1:
        return bg  # solid color
    # This is preparation for an arena-backed copy; do the resampling on the
    # host even if a direct caller supplied a render-device background.
    # bg = bg.detach().cpu()
    C = bg.shape[-1]
    body = bg.reshape(-1, C)[1:]  # drop the leading padding row
    h_aa, w_aa = screen_height * aa, screen_width * aa
    if body.shape[0] != num_frames * h_aa * w_aa:
        return bg  # not a super-sampled image background; leave as-is
    img = body.view(num_frames, h_aa, w_aa, C).float().permute(0, 3, 1, 2)
    ds = F.avg_pool2d(img, aa).permute(0, 2, 3, 1).reshape(-1, C)
    ds = (ds + 0.5).clamp_(0, 255).to(bg.dtype)
    return torch.cat((ds[:1], ds), 0)
