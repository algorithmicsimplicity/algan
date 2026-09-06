"""Collection of helper functions used to combine collections of primitives
into contiguous tensor data-structures, ready to be shipped to ray tracing kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.environment import env_int
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.bezier_acceleration import (
    build_bezier_edge_acceleration,
)
from algan.rendering.raytracing.primitives import (
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _EXTRA_W,
    _M_REFLECTIVITY,
    _M_TRANSMISSION,
    _M_WIDTH,
)
from algan.rendering.raytracing.refit_bvh import build_refit_bvh
from algan.rendering.raytracing.settings import (
    _MAT_SLOTS,
    _USER_PIPELINE_BASE,
    _constant_promotion_active,
)
from algan.rendering.raytracing.shading_taichi import MAT_W
from algan.rendering.raytracing.sliver_split import sliver_leaf_columns
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO, STBVH, build_stbvh
from algan.rendering.raytracing.utils import (
    _cat_collections,
    _cat_mat_blocks,
    _expand_frames,
    _flat_frames,
)
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.taichi_compat import is_compiler_func
from algan.utils.color_space import srgb_to_linear
from algan.utils.memory_utils import (
    InsufficientMemoryException,
    begin_cuda_peak,
    end_cuda_peak,
    release_torch_memory,
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
        self.is_taichi_func = is_compiler_func(callback)


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
    ``settings.merge_gpu_peak_factor`` and ``RenderLoopMixin``).
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
    # The default-material seed is a vertex-shader input like any other. A
    # primitive built by the no-material fallback carries the process default
    # material's parameter values (SETTINGS.style.default_material), and the
    # tensor-valued ones -- DiffuseMaterial's ``emissive`` is a Color, built
    # wherever the material was -- reach ``lambert_shader`` beside ``corners``
    # and ``colors``. Left behind, they are the one CPU operand in an
    # otherwise-uploaded expression, which is invisible on a CPU render and a
    # device-mismatch RuntimeError on a CUDA one.
    seed = getattr(primitive, "default_material_params", None)
    if seed:
        primitive.default_material_params = {
            name: value.to(device)
            if torch.is_tensor(value) and value.device != device
            else value
            for name, value in seed.items()
        }
    if not hasattr(primitive, "shader_param_values"):
        return
    for i in range(len(primitive.shader_param_values)):
        value = primitive.shader_param_values[i]
        if value.device != device:
            primitive.shader_param_values[i] = value.to(device)


def gpu_project_input_bytes(primitives):
    """Total bytes of a batch's pre-projection source geometry.

    Feeds the projection's transient-peak estimate used by the render-arena
    preflight (see ``settings.project_gpu_peak_factor`` and
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


#: Byte boundary the Apple GPU's zero-copy path needs an uploaded storage to
#: start on, or 1 for "whatever the element needs" -- which is what every other
#: backend wants and what this used to be unconditionally.
#:
#: A kernel there binds torch's own ``MTLBuffer`` at the storage's byte offset,
#: and a **vector-element** array is loaded as one vector: the BVH's sibling
#: blocks are ``ndarray(dtype=vector(4, f16))``, an 8-byte element. Aligning to
#: the scalar element (2 bytes for f16) is not enough for that, and it is not
#: hypothetical -- the triangle BVH's blocks landed at byte 799066188, a
#: multiple of 4 and not of 8, so ``mps_zero_copy`` refused to import them and
#: they went back on Taichi's host-staging path: four copies and a stream sync
#: per launch, on the widest array the tracer reads.
#: (``DESIGN_mps_zero_copy.md`` §3.3 asks for this.)
#:
#: 16 covers every vector element Metal binds as one (up to 4 x f32). A wider
#: one -- ``ALGAN_BVH_ARITY=8`` with f32 blocks would be 32 bytes -- is not
#: silently mis-bound: the import checks the offset against the element size
#: itself, declines, and the bus report names it.
#:
#: Applied HERE rather than in ``ManualMemory``, which was the first attempt and
#: was too broad by half: a floor on every arena slice changes every arena's
#: layout and size, which the memory model derives chunk lengths from, and it
#: broke six tests that assert exact byte accounting. This is the only path that
#: places a vector-element array, and its two halves -- the copy and
#: ``get_merged_scene_arena_nbytes``, which predicts the cost -- share this
#: function, so the accounting stays exact.
#:
#: Resolved once: whether the conversion is installed is decided while ``algan``
#: is still importing, long before a scene is uploaded.
_UPLOAD_ALIGNMENT_FLOOR = None


def _upload_alignment_floor():
    global _UPLOAD_ALIGNMENT_FLOOR
    if _UPLOAD_ALIGNMENT_FLOOR is None:
        _UPLOAD_ALIGNMENT_FLOOR = 1
        try:
            from algan.rendering.mps_zero_copy import installed

            if installed():
                _UPLOAD_ALIGNMENT_FLOOR = 16
        except Exception:
            _UPLOAD_ALIGNMENT_FLOOR = 1
    return _UPLOAD_ALIGNMENT_FLOOR


def _storage_alignment(group):
    element = max((tensor.element_size() for tensor in group["tensors"]), default=1)
    return max(element, _upload_alignment_floor())


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
            # The START of the block is what a kernel binds, and the reverse
            # arena grows downward, so subtract first and align that. While the
            # alignment was the element size this was the same thing -- a
            # storage is always a whole number of elements, so aligning the end
            # aligned the start too -- and it stops being the same as soon as
            # `_upload_alignment_floor` exceeds the element.
            pointer -= nbytes
            pointer -= pointer % alignment
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
    # Reverse allocations grow downward, so the block's START is
    # ``pointer - nbytes`` and that is what has to land on the boundary (see
    # get_merged_scene_arena_nbytes, which predicts the same padding).
    padding = (pointer - nbytes) % alignment if persist else (-pointer) % alignment
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
    length 1, so a temporally-constant map/color is stored once instead of T
    times. The kernels index the time axis as ``f % shape[0]``, so a length-1
    axis is read by every frame.

    The ``bool()`` is a device sync, and the merge runs on the prefetch
    worker while the previous batch renders -- every sync here waits out the
    whole queued render. Prefer :func:`_dedup_time_group` for anything called
    per merge; this stays for one-off probes outside the merge's hot path.
    """
    if x.shape[0] > 1 and bool((x == x[:1]).all()):
        return x[:1].contiguous()
    return x


def _dedup_time_group(scene, keys):
    """Collapse every temporally-constant ``scene[key]`` to one frame with a
    SINGLE device sync for the whole group.

    Each probe's reduction stays on the device; one stacked host transfer
    answers all of them. Measured on the nn UHD benchmark: per-table syncs
    inside the merge cost +5.3 s of a 24 s render (the merge overlaps the
    render, so each sync drains the full queued chunk), while the collapse
    itself is milliseconds.
    """
    probes = [k for k in keys if scene[k].shape[0] > 1]
    if not probes:
        return
    flags = torch.stack([(scene[k] == scene[k][:1]).all() for k in probes]).cpu()
    for k, flag in zip(probes, flags.tolist()):
        if flag:
            scene[k] = scene[k][:1].contiguous()


#: Texture-meta row width: cols 0-2 color map (offset, w, h), 3-5 material
#: map, 6-8 normal map, 9 the texture-driven-property bitmask, 10-12 the
#: per-map TIME lengths (see ``_append_texture``; 1 when the buffer's own
#: time axis carries the frames). Cols 13-14 are the color map's opacity
#: region (base row in the bank / frame count; -1 = premultiplied on the host,
#: texture_opacity_in_kernel), col 15 its u8-storage LUT base row (-1 =
#: plain f32 rows, -2 = u8-packed WITHOUT a LUT -- an endpoint stack decodes
#: bytes as authored k/255 directly; texture_u8_storage) and cols 16-17 its
#: endpoint-interpolation region (base row / frame count; -1 = the map's
#: leading axis is time, texture_time_lerp) -- new capabilities travel as
#: DATA in this table because the resolve kernel sits at Taichi's
#: runtime-argument ceiling and cannot take new arrays.
_TEX_META_W = 18


def _tex_meta_placeholder(device):
    """One all-absent tex-meta row (offsets -1, time lengths 1)."""
    meta = torch.full((1, _TEX_META_W), -1, dtype=torch.int32, device=device)
    meta[:, 10:13] = 1
    meta[:, 14] = 1
    meta[:, 17] = 1
    return meta


#: Texel count below which texture content dedup is not attempted (see
#: ``_append_texture``): each candidate match is a synchronizing
#: ``torch.equal``, worth it for a shared image, not for a promoted 1x1 map.
#: Output-neutral (dedup only ever shares identical texels), so this trades
#: merge-time syncs against texture memory and is exposed for that.
content_dedup_min_texels = max(0, env_int("ALGAN_CONTENT_DEDUP_MIN_TEXELS", 4096))


def _split_promotable(p, _append_texture, device, scene):
    """Partition a non-textured triangle primitive into the triangles that must
    stay per-vertex and the triangles whose color + material are constant
    across their three corners and every frame (and are non-glowing). The
    constant triangles are grouped by value -- so a uniform mob is one group even
    when it was batched into a primitive alongside differently-colored mobs --
    and each group is promoted to one shared 1x1 color map + 1x1 material map
    (appended here to the shared texel buffer).

    Returns ``(keep_idx, promo_idx, promo_meta)``: ascending ``keep_idx`` selects
    the per-vertex triangles; ``promo_idx`` selects the promoted triangles
    grouped by value; ``promo_meta`` is the ``[len(promo_idx), 10]`` tex-meta
    (color map cols 0-2, material map 3-5, no normal map 6-8 = -1, bitmask 9 =
    refl|rough|ior) aligned to ``promo_idx``. The kernel reads all three material
    properties from the material map, so promoted triangles need no per-vertex
    ``tri_colors``/``tri_extra`` row.
    """
    colors = p._rt_tri_colors  # [Tc, N, 3, 5]
    extra = p._rt_tri_extra  # [Te, N, _EXTRA_W] (see _pack_surface_extra)
    N = colors.shape[1]
    all_idx = torch.arange(N, device=device)
    if N == 0:
        empty = torch.zeros((0, _TEX_META_W), dtype=torch.int32, device=device)
        return all_idx, all_idx, empty
    # Per-triangle promotable: the three corners share one color (all channels,
    # all frames) and one material (reflectivity 0/2/4, roughness 1/3/5, index of
    # refraction 6/7/8), and the triangle is non-glowing (glow magnitude cols
    # 9-11 -- which hold per-corner transmission since the transmission work;
    # the name survives from when those columns were glow). Only such a triangle
    # is fully described by a single 1x1 texel.
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
        empty = torch.zeros((0, _TEX_META_W), dtype=torch.int32, device=device)
        return keep_idx, promo_all, empty

    # Group promoted triangles by their (per-frame) constant color + material
    # value, so identical mobs share one pair of maps. The key is the corner-0
    # color [T,5] plus material (refl, rough, ior) [T,3] over all frames.
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

    # One color + material map per distinct value; each promoted triangle's meta
    # row points at its group's maps.
    group_meta = []
    for gid in range(uniq.shape[0]):
        rep = int(promo_all[int((inv == gid).nonzero()[0])])
        cmap = _dedup_time(colors[:, rep : rep + 1, 0, :].contiguous())  # [T',1,5]
        color_meta = _append_texture(
            cmap.reshape(cmap.shape[0], 1, 1, 5).float().contiguous(), is_color=True
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
        group_meta.append(
            [
                *color_meta[:3],
                *material_meta[:3],
                -1,
                0,
                0,
                1 | 2 | 4 | 8,
                color_meta[3],
                material_meta[3],
                1,
                # A promoted map is sliced out of per-vertex colors whose
                # coverage already carries the mob opacity, so it takes no
                # opacity region, stays f32 (it is 1x1) and its leading axis
                # is time (no endpoint interpolation).
                -1,
                1,
                -1,
                -1,
                1,
            ]
        )
    group_meta = torch.tensor(group_meta, dtype=torch.int32, device=device)
    promo_meta = group_meta[inv_sorted]  # [P, _TEX_META_W]
    return keep_idx, promo_idx, promo_meta


def _build_accel(
    lo,
    hi,
    num_frames,
    tightness,
    opaque=None,
    builder="morton",
    refit=None,
    casts=None,
    leaf_prim=None,
):
    """Build one geometry type's acceleration structure: the classic
    spatio-temporal instance tree, or -- under ``settings.bvh_refit`` -- the
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
        return build_refit_bvh(
            lo,
            hi,
            num_frames=num_frames,
            opaque=opaque,
            casts=casts,
            leaf_prim=leaf_prim,
        )
    if leaf_prim is not None:
        raise ValueError("only the refit BVH takes a leaf -> primitive map")
    return build_stbvh(
        lo,
        hi,
        num_frames=num_frames,
        tightness=tightness,
        opaque=opaque,
        builder=builder,
        casts=casts,
    )


def _tri_leaf_columns(tri_pos, lo, hi, opaque, casts, refit):
    """The triangle tree's leaf columns: the per-primitive boxes as they are,
    or -- under the refit tree, which is the only one whose leaves carry an
    explicit primitive id -- with every sliver cut into per-strip leaves
    (``sliver_split``). Returns ``(lo, hi, opaque, casts, leaf_prim)``.
    """
    if refit and tri_pos is not None:
        cols = sliver_leaf_columns(tri_pos, lo, hi, opaque, casts)
        if cols is not None:
            return cols
    return lo, hi, opaque, casts, None


def _empty_scene_part(device, refit=None):
    """Placeholder BVH + arrays for an absent geometry type (same tree kind
    as the batch's real trees, so one compile-time flag covers all four).
    """
    lo = torch.full((1, 1, 3), EMPTY_LO, device=device)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=device)
    return _build_accel(lo, hi, num_frames=1, tightness=2.0, refit=refit)


def _build_opaque_bvh(
    lo, hi, opaque, num_frames, tightness, builder="morton", refit=None, casts=None
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
        casts=casts,
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
    if not _rts.bvh_defer:
        return False
    # The sheet route is what resolves primaries without a tree, so every
    # switch that vetoes it outright (``tracer.analytic_raster_route_active``)
    # sends the batch to the classic wavefront, which traverses for every
    # primary ray. All of them are readable here, so a settings-driven
    # fallback -- ``analytic_aa=False``, the reference arm for judging a
    # sheet-resolve change, most of all -- is planned for rather than
    # discovered mid-render: its trees are built on the prefetch worker with
    # the rest of the merge and uploaded with it, instead of stalling the
    # render and being re-homed into the arena afterwards. The route's
    # remaining vetoes are scene- or camera-dependent and stay runtime-only.
    if not (
        _rts.hybrid_raster
        and _rts.analytic_aa
        and _rts.sheet_resolve
        and _rts.analytic_aa_run
        and _rts.raster_sparse_coverage
        and _rts.raster_empty_skip
        and _rts.raster_covered_shade
    ):
        return False
    if int(_rts.samples_per_pixel) > 1 or _rts.shadows or _rts.inplace_aa:
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
    # wf_opaque_closest / wf_opaque_prepass rollouts (both default OFF): the
    # tracer's opaque_closest/opaque_prepass templates compile every read out
    # otherwise. With neither live (read at merge time), alias the main tree
    # instead of building a second one -- ~40% of the per-batch BVH build.
    # ``opaque_bvh_skipped`` lets the tracer keep those features off for a
    # batch merged without real opaque trees if a toggle flips mid-render.
    opq_live = (
        not SETTINGS.raytracing.opaque_bvh_skip_dead
        or SETTINGS.raytracing.wf_opaque_closest
        or SETTINGS.raytracing.wf_opaque_prepass
    )
    scene["opaque_bvh_skipped"] = not opq_live
    if _bvh_deferral_eligible(scene) and (
        tri_inputs is not None or bez_inputs is not None
    ):
        _rts = SETTINGS.raytracing
        placeholder = _empty_scene_part(device)
        if tri_inputs is not None:
            lo, hi, opaque, casts = tri_inputs
            # The caster mask is retained beside the bounds: the on-demand
            # build needs it to stamp the leaf words (``build_deferred_bvhs``).
            scene["tri_frame_casts"] = casts.contiguous()
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
        lo, hi, opaque, casts = tri_inputs
        # Median-split ordering: ~25% faster traversal than Morton at ~0.2s
        # extra build per batch; byte-identical for triangles (the depth-peel
        # is arrangement-invariant). PN/bezier BVHs stay Morton -- their
        # seam de-dup is discovery-order sensitive (see stbvh.bvh_build).
        leaf_lo, leaf_hi, leaf_opq, leaf_casts, leaf_prim = _tri_leaf_columns(
            scene.get("tri_pos"),
            lo,
            hi,
            opaque,
            casts,
            SETTINGS.raytracing.refit_bvh_active(),
        )
        scene["tri_bvh"] = _build_accel(
            leaf_lo,
            leaf_hi,
            num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=leaf_opq,
            casts=leaf_casts,
            builder="split",
            leaf_prim=leaf_prim,
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
                casts=casts,
            )
    if bez_inputs is not None:
        lo, hi, opaque, casts = bez_inputs
        # ss3.4: bezier was the last type still pinned to Morton (PN, the other,
        # was deleted). Split ordering is a pure reorder, but a circuit's seam
        # de-dup is discovery-order sensitive, so it moves output at the epsilon
        # level -- hence the gate rather than a straight flip.
        bez_builder = "split" if SETTINGS.raytracing.bez_bvh_split else "morton"
        scene["bez_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque,
            casts=casts,
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
                casts=casts,
            )


#: The merged-scene keys ``build_deferred_bvhs`` writes. Re-homed into the
#: arena as one group so an opaque tree aliased to its main tree stays aliased.
_DEFERRED_BVH_KEYS = ("tri_bvh", "tri_opaque_bvh", "bez_bvh", "bez_opaque_bvh")


def rehome_deferred_bvhs_to_arena(merged, memory):
    """Move on-demand-built STBVHs into the arena the rest of the scene lives in.

    The merged scene is uploaded as one ManualMemory allocation per dtype
    (:func:`copy_merged_scene_to_arena`) and the widest kernels bind their
    scene-indexed tables as offsets into that single buffer -- every array so
    bound has to be a view of it (`arena_args_taichi`). A tree built by
    :func:`build_deferred_bvhs` is an ordinary torch allocation made long after
    that upload, so leaving it where the builder put it makes every one of
    those launches raise ``ArenaBindingError``: ``t_leaf_prim`` in a different
    allocation from the ``edge_accel`` bound beside it.

    Copied at the arena's persistent (reverse) end, because these trees live
    for the whole batch while this runs from inside a chunk. The caller
    publishes the pointer reached so the per-chunk rewind and the render loop's
    between-chunk restore hold the arena open exactly that far -- the same
    treatment the batch-wide raster tables get (``tracer.rewind_to``).

    Idempotent: a group already backed by ``memory`` is skipped, so a second
    call after an out-of-memory retry costs nothing and copies nothing.
    """
    group = {
        key: merged[key]
        for key in _DEFERRED_BVH_KEYS
        if isinstance(merged.get(key), STBVH)
    }
    if not group:
        return
    merged.update(_copy_merged_scene_to_arena(group, memory, persist=True))


def build_deferred_bvhs(merged, memory=None):
    """Build the STBVHs a deferred batch skipped (see ``_finalize_bvhs``).

    Called by the tracer the moment anything actually needs a tree: shadows,
    classic-wavefront routing, an actually spawned continuation ray, or the
    Monte Carlo path. Idempotent, and forces the tree kind recorded at merge
    time so the batch's placeholder and real trees always agree on the
    ``refit`` kernel template.

    ``memory`` is the arena the merged scene was uploaded into; pass it so the
    freshly built trees are re-homed there rather than left in the ordinary
    torch allocations the builder returns
    (:func:`rehome_deferred_bvhs_to_arena` says why that matters).
    """
    if not merged.get("bvh_deferred"):
        # An out-of-memory retry re-enters here after the build itself
        # succeeded and cleared the flag: the re-home may still be owed, and
        # costs nothing once the trees are arena-backed.
        if memory is not None:
            rehome_deferred_bvhs_to_arena(merged, memory)
        return
    refit = bool(merged.get("bvh_deferred_refit"))
    num_frames = int(merged["num_frames"])
    # Same opaque-tree skip as _finalize_bvhs (read live at this build).
    opq_live = (
        not SETTINGS.raytracing.opaque_bvh_skip_dead
        or SETTINGS.raytracing.wf_opaque_closest
        or SETTINGS.raytracing.wf_opaque_prepass
    )
    merged["opaque_bvh_skipped"] = not opq_live
    if merged.get("num_triangles", 0) > 0 and merged.get("tri_frame_lo") is not None:
        lo = merged["tri_frame_lo"]
        hi = merged["tri_frame_hi"]
        opaque = merged["tri_frame_opaque"]
        casts = merged.get("tri_frame_casts")
        leaf_lo, leaf_hi, leaf_opq, leaf_casts, leaf_prim = _tri_leaf_columns(
            merged.get("tri_pos"), lo, hi, opaque, casts, refit
        )
        merged["tri_bvh"] = _build_accel(
            leaf_lo,
            leaf_hi,
            num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=leaf_opq,
            builder="split",
            refit=refit,
            casts=leaf_casts,
            leaf_prim=leaf_prim,
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
                casts=casts,
            )
        merged["tri_frame_lo"] = None
        merged["tri_frame_hi"] = None
    if merged.get("num_circuits", 0) > 0:
        lo = merged["bez_frame_lo"]
        hi = merged["bez_frame_hi"]
        opaque = merged["bez_frame_opaque"]
        casts = merged.get("bez_frame_casts")
        merged["bez_bvh"] = _build_accel(
            lo,
            hi,
            num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque,
            refit=refit,
            casts=casts,
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
                casts=casts,
            )
    merged["bvh_deferred"] = False
    if memory is not None:
        rehome_deferred_bvhs_to_arena(merged, memory)


def _build_mem_trim(scene, lo, hi, opaque, num_frames, device):
    """Build the 'Family A+B' memory-trim triangle arrays (see
    settings.wf_mem_trim). Reorders prims into material-class bands -- band 0
    ``needs_mat`` (lit), band 1 ``needs_norm`` only (reflective / normal-mapped /
    promoted), band 2 bare (unlit matte) -- so that ``tri_norm`` and ``tri_mat``
    become compacted PREFIXES (needs_mat subset needs_norm, so both nest under a
    single permutation). ``tri_colors``/``tri_extra`` stay in their original
    (promotion-compacted) order, addressed by a per-prim remap ``col_row`` (-1 =
    promoted, color/material from its 1x1 maps); ``tex_meta``/``uvs`` are widened
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
    Nc = tri_extra.shape[1]  # prims with a per-vertex color/extra row

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

    tex_meta_t = torch.zeros((N, _TEX_META_W), dtype=torch.int32, device=device)
    tex_meta_t[:, 0] = -1
    tex_meta_t[:, 3] = -1
    tex_meta_t[:, 6] = -1
    tex_meta_t[:, 10:13] = 1
    # No-map rows: no opacity region (col 13 = -1, col 14 = 1 frame), f32
    # storage (col 15 = -1) and no endpoint interpolation (col 16 = -1,
    # col 17 = 1); real rows are index_selected below and carry their own
    # values.
    tex_meta_t[:, 13] = -1
    tex_meta_t[:, 14] = 1
    tex_meta_t[:, 15] = -1
    tex_meta_t[:, 16] = -1
    tex_meta_t[:, 17] = 1
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


def _densify_frag_pipeline_ids(scene):
    """Renumber this batch's user fragment-pipeline ids into a dense range.

    A pipeline's id is its position in a process-global, append-only registry
    (``fragment_shaders.register_pipeline``), and the shade kernel's injected
    ``frag_pipelines`` tuple is indexed by that id. Taichi specialises on the
    tuple, so with global ids the SAME custom shader compiles a different
    kernel depending on how many other pipelines the process registered before
    it: in one process a lone pipeline is ``(fn,)``, in the next it is
    ``(None, None, fn)``. Nothing reuses a variant across those, in the
    process or in the offline cache, and a test suite pays a cold compile per
    custom-shader render.

    So the batch renumbers: the user ids it actually carries, in ascending
    global order, become ``_USER_PIPELINE_BASE + 0, +1, ...``. The injected
    tuple's shape is then a function of the batch's own content -- one
    pipeline is always ``(fn,)`` -- while ``scene["frag_pipeline_ids"]`` keeps
    the global ids, in slot order, so the tracer can still find the funcs.

    Built-in ids (below ``_USER_PIPELINE_BASE``) and the negative sentinels
    are left exactly as they are; only user pipelines move. Sets
    ``scene["tri_material_ids"]`` as a side effect, since it has the unique
    ids in hand and the merge would otherwise recompute them.
    """
    mat_id = scene["tri_mat_id"]
    present = tuple(int(v) for v in torch.unique(mat_id.detach().cpu()).tolist())
    user = tuple(v for v in present if v >= _USER_PIPELINE_BASE)
    scene["frag_pipeline_ids"] = user
    dense = tuple(range(_USER_PIPELINE_BASE, _USER_PIPELINE_BASE + len(user)))
    if user and user != dense:
        # One gather over a [0, max] lookup table; the ``>=`` guard keeps
        # built-ins and sentinels off it rather than relying on the table's
        # identity prefix, so a negative id can never index it.
        table = torch.arange(user[-1] + 1, dtype=mat_id.dtype, device=mat_id.device)
        for old, new in zip(user, dense):
            table[old] = new
        scene["tri_mat_id"] = torch.where(
            mat_id >= _USER_PIPELINE_BASE,
            table.index_select(0, mat_id.clamp_min(0).reshape(-1).long()).view_as(
                mat_id
            ),
            mat_id,
        ).contiguous()
        renumbered = dict(zip(user, dense))
        present = tuple(sorted(renumbered.get(v, v) for v in present))
    scene["tri_material_ids"] = present


def _merge_scene(primitives, *, track_peak=None):
    """Merge the batch's collections into one set per geometry type --
    triangles and bezier circuits, each with a single STBVH
    over all frames -- cached for the batch.

    ``track_peak`` overrides the ``merge_track_peak`` setting for this one
    build: ``False`` skips the measurement entirely. The overlapped batch
    prep passes that, because a peak measured beside a live render counts
    the render's own allocations -- and the counter reset it would need
    fires under that render (see ``RenderLoopMixin._prepare_batch_on_worker``).
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
    # bounded by the render-arena preflight's ``merge_gpu_peak_factor``
    # estimate. ``merge_track_peak`` optionally measures the exact peak here to
    # calibrate that factor (it resets the process peak counter, so it stays
    # opt-in and off during profiling runs).
    gpu_merge = _rts.merge_on_gpu_active()
    if track_peak is None:
        track_peak = gpu_merge and _rts.merge_track_peak
    else:
        track_peak = bool(track_peak) and gpu_merge
    peak_token = None
    if gpu_merge:
        device = render_device()
        if track_peak:
            peak_token = begin_cuda_peak(device)
        _upload_primitive_inputs(primitives, device)
        release_torch_memory(force_gc=False)
    else:
        device = _projected_scene_device(primitives)
        if device.type != "cpu":
            release_torch_memory(force_gc=False)
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
    # normal). Each map is appended once, padded to 5 channels and flattened;
    # its placement is an ``(offset, w, h, t)`` quadruple recorded in the
    # consuming geometry's metadata (offset -1 = no map), keyed by
    # tri_tex_meta (cols 0-2 / 3-5 / 6-8 hold the triplets, cols 10-12 the
    # per-map time lengths). Assembled into scene["textures"] once the
    # geometry blocks below have appended. Under texture_time_flat a map's
    # frames are flattened along the texel axis (frame f starts at
    # ``offset + (f % t) * w * h``), so the assembled buffer keeps time
    # length 1 and one animated map no longer re-expands every static one to
    # the batch maximum at assembly.
    _texture_tensors = []
    _texel_offset = [0]
    # Content dedup (texture_content_dedup): processed maps already appended,
    # bucketed by shape/placement for the cheap prefilter; matching is exact
    # (torch.equal), so a reused placement reads byte-identical texels.
    _texture_index = {}

    def _append_u8_lut(rgb, q_rgb):
        """Append one u8 map's 256-entry decode LUT; returns its base row.

        PER MAP, and built by scattering the map's OWN direct decode (the
        exact tensor the f32 arm would have stored) into byte slots -- not by
        decoding ``arange(256) / 255``: torch's decode is not bit-stable
        across tensor sizes (its scalar and SIMD libm paths can disagree in
        the last ulp), so a shared vector-decoded table matches the f32 arm
        only approximately, while the scatter copies the arm's own bits.
        Every byte the sampler can fetch occurs in the map, so every read
        slot is written; col 1 is ``k / 255`` for the coverage byte, exact by
        IEEE division. The transient full-map decode this costs is the same
        pass the f32 arm runs per batch on the (collapsed, one-frame) map.
        """
        dec = srgb_to_linear(rgb) if rt_settings.linear_color_space else rgb
        rows = torch.zeros((1, 256, 5), dtype=torch.float32, device=device)
        rows[0, :, 0].scatter_(0, q_rgb.reshape(-1).long(), dec.reshape(-1))
        rows[0, :, 1] = torch.arange(256, dtype=torch.float32, device=device) / 255.0
        base = _texel_offset[0]
        _texture_tensors.append(rows)
        _texel_offset[0] += 256
        return base

    def _append_tex_opacity(op):
        """Append one color map's per-frame opacity region; ``(row, frames)``.

        The region is the mob's animated opacity as bank rows (value in col
        0), one row per frame of the batch window -- the sampler reads
        ``textures[tc, row + (f % frames), 0]`` (meta cols 13-14) and
        multiplies the sampled coverage by it (texture_opacity_in_kernel).
        ``(-1, 1)`` when the primitive premultiplied on the host. Tiny (20
        bytes x frames), so no dedup and no constancy probe: a probe is a
        device sync, and on the prefetch worker a sync waits out the whole
        queued chunk.
        """
        if op is None:
            return (-1, 1)
        vals = op.detach().reshape(-1).float().to(device)
        rows = torch.zeros((1, vals.numel(), 5), dtype=torch.float32, device=device)
        rows[0, :, 0] = vals
        off = _texel_offset[0]
        _texture_tensors.append(rows)
        _texel_offset[0] += vals.numel()
        return (off, vals.numel())

    def _append_tex_lerp(lerp):
        """Append one color map's endpoint-interpolation region; ``(row,
        frames)``.

        ``lerp`` is the primitive's ``[T, 3]`` (i0, i1, w) rows
        (texture_time_lerp); the sampler reads row ``off + (f % frames)``
        (meta cols 16-17), fetches endpoint texels i0 and i1 of the stack in
        AUTHORED space, lerps by w, and -- when col 3 of the row says so --
        decodes the lerped rgb into linear light, which is the merge-side
        decode of the dense path moved past the lerp. ``(-1, 1)`` when the
        map's leading axis is time. Tiny, like the opacity region: no dedup,
        no probes.
        """
        if lerp is None:
            return (-1, 1)
        vals = lerp.detach().reshape(-1, 3).float().to(device)
        rows = torch.zeros((1, vals.shape[0], 5), dtype=torch.float32, device=device)
        rows[0, :, :3] = vals
        rows[0, :, 3] = 1.0 if rt_settings.linear_color_space else 0.0
        off = _texel_offset[0]
        _texture_tensors.append(rows)
        _texel_offset[0] += vals.shape[0]
        return (off, vals.shape[0])

    def _append_texture(tex, is_color=False, u8_ok=False, authored_stack=False):
        """Append one texture map; returns ``(offset, w, h, t, lut_base)``.

        ``is_color`` says the map holds authored COLOUR, so it crosses the same
        render boundary ``_decode_merged_colors`` takes the merged color
        arrays across and is decoded into linear light here. A material map
        (metalness / roughness / IOR / transmission) or a normal map is not
        color and must not be touched, which is why the caller declares it
        rather than the decode guessing from the buffer.

        Decoded on the way IN rather than by ``_decode_merged_colors`` on the
        assembled ``scene["textures"]``: a promoted 1x1 color map is sliced out
        of the primitive's own per-vertex colors and may share storage with
        them (``_cat_collections`` also passes a lone collection through
        uncopied), so decoding the assembled buffer in place could decode the
        same values twice. ``srgb_to_linear`` returns a fresh tensor, which
        breaks any such aliasing. Only channels 0:2 are color -- 3 is additive
        glow and 4 is coverage.

        ``u8_ok`` (color maps only) says the AUTHORING side proved every
        texel is exactly ``k / 255`` with zero glow (``texture_u8_ok`` --
        proved once at assignment, so this function never probes texels).
        Such a map, when its window arrived collapsed to one frame, is stored
        as RGBA bytes bit-packed one texel per f32 lane of this same bank and
        decoded in-kernel through ``_ensure_u8_lut``'s table: x5 fewer bytes
        on the widest array of a textured merge, byte-identical by the LUT's
        construction (texture_u8_storage; ``lut_base`` >= 0 marks the layout
        in meta col 15). An interpolating window (t > 1) keeps f32 rows --
        its in-between texels are not ``k / 255``.

        ``authored_stack`` (color maps only) says the tensor is a
        ``[1, K, H, W, 5]`` endpoint stack (texture_time_lerp) whose texels
        must stay in AUTHORED space: the sampler lerps two endpoint texels
        and THEN decodes, matching the dense path's order (timeline lerp,
        then this function's decode). So the linear-light decode is skipped
        here, and a u8-eligible stack packs its bytes with NO LUT (meta
        marker -2): the bytes ARE the authored ``k / 255`` values, decoded
        by an IEEE division in the sampler. The stack's frames flatten along
        the texel axis exactly like time frames (endpoint k starts at
        ``offset + k * w * h``), with meta time length 1 -- the sampler
        addresses endpoints through the interpolation region, never through
        ``f % t``.
        """
        if tex is None:
            return (-1, 0, 0, 1, -1)
        if tex.device != device:
            # Maps arrive on whatever device built them -- a color map's
            # frame window materializes on the render device, a material or
            # normal map is a plain host tensor -- and the buffer they share
            # is built on the merge device.
            tex = tex.to(device)
        if tex.dim() == 3:  # [W, H, C]
            tex = tex.unsqueeze(0)  # [1, W, H, C]
        as_u8 = (
            is_color
            and u8_ok
            and SETTINGS.raytracing.texture_u8_storage
            and SETTINGS.raytracing.texture_time_flat
            and tex.shape[0] == 1
            and tex.shape[-1] == 5
        )
        if as_u8:
            w, h = tex.shape[-3], tex.shape[-2]
            # (r, g, b, a) bytes, little-endian packed into one i32 = one f32
            # lane. round() is exact: every channel is k/255 by admission.
            # For an authored stack tex[0] is the [K, H, W, 5] endpoint axis,
            # packed frame-major so endpoint k's texels start at lane k*w*h;
            # the tail pad is global, after the last endpoint.
            q = (
                torch.round(tex[0][..., (0, 1, 2, 4)] * 255.0)
                .clamp_(0.0, 255.0)
                .to(torch.uint8)
                .reshape(-1, 4)
                .contiguous()
            )
            packed = q.view(torch.int32).view(torch.float32).reshape(-1)
            tail = (-packed.numel()) % 5
            if tail:
                packed = torch.cat((packed, packed.new_zeros(tail)))
            flat = packed.reshape(1, -1, 5)
            dedup = (
                SETTINGS.raytracing.texture_content_dedup
                and w * h >= content_dedup_min_texels
            )
            if dedup:
                key = ("u8a" if authored_stack else "u8", tuple(flat.shape), w, h)
                for prior_flat, prior_meta in _texture_index.get(key, ()):
                    # Compared as i32: packed byte patterns can form float
                    # NaNs, and float torch.equal treats NaN != NaN -- which
                    # silently stored every shared image twice.
                    if torch.equal(
                        prior_flat.view(torch.int32), flat.view(torch.int32)
                    ):
                        return prior_meta
            o = _texel_offset[0]
            _texture_tensors.append(flat)
            _texel_offset[0] += flat.shape[1]
            if authored_stack:
                # The bytes ARE the authored k/255 values; the sampler
                # divides by 255 and decodes AFTER the endpoint lerp, so no
                # LUT exists. -2 marks u8 packing without one.
                lut_base = -2
            else:
                # Decode EXACTLY the tensor the f32 arm would decode (same
                # shape, same slice), so the scattered LUT carries that
                # arm's own bits.
                lut_base = _append_u8_lut(tex[..., :3].float(), q[..., :3])
            meta = (o, w, h, 1, lut_base)
            if dedup:
                _texture_index.setdefault(key, []).append((flat, meta))
            return meta
        if (
            is_color
            and not authored_stack
            and rt_settings.linear_color_space
            and tex.shape[-1] >= 3
        ):
            tex = torch.cat(
                (srgb_to_linear(tex[..., :3].float()), tex[..., 3:].float()), -1
            )
        w, h, c = tex.shape[-3], tex.shape[-2], tex.shape[-1]
        if c < 5:
            tex = torch.cat((tex, tex.new_zeros((*tex.shape[:-1], 5 - c))), -1)
        if SETTINGS.raytracing.texture_time_flat:
            # No equality probe here: a static color window arrives already
            # collapsed to one frame (texture_window_collapse, upstream of
            # the merge) and material/normal maps are single-frame tensors,
            # so a map still carrying T frames here is genuinely animated
            # and a probe would only cost a device sync -- which, on the
            # prefetch worker mid-render, waits out the whole queued chunk
            # (measured +5.3 s of a 24 s nn UHD render).
            t = tex.shape[0]
            flat = tex.reshape(1, -1, 5)
        else:
            # Legacy shared time axis: the buffer's leading dim carries the
            # frames (unified at assembly), so the per-map length is 1.
            t = 1
            flat = tex.reshape(tex.shape[0], -1, 5)
        # Content dedup pays only on real images (the N-mobs-one-file case):
        # tiny maps -- the promoted 1x1s -- are already grouped by value per
        # primitive, and each ``torch.equal`` is a device sync with the same
        # queue-drain cost as above, so small maps skip the index entirely.
        dedup = (
            SETTINGS.raytracing.texture_content_dedup
            and flat.shape[1] >= content_dedup_min_texels
        )
        if dedup:
            # An authored stack's texels are pre-decode; keying it apart
            # keeps it from ever answering for (or being answered by) a
            # decoded map that happens to hold equal bits.
            key = (tuple(flat.shape), w, h, t, bool(authored_stack))
            for prior_flat, prior_meta in _texture_index.get(key, ()):
                if torch.equal(prior_flat, flat):
                    return prior_meta
        _texture_tensors.append(flat)
        o = _texel_offset[0]
        _texel_offset[0] += flat.shape[1]
        meta = (o, w, h, t, -1)
        if dedup:
            _texture_index.setdefault(key, []).append((flat, meta))
        return meta

    scene["tex_has_refractive"] = False
    scene["tex_has_reflective"] = False
    # Per-geometry BVH build inputs, captured by the merge sections below and
    # consumed at the end by ``_finalize_bvhs`` (which either builds the trees
    # or, for batches that provably never traverse one, defers them).
    tri_bvh_inputs = bez_bvh_inputs = None
    if triangles:
        # Constant-property promotion: triangles whose color + material params
        # are constant across their corners (and frames) are rendered from a
        # shared 1x1 color + material map instead of per-vertex tri_colors /
        # tri_extra rows (see _split_promotable). Detection is per triangle and
        # grouped by value, so a uniform mob is promoted even when it was batched
        # into one primitive alongside differently-colored mobs. Promoted
        # triangles are ordered LAST (their prims sit past the shrunk arrays,
        # which the guarded kernel reads never index). With promotion inactive
        # every triangle is kept and this reduces byte-identically to the plain
        # per-vertex merge (see _sel: an all-keep selection returns the original
        # tensor, uncopied).
        promote = _constant_promotion_active()
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
                meta = torch.zeros((0, _TEX_META_W), dtype=torch.int32, device=device)
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
        # Which block of global ids each primitive owns, and the mesh keys its
        # surfaces were built from. Plain Python beside the tensor, for tools
        # that need to name the Mob behind a rendered surface (the GUI viewer's
        # pixel inspector); no kernel reads it.
        _obj_sources = []
        for p in plain_triangles + textured_triangles:
            p._rt_tri_obj_global = p._rt_tri_obj + _obj_base
            _obj_n = int(getattr(p, "_rt_tri_obj_n", 1))
            _obj_sources.append((_obj_base, _obj_n, getattr(p, "_obj_keys", None)))
            _obj_base += _obj_n
        scene["tri_obj_sources"] = _obj_sources
        scene["tri_obj"] = (
            _cat_collections(_geom("_rt_tri_obj_global"), 1, "triangle merge")
            .to(torch.int32)
            .contiguous()
        )
        # Per-triangle closed-shell declaration, folded with its transmission
        # exemption at pack time (primitives._pack_projected_flat_geometry):
        # 1.0 where the surface may be coverage-ceilinged as a closed shell.
        # Read only by the sheet compaction, and only under
        # ``solid_shell_alpha`` -- the field is built unconditionally because it
        # is one [T?, N] tensor against a whole batch, but consumed strictly
        # behind the toggle so disabling it changes nothing downstream.
        scene["tri_closed"] = _cat_collections(
            _geom("_rt_tri_closed"), 1, "triangle merge"
        ).contiguous()
        scene["tri_pos"] = _cat_collections(_tri_parts, 1, "triangle merge")
        scene["tri_norm"] = _cat_collections(_geom("_rt_tri_norm"), 1, "triangle merge")
        scene["tri_mat_id"] = _cat_collections(
            _geom("_rt_tri_mat_id"), 1, "triangle merge"
        )
        # Before anything reads the ids: the memory trim below compacts this
        # very table, and the host-side classification at the end of the merge
        # reuses the unique ids this leaves behind.
        _densify_frag_pipeline_ids(scene)
        scene["tri_mat"] = _cat_mat_blocks(_geom("_rt_tri_mat"), "triangle merge")
        lo = _cat_collections(_geom("_rt_frame_lo"), 1, "triangle merge")
        hi = _cat_collections(_geom("_rt_frame_hi"), 1, "triangle merge")
        opaque = _cat_collections(_geom("_rt_frame_opaque"), 1, "triangle merge")
        casts = _cat_collections(_geom("_rt_frame_casts"), 1, "triangle merge")
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
        # normal maps and fall back to per-vertex color, color-map offset -1).
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
            # Width must agree with the real packing (_EXTRA_W): these rows
            # are write-only filler, but a kernel variant compiled against
            # one width and a merge that produces another is how silent
            # drift starts.
            scene["tri_extra"] = torch.zeros((1, 1, _EXTRA_W), device=device)

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
        # map -> per-vertex fallback), 10-12 the per-map time lengths (1 when
        # the buffer's own time axis carries the frames -- see _append_texture),
        # 13-14 the color map's opacity region (row / frames; -1 = host
        # premultiply), 15 its u8 LUT base row (-1 = f32 rows, -2 = u8 bytes
        # with no LUT: an authored endpoint stack) and 16-17 its
        # endpoint-interpolation region (row / frames; -1 = leading axis is
        # time).
        meta_parts, uvs_parts = [], []
        for p in textured_triangles:
            tex_lerp = getattr(p, "_rt_tex_lerp", None)
            color_meta = _append_texture(
                p._rt_texture_map,
                is_color=True,
                u8_ok=bool(getattr(p, "_rt_texture_u8_ok", False)),
                authored_stack=tex_lerp is not None,
            )
            op_off, op_len = _append_tex_opacity(
                getattr(p, "_rt_tex_opacity", None) if color_meta[0] >= 0 else None
            )
            lerp_off, lerp_len = _append_tex_lerp(
                tex_lerp if color_meta[0] >= 0 else None
            )
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
                    [
                        *color_meta[:3],
                        *material_meta[:3],
                        *normal_meta[:3],
                        flags,
                        color_meta[3],
                        material_meta[3],
                        normal_meta[3],
                        op_off,
                        op_len,
                        color_meta[4],
                        lerp_off,
                        lerp_len,
                    ],
                    dtype=torch.int32,
                    device=device,
                )
                .view(1, _TEX_META_W)
                .expand(p._rt_tri_pos.shape[1], _TEX_META_W)
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
            scene["tri_tex_meta"] = _tex_meta_placeholder(device)

        # Collapse temporally-constant triangle tables to one frame. Every
        # consumer reads their time axis as ``f % shape[0]`` (kernels) or
        # ``_expand_frames`` (raster host tables), and _build_mem_trim below
        # is T-agnostic, so a batch whose materials/normals/colors do not
        # animate stores one row instead of T -- tri_mat alone is [T, N, 26],
        # tens of MB of identical frames on ordinary scenes. Under
        # merge_dedup_geometry the same collapse covers ``tri_pos`` and the
        # per-frame id/flag tables it used to skip ("rigid motion lives in
        # tri_pos" forfeited the static case, where the probe is one pass and
        # the saving is (T-1)/T of the largest geometry array).
        if SETTINGS.raytracing.merge_dedup_time:
            keys = [
                "tri_norm",
                "tri_mat_id",
                "tri_mat",
                "tri_colors",
                "tri_extra",
                "tri_uvs",
            ]
            if SETTINGS.raytracing.merge_dedup_geometry:
                keys += ["tri_pos", "tri_obj", "tri_closed"]
                # The per-frame bounds feed the BVH builds (both builders
                # accept ``Tc == 1`` -- one instance spanning all frames --
                # and both reduce a still-per-frame opacity mask over the
                # batch when they do, since each key here collapses on its
                # own: static geometry under a fading mob lands as Tc == 1,
                # To == T) and the raster host tables (all ``f % shape[0]``).
                # lo/hi collapse together or not at all: the builders require
                # one frame count across the pair. Probed through the scene
                # dict so the block shares _dedup_time_group's single sync.
                scene["_probe_lo"], scene["_probe_hi"] = lo, hi
                scene["_probe_opaque"], scene["_probe_casts"] = opaque, casts
                keys += ["_probe_lo", "_probe_hi", "_probe_opaque", "_probe_casts"]
            _dedup_time_group(scene, keys)
            if SETTINGS.raytracing.merge_dedup_geometry:
                lo2, hi2 = scene.pop("_probe_lo"), scene.pop("_probe_hi")
                if lo2.shape[0] == hi2.shape[0]:
                    lo, hi = lo2, hi2
                opaque = scene.pop("_probe_opaque")
                casts = scene.pop("_probe_casts")

        # Per-(frame, prim) visibility/opacity masks for the hybrid raster
        # front-end (settings.hybrid_raster): candidate emission skips
        # invisible triangles and routes proven-opaque ones to the z-prepass.
        # Derived from the same bounds/opacity arrays the STBVH build uses.
        scene["tri_frame_valid"] = (hi >= lo).all(-1).contiguous()
        scene["tri_frame_opaque"] = opaque.contiguous()
        scene["tri_frame_casts"] = casts.contiguous()
        # Triangle STBVHs are built (or deferred) in _finalize_bvhs once every
        # routing flag this batch needs is known.
        tri_bvh_inputs = (lo, hi, opaque, casts)
        if _rts.wf_mem_trim:
            _build_mem_trim(scene, lo, hi, opaque, num_frames, device)
    else:
        scene["tri_pos"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_extra"] = torch.zeros((1, 1, _EXTRA_W), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
        scene["tri_tex_meta"] = _tex_meta_placeholder(device)
        scene["num_colored_triangles"] = 0
        scene["has_material_textures"] = False
        scene["tri_mat_id"] = torch.zeros((1, 1), dtype=torch.int32, device=device)
        scene["tri_obj"] = torch.zeros((1, 1), dtype=torch.int32, device=device)
        scene["tri_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)
        scene["tri_opaque_bvh"] = scene["tri_bvh"]
        scene["tri_frame_valid"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["tri_frame_opaque"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
        scene["tri_frame_casts"] = torch.ones((1, 1), dtype=torch.bool, device=device)
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
        if SETTINGS.raytracing.merge_dedup_time:
            _dedup_time_group(
                scene,
                [
                    "circuit_meta",
                    "circuit_colors",
                    "circuit_border_colors",
                    "edges_2d",
                ],
            )
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
        casts = _cat_collections(
            [p._rt_frame_casts for p in beziers], 1, "bezier merge"
        )
        _record_visibility("bez", lo, hi, opaque)
        # Same static collapse as the triangle bounds above: the builders
        # accept ``Tc == 1`` and every host/kernel consumer reads
        # ``f % shape[0]``; lo/hi must collapse together. One grouped sync,
        # like the circuit tables above.
        if (
            SETTINGS.raytracing.merge_dedup_time
            and SETTINGS.raytracing.merge_dedup_geometry
        ):
            scene["_probe_lo"], scene["_probe_hi"] = lo, hi
            scene["_probe_opaque"], scene["_probe_casts"] = opaque, casts
            _dedup_time_group(
                scene, ["_probe_lo", "_probe_hi", "_probe_opaque", "_probe_casts"]
            )
            lo2, hi2 = scene.pop("_probe_lo"), scene.pop("_probe_hi")
            if lo2.shape[0] == hi2.shape[0]:
                lo, hi = lo2, hi2
            opaque = scene.pop("_probe_opaque")
            casts = scene.pop("_probe_casts")
        # Per-(frame, circuit) visibility, opacity, and AABBs for the hybrid
        # raster frontend.  Proven-opaque circuits now participate in the typed
        # visibility buffer and cull geometry behind large filled 2D shapes;
        # translucent / reflective panes remain in the ordered fragment stream.
        scene["bez_frame_valid"] = (hi >= lo).all(-1).contiguous()
        scene["bez_frame_opaque"] = opaque.contiguous()
        scene["bez_frame_casts"] = casts.contiguous()
        scene["bez_frame_lo"] = lo.contiguous()
        scene["bez_frame_hi"] = hi.contiguous()
        # Bezier STBVHs are built (or deferred) in _finalize_bvhs.
        bez_bvh_inputs = (lo, hi, opaque, casts)
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
        scene["bez_frame_casts"] = torch.ones((1, 1), dtype=torch.bool, device=device)
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
    if "tri_material_ids" not in scene:
        # The triangle branch's ``_densify_frag_pipeline_ids`` already took
        # this unique; only a batch with no triangles at all reaches here.
        ids = scene["tri_mat_id"].detach().cpu()
        scene["tri_material_ids"] = tuple(
            int(value) for value in torch.unique(ids).tolist()
        )
        scene["frag_pipeline_ids"] = ()
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
        p._rt_tex_opacity = p._rt_tex_lerp = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None

    if device.type != "cpu":
        release_torch_memory(force_gc=False)
    # Measured transient device bytes the build allocated above the pre-merge
    # baseline, when opt-in peak tracking is on (see settings.merge_track_peak);
    # -1 marks "not measured". Purely diagnostic -- the arena preflight bounds
    # the build with the merge_gpu_peak_factor estimate, not this value.
    if track_peak:
        scene["_gpu_merge_peak_bytes"] = int(end_cuda_peak(peak_token))
    else:
        scene["_gpu_merge_peak_bytes"] = -1
    _decode_merged_colors(scene)
    first._rt_merged_scene = scene
    return scene


#: The merged arrays holding authored color. Each is ``[..., 5]`` --
#: ``[r, g, b, glow, alpha]`` -- so only channels 0:3 are color. Glow is an
#: additive emissive strength and alpha is coverage; neither is a color and
#: neither is decoded.
_MERGED_COLOR_KEYS = ("tri_colors", "circuit_colors", "circuit_border_colors")

#: Color-valued entries of the built-in material parameter block, by name in
#: ``_MAT_SLOTS`` (resolved rather than hard-coded so a renumbered slot map
#: carries this with it). Authored like every other color, and consumed by
#: arithmetic that runs in linear light: emissive is light the surface adds to
#: the frame, and the two specular tints and the sheen tint multiply lobes the
#: shading stages compute. Everything else in the block is a scalar coefficient
#: -- roughness, metalness, an IOR, an absorption coefficient -- and decoding
#: one would corrupt it.
_MAT_COLOR_SLOT_NAMES = ("emissive", "specular", "specular_color", "sheen_color")


def _decode_merged_colors(scene):
    """Decode the batch's authored color into the linear working space.

    This is the geometry half of the render boundary -- the single point where
    every primitive's color crosses from display-referred (what the author
    typed, and what ``Mob.color`` still reads back) into the linear light the
    shading and compositing arithmetic needs. It runs once per batch, on the
    merged arrays, just before they are cached.

    Deliberately *not* done in :class:`~algan.constants.color.Color`: that is a
    ``torch.Tensor`` subclass which flows through the animation timeline, so
    decoding there would change what ``mob.color`` reads back and would make
    color tweens interpolate in linear light. three.js does decode at its
    ``Color``; Algan does not, and a red-to-blue tween staying perceptually
    even is the reason.

    Three kinds of authored color reach the renderer and all three cross here:

    * the per-vertex color arrays (``_MERGED_COLOR_KEYS``);
    * the **color texture maps**, which are decoded as they are appended
      (``_append_texture(..., is_color=True)``) because a promoted map can alias
      the per-vertex array it came from -- and this is not a corner: with
      constant-property promotion on (the default) a mob whose color and
      material are uniform is rendered from a 1x1 color map, so most content
      arrives that way rather than through ``tri_colors``;
    * the **color slots of the material parameter block**
      (``_MAT_COLOR_SLOT_NAMES``), for primitives on a built-in pipeline. A
      custom fragment pipeline's block is its own layout, so those slots are not
      colors there and are left alone.
    """
    if not rt_settings.linear_color_space:
        return
    for key in _MERGED_COLOR_KEYS:
        arr = scene.get(key)
        if arr is None or arr.shape[-1] < 3:
            continue
        arr[..., :3] = srgb_to_linear(arr[..., :3])
    _decode_material_block_colors(scene)


def _decode_material_block_colors(scene):
    """Decode the color slots of ``tri_mat`` for built-in-pipeline primitives.

    ``tri_mat`` is ``[Tm, N, MAT_W]`` and ``tri_mat_id`` ``[Tm', N]``; a
    primitive whose pipeline id is at or above ``_USER_PIPELINE_BASE`` carries a
    custom layout in the same array and is excluded. Nothing happens at all when
    every material leaves emissive black and the three tints white, which decode
    to themselves -- so the ordinary scene is untouched by this.
    """
    mat = scene.get("tri_mat")
    mat_id = scene.get("tri_mat_id")
    if mat is None or mat_id is None or mat.numel() == 0:
        return
    builtin = (mat_id < _USER_PIPELINE_BASE).all(0)  # [N]
    if not bool(builtin.any()):
        return
    idx = builtin.nonzero(as_tuple=True)[0]
    for name in _MAT_COLOR_SLOT_NAMES:
        start, width = _MAT_SLOTS[name]
        if start + width > mat.shape[-1]:
            continue
        block = mat[:, idx, start : start + width]
        mat[:, idx, start : start + width] = srgb_to_linear(block)


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

    For ``ltype == 5`` (an area-sample row) columns 9/10 instead carry the
    emitter cell's half-extents, column 11 the cell's equal-area radius, and
    columns 12:14 the rectangle's right axis; each of those readers is
    type-guarded.

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
    release_torch_memory(force_gc=False)


def _prefill_background(out, background, frame_offset, device, background_frames=None):
    """Fill the output buffer with the background. Solid colors arrive as a
    float [channels] tensor in [0, 1]; animated/image backgrounds arrive as a
    uint8 row tensor [1 + frames * pixels, channels] (leading padding row).

    ``background_frames`` is how many frames that row tensor covers. Callers
    that know it should pass it: it is the only way to tell an image
    background's own resolution apart from the output's, and a mismatch there
    scrolls a different slice of the background into every frame rather than
    failing (the deferred path already raises for the same reason).
    """
    if isinstance(background, _DeferredBackground):
        _prefill_deferred_background(out, background, frame_offset)
        return

    num_frames, num_pixels, C_out = out.shape
    # Keep the background on its source device. ``Tensor.to(device)`` used to
    # materialize a full, untracked peer tensor on the rendering device before
    # writing the arena-backed output. ``copy_`` below performs device and dtype
    # conversion directly into the reserved destination instead.
    bg = background
    linear = rt_settings.linear_color_space
    if bg.dim() <= 1 or bg.shape[0] == 1:  # solid color (in [0, 1] floats)
        vals = bg.float().flatten()[:5]
        if linear:
            # The background is the second color ingest, and it composites
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
            # decodes the same way the solid color above does. Done in float
            # at 0-255 scale to match what the composite expects.
            head = rows[..., : min(3, k)].float() * (1.0 / 255.0)
            out[..., : min(3, k)].copy_(srgb_to_linear(head) * 255.0)
            if k > 3:
                out[..., 3:k].copy_(rows[..., 3:k])
        else:
            out[..., :k].copy_(rows[..., :k])
        if C_out > k:
            out[..., k:].copy_(rows[..., -1:])


def _downsample_background(background, aa, num_frames, screen_height, screen_width):
    """Average a super-sampled animated/image background down to the output
    resolution (box filter, matching ``post_process_frames``), so the in-place
    anti-aliased renderer -- which samples the background once per output pixel
    -- gets a background at the right resolution.

    Solid colors (resolution-free) and backgrounds that are not super-sampled
    (row count not ``num_frames * (screen_height*aa) * (screen_width*aa)``) are
    returned unchanged.
    """
    bg = background
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
