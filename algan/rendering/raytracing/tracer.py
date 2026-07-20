"""Ray-traced render orchestration: renderer dispatch and the deterministic
wavefront's host-side tile / iteration loop.

:func:`render_batch_raytraced` is the entry point called by the render loop
for a batch of frames. It obtains the merged per-geometry-type arrays +
STBVHs (``scene_builder``), prepares camera / light / environment tensors in
the render arena, and dispatches on the sample count:

* ``samples_per_pixel == 1`` (default) -- the deterministic *wavefront*
  tracer (:func:`raytrace_render_wavefront`): bounded ray tiles run
  generate -> traverse -> shade -> composite kernel stages
  (``wavefront_kernels_taichi.py``), with per-ray state pool-allocated from
  ``ManualMemory``, Taichi-side compaction of the still-active rays between
  host iterations, and a shared continuation pool for reflective /
  refractive splits (an overflowing tile is discarded and retried with fewer
  primaries, never approximated).
* ``samples_per_pixel > 1`` -- the Monte Carlo path-tracing megakernel
  (``path_trace_scene_stbvh``), one thread per (frame, pixel, sample) path,
  accumulating into a float32 per-pixel buffer that ``finalize_samples``
  averages.

Reflections and refraction are inferred from the mob's Three.js-style
material properties. Use ``MeshStandardMaterial(metalness=..., roughness=...)``
or ``MeshPhysicalMaterial`` before spawning; rays bounce up to
``MAX_BOUNCES`` times.

On out-of-memory the frame window is halved and retried
(``OutOfRenderMemory``); see ``render_batch_raytraced``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from algan.errors import UnsupportedFeatureError

import os
import sys
import traceback
from algan.rendering.post_processing.post_process import post_process_frames
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing.settings import _scene_has_user_pipeline
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    KBUF,
    MAX_SURFACES_PER_RAY, path_trace_scene_stbvh, finalize_samples,
)
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _merge_scene,
    _pack_lights,
    _prefill_background,
    copy_merged_scene_to_arena,
)
# NOTE: only immutable settings values may be imported by value here; the
# mutable module globals (SAMPLES_PER_PIXEL, TONEMAP_*, SHADOWS, ...) must be
# read live as ``rt_settings.X`` or their setters silently stop working.
from algan.rendering.raytracing.settings import _get_tonemap_t_val, REFRACT_INITIAL_POOL_RATIO, \
    GATE_EMPTY_TRAVERSALS, is_post_process_tonemap_enabled

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE

# Diagnostics: bumped each time the wavefront engages the Family A+B memory-trim
# path (used by benchmarks/_wf_mem_trim_ab.py to confirm the trim actually fired).
_MEM_TRIM_ENGAGED = [0]
# Number of tile attempts discarded and retried after the shared continuation
# allocator reported overflow. Kept as a list for low-overhead in-process tests.
_WAVEFRONT_POOL_RETRIES = [0]
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _pixel_bases
# ``build_frag_pipelines`` is imported lazily in the render dispatch to avoid a
# module-load import cycle (fragment_shaders -> shading_taichi -> raytracing
# package __init__ -> primitives).
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    compact_ray_slots,
    wf_composite_accum,
    wf_composite_accum_aa,
    wf_finalize_aa,
    wavefront_generate_rays,
    wavefront_shade,
    wavefront_traverse,
    wavefront_traverse_events,
)
from algan.utils.memory_utils import InsufficientMemoryException, empty_cache
from algan.logging.logger import get_logger

logger = get_logger("raytracing")


@dataclass(frozen=True)
class RenderPlan:
    """Resolved renderer route and capability assessment for one batch.

    The plan is attached to ``scene.last_render_plan`` and returned through
    :class:`algan.utils.algan_utils.RenderResult` after a file render.  It is
    intentionally data-only so applications can log or serialize it without
    depending on Taichi/Torch implementation objects.
    """

    backend: Literal["deterministic_wavefront", "monte_carlo"]
    samples_per_pixel: int
    requested_features: tuple[str, ...]
    unsupported_features: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        return not self.unsupported_features

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "samples_per_pixel": self.samples_per_pixel,
            "requested_features": list(self.requested_features),
            "unsupported_features": list(self.unsupported_features),
        }


def _host_tensor(value):
    """Return a detached CPU tensor for render-input preparation.

    Camera/environment/light arithmetic is preparation work.  Keeping it on
    the host prevents a caller-provided render-device tensor from turning a
    subtraction, norm, stack, or cast into an allocation beside the arena.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value, device="cpu")


def _arena_copy(memory, tensor, dtype=None):
    """Copy ``tensor`` into the render arena, optionally casting on copy.

    The source may live on the animation device (normally CPU); ``copy_``
    performs the transfer directly into the reserved byte buffer instead of
    creating a transient ``tensor.to(render_device)`` allocation.
    """
    dtype = tensor.dtype if dtype is None else dtype
    out = memory.get_tensor(tensor.shape, dtype)
    out.copy_(tensor)
    return out


def _arena_values(memory, values, dtype=torch.float32):
    source = torch.tensor(values, dtype=dtype, device="cpu")
    return _arena_copy(memory, source)


def _alloc_wavefront_state(memory, tn, sca_width, *, global_hits=True):
    """Allocate the wavefront's per-ray global state from the render memory pool
    (a bump allocator) rather than fresh ``torch.empty`` tensors.

    The caller snapshots ``memory.get_pointers()`` before each tile and restores
    them after, so this ~hundreds-of-MB of state is released *deterministically*
    at the end of every iteration and the next tile reuses the same arena bytes.
    Previously these were ``torch.empty`` allocations that the CUDA caching
    allocator / Python GC didn't reclaim before the next tile asked for its own,
    so consecutive tiles' state piled up and OOMed the GPU at HD/AA>=2.
    """
    f32 = torch.float32
    i32 = torch.int32
    core = (
        memory.get_tensor((tn, 3), f32),          # rs_ro
        memory.get_tensor((tn, 3), f32),          # rs_rd
        memory.get_tensor((tn, 4), f32),          # rs_acc
        # rs_sca: 0 weight red, 1 t_prev, 2 layer_prev, 3 seam_t, 4 base_dist,
        # 5 weight green, 6 weight blue (colour transport).
        memory.get_tensor((tn, sca_width), f32),  # rs_sca (7 general)
        # rs_int: 0 bounces_left, 1 processed, 2 status, 3 num_hits, 4 drained
        # (column 4 is used only by the legacy sorted-material path
        # (unsupported); the classic kernels index columns 0-3 and never
        # read it).
        memory.get_tensor((tn, 5), i32),          # rs_int
    )
    if global_hits:
        return core + (
            memory.get_tensor((tn, KBUF), f32),   # rs_kt
            memory.get_tensor((tn, KBUF), f32),   # rs_kl
            memory.get_tensor((tn, KBUF), f32),   # rs_ka
            memory.get_tensor((tn, KBUF), f32),   # rs_kb
            memory.get_tensor((tn, KBUF), i32),   # rs_kp
            memory.get_tensor((tn, KBUF), i32),   # rs_kf
        )

    # The supported general renderer no longer attaches a K-buffer to every
    # continuation-pool slot. Keep six tiny placeholders solely so the state
    # tuple remains ABI-compatible with the hybrid raster frontend and the
    # unsupported legacy orchestrators. The general traverse/shade pair uses an
    # exact-size transient surface-event batch allocated for the current active
    # queue instead.
    stub_f = memory.get_tensor((1, 1), f32)
    stub_i = memory.get_tensor((1, 1), i32)
    return core + (stub_f, stub_f, stub_f, stub_f, stub_i, stub_i)


def _wavefront_state_bytes_per_primary(pool_ratio, extra_bytes_per_slot=0,
                                       extra_bytes_per_primary=0):
    """Bytes charged to one initial primary when sizing a wavefront tile.

    Each primary contributes ``pool_ratio`` slots to one *shared* continuation
    pool, rather than owning a private block. ``pix_accum`` remains per primary;
    the two-word shared allocator (next slot + overflow flag) is fixed per tile
    and is accounted separately by the callers. Orchestrator-specific extras
    (for example the sorted path's event record/key arrays) are passed in so
    adaptive tile sizing can account for them."""
    # rs_ro(3) + rs_rd(3) + rs_acc(4) + rs_sca(7) f32, rs_int(5) i32,
    # rs_pix i32 and two arena-backed active-index ping-pong buffers are
    # permanent per pool slot. The 6*KBUF words are now a transient event batch
    # sized to the current active queue. Charge the conservative pool-sized
    # worst case here so automatic tiling cannot overrun the arena even if every
    # continuation slot becomes active simultaneously.
    per_slot = ((3 + 3 + 4 + 7 + 5) * 4 + 6 * KBUF * 4
                + 4 + 2 * 4 + extra_bytes_per_slot)
    # pix_accum(7) f32 -- per primary (RGB+glow premultiplied colour plus the
    # per-channel leftover/background weight of the colour transport).
    return pool_ratio * per_slot + 7 * 4 + extra_bytes_per_primary


def get_wavefront_memory_required(
    merged,
    num_frames,
    width,
    height,
    *,
    light_sources=(),
    environment_map=None,
    near_clip=0.0,
    far_clip=0.0,
):
    """Exact deterministic-wavefront temporary bytes for batch sizing.

    This mirrors the selected route -- the general wavefront, or the legacy
    textured / material-sorted variants (unsupported, but still selectable
    via their settings toggles) -- including route metadata, event records,
    shadow visibility, active-index ping-pong buffers, and fixed counter
    words.  It excludes the possible
    initial four-byte alignment; the caller knows the output buffer's actual
    end pointer and adds that padding exactly.

    With adaptive tiling enabled this returns the exact minimum viable
    one-primary allocation; the runtime grows that tile into whatever remaining
    arena space is available.  With adaptive tiling disabled this returns the
    exact configured static-target allocation peak.
    """
    if int(rt_settings.SAMPLES_PER_PIXEL) > 1:
        return 0

    total_primary = max(
        1, int(num_frames) * int(width) * int(height)
    )
    shadow = bool(rt_settings.SHADOWS)
    lights_extended = any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    )
    has_environment = environment_map is not None
    frag = (
        bool(rt_settings.FRAGMENT_SHADING)
        or shadow
        or _scene_has_user_pipeline(merged)
        or lights_extended
        or has_environment
    )
    has_custom_scatter = bool(frag and _scene_has_custom_scatter(merged))
    # Must stay in step with ``refraction_flag`` in render_batch_raytraced (it
    # sizes the split pool the kernel then writes into).
    refraction = bool(merged.get("has_refractive") or has_custom_scatter
                      or merged.get("has_refl_transparent"))
    pool_ratio = int(REFRACT_INITIAL_POOL_RATIO) if refraction else 1
    target_slots = max(1, int(rt_settings.WAVEFRONT_TILE_RAYS))

    def _route_primary(static_target):
        if rt_settings.WAVEFRONT_TILE_AUTO:
            return 1
        return min(max(1, int(static_target)), total_primary)

    # Remove the per-primary pix_accum tail to recover one pool slot's core
    # state (including rs_pix and both compaction index arrays). The shared
    # allocator is two int32 words per tile and is included in ``fixed`` below.
    per_primary_tail = 7 * torch.float32.itemsize
    base_slot = _wavefront_state_bytes_per_primary(1) - per_primary_tail

    uses_extended_features = (
        lights_extended
        or has_environment
        or float(near_clip) > 0.0
        or float(far_clip) > 0.0
    )
    if merged.get("textured_active") and not uses_extended_features:
        primary = _route_primary(target_slots // pool_ratio)
        pool = primary * pool_ratio
        # gen_meta + compactor output count + shared allocator words.
        fixed = 4 * torch.int32.itemsize
        return pool * base_slot + primary * per_primary_tail + fixed

    if (frag and rt_settings.WAVEFRONT_SORT_MATERIALS is True
            and not uses_extended_features):
        primary = _route_primary((target_slots * 2) // (3 * pool_ratio))
        pool = primary * pool_ratio
        # rs_hit(16 f32), rs_key and rs_eprim, plus per-slot rs_vis when
        # shadows are enabled.
        event_slot = (
            16 * torch.float32.itemsize
            + 2 * torch.int32.itemsize
            + (torch.int32.itemsize if shadow else 0)
        )
        # gen_meta + compactor count, the rs_vis placeholder when shadows
        # compile out, and two shared allocator words.
        fixed = torch.int32.itemsize * (4 if shadow else 5)
        return (
            pool * (base_slot + event_slot)
            + primary * per_primary_tail
            + fixed
        )

    primary = _route_primary(target_slots // pool_ratio)
    pool = primary * pool_ratio
    mem_trim = bool(
        rt_settings.WF_MEM_TRIM
        and merged.get("mem_trim_active")
        and not shadow
        and not has_custom_scatter
        and not refraction
    )
    gen_fused = bool(
        rt_settings.wf_gen_fused_active()
        and pool_ratio == 1
        and float(near_clip) <= 0.0
    )
    # col_row placeholder (unless the scene-owned remap is used), gen_meta,
    # layer_offsets, rs_vis placeholder, and the compactor count.
    metadata = 0 if mem_trim else torch.int32.itemsize
    metadata += (4 if gen_fused else 1) * torch.float32.itemsize
    metadata += (
        8 if (gen_fused or has_environment or float(far_clip) > 0.0) else 2
    ) * torch.float32.itemsize
    # layer/route metadata above, plus rs_vis placeholder + compactor count +
    # the shared allocator's next-slot and overflow words.
    metadata += 4 * torch.int32.itemsize
    return pool * base_slot + primary * per_primary_tail + metadata


def _auto_primary_per_tile(memory, pool_ratio, static_primary,
                           extra_bytes_per_slot=0, extra_bytes_per_primary=0,
                           fixed_bytes=0):
    """Primary rays per wavefront tile, sized from the render pool's free
    bytes when ``settings.WAVEFRONT_TILE_AUTO`` is on (see settings.py for the
    rationale: fewer, bigger tiles amortize the fixed host-side kernel-launch
    cost). Falls back to ``static_primary`` (the WAVEFRONT_TILE_RAYS-derived
    value) for unmanaged pools or when auto is disabled. Byte-identical to any
    other tile size: tiles partition pixels, and every per-pixel computation
    is independent of its tile."""
    if not rt_settings.WAVEFRONT_TILE_AUTO or not getattr(memory, "managed", False):
        return static_primary
    bytes_per_primary = _wavefront_state_bytes_per_primary(
        pool_ratio, extra_bytes_per_slot, extra_bytes_per_primary)
    free = memory.get_num_bytes_remaining()
    # Every per-tile allocation is f32/i32.  The output immediately before it
    # can be uint8 (including an odd-sized five-channel transparent frame), so
    # mirror ManualMemory's one initial four-byte alignment exactly.  All
    # subsequent allocations remain aligned because their sizes are multiples
    # of four.
    alignment_bytes = (-memory.current_pointer) % torch.float32.itemsize
    safety = min(1.0, max(0.0, float(rt_settings.WAVEFRONT_TILE_SAFETY)))
    usable = (int(free * safety)
              - alignment_bytes - int(fixed_bytes))
    budget = max(0, usable) // bytes_per_primary
    hi = max(1, rt_settings.WAVEFRONT_TILE_MAX // pool_ratio)
    lo = min(hi, max(1, rt_settings.WAVEFRONT_TILE_MIN // pool_ratio))
    # The minimum is a launch-amortisation preference, not permission to
    # overrun the arena.  When less than the preferred floor fits, use the
    # exact smaller value; a one-primary allocation is attempted only when no
    # primary can fit, preserving the normal single-frame OOM diagnostic.
    if budget < lo:
        return max(1, budget)
    return min(budget, hi)


class _ArenaRayCompactor:
    """Stable-lifetime ray-index buffers owned by ``ManualMemory``.

    PyTorch's comparison/advanced-index/nonzero chain created several fresh
    CUDA tensors after the render arena had already been reserved.  A Taichi
    filter kernel now writes directly into these two ping-pong buffers and a
    one-word counter, so compaction cannot exceed the arena allowance.
    """

    def __init__(self, memory, capacity, dtype=torch.int32):
        self.capacity = int(capacity)
        self.a = memory.get_tensor((self.capacity,), dtype)
        self.b = memory.get_tensor((self.capacity,), dtype)
        self.count = memory.get_tensor((1,), dtype)
        self.current = self.a
        self.spare = self.b
        self.size = 0

    def initial(self, size):
        self.size = int(size)
        torch.arange(self.size, out=self.current[:self.size])
        return self.current[:self.size]

    def select(self, rs_int, desired_status, *, source=None, scan_pool=False,
               rs_key=None, desired_key=0):
        if source is None:
            source = self.current[:self.size]
        source_size = self.capacity if scan_pool else int(source.numel())
        self.count.zero_()
        compact_ray_slots(
            source, source_size, bool(scan_pool), int(desired_status),
            # ``rs_key`` is a compile-time-unused argument without a key
            # predicate. Reuse the current index array instead of reserving an
            # otherwise dead placeholder word.
            rs_int, self.current if rs_key is None else rs_key,
            rs_key is not None, int(desired_key), self.spare, self.count)
        size = int(self.count.item())
        self.current, self.spare = self.spare, self.current
        self.size = size
        return self.current[:size]


_kernel_compile_notice_shown = False


def _show_kernel_compile_notice():
    global _kernel_compile_notice_shown
    if _kernel_compile_notice_shown:
        return
    _kernel_compile_notice_shown = True
    logger.info(
        "Preparing render kernels. If this is the first render on this machine"
        " (or after an update), compiling the GPU kernels can take several"
        " minutes. Compiled kernels are cached, so subsequent renders start"
        " immediately."
    )


def _append_env_texture(textures, env, intensity, device):
    """Append an equirect environment map to the shared flat texel buffer.

    Returns the widened buffer and the map's placement meta
    ``(offset, width, height, intensity)`` for the shade kernel (packed into
    the ``layer_offsets`` ndarray -- the kernel is at the 64-arg ceiling).
    Texels are stored column-major (``offset + x * height + y``) to match
    ``_sample_tex_vec5``.
    """
    # Environment resampling and concatenation are scene preparation, not
    # rendering.  Always perform them on the host even when a direct caller
    # supplied render-device textures; the completed storage is uploaded once
    # through ``copy_merged_scene_to_arena``.
    device = torch.device("cpu")
    textures = textures.detach().cpu()
    env = _host_tensor(env).float()
    max_w = 2048
    if env.shape[1] > max_w:
        scale = max_w / env.shape[1]
        env = torch.nn.functional.interpolate(
            env.permute(2, 0, 1).unsqueeze(0), scale_factor=scale,
            mode="area")[0].permute(1, 2, 0)
    h, w = int(env.shape[0]), int(env.shape[1])
    texels = torch.zeros((w * h, 5), dtype=torch.float32, device=device)
    texels[:, :3] = env.permute(1, 0, 2).reshape(w * h, 3)
    texels[:, 3] = 1.0
    if textures.dtype != torch.float32:
        texels = texels.to(textures.dtype)
    offset = int(textures.shape[1])
    texels = texels.unsqueeze(0).expand(textures.shape[0], -1, -1)
    textures = torch.cat((textures, texels), 1).contiguous()
    return textures, (offset, w, h, float(intensity))


def _env_sh_coeffs(env, intensity):
    """Order-1 spherical-harmonics irradiance of an equirect environment map,
    as the linear form ``E(n) / pi = A + Bx*nx + By*ny + Bz*nz`` consumed by
    the in-kernel ENV_SH light row. A uniform map of colour ``c`` yields
    ``A = c, B = 0`` -- i.e. it lights like an ambient light of colour ``c``.
    """
    import math

    e = _host_tensor(env).float()
    if e.shape[0] > 32 or e.shape[1] > 64:
        e = torch.nn.functional.adaptive_avg_pool2d(
            e.permute(2, 0, 1).unsqueeze(0), (16, 32))[0].permute(1, 2, 0)
    H, W = int(e.shape[0]), int(e.shape[1])
    v = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    theta = math.pi * v                       # 0 at the top row (y = +1)
    y = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    phi = (u - 0.5) * (2.0 * math.pi)         # matches _sample_env_map
    x = torch.cos(phi).unsqueeze(0) * sin_t.unsqueeze(1)
    z = torch.sin(phi).unsqueeze(0) * sin_t.unsqueeze(1)
    yy = y.unsqueeze(1).expand(H, W)
    dw = (sin_t * (math.pi / H) * (2.0 * math.pi / W)).unsqueeze(1)
    L = e * float(intensity)
    w = dw.expand(H, W).unsqueeze(-1)
    Y00 = 0.28209479177387814
    Y1 = 0.4886025119029199
    L00 = (L * (Y00 * w)).sum((0, 1))
    L1x = (L * ((Y1 * x).unsqueeze(-1) * w)).sum((0, 1))
    L1y = (L * ((Y1 * yy).unsqueeze(-1) * w)).sum((0, 1))
    L1z = (L * ((Y1 * z).unsqueeze(-1) * w)).sum((0, 1))
    a0 = math.pi              # irradiance convolution coefficients
    a1 = 2.0 * math.pi / 3.0
    A = a0 * Y00 * L00 / math.pi
    Bx = a1 * Y1 * L1x / math.pi
    By = a1 * Y1 * L1y / math.pi
    Bz = a1 * Y1 * L1z / math.pi
    return A, Bx, By, Bz


def _append_env_sh_light(light_pos, light_col, num_lights, env, intensity,
                         device):
    """Add the environment map's diffuse irradiance as one ENV_SH light row
    (type 6) to the packed lights, widening the color rows to 16 columns if
    they are still in the compact point-light packing."""
    A, Bx, By, Bz = _env_sh_coeffs(env, intensity)
    row = torch.zeros(16)
    row[0:3] = A
    row[3] = 6.0  # LIGHT_ENV_SH
    row[6:9] = Bx
    row[9:12] = By
    row[12:15] = Bz
    T = light_pos.shape[0] if num_lights > 0 else 1
    row = row.view(1, 1, 16).expand(T, 1, 16).to(device)
    zero_pos = torch.zeros((T, 1, 3), device=device)
    if num_lights == 0:
        return zero_pos.contiguous(), row.contiguous(), 1
    if light_col.shape[2] < 16:
        pad = torch.zeros((light_col.shape[0], light_col.shape[1],
                           16 - light_col.shape[2]), device=device)
        light_col = torch.cat((light_col, pad), -1)
    light_pos = torch.cat((light_pos, zero_pos), 1).contiguous()
    light_col = torch.cat((light_col, row), 1).contiguous()
    return light_pos, light_col, num_lights + 1


def _build_render_plan(samples_per_pixel, scene_environment_map, merged,
                       light_sources=()):
    """Resolve the renderer route and feature compatibility for a batch."""
    samples_requested = max(1, int(samples_per_pixel))
    backend = (
        "monte_carlo" if samples_requested > 1 else "deterministic_wavefront"
    )
    requested = []
    if scene_environment_map is not None:
        requested.append("environment maps")
    if bool(merged.get("has_refractive")):
        requested.append("refractive materials")
    if _scene_has_user_pipeline(merged):
        requested.append("custom fragment-shader pipelines")
    if any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    ):
        requested.append("extended lights")

    unsupported = tuple(requested) if samples_requested > 1 else ()
    return RenderPlan(
        backend=backend,
        samples_per_pixel=samples_requested,
        requested_features=tuple(requested),
        unsupported_features=unsupported,
    )


def _validate_render_capabilities(samples_per_pixel, scene_environment_map,
                                  merged, light_sources=()):
    """Validate that the selected renderer can honor the authored scene.

    ``samples_per_pixel > 1`` selects the Monte Carlo megakernel. Several
    features currently exist only in the deterministic wavefront renderer;
    silently discarding them is more dangerous than failing early. The global
    unsupported-feature policy permits an explicit warning/ignore migration
    mode for benchmarks and legacy projects.
    """
    plan = _build_render_plan(
        samples_per_pixel,
        scene_environment_map,
        merged,
        light_sources,
    )
    if plan.unsupported_features:
        feature_list = ", ".join(plan.unsupported_features)
        rt_settings.report_unsupported_features(
            "The Monte Carlo renderer selected by samples_per_pixel > 1 "
            f"cannot honor: {feature_list}. Set samples_per_pixel to 1 to use "
            "the deterministic wavefront renderer, remove those features, or "
            "set_unsupported_feature_policy('warn'/'ignore') explicitly."
        )

    # Keep direct mutation of the legacy globals from bypassing the guarded
    # public setters. These backends are known-broken and must not render a
    # misleading result.
    if bool(getattr(rt_settings, "WF_TEXTURED", False)):
        raise UnsupportedFeatureError(
            "The legacy textured wavefront renderer is unsupported."
        )
    if getattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", "auto") is True:
        raise UnsupportedFeatureError(
            "The legacy sorted-material wavefront renderer is unsupported."
        )
    return plan


def render_batch_raytraced(primitives, scene, screen_width, screen_height,
                            time_start, time_end, background_color,
                            transparent_background, ray_origin, screen_point,
                            screen_basis, anti_alias_level=1, light_sources=(),
                            memory=None, post_processes=(), **kwargs):
    """Render frames [time_start, time_end) of a primitive batch by ray
    tracing into a fixed [frames, pixels, channels] buffer.

    On out-of-memory the time window is halved and retried; per-frame memory
    is just the output buffer (plus post-processing), independent of scene
    depth complexity or bounce count.
    """
    _show_kernel_compile_notice()
    # Read the user-toggleable settings *live* from the settings module.
    # These names used to be imported by value at module-import time, which
    # froze them before user code ran -- silently disabling
    # set_ray_traced_shadows() / set_samples_per_pixel() / etc. for anyone
    # calling the setters after `import algan` (i.e. everyone).
    SAMPLES_PER_PIXEL = rt_settings.SAMPLES_PER_PIXEL
    SHADOWS = rt_settings.SHADOWS
    FRAGMENT_SHADING = rt_settings.FRAGMENT_SHADING
    MAX_BOUNCES = rt_settings.MAX_BOUNCES
    TONEMAP_EXPOSURE = rt_settings.TONEMAP_EXPOSURE
    INDIRECT_BOUNCE_STRENGTH = rt_settings.INDIRECT_BOUNCE_STRENGTH
    scene_env_map = getattr(scene, "environment_map", None)
    env_map = scene_env_map if int(SAMPLES_PER_PIXEL) <= 1 else None
    env_source = (
        env_map.detach().cpu() if torch.is_tensor(env_map) else env_map
    )
    env_meta = getattr(primitives[0], "_rt_env_meta", None)
    merged = getattr(primitives[0], "_rt_device_scene", None)
    if merged is None:
        merged_host = _merge_scene(primitives)
        # Validate on host metadata before reserving/copying the persistent
        # device scene. Unsupported combinations therefore fail before costly
        # arena allocations or any Taichi kernel compilation.
        plan = _validate_render_capabilities(
            SAMPLES_PER_PIXEL,
            scene_env_map,
            merged_host,
            light_sources,
        )
        if env_map is not None:
            merged_host = dict(merged_host)
            texture_device = merged_host["textures"].device
            merged_host["textures"], env_meta = _append_env_texture(
                merged_host["textures"], env_source,
                float(getattr(scene, "environment_intensity", 1.0)),
                texture_device)
        merged = copy_merged_scene_to_arena(
            merged_host, memory, persist=True)
    else:
        plan = _validate_render_capabilities(
            SAMPLES_PER_PIXEL,
            scene_env_map,
            merged,
            light_sources,
        )
    scene.last_render_plan = plan

    aa = max(1, int(anti_alias_level))
    # Refraction is only implemented by the general wavefront tracer, which is
    # already where every deterministic (samples <= 1) batch goes -- so this
    # only gates the refraction template, not routing. The Monte Carlo
    # megakernel (samples > 1) ignores the refractive index.
    refractive_det = (bool(merged.get("has_refractive"))
                      and int(SAMPLES_PER_PIXEL) <= 1)
    # Semi-transparent PBR surfaces split off a reflection branch, so they need
    # the same pool + split code the refraction path compiles in. No routing
    # implication: the deterministic (samples <= 1) path is already wavefront.
    refl_transparent_det = (bool(merged.get("has_refl_transparent"))
                            and int(SAMPLES_PER_PIXEL) <= 1)

    # Extended lights (directional / ambient / hemisphere / spot / area /
    # falloff / soft shadows) and environment maps are features of the
    # deterministic general wavefront with per-fragment lighting: their
    # presence forces fragment shading on and routes away from the textured /
    # sorted variants. Plain point-light scenes keep the compact light packing
    # and are untouched.
    lights_extended = (int(SAMPLES_PER_PIXEL) <= 1 and any(
        getattr(l, "_render_aux", None) is not None
        for l in (light_sources or ())))
    cam = getattr(scene, "camera", None)
    near_clip = float(getattr(cam, "near", 0.0) or 0.0)
    far_clip = float(getattr(cam, "far", 0.0) or 0.0)

    # Anti-aliasing strategy. Default: render super-sampled (screen size * aa)
    # and let post-processing box-average down. ALGAN_INPLACE_AA=1 instead
    # averages ``aa^2`` jittered sub-pixel rays *in place* at the output
    # resolution, so the frame buffer stays ``screen_width x screen_height``
    # regardless of ``aa`` (aa^2x less render memory than super-sampling): the
    # wavefront runs the full gen→traverse→shade→compact→composite pipeline
    # once per sub-pixel sample, accumulating into a float buffer and
    # averaging at the end, while the Monte Carlo megakernel folds the aa^2
    # factor into its per-pixel sample count (see samples_eff below).
    inplace_aa = os.getenv("ALGAN_INPLACE_AA", "0") != "0"
    if inplace_aa:
        width = screen_width
        height = screen_height
        kernel_aa = aa  # in-kernel sub-pixel averaging factor
        post_aa = 1  # post-processing does not down-sample
    else:
        width = screen_width * aa
        height = screen_height * aa
        kernel_aa = 1
        post_aa = aa

    C_out = 5 if transparent_background else 4
    device = memory.data.device
    num_frames = merged["num_frames"]

    # Camera snapshots stay on the animation/source device during prefetch.
    # Complete their small vector math there, then copy only the kernel-facing
    # results into arena-backed render tensors.
    cam_origin_host = _expand_frames(
        _flat_frames(_host_tensor(ray_origin), (3,)),
        num_frames).contiguous()
    sp_host = _expand_frames(
        _flat_frames(_host_tensor(screen_point), (3,)),
        num_frames).contiguous()
    sb_host = _expand_frames(
        _flat_frames(_host_tensor(screen_basis), (3, 3)), num_frames)
    pbx_host, pby_host = _pixel_bases(sb_host)
    # World units per screen pixel per unit distance (for border widths). Border
    # widths are authored in *anti-aliased* pixels (see BezierCircuit), so this
    # always uses the super-sampled height (screen_height * aa), whether or not
    # the frame buffer itself is super-sampled.
    b1_norm = sb_host[:, 1].norm(p=2, dim=-1)
    screen_dist = (sp_host - cam_origin_host).norm(p=2, dim=-1)
    pixel_world_scale_host = (
        2.0 / (screen_height * aa * b1_norm * screen_dist).clamp_min(1e-12)
    ).contiguous()
    cam_origin = _arena_copy(memory, cam_origin_host)
    sp = _arena_copy(memory, sp_host)
    pbx = _arena_copy(memory, pbx_host)
    pby = _arena_copy(memory, pby_host)
    pixel_world_scale = _arena_copy(memory, pixel_world_scale_host)

    # In-place AA samples the background once per output pixel, so an
    # animated/image background that arrived super-sampled must be averaged
    # down to the output resolution first (solid colors are resolution-free).
    if aa > 1 and inplace_aa:
        background_color = _downsample_background(
            background_color, aa, time_end - time_start,
            screen_height, screen_width)

    # A deferred-BVH batch (scene_builder._finalize_bvhs) holds placeholder
    # trees; the Monte Carlo megakernel traverses unconditionally, so build
    # the real trees now if that is where this batch is headed. (The
    # deterministic wavefront has its own later, finer-grained check.)
    if merged.get("bvh_deferred") and int(SAMPLES_PER_PIXEL) > 1:
        from algan.rendering.raytracing.scene_builder import (
            build_deferred_bvhs)
        build_deferred_bvhs(merged)
    tri_bvh = merged["tri_bvh"]
    pn_bvh = merged["pn_bvh"]
    bez_bvh = merged["bez_bvh"]
    # A geometry type absent from the whole batch has only a placeholder BVH;
    # tell the deterministic kernel so it skips that empty traversal per ray.
    if GATE_EMPTY_TRAVERSALS:
        has_tri = 1 if merged["num_triangles"] > 0 else 0
        has_pn = 1 if merged["num_pn"] > 0 else 0
        has_bez = 1 if merged["num_circuits"] > 0 else 0
    else:  # benchmarking escape hatch: traverse every (possibly empty) tree
        has_tri = has_pn = has_bez = 1
    t_val = _get_tonemap_t_val()

    samples = max(1, int(SAMPLES_PER_PIXEL))
    # In-place AA folds the anti-alias super-sampling into the Monte Carlo
    # sample count: each of the ``aa^2`` sub-pixels would have drawn ``samples``
    # random rays jittered over its own cell and then been averaged down, which
    # is equivalent (same total, same expectation) to drawing ``samples * aa^2``
    # rays jittered over the whole output pixel. (The wavefront/super-sample
    # path keeps ``kernel_aa == 1``, so ``samples_eff == samples`` there.)
    samples_eff = samples * (kernel_aa * kernel_aa)

    # Deterministic per-fragment shading is active for a single-sample,
    # non-physical render with the toggle on; it needs the scene's point lights
    # in the kernel. (Physical mode packs the same lights for its own path.)
    # Deterministic hard shadows are evaluated inside the per-fragment lighting
    # model, so enabling them implies fragment shading for this render.
    det_shadows = bool(SHADOWS) and samples <= 1
    # A mob with a custom fragment pipeline (Mob.set_fragment_shader) forces
    # fragment shading on for this render, without a persistent global toggle.
    scene_has_frag_pipeline = _scene_has_user_pipeline(merged)
    det_frag = ((bool(FRAGMENT_SHADING) or det_shadows or scene_has_frag_pipeline
                 or lights_extended or env_map is not None)
                and samples <= 1)
    frag_flag = 1 if det_frag else 0
    shadow_flag = 1 if det_shadows else 0
    # Composed custom fragment-shader pipelines injected into the shade kernel as
    # a flat ti.template() tuple; empty () keeps the built-in / vertex-shaded
    # kernel specialization unchanged (see shading_taichi._run_frag_pipeline).
    # A non-empty ``frag_scatters`` tuple switches the monolithic shade kernel's
    # bounce block to per-material scatter dispatch (custom ray bouncing); it is
    # only assembled when a pipeline in *this* scene overrides bouncing, so an
    # ordinary scene keeps the byte-identical built-in bounce block (empty ()).
    if det_frag:
        from algan.rendering.shaders.fragment_shaders import (
            build_frag_pipelines, build_frag_scatters)
        frag_pipelines = build_frag_pipelines()
        frag_scatters = (build_frag_scatters()
                         if _scene_has_custom_scatter(merged) else ())
    else:
        frag_pipelines = ()
        frag_scatters = ()
    # Refraction (general wavefront only; see refractive_det above). A custom
    # scatter may spawn a transmitted branch, so it needs the same split pool +
    # transmitted-branch code the refraction path compiles in.
    refraction_flag = 1 if (refractive_det or refl_transparent_det
                            or frag_scatters) else 0
    # Environment map: append its texels to the shared texture buffer (the
    # merged dict is shallow-copied -- it is cached across batches) and, when
    # its ambient lighting is enabled, its SH irradiance as an extra light row.
    if det_frag:
        light_device = torch.device("cpu")
        light_pos_host, light_col_host, num_lights = _pack_lights(
            light_sources, num_frames, light_device)
        if env_map is not None and getattr(scene, "environment_ambient", True):
            light_pos_host, light_col_host, num_lights = _append_env_sh_light(
                light_pos_host, light_col_host, num_lights, env_source,
                float(getattr(scene, "environment_intensity", 1.0)),
                light_device)
        light_pos = _arena_copy(memory, light_pos_host)
        light_col = _arena_copy(memory, light_col_host)
    elif samples > 1:
        light_pos = light_col = None
        num_lights = 0
    else:
        # Deterministic, fragment shading off: tiny placeholders for the
        # (compiled-out) material/light kernel args.
        light_pos = memory.get_tensor((1, 1, 3), torch.float32)
        light_col = memory.get_tensor((1, 1, 3), torch.float32)
        light_pos.zero_()
        light_col.zero_()
        num_lights = 0

    def render_chunk(start, end):
        # The Monte Carlo kernels launch one thread per (frame, pixel,
        # sample) path; keep the flattened index within int32 range. (The
        # deterministic kernels loop the aa^2 sub-pixels serially per pixel, so
        # only the Monte Carlo path multiplies the thread count by the samples.)
        if (samples > 1 and
                (end - start) * width * height * samples_eff >= 1 << 31):
            logger.warning(f'Render OOM, splitting {start}:{end}')
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "samples_per_pixel * resolution exceeds the ray tracer's "
                    "per-launch path budget (2^31). Please lower the sample "
                    "count, resolution or anti-alias level.")
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)
        entry_pointers = memory.get_pointers()
        try:
            out_dtype = torch.float32 if is_post_process_tonemap_enabled() else torch.uint8
            out = memory.get_tensor((end - start, width * height, C_out),
                                    out_dtype)
            _prefill_background(out, background_color, start - time_start,
                                device)
            accum = None
            if samples > 1:
                # f32 per-pixel sample sums, averaged by finalize_samples.
                accum = memory.get_tensor((end - start, width * height, 5),
                                          torch.float32)
                accum.zero_()
            # Coplanar layer order: circuits < triangles < PN patches.
            layer_offset_triangles = float(merged["num_circuits"])
            layer_offset_pn = layer_offset_triangles + float(
                merged["num_triangles"])
            shared_args = (
                tri_bvh.blocks, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
                merged["tri_colors"], merged["tri_uvs"], merged["tri_tex_meta"],
                merged["textures"], int(merged["num_colored_triangles"]),
                pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan, pn_bvh.first_leaf,
                merged["pn_ctrl"], merged["pn_norm"], merged["pn_extra"],
                merged["pn_colors"],
                bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan, bez_bvh.first_leaf,
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"], merged["edges_2d"],
                merged["edge_accel"],
                cam_origin, sp, pbx, pby, pixel_world_scale,
                int(start), int(end), int(width), int(height),
                float(width // 2), float(height // 2),
                layer_offset_triangles, layer_offset_pn, int(MAX_BOUNCES),
                1 if transparent_background else 0)
            if samples > 1:
                from algan.rendering.raytracing.refit_bvh import RefitBVH
                path_trace_scene_stbvh(1 if isinstance(tri_bvh, RefitBVH)
                                       else 0,
                                       *shared_args, samples_eff,
                                       float(INDIRECT_BOUNCE_STRENGTH),
                                       merged["pn_obb"], out, accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 t_val, float(TONEMAP_EXPOSURE),
                                 accum, out)
            else:
                # col_row/gen/layer metadata, AA accumulation and every tile
                # buffer are wavefront-only. Release them before post
                # processing so the two phases share the same temporary arena
                # range (the batch estimator models max(wavefront, post), not
                # their sum).
                with memory.temp():
                    raytrace_render_wavefront(
                        tri_bvh, pn_bvh, bez_bvh, merged,
                        cam_origin, sp, pbx, pby, pixel_world_scale,
                        int(start), int(end), int(width), int(height),
                        float(width // 2), float(height // 2),
                        layer_offset_triangles, layer_offset_pn,
                        has_tri, has_pn, has_bez, int(MAX_BOUNCES),
                        light_pos, light_col, int(num_lights),
                        frag_flag, frag_pipelines, frag_scatters, shadow_flag,
                        refraction_flag,
                        1 if transparent_background else 0, memory, out,
                        kernel_aa, lights_extended=lights_extended,
                        env_meta=env_meta, near_clip=near_clip,
                        far_clip=far_clip)
            frames = out.view(end - start, height, width, C_out)
            frames = post_process_frames(memory,
                frames, anti_alias_level=post_aa,
                post_processes=list(post_processes), apply_fxaa=scene.render_settings.fxaa)
            memory.set_pointers(entry_pointers)
            return [frames]
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            logger.warning(f'Render OOM, splitting {start}:{end}')
            memory.set_pointers(entry_pointers)
            # All this stuff is necessary to free local variables assigned during the previous render attempt.
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.clear_frames(exc_traceback)
            # traceback.print_tb(exc_traceback)
            # exc_traceback.tb_next.tb_frame.clear()
            # Release the failed allocation (e.g. the wavefront's large per-ray
            # state) so it doesn't fragment/block the smaller retry.
            empty_cache()
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "Insufficient memory to ray trace a single frame. "
                    "Please lower the resolution or anti-alias level.") from None
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)

    chunks = render_chunk(time_start, time_end)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, 0)


def _run_wavefront_tiles(memory, out, *, n, width, height, time_start,
                         transparent, aa_level, pool_ratio, primary_per_tile,
                         cam_origin, screen_point, pixel_basis_x,
                         pixel_basis_y, half_screen_w, half_screen_h,
                         max_bounces, near_clip, run_tile,
                         auto_extra_slot_bytes=0, auto_extra_primary_bytes=0,
                         auto_fixed_bytes=0, gen_fused=False, raster=False,
                         global_hits=True):
    """Run deterministic-wavefront screen tiles with a shared split pool.

    ``run_tile(tile_start, tn_primary, pool, state, rs_pix, pix_accum,
    rs_alloc)`` supplies the variant-specific traverse/shade iteration. The
    first ``tn_primary`` slots hold primary rays and every spawned continuation
    atomically appends to the shared remainder of the pool. ``rs_alloc`` is a
    two-word counter: next free slot and overflow flag.

    Pool exhaustion is never accepted as a rendering approximation. An
    overflowing attempt is discarded before compositing and retried with half
    as many primaries while retaining the same pool capacity. Thus the
    continuation headroom doubles on every retry without increasing arena
    memory, and there is no fixed per-pixel split limit.
    """
    t_val = _get_tonemap_t_val()
    i32 = torch.int32
    f32 = torch.float32
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    # Constant primary-ray init rows (rs_sca / rs_int). When there is no split
    # pool and no near clip these are identical for every primary, so they are
    # filled with coalesced broadcast copies here rather than by the strided
    # per-ray stores in the memory-bound generate kernel (``write_const == 0``
    # tells the kernel the host already filled them). base_dist (rs_sca[4]) is
    # 0 without a near clip; _ACTIVE == 0 (rs_int cols 1-3 are all zero).
    const_fill = (pool_ratio == 1 and near_clip <= 0.0)
    if const_fill:
        sca_init = torch.tensor([1.0, 0.0, 1e30, -1e30, 0.0, 1.0, 1.0],
                                dtype=f32, device=out.device)
        int_init = torch.tensor([int(max_bounces), 0, 0, 0],
                                dtype=i32, device=out.device)

    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    # Adaptive tile sizing (after aa_accum so free-bytes accounting sees it).
    # The allocator's two int32 words are fixed per tile rather than per ray.
    primary_per_tile = _auto_primary_per_tile(
        memory, pool_ratio, primary_per_tile,
        auto_extra_slot_bytes, auto_extra_primary_bytes,
        auto_fixed_bytes + 2 * torch.int32.itemsize)
    primary_capacity = min(max(1, int(primary_per_tile)), max(1, int(n)))
    shared_pool_capacity = primary_capacity * int(pool_ratio)

    # Remember a successful reduced tile size after an overflow so every
    # subsequent tile does not repeat the same failed first attempt. The pool
    # itself remains at ``shared_pool_capacity`` for all splitting tiles, so a
    # smaller final tile automatically receives the otherwise-unused slots.
    learned_primary_cap = primary_capacity

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5
            tile_start = 0

            while tile_start < n:
                remaining = n - tile_start
                attempt_primary = min(learned_primary_cap, remaining)
                # Split-free renders do not need a shared reserve; keeping their
                # final tile exact avoids scanning unused slots. Splitting
                # renders retain the full fixed pool across retries and tiles.
                pool = (shared_pool_capacity if pool_ratio > 1
                        else attempt_primary)

                while True:
                    state_ptrs = memory.get_pointers()
                    try:
                        state = _alloc_wavefront_state(
                            memory, pool, 7, global_hits=global_hits)
                        (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                         rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = state
                        rs_pix = memory.get_tensor((pool,), i32)
                        pix_accum = memory.get_tensor((attempt_primary, 7),
                                                      f32)
                        # [0] next free shared slot, [1] overflow flag. The
                        # classic generation kernel initialises both. Fused
                        # generation is split-free, but zeroing keeps the
                        # state well-defined.
                        rs_alloc = memory.get_tensor((2,), i32)

                        if raster:
                            # Hybrid raster front-end: no generate pass.
                            # Primary slots are written (or retired) by
                            # raster_first_shade; pre-mark every pool slot
                            # DONE (status 1) so the post-raster full-pool
                            # compaction sees only the continuations the
                            # raster actually spawned, and seed the shared
                            # allocator past the primary slots.
                            pix_accum.zero_()
                            rs_int[:, 2].fill_(1)
                            rs_alloc.zero_()
                            rs_alloc[0] = attempt_primary
                        elif gen_fused:
                            pix_accum.zero_()
                            rs_alloc.zero_()
                        else:
                            # rs_acc and pix_accum start all-zero, and the
                            # constant rs_sca / rs_int primary init rows are
                            # filled here for the split-free, near-clip-free
                            # case. Doing this as contiguous memsets /
                            # broadcast copies is far cheaper than the strided
                            # per-ray stores the generate kernel otherwise
                            # does through the AoS [ray, channel] layout
                            # (memory-bound kernel); byte-identical -- same
                            # values, just coalesced.
                            rs_acc.zero_()
                            pix_accum.zero_()
                            if const_fill:
                                rs_sca[:attempt_primary].copy_(sca_init)
                                # rs_int is 5 wide; generate only wrote cols
                                # 0-3 (col 4 is the legacy sorted-path
                                # "drained" field it never touched), so fill
                                # only 0-3 to leave col 4 exactly as before --
                                # byte-identical.
                                rs_int[:attempt_primary, :4].copy_(int_init)
                            wavefront_generate_rays(
                                cam_origin, screen_point, pixel_basis_x,
                                pixel_basis_y, int(time_start), int(width),
                                int(height), float(half_screen_w),
                                float(half_screen_h), int(max_bounces),
                                int(tile_start), int(attempt_primary),
                                float(jx), float(jy), float(near_clip),
                                0 if const_fill else 1,
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                rs_pix, pix_accum, rs_alloc)

                        run_tile(tile_start, attempt_primary, pool, state,
                                 rs_pix, pix_accum, rs_alloc)
                    except (InsufficientMemoryException,
                            torch.OutOfMemoryError) as exc:
                        memory.set_pointers(state_ptrs)
                        if not raster:
                            raise
                        # Raster scratch (fragment records, sort scratch, the
                        # sparse shadow-event queue) scales with the tile's
                        # fragment volume, which the up-front tile sizing
                        # cannot know. Retry the tile with half the primaries
                        # rather than discarding the whole frame window.
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt_primary <= 1:
                            raise OutOfRenderMemory(
                                "Raster scratch did not fit for a single "
                                "pixel. Lower the resolution or transparency "
                                "complexity.") from exc
                        next_primary = max(1, attempt_primary // 2)
                        _WAVEFRONT_POOL_RETRIES[0] += 1
                        logger.warning(
                            "Hybrid raster tile allocation failed for "
                            f"{tile_start}:{tile_start + attempt_primary}; "
                            f"retrying with {next_primary} primaries")
                        learned_primary_cap = min(learned_primary_cap,
                                                  next_primary)
                        attempt_primary = next_primary
                        continue

                    overflow = (pool_ratio > 1
                                and int(rs_alloc[1].item()) != 0)
                    if overflow:
                        memory.set_pointers(state_ptrs)
                        if attempt_primary <= 1:
                            raise OutOfRenderMemory(
                                "A single pixel's deterministic ray tree "
                                f"exceeded the shared wavefront pool of {pool} "
                                "slots. Lower MAX_BOUNCES / transparency "
                                "complexity, or increase WAVEFRONT_TILE_RAYS.")
                        next_primary = max(1, attempt_primary // 2)
                        _WAVEFRONT_POOL_RETRIES[0] += 1
                        logger.warning(
                            "Wavefront continuation pool overflow for tile "
                            f"{tile_start}:{tile_start + attempt_primary}; "
                            f"retrying with {next_primary} primaries and the "
                            f"same {pool}-slot pool")
                        learned_primary_cap = min(learned_primary_cap,
                                                  next_primary)
                        attempt_primary = next_primary
                        continue

                    if do_aa:
                        wf_composite_accum_aa(
                            int(time_start), int(width), int(height),
                            1 if transparent else 0, int(tile_start),
                            pix_accum, out, aa_accum)
                    else:
                        wf_composite_accum(
                            int(time_start), int(width), int(height),
                            1 if transparent else 0, int(tile_start),
                            pix_accum, t_val,
                            float(rt_settings.TONEMAP_EXPOSURE), out)
                    memory.set_pointers(state_ptrs)
                    tile_start += attempt_primary
                    break

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val,
                       float(rt_settings.TONEMAP_EXPOSURE), aa_accum, out)


def raytrace_render_wavefront(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, frag_flag, frag_pipelines,
        frag_scatters, shadow_flag,
        refraction_flag, transparent, memory, out, aa_level=1,
        lights_extended=False, env_meta=None, near_clip=0.0, far_clip=0.0):
    """Wavefront orchestration for the general triangle/PN/bezier path.

    Persistent continuation state is stage-split in global memory and PyTorch
    compacts ray indices between host iterations. Hit records are different:
    traversal writes one exact-size ``[num_active, KBUF]`` transient event
    batch, shade consumes it immediately, and the arena range is then reused.
    No pool-wide K-buffer is attached to secondary radiance ray slots. The
    persistent scalar state carries ``base_dist`` for Bezier border widths
    across bounces.

    ``frag_flag``/``shadow_flag`` select the deterministic per-fragment shading
    and binary hard-shadow paths (compile-time templates of the shade kernel);
    ``light_pos``/``light_col`` feed both.

    ``frag_scatters`` is the per-pipeline custom ray-continuation (scatter) tuple
    (empty when no scene pipeline overrides bouncing); a non-empty tuple switches
    the monolithic shade kernel's bounce block from the built-in
    opacity/reflectivity/Fresnel logic to per-material scatter dispatch
    (``_run_frag_scatter``), so users can customise reflection / refraction /
    pass-through. Empty keeps the built-in bounce block byte-identical.

    ``refraction_flag`` enables simultaneous reflection + refraction (glass): the
    shade kernel SPLITS such a ray, continuing the reflected branch in place and
    spawning the refracted branch into a free pool slot. The pool is therefore
    over-allocated by ``pool_ratio`` (only when refraction / custom scatter is on)
    -- it holds ``primary_per_tile`` one-per-pixel rays plus spare slots for
    split branches, at fixed total memory (fewer pixels per tile instead of
    bigger per-ray state). Each ray commits into a shared per-pixel accumulator
    (``pix_accum``) on termination, so a pixel's reflected and refracted branches
    sum.

    When fragment shading is active, ``settings.WAVEFRONT_SORT_MATERIALS``
    selects the shade architecture: the monolithic ``wavefront_shade`` kernel
    below (the default, and the only supported path -- it handles custom
    scatter and normal-mapped lighting, and on the built-in materials it is
    faster because it drains up to KBUF hits per launch while sorting pays
    per-event kernel round trips and host syncs), or the UNSUPPORTED legacy
    Cycles-style *sorted* pipeline (rays suspended at their material events,
    bucketed by (geometry type, material pipeline id) and shaded by dedicated
    per-material kernels -- see ``wavefront_sorted_kernels_taichi``), routed
    only when explicitly forced (``set_material_sorting(True)``) and kept for
    reference only. The vertex-shaded path below is unaffected either way.
    """
    from algan.rendering.raytracing import settings as rt_settings
    # UNSUPPORTED legacy textured-surface shader (Surface / flat-triangle
    # scenes): shades from three per-triangle texture lookups instead of
    # per-vertex arrays. Only reachable via the opt-in WF_TEXTURED toggle;
    # kept for reference. Extended lights, environment maps and near/far
    # clipping live in the monolithic general shade kernel below; a scene
    # using any of them skips the textured / sorted variants.
    uses_extended_features = (bool(lights_extended) or env_meta is not None
                              or near_clip > 0.0 or far_clip > 0.0)
    if merged.get("textured_active") and not uses_extended_features:
        return _raytrace_render_wavefront_textured(
            tri_bvh, pn_bvh, bez_bvh, merged,
            cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
            pixel_world_scale, time_start, time_end, width, height,
            half_screen_w, half_screen_h, layer_offset_triangles,
            layer_offset_pn, has_tri, has_pn, has_bez, max_bounces,
            light_pos, light_col, num_lights, refraction_flag,
            transparent, memory, out, aa_level)
    sort_mode = rt_settings.WAVEFRONT_SORT_MATERIALS
    # The monolith handles custom scatter + normal maps, so it is the default
    # for every fragment-shaded scene; the UNSUPPORTED legacy sorted pipeline
    # runs only when explicitly forced (kept for reference -- see settings).
    use_sorted = (bool(frag_flag) and (sort_mode is True)
                  and not uses_extended_features
                  and not merged.get("bez_has_reflective", False))
    if use_sorted:
        return _raytrace_render_wavefront_sorted(
            tri_bvh, pn_bvh, bez_bvh, merged,
            cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
            pixel_world_scale, time_start, time_end, width, height,
            half_screen_w, half_screen_h, layer_offset_triangles,
            layer_offset_pn, has_tri, has_pn, has_bez, max_bounces,
            light_pos, light_col, num_lights, frag_pipelines, shadow_flag,
            refraction_flag, transparent, memory, out, aa_level)
    device = out.device
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height

    # Compile-time walk selector: the merge builds either all-classic or
    # all-refit trees for a batch (see scene_builder._build_accel), so the
    # tree object's type is the authority -- never the live toggle, which the
    # user may have flipped since this batch was merged/prewarmed.
    from algan.rendering.raytracing.refit_bvh import RefitBVH
    bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    # Pool over-allocation for ray splitting. Only glass (reflective+refractive)
    # surfaces split, so spare slots are reserved only when refraction is on; the
    # non-refractive path keeps pool_ratio == 1 (one slot per pixel, as before).
    pool_ratio = REFRACT_INITIAL_POOL_RATIO if refraction_flag else 1
    # Read live (settings convention): runtime-mutable for tile-size A/B.
    primary_per_tile = max(1, rt_settings.WAVEFRONT_TILE_RAYS // pool_ratio)

    # Family A+B memory-trim: engage only for the no-shadow, non-refractive,
    # scatter-free triangle path (the trim arrays are built by scene_builder
    # only when ALGAN_WF_MEM_TRIM). Rebinds the triangle geometry + BVH to the
    # band-reordered/compacted variants and supplies the col_row remap; PN and
    # bezier are untouched. tri_colors/tri_extra stay in their original order
    # (addressed via col_row). ``mem_trim == 0`` leaves everything byte-identical.
    mem_trim = (1 if (rt_settings.WF_MEM_TRIM and merged.get("mem_trim_active")
                      and shadow_flag == 0 and len(frag_scatters) == 0
                      and refraction_flag == 0) else 0)
    if mem_trim:
        _MEM_TRIM_ENGAGED[0] += 1
        t_bvh = merged["tri_bvh_t"]
        a_pos, a_norm = merged["tri_pos_t"], merged["tri_norm_t"]
        a_mat, a_matid = merged["tri_mat_t"], merged["tri_mat_id_t"]
        a_uvs, a_meta = merged["tri_uvs_t"], merged["tri_tex_meta_t"]
        col_row_arr = merged["tri_col_row"]
    else:
        t_bvh = tri_bvh
        a_pos, a_norm = merged["tri_pos"], merged["tri_norm"]
        a_mat, a_matid = merged["tri_mat"], merged["tri_mat_id"]
        a_uvs, a_meta = merged["tri_uvs"], merged["tri_tex_meta"]
        col_row_arr = memory.get_tensor((1,), i32)
        col_row_arr.zero_()
    opaque_closest = int(
        rt_settings.WF_OPAQUE_CLOSEST
        and merged.get("all_visible_opaque", False)
        and not refraction_flag
        and len(frag_scatters) == 0
        and not mem_trim
        and not merged.get("textured_active", False))
    opaque_prepass = int(
        rt_settings.WF_OPAQUE_PREPASS
        and merged.get("has_any_opaque", False)
        and merged.get("has_any_translucent", False)
        and not merged.get("has_uncertain_texture_alpha", False)
        and not refraction_flag
        and len(frag_scatters) == 0
        and not mem_trim
        and not merged.get("textured_active", False))
    # Hybrid raster front-end: replace iteration zero with an opaque typed
    # visibility buffer plus ordered transparent fragment runs. PN patches are
    # conservatively routed to the classic path without altering their geometry.
    # Primary hard and soft shadows use the exact sparse event queue. Non-zero
    # emitter radii are sampled with the same deterministic golden-angle fan as
    # the classic wavefront path.
    use_raster = (
        rt_settings.HYBRID_RASTER
        and merged.get("tri_frame_valid") is not None
        and (merged["num_triangles"] > 0 or merged["num_circuits"] > 0)
        and merged["num_pn"] == 0
        and not merged.get("textured_active")
        and mem_trim == 0
        and len(frag_scatters) == 0
        and near_clip <= 0.0
        and max(1, int(aa_level)) <= 1
    )
    # Fused primary-ray generation (settings.WF_GEN_FUSED): the tile's first
    # traverse generates its rays in-kernel and the first shade uses the
    # implicit initial state, skipping the standalone generate pass. Only for
    # split-free (one slot per pixel, so pix == r), near-clip-free (implicit
    # base_dist == 0) renders on the one-sample-per-pixel AA path (fixed
    # 0.5/0.5 jitter). Everything else keeps the classic generate kernel.
    def _ensure_bvhs():
        # Deferred-BVH batch (scene_builder._finalize_bvhs): build the real
        # trees and rebind everything derived from the placeholders. Deferral
        # implies mem_trim was inactive at merge, so t_bvh is plain tri_bvh.
        nonlocal tri_bvh, pn_bvh, bez_bvh, t_bvh, bvh_refit
        from algan.rendering.raytracing.scene_builder import (
            build_deferred_bvhs)
        build_deferred_bvhs(merged)
        tri_bvh = merged["tri_bvh"]
        pn_bvh = merged["pn_bvh"]
        bez_bvh = merged["bez_bvh"]
        t_bvh = tri_bvh
        bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    if merged.get("bvh_deferred") and (shadow_flag != 0 or not use_raster):
        # Runtime routing needs the trees after all: primary shadows trace
        # them inside iteration zero, and a batch that fell back to classic
        # primary traversal (near clip, in-place AA, flipped toggles, ...)
        # walks them for every primary ray.
        _ensure_bvhs()

    gen_fused = (rt_settings.wf_gen_fused_active() and pool_ratio == 1
                 and near_clip <= 0.0 and max(1, int(aa_level)) <= 1
                 and not use_raster)
    if gen_fused or use_raster:
        gen_meta = _arena_values(
            memory, [0.5, 0.5, float(half_screen_w),
                     float(half_screen_h)], f32)
    else:
        gen_meta = memory.get_tensor((1,), f32)
        gen_meta.zero_()
    if gen_fused or use_raster or env_meta is not None or far_clip > 0.0:
        # Extras packed behind the two layer offsets (the shade kernel is at
        # the 64-arg ceiling): env map placement in the shared texel buffer +
        # the camera's far clip distance, and -- read only by the fused first
        # shade iteration -- max_bounces. The kernel detects them by length.
        eo, ew, eh, ei = env_meta if env_meta is not None else (0, 0, 0, 0.0)
        layer_offsets_t = _arena_values(
            memory,
            [float(layer_offset_triangles), float(layer_offset_pn),
             float(eo), float(ew), float(eh), float(ei), float(far_clip),
             float(max_bounces)], f32)
    else:
        layer_offsets_t = _arena_values(
            memory,
            [float(layer_offset_triangles), float(layer_offset_pn)], f32)

    tri_screen = None
    if use_raster:
        from algan.rendering.raytracing.raster_pipeline import (
            precompute_triangle_projection)
        tri_screen = precompute_triangle_projection(
            merged, cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
            half_screen_w, half_screen_h, memory)

    def run_tile(tile_start, tn_primary, pool, state, rs_pix,
                 pix_accum, rs_alloc):
        (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
         rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = state
        # One-element placeholder for the classic shade kernel's legacy
        # deferred-visibility argument. Raster primary shadows use their own
        # compact sparse any-hit event queue inside iteration zero.
        rs_vis = memory.get_tensor((1,), i32)
        compactor = _ArenaRayCompactor(memory, pool, i32)
        it = 0
        if use_raster:
            # Iteration 0 via the raster front-end: primary visibility is
            # resolved and shaded in full (straight-ray transparency capped
            # only by MAX_SURFACES_PER_RAY); only bounced continuations enter
            # the classic loop below. Raster scratch (z-buffer, fragment
            # records, CSR runs and the sparse shadow-event queue) is
            # phase-local: the temporary arena scope releases it before the
            # bounce loop's per-iteration surface-event batches are allocated.
            from algan.rendering.raytracing.raster_pipeline import (
                raster_iteration_zero)
            with memory.temp():
                raster_iteration_zero(
                    merged, tri_screen, memory, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, pixel_world_scale,
                    layer_offsets_t, gen_meta, light_pos, light_col,
                    num_lights, col_row_arr, frag_flag, frag_pipelines,
                    int(rt_settings.WF_SKIP_UNLIT_NORMAL), refraction_flag,
                    time_start, width, height, half_screen_w, half_screen_h,
                    tile_start, tn_primary, state, rs_pix, pix_accum,
                    rs_alloc, shadow_flag, t_bvh, pn_bvh, bez_bvh,
                    layer_offset_triangles, layer_offset_pn, max_bounces)
            # A continuation-pool overflow is detected and retried by the tile
            # host (with half as many primaries); skip the bounce loop for the
            # doomed attempt.
            if pool_ratio > 1 and int(rs_alloc[1].item()) != 0:
                return
            active = compactor.select(rs_int, 0, source=compactor.current,
                                      scan_pool=True)
            if active.numel() > 0 and merged.get("bvh_deferred"):
                # A continuation actually spawned (a reflective/refractive
                # surface the merge-time deferral analysis missed): build the
                # deferred trees before the bounce loop traverses them.
                _ensure_bvhs()
            it = 1
        else:
            active = compactor.initial(tn_primary)
        while active.numel() > 0 and it < max_iters:
            na = int(active.numel())
            # Fused generation: the tile's first iteration generates rays in
            # traversal and shades with implicit initial state (separate
            # compile-time instantiations). Later iterations and unfused
            # renders use the same compact event-batch kernels with materialized
            # persistent state.
            first = 1 if (gen_fused and it == 0) else 0
            # The hit batch is phase-local: traversal writes one compact
            # [active ray, KBUF] surface-event record and shade consumes it in
            # the same host iteration. Releasing this arena scope before
            # compaction removes the six permanent [pool, KBUF] arrays from
            # secondary radiance state while preserving the existing four-hit
            # traversal/shading behavior.
            with memory.temp():
                hit_f = memory.get_tensor((na, KBUF, 4), f32)
                hit_i = memory.get_tensor((na, KBUF, 2), i32)
                wavefront_traverse_events(
                    active, na,
                    t_bvh.blocks, t_bvh.node_miss, t_bvh.leaf_prim,
                    t_bvh.leaf_tspan, int(t_bvh.first_leaf),
                    a_pos,
                    pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                    pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                    merged["pn_ctrl"],
                    merged["pn_obb"],
                    bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
                    bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                    merged["circuit_meta"],
                    merged["edges_2d"], merged["edge_accel"],
                    merged["tri_opaque_bvh"].blocks,
                    merged["tri_opaque_bvh"].node_miss,
                    merged["tri_opaque_bvh"].leaf_prim,
                    merged["tri_opaque_bvh"].leaf_tspan,
                    int(merged["tri_opaque_bvh"].first_leaf),
                    merged["pn_opaque_bvh"].blocks,
                    merged["pn_opaque_bvh"].node_miss,
                    merged["pn_opaque_bvh"].leaf_prim,
                    merged["pn_opaque_bvh"].leaf_tspan,
                    int(merged["pn_opaque_bvh"].first_leaf),
                    merged["bez_opaque_bvh"].blocks,
                    merged["bez_opaque_bvh"].node_miss,
                    merged["bez_opaque_bvh"].leaf_prim,
                    merged["bez_opaque_bvh"].leaf_tspan,
                    int(merged["bez_opaque_bvh"].first_leaf),
                    pixel_world_scale,
                    float(layer_offset_triangles), float(layer_offset_pn),
                    bvh_refit,
                    int(has_tri), int(has_pn), int(has_bez),
                    opaque_closest,
                    opaque_prepass,
                    int(time_start), int(width), int(height),
                    int(tile_start),
                    rs_ro, rs_rd, rs_sca, rs_int,
                    hit_f, hit_i, rs_pix,
                    first, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, gen_meta)
                wavefront_shade(
                    active, na,
                    t_bvh.blocks, t_bvh.node_miss, t_bvh.leaf_prim,
                    t_bvh.leaf_tspan, int(t_bvh.first_leaf),
                    a_pos, a_norm,
                    merged["tri_extra"],
                    merged["tri_colors"], a_uvs,
                    a_meta,
                    merged["textures"],
                    int(merged["num_colored_triangles"]),
                    col_row_arr,
                    pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                    pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                    merged["pn_ctrl"], merged["pn_norm"],
                    merged["pn_extra"],
                    merged["pn_colors"], merged["pn_obb"],
                    bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
                    bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                    merged["circuit_meta"], merged["circuit_colors"],
                    merged["circuit_border_colors"],
                    merged["edges_2d"], merged["edge_accel"],
                    pixel_world_scale,
                    layer_offsets_t,
                    int(frag_flag), frag_pipelines, frag_scatters,
                    int(shadow_flag),
                    int(refraction_flag),
                    bvh_refit,
                    int(has_tri), int(has_pn), int(has_bez),
                    0,
                    int(rt_settings.WF_SKIP_UNLIT_NORMAL),
                    int(mem_trim),
                    opaque_closest,
                    first,
                    a_matid, a_mat,
                    merged["pn_mat_id"], merged["pn_mat"],
                    light_pos, light_col, int(num_lights),
                    int(time_start), int(width), int(height),
                    int(tile_start),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                    hit_f, hit_i,
                    rs_pix, pix_accum, rs_alloc, rs_vis)
            active = compactor.select(
                rs_int, 0, source=active,
                scan_pool=(pool_ratio != 1
                           or not rt_settings.WF_COMPACT_ACTIVE_ONLY))
            it += 1

    _run_wavefront_tiles(
        memory, out, n=n, width=width, height=height,
        time_start=time_start, transparent=transparent,
        aa_level=aa_level, pool_ratio=pool_ratio,
        primary_per_tile=primary_per_tile,
        cam_origin=cam_origin, screen_point=screen_point,
        pixel_basis_x=pixel_basis_x, pixel_basis_y=pixel_basis_y,
        half_screen_w=half_screen_w, half_screen_h=half_screen_h,
        max_bounces=max_bounces, near_clip=near_clip,
        run_tile=run_tile,
        # rs_vis placeholder + the compactor's output-count word.
        auto_fixed_bytes=2 * torch.int32.itemsize,
        gen_fused=gen_fused, raster=use_raster, global_hits=False)


def _raytrace_render_wavefront_textured(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, refraction_flag,
        transparent, memory, out, aa_level=1):
    """UNSUPPORTED legacy textured-surface wavefront orchestration (Surface /
    flat-triangle scenes only; no longer maintained, kept for reference --
    see ``settings.WF_TEXTURED``). Same generate -> traverse -> shade ->
    composite tile loop as the monolithic :func:`raytrace_render_wavefront`,
    but the shade stage is ``wf_shade_textured`` reading the three
    per-triangle texture banks built by
    ``scene_builder._build_textured_scene``. PN and bezier traversals gate out
    (the scene is all flat triangles)."""
    from algan.rendering.raytracing.wavefront_textured_kernels_taichi import (
        wf_shade_textured)
    from algan.rendering.raytracing import settings as rt_settings

    device = out.device
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height

    # Feature templates (compiled into the shade kernel one at a time to measure
    # each feature's cost -- see settings.WF_TEXTURED_FEATURES).
    feat = int(rt_settings.WF_TEXTURED_FEATURES)
    feat_bez = 1 if (feat & rt_settings.WF_TEX_BEZ) else 0
    feat_scatter = 1 if (feat & rt_settings.WF_TEX_SCATTER) else 0
    feat_shadows = 1 if (feat & rt_settings.WF_TEX_SHADOWS) else 0
    feat_normalmap = 1 if (feat & rt_settings.WF_TEX_NORMALMAP) else 0
    # Compile the bezier traversal into the (shared) traverse kernel when the
    # bezier feature is on, so its cost is included even on a bezier-free scene.
    has_bez_eff = 1 if (feat_bez or has_bez) else 0

    pool_ratio = REFRACT_INITIAL_POOL_RATIO if refraction_flag else 1
    primary_per_tile = max(1, rt_settings.WAVEFRONT_TILE_RAYS // pool_ratio)
    # Placeholder for the fused-generation traverse args (classic generate
    # kernel is kept on this path; the gen block compiles out).
    gen_meta = memory.get_tensor((1,), f32)
    gen_meta.zero_()

    def run_tile(tile_start, tn_primary, pool, state, rs_pix,
                 pix_accum, rs_alloc):
        (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
         rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = state
        compactor = _ArenaRayCompactor(memory, pool, i32)
        active = compactor.initial(tn_primary)
        it = 0
        while active.numel() > 0 and it < max_iters:
            na = int(active.numel())
            wavefront_traverse(
                active, na,
                tri_bvh.blocks, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                merged["tri_pos"],
                pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                merged["pn_ctrl"], merged["pn_obb"],
                bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                merged["circuit_meta"],
                merged["edges_2d"], merged["edge_accel"],
                merged["tri_opaque_bvh"].blocks,
                merged["tri_opaque_bvh"].node_miss,
                merged["tri_opaque_bvh"].leaf_prim,
                merged["tri_opaque_bvh"].leaf_tspan,
                int(merged["tri_opaque_bvh"].first_leaf),
                merged["pn_opaque_bvh"].blocks,
                merged["pn_opaque_bvh"].node_miss,
                merged["pn_opaque_bvh"].leaf_prim,
                merged["pn_opaque_bvh"].leaf_tspan,
                int(merged["pn_opaque_bvh"].first_leaf),
                merged["bez_opaque_bvh"].blocks,
                merged["bez_opaque_bvh"].node_miss,
                merged["bez_opaque_bvh"].leaf_prim,
                merged["bez_opaque_bvh"].leaf_tspan,
                int(merged["bez_opaque_bvh"].first_leaf),
                pixel_world_scale,
                float(layer_offset_triangles), float(layer_offset_pn),
                0,
                int(has_tri), int(has_pn), int(has_bez_eff),
                0,
                0,
                int(time_start), int(width), int(height),
                int(tile_start),
                rs_ro, rs_rd, rs_sca, rs_int,
                rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix,
                0, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, gen_meta)
            wf_shade_textured(
                active, na,
                merged["tri_pos"], merged["tri_norm"],
                merged["tx_uv"],
                merged["tx_color_idx"], merged["tx_mat_idx"],
                merged["tx_surf_idx"],
                merged["tx_color_bank"], merged["tx_color_meta"],
                merged["tx_mat_bank"], merged["tx_mat_meta"],
                merged["tx_surf_bank"], merged["tx_surf_meta"],
                merged["tx_nmap_idx"], merged["tx_nmap_bank"],
                merged["tx_nmap_meta"],
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"],
                tri_bvh.blocks, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                pixel_world_scale,
                float(layer_offset_triangles),
                light_pos, light_col, int(num_lights),
                int(refraction_flag),
                int(feat_bez), int(feat_scatter), int(feat_shadows),
                int(feat_normalmap),
                int(time_start), int(width), int(height),
                int(tile_start),
                rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf,
                rs_pix, pix_accum, rs_alloc)
            active = compactor.select(
                rs_int, 0, source=active,
                scan_pool=(pool_ratio != 1
                           or not rt_settings.WF_COMPACT_ACTIVE_ONLY))
            it += 1

    _run_wavefront_tiles(
        memory, out, n=n, width=width, height=height,
        time_start=time_start, transparent=transparent,
        aa_level=aa_level, pool_ratio=pool_ratio,
        primary_per_tile=primary_per_tile,
        cam_origin=cam_origin, screen_point=screen_point,
        pixel_basis_x=pixel_basis_x, pixel_basis_y=pixel_basis_y,
        half_screen_w=half_screen_w, half_screen_h=half_screen_h,
        max_bounces=max_bounces, near_clip=0.0,
        run_tile=run_tile,
        # Compactor output-count word.
        auto_fixed_bytes=torch.int32.itemsize)


def _scene_has_custom_scatter(merged):
    """True if any merged primitive's material pipeline carries a custom
    scatter func (user-controlled ray bouncing). The monolithic wavefront
    shade kernel dispatches these directly; this check only decides whether
    the scatter templates get compiled in (scatter-free scenes stay on the
    scatter-free, byte-identical default path). Cheap: exits on the
    tensor-max user-pipeline pre-check for the (overwhelmingly common)
    all-built-in scene."""
    cached = merged.get("has_custom_scatter")
    if cached is not None:
        return bool(cached)
    if not _scene_has_user_pipeline(merged):
        merged["has_custom_scatter"] = False
        return False
    from algan.rendering.shaders.fragment_shaders import build_frag_scatters
    scatters = build_frag_scatters()
    for prefix in ("tri", "pn"):
        material_ids = merged.get(f"{prefix}_material_ids")
        if material_ids is None:
            arr = merged.get(f"{prefix}_mat_id")
            material_ids = (() if arr is None or not arr.numel() else
                            torch.unique(arr.detach().cpu()).tolist())
        for pid in material_ids:
            i = int(pid) - _USER_PIPELINE_BASE
            if 0 <= i < len(scatters) and scatters[i] is not None:
                merged["has_custom_scatter"] = True
                return True
    merged["has_custom_scatter"] = False
    return False


def _raytrace_render_wavefront_sorted(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, frag_pipelines, shadow_flag,
        refraction_flag, transparent, memory, out, aa_level=1):
    """UNSUPPORTED legacy Cycles-style sorted-material orchestration of the
    fragment-shading wavefront (no longer maintained, kept for reference; only
    reachable via ``set_material_sorting(True)`` -- see
    ``wavefront_sorted_kernels_taichi`` for the kernel split).

    Per host iteration: rays needing a K-buffer refill are traversed
    (``wavefront_traverse``, unchanged), ``wf_peel`` advances every ray with
    unconsumed hits to its next material event (compositing bezier hits and
    background escapes inline), the pending events get their shadow bits from
    one ``wf_shadow_event`` launch, and each material bucket -- rays whose
    event key ``(geometry type << 8) | pipeline id`` matches -- is shaded by a
    dedicated ``wf_shade_event`` instantiation with that material's pipeline
    and scatter funcs as compile-time templates. The bucket table is built
    once per chunk from the merged scene's material ids, so only materials
    actually present cost a kernel instantiation.
    """
    from algan.rendering.shaders.fragment_shaders import build_frag_scatters
    from algan.rendering.raytracing.shading_taichi import (
        _USER_PIPELINE_BASE, builtin_pipeline_fn)
    from algan.rendering.raytracing.wavefront_sorted_kernels_taichi import (
        ST_PEEL, ST_SHADE, ST_TRAVERSE,
        default_scatter, wf_peel, wf_shade_event, wf_shadow_event)

    device = out.device
    i32 = torch.int32
    f32 = torch.float32
    n = (time_end - time_start) * width * height

    # Material bucket table: one entry per (geometry type, pipeline id) pair
    # present in the merged scene. Each bucket carries the composed pipeline
    # func + scatter func to inject and the geometry type's parameter block.
    scatters = build_frag_scatters()

    def _resolve(pid):
        if pid < _USER_PIPELINE_BASE:
            return builtin_pipeline_fn(pid), default_scatter
        fn = frag_pipelines[pid - _USER_PIPELINE_BASE]
        sc = scatters[pid - _USER_PIPELINE_BASE]
        return fn, (sc if sc is not None else default_scatter)

    buckets = []
    has_custom_scatter = False
    if merged["num_triangles"] > 0:
        tri_material_ids = merged.get("tri_material_ids")
        if tri_material_ids is None:
            tri_material_ids = torch.unique(
                merged["tri_mat_id"].detach().cpu()).tolist()
        for pid in tri_material_ids:
            fn, sc = _resolve(int(pid))
            has_custom_scatter |= sc is not default_scatter
            buckets.append(((1 << 8) | int(pid), fn, sc, merged["tri_mat"]))
    if merged["num_pn"] > 0:
        pn_material_ids = merged.get("pn_material_ids")
        if pn_material_ids is None:
            pn_material_ids = torch.unique(
                merged["pn_mat_id"].detach().cpu()).tolist()
        for pid in pn_material_ids:
            fn, sc = _resolve(int(pid))
            has_custom_scatter |= sc is not default_scatter
            buckets.append(((2 << 8) | int(pid), fn, sc, merged["pn_mat"]))
    # A custom scatter may spawn transmitted branches, which need the glass
    # split pool (and the peel's IOR sampling) even in a scene with no
    # refractive surface.
    refraction_flag = 1 if (refraction_flag or has_custom_scatter) else 0

    # Worst case: every one of MAX_SURFACES_PER_RAY hits is a material event
    # (one peel+shade pass each) plus a traverse per K-buffer refill / bounce.
    max_iters = (MAX_SURFACES_PER_RAY + MAX_SURFACES_PER_RAY // KBUF
                 + max_bounces * 2 + 8)

    pool_ratio = REFRACT_INITIAL_POOL_RATIO if refraction_flag else 1
    # The sorted path carries ~1.5x the classic per-ray state (the event
    # record + keys), so tiles hold fewer rays for the same memory envelope.
    primary_per_tile = max(1, (rt_settings.WAVEFRONT_TILE_RAYS * 2)
                           // (3 * pool_ratio))
    # Placeholder for the fused-generation traverse args (classic generate
    # kernel is kept on this path; the gen block compiles out).
    gen_meta = memory.get_tensor((1,), f32)
    gen_meta.zero_()

    def run_tile(tile_start, tn_primary, pool, state, rs_pix,
                 pix_accum, rs_alloc):
        (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
         rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = state
        # Event state: hit record, sort key, event primitive index and
        # per-event shadow visibility bits (placeholder when unused).
        rs_hit = memory.get_tensor((pool, 16), f32)
        rs_key = memory.get_tensor((pool,), i32)
        rs_eprim = memory.get_tensor((pool,), i32)
        rs_vis = memory.get_tensor((pool,) if shadow_flag else (1,),
                                   i32)
        compactor = _ArenaRayCompactor(memory, pool, i32)
        # The drained counter (rs_int col 4, pool garbage after
        # allocation) must be 0 for every ray entering ST_TRAVERSE;
        # the kernels maintain that invariant from here on.
        rs_int[:, 4].zero_()

        it = 0
        while it < max_iters:
            # Build every work list directly in the arena. Pending SHADE
            # events never survive an iteration, so TRAVERSE then PEEL scans
            # are sufficient to decide termination.
            trav = compactor.select(
                rs_int, ST_TRAVERSE, scan_pool=True)
            if trav.numel():
                wavefront_traverse(
                    trav, int(trav.numel()),
                    tri_bvh.blocks, tri_bvh.node_miss,
                    tri_bvh.leaf_prim, tri_bvh.leaf_tspan,
                    int(tri_bvh.first_leaf),
                    merged["tri_pos"],
                    pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                    pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                    merged["pn_ctrl"],
                    merged["pn_obb"],
                    bez_bvh.blocks, bez_bvh.node_miss,
                    bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
                    int(bez_bvh.first_leaf),
                    merged["circuit_meta"],
                    merged["edges_2d"], merged["edge_accel"],
                    merged["tri_opaque_bvh"].blocks,
                    merged["tri_opaque_bvh"].node_miss,
                    merged["tri_opaque_bvh"].leaf_prim,
                    merged["tri_opaque_bvh"].leaf_tspan,
                    int(merged["tri_opaque_bvh"].first_leaf),
                    merged["pn_opaque_bvh"].blocks,
                    merged["pn_opaque_bvh"].node_miss,
                    merged["pn_opaque_bvh"].leaf_prim,
                    merged["pn_opaque_bvh"].leaf_tspan,
                    int(merged["pn_opaque_bvh"].first_leaf),
                    merged["bez_opaque_bvh"].blocks,
                    merged["bez_opaque_bvh"].node_miss,
                    merged["bez_opaque_bvh"].leaf_prim,
                    merged["bez_opaque_bvh"].leaf_tspan,
                    int(merged["bez_opaque_bvh"].first_leaf),
                    pixel_world_scale,
                    float(layer_offset_triangles),
                    float(layer_offset_pn),
                    0,
                    int(has_tri), int(has_pn), int(has_bez),
                    0,
                    0,
                    int(time_start), int(width), int(height),
                    int(tile_start),
                    rs_ro, rs_rd, rs_sca, rs_int,
                    rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix,
                    0, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, gen_meta)
            # Traversal transitions its rays to PEEL, so one post-traverse
            # scan naturally includes both those rays and previously-peeling
            # rays without a temporary torch.cat.
            peel_idx = compactor.select(rs_int, ST_PEEL, scan_pool=True)
            if trav.numel() == 0 and peel_idx.numel() == 0:
                break
            if peel_idx.numel():
                wf_peel(
                    peel_idx, int(peel_idx.numel()),
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    merged["tri_uvs"], merged["tri_tex_meta"],
                    merged["textures"],
                    int(merged["num_colored_triangles"]),
                    merged["pn_ctrl"], merged["pn_norm"],
                    merged["pn_extra"], merged["pn_colors"],
                    merged["circuit_meta"], merged["circuit_colors"],
                    merged["circuit_border_colors"],
                    merged["tri_mat_id"], merged["pn_mat_id"],
                    int(refraction_flag),
                    int(has_tri), int(has_pn), int(has_bez),
                    int(time_start), int(width), int(height),
                    int(tile_start),
                    rs_acc, rs_sca, rs_int,
                    rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf,
                    rs_pix, rs_hit, rs_key, rs_eprim, pix_accum)
            shade_all = compactor.select(
                rs_int, ST_SHADE, scan_pool=True)
            if shade_all.numel():
                if shadow_flag:
                    wf_shadow_event(
                        shade_all, int(shade_all.numel()),
                        tri_bvh.blocks, tri_bvh.node_miss,
                        tri_bvh.leaf_prim, tri_bvh.leaf_tspan,
                        int(tri_bvh.first_leaf),
                        merged["tri_pos"], merged["tri_colors"],
                        merged["tri_uvs"], merged["tri_tex_meta"],
                        merged["textures"],
                        int(merged["num_colored_triangles"]),
                        pn_bvh.blocks, pn_bvh.node_miss,
                        pn_bvh.leaf_prim, pn_bvh.leaf_tspan,
                        int(pn_bvh.first_leaf),
                        merged["pn_ctrl"], merged["pn_obb"],
                        merged["pn_colors"],
                        bez_bvh.blocks, bez_bvh.node_miss,
                        bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
                        int(bez_bvh.first_leaf),
                        merged["circuit_meta"],
                        merged["circuit_colors"],
                        merged["circuit_border_colors"],
                        merged["edges_2d"], merged["edge_accel"],
                        pixel_world_scale,
                        float(layer_offset_triangles),
                        float(layer_offset_pn),
                        int(has_tri), int(has_pn), int(has_bez),
                        light_pos, int(num_lights),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_sca, rs_hit, rs_pix, rs_vis)
                for key_val, fn, sc, mat in buckets:
                    bidx = compactor.select(
                        rs_int, ST_SHADE, scan_pool=True,
                        rs_key=rs_key, desired_key=key_val)
                    cnt = int(bidx.numel())
                    if cnt == 0:
                        continue
                    wf_shade_event(
                        bidx, cnt,
                        mat, light_pos, light_col,
                        int(num_lights),
                        fn, sc, int(shadow_flag),
                        int(refraction_flag),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                        rs_hit, rs_eprim, rs_pix, pix_accum,
                        rs_alloc, rs_vis)
            it += 1

    _run_wavefront_tiles(
        memory, out, n=n, width=width, height=height,
        time_start=time_start, transparent=transparent,
        aa_level=aa_level, pool_ratio=pool_ratio,
        primary_per_tile=primary_per_tile,
        cam_origin=cam_origin, screen_point=screen_point,
        pixel_basis_x=pixel_basis_x, pixel_basis_y=pixel_basis_y,
        half_screen_w=half_screen_w, half_screen_h=half_screen_h,
        max_bounces=max_bounces, near_clip=0.0,
        run_tile=run_tile,
        # rs_hit(16 f32) + rs_key + rs_eprim (+ rs_vis with shadows) per slot.
        auto_extra_slot_bytes=16 * 4 + 4 + 4 + (4 if shadow_flag else 0),
        # Compactor output-count word, plus the one-word rs_vis placeholder
        # when shadow visibility is not a per-slot array.
        auto_fixed_bytes=(torch.int32.itemsize
                          * (1 if shadow_flag else 2)))


_originals = {}


def is_ray_tracing_enabled():
    """Vestigial: always False. The ray-traced primitive classes are now the
    engine's only renderer (``RENDERER_SETTINGS`` binds them by default), and
    the ``enable_ray_tracing`` toggle that used to populate ``_originals`` was
    removed with the rasterizer. Kept only because ``post_processing.bloom``
    probes for it defensively."""
    return bool(_originals)
