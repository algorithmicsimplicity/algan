"""Ray traced drop-in replacements for Algan's rasterized render primitives.

``RayTracedTrianglePrimitive`` and ``RayTracedBezierCircuitPrimitive`` subclass
the rasterized primitives to keep their construction and batching, but render
through a self-contained ray tracing pipeline. ``RayTracedPNTrianglePrimitive``
additionally renders each triangle as a curved point-normal (PN) patch -- a
quadratic Bezier triangle bent to match the vertex normals (enable with
``enable_ray_tracing(pn_triangles=True)``):

* ``project_to_screen`` shades vertices and packs geometry + per-frame bounds
  for the whole batch of frames (the spatio-temporal BVH of ``stbvh.py``).
* ``render`` traces one ray per (frame, pixel) with the unified Taichi kernel
  (``ray_trace_taichi.py``), which alpha-blends every surface it encounters
  -- including mirror bounces -- directly into a fixed
  ``[num_frames, num_pixels, channels]`` output buffer. Memory use is
  independent of depth complexity and bounce count, and there is no fragment
  buffer or sorting pass. The Monte Carlo kernels flatten the parallel loop
  further to one thread per (frame, pixel, sample) path, accumulating into a
  float32 per-pixel buffer with atomic adds.

Mirrors: give a mob a reflectivity with :func:`set_reflectivity` (before
spawning); the value is per-vertex, animatable, and bounces up to
``MAX_BOUNCES`` times.

Call :func:`enable_ray_tracing` *before constructing any mobs* to make new
mobs render through this pipeline.
"""
from __future__ import annotations

import gc
import os


import torch

from algan.rendering.post_processing.post_process import post_process_frames
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing.settings import _scene_has_user_pipeline
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    KBUF,
    MAX_SURFACES_PER_RAY, path_trace_scene_stbvh, finalize_samples,
)
from algan.rendering.raytracing.scene_builder import _merge_scene, _downsample_background, _pack_lights, \
    _prefill_background
from algan.rendering.raytracing.settings import _get_tonemap_t_val, REFRACT_SPLIT_SLOTS, WAVEFRONT_TILE_RAYS, \
    TONEMAP_EXPOSURE, SAMPLES_PER_PIXEL, GATE_EMPTY_TRAVERSALS, SHADOWS, FRAGMENT_SHADING, \
    is_post_process_tonemap_enabled, MAX_BOUNCES, INDIRECT_BOUNCE_STRENGTH

from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.rendering.raytracing.shading_taichi import MAT_W, _USER_PIPELINE_BASE
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _pixel_bases
# ``build_frag_pipelines`` is imported lazily in the render dispatch to avoid a
# module-load import cycle (fragment_shaders -> shading_taichi -> raytracing
# package __init__ -> primitives).
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    wf_composite,
    wf_composite_aa,
    wf_composite_accum,
    wf_composite_accum_aa,
    wf_finalize_aa,
    wavefront_generate_rays,
    wavefront_shade,
    wavefront_traverse,
    wavefront_shadow
)
from algan.utils.memory_utils import InsufficientMemoryException


def _alloc_wavefront_state(memory, tn, sca_width):
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
    return (
        memory.get_tensor((tn, 3), f32),          # rs_ro
        memory.get_tensor((tn, 3), f32),          # rs_rd
        memory.get_tensor((tn, 4), f32),          # rs_acc
        memory.get_tensor((tn, sca_width), f32),  # rs_sca (4 tri / 5 general)
        # rs_int: 0 bounces_left, 1 processed, 2 status, 3 num_hits, 4 drained
        # (column 4 is used only by the sorted-material path; the classic
        # kernels index columns 0-3 and never read it).
        memory.get_tensor((tn, 5), i32),          # rs_int
        memory.get_tensor((tn, KBUF), f32),       # rs_kt
        memory.get_tensor((tn, KBUF), f32),       # rs_kl
        memory.get_tensor((tn, KBUF), f32),       # rs_ka
        memory.get_tensor((tn, KBUF), f32),       # rs_kb
        memory.get_tensor((tn, KBUF), i32),       # rs_kp
        memory.get_tensor((tn, KBUF), i32),       # rs_kf
    )


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
    merged = _merge_scene(primitives)
    aa = max(1, int(anti_alias_level))
    # Refraction is only implemented by the general wavefront tracer, so a
    # deterministic batch that contains a refractive surface is routed there
    # regardless of USE_WAVEFRONT (the megakernel / Monte Carlo paths ignore the
    # refractive index). Computed before the AA strategy because it, like
    # USE_WAVEFRONT, forces the super-sampled (non-in-place) AA path.
    refractive_det = (bool(merged.get("has_refractive"))
                      and int(SAMPLES_PER_PIXEL) <= 1)

    # Anti-aliasing strategy. All deterministic renderers (megakernels and
    # wavefront) average ``aa^2`` jittered sub-pixel rays *in place* at the
    # output resolution, so the frame buffer stays ``screen_width x
    # screen_height`` regardless of ``aa`` (aa^2× less render memory than
    # super-sampling). The wavefront path runs the full gen→traverse→shade→
    # compact→composite pipeline once per sub-pixel sample, accumulating into
    # a float buffer and averaging at the end.
    inplace_aa = False
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
    device = COMPUTING_DEFAULTS.render_device
    num_frames = merged["num_frames"]

    cam_origin = _expand_frames(_flat_frames(ray_origin, (3,)),
                                num_frames).contiguous()
    sp = _expand_frames(_flat_frames(screen_point, (3,)),
                        num_frames).contiguous()
    sb = _expand_frames(_flat_frames(screen_basis, (3, 3)), num_frames)
    pbx, pby = _pixel_bases(sb)
    # World units per screen pixel per unit distance (for border widths). Border
    # widths are authored in *anti-aliased* pixels (see BezierCircuit), so this
    # always uses the super-sampled height (screen_height * aa), whether or not
    # the frame buffer itself is super-sampled.
    b1_norm = sb[:, 1].norm(p=2, dim=-1)
    screen_dist = (sp - cam_origin).norm(p=2, dim=-1)
    pixel_world_scale = (
        2.0 / (screen_height * aa * b1_norm * screen_dist).clamp_min(1e-12)
    ).contiguous()

    # In-place AA samples the background once per output pixel, so an
    # animated/image background that arrived super-sampled must be averaged
    # down to the output resolution first (solid colors are resolution-free).
    if aa > 1:
        background_color = _downsample_background(
            background_color, aa, time_end - time_start,
            screen_height, screen_width)

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
    det_frag = ((bool(FRAGMENT_SHADING) or det_shadows or scene_has_frag_pipeline)
                and samples <= 1)
    frag_flag = 1 if det_frag else 0
    shadow_flag = 1 if det_shadows else 0
    # Composed custom fragment-shader pipelines injected into the shade kernel as
    # a flat ti.template() tuple; empty () keeps the built-in / vertex-shaded
    # kernel specialization unchanged (see shading_taichi._run_frag_pipeline).
    if det_frag:
        from algan.rendering.shaders.fragment_shaders import build_frag_pipelines
        frag_pipelines = build_frag_pipelines()
    else:
        frag_pipelines = ()
    # Refraction (general wavefront only; see refractive_det above).
    refraction_flag = 1 if refractive_det else 0
    if det_frag:
        light_pos, light_col, num_lights = _pack_lights(
            light_sources, num_frames, device)
    elif samples > 1:
        light_pos = light_col = None
        num_lights = 0
    else:
        # Deterministic, fragment shading off: tiny placeholders for the
        # (compiled-out) material/light kernel args.
        light_pos = torch.zeros((1, 1, 3), device=device)
        light_col = torch.zeros((1, 1, 3), device=device)
        num_lights = 0

    def render_chunk(start, end):
        # The Monte Carlo kernels launch one thread per (frame, pixel,
        # sample) path; keep the flattened index within int32 range. (The
        # deterministic kernels loop the aa^2 sub-pixels serially per pixel, so
        # only the Monte Carlo path multiplies the thread count by the samples.)
        if (samples > 1 and
                (end - start) * width * height * samples_eff >= 1 << 31):
            print(f'Render OOM, splitting {start}:{end}')
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
                tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
                merged["tri_colors"], merged["tri_uvs"], merged["tri_tex_meta"],
                merged["textures"], int(merged["num_colored_triangles"]),
                pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan, pn_bvh.first_leaf,
                merged["pn_ctrl"], merged["pn_norm"], merged["pn_extra"],
                merged["pn_colors"],
                bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan, bez_bvh.first_leaf,
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"], merged["edges_2d"],
                merged["edge_offsets"],
                cam_origin, sp, pbx, pby, pixel_world_scale,
                int(start), int(end), int(width), int(height),
                float(width // 2), float(height // 2),
                layer_offset_triangles, layer_offset_pn, int(MAX_BOUNCES),
                1 if transparent_background else 0)
            if samples > 1:
                path_trace_scene_stbvh(*shared_args, samples_eff,
                                       float(INDIRECT_BOUNCE_STRENGTH),
                                       merged["pn_obb"], out, accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 t_val, float(TONEMAP_EXPOSURE),
                                 accum, out)
            else:
                raytrace_render_wavefront(
                    tri_bvh, pn_bvh, bez_bvh, merged,
                    cam_origin, sp, pbx, pby, pixel_world_scale,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, layer_offset_pn,
                    has_tri, has_pn, has_bez, int(MAX_BOUNCES),
                    light_pos, light_col, int(num_lights),
                    frag_flag, frag_pipelines, shadow_flag, refraction_flag,
                    1 if transparent_background else 0, memory, out,
                    kernel_aa)
            frames = out.view(end - start, height, width, C_out)
            frames = post_process_frames(memory,
                frames, anti_alias_level=post_aa,
                post_processes=list(post_processes), apply_fxaa=scene.render_settings.fxaa)
            memory.set_pointers(entry_pointers)
            return [frames]
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            print(f'Render OOM, splitting {start}:{end}')
            memory.set_pointers(entry_pointers)
            # Release the failed allocation (e.g. the wavefront's large per-ray
            # state) so it doesn't fragment/block the smaller retry.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


def raytrace_render_wavefront(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, frag_flag, frag_pipelines, shadow_flag,
        refraction_flag, transparent, memory, out, aa_level=1):
    """Wavefront orchestration for the general (triangle + PN + bezier) case:
    stage-split over per-ray global state, with PyTorch ray compaction between host iterations. State carries a
    5th scalar (base_dist) for bezier border widths across bounces.

    ``frag_flag``/``shadow_flag`` select the deterministic per-fragment shading
    and binary hard-shadow paths (compile-time templates of the shade kernel,
    matching ``render_scene_stbvh``); ``light_pos``/``light_col`` feed both.

    ``refraction_flag`` enables simultaneous reflection + refraction (glass): the
    shade kernel SPLITS such a ray, continuing the reflected branch in place and
    spawning the refracted branch into a free pool slot. The pool is therefore
    over-allocated by ``split_k`` (only when refraction is on) -- it holds
    ``primary_per_tile`` one-per-pixel rays plus spare slots for split branches,
    at fixed total memory (fewer pixels per tile instead of bigger per-ray
    state). Each ray commits into a shared per-pixel accumulator (``pix_accum``)
    on termination, so a pixel's reflected and refracted branches sum.

    When fragment shading is active, ``settings.WAVEFRONT_SORT_MATERIALS``
    selects the shade architecture: the Cycles-style *sorted* pipeline (rays
    suspended at their material events, bucketed by (geometry type, material
    pipeline id) and shaded by dedicated per-material kernels -- see
    ``wavefront_sorted_kernels_taichi``) or the monolithic ``wavefront_shade``
    kernel below. The default ``"auto"`` sorts only when the scene *requires*
    it (a pipeline with a custom scatter func, which the monolith cannot run):
    on the built-in analytic materials the sorted path renders byte-identical
    but slower, since the monolith drains up to KBUF hits per launch while
    sorting pays per-event kernel round trips and host syncs. The
    vertex-shaded path below is unaffected either way.
    """
    from algan.rendering.raytracing import settings as rt_settings
    sort_mode = rt_settings.WAVEFRONT_SORT_MATERIALS
    use_sorted = bool(frag_flag) and (
        sort_mode is True
        or (sort_mode == "auto" and _scene_has_custom_scatter(merged)))
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
    t_val = _get_tonemap_t_val()
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    # Pool over-allocation for ray splitting. Only glass (reflective+refractive)
    # surfaces split, so spare slots are reserved only when refraction is on; the
    # non-refractive path keeps split_k == 1 (one slot per pixel, as before).
    split_k = REFRACT_SPLIT_SLOTS if refraction_flag else 1
    primary_per_tile = max(1, WAVEFRONT_TILE_RAYS // split_k)

    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5

            # Ray-offset screen tiling (like render_triangles_wavefront):
            # process the chunk's pixels in bounded tiles, so per-tile state
            # stays bounded *regardless of frame size* -- a single UHD frame
            # just splits into several tiles. ``tile_start`` is the first
            # pixel's global ray index; the pool slot r (< primary count)
            # renders pixel ``tile_start + r``. rs_sca has a 5th column
            # (base_dist). State is pool-allocated and freed after every tile.
            for tile_start in range(0, n, primary_per_tile):
                tn_primary = min(primary_per_tile, n - tile_start)
                pool = tn_primary * split_k
                state_ptrs = memory.get_pointers()
                (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                 rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = \
                    _alloc_wavefront_state(memory, pool, 5)
                rs_pix = memory.get_tensor((pool,), i32)
                pix_accum = memory.get_tensor((tn_primary, 5), f32)
                # Per-pixel spare-slot counter (zeroed by wf_gen_general): a
                # split ray bumps rs_used[its pixel], so distinct pixels touch
                # distinct addresses -- no single global atomic to serialise on.
                rs_used = memory.get_tensor((tn_primary,), i32)
                # Packed per-ray shadow visibility bits (deferred shadows only);
                # a 1-element placeholder otherwise (the reader compiles out).
                rs_vis = memory.get_tensor(
                    (1,), i32)

                wavefront_generate_rays(
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    int(time_start), int(width), int(height),
                    float(half_screen_w), float(half_screen_h),
                    int(max_bounces),
                    int(tile_start), int(tn_primary), float(jx), float(jy),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                    rs_pix, pix_accum, rs_used)

                active = torch.arange(tn_primary, dtype=i32, device=device)
                it = 0
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    wavefront_traverse(
                        active, na,
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                        merged["tri_pos"],
                        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                        pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                        merged["pn_ctrl"],
                        merged["pn_obb"],
                        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                        bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                        merged["circuit_meta"],
                        merged["edges_2d"], merged["edge_offsets"],
                        pixel_world_scale,
                        float(layer_offset_triangles), float(layer_offset_pn),
                        int(has_tri), int(has_pn), int(has_bez),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix)
                    wavefront_shade(
                        active, na,
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                        merged["tri_pos"], merged["tri_norm"],
                        merged["tri_extra"],
                        merged["tri_colors"], merged["tri_uvs"],
                        merged["tri_tex_meta"],
                        merged["textures"],
                        int(merged["num_colored_triangles"]),
                        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                        pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                        merged["pn_ctrl"], merged["pn_norm"],
                        merged["pn_extra"],
                        merged["pn_colors"], merged["pn_obb"],
                        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                        bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                        merged["circuit_meta"], merged["circuit_colors"],
                        merged["circuit_border_colors"],
                        merged["edges_2d"], merged["edge_offsets"],
                        pixel_world_scale,
                        float(layer_offset_triangles), float(layer_offset_pn),
                        int(frag_flag), frag_pipelines, int(shadow_flag),
                        int(refraction_flag),
                        int(has_tri), int(has_pn), int(has_bez),
                        0,
                        merged["tri_mat_id"], merged["tri_mat"],
                        merged["pn_mat_id"], merged["pn_mat"],
                        light_pos, light_col, int(num_lights),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf,
                        rs_pix, pix_accum, rs_used, rs_vis)
                    active = (rs_int[:, 2] == 0).nonzero(
                        as_tuple=True)[0].to(i32)
                    it += 1

                if do_aa:
                    wf_composite_accum_aa(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, out, aa_accum)
                else:
                    wf_composite_accum(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, t_val, float(TONEMAP_EXPOSURE), out)
                # Release this tile's state back to the pool before the next
                # tile.
                memory.set_pointers(state_ptrs)

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val, float(TONEMAP_EXPOSURE), aa_accum, out)


def _scene_has_custom_scatter(merged):
    """True if any merged primitive's material pipeline carries a custom
    scatter func (user-controlled ray bouncing). Such a scene must render
    through the sorted-material pipeline -- the monolithic shade kernel has no
    way to run a per-pipeline scatter. Cheap: exits on the tensor-max
    user-pipeline pre-check for the (overwhelmingly common) all-built-in
    scene."""
    if not _scene_has_user_pipeline(merged):
        return False
    from algan.rendering.shaders.fragment_shaders import build_frag_scatters
    scatters = build_frag_scatters()
    for key in ("tri_mat_id", "pn_mat_id"):
        arr = merged.get(key)
        if arr is None or not arr.numel():
            continue
        for pid in torch.unique(arr).tolist():
            i = int(pid) - _USER_PIPELINE_BASE
            if 0 <= i < len(scatters) and scatters[i] is not None:
                return True
    return False


def _raytrace_render_wavefront_sorted(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, frag_pipelines, shadow_flag,
        refraction_flag, transparent, memory, out, aa_level=1):
    """Cycles-style sorted-material orchestration of the fragment-shading
    wavefront (see ``wavefront_sorted_kernels_taichi`` for the kernel split).

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
        ST_DONE, ST_PEEL, ST_SHADE, ST_TRAVERSE,
        default_scatter, wf_peel, wf_shade_event, wf_shadow_event)

    device = out.device
    t_val = _get_tonemap_t_val()
    i32 = torch.int32
    f32 = torch.float32
    n = (time_end - time_start) * width * height
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

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
        for pid in torch.unique(merged["tri_mat_id"]).tolist():
            fn, sc = _resolve(int(pid))
            has_custom_scatter |= sc is not default_scatter
            buckets.append(((1 << 8) | int(pid), fn, sc, merged["tri_mat"]))
    if merged["num_pn"] > 0:
        for pid in torch.unique(merged["pn_mat_id"]).tolist():
            fn, sc = _resolve(int(pid))
            has_custom_scatter |= sc is not default_scatter
            buckets.append(((2 << 8) | int(pid), fn, sc, merged["pn_mat"]))
    # A custom scatter may spawn transmitted branches, which need the glass
    # split pool (and the peel's IOR sampling) even in a scene with no
    # refractive surface.
    refraction_flag = 1 if (refraction_flag or has_custom_scatter) else 0
    bucket_map = {key: (fn, sc, mat) for key, fn, sc, mat in buckets}

    # Worst case: every one of MAX_SURFACES_PER_RAY hits is a material event
    # (one peel+shade pass each) plus a traverse per K-buffer refill / bounce.
    max_iters = (MAX_SURFACES_PER_RAY + MAX_SURFACES_PER_RAY // KBUF
                 + max_bounces * 2 + 8)

    split_k = REFRACT_SPLIT_SLOTS if refraction_flag else 1
    # The sorted path carries ~1.5x the classic per-ray state (the event
    # record + keys), so tiles hold fewer rays for the same memory envelope.
    primary_per_tile = max(1, (WAVEFRONT_TILE_RAYS * 2) // (3 * split_k))

    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5

            for tile_start in range(0, n, primary_per_tile):
                tn_primary = min(primary_per_tile, n - tile_start)
                pool = tn_primary * split_k
                state_ptrs = memory.get_pointers()
                (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                 rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = \
                    _alloc_wavefront_state(memory, pool, 5)
                rs_pix = memory.get_tensor((pool,), i32)
                pix_accum = memory.get_tensor((tn_primary, 5), f32)
                rs_used = memory.get_tensor((tn_primary,), i32)
                # Event state: hit record, sort key, event primitive index and
                # per-event shadow visibility bits (placeholder when unused).
                rs_hit = memory.get_tensor((pool, 15), f32)
                rs_key = memory.get_tensor((pool,), i32)
                rs_eprim = memory.get_tensor((pool,), i32)
                rs_vis = memory.get_tensor((pool,) if shadow_flag else (1,),
                                           i32)

                wavefront_generate_rays(
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    int(time_start), int(width), int(height),
                    float(half_screen_w), float(half_screen_h),
                    int(max_bounces),
                    int(tile_start), int(tn_primary), float(jx), float(jy),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                    rs_pix, pix_accum, rs_used)
                # The drained counter (rs_int col 4, pool garbage after
                # allocation) must be 0 for every ray entering ST_TRAVERSE;
                # the kernels maintain that invariant from here on.
                rs_int[:, 4].zero_()

                it = 0
                while it < max_iters:
                    # At the top of an iteration every ray is DONE, TRAVERSE
                    # or PEEL (pending SHADE events never survive their
                    # discovery iteration), so two index builds decide both
                    # what to launch and when to stop.
                    status = rs_int[:, 2]
                    trav = (status == ST_TRAVERSE).nonzero(
                        as_tuple=True)[0].to(i32)
                    peel_extra = (status == ST_PEEL).nonzero(
                        as_tuple=True)[0].to(i32)
                    if trav.numel() == 0 and peel_extra.numel() == 0:
                        break
                    if trav.numel():
                        wavefront_traverse(
                            trav, int(trav.numel()),
                            tri_bvh.nodes, tri_bvh.node_miss,
                            tri_bvh.leaf_prim, tri_bvh.leaf_tspan,
                            int(tri_bvh.first_leaf),
                            merged["tri_pos"],
                            pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                            pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                            merged["pn_ctrl"],
                            merged["pn_obb"],
                            bez_bvh.nodes, bez_bvh.node_miss,
                            bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
                            int(bez_bvh.first_leaf),
                            merged["circuit_meta"],
                            merged["edges_2d"], merged["edge_offsets"],
                            pixel_world_scale,
                            float(layer_offset_triangles),
                            float(layer_offset_pn),
                            int(has_tri), int(has_pn), int(has_bez),
                            int(time_start), int(width), int(height),
                            int(tile_start),
                            rs_ro, rs_rd, rs_sca, rs_int,
                            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix)
                    if peel_extra.numel():
                        peel_idx = (torch.cat((trav, peel_extra))
                                    if trav.numel() else peel_extra)
                    else:
                        peel_idx = trav
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
                    # Sort the pending events by material key once: the
                    # buckets become contiguous slices of one index array
                    # (coalesced kernel reads) and the whole dispatch costs a
                    # single host sync instead of one nonzero per bucket.
                    shade_idx = (rs_int[:, 2] == ST_SHADE).nonzero(
                        as_tuple=True)[0]
                    if shade_idx.numel():
                        keys_shade = rs_key[shade_idx]
                        sorted_keys, order = torch.sort(keys_shade)
                        shade_sorted = shade_idx[order].to(i32)
                        uniq, counts = torch.unique_consecutive(
                            sorted_keys, return_counts=True)
                        bucket_sizes = torch.stack(
                            (uniq.long(), counts)).tolist()
                        if shadow_flag:
                            shade_all = shade_sorted
                            wf_shadow_event(
                                shade_all, int(shade_all.numel()),
                                tri_bvh.nodes, tri_bvh.node_miss,
                                tri_bvh.leaf_prim, tri_bvh.leaf_tspan,
                                int(tri_bvh.first_leaf),
                                merged["tri_pos"], merged["tri_colors"],
                                merged["tri_uvs"], merged["tri_tex_meta"],
                                merged["textures"],
                                int(merged["num_colored_triangles"]),
                                pn_bvh.nodes, pn_bvh.node_miss,
                                pn_bvh.leaf_prim, pn_bvh.leaf_tspan,
                                int(pn_bvh.first_leaf),
                                merged["pn_ctrl"], merged["pn_obb"],
                                merged["pn_colors"],
                                bez_bvh.nodes, bez_bvh.node_miss,
                                bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
                                int(bez_bvh.first_leaf),
                                merged["circuit_meta"],
                                merged["circuit_colors"],
                                merged["circuit_border_colors"],
                                merged["edges_2d"], merged["edge_offsets"],
                                pixel_world_scale,
                                float(layer_offset_triangles),
                                float(layer_offset_pn),
                                int(has_tri), int(has_pn), int(has_bez),
                                light_pos, int(num_lights),
                                int(time_start), int(width), int(height),
                                int(tile_start),
                                rs_ro, rs_rd, rs_sca, rs_hit, rs_pix, rs_vis)
                        off = 0
                        for key_val, cnt in zip(*bucket_sizes):
                            fn, sc, mat = bucket_map[int(key_val)]
                            cnt = int(cnt)
                            bidx = shade_sorted[off:off + cnt]
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
                                rs_used, rs_vis)
                            off += cnt
                    it += 1

                if do_aa:
                    wf_composite_accum_aa(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, out, aa_accum)
                else:
                    wf_composite_accum(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, t_val, float(TONEMAP_EXPOSURE), out)
                memory.set_pointers(state_ptrs)

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val,
                       float(TONEMAP_EXPOSURE), aa_accum, out)


_originals = {}


def is_ray_tracing_enabled():
    """True if the ray traced primitive classes are currently active (i.e.
    :func:`enable_ray_tracing` has been called and not yet disabled)."""
    return bool(_originals)
