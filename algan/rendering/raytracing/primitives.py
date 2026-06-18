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

import taichi as ti
import torch
import torch.nn.functional as F

from algan import CudaStream, csync
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
    batch_arange,
)
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.rendering.raytracing.pn_patch import (
    pn_control_points,
    pn_patch_coefficients,
)
from algan.rendering.raytracing.ray_trace_taichi import (
    KBUF,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    finalize_samples,
    path_trace_physical_stbvh,
    path_trace_scene_stbvh,
    render_scene_stbvh,
    render_triangles_stbvh,
)
from algan.rendering.raytracing.no_pn_taichi import render_no_pn_stbvh
from algan.rendering.raytracing.wavefront_taichi import (
    wf_composite,
    wf_gen_general,
    wf_gen_triangle,
    wf_shade_general,
    wf_shade_triangle,
    wf_traverse_general,
    wf_traverse_triangle,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO, build_stbvh
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.utils.memory_utils import InsufficientMemoryException, empty_cache
from algan.utils.tensor_utils import broadcast_all, cast_to_tensor, unsquish

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
MAX_BOUNCES = 4
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo path tracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
SAMPLES_PER_PIXEL = 1
# Strength of diffuse indirect bounces in the Monte Carlo renderer: 0 keeps
# surfaces purely (vertex-shader) lit, > 0 scatters paths on diffuse hits
# with throughput ``albedo * strength`` for color bleeding.
INDIRECT_BOUNCE_STRENGTH = 0.0
# Fully physical mode: vertex shading is skipped (colors are raw albedo) and
# the path tracer computes all lighting from the scene's explicit point
# lights (with shadow rays), glow emission and the background environment.
# Requires SAMPLES_PER_PIXEL > 1.
PHYSICAL_LIGHTING = False
# Radiance scale of explicit point lights in physical mode. The default of
# pi makes a white light produce roughly albedo-level Lambertian brightness.
LIGHT_INTENSITY = 3.141592653589793
# Constant ambient term added per diffuse interaction in physical mode.
AMBIENT_LIGHT = 0.0
# When True, the deterministic trace kernel is told which geometry types are
# actually present and skips the per-ray traversal of any type whose tree is
# just the empty placeholder (a launch-uniform branch, no divergence). Set
# False to force all three traversals -- used by the A/B benchmark to measure
# the gain in isolation.
GATE_EMPTY_TRAVERSALS = True
# When True, a batch containing only flat triangles is rendered by the
# specialized ``render_triangles_stbvh`` kernel instead of the general
# three-geometry-type ``render_scene_stbvh``. Output is identical; the
# specialized kernel just carries no PN/bezier code, so it has lower register
# pressure (higher GPU occupancy). Set False to force the general kernel (A/B).
USE_TRIANGLE_ONLY_KERNEL = True
# When True, a batch with NO PN patches but containing bezier circuits (text,
# 2D shapes), optionally mixed with flat triangles, is rendered by the
# specialized ``render_no_pn_stbvh`` kernel instead of the general
# ``render_scene_stbvh``. The general kernel always compiles the PN Matrix Pencil
# solver into its call graph (its dominant register cost); the no-PN kernel
# omits it, so a no-PN scene runs at higher occupancy. Output is byte-identical
# (the omitted PN paths are inert when no PN patches are present). The pure
# triangle-only case is handled by the even-leaner kernel above; this catches
# the remaining default (pn_triangles=False) scenes. Env ALGAN_NO_PN=0 forces
# the general kernel (A/B against this specialization).
USE_NO_PN_KERNEL = os.environ.get("ALGAN_NO_PN", "1") == "1"
# When True, ray-traced batches are rendered by the experimental wavefront
# (stage-split) path instead of the single-megakernel path: per-ray state lives
# in global memory and small per-stage kernels (gen, traverse, shade, composite)
# run in a host loop with PyTorch ray compaction between iterations, for higher
# occupancy and less divergence. Output is byte-identical. Covers both the
# triangle-only and the general (PN/bezier/mixed) cases. Off by default (env
# ALGAN_WAVEFRONT=1): measured SLOWER than the megakernels on the GTX 1050
# (global state-I/O cost exceeds the occupancy gain on a bandwidth-limited GPU),
# kept for validation and for higher-bandwidth hardware.
USE_WAVEFRONT = os.environ.get("ALGAN_WAVEFRONT", "0") == "1"


def set_physical_lighting(enabled):
    """Toggle fully physical lighting for the Monte Carlo renderer: vertex
    shading is skipped and illumination comes from the scene's point lights
    (sampled with shadow rays), surface ``glow`` emission, and the background
    environment. Requires ``set_samples_per_pixel`` > 1. Set before
    rendering.
    """
    global PHYSICAL_LIGHTING
    PHYSICAL_LIGHTING = bool(enabled)


def set_light_intensity(intensity):
    """Radiance scale applied to explicit point lights in physical mode."""
    global LIGHT_INTENSITY
    LIGHT_INTENSITY = float(intensity)


def set_ambient_light(intensity):
    """Constant ambient lighting term used in physical mode."""
    global AMBIENT_LIGHT
    AMBIENT_LIGHT = float(intensity)


def set_samples_per_pixel(samples):
    """Set how many rays are averaged per pixel. 1 (the default) uses the
    exact deterministic renderer; larger values enable Monte Carlo path
    tracing with that many samples.
    """
    global SAMPLES_PER_PIXEL
    SAMPLES_PER_PIXEL = max(1, int(samples))


def set_indirect_bounce_strength(strength):
    """Set the diffuse indirect lighting strength of the Monte Carlo
    renderer (0 disables diffuse bounces).
    """
    global INDIRECT_BOUNCE_STRENGTH
    INDIRECT_BOUNCE_STRENGTH = float(strength)


def _set_surface_param(mob, name, value):
    value = cast_to_tensor(float(value)).view(1, 1)
    for descendant in reversed(mob.get_descendants()):
        setattr(descendant, name, value)
        names = list(getattr(descendant, "shader_specific_param_names", []))
        if name not in names:
            names.append(name)
        descendant.shader_specific_param_names = names
    return mob


def set_reflectivity(mob, reflectivity):
    """Make a mob a mirror under the ray traced renderer.

    Attaches a (static) ``reflectivity`` value (0 = matte, 1 = perfect
    mirror) to the mob and its descendants, exposed to the renderer as a
    shader parameter. Call before the mob is spawned; only the ray traced
    pipeline uses it (the rasterizer's shaders would reject the extra
    parameter).
    """
    return _set_surface_param(mob, "reflectivity", reflectivity)


def set_roughness(mob, roughness):
    """Set the glossiness of a mirror mob: 0 is a sharp mirror, larger
    values blur its reflections. Only used by the Monte Carlo renderer
    (``set_samples_per_pixel`` > 1); the deterministic renderer reflects
    sharply. Call before the mob is spawned.
    """
    return _set_surface_param(mob, "roughness", roughness)


def _flat_frames(x, last_dims):
    """Collapse camera tensors like [T, 1, 1, 3] to [T, *last_dims]."""
    return x.reshape(x.shape[0], *last_dims).float()


def _expand_frames(x, num_frames):
    if x.shape[0] == num_frames:
        return x
    return x.expand(num_frames, *x.shape[1:])


def _pixel_bases(screen_basis):
    """Per-frame world-space steps corresponding to one unit of normalized
    screen coordinate, matching the camera's projection exactly.

    The camera projects a world point by intersecting its view ray with the
    plane ``normal = basis_row_2`` through the screen center, then taking raw
    dot products with ``basis_row_0/1``. The screen basis is rotation x
    non-uniform scale, so under camera rotation its rows are *not* mutually
    orthogonal (``row0 . row2 != 0``) -- the projection is anisotropic and
    changes with orientation. The exact inverse image of screen coordinate
    (u, v) is ``screen_point + u * d0 + v * d1`` where ``d0, d1`` are the
    first two columns of the inverse basis matrix (the reciprocal basis:
    ``d_i . row_j = delta_ij``), which both lies on the projection plane and
    reproduces the dot products.
    """
    eye = torch.eye(3, device=screen_basis.device).unsqueeze(0) * 1e-12
    dual = torch.linalg.inv(screen_basis + eye)
    return dual[:, :, 0].contiguous(), dual[:, :, 1].contiguous()


def _unify_time(tensors, error_context):
    """Expand a set of tensors whose leading (time) dims are each 1 or T to a
    common T. Returns the expanded tensors and T.
    """
    T = max(t.shape[0] for t in tensors)
    for t in tensors:
        if t.shape[0] not in (1, T):
            raise ValueError(
                f"{error_context}: incompatible frame counts "
                f"{[tuple(t.shape) for t in tensors]}")
    return [_expand_frames(t, T) for t in tensors], T


def _cat_collections(tensors, dim, error_context):
    """Concatenate per-collection tensors along ``dim``, broadcasting their
    (possibly different) time dimensions to a common length first. A single
    collection is passed through without copying (the kernel indexes each
    array's time dimension independently, so no expansion is needed).
    """
    if len(tensors) == 1:
        return tensors[0]
    tensors, _ = _unify_time(tensors, error_context)
    return torch.cat(tensors, dim).contiguous()


class RayTracedTrianglePrimitive(TrianglePrimitive):
    """Triangle batch rendered by ray tracing a spatio-temporal BVH."""

    stbvh_tightness = 2.0

    # Per-vertex surface parameters consumed by the trace kernels rather
    # than by a shader; popped from the shader kwargs (see set_reflectivity
    # and set_roughness).
    _surface_params = ("reflectivity", "roughness")

    def __init__(self, corners=None, colors=None, opacity=1, normals=None,
                 perimeter_points=None, reverse_perimeter=False,
                 triangle_collection=None, glow=0, shader=None,
                 **shader_kwargs):
        if triangle_collection is not None:
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader,
                             **shader_kwargs)
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
                        v = torch.zeros_like(triangle.colors[:1, ..., :1])
                    v = broadcast_all(
                        [triangle.corners[:1], triangle.colors[:1],
                         triangle.normals[:1], v], ignored_dims=[-1]
                    )[-1][..., :1]
                    values.append(v)
                setattr(self, name, unsquish(torch.cat(values, 1), -2, 3
                                             ).to(COMPUTING_DEFAULTS.render_device))
        else:
            params = {name: shader_kwargs.pop(name, None)
                      for name in self._surface_params}
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader,
                             **shader_kwargs)
            for name, value in params.items():
                if value is None:
                    setattr(self, name,
                            torch.zeros_like(self.colors[:1, ..., :1]))
                else:
                    value = cast_to_tensor(value).to(self.colors.device)
                    setattr(self, name, broadcast_all(
                        [self.corners[:1], self.colors[:1], value],
                        ignored_dims=[-1])[-1][..., :1])

    def _shade_vertex_colors(self, camera, light_sources):
        """Vertex shading, identical to the rasterized pipeline. Skipped in
        physical mode, where colors are raw albedo and the path tracer
        computes all lighting itself.
        """
        d = -1
        if not PHYSICAL_LIGHTING and getattr(self, "shader", None) is not None:
            for light_source in light_sources:
                with self.memory.temp():
                    self.colors[..., :d] = self.shader(
                        self.memory,
                        self.corners,
                        self.normals,
                        self.colors[..., :d],
                        camera.ray_origin,
                        light_source.origin,
                        light_source.light_color,
                        1,
                        1,
                        *self.shader_param_values,
                    )

    def _pack_surface_extra(self, error_context):
        """Per-corner (reflectivity, roughness) pairs [Te, N, 6]."""
        (reflectivity_e, roughness_e), _ = _unify_time(
            [self.reflectivity.float(), self.roughness.float()],
            error_context)
        return torch.cat((reflectivity_e, roughness_e), -1).reshape(
            reflectivity_e.shape[0], reflectivity_e.shape[1], 6).contiguous()

    def _pack_frame_visibility(self, lo, hi, colors, error_context):
        """Per-frame bounds; frames where a primitive is fully transparent
        are marked empty so they never enter the BVH. Fully opaque frames
        are flagged so the trace kernel can prune hits behind them while
        gathering.
        """
        alpha = colors.opacity.squeeze(-1)
        visible = alpha.amax(-1) > MIN_ALPHA
        opaque = alpha.amin(-1) >= 1.0 - 1e-6
        (lo, hi, visible, opaque), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), opaque.unsqueeze(-1)],
            error_context)
        visible = visible.squeeze(-1)
        self._rt_frame_opaque = opaque.squeeze(-1).contiguous()
        self._rt_frame_lo = torch.where(
            visible.unsqueeze(-1), lo,
            torch.tensor(EMPTY_LO, device=lo.device)).contiguous()
        self._rt_frame_hi = torch.where(
            visible.unsqueeze(-1), hi,
            torch.tensor(EMPTY_HI, device=hi.device)).contiguous()

    def _set_frame_buffer_bytes(self, camera):
        """Per-frame buffer bytes: the u8 output, plus the f32 sample
        accumulator in Monte Carlo mode.
        """
        mc = PHYSICAL_LIGHTING or SAMPLES_PER_PIXEL > 1
        self._rt_frame_bytes = int(
            camera.screen_width * camera.screen_height * 5 * 4
            * (2 if mc else 1))

    @csync
    def project_to_screen(self, camera, light_sources):
        with CudaStream():
            self._shade_vertex_colors(camera, light_sources)

            corners = self.corners.float()
            normals = self.normals.float()
            # Hot/cold split, each array with its own (independent) time
            # dimension: positions are touched by every candidate
            # intersection, normals only by hits that bounce or scatter, and
            # reflectivity/roughness (usually static) only by confirmed hits.
            self._rt_tri_pos = corners.reshape(
                corners.shape[0], corners.shape[1], 9).contiguous()
            self._rt_tri_norm = normals.reshape(
                normals.shape[0], normals.shape[1], 9).contiguous()
            self._rt_tri_extra = self._pack_surface_extra(
                "triangle surface params")
            self._rt_tri_colors = self.colors.float().contiguous()
            self._rt_num_frames = camera.ray_origin.shape[0]

            self._pack_frame_visibility(corners.amin(-2), corners.amax(-2),
                                        self._rt_tri_colors,
                                        "triangle bounds/colors")

            # Everything the renderer needs now lives in the packed arrays;
            # release the unpacked geometry to halve resident GPU memory.
            self.corners = self.normals = None
            self.reflectivity = self.roughness = None
            self.colors = None

            self._set_frame_buffer_bytes(camera)

            # Ensure released geometry is actually freed before rendering.
            empty_cache()
        return self

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return render_batch_ray_traced(
            primitives, scene, screen_width, screen_height, time_start,
            time_end, background_color, transparent_background, *args,
            **kwargs)


class RayTracedPNTrianglePrimitive(RayTracedTrianglePrimitive):
    """Curved point-normal (PN) triangle batch: each triangle is rendered as
    the quadratic Bezier (Steiner) triangle whose mid-edge control points
    bend the surface to respect the vertex normals (see
    :mod:`algan.rendering.raytracing.pn_patch`), so coarsely tessellated
    smooth surfaces keep smooth silhouettes. Construction, batching and
    vertex shading are inherited from the flat triangles; only the packed
    geometry differs (monomial patch coefficients instead of corner
    positions) and the trace kernels intersect rays with the curved patch
    (up to four hits per ray). Triangles with zero or face-constant normals
    stay exactly flat, and adjacent patches share boundary curves, so PN
    meshes stay watertight.
    """

    @csync
    def project_to_screen(self, camera, light_sources):
        with CudaStream():
            self._shade_vertex_colors(camera, light_sources)

            corners = self.corners.float()
            normals = self.normals.float()
            # Hot/cold split as for flat triangles, with the patch's
            # monomial coefficients as the hot geometry. corners and
            # normals share a time dimension by construction (the batching
            # constructor broadcasts them together).
            control_points = pn_control_points(corners, normals)
            self._rt_pn_ctrl = pn_patch_coefficients(
                control_points).contiguous()
            self._rt_pn_norm = normals.reshape(
                normals.shape[0], normals.shape[1], 9).contiguous()
            self._rt_pn_extra = self._pack_surface_extra("pn surface params")
            self._rt_pn_colors = self.colors.float().contiguous()
            self._rt_num_frames = camera.ray_origin.shape[0]

            # The patch lies in the convex hull of its control points, so
            # the control net bounds it.
            self._pack_frame_visibility(control_points.amin(-2),
                                        control_points.amax(-2),
                                        self._rt_pn_colors,
                                        "pn bounds/colors")

            self.corners = self.normals = None
            self.reflectivity = self.roughness = None
            self.colors = None

            self._set_frame_buffer_bytes(camera)

            # Ensure released geometry is actually freed before rendering.
            empty_cache()
        return self


def _evaluate_cubic_bezier_batch(p, t):
    """p: [..., 4, 3] control points, t: broadcastable parameter in [0, 1)."""
    mt = 1.0 - t
    return ((mt * mt * mt) * p[..., 0, :]
            + (3.0 * mt * mt * t) * p[..., 1, :]
            + (3.0 * mt * t * t) * p[..., 2, :]
            + (t * t * t) * p[..., 3, :])


class RayTracedBezierCircuitPrimitive(BezierCircuitPrimitive):
    """Planar bezier circuits rendered by ray tracing a spatio-temporal BVH.

    Circuits are sampled into polylines at a screen-space density (matching
    the rasterizer's sampling rule) but expressed in each circuit's own plane
    coordinates; the trace kernel intersects rays with the plane and
    classifies hits by an even-odd crossing test (fill) plus a min distance
    to the polyline (border). Texture-mapped circuits (``ImageMob`` etc.) are
    sampled bilinearly in-kernel from their texture grid.
    """

    stbvh_tightness = 2.0
    max_samples_per_segment = 512

    @csync
    def project_to_screen(self, camera, light_sources):
        with CudaStream():
            corners = self.corners.float().contiguous()  # [Tc, S, 4, 3]
            num_frames = camera.ray_origin.shape[0]
            self._rt_num_frames = num_frames

            device = corners.device
            cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)),
                                   num_frames).to(device)
            sp = _expand_frames(_flat_frames(camera.screen_point, (3,)),
                                num_frames).to(device)
            sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)),
                                num_frames).to(device)

            num_samples = self._compute_samples_per_segment(
                corners, cam_o, sp, sb, camera.screen_height)
            self._build_circuit_geometry(corners, num_samples)
            self._build_frame_bounds(corners, cam_o, sp, sb,
                                     camera.screen_height)

            # The polylines/metadata now carry everything the renderer needs;
            # release the control points to reduce resident GPU memory.
            self.corners = None

            # Per-frame buffer bytes: the u8 output, plus the f32 sample
            # accumulator in Monte Carlo mode.
            mc = PHYSICAL_LIGHTING or SAMPLES_PER_PIXEL > 1
            self._rt_frame_bytes = int(
                camera.screen_width * camera.screen_height * 5 * 4
                * (2 if mc else 1))

            # Ensure released geometry is actually freed before rendering.
            empty_cache()
        return self

    def _compute_samples_per_segment(self, corners, cam_o, sp, sb, screen_h):
        """Choose the polyline density per bezier segment from the maximum
        projected control-net length over the batch (the rasterizer's rule:
        one sample every ``num_pixels_per_sample`` screen pixels).
        """
        device = corners.device
        T = cam_o.shape[0]
        Tc = corners.shape[0]
        S = corners.shape[1]
        net_max = torch.zeros((S,), device=device)
        chunk = max(1, int(2e6 // max(S * 4, 1)))
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            cor = corners if Tc == 1 else corners[s:e]
            cam_c = cam_o[s:e].view(-1, 1, 1, 3)
            sp_c = sp[s:e].view(-1, 1, 1, 3)
            n_c = sb[s:e, 2].view(-1, 1, 1, 3)
            rays = cor - cam_c
            denom = (rays * n_c).sum(-1, keepdim=True)
            t_plane = ((sp_c - cam_c) * n_c).sum(-1, keepdim=True) / denom
            proj = cam_c + t_plane * rays
            rel = proj - sp_c
            pts = torch.stack(((rel * sb[s:e, 0].view(-1, 1, 1, 3)).sum(-1),
                               (rel * sb[s:e, 1].view(-1, 1, 1, 3)).sum(-1)), -1)
            pts = pts.nan_to_num_() * (screen_h // 2)
            net = (pts[..., 1:, :] - pts[..., :-1, :]).norm(p=2, dim=-1).sum(-1)
            net_max = torch.maximum(net_max, net.amax(0))
        return (net_max / self.num_pixels_per_sample).ceil().long().clamp_(
            min=1, max=self.max_samples_per_segment)

    def _build_circuit_geometry(self, corners, num_samples):
        """Sample world-space polylines into per-circuit plane coordinates and
        pack the per-circuit metadata the trace kernel consumes.
        """
        device = corners.device
        S = corners.shape[1]
        num_segments = self.num_segments_per_object.to(device).view(-1).long()
        C = num_segments.shape[0]

        circuit_of_segment = torch.repeat_interleave(
            torch.arange(C, device=device), num_segments)
        vert_circuit = torch.repeat_interleave(circuit_of_segment, num_samples)
        V = int(num_samples.sum())

        t_params = (batch_arange(num_samples)
                    / torch.repeat_interleave(num_samples, num_samples))
        ctrl = torch.repeat_interleave(corners, num_samples, dim=1)
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

        # Absolute polyline index of the first sample of each segment, and of
        # the sample each segment's last sample connects to (closing each
        # subpath through next_segment_inds, exactly like the rasterizer).
        seg_starts = num_samples.cumsum(0) - num_samples
        seg_ends = seg_starts - 1
        seg_ends[0] = V - 1
        seg_ends = torch.roll(seg_ends, -1, 0)
        nsi = self.next_segment_inds.to(device).reshape(
            self.next_segment_inds.shape[0], S).long()
        next_start = seg_starts[nsi]  # [Tn, S]

        (verts_e, centers_e, basis_u_e, basis_v_e, next_start_e), T_geo = _unify_time(
            [verts, centers, basis_u, basis_v, next_start.unsqueeze(-1)],
            "bezier geometry")
        next_start_e = next_start_e.squeeze(-1)

        rel = verts_e - centers_e[:, vert_circuit]
        u = (rel * basis_u_e[:, vert_circuit]).sum(-1)
        v = (rel * basis_v_e[:, vert_circuit]).sum(-1)
        locals_uv = torch.stack((u, v), -1)  # [T_geo, V, 2]
        next_uv = locals_uv.roll(-1, dims=1)
        gather_inds = next_start_e.unsqueeze(-1).expand(T_geo, -1, 2)
        next_uv[:, seg_ends] = torch.gather(locals_uv, 1, gather_inds)
        self._rt_edges = torch.cat((locals_uv, next_uv), -1).float().contiguous()

        samples_per_circuit = torch.zeros((C,), dtype=torch.long, device=device)
        samples_per_circuit.index_add_(0, circuit_of_segment, num_samples)
        edge_offsets = torch.zeros((C + 1,), dtype=torch.long, device=device)
        edge_offsets[1:] = samples_per_circuit.cumsum(0)
        self._rt_edge_offsets = edge_offsets.to(torch.int32).contiguous()
        self._rt_circuit_of_segment = circuit_of_segment

        # Texture-grid transform: maps plane (u, v) displacements to the
        # mob-basis coordinates used by the texture lookup.
        def scaled(basis):
            basis = basis.float()
            return basis / basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-12)

        basis1, basis2 = scaled(self.basis1), scaled(self.basis2)
        border_width = self.border_width.float().reshape(
            self.border_width.shape[0], C)
        grid_w = self.grid_width.float().reshape(self.grid_width.shape[0], C)
        grid_h = self.grid_height.float().reshape(self.grid_height.shape[0], C)
        (centers_m, normals_m, bu_m, bv_m, b1_m, b2_m, bw_m, gw_m, gh_m), Tm = _unify_time(
            [centers, normals, basis_u, basis_v, basis1, basis2,
             border_width.unsqueeze(-1), grid_w.unsqueeze(-1),
             grid_h.unsqueeze(-1)], "bezier metadata")
        filled = torch.full((Tm, C, 1), 1.0 if self.filled else 0.0,
                            device=device)
        tex = torch.stack((
            (b1_m * bu_m).sum(-1), (b1_m * bv_m).sum(-1),
            (b2_m * bu_m).sum(-1), (b2_m * bv_m).sum(-1)), -1).nan_to_num_()
        self._rt_circuit_meta = torch.cat(
            (centers_m, normals_m, bu_m, bv_m, bw_m, filled, gw_m, gh_m, tex),
            -1).contiguous()

        colors = self.colors.float()
        if colors.dim() == 3:  # plain fills: a 1x1 "texture" grid
            colors = colors.unsqueeze(-2)
        self._rt_circuit_colors = colors.contiguous()
        self._rt_circuit_border_colors = self.border_color.float().contiguous()
        self._rt_border_width = border_width

    def _build_frame_bounds(self, corners, cam_o, sp, sb, screen_h):
        """Per-frame circuit AABBs (from control-point hulls, inflated by the
        screen-space border width), with invisible frames marked empty.
        """
        device = corners.device
        C = self._rt_edge_offsets.shape[0] - 1
        circuit_of_segment = self._rt_circuit_of_segment

        seg_lo = corners.amin(-2)
        seg_hi = corners.amax(-2)
        Tb = seg_lo.shape[0]
        idx = circuit_of_segment.view(1, -1, 1).expand(Tb, -1, 3)
        lo = torch.full((Tb, C, 3), EMPTY_LO, device=device).scatter_reduce_(
            1, idx, seg_lo, "amin", include_self=True)
        hi = torch.full((Tb, C, 3), EMPTY_HI, device=device).scatter_reduce_(
            1, idx, seg_hi, "amax", include_self=True)

        fill_alpha = self._rt_circuit_colors.opacity.squeeze(-1).amax(-1)  # over texture
        fill_min = self._rt_circuit_colors.opacity.squeeze(-1).amin(-1)
        if not self.filled:
            fill_alpha = torch.zeros_like(fill_alpha)
        border_alpha = self._rt_circuit_border_colors.opacity.squeeze(-1)
        border_on = self._rt_border_width > 1e-3
        visible = (fill_alpha > MIN_ALPHA) | (
            (border_alpha > MIN_ALPHA) & border_on)
        (lo, hi, visible, fill_min, border_alpha, border_on), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), fill_min.unsqueeze(-1),
             border_alpha.unsqueeze(-1), border_on.unsqueeze(-1)],
            "bezier bounds/colors")
        visible = visible.squeeze(-1)
        # A circuit is opaque (prunes hits behind it while gathering) only if
        # every region a hit can land in -- the fill/texture and, when shown,
        # the border -- is fully opaque.
        opaque = (fill_min.squeeze(-1) >= 1.0 - 1e-6) & (
            (~border_on.squeeze(-1))
            | (border_alpha.squeeze(-1) >= 1.0 - 1e-6))
        if not self.filled:
            opaque = torch.zeros_like(opaque)
        self._rt_frame_opaque = opaque.contiguous()
        lo = torch.where(visible.unsqueeze(-1), lo,
                         torch.tensor(EMPTY_LO, device=device))
        hi = torch.where(visible.unsqueeze(-1), hi,
                         torch.tensor(EMPTY_HI, device=device))

        # Inflate by the border (+ anti-crack outline) width converted to
        # world units at each circuit's distance from the camera.
        b1_norm = sb[:, 1].norm(p=2, dim=-1)
        screen_dist = (sp - cam_o).norm(p=2, dim=-1)
        pixel_world_scale = 2.0 / (screen_h * b1_norm * screen_dist).clamp_min(1e-12)
        centers = self._rt_circuit_meta[..., :3]
        dist = (centers - cam_o.view(-1, 1, 3)).norm(p=2, dim=-1)
        world_per_px = (pixel_world_scale.view(-1, 1) * dist).amax(0)
        inflate = (self._rt_border_width.amax(0) + 1.0) * world_per_px
        self._rt_frame_lo = (lo - inflate.view(1, -1, 1)).contiguous()
        self._rt_frame_hi = (hi + inflate.view(1, -1, 1)).contiguous()

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return render_batch_ray_traced(
            primitives, scene, screen_width, screen_height, time_start,
            time_end, background_color, transparent_background, *args,
            **kwargs)


def _empty_scene_part(device):
    """Placeholder STBVH + arrays for an absent geometry type."""
    lo = torch.full((1, 1, 3), EMPTY_LO, device=device)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=device)
    return build_stbvh(lo, hi, num_frames=1)


def _merge_scene(primitives):
    """Merge the batch's collections into one set per geometry type --
    triangles, PN patches and bezier circuits, each with a single STBVH
    over all frames -- cached for the batch.
    """
    first = primitives[0]
    cached = getattr(first, "_rt_merged_scene", None)
    if cached is not None:
        return cached

    empty_cache()
    device = COMPUTING_DEFAULTS.render_device
    pn_patches = [p for p in primitives
                  if isinstance(p, RayTracedPNTrianglePrimitive)]
    triangles = [p for p in primitives
                 if isinstance(p, RayTracedTrianglePrimitive)
                 and not isinstance(p, RayTracedPNTrianglePrimitive)]
    beziers = [p for p in primitives
               if isinstance(p, RayTracedBezierCircuitPrimitive)]
    unknown = [p for p in primitives
               if p not in triangles and p not in pn_patches
               and p not in beziers]
    if unknown:
        raise TypeError(
            "The ray traced renderer can only draw ray traced primitives; "
            f"got {[type(p).__name__ for p in unknown]}. Was "
            "enable_ray_tracing() called before the mobs were created?")
    num_frames = max(p._rt_num_frames for p in primitives)

    scene = {}
    if triangles:
        scene["tri_pos"] = _cat_collections(
            [p._rt_tri_pos for p in triangles], 1, "triangle merge")
        scene["tri_norm"] = _cat_collections(
            [p._rt_tri_norm for p in triangles], 1, "triangle merge")
        scene["tri_extra"] = _cat_collections(
            [p._rt_tri_extra for p in triangles], 1, "triangle merge")
        scene["tri_colors"] = _cat_collections(
            [p._rt_tri_colors for p in triangles], 1, "triangle merge")
        lo = _cat_collections([p._rt_frame_lo for p in triangles], 1,
                              "triangle merge")
        hi = _cat_collections([p._rt_frame_hi for p in triangles], 1,
                              "triangle merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in triangles], 1,
                                  "triangle merge")
        scene["tri_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=opaque)
    else:
        scene["tri_pos"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_extra"] = torch.zeros((1, 1, 6), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)
    scene["num_triangles"] = scene["tri_pos"].shape[1] if triangles else 0

    if pn_patches:
        scene["pn_ctrl"] = _cat_collections(
            [p._rt_pn_ctrl for p in pn_patches], 1, "pn merge")
        scene["pn_norm"] = _cat_collections(
            [p._rt_pn_norm for p in pn_patches], 1, "pn merge")
        scene["pn_extra"] = _cat_collections(
            [p._rt_pn_extra for p in pn_patches], 1, "pn merge")
        scene["pn_colors"] = _cat_collections(
            [p._rt_pn_colors for p in pn_patches], 1, "pn merge")
        lo = _cat_collections([p._rt_frame_lo for p in pn_patches], 1,
                              "pn merge")
        hi = _cat_collections([p._rt_frame_hi for p in pn_patches], 1,
                              "pn merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in pn_patches],
                                  1, "pn merge")
        scene["pn_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedPNTrianglePrimitive.stbvh_tightness,
            opaque=opaque)
    else:
        scene["pn_ctrl"] = torch.zeros((1, 1, 18), device=device)
        scene["pn_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["pn_extra"] = torch.zeros((1, 1, 6), device=device)
        scene["pn_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["pn_bvh"] = _empty_scene_part(device)
    scene["num_pn"] = scene["pn_ctrl"].shape[1] if pn_patches else 0

    if beziers:
        scene["circuit_meta"] = _cat_collections(
            [p._rt_circuit_meta for p in beziers], 1, "bezier merge")
        scene["circuit_border_colors"] = _cat_collections(
            [p._rt_circuit_border_colors for p in beziers], 1, "bezier merge")
        max_points = max(p._rt_circuit_colors.shape[2] for p in beziers)
        padded = []
        for p in beziers:
            c = p._rt_circuit_colors
            if c.shape[2] < max_points:
                pad = torch.zeros((c.shape[0], c.shape[1],
                                   max_points - c.shape[2], c.shape[3]),
                                  device=c.device)
                c = torch.cat((c, pad), 2)
            padded.append(c)
        scene["circuit_colors"] = _cat_collections(padded, 1, "bezier merge")
        scene["edges_2d"] = _cat_collections(
            [p._rt_edges for p in beziers], 1, "bezier merge")
        offsets, shift = [torch.zeros((1,), dtype=torch.int32, device=device)], 0
        for p in beziers:
            offsets.append(p._rt_edge_offsets[1:].long() + shift)
            shift = shift + p._rt_edges.shape[1]
        scene["edge_offsets"] = torch.cat(
            [o.to(torch.int32) for o in offsets]).contiguous()
        lo = _cat_collections([p._rt_frame_lo for p in beziers], 1,
                              "bezier merge")
        hi = _cat_collections([p._rt_frame_hi for p in beziers], 1,
                              "bezier merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in beziers], 1,
                                  "bezier merge")
        scene["bez_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque)
        scene["num_circuits"] = scene["circuit_meta"].shape[1]
    else:
        scene["circuit_meta"] = torch.zeros((1, 1, 20), device=device)
        scene["circuit_colors"] = torch.zeros((1, 1, 1, 5), device=device)
        scene["circuit_border_colors"] = torch.zeros((1, 1, 5), device=device)
        scene["edges_2d"] = torch.zeros((1, 1, 4), device=device)
        scene["edge_offsets"] = torch.zeros((2,), dtype=torch.int32,
                                            device=device)
        scene["bez_bvh"] = _empty_scene_part(device)
        scene["num_circuits"] = 0

    scene["num_frames"] = num_frames
    # The merged tensors replace the per-collection ones; release the
    # originals so peak GPU memory stays close to one copy of the scene.
    for p in triangles:
        p._rt_tri_pos = p._rt_tri_norm = None
        p._rt_tri_extra = p._rt_tri_colors = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in pn_patches:
        p._rt_pn_ctrl = p._rt_pn_norm = None
        p._rt_pn_extra = p._rt_pn_colors = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None

    empty_cache()
    first._rt_merged_scene = scene
    return scene


def _pack_lights(light_sources, num_frames, device):
    """Per-frame positions [T, L, 3] and RGB radiances [T, L, 3] of the
    scene's point lights (as prepared by the scene before rendering).
    """
    positions, colors = [], []
    for light in light_sources or ():
        positions.append(_expand_frames(
            _flat_frames(light.origin, (3,)), num_frames))
        col = light.light_color.reshape(light.light_color.shape[0], -1)
        colors.append(_expand_frames(col[:, :3].float(), num_frames))
    if not positions:
        return (torch.zeros((1, 1, 3), device=device),
                torch.zeros((1, 1, 3), device=device), 0)
    light_pos = torch.stack(positions, 1).to(device).contiguous()
    light_col = torch.stack(colors, 1).to(device).contiguous()
    return light_pos, light_col, light_pos.shape[1]


def _prefill_background(out, background_color, frame_offset, device):
    """Fill the output buffer with the background. Solid colors arrive as a
    float [channels] tensor in [0, 1]; animated/image backgrounds arrive as a
    uint8 row tensor [1 + frames * pixels, channels] (leading padding row).
    """
    num_frames, num_pixels, C_out = out.shape
    bg = background_color.to(device)
    if bg.dim() <= 1 or bg.shape[0] == 1:  # solid color (in [0, 1] floats)
        vals = (bg.float().flatten()[:5] * 255).round_().clamp_(0, 255)
        k = min(vals.shape[0], C_out)
        out[..., :k] = vals[:k].to(torch.uint8)
        if C_out > k:
            # Alpha (and any missing channel) defaults to the background's
            # last channel, matching opaque-by-default behavior.
            out[..., k:] = vals[-1].to(torch.uint8)
    else:
        rows = bg.reshape(-1, bg.shape[-1])[1:]
        rows = rows[frame_offset * num_pixels:
                    (frame_offset + num_frames) * num_pixels]
        rows = rows.view(num_frames, num_pixels, -1)
        k = min(rows.shape[-1], C_out)
        out[..., :k] = rows[..., :k].to(torch.uint8)
        if C_out > k:
            out[..., k:] = rows[..., -1:].to(torch.uint8)


def _downsample_background(background_color, aa, num_frames, screen_height,
                           screen_width):
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
    C = bg.shape[-1]
    body = bg.reshape(-1, C)[1:]  # drop the leading padding row
    h_aa, w_aa = screen_height * aa, screen_width * aa
    if body.shape[0] != num_frames * h_aa * w_aa:
        return bg  # not a super-sampled image background; leave as-is
    img = body.view(num_frames, h_aa, w_aa, C).float().permute(0, 3, 1, 2)
    ds = F.avg_pool2d(img, aa).permute(0, 2, 3, 1).reshape(-1, C)
    ds = (ds + 0.5).clamp_(0, 255).to(bg.dtype)
    return torch.cat((ds[:1], ds), 0)


def render_triangles_wavefront(
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
        tri_pos, tri_norm, tri_extra, tri_colors,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        time_start, time_end, width, height, half_screen_w, half_screen_h,
        layer_offset_triangles, max_bounces, transparent, out):
    """Wavefront orchestration for a triangle-only batch: byte-identical to
    ``render_triangles_stbvh`` but split into per-stage kernels over per-ray
    global state, with PyTorch ray compaction between host iterations.

    Geometry/intersection/shading is the megakernel's (see
    :mod:`algan.rendering.raytracing.wavefront_taichi`); this just owns the
    state buffers and the gen -> (traverse -> shade -> compact)* -> composite
    loop.
    """
    device = out.device
    n = (time_end - time_start) * width * height

    f32 = torch.float32
    i32 = torch.int32
    rs_ro = torch.empty((n, 3), dtype=f32, device=device)
    rs_rd = torch.empty((n, 3), dtype=f32, device=device)
    rs_acc = torch.empty((n, 4), dtype=f32, device=device)
    rs_sca = torch.empty((n, 4), dtype=f32, device=device)
    rs_int = torch.empty((n, 4), dtype=i32, device=device)
    rs_kt = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kl = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_ka = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kb = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kp = torch.empty((n, KBUF), dtype=i32, device=device)
    rs_kf = torch.empty((n, KBUF), dtype=i32, device=device)

    wf_gen_triangle(
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        int(time_start), int(width), int(height),
        float(half_screen_w), float(half_screen_h), int(max_bounces),
        rs_ro, rs_rd, rs_acc, rs_sca, rs_int)

    _timing = os.environ.get("ALGAN_WF_TIMING", "0") == "1"
    if _timing:
        import time as _t
        torch.cuda.synchronize(); ti.sync()
        _t_loop = _t.perf_counter()
        _t_traverse = 0.0
        _t_shade = 0.0
        _t_compact = 0.0

    active = torch.arange(n, dtype=i32, device=device)
    # Worst case: MAX_SURFACES_PER_RAY layers + a bounce-triggered re-traversal
    # per bounce, plus margin. Loop also exits as soon as no ray is active.
    max_iters = MAX_SURFACES_PER_RAY + max_bounces + 2
    it = 0
    while active.numel() > 0 and it < max_iters:
        na = int(active.numel())
        if _timing:
            import time as _t
            torch.cuda.synchronize(); ti.sync(); _s = _t.perf_counter()
        wf_traverse_triangle(
            active, na, t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
            int(t_first_leaf), tri_pos, float(layer_offset_triangles),
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
        if _timing:
            torch.cuda.synchronize(); ti.sync(); _m = _t.perf_counter()
            _t_traverse += _m - _s
        wf_shade_triangle(
            active, na, tri_pos, tri_norm, tri_extra, tri_colors,
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
        if _timing:
            torch.cuda.synchronize(); ti.sync(); _e = _t.perf_counter()
            _t_shade += _e - _m
        # Compaction: rays still flagged active (status column 2 == 0).
        active = (rs_int[:, 2] == 0).nonzero(as_tuple=True)[0].to(i32)
        if _timing:
            torch.cuda.synchronize(); ti.sync()
            _t_compact += _t.perf_counter() - _e
        it += 1

    if _timing:
        import time as _t
        torch.cuda.synchronize(); ti.sync()
        print(f"[wf] iters={it} traverse={_t_traverse*1e3:.1f}ms "
              f"shade={_t_shade*1e3:.1f}ms compact={_t_compact*1e3:.1f}ms "
              f"loop_total={(_t.perf_counter()-_t_loop)*1e3:.1f}ms")

    wf_composite(int(time_start), int(width), int(height),
                 1 if transparent else 0, rs_acc, rs_sca, out)


def render_general_wavefront(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces, transparent, out):
    """Wavefront orchestration for the general (triangle + PN + bezier) case:
    byte-identical to ``render_scene_stbvh`` but stage-split over per-ray global
    state, with PyTorch ray compaction between host iterations. State carries a
    5th scalar (base_dist) for bezier border widths across bounces."""
    device = out.device
    n = (time_end - time_start) * width * height
    f32 = torch.float32
    i32 = torch.int32
    rs_ro = torch.empty((n, 3), dtype=f32, device=device)
    rs_rd = torch.empty((n, 3), dtype=f32, device=device)
    rs_acc = torch.empty((n, 4), dtype=f32, device=device)
    rs_sca = torch.empty((n, 5), dtype=f32, device=device)
    rs_int = torch.empty((n, 4), dtype=i32, device=device)
    rs_kt = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kl = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_ka = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kb = torch.empty((n, KBUF), dtype=f32, device=device)
    rs_kp = torch.empty((n, KBUF), dtype=i32, device=device)
    rs_kf = torch.empty((n, KBUF), dtype=i32, device=device)

    wf_gen_general(
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        int(time_start), int(width), int(height),
        float(half_screen_w), float(half_screen_h), int(max_bounces),
        rs_ro, rs_rd, rs_acc, rs_sca, rs_int)

    _timing = os.environ.get("ALGAN_WF_TIMING", "0") == "1"
    _tt = _ts = _tc = 0.0
    if _timing:
        import time as _t
        torch.cuda.synchronize(); ti.sync(); _loop0 = _t.perf_counter()

    active = torch.arange(n, dtype=i32, device=device)
    max_iters = MAX_SURFACES_PER_RAY + max_bounces + 2
    it = 0
    while active.numel() > 0 and it < max_iters:
        na = int(active.numel())
        if _timing:
            import time as _t
            torch.cuda.synchronize(); ti.sync(); _s = _t.perf_counter()
        wf_traverse_general(
            active, na,
            tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
            tri_bvh.leaf_tspan, int(tri_bvh.first_leaf), merged["tri_pos"],
            pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
            pn_bvh.leaf_tspan, int(pn_bvh.first_leaf), merged["pn_ctrl"],
            bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
            bez_bvh.leaf_tspan, int(bez_bvh.first_leaf), merged["circuit_meta"],
            merged["edges_2d"], merged["edge_offsets"], pixel_world_scale,
            float(layer_offset_triangles), float(layer_offset_pn),
            int(has_tri), int(has_pn), int(has_bez),
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
        if _timing:
            torch.cuda.synchronize(); ti.sync(); _m = _t.perf_counter()
            _tt += _m - _s
        wf_shade_general(
            active, na,
            merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
            merged["tri_colors"],
            merged["pn_ctrl"], merged["pn_norm"], merged["pn_extra"],
            merged["pn_colors"],
            merged["circuit_meta"], merged["circuit_colors"],
            merged["circuit_border_colors"],
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
        if _timing:
            torch.cuda.synchronize(); ti.sync(); _e = _t.perf_counter()
            _ts += _e - _m
        ti.sync()
        active = (rs_int[:, 2] == 0).nonzero(as_tuple=True)[0].to(i32)
        if _timing:
            torch.cuda.synchronize(); ti.sync(); _tc += _t.perf_counter() - _e
        it += 1

    if _timing:
        import time as _t
        torch.cuda.synchronize(); ti.sync()
        print(f"[wf-general] iters={it} traverse={_tt*1e3:.1f}ms "
              f"shade={_ts*1e3:.1f}ms compact={_tc*1e3:.1f}ms "
              f"loop_total={(_t.perf_counter()-_loop0)*1e3:.1f}ms")

    wf_composite(int(time_start), int(width), int(height),
                 1 if transparent else 0, rs_acc, rs_sca, out)


@csync
def render_batch_ray_traced(primitives, scene, screen_width, screen_height,
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
    # Anti-aliasing strategy. The deterministic megakernels and the Monte
    # Carlo path tracers average ``aa^2`` jittered sub-pixel rays *in place* at
    # the output resolution, so the frame buffer stays ``screen_width x
    # screen_height`` regardless of ``aa`` (a^2x less render memory than
    # super-sampling). The wavefront path can't do that (its per-ray state is
    # one global cell per ray), so when it is selected we fall back to the
    # classic super-sample-then-average-down path: render at ``aa``x resolution
    # and let ``post_process_frames`` box-filter back down.
    inplace_aa = not USE_WAVEFRONT
    if inplace_aa:
        width = screen_width
        height = screen_height
        kernel_aa = aa          # in-kernel sub-pixel averaging factor
        post_aa = 1             # post-processing does not down-sample
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
    if inplace_aa and aa > 1:
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
    first = primitives[0]
    first.memory = memory

    samples = max(1, int(SAMPLES_PER_PIXEL))
    # In-place AA folds the anti-alias super-sampling into the Monte Carlo
    # sample count: each of the ``aa^2`` sub-pixels would have drawn ``samples``
    # random rays jittered over its own cell and then been averaged down, which
    # is equivalent (same total, same expectation) to drawing ``samples * aa^2``
    # rays jittered over the whole output pixel. (The wavefront/super-sample
    # path keeps ``kernel_aa == 1``, so ``samples_eff == samples`` there.)
    samples_eff = samples * (kernel_aa * kernel_aa)
    physical = bool(PHYSICAL_LIGHTING)
    if physical and samples <= 1:
        raise ValueError(
            "Physical lighting is a Monte Carlo mode; call "
            "set_samples_per_pixel(n) with n > 1 (e.g. 32) to use it.")
    if physical:
        light_pos, light_col, num_lights = _pack_lights(
            light_sources, num_frames, device)
    else:
        light_pos = light_col = None
        num_lights = 0

    def render_chunk(start, end):
        # The Monte Carlo kernels launch one thread per (frame, pixel,
        # sample) path; keep the flattened index within int32 range. (The
        # deterministic kernels loop the aa^2 sub-pixels serially per pixel, so
        # only the Monte Carlo path multiplies the thread count by the samples.)
        if (samples > 1 and
                (end - start) * width * height * samples_eff >= 1 << 31):
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "samples_per_pixel * resolution exceeds the ray tracer's "
                    "per-launch path budget (2^31). Please lower the sample "
                    "count, resolution or anti-alias level.")
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)
        entry_pointers = memory.get_pointers()
        try:
            out = memory.get_tensor((end - start, width * height, C_out),
                                    torch.uint8)
            _prefill_background(out, background_color, start - time_start,
                                device)
            accum = None
            if physical or samples > 1:
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
                merged["tri_colors"],
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
            if physical:
                path_trace_physical_stbvh(
                    *shared_args, samples_eff, light_pos, light_col,
                    int(num_lights), float(LIGHT_INTENSITY),
                    float(AMBIENT_LIGHT), out, accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 accum, out)
            elif samples > 1:
                path_trace_scene_stbvh(*shared_args, samples_eff,
                                       float(INDIRECT_BOUNCE_STRENGTH), out,
                                       accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 accum, out)
            elif (USE_WAVEFRONT and has_tri
                  and not has_pn and not has_bez):
                # Triangle-only batch via the wavefront (stage-split) path:
                # byte-identical output, higher occupancy / less divergence.
                render_triangles_wavefront(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    cam_origin, sp, pbx, pby,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0, out)
            elif USE_WAVEFRONT:
                # General (PN/bezier/mixed) batch via the wavefront path.
                render_general_wavefront(
                    tri_bvh, pn_bvh, bez_bvh, merged,
                    cam_origin, sp, pbx, pby, pixel_world_scale,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, layer_offset_pn,
                    has_tri, has_pn, has_bez, int(MAX_BOUNCES),
                    1 if transparent_background else 0, out)
            elif (USE_TRIANGLE_ONLY_KERNEL and has_tri
                  and not has_pn and not has_bez):
                # Triangle-only batch: the lean kernel (no PN/bezier code)
                # gives identical output at lower register pressure.
                render_triangles_stbvh(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    cam_origin, sp, pbx, pby,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0, kernel_aa, out)
            elif USE_NO_PN_KERNEL and not has_pn:
                # No PN patches (bezier circuits, optionally with flat
                # triangles): the no-PN kernel omits the Matrix Pencil solver from
                # its call graph for identical output at lower register
                # pressure. (Pure triangle-only was handled above.)
                render_no_pn_stbvh(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                    bez_bvh.leaf_tspan, bez_bvh.first_leaf,
                    merged["circuit_meta"], merged["circuit_colors"],
                    merged["circuit_border_colors"], merged["edges_2d"],
                    merged["edge_offsets"],
                    cam_origin, sp, pbx, pby, pixel_world_scale,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0,
                    has_tri, has_bez, kernel_aa, out)
            else:
                render_scene_stbvh(*shared_args, has_tri, has_pn, has_bez,
                                   kernel_aa, out)
            ti.sync()
            frames = out.view(end - start, height, width, C_out)
            frames = first.post_process_frames(
                frames, anti_alias_level=post_aa,
                post_processes=list(post_processes))
            memory.set_pointers(entry_pointers)
            return [frames]
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            memory.set_pointers(entry_pointers)
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


_originals = {}


def enable_ray_tracing(samples_per_pixel=None, indirect_bounce_strength=None,
                       physical_lighting=None, pn_triangles=False):
    """Route newly created mobs through the ray traced render pipeline.

    Rebinds the primitive classes used by the mob modules; call this before
    constructing the mobs that should be ray traced (bezier mobs bind their
    primitive class at construction time).

    Parameters
    ----------
    samples_per_pixel
        Rays averaged per pixel. 1 (default) renders with the exact
        deterministic kernel; larger values enable the Monte Carlo path
        tracer (see :func:`set_samples_per_pixel`).
    indirect_bounce_strength
        Diffuse indirect lighting strength for the Monte Carlo renderer
        (see :func:`set_indirect_bounce_strength`).
    physical_lighting
        Fully physical mode: skip vertex shading and light the scene with
        the explicit point lights, glow emission and the background
        environment (see :func:`set_physical_lighting`).
    pn_triangles
        Render triangle mobs as curved point-normal (PN) patches --
        quadratic Bezier triangles bent to match the vertex normals -- so
        coarsely tessellated smooth surfaces (spheres, parametric surfaces)
        keep smooth silhouettes. Triangles whose normals are zero or
        constant across the face stay exactly flat.
    """
    if samples_per_pixel is not None:
        set_samples_per_pixel(samples_per_pixel)
    if indirect_bounce_strength is not None:
        set_indirect_bounce_strength(indirect_bounce_strength)
    if physical_lighting is not None:
        set_physical_lighting(physical_lighting)
    triangle_cls = (RayTracedPNTrianglePrimitive if pn_triangles
                    else RayTracedTrianglePrimitive)
    targets = []
    import algan.mobs.bezier_circuit as bezier_circuit
    import algan.mobs.shapes_2d as shapes_2d
    import algan.mobs.surfaces.surface as surface
    targets.append((shapes_2d, "TrianglePrimitive", triangle_cls))
    targets.append((surface, "TrianglePrimitive", triangle_cls))
    targets.append((bezier_circuit, "BezierCircuitPrimitive",
                    RayTracedBezierCircuitPrimitive))
    try:
        import algan.mobs.plots as plots
        targets.append((plots, "TrianglePrimitive", triangle_cls))
    except Exception:
        pass  # plots has a broken legacy import; skip it.
    for module, name, cls in targets:
        _originals.setdefault((module, name), getattr(module, name))
        setattr(module, name, cls)


def disable_ray_tracing():
    """Restore the rasterized primitive classes for newly created mobs."""
    for (module, name), original in _originals.items():
        setattr(module, name, original)
    _originals.clear()


def is_ray_tracing_enabled():
    """True if the ray traced primitive classes are currently active (i.e.
    :func:`enable_ray_tracing` has been called and not yet disabled)."""
    return bool(_originals)
