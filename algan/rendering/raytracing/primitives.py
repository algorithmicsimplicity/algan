"""Ray traced drop-in replacements for Algan's rasterized render primitives.

``RayTracedTrianglePrimitive`` and ``RayTracedBezierCircuitPrimitive`` subclass
the rasterized primitives to keep their construction and batching, but render
through a self-contained ray tracing pipeline:

* ``project_to_screen`` shades vertices and packs geometry + per-frame bounds
  for the whole batch of frames (the spatio-temporal BVH of ``stbvh.py``).
* ``render`` traces one ray per (frame, pixel) with the unified Taichi kernel
  (``ray_trace_taichi.py``), which alpha-blends every surface it encounters
  -- including mirror bounces -- directly into a fixed
  ``[num_frames, num_pixels, channels]`` output buffer. Memory use is
  independent of depth complexity and bounce count, and there is no fragment
  buffer, sorting pass or atomic contention.

Mirrors: give a mob a reflectivity with :func:`set_reflectivity` (before
spawning); the value is per-vertex, animatable, and bounces up to
``MAX_BOUNCES`` times.

Call :func:`enable_ray_tracing` *before constructing any mobs* to make new
mobs render through this pipeline.
"""
from __future__ import annotations

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
from algan.rendering.raytracing.ray_trace_taichi import (
    MIN_ALPHA,
    path_trace_scene_stbvh,
    render_scene_stbvh,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO, build_stbvh
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.utils.memory_utils import InsufficientMemoryException
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
            # line up.
            for name in self._surface_params:
                values = []
                for triangle in triangle_collection:
                    v = getattr(triangle, name, None)
                    if v is None:
                        v = torch.zeros_like(triangle.colors[..., :1])
                    v = broadcast_all(
                        [triangle.corners, triangle.colors, triangle.normals,
                         v], ignored_dims=[-1])[-1][..., :1]
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
                            torch.zeros_like(self.colors[..., :1]))
                else:
                    value = cast_to_tensor(value).to(self.colors.device)
                    setattr(self, name, broadcast_all(
                        [self.corners, self.colors, value],
                        ignored_dims=[-1])[-1][..., :1])

    @csync
    def project_to_screen(self, camera, light_sources):
        with CudaStream():
            # Vertex shading, identical to the rasterized pipeline.
            d = -1
            if getattr(self, "shader", None) is not None:
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

            corners = self.corners.float()
            normals = self.normals.float()
            reflectivity = self.reflectivity.float()
            roughness = self.roughness.float()
            (corners_e, normals_e, reflectivity_e, roughness_e), _ = _unify_time(
                [corners, normals, reflectivity, roughness],
                "triangle vertex data")
            # Packed per-corner data: position, shading normal, reflectivity,
            # roughness.
            self._rt_tri_verts = torch.cat(
                (corners_e, normals_e, reflectivity_e, roughness_e),
                -1).contiguous()
            self._rt_tri_colors = self.colors.float().contiguous()
            num_frames = camera.ray_origin.shape[0]
            self._rt_num_frames = num_frames

            # Per-frame bounds; frames where a triangle is fully transparent
            # are marked empty so they never enter the BVH.
            lo = corners.amin(-2)
            hi = corners.amax(-2)
            visible = self._rt_tri_colors[..., -1].amax(-1) > MIN_ALPHA
            (lo, hi, visible), _ = _unify_time(
                [lo, hi, visible.unsqueeze(-1)], "triangle bounds/colors")
            visible = visible.squeeze(-1)
            self._rt_frame_lo = torch.where(
                visible.unsqueeze(-1), lo,
                torch.tensor(EMPTY_LO, device=lo.device)).contiguous()
            self._rt_frame_hi = torch.where(
                visible.unsqueeze(-1), hi,
                torch.tensor(EMPTY_HI, device=hi.device)).contiguous()

            # Everything the renderer needs now lives in the packed arrays;
            # release the unpacked geometry to halve resident GPU memory.
            self.corners = self.normals = None
            self.reflectivity = self.roughness = None
            self.colors = None

            self._rt_frame_bytes = int(
                camera.screen_width * camera.screen_height * 5 * 4)
        torch.cuda.synchronize()
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

            self._rt_frame_bytes = int(
                camera.screen_width * camera.screen_height * 5 * 4)
        torch.cuda.synchronize()
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

        fill_alpha = self._rt_circuit_colors[..., -1].amax(-1)  # over texture
        if not self.filled:
            fill_alpha = torch.zeros_like(fill_alpha)
        border_alpha = self._rt_circuit_border_colors[..., -1]
        border_on = self._rt_border_width > 1e-3
        visible = (fill_alpha > MIN_ALPHA) | (
            (border_alpha > MIN_ALPHA) & border_on)
        (lo, hi, visible), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1)], "bezier bounds/colors")
        visible = visible.squeeze(-1)
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
    """Merge the batch's collections into one triangle set and one bezier set
    (each with a single STBVH over all frames), cached for the batch.
    """
    first = primitives[0]
    cached = getattr(first, "_rt_merged_scene", None)
    if cached is not None:
        return cached

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = COMPUTING_DEFAULTS.render_device
    triangles = [p for p in primitives
                 if isinstance(p, RayTracedTrianglePrimitive)]
    beziers = [p for p in primitives
               if isinstance(p, RayTracedBezierCircuitPrimitive)]
    unknown = [p for p in primitives
               if p not in triangles and p not in beziers]
    if unknown:
        raise TypeError(
            "The ray traced renderer can only draw ray traced primitives; "
            f"got {[type(p).__name__ for p in unknown]}. Was "
            "enable_ray_tracing() called before the mobs were created?")
    num_frames = max(p._rt_num_frames for p in primitives)

    scene = {}
    if triangles:
        scene["tri_verts"] = _cat_collections(
            [p._rt_tri_verts for p in triangles], 1, "triangle merge")
        scene["tri_colors"] = _cat_collections(
            [p._rt_tri_colors for p in triangles], 1, "triangle merge")
        lo = _cat_collections([p._rt_frame_lo for p in triangles], 1,
                              "triangle merge")
        hi = _cat_collections([p._rt_frame_hi for p in triangles], 1,
                              "triangle merge")
        scene["tri_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness)
    else:
        scene["tri_verts"] = torch.zeros((1, 1, 3, 8), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)

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
        scene["bez_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness)
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
        p._rt_tri_verts = p._rt_tri_colors = None
        p._rt_frame_lo = p._rt_frame_hi = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    first._rt_merged_scene = scene
    return scene


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
    width = screen_width * anti_alias_level
    height = screen_height * anti_alias_level
    C_out = 5 if transparent_background else 4
    device = COMPUTING_DEFAULTS.render_device
    num_frames = merged["num_frames"]

    cam_origin = _expand_frames(_flat_frames(ray_origin, (3,)),
                                num_frames).contiguous()
    sp = _expand_frames(_flat_frames(screen_point, (3,)),
                        num_frames).contiguous()
    sb = _expand_frames(_flat_frames(screen_basis, (3, 3)), num_frames)
    pbx, pby = _pixel_bases(sb)
    # World units per screen pixel per unit distance (for border widths).
    b1_norm = sb[:, 1].norm(p=2, dim=-1)
    screen_dist = (sp - cam_origin).norm(p=2, dim=-1)
    pixel_world_scale = (2.0 / (height * b1_norm * screen_dist).clamp_min(1e-12)
                         ).contiguous()

    tri_bvh = merged["tri_bvh"]
    bez_bvh = merged["bez_bvh"]
    first = primitives[0]
    first.memory = memory

    samples = max(1, int(SAMPLES_PER_PIXEL))

    def render_chunk(start, end):
        entry_pointers = memory.get_pointers()
        try:
            out = memory.get_tensor((end - start, width * height, C_out),
                                    torch.uint8)
            _prefill_background(out, background_color, start - time_start,
                                device)
            torch.cuda.synchronize()
            shared_args = (
                tri_bvh.node_lo, tri_bvh.node_hi, tri_bvh.node_tmin,
                tri_bvh.node_tmax, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.first_leaf,
                merged["tri_verts"], merged["tri_colors"],
                bez_bvh.node_lo, bez_bvh.node_hi, bez_bvh.node_tmin,
                bez_bvh.node_tmax, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.first_leaf,
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"], merged["edges_2d"],
                merged["edge_offsets"],
                cam_origin, sp, pbx, pby, pixel_world_scale,
                int(start), int(end), int(width), int(height),
                float(width // 2), float(height // 2),
                float(merged["num_circuits"]), int(MAX_BOUNCES),
                1 if transparent_background else 0)
            if samples > 1:
                path_trace_scene_stbvh(*shared_args, samples,
                                       float(INDIRECT_BOUNCE_STRENGTH), out)
            else:
                render_scene_stbvh(*shared_args, out)
            ti.sync()
            frames = out.view(end - start, height, width, C_out)
            frames = first.post_process_frames(
                frames, anti_alias_level=anti_alias_level,
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


def enable_ray_tracing(samples_per_pixel=None, indirect_bounce_strength=None):
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
    """
    if samples_per_pixel is not None:
        set_samples_per_pixel(samples_per_pixel)
    if indirect_bounce_strength is not None:
        set_indirect_bounce_strength(indirect_bounce_strength)
    targets = []
    import algan.mobs.bezier_circuit as bezier_circuit
    import algan.mobs.shapes_2d as shapes_2d
    import algan.mobs.surfaces.surface as surface
    targets.append((shapes_2d, "TrianglePrimitive", RayTracedTrianglePrimitive))
    targets.append((surface, "TrianglePrimitive", RayTracedTrianglePrimitive))
    targets.append((bezier_circuit, "BezierCircuitPrimitive",
                    RayTracedBezierCircuitPrimitive))
    try:
        import algan.mobs.plots as plots
        targets.append((plots, "TrianglePrimitive", RayTracedTrianglePrimitive))
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
