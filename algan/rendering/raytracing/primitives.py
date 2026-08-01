from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F

from algan.constants.color import Color
from algan.rendering.logical_pn import (
    evaluate_cubic_curve,
    evaluate_logical_pn,
    evaluate_logical_pn_normals,
    interpolate_patch_attribute,
    logical_pn_control_points,
    logical_pn_edge_control_points,
    logical_pn_normal_control_points,
    snap_boundary_values,
    subdivision_boundary_map,
    subdivision_triangle_indices,
    subdivision_triangle_uvs,
    subdivision_vertex_uvs,
)
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
    batch_arange,
)
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.rendering.raytracing import pn_control_points, pn_patch_coefficients
from algan.rendering.raytracing.pn_patch import pn_obb
from algan.rendering.raytracing.raytrace_kernels_taichi import MIN_ALPHA
from algan.rendering.raytracing.settings import (
    _MAT_DEFAULTS,
    _MAT_SLOTS,
    _shader_is_core,
    _shader_material_id,
)
from algan.rendering.raytracing.shading_taichi import MAT_W
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _unify_time
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

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))

    # Renderer-internal transport channels, shared with
    # ``RayTracedBezierCircuitPrimitive``. ``reflectivity`` stores material
    # metalness for historical packed-layout compatibility; a negative value
    # marks a non-PBR material. ``refractive_index`` is an unsigned magnitude
    # (0 = non-PBR) feeding dielectric F0 and Snell; ``transmission`` alone says
    # whether -- and how much -- the surface transmits. All are derived from the
    # material alone (see ``_derive_material_surface_params``) -- there is no
    # user-facing renderer control, matching the Three.js material interface.
    _surface_params = ("reflectivity", "roughness", "refractive_index", "transmission")

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
            self._derive_material_surface_params()

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
        remain robust to missing optional parameters.
        """
        import inspect

        from algan.rendering.shaders.pbr_shaders import default_shader

        sig = inspect.signature(self.shader).parameters
        num_fixed = len(inspect.signature(default_shader).parameters)
        extra_names = list(sig.keys())[num_fixed:]

        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        by_name = dict(zip(names, values))

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
                        *param_values,
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
        pairs = []
        if _shader_is_core(shader):
            # The material's shader params, addressed by their real names.
            names = list(getattr(self, "shader_param_names", None) or [])
            values = list(getattr(self, "shader_param_values", None) or [])
            for name, value in zip(names, values):
                if name in _MAT_SLOTS and value is not None:
                    pairs.append((name, per_triangle(value)))
        Tm = max([1] + [v.shape[0] for _n, v in pairs])
        mat = (
            torch.tensor(_MAT_DEFAULTS, device=device)
            .view(1, 1, MAT_W)
            .expand(Tm, N, MAT_W)
            .contiguous()
        )
        for name, v in pairs:
            start, width = _MAT_SLOTS[name]
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
        """Per-primitive surface params [Te, N, 12]: the interleaved per-corner
        (reflectivity, roughness) pairs in columns 0-5 (consumed by
        ``_triangle_extra`` in every kernel), followed by the per-corner
        refractive index in columns 6-8 (unsigned magnitude, 0 = non-PBR; read
        by the wavefront's ``_corner_ior``), followed by the per-corner
        transmission in columns 9-11 (0 = opaque to light passing through; read
        by ``_corner_transmission``).
        """
        (reflectivity_e, roughness_e, ior_e, transmission_e), _ = _unify_time(
            [
                self.reflectivity.float(),
                self.roughness.float(),
                self.refractive_index.float(),
                self.transmission.float(),
            ],
            error_context,
        )
        n_t, n_p = reflectivity_e.shape[0], reflectivity_e.shape[1]
        refl_rough = torch.cat((reflectivity_e, roughness_e), -1).reshape(n_t, n_p, 6)
        ior = ior_e.reshape(n_t, n_p, 3)
        transmission = transmission_e.reshape(n_t, n_p, 3)
        return torch.cat((refl_rough, ior, transmission), -1).contiguous()

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
        # ...but full coverage is not enough to prune hits behind: a
        # transmissive surface still lets light through at alpha 1.
        opaque = alpha.amin(-1) >= 1.0 - 1e-6
        transmission = getattr(self, "transmission", None)
        if transmission is not None:
            opaque = opaque & (transmission[..., 0] <= 1e-6).all(-1)

        (lo, hi, visible, opaque), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), opaque.unsqueeze(-1)], error_context
        )
        visible = visible.squeeze(-1)
        self._rt_frame_opaque = opaque.squeeze(-1).contiguous()
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
        self._rt_num_frames = camera.ray_origin.shape[0]

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

    This class is deliberately unrelated to
    :class:`RayTracedPNTrianglePrimitive`: the latter ray-intersects curved
    patches directly and is retained only for legacy callers.  Logical PN
    patches use their fixed construction-time topology as source geometry and
    dice into flat triangles for each materialized camera frame.  The packed
    result follows the normal flat-triangle/STBVH path.

    **Every patch picks its own subdivision level, in every frame.**  A patch
    that fills the screen costs what it needs and nothing else pays for it --
    neither the other patches of the same mesh in that frame, nor the same patch
    in the frames where it is small or off screen.  Only the padded width of the
    output tensor is shared: each frame's patches are packed back to back, and
    the batch is padded to the largest per-frame total (surplus rows are marked
    invisible, exactly as before).

    Independent per-patch levels would crack the mesh open along its seams, so
    the level of a patch's three boundary curves is decided separately from its
    interior:

    * A boundary curve's level is a function of that curve alone -- its two
      endpoints and their normals, which the two patches sharing it hold in
      common -- evaluated on canonically ordered controls
      (:func:`~algan.rendering.logical_pn.logical_pn_edge_control_points`) so
      both neighbours reach a bit-identical answer without any adjacency
      information.
    * A patch's own level is at least the largest of its three boundary levels,
      and is then raised until its interior is flat enough.
    * Where the interior level exceeds a boundary level, the boundary vertices
      of the finer grid are snapped back onto the coarser boundary polyline
      (:func:`~algan.rendering.logical_pn.snap_boundary_values`).  Levels are
      powers of two, so the coarse polyline's knots are always vertices of the
      finer grid and the snapped boundary reproduces it exactly: the seam is
      watertight whatever the two neighbours chose.

    The tolerance guarantee is therefore stated per component: the diced
    boundary lands within ``render_tolerance`` of the true boundary curve and
    the diced interior within ``render_tolerance`` of the true patch, both in
    output pixels.  In the band of microtriangles touching a snapped boundary
    the two displacements can add, for a worst case of twice the tolerance.
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

    def __init__(self, *args, render_tolerance=0.5, **kwargs):
        collection = kwargs.get("triangle_collection")
        if collection is not None:
            tolerances = [
                float(getattr(p, "render_tolerance", render_tolerance))
                for p in collection
            ]
            render_tolerance = min(tolerances)
        super().__init__(*args, **kwargs)
        self.render_tolerance = float(render_tolerance)
        if not torch.isfinite(torch.tensor(self.render_tolerance)):
            raise ValueError("render_tolerance must be finite")
        if self.render_tolerance <= 0:
            raise ValueError("render_tolerance must be greater than zero")

    def get_batch_identifier(self):
        return (
            f"{super().get_batch_identifier()}"
            f"_logical_pn_render_tolerance={self.render_tolerance}"
        )

    @staticmethod
    def _project_to_output_pixels(points, cam_o, sp, sb, screen_height):
        """Perspective-project ``[T, ... ,3]`` points into output pixels."""
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
        return pixels * (float(screen_height) / 2.0), depth.squeeze(-1)

    def _guarded_pixel_error(self, exact, approximated, cam, front_sign, screen_height):
        """Guarded projected pixel deviation between matching point sets.

        ``exact`` and ``approximated`` are ``[K, ..., 3]``; ``cam`` and
        ``front_sign`` carry one camera row per leading element.

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
        exact_pixels, exact_depth = self._project_to_output_pixels(
            exact, *cam, screen_height
        )
        approximated_pixels, approximated_depth = self._project_to_output_pixels(
            approximated, *cam, screen_height
        )
        error = (
            exact_pixels.clamp(-guard, guard) - approximated_pixels.clamp(-guard, guard)
        ).norm(dim=-1)
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
        """``4 ** levels``, the diced triangle count of each patch."""
        return torch.bitwise_left_shift(torch.ones_like(levels), 2 * levels)

    def _required_subdivision_levels(
        self, control_points, edge_controls, cam_o, sp, sb, screen_height
    ):
        """Choose the crack-free logical PN levels of every patch and edge.

        Returns per-patch interior levels ``[T, P]`` and per-edge boundary
        levels ``[T, P, 3]``, both of which vary freely from patch to patch and
        from frame to frame.
        """
        # Which side of the camera plane is in front: the screen plane's own
        # side, exactly as the renderer's front test decides it.
        front_sign = torch.sign(((sp - cam_o) * sb[:, 2]).sum(-1))
        cam = (cam_o, sp, sb)
        edge_levels, edge_capped = self._required_edge_levels(
            edge_controls, cam, front_sign, screen_height
        )
        levels, patch_capped = self._required_patch_levels(
            control_points,
            edge_levels.amax(-1),
            cam,
            front_sign,
            screen_height,
        )
        if edge_capped or patch_capped:
            warnings.warn(
                "Logical PN render tessellation reached its safety cap before "
                "meeting render_tolerance for every patch.",
                RuntimeWarning,
                stacklevel=3,
            )
        return levels, edge_levels

    def _required_edge_levels(self, edge_controls, cam, front_sign, screen_height):
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
        threshold = self.render_tolerance * float(screen_height)

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
        self, edge_controls, active, level, cam, front_sign, samples, screen_height
    ):
        """Peak pixel deviation of each active curve from its chord polyline.

        The polyline has ``2 ** level`` chords; every chord is compared against
        the curve at ``_edge_sample_parameters``.  Work is streamed in chunks so
        scratch stays inside ``max_scratch_triangles`` however many curves are
        still looking for a level.
        """
        device = edge_controls.device
        dtype = edge_controls.dtype
        num_patches = edge_controls.shape[1]
        segments = 1 << level
        num_samples = samples.numel()
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
            ).amax(dim=(1, 2))
        return error

    def _required_patch_levels(
        self, control_points, start, cam, front_sign, screen_height
    ):
        """Per-patch interior subdivision levels, shape ``[T, P]``.

        Every patch starts at the largest of its three boundary levels -- the
        floor imposed by the snap in
        :func:`~algan.rendering.logical_pn.snap_boundary_values` -- and climbs
        only while its *own* dice misses ``render_tolerance``.  Because the
        active set shrinks as patches resolve, the whole search costs about a
        third more than the tessellation it settles on, rather than one full
        trial tessellation of the entire mesh per level tried.

        The criterion measures the *unsnapped* dice.  Folding the boundary snap
        in instead would be measuring against a floor the interior cannot get
        under -- the snap displacement is fixed by the boundary level, and is
        itself allowed to reach the tolerance -- so patches whose boundary
        resolved just inside the tolerance would climb to the safety cap
        without ever passing.  The two approximations are held to the tolerance
        separately (see the class docstring).
        """
        max_level = int(self.max_subdivision_level)
        budget = max(1, int(self.max_diced_triangles))
        threshold = self.render_tolerance * float(screen_height)
        levels = start.clone()
        if levels.numel() == 0:
            return levels, False
        unresolved = torch.ones_like(levels, dtype=torch.bool)
        # Accumulated on the device and read back once: a per-iteration
        # ``.any()`` would stall the queue at every level for a flag that only
        # decides whether to warn.
        capped = torch.zeros((), dtype=torch.bool, device=levels.device)

        for level in range(int(levels.amin().item()), max_level + 1):
            selected = (unresolved & (levels == level)).nonzero()
            if not selected.shape[0]:
                if not bool(unresolved.any()):
                    break
                continue
            frames, patches = selected[:, 0], selected[:, 1]
            error = self._patch_flatness_error(
                control_points,
                selected,
                level,
                cam,
                front_sign,
                screen_height,
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
        self, control_points, selected, level, cam, front_sign, screen_height
    ):
        """Peak pixel deviation of each selected patch's level-``level`` dice,
        sampled at ``_flatness_sample_weights`` within every microtriangle.
        """
        device = control_points.device
        dtype = control_points.dtype
        vertex_uv = subdivision_vertex_uvs(level, device=device, dtype=dtype)
        triangle_indices = subdivision_triangle_indices(level, device=device)
        corner_uv = subdivision_triangle_uvs(level, device=device, dtype=dtype)
        weights = _sample_tensor(self._flatness_sample_weights, device, dtype)
        sample_uv = torch.einsum("sk,mka->msa", weights, corner_uv)
        chunk = max(
            1,
            int(self.max_scratch_triangles)
            // max(1, triangle_indices.shape[0] * weights.shape[0]),
        )

        error = torch.empty(selected.shape[0], device=device, dtype=dtype)
        for start in range(0, selected.shape[0], chunk):
            rows = selected[start : start + chunk]
            frames, patches = rows[:, 0], rows[:, 1]
            controls = control_points[frames, patches].unsqueeze(0)
            vertices = evaluate_logical_pn(controls, vertex_uv)[0]
            approximated = torch.einsum(
                "sk,pmkc->pmsc", weights, vertices[:, triangle_indices]
            )
            error[start : start + chunk] = self._guarded_pixel_error(
                evaluate_logical_pn(controls, sample_uv)[0],
                approximated,
                tuple(value.index_select(0, frames) for value in cam),
                front_sign.index_select(0, frames),
                screen_height,
            ).amax(dim=(1, 2))
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

    def _dice_logical_pn(self, camera):
        num_frames = int(camera.ray_origin.shape[0])
        source_corners = self.corners.float()
        source_normals = self.normals.float()
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
        levels, edge_levels = self._required_subdivision_levels(
            control_points,
            edge_controls,
            cam_o,
            sp,
            sb,
            output_height,
        )

        # Each frame packs its patches back to back at their own diced sizes;
        # only the batch's widest frame sets the padded width. A frame that
        # needs a fraction of the detail no longer pays for the frame that
        # needs the most, and neither does a patch for its neighbours.
        counts = self._triangle_counts(levels)
        offsets = counts.cumsum(1) - counts
        max_triangles = int(counts.sum(1).amax().item()) if counts.numel() else 0

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
        padding = torch.ones(
            (num_frames, max_triangles),
            dtype=torch.bool,
            device=device,
        )

        for level in levels.unique(sorted=True).tolist():
            level = int(level)
            selected = (levels == level).nonzero()
            vertex_uv = subdivision_vertex_uvs(level, device=device, dtype=dtype)
            triangle_indices = subdivision_triangle_indices(level, device=device)
            corner_uv = subdivision_triangle_uvs(level, device=device, dtype=dtype)
            boundary = subdivision_boundary_map(level, device=device)
            num_triangles = triangle_indices.shape[0]
            columns = torch.arange(num_triangles, device=device)
            chunk = max(1, int(self.max_scratch_triangles) // num_triangles)

            for start in range(0, selected.shape[0], chunk):
                rows = selected[start : start + chunk]
                frames, patches = rows[:, 0], rows[:, 1]
                edges = edge_levels[frames, patches]
                # The patch is evaluated once per shared subdivision vertex
                # (each is a corner of up to six microtriangles), snapped onto
                # its boundary polylines, and only then expanded to the
                # triangle-soup layout the packed geometry wants.
                positions = snap_boundary_values(
                    evaluate_logical_pn(
                        control_points[frames, patches].unsqueeze(0),
                        vertex_uv,
                    )[0],
                    level,
                    edges,
                    boundary,
                )
                vertex_normals = F.normalize(
                    snap_boundary_values(
                        evaluate_logical_pn_normals(
                            normal_control_points[frames, patches].unsqueeze(0),
                            vertex_uv,
                        )[0],
                        level,
                        edges,
                        boundary,
                    ),
                    p=2,
                    dim=-1,
                )
                target_rows = frames.unsqueeze(1).expand(-1, num_triangles)
                target_columns = offsets[frames, patches].unsqueeze(1) + columns

                diced_corners[target_rows, target_columns] = positions[
                    :, triangle_indices
                ]
                diced_normals[target_rows, target_columns] = vertex_normals[
                    :, triangle_indices
                ]
                diced_colors[target_rows, target_columns] = interpolate_patch_attribute(
                    colors[frames, patches], corner_uv
                )
                for name, output in diced_surface_params.items():
                    output[target_rows, target_columns] = interpolate_patch_attribute(
                        surface_sources[name][frames, patches], corner_uv
                    )
                for output, source in zip(diced_shader_params, shader_sources):
                    output[target_rows, target_columns] = interpolate_patch_attribute(
                        source[frames, patches], corner_uv
                    )
                if diced_uvs is not None:
                    diced_uvs[target_rows, target_columns] = (
                        interpolate_patch_attribute(
                            uv_source[frames, patches], corner_uv
                        )
                    )
                padding[target_rows, target_columns] = False

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

    def project_to_screen(self, camera, light_sources):
        self._shade_vertex_colors(camera, light_sources)

        corners = self.corners.float()
        normals = self.normals.float()
        # Hot/cold split as for flat triangles, with the patch's
        # monomial coefficients as the hot geometry. corners and
        # normals share a time dimension by construction (the batching
        # constructor broadcasts them together).
        control_points = pn_control_points(corners, normals)
        self._rt_pn_ctrl = pn_patch_coefficients(control_points).contiguous()
        # Tight oriented bounding box per patch: the trace kernel tests it
        # before the matrix-pencil solve to reject the (many) candidates
        # whose loose axis-aligned leaf box the ray pierces but whose actual
        # (often thin, diagonal) patch it misses.
        self._rt_pn_obb = pn_obb(control_points).contiguous()
        self._rt_pn_norm = normals.reshape(
            normals.shape[0], normals.shape[1], 9
        ).contiguous()
        self._rt_pn_extra = self._pack_surface_extra("pn surface params")
        self._rt_pn_colors = self.colors.float().contiguous()
        self._rt_pn_mat_id, self._rt_pn_mat = self._pack_material()
        self._rt_num_frames = camera.ray_origin.shape[0]

        # Texture maps (color / material / normal). PN patches have no kernel
        # argument budget left (the general wavefront shade kernel is at
        # Taichi's 64-arg ceiling), so unlike flat triangles the UVs and the
        # per-patch texture metadata are folded into the cold pn_extra array at
        # merge time (see _merge_scene); here we just stash the raw maps + UVs.
        self._rt_pn_uvs = self._stash_texture_maps()

        # The patch lies in the convex hull of its control points, so
        # the control net bounds it.
        self._pack_frame_visibility(
            control_points.amin(-2),
            control_points.amax(-2),
            self._rt_pn_colors,
            "pn bounds/colors",
        )

        self._release_unpacked_geometry()
        return self


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
    _surface_params = ("reflectivity", "roughness", "refractive_index", "transmission")

    # Non-PBR sentinel for metalness; the other channels are inert at 0.
    _surface_param_fill = {"reflectivity": -1.0}

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

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))
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

        while active.numel() > 0:
            num_active = active.shape[0]
            max_error_squared = torch.zeros(
                (num_active,), dtype=corners.dtype, device=device
            )

            # Bound the largest temporary by projected control-point count.
            # The subcurve construction and projection use several arrays of
            # this shape, so a lower budget than the old single-pass sampler is
            # intentionally used here.
            chunk = max(1, int(5e5 // max(num_active * num_subdivisions * 4, 1)))
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
        self._rt_edges = (
            torch.cat((locals_uv, next_uv, border_visible_e.unsqueeze(-1)), -1)
            .float()
            .contiguous()
        )
        self._rt_edges = torch.where(
            edge_degenerate_e.unsqueeze(-1),
            torch.tensor([1e9, 1e9, 1e9, 1e9, 0.0], device=device),
            self._rt_edges,
        )

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
        self._rt_circuit_border_colors = (
            self.border_color.float().contiguous().as_subclass(Color)
        )
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
        border_alpha = self._rt_circuit_border_colors.opacity.squeeze(-1)
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
        (lo, hi, visible, fill_min, border_alpha, border_on, transmissive), _ = (
            _unify_time(
                [
                    lo,
                    hi,
                    visible.unsqueeze(-1),
                    fill_min.unsqueeze(-1),
                    border_alpha.unsqueeze(-1),
                    border_on.unsqueeze(-1),
                    transmissive.unsqueeze(-1),
                ],
                "bezier bounds/colors",
            )
        )
        visible = visible.squeeze(-1)
        # A circuit is opaque (prunes hits behind it while gathering) only if
        # every region a hit can land in -- the fill/texture and, when shown,
        # the border -- is fully opaque.
        opaque = (fill_min.squeeze(-1) >= 1.0 - 1e-6) & (
            (~border_on.squeeze(-1)) | (border_alpha.squeeze(-1) >= 1.0 - 1e-6)
        )
        # A transmissive circuit lets light through even at full coverage, so
        # it can never prune hits behind it.
        opaque = opaque & ~transmissive.squeeze(-1)
        if not self.filled:
            opaque = torch.zeros_like(opaque)
        self._rt_frame_opaque = opaque.contiguous()
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
