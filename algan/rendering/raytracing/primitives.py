import torch.nn.functional as F

from algan.constants.color import Color
from algan.utils.memory_utils import empty_cache
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.rendering.primitives.bezier_circuit_primitive import batch_arange, BezierCircuitPrimitive
from algan.rendering.raytracing import pn_control_points, pn_patch_coefficients
from algan.rendering.raytracing.pn_patch import pn_obb
from algan.rendering.raytracing.raytrace_kernels_taichi import MIN_ALPHA, KBUF
from algan.rendering.raytracing.settings import _shader_is_core, _shader_material_id, _MAT_SLOTS, _MAT_DEFAULTS
from algan.rendering.raytracing.shading_taichi import MAT_W, _USER_PIPELINE_BASE
from algan.rendering.raytracing.stbvh import EMPTY_LO, EMPTY_HI
from algan.rendering.raytracing.utils import _expand_frames, _unify_time, _flat_frames
from algan.utils.tensor_utils import *
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.rendering.raytracing.settings import *
from algan.settings.kernel_settings import KERNEL_SETTINGS


def _set_surface_param(mob, name, value):
    value = cast_to_tensor(float(value)).view(1, 1)
    for descendant in reversed(mob.get_descendants()):
        # Register the attr as animatable BEFORE setting it. A plain
        # ``setattr`` here would store an instance attribute that is later
        # shadowed if another mob's shader registers ``name`` as a class-level
        # animatable property (e.g. standard_shader registering ``roughness``
        # after a phong mob's set_material routed its roughness through this
        # helper) -- the shadowed value then reads as AttributeError at batch
        # prep. Registering first routes the value through the animated
        # storage, which stays readable regardless of registration order.
        descendant.register_attrs_as_animatable([name])
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

    This is a convenience shortcut for the corresponding
    :class:`~algan.rendering.shaders.materials.Material` property: applying a
    material with :meth:`~algan.mobs.mob.Mob.set_material` routes the same
    surface parameter (from :meth:`Material.physical_surface_params`). Use this
    setter directly when you want mirror-ness without configuring a full
    material (e.g. a mirror in the deterministic renderer, where ``set_material``
    deliberately does not auto-route metalness).
    """
    return _set_surface_param(mob, "reflectivity", reflectivity)


def set_roughness(mob, roughness):
    """Set the glossiness of a mirror mob: 0 is a sharp mirror, larger
    values blur its reflections. Only used by the Monte Carlo renderer
    (``set_samples_per_pixel`` > 1); the deterministic renderer reflects
    sharply. Call before the mob is spawned.

    Convenience shortcut for the ``roughness`` surface parameter that
    :meth:`~algan.mobs.mob.Mob.set_material` routes from a material's
    :meth:`Material.physical_surface_params`.
    """
    return _set_surface_param(mob, "roughness", roughness)


def set_refractive_index(mob, ior):
    """Make a mob refract light (glass) under the ray traced renderer.

    Convenience shortcut for the refractive-index surface parameter that
    :meth:`~algan.mobs.mob.Mob.set_material` routes from a transmissive
    :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial`
    (``transmission > 0``); use it directly to make a mob glass without
    configuring a full material.

    Attaches a (static) index of refraction to the mob and its descendants:
    1.0 means no bending (the default for unset mobs is 0, treated as "not
    refractive"), ~1.33 water, ~1.5 glass, ~2.4 diamond. The *transmitted*
    fraction of each ray (the part not reflected or absorbed -- so give the mob
    some transparency via its colour's opacity) is bent at the surface by
    Snell's law and traced onward, with total internal reflection handled.

    Only the **general wavefront** tracer implements refraction, so it is used
    when that path renders the batch (``set_wavefront(True)`` /
    ``ALGAN_WAVEFRONT=1``, or automatically for any batch that contains a
    refractive mob). The megakernel and Monte Carlo paths ignore it. Reflection
    takes priority over refraction if both are set on the same mob. Call before
    the mob is spawned.
    """
    return _set_surface_param(mob, "refractive_index", ior)


class RayTracedTrianglePrimitive(TrianglePrimitive):
    """Triangle batch rendered by ray tracing a spatio-temporal BVH."""

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))

    # Per-vertex surface parameters consumed by the trace kernels rather
    # than by a shader; popped from the shader kwargs (see set_reflectivity
    # and set_roughness).
    _surface_params = ("reflectivity", "roughness", "refractive_index")

    def __init__(self, corners=None, colors=None, opacity=1, normals=None,
                 perimeter_points=None, reverse_perimeter=False,
                 triangle_collection=None, glow=0, shader=None,
                 uvs=None, texture_map=None,
                 material_texture_map=None, material_texture_flags=0,
                 normal_texture_map=None,
                 **shader_kwargs):
        if triangle_collection is not None:
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader,
                             uvs=uvs, texture_map=texture_map,
                             material_texture_map=material_texture_map,
                             material_texture_flags=material_texture_flags,
                             normal_texture_map=normal_texture_map,
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
                # A registered (animatable) surface param on an *animated* mob
                # materializes per batch timestep ([T, ...]) while static
                # mobs' params stay single-frame; unify the time dims before
                # the cat (the kernels index time as ``f % T`` either way).
                values, _ = _unify_time(values, "surface param merge")
                setattr(self, name, unsquish(torch.cat(values, 1), -2, 3
                                             ).to(COMPUTING_DEFAULTS.render_device))
        else:
            params = {name: shader_kwargs.pop(name, None)
                      for name in self._surface_params}
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader=shader,
                             uvs=uvs, texture_map=texture_map,
                             material_texture_map=material_texture_map,
                             material_texture_flags=material_texture_flags,
                             normal_texture_map=normal_texture_map,
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

    def _shaded_per_fragment(self):
        """True when this primitive's hits are shaded per fragment in-kernel
        (deterministic renderer, fragment shading on, core lit material or a
        custom fragment pipeline) rather than baked per vertex -- in which case
        ``colors`` stays raw albedo."""
        shader = getattr(self, "shader", None)
        if getattr(shader, "_frag_pipeline_id", None) is not None:
            # A custom pipeline always shades in-kernel on the deterministic
            # renderer (fragment shading is forced on for such a scene).
            return SAMPLES_PER_PIXEL <= 1
        return (FRAGMENT_SHADING and SAMPLES_PER_PIXEL <= 1
                and _shader_is_core(shader))

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
                        *self.shader_param_values,
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

        mat_id = torch.full((1, N), _shader_material_id(shader),
                            dtype=torch.int32, device=device)
        pairs = []
        if _shader_is_core(shader):
            # The material's shader params, addressed by their real names (the
            # signature is not reliable: the ray tracer pops ``roughness`` out
            # of the shader kwargs into a surface param, see _surface_params).
            names = list(getattr(self, "shader_param_names", None) or [])
            values = list(getattr(self, "shader_param_values", None) or [])
            for name, value in zip(names, values):
                if name in _MAT_SLOTS and value is not None:
                    pairs.append((name, per_triangle(value)))
            # ``roughness`` (MeshStandardMaterial) was popped into self.roughness;
            # feed it into the roughness slot (ignored by the non-PBR branches).
            roughness = getattr(self, "roughness", None)
            if roughness is not None:
                pairs.append(("roughness", per_triangle(roughness)))
        Tm = max([1] + [v.shape[0] for _n, v in pairs])
        mat = torch.tensor(_MAT_DEFAULTS, device=device).view(
            1, 1, MAT_W).expand(Tm, N, MAT_W).contiguous()
        for name, v in pairs:
            start, width = _MAT_SLOTS[name]
            if v.shape[-1] != width:  # broadcast a scalar into a vector slot
                v = v.expand(*v.shape[:-1], width)
            mat[:, :, start:start + width] = v
        return mat_id.contiguous(), mat.contiguous()

    def _pack_frag_pipeline(self, shader, N, device, per_triangle):
        """Per-primitive pipeline id ``[1, N]`` and the custom-pipeline parameter
        block ``[Tm, N, W]`` for a mob with a fragment pipeline
        (:meth:`~algan.mobs.mob.Mob.set_fragment_shader`). Each stage's
        parameters occupy a contiguous slot range (the marker shader's
        ``_frag_param_layout`` maps attr name -> absolute slot); values are the
        materialised animated ``shader_param_values``, with defaults filling any
        slot whose attr is absent. A param whose name collides with a popped
        surface param (e.g. ``roughness``) is read back from that attribute, as
        the built-in :meth:`_pack_material` path does."""
        pid = int(shader._frag_pipeline_id)
        W = int(shader._frag_total_width)
        layout = shader._frag_param_layout  # list of (name, slot, width, default)
        mat_id = torch.full((1, N), pid, dtype=torch.int32, device=device)

        names = list(getattr(self, "shader_param_names", None) or [])
        values = list(getattr(self, "shader_param_values", None) or [])
        val_by_name = {n: v for n, v in zip(names, values)}

        # Default row (every slot is covered by exactly one layout entry).
        default_row = torch.zeros(W, dtype=torch.float32, device=device)
        for name, slot, width, default in layout:
            dv = torch.as_tensor(default, dtype=torch.float32,
                                 device=device).flatten()
            if dv.numel() == 1 and width > 1:
                dv = dv.expand(width)
            default_row[slot:slot + width] = dv[:width]

        pairs = []
        for name, slot, width, default in layout:
            v = val_by_name.get(name, None)
            if v is None and name in self._surface_params:
                v = getattr(self, name, None)  # popped into a surface attribute
            if v is not None:
                pairs.append((slot, width, per_triangle(v)))
        Tm = max([1] + [v.shape[0] for _s, _w, v in pairs])
        mat = default_row.view(1, 1, W).expand(Tm, N, W).contiguous()
        for slot, width, v in pairs:
            if v.shape[-1] != width:  # broadcast a scalar into a vector slot
                v = v.expand(*v.shape[:-1], width)
            mat[:, :, slot:slot + width] = v
        return mat_id.contiguous(), mat.contiguous()

    def _pack_surface_extra(self, error_context):
        """Per-primitive surface params [Te, N, 15]: the interleaved per-corner
        (reflectivity, roughness) pairs in columns 0-5 (unchanged; consumed by
        ``_triangle_extra`` in every kernel), followed by the per-corner
        refractive index in columns 6-8 (0 = not refractive; read by the
        wavefront's ``_corner_ior`` for the refraction path), followed by the
        per-corner glow strength in columns 9-11, followed by the per-corner
        glow radius in columns 12-14."""
        (reflectivity_e, roughness_e, ior_e), _ = _unify_time(
            [self.reflectivity.float(), self.roughness.float(),
             self.refractive_index.float()], error_context)
        n_t, n_p = reflectivity_e.shape[0], reflectivity_e.shape[1]
        refl_rough = torch.cat((reflectivity_e, roughness_e), -1).reshape(
            n_t, n_p, 6)
        ior = ior_e.reshape(n_t, n_p, 3)
        return torch.cat((refl_rough, ior), -1).contiguous()

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

        visible = (alpha.amax(-1) > MIN_ALPHA)
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
        accumulator in Monte Carlo mode, plus the wavefront's per-ray global
        state.
        """
        mc = SAMPLES_PER_PIXEL > 1
        self._rt_frame_bytes = int(
            camera.screen_width * camera.screen_height * 5 * 4
            * (2 if mc else 1))

        # Per ray: rs_ro/rd/acc/sca/int (~18 floats) + 6 KBUF-wide hit
        # buffers, all float/int32 (4 bytes), held for the whole chunk.
        self._rt_frame_bytes += int(
            camera.screen_width * camera.screen_height
            * (18 + 6 * KBUF) * 4)

    def project_to_screen(self, camera, light_sources):
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
        self._rt_tri_mat_id, self._rt_tri_mat = self._pack_material()
        self._rt_num_frames = camera.ray_origin.shape[0]

        if self.uvs is not None:
            self._rt_tri_uvs = self.uvs.float().reshape(self.uvs.shape[0], self.uvs.shape[1], 6).contiguous().to(COMPUTING_DEFAULTS.render_device)
            self._rt_texture_map = self.texture_map.float().contiguous() if self.texture_map is not None else None
            mtex = getattr(self, "material_texture_map", None)
            self._rt_material_texture = (mtex.float().contiguous()
                                         if mtex is not None else None)
            self._rt_material_flags = int(
                getattr(self, "material_texture_flags", 0) or 0)
            ntex = getattr(self, "normal_texture_map", None)
            self._rt_normal_texture = (ntex.float().contiguous()
                                       if ntex is not None else None)
        else:
            self._rt_tri_uvs = None
            self._rt_texture_map = None
            self._rt_material_texture = None
            self._rt_material_flags = 0
            self._rt_normal_texture = None

        self._pack_frame_visibility(corners.amin(-2), corners.amax(-2),
                                    self._rt_tri_colors,
                                    "triangle bounds/colors")

        # Everything the renderer needs now lives in the packed arrays;
        # release the unpacked geometry to halve resident GPU memory.
        self.corners = self.normals = None
        self.reflectivity = self.roughness = self.refractive_index = None
        self.colors = self.shader_param_values = None
        self.uvs = self.texture_map = None
        self.material_texture_map = self.normal_texture_map = None

        self._set_frame_buffer_bytes(camera)

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
        return self

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return KERNEL_SETTINGS.render_kernel(
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

    def project_to_screen(self, camera, light_sources):
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
        # Tight oriented bounding box per patch: the trace kernel tests it
        # before the matrix-pencil solve to reject the (many) candidates
        # whose loose axis-aligned leaf box the ray pierces but whose actual
        # (often thin, diagonal) patch it misses.
        self._rt_pn_obb = pn_obb(control_points).contiguous()
        self._rt_pn_norm = normals.reshape(
            normals.shape[0], normals.shape[1], 9).contiguous()
        self._rt_pn_extra = self._pack_surface_extra("pn surface params")
        self._rt_pn_colors = self.colors.float().contiguous()
        self._rt_pn_mat_id, self._rt_pn_mat = self._pack_material()
        self._rt_num_frames = camera.ray_origin.shape[0]

        # Texture maps (color / material / normal). PN patches have no kernel
        # argument budget left (the general wavefront shade kernel is at
        # Taichi's 64-arg ceiling), so unlike flat triangles the UVs and the
        # per-patch texture metadata are folded into the cold pn_extra array at
        # merge time (see _merge_scene); here we just stash the raw maps + UVs.
        if self.uvs is not None:
            self._rt_pn_uvs = self.uvs.float().reshape(
                self.uvs.shape[0], self.uvs.shape[1], 6).contiguous()
            self._rt_texture_map = (self.texture_map.float().contiguous()
                                    if self.texture_map is not None else None)
            mtex = getattr(self, "material_texture_map", None)
            self._rt_material_texture = (mtex.float().contiguous()
                                         if mtex is not None else None)
            self._rt_material_flags = int(
                getattr(self, "material_texture_flags", 0) or 0)
            ntex = getattr(self, "normal_texture_map", None)
            self._rt_normal_texture = (ntex.float().contiguous()
                                       if ntex is not None else None)
        else:
            self._rt_pn_uvs = None
            self._rt_texture_map = None
            self._rt_material_texture = None
            self._rt_material_flags = 0
            self._rt_normal_texture = None

        # The patch lies in the convex hull of its control points, so
        # the control net bounds it.
        self._pack_frame_visibility(control_points.amin(-2),
                                    control_points.amax(-2),
                                    self._rt_pn_colors,
                                    "pn bounds/colors")

        self.corners = self.normals = None
        self.reflectivity = self.roughness = self.refractive_index = None
        self.colors = self.shader_param_values = None
        self.uvs = self.texture_map = None
        self.material_texture_map = self.normal_texture_map = None

        self._set_frame_buffer_bytes(camera)

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
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

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))
    max_samples_per_segment = 512

    def project_to_screen(self, camera, light_sources):
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
        mc = SAMPLES_PER_PIXEL > 1
        self._rt_frame_bytes = int(
            camera.screen_width * camera.screen_height * 5 * 4
            * (2 if mc else 1))

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
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

        segment_lengths = (corners[..., 1:, :] - corners[..., :-1, :]).square().sum(-1).sum(-1)
        is_degenerate = segment_lengths < 1e-9
        edge_degenerate = torch.repeat_interleave(is_degenerate, num_samples, dim=1)

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

        Tn = nsi.shape[0]
        border_visible = torch.ones((Tn, V), device=device, dtype=torch.float32)
        closing_mask = nsi <= torch.arange(S, device=device).view(1, -1)
        seg_ends_expanded = seg_ends.view(1, -1).expand(Tn, -1)
        border_visible.scatter_(1, seg_ends_expanded, torch.where(closing_mask, torch.tensor(0.0, device=device),
                                                                  torch.tensor(1.0, device=device)))

        (verts_e, centers_e, basis_u_e, basis_v_e, next_start_e, edge_degenerate_e,
         border_visible_e), T_geo = _unify_time(
            [verts, centers, basis_u, basis_v, next_start.unsqueeze(-1), edge_degenerate.unsqueeze(-1),
             border_visible.unsqueeze(-1)],
            "bezier geometry")
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
        self._rt_edges = torch.cat((locals_uv, next_uv, border_visible_e.unsqueeze(-1)), -1).float().contiguous()
        self._rt_edges = torch.where(
            edge_degenerate_e.unsqueeze(-1),
            torch.tensor([1e9, 1e9, 1e9, 1e9, 0.0], device=device),
            self._rt_edges
        )

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
        glow_radius = self.glow_radius.float().reshape(
            self.glow_radius.shape[0], C)
        (centers_m, normals_m, bu_m, bv_m, b1_m, b2_m, bw_m, gw_m, gh_m, glow_radius_m), Tm = _unify_time(
            [centers, normals, basis_u, basis_v, basis1, basis2,
             border_width.unsqueeze(-1), grid_w.unsqueeze(-1),
             grid_h.unsqueeze(-1), glow_radius.unsqueeze(-1)], "bezier metadata")
        filled = torch.full((Tm, C, 1), 1.0 if self.filled else 0.0,
                            device=device)
        tex = torch.stack((
            (b1_m * bu_m).sum(-1), (b1_m * bv_m).sum(-1),
            (b2_m * bu_m).sum(-1), (b2_m * bv_m).sum(-1)), -1).nan_to_num_()
        self._rt_circuit_meta = torch.cat(
            (centers_m, normals_m, bu_m, bv_m, bw_m, filled, gw_m, gh_m, tex, glow_radius_m),
            -1).contiguous()

        colors = self.colors.float()
        if colors.dim() == 3:  # plain fills: a 1x1 "texture" grid
            colors = colors.unsqueeze(-2)
        self._rt_circuit_colors = colors.contiguous().as_subclass(Color)
        self._rt_circuit_border_colors = self.border_color.float().contiguous().as_subclass(Color)
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
            1, idx, seg_lo, "amin", include_self=True)
        hi = torch.full((Tb, C, 3), EMPTY_HI, device=device).scatter_reduce_(
            1, idx, seg_hi, "amax", include_self=True)

        fill_alpha = self._rt_circuit_colors.opacity.squeeze(-1).amax(-1)  # over texture
        fill_min = self._rt_circuit_colors.opacity.squeeze(-1).amin(-1)
        if not self.filled:
            fill_alpha = torch.zeros_like(fill_alpha)
        border_alpha = self._rt_circuit_border_colors.opacity.squeeze(-1)
        border_on = self._rt_border_width > 1e-3
        glow_alpha = self._rt_circuit_colors[..., 3].amax(-1)
        visible = (fill_alpha > MIN_ALPHA) | (
                (border_alpha > MIN_ALPHA) & border_on) | (glow_alpha > 0.0)
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
        # world units at each circuit's distance from the camera, plus the glow radius.
        b1_norm = sb[:, 1].norm(p=2, dim=-1)
        screen_dist = (sp - cam_o).norm(p=2, dim=-1)
        pixel_world_scale = 2.0 / (screen_h * b1_norm * screen_dist).clamp_min(1e-12)
        centers = self._rt_circuit_meta[..., :3]
        dist = (centers - cam_o.view(-1, 1, 3)).norm(p=2, dim=-1)
        world_per_px = (pixel_world_scale.view(-1, 1) * dist).amax(0)

        glow_rad = torch.where(glow_alpha > 0.0, self.glow_radius.squeeze(-1), 0.0)
        glow_rad_max = glow_rad.amax(0)

        inflate = (self._rt_border_width.amax(0) + 1.0) * world_per_px + glow_rad_max
        self._rt_frame_lo = (lo - inflate.view(1, -1, 1)).contiguous()
        self._rt_frame_hi = (hi + inflate.view(1, -1, 1)).contiguous()

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return KERNEL_SETTINGS.render_kernel(
            primitives, scene, screen_width, screen_height, time_start,
            time_end, background_color, transparent_background, *args,
            **kwargs)