from algan.settings import SETTINGS
import torch.nn.functional as F

from algan.constants.spatial import OUT, RIGHT
from algan.settings.video_settings import PREVIEW
from algan.geometry.geometry import rotate_vector_around_axis
from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import *
from algan.animatable_base.mob import Mob

from algan.animatable_base.animatable import animated_function
from algan.utils.tensor_utils import *
from algan.settings.renderer_settings import RENDERER_REGISTRY
from algan.rendering.raytracing.utils import _unify_time


# Three.js's fixed dielectric F0 = 0.04 corresponds to IOR 1.5; MeshStandard
# has no ``ior`` of its own, so that is the default a circuit falls back to.
DIELECTRIC_IOR = 1.5


def _circuit_ior(ior, metalness):
    """Pack a material's IOR into a circuit's transport channel.

    Mirrors the triangle primitive's ``_derive_material_surface_params``: an
    unsigned magnitude feeding dielectric F0. Whether the circuit transmits is
    carried by the separate ``transmission`` channel, not by this one's sign.
    Non-PBR circuits (metalness < 0) get 0: inert, since their reflectance is 0
    anyway.
    """
    return torch.where(metalness >= 0.0, ior.abs(), torch.zeros_like(ior))


def _circuit_location_and_basis(control_points):
    """Return the same local frame used by a standalone bezier circuit."""
    control_points = control_points.reshape(-1, 3)
    mn = control_points.amin(-2)
    mx = control_points.amax(-2)
    location = (mn + mx) * 0.5
    if (mx - mn).norm(p=2, dim=-1) <= 1e-6:
        basis = squish(torch.eye(3, device=control_points.device,
                                  dtype=control_points.dtype))
        return location, basis.reshape(-1)

    disps = control_points - location
    dists = disps.norm(p=2, dim=-1, keepdim=True)
    first_basis = disps[
        ..., dists.argmax(-2, keepdim=True).squeeze(), :
    ].unsqueeze(-2)
    if first_basis.norm(p=2, dim=-1) <= 1e-4:
        first_basis = RIGHT.to(control_points) * 1e-4
    first_basis_n = F.normalize(first_basis, p=2, dim=-1)

    planar_disps = disps - dot_product(disps, first_basis_n) * first_basis_n
    dists = planar_disps.norm(p=2, dim=-1, keepdim=True)
    second_basis = planar_disps[
        ..., dists.argmax(-2, keepdim=True).squeeze(), :
    ].unsqueeze(-2)
    if second_basis.norm(p=2, dim=-1) <= 1e-4:
        second_basis = rotate_vector_around_axis(first_basis, 90, OUT, -1)
    second_basis = (
        second_basis
        * first_basis.norm(p=2, dim=-1, keepdim=True)
        / second_basis.norm(p=2, dim=-1, keepdim=True)
    )
    third_basis_n = F.normalize(
        broadcast_cross_product(first_basis_n, second_basis), p=2, dim=-1
    )
    basis = torch.cat((first_basis, second_basis, third_basis_n), -1)
    return location, basis.reshape(-1)



class BezierCircuitCubic(Mob):
    def __init__(
        self,
        control_points,
        normals=None,
        border_width=5,
        border_color=WHITE,
        portion_of_curve_drawn=1.0,
        filled=True,
        add_texture_grid=True,
        texture_grid_size=1,
        empty=False,
        **kwargs,
    ):
        self.num_bezier_parameters = 4
        control_points = control_points.view(-1, control_points.shape[-1])
        """ucp = unsquish(control_points, -2, self.num_bezier_parameters)
        start_points = ucp[...,:1,:]
        end_points = ucp[...,-1:,:]
        circuit_start_mask = ((start_points - end_points.roll(1, -3)).norm(p=2, dim=-1, keepdim=True) > 1e-5)
        circuit_end_mask = ((end_points - start_points.roll(-1, -3)).norm(p=2, dim=-1, keepdim=True) > 1e-5)

        circuit_start_inds = circuit_start_mask.view(-1).nonzero()
        circuit_end_inds = circuit_end_mask.view(-1).nonzero()
        out = []

        def get_connecting_bezier(start, end):
            return torch.stack([start * (1-a) + a * end for a in torch.linspace(0,1, self.num_bezier_parameters)]), -2

        for s, e in zip(circuit_start_inds, circuit_end_inds):
            out.append(ucp[..., :e, :, :])
            n = (e+1) % ucp.shape[-3]
            out.append(get_connecting_bezier(ucp[..., e:e+1, -1,:], ucp[..., n:n+1, 0,:]))
        if len(out) > 0:
            out.append(ucp[..., e+1:,:,:])
            ucp = torch.cat(out, dim=-3)
        control_points = squish(ucp, -3, -2)"""

        kwargs2 = {k: v for k, v in kwargs.items()}

        if "color" in kwargs2:
            kwargs2["color"] = (
                kwargs2["color"].reshape(-1, kwargs2["color"].shape[-1]).mean(-2)
            )
        if normals is not None:
            normals = normals.reshape(-1, 3)
        kwargs2["location"], kwargs2["basis"] = _circuit_location_and_basis(
            control_points
        )

        self.grid_width = self.grid_height = 1
        self.num_texture_points = 0
        first_basis = kwargs2["basis"][..., :3]
        second_basis = kwargs2["basis"][..., 3:6]
        self.first_basis = first_basis
        self.second_basis = second_basis

        super().__init__(**kwargs2)
        kwargs["scene"] = self.scene
        self.register_attrs_as_animatable(
            ["border_width", "border_color", "portion_of_curve_drawn"],
            BezierCircuitCubic,
        )
        self.filled = filled
        self.empty = empty
        if self.empty:
            self.color = self.color.as_subclass(Color).set_opacity(0)

        texture_triangle_vertices = self.location.squeeze(0)
        if add_texture_grid:
            aspect_ratio = second_basis.norm(p=2, dim=-1) / first_basis.norm(
                p=2, dim=-1
            )

            a1 = torch.linspace(-1, 1, texture_grid_size).view(-1, 1, 1) * (1 + 1e-5)
            a2 = torch.linspace(
                -1, 1, int((texture_grid_size * aspect_ratio).round())
            ).view(1, -1, 1) * (1 + 1e-5)
            texture_grid_points = (a1 * first_basis + a2 * second_basis) + self.location
            texture_triangle_vertices = texture_grid_points
            self.grid_width = texture_triangle_vertices.shape[-2]
            self.grid_height = texture_triangle_vertices.shape[-3]
            texture_triangle_vertices = texture_triangle_vertices.reshape(
                -1, texture_triangle_vertices.shape[-1]
            )
            self.num_texture_points = texture_triangle_vertices.shape[-2]

            # control_points = torch.cat((control_points, texture_triangle_vertices), -2)
        self.border_width = cast_to_tensor(border_width)
        self.border_color = cast_to_tensor(
            border_color if not self.empty else border_color.set_opacity(0)
        )
        kwargs["color"] = self.color if self.filled else self.border_color
        with Off(animation_manager=self.animation_manager):
            self.texture_points = Mob(texture_triangle_vertices, **kwargs)
            self.texture_points.exclude_from_boundary = True
            self.texture_points.is_primitive = True
            self.add_children(self.texture_points)

            self.control_points = Mob(control_points, **kwargs)
            self.control_points.is_primitive = True
            self.add_children(self.control_points)
            self.control_points.num_points_per_object = 4
            self.components = [self.texture_points, self.control_points]

        self.normals = normals
        self.is_primitive = True
        self.render_primitive = RENDERER_REGISTRY.bezier_circuit_primitive

    @classmethod
    def from_batches(cls, control_point_batches, *args, **kwargs):
        """Build many independently indexable circuits without per-circuit mobs.

        ``control_point_batches`` contains one cubic-bezier tensor per logical
        object.  Geometry is concatenated once while ``parent_batch_sizes``
        retains the control-point boundaries used by rendering and indexed
        views.
        """
        batches = [
            cast_to_tensor(points).reshape(-1, 3)
            for points in control_point_batches
        ]
        if not batches:
            raise ValueError("from_batches requires at least one bezier circuit")
        point_counts = torch.tensor(
            [len(points) for points in batches], dtype=torch.long
        )
        if bool((point_counts % 4 != 0).any()):
            raise ValueError(
                "every cubic bezier circuit must contain a multiple of 4 points"
            )

        mob = cls(torch.cat(batches, -2), *args, **kwargs)
        locations, bases = zip(
            *[_circuit_location_and_basis(points) for points in batches]
        )
        locations = torch.stack(locations, -2).unsqueeze(0)
        bases = torch.stack(bases, -2).unsqueeze(0)
        count = len(batches)

        with Off(record_funcs=False, record_attr_modifications=False, animation_manager=mob.animation_manager):
            for attr in mob.animatable_attrs:
                try:
                    value = getattr(mob, attr)
                except AttributeError:
                    continue
                if attr == "location":
                    value = locations
                elif attr == "basis":
                    value = bases
                elif value.shape[-2] == 1:
                    value = value.expand(
                        *value.shape[:-2], count, value.shape[-1]
                    ).contiguous()
                mob.setattr_and_rebatch_without_record(attr, value)

            mob.texture_points.parent_batch_sizes = torch.ones(
                count, dtype=torch.long
            )
            for attr in mob.texture_points.animatable_attrs:
                try:
                    value = getattr(mob.texture_points, attr)
                except AttributeError:
                    continue
                if attr == "location":
                    value = locations
                elif value.shape[-2] == 1:
                    value = value.expand(
                        *value.shape[:-2], count, value.shape[-1]
                    ).contiguous()
                mob.texture_points.setattr_and_rebatch_without_record(attr, value)

            mob.control_points.parent_batch_sizes = point_counts
            mob.parent_batch_sizes = torch.tensor((count,), dtype=torch.long)
            mob.singleton_batch_indexing = True
        return mob

    def get_animatable_attrs(self):
        return {"border_width", "border_color"}.union(
            super().get_animatable_attrs()
        )

    def get_default_color(self):
        return PURPLE

    def get_memory_used_per_timestep(self):
        # Called for every circuit every render batch just to size batches;
        # the shape reads below go through the animated-attribute machinery,
        # so cache the result against the global structure version (row
        # re-allocation bumps it).
        from algan.animation_timeline.timeline import STRUCTURE_VERSION

        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == STRUCTURE_VERSION[0]:
            return cache[1]
        n_ctrl = self.control_points.location.shape[-2]
        n_tex = self.texture_points.location.shape[-2]
        n_loc = self.location.shape[-2]
        n_segments = max(n_ctrl // 4, 1)  # cubic beziers have 4 control points each
        # Animation state: control points (3 floats), texture (5), location (6).
        animation_bytes = (n_ctrl * 3 + n_tex * 5 + n_loc * 6) * 4
        # Primitive output: control point corners, colors, normals, border data.
        primitive_bytes = n_segments * 4 * 3 * 4 + n_tex * 5 * 4 + n_loc * 12
        # Sampled edges, metadata and the content-dependent STBVH are charged
        # exactly by the final scene upload instead of guessed here (the old
        # fixed 100-sample estimate was wrong for the actual 1..512 range).
        result = int(animation_bytes + primitive_bytes)
        self._memory_per_timestep_cache = (STRUCTURE_VERSION[0], result)
        return result

    def get_render_primitives(self):
        if self.empty:
            return None
        # Derive transport directly from the material shader parameters. A
        # negative metalness sentinel marks non-PBR materials; Standard and
        # Physical materials expose metalness/roughness as animatable attrs.
        surface_template = self.opacity[..., :1]

        def material_param(name, default):
            if name in self.animatable_attrs:
                return getattr(self, name)
            return torch.full_like(surface_template, default)

        metalness = material_param("metalness", -1.0)
        roughness = material_param("roughness", 0.0)
        # Opacity is coverage and transmission is transparency: independent
        # channels, never folded together (see _derive_material_surface_params).
        transmission = material_param("transmission", 0.0).clamp(0.0, 1.0)
        ior = _circuit_ior(material_param("ior", DIELECTRIC_IOR), metalness)

        vars = broadcast_all(
            [
                self.opacity,
                self.basis,
                self.glow,
                self.border_width
                * self.scene.video_settings.resolution[1] * self.scene.video_settings.anti_alias_level
                / (PREVIEW.resolution[1] * 2),
                self.border_color,
                metalness,
                roughness,
                ior,
                transmission,
            ],
            ignored_dims=[-1],
        )
        num_control_points = 4  # cubic beziers
        if self.control_points.parent_batch_sizes is None:
            return self._get_render_primitives(
                unsquish(self.control_points.location, -2, num_control_points),
                self.texture_points.color,
                self.location,
                self.basis,
                *vars,
            )
        x = self.control_points.location
        tpc = self.texture_points.color
        num_segments_per_circuit = (
            self.control_points.parent_batch_sizes // num_control_points
        )
        return self._get_render_primitives(
            unsquish((x), -2, num_control_points),
            (tpc),
            self.location,
            self.basis,
            *vars,
            num_segments_per_circuit,
        )

    def _get_render_primitives(
        self, x, tpc, loc, basis, o, n, g, bw, bc, reflectivity,
        roughness, refractive_index, transmission,
        num_segments_per_circuit=None
    ):
        num_control_points = 4  # cubic beziers
        # x = unsquish(x, -2, num_control_points)
        # assert x.shape == [*, N, num_control_points, 3], where N is number of bezier segments.
        start_points = x[..., :1, :]
        end_points = x[..., -1:, :]

        # We allow for rendering circuits with holes,
        # we treat beziers which don't start at the previous one's end as marking the start of a new circuit (i.e. a hole).
        circuit_start_mask = (start_points - end_points.roll(1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5
        circuit_end_mask = (end_points - start_points.roll(-1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5

        inds = torch.arange(x.shape[-3], device=x.device).view(-1, 1, 1)
        circuit_start_inds = torch.where(circuit_start_mask, inds, 0)
        circuit_start_inds = torch.cummax(circuit_start_inds, -3)[0]
        # circuit_start_inds now contains the index of the start of the current index's circuit.

        next_segment_inds = (inds + 1) % x.shape[-3]
        # If the current ind is the end of the circuit, then the next segment is the first ind of this circuit, otherwise it is the next ind.
        next_segment_inds = torch.where(
            circuit_end_mask, circuit_start_inds, next_segment_inds
        )
        # We subtract inds so that each ind is represented as an offset from the current ind.
        # This way, we can concatenate together offsets from different objects, and then just add a torch.arange during rendering
        # to recover the index in the new concatenated tensor.
        next_segment_inds_offset = next_segment_inds - inds

        if num_segments_per_circuit is None:
            starting_inds = circuit_start_mask[0, :, 0, 0].nonzero()[:, 0]
            num_segments_per_circuit = []
            if len(starting_inds) == 0:
                num_segments_per_circuit.append(
                    torch.tensor(
                        (circuit_start_mask.shape[-3],),
                        device=next_segment_inds.device,
                        dtype=next_segment_inds.dtype,
                    ).squeeze()
                )
            else:
                for i in range(len(starting_inds)):
                    num_segments_per_circuit.append(
                        (
                            starting_inds[(i + 1)]
                            if (i + 1) < len(starting_inds)
                            else circuit_start_mask.shape[-3]
                        )
                        - starting_inds[i]
                    )
            #num_segments_per_circuit = torch.stack(num_segments_per_circuit, 0)
            num_segments_per_circuit = torch.tensor(
                [x.shape[-3]], device=x.device, dtype=torch.long
            )
            c = tpc.unsqueeze(-3)
            if self.num_texture_points > c.shape[-2]:
                c = c.expand([-1, -1, self.num_texture_points, -1])
        else:
            c = unsquish(tpc, -2, self.num_texture_points)

        prim = self.render_primitive(
            x,
            next_segment_inds_offset,
            num_segments_per_circuit,
            c,
            o,
            basis[..., -3:],
            bw,
            bc,
            loc,
            cast_to_tensor(self.grid_width).expand(-1, loc.shape[1], -1),
            cast_to_tensor(self.grid_height).expand(-1, loc.shape[1], -1),
            basis[..., :3],
            basis[..., 3:6],
            glow=g,
            num_texture_points=self.num_texture_points,
            filled=self.filled,
            reflectivity=reflectivity,
            roughness=roughness,
            refractive_index=refractive_index,
            transmission=transmission,
        )
        prim.num_texture_points = self.num_texture_points
        return prim

    @animated_function(animated_args={"t": 0.0})
    def draw(self, t=1.0):
        self._original_control_points = self.control_points.location.clone()
        num_frames = self.control_points.location.shape[0]
        total_control_points = self._original_control_points.shape[-2]
        points = self._original_control_points.expand(num_frames, -1, -1)


        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        # Reshape points to (num_frames, num_mobs, N_per_mob, 4, 3)
        points_reshaped = points.view(num_frames, num_mobs, N_per_mob, 4, 3)

        # Ensure t is a tensor and has shape (num_frames, num_mobs, 1, 1)
        t = cast_to_tensor(t).to(points.device)
        while t.dim() < 3:
            t = t.unsqueeze(0)
        if t.shape[1] != num_mobs:
            t = t.expand(-1, num_mobs, -1)
        t = t.unsqueeze(-1) # (num_frames, num_mobs, 1, 1)

        # Calculate local b parameters
        inds_local = torch.arange(N_per_mob, device=points.device, dtype=points.dtype)
        b = (N_per_mob * t - inds_local.view(1, 1, N_per_mob, 1)).clamp(0.0, 1.0) # (num_frames, num_mobs, N_per_mob, 1, 1)

        # Portion matrix coefficients for each segment
        mb = 1.0 - b
        b2 = b * b
        mb2 = mb * mb
        b3 = b2 * b
        mb3 = mb2 * mb

        # Construct portion_matrix of shape (num_frames, num_mobs, N_per_mob, 4, 4)
        portion_matrix = torch.zeros((num_frames, num_mobs, N_per_mob, 4, 4), device=points.device, dtype=points.dtype)
        portion_matrix[..., 0, 0] = 1.0

        portion_matrix[..., 1, 0] = mb.squeeze(-1)
        portion_matrix[..., 1, 1] = b.squeeze(-1)

        portion_matrix[..., 2, 0] = mb2.squeeze(-1)
        portion_matrix[..., 2, 1] = 2.0 * mb.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 2, 2] = b2.squeeze(-1)

        portion_matrix[..., 3, 0] = mb3.squeeze(-1)
        portion_matrix[..., 3, 1] = 3.0 * mb2.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 3, 2] = 3.0 * mb.squeeze(-1) * b2.squeeze(-1)
        portion_matrix[..., 3, 3] = b3.squeeze(-1)

        # Compute new control points
        new_points = torch.matmul(portion_matrix, points_reshaped)

        # Reshape back to (num_frames, total_control_points, 3)
        new_points = new_points.view(num_frames, total_control_points, 3)

        # Set the control points location absolute
        self.control_points.location = new_points
        return self

    def set_control_points_to_partial(self, full_control_points, start_t, end_t):
        full_control_points = cast_to_tensor(full_control_points)
        start_t = cast_to_tensor(start_t).to(full_control_points)
        end_t = cast_to_tensor(end_t).to(full_control_points)

        def frame_values(value, name):
            if value.numel() == 1:
                return value.reshape(1)
            values = value.reshape(value.shape[0], -1)
            if values.shape[1] != 1:
                raise ValueError(f"{name} must contain one value per frame")
            return values[:, 0]

        start_t = frame_values(start_t, "start_t")
        end_t = frame_values(end_t, "end_t")
        num_frames = max(
            full_control_points.shape[0], start_t.numel(), end_t.numel()
        )
        if full_control_points.shape[0] == 1:
            full_control_points = full_control_points.expand(
                num_frames, -1, -1
            )
        elif full_control_points.shape[0] != num_frames:
            raise ValueError(
                "full_control_points must have one row or one row per frame"
            )
        if start_t.numel() == 1:
            start_t = start_t.expand(num_frames)
        elif start_t.numel() != num_frames:
            raise ValueError("start_t must have one value per frame")
        if end_t.numel() == 1:
            end_t = end_t.expand(num_frames)
        elif end_t.numel() != num_frames:
            raise ValueError("end_t must have one value per frame")

        total_control_points = full_control_points.shape[-2]

        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        points_reshaped = full_control_points.view(
            num_frames, num_mobs, N_per_mob, 4, 3
        )

        j = torch.arange(
            N_per_mob,
            device=full_control_points.device,
            dtype=full_control_points.dtype,
        ).view(1, 1, N_per_mob, 1, 1)
        s_start = j / N_per_mob
        s_end = (j + 1) / N_per_mob

        a = torch.clamp(start_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)
        b = torch.clamp(end_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)

        local_a = (a - s_start) * N_per_mob
        local_b = (b - s_start) * N_per_mob

        P0 = points_reshaped[..., 0, :]
        P1 = points_reshaped[..., 1, :]
        P2 = points_reshaped[..., 2, :]
        P3 = points_reshaped[..., 3, :]

        b_t = local_b.squeeze(-1)
        mb_t = 1.0 - b_t

        Q0 = P0
        Q1 = mb_t * P0 + b_t * P1
        Q2 = mb_t**2 * P0 + 2.0 * mb_t * b_t * P1 + b_t**2 * P2
        Q3 = (
            mb_t**3 * P0
            + 3.0 * mb_t**2 * b_t * P1
            + 3.0 * mb_t * b_t**2 * P2
            + b_t**3 * P3
        )

        u = torch.where(
            b_t > 1e-6, local_a.squeeze(-1) / b_t, torch.zeros_like(b_t)
        )
        u = torch.clamp(u, 0.0, 1.0)
        mu = 1.0 - u

        R3 = Q3
        R2 = u * Q3 + mu * Q2
        R1 = u**2 * Q3 + 2.0 * u * mu * Q2 + mu**2 * Q1
        R0 = u**3 * Q3 + 3.0 * u**2 * mu * Q2 + 3.0 * u * mu**2 * Q1 + mu**3 * Q0

        new_points = torch.stack([R0, R1, R2, R3], -2).view(
            num_frames, total_control_points, 3
        )
        self.control_points.location = new_points
        return self





class BezierCurveCubic(BezierCircuitCubic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, filled=False, **kwargs)


def build_render_primitives_batched(actors, scene):
    """Build the merged (collection-level) bezier render primitive for
    ``actors`` in one vectorized pass.

    Byte-identical replacement for calling ``get_render_primitives()`` on
    every actor and concatenating the per-actor primitives through
    ``BezierCircuitPrimitive(triangle_collection=...)``: each attribute is
    read from its timeline once for the whole group (contiguous rows read as
    a single slice), and the per-segment circuit topology (subpath start/end
    masks, next-segment indices) is computed with per-actor index maps that
    reproduce each actor's local ``roll``/``cummax`` wrap-around semantics.

    Callers must guarantee (see ``RenderLoopMixin._is_batchable_bezier`` and
    ``_build_deferred_beziers``): stock ``BezierCircuitCubic`` build methods,
    not ``empty``, un-batched control points, singleton rows for the scalar
    attributes, and uniform ``num_texture_points`` / ``filled`` /
    texture-color row count / primitive class across the group.
    """
    from algan.animation_timeline.timeline import RowRanges

    timeline = scene.timeline_manager
    first = actors[0]
    ntp = first.num_texture_points
    M = len(actors)

    def read(attr, mobs):
        tl = timeline.attr_to_timeline[attr]
        # Merge the per-mob cached [begin, end) runs (ranges_for) instead of
        # rebuilding them from the index tensors: this is called every frame
        # batch, and tensor->int conversion per mob dominates otherwise.
        pairs = []
        for m in mobs:
            r = tl.ranges_for(m.id)
            if r.pairs is None:  # non-contiguous rows (defensive)
                return tl.get(RowRanges(None, tensor=torch.cat(
                    [tl.mob_id_to_inds[mm.id] for mm in mobs])))
            for b, e in r.pairs:
                if pairs and pairs[-1][1] == b:
                    pairs[-1] = (pairs[-1][0], e)
                else:
                    pairs.append((b, e))
        return tl.get(RowRanges(pairs))

    # --- batched attribute reads (mirrors the per-actor property reads and
    # the ``vars`` broadcast in get_render_primitives) ---
    o = read("opacity", actors)

    def read_optional_material(attr, default):
        values = []
        for actor in actors:
            if attr in actor.animatable_attrs:
                tl = timeline.attr_to_timeline[attr]
                values.append(tl.get(tl.ranges_for(actor.id)))
            else:
                values.append(torch.full_like(o[:, :1, :1], default))
        values, _ = _unify_time(values, f"bezier {attr} merge")
        return torch.cat(values, 1)

    reflectivity = read_optional_material("metalness", -1.0)
    roughness = read_optional_material("roughness", 0.0)
    # Opacity is coverage, transmission is transparency: independent channels
    # (see _derive_material_surface_params). ``o`` is left alone.
    transmission = read_optional_material("transmission", 0.0).clamp(0.0, 1.0)
    refractive_index = _circuit_ior(
        read_optional_material("ior", DIELECTRIC_IOR), reflectivity)
    basis = read("basis", actors)
    g = read("glow", actors)
    bw = read("border_width", actors) * (
        scene.video_settings.resolution[1] * scene.video_settings.anti_alias_level
        / (PREVIEW.resolution[1] * 2))
    bc = read("border_color", actors)
    loc = read("location", actors)
    o, basis, g, bw, bc = broadcast_all([o, basis, g, bw, bc],
                                            ignored_dims=[-1])
    cp = read("location", [a.control_points for a in actors])
    tpc = read("color", [a.texture_points for a in actors])

    # --- circuit topology (mirrors _get_render_primitives) ---
    loc_inds = timeline.attr_to_timeline["location"].mob_id_to_inds
    seg_counts = torch.tensor(
        [loc_inds[a.control_points.id].numel() // 4 for a in actors],
        dtype=torch.long)
    x = unsquish(cp, -2, 4)  # [T, S_total, 4, 3]
    S_tot = x.shape[-3]
    seg_offsets = seg_counts.cumsum(0) - seg_counts
    mob_of_seg = torch.repeat_interleave(torch.arange(M), seg_counts)
    off_of_seg = seg_offsets[mob_of_seg]
    gidx = torch.arange(S_tot)
    local = gidx - off_of_seg
    last_local = seg_counts[mob_of_seg] - 1

    start_points = x[..., :1, :]
    end_points = x[..., -1:, :]
    # Per-actor wrap-around neighbours: each actor's own roll(+-1, -3).
    prev_idx = torch.where(local == 0, off_of_seg + last_local, gidx - 1)
    next_idx = torch.where(local == last_local, off_of_seg, gidx + 1)
    circuit_start_mask = (
        start_points - end_points.index_select(-3, prev_idx)
    ).norm(p=2, dim=-1, keepdim=True) > 1e-5
    circuit_end_mask = (
        end_points - start_points.index_select(-3, next_idx)
    ).norm(p=2, dim=-1, keepdim=True) > 1e-5

    local_col = local.view(-1, 1, 1)
    off_col = off_of_seg.view(-1, 1, 1)
    # The per-actor where(mask, local_ind, 0) + cummax scan, run in global
    # index space: candidate values are per-actor monotone blocks (every
    # actor's candidates are >= its offset and below the next actor's), so
    # one global cummax restarts cleanly at every actor boundary.
    circuit_start_inds = torch.where(circuit_start_mask, local_col + off_col,
                                     off_col)
    circuit_start_inds = torch.cummax(circuit_start_inds, -3)[0] - off_col
    next_segment_inds = torch.where(
        local == last_local, torch.zeros_like(local), local + 1).view(-1, 1, 1)
    next_segment_inds = torch.where(circuit_end_mask, circuit_start_inds,
                                    next_segment_inds)
    next_segment_inds_offset = next_segment_inds - local_col  # [T, S, 1, 1]

    # --- texture colors (mirrors the ``c`` construction) ---
    c = unsquish(tpc, -2, tpc.shape[-2] // M)  # [T, M, P, 5]
    if ntp > c.shape[-2]:
        c = c.expand([-1, -1, ntp, -1])

    # --- per-primitive color/border math (mirrors
    # BezierCircuitPrimitive.__init__'s scalar path) ---
    normals = basis[..., -3:]
    bc, o, g = broadcast_all([bc, o, g], ignored_dims=[-1])
    colors = c.clone()
    colors[..., -2:-1] += g.unsqueeze(-2)
    colors[..., -1:] *= o.unsqueeze(-2)
    bc[..., -2:-1] += g
    bc[..., -1:] *= o

    # --- collection-level assembly (mirrors the triangle_collection branch
    # of BezierCircuitPrimitive.__init__) ---
    # Keep the deferred mega-primitive on the materialized animation/source
    # device.  The prefetch worker must not upload the next batch while the
    # current one occupies the render device; upload happens at the managed
    # render-memory boundary.
    device = x.device
    cls = first.render_primitive
    mega = cls.__new__(cls)
    # The ray tracer interprets this legacy density setting as the maximum
    # screen-space curve-to-chord error in pixels.
    mega.num_pixels_per_sample = 1
    mega.num_bezier_parameters = 4
    mega.num_texture_points = ntp
    mega.filled = first.filled
    mega.num_segments_per_object = seg_counts.to(device)
    mega.corners = x.to(device)
    cols = colors.to(device)
    if ntp == 0:
        cols = cols.squeeze(-2)
    mega.next_segment_inds = (
        next_segment_inds_offset.to(device)
        + torch.arange(S_tot, device=device).view(-1, 1, 1))
    mega.normals = normals.to(device)
    mega.border_width = bw.to(device)
    mega.border_color = bc.to(device)

    T = loc.shape[0]

    def per_actor_int(vals):
        return (torch.tensor([float(v) for v in vals]).view(1, M, 1).int()
                .expand(T, -1, -1))

    mega.mob_center = loc.to(device)
    # NB: the triangle_collection constructor assigns each primitive's
    # grid_height into the collection's .grid_width and vice versa;
    # reproduced as-is for byte-identity.
    mega.grid_width = per_actor_int([a.grid_height for a in actors]).to(device)
    mega.grid_height = per_actor_int([a.grid_width for a in actors]).to(device)
    mega.basis1 = basis[..., :3].to(device)
    mega.basis2 = basis[..., 3:6].to(device)
    mega.reflectivity = reflectivity.to(device)
    mega.roughness = roughness.to(device)
    mega.refractive_index = refractive_index.to(device)
    mega.transmission = transmission.to(device)
    if ntp > 0:
        cols = cols[..., -ntp:, :]
    mega.colors = cols
    return mega
