import gc

import torch
from algan.constants.spatial import OUT
from algan import RIGHT, PREVIEW, rotate_vector_around_axis
from algan.animation.animation_contexts import Off
from algan.logging.logger import LoggerManager
from algan.mobs.renderable import Renderable
from algan.constants.color import *
from algan.geometry.geometry import get_roots_of_quadratic, get_orthonormal_vector
from algan.mobs.mob import Mob
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
)

from algan.animation.animatable import animated_function
from algan.utils.tensor_utils import *



class BezierCircuitCubic(Renderable):
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
        mn = control_points.reshape(-1, 3).amin(-2)
        mx = control_points.reshape(-1, 3).amax(-2)
        kwargs2["location"] = (mn + mx) * 0.5

        self.grid_width = self.grid_height = 1
        self.num_texture_points = 0
        if (mx - mn).norm(p=2, dim=-1) <= 1e-6:
            kwargs2["basis"] = squish(torch.eye(3))
            first_basis = kwargs2["basis"][..., :3]
            second_basis = kwargs2["basis"][..., 3:6]
        else:
            disps = control_points - kwargs2["location"]
            dists = (disps).norm(p=2, dim=-1, keepdim=True)
            first_basis = disps[
                ..., dists.argmax(-2, keepdim=True).squeeze(), :
            ].unsqueeze(-2)
            if first_basis.norm(p=2, dim=-1) <= 1e-4:
                first_basis = RIGHT * 1e-4
            self.first_basis = first_basis
            first_basis_n = F.normalize(first_basis, p=2, dim=-1)

            disps = disps - dot_product(disps, first_basis_n) * first_basis_n

            dists = (disps).norm(p=2, dim=-1, keepdim=True)
            second_basis = disps[
                ..., dists.argmax(-2, keepdim=True).squeeze(), :
            ].unsqueeze(-2)
            if second_basis.norm(p=2, dim=-1) <= 1e-4:
                second_basis = rotate_vector_around_axis(first_basis, 90, OUT, -1)
            second_basis = (
                second_basis
                * first_basis.norm(p=2, dim=-1, keepdim=True)
                / second_basis.norm(p=2, dim=-1, keepdim=True)
            )
            self.second_basis = second_basis
            third_basis_n = F.normalize(
                broadcast_cross_product(first_basis_n, second_basis), p=2, dim=-1
            )
            kwargs2["basis"] = torch.cat((first_basis, second_basis, third_basis_n), -1)

        super().__init__(**kwargs2)
        self.register_attrs_as_animatable(
            ["border_width", "border_color", "portion_of_curve_drawn"],
            BezierCircuitCubic,
        )
        self.filled = filled
        self.empty = empty
        if self.empty:
            self.color = self.color.set_opacity(0)

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
        with Off():
            self.texture_points = Mob(texture_triangle_vertices, **kwargs)
            self.texture_points.exclude_from_boundary = True
            self.texture_points.is_primitive = True
            self.add_children(self.texture_points)

            self.control_points = Mob(control_points, **kwargs)
            self.control_points.is_primitive = True
            self.add_children(self.control_points)
            self.control_points.num_points_per_object = 4
            self.components = [self.texture_points, self.control_points]

        self.portion_of_curve_drawn = cast_to_tensor(portion_of_curve_drawn)
        self.normals = normals
        self.is_primitive = True
        self.render_primitive = BezierCircuitPrimitive

    def get_animatable_attrs(self):
        return {"border_width", "border_color", "portion_of_curve_drawn"}.union(
            super().get_animatable_attrs()
        )

    def get_default_color(self):
        return PURPLE

    def get_memory_used_per_timestep(self):
        n_ctrl = self.control_points.location.shape[-2]
        n_tex = self.texture_points.location.shape[-2]
        n_loc = self.location.shape[-2]
        n_segments = max(n_ctrl // 4, 1)  # cubic beziers have 4 control points each
        # Animation state: control points (3 floats), texture (5), location (6).
        animation_bytes = (n_ctrl * 3 + n_tex * 5 + n_loc * 6) * 4
        # Primitive output: control point corners, colors, normals, border data.
        primitive_bytes = n_segments * 4 * 3 * 4 + n_tex * 5 * 4 + n_loc * 12
        # RT polyline edges: ~100 samples per segment, 4 floats (16 bytes) each.
        rt_edge_bytes = n_segments * 100 * 16
        # Per-circuit RT metadata (20 floats), frame bounds (6 floats), BVH (~64 bytes).
        n_circuits = max(n_loc, 1)
        rt_meta_bytes = n_circuits * (80 + 24 + 64)
        return int(animation_bytes + primitive_bytes + rt_edge_bytes + rt_meta_bytes)

    def get_render_primitives(self):
        if self.empty:
            return None
        self.texture_points.set_time_inds_to(self)
        self.control_points.set_time_inds_to(self)

        vars = broadcast_all(
            [
                self.opacity,# * self.max_opacity,
                self.basis,
                self.glow,
                self.border_width
                * self.scene.render_settings.resolution[1] * self.scene.render_settings.anti_alias_level
                / (PREVIEW.resolution[1] * 2),
                self.border_color,
                self.portion_of_curve_drawn,
                self.glow_radius,
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
        self, x, tpc, loc, basis, o, n, g, bw, bc, pc, gr, num_segments_per_circuit=None
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
        #LoggerManager.instance().set_class("batching").log_message(
        #    f"Making bezier with num_segments_per_circuit: {num_segments_per_circuit}"
        #)
        # num_segments_per_circuit = torch.cat((starting_inds, torch.tensor((len(inds)-(starting_inds.amax() if len(starting_inds) > 0 else 0),), device=x.device)), -1)

        prim = self.render_primitive(
            x,
            next_segment_inds_offset,
            num_segments_per_circuit,
            c,
            o,
            basis[..., -3:],
            bw,
            bc,
            pc,
            loc,
            cast_to_tensor(self.grid_width).expand(-1, loc.shape[1], -1),
            cast_to_tensor(self.grid_height).expand(-1, loc.shape[1], -1),
            basis[..., :3],
            basis[..., 3:6],
            glow=g,
            glow_radius=gr,
            num_texture_points=self.num_texture_points,
            filled=self.filled,
        )
        prim.num_texture_points = self.num_texture_points
        return prim

    @animated_function(animated_args={"t": 0.0})
    def draw(self, t=1.0):
        self.control_points.set_time_inds_to(self)
        #if not hasattr(self, "_original_control_points"):
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
        start_t = cast_to_tensor(start_t).to(full_control_points.device)
        end_t = cast_to_tensor(end_t).to(full_control_points.device)

        num_frames = full_control_points.shape[0]
        total_control_points = full_control_points.shape[-2]

        if start_t.dim() == 0:
            start_t = start_t.view(1).expand(num_frames)

        if end_t.dim() == 0:
            end_t = end_t.view(1).expand(num_frames)
        else:
            end_t = end_t.view(num_frames)

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
