from algan import RIGHT
from algan.animation.animation_contexts import Off
from algan.mobs.renderable import Renderable
from algan.constants.color import *
from algan.geometry.geometry import get_roots_of_quadratic, get_orthonormal_vector
from algan.mobs.mob import Mob
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive, evaluate_cubic_bezier_derivative_old, evaluate_cubic_bezier_old2

from algan.utils.tensor_utils import *


class BezierCircuitCubic(Renderable):
    def __init__(self, control_points, normals=None, border_width=5, border_color=WHITE, portion_of_curve_drawn=1.0,
                 filled=True, add_texture_grid=False, texture_grid_size = 10, **kwargs):

        self.num_bezier_parameters = 4
        control_points = control_points.view(-1, control_points.shape[-1])
        '''ucp = unsquish(control_points, -2, self.num_bezier_parameters)
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
        control_points = squish(ucp, -3, -2)'''

        kwargs2 = {k: v for k, v in kwargs.items()}

        if 'color' in kwargs2:
            kwargs2['color'] = kwargs2['color'].reshape(-1,kwargs2['color'].shape[-1]).mean(-2)
        if normals is not None:
            normals = normals.reshape(-1, 3)
        mn = control_points.reshape(-1, 3).amin(-2)
        mx = control_points.reshape(-1, 3).amax(-2)
        kwargs2['location'] = (mn + mx) * 0.5

        self.grid_width = self.grid_height = 1
        self.num_texture_points = 0
        if (mx - mn).norm(p=2,dim=-1) <= 1e-6:
            kwargs2['basis'] = squish(torch.eye(3))
            first_basis = kwargs2['basis'][...,:3]
            second_basis = kwargs2['basis'][...,3:6]
        else:
            disps = control_points - kwargs2['location']
            dists = (disps).norm(p=2, dim=-1, keepdim=True)
            first_basis = disps[...,dists.argmax(-2, keepdim=True).squeeze(),:].unsqueeze(-2)
            if first_basis.norm(p=2,dim=-1) <= 1e-4:
                first_basis = RIGHT * 1e-4
            self.first_basis = first_basis
            first_basis_n = F.normalize(first_basis, p=2, dim=-1)

            disps = disps - dot_product(disps, first_basis_n) * first_basis_n

            dists = (disps).norm(p=2, dim=-1, keepdim=True)
            second_basis = disps[..., dists.argmax(-2, keepdim=True).squeeze(),:].unsqueeze(-2)
            if second_basis.norm(p=2,dim=-1) <= 1e-4:
                second_basis = get_orthonormal_vector(first_basis)
            second_basis = second_basis * first_basis.norm(p=2,dim=-1, keepdim=True) / second_basis.norm(p=2,dim=-1, keepdim=True)
            self.second_basis = second_basis
            third_basis_n = F.normalize(broadcast_cross_product(first_basis_n, second_basis), p=2, dim=-1)
            kwargs2['basis'] = torch.cat((first_basis, second_basis, third_basis_n), -1)

        self.register_attrs_as_animatable({'border_width', 'border_color', 'portion_of_curve_drawn'}, BezierCircuitCubic)
        super().__init__(**kwargs2)
        self.filled = filled

        texture_triangle_vertices = self.location.squeeze(0)
        if add_texture_grid:
            aspect_ratio = second_basis.norm(p=2, dim=-1) / first_basis.norm(p=2, dim=-1)

            a1 = torch.linspace(-1, 1, texture_grid_size).view(-1, 1, 1) * (1+1e-5)
            a2 = torch.linspace(-1, 1, int(texture_grid_size * aspect_ratio)).view(1, -1, 1) * (1+1e-5)
            texture_grid_points = (a1 * first_basis + a2 * second_basis) + self.location
            texture_triangle_vertices = texture_grid_points
            self.grid_width = texture_triangle_vertices.shape[-2]
            self.grid_height = texture_triangle_vertices.shape[-3]
            texture_triangle_vertices = texture_triangle_vertices.reshape(-1, texture_triangle_vertices.shape[-1])
            self.num_texture_points = texture_triangle_vertices.shape[-2]

            #control_points = torch.cat((control_points, texture_triangle_vertices), -2)
        kwargs['color'] = self.color
        with Off():
            self.texture_points = Mob(texture_triangle_vertices, **kwargs)
            self.texture_points.exclude_from_boundary = True
            self.texture_points.is_primitive = True
            self.add_children(self.texture_points)

        with Off():
            self.control_points = Mob(control_points, **kwargs)
            self.control_points.is_primitive = True
            self.add_children(self.control_points)
            self.control_points.num_points_per_object = 4

        self.border_width = cast_to_tensor(border_width)
        self.border_color = cast_to_tensor(border_color)
        self.portion_of_curve_drawn = cast_to_tensor(portion_of_curve_drawn)
        self.normals = normals
        self.is_primitive = True
        self.render_primitive = BezierCircuitPrimitive

    def get_animatable_attrs(self):
        return {'border_width', 'border_color', 'portion_of_curve_drawn'}.union(super().get_animatable_attrs())

    def get_default_color(self):
        return PURPLE

    def get_render_primitives(self):
        o, n, g, bw, bc, pc = broadcast_all([self.opacity * self.max_opacity, self.basis, self.glow, self.border_width, self.border_color, self.portion_of_curve_drawn], ignored_dims=[-1])

        num_control_points = 4 # cubic beziers
        x = unsquish(self.control_points.location, -2, num_control_points)
        # assert x.shape == [*, N, num_control_points, 3], where N is number of bezier segments.
        start_points = x[...,:1,:]
        end_points = x[...,-1:,:]

        # We allow for rendering circuits with holes,
        # we treat beziers which don't start at the previous one's end as marking the start of a new circuit (i.e. a hole).
        circuit_start_mask = ((start_points - end_points.roll(1, -3)).norm(p=2,dim=-1, keepdim=True) > 1e-5)
        circuit_end_mask = ((end_points - start_points.roll(-1, -3)).norm(p=2, dim=-1, keepdim=True) > 1e-5)

        inds = torch.arange(x.shape[-3], device=x.device).view(-1,1,1)
        circuit_start_inds = torch.where(circuit_start_mask, inds, 0)
        circuit_start_inds = torch.cummax(circuit_start_inds, -3)[0]
        # circuit_start_inds now contains the index of the start of the current index's circuit.

        next_segment_inds = (inds + 1) % x.shape[-3]
        # If the current ind is the end of the circuit, then the next segment is the first ind of this circuit, otherwise it is the next ind.
        next_segment_inds = torch.where(circuit_end_mask, circuit_start_inds, next_segment_inds)
        # We subtract inds so that each ind is represented as an offset from the current ind.
        # This way, we can concatenate together offsets from different objects, and then just add a torch.arange during rendering
        # to recover the index in the new concatenated tensor.
        next_segment_inds_offset = next_segment_inds - inds

        starting_inds = circuit_start_mask[0,:,0,0].nonzero()[:,0]
        num_segments_per_circuit = []
        if len(starting_inds) == 0:
            num_segments_per_circuit.append(torch.tensor((circuit_start_mask.shape[-3],), device=next_segment_inds.device, dtype=next_segment_inds.dtype).squeeze())
        else:
            for i in range(len(starting_inds)):
                num_segments_per_circuit.append((starting_inds[(i+1)] if (i+1) < len(starting_inds) else
                                                 circuit_start_mask.shape[-3]) - starting_inds[i])
        num_segments_per_circuit = torch.stack(num_segments_per_circuit, 0)
        #num_segments_per_circuit = torch.cat((starting_inds, torch.tensor((len(inds)-(starting_inds.amax() if len(starting_inds) > 0 else 0),), device=x.device)), -1)

        c = self.texture_points.color.unsqueeze(-3)
        if self.num_texture_points > c.shape[-2]:
            c = c.expand(-1,-1,self.num_texture_points,-1)

        prim = self.render_primitive(x, next_segment_inds_offset, num_segments_per_circuit, c, o, self.basis[..., -3:],
                                     bw, bc, pc, self.location, cast_to_tensor(self.grid_width),
                                     cast_to_tensor(self.grid_height), self.basis[...,:3], self.basis[...,3:6],
                                     glow=g, filled=self.filled)
        prim.num_texture_points = self.num_texture_points
        return prim


class BezierCurveCubic(BezierCircuitCubic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, filled=False, **kwargs)