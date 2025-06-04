from algan import RIGHT
from algan.animation.animation_contexts import Off
from algan.mobs.renderable import Renderable
from algan.constants.color import *
from algan.geometry.geometry import get_roots_of_quadratic, get_orthonormal_vector
from algan.mobs.mob import Mob
from algan.rendering.primitives.bezier_circuit_with_border_then_fill import BezierCircuitPrimitiveWithBorderFillRendering, evaluate_cubic_bezier_derivative_old, evaluate_cubic_bezier_old2

from algan.utils.tensor_utils import *


class BezierCircuitCubic(Renderable):
    def __init__(self, control_points, normals=None, border_width=5, border_color=WHITE, portion_of_curve_drawn=1.0,
                 filled=True, render_with_distance_to_curve=False, add_texture_grid=False, texture_grid_size = 10, **kwargs):

        control_points = control_points.view(-1, control_points.shape[-1])
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

            control_points = torch.cat((control_points, texture_triangle_vertices), -2)


        with Off():
            self.control_points = Mob(control_points, **kwargs)
            self.control_points.is_primitive = True
            self.add_children(self.control_points)

        self.border_width = cast_to_tensor(border_width)
        self.border_color = cast_to_tensor(border_color)
        self.portion_of_curve_drawn = cast_to_tensor(portion_of_curve_drawn)
        self.normals = normals
        self.is_primitive = True
        self.control_points.num_points_per_object = 4
        self.render_primitive = BezierCircuitPrimitiveWithBorderFillRendering if not render_with_distance_to_curve else BezierCircuitPrimitiveWithDistanceRendering
        self.render_with_distance = render_with_distance_to_curve

    def get_animatable_attrs(self):
        return {'border_width', 'border_color', 'portion_of_curve_drawn'}.union(super().get_animatable_attrs())

    def get_default_color(self):
        return PURPLE

    def get_render_primitives(self):
        def rt(x):
            if self.num_texture_points <= 0 or x.shape[-2] == 1:
                return x
            return x[...,:-self.num_texture_points,:]
        points = rt(torch.cat([dot_product(_, self.control_points.location) for _ in [self.basis[...,:3], self.basis[...,3:6]]], -1)[0])
        points = unsquish(points, -2, 4)
        # points.shape == [*, N, 4, 2], N is number of bezier curves that make up the circuit, 4 is number of control points per cubic bezier.

        # We allow for multiple separate circuits to be present. If the next beziers starting point is not equal to the current beziers ending point,
        # then we treat this as the end of the current bezier circuit, the next bezier marks the start of the next circuit.
        # for testing the signed area, we only use the first circuit.
        next_points = points.roll(-1,-3)
        loop_end_ind = ((next_points[...,0,:] - points[...,-1,:]).norm(p=2,dim=-1) > 1e-5).nonzero()
        if loop_end_ind.numel() != 0:
            loop_end_ind = loop_end_ind[0]
            if 0 < loop_end_ind < points.shape[-3]:
                points = points[...,:loop_end_ind,:]
        points = squish(points, -3, -2)
        next_points = points.roll(-1, -2)

        # If the signed area is positive, we reverse the order of the control points to flip inside and outside.
        reverse_order = (points[...,0] * next_points[...,1] - next_points[...,0] * points[...,1]).sum(-1) > 0
        def set_order(x):
            if reverse_order:
                return x.flip(-2)
            return x
        l, c, o, n, g, bw, bc, pf = [set_order(_) for _ in broadcast_all([rt(self.control_points.location),
                                                                          self.color if self.num_texture_points == 0 else rt(self.control_points.color),
                                        self.opacity * self.max_opacity, self.basis, self.glow, self.border_width, self.border_color, self.portion_of_curve_drawn], ignore_dims=[-1])]


        x = unsquish(l, -2, 4)
        # Delete all bezier curves where all control points are equal (i.e. the curve is a single point).
        #m = (x[0] - x[0].mean(-2, keepdim=True)).norm(p=2,dim=-1).mean(-1) > 1e-5
        #m = m.unsqueeze(-1).expand(-1,4).reshape(-1)
        #l, c, o, n, g,  bw, bc, pf = [_[:,m] for _ in [l, c, o, n, g,  bw, bc, pf]]
        if min(l.shape) == 0:
            return None

        def split_cubic_bezier_at_perpendiculars(control_points):
            """
            splits cubic beziers at the first point where the tangent direction is perpendicular to the starting tangent, or else at the midpoint of the tangent never passes perpendicular
            """

            #<B`(t), B`(0)> == 0
            #  ( -P₀ + 3P₁ - 3P₂ + P₃ )t³ + ( 3P₀ - 6P₁ + 3P₂ )t² + ( -3P₀ + 3P₁ )t + P₀
            #  3( -P₀ + 3P₁ - 3P₂ + P₃ )t^2 + 2( 3P₀ - 6P₁ + 3P₂ )t + ( -3P₀ + 3P₁ )
            '''
            (p[3] - p[0] + 3 * (p[1] - p[2]))
            b = (-6 * p[1] + p[2] * 3 + 3 * p[0])
            c = (3 * p[1] - 3 * p[0])
            d = p[0] - point

            '''
            control_points = unsquish(control_points, -2, 4)
            cp = control_points#[...,:2]
            P0 = cp[..., 0, :]
            P1 = cp[..., 1, :]
            P2 = cp[..., 2, :]
            P3 = cp[..., 3, :]
            d0 = evaluate_cubic_bezier_derivative_old(control_points, 0)
            a = dot_product(3*(-P0+3*P1-3*P2+P3), d0)
            b = dot_product(2*(3*P0-6*P1+3*P2), d0)
            c = dot_product((-3*P0+3*P1), d0)
            roots = get_roots_of_quadratic(a, b, c, 2)
            m = (roots > 0) & (roots < 1)
            roots = roots * m + (~m) * 2
            roots = roots.amin(-1)
            m = roots < 1
            roots = roots * m + (~m) * 0.5
            roots = torch.full_like(roots, 0.5)
            t = t_float = roots
            one_minus_t = 1-t
            # --- De Casteljau's Algorithm Steps ---
            # First level of interpolation
            P01 = one_minus_t * P0 + t_float * P1
            P12 = one_minus_t * P1 + t_float * P2
            P23 = one_minus_t * P2 + t_float * P3

            # Second level of interpolation
            P012 = one_minus_t * P01 + t_float * P12
            P123 = one_minus_t * P12 + t_float * P23

            # Third level of interpolation (this is the point on the curve at t)
            P0123 = one_minus_t * P012 + t_float * P123

            # --- Assemble the Control Points for the New Curves ---
            # The control points generated during the algorithm form the new curves' points.

            # Curve 1: Uses points from the "left" side of the De Casteljau diagram
            # Control Points: P0, P01, P012, P0123
            curve1_cp = torch.stack([P0, P01, P012, P0123], dim=-2)

            # Curve 2: Uses points from the "right" side of the De Casteljau diagram
            # Control Points: P0123, P123, P23, P3
            curve2_cp = torch.stack([P0123, P123, P23, P3], dim=-2)

            return squish(torch.cat((curve1_cp, curve2_cp), -2), -3, -2)


        # Because the rendering algorithm we use uses a binary search to find the nearest point on the curve,
        # the rendering algorithm can fail for bezier curves which turn more than 90 degrees.
        # We therefore split all of the bezier curves to ensure that none can turn for more than 90 degrees.
        l = split_cubic_bezier_at_perpendiculars(l)
        l = split_cubic_bezier_at_perpendiculars(l)
        c, o, n, g, bw, bc, pf = [squish(torch.cat([(unsquish(_, -2, 4)) for __ in range(4)], -2), -3, -2) for _ in [c, o, n, g, bw, bc, pf]]
        x = unsquish(l, -2, 4)

        # For rendering sharp corners correctly, we add 2 extra control points representing the average starting
        # and ending derivatives. Note that the averaging is done between the derivative at the end of this curve
        # and the derivative at the start of the next curve (and analogously for the starting derivative).
        end_points = evaluate_cubic_bezier_old2(x, 1)
        start_points = evaluate_cubic_bezier_old2(x, 0)
        end_derivs = evaluate_cubic_bezier_derivative_old(x, 1)
        start_derivs = evaluate_cubic_bezier_derivative_old(x, 0)

        # Identify where each circuit ends. Remember, there can be multiple.
        loop_end_mask = (end_points - start_points.roll(-1, -2)).norm(p=2,dim=-1, keepdim=True) > 1e-5
        loop_start_mask = (start_points - end_points.roll(1, -2)).norm(p=2,dim=-1, keepdim=True) > 1e-5

        loop_start_inds = loop_start_mask[0,:,0].nonzero()
        loop_end_inds = loop_end_mask[0, :, 0].nonzero()
        if loop_start_inds.numel() == 0 or loop_end_inds.numel() == 0:
            loop_start_derivs = start_derivs
            loop_end_derivs = end_derivs
        else:
            repeats = (loop_end_inds - loop_start_inds)+1
            gather_inds = torch.repeat_interleave(loop_start_inds, repeats.squeeze(-1), 0)
            loop_start_derivs = broadcast_gather(start_derivs, -2, gather_inds)

            gather_inds = torch.repeat_interleave(loop_end_inds, repeats.squeeze(-1), 0)
            loop_end_derivs = broadcast_gather(end_derivs, -2, gather_inds)

        def get_avg(x, y):
            return F.normalize(F.normalize(x, p=2, dim=-1) + F.normalize(y, p=2, dim=-1), p=2, dim=-1) * 0.02

        next_derivs = start_derivs.roll(-1,-2) * (~loop_end_mask) + loop_end_mask * loop_start_derivs
        end_avg_deriv = get_avg(next_derivs, next_derivs) + end_points

        prev_derivs = end_derivs.roll(1, -2) * (~loop_start_mask) + loop_start_mask * loop_end_derivs
        start_avg_deriv = get_avg(prev_derivs, prev_derivs) + start_points

        x = torch.cat((x, start_avg_deriv.unsqueeze(-2), end_avg_deriv.unsqueeze(-2)), -2)
        l = squish(x, -3, -2)

        def padexp(_):
            _ = unsquish(_, -2, 4)
            _ = torch.cat((_, _[...,-2:,:]), -2)
            return squish(_, -3, -2)
        c, o, n, g, bw, bc, pf = [padexp(_) for _ in [c, o, n, g, bw, bc, pf]]

        def trim(_):
            _ = unsquish(_, -2, 6)
            return squish(_, -3, -2)

        l, c, o, n, g, bw, bc, pf = [(trim(_)) for _ in [l, c, o, n, g, bw, bc, pf]]

        if self.num_texture_points > 0:
            l = torch.cat((l, self.control_points.location[...,-self.num_texture_points:,:]), -2)
            colo = self.control_points.color
            if colo.shape[-2] == 1:
                colo = colo.expand(-1,self.num_texture_points,-1)
            else:
                colo = colo[...,-self.num_texture_points:,:]
            c = torch.cat((c, colo), -2)

        prim = self.render_primitive(l, c, o[..., :1, :], self.basis[..., -3:], bw[..., :1, :], bc[..., :1, :], pf[..., :1, :],
                             *broadcast_all([self.location, cast_to_tensor(self.grid_width), cast_to_tensor(self.grid_height),
                                             self.basis[...,:3], self.basis[...,3:6]], [-1]),
                             glow=g[...,:1,:], filled=self.filled
                             )
        prim.num_texture_points = self.num_texture_points
        prim.render_with_distance = self.render_with_distance
        return prim


class BezierCurveCubic(BezierCircuitCubic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, filled=False, **kwargs)