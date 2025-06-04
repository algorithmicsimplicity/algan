import math

import torch
import torch.nn.functional as F

from algan.constants.color import BLUE, BLACK
from algan.defaults.device_defaults import DEFAULT_RENDER_DEVICE
from algan.geometry.geometry import intersect_line_with_plane
from algan.rendering.primitives.primitive import InsufficientMemoryException, RenderPrimitive2D
from algan.utils.tensor_utils import broadcast_all
from algan.utils.tensor_utils import dot_product, squish, broadcast_gather, expand_as_left, unsquish, unsqueeze_right


def evaluate_cubic_bezier_old2(p, t):
    out = ((1 - t) ** 3) * p[..., 0, :]
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :]
    out[:] += (t ** 3) * p[..., 3, :]
    return out


def evaluate_cubic_bezier_old(p, t, out, mem):
    out[:] = ((1 - t) ** 3) * p[..., 0, :]
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :]
    out[:] += (t ** 3) * p[..., 3, :]
    return out


def evaluate_cubic_bezier(p, t, out, mem):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    mem.save_pointer()
    temp = mem.get_tensor([*p0.shape])
    temp2 = mem.get_tensor([*p0.shape])
    torch.subtract(1, t, out=temp[...,:1])
    torch.pow(temp[...,:1], 3, out=temp2[...,:1])
    torch.mul(p0, temp2[...,:1], out=out)
    # out[:] = ((1 - t) ** 3) * p[..., 0, :]
    torch.mul(t, p1, out=temp2)
    torch.pow(temp[...,:1], 2, out=temp[...,:1])
    torch.addcmul(out, temp[...,:1], temp2, value=3, out=out)
    #out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]

    torch.square(t, out=temp[...,:1])
    torch.subtract(1,t,out=temp2[...,:1])
    torch.mul(temp[...,:1],temp2[...,:1],out=temp[...,:1])
    torch.addcmul(out, temp[...,:1], p2, value=3, out=out)
    # out[:] += 3 * (1 - t) * t * t * p[..., 2, :]

    torch.pow(t, 3, out=temp[...,:1])
    torch.addcmul(out, temp[...,:1], p3, out=out)
    #out[:] += (t ** 3) * p[..., 3, :]
    mem.reset_pointer()
    return out


def evaluate_cubic_bezier_derivative_old(p, t):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    return 3 * ((1 - t) ** 2) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * (t * t) * (p3 - p2)


def evaluate_cubic_bezier_derivative_with_end(p, t, end_portion=0.01):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    start_deriv = p[..., 4, :] - p0
    end_deriv = p[...,5,:] - p3
    out = 3 * ((1 - t) ** 2) * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * (t * t) * (p3 - p2)
    m = (t < end_portion)
    out = out * (~m) + m * start_deriv
    m = (t > (1-end_portion))
    out = out * (~m) + m * end_deriv
    return out

def evaluate_cubic_bezier_derivative(p, t, out, mem, end_portion=-0.05):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    start_deriv = p[..., 4, :] - p0
    end_deriv = p[..., 5, :] - p3
    mem.save_pointer()
    temp = mem.get_tensor([*p0.shape])
    temp2 = mem.get_tensor([*p0.shape])
    out[:] = 0
    torch.subtract(p3, p2, out=temp)
    torch.pow(t, 2, out=temp2[...,:1])
    torch.addcmul(out, temp, temp2[...,:1], value=3, out=out)
    #out[:] += 3 * (t * t) * (p3 - p2)
    torch.subtract(p2, p1, out=temp)
    torch.mul(temp, t, out=temp)
    torch.subtract(1, t, out=temp2[...,:1])
    torch.addcmul(out, temp, temp2[...,:1], value=6, out=out)
    #out[:] += 6 * (1 - t) * t * (p2 - p1)
    torch.pow(temp2[...,:1], 2, out=temp2[...,:1])
    torch.subtract(p1, p0, out=temp)
    torch.addcmul(out, temp, temp2[...,:1], value=3, out=out)
    #out[:] += 3 * ((1 - t) ** 2) * (p1 - p0)

    m = (t < end_portion)
    torch.where(m, start_deriv, out, out=out)
    #out = out * (~m) + m * start_deriv
    m = (t > (1 - end_portion))
    torch.where(m, end_deriv, out, out=out)
    #out = out * (~m) + m * end_deriv
    mem.reset_pointer()
    return out


def evaluate_cubic_bezier_second_derivative(p, t):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    return 6 * t * (p3 + 3 * (p1-p2) - p0) + 6 * (p0 - 2 * p1 + p2)


def solve_cubic_bezier_second_derivative_equal_to_0(p):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    denom = (p3 + 3 * (p1 - p2) - p0)
    m = denom.abs() <= 1e-5
    t = -(p0 - 2 * p1 + p2) / denom
    t = t.nan_to_num(2)
    m = m | (t <= 0) | (t >= 1)
    t = t*(~m) + m * 2
    t = t.amin(-1, keepdim=True)
    m = (t >= 1)
    t = t*(~m) + m * 0.5
    return t


class BezierCircuitPrimitiveWithBorderFillRendering(RenderPrimitive2D):
    def __init__(self, corners=None, colors=BLUE, opacity=1, normals=None, border_width=None, border_color=None, portion_of_curve_drawn=None,
                 mob_center=None, grid_width=None, grid_height=None, first_basis=None, second_basis=None,
                 loop_starts=None, loop_ends=None, perimeter_points=None,
                 reverse_perimeter=False, triangle_collection=None, glow=0, num_texture_points=0, render_with_distance=True, filled=True):

        self.reverse_perimeter = reverse_perimeter
        self.num_texture_points = num_texture_points
        self.render_with_distance = render_with_distance
        self.filled = filled
        if triangle_collection is not None:
            max_num_pieces = max([_.corners.shape[-2] for _ in triangle_collection])

            def pad(x):
                return torch.cat((x, x.mean(-2, keepdim=True).expand(*[-1 for _ in range(x.dim()-2)], max_num_pieces-x.shape[-2], x.shape[-1])), -2)
            self.num_texture_points = triangle_collection[0].num_texture_points
            self.render_with_distance = triangle_collection[0].render_with_distance
            self.filled = triangle_collection[0].filled
            self.corners, self.colors, self.normals, self.border_width, self.border_color, self.portion_of_curve_drawn = (
                ((torch.stack([pad(__) for __ in _], 1))).to(DEFAULT_RENDER_DEVICE, non_blocking=True) for _ in
                zip(*((
                    triangle.corners, expand_as_left(triangle.colors, triangle.corners), expand_as_left(triangle.normals, triangle.corners),
                    expand_as_left(triangle.border_width, triangle.corners), expand_as_left(triangle.border_color, triangle.corners),
                    expand_as_left(triangle.portion_of_curve_drawn, triangle.corners),
                                      ) for triangle in triangle_collection)))
            self.mob_center, self.grid_width, self.grid_height, self.basis1, self.basis2 = (
                ((torch.stack([(__) for __ in _], 1))).to(DEFAULT_RENDER_DEVICE, non_blocking=True) for _ in
                zip(*(broadcast_all((
                    triangle.mob_center,
                    triangle.grid_height,
                    triangle.grid_width,
                    triangle.basis1,
                    triangle.basis2,
                ), [-1]) for triangle in triangle_collection)))
            if min(self.corners.shape)==0:
                print('0 dim')
            self.border_width = self.border_width[...,0,:1]
            self.border_color = self.border_color[...,0,:]
            if self.num_texture_points <= 0:
                self.colors = self.colors[..., 0, :]
            else:
                self.colors = self.colors[...,(-self.num_texture_points):, :]
            self.portion_of_curve_drawn = self.portion_of_curve_drawn[...,0,:1]
            return
        self.corners = corners
        border_color, opacity, glow = broadcast_all([border_color, opacity, glow], ignore_dims=[-1])
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow
        self.colors[..., -1:] *= opacity
        self.normals = normals
        self.loop_starts = loop_starts
        self.loop_ends = loop_ends
        self.border_width, self.border_color, self.portion_of_curve_drawn = border_width, border_color, portion_of_curve_drawn
        self.border_color[..., -2:-1] += glow
        self.border_color[..., -1:] *= opacity
        self.mob_center = mob_center
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.basis1 = first_basis
        self.basis2 = second_basis

    def get_batch_identifier(self):
        return f'{__class__}_{self.num_texture_points}_{self.render_with_distance}_{self.filled}'

    def render_(self, time_start, time_end, object_start, object_end, ray_origin, screen_point, screen_basis,
               background_color=BLACK, anti_alias=False, anti_alias_offset=[0.5, 0.5], anti_alias_level=1,
               light_origin=None, light_color=None, screen_width=2000, screen_height=2000, window_coords=None, memory=None, primitive_type=None):

        ray_origin = ray_origin.unsqueeze(-2)
        screen_point = screen_point.unsqueeze(-2)
        screen_basis = unsquish(screen_basis, -1, 3)

        def select_time(x, texture=False):
            x = x if len(x) == 1 else x[time_start:time_end]
            x = x if x.shape[1] == 1 else x[:, int(x.shape[1]*object_start):int(x.shape[1]*object_end)]
            return x
        corners = select_time(self.corners)
        if corners.numel() == 0:
            return None
        normals = select_time(self.normals)
        normals = unsquish(normals, -1, 3)
        colors = select_time(self.colors)
        border_colors = select_time(self.border_color)
        border_width = select_time(self.border_width)
        portion_of_curve_drawn = select_time(self.portion_of_curve_drawn)
        screen_point = select_time(screen_point)
        screen_basis = select_time(screen_basis)
        ray_origin = select_time(ray_origin)

        if window_coords is None:
            window_coords = 0, 0, screen_width, screen_height
        window_height = window_coords[-1] - window_coords[1]
        window_width = window_coords[-2] - window_coords[0]
        start_x, start_y, end_x, end_y = window_coords
        end_x = end_x - 1
        end_y = end_y - 1
        def project_onto_screen(x):
            rays = F.normalize(x - ray_origin, p=2, dim=-1)
            projected_corners, _ = intersect_line_with_plane(rays, screen_point, screen_basis[..., -1:, :], ray_origin)
            projected_corners.nan_to_num_()
            projected_distances = (x - ray_origin).norm(p=2, dim=-1, keepdim=True)
            projected_corners -= screen_point
            corners_2d = dot_product(projected_corners.unsqueeze(-2), screen_basis[...,:-1, :].unsqueeze(-3), -1, keepdim=False)
            corners_2d.nan_to_num_()
            corners_2d = corners_2d * (screen_height//2)
            corners_2d[..., 0] += screen_width // 2
            corners_2d[..., 1] += screen_height // 2
            corners_locs = corners_2d
            corners_inds = corners_locs.long()
            return corners_locs, corners_inds, projected_distances

        corners_locs, corners_inds, projected_distances = project_onto_screen(corners)
        padding = max(20, self.border_width.amax().ceil().long()+1)

        def get_fragment_coords(x, get_circuit_inds=False, batch_time_dim=False):
            batch_time_dim = False
            bounding_corners = torch.stack(((x.amin(-2) - padding), (x.amax(-2) + padding)), -2).clamp_(
                min=torch.tensor((start_x, start_y), device=x.device),
                max=torch.tensor((end_x, end_y), device=x.device))
            bounding_box_sizes = (bounding_corners[..., 1, :] - bounding_corners[..., 0, :]).amax(0, keepdim=True)
            if batch_time_dim:
                bounding_box_sizes_unbatched = bounding_box_sizes
            else:
                bounding_box_sizes_unbatched = (bounding_corners[..., 1, :] - bounding_corners[..., 0, :])
            bbss = bounding_box_sizes.prod(-1, keepdim=True)

            bounding_box_num_pixels = bbss.amax(0)
            num_fragments = bounding_box_num_pixels.sum() * (time_end - time_start)
            mem_per_fragment = 256
            total_mem_required = num_fragments * mem_per_fragment

            free_mem = self.memory.get_num_bytes_remaining()

            if free_mem < total_mem_required:
                raise InsufficientMemoryException

            repeats = bounding_box_num_pixels.view(-1)
            num_frags = repeats.sum()
            repeats_inds = torch.repeat_interleave(torch.arange(len(repeats), device=repeats.device), repeats, -1,
                                                   output_size=num_frags).unsqueeze(-1)
            if get_circuit_inds:
                repeats_circuit = unsquish(repeats, -1, x.shape[-3]).sum(-1)
                repeats_inds_circuit = torch.repeat_interleave(
                    torch.arange(len(repeats_circuit), device=repeats.device), repeats_circuit, -1,
                    output_size=num_frags).unsqueeze(-1)
            else:
                repeats_inds_circuit = None

            def s(_):
                if not get_circuit_inds:
                    return _
                return _.view(_.shape[0], -1, _.shape[-1])

            def s2(_):
                if not get_circuit_inds:
                    return _
                return squish(_, 1, 2)

            nps = (bounding_box_num_pixels).view(-1,bounding_box_num_pixels.shape[-1])
            nps = (nps.cumsum(-2) - nps)
            offsets = self.expand_verts_to_frags(nps, repeats_inds, -2)

            fragment_inds = torch.arange(offsets.shape[-2], device=offsets.device).view(-1, 1) - offsets
            bounding_box_widths = self.expand_verts_to_frags(s(bounding_box_sizes_unbatched)[..., :1], repeats_inds,
                                                             -2).clamp_min_(1)
            bounding_box_heights = self.expand_verts_to_frags(s(bounding_box_sizes_unbatched)[..., 1:], repeats_inds,
                                                              -2).clamp_min_(1)

            bounding_corners_rep = self.expand_verts_to_frags(s2(bounding_corners)[..., 0, :], repeats_inds, -2)
            fragment_x = (fragment_inds % (bounding_box_widths)) + bounding_corners_rep[..., :1]
            fragment_y = (fragment_inds // (bounding_box_widths)) + bounding_corners_rep[..., 1:]

            fragment_ind_vertical = (fragment_inds % (bounding_box_widths)) * bounding_box_heights + (
                        fragment_inds // (bounding_box_widths))
            return fragment_x, fragment_y, repeats_inds, s2(bounding_corners), s(
                bounding_box_sizes_unbatched), repeats_inds_circuit, bounding_box_num_pixels, (
                        fragment_inds % (bounding_box_widths)), fragment_ind_vertical

        num_bezier_params = 6
        fragment_x, fragment_y, repeats_inds, bounding_corners, bounding_box_sizes, repeats_inds_circuit, _, fragment_inds_circuit, fragment_ind_vertical_circuit = get_fragment_coords(
            unsquish(corners_inds if self.num_texture_points <= 0 else corners_inds[...,:-self.num_texture_points, :], -2, num_bezier_params), True)

        if fragment_x.numel() == 0:
            return None
        control_points = corners_locs
        if self.num_texture_points > 0:
            control_points = control_points[..., :(-self.num_texture_points), :]
        control_points = unsquish(control_points, -2, num_bezier_params)
        bezier_normals = self.expand_verts_to_frags(normals[...,0,-1,:], repeats_inds_circuit, -2)
        view_angle = dot_product(F.normalize(bezier_normals,p=2,dim=-1),
                                    self.expand_verts_to_frags((F.normalize(select_time(self.mob_center) -
                                                                            ray_origin, p=2,dim=-1)).squeeze(-2),
                                                               repeats_inds_circuit))
        reverse_mask = (view_angle >= 0).float() * 2 - 1

        def get_within_circuit_mask_without_dist(cubic_control_points_verts, fragment_coords, repeats_inds=repeats_inds):
            flat_mask = view_angle.abs() >=  1e-3
            cubic_control_points_verts = squish(cubic_control_points_verts, 1, 2)
            batch_shape = fragment_coords.shape[:-1]
            self.memory.save_pointer()
            closest_dist = self.get_tensor([*batch_shape,1])
            closest_dist[:] = 1e12
            closest_dot = self.get_tensor([*batch_shape, 1])
            closest_dot[:] = 0

            closest_mask = self.get_tensor([*batch_shape,1], torch.bool)
            closest_t = self.get_tensor([*batch_shape,1])
            current_m = self.get_tensor([*batch_shape, 1], torch.bool)
            current_m2 = self.get_tensor([*batch_shape, 1], torch.bool)
            current_m3 = self.get_tensor([*batch_shape,  1], torch.bool)
            current_dot = self.get_tensor([*batch_shape, 1])
            current_dist = self.get_tensor([*batch_shape, 1])
            current_dist[:] = 1e12
            current_temp = self.get_tensor([*batch_shape, 1])
            current_point = self.get_tensor([*batch_shape, 2])
            current_derivs = self.get_tensor([*batch_shape, 2])
            current_perp = self.get_tensor([*batch_shape, 2])
            current_control_points = self.get_tensor([*batch_shape, num_bezier_params,2])
            ts = self.get_tensor([*batch_shape,1])

            def dp2(x, y, out):
                out[:] = 0
                torch.addcmul(out[...,0],x[...,0],y[...,0], out=out[...,0])
                torch.addcmul(out[..., 0], x[..., 1], y[..., 1], out=out[..., 0])
                return out

            n = bounding_box_sizes.amax().log2().int()+1

            def update_closest_dist_for_t(t):
                for i in range(0, 1):
                    cubic_control_points = self.expand_verts_to_frags(cubic_control_points_verts, repeats_inds.unsqueeze(-2), -3, out=current_control_points)

                    torch.mean(cubic_control_points, -2, keepdim=False, out=current_point)
                    torch.subtract(cubic_control_points, current_point.unsqueeze(-2), out=cubic_control_points)
                    torch.norm(cubic_control_points, p=1, dim=(-2,-1), out=current_dot.squeeze(-1))
                    pad_m = torch.less_equal(current_dot, 1e-3, out=current_m3)
                    torch.add(cubic_control_points, current_point.unsqueeze(-2), out=cubic_control_points)
                    t[:] = 0.5

                    current_dist2 = current_dist
                    for iteration in range(n):
                        evaluate_cubic_bezier(cubic_control_points, t, current_point, self.memory)
                        disps = torch.subtract(fragment_coords, current_point, out=current_point)
                        dp2(disps, disps, out=current_dist2)
                        current_dist2.sqrt_()
                        current_temp[:] = 1e12
                        torch.where(pad_m, current_temp, current_dist2, out=current_dist2)
                        derivs = evaluate_cubic_bezier_derivative(cubic_control_points, t, current_derivs, self.memory)
                        derivs = F.normalize(derivs, p=2, dim=-1, out=derivs)

                        dist_m = torch.lt(current_dist2, closest_dist, out=current_m2)
                        torch.minimum(closest_dist, current_dist2, out=closest_dist)
                        perp = torch.stack((derivs[...,1], -derivs[...,0]), -1)


                        torch.where(dist_m, (t), closest_t, out=closest_t)
                        next_derivs = cubic_control_points[...,-1,:] - cubic_control_points[...,3,:]
                        prev_derivs = cubic_control_points[...,-2,:] - cubic_control_points[...,0,:]
                        next_perp = torch.stack((next_derivs[...,1], -next_derivs[...,0]), -1)
                        prev_perp = torch.stack((prev_derivs[..., 1], -prev_derivs[..., 0]), -1)
                        m_conjunction_end = dot_product(perp, next_derivs) >= 0
                        m_conjunction_start = dot_product(perp, prev_derivs) < 0
                        eps = 1e-2
                        m_end = t > (1-eps)
                        m_start = t < (eps)
                        interior_mask = ((dot_product(perp, disps) > 0) )
                        end_interior_mask = ((dot_product(next_perp, disps) > 0) )
                        start_interior_mask = ((dot_product(prev_perp, disps) > 0))
                        conjunct_mask_end = end_interior_mask & interior_mask
                        disjunct_mask_end = end_interior_mask | interior_mask
                        conjunct_mask_start = start_interior_mask & interior_mask
                        disjunct_mask_start = start_interior_mask | interior_mask
                        interior_mask = torch.where(m_end, torch.where(m_conjunction_end, conjunct_mask_end, disjunct_mask_end), interior_mask)
                        interior_mask = torch.where(m_start, torch.where(m_conjunction_start, conjunct_mask_start, disjunct_mask_start), interior_mask)
                        torch.where(dist_m & flat_mask & (current_dist2 < 3), (interior_mask * 2 - 1) * reverse_mask, closest_dot, out=closest_dot)

                        #Update t
                        dp2(disps, derivs, out=current_dist2)
                        m = torch.gt(current_dist2, 0, out=current_m)
                        s = (2 ** (-(iteration + 2)))
                        current_dist2 = s
                        current_dist2 *= m
                        t += current_dist2
                        current_dist2 -= s
                        t += current_dist2

                        current_perp[..., 0] = derivs[..., 1]
                        torch.mul(derivs[..., 0], -1, out=current_perp[..., 1])
                        dp2(current_perp, disps, out=current_dot)
                        m = torch.gt(current_dot * reverse_mask, 0, out=current_m)

                        torch.where(dist_m, m, closest_mask, out=closest_mask)
                bws = self.expand_verts_to_frags(border_width, repeats_inds_circuit)
                m2 = (closest_dist < bws)
                #closest_t2 = closest_t + self.expand_verts_to_frags(torch.arange(portion_of_curve_drawn.shape[2], device=t.device, dtype=torch.float).reshape(1,-1).expand(self.portion_of_curve_drawn.shape[1],-1).reshape(-1), repeats_inds_circuit.view(-1), 0).reshape(-1,1)
                #m3 = closest_t2 <= self.expand_verts_to_frags(portion_of_curve_drawn, repeats_inds_circuit)
                # TODO make this work with m3, i.e. partial borders. We probably need to render 2 borders,
                # one with partial, and one used purely for filling (which is capped at a minimum width.
                return closest_mask, m2, closest_dot.clone(), closest_dist.clone()# & m3

            interior_mask, border_mask, closest_dot, closest_dist = update_closest_dist_for_t(ts)

            self.memory.reset_pointer()
            return interior_mask, border_mask, closest_dot, closest_dist

        def get_within_circuit_mask_with_distance(cubic_control_points_verts, fragment_coords, repeats_inds=repeats_inds_circuit):
            # cubic_control_points.shape == [*,n,4,2], n is number of pieces (padded), 4 is number of control points per piece (cubic), points are in 2d (screen) space.
            # fragment_coords.shape == [*, 1, 2]

            num_segments = cubic_control_points_verts.shape[-3]

            batch_shape = fragment_coords.shape[:-1]
            num_bytes_per_segment = math.prod(batch_shape) * (30+num_bezier_params*2)
            self.memory.save_pointer()
            remaining_mem = len(self.memory) - self.memory.current_pointer
            batch_size_segments = min(max((remaining_mem // num_bytes_per_segment)-1, 1), num_segments)
            closest_dist = self.get_tensor([*batch_shape,batch_size_segments,1])
            closest_dist[:] = 1e12
            closest_mask = self.get_tensor([*batch_shape, batch_size_segments,1], torch.bool)
            closest_t = self.get_tensor([*batch_shape, batch_size_segments,1])
            current_m = self.get_tensor([*batch_shape, batch_size_segments, 1], torch.bool)
            current_m2 = self.get_tensor([*batch_shape, batch_size_segments, 1], torch.bool)
            current_m3 = self.get_tensor([*batch_shape, batch_size_segments, 1], torch.bool)
            current_dot = self.get_tensor([*batch_shape, batch_size_segments,1])
            current_dist = self.get_tensor([*batch_shape,batch_size_segments,1])
            current_temp = self.get_tensor([*batch_shape, batch_size_segments,1])
            current_point = self.get_tensor([*batch_shape,batch_size_segments,2])
            current_derivs = self.get_tensor([*batch_shape, batch_size_segments,2])
            current_perp = self.get_tensor([*batch_shape,batch_size_segments,2])
            current_control_points = self.get_tensor([*batch_shape,batch_size_segments, num_bezier_params,2])
            ts = self.get_tensor([*batch_shape,batch_size_segments,1])

            def dp2(x, y, out):
                out[:] = 0
                torch.addcmul(out[...,0],x[...,0],y[...,0], out=out[...,0])
                torch.addcmul(out[..., 0], x[..., 1], y[..., 1], out=out[..., 0])
                return out

            n = bounding_box_sizes.amax().log2().int()+1

            def update_closest_dist_for_t(t):
                for i in range(0, num_segments, batch_size_segments):
                    ccpv = cubic_control_points_verts[..., i:i+batch_size_segments, :, :]
                    b = ccpv.shape[-3]
                    cubic_control_points = self.expand_verts_to_frags(ccpv, repeats_inds.unsqueeze(-2).unsqueeze(-2), -4, out=current_control_points[...,:b,:,:])
                    segment_inds = torch.arange(i, i+b, device=t.device).float().unsqueeze(-1)

                    current_dist_b = current_dist[..., :b, :]
                    current_dot_b = current_dot[..., :b, :]
                    current_point_b = current_point[..., :b, :]
                    torch.mean(cubic_control_points, -2, keepdim=False, out=current_point_b)
                    torch.subtract(cubic_control_points, current_point_b.unsqueeze(-2), out=cubic_control_points)
                    torch.norm(cubic_control_points, p=1, dim=(-2,-1), out=current_dot_b.squeeze(-1))
                    pad_m = torch.less_equal(current_dot_b, 1e-6, out=current_m3[...,:b,:])
                    torch.add(cubic_control_points, current_point_b.unsqueeze(-2), out=cubic_control_points)


                    current_temp_b = current_temp[..., :b, :]
                    current_derivs_b = current_derivs[..., :b, :]
                    current_perp_b = current_perp[..., :b, :]
                    closest_dist_b = closest_dist[..., :b, :]
                    current_m_b = current_m[..., :b, :]
                    current_m2_b = current_m2[..., :b, :]
                    t_b = t[...,:b,:]
                    closest_t_b = closest_t[..., :b, :]
                    closest_mask_b = closest_mask[..., :b, :]

                    t_b[:] = 0.5

                    for iteration in range(n):
                        #t[:] = iteration / n#0.5 #########SWITCH HERE FOR BRUTE FORCE MODE
                        evaluate_cubic_bezier(cubic_control_points, t_b, current_point_b, self.memory)
                        disps = torch.subtract(fragment_coords.unsqueeze(-2), current_point_b, out=current_point_b)
                        dp2(disps, disps, out=current_dist_b)
                        current_temp_b[:] = 1e12
                        torch.where(pad_m, current_temp_b, current_dist_b, out=current_dist_b)
                        derivs = evaluate_cubic_bezier_derivative(cubic_control_points, t_b, current_derivs_b, self.memory, end_portion=0.05)
                        dist_m = torch.lt(current_dist_b, closest_dist_b, out=current_m2_b)
                        torch.minimum(closest_dist_b, current_dist_b, out=closest_dist_b)
                        torch.where(dist_m, (t_b + segment_inds), closest_t_b, out=closest_t_b)


                        #Update t
                        dp2(disps, derivs, out=current_dist_b)
                        m = torch.gt(current_dist_b, 0, out=current_m_b)
                        s = (2 ** (-(iteration + 2)))
                        current_dist_b = s
                        current_dist_b *= m
                        t_b += current_dist_b
                        current_dist_b -= s
                        t_b += current_dist_b

                        current_perp_b[..., 0] = derivs[..., 1]
                        torch.mul(derivs[..., 0], -1, out=current_perp_b[..., 1])
                        dp2(current_perp_b, disps, out=current_dot_b)

                        next_derivs = cubic_control_points[..., -1, :] - cubic_control_points[..., 3, :]
                        prev_derivs = cubic_control_points[..., -2, :] - cubic_control_points[..., 0, :]
                        next_perp = torch.stack((next_derivs[..., 1], -next_derivs[..., 0]), -1)
                        prev_perp = torch.stack((prev_derivs[..., 1], -prev_derivs[..., 0]), -1)
                        m_conjunction_end = dot_product(current_perp, next_derivs) >= 0
                        m_conjunction_start = dot_product(current_perp, prev_derivs) < 0
                        eps = 1e-2
                        m_end = t > (1 - eps)
                        m_start = t < (eps)
                        interior_mask = current_dot_b > 0
                        end_interior_mask = ((dot_product(next_perp, disps) > 0))
                        start_interior_mask = ((dot_product(prev_perp, disps) > 0))
                        conjunct_mask_end = end_interior_mask & interior_mask
                        disjunct_mask_end = end_interior_mask | interior_mask
                        conjunct_mask_start = start_interior_mask & interior_mask
                        disjunct_mask_start = start_interior_mask | interior_mask
                        interior_mask = torch.where(m_end, torch.where(m_conjunction_end, conjunct_mask_end,
                                                                       disjunct_mask_end),
                                                    interior_mask)
                        interior_mask = torch.where(m_start, torch.where(m_conjunction_start, conjunct_mask_start,
                                                                         disjunct_mask_start),
                                                    interior_mask)

                        m = torch.where((reverse_mask.unsqueeze(-2) > 0), interior_mask, ~interior_mask)
                        torch.where(dist_m, m, closest_mask_b, out=closest_mask_b)

                best_ind = (closest_dist).argmin(-2, keepdim=True)
                torch.amin(closest_dist, -2, out=current_dist[...,0,:])
                best_m = torch.eq(closest_dist, current_dist[...,:1,:], out=current_m)
                closest_mask[:] &= best_m
                m = torch.any(closest_mask, dim=-2, out=current_m[...,0,:])
                m2 = current_dist[...,0,:] < self.expand_verts_to_frags((border_width), repeats_inds)
                t = broadcast_gather(closest_t, -2, best_ind, keepdim=False)
                t = t / t.amax()
                m3 = t <= self.expand_verts_to_frags((portion_of_curve_drawn), repeats_inds)
                return m, m2 & m3

            interior_mask, border_mask = update_closest_dist_for_t(ts)

            self.memory.reset_pointer
            return interior_mask, border_mask

        aa_offsets = torch.linspace(0, 1, anti_alias_level * 2 + 1, device=fragment_x.device)[1:-1:2]
        fragment_coords = torch.cat((fragment_x, fragment_y), -1).float()
        if self.render_with_distance:
            all_mask, border_mask = get_within_circuit_mask_with_distance(control_points, fragment_coords)
            closest_dot = None
        else:
            all_mask, border_mask, closest_dot, closest_dist = get_within_circuit_mask_without_dist(control_points, fragment_coords)
            all_mask[:] = False

        if self.render_with_distance:
            inds = fragment_x + (fragment_y) * window_width
            repeats_inds = repeats_inds_circuit
        else:
            fragment_x_full, fragment_y_full, repeats_inds_full, bounding_corners_full, bounding_box_sizes_full, _, bb_num_pixels_full, fragment_x_base_full, fragment_ind_vertical_full = get_fragment_coords(
                corners_inds, batch_time_dim=True)

            row_start_mask = (fragment_x_full - self.expand_verts_to_frags(bounding_corners_full[...,0,:1], repeats_inds_full)) == 0

            fragment_coords = torch.cat((fragment_x_full, fragment_y_full), -1).float()
            bounding_corners_bl_circuit = self.expand_verts_to_frags(bounding_corners_full[..., 0, :],
                                                                     repeats_inds_circuit)
            bounding_box_widths_circuit = self.expand_verts_to_frags(bounding_box_sizes_full[..., :1],
                                                                     repeats_inds_circuit).clamp_min_(1)

            fragment_inds_circuit_horizontal = (fragment_x - bounding_corners_bl_circuit[..., :1]) + (
                    fragment_y - bounding_corners_bl_circuit[...,
                                 1:]) * bounding_box_widths_circuit

            def get_extended_mask(fragment_inds_circuit, vert=False):
                bb_offsets = bounding_box_sizes_full.amax(0, keepdim=True).prod(-1, keepdim=True)
                bb_offsets = bb_offsets.cumsum(-2) - bb_offsets
                fragment_inds_circuit = fragment_inds_circuit + self.expand_verts_to_frags(bb_offsets, repeats_inds_circuit)

                full_mask = self.get_tensor(fragment_x_full.shape)
                full_mask[:] = 0

                full_dist_mask = self.get_tensor(fragment_x_full.shape)
                full_dist_mask[:] = 0
                # TODO try changing this fragment_inds_circuit t- just fragment_inds_circuit (i.e. the first one).
                # I think that we can just scatter into the correct index and itll work fine.
                fragment_inds_circuit = fragment_inds_circuit.clamp_max(full_mask.shape[1] - 1)
                full_mask_border = torch.scatter_reduce(full_mask, -2, fragment_inds_circuit, (border_mask.float()),
                                                        reduce='amax')

                full_mask2 = self.get_tensor(fragment_x_full.shape)
                full_mask2[:] = 1e12

                full_mask_combine = torch.scatter_reduce(full_mask2, -2, fragment_inds_circuit, (closest_dist), reduce='amin', include_self=True)
                min_dists = torch.gather(full_mask_combine, -2, fragment_inds_circuit) + 1e-1
                m = closest_dist > min_dists
                closest_dot2 = closest_dot
                closest_dot2[m] = 0
                full_mask_edge = torch.scatter_reduce(full_mask, -2, fragment_inds_circuit, (closest_dot2), reduce='sum', include_self=False)
                full_mask_pos = torch.scatter_reduce(full_mask, -2, fragment_inds_circuit, (closest_dot2), reduce='amax', include_self=True) >= 0.5
                full_mask_neg = torch.scatter_reduce(full_mask, -2, fragment_inds_circuit, (closest_dot2), reduce='amin', include_self=True) <= -0.5
                full_mask_edge[full_mask_pos & full_mask_neg] = -1

                full_mask = full_mask_edge
                full_mask = torch.where(row_start_mask, -1, full_mask)
                switch_on = full_mask >= 1
                switch_off = (full_mask <= -1)

                fill_inds = torch.arange(full_mask.shape[-2], device=full_mask.device).unsqueeze(-1)
                m = switch_on | switch_off
                fill_inds = fill_inds * m + (~m) * 0
                fill_inds = torch.cummax(fill_inds, -2)[0]
                all_mask = broadcast_gather(full_mask, -2, fill_inds) > 0.5
                return all_mask, full_mask_border * all_mask, None

            all_mask, full_mask_border, full_mask_edge = get_extended_mask(fragment_inds_circuit_horizontal)
            border_mask = full_mask_border > 0.5
            repeats_inds = repeats_inds_full

            max_height = self.expand_verts_to_frags(bounding_corners_full[..., 0, 1:] + bounding_box_sizes_full[..., 1:], repeats_inds_full)
            m_height = (fragment_y_full <= max_height).view(-1,1)
            fragment_x_full = fragment_x_full.reshape(-1, 1)
            fragment_y_full = fragment_y_full.reshape(-1, 1)
            m = (fragment_x_full < window_width) & (fragment_y_full < window_height) & m_height
            m = m.view(-1)
            inds = fragment_x_full + (fragment_y_full) * window_width
            inds = inds[m]
            all_mask = all_mask.reshape(-1,1)[m]
            border_mask = border_mask.reshape(-1,1)[m]
            m_outside_screen = m
        screen_size = screen_width * screen_height

        if not self.filled:
            all_mask[:] = 0
        m = (inds < (screen_size)) & ((all_mask > 0) | (border_mask > 0))
        m = m.reshape(-1)
        border_mask = border_mask.view(-1)[m].unsqueeze(-1)
        g_offsets = torch.arange(fragment_x.shape[0], device=inds.device) * screen_size
        if not self.render_with_distance:
            g_offsets = self.expand_verts_to_frags(g_offsets.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(repeats_inds_full))
            g_offsets = squish(g_offsets,0,1)[m_outside_screen]
        else:
            g_offsets = unsqueeze_right(g_offsets, inds)
        inds = inds + g_offsets
        inds = inds.view(-1)
        inds = inds[m]
        unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        expanded_corners = self.expand_verts_to_frags(corners[..., 0, :], repeats_inds, -2)
        normals_ = unsquish(self.expand_verts_to_frags(squish(normals[..., 0, :, :], -2, -1), repeats_inds, -2), -1,
                            3)
        ray_direction = F.normalize(
            (screen_point.squeeze(-2) + (
            ((fragment_coords[..., 0].unsqueeze(-1) - screen_width * 0.5) / (screen_height * 0.5))) * screen_basis[..., 0, :] +
             (((fragment_coords[..., 1].unsqueeze(-1) - screen_height * 0.5) / (screen_height * 0.5))) * screen_basis[..., 1,
                                                                                                 :]) - ray_origin.squeeze(
                -2), p=2, dim=-1)
        dists = self.raycast_onto_plane(ray_origin.squeeze(-2), ray_direction,
                                        expanded_corners, normals_[..., -1, :])
        if self.num_texture_points > 0:
            proj_onto_mobs = ray_origin.squeeze(-2) + dists * ray_direction

            def expo(x):
                x = select_time(x)
                return self.expand_verts_to_frags(x.squeeze(-2), repeats_inds)
            mob_centers = expo(self.mob_center)
            mob_basis1 = expo(self.basis1)
            mob_basis2 = expo(self.basis2)
            grid_width = expo(self.grid_width).long()
            grid_height = expo(self.grid_height).long()
            disps_from_mobs = proj_onto_mobs - mob_centers
            offsets = expo((torch.arange(self.colors.shape[1], device=self.colors.device)*self.colors.shape[2]).view(1,-1,1,1))
            def get_c(b):
                c = dot_product(F.normalize(b, p=2, dim=-1), disps_from_mobs) / b.norm(p=2, dim=-1, keepdim=True)
                return (c * 0.5 + 0.5)
            x = (get_c(mob_basis2) * (grid_height)).clamp_max_(grid_height-1).clamp_min_(0)
            y = (get_c(mob_basis1) * grid_width).clamp_max_(grid_width-1).clamp_min_(0)
            xr = x % 1
            yr = y % 1
            w1 = (1-xr) * (1-yr)
            w2 = xr * (1-yr)
            w3 = (1-xr)*yr
            w4 = xr * yr

            x_floor = (x).floor().long()
            x_ciel = (x).ceil().long()
            y_floor = (y).floor().long()
            y_ciel = (y).ceil().long()

            colos = squish(select_time(self.colors), 1, 2)
            interpolated_colors = 0
            sum_w = 0
            for w, x, y in [(w1, x_floor, y_floor), (w2, x_ciel, y_floor), (w3, x_floor, y_ciel), (w4, x_ciel, y_ciel)]:
                c = (x + (grid_height) * y) + offsets
                sum_w = sum_w + w
                interpolated_colors = interpolated_colors + w * broadcast_gather(colos, -2, (c), keepdim=True)
            interpolated_colors /= sum_w

        else:
            interpolated_colors = self.expand_verts_to_frags(colors, repeats_inds, -2)
        if not self.render_with_distance:
            interpolated_colors = interpolated_colors.view(-1,interpolated_colors.shape[-1])[m_outside_screen]

        output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        output_frags[:] = 0
        current_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))

        if unique_counts.numel() == 0:
            max_buffer_depth = 1
        else:
            max_buffer_depth = unique_counts.amax()

        def get_frags(ws, fragment_coords=fragment_coords):

            def get_colors():
                colors = interpolated_colors
                border_colors_frags = self.expand_verts_to_frags(border_colors, repeats_inds)
                colors = colors.reshape(-1, colors.shape[-1])
                colors = colors[m]

                border_colors_frags = border_colors_frags.reshape(-1, border_colors_frags.shape[-1])
                if not self.render_with_distance:
                    border_colors_frags = border_colors_frags[m_outside_screen]
                border_colors_frags = border_colors_frags[m]
                colors[...,:] = colors[...,:] * (~border_mask) + border_mask * border_colors_frags
                return colors

            colors = get_colors()
            dists2 = dists.reshape(-1)
            if not self.render_with_distance:
                dists2 = dists2[m_outside_screen]
            dists2 = dists2[m]

            return colors, dists2

        colors, dists = get_frags(1)
        return colors, dists, inds
