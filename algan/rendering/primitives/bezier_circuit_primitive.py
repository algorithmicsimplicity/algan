import math

import torch
import torch.nn.functional as F
import torch_scatter

from algan.constants.color import BLUE, BLACK
from algan.defaults.device_defaults import DEFAULT_RENDER_DEVICE
from algan.geometry.geometry import intersect_line_with_plane, project_point_onto_line, project_point_onto_line_segment
from algan.rendering.primitives.primitive import InsufficientMemoryException, RenderPrimitive2D
from algan.utils.plotting_utils import plot_tensor
from algan.utils.tensor_utils import broadcast_all, broadcast_scatter
from algan.utils.tensor_utils import dot_product, squish, broadcast_gather, expand_as_left, unsquish, unsqueeze_right


def evaluate_cubic_bezier_old3(p, t):
    out = ((1 - t) ** 3) * p[..., 0, :].unsqueeze(-2)
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :].unsqueeze(-2)
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :].unsqueeze(-2)
    out[:] += (t ** 3) * p[..., 3, :].unsqueeze(-2)
    return out


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
    p0 = p[..., 0, :].unsqueeze(-2)
    p1 = p[..., 1, :].unsqueeze(-2)
    p2 = p[..., 2, :].unsqueeze(-2)
    p3 = p[..., 3, :].unsqueeze(-2)
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


class BezierCircuitPrimitive(RenderPrimitive2D):
    def __init__(self, corners=None, next_segment_inds=None, num_segments_per_circuit=None, colors=BLUE, opacity=1, normals=None, border_width=None, border_color=None, portion_of_curve_drawn=None,
                 mob_center=None, grid_width=None, grid_height=None, first_basis=None, second_basis=None,
                 triangle_collection=None, glow=0, num_texture_points=0, filled=True):

        self.num_bezier_parameters = 4
        self.num_texture_points = num_texture_points
        self.filled = filled
        if triangle_collection is not None:
            self.num_segments_per_circuit = torch.cat([_.num_segments_per_circuit for _ in triangle_collection]).to(DEFAULT_RENDER_DEVICE, non_blocking=True)
            self.num_segments_per_object = torch.stack([_.num_segments_per_circuit.sum() for _ in triangle_collection]).to(DEFAULT_RENDER_DEVICE, non_blocking=True)

            self.num_texture_points = triangle_collection[0].num_texture_points
            self.filled = triangle_collection[0].filled
            self.corners = torch.cat([_.corners for _ in triangle_collection], -3).to(DEFAULT_RENDER_DEVICE, non_blocking=True)
            self.colors = torch.cat([_.colors for _ in triangle_collection], -3).to(DEFAULT_RENDER_DEVICE, non_blocking=True)
            if self.num_texture_points == 0:
                self.colors = self.colors.squeeze(-2)
            self.next_segment_inds = torch.cat([_.next_segment_inds for _ in triangle_collection], -3).to(DEFAULT_RENDER_DEVICE, non_blocking=True)
            self.next_segment_inds = self.next_segment_inds + torch.arange(self.next_segment_inds.shape[-3], device=self.next_segment_inds.device).view(-1,1,1)

            self.normals, self.border_width, self.border_color, self.portion_of_curve_drawn = (
                ((torch.cat([(__) for __ in _], -2))).to(DEFAULT_RENDER_DEVICE, non_blocking=True) for _ in
                zip(*((
                    triangle.normals, triangle.border_width, triangle.border_color, triangle.portion_of_curve_drawn)
                    for triangle in triangle_collection)))

            self.mob_center, self.grid_width, self.grid_height, self.basis1, self.basis2 = (
                ((torch.stack([(__) for __ in _], 1))).to(DEFAULT_RENDER_DEVICE, non_blocking=True) for _ in
                zip(*(broadcast_all((
                    triangle.mob_center,
                    triangle.grid_height,
                    triangle.grid_width,
                    triangle.basis1,
                    triangle.basis2,
                ), [-1]) for triangle in triangle_collection)))
            #self.border_width = self.border_width[...,0,:1]
            #self.border_color = self.border_color[...,0,:]
            if self.num_texture_points <= 0:
                self.colors = self.colors#[..., 0, :]
            else:
                self.colors = self.colors[...,(-self.num_texture_points):, :]
            #self.portion_of_curve_drawn = self.portion_of_curve_drawn[...,0,:1]
            return
        self.corners = corners
        self.next_segment_inds = next_segment_inds
        self.num_segments_per_circuit = num_segments_per_circuit
        border_color, opacity, glow = broadcast_all([border_color, opacity, glow], ignored_dims=[-1])
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow.unsqueeze(-2)
        self.colors[..., -1:] *= opacity.unsqueeze(-2)
        self.normals = normals
        self.border_width, self.border_color, self.portion_of_curve_drawn = border_width, border_color, portion_of_curve_drawn
        self.border_color[..., -2:-1] += glow
        self.border_color[..., -1:] *= opacity
        self.mob_center = mob_center
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.basis1 = first_basis
        self.basis2 = second_basis

    def get_batch_identifier(self):
        return f'{__class__}_{self.num_texture_points}_{self.filled}'

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
        mob_center = select_time(self.mob_center)
        colors = select_time(self.colors)
        border_colors = select_time(self.border_color)
        border_width = select_time(self.border_width)
        portion_of_curve_drawn = select_time(self.portion_of_curve_drawn)
        screen_point = select_time(screen_point)
        screen_basis = select_time(screen_basis)
        ray_origin = select_time(ray_origin)

        num_objects = len(self.num_segments_per_object)#

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

        padding = max(20, border_width.amax().ceil().long()+1)

        def get_bounding_box_fragment_coords(x):
            arange_num_segments_per_oject = torch.arange(len(self.num_segments_per_object), device=x.device)
            segment_to_object_scatter_inds = torch.repeat_interleave(arange_num_segments_per_oject,
                                                             self.num_segments_per_object*self.num_bezier_parameters, -1).view(1,-1,1)

            arange_num_segments_per_oject = arange_num_segments_per_oject.view(1,-1,1)
            object_bounding_corners_bottom_left = (broadcast_scatter(arange_num_segments_per_oject, -2,
                        segment_to_object_scatter_inds, x, reduce='amin', include_self=False) - padding).clamp_(
                min=torch.tensor((start_x, start_y), device=x.device),
                max=torch.tensor((end_x, end_y), device=x.device))
            object_bounding_corners_top_right = (broadcast_scatter(arange_num_segments_per_oject, -2,
                        segment_to_object_scatter_inds, x, reduce='amax', include_self=False) + padding).clamp_(
                min=torch.tensor((start_x, start_y), device=x.device),
                max=torch.tensor((end_x, end_y), device=x.device))

            object_bounding_box_dimensions = object_bounding_corners_top_right - object_bounding_corners_bottom_left
            object_bounding_box_num_pixels = object_bounding_box_dimensions.prod(-1, keepdim=True)

            num_fragments = object_bounding_box_num_pixels.sum()
            mem_per_fragment = 256
            total_mem_required = num_fragments * mem_per_fragment

            free_mem = self.memory.get_num_bytes_remaining()

            if free_mem < total_mem_required:
                raise InsufficientMemoryException

            object_to_fragment_gather_inds = torch.repeat_interleave(
                torch.arange(object_bounding_box_num_pixels.numel(),
                             device=x.device), object_bounding_box_num_pixels.view(-1), -1,
                output_size=num_fragments).unsqueeze(-1)

            object_fragment_inds = torch.arange(num_fragments, device=x.device).view(-1,1)

            object_offsets = (object_bounding_box_num_pixels.view(-1).cumsum(-1) - object_bounding_box_num_pixels.view(-1)).view(-1,1)
            object_fragment_inds = object_fragment_inds - broadcast_gather(object_offsets, -2, object_to_fragment_gather_inds)

            object_bounding_box_dimensions_for_frags = broadcast_gather(squish(object_bounding_box_dimensions,0,1), -2, object_to_fragment_gather_inds)
            object_bounding_corners_bottom_left_for_frags = broadcast_gather(squish(object_bounding_corners_bottom_left,0,1), -2, object_to_fragment_gather_inds)
            object_fragment_x = (object_fragment_inds % object_bounding_box_dimensions_for_frags[...,:1]) + object_bounding_corners_bottom_left_for_frags[...,:1]
            object_fragment_y_bbox = (object_fragment_inds // object_bounding_box_dimensions_for_frags[..., :1])
            object_fragment_y = object_fragment_y_bbox + object_bounding_corners_bottom_left_for_frags[...,1:]

            return (object_fragment_x, object_fragment_y, object_fragment_y_bbox, object_fragment_inds, object_bounding_box_dimensions,
                    object_bounding_corners_bottom_left, object_to_fragment_gather_inds)

        (fragment_x, fragment_y, fragment_y_bbox, fragment_inds,
            object_bounding_box_dimensions, object_bounding_corners_bottom_left,
         object_to_fragment_gather_inds
         ) = get_bounding_box_fragment_coords(squish(corners_inds, -3, -2))

        if fragment_x.numel() == 0:
            return None
        control_points = corners_locs

        control_net_lengths = (control_points[...,1:,:] - control_points[...,:-1,:]).norm(p=2, dim=-1).sum(-1)
        maximum_net_length = control_net_lengths.amax()
        num_sampled_points = (maximum_net_length).ceil().long() # 1 sample per 2 pixel widths.
        t = torch.linspace(0, 1, num_sampled_points, device=control_points.device)
        ##polygon_vertices = self.get_tensor((*control_points.shape[:3], num_sampled_points, 2))
        polygon_vertices = evaluate_cubic_bezier_old3(control_points, t.unsqueeze(-1))#, polygon_vertices, self.memory)
        # assert polygon_vertices.shape == [T, N, P, 2] (time (frames), num segments, num control points per segment, 2D)
        #polygon_vertices = squish(polygon_vertices, -3, -2)  # shape [T, N, S*P, 2]
        next_polygon_vertices = polygon_vertices.roll(shifts=-1, dims=-2)

        # Change the last next_vertice from the start of this segment to the start of the next segment.
        next_segments = broadcast_gather(polygon_vertices, -3, self.next_segment_inds)
        next_polygon_vertices[...,-1,:] = next_segments[...,0,:]

        line_segments = next_polygon_vertices - polygon_vertices
        line_segment_lengths = line_segments.norm(p=2, dim=-1)

        # Now that we have approximated the bezier circuits as polygons, we need to rasterize the polygons.
        # The basic plan is, around each polygon vertex we look at the local neighbourhood of pixels.
        # In this local window, we identify all pixels that cross the line segment joining this vertex to the next.
        # Then we scatter_add all local windows together into the final image. The result is that in the final
        # image each pixel will contain a count of the number of line segments it intersects.
        # When then cumsum across rows to get the number of intersections to the left of each pixel,
        # and use the polarity rule num_intersections % 2 == 1 to determine the interior.

        # We need to ensure that the local window is large enough to completely cover the largest line segment,
        # otherwise there will be holes in the border.
        if self.filled:
            half_local_window_size = max(border_width.amax().ceil().long(), line_segment_lengths.amax().ceil().long())
        else:
            half_local_window_size = border_width.amax().ceil().long()
        local_window_size = half_local_window_size * 2 + 1 + 1
        local_window_inds = torch.arange(local_window_size * local_window_size, device=control_points.device)

        # we subtract half_local_window_size so that in local coord (0,0) is the center (i.e. line start).
        local_window_x = local_window_inds % local_window_size - half_local_window_size
        local_window_y = local_window_inds // local_window_size - half_local_window_size

        line_start_x = polygon_vertices[...,:1]# % 1
        line_start_y = polygon_vertices[...,1:]# % 1

        # Note all line segments are centered, so they start at (0, 0)
        line_end_x = next_polygon_vertices[...,:1]#line_segments[..., :1] + line_start_x
        line_end_y = next_polygon_vertices[...,1:]#line_segments[..., 1:] + line_start_y

        local_window_x = local_window_x + line_start_x.floor().long()
        local_window_y = local_window_y + line_start_y.floor().long()

        # First we check that the local pixel is within the horizontal extent of the line segment.
        horizontal_mask = (local_window_y < line_end_y) != (local_window_y < line_start_y)
        # I think the current horizontal mask double counts the end points if local_window_y == line_start_y,
        # might need to switch to something like:
        #horizontal_mask = (((line_start_y < local_window_y) & (local_window_y < line_end_y)) |
        #                   ((line_start_y > local_window_y) & (local_window_y > line_end_y)))
        #end_point_mask = ((line_start_y == local_window_y) | (local_window_y == line_end_y))


        # Then we check that if we move from the pixel center to the right edge of the pixel
        # we move from one side of the line to the other. If so, that means that this pixel intersects
        # with the line (i.e. polygon boundary).
        intersect_mask = (((local_window_x - line_start_x) * -line_segments[...,1:]
                           + (local_window_y - line_start_y) * line_segments[...,:1]) < 0) != \
                         (((local_window_x + 1 - line_start_x) * -line_segments[..., 1:]
                           + (local_window_y - line_start_y) * line_segments[..., :1]) < 0)

        local_intersection_counts = (horizontal_mask & intersect_mask).float()#.view(-1)

        '''local_to_global_inds = ((polygon_vertices_int[...,:1] + local_window_x).clamp_min(1) +
                                (polygon_vertices_int[...,1:] + local_window_y) * bounding_box_widths.unsqueeze(-1)
                                ).clamp_(min=torch.zeros_like(bounding_box_num_pixels.unsqueeze(-1)), max=bounding_box_num_pixels.unsqueeze(-1)-1)'''
        object_bounding_box_dimensions_for_segments = torch.repeat_interleave(object_bounding_box_dimensions, self.num_segments_per_object, -2).unsqueeze(-1)
        object_bounding_corners_bottom_left_for_segments = torch.repeat_interleave(object_bounding_corners_bottom_left, self.num_segments_per_object, -2).unsqueeze(-1)
        bbox_x = local_window_x - object_bounding_corners_bottom_left_for_segments[...,:1,:]
        bbox_y = local_window_y - object_bounding_corners_bottom_left_for_segments[...,1:,:]
        bbox_num_pixels = object_bounding_box_dimensions_for_segments.prod(-2, keepdim=True)
        local_to_bbox_inds = (bbox_x.clamp_min(0) + bbox_y * object_bounding_box_dimensions_for_segments[...,:1,:]
                                ).clamp_(min=torch.zeros_like(bbox_num_pixels),
                                         max=bbox_num_pixels - 1)

        # local_to_bbox_inds scatters from local_window into object level bounding box.
        # Now we need to add offsets so that inds from different objects end up in different output frames.
        offsets = object_bounding_box_dimensions.prod(-1, keepdims=True).view(-1,1)
        offsets = offsets.cumsum(-2)  - offsets
        offsets_for_segments = squish(torch.repeat_interleave(unsquish(offsets, 0, -corners.shape[0]), self.num_segments_per_object, -2).unsqueeze(-1), 0, 1)
        local_to_global_inds = (squish(local_to_bbox_inds, 0, 1) + offsets_for_segments.view(-1,1,1)).view(-1)

        #invalid_mask = ((bbox_x < 0) | (bbox_x > bounding_box_widths.unsqueeze(-2))) | (((bbox_y < 0) | (bbox_y > bounding_box_heights.unsqueeze(-2))))
        invalid_mask = ((bbox_x >= object_bounding_box_dimensions_for_segments[...,:1,:]) |
                        (bbox_y < 0) | (bbox_y > object_bounding_box_dimensions_for_segments[...,1:,:]))
        # Note we need to keep negative x inds around for now, because we cumsum across rows from the left
        # to count intersections, we will cull negative x inds later.

        local_intersection_counts = torch.where(invalid_mask, 0, local_intersection_counts)

        global_intersection_counts = torch_scatter.scatter_sum(local_intersection_counts.view(-1),
                                                               local_to_global_inds.clamp(min=0, max=fragment_x.shape[-2]-1), -1, dim_size=fragment_x.shape[-2])

        # Now do border mask.
        local_window_xy = torch.stack((local_window_x, local_window_y), -1)
        local_proj_onto_line = project_point_onto_line_segment(local_window_xy, polygon_vertices.unsqueeze(-2), next_polygon_vertices.unsqueeze(-2))
        local_dist = (local_window_xy - local_proj_onto_line).norm(p=2, dim=-1)
        dist_invalid_mask = invalid_mask | (bbox_x <0)
        local_dist = torch.where(dist_invalid_mask, 1e12, local_dist)
        global_dists = torch.empty((fragment_x.shape[-2],), device=control_points.device)
        global_dists[:] = 1e12
        global_dists = torch_scatter.scatter_min(local_dist.view(-1),
                                                               local_to_global_inds.clamp(min=0,
                                                                                          max=fragment_x.shape[-2] - 1),
                                                               -1, out=global_dists)[0]

        border_mask = (global_dists.unsqueeze(-1) < self.expand_verts_to_frags(squish(border_width, 0, 1), object_to_fragment_gather_inds)).float()

        # Count the number of intersections in the horizontal ray to this pixel's left.
        left_intersection_counts = global_intersection_counts.cumsum(-1)
        row_start_ind = (self.expand_verts_to_frags(offsets.view(-1,1), object_to_fragment_gather_inds) +
                         (fragment_y_bbox)*self.expand_verts_to_frags(object_bounding_box_dimensions[...,:1].view(-1,1), object_to_fragment_gather_inds)).view(-1)
        row_start_ind = (row_start_ind-1).clamp_min(0)

        left_intersection_counts = left_intersection_counts - broadcast_gather(left_intersection_counts, -1, row_start_ind)

        interior_mask = ((left_intersection_counts % 2) == 1).float().unsqueeze(-1)

        fragment_coords = torch.cat((fragment_x, fragment_y), -1).float()

        inds = fragment_x + (fragment_y) * screen_width
        screen_size = screen_width * screen_height

        if not self.filled:
            interior_mask[:] = 0
        m = (inds < screen_size) & ((interior_mask > 0) | (border_mask > 0))
        m = m.reshape(-1)
        border_mask = border_mask.view(-1)[m].unsqueeze(-1)
        g_offsets = torch.arange(0, corners.shape[0], device=inds.device) * screen_size
        frame_to_fragment_gather_inds = object_to_fragment_gather_inds // num_objects
        g_offsets = self.expand_verts_to_frags(g_offsets.unsqueeze(-1), frame_to_fragment_gather_inds)
        inds = inds + g_offsets
        inds = inds.view(-1)
        inds = inds[m]
        #unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        mob_center_for_frags = self.expand_verts_to_frags(squish(mob_center, 0, 1).squeeze(-2), object_to_fragment_gather_inds)
        normals_for_frags = self.expand_verts_to_frags(squish(normals,0,1), object_to_fragment_gather_inds)

        def expo(x, gather_inds=object_to_fragment_gather_inds):
            x = select_time(x)
            x = x.view(-1,x.shape[-1])
            return self.expand_verts_to_frags(x, gather_inds)

        screen_basis = unsquish(expo(squish(screen_basis,-2,-1), gather_inds=frame_to_fragment_gather_inds), -1, 3)
        screen_point = expo(screen_point, gather_inds=frame_to_fragment_gather_inds)
        ray_origin = expo(ray_origin, gather_inds=frame_to_fragment_gather_inds)
        screen_basis = screen_basis / screen_basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        ray_direction = F.normalize(
            (screen_point + (((fragment_coords[..., :1] - screen_width * 0.5) /
                              (screen_height * 0.5))) * screen_basis[..., 0, :] +
             (((fragment_coords[..., 1:] - screen_height * 0.5) /
                                (screen_height * 0.5))) * screen_basis[..., 1, :]) - ray_origin, p=2, dim=-1)
        dists = (self.raycast_onto_plane(ray_origin, ray_direction,
                                        mob_center_for_frags, normals_for_frags))
        if self.num_texture_points > 0:
            proj_onto_mobs = ray_origin + dists * ray_direction
            mob_centers = expo(self.mob_center)
            mob_basis1 = expo(self.basis1)
            mob_basis2 = expo(self.basis2)
            grid_width = expo(self.grid_width).long()
            grid_height = expo(self.grid_height).long()
            disps_from_mobs = proj_onto_mobs - mob_centers
            offsets = expo((torch.arange(self.mob_center.shape[1]*(time_end - time_start), device=self.colors.device)*self.colors.shape[-2]).view(1,-1,1,1))
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

            colos = squish(select_time(self.colors), 0, 2)
            interpolated_colors = 0
            sum_w = 0
            for w, x, y in [(w1, x_floor, y_floor), (w2, x_ciel, y_floor), (w3, x_floor, y_ciel), (w4, x_ciel, y_ciel)]:
                c = (x + (grid_height) * y) + offsets
                sum_w = sum_w + w
                interpolated_colors = interpolated_colors + w * broadcast_gather(colos, -2, (c), keepdim=True)
            interpolated_colors /= sum_w
        else:
            interpolated_colors = self.expand_verts_to_frags(squish(colors, 0, 1), object_to_fragment_gather_inds, -2)

        #output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        #output_frags[:] = 0

        def get_frags(ws, fragment_coords=fragment_coords):

            def get_colors():
                colors = interpolated_colors
                border_colors_frags = self.expand_verts_to_frags(squish(border_colors,0,1), object_to_fragment_gather_inds)
                colors = colors.reshape(-1, colors.shape[-1])
                colors = colors[m]

                border_colors_frags = border_colors_frags.reshape(-1, border_colors_frags.shape[-1])
                border_colors_frags = border_colors_frags[m]
                colors[...,:] = colors[...,:] * (1-border_mask) + border_mask * border_colors_frags
                return colors

            colors = get_colors()
            dists2 = dists.reshape(-1)
            dists2 = dists2[m]

            return colors, dists2

        colors, dists = get_frags(1)
        return colors, dists, inds
