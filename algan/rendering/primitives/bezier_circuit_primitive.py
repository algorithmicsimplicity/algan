import math

import torch
import torch.nn.functional as F

from algan import compiled
from algan.constants.color import BLUE, BLACK
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.geometry.geometry import (
    intersect_line_with_plane,
    project_point_onto_line,
    project_point_onto_line_segment,
)
from algan.rendering.primitives.primitive import (
    InsufficientMemoryException,
    RenderPrimitive2D,
)
from algan.utils.tensor_utils import broadcast_all, broadcast_scatter
from algan.utils.tensor_utils import (
    dot_product,
    squish,
    broadcast_gather,
    expand_as_left,
    unsquish,
    unsqueeze_right,
)


def evaluate_cubic_bezier(p, t, out, mem, i=-1):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    mem.save_pointer()
    temp_t = mem.get_tensor(t.shape)
    temp_t_2 = mem.get_tensor(t.shape)
    temp = mem.get_tensor(out.shape)
    _1_m_t = torch.subtract(1, t, out=temp_t)
    _1_m_t3 = torch.pow(_1_m_t, 3, out=temp_t)
    _1_m_t3p0 = torch.mul(p0, _1_m_t3, out=out)
    if i == 0:
        return out
    # out[:] = ((1 - t) ** 3) * p[..., 0, :]
    tp1 = torch.mul(t, p1, out=temp)
    _1_m_t = torch.subtract(1, t, out=temp_t)
    _1_m_t2 = torch.pow(_1_m_t, 2, out=_1_m_t)
    torch.addcmul(_1_m_t3p0, _1_m_t2, tp1, value=3, out=out)
    # out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]
    if i == 1:
        return out
    # Is this a Pytorch bug?
    # torch.square(t, out=temp[...,:1])
    temp_t[:] = t
    temp_t.square_()
    # torch.subtract(1,t,out=temp2[...,:1])
    temp_t_2[:] = t
    temp_t_2 *= -1
    temp_t_2 += 1
    temp_t *= temp_t_2
    # torch.mul(temp[...,:1],temp2[...,:1],out=temp[...,:1])
    torch.addcmul(out, temp_t, p2, value=3, out=out)
    # out[:] += 3 * (1 - t) * t * t * p[..., 2, :]
    if i == 2:
        return out
    # torch.pow(t, 3, out=temp[...,:1])
    temp_t[:] = t
    temp_t **= 3
    torch.addcmul(out, temp_t, p3, out=out)
    # out[:] += (t ** 3) * p[..., 3, :]
    mem.reset_pointer()
    return out


def batch_arange(lengths, memory=None):
    if memory is None:
        offsets = lengths.cumsum(0)
        n = offsets[-1].clone()
        offsets -= lengths
        offsets = torch.repeat_interleave(offsets, lengths, output_size=n)
        return torch.arange(n, device=lengths.device) - offsets

    start_pointer = memory.current_pointer
    start_reverse_pointer = memory.current_reverse_pointer
    offsets = torch.cumsum(lengths, 0, out=memory.get_tensor(lengths.shape, lengths.dtype))
    n = offsets[-1].clone()
    offsets -= lengths
    offsets = torch.repeat_interleave(offsets, lengths, output_size=n)
    inds = torch.arange(n, device=lengths.device, out=memory.get_tensor((n,), dtype=torch.long, persist=True))
    inds -= offsets
    memory.current_pointer = start_pointer
    inds = memory.cast(inds, torch.int)
    memory.current_reverse_pointer = start_reverse_pointer
    return inds


def get_distance_to_line_segment(points, p1, p2, memory=None):
    projected_points = project_point_onto_line_segment(
        points,
        p1, p2,
        memory=memory
    )
    disps = torch.sub(points, projected_points, out=projected_points)
    disps.square_()
    dists = torch.sum(disps, -1, keepdim=True, out=memory.get_tensor([*disps.shape[:-1], 1], persist=True))
    return dists
    #return torch.zeros_like(dists)


def rasterize_rectangle_by_grid_transform(c1, c2, edge2, memory):
    """Rasterizes a batch of 2D rectangles using a grid transformation method.

    This algorithm creates a dense grid in the rectangle's local (u,v) space
    and transforms these points into world coordinates. It guarantees full coverage
    but may produce duplicate points.

    Parameters
    ----------
    rectangle : (torch.Tensor)
        A tensor of shape [n, 5] where n is the
        batch size. Each row contains the parameters of a rectangle:
        [c1_x, c1_y, c2_x, c2_y, height], where (c1_x, c1_y) is the first
        corner, (c2_x, c2_y) is an adjacent corner, and height is the
        perpendicular distance to the opposite edge.

    Returns
    -------
    torch.Tensor
        A tensor of shape [m, 2] containing the integer
        coordinates of all points inside the rectangles. 'm' is the
        total number of such points (including duplicates).
    """
    device = c1.device
    n = c1.shape[1]

    with memory.temp():
        edge1 = torch.sub(c2, c1, out=memory.get_tensor(c1.shape))

        # === Determine Grid Density for Each Parallelogram ===
        # We need enough steps along each edge to not miss any integer grid lines.
        # The theoretical number of steps for right-angled rectangles is the
        # ceiling of the vector's norm times sqrt(2),
        # but in practice we need a bit of leeway, so we use 1.75 to be safe.
        sqrt_2 = 1.75#

        padding = 1

        def get_num_steps(edge):
            edge_len = torch.norm(edge, p=2, dim=-1, keepdim=True, out=memory.get_tensor([*c1.shape[:-1], 1]))
            max_len = torch.amax(edge_len, 0, out=memory.get_tensor(edge_len.shape[1:]))
            max_len *= sqrt_2
            max_len.ceil_()
            max_len = memory.cast(max_len, torch.int)
            max_len += padding * 2
            return max_len

        num_steps = [get_num_steps(edge) for edge in [edge1, edge2]]
        num_points_per_para = torch.mul(num_steps[0], num_steps[1], out=memory.get_tensor(num_steps[0].shape, torch.int)).view(-1)

        total_points = num_points_per_para.sum()

        # Create an index mapping each final point back to its original parallelogram
        para_indices = torch.repeat_interleave(torch.arange(n, device=device), num_points_per_para).unsqueeze(-1)

        # Generate local grid indices (0, 1, 2, ...) for each parallelogram's flattened grid
        end_indices = torch.cumsum(num_points_per_para, dim=0, out=memory.get_tensor(num_points_per_para.shape, dtype=torch.int))
        start_indices = torch.sub(end_indices, num_points_per_para, out=end_indices)
        local_indices = torch.arange(total_points, device=device, out=memory.get_tensor((total_points,), torch.long))
        local_indices -= torch.repeat_interleave(start_indices, num_points_per_para)
        local_indices = local_indices.unsqueeze(-1)

        # Gather the number of u-steps corresponding to each point
        steps_for_points = [broadcast_gather(ns, 0, para_indices, keepdim=True, out=memory) for ns in num_steps]

        # Calculate the u and v indices for each point within its local grid
        u_idx = torch.remainder(local_indices, steps_for_points[0], out=memory.get_tensor(local_indices.shape, torch.int))
        u_idx -= padding
        v_idx = torch.floor_divide(local_indices, steps_for_points[0], out=memory.get_tensor(local_indices.shape, torch.int))
        v_idx -= padding

        # Convert indices to normalized [0, 1] coordinates
        # Denominator is clamped to 1 to avoid division by zero if a dimension has only one step
        for sp in steps_for_points:
            sp -= padding * 2
            sp -= 1
            sp.clamp_min_(1)
        u = torch.divide(u_idx, steps_for_points[0], out=u_idx.view(torch.float))
        edge1_mapped = temp = broadcast_gather(edge1, -2, para_indices, out=memory)
        c1_mapped = broadcast_gather(c1, -2, para_indices, out=memory)#.get_tensor(temp.shape, persist=True))
        torch.addcmul(c1_mapped, edge1_mapped, u, out=c1_mapped)
        v = torch.divide(v_idx, steps_for_points[1], out=v_idx.view(torch.float))
        edge2_mapped = broadcast_gather(edge2, -2, para_indices, out=temp)
        points = torch.addcmul(c1_mapped, edge2_mapped, v, out=c1_mapped)
        points.round_()
        points = memory.cast(points, torch.int, persist=True)
        return points, para_indices


def rasterize_polygon_border(vertices, next_vertices, next_perpendiculars, widths, memory):
    vshape = vertices.shape
    persist_pointer = memory.current_reverse_pointer
    with memory.temp():
        line_segments_orig = torch.sub(next_vertices, vertices, out=memory.get_tensor(vshape))
        line_segments = memory.clone(line_segments_orig)
        line_lengths = torch.norm(line_segments, p=1, dim=-1, keepdim=True, out=memory.get_tensor([*vshape[:-1], 1]))
        eps = 1e-5
        null_mask = torch.lt(line_lengths, eps, out=memory.get_tensor(line_lengths.shape, torch.bool))
        line_lengths.clamp_min_(eps)
        line_segments /= line_lengths
        line_segments = torch.where(null_mask, torch.tensor((1/math.sqrt(2),), device=vertices.device), line_segments, out=line_segments)
        line_perpendiculars = memory.get_tensor(line_segments.shape)
        line_perpendiculars[..., 1] = line_segments[...,0]
        line_perpendiculars[..., 0] = torch.mul(line_segments[..., 1], -1, out=line_perpendiculars[..., 0])
        line_perpendiculars *= widths
        start_corners = memory.get_tensor([2, *vshape])
        start_corners[0] = torch.add(vertices, line_perpendiculars, out=start_corners[0])
        start_corners[1] = torch.sub(vertices, line_perpendiculars, out=start_corners[1])
        #start_corners = [vertices + line_perpendiculars, vertices - line_perpendiculars]
        null_start = memory.get_tensor(vshape)
        line_segments_width = torch.mul(line_segments, widths, out=memory.get_tensor(vshape))
        for start_corner in start_corners:
            null_start = torch.sub(start_corner, line_segments_width, out=null_start)
            torch.where(null_mask, null_start, start_corner, out=start_corner)

        next_perpendiculars = F.normalize(next_perpendiculars, p=2, dim=-1, out=next_perpendiculars)
        next_perpendiculars *= widths
        end_corners = [
            torch.add(line_segments_orig, next_perpendiculars, out=null_start),
            torch.sub(line_segments_orig, next_perpendiculars, out=line_segments_width)
        ]

        dots = [dot_product(end_corners[i], line_segments, out=next_perpendiculars[...,i:i+1]) for i in range(2)]
        dot_abs = [torch.abs(dots[i], out=end_corners[0][...,i:i+1]) for i in range(2)]
        max_dot = torch.where(dot_abs[0] >= dot_abs[1], dots[0], dots[1], out=dots[0])
        widths *= 2 # TODO check the minimum value needed here.
        max_dot = torch.where(null_mask, widths, max_dot, out=max_dot)
        edge = torch.mul(max_dot, line_segments, out=line_segments)
        points, inds = rasterize_rectangle_by_grid_transform(*start_corners, edge, memory)
        local_distance = get_distance_to_line_segment(points, broadcast_gather(vertices, -2, inds, out=memory), broadcast_gather(next_vertices, -2, inds, out=memory), memory)
    points = memory.clone(points)
    local_distance = memory.clone(local_distance)
    memory.current_reverse_pointer = persist_pointer
    return points[...,:1], points[...,1:], inds, local_distance


class TensorExpander:
    def __init__(self, num_repeats=None, expand_gather_inds=None, dim=None, memory=None):
        if num_repeats is not None and expand_gather_inds is None:
            expand_gather_inds = torch.repeat_interleave(torch.arange(len(num_repeats), device=num_repeats.device), num_repeats)
        self.expand_gather_inds = expand_gather_inds
        self.num_repeats = num_repeats
        self.dim = dim
        self.memory = memory

    def __call__(self, x, dim=None):
        d = self.dim
        if dim is not None:
            d = dim
        #return torch.repeat_interleave(x, self.num_repeats, d)
        if d < 0:
            d = d + x.dim()
        inds = self.expand_gather_inds
        for i in range(x.dim() - (d+1)):
            inds = inds.unsqueeze(-1)
        return broadcast_gather(x, d, inds, out=self.memory)


#@squish_batch_dims
def rasterize_polygon(vertices, next_vertices, num_vertices_per_object, memory):
    num_vertices_per_object = num_vertices_per_object.view(-1)
    v_y = vertices[...,1:]
    nv_y = next_vertices[...,1:]
    v_x = vertices[...,:1]
    nv_x = next_vertices[...,:1]
    reverse_pointer = memory.current_reverse_pointer
    with memory.temp():
        vshape = v_x.shape
        vertex_vars = memory.get_tensor([4, *vshape])
        slopes = memory.get_tensor(vshape)
        with memory.temp():
            rises = torch.sub(nv_y, v_y, out=slopes)
            runs = torch.sub(nv_x, v_x, out=memory.get_tensor(vshape))
            slopes = torch.div(runs, rises, out=slopes)

        min_y = torch.minimum(nv_y, v_y, out=vertex_vars[1])
        max_y = torch.maximum(nv_y, v_y, out=vertex_vars[2])
        widths = torch.sub(nv_x, v_x, out=vertex_vars[3])
        widths = widths.abs_().floor_()
        widths = memory.cast(widths, torch.int)
        temp = memory.get_tensor(widths.shape, torch.int)
        temp[:] = widths
        widths = temp

        pos_slope_mask = torch.lt(nv_y, v_y, out=memory.get_tensor(vshape, dtype=torch.bool))

        vertices_y = memory.get_tensor(vshape[:-1], torch.int)
        with memory.temp():
            temp_vertices_y = memory.get_tensor(vshape[:-1], torch.int)
            temp_next_vertices_y = memory.get_tensor(vshape[:-1], torch.int)
            next_vertices_y = memory.get_tensor(vshape[:-1], torch.float)
            next_vertices_y_int = torch.ceil(nv_y.squeeze(-1), out=next_vertices_y)
            temp_next_vertices_y[:] = next_vertices_y_int
            next_vertices_y_int = temp_next_vertices_y

            vertices_y_int = torch.ceil(v_y.squeeze(-1), out=next_vertices_y)
            temp_vertices_y[:] = vertices_y_int
            vertices_y_int = temp_vertices_y

            y_ranges = torch.sub(next_vertices_y_int, vertices_y_int, out=vertices_y)
            y_ranges.abs_()
            y_ranges = torch.amax(y_ranges, 0, out=vertices_y[0])

        expa = TensorExpander(num_repeats=y_ranges, dim=-2, memory=memory)

        vertex_vars_n = expa(vertex_vars[:-1])
        slopes_n, min_y_n, max_y_n = vertex_vars_n

        #y_ranges = (next_vertices[...,1].ceil().int() - vertices[...,1].ceil().int()).abs().amax(0)
        fragment_to_object_inds = torch.repeat_interleave(torch.arange(num_vertices_per_object.shape[0], device=vertices.device), num_vertices_per_object)
        fragment_to_object_inds = torch.repeat_interleave(fragment_to_object_inds, y_ranges)
        inds_y = batch_arange(y_ranges, memory).unsqueeze(-1)
        inds_y_neg = torch.add(inds_y, 1, out=memory.get_tensor(inds_y.shape, inds_y.dtype))
        inds_y_neg *= -1
        vertices = torch.ceil(vertices, out=memory.get_tensor(vertices.shape))
        vertices = memory.cast(vertices, torch.int)
        vertices_n = expa(vertices)
        pos_slope_mask_n = expa(pos_slope_mask)
        widths_n = expa(widths)
        #next_vertices = torch.repeat_interleave(next_vertices, y_ranges, -2)
        inds_y = torch.where(pos_slope_mask_n, inds_y_neg, inds_y)
        inds_x = torch.mul(inds_y, slopes_n, out=memory.get_tensor(inds_y.shape)).round_()
        inds_x = memory.cast(inds_x, torch.int, persist=True)

        inf = torch.tensor(1000000000, device=v_x.device, dtype=torch.int)
        inds_x.nan_to_num_(inf, inf, -inf)
        inds_x.clamp_max_(widths_n)
        widths_n *= -1
        inds_x.clamp_min_(widths_n)
        inds_y += vertices_n[...,1:]#.ceil().int()
        inds_x += vertices_n[..., :1]#.ceil().int()
        #min_y = torch.minimum(vertices[...,1:], next_vertices[...,1:])
        #max_y = torch.maximum(vertices[...,1:], next_vertices[...,1:])
        m = torch.lt(inds_y, min_y_n, out=memory.get_tensor(inds_y.shape, torch.bool))
        m2 = torch.lt(inds_y, max_y_n, out=memory.get_tensor(inds_y.shape, torch.bool))
        m = torch.not_equal(m, m2, out=m)
        #m = (inds_y < min_y) != (inds_y < max_y)
        inds_y = torch.where(m, inds_y, inf, out=memory.get_tensor(inds_y.shape, inds_y.dtype, persist=True))
        #inds_y = torch.where(m, inds_y, torch.full_like(inds_y, inf))
    # Move inds_x out of persistent memory.
    inds_x = memory.clone(inds_x)
    inds_y = memory.clone(inds_y)
    memory.current_reverse_pointer = reverse_pointer
    return inds_x, inds_y, fragment_to_object_inds.unsqueeze(-1)


class BezierCircuitPrimitive(RenderPrimitive2D):
    def __init__(
        self,
        corners=None,
        next_segment_inds=None,
        num_segments_per_circuit=None,
        colors=BLUE,
        opacity=1,
        normals=None,
        border_width=None,
        border_color=None,
        portion_of_curve_drawn=None,
        mob_center=None,
        grid_width=None,
        grid_height=None,
        first_basis=None,
        second_basis=None,
        triangle_collection=None,
        glow=0,
        num_texture_points=0,
        filled=True,
            num_pixels_per_sample=2
    ):
        self.num_pixels_per_sample = num_pixels_per_sample
        self.num_bezier_parameters = 4
        self.num_texture_points = num_texture_points
        self.filled = filled
        if triangle_collection is not None:
            device = COMPUTING_DEFAULTS.render_device
            self.num_segments_per_object = torch.cat(
                [_.num_segments_per_circuit.view(-1) for _ in triangle_collection]
            ).to(device)

            self.num_texture_points = triangle_collection[0].num_texture_points
            self.filled = triangle_collection[0].filled
            self.corners = torch.cat([_.corners for _ in triangle_collection], -3).to(
                device
            )
            self.colors = torch.cat([_.colors for _ in triangle_collection], -3).to(
                device
            )
            if self.num_texture_points == 0:
                self.colors = self.colors.squeeze(-2)
            self.next_segment_inds = torch.cat(
                [_.next_segment_inds for _ in triangle_collection], -3
            ).to(device)
            self.next_segment_inds = self.next_segment_inds + torch.arange(
                self.next_segment_inds.shape[-3], device=self.next_segment_inds.device
            ).view(-1, 1, 1)

            (
                self.normals,
                self.border_width,
                self.border_color,
                self.portion_of_curve_drawn,
            ) = (
                (torch.cat([(__) for __ in _], -2)).to(device)
                for _ in zip(
                    *(
                        (
                            triangle.normals,
                            triangle.border_width,
                            triangle.border_color,
                            triangle.portion_of_curve_drawn,
                        )
                        for triangle in triangle_collection
                    )
                )
            )

            (
                self.mob_center,
                self.grid_width,
                self.grid_height,
                self.basis1,
                self.basis2,
            ) = (
                (torch.cat([(__) for __ in _], 1)).to(device)
                for _ in zip(
                    *(
                        broadcast_all(
                            (
                                triangle.mob_center,
                                triangle.grid_height.int(),
                                triangle.grid_width.int(),
                                triangle.basis1,
                                triangle.basis2,
                            ),
                            [-2, -1],
                        )
                        for triangle in triangle_collection
                    )
                )
            )
            # self.border_width = self.border_width[...,0,:1]
            # self.border_color = self.border_color[...,0,:]
            if self.num_texture_points <= 0:
                self.colors = self.colors  # [..., 0, :]
            else:
                self.colors = self.colors[..., (-self.num_texture_points) :, :]
            self.padding = max(self.border_width.amax().ceil().long()+1, 2)
            # self.portion_of_curve_drawn = self.portion_of_curve_drawn[...,0,:1]
            return
        self.corners = corners
        self.next_segment_inds = next_segment_inds
        self.num_segments_per_circuit = num_segments_per_circuit
        border_color, opacity, glow = broadcast_all(
            [border_color, opacity, glow], ignored_dims=[-1]
        )
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow.unsqueeze(-2)
        self.colors[..., -1:] *= opacity.unsqueeze(-2)
        self.normals = normals
        self.border_width, self.border_color, self.portion_of_curve_drawn = (
            border_width,
            border_color,
            portion_of_curve_drawn,
        )
        self.border_color[..., -2:-1] += glow
        self.border_color[..., -1:] *= opacity
        self.mob_center = mob_center
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.basis1 = first_basis
        self.basis2 = second_basis

    def get_windowed_bounding_boxes(
        self, bounding_corners, screen_width, screen_height, window_coords=None
    ):
        if window_coords is None:
            window_coords = (0, 0, screen_width, screen_height)
        start_x, start_y, end_x, end_y = window_coords
        # end_x = end_x - 1
        # end_y = end_y - 1
        bounding_corners = bounding_corners.clamp(
            min=torch.tensor((start_x, start_y), device=bounding_corners.device),
            max=torch.tensor((end_x, end_y), device=bounding_corners.device),
        )
        # bounding_box_sizes = (bounding_corners[..., 1, :] - bounding_corners[..., 0, :])
        bounding_box_sizes = self.get_tensor(
            bounding_corners[..., 1, :].shape, dtype=bounding_corners.dtype
        )
        torch.subtract(
            bounding_corners[..., 1, :],
            bounding_corners[..., 0, :],
            out=bounding_box_sizes,
        )
        # bbss = bounding_box_sizes.prod(-1, keepdim=True)
        bbss = self.get_tensor(
            [*bounding_box_sizes.shape[:-1], 1], dtype=bounding_box_sizes.dtype
        )
        torch.prod(bounding_box_sizes, -1, keepdim=True, out=bbss)
        # num_fragments_per_object = bbss.amax(0)
        num_fragments_per_object = self.get_tensor(bbss.shape[1:], dtype=bbss.dtype)
        torch.amax(bbss, 0, out=num_fragments_per_object)
        # num_fragments_per_frame = num_fragments_per_object.sum()
        num_fragments_per_frame = torch.sum(num_fragments_per_object)
        num_fragments = num_fragments_per_frame * bbss.shape[0]

        def get_bounding_box_fragment_coords(x):
            # arange_num_segments_per_oject = torch.arange(len(self.num_segments_per_object), device=x.device)
            arange_num_segments_per_oject = self.get_tensor(
                [len(self.num_segments_per_object)], dtype=torch.long
            )
            torch.arange(
                len(self.num_segments_per_object),
                device=arange_num_segments_per_oject.device,
                out=arange_num_segments_per_oject,
            )
            segment_to_object_scatter_inds = torch.repeat_interleave(
                arange_num_segments_per_oject, self.num_segments_per_object, -1
            ).view(1, -1, 1)
            self.segment_to_object_scatter_inds = segment_to_object_scatter_inds

            arange_num_segments_per_oject = arange_num_segments_per_oject.view(1, -1, 1)
            object_bounding_corners_bottom_left = (
                broadcast_scatter(
                    arange_num_segments_per_oject,
                    -2,
                    segment_to_object_scatter_inds,
                    x[..., 0, :],
                    reduce="amin",
                    include_self=False,
                )
            ).clamp_(
                min=torch.tensor((start_x, start_y), device=x.device),
                max=torch.tensor((end_x, end_y), device=x.device),
            )
            #log_var(
            #    "object_bounding_corners_bottom_left",
            #    object_bounding_corners_bottom_left,
            #)
            object_bounding_corners_top_right = (
                broadcast_scatter(
                    arange_num_segments_per_oject,
                    -2,
                    segment_to_object_scatter_inds,
                    x[..., 1, :],
                    reduce="amax",
                    include_self=False,
                )
            ).clamp_(
                min=torch.tensor((start_x, start_y), device=x.device),
                max=torch.tensor((end_x, end_y), device=x.device),
            )
            #log_var(
            #    "object_bounding_corners_top_right", object_bounding_corners_top_right
            #)

            # object_bounding_box_dimensions = object_bounding_corners_top_right - object_bounding_corners_bottom_left
            object_bounding_box_dimensions = self.get_tensor(
                object_bounding_corners_top_right.shape,
                dtype=object_bounding_corners_top_right.dtype,
            )
            torch.subtract(
                object_bounding_corners_top_right,
                object_bounding_corners_bottom_left,
                out=object_bounding_box_dimensions,
            )
            # object_bounding_box_num_pixels = object_bounding_box_dimensions.prod(-1, keepdim=True)
            object_bounding_box_num_pixels = self.get_tensor(
                [*object_bounding_box_dimensions.shape[:-1], 1], dtype=torch.long
            )
            torch.prod(
                object_bounding_box_dimensions,
                -1,
                keepdim=True,
                out=object_bounding_box_num_pixels,
            )

            #log_var("object_bounding_box_dimensions", object_bounding_box_dimensions)
            #log_var("object_bounding_box_num_pixels", object_bounding_box_num_pixels)
            # num_fragments = object_bounding_box_num_pixels.long().sum()
            num_fragments = torch.sum(object_bounding_box_num_pixels).item()
            self.num_fragments_fill = num_fragments / len(
                object_bounding_box_num_pixels
            )
            if self.first_projection:
                return None

            arange_numel = self.get_tensor(
                [object_bounding_box_num_pixels.numel()], dtype=torch.long
            )
            torch.arange(
                object_bounding_box_num_pixels.numel(),
                device=arange_numel.device,
                out=arange_numel,
            )
            object_to_fragment_gather_inds = torch.repeat_interleave(
                arange_numel,
                object_bounding_box_num_pixels.view(-1),
                -1,
                output_size=num_fragments,
            ).unsqueeze(-1)

            # object_fragment_inds = torch.arange(num_fragments, device=x.device).view(-1, 1)
            object_fragment_inds = self.get_tensor([num_fragments], dtype=torch.long)
            torch.arange(
                num_fragments,
                device=object_fragment_inds.device,
                out=object_fragment_inds,
            )
            object_fragment_inds = object_fragment_inds.view(-1, 1)

            # object_offsets = (object_bounding_box_num_pixels.view(-1).cumsum(-1) - object_bounding_box_num_pixels.view(-1)).view(-1, 1)
            flattened_pixels = object_bounding_box_num_pixels.view(-1)
            object_offsets_flat = self.get_tensor(
                flattened_pixels.shape, dtype=flattened_pixels.dtype
            )
            cumsum_pixels = self.get_tensor(
                flattened_pixels.shape, dtype=flattened_pixels.dtype
            )
            torch.cumsum(flattened_pixels, -1, out=cumsum_pixels)
            torch.subtract(cumsum_pixels, flattened_pixels, out=object_offsets_flat)
            # Free cumsum_pixels as it's no longer needed (most recently allocated)
            cumsum_size = cumsum_pixels.numel() * cumsum_pixels.element_size()
            self.memory.current_pointer -= cumsum_size
            object_offsets = object_offsets_flat.view(-1, 1)

            temp_gathered = broadcast_gather(
                object_offsets,
                -2,
                object_to_fragment_gather_inds,
                keepdim=True,
                out=self.memory,
            )
            torch.subtract(
                object_fragment_inds, temp_gathered, out=object_fragment_inds
            )

            # object_bounding_box_dimensions_for_frags = broadcast_gather(squish(object_bounding_box_dimensions, 0, 1), -2, object_to_fragment_gather_inds, keepdim=True)
            squished_dims = squish(object_bounding_box_dimensions, 0, 1)
            object_bounding_box_dimensions_for_frags = broadcast_gather(
                squished_dims,
                -2,
                object_to_fragment_gather_inds,
                keepdim=True,
                out=self.memory,
            )

            squished_corners = squish(object_bounding_corners_bottom_left, 0, 1)
            object_bounding_corners_bottom_left_for_frags = broadcast_gather(
                squished_corners,
                -2,
                object_to_fragment_gather_inds,
                keepdim=True,
                out=self.memory,
            )
            # object_fragment_x = (object_fragment_inds % object_bounding_box_dimensions_for_frags[..., :1]) + object_bounding_corners_bottom_left_for_frags[..., :1]
            object_fragment_x = self.get_tensor(
                object_fragment_inds.shape, dtype=object_fragment_inds.dtype
            )
            temp_remainder = self.get_tensor(
                object_fragment_inds.shape, dtype=object_fragment_inds.dtype
            )
            torch.remainder(
                object_fragment_inds,
                object_bounding_box_dimensions_for_frags[..., :1],
                out=temp_remainder,
            )
            torch.add(
                temp_remainder,
                object_bounding_corners_bottom_left_for_frags[..., :1],
                out=object_fragment_x,
            )
            # Free temp_remainder as it's no longer needed
            temp_size = temp_remainder.numel() * temp_remainder.element_size()
            self.memory.current_pointer -= temp_size
            # object_fragment_y_bbox = (object_fragment_inds // object_bounding_box_dimensions_for_frags[..., :1])
            object_fragment_y_bbox = self.get_tensor(
                object_fragment_inds.shape, dtype=object_fragment_inds.dtype
            )
            torch.floor_divide(
                object_fragment_inds,
                object_bounding_box_dimensions_for_frags[..., :1],
                out=object_fragment_y_bbox,
            )
            # object_fragment_y = object_fragment_y_bbox + object_bounding_corners_bottom_left_for_frags[..., 1:]
            object_fragment_y = self.get_tensor(
                object_fragment_y_bbox.shape, dtype=object_fragment_y_bbox.dtype
            )
            torch.add(
                object_fragment_y_bbox,
                object_bounding_corners_bottom_left_for_frags[..., 1:],
                out=object_fragment_y,
            )

            return (
                object_fragment_x,
                object_fragment_y,
                object_fragment_y_bbox,
                object_fragment_inds,
                object_bounding_box_dimensions,
                object_bounding_corners_bottom_left,
                object_to_fragment_gather_inds,
            )

        return (
            bounding_corners,
            bounding_box_sizes,
            bbss,
            num_fragments_per_object,
            num_fragments_per_frame,
            num_fragments,
            get_bounding_box_fragment_coords(bounding_corners),
        )

    def project_to_screen(self, camera, light_sources):
        self.distance_to_control_points = (self.corners - camera.ray_origin).norm(p=2, dim=-1, keepdim=True)
        super().project_to_screen(camera, light_sources)
        control_points = self.corners

        # control_net_lengths = (control_points[..., 1:, :] - control_points[..., :-1, :]).norm(p=2, dim=-1).sum(-1)
        control_point_diffs = self.get_tensor(
            control_points[..., 1:, :].shape, dtype=control_points.dtype
        )
        torch.subtract(
            control_points[..., 1:, :],
            control_points[..., :-1, :],
            out=control_point_diffs,
        )
        control_point_norms = self.get_tensor(
            [*control_point_diffs.shape[:-1]], dtype=control_points.dtype
        )
        torch.norm(control_point_diffs, p=2, dim=-1, out=control_point_norms)
        control_net_lengths = self.get_tensor(
            [*control_point_norms.shape[:-1]], dtype=control_points.dtype
        )
        torch.sum(control_point_norms, -1, out=control_net_lengths)
        self.num_samples_per_segment = (control_net_lengths.amax(0) / self.num_pixels_per_sample).ceil().long().clamp_min_(1)
        maximum_net_length = control_net_lengths.amax()
        #self.num_sampled_points = self.num_samples_per_segment.amax().item()
        return self

    def get_batch_identifier(self):
        return f"{__class__}_{self.num_texture_points}_{self.filled}"

    def get_memory_used_per_timestep(self):
        return self.num_fragments_fill * (128 + 64)

    #@compiled
    def render_(
        self,
        time_start,
        time_end,
        object_start,
        object_end,
        ray_origin,
        screen_point,
        screen_basis,
        background_color=BLACK,
        anti_alias=False,
        anti_alias_offset=[0.5, 0.5],
        anti_alias_level=1,
        light_sources=[],
        screen_width=2000,
        screen_height=2000,
        window_coords=None,
        memory=None,
        primitive_type=None,
    ):
        def select_time(x, texture=False):
            x = x if len(x) == 1 else x[time_start:time_end]
            x = (
                x
                if x.shape[1] == 1
                else x[:, int(x.shape[1] * object_start) : int(x.shape[1] * object_end)]
            )
            return x

        corners = select_time(self.corners)
        distance_to_control_points = select_time(self.distance_to_control_points)
        corners_int = select_time(self.corners_int)
        projected_distances = select_time(self.projected_distances)
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
        next_segment_inds = select_time(self.next_segment_inds)
        num_segments_per_object = self.num_segments_per_object

        num_objects = len(num_segments_per_object)  #

        if window_coords is None:
            window_coords = 0, 0, screen_width, screen_height
        window_height = window_coords[-1] - window_coords[1]
        window_width = window_coords[-2] - window_coords[0]
        start_x, start_y, end_x, end_y = window_coords

        bounding_corners = select_time(self.bounding_corners)

        initial_pointer = self.memory.current_pointer
        initial_persist_pointer = self.memory.current_reverse_pointer
        (
            fragment_x,
            fragment_y,
            fragment_y_bbox,
            fragment_inds,
            object_bounding_box_dimensions,
            object_bounding_corners_bottom_left,
            object_to_fragment_gather_inds,
        ) = self.get_windowed_bounding_boxes(
            bounding_corners, screen_width, screen_height, window_coords
        )[-1]

        if fragment_x.numel() == 0:
            return None

        memory = self.memory
        fragment_x = memory.clone(fragment_x, persist=True)
        fragment_y = memory.clone(fragment_y, persist=True)
        control_points = corners
        control_points = torch.repeat_interleave(control_points, self.num_samples_per_segment, 1)
        # t = torch.linspace(0, 1, self.num_sampled_points, device=control_points.device)
        polygon_vertices = self.get_tensor([*control_points.shape[:-2], control_points.shape[-1]])
        self.memory.save_pointer()
        t = batch_arange(self.num_samples_per_segment) / torch.repeat_interleave(self.num_samples_per_segment, self.num_samples_per_segment)

        polygon_vertices = evaluate_cubic_bezier(
            control_points, t.unsqueeze(-1), polygon_vertices, self.memory
        )

        # polygon_vertices = evaluate_cubic_bezier_old3(control_points, t.unsqueeze(-1))
        # assert polygon_vertices.shape == [T, N, P, 2] (time (frames), num segments, num control points per segment, 2D)
        # polygon_vertices = squish(polygon_vertices, -3, -2)  # shape [T, N, S*P, 2]
        next_polygon_vertices = polygon_vertices.roll(shifts=-1, dims=-2)

        # Change the last next_vertice from the start of this segment to the start of the next segment.
        # next_segments = broadcast_gather(polygon_vertices, -3, next_segment_inds, keepdim=True)
        segment_start_inds = self.num_samples_per_segment.cumsum(-1) - self.num_samples_per_segment
        segment_end_inds = segment_start_inds - 1
        segment_end_inds[0] = polygon_vertices.shape[-2]-1
        segment_end_inds = torch.roll(segment_end_inds, -1, -1)
        next_segment_inds = segment_start_inds[next_segment_inds].squeeze(-1)
        next_polygon_vertices[:,segment_end_inds] = broadcast_gather(polygon_vertices, -2, next_segment_inds, keepdim=True, out=self.memory)

        next_polygon_perpendiculars = next_polygon_vertices.roll(shifts=-1, dims=-2)
        next_polygon_perpendiculars[:,segment_end_inds] = broadcast_gather(next_polygon_vertices, -2, next_segment_inds, keepdim=True, out=self.memory)

        next_polygon_perpendiculars = next_polygon_perpendiculars - next_polygon_vertices
        next_polygon_perpendiculars = torch.stack(
            (-next_polygon_perpendiculars[..., 1], next_polygon_perpendiculars[..., 0]),
            dim=-1,
        )

        self.memory.reset_pointer()

        # line_segments = next_polygon_vertices - polygon_vertices
        line_segments = self.get_tensor(
            polygon_vertices.shape, dtype=polygon_vertices.dtype
        )
        self.memory.save_pointer()
        torch.subtract(next_polygon_vertices, polygon_vertices, out=line_segments)
        # line_segment_lengths = line_segments.norm(p=2, dim=-1)
        line_segment_lengths = self.get_tensor(
            [*line_segments.shape[:-1]], dtype=line_segments.dtype
        )
        torch.norm(line_segments, p=2, dim=-1, out=line_segment_lengths)

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
            if line_segment_lengths.amax() < 0.1:
                return None
        else:
            if (border_width.amax() < 0.1) | (line_segment_lengths.amax() < 0.1):
                return None

        self.memory.reset_pointer()

        num_samples_per_object = torch.stack([sum(_) for _ in self.num_samples_per_segment.split([__ for __ in num_segments_per_object])])
        local_window_x, local_window_y, local_window_fragment_to_object_inds = rasterize_polygon(polygon_vertices, next_polygon_vertices, num_samples_per_object, self.memory)

        def get_local_to_global_inds(local_window_x, local_window_y, local_window_fragment_to_object_inds):
            object_bounding_box_dimensions_for_segments = (
                broadcast_gather(
                    object_bounding_box_dimensions,
                    -2,
                    local_window_fragment_to_object_inds,
                    keepdim=True,
                )
            )#.unsqueeze(-1)

            object_bounding_corners_bottom_left_for_segments = (
                broadcast_gather(
                    object_bounding_corners_bottom_left,
                    -2,
                    local_window_fragment_to_object_inds,
                    keepdim=True,
                )
            )#.unsqueeze(-1)

            bbox_x = self.get_tensor(local_window_x.shape, dtype=torch.long)
            torch.subtract(
                local_window_x,
                object_bounding_corners_bottom_left_for_segments[..., :1],
                out=bbox_x,
            )
            bbox_y = self.get_tensor(local_window_y.shape, dtype=torch.long)
            torch.subtract(
                local_window_y,
                object_bounding_corners_bottom_left_for_segments[..., 1:],
                out=bbox_y,
            )
            # bbox_num_pixels = object_bounding_box_dimensions_for_segments.prod(-2, keepdim=True)
            bbox_num_pixels = self.get_tensor(
                [
                    *object_bounding_box_dimensions_for_segments.shape[:-1],
                    1,
                ],
                dtype=object_bounding_box_dimensions_for_segments.dtype,
            )
            torch.prod(
                object_bounding_box_dimensions_for_segments,
                -1,
                keepdim=True,
                out=bbox_num_pixels,
            )
            # local_to_bbox_inds = (bbox_x.clamp_min(0) + bbox_y * object_bounding_box_dimensions_for_segments[...,:1,:]
            #                        ).clamp_(min=torch.zeros_like(bbox_num_pixels),
            #                                 max=bbox_num_pixels - 1)
            local_to_bbox_inds = self.get_tensor(bbox_x.shape, dtype=torch.long)
            torch.clamp_min(bbox_x, 0, out=local_to_bbox_inds)
            torch.addcmul(
                local_to_bbox_inds,
                object_bounding_box_dimensions_for_segments[..., :1],
                bbox_y,
                value=1,
                out=local_to_bbox_inds,
            )
            local_to_bbox_inds.clamp_min_(0)
            local_to_bbox_inds.clamp_max_(bbox_num_pixels - 1)

            # local_to_bbox_inds scatters from local_window into object level bounding box.
            # Now we need to add offsets so that inds from different objects end up in different output frames.
            # offsets = object_bounding_box_dimensions.prod(-1, keepdims=True).view(-1,1)
            # offsets = offsets.cumsum(-2)  - offsets
            offsets = self.get_tensor(
                [*object_bounding_box_dimensions.shape[:-1], 1], dtype=torch.long
            )
            torch.prod(object_bounding_box_dimensions, -1, keepdim=True, out=offsets)
            offsets = offsets.view(-1, 1)
            temp_offsets = self.get_tensor(offsets.shape, dtype=torch.long)
            temp_offsets.copy_(offsets)
            torch.cumsum(offsets, -2, out=offsets)
            offsets -= temp_offsets

            # offsets_for_segments = squish(torch.repeat_interleave(unsquish(offsets, 0, -corners.shape[0]), num_segments_per_object, -2).unsqueeze(-1), 0, 1)
            offsets_for_segments = squish(
                broadcast_gather(
                    unsquish(offsets, 0, -corners.shape[0]),
                    -2,
                    local_window_fragment_to_object_inds,
                    keepdim=True,
                ),
                0,
                1,
            )

            local_to_global_inds = squish(local_to_bbox_inds, 0, 1)
            local_to_global_inds += offsets_for_segments
            local_to_global_inds = local_to_global_inds.view(-1)

            local_to_global_inds.clamp_(min=0, max=fragment_x.shape[-2] - 1)
            return local_to_global_inds, bbox_x, bbox_y, object_bounding_box_dimensions_for_segments, offsets

        (local_to_global_inds, bbox_x,
            bbox_y, object_bounding_box_dimensions_for_segments, offsets
         ) = get_local_to_global_inds(local_window_x, local_window_y, local_window_fragment_to_object_inds)

        # invalid_mask = ((bbox_x < 0) | (bbox_x > bounding_box_widths.unsqueeze(-2))) | (((bbox_y < 0) | (bbox_y > bounding_box_heights.unsqueeze(-2))))
        # invalid_mask = ((bbox_x >= object_bounding_box_dimensions_for_segments[...,:1,:]) |
        #                (bbox_y < 0) | (bbox_y > object_bounding_box_dimensions_for_segments[...,1:,:]))
        invalid_mask = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        pointer = self.memory.current_pointer
        temp_bool = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        torch.greater_equal(
            bbox_x,
            object_bounding_box_dimensions_for_segments[..., :1],
            out=invalid_mask,
        )
        torch.lt(bbox_y, 0, out=temp_bool)
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        torch.gt(
            bbox_y,
            object_bounding_box_dimensions_for_segments[..., 1:],
            out=temp_bool,
        )
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        self.memory.current_pointer = pointer

        # Note we need to keep negative x inds around for now, because we cumsum across rows from the left
        # to count intersections, we will cull negative x inds later.
        zero = self.get_tensor([1])
        zero[:] = 0
        local_intersection_counts = torch.ones(local_window_y.shape, dtype=torch.float, device=local_window_y.device)
        local_intersection_counts = torch.where(
            invalid_mask, zero, local_intersection_counts, out=local_intersection_counts
        )

        # global_intersection_counts = torch_scatter.scatter_sum(local_intersection_counts.view(-1), local_to_global_inds, -1, dim_size=fragment_x.shape[-2])
        # out = torch.zeros((fragment_x.shape[-2],), device=fragment_x.device)
        out = self.get_tensor([fragment_x.shape[-2]])
        out[:] = 0

        global_intersection_counts = torch.scatter_add(
            out, -1, local_to_global_inds, local_intersection_counts.view(-1), out=out
        )

        # self.memory.current_pointer = local_intersection_counts_pointer

        # Now do border mask.
        # local_window_xy = torch.stack((local_window_x, local_window_y), -1)

        global_dists = self.get_tensor([fragment_x.shape[-2]], dtype=torch.float)
        global_dists[:] = 1e12
        local_dist_pointer = self.memory.current_pointer

        border_width_o = broadcast_gather(
            border_width, -2, self.segment_to_object_scatter_inds
        )
        border_width_o = torch.repeat_interleave(border_width_o, self.num_samples_per_segment, -2)
        (local_window_x, local_window_y, local_window_fragment_to_sample_inds, local_dist
        ) = rasterize_polygon_border(
            polygon_vertices,
            next_polygon_vertices,
            next_polygon_perpendiculars,
            border_width_o,
            self.memory
        )

        sample_to_object_ind = torch.repeat_interleave(torch.arange(num_samples_per_object.shape[0], device=num_samples_per_object.device),
                                                                       num_samples_per_object)
        #local_window_fragment_to_sample_inds //= self.num_sampled_points
        local_window_fragment_to_object_inds = broadcast_gather(sample_to_object_ind.view(1,-1,1), -2,
                                                                local_window_fragment_to_sample_inds, keepdim=True)

        #local_window_fragment_to_object_inds =  broadcast_gather(self.segment_to_object_scatter_inds, -2, local_window_fragment_to_segment_inds, keepdim=True)

        (local_to_global_inds, bbox_x,
         bbox_y, object_bounding_box_dimensions_for_segments, _
         ) = get_local_to_global_inds(local_window_x, local_window_y, local_window_fragment_to_object_inds)

        invalid_mask = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        pointer = self.memory.current_pointer
        temp_bool = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        torch.greater_equal(
            bbox_x,
            object_bounding_box_dimensions_for_segments[..., :1],
            out=invalid_mask,
        )
        torch.lt(bbox_y, 0, out=temp_bool)
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        torch.gt(
            bbox_y,
            object_bounding_box_dimensions_for_segments[..., 1:],
            out=temp_bool,
        )
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        torch.lt(bbox_x, 0, out=temp_bool)
        dist_invalid_mask = torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        self.memory.current_pointer = pointer

        posinf = zero
        posinf[:] = 1e12
        local_dist = torch.where(dist_invalid_mask, posinf, local_dist, out=local_dist)

                # Handle portion_of_curve_drawn
        # self.expand_verts_to_frags(self.portion_of_curve_drawn)

        """num_vertices_per_object = num_samples_per_object.view(-1, 1)
        num_vertices_per_object += 1
        threshold_for_drawing = broadcast_gather(
            portion_of_curve_drawn * num_vertices_per_object,
            -2,
            self.segment_to_object_scatter_inds,
            keepdim=True,
        )
        threshold_for_drawing = threshold_for_drawing.unsqueeze(
            -2
        )  # .expand(-1,-1,self.num_sampled_points, -1)

        vertex_number = torch.arange(
            threshold_for_drawing.shape[-3] * self.num_sampled_points,
            device=threshold_for_drawing.device,
        )
        vertex_offsets = num_segments_per_object * self.num_sampled_points
        vertex_offsets = vertex_offsets.cumsum(-1) - vertex_offsets
        vertex_offsets = vertex_offsets.gather(
            -1, self.segment_to_object_scatter_inds.view(-1)
        ).view(-1, 1)
        vertex_offsets = (
            vertex_offsets.expand(-1, self.num_sampled_points).clone().view(-1)
        )
        # vertex_offsets = torch.repeat_interleave(
        #    vertex_offsets, num_segments_per_object * self.num_sampled_points, -1
        # )
        vertex_number -= vertex_offsets
        vertex_number = unsquish(vertex_number, 0, self.num_sampled_points).unsqueeze(
            -1
        )
        posinf = zero
        posinf[:] = 1e12
        #local_dist = torch.where(
        #    vertex_number >= threshold_for_drawing, posinf, local_dist, out=local_dist
        #)

        # global_dists = torch.empty((fragment_x.shape[-2],), device=control_points.device)

        """
        self.memory.current_pointer = pointer

        global_dists = torch.scatter_reduce(
            global_dists,
            -1,
            local_to_global_inds,
            local_dist.view(-1),
            reduce="amin",
            out=global_dists,
        )

        self.memory.current_pointer = local_dist_pointer

        # border_mask = (global_dists.unsqueeze(-1) < self.expand_verts_to_frags(squish(border_width, 0, 1), object_to_fragment_gather_inds)).float()

        border_mask = self.expand_verts_to_frags(
            squish(border_width, 0, 1), object_to_fragment_gather_inds
        )
        #global_dists -= 1e-3

        # Count the number of intersections in the horizontal ray to this pixel's left.
        left_intersection_counts = torch.cumsum(
            global_intersection_counts, -1, out=global_intersection_counts
        )

        pointer = self.memory.current_pointer
        row_start_counts = self.get_tensor(left_intersection_counts.shape)
        row_start_ind_local = self.expand_verts_to_frags(
            object_bounding_box_dimensions[..., :1].view(-1, 1),
            object_to_fragment_gather_inds,
        )
        row_start_ind_local *= fragment_y_bbox
        row_start_offset = self.expand_verts_to_frags(
            offsets.view(-1, 1), object_to_fragment_gather_inds
        )
        row_start_ind = torch.add(
            row_start_offset, row_start_ind_local, out=row_start_ind_local
        ).view(-1)
        # row_start_ind = (row_start_ind-1).clamp_min(0)
        row_start_ind -= 1
        row_start_ind.clamp_min_(0)

        broadcast_gather(
            left_intersection_counts,
            -1,
            row_start_ind,
            keepdim=True,
            out=row_start_counts,
        )
        left_intersection_counts -= row_start_counts
        self.memory.current_pointer = pointer

        # interior_mask = ((left_intersection_counts % 2) == 1).float().unsqueeze(-1)
        left_intersection_counts %= 2
        interior_mask = torch.eq(
            left_intersection_counts, 1, out=left_intersection_counts
        ).unsqueeze(-1)

        global_dists = global_dists.unsqueeze(-1)
        if self.filled:
            with memory.temp():
                outline = torch.lt(global_dists, 0.6 + 1e-3, out=memory.get_tensor(global_dists.shape))#, torch.bool))
                interior_mask = torch.maximum(interior_mask, outline, out=interior_mask)
            self.memory.current_pointer = pointer
        border_mask = torch.lt(global_dists, border_mask.square_(), out=memory.get_tensor(border_mask.shape, torch.bool))
        #border_mask = torch.zeros_like(interior_mask)

        # fragment_coords = torch.cat((fragment_x, fragment_y), -1).float()

        # TODO subtract window_start from x and y (so they are 0 centered.
        # inds = (fragment_x - start_x) + (fragment_y - start_y) * window_width
        interior_mask = memory.clone(interior_mask, persist=True)
        border_mask = memory.clone(border_mask, persist=True)
        memory.current_pointer = initial_pointer
        inds = torch.multiply(fragment_y, window_width, out=self.get_tensor(fragment_x.shape, dtype=torch.int))
        inds -= start_y * window_width + start_x
        inds += fragment_x

        window_size = window_width * window_height

        if self.filled:
            pass  # border_mask *= interior_mask
        else:
            interior_mask[:] = 0
        # TODO does this need to clip based on x and y instead of inds for window?
        #m = (inds < window_size) & ((interior_mask > 0) | (border_mask > 0))
        #m = m.reshape(-1)
        m1 = torch.gt(interior_mask, 0, out=memory.get_tensor(interior_mask.shape, torch.bool))
        m2 = memory.clone(border_mask)#torch.gt(border_mask, 0, out=memory.get_tensor(interior_mask.shape, torch.bool))
        m = torch.logical_or(m1, m2, out=memory.get_tensor(m1.shape, m1.dtype, persist=True)).view(-1)
        num_masked_frags = m.sum()
        border_mask = torch.masked_select(border_mask.view(-1), m, out=memory.get_tensor((num_masked_frags,), border_mask.dtype, persist=True)).unsqueeze(-1)

        # g_offsets = torch.arange(0, corners.shape[0], device=inds.device) * window_size
        g_offsets = self.get_tensor([corners.shape[0]], dtype=torch.long)
        torch.arange(0, corners.shape[0], device=g_offsets.device, out=g_offsets)
        g_offsets *= window_size
        #object_to_fragment_gather_inds = memory.clone(object_to_fragment_gather_inds, persist=True)
        frame_to_fragment_gather_inds = torch.floor_divide(object_to_fragment_gather_inds, num_objects, out=memory.get_tensor(object_to_fragment_gather_inds.shape, object_to_fragment_gather_inds.dtype, persist=True))
        g_offsets = self.expand_verts_to_frags(
            g_offsets.unsqueeze(-1), frame_to_fragment_gather_inds
        )
        inds = memory.cast(inds, torch.long)
        inds += g_offsets
        inds = torch.masked_select(inds.view(-1), m, out=memory.get_tensor((num_masked_frags,), inds.dtype, persist=True))
        memory.current_pointer = initial_pointer
        inds = memory.clone(inds)
        initial_pointer = memory.current_pointer
        # unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        mob_center_for_frags = self.expand_verts_to_frags(
            squish(mob_center, 0, 1), object_to_fragment_gather_inds
        )
        normals_for_frags = self.expand_verts_to_frags(
            squish(normals, 0, 1), object_to_fragment_gather_inds
        )

        def expo(x, select=True, gather_inds=object_to_fragment_gather_inds, persist=False):
            if select:
                x = select_time(x)
            x = x.view(-1, x.shape[-1])
            return self.expand_verts_to_frags(x, gather_inds, persist=persist)

        '''screen_basis = unsquish(
            expo(
                squish(screen_basis, -2, -1),
                False,
                gather_inds=frame_to_fragment_gather_inds,
            ),
            -1,
            3,
        )'''
        screen_point = expo(
            screen_point, False, gather_inds=frame_to_fragment_gather_inds
        )
        ray_origin = expo(ray_origin, False, gather_inds=frame_to_fragment_gather_inds, persist=True)
        # screen_basis = screen_basis / screen_basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        screen_basis_norm = torch.norm(screen_basis[...,:2,:], p=2, dim=-1, keepdim=True, out=self.get_tensor([*screen_basis.shape[:-2], 2, 1], persist=True))
        #screen_basis_norm_sq = self.get_tensor(
        #    screen_basis_norm.shape, dtype=screen_basis.dtype
        #)
        screen_basis_norm_sq = torch.square(screen_basis_norm, out=screen_basis_norm)
        screen_basis_norm_sq.clamp_min_(1e-6)
        screen_basis = torch.div(screen_basis[...,:2,:], screen_basis_norm_sq, out=memory.get_tensor([*screen_basis.shape[:-2], 2, 3], persist=True))
        screen_basis = unsquish(expo(squish(screen_basis, -2, -1), False, gather_inds=frame_to_fragment_gather_inds, persist=True), -1, 3)

        fragment_x = torch.sub(fragment_x, screen_width * 0.5, out=memory.get_tensor(fragment_x.shape))
        #fragment_x -= screen_width * 0.5
        #fragment_x /=  screen_height * 0.5

        #fragment_y -= screen_height * 0.5
        fragment_y = torch.sub(fragment_y, screen_height * 0.5, out=memory.get_tensor(fragment_y.shape))
        s = 2/screen_height
        #fragment_y /= screen_height * 0.5

        ray_direction = torch.addcmul(screen_point, fragment_x, screen_basis[...,0,:], value=s, out=screen_point)
        ray_direction = torch.addcmul(ray_direction, fragment_y, screen_basis[...,1,:], value=s, out=ray_direction)
        ray_direction -= ray_origin
        ray_direction = F.normalize(ray_direction, p=2, dim=-1, out=ray_direction)

        dists = memory.get_tensor([*ray_origin.shape[:-1], 1])
        with memory.temp():
            plane_dir = torch.sub(mob_center_for_frags, ray_origin, out=memory.get_tensor(ray_origin.shape))
            dot1 = dot_product(plane_dir, normals_for_frags, out=memory.get_tensor([*ray_origin.shape[:-1], 1]))
            dot2 = dot_product(ray_direction, normals_for_frags, out=dists)
            #if not self.filled:
            #    mdeg = dot2.abs() < 0.075
            dists = torch.divide(dot1, dot2, out=dot2)
            #if not self.filled:
            #dists = torch.where(mdeg, torch.tensor((-1,), device=dists.device), dists, out=dists)
        #m_parallel = squish((dists < min_dists_for_frag) | (dists > max_dists_for_frag))
        #colors = torch.where(m_parallel, torch.tensor((0,), device=dists.device), colors)
        #dists = torch.where(m_parallel, torch.tensor((-1,), device=dists.device), dists)
        dists.nan_to_num_()
        texture_start_pointer = memory.current_pointer
        if self.num_texture_points > 1:
            proj_onto_mobs = torch.addcmul(ray_origin, dists, ray_direction, out=ray_direction)
            memory.current_reverse_pointer = initial_persist_pointer
            with memory.temp():
                mob_centers = expo(self.mob_center)
                disps_from_mobs = torch.subtract(proj_onto_mobs, mob_centers, out=proj_onto_mobs)
            arange_size = self.mob_center.shape[1] * (time_end - time_start)
            arange_offsets = self.get_tensor([arange_size], dtype=torch.long)
            torch.arange(arange_size, device=arange_offsets.device, out=arange_offsets)
            arange_offsets *= self.colors.shape[-2]
            offsets = expo(arange_offsets.view(1, -1, 1, 1))

            def get_c(b):
                dot_out = self.get_tensor([*b.shape[:-1], 1], persist=True)
                # c = dot_product(F.normalize(b, p=2, dim=-1), disps_from_mobs) / b.norm(p=2, dim=-1, keepdim=True)
                # b_normalized = F.normalize(b, p=2, dim=-1)
                dot_result = dot_product(b, disps_from_mobs, out=dot_out)
                b_norm = self.get_tensor([*b.shape[:-1], 1], dtype=b.dtype)
                torch.norm(b, p=2, dim=-1, keepdim=True, out=b_norm)
                b_norm.square_()
                c = torch.div(dot_result, b_norm, out=dot_result)
                c *= 0.5
                c += 0.5
                return c

            # x = (get_c(mob_basis2) * (grid_height)).clamp_max_(grid_height-1).clamp_min_(0)
            grid_height = expo(self.grid_height)
            with memory.temp():
                mob_basis2 = expo(self.basis2)
                x = get_c(mob_basis2)
            x *= grid_height
            x.clamp_max_(grid_height - 1).clamp_min_(0)

            # y = (get_c(mob_basis1) * grid_width).clamp_max_(grid_width-1).clamp_min_(0)
            with memory.temp():
                grid_width = expo(self.grid_width)
                mob_basis1 = expo(self.basis1)
                y = get_c(mob_basis1)
            y *= grid_width
            y.clamp_max_(grid_width - 1).clamp_min_(0)

            # xr = x % 1
            xr = self.get_tensor(x.shape, dtype=x.dtype)
            torch.remainder(x, 1, out=xr)
            # yr = y % 1
            yr = self.get_tensor(y.shape, dtype=y.dtype)
            torch.remainder(y, 1, out=yr)
            # w1 = (1-xr) * (1-yr)
            w4 = torch.mul(xr, yr, out=self.get_tensor(xr.shape))
            minus_xr = torch.sub(1, xr, out=self.get_tensor(xr.shape))
            w3 = torch.mul(minus_xr, yr, out=self.get_tensor(xr.shape))
            #with memory.temp():
            minus_yr = torch.sub(1, yr, out=yr)
            w2 = torch.mul(xr, minus_yr, out=xr)
            w1 = torch.mul(minus_yr, minus_xr, out=minus_yr)

            # x_floor = (x).floor().long()
            x_floor = self.get_tensor(x.shape, dtype=torch.int)
            x_ceil = self.get_tensor(x.shape, dtype=torch.int)
            y_floor = self.get_tensor(y.shape, dtype=torch.int)
            y_ceil = self.get_tensor(y.shape, dtype=torch.int)
            with memory.temp():
                temp = self.get_tensor(y.shape)
                x_floor_float = torch.floor(x, out=temp)
                x_floor[:] = x_floor_float
                x_ceil_float = torch.ceil(x, out=temp)
                x_ceil[:] = x_ceil_float
                y_floor_float = torch.floor(y, out=temp)
                y_floor[:] = y_floor_float
                y_ceil_float = torch.ceil(y, out=temp)
                y_ceil[:] = y_ceil_float

            colos = squish(colors, 0, 2)
            interpolated_colors = 0
            temp_long = self.get_tensor(w1.shape, torch.int)
            sum_w = self.get_tensor(w1.shape, w1.dtype)
            sum_w[:] = 0
            gathered_colors = self.get_tensor([*w1.shape[:-1], 5], w1.dtype)
            memory.current_reverse_pointer = initial_persist_pointer
            interpolated_colors = self.get_tensor([*w1.shape[:-1], 5], w1.dtype, persist=True)
            interpolated_colors[:] = 0
            for w, x, y in [
                (w1, x_floor, y_floor),
                (w2, x_ceil, y_floor),
                (w3, x_floor, y_ceil),
                (w4, x_ceil, y_ceil),
            ]:
                x = torch.addcmul(x, y, grid_height, out=temp_long)
                # x = x + y * grid_height
                x += offsets
                sum_w = torch.add(w, sum_w, out=sum_w)
                with memory.temp():
                    c = broadcast_gather(colos, -2, memory.cast(x, torch.long), out=gathered_colors, keepdim=True)
                interpolated_colors = torch.addcmul(
                    interpolated_colors, c, w, out=interpolated_colors
                )
            interpolated_colors /= sum_w
            memory.current_pointer = texture_start_pointer
            interpolated_colors = memory.clone(interpolated_colors)
            memory.current_reverse_pointer = initial_persist_pointer
        else:
            memory.current_pointer = texture_start_pointer
            memory.current_reverse_pointer = initial_persist_pointer
            interpolated_colors = self.expand_verts_to_frags(
                colors.view(-1,colors.shape[-1]), object_to_fragment_gather_inds, -2
            )

        with memory.temp():
            dists_per_object = torch.split(distance_to_control_points, [_ for _ in self.num_segments_per_object], -3)
            dshape = [*distance_to_control_points.shape[:-3], len(dists_per_object), *distance_to_control_points.shape[-1:]]
            min_dists_per_object = torch.cat([squish(_, 1, 2).amin(-2, keepdim=True) for _ in dists_per_object], -2,
                                             out=self.memory.get_tensor(dshape))  # * 0.9
            max_dists_per_object = torch.cat([squish(_, 1, 2).amax(-2, keepdim=True) for _ in dists_per_object], -2,
                                             out=self.memory.get_tensor(dshape))  # * 0.9
            border_dists = broadcast_gather(squish(border_width) * 10 / screen_height, -2, object_to_fragment_gather_inds, out=memory)
            min_dists_for_frag = broadcast_gather(squish(min_dists_per_object), -2,
                                                  object_to_fragment_gather_inds, out=memory)
            min_dists_for_frag -= border_dists
            max_dists_for_frag = broadcast_gather(squish(max_dists_per_object), -2,
                                                  object_to_fragment_gather_inds, out=memory)
            max_dists_for_frag += border_dists
            dists = dists.view(-1,1).clamp_(min=min_dists_for_frag, max=max_dists_for_frag)

        # output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        # output_frags[:] = 0

        def get_frags(ws, fragment_coords=None):
            def get_colors():
                colors = interpolated_colors
                colors = colors.view(-1, colors.shape[-1])
                colors = torch.masked_select(colors, m.unsqueeze(-1), out=memory.get_tensor((num_masked_frags * colors.shape[-1],), persist=True)).view(-1,colors.shape[-1])
                #colors = colors[m]

                if self.filled:
                    border_colors_frags = self.expand_verts_to_frags(
                        squish(border_colors, 0, 1), object_to_fragment_gather_inds
                    )
                    border_colors_frags = border_colors_frags.view(
                        -1, border_colors_frags.shape[-1]
                    )
                    #border_colors_frags = border_colors_frags[m]
                    border_colors_frags = torch.masked_select(border_colors_frags, m.unsqueeze(-1),
                                        out=memory.get_tensor((num_masked_frags * colors.shape[-1],))).view(-1, colors.shape[-1])
                    colors = torch.where(memory.cast(border_mask, torch.bool), border_colors_frags, colors, out=colors)
                return colors

            colors = get_colors()
            dists2 = dists.view(-1)
            dists2 = torch.masked_select(dists2, m, out=memory.get_tensor((num_masked_frags,), persist=True))
            #dists2 = dists2[m]

            return colors, dists2

        colors, dists = get_frags(1)
        self.memory.current_pointer = initial_pointer
        return colors, dists, inds
