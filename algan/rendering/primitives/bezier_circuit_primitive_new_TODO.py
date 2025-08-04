import math
from functools import wraps

import torch
import torch.nn.functional as F

from algan.constants.color import BLUE, BLACK
from algan.logging.logger import LoggerManager
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


def evaluate_cubic_bezier_old3(p, t, i=-1):
    # out = ((1 - t) ** 3) * p[..., 0, :].unsqueeze(-2)
    # out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :].unsqueeze(-2)
    # out[:] += 3 * (1 - t) * t * t * p[..., 2, :].unsqueeze(-2)
    # out[:] += (t ** 3) * p[..., 3, :].unsqueeze(-2)
    out = ((1 - t) ** 3) * p[..., 0, :].unsqueeze(-2)
    if i == 0:
        return out
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :].unsqueeze(-2)
    if i == 1:
        return out
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :].unsqueeze(-2)
    if i == 2:
        return out
    out[:] += (t**3) * p[..., 3, :].unsqueeze(-2)
    return out


def evaluate_cubic_bezier_old2(p, t):
    out = ((1 - t) ** 3) * p[..., 0, :]
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :]
    out[:] += (t**3) * p[..., 3, :]
    return out


def evaluate_cubic_bezier_old(p, t, out, mem):
    out[:] = ((1 - t) ** 3) * p[..., 0, :]
    out[:] += 3 * ((1 - t) ** 2) * t * p[..., 1, :]
    out[:] += 3 * (1 - t) * t * t * p[..., 2, :]
    out[:] += (t**3) * p[..., 3, :]
    return out


def evaluate_cubic_bezier(p, t, out, mem, i=-1):
    p0 = p[..., 0, :].unsqueeze(-2)
    p1 = p[..., 1, :].unsqueeze(-2)
    p2 = p[..., 2, :].unsqueeze(-2)
    p3 = p[..., 3, :].unsqueeze(-2)
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


def evaluate_cubic_bezier_derivative_old(p, t):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    return (
        3 * ((1 - t) ** 2) * (p1 - p0)
        + 6 * (1 - t) * t * (p2 - p1)
        + 3 * (t * t) * (p3 - p2)
    )


def evaluate_cubic_bezier_derivative_with_end(p, t, end_portion=0.01):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    start_deriv = p[..., 4, :] - p0
    end_deriv = p[..., 5, :] - p3
    out = (
        3 * ((1 - t) ** 2) * (p1 - p0)
        + 6 * (1 - t) * t * (p2 - p1)
        + 3 * (t * t) * (p3 - p2)
    )
    m = t < end_portion
    out = out * (~m) + m * start_deriv
    m = t > (1 - end_portion)
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
    torch.pow(t, 2, out=temp2[..., :1])
    torch.addcmul(out, temp, temp2[..., :1], value=3, out=out)
    # out[:] += 3 * (t * t) * (p3 - p2)
    torch.subtract(p2, p1, out=temp)
    torch.mul(temp, t, out=temp)
    torch.subtract(1, t, out=temp2[..., :1])
    torch.addcmul(out, temp, temp2[..., :1], value=6, out=out)
    # out[:] += 6 * (1 - t) * t * (p2 - p1)
    torch.pow(temp2[..., :1], 2, out=temp2[..., :1])
    torch.subtract(p1, p0, out=temp)
    torch.addcmul(out, temp, temp2[..., :1], value=3, out=out)
    # out[:] += 3 * ((1 - t) ** 2) * (p1 - p0)

    m = t < end_portion
    torch.where(m, start_deriv, out, out=out)
    # out = out * (~m) + m * start_deriv
    m = t > (1 - end_portion)
    torch.where(m, end_deriv, out, out=out)
    # out = out * (~m) + m * end_deriv
    mem.reset_pointer()
    return out


def evaluate_cubic_bezier_second_derivative(p, t):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    return 6 * t * (p3 + 3 * (p1 - p2) - p0) + 6 * (p0 - 2 * p1 + p2)


def solve_cubic_bezier_second_derivative_equal_to_0(p):
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]
    denom = p3 + 3 * (p1 - p2) - p0
    m = denom.abs() <= 1e-5
    t = -(p0 - 2 * p1 + p2) / denom
    t = t.nan_to_num(2)
    m = m | (t <= 0) | (t >= 1)
    t = t * (~m) + m * 2
    t = t.amin(-1, keepdim=True)
    m = t >= 1
    t = t * (~m) + m * 0.5
    return t


def batch_arange(lengths):
    offsets = lengths.cumsum(0)
    n = offsets[-1].clone()
    offsets -= lengths
    offsets = torch.repeat_interleave(offsets, lengths)
    return torch.arange(n, device=lengths.device) - offsets


def rasterize_axis_aligned_parallelogram(
    start_corner_1, start_corner_2, widths, horizontal_mask
):
    # Inputs must be a parallelogram with its top and bottom sides aligned with the x-axis (flat horizontal lines)
    # and corresponding horizontal_mask set to True, or else its left and right sides aligned with the y-axis (flat vertical lines).
    #horizontal_mask = torch.full_like(horizontal_mask, True)
    bottom_left_corners = start_corner_1#torch.minimum(start_corner_1, start_corner_2)
    top_left_corners = start_corner_2#torch.maximum(start_corner_1, start_corner_2)
    heights = torch.where(
        horizontal_mask,
        top_left_corners[..., 1:] - bottom_left_corners[..., 1:],
        top_left_corners[..., :1] - bottom_left_corners[..., :1],
    )#.abs().ceil()
    orig_heights = heights
    heights = torch.where(horizontal_mask, heights, widths)
    widths = torch.where(horizontal_mask, widths, orig_heights)

    width_sign_mask = widths >= 0
    height_sign_mask = heights >= 0
    widths = widths.abs_().ceil_() + 1
    heights = heights.abs_().ceil_()

    n = (heights * widths).long().amax(0).view(-1)
    fragment_to_segment_inds = torch.repeat_interleave(torch.arange(n.shape[0], device=n.device), n)
    inds = batch_arange(n).unsqueeze(-1)
    widths_n = torch.repeat_interleave(widths, n, -2)
    ind_x = inds % widths_n
    ind_y = inds // widths_n
    ind_x = torch.where(torch.repeat_interleave(width_sign_mask, n, -2), ind_x, -(ind_x+1))
    ind_y = torch.where(torch.repeat_interleave(height_sign_mask, n, -2), ind_y, -(ind_y + 1))
    slope = torch.where(
        horizontal_mask,
        (start_corner_2[..., :1] - start_corner_1[..., :1]),
        (start_corner_2[..., 1:] - start_corner_1[..., 1:]),
    ) / (heights + 1e-4)

    slope = torch.where(height_sign_mask, slope, -slope)

    horizontal_mask_n = torch.repeat_interleave(horizontal_mask, n, -2)
    slope_n = torch.repeat_interleave(slope, n, -2)
    ind_x = torch.where(horizontal_mask_n, ind_x + (slope_n * ind_y).floor(), ind_x)
    #ind_y = torch.where(horizontal_mask_n, ind_y, ind_y + (slope_n * ind_x).floor())

    """bottom_left_corners[..., :1] = torch.where(
        horizontal_mask & ~width_sign_mask,
        bottom_left_corners[..., :1],# - (widths),
        bottom_left_corners[..., :1],
    )
    bottom_left_corners[..., 1:] = torch.where(
        ~horizontal_mask & ~width_sign_mask,
        bottom_left_corners[..., 1:],# - (heights),
        bottom_left_corners[..., 1:],
    )"""

    bottom_left_corners_n = torch.repeat_interleave(bottom_left_corners, n, -2)
    ind_x += bottom_left_corners_n[..., :1].floor()
    ind_y += bottom_left_corners_n[..., 1:].floor()
    local_dists = torch.zeros_like(ind_x)#((ind_x - bottom_left_corners[..., :1]).square_() + (ind_y - bottom_left_corners[..., 1:]).square_()).sqrt_()
    return ind_x.int(), ind_y.int(), fragment_to_segment_inds.unsqueeze(-1), local_dists


def rasterize_rectangle_no_duplicates(corner1, edge1, edge2):
    quadrant_mask = edge1 >= 0
    #++ -> corner1 + edge2, corner1, corner1 + edge1
    #+- -> corner1 + edge2 + edge1, corner1 + edge2, corner1
    #-- -> corner1 + edge1, corner1 + edge1 + edge2, corner1 + edge2
    #-+ -> corner1, corner1 + edge1, corner1 + edge1 + edge2
    corners = [corner1, corner1 + edge1, corner1 + edge1 + edge2, corner1 + edge2]
    m1 = quadrant_mask[...,:1]
    m2 = quadrant_mask[...,1:]
    c1 = torch.where(m1, torch.where(m2, corners[3], corners[2]), torch.where(m2, corners[0], corners[1]))
    c2 = torch.where(m1, torch.where(m2, corners[0], corners[3]), torch.where(m2, corners[1], corners[2]))
    c3 = torch.where(m1, torch.where(m2, corners[1], corners[0]), torch.where(m2, corners[2], corners[3]))

    c2 - c1
    raise NotImplementedError


def get_distance_to_line_segment(points, p1, p2, memory=None):
    projected_points = project_point_onto_line_segment(
        points,
        p1, p2,
        memory=None
    )
    return (points - projected_points).norm(p=2,dim=-1, keepdim=True)

    # local_dist = (local_window_xy - local_proj_onto_line).norm(p=2, dim=-1)
    local_window_xy_centered = torch.subtract(
        local_window_xy, local_proj_onto_line, out=local_proj_onto_line
    )

    global_dists = self.get_tensor([fragment_x.shape[-2]], dtype=torch.float)
    global_dists[:] = 1e12
    local_dist_pointer = self.memory.current_pointer
    local_dist = self.get_tensor(local_window_x.shape, dtype=torch.float)
    local_dist = torch.norm(local_window_xy_centered, p=2, dim=-1, out=local_dist)


def rasterize_rectangle_by_grid_transform(c1, c2, edge2):
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

    # === 1. Vectorized Calculation of Edge Vectors ===

    edge1 = c2 - c1

    # Add a small epsilon to prevent division by zero for zero-length edges
    edge1_len = torch.norm(edge1, dim=-1, keepdim=True)
    #edge1_len_safe = edge1_len + 1e-9
    #edge1_norm = edge1 / edge1_len_safe.unsqueeze(1)

    #perp_vec = torch.stack([-edge1_norm[:, 1], edge1_norm[:, 0]], dim=1)
    #edge2 = perp_vec * heights.unsqueeze(1)

    # === 2. Determine Grid Density for Each Parallelogram ===
    # We need enough steps along each edge to not miss any integer grid lines.
    # The number of steps is the ceiling of the vector's norm (its length).
    edge2_len = torch.norm(edge2, dim=-1, keepdim=True)  # heights are the lengths of edge2

    # Use clamp(min=1) to handle degenerate zero-area rectangle
    sqrt_2 = math.sqrt(2)
    num_u_steps = torch.ceil(edge1_len.amax(0) * sqrt_2).long() + 1
    num_v_steps = torch.ceil(edge2_len.amax(0) * sqrt_2).long() + 1

    # === 3. Vectorized Generation of (u,v) Coordinates ===
    # This section creates a "ragged" tensor of grid coordinates without loops.
    num_points_per_para = (num_u_steps * num_v_steps).long().view(-1)
    total_points = num_points_per_para.sum()

    # Create an index mapping each final point back to its original parallelogram
    para_indices = torch.repeat_interleave(torch.arange(n, device=device), num_points_per_para)

    # Generate local grid indices (0, 1, 2, ...) for each parallelogram's flattened grid
    end_indices = torch.cumsum(num_points_per_para, dim=0)
    start_indices = end_indices - num_points_per_para
    local_indices = (torch.arange(total_points, device=device) - torch.repeat_interleave(start_indices,
                                                                                        num_points_per_para)).unsqueeze(-1)

    # Gather the number of u-steps corresponding to each point
    u_steps_for_points = num_u_steps[para_indices]

    # Calculate the u and v indices for each point within its local grid
    u_idx = local_indices % u_steps_for_points
    v_idx = local_indices // u_steps_for_points

    # Convert indices to normalized [0, 1] coordinates
    # Denominator is clamped to 1 to avoid division by zero if a dimension has only one step
    u = u_idx / (num_u_steps[para_indices] - 1).clamp(min=1).float()
    v = v_idx / (num_v_steps[para_indices] - 1).clamp(min=1).float()

    # === 4. Transform Grid Points to World Coordinates ===
    # Gather the geometric data for each point using the parallelogram index
    c1_mapped = c1[:, para_indices]
    edge1_mapped = edge1[:, para_indices]
    edge2_mapped = edge2[:, para_indices]

    # Apply the transformation: P = C1 + u * Edge1 + v * Edge2
    # u and v need to be unsqueezed to correctly broadcast with the [M, 2] edge vectors
    points_float = c1_mapped + u * edge1_mapped + v * edge2_mapped

    # === 5. Round to Get Final Integer Coordinates ===
    # Rounding is used instead of flooring/ceiling to get the nearest integer coordinate.
    # This provides good coverage centered on the parallelogram's area.
    points = torch.round(points_float).int()

    #local_dists = torch.zeros_like(ind_x, dtype=torch.float)  # ((ind_x - bottom_left_corners[..., :1]).square_() + (ind_y - bottom_left_corners[..., 1:]).square_()).sqrt_()
    return points, para_indices


if __name__ == '__main__':
    # A batch of three rectangle to test various cases
    # 1. A simple axis-aligned rectangle
    # 2. A thin, long parallelogram at a 45-degree angle
    # 3. A general parallelogram
    rectangle_batch = torch.tensor([
        [1.0, 1.0, 6.0, 1.0, 3.0],  # A 5x3 rectangle. ||E1||=5, ||E2||=3.
        [10.0, 10.0, 10.707, 10.707, 10.0],  # Thin (width=1) and long (height=10) at 45 deg
        [20.0, 5.0, 23.0, 6.0, 4.0]  # Slanted parallelogram
    ])

    # Move to GPU if available
    if torch.cuda.is_available():
        rectangle_batch = rectangle_batch.cuda()
        print("Running on GPU")
    else:
        print("Running on CPU")

    print("\n--- Running Grid Transform Rasterizer (allows duplicates) ---")
    rasterized_points = rasterize_rectangle_by_grid_transform(rectangle_batch)

    # For inspection, let's also find the unique points
    unique_points = torch.unique(rasterized_points, dim=0)

    print(f"\nInput rectangle (Batch of {rectangle_batch.shape[0]}):")
    print(rectangle_batch)
    print(f"\nTotal Rasterized Points Generated (with duplicates): {rasterized_points.shape[0]}")
    print(f"Number of Unique Points Found: {unique_points.shape[0]}")

    print("\nRasterized Points (showing first 50):")
    print(rasterized_points[:50])

    print("\nUnique Rasterized Points (showing first 50):")
    print(unique_points[:50])

    # Verify the coverage for the first parallelogram (rectangle from [1,1] to [6,4])
    # Expected unique points: x in [1..6], y in [1..4]. Total = 6*4 = 24.
    p1_mask = (unique_points[:, 0] >= 1) & (unique_points[:, 0] <= 6) & \
              (unique_points[:, 1] >= 1) & (unique_points[:, 1] <= 4)
    count_p1_unique = p1_mask.sum()
    print(f"\nUnique points found within the bounds of the first parallelogram: {count_p1_unique} (Expected: ~24)")


def squish_batch_dims(func, start=1, end=-1):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        def s(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.view([*x.shape[:start], math.prod(x.shape[start:end]), *x.shape[end:]])
        args = [s(_) for _ in args]
        kwargs = {k: s(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper_func


#@squish_batch_dims
def rasterize_polygon_border(vertices, next_vertices, next_perpendiculars, widths):
    line_segments = next_vertices - vertices
    line_segments = F.normalize(line_segments, p=2, dim=-1)
    null_mask = (next_vertices - vertices).norm(p=1, dim=-1, keepdim=True) < 1e-5
    line_segments = torch.where(null_mask, torch.full_like(line_segments, 1/math.sqrt(2)), line_segments)
    line_perpendiculars = torch.stack(
        (-line_segments[..., 1], line_segments[..., 0]), -1
    )
    line_perpendiculars *= widths
    next_perpendiculars = F.normalize(next_perpendiculars, p=2, dim=-1)
    next_perpendiculars *= widths
    start_corners = [vertices + line_perpendiculars, vertices - line_perpendiculars]
    start_corners = [torch.where(null_mask, start_corner - line_segments * widths, start_corner) for start_corner in start_corners]

    end_corners = [
        next_vertices + next_perpendiculars,
        next_vertices - next_perpendiculars,
    ]

    dots = [dot_product(end_corner - vertices, line_segments) for end_corner in end_corners]
    max_dot = torch.where(dots[0].abs() >= dots[1].abs(), dots[0], dots[1])
    max_dot = torch.where(null_mask, widths*4, max_dot)
    edge = (max_dot * line_segments)
    points, inds = rasterize_rectangle_by_grid_transform(*start_corners, edge)
    local_distance = get_distance_to_line_segment(points, vertices[:, inds], next_vertices[:,inds])
    return points[...,:1], points[...,1:], inds.unsqueeze(-1), local_distance

    # dot_product(line_segments, end_corner + a * e1) = 0
    # dot_product(line_segments, end_corner) + a * line_segments[...,:1] = 0
    # a = -dot_product(line_segments, end_corner) / line_segments[...,:1]
    # TOOD deal with divide by 0 here properly
    dots_horizontal = [
        (
            -dot_product(line_segments, end_corner - vertices) / line_segments[..., :1]
        ).nan_to_num_(1e12, 1e12, -1e12)
        for end_corner in end_corners
    ]

    line_dir = start_corners[1] - start_corners[0]
    line_norm = line_dir.norm(p=2, dim=-1, keepdim=True)
    line_dir_normed = F.normalize(line_dir, p=2, dim=-1) / (line_norm + 1e-6)

    def expand_along_line(projected_points):
        dots = [dot_product(p - start_corners[0], line_dir_normed) for p in projected_points]
        min_dots = torch.minimum(*dots).clamp_max_(0)
        max_dots = torch.maximum(*dots).clamp_min_(1)
        return start_corners[0] + min_dots * line_dir, start_corners[0] + max_dots * line_dir

    def proj_along_axis(x, d, axis=0):
        x = x.clone()
        x[..., axis:axis+1] += d
        return x

    proj_horizontal = [proj_along_axis(ec, d, 0) for ec, d in zip(end_corners, dots_horizontal)]
    expanded_start_horizontal = expand_along_line(proj_horizontal)

    a_horizontal = torch.where(
        dots_horizontal[0].abs() >= dots_horizontal[1].abs(),
        dots_horizontal[0],
        dots_horizontal[1],
    )
    dots_vertical = [
        (
            -dot_product(line_segments, end_corner - vertices) / line_segments[..., 1:]
        ).nan_to_num_(1e12, 1e12, -1e12)
        for end_corner in end_corners
    ]

    proj_vertical = [proj_along_axis(ec, d, 1) for ec, d in zip(end_corners, dots_vertical)]
    expanded_start_vertical = expand_along_line(proj_vertical)

    a_vertical = torch.where(
        dots_vertical[0].abs() >= dots_vertical[1].abs(),
        dots_vertical[0],
        dots_vertical[1],
    )

    horizontal_mask = a_horizontal.abs() < a_vertical.abs()
    horizontal_mask = torch.full_like(horizontal_mask, True)
    horizontal_widths = torch.where(horizontal_mask, a_horizontal, a_vertical)

    start_corners = [torch.where(horizontal_mask, h, v) for h, v in (expanded_start_horizontal, expanded_start_vertical)]

    max_width = widths.max()
    horizontal_widths.clamp_(min=-max_width, max=max_width)

    return rasterize_axis_aligned_parallelogram(
        *[
            _#.view(-1, _.shape[-1])
            for _ in start_corners + [-horizontal_widths, horizontal_mask]
        ]
    )


#@squish_batch_dims
def rasterize_polygon(vertices, next_vertices, num_vertices_per_object):
    num_vertices_per_object = num_vertices_per_object.view(-1)
    y_ranges = (next_vertices[...,1].ceil().int() - vertices[...,1].ceil().int()).abs().amax(0)
    fragment_to_object_inds = torch.repeat_interleave(torch.arange(num_vertices_per_object.shape[0], device=vertices.device), num_vertices_per_object)
    fragment_to_object_inds = torch.repeat_interleave(fragment_to_object_inds, y_ranges)
    inds_y = batch_arange(y_ranges).unsqueeze(-1)
    vertices = torch.repeat_interleave(vertices, y_ranges, -2)
    next_vertices = torch.repeat_interleave(next_vertices, y_ranges, -2)
    inds_y = torch.where(next_vertices[...,1:] < vertices[...,1:], -(inds_y+1), inds_y)
    slopes = (next_vertices[...,:1] - vertices[...,:1]) / (next_vertices[...,1:] - vertices[...,1:])
    inds_x = (slopes * inds_y).round().int()
    inf = 10000000000
    inds_x.nan_to_num_(inf, inf, -inf)
    widths = (next_vertices[...,:1] - vertices[...,:1]).abs().floor().int()
    inds_x.clamp_(min=-widths, max=widths)
    inds_y += vertices[...,1:].ceil().int()
    inds_x += vertices[..., :1].ceil().int()
    min_y = torch.minimum(vertices[...,1:], next_vertices[...,1:])
    max_y = torch.maximum(vertices[...,1:], next_vertices[...,1:])
    m = (inds_y < min_y) != (inds_y < max_y)
    inds_y = torch.where(m, inds_y, torch.full_like(inds_y, inf))
    return inds_x, inds_y, fragment_to_object_inds.unsqueeze(-1)


def scatter_screen_inds_into_bounding_box(x, screen_ind_x, screen_ind_y, values, bounding_box_bottom_left_corner, bounding_box_width, offset):
    screen_ind_x -= bounding_box_bottom_left_corner[...,:1]
    screen_ind_y -= bounding_box_bottom_left_corner[..., 1:]
    bbox_ind = screen_ind_x + screen_ind_y * bounding_box_width
    bbox_ind += offset
    x.scatter_(-2, bbox_ind, values)
    return x


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
    ):
        self.num_bezier_parameters = 4
        self.num_texture_points = num_texture_points
        self.filled = filled
        if triangle_collection is not None:
            device = COMPUTING_DEFAULTS.render_device
            # logger = LoggerManager.instance().set_class("batching")
            # logger.log_message(
            #    f"{[_.num_segments_per_circuit for _ in triangle_collection]}"
            # )
            # self.num_segments_per_circuit = torch.cat([_.num_segments_per_circuit for _ in triangle_collection]).to(device)
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
                                triangle.grid_height,
                                triangle.grid_width,
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
            self.padding = max(self.border_width.amax().ceil().long() + 1, 5)
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
            # logger = LoggerManager.instance().set_class("rendering")
            # logger.log_message(
            #    f"Attempting repeat_interleave with arguments {arange_num_segments_per_oject},"
            #    f"{self.num_segments_per_object}, device: {x.device}, x.shape: {x.shape}"
            # )
            segment_to_object_scatter_inds = torch.repeat_interleave(
                arange_num_segments_per_oject, self.num_segments_per_object, -1
            ).view(1, -1, 1)
            self.segment_to_object_scatter_inds = segment_to_object_scatter_inds

            def log_var(name, var):
                pass
                # logger.log_message(f"{name} {var.shape},\n{var.dtype},\n {var}\n")

            log_var("segment_to_object_scatter_inds", segment_to_object_scatter_inds)
            arange_num_segments_per_oject = arange_num_segments_per_oject.view(1, -1, 1)
            log_var("arange_num_segments_per_oject", arange_num_segments_per_oject)
            log_var("x0", x[..., 0])
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
            log_var(
                "object_bounding_corners_bottom_left",
                object_bounding_corners_bottom_left,
            )
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
            log_var(
                "object_bounding_corners_top_right", object_bounding_corners_top_right
            )

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

            log_var("object_bounding_box_dimensions", object_bounding_box_dimensions)
            log_var("object_bounding_box_num_pixels", object_bounding_box_num_pixels)
            # num_fragments = object_bounding_box_num_pixels.long().sum()
            num_fragments = torch.sum(object_bounding_box_num_pixels).item()
            self.num_fragments_fill = num_fragments / len(
                object_bounding_box_num_pixels
            )
            if self.first_projection:
                return None

            # LoggerManager.instance().set_class("rendering").log_message(
            #    f"num_fragments:  {num_fragments}"
            # )
            # object_to_fragment_gather_inds = torch.repeat_interleave(
            #     torch.arange(object_bounding_box_num_pixels.numel(),
            #                  device=x.device), object_bounding_box_num_pixels.view(-1), -1,
            #     output_size=num_fragments).unsqueeze(-1)
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

            # LoggerManager.instance().set_class("rendering").log_message(
            #    f"first bg:  {object_offsets}, {object_to_fragment_gather_inds}"
            # )
            # object_fragment_inds = object_fragment_inds - broadcast_gather(object_offsets, -2, object_to_fragment_gather_inds, keepdim=True)
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

            # LoggerManager.instance().set_class("rendering").log_message(f"bg3")
            # object_bounding_corners_bottom_left_for_frags = broadcast_gather(squish(object_bounding_corners_bottom_left, 0, 1), -2, object_to_fragment_gather_inds, keepdim=True)
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

            # LoggerManager.instance().set_class("rendering").log_message(f"done")

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
        maximum_net_length = control_net_lengths.amax()
        self.num_sampled_points = (
            (maximum_net_length * 0.25).ceil().long().clamp_min_(1)
        ).item()  # 1 sample per 4 pixel widths.
        return self

    def get_batch_identifier(self):
        return f"{__class__}_{self.num_texture_points}_{self.filled}"

    def get_memory_used_per_timestep(self):
        num_fragments_border_segments = self.num_fragments_per_frame
        num_fragments_border_samples = (
            self.num_sampled_points
            * int((self.padding * 2 + 1) ** 2)
            * self.corners.shape[-3]
        )
        return (
            self.num_fragments_fill * 256
            + num_fragments_border_segments * 256
            + num_fragments_border_samples * 128
        )

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

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"starting rendering with {corners.shape}, {normals.shape}, {mob_center.shape}, "
        #    f"{colors.shape}, {ray_origin.shape}"
        # )

        num_objects = len(num_segments_per_object)  #

        if window_coords is None:
            window_coords = 0, 0, screen_width, screen_height
        window_height = window_coords[-1] - window_coords[1]
        window_width = window_coords[-2] - window_coords[0]
        start_x, start_y, end_x, end_y = window_coords

        bounding_corners = select_time(self.bounding_corners)

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
        inds = self.get_tensor(fragment_x.shape, dtype=torch.int)
        inds_pointer = self.memory.current_pointer

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got bounding boxes"
        # )
        control_points = corners
        # t = torch.linspace(0, 1, self.num_sampled_points, device=control_points.device)
        polygon_vertices = self.get_tensor(
            [
                *control_points.shape[:-2],
                self.num_sampled_points,
                control_points.shape[-1],
            ]
        )
        self.memory.save_pointer()
        t = self.get_tensor([self.num_sampled_points], dtype=torch.float)
        torch.linspace(0, 1, self.num_sampled_points, device=t.device, out=t)
        ##polygon_vertices = self.get_tensor((*control_points.shape[:3], num_sampled_points, 2))
        polygon_vertices = evaluate_cubic_bezier(
            control_points, t.unsqueeze(-1), polygon_vertices, self.memory
        )

        # polygon_vertices = evaluate_cubic_bezier_old3(control_points, t.unsqueeze(-1))
        # assert polygon_vertices.shape == [T, N, P, 2] (time (frames), num segments, num control points per segment, 2D)
        # polygon_vertices = squish(polygon_vertices, -3, -2)  # shape [T, N, S*P, 2]
        next_polygon_vertices = polygon_vertices.roll(shifts=-1, dims=-2)

        # Change the last next_vertice from the start of this segment to the start of the next segment.
        # next_segments = broadcast_gather(polygon_vertices, -3, next_segment_inds, keepdim=True)
        next_segments = broadcast_gather(
            polygon_vertices, -3, next_segment_inds, keepdim=True, out=self.memory
        )
        next_polygon_vertices[..., -1, :] = next_segments[..., 0, :]
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

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got line_segment_lengths"
        # )

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
            half_local_window_size = max(
                border_width.amax().ceil().long(),
                line_segment_lengths.amax().ceil().long(),
            )
        else:
            if (border_width.amax() < 0.1) | (line_segment_lengths.amax() < 0.1):
                return None
            half_local_window_size = border_width.amax().ceil().long()
        half_local_window_size = half_local_window_size.item()
        local_window_size = half_local_window_size * 2 + 1
        # if local_window_size > 50:
        #    raise RuntimeError(
        #        "Filled Bezier Circuit is not closed, make sure that the starting and ending points"
        #        "of your Bezier circuits are the same, or else set filled=False."
        #    )

        self.memory.reset_pointer()

        next_polygon_perpendiculars = next_polygon_vertices.roll(shifts=-1, dims=-2)
        next_segments = broadcast_gather(
            next_polygon_vertices, -3, next_segment_inds, keepdim=True, out=self.memory
        )
        next_polygon_perpendiculars[..., -1, :] = next_segments[..., 0, :]
        next_polygon_perpendiculars = (
            next_polygon_perpendiculars - next_polygon_vertices
        )
        next_polygon_perpendiculars = torch.stack(
            (-next_polygon_perpendiculars[..., 1], next_polygon_perpendiculars[..., 0]),
            dim=-1,
        )

        border_width_o = broadcast_gather(
            border_width, -2, self.segment_to_object_scatter_inds
        )
        local_fragment_x, local_fragment_y = rasterize_polygon_border(
            polygon_vertices,
            next_polygon_vertices,
            next_polygon_perpendiculars,
            border_width_o.unsqueeze(-1),
        )

        # self.memory.save_pointer()

        object_bounding_box_dimensions_for_segments = (
            broadcast_gather(
                object_bounding_box_dimensions, -2, self.segment_to_object_scatter_inds
            )
        ).unsqueeze(-1)
        object_bounding_corners_bottom_left_for_segments = (
            broadcast_gather(
                object_bounding_corners_bottom_left,
                -2,
                self.segment_to_object_scatter_inds,
            )
        ).unsqueeze(-1)
        # bbox_x = local_window_x - object_bounding_corners_bottom_left_for_segments[...,:1,:]
        # bbox_y = local_window_y - object_bounding_corners_bottom_left_for_segments[...,1:,:]
        # pointer = self.memory.current_pointer
        bbox_x = self.get_tensor(local_window_x.shape, dtype=torch.long)
        torch.subtract(
            local_window_x,
            object_bounding_corners_bottom_left_for_segments[..., :1, :],
            out=bbox_x,
        )
        bbox_y = self.get_tensor(local_window_y.shape, dtype=torch.long)
        torch.subtract(
            local_window_y,
            object_bounding_corners_bottom_left_for_segments[..., 1:, :],
            out=bbox_y,
        )
        # bbox_num_pixels = object_bounding_box_dimensions_for_segments.prod(-2, keepdim=True)
        bbox_num_pixels = self.get_tensor(
            [
                *object_bounding_box_dimensions_for_segments.shape[:-2],
                1,
                object_bounding_box_dimensions_for_segments.shape[-1],
            ],
            dtype=object_bounding_box_dimensions_for_segments.dtype,
        )
        torch.prod(
            object_bounding_box_dimensions_for_segments,
            -2,
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
            object_bounding_box_dimensions_for_segments[..., :1, :],
            bbox_y,
            value=1,
            out=local_to_bbox_inds,
        )
        local_to_bbox_inds.clamp_min_(0)
        local_to_bbox_inds.clamp_max_(bbox_num_pixels - 1)

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got local_to_bbox"
        # )
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
        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"attempting repeat interleave3 {offsets} {corners.shape[0]}"
        # )
        # offsets_for_segments = squish(torch.repeat_interleave(unsquish(offsets, 0, -corners.shape[0]), num_segments_per_object, -2).unsqueeze(-1), 0, 1)
        offsets_for_segments = squish(
            broadcast_gather(
                unsquish(offsets, 0, -corners.shape[0]),
                -2,
                self.segment_to_object_scatter_inds,
            ).unsqueeze(-1),
            0,
            1,
        )
        local_to_global_inds = squish(local_to_bbox_inds, 0, 1)
        local_to_global_inds += offsets_for_segments.view(-1, 1, 1)
        local_to_global_inds = local_to_global_inds.view(-1)

        local_to_global_inds.clamp_(min=0, max=fragment_x.shape[-2] - 1)

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got local to global"
        # )

        # invalid_mask = ((bbox_x < 0) | (bbox_x > bounding_box_widths.unsqueeze(-2))) | (((bbox_y < 0) | (bbox_y > bounding_box_heights.unsqueeze(-2))))
        # invalid_mask = ((bbox_x >= object_bounding_box_dimensions_for_segments[...,:1,:]) |
        #                (bbox_y < 0) | (bbox_y > object_bounding_box_dimensions_for_segments[...,1:,:]))
        invalid_mask = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        pointer = self.memory.current_pointer
        temp_bool = self.get_tensor(bbox_x.shape, dtype=torch.bool)
        torch.greater_equal(
            bbox_x,
            object_bounding_box_dimensions_for_segments[..., :1, :],
            out=invalid_mask,
        )
        torch.lt(bbox_y, 0, out=temp_bool)
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        torch.gt(
            bbox_y,
            object_bounding_box_dimensions_for_segments[..., 1:, :],
            out=temp_bool,
        )
        torch.logical_or(invalid_mask, temp_bool, out=invalid_mask)
        self.memory.current_pointer = pointer

        # LoggerManager.instance().set_class("rendering").log_message(f"got invalid mask")

        # Note we need to keep negative x inds around for now, because we cumsum across rows from the left
        # to count intersections, we will cull negative x inds later.
        zero = self.get_tensor([1])
        zero[:] = 0
        local_intersection_counts = torch.where(
            invalid_mask, zero, local_intersection_counts, out=local_intersection_counts
        )

        # global_intersection_counts = torch_scatter.scatter_sum(local_intersection_counts.view(-1), local_to_global_inds, -1, dim_size=fragment_x.shape[-2])
        # out = torch.zeros((fragment_x.shape[-2],), device=fragment_x.device)
        out = self.get_tensor([fragment_x.shape[-2]])
        out[:] = 0

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"attempting scatter_add {local_intersection_counts}"
        # )
        global_intersection_counts = torch.scatter_add(
            out, -1, local_to_global_inds, local_intersection_counts.view(-1), out=out
        )
        # self.memory.current_pointer = local_intersection_counts_pointer

        # Now do border mask.
        # local_window_xy = torch.stack((local_window_x, local_window_y), -1)
        local_window_xy = self.get_tensor([*local_window_x.shape, 2], dtype=torch.long)
        local_window_xy[..., 0] = local_window_x
        local_window_xy[..., 1] = local_window_y
        local_proj_onto_line = project_point_onto_line_segment(
            local_window_xy,
            polygon_vertices.unsqueeze(-2),
            next_polygon_vertices.unsqueeze(-2),
            memory=self.memory,
        )

        # local_dist = (local_window_xy - local_proj_onto_line).norm(p=2, dim=-1)
        local_window_xy_centered = torch.subtract(
            local_window_xy, local_proj_onto_line, out=local_proj_onto_line
        )

        global_dists = self.get_tensor([fragment_x.shape[-2]], dtype=torch.float)
        global_dists[:] = 1e12
        local_dist_pointer = self.memory.current_pointer
        local_dist = self.get_tensor(local_window_x.shape, dtype=torch.float)
        local_dist = torch.norm(local_window_xy_centered, p=2, dim=-1, out=local_dist)

        # dist_invalid_mask = invalid_mask | (bbox_x < 0)
        pointer = self.memory.current_pointer
        dist_invalid_mask = self.get_tensor(invalid_mask.shape, dtype=torch.bool)
        torch.lt(bbox_x, 0, out=dist_invalid_mask)
        torch.logical_or(dist_invalid_mask, invalid_mask, out=dist_invalid_mask)

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got dist_invalid mask"
        # )

        posinf = zero
        posinf[:] = 1e12
        local_dist = torch.where(dist_invalid_mask, posinf, local_dist, out=local_dist)

        # Handle portion_of_curve_drawn
        # self.expand_verts_to_frags(self.portion_of_curve_drawn)
        num_vertices_per_object = (
            num_segments_per_object.view(-1, 1) * self.num_sampled_points
        )
        num_vertices_per_object += 1
        threshold_for_drawing = broadcast_gather(
            portion_of_curve_drawn * num_vertices_per_object,
            -2,
            self.segment_to_object_scatter_inds,
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
        local_dist = torch.where(
            vertex_number >= threshold_for_drawing, posinf, local_dist, out=local_dist
        )

        self.memory.current_pointer = pointer
        # global_dists = torch.empty((fragment_x.shape[-2],), device=control_points.device)
        """global_dists = torch_scatter.scatter_min(local_dist.view(-1),
                                                               local_to_global_inds.clamp(min=0,
                                                                                          max=fragment_x.shape[-2] - 1),
                                                               -1, out=global_dists)[0]"""

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"attempting scatter_reduce"
        # )
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
        global_dists -= 1e-3

        # Count the number of intersections in the horizontal ray to this pixel's left.
        left_intersection_counts = torch.cumsum(
            global_intersection_counts, -1, out=global_intersection_counts
        )
        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"got left_intersection_counts {local_intersection_counts.shape}"
        # )

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

        if self.filled:
            bool_interior_mask = self.get_tensor(interior_mask.shape, torch.bool)
            min_border = self.get_tensor(border_mask.shape, border_mask.dtype)
            min_border = torch.minimum(
                border_mask,
                torch.tensor((1.5,), dtype=torch.float, device=min_border.device),
                out=min_border,
            )
            bool_interior_mask[:] = interior_mask
            border_mask = torch.where(
                bool_interior_mask, border_mask, min_border, out=border_mask
            )
            self.memory.current_pointer = pointer
        torch.less_equal(global_dists.unsqueeze(-1), border_mask, out=border_mask)

        # fragment_coords = torch.cat((fragment_x, fragment_y), -1).float()

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"finished constructing mask {fragment_x.shape},"
        # )

        # TODO subtract window_start from x and y (so they are 0 centered.
        # inds = (fragment_x - start_x) + (fragment_y - start_y) * window_width
        torch.multiply(fragment_y, window_width, out=inds)
        inds -= start_y * window_width + start_x
        inds += fragment_x

        window_size = window_width * window_height

        if self.filled:
            pass  # border_mask *= interior_mask
        else:
            interior_mask[:] = 0
        # TODO does this need to clip based on x and y instead of inds for window?
        m = (inds < window_size) & ((interior_mask > 0) | (border_mask > 0))
        m = m.reshape(-1)
        border_mask = border_mask.view(-1)[m].unsqueeze(-1)
        # g_offsets = torch.arange(0, corners.shape[0], device=inds.device) * window_size
        g_offsets = self.get_tensor([corners.shape[0]], dtype=torch.long)
        torch.arange(0, corners.shape[0], device=g_offsets.device, out=g_offsets)
        g_offsets *= window_size
        frame_to_fragment_gather_inds = object_to_fragment_gather_inds // num_objects
        g_offsets = self.expand_verts_to_frags(
            g_offsets.unsqueeze(-1), frame_to_fragment_gather_inds
        )
        inds = inds + g_offsets
        inds = inds.view(-1)
        inds = inds[m]
        # unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        mob_center_for_frags = self.expand_verts_to_frags(
            squish(mob_center, 0, 1), object_to_fragment_gather_inds
        )
        normals_for_frags = self.expand_verts_to_frags(
            squish(normals, 0, 1), object_to_fragment_gather_inds
        )

        def expo(x, select=True, gather_inds=object_to_fragment_gather_inds):
            if select:
                x = select_time(x)
            x = x.view(-1, x.shape[-1])
            return self.expand_verts_to_frags(x, gather_inds)

        screen_basis = unsquish(
            expo(
                squish(screen_basis, -2, -1),
                False,
                gather_inds=frame_to_fragment_gather_inds,
            ),
            -1,
            3,
        )
        screen_point = expo(
            screen_point, False, gather_inds=frame_to_fragment_gather_inds
        )
        ray_origin = expo(ray_origin, False, gather_inds=frame_to_fragment_gather_inds)
        # screen_basis = screen_basis / screen_basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        screen_basis_norm = self.get_tensor(
            [*screen_basis.shape[:-1], 1], dtype=screen_basis.dtype
        )
        torch.norm(screen_basis, p=2, dim=-1, keepdim=True, out=screen_basis_norm)
        screen_basis_norm_sq = self.get_tensor(
            screen_basis_norm.shape, dtype=screen_basis.dtype
        )
        torch.square(screen_basis_norm, out=screen_basis_norm_sq)
        screen_basis_norm_sq.clamp_min_(1e-6)
        torch.div(screen_basis, screen_basis_norm_sq, out=screen_basis)
        ray_direction = F.normalize(
            (
                screen_point
                + ((fragment_x - screen_width * 0.5) / (screen_height * 0.5))
                * screen_basis[..., 0, :]
                + ((fragment_y - screen_height * 0.5) / (screen_height * 0.5))
                * screen_basis[..., 1, :]
            )
            - ray_origin,
            p=2,
            dim=-1,
        )
        dists = self.raycast_onto_plane(
            ray_origin, ray_direction, mob_center_for_frags, normals_for_frags
        )
        if self.num_texture_points > 0:
            # LoggerManager.instance().set_class("rendering").log_message(
            #    f"starting coloring process {dists.shape},"
            # )
            proj_onto_mobs = ray_origin + dists * ray_direction
            mob_centers = expo(self.mob_center)
            mob_basis1 = expo(self.basis1)
            mob_basis2 = expo(self.basis2)
            # grid_width = expo(self.grid_width).long()
            grid_width = self.get_tensor(expo(self.grid_width).shape, dtype=torch.long)
            grid_height = self.get_tensor(
                expo(self.grid_height).shape, dtype=torch.long
            )
            grid_width_float = self.get_tensor(
                expo(self.grid_width).shape, dtype=torch.float
            )
            grid_height_float = self.get_tensor(
                expo(self.grid_height).shape, dtype=torch.float
            )
            grid_width_float[:] = expo(self.grid_width)
            grid_height_float[:] = expo(self.grid_height)
            grid_width[:] = grid_width_float
            grid_height[:] = grid_height_float
            # Free float versions as they're no longer needed
            float_size = grid_height_float.numel() * grid_height_float.element_size()
            self.memory.current_pointer -= float_size  # Free grid_height_float
            float_size = grid_width_float.numel() * grid_width_float.element_size()
            self.memory.current_pointer -= float_size  # Free grid_width_float
            # disps_from_mobs = proj_onto_mobs - mob_centers
            disps_from_mobs = self.get_tensor(
                proj_onto_mobs.shape, dtype=proj_onto_mobs.dtype
            )
            torch.subtract(proj_onto_mobs, mob_centers, out=disps_from_mobs)
            # offsets = expo((torch.arange(self.mob_center.shape[1]*(time_end - time_start), device=self.colors.device)*self.colors.shape[-2]).view(1,-1,1,1))
            arange_size = self.mob_center.shape[1] * (time_end - time_start)
            arange_offsets = self.get_tensor([arange_size], dtype=torch.long)
            torch.arange(arange_size, device=arange_offsets.device, out=arange_offsets)
            arange_offsets *= self.colors.shape[-2]
            offsets = expo(arange_offsets.view(1, -1, 1, 1))

            def get_c(b):
                dot_out = self.get_tensor([*b.shape[:-1], 1])
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
            x = get_c(mob_basis2)
            x *= grid_height
            x = x.clamp_max_(grid_height - 1).clamp_min_(0)

            # y = (get_c(mob_basis1) * grid_width).clamp_max_(grid_width-1).clamp_min_(0)
            y = get_c(mob_basis1)
            y *= grid_width
            y = y.clamp_max_(grid_width - 1).clamp_min_(0)

            # xr = x % 1
            xr = self.get_tensor(x.shape, dtype=x.dtype)
            torch.remainder(x, 1, out=xr)
            # yr = y % 1
            yr = self.get_tensor(y.shape, dtype=y.dtype)
            torch.remainder(y, 1, out=yr)
            # w1 = (1-xr) * (1-yr)
            w1 = self.get_tensor(xr.shape, dtype=xr.dtype)
            w2 = self.get_tensor(xr.shape, dtype=xr.dtype)
            w3 = self.get_tensor(xr.shape, dtype=xr.dtype)
            w4 = self.get_tensor(xr.shape, dtype=xr.dtype)
            temp2 = self.get_tensor(yr.shape, dtype=yr.dtype)
            temp1 = self.get_tensor(xr.shape, dtype=xr.dtype)
            torch.subtract(1, xr, out=temp1)
            torch.subtract(1, yr, out=temp2)
            torch.mul(temp1, temp2, out=w1)
            # w2 = xr * (1-yr)
            torch.mul(xr, temp2, out=w2)
            # w3 = (1-xr)*yr
            torch.mul(temp1, yr, out=w3)
            # w4 = xr * yr
            torch.mul(xr, yr, out=w4)
            # Free temp1 and temp2 as they're no longer needed
            temp_size = temp1.numel() * temp1.element_size()
            self.memory.current_pointer -= temp_size  # Free temp1
            temp_size = temp2.numel() * temp2.element_size()
            self.memory.current_pointer -= temp_size  # Free temp2

            # x_floor = (x).floor().long()
            x_floor = self.get_tensor(x.shape, dtype=torch.long)
            x_ciel = self.get_tensor(x.shape, dtype=torch.long)
            y_floor = self.get_tensor(y.shape, dtype=torch.long)
            y_ciel = self.get_tensor(y.shape, dtype=torch.long)
            y_ciel_float = self.get_tensor(y.shape, dtype=y.dtype)
            y_floor_float = self.get_tensor(y.shape, dtype=y.dtype)
            x_ciel_float = self.get_tensor(x.shape, dtype=x.dtype)
            x_floor_float = self.get_tensor(x.shape, dtype=x.dtype)
            torch.floor(x, out=x_floor_float)
            x_floor[:] = x_floor_float
            # x_ciel = (x).ceil().long()
            torch.ceil(x, out=x_ciel_float)
            x_ciel[:] = x_ciel_float
            # y_floor = (y).floor().long()
            torch.floor(y, out=y_floor_float)
            y_floor[:] = y_floor_float
            # y_ciel = (y).ceil().long()
            torch.ceil(y, out=y_ciel_float)
            y_ciel[:] = y_ciel_float
            # Free float versions as they're no longer needed
            float_size = y_ciel_float.numel() * y_ciel_float.element_size()
            self.memory.current_pointer -= float_size  # Free y_ciel_float
            float_size = y_floor_float.numel() * y_floor_float.element_size()
            self.memory.current_pointer -= float_size  # Free y_floor_float
            float_size = x_ciel_float.numel() * x_ciel_float.element_size()
            self.memory.current_pointer -= float_size  # Free x_ciel_float
            float_size = x_floor_float.numel() * x_floor_float.element_size()
            self.memory.current_pointer -= float_size  # Free x_floor_float

            colos = squish(select_time(self.colors), 0, 2)
            interpolated_colors = 0
            temp_long = self.get_tensor(w1.shape, torch.long)
            sum_w = self.get_tensor(w1.shape, w1.dtype)
            sum_w[:] = 0
            gathered_colors = self.get_tensor([*w1.shape[:-1], 5], w1.dtype)
            interpolated_colors = self.get_tensor([*w1.shape[:-1], 5], w1.dtype)
            interpolated_colors[:] = 0
            for w, x, y in [
                (w1, x_floor, y_floor),
                (w2, x_ciel, y_floor),
                (w3, x_floor, y_ciel),
                (w4, x_ciel, y_ciel),
            ]:
                x = torch.addcmul(x, y, grid_height, out=temp_long)
                # x = x + y * grid_height
                x += offsets
                sum_w = torch.add(w, sum_w, out=sum_w)
                c = broadcast_gather(colos, -2, x, out=gathered_colors, keepdim=True)
                interpolated_colors = torch.addcmul(
                    interpolated_colors, c, w, out=interpolated_colors
                )
            interpolated_colors /= sum_w
        else:
            interpolated_colors = self.expand_verts_to_frags(
                squish(colors, 0, 1), object_to_fragment_gather_inds, -2
            )

        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"finished coloring {interpolated_colors.shape},"
        # )

        # output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        # output_frags[:] = 0

        def get_frags(ws, fragment_coords=None):
            def get_colors():
                colors = interpolated_colors
                colors = colors.reshape(-1, colors.shape[-1])
                colors = colors[m]

                if self.filled:
                    border_colors_frags = self.expand_verts_to_frags(
                        squish(border_colors, 0, 1), object_to_fragment_gather_inds
                    )
                    border_colors_frags = border_colors_frags.reshape(
                        -1, border_colors_frags.shape[-1]
                    )
                    border_colors_frags = border_colors_frags[m]
                    colors[..., :] = (
                        colors[..., :] * (1 - border_mask)
                        + border_mask * border_colors_frags
                    )
                return colors

            colors = get_colors()
            dists2 = dists.reshape(-1)
            dists2 = dists2[m]

            return colors, dists2

        colors, dists = get_frags(1)
        # LoggerManager.instance().set_class("rendering").log_message(
        #    f"finished getting frags {colors.shape},"
        # )
        self.memory.current_pointer = inds_pointer
        return colors, dists, inds
