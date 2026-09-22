"""Turn explicitly styled planar strokes into ordinary filled circuits.

Skia (already a dependency for vector boolean operations) handles cubic offsets,
joins, caps, self intersections and miter clipping. Expansion happens during
projection, from the current frame's camera and stroke width. No authored Mob or
timeline is changed. The resulting boundaries use the usual AA, BVH, shadow and
path-tracing paths; default round strokes retain their existing distance bands.
"""

from __future__ import annotations

import numpy as np
import pathops
import torch
import torch.nn.functional as F

from algan.rendering.mps_compat import clamp_floor
from algan.settings import SETTINGS


def _path(cubics):
    path = pathops.Path()
    path.fillType = pathops.FillType.EVEN_ODD
    first = previous = None
    for cubic in cubics:
        if np.max(np.abs(cubic - cubic[0])) < 1e-8:
            continue
        if previous is None or np.linalg.norm(cubic[0] - previous) > 1e-5:
            if first is not None and np.linalg.norm(previous - first) <= 1e-5:
                path.close()
            path.moveTo(*cubic[0])
            first = cubic[0]
        path.cubicTo(*cubic[1:].reshape(-1))
        previous = cubic[-1]
    if first is not None and np.linalg.norm(previous - first) <= 1e-5:
        path.close()
    return path


def _cubics(path, tolerance):
    """Closed contours as cubics and relative next-segment indices."""
    path.convertConicsToQuads(tolerance)
    curves, following = [], []
    point = first = None
    contour_start = 0

    def line(end):
        nonlocal point
        curves.append(
            np.stack(
                (point, point + (end - point) / 3, point + 2 * (end - point) / 3, end)
            )
        )
        following.append(1)
        point = end

    for verb, points in path:
        points = np.asarray(points, dtype=np.float64)
        if verb == pathops.PathVerb.MOVE:
            point = first = points[0]
            contour_start = len(curves)
        elif verb == pathops.PathVerb.LINE:
            line(points[0])
        elif verb == pathops.PathVerb.QUAD:
            control, end = points
            curves.append(
                np.stack(
                    (
                        point,
                        point + (control - point) * 2 / 3,
                        end + (control - end) * 2 / 3,
                        end,
                    )
                )
            )
            following.append(1)
            point = end
        elif verb == pathops.PathVerb.CUBIC:
            curves.append(np.vstack((point, points)))
            following.append(1)
            point = points[-1]
        elif verb == pathops.PathVerb.CLOSE:
            if np.linalg.norm(point - first) > 1e-8:
                line(first)
            if len(curves) > contour_start:
                following[-1] = contour_start - len(curves) + 1
        else:
            raise RuntimeError(f"Unexpected stroke outline verb: {verb}")
    return curves, following


def _regions(cubics, width, style, filled, inside, tolerance):
    cap, join, limit = style
    source = _path(cubics)
    stroke = pathops.Path(source)
    if width > 1e-9:
        stroke.stroke(
            width * (2 if filled and inside else 1),
            getattr(pathops.LineCap, cap.upper() + "_CAP"),
            getattr(pathops.LineJoin, join.upper() + "_JOIN"),
            limit,
        )
        stroke.convertConicsToQuads(tolerance)
        stroke = pathops.simplify(stroke)
    else:
        stroke = pathops.Path()
    if not filled:
        return [_cubics(stroke, tolerance)]
    if inside:
        stroke = pathops.op(stroke, source, pathops.PathOp.INTERSECTION)
    interior = pathops.op(source, stroke, pathops.PathOp.DIFFERENCE)
    return [_cubics(interior, tolerance), _cubics(stroke, tolerance)]


def _expand_stroke(primitive, camera):
    """Replace this disposable render primitive's source with filled outlines."""
    style = getattr(primitive, "stroke_style", None)
    if style is None:
        return
    frames = camera.ray_origin.shape[0]
    device = primitive.corners.device

    def host(value):
        return value.detach().float().cpu().expand(frames, *value.shape[1:])

    centers = host(primitive.mob_center)
    normals = F.normalize(host(primitive.normals), dim=-1)
    helper = torch.zeros_like(normals)
    helper[..., 0] = 1
    alternate = torch.zeros_like(normals)
    alternate[..., 1] = 1
    helper = torch.where(normals[..., :1].abs() < 0.9, helper, alternate)
    u = F.normalize(torch.cross(helper, normals, dim=-1), dim=-1)
    v = torch.cross(u, normals, dim=-1)
    eye = host(camera.ray_origin).reshape(frames, 1, 3)
    screen = host(camera.screen_point).reshape(frames, 1, 3)
    screen_basis = host(camera.screen_basis).reshape(frames, 3, 3)
    height = float(getattr(camera, "output_screen_height", camera.screen_height))
    # Match the renderer's pixel_world_scale: perpendicular depth, not slant
    # range. Otherwise strokes get wider as they move towards a screen edge.
    depth = ((centers - eye) * F.normalize(screen - eye, dim=-1)).sum(-1).abs()
    pixel_size = (
        2
        * depth
        / clamp_floor(
            height
            * screen_basis[:, 1].norm(dim=-1)[:, None]
            * (screen - eye).norm(dim=-1),
            1e-12,
        )
    )
    widths = host(primitive.stroke_width)[..., 0].abs() * pixel_size
    counts = primitive.num_segments_per_object.detach().cpu().long().tolist()
    source = host(primitive.corners)
    parts = 2 if primitive.filled else 1
    circuit_count = len(counts)
    outlines = [[] for _ in range(circuit_count * parts)]
    offset = 0
    for circuit, count in enumerate(counts):
        relative = source[:, offset : offset + count] - centers[:, circuit, None, None]
        local = torch.stack(
            (
                (relative * u[:, circuit, None, None]).sum(-1),
                (relative * v[:, circuit, None, None]).sum(-1),
            ),
            -1,
        ).numpy()
        for frame in range(frames):
            regions = _regions(
                local[frame],
                float(widths[frame, circuit]),
                style,
                primitive.filled,
                SETTINGS.style.border_placement != "centered",
                max(float(pixel_size[frame, circuit]) * 0.025, 1e-8),
            )
            for part, region in enumerate(regions):
                outlines[circuit * parts + part].append(region)
        offset += count

    sizes = [max(1, max(len(curves) for curves, _ in region)) for region in outlines]
    corners = torch.empty((frames, sum(sizes), 4, 3), dtype=torch.float32, device="cpu")
    following = torch.zeros((frames, sum(sizes), 1, 1), dtype=torch.long, device="cpu")
    offset = 0
    for index, (size, region) in enumerate(zip(sizes, outlines)):
        circuit = index // parts
        for frame, (curves, next_indices) in enumerate(region):
            center = centers[frame, circuit]
            corners[frame, offset : offset + size] = center
            if curves:
                xy = torch.from_numpy(np.asarray(curves)).float()
                world = (
                    center
                    + xy[..., :1] * u[frame, circuit]
                    + xy[..., 1:] * v[frame, circuit]
                )
                corners[frame, offset : offset + len(curves)] = world
                corners[frame, offset + len(curves) : offset + size] = world[-1, -1]
                following[frame, offset : offset + len(curves), 0, 0] = torch.tensor(
                    next_indices, device="cpu"
                )
        offset += size
    primitive.corners = corners.to(device)
    primitive.next_segment_inds = following.to(device) + torch.arange(
        sum(sizes), device=device
    ).view(1, -1, 1, 1)
    primitive.num_segments_per_object = torch.tensor(
        sizes, dtype=torch.long, device=device
    )
    colors = primitive.colors
    border = primitive.stroke_color
    if colors.ndim == 3:
        colors = colors.unsqueeze(-2)
    if border.ndim == 3:
        border = border.unsqueeze(-2)
    if parts == 2:
        colors, border = torch.broadcast_tensors(colors, border)
        primitive.colors = torch.stack((colors, border), 2).flatten(1, 2)
    else:
        primitive.colors = border
    # A zero-width stroke or an entirely consumed interior has no region.
    # Degenerate padding alone is insufficient: ordinary filled circuits draw
    # a small anti-crack hairline even around a point. Hide empty regions in
    # each frame, including glow, while retaining their batch slots.
    visible = torch.tensor(
        [[bool(region[frame][0]) for region in outlines] for frame in range(frames)],
        device=device,
    )
    primitive.colors = primitive.colors * visible[:, :, None, None]
    for name in (
        "mob_center",
        "normals",
        "basis1",
        "basis2",
        "grid_width",
        "grid_height",
        "z_index",
        *primitive._surface_params,
    ):
        value = getattr(primitive, name, None)
        if value is not None:
            setattr(primitive, name, value.repeat_interleave(parts, dim=1))
    primitive.stroke_color = torch.zeros_like(primitive.colors)
    primitive.stroke_width = torch.zeros_like(primitive.mob_center[..., :1])
    primitive.filled = True
