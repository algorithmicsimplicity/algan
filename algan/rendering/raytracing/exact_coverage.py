"""Host construction of exact screen-space Bezier coverage contours.

The ray tracer's Bezier primitive already flattens every cubic with a bounded
screen-space chord error.  This module turns those directed polylines into the
explicit boundaries used by the exact analytic-AA kernels:

* filled circuits get an outward outline and, when they have a border, an
  inward fill-only contour;
* unfilled circuits become round-joined, round-capped stroke outlines;
* nested contours are reoriented so the filled region is always on their left.

The GPU can then integrate each boundary as a signed fan of clipped triangles.
That construction is linear in oriented boundaries, so concavity and holes do
not require triangulation or a sample mask.  Any projection or topology which
cannot be certified here is deliberately marked for primary-ray fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# These values mirror the renderer's public diagnostic bitset.  They live here
# as well to keep this host-only geometry module independent of Taichi imports.
EXACT_REASON_SELF_OVERLAP = 1 << 2
EXACT_REASON_PROJECTION_FAILURE = 1 << 3
EXACT_REASON_COMPLEXITY_CAP = 1 << 4

_GEOM_EPS = 1e-7
_CONNECT_EPS = 2e-4
_MAX_EXACT_EDGES_PER_CIRCUIT = 4096


@dataclass
class ExactCircuitContours:
    """Packed, frame-varying projected boundaries for a circuit batch."""

    total_edges: torch.Tensor
    total_offsets: torch.Tensor
    fill_edges: torch.Tensor
    fill_offsets: torch.Tensor
    origins: torch.Tensor
    reasons: torch.Tensor


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area(points):
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1))
        - np.sum(points[:, 1] * np.roll(points[:, 0], -1))
    )


def _clean_points(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 2)
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > _GEOM_EPS:
            keep.append(i)
    out = points[keep]
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= _GEOM_EPS:
        out = out[:-1]
    return out


def _point_on_segment(p, a, b, eps=_CONNECT_EPS):
    ab = b - a
    ap = p - a
    scale = max(1.0, float(np.linalg.norm(ab)))
    if abs(_cross(ab, ap)) > eps * scale:
        return False
    dot = float(np.dot(ap, ab))
    return -eps <= dot <= float(np.dot(ab, ab)) + eps


def _point_in_polygon(point, polygon):
    """Even-odd containment; a boundary point is reported as indeterminate."""
    inside = False
    x, y = map(float, point)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if _point_on_segment(np.asarray(point), a, b):
            return None
        ay, by = float(a[1]), float(b[1])
        if (ay > y) != (by > y):
            hit_x = float(a[0]) + (y - ay) * float(b[0] - a[0]) / (by - ay)
            if hit_x > x:
                inside = not inside
    return inside


def _segment_relation(a, b, c, d):
    """Return whether two closed segments touch or cross."""
    ab = b - a
    cd = d - c
    scale = max(
        1.0,
        float(np.linalg.norm(ab)),
        float(np.linalg.norm(cd)),
    )
    eps = _CONNECT_EPS * scale
    o1 = _cross(ab, c - a)
    o2 = _cross(ab, d - a)
    o3 = _cross(cd, a - c)
    o4 = _cross(cd, b - c)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return any(
        (
            abs(o) <= eps and _point_on_segment(p, s0, s1, eps)
            for o, p, s0, s1 in (
                (o1, c, a, b),
                (o2, d, a, b),
                (o3, a, c, d),
                (o4, b, c, d),
            )
        )
    )


def _simple_polygon(points):
    n = len(points)
    if n < 3 or abs(_signed_area(points)) <= _GEOM_EPS:
        return False
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        if np.linalg.norm(b - a) <= _GEOM_EPS:
            return False
        for j in range(i + 1, n):
            # Adjacent edges share their authored endpoint and are allowed.
            if j == i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            c, d = points[j], points[(j + 1) % n]
            if (
                max(a[0], b[0]) + _CONNECT_EPS < min(c[0], d[0])
                or max(c[0], d[0]) + _CONNECT_EPS < min(a[0], b[0])
                or max(a[1], b[1]) + _CONNECT_EPS < min(c[1], d[1])
                or max(c[1], d[1]) + _CONNECT_EPS < min(a[1], b[1])
            ):
                continue
            if _segment_relation(a, b, c, d):
                return False
    return True


def _contours_intersect(first, second):
    for a, b in zip(first, np.roll(first, -1, axis=0)):
        for c, d in zip(second, np.roll(second, -1, axis=0)):
            if (
                max(a[0], b[0]) + _CONNECT_EPS < min(c[0], d[0])
                or max(c[0], d[0]) + _CONNECT_EPS < min(a[0], b[0])
                or max(a[1], b[1]) + _CONNECT_EPS < min(c[1], d[1])
                or max(c[1], d[1]) + _CONNECT_EPS < min(a[1], b[1])
            ):
                continue
            if _segment_relation(a, b, c, d):
                return True
    return False


def _interior_probe(points):
    """Find a point just inside one non-degenerate boundary edge."""
    orientation = 1.0 if _signed_area(points) > 0 else -1.0
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length <= _GEOM_EPS:
            continue
        left = np.asarray((-direction[1], direction[0])) / length
        midpoint = 0.5 * (a + b)
        for fraction in (1e-5, 1e-4, 1e-3):
            probe = midpoint + left * orientation * extent * fraction
            if _point_in_polygon(probe, points) is True:
                return probe
    return None


def _normalise_fill_contours(contours):
    """Orient even-depth contours CCW and odd-depth contours CW."""
    clean = [_clean_points(p) for p in contours]
    if not clean or any(not _simple_polygon(p) for p in clean):
        return None
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            if _contours_intersect(clean[i], clean[j]):
                return None
    probes = [_interior_probe(p) for p in clean]
    if any(p is None for p in probes):
        return None
    result = []
    for i, (polygon, probe) in enumerate(zip(clean, probes)):
        depth = 0
        for j, other in enumerate(clean):
            if i == j:
                continue
            contained = _point_in_polygon(probe, other)
            if contained is None:
                return None
            depth += int(contained)
        desired = 1.0 if depth % 2 == 0 else -1.0
        if _signed_area(polygon) * desired < 0.0:
            polygon = polygon[::-1].copy()
        result.append(polygon)
    return result


def _line_intersection(point_a, direction_a, point_b, direction_b):
    denominator = _cross(direction_a, direction_b)
    if abs(denominator) <= 1e-10:
        return 0.5 * (point_a + point_b)
    amount = _cross(point_b - point_a, direction_b) / denominator
    return point_a + amount * direction_a


def _arc_points(center, start_vector, turn, chord_tolerance, include_start=True):
    radius = float(np.linalg.norm(start_vector))
    if radius <= _GEOM_EPS or abs(turn) <= 1e-10:
        return [center + start_vector] if include_start else []
    tolerance = max(1e-4, min(float(chord_tolerance), radius * 0.5))
    cosine = max(-1.0, min(1.0, 1.0 - tolerance / radius))
    max_step = max(1e-3, 2.0 * math.acos(cosine))
    pieces = max(1, int(math.ceil(abs(turn) / max_step)))
    angle = math.atan2(float(start_vector[1]), float(start_vector[0]))
    first = 0 if include_start else 1
    return [
        center
        + radius
        * np.asarray(
            (
                math.cos(angle + turn * k / pieces),
                math.sin(angle + turn * k / pieces),
            )
        )
        for k in range(first, pieces + 1)
    ]


def _offset_closed(points, distance, chord_tolerance):
    points = _clean_points(points)
    if len(points) < 3:
        return None
    if abs(distance) <= _GEOM_EPS:
        return points.copy()
    out = []
    n = len(points)
    for i in range(n):
        previous = points[(i - 1) % n]
        current = points[i]
        following = points[(i + 1) % n]
        incoming = current - previous
        outgoing = following - current
        li = float(np.linalg.norm(incoming))
        lo = float(np.linalg.norm(outgoing))
        if li <= _GEOM_EPS or lo <= _GEOM_EPS:
            return None
        incoming /= li
        outgoing /= lo
        right_in = np.asarray((incoming[1], -incoming[0]))
        right_out = np.asarray((outgoing[1], -outgoing[0]))
        q_in = current + distance * right_in
        q_out = current + distance * right_out
        turn = math.atan2(_cross(incoming, outgoing), float(np.dot(incoming, outgoing)))
        if turn * distance > 1e-10:
            out.extend(
                _arc_points(
                    current,
                    distance * right_in,
                    turn,
                    chord_tolerance,
                    include_start=True,
                )[:-1]
            )
            out.append(q_out)
        else:
            point = _line_intersection(q_in, incoming, q_out, outgoing)
            if not np.isfinite(point).all():
                return None
            # Nearly reversing segments produce an unbounded miter and an
            # ambiguous offset topology.  Rays are the safe answer there.
            if np.linalg.norm(point - current) > max(8.0 * abs(distance), 4.0):
                return None
            out.append(point)
    out = _clean_points(np.asarray(out))
    if not _simple_polygon(out):
        return None
    if _signed_area(out) * _signed_area(points) <= 0.0:
        return None
    return out


def _offset_open_side(points, distance, chord_tolerance):
    points = _clean_points(points)
    if len(points) < 2:
        return None
    directions = np.diff(points, axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= _GEOM_EPS):
        return None
    directions /= lengths[:, None]
    right = np.stack((directions[:, 1], -directions[:, 0]), axis=-1)
    out = [points[0] + distance * right[0]]
    for i in range(1, len(points) - 1):
        incoming, outgoing = directions[i - 1], directions[i]
        q_in = points[i] + distance * right[i - 1]
        q_out = points[i] + distance * right[i]
        turn = math.atan2(_cross(incoming, outgoing), float(np.dot(incoming, outgoing)))
        if turn * distance > 1e-10:
            out.extend(
                _arc_points(
                    points[i],
                    distance * right[i - 1],
                    turn,
                    chord_tolerance,
                    include_start=False,
                )
            )
        else:
            point = _line_intersection(q_in, incoming, q_out, outgoing)
            if not np.isfinite(point).all():
                return None
            if np.linalg.norm(point - points[i]) > max(8.0 * abs(distance), 4.0):
                return None
            out.append(point)
    out.append(points[-1] + distance * right[-1])
    return _clean_points(np.asarray(out))


def _stroke_outline(points, closed, half_width, chord_tolerance):
    if half_width <= _GEOM_EPS:
        return []
    points = _clean_points(points)
    if closed:
        if not _simple_polygon(points):
            return None
        if _signed_area(points) < 0.0:
            points = points[::-1].copy()
        outer = _offset_closed(points, half_width, chord_tolerance)
        inner = _offset_closed(points, -half_width, chord_tolerance)
        if outer is None or inner is None or _contours_intersect(outer, inner):
            return None
        return [outer, inner[::-1].copy()]

    if len(points) < 2:
        return None
    right = _offset_open_side(points, half_width, chord_tolerance)
    left = _offset_open_side(points, -half_width, chord_tolerance)
    if right is None or left is None:
        return None
    end_direction = points[-1] - points[-2]
    end_direction /= np.linalg.norm(end_direction)
    end_right = half_width * np.asarray((end_direction[1], -end_direction[0]))
    start_direction = points[1] - points[0]
    start_direction /= np.linalg.norm(start_direction)
    start_left = -half_width * np.asarray((start_direction[1], -start_direction[0]))
    outline = list(right)
    outline.extend(
        _arc_points(
            points[-1], end_right, math.pi, chord_tolerance, include_start=False
        )
    )
    outline.extend(left[-2::-1])
    outline.extend(
        _arc_points(
            points[0], start_left, math.pi, chord_tolerance, include_start=False
        )
    )
    outline = _clean_points(np.asarray(outline))
    if not _simple_polygon(outline):
        return None
    if _signed_area(outline) < 0.0:
        outline = outline[::-1].copy()
    return [outline]


def _split_edge_contours(edges):
    """Split one circuit's ordered edge rows at discontinuities."""
    count = len(edges)
    if count == 0:
        return []
    breaks = []
    for i in range(count):
        following = (i + 1) % count
        if np.linalg.norm(edges[i, 2:4] - edges[following, 0:2]) > _CONNECT_EPS:
            breaks.append(i)
    if not breaks:
        return [edges]
    groups = []
    for j, end in enumerate(breaks):
        start = (breaks[j - 1] + 1) % count
        if start <= end:
            inds = np.arange(start, end + 1)
        else:
            inds = np.concatenate((np.arange(start, count), np.arange(0, end + 1)))
        groups.append(edges[inds])
    return groups


def _edges_from_contours(contours):
    if not contours:
        return np.empty((0, 4), dtype=np.float64)
    return np.concatenate(
        [np.concatenate((p, np.roll(p, -1, axis=0)), axis=1) for p in contours],
        axis=0,
    )


def _build_one_circuit(
    edge_rows,
    filled,
    border_width,
    outline_width,
    chord_tolerance,
):
    """Return ``(total_edges, fill_edges, reason)`` for one frame/circuit."""
    if len(edge_rows) == 0 or not np.isfinite(edge_rows).all():
        return None, None, EXACT_REASON_PROJECTION_FAILURE
    if len(edge_rows) > _MAX_EXACT_EDGES_PER_CIRCUIT:
        return None, None, EXACT_REASON_COMPLEXITY_CAP
    groups = _split_edge_contours(edge_rows)
    if not groups:
        return None, None, EXACT_REASON_PROJECTION_FAILURE

    if filled:
        contours = []
        for group in groups:
            if np.linalg.norm(group[-1, 2:4] - group[0, 0:2]) > _CONNECT_EPS:
                return None, None, EXACT_REASON_PROJECTION_FAILURE
            points = _clean_points(group[:, 0:2])
            if len(points) < 3:
                edge_lengths = np.linalg.norm(group[:, 2:4] - group[:, 0:2], axis=1)
                # Font outlines commonly carry a one-row, invisible zero-area
                # separator between subpaths.  It is neither fill boundary nor
                # border geometry and can be discarded exactly.
                if np.all(edge_lengths <= _CONNECT_EPS) and np.all(group[:, 4] < 0.5):
                    continue
                return None, None, EXACT_REASON_PROJECTION_FAILURE
            # A synthetic closure is valid for fill, but a visible border on
            # that open contour cannot be represented by a whole-contour
            # erosion.  Send just that uncommon case to rays.
            if border_width > _GEOM_EPS and np.any(group[:, 4] < 0.5):
                return None, None, EXACT_REASON_PROJECTION_FAILURE
            contours.append(points)
        if not contours:
            return (
                np.empty((0, 4), dtype=np.float64),
                np.empty((0, 4), dtype=np.float64),
                0,
            )
        contours = _normalise_fill_contours(contours)
        if contours is None:
            return None, None, EXACT_REASON_SELF_OVERLAP
        total = [_offset_closed(p, outline_width, chord_tolerance) for p in contours]
        if any(p is None for p in total):
            return None, None, EXACT_REASON_SELF_OVERLAP
        total = _normalise_fill_contours(total)
        if total is None:
            return None, None, EXACT_REASON_SELF_OVERLAP
        if border_width <= _GEOM_EPS:
            fill_only = [p.copy() for p in total]
        else:
            fill_only = [
                _offset_closed(p, -border_width, chord_tolerance) for p in contours
            ]
            if any(p is None for p in fill_only):
                return None, None, EXACT_REASON_SELF_OVERLAP
            fill_only = _normalise_fill_contours(fill_only)
            if fill_only is None:
                return None, None, EXACT_REASON_SELF_OVERLAP
        return _edges_from_contours(total), _edges_from_contours(fill_only), 0

    stroke_regions = []
    for group in groups:
        edge_lengths = np.linalg.norm(group[:, 2:4] - group[:, 0:2], axis=1)
        if np.all(edge_lengths <= _CONNECT_EPS) and np.all(group[:, 4] < 0.5):
            continue
        invisible = np.flatnonzero(group[:, 4] < 0.5)
        if len(invisible) > 1:
            return None, None, EXACT_REASON_PROJECTION_FAILURE
        if len(invisible) == 1:
            closure = int(invisible[0])
            group = np.concatenate((group[closure + 1 :], group[: closure + 1]))
            if int(group[-1, 4] >= 0.5) != 0:
                return None, None, EXACT_REASON_PROJECTION_FAILURE
            points = _clean_points(group[:, 0:2])
            closed = False
        else:
            points = _clean_points(group[:, 0:2])
            closed = True
        outlines = _stroke_outline(points, closed, 0.5 * border_width, chord_tolerance)
        if outlines is None:
            return None, None, EXACT_REASON_SELF_OVERLAP
        stroke_regions.extend(outlines)

    # Stroke subpaths are independent positive regions.  Containment or an
    # intersection means their union is not the sum of their scalar areas.
    positives = [p for p in stroke_regions if _signed_area(p) > 0.0]
    for i in range(len(positives)):
        for j in range(i + 1, len(positives)):
            if _contours_intersect(positives[i], positives[j]):
                return None, None, EXACT_REASON_SELF_OVERLAP
            probe = _interior_probe(positives[i])
            other_probe = _interior_probe(positives[j])
            if probe is None or other_probe is None:
                return None, None, EXACT_REASON_PROJECTION_FAILURE
            if _point_in_polygon(probe, positives[j]) or _point_in_polygon(
                other_probe, positives[i]
            ):
                return None, None, EXACT_REASON_SELF_OVERLAP
    total_edges = _edges_from_contours(stroke_regions)
    return total_edges, np.empty((0, 4), dtype=np.float64), 0


def build_exact_circuit_contours(
    edges_2d,
    edge_offsets,
    edge_circuit,
    circuit_meta,
    cam_origin,
    screen_point,
    screen_basis,
    screen_width,
    screen_height,
    outline_width,
    chord_tolerance,
):
    """Project flattened circuits and construct exact oriented boundaries.

    All returned tensors live on ``edges_2d.device``.  Counts are padded per
    circuit to the maximum required by any frame, keeping offsets static while
    animation changes projection and join angles.
    """
    device = edges_2d.device
    frame_count = max(
        int(edges_2d.shape[0]),
        int(circuit_meta.shape[0]),
        int(cam_origin.shape[0]),
        int(screen_point.shape[0]),
        int(screen_basis.shape[0]),
    )
    frame_ids = torch.arange(frame_count, device=device)
    edges = edges_2d.index_select(0, frame_ids % edges_2d.shape[0])
    meta = circuit_meta.index_select(0, frame_ids % circuit_meta.shape[0])
    ro = cam_origin.index_select(0, frame_ids % cam_origin.shape[0])
    sp = screen_point.index_select(0, frame_ids % screen_point.shape[0])
    basis = screen_basis.index_select(0, frame_ids % screen_basis.shape[0])

    edge_circuit = edge_circuit.long()
    center = meta[:, edge_circuit, 0:3]
    basis_u = meta[:, edge_circuit, 6:9]
    basis_v = meta[:, edge_circuit, 9:12]
    local = edges[..., :4].reshape(frame_count, -1, 2, 2)
    world = (
        center.unsqueeze(2)
        + local[..., :1] * basis_u.unsqueeze(2)
        + local[..., 1:] * basis_v.unsqueeze(2)
    )

    # Match Camera.get_render_screen_basis exactly.  Its rows are generally
    # non-orthogonal after a rotated, non-uniform screen scale: projection is
    # defined by the row-2 plane normal followed by raw dot products against
    # rows 0/1.  Treating rows 0/1 as a Euclidean plane basis (and solving for
    # their coefficients) only works for an unrotated camera and displaces
    # tilted circuits into unrelated pixels.
    normal = basis[:, 2]
    direction = world - ro[:, None, None]
    plane_depth = (direction * normal[:, None, None]).sum(-1)
    screen_depth = ((sp - ro) * normal).sum(-1)
    safe = torch.where(
        plane_depth.abs() > 1e-12, plane_depth, torch.ones_like(plane_depth)
    )
    amount = screen_depth[:, None, None] / safe
    hit = ro[:, None, None] + amount.unsqueeze(-1) * direction
    relative = hit - sp[:, None, None]
    u = (relative * basis[:, None, None, 0]).sum(-1)
    v = (relative * basis[:, None, None, 1]).sum(-1)
    projected = torch.stack(
        (
            u * (float(screen_height) / 2.0) + float(screen_width) / 2.0,
            v * (float(screen_height) / 2.0) + float(screen_height) / 2.0,
        ),
        dim=-1,
    )
    front = (
        (plane_depth.abs() > 1e-12) & (amount > 0.0) & torch.isfinite(projected).all(-1)
    )

    projected_cpu = projected.detach().cpu().double().numpy()
    visible_cpu = edges[..., 4].detach().cpu().double().numpy()
    front_cpu = front.detach().cpu().numpy()
    meta_cpu = meta.detach().cpu().double().numpy()
    offsets_cpu = edge_offsets.detach().cpu().long().numpy()
    circuits = int(circuit_meta.shape[1])

    totals = [[None for _ in range(circuits)] for _ in range(frame_count)]
    fills = [[None for _ in range(circuits)] for _ in range(frame_count)]
    reasons = np.zeros((frame_count, circuits), dtype=np.int32)
    max_total = np.zeros((circuits,), dtype=np.int64)
    max_fill = np.zeros((circuits,), dtype=np.int64)

    for frame in range(frame_count):
        for circuit in range(circuits):
            start, end = int(offsets_cpu[circuit]), int(offsets_cpu[circuit + 1])
            if end <= start or not bool(front_cpu[frame, start:end].all()):
                reasons[frame, circuit] = EXACT_REASON_PROJECTION_FAILURE
                continue
            source = projected_cpu[frame, start:end]
            rows = np.concatenate(
                (
                    source.reshape(end - start, 4),
                    visible_cpu[frame, start:end, None],
                ),
                axis=1,
            )
            filled = bool(meta_cpu[frame, circuit, 13] > 0.5)
            border_width = abs(float(meta_cpu[frame, circuit, 12]))
            total, fill, reason = _build_one_circuit(
                rows,
                filled,
                border_width,
                float(outline_width),
                float(chord_tolerance),
            )
            if reason:
                reasons[frame, circuit] = reason
                continue
            if len(total) > _MAX_EXACT_EDGES_PER_CIRCUIT or len(fill) > (
                _MAX_EXACT_EDGES_PER_CIRCUIT
            ):
                reasons[frame, circuit] = EXACT_REASON_COMPLEXITY_CAP
                continue
            totals[frame][circuit] = total
            fills[frame][circuit] = fill
            max_total[circuit] = max(max_total[circuit], len(total))
            max_fill[circuit] = max(max_fill[circuit], len(fill))

    total_offsets = np.zeros((circuits + 1,), dtype=np.int32)
    fill_offsets = np.zeros((circuits + 1,), dtype=np.int32)
    total_offsets[1:] = np.cumsum(max_total, dtype=np.int64).astype(np.int32)
    fill_offsets[1:] = np.cumsum(max_fill, dtype=np.int64).astype(np.int32)
    total_packed = np.zeros((frame_count, int(total_offsets[-1]), 4), np.float32)
    fill_packed = np.zeros((frame_count, int(fill_offsets[-1]), 4), np.float32)
    origins = np.zeros((frame_count, circuits, 2), np.float32)
    for frame in range(frame_count):
        for circuit in range(circuits):
            total = totals[frame][circuit]
            fill = fills[frame][circuit]
            if total is None:
                continue
            ts = int(total_offsets[circuit])
            fs = int(fill_offsets[circuit])
            total_packed[frame, ts : ts + len(total)] = total.astype(np.float32)
            if len(fill):
                fill_packed[frame, fs : fs + len(fill)] = fill.astype(np.float32)
            origins[frame, circuit] = total[:, :2].mean(axis=0).astype(np.float32)

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=device).contiguous()

    return ExactCircuitContours(
        total_edges=tensor(total_packed),
        total_offsets=tensor(total_offsets, torch.int32),
        fill_edges=tensor(fill_packed),
        fill_offsets=tensor(fill_offsets, torch.int32),
        origins=tensor(origins),
        reasons=tensor(reasons, torch.int32),
    )
