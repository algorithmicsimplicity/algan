"""Bounded, render-local reuse of unchanged circuit polylines."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import torch


class _BezierGeometryCache:
    # Outside the arena: retain at most 16 MiB of keys and device geometry,
    # regardless of scene length. Pool-headroom checks see these live tensors.
    def __init__(self, max_bytes=16 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._entries = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            tensors, event, _ = entry
            if event is not None:
                stream = torch.cuda.current_stream(tensors[0].device)
                stream.wait_event(event)
                for tensor in tensors:
                    tensor.record_stream(stream)
            return tensors

    def put(self, key, tensors):
        size = len(key[-1]) + sum(t.untyped_storage().nbytes() for t in tensors)
        if size > self.max_bytes:
            return
        event = None
        if tensors[0].is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensors[0].device))
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.bytes -= old[2]
            while self._entries and (
                self.bytes + size > self.max_bytes or len(self._entries) >= 32
            ):
                self.bytes -= self._entries.popitem(last=False)[1][2]
            self._entries[key] = (tensors, event, size)
            self.bytes += size

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.bytes = 0


def _constant_rows(tensor):
    # Equality is exact. NaNs remain on the ordinary rebuild path.
    if tensor.shape[0] == 1:
        return torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
    return (
        (tensor[1:] == tensor[:1])
        .reshape(tensor.shape[0] - 1, tensor.shape[1], -1)
        .all(2)
        .all(0)
    )


def _rigid_segments(corners, centers, circuit_ids, frame_chunk=8):
    """Segments whose controls relative to their circuit's center stay put.

    Compared to frame 0 within 16 float epsilons of the coordinate scale,
    which absorbs the rounding of a translation. Streamed over frame chunks,
    so the temporaries are a few frames' worth of controls rather than the
    batch's. A NaN compares false and stays on the rebuild path.
    """
    num_frames = max(corners.shape[0], centers.shape[0])
    rigid = torch.ones(corners.shape[1], dtype=torch.bool, device=corners.device)
    if num_frames == 1:
        return rigid
    low, high = torch.aminmax(corners)
    scale = torch.stack((-low, high, centers.abs().amax())).amax().clamp_min(1.0)
    tolerance = 16 * torch.finfo(corners.dtype).eps * scale

    def relative(start, end):
        c = corners[start:end] if corners.shape[0] > 1 else corners
        m = centers[start:end] if centers.shape[0] > 1 else centers
        return c - m[:, circuit_ids].unsqueeze(-2)

    reference = relative(0, 1)
    for start in range(1, num_frames, frame_chunk):
        deviation = (relative(start, start + frame_chunk) - reference).abs()
        rigid &= (
            (deviation <= tolerance)
            .reshape(deviation.shape[0], deviation.shape[1], -1)
            .all(2)
            .all(0)
        )
    return rigid


def _cached_static_edges(cache, build, args, inward_signs):
    # The actual pre-bias geometry, plane frame, topology and chosen chord
    # counts are the key. Camera/resolution/tolerance changes still run the
    # criterion; reuse is valid only if its resulting geometry is identical.
    # One packed transfer avoids a device synchronization per source array.
    key_bytes = sum(t.numel() * t.element_size() for t in args)
    key = None
    if cache is not None and key_bytes <= cache.max_bytes:
        layouts = tuple((tuple(t.shape), t.dtype, t.device) for t in args)
        data = torch.cat(
            [t.detach().contiguous().reshape(-1).view(torch.uint8) for t in args]
        )
        key = (inward_signs, layouts, data.cpu().numpy().tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
    result = build(*args, inward_signs)
    if key is not None:
        cache.put(key, result)
    return result


def _build_cached_circuit_edges(scene, build, args, inward_signs):
    """Reuse static circuits even when the collection also contains animation.

    Edges are plane coordinates relative to each circuit's center, so a
    circuit whose shape and plane do not change has the same edges in every
    frame, whether it is still or moving rigidly -- a line of text scrolling
    up a terminal, or a label in a group that slides across the frame. Such a
    circuit is built once, from the batch's first frame, and the caller places
    it at each frame's center. For a still circuit that is exact. For a
    translated one the control points relative to the center agree between
    frames only up to the rounding of the translation, so they are compared to
    within 16 float epsilons of the coordinate scale, and the reused edges
    differ from per-frame ones by that rounding alone. Rotation, scaling and
    any change of shape rebuild the circuit every frame.

    Only edge geometry is retained. Materials, texture transforms and frame
    bounds are rebuilt from the current batch by the caller. The result keeps
    the original circuit/edge order and supports a singleton frame dimension.
    """
    if any(t.requires_grad for t in args):
        return build(*args, inward_signs)
    corners, samples, segments, next_inds, centers, basis_u, basis_v = args
    device = corners.device
    circuit_ids = torch.repeat_interleave(
        torch.arange(len(segments), device=device), segments
    )
    static = _constant_rows(basis_u) & _constant_rows(basis_v)
    segment_static = _rigid_segments(corners, centers, circuit_ids) & _constant_rows(
        next_inds
    )
    static = (
        static.int()
        .scatter_reduce_(0, circuit_ids, segment_static.int(), "amin")
        .bool()
    )
    if not bool(static.any()):
        return build(*args, inward_signs)

    cache = None
    if scene is not None:
        cache = getattr(scene, "_bezier_geometry_cache", None)
        if cache is None:
            cache = scene.__dict__.setdefault(
                "_bezier_geometry_cache", _BezierGeometryCache()
            )
    if bool(static.all()):
        single = (
            corners[:1],
            samples,
            segments,
            next_inds[:1],
            centers[:1],
            basis_u[:1],
            basis_v[:1],
        )
        return _cached_static_edges(cache, build, single, inward_signs)

    # Ordinary circuits connect only within themselves. Keep the original path
    # for an exotic cross-circuit link that would cross this partition.
    segment_static = static[circuit_ids]
    if not bool((segment_static[next_inds] == segment_static.unsqueeze(0)).all()):
        return build(*args, inward_signs)

    parts = []
    counts = torch.empty_like(segments)
    for is_static in (True, False):
        selected = static if is_static else ~static
        circuits = selected.nonzero().flatten()
        indices = selected[circuit_ids].nonzero().flatten()
        remap = torch.empty(corners.shape[1], dtype=torch.long, device=device)
        remap[indices] = torch.arange(len(indices), device=device)
        time = slice(0, 1) if is_static else slice(None)
        subset = (
            corners[time][:, indices],
            samples[indices],
            segments[circuits],
            remap[next_inds[time][:, indices]],
            centers[time][:, circuits],
            basis_u[time][:, circuits],
            basis_v[time][:, circuits],
        )
        edges, offsets = (
            _cached_static_edges(cache, build, subset, inward_signs)
            if is_static
            else build(*subset, inward_signs)
        )
        offsets = offsets.long()
        counts[circuits] = offsets[1:] - offsets[:-1]
        parts.append((circuits, edges, offsets))

    offsets = torch.cat((segments.new_zeros(1), counts.cumsum(0)))
    num_frames = max(edges.shape[0] for _, edges, _ in parts)
    result = corners.new_empty((num_frames, int(offsets[-1]), 6))
    for circuits, edges, part_offsets in parts:
        destinations = torch.repeat_interleave(
            offsets[circuits] - part_offsets[:-1], part_offsets[1:] - part_offsets[:-1]
        ) + torch.arange(edges.shape[1], device=device)
        result[:, destinations] = edges
    return result, offsets.to(torch.int32)


#: The row ``_sample_circuit_edges`` gives a degenerate cubic's edges, which
#: every bound, index and kernel skips.
_SENTINEL_EDGE = (1e9, 1e9, 1e9, 1e9, 0.0, 0.0)


def _padding_edges(edges, offsets):
    """One inert edge per circuit, to fill the slots it leaves unused.

    A zero-length edge on the circuit's first real vertex, hidden from the
    border: the crossing index skips a horizontal edge and the border index an
    invisible one, the circuit bounds already hold that vertex, and the row
    scans that walk every edge of a circuit (``sheet_compact_taichi``) meet a
    point some real edge of it starts at. A circuit with no real edge gets the
    degenerate-cubic sentinel it already holds.
    """
    counts = offsets[1:] - offsets[:-1]
    num_circuits, num_edges = counts.numel(), edges.shape[0]
    sentinel = edges.new_tensor(_SENTINEL_EDGE).expand(num_circuits, -1)
    if num_edges == 0:
        return sentinel.clone()
    real = (edges[:, :4].abs() < 1e8).all(-1)
    circuit = torch.repeat_interleave(
        torch.arange(num_circuits, device=edges.device), counts
    )
    position = torch.arange(num_edges, device=edges.device)
    first = torch.full_like(counts, num_edges).scatter_reduce_(
        0, circuit[real], position[real], "amin", include_self=True
    )
    start = edges[first.clamp(max=num_edges - 1), :2]
    padding = torch.cat((start, start, torch.zeros_like(start)), -1)
    return torch.where((first < num_edges).unsqueeze(-1), padding, sentinel)


def _build_frame_local_circuit_edges(scene, build, args, inward_signs, cache=True):
    """Every frame's circuit edges exactly as that frame would build them alone.

    For stills that share a render batch (see ``RenderLoopMixin.get_frames``).
    ``args`` are ``_build_cached_circuit_edges``'s, except that the chord
    counts are ``[T, S]``: each frame's own. Nothing is reused between frames
    within a tolerance, and nothing is shared that a frame alone would choose
    differently:

    - geometry exactly equal in every frame (controls, plane, topology and
      chord counts alike) is built once, through the single-frame path a still
      rendered alone takes (its cache included);
    - otherwise, where every frame takes the same chord counts and closing
      vertices -- all a batched build shares between frames -- one batched
      build is each frame's own;
    - otherwise every frame is built alone, and frames whose circuits come out
      with different edge counts are packed to the widest, each circuit's
      spare slots holding inert edges (:func:`_padding_edges`), so a frame's
      real edges -- the ones every query reads, in their own order -- are its
      alone ones.
    """
    corners, samples, segments, next_inds, centers, basis_u, basis_v = args
    num_frames = samples.shape[0]

    def frame(index):
        def at(tensor):
            return tensor if tensor.shape[0] == 1 else tensor[index : index + 1]

        return (
            at(corners),
            samples[index],
            segments,
            at(next_inds),
            at(centers),
            at(basis_u),
            at(basis_v),
        )

    if samples.numel() == 0 or all(
        bool(_constant_rows(tensor).all())
        for tensor in (corners, samples, next_inds, centers, basis_u, basis_v)
    ):
        if cache:
            return _build_cached_circuit_edges(scene, build, frame(0), inward_signs)
        return build(*frame(0), inward_signs)

    if bool(_constant_rows(samples).all()):
        from algan.rendering.raytracing.primitives import (
            _bezier_connection_visibility,
        )

        circuit_ids = torch.repeat_interleave(
            torch.arange(len(segments), device=corners.device), segments
        )
        closing = ~_bezier_connection_visibility(corners, next_inds, circuit_ids)
        if bool(_constant_rows(closing).all()):
            # The rest of a batched build is per-frame arithmetic, element for
            # element what one frame computes alone. Built directly: the
            # cached path would reuse contours within a tolerance.
            return build(
                corners,
                samples[0],
                segments,
                next_inds,
                centers,
                basis_u,
                basis_v,
                inward_signs,
            )

    built = [build(*frame(index), inward_signs) for index in range(num_frames)]
    frame_offsets = torch.stack([offsets.long() for _, offsets in built])
    counts = frame_offsets[:, 1:] - frame_offsets[:, :-1]  # [T, C]
    widths = counts.amax(0)
    offsets = torch.cat((widths.new_zeros(1), widths.cumsum(0)))
    first_edges = built[0][0]
    result = first_edges.new_empty((num_frames, int(offsets[-1]), 6))
    slot_circuit = torch.repeat_interleave(
        torch.arange(widths.numel(), device=widths.device), widths
    )
    for index, (edges, own_offsets) in enumerate(built):
        edges = edges[0]
        own_offsets = own_offsets.long()
        result[index] = _padding_edges(edges, own_offsets)[slot_circuit]
        destinations = torch.repeat_interleave(
            offsets[:-1] - own_offsets[:-1], counts[index]
        ) + torch.arange(edges.shape[0], device=edges.device)
        result[index, destinations] = edges
    return result, offsets.to(torch.int32)
