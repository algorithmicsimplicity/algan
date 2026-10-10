from __future__ import annotations

import torch


def _flat_frames(x, last_dims):
    """Collapse camera tensors like [T, 1, 1, 3] to [T, *last_dims]."""
    return x.reshape(x.shape[0], *last_dims).float()


def _expand_frames(x, num_frames):
    if x.shape[0] == num_frames:
        return x
    return x.expand(num_frames, *x.shape[1:])


def _pixel_bases(screen_basis):
    """Per-frame world-space steps corresponding to one unit of normalized
    screen coordinate, matching the camera's projection exactly.

    The camera projects a world point by intersecting its view ray with the
    plane ``normal = basis_row_2`` through the screen center, then taking raw
    dot products with ``basis_row_0/1``. The screen basis is rotation x
    non-uniform scale, so under camera rotation its rows are *not* mutually
    orthogonal (``row0 . row2 != 0``) -- the projection is anisotropic and
    changes with orientation. The exact inverse image of screen coordinate
    (u, v) is ``screen_point + u * d0 + v * d1`` where ``d0, d1`` are the
    first two columns of the inverse basis matrix (the reciprocal basis:
    ``d_i . row_j = delta_ij``), which both lies on the projection plane and
    reproduces the dot products.
    """
    eye = torch.eye(3, device=screen_basis.device).unsqueeze(0) * 1e-12
    dual = torch.linalg.inv(screen_basis + eye)
    return dual[:, :, 0].contiguous(), dual[:, :, 1].contiguous()


#: The two circuit-topology tolerances below are world-unit sizes, right for a
#: circuit at least this large (its control points' box diagonal, in world
#: units) and scaled down in proportion for a smaller one. Fixed sizes broke
#: a HUD -- text shrunk ~1/107 to sit a unit in front of a distant camera, its
#: glyphs a few thousandths of a unit across: 23 real cubics of
#: ``E_new - E_old`` fell under the degenerate size and were dropped, and each
#: gap they left flipped the even-odd parity into a one-pixel dashed streak. A
#: tenth of a unit from the eye the gap size also read a letter's jump to its
#: hole as a continuation and drew a seam through it.
#:
#: Relative to the CIRCUIT, not to each cubic, because animation leaves cubics
#: that are nothing: a partial draw (``Create``, ``Write``,
#: ``ShowPassingFlash``) collapses the cubics outside its window onto their
#: own end points and leaves the one at the window's edge rounding-small, and
#: those must stay dropped -- kept, each draws a stroke-wide dot. A collapsed
#: cubic still sits on the shape's outline, so the circuit keeps its full size
#: while it is drawn. Above the reference size the tolerances are exactly the
#: absolute ones the renderer always used, and glyphs and shapes at ordinary
#: sizes (a tenth of a unit and up) render as they did.
CIRCUIT_REFERENCE_SIZE = 0.1

#: A cubic whose control legs' squared lengths sum below this, times the square
#: of its circuit's :func:`_circuit_scale`, is a point and draws nothing.
DEGENERATE_CUBIC_LEGS_SQUARED = 1e-9

#: Consecutive cubics are one contour when the gap from the first one's end to
#: the next one's start is at most this, times the circuit's scale. Real joins
#: are exact copies of one point and real jumps (a letter's outline to its hole)
#: a visible fraction of the shape, so the band between them is wide.
CONNECTION_GAP = 1e-5


def _circuit_scale(corners, circuit_of_segment=None):
    """Each cubic's tolerance scale: its circuit's size over the reference, at most 1.

    ``corners`` is ``[..., S, 4, 3]``; ``circuit_of_segment`` (``[S]``, long) names
    each cubic's circuit, ``None`` meaning they form one. The size is the
    diagonal of the circuit's control-point box in each frame. Returns
    ``[..., S]``. See :data:`CIRCUIT_REFERENCE_SIZE`.
    """
    lo = corners.amin(-2)  # [..., S, 3]
    hi = corners.amax(-2)
    if circuit_of_segment is None:
        size = (hi.amax(-2) - lo.amin(-2)).norm(p=2, dim=-1, keepdim=True)
        size = size.expand(lo.shape[:-1])
    else:
        circuit_of_segment = circuit_of_segment.to(corners.device)
        num_circuits = int(circuit_of_segment.max()) + 1 if lo.shape[-2] else 0
        index = circuit_of_segment.view(-1, 1).expand(lo.shape)
        boxes = lo.shape[:-2] + (num_circuits, 3)
        circuit_lo = lo.new_full(boxes, float("inf")).scatter_reduce(
            -2, index, lo, "amin", include_self=True
        )
        circuit_hi = hi.new_full(boxes, float("-inf")).scatter_reduce(
            -2, index, hi, "amax", include_self=True
        )
        size = (circuit_hi - circuit_lo).norm(p=2, dim=-1)[..., circuit_of_segment]
    return (size / CIRCUIT_REFERENCE_SIZE).clamp(max=1.0)


def _degenerate_cubics(corners, scale):
    """Whether each cubic of ``corners`` (``[..., S, 4, 3]``) is a point.

    ``scale`` is :func:`_circuit_scale`; see :data:`DEGENERATE_CUBIC_LEGS_SQUARED`.
    A cubic of no length is a point whatever its circuit's size -- a circuit
    collapsed whole (scaled to nothing, or an empty stroke region) has scale 0.
    Returns ``[..., S]``.
    """
    legs = (corners[..., 1:, :] - corners[..., :-1, :]).square().sum(-1).sum(-1)
    return (legs < DEGENERATE_CUBIC_LEGS_SQUARED * scale.square()) | (legs == 0)


def _cubics_disconnected(gap, scale):
    """Whether one cubic's end and the next one's start are different points.

    ``gap`` is ``end - start`` (``[..., 3]``) and ``scale`` the circuit's
    :func:`_circuit_scale`, broadcastable against ``gap`` with its last
    dimension reduced to 1, which is the result's shape. Every place that splits
    a circuit into sub-paths uses this one rule (see :data:`CONNECTION_GAP`), so
    the mob, the non-planar classifier and the renderer agree on what a sub-path
    is.
    """
    return gap.norm(p=2, dim=-1, keepdim=True) > CONNECTION_GAP * scale


def _unify_time(tensors, error_context):
    """Expand a set of tensors whose leading (time) dims are each 1 or T to a
    common T. Returns the expanded tensors and T.
    """
    T = max(t.shape[0] for t in tensors)
    for t in tensors:
        if t.shape[0] not in (1, T):
            raise ValueError(
                f"{error_context}: incompatible frame counts "
                f"{[tuple(t.shape) for t in tensors]}"
            )
    return [_expand_frames(t, T) for t in tensors], T


def _cat_collections(tensors, dim, error_context):
    """Concatenate per-collection tensors along ``dim``, broadcasting their
    (possibly different) time dimensions to a common length first. A single
    collection is passed through without copying (the kernel indexes each
    array's time dimension independently, so no expansion is needed).
    """
    if len(tensors) == 1:
        return tensors[0]
    tensors, _ = _unify_time(tensors, error_context)
    return torch.cat(tensors, dim).contiguous()


def _cat_mat_blocks(blocks, error_context):
    """Concatenate per-collection parameter blocks ``[Tm, N, W]`` along the
    primitive axis, right-zero-padding narrower blocks to the widest ``W`` first.

    Built-in materials pack a 12-slot block while custom fragment pipelines pack
    a wider one; padding lets them share a single per-scene array. The padding
    slots are never read (each stage reads only its own slice), so a built-in
    (or built-in-only) scene is unaffected -- with no wide blocks present ``W``
    stays 12 and no padding happens.
    """
    if len(blocks) == 1:
        return blocks[0]
    max_w = max(b.shape[-1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[-1] < max_w:
            pad = torch.zeros(
                (*b.shape[:-1], max_w - b.shape[-1]), dtype=b.dtype, device=b.device
            )
            b = torch.cat([b, pad], dim=-1)
        padded.append(b)
    return _cat_collections(padded, 1, error_context)
