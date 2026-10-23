"""Batched 3-D geometry primitives shared by the animation and render paths.

Every function here is vectorized over leading batch dimensions and operates on
torch tensors, because it is called once per frame batch on whole arrays of
points rather than per point.

The module covers rotations (building a rotation from an axis and angle, or
between two vectors or two bases), projection and intersection (point onto line,
segment or plane; line against plane), and basis changes between a Mob's local
frame and world space.

It used to carry polynomial root finding as well -- closed-form quadratic and
cubic solvers, a companion-matrix eigenvalue solver, and the recursive
lower-degree fallbacks around them. All of it was unreachable, and the cubic
had never worked: it raised on ordinary inputs, and the shapes it produced
could not reconcile with its own fallback. It is gone rather than repaired.
The ray tracer does its curve and surface intersection elsewhere, and always
did.

These are internal building blocks: user-facing spatial operations live on
:class:`~algan.animatable_base.mob.Mob`.
"""

from __future__ import annotations

import math
from string import ascii_lowercase

import torch
import torch.nn.functional as F

from algan.constants.math import DEGREES_TO_RADIANS, RADIANS_TO_DEGREES
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    broadcast_gather,
    dot_product,
    unsqueeze_left,
    unsquish,
)


def distance(x, y, memory=None, *args, **kwargs):
    """Return the Euclidean distance between ``x`` and ``y`` along the last dim.

    Parameters
    ----------
    x, y : torch.Tensor
        Points of shape ``(*, D)``; leading dims broadcast against each other.
    memory : optional
        Scratch-tensor allocator. When given, the result is written into a tensor
        from it instead of being computed with :func:`torch.cdist`.
    *args, **kwargs
        Passed to :func:`torch.cdist` (only used when ``memory`` is None).

    Returns
    -------
    torch.Tensor
        Distances of shape ``(*, 1)`` (the last dim is kept with size 1).
    """
    if memory is None:
        return torch.cdist(x.unsqueeze(-2), y.unsqueeze(-2), *args, **kwargs).squeeze(
            -2
        )
    out = memory.get_tensor([*x.shape[:-1], 1])
    with memory.temp():
        dif = torch.sub(x, y, out=memory.get_tensor(x.shape))
        dif.square_()
        dist = torch.sum(dif, -1, keepdim=True, out=out)
        dist = dist.sqrt_()
    return dist


def intersect_line_with_plane(
    line_direction, plane_point, plane_normal, line_point=0, dim=-1, memory=None
):
    """Intersect lines with planes.

    The line is ``line_point + t * line_direction``; the plane passes through
    ``plane_point`` with normal ``plane_normal`` (normalized internally, so its
    length does not matter). All arguments broadcast over leading dims, with
    the 3 coordinates on ``dim``.

    Parameters
    ----------
    line_direction : torch.Tensor
        Line directions. Need not be unit length; ``t`` is measured in units
        of this vector.
    plane_point, plane_normal : torch.Tensor
        A point on each plane and its normal.
    line_point : torch.Tensor or float, default 0
        A point on each line (the origin by default).
    dim : int, default -1
        The coordinate dimension. A non-negative ``dim`` is converted to a
        negative one relative to the highest-rank argument.
    memory : optional
        Scratch-tensor allocator for the in-place path. That path overwrites
        ``plane_normal`` and ``line_direction`` and returns ``t`` with the
        opposite sign to the default path (the points are the same).

    Returns
    -------
    intersection_points : torch.Tensor
        The intersection points.
    intersection_distances : torch.Tensor
        The line parameter ``t`` of each intersection, with ``dim`` kept as
        size 1. Infinite or NaN where a line is parallel to its plane.
    """
    if dim > 0:
        maxdim = max(
            [
                _.dim()
                for _ in (line_direction, plane_point, plane_normal, line_point)
                if hasattr(_, "dim")
            ]
        )
        dim = dim - maxdim
    if memory is None:
        lp = line_point - plane_point
        plane_normal = F.normalize(plane_normal, p=2, dim=dim)
        intersection_distances = -(
            dot_product(lp, plane_normal, dim)
            / dot_product(line_direction, plane_normal, dim)
        )
        intersection_points = line_point + line_direction * intersection_distances
        return intersection_points, intersection_distances
    with memory.temp():
        lp = torch.sub(line_point, plane_point, out=memory.get_tensor(line_point.shape))
        plane_normal = F.normalize(plane_normal, p=2, dim=dim, out=plane_normal)
        dot1 = dot_product(lp, plane_normal, dim, out=memory)
        dot2 = dot_product(line_direction, plane_normal, dim, out=memory)
        intersection_distances = torch.divide(dot1, dot2, out=dot2)
        intersection_points = torch.addcmul(
            line_point,
            line_direction,
            intersection_distances,
            value=-1,
            out=line_direction,
        )
        return intersection_points, intersection_distances


def intersect_line_with_plane_colinear(
    line_direction, plane_point, plane_co1, plane_co2, line_point=0
):
    """Intersect lines with planes given by two in-plane vectors.

    The plane normal is the normalized cross product ``plane_co1 x plane_co2``
    (last dim); everything else is as in :func:`intersect_line_with_plane`,
    with coordinates on the last dim.

    Returns
    -------
    tuple of torch.Tensor
        ``(intersection_points, intersection_distances)``, as returned by
        :func:`intersect_line_with_plane`.
    """
    plane_normal = F.normalize(
        broadcast_cross_product(plane_co1, plane_co2, dim=-1), p=2, dim=-1
    )
    return intersect_line_with_plane(
        line_direction, plane_point, plane_normal, line_point
    )


"""def intersect_line_with_line(line_point1, line_direction1, line_point2, line_direction2, dim=-1):
    lp = line_point - plane_point
    plane_normal = F.normalize(plane_normal, p=2, dim=dim)
    intersection_distances = -(dot_product(lp, plane_normal, dim) /
                              dot_product(line_direction, plane_normal, dim))
    intersection_points = line_point + line_direction * intersection_distances
    return intersection_points, intersection_distances"""


def get_rotation_around_axis(num_degrees, axis, dim=0):
    """Build the rotation of ``num_degrees`` about ``axis``, right-handed.

    Every call site applies the result to **row** vectors -- ``basis @ R``,
    ``(location - about) @ R`` -- so the matrix returned is the transpose of
    the Rodrigues form written below, and ``v @ R`` turns ``v`` counter-
    clockwise seen from the tip of ``axis``.

    That transpose is the whole of Algan's rotation handedness, and it is tied
    to ``OUTWARD`` being ``+z`` (:mod:`algan.constants.spatial`): with a
    right-handed world basis, ``rotate(90, OUTWARD)`` has to take ``RIGHT`` to
    ``UP``, and a row-vector product against an untransposed Rodrigues matrix
    takes it to ``DOWN`` instead. :func:`get_rotation_between_3d_vectors` pairs
    with this -- it returns ``cross(v1, v2)`` unnegated for the same reason.
    """
    num_radians = num_degrees * DEGREES_TO_RADIANS

    def cast_to_tensor(_):
        return (
            _
            if isinstance(_, torch.Tensor)
            else torch.tensor((_,), dtype=torch.get_default_dtype())
        )

    num_radians = cast_to_tensor(num_radians)
    c = num_radians.cos()
    s = num_radians.sin()
    n = F.normalize(axis, p=2, dim=dim)
    n2 = n * n
    n_0 = n.select(dim, 0).unsqueeze(dim)
    n_1 = n.select(dim, 1).unsqueeze(dim)
    n_2 = n.select(dim, 2).unsqueeze(dim)
    n2_0 = n2.select(dim, 0).unsqueeze(dim)
    n2_1 = n2.select(dim, 1).unsqueeze(dim)
    n2_2 = n2.select(dim, 2).unsqueeze(dim)
    R = torch.cat(
        (
            c + (n2_0) * (1 - c),
            n_0 * n_1 * (1 - c) - n_2 * s,
            n_0 * n_2 * (1 - c) + (n_1) * s,
            n_0 * n_1 * (1 - c) + n_2 * s,
            c + n2_1 * (1 - c),
            n_1 * n_2 * (1 - c) - n_0 * s,
            n_0 * n_2 * (1 - c) - n_1 * s,
            n_1 * n_2 * (1 - c) + n_0 * s,
            c + n2_2 * (1 - c),
        ),
        dim,
    )
    if dim < 0:
        dim = dim + R.dim()
    R = R.reshape(*R.shape[:dim], 3, 3, *R.shape[dim + 1 :])
    # Transposed for the row-vector application described above.
    return R.transpose(dim, dim + 1)


def rotate_vector_around_axis(vector, num_degrees, axis, dim=0):
    """Rotate ``vector`` by ``num_degrees`` about ``axis``.

    Uses the rotation built by :func:`get_rotation_around_axis`, applied to the
    vector as a row vector, so the rotation is counter-clockwise seen from the
    tip of ``axis``.

    Parameters
    ----------
    vector : torch.Tensor
        Vectors to rotate, with 3 coordinates on ``dim``.
    num_degrees : torch.Tensor or float
        Rotation angle in **degrees**.
    axis : torch.Tensor
        Rotation axis, with 3 coordinates on ``dim``; normalized internally.
    dim : int, default 0
        The coordinate dimension of ``vector`` and ``axis``. Tensor arguments
        are first left-padded with size-1 dims to a common rank.

    Returns
    -------
    torch.Tensor
        The rotated vectors, with the same shape as ``vector``.
    """
    vshape = vector.shape

    def get_dim(x):
        return x.dim() if hasattr(x, "dim") else 0

    max_dim = max([get_dim(_) for _ in [vector, num_degrees, axis]])

    def unsq(x):
        if not isinstance(x, torch.Tensor):
            return x
        return x.view(*([1] * (max_dim - x.dim())), *x.shape)

    vector, num_degrees, axis = [unsq(_) for _ in [vector, num_degrees, axis]]
    vector = unsqueeze_left(vector, axis)
    num_degrees = unsqueeze_left(num_degrees, axis)
    R = get_rotation_around_axis(num_degrees, axis, dim=dim)
    a = ascii_lowercase[: R.dim()]
    ma = ascii_lowercase[R.dim() : R.dim() + 3]
    vector = unsqueeze_left(vector.unsqueeze(dim - 1), R)
    if dim < 0:
        dim = dim + R.dim()
    a1 = "".join([a[: dim - 1], ma[:2], a[dim + 1 :]])
    a2 = "".join([a[: dim - 1], ma[1:], a[dim + 1 :]])
    a3 = "".join([a[: dim - 1], "".join([ma[0], ma[2]]), a[dim + 1 :]])
    return torch.einsum(f"{a1},{a2}->{a3}", vector, R).reshape(vshape)


def get_rotation_between_3d_vectors(vector1, vector2, dim=-1):
    """Return the angle and axis of the rotation carrying ``vector1`` to ``vector2``.

    The axis is the normalized ``cross(vector1, vector2)``, matching the
    handedness of :func:`get_rotation_around_axis`, so rotating ``vector1``
    by the returned angle about the returned axis points it along ``vector2``.
    When the vectors are (anti)parallel, i.e. the angle is within 1e-4 degrees
    of 0 or 180, the cross product is degenerate and an arbitrary but
    deterministic unit vector orthogonal to both is used instead
    (:func:`get_orthonormal_vector`).

    Parameters
    ----------
    vector1, vector2 : torch.Tensor
        Vectors of any length with 3 coordinates on ``dim``; leading dims
        broadcast.
    dim : int, default -1
        The coordinate dimension. The fallback axis always works on the last
        dim, so other values are only reliable when no pair is (anti)parallel.

    Returns
    -------
    degrees : torch.Tensor
        The angle in **degrees**, in ``[0, 180]``, with ``dim`` kept as size 1.
    axis : torch.Tensor
        Unit rotation axis.
    """
    # cross(v1, v2) unnegated: the axis that carries v1 to v2 under the
    # right-handed rotation get_rotation_around_axis builds. It was negated
    # while that rotation ran the other way.
    normal_vector = F.normalize(
        broadcast_cross_product(vector1, vector2, dim=dim), p=2, dim=dim
    )
    normal_vector_r = get_orthonormal_vector(vector1, vector2)
    radians_to_rotate = (
        F.cosine_similarity(vector1, vector2, dim=dim, eps=1e-12)
        .arccos()
        .unsqueeze(dim)
    )
    degrees_to_rotate = radians_to_rotate * RADIANS_TO_DEGREES
    normal_vector = torch.where(
        (degrees_to_rotate.abs() <= 1e-4) | ((degrees_to_rotate - 180).abs() <= 1e-4),
        normal_vector_r,
        normal_vector,
    )
    return degrees_to_rotate, normal_vector


def normalize(x, dim=-1, p=2, memory=None):
    """Scale ``x`` to unit ``p``-norm along ``dim``.

    Parameters
    ----------
    x : torch.Tensor
        The vectors to normalize.
    dim : int, default -1
        The dimension to normalize over.
    p : float, default 2
        The norm order.
    memory : optional
        Scratch-tensor allocator. When given, ``x`` is normalized **in place**
        and returned, always with the 2-norm over the last dim (``dim`` and
        ``p`` are ignored); norms are clamped to at least 1e-8.

    Returns
    -------
    torch.Tensor
        The normalized vectors, same shape as ``x``.
    """
    if memory is None:
        return F.normalize(x, p=p, dim=dim)
    with memory.temp():
        norm = torch.norm(
            x, p=2, dim=-1, keepdim=True, out=memory.get_tensor([*x.shape[:-1], 1])
        ).clamp_min_(1e-8)
        x /= norm
    return x


def invert_row_basis(basis):
    """Invert a batch of row-major 3x3 bases, shape ``(*, 3, 3)``.

    Built from the adjugate (three cross products and a determinant) rather than
    ``torch.linalg.inv``: at this size it is cheaper, it batches without a LAPACK
    call, and it lets a degenerate basis be handled rather than raised on.

    A basis whose rows are coplanar -- a Mob scaled flat along an axis, say --
    has no inverse at all. Those return the identity, i.e. "no change", so that
    a basis assignment leaves such a Mob as it is instead of turning its
    geometry into inf or NaN.
    """
    r0, r1, r2 = basis.unbind(-2)
    # Columns of the adjugate: each is orthogonal to two of the rows, so
    # ``basis @ stack(columns) == determinant * I``.
    c0 = torch.linalg.cross(r1, r2, dim=-1)
    c1 = torch.linalg.cross(r2, r0, dim=-1)
    c2 = torch.linalg.cross(r0, r1, dim=-1)
    determinant = (r0 * c0).sum(-1, keepdim=True).unsqueeze(-1)
    inverse = torch.stack((c0, c1, c2), dim=-1) / determinant
    # |determinant| is the product of the row norms exactly when the rows are
    # orthogonal and falls to zero as they become coplanar, so comparing the two
    # tests degeneracy free of scale: a Mob scaled to a millionth inverts fine,
    # a flattened one does not invert at any scale.
    volume = basis.norm(p=2, dim=-1).prod(-1).unsqueeze(-1).unsqueeze(-1)
    identity = torch.eye(3, dtype=basis.dtype, device=basis.device).expand_as(inverse)
    return torch.where(determinant.abs() > 1e-9 * volume, inverse, identity)


def get_rotation_between_bases(basis1, basis2):
    """Return the right-side transform taking row basis1 to row basis2.

    That is, ``basis1 @ get_rotation_between_bases(basis1, basis2) == basis2``.

    This has to be the exact inverse of ``basis1``, not the transpose of its
    normalized form. Mob.basis's setter turns an absolute basis into a relative
    change through this function and applies it with a right-multiply (so that
    concurrent writers compose), which means any error here is re-applied on top
    of the value it was measured from. On a sheared basis -- where the rows are
    not orthogonal -- the normalized transpose left a residual that the
    round-trip *amplified* roughly threefold, so float-noise shear grew into
    total collapse over a couple of dozen assignments (each ``detach_history``
    clone performs one). The two forms agree exactly for an orthogonal basis.
    """
    return invert_row_basis(basis1) @ basis2


def project_point_onto_line(point, line_direction, line_start=0, dim=-1):
    """Return the point on a line closest to ``point``.

    The line passes through ``line_start`` with direction ``line_direction``
    (normalized internally); coordinates lie on ``dim``.
    """
    line_direction = F.normalize(line_direction, p=2, dim=dim)
    return line_start + line_direction * dot_product(
        point - line_start, line_direction, dim=dim
    )


def project_onto_basis(vector, basis):
    """Project ``vector`` onto the span of the vectors in ``basis``.

    Computes ``sum(dot(vector, b) * b for b in basis)``, iterating over the
    first dim of ``basis``. This is the orthogonal projection only when the
    basis vectors are orthonormal; they are not normalized here.

    Parameters
    ----------
    vector : torch.Tensor
        Vectors with coordinates on the last dim.
    basis : sequence of torch.Tensor
        Basis vectors, each broadcastable against ``vector``.

    Returns
    -------
    torch.Tensor
        The projected vectors.
    """
    return sum([dot_product(vector, b) * b for b in basis])


def get_orthonormal_vector(*vectors):
    """A unit vector orthogonal to every vector in ``vectors`` (batched over the
    leading dims). The choice among the valid orthogonal directions is
    *deterministic*.

    It used to seed the Gram-Schmidt with ``torch.randn_like``, which re-rolls
    every render. Since this builds the perpendicular basis of surfaces of
    revolution (e.g. Cylinders, via ``Cylinder._move_between_points``), a random
    seed spun those meshes to a random angle about their axis on each render. The
    silhouette is rotation-symmetric so it looked stable, but which tessellated
    facet faced the light changed -- making per-facet shading and ray-traced
    shadows flicker randomly between renders (most visible on thin tubes such as
    neural-net synapses). Seeding from the fixed standard basis instead keeps the
    orientation reproducible.
    """
    vectors = [F.normalize(v, p=2, dim=-1) for v in vectors]
    v0 = vectors[0]
    # Try each standard-basis axis as the seed, project out all input vectors,
    # and keep the best-conditioned residual per batch element -- a deterministic,
    # well-conditioned choice for any input (even when an axis lies in the span of
    # ``vectors``). The *which* orthogonal direction does not matter to callers;
    # only that it is reproducible.
    #
    # All axes are projected in ONE batched pass over a new axis dim rather
    # than a per-axis Python loop: the per-element arithmetic (projection
    # chain, 3-element norm) and the sequential strictly-greater selection are
    # unchanged, so the result is bit-identical while launching ~1/d as many
    # ops. This is called per animated Cylinder point-move, thousands of times
    # per window in an updater-heavy scene -- dispatch count is its whole cost.
    d = v0.shape[-1]
    seeds = torch.eye(d, dtype=v0.dtype, device=v0.device)
    if d == 3:
        # Seed from RIGHT, UP, -z rather than the raw identity rows. They differ
        # only in the sign of the third, and the sign is not a convention this
        # could read off DEFAULT_BASIS -- it is the mirror the z flip put through
        # the whole world: a +z seed picks the opposite perpendicular to the one
        # mirrored geometry wants, which rolls a surface of revolution 180
        # degrees about its axis. The silhouette is rotation-symmetric so it
        # looks the same, but which tessellated facet faces the light changes,
        # and the pixel baselines hold this seed.
        seeds = seeds * torch.tensor(
            (1.0, 1.0, -1.0), dtype=v0.dtype, device=v0.device
        ).unsqueeze(-1)
    seeds = seeds.expand(v0.shape[:-1] + (d, d))
    r = seeds
    for vn in vectors:
        vn = vn.unsqueeze(-2)
        r = r - dot_product(r, vn) * vn
    n = r.norm(p=2, dim=-1, keepdim=True)
    # The original per-axis update started from zeros and took an axis only on
    # a strictly greater norm (first-max-wins; NaN, being > nothing, is never
    # taken and the zeros survive); replicate that exact chain over the axis
    # dim rather than argmax, whose NaN and init rules both differ.
    best = torch.zeros_like(v0)
    best_norm = torch.zeros_like(v0[..., :1])
    for axis in range(d):
        take = n[..., axis, :] > best_norm
        best = torch.where(take, r[..., axis, :], best)
        best_norm = torch.where(take, n[..., axis, :], best_norm)
    return F.normalize(best, p=2, dim=-1)


def get_2d_polygon_mask(polygon_vertices, grid_points, eps=1e-6):
    # TODO change this to use scanline
    """
    polygon_vertices: Tensor[batch[*], num_vertices, 2]
    grid_points: Tensor[batch[*], num_grid_points, 2]
    """
    pp2d = polygon_vertices
    bounded_pixels = grid_points
    parallel = torch.cat((pp2d[..., 1:, :], pp2d[..., :1, :]), -2) - pp2d
    m_ignore = (pp2d.amin(-1, keepdim=True) <= -1e12).float().unsqueeze(-3)

    parallel = F.normalize(parallel, p=2, dim=-1)
    parallel2 = -torch.cat((parallel[..., -1:, :], parallel[..., :-1, :]), -2)

    dists = torch.cdist(bounded_pixels.float(), pp2d).unsqueeze(-1)
    dists = dists * (1 - m_ignore) + m_ignore * 1e12
    nearest_ind = dists.argmin(-2, keepdim=True)
    nearest_par1 = broadcast_gather(
        parallel.unsqueeze(-3), -2, nearest_ind, keepdim=False
    )
    nearest_par2 = broadcast_gather(
        parallel2.unsqueeze(-3), -2, nearest_ind, keepdim=False
    )
    nearest_point = broadcast_gather(pp2d.unsqueeze(-3), -2, nearest_ind, keepdim=False)

    bounded_pixels = bounded_pixels.float() - nearest_point
    bounded_pixels = F.normalize(bounded_pixels, p=2, dim=-1)

    def angle(x):
        a = torch.complex(x[..., 0], x[..., 1]).angle()
        m = (a >= 0).float()
        return a * m + (1 - m) * (2 * math.pi + a)

    angles = torch.stack(
        [angle(_) for _ in [nearest_par1, nearest_par2, bounded_pixels]], -1
    )
    i = angles.argsort(-1)
    m2 = (
        broadcast_gather(
            i, -1, ((i == 2).float().argmax(-1, keepdim=True) + 1) % 3, keepdim=False
        )
        != 1
    )

    return m2.float()


def map_global_to_local_coords(location, basis, global_coords):
    """Express world-space points in a Mob's local frame.

    Inverse of :func:`map_local_to_global_coords`: finds ``c`` with
    ``location + c @ basis == global_coords``. For an orthogonal basis this
    projects onto each normalized row and divides by the row's length; for
    a sheared basis it multiplies by the exact inverse
    (:func:`invert_row_basis`, which yields the identity for a degenerate
    basis).

    Parameters
    ----------
    location : torch.Tensor
        Frame origin, shape ``(*, 3)``.
    basis : torch.Tensor
        Flattened row basis, shape ``(*, 9)``: rows are the frame's x, y and z
        axes (not necessarily unit length).
    global_coords : torch.Tensor
        World-space points, shape ``(*, 3)``.

    Returns
    -------
    torch.Tensor
        Local coordinates, shape ``(*, 3)``.
    """
    basis = unsquish(basis, -1, 3)
    offset = (global_coords - location).unsqueeze(-2)
    normalized = F.normalize(basis, p=2, dim=-1)
    gram = normalized @ normalized.transpose(-2, -1)
    identity = torch.eye(3, device=basis.device, dtype=basis.dtype)
    orthogonal = (gram - identity).abs().amax(dim=(-2, -1)) <= (
        4 * torch.finfo(basis.dtype).eps
    )
    # Keep established rounding for orthogonal frames (including rotation
    # round-off): changing it can move a raster edge across a pixel boundary.
    # Sheared frames require the actual inverse, not row projections.
    projected = dot_product(normalized, offset, -1, keepdim=False) / basis.norm(
        p=2, dim=-1
    )
    inverted = (offset @ invert_row_basis(basis)).squeeze(-2)
    return torch.where(orthogonal.unsqueeze(-1), projected, inverted)


def map_local_to_global_coords(location, basis, local_coords):
    """Express points given in a Mob's local frame in world space.

    Computes ``location + local_coords @ basis``, i.e. each local coordinate
    scales the corresponding basis row.

    Parameters
    ----------
    location : torch.Tensor
        Frame origin, shape ``(*, 3)``.
    basis : torch.Tensor
        Flattened row basis, shape ``(*, 9)``.
    local_coords : torch.Tensor
        Local coordinates, shape ``(*, 3)``.

    Returns
    -------
    torch.Tensor
        World-space points, shape ``(*, 3)``.
    """
    basis = unsquish(basis, -1, 3)
    return dot_product(basis, local_coords.unsqueeze(-1), -2, keepdim=False) + location
