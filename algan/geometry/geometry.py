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
    to :data:`~algan.constants.spatial.OUTWARD` being ``+z``: with a
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
    [a[i] for i in [dim]]
    a1 = "".join([a[: dim - 1], ma[:2], a[dim + 1 :]])
    a2 = "".join([a[: dim - 1], ma[1:], a[dim + 1 :]])
    a3 = "".join([a[: dim - 1], "".join([ma[0], ma[2]]), a[dim + 1 :]])
    return torch.einsum(f"{a1},{a2}->{a3}", vector, R).reshape(vshape)
    # return torch.einsum('ij...,jk...->ik...', vector, R)
    return (vector.unsqueeze(-2) @ R).squeeze(-2)


def get_rotation_between_3d_vectors(vector1, vector2, dim=-1):
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


def rotate_basis_to_direction(basis, direction, axis=-1, dim=-1):
    angle, axis = get_rotation_between_3d_vectors(
        basis[..., axis, :], direction, dim=dim
    )
    return rotate_vector_around_axis(
        basis, angle.unsqueeze(dim - 1), axis.unsqueeze(dim - 1), dim=dim
    )


def normalize(x, dim=-1, p=2, memory=None):
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


def get_rotation_between_orthonormal_bases(basis1, basis2):
    return basis1.transpose(-2, -1) @ basis2


def project_point_onto_line(point, line_direction, line_start=0, dim=-1):
    """Projects point x to the closest point on a line defined by a starting point and a direction"""
    line_direction = F.normalize(line_direction, p=2, dim=dim)
    return line_start + line_direction * dot_product(
        point - line_start, line_direction, dim=dim
    )


def project_point_onto_line_segment(point, line_start, line_end, dim=-1, memory=None):
    """Projects point x to the closest point on a line segment defined by its start and end points."""
    if memory is None:
        line_direction = F.normalize(line_end - line_start, p=2, dim=dim)
        line_lengths = (line_end - line_start).norm(p=2, dim=dim, keepdim=True)
        return line_start + line_direction * dot_product(
            point - line_start, line_direction, dim=dim
        ).clamp(min=torch.zeros_like(line_lengths), max=line_lengths)

    out = memory.get_tensor(point.shape, dtype=torch.float)
    pointer = memory.current_pointer
    lines = memory.get_tensor(line_start.shape, dtype=torch.float)
    lines = torch.subtract(line_end, line_start, out=lines)

    line_lengths = memory.get_tensor(
        [*line_start.shape[:dim], 1, *line_start.shape[dim:][1:]], dtype=torch.float
    )
    line_lengths = torch.norm(lines, p=2, dim=dim, out=line_lengths, keepdim=True)

    line_direction = F.normalize(lines, p=2, dim=dim, out=lines)

    torch.subtract(point, line_start, out=out)

    dots = memory.get_tensor(
        [*point.shape[:dim], 1, *point.shape[dim:][1:]], dtype=torch.float
    )
    dots = dot_product(out, line_direction, dim=dim, out=dots)
    dots.clamp_min_(0)
    dots.clamp_max_(line_lengths)

    out = torch.addcmul(line_start, line_direction, dots, value=1, out=out)
    memory.current_pointer = pointer
    return out


def project_point_onto_plane(point, plane_normal, plane_point=0, dim=-1):
    """Projects point x onto a plane defined by a point and normal direction"""
    return project_point_onto_line(
        point, get_orthonormal_vector(plane_normal), plane_point, dim
    )


def project_onto_basis(vector, basis):
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
    # parallel = pp2d[..., 1:, :] - pp2d[..., :-1, :]
    # pp2d = pp2d[..., :-1, :]
    parallel = torch.cat((pp2d[..., 1:, :], pp2d[..., :1, :]), -2) - pp2d
    m_ignore = (pp2d.amin(-1, keepdim=True) <= -1e12).float().unsqueeze(-3)

    parallel = F.normalize(parallel, p=2, dim=-1)
    parallel2 = -torch.cat((parallel[..., -1:, :], parallel[..., :-1, :]), -2)

    # parallel[...,-1:,:] = parallel[...,-2:-1,:]
    # parallel2[...,:1,:] = parallel2[...,1:2,:]

    # perp = torch.stack((parallel[..., 1], -parallel[..., 0]), -1)
    # perp2 = torch.cat((perp[..., -1:, :], perp[..., :-1, :]), -2)

    """
    bounded_pixels: Tensor[
    pp: Tensor[frames, Batch[*], num_points, 3] 
    """

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
    # nearest_dists = dists.amin(-2, keepdim=True)
    # m = (dists <= nearest_dists + eps).float()

    bounded_pixels = bounded_pixels.float() - nearest_point
    bounded_pixels = F.normalize(bounded_pixels, p=2, dim=-1)

    def angle(x):
        a = torch.complex(x[..., 0], x[..., 1]).angle()
        m = (a >= 0).float()
        return a * m + (1 - m) * (2 * math.pi + a)

    # dots1 = dot_product(nearest_perp1, bounded_pixels, dim=-1, keepdim=True)
    # dots2 = dot_product(nearest_perp2, bounded_pixels, dim=-1, keepdim=True)
    angles = torch.stack(
        [angle(_) for _ in [nearest_par1, nearest_par2, bounded_pixels]], -1
    )
    ##plot_tensor(bounded_pixels[0,...,1].view(1, 251, 205).abs())
    ## plot_tensor(nearest_ind[0].view(1, 251, 205)==2)
    i = angles.argsort(-1)
    m2 = (
        broadcast_gather(
            i, -1, ((i == 2).float().argmax(-1, keepdim=True) + 1) % 3, keepdim=False
        )
        != 1
    )

    def rs(x):
        return x[0, 0].view(107, 96, -1)

    return (m2).float()  # .squeeze(-1)


def get_2d_polygon_mask2(polygon_vertices, grid_points, eps=1e-6):
    """
    polygon_vertices: Tensor[batch[*], num_vertices, 2]
    grid_points: Tensor[batch[*], num_grid_points, 2]
    """
    pp2d = polygon_vertices
    bounded_pixels = grid_points
    paralel = torch.cat((pp2d[..., 1:, :], pp2d[..., :1, :]), -2) - pp2d
    paralel[..., -1:, :] = 0
    perp1 = torch.stack((paralel[..., 1], -paralel[..., 0]), -1)
    paralel2 = torch.cat((pp2d[..., -1:, :], pp2d[..., :-1, :]), -2) - pp2d
    paralel2[..., :1, :] = 0
    perp2 = torch.stack((-paralel2[..., 1], paralel2[..., 0]), -1)

    """
    bounded_pixels: Tensor[
    pp: Tensor[frames, Batch[*], num_points, 3] 
    """

    dists = torch.cdist(bounded_pixels.float(), pp2d).unsqueeze(-1)
    nearest_dists = dists.amin(-2, keepdim=True)
    m = dists <= nearest_dists + eps  # .float()
    # dots1 = dot_product(perp1.unsqueeze(-3), bounded_pixels.unsqueeze(-2) - pp2d.unsqueeze(-3), dim=-1, keepdim=True)
    perp1 = F.normalize(perp1, p=2, dim=-1)
    perp2 = F.normalize(perp2, p=2, dim=-1)
    dots1 = dot_product(
        perp1.unsqueeze(-3), bounded_pixels.float().unsqueeze(-2), dim=-1, keepdim=True
    ) - dot_product(perp1, pp2d, dim=-1, keepdim=True).unsqueeze(-3)
    # dots2 = dot_product(perp2.unsqueeze(-3), bounded_pixels.unsqueeze(-2) - pp2d.unsqueeze(-3), dim=-1, keepdim=True)
    dots2 = dot_product(
        perp2.unsqueeze(-3), bounded_pixels.float().unsqueeze(-2), dim=-1, keepdim=True
    ) - dot_product(perp2, pp2d, dim=-1, keepdim=True).unsqueeze(-3)

    max_dot = torch.minimum(dots1.abs(), dots2.abs())
    mf = m.float()
    max_ind = (max_dot * mf + (1 - mf) * -1e12).argmax(-2, keepdim=True)

    dots1 = broadcast_gather(dots1, -2, max_ind, keepdim=False)
    dots2 = broadcast_gather(dots2, -2, max_ind, keepdim=False)

    md = ~((dots1 > 0) & (dots2 > 0))  # .float()
    return md.float().squeeze(-1)
    return (m & md).any(-2).float().squeeze(-1)

    pp2d = pp2d.unsqueeze(-3)
    nearest_ind = (
        (bounded_pixels.unsqueeze(-2) - pp2d)
        .norm(p=2, dim=-1, keepdim=True)
        .argmin(-2, keepdim=True)
    )
    nearest_point = broadcast_gather(pp2d, -2, nearest_ind, keepdim=False)

    def get_dots(perps):
        nearest_normal = broadcast_gather(
            perps.unsqueeze(-3), -2, nearest_ind, keepdim=False
        )
        return (
            dot_product(
                nearest_normal, (bounded_pixels - nearest_point), dim=-1, keepdim=False
            )
            > 0
        )

    d1, d2 = [get_dots(_) for _ in (perp1, perp2)]
    return (~(d1 & d2)).float()


def map_global_to_local_coords(location, basis, global_coords):
    basis = unsquish(basis, -1, 3)
    scale = basis.norm(p=2, dim=-1)
    basis = F.normalize(basis, p=2, dim=-1)
    return (
        dot_product(basis, (global_coords - location).unsqueeze(-2), -1, keepdim=False)
        / scale
    )


def map_local_to_global_coords(location, basis, local_coords):
    basis = unsquish(basis, -1, 3)
    return dot_product(basis, local_coords.unsqueeze(-1), -2, keepdim=False) + location
    scale = basis.norm(p=2, dim=-1)
    basis = F.normalize(basis, p=2, dim=-1)
    return (
        dot_product(basis, local_coords.unsqueeze(-1), -2, keepdim=False) * scale
        + location
    )
