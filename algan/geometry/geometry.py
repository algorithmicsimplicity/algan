import math
from string import ascii_lowercase

import torch
import torch.nn.functional as F

from algan.constants.spatial import DEGREES_TO_RADIANS, RADIANS_TO_DEGREES
from algan.utils.tensor_utils import expand_as_left, squish
from algan.utils.tensor_utils import broadcast_cross_product, dot_product, unsqueeze_left, broadcast_gather, unsquish


def intersect_line_with_plane(line_direction, plane_point, plane_normal, line_point=0, dim=-1):
    if dim > 0:
        maxdim = max([_.dim() for _ in (line_direction, plane_point, plane_normal, line_point) if hasattr(_, 'dim')])
        dim = dim - maxdim
    lp = line_point - plane_point
    plane_normal = F.normalize(plane_normal, p=2, dim=dim)
    intersection_distances = -(dot_product(lp, plane_normal, dim) /
                              dot_product(line_direction, plane_normal, dim))
    intersection_points = line_point + line_direction * intersection_distances
    return intersection_points, intersection_distances


def intersect_line_with_plane_colinear(line_direction, plane_point, plane_co1, plane_co2, line_point=0):
    plane_normal = F.normalize(broadcast_cross_product(plane_co1, plane_co2, dim=-1), p=2, dim=-1)
    return intersect_line_with_plane(line_direction, plane_point, plane_normal, line_point)


'''def intersect_line_with_line(line_point1, line_direction1, line_point2, line_direction2, dim=-1):
    lp = line_point - plane_point
    plane_normal = F.normalize(plane_normal, p=2, dim=dim)
    intersection_distances = -(dot_product(lp, plane_normal, dim) /
                              dot_product(line_direction, plane_normal, dim))
    intersection_points = line_point + line_direction * intersection_distances
    return intersection_points, intersection_distances'''


def get_rotation_around_axis(num_degrees, axis, dim=0):
    num_radians = num_degrees * DEGREES_TO_RADIANS

    def cast_to_tensor(_):
        return _ if isinstance(_, torch.Tensor) else torch.tensor((_,), dtype=torch.get_default_dtype())
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
    R = torch.cat((c + (n2_0) * (1 - c), n_0 * n_1 * (1 - c) - n_2 * s, n_0 * n_2 * (1 - c) + (n_1) * s,
                     n_0 * n_1 * (1 - c) + n_2 * s, c + n2_1 * (1 - c), n_1 * n_2 * (1 - c) - n_0 * s,
                     n_0 * n_2 * (1 - c) - n_1 * s, n_1 * n_2 * (1 - c) + n_0 * s, c + n2_2 * (1 - c)), dim)
    if dim < 0:
        dim = dim + R.dim()
    R = R.reshape(*R.shape[:dim], 3, 3, *R.shape[dim+1:])
    #R = R.transpose(dim, dim+1)
    #R = R.permute(*range(R.dim()-1,-1,-1)).squeeze(0)
    #TODO change this permute to something that accepts arbitrray number of batch dims
    return R


def rotate_vector_around_axis(vector, num_degrees, axis, dim=0):
    vshape = vector.shape

    def get_dim(x):
        return x.dim() if hasattr(x, 'dim') else 0
    max_dim= max([get_dim(_) for _ in [vector, num_degrees, axis]])
    def unsq(x):
        if not isinstance(x, torch.Tensor):
            return x
        return x.view(*([1]*(max_dim-x.dim())), *x.shape)
    vector, num_degrees, axis = [unsq(_) for _ in [vector, num_degrees, axis]]
    vector = unsqueeze_left(vector, axis)
    num_degrees = unsqueeze_left(num_degrees, axis)
    R = get_rotation_around_axis(num_degrees, axis, dim=dim)
    a = ascii_lowercase[:R.dim()]
    ma = ascii_lowercase[R.dim():R.dim()+3]
    vector = unsqueeze_left(vector.unsqueeze(dim-1), R)
    if dim < 0:
        dim = dim + R.dim()
    r = [a[i] for i in [dim]]
    a1 = ''.join([a[:dim-1], ma[:2], a[dim+1:]])
    a2 = ''.join([a[:dim-1], ma[1:], a[dim+1:]])
    a3 = ''.join([a[:dim-1], ''.join([ma[0], ma[2]]), a[dim+1:]])
    return torch.einsum(f'{a1},{a2}->{a3}', vector, R).reshape(vshape)
    #return torch.einsum('ij...,jk...->ik...', vector, R)
    return (vector.unsqueeze(-2) @ R).squeeze(-2)


def get_rotation_between_3d_vectors(vector1, vector2, dim=-1):
    normal_vector = -F.normalize(broadcast_cross_product(vector1, vector2, dim=dim), p=2, dim=dim)
    radians_to_rotate = F.cosine_similarity(vector1, vector2, dim=dim, eps=1e-12).arccos().unsqueeze(dim)
    return radians_to_rotate*RADIANS_TO_DEGREES, normal_vector


def rotate_basis_to_direction(basis, direction, axis=-1, dim=-1):
    angle, axis = get_rotation_between_3d_vectors(basis[..., axis, :], direction, dim=dim)
    return rotate_vector_around_axis(basis, angle.unsqueeze(dim-1), axis.unsqueeze(dim-1), dim=dim)


def get_rotation_between_bases(basis1, basis2):
    n1 = basis1.norm(p=2, dim=-1, keepdim=True)
    n2 = basis2.norm(p=2, dim=-1, keepdim=True)
    o1 = F.normalize(basis1, p=2, dim=-1)
    o2 = F.normalize(basis2, p=2, dim=-1)
    rot = o1.transpose(-2,-1) @ o2
    #scale = n2 / n1
    return (1/n1) * rot * n2#.transpose(-2,-1)


def get_rotation_between_orthonormal_bases(basis1, basis2):
    return basis1.transpose(-2,-1) @ basis2


def get_roots_of_normalized_polynomial(coefs):
    n = coefs.shape[-1]-1
    base_matrix = torch.cat((torch.zeros((1,n), device=coefs.device), torch.eye(n, device=coefs.device)), -2)
    coefs = coefs.unsqueeze(-1)
    companion_matrix = torch.cat((expand_as_left(base_matrix, coefs), -coefs.flip(-2)), -1)
    roots = torch.linalg.eigvals(companion_matrix)
    m = (roots.imag.abs() < 1e-12).type(coefs.dtype)
    return roots.real * m + (1-m) * 2e12


def pad_to_length(x, length):
    return torch.cat((x, torch.zeros((list(x.shape[:-1]) + [length-x.shape[-1]]), device=x.device)), -1)


def project_point_onto_line(point, line_direction, line_start=0, dim=-1):
    """
    Projects point x to the closest point on a line defined by a starting point and a direction
    """
    line_direction = F.normalize(line_direction, p=2, dim=dim)
    return line_start + line_direction * dot_product(point - line_start, line_direction, dim=dim)


def project_point_onto_line_segment(point, line_start, line_end, dim=-1):
    """
    Projects point x to the closest point on a line segment defined by its start and end points.
    """
    line_direction = F.normalize(line_end-line_start, p=2, dim=dim)
    line_lengths = (line_end - line_start).norm(p=2,dim=-1, keepdim=True)
    return line_start + line_direction * dot_product(point - line_start, line_direction, dim=dim).clamp(min=torch.zeros_like(line_lengths), max=line_lengths)


def project_point_onto_plane(point, plane_normal, plane_point=0, dim=-1):
    """
    Projects point x onto a plane defined by a point and normal direction
    """
    return project_point_onto_line(point, get_orthonormal_vector(plane_normal), plane_point, dim)


def get_roots_of_quadratic_no_backup(a, b, c, fill_value:float=2e12):
    out = torch.empty([max([_.shape[i] for _ in [a, b, c]]) for i in range(a.dim())] + [2], dtype=a.dtype,
                      device=a.device)  # [...,:2])
    disc = a  # .clone()
    disc = disc * -4 * c
    disc += b.square()
    disc.sqrt_()
    # disc = (b * b - 4 * a * c).sqrt_()
    q = b.clone()
    q = q + ((b >= 0).type(a.dtype)*2-1) * disc
    q *= -0.5
    # q = -0.5 * (b + ((b >= 0).float()*2-1) * disc)
    out[..., 0] = c  # / q
    out[..., 0] /= q
    out[..., 1] = q  # / a
    out[..., 1] /= a
    # out = out
    '''out *= m
    m *= -1
    m += 1
    m *= -c.unsqueeze(-1)
    m /= b.unsqueeze(-1)
    out += m#(1-m)'''
    out.nan_to_num_(nan=fill_value, posinf=fill_value, neginf=fill_value)
    return out

#@torch.jit.script
def get_roots_of_quadratic(a, b, c, fill_value:float=2e12):
    m = (a.abs() <= 1e-7).unsqueeze(-1)
    m2 = (b.abs() <= 1e-7)#.unsqueeze(-1)
    backup = (-c / b).nan_to_num_(nan=fill_value, posinf=fill_value, neginf=fill_value)
    backup = backup * (~m2) + m2 * fill_value#(-c / b).nan_to_num_(nan=fill_value, posinf=fill_value, neginf=fill_value)
    backup = torch.stack((backup, torch.full_like(backup, fill_value)), -1)
    #a = coefs[...,0]
    #b = coefs[...,1]
    #c = coefs[...,2]
    #m = (a.abs() > 1e-12).float().unsqueeze(-1)
    out = get_roots_of_quadratic_no_backup(a, b, c, fill_value)
    return out * (~m) + m * backup
    return torch.cat((out, backup.unsqueeze(-1)), -1)
    #out = (out * m + (1-m) * (-c/b).unsqueeze(-1)).nan_to_num_(nan=fill_value, posinf=fill_value, neginf=fill_value)
    return out


@torch.jit.script
def nth_root(z, n: int):
    theta = z.angle()
    angles = torch.stack([(theta + k * math.pi * 2) / n for k in range(n)], -1)
    roots = torch.view_as_complex(torch.stack((angles.cos(), angles.sin()), -1))
    return roots * z.unsqueeze(-1).abs().pow_(1 / n)


#@torch.jit.script
def get_roots_of_cubic(a, b, c, d, fill_value: float = 2e12):
    m_backup = (a.abs() <= 1e-7).unsqueeze(-1)

    backup_roots = get_roots_of_quadratic(expand_as_left(b, d), expand_as_left(c, d), d, fill_value)
    backup_roots = torch.cat((backup_roots, torch.full_like(backup_roots, fill_value)), -2)

    m = 10000
    b = (b / a).clamp(min=-m, max=m)
    c = (c / a).clamp(min=-m, max=m)
    d = (d / a).clamp(min=-m, max=m)

    #p = b - c.square()/3
    #q = (9*b*c-27*b-2*d.pow(3))/27
    #C = (0.5)*q*(3/p.abs()).pow(1.5)

    """def make_nonzero(x):
        m = (x.abs() > 1e-5).float()
        x = x * m + (1-m) * 1e-5
        return x"""

    a_inv = 1/a
    p = -(b.pow(3)) * a_inv.pow(3) / 27 + b * c * a_inv.square() / 6 - d * a_inv * 0.5
    #p = a_inv * (-b * (b.square()*a_inv.square())/27 + c*a_inv/6 - d*0.5)
    q = (c*a_inv/3 - b.square()*a_inv.square()/9).pow(3)
    z = (p.square()+q)
    z = torch.view_as_complex(torch.stack((z, torch.zeros_like(z)), -1))

    z_roots = nth_root(z, 2)
    p = p.unsqueeze(-1)
    all_roots = squish(nth_root(p-z_roots, 3), -2, -1).unsqueeze(-1).real + squish(nth_root(p+z_roots, 3), -2, -1).unsqueeze(-2).real
    all_roots = squish(all_roots, -2, -1) - (b*a_inv/3).unsqueeze(-1)

    all_roots.nan_to_num_(nan=fill_value,posinf=fill_value,neginf=fill_value)
    all_roots = all_roots * (~m) + m * backup_roots
    return all_roots

    #return (all_roots.nan_to_num().nan_to_num(nan=0,posinf=0,neginf=0)).float()
    return torch.cat((all_roots.nan_to_num_(nan=fill_value,posinf=fill_value,neginf=fill_value), backup_roots), -1)
    return (all_roots.nan_to_num(nan=fill_value,posinf=fill_value,neginf=fill_value) * m + (1-m) * pad_to_length(backup_roots, all_roots.shape[-1])).float()


def get_roots_of_quadratic_backup_recurse_clean(coefs, fill_value:float=2e12):
    a = coefs[...,0]
    b = coefs[...,1]
    c = coefs[...,2]
    m = (a.abs() > 1e-12).type(coefs.dtype).unsqueeze(-1)
    out = torch.empty_like(coefs[...,:2])
    disc = (b * b - 4 * a * c).sqrt_()
    q = -0.5 * (b + (b >= 0).type(coefs.dtype) * disc)
    out[...,0] = c / q
    out[...,1] = q / a
    #out = out
    out = (out * m + (1-m) * (-c/b).unsqueeze(-1)).nan_to_num_(nan=fill_value, posinf=fill_value, neginf=fill_value)
    return out


def get_roots_of_polynomial_backup_recurse(coefs):
    m = (coefs[...,:1].abs() > 0).type(coefs.dtype)

    normalized_coefs = coefs[...,1:] / (coefs[...,:1] * m + (1-m))
    roots = get_roots_of_normalized_polynomial(normalized_coefs)
    backup_roots = get_roots_of_polynomial_backup_recurse(coefs[...,1:]) if (coefs.shape[-1] > 3) else\
                   (-coefs[...,-1] / coefs[...,-2]).nan_to_num(nan=0,posinf=0,neginf=0).unsqueeze(-1)

    return roots * m + (1 - m) * pad_to_length(backup_roots, roots.shape[-1])


def get_orthonormal_vector(vector):
    r = torch.randn_like(vector)
    vn = F.normalize(vector, p=2, dim=-1)
    r = r - dot_product(r, vn) * vn
    return F.normalize(r, p=2, dim=-1)


def get_2d_polygon_mask(polygon_vertices, grid_points, eps=1e-6):
    """
    polygon_vertices: Tensor[batch[*], num_vertices, 2]
    grid_points: Tensor[batch[*], num_grid_points, 2]
    """
    pp2d = polygon_vertices
    bounded_pixels = grid_points
    #parallel = pp2d[..., 1:, :] - pp2d[..., :-1, :]
    #pp2d = pp2d[..., :-1, :]
    parallel = torch.cat((pp2d[...,1:,:], pp2d[...,:1,:]), -2) - pp2d
    m_ignore = (pp2d.amin(-1, keepdim=True) <= -1e12).float().unsqueeze(-3)

    parallel = F.normalize(parallel, p=2, dim=-1)
    parallel2 = -torch.cat((parallel[..., -1:, :], parallel[..., :-1, :]), -2)

    #parallel[...,-1:,:] = parallel[...,-2:-1,:]
    #parallel2[...,:1,:] = parallel2[...,1:2,:]

    #perp = torch.stack((parallel[..., 1], -parallel[..., 0]), -1)
    #perp2 = torch.cat((perp[..., -1:, :], perp[..., :-1, :]), -2)

    """
    bounded_pixels: Tensor[
    pp: Tensor[frames, Batch[*], num_points, 3] 
    """

    dists = torch.cdist(bounded_pixels.float(), pp2d).unsqueeze(-1)
    dists = dists * (1-m_ignore) + m_ignore * 1e12
    nearest_ind = dists.argmin(-2, keepdim=True)
    nearest_par1 = broadcast_gather(parallel.unsqueeze(-3), -2, nearest_ind, keepdim=False)
    nearest_par2 = broadcast_gather(parallel2.unsqueeze(-3), -2, nearest_ind, keepdim=False)
    nearest_point = broadcast_gather(pp2d.unsqueeze(-3), -2, nearest_ind, keepdim=False)
    #nearest_dists = dists.amin(-2, keepdim=True)
    #m = (dists <= nearest_dists + eps).float()

    bounded_pixels = bounded_pixels.float() - nearest_point
    bounded_pixels = F.normalize(bounded_pixels, p=2, dim=-1)

    def angle(x):
        a = torch.complex(x[..., 0], x[..., 1]).angle()
        m = (0 <= a).float()
        return a * m + (1-m) * (2*math.pi + a)
    #dots1 = dot_product(nearest_perp1, bounded_pixels, dim=-1, keepdim=True)
    #dots2 = dot_product(nearest_perp2, bounded_pixels, dim=-1, keepdim=True)
    angles = torch.stack([angle(_) for _ in [nearest_par1, nearest_par2, bounded_pixels]], -1)
    ##plot_tensor(bounded_pixels[0,...,1].view(1, 251, 205).abs())
    ## plot_tensor(nearest_ind[0].view(1, 251, 205)==2)
    m1 = (angles[...,-1:] - angles[..., :-1]).abs().amin(-1) <= 0.0001
    i = angles.argsort(-1)
    m2 = broadcast_gather(i, -1, ((i == 2).float().argmax(-1, keepdim=True)+1)%3, keepdim=False) != 1

    def rs(x):
        return x[0,0].view(107,96,-1)

    return (m2).float()#.squeeze(-1)
    ##plot_tensor((broadcast_gather(i, -1, ((i == 2).float().argmax(-1, keepdim=True)+1)%3, keepdim=False) != 1)[-1].view(1, 251, 205).float())
    #plot_tensor((m2.float())[0, 0].view(1,107,96))


    nearest_perp = torch.stack((bounded_pixels[..., 1], -bounded_pixels[..., 0]), -1)

    def get_dots(x):
        d1 = dot_product(x, nearest_par1, dim=-1)
        d2 = dot_product(x, nearest_par2, dim=-1)
        return d1, d2

    d1_perp, d2_perp = get_dots(nearest_perp)
    d1_par, d2_par = get_dots(bounded_pixels)
    m = (d1_perp.abs() <= 0.1) | (d2_perp.abs() <= 0.1)
    return ((((d1_perp >= 0) != (d2_perp > 0)) & ~m) | (m & ((d1_par > 0.1) != (d2_par > 0.1)))).float().squeeze(-1)

    def angle(x):
        a = torch.complex(x[..., 1], x[..., 0]).angle()
        m = (0 <= a).float()
        return a * m + (1-m) * (2*math.pi + a)
    #dots1 = dot_product(nearest_perp1, bounded_pixels, dim=-1, keepdim=True)
    #dots2 = dot_product(nearest_perp2, bounded_pixels, dim=-1, keepdim=True)
    a1, a2, ab = [angle(_) for _ in [nearest_par1, nearest_par2, bounded_pixels]]
    a1 = a1 - 0.1
    a2 = a2 + 0.1

    return (((a1 <= ab) & (ab <= a2))).float()#.squeeze(-1)

    dots = torch.minimum(dots1, dots2)
    #dots = (dots * m + (1-m) * 1e12).amin(-2)
    return (dots <= 0).float().squeeze(-1)

    #dots1 = dot_product(perp1.unsqueeze(-3), bounded_pixels.unsqueeze(-2) - pp2d.unsqueeze(-3), dim=-1, keepdim=True)
    bounded_pixels = bounded_pixels.float().unsqueeze(-2)
    parallel = parallel.unsqueeze(-3)
    perp_dists = (bounded_pixels - dot_product(bounded_pixels, parallel) * parallel).norm(p=2,dim=-1, keepdim=True)
    max_ind = (perp_dists * m + (1-m) * -1e12).argmax(-2, keepdim=True)
    dots = dot_product(perp.unsqueeze(-3), bounded_pixels, dim=-1, keepdim=True) - \
            dot_product(perp, pp2d, dim=-1, keepdim=True).unsqueeze(-3)

    #max_ind = (dots.abs() * m + (1-m) * -1e12).argmax(-2, keepdim=True)

    dots = broadcast_gather(dots, -2, max_ind, keepdim=False)

    #md = (dots > 0)
    return (dots <= 0).float().squeeze(-1)
    md = ((dots.unsqueeze(-2) * mf + (1-mf) * 1e12).amin(-2, keepdim=False) < 0)
    return md.float().squeeze(-1)


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
    m = (dists <= nearest_dists + eps)#.float()
    #dots1 = dot_product(perp1.unsqueeze(-3), bounded_pixels.unsqueeze(-2) - pp2d.unsqueeze(-3), dim=-1, keepdim=True)
    perp1 = F.normalize(perp1, p=2, dim=-1)
    perp2 = F.normalize(perp2, p=2, dim=-1)
    dots1 = dot_product(perp1.unsqueeze(-3), bounded_pixels.float().unsqueeze(-2), dim=-1, keepdim=True) - \
            dot_product(perp1, pp2d, dim=-1, keepdim=True).unsqueeze(-3)
    #dots2 = dot_product(perp2.unsqueeze(-3), bounded_pixels.unsqueeze(-2) - pp2d.unsqueeze(-3), dim=-1, keepdim=True)
    dots2 = dot_product(perp2.unsqueeze(-3), bounded_pixels.float().unsqueeze(-2), dim=-1, keepdim=True) - \
            dot_product(perp2, pp2d, dim=-1, keepdim=True).unsqueeze(-3)

    max_dot = torch.minimum(dots1.abs(), dots2.abs())
    mf = m.float()
    max_ind = (max_dot * mf + (1-mf) * -1e12).argmax(-2, keepdim=True)

    dots1 = broadcast_gather(dots1, -2, max_ind, keepdim=False)
    dots2 = broadcast_gather(dots2, -2, max_ind, keepdim=False)

    md = (~((dots1 > 0) & (dots2 > 0)))#.float()
    return md.float().squeeze(-1)
    return (m & md).any(-2).float().squeeze(-1)

    pp2d = pp2d.unsqueeze(-3)
    nearest_ind = (bounded_pixels.unsqueeze(-2) - pp2d).norm(p=2, dim=-1, keepdim=True).argmin(-2, keepdim=True)
    nearest_point = broadcast_gather(pp2d, -2, nearest_ind, keepdim=False)

    def get_dots(perps):
        nearest_normal = broadcast_gather(perps.unsqueeze(-3), -2, nearest_ind, keepdim=False)
        return (dot_product(nearest_normal, (bounded_pixels - nearest_point), dim=-1, keepdim=False) > 0)

    d1, d2 = [get_dots(_) for _ in (perp1, perp2)]
    return (~(d1 & d2)).float()


def map_global_to_local_coords(location, basis, global_coords):
    basis = unsquish(basis, -1 ,3)
    scale = basis.norm(p=2, dim=-1)
    basis = F.normalize(basis, p=2, dim=-1)
    return dot_product(basis, (global_coords - location).unsqueeze(-2), -1, keepdim=False) / scale


def map_local_to_global_coords(location, basis, local_coords):
    basis = unsquish(basis, -1, 3)
    return dot_product(basis, local_coords.unsqueeze(-1), -2, keepdim=False) + location
    scale = basis.norm(p=2, dim=-1)
    basis = F.normalize(basis, p=2, dim=-1)
    return dot_product(basis, local_coords.unsqueeze(-1), -2, keepdim=False) * scale + location