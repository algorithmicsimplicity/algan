import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import svgelements

from algan.animation.animation_contexts import Off, Sync
from algan.constants.rate_funcs import ease_out_exp
from algan.constants.color import RED, WHITE, GREEN, RED_A
from algan.constants.spatial import RIGHT, DOWN
from algan.defaults.device_defaults import DEFAULT_DEVICE
from algan.defaults.directory_defaults import DEFAULT_DIR
from algan.geometry.geometry import get_roots_of_cubic, get_roots_of_quadratic, \
    get_2d_polygon_mask
from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import Quad, TriangleTriangulated
from algan.utils.tensor_utils import dot_product, squish, broadcast_gather, expand_as_left, unsquish, \
    unsqueeze_left, packed_reorder, unpack_tensor


def get_corners(g, i, j):
    return torch.stack([g[i, j], g[i, j + 1], g[i + 1, j + 1], g[i + 1, j]], -2)


def get_points_per_tile(grid, perimeter_points, max_pp=500):
    grid = torch.stack([grid[...,:-1,:-1,:], grid[...,:-1,1:,:], grid[...,1:, 1:,:], grid[...,1:,:-1,:]], -2)
    perimeter_points = perimeter_points.squeeze(-2).squeeze(-2)
    mask_ignore = (perimeter_points.amin(-1) > -1e12).float()
    m = get_2d_polygon_mask(grid, perimeter_points)
    m = 1-m
    m = unsquish(F.conv1d(squish(m).unsqueeze(1), torch.ones((1,1,3)), padding=1).clamp(max=1).squeeze(1), 0, m.shape[1])
    perimeter_points = perimeter_points.unsqueeze(0).unsqueeze(0).expand(m.shape[0], m.shape[1], -1, -1)
    m = m * mask_ignore + (1-mask_ignore)

    def mask_to_inds(x):
        i = x.nonzero().squeeze(-1)
        if len(i) == 0:
            return torch.zeros((max_pp,), dtype=torch.long)
        i = i[:max_pp]
        i = torch.cat((i, torch.full_like(i[-1:].expand(max_pp-i.shape[0]), m.shape[-1])))
        return i

    inds = torch.stack([torch.stack([mask_to_inds(m[i,j]) for i in range(m.shape[0])]) for j in range(m.shape[1])], 1)
    return broadcast_gather(torch.cat((perimeter_points, torch.full_like(perimeter_points[...,:1,:], -1e12)), -2), -2, inds.unsqueeze(-1), keepdim=True)


from algan.external_libraries.ground.base import get_context
from algan.external_libraries.sect.triangulation import Triangulation


"""import triangle as tr
import numpy as np


def circle(N, R):
    i = np.arange(N)
    theta = i * 2 * np.pi / N
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    seg = np.stack([i, i + 1], axis=1) % N
    return pts, seg


pts0, seg0 = circle(30, 1.4)
pts1, seg1 = circle(16, 0.6)
pts = np.vstack([pts0, pts1])
seg = np.vstack([seg0, seg1 + seg0.shape[0]])

A = dict(vertices=pts, segments=seg, holes=[[0, 0]])
B = tr.triangulate(A, 'qpa0.05')"""


def triangulate_simple_polygon(polygons):
    all_triangles = []
    count = 0
    context = get_context()
    Polygon = context.polygon_cls
    Contour, Point = context.contour_cls, context.point_cls
    triangle_counts = []

    for grid in polygons:
        grid_triangles = []
        for i, vertices in enumerate(grid):
            if len(vertices) == 0:
                continue
            all_verts = vertices
            points = Polygon(Contour([Point(*[__.item() for __ in _]) for _ in vertices]), [])

            try:
                triangles = Triangulation.constrained_delaunay(points, context=context).triangles()
            except KeyError:
                continue
            if len(triangles) > 0:
                grid_triangles.extend([torch.stack([torch.tensor((_.x, _.y)) for _ in c.vertices]) for c in triangles])
        all_triangles.extend(grid_triangles)
        triangle_counts.append(len(grid_triangles))
    return torch.stack(all_triangles) if len(all_triangles) > 0 else torch.zeros((1,3,2)), torch.tensor(triangle_counts)


def tile_region(perimeter_points, tile_size, random_perturbation=0.0, reverse_points=False, color=GREEN, **kwargs):
    """
    perimeter_points: Tensor[num_points, 2]: collection of points outlining the perimeter of the region to be tiled.
    perimeter_normals: Tensor[num_points, 2]: unit vectors pointing in direction from perimeter_point out of the region.
    tile_size: size of each tile.
    random_perturbation: strength of random perturbation applied to tile corners.
    """

    m = (perimeter_points > -1e11).float()
    mn_corner, mx_corner = (perimeter_points * m + (1-m) * 1e12).amin(0)-1e-5, perimeter_points.amax(0)+1e-5
    bounding_width, bounding_height = mx_corner - mn_corner
    grid_x = torch.linspace(mn_corner[0], mx_corner[0], steps=int(bounding_width // tile_size)+2)
    grid_y = torch.linspace(mn_corner[1], mx_corner[1], steps=int(bounding_height // tile_size)+2)
    grid = torch.stack((grid_x.view(1,-1).expand(len(grid_y),-1), grid_y.view(-1, 1).expand(-1, len(grid_x))), -1)
    te = tile_size*0.3 - 1e-5
    torch.manual_seed(42)
    grid[1:-1,1:-1] = grid[1:-1,1:-1] + (torch.randn_like(grid[1:-1,1:-1]) * random_perturbation*te).clamp_(min=-te, max=te)

    grid4 = torch.stack([grid[...,:-1,:-1,:], grid[...,:-1,1:,:], grid[...,1:, 1:,:], grid[...,1:,:-1,:]], -2)

    m = ((perimeter_points - torch.cat((perimeter_points[-1:], perimeter_points[:-1]))).norm(p=2,dim=-1) > 1e-6)
    perimeter_points = perimeter_points[m]
    if len(perimeter_points) == 0:
        return None, None
    grid_interior_mask = get_2d_polygon_mask(((perimeter_points) if reverse_points else perimeter_points), squish(grid))

    torch.set_default_device(torch.device('cpu'))
    perimeter_points = perimeter_points.cpu()
    grid4 = grid4.cpu()

    def intersect_line_segments(s1, e1, s2, e2):
        origs = [_.clone() for _ in [s1, e1, s2, e2]]
        e1, s2, e2 = [_ - s1 for _ in [e1, s2, e2]]
        p = torch.stack((e1[...,1], -e1[...,0]), -1)
        x = e2 - s2
        b = s2

        a = -dot_product(b, p, dim=-1, keepdim=True)/dot_product(x, p, dim=-1, keepdim=True)

        y = a*x + b
        d1 = dot_product(y, e1, dim=-1, keepdim=True)
        m1 = (0 <= d1) & (d1 <= dot_product(e1, e1, dim=-1, keepdim=True))
        d2 = dot_product(y-s2, x, dim=-1, keepdim=True)
        m2 = (0 <= d2) & (d2 <= dot_product(x, x, dim=-1, keepdim=True))

        return (m1 & m2).float(), y+s1, d1 / dot_product(e1, e1, dim=-1, keepdim=True), d2 / dot_product(x, x, dim=-1, keepdim=True)

    cell_to_paths = defaultdict(list)
    cell_to_enters = defaultdict(list)
    cell_to_exits = defaultdict(list)
    grid4 = squish(grid4)
    grid4_offset = torch.cat((grid4[...,1:,:], grid4[...,:1,:]), -2)
    gridl = grid4_offset - grid4
    gridp = torch.stack((gridl[...,1], -gridl[...,0]), -1)
    prev_ind = None
    prev_p = None

    def attempt_add(ctp, cell, point):
        if len(ctp[cell][0]) == 0:
            ctp[cell][-1].append(point)
            return
        first = ctp[cell][0][0]
        prev = ctp[cell][-1][-1] if len(ctp[cell][-1]) > 0 else first
        if torch.minimum((prev-point).norm(p=2, dim=-1), (first-point).norm(p=2, dim=-1)) <= 1e-5:
            return

        if len(ctp[cell][-1]) >= 2:
            p1, p2 = ctp[cell][-1][-2], ctp[cell][-1][-1]
            a = p2-p1
            a = F.normalize(a, p=2, dim=-1)
            pd = point - p1
            if (pd - dot_product(pd, a) * a).norm(p=2, dim=-1) <= 1e-6:
                ctp[cell][-1] = ctp[cell][-1][:-1]
        ctp[cell][-1].append(point)

    ignore_inds = []
    move_from_ind = None

    def get_cell_hits(g, go, p1, p2, ind):
        cell_walls = g[ind]
        cell_walls_offset = go[ind]
        hit_walls, intx, hit_portion, hit_portion2 = intersect_line_segments(cell_walls, cell_walls_offset, p1, p2)
        hit_walls = hit_walls.argmax(0).item()
        intx = intx[hit_walls]
        hit_portion = hit_portion[hit_walls]
        return hit_walls, intx, hit_portion

    def get_cell_hits_multi(g, go, p1, p2):
        cell_walls = g
        cell_walls_offset = go
        return intersect_line_segments(cell_walls, cell_walls_offset, p1, p2)

    curve_begin_ind = None
    curve_begin_coords = None

    def add_edge(prev_p, pp, prev_ind, hit_ind):
        prev_hit_walls, prev_intx, prev_hit_portion = get_cell_hits(grid4, grid4_offset, prev_p, pp, prev_ind)
        w = len(grid_x) - 1
        now_hit_walls, now_intx, now_hit_portion = get_cell_hits(grid4, grid4_offset, prev_p, pp, hit_ind)
        attempt_add(cell_to_paths, prev_ind, prev_intx)
        cell_to_exits[prev_ind].append((prev_hit_walls, prev_hit_portion))
        cell_to_enters[hit_ind].append((now_hit_walls, now_hit_portion))
        attempt_add(cell_to_paths, hit_ind, now_intx)

    for i in list(range(len(perimeter_points))):
        pp = perimeter_points[i]
        if pp.amin(-1) < -1e11:
            ignore_inds.append(prev_ind)
            if prev_ind is not None:
                if prev_ind != curve_begin_ind:
                    cell_to_paths[curve_begin_ind].append([])
                    add_edge(prev_p, curve_begin_coords, prev_ind, curve_begin_ind)
                    prev_ind = curve_begin_ind
                move_from_ind = prev_ind
            prev_ind = None
            continue
        hits = (dot_product(pp - grid4, gridp, dim=-1, keepdim=True) <= 0).all(-2)
        if hits.sum(0) == 0:
            continue
        if hits.sum(0) > 1 and prev_p is not None:
            ds = dot_product(pp - prev_p, grid4-prev_p, dim=-1, keepdim=True).amax(-2)
            m = hits.float()
            hit_ind = (ds * m + (1-m) * -1e12).argmax(0).item()
        else:
            hit_ind = hits.float().argmax(0, keepdim=True).item()
        if curve_begin_ind is None:
            curve_begin_ind = hit_ind
            curve_begin_coords = pp
        if hit_ind != prev_ind:
            cell_to_paths[hit_ind].append([])
            if prev_ind is not None:
                dx = abs((hit_ind // (len(grid_x)-1)) - (prev_ind // (len(grid_x)-1)))
                dy = abs((hit_ind % (len(grid_x)-1)) - (prev_ind % (len(grid_x)-1)))
                if not ((dx <= 1 and dy == 0) or (dy <= 1 and dx == 0)):
                    hit_walls, intx, hit_portion, hit_portion_2 = get_cell_hits_multi(grid4, grid4_offset, prev_p, pp)
                    int_inds = hit_walls.sum(1).squeeze(-1).nonzero().view(-1)
                    for int_ind in int_inds:
                        if int_ind in [prev_ind, hit_ind]:
                            continue
                        if hit_walls[int_ind].sum() < 1.5:
                            continue
                        sorted_ps, argsort_ps = hit_portion_2[int_ind].view(-1).sort()
                        argsort_ps = argsort_ps[(~sorted_ps.isnan() & (0 <= sorted_ps) & (sorted_ps <= 1))]
                        if len(argsort_ps) <= 1:
                            continue
                        cell_to_exits[int_ind.item()].append((argsort_ps[1].item(), hit_portion[int_ind, argsort_ps[1]]))
                        cell_to_enters[int_ind.item()].append((argsort_ps[0].item(), hit_portion[int_ind, argsort_ps[0]]))
                        cell_to_paths[int_ind.item()].append([intx[int_ind, argsort_ps[0]], intx[int_ind, argsort_ps[1]]])
                add_edge(prev_p, pp, prev_ind, hit_ind)
            else:
                cell_to_enters[hit_ind].append((torch.tensor((-1,)), torch.tensor((-1,))))
                if move_from_ind is not None:
                    cell_to_exits[move_from_ind].append((torch.tensor((5,)), torch.tensor((5,))))
                    move_from_ind = None
                    curve_begin_ind = hit_ind
                    curve_begin_coords = pp

        attempt_add(cell_to_paths, hit_ind, pp)
        prev_ind = hit_ind
        prev_p = pp
    cell_to_exits[curve_begin_ind].append((torch.tensor((5,)), torch.tensor((5,))))
    if len(cell_to_exits[prev_ind]) < len(cell_to_enters[prev_ind]):
        cell_to_exits[prev_ind].append((torch.tensor((5,)), torch.tensor((5,))))
    end_ind = hit_ind

    for c in cell_to_paths:
        cell_to_paths[c] = [l for l in cell_to_paths[c] if len(l) > 0]

    def get_peri_dist(wp):
        wall_ind, portion = wp
        return wall_ind + portion
    all_polygons = []
    all_grid_ids = []
    total_num_polygons = 0
    for c in cell_to_paths:
        pee = list(zip(*(cell_to_paths[c], cell_to_enters[c], cell_to_exits[c])))
        polygons = [[]]
        current_ind = 0
        initial_inds = [j for j, (p, s, e) in enumerate(pee) if get_peri_dist(s) < -0.5]
        if len(initial_inds) > 0:
            current_ind = initial_inds[0]
            initial_inds = initial_inds[1:]
        used_paths = []
        first_enter = None

        while True:
            try:
                path, enter, exit = pee[current_ind]
            except IndexError:
                path, enter, exit = pee[current_ind]
            if first_enter is None:
                first_enter = enter
            polygons[-1].extend(path)
            if exit[0] > 4.5:
                prev_end = path[-1]
                closest_j = -1
                closest_dist = 1e12
                for j, (pathj, enterj, exitj) in enumerate(pee):
                    if j in used_paths + [current_ind]:
                        continue
                    dist = (torch.stack(pathj) - prev_end).norm(p=2,dim=-1).amin(0)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_j = j
                if closest_j != -1:
                    used_paths.append(current_ind)
                    current_ind = closest_j
                    first_enter = None
                    polygons.append([])
                    continue
            s, e = (get_peri_dist(_) for _ in (first_enter, exit))
            next_enters = []
            for j, (pathj, enterj, exitj) in enumerate(pee):
                if j in used_paths + [current_ind]:
                    continue
                if get_peri_dist(enterj) < -0.5:
                    continue
                if get_peri_dist(exitj) > 4.5:
                    if s > -0.5:
                        continue
                if s < -0.5:
                    next_enters.append([j, enterj])
                    continue
                q = get_peri_dist(enterj)
                ordered = list(sorted([(q, 0), (s, 1), (e, 2)], key=lambda x: x[0]))
                for i, (v, k) in enumerate(ordered):
                    if k == 0:
                        if ordered[(i+1)%len(ordered)][1] == 2:
                            next_enters.append([j, enterj])
                            break
                """for i in range(4):
                    ep = (e - i) % 4
                    if i == 0:
                        if ((exit[0] - i) % 4 == first_enter[0]):
                            if s <= q <= (e - i) % 4:
                                next_enters.append([j, enterj])
                                break
                        elif ((ep - 1) <= q <= (ep)) or (ep + 3 <= q <= ep + 4):
                            next_enters.append([j, enterj])
                            break
                    if ((exit[0] - i) % 4 == first_enter[0]):
                        if s <= q <= (e-i)%4:
                            next_enters.append([j, enterj])
                        break
                    elif ((ep - 1) <= q <= (ep)) or (ep+3 <= q <= ep+4):
                        next_enters.append([j, enterj])
                        break"""

            used_paths.append(current_ind)

            def add_corners(s, e):
                if e[0] < -0.5 and s[0] > 4.5:
                    return
                s = s[0] + s[1]
                e = e[0] + e[1]
                if e > s:
                    e = e - 4

                sf = s.ceil().long()

                for i in range(4):
                    k = (sf - (i + 1))
                    if k <= e:
                        break
                    polygons[-1].append(grid4[c, k.item() % 4])

                """for i in range(3):
                    if (i == 0) and ((s[0]-i) % 4 == e[0]) and ((get_peri_dist(s) - i)%4 >= get_peri_dist(e)):
                        break
                    if ((s[0]-i) % 4 == e[0]):
                        break
                    polygons[-1].append(grid4[c, (s[0]-i) % 4])"""
            if len(next_enters) == 0:
                add_corners(exit, first_enter)
                if len(initial_inds) > 0:
                    current_ind = initial_inds[0]
                    initial_inds = initial_inds[1:]
                    break
                for k in range(len(pee)):
                    if k not in used_paths:
                        current_ind = k
                        break
                if len(used_paths) == len(pee):
                    break
                first_enter = None
                polygons.append([])
                continue
            e = get_peri_dist(exit)
            next_enter = list(sorted(next_enters, key=lambda x: (get_peri_dist(x[1])-e)%4))[-1]
            add_corners(exit, next_enter[1])
            current_ind = next_enter[0]
        total_num_polygons += len(polygons)

        def shift(_):
            return _
        ps = [shift(torch.stack(polygon)) for polygon in polygons if len(polygon) >= 3]
        all_polygons.append(ps)
        all_grid_ids.append(c)

    grid_interior_mask = grid_interior_mask.cpu().view(-1)
    for i in grid_interior_mask.nonzero().view(-1):
        rix = (i % len(grid_x))
        riy = (i // len(grid_x))
        if (rix == (len(grid_x)-1)) or (riy == (len(grid_y)-1)):
            continue
        if (i+len(grid_x)+1) >= grid_interior_mask.shape[0]:
            continue
        ri = riy * (len(grid_x)-1) + rix

        if (sum([grid_interior_mask[j] for j in [i+1, i+len(grid_x), i+len(grid_x)+1]]) > 2.5):
            all_polygons.append([grid4[ri]])
            all_grid_ids.append(ri.item())

    out = [*triangulate_simple_polygon(all_polygons)]
    out = [_.to(DEFAULT_DEVICE, non_blocking=True) for _ in out]
    torch.set_default_device(DEFAULT_DEVICE)
    out[1] = [out[1], torch.tensor(all_grid_ids), (len(grid_x) - 1), len(grid_y)-1]
    return out


def tile_region2(perimeter_points, perimeter_normals=None, tile_size=20, random_perturbation=0, color=WHITE, **kwargs):
    """
    perimeter_points: Tensor[num_points, 2]: collection of points outlining the perimeter of the region to be tiled.
    perimeter_normals: Tensor[num_points, 2]: unit vectors pointing in direction from perimeter_point out of the region.
    tile_size: size of each tile.
    random_perturbation: strength of random perturbation applied to tile corners.
    """

    m = (perimeter_points > -1e11).float()
    mn_corner, mx_corner = (perimeter_points * m + (1-m) * 1e12).amin(0), perimeter_points.amax(0)
    bounding_width, bounding_height = mx_corner - mn_corner
    grid_x = torch.linspace(mn_corner[0], mx_corner[0], steps=int(bounding_width // tile_size)+2)
    grid_y = torch.linspace(mn_corner[1], mx_corner[1], steps=int(bounding_height // tile_size)+2)
    grid = torch.stack((grid_x.view(1,-1).expand(len(grid_y),-1), grid_y.view(-1, 1).expand(-1, len(grid_x))), -1)
    te = tile_size*0.5 - 1e-5
    grid = grid + (torch.randn_like(grid) * random_perturbation).clamp_(min=-te, max=te)

    prev_loc = torch.tensor((-1e12, -1e12))
    kept_inds = []
    for i in range(len(perimeter_points)):
        d = (perimeter_points[i] - prev_loc).norm(p=2, dim=-1)
        if d > 1e-3:
            kept_inds.append(i)
            prev_loc = perimeter_points[i]

    perimeter_points = perimeter_points[kept_inds]
    perimeter_normals = perimeter_normals[kept_inds]

    perimeter_points = perimeter_points.unsqueeze(-2).unsqueeze(-2)

    edge_polygons = get_points_per_tile(grid, perimeter_points)

    perimeter_normals = perimeter_normals.unsqueeze(-2).unsqueeze(-2)
    dists = (perimeter_points - grid).norm(p=2,dim=-1, keepdim=True)
    closest_perimeter_ind = dists.argmin(0, keepdim=True)
    closest_normal = broadcast_gather(perimeter_normals, 0, closest_perimeter_ind, keepdim=False)
    closest_point = broadcast_gather(perimeter_points, 0, closest_perimeter_ind, keepdim=False)
    mask = (dot_product(grid - closest_point, closest_normal, -1) <= 1e-6).float()
    color = color.unsqueeze(0).unsqueeze(0).expand(grid.shape[0], grid.shape[1], -1)

    grid = torch.cat((torch.zeros_like(grid[...,:1]), grid.flip(-1)), -1)

    inds = torch.arange((grid.shape[0]-1)*(grid.shape[1]-1))
    n = grid.shape[1]-1
    x = inds % n
    y = inds // n
    n += 1
    inds = torch.stack([x+y*n,x+1+y*n,x+1+(y+1)*n,
                        x+1+(y+1)*n, x + (y+1)*n, x+y*n], -1).unsqueeze(-1).unsqueeze(1)

    def get_inds(g):
        return squish(unsquish(broadcast_gather(squish(g).unsqueeze(-2).unsqueeze(0), 1, inds, keepdim=False), -2, 3))
    corners, colors, transparencies = [get_inds(_) for _ in [grid, color, mask]]
    pp = squish(edge_polygons.unsqueeze(2).expand(-1,-1,2,-1,-1),0,2)
    pp = torch.cat((torch.zeros_like(pp[...,:1]), pp.flip(-1)), -1)
    return TriangleTriangulated(corners, color=colors, transparency=torch.zeros_like(1 - transparencies), perimeter_poins=pp)

def cubic_bezier_eval(p, t):
    return ((1 - t) ** 3) * p[:, 0] + 3 * ((1 - t) ** 2) * t * p[:, 1] + 3 * (1 - t) * t * t * p[:, 2] + (t ** 3) * p[:, 3]


def cubic_bezier_derivative_eval(p, t):
    p0 = p[:, 0]
    p1 = p[:, 1]
    p2 = p[:, 2]
    p3 = p[:, 3]
    return 3*((1-t)**2)*(p1-p0) + 6*(1-t)*t*(p2-p1) + 3*(t*t)*(p3-p2)


def point_to_tensor2(point):
    return torch.tensor((point.x, point.y))


def point_to_tensor(point):
    return torch.tensor((2, point.y, point.x))


def get_roots_of_l2_proj_on_cubic_bezier(a, b, c, d):
    rc = get_roots_of_cubic(a, b, c, d, fill_value=0)
    rq = get_roots_of_quadratic(3*a,2*b,c, fill_value=0)
    return torch.cat((expand_as_left(rq.clamp_(0, 1), rc), rc.clamp_(0, 1), torch.zeros_like(rc[..., :1]), torch.ones_like(rc[..., :1])), -1)

eps=1e-12


def params_to_tensor(params):
    p = [point_to_tensor2(_) for _ in params]
    return torch.stack(p, 0).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)


num_points_per_curve = 20


def get_points_along_cubic_bezier(params, invert=False):
    p = params.unsqueeze(0)

    roots = torch.linspace(0, 1, num_points_per_curve+1)
    if invert:
        roots = roots.flip(-1)
    roots = roots[:num_points_per_curve]
    critical_points = cubic_bezier_eval(p.unsqueeze(-1), roots)
    parallel_vec = cubic_bezier_derivative_eval(p.unsqueeze(-1), roots)
    parallel_vec = F.normalize(parallel_vec, p=2, dim=-1, eps=eps)
    perp_vec = torch.stack([-parallel_vec[..., 1, :], parallel_vec[..., 0, :]], -2)
    if invert:
        perp_vec *= -1
    return critical_points.squeeze(0).squeeze(0).squeeze(0).squeeze(0).t(), perp_vec.squeeze(0).squeeze(0).squeeze(0).squeeze(0).t()


def get_points_along_line(params, invert=False):
    p = params.unsqueeze(0)
    b = p[:, -1]
    a = p[:, 0]
    v = (b - a)
    parallel_vec = v
    perp_vec = torch.stack([-parallel_vec[..., 1], parallel_vec[..., 0]], -1)
    if invert:
        perp_vec *= -1

    t = torch.linspace(0, 1, num_points_per_curve+1)
    if invert:
        t = t.flip(-1)
    t = t[: num_points_per_curve].unsqueeze(-1)
    points = a + v * t
    return points.squeeze(0).squeeze(0).squeeze(0), perp_vec.squeeze(0).squeeze(0).squeeze(0).expand(points.shape[-2], -1)


def project_onto_cubic_bezier(params, point, invert=False):
    p = params

    roots = torch.linspace(0, 1, 20)
    critical_points = cubic_bezier_eval(p.unsqueeze(-1), roots)

    dists = (critical_points - point.unsqueeze(-1))
    dists = dists.square_().sum(-2, keepdim=True)
    closest_dist, closest_ind = dists.min(-1, keepdim=True)
    closest_point = broadcast_gather(critical_points, -1, closest_ind, keepdim=False)
    closest_root = broadcast_gather(unsqueeze_left(roots, closest_ind), -1, closest_ind, keepdim=False)

    t = closest_ind.squeeze(-1) / (roots.shape[0]-1)

    parallel_vec = cubic_bezier_derivative_eval(p, closest_root)
    parallel_vec = F.normalize(parallel_vec, p=2, dim=-1, eps=eps)
    perp_vec = torch.stack([-parallel_vec[..., 1], parallel_vec[..., 0]], -1)
    if invert:
        perp_vec *= -1
    return dot_product(F.normalize(point - closest_point, p=2, dim=-1), perp_vec, -1, keepdim=True), closest_dist.squeeze(-1).sqrt_(), t


def project_onto_line(params, point, invert=False):
    p = params
    b = p[:, -1]
    a = p[:, 0]
    v = (b - a)
    v_len = v.norm(p=2, dim=-1, keepdim=True)
    v = F.normalize(v, p=2, dim=-1, eps=eps)
    t = dot_product(v, point-a, -1, keepdim=True).clamp_(min=torch.tensor((0,)),max=v_len)
    closest_point = v * t + a
    closest_dist = (point - closest_point).norm(p=2,dim=-1)
    parallel_vec = v
    perp_vec = torch.stack([-parallel_vec[...,1], parallel_vec[...,0]], -1)
    if invert:
        perp_vec *= -1
    return dot_product(F.normalize(point - closest_point, p=2, dim=-1), perp_vec, -1, keepdim=True), closest_dist.unsqueeze(-1), t / v_len


class TriangulatedBezierCircuit(Mob):
    def __init__(self, paths, invert=False, border_width=0.1, tile_size=0.015, debug=False, hash_keys=None, use_cache=True,
                 reverse_points=False, color=WHITE, create_direction=F.normalize(RIGHT*2+DOWN, p=2, dim=-1), *args, **kwargs):
        self.invert = invert

        self.funcs = []
        self.cubic_params = []
        self.linear_params = []
        self.create_direction = create_direction

        def get_center(x):
            mn = x.amin((0, 1), keepdim=True)
            mx = x.amax((0, 1), keepdim=True)
            return (mx + mn) / 2

        if not (isinstance(paths, list) or isinstance(paths, tuple)):
            paths = [paths]
            hash_keys = [hash_keys]
        all_triangles = []
        all_tiles = []
        all_pack_counts = []
        if hash_keys is None:
            hash_keys = [None for _ in range(len(paths))]
        for path, hash_key in zip(paths, hash_keys):
            found_hash = False
            just_moved = False
            if hash_key is not None:
                n = 12
                hash_key = torch.from_numpy(hash_key).to(DEFAULT_DEVICE)
                offset = hash_key.amin(0)
                hash_key = hash_key - offset
                hash_key = (hash_key.round(decimals=n) * (10**n)).long()
                hash_bytes = torch.cat((torch.tensor((1 if invert else 0,), dtype=torch.long),
                                        (torch.tensor((tile_size,)).round(decimals=n) * (10**n)).long(), hash_key.view(-1)))
                hash_bytes = ''.join([str(_.item()) for _ in hash_bytes.cpu()])

                hasher = hashlib.sha256()
                hasher.update(hash_bytes.encode())
                hash_bytes = hasher.hexdigest()[:32]
                file_path = os.path.join(DEFAULT_DIR, 'algan_cache', f'{hash_bytes}.txt')
                if os.path.exists(file_path):
                    tiles, tile_counts = torch.load(file_path, map_location=DEFAULT_DEVICE)
                    tiles = tiles + offset.float()[:2]
                    found_hash = True

            points = []
            if (not use_cache) or (use_cache and not found_hash):
                for i, element in enumerate(path if not self.invert else reversed(path)):
                    if isinstance(element, svgelements.Move):
                        if i == 0:
                            continue
                        just_moved = True
                        points[-1][0][-1] = -2e12
                        continue
                    elif isinstance(element, svgelements.Line) or isinstance(element, svgelements.Close):
                        params = (params_to_tensor([element.start, element.end]))
                        points.append(get_points_along_line(params, invert=self.invert))
                    elif isinstance(element, svgelements.CubicBezier):
                        params = (params_to_tensor([element.start, element.control1, element.control2, element.end]))
                        points.append(get_points_along_cubic_bezier(params, invert=self.invert))
                    else:
                        raise NotImplementedError(f'{type(element)}')
                    if just_moved:
                        points[-1][0][0] = -2e12
                        just_moved = False

                self.num_curves = len(points)

                points, normals = [torch.cat(_) for _ in zip(*points)]
                tiles, tile_counts = tile_region(points, tile_size=tile_size, reverse_points=reverse_points)
                if tiles is None:
                    continue
                if hash_key is not None:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save((tiles - tiles.amin((0,1), keepdim=True), tile_counts), file_path)

            tile_sizes, tile_grid_id, grid_width, grid_height = tile_counts

            k = 2
            k_id = (tile_grid_id % grid_width) // k + ((tile_grid_id // grid_width) // k) * ((grid_width // k) + 1)
            k_id = k_id.unique(return_inverse=True)[1]

            tiles, pack_counts = packed_reorder(tiles, tile_sizes, k_id)
            triangle_corners = torch.cat((tiles, torch.zeros_like(tiles[...,:1])), -1)
            tile_centers = torch.cat([get_center(_) for _ in unpack_tensor(triangle_corners, (pack_counts, 2))])
            all_tiles.append(tile_centers)
            all_triangles.append(triangle_corners)
            all_pack_counts.append(pack_counts)
        tiles = torch.cat(all_tiles) if len(all_tiles) > 0 else torch.tensor((0,0,0)).view(1,1,3)
        triangles = torch.cat(all_triangles) if len(all_triangles) > 0 else torch.tensor((0,0,0)).view(1,1,3).expand(-1,3,-1)
        #create = True
        #if 'create' in kwargs:
        #    create = kwargs['create']
        #    del kwargs['create']

        animate_creation = True
        if 'animate_creation' in kwargs:
            animate_creation = kwargs['animate_creation']
            del kwargs['animate_creation']
        super().__init__(*args, **kwargs)
        self.border_width = 0.1
        self.animatable_attrs.update({'fill_portion'})
        self.debug = debug
        self.fill_portion = 1
        self.color = color

        self.location = torch.stack([get_center(_).squeeze(0) for _ in all_tiles]).squeeze(1)

        packing = torch.cat(all_pack_counts)
        self.tiles = Mob(location=tiles.squeeze(1), parent_batch_sizes=(torch.tensor([len(_) for _ in all_tiles])), **kwargs)
        triangles = TriangleTriangulated(triangles.squeeze(1), color=color, parent_batch_sizes=packing, **kwargs)
        self.tiles.add_children(triangles)
        self.add_children(self.tiles)
        #if create and not self.animation_manager.context.delay_creation:
        #    self.spawn(animate_creation)
        self.parents = []

    def highlight(self):
        self.color = RED_A
        return self

    def highlight_off(self):
        self.color = WHITE
        return self

    def get_local_coord_bounding_box(self):
        all_points = torch.stack([point_to_tensor2(_.end) for _ in self.path], 0)
        mn, mx = (all_points.amin(-2)), (all_points.amax(-2))
        return torch.stack((mn, torch.stack((mn[...,0], mx[...,1]), -1), mx, torch.stack((mx[...,0], mn[...,1]), -1)), -1).unsqueeze(-3)
