"""How many BVH nodes does a ray actually visit? -- ssE's missing instrument.

Nothing in this repo counts traversal steps, so two standing claims about BVH
ORDERING cannot be confirmed or refuted: the bezier median split's inherited
"~20-25% fewer traversal steps" (ssF), and the general case for instancing.
Wall-clock cannot settle either on this machine -- it throttles hard enough that
a control kernel the change cannot touch drifts as much as the target -- but a
step count is DETERMINISTIC, which is exactly what a machine without a usable
clock needs.

WHAT IS COUNTED
----------------
Per ray, for one geometry type's STBVH:

    groups   sibling-block tests (``_group_test``) -- one aligned fetch that
             tests all bvh_arity children of a node at once. This is the
             traversal step: it is what a tighter box removes.
    leaves   leaf slots reached (bvh_leaf_size per leaf visited)
    prims    primitive intersections actually performed -- slots holding a live
             instance for this frame. The second thing a tighter box removes.

HOW IT IS KEPT HONEST
----------------------
The walk here is a transcription of ``_nearest_triangle_hit`` /
``_nearest_bezier_hit`` with counters added: same ``_group_test``, same
``_nearest_pending_child``, same ``best_t`` pruning, same leaf predicate, and
the leaf bodies call the SAME production ``@ti.func``s. A transcription can
still drift, so ``--verify`` runs the production function over the identical
rays and arrays and compares the hit it returns:

    same (t, prim) for every ray  =>  the same nodes were visited,

because the only thing that decides which nodes a ray enters is the block data,
the ray, and the sequence of ``best_t`` updates -- and the last of those is
observable as the hit. A transcription that agreed on nothing but the count
would be measuring itself (ss0.1 rule 2).

The rays are the scene's own PRIMARY rays, generated with the production
``_generate_ray`` from the captured camera arrays, so the coherence is a real
render's rather than a random sample's. ``--random N`` adds an incoherent set
aimed at the root box, which is the harder case for any ordering.

Usage:
    <venv-python> benchmarks/_bvh_steps.py --verify              # bez scene
    <venv-python> benchmarks/_bvh_steps.py --scene solids --geom tri
    <venv-python> benchmarks/_bvh_steps.py --compare bez_split   # A vs B
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from algan import LD, MD, SETTINGS, Off, Scene  # noqa: E402,F401
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing.raytrace_kernels_taichi import (  # noqa: E402
    _GROUP_MASK,
    _GROUP_STACK,
    _M_BASIS_U,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    _M_NORMAL,
    NODE_ARG,
    _bezier_point_metrics,
    _circuit_point_region,
    _circuit_query_radius,
    _comes_after,
    _generate_ray,
    _group_test,
    _nearest_bezier_hit,
    _nearest_pending_child,
    _nearest_triangle_hit,
    _tri_hit,
    bvh_arity,
    bvh_leaf_size,
    depth_tie_epsilon,
    min_hit_distance,
)
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.taichi_compat import ti  # noqa: E402

DEBUG = bool(os.environ.get("STEPS_DEBUG"))
PINNED_BYTES = 1_400_000_000
FONT = "Algan Test Sans"


# ---------------------------------------------------------------- the walks


@ti.kernel
def _steps_tri(
    n: ti.i32,
    f: ti.i32,
    rays: ti.types.ndarray(),
    blocks: NODE_ARG,
    leaf_prim: ti.types.ndarray(),
    leaf_tspan: ti.types.ndarray(),
    first_leaf: ti.i32,
    tri_pos: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """``_nearest_triangle_hit`` (refit == 0) with three counters added.

    Every line that decides which node is entered next is the production
    line; ``out`` carries (groups, leaves, prims, best_t, best_prim) so
    ``--verify`` can compare the hit as well as the cost.
    """
    for r in range(n):
        ro = ti.math.vec3(rays[r, 0], rays[r, 1], rays[r, 2])
        rd = ti.math.vec3(rays[r, 3], rays[r, 4], rays[r, 5])
        inv_rd = ti.math.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
        t_prev = 0.0
        layer_prev = 1e30
        t_cap = 1e30
        best_t = 1e30
        best_layer = -1e30
        best_prim = -1
        n_grp = 0
        n_leaf = 0
        n_prim = 0
        tp = f % tri_pos.shape[0]
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            0,
            0,
            0,
            f,
            ro,
            inv_rd,
            t_prev - depth_tie_epsilon,
            ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
            blocks,
        )
        n_grp += 1
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> bvh_arity
                saved_mask = saved & _GROUP_MASK
                fresh_mask, g_near = _group_test(
                    0,
                    0,
                    g_cur,
                    f,
                    ro,
                    inv_rd,
                    t_prev - depth_tie_epsilon,
                    ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
                    blocks,
                )
                n_grp += 1
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                descend = 0
                child_blk = 0
                l_base = 0
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
                if descend == 0:
                    n_leaf += bvh_leaf_size
                    for j in ti.static(range(bvh_leaf_size)):
                        prim = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if (
                            (p0 >= 0)
                            and ((tspan & 0x7FFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))
                        ):
                            prim = p0
                        if prim >= 0:
                            n_prim += 1
                            v0 = ti.math.vec3(
                                tri_pos[tp, prim, 0],
                                tri_pos[tp, prim, 1],
                                tri_pos[tp, prim, 2],
                            )
                            v1 = ti.math.vec3(
                                tri_pos[tp, prim, 3],
                                tri_pos[tp, prim, 4],
                                tri_pos[tp, prim, 5],
                            )
                            v2 = ti.math.vec3(
                                tri_pos[tp, prim, 6],
                                tri_pos[tp, prim, 7],
                                tri_pos[tp, prim, 8],
                            )
                            hit_ok, w1, w2, t = _tri_hit(ro, rd, v0, v1, v2)
                            if hit_ok != 0:
                                layer = ti.cast(prim, ti.f32)
                                if (
                                    (t > min_hit_distance)
                                    and _comes_after(t, layer, t_prev, layer_prev)
                                    and _comes_after(best_t, best_layer, t, layer)
                                ):
                                    best_t = t
                                    best_layer = layer
                                    best_prim = prim
                else:
                    if g_pend != 0:
                        g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    g_pend, g_near = _group_test(
                        0,
                        0,
                        g_cur,
                        f,
                        ro,
                        inv_rd,
                        t_prev - depth_tie_epsilon,
                        ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
                        blocks,
                    )
                    n_grp += 1
        out[r, 0] = ti.cast(n_grp, ti.f32)
        out[r, 1] = ti.cast(n_leaf, ti.f32)
        out[r, 2] = ti.cast(n_prim, ti.f32)
        out[r, 3] = best_t
        out[r, 4] = ti.cast(best_prim, ti.f32)


@ti.kernel
def _steps_bez(
    n: ti.i32,
    f: ti.i32,
    pixel_size_per_t: ti.f32,
    rays: ti.types.ndarray(),
    blocks: NODE_ARG,
    leaf_prim: ti.types.ndarray(),
    leaf_tspan: ti.types.ndarray(),
    first_leaf: ti.i32,
    circuit_meta: ti.types.ndarray(),
    edges_2d: ti.types.ndarray(),
    edge_accel: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """``_nearest_bezier_hit`` (refit == 0) with the same three counters."""
    for r in range(n):
        ro = ti.math.vec3(rays[r, 0], rays[r, 1], rays[r, 2])
        rd = ti.math.vec3(rays[r, 3], rays[r, 4], rays[r, 5])
        inv_rd = ti.math.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
        t_prev = 0.0
        layer_prev = 1e30
        t_cap = 1e30
        base_dist = 0.0
        best_t = 1e30
        best_layer = -1e30
        best_circuit = -1
        n_grp = 0
        n_leaf = 0
        n_prim = 0
        num_meta_frames = circuit_meta.shape[0]
        num_edge_frames = edges_2d.shape[0]
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            0,
            0,
            0,
            f,
            ro,
            inv_rd,
            t_prev - depth_tie_epsilon,
            ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
            blocks,
        )
        n_grp += 1
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> bvh_arity
                saved_mask = saved & _GROUP_MASK
                fresh_mask, g_near = _group_test(
                    0,
                    0,
                    g_cur,
                    f,
                    ro,
                    inv_rd,
                    t_prev - depth_tie_epsilon,
                    ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
                    blocks,
                )
                n_grp += 1
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                descend = 0
                child_blk = 0
                l_base = 0
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
                if descend == 0:
                    n_leaf += bvh_leaf_size
                    for j in ti.static(range(bvh_leaf_size)):
                        circuit = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if (
                            (p0 >= 0)
                            and ((tspan & 0x7FFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))
                        ):
                            circuit = p0
                        if circuit >= 0:
                            n_prim += 1
                            tm = f % num_meta_frames
                            nrm = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_NORMAL],
                                circuit_meta[tm, circuit, _M_NORMAL + 1],
                                circuit_meta[tm, circuit, _M_NORMAL + 2],
                            )
                            denom = rd.dot(nrm)
                            layer = ti.cast(circuit, ti.f32)
                            if ti.abs(denom) > 1e-9:
                                center = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_CENTER],
                                    circuit_meta[tm, circuit, _M_CENTER + 1],
                                    circuit_meta[tm, circuit, _M_CENTER + 2],
                                )
                                t = (center - ro).dot(nrm) / denom
                                if (
                                    (t > min_hit_distance)
                                    and _comes_after(t, layer, t_prev, layer_prev)
                                    and _comes_after(best_t, best_layer, t, layer)
                                ):
                                    hit = ro + t * rd - center
                                    bu = ti.math.vec3(
                                        circuit_meta[tm, circuit, _M_BASIS_U],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 2],
                                    )
                                    bv = ti.math.vec3(
                                        circuit_meta[tm, circuit, _M_BASIS_U + 3],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 4],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 5],
                                    )
                                    u = hit.dot(bu)
                                    v = hit.dot(bv)
                                    pixel_size = pixel_size_per_t * (base_dist + t)
                                    border_w = (
                                        circuit_meta[tm, circuit, _M_BORDER_W]
                                        * pixel_size
                                    )
                                    outline_w = 0.6 * pixel_size
                                    filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
                                    query_radius = _circuit_query_radius(
                                        border_w, outline_w, filled
                                    )
                                    te = f % num_edge_frames
                                    (
                                        crossings,
                                        min_dist_sq,
                                        _ccu,
                                        _ccv,
                                        _e1x,
                                        _e1y,
                                        _sg1,
                                        _s2,
                                        _s2u,
                                        _s2v,
                                        _e2x,
                                        _e2y,
                                        _sg2,
                                    ) = _bezier_point_metrics(
                                        circuit,
                                        te,
                                        u,
                                        v,
                                        query_radius,
                                        circuit_meta.shape[1],
                                        edges_2d,
                                        edge_accel,
                                    )
                                    inside, _in_border = _circuit_point_region(
                                        border_w,
                                        outline_w,
                                        filled,
                                        crossings,
                                        min_dist_sq,
                                    )
                                    if inside:
                                        best_t = t
                                        best_layer = layer
                                        best_circuit = circuit
                else:
                    if g_pend != 0:
                        g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    g_pend, g_near = _group_test(
                        0,
                        0,
                        g_cur,
                        f,
                        ro,
                        inv_rd,
                        t_prev - depth_tie_epsilon,
                        ti.min(best_t + depth_tie_epsilon, t_cap + depth_tie_epsilon),
                        blocks,
                    )
                    n_grp += 1
        out[r, 0] = ti.cast(n_grp, ti.f32)
        out[r, 1] = ti.cast(n_leaf, ti.f32)
        out[r, 2] = ti.cast(n_prim, ti.f32)
        out[r, 3] = best_t
        out[r, 4] = ti.cast(best_circuit, ti.f32)


# ------------------------------------------------------- the reference walks


@ti.kernel
def _ref_tri(
    n: ti.i32,
    f: ti.i32,
    rays: ti.types.ndarray(),
    blocks: NODE_ARG,
    node_miss: ti.types.ndarray(),
    leaf_prim: ti.types.ndarray(),
    leaf_tspan: ti.types.ndarray(),
    first_leaf: ti.i32,
    tri_pos: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """The PRODUCTION walk over the same rays -- the thing the counter must
    agree with, called rather than re-derived.
    """
    for r in range(n):
        ro = ti.math.vec3(rays[r, 0], rays[r, 1], rays[r, 2])
        rd = ti.math.vec3(rays[r, 3], rays[r, 4], rays[r, 5])
        inv_rd = ti.math.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
        t, prim, _w1, _w2, _layer = _nearest_triangle_hit(
            0,
            ro,
            rd,
            inv_rd,
            f,
            ti.cast(f, ti.f32),
            0.0,
            1e30,
            1e30,
            0.0,
            blocks,
            node_miss,
            leaf_prim,
            leaf_tspan,
            first_leaf,
            tri_pos,
        )
        out[r, 0] = t
        out[r, 1] = ti.cast(prim, ti.f32)


@ti.kernel
def _ref_bez(
    n: ti.i32,
    f: ti.i32,
    pixel_size_per_t: ti.f32,
    rays: ti.types.ndarray(),
    blocks: NODE_ARG,
    node_miss: ti.types.ndarray(),
    leaf_prim: ti.types.ndarray(),
    leaf_tspan: ti.types.ndarray(),
    first_leaf: ti.i32,
    circuit_meta: ti.types.ndarray(),
    edges_2d: ti.types.ndarray(),
    edge_accel: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for r in range(n):
        ro = ti.math.vec3(rays[r, 0], rays[r, 1], rays[r, 2])
        rd = ti.math.vec3(rays[r, 3], rays[r, 4], rays[r, 5])
        inv_rd = ti.math.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
        t, circuit, _b, _u, _v, _layer = _nearest_bezier_hit(
            0,
            ro,
            rd,
            inv_rd,
            f,
            ti.cast(f, ti.f32),
            0.0,
            1e30,
            1e30,
            pixel_size_per_t,
            0.0,
            blocks,
            node_miss,
            leaf_prim,
            leaf_tspan,
            first_leaf,
            circuit_meta,
            edges_2d,
            edge_accel,
        )
        out[r, 0] = t
        out[r, 1] = ti.cast(circuit, ti.f32)


@ti.kernel
def _gen_rays(
    n: ti.i32,
    f: ti.i32,
    stride: ti.i32,
    width: ti.i32,
    half_w: ti.f32,
    half_h: ti.f32,
    cam_origin: ti.types.ndarray(),
    screen_point: ti.types.ndarray(),
    pixel_basis_x: ti.types.ndarray(),
    pixel_basis_y: ti.types.ndarray(),
    rays: ti.types.ndarray(),
):
    """The scene's own primary rays, on a regular stride over the frame."""
    for r in range(n):
        g = r * stride
        px = g % width
        py = g // width
        ro, rd = _generate_ray(
            f,
            px,
            py,
            0.5,
            0.5,
            half_w,
            half_h,
            cam_origin,
            screen_point,
            pixel_basis_x,
            pixel_basis_y,
        )
        rays[r, 0] = ro[0]
        rays[r, 1] = ro[1]
        rays[r, 2] = ro[2]
        rays[r, 3] = rd[0]
        rays[r, 4] = rd[1]
        rays[r, 5] = rd[2]


# ------------------------------------------------------------------ capture


class _Capture:
    """Clone one batch's BVH, geometry and camera arrays out of the arena.

    They are arena tensors, so they are recycled the moment the render moves
    on; anything kept has to be copied while the hook is on the stack.
    """

    def __init__(self):
        self.data = None

    def __enter__(self):
        original = rp.prepare_sparse_raster_coverage

        def spy(*args, **kwargs):
            if self.data is None:

                def arg(name, pos):
                    return kwargs[name] if name in kwargs else args[pos]

                merged = arg("merged", 0)
                keep = {
                    "cam_origin": arg("cam_origin", 5).clone(),
                    "screen_point": arg("screen_point", 6).clone(),
                    "pixel_basis_x": arg("pixel_basis_x", 7).clone(),
                    "pixel_basis_y": arg("pixel_basis_y", 8).clone(),
                    "time_start": int(arg("time_start", 11)),
                    "width": int(arg("width", 13)),
                    "height": int(arg("height", 14)),
                    "half_w": float(arg("half_w", 15)),
                    "half_h": float(arg("half_h", 16)),
                    "num_triangles": int(merged["num_triangles"]),
                    "num_circuits": int(merged["num_circuits"]),
                }
                for geom, bvh_key, prim_keys in (
                    ("tri", "tri_bvh", ("tri_pos",)),
                    ("bez", "bez_bvh", ("circuit_meta", "edges_2d", "edge_accel")),
                ):
                    bvh = merged.get(bvh_key)
                    if bvh is None:
                        continue
                    if DEBUG:
                        print(
                            f"  [dbg] {bvh_key}: {type(bvh).__name__} "
                            f"nodes={tuple(bvh.nodes.shape)} "
                            f"blocks={tuple(bvh.blocks.shape)} "
                            f"leaf_prim={tuple(bvh.leaf_prim.shape)} "
                            f"first_leaf={bvh.first_leaf} "
                            f"deferred={merged.get('bvh_deferred')} "
                            f"ncirc={merged.get('num_circuits')} "
                            f"ntri={merged.get('num_triangles')}"
                        )
                    keep[f"{geom}_nodes"] = bvh.nodes.clone()
                    keep[f"{geom}_blocks"] = bvh.blocks.clone()
                    keep[f"{geom}_node_miss"] = bvh.node_miss.clone()
                    keep[f"{geom}_leaf_prim"] = bvh.leaf_prim.clone()
                    keep[f"{geom}_leaf_tspan"] = bvh.leaf_tspan.clone()
                    keep[f"{geom}_first_leaf"] = int(bvh.first_leaf)
                    for k in prim_keys:
                        v = merged.get(k)
                        keep[k] = v.clone() if torch.is_tensor(v) else v
                self.data = keep
            return original(*args, **kwargs)

        self.original = original
        rp.prepare_sparse_raster_coverage = spy
        return self

    def __exit__(self, *exc):
        rp.prepare_sparse_raster_coverage = self.original
        return False


# ------------------------------------------------------------------- scenes


def _scene_bez():
    """Many small independent circuits, which is what a bezier tree holds.

    A metallic sphere is in the scene for one reason: without a SECONDARY ray
    the batch provably never traverses an STBVH, so the builder defers it and
    hands the tracer a placeholder -- 1 block, 1 leaf slot, no live primitive,
    an all-zero root. A counter pointed at that reports 1 group per ray and no
    hits, agrees with the production walk (which finds nothing either), and
    means nothing at all.
    """
    from algan import (  # noqa: PLC0415
        BLUE,
        DARKER_GRAY,
        GREEN,
        IN,
        OUT,
        RED,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        Circle,
        MeshStandardMaterial,
        Sphere,
        Square,
        Star,
        Sync,
        Tex,
        Text,
        Triangle,
    )

    Scene.set_background(DARKER_GRAY)
    shapes = []
    with Off():
        for i in range(7):
            for j in range(5):
                kind = (i + j) % 4
                mob = (
                    Circle(color=RED).scale(0.32)
                    if kind == 0
                    else Square(color=GREEN).scale(0.3)
                    if kind == 1
                    else Triangle(color=BLUE).scale(0.34)
                    if kind == 2
                    else Star(color=YELLOW).scale(0.34)
                )
                mob.move(RIGHT * ((i - 3) * 1.05) + UP * ((j - 2) * 1.0))
                mob.spawn(animate=False)
                shapes.append(mob)
        Text("bezier bvh ordering", font_size=26, color=WHITE, font=FONT).move(
            UP * -2.6
        ).spawn(animate=False)
        Tex(r"\sum_{i=0}^{n} x_i^2", font_size=30, color=WHITE).move(UP * 2.6).spawn(
            animate=False
        )
        mirror = Sphere(color=WHITE).scale(0.9).move(IN * 2.2)
        mirror.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
        mirror.spawn(animate=False)
    with Sync(duration=0.4):
        for i, mob in enumerate(shapes):
            mob.rotate(20 * (1 + i % 5), OUT)


def _scene_tri():
    """Many small solids, so the triangle tree holds many instances."""
    from algan import (  # noqa: PLC0415
        BLUE,
        DARKER_GRAY,
        GREEN,
        IN,
        ORANGE,
        OUT,
        PURPLE,
        RED,
        RIGHT,
        UP,
        WHITE,
        Cube,
        MeshStandardMaterial,
        Sphere,
        Sync,
    )

    Scene.set_background(DARKER_GRAY)
    solids = []
    tints = (RED, GREEN, BLUE, ORANGE, PURPLE)
    with Off():
        for i in range(6):
            for j in range(4):
                tint = tints[(i + j) % len(tints)]
                mob = (
                    Cube(color=tint).scale(0.28)
                    if (i + j) % 2
                    else Sphere(color=tint).scale(0.3)
                )
                mob.move(RIGHT * ((i - 2.5) * 1.1) + UP * ((j - 1.5) * 1.1))
                mob.spawn(animate=False)
                solids.append(mob)
        # Same reason as _scene_bez: a secondary ray is what makes the builder
        # build the tree instead of deferring it.
        mirror = Sphere(color=WHITE).scale(0.8).move(IN * 2.4)
        mirror.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
        mirror.spawn(animate=False)
        solids.append(mirror)
    with Sync(duration=0.4):
        for i, mob in enumerate(solids):
            mob.rotate(25 * (1 + i % 4), OUT)


SCENES = {"bez": _scene_bez, "tri": _scene_tri}


def _register_test_fonts():
    import importlib.util  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    conftest = Path(__file__).resolve().parent.parent / "tests" / "conftest.py"
    if not conftest.exists():
        return
    spec = importlib.util.spec_from_file_location("_algan_steps_conf", conftest)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        for name in ("_register_bundled_fonts", "register_test_fonts"):
            fn = getattr(module, name, None)
            if callable(fn):
                fn()
                return
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"(font registration skipped: {exc})")


def _capture_scene(which, quality):
    # The walks transcribed here are the ``refit == 0`` ones -- the STBVH, which
    # is also the only tree the median-split ordering under test applies to. A
    # RefitBVH stores its topology as per-(frame, child) link words and leaves
    # the leaf-slot arrays unused, so a counter written for the STBVH walking one
    # reports 1 group per ray and no hits at all. That is what it did.
    from algan.rendering.raytracing import settings as _rts  # noqa: PLC0415

    if _rts.refit_bvh_active():
        raise SystemExit(
            "the batch would build a RefitBVH, whose topology is per-(frame, "
            "child) link words and whose leaf-slot arrays are unused. This "
            "counter transcribes the STBVH walk, so pointed at one it reports "
            "1 group per ray and no hits -- and agrees with the production "
            "walk, which finds nothing either.\n\n"
            "Re-run with ALGAN_BVH_REFIT=0. Note what that implies for ssF: "
            "BVH_REFIT DEFAULTS ON, and _build_accel's refit branch ignores "
            "``builder`` outright, so ALGAN_BVH_BUILD and ALGAN_BEZ_BVH_SPLIT "
            "reorder a tree the shipped renderer does not build."
        )
    _register_test_fonts()
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    SCENES[which]()
    cap = _Capture()
    with cap:
        scene.save_frame(
            os.path.join("algan_outputs", "bvh_steps", f"{which}.png"),
            video_settings=quality,
            overwrite=True,
        )
    if cap.data is None:
        raise SystemExit(
            "no raster coverage batch was emitted -- the scene rendered "
            "nothing, so there is no tree to walk"
        )
    return cap.data


# -------------------------------------------------------------------- driver


def _to_ti(t):
    return t.contiguous()


def _run(geom, data, n_rays, verify, random_rays):
    dev = data["tri_blocks"].device if "tri_blocks" in data else None
    dev = data.get(f"{geom}_blocks").device
    width, height = data["width"], data["height"]
    frame = data["time_start"]
    total = width * height
    stride = max(1, total // n_rays)
    n = total // stride

    rays = torch.zeros((n, 6), dtype=torch.float32, device=dev)
    _gen_rays(
        n,
        frame,
        stride,
        width,
        float(data["half_w"]),
        float(data["half_h"]),
        _to_ti(data["cam_origin"]),
        _to_ti(data["screen_point"]),
        _to_ti(data["pixel_basis_x"]),
        _to_ti(data["pixel_basis_y"]),
        rays,
    )
    if random_rays:
        # Incoherent rays through the same volume: the harder case for any
        # ordering, and the one a coherent primary sweep cannot speak for.
        rng = np.random.default_rng(0)
        origins = rays[:, :3].cpu().numpy()
        targets = origins + rays[:, 3:].cpu().numpy() * 8.0
        jitter = rng.normal(0.0, 1.5, size=targets.shape).astype(np.float32)
        d = (targets + jitter) - origins
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rays[:, 3:] = torch.from_numpy(d).to(dev)

    if DEBUG:
        nd = data.get(f"{geom}_nodes")
        print(
            f"  [dbg] {geom} root node row: {nd[0].tolist() if nd is not None else None}"
        )
        print(
            f"  [dbg] blocks {tuple(data[f'{geom}_blocks'].shape)} "
            f"first_leaf {data[f'{geom}_first_leaf']} "
            f"leaf_prim {tuple(data[f'{geom}_leaf_prim'].shape)} "
            f"live slots {int((data[f'{geom}_leaf_prim'] >= 0).sum())}"
        )
        print(
            f"  [dbg] frame {frame} width {width} height {height} "
            f"half {data['half_w']},{data['half_h']}"
        )
        print(f"  [dbg] ray0 o={rays[0, :3].tolist()} d={rays[0, 3:].tolist()}")
        print(
            f"  [dbg] ray mid o={rays[n // 2, :3].tolist()} "
            f"d={rays[n // 2, 3:].tolist()}"
        )
    out = torch.zeros((n, 5), dtype=torch.float32, device=dev)
    ref = torch.zeros((n, 2), dtype=torch.float32, device=dev)
    if geom == "tri":
        _steps_tri(
            n,
            frame,
            rays,
            _to_ti(data["tri_blocks"]),
            _to_ti(data["tri_leaf_prim"]),
            _to_ti(data["tri_leaf_tspan"]),
            data["tri_first_leaf"],
            _to_ti(data["tri_pos"]),
            out,
        )
        if verify:
            _ref_tri(
                n,
                frame,
                rays,
                _to_ti(data["tri_blocks"]),
                _to_ti(data["tri_node_miss"]),
                _to_ti(data["tri_leaf_prim"]),
                _to_ti(data["tri_leaf_tspan"]),
                data["tri_first_leaf"],
                _to_ti(data["tri_pos"]),
                ref,
            )
    else:
        # Screen-constant border widths need the render's own pixel scale;
        # the same value goes to both walks, so it cannot favour either.
        psz = 2.0 / max(1.0, float(data["half_h"]) * 2.0)
        _steps_bez(
            n,
            frame,
            psz,
            rays,
            _to_ti(data["bez_blocks"]),
            _to_ti(data["bez_leaf_prim"]),
            _to_ti(data["bez_leaf_tspan"]),
            data["bez_first_leaf"],
            _to_ti(data["circuit_meta"]),
            _to_ti(data["edges_2d"]),
            _to_ti(data["edge_accel"]),
            out,
        )
        if verify:
            _ref_bez(
                n,
                frame,
                psz,
                rays,
                _to_ti(data["bez_blocks"]),
                _to_ti(data["bez_node_miss"]),
                _to_ti(data["bez_leaf_prim"]),
                _to_ti(data["bez_leaf_tspan"]),
                data["bez_first_leaf"],
                _to_ti(data["circuit_meta"]),
                _to_ti(data["edges_2d"]),
                _to_ti(data["edge_accel"]),
                ref,
            )
    _sync_devices()
    o = out.cpu().numpy()
    result = {
        "rays": n,
        "groups": float(o[:, 0].mean()),
        "leaves": float(o[:, 1].mean()),
        "prims": float(o[:, 2].mean()),
        "groups_total": int(o[:, 0].sum()),
        "prims_total": int(o[:, 2].sum()),
        "hit_frac": float((o[:, 4] >= 0).mean()),
    }
    if verify:
        rf = ref.cpu().numpy()
        same_prim = int((rf[:, 1] == o[:, 4]).sum())
        dt = np.abs(
            np.where(o[:, 3] < 1e29, o[:, 3], 0.0)
            - np.where(rf[:, 0] < 1e29, rf[:, 0], 0.0)
        )
        result["verify"] = (same_prim, n, float(dt.max()))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", choices=sorted(SCENES), default=None)
    ap.add_argument("--geom", choices=("tri", "bez"), default=None)
    ap.add_argument("--rays", type=int, default=200000)
    ap.add_argument("--res", choices=("ld", "md"), default="md")
    ap.add_argument(
        "--random",
        action="store_true",
        help="incoherent rays instead of the primary sweep",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="also run the PRODUCTION walk and compare its hit",
    )
    ap.add_argument(
        "--compare",
        default=None,
        metavar="ENV=VAL[,ENV=VAL]",
        help="rerun in a second process with these env vars set and print "
        "both step counts side by side (BVH build order is read at build "
        "time, so the arms must not share a process)",
    )
    args = ap.parse_args()

    geom = args.geom or (args.scene or "bez")
    scene = args.scene or ("tri" if geom == "tri" else "bez")
    quality = {"ld": LD, "md": MD}[args.res]

    data = _capture_scene(scene, quality)
    if geom == "tri" and not data.get("num_triangles"):
        raise SystemExit("scene has no triangles")
    if geom == "bez" and not data.get("num_circuits"):
        raise SystemExit("scene has no circuits")

    res = _run(geom, data, args.rays, args.verify, args.random)
    label = os.environ.get("ALGAN_BVH_BUILD", "default")
    if os.environ.get("ALGAN_BEZ_BVH_SPLIT"):
        label += "+bez_split"
    print(
        f"\n{scene}/{geom:3s} order={label:16s} rays={res['rays']}  "
        f"groups/ray {res['groups']:8.3f}  leaf slots/ray {res['leaves']:8.3f}  "
        f"prim tests/ray {res['prims']:8.3f}  hit {res['hit_frac'] * 100:.1f}%"
    )
    if "verify" in res:
        same, n, dt = res["verify"]
        verdict = "AGREES" if same == n and dt == 0.0 else "DISAGREES"
        print(
            f"  verify vs the production walk: {same}/{n} rays return the same "
            f"primitive, worst |dt| {dt:.3e}  -> {verdict}"
        )
    if args.compare:
        env = dict(os.environ)
        for pair in args.compare.split(","):
            k, _, v = pair.partition("=")
            env[k.strip()] = v.strip() or "1"
        cmd = [
            sys.executable,
            __file__,
            "--scene",
            scene,
            "--geom",
            geom,
            "--rays",
            str(args.rays),
            "--res",
            args.res,
        ]
        if args.random:
            cmd.append("--random")
        if args.verify:
            cmd.append("--verify")
        print("\n-- second arm --")
        subprocess.run(cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
