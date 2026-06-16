"""Standalone unit tests for the spatio-temporal BVH ray tracer.

Run directly: python tests/test_raytracing_unit.py

Validates, without the full scene pipeline:
1. The 4D Morton bit-spreading against a slow bit-loop reference.
2. STBVH structural invariants (parents bound children, leaves cover every
   visible (frame, primitive) pair exactly once).
3. The unified render kernel against a brute-force PyTorch renderer that
   intersects every (ray, triangle) pair, sorts hits by depth and
   alpha-blends them in order (transparency included).
4. Mirror reflections: a reflective floor showing geometry that sits behind
   the camera.
5. PN (quadratic Bezier / Steiner) triangle patches: flat patches against
   the triangle brute force, curved patches against an exact float64
   paraboloid reference (including rays that pierce a patch twice),
   watertight seams between sub-patches, and mirror/Monte Carlo/physical
   smoke tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from algan.rendering.raytracing.pn_patch import (
    pn_control_points,
    pn_patch_coefficients,
)
from algan.rendering.raytracing.ray_trace_taichi import (
    MIN_ALPHA,
    finalize_samples,
    path_trace_physical_stbvh,
    path_trace_scene_stbvh,
    render_scene_stbvh,
)
from algan.rendering.raytracing.stbvh import (
    EMPTY_HI,
    EMPTY_LO,
    _spread_bits_4,
    build_stbvh,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_morton_spread():
    x = torch.arange(0, 1 << 15, 7, dtype=torch.long)
    spread = _spread_bits_4(x)
    # Slow reference: place bit i at position 4 * i.
    ref = torch.zeros_like(x)
    for i in range(16):
        ref |= ((x >> i) & 1) << (4 * i)
    assert torch.equal(spread, ref), "morton bit spread mismatch"
    print("ok: morton bit spread")


def _leaf_coverage(bvh, num_frames, num_prims):
    """Return int [T, N]: how many leaf slots cover each (frame, prim) pair."""
    covered = torch.zeros((num_frames, num_prims), dtype=torch.int32)
    prim = bvh.leaf_prim.long().cpu()
    tspan = bvh.leaf_tspan.long().cpu()
    t0 = tspan & 0xFFFF
    t1 = (tspan >> 16) & 0x7FFF
    for i in range(prim.shape[0]):
        if prim[i] >= 0:
            covered[t0[i]:t1[i] + 1, prim[i]] += 1
    return covered


def test_stbvh_structure():
    torch.manual_seed(0)
    T, N = 37, 23
    centers = torch.randn(1, N, 3, device=DEVICE) * 3
    velocity = torch.randn(1, N, 3, device=DEVICE) * 0.05
    # Half the primitives are static, half drift over time.
    velocity[:, ::2] = 0
    t = torch.arange(T, device=DEVICE).view(T, 1, 1).float()
    c = centers + velocity * t
    half = torch.rand(1, N, 3, device=DEVICE) * 0.5 + 0.05
    lo, hi = c - half, c + half
    # Make some (frame, prim) pairs invisible.
    visible = torch.rand(T, N, device=DEVICE) > 0.2
    lo = torch.where(visible.unsqueeze(-1), lo, torch.full_like(lo, EMPTY_LO))
    hi = torch.where(visible.unsqueeze(-1), hi, torch.full_like(hi, EMPTY_HI))

    bvh = build_stbvh(lo, hi, num_frames=T)

    coverage = _leaf_coverage(bvh, T, N)
    visible_cpu = visible.cpu()
    assert not (coverage > 1).any(), "a (frame, prim) pair is covered twice"
    missing = visible_cpu & (coverage == 0)
    assert not missing.any(), "a visible (frame, prim) pair is not covered"

    # Parents must bound children spatially and temporally.
    P = bvh.num_leaves
    for parent in range(P - 1):
        for child in (2 * parent + 1, 2 * parent + 2):
            assert (bvh.nodes[parent, 0:3] <= bvh.nodes[child, 0:3] + 1e-5).all()
            assert (bvh.nodes[parent, 3:6] >= bvh.nodes[child, 3:6] - 1e-5).all()
            assert bvh.nodes[parent, 6] <= bvh.nodes[child, 6]
            assert bvh.nodes[parent, 7] >= bvh.nodes[child, 7]
    # Leaf slots must lie within their leaf node's bounds/interval.
    from algan.rendering.raytracing.stbvh import LEAF_SIZE
    prim = bvh.leaf_prim.long().cpu()
    tspan = bvh.leaf_tspan.long().cpu()
    nodes_cpu = bvh.nodes.cpu()
    for i in range(prim.shape[0]):
        if prim[i] >= 0:
            leaf_node = bvh.first_leaf + i // LEAF_SIZE
            assert nodes_cpu[leaf_node, 6] <= (tspan[i] & 0xFFFF)
            assert nodes_cpu[leaf_node, 7] >= ((tspan[i] >> 16) & 0x7FFF)
    instances = int((bvh.leaf_prim >= 0).sum())
    print(f"ok: stbvh structure ({instances} instances for "
          f"{int(visible_cpu.sum())} visible (frame, prim) pairs)")


def _dummy_bezier_parts():
    lo = torch.full((1, 1, 3), EMPTY_LO, device=DEVICE)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=DEVICE)
    bvh = build_stbvh(lo, hi, num_frames=1)
    return (bvh,
            torch.zeros((1, 1, 20), device=DEVICE),
            torch.zeros((1, 1, 1, 5), device=DEVICE),
            torch.zeros((1, 1, 5), device=DEVICE),
            torch.zeros((1, 1, 4), device=DEVICE),
            torch.zeros((2,), dtype=torch.int32, device=DEVICE))


def _dummy_pn_parts():
    lo = torch.full((1, 1, 3), EMPTY_LO, device=DEVICE)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=DEVICE)
    bvh = build_stbvh(lo, hi, num_frames=1)
    return (bvh,
            torch.zeros((1, 1, 18), device=DEVICE),
            torch.zeros((1, 1, 9), device=DEVICE),
            torch.zeros((1, 1, 6), device=DEVICE),
            torch.zeros((1, 1, 3, 5), device=DEVICE))


def _dummy_triangle_parts():
    """Empty triangle set (BVH, packed verts, colors) for PN-only scenes."""
    lo = torch.full((1, 1, 3), EMPTY_LO, device=DEVICE)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=DEVICE)
    bvh = build_stbvh(lo, hi, num_frames=1)
    return (bvh, torch.zeros((1, 1, 3, 8), device=DEVICE),
            torch.zeros((1, 1, 3, 5), device=DEVICE))


def _split_tri_verts(tri_verts):
    """Split the legacy packed per-corner layout [T, N, 3, 8] (position,
    normal, reflectivity, roughness) into the renderer's hot/cold arrays:
    positions [T, N, 9], normals [T, N, 9] and (reflectivity, roughness)
    pairs [T, N, 6].
    """
    T, N = tri_verts.shape[0], tri_verts.shape[1]
    pos = tri_verts[..., 0:3].reshape(T, N, 9).contiguous()
    norm = tri_verts[..., 3:6].reshape(T, N, 9).contiguous()
    extra = tri_verts[..., 6:8].reshape(T, N, 6).contiguous()
    return pos, norm, extra


def _run_kernel(tri_bvh, tri_verts, tri_colors, cam, sp, pbx, pby, T, W, H,
                bg=20, max_bounces=0, samples_per_pixel=0, indirect=0.0,
                physical=False, light_pos=None, light_col=None,
                light_intensity=3.141592653589793, ambient=0.0,
                pn_parts=None):
    """Launch the deterministic kernel, the Monte Carlo kernel
    (``samples_per_pixel > 0``), or the physical path tracer
    (``physical=True``). ``pn_parts`` optionally adds PN patch geometry as
    ``(bvh, ctrl, norm, extra, colors)`` (defaults to an empty set).
    """
    bez_bvh, meta, ccolors, bcolors, edges, offsets = _dummy_bezier_parts()
    if pn_parts is None:
        pn_parts = _dummy_pn_parts()
    pn_bvh, pn_ctrl, pn_norm, pn_extra, pn_colors = pn_parts
    out = torch.full((T, W * H, 4), bg, dtype=torch.uint8, device=DEVICE)
    scale = torch.full((T,), 1e-3, device=DEVICE)
    tri_pos, tri_norm, tri_extra = _split_tri_verts(tri_verts)
    shared = (
        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
        tri_bvh.leaf_tspan, tri_bvh.first_leaf,
        tri_pos, tri_norm, tri_extra, tri_colors.contiguous(),
        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
        pn_bvh.leaf_tspan, pn_bvh.first_leaf,
        pn_ctrl.contiguous(), pn_norm.contiguous(), pn_extra.contiguous(),
        pn_colors.contiguous(),
        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
        bez_bvh.leaf_tspan, bez_bvh.first_leaf,
        meta, ccolors, bcolors, edges, offsets,
        cam.contiguous(), sp.contiguous(), pbx.contiguous(), pby.contiguous(),
        scale, 0, T, W, H, float(W // 2), float(H // 2),
        0.0, float(tri_verts.shape[1]), max_bounces, 0)
    if physical:
        if light_pos is None:
            light_pos = torch.zeros((1, 1, 3), device=DEVICE)
            light_col = torch.zeros((1, 1, 3), device=DEVICE)
            num_lights = 0
        else:
            num_lights = light_pos.shape[1]
        accum = torch.zeros((T, W * H, 5), device=DEVICE)
        path_trace_physical_stbvh(
            *shared, samples_per_pixel, light_pos.contiguous(),
            light_col.contiguous(), num_lights, light_intensity, ambient,
            out, accum)
        finalize_samples(samples_per_pixel, 0, accum, out)
    elif samples_per_pixel > 0:
        accum = torch.zeros((T, W * H, 5), device=DEVICE)
        path_trace_scene_stbvh(*shared, samples_per_pixel, indirect, out,
                               accum)
        finalize_samples(samples_per_pixel, 0, accum, out)
    else:
        # has_tri/has_pn/has_bez = 1, 1, 1: traverse every geometry type, as
        # the kernel did before the empty-type gating was added (the dummy
        # PN/bezier BVHs here are empty, so traversing them is a no-op).
        render_scene_stbvh(*shared, 1, 1, 1, out)
    torch.cuda.synchronize()
    return out


def _reference_blend(tri_verts, tri_colors, cam, sp, pbx, pby, T, W, H, bg=20):
    """Brute-force renderer: intersect every (ray, triangle) pair, order hits
    by (distance, layer) and alpha-blend front-to-back over the background.
    """
    Tc = tri_verts.shape[0]
    Tcol = tri_colors.shape[0]
    N = tri_verts.shape[1]
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE),
                            torch.arange(W, device=DEVICE), indexing="ij")
    u = (xs.float() + 0.5 - W // 2) / (H // 2)
    v = (ys.float() + 0.5 - H // 2) / (H // 2)
    frames = []
    for f in range(T):
        pix = sp[f] + u.unsqueeze(-1) * pbx[f] + v.unsqueeze(-1) * pby[f]
        ro = cam[f]
        rd = torch.nn.functional.normalize(pix - ro, dim=-1).view(-1, 1, 3)
        tri = tri_verts[f % Tc, :, :, :3].unsqueeze(0)
        e1 = tri[..., 1, :] - tri[..., 0, :]
        e2 = tri[..., 2, :] - tri[..., 0, :]
        pv = torch.cross(rd.expand(-1, N, -1), e2, dim=-1)
        det = (e1 * pv).sum(-1)
        tv = ro - tri[..., 0, :]
        w1 = (tv * pv).sum(-1) / det
        qv = torch.cross(tv.expand(rd.shape[0], -1, -1), e1, dim=-1)
        w2 = (rd * qv).sum(-1) / det
        t_hit = (e2 * qv).sum(-1) / det
        valid = ((det.abs() > 1e-12) & (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
                 & (t_hit > 1e-4))
        w0 = 1 - w1 - w2
        cols = tri_colors[f % Tcol]
        col = (w0.unsqueeze(-1) * cols[:, 0] + w1.unsqueeze(-1) * cols[:, 1]
               + w2.unsqueeze(-1) * cols[:, 2])  # [rays, N, 5]
        alpha = col[..., 4].clamp(0, 1) * valid
        # Order hits per ray by distance (ties by descending layer = index).
        # float64 keys: the tie-break perturbation must survive rounding at
        # any hit distance (in float32 it underflows for exactly coplanar
        # triangles, leaving the tie order to argsort's whim).
        order_key = torch.where(valid, t_hit,
                                torch.full_like(t_hit, 1e30)).double()
        order_key = order_key - torch.arange(N, device=DEVICE) * 1e-7
        order = order_key.argsort(dim=-1)
        alpha_sorted = alpha.gather(-1, order)
        col_sorted = col[..., :4].gather(
            -2, order.unsqueeze(-1).expand(-1, -1, 4))
        trans = torch.cumprod(1 - alpha_sorted, dim=-1)
        weight = torch.cat((torch.ones_like(trans[:, :1]), trans[:, :-1]), -1)
        acc = (weight * alpha_sorted).unsqueeze(-1) * col_sorted
        acc = acc.sum(1)
        remaining = trans[:, -1:]
        frame = (acc * 255 + remaining * bg + 0.5).clamp(0, 255).to(torch.uint8)
        frames.append(frame)
    return torch.stack(frames)


def _random_triangle_scene(T=7, N=30):
    """Animated random triangles with mixed transparency, plus a camera."""
    torch.manual_seed(1)
    base = torch.randn(1, N, 1, 3, device=DEVICE) * 1.5
    spread = torch.randn(1, N, 3, 3, device=DEVICE) * 0.5
    drift = torch.randn(1, N, 1, 3, device=DEVICE) * 0.05
    drift[:, ::3] = 0
    t = torch.arange(T, device=DEVICE).view(T, 1, 1, 1).float()
    corners = (base + spread + drift * t).contiguous()
    tri_verts = torch.cat(
        (corners, torch.zeros(T, N, 3, 5, device=DEVICE)), -1)
    colors = torch.rand(1, N, 3, 5, device=DEVICE)
    colors[..., 4] = (torch.rand(1, N, 1, 1, device=DEVICE) * 0.8 + 0.2
                      ).expand(-1, -1, 3, -1).squeeze(-1)
    colors[:, 5::7, :, 4] = 0.0  # some invisible triangles
    colors[:, 3::5, :, 4] = 1.0  # some opaque (early termination path)

    cam = torch.tensor([0.0, 0.0, 8.0], device=DEVICE).repeat(T, 1)
    sp = torch.tensor([0.0, 0.0, 5.0], device=DEVICE).repeat(T, 1)
    pbx = torch.tensor([1.0, 0.0, 0.0], device=DEVICE).repeat(T, 1)
    pby = torch.tensor([0.0, 1.0, 0.0], device=DEVICE).repeat(T, 1)

    lo = corners.amin(-2)
    hi = corners.amax(-2)
    vis = (colors[..., 4].amax(-1) > MIN_ALPHA).expand(T, -1)
    lo = torch.where(vis.unsqueeze(-1), lo, torch.full_like(lo, EMPTY_LO))
    hi = torch.where(vis.unsqueeze(-1), hi, torch.full_like(hi, EMPTY_HI))
    opaque = colors[..., 4].amin(-1) >= 1.0 - 1e-6  # [1, N], as in production
    bvh = build_stbvh(lo.contiguous(), hi.contiguous(), num_frames=T,
                      opaque=opaque)
    return bvh, tri_verts, colors, cam, sp, pbx, pby


def test_blended_render_vs_brute_force():
    T, W, H = 7, 64, 48
    bvh, tri_verts, colors, cam, sp, pbx, pby = _random_triangle_scene(T)
    got = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, T, W, H)
    ref = _reference_blend(tri_verts, colors, cam, sp, pbx, pby, T, W, H)
    got = got.view(T, H * W, 4).float()
    ref = ref.view(T, H * W, 4).float()
    err = (got - ref).abs()
    bad = (err > 2).float().mean()
    assert bad < 2e-3, f"blended output mismatch: {bad:.2%} of channels off by >2"
    print(f"ok: blended render matches brute force "
          f"(max err {err.max():.0f}, {bad:.3%} channels off by >2)")


def test_deep_translucent_stack():
    """A stack of translucent sheets (with coplanar pairs) deeper than the
    traversal's per-pass hit batch must blend exactly like the brute-force
    reference: every ray needs several gather/drain rounds, and the coplanar
    pairs exercise the (distance, layer) tie ordering across batches. One
    fully opaque sheet sits mid-stack, exercising the gatherer's opaque
    pruning (the sheets behind it must not change the image).
    """
    T, W, H = 2, 48, 36
    z = torch.tensor([3.0, 2.6, 2.6, 2.2, 1.8, 1.8, 1.4, 1.0, 0.6, 0.6,
                      0.2, -0.2], device=DEVICE)
    N = z.shape[0]
    corners = torch.zeros(1, N, 3, 3, device=DEVICE)
    corners[0, :, 0] = torch.tensor([-9.0, -9.0, 0.0], device=DEVICE)
    corners[0, :, 1] = torch.tensor([9.0, -9.0, 0.0], device=DEVICE)
    corners[0, :, 2] = torch.tensor([0.0, 14.0, 0.0], device=DEVICE)
    corners[0, :, :, 2] = z.view(N, 1)
    tri_verts = torch.cat(
        (corners, torch.zeros(1, N, 3, 5, device=DEVICE)), -1)
    torch.manual_seed(3)
    colors = torch.rand(1, N, 3, 5, device=DEVICE)
    colors[..., 4] = 0.3  # translucent layers stay above MIN_WEIGHT
    # Opaque sheet at z=0.6; its coplanar partner (index 9, higher layer)
    # still peels first, and the sheets behind it are absorbed.
    colors[0, 8, :, 4] = 1.0

    cam = torch.tensor([0.0, 0.0, 8.0], device=DEVICE).repeat(T, 1)
    sp = torch.tensor([0.0, 0.0, 5.0], device=DEVICE).repeat(T, 1)
    pbx = torch.tensor([1.0, 0.0, 0.0], device=DEVICE).repeat(T, 1)
    pby = torch.tensor([0.0, 1.0, 0.0], device=DEVICE).repeat(T, 1)
    opaque = colors[..., 4].amin(-1) >= 1.0 - 1e-6
    bvh = build_stbvh(corners.amin(-2).contiguous(),
                      corners.amax(-2).contiguous(), num_frames=T,
                      opaque=opaque)
    got = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, T, W, H
                      ).view(T, H * W, 4).float()
    ref = _reference_blend(tri_verts, colors, cam, sp, pbx, pby, T, W, H
                           ).view(T, H * W, 4).float()
    err = (got - ref).abs()
    bad = (err > 2).float().mean()
    assert bad < 1e-4, (
        f"deep stack mismatch: {bad:.2%} of channels off by >2 "
        f"(max err {err.max():.0f})")
    print(f"ok: 12-deep translucent stack (opaque sheet mid-stack) blends "
          f"exactly (max err {err.max():.0f})")


def test_monte_carlo_converges_to_blend():
    """At many samples per pixel, stochastic transparency (random
    pass-through by opacity) must converge to exact alpha blending.
    Edge pixels are excluded: the Monte Carlo kernel jitters sub-pixel ray
    positions (anti-aliasing), so they legitimately differ from the
    deterministic center samples.
    """
    T, W, H = 7, 64, 48
    bvh, tri_verts, colors, cam, sp, pbx, pby = _random_triangle_scene(T)
    det = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, T, W, H
                      ).view(T, H, W, 4).float()
    mc = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, T, W, H,
                     samples_per_pixel=256).view(T, H, W, 4).float()
    pooled_max = torch.nn.functional.max_pool2d(
        det.permute(0, 3, 1, 2), 3, stride=1, padding=1)
    pooled_min = -torch.nn.functional.max_pool2d(
        -det.permute(0, 3, 1, 2), 3, stride=1, padding=1)
    flat = ((pooled_max - pooled_min).amax(1) < 6)  # locally uniform pixels
    err = (det - mc).abs().amax(-1)
    flat_err = err[flat].mean()
    assert flat_err < 3, (
        f"Monte Carlo did not converge to the blend (flat-region mean err "
        f"{flat_err:.2f})")
    print(f"ok: 256-spp Monte Carlo converges to exact blending "
          f"(flat-region mean err {flat_err:.2f}/255 over "
          f"{int(flat.sum())} pixels)")


def test_mirror_reflection():
    # A reflective floor (z=0, normal +z) seen from above; a red panel
    # *behind* the camera (z=9) is visible only via the reflection.
    T, W, H = 1, 48, 48
    floor = torch.tensor([[[-20.0, -20, 0], [20, -20, 0], [0, 40, 0]]],
                         device=DEVICE)
    panel = torch.tensor([[[-30.0, -30, 9], [30, -30, 9], [0, 60, 9]]],
                         device=DEVICE)
    corners = torch.stack((floor[0], panel[0])).unsqueeze(0)  # [1, 2, 3, 3]
    normals = torch.zeros(1, 2, 3, 3, device=DEVICE)
    normals[0, 0, :, 2] = 1.0
    reflectivity = torch.zeros(1, 2, 3, 1, device=DEVICE)
    reflectivity[0, 0] = 1.0  # the floor is a perfect mirror
    roughness = torch.zeros(1, 2, 3, 1, device=DEVICE)
    tri_verts = torch.cat((corners, normals, reflectivity, roughness), -1)
    colors = torch.zeros(1, 2, 3, 5, device=DEVICE)
    colors[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], device=DEVICE)
    colors[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)

    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)

    lo = corners.amin(-2)
    hi = corners.amax(-2)
    bvh = build_stbvh(lo.contiguous(), hi.contiguous(), num_frames=T)

    with_bounce = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby,
                              T, W, H, max_bounces=2)
    no_bounce = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby,
                            T, W, H, max_bounces=0)
    center = with_bounce.view(T, H, W, 4)[0, H // 2, W // 2]
    assert center[0] > 200, f"mirror should reflect red, got {center.tolist()}"
    assert center[2] < 50, f"mirror should not show blue, got {center.tolist()}"
    center_flat = no_bounce.view(T, H, W, 4)[0, H // 2, W // 2]
    assert center_flat[2] > 200, (
        "with bounces disabled the mirror should show its own blue color, "
        f"got {center_flat.tolist()}")
    print("ok: mirror reflection (red panel behind the camera is visible "
          "in the floor)")


def _mirror_floor_scene(panel_half_width, floor_roughness):
    """Mirror floor at z=0 seen from above; a red panel *behind* the camera
    (z=9) covers reflected directions with |x| < panel_half_width.
    """
    s = panel_half_width
    floor = torch.tensor([[[-20.0, -20, 0], [20, -20, 0], [0, 40, 0]]],
                         device=DEVICE)
    panel = torch.tensor([[[-s, -30.0, 9], [s, -30.0, 9], [0.0, 60.0, 9]]],
                         device=DEVICE)
    corners = torch.stack((floor[0], panel[0])).unsqueeze(0)
    normals = torch.zeros(1, 2, 3, 3, device=DEVICE)
    normals[0, 0, :, 2] = 1.0
    reflectivity = torch.zeros(1, 2, 3, 1, device=DEVICE)
    reflectivity[0, 0] = 1.0
    roughness = torch.zeros(1, 2, 3, 1, device=DEVICE)
    roughness[0, 0] = floor_roughness
    tri_verts = torch.cat((corners, normals, reflectivity, roughness), -1)
    colors = torch.zeros(1, 2, 3, 5, device=DEVICE)
    colors[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], device=DEVICE)
    colors[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)
    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)
    bvh = build_stbvh(corners.amin(-2).contiguous(),
                      corners.amax(-2).contiguous(), num_frames=1)
    return bvh, tri_verts, colors, cam, sp, pbx, pby


def test_glossy_reflection_blurs():
    """A rough mirror must blur the edge of a reflected panel: with a sharp
    mirror almost every floor pixel is purely red (panel) or purely blue
    (no specular contribution? -- with reflectivity 1 the floor shows the
    background), while a glossy lobe produces many intermediate pixels.
    """
    W = H = 48

    def intermediate_count(roughness):
        bvh, tv, tc, cam, sp, pbx, pby = _mirror_floor_scene(8.0, roughness)
        img = _run_kernel(bvh, tv, tc, cam, sp, pbx, pby, 1, W, H,
                          max_bounces=2, samples_per_pixel=128
                          ).view(H, W, 4).float()
        red = img[..., 0]
        return int(((red > 60) & (red < 195)).sum())

    sharp = intermediate_count(0.0)
    rough = intermediate_count(0.5)
    assert rough > sharp * 2 + 20, (
        f"glossy mirror did not blur the reflection "
        f"(intermediate pixels sharp={sharp} rough={rough})")
    print(f"ok: glossy roughness blurs reflections "
          f"(edge pixels sharp={sharp} -> rough={rough})")


def test_indirect_color_bleed():
    """With indirect bounces enabled, a white floor next to an (edge-on,
    directly invisible) red wall must pick up red near the wall.
    """
    W = H = 48
    floor = torch.tensor([[[-3.0, -3, 0], [3, -3, 0], [0, 4, 0]]],
                         device=DEVICE)
    wall = torch.tensor([[[1.5, -3.0, 0.0], [1.5, 3.0, 0.0], [1.5, 0.0, 4.0]]],
                        device=DEVICE)
    corners = torch.stack((floor[0], wall[0])).unsqueeze(0)
    tri_verts = torch.cat(
        (corners, torch.zeros(1, 2, 3, 5, device=DEVICE)), -1)
    colors = torch.zeros(1, 2, 3, 5, device=DEVICE)
    colors[0, 0] = torch.tensor([0.7, 0.7, 0.7, 0.0, 1.0], device=DEVICE)
    colors[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)
    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)
    bvh = build_stbvh(corners.amin(-2).contiguous(),
                      corners.amax(-2).contiguous(), num_frames=1)

    def render(indirect):
        return _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby,
                           1, W, H, max_bounces=3, samples_per_pixel=512,
                           indirect=indirect).view(H, W, 4).float()

    flat = render(0.0)
    lit = render(0.8)
    # Floor pixels just left of the wall (world x in ~[0.6, 1.4]).
    near = slice(int(0.26 * (H // 2) + W // 2), int(0.5 * (H // 2) + W // 2))
    rows = slice(H // 3, 2 * H // 3)
    bleed_on = (lit[rows, near, 0] - lit[rows, near, 1]).mean()
    bleed_off = (flat[rows, near, 0] - flat[rows, near, 1]).mean()
    assert bleed_on > bleed_off + 8, (
        f"no red color bleed from indirect bounces "
        f"(R-G near wall: off={bleed_off:.1f} on={bleed_on:.1f})")
    print(f"ok: indirect bounces bleed red onto the white floor "
          f"(R-G near wall {bleed_off:.1f} -> {bleed_on:.1f})")


def _floor_pixel(world_x, W, H):
    """Pixel column of floor point (world_x, 0, 0) for the standard camera
    at (0, 0, 8) with screen plane z=5 (ray hits z=0 at 8/3 the screen
    offset, so u = 3x/8).
    """
    return int(round(0.375 * world_x * (H // 2) + W // 2))


def test_physical_direct_lighting_and_shadows():
    """A point light over a matte floor: brightness must follow the Lambert
    cosine (no vertex shading involved) and an occluder must cast a shadow
    via the explicit shadow rays.
    """
    W, H = 64, 48
    floor = torch.tensor([[[-8.0, -8, 0], [8, -8, 0], [0, 12, 0]]],
                         device=DEVICE)
    occluder = torch.tensor([[[1.8, -0.2, 2.0], [2.2, -0.2, 2.0],
                              [2.0, 0.3, 2.0]]], device=DEVICE)
    corners = torch.stack((floor[0], occluder[0])).unsqueeze(0)
    tri_verts = torch.cat(
        (corners, torch.zeros(1, 2, 3, 5, device=DEVICE)), -1)
    colors = torch.zeros(1, 2, 3, 5, device=DEVICE)
    colors[0, 0] = torch.tensor([0.8, 0.8, 0.8, 0.0, 1.0], device=DEVICE)
    colors[0, 1] = torch.tensor([0.3, 0.3, 0.3, 0.0, 1.0], device=DEVICE)
    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)
    bvh = build_stbvh(corners.amin(-2).contiguous(),
                      corners.amax(-2).contiguous(), num_frames=1)
    light_pos = torch.tensor([[[2.0, 0.0, 4.0]]], device=DEVICE)
    light_col = torch.tensor([[[1.0, 1.0, 1.0]]], device=DEVICE)

    img = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, 1, W, H,
                      bg=0, max_bounces=0, samples_per_pixel=256,
                      physical=True, light_pos=light_pos,
                      light_col=light_col).view(H, W, 4).float()
    row = H // 2
    lit_near = img[row, _floor_pixel(0.8, W, H), 0]      # cos ~ 0.96
    lit_far = img[row, _floor_pixel(-2.5, W, H), 0]      # cos ~ 0.66
    shadowed = img[row, _floor_pixel(2.0, W, H), 0]      # under the occluder
    print(f"physical floor brightness: near={lit_near:.0f} far={lit_far:.0f} "
          f"shadow={shadowed:.0f}")
    # Lambert: brightness ~ albedo * cos -> 0.8 * 0.96 * 255 ~ 196.
    assert 150 < lit_near < 240, f"direct lighting off: {lit_near}"
    assert lit_far < lit_near - 20, "no cosine falloff with light distance"
    assert lit_far > 60, f"far floor unexpectedly dark: {lit_far}"
    assert shadowed < 30, f"occluder cast no shadow: {shadowed}"
    print("ok: physical direct lighting follows Lambert cosine and shadows")


def test_physical_emissive_surface():
    """With no lights at all, a glowing panel must illuminate a matte floor
    through indirect bounces (emission picked up by scattered paths).
    """
    W, H = 64, 48
    floor = torch.tensor([[[-8.0, -8, 0], [8, -8, 0], [0, 12, 0]]],
                         device=DEVICE)
    panel = torch.tensor([[[1.2, -0.8, 2.5], [2.4, -0.8, 2.5],
                           [1.8, 0.8, 2.5]]], device=DEVICE)
    corners = torch.stack((floor[0], panel[0])).unsqueeze(0)
    tri_verts = torch.cat(
        (corners, torch.zeros(1, 2, 3, 5, device=DEVICE)), -1)
    colors = torch.zeros(1, 2, 3, 5, device=DEVICE)
    colors[0, 0] = torch.tensor([0.8, 0.8, 0.8, 0.0, 1.0], device=DEVICE)
    colors[0, 1] = torch.tensor([1.0, 0.15, 0.15, 6.0, 1.0], device=DEVICE)
    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)
    bvh = build_stbvh(corners.amin(-2).contiguous(),
                      corners.amax(-2).contiguous(), num_frames=1)

    img = _run_kernel(bvh, tri_verts, colors, cam, sp, pbx, pby, 1, W, H,
                      bg=0, max_bounces=2, samples_per_pixel=512,
                      physical=True).view(H, W, 4).float()
    row = H // 2
    near = img[row, _floor_pixel(0.9, W, H)]
    far = img[row, _floor_pixel(-3.0, W, H)]
    print(f"emissive bleed: near R={near[0]:.0f} G={near[1]:.0f}, "
          f"far R={far[0]:.0f}")
    assert near[0] > far[0] + 8, "emissive panel did not light the floor"
    assert near[0] > near[1] + 4, "emissive lighting lost the panel's color"
    print("ok: emissive (glow) surfaces light their surroundings")


# ---------------------------------------------------------------------------
# PN (quadratic Bezier / Steiner) triangle patches
# ---------------------------------------------------------------------------

def _pn_hull_points(coeffs):
    """Bezier control points recovered from monomial coefficient rows
    [..., 18]; the patch lies in their convex hull."""
    k = coeffs.unflatten(-1, (6, 3))
    k0, ku, kv, kuu, kvv, kuv = k.unbind(-2)
    return torch.stack((k0, k0 + ku + kuu, k0 + kv + kvv,
                        k0 + 0.5 * ku, k0 + 0.5 * (ku + kv + kuv),
                        k0 + 0.5 * kv), -2)


def _bvh_from_hull(hull, colors, num_frames):
    """STBVH over per-frame control-net bounds, with the production rules:
    fully transparent frames are empty, fully opaque primitives flagged."""
    lo = hull.amin(-2)
    hi = hull.amax(-2)
    vis = colors[..., 4].amax(-1) > MIN_ALPHA
    if vis.shape[0] != lo.shape[0]:
        vis = vis.expand(lo.shape[0], -1)
    lo = torch.where(vis.unsqueeze(-1), lo, torch.full_like(lo, EMPTY_LO))
    hi = torch.where(vis.unsqueeze(-1), hi, torch.full_like(hi, EMPTY_HI))
    opaque = colors[..., 4].amin(-1) >= 1.0 - 1e-6
    return build_stbvh(lo.contiguous(), hi.contiguous(),
                       num_frames=num_frames, opaque=opaque)


def _pn_parts_from_coeffs(coeffs, colors, normals9=None, extra=None,
                          num_frames=None):
    """Packed PN arrays + STBVH for explicit patch coefficients [Tp, N, 18]
    and per-corner colors [Tc, N, 3, 5]."""
    n = coeffs.shape[1]
    if num_frames is None:
        num_frames = coeffs.shape[0]
    if normals9 is None:
        normals9 = torch.zeros((1, n, 9), device=DEVICE)
    if extra is None:
        extra = torch.zeros((1, n, 6), device=DEVICE)
    bvh = _bvh_from_hull(_pn_hull_points(coeffs), colors, num_frames)
    return (bvh, coeffs.contiguous(), normals9.contiguous(),
            extra.contiguous(), colors.contiguous())


def _pn_parts_from_mesh(corners, normals, colors):
    """Production-style PN parts from per-corner positions/normals
    [T, N, 3, 3] (the construction shared with
    ``RayTracedPNTrianglePrimitive``)."""
    T, n = corners.shape[0], corners.shape[1]
    control = pn_control_points(corners, normals)
    coeffs = pn_patch_coefficients(control)
    bvh = _bvh_from_hull(control, colors, T)
    return (bvh, coeffs.contiguous(),
            normals.reshape(T, n, 9).contiguous(),
            torch.zeros((1, n, 6), device=DEVICE),
            colors.contiguous())


def _camera_frame(position, target, T=1):
    """Camera arrays (origin, screen point, pixel bases) for a viewpoint:
    screen plane 3 units toward ``target``, unit pixel bases orthogonal to
    the view direction. The references below use the same convention."""
    pos = torch.tensor(position, device=DEVICE, dtype=torch.float32)
    tgt = torch.tensor(target, device=DEVICE, dtype=torch.float32)
    fwd = torch.nn.functional.normalize(tgt - pos, dim=-1)
    up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    if fwd[2].abs() > 0.99:
        up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
    right = torch.nn.functional.normalize(torch.linalg.cross(fwd, up),
                                          dim=-1)
    down = torch.linalg.cross(fwd, right)

    def rep(x):
        return x.view(1, 3).repeat(T, 1).contiguous()

    return rep(pos), rep(pos + 3.0 * fwd), rep(right), rep(down)


# One curved patch covering the exact surface (x0 + su*u, y0 + sv*v,
# h*(u^2 + v^2)) over the barycentric domain: rays intersect it through a
# plain quadratic in t, giving an exact float64 reference.
_PARABOLOID = {"x0": -1.0, "y0": -1.0, "su": 3.0, "sv": 3.0, "h": 2.0}


def _paraboloid_coeffs():
    p = _PARABOLOID
    return torch.tensor(
        [p["x0"], p["y0"], 0.0,
         p["su"], 0.0, 0.0,
         0.0, p["sv"], 0.0,
         0.0, 0.0, p["h"],
         0.0, 0.0, p["h"],
         0.0, 0.0, 0.0], device=DEVICE).view(1, 1, 18)


def _paraboloid_point(u, v):
    p = _PARABOLOID
    return torch.stack((p["x0"] + p["su"] * u, p["y0"] + p["sv"] * v,
                        p["h"] * (u * u + v * v)), -1)


def _uv_corner_colors(uvs, alpha):
    """Per-corner colors (R, G, B) = (u, v, 1 - u - v) at the given domain
    corners [N, 3, 2]: every hit then renders the *global* domain
    coordinates (linear interpolation is exact for affine data), which
    makes split and unsplit patches directly comparable."""
    uvs = torch.as_tensor(uvs, device=DEVICE, dtype=torch.float32)
    c = torch.zeros((1, uvs.shape[0], 3, 5), device=DEVICE)
    c[0, :, :, 0] = uvs[..., 0]
    c[0, :, :, 1] = uvs[..., 1]
    c[0, :, :, 2] = 1.0 - uvs[..., 0] - uvs[..., 1]
    c[0, :, :, 4] = alpha
    return c


def _reference_paraboloid(cam, sp, pbx, pby, W, H, alpha, bg):
    """Exact float64 renderer of the paraboloid patch: each ray's quadratic
    is solved in closed form and the in-domain hits are alpha-blended
    front-to-back over the background.

    Returns (image [H*W, 4] as rounded floats, reliable-pixel mask, number
    of pixels with two in-domain hits). The mask excludes pixels whose
    classification is decided by less than ~2e-3 in domain coordinates or
    by a near-zero discriminant -- silhouette/boundary pixels the f32
    kernel may legitimately resolve differently.
    """
    p = _PARABOLOID
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE),
                            torch.arange(W, device=DEVICE), indexing="ij")
    su_s = (xs.double() + 0.5 - W // 2) / (H // 2)
    sv_s = (ys.double() + 0.5 - H // 2) / (H // 2)
    ro = cam[0].double()
    pix = (sp[0].double() + su_s.unsqueeze(-1) * pbx[0].double()
           + sv_s.unsqueeze(-1) * pby[0].double())
    rd = torch.nn.functional.normalize(pix - ro, dim=-1).view(-1, 3)

    u0 = (ro[0] - p["x0"]) / p["su"]
    v0 = (ro[1] - p["y0"]) / p["sv"]
    du = rd[:, 0] / p["su"]
    dv = rd[:, 1] / p["sv"]
    h = p["h"]
    a = h * (du * du + dv * dv)
    b = 2.0 * h * (u0 * du + v0 * dv) - rd[:, 2]
    c = h * (u0 * u0 + v0 * v0) - ro[2]
    disc = b * b - 4.0 * a * c
    disc_scale = torch.maximum(b * b, (4.0 * a * c).abs()) + 1e-30
    reliable = (disc.abs() / disc_scale) > 1e-6
    sq = torch.sqrt(disc.clamp_min(0))
    qq = -0.5 * (b + torch.where(b >= 0, sq, -sq))
    t1 = qq / a
    t2 = c / qq
    t_near = torch.minimum(t1, t2)
    t_far = torch.maximum(t1, t2)

    acc = torch.zeros((W * H, 4), dtype=torch.float64, device=DEVICE)
    weight = torch.ones((W * H,), dtype=torch.float64, device=DEVICE)
    n_valid = torch.zeros((W * H,), dtype=torch.long, device=DEVICE)
    for t in (t_near, t_far):
        u = u0 + t * du
        v = v0 + t * dv
        margin = torch.minimum(torch.minimum(u, v), 1.0 - u - v)
        exists = (disc > 0) & torch.isfinite(t) & (t > 1e-4)
        reliable &= ~(exists & (margin.abs() < 2e-3))
        valid = exists & (margin >= -1e-4)
        n_valid += valid.long()
        col = torch.stack((u, v, 1.0 - u - v, torch.zeros_like(u)), -1)
        a_hit = valid.double() * alpha
        acc += (weight * a_hit).unsqueeze(-1) * col
        weight = weight * (1.0 - a_hit)
    img = (acc * 255.0 + weight.unsqueeze(-1) * bg + 0.5).clamp(0, 255)
    return img.floor(), reliable, int((n_valid == 2).sum())


def test_pn_flat_matches_triangle_reference():
    """PN patches with zero normals are exactly flat triangles, so the
    curved intersector (running its degenerate linear-in-v branch) must
    reproduce the brute-force triangle reference -- animated bounds, mixed
    transparency and opaque pruning included."""
    T, W, H = 7, 64, 48
    _, tri_verts, colors, cam, sp, pbx, pby = _random_triangle_scene(T)
    corners = tri_verts[..., :3].contiguous()
    pn_parts = _pn_parts_from_mesh(corners, torch.zeros_like(corners),
                                   colors)
    got = _run_kernel(*_dummy_triangle_parts(), cam, sp, pbx, pby, T, W, H,
                      pn_parts=pn_parts)
    ref = _reference_blend(tri_verts, colors, cam, sp, pbx, pby, T, W, H)
    got = got.view(T, H * W, 4).float()
    ref = ref.view(T, H * W, 4).float()
    err = (got - ref).abs()
    bad = (err > 2).float().mean()
    assert bad < 2e-3, f"flat PN mismatch: {bad:.2%} of channels off by >2"
    print(f"ok: flat PN patches match the triangle brute force "
          f"(max err {err.max():.0f}, {bad:.3%} channels off by >2)")


def test_pn_paraboloid_analytic():
    """Curved-patch correctness against the exact float64 paraboloid
    reference. The top-down view exercises mostly single hits; the low
    diagonal view sends many rays through the bowl twice, exercising
    multiple hits per patch and their front-to-back transparency order."""
    W = H = 96
    alpha = 0.5
    full_uvs = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    parts = _pn_parts_from_coeffs(_paraboloid_coeffs(),
                                  _uv_corner_colors(full_uvs, alpha))
    for name, pos, tgt, want_two in (
            ("top", (0.4, 0.2, 9.0), (0.5, 0.5, 0.0), 0),
            ("side", (2.6, -1.9, 1.3), (-1.9, 2.6, 0.6), 100)):
        cam, sp, pbx, pby = _camera_frame(pos, tgt)
        got = _run_kernel(*_dummy_triangle_parts(), cam, sp, pbx, pby,
                          1, W, H, pn_parts=parts).view(W * H, 4).float()
        ref, reliable, two = _reference_paraboloid(cam, sp, pbx, pby, W, H,
                                                   alpha, 20.0)
        cover = reliable.float().mean()
        err = (got - ref.float()).abs()[reliable]
        bad = (err > 3).float().mean()
        assert cover > 0.9, f"{name}: only {cover:.0%} of pixels reliable"
        assert two >= want_two, (
            f"{name}: expected >= {want_two} two-hit pixels, got {two}")
        assert bad < 2e-3, (
            f"{name} view mismatch: {bad:.2%} of channels off by >3 "
            f"(max err {err.max():.0f})")
        print(f"ok: paraboloid {name} view matches the float64 reference "
              f"({two} two-hit pixels, max err {err.max():.0f})")


def test_pn_watertight_seam():
    """Splitting the paraboloid patch into two sub-patches along a median
    must render pixel-identically to the unsplit patch: both sub-patches
    report hits on the shared boundary curve, which must blend exactly once
    (the PN analogue of the triangle mesh seam rule), with no holes. The
    sub-patches also exercise the uv cross term (the unsplit patch has
    none)."""
    W = H = 96
    alpha = 0.5
    full_uvs = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    one_parts = _pn_parts_from_coeffs(_paraboloid_coeffs(),
                                      _uv_corner_colors(full_uvs, alpha))

    def subpatch(q):
        """Control points of the paraboloid restricted to domain triangle
        ``q`` [3, 2]: corner samples plus mid-edge controls from the exact
        edge midpoint values (the restriction of a quadratic to a straight
        parameter line is a quadratic curve)."""
        q = torch.tensor(q, device=DEVICE, dtype=torch.float32)
        pts = _paraboloid_point(q[:, 0], q[:, 1])
        mids = (q + q.roll(-1, 0)) * 0.5  # edge midpoints (01, 12, 20)
        mpts = _paraboloid_point(mids[:, 0], mids[:, 1])
        e01 = 2.0 * mpts[0] - 0.5 * (pts[0] + pts[1])
        e12 = 2.0 * mpts[1] - 0.5 * (pts[1] + pts[2])
        e02 = 2.0 * mpts[2] - 0.5 * (pts[2] + pts[0])
        return torch.stack((pts[0], pts[1], pts[2], e01, e12, e02), 0)

    sub_a = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
    sub_b = [[0.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    ctrl = torch.stack((subpatch(sub_a), subpatch(sub_b)), 0).unsqueeze(0)
    two_parts = _pn_parts_from_coeffs(
        pn_patch_coefficients(ctrl), _uv_corner_colors([sub_a, sub_b], alpha))

    for name, pos, tgt in (("top", (0.4, 0.2, 9.0), (0.5, 0.5, 0.0)),
                           ("side", (2.6, -1.9, 1.3), (-1.9, 2.6, 0.6))):
        cam, sp, pbx, pby = _camera_frame(pos, tgt)
        one = _run_kernel(*_dummy_triangle_parts(), cam, sp, pbx, pby,
                          1, W, H, pn_parts=one_parts).view(-1, 4).float()
        two = _run_kernel(*_dummy_triangle_parts(), cam, sp, pbx, pby,
                          1, W, H, pn_parts=two_parts).view(-1, 4).float()
        err = (one - two).abs()
        bad = (err > 2).float().mean()
        assert bad < 2e-3, (
            f"{name}: split patch differs from unsplit "
            f"({bad:.2%} of channels off by >2, max err {err.max():.0f})")
        print(f"ok: {name} view of split patch matches unsplit "
              f"(max err {err.max():.0f}, watertight seam)")


def test_pn_mirror_and_monte_carlo():
    """A perfectly reflective PN floor must mirror a red triangle panel
    that sits behind the camera -- exercising the PN normal fetch on the
    deterministic bounce path -- and the Monte Carlo kernel must agree
    (smoke for its PN dispatch)."""
    T, W, H = 1, 48, 48
    floor_corners = torch.tensor(
        [[[-20.0, -20, 0], [20, -20, 0], [0, 40, 0]]],
        device=DEVICE).unsqueeze(0)
    normals = torch.zeros_like(floor_corners)
    normals[..., 2] = 1.0
    colors = torch.zeros((1, 1, 3, 5), device=DEVICE)
    colors[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], device=DEVICE)
    bvh, coeffs, norm9, extra, pcol = _pn_parts_from_mesh(
        floor_corners, normals, colors)
    extra = extra.clone()
    extra[..., 0::2] = 1.0  # reflectivity 1 at every corner
    pn_parts = (bvh, coeffs, norm9, extra, pcol)

    panel = torch.tensor([[[-30.0, -30, 9], [30, -30, 9], [0, 60, 9]]],
                         device=DEVICE).unsqueeze(0)
    tri_verts = torch.cat(
        (panel, torch.zeros(1, 1, 3, 5, device=DEVICE)), -1)
    tri_colors = torch.zeros((1, 1, 3, 5), device=DEVICE)
    tri_colors[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0],
                                    device=DEVICE)
    tri_bvh = build_stbvh(panel.amin(-2).contiguous(),
                          panel.amax(-2).contiguous(), num_frames=T)

    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)

    det = _run_kernel(tri_bvh, tri_verts, tri_colors, cam, sp, pbx, pby,
                      T, W, H, max_bounces=2, pn_parts=pn_parts)
    center = det.view(T, H, W, 4)[0, H // 2, W // 2]
    assert center[0] > 200, f"PN mirror should reflect red: {center.tolist()}"
    assert center[2] < 50, f"PN mirror should not show blue: {center.tolist()}"

    mc = _run_kernel(tri_bvh, tri_verts, tri_colors, cam, sp, pbx, pby,
                     T, W, H, max_bounces=2, samples_per_pixel=64,
                     pn_parts=pn_parts)
    center_mc = mc.view(T, H, W, 4)[0, H // 2, W // 2]
    assert center_mc[0] > 150, (
        f"Monte Carlo PN mirror lost the reflection: {center_mc.tolist()}")
    print("ok: PN mirror reflects via interpolated patch normals "
          "(deterministic + Monte Carlo)")


def test_pn_physical_shadow():
    """Physical mode with a PN occluder over a triangle floor: the explicit
    shadow rays must see the patch (the transmittance kernel's PN path) and
    block the point light."""
    W, H = 64, 48
    floor = torch.tensor([[[-8.0, -8, 0], [8, -8, 0], [0, 12, 0]]],
                         device=DEVICE).unsqueeze(0)
    tri_verts = torch.cat(
        (floor, torch.zeros(1, 1, 3, 5, device=DEVICE)), -1)
    tri_colors = torch.zeros((1, 1, 3, 5), device=DEVICE)
    tri_colors[0, 0] = torch.tensor([0.8, 0.8, 0.8, 0.0, 1.0], device=DEVICE)
    tri_bvh = build_stbvh(floor.amin(-2).contiguous(),
                          floor.amax(-2).contiguous(), num_frames=1)

    occluder = torch.tensor([[[1.8, -0.2, 2.0], [2.2, -0.2, 2.0],
                              [2.0, 0.3, 2.0]]], device=DEVICE).unsqueeze(0)
    occ_colors = torch.zeros((1, 1, 3, 5), device=DEVICE)
    occ_colors[0, 0] = torch.tensor([0.3, 0.3, 0.3, 0.0, 1.0], device=DEVICE)
    pn_parts = _pn_parts_from_mesh(occluder, torch.zeros_like(occluder),
                                   occ_colors)

    cam = torch.tensor([[0.0, 0.0, 8.0]], device=DEVICE)
    sp = torch.tensor([[0.0, 0.0, 5.0]], device=DEVICE)
    pbx = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    pby = torch.tensor([[0.0, 1.0, 0.0]], device=DEVICE)
    light_pos = torch.tensor([[[2.0, 0.0, 4.0]]], device=DEVICE)
    light_col = torch.tensor([[[1.0, 1.0, 1.0]]], device=DEVICE)

    img = _run_kernel(tri_bvh, tri_verts, tri_colors, cam, sp, pbx, pby,
                      1, W, H, bg=0, max_bounces=0, samples_per_pixel=256,
                      physical=True, light_pos=light_pos,
                      light_col=light_col, pn_parts=pn_parts
                      ).view(H, W, 4).float()
    row = H // 2
    lit = img[row, _floor_pixel(-2.5, W, H), 0]
    shadowed = img[row, _floor_pixel(2.0, W, H), 0]
    assert lit > 60, f"floor unexpectedly dark away from the occluder: {lit}"
    assert shadowed < 30, f"PN occluder cast no shadow: {shadowed}"
    print(f"ok: PN occluder shadows the point light "
          f"(lit={lit:.0f} shadow={shadowed:.0f})")


if __name__ == "__main__":
    test_morton_spread()
    test_stbvh_structure()
    test_blended_render_vs_brute_force()
    test_deep_translucent_stack()
    test_monte_carlo_converges_to_blend()
    test_mirror_reflection()
    test_glossy_reflection_blurs()
    test_indirect_color_bleed()
    test_physical_direct_lighting_and_shadows()
    test_physical_emissive_surface()
    test_pn_flat_matches_triangle_reference()
    test_pn_paraboloid_analytic()
    test_pn_watertight_seam()
    test_pn_mirror_and_monte_carlo()
    test_pn_physical_shadow()
    print("all raytracing unit tests passed")
