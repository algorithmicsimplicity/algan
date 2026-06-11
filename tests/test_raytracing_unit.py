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
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from algan.rendering.raytracing.ray_trace_taichi import (
    MIN_ALPHA,
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
    """Return int [T, N]: how many leaves cover each (frame, prim) pair."""
    covered = torch.zeros((num_frames, num_prims), dtype=torch.int32)
    first = bvh.first_leaf
    prim = bvh.leaf_prim.long().cpu()
    t0 = bvh.node_tmin[first:].long().cpu()
    t1 = bvh.node_tmax[first:].long().cpu()
    for i in range(bvh.num_leaves):
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
            assert (bvh.node_lo[parent] <= bvh.node_lo[child] + 1e-5).all()
            assert (bvh.node_hi[parent] >= bvh.node_hi[child] - 1e-5).all()
            assert bvh.node_tmin[parent] <= bvh.node_tmin[child]
            assert bvh.node_tmax[parent] >= bvh.node_tmax[child]
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


def _run_kernel(tri_bvh, tri_verts, tri_colors, cam, sp, pbx, pby, T, W, H,
                bg=20, max_bounces=0, samples_per_pixel=0, indirect=0.0):
    """Launch the deterministic kernel, or the Monte Carlo kernel when
    ``samples_per_pixel > 0``.
    """
    bez_bvh, meta, ccolors, bcolors, edges, offsets = _dummy_bezier_parts()
    out = torch.full((T, W * H, 4), bg, dtype=torch.uint8, device=DEVICE)
    scale = torch.full((T,), 1e-3, device=DEVICE)
    shared = (
        tri_bvh.node_lo, tri_bvh.node_hi, tri_bvh.node_tmin, tri_bvh.node_tmax,
        tri_bvh.node_miss, tri_bvh.leaf_prim, tri_bvh.first_leaf,
        tri_verts.contiguous(), tri_colors.contiguous(),
        bez_bvh.node_lo, bez_bvh.node_hi, bez_bvh.node_tmin, bez_bvh.node_tmax,
        bez_bvh.node_miss, bez_bvh.leaf_prim, bez_bvh.first_leaf,
        meta, ccolors, bcolors, edges, offsets,
        cam.contiguous(), sp.contiguous(), pbx.contiguous(), pby.contiguous(),
        scale, 0, T, W, H, float(W // 2), float(H // 2),
        0.0, max_bounces, 0)
    if samples_per_pixel > 0:
        path_trace_scene_stbvh(*shared, samples_per_pixel, indirect, out)
    else:
        render_scene_stbvh(*shared, out)
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
        order_key = torch.where(valid, t_hit, torch.full_like(t_hit, 1e30))
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
    bvh = build_stbvh(lo.contiguous(), hi.contiguous(), num_frames=T)
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


if __name__ == "__main__":
    test_morton_spread()
    test_stbvh_structure()
    test_blended_render_vs_brute_force()
    test_monte_carlo_converges_to_blend()
    test_mirror_reflection()
    test_glossy_reflection_blurs()
    test_indirect_color_bleed()
    print("all raytracing unit tests passed")
