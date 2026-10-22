"""The OptiX side path: GAS build, shadow and gather queries, and the fallback.

The device tests need CUDA and a driver with OptiX 9.1 and RT cores, and skip
otherwise. They check the hardware queries against a float64 brute force over
the same triangles, with the exact set semantics the software walk has:
visibility and shadow casting per primitive, the opaque/translucent split,
``_comes_after`` order and the k-buffer's cut after the first opaque hit. A
handful of rays whose hit lies within float32 noise of a triangle edge may
legitimately differ (the hardware's watertight test is its own), so each
comparison allows a small fraction.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing import settings as rt_settings

_BIN = 1e-4
_MIN_HIT = 1e-4


def _runtime():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from algan.rendering.raytracing.optix import runtime_for

    runtime = runtime_for("cuda")
    if runtime is None:
        pytest.skip("OptiX with RT cores is not available here")
    return runtime


def _scene(seed, num, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    centre = torch.rand((num, 1, 3), generator=g) * 4.0 - 2.0
    verts = centre + (torch.rand((num, 3, 3), generator=g) - 0.5) * 0.8
    tri_pos = verts.reshape(1, num, 9).float().to(device)
    valid = torch.rand((1, num), generator=g) > 0.1
    opaque = torch.rand((1, num), generator=g) > 0.5
    casts = torch.rand((1, num), generator=g) > 0.1
    return tri_pos, valid.to(device), opaque.to(device), casts.to(device)


def _rays(seed, count, device="cuda"):
    g = torch.Generator().manual_seed(seed + 1)
    origin = torch.rand((count, 3), generator=g) * 6.0 - 3.0
    target = torch.rand((count, 3), generator=g) * 2.0 - 1.0
    direction = target - origin
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return origin.to(device), direction.to(device)


def _brute_hits(tri_pos, origin, direction):
    """float64 Moller-Trumbore: ``t, b1, b2, hit`` of every (ray, triangle)."""
    v = tri_pos[0].double().view(-1, 3, 3)
    o = origin.double()[:, None, :]
    d = direction.double()[:, None, :]
    e1 = (v[:, 1] - v[:, 0])[None]
    e2 = (v[:, 2] - v[:, 0])[None]
    d, e1, e2 = torch.broadcast_tensors(d, e1, e2)
    pv = torch.cross(d, e2, dim=-1)
    det = (e1 * pv).sum(-1)
    inv = 1.0 / det
    tv = o - v[None, :, 0]
    b1 = (tv * pv).sum(-1) * inv
    qv = torch.cross(tv, e1, dim=-1)
    b2 = (d * qv).sum(-1) * inv
    t = (e2 * qv).sum(-1) * inv
    hit = (det.abs() > 1e-12) & (b1 >= 0) & (b2 >= 0) & (b1 + b2 <= 1)
    return t, b1, b2, hit


def _order_key(t, prim):
    """_comes_after as a sort key: depth bin, then descending layer."""
    return (int(torch.floor(torch.tensor(float(t), dtype=torch.float32) / _BIN)), -prim)


def _edge_distance(b1, b2):
    return torch.minimum(torch.minimum(b1, b2), 1.0 - b1 - b2)


def test_gas_builds_one_structure_per_distinct_frame():
    from algan.rendering.raytracing.optix.gas import build_batch_gas

    runtime = _runtime()
    tri_pos, valid, opaque, casts = _scene(0, 500)
    gas = build_batch_gas(runtime, tri_pos, valid, opaque, casts=casts)
    assert gas.num_handles == 1
    assert gas.triangles == 500
    assert int(gas.handles[0]) != 0
    # A per-frame table makes it one GAS per frame.
    frames = valid.expand(3, -1).clone()
    frames[1] = False
    gas3 = build_batch_gas(runtime, tri_pos, frames, opaque)
    assert gas3.num_handles == 3
    assert len(set(gas3.handles.tolist())) == 3
    # Tables that disagree on the frame count are refused, not guessed at.
    assert build_batch_gas(runtime, tri_pos, frames, opaque.expand(2, -1)) is None


def _shadow_launch(runtime, gas, origin, direction, tmax):
    from algan.rendering.raytracing.optix.runtime import SHADOW_HITS, new_params

    rays = origin.shape[0]
    queue = torch.cat([origin, direction, tmax[:, None]], -1).contiguous()
    valid = torch.ones(rays, dtype=torch.int32, device="cuda")
    layout = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    occ = torch.full((rays, 3), -1.0, device="cuda")
    frame = torch.zeros(rays, dtype=torch.int32, device="cuda")
    status = torch.full((rays,), -7, dtype=torch.int32, device="cuda")
    hits = torch.zeros((rays, 4 * SHADOW_HITS), device="cuda")
    p = new_params()
    p.handles = gas.handles.data_ptr()
    p.num_handles = gas.num_handles
    p.min_hit_distance = _MIN_HIT
    p.inv_bin = 1.0 / _BIN
    p.bin_width = _BIN
    p.layer_offset = 0.0
    p.queue_data = queue.data_ptr()
    p.queue_valid = valid.data_ptr()
    p.layout = layout.data_ptr()
    p.occ_out = occ.data_ptr()
    p.event_frame = frame.data_ptr()
    p.status = status.data_ptr()
    p.hits = hits.data_ptr()
    p.num_rays = rays
    p.num_events = rays
    p.event_first = 0
    runtime.launch("shadow", p, rays)
    torch.cuda.synchronize()
    return status.cpu(), occ.cpu(), hits.cpu()


def test_shadow_rays_match_a_brute_force_over_the_same_triangles():
    from algan.rendering.raytracing.optix.gas import build_batch_gas
    from algan.rendering.raytracing.optix.runtime import SHADOW_HITS

    runtime = _runtime()
    tri_pos, valid, opaque, casts = _scene(1, 400)
    gas = build_batch_gas(runtime, tri_pos, valid, opaque, casts=casts)
    origin, direction = _rays(1, 4096)
    tmax = torch.full((4096,), 4.0, device="cuda")
    status, occ, hits = _shadow_launch(runtime, gas, origin, direction, tmax)

    t, b1, b2, hit = _brute_hits(tri_pos, origin, direction)
    live = (valid[0] & casts[0]).cpu()
    opq = opaque[0].cpu()
    t, b1, b2, hit = t.cpu(), b1.cpu(), b2.cpu(), hit.cpu()
    inside = hit & live[None] & (t >= _MIN_HIT) & (t <= 4.0)
    # Rays within float32 noise of a decision: a triangle edge, or the ends
    # of the segment.
    near_edge = (_edge_distance(b1, b2).abs() < 1e-5) | (
        hit & (((t - 4.0).abs() < 1e-5) | ((t - _MIN_HIT).abs() < 1e-5))
    )
    mismatches = 0
    answered = listed = 0
    for r in range(4096):
        if (near_edge[r] & live).any():
            continue
        prims = inside[r].nonzero().flatten().tolist()
        blocked = any(bool(opq[q]) for q in prims)
        if blocked:
            ok = int(status[r]) == 1 and occ[r].tolist() == [1.0, 1.0, 1.0]
            answered += 1
        elif not prims:
            ok = int(status[r]) == 1 and occ[r].tolist() == [0.0, 0.0, 0.0]
            answered += 1
        else:
            listed += 1
            ok = int(status[r]) == 2 + len(prims)
            want = sorted(prims, key=lambda q: _order_key(t[r, q], q))
            got = hits[r].view(SHADOW_HITS, 4)
            for k, q in enumerate(want[:SHADOW_HITS]):
                ok = ok and int(got[k, 1].view(torch.int32)) == q
                ok = ok and abs(float(got[k, 0]) - float(t[r, q])) < 1e-4
                ok = ok and abs(float(got[k, 2]) - float(b1[r, q])) < 1e-3
                ok = ok and abs(float(got[k, 3]) - float(b2[r, q])) < 1e-3
        mismatches += not ok
    assert answered > 1000
    assert listed > 100
    assert mismatches == 0


def _collect_reference(t, b1, b2, hit, live, opq, t_prev, layer_prev, kbuf):
    """_collect_hits' consumable content for one ray.

    The first ``kbuf`` hits after the cursor in order, cut after the first
    opaque one.
    """
    cursor = _order_key(t_prev, int(layer_prev)) if layer_prev < 1e29 else None
    cands = []
    for q in (hit & live).nonzero().flatten().tolist():
        if not float(t[q]) > _MIN_HIT:
            continue
        key = _order_key(t[q], q)
        if cursor is not None and not key > cursor:
            continue
        cands.append((key, q))
    cands.sort()
    out = []
    for _, q in cands:
        out.append(q)
        if bool(opq[q]) or len(out) == kbuf:
            break
    return out


def test_gather_matches_the_kbuffer_semantics_of_collect_hits():
    from algan.rendering.raytracing.optix import gather_kbuf
    from algan.rendering.raytracing.optix.gas import build_batch_gas
    from algan.rendering.raytracing.optix.gather import gather

    runtime = _runtime()
    kbuf = gather_kbuf()
    tri_pos, valid, opaque, _ = _scene(2, 300)
    # Coplanar duplicates on a few triangles: same depth bin, layer decides.
    tri_pos[0, 1::7] = tri_pos[0, 0:-1:7][: tri_pos[0, 1::7].shape[0]]
    gas = build_batch_gas(runtime, tri_pos, valid, opaque)
    rays = 4096
    origin, direction = _rays(2, rays)
    t, b1, b2, hit = (x.cpu() for x in _brute_hits(tri_pos, origin, direction))
    # Half the rays restart behind a hit they already consumed.
    g = torch.Generator().manual_seed(5)
    t_prev = torch.zeros(rays)
    layer_prev = torch.full((rays,), 1e30)
    live = valid[0].cpu()
    for r in range(0, rays, 2):
        prims = (hit[r] & live & (t[r] > _MIN_HIT)).nonzero().flatten()
        if prims.numel():
            q = int(prims[torch.randint(prims.numel(), (1,), generator=g)])
            t_prev[r] = t[r, q]
            layer_prev[r] = float(q)
    sca = torch.zeros((rays, 8))
    sca[:, 1] = t_prev
    sca[:, 2] = layer_prev
    rs_int = torch.full((rays, 5), -3, dtype=torch.int32, device="cuda")
    hit_f = torch.zeros((kbuf, 4, rays), device="cuda")
    hit_i = torch.zeros((kbuf, 2, rays), dtype=torch.int32, device="cuda")
    active = torch.arange(rays, dtype=torch.int32, device="cuda")
    gather(
        runtime,
        gas,
        active=active,
        num_active=rays,
        rs_ro=origin.contiguous(),
        rs_rd=direction.contiguous(),
        rs_sca=sca.cuda(),
        rs_int=rs_int,
        rs_pix=torch.zeros(rays, dtype=torch.int32, device="cuda"),
        hit_f=hit_f,
        hit_i=hit_i,
        time_start=0,
        ray_offset=0,
        pixels_per_frame=1 << 20,
        layer_offset=0.0,
    )
    torch.cuda.synchronize()
    counts = rs_int[:, 3].cpu()
    hf, hi = hit_f.cpu(), hit_i.cpu()
    opq = opaque[0].cpu()
    edge = _edge_distance(b1, b2).abs() < 1e-5
    mismatches = checked = 0
    for r in range(rays):
        if (edge[r] & live).any():
            continue
        want = _collect_reference(
            t[r],
            b1[r],
            b2[r],
            hit[r],
            live,
            opq,
            float(t_prev[r]),
            float(layer_prev[r]),
            kbuf,
        )
        checked += 1
        got = sorted(
            (int(hi[q, 0, r]) for q in range(int(counts[r]))),
            key=lambda q: _order_key(t[r, q], q),
        )
        ok = got == want
        for q in range(int(counts[r])):
            prim = int(hi[q, 0, r])
            ok = ok and abs(float(hf[q, 0, r]) - float(t[r, prim])) < 1e-4
            ok = ok and float(hf[q, 1, r]) == float(prim)
            ok = ok and (int(hi[q, 1, r]) & 3) == 1
        mismatches += not ok
    assert checked > 3500
    assert mismatches == 0


def test_shadow_tracer_falls_back_when_optix_is_unavailable(monkeypatch):
    from algan.rendering.raytracing.optix import shadows
    from algan.rendering.raytracing.shadow_queue import make_shadow_tracer
    from algan.utils.memory_utils import ManualMemory

    monkeypatch.setattr(rt_settings, "optix", True)
    monkeypatch.setattr(rt_settings, "shadow_ray_parallel", False)
    monkeypatch.setattr(shadows, "batch_runtime", lambda device: None)
    calls = []
    monkeypatch.setattr(kernels, "raster_shadow_trace", lambda *a: calls.append(a))
    arena = ManualMemory(0, device="cpu", num_bytes=4096)
    tracer = make_shadow_tracer(arena, merged={})
    assert tracer is not kernels.raster_shadow_trace
    params = dict.fromkeys(kernels._RASTER_SHADOW_TRACE_PARAMS, 0)
    params.update(
        num_events=3,
        num_lights=1,
        has_tri=1,
        has_bez=0,
        tri_pos=torch.zeros(
            (1, 2, 9), device="cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    tracer(*params.values())
    assert len(calls) == 1
    # Off, or without a merged scene, the software tracer is returned as is.
    monkeypatch.setattr(rt_settings, "optix", False)
    assert make_shadow_tracer(arena, merged={}) is kernels.raster_shadow_trace
    monkeypatch.setattr(rt_settings, "optix", True)
    assert make_shadow_tracer(arena) is kernels.raster_shadow_trace


def _mixed_triangle_scene():
    """Opaque, half-transparent and glass solids and a mirror on a lit wall."""
    from algan import (
        BLUE,
        OUT,
        RED,
        RIGHT,
        UP,
        WHITE,
        Cube,
        MeshLambertMaterial,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        PointLight,
        Prism,
        Sphere,
    )

    PointLight(
        location=RIGHT * 1.0 + OUT * 6.0 + UP * 1.5, color=WHITE, intensity=20.0
    ).spawn(animate=False)
    Prism(width=14.0, height=9.0, depth=0.1).set_material(
        MeshLambertMaterial(color=WHITE)
    ).spawn(animate=False)
    Cube(size=1.6).set_material(MeshLambertMaterial(color=RED)).move(
        RIGHT * -2.5 + OUT * 1.0
    ).spawn(animate=False)
    Cube(size=1.8, opacity=0.5).set_material(MeshLambertMaterial(color=BLUE)).move(
        RIGHT * 0.5 + OUT * 1.4
    ).spawn(animate=False)
    Sphere(radius=1.0).set_material(
        MeshPhysicalMaterial(color=WHITE, transmission=1.0, roughness=0.0, ior=1.5)
    ).move(RIGHT * 3.0 + OUT * 1.5 + UP * 1.0).spawn(animate=False)
    Sphere(radius=0.7).set_material(
        MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.0)
    ).move(RIGHT * -1.0 + OUT * 2.2 + UP * -1.5).spawn(animate=False)


def _render_mixed(path, optix, max_bounces=0, circuits=False):
    import numpy as np
    from PIL import Image

    from algan import BLACK, SETTINGS, SMOKE_TEST, Off, Scene
    from algan.scene_manager import SceneManager

    video = SMOKE_TEST.set(resolution=(128, 96))
    SceneManager.reset()
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(shadows=True, max_bounces=max_bounces)
        SETTINGS.raytracing.experimental.set(optix=optix)
        with Scene(video_settings=video) as scene:
            scene.set_background(BLACK)
            with Off():
                _mixed_triangle_scene()
                if circuits:
                    from algan import OUT, RED, RIGHT, UP, Circle, Square

                    Square(size=1.5, color=RED, opacity=0.6).move(
                        RIGHT * -0.5 + OUT * 3.0 + UP * -1.0
                    ).spawn(animate=False)
                    Circle(radius=0.5, color=RED).move(
                        RIGHT * 1.5 + OUT * 2.5 + UP * -1.8
                    ).spawn(animate=False)
            result = scene.save_frame(str(path), video_settings=video, overwrite=True)
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGB")).astype(np.int16)
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


@pytest.mark.parametrize("circuits", [False, True])
def test_shadow_rays_of_a_render_agree_with_the_software_walk(
    tmp_path, monkeypatch, circuits
):
    """Every phase-0 queue ray of a real render, traced both ways.

    With circuits in the batch the triangles still go to the RT cores and the
    circuits' transmittance is marched in software and multiplied on.
    """
    import numpy as np

    from algan.rendering.raytracing.optix import shadows

    _runtime()
    monkeypatch.setattr(shadows, "VERIFY", True)
    monkeypatch.setattr(shadows, "COUNTING", True)
    for stats in (shadows.VERIFY_STATS, shadows.STATS):
        for key in stats:
            stats[key] = 0 if key != "max_abs" else 0.0
    hardware = _render_mixed(tmp_path / "on.png", True, circuits=circuits)
    software = _render_mixed(tmp_path / "off.png", False, circuits=circuits)
    verify = shadows.VERIFY_STATS
    assert verify["rays"] > 5000
    assert shadows.STATS["listed"] > 100  # the translucent casters were listed
    assert shadows.STATS["answered"] > 1000
    # Last-bit differences of the triangle test only: a ray may change sides
    # of an edge, nothing more.
    assert verify["within_1e-6"] >= 0.995 * verify["rays"]
    assert verify["lit_vs_blocked"] <= 0.003 * verify["rays"]
    delta = np.abs(hardware - software).max(-1)
    assert int((delta > 2).sum()) <= 0.002 * delta.size


def test_one_launch_over_every_tap_matches_the_two_phase_schedule(
    tmp_path, monkeypatch
):
    """One launch over every tap renders what the two-phase schedule does.

    The single launch traces a hard fan's off-diagonal taps whether or not
    the reduce reads them.
    """
    import numpy as np

    from algan.rendering.raytracing.optix import shadows

    _runtime()
    monkeypatch.setattr(shadows, "TWO_PHASE", True)
    two = _render_mixed(tmp_path / "two.png", True)
    monkeypatch.setattr(shadows, "TWO_PHASE", False)
    one = _render_mixed(tmp_path / "one.png", True)
    assert np.array_equal(one, two)

