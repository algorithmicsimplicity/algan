"""Opaque-first shadow queries (shadow mode 5) answer exactly as the march does.

``raytrace_kernels_taichi._shadow_opaque_first`` settles a shadow ray with one
any-hit pass when it is blocked by an opaque caster the ordered march is
certain to reach, or touches nothing at all, and hands everything else to the
ordered march. The claim is bit-identity with the march (shadow mode 1), so
the tests here hold it to that:

* the host selection (``shadow_dispatch``) and its kill switch;
* per ray, over a scene captured from a real render -- opaque, half
  transparent, glass and circuit casters, both BVH kinds -- with rays aimed at
  surface points, triangle edges and vertices (``max_t`` landing on a surface:
  the light's own depth bin) and non-finite rays, comparing the occlusion bits
  of modes 1 and 5 and requiring every outcome of the first pass to occur;
* whole frames on both deterministic routes, mode 5 against the
  ``opaquefirst`` kill switch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from algan import (
    BLACK,
    BLUE,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    Circle,
    Cube,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    Off,
    PointLight,
    Polyhedron,
    Prism,
    Scene,
    Sphere,
    Square,
)
from algan.animation_timeline import timeline
from algan.rendering.raytracing import raytrace_kernels_taichi as rk
from algan.rendering.raytracing import scene_builder
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shadow_dispatch import _select_shadow_mode
from algan.rendering.taichi_runtime import init_taichi
from algan.scene_manager import SceneManager
from algan.taichi_compat import ti

_SETTINGS = SMOKE_TEST.set(resolution=(96, 64))

_FACTS = (
    "has_transmissive",
    "tri_has_translucent",
    "bez_has_translucent",
    "has_uncertain_texture_alpha",
)


@pytest.fixture
def opt_disabled():
    """Swap ``ALGAN_OPT_DISABLE`` in-process, as the A/B scripts do."""
    saved = timeline._OPT_DISABLED

    def use(*names):
        timeline._OPT_DISABLED = frozenset(names)

    yield use
    timeline._OPT_DISABLED = saved


def _mixed_batch(**changes):
    merged = dict.fromkeys(_FACTS, False)
    merged.update(
        tri_has_translucent=True,
        has_transmissive=True,
        tri_has_opaque=True,
        bez_has_opaque=False,
    )
    merged.update(changes)
    return merged


def test_auto_takes_opaque_first_for_mixed_batches_with_opaque_casters(opt_disabled):
    opt_disabled()
    assert _select_shadow_mode(True, "auto", _mixed_batch()) == 5
    # Circuits alone can be the opaque casters.
    assert (
        _select_shadow_mode(
            True, "auto", _mixed_batch(tri_has_opaque=False, bez_has_opaque=True)
        )
        == 5
    )
    # Nothing opaque to stop a ray early: every ray would march anyway.
    assert _select_shadow_mode(True, "auto", _mixed_batch(tri_has_opaque=False)) == 1
    # Missing opacity facts are uncertainty, never a reason to skip the march.
    merged = _mixed_batch()
    del merged["tri_has_opaque"]
    assert _select_shadow_mode(True, "auto", merged) == 1
    # A provably opaque batch keeps the plain any-hit, and the explicit
    # experimental modes are untouched.
    assert _select_shadow_mode(True, "auto", dict.fromkeys(_FACTS, False)) == 3
    assert _select_shadow_mode(True, True, _mixed_batch(has_transmissive=False)) == 2
    assert _select_shadow_mode(True, "gather", _mixed_batch()) == 4
    assert _select_shadow_mode(True, False, _mixed_batch()) == 1
    assert _select_shadow_mode(False, "auto", _mixed_batch()) == 0


def test_opaque_first_kill_switch_and_its_exactness_preconditions(
    opt_disabled, monkeypatch
):
    opt_disabled("opaquefirst")
    assert _select_shadow_mode(True, "auto", _mixed_batch()) == 1
    opt_disabled()
    # Under a min_alpha below what an opaque blocker leaves of the light, the
    # march would not end at exactly zero: no opaque-first.
    monkeypatch.setattr(rk, "min_alpha", 1e-6)
    assert _select_shadow_mode(True, "auto", _mixed_batch()) == 1
    monkeypatch.setattr(rk, "min_alpha", 1e-3)
    # A short surface cap could stop the march before the blocker.
    monkeypatch.setattr(rk, "max_surfaces_per_ray", 64)
    assert _select_shadow_mode(True, "auto", _mixed_batch()) == 1


def _build_mixed_scene():
    """Opaque, half transparent and glass solids, a translucent pane and two
    circuits (one opaque, one half transparent) in front of a lit wall.
    """
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
    Cube(size=0.9).set_material(MeshLambertMaterial(color=WHITE)).move(
        RIGHT * 0.2 + OUT * 0.5 + UP * 0.2
    ).spawn(animate=False)
    Sphere(radius=1.0).set_material(
        MeshPhysicalMaterial(color=WHITE, transmission=1.0, roughness=0.0, ior=1.5)
    ).move(RIGHT * 3.0 + OUT * 1.5 + UP * 1.0).spawn(animate=False)
    Square(size=2.0, color=WHITE, opacity=0.4).move(
        RIGHT * -0.8 + OUT * 3.0 + UP * 1.2
    ).spawn(animate=False)
    Circle(radius=0.7, color=RED).move(RIGHT * -2.0 + OUT * 3.6 + UP * 1.8).spawn(
        animate=False
    )
    Circle(radius=0.6, color=WHITE, opacity=0.5).move(
        RIGHT * 2.0 + OUT * 3.2 + UP * -1.5
    ).spawn(animate=False)


def _render(path, *, hybrid_raster=True):
    SceneManager.reset()
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(shadows=True, max_bounces=0)
        SETTINGS.raytracing.experimental.set(hybrid_raster=hybrid_raster)
        with Scene(video_settings=_SETTINGS) as scene:
            scene.set_background(BLACK)
            with Off():
                _build_mixed_scene()
            result = scene.save_frame(
                str(path), video_settings=_SETTINGS, overwrite=True
            )
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGBA")).copy()
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


@pytest.fixture(scope="module")
def captured_scenes(tmp_path_factory):
    """The merged scene of a real render, once per BVH kind."""
    init_taichi()
    real_merge = scene_builder._merge_scene
    saved_refit = rt_settings.bvh_refit
    scenes = {}
    try:
        for tree, refit in (("refit", True), ("stbvh", False)):
            captured = []

            def capture(*args, _into=captured, **kwargs):
                merged = real_merge(*args, **kwargs)
                _into.append(merged)
                return merged

            scene_builder._merge_scene = capture
            rt_settings.bvh_refit = refit
            _render(tmp_path_factory.mktemp(tree) / "capture.png")
            assert captured, "the render merged no scene"
            scenes[tree] = captured[-1]
    finally:
        scene_builder._merge_scene = real_merge
        rt_settings.bvh_refit = saved_refit
    return scenes


@ti.kernel
def _occlusion_probe(
    mode: ti.template(),
    refit: ti.template(),
    has_tri: ti.template(),
    has_bez: ti.template(),
    ident: ti.template(),
    rays: ti.types.ndarray(),
    src: ti.types.ndarray(),
    occ: ti.types.ndarray(),
    outcome: ti.types.ndarray(),
    t_nodes: rk.NODE_ARG,
    t_node_miss: ti.types.ndarray(),
    t_leaf_prim: ti.types.ndarray(),
    t_leaf_tspan: ti.types.ndarray(),
    t_first_leaf: ti.i32,
    tri_pos: ti.types.ndarray(),
    tri_colors: ti.types.ndarray(),
    tri_uvs: ti.types.ndarray(),
    tri_tex_meta: ti.types.ndarray(),
    textures: ti.types.ndarray(),
    tri_extra: ti.types.ndarray(),
    num_colored_triangles: ti.i32,
    b_nodes: rk.NODE_ARG,
    b_node_miss: ti.types.ndarray(),
    b_leaf_prim: ti.types.ndarray(),
    b_leaf_tspan: ti.types.ndarray(),
    b_first_leaf: ti.i32,
    circuit_meta: ti.types.ndarray(),
    circuit_colors: ti.types.ndarray(),
    circuit_border_colors: ti.types.ndarray(),
    edges_2d: ti.types.ndarray(),
    edge_accel: ti.types.ndarray(),
    tri_obj: ti.types.ndarray(),
    pixel_size_per_t: ti.f32,
    layer_offset_triangles: ti.f32,
    eps_self: ti.f32,
    eps_near: ti.f32,
):
    for r in range(rays.shape[0]):
        ro = ti.math.vec3(rays[r, 0], rays[r, 1], rays[r, 2])
        rd = ti.math.vec3(rays[r, 3], rays[r, 4], rays[r, 5])
        max_t = rays[r, 6]
        src_prim = src[r]
        src_sid = -1
        # Nested on purpose: one `and` would make the compile-time gate a
        # runtime test (see AGENTS.md, ruff and kernels).
        if ti.static(ident != 0):  # noqa: SIM102
            if src_prim >= 0:
                src_sid = tri_obj[0, src_prim]
        o = rk._shadow_occluded(
            refit,
            mode,
            ro,
            rd,
            0,
            0.0,
            max_t,
            pixel_size_per_t,
            0.0,
            None,
            layer_offset_triangles,
            has_tri,
            has_bez,
            t_nodes,
            t_node_miss,
            t_leaf_prim,
            t_leaf_tspan,
            t_first_leaf,
            tri_pos,
            tri_colors,
            tri_uvs,
            tri_tex_meta,
            textures,
            tri_extra,
            num_colored_triangles,
            b_nodes,
            b_node_miss,
            b_leaf_prim,
            b_leaf_tspan,
            b_first_leaf,
            circuit_meta,
            circuit_colors,
            circuit_border_colors,
            edges_2d,
            edge_accel,
            src_sid,
            src_prim,
            eps_self,
            eps_near,
            tri_obj,
            ident,
        )
        for c in ti.static(range(3)):
            occ[r, c] = o[c]
        if ti.static(mode == 5):
            inv_rd = ti.math.vec3(
                rk._safe_inverse(rd[0]),
                rk._safe_inverse(rd[1]),
                rk._safe_inverse(rd[2]),
            )
            outcome[r] = rk._shadow_opaque_first(
                refit,
                has_tri,
                has_bez,
                ro,
                rd,
                inv_rd,
                0,
                max_t,
                pixel_size_per_t,
                0.0,
                None,
                t_nodes,
                t_leaf_prim,
                t_leaf_tspan,
                t_first_leaf,
                tri_pos,
                b_nodes,
                b_leaf_prim,
                b_leaf_tspan,
                b_first_leaf,
                circuit_meta,
                edges_2d,
                edge_accel,
                src_sid,
                src_prim,
                eps_self,
                eps_near,
                tri_obj,
                ident,
            )


def _rays(merged, light, count, seed):
    """Shadow rays the march finds hard: from points on (and just off) the
    scene's triangles, triangle edges and vertices, toward surface points
    (``max_t`` exactly on a surface), the light and far away, plus rays that
    are not finite at all. Returns ``(rays [N, 7], src_prim [N])``.
    """
    gen = torch.Generator().manual_seed(seed)
    tri = merged["tri_pos"][0].reshape(-1, 3, 3).double().cpu()
    n = tri.shape[0]

    def surface_points(k, where):
        idx = torch.randint(0, n, (k,), generator=gen)
        if where == "inside":
            w = torch.rand((k, 3), generator=gen, dtype=torch.float64)
            w = w / w.sum(1, keepdim=True)
        elif where == "edge":
            a = torch.rand((k,), generator=gen, dtype=torch.float64)
            w = torch.zeros((k, 3), dtype=torch.float64)
            which = torch.randint(0, 3, (k,), generator=gen)
            w[torch.arange(k), which] = a
            w[torch.arange(k), (which + 1) % 3] = 1 - a
        else:
            w = torch.zeros((k, 3), dtype=torch.float64)
            w[torch.arange(k), torch.randint(0, 3, (k,), generator=gen)] = 1.0
        return (w.unsqueeze(-1) * tri[idx]).sum(1), idx

    lo = tri.reshape(-1, 3).amin(0) - 0.5
    hi = tri.reshape(-1, 3).amax(0) + 0.5
    origins, srcs, targets = [], [], []
    for where in ("inside", "edge", "vertex"):
        p, idx = surface_points(count, where)
        e0 = tri[idx, 1] - tri[idx, 0]
        e1 = tri[idx, 2] - tri[idx, 0]
        nrm = torch.linalg.cross(e0, e1)
        nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-30)
        lift = (torch.rand((count, 1), generator=gen, dtype=torch.float64) - 0.3) * 1e-3
        origins.append(p + nrm * lift)
        srcs.append(idx)
        q, _ = surface_points(
            count,
            ("inside", "edge", "vertex")[int(torch.randint(0, 3, (1,), generator=gen))],
        )
        far = lo + (hi - lo) * torch.rand(
            (count, 3), generator=gen, dtype=torch.float64
        )
        pick = torch.randint(0, 3, (count, 1), generator=gen)
        target = torch.where(
            pick == 0, q, torch.where(pick == 1, far, light.expand(count, 3))
        )
        targets.append(target)
    o = torch.cat(origins)
    t = torch.cat(targets)
    src = torch.cat(srcs).to(torch.int32)
    d = t - o
    dist = d.norm(dim=1, keepdim=True)
    d = d / dist.clamp_min(1e-30)
    max_t = dist.clone()
    max_t[torch.rand((max_t.shape[0], 1), generator=gen) < 0.1] = 1e7
    rays = torch.cat([o, d, max_t], 1).float()
    # Half the sources forget their triangle (the circuit-sourced convention).
    src[torch.rand(src.shape, generator=gen) < 0.5] = -1
    nan, inf = float("nan"), float("inf")
    special = torch.tensor(
        [
            [nan, 0, 0, 0, 0, 1, 10],
            [0, 0, 5, nan, 0, -1, 10],
            [0, 0, 5, 0, 0, -1, nan],
            [0, 0, 5, 0, 0, -1, inf],
            [0, 0, 5, inf, 0, -1, 10],
            [inf, 0, 5, -1, 0, 0, 10],
            [0, 0, 5, 0, 0, 0, 10],
            [0, 0, 5, 0, 0, -1, -1],
            [0, 0, 5, 0, 0, -1, 0],
            [0, 0, 5, 1e-30, 1e-30, -1, 10],
        ],
        dtype=torch.float32,
    )
    rays = torch.cat([rays, special])
    src = torch.cat([src, torch.full((special.shape[0],), -1, dtype=torch.int32)])
    return rays.contiguous(), src.contiguous()


def _launch(mode, merged, rays, src, ident):
    from algan.rendering.raytracing.refit_bvh import RefitBVH

    tri_bvh = merged["tri_bvh"]
    bez_bvh = merged["bez_bvh"]
    refit = 1 if isinstance(tri_bvh, RefitBVH) else 0
    device = merged["tri_pos"].device
    rays = rays.to(device)
    src = src.to(device)
    occ = torch.zeros((rays.shape[0], 3), dtype=torch.float32, device=device)
    outcome = torch.full((rays.shape[0],), -1, dtype=torch.int32, device=device)
    tri_obj = merged.get("tri_obj")
    if tri_obj is None:
        tri_obj = torch.zeros((1, 1), dtype=torch.int32, device=device)
    _occlusion_probe(
        mode,
        refit,
        1 if int(merged.get("num_triangles", 0)) > 0 else 0,
        1 if int(merged.get("num_circuits", 0)) > 0 else 0,
        ident,
        rays,
        src,
        occ,
        outcome,
        tri_bvh.kernel_blocks,
        tri_bvh.node_miss,
        tri_bvh.leaf_prim,
        tri_bvh.leaf_tspan,
        int(tri_bvh.first_leaf),
        merged["tri_pos"],
        merged["tri_colors"],
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        merged["tri_extra"],
        int(merged["num_colored_triangles"]),
        bez_bvh.kernel_blocks,
        bez_bvh.node_miss,
        bez_bvh.leaf_prim,
        bez_bvh.leaf_tspan,
        int(bez_bvh.first_leaf),
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        merged["edges_2d"],
        merged["edge_accel"],
        tri_obj.to(torch.int32),
        1e-3,
        float(merged.get("num_circuits", 0)),
        2e-4,
        0.0,
    )
    return occ.cpu(), outcome.cpu()


@pytest.mark.parametrize("tree", ["refit", "stbvh"])
@pytest.mark.parametrize("ident", [0, 1])
def test_opaque_first_matches_the_march_ray_for_ray(captured_scenes, tree, ident):
    merged = captured_scenes[tree]
    assert int(merged.get("num_circuits", 0)) > 0, "the scene must hold circuits"
    light = torch.tensor([[1.0, 1.5, 6.0]], dtype=torch.float64)
    rays, src = _rays(merged, light, 3000, seed=7 + ident)
    march, _ = _launch(1, merged, rays, src, ident)
    first, outcome = _launch(5, merged, rays, src, ident)
    differs = (march.view(torch.int32) != first.view(torch.int32)).any(1)
    assert int(differs.sum()) == 0, (
        f"{int(differs.sum())} of {rays.shape[0]} rays differ from the march, "
        f"first at {rays[differs][:3].tolist()}"
    )
    counts = {k: int((outcome == k).sum()) for k in (0, 1, 2)}
    # Every answer of the first pass actually happened, so each was compared.
    assert min(counts.values()) >= 50, counts
    # The non-finite rays (the first six specials) always fall through to the
    # march; the finite degenerate ones (zero direction, max_t 0 or -1, an
    # almost-axis direction) take the walk like any other ray.
    assert (outcome[-10:-4] == 2).all()


def _frame(tmp_path, name, *, hybrid_raster):
    return _render(tmp_path / f"{name}.png", hybrid_raster=hybrid_raster)


@pytest.mark.parametrize("hybrid_raster", [True, False], ids=("sheet", "classic"))
@pytest.mark.parametrize("refit", [True, False], ids=("refit", "stbvh"))
def test_opaque_first_frames_match_the_kill_switch(
    tmp_path, opt_disabled, monkeypatch, hybrid_raster, refit
):
    from algan.rendering.raytracing import tracer

    monkeypatch.setattr(rt_settings, "bvh_refit", refit)
    modes = []
    select = tracer._select_shadow_mode

    def recording(*args, **kwargs):
        mode = select(*args, **kwargs)
        modes.append(mode)
        return mode

    monkeypatch.setattr(tracer, "_select_shadow_mode", recording)
    opt_disabled()
    on = _frame(tmp_path, "on", hybrid_raster=hybrid_raster)
    assert set(modes) == {5}, modes
    modes.clear()
    opt_disabled("opaquefirst")
    off = _frame(tmp_path, "off", hybrid_raster=hybrid_raster)
    assert set(modes) == {1}, modes
    assert np.array_equal(on, off)
    # The shadows are really there: the frame is not one flat wall.
    assert len(np.unique(on.reshape(-1, 4), axis=0)) > 20


def _capture(path, build):
    """Render ``build``'s scene once and return its merged scene."""
    captured = []
    real_merge = scene_builder._merge_scene

    def capture(*args, **kwargs):
        merged = real_merge(*args, **kwargs)
        captured.append(merged)
        return merged

    scene_builder._merge_scene = capture
    SceneManager.reset()
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(shadows=True, max_bounces=0)
        with Scene(video_settings=_SETTINGS) as scene:
            scene.set_background(BLACK)
            with Off():
                build()
            scene.save_frame(str(path), video_settings=_SETTINGS, overwrite=True)
    finally:
        scene_builder._merge_scene = real_merge
        SETTINGS.restore(snapshot)
        SceneManager.reset()
    assert captured
    return captured[-1]


def _sheet(kind, x, z):
    """A one-sided unit sheet at depth ``z``: a circuit or two triangles."""
    if kind == "circuit":
        return Square(size=1.0, color=WHITE).move(RIGHT * x + OUT * z)
    corners = [
        [x - 0.5, -0.5, z],
        [x + 0.5, -0.5, z],
        [x + 0.5, 0.5, z],
        [x - 0.5, 0.5, z],
    ]
    return Polyhedron(corners, [[0, 1, 2, 3]]).set_material(
        MeshLambertMaterial(color=WHITE)
    )


@pytest.mark.parametrize("kind", ["circuit", "triangle"])
def test_a_blocker_in_the_lights_depth_bin_is_left_to_the_march(tmp_path, kind):
    """The march orders hits by depth bin, then by descending layer, and stops
    at the first one at or past ``max_t``. An opaque blocker just short of the
    light, in the light's own bin, behind (in that order) a surface past the
    light, is therefore never reached: the march says lit. The first pass must
    not call it blocked -- it leaves any opaque hit in the light's bin to the
    march, and dropping that rule (in either arm) makes this test fail.

    Two opaque sheets 5e-5 apart, both inside one ``depth_tie_epsilon`` bin
    along the rays, created in both orders so one pair has the far sheet on
    the higher layer; ``max_t`` falls between them.
    """
    init_taichi()
    z_near, z_far = 0.50008, 0.50003

    def build():
        PointLight(location=OUT * 6.0, color=WHITE, intensity=10.0).spawn(animate=False)
        for x, order in ((-1.0, (z_near, z_far)), (1.0, (z_far, z_near))):
            for z in order:
                _sheet(kind, x, z).spawn(animate=False)

    merged = _capture(tmp_path / "bin.png", build)
    assert (
        int(merged.get("num_circuits" if kind == "circuit" else "num_triangles", 0)) > 0
    )
    gen = torch.Generator().manual_seed(3)
    k = 400
    x = (
        torch.where(torch.arange(k) < k // 2, -1.0, 1.0)
        + (torch.rand(k, generator=gen) - 0.5) * 0.8
    )
    y = (torch.rand(k, generator=gen) - 0.5) * 0.8
    origin = torch.stack([x, y, torch.full((k,), 2.0)], 1)
    direction = torch.tensor([0.0, 0.0, -1.0]).expand(k, 3)
    # t_near = 1.49992 and t_far = 1.49997, both in bin 14999; max_t between.
    max_t = torch.full((k, 1), 2.0 - 0.500055)
    rays = torch.cat([origin, direction, max_t], 1).float().contiguous()
    src = torch.full((k,), -1, dtype=torch.int32)
    march, _ = _launch(1, merged, rays, src, 0)
    first, outcome = _launch(5, merged, rays, src, 0)
    assert torch.equal(march.view(torch.int32), first.view(torch.int32))
    # The pair whose far sheet takes the higher layer is lit by the march;
    # the other is blocked. Both happen, so the rule was exercised.
    lit = (march == 0).all(1)
    assert 0 < int(lit.sum()) < k
    # Neither may be answered "blocked" on a hit inside the light's bin.
    assert int((outcome == 1).sum()) == 0
