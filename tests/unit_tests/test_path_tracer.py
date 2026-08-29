"""Unit tests for the ``samples_per_pixel > 1`` path tracer.

Two layers:

* **Sampler tests** drive ``pt_sampler_probe`` directly -- the Sobol-Owen
  sampler is a pure function of ``(seed, frame, pixel, pair, sample index)``,
  so its stratification, purity and decorrelation are testable without the
  scene pipeline. That purity is a sampling-quality property (stratification,
  and independence from how a render was split into tiles and waves); the
  renderer does **not** promise that two runs produce identical frames, and
  nothing here asserts it (see ``DESIGN_path_tracer_roadmap.md``).
* **Render tests** drive the real dispatch through ``Scene.save_frame``:
  the path tracer's deterministic transparency must reproduce the
  deterministic renderer's composite on unlit 2-D content away from edges
  (edges legitimately differ: jittered-sample anti-aliasing vs analytic
  coverage), and its estimators must land on independently computed
  reference integrals.

The whole module sits outside the fast suite: the render tests compile the
path tracer's kernel variants, tens of seconds charged to whichever test
reaches them first.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    GREEN,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    PointLight,
    Prism,
    RectAreaLight,
    Scene,
    SceneManager,
    Sphere,
    Square,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 64x64 rather than SMOKE_TEST's 32x32: the stack scene needs interior area
# between the squares' borders for the flat-region comparison to bite.
STACK_SETTINGS = SMOKE_TEST.set(resolution=(64, 64))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def _probe(seed=0, f=0, pixel=0, pair=0, n=64):
    from algan.rendering.raytracing.path_tracer_taichi import pt_sampler_probe
    from algan.rendering.taichi_runtime import init_taichi

    # Algan's own init, never a bare ``ti.init`` (which would re-enable
    # advanced_optimization for everything compiled after it -- see
    # test_taichi_runtime_config.py).
    init_taichi()
    out = torch.zeros((n, 2), dtype=torch.float32, device=DEVICE)
    pt_sampler_probe(int(seed), int(f), int(pixel), int(pair), out)
    return out.cpu()


def _strata_counts(values, buckets):
    idx = (values * buckets).long().clamp_(0, buckets - 1)
    return torch.bincount(idx, minlength=buckets)


def test_sampler_prefixes_are_stratified():
    """(0,2)-sequence property, kept through the Owen shuffle/scramble: any
    prefix of ``4^m`` samples fills every elementary interval exactly once.
    """
    samples = _probe(n=16)
    for dim in range(2):
        assert (_strata_counts(samples[:4, dim], 4) == 1).all(), (
            f"first 4 samples not stratified in dim {dim}: {samples[:4, dim]}"
        )
        assert (_strata_counts(samples[:16, dim], 16) == 1).all(), (
            f"first 16 samples not 16-stratified in dim {dim}"
        )
    # Joint 2D stratification: one sample in each cell of the 2x2 and 4x4
    # grids.
    for m, count in ((2, 4), (4, 16)):
        cells = (samples[:count, 0] * m).long().clamp_(0, m - 1) * m + (
            samples[:count, 1] * m
        ).long().clamp_(0, m - 1)
        assert (torch.bincount(cells, minlength=m * m) == 1).all(), (
            f"first {count} samples not jointly {m}x{m} stratified"
        )


def test_sampler_is_reproducible_and_seed_sensitive():
    a = _probe(n=32)
    b = _probe(n=32)
    assert torch.equal(a, b), "identical inputs produced different samples"
    c = _probe(seed=1, n=32)
    assert not torch.equal(a, c), "the seed does not reach the sampler"


def test_sampler_decorrelates_pixels_and_pairs():
    base = _probe(n=32)
    for kwargs in ({"pixel": 1}, {"f": 1}, {"pair": 1}):
        other = _probe(n=32, **kwargs)
        assert not torch.equal(base, other), f"{kwargs} did not decorrelate"
        # Distinct sequences, but each still uniform: their difference should
        # not be a constant offset either.
        assert (base - other).abs().amax() > 0.05


def test_sampler_is_uniform():
    samples = _probe(n=1024)
    mean = samples.mean(0)
    assert (mean - 0.5).abs().max() < 0.01, f"sampler mean off: {mean}"
    assert samples.min() >= 0.0
    assert samples.max() < 1.0


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


def _render_stack_frame(tmp_path, name, samples_per_pixel):
    """One 64x64 frame of three overlapping translucent unlit squares.

    2-D circuits are unlit, so the frame isolates exactly what the path
    tracer's transport skeleton owns: deterministic front-to-back alpha
    compositing in author order, plus sub-pixel anti-aliasing.
    """
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        # denoise off throughout this module: it tests the estimator, and CI
        # must not depend on the denoiser weights being downloadable.
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        with Scene(video_settings=STACK_SETTINGS) as scene:
            with Off():
                Square(side_length=6.0, color=BLUE).spawn(animate=False)
                red = Square(side_length=4.0, color=RED).set_opacity(0.5)
                red.spawn(animate=False)
                green = Square(side_length=2.0, color=GREEN).set_opacity(0.25)
                green.spawn(animate=False)
            result = scene.save_frame(
                tmp_path / name,
                video_settings=STACK_SETTINGS,
                overwrite=True,
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    return result


def _read(result):
    frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    assert frame is not None, f"unreadable frame at {result.output_path}"
    return torch.from_numpy(frame.astype(np.int32))


def test_path_traced_transparency_matches_deterministic_compositing(tmp_path):
    """Away from geometry edges, the path-traced composite of an unlit
    translucent stack equals the deterministic renderer's: transparency is
    throughput-weighted (zero variance), never stochastic alpha.
    """
    det = _read(_render_stack_frame(tmp_path, "stack_det.png", 1))
    pt = _read(_render_stack_frame(tmp_path, "stack_pt.png", 8))
    assert det.shape == pt.shape

    # Flat-region mask from the deterministic frame: 3x3 neighborhoods whose
    # channel range is tiny are interior; the rest are edges where jittered
    # AA and analytic coverage legitimately differ.
    det_f = det.float().permute(2, 0, 1).unsqueeze(0)
    pooled_max = torch.nn.functional.max_pool2d(det_f, 3, stride=1, padding=1)
    pooled_min = -torch.nn.functional.max_pool2d(-det_f, 3, stride=1, padding=1)
    flat = (pooled_max - pooled_min).squeeze(0).amax(0) < 2
    assert flat.sum() > det.shape[0] * det.shape[1] // 4, (
        "the scene left too few interior pixels to compare"
    )
    err = (det - pt).abs().amax(-1)
    max_err = err[flat].max()
    assert max_err <= 1, (
        f"path-traced interior deviates from the deterministic composite by "
        f"{int(max_err)} (expected exact up to rounding); "
        f"{int((err[flat] > 1).sum())} of {int(flat.sum())} flat pixels off"
    )


def test_path_traced_plan_reports_the_backend(tmp_path):
    result = _render_stack_frame(tmp_path, "plan.png", 4)
    assert result.render_plan.samples_per_pixel == 4
    assert result.render_plan.backend == "path_tracer"


def test_seed_changes_the_noise_but_not_the_flat_interior(tmp_path):
    """pt_seed reaches the render (edge jitter changes) without disturbing
    the deterministic interior composite.
    """
    a = _read(_render_stack_frame(tmp_path, "seed0.png", 4))
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.experimental.set(pt_seed=7)
        b = _read(_render_stack_frame(tmp_path, "seed7.png", 4))
    finally:
        SETTINGS.restore(snapshot)
    assert not torch.equal(a, b), "pt_seed does not reach the sampler"
    det_f = a.float().permute(2, 0, 1).unsqueeze(0)
    pooled_max = torch.nn.functional.max_pool2d(det_f, 3, stride=1, padding=1)
    pooled_min = -torch.nn.functional.max_pool2d(-det_f, 3, stride=1, padding=1)
    flat = (pooled_max - pooled_min).squeeze(0).amax(0) < 2
    err = (a - b).abs().amax(-1)
    assert err[flat].max() <= 1, "the seed disturbed interior pixels"


# ---------------------------------------------------------------------------
# Transport (Stage 2: NEE + BSDF sampling)
# ---------------------------------------------------------------------------


def _render_scene(tmp_path, name, build, samples_per_pixel, video=None, **rt_kwargs):
    """Render one frame of ``build()``'s scene under the given settings."""
    settings = video if video is not None else STACK_SETTINGS
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        for key, value in rt_kwargs.items():
            SETTINGS.raytracing.set(**{key: value})
        with Scene(video_settings=settings) as scene:
            with Off():
                build(scene)
            result = scene.save_frame(
                tmp_path / name, video_settings=settings, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    return _read(result)


def test_camera_clip_planes_apply_under_path_tracing(tmp_path):
    """``camera.near`` and ``camera.far`` clip path-traced frames too.

    Both were regressions of the fallback role rather than of parity: a scene
    that needs the path tracer (for GI, memory or light count) must not
    silently lose a camera setting the deterministic renderer honours. Near
    clipping used to be inert here -- ``pt_generate`` built primaries without
    it -- while the feature matrix advertised it, and far clipping was
    unimplemented.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        Square(side_length=3.0, color=RED).spawn(animate=False)

    def with_camera(**clip):
        def inner(scene):
            build(scene)
            for name, value in clip.items():
                setattr(scene.camera, name, value)

        return inner

    base = _render_scene(tmp_path, "clip_base.png", build, 8).float()
    assert base.mean() > 5.0, "the unclipped scene rendered (nearly) empty"

    # A near plane in front of everything, and a far plane behind everything,
    # each leave the background alone.
    near = _render_scene(
        tmp_path, "clip_near.png", with_camera(near=100.0), 8
    ).float()
    assert near.max() == 0, f"near plane did not clip (max {int(near.max())})"
    far = _render_scene(tmp_path, "clip_far.png", with_camera(far=0.5), 8).float()
    assert far.max() == 0, f"far plane did not clip (max {int(far.max())})"

    # Planes generous enough to contain the scene change nothing.
    wide = _render_scene(
        tmp_path, "clip_wide.png", with_camera(near=0.1, far=1000.0), 8
    ).float()
    assert (wide - base).abs().max() <= 1, (
        "clip planes that contain the whole scene moved the image by "
        f"{int((wide - base).abs().max())}"
    )


def test_lambert_furnace_is_lossless(tmp_path):
    """White furnace, diffuse: a pure-white Lambert sphere in front of a
    pure-white background with no lights must render white everywhere -- the
    diffuse continuation carries exactly the albedo (no ambient fill, no
    hidden loss), and the leftover throughput picks up the background.
    """

    def build(scene):
        scene.set_background(WHITE)
        Scene.clear_light_sources()
        sphere = Sphere(radius=1.0)
        sphere.set_material(MeshLambertMaterial(color=WHITE))
        sphere.spawn(animate=False)

    img = _render_scene(tmp_path, "furnace_lambert.png", build, 16)
    lo = int(img.amin())
    assert lo >= 254, (
        f"diffuse furnace lost energy: darkest channel {lo} (expected white everywhere)"
    )


def test_ggx_furnace_keeps_energy_with_compensation(tmp_path):
    """White furnace, specular: a white metallic sphere (roughness 0.5) under
    a white background must stay near-white -- VNDF sampling with the Turquin
    compensation recovers the multiple-scattering energy single-scatter GGX
    loses (uncompensated, rough metal renders visibly dark).
    """

    def build(scene):
        scene.set_background(WHITE)
        Scene.clear_light_sources()
        sphere = Sphere(radius=1.0)
        sphere.set_material(
            MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.5)
        )
        sphere.spawn(animate=False)

    img = _render_scene(tmp_path, "furnace_ggx.png", build, 64).float()
    h, w = img.shape[0], img.shape[1]
    # The sphere covers the frame centre; sample its disc.
    disc = img[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8]
    mean = float(disc.mean())
    assert mean > 0.93 * 255, (
        f"GGX furnace lost energy: sphere disc mean {mean:.1f}/255"
    )
    assert mean <= 256, f"GGX furnace GAINED energy: {mean:.1f}/255"


def test_nee_direct_lighting_matches_deterministic(tmp_path):
    """A Lambert plane under one point light, no shadows: the path tracer's
    NEE uses the same ``_light_eval`` radiometry and stage formulas as the
    deterministic renderer, so flat interiors agree up to the deterministic
    renderer's small ambient fill (which real GI replaces) -- nothing here
    for GI to add (black background, single surface).
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=OUT * 5.0, color=WHITE, intensity=1.0).spawn(animate=False)
        plane = Prism(dimensions=(7.0, 7.0, 0.1))
        plane.set_material(MeshLambertMaterial(color=RED))
        plane.spawn(animate=False)

    det = _render_scene(tmp_path, "nee_det.png", build, 1)
    pt = _render_scene(tmp_path, "nee_pt.png", build, 32)
    det_f = det.float().permute(2, 0, 1).unsqueeze(0)
    pooled_max = torch.nn.functional.max_pool2d(det_f, 3, stride=1, padding=1)
    pooled_min = -torch.nn.functional.max_pool2d(-det_f, 3, stride=1, padding=1)
    flat = (pooled_max - pooled_min).squeeze(0).amax(0) < 2
    err = (det - pt).abs().amax(-1)
    assert flat.sum() > 500, "not enough flat pixels to compare"
    max_err = int(err[flat].max())
    assert max_err <= 5, (
        f"path-traced direct lighting deviates from the deterministic stage "
        f"by {max_err} (expected within the ambient-fill difference); "
        f"{int((err[flat] > 5).sum())} of {int(flat.sum())} flat pixels off"
    )


def test_point_light_shadow_under_path_tracing(tmp_path):
    """With ``shadows`` on, NEE visibility rays darken occluded geometry.

    Asserted as the physical invariant rather than at a guessed pixel: a
    shadow can only ever *remove* light, so turning ``shadows`` on must
    darken a meaningful part of the frame and brighten none of it.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=OUT * 6.0, color=WHITE, intensity=1.0).spawn(animate=False)
        floor = Prism(dimensions=(7.0, 7.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        blocker = Square(side_length=2.0, color=BLUE)
        blocker.move(OUT * 2.0)
        blocker.spawn(animate=False)

    lit = _render_scene(tmp_path, "shadow_off.png", build, 24, shadows=False)
    shadowed = _render_scene(tmp_path, "shadow_on.png", build, 24, shadows=True)
    delta = (lit - shadowed).amax(-1)
    darkened = int((delta > 2).sum())
    brightened = int((delta < -2).sum())
    assert darkened > 20, (
        f"shadows changed almost nothing: {darkened} pixels darkened "
        f"(max drop {int(delta.max())})"
    )
    assert brightened == 0, (
        f"{brightened} pixels got BRIGHTER with shadows on; a shadow can only "
        "remove light"
    )


def test_indirect_light_bleeds_color(tmp_path):
    """Global illumination: a red wall beside a white floor bleeds red onto
    it. The deterministic renderer cannot do this at all and the old Monte
    Carlo kernel needed an opt-in hack; here it is simply what the transport
    does.

    Isolated as a ``max_bounces`` A/B rather than a position probe: 0 bounces
    gates scattering off entirely, so the two renders share geometry, direct
    lighting and tonemapping, and every difference between them IS the
    indirect transport. Red must arrive, and it must arrive as *red*.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=(UP * 2.0 + OUT * 5.0), color=WHITE, intensity=1.2).spawn(
            animate=False
        )
        floor = Prism(dimensions=(8.0, 8.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        wall = Prism(dimensions=(0.1, 8.0, 3.0))
        wall.set_material(MeshLambertMaterial(color=RED))
        wall.move(RIGHT * 2.5 + OUT * 1.5)
        wall.spawn(animate=False)

    direct = _render_scene(tmp_path, "bleed_off.png", build, 48, max_bounces=0).float()
    gi = _render_scene(tmp_path, "bleed_on.png", build, 48).float()
    # OpenCV loads BGR: channel 2 is red, 0 is blue.
    gained = gi - direct
    red_gain = float(gained[..., 2].max())
    assert red_gain > 4, (
        f"indirect bounces added no red light (max red gain {red_gain:.1f})"
    )
    # The bounce came off a red wall, so it must carry red rather than lift
    # every channel equally.
    lit = gained[..., 2] > 4
    assert float(gained[..., 2][lit].mean() - gained[..., 0][lit].mean()) > 2, (
        "indirect light arrived achromatic; the red wall did not tint it"
    )


# ---------------------------------------------------------------------------
# Lights (Stage 3: the power-weighted next-event table -- area radiometry,
# emissive triangles with MIS, environment sampling)
# ---------------------------------------------------------------------------


def _render_scene_exp(
    tmp_path, name, build, samples_per_pixel, video=None, experimental=None, **rt_kwargs
):
    """``_render_scene`` plus experimental raytracing overrides (the parent
    section refuses experimental fields by design).
    """
    settings = video if video is not None else STACK_SETTINGS
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        for key, value in rt_kwargs.items():
            SETTINGS.raytracing.set(**{key: value})
        for key, value in (experimental or {}).items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        with Scene(video_settings=settings) as scene:
            with Off():
                build(scene)
            result = scene.save_frame(
                tmp_path / name, video_settings=settings, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    return _read(result)


def _center_patch_mean(img, half=2):
    """Mean over the RGB channels of the ``2 half x 2 half`` pixel patch at
    the image centre (BGR/BGRA tensors from ``_read``).
    """
    h, w = img.shape[0], img.shape[1]
    patch = img[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half]
    return float(patch[..., :3].double().mean())


def test_area_light_matches_the_deterministic_grid_limit(tmp_path):
    """A RectAreaLight under the path tracer samples a uniform point inside
    the selected row's cell and evaluates the falloff and one-sided cosine
    AT that point, so its expectation is the continuous area integral -- the
    limit the deterministic K-cell grid approximates. With a fine grid the
    two must agree on flat interior pixels. Runs with pt_light_samples = 2,
    so the 1/N weighting of multi-sample NEE is under test too.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        RectAreaLight(
            location=OUT * 3.0,
            width=3.0,
            height=3.0,
            samples=64,
            color=WHITE,
            intensity=1.0,
        ).spawn(animate=False)
        floor = Prism(dimensions=(7.0, 7.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)

    det = _render_scene(tmp_path, "area_det.png", build, 1, shadows=False)
    pt = _render_scene_exp(
        tmp_path,
        "area_pt.png",
        build,
        96,
        shadows=False,
        experimental={"pt_light_samples": 2},
    )
    # The lit floor is a smooth gradient (per-cell cosines), so a flatness
    # mask has nothing to grab; compare per-pixel over the floor's interior
    # (the central region sits well inside the 7x7 floor at this framing).
    h, w = det.shape[0], det.shape[1]
    core = (slice(h // 2 - 20, h // 2 + 20), slice(w // 2 - 20, w // 2 + 20))
    err = (det[..., :3] - pt[..., :3]).abs().amax(-1).float()[core]
    assert float(det[core][..., :3].float().mean()) > 40.0, (
        "the area light did not light the floor at all"
    )
    assert float(err.mean()) <= 4.0, (
        f"area-light radiometry drifted from the deterministic grid limit "
        f"(mean interior error {float(err.mean()):.2f})"
    )
    assert float(err.max()) <= 30.0, (
        f"area-light sampling left an outlier on the floor interior "
        f"(max interior error {float(err.max()):.1f})"
    )


def test_emissive_quad_matches_the_reference_integral(tmp_path):
    """An emissive quad lighting a diffuse floor, against the closed-form
    direct integral evaluated by torch quadrature.

    This is the MIS correctness test: the quad's light reaches the floor
    through BOTH strategies (next-event samples toward the quad, and
    diffuse-sampled rays that hit it), each carrying a power-heuristic
    weight. Double counting reads ~2x the reference and a lost strategy
    reads low, either far outside the tolerance, while correct weights land
    on the integral with only sampling noise. Tonemapping is disabled so
    pixel values are raw linear radiance times 255.
    """
    quad_center = torch.tensor([2.0, 0.0, 1.6], dtype=torch.float64)
    quad_half = 1.0
    emissive_intensity = 6.0

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        floor = Prism(dimensions=(8.0, 8.0, 0.2))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        # A thin prism, not a Square: the emitter must be triangle geometry
        # with a material block (a 2-D Square is a bezier circuit).
        quad = Prism(dimensions=(2.0 * quad_half, 2.0 * quad_half, 0.02))
        quad.set_material(
            MeshLambertMaterial(
                color=BLACK,
                emissive=WHITE,
                emissive_intensity=emissive_intensity,
            )
        )
        quad.move(RIGHT * float(quad_center[0]) + OUT * float(quad_center[2]))
        quad.spawn(animate=False)

    img = _render_scene_exp(
        tmp_path,
        "emissive_quad.png",
        build,
        96,
        shadows=True,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False, "pt_light_samples": 2},
    )

    # Torch quadrature of L = (albedo / pi) Le \int cos_p cos_q / r^2 dA over
    # the quad, averaged over a small neighbourhood of the floor point the
    # centre pixel sees (the camera looks at the origin; the floor's top face
    # is at z = 0.1).
    n = 128
    cell = 2.0 * quad_half / n
    axis = torch.arange(n, dtype=torch.float64) * cell - quad_half + cell / 2
    qx = quad_center[0] + axis.view(-1, 1)
    qy = quad_center[1] + axis.view(1, -1)
    refs = []
    for px in (-0.2, 0.0, 0.2):
        for py in (-0.2, 0.0, 0.2):
            dx = qx - px
            dy = qy - py
            dz = float(quad_center[2]) - 0.1
            r2 = dx * dx + dy * dy + dz * dz
            cos_pq = (dz * dz) / r2  # cos_p * cos_q, both against +-z
            integral = float((cos_pq / r2).sum()) * cell * cell
            refs.append(emissive_intensity * integral / np.pi)
    reference = 255.0 * float(np.mean(refs))
    measured = _center_patch_mean(img, half=2)
    assert abs(measured - reference) <= max(8.0, 0.12 * reference), (
        f"emissive direct lighting off the reference integral: measured "
        f"{measured:.1f}, reference {reference:.1f} (a ~2x error here means "
        f"MIS double counting; ~0.5x a lost strategy)"
    )


def _sun_env_map():
    """A dim sky with one bright rectangular sun centred on the camera side
    of the scene (``OUT`` is -z: phi = -pi/2 -> u = 0.25, theta = pi/2 ->
    v = 0.5), so it shines onto the camera-facing floor surface.
    """
    env = torch.full((64, 128, 3), 0.05)
    env[26:38, 28:36] = 6.0
    return env


def _env_irradiance_on_floor(env, intensity):
    """Torch quadrature of the irradiance the map sends onto the floor's
    camera-facing (-z) surface, per the kernel's equirect convention
    (y = cos theta up, phi = atan2(z, x)).
    """
    e = env.double()
    h, w = e.shape[0], e.shape[1]
    v = (torch.arange(h, dtype=torch.float64) + 0.5) / h
    u = (torch.arange(w, dtype=torch.float64) + 0.5) / w
    theta = np.pi * v
    phi = 2.0 * np.pi * (u - 0.5)
    sin_t = torch.sin(theta).view(-1, 1)
    dir_z = torch.sin(phi).view(1, -1) * sin_t
    weight = (-dir_z).clamp_min(0.0) * sin_t * (np.pi / h) * (2.0 * np.pi / w)
    return float((e.mean(-1) * weight).sum()) * intensity


def test_env_map_lighting_matches_the_reference_integral(tmp_path):
    """Environment lighting under the path tracer, against the torch
    irradiance integral: with env NEE on, the CDF-sampled sun converges to
    the reference at modest sample counts; with ``pt_env_nee`` off the
    BSDF-escape arm alone must land on the SAME value (the two strategies
    bracket the MIS estimate -- both are unbiased or one of them is wrong).
    A sky pixel checks the camera-escape fold against the map itself.
    """
    env = _sun_env_map()

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        scene.set_environment_map(env, ambient=False)
        floor = Prism(dimensions=(5.0, 5.0, 0.2))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)

    reference = 255.0 * _env_irradiance_on_floor(env, 1.0) / np.pi
    assert 60.0 < reference < 240.0, "test scene poorly scaled"

    nee = _render_scene_exp(
        tmp_path,
        "env_nee.png",
        build,
        32,
        shadows=True,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False},
    )
    measured = _center_patch_mean(nee, half=2)
    assert abs(measured - reference) <= max(8.0, 0.12 * reference), (
        f"env NEE off the irradiance reference: measured {measured:.1f}, "
        f"reference {reference:.1f}"
    )
    # The sky shows the map itself through the camera-escape fold.
    sky = float(nee[2, 2, :3].double().mean())
    assert abs(sky - 255.0 * 0.05) <= 6.0, (
        f"sky pixel {sky:.1f} does not show the environment base radiance"
    )

    bsdf_only = _render_scene_exp(
        tmp_path,
        "env_bsdf.png",
        build,
        128,
        shadows=True,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False, "pt_env_nee": False},
    )
    measured_b = _center_patch_mean(bsdf_only, half=2)
    assert abs(measured_b - reference) <= max(12.0, 0.2 * reference), (
        f"BSDF-only env arm off the irradiance reference: measured "
        f"{measured_b:.1f}, reference {reference:.1f} -- the two strategies "
        f"no longer estimate the same integral"
    )


# ---------------------------------------------------------------------------
# Stage-3 probes (table search + environment CDF, no render pipeline)
# ---------------------------------------------------------------------------


def test_nee_table_search_probe():
    """The CDF binary search returns the bracketing entry and its exact
    selection probability, boundaries included.
    """
    from algan.rendering.raytracing.path_tracer_taichi import pt_nee_pick_probe
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    cdf = torch.tensor([0.1, 0.3, 1.0], dtype=torch.float32, device=DEVICE)
    u = torch.tensor(
        [0.0, 0.05, 0.1, 0.15, 0.3, 0.9999], dtype=torch.float32, device=DEVICE
    )
    out = torch.zeros((u.shape[0], 2), dtype=torch.float32, device=DEVICE)
    pt_nee_pick_probe(cdf, 3, u, out)
    out = out.cpu()
    assert out[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    expected_p = [0.1, 0.1, 0.2, 0.2, 0.7, 0.7]
    assert torch.allclose(out[:, 1], torch.tensor(expected_p), atol=1e-6)


def test_env_cdf_sampling_is_consistent_and_normalized():
    """Three properties the escape-MIS weight rests on: the pdf returned by
    the sampler equals the pdf re-evaluated from the sampled direction, the
    pdf integrates to one over the sphere, and the sampler concentrates its
    draws on the bright region.
    """
    from algan.rendering.raytracing.path_tracer import _build_env_cdf
    from algan.rendering.raytracing.path_tracer_taichi import (
        pt_env_pdf_probe,
        pt_env_sample_probe,
    )
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    env = _sun_env_map()
    env_cdf, power = _build_env_cdf(env)
    assert power > 0
    ch, cw = int(env_cdf.shape[0]), int(env_cdf.shape[1]) - 1
    # CDF sanity: monotone, ending exactly at 1.
    assert float(env_cdf[:, cw].diff().min()) >= 0.0
    assert float(env_cdf[:, cw][-1]) == 1.0
    assert float(env_cdf[:, :cw].diff(dim=1).min()) >= 0.0

    env_cdf_dev = env_cdf.to(DEVICE)
    gen = torch.Generator().manual_seed(7)
    u = torch.rand((4096, 2), generator=gen).float().to(DEVICE)
    out = torch.zeros((4096, 5), dtype=torch.float32, device=DEVICE)
    pt_env_sample_probe(env_cdf_dev, ch, cw, u, out)
    out = out.cpu()
    dirs = out[:, :3]
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(4096), atol=1e-4)
    # Sampled pdf == evaluated pdf for the same direction.
    rel = (out[:, 3] - out[:, 4]).abs() / out[:, 3].clamp_min(1e-9)
    assert float(rel.max()) < 1e-3
    # Importance: the sun (on the -z camera side; see _sun_env_map) subtends
    # far less than half the sphere, but with luminance 120x the sky it must
    # draw the majority of the samples.
    in_sun = dirs[:, 2] < -0.85
    assert float(in_sun.float().mean()) > 0.5

    # The pdf integrates to one over the sphere (quadrature on a (u, v)
    # grid mapped through the equirect parameterisation).
    gh, gw = 128, 256
    v = (torch.arange(gh, dtype=torch.float64) + 0.5) / gh
    ug = (torch.arange(gw, dtype=torch.float64) + 0.5) / gw
    theta = np.pi * v
    phi = 2.0 * np.pi * (ug - 0.5)
    sin_t = torch.sin(theta).view(-1, 1).expand(gh, gw)
    dx = torch.cos(phi).view(1, -1) * sin_t
    dy = torch.cos(theta).view(-1, 1).expand(gh, gw)
    dz = torch.sin(phi).view(1, -1) * sin_t
    grid = torch.stack((dx, dy, dz), -1).reshape(-1, 3).float().to(DEVICE)
    pdf = torch.zeros((grid.shape[0],), dtype=torch.float32, device=DEVICE)
    pt_env_pdf_probe(env_cdf_dev, ch, cw, grid, pdf)
    dw = sin_t.reshape(-1).double() * (np.pi / gh) * (2.0 * np.pi / gw)
    total = float((pdf.cpu().double() * dw).sum())
    assert abs(total - 1.0) < 0.02, f"env pdf integrates to {total:.4f}"


# ---------------------------------------------------------------------------
# Closed shells + author order (Stage 4: 2D/parity polish)
# ---------------------------------------------------------------------------


def _render_scene_result(
    tmp_path, name, build, samples_per_pixel, video=None, experimental=None, **rt_kwargs
):
    """``_render_scene_exp`` that also returns the ``RenderResult`` (for the
    truncation counters riding ``render_plan``).
    """
    settings = video if video is not None else STACK_SETTINGS
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        for key, value in rt_kwargs.items():
            SETTINGS.raytracing.set(**{key: value})
        for key, value in (experimental or {}).items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        with Scene(video_settings=settings) as scene:
            with Off():
                build(scene)
            result = scene.save_frame(
                tmp_path / name, video_settings=settings, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    return _read(result), result


# Raw byte output (no colour management), so composites read off directly:
# the public fields, passed as rt_kwargs, plus the experimental HDR switch.
_RAW_KW = {"linear_color_space": False, "tonemapping": False}
_RAW_EXP = {"post_process_tonemap": False}


def _emissive_shell_cube(dimensions=(2.0, 2.0, 2.0), opacity=0.6):
    """A translucent closed-shell cube whose only radiance is its emission.

    Black albedo kills the diffuse lobe (and every NEE response), emission is
    exact, and the continuation is a pure pass-through -- so the composite is
    deterministic per sample and the authored-opacity oracle is sharp: the
    interior reads ``opacity * emissive`` if the shell attenuates once and
    ``opacity * emissive * (2 - opacity)`` if both crossings composite.
    Rotated off-axis because exactly axis-aligned coincident edges lose
    occasional seam hits per sample (pre-existing, ring-independent), which
    would blur the doubled arm.
    """
    cube = Prism(dimensions=dimensions)
    cube.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.0)
    )
    cube.set_opacity(opacity)
    cube.rotate(17, UP).rotate(9, RIGHT)
    return cube


def test_closed_shell_attenuates_once_at_authored_opacity(tmp_path):
    """The opacity oracle. A declared closed shell at ``opacity=0.6`` must
    render its interior at exactly ``0.6 * emissive`` under the path tracer
    -- one attenuation per entry/exit pair, the per-ray form of the sheet
    route's ``solid_shell_alpha`` coverage ceiling -- and must agree with the
    deterministic route's ceilinged composite pixel-for-pixel.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        _emissive_shell_cube().spawn(animate=False)

    pt, result = _render_scene_result(
        tmp_path, "shell_pt.png", build, 8, experimental=_RAW_EXP, **_RAW_KW
    )
    det, _ = _render_scene_result(
        tmp_path, "shell_det.png", build, 1, experimental=_RAW_EXP, **_RAW_KW
    )
    expected = 0.6 * 255.0
    h, w = pt.shape[0], pt.shape[1]
    core = pt[h // 2 - 6 : h // 2 + 6, w // 2 - 6 : w // 2 + 6, :3].float()
    assert abs(float(core.mean()) - expected) <= 2.0, (
        f"closed shell composited at {float(core.mean()):.1f}, expected "
        f"{expected:.1f} (authored opacity once); doubled would read "
        f"{0.6 * 255 * 1.4:.1f}"
    )
    assert float((core - expected).abs().max()) <= 2.0, (
        "the interior is not uniform at the authored opacity"
    )
    err = (pt[..., :3] - det[..., :3]).abs().amax(-1).float()
    interior = err[h // 2 - 6 : h // 2 + 6, w // 2 - 6 : w // 2 + 6]
    assert float(interior.max()) <= 2.0, (
        f"path-traced closed shell deviates from the deterministic ceiling "
        f"by {float(interior.max()):.0f} on the interior"
    )
    assert result.render_plan.truncations.closed_shell_ring == 0


def test_closed_shell_ceiling_off_restores_per_crossing_attenuation(tmp_path):
    """``solid_shell_alpha=False`` must disable the ring entirely: both shell
    crossings composite, the pre-ceiling behaviour (an interior near
    ``a * (2 - a) * emissive``). This is the byte-parity escape hatch, and it
    proves the oracle above is measuring the ring rather than an accident of
    the scene.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        _emissive_shell_cube().spawn(animate=False)

    pt, _ = _render_scene_result(
        tmp_path,
        "shell_off.png",
        build,
        8,
        experimental=dict(_RAW_EXP, solid_shell_alpha=False),
        **_RAW_KW,
    )
    h, w = pt.shape[0], pt.shape[1]
    core = pt[h // 2 - 6 : h // 2 + 6, w // 2 - 6 : w // 2 + 6, :3].float()
    once = 0.6 * 255.0
    doubled = 0.6 * 255.0 * 1.4
    mean = float(core.mean())
    assert mean > (once + doubled) / 2.0, (
        f"with solid_shell_alpha off the interior reads {mean:.1f}; expected "
        f"near the doubled composite {doubled:.1f}, not the ceiling {once:.1f}"
    )


def test_shell_ring_overflow_is_counted_not_silent(tmp_path):
    """Five nested closed shells overflow the four-slot ring; the surplus
    crossings must be tallied on the render plan (an instrument that reports
    zero may not be looking), never dropped silently.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        for k in range(5):
            _emissive_shell_cube(dimensions=(0.6 + 0.4 * k,) * 3, opacity=0.3).spawn(
                animate=False
            )

    _pt, result = _render_scene_result(
        tmp_path,
        "shell_overflow.png",
        build,
        4,
        experimental=_RAW_EXP,
        **_RAW_KW,
    )
    assert result.render_plan.truncations.closed_shell_ring > 0, (
        "five nested closed shells did not report the ring ceiling"
    )


def test_author_order_and_depth_compose_like_the_deterministic_route(tmp_path):
    """Author-order edge cases: two translucent squares at the SAME depth
    composite in spawn order (the layer / ``_comes_after`` tie-break both
    routes share), and a third square spawned LAST but placed behind them
    composites underneath -- depth beats author order. Flat interiors of the
    path-traced frame must match the deterministic route exactly (up to
    rounding) for BOTH spawn orders, and the two orders must produce
    different composites in the overlap -- proof the scene actually
    exercises the tie-break rather than being order-blind.
    """

    def _order_scene(first, second):
        def build(scene):
            scene.set_background(BLACK)
            a = Square(side_length=3.0, color=first).set_opacity(0.5)
            a.spawn(animate=False)
            b = Square(side_length=3.0, color=second).set_opacity(0.5)
            b.move(RIGHT * 1.0 + UP * 1.0)
            b.spawn(animate=False)
            back = Square(side_length=5.0, color=BLUE).set_opacity(0.5)
            back.move(-OUT * 2.0)  # OUT is toward the camera; behind is -OUT
            back.spawn(animate=False)

        return build

    frames = {}
    for tag, build in (
        ("rg", _order_scene(RED, GREEN)),
        ("gr", _order_scene(GREEN, RED)),
    ):
        det, _ = _render_scene_result(tmp_path, f"order_det_{tag}.png", build, 1)
        pt, _ = _render_scene_result(tmp_path, f"order_pt_{tag}.png", build, 8)
        det_f = det.float().permute(2, 0, 1).unsqueeze(0)
        pooled_max = torch.nn.functional.max_pool2d(det_f, 3, stride=1, padding=1)
        pooled_min = -torch.nn.functional.max_pool2d(-det_f, 3, stride=1, padding=1)
        flat = (pooled_max - pooled_min).squeeze(0).amax(0) < 2
        assert flat.sum() > det.shape[0] * det.shape[1] // 4
        err = (det - pt).abs().amax(-1)
        assert int(err[flat].max()) <= 1, (
            f"author-order composite ({tag}) deviates from the deterministic "
            f"route by {int(err[flat].max())} on flat interiors"
        )
        frames[tag] = pt

    # The overlap of the two same-depth squares (world x, y in [-0.5, 1.5]^2,
    # ~8 px per world unit at this framing) must depend on spawn order: with
    # the second square offset toward the corner, only author order separates
    # the two composites there.
    h, w = frames["rg"].shape[0], frames["rg"].shape[1]
    patch = (slice(h // 2 - 8, h // 2 - 4), slice(w // 2 + 4, w // 2 + 8))
    delta = (
        (frames["rg"][patch][..., :3].float() - frames["gr"][patch][..., :3].float())
        .abs()
        .amax(-1)
    )
    assert float(delta.mean()) > 5.0, (
        f"swapping the spawn order did not change the overlap composite "
        f"(mean channel delta {float(delta.mean()):.1f}) -- the scene cannot "
        f"see the author-order tie-break"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
