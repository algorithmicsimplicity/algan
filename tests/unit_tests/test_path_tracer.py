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
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    AmbientLight,
    Circle,
    HemisphereLight,
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


def test_dimension_pairs_never_collide():
    """The dimension table is a partition, not a convention.

    Two consumers sharing a pair would silently correlate two decisions --
    the failure mode has no symptom other than variance -- so the arithmetic
    in ``pt_shade`` is checked here directly, in Python, over the shapes a
    render can take. The pair the stratified lobe select gained (roadmap
    section 7) widened one crossing's block from ``2L`` to ``2L + 1``, which
    is exactly the kind of change this test exists to catch.
    """
    from algan.rendering.raytracing.path_tracer_taichi import (
        PAIR_BOUNCE_BASE,
        PAIR_LENS,
        PAIR_PIXEL,
        PAIRS_PER_BOUNCE,
    )

    for max_bounces in (0, 1, 2, 4, 8):
        for light_samples in (1, 2, 4):
            owner = {PAIR_PIXEL: "pixel jitter", PAIR_LENS: "lens"}

            shape = f"B={max_bounces}, L={light_samples}"

            def claim(pair, who, owner=owner, shape=shape):
                assert pair >= 0, f"negative pair {pair} for {who!r} ({shape})"
                assert pair not in owner, (
                    f"pair {pair} claimed by both {owner[pair]!r} and {who!r} ({shape})"
                )
                owner[pair] = who

            # A scatter needs a bounce left, so the ordinal tops out at
            # ``max_bounces - 1`` and the crossing block starts right after.
            for b in range(max_bounces):
                for slot in range(PAIRS_PER_BOUNCE):
                    claim(
                        PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * b + slot,
                        f"bounce {b} slot {slot}",
                    )
            # Crossings are indexed by ``processed``, which ``pt_shade``
            # increments BEFORE the block, so c starts at 1; c = 0 is checked
            # too because nothing but the arithmetic keeps it clear.
            cross0 = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * max_bounces
            for c in range(0, 12):
                base = cross0 + (2 * light_samples + 1) * c
                for s in range(light_samples):
                    claim(base + 2 * s, f"crossing {c} NEE {s} select")
                    claim(base + 2 * s + 1, f"crossing {c} NEE {s} point")
                claim(base + 2 * light_samples, f"crossing {c} lobe select")


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
                Square(size=6.0, color=BLUE).spawn(animate=False)
                red = Square(size=4.0, color=RED).set_opacity(0.5)
                red.spawn(animate=False)
                green = Square(size=2.0, color=GREEN).set_opacity(0.25)
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
        Scene.clear_lights()
        Square(size=3.0, color=RED).spawn(animate=False)

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
    near = _render_scene(tmp_path, "clip_near.png", with_camera(near=100.0), 8).float()
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
        Scene.clear_lights()
        sphere = Sphere(radius=1.0)
        sphere.set_material(MeshLambertMaterial(color=WHITE))
        sphere.spawn(animate=False)

    img = _render_scene(tmp_path, "furnace_lambert.png", build, 16)
    lo = int(img.amin())
    assert lo >= 254, (
        f"diffuse furnace lost energy: darkest channel {lo} (expected white everywhere)"
    )


def test_lambert_furnace_is_lossless_under_a_light_row(tmp_path):
    """The furnace check, with the uniform environment delivered as a light
    ROW instead of as the background.

    An ``AmbientLight`` is a direction-less packed row: constant radiance
    ``L`` from every direction. Integrated against a white Lambert lobe that
    is exactly ``e_diff * L`` -- which is what the fill computes since the
    one-BSDF change (roadmap section 5), in place of the deterministic
    stage's ``albedo * L * (n . n)``. So a white sphere lit only by an
    ambient row of radiance 1 must read white: no energy lost, and none
    gained. Tonemapping and the sRGB transfer are off so a byte IS the
    radiance times 255.

    This is the light-row arm of the furnace test above; the NEE-sampled rows
    are pinned against an independently-built emitter of matched radiance by
    ``test_area_light_row_and_emissive_quad_agree``.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        AmbientLight(color=WHITE, intensity=1.0).spawn(animate=False)
        sphere = Sphere(radius=1.0)
        sphere.set_material(MeshLambertMaterial(color=WHITE))
        sphere.spawn(animate=False)

    img = _render_scene_exp(
        tmp_path,
        "furnace_light_row.png",
        build,
        16,
        shadows=False,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False},
    ).float()
    h, w = img.shape[0], img.shape[1]
    # 8x8: the sphere's disc is ~20 px across at this framing, so a wider
    # patch would average in the black background.
    disc = img[h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4, :3]
    mean = float(disc.mean())
    assert mean > 0.98 * 255, (
        f"ambient-row furnace lost energy: sphere disc mean {mean:.1f}/255"
    )
    assert mean <= 256, f"ambient-row furnace GAINED energy: {mean:.1f}/255"


def test_ggx_furnace_keeps_energy_with_compensation(tmp_path):
    """White furnace, specular: a white metallic sphere (roughness 0.5) under
    a white background must stay near-white -- VNDF sampling with the Turquin
    compensation recovers the multiple-scattering energy single-scatter GGX
    loses (uncompensated, rough metal renders visibly dark).
    """

    def build(scene):
        scene.set_background(WHITE)
        Scene.clear_lights()
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


def test_nee_light_row_direct_lighting_is_the_physical_bsdf(tmp_path):
    """A Lambert plane under one point light, against the closed-form
    physical answer rather than against the deterministic renderer.

    Since the one-BSDF change (``DESIGN_path_tracer_roadmap.md`` section 5) a
    packed light row goes through ``_pt_lit_f_pdf`` like every other emitter
    kind, so a Lambert surface returns ``albedo / pi * L * cos`` -- ``pi``
    times dimmer than the deterministic stage, which has no ``1/pi``. That is
    deliberate: parity with a renderer the user fell back FROM is not a goal,
    and one response is what makes MIS weights sum to one. This test pins the
    physical value, which is the thing that must not drift.

    The light sits on the surface normal above the measured point, so
    ``cos = 1`` and the answer is simply ``albedo * intensity / pi``.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        PointLight(location=OUT * 5.0, color=WHITE, intensity=1.0).spawn(animate=False)
        plane = Prism(width=7.0, height=7.0, depth=0.1)
        plane.set_material(MeshLambertMaterial(color=WHITE))
        plane.spawn(animate=False)

    img = _render_scene_exp(
        tmp_path,
        "nee_row_bsdf.png",
        build,
        32,
        shadows=False,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False},
    )
    reference = 255.0 / np.pi
    measured = _center_patch_mean(img, half=2)
    assert abs(measured - reference) <= max(4.0, 0.06 * reference), (
        f"light-row direct lighting off the physical BSDF: measured "
        f"{measured:.1f}, reference {reference:.1f} (a ~pi x reading means "
        f"the deterministic stage formula is back)"
    )


def test_point_light_shadow_under_path_tracing(tmp_path):
    """With ``shadows`` on, NEE visibility rays darken occluded geometry.

    Asserted as the physical invariant rather than at a guessed pixel: a
    shadow can only ever *remove* light, so turning ``shadows`` on must
    darken a meaningful part of the frame and brighten none of it.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        PointLight(location=OUT * 6.0, color=WHITE, intensity=1.0).spawn(animate=False)
        floor = Prism(width=7.0, height=7.0, depth=0.1)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        blocker = Square(size=2.0, color=BLUE)
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
        Scene.clear_lights()
        PointLight(location=(UP * 2.0 + OUT * 5.0), color=WHITE, intensity=1.2).spawn(
            animate=False
        )
        floor = Prism(width=8.0, height=8.0, depth=0.1)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        wall = Prism(width=0.1, height=8.0, depth=3.0)
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


def _rect_light_lambert_reference(half_u, half_v, height, radiance, decay=0.0, n=192):
    """Torch quadrature of the radiance a white Lambert surface returns under
    a rectangular emitter directly above it.

    The rectangle is centred on the surface normal at ``height``, faces
    straight down, and carries ``radiance`` per unit area's worth of the
    light's power (``color * intensity / area``, which is what
    ``_materialize_render_state``'s ``1/K`` split plus ``_light_eval``'s
    per-row radiometry adds up to). ``decay`` is the light's falloff exponent
    -- ``2`` is the physical inverse-square, ``0`` the default "no falloff",
    which the emitter model applies verbatim and this reference therefore
    reproduces verbatim.

    Returns ``(albedo / pi) * radiance * integral(cos_p cos_l / d^decay dA)``.
    """
    cell_u = 2.0 * half_u / n
    cell_v = 2.0 * half_v / n
    u = torch.arange(n, dtype=torch.float64) * cell_u - half_u + cell_u / 2
    v = torch.arange(n, dtype=torch.float64) * cell_v - half_v + cell_v / 2
    du = u.view(-1, 1)
    dv = v.view(1, -1)
    d2 = du * du + dv * dv + height * height
    d = d2.sqrt()
    cos_pq = (height * height) / d2  # cos_p * cos_l, both against +-z
    fall = torch.ones_like(d) if decay == 0.0 else d.pow(-decay)
    integral = float((cos_pq * fall).sum()) * cell_u * cell_v
    return radiance * integral / np.pi


def test_area_light_matches_the_reference_integral(tmp_path):
    """A RectAreaLight under the path tracer samples a uniform point inside
    the selected row's cell and evaluates the falloff and one-sided cosine AT
    that point; every cell carries equal power, so selecting a cell and then
    a point inside it IS a uniform point on the whole rectangle. Its
    expectation is therefore the continuous area integral, whatever ``k`` is.

    Checked against that integral in torch rather than against the
    deterministic renderer: since the one-BSDF change (roadmap section 5) the
    two no longer agree by design, and a parity assertion here would only
    re-assert what section 5 deleted. Runs with pt_light_samples = 2, so the
    1/N weighting of multi-sample NEE is under test too.
    """
    width = height = 3.0
    intensity = 1.0
    plane_z = 3.0

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        RectAreaLight(
            location=OUT * plane_z,
            width=width,
            height=height,
            samples=64,
            color=WHITE,
            intensity=intensity,
        ).spawn(animate=False)
        floor = Prism(width=7.0, height=7.0, depth=0.1)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)

    img = _render_scene_exp(
        tmp_path,
        "area_pt.png",
        build,
        96,
        shadows=False,
        linear_color_space=False,
        tonemapping=False,
        experimental={"post_process_tonemap": False, "pt_light_samples": 2},
    )
    # The floor's top face is at z = 0.05 (a 0.1-deep prism at the origin).
    reference = 255.0 * _rect_light_lambert_reference(
        0.5 * width,
        0.5 * height,
        plane_z - 0.05,
        intensity / (width * height),
    )
    measured = _center_patch_mean(img, half=2)
    assert reference > 20.0, "test scene poorly scaled"
    assert abs(measured - reference) <= max(4.0, 0.06 * reference), (
        f"area-light radiometry off the reference integral: measured "
        f"{measured:.1f}, reference {reference:.1f}"
    )


def test_area_light_row_and_emissive_quad_agree(tmp_path):
    """**The section-5 acceptance test.** A ``RectAreaLight`` and an emissive
    quad of matched radiance, in the same place, must light the same surface
    identically -- for a Lambert surface and for a GGX one.

    They did not before: light rows went through ``_pt_direct_response`` (the
    deterministic stage formula, term for term) while emissive triangles went
    through ``_pt_lit_f_pdf`` (the physical BSDF the continuation samples),
    and the two diverge most on smooth metals, where their ``G`` terms
    disagree. Now both ends call ``_pt_lit_f_pdf``.

    **Matching the radiance.** A ``RectAreaLight`` of colour ``C`` and
    intensity ``I`` expands into ``K`` cell rows carrying ``C * I / K`` each
    (``_materialize_render_state``), and ``_light_eval`` /
    ``_pt_nee_light_row`` apply the falloff ``d^-decay``, the range fade and
    the one-sided cosine at the sampled point. With ``decay = 2`` and no
    range limit, the ``K``-row sum is a Riemann sum of
    ``integral Le cos_l / d^2 dA`` over the rectangle with
    ``Le = C * I / K / (A / K) = C * I / A``. So an emissive quad of the same
    size and place matches when ``emissive * emissive_intensity`` equals
    ``C * I / area``. (``decay = 2`` is not incidental: the light row's
    default ``decay = 0`` has no inverse-square term at all, so no emissive
    quad can match it -- that is the emitter model, not the surface
    response, and section 5 leaves it exactly as it was.)
    """
    size = 1.2
    area = size * size
    radiance = 4.0
    intensity = radiance * area
    emitter_at = RIGHT * 2.0 + OUT * 1.6

    def floor_of(material):
        def build(scene):
            scene.set_background(BLACK)
            Scene.clear_lights()
            floor = Prism(width=8.0, height=8.0, depth=0.2)
            floor.set_material(material())
            floor.spawn(animate=False)

        return build

    def with_row(material):
        base = floor_of(material)

        def build(scene):
            base(scene)
            RectAreaLight(
                location=emitter_at,
                width=size,
                height=size,
                samples=64,
                color=WHITE,
                intensity=intensity,
                decay=2.0,
                # Straight down, so the rectangle's one-sided cosine is
                # measured against the same normal the quad's -z face has.
                # A ``target`` of ORIGIN would tilt it and change the
                # emitter, not the response.
                target=RIGHT * 2.0,
            ).spawn(animate=False)

        return build

    def with_quad(material):
        base = floor_of(material)

        def build(scene):
            base(scene)
            # A thin prism, not a Square: the emitter must be triangle
            # geometry with a material block (a 2-D Square is a circuit).
            quad = Prism(width=size, height=size, depth=0.02)
            quad.set_material(
                MeshLambertMaterial(
                    color=BLACK,
                    emissive=WHITE,
                    emissive_intensity=radiance,
                )
            )
            quad.move(emitter_at)
            quad.spawn(animate=False)

        return build

    surfaces = {
        "lambert": lambda: MeshLambertMaterial(color=WHITE),
        "ggx": lambda: MeshStandardMaterial(color=WHITE, metalness=0.0, roughness=0.35),
    }
    for name, material in surfaces.items():
        row = _render_scene_exp(
            tmp_path,
            f"matched_row_{name}.png",
            with_row(material),
            96,
            shadows=True,
            linear_color_space=False,
            tonemapping=False,
            experimental={"post_process_tonemap": False, "pt_light_samples": 2},
        )
        quad = _render_scene_exp(
            tmp_path,
            f"matched_quad_{name}.png",
            with_quad(material),
            96,
            shadows=True,
            linear_color_space=False,
            tonemapping=False,
            experimental={"post_process_tonemap": False, "pt_light_samples": 2},
        )
        row_mean = _center_patch_mean(row, half=4)
        quad_mean = _center_patch_mean(quad, half=4)
        assert row_mean > 10.0, (
            f"the {name} patch is barely lit by the light row "
            f"({row_mean:.1f}/255); the comparison has no content"
        )
        rel = abs(row_mean - quad_mean) / max(row_mean, quad_mean)
        assert rel <= 0.06, (
            f"{name}: a RectAreaLight and an emissive quad of matched "
            f"radiance disagree by {100 * rel:.1f}% (row {row_mean:.1f}, "
            f"quad {quad_mean:.1f}) -- they must go through ONE BSDF"
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
        Scene.clear_lights()
        floor = Prism(width=8.0, height=8.0, depth=0.2)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        # A thin prism, not a Square: the emitter must be triangle geometry
        # with a material block (a 2-D Square is a bezier circuit).
        quad = Prism(width=2.0 * quad_half, height=2.0 * quad_half, depth=0.02)
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
    of the scene (``OUT`` is +z: phi = +pi/2 -> u = 0.75, theta = pi/2 ->
    v = 0.5), so it shines onto the camera-facing floor surface.
    """
    env = torch.full((64, 128, 3), 0.05)
    env[26:38, 92:100] = 6.0
    return env


def _env_irradiance_on_floor(env, intensity):
    """Torch quadrature of the irradiance the map sends onto the floor's
    camera-facing (+z) surface, per the kernel's equirect convention
    (y = cos theta up, phi = atan2(z, x)) -- the same one Three.js uses.
    """
    e = env.double()
    h, w = e.shape[0], e.shape[1]
    v = (torch.arange(h, dtype=torch.float64) + 0.5) / h
    u = (torch.arange(w, dtype=torch.float64) + 0.5) / w
    theta = np.pi * v
    phi = 2.0 * np.pi * (u - 0.5)
    sin_t = torch.sin(theta).view(-1, 1)
    dir_z = torch.sin(phi).view(1, -1) * sin_t
    weight = dir_z.clamp_min(0.0) * sin_t * (np.pi / h) * (2.0 * np.pi / w)
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
        Scene.clear_lights()
        scene.set_environment_map(env, ambient=False)
        floor = Prism(width=5.0, height=5.0, depth=0.2)
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


def test_offset_ray_origin_scales_with_the_hit_point():
    """``_pt_offset_ray_origin`` (Wachter & Binder, Ray Tracing Gems 2019)
    replaces the fixed ``10 * min_hit_distance`` world epsilon the path
    tracer used to spawn rays with.

    The property that matters is the one the fixed epsilon did not have: the
    step is tied to the representable spacing AT the hit point, so it grows
    with the scene's coordinates instead of being simultaneously too small
    far from the origin (acne) and too large near it (light leaks). Three
    decades of magnitude, and the point must move at every one of them --
    an offset that returned its input would put the spawn origin back on the
    surface.
    """
    from algan.rendering.raytracing.path_tracer_taichi import pt_offset_probe
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    magnitudes = [1e-3, 1.0, 1e3]
    points = torch.tensor(
        [[m, m, m] for m in magnitudes], dtype=torch.float32, device=DEVICE
    )
    normals = torch.tensor(
        [[0.0, 0.0, 1.0]] * len(magnitudes), dtype=torch.float32, device=DEVICE
    )
    out = torch.zeros((len(magnitudes), 3), dtype=torch.float32, device=DEVICE)
    pt_offset_probe(points, normals, out)
    out = out.double().cpu()

    stored = points.double().cpu()
    steps = []
    for i, m in enumerate(magnitudes):
        # Against the f32 value the kernel actually saw, not the f64 literal.
        m32 = float(stored[i, 2])
        step = float(out[i, 2]) - m32
        assert step > 0.0, (
            f"the offset did not move a point at magnitude {m} along the "
            f"normal (step {step!r})"
        )
        # Only the normal's axis moves; the other two are untouched.
        untouched = [float(out[i, 0]), float(out[i, 1])]
        assert untouched == [float(stored[i, 0]), float(stored[i, 1])], (
            f"the offset moved axes the normal does not point along at "
            f"magnitude {m}: {out[i].tolist()}"
        )
        steps.append(step)

    # It scales: a thousand-fold larger hit point takes a far larger step.
    assert steps[2] > 100.0 * steps[1] > steps[0], (
        f"the offset did not scale with the hit point's magnitude: {steps}"
    )
    # And it stays an epsilon at every scale: bounded by a ten-thousandth of
    # the coordinate above the near-origin threshold, and by the paper's fixed
    # float offset (1/65536) below it, where a relative step would round to
    # nothing. An offset growing faster than that would push spawn origins off
    # the geometry they belong to.
    for m, step in zip(magnitudes, steps):
        bound = max(1e-4 * m, 2.0 / 65536.0)
        assert step <= bound, (
            f"the offset is {step} at magnitude {m} (bound {bound}): too "
            "coarse to be an epsilon"
        )


def test_lobe_select_is_stratified_and_seed_stable(tmp_path):
    """The pass/diffuse/specular/transmit pick draws its own crossing-indexed
    Sobol pair now (roadmap section 7) rather than white noise.

    A stratification assertion on one pixel's lobe draws would only re-test
    ``pt_sample_2d`` (``test_sampler_prefixes_are_stratified`` already owns
    that), and the pair itself is checked by
    ``test_dimension_pairs_never_collide``. What is left to check is that the
    kernel wiring survives: the same lit scene at a small sample count under
    two different ``pt_seed`` values must both render, must differ (the pick
    really is re-rolled by the seed), and must agree on the mean -- two
    realisations of one estimator.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        PointLight(location=OUT * 5.0 + UP * 1.5, color=WHITE, intensity=2.0).spawn(
            animate=False
        )
        floor = Prism(width=7.0, height=7.0, depth=0.1)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        blocker = Prism(width=1.6, height=1.6, depth=0.1)
        blocker.set_material(MeshStandardMaterial(color=RED, roughness=0.4))
        blocker.move(OUT * 1.5)
        blocker.spawn(animate=False)

    a = _render_scene_exp(
        tmp_path, "lobe_seed0.png", build, 8, shadows=True, experimental={"pt_seed": 0}
    ).float()
    b = _render_scene_exp(
        tmp_path, "lobe_seed7.png", build, 8, shadows=True, experimental={"pt_seed": 7}
    ).float()
    assert float(a.mean()) > 5.0, "the probe scene rendered (nearly) empty"
    assert not torch.equal(a, b), "pt_seed does not reach the lobe select"
    assert abs(float(a.mean()) - float(b.mean())) <= 2.0, (
        f"two seeds of the same scene disagree on the mean "
        f"({float(a.mean()):.2f} vs {float(b.mean()):.2f}); the lobe select's "
        "reweighting is not unbiased"
    )


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
    # Importance: the sun (on the +z camera side; see _sun_env_map) subtends
    # far less than half the sphere, but with luminance 120x the sky it must
    # draw the majority of the samples.
    in_sun = dirs[:, 2] > 0.85
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


def _emissive_shell_cube(width=2.0, height=2.0, depth=2.0, opacity=0.6):
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
    cube = Prism(width=width, height=height, depth=depth)
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
        Scene.clear_lights()
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
        Scene.clear_lights()
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
        Scene.clear_lights()
        for k in range(5):
            _emissive_shell_cube(*((0.6 + 0.4 * k,) * 3), opacity=0.3).spawn(
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
            a = Square(size=3.0, color=first).set_opacity(0.5)
            a.spawn(animate=False)
            b = Square(size=3.0, color=second).set_opacity(0.5)
            b.move(RIGHT * 1.0 + UP * 1.0)
            b.spawn(animate=False)
            back = Square(size=5.0, color=BLUE).set_opacity(0.5)
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


# ---------------------------------------------------------------------------
# Throughput switches that must not move a pixel
# ---------------------------------------------------------------------------


def _lit_shadowed_scene(scene):
    """A lit, shadowed, all-opaque scene: a floor, a blocker above it and one
    point light, plus the two direction-less rows (ambient + hemisphere) that
    the packed-ambient-row path exists for.
    """
    scene.set_background(BLACK)
    Scene.clear_lights()
    PointLight(location=OUT * 6.0 + UP * 2.0, color=WHITE, intensity=1.0).spawn(
        animate=False
    )
    AmbientLight(color=WHITE, intensity=0.15).spawn(animate=False)
    HemisphereLight(color=WHITE, ground_color=BLUE, intensity=0.1).spawn(animate=False)
    floor = Prism(width=7.0, height=7.0, depth=0.1)
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.spawn(animate=False)
    blocker = Prism(width=2.0, height=2.0, depth=0.1)
    blocker.set_material(MeshLambertMaterial(color=BLUE))
    blocker.move(OUT * 2.0)
    blocker.spawn(animate=False)


def test_packed_ambient_rows_render_exactly_as_the_scan(tmp_path):
    """``pt_ambient_rows`` moves the ambient / hemisphere row lookup out of
    the kernel's per-crossing type scan and into a host-packed list on the
    tail of ``nee_ref``. Same rows, same ascending order, same arithmetic --
    so the frame must not move by a bit.

    Both arms run in one process on purpose: this switch is a runtime word in
    ``nee_meta``, not a ``ti.static`` gate, so one compiled kernel serves both
    (which is also what makes the comparison cheap).
    """
    packed, _ = _render_scene_result(
        tmp_path,
        "ambient_packed.png",
        _lit_shadowed_scene,
        8,
        experimental={"pt_ambient_rows": True},
        shadows=True,
    )
    scanned, _ = _render_scene_result(
        tmp_path,
        "ambient_scanned.png",
        _lit_shadowed_scene,
        8,
        experimental={"pt_ambient_rows": False},
        shadows=True,
    )
    assert torch.equal(packed, scanned), (
        "packing the ambient rows changed the image by up to "
        f"{int((packed - scanned).abs().max())} of 255"
    )


def _render_raw_frames(frames, **experimental):
    """The first ``frames`` raw rendered frames of the static lit scene.

    Through ``Scene.get_frames`` rather than ``save_frame``: the frame index
    the sampler keys on is the one INSIDE the render job (``render_loop``
    hands the kernels ``current_ind - start_ind``), so a still rendered at a
    later timestamp is still frame 0 to the sampler and could not tell the
    two seed policies apart. Raw, so nothing is compared through a lossy
    video codec.
    """
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=8, denoise=False, shadows=True)
        for key, value in experimental.items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        with Scene(video_settings=STACK_SETTINGS) as scene:
            with Off():
                _lit_shadowed_scene(scene)
            out = torch.cat([f.cpu() for f in scene.get_frames(0, frames)])
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    assert out.shape[0] == frames
    return out.to(torch.int32)


def test_a_static_scene_draws_the_same_noise_every_frame():
    """``pt_animated_seed`` (off by default) folds the frame out of the
    sampler key, so a region that does not move draws the IDENTICAL samples
    at every frame: its Monte Carlo error is a fixed grain rather than
    per-frame shimmer. Nothing in this scene moves, so two frames of it must
    come out bit-identical -- and must stop being so with the switch on,
    which is what says the frame reaches the key at all.
    """
    fixed = _render_raw_frames(2, pt_animated_seed=False)
    assert torch.equal(fixed[0], fixed[1]), (
        "a static scene rendered two frames differently under the fixed "
        "sampler seed: the frame is still reaching the key (max channel "
        f"delta {int((fixed[0] - fixed[1]).abs().max())})"
    )
    animated = _render_raw_frames(2, pt_animated_seed=True)
    assert torch.equal(fixed[0], animated[0]), (
        "frame 0 must render identically under either seed policy (max "
        f"channel delta {int((fixed[0] - animated[0]).abs().max())})"
    )
    assert not torch.equal(fixed[1], animated[1]), (
        "pt_animated_seed=True did not change frame 1, so the switch does "
        "not reach the sampler"
    )


# ---------------------------------------------------------------------------
# Completeness: the fallback must not refuse anything
# (DESIGN_path_tracer_roadmap.md section 9)
# ---------------------------------------------------------------------------


def test_the_fallback_refuses_nothing():
    """``_build_render_plan`` returns an EMPTY ``unsupported_features`` for
    ``samples_per_pixel > 1``, whatever the scene carries.

    The path tracer is the fallback for scenes the deterministic renderer
    cannot do, so a refusal leaves the user with no renderer at all and an
    error message naming the setting that did not work. This is the
    machine-checkable form of that rule, and it is built by enumerating the
    features ``_build_render_plan`` actually inspects -- an environment map,
    ``has_refractive``, a user fragment pipeline, a custom scatter override
    on it, and an extended light -- rather than by listing scenes, so a
    rejection added to that function fails this test the moment it is
    written. Every feature is set at once *and* one at a time: a refusal
    conditioned on a combination has to fail here too.
    """
    from types import SimpleNamespace

    from algan.rendering.raytracing.tracer import _build_render_plan

    # Every input _build_render_plan reads, in the form it reads it.
    features = {
        "environment_map": lambda kw: kw.update(
            scene_environment_map=torch.zeros((1, 1, 3))
        ),
        "refractive": lambda kw: kw["merged"].update(has_refractive=True),
        "user_pipeline": lambda kw: kw["merged"].update(has_user_pipeline=True),
        "custom_scatter": lambda kw: kw["merged"].update(
            has_user_pipeline=True, has_custom_scatter=True
        ),
        "extended_light": lambda kw: kw["light_sources"].append(
            SimpleNamespace(_render_aux=object())
        ),
    }

    def plan_for(names):
        kwargs = {
            "scene_environment_map": None,
            "merged": {},
            "light_sources": [],
        }
        for name in names:
            features[name](kwargs)
        return _build_render_plan(
            4,
            kwargs["scene_environment_map"],
            kwargs["merged"],
            kwargs["light_sources"],
        )

    everything = plan_for(features)
    assert everything.backend == "path_tracer"
    assert everything.unsupported_features == (), (
        "the path tracer refused "
        f"{list(everything.unsupported_features)}; it is the fallback, so a "
        "refusal leaves that scene with no renderer at all"
    )
    assert everything.is_supported
    # Each feature alone, so a future refusal cannot hide behind a
    # combination the merged dict above happens to satisfy.
    for name in features:
        assert plan_for([name]).unsupported_features == (), (
            f"the path tracer refused a scene carrying only {name}"
        )
    # ... and the features it does honour are still *reported* as requested.
    assert set(everything.requested_features) == {
        "environment maps",
        "refractive materials",
        "custom fragment-shader pipelines",
        "extended lights",
    }


def _mirror_scene(mirror_stage):
    """A black 45-degree panel between two red emissive walls, out of frame.

    The panel is the only thing the camera sees. Its own albedo is black, so
    the ONLY way red can reach its pixels is a ray that leaves it sideways
    and finds a wall -- which is exactly what a custom scatter is for, and
    which nothing else in this scene can produce.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        mirror = Prism(width=3.0, height=3.0, depth=0.2)
        mirror.color = BLACK
        mirror.set_fragment_shader(mirror_stage)
        # Tilted 45 degrees, so a camera ray leaves it sideways -- toward one
        # of the walls -- rather than straight back at the camera.
        mirror.rotate(45, UP)
        mirror.spawn(animate=False)
        # Both sides, so the test does not depend on which way the rotation
        # tips the panel. Far enough out to sit outside the camera frustum
        # (the visible half-width at the origin plane is 4), so the only red
        # that can reach the frame is red that bounced.
        for side in (RIGHT, -RIGHT):
            wall = Prism(width=0.4, height=8.0, depth=6.0)
            wall.set_material(
                MeshLambertMaterial(color=BLACK, emissive=RED, emissive_intensity=1.0)
            )
            wall.move(side * 5.0)
            wall.spawn(animate=False)

    return build


def _mirror_patch(img, half=4):
    """The mirror panel's own pixels: the image centre (the panel is the only
    geometry inside the camera frustum).
    """
    h, w = img.shape[0], img.shape[1]
    return img[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half]


def test_custom_scatter_renders_as_a_delta_continuation(tmp_path):
    """A custom scatter is honoured by the path tracer, as a delta lobe.

    It used to be the fallback's one hard refusal (``_build_render_plan``
    put "custom scatter overrides" in ``unsupported_features``), which is a
    bug against the renderer's role: a scene that needs the path tracer for
    memory, light count or GI *and* authors a scatter had nowhere to go. It
    now continues along the branch the user's function returns, weight 1 and
    ``prev_pdf = 0`` -- the same contract refraction and the tinted pane get.

    Asserted against the same panel wearing a plain unlit stage, which is
    the same geometry, the same shading and the same black albedo, differing
    only in whether rays bounce: red in the mirror's pixels can therefore
    only have come from the scatter.
    """
    from algan.rendering.shaders.fragment_shaders import (
        _BUILTIN_MAT_SPECS,
        STAGE_UNLIT,
        FragmentStage,
        forced_mirror_scatter,
    )

    # Same stage, same params: only the scatter differs.
    plain = FragmentStage(STAGE_UNLIT.ti_func, _BUILTIN_MAT_SPECS)
    mirrored = _render_scene(
        tmp_path, "scatter_mirror.png", _mirror_scene(forced_mirror_scatter), 16
    ).float()
    flat = _render_scene(
        tmp_path, "scatter_plain.png", _mirror_scene(plain), 16
    ).float()

    # OpenCV loads BGR: channel 2 is red.
    red_mirror = float(_mirror_patch(mirrored)[..., 2].mean())
    red_flat = float(_mirror_patch(flat)[..., 2].mean())
    assert red_flat < 3.0, (
        f"the control panel is not black ({red_flat:.1f}/255 red); the "
        "comparison below would not isolate the scatter"
    )
    assert red_mirror > red_flat + 50.0, (
        "the custom scatter reflected nothing: the mirror panel reads "
        f"{red_mirror:.1f}/255 red against the plain stage's {red_flat:.1f}"
    )
    # It reflected the RED walls rather than lifting every channel.
    patch = _mirror_patch(mirrored)
    assert float(patch[..., 2].mean() - patch[..., 0].mean()) > 30.0, (
        "the reflection arrived achromatic; it did not come off the red walls"
    )


def test_custom_scatter_plan_does_not_refuse_an_authored_scene(tmp_path):
    """The wiring, end to end: a mob authored with a custom scatter renders
    under the path tracer and its plan refuses nothing. The unit test above
    proves the message; this proves a real pipeline registers as one and
    still reaches the kernel.
    """
    from algan.rendering.shaders.fragment_shaders import forced_mirror_scatter

    _img, result = _render_scene_result(
        tmp_path,
        "scatter_plan.png",
        _mirror_scene(forced_mirror_scatter),
        4,
    )
    assert result.render_plan.backend == "path_tracer"
    assert result.render_plan.unsupported_features == ()


# ---------------------------------------------------------------------------
# Adaptive sampling (roadmap section 2)
# ---------------------------------------------------------------------------
#: The ceiling every adaptive test renders against. Big enough that stopping
#: at the floor of 4 is unmistakable in the mean, small enough to stay cheap.
_ADAPTIVE_SPP = 32


def _flat_2d_scene(scene):
    """Unlit 2-D only: no lights, no 3-D geometry, nothing stochastic.

    Every interior pixel here is zero-variance by construction -- the camera
    segment composites deterministically (roadmap contract 4) -- so its two
    sample halves agree exactly and it must stop at ``pt_min_samples``. The
    shapes' edges are the exception and are meant to be: anti-aliasing is
    jittered sampling, so an edge pixel legitimately has variance and may run
    on, which is why the assertions below are about the mean being far from
    the ceiling rather than exactly at the floor.
    """
    scene.set_background(BLACK)
    Scene.clear_lights()
    # The first square covers the frame: a background pixel would stop at the
    # floor too, and counting it would make the assertion pass for the wrong
    # reason.
    Square(size=12.0, color=BLUE).spawn(animate=False)
    Square(size=3.0, color=RED).set_opacity(0.5).spawn(animate=False)
    Circle(radius=1.2, color=GREEN).set_opacity(0.6).spawn(animate=False)


def _lit_shadowed_scene(scene):
    """A lit, shadowed 3-D scene: real Monte Carlo variance to converge."""
    scene.set_background(BLACK)
    Scene.clear_lights()
    PointLight(location=OUT * 5.0 + UP * 2.0, color=WHITE, intensity=2.0).spawn(
        animate=False
    )
    floor = Prism(width=7.0, height=0.2, depth=5.0)
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.move(DOWN * 1.5)
    floor.spawn(animate=False)
    ball = Sphere(radius=1.0)
    ball.set_material(MeshStandardMaterial(color=WHITE, roughness=0.4))
    ball.spawn(animate=False)


def test_adaptive_sampling_stops_unlit_2d_content_near_the_floor(tmp_path):
    """The case section 2 exists for: unlit 2-D content converges at the
    floor, so ``samples_per_pixel`` is a ceiling it never reaches.
    """
    floor = int(SETTINGS.raytracing.experimental.pt_min_samples)
    _img, result = _render_scene_result(
        tmp_path,
        "adaptive_2d.png",
        _flat_2d_scene,
        _ADAPTIVE_SPP,
        experimental={"pt_error_target": 0.02},
    )
    mean = result.render_plan.path_samples_mean
    assert mean >= floor, f"a pixel got fewer than the floor {floor}: mean {mean}"
    # Far below the ceiling, and within a couple of floors of it: what is
    # above the floor is the jittered edges of three shapes, not the
    # interiors.
    assert mean < 0.4 * _ADAPTIVE_SPP, (
        f"unlit 2-D content averaged {mean:.2f} of {_ADAPTIVE_SPP} samples; "
        "zero-variance pixels should stop at the floor"
    )


def test_adaptive_sampling_keeps_sampling_a_lit_scene(tmp_path):
    """The other half of the contract: real variance is not declared
    converged. A lit, shadowed scene must climb well above the floor.
    """
    floor = int(SETTINGS.raytracing.experimental.pt_min_samples)
    _img, result = _render_scene_result(
        tmp_path,
        "adaptive_lit.png",
        _lit_shadowed_scene,
        _ADAPTIVE_SPP,
        shadows=True,
        experimental={"pt_error_target": 0.02},
    )
    mean = result.render_plan.path_samples_mean
    assert mean > 1.5 * floor, (
        f"a lit shadowed scene averaged only {mean:.2f} samples of "
        f"{_ADAPTIVE_SPP} (floor {floor}); the estimator is calling noise "
        "converged"
    )
    assert mean <= _ADAPTIVE_SPP


def test_adaptive_sampling_never_stops_a_stochastic_pixel(tmp_path):
    """The safety property the whole mechanism rests on.

    A pixel whose light was estimated by sampling is run to the ceiling
    unconditionally -- ``pt_shade`` flags every path that takes a random
    decision and the host refuses to stop a pixel that has one -- so a lit,
    shadowed scene must render BYTE-IDENTICALLY to its uniform arm even
    though it takes far fewer samples overall (the ones it drops are the
    background's, which are deterministic).

    Without that gate this scene renders black dots on lit surfaces: a pixel
    whose first samples all miss the light has two sample halves that agree
    exactly, and no error threshold can tell that apart from convergence.
    """
    uniform, _ = _render_scene_result(
        tmp_path,
        "stoch_uniform.png",
        _lit_shadowed_scene,
        _ADAPTIVE_SPP,
        shadows=True,
        experimental={"pt_error_target": 0.0},
    )
    adaptive, result = _render_scene_result(
        tmp_path,
        "stoch_adaptive.png",
        _lit_shadowed_scene,
        _ADAPTIVE_SPP,
        shadows=True,
        experimental={"pt_error_target": 0.02},
    )
    assert result.render_plan.path_samples_mean < _ADAPTIVE_SPP, (
        "the scene did not exercise adaptive sampling at all"
    )
    assert torch.equal(uniform, adaptive), (
        "adaptive sampling moved a lit scene by up to "
        f"{int((uniform - adaptive).abs().max())} counts; a pixel whose "
        "samples gambled must run to the ceiling"
    )


def test_uniform_sampling_spends_the_whole_budget(tmp_path):
    """``pt_error_target = 0`` is the byte-parity escape hatch, so every
    pixel gets exactly ``samples_per_pixel`` -- the reported mean is the
    ceiling, exactly, with nothing rescaled.
    """
    _img, result = _render_scene_result(
        tmp_path,
        "uniform_2d.png",
        _flat_2d_scene,
        _ADAPTIVE_SPP,
        experimental={"pt_error_target": 0.0},
    )
    assert result.render_plan.path_samples_mean == float(_ADAPTIVE_SPP)


def test_a_floor_at_the_ceiling_renders_exactly_like_uniform_sampling(tmp_path):
    """Adaptive sampling's plumbing must not move a pixel by itself.

    With ``pt_min_samples`` at the ceiling no pixel can stop early, so the
    pixel list stays the tile's, every ``samples / n_p`` rescale is exactly
    1.0, and the frame must match the uniform arm byte for byte -- even
    though the waves are cut differently, which is also what proves the
    sampler's per-pixel prefix survives re-waving.
    """
    uniform, _ = _render_scene_result(
        tmp_path,
        "parity_uniform.png",
        _lit_shadowed_scene,
        8,
        shadows=True,
        experimental={"pt_error_target": 0.0},
    )
    floored, result = _render_scene_result(
        tmp_path,
        "parity_floored.png",
        _lit_shadowed_scene,
        8,
        shadows=True,
        experimental={"pt_error_target": 0.02, "pt_min_samples": 8},
    )
    assert result.render_plan.path_samples_mean == 8.0
    assert torch.equal(uniform, floored), (
        "adaptive sampling with nothing to stop early changed the image by up "
        f"to {int((uniform - floored).abs().max())} counts"
    )


def test_the_deterministic_renderer_reports_no_path_samples(tmp_path):
    """Zero means "the path tracer did not run", not "nothing measured"."""
    _img, result = _render_scene_result(
        tmp_path, "deterministic_spp.png", _flat_2d_scene, 1
    )
    assert result.render_plan.backend == "deterministic_wavefront"
    assert result.render_plan.path_samples_mean == 0.0
    assert result.render_plan.as_dict()["path_samples_mean"] == 0.0


# ---------------------------------------------------------------------------
# The light tree (DESIGN_path_tracer_roadmap.md 6a / 6b)
# ---------------------------------------------------------------------------


def _random_light_tree(seed=3, entries=17):
    """One frame's tree over a random mix of emitter shapes.

    Deliberately mixed: full cones (point rows), half-spread cones (two-sided
    triangles) and zero-spread ones (one-sided emitters), with and without
    distance falloff, so the probes exercise every branch of the importance
    function rather than one uniform kind.
    """
    import numpy as np

    from algan.rendering.raytracing.light_tree import build_light_tree

    rng = np.random.default_rng(seed)
    power = rng.random(entries) + 0.05
    center = rng.random((entries, 3)) * 8.0 - 4.0
    radius = rng.random((entries, 1)) * 0.3
    axis = rng.normal(size=(entries, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    theta_o = rng.choice([0.0, np.pi / 2, np.pi], entries)
    theta_e = np.full(entries, np.pi / 2)
    decay = rng.choice([0.0, 2.0], entries)
    node_f, node_i, leaf = build_light_tree(
        power, center - radius, center + radius, axis, theta_o, theta_e, decay
    )
    return node_f, node_i, leaf


def test_light_tree_descent_and_pmf_agree():
    """The MIS identity: the probability the descent returns for the leaf it
    lands on equals the probability the upward walk computes for that same
    leaf and point.

    This is the single test that keeps multiple importance sampling correct
    once selection depends on position. Next-event estimation divides by the
    descent's probability; a BSDF ray that later hits the same emitter forms
    its ``pdf_ne`` from the upward walk at the PREVIOUS vertex. If the two
    disagree the power-heuristic weights stop summing to one and the
    estimator is quietly biased -- nothing else in the suite would see it.
    """
    import numpy as np
    import torch as _torch

    from algan.rendering.raytracing.path_tracer_taichi import pt_light_tree_probe
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    node_f, node_i, _leaf = _random_light_tree()
    rng = np.random.default_rng(11)
    n = 4096
    pts = _torch.from_numpy((rng.random((n, 3)) * 24.0 - 12.0).astype(np.float32))
    u = _torch.from_numpy(rng.random(n).astype(np.float32))
    out = _torch.zeros((n, 4), dtype=_torch.float32)
    pt_light_tree_probe(
        _torch.from_numpy(node_f[None].copy()),
        _torch.from_numpy(node_i[None].copy()),
        0,
        pts,
        u,
        out,
    )
    got = out.numpy()
    assert (got[:, 0] >= 0).all(), "the descent failed to reach a leaf"
    rel = np.abs(got[:, 1] - got[:, 2]) / np.maximum(got[:, 1], 1e-30)
    assert rel.max() < 1e-5, (
        f"descent and upward-walk probabilities disagree by {rel.max():.3e} "
        "relative -- the two MIS ends are not evaluating the same selection "
        "pdf"
    )
    # And the probe really did exercise the whole tree, not one bright leaf.
    assert len(set(got[:, 0].astype(int).tolist())) > 8


def test_light_tree_selection_probabilities_sum_to_one():
    """Over every leaf, the descent's probabilities are a distribution.

    A tree that merely aims well is not enough: next-event estimation
    divides by the selection probability, so the probabilities have to be a
    normalized pmf over the entries or the estimate is scaled wrong. The
    interesting case is a node whose two children both score zero at a point
    while the parent does not -- the descent splits evenly there rather than
    dropping the sample, which is exactly what keeps this sum at one.
    """
    import numpy as np
    import torch as _torch

    from algan.rendering.raytracing.path_tracer_taichi import (
        pt_light_tree_pmf_probe,
    )
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    node_f, node_i, leaf = _random_light_tree()
    rng = np.random.default_rng(5)
    pts = _torch.from_numpy((rng.random((64, 3)) * 24.0 - 12.0).astype(np.float32))
    out = _torch.zeros((64, leaf.shape[0]), dtype=_torch.float32)
    pt_light_tree_pmf_probe(
        _torch.from_numpy(node_f[None].copy()),
        _torch.from_numpy(node_i[None].copy()),
        0,
        pts,
        _torch.from_numpy(leaf.astype(np.int32)),
        out,
    )
    sums = out.numpy().sum(1)
    assert np.abs(sums - 1.0).max() < 1e-4, (
        f"leaf selection probabilities sum to [{sums.min():.6f}, "
        f"{sums.max():.6f}], not 1"
    )


def test_light_tree_build_is_deterministic():
    """Two builds of one input give byte-identical tensors.

    The tree is rebuilt every render call, and ``tests/path_traced`` pixel-
    compares, so a build that depended on dictionary or sort order would show
    up as unexplainable frame drift rather than as a failure here.
    """
    first = _random_light_tree(seed=8, entries=64)
    second = _random_light_tree(seed=8, entries=64)
    for a, b, name in zip(first, second, ("node_f", "node_i", "entry_leaf")):
        assert (a == b).all(), f"{name} differs between two identical builds"


def test_light_tree_follows_a_light_that_moves_between_frames(tmp_path):
    """A moving light gets a different tree at each frame.

    ``light_pos`` is a per-frame tensor, so a tree built once per render call
    from frame 0 would aim every later frame's shadow rays at where the light
    used to be. The build is per frame (frames with identical emitter
    geometry share a row), which is what this pins.
    """
    import numpy as np

    from algan.rendering.raytracing import path_tracer as pt_host
    from algan.rendering.raytracing.light_tree import LT_BMIN

    captured = []
    original = pt_host._build_nee_tables

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append((out[6].detach().cpu().numpy().copy(), out[9].cpu().tolist()))
        return out

    settings = SMOKE_TEST.set(resolution=(32, 32), frames_per_second=2)
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    pt_host._build_nee_tables = capture
    try:
        SETTINGS.raytracing.set(samples_per_pixel=2, denoise=False)
        with Scene(video_settings=settings) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                floor = Prism(width=7.0, height=7.0, depth=0.1)
                floor.set_material(MeshLambertMaterial(color=WHITE))
                floor.spawn(animate=False)
                light = PointLight(
                    location=OUT * 4.0 + LEFT * 3.0, color=WHITE, intensity=1.0
                ).spawn(animate=False)
            light.move(RIGHT * 6.0)
            scene.save_frame(
                tmp_path / "lt_moving.png",
                video_settings=settings,
                at=[0, 1],
                overwrite=True,
            )
    finally:
        pt_host._build_nee_tables = original
        SceneManager.reset()
        SETTINGS.restore(snapshot)

    assert captured, "the path tracer never built a next-event table"
    # Every distinct tree the render built, whether the chunk held both
    # frames (two rows in one call) or one frame each (two calls).
    trees = [nodes[r] for nodes, rows in captured for r in sorted(set(rows))]
    assert len(trees) >= 2, "the render produced only one tree for two frames"
    spread = max(
        float(np.abs(a[:, LT_BMIN] - b[:, LT_BMIN]).max()) for a in trees for b in trees
    )
    assert spread > 1.0, (
        f"the two frames' trees bound the light in the same place (max x-min "
        f"difference {spread:.3f}) -- the build is not per frame"
    )


def _many_light_ring(scene):
    """A Lambert floor under 32 point lights on a ring.

    ``decay=2`` on purpose: Algan's light rows default to no distance
    falloff at all, and a light that does not fade with distance is one the
    tree has nothing to discriminate by -- the tree matches the flat CDF
    there rather than beating it. Physical falloff is the case the
    "too many lights" use case actually means.
    """
    import math as _math

    scene.set_background(BLACK)
    Scene.clear_lights()
    floor = Prism(width=9.0, height=9.0, depth=0.2)
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.spawn(animate=False)
    for i in range(32):
        a = 2.0 * _math.pi * i / 32
        PointLight(
            location=OUT * 2.5
            + RIGHT * (3.2 * _math.cos(a))
            + UP * (3.2 * _math.sin(a)),
            color=WHITE,
            intensity=0.15,
            decay=2.0,
        ).spawn(animate=False)


def test_light_tree_cuts_many_light_variance(tmp_path):
    """The point of the whole structure: at equal spp, aiming shadow rays by
    distance and orientation converges faster than aiming them by power.

    Both arms are unbiased estimators of the same integral, so this is a
    noise comparison against a high-sample reference, not a pixel match.
    """
    video = SMOKE_TEST.set(resolution=(64, 64))
    reference = _render_scene_exp(
        tmp_path,
        "lt_ref.png",
        _many_light_ring,
        128,
        video=video,
        experimental={"pt_light_tree": False, "pt_light_samples": 1},
    ).double()
    tree = _render_scene_exp(
        tmp_path,
        "lt_on.png",
        _many_light_ring,
        4,
        video=video,
        experimental={"pt_light_tree": True, "pt_light_samples": 1},
    ).double()
    flat = _render_scene_exp(
        tmp_path,
        "lt_off.png",
        _many_light_ring,
        4,
        video=video,
        experimental={"pt_light_tree": False, "pt_light_samples": 1},
    ).double()
    assert reference.mean() > 20, "the reference scene rendered (nearly) black"
    assert (tree - flat).abs().max() > 0, (
        "the two arms produced identical frames -- pt_light_tree did not "
        "reach the kernel"
    )
    mse_tree = float(((tree - reference) ** 2).mean())
    mse_flat = float(((flat - reference) ** 2).mean())
    assert mse_tree > 0
    assert mse_flat / mse_tree > 3.0, (
        f"the light tree cut mean squared error by only "
        f"{mse_flat / mse_tree:.2f}x (tree {mse_tree:.2f}, flat CDF "
        f"{mse_flat:.2f}) on 32 lights"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
