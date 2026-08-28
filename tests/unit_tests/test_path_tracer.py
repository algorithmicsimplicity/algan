"""Unit tests for the ``samples_per_pixel > 1`` path tracer.

Two layers:

* **Sampler tests** drive ``pt_sampler_probe`` directly -- the Sobol-Owen
  sampler is a pure function of ``(seed, frame, pixel, pair, sample index)``,
  so its stratification, reproducibility and decorrelation are testable
  without the scene pipeline.
* **Render tests** drive the real dispatch through ``Scene.save_frame``:
  the path tracer's deterministic transparency must reproduce the
  deterministic renderer's composite on unlit 2-D content away from edges
  (edges legitimately differ: jittered-sample anti-aliasing vs analytic
  coverage), and a path-traced frame must be byte-reproducible run-to-run.

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
    ORIGIN,
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
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel)
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


def test_path_traced_frame_is_reproducible(tmp_path):
    """Two renders of the same frame are byte-identical: the sampler is a
    pure function of (seed, frame, pixel, pair, sample) and accumulation is
    atomic-free with a fixed summation order.
    """
    a = _read(_render_stack_frame(tmp_path, "repro_a.png", 8))
    b = _read(_render_stack_frame(tmp_path, "repro_b.png", 8))
    assert torch.equal(a, b), (
        f"path-traced output changed between identical runs "
        f"(max diff {int((a - b).abs().max())})"
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


def _render_scene(tmp_path, name, build, samples_per_pixel, video=None,
                  **rt_kwargs):
    """Render one frame of ``build()``'s scene under the given settings."""
    settings = video if video is not None else STACK_SETTINGS
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel)
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
        f"diffuse furnace lost energy: darkest channel {lo} (expected white "
        "everywhere)"
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
        PointLight(location=OUT * 5.0, color=WHITE, intensity=1.0).spawn(
            animate=False
        )
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
    """With ``shadows`` on, NEE visibility rays darken occluded geometry."""

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=OUT * 6.0, color=WHITE, intensity=1.0).spawn(
            animate=False
        )
        floor = Prism(dimensions=(7.0, 7.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        blocker = Square(side_length=2.0, color=BLUE)
        blocker.move(OUT * 2.0)
        blocker.spawn(animate=False)

    lit = _render_scene(tmp_path, "shadow_off.png", build, 24, shadows=False)
    shadowed = _render_scene(tmp_path, "shadow_on.png", build, 24, shadows=True)
    h, w = lit.shape[0], lit.shape[1]
    # Just outside the blocker's straight-down projection the floor stays lit
    # in both renders; inside it only the shadowed render darkens. The
    # blocker itself is unlit 2-D and draws on top, so probe the floor next
    # to the frame centre... the blocker covers the centre, so probe under
    # its edge shadow: a point light above the centre projects the blocker's
    # shadow around the centre; the blocker occludes the view there too.
    # Probe a ring between the blocker's silhouette and the shadow edge.
    centre_bright_lit = float(lit[h // 2 + 12, w // 2, :3].float().mean())
    centre_bright_shadowed = float(
        shadowed[h // 2 + 12, w // 2, :3].float().mean()
    )
    assert centre_bright_shadowed < 0.75 * centre_bright_lit, (
        f"no shadow: lit {centre_bright_lit:.0f} vs shadowed "
        f"{centre_bright_shadowed:.0f}"
    )


def test_indirect_light_bleeds_color(tmp_path):
    """Global illumination: a red wall beside a white floor bleeds red onto
    it (the deterministic renderer cannot; the old Monte Carlo kernel needed
    an opt-in hack). Compared against its own far side rather than an
    absolute value so the test tracks lighting, not tonemapping.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=(UP * 2.0 + OUT * 5.0), color=WHITE,
                   intensity=1.2).spawn(animate=False)
        floor = Prism(dimensions=(8.0, 8.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        wall = Prism(dimensions=(0.1, 8.0, 3.0))
        wall.set_material(MeshLambertMaterial(color=RED))
        wall.move(RIGHT * 2.5 + OUT * 1.5)
        wall.spawn(animate=False)

    img = _render_scene(tmp_path, "bleed.png", build, 48).float()
    h, w = img.shape[0], img.shape[1]
    # OpenCV loads BGR: channel 2 is red, 0 is blue.
    near = img[h // 2, w // 2 + 6]
    far = img[h // 2, w // 2 - 20]
    near_redness = float(near[2] - near[0])
    far_redness = float(far[2] - far[0])
    assert near_redness > far_redness + 6, (
        f"no red bleed: near {near.tolist()} far {far.tolist()}"
    )


def test_lit_scene_render_is_reproducible(tmp_path):
    """The full Stage-2 transport (NEE jitter, lobe choices, RR) draws every
    random number from pure functions of the path identity, so a lit GI
    render reproduces byte-for-byte, exactly like the unlit stack.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_light_sources()
        PointLight(location=OUT * 5.0, color=WHITE, intensity=1.0).spawn(
            animate=False
        )
        floor = Prism(dimensions=(6.0, 6.0, 0.1))
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        ball = Sphere(radius=0.7)
        ball.set_material(
            MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3)
        )
        ball.move(OUT * 1.0)
        ball.spawn(animate=False)

    a = _render_scene(tmp_path, "lit_a.png", build, 12, shadows=True)
    b = _render_scene(tmp_path, "lit_b.png", build, 12, shadows=True)
    assert torch.equal(a, b), (
        f"lit path-traced output changed between identical runs "
        f"(max diff {int((a - b).abs().max())})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
