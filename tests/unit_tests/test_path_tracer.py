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
    BLUE,
    GREEN,
    RED,
    SETTINGS,
    SMOKE_TEST,
    Off,
    Scene,
    SceneManager,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
