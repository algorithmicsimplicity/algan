"""The glossy prefilter's fused finish and its exact buffer reset.

``gloss_finish`` replaces a finished frame's per-level ``gloss_pyramid_level``
launches, its ``gloss_composite`` and the ``_gloss_clear`` after it with one
launch (``ALGAN_OPT_DISABLE=glossfuse`` restores them), and ``gloss_clear``
replaces the clear's strided column write (``ALGAN_OPT_DISABLE=exactfills``).
Both are held here to the launches they replace, bit for bit, on the CPU
kernels: several frames through the same buffers, so every frame reads the
state the previous frame's reset left, and everything the new reset leaves
alone is poisoned with NaN first -- a read of any of it would reach the output.

Outside the fast suite: only a change to the glossy route can break it.
"""

import pytest
import torch

import algan  # noqa: F401  (initialises Taichi)
import algan.animation_timeline.timeline as timeline_module
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.glossy_prefilter_taichi import (
    _BOX_SIGMA,
    GL_MAIN_SIGMA,
    GL_MAIN_WIDTH,
    GL_PYR_WIDTH,
    GL_ROW_DIST,
    GL_ROW_DP,
    GL_ROW_MAG,
    GL_ROW_SIGMA_SCALE,
    GL_ROW_W,
    GL_ROW_WIDTH,
    gloss_composite,
    gloss_pyramid_level,
    gloss_scatter,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    wf_composite_accum_sparse,
)

NAN = float("nan")


def _bits(t):
    """A tensor's bytes, comparable with NaNs in it."""
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    return t


def _frames(width, height, frames, tonemapping, seed):
    """One window's tile: covered pixels, accumulator rows and the frame buffer.

    Three quarters of each frame's pixels are covered and about half of those
    carry a prefiltered glossy branch; the rest of every row is noise, which is
    what lets a read of the wrong column show.
    """
    gen = torch.Generator().manual_seed(seed)
    ppf = width * height
    idx = []
    for f in range(frames):
        keep = max(1, (3 * ppf) // 4)
        pix = torch.randperm(ppf, generator=gen)[:keep].sort().values
        idx.append(pix + f * ppf)
    covered_idx = torch.cat(idx).to(torch.int32)
    n = int(covered_idx.numel())

    def rand(*shape):
        return torch.rand(*shape, generator=gen)

    pix_accum = torch.zeros(2 * n, GL_ROW_WIDTH)
    pix_accum[:, 0:4] = rand(2 * n, 4) * 0.5
    pix_accum[:, 4:7] = rand(2 * n, 3)
    gloss = pix_accum[n:]
    gloss[:, GL_ROW_DIST] = rand(n) * (rand(n) > 0.3)
    gloss[:, GL_ROW_W : GL_ROW_W + 3] = rand(n, 3)
    gloss[:, GL_ROW_SIGMA_SCALE] = torch.where(
        rand(n) > 0.45, rand(n) * 40.0 + 0.05, torch.zeros(n)
    )
    gloss[:, GL_ROW_DP] = rand(n) * 3.0
    gloss[:, GL_ROW_MAG] = 1.0 + 2.0 * rand(n) * (rand(n) > 0.5)
    if tonemapping == 3:
        out = rand(frames, ppf, 4) * 255.0
    else:
        out = torch.randint(0, 256, (frames, ppf, 4), generator=gen).to(torch.uint8)
    bounds = torch.searchsorted(
        covered_idx, torch.arange(frames + 1, dtype=torch.int32) * ppf
    ).tolist()
    return covered_idx, pix_accum, out, bounds


def _render_window(width, height, frames, max_levels, tonemapping, seed, finish):
    """Scatter, composite and finish every frame of a window in turn.

    ``finish(frame, levels_t, gl_main, gl_pyr, out)`` is the arm under test.
    Returns the frame buffer and each frame's pyramid above level 0, read
    right after the frame's reduction.
    """
    covered_idx, pix_accum, out, bounds = _frames(
        width, height, frames, tonemapping, seed
    )
    n = int(covered_idx.numel())
    levels, total = tracer._gloss_pyramid_levels(width, height, max_levels)
    levels_t = torch.tensor([c for row in levels for c in row], dtype=torch.int32).view(
        len(levels), 3
    )
    sigma_max = _BOX_SIGMA * float(1 << (len(levels) - 1))
    ppf = width * height
    # Arena noise: whatever the reset leaves unwritten must not be read.
    gl_main = torch.full((ppf, GL_MAIN_WIDTH), NAN)
    gl_pyr = torch.full((total, GL_PYR_WIDTH), NAN)
    stand_in = torch.zeros(frames, 3)
    tracer._gloss_clear(gl_main, gl_pyr)
    pyramids = []
    for f in range(frames):
        a, b = bounds[f], bounds[f + 1]
        gloss_scatter(
            b - a,
            n,
            f * ppf,
            f,
            width,
            float(sigma_max),
            covered_idx[a:b],
            pix_accum,
            gl_main,
            gl_pyr,
            out,
            a,
            stand_in,
            False,
            f,
        )
        wf_composite_accum_sparse(
            0,
            width,
            height,
            0,
            0,
            covered_idx[a:b],
            pix_accum[a:b],
            tonemapping,
            float(tracer.rt_settings.tonemap_exposure),
            0,
            out,
        )
        pyramids.append(finish(f, levels_t, gl_main, gl_pyr, out))
    return out, pyramids


def _reference_finish(tonemapping, width, height):
    """The launches ``gloss_finish`` replaces, spelled out: one reduction per
    level, the composite, and the strided clear.
    """

    def finish(f, levels_t, gl_main, gl_pyr, out):
        num_levels = int(levels_t.shape[0])
        for lvl in range(1, num_levels):
            gloss_pyramid_level(lvl - 1, lvl, levels_t, gl_pyr)
        pyramid = gl_pyr[width * height :].clone()
        gloss_composite(
            width * height,
            width,
            height,
            num_levels,
            f,
            tonemapping,
            float(tracer.rt_settings.tonemap_exposure),
            levels_t,
            gl_main,
            gl_pyr,
            out,
        )
        gl_main.zero_()
        gl_main[:, GL_MAIN_SIGMA] = -1.0
        gl_pyr.zero_()
        return pyramid

    return finish


def _tracer_finish(tonemapping, width, height, poison):
    """``tracer._gloss_finish_frame`` as the render calls it.

    With ``poison``, everything the fused reset does not rewrite is set to
    NaN after it -- after its own two promises are checked: every blur radius
    back at -1 and level 0 back at zero.
    """
    ppf = width * height

    def finish(f, levels_t, gl_main, gl_pyr, out):
        tracer._gloss_finish_frame(
            f, levels_t, gl_main, gl_pyr, width, height, tonemapping, out
        )
        pyramid = gl_pyr[ppf:].clone()
        if poison:
            assert torch.all(gl_main[:, GL_MAIN_SIGMA] == -1.0)
            assert torch.equal(
                _bits(gl_pyr[:ppf]), _bits(torch.zeros(ppf, GL_PYR_WIDTH))
            )
            sigma = gl_main[:, GL_MAIN_SIGMA].clone()
            gl_main.fill_(NAN)
            gl_main[:, GL_MAIN_SIGMA] = sigma
            gl_pyr[ppf:] = NAN
        return pyramid

    return finish


def _disable(monkeypatch, *names):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset(names))


@pytest.mark.parametrize("tonemapping", [3, 0, 2])
@pytest.mark.parametrize(
    ("width", "height", "max_levels"),
    [(7, 5, 10), (16, 9, 10), (1, 1, 10), (1, 6, 10), (33, 17, 10), (33, 17, 3)],
)
def test_the_fused_finish_matches_the_launches_it_replaces(
    monkeypatch, width, height, max_levels, tonemapping
):
    frames, seed = 3, width * 131 + height * 7 + max_levels + tonemapping
    _disable(monkeypatch, "glossfuse", "exactfills")
    ref_out, ref_pyr = _render_window(
        width,
        height,
        frames,
        max_levels,
        tonemapping,
        seed,
        _reference_finish(tonemapping, width, height),
    )
    # The switches off: the tracer's own fallback is the reference.
    off_out, off_pyr = _render_window(
        width,
        height,
        frames,
        max_levels,
        tonemapping,
        seed,
        _tracer_finish(tonemapping, width, height, poison=False),
    )
    _disable(monkeypatch)
    on_out, on_pyr = _render_window(
        width,
        height,
        frames,
        max_levels,
        tonemapping,
        seed,
        _tracer_finish(tonemapping, width, height, poison=True),
    )
    assert torch.equal(_bits(off_out), _bits(ref_out))
    assert torch.equal(_bits(on_out), _bits(ref_out))
    for f in range(frames):
        assert torch.equal(_bits(on_pyr[f]), _bits(ref_pyr[f])), f"frame {f}"
    # The window drew something through the prefilter, or this proves nothing.
    assert not torch.equal(
        _bits(ref_out), _bits(_frames(width, height, frames, tonemapping, seed)[2])
    )


def test_gloss_clear_writes_the_bytes_the_strided_clear_wrote(monkeypatch):
    width, height = 13, 7
    levels, total = tracer._gloss_pyramid_levels(width, height, 10)
    ppf = width * height
    _disable(monkeypatch, "exactfills")
    ref_main = torch.full((ppf, GL_MAIN_WIDTH), NAN)
    ref_pyr = torch.full((total, GL_PYR_WIDTH), NAN)
    tracer._gloss_clear(ref_main, ref_pyr)
    _disable(monkeypatch)
    new_main = torch.full((ppf, GL_MAIN_WIDTH), NAN)
    new_pyr = torch.full((total, GL_PYR_WIDTH), NAN)
    tracer._gloss_clear(new_main, new_pyr)
    assert torch.equal(_bits(new_main), _bits(ref_main))
    assert torch.equal(_bits(new_pyr[:ppf]), _bits(ref_pyr[:ppf]))
    # The levels above 0 are left for the frame's reduction to write.
    assert torch.isnan(new_pyr[ppf:]).all()


def test_the_switches_are_read_at_every_use(monkeypatch):
    for name in ("glossfuse", "exactfills"):
        _disable(monkeypatch)
        assert not tracer._tracer_opt_disabled(name)
        _disable(monkeypatch, name)
        assert tracer._tracer_opt_disabled(name)
        _disable(monkeypatch)
        assert not tracer._tracer_opt_disabled(name)
