"""The glossy prefilter's fused finish and its exact buffer reset.

``gloss_finish`` replaces a finished frame's per-level ``gloss_pyramid_level``
launches, its ``gloss_composite`` and the ``_gloss_clear`` after it with one
launch (``ALGAN_OPT_DISABLE=glossfuse`` restores them), and ``gloss_clear``
replaces the clear's strided column write (``ALGAN_OPT_DISABLE=exactfills``).
Both are held here to the launches they replace, bit for bit, on the CPU
kernels: several frames through the same buffers, so every frame reads the
state the previous frame's reset left, and everything the new reset leaves
alone is poisoned with NaN first -- a read of any of it would reach the output.

The fused finish works only inside the frame's glossy bounding box
(``gl_box``), so the windows include glossy pixels confined to an odd corner of
the frame and frames with none at all, where the separate launches still
reduce and composite the whole frame.

Outside the fast suite: only a change to the glossy route can break it.
"""

import pytest
import torch

import algan  # noqa: F401  (initialises Taichi)
import algan.animation_timeline.timeline as timeline_module
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.glossy_prefilter_taichi import (
    _BOX_SIGMA,
    GL_BOX_EMPTY,
    GL_BOX_WIDTH,
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


def _frames(width, height, frames, tonemapping, seed, region=None):
    """One window's tile: covered pixels, accumulator rows and the frame buffer.

    Three quarters of each frame's pixels are covered and about half of those
    carry a prefiltered glossy branch; the rest of every row is noise, which is
    what lets a read of the wrong column show.

    ``region`` ``(x0, y0, x1, y1)``, inclusive, confines the glossy pixels to
    that rectangle, and the window's middle frame then has none at all.
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
    glossy = rand(n) > 0.45
    if region is not None:
        x0, y0, x1, y1 = region
        frame, pixel = covered_idx // ppf, covered_idx % ppf
        x, y = pixel % width, pixel // width
        glossy &= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        glossy &= frame != frames // 2
    gloss[:, GL_ROW_SIGMA_SCALE] = torch.where(
        glossy, rand(n) * 40.0 + 0.05, torch.zeros(n)
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


def _render_window(
    width, height, frames, max_levels, tonemapping, seed, finish, region=None
):
    """Scatter, composite and finish every frame of a window in turn.

    ``finish(frame, levels_t, gl_main, gl_pyr, gl_box, out)`` is the arm under
    test. Returns the frame buffer and what ``finish`` returned per frame.
    """
    covered_idx, pix_accum, out, bounds = _frames(
        width, height, frames, tonemapping, seed, region
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
    gl_box = torch.full((GL_BOX_WIDTH,), 12345, dtype=torch.int32)
    stand_in = torch.zeros(frames, 3)
    tracer._gloss_clear(gl_main, gl_pyr, gl_box)
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
            gl_box,
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
        pyramids.append(finish(f, levels_t, gl_main, gl_pyr, gl_box, out))
    return out, pyramids


def _reference_finish(tonemapping, width, height):
    """The launches ``gloss_finish`` replaces, spelled out: one reduction per
    level, the composite, and the strided clear, all over the whole frame.
    Returns the frame's pyramid above level 0, read right after its reduction.
    """

    def finish(f, levels_t, gl_main, gl_pyr, gl_box, out):
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
        gl_box[:2] = GL_BOX_EMPTY
        gl_box[2:] = -1
        return pyramid

    return finish


def _tracer_finish(tonemapping, width, height, poison):
    """``tracer._gloss_finish_frame`` as the render calls it. Returns the
    glossy bounding box the frame's scatter left for it.

    With ``poison``, everything the fused reset does not rewrite is set to
    NaN after it -- after its own three promises are checked: every blur
    radius back at -1, the whole pyramid back at zero, and the box empty.
    """

    def finish(f, levels_t, gl_main, gl_pyr, gl_box, out):
        box = gl_box.tolist()
        tracer._gloss_finish_frame(
            f, levels_t, gl_main, gl_pyr, gl_box, width, height, tonemapping, out
        )
        if poison:
            assert torch.all(gl_main[:, GL_MAIN_SIGMA] == -1.0)
            assert torch.equal(_bits(gl_pyr), _bits(torch.zeros_like(gl_pyr)))
            assert gl_box.tolist() == [GL_BOX_EMPTY, GL_BOX_EMPTY, -1, -1]
            sigma = gl_main[:, GL_MAIN_SIGMA].clone()
            gl_main.fill_(NAN)
            gl_main[:, GL_MAIN_SIGMA] = sigma
        return box

    return finish


def _disable(monkeypatch, *names):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset(names))


def _beyond_reach(levels, width, height, box):
    """Which texels above level 0 the level-0 box ``box`` does not reach
    through the reduction (``glossy_prefilter_taichi._box_extent``).
    """
    x0, y0, x1, y1 = box
    beyond = []
    for lvl, (_off, w, h) in enumerate(levels[1:], start=1):
        cols = torch.arange(w)
        rows = torch.arange(h)
        if x1 >= x0:
            inside = ((cols >= x0 >> lvl) & (cols <= x1 >> lvl))[None, :] & (
                (rows >= y0 >> lvl) & (rows <= y1 >> lvl)
            )[:, None]
        else:
            inside = torch.zeros(h, w, dtype=torch.bool)
        beyond.append(~inside.reshape(-1))
    return torch.cat(beyond) if beyond else torch.zeros(0, dtype=torch.bool)


def _case(width, height, max_levels, region=None):
    """A frame size, with its glossy pixels confined to ``region`` if given.

    Unconfined cases keep the node ids they had before regions existed.
    """
    name = f"{width}-{height}-{max_levels}"
    if region is not None:
        name += "-box" + "-".join(map(str, region))
    return pytest.param(width, height, max_levels, region, id=name)


@pytest.mark.parametrize("tonemapping", [3, 0, 2])
@pytest.mark.parametrize(
    ("width", "height", "max_levels", "region"),
    [
        _case(7, 5, 10),
        _case(16, 9, 10),
        _case(1, 1, 10),
        _case(1, 6, 10),
        _case(33, 17, 10),
        _case(33, 17, 3),
        # Glossy pixels in an odd corner of the frame, and a frame with none:
        # the box, and its reach at every level, is then a strict part of
        # the frame, which the full-frame launches still reduce and scan.
        _case(33, 17, 10, (19, 3, 26, 9)),
        _case(33, 17, 3, (0, 11, 4, 16)),
        _case(16, 9, 10, (5, 0, 5, 8)),
        _case(1, 6, 10, (0, 2, 0, 4)),
        _case(40, 23, 10, (31, 17, 39, 22)),
    ],
)
def test_the_fused_finish_matches_the_launches_it_replaces(
    monkeypatch, width, height, max_levels, region, tonemapping
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
        region,
    )
    # The switches off: the tracer's own fallback is the reference.
    off_out, _ = _render_window(
        width,
        height,
        frames,
        max_levels,
        tonemapping,
        seed,
        _tracer_finish(tonemapping, width, height, poison=False),
        region,
    )
    _disable(monkeypatch)
    on_out, boxes = _render_window(
        width,
        height,
        frames,
        max_levels,
        tonemapping,
        seed,
        _tracer_finish(tonemapping, width, height, poison=True),
        region,
    )
    assert torch.equal(_bits(off_out), _bits(ref_out))
    assert torch.equal(_bits(on_out), _bits(ref_out))
    # What the box finish rests on: the full-frame reduction of each frame is
    # exactly zero wherever its box does not reach.
    levels, _total = tracer._gloss_pyramid_levels(width, height, max_levels)
    for f in range(frames):
        beyond = _beyond_reach(levels, width, height, boxes[f])
        assert torch.equal(
            _bits(ref_pyr[f][beyond]), _bits(torch.zeros_like(ref_pyr[f][beyond]))
        ), f"frame {f}"
    if region is not None:
        x0, y0, x1, y1 = region
        for f, box in enumerate(boxes):
            if f == frames // 2:
                assert box == [GL_BOX_EMPTY, GL_BOX_EMPTY, -1, -1]
            else:
                assert x0 <= box[0] <= box[2] <= x1, (f, box)
                assert y0 <= box[1] <= box[3] <= y1, (f, box)
    # The window drew something through the prefilter, or this proves nothing.
    assert not torch.equal(
        _bits(ref_out),
        _bits(_frames(width, height, frames, tonemapping, seed, region)[2]),
    )


def test_gloss_clear_writes_the_bytes_the_strided_clear_wrote(monkeypatch):
    width, height = 13, 7
    levels, total = tracer._gloss_pyramid_levels(width, height, 10)
    ppf = width * height
    _disable(monkeypatch, "exactfills")
    ref_main = torch.full((ppf, GL_MAIN_WIDTH), NAN)
    ref_pyr = torch.full((total, GL_PYR_WIDTH), NAN)
    ref_box = torch.full((GL_BOX_WIDTH,), 7, dtype=torch.int32)
    tracer._gloss_clear(ref_main, ref_pyr, ref_box)
    _disable(monkeypatch)
    new_main = torch.full((ppf, GL_MAIN_WIDTH), NAN)
    new_pyr = torch.full((total, GL_PYR_WIDTH), NAN)
    new_box = torch.full((GL_BOX_WIDTH,), 7, dtype=torch.int32)
    tracer._gloss_clear(new_main, new_pyr, new_box)
    assert torch.equal(_bits(new_main), _bits(ref_main))
    # Every level: the finish reduces only inside the box and leaves the rest
    # of the pyramid to be the zero this wrote.
    assert torch.equal(_bits(new_pyr), _bits(ref_pyr))
    assert torch.equal(_bits(new_pyr), _bits(torch.zeros_like(new_pyr)))
    assert new_box.tolist() == ref_box.tolist() == [GL_BOX_EMPTY, GL_BOX_EMPTY, -1, -1]


def test_the_switches_are_read_at_every_use(monkeypatch):
    for name in ("glossfuse", "exactfills"):
        _disable(monkeypatch)
        assert not tracer._tracer_opt_disabled(name)
        _disable(monkeypatch, name)
        assert tracer._tracer_opt_disabled(name)
        _disable(monkeypatch)
        assert not tracer._tracer_opt_disabled(name)
