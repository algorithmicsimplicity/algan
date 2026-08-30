"""One frame through the whole renderer, on whatever device is configured.

The smallest thing that answers "does Algan render here at all". It exists for
the Apple GPU (``DESIGN_mps_support.md``), where the interesting failures are
not assertion failures: Metal answers an over-wide kernel with ``computeFunction
must not be nil`` and an int64 atomic with ``bind_pipeline`` -- both ``SIGABRT``
inside Taichi rather than exceptions Python can catch -- so a run that dies
without a traceback is itself the result. Running this before the test suite in
``.github/workflows/mps_probe.yaml`` costs a minute and separates "the renderer
aborts on this backend" from "one test disagrees about a pixel", which the
suite's output otherwise blends together.

It renders through ``save_frame``, so it needs no encoder and no LaTeX; the
geometry is chosen to reach the parts MPS-friendly mode touches -- overlapping
opaque solids (the one-mesh coverage ceiling and its facing-split sums), a
bezier circuit (the raster path), and a shadow (the sheet compaction).

    uv run python benchmarks/_mps_render_smoke.py
    uv run python benchmarks/_mps_render_smoke.py --verify-torch-ops

Prints what it resolved, writes ``mps_render_smoke/frame.png``, and exits
non-zero if the frame is empty, uniform, or not finite.

``--verify-torch-ops`` adds the attribution pass: every suspect torch op is
re-run on the CPU over its own real inputs and the two answers compared, so a
frame that comes out wrong names the op that made it wrong instead of leaving a
choice between a miscompiled kernel and a mis-substituted dtype. It is the
in-situ counterpart to ``_mps_torch_op_probe.py``, which asks the same question
of synthetic data: the probe isolates, this attributes. Each op is checked
against ITS OWN inputs rather than against a CPU run of the whole pipeline, so
only the FIRST op to go wrong is reported -- everything downstream computes
correctly on the wrong values it was handed, and says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from algan import LD, OUTWARD, RIGHT, UP, Off, Scene, Sphere, Square  # noqa: E402
from algan.rendering.mps_compat import (  # noqa: E402
    accumulate_dtype,
    mps_friendly,
    reduction_index_dtype,
)
from algan.settings import SETTINGS  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "mps_render_smoke"


def _host(tensor):
    """A float64 CPU copy of ``tensor``, for a checksum both arms can share.

    ``.to(torch.float64)`` on an MPS tensor raises -- Metal has no float64 at
    all, which is the whole subject here -- so the move to the host comes
    FIRST. Getting that order wrong is what took down the first run that
    carried these diagnostics.
    """
    return tensor.detach().cpu().to(torch.float64)


#: Every diagnostic line, in order, so the run can repeat them at the end.
_NOTES: list = []


def _say(build, *args):
    """Print what ``build`` returns, or why it could not be built.

    A diagnostic must never be able to take down the render it instruments:
    these run inside the pipeline, and a bad one turns a run that would have
    produced a frame and a table into a traceback and neither. Losing one line
    of a report is a much smaller cost than losing the round.

    The line is also kept, so :func:`_replay_notes` can repeat the whole set
    after the render. These lines are emitted from inside the pipeline, which
    puts them thousands of Taichi compile messages above the end of the log --
    and the GitHub Actions API serves a fixed window at the END of a job's log,
    so an inline diagnostic is one that cannot be read without fetching the
    whole thing. Repeating them costs nothing and makes the report reachable.
    """
    try:
        line = build(*args)
    except Exception as exc:  # noqa: BLE001
        line = f"  [report] {build.__name__} failed: {type(exc).__name__}: {exc}"
    _NOTES.append(line)
    print(line)


def _replay_notes():
    """Repeat every diagnostic line, where the log tail can reach it."""
    if not _NOTES:
        return
    print("\npipeline diagnostics, repeated so the log tail carries them:")
    for line in _NOTES:
        print(line)


def _install_pipeline_report():
    """Report what the raster pipeline found, so a black frame is localizable.

    A frame that renders to completion and comes out flat says nothing about
    where it went flat: no geometry reached the rasteriser, no fragment
    survived, or the shading wrote somewhere the frame buffer is not. These
    three numbers separate those, and they cost one print per chunk.

    The geometry line separates one more thing, and it is the one that made the
    fragment counts confusing: the spheres are PN-tessellated adaptively, and
    the level criterion is a THRESHOLD on float arithmetic, so two devices may
    legitimately emit different triangles and hence different fragment counts.
    A fragment count that differs with the triangle count is a tessellation
    difference and expected; one that differs with the triangle count EQUAL is
    a rasteriser defect.

    Every number below is computed on the CPU, for two reasons that both
    matter. A diagnostic must not depend on the thing it is diagnosing -- these
    exist to judge MPS, so reducing on MPS would let the defect under
    investigation quietly shape the evidence. And the arms are compared to each
    other, so the reduction has to be the same arithmetic on both; a float32
    device sum against a float32 host sum differs in the last digits for
    reasons that have nothing to do with the question.
    """
    from algan.rendering.raytracing import raster_pipeline

    original = raster_pipeline.prepare_sparse_raster_coverage

    def geometry_line(merged):
        tri_pos = _host(merged["tri_pos"])
        return (
            f"  [geometry] triangles={tri_pos.shape[1]} "
            f"pos_sum={float(tri_pos.sum()):.6f}"
        )

    def coverage_line(cov):
        cov = _host(cov)
        return (
            f"  [pipeline] frag_cov min={float(cov.min()):.6f} "
            f"max={float(cov.max()):.6f} sum={float(cov.sum()):.3f}"
        )

    def reporting(merged, *args, **kwargs):
        if isinstance(merged, dict) and merged.get("tri_pos") is not None:
            _say(geometry_line, merged)
        coverage = original(merged, *args, **kwargs)
        if coverage is None:
            print("  [pipeline] no coverage at all -- nothing rasterised")
            return coverage
        print(
            f"  [pipeline] fragments={coverage.get('num_fragments')} "
            f"covered_pixels={coverage.get('num_covered')} "
            f"sheets={coverage.get('num_sheets')}"
        )
        cov = coverage.get("frag_cov")
        if cov is not None and cov.numel():
            _say(coverage_line, cov)
        return coverage

    raster_pipeline.prepare_sparse_raster_coverage = reporting

    # The compaction's INPUT, which is what separates "the sheet count is
    # wrong" from "what the sheet count was computed from is wrong". Every
    # band boundary is a change in ``frag_key >> 32``, so the number of
    # distinct pixels in the key is a floor on the sheet count: if it is
    # ~30929 and the sheets are 128, the compaction lost them; if it is itself
    # ~128, the key arrived broken and the rasteriser or the buffer binding
    # under it is what to look at, not a torch op.
    from algan.rendering.raytracing import sheets as _sheets

    original_compact = _sheets.compact_sheets

    def key_lines(coverage):
        # ``frag_key`` is ``pixel << 32 | bit_cast(depth)``, written by one
        # statement of raster_tri_write beside frag_ref / frag_cov / frag_msk.
        # Reporting the two words SEPARATELY, and the siblings alongside, is
        # what tells a bad value apart from a bad binding: a constant depth
        # word with a healthy pixel word and healthy siblings is the kernel's
        # arithmetic, while everything garbled together is the buffer.
        n = int(coverage["num_fragments"])
        key = coverage["frag_key"][:n].cpu()
        pixels = key >> 32
        low = (key & 0xFFFFFFFF).to(torch.int32)
        depth = low.view(torch.float32)
        # Compact on purpose: these lines have to survive the log tail, and
        # every line spent here is one the report loses at the far end.
        parts = [
            f"  [compact-in] n={n} pix[{int(pixels.min())}..{int(pixels.max())}]"
            f"x{int(torch.unique(pixels).numel())}"
            f" depth[{float(depth.min()):.4f}..{float(depth.max()):.4f}]"
            f"x{int(torch.unique(depth).numel())}"
            f" lo0={int(low[0]) & 0xFFFFFFFF:08x}"
        ]
        stats = []
        for name in ("frag_ref", "frag_msk", "frag_cap"):
            sibling = coverage.get(name)
            if sibling is None:
                continue
            sibling = sibling[:n].cpu()
            lo_v, hi_v = sibling.min().item(), sibling.max().item()
            fmt = "{:.4f}" if sibling.is_floating_point() else "{:.0f}"
            stats.append(
                f"{name.removeprefix('frag_')}[{fmt.format(lo_v)}..{fmt.format(hi_v)}]"
                f"x{int(torch.unique(sibling).numel())}"
            )
        parts.append("  [compact-in] " + " ".join(stats))
        return "\n".join(parts)

    def group_line(stream):
        # ``num_groups`` counts (pixel, mesh, facing) triples -- the banding
        # BEFORE the conflict-rank split and before the shading-class split.
        # It brackets where a collapsed sheet count collapsed: a healthy
        # num_groups with 128 sheets puts it in the two splits (both of which
        # build a wide composite key -- ``band * 16 + rank``, then ``band *
        # 2**25 + class``, the second reaching ~2**40 and so past where MPS
        # int64 stops being exact); a num_groups already near 128 puts it in
        # the sort and the group detection ahead of them.
        return (
            f"  [compact-out] groups={int(stream['num_groups'])} "
            f"split_groups={int(stream['num_split_groups'])} "
            f"sheets={int(stream['num_sheets'])}"
        )

    def split_line(coverage, args, kwargs, real):
        # The SAME inputs compacted with the shading-class split off. That
        # split is the one step that builds a key wide enough to be at risk --
        # ``band * 2**25 + cls`` reaches ~2**40, where MPS int64 stops being
        # exact -- and the two counts separate the cases without needing to see
        # inside the function: a re-run that finds the CPU's ~17065-plus bands
        # puts the collapse in the split, and one that finds 128 again puts it
        # before, in the sort and the group detection.
        #
        # Advisory rather than authoritative: the compaction writes through
        # some of what it is handed, so this second pass sees inputs the first
        # may have touched. It runs after the real call for that reason.
        kwargs = {**kwargs, "shade_split": False}
        again = original_compact(coverage, *args, **kwargs)
        return (
            f"  [compact-alt] shade_split=False -> "
            f"sheets={0 if again is None else int(again['num_sheets'])} "
            f"(with the split: {0 if real is None else int(real['num_sheets'])})"
        )

    def reporting_compact(coverage, *args, **kwargs):
        _say(key_lines, coverage)
        stream = original_compact(coverage, *args, **kwargs)
        if stream is not None:
            _say(group_line, stream)
        _say(split_line, coverage, args, kwargs, stream)
        return stream

    # The module attribute is the only binding to patch: ``raster_pipeline``
    # imports the name inside the function that calls it, so it resolves this
    # attribute per call rather than having captured the original at import.
    _sheets.compact_sheets = reporting_compact


#: name -> [calls checked, mismatching calls, first mismatch detail]
_OP_STATS: dict = {}


def _to_cpu(value):
    """A CPU twin of one argument, leaving non-tensors alone."""
    return value.cpu() if isinstance(value, torch.Tensor) else value


def _same(a, b):
    """Do two results agree? Returns None when they do, else a description.

    Integers must match exactly -- they are ids, counts and offsets, and an
    off-by-one is the whole defect. Floats get a tolerance, because the CPU and
    the GPU reassociate reductions differently and always will; the point here
    is to catch an op that is WRONG, not one that rounds elsewhere.
    """
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return f"{len(a)} results vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            detail = _same(x, y)
            if detail is not None:
                return f"[{i}] {detail}"
        return None
    if not isinstance(a, torch.Tensor):
        return None if a == b else f"{a!r} vs {b!r}"
    b = b.cpu()
    a = a.cpu()
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} vs {b.dtype}"
    if a.is_floating_point():
        scale = a.abs().max().clamp_min(1.0)
        bad = (a - b).abs() > 1e-3 * scale
    else:
        bad = a != b
    count = int(bad.sum())
    if count == 0:
        return None
    where = int(bad.reshape(-1).nonzero()[0][0])
    return (
        f"{count}/{a.numel()} differ, first at {where}: "
        f"cpu {b.reshape(-1)[where]!s} vs mps {a.reshape(-1)[where]!s}"
    )


def _record(name, detail):
    entry = _OP_STATS.setdefault(name, [0, 0, ""])
    entry[0] += 1
    if detail is not None:
        entry[1] += 1
        if not entry[2]:
            entry[2] = detail


def _verified(name, function, inplace_self=False):
    """``function``, wrapped so each MPS call is re-checked against the CPU.

    ``inplace_self`` marks a mutating method: what has to be compared is the
    tensor the call wrote through, so the twin needs a clone taken BEFORE the
    real call runs -- afterwards there is nothing left to clone.
    """

    def wrapper(*args, **kwargs):
        on_mps = any(
            isinstance(a, torch.Tensor) and a.device.type == "mps"
            for a in (*args, *kwargs.values())
        )
        if not on_mps:
            return function(*args, **kwargs)
        twin_args = [_to_cpu(a) for a in args]
        twin_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}
        result = function(*args, **kwargs)
        try:
            twin = function(*twin_args, **twin_kwargs)
        except Exception as exc:  # noqa: BLE001
            _record(name, f"the CPU twin raised {type(exc).__name__}: {exc}")
            return result
        _record(
            name,
            _same(
                args[0] if inplace_self else result,
                twin_args[0] if inplace_self else twin,
            ),
        )
        return result

    return wrapper


def _install_torch_op_verifier():
    """Check the torch ops the compaction is built out of, against the CPU.

    Wrapped at ``torch``/``torch.Tensor`` rather than at the call sites,
    because there are 67 gathers and 9 segmented reductions in ``sheets.py``
    alone and the interesting one is whichever nobody suspected.
    """
    for name in ("cumsum", "unique", "argsort", "sort", "searchsorted", "cumprod"):
        setattr(torch, name, _verified(f"torch.{name}", getattr(torch, name)))
    for name in ("scatter_add_", "scatter_reduce_"):
        setattr(
            torch.Tensor,
            name,
            _verified(f"Tensor.{name}", getattr(torch.Tensor, name), inplace_self=True),
        )
    for name in ("index_select", "amin", "amax", "cummax", "cummin"):
        setattr(
            torch.Tensor, name, _verified(f"Tensor.{name}", getattr(torch.Tensor, name))
        )


def _report_torch_ops() -> bool:
    """Print the attribution table. True when every checked op agreed."""
    print("\ntorch ops checked against the CPU over their own real inputs:")
    checked = [(n, s) for n, s in sorted(_OP_STATS.items()) if s[0]]
    if not checked:
        print("  (nothing ran on MPS -- no op was checked)")
        return True
    clean = True
    for name, (calls, bad, detail) in checked:
        if bad:
            clean = False
            print(f"  FAIL  {name}: {bad}/{calls} calls differ -- {detail}")
        else:
            print(f"  ok    {name}: {calls} calls agree")
    return clean


def main() -> int:
    verify = "--verify-torch-ops" in sys.argv[1:]
    print(f"render device    : {SETTINGS.computing.render_device}")
    print(f"mps_friendly     : {SETTINGS.computing.mps_friendly} -> {mps_friendly()}")
    print(f"accumulate dtype : {accumulate_dtype()}")
    print(f"reduction dtype  : {reduction_index_dtype()}")
    print(f"torch            : {torch.__version__}")
    print(f"verify torch ops : {verify}")

    _install_pipeline_report()
    if verify:
        _install_torch_op_verifier()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = OUTPUT_DIR / "frame.png"
    frame_path.unlink(missing_ok=True)

    with Scene() as scene:
        with Off():
            Square(size=2.4).spawn()
            Sphere(radius=0.65).move(RIGHT * 1.1 + UP * 0.35).spawn()
            Sphere(radius=0.5).move(OUTWARD * 1.2 + RIGHT * 0.3).spawn()
        scene.save_frame(str(frame_path), video_settings=LD)

    _replay_notes()

    # Whether the fork was in the path at all. A wrapper that installed and
    # converted nothing leaves every argument on Taichi's host-staging path,
    # which is not slow-but-correct for the arena convention -- it is wrong
    # (DESIGN_mps_support.md 1.3b) -- so "0 converted launches" and "the frame
    # is wrong" is one finding rather than two.
    from algan.rendering.mps_zero_copy import STATS, installed, zero_copy_available

    print(
        f"\nzero copy        : available={zero_copy_available()} "
        f"installed={installed()}"
    )
    print(
        f"                   converted={STATS['converted_launches']} launches "
        f"({STATS['arguments']} args), "
        f"passthrough={STATS['passthrough_launches']}"
    )

    # Before the frame verdict, because every verdict below can return early
    # and the attribution table is the thing worth having when one does: a
    # frame that failed to draw is exactly the run whose op table you want.
    ops_clean = _report_torch_ops() if verify else True

    if not frame_path.exists() or frame_path.stat().st_size == 0:
        print("FAIL: no frame was written")
        return 1

    frame = np.asarray(Image.open(frame_path).convert("RGB")).astype(np.float64)
    print(f"frame            : {frame.shape} min {frame.min()} max {frame.max()}")
    if not np.isfinite(frame).all():
        print("FAIL: the frame is not finite")
        return 1
    if frame.max() == frame.min():
        print("FAIL: the frame is a single flat colour -- nothing was drawn")
        return 1
    # A frame of pure background would pass the test above on the vignette
    # alone, so ask for real ink: distinct values over a real share of it.
    if len(np.unique(frame.astype(np.uint8))) < 16:
        print("FAIL: the frame holds too few distinct values to be a render")
        return 1

    if not ops_clean:
        print("FAIL: the frame drew, but a torch op disagrees with the CPU")
        return 1

    print(f"OK: rendered {frame_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
