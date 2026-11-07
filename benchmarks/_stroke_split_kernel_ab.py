"""The stroke split's kernel arm against its torch arm, on a real frame.

``sheets._split_stroke_pixels`` compacts a mixed fragment stream's triangle
pixels apart from its stroke pixels and interleaves the two sheet streams back
into pixel order. Under ``sheet_split_kernel`` its classification, gather and
interleave are three kernels (``sheet_compact_taichi.split_mixed_pixels``,
``split_gather_pixels``, ``split_merge_sheets``); on an Apple GPU they also
wait for ``sheet_split_kernel_mps``. Otherwise they are some twenty torch
passes.

This renders one frame of the explainer workload
(``benchmarks/performance/explainer_scene.py``: glyphs, plots and diagrams with
one ``Sphere`` and one ``Cube``, so most covered pixels hold strokes alone and
the compaction takes the split) and captures what the render handed
``compact_sheets``. On that input it then:

* checks that the kernel arm, the torch arm and the unsplit compaction
  (``ALGAN_OPT_DISABLE=strokepixels``) return the same sheets, bit for bit;
* checks each arm against a second replay of itself. Where the general path
  is not reproducible run to run, the arms can differ for no fault of the
  split's: MPS sums the areas in float32 atomics (``mps_compat.
  accumulate_dtype``), whose order a band of three or more fragments sees;
* replays the two split arms sharing one general-path result for the
  triangle pixels, after checking that the kernel arm handed the general path
  the same fragments, bit for bit. What can still differ is the split's own
  work: its classification, gather and interleave;
* where any of these differ, says which sheets, in what kind of pixel, and
  with which bits in each arm;
* times ``compact_sheets`` under each arm, interleaved, with the device
  synchronized before and after every call, so each reading is one
  compaction's whole latency;
* times whole warm renders of the frame under the two split arms, interleaved,
  and compares their pixels.

    python benchmarks/_stroke_split_kernel_ab.py --quality HD --reps 7

It prints a line per measurement as it goes, so a run cut short keeps what it
measured (``agent_guidance/gpu_harnesses.md``). An arm that raises is reported
with its traceback and left out of the rest, which is how this reproduces a
kernel arm that does not run on a device.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
import traceback

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance")
)

import explainer_scene  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import algan  # noqa: E402
import algan.animation_timeline.timeline as tl  # noqa: E402
import algan.rendering.raytracing.sheets as sheets  # noqa: E402
from algan.rendering.mps_compat import accumulate_dtype, mps_friendly  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402

ARMS = ("kernel", "torch", "unsplit")
#: The fields of ``merged`` that ``compact_sheets`` reads.
MERGED_FIELDS = ("tri_obj", "tri_pos", "tri_norm", "tri_closed")

_real_compact = sheets.compact_sheets
_captured: list[dict] = []
_capturing = False
_broken: dict[str, str] = {}
#: While set, the general-path calls ``_split_stroke_pixels`` makes for its
#: triangle pixels go through ``_pinned_sub``: in ``record`` mode it keeps each
#: call's input and result, and in ``reuse`` mode it checks each call's input
#: against the recorded one, bit for bit, and returns the recorded result.
_pin: dict | None = None


def _sync():
    device = render_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _set_arm(arm):
    """Select one arm: the split's kernels, its torch passes, or no split."""
    tl._opt_disabled("strokepixels")  # initialise the set from the environment
    base = tl._OPT_DISABLED - {"strokepixels"}
    tl._OPT_DISABLED = base | {"strokepixels"} if arm == "unsplit" else base
    sheets.sheet_split_kernel = arm == "kernel"
    sheets.sheet_split_kernel_mps = arm == "kernel"


def _clone(value):
    return value.clone() if torch.is_tensor(value) else value


def _capturing_compact(coverage, merged, *args, **kwargs):
    """``compact_sheets`` that keeps a copy of each top-level call's input,
    and hands the split's own general-path calls to ``_pinned_sub`` while
    ``_pin`` is set.
    """
    top = kwargs.get("_split", True)
    if _capturing and top:
        _captured.append(
            {
                "coverage": {k: _clone(v) for k, v in coverage.items()},
                "merged": {k: _clone(merged[k]) for k in MERGED_FIELDS if k in merged},
                "args": [_clone(a) for a in args],
                "kwargs": {
                    k: _clone(v) for k, v in kwargs.items() if k not in ("memory",)
                },
            }
        )
    if _pin is not None and not top:
        return _pinned_sub(coverage, merged, *args, **kwargs)
    return _real_compact(coverage, merged, *args, **kwargs)


def _pinned_sub(coverage, merged, *args, **kwargs):
    """The general path on a split's triangle pixels, recorded or replayed.

    A replayed call whose input differs from the recorded one in any field
    is noted in ``_pin["diffs"]`` and compacted afresh, so the arm's output
    stays its own.
    """
    i = _pin["calls"]
    _pin["calls"] += 1
    if _pin["mode"] == "record":
        _pin["inputs"].append({k: _clone(v) for k, v in coverage.items()})
        out = _real_compact(coverage, merged, *args, **kwargs)
        _pin["outputs"].append(
            None if out is None else {k: _clone(v) for k, v in out.items()}
        )
        return out
    want = _pin["inputs"][i] if i < len(_pin["inputs"]) else {}
    bad = sorted(
        k for k in set(want) | set(coverage) if not _same(coverage.get(k), want.get(k))
    )
    if bad:
        _pin["diffs"].append(f"call {i}: {', '.join(bad)}")
        return _real_compact(coverage, merged, *args, **kwargs)
    out = _pin["outputs"][i]
    return None if out is None else {k: _clone(v) for k, v in out.items()}


def _replay(capture, **override):
    kwargs = dict(capture["kwargs"])
    kwargs["memory"] = None
    kwargs.update(override)
    return _real_compact(
        dict(capture["coverage"]), capture["merged"], *capture["args"], **kwargs
    )


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _same(got, want):
    """Bit-for-bit equality of two entries of a coverage or sheet dict."""
    if torch.is_tensor(want):
        return (
            torch.is_tensor(got)
            and got.dtype == want.dtype
            and got.shape == want.shape
            and torch.equal(_bits(got), _bits(want))
        )
    return not torch.is_tensor(got) and got == want


def _compare(name, got, want):
    """Bit-for-bit comparison of two compaction outputs; returns mismatches."""
    bad = []
    if set(got) != set(want):
        return [f"{name}: fields {sorted(set(got) ^ set(want))}"]
    for field, w in want.items():
        g = got[field]
        if torch.is_tensor(w):
            if g.dtype != w.dtype or g.shape != w.shape:
                bad.append(
                    f"{field} {g.dtype}{tuple(g.shape)} vs {w.dtype}{tuple(w.shape)}"
                )
                continue
            diff = _bits(g) != _bits(w)
            if diff.any():
                first = int(diff.reshape(-1).nonzero()[0])
                bad.append(
                    f"{field}: {int(diff.sum())}/{w.numel()} differ, first at {first}"
                )
        elif g != w:
            bad.append(f"{field}: {g!r} vs {w!r}")
    return bad


def _verdict(got, want):
    """One line on two lists of compaction outputs, call by call."""
    bad = [
        f"call {k}: {m}"
        for k, (g, w) in enumerate(zip(got, want))
        for m in _compare(f"call {k}", g, w)
    ]
    if len(got) != len(want):
        bad.append(f"{len(got)} calls against {len(want)}")
    return "bit-identical" if not bad else "DIFFERS: " + "; ".join(bad[:8])


def _sheet_nfrag(capture, want):
    """How many fragments each of the torch arm's sheets sums, or ``None``."""
    if "sheet_nfrag" in want:
        return want["sheet_nfrag"].cpu()
    _set_arm("torch")
    nfrag = _replay(capture, resolve_only=False).get("sheet_nfrag")
    if nfrag is None or nfrag.shape != want["sheet_wgt"].shape:
        return None
    return nfrag.cpu()


def _float_class(bits):
    """A float32's kind from its bits: what a flush, a sign or a NaN changes."""
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    sign = "-" if bits >> 31 else "+"
    if exponent == 0:
        return sign + ("0" if mantissa == 0 else "denormal")
    if exponent == 0xFF:
        return "nan" if mantissa else sign + "inf"
    return sign + "normal"


def _pixels_of(capture):
    """A capture's covered pixels: the CSR, each fragment's pixel, and which
    pixels hold a triangle fragment.
    """
    cov = capture["coverage"]
    n = int(cov["num_fragments"])
    nc = int(cov["num_covered"])
    ref = cov["frag_ref"][:n].cpu()
    run = cov["run_offsets"][: nc + 1].to(torch.int64).cpu()
    counts = run[1:] - run[:-1]
    pixel = torch.repeat_interleave(torch.arange(nc), counts)
    tri_pixel = torch.zeros(nc, dtype=torch.bool)
    tri_pixel[pixel[ref >= 0]] = True
    return run, pixel, tri_pixel


def _float_kinds(bits):
    """Masks of each float32 kind over a tensor of bit patterns."""
    bits = bits.to(torch.int64) & 0xFFFFFFFF
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    neg = (bits >> 31) == 1
    normal = (exponent > 0) & (exponent < 0xFF)
    tiny = exponent == 0
    return {
        "+normal": ~neg & normal,
        "-normal": neg & normal,
        "+0": ~neg & tiny & (mantissa == 0),
        "-0": neg & tiny & (mantissa == 0),
        "+denormal": ~neg & tiny & (mantissa != 0),
        "-denormal": neg & tiny & (mantissa != 0),
        "inf": (exponent == 0xFF) & (mantissa == 0),
        "nan": (exponent == 0xFF) & (mantissa != 0),
    }


def _area_kinds(capture):
    """The kinds of value the input fragments' areas take, in stroke pixels
    and in triangle pixels: which of the area rule's cases a real frame
    reaches (``sheet_compact_taichi.split_merge_sheets``).
    """
    cov = capture["coverage"]
    n = int(cov["num_fragments"])
    _, pixel, tri_pixel = _pixels_of(capture)
    in_tri = tri_pixel[pixel]
    kinds = _float_kinds(cov["frag_cov"][:n].cpu().view(torch.int32))
    lines = []
    for where, sel in (("stroke", ~in_tri), ("triangle", in_tri)):
        parts = [
            f"{name} {int((mask & sel).sum()):,}"
            for name, mask in kinds.items()
            if bool((mask & sel).any())
        ]
        lines.append(f"{where} pixels: " + ", ".join(parts))
    return "; ".join(lines)


def _explain(capture, outputs, k, nfrag=None, shown=6):
    """Where call ``k``'s float sheet fields differ between the arms.

    ``outputs`` maps a column (an arm, or a replay of one) to its outputs,
    call by call, and every column is compared with ``torch``. Per field: how
    many sheets differ, by the kind of pixel they sit in and, given the torch
    arm's ``sheet_nfrag`` as ``nfrag``, by how many fragments each sheet
    sums, and by the kind of value each column holds there; then the first
    few, with every column's bits and their distance in units in the last
    place from the torch arm's, the pixel's kind and fragments, and for a
    stroke pixel the area of the fragment the sheet copies.
    """
    want = outputs["torch"][k]
    cov = capture["coverage"]
    run, _, tri_pixel = _pixels_of(capture)
    frag_cov = cov["frag_cov"].cpu().view(torch.int32)
    offsets = want["sheet_offsets"].to(torch.int64).cpu()
    for field, w in want.items():
        if not torch.is_tensor(w) or w.dtype != torch.float32 or w.dim() != 1:
            continue
        bits = {"torch": w.cpu().view(torch.int32)}
        for arm, out in outputs.items():
            got = out[k].get(field) if arm != "torch" and len(out) > k else None
            if torch.is_tensor(got) and got.shape == w.shape:
                bits[arm] = got.cpu().view(torch.int32)
        differ = torch.zeros(w.shape[0], dtype=torch.bool)
        for b in bits.values():
            differ |= b != bits["torch"]
        rows = differ.nonzero().squeeze(1)
        if not rows.numel():
            continue
        pix = torch.searchsorted(offsets, rows, right=True) - 1
        tri = tri_pixel[pix]
        line = (
            f"  {field}: {rows.numel()} sheet(s) differ in some column, "
            f"{int(tri.sum())} in triangle pixels and {int((~tri).sum())} in "
            "stroke pixels"
        )
        if nfrag is not None:
            sums = {}
            for c in nfrag[rows].tolist():
                sums[c] = sums.get(c, 0) + 1
            line += "; they sum " + ", ".join(
                f"{c} fragment(s): {sums[c]}" for c in sorted(sums)
            )
        print(line)
        kinds = {}
        for o in rows.tolist():
            kind = ", ".join(
                f"{a} {_float_class(int(b[o]) & 0xFFFFFFFF)}" for a, b in bits.items()
            )
            kinds[kind] = kinds.get(kind, 0) + 1
        print("    by kind: " + "; ".join(f"{c}x ({kd})" for kd, c in kinds.items()))
        for o, p in list(zip(rows.tolist(), pix.tolist()))[:shown]:
            s, e = int(run[p]), int(run[p + 1])
            line = (
                f"    sheet {o}, pixel {p} "
                f"({'triangle' if tri_pixel[p] else 'stroke'}, {e - s} fragments"
            )
            if nfrag is not None:
                line += f", the sheet sums {int(nfrag[o])}"
            line += ")"
            if not tri_pixel[p]:
                i = s + o - int(offsets[p])
                line += f", fragment area 0x{int(frag_cov[i]) & 0xFFFFFFFF:08x}"
            ref_bits = int(bits["torch"][o])
            print(
                line
                + ": "
                + ", ".join(
                    f"{a} 0x{int(b[o]) & 0xFFFFFFFF:08x}"
                    + (f" ({int(b[o]) - ref_bits:+d})" if a != "torch" else "")
                    for a, b in bits.items()
                )
            )


def _stream_stats(capture):
    n = int(capture["coverage"]["num_fragments"])
    run, _, tri_pixel = _pixels_of(capture)
    counts = run[1:] - run[:-1]
    in_tri = int(counts[tri_pixel].sum())
    return (
        f"{n:,} fragments over {counts.numel():,} covered pixels; "
        f"{int(tri_pixel.sum()):,} pixels hold a triangle, with {in_tri:,} "
        f"fragments ({in_tri / max(n, 1):.1%})"
    )


def _render(path, quality, at):
    _sync()
    t0 = time.perf_counter()
    algan.Scene.save_frame(path, quality, at=at, overwrite=True)
    _sync()
    return time.perf_counter() - t0


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def _note_failure(arm, exc):
    if arm not in _broken:
        _broken[arm] = f"{type(exc).__name__}: {exc}"
        print(f"  arm {arm} RAISED -- left out of the rest:")
        traceback.print_exc()


def _stroke_area_rule():
    """The stroke-pixel area rule on the render device against the CPU, over
    values a real area can take and a few it cannot: as the float add the
    stroke-only shortcut takes, ``(cov + 0.0).clamp_min_(0.0)``, and as the
    bit rule both arms of the split take (``sheets._stroke_area_bits``). The
    host's add is IEEE here, because this runs before the first render
    initializes Taichi, which turns flush-to-zero on for the thread.
    """
    values = [0.0, -0.0, 1e-9, 1e-30, 1.4e-45, 5.9604645e-8, 3e-8, -1e-9, 0.5, 1.0]
    host = torch.tensor(values, dtype=torch.float32)
    rules = {
        "(cov + 0.0).clamp_min_(0.0)": lambda x: (x + 0.0).clamp_min_(0.0),
        "sheets._stroke_area_bits": sheets._stroke_area_bits,
    }
    for name, rule in rules.items():
        want = rule(host.clone())
        got = rule(host.to(render_device())).cpu()
        same = torch.equal(want.view(torch.int32), got.view(torch.int32))
        print(
            f"stroke area rule {name} on the device: {'exact' if same else 'DIFFERS'}"
        )
        if not same:
            for v, w, g in zip(values, want.tolist(), got.tolist()):
                if w != g or str(w) != str(g):
                    print(f"    {v!r}: cpu {w!r} device {g!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", default="HD")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument(
        "--at",
        type=float,
        default=4.0,
        help="frame time (default: act 3, all on screen)",
    )
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--renders", type=int, default=5)
    args = parser.parse_args(argv)

    global _capturing, _pin
    device = render_device()
    quality = getattr(algan, args.quality)
    print(
        f"device {device.type}, torch {torch.__version__}, mps_friendly "
        f"{mps_friendly()}, areas summed in {accumulate_dtype()}, "
        f"{args.quality} frame at {args.at}s"
    )
    _stroke_area_rule()
    sheets.compact_sheets = _capturing_compact
    explainer_scene.scene(args.seconds)
    out = tempfile.mkdtemp(prefix="split_ab_")
    frames = {}

    # 1. One render per split arm. The torch arm's render is captured, cold
    #    (it compiles everything); the kernel arm's then compiles its kernels.
    for arm in ("torch", "kernel"):
        _set_arm(arm)
        _capturing = arm == "torch"
        path = os.path.join(out, f"{arm}.png")
        try:
            seconds = _render(path, quality, args.at)
        except Exception as exc:  # noqa: BLE001
            _note_failure(arm, exc)
            continue
        finally:
            _capturing = False
        frames[arm] = _pixels(path)
        print(f"first render, {arm} arm: {seconds:.2f} s")
    if not _captured:
        print("nothing captured: the render made no compaction call")
        return 1
    split_calls = [c for c in _captured if c["coverage"].get("num_tri_fragments")]
    print(
        f"captured {len(_captured)} compaction call(s), {len(split_calls)} with triangles"
    )
    for k, capture in enumerate(_captured):
        print(f"  call {k}: {_stream_stats(capture)}")

    # 2. The three arms on the captured input, bit for bit.
    outputs = {arm: [] for arm in ARMS}
    for arm in ARMS:
        if arm in _broken:
            continue
        _set_arm(arm)
        try:
            for capture in _captured:
                outputs[arm].append(_replay(capture))
        except Exception as exc:  # noqa: BLE001
            _note_failure(arm, exc)
    ref = outputs["torch"]
    for arm in ("kernel", "unsplit"):
        if arm in _broken or not ref:
            continue
        print(
            f"{arm} arm against the torch arm, every call: {_verdict(outputs[arm], ref)}"
        )
    # 2b. Each arm against a second replay of itself: a difference the arms
    #     show only where an arm also differs from itself is not the split's.
    for arm in ARMS:
        if arm in _broken or not outputs[arm]:
            continue
        _set_arm(arm)
        again = outputs[f"{arm} again"] = [_replay(c) for c in _captured]
        print(
            f"{arm} arm against a second replay of itself: {_verdict(again, outputs[arm])}"
        )
    # 2c. The two split arms sharing one general-path result for the triangle
    #     pixels, the torch arm's, once the kernel arm has handed the general
    #     path the same fragments.
    if ref and "kernel" not in _broken:
        _pin = {"mode": "record", "calls": 0, "inputs": [], "outputs": [], "diffs": []}
        try:
            _set_arm("torch")
            shared = outputs["torch shared"] = [_replay(c) for c in _captured]
            _pin.update(mode="reuse", calls=0)
            _set_arm("kernel")
            outputs["kernel shared"] = [_replay(c) for c in _captured]
            handed, diffs = len(_pin["inputs"]), _pin["diffs"]
        finally:
            _pin = None
        print(
            f"triangle pixels the kernel arm handed the general path, against the "
            f"torch arm's ({handed} call(s)): "
            + ("bit-identical" if not diffs else "DIFFERS: " + "; ".join(diffs))
        )
        print(
            "kernel arm against the torch arm, the two sharing one general-path "
            f"result for the triangle pixels: {_verdict(outputs['kernel shared'], shared)}"
        )
    if ref:
        for k, capture in enumerate(_captured):
            print(f"  call {k} fragment areas: {_area_kinds(capture)}")
            _explain(capture, outputs, k, nfrag=_sheet_nfrag(capture, ref[k]))
    del outputs

    # 3. The compaction alone, interleaved, fenced on both sides.
    times = {arm: [] for arm in ARMS if arm not in _broken}
    for rep in range(args.reps):
        order = list(times) if rep % 2 == 0 else list(reversed(times))
        for arm in order:
            _set_arm(arm)
            _sync()
            t0 = time.perf_counter()
            for capture in _captured:
                _replay(capture)
            _sync()
            times[arm].append(time.perf_counter() - t0)
        print(
            f"compaction rep {rep}: "
            + ", ".join(f"{a} {times[a][-1] * 1e3:.1f} ms" for a in times)
        )
    for arm, ts in times.items():
        print(
            f"compaction {arm:<8} median {statistics.median(ts) * 1e3:8.1f} ms  "
            f"min {min(ts) * 1e3:8.1f} ms  over {len(ts)} (all calls of the frame)"
        )

    # 4. Whole warm renders under the two split arms, interleaved.
    sheets.compact_sheets = _real_compact
    render_times = {arm: [] for arm in ("kernel", "torch") if arm not in _broken}
    for rep in range(args.renders):
        order = list(render_times) if rep % 2 == 0 else list(reversed(render_times))
        for arm in order:
            _set_arm(arm)
            path = os.path.join(out, f"{arm}_{rep}.png")
            render_times[arm].append(_render(path, quality, args.at))
            if rep == 0:
                frames[arm + "_warm"] = _pixels(path)
        print(
            f"render rep {rep}: "
            + ", ".join(f"{a} {render_times[a][-1]:.3f} s" for a in render_times)
        )
    for arm, ts in render_times.items():
        print(
            f"render {arm:<8} median {statistics.median(ts):7.3f} s  "
            f"min {min(ts):7.3f} s  over {len(ts)}"
        )
    if "kernel_warm" in frames and "torch_warm" in frames:
        a = frames["kernel_warm"].astype(np.int16)
        b = frames["torch_warm"].astype(np.int16)
        moved = int((np.abs(a - b).max(axis=-1) > 0).sum())
        print(
            f"warm frames, kernel against torch arm: {moved} pixels differ, "
            f"max channel difference {int(np.abs(a - b).max())}"
        )
    if _broken:
        print("arms that raised: " + "; ".join(f"{a}: {m}" for a, m in _broken.items()))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
