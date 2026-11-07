"""Replay noise in the compaction's float32 area sums, and what fixing it costs.

In MPS-friendly mode (``mps_compat.accumulate_dtype`` is float32: Metal has no
float64) the compaction's band area sums were float32 atomics
(``sheet_compact_taichi.sheet_band_reduce``), whose order the GPU does not fix,
so a band of three or more fragments could round differently from one launch
to the next. ``sheets.sheet_area_fixed_order`` sums each band in stream order
instead (``sheet_band_reduce_runs``). This renders one frame, captures what the
render handed ``compact_sheets``, and for each arm (``atomic``: the unordered
atomics; ``fixed``: the ordered walk):

* replays the compaction on the captured input ``--replays`` times and
  compares every replay with the first, bit for bit: how many sheets differ,
  in which fields, and how many fragments each of those sheets sums;
* times the compaction, the arms interleaved and the device synchronized
  around every call, so a reading is one compaction's whole latency;
* renders the frame ``--renders`` times warm, the arms interleaved, and
  compares every frame with its arm's first and with the other arm's:
  whether any final pixel moves.

``--inject-denormal`` also renders the frame once with every sheet whose
weight is exactly +0.0 given the smallest denormal instead, and compares it
with a plain render: whether the resolve can tell the two apart.

    python benchmarks/_sheet_area_replay.py --scene explainer --quality HD
    python benchmarks/_sheet_area_replay.py --scene solids --quality HD

It prints a line per measurement as it goes, so a run cut short keeps what it
measured (``agent_guidance/gpu_harnesses.md``).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance")
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import algan  # noqa: E402
import algan.rendering.raytracing.sheets as sheets  # noqa: E402
from algan.rendering.mps_compat import accumulate_dtype, mps_friendly  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402

ARMS = {"atomic": False, "fixed": True}
#: The fields of ``merged`` that ``compact_sheets`` reads.
MERGED_FIELDS = ("tri_obj", "tri_pos", "tri_norm", "tri_closed")

_real_compact = sheets.compact_sheets
_captured: list[dict] = []
_capturing = False
#: While set, every top-level compaction's zero weights become this bit
#: pattern (``--inject-denormal``); the count of sheets so changed is kept.
_inject: dict | None = None


def _sync():
    device = render_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _clone(value):
    return value.clone() if torch.is_tensor(value) else value


def _hooked_compact(coverage, merged, *args, **kwargs):
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
    out = _real_compact(coverage, merged, *args, **kwargs)
    if _inject is not None and top and out is not None:
        wgt = out["sheet_wgt"]
        zero = wgt.view(torch.int32) == 0
        _inject["sheets"] += int(zero.sum())
        wgt.view(torch.int32).masked_fill_(zero, _inject["bits"])
    return out


def _replay(capture, **override):
    kwargs = dict(capture["kwargs"])
    kwargs["memory"] = None
    kwargs.update(override)
    return _real_compact(
        dict(capture["coverage"]), capture["merged"], *capture["args"], **kwargs
    )


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _differing_rows(got, want):
    """Per field, the rows whose bits differ; ``None`` where shapes differ."""
    rows = {}
    for field, w in want.items():
        if not torch.is_tensor(w):
            if got.get(field) != w:
                rows[field] = None
            continue
        g = got[field]
        if g.shape != w.shape or g.dtype != w.dtype:
            rows[field] = None
            continue
        diff = _bits(g) != _bits(w)
        if diff.dim() > 1:
            diff = diff.reshape(diff.shape[0], -1).any(1)
        if bool(diff.any()):
            rows[field] = diff.nonzero().squeeze(1).cpu()
    return rows


def _histogram(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return ", ".join(f"{k}: {counts[k]}" for k in sorted(counts))


def _nfrag(capture):
    """Fragments per sheet, in the order the resolve-only replay returns."""
    out = _replay(capture, resolve_only=False)
    return None if out is None else out["sheet_nfrag"].cpu()


def _stream_stats(capture, nfrag):
    cov = capture["coverage"]
    n = int(cov["num_fragments"])
    kinds = _float_kinds(cov["frag_cov"][:n].cpu().view(torch.int32))
    line = f"{n:,} fragments over {int(cov['num_covered']):,} covered pixels"
    if nfrag is not None:
        line += (
            f"; {nfrag.numel():,} sheets, {int((nfrag >= 3).sum()):,} of them "
            f"summing 3 or more fragments (most {int(nfrag.max())})"
        )
    line += "; areas: " + ", ".join(
        f"{k} {int(m.sum()):,}" for k, m in kinds.items() if bool(m.any())
    )
    return line


def _float_kinds(bits):
    bits = bits.to(torch.int64) & 0xFFFFFFFF
    exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    neg = (bits >> 31) == 1
    tiny = exponent == 0
    return {
        "+normal": ~neg & (exponent > 0) & (exponent < 0xFF),
        "-normal": neg & (exponent > 0) & (exponent < 0xFF),
        "+0": ~neg & tiny & (mantissa == 0),
        "-0": neg & tiny & (mantissa == 0),
        "+denormal": ~neg & tiny & (mantissa != 0),
        "-denormal": neg & tiny & (mantissa != 0),
        "nonfinite": exponent == 0xFF,
    }


def _render(path, quality, at):
    _sync()
    t0 = time.perf_counter()
    algan.Scene.save_frame(path, quality, at=at, overwrite=True)
    _sync()
    return time.perf_counter() - t0


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.int16)


def _frame_diff(a, b):
    d = np.abs(a - b).max(axis=-1)
    return int((d > 0).sum()), int(d.max())


def _solids(seconds):
    """Eight PN spheres, half of them translucent, a translucent torus and a
    translucent cube in front: a frame of triangle pixels, many of whose
    sheets sum several fragments, through the closed-shell ceiling and the
    rank pooling as well as the shading-class split.
    """
    from algan import (
        BLUE,
        DARKER_GRAY,
        GRAY_A,
        GREEN,
        MAROON_B,
        ORANGE,
        OUT,
        PURPLE_A,
        RED,
        RIGHT,
        SETTINGS,
        TEAL_A,
        UP,
        WHITE,
        YELLOW,
        Cube,
        Off,
        Scene,
        Sphere,
        Sync,
        Torus,
    )

    SETTINGS.raytracing.set(shadows=False)
    Scene.set_background(DARKER_GRAY)
    colors = [BLUE, RED, GREEN, YELLOW, PURPLE_A, TEAL_A, ORANGE, MAROON_B]
    mobs = []
    with Off():
        k = 0
        for x in (-4.8, -1.6, 1.6, 4.8):
            for y in (-1.7, 1.7):
                sphere = Sphere(
                    radius=1.3, color=colors[k], opacity=0.55 if k % 2 else 1.0
                )
                mobs.append(sphere.move(RIGHT * x + UP * y))
                k += 1
        mobs.append(
            Torus(ring_radius=1.5, tube_radius=0.5, color=WHITE, opacity=0.7).move(
                OUT * 2.0
            )
        )
        mobs.append(Cube(size=1.6, color=GRAY_A, opacity=0.6).move(OUT + UP * 0.2))
    with Sync(runtime=seconds * 0.2):
        for mob in mobs:
            mob.spawn()
    with Sync(runtime=seconds * 0.8):
        for mob in mobs:
            mob.rotate(90, UP)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("explainer", "solids"), default="explainer")
    parser.add_argument("--quality", default="HD")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--at", type=float, default=None)
    parser.add_argument("--replays", type=int, default=6)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--renders", type=int, default=4)
    parser.add_argument("--inject-denormal", action="store_true")
    args = parser.parse_args(argv)

    global _capturing, _inject
    device = render_device()
    quality = getattr(algan, args.quality)
    at = args.at if args.at is not None else (4.0 if args.scene == "explainer" else 3.6)
    print(
        f"device {device.type}, torch {torch.__version__}, mps_friendly "
        f"{mps_friendly()}, areas summed in {accumulate_dtype()}, "
        f"{args.scene} {args.quality} frame at {at}s",
        flush=True,
    )
    sheets.compact_sheets = _hooked_compact
    if args.scene == "explainer":
        import explainer_scene

        explainer_scene.scene(args.seconds)
    else:
        _solids(args.seconds)
    out = tempfile.mkdtemp(prefix="area_replay_")

    # 1. One render, captured (cold: it compiles everything the frame needs).
    sheets.sheet_area_fixed_order = True
    _capturing = True
    try:
        seconds = _render(os.path.join(out, "capture.png"), quality, at)
    finally:
        _capturing = False
    print(f"first render (captured): {seconds:.2f} s", flush=True)
    if not _captured:
        print("nothing captured: the render made no compaction call")
        return 1
    nfrags = [_nfrag(c) for c in _captured]
    for k, capture in enumerate(_captured):
        print(f"  call {k}: {_stream_stats(capture, nfrags[k])}", flush=True)

    # 2. Replays, per arm, each against the arm's first.
    firsts = {}
    for arm, fixed in ARMS.items():
        sheets.sheet_area_fixed_order = fixed
        first = [_replay(c) for c in _captured]
        firsts[arm] = first
        total = sum(int(o["num_sheets"]) for o in first if o is not None)
        moved_any = set()
        for r in range(1, args.replays):
            again = [_replay(c) for c in _captured]
            parts = []
            for k, (g, w) in enumerate(zip(again, first)):
                if g is None or w is None:
                    continue
                rows = _differing_rows(g, w)
                if not rows:
                    continue
                for field, idx in rows.items():
                    if idx is None:
                        parts.append(f"call {k} {field}: shape/value differs")
                        continue
                    moved_any.update((k, int(i)) for i in idx)
                    sums = ""
                    if (
                        nfrags[k] is not None
                        and nfrags[k].numel() == w["sheet_wgt"].shape[0]
                    ):
                        sums = " summing " + _histogram(nfrags[k][idx].tolist())
                    parts.append(
                        f"call {k} {field}: {idx.numel()} sheets{sums} fragment(s)"
                    )
            print(
                f"{arm} arm, replay {r} against replay 0 ({total:,} sheets): "
                + ("bit-identical" if not parts else "DIFFERS: " + "; ".join(parts)),
                flush=True,
            )
        print(
            f"{arm} arm: {len(moved_any)} distinct sheet(s) moved over "
            f"{args.replays} replays",
            flush=True,
        )
    parts = []
    for k, (g, w) in enumerate(zip(firsts["fixed"], firsts["atomic"])):
        if g is None or w is None:
            continue
        for field, idx in _differing_rows(g, w).items():
            parts.append(f"call {k} {field}: {'?' if idx is None else idx.numel()}")
    print(
        "fixed arm against the atomic arm's replay 0: "
        + ("bit-identical" if not parts else "differs: " + "; ".join(parts)),
        flush=True,
    )
    del firsts

    # 3. The compaction alone, interleaved, fenced on both sides.
    times = {arm: [] for arm in ARMS}
    for rep in range(args.reps):
        order = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            sheets.sheet_area_fixed_order = ARMS[arm]
            _sync()
            t0 = time.perf_counter()
            for capture in _captured:
                _replay(capture)
            _sync()
            times[arm].append(time.perf_counter() - t0)
        print(
            f"compaction rep {rep}: "
            + ", ".join(f"{a} {times[a][-1] * 1e3:.1f} ms" for a in times),
            flush=True,
        )
    for arm, ts in times.items():
        print(
            f"compaction {arm:<7} median {statistics.median(ts) * 1e3:8.2f} ms  "
            f"min {min(ts) * 1e3:8.2f} ms  over {len(ts)} (all calls of the frame)",
            flush=True,
        )
    _captured.clear()

    # 4. Whole warm renders, interleaved.
    frames = {arm: [] for arm in ARMS}
    render_times = {arm: [] for arm in ARMS}
    for rep in range(args.renders):
        order = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            sheets.sheet_area_fixed_order = ARMS[arm]
            path = os.path.join(out, f"{arm}_{rep}.png")
            render_times[arm].append(_render(path, quality, at))
            frames[arm].append(_pixels(path))
        line = f"render rep {rep}: " + ", ".join(
            f"{a} {render_times[a][-1]:.3f} s" for a in render_times
        )
        for arm in ARMS:
            moved, worst = _frame_diff(frames[arm][-1], frames[arm][0])
            line += f"; {arm} against its first: {moved} pixels (max {worst})"
        print(line, flush=True)
    for arm, ts in render_times.items():
        print(
            f"render {arm:<7} median {statistics.median(ts):7.3f} s  "
            f"min {min(ts):7.3f} s  over {len(ts)}",
            flush=True,
        )
    moved, worst = _frame_diff(frames["fixed"][0], frames["atomic"][0])
    print(
        f"fixed arm's first frame against the atomic arm's: {moved} pixels "
        f"differ, max channel difference {worst}",
        flush=True,
    )

    # 5. Zero weights against the smallest denormal, through the resolve.
    if args.inject_denormal:
        sheets.sheet_area_fixed_order = True
        _inject = {"bits": 1, "sheets": 0}
        try:
            path = os.path.join(out, "denormal.png")
            _render(path, quality, at)
        finally:
            injected, _inject = _inject, None
        moved, worst = _frame_diff(_pixels(path), frames["fixed"][0])
        print(
            f"frame with {injected['sheets']:,} zero sheet weights made the "
            f"smallest denormal, against the plain frame: {moved} pixels differ, "
            f"max channel difference {worst}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
