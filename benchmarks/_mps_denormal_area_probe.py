"""Where a denormal fragment area goes on the render device, op by op and arm by arm.

``DESIGN_mps_support.md`` §4.10 records that on MPS the stroke split's torch
arm (``sheets._interleave_split`` -> ``sheets._stroke_sheets``) flushes a
denormal area to zero where its kernel arm (``sheet_compact_taichi.
split_merge_sheets``) keeps it. This finds the op that flushes and what every
arm of the compaction makes of such an area, in three parts:

1. **torch ops** the torch arm and the general path run on an area -- the
   gathers, ``+ 0.0``, ``clamp_min_(0.0)``, the scatters -- on the device,
   against the IEEE answer. The reference is taken on the CPU *before* Taichi
   initializes, because initializing it turns flush-to-zero on for the
   thread, and from then on the CPU's own add flushes too.
2. **Taichi kernels** on the device: a copy, ``x + 0.0``, ``max(x, 0)``, a
   float32 atomic add into zero (``sheet_band_reduce``'s), and a plain add into
   zero (``sheet_band_reduce_runs``').
3. **compact_sheets** on a device stream whose stroke pixels carry such areas:
   the split's kernel arm, its torch arm and the unsplit compaction, and the
   stroke-only shortcut against its own general path.

    python benchmarks/_mps_denormal_area_probe.py

Prints a line per case as it goes, and exits 0 whatever it finds: it is a
measurement, not a check.
"""

# No ``from __future__ import annotations``: the probe kernels below are
# annotated ``ti.types.ndarray()``, which Taichi evaluates at run time.

import os
import struct
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from algan.settings._startup import render_device  # noqa: E402

#: The bit patterns probed: the denormals of either sign, both zeros, the
#: smallest normal, ordinary areas, and NaNs.
PATTERNS = {
    "+den min": 0x00000001,
    "+den mid": 0x00400000,
    "+den max": 0x007FFFFF,
    "-den min": 0x80000001,
    "-den max": 0x807FFFFF,
    "+0": 0x00000000,
    "-0": 0x80000000,
    "+min normal": 0x00800000,
    "0.1": 0x3DCCCCCD,
    "1.0": 0x3F800000,
    "-1.0": 0xBF800000,
    "qNaN": 0x7FC00000,
    "sNaN": 0x7FA00000,
}


def _bits_tensor(device="cpu"):
    values = [b - (1 << 32) if b >= 1 << 31 else b for b in PATTERNS.values()]
    return torch.tensor(values, dtype=torch.int32).view(torch.float32).to(device)


def _hex(t):
    return [f"{int(v) & 0xFFFFFFFF:08x}" for v in t.cpu().view(torch.int32).tolist()]


def _ops():
    """Each op the torch arm or the general path applies to an area, as a
    function of a float32 tensor on some device.
    """
    n = len(PATTERNS)

    def idx(x):
        return torch.arange(n, device=x.device)

    def scatter_add(x):
        return torch.zeros(n, dtype=x.dtype, device=x.device).scatter_add_(0, idx(x), x)

    def index_add(x):
        return torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, idx(x), x)

    def index_copy(x):
        return torch.empty(n, dtype=x.dtype, device=x.device).index_copy_(0, idx(x), x)

    def bool_mask(x):
        return x[torch.ones(n, dtype=torch.bool, device=x.device)]

    return {
        "clone": lambda x: x.clone(),
        "index_select": lambda x: x.index_select(0, idx(x)),
        "advanced index x[i]": lambda x: x[idx(x)],
        "boolean mask x[m]": bool_mask,
        "index_copy_": index_copy,
        "x + 0.0": lambda x: x + 0.0,
        "x * 1.0": lambda x: x * 1.0,
        "x.clamp_min(0.0)": lambda x: x.clamp_min(0.0),
        "x.clone().clamp_min_(0.0)": lambda x: x.clone().clamp_min_(0.0),
        "(x + 0.0).clamp_min_(0.0) [torch arm]": lambda x: (x + 0.0).clamp_min_(0.0),
        "torch.maximum(x, 0)": lambda x: torch.maximum(x, torch.zeros_like(x)),
        "where(x < 0, 0, x)": lambda x: torch.where(x < 0, torch.zeros_like(x), x),
        "zeros.scatter_add_ [general path, torch arm]": scatter_add,
        "zeros.index_add_": index_add,
    }


def part_torch(device):
    print(f"== 1. torch ops on {device} against the CPU's IEEE answer")
    host = _bits_tensor()
    names = list(PATTERNS)
    for name, op in _ops().items():
        want = _hex(op(host.clone()))
        try:
            got = _hex(op(_bits_tensor(device)))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: RAISED {type(exc).__name__}: {exc}")
            continue
        bad = [
            f"{names[i]} {w}->{g}" for i, (w, g) in enumerate(zip(want, got)) if w != g
        ]
        print(
            f"  {name}: " + ("exact" if not bad else "DIFFERS: " + ", ".join(bad)),
            flush=True,
        )
    return host


def part_kernels(device):
    from algan.rendering.taichi_runtime import init_taichi
    from algan.taichi_compat import ti

    init_taichi()

    @ti.kernel
    def copy(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = x[i]

    @ti.kernel
    def add_zero(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = x[i] + 0.0

    @ti.kernel
    def max_zero(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = ti.max(x[i], 0.0)

    @ti.kernel
    def atomic_into_zero(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            ti.atomic_add(out[i], x[i])

    @ti.kernel
    def plain_into_zero(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = out[i] + x[i]

    print(f"== 2. Taichi kernels on {device}: the bits each writes, per input")
    n = len(PATTERNS)
    kernels = {
        "copy": copy,
        "x+0.0": add_zero,
        "max(x,0)": max_zero,
        "atomic 0+x": atomic_into_zero,
        "plain 0+x": plain_into_zero,
    }
    columns = {}
    for name, kernel in kernels.items():
        x = _bits_tensor(device)
        out = torch.zeros(n, dtype=torch.float32, device=device)
        try:
            kernel(x, out, n)
            columns[name] = _hex(out)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: RAISED {type(exc).__name__}: {exc}")
    print("  input bits: " + ", ".join(f"{k}" for k in columns))
    for i, (pattern, bits) in enumerate(PATTERNS.items()):
        cells = ", ".join(f"{k} {v[i]}" for k, v in columns.items())
        print(f"  {pattern} ({bits:08x}): {cells}", flush=True)


def _stroke_rows(coverage):
    """Rows of the stream's stroke pixels' fragments, and for each the row of
    its sheet: a stroke pixel compacts one fragment to one sheet, in stream
    order.
    """
    n = int(coverage["num_fragments"])
    ref = coverage["frag_ref"][:n].cpu()
    offsets = coverage["run_offsets"].to(torch.int64).cpu()
    counts = offsets[1:] - offsets[:-1]
    pixel = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    tri_pixel = torch.zeros(counts.numel(), dtype=torch.bool)
    tri_pixel[pixel[ref >= 0]] = True
    rows = (~tri_pixel[pixel]).nonzero().squeeze(1)
    return rows, pixel


#: The patterns the general path can take. Its dominant-fragment search starts
#: from +0.0 (``band_stats_rep_orig``'s ``cov >= cmax``), so a band whose only
#: area is negative or a NaN has no dominant fragment and the compaction
#: indexes past its stream; ``test_stroke_sheets._stroke_stream`` keeps its
#: areas non-negative for the same reason.
GENERAL_SAFE = ("+den min", "+den mid", "+den max", "+0", "-0", "+min normal", "0.1")


def _inject(stream, rows, names=None):
    coverage = dict(stream[0])
    names = list(PATTERNS) if names is None else list(names)
    values = [PATTERNS[k] for k in names]
    values = [b - (1 << 32) if b >= 1 << 31 else b for b in values]
    picks = rows[: len(values)]
    cov = coverage["frag_cov"].clone()
    cov[picks.to(cov.device)] = (
        torch.tensor(values[: picks.numel()], dtype=torch.int32)
        .view(torch.float32)
        .to(cov.device)
    )
    coverage["frag_cov"] = cov
    return (coverage, *stream[1:]), picks


def _sheet_rows(out, picks, pixel, coverage):
    """The rows of the sheets the picked stroke fragments make."""
    offsets = coverage["run_offsets"].to(torch.int64).cpu()
    sheet_offsets = out["sheet_offsets"].to(torch.int64).cpu()
    p = pixel[picks]
    return sheet_offsets[p] + (picks - offsets[p])


def _sheet_bits(out, picks, pixel, coverage):
    """The ``sheet_wgt`` bits of the sheets the picked stroke fragments make."""
    return _hex(out["sheet_wgt"].cpu()[_sheet_rows(out, picks, pixel, coverage)])


def _whole_output_diff(got, want, rows):
    """Everything else two arms returned: whether the sheet count and CSR
    agree, and per field how many rows differ outside ``rows``.
    """
    if int(got["num_sheets"]) != int(want["num_sheets"]):
        return f"num_sheets {int(got['num_sheets'])} vs {int(want['num_sheets'])}"
    inside = torch.zeros(int(want["num_sheets"]), dtype=torch.bool)
    inside[rows] = True
    parts = []
    for name, w in want.items():
        if not torch.is_tensor(w) or w.dim() == 0:
            continue
        g = got[name].cpu()
        w = w.cpu()
        d = (
            g.view(torch.int32) != w.view(torch.int32)
            if w.is_floating_point()
            else g != w
        )
        d = d.reshape(d.shape[0], -1).any(1)
        if name == "sheet_offsets":
            if bool(d.any()):
                parts.append(f"sheet_offsets {int(d.sum())} entries")
            continue
        out_rows = int((d & ~inside).sum())
        in_rows = int((d & inside).sum())
        if out_rows or in_rows:
            parts.append(f"{name} {in_rows} picked + {out_rows} other rows")
    return "same sheets, same order; " + (
        "differ only in: " + ", ".join(parts) if parts else "bit-identical"
    )


def part_compaction(device):
    import algan.animation_timeline.timeline as tl
    import algan.rendering.raytracing.sheets as sheets
    from tests.unit_tests import test_stroke_sheets as ts

    print(f"== 3. compact_sheets arms on {device} tensors")
    names = list(PATTERNS)
    tl._opt_disabled("strokepixels")  # initialise the set from the environment
    base = tl._OPT_DISABLED - {"strokepixels"}
    flags = {"shade_split": True, "positioned_depth": True, "sample_depth": True}

    def arm(name, stream):
        tl._OPT_DISABLED = base | {"strokepixels"} if name == "unsplit" else base
        sheets.sheet_split_kernel = name == "kernel"
        sheets.sheet_split_kernel_mps = name == "kernel"
        return ts._run(stream, shortcut=False, resolve_only=True, **flags)

    plain = ts._mixed_stream(0, device, True, False, tri_share=0.08)
    rows, pixel = _stroke_rows(plain[0])
    # Every pattern through the split's two arms, which never send a stroke
    # pixel to the general path; the non-negative ones through all three.
    for patterns, arms in ((names, ("kernel", "torch")), (GENERAL_SAFE, ("unsplit",))):
        stream, picks = _inject(plain, rows, patterns)
        results = {}
        outs = {}
        for name in arms if arms != ("unsplit",) else ("kernel", "torch", "unsplit"):
            try:
                outs[name] = arm(name, stream)
                results[name] = _sheet_bits(outs[name], picks, pixel, stream[0])
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} arm RAISED {type(exc).__name__}: {exc}")
        for i, pattern in enumerate(list(patterns)[: picks.numel()]):
            cells = ", ".join(f"{a} {bits[i]}" for a, bits in results.items())
            print(f"  split, stroke area {pattern}: {cells}", flush=True)
        if "torch" in outs:
            picked = _sheet_rows(outs["torch"], picks, pixel, stream[0])
            for name in ("kernel", "unsplit"):
                if name in outs:
                    print(
                        f"  {name} arm against the torch arm, all of "
                        f"{int(outs['torch']['num_sheets'])} sheets: "
                        + _whole_output_diff(outs[name], outs["torch"], picked),
                        flush=True,
                    )
    tl._OPT_DISABLED = base
    sheets.sheet_split_kernel = True
    sheets.sheet_split_kernel_mps = False
    names = list(GENERAL_SAFE)

    # The stroke-only shortcut against the general path it short-circuits.
    sstream = ts._stroke_stream(3, device)
    srows = torch.arange(int(sstream[0]["num_fragments"]))
    sstream, spicks = _inject(sstream, srows, GENERAL_SAFE)
    spixel = torch.repeat_interleave(
        torch.arange(int(sstream[0]["num_covered"])),
        (
            sstream[0]["run_offsets"][1:].to(torch.int64)
            - sstream[0]["run_offsets"][:-1].to(torch.int64)
        ).cpu(),
    )
    out = {}
    for name, shortcut in (("shortcut", True), ("general", False)):
        res = ts._run(sstream, shortcut=shortcut, resolve_only=True, shade_split=True)
        out[name] = _sheet_bits(res, spicks, spixel, sstream[0])
    for i, pattern in enumerate(names[: spicks.numel()]):
        cells = ", ".join(f"{a} {bits[i]}" for a, bits in out.items())
        print(f"  stroke-only stream, area {pattern}: {cells}", flush=True)


def main() -> int:
    device = render_device()
    print(
        f"device {device.type}, torch {torch.__version__}, "
        f"float32 smallest denormal {struct.unpack('<f', struct.pack('<I', 1))[0]!r}"
    )
    part_torch(device)
    part_kernels(device)
    part_compaction(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
