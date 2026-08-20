"""Sheet compaction over the six full-render scenes: the Phase-1 reality check.

``DESIGN_sheet_resolve.md`` Phase 1 scores the compaction on synthetic cases
(``_aa_run_gate_check --sheets``); this instrument runs the same compaction on
the SIX real scenes' fragment streams -- every batch of every frame, text and
PN dicing and materials included -- and reports what the sheet representation
looks like there:

* the compaction ratio S/F (what the per-sheet shading of P5 shrinks by),
* the per-pixel sheet-count distribution and its tail against the design's
  starting K = 24 (overflow policy sizing, ss4.6),
* the FUSED count (a sample bit contributed twice within one band -- the
  fill-rule partition violation; nonzero at curved folds even for a perfect
  band rule, because a fold is where projection stops being injective),
* the split-group count (benign; the band rule erring the right way), and
* host seconds inside ``compact_sheets`` per scene (indicative -- the real
  cost gate is Phase 2's, ss6.4).

Borrows ``_notch_scene_check``'s machinery -- the vendored-font registration
and the render-and-diff-against-baseline discipline -- by importing it and
swapping in a sheet-scoring spy, so this probe cannot silently render a
different scene than the suite does (ss7.19).

Run:  <venv-python> benchmarks/_sheet_scene_stats.py [--scenes solids text]
      [--rule prim|facing] [--band-c 4.0] [--res preview]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import _notch_scene_check as nsc  # noqa: E402
import torch  # noqa: E402

from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing.sheets import compact_sheets  # noqa: E402


class SheetStats:
    def __init__(self, name):
        self.name = name
        self.baseline = "(not diffed)"
        self.batches = 0
        self.frags = 0
        self.sheets = 0
        self.fused = 0
        self.groups = 0
        self.split_groups = 0
        self.covered = 0
        self.max_sheets = 0
        self.over_k = 0
        self.seconds = 0.0
        # Sheet-count histogram, clamped at 32.
        self.hist = torch.zeros(33, dtype=torch.int64)


RULE = "prim"
BAND_C = 4.0


class _SheetSpy:
    """Duck-typed replacement for ``_notch_scene_check._Spy``."""

    def __init__(self, stats):
        self.stats = stats
        self.original = rp.prepare_sparse_raster_coverage

    def __enter__(self):
        original = self.original
        stats = self.stats

        def spy(*args, **kwargs):
            coverage = original(*args, **kwargs)
            if coverage is None:
                return coverage

            def arg(name, pos):
                return kwargs[name] if name in kwargs else args[pos]

            t0 = time.perf_counter()
            stream = compact_sheets(
                coverage,
                arg("merged", 0),
                arg("cam_origin", 5),
                arg("pixel_world_scale", 9),
                int(arg("time_start", 11)),
                int(arg("width", 13)),
                int(arg("height", 14)),
                band_rule=RULE,
                band_c=BAND_C,
            )
            if stream is not None and stream["sheet_key"].is_cuda:
                torch.cuda.synchronize()
            stats.seconds += time.perf_counter() - t0
            if stream is None:
                return coverage
            stats.batches += 1
            stats.frags += int(coverage["num_fragments"])
            stats.sheets += stream["num_sheets"]
            stats.fused += int(stream["sheet_fused"].sum().item())
            stats.groups += stream["num_groups"]
            stats.split_groups += stream["num_split_groups"]
            counts = stream["sheet_offsets"].diff()
            stats.covered += int(counts.numel())
            stats.max_sheets = max(stats.max_sheets, int(counts.max().item()))
            stats.over_k += int((counts > 24).sum().item())
            stats.hist += torch.bincount(counts.clamp(max=32).cpu(), minlength=33)
            return coverage

        rp.prepare_sparse_raster_coverage = spy
        return self

    def __exit__(self, *exc):
        rp.prepare_sparse_raster_coverage = self.original
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--rule", choices=("prim", "facing"), default="prim")
    ap.add_argument("--band-c", type=float, default=4.0)
    ap.add_argument("--res", choices=sorted(nsc.RESOLUTIONS), default="preview")
    args = ap.parse_args()
    global RULE, BAND_C
    RULE = args.rule
    BAND_C = args.band_c
    quality = nsc.RESOLUTIONS[args.res]

    nsc._register_test_fonts()
    nsc._Spy = _SheetSpy  # _render_full_render_scene instantiates by name

    paths = sorted((nsc.FULL_RENDERS / "scenes").glob("*.py"))
    if args.scenes:
        paths = [p for p in paths if any(s in p.stem for s in args.scenes)]
    rows = []
    for path in paths:
        stats = SheetStats(path.stem)
        print(f"-- {path.stem} ...", flush=True)
        nsc._render_full_render_scene(path, stats, quality, None)
        rows.append(stats)

    print(f"\nband rule = {RULE} (c = {BAND_C})")
    head = (
        f"{'scene':26s} {'batches':>7s} {'frags':>11s} {'sheets':>11s} "
        f"{'S/F':>6s} {'fused':>8s} {'split':>8s} {'maxS':>5s} {'>K':>6s} "
        f"{'compact s':>10s}"
    )
    print(head)
    print("-" * len(head))
    for s in rows:
        sf = s.sheets / max(s.frags, 1)
        print(
            f"{s.name:26s} {s.batches:7d} {s.frags:11d} {s.sheets:11d} "
            f"{sf:6.2f} {s.fused:8d} {s.split_groups:8d} {s.max_sheets:5d} "
            f"{s.over_k:6d} {s.seconds:10.2f}"
        )
        tail = {k: int(s.hist[k].item()) for k in range(9, 33) if int(s.hist[k].item())}
        print(f"{'':26s} {s.baseline}")
        if tail:
            print(f"{'':26s} sheets/pixel tail (>=9): {tail}")
    print(
        "\n'fused' is the fill-rule partition violation inside one band; a\n"
        "curved fold makes some irreducible (projection stops being\n"
        "injective there), so compare rules rather than expecting zero.\n"
        "'>K' counts covered pixels holding more sheets than the design's\n"
        "starting K = 24."
    )


if __name__ == "__main__":
    main()
