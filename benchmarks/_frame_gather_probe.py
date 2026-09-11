"""What the frame-table lookup costs as one indexing op versus as a flat index.

``sheets._shade_class`` and ``sheets._prim_split_after`` each gather one entry
per FRAGMENT out of a small ``[frames, primitives]`` table. Master spells that
as a single advanced index, ``table[frame_rel, safe_ref]``. The ownership
refactor cannot, because advanced indexing has no destination argument, so
``array_ops.gather_frame_table`` builds the flat index explicitly -- widen,
add the time offset, wrap, multiply by the row length, add the column -- and
then calls ``index_select`` into the caller's array.

Both forms do the same lookup. The difference is that the second one walks the
whole fragment stream several more times, and the fragment stream is the
largest array in the compaction. This measures that difference on its own, with
no renderer state: the same table, the same indices, the same result, once each
way, plus a fused single-pass form for what a kernel would cost.

The fragment count dominates, so size it from a real chunk rather than a batch:
the ``compact_sheets`` calls in a 4K render cover three or four frames each.
Sizes come from parameters, so the footprint is checked before allocating.
"""

from __future__ import annotations

import argparse
import json
import time


def arms(table, frames, columns, out, index, time_start, device):
    """``{name: callable}`` for each spelling of the same gather."""
    import torch

    rows, width = table.shape

    def advanced_index():
        """Master's form: one fused operation, its own result."""
        return table[frames, columns]

    def flat_index():
        """The destination form: build the flat index, then index_select."""
        torch.add(frames, time_start, out=index)
        index.remainder_(rows)
        index.mul_(width).add_(columns)
        return torch.index_select(table.reshape(-1), 0, index, out=out)

    def fused_index():
        """The same flat index in one pass, standing in for a kernel."""
        torch.add(frames, time_start, out=index)
        # remainder/mul/add as one expression still allocates, but it is ONE
        # traversal of the stream instead of three in-place ones.
        return torch.index_select(
            table.reshape(-1),
            0,
            index.remainder_(rows).mul_(width).add_(columns),
            out=out,
        )

    return {
        "advanced_index": advanced_index,
        "flat_index": flat_index,
        "fused_index": fused_index,
    }


def measure(call, reps, device):
    import torch

    for _ in range(3):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tris", type=int, default=44444)
    parser.add_argument("--frags", type=int, default=8_000_000)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--max-gb", type=float, default=6.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if min(args.frames, args.tris, args.frags, args.reps) < 1:
        parser.error("frames, tris, frags and reps must all be positive")
    # table + three int64 fragment arrays + one int64 destination.
    wanted = args.frames * args.tris * 8 + args.frags * 8 * 4
    if wanted > args.max_gb * (1 << 30):
        parser.error(
            f"these sizes would allocate about {wanted / (1 << 30):.1f} GB, past "
            f"the {args.max_gb} GB ceiling; raise --max-gb deliberately"
        )

    import torch

    from algan import SETTINGS

    device = SETTINGS.computing.render_device
    n = args.frags
    table = torch.randint(
        0, 1 << 20, (args.frames, args.tris), dtype=torch.int64, device=device
    )
    frames = torch.randint(0, args.frames, (n,), dtype=torch.int64, device=device)
    columns = torch.randint(0, args.tris, (n,), dtype=torch.int64, device=device)
    out = torch.empty(n, dtype=torch.int64, device=device)
    index = torch.empty(n, dtype=torch.int64, device=device)

    calls = arms(table, frames, columns, out, index, 0, device)
    reference = calls["advanced_index"]()
    print(
        f"# {args.label or '(unlabelled)'}: device={device} frames={args.frames} "
        f"tris={args.tris} frags={n} reps={args.reps} torch={torch.__version__}",
        flush=True,
    )
    payload = {
        "benchmark": "frame_gather_probe",
        "label": args.label,
        "device": str(device),
        "frames": args.frames,
        "tris": args.tris,
        "frags": n,
        "arms": {},
    }
    baseline = None
    for name, call in calls.items():
        result = call()
        if not torch.equal(result, reference):
            raise SystemExit(f"{name} does not reproduce the advanced-index result")
        seconds = measure(call, args.reps, device)
        baseline = baseline if baseline is not None else seconds
        print(
            f"{name:16} {seconds * 1e3:8.3f} ms/call  {seconds / baseline:5.2f}x",
            flush=True,
        )
        payload["arms"][name] = {
            "ms": round(seconds * 1e3, 4),
            "ratio": round(seconds / baseline, 4),
        }
    print("")
    print("RESULTS " + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
