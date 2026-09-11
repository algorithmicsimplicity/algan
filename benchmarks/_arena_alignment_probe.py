"""What a bump allocator's alignment costs a bandwidth-bound elementwise op.

``ManualMemory`` aligns each allocation to its element size -- four bytes for
float32 -- while a torch caching-allocator block is aligned far more coarsely.
That difference is invisible on a CPU and invisible in an operation count, but
CUDA elementwise kernels choose a vectorised load width from the pointer's
alignment, so a four-byte-aligned buffer can be served by the scalar path where
a sixteen-byte-aligned one is not.

This measures that directly, on plain torch tensors and no Algan code: the same
elementwise work over the same bytes, once from a base pointer aligned to each
width and once from the same buffer offset by one element. Nothing here depends
on the renderer, so it says whether the mechanism exists on this device before
anything is concluded about a stage that allocates from the arena.

Sizes come from parameters, so the footprint is checked before allocating.
"""

from __future__ import annotations

import argparse
import json
import time

#: Widths to compare, in bytes. 4 is what the arena guarantees for float32; 16
#: is a float4; 128 and 512 are what the CUDA allocator's blocks tend to be.
WIDTHS = (4, 16, 128, 512)


def measure(tensor_a, tensor_b, out, reps, device):
    import torch

    for _ in range(3):
        torch.add(tensor_a, tensor_b, out=out)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        torch.add(tensor_a, tensor_b, out=out)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=23 * 44444 * 9)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--max-gb", type=float, default=4.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    if args.elements < 1 or args.reps < 1:
        parser.error("elements and reps must be positive")
    # Three float32 buffers, each with room for the largest offset tried.
    wanted = 3 * (args.elements + 256) * 4
    if wanted > args.max_gb * (1 << 30):
        parser.error(
            f"these sizes would allocate about {wanted / (1 << 30):.1f} GB, past "
            f"the {args.max_gb} GB ceiling; raise --max-gb deliberately"
        )

    import torch

    from algan import SETTINGS

    device = SETTINGS.computing.render_device
    n = args.elements
    # One oversized buffer per operand, so every arm reads the same bytes and
    # only the base pointer moves.
    buffers = [
        torch.rand(n + 256, dtype=torch.float32, device=device) for _ in range(3)
    ]
    gigabytes = 3 * n * 4 / 1e9

    print(
        f"# {args.label or '(unlabelled)'}: device={device} elements={n} "
        f"reps={args.reps} torch={torch.__version__} "
        f"({gigabytes:.2f} GB touched per call)",
        flush=True,
    )
    payload = {
        "benchmark": "arena_alignment_probe",
        "label": args.label,
        "device": str(device),
        "elements": n,
        "arms": {},
    }
    for width in WIDTHS:
        # Offset each view so its base pointer is aligned to `width` exactly and
        # not to the next one up, which is what an arbitrary bump offset gives.
        views = []
        for buffer in buffers:
            base = buffer.data_ptr()
            step = (-base) % width
            offset = step // 4
            while (buffer[offset:].data_ptr() % (2 * width)) == 0:
                offset += width // 4
            views.append(buffer[offset : offset + n])
        seconds = measure(views[0], views[1], views[2], args.reps, device)
        alignment = views[0].data_ptr() % 512
        print(
            f"aligned to {width:4d} B (ptr % 512 = {alignment:3d}): "
            f"{seconds * 1e3:8.3f} ms/call  {gigabytes / seconds:7.1f} GB/s",
            flush=True,
        )
        payload["arms"][str(width)] = {
            "ms": round(seconds * 1e3, 4),
            "gbps": round(gigabytes / seconds, 2),
        }
    print("")
    print("RESULTS " + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
