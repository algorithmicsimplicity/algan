"""Does Metal agree with the CPU on the integer arithmetic the sheet route uses?

Why this exists. The Metal render arena used to be clamped to 1 GiB
(``memory_utils.get_num_available_bytes``), which made **every** frame window
on **every** Mac one frame per render chunk. A one-frame chunk has all its
fragment pixel indices below ``width * height``, so the whole high half of the
sheet route's 64-bit fragment key is zero and its frame ordinal is always 0.
The first render taken with the clamp lifted -- a multi-frame chunk, for the
first time on that backend -- failed two different ways:

    nn_scene_PREVIEW  RuntimeError: Invalid buffer size: 6.45 GB
                      in sheets._shade_class, whose table is
                      [num_frames, num_triangles, 9] and is therefore a direct
                      readout of ``int(frame_rel.amax()) + 1``
    nn_scene_UHD      IndexError at tracer.py's ``gl_bounds[gl_frame + 1]``,
                      whose bounds come from a ``searchsorted`` over the
                      covered-pixel ordinals

Neither reproduces on the CPU at the same pool size and the same 50-frame
window, so the suspect is the arithmetic itself rather than the logic above it.
This script runs exactly the operations that path performs -- the packed
``(pixel << 32) | float_bits`` key, its two halves, the frame ordinal, the
searchsorted over frame edges, and the sorts the compaction takes -- on the
render device and on the CPU, and reports the first one that disagrees.

Run it on the Mac harness with ``ALGAN_RENDER_DEVICE=mps``; it needs no scene,
no kernels and no compile, so it costs seconds rather than a render.
"""

from __future__ import annotations

import torch

from algan.settings._startup import render_device

# Two frames of a 4K-ish grid: the point is that the pixel ordinal exceeds the
# 2**23 that a float32 can count and that the packed key needs the full 64.
WIDTH, HEIGHT = 3840, 2160
PPF = WIDTH * HEIGHT
NUM_FRAMES = 4
NUM_FRAGS = 200_000


def _reference(device):
    """The sheet route's key build, on one device, as plain torch."""
    g = torch.Generator(device="cpu").manual_seed(20260907)
    pixel = torch.randint(0, NUM_FRAMES * PPF, (NUM_FRAGS,), generator=g)
    depth = torch.rand(NUM_FRAGS, generator=g) * 40.0 + 0.5
    pixel = pixel.sort().values.to(device)
    depth = depth.to(device)

    # raster_taichi's frag_key: (pixel << 32) | bitcast(t).
    bits = depth.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    frag_key = (pixel.to(torch.int64) << 32) | bits

    out = {}
    out["key"] = frag_key
    out["pix"] = frag_key >> 32
    out["t"] = (frag_key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    out["frame_rel"] = out["pix"] // PPF
    out["frame_max"] = out["frame_rel"].amax()
    out["argsort_i64"] = torch.argsort(out["pix"], stable=True)
    out["argsort_i32"] = torch.argsort(out["pix"].to(torch.int32), stable=True)

    # _gloss_frame_bounds: searchsorted of per-frame edges into ascending
    # covered ordinals, in the int32 dtype the covered index is stored as.
    covered = torch.unique_consecutive(out["pix"]).to(torch.int32)
    edges = torch.arange(NUM_FRAMES + 1, device=device, dtype=torch.int32) * PPF
    out["covered_n"] = torch.tensor(covered.numel(), device=device)
    out["bounds"] = torch.searchsorted(covered, edges)
    return out


def main():
    device = render_device()
    print(f"render device: {device}")
    print(f"{NUM_FRAGS} fragments over {NUM_FRAMES} frames of {WIDTH}x{HEIGHT}")
    if device.type == "cpu":
        print("nothing to compare: the render device IS the reference")
        return 0

    got = _reference(device)
    want = _reference(torch.device("cpu"))

    failures = []
    for name in want:
        a = got[name].to("cpu")
        b = want[name]
        if a.dtype != b.dtype or a.shape != b.shape:
            failures.append(
                f"{name}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
            )
            continue
        if torch.equal(a, b):
            print(f"  ok       {name}")
            continue
        if a.numel() == 1:
            failures.append(f"{name}: device {a.item()} vs cpu {b.item()}")
            continue
        bad = (a != b).nonzero().flatten()
        first = int(bad[0])
        failures.append(
            f"{name}: {bad.numel()} of {a.numel()} differ; "
            f"first at {first}: device {a[first].item()} vs cpu {b[first].item()}"
        )
        print(f"  MISMATCH {name}")

    if failures:
        print("\nDISAGREEMENTS:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nevery operation agrees with the CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
