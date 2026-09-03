"""The smallest thing worth running on a remote GPU: does a render happen?

Both GPU harnesses (`agent_guidance/gpu_harnesses.md`) point at this to prove
their plumbing before anything expensive is launched through them. It renders a
handful of frames of one moving square, so what it exercises is the whole path
-- import, device selection, Taichi kernel compilation, the render loop, the
encoder -- and not the renderer's interesting parts.

It says which device it resolved, in the clear, because that is what a broken
GPU harness gets wrong: an arm that quietly fell back to the CPU produces a
green run and a plausible number.

    uv run python scripts/gpu_smoke.py [--quality PREVIEW] [--runs 2]

With `--runs 2` the second render is the warm one; the first pays the Taichi
JIT, which on a cold cache is tens of seconds and is not a measurement.
"""

from __future__ import annotations

import argparse
import os
import time

# Never measured inside a warm daemon: it keeps adaptive renderer state (the
# memory model's batch-size fit) across runs, so a timing would be taken
# against whatever ran before it -- and everything above `import algan` would
# run twice. Set before the import, which is why it is at module scope.
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", default="PREVIEW")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)

    import algan
    from algan.settings import _startup

    print(f"algan {algan.__version__}", flush=True)
    print(f"ALGAN_DEVICE {_startup.render_device().type}", flush=True)

    import torch

    print(f"TORCH {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    if getattr(torch.backends, "mps", None) is not None:
        print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)

    from algan import OUT, RIGHT, Scene, Square, Sync

    quality = getattr(algan, args.quality)

    # Authored once and rendered `--runs` times: `save_video` leaves the Scene
    # exactly as authored, so re-authoring per run would stack squares and make
    # the runs different scenes rather than a cold/warm pair.
    square = Square().spawn()
    with Sync():
        square.move(RIGHT)
        square.rotate(90, OUT)

    for run in range(1, args.runs + 1):
        started = time.time()
        result = Scene.save_video("gpu_smoke", quality)
        seconds = time.time() - started
        label = "cold" if run == 1 else "warm (steady state)"
        print(
            f"RUN {run} ({label}): {seconds:.2f} s -> {result.output_path}",
            flush=True,
        )

    print("SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
