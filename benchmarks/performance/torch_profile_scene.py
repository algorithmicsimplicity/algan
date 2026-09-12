"""torch.profiler capture (CPU + CUDA) of a short warm render of a workload scene.

    python benchmarks/performance/torch_profile_scene.py explainer --quality UHD --frames 12
    python benchmarks/performance/torch_profile_scene.py graphics --quality UHD --frames 6

The stage profiler (``profile_scene``) syncs at stage boundaries, so it says
how long ``compact_sheets`` took and not what it did; this says what it did.
One warm-up render compiles the kernels and fills the memory model, then the
second render is captured and reported three ways:

* **by CUDA time** -- which device ops (torch kernels and Taichi launches)
  the GPU actually spent its time on;
* **by CPU self time** -- which host-side calls the render thread spent its
  time issuing; on a launch-bound chain this is the table that matters;
* **the sync census** -- every host<->device synchronisation the capture saw
  (``cudaStreamSynchronize``, ``cudaDeviceSynchronize``, ``Memcpy DtoH``)
  with counts, which is the number of times the queue was drained per chunk.

The scene functions are the workload benchmarks' (``explainer_scene.scene``,
``graphics_scene.scene``), unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import algan  # noqa: E402
from algan import Scene  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

SCENES = {"explainer": "explainer_scene", "graphics": "graphics_scene"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=sorted(SCENES))
    parser.add_argument("--quality", default="UHD")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--rows", type=int, default=45)
    args = parser.parse_args(argv)

    module = __import__(SCENES[args.scene])
    settings = getattr(algan, args.quality)
    seconds = args.frames / settings.frames_per_second
    ffmpeg = ["-crf", "17", "-preset", "ultrafast"]

    def render(tag):
        SceneManager.reset()
        module.scene(seconds)
        return Scene.save_video(
            os.path.join("algan_outputs", "torch_profile", f"{args.scene}_{tag}.mp4"),
            settings,
            overwrite=True,
            reset=True,
            ffmpeg_params=ffmpeg,
        )

    print(
        f"[tp] warm-up render: {args.scene} {args.quality} {args.frames} frames",
        flush=True,
    )
    render("warm")
    print("[tp] profiled render", flush=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        result = render("prof")
    print(f"[tp] profiled render took {result.walltime_seconds:.2f}s wall", flush=True)

    ka = prof.key_averages()
    width = 70
    if torch.cuda.is_available():
        print("================ BY CUDA TIME ================", flush=True)
        print(
            ka.table(
                sort_by="cuda_time_total",
                row_limit=args.rows,
                max_name_column_width=width,
            )
        )
    print("================ BY CPU SELF TIME =============", flush=True)
    print(
        ka.table(
            sort_by="self_cpu_time_total",
            row_limit=args.rows,
            max_name_column_width=width,
        )
    )
    print("================ SYNC CENSUS =================", flush=True)
    total_cpu = sum(e.self_cpu_time_total for e in ka)
    for e in sorted(ka, key=lambda e: -e.count):
        name = e.key
        if any(
            s in name
            for s in (
                "Synchronize",
                "Memcpy DtoH",
                "Memcpy HtoD",
                "aten::item",
                "aten::_local_scalar_dense",
                "aten::nonzero",
                "aten::unique_consecutive",
                "aten::_unique2",
                "aten::sort",
                "aten::argsort",
            )
        ):
            print(
                f"  {name[:60]:<60} count {e.count:>7}  cpu total {e.cpu_time_total / 1e3:9.1f} ms"
                f"  ({100.0 * e.self_cpu_time_total / max(1.0, total_cpu):5.1f}% of self cpu)"
            )
    n_kernels = sum(
        e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA
    )
    print(f"  device-side records: {n_kernels}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
