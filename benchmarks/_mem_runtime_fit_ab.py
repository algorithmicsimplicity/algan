"""Can a runtime two-point fit replace the offline calibration tables?

The proposal: render a batch's first two frames, measure the arena's peak at
n=1 and n=2, solve ``peak = a + b*n``, then size the rest of the batch from
``n = (M - a) / b``. No corpus, no route keys, no coefficient fitting -- and it
captures any new rendering code for free.

This measures whether that holds up, on two questions that decide it:

1. **Is the peak actually affine in the frame count?** If it is, two points
   determine it and points 3/5/8 will land exactly. If it is not, the fit is
   wrong in a direction that matters.
2. **Does ``wavefront_tile_auto`` break the measurement?** The wavefront tile
   sizes itself from whatever arena is free, so it is *elastic*: measuring a
   peak that includes it may just measure the arena. Every scene is run with
   the toggle both ways.

    .venv/Scripts/python.exe benchmarks/_mem_runtime_fit_ab.py
    .venv/Scripts/python.exe benchmarks/_mem_runtime_fit_ab.py --scenes shapes
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

FRAME_COUNTS = (1, 2, 3, 5, 8)


def measure_scene(scene_func, name, frame_counts, tile_auto):
    """Peak arena bytes for a render chunk of each frame count."""
    from algan.rendering.post_processing import post_process as pp
    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.raytracing import tracer as rtr
    from algan.scene import Scene
    from algan.scene_manager import SceneManager
    from algan.settings import LD, SETTINGS

    peaks = {}
    original_post = pp.post_process_frames
    saved_auto = rt_settings.wavefront_tile_auto
    saved_batch = SETTINGS.computing.max_animation_batch_size

    for frames in frame_counts:
        observed = []

        def _capturing_post(memory, *args, observed=observed, **kwargs):
            result = original_post(memory, *args, **kwargs)
            # Post-processing is the chunk's last arena consumer, so the
            # high-water mark is complete by the time it returns.
            observed.append(int(memory.max_pointer))
            return result

        rt_settings.wavefront_tile_auto = tile_auto
        # Force one batch (and therefore one render chunk) per frame count.
        SETTINGS.computing.set(max_animation_batch_size=frames)
        pp.post_process_frames = _capturing_post
        rtr.post_process_frames = _capturing_post
        try:
            SceneManager.reset()
            scene_func()
            Scene.save_video(f"_fit_{name}_{frames}", LD, reset=True, overwrite=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name} n={frames}] failed: {type(exc).__name__}: {exc}")
            continue
        finally:
            pp.post_process_frames = original_post
            rtr.post_process_frames = original_post
            rt_settings.wavefront_tile_auto = saved_auto
            SETTINGS.computing.set(max_animation_batch_size=saved_batch)
        if observed:
            peaks[frames] = max(observed)
    return peaks


def report(name, peaks, label):
    if 1 not in peaks or 2 not in peaks:
        print(f"  {label:<12} insufficient data {sorted(peaks)}")
        return None

    # Two-point solve, exactly as the runtime scheme would do it.
    b = peaks[2] - peaks[1]
    a = peaks[1] - b
    print(f"  {label:<12} a={a / 1e6:10.3f} MB  b={b / 1e6:8.3f} MB/frame")

    worst = 0.0
    for frames in sorted(peaks):
        if frames <= 2:
            continue
        predicted = a + b * frames
        actual = peaks[frames]
        error = (predicted - actual) / max(1, actual)
        worst = min(worst, error) if error < 0 else worst
        flag = "UNDER" if predicted < actual else "ok"
        print(
            f"    n={frames:<3} predicted {predicted / 1e6:10.3f} MB  "
            f"actual {actual / 1e6:10.3f} MB  {error:+7.2%}  {flag}"
        )
    # Elasticity check: a scheme that measures a self-sizing allocation reads
    # back the arena rather than the workload.
    if b <= 0:
        print(
            "    b <= 0: peak does not grow with frames -- the measurement "
            "is dominated by an elastic (self-sizing) allocation."
        )
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    from algan.utils.calibration_corpus import RENDER_SCENES

    selected = [
        entry
        for entry in RENDER_SCENES
        if (args.scenes is None or entry[0] in args.scenes) and not entry[2]
    ]  # default settings only

    with torch.inference_mode():
        for name, scene_func, _overrides in selected:
            print(f"\n=== {name} ===")
            for tile_auto in (True, False):
                label = "TILE_AUTO" if tile_auto else "tile fixed"
                peaks = measure_scene(scene_func, name, FRAME_COUNTS, tile_auto)
                report(name, peaks, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
