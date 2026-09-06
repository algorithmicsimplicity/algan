"""Alternate warm-render optimizations on the exact UHD workload.

Run with the venv interpreter on CUDA. The first two passes warm both arms. No profiling
hooks or per-kernel synchronizations affect the steady-state measurements.
Use --verify separately to compare every raw output channel before encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import time
from pathlib import Path

os.environ["ALGAN_USE_DAEMON"] = "0"

import numpy as np
import torch

from algan import SETTINGS, Scene
from algan.render_loop import RenderLoopMixin
from algan.scene_manager import SceneManager
from algan.utils import profiling_utils
from algan.utils.memory_utils import peak_allocated, reset_peak_floor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tag", default="depth_ab")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--capture-depth", action="store_true")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("at least two runs are required to exercise both arms")
    captured = {}

    def capture(builder, quality, tag, **kwargs):
        captured.update(
            builder=builder, quality=quality, kwargs=kwargs["save_video_kwargs"]
        )

    original = profiling_utils.profile_scene
    profiling_utils.profile_scene = capture
    try:
        runpy.run_path(
            str(Path(__file__).with_name("nn_scene_UHD.py")), run_name="__main__"
        )
    finally:
        profiling_utils.profile_scene = original

    output = Path(__file__).parent / args.tag
    output.mkdir(exist_ok=True)
    if args.capture_depth:
        from algan.rendering.raytracing import sheets

        reference_depth = sheets._sample_depth_lose_reference

        def capture_depth(*values):
            torch.save(
                tuple(value.cpu() for value in values), output / "depth_inputs.pt"
            )
            sheets._sample_depth_lose_reference = reference_depth
            return reference_depth(*values)

        sheets._sample_depth_lose_reference = capture_depth
    results = []
    get_frames = RenderLoopMixin.get_frames
    note_cost = RenderLoopMixin._note_batch_cost
    preflight = RenderLoopMixin._prepared_batch_fits_render_arena
    current = {}

    def ab_note_cost(self, term, num_frames, needed_bytes, usable_bytes):
        if not args.depth_only and not current["on"] and usable_bytes <= 0:
            return
        return note_cost(self, term, num_frames, needed_bytes, usable_bytes)

    def count_preflight(self, *a, **kw):
        current["preflight_calls"] += 1
        return preflight(self, *a, **kw)

    RenderLoopMixin._note_batch_cost = ab_note_cost
    RenderLoopMixin._prepared_batch_fits_render_arena = count_preflight

    def checked_frames(self, *a, **kw):
        with (output / "reference.rgb").open("rb" if current["on"] else "wb") as stream:
            for batch in get_frames(self, *a, **kw):
                frames = np.asarray(batch)
                if current["on"]:
                    raw = stream.read(frames.nbytes)
                    assert len(raw) == frames.nbytes, "frame stream grew"
                    reference = np.frombuffer(raw, dtype=frames.dtype).reshape(
                        frames.shape
                    )
                    diff = np.abs(frames.astype(np.int16) - reference.astype(np.int16))
                    current["max_diff"] = max(
                        current["max_diff"], int(diff.max(initial=0))
                    )
                    current["different_channels"] += int(np.count_nonzero(diff))
                    current["channels"] += frames.size
                else:
                    stream.write(frames.tobytes())
                yield batch
            if current["on"]:
                assert not stream.read(1), "frame stream shrank"

    if args.verify:
        RenderLoopMixin.get_frames = checked_frames
    try:
        for i in range(args.runs):
            on = bool(i % 2)
            SETTINGS.raytracing.experimental.sheet_depth_reduce_kernel = on
            scene = SceneManager.reset()
            scene.set_video_settings(captured["quality"])
            captured["builder"]()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            reset_peak_floor()
            current.clear()
            current.update(
                on=on, max_diff=0, different_channels=0, channels=0, preflight_calls=0
            )
            started = time.perf_counter()
            result = Scene.save_video(
                str(output / f"run{i}.mp4"), reset=True, **captured["kwargs"]
            )
            torch.cuda.synchronize()
            record = {
                "run": i,
                "kernel": on,
                "warm": i >= 2,
                "seconds": time.perf_counter() - started,
                "peak_alloc_mb": peak_allocated() / 2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                "preflight_calls": current["preflight_calls"],
                "plan": str(result.render_plan),
            }
            if args.verify:
                record.update(current)
                assert current["max_diff"] <= 2, record
            results.append(record)
            print(json.dumps(record), flush=True)
            (output / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        RenderLoopMixin.get_frames = get_frames
        RenderLoopMixin._note_batch_cost = note_cost
        RenderLoopMixin._prepared_batch_fits_render_arena = preflight


if __name__ == "__main__":
    main()
