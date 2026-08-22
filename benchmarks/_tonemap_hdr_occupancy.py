"""How much of a real frame is actually above the display range?

The tonemap's justification is HDR: values above 1.0 have to be brought into
range, and clamping them looks worse than rolling them off. That justification
is only worth what the scenes actually contain, and the cost is paid on every
pixel -- so this measures the occupancy directly, on the repo's own dense
full-render scenes.

It intercepts the linear-HDR frame at the point the post stage hands it to the
tonemap (``_finalize_on_device``), before any curve or quantization, and
histograms it. Two numbers matter per scene: the fraction of channels above
1.0 (what the tonemap exists for) and the fraction in 0 < x <= 1.0 (what it
charges for that).

    <venv-python> benchmarks/_tonemap_hdr_occupancy.py                  # all scenes
    <venv-python> benchmarks/_tonemap_hdr_occupancy.py materials_and_lighting
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from algan import PREVIEW, SETTINGS, Scene
from algan.rendering.post_processing import post_process
from algan.scene_manager import SceneManager

REPO = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO / "tests" / "full_renders" / "scenes"
OUT_DIR = REPO / "algan_outputs" / "tonemap_check"


class Occupancy:
    """Running channel counts over every frame handed to the tonemap."""

    def __init__(self):
        self.total = 0
        self.zero = 0
        self.sdr = 0  # 0 < x <= 1.0: inside the display range already
        self.over = 0  # x > 1.0: the values the tonemap exists to handle
        self.over_1_05 = 0  # x > 1.05: over by more than quantization noise
        self.peak = 0.0
        self.frames = 0

    def observe(self, frame):
        # Channels 0-2 are the colour; 3 is glow and 4 (if present) is alpha,
        # neither of which the tonemap curve touches.
        rgb = frame[..., :3].float()
        self.total += rgb.numel()
        self.zero += int((rgb <= 0).sum())
        self.sdr += int(((rgb > 0) & (rgb <= 1.0)).sum())
        self.over += int((rgb > 1.0).sum())
        self.over_1_05 += int((rgb > 1.05).sum())
        self.peak = max(self.peak, float(rgb.max()))
        self.frames += frame.shape[0]

    def report(self, name):
        if not self.total:
            print(f"{name:>28}  (no frames observed)")
            return
        pct = 100.0 / self.total
        print(
            f"{name:>28} {self.frames:>7} {self.peak:>9.3f} "
            f"{self.zero * pct:>9.2f}% {self.sdr * pct:>9.2f}% "
            f"{self.over * pct:>9.4f}% {self.over_1_05 * pct:>10.4f}%"
        )


def _instrument(counter):
    """Wrap ``_finalize_on_device`` so every frame is counted on its way in."""
    original = post_process._finalize_on_device

    def wrapper(frame, *args, **kwargs):
        if frame.dtype != torch.uint8:
            counter.observe(frame)
        return original(frame, *args, **kwargs)

    post_process._finalize_on_device = wrapper
    return original


def _load_scene(scene_path):
    module_name = f"_algan_hdr_occupancy_{scene_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, scene_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)


def run(scene_path):
    counter = Occupancy()
    original = _instrument(counter)
    snapshot = SETTINGS.snapshot()
    cwd = Path.cwd()
    import os

    os.chdir(SCENES_DIR.parent)
    SceneManager.reset()
    try:
        with Scene() as scene:
            _load_scene(scene_path)
            scene.save_video(
                str(OUT_DIR / f"occupancy_{scene_path.stem}.mp4"),
                video_settings=PREVIEW,
                overwrite=True,
            )
    finally:
        post_process._finalize_on_device = original
        SETTINGS.restore(snapshot)
        SceneManager.reset()
        os.chdir(cwd)
    return counter


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    scenes = sorted(SCENES_DIR.glob("*.py"))
    if wanted:
        scenes = [p for p in scenes if p.stem in wanted]
    if not scenes:
        raise SystemExit(f"no scenes matched {wanted} in {SCENES_DIR}")

    print()
    print("Linear-HDR channel occupancy, sampled where the tonemap receives it")
    print(
        f"{'scene':>28} {'frames':>7} {'peak':>9} {'== 0':>10} "
        f"{'0 < x <= 1':>10} {'x > 1':>10} {'x > 1.05':>11}"
    )
    print("-" * 92)
    counters = []
    for scene_path in scenes:
        counter = run(scene_path)
        counter.report(scene_path.stem)
        counters.append(counter)

    total = Occupancy()
    for c in counters:
        total.total += c.total
        total.zero += c.zero
        total.sdr += c.sdr
        total.over += c.over
        total.over_1_05 += c.over_1_05
        total.frames += c.frames
        total.peak = max(total.peak, c.peak)
    print("-" * 92)
    total.report("ALL")
    if total.total:
        print()
        print(f"channels the tonemap exists for (> 1.0)   : {total.over}")
        print(f"channels it alters anyway (0 < x <= 1.0)  : {total.sdr}")
        if total.over:
            print(
                f"ratio                                     : "
                f"1 : {total.sdr / total.over:,.0f}"
            )


if __name__ == "__main__":
    main()
