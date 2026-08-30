"""P9 frame-level A/B: run splitting vs the all-or-nothing group revert.

Renders the clashing scene twice IN ONE PROCESS -- once with
ALGAN_BEZIER_GROUP_RUNS=0 (the old wholesale revert) and once with =1 (run
splitting) -- flipping os.environ between the renders, which is legal because
the variable is read live at the point of use. The two videos are compared
frame by frame, pixel by pixel; the requirement is 0 differing channels.

Both arms must also see IDENTICAL batch windows, or the comparison would be
confounded by re-windowed state rather than by the change; the script wraps
get_batch_of_primitives and asserts it.

The scene is the constructed clash scene from _bezier_batchability.py (a
packed Text whose glyph primitives share a batch identifier with ordinary
circuits): the benchmark bezier_rendering scene measures ZERO group clashes,
so an A/B on it would prove nothing. On this scene the probe measures 240 of
246 circuit appearances reverted per window set under the old arm and 0 under
the new one.

Videos are written lossless (libx264rgb crf 0) so a zero diff is the
renderer's own output, not two equally-lossy encodes.

    .venv/bin/python benchmarks/_bezier_run_split_ab.py
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
os.environ["ALGAN_ADV_OPT"] = "0"
# A warm daemon keeps adaptive renderer state across runs; this A/B must run
# both arms in one process of its own.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "algan_outputs" / "_p9_ab"

from algan import *  # noqa: E402
from algan.mobs.bezier_circuit import BezierCircuitCubic  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


def clash_scene():
    with Off():
        txt = Text("Hello World").spawn()
        others = []
        for i in range(20):
            others.append(Circle(color=BLUE).move(RIGHT * (i - 10)).spawn())
            others.append(Square(color=RED).move(RIGHT * (i - 10) + UP * 2).spawn())
    with Sync(duration=1):
        txt.move(LEFT)
        for m in others:
            m.move(UP)


class ArmWatcher:
    """Counts where circuits were built, and records the batch windows."""

    def __init__(self):
        self.per_actor_builds = 0
        self.batched_actors = 0
        self.windows = []
        self._orig_grp = None
        self._orig_brb = None

    def attach(self):
        import algan.mobs.bezier_circuit as bez_mod

        watcher = self

        def counting_grp(self_mob):
            watcher.per_actor_builds += 1
            return watcher._orig_grp(self_mob)

        def counting_brb(actors, scene):
            watcher.batched_actors += len(actors)
            return watcher._orig_brb(actors, scene)

        self._orig_grp = BezierCircuitCubic.get_render_primitives
        self._orig_brb = bez_mod.build_render_primitives_batched
        BezierCircuitCubic.get_render_primitives = counting_grp
        bez_mod.build_render_primitives_batched = counting_brb

    def wrap_scene(self, scene):
        watcher = self
        orig = scene.get_batch_of_primitives

        def recording(start_ind, end_ind, actors, mem):
            watcher.windows.append((start_ind, end_ind))
            return orig(start_ind, end_ind, actors, mem)

        scene.get_batch_of_primitives = recording


def render_arm(flag_value, out_name):
    os.environ["ALGAN_BEZIER_GROUP_RUNS"] = flag_value
    watcher = ArmWatcher()
    watcher.attach()
    SceneManager.reset()
    with Scene() as scene:
        watcher.wrap_scene(scene)
        clash_scene()
        result = scene.save_video(
            str(OUT_DIR / out_name),
            video_settings=PREVIEW,
            overwrite=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )
    windows_text = ", ".join(f"{a}:{b}" for a, b in watcher.windows)
    print(
        f"arm ALGAN_BEZIER_GROUP_RUNS={flag_value}: "
        f"{watcher.per_actor_builds} per-actor circuit builds, "
        f"{watcher.batched_actors} circuits merged batched; "
        f"windows [{windows_text}]"
    )
    return watcher.windows, result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Warm-up render first, discarded: the first render on a fresh process
    # populates glyph/SVG caches whose output differs from every later run
    # (see tests/README.md), and an arm must not be the one that pays it.
    render_arm("1", "_warmup.mp4")
    old_windows, _ = render_arm("0", "runs_off.mp4")
    new_windows, _ = render_arm("1", "runs_on.mp4")
    if old_windows != new_windows:
        print(
            "BATCH WINDOWS DIFFER BETWEEN ARMS -- the comparison would be "
            f"confounded:\n  runs=0: {old_windows}\n  runs=1: {new_windows}"
        )
        sys.exit(1)

    import cv2
    import numpy as np

    cap_a = cv2.VideoCapture(str(OUT_DIR / "runs_off.mp4"))
    cap_b = cv2.VideoCapture(str(OUT_DIR / "runs_on.mp4"))
    worst = 0
    worst_frame = -1
    differing_pixels = 0
    frames = 0
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                print(f"FRAME COUNT MISMATCH at {frames}")
                sys.exit(1)
            break
        delta = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
        d = int(delta.max())
        differing_pixels += int((delta.max(axis=2) > 0).sum())
        if d > worst:
            worst = d
            worst_frame = frames
        frames += 1
    cap_a.release()
    cap_b.release()
    print(f"frames compared: {frames}")
    print(
        f"max absolute channel difference: {worst} "
        f"(frame {worst_frame}); differing pixels total: {differing_pixels}"
    )
    if worst != 0 or differing_pixels != 0:
        print("FAIL: outputs are not byte-identical")
        sys.exit(1)
    print("PASS: every frame of both renders is byte-identical")


if __name__ == "__main__":
    main()
