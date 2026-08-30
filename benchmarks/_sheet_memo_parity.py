"""Frame-level A/B for the shadowed resolve's cross-pass material memo.

``sheet_resolve_memo`` (RENDERER_WORK_QUEUE.md item 9) has mode 1 of
``sheet_resolve_shade`` store each processed triangle sheet's fetched
material -- colour(4), alpha, reflectivity, roughness, IOR, transmission,
surface point, twelve floats -- and mode 2 read it back instead of calling
``_tri_color_g`` / ``_tri_extra_g`` / ``_tri_ior_transmission_g`` /
``_tri_surface_point`` a second time. The values are copied verbatim through
f32, so the render must be BYTE-IDENTICAL.

Three arms, one process:

* ``off``    -- the toggle off, i.e. mode 2 re-fetches (the legacy path);
* ``on``     -- the toggle on. MUST match ``off`` byte for byte;
* ``poison`` -- the toggle on, with the memo table scribbled between the two
  launches. MUST DIFFER from ``on``. This is the non-vacuity proof that
  actually bites: it shows mode 2 *reads* the memo rather than the table
  being written and ignored, which a passing byte-identical A/B alone cannot
  distinguish from the feature being dead.

In-process flipping is legal here: ``memo`` reaches the kernel as a
``ti.template()`` ARGUMENT, which Taichi specialises on (a fresh variant per
value), not as a module-level ``ti.static`` gate resolved once per process --
see CLAUDE.md's Taichi gotchas.

Fixture notes. ``max_bounces`` is pinned to 0 so no continuation ever reaches
the shared pool: a pixel carrying three or more branches is not byte-
reproducible run to run (agent_guidance/memory_perf.md), which would mask the
comparison this script exists to make. Everything else is chosen to make the
memo's columns live rather than incidental -- see ``build_scene``.

    uv run python benchmarks/_sheet_memo_parity.py
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
# A warm daemon keeps adaptive renderer state across runs; both arms must run
# in one process of their own.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "algan_outputs" / "_sheet_memo_parity"
IMAGE = REPO / "benchmarks" / "performance" / "world_map.png"

from algan import *  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.raytracing import sheet_resolve_taichi as srt  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

#: Positions in ``sheet_resolve_shade``'s argument list, after the 38 shared
#: ``pre_args`` the two launch sites build in raster_pipeline. Asserted at
#: every call rather than trusted, so a signature change fails loudly here
#: instead of silently measuring the wrong thing.
_MODE_ARG = 38
_MEMO_ARG = 40
_SHEET_MEMO_ARG = 41
_SHEET_ACCEPT_ARG = 42


def build_scene():
    """A scene that makes every memo column load-bearing.

    * lit matte spheres over a ground plane -- accepted shadow events, the
      ordinary case;
    * an UNLIT cube -- a processed triangle sheet that builds NO event, which
      is the case the memo covers and the existing event tables do not (they
      are written only under ``sheet_accept``);
    * a translucent cube -- keeps ``alpha`` below 1 and drives transmission,
      so the IOR/transmission columns reach the output through
      ``trans_share`` and pixels carry several sheets rather than one;
    * a textured ImageMob -- routes the colour fetch through the texture
      sampler, which is the expensive fetch the memo is there to skip;
    * a Text -- interleaves BEZIER sheets, which the memo deliberately skips
      (circuit fetches are cheap and hit no texture), so both arms have to
      walk a mixed sheet stream.
    """
    torch.manual_seed(1234)
    with Off():
        ground = Cube(color=WHITE).scale(4)
        ground.move(DOWN * 5.4)
        ground.spawn()
        Sphere().scale(0.8).move(LEFT * 2.4 + UP * 1.2).spawn()
        Sphere().scale(0.8).move(UP * 1.2).spawn()
        unlit = Cube(color=RED).scale(0.6).move(RIGHT * 2.4 + UP * 1.2)
        unlit.set_material(MeshBasicMaterial())
        unlit.spawn()
        glassy = Cube(color=BLUE).scale(0.7).move(DOWN * 0.6)
        glassy.opacity = 0.45
        glassy.spawn()
        ImageMob(str(IMAGE)).scale(0.5).move(LEFT * 3.2 + DOWN * 1.6).spawn()
        Text("memo").scale(0.8).move(DOWN * 2.9).spawn()
        mover = Cube(color=GREEN).scale(0.4).move(UP * 2.6).spawn()
    with Sync(duration=0.6):
        mover.move(RIGHT * 1.5)


class LaunchWatcher:
    """Counts resolve launches per mode and optionally poisons the memo.

    ``poison`` scribbles the memo table immediately before the mode-2 launch,
    which is exactly the window in which mode 2's reads are the only consumer
    of those bytes.
    """

    def __init__(self, poison=False):
        self.poison = poison
        self.counts = {}
        self.memo_rows = 0
        self.accepted = 0
        self.processed_sheets = 0
        self._orig = None

    def attach(self):
        watcher = self
        self._orig = srt.sheet_resolve_shade

        def watching(*args, **kwargs):
            mode = args[_MODE_ARG]
            assert mode in (0, 1, 2), (
                f"arg {_MODE_ARG} is {mode!r}, not a resolve mode -- the "
                "launch signature moved; recount pre_args in raster_pipeline"
            )
            memo_flag = args[_MEMO_ARG]
            assert memo_flag in (0, 1), (
                f"arg {_MEMO_ARG} is {memo_flag!r}, not a memo flag -- the "
                "launch signature moved"
            )
            sheet_memo = args[_SHEET_MEMO_ARG]
            assert sheet_memo.ndim == 2, (
                f"arg {_SHEET_MEMO_ARG} has {sheet_memo.ndim} dims, not 2 -- "
                "the launch signature moved"
            )
            assert sheet_memo.shape[1] == 12, (
                f"arg {_SHEET_MEMO_ARG} has width {sheet_memo.shape[1]}, not "
                "12 -- the memo column layout moved"
            )
            watcher.counts[mode] = watcher.counts.get(mode, 0) + 1
            if mode == 1 and memo_flag:
                watcher.memo_rows = max(watcher.memo_rows, sheet_memo.shape[0])
            if mode == 2 and watcher.poison:
                # Anything mode 2 then reads back is wrong on purpose. NaN
                # would propagate to a uniformly broken frame; a finite,
                # plainly-wrong value keeps the render sane and still moves
                # every pixel whose sheet read a memoized column.
                sheet_memo.fill_(0.5)
            out = watcher._orig(*args, **kwargs)
            if mode == 1:
                accept = args[_SHEET_ACCEPT_ARG]
                watcher.accepted += int(accept.sum())
                watcher.processed_sheets += int(accept.numel())
            return out

        srt.sheet_resolve_shade = watching
        # raster_pipeline imports the kernel inside the call, so rebinding the
        # module attribute is enough -- but pin the import site too in case
        # that ever hoists to module scope.
        import algan.rendering.raytracing.raster_pipeline as rp

        if hasattr(rp, "sheet_resolve_shade"):
            rp.sheet_resolve_shade = watching

    def detach(self):
        srt.sheet_resolve_shade = self._orig
        import algan.rendering.raytracing.raster_pipeline as rp

        if hasattr(rp, "sheet_resolve_shade"):
            rp.sheet_resolve_shade = self._orig


def render_arm(on, out_name, poison=False):
    rt_settings.set_sheet_resolve_memo(on)
    watcher = LaunchWatcher(poison=poison)
    watcher.attach()
    SceneManager.reset()
    try:
        with Scene() as scene:
            build_scene()
            SETTINGS.raytracing.shadows = True
            # Pin continuations off: a pixel with three or more pool branches
            # is not byte-reproducible (agent_guidance/memory_perf.md), and
            # this comparison must see only the memo.
            SETTINGS.raytracing.max_bounces = 0
            scene.save_video(
                str(OUT_DIR / out_name),
                video_settings=PREVIEW,
                overwrite=True,
                codec="libx264rgb",
                ffmpeg_params=["-crf", "0", "-preset", "fast"],
            )
    finally:
        watcher.detach()
        SETTINGS.raytracing.shadows = False
        rt_settings.set_sheet_resolve_memo(True)
    print(
        f"arm on={on} poison={poison}: launches={watcher.counts} "
        f"memo_rows={watcher.memo_rows} "
        f"accepted={watcher.accepted}/{watcher.processed_sheets} sheets"
    )
    return watcher


def compare(a_name, b_name):
    import cv2
    import numpy as np

    cap_a = cv2.VideoCapture(str(OUT_DIR / a_name))
    cap_b = cv2.VideoCapture(str(OUT_DIR / b_name))
    worst = 0
    differing_pixels = 0
    frames = 0
    mismatch = False
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                mismatch = True
            break
        delta = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
        worst = max(worst, int(delta.max()))
        differing_pixels += int((delta.max(axis=2) > 0).sum())
        frames += 1
    cap_a.release()
    cap_b.release()
    return frames, worst, differing_pixels, mismatch


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Both arms must choose the same batch windows, or the comparison is
    # confounded by re-windowed state. The memo adds 48 B per sheet of arena,
    # which the runtime memory model can see -- so hand the sizer far more
    # than the scene needs and cap the window well below the budget.
    SETTINGS.computing.set(
        max_animation_batch_size=5, max_cpu_memory_used=8 * (1 << 30)
    )
    # Warm-up render, discarded: the first render of a fresh process populates
    # the Tex glyph cache, whose antialiasing differs from every later run
    # (tests/README.md), and no arm may be the one that pays it.
    render_arm(True, "_warmup.mp4")
    off = render_arm(False, "arm_off.mp4")
    on = render_arm(True, "arm_on.mp4")
    poison = render_arm(True, "arm_poison.mp4", poison=True)

    problems = []
    for name, w in (("off", off), ("on", on), ("poison", poison)):
        if w.counts.get(1, 0) == 0 or w.counts.get(2, 0) == 0:
            problems.append(
                f"arm {name} never took the shadowed two-launch path "
                f"(launches={w.counts})"
            )
    if on.memo_rows <= 1:
        problems.append(
            "the ON arm never allocated a real memo table (memo_rows="
            f"{on.memo_rows}) -- no sheets, so nothing was memoized"
        )
    if on.accepted >= on.processed_sheets:
        problems.append(
            "every sheet of the ON arm built a shadow event, so the "
            "'processed but not accepted' rows the memo exists to cover "
            "were never exercised -- the unlit mob did not reach the resolve"
        )
    if on.accepted == 0:
        problems.append("the ON arm accepted no shadow events at all")
    for p in problems:
        print(f"VACUOUS: {p}")
    if problems:
        sys.exit(1)

    frames, worst, diff_px, mismatch = compare("arm_off.mp4", "arm_on.mp4")
    if mismatch:
        print("FRAME COUNT MISMATCH between arm_off and arm_on")
        sys.exit(1)
    print(f"frames compared: {frames}")
    print(f"off vs on: max |d| = {worst}, differing pixels = {diff_px}")

    p_frames, p_worst, p_diff, p_mismatch = compare("arm_on.mp4", "arm_poison.mp4")
    print(f"on vs poison: max |d| = {p_worst}, differing pixels = {p_diff}")

    if worst != 0 or diff_px != 0:
        print("FAIL: the memo changed the render -- it is not byte-identical")
        sys.exit(1)
    if p_mismatch or p_diff == 0:
        print(
            "VACUOUS: poisoning the memo changed nothing, so mode 2 never "
            "read it -- the byte-identical result above proves nothing"
        )
        sys.exit(1)
    print(
        "PASS: memo on == memo off byte for byte, and poisoning the memo "
        f"moves {p_diff} pixels, so the read is live"
    )


if __name__ == "__main__":
    main()
