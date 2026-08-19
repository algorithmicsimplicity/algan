"""Is the render a function of the sorted hit list alone? -- ss4.8's missing check.

``DESIGN_mesh_identity_open.md`` ssJ asks for a property, not a number: the
resolution of a pixel should depend on the canonically sorted list of hits at
that pixel and on nothing else -- not on the K-buffer's width, not on how the
BVH happened to order instances, not on how many rays a wavefront tile holds,
and not on where the render's frame batches happened to fall. Each of those is
a bookkeeping choice with no place in the answer.

That property is *claimed* to hold at shipped defaults, because the greedy
``seam_t`` dedup that broke it is compiled out. Nothing demonstrated it. This
does, by rendering ONE scene under each lever and diffing:

    KBUF                 1 / 4 / 8        depth-window width of the hit gather
    BVH instance order   morton / split   which order leaves reach the walk in
    wavefront tile       2M / 128k rays   how many rays are in flight at once
    frame batch window   pinned large / pinned small

WHAT COUNTS AS A PASS, AND WHY IT IS NOT ALWAYS ZERO
-----------------------------------------------------
Two known effects put a floor under the diff, and both are recorded elsewhere in
this repo rather than discovered here:

* **Split-pixel nondeterminism.** A pixel resolved through three or more
  branches accumulates through a float ``atomic_add`` on ``pix_accum``, whose
  order is not reproducible. Measured cap: ``|d| = 1``. So the honest reference
  is not zero but the scene's OWN run-to-run noise floor, which this measures
  first, from two renders that differ in nothing at all.
* **Re-windowed rate functions.** Torch's CPU evaluation of a rate function
  rounds differently depending on the materialization window, so a scene
  rendered in two different frame batchings does not have identical *inputs* --
  the geometry itself moves in the last bits before any renderer choice
  applies. That is a property of the timeline, not of the hit list, so the
  window arm is additionally run on a STATIC scene, where the inputs are
  identical by construction and the answer must be exact.

So each arm is reported against the noise floor, and the window arm gets both a
moving and a static reading. A lever that lands inside the floor has been shown
not to matter; a lever that lands outside it has been shown to matter, which is
the finding either way.

WHY SUBPROCESSES
-----------------
``KBUF`` is a module-level constant read at import and baked into array widths,
and ``ALGAN_WAVEFRONT_TILE`` is read the same way. Neither can be changed in a
live process, so every arm is its own interpreter. That also removes any
question of stale module state carrying between arms.

Usage:
    <venv-python> benchmarks/_order_window_check.py            # every arm
    <venv-python> benchmarks/_order_window_check.py --arms kbuf1 split
    <venv-python> benchmarks/_order_window_check.py --res ld   # cheaper
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "algan_outputs" / "order_window"

#: Pinned so the frame batching is a CHOICE this script makes rather than a
#: consequence of whatever the driver reports free at the moment (batch windows
#: are sized from live free VRAM, which is not reproducible run to run).
WINDOW_BIG = 1_400_000_000
WINDOW_SMALL = 150_000_000

#: tag -> (extra env, static scene?, pinned memory override). The frame batch
#: window is a SETTING rather than an env var, so it is passed on the command
#: line: ``algan/environment.py`` rejects names it does not declare, and a
#: benchmark has no business adding one to the package's list.
ARMS = {
    "ref": ({}, False, WINDOW_BIG),
    "noise": ({}, False, WINDOW_BIG),
    "kbuf1": ({"ALGAN_KBUF": "1"}, False, WINDOW_BIG),
    "kbuf8": ({"ALGAN_KBUF": "8"}, False, WINDOW_BIG),
    # The instance-ORDER arms must also turn the refit tree off, and are
    # compared against ``refit_off`` rather than ``ref``. ``BVH_REFIT`` defaults
    # ON, and ``_build_accel``'s refit branch ignores ``builder`` outright, so
    # with it on both order arms build the identical RefitBVH: the leg would
    # report byte-identity for a lever that never moved. (That is also why
    # ``_bez_bvh_ab.py`` found ALGAN_BEZ_BVH_SPLIT byte-identical at 0.993x --
    # it was A/B-ing one render against itself.)
    "refit_off": ({"ALGAN_BVH_REFIT": "0"}, False, WINDOW_BIG),
    "morton": (
        {"ALGAN_BVH_REFIT": "0", "ALGAN_BVH_BUILD": "morton"},
        False,
        WINDOW_BIG,
    ),
    "split": (
        {
            "ALGAN_BVH_REFIT": "0",
            "ALGAN_BVH_BUILD": "split",
            "ALGAN_BEZ_BVH_SPLIT": "1",
        },
        False,
        WINDOW_BIG,
    ),
    "tile_small": ({"ALGAN_WAVEFRONT_TILE": str(1 << 17)}, False, WINDOW_BIG),
    "window_small": ({}, False, WINDOW_SMALL),
    "static_ref": ({}, True, WINDOW_BIG),
    "static_window": ({}, True, WINDOW_SMALL),
}

#: What each arm is compared against, and what the comparison is allowed to be.
#: "floor" means the run-to-run noise of the same configuration; "exact" means
#: byte-identity is available in principle and anything else is a finding.
COMPARISONS = [
    ("noise", "ref", "floor", "the scene's own run-to-run noise"),
    ("kbuf1", "ref", "floor", "K-buffer width 1 (gather refills every hit)"),
    ("kbuf8", "ref", "floor", "K-buffer width 8"),
    ("morton", "refit_off", "floor", "Morton instance order, STBVH (refit off)"),
    ("split", "refit_off", "floor", "median-split order, STBVH (refit off)"),
    ("tile_small", "ref", "floor", "16x more wavefront tiles"),
    ("window_small", "ref", "floor", "a third of the frame-batch memory"),
    ("static_window", "static_ref", "exact", "same, on a scene with no animation"),
]


def build_scene(static):
    """Depth complexity, several meshes, and translucency -- the three things
    the levers under test could plausibly reach.

    A stack of partly transparent sheets in front of solids is what makes the
    hit list at a pixel LONGER than any K-buffer, which is the only regime in
    which KBUF could change an answer. Reflective and refractive members put the
    secondary continuations through the same question. ``static`` drops the
    animation so the timeline hands every batching the same numbers.
    """
    from algan import (  # noqa: PLC0415
        BLUE,
        DARKER_GRAY,
        GREEN,
        IN,
        LEFT,
        OUT,
        PURE_RED,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Off,
        Scene,
        Sphere,
        Square,
        Sync,
    )

    Scene.set_background_color(DARKER_GRAY)
    sheets = []
    solids = []
    with Off():
        # Six coplanar-ish translucent sheets: the hit list at a centre pixel is
        # six long before any solid behind them, so KBUF = 1, 4 and 8 all have
        # to refill a different number of times to resolve it.
        for i in range(6):
            sheet = Square(color=(WHITE if i % 2 else YELLOW), opacity=0.35).scale(1.5)
            sheet.move(OUT * (1.2 - 0.35 * i) + RIGHT * (0.06 * i))
            sheet.spawn(animate=False)
            sheets.append(sheet)
        glass = Sphere(color=WHITE).scale(0.7).move(LEFT * 1.9 + IN * 0.4)
        glass.set_material(
            MeshPhysicalMaterial(transmission=0.9, ior=1.5, roughness=0.0)
        )
        glass.spawn(animate=False)
        solids.append(glass)
        mirror = Sphere(color=BLUE).scale(0.65).move(RIGHT * 2.0 + IN * 0.4)
        mirror.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
        mirror.spawn(animate=False)
        solids.append(mirror)
        for i, tint in enumerate((PURE_RED, GREEN)):
            solid = Sphere(color=tint).scale(0.5)
            solid.move(UP * (1.4 - 2.8 * i) + IN * 1.1)
            solid.spawn(animate=False)
            solids.append(solid)

    if static:
        return
    with Sync(run_time=1.5):
        for i, sheet in enumerate(sheets):
            sheet.rotate(9 * (i + 1), OUT)
        for i, solid in enumerate(solids):
            solid.move(UP * (0.22 if i % 2 else -0.22))
        Scene.get_camera().move(RIGHT * 0.2)


def render_arm(tag, out_path, res, static, window):
    from algan import KERNEL_REGISTRY, LD, MD, SETTINGS, Scene  # noqa: PLC0415
    from algan.scene_manager import SceneManager  # noqa: PLC0415

    quality = {"ld": LD, "md": MD}[res]
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=int(window))
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene(static)
    # Count the frame BATCHES this arm actually rendered in and write it beside
    # the video. Without it the window arm proves nothing: a short scene can fit
    # in one batch at both memory settings, and "byte-identical" would then be a
    # statement about a lever that never moved (ss0.1 rule 1).
    batches = {"n": 0}
    # Hook KERNEL_REGISTRY.render_kernel, not the module attribute: the render
    # loop resolves the tracer through the registry, so rebinding
    # ``tracer.render_batch_raytraced`` counts nothing and reports a confident
    # zero -- which it did.
    orig_render = KERNEL_REGISTRY.render_kernel

    def counting(*a, **k):
        batches["n"] += 1
        return orig_render(*a, **k)

    KERNEL_REGISTRY.render_kernel = counting
    try:
        Scene.save_video(str(out_path), quality, overwrite=True)
    finally:
        KERNEL_REGISTRY.render_kernel = orig_render
    Path(str(out_path) + ".batches").write_text(str(batches["n"]))


def diff(a, b):
    """Worst channel delta and how many pixels moved past the suites' tolerance."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    worst = 0
    moved = 0
    frames = 0
    while True:
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not ok_a or not ok_b:
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        worst = max(worst, int(d.max()))
        moved += int((d.max(axis=2) > 2).sum())
        frames += 1
    ca.release()
    cb.release()
    return worst, moved, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--res", choices=("ld", "md"), default="ld")
    ap.add_argument("--render", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.render:
        _env, static, window = ARMS[args.render]
        render_arm(
            args.render, OUT_DIR / f"{args.render}.mp4", args.res, static, window
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(args.arms) if args.arms else None
    needed = []
    for src, ref, _kind, _why in COMPARISONS:
        if wanted and src not in wanted:
            continue
        for tag in (ref, src):
            if tag not in needed:
                needed.append(tag)
    if wanted is None:
        needed = list(ARMS)

    for tag in needed:
        env = dict(os.environ)
        env.update(ARMS[tag][0])
        # Every arm renders in its OWN interpreter: KBUF and the tile size are
        # read at import and baked into array widths, so an in-process sweep
        # would silently measure the first arm's constants under later arms'
        # labels.
        print(f"-- rendering {tag} {ARMS[tag][0] or '(defaults)'}", flush=True)
        cmd = [sys.executable, __file__, "--render", tag, "--res", args.res]
        r = subprocess.run(cmd, env=env, cwd=str(REPO))
        if r.returncode != 0:
            print(f"   arm {tag} FAILED (exit {r.returncode})")
            return 1

    print()
    floor = None
    head = (
        f"{'lever':16s} {'vs':12s} {'worst':>6s} {'moved px':>9s} {'verdict':>10s}  why"
    )
    print(head)
    print("-" * len(head))
    for src, ref, kind, why in COMPARISONS:
        if wanted and src not in wanted:
            continue
        a, b = OUT_DIR / f"{ref}.mp4", OUT_DIR / f"{src}.mp4"
        if not a.exists() or not b.exists():
            continue
        worst, moved, frames = diff(a, b)
        nb_a = (
            Path(str(a) + ".batches").read_text()
            if Path(str(a) + ".batches").exists()
            else "?"
        )
        nb_b = (
            Path(str(b) + ".batches").read_text()
            if Path(str(b) + ".batches").exists()
            else "?"
        )
        if src == "noise":
            floor = worst
            verdict = "FLOOR"
        elif kind == "exact":
            verdict = "ok" if worst == 0 else "MOVES"
        elif floor is None:
            verdict = "?"
        else:
            verdict = "ok" if worst <= floor else "MOVES"
        print(
            f"{src:16s} {ref:12s} {worst:6d} {moved:9d} {verdict:>10s}  {why}"
            f"  ({frames} frames, batches {nb_a}->{nb_b})"
        )
    if floor is not None:
        print(f"\nnoise floor = {floor} channel values; 'ok' means at or under it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
