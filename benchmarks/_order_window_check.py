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

    kbuf                 1 / 4 / 8        depth-window width of the hit gather
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

LOSSLESS, AND THE TWO SCENES THE OLD GATES MADE UNTESTABLE
-----------------------------------------------------------
Two extensions from ``DESIGN_sheet_resolve.md`` Phase 0:

* **Every arm renders lossless** (``codec="libx264rgb"``, ``-crf 0``): a pixel
  diff read from a lossy MP4 measures the encoder, not the renderer -- measured
  at up to ~2,000x inflation (``DESIGN_mesh_identity.md`` ss6.7.3). Byte-identity
  verdicts were always safe; the *moved-pixel* columns only mean something now.
* **An env-mapped scene and a non-default-tonemap scene**, as ``env_*`` and
  ``tm_*`` arms. The sparse-coverage route requires ``not env_active`` and the
  default tonemap (``_get_tonemap_t_val() == 3``), so those two configurations
  run the DENSE resolve and could never be exercised by the base scene at
  shipped defaults. The sheet redesign deletes both gates; these arms are the
  regression net that has to stay byte-inert through every phase of it. The
  tonemap arm flips ``post_process_tonemap`` (the experimental toggle), because
  that is the one the route gate actually reads -- the public ``tonemapping``
  flag does not change path selection.

One falsifiable prediction, recorded here because this harness owns the number:
at ``--res md`` the base scene's run-to-run NOISE FLOOR was measured at 46
channel values over 212k pixels (translucent-stack edges + the glass sphere)
-- far above the |d| = 1 split-pixel cap, unexplained, and tolerated. The sheet
resolve's no-atomics rule (``DESIGN_sheet_resolve.md`` ss2.2) predicts that
floor goes to exactly ZERO once the resolve ships. Check it with
``--arms noise --res md`` before and after.

WHY SUBPROCESSES
-----------------
``kbuf`` is a module-level constant read at import and baked into array widths,
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

#: tag -> (extra env, scene kind, pinned memory override). The frame batch
#: window is a SETTING rather than an env var, so it is passed on the command
#: line: ``algan/environment.py`` rejects names it does not declare, and a
#: benchmark has no business adding one to the package's list.
#:
#: Scene kinds: "moving" is the base scene; "static" drops its animation;
#: "env" adds an environment map (dense resolve, empty pixels sample the sky);
#: "tonemap" is the base scene with post_process_tonemap OFF (dense resolve,
#: in-kernel tonemap).
ARMS = {
    "ref": ({}, "moving", WINDOW_BIG),
    "noise": ({}, "moving", WINDOW_BIG),
    "kbuf1": ({"ALGAN_KBUF": "1"}, "moving", WINDOW_BIG),
    "kbuf8": ({"ALGAN_KBUF": "8"}, "moving", WINDOW_BIG),
    # The instance-ORDER arms must also turn the refit tree off, and are
    # compared against ``refit_off`` rather than ``ref``. ``bvh_refit`` defaults
    # ON, and ``_build_accel``'s refit branch ignores ``builder`` outright, so
    # with it on both order arms build the identical RefitBVH: the leg would
    # report byte-identity for a lever that never moved. (That is also why
    # ``_bez_bvh_ab.py`` found ALGAN_BEZ_BVH_SPLIT byte-identical at 0.993x --
    # it was A/B-ing one render against itself.)
    "refit_off": ({"ALGAN_BVH_REFIT": "0"}, "moving", WINDOW_BIG),
    "morton": (
        {"ALGAN_BVH_REFIT": "0", "ALGAN_BVH_BUILD": "morton"},
        "moving",
        WINDOW_BIG,
    ),
    "split": (
        {
            "ALGAN_BVH_REFIT": "0",
            "ALGAN_BVH_BUILD": "split",
            "ALGAN_BEZ_BVH_SPLIT": "1",
        },
        "moving",
        WINDOW_BIG,
    ),
    "tile_small": ({"ALGAN_WAVEFRONT_TILE": str(1 << 17)}, "moving", WINDOW_BIG),
    "window_small": ({}, "moving", WINDOW_SMALL),
    "static_ref": ({}, "static", WINDOW_BIG),
    "static_window": ({}, "static", WINDOW_SMALL),
    # DESIGN_sheet_resolve.md Phase 0: the two configurations the sparse gate
    # excludes today, exercised with the levers that could plausibly reach
    # their dense resolve. The env scene's mirror sphere reflects the map, so
    # kbuf and instance order reach it through secondary rays.
    "env_ref": ({}, "env", WINDOW_BIG),
    "env_noise": ({}, "env", WINDOW_BIG),
    "env_kbuf1": ({"ALGAN_KBUF": "1"}, "env", WINDOW_BIG),
    "env_kbuf8": ({"ALGAN_KBUF": "8"}, "env", WINDOW_BIG),
    "env_refit_off": ({"ALGAN_BVH_REFIT": "0"}, "env", WINDOW_BIG),
    "env_morton": (
        {"ALGAN_BVH_REFIT": "0", "ALGAN_BVH_BUILD": "morton"},
        "env",
        WINDOW_BIG,
    ),
    "env_split": (
        {
            "ALGAN_BVH_REFIT": "0",
            "ALGAN_BVH_BUILD": "split",
            "ALGAN_BEZ_BVH_SPLIT": "1",
        },
        "env",
        WINDOW_BIG,
    ),
    "env_tile_small": ({"ALGAN_WAVEFRONT_TILE": str(1 << 17)}, "env", WINDOW_BIG),
    "env_window_small": ({}, "env", WINDOW_SMALL),
    # Same geometry as the base scene, so the order arms above already cover
    # instance order at this scene's shape; what tonemap-off adds is the
    # in-kernel tonemap in the resolve/composite, which tiles, windows and the
    # gather width could interact with.
    "tm_ref": ({}, "tonemap", WINDOW_BIG),
    "tm_noise": ({}, "tonemap", WINDOW_BIG),
    "tm_kbuf8": ({"ALGAN_KBUF": "8"}, "tonemap", WINDOW_BIG),
    "tm_tile_small": ({"ALGAN_WAVEFRONT_TILE": str(1 << 17)}, "tonemap", WINDOW_BIG),
    "tm_window_small": ({}, "tonemap", WINDOW_SMALL),
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
    ("env_noise", "env_ref", "floor", "env-mapped: run-to-run noise"),
    ("env_kbuf1", "env_ref", "floor", "env-mapped: K-buffer width 1"),
    ("env_kbuf8", "env_ref", "floor", "env-mapped: K-buffer width 8"),
    ("env_morton", "env_refit_off", "floor", "env-mapped: Morton order (refit off)"),
    ("env_split", "env_refit_off", "floor", "env-mapped: split order (refit off)"),
    ("env_tile_small", "env_ref", "floor", "env-mapped: 16x more wavefront tiles"),
    ("env_window_small", "env_ref", "floor", "env-mapped: a third of the memory"),
    ("tm_noise", "tm_ref", "floor", "tonemap-in-kernel: run-to-run noise"),
    ("tm_kbuf8", "tm_ref", "floor", "tonemap-in-kernel: K-buffer width 8"),
    ("tm_tile_small", "tm_ref", "floor", "tonemap-in-kernel: 16x more tiles"),
    ("tm_window_small", "tm_ref", "floor", "tonemap-in-kernel: a third of the memory"),
]


def build_scene(scene_kind):
    """Depth complexity, several meshes, and translucency -- the three things
    the levers under test could plausibly reach.

    A stack of partly transparent sheets in front of solids is what makes the
    hit list at a pixel LONGER than any K-buffer, which is the only regime in
    which kbuf could change an answer. Reflective and refractive members put the
    secondary continuations through the same question. ``static`` drops the
    animation so the timeline hands every batching the same numbers. ``env``
    surrounds the same moving scene with a deterministic gradient environment
    map -- authored as a float tensor, no file dependency -- so empty pixels
    sample the sky and the mirror sphere reflects it.
    """
    static = scene_kind == "static"
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
    if scene_kind == "env":
        import torch  # noqa: PLC0415

        # Deterministic horizontal hue ramp with a vertical brightness ramp:
        # enough structure that a shifted reflection or sky sample cannot land
        # on the same value it left. Float tensor => taken as authored (0..1).
        h, w = 8, 16
        xs = torch.linspace(0.0, 1.0, w).view(1, w, 1).expand(h, w, 1)
        ys = torch.linspace(0.15, 0.9, h).view(h, 1, 1).expand(h, w, 1)
        env = torch.cat([xs * ys, ys, (1.0 - xs) * ys], dim=2).contiguous()
        Scene.set_environment_map(env, intensity=1.0, ambient=True)
    sheets = []
    solids = []
    with Off():
        # Six coplanar-ish translucent sheets: the hit list at a centre pixel is
        # six long before any solid behind them, so kbuf = 1, 4 and 8 all have
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


def render_arm(tag, out_path, res, scene_kind, window):
    from algan import KERNEL_REGISTRY, LD, MD, SETTINGS, Scene  # noqa: PLC0415
    from algan.scene_manager import SceneManager  # noqa: PLC0415

    quality = {"ld": LD, "md": MD}[res]
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=int(window))
    if scene_kind == "tonemap":
        # The route gate reads _get_tonemap_t_val() == 3, which is
        # post_process_tonemap -- the public ``tonemapping`` flag does not
        # change path selection, so this is the lever that actually forces the
        # in-kernel tonemap (and, today, the dense resolve).
        SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene(scene_kind)
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
        # Lossless on purpose (DESIGN_mesh_identity.md ss6.7.3): the moved-pixel
        # columns this harness prints are only the renderer's if the codec adds
        # nothing. libx264rgb at crf 0 is bit-exact RGB.
        Scene.save_video(
            str(out_path),
            quality,
            overwrite=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0"],
        )
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
        _env, scene_kind, window = ARMS[args.render]
        render_arm(
            args.render, OUT_DIR / f"{args.render}.mp4", args.res, scene_kind, window
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
        # Every arm renders in its OWN interpreter: kbuf and the tile size are
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
    # One noise floor per SCENE FAMILY: the env and tonemap scenes have their
    # own noise arms, and comparing an env lever against the base scene's floor
    # would let a genuinely noisy family hide a lever (or damn a quiet one).
    floors = {}

    def family(tag):
        for prefix in ("env_", "tm_", "static_"):
            if tag.startswith(prefix):
                return prefix
        return ""

    head = (
        f"{'lever':16s} {'vs':14s} {'worst':>6s} {'moved px':>9s} {'verdict':>10s}  why"
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
        floor = floors.get(family(src))
        if src.endswith("noise"):
            floors[family(src)] = worst
            verdict = "FLOOR"
        elif kind == "exact":
            verdict = "ok" if worst == 0 else "MOVES"
        elif floor is None:
            verdict = "?"
        else:
            verdict = "ok" if worst <= floor else "MOVES"
        print(
            f"{src:16s} {ref:14s} {worst:6d} {moved:9d} {verdict:>10s}  {why}"
            f"  ({frames} frames, batches {nb_a}->{nb_b})"
        )
    for fam, floor in floors.items():
        label = {"": "base", "env_": "env", "tm_": "tonemap"}.get(fam, fam)
        print(f"\n{label} noise floor = {floor} channel values ('ok' = at or under)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
