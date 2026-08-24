"""Acceptance harness for the per-mob shadow flags (``Mob.casts_shadows`` /
``Mob.receives_shadows``).

The flags are checked against EXTERNAL oracles rather than against themselves --
each arm below has an independently-rendered image it must reproduce, so a
mechanism that is plumbed but inert cannot pass:

* ``noncaster``  -- one cube stops casting. Inside that cube's own silhouette the
  frame must be BYTE-IDENTICAL to ``control`` (a non-caster still renders
  normally); everywhere else it must match ``drop`` -- the same scene with that
  cube deleted outright -- because "casts no shadow" and "is not there" are the
  same statement as far as every other pixel is concerned. This is the arm that
  catches a flag that removes too much or too little.
* ``allcast`` / ``allrecv`` -- EVERY mob opts out of casting (resp. receiving).
  Both must reproduce ``noshadow``, the same scene rendered with the global
  ``SETTINGS.raytracing.shadows`` switched off, since a scene where nothing casts
  and a scene where nothing receives are both a scene with no shadows in it. Two
  independent routes to one known image.
* ``control`` -- every flag left at its default. Must be byte-identical to the
  same arm rendered before the feature existed, which is what establishes that
  ordinary scenes did not move.

The silhouette mask is eroded before the "unchanged" test and dilated before the
"matches the oracle" test: an antialiased edge pixel blends the cube with the
ground behind it, so it legitimately moves when that ground's shadow does, and
belongs to neither claim.

Usage (one process per arm -- the render route and several kernel gates are
``ti.static``, so two arms in one process silently share the first one's code)::

    for arm in control noshadow drop noncaster allcast allrecv silhouette; do
        .venv/bin/python benchmarks/_shadow_flags_check.py $arm
    done
    .venv/bin/python benchmarks/_shadow_flags_check.py compare

Add ``ALGAN_ANALYTIC_AA=0`` to every render to repeat the whole set on the
classic wavefront instead of the sheet route; the arms are then tagged ``_wf``
and compared among themselves. Frames are kept as .npy so the comparison is on
the exact channel values, not on a re-encoded video.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A warm daemon carries the previous arm's import-time renderer state, and these
# arms differ in exactly that; see CLAUDE.md on _IMPORT_TIME_VARIABLES.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "shadow_flags")
ARMS = ("control", "noshadow", "drop", "noncaster", "allcast", "allrecv", "silhouette")

# The suite-wide tolerance (CLAUDE.md). Demanded as an exact 0 only where the two
# arms are the same render of the same geometry.
TOL = 2


def _suffix():
    """Arms are tagged by the render route AND the shadow-query mode.

    The two are separate axes and both matter here: the route decides which
    kernel spawns the shadow ray (``receives_shadows``), and
    ``ALGAN_SHADOW_ANYHIT`` decides which of FOUR traversal bodies answers it
    (``casts_shadows``) -- the ordered march by default, the opaque any-hit
    walks at ``1``, the KBUF gather march at ``gather``. Every one of those
    bodies carries its own copy of the leaf test, so a set rendered at the
    default proves nothing about the other two.
    """
    tag = "_wf" if os.environ.get("ALGAN_ANALYTIC_AA") == "0" else ""
    anyhit = (os.environ.get("ALGAN_SHADOW_ANYHIT") or "0").strip().lower()
    if anyhit not in ("0", "false", ""):
        tag += "_ah" if anyhit != "gather" else "_gather"
    return tag


def _path(arm):
    return os.path.join(OUT_DIR, f"{arm}{_suffix()}.npy")


def build_scene(arm):
    """One frame: a lit ground plane, a cube that casts across it (A), a second
    cube whose shadow is the thing under test (B), and a flat plate lying inside
    A's shadow (C) whose shading is what ``receives_shadows`` changes.

    Flags are applied BEFORE spawn -- the render primitive reads them once, the
    same contract ``two_sided`` and ``closed_shell`` keep.
    """
    Scene.set_background_color(DARKER_GRAY)
    SETTINGS.raytracing.set(shadows=(arm != "noshadow"))

    with Off():
        AmbientLight(color=WHITE, intensity=0.25).spawn(animate=False)
        # shadow_angle 0 keeps the light a point emitter: one hard shadow ray
        # per fragment, so a visibility difference cannot hide inside a soft
        # fan's averaging.
        DirectionalLight(
            location=RIGHT * 2 + UP * 6 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=0.0,
        ).spawn(animate=False)

        # Everything here is a Polyhedron -- a lit triangle mesh. It has to be:
        # a 2-D Square floor is a bezier circuit, which the renderer draws
        # unlit, and an unlit surface builds no shadow event and so cannot
        # receive a shadow however the flags are set. (Rendering this scene on
        # 2-D shapes showed zero difference between shadows on and off, which is
        # what caught it.) fill_opacity is pinned to 1 because Cube's default is
        # Manim's 0.75, and a translucent occluder only attenuates.
        mobs = []
        # The silhouette arm renders B alone against the background, so the
        # pixels B covers can be told from the pixels its shadow falls on.
        if arm != "silhouette":
            floor = Prism(
                dimensions=(16, 0.5, 9), fill_color=GRAY, fill_opacity=1
            ).move(DOWN * 2.2)
            mobs.append(floor)

            cube_a = Cube(side_length=1.5, fill_color=RED, fill_opacity=1).move(
                LEFT * 2.2 + UP * 0.4
            )
            mobs.append(cube_a)

            # C lies flat on the floor inside A's shadow, so "does not receive"
            # is visible as the plate lighting up.
            plate = Prism(
                dimensions=(2.6, 0.12, 2.2), fill_color=YELLOW, fill_opacity=1
            ).move(LEFT * 2.2 + DOWN * 1.9 + IN * 0.2)
            mobs.append(plate)

        # B: the cube under test. Absent entirely in the ``drop`` arm, which is
        # the oracle the ``noncaster`` arm's ground pixels must reproduce.
        cube_b = None
        if arm != "drop":
            cube_b = Cube(side_length=1.5, fill_color=BLUE, fill_opacity=1).move(
                RIGHT * 2.2 + UP * 0.4
            )
            mobs.append(cube_b)

        if arm == "noncaster":
            cube_b.casts_shadows = False
        elif arm == "allcast":
            for mob in mobs:
                mob.casts_shadows = False
        elif arm == "allrecv":
            for mob in mobs:
                mob.receives_shadows = False

        for mob in mobs:
            mob.spawn(animate=False)


def render(arm):
    os.makedirs(OUT_DIR, exist_ok=True)
    SETTINGS.computing.set(available_memory_override=2_400_000_000)
    SETTINGS.video.set(PREVIEW)
    SceneManager.reset()
    build_scene(arm)
    result = Scene.save_frame(
        os.path.join(OUT_DIR, f"{arm}{_suffix()}.png"), overwrite=True
    )
    import cv2

    frame = cv2.imread(result.output_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"render produced no readable frame at {result.output_path}")
    np.save(_path(arm), frame)
    print(f"{arm}{_suffix()}: saved {frame.shape} -> {_path(arm)}")


def _load(arm):
    p = _path(arm)
    if not os.path.exists(p):
        raise SystemExit(f"missing arm '{arm}' -- render it first ({p})")
    return np.load(p).astype(np.int32)


def _report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def compare():
    from scipy import ndimage

    control = _load("control")
    noshadow = _load("noshadow")
    drop = _load("drop")
    noncaster = _load("noncaster")
    allcast = _load("allcast")
    allrecv = _load("allrecv")
    sil = _load("silhouette")

    # B's silhouette: the pixels B covers, rendered alone against the background.
    bg = np.array(sil[0, 0], dtype=np.int32)
    mask_b = np.abs(sil - bg).max(-1) > TOL
    interior = ndimage.binary_erosion(mask_b, iterations=2)
    touched = ndimage.binary_dilation(mask_b, iterations=2)
    print(
        f"\nB silhouette {int(mask_b.sum())} px "
        f"(interior {int(interior.sum())}) of {mask_b.size}"
    )

    ok = True

    # The feature must actually move pixels, or every other check below passes
    # vacuously on an inert mechanism.
    d_ctl = np.abs(noncaster - control).max(-1)
    ok &= _report(
        "noncaster differs from control",
        int((d_ctl > TOL).sum()) > 0,
        f"{int((d_ctl > TOL).sum())} px changed (max delta {int(d_ctl.max())})",
    )

    # Inside B's own silhouette a non-caster renders EXACTLY as before.
    inside = d_ctl[interior]
    ok &= _report(
        "non-caster renders unchanged inside its own silhouette",
        inside.size and int(inside.max()) == 0,
        f"max delta {int(inside.max()) if inside.size else 'EMPTY MASK'} (want 0)",
    )

    # Outside it, the frame must match the scene with B deleted: its shadow is
    # exactly gone, and nothing else moved.
    d_drop = np.abs(noncaster - drop).max(-1)
    outside = d_drop[~touched]
    ok &= _report(
        "non-caster matches the B-deleted oracle outside the silhouette",
        outside.size and int(outside.max()) <= TOL,
        f"max delta {int(outside.max()) if outside.size else 'EMPTY'} (tol {TOL})",
    )

    # Both all-opt-out arms must reproduce the globally-shadowless render.
    for name, arm in (("allcast", allcast), ("allrecv", allrecv)):
        d = np.abs(arm - noshadow).max(-1)
        ok &= _report(
            f"{name} reproduces the global shadows=False render",
            int(d.max()) <= TOL,
            f"max delta {int(d.max())} over {int((d > TOL).sum())} px (tol {TOL})",
        )

    base = _path("control").replace(".npy", ".baseline.npy")
    if os.path.exists(base):
        d = np.abs(control - np.load(base).astype(np.int32)).max(-1)
        ok &= _report(
            "control is byte-identical to the pre-change baseline",
            int(d.max()) == 0,
            f"max delta {int(d.max())} over {int((d > 0).sum())} px (want 0)",
        )
    else:
        print(
            f"  [SKIP] pre-change baseline absent ({base}); capture it on the "
            f"base commit with the 'baseline' arm"
        )

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "control"
    if what == "compare":
        raise SystemExit(compare())
    if what == "baseline":
        # Render the control arm on the BASE commit and keep it as the
        # byte-identity reference for the feature branch.
        render("control")
        os.replace(_path("control"), _path("control").replace(".npy", ".baseline.npy"))
        print("kept as pre-change baseline")
        return
    if what not in ARMS:
        raise SystemExit(f"unknown arm '{what}'; one of {ARMS}, baseline, compare")
    render(what)


if __name__ == "__main__":
    main()
