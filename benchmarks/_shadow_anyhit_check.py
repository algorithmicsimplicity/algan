"""The purpose-built scene ss4.6 asks for: do the three SHADOW_ANYHIT modes agree?

``DESIGN_mesh_identity.md`` ss4.6 predicted that once mesh identity replaced the
seam epsilon, the three ``ALGAN_SHADOW_ANYHIT`` modes -- the plain march (0), the
any-hit early-out (1) and the gather-march ("gather") -- would stop disagreeing.
Running the suite could not test it: all three already produce the identical
sha256 there, because ``materials_and_lighting`` is the suite's only shadowed
scene and it reaches neither corner case. "The three modes agree" was a statement
about that scene, not about the renderer.

``_shadow_occlusion``'s own docstring names the two cases where the early-out
deliberately overrules the march, and they have different causes:

1. **An opaque edge hit** the seam merge would have folded into an earlier
   TRANSLUCENT edge hit within ``DEPTH_TIE_EPSILON`` (1e-4 world units). This one
   is identity-related, so it is the case ss3.3 could remove.
2. **An opaque blocker past ``MAX_SURFACES_PER_RAY``** (256) peeled surfaces. Not
   identity-related at all -- it is the peel depth, and no amount of identity work
   touches it.

So this builds one scene per case and renders each under all three modes.

**What counts as a result here is not "they agree".** Case 2's disagreement is
the early-out being RIGHT -- the docstring's last clause is that full occlusion is
the physically correct answer in both cases -- so a difference there is the
expected outcome and its absence means the scene failed to reach the case. Each
scene therefore reports whether it actually reached its corner case, and that
report is worth more than the agreement column: a green "all modes agree" from a
scene that reached nothing is exactly the false negative ss4.6 already fell into
once.

Usage:
    .venv/Scripts/python.exe benchmarks/_shadow_anyhit_check.py [quality]
"""

import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join("algan_outputs", "shadow_anyhit")
MODES = ("0", "1", "gather")

#: ss4.6 case 1 needs an opaque edge and a translucent edge within this of each
#: other along the shadow ray. Read from the kernel so the scene cannot drift
#: away from the constant it is built against.
from algan.rendering.raytracing.raytrace_kernels_taichi import (  # noqa: E402
    DEPTH_TIE_EPSILON,
    MAX_SURFACES_PER_RAY,
)

#: Deliberately INSIDE the tie window, by an order of magnitude: the point is to
#: land in the band where the merge folds the two hits together, not near its
#: edge where float noise decides.
TIE_GAP = DEPTH_TIE_EPSILON * 0.1

#: Comfortably past the peel depth, so the blocker is unreachable by the march.
STACK_COUNT = MAX_SURFACES_PER_RAY + 48


def build_tie_scene():
    """Case 1: an opaque quad edge a hair in front of a translucent quad edge.

    Both quads are edge-on to the light, so the shadow ray from the receiver
    grazes both edges and the two hits land within ``DEPTH_TIE_EPSILON`` of each
    other -- which is the band the seam merge folds.
    """
    from algan import (
        BLUE,
        DOWN,
        LEFT,
        ORIGIN,
        RIGHT,
        UP,
        WHITE,
        Off,
        PointLight,
        Scene,
        Square,
        Sync,
    )

    with Off():
        PointLight(location=UP * 4 + LEFT * 0.001, intensity=2.0).spawn(animate=False)
        ground = Square(color=WHITE).scale(6)
        ground.rotate(90, RIGHT).move(DOWN * 1.5)
        ground.spawn(animate=False)
        # Two coplanar-ish blockers standing edge-on to the light. The
        # translucent one sits TIE_GAP behind the opaque one along the ray, so
        # their edge hits fall inside the merge window.
        opaque = Square(color=BLUE).scale(1.5)
        opaque.rotate(90, RIGHT).move(UP * 0.5)
        opaque.spawn(animate=False)
        glass = Square(color=BLUE, opacity=0.4).scale(1.5)
        glass.rotate(90, RIGHT).move(UP * (0.5 - TIE_GAP))
        glass.spawn(animate=False)
    with Sync(run_time=1):
        Scene.get_camera().move(RIGHT * 0.2 + UP * 0.1)
    return ORIGIN


def build_stack_scene(count=None):
    """Case 2: an opaque blocker behind more than ``MAX_SURFACES_PER_RAY`` sheets.

    The march peels 256 surfaces and stops; the opaque blocker is past that, so
    the two disagree by construction -- and the early-out's answer (full
    occlusion) is the correct one.

    ``count`` exists for the REACH CHECK. Rendering the same scene with a stack
    far below the peel limit and comparing tells us whether the limit is being
    reached at all; without it, "the modes agree" cannot be distinguished from
    "the scene never got near the case".
    """
    count = STACK_COUNT if count is None else int(count)
    from algan import (
        DOWN,
        OUT,
        RED,
        RIGHT,
        UP,
        WHITE,
        Off,
        PointLight,
        Scene,
        Square,
        Sync,
    )

    with Off():
        # OFF-AXIS ON PURPOSE. A light directly overhead casts the stack's
        # shadow straight down onto ground the camera sees edge-on at the
        # horizon, so the one region that answers the question is the one region
        # not in frame -- which is how the first run of this harness reported
        # "all three modes agree" from a scene that showed no shadow at all.
        # Behind the stack throws it forward, onto ground the camera faces.
        PointLight(location=UP * 5 - OUT * 2.5, intensity=2.0).spawn(animate=False)
        ground = Square(color=WHITE).scale(6)
        ground.rotate(90, RIGHT).move(DOWN * 1.5)
        ground.spawn(animate=False)
        for i in range(count):
            sheet = Square(color=WHITE, opacity=0.02).scale(2.0)
            sheet.rotate(90, RIGHT).move(UP * (0.2 + 0.004 * i))
            sheet.spawn(animate=False)
        blocker = Square(color=RED).scale(2.0)
        blocker.rotate(90, RIGHT).move(UP * (0.2 + 0.004 * count + 0.05))
        blocker.spawn(animate=False)
    with Sync(run_time=1):
        Scene.get_camera().move(OUT * 0.2)


SCENES = {"tie": "build_tie_scene", "stack": "build_stack_scene"}


def _render(scene_key, mode, quality_name, arg=None, tag=""):
    """One render in a FRESH process.

    ``SHADOW_ANYHIT`` is read at import into a module global, so switching modes
    in-process would need a setter and a guarantee that nothing cached the old
    value. A subprocess per arm is slower and cannot be wrong.

    ``arg`` is passed to the scene builder (the reach check uses it to vary the
    stack depth) and ``tag`` keeps that render's output beside the others.
    """
    path = os.path.join(OUT_DIR, f"shadow_{scene_key}_{mode}{tag}.mp4")
    env = dict(os.environ)
    env["ALGAN_SHADOW_ANYHIT"] = mode
    code = (
        "import benchmarks._shadow_anyhit_check as c;"
        "from algan import Scene, SETTINGS;"
        "from algan.scene_manager import SceneManager;"
        "import algan as A;"
        "SceneManager.reset();"
        "SETTINGS.raytracing.set(shadows=True);"
        f"q = getattr(A, {quality_name!r});"
        "SceneManager.instance().current_scene.set_video_settings(q);"
        f"getattr(c, {SCENES[scene_key]!r})({'' if arg is None else arg});"
        f"Scene.save_video({path!r}, q, overwrite=True)"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=os.getcwd())
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest(), path


def main():
    quality = sys.argv[1] if len(sys.argv) > 1 else "MD"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"DEPTH_TIE_EPSILON={DEPTH_TIE_EPSILON} tie gap={TIE_GAP}  "
        f"MAX_SURFACES_PER_RAY={MAX_SURFACES_PER_RAY} stack={STACK_COUNT}",
        flush=True,
    )
    import numpy as np

    from benchmarks._one_mesh_ab import read_frames

    want = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for key in SCENES:
        if want and key not in want:
            continue
        shas, paths = {}, {}
        for mode in MODES:
            shas[mode], paths[mode] = _render(key, mode, quality)
        base = MODES[0]
        print(f"\n{key}: march sha {shas[base][:12]}")
        if key == "stack":
            # REACH CHECK. The same scene with a stack far below the peel limit.
            # If the deep stack renders the same as the shallow one, the limit
            # was never reached and the agreement column below says nothing
            # about the corner case -- only that two scenes which both avoid it
            # agree. The first run of this harness reported exactly that, from a
            # scene whose shadow fell outside the frame entirely.
            _sha, shallow = _render(key, base, quality, arg=8, tag="_shallow")
            a, b = read_frames(paths[base]), read_frames(shallow)
            n = min(len(a), len(b))
            d = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
            print(
                f"  reach check: {STACK_COUNT}-deep vs 8-deep max|d|={int(d.max())}"
                f"  -> peel limit {'REACHED' if int(d.max()) > 2 else 'NOT reached'}"
            )
        for mode in MODES[1:]:
            a, b = read_frames(paths[base]), read_frames(paths[mode])
            n = min(len(a), len(b))
            d = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
            moved = int((d.max(axis=-1) > 2).sum())
            print(
                f"  {mode:7s} sha {shas[mode][:12]} "
                f"{'IDENTICAL' if shas[mode] == shas[base] else 'DIFFERS  '} "
                f"max|d|={int(d.max()):3d} px>2={moved}"
            )


if __name__ == "__main__":
    main()
