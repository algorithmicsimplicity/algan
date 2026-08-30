"""Does a mixed-flag collection of DICED surfaces lose a legitimate shadow?

``shadow_cast_flag`` reduces the per-corner declaration to one bool per merged
primitive column, taking amax over frames. That is exact while one column means
one mob for the whole batch -- but a collection of logical-PN surfaces dices
adaptively per frame, so a column can host a patch of mob A in one frame and a
patch of mob B in the next. If A declines to cast and B does not, the
over-frames reduction makes the column non-casting on every frame, and B loses
part of its shadow.

Two arms, several frames, geometry moving so the dice levels shift:

* ``mixed``  -- sphere A (``casts_shadows=False``) beside sphere B (default).
* ``onlyb``  -- the same scene with A deleted outright.

B's shadow is the same in both, so on the right-hand side of the frame -- which
holds B and its shadow and nothing of A -- every frame must match. A mismatch
there is B's shadow being eaten by A's flag.

    .venv/bin/python benchmarks/_shadow_flags_mixed_dice_check.py mixed
    .venv/bin/python benchmarks/_shadow_flags_mixed_dice_check.py bothcast
    .venv/bin/python benchmarks/_shadow_flags_mixed_dice_check.py onlyb
    .venv/bin/python benchmarks/_shadow_flags_mixed_dice_check.py compare mixed
    .venv/bin/python benchmarks/_shadow_flags_mixed_dice_check.py compare bothcast

``bothcast`` is the control arm: it isolates how much of any difference is A's
mere presence perturbing B's adaptive dice rather than the flag.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "shadow_flags_dice")
TOL = 2
# The frame is 704 wide; A sits left of centre and B right of it, so this column
# holds B, B's shadow, and no part of A.
RIGHT_FROM = 400


def build_scene(arm):
    Scene.set_background(DARKER_GRAY)
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        AmbientLight(color=WHITE, intensity=0.25).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 2 + UP * 6 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=0.0,
        ).spawn(animate=False)
        floor = Prism(
            width=16, height=0.5, depth=9, fill_color=GRAY, fill_opacity=1
        ).move(DOWN * 2.2)
        floor.spawn(animate=False)

        # Curved surfaces, so both reach the renderer as logical-PN patches
        # diced per frame -- which is the whole point of this check.
        if arm != "onlyb":
            sphere_a = Sphere(radius=1.0, color=RED).move(LEFT * 2.4 + UP * 0.2)
            # ``bothcast`` is the CONTROL: A present and casting normally, so
            # the flag plays no part. Whatever it shows against ``onlyb`` is A's
            # mere presence perturbing B -- the adaptive dice is chosen per
            # batch, so a second surface in the batch can move B's levels and
            # with them its silhouette by a channel value or two. Only the
            # excess of ``mixed`` over ``bothcast`` is the flag's doing.
            if arm != "bothcast":
                sphere_a.casts_shadows = False
            sphere_a.spawn(animate=False)

        sphere_b = Sphere(radius=1.0, color=BLUE).move(RIGHT * 2.4 + UP * 0.2)
        sphere_b.spawn(animate=False)

    # Move the spheres toward and away from the camera so the adaptive dice
    # level changes frame to frame, which is what shuffles column ownership.
    with Sync(run_time=1.0):
        if arm != "onlyb":
            sphere_a.move(OUT * 1.6)
        sphere_b.move(IN * 1.2)


def render(arm):
    os.makedirs(OUT_DIR, exist_ok=True)
    SETTINGS.computing.set(available_memory_override=2_400_000_000)
    SETTINGS.video.set(PREVIEW.set(frames_per_second=8))
    SceneManager.reset()
    build_scene(arm)
    path = os.path.join(OUT_DIR, f"{arm}.mp4")
    Scene.save_video(path, overwrite=True)

    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    arr = np.stack(frames)
    np.save(os.path.join(OUT_DIR, f"{arm}.npy"), arr)
    print(f"{arm}: {arr.shape} -> {OUT_DIR}/{arm}.npy")


def compare():
    arm = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    a = np.load(os.path.join(OUT_DIR, f"{arm}.npy")).astype(np.int32)
    b = np.load(os.path.join(OUT_DIR, "onlyb.npy")).astype(np.int32)
    print(f"arm under test: {arm}")
    n = min(len(a), len(b))
    print(f"comparing {n} frames, columns {RIGHT_FROM}+ (B and its shadow only)")
    worst = 0
    bad = []
    for i in range(n):
        d = np.abs(a[i, :, RIGHT_FROM:] - b[i, :, RIGHT_FROM:]).max(-1)
        m = int(d.max())
        worst = max(worst, m)
        if m > TOL:
            bad.append((i, m, int((d > TOL).sum())))
    for i, m, px in bad:
        print(f"  frame {i}: max delta {m} over {px} px")
    verdict = (
        "PASS -- B's shadow is intact"
        if worst <= TOL
        else ("FAIL -- B's shadow differs when A declines to cast")
    )
    print(f"worst delta {worst} (tol {TOL})\n{verdict}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    if what == "compare":
        raise SystemExit(compare())
    render(what)
