"""Acceptance harness for the path tracer's four throughput switches.

Each switch is meant to change how the ``samples_per_pixel > 1`` renderer
spends its time, not what it computes. This renders ONE all-opaque, lit,
shadowed scene under the shipped defaults and then re-renders it with one
switch flipped at a time, comparing the frames:

===========================  ==============================================
switch                       expected
===========================  ==============================================
``pt_shadow_anyhit``         identical (the opaque any-hit walk answers the
                             same question as the ordered march on a batch
                             with nothing partly transparent in it; the two
                             corner cases ``_shadow_occluded``'s docstring
                             names need a translucent blocker, which this
                             scene has none of)
``pt_opaque_closest``        identical (with every visible primitive opaque
                             the peel ends at the first crossing, so the
                             k-buffer's other slots were filled for nothing)
``pt_ambient_rows``          identical (the same ambient / hemisphere rows,
                             in the same ascending order, found by a packed
                             list instead of a per-crossing type scan)
``pt_animated_seed``         FRAME 0 identical, later frames differ -- that
                             is the whole point of the switch: with the
                             frame folded out of the sampler key a static
                             region draws the same samples every frame, so
                             its noise stops shimmering
===========================  ==============================================

**A pass is only worth something if the switch was live**, so every arm also
reports what the host actually handed the kernels -- the shadow mode, the
``opaque_closest`` template, the packed ambient-row count. An arm that
reports mode 1 on both sides proved nothing.

One process per arm: two of these four are ``ti.static`` gates or template
arguments read when the kernel compiles, and a second arm in the same process
would silently reuse the first arm's code (CLAUDE.md, the ti.static hazard).

Usage::

    uv run python benchmarks/_pt_shadow_anyhit_check.py            # all arms
    uv run python benchmarks/_pt_shadow_anyhit_check.py shadow_anyhit
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

OUT_DIR = os.path.join(_REPO_ROOT, "algan_outputs", "pt_switches")

#: Tiny on purpose: the question is per-pixel agreement, not image quality,
#: and every arm pays a cold kernel compile.
RESOLUTION = (96, 54)
SAMPLES = 4
DURATION = 1.0  # at SMOKE_TEST's 2 fps: 3 rendered frames

#: Frames each arm renders. Every comparison is made on the RAW frames
#: (``Scene.get_frames``, saved beside the video as ``.npy``) rather than on
#: the encoded video: mp4 is a lossy inter-frame codec, so a genuine change
#: in frame 1 leaks a few counts into frame 0's decoded pixels (measured: 7
#: of 255) even when the rendered frame 0 is bit-identical -- which is
#: precisely the distinction the animated-seed arm is about. The video is
#: still written, and its own per-frame deltas reported, because it is what a
#: user would look at.
FRAMES = 2

#: name -> (environment variable, flipped value, expectation)
#: ``"same"`` = every frame must match; ``"same_first"`` = only frame 0 must
#: match and a later frame is EXPECTED to differ.
ARMS = {
    "shadow_anyhit": ("ALGAN_PT_SHADOW_ANYHIT", "0", "same"),
    "opaque_closest": ("ALGAN_PT_OPAQUE_CLOSEST", "0", "same"),
    "ambient_rows": ("ALGAN_PT_AMBIENT_ROWS", "0", "same"),
    "animated_seed": ("ALGAN_PT_ANIMATED_SEED", "1", "same_first"),
}


def build_scene():
    """An all-opaque lit scene with a real cast shadow.

    Every surface is a MESH. A ``Square`` would be a bezier circuit, and
    circuits do not cast shadows -- a scene built from them renders the same
    with shadows on and off, so every "the arms agree" it produced would be
    agreement about a scene with no shadow query in it
    (``_shadow_anyhit_check`` learned that the expensive way).

    The ambient and hemisphere lights are not decoration either: they are the
    direction-less rows ``pt_ambient_rows`` packs, and without one of each
    that arm compares two identical code paths.

    Every mob is spelled so the merge can PROVE it opaque, which is what the
    first two switches are gated on -- measured: a bare ``Cube(color=WHITE)``
    lands in the batch as translucent (its alpha does not read as a proven
    1.0) and takes ``all_visible_opaque`` down with it, while a ``Prism``, a
    ``Sphere`` and a ``Cube`` with an explicit ``set_opacity(1.0)`` do not.
    The baseline's engagement report is what catches a drift here.
    """
    from algan import (
        BLUE,
        DOWN,
        LEFT,
        ORANGE,
        OUT,
        RIGHT,
        UP,
        WHITE,
        AmbientLight,
        HemisphereLight,
        MeshLambertMaterial,
        Off,
        PointLight,
        Prism,
        Sphere,
        Sync,
    )

    with Off():
        ground = Prism(width=9.0, height=9.0, depth=0.2)
        ground.set_material(MeshLambertMaterial(color=WHITE))
        ground.move(DOWN * 2.2)
        ground.spawn(animate=False)

        caster = Sphere(color=BLUE).scale(0.9)
        caster.move(UP * 0.6 + LEFT * 0.3)
        caster.spawn(animate=False)

        box = Prism(width=1.0, height=1.0, depth=1.0)
        box.set_material(MeshLambertMaterial(color=ORANGE))
        box.move(RIGHT * 1.4 - UP * 0.2)
        box.spawn(animate=False)

        PointLight(location=UP * 4 - OUT * 2.5 + LEFT * 0.5, intensity=2.0).spawn(
            animate=False
        )
        AmbientLight(color=WHITE, intensity=0.15).spawn(animate=False)
        HemisphereLight(color=WHITE, ground_color=BLUE, intensity=0.1).spawn(
            animate=False
        )

    # Something has to move, or every frame is the same picture and the
    # animated-seed arm cannot show a difference where one is expected.
    with Sync(runtime=DURATION):
        caster.move(RIGHT * 0.8)


#: What the child process runs. Wraps the two kernels the switches reach so
#: the arm reports the values the host actually passed, then renders.
_CHILD = """
import json, sys
sys.path.insert(0, {root!r})
import algan as A
from algan import Scene, SETTINGS
from algan.scene_manager import SceneManager
from algan.rendering.raytracing import path_tracer as pt
from algan.rendering.raytracing import wavefront_kernels_taichi as wf
from algan.rendering.raytracing import path_tracer_taichi as ptk
import benchmarks._pt_shadow_anyhit_check as c

seen = {{}}
_shade_params = ptk._pt_shade_launch.call_params
_i_mode = _shade_params.index("shadow_mode")
_trav_params = wf._wavefront_traverse_events_launch.call_params
_i_closest = _trav_params.index("opaque_closest")
_shade, _trav = pt.pt_shade, pt.wavefront_traverse_events


def _wrapped_shade(*a):
    seen["shadow_mode"] = int(a[_i_mode])
    meta = a[_shade_params.index("nee_meta")]
    seen["ambient_packed"] = float(meta[ptk._NM_AMBIENT_PACKED])
    seen["ambient_count"] = float(meta[ptk._NM_AMBIENT_COUNT])
    seen["animated_seed"] = float(meta[ptk._NM_ANIM_SEED])
    seen["nee_entries"] = float(meta[ptk._NM_COUNT])
    return _shade(*a)


def _wrapped_traverse(*a):
    seen["opaque_closest"] = int(a[_i_closest])
    return _trav(*a)


pt.pt_shade = _wrapped_shade
pt.wavefront_traverse_events = _wrapped_traverse

SceneManager.reset()
SETTINGS.raytracing.set(samples_per_pixel={samples}, shadows=True, denoise=False)
q = A.SMOKE_TEST.set(resolution={resolution})
SceneManager.instance().current_scene.set_video_settings(q)
c.build_scene()
# animate_fade_out=False, or the scene ends with everything fading: a
# primitive at alpha < 1 in ANY frame of the batch is a translucent one, and
# the batch stops being provably all-opaque -- which is exactly the
# classification both of the first two switches are gated on.
Scene.save_video({path!r}, q, overwrite=True, animate_fade_out=False)
# ... and the same frames again, RAW, for the per-frame comparisons. Rendered
# through the render loop's own entry point rather than save_frame: the frame
# index the sampler keys on is the one INSIDE the render job (render_loop
# passes ``current_ind - start_ind``), so a still rendered at t=0.5 is frame 0
# to the sampler and could not tell the two seed policies apart.
import numpy as np, torch
_scene = SceneManager.instance().current_scene
np.save(
    {path!r}[:-4] + "_raw.npy",
    torch.cat([f.cpu() for f in _scene.get_frames(0, {frames})]).numpy(),
)
print("PROBE " + json.dumps(seen))
"""


def _render(env_overrides, path):
    """One arm, in a fresh process. Returns what the kernels were handed."""
    env = dict(os.environ)
    env.update(env_overrides)
    # A warm daemon would carry the previous arm's import-time toggles.
    env["ALGAN_USE_DAEMON"] = "0"
    code = _CHILD.format(
        root=_REPO_ROOT,
        samples=SAMPLES,
        resolution=RESOLUTION,
        path=path,
        frames=FRAMES,
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # The arm's own traceback, not this harness's: a child that dies of a
        # scene-building or settings mistake says so on ITS stderr, and
        # swallowing that costs a whole debugging round.
        raise SystemExit(
            f"arm failed ({env_overrides}):\n{proc.stdout[-4000:]}\n"
            f"{proc.stderr[-4000:]}"
        )
    probe = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            probe = json.loads(line[len("PROBE ") :])
    return probe


def _frames(path):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame.copy())
    cap.release()
    return np.stack(out)


def _raw(path):
    """The arm's raw rendered frames, as ``[frames, ...]`` int32."""
    import numpy as np

    return np.load(path[:-4] + "_raw.npy").astype(np.int32)


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    import numpy as np

    wanted = sys.argv[1:] or list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; pick from {list(ARMS)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    base_path = os.path.join(OUT_DIR, "pt_switches_base.mp4")
    base_probe = _render({}, base_path)
    base = _frames(base_path)
    print(
        f"baseline: {base.shape[0]} frames {RESOLUTION[0]}x{RESOLUTION[1]} "
        f"spp={SAMPLES}  sha={_sha(base_path)[:12]}\n"
        f"  host handed the kernels: {base_probe}",
        flush=True,
    )
    if base_probe.get("shadow_mode") != 3:
        print(
            "  !! the baseline did NOT reach the opaque any-hit shadow mode; "
            "the shadow_anyhit arm below compares two identical paths",
            flush=True,
        )
    if not base_probe.get("opaque_closest"):
        print(
            "  !! the baseline did NOT reach closest-hit traversal; the "
            "opaque_closest arm below compares two identical paths",
            flush=True,
        )
    if not base_probe.get("ambient_count"):
        print(
            "  !! no ambient/hemisphere rows were packed; the ambient_rows "
            "arm below compares two identical paths",
            flush=True,
        )

    base_raw = _raw(base_path)
    failures = []
    for name in wanted:
        var, value, expectation = ARMS[name]
        path = os.path.join(OUT_DIR, f"pt_switches_{name}.mp4")
        probe = _render({var: value}, path)
        arm = _frames(path)
        if arm.shape != base.shape:
            failures.append(f"{name}: frame count/shape differs {arm.shape}")
            continue
        video_delta = [
            int(np.abs(arm[i].astype(np.int32) - base[i].astype(np.int32)).max())
            for i in range(arm.shape[0])
        ]
        arm_raw = _raw(path)
        raw_delta = [
            int(np.abs(arm_raw[i] - base_raw[i]).max())
            for i in range(min(arm_raw.shape[0], base_raw.shape[0]))
        ]
        if expectation == "same":
            bad = [i for i, d in enumerate(raw_delta) if d]
            ok = not bad
            verdict = "IDENTICAL" if ok else "DIFFERS"
        else:  # same_first: frame 0 pinned, a later frame must move
            bad = [0] if raw_delta[0] else []
            ok = not bad
            moved = any(d for d in raw_delta[1:])
            verdict = (
                "frame 0 IDENTICAL, later frames differ"
                if ok and moved
                else (
                    "frame 0 IDENTICAL, later frames did NOT move" if ok else "DIFFERS"
                )
            )
            if ok and not moved:
                failures.append(
                    f"{name}: no later frame changed, so the frame is not "
                    "reaching the sampler key at all"
                )
        print(
            f"{name} ({var}={value}): raw max |diff| {raw_delta}  "
            f"video max |diff| {video_delta}  {verdict}\n"
            f"  host handed the kernels: {probe}",
            flush=True,
        )
        if not ok:
            failures.append(
                f"{name}: raw frames {bad} differ (max {max(raw_delta)} of 255)"
            )

    print("\n" + ("all arms agree" if not failures else "FAILURES:"), flush=True)
    for f in failures:
        print("  " + f, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
