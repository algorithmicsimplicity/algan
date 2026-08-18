"""A/B for scaling a capped fragment's occlusion write (``ALGAN_ANALYTIC_AA_ONE_MESH_DENS``).

``DESIGN_mesh_identity.md`` ss6.6.2 diagnoses a defect inside the shipped
one-mesh cap: the cap clips a fragment's CLAIM (``eff``) while the per-sample
transmittance write keeps using the uncapped ``dens``, so a PARTIALLY capped
fragment hides more background than it paints and the pixel loses that energy.
``_aa_run_gate_check`` scores that directly, and the fix takes its
claim-vs-occlusion column from 2.2e-01 to float dust on all eleven cases.

What that harness CANNOT answer is what the fix looks like, and there is a real
question there rather than a formality. The residual claim shortfall is NOT
removed by this fix -- ss6.6.2 predicted it would be and the measurement refuted
that, because the harness scores notches on ``actual``, which is the claim. So an
interior pixel that the ceiling over-bites used to render too DARK (it painted
0.95 and hid 1.00) and now renders with 5% of the background showing through
instead. Against a dark background that is invisible; against a bright one it is
the same error wearing different clothes, and it has to be looked at rather than
argued about.

Hence the background arm below: the same solids over ``DARKER_GRAY`` and over
``WHITE``. A fix that is neutral on one and ugly on the other is not ready.

Both arms hold ``ANALYTIC_AA_ONE_MESH`` ON, so this isolates ss6.6.2 from ss6.6.
Byte-identity is the wrong gate (the whole point is that the occlusion write
moves), so the delta is reported as magnitude. The A/A arm is not decoration:
the cap's ceiling had a reproducibility bug found exactly this way (ss6.6.4), and
this change feeds the same threshold, so "twice, same settings, compare" runs
before any number here is trusted.

Usage:
    .venv/Scripts/python.exe benchmarks/_one_mesh_dens_ab.py [reps] [quality]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from benchmarks._one_mesh_ab import (  # noqa: E402
    PINNED_BYTES,
    build_scene,
    read_frames,
)

OUT_DIR = os.path.join("algan_outputs", "one_mesh_dens_ab")

#: (shape, background). ``cylfine`` is ss6.6's pathological case -- a
#: 0.045-radius rod diced 256x, nearly all boundary -- and carries 253 of the 257
#: residual notches, so it is where a changed notch CHARACTER would show.
ARMS = (
    ("diced", "dark"),
    ("diced", "white"),
    ("mixed", "dark"),
    ("cylfine", "white"),
)

BACKGROUNDS = {"dark": DARKER_GRAY, "white": WHITE}


def build_cylfine():
    """ss6.6's fine rod: sub-pixel facets, nearly every pixel a boundary."""
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        for i, x in enumerate((-2.0, 0.0, 2.0)):
            rod = Cylinder(
                radius=0.045, height=3.0, color=RED, resolution=(256, 2)
            ).move(RIGHT * x)
            rod.rotate(33 + 12 * i, OUT)
            rod.spawn(animate=False)
    with Sync(run_time=2):
        Scene.get_camera().move(RIGHT * 0.4 + UP * 0.2)


def render_once(shape, background, dens, quality, tag=""):
    name = f"dens_{shape}_{background}_{'on' if dens else 'off'}{tag}"
    path = os.path.join(OUT_DIR, f"{name}.mp4")
    rt_settings.set_analytic_aa(True, one_mesh=True, one_mesh_dens=bool(dens))
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    if shape == "mixed":
        SETTINGS.raytracing.set(shadows=True)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    Scene.set_background_color(BACKGROUNDS[background])
    if shape == "cylfine":
        build_cylfine()
    else:
        build_scene(shape)
        Scene.set_background_color(BACKGROUNDS[background])
    t0 = time.perf_counter()
    Scene.save_video(path, quality, overwrite=True)
    return path, time.perf_counter() - t0


def _delta(a_path, b_path):
    a, b = read_frames(a_path), read_frames(b_path)
    n = min(len(a), len(b))
    delta = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
    moved = int((delta.max(axis=-1) > 2).sum())
    px = int(delta.shape[0] * delta.shape[1] * delta.shape[2])
    return int(delta.max()), moved, px


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    quality = globals()[sys.argv[2]] if len(sys.argv) > 2 else MD
    # Third argument filters the arms, e.g. "mixed" for the cost question alone:
    # ss6.6.3's reasoning applies here too, that only the seconds-long shadowed
    # scene measures the clamp rather than fixed per-render overhead.
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    arms = [a for a in ARMS if not want or a[0] in want]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"quality={quality.__class__.__name__} reps={reps}", flush=True)
    for shape, background in arms:
        t_off, t_on = [], []
        for rep in range(reps):
            # ALTERNATE THE ORDER, do not merely interleave. A fixed off,on,off,on
            # gives one arm the cooler slot in every pair, and this box throttles:
            # the memory note "algan measurement traps" records an 8-16% uniform
            # bias produced exactly this way, on a control kernel the change could
            # not touch. Odd reps run on-first so the thermal ramp lands on both.
            order = (False, True) if rep % 2 == 0 else (True, False)
            for dens in order:
                _p, dt = render_once(shape, background, dens, quality)
                (t_on if dens else t_off).append(dt)
        off_path = os.path.join(OUT_DIR, f"dens_{shape}_{background}_off.mp4")
        on_path = os.path.join(OUT_DIR, f"dens_{shape}_{background}_on.mp4")
        # A/A on the ON arm: the same configuration rendered twice. Any movement
        # here is nondeterminism and invalidates the A/B column beside it.
        aa_path, _dt = render_once(shape, background, True, quality, tag="_aa")
        aa_max, aa_moved, _px = _delta(on_path, aa_path)
        mx, moved, px = _delta(off_path, on_path)
        keep_off = t_off[1:] if len(t_off) > 1 else t_off
        keep_on = t_on[1:] if len(t_on) > 1 else t_on
        print(
            f"{shape:8s} {background:5s}: A/A max|d|={aa_max} px>2={aa_moved} | "
            f"A/B max|d|={mx:3d} px>2={moved} of {px} ({moved / max(px, 1):.3%}) "
            f"off={min(keep_off):6.2f}s on={min(keep_on):6.2f}s "
            f"ratio={min(keep_on) / min(keep_off):5.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
