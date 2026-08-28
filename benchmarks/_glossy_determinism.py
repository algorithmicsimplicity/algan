"""Determinism and flicker checks for the glossy reflection lobe.

Three things a deterministic renderer must keep, and that a stochastic lobe
would each break:

  1. SAME FRAME TWICE is byte-identical. Rendered twice in one process, so a
     re-used kernel and a re-used arena are both exercised.
  2. A MOVING glossy object does not twinkle. The lobe's tap rotation is fixed
     in SCREEN space, so the worry is not noise but CRAWL: the pattern stands
     still while the object moves under it. Scored as the frame-to-frame
     difference of the reflected region, which for a smoothly moving object must
     itself vary smoothly -- an isolated spike is a frame that sampled the lobe
     differently from its neighbours.
  3. An ANIMATED ROUGHNESS sweep does not flicker: the same measure while the
     material -- not the object -- changes.

Reported against the SHARP arm (glossy off), which is the behaviour the tests
already baseline, so the numbers say how much the lobe adds rather than how
large they are in absolute terms.

Run: .venv/Scripts/python.exe benchmarks/_glossy_determinism.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    DOWN,
    GRAY_A,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    Off,
    RenderSettings,
    SceneManager,
    Seq,
    Sphere,
    Sync,
    Text,
    render_to_file,
)
from algan.rendering.lights import AmbientLight, DirectionalLight  # noqa: E402
from algan.rendering.raytracing import set_fragment_shading  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import MeshStandardMaterial  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)
W, H, FPS = 480, 270, 12


def _stage(roughness):
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 4 + UP * 4 + OUT * 3, color=WHITE, intensity=1.0
        ).spawn(animate=False)
        Text("MATERIAL STUDY", font_size=52, weight="BOLD", color=WHITE).move(
            UP * 1.55
        ).spawn()
        Text("Standard", font_size=24, color=GRAY_A).move(DOWN * 1.4).spawn()
        ball = Sphere(radius=0.48).scale(2.2)
        ball.set_material(
            MeshStandardMaterial(color=RED, roughness=roughness, metalness=0.75)
        )
        ball.spawn()
    return ball


def build_still(roughness):
    _stage(roughness)


def build_spin(roughness):
    ball = _stage(roughness)
    with Seq(), Sync(run_time=1.0):
        ball.rotate(60, UP)


def build_rough_sweep(_unused):
    ball = _stage(0.02)
    with Seq(), Sync(run_time=1.0):
        ball.roughness = 0.55


def render(builder, roughness, tag):
    path = os.path.join(OUT_DIR, f"glossyDet_{tag}.mp4")
    SceneManager.reset()
    set_fragment_shading(True)
    rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    builder(roughness)
    render_to_file(
        file_path=path,
        video_settings=RenderSettings((W, H), FPS, super_sampling_anti_aliasing=1),
    )
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float64))
    cap.release()
    return frames


def temporal_spikiness(frames):
    """How ragged the frame-to-frame change is.

    For smooth motion the per-frame mean |delta| should itself be a smooth
    sequence. Twinkling shows up as its ratio to its own median: a frame whose
    change is far out of line with its neighbours sampled the lobe differently.
    """
    d = np.array([np.abs(b - a).mean() for a, b in zip(frames, frames[1:])])
    if len(d) < 3 or np.median(d) < 1e-9:
        return float("nan"), d
    return float(d.max() / np.median(d)), d


def main():
    print("1. SAME FRAME TWICE (must be byte-identical)")
    for glossy in (False, True):
        rt_settings.set_glossy_reflection(glossy)
        a = render(build_still, 0.35, f"still_a_{glossy}")
        b = render(build_still, 0.35, f"still_b_{glossy}")
        same = len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
        print(f"   glossy={str(glossy):5s}  identical={same}")

    print("\n2/3. TEMPORAL STABILITY  (max/median of per-frame mean |delta|;")
    print("     a value near the sharp arm's means the lobe added no flicker)")
    print(f"{'scene':14s} {'glossy':>7s} {'spikiness':>10s}  per-frame deltas")
    for name, builder, rough in (
        ("spin r=0.35", build_spin, 0.35),
        ("roughness ramp", build_rough_sweep, 0.0),
    ):
        for glossy in (False, True):
            rt_settings.set_glossy_reflection(glossy)
            fr = render(builder, rough, f"{name.split()[0]}_{glossy}")
            sp, d = temporal_spikiness(fr)
            print(
                f"{name:14s} {str(glossy):>7s} {sp:10.2f}  "
                + " ".join(f"{v:.3f}" for v in d[:10])
            )
    rt_settings.set_glossy_reflection(True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
