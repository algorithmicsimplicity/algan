"""Visual repro / sweep for the roughness-driven glossy reflection lobe.

Renders ONE frame per roughness value: a metallic sphere with text above and
below it, so the reflected text is an asymmetric target (a centred sphere makes
the vertical flip ambiguous -- the title is at the TOP of the image, so it
reflects into the UPPER part of the sphere).

    .venv/Scripts/python.exe benchmarks/_glossy_sweep.py            # sweep
    .venv/Scripts/python.exe benchmarks/_glossy_sweep.py 0.18       # one value
    .venv/Scripts/python.exe benchmarks/_glossy_sweep.py --tag pre  # name frames

Frames land in ``benchmarks/_rt2_out/glossy_<tag>_r<roughness>.png``.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

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
    Sphere,
    Text,
    render_to_file,
)
from algan.rendering.lights import AmbientLight, DirectionalLight  # noqa: E402
from algan.rendering.raytracing import set_fragment_shading  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import MeshStandardMaterial  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 864, 486
DEFAULT_SWEEP = (0.0, 0.05, 0.18, 0.35, 0.6, 1.0)


def build(roughness):
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


def render(roughness, tag):
    name = f"glossy_{tag}_r{roughness:g}"
    path = os.path.join(OUT_DIR, name + ".mp4")
    SceneManager.reset()
    set_fragment_shading(True)
    rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    build(roughness)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    render_to_file(
        file_path=path,
        video_settings=RenderSettings((W, H), 1, super_sampling_anti_aliasing=1),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    import cv2

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    png = os.path.join(OUT_DIR, name + ".png")
    if ok:
        cv2.imwrite(png, frame)
    return png, dt, (frame.astype(np.float64) if ok else None)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "run"
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        args = [a for a in args if a != tag]
    values = [float(a) for a in args] or list(DEFAULT_SWEEP)
    prev = None
    for r in values:
        png, dt, frame = render(r, tag)
        line = f"roughness {r:5.2f}  {dt:6.2f}s  {os.path.basename(png)}"
        if frame is not None and prev is not None:
            line += f"   max|delta| vs previous {np.abs(frame - prev).max():6.1f}"
        prev = frame
        print(line, flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
