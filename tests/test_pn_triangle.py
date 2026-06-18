"""End-to-end comparison of the rasterized and ray traced render pipelines.

Run directly: python tests/test_raytracing_render.py

Renders the same animated scene through both pipelines, reports per-frame
PSNR between the two videos, and writes a side-by-side comparison image.
The two renderers are expected to differ slightly (screen-space vs exact
depth/interpolation, polyline sampling), so this checks for gross agreement
rather than pixel equality. Coplanar overlaps are deliberately avoided by
small z offsets: depth ties are inherently ambiguous and the two pipelines
break them differently.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (
    Sphere, Scene
)
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_v2")
SETTINGS = RenderSettings((640, 400), 10, anti_alias_level=2, fxaa=False)


def build_and_animate_scene():
    Sphere(grid_height=10, opacity=0.5).scale(3).spawn()
    #Scene.instance().save_frame('pn_sphere.png')


def render(name, ray_traced):
    if ray_traced:
        from algan.rendering.raytracing import enable_ray_tracing

        enable_ray_tracing(1, pn_triangles=True)
    else:
        try:
            from algan.rendering.raytracing import disable_ray_tracing

            disable_ray_tracing()
        except ImportError:
            pass
    build_and_animate_scene()
    render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                   render_settings=SETTINGS, file_extension="mp4")
    return os.path.join(OUT_DIR, f"{name}.mp4")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rt_path = render("raytraced", ray_traced=True)


if __name__ == "__main__":
    main()
