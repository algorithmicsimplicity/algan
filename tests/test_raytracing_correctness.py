"""Geometric correctness checks for the renderers.

Run directly: python tests/test_raytracing_correctness.py [raster]

1. A triangulated image grid (ImageMob) must not show cracks along the
   shared diagonal edges of its quads (rays/pixels exactly on an edge must
   hit at least one of the adjacent triangles).
2. A sphere centered on the camera's orbit axis must keep a circular
   silhouette while the camera orbits (the projection must stay isotropic
   at every camera orientation).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from algan import ORIGIN, UP, ImageMob, Sphere
from algan.constants.color import PURE_GREEN
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_v2")
SETTINGS = RenderSettings((480, 300), 10, anti_alias_level=1, fxaa=False)
USE_RASTERIZER = "raster" in sys.argv[1:]


def _enable_renderer():
    from algan import SceneManager

    SceneManager.reset()
    if not USE_RASTERIZER:
        from algan.rendering.raytracing import enable_ray_tracing

        enable_ray_tracing()


def _grab(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame {index} of {path}"
    return frame.astype(np.float64)


def _render_uniform_image(name, alpha):
    """Render a uniform white image with the given opacity over a blue
    background, returning the strict interior of the image region.
    """
    from algan.constants.color import BLUE

    _enable_renderer()
    image = torch.ones(24, 24, 4)
    image[..., 3] = alpha
    mob = ImageMob(image).spawn(animate=False)
    mob.scale(1.6)
    mob.wait(0.3)
    render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                   render_settings=SETTINGS, file_extension="mp4",
                   background_color=BLUE)
    frame = _grab(os.path.join(OUT_DIR, f"{name}.mp4"), 1)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_frame.png"),
                frame.astype(np.uint8))
    # The (partially) white image is the region brighter than the blue
    # background in the red channel (BGR index 2).
    bright = frame[..., 2] > 120
    ys, xs = np.nonzero(bright)
    y0, y1 = ys.min() + 3, ys.max() - 2
    x0, x1 = xs.min() + 3, xs.max() - 2
    return frame[y0:y1, x0:x1]


def test_image_grid_has_no_cracks():
    """An opaque uniform white image must contain no dark pixels: a dark
    pixel inside it is a crack between the triangles of its grid (a ray on
    the shared diagonal edge missing both adjacent triangles).
    """
    name = "cracks_raster" if USE_RASTERIZER else "cracks_rt"
    interior = _render_uniform_image(name, alpha=1.0)
    cracks = int((interior.min(-1) < 160).sum())
    print(f"{name}: {cracks} crack pixels inside {interior.size // 3} "
          f"interior pixels")
    assert cracks == 0, f"image grid shows {cracks} crack pixels"
    print("ok: no cracks in the triangulated image grid")


def test_transparent_image_grid_is_uniform():
    """A half-transparent uniform image over a colored background must be
    uniform: seams must neither leave cracks (background showing through)
    nor blend twice (a brighter line where a ray interacted with both
    triangles adjacent to the shared edge).
    """
    if USE_RASTERIZER:
        # The rasterizer's fragment pipeline has no seam deduplication; its
        # coverage epsilon is kept tiny instead. Only the ray tracer makes
        # this guarantee.
        print("skip: transparent seam uniformity (rasterizer)")
        return
    name = "cracks_rt_transparent"
    interior = _render_uniform_image(name, alpha=0.5)
    median = np.median(interior, axis=(0, 1))
    deviation = np.abs(interior - median).max(-1)
    bad = int((deviation > 12).sum())
    print(f"{name}: median {median.astype(int).tolist()}, "
          f"{bad} non-uniform pixels (max deviation {deviation.max():.0f})")
    assert bad == 0, (
        f"semi-transparent image grid is not uniform across seams "
        f"({bad} pixels deviate by more than 12)")
    print("ok: semi-transparent image grid blends exactly once per pixel")


def test_sphere_circular_under_orbit():
    """A sphere on the orbit axis must keep a ~1:1 silhouette while the
    camera orbits around it.
    """
    _enable_renderer()
    sphere = Sphere(radius=0.8, color=PURE_GREEN).spawn(animate=False)
    sphere.scene.camera.orbit_around_line(ORIGIN, UP, 45)
    name = "orbit_sphere_raster" if USE_RASTERIZER else "orbit_sphere_rt"
    render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                   render_settings=SETTINGS, file_extension="mp4")
    path = os.path.join(OUT_DIR, f"{name}.mp4")
    worst = 1.0
    for index in [0, 4, 9]:
        frame = _grab(path, index)
        mask = frame[..., 1] > 60  # green silhouette
        ys, xs = np.nonzero(mask)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        aspect = width / height
        worst = max(worst, max(aspect, 1 / aspect))
        print(f"{name} frame {index}: silhouette {width}x{height} "
              f"(aspect {aspect:.3f})")
    assert worst < 1.06, (
        f"sphere silhouette distorts under camera orbit (aspect {worst:.3f})")
    print("ok: sphere stays circular while the camera orbits")


if __name__ == "__main__":
    test_image_grid_has_no_cracks()
    test_transparent_image_grid_is_uniform()
    test_sphere_circular_under_orbit()
    print("all correctness tests passed")
