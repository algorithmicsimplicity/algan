"""Shading defects the sheet resolve review found, each pinned by a render.

Each case renders through the sheet route, once with one input corrupted, and
asserts on the frame:

* a clear glass sphere on a white background lost the light passing beside it
  at its silhouette, a dark one-pixel outline;
* a NaN sheet area painted its pixel opaque;
* an environment map brighter than 1 wrapped round in the byte frame buffer.

Feature tests of the resolve, which nothing outside it can break, so none is
marked fast.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from algan import BLACK, OUT, RED, WHITE, Scene, Sphere, Square, VideoSettings
from algan.rendering.shaders.materials import MeshPhysicalMaterial
from algan.settings import SETTINGS


def _luminance(path):
    """Mean of the encoded RGB channels, per pixel, as float."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    return im.mean(-1)


def test_clear_glass_on_white_has_no_dark_rim(tmp_path):
    """A clear sphere on a white background shows no dark outline.

    With no lights, every continuation a smooth white glass sphere spawns --
    reflection, refraction, and the light passing beside it in a partially
    covered silhouette pixel -- ends on the same white, so the sphere should
    all but vanish: a 4x supersampled render of it never drops below 250. The
    glass branch made the primary ray the reflection whenever
    ``alpha * R >= 1 - alpha``, which a grazing silhouette pixel usually
    satisfies, and dropped the pass-through. Darkest pixel at this size: 204,
    with 18 pixels below 230, before; 238 and none, after. What remains below
    250 is a smaller limb effect of its own.
    """
    SETTINGS.raytracing.set(shadows=False)
    glass = MeshPhysicalMaterial(
        color=WHITE, roughness=0.0, metalness=0.0, transmission=1.0, ior=1.5
    )
    out = tmp_path / "glass_on_white.png"
    with Scene() as scene:
        Sphere(radius=1.4).set_material(glass).spawn(animate=False)
        scene.save_frame(
            str(out), VideoSettings((192, 108), 1, supersampling=1), background=WHITE
        )
    lum = _luminance(out)
    dark = int((lum < 225).sum())
    assert dark == 0, (
        f"{dark} pixels of a clear sphere on white are darker than 225 "
        f"(darkest {lum.min():.0f}): the silhouette is dropping its pass-through"
    )


def test_a_nan_sheet_area_paints_nothing(tmp_path, monkeypatch):
    """A sheet whose exact area is NaN claims nothing.

    The resolve clamps an area with ``min(area, 1)``, and on the CPU backend
    that returns 1 for a NaN, so one bad area painted its pixel opaque. Nothing
    upstream produces one today (the compaction clamps its areas), so this hands
    the resolve NaN for EVERY sheet of a white square on black. Before, all 460
    covered pixels came out pure white, silhouette pixels included; now none is
    painted. The oracle, ``sheets.resolve_pixel_reference``, drops a NaN area
    the same way (``test_sheet_compaction``).
    """
    from algan.rendering.raytracing import raster_pipeline

    launch = raster_pipeline.shade_sparse_raster_coverage
    poisoned = []

    def with_nan_areas(coverage, *args, **kwargs):
        coverage["sheet_cov"].fill_(float("nan"))
        poisoned.append(int(coverage["sheet_cov"].numel()))
        return launch(coverage, *args, **kwargs)

    monkeypatch.setattr(raster_pipeline, "shade_sparse_raster_coverage", with_nan_areas)
    SETTINGS.raytracing.set(shadows=False)
    out = tmp_path / "nan_areas.png"
    with Scene() as scene:
        Square(color=WHITE).scale(1.5).rotate(17, OUT).spawn(animate=False)
        scene.save_frame(
            str(out), VideoSettings((96, 54), 1, supersampling=1), background=BLACK
        )
    assert max(poisoned, default=0) > 100, "the resolve never saw the square"
    lum = _luminance(out)
    painted = int((lum > 0).sum())
    assert painted == 0, f"{painted} pixels were painted by sheets with NaN areas"


def test_a_bright_environment_saturates_the_byte_buffer(tmp_path):
    """An environment map brighter than 1 shows as white on the byte buffer.

    With post-process tonemapping off the frame buffer is uint8, and the sheet
    route prefills it with the environment map, pixel by pixel. The prefill
    stored ``value * 255`` unclamped, so a map at 1.5 (382) wrapped to 126: a
    grey sky behind a scene lit by a white one. It now rounds and clamps the
    way the host's solid-colour prefill does. ``linear_color_space`` has to be
    off as well, because the linear space refuses the byte buffer outright.
    """
    SETTINGS.raytracing.set(shadows=False, linear_color_space=False, tonemapping=False)
    SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
    out = tmp_path / "bright_sky.png"
    with Scene() as scene:
        scene.set_environment_map(torch.full((4, 8, 3), 1.5), ambient=False)
        Sphere(radius=0.5, color=RED).spawn(animate=False)
        scene.save_frame(str(out), VideoSettings((64, 36), 1, supersampling=1))
    im = np.asarray(Image.open(out).convert("RGB"))
    corners = im[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert (corners == 255).all(), (
        f"the sky's corners read {corners.tolist()}, not white: the "
        "environment prefill wrapped in the byte buffer"
    )
