"""Focused render regression for deterministic translucent shadows."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from algan import (
    BLACK,
    OUT,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    WHITE,
    MeshLambertMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    Square,
)
from algan.scene_manager import SceneManager

SHADOW_TEST_SETTINGS = SMOKE_TEST.set(resolution=(64, 64))


def _render_shadow_luminance(tmp_path, name, blocker_opacities):
    output_path = tmp_path / name
    SceneManager.reset()
    with Scene(video_settings=SHADOW_TEST_SETTINGS) as scene:
        scene.set_background_color(BLACK)
        with Off():
            PointLight(
                location=RIGHT * 3.0 + OUT * 3.0,
                color=WHITE,
                intensity=0.08,
            ).spawn(animate=False)
            (
                Prism(dimensions=(6.0, 6.0, 0.1))
                .set_material(MeshLambertMaterial(color=WHITE))
                .spawn(animate=False)
            )
            for opacity, distance in zip(blocker_opacities, (1.0, 1.1)):
                (
                    Square(side_length=1.2, color=WHITE, opacity=opacity)
                    .move(RIGHT * distance + OUT * distance)
                    .spawn(animate=False)
                )

        result = scene.save_frame(
            output_path,
            video_settings=SHADOW_TEST_SETTINGS,
            overwrite=True,
        )

    with Image.open(result.output_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return pixels.mean(axis=2)


@pytest.mark.parametrize("hybrid_raster", [True, False], ids=("raster", "wavefront"))
def test_deterministic_shadows_accumulate_every_blocker_opacity(
    tmp_path,
    hybrid_raster,
):
    snapshot = SETTINGS.snapshot()
    try:
        # ambient_light is not set here: it is inert (no renderer this build
        # can launch reads it) and writing it now raises rather than pretending.
        SETTINGS.raytracing.set(
            shadows=True,
            tonemapping=False,
        )
        SETTINGS.raytracing.experimental.set(hybrid_raster=hybrid_raster)
        one_image = _render_shadow_luminance(tmp_path, "one", (0.25, 0.0))
        two_image = _render_shadow_luminance(tmp_path, "two", (0.25, 0.5))
        opaque_image = _render_shadow_luminance(tmp_path, "opaque", (1.0, 0.0))
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    h, w = one_image.shape
    center = np.zeros_like(one_image, dtype=bool)
    center[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8] = True
    # Keep only the opaque blocker's fully covered interior. This excludes the
    # two shadow silhouettes' slightly different antialiased boundaries.
    shared_shadow = center & (opaque_image < 80.0) & (one_image - opaque_image > 30.0)
    assert shared_shadow.sum() >= 9

    one_blocker = float(one_image[shared_shadow].mean())
    two_blockers = float(two_image[shared_shadow].mean())
    opaque_blocker = float(opaque_image[shared_shadow].mean())
    assert one_blocker > two_blockers > opaque_blocker
    assert two_blockers == pytest.approx(
        0.5 * (one_blocker + opaque_blocker),
        abs=4.0,
    )
