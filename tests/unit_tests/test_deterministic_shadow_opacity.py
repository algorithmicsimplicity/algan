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


def _to_linear(byte_value):
    """Decode a 0-255 luminance back to linear light.

    Written out from the sRGB specification rather than imported from
    ``algan.utils.color_space``, so this test measures the renderer against the
    standard instead of against the renderer's own transcription of it.
    """
    c = byte_value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _render_shadow_luminance(tmp_path, name, blocker_opacities):
    output_path = tmp_path / name
    SceneManager.reset()
    with Scene(video_settings=SHADOW_TEST_SETTINGS) as scene:
        scene.set_background(BLACK)
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
    #
    # Selected relative to the darkest pixel in the centre rather than against
    # an absolute byte. A fully shadowed surface sits at the ambient floor, and
    # where that floor lands depends on the working space: ambient_strength is
    # 0.1 either way, but 0.1 of linear light encodes to byte 89 where 0.1 of
    # an encoded value is byte 26. An absolute "< 80" silently selected nothing
    # at all under the linear space.
    floor = float(opaque_image[center].min())
    shared_shadow = (
        center & (opaque_image <= floor + 2.0) & (one_image - opaque_image > 10.0)
    )
    assert shared_shadow.sum() >= 9

    one_blocker = float(one_image[shared_shadow].mean())
    two_blockers = float(two_image[shared_shadow].mean())
    opaque_blocker = float(opaque_image[shared_shadow].mean())
    assert one_blocker > two_blockers > opaque_blocker

    # A half-opaque second blocker must land the shadow exactly halfway between
    # the one-blocker and opaque cases -- that is what "accumulates every
    # blocker's opacity" means.
    #
    # The halfway point is a statement about *light*, so it has to be measured
    # in the space the renderer composes in. It used to be asserted on the
    # output bytes, which only worked because there was no transfer function:
    # shading was linear in encoded values, so the encoded midpoint was the
    # light midpoint by accident. With the linear working space the OETF is
    # concave and the two part company -- measured on this scene, the byte
    # midpoint is off by 18.6 while the linear one is off by 0.0027 (0.68 of a
    # byte). Decoding first states the invariant that was always intended, and
    # it holds in both arms: 0.42 bytes of error with the linear space off,
    # 0.68 with it on.
    if SETTINGS.raytracing.linear_color_space:
        one_blocker, two_blockers, opaque_blocker = (
            float(_to_linear(v)) for v in (one_blocker, two_blockers, opaque_blocker)
        )
        tolerance = 4.0 / 255.0
    else:
        tolerance = 4.0

    assert two_blockers == pytest.approx(
        0.5 * (one_blocker + opaque_blocker),
        abs=tolerance,
    )
