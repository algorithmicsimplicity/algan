"""Render/scene settings at unusual values: degenerate inputs must fail with a
clear configuration error rather than an obscure crash deep in the renderer."""

import pytest

from algan.errors import AlganConfigurationError
from algan.settings.video_settings import SMOKE_TEST


@pytest.mark.parametrize("resolution", [(1, 1), (32, 1), (1, 32)])
def test_one_pixel_side_is_refused_with_a_clear_error(resolution):
    # A 1-pixel side made the renderer's half-screen ``side // 2`` zero, and
    # save_video crashed with ZeroDivisionError in the raster pass.
    with pytest.raises(AlganConfigurationError, match="at least 2 pixels"):
        SMOKE_TEST.set(resolution=resolution, supersampling=1)


def test_one_pixel_side_is_fine_once_supersampled():
    settings = SMOKE_TEST.set(resolution=(1, 1), supersampling=2)
    assert settings.resolution == (1, 1)


def test_smallest_and_odd_resolutions_are_accepted():
    assert SMOKE_TEST.set(resolution=(2, 2), supersampling=1).resolution == (2, 2)
    assert SMOKE_TEST.set(resolution=(33, 17)).resolution == (33, 17)


def test_odd_resolution_still_is_centred(tmp_path):
    # The tracer used ``side // 2`` as the half-screen, so an odd-sized frame
    # was projected half a pixel off centre (and slightly too small).
    import numpy as np
    from PIL import Image

    from algan import Scene, Square

    with Scene() as scene:
        scene.set_video_settings(SMOKE_TEST.set(resolution=(33, 17), supersampling=1))
        Square().spawn(animate=False)
        result = scene.save_frame(tmp_path / "odd.png")
    image = np.asarray(Image.open(result.output_path)).astype(float)[..., :3]
    luminance = image.sum(-1)
    rows, cols = np.nonzero(luminance > luminance.min() + 30)
    assert (rows.min() + rows.max()) / 2 == 8
    assert (cols.min() + cols.max()) / 2 == 16
