"""A mob's authored colour survives its lighting rig.

``default_shader`` -- the shader every mob that was never given a material
falls to -- tints the albedo toward the light colours rather than multiplying
by them. That is a stylistic choice and not in question here. What *is* in
question is that the tint has to leave the albedo behind it: a
``Line3D(color=MAGENTA)`` must render magenta, not white, however many lights
the Scene has.

It did not. The in-kernel stage summed a per-light fade weight and clamped the
total at 1, and an ambient-like light (ambient / hemisphere / environment SH)
contributed the *maximum* weight of 0.5 whatever its intensity, because its
shading direction is the surface normal so ``n . l`` is 1. Two of them summed
to exactly 1, the albedo term was multiplied by ``1 - 1``, and every
default-shaded mob in the Scene rendered as pure light colour -- indifferent to
what the author had asked for. It reads as "one ``set_material`` silenced
everyone else's ``color=``" because the mobs that *do* carry a material shade
through a different stage and keep their colour.

These are renders rather than tensor assertions because the arithmetic under
test lives in a Taichi kernel and only a render reaches it.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from algan import (
    BLACK,
    BLUE_A,
    GRAY_A,
    LEFT,
    MAROON_E,
    OUT,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    AmbientLight,
    HemisphereLight,
    Line3D,
    Off,
    PointLight,
    Scene,
)
from algan.scene_manager import SceneManager

ALBEDO_TEST_SETTINGS = SMOKE_TEST.set(resolution=(64, 64))

#: The rig from ``tests/full_renders/scenes/solids_and_camera.py``, trimmed to
#: the two ambient-like lights that saturated the fade. Their intensities are
#: deliberately low: under the old rule intensity did not enter the fade at
#: all, so a pair this dim still erased the albedo completely.
_AMBIENT_INTENSITY = 0.45
_HEMISPHERE_INTENSITY = 0.3


def _two_ambient_lights():
    AmbientLight(color=WHITE, intensity=_AMBIENT_INTENSITY).spawn(animate=False)
    HemisphereLight(
        color=BLUE_A,
        ground_color=MAROON_E,
        intensity=_HEMISPHERE_INTENSITY,
    ).spawn(animate=False)


def _one_point_light():
    PointLight(
        location=RIGHT * 5.0 + UP * 1.0 + OUT * 5.0,
        color=WHITE,
    ).spawn(animate=False)


def _render_bar(tmp_path, name, color, lights):
    """Render one default-shaded bar of ``color`` and return its mean RGB.

    A ``Line3D`` rather than a ``Polyhedron``: a polyhedron's faces carry their
    own styling, which would stand between ``color=`` and the pixel and make a
    failure ambiguous. The Scene's own lights are cleared first so the rig under
    test is the whole rig -- ``Scene`` seeds one point light of its own.
    """
    output_path = tmp_path / name
    SceneManager.reset()
    with Scene(video_settings=ALBEDO_TEST_SETTINGS) as scene:
        scene.set_background(BLACK)
        scene.clear_light_sources()
        with Off():
            lights()
            Line3D(
                start=LEFT * 6,
                end=RIGHT * 6,
                thickness=2.0,
                color=color,
            ).spawn(animate=False)
        scene.save_frame(str(output_path), overwrite=True)
    frame = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.float64)
    # The bar spans the frame horizontally through the middle; keep well
    # inside it so no antialiased edge pixel or background sliver is averaged.
    return frame[30:34, 8:56].reshape(-1, 3).mean(axis=0)


@pytest.mark.parametrize("lights", [_one_point_light, _two_ambient_lights])
def test_authored_hue_survives(tmp_path, lights):
    """A red mob renders red -- under one light and under two ambient ones."""
    rgb = _render_bar(tmp_path, f"hue_{lights.__name__}.png", RED, lights)
    assert rgb[0] > max(rgb[1], rgb[2]) + 30, (
        f"a RED mob rendered {tuple(rgb.round(1))}: the authored hue was lost"
    )


def test_two_ambient_lights_keep_albedos_apart(tmp_path):
    """Two mobs of different colours do not collapse onto the same colour.

    The sharpest form of the defect: ``GRAY_A`` (#DDDDDD) and ``WHITE`` are
    close enough that "it renders white" is easy to miss by eye, and under the
    old rule they rendered byte-identically because neither albedo reached the
    frame at all.
    """
    gray = _render_bar(tmp_path, "gray.png", GRAY_A, _two_ambient_lights)
    white = _render_bar(tmp_path, "white.png", WHITE, _two_ambient_lights)
    assert white.mean() - gray.mean() > 4.0, (
        f"GRAY_A rendered {tuple(gray.round(1))} and WHITE {tuple(white.round(1))}: "
        "two distinct albedos collapsed onto one colour"
    )
