"""Acceptance harness for the default 3-D shading of a Mob with no material.

Renders one still of the same three solids under each of two lighting profiles
and writes them beside each other, so a change to what
``SETTINGS.style.default_material`` holds can be judged by looking at it rather
than by reading a diff:

* ``default`` -- Algan's own defaults, i.e. whatever
  ``SETTINGS.style.default_material`` is on a fresh import.
* ``manim`` -- after ``Scene.use_manim_defaults()``, which repoints the default
  material at Manim's own 3-D shading (``ManimMaterial`` /
  ``get_shaded_rgb``) along with Manim's camera, light and background.

None of the solids sets a material, which is the whole point: they are the
geometry that reads the default.

Run one arm per process::

    <venv-python> benchmarks/_default_shading_check.py default
    <venv-python> benchmarks/_default_shading_check.py manim

Two processes rather than one because several of the renderer's shading gates
are ``ti.static`` and resolve when the kernel first compiles (see ``CLAUDE.md``),
and because ``use_manim_defaults`` mutates process-global ``SETTINGS``.

Frames land in ``algan_outputs/default_shading_<arm>.png``.
"""

from __future__ import annotations

import sys

from algan import (
    BLUE,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    Cube,
    Scene,
    Sphere,
    Torus,
)

ARMS = ("default", "manim")


def build_scene():
    """Three solids with no material of their own, spread across the frame."""
    Sphere(radius=0.9, color=BLUE).move(LEFT * 2.2).spawn()
    Cube(size=1.4).move(ORIGIN).spawn()
    Torus(ring_radius=0.75, tube_radius=0.28).move(RIGHT * 2.2 + DOWN * 0.1).spawn()


def main(arm):
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {ARMS}, got {arm!r}")
    if arm == "manim":
        # Camera, background, coordinates and lighting all move together; the
        # shading half is the one under test.
        Scene.use_manim_defaults()
    # Neither arm adds a light. Each profile brings its own and they are part
    # of what is being compared: Algan's stock Scene carries one PointLight
    # near the camera (``default_scene_initializer``), and
    # ``use_manim_defaults`` clears that and installs Manim's, down and to the
    # left. Adding one here would have made the arms differ by the rig on top
    # of the shading.
    build_scene()
    result = Scene.save_frame(f"default_shading_{arm}")
    print(f"{arm}: {result.output_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "default")
