"""Direction constants and Algan's coordinate conventions.

``RIGHT``/``LEFT`` run along +x/-x, ``UP``/``DOWN`` along +y/-y, and
``IN``/``OUT`` along **+z/-z**: ``OUT`` points out of the screen towards the
viewer, so it is ``(0, 0, -1)`` and the +z axis runs *away* from the viewer,
into the scene. ``CAMERA_ORIGIN`` is correspondingly at negative z. This is the
opposite of Three.js's and glTF's sign convention, so a scene ported from
either has every z coordinate (and every z direction) negated. Each constant is
a unit tensor of shape ``(1, 1, 3)``, so they compose by ordinary arithmetic:
``UP * 2 + LEFT``.

``ORIGIN`` is the zero vector, ``DEFAULT_BASIS`` the identity orientation every
Mob starts with, and ``CAMERA_ORIGIN`` where a new Scene's camera sits.

Distances are in world units throughout; angles are in **degrees**, which is the
convention that most often surprises users arriving from Manim.
"""

from __future__ import annotations

from algan.constants.color import *

RIGHT = torch.tensor((1, 0, 0), dtype=torch.get_default_dtype())
LEFT = -RIGHT
UP = torch.tensor((0, 1, 0), dtype=torch.get_default_dtype())
DOWN = -UP
IN = torch.tensor((0, 0, 1), dtype=torch.get_default_dtype())
OUT = -IN

DEFAULT_BASIS = torch.stack((RIGHT, UP, OUT))

ORIGIN = torch.zeros_like(OUT)
CAMERA_ORIGIN = ORIGIN + OUT * 7

NUM_DIMENSIONS = 3
