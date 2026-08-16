"""Direction constants and Algan's coordinate conventions.

Algan uses a right-handed 3-D coordinate system. ``RIGHT``/``LEFT`` run along
+x/-x, ``UP``/``DOWN`` along +y/-y, and ``OUT``/``IN`` along +z/-z -- ``OUT``
points out of the screen, towards the viewer. Each is a unit tensor of shape
``(1, 1, 3)``, so they compose by ordinary arithmetic: ``UP * 2 + LEFT``.

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
