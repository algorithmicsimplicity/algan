"""Direction constants and Algan's coordinate conventions.

``RIGHT``/``LEFT`` run along +x/-x, ``UP``/``DOWN`` along +y/-y, and
``INWARD``/``OUTWARD`` along **+z/-z**: ``OUTWARD`` points out of the screen
towards the viewer, so it is ``(0, 0, -1)`` and the +z axis runs *away* from the
viewer, into the scene. ``CAMERA_ORIGIN`` is correspondingly at negative z. This
is the opposite of Three.js's and glTF's sign convention, so a scene ported from
either has every z coordinate (and every z direction) negated. Each constant is
a unit tensor of shape ``(1, 1, 3)``, so they compose by ordinary arithmetic:
``UP * 2 + LEFT``.

``IN`` and ``OUT`` are those same two vectors under shorter names -- the same
objects, not copies -- and are what most scripts write; ``OUT`` is what Manim
calls it too. They are the one place Algan deliberately carries two names for
one thing. ``in`` and ``out`` are such ordinary words that a script is liable to
want them for something of its own, and a name the library reads is a poor thing
to leave lying in the way. So Algan's own source says ``INWARD`` and ``OUTWARD``
throughout and never reads ``IN`` or ``OUT`` (enforced by
``tests/unit_tests/test_spatial_constants.py``), and the short names are the
script's to keep or to shadow.

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
INWARD = torch.tensor((0, 0, 1), dtype=torch.get_default_dtype())
OUTWARD = -INWARD

#: Shorthands for the two z directions. See the module docstring: these are the
#: names to write in a script, and the ones Algan's own source never reads.
IN = INWARD
OUT = OUTWARD

DEFAULT_BASIS = torch.stack((RIGHT, UP, OUTWARD))

ORIGIN = torch.zeros_like(OUTWARD)
CAMERA_ORIGIN = ORIGIN + OUTWARD * 7

NUM_DIMENSIONS = 3
