"""Direction constants and Algan's coordinate conventions.

``RIGHT``/``LEFT`` run along +x/-x, ``UP``/``DOWN`` along +y/-y, and
``OUTWARD``/``INWARD`` along **+z/-z**: ``OUTWARD`` points out of the screen
towards the viewer, so it is ``(0, 0, 1)`` and the +z axis runs *towards* the
viewer, out of the scene. ``CAMERA_ORIGIN`` is correspondingly at positive z.
This is the same sign convention as Manim, Three.js and glTF, so a scene ported
from any of them keeps its z coordinates as written. Each constant is a unit
tensor of shape ``(3,)``, so they compose by ordinary arithmetic:
``UP * 2 + LEFT``. (``DEFAULT_BASIS``, being a matrix, is ``(3, 3)``.)

``(RIGHT, UP, OUTWARD)`` is therefore a right-handed basis, and a rotation of a
positive angle about an axis is counter-clockwise seen from the tip of that
axis: ``rotate(90, OUTWARD)`` turns anti-clockwise on screen.

``IN`` and ``OUT`` are those same two vectors under shorter names -- the same
objects, not copies -- and are what most scripts write; ``OUT`` is what Manim
calls it too. They are the one place Algan deliberately carries two names for
one thing. ``in`` and ``out`` are such ordinary words that a script is liable to
want them for something of its own, and a name the library reads is a poor thing
to leave lying in the way. So Algan's own source says ``INWARD`` and ``OUTWARD``
throughout and never reads ``IN`` or ``OUT`` (enforced by
``tests/unit_tests/test_spatial_constants.py``), and the short names are the
script's to keep or to shadow.

``ORIGIN`` is the zero vector, ``DEFAULT_BASIS`` the orientation every Mob
starts with -- facing ``OUTWARD``, towards the viewer -- and ``CAMERA_ORIGIN``
where a new Scene's camera sits, looking ``INWARD`` at the Mobs that face it.

Distances are in world units throughout; angles are in **degrees**, which is the
convention that most often surprises users arriving from Manim.
"""

from __future__ import annotations

from algan.constants.color import *

RIGHT = torch.tensor((1, 0, 0), dtype=torch.get_default_dtype())
LEFT = -RIGHT
UP = torch.tensor((0, 1, 0), dtype=torch.get_default_dtype())
DOWN = -UP
OUTWARD = torch.tensor((0, 0, 1), dtype=torch.get_default_dtype())
INWARD = -OUTWARD

#: Shorthands for the two z directions. See the module docstring: these are the
#: names to write in a script, and the ones Algan's own source never reads.
IN = INWARD
OUT = OUTWARD

#: The orientation every Mob starts with: ``RIGHT``, ``UP`` and a forward axis
#: that points ``OUTWARD``, **towards** the viewer -- so a new Mob faces the
#: camera, the way a model's front faces +z in Three.js and glTF. It is the
#: identity matrix, and it is right-handed, like ``(RIGHT, UP, OUTWARD)``
#: itself. This is what ``Mob.__init__`` defaults ``basis`` to, and what
#: ``get_forward_direction()`` returns for an unrotated Mob.
#:
#: A camera is the one thing built the other way round: its forward axis is
#: where it looks, so it starts at ``(RIGHT, UP, INWARD)`` -- ``_CAMERA_BASIS``
#: in ``algan/rendering/camera.py`` -- and a Mob at the default orientation
#: faces it.
DEFAULT_BASIS = torch.stack((RIGHT, UP, OUTWARD))

ORIGIN = torch.zeros_like(OUTWARD)
CAMERA_ORIGIN = ORIGIN + OUTWARD * 7

NUM_DIMENSIONS = 3
