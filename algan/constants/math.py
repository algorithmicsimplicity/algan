from __future__ import annotations

import math

PI = math.pi
TAU = PI * 2

DEGREES: float = 1.0
"""Unit multiplier for an angle already written in degrees, Algan's native unit.

Algan's angle arguments are in degrees, so this is 1 and ``rotate(180)`` and
``rotate(180 * DEGREES)`` are the same call. Write the multiplier when you want the
unit to be explicit at the call site: ``square.rotate(180 * DEGREES)``.

Note this is the reciprocal of Manim's constant of the same name -- Manim measures
angles in radians, so its ``DEGREES`` converts *to* radians.
"""

RADIANS: float = 180 / PI
"""Multiplier converting an angle written in radians to Algan's degrees.

Use it when you think in radians: ``square.rotate(PI * RADIANS)`` turns half a
circle, exactly as ``square.rotate(180)`` does.
"""

# Internal degree/radian boundary conversions, kept distinct from the user-facing
# DEGREES / RADIANS multipliers above. RADIANS_TO_DEGREES aliases RADIANS on
# purpose; algan.geometry.geometry and algan.animations.indication consume it.
RADIANS_TO_DEGREES = RADIANS
DEGREES_TO_RADIANS = PI / 180

KILOBYTES = 1000
MEGABYTES = KILOBYTES * KILOBYTES
GIGABYTES = MEGABYTES * KILOBYTES
