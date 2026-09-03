"""Stand-ins for Manim's animation system, which Algan does not vendor.

Algan records and plays its own animations, and a Manim ``Animation`` has no
way to reach a frame here: the vendored subset ships neither ``Scene`` nor a
renderer. A handful of geometry classes still *name* animation classes --
``Brace.creation_anim``, ``Table.create``, ``ManimBanner.create``,
``StreamLines.create``, and ``Graph``'s ``@override_animation(Create)`` -- so
the names exist, the override registry still works, and constructing one
points at :mod:`algan.animations` instead.
"""

from .animation import Animation, override_animation
from .composition import AnimationGroup, LaggedStart, Succession
from .creation import Create, SpiralIn, Uncreate, Write
from .fading import FadeIn, FadeOut
from .growing import GrowFromCenter, GrowFromPoint, SpinInFromNothing
from .indication import ShowPassingFlash
from .transform import Transform, _MethodAnimation
from .updaters.update import UpdateFromAlphaFunc, UpdateFromFunc

__all__ = [
    "Animation",
    "AnimationGroup",
    "Create",
    "FadeIn",
    "FadeOut",
    "GrowFromCenter",
    "GrowFromPoint",
    "LaggedStart",
    "ShowPassingFlash",
    "SpinInFromNothing",
    "SpiralIn",
    "Succession",
    "Transform",
    "Uncreate",
    "UpdateFromAlphaFunc",
    "UpdateFromFunc",
    "Write",
    "override_animation",
]
