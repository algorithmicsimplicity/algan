"""Manim's Mobjects, by Manim's conventions, in a namespace that says so.

Algan's own classes and Manim's used to share one namespace, which meant a
script could not tell them apart and the conventions silently disagreed --
``Square`` was Algan's and took degrees, ``Arc`` was Manim's and took radians,
and nothing in either name said which. This module is the other half of that
split::

    from algan import *
    import algan.manim as mn

    Square().spawn()  # Algan's: degrees, Algan's stroke width
    mn.Arc(angle=PI / 2)  # Manim's: radians, Manim's stroke width

Every name here is Manim's, wrapped by :class:`~algan.mobs.manim_mob.ManimMob`
so that Algan's timeline, materials and renderer apply to it -- Manim's
renderer is never involved. Names that also exist natively in Algan are
wrapped too, so the namespace has no holes: ``mn.Sphere`` is Manim's sphere,
``Sphere`` is Algan's, and both work.

Some of these classes also have a native Algan counterpart under the same name
at the root (``Arc``, ``Axes``, ``Brace`` and the rest of the curated set).
Those counterparts are thin adapters that convert Manim's conventions to
Algan's and delegate here, so ``Arc(angle=90)`` and ``mn.Arc(angle=PI / 2)``
build the same geometry. Reach for this module when you want Manim's
conventions, or a class with no native counterpart at all.

Nothing here is exported by ``from algan import *``; it is reached by import,
which is what makes the boundary visible at the call site.

See :doc:`/advanced_user_tutorials/importing_from_manim`.
"""

from __future__ import annotations

from algan.animatable_base.mob import Mob
from algan.mobs import image_compat as _image_compat
from algan.mobs import manim_compat as _manim_compat
from algan.mobs import manim_parity as _manim_parity
from algan.mobs import opengl_compat as _opengl_compat
from algan.mobs import point_cloud as _point_cloud
from algan.mobs.manim_compat import install_opengl_aliases
from algan.mobs.manim_mob import ManimMob

_SOURCE_MODULES = (
    _manim_compat,
    _opengl_compat,
    _point_cloud,
    _image_compat,
    _manim_parity,
)

for _module in _SOURCE_MODULES:
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)

# Manim names its root class Mobject; Algan's native equivalent is Mob. Its
# abstract graph base likewise maps onto Algan's renderer-independent Graph,
# which in this namespace is the wrapped Manim class.
Mobject = Mob
GenericGraph = globals()["Graph"]

# The OpenGL* names are Manim's renderer-specific spellings of classes that are
# renderer-independent in Algan. They resolve against *this* namespace, so an
# ``OpenGLSquare`` is Manim's Square -- an OpenGL name is a Manim name and
# follows Manim's conventions, not Algan's.
_INSTALLED_OPENGL_ALIASES = install_opengl_aliases(globals())

__all__ = sorted(
    {
        *(name for module in _SOURCE_MODULES for name in module.__all__),
        *_INSTALLED_OPENGL_ALIASES,
        "GenericGraph",
        "ManimMob",
        "Mobject",
        "install_opengl_aliases",
    }
)

del _module, _name
