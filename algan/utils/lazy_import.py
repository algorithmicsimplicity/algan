"""Import-on-first-use proxies for heavy third-party dependencies.

``import algan`` pays for everything imported at module scope anywhere in the
package. The dominant removable cost is manim's dependency chain (sympy,
networkx, scipy, bs4, rich, ... -- ~2 s), which only matters once a
Text/Tex/ManimMob is actually constructed. Because the documented usage is
``from algan import *`` (which forces every name in the package namespace),
lazy *submodules* would be defeated immediately -- so laziness lives at the
third-party boundary instead: modules bind a :class:`LazyModule` and use it
exactly like the real module; the real import happens on first attribute
access.
"""

from __future__ import annotations

import importlib
import sys


class LazyModule:
    """Proxy that imports ``name`` on first attribute access.

    ``extras`` are imported immediately after the target the first time it
    loads -- e.g. ``algan.utils.manim_svg_cache``, which must patch manim as
    soon as (but not before) manim exists.
    """

    __slots__ = ("_name", "_extras", "_module")

    def __init__(self, name, extras=()):
        self._name = name
        self._extras = tuple(extras)
        self._module = None

    def _load(self):
        module = importlib.import_module(self._name)
        for extra in self._extras:
            importlib.import_module(extra)
        self._module = module
        return module

    def __getattr__(self, attr):
        module = self._module
        if module is None:
            module = self._load()
        return getattr(module, attr)


def isinstance_if_loaded(obj, lazy_module, class_name):
    """``isinstance(obj, getattr(module, class_name))`` without forcing the
    import: if the module was never loaded, ``obj`` cannot be an instance of
    one of its classes, so the answer is False for free.
    """
    if lazy_module._module is None and lazy_module._name not in sys.modules:
        return False
    return isinstance(obj, getattr(lazy_module, class_name))
