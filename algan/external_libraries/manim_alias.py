"""Make ``import manim`` resolve to Algan's vendored Manim subset.

Algan ships the geometry half of Manim Community under
:mod:`algan.external_libraries.manim` (see ``VENDORING.md`` there for why),
and both Algan's own modules and user scripts reach it by Manim's ordinary
name: ``import manim as mn``, ``ManimMob(manim.Circle())``. This module is
what makes that name mean the vendored copy.

Aliasing the top-level package alone is not enough, and getting that wrong is
what the old ``sys.modules.setdefault("manim", ...)`` did. ``sys.modules``
holds one entry per module, not per package: with only ``manim`` aliased,
``from manim.mobject.svg.brace import BraceText`` would find ``brace.py``
again through the aliased package's ``__path__`` and **execute it a second
time**, producing a ``BraceText`` that is not the ``BraceText`` the rest of
the tree holds. Every ``isinstance`` and ``issubclass`` across that seam then
answers ``False``.

So the alias is a meta-path finder: any ``manim.X`` import returns the
*already-imported object* ``algan.external_libraries.manim.X``, never a second
execution of the same file.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder
from types import ModuleType
from typing import Any

__all__ = ["install"]

_ALIAS = "manim"
_TARGET = "algan.external_libraries.manim"


class _AliasLoader(Loader):
    """A loader that hands back a module which is already imported."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def create_module(self, spec: Any) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:
        """Already executed under its real name; running it again is the bug."""


class _VendoredManimFinder(MetaPathFinder):
    """Route ``manim.<rest>`` to ``algan.external_libraries.manim.<rest>``."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _ALIAS and not fullname.startswith(_ALIAS + "."):
            return None
        real_name = _TARGET + fullname[len(_ALIAS) :]
        try:
            module = importlib.import_module(real_name)
        except ImportError:
            # Not part of the vendored subset -- Manim's animations, scenes,
            # cameras and renderers are all absent. Returning None lets the
            # normal machinery produce a ModuleNotFoundError naming what was
            # asked for, which is the useful message.
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(module), is_package=hasattr(module, "__path__")
        )


def install() -> ModuleType:
    """Install the alias and return the vendored package.

    Idempotent, and safe to call before the rest of Algan is imported -- which
    is where it must happen, since Algan's Mob modules import ``manim`` at
    their own import time.
    """
    vendored = importlib.import_module(_TARGET)

    existing = sys.modules.get(_ALIAS)
    if existing is not None and existing is not vendored:
        warnings.warn(
            "`manim` was already imported when algan loaded, and algan has "
            "replaced it with its own vendored Manim subset "
            f"({_TARGET}). Algan's compatibility layer requires a single Manim "
            "identity: a Mobject built by the copy you imported first is not "
            "an instance of the classes algan checks against, so ManimMob and "
            "algan.manim would silently disagree about it. Import algan "
            "before manim, or drop the separate manim import and use "
            "`import algan.manim as mn`.",
            RuntimeWarning,
            stacklevel=2,
        )

    sys.modules[_ALIAS] = vendored
    if not any(isinstance(f, _VendoredManimFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _VendoredManimFinder())

    # The submodules the package's own __init__ already executed are reachable
    # under their real names; alias each so an `import manim.x` that races the
    # finder (or a `from manim import x` on a submodule) sees the same object.
    for name, module in list(sys.modules.items()):
        if name.startswith(_TARGET + "."):
            sys.modules[_ALIAS + name[len(_TARGET) :]] = module

    return vendored
