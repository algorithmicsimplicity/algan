# Vendoring Manim

This directory is Manim Community **0.21.0**, cut down to the geometry Algan
converts and rewritten so it imports itself rather than whatever happens to own
`sys.modules["manim"]`.

Rebuild it with:

```bash
uv run python scripts/vendor_manim.py --download 0.21.0
```

That script is the specification -- everything below is a description of what
it does. It asserts on each targeted patch, so an upstream bump that moves one
of these lines fails the build instead of quietly producing a tree that imports
the wrong thing.

## Why vendor at all

Depending on the `manim` distribution drags in `pycairo` and `manimpango`.
Neither publishes a Linux wheel, so `pip install algan` would compile Cairo and
Pango bindings from source on every Linux box without `libcairo2-dev
libpango1.0-dev pkg-config`. Algan needs neither: it uses Manim to *build
Bezier geometry* and renders that with its own ray tracer.

## What is kept

`_config`, `constants`, `typing`, `data_structures`, the whole `mobject/` tree
except the OpenGL hierarchy and the Typst classes, and the `utils/` modules
those need.

## What is dropped

`animation/`, `camera/`, `renderer/`, `scene/`, `cli/`, `plugins/`, `opengl/`,
`mobject/opengl/`, `mobject/text/typst_mobject.py`,
`utils/typst_file_writing.py`, `utils/docbuild/`, `utils/testing/`,
`utils/{caching,commands,debug,hashing,ipython_magic,module_ops,opengl,sounds}.py`,
and `_config/{logger_utils,cli_colors}.py`.

Two of those are referenced from geometry modules, and are replaced by shims:

* `mobject/opengl/` -- `ConvertToOpenGL` (the real metaclass minus its renderer
  branch) plus inert `OpenGL*` classes for the `isinstance` checks. What it
  replaces is ~7000 lines that only `manim --renderer=opengl` can reach.
* `animation/` -- inert `Animation` subclasses and the real
  `override_animation` decorator, for the five geometry classes that name an
  animation (`Brace.creation_anim`, `Table.create`, `ManimBanner.create`,
  `StreamLines.create`, and `Graph`'s `@override_animation(Create)`).

## Edits to upstream source

Mechanical, applied to every file:

1. Every `from manim... import` becomes relative. Nothing in the tree can
   resolve to an installed `manim`; `scripts/vendor_manim.py` fails the build
   if one survives.
2. `typing.Self` and `typing.TypeAlias` come from `manim/_compat.py`, which
   falls back to `typing_extensions` -- upstream requires Python 3.11, Algan
   supports 3.9.
3. Module-level `X: TypeAlias = A | B` becomes `Union[A, B]`, for the same
   reason: `|` between typing objects is 3.10, and an assignment is not
   deferred by `from __future__ import annotations`.
4. `np.trapezoid` goes through `manim/_compat.py`, which falls back to
   `np.trapz` -- upstream requires NumPy 2.1, Algan supports 1.20.
5. `zip(..., strict=...)` goes through `manim/_compat.py` too; the keyword is
   3.10.

Rewritten wholesale:

6. `_config/__init__.py`. Upstream builds a `rich` logger and installs it on
   the **root** logger at import time. This copy hands out a plain
   `logging.getLogger("manim")` with a `NullHandler`, which is also how `rich`
   stays out of Algan's dependency set.
7. `__init__.py`, to export the kept subset, with the Pango classes withheld
   unless `manimpango` imports (see `PANGO_AVAILABLE`).

Targeted, asserted, one dropped reference each:

8. `constants.py` -- the `cloup` import and `CONTEXT_SETTINGS`, which only the
   CLI used. Drops the `cloup` dependency.
9. `typing.py`, `mobject/geometry/labeled.py`, `mobject/graphing/number_line.py`
   -- the `Typst` branches.
10. `mobject/types/image_mobject.py` -- the unused runtime `MovingCamera`
    import (it is re-imported under `TYPE_CHECKING` a few lines below).
11. `_config/utils.py` -- `ManimConfig.renderer`'s setter rejects `"opengl"`
    outright rather than rebasing the geometry classes onto the stand-ins.
12. `mobject/text/text_mobject.py` -- `manimpango` is reached through the lazy
    proxies in `manim/_pango.py`, so the module imports without the optional
    extra. It has to: `Text` is `Brace`'s default label class and `Paragraph`
    is `Table`'s default entry class, and both are imported at module level by
    modules that have nothing to do with Pango.
