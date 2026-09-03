#!/usr/bin/env python
"""Rebuild ``algan/external_libraries/manim`` from an upstream Manim sdist.

Algan needs Manim's *geometry* -- the Mobject graph, its Bezier machinery and
the SVG/LaTeX pipeline -- and nothing else: Algan animates and renders on its
own. Depending on the real ``manim`` distribution for that costs a build of
``pycairo`` and ``manimpango`` from source on every Linux install, because
neither publishes a Linux wheel. So the geometry subset is vendored.

This script produces that subset. It is committed so a version bump is a
re-run rather than an archaeology session; every edit it makes to upstream
source is mechanical and listed in ``VENDORING.md`` next to the output.

Usage::

    uv run python scripts/vendor_manim.py --download 0.21.0
    uv run python scripts/vendor_manim.py <path-to-unpacked-manim-sdist>

The unpacked sdist is the directory *containing* the ``manim`` package.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DST = REPO_ROOT / "algan" / "external_libraries" / "manim"

# --------------------------------------------------------------------------
# What is kept.
#
# Everything else upstream ships -- ``animation/``, ``camera/``, ``renderer/``,
# ``scene/``, ``cli/``, ``plugins/``, ``opengl/``, the Typst pipeline, the
# docs and test helpers -- is dropped. Two of those are referenced from the
# geometry modules and are replaced by the shims further down.
#
# Names ending a package (``mobject``, ``utils.color``) are that package's
# ``__init__``; the rest are plain modules.
# --------------------------------------------------------------------------
KEEP_MODULES = """
constants typing data_structures
_config.utils
mobject
mobject.frame
mobject.geometry mobject.geometry.arc mobject.geometry.boolean_ops
mobject.geometry.labeled mobject.geometry.line mobject.geometry.polygram
mobject.geometry.shape_matchers mobject.geometry.tips
mobject.graph
mobject.graphing mobject.graphing.coordinate_systems mobject.graphing.functions
mobject.graphing.number_line mobject.graphing.probability mobject.graphing.scale
mobject.logo mobject.matrix mobject.mobject mobject.utils
mobject.value_tracker mobject.vector_field
mobject.svg mobject.svg.brace mobject.svg.svg_mobject
mobject.table
mobject.text mobject.text.code_mobject mobject.text.numbers
mobject.text.tex_mobject mobject.text.text_mobject
mobject.three_d mobject.three_d.polyhedra mobject.three_d.three_d_utils
mobject.three_d.three_dimensions
mobject.types mobject.types.image_mobject mobject.types.point_cloud_mobject
mobject.types.vectorized_mobject
utils utils.bezier utils.config_ops utils.deprecation utils.exceptions
utils.family utils.family_ops utils.file_ops utils.images utils.iterables
utils.parameter_parsing utils.paths utils.polylabel utils.qhull
utils.rate_functions utils.simple_functions utils.space_ops utils.tex
utils.tex_file_writing utils.tex_templates utils.unit
utils.color utils.color.core utils.color.manim_colors utils.color.AS2700
utils.color.BS381 utils.color.DVIPSNAMES utils.color.SVGNAMES utils.color.X11
utils.color.XKCD
""".split()  # noqa: SIM905 - a wrapped block reads better than 70 quoted strings

#: Non-Python files copied verbatim.
DATA_FILES = ["_config/default.cfg"]


# --------------------------------------------------------------------------
# Import rewriting
# --------------------------------------------------------------------------

#: ``from manim.a.b import x`` and ``from manim import x``. Every one becomes
#: relative, so the tree imports itself no matter what owns
#: ``sys.modules["manim"]``.
_ABS_FROM = re.compile(r"^(\s*)from manim(\.[A-Za-z0-9_.]+)? import ", re.MULTILINE)

#: ``import manim.a.b as c``, the one plain-import spelling upstream uses in
#: real code (the others are inside ``.. manim::`` docstring examples, which
#: never execute and are left as written).
_ABS_IMPORT_AS = re.compile(
    r"^(\s*)import manim\.([A-Za-z0-9_.]+) as (\w+)\s*$", re.MULTILINE
)

#: The names the vendored ``_compat`` supplies in place of ``typing``'s.
_COMPAT_TYPING_NAMES = ("Self", "TypeAlias")


def root_prefix(module: str, is_package: bool) -> str:
    """The relative-import prefix reaching the package root from ``module``."""
    depth = module.count(".") + (1 if is_package else 0)
    return "." * (depth + 1)


def rewrite_absolute_imports(text: str, module: str, root: str) -> str:
    def sub_from(m: re.Match) -> str:
        indent, tail = m.group(1), (m.group(2) or "").lstrip(".")
        return f"{indent}from {root}{tail} import "

    def sub_import_as(m: re.Match) -> str:
        indent, dotted, alias = m.groups()
        package, _, name = dotted.rpartition(".")
        return f"{indent}from {root}{package} import {name} as {alias}"

    text = _ABS_IMPORT_AS.sub(sub_import_as, _ABS_FROM.sub(sub_from, text))
    for leftover in absolute_manim_imports(text):
        raise AssertionError(f"{module}: `{leftover}` needs a hand rewrite")
    return text


def absolute_manim_imports(text: str) -> list[str]:
    """Executable ``manim``-absolute imports in ``text``.

    AST-based rather than textual: ``.. manim::`` docstring examples are full
    of ``import manim``, and those are documentation, not code.
    """
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found += [
                f"import {a.name}"
                for a in node.names
                if a.name == "manim" or a.name.startswith("manim.")
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module == "manim" or (node.module or "").startswith("manim."))
        ):
            found.append(f"from {node.module} import ...")
    return found


def rewrite_typing_compat(text: str, module: str, root: str) -> str:
    """Route ``Self`` and ``TypeAlias`` through the vendored ``_compat``."""
    out = []
    for line in text.splitlines(keepends=True):
        m = re.match(r"^(\s*)from typing import ([^(\n]+?)(\s*)$", line)
        if not m:
            if re.match(r"^\s*from typing import \(", line):
                raise AssertionError(f"{module}: parenthesised typing import")
            out.append(line)
            continue
        indent, names, eol = m.group(1), m.group(2), m.group(3)
        names = [n.strip() for n in names.split(",") if n.strip()]
        moved = [n for n in names if n in _COMPAT_TYPING_NAMES]
        if not moved:
            out.append(line)
            continue
        kept = [n for n in names if n not in _COMPAT_TYPING_NAMES]
        if kept:
            out.append(f"{indent}from typing import {', '.join(kept)}{eol}")
        out.append(f"{indent}from {root}_compat import {', '.join(moved)}{eol}")
    return "".join(out)


def rewrite_runtime_type_aliases(text: str, module: str) -> str:
    """Turn module-level ``X: TypeAlias = A | B`` into ``Union[A, B]``.

    An assignment is not deferred by ``from __future__ import annotations``, so
    the ``|`` runs at import; ``__or__`` between typing objects is Python 3.10.
    Aliases inside ``if TYPE_CHECKING`` never execute and are left alone.
    """
    edits = []
    for node in ast.parse(text).body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        ann = node.annotation
        is_alias = (isinstance(ann, ast.Name) and ann.id == "TypeAlias") or (
            isinstance(ann, ast.Constant) and ann.value == "TypeAlias"
        )
        if not is_alias or not _has_union(node.value):
            continue
        rewritten = ast.unparse(
            ast.AnnAssign(
                target=node.target,
                annotation=ann,
                value=_union_to_typing_union(node.value),
                simple=node.simple,
            )
        )
        edits.append((node.lineno, node.end_lineno, rewritten))

    if not edits:
        return text

    lines = text.splitlines(keepends=True)
    for start, end, replacement in reversed(edits):
        lines[start - 1 : end] = [replacement + "\n"]
    return _ensure_union_import("".join(lines), module)


def rewrite_zip_strict(text: str, module: str, root: str) -> str:
    """Route ``zip(..., strict=...)`` through the vendored ``_compat``.

    ``zip``'s ``strict`` keyword is Python 3.10. Rewritten by AST position
    rather than by regex, because several of these calls wrap across lines.
    """
    spots = [
        (node.func.lineno, node.func.col_offset)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(kw.arg == "strict" for kw in node.keywords)
    ]
    if not spots:
        return text

    lines = text.splitlines(keepends=True)
    for lineno, col in sorted(set(spots), reverse=True):
        # ``col_offset`` counts UTF-8 bytes, so splice on bytes.
        raw = lines[lineno - 1].encode("utf-8")
        if raw[col : col + 3] != b"zip":
            raise AssertionError(f"{module}: no `zip` at line {lineno} col {col}")
        lines[lineno - 1] = (raw[:col] + b"_zip" + raw[col + 3 :]).decode("utf-8")
    return _insert_after_future(
        "".join(lines), f"\nfrom {root}_compat import zip_strict as _zip\n", module
    )


def _has_union(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(node)
    )


def _union_to_typing_union(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        members: list[ast.expr] = []

        def flatten(n: ast.expr) -> None:
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                flatten(n.left)
                flatten(n.right)
            else:
                members.append(_union_to_typing_union(n))

        flatten(node)
        return ast.Subscript(
            value=ast.Name(id="Union", ctx=ast.Load()),
            slice=ast.Tuple(elts=members, ctx=ast.Load()),
            ctx=ast.Load(),
        )
    if isinstance(node, ast.Subscript):
        return ast.Subscript(
            value=node.value, slice=_union_to_typing_union(node.slice), ctx=node.ctx
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(
            elts=[_union_to_typing_union(e) for e in node.elts], ctx=node.ctx
        )
    return node


def _ensure_union_import(text: str, module: str) -> str:
    if re.search(r"^from typing import .*\bUnion\b", text, re.MULTILINE):
        return text
    return _insert_after_future(text, "\nfrom typing import Union\n", module)


def _insert_after_future(text: str, insertion: str, module: str) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            lines.insert(i + 1, insertion)
            return "".join(lines)
    raise AssertionError(f"{module}: no `from __future__` line to anchor an import")


def apply_numpy_compat(text: str, module: str, root: str) -> str:
    if "np.trapezoid" not in text:
        return text
    text = text.replace("np.trapezoid(", "_trapezoid(")
    return _insert_after_future(
        text, f"\nfrom {root}_compat import trapezoid as _trapezoid\n", module
    )


# --------------------------------------------------------------------------
# Targeted patches
#
# Each one removes a reference to something the vendored subset drops. They
# assert rather than try: an upstream bump that moves one of these lines fails
# the build here instead of producing a tree that imports the wrong thing.
# --------------------------------------------------------------------------


def apply_targeted_patches(text: str, module: str) -> str:
    def cut(old: str, new: str = "") -> None:
        nonlocal text
        if old not in text:
            raise AssertionError(f"{module}: patch target not found:\n{old!r}")
        text = text.replace(old, new, 1)

    if module == "constants":
        # cloup exists here only to build the CLI's click context, and the CLI
        # is not vendored. Dropping it drops the dependency.
        cut("from cloup import Context\n")
        cut('    "CONTEXT_SETTINGS",\n')
        cut(
            "CONTEXT_SETTINGS = Context.settings(\n"
            "    align_option_groups=True,\n"
            "    align_sections=True,\n"
            "    show_constraints=True,\n"
            ")\n"
        )

    elif module == "typing":
        cut("    from .mobject.text.typst_mobject import Typst\n")
        cut('"Text | MathTex | Typst"', '"Text | MathTex"')
        cut(
            "This includes :class:`~.Text`, :class:`~.MathTex`, and "
            ":class:`~.Typst`.\n",
            "This includes :class:`~.Text` and :class:`~.MathTex`.\n",
        )

    elif module == "_config.utils":
        # The vendored subset ships no OpenGL renderer, and the base-swapping
        # this setter does would rebase the geometry classes onto the inert
        # stand-ins in mobject/opengl/.
        cut(
            "        renderer = RendererType(value)\n",
            "        renderer = RendererType(value)\n"
            "        if renderer == RendererType.OPENGL:\n"
            "            raise ValueError(\n"
            "                \"Algan's vendored Manim subset ships no renderer, "
            'so config."\n'
            "                \"renderer cannot be set to 'opengl'. Algan renders \"\n"
            '                "Manim geometry with its own ray tracer."\n'
            "            )\n",
        )

    elif module == "mobject.types.image_mobject":
        # Used only in an annotation, and re-imported under TYPE_CHECKING a few
        # lines down, so the runtime import is pure cost.
        cut("from ...camera.moving_camera import MovingCamera\n")
        cut(
            "    from ...camera.moving_camera import MovingCamera\n",
            "    from typing import Any as MovingCamera\n",
        )

    elif module == "mobject.geometry.labeled":
        # Typst needs the `typst` package and a second document pipeline; the
        # LaTeX and Pango label paths cover everything Algan converts.
        cut("from ...mobject.text.typst_mobject import Typst\n")
        cut("(MathTex, Text, Typst)", "(MathTex, Text)")
        cut(
            "Must be MathTex, Tex, Text, Typst, or MathTypst.",
            "Must be MathTex, Tex, or Text.",
        )

    elif module == "mobject.graphing.number_line":
        cut("from ...mobject.text.typst_mobject import MathTypst, Typst\n")
        cut(
            "                elif label_constructor is MathTypst:\n"
            "                    label = Typst(label)\n"
        )

    elif module == "mobject.text.text_mobject":
        # manimpango is the optional `algan[pango]` extra, but this module has
        # to import even without it: `Text` and `Paragraph` are the default
        # `label_constructor` / `element_to_mobject` of Brace, Table and the
        # labelled geometry, and those modules import them at module level. So
        # the dependency moves behind the lazy proxies in manim/_pango.py --
        # importing this module costs nothing, and only *building* a Text says
        # what is missing. `manim/__init__.py` is what withholds the classes.
        cut(
            "import manimpango\n",
            "from ..._pango import manimpango\n",
        )
        cut(
            "from manimpango import MarkupUtils, PangoUtils, TextSetting\n",
            "from ..._pango import MarkupUtils, PangoUtils, TextSetting\n",
        )

    return text


# --------------------------------------------------------------------------
# Generated files
# --------------------------------------------------------------------------

COMPAT_PY = '''\
"""Typing and NumPy shims, so the vendored subset runs on Algan's floor.

Upstream Manim requires Python 3.11 and NumPy 2.1; Algan supports Python 3.9
and NumPy 1.20 upwards. Three upstream spellings need a fallback, and the
vendoring script rewrites every use to come through here:

``Self``
    :data:`typing.Self` is 3.11. It is only ever used in an annotation, and
    every vendored module carries ``from __future__ import annotations``, so an
    older interpreter never evaluates it -- the name just has to exist.
``TypeAlias``
    :data:`typing.TypeAlias` is 3.10.
``trapezoid``
    ``np.trapezoid`` is NumPy 2.0's spelling of ``np.trapz``.
``zip_strict``
    ``zip``'s ``strict=`` keyword is 3.10.
"""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy as np

__all__ = ["Self", "TypeAlias", "trapezoid", "zip_strict"]

if sys.version_info >= (3, 11):
    from typing import Self, TypeAlias
else:  # pragma: no cover - exercised only on Python 3.9 and 3.10
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        try:
            from typing_extensions import TypeAlias
        except ImportError:
            TypeAlias = Any  # type: ignore[misc, assignment]
    try:
        from typing_extensions import Self
    except ImportError:
        Self = Any  # type: ignore[misc, assignment]

#: ``np.trapz`` was renamed in NumPy 2.0 and removed from the main namespace.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


if sys.version_info >= (3, 10):
    zip_strict = zip
else:  # pragma: no cover - exercised only on Python 3.9

    def _raise_on_ragged(iterables: tuple[Any, ...]) -> Any:
        sentinel = object()
        for row in itertools.zip_longest(*iterables, fillvalue=sentinel):
            if any(value is sentinel for value in row):
                raise ValueError("zip() arguments are of unequal length")
            yield row

    def zip_strict(*iterables: Any, strict: bool = False) -> Any:
        """``zip``, with 3.10's ``strict=`` keyword.

        Lazy either way, and ``strict=True`` raises ``ValueError`` on ragged
        input as the builtin does -- the message differs, the type does not.
        """
        if not strict:
            return zip(*iterables)
        return _raise_on_ragged(iterables)
'''

PANGO_PY = '''\
"""Lazy access to ``manimpango``, which is optional for Algan.

Manim's Pango text renderer needs ``manimpango``, and manimpango publishes no
Linux wheel -- requiring it would put a from-source build of Pango in front of
every Linux ``pip install algan``, which is the cost the vendoring exists to
avoid. It is the ``algan[pango]`` extra instead.

That makes it optional, not detachable. ``Text`` is the default
``label_constructor`` of ``Brace`` and the labelled geometry and ``Paragraph``
is ``Table``'s default ``element_to_mobject``, so
``manim.mobject.text.text_mobject`` has to *import* on a machine without
Pango even though nothing in it can *run* there. The proxies below make that
work: importing costs nothing, and the first actual use raises with the one
sentence that fixes it.

Whether the classes are usable is a separate question, and
``manim.PANGO_AVAILABLE`` answers it -- ``Text``, ``MarkupText`` and
``Paragraph`` are exported from the package only when it is true, so
``hasattr(manim, "Text")`` stays the honest test that Algan's own ``Text``
(which falls back to LaTeX's text mode) and its compatibility layer both use.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MarkupUtils", "PangoUtils", "TextSetting", "available", "manimpango"]

_MESSAGE = (
    "Pango text rendering needs the `manimpango` package, which Algan does "
    "not install by default -- it publishes no Linux wheel, so requiring it "
    'would mean building Pango from source. Install it with `pip install '
    '"algan[pango]"` (or `pip install manimpango`). Without it, use '
    "`algan.Text`, which typesets through LaTeX's text mode instead, or "
    "`algan.Tex` / `mn.MathTex` for mathematics."
)


def _module() -> Any:
    try:
        import manimpango
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(_MESSAGE) from exc
    return manimpango


def available() -> bool:
    """Whether ``manimpango`` can be imported."""
    try:
        _module()
    except ImportError:
        return False
    return True


class _LazyName:
    """A stand-in for one top-level ``manimpango`` name.

    Resolves on first use, and forwards both attribute access
    (``MarkupUtils.text2svg``) and calls (``TextSetting(...)``), which is
    every shape the vendored ``text_mobject`` uses these in.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name

    def _resolve(self) -> Any:
        module = _module()
        return module if self._name is None else getattr(module, self._name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._resolve(), attr)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<lazy manimpango{'.' + self._name if self._name else ''}>"


manimpango = _LazyName()
MarkupUtils = _LazyName("MarkupUtils")
PangoUtils = _LazyName("PangoUtils")
TextSetting = _LazyName("TextSetting")
'''

CONFIG_INIT_PY = '''\
"""The global Manim config object.

Upstream also builds a ``rich``-backed logger here and installs it on the
*root* logger at import time. Algan owns its own console output and must not
have a library reconfigure logging out from under the importing application,
so this copy hands out a plain :mod:`logging` logger with a ``NullHandler``
and a ``console`` that is a thin ``print`` wrapper. That is everything the
vendored geometry subset asks for, and it keeps ``rich`` out of Algan's
dependency set.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .utils import ManimConfig, ManimFrame, make_config_parser

__all__ = [
    "config",
    "console",
    "error_console",
    "frame",
    "logger",
    "tempconfig",
]

_RICH_MARKUP = re.compile(r"\\[/?[a-z_ ]+\\]")


class _Console:
    """The sliver of ``rich.console.Console`` the vendored subset calls."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def print(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("style", None)
        text = " ".join(_RICH_MARKUP.sub("", str(a)) for a in args)
        print(text, file=self._stream, **kwargs)


#: Reachable as ``manim.logger`` or ``logging.getLogger("manim")``.
logger = logging.getLogger("manim")
logger.addHandler(logging.NullHandler())

console = _Console(sys.stdout)
error_console = _Console(sys.stderr)

parser = make_config_parser()
config = ManimConfig().digest_parser(parser)
frame = ManimFrame(config)


@contextmanager
def tempconfig(temp: ManimConfig | dict[str, Any]) -> Generator[None, None, None]:
    """Temporarily modify the global ``config`` object.

    Inside the ``with`` statement the modified config is in force; on exit the
    original values are restored.

    Examples
    --------
    .. code-block:: pycon

       >>> config["frame_height"]
       8.0
       >>> with tempconfig({"frame_height": 100.0}):
       ...     print(config["frame_height"])
       100.0
       >>> config["frame_height"]
       8.0
    """
    global config
    original = config.copy()

    temp = {k: v for k, v in temp.items() if k in original}

    # update(), never assignment: every module holds a reference to this one
    # object, and rebinding the name here would not reach any of them.
    config.update(temp)
    try:
        yield
    finally:
        config.update(original)
'''

OPENGL_SHIM = {
    "__init__.py": '''\
"""Stand-ins for Manim's OpenGL Mobject tree, which Algan does not vendor.

Upstream carries a parallel ~7000-line hierarchy (``OpenGLMobject`` and
friends) that exists to serve ``manim --renderer=opengl``. The vendored subset
has no renderer at all, so none of it can ever run; what the geometry modules
actually mention is a metaclass plus a handful of classes used in
``isinstance`` checks and annotations.

The metaclass here is the real one minus its renderer branch. The classes are
inert: nothing is ever an instance of them, which is exactly the answer those
``isinstance`` checks want under the Cairo-shaped code path, and constructing
one says why it cannot work.
"""
''',
    "_placeholder.py": '''\
"""The base every OpenGL stand-in shares."""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = ["_OpenGLPlaceholder"]


class _OpenGLPlaceholder:
    """A class no object is ever an instance of.

    The geometry modules reach for the OpenGL types two ways, and both survive:
    ``isinstance(x, (Mobject, OpenGLMobject))`` answers exactly as it would
    upstream under the Cairo renderer, and annotations are never evaluated.
    Construction is the one thing that cannot work, so it says so.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} belongs to Manim's OpenGL renderer, which "
            "Algan's vendored Manim subset does not include. Algan renders "
            "Manim geometry with its own ray tracer; build the Cairo-side "
            "class instead."
        )
''',
    "opengl_compatibility.py": '''\
"""The ``ConvertToOpenGL`` metaclass, minus the renderer swap."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any

__all__ = ["ConvertToOpenGL"]


class ConvertToOpenGL(ABCMeta):
    """Upstream swaps a class's bases for their OpenGL counterparts here when
    ``config.renderer`` is OpenGL. The vendored subset ships no OpenGL
    renderer -- ``ManimConfig.renderer`` rejects ``"opengl"`` outright -- so
    this is the Cairo branch alone: an ``ABCMeta`` that records the classes
    built with it, because that registry is still walked by the config setter.
    """

    _converted_classes: list[type] = []

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
''',
    "opengl_mobject.py": '''\
"""Inert stand-ins for ``OpenGLMobject`` and ``OpenGLGroup``."""

from __future__ import annotations

from ._placeholder import _OpenGLPlaceholder

__all__ = ["OpenGLGroup", "OpenGLMobject"]


class OpenGLMobject(_OpenGLPlaceholder):
    """See :mod:`~manim.mobject.opengl`."""


class OpenGLGroup(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
''',
    "opengl_vectorized_mobject.py": '''\
"""Inert stand-ins for ``OpenGLVMobject`` and ``OpenGLVGroup``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLVGroup", "OpenGLVMobject"]


class OpenGLVMobject(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""


class OpenGLVGroup(OpenGLVMobject):
    """See :mod:`~manim.mobject.opengl`."""
''',
    "opengl_surface.py": '''\
"""Inert stand-in for ``OpenGLSurface``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLSurface"]


class OpenGLSurface(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
''',
    "opengl_three_dimensions.py": '''\
"""``OpenGLSurface`` under the name upstream's compatibility module uses."""

from __future__ import annotations

from .opengl_surface import OpenGLSurface

__all__ = ["OpenGLSurface"]
''',
    "opengl_point_cloud_mobject.py": '''\
"""Inert stand-in for ``OpenGLPMobject``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLPMobject"]


class OpenGLPMobject(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
''',
}

ANIMATION_SHIM = {
    "__init__.py": '''\
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
''',
    "animation.py": '''\
"""``Animation`` itself, and the ``override_animation`` decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

__all__ = ["Animation", "override_animation"]


class Animation:
    """A Manim animation, as far as the vendored geometry subset needs one.

    Manim's animations drive a Manim ``Scene`` through a Manim renderer, and
    this subset vendors neither. Algan's own animation system replaces them;
    see :mod:`algan.animations`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} is part of Manim's animation system, which "
            "Algan's vendored Manim subset does not include -- Manim geometry "
            "is converted to Algan Mobs and animated by Algan. Use Algan's "
            "animations on the converted Mob instead."
        )


def override_animation(animation_class: type) -> Callable[[Callable], Callable]:
    """Mark a Mobject method as the override for ``animation_class``.

    Upstream's implementation verbatim. It only tags the function;
    :class:`~manim.mobject.mobject.Mobject.__init_subclass__` is what collects
    the tags into ``animation_overrides``. Nothing here runs an animation, so
    the registration side works unchanged.
    """

    def decorator(func: Callable) -> Callable:
        func._override_animation = animation_class
        return func

    return decorator
''',
    "composition.py": '''\
"""Inert stand-ins for Manim's animation containers."""

from __future__ import annotations

from .animation import Animation

__all__ = ["AnimationGroup", "LaggedStart", "Succession"]


class AnimationGroup(Animation):
    """See :mod:`~manim.animation`."""


class Succession(AnimationGroup):
    """See :mod:`~manim.animation`."""


class LaggedStart(AnimationGroup):
    """See :mod:`~manim.animation`."""
''',
    "creation.py": '''\
"""Inert stand-ins for Manim's creation animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["Create", "ShowPartial", "SpiralIn", "Uncreate", "Write"]


class ShowPartial(Animation):
    """See :mod:`~manim.animation`."""


class Create(ShowPartial):
    """See :mod:`~manim.animation`."""


class Uncreate(Create):
    """See :mod:`~manim.animation`."""


class Write(Create):
    """See :mod:`~manim.animation`."""


class SpiralIn(Animation):
    """See :mod:`~manim.animation`."""
''',
    "fading.py": '''\
"""Inert stand-ins for Manim's fading animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["FadeIn", "FadeOut"]


class _Fade(Animation):
    """See :mod:`~manim.animation`."""


class FadeIn(_Fade):
    """See :mod:`~manim.animation`."""


class FadeOut(_Fade):
    """See :mod:`~manim.animation`."""
''',
    "growing.py": '''\
"""Inert stand-ins for Manim's growing animations."""

from __future__ import annotations

from .transform import Transform

__all__ = ["GrowFromCenter", "GrowFromPoint", "SpinInFromNothing"]


class GrowFromPoint(Transform):
    """See :mod:`~manim.animation`."""


class GrowFromCenter(GrowFromPoint):
    """See :mod:`~manim.animation`."""


class SpinInFromNothing(GrowFromPoint):
    """See :mod:`~manim.animation`."""
''',
    "indication.py": '''\
"""Inert stand-ins for Manim's indication animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["ShowPassingFlash"]


class ShowPassingFlash(Animation):
    """See :mod:`~manim.animation`."""
''',
    "transform.py": '''\
"""Inert stand-ins for Manim's transform animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["Transform", "_MethodAnimation"]


class Transform(Animation):
    """See :mod:`~manim.animation`."""


class _MethodAnimation(Transform):
    """What upstream builds behind ``mob.animate.shift(...)``.

    Reached only through ``Mobject.animate``, which needs a Manim ``Scene`` to
    play what it builds. Algan records the plain method call instead: call
    ``mob.shift(...)`` inside an Algan animation context.
    """
''',
    "updaters/__init__.py": '''\
"""Stand-ins for Manim's updater animations."""
''',
    "updaters/update.py": '''\
"""Inert stand-ins for Manim's updater animations."""

from __future__ import annotations

from ..animation import Animation

__all__ = ["UpdateFromAlphaFunc", "UpdateFromFunc"]


class UpdateFromFunc(Animation):
    """See :mod:`~manim.animation`."""


class UpdateFromAlphaFunc(UpdateFromFunc):
    """See :mod:`~manim.animation`."""
''',
}

INIT_PY = '''\
"""Manim Community's geometry subset, vendored for Algan.

This is not Manim. It is the part of Manim that *builds Mobjects* -- the
Mobject graph, the Bezier and SVG/LaTeX machinery, and the shape, graphing,
text and 3-D classes on top of them. Manim's animations, scenes, cameras,
renderers, CLI and plugin system are absent, because Algan supplies all of
those itself: :class:`~algan.mobs.manim_mob.ManimMob` takes the cubic Bezier
circuits a Manim Mobject produces and turns them into Algan render primitives,
and everything after that is Algan's.

Reachable as ``manim`` -- Algan registers it under that name before importing
any Mob module -- and as ``algan.external_libraries.manim``. See
``VENDORING.md`` in this directory for the provenance and the exact set of
edits made to upstream.
"""

from __future__ import annotations

#: The upstream Manim Community release this subset was taken from.
__version__ = "{version}"

# isort: off

# Config first: every module below reads the global config as it is imported.
from ._config import config, console, error_console, frame, logger, tempconfig

# isort: on

from .constants import *
from .mobject.frame import *
from .mobject.geometry.arc import *
from .mobject.geometry.boolean_ops import *
from .mobject.geometry.labeled import *
from .mobject.geometry.line import *
from .mobject.geometry.polygram import *
from .mobject.geometry.shape_matchers import *
from .mobject.geometry.tips import *
from .mobject.graph import *
from .mobject.graphing.coordinate_systems import *
from .mobject.graphing.functions import *
from .mobject.graphing.number_line import *
from .mobject.graphing.probability import *
from .mobject.graphing.scale import *
from .mobject.logo import *
from .mobject.matrix import *
from .mobject.mobject import *
from .mobject.svg.brace import *
from .mobject.svg.svg_mobject import *
from .mobject.table import *
from .mobject.text.code_mobject import *
from .mobject.text.numbers import *
from .mobject.text.tex_mobject import *
from .mobject.three_d.polyhedra import *
from .mobject.three_d.three_d_utils import *
from .mobject.three_d.three_dimensions import *
from .mobject.types.image_mobject import *
from .mobject.types.point_cloud_mobject import *
from .mobject.types.vectorized_mobject import *
from .mobject.value_tracker import *
from .mobject.vector_field import *
from .utils import color, rate_functions, unit
from .utils.color import *
from .utils.config_ops import *
from .utils.file_ops import *
from .utils.images import *
from .utils.iterables import *
from .utils.paths import *
from .utils.rate_functions import *
from .utils.simple_functions import *
from .utils.space_ops import *
from .utils.tex import *
from .utils.tex_templates import *

#: Whether Pango-rendered text is available.
#:
#: ``Text``, ``MarkupText`` and ``Paragraph`` are the only classes that need
#: ``manimpango``, which publishes no Linux wheel -- requiring it would put a
#: source build of Pango in front of every Linux ``pip install algan``, which
#: is the cost this whole directory exists to avoid. Install the optional
#: extra (``pip install "algan[pango]"``) to get them.
#:
#: The three classes are withheld rather than left importable-and-broken,
#: because that is what Algan tests: :class:`algan.Text` renders through
#: LaTeX's text mode when ``hasattr(manim, "Text")`` is false, and
#: :mod:`algan.mobs.manim_compat` leaves them out of the compatibility
#: registry. (The *module* still imports either way -- ``Text`` is
#: ``Brace``'s default label class and ``Paragraph`` is ``Table``'s default
#: entry class, so ``mobject/text/text_mobject.py`` reaches manimpango through
#: the lazy proxies in ``manim/_pango.py``.)
from ._pango import available as _pango_available

PANGO_AVAILABLE = _pango_available()

if PANGO_AVAILABLE:
    from .mobject.text.text_mobject import *  # noqa: F401
'''

VENDORING_MD = """\
# Vendoring Manim

This directory is Manim Community **{version}**, cut down to the geometry Algan
converts and rewritten so it imports itself rather than whatever happens to own
`sys.modules["manim"]`.

Rebuild it with:

```bash
uv run python scripts/vendor_manim.py --download {version}
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
`utils/{{caching,commands,debug,hashing,ipython_magic,module_ops,opengl,sounds}}.py`,
and `_config/{{logger_utils,cli_colors}}.py`.

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
"""


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build(src_root: Path, version: str) -> None:
    src = src_root / "manim"
    if not (src / "mobject" / "mobject.py").is_file():
        raise SystemExit(f"{src} does not look like a Manim package")

    license_text = next(
        (
            (src_root / name).read_text(encoding="utf-8")
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
            if (src_root / name).is_file()
        ),
        None,
    )
    if license_text is None:
        if not (DST / "LICENSE").is_file():
            raise SystemExit("no LICENSE in the sdist and none to carry over")
        license_text = (DST / "LICENSE").read_text(encoding="utf-8")

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    for module in KEEP_MODULES:
        parts = module.split(".")
        is_package = (src / Path(*parts)).is_dir()
        rel = (
            Path(*parts) / "__init__.py"
            if is_package
            else Path(*parts[:-1]) / f"{parts[-1]}.py"
        )
        source = src / rel
        if not source.is_file():
            raise SystemExit(f"upstream is missing {rel} (module {module})")

        root = root_prefix(module, is_package)
        text = source.read_text(encoding="utf-8")
        text = rewrite_absolute_imports(text, module, root)
        text = apply_targeted_patches(text, module)
        text = rewrite_zip_strict(text, module, root)
        text = rewrite_typing_compat(text, module, root)
        text = rewrite_runtime_type_aliases(text, module)
        text = apply_numpy_compat(text, module, root)

        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")

    for rel in DATA_FILES:
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, DST / rel)

    for subdir, shim in (
        ("mobject/opengl", OPENGL_SHIM),
        ("animation", ANIMATION_SHIM),
    ):
        for name, body in shim.items():
            path = DST / subdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8", newline="\n")

    for name, body in (
        ("_compat.py", COMPAT_PY),
        ("_pango.py", PANGO_PY),
        ("_config/__init__.py", CONFIG_INIT_PY),
        ("__init__.py", INIT_PY.format(version=version)),
        ("LICENSE", license_text),
        ("VENDORING.md", VENDORING_MD.format(version=version)),
    ):
        (DST / name).write_text(body, encoding="utf-8", newline="\n")

    survivors = sorted(
        str(p.relative_to(DST))
        for p in DST.rglob("*.py")
        if absolute_manim_imports(p.read_text(encoding="utf-8"))
    )
    if survivors:
        raise SystemExit(f"absolute `manim` imports survived in: {survivors}")

    modules = len(list(DST.rglob("*.py")))
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6
    print(f"vendored manim {version}: {modules} modules, {size:.2f} MB -> {DST}")


def download(version: str) -> Path:
    with urllib.request.urlopen(f"https://pypi.org/pypi/manim/{version}/json") as fh:
        meta = json.load(fh)
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    print(f"downloading {sdist['filename']} ...")
    with urllib.request.urlopen(sdist["url"]) as fh:
        payload = fh.read()
    tmp = Path(tempfile.mkdtemp(prefix="manim-vendor-"))
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        tar.extractall(tmp)
    return tmp / f"manim-{version}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", help="unpacked Manim sdist directory")
    ap.add_argument("--download", metavar="VERSION", help="fetch the sdist from PyPI")
    args = ap.parse_args(argv)

    if args.download:
        root, version = download(args.download), args.download
    elif args.source:
        root = Path(args.source).resolve()
        m = re.search(r"manim-([0-9][^/\\]*)$", root.name)
        version = m.group(1) if m else "unknown"
    else:
        ap.error("give a source directory or --download VERSION")
    build(root, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
