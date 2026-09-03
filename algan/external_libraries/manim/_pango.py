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
