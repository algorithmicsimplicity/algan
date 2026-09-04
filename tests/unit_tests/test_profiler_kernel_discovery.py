"""Kernel discovery must survive a lazy proxy in an algan module namespace.

``discover_taichi_kernels`` reads a marker attribute off *every* value in
*every* imported ``algan`` module. Some of those values are lazy stand-ins for
optional dependencies, and such a proxy answers an attribute probe by importing
the thing it stands for -- which raises when the extra is not installed.

That took the whole profiler down on the Kaggle T4 box, before a single kernel
was hooked and before the scene under test ever ran: the vendored manim text
stack holds ``manimpango`` as a proxy, ``algan[pango]`` was not installed, and
``getattr(obj, "_is_wrapped_kernel", False)`` raised ImportError instead of
answering False.

Two things are pinned here, because either one alone closes the hole and both
are worth keeping: the proxy answers private-name probes with AttributeError,
and the profiler's own probe treats a raising ``__getattr__`` as "not a kernel".
"""

import pytest

from algan.external_libraries.manim import _pango
from algan.taichi_compat import FUNC_MARKER_ATTR
from algan.utils.profiling_utils import _flag, _is_taichi_func, _is_taichi_kernel


class _Hostile:
    """Anything at all that raises out of ``__getattr__``."""

    def __getattr__(self, attr):
        raise ImportError("this optional dependency is not installed")


def test_lazy_pango_name_answers_private_probes_without_importing():
    lazy = _pango._LazyName("MarkupUtils")
    with pytest.raises(AttributeError):
        lazy._is_wrapped_kernel
    # ``getattr`` with a default is the shape the profiler actually uses.
    assert getattr(lazy, "_is_wrapped_kernel", False) is False
    assert getattr(lazy, "_profiling_kernel_wrapper", False) is False


def test_public_names_still_resolve_or_report_the_missing_extra():
    """The proxy's whole purpose -- a real use says how to fix it -- survives."""
    lazy = _pango._LazyName("MarkupUtils")
    if _pango.available():
        assert lazy.text2svg is not None
    else:
        with pytest.raises(ImportError, match="manimpango"):
            lazy.text2svg


def test_kernel_probe_treats_a_raising_getattr_as_false():
    hostile = _Hostile()
    assert _flag(hostile, "_is_wrapped_kernel") is False
    assert _is_taichi_kernel(hostile) is False
    assert _is_taichi_func(hostile) is False


def test_kernel_probe_still_recognizes_a_kernel_and_a_func():
    class _Kernel:
        _is_wrapped_kernel = True

        def __call__(self):
            pass

    class _Func:
        pass

    setattr(_Func, FUNC_MARKER_ATTR, True)

    assert _is_taichi_kernel(_Kernel()) is True
    assert _is_taichi_func(_Func()) is True
    # A marker without a call is not a kernel, and a plain object is neither.
    assert _is_taichi_kernel(object()) is False
    assert _is_taichi_func(object()) is False