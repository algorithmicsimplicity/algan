"""Early ``return`` inside an inlined ``@ti.func``, compiled for real.

``algan/utils/taichi_early_return.py`` rewrites a func body whose ``return``
sits under runtime control flow -- which both compilers reject -- into
single-exit form before the transformer sees it. Nothing in Algan's own kernels
has such a ``return``, so every claim in that module is held here instead, and
held the only way that means anything: **the rewritten func is compiled by the
live compiler and its answer compared against a hand-written single-exit func
computing the same thing on the same inputs**, plus the value Python itself
would produce.

The backend is process-global (``algan.taichi_compat`` binds it on first use),
so this file cannot parametrise over both. Run it twice::

    ALGAN_TAICHI_BACKEND=quadrants .venv/Scripts/python.exe -m pytest -q tests/unit_tests/test_taichi_early_return.py
    ALGAN_TAICHI_BACKEND=taichi    .venv/Scripts/python.exe -m pytest -q tests/unit_tests/test_taichi_early_return.py

Two things the rewrite deliberately does *not* do, checked here so they stay
deliberate:

* it **refuses** a ``return`` anywhere inside a func's outermost runtime
  ``for`` -- that loop is the offloaded, parallel one wherever the func is
  inlined at a kernel's top level, so a body guard would compile into a race
  and ``break`` would not compile at all. The ``while`` spelling is tested
  beside it as the alternative;
* a body with no nested ``return`` is handed back untouched, so no existing
  kernel's frontend IR (and no offline-cache key) moves.
"""

# No ``from __future__ import annotations``: the kernels below carry
# ``ti.types.ndarray(...)`` annotations the compilers read as objects.
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from algan.errors import AlganWarning
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import BACKEND, ti
from algan.utils import taichi_early_return as er
from algan.utils import taichi_source_key as sk

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_source_key_shortcut():
    """Nothing in this file may be served from the source-keyed cache.

    This file tests an **AST transform**, and 28 of its tests assert the tag
    ``_rewrite_tree`` leaves on a function when it rewrites it. The
    source-keyed index (``algan/utils/taichi_source_key.py``, on by default
    since 2026-09-05) exists precisely to *skip* that transform for a kernel
    some previous process already compiled -- so on a hit the tag is never
    set and the assertion reads as "the rewrite did not fire" when in truth
    it fired earlier and was cached. The kernel is still correct; the test is
    simply blind.

    That makes every one of those 28 cache-dependent: green on a cold index,
    red on a warm one, which is worse than either. Restoring the compiler's
    own ``_try_load_fastcache`` for the duration of each test forces the full
    transform, so they test what they claim whatever is on disk. The store
    hook is deliberately left installed -- writing an entry harms nothing, and
    leaving it means this file does not quietly change what the index holds.

    The alternative -- teaching 28 assertions to accept ``None`` -- would
    blind them to a rewrite that really had stopped firing, which is the one
    thing they exist to catch.
    """
    if not sk.is_applied():
        yield
        return
    from algan.taichi_compat import submodule

    kernel_cls = submodule("lang.kernel").Kernel
    patched = kernel_cls._try_load_fastcache
    original = getattr(patched, "_algan_original", None)
    if original is None:
        yield  # not our hook after all; leave it alone
        return
    kernel_cls._try_load_fastcache = original
    try:
        yield
    finally:
        kernel_cls._try_load_fastcache = patched


#: Off the daemon, and on this process's compiler, for every child below.
CHILD_ENV = {
    "ALGAN_USE_DAEMON": "0",
    "ALGAN_AUTO_DAEMON": "0",
    "ALGAN_TAICHI_BACKEND": BACKEND,
}

#: What both compilers say about a ``return`` they will not take.
REFUSAL = "Return inside non-static if/for"


@pytest.fixture(scope="module", autouse=True)
def _compiler():
    """One runtime for the file; every kernel below is materialized for real."""
    init_taichi()


def _raw(decorated):
    """The undecorated Python function behind a ``@ti.func``.

    ``update_wrapper``/``functools.wraps`` leaves ``__wrapped__`` on the
    decorator's callable on both compilers, and the rewrite records its
    decision there.
    """
    return inspect.unwrap(decorated)


def _outcome(decorated):
    return getattr(_raw(decorated), "_algan_early_return", None)


def _f32(values):
    return torch.tensor(values, dtype=torch.float32)


def _zeros(n):
    return torch.zeros(n, dtype=torch.float32)


def _i32(n):
    return torch.zeros(n, dtype=torch.int32)


# ---------------------------------------------------------------------------
# The hook itself
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_the_rewrite_is_live_on_this_compiler():
    """A compiler release the rewrite does not know turns it off silently.

    Marked ``fast`` alone in this file: it is the one test here that a change
    *elsewhere* breaks (a compiler bump in ``pyproject.toml``, a new backend in
    ``taichi_compat``, a rename in the compile path the hook wraps), and when
    it is off the only symptom is a shader stage that used to compile now
    failing with the compiler's own message. The rest are feature tests.
    """
    assert er.skipped_reason() is None, er.skipped_reason()
    assert er._APPLIED

    @ti.func
    def clamp01(x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = clamp01(src[i])

    src = _f32([-3.0, 0.25, 5.0])
    got = _zeros(3)
    run(got, src)
    # The module's `_no_source_key_shortcut` fixture guarantees a full
    # transform here, so a missing tag really does mean the rewrite did not
    # fire rather than that a cache answered first.
    assert _outcome(clamp01) == er.REWRITTEN
    assert got.tolist() == [0.0, 0.25, 1.0]


# ---------------------------------------------------------------------------
# `if` / `elif` / nested `if`
# ---------------------------------------------------------------------------


def test_a_return_under_a_runtime_if():
    @ti.func
    def early(x):
        if x < 0.0:
            return 0.0
        return x

    @ti.func
    def single_exit(x):
        v = x
        if x < 0.0:
            v = 0.0
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])
            want[i] = single_exit(src[i])

    src = _f32([-2.0, -0.5, 0.0, 0.5, 3.0])
    got, want = _zeros(5), _zeros(5)
    run(got, want, src)
    assert _outcome(early) == er.REWRITTEN
    assert _outcome(single_exit) == er.UNTOUCHED
    assert got.tolist() == want.tolist() == [0.0, 0.0, 0.0, 0.5, 3.0]


def test_returns_under_if_elif_else():
    @ti.func
    def early(x):
        if x < -1.0:
            return -1.0
        elif x > 1.0:
            return 1.0
        else:
            return x * 2.0

    @ti.func
    def single_exit(x):
        v = x * 2.0
        if x < -1.0:
            v = -1.0
        elif x > 1.0:
            v = 1.0
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])
            want[i] = single_exit(src[i])

    src = _f32([-4.0, -1.0, 0.0, 0.5, 9.0])
    got, want = _zeros(5), _zeros(5)
    run(got, want, src)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [-1.0, -2.0, 0.0, 1.0, 1.0]


def test_returns_under_nested_ifs():
    @ti.func
    def early(x, y):
        if x > 0.0:
            if y > 0.0:
                return 1.0
            return 2.0
        if y > 0.0:
            return 3.0
        return 4.0

    @ti.func
    def single_exit(x, y):
        v = 4.0
        if x > 0.0:
            v = 2.0
            if y > 0.0:
                v = 1.0
        elif y > 0.0:
            v = 3.0
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        xs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ys: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(xs.shape[0]):
            got[i] = early(xs[i], ys[i])
            want[i] = single_exit(xs[i], ys[i])

    xs = _f32([1.0, 1.0, -1.0, -1.0])
    ys = _f32([1.0, -1.0, 1.0, -1.0])
    got, want = _zeros(4), _zeros(4)
    run(got, want, xs, ys)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


def test_a_return_inside_a_while_breaks_out_of_it():
    """The ``while`` must stop at the hit, not merely stop *storing*.

    ``marks`` records every iteration the loop actually entered, so a rewrite
    that guarded the body instead of breaking would leave the whole array
    marked and the trip count wrong.
    """

    @ti.func
    def early(src: ti.template(), marks: ti.template(), t):
        i = 0
        while i < src.shape[0]:
            marks[i] = 1
            if src[i] >= t:
                return i
            i = i + 1
        return -1

    @ti.func
    def single_exit(src: ti.template(), marks: ti.template(), t):
        found = -1
        i = 0
        while i < src.shape[0]:
            marks[i] = 1
            if src[i] >= t:
                found = i
                break
            i = i + 1
        return found

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        got_marks: ti.types.ndarray(dtype=ti.i32, ndim=1),
        want_marks: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, got_marks, t)
        out[1] = single_exit(src, want_marks, t)

    src = _f32([0.0, 1.0, 5.0, 6.0, 7.0])
    out, got_marks, want_marks = _i32(2), _i32(5), _i32(5)
    run(out, got_marks, want_marks, src, 5.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [2, 2]
    assert got_marks.tolist() == want_marks.tolist() == [1, 1, 1, 0, 0]

    # And the not-found path still runs the loop out and returns the fallback.
    out, got_marks, want_marks = _i32(2), _i32(5), _i32(5)
    run(out, got_marks, want_marks, src, 99.0)
    assert out.tolist() == [-1, -1]
    assert got_marks.tolist() == want_marks.tolist() == [1, 1, 1, 1, 1]


def test_a_return_inside_a_for_nested_in_a_while_breaks_the_inner_loop():
    """A ``for`` inside another runtime loop *is* breakable, and is broken.

    The outer loop is a ``while``: a ``for`` there would be the func's
    outermost runtime loop, which is refused (see the decline tests below).
    """

    @ti.func
    def early(src: ti.template(), marks: ti.template(), t):
        r = 0
        while r < src.shape[0]:
            for c in range(src.shape[1]):
                marks[r, c] = 1
                if src[r, c] >= t:
                    return r * src.shape[1] + c
            r = r + 1
        return -1

    @ti.func
    def single_exit(src: ti.template(), marks: ti.template(), t):
        found = -1
        r = 0
        while r < src.shape[0]:
            for c in range(src.shape[1]):
                marks[r, c] = 1
                if src[r, c] >= t:
                    found = r * src.shape[1] + c
                    break
            if found >= 0:
                break
            r = r + 1
        return found

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        got_marks: ti.types.ndarray(dtype=ti.i32, ndim=2),
        want_marks: ti.types.ndarray(dtype=ti.i32, ndim=2),
        src: ti.types.ndarray(dtype=ti.f32, ndim=2),
        t: ti.f32,
    ):
        out[0] = early(src, got_marks, t)
        out[1] = single_exit(src, want_marks, t)

    src = _f32([[0.0, 1.0], [9.0, 9.0]])
    out = _i32(2)
    got_marks = torch.zeros(2, 2, dtype=torch.int32)
    want_marks = torch.zeros(2, 2, dtype=torch.int32)
    run(out, got_marks, want_marks, src, 5.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [2, 2]
    # Row 0 fully scanned, row 1 stopped at its first column: the inner loop
    # broke rather than idling to the end, and the `while` broke after it.
    assert got_marks.tolist() == want_marks.tolist() == [[1, 1], [1, 0]]


def test_a_return_in_the_funcs_outermost_for_is_refused():
    """The rewrite refuses it rather than body-guarding it.

    Guarding the body *compiles*, which is the problem: wherever the func is
    inlined at a kernel's top level that ``for`` is the offloaded loop, its
    iterations run in parallel, and the flag and value declared outside it are
    shared by iterations racing to set them -- one kernel, one input with two
    matches, 20 launches, both indices returned. The pass rewrites source and
    cannot see the call site, so it refuses the shape.
    """

    @ti.func
    def early(src: ti.template(), t):
        for i in range(src.shape[0]):
            if src[i] >= t:
                return i
        return -1

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, t)

    with (
        pytest.warns(AlganWarning, match="outermost `for`"),
        pytest.raises(Exception, match=REFUSAL),
    ):
        run(_i32(1), _f32([0.0, 9.0]), 5.0)
    outcome = str(_outcome(early))
    assert outcome.startswith("declined:")
    assert "`while`" in outcome, outcome


def test_the_refusal_is_lexical_and_does_not_look_at_the_call_site():
    """Called from inside a kernel's own loop the func's ``for`` would be
    serial and the guard would be correct -- and it is still refused. The
    decision is made when the source is parsed, once, before any call site is
    known; guessing at one is what this rule exists to avoid.
    """

    @ti.func
    def early(src: ti.template(), t):
        for i in range(src.shape[0]):
            if src[i] >= t:
                return i
        return -1

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        for _ in range(1):  # the func's `for` would be nested and serial here
            out[0] = early(src, t)

    with pytest.warns(AlganWarning), pytest.raises(Exception, match=REFUSAL):
        run(_i32(1), _f32([0.0, 9.0]), 5.0)
    assert str(_outcome(early)).startswith("declined:")


def test_the_hand_written_break_does_not_compile_there_either():
    """The other half of the argument: a ``break`` in that loop is rejected by
    the compiler at a kernel's top level, so there is no rewrite to make. Both
    spellings of the ``for`` are out; the ``while`` below is the way.
    """

    @ti.func
    def with_break(src: ti.template(), t):
        found = -1
        for i in range(src.shape[0]):
            if src[i] >= t:
                found = i
                break
        return found

    @ti.kernel
    def run_break(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = with_break(src, t)

    with pytest.raises(Exception, match="outermost loop"):
        run_break(_i32(1), _f32([0.0, 1.0, 9.0, 2.0]), 5.0)


def test_the_while_spelling_of_the_same_search_compiles_at_a_kernels_top_level():
    """The documented alternative, in the position the ``for`` is refused in.

    A ``while`` stays serial wherever it is inlined, so its ``break`` compiles
    at a kernel's top level and the first hit is the answer -- deterministically,
    with several matches, over repeated launches.
    """

    @ti.func
    def early(src: ti.template(), t):
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                return i
            i = i + 1
        return -1

    @ti.func
    def single_exit(src: ti.template(), t):
        found = -1
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                found = i
                break
            i = i + 1
        return found

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, t)
        out[1] = single_exit(src, t)

    src = _f32([0.0, 1.0, 9.0, 2.0, 8.0])
    for _ in range(5):
        out = _i32(2)
        run(out, src, 5.0)
        assert out.tolist() == [2, 2]
    assert _outcome(early) == er.REWRITTEN

    out = _i32(2)
    run(out, src, 99.0)
    assert out.tolist() == [-1, -1]


def test_a_return_inside_a_static_for():
    """A statically unrolled loop cannot be broken from a runtime ``if``
    either, so it gets the same body guard.
    """

    @ti.func
    def early(v: ti.template(), t):
        for k in ti.static(range(4)):
            if v[k] >= t:
                return k
        return -1

    @ti.func
    def single_exit(v: ti.template(), t):
        found = -1
        for k in ti.static(range(4)):
            if found < 0:  # noqa: SIM102 -- the nesting mirrors the rewrite's own guard
                if v[k] >= t:
                    found = k
        return found

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        for _ in range(1):
            out[0] = early(src, t)
            out[1] = single_exit(src, t)

    src = _f32([0.0, 6.0, 7.0, 8.0])
    out = _i32(2)
    run(out, src, 5.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [1, 1]

    out = _i32(2)
    run(out, src, 99.0)
    assert out.tolist() == [-1, -1]


def test_a_return_after_a_loop_sees_what_the_loop_computed():
    """The statements after the converted loop are wrapped, not moved: they
    still run, and still see the loop's accumulator.
    """

    @ti.func
    def early(src: ti.template(), t):
        total = 0.0
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                return -1.0
            total = total + src[i]
            i = i + 1
        total = total * 2.0
        return total

    @ti.func
    def single_exit(src: ti.template(), t):
        found = 0
        total = 0.0
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                found = 1
                break
            total = total + src[i]
            i = i + 1
        out = -1.0
        if found == 0:
            out = total * 2.0
        return out

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        for _ in range(1):
            out[0] = early(src, t)
            out[1] = single_exit(src, t)

    src = _f32([1.0, 2.0, 3.0])
    out = _zeros(2)
    run(out, src, 99.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [12.0, 12.0]

    out = _zeros(2)
    run(out, src, 2.5)
    assert out.tolist() == [-1.0, -1.0]


def test_a_continue_beside_an_early_return():
    """The loop's own ``continue`` keeps working beside the converted return."""

    @ti.func
    def early(src: ti.template(), t):
        total = 0.0
        i = 0
        while i < src.shape[0]:
            i = i + 1
            if src[i - 1] < 0.0:
                continue
            if src[i - 1] >= t:
                return -1.0
            total = total + src[i - 1]
        return total

    @ti.func
    def single_exit(src: ti.template(), t):
        found = 0
        total = 0.0
        i = 0
        while i < src.shape[0]:
            i = i + 1
            if src[i - 1] < 0.0:
                continue
            if src[i - 1] >= t:
                found = 1
                break
            total = total + src[i - 1]
        v = total
        if found == 1:
            v = -1.0
        return v

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, t)
        out[1] = single_exit(src, t)

    src = _f32([1.0, -5.0, 2.0])
    out = _zeros(2)
    run(out, src, 99.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [3.0, 3.0]

    out = _zeros(2)
    run(out, src, 1.5)
    assert out.tolist() == [-1.0, -1.0]


def test_a_while_inside_the_funcs_outermost_for_is_refused_too():
    """The refusal is about where the ``return`` *is*, not what it is directly
    under: a hit inside the ``while`` still leaves the enclosing ``for`` -- the
    offloaded one -- to be exited by a shared flag.
    """

    @ti.func
    def early(src: ti.template(), t):
        for r in range(src.shape[0]):
            c = 0
            while c < src.shape[1]:
                if src[r, c] >= t:
                    return r * 10 + c
                c = c + 1
        return -1

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.i32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=2),
        t: ti.f32,
    ):
        out[0] = early(src, t)

    with (
        pytest.warns(AlganWarning, match="outermost `for`"),
        pytest.raises(Exception, match=REFUSAL),
    ):
        run(_i32(1), _f32([[0.0, 0.0], [0.0, 9.0]]), 5.0)
    assert str(_outcome(early)).startswith("declined:")


def test_a_runtime_return_nested_inside_a_static_if():
    """A compile-time gate around a runtime early return: the declarations go
    in ahead of the gate, so the body is whole whichever way it resolves.
    """

    @ti.func
    def early(x, gate: ti.template()):
        if ti.static(gate):  # noqa: SIM102 -- `and` would demote the compile-time gate to a runtime one
            if x < 0.0:
                return 0.0
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        gate: ti.template(),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i], gate)

    src = _f32([-2.0, 3.0])
    on, off = _zeros(2), _zeros(2)
    run(on, src, 1)
    run(off, src, 0)
    assert _outcome(early) == er.REWRITTEN
    assert on.tolist() == [0.0, 3.0]
    assert off.tolist() == [-2.0, 3.0]


# ---------------------------------------------------------------------------
# What is returned: void, several values, a vector, an annotated type
# ---------------------------------------------------------------------------


def test_a_void_func_with_a_bare_return():
    @ti.func
    def early(out: ti.template(), i, x):
        if x < 0.0:
            return
        out[i] = x
        out[i] = out[i] + 1.0

    @ti.func
    def single_exit(out: ti.template(), i, x):
        if x >= 0.0:
            out[i] = x
            out[i] = out[i] + 1.0

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            early(got, i, src[i])
            single_exit(want, i, src[i])

    src = _f32([-1.0, 2.0, -3.0, 4.0])
    got, want = _zeros(4), _zeros(4)
    run(got, want, src)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [0.0, 3.0, 0.0, 5.0]


def test_a_multi_value_return():
    @ti.func
    def early(a, b):
        if b == 0.0:
            return -1.0, -2.0
        return a / b, a * b

    @ti.func
    def single_exit(a, b):
        u = a / b
        v = a * b
        if b == 0.0:
            u = -1.0
            v = -2.0
        return u, v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a: ti.f32,
        b: ti.f32,
    ):
        got[0], got[1] = early(a, b)
        want[0], want[1] = single_exit(a, b)

    got, want = _zeros(2), _zeros(2)
    run(got, want, 3.0, 2.0)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [1.5, 6.0]

    got, want = _zeros(2), _zeros(2)
    run(got, want, 3.0, 0.0)
    assert got.tolist() == want.tolist() == [-1.0, -2.0]


def test_a_vector_return_declared_from_the_annotation():
    @ti.func
    def early(x) -> ti.math.vec3:
        if x < 0.0:
            return ti.math.vec3(1.0, 0.0, 0.0)
        if x > 1.0:
            return ti.math.vec3(0.0, 0.0, 1.0)
        return ti.math.vec3(0.0, x, 0.0)

    @ti.func
    def single_exit(x) -> ti.math.vec3:
        v = ti.math.vec3(0.0, x, 0.0)
        if x < 0.0:
            v = ti.math.vec3(1.0, 0.0, 0.0)
        elif x > 1.0:
            v = ti.math.vec3(0.0, 0.0, 1.0)
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=2),
        want: ti.types.ndarray(dtype=ti.f32, ndim=2),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            a = early(src[i])
            b = single_exit(src[i])
            for k in ti.static(range(3)):
                got[i, k] = a[k]
                want[i, k] = b[k]

    src = _f32([-1.0, 0.5, 4.0])
    got = torch.zeros(3, 3, dtype=torch.float32)
    want = torch.zeros(3, 3, dtype=torch.float32)
    run(got, want, src)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist()
    assert got.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_an_annotation_declares_a_value_no_return_expression_could():
    """Every ``return`` here is unhoistable (a non-constant subscript), so the
    annotation is the only thing that can type the value variable -- and
    without it the pass declines and the compiler refuses the func.
    """

    @ti.func
    def early(src: ti.template(), t) -> ti.f32:
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                return src[i] - t
            i = i + 1
        return src[src.shape[0] - 1]

    @ti.func
    def single_exit(src: ti.template(), t) -> ti.f32:
        v = src[src.shape[0] - 1]
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                v = src[i] - t
                break
            i = i + 1
        return v

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, t)
        out[1] = single_exit(src, t)

    src = _f32([1.0, 8.5, 9.0])
    out = _zeros(2)
    run(out, src, 5.0)
    assert _outcome(early) == er.REWRITTEN
    assert out.tolist() == [3.5, 3.5]

    out = _zeros(2)
    run(out, src, 99.0)
    assert out.tolist() == [9.0, 9.0]


def test_without_the_annotation_the_same_body_is_declined():
    @ti.func
    def early(src: ti.template(), t):
        i = 0
        while i < src.shape[0]:
            if src[i] >= t:
                return src[i] - t
            i = i + 1
        return src[src.shape[0] - 1]

    @ti.kernel
    def run(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.f32,
    ):
        out[0] = early(src, t)

    with (
        pytest.warns(AlganWarning, match="cannot handle"),
        pytest.raises(Exception, match=REFUSAL),
    ):
        run(_zeros(1), _f32([1.0, 2.0]), 5.0)
    outcome = str(_outcome(early))
    assert outcome.startswith("declined:")
    assert "annotate the return type" in outcome, outcome


def test_the_value_variable_is_typed_from_a_return_that_carries_a_type():
    """A bare literal says nothing about the func's value type, and taichi
    types a local from its first assignment: initialising from ``0`` and then
    assigning an ``f32`` truncates every answer. Where a typed return
    expression is hoistable, that is the one to declare from.
    """

    @ti.func
    def early(x):
        if x < 0.0:
            return 0
        return x

    @ti.func
    def single_exit(x):
        v = x
        if x < 0.0:
            v = 0
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])
            want[i] = single_exit(src[i])

    src = _f32([-1.5, 0.5, 2.25])
    got, want = _zeros(3), _zeros(3)
    run(got, want, src)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [0.0, 0.5, 2.25]


def test_a_negative_literal_does_not_pass_for_a_typed_expression():
    """``-1`` is a ``UnaryOp`` over a ``Constant``, not a ``Constant``: taken
    for an expression it would declare an ``i32`` and truncate ``x``.
    """

    @ti.func
    def early(x):
        if x < 0.0:
            return -1
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])

    got = _zeros(2)
    run(got, _f32([-1.0, 2.5]))
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == [-1.0, 2.5]


def test_a_float_constant_return_is_not_truncated_by_an_integer_one():
    """Same hazard, both returns literal: ``0`` first, ``1.5`` second. The
    value is a float because one of the answers is.
    """

    @ti.func
    def early(x):
        if x < 0.0:
            return 0
        return 1.5

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])

    got = _zeros(2)
    run(got, _f32([-1.0, 1.0]))
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == [0.0, 1.5]


# ---------------------------------------------------------------------------
# Composition: two call sites, two funcs, two compiles
# ---------------------------------------------------------------------------


def test_one_func_called_from_two_kernels_and_from_two_places():
    """The rewrite runs per call site (the tree is re-parsed each time), so a
    second kernel must get the same body, not a doubly-rewritten one.
    """

    @ti.func
    def early(x):
        if x < 0.0:
            return -x
        return x

    @ti.kernel
    def one(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i]) + early(-src[i])

    @ti.kernel
    def two(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])

    src = _f32([-2.0, 3.0])
    a, b = _zeros(2), _zeros(2)
    one(a, src)
    two(b, src)
    assert _outcome(early) == er.REWRITTEN
    assert a.tolist() == [4.0, 6.0]
    assert b.tolist() == [2.0, 3.0]


def test_a_func_with_an_early_return_calling_another_one():
    """Both bodies introduce ``__algan_ret_flag``/``__algan_ret_val``; the
    inlined callee must not clobber the caller's.
    """

    @ti.func
    def inner(x):
        if x < 0.0:
            return 0.0
        return x * 2.0

    @ti.func
    def outer(x, y):
        if x > 10.0:
            return 100.0
        if y > 0.0:
            return inner(x) + inner(y)
        return inner(x)

    @ti.func
    def inner_ref(x):
        v = x * 2.0
        if x < 0.0:
            v = 0.0
        return v

    @ti.func
    def outer_ref(x, y):
        v = 0.0
        if x > 10.0:
            v = 100.0
        elif y > 0.0:
            v = inner_ref(x) + inner_ref(y)
        else:
            v = inner_ref(x)
        return v

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        want: ti.types.ndarray(dtype=ti.f32, ndim=1),
        xs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ys: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(xs.shape[0]):
            got[i] = outer(xs[i], ys[i])
            want[i] = outer_ref(xs[i], ys[i])

    xs = _f32([20.0, 3.0, 3.0, -4.0])
    ys = _f32([0.0, 5.0, -1.0, -1.0])
    got, want = _zeros(4), _zeros(4)
    run(got, want, xs, ys)
    assert _outcome(outer) == er.REWRITTEN
    assert _outcome(inner) == er.REWRITTEN
    assert got.tolist() == want.tolist() == [100.0, 16.0, 6.0, 0.0]


def test_calling_an_early_return_func_from_inside_runtime_control_flow():
    """The rewritten body ends in a top-level ``return``, which is legal in a
    func inlined under a runtime ``if`` or ``for`` -- as long as nothing in it
    is left under one.
    """

    @ti.func
    def early(x):
        if x < 0.0:
            return 0.0
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            if src[i] != 0.0:
                acc = 0.0
                for _ in range(2):
                    acc = acc + early(src[i])
                got[i] = acc

    src = _f32([-2.0, 0.0, 3.0])
    got = _zeros(3)
    run(got, src)
    assert _outcome(early) == er.REWRITTEN
    assert got.tolist() == [0.0, 0.0, 6.0]


def test_compiling_the_same_body_twice_in_one_process_agrees():
    """The second compile of an identical func/kernel pair -- a cache hit on
    the offline kernel cache, whose key is the frontend IR the rewrite
    produced -- must give the same answer as the first.
    """

    def build():
        @ti.func
        def early(x):
            if x < 0.0:
                return 0.0
            return x * 3.0

        @ti.kernel
        def run(
            got: ti.types.ndarray(dtype=ti.f32, ndim=1),
            src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ):
            for i in range(src.shape[0]):
                got[i] = early(src[i])

        return early, run

    src = _f32([-1.0, 2.0])
    first_func, first = build()
    second_func, second = build()
    a, b = _zeros(2), _zeros(2)
    first(a, src)
    second(b, src)
    assert _outcome(first_func) == _outcome(second_func) == er.REWRITTEN
    assert a.tolist() == b.tolist() == [0.0, 6.0]

    # And re-launching the first kernel after the second compiled still runs
    # the body it was compiled from.
    c = _zeros(2)
    first(c, src)
    assert c.tolist() == [0.0, 6.0]


# ---------------------------------------------------------------------------
# What the rewrite must not touch
# ---------------------------------------------------------------------------


def test_a_kernel_is_not_rewritten():
    """The hook fires only for a non-kernel, non-real func: a kernel with the
    same body still gets the compiler's own refusal.
    """

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            if src[i] < 0.0:
                return
            got[i] = src[i]

    with pytest.raises(Exception, match=REFUSAL):
        run(_zeros(2), _f32([1.0, -1.0]))
    assert _outcome(run) is None


def test_a_real_function_is_not_rewritten():
    """A real function has an exit of its own; the compiler takes its early
    ``return`` as written and the pass must keep its hands off.
    """

    @ti.real_func
    def early(x: ti.f32) -> ti.f32:
        if x < 0.0:
            return 0.0
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i])

    got = _zeros(3)
    run(got, _f32([-2.0, 0.5, 3.0]))
    assert got.tolist() == [0.0, 0.5, 3.0]
    assert _outcome(early) is None


def test_a_body_with_no_nested_return_is_handed_back_untouched():
    """The identity claim: an ordinary func's tree is not rebuilt, not
    reordered and not annotated, so its frontend IR -- and the offline-cache
    key hashed from it -- is exactly what it was before the pass existed.
    """
    source = (
        "def f(x, y):\n"
        "    z = x + y\n"
        "    if x > 0.0:\n"
        "        z = z * 2.0\n"
        "    for i in range(4):\n"
        "        z = z + i\n"
        "    while z > 100.0:\n"
        "        z = z - 1.0\n"
        "    return z\n"
    )
    tree = ast.parse(source)
    func_def = tree.body[0]
    before = ast.dump(func_def)
    assert er.rewrite_function_def(func_def) == er.UNTOUCHED
    assert ast.dump(func_def) == before
    assert tree.body[0] is func_def


def test_a_static_only_return_is_left_alone():
    """``return`` under ``ti.static`` has always been legal, so it is not a
    nested return and the body is untouched.
    """
    source = (
        "def f(x, flag):\n"
        "    if ti.static(flag):\n"
        "        return x\n"
        "    for k in ti.static(range(3)):\n"
        "        return x + k\n"
        "    return 0.0\n"
    )
    func_def = ast.parse(source).body[0]
    before = ast.dump(func_def)
    assert er.rewrite_function_def(func_def) == er.UNTOUCHED
    assert ast.dump(func_def) == before


def test_a_static_if_gate_keeps_working_around_an_early_return():
    """A compile-time gate and a runtime early return in one body."""

    @ti.func
    def early(x, gate: ti.template()):
        if ti.static(gate):
            return 42.0
        if x < 0.0:
            return 0.0
        return x

    @ti.kernel
    def run(
        got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1),
        gate: ti.template(),
    ):
        for i in range(src.shape[0]):
            got[i] = early(src[i], gate)

    src = _f32([-1.0, 2.0])
    on, off = _zeros(2), _zeros(2)
    run(on, src, 1)
    run(off, src, 0)
    assert on.tolist() == [42.0, 42.0]
    assert off.tolist() == [0.0, 2.0]


def test_statements_after_an_unconditional_return_are_dropped():
    source = (
        "def f(out, x):\n"
        "    if x < 0.0:\n"
        "        return 1.0\n"
        "        out[0] = 99.0\n"
        "    return 2.0\n"
    )
    rewritten = er.rewrite_source(source)
    assert "99.0" not in rewritten
    assert rewritten.count(er.FLAG) >= 2


# ---------------------------------------------------------------------------
# Bodies the pass declines
# ---------------------------------------------------------------------------


DECLINED = {
    "with_block": (
        "def f(x):\n"
        "    with ti.static(x):\n"
        "        if x > 0.0:\n"
        "            return 1.0\n"
        "    return 0.0\n"
    ),
    "static_loop_with_its_own_break": (
        "def f(v, t):\n"
        "    for k in ti.static(range(4)):\n"
        "        if v[k] > t:\n"
        "            return k\n"
        "        if v[k] < 0.0:\n"
        "            break\n"
        "    return -1\n"
    ),
    # A nested scope's `return` is its own, but the pass walks into one and
    # refuses -- *after* the conversion has already rewritten the `if` above
    # it in place, unless the conversion works on a copy. What matters is the
    # identity assertion below: a decline must leave the body as written, or
    # the compiler is handed a body assigning to a variable nothing declared.
    "a_nested_def_after_the_hoist_point": (
        "def f(x):\n"
        "    if x > 0.0:\n"
        "        return 1.0\n"
        "    def g():\n"
        "        return 2.0\n"
        "    return 3.0\n"
    ),
    "bare_return_mixed_with_a_valued_one": (
        "def f(out, x):\n"
        "    if x < 0.0:\n"
        "        return\n"
        "    if x > 1.0:\n"
        "        return 1.0\n"
        "    out[0] = x\n"
    ),
    "tuples_of_different_length": (
        "def f(x):\n    if x < 0.0:\n        return 0.0, 0.0\n    return 1.0, 2.0, 3.0\n"
    ),
    "no_declarable_value": (
        "def f(src, t):\n"
        "    i = 0\n"
        "    while i < src.shape[0]:\n"
        "        if src[i] >= t:\n"
        "            return src[i]\n"
        "        i = i + 1\n"
        "    return src[src.shape[0] - 1]\n"
    ),
    # The func's outermost runtime `for`: refused because guarding its body
    # compiles into a race wherever the func is inlined at a kernel's top
    # level (`test_a_return_in_the_funcs_outermost_for_is_refused`).
    "a_return_in_the_funcs_outermost_for": (
        "def f(src, t):\n"
        "    for i in range(src.shape[0]):\n"
        "        if src[i] >= t:\n"
        "            return i\n"
        "    return -1\n"
    ),
    "a_return_under_a_for_that_only_a_static_loop_encloses": (
        "def f(src, t):\n"
        "    for k in ti.static(range(2)):\n"
        "        for i in range(src.shape[0]):\n"
        "            if src[k, i] >= t:\n"
        "                return i\n"
        "    return -1\n"
    ),
    "an_impure_call_is_not_hoistable": (
        "def f(x):\n    if x < 0.0:\n        return ti.random()\n    return ti.random()\n"
    ),
}


@pytest.mark.parametrize("case", sorted(DECLINED))
def test_an_unsupported_body_is_declined_and_left_alone(case):
    """Declining must leave the body byte-identical: the compiler then reports
    its own error against the source the user actually wrote.
    """
    func_def = ast.parse(DECLINED[case]).body[0]
    before = ast.dump(func_def)
    with pytest.raises(er.EarlyReturnUnsupported):
        er.rewrite_function_def(func_def)
    assert ast.dump(func_def) == before


# ---------------------------------------------------------------------------
# Inertness against Algan's own kernels
# ---------------------------------------------------------------------------


def _is_compiler_function(node):
    """``@ti.func`` / ``@ti.kernel`` / ``@ti.pyfunc`` / ``@ti.real_func``.

    Attribute spellings only: every compiler function in ``algan/`` reaches
    the decorator through the bound module (``from algan.taichi_compat import
    ti``), and a bare ``@func`` in some unrelated module is not one of these.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and target.attr in {"func", "pyfunc", "real_func", "kernel"}
            and isinstance(target.value, ast.Name)
        ):
            return True
    return False


def test_no_function_in_algans_own_kernel_modules_is_rewritten():
    """The pass must be inert over the whole engine: every ``@ti.func`` and
    ``@ti.kernel`` Algan ships goes through the hook and comes back
    :data:`UNTOUCHED`, so no kernel's IR -- and no offline-cache key -- moves
    because this module is installed.

    The whole package, not just ``*_taichi.py``: a handful of modules outside
    that naming convention define kernels too (``pyproject.toml`` lists them
    for the same reason).
    """
    paths = [
        path
        for path in sorted((REPO_ROOT / "algan").rglob("*.py"))
        if "external_libraries" not in path.parts
    ]
    assert len(paths) > 100, f"only found {len(paths)} modules"
    seen = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not _is_compiler_function(node):
                continue
            seen += 1
            assert er.rewrite_function_def(node) == er.UNTOUCHED, (
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} `{node.name}` would be "
                "rewritten by the early-return pass"
            )
    assert seen > 200, f"only found {seen} compiler functions to check"


def test_the_counters_stay_at_zero_for_algans_own_funcs():
    """The runtime half of the same claim: importing every kernel module and
    materializing nothing new must not move the rewrite counter.
    """
    import importlib

    before = dict(er.STATS)
    for path in sorted((REPO_ROOT / "algan").rglob("*_taichi.py")):
        module = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        importlib.import_module(module)
    assert er.STATS["rewritten"] == before["rewritten"]


# ---------------------------------------------------------------------------
# The off switch
# ---------------------------------------------------------------------------


_CHILD = """
import json, sys
import algan  # noqa: F401
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils import taichi_early_return as er
import torch

init_taichi()


@ti.func
def clamp01(x):
    if x < 0.0:
        return 0.0
    return x


@ti.kernel
def run(got: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in range(src.shape[0]):
        got[i] = clamp01(src[i])


report = {"skipped_reason": er.skipped_reason(), "applied": er._APPLIED}
got = torch.zeros(2, dtype=torch.float32)
try:
    run(got, torch.tensor([-1.0, 2.0], dtype=torch.float32))
except Exception as exc:
    report["error"] = type(exc).__name__ + ": " + str(exc)
else:
    report["got"] = got.tolist()
print("REPORT " + json.dumps(report))
"""


def _child(env, tmp_path):
    # A real file, not ``-c``: the compilers read a kernel's *source* to parse
    # it, and a body typed on the command line has none.
    script = tmp_path / "early_return_child.py"
    script.write_text(_CHILD, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, **CHILD_ENV, **env},
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    for line in result.stdout.splitlines():
        if line.startswith("REPORT "):
            return json.loads(line[len("REPORT ") :])
    raise AssertionError(f"no REPORT line:\n{result.stdout}\n{result.stderr}")


def test_the_env_var_turns_the_rewrite_off(tmp_path):
    """``ALGAN_TAICHI_EARLY_RETURN=0`` hands the body back to the compiler,
    which rejects it exactly as it did before this module existed.
    """
    off = _child({"ALGAN_TAICHI_EARLY_RETURN": "0"}, tmp_path)
    assert off["skipped_reason"] == "ALGAN_TAICHI_EARLY_RETURN=0"
    assert off["applied"] is False
    assert REFUSAL in off.get("error", ""), off

    on = _child({"ALGAN_TAICHI_EARLY_RETURN": "1"}, tmp_path)
    assert on["skipped_reason"] is None
    assert on["applied"] is True
    assert on["got"] == [0.0, 2.0], on
