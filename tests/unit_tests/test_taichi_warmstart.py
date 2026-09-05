"""The warm-start memoization must be invisible except in the clock.

``algan/utils/taichi_warmstart.py`` replaces two compiler internals with
memoizing copies. Its whole claim is byte-identity: the strings it hands back
are exactly the ones the compiler would have recomputed. These tests hold it to
that on whichever compiler is live, without compiling a kernel -- the memo is a
pure function of source text and node positions, so it can be exercised against
a duck-typed context and the implementation it replaced.

The end-to-end check is ``benchmarks/_taichi_warmstart_check.py``, which renders
with ``ALGAN_TAICHI_WARMSTART_VERIFY=1`` so every memoized value is recomputed
the original way inside a real materialization.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from textwrap import TextWrapper

import pytest

from algan.taichi_compat import BACKEND, submodule
from algan.utils.taichi_warmstart import (
    _make_fast_inside_class,
    _MemoizingTextwrap,
    _wrap80,
    skipped_reason,
)

#: Lines chosen for the edges of ``_wrap80``'s fast path rather than for
#: realism: the fast path is only valid for a short line whose whitespace is
#: all plain spaces, and every other case has to fall back.
TRICKY_LINES = [
    "",
    " ",
    "    ",
    "x",
    "    indented = 1",
    "    trailing = 1   ",
    "\ttabbed = 1",
    "mixed nbsp = 1",
    "line\x0bwith\x0cvertical = 1",
    "    " + "a" * 76,
    "    " + "a" * 77,
    "w " * 60,
    "unicode_ident_ᴀ = 1",
    "        ^^^^^^^^",
]


def _context_class():
    """The transformer context class on the live compiler, or ``None``."""
    utils = submodule("lang.ast.ast_transformer_utils")
    name = (
        "ASTTransformerContext" if BACKEND == "taichi" else "ASTTransformerFuncContext"
    )
    return getattr(utils, name, None)


class _FakeFunc:
    """Stands in for the decorated wrapper: a name, and somewhere to cache."""

    def __init__(self, name="fake_kernel"):
        self.func = type("_Raw", (), {"__name__": name})


class _FakeContext:
    """Everything ``get_pos_info`` reads, and nothing else."""

    def __init__(self, src, indent=0, lineno_offset=0):
        self.src = src
        self.file = "fake_kernel.py"
        self.indent = indent
        self.lineno_offset = lineno_offset
        self.func = _FakeFunc()


SAMPLE_SOURCE = '''\
def stage(x, y):
    """A docstring, because those are nodes too."""
    total = x + y
    if total > 3:
        for i in range(4):
            total += i * 2
    while total > 100:
        total -= 1
    return (
        total
        + 1
    )
'''


@pytest.mark.fast
def test_the_memoization_is_live_on_this_compiler():
    """A compiler release the patch does not know turns it off silently.

    Marked ``fast`` alone in this file, and for the reason `tests/README.md`
    gives: it is the one test here that a change *elsewhere* breaks -- a
    compiler bump in ``pyproject.toml``, a new backend in ``taichi_compat``, an
    env-var rename. The rest are feature tests for the memo itself.

    That is the failure `taichi_patches/PLAN.md` §6.1 found in the field: the
    quadrants port did not exist, the taichi version gate refused to fire, and
    renders paid ~25 s of frontend per process for it. If this fails after a
    compiler upgrade, the patch needs porting to the new internals -- not
    deleting.
    """
    reason = skipped_reason()
    assert reason is None, f"warm-start memoization is not installed: {reason}"


@pytest.mark.parametrize("line", TRICKY_LINES)
def test_wrap80_matches_the_textwrapper_it_replaces(line):
    assert _wrap80(line) == TextWrapper(width=80).wrap(line)


@pytest.mark.parametrize("line", TRICKY_LINES)
def test_memoized_fill_matches_textwrap_fill(line):
    shim = _MemoizingTextwrap()
    expected = textwrap.fill(line, tabsize=4, width=9999)
    assert shim.fill(line, tabsize=4, width=9999) == expected
    # Again, to take the cached path rather than the computing one.
    assert shim.fill(line, tabsize=4, width=9999) == expected


def test_memoized_fill_delegates_calls_it_was_not_written_for():
    shim = _MemoizingTextwrap()
    line = "a b c d e f g h"
    assert shim.fill(line, width=5) == textwrap.fill(line, width=5)
    assert shim._cache == {}, "a differently-parameterised fill must not be cached"


def test_memoizing_textwrap_delegates_every_other_attribute():
    shim = _MemoizingTextwrap()
    assert shim.dedent("    x\n    y\n") == textwrap.dedent("    x\n    y\n")
    assert shim.TextWrapper is textwrap.TextWrapper


def test_the_fill_cache_is_bounded():
    """A program that generates kernels from unique source must not leak."""
    shim = _MemoizingTextwrap()
    for i in range(100_002):
        shim.fill(f"line_{i} = {i}", tabsize=4, width=9999)
    assert len(shim._cache) < 100_000


def test_pos_info_memo_reproduces_the_original_for_every_node():
    """The memo against the implementation it replaced, node by node.

    A duck-typed context rather than a compiled kernel: ``get_pos_info`` reads
    only the source lines, the two offsets and the function name, so this
    exercises the real installed memo (including its cache) over a wider set of
    node shapes than one kernel would contain -- multi-line calls, `for`,
    `while`, `if`, a docstring, a trailing return.
    """
    ctx_cls = _context_class()
    if ctx_cls is None:
        pytest.skip(f"no transformer context class on {BACKEND}")
    memoized = ctx_cls.get_pos_info
    original = getattr(memoized, "_algan_original", None)
    if original is None:
        pytest.skip("get_pos_info is not the memoized version on this compiler")

    src = SAMPLE_SOURCE.splitlines()
    tree = ast.parse(SAMPLE_SOURCE)
    nodes = [node for node in ast.walk(tree) if hasattr(node, "lineno")]
    assert len(nodes) > 20, "the sample source stopped covering enough node kinds"

    for indent, lineno_offset in ((0, 0), (4, 17)):
        ctx = _FakeContext(src, indent=indent, lineno_offset=lineno_offset)
        for node in nodes:
            assert memoized(ctx, node) == original(ctx, node)
        # Second pass: every one of these is now a cache hit.
        for node in nodes:
            assert memoized(ctx, node) == original(ctx, node)
        assert ctx.func._algan_pos_info_cache, "nothing was cached"


def test_pos_info_memo_keys_on_the_context_offsets():
    """Two contexts over one function must not read each other's entries.

    The cache lives on the decorated wrapper, which outlives any single
    transform, and the same wrapper is transformed at different indents and
    line offsets (an inlined func, a re-materialized template). Those offsets
    are in the key for that reason.
    """
    ctx_cls = _context_class()
    memoized = getattr(ctx_cls, "get_pos_info", None)
    if memoized is None or not hasattr(memoized, "_algan_original"):
        pytest.skip("get_pos_info is not the memoized version on this compiler")

    src = SAMPLE_SOURCE.splitlines()
    node = ast.parse(SAMPLE_SOURCE).body[0].body[2]
    shared_func = _FakeFunc()

    first = _FakeContext(src, indent=0, lineno_offset=0)
    second = _FakeContext(src, indent=4, lineno_offset=17)
    first.func = second.func = shared_func

    assert memoized(first, node) != memoized(second, node)


#: A module that asks ``_inside_class`` from the places a decorator would: the
#: frame *enclosing* a class body (the one the compilers look at, ``level`` 3
#: from ``ask``), the module body itself, a plain function, and a class that
#: carries a decorator of its own. ``ASK`` is the function under test, bound in
#: by the test before the module runs.
_INSIDE_CLASS_SOURCE = """\
def ask(level):
    return ASK(level)


def deco(cls):
    return cls


class Holder:
    enclosing = ask(3)
    own_body = ask(2)


@deco
class Decorated:
    enclosing = ask(3)


module_body = ask(2)
too_deep = ask(200)


def in_function():
    return ask(2)


function_body = in_function()
"""


def _run_inside_class_module(tmp_path, ask):
    """Execute the module from a real file, so ``linecache`` can see it."""
    path = tmp_path / "inside_class_probe.py"
    path.write_text(_INSIDE_CLASS_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_inside_class_probe", path)
    module = importlib.util.module_from_spec(spec)
    module.ASK = ask
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return {
        "Holder.enclosing": module.Holder.enclosing,
        "Holder.own_body": module.Holder.own_body,
        "Decorated.enclosing": module.Decorated.enclosing,
        "module_body": module.module_body,
        "too_deep": module.too_deep,
        "function_body": module.function_body,
    }


def _installed_inside_class():
    fast = submodule("lang.kernel_impl")._inside_class
    original = getattr(fast, "_algan_original", None)
    if original is None:
        pytest.skip("_inside_class is not the patched version on this compiler")
    return fast, original


def test_inside_class_patch_answers_exactly_as_the_frame_walk(tmp_path):
    """The patch against the implementation it replaced, frame by frame.

    Both are handed the same ``level`` from the same call sites; the original
    is *called* one frame deeper here (through a lambda), which is the same
    adjustment the verify arm makes, so the frame each inspects is the same.
    The class case is asserted outright so the comparison cannot pass by both
    sides answering ``False`` to everything.
    """
    fast, original = _installed_inside_class()

    from_fast = _run_inside_class_module(tmp_path, fast)
    from_original = _run_inside_class_module(
        tmp_path, lambda level: original(level + 1)
    )

    assert from_fast == from_original
    assert from_fast["Holder.enclosing"] is True
    assert from_fast["module_body"] is False
    assert from_fast["too_deep"] is False
    assert from_fast["function_body"] is False


def test_inside_class_patch_verify_arm_agrees_with_the_original(tmp_path):
    """``ALGAN_TAICHI_WARMSTART_VERIFY=1`` builds this instance; it raises on
    the first disagreement, so running it over the same module is the check
    that the verify arm's own frame arithmetic is right.
    """
    _, original = _installed_inside_class()
    patterns = submodule("lang.kernel_impl")._KERNEL_CLASS_STACKFRAME_STMT_RES
    verifying = _make_fast_inside_class(original, patterns, verify=True)

    answers = _run_inside_class_module(tmp_path, verifying)

    assert answers["Holder.enclosing"] is True
    assert answers["module_body"] is False


def test_inside_class_patch_without_source_on_disk_answers_false():
    """A ``<string>`` module has no line to match, on either implementation."""
    fast, original = _installed_inside_class()
    namespace = {"FAST": fast, "ORIG": original}
    exec(  # noqa: S102 -- the point is a frame whose file does not exist
        compile(
            "class NoFile:\n    answers = (FAST(2), ORIG(2))\n",
            "<inside-class-probe>",
            "exec",
        ),
        namespace,
    )
    assert namespace["NoFile"].answers == (False, False)
