"""Prove `0009-python-3.14.patch` did what it claims, on whatever Python runs it.

Python 3.14 removed `ast.Str`, and Quadrants v1.3.0's frontend names it at four
sites. Merely *evaluating* the name raises `AttributeError`, so on 3.14 an
unpatched wheel does not miscompile these kernels -- it refuses them, as a
`QuadrantsCompilationError` wrapping `module 'ast' has no attribute 'Str'`. One
kernel per site, each compiled and launched on the CPU backend::

    python verify_python_314.py

and a nonzero exit if any of them fails. On 3.10-3.13 every case passes with or
without the patch (the removed branches were dead code there), so this is a
regression gate for cp314 and a cheap sanity run everywhere else -- which is why
the wheel workflow runs it on every leg rather than only on 3.14.

Measured before the patch was written, on a cp314 wheel with 0001-0008 only:
the first three cases fail and the f-string `print` passes. `build_JoinedStr`
reads `ast.Str` only on its error path (an f-string part that is neither a
`FormattedValue` nor a `Constant`), so the last case is a control that the
f-string machinery itself works, not a reproduction.

On linux/aarch64 `build_Assert` returns before reading the message ("Assert not
supported on linux arm64 currently"), so the two assert cases compile there
without reaching their sites. They still pass; they just prove less.
"""

# ruff: noqa: I002 -- I002 would insert `from __future__ import annotations`,
# which turns a kernel's runtime-evaluated annotations into strings and breaks
# decoration; see `verify_invariant_load.py` beside this file for the long form.
import sys
import traceback

import numpy as np

# Module level on purpose: kernel annotations resolve against this module's
# globals, so a function-local import would fail to decorate them.
import quadrants as qd


@qd.kernel
def assert_percent_message(a: qd.types.ndarray()):
    # ASTTransformer._is_string_mod_args
    for i in range(a.shape[0]):
        assert a[i] >= 0, "negative at %d" % i  # noqa: UP031 -- the %-format is the site


@qd.kernel
def assert_fstring_message(a: qd.types.ndarray()):
    # ASTTransformer.build_Assert, the non-Constant message branch
    for i in range(a.shape[0]):
        assert a[i] >= 0, f"negative {a[i]}"


@qd.kernel(graph=True)
def docstring_then_graph_do_while(
    x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)
):
    """FunctionDefTransformer._is_docstring, via the graph do_while validator."""
    while qd.graph.do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for _ in range(1):
            counter[()] = counter[()] - 1


@qd.kernel
def print_fstring(a: qd.types.ndarray()):
    # ASTTransformer.build_JoinedStr -- the control, see the module docstring
    for i in range(a.shape[0]):
        print(f"a[{i}] = {a[i]}")


def _assert_percent() -> None:
    assert_percent_message(np.ones(4, dtype=np.float32))


def _assert_fstring() -> None:
    assert_fstring_message(np.ones(4, dtype=np.float32))


def _graph_do_while() -> None:
    x = qd.ndarray(qd.i32, shape=(4,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(4, dtype=np.int32))
    counter.from_numpy(np.array(3, dtype=np.int32))
    docstring_then_graph_do_while(x, counter)
    qd.sync()
    # Three trips round the loop, so this also proves it ran, not just compiled.
    got = x.to_numpy()
    if not (got == 3).all():
        raise AssertionError(f"do_while body ran the wrong number of times: {got}")


def _print() -> None:
    print_fstring(np.ones(2, dtype=np.float32))


CASES = (
    ("assert with a %-formatted message", _assert_percent),
    ("assert with an f-string message", _assert_fstring),
    ("docstring + qd.graph.do_while", _graph_do_while),
    ("print of an f-string (control)", _print),
)


def main() -> int:
    # debug=True, so the assert kernels compile their checks rather than
    # having the frontend drop them. The sites are read either way; this is
    # about proving the compiled assert still behaves.
    qd.init(arch=qd.cpu, debug=True)
    failed = []
    for name, case in CASES:
        try:
            case()
            qd.sync()
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    version = ".".join(map(str, sys.version_info[:3]))
    if failed:
        print(f"0009 gate: {len(failed)} of {len(CASES)} failed on Python {version}")
        return 1
    print(f"0009 gate: all {len(CASES)} passed on Python {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
