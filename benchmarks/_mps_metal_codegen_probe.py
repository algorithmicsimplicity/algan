"""Which kernel *spellings* Taichi can actually compile on the Metal backend.

The first render on an Apple GPU after MPS-friendly mode landed
(``DESIGN_mps_support.md`` §1.2) got past the f64 and the argument-count
blockers and died in the MSL compiler instead, on Taichi's own generated
source:

    RHI Error: cannot compile metal library from source:
        error: indirection requires pointer operand ('int' invalid)
            int tmp16_i32 = (int(long(_76))) * 8;
        error: cannot initialize a variable of type 'int' with an rvalue of
               type 'int (long)'

That is C++'s most vexing parse, in generated code: ``int(long(_76))`` reads
as the *function type* ``int(long)`` with a parameter named ``_76``, so the
``* 8`` after it is then parsed as a dereference. Nothing in Algan's kernel
source says it -- the nested functional cast is what Taichi's SPIR-V-to-MSL
step emits for a narrowing cast of a 64-bit ndarray load, and the pattern
(``ti.cast(some_i64_array[i], ti.i32)``) is everywhere in the renderer.

So the question this probe answers is not "does Algan render" but "which of
the ways of writing that narrowing survive MSL", which is what says whether
the fix is a spelling, an argument dtype, or an upstream patch.

**Every case runs in its own subprocess**, because the interesting failures
are not exceptions: Metal answers a shader that will not compile with
``computeFunction must not be nil`` and an unsupported atomic with
``bind_pipeline``, both ``SIGABRT``. A case that takes the process down has
still reported, by its exit code.

    uv run python benchmarks/_mps_metal_codegen_probe.py

Runs on whatever backend ``ALGAN_RENDER_DEVICE`` selects, so the Linux arm of
``mps_probe.yaml`` runs the identical cases on the CPU arch and establishes
that a failure is Metal's rather than the probe's.
"""

# No ``from __future__ import annotations`` here, deliberately, and
# ``benchmarks/*`` is exempted from the rule that would add one: every kernel
# below is annotated ``ti.types.ndarray()`` / ``ti.template()``, which Taichi
# evaluates at run time, and the import turns those into strings that fail
# with "Invalid type annotation".

import argparse
import os
import subprocess
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import taichi as ti  # noqa: E402
import torch  # noqa: E402

from algan.rendering.taichi_runtime import init_taichi  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402

N = 16
STRIDE = 8


# --------------------------------------------------------------- the cases
#
# Each builds a kernel, launches it, and checks the answer. The check matters:
# a backend that miscompiles a narrowing cast rather than refusing it is a
# worse outcome than an abort, and only a comparison can see it.


def case_baseline():
    """i32 in, i32 out, no casts. If this fails, nothing below means anything."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = x[i] + 1

    x = torch.arange(N, dtype=torch.int32)
    out = torch.zeros(N, dtype=torch.int32)
    k(x, out, N)
    assert torch.equal(out, x + 1), out


def case_i64_cast_only():
    """``ti.cast(i64_load, ti.i32)``, used as a plain value."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[i] = ti.cast(x[i], ti.i32)

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros(N, dtype=torch.int32)
    k(x, out, N)
    assert torch.equal(out, x.to(torch.int32)), out


def case_i64_cast_mul():
    """The suspect: a narrowed i64 load multiplied by a compile-time stride.

    ``sheet_lane_first_owner``'s ``base = ti.cast(band[i], ti.i32) *
    _AA_NUM_SAMPLES``, and ``_AA_NUM_SAMPLES`` is 8 -- which is the ``* 8`` in
    the error.
    """

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            b = ti.cast(x[i], ti.i32)
            out[b * STRIDE] = 1

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros(N * STRIDE, dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i64_cast_mul_via_temp():
    """The same, with the load pulled into its own name first.

    If Taichi's IR fuses the load into the cast either way this compiles to
    the same MSL, and the answer is that a source-level rewrite cannot help.
    """

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            raw = x[i]
            b = ti.cast(raw, ti.i32)
            out[b * STRIDE] = 1

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros(N * STRIDE, dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i64_mul_then_cast():
    """Multiply at i64 and narrow the product, so the cast is not a factor."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[ti.cast(x[i] * STRIDE, ti.i32)] = 1

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros(N * STRIDE, dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i32_array_mul():
    """The narrowing removed at the boundary: pass the array as i32.

    This is the fix MPS-friendly mode can actually apply, so whether it
    compiles is the question that decides the next commit.
    """

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            b = x[i]
            out[b * STRIDE] = 1

    x = torch.arange(N, dtype=torch.int32)
    out = torch.zeros(N * STRIDE, dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i64_cast_2d_index():
    """A narrowed i64 load as the ROW of a 2-D ndarray.

    Taichi computes a flat offset as ``row * row_stride``, so this reaches the
    same multiply without the source ever writing one -- which would make the
    pattern unavoidable by rewriting, and every ``tri_obj[row, r]``-shaped
    index in the renderer a site.
    """

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[ti.cast(x[i], ti.i32), 0] = 1

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros((N, STRIDE), dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i64_index_direct():
    """An i64 load used as an index with no cast at all."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            out[x[i]] = 1

    x = torch.arange(N, dtype=torch.int64)
    out = torch.zeros(N, dtype=torch.int32)
    k(x, out, N)
    assert int(out.sum()) == N, out


def case_i64_atomic_min():
    """§1.2's other Metal abort, re-asked now that the mode avoids it."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            ti.atomic_min(out[0], x[i])

    x = torch.arange(N, dtype=torch.int64) + 3
    out = torch.full((1,), 1 << 40, dtype=torch.int64)
    k(x, out, N)
    assert int(out[0]) == 3, out


def case_i32_atomic_min():
    """The mode's replacement for it."""

    @ti.kernel
    def k(x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
        for i in range(n):
            ti.atomic_min(out[0], ti.cast(x[i], ti.i32))

    x = torch.arange(N, dtype=torch.int64) + 3
    out = torch.full((1,), 2147483647, dtype=torch.int32)
    k(x, out, N)
    assert int(out[0]) == 3, out


def case_f32_accumulate():
    """The mode's replacement for the f64 accumulators."""

    @ti.kernel
    def k(
        x: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32, acc_t: ti.template()
    ):
        for i in range(n):
            total = ti.cast(0.0, acc_t)
            for _j in range(3):
                total += ti.cast(x[i], acc_t)
            out[i] = total

    x = torch.full((N,), 0.25, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    k(x, out, N, ti.f32)
    assert torch.allclose(out, torch.full((N,), 0.75)), out


CASES = {
    "baseline": case_baseline,
    "i64_cast_only": case_i64_cast_only,
    "i64_cast_mul": case_i64_cast_mul,
    "i64_cast_mul_via_temp": case_i64_cast_mul_via_temp,
    "i64_mul_then_cast": case_i64_mul_then_cast,
    "i32_array_mul": case_i32_array_mul,
    "i64_cast_2d_index": case_i64_cast_2d_index,
    "i64_index_direct": case_i64_index_direct,
    "i64_atomic_min": case_i64_atomic_min,
    "i32_atomic_min": case_i32_atomic_min,
    "f32_accumulate": case_f32_accumulate,
}


# ------------------------------------------------------------- the harness


def run_one(name: str) -> int:
    init_taichi()
    CASES[name]()
    print(f"{name}: OK")
    return 0


def run_all() -> int:
    device = render_device()
    print(f"render device : {device}")
    print(f"torch         : {torch.__version__}")
    print(f"taichi        : {ti.__version__}")
    print()
    width = max(len(name) for name in CASES)
    failures = []
    for name in CASES:
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", name],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print(f"  {name:<{width}}  PASS")
            continue
        failures.append(name)
        detail = _first_interesting_line(completed.stderr or completed.stdout)
        print(f"  {name:<{width}}  FAIL (exit {completed.returncode}) {detail}")
    print()
    if failures:
        print(f"{len(failures)} of {len(CASES)} cases failed: {', '.join(failures)}")
    else:
        print(f"all {len(CASES)} cases passed")
    # The probe reports; it does not judge. A failing case on a backend that
    # cannot run it is the measurement, not an error.
    return 0


def _first_interesting_line(text: str) -> str:
    """The line worth putting in a one-line table row."""
    markers = ("error:", "Error", "assertion", "Assertion", "Error:")
    for line in text.splitlines():
        if any(marker in line for marker in markers):
            return "-- " + line.strip()[:140]
    tail = [line for line in text.splitlines() if line.strip()]
    return "-- " + tail[-1].strip()[:140] if tail else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    args = parser.parse_args()
    sys.exit(run_one(args.case) if args.case else run_all())
