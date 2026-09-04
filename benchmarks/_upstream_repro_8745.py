"""Upstream repro: taichi-dev/taichi#8745 -- Metal result depends on an unrelated field's shape.

https://github.com/taichi-dev/taichi/issues/8745 ("Incorrect running result",
reporter ``xiaobo-lab``, open, Taichi 1.7.3, Apple M2 Pro, arch **metal**).

**Symptom.** A kernel fills a ``f32`` field, then copies a small ``u16`` lookup field
into a ``u16`` output field. ``shape_field`` holds ``[0, 0, 6]``, so
``index.to_numpy()[0, 1, 1]`` must be ``[0 0 6]``. On Metal it is, but only when
``shape_field`` is declared with shape ``(4,)``: declared ``(3,)`` -- the size the
kernel actually uses -- the answer comes back ``[0 0 0]``. Deleting the first loop (the
``input_`` fill, which the second loop does not read) also makes it correct. So the
result depends both on the *padding* of an unrelated field and on the presence of a
loop that writes nothing the answer depends on; the shapes point at 16-bit storage
packing on the SPIR-V/MSL path.

**Arch.** Reported on **Metal only**, and this is the arm that needs the Mac runner.
Run it on ``cpu`` / ``cuda`` too: those establish the reference answer and say whether
the bug is Metal-specific, which is what decides how much a rebase has to care.

**Deviations from the issue's code.** (1) ``index[*input_i, shape_i]`` is rewritten as
``index[input_i[0], input_i[1], input_i[2], shape_i]`` -- identical semantics, but the
starred subscript is Python 3.11 syntax and this must run on the 3.10 the Quadrants
wheels also target. (2) Each variant runs in its own child process, because a field's
shape is baked into the SNode tree at declaration and the bug is precisely about layout:
declaring both a ``(3,)`` and a ``(4,)`` ``shape_field`` in one process would not be the
reported program. (3) The fill loop is gated on a ``ti.template()`` argument read through
``ti.static``, so the ``shape3`` variant compiles to the same IR as the issue's kernel
while ``shape3_nofill`` compiles to the issue's control.

Three variants:

``shape3``
    the reported failing program (``shape_field`` shape ``(3,)``).
``shape4``
    the reporter's control (``shape_field`` shape ``(4,)``, one unused slot).
``shape3_nofill``
    shape ``(3,)`` with the first loop deleted, the reporter's other control.

All three must print ``[0 0 6]``.

Usage::

    REPRO_BACKEND=taichi    REPRO_ARCH=metal python benchmarks/_upstream_repro_8745.py
    REPRO_BACKEND=quadrants REPRO_ARCH=cpu   python benchmarks/_upstream_repro_8745.py

On the Mac runner (`agent_guidance/gpu_harnesses.md`, "Mac runner"), dispatch
``run_on_mac.yaml`` with ``arms: mac-mps,linux-cpu`` and ``env`` (newline-separated)
holding ``REPRO_ARCH=metal`` and ``REPRO_BACKEND=...``; ``command`` is
``uv run python benchmarks/_upstream_repro_8745.py`` for the Taichi arm and
``uv run --with quadrants==1.3.0 python benchmarks/_upstream_repro_8745.py`` for the
Quadrants one. The MPS arm's patched-Taichi wheel is irrelevant here (this script never
imports algan and never touches an MPS tensor) but harmless; ``taichi_wheel_run_id:
none`` measures what an unpatched Mac user gets, which for *this* question is the same
thing. The ``linux-cpu`` control should print CLEAN, since it does below.

Prints one verdict line, ``REPRO-8745: REPRODUCES`` or ``REPRO-8745: CLEAN``, plus the
vector each variant produced. Exits 0 either way; a non-zero exit means the script
itself failed.

**Candidate fix to test on the Mac.** Quadrants commit ``7a9b6cb23`` (#384) decorates
every SPIR-V storage buffer ``Volatile`` when ``arch == metal``, with the comment that
the Metal/MoltenVK shader compiler "incorrectly hoists storage buffer loads out of loops
(LICM), causing stale reads when a buffer is written and re-read within the same loop".
That is this issue's shape exactly -- one loop writes ``input_``, the next re-reads the
same root buffer -- so the Quadrants arm is a direct test of that hypothesis, and
``taichi_patches/PLAN.md`` §5's "copy if" for #384 becomes "copy" if it comes back CLEAN
where Taichi 1.7.4 reproduces. Algan's ndarrays are storage buffers on the same path.
"""

import importlib
import os
import re
import subprocess
import sys

BACKEND = os.environ.get("REPRO_BACKEND", "taichi")
ARCH_NAME = os.environ.get("REPRO_ARCH", "cpu")
EXPECTED = [0, 0, 6]
VARIANTS = {
    # name: (shape_field length, run the input_ fill loop)
    "shape3": (3, True),
    "shape4": (4, True),
    "shape3_nofill": (3, False),
}


def child(name: str) -> int:
    """One variant, in its own process. Prints ``RESULT=[a, b, c]``."""
    n_shape, do_fill = VARIANTS[name]
    ti = importlib.import_module(BACKEND)
    ti.init(arch=getattr(ti, ARCH_NAME))

    input_ = ti.field(ti.f32, shape=(3, 2, 2))
    index = ti.field(ti.u16, shape=(3, 2, 2, 3))
    shape_field = ti.field(ti.u16, shape=(n_shape,))
    shape_field[2] = 6

    @ti.kernel
    def test(fill: ti.template()):
        if ti.static(fill):
            for input_i in ti.grouped(input_):
                input_[input_i] = 3
        for input_i in ti.grouped(input_):
            for shape_i in ti.ndrange(3):
                index[input_i[0], input_i[1], input_i[2], shape_i] = shape_field[shape_i]

    test(do_fill)
    print(f"RESULT={index.to_numpy()[0, 1, 1].tolist()}", flush=True)
    return 0


def run_variant(name: str) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name],
        capture_output=True,
        text=True,
        env={**os.environ, "REPRO_BACKEND": BACKEND, "REPRO_ARCH": ARCH_NAME},
    )
    match = re.search(r"^RESULT=(\[.*\])$", proc.stdout, re.M)
    value = None
    if match:
        value = [int(v) for v in re.findall(r"-?\d+", match.group(1))]
    tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()][-3:]
    return {"name": name, "value": value, "returncode": proc.returncode, "tail": tail}


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2])

    print(f"backend={BACKEND} arch={ARCH_NAME} python={sys.version.split()[0]}")
    print(f"  expected, every variant: {EXPECTED}")

    results = [run_variant(name) for name in VARIANTS]
    wrong = []
    for r in results:
        if r["value"] is None:
            print(f"  {r['name']:14s} -> FAILED (exit {r['returncode']})")
            for line in r["tail"]:
                print(f"      | {line}")
            wrong.append(r)
            continue
        verdict = "ok" if r["value"] == EXPECTED else "WRONG"
        print(f"  {r['name']:14s} -> {r['value']}  {verdict}")
        if r["value"] != EXPECTED:
            wrong.append(r)

    reported = next(r for r in results if r["name"] == "shape3")
    if reported["value"] != EXPECTED:
        others = [r["name"] for r in results if r["name"] != "shape3" and r not in wrong]
        print(
            f"REPRO-8745: REPRODUCES (shape3 gave {reported['value']}, expected "
            f"{EXPECTED}; variants that stay correct: {others or 'none'})"
        )
    elif wrong:
        print(
            f"REPRO-8745: CLEAN for the reported variant, but "
            f"{[r['name'] for r in wrong]} came back wrong -- investigate"
        )
    else:
        print(f"REPRO-8745: CLEAN (all {len(results)} variants gave {EXPECTED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
