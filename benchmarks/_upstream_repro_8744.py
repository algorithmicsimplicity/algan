"""Upstream repro: taichi-dev/taichi#8744 -- a dead branch changes a ``@ti.func``'s result.

https://github.com/taichi-dev/taichi/issues/8744 ("Incorrect running result",
reporter ``emailweixu``, open, filed against Taichi 1.7.3, arch **cuda**).

**Symptom.** A ``@ti.func`` reads element 0 of one vector into ``x0``, then rebinds the
same Python name to a *second* vector and reads element 0 into ``x1``, and returns
``x0 == x1``. The two vectors hold different values, so the answer is False. Insert a
branch that is never taken at runtime (``if p: x0 = x0 + 0``, with ``p`` false for every
element) and the answer flips to True for every element: ``x0`` is miscompiled into a
second read of the *rebound* name. Renaming the second binding to a fresh Python name
also makes the bug go away, which is what points at the Python-name-to-IR-local
mapping in the frontend rather than at anything the branch itself does.

**Arch.** The report is CUDA-only and has no comments; nobody has said whether it is
GPU-specific. This script runs on whatever ``REPRO_ARCH`` names (default ``cpu``), so
the x64 arm of the question can be answered without a GPU.

**Deviation from the issue's code.** The original seeds the fields from torch with
``torch.set_default_device("cuda")`` and passes a torch bool tensor as the ndarray
result. This script uses numpy (fixed seed) and an ``i32`` result array so it needs no
torch and no GPU; the kernel and the ``@ti.func`` under test are otherwise character
-for-character the issue's. The issue's own ``expected`` line compares ``boxes0[:, 0]``
against ``boxes1[:, 1]``; that is a typo in the report (the func compares element 0 of
both), harmless there because random values differ either way. This script compares
element 0 of both, which is what the func actually computes.

**Extra variants**, because the gate needs to know how exposed Algan's own kernels are
and whether a workaround is bounded:

``control``
    the reported func with the two "useless" lines deleted -- says whether the dead
    branch is what moves the answer.
``rename``
    the dead branch kept, but the second binding uses fresh Python names
    (``bbox1``/``a1``). This is the workaround the reporter names; if it is clean the
    trigger is Python-name *rebinding* and avoiding it is a mechanical fix.
``inline``
    the same statements written straight into the kernel body with no ``@ti.func`` at
    all -- says whether inlining is required to hit it.

**Known workaround** (measured on Linux x64, both compilers, 2026-09-04): the miscompile
is gated on the optimizer. ``ti.init(advanced_optimization=False)`` -- which is what
Algan already runs with (``taichi_runtime.py``'s ``taichi_init_kwargs``, env
``ALGAN_ADV_OPT``, default off) -- makes every variant correct, and so does
``cfg_optimization=False`` alone, which narrows the suspect to the CFG store-to-load
forwarding pass. Set ``REPRO_ADV_OPT=0`` (or ``=1``) to pin the flag; unset means the
compiler's own default, which is what the issue reports against.

Usage::

    REPRO_BACKEND=taichi   REPRO_ARCH=cpu  python benchmarks/_upstream_repro_8744.py
    REPRO_BACKEND=quadrants REPRO_ARCH=cuda python benchmarks/_upstream_repro_8744.py

Prints one verdict line, ``REPRO-8744: REPRODUCES`` or ``REPRO-8744: CLEAN``, plus the
numbers behind it. Exits 0 either way; a non-zero exit means the script itself failed.
"""

import importlib
import os
import sys

import numpy as np

ti = importlib.import_module(os.environ.get("REPRO_BACKEND", "taichi"))

ARCH_NAME = os.environ.get("REPRO_ARCH", "cpu")
N = 10
SENTINEL = 15  # > N, so ``i == n`` is false for every element: the branch is dead.

INIT_KWARGS = {}
if "REPRO_ADV_OPT" in os.environ:
    INIT_KWARGS["advanced_optimization"] = bool(int(os.environ["REPRO_ADV_OPT"]))
if "REPRO_CFG_OPT" in os.environ:
    INIT_KWARGS["cfg_optimization"] = bool(int(os.environ["REPRO_CFG_OPT"]))

ti.init(arch=getattr(ti, ARCH_NAME), default_ip=ti.i32, default_fp=ti.f32, **INIT_KWARGS)

BOOL_T = getattr(ti, "u1", ti.i32)


@ti.func
def testf(box0, box1, p) -> BOOL_T:
    bbox = box0
    a = bbox[0]
    x0 = a
    # If the following two lines are commented out, the assert will pass.
    if p:
        x0 = x0 + 0
    # If the following variable is renamed to `bbox1`, the assert will pass
    # (bbox at the line after is also changed accordingly to `bbox1`).
    bbox = box1
    a = bbox[0]
    x1 = a
    return x0 == x1


@ti.func
def testf_control(box0, box1, p) -> BOOL_T:
    # Identical, minus the dead branch. `p` is still an argument so the two funcs
    # take the same signature.
    bbox = box0
    a = bbox[0]
    x0 = a
    bbox = box1
    a = bbox[0]
    x1 = a
    return x0 == x1


@ti.func
def testf_rename(box0, box1, p) -> BOOL_T:
    # The dead branch is kept; only the second binding gets fresh names.
    bbox = box0
    a = bbox[0]
    x0 = a
    if p:
        x0 = x0 + 0
    bbox1 = box1
    a1 = bbox1[0]
    x1 = a1
    return x0 == x1


@ti.kernel
def test(boxes0: ti.template(), boxes1: ti.template(), res: ti.types.ndarray(), n: int):
    for i in ti.ndrange(boxes0.shape[0]):
        box0 = boxes0[i]
        box1 = boxes1[i]
        res[i] = testf(box0, box1, i == n)


@ti.kernel
def test_control(
    boxes0: ti.template(), boxes1: ti.template(), res: ti.types.ndarray(), n: int
):
    for i in ti.ndrange(boxes0.shape[0]):
        box0 = boxes0[i]
        box1 = boxes1[i]
        res[i] = testf_control(box0, box1, i == n)


@ti.kernel
def test_rename(
    boxes0: ti.template(), boxes1: ti.template(), res: ti.types.ndarray(), n: int
):
    for i in ti.ndrange(boxes0.shape[0]):
        box0 = boxes0[i]
        box1 = boxes1[i]
        res[i] = testf_rename(box0, box1, i == n)


@ti.kernel
def test_inline(
    boxes0: ti.template(), boxes1: ti.template(), res: ti.types.ndarray(), n: int
):
    # No @ti.func at all: the same statements in the kernel body.
    for i in ti.ndrange(boxes0.shape[0]):
        box0 = boxes0[i]
        box1 = boxes1[i]
        p = i == n
        bbox = box0
        a = bbox[0]
        x0 = a
        if p:
            x0 = x0 + 0
        bbox = box1
        a = bbox[0]
        x1 = a
        res[i] = x0 == x1


def main() -> int:
    print(
        f"backend={ti.__name__} version={ti.__version__} arch={ARCH_NAME} "
        f"python={sys.version.split()[0]} init_kwargs={INIT_KWARGS or 'compiler defaults'}"
    )

    rng = np.random.default_rng(8744)
    b0 = rng.standard_normal((N, 2)).astype(np.float32)
    b1 = rng.standard_normal((N, 2)).astype(np.float32)

    boxes0 = ti.math.vec2.field(shape=(N,))
    boxes1 = ti.math.vec2.field(shape=(N,))
    boxes0.from_numpy(b0)
    boxes1.from_numpy(b1)

    expected = (b0[:, 0] == b1[:, 0]).astype(np.int32)
    variants = {
        "reported (func, dead branch, rebound name)": test,
        "control  (func, no dead branch)": test_control,
        "rename   (func, dead branch, fresh names)": test_rename,
        "inline   (no func, dead branch, rebound name)": test_inline,
    }

    print(f"  boxes0[:, 0]      = {np.array2string(b0[:, 0], precision=4)}")
    print(f"  boxes1[:, 0]      = {np.array2string(b1[:, 0], precision=4)}")
    print(f"  expected (x0==x1) = {expected.tolist()}")

    wrong = {}
    for label, kern in variants.items():
        res = np.zeros((N,), dtype=np.int32)
        kern(boxes0, boxes1, res, SENTINEL)
        wrong[label] = int((res != expected).sum())
        print(f"  {label:46s} -> {res.tolist()}  wrong {wrong[label]}/{N}")

    bad = wrong["reported (func, dead branch, rebound name)"]
    clean_variants = [label for label, n_bad in wrong.items() if not n_bad]
    if bad:
        print(
            f"REPRO-8744: REPRODUCES ({bad}/{N} elements wrong in the reported variant; "
            f"variants that stay correct: {len(clean_variants)}/{len(variants)})"
        )
        for label in clean_variants:
            print(f"  workaround that holds: {label.strip()}")
    elif len(clean_variants) == len(variants):
        print(f"REPRO-8744: CLEAN (0/{N} elements wrong in all {len(variants)} variants)")
    else:
        print(
            "REPRO-8744: CLEAN for the reported variant, but another variant is wrong "
            "-- investigate, that is a different bug"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
