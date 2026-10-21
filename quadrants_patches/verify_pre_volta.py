"""Prove `0003-cuda-pre-volta.patch` did what it claims, on a pre-Volta CUDA card.

0003 makes Quadrants usable below sm_70 (and its warp-reduction gate below
sm_75). Without it, pristine Quadrants dies in `qd.init(arch=qd.cuda)` on such
a card: the runtime module's one system-scoped atomic, an `atom.sys.cas.b64` in
`runtime_eval_adstack_max_reduce`, makes `cuModuleLoadDataEx` answer
`CUDA_ERROR_NOT_SUPPORTED` (`../taichi_patches/MIGRATION.md` §7, §11.1). With
it, that atomic is `atom.gpu.cas.b64`, the reduction kernels compile without
the `Cannot select: intrinsic %llvm.nvvm.activemask` abort, and the answers are
right. One command::

    python verify_pre_volta.py

It brings CUDA up, runs three reductions (f32 sum, i32 `atomic_max`, f64 sum)
and checks them against numpy, then reads every PTX module the run dumped and
reports the `atom.sys` count (must be 0 on a pre-Volta card), the device-scoped
atomics, and the `.target` / `.version` header (`sm_61` and `5.0` on a GTX
1050). Exit status 0 means all of that held.

On sm_70 and up the patch is a no-op by design, so this passes on any modern
card with or without it and proves nothing there: it prints `NOT PRE-VOLTA` so
the result is not mistaken for coverage. The negative control is the same
script against stock `quadrants` of the same version on the same card, which
must fail at `qd.init`.

First run on Kaggle's T4 (sm_75) with the v1.3.3 rebase wheel, 2026-10-21:
both the patched and the stock wheel pass, as they should above sm_70.
"""

# ruff: noqa: I002 -- I002 would insert `from __future__ import annotations`,
# which turns a kernel's runtime-evaluated annotations into strings and breaks
# decoration; see `verify_invariant_load.py` beside this file for the long form.
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

# Module level on purpose: kernel annotations resolve against this module's
# globals, so a function-local import would fail to decorate them.
import quadrants as qd


@qd.kernel
def reduce_sum(a: qd.types.ndarray(), out: qd.types.ndarray()):
    for i in range(a.shape[0]):
        out[0] += a[i]


@qd.kernel
def reduce_max(a: qd.types.ndarray(), out: qd.types.ndarray()):
    for i in range(a.shape[0]):
        qd.atomic_max(out[0], a[i])


def main():
    dump_dir = tempfile.mkdtemp(prefix="qd_pre_volta_")
    os.chdir(dump_dir)  # every PTX dump lands in the CWD
    qd.init(arch=qd.cuda, offline_cache=False, print_kernel_asm=True)
    from quadrants.lang import impl

    arch = str(impl.current_cfg().arch)
    if arch != str(qd.cuda):
        print(f"FAIL: qd.init(arch=qd.cuda) landed on {arch}")
        return 1

    n = 1 << 20
    rng = np.random.default_rng(0)
    f32 = rng.random(n, dtype=np.float32)
    sum32 = np.zeros(1, np.float32)
    reduce_sum(f32, sum32)
    i32 = rng.integers(0, 1 << 30, n, dtype=np.int32)
    max32 = np.zeros(1, np.int32)
    reduce_max(i32, max32)
    f64 = rng.random(n)
    sum64 = np.zeros(1, np.float64)
    reduce_sum(f64, sum64)
    qd.sync()
    checks = {
        "sum_f32": abs(float(sum32[0]) - float(f32.astype(np.float64).sum()))
        < 1e-3 * n,
        "max_i32": int(max32[0]) == int(i32.max()),
        "sum_f64": abs(float(sum64[0]) - float(f64.sum())) < 1e-6 * n,
    }

    ptx = ""
    for path in glob.glob(os.path.join(dump_dir, "*")):
        with open(path, errors="replace") as fh:
            text = fh.read()
        if ".target" in text:
            ptx += text + "\n"
    targets = sorted(set(re.findall(r"\.target\s+sm_(\d+)", ptx)))
    result = {
        "checks": checks,
        "atom_sys": len(re.findall(r"\b(?:atom|red)\.sys\b", ptx)),
        "device_scoped": sorted(set(re.findall(r"\b(?:atom|red)\.gpu\.[\w.]+", ptx))),
        "atomics": len(re.findall(r"\b(?:atom|red)\.", ptx)),
        "targets": ["sm_" + t for t in targets],
        "versions": sorted(set(re.findall(r"\.version\s+([\d.]+)", ptx))),
    }
    print(json.dumps(result, indent=2))

    failures = [f"{name} is wrong" for name, ok in checks.items() if not ok]
    if not ptx:
        failures.append("no PTX was dumped")
    if result["atom_sys"]:
        failures.append(f"{result['atom_sys']} system-scoped atomics remain")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    if not targets or min(int(t) for t in targets) >= 70:
        print(
            f"PASS, but NOT PRE-VOLTA ({', '.join(result['targets'])}): 0003 is a no-op here"
        )
    else:
        print(
            f"PASS on {', '.join(result['targets'])}: CUDA up, 0 atom.sys, answers right"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
