"""Prove the three CUDA codegen patches did what they claim, on a CUDA box.

`0005-cuda-max-reg.patch`, `0006-cuda-readonly-ndarray-ldg.patch` and
`0007-cuda-fast-expf.patch` each change what the NVPTX backend is handed, so
each is visible in the PTX or the pre-O3 LLVM IR of one small kernel:

    0005  qd.loop_config(max_reg=N)   ->  `.maxnreg N` in that kernel's PTX
    0006  readonly_ndarray_ldg        ->  `ld.global.nc` for the loads of the
                                          ndarrays the kernel never writes
    0007  fast_math                   ->  `@__nv_fast_expf` in place of
                                          `@__nv_expf` in the *unoptimized* IR

The exp check has to read the unoptimized IR: O3 inlines both libdevice
routines, and both end in an `ex2.approx`, so neither the optimized IR nor the
PTX can tell them apart by name.

Run once per arm -- the arms differ by `CompileConfig` fields, `qd.init` runs
once per process, and the fields are in the offline cache key so the arms
would not share a compiled artifact even if they could share a process::

    python verify_cuda_patches.py on  on.json
    python verify_cuda_patches.py off off.json
    python verify_cuda_patches.py --compare on.json off.json

The `on` arm is Algan's configuration: `fast_math=True`,
`readonly_ndarray_ldg=True`, and the loop carries `max_reg=64`. The `off` arm
turns all three off. Both leave `invariant_arg_loads` (0004) at its default,
so the comparison isolates 0005-0007.

**No CUDA device is a skip, not a failure.** An arm run prints why and exits 0
after writing `{"skipped": true}` to its file, and `--compare` exits 0 with
`SKIP` when either file says so. That is what lets the build workflow call this
on the GPU-less runner without lying about a pass.

The arm result is written to the named file rather than to stdout, because
`qd.init()` prints a banner there that a shell redirect would fold into the
JSON.

What this cannot see: the module-wide `qd.init(gpu_max_reg=N)` half of 0005 is
a `CU_JIT_MAX_REGISTERS` option at `cuModuleLoadDataEx`, not a PTX directive,
so it leaves no trace in any dump. Both arms pass `gpu_max_reg=0` so it cannot
confound the `.maxnreg` count; measure it with `cuobjdump
--dump-resource-usage` on the driver's cubin, or the profiler's register count.
"""

# ruff: noqa: I002 -- I002 would insert `from __future__ import annotations`,
# which breaks this file. The comment above the `quadrants` import below says
# why. It is the same hazard `pyproject.toml` disables I002 for in `*_taichi.py`;
# this file is not named that way because it is a standalone gate script rather
# than an Algan module, so the suffix-keyed config does not reach it.
import glob
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

import numpy as np

# Both imports are module-level, and NEITHER is incidental.
#
# There is no `from __future__ import annotations` in this file, and there must
# not be: a kernel's parameter annotations are evaluated at *runtime*, so the
# future import turns `qd.types.ndarray()` into the string "qd.types.ndarray()"
# and decoration dies with "Invalid type annotation ... name 'qd' is not
# defined". This is the same hazard `CLAUDE.md` records for `*_taichi.py` files,
# where ruff's I002 is switched off for exactly this reason.
#
# And `qd` is bound here rather than inside `measure()` because annotations
# resolve against the enclosing function's *globals*, not its locals -- a
# function-local `import quadrants as qd` fails the same way even without the
# future import.
import quadrants as qd

N = 4096
MAX_REG = 64  # the per-loop cap the `on` arm asks for; any value ptxas accepts will do, it only has to show up
KERNEL_NAME = "verify_cuda_probe"  # every dump is filtered on this, so the runtime module's PTX is never counted


def _cuda_available():
    """Whether `qd.init(arch=qd.cuda)` can end up on a CUDA device, without initializing anything.

    `with_cuda()` is the probe `qd.init` itself uses to decide whether CUDA is even a candidate (it looks for the
    driver library, so it is False on every GPU-less runner however the binary was built). A True here can still fall
    back to CPU at init time if the library is present but no device is, which `measure()` re-checks after `qd.init`.
    """
    try:
        from quadrants._lib import core as qd_core

        return bool(qd_core.with_cuda())
    except Exception as exc:  # noqa: BLE001 -- the probe's absence is itself the answer
        print(f"[verify_cuda_patches] cannot probe for CUDA ({exc!r}); treating as absent")
        return False


def _dump_dir():
    d = tempfile.mkdtemp(prefix="qd_cuda_patches_")
    os.chdir(d)  # every FileSequenceWriter dump lands in the CWD
    return d


def _cfg():
    """The three flags read back off the live config, so the JSON says what the arm actually ran with."""
    try:
        from quadrants.lang import impl

        cfg = impl.current_cfg()
        return {
            "arch": str(cfg.arch),
            "fast_math": bool(cfg.fast_math),
            "readonly_ndarray_ldg": bool(cfg.readonly_ndarray_ldg),
            "gpu_max_reg": int(cfg.gpu_max_reg),
        }
    except Exception as exc:  # noqa: BLE001 -- the nanobind is the thing under test; do not mask its absence
        return {"unreadable": repr(exc)}


def _texts(pattern, dump_dir):
    """The dumps matching `pattern` that belong to the probe kernel, as text, oldest first."""
    out = []
    for f in sorted(glob.glob(os.path.join(dump_dir, pattern))):
        text = pathlib.Path(f).read_text()
        if KERNEL_NAME in text:
            out.append(text)
    return out


def build_kernel(max_reg):
    """The probe: two ndarrays it only reads, one it reads and writes, one f32 exp.

    `qd.loop_config(max_reg=None)` is a no-op, which is how the `off` arm gets the same kernel with no cap.
    """

    @qd.kernel
    def verify_cuda_probe(n: int, a: qd.types.ndarray(), b: qd.types.ndarray(), out: qd.types.ndarray()):
        qd.loop_config(max_reg=max_reg)
        for i in range(n):
            # `a` and `b` are never written, so 0006 may fetch them through the non-coherent cache; `out` is read
            # *and* written, so its load must stay a plain `ld.global` -- the count of those is the control.
            out[i] = qd.exp(-a[i] * b[i]) + out[i]

    return verify_cuda_probe


def measure(arm):
    if not _cuda_available():
        print("[verify_cuda_patches] SKIP: no CUDA driver on this machine; nothing to measure")
        return {"arm": arm, "skipped": True, "reason": "no CUDA driver"}

    dump_dir = _dump_dir()  # chdir first: the dumps land in the CWD
    on = arm == "on"
    qd.init(
        arch=qd.cuda,
        offline_cache=False,  # force a real compile so the dumps are emitted
        advanced_optimization=False,  # the config Algan actually renders with
        fast_math=on,  # 0007
        readonly_ndarray_ldg=on,  # 0006
        gpu_max_reg=0,  # keep the module-wide cap out of the way of the per-loop one
        print_kernel_llvm_ir=True,  # unoptimized IR: where the libdevice call is still a call by name
        print_kernel_asm=True,  # the PTX
    )
    cfg = _cfg()
    if cfg.get("arch") != str(qd.cuda):
        os.chdir("/")
        shutil.rmtree(dump_dir, ignore_errors=True)
        print(f"[verify_cuda_patches] SKIP: qd.init(arch=qd.cuda) landed on {cfg.get('arch')}; no CUDA device")
        return {"arm": arm, "skipped": True, "reason": f"init fell back to {cfg.get('arch')}"}

    kernel = build_kernel(MAX_REG if on else None)
    a = np.full(N, 0.5, dtype=np.float32)
    b = np.full(N, 2.0, dtype=np.float32)
    out = np.zeros(N, dtype=np.float32)
    kernel(N, a, b, out)
    qd.sync()

    ptx = "\n".join(_texts("quadrants_kernel_nvptx_*.ptx", dump_dir))
    ir = "\n".join(_texts("quadrants_kernel_cuda_llvm_ir_[0-9]*.ll", dump_dir))
    result = {
        "arm": arm,
        "skipped": False,
        "config": cfg,
        "ptx_found": bool(ptx),
        "ir_found": bool(ir),
        # 0005: the per-kernel directive.
        "maxnreg_directives": len(re.findall(r"^\s*\.maxnreg\s+\d+", ptx, re.M)),
        "maxnreg_values": sorted(set(re.findall(r"^\s*\.maxnreg\s+(\d+)", ptx, re.M))),
        # 0006: the non-coherent loads, and the plain global loads left beside them (the `out` read is the control).
        "ld_global_nc": len(re.findall(r"\bld\.global\.nc\b", ptx)),
        "ld_global_plain": len(re.findall(r"\bld\.global\.(?!nc\b)[a-z0-9.]*\b", ptx)),
        "st_global": len(re.findall(r"\bst\.global\b", ptx)),
        # 0007: which libdevice exp the codegen asked for, before O3 inlines it.
        "fast_expf_calls": len(re.findall(r"@__nv_fast_expf\b", ir)),
        "expf_calls": len(re.findall(r"@__nv_expf\b", ir)),
    }
    os.chdir("/")  # leave the directory before removing it
    shutil.rmtree(dump_dir, ignore_errors=True)
    return result


def compare(on_path, off_path):
    on = json.loads(pathlib.Path(on_path).read_text())
    off = json.loads(pathlib.Path(off_path).read_text())
    if on.get("skipped") or off.get("skipped"):
        print(f"SKIP: {on.get('reason') or off.get('reason')} -- the CUDA patches were not exercised")
        return 0

    print(f"{'':24s} {'off':>8s} {'on':>8s}")
    for k in ("maxnreg_directives", "ld_global_nc", "ld_global_plain", "st_global", "fast_expf_calls", "expf_calls"):
        print(f"{k:24s} {off[k]:8d} {on[k]:8d}")

    failures = []
    if not (on["ptx_found"] and off["ptx_found"] and on["ir_found"] and off["ir_found"]):
        failures.append("a dump is missing for at least one arm; the glob or the kernel-name filter is wrong, not the patch")
    else:
        # 0005
        if on["maxnreg_directives"] < 1:
            failures.append("the on arm's PTX has no .maxnreg; loop_config(max_reg=) did not reach the kernel")
        elif str(MAX_REG) not in on["maxnreg_values"]:
            failures.append(f"the on arm's .maxnreg is {on['maxnreg_values']}, not {MAX_REG}")
        if off["maxnreg_directives"] != 0:
            failures.append("the off arm's PTX carries a .maxnreg; a cap leaked in from somewhere")
        # 0006
        if on["ld_global_nc"] < 1:
            failures.append("the on arm has no ld.global.nc; the read-only ndarray loads did not take the ldg path")
        if on["ld_global_plain"] < 1:
            failures.append("the on arm has no plain ld.global left; the read-and-written array went .nc -- unsound")
        if off["ld_global_nc"] != 0:
            failures.append("the off arm emitted ld.global.nc; readonly_ndarray_ldg=False does not gate")
        # 0007
        if on["fast_expf_calls"] < 1 or on["expf_calls"] != 0:
            failures.append("the on arm does not call __nv_fast_expf (and only it); fast_math did not pick the fast exp")
        if off["expf_calls"] < 1 or off["fast_expf_calls"] != 0:
            failures.append("the off arm does not call __nv_expf (and only it); fast_math=False did not restore exact exp")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"\nPASS: .maxnreg {on['maxnreg_values']} lands (0005), "
        f"{on['ld_global_nc']} ld.global.nc with {on['ld_global_plain']} plain load kept for the written array (0006), "
        f"and exp goes through __nv_fast_expf only under fast_math (0007)."
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    # The result goes to a FILE named on the command line, never to stdout: `qd.init()` prints its banner there
    # unconditionally. Absolute path first, because measure() leaves the CWD.
    arm, out_path = sys.argv[1], os.path.abspath(sys.argv[2])
    if arm not in ("on", "off"):
        sys.exit(f"arm must be 'on' or 'off', not {arm!r}")
    result = measure(arm)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
