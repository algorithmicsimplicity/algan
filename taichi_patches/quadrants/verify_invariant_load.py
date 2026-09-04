"""Prove `quadrants/0001-invariant-load-argument-loads.patch` did what it claims.

`PLAN.md` §5 step 2: confirm the argument loads have left the loop body
**before timing anything**. This is that check, on the CPU backend, where it
needs no GPU and the optimized LLVM IR is dumped by the same flag CUDA uses.

Run once per arm -- the arms differ by a `CompileConfig` field, so they must not
share a process (and, because the field is in the offline cache key, they would
not share a compiled artifact even if they did)::

    python verify_invariant_load.py on  > on.json
    python verify_invariant_load.py off > off.json
    python verify_invariant_load.py --compare on.json off.json

The kernel is eight ndarray arguments summed in a loop, which is the shape the
patch targets in miniature: every one of those eight base pointers is re-loaded
from the argument buffer on every iteration unless LICM can hoist it, and LICM
cannot until `!invariant.load` tells it the buffer is never written.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import tempfile

NARR = 8
N = 4096


def _dump_dir():
    d = tempfile.mkdtemp(prefix="qd_ir_")
    os.chdir(d)  # the CPU dump lands in the CWD
    return d


def _cfg_flag():
    """Read the flag back off the live config, without assuming the accessor path."""
    try:
        from quadrants.lang import impl

        return bool(impl.current_cfg().invariant_arg_loads)
    except Exception as exc:  # the pybind is the thing under test; do not mask its absence
        return f"unreadable: {exc!r}"


def _collect_ir(dump_dir):
    """Every optimized-IR dump this process produced, oldest first."""
    pats = [
        os.path.join(dump_dir, "*llvm_ir_optimized*.ll"),
        "/tmp/ir/*llvm_ir_optimized*.ll",
    ]
    out = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return out


def _loop_body(ir_text):
    """The block the backedge returns to -- i.e. the hot loop."""
    m = re.search(r"define internal void @function_body.*?\n\}", ir_text, re.S)
    if not m:
        return ""
    body = m.group(0)
    for block in re.split(r"\n(?=\w[\w.]*:)", body):
        name = block.split(":")[0].strip().split("\n")[-1]
        if re.search(r"br i1 .*?, label %" + re.escape(name), block):
            return block
    return ""


def measure(arm):
    dump_dir = _dump_dir()
    import numpy as np
    import quadrants as qd

    qd.init(
        arch=qd.cpu,
        offline_cache=False,               # force a real compile so IR is emitted
        advanced_optimization=False,       # the config Algan actually renders with
        fast_math=True,
        print_kernel_llvm_ir_optimized=True,
        invariant_arg_loads=(arm == "on"),
    )

    @qd.kernel
    def sum8(n: int, a0: qd.types.ndarray(), a1: qd.types.ndarray(), a2: qd.types.ndarray(),
             a3: qd.types.ndarray(), a4: qd.types.ndarray(), a5: qd.types.ndarray(),
             a6: qd.types.ndarray(), a7: qd.types.ndarray()):
        for i in range(n):
            a0[i] = a0[i] + a1[i] + a2[i] + a3[i] + a4[i] + a5[i] + a6[i] + a7[i]

    arrs = [np.zeros(N, dtype=np.float32) for _ in range(NARR)]
    sum8(N, *arrs)

    files = _collect_ir(dump_dir)
    text = "\n".join(open(f).read() for f in files)
    body = _loop_body(text)
    result = {
        "arm": arm,
        "config_flag_readable": _cfg_flag(),
        "ir_files": len(files),
        # The metadata itself.
        "invariant_load_total": len(re.findall(r"!invariant\.load", text)),
        "dereferenceable_total": len(re.findall(r"!dereferenceable", text)),
        # What it is supposed to buy: pointer re-loads left inside the loop.
        "loop_base_ptr_loads": len(re.findall(r"= load ptr, ptr |= load float\*, float\*\* ", body)),
        "loop_data_loads": len(re.findall(r"= load float, ", body)),
        "loop_lines": len(body.splitlines()),
    }
    os.chdir("/")  # leave the directory before removing it
    shutil.rmtree(dump_dir, ignore_errors=True)
    return result


def compare(on_path, off_path):
    on = json.load(open(on_path))
    off = json.load(open(off_path))
    print(f"{'':24s} {'off':>8s} {'on':>8s}")
    for k in ("invariant_load_total", "dereferenceable_total", "loop_base_ptr_loads",
              "loop_data_loads", "loop_lines"):
        print(f"{k:24s} {off[k]:8d} {on[k]:8d}")

    failures = []
    if off["invariant_load_total"] != 0:
        failures.append("the off arm emitted !invariant.load; the gate does not gate")
    if on["invariant_load_total"] <= 0:
        failures.append("the on arm emitted no !invariant.load; the patch did not take")
    if on["dereferenceable_total"] <= 0:
        failures.append("the on arm emitted no !dereferenceable on the arg-buffer pointer")
    if on["loop_data_loads"] == 0 or off["loop_data_loads"] == 0:
        failures.append("no loop body was found in the dumped IR; the parse is wrong, not the patch")
    elif on["loop_base_ptr_loads"] >= off["loop_base_ptr_loads"]:
        failures.append(
            f"base-pointer loads did not leave the loop "
            f"({off['loop_base_ptr_loads']} -> {on['loop_base_ptr_loads']}); "
            f"the metadata landed but LICM did not act on it")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nPASS: !invariant.load lands ({on['invariant_load_total']} sites) and the "
          f"argument base-pointer re-loads leave the loop "
          f"({off['loop_base_ptr_loads']} -> {on['loop_base_ptr_loads']}).")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    print(json.dumps(measure(sys.argv[1]), indent=2))
