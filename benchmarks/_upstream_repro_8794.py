"""Upstream repro: taichi-dev/taichi#8794 -- segfault on the 512th SNode tree.

https://github.com/taichi-dev/taichi/issues/8794 ("Taichi apparently not cleaning
memory on Mac (arm64)", reporter ``lyounes``, open, Taichi 1.7.4, macOS arm64,
banner ``arch=arm64`` -- i.e. the **LLVM CPU backend**, not Metal).

**Symptom.** A loop prints ``rep=0`` .. ``rep=511`` and then dies with signal 11 inside
``taichi::ThreadPool::run`` / ``std::mutex::lock``, reached from
``taichi::lang::Program::launch_kernel``. The reporter reads this as "memory not being
cleaned"; it is consistent to the iteration on every run.

**The issue carries no reproduction script** (the reporter names a local
``taichi_bug2.py`` that was never attached), so this file is a *reconstruction* from the
iteration number and the stack trace, not the reporter's code. What makes it more than a
guess:

* ``kMaxNumSnodeTreesLlvm = 512`` (``taichi/inc/constants.h``) sizes two fixed C arrays
  in the LLVM runtime struct, ``Ptr roots[512]`` and ``std::size_t root_mem_sizes[512]``
  (``taichi/runtime/llvm/runtime_module/llvm_runtime.h``), and nothing anywhere checks
  ``tree_id`` against it -- the constant has no other use in the tree.
* The field declared immediately after those two arrays is ``Ptr thread_pool``. So
  materializing SNode tree 512 writes ``root_mem_sizes[512]``, which *is* the
  ``thread_pool`` pointer, and the next parallel launch jumps through it. That is the
  reported crash site, reached from the reported caller, at the reported iteration.

So: any program that materializes more than 512 SNode trees -- i.e. creates fields in a
loop without destroying them -- corrupts the runtime and segfaults on its next launch.
It is neither Mac- nor arm64- nor Metal-specific.

**Arch.** Any LLVM backend (cpu / cuda / amdgpu); ``REPRO_ARCH`` picks one, default
``cpu``. It says nothing about the SPIR-V backends, which do not use this runtime struct.

**Relevance to Algan.** Algan uses no ``ti.field`` at all (every kernel argument is a
``ti.types.ndarray()`` over a torch tensor), so it materializes one SNode tree per
process and is structurally immune. This matters for reading the Taichi-fork plan:
``taichi_patches/PLAN.md`` row 30 offers Quadrants' Metal "pending-launch drain valve"
as a hypothesis for #8794, and that cannot be the fix -- the crash is in the LLVM CPU
runtime, on a code path the Metal/gfx runtime does not share.

Three modes are run, each in its own child process because the failure is a signal:

``fields``
    a new ``FieldsBuilder`` finalized every iteration, never destroyed -- the reported
    shape.
``fields_destroy``
    the same, with ``tree.destroy()`` each iteration -- tests whether tree ids are
    recycled, i.e. whether "destroy your trees" is a bounded workaround.
``launch``
    one tree, the same kernel launched every iteration -- tests the rival hypothesis
    that repeated *launches* are what break, independent of tree count.

Usage::

    REPRO_BACKEND=taichi    REPRO_ARCH=cpu   python benchmarks/_upstream_repro_8794.py
    REPRO_BACKEND=quadrants REPRO_ARCH=cuda  python benchmarks/_upstream_repro_8794.py

``REPRO_ITERS`` (default 600) sets the loop length; ``REPRO_MODE`` runs a single mode
instead of all three. Prints one verdict line, ``REPRO-8794: REPRODUCES`` or
``REPRO-8794: CLEAN``, plus the iteration each mode reached. Exits 0 either way; a
non-zero exit means the script itself failed.
"""

import importlib
import os
import re
import subprocess
import sys

BACKEND = os.environ.get("REPRO_BACKEND", "taichi")
ARCH_NAME = os.environ.get("REPRO_ARCH", "cpu")
ITERS = int(os.environ.get("REPRO_ITERS", "600"))
MODES = ("fields", "fields_destroy", "launch")


def child(mode: str) -> int:
    """The loop under test. Runs in its own process; may die by signal."""
    ti = importlib.import_module(BACKEND)
    ti.init(arch=getattr(ti, ARCH_NAME))

    x = ti.field(ti.f32, shape=(16,))  # tree 0

    @ti.kernel
    def bump():
        for i in x:
            x[i] += 1.0

    kept = []
    for rep in range(ITERS):
        if mode in ("fields", "fields_destroy"):
            builder = ti.FieldsBuilder()
            y = ti.field(ti.f32)
            builder.dense(ti.i, 4).place(y)
            tree = builder.finalize()
            if mode == "fields_destroy":
                tree.destroy()
            else:
                kept.append(tree)  # keep it alive: the reported shape
        bump()
        print(f"rep={rep}", flush=True)
    print(f"COMPLETED {ITERS} iterations, x[0]={x.to_numpy()[0]}", flush=True)
    return 0


def run_mode(mode: str) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", mode],
        capture_output=True,
        text=True,
        env={**os.environ, "REPRO_BACKEND": BACKEND, "REPRO_ARCH": ARCH_NAME,
             "REPRO_ITERS": str(ITERS)},
    )
    reps = [int(m) for m in re.findall(r"^rep=(\d+)$", proc.stdout, re.M)]
    last = reps[-1] if reps else None
    tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()][-3:]
    return {
        "mode": mode,
        "returncode": proc.returncode,
        "last_rep": last,
        "completed": "COMPLETED" in proc.stdout,
        "signal": -proc.returncode if proc.returncode < 0 else None,
        "tail": tail,
    }


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2])

    print(f"backend={BACKEND} arch={ARCH_NAME} iters={ITERS} python={sys.version.split()[0]}")
    modes = (os.environ["REPRO_MODE"],) if "REPRO_MODE" in os.environ else MODES

    results = [run_mode(m) for m in modes]
    crashed = []
    for r in results:
        if r["completed"]:
            outcome = f"completed all {ITERS}"
        elif r["signal"]:
            outcome = f"KILLED by signal {r['signal']} after rep={r['last_rep']}"
            crashed.append(r)
        else:
            outcome = f"exited {r['returncode']} after rep={r['last_rep']}"
            if r["returncode"]:
                crashed.append(r)
        print(f"  {r['mode']:16s} -> {outcome}")
        if not r["completed"]:
            for line in r["tail"]:
                print(f"      | {line}")

    if crashed:
        first = crashed[0]
        n_trees = (first["last_rep"] or 0) + 2  # tree 0, plus one per completed rep
        clean_modes = [r["mode"] for r in results if r["completed"]]
        print(
            f"REPRO-8794: REPRODUCES ({first['mode']} died with signal {first['signal']} "
            f"after rep={first['last_rep']}, i.e. on SNode tree {n_trees} of a "
            f"512-entry table; modes that survived: {clean_modes or 'none'})"
        )
    else:
        print(f"REPRO-8794: CLEAN (every mode completed {ITERS} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
