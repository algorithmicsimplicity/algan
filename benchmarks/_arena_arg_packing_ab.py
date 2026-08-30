"""A/B: many ndarray kernel arguments vs one arena buffer + offsets.

Question under test (the Metal backend has a hard ceiling on buffer bindings,
so the render megakernels' 30-49 ndarray parameters would have to collapse into
a handful of arena buffers plus an offset table): what does that indirection
cost?

Three arms, all doing *identical* work over *identical* memory:

* ``split``  -- NF f32 ndarrays + NI i32 ndarrays, one kernel parameter each
               (what the renderer does today).
* ``arena``  -- one f32 arena ndarray + one i32 arena ndarray + two i32 offset
               ndarrays; every access is ``af[off_f[k] + idx]``.
* ``arenak`` -- same two arenas, but the offsets arrive as plain i32 *scalar*
               parameters instead of a lookup table. Scalars share one packed
               argument buffer on every backend, so this is the shape an actual
               port would take; the table arm bounds the cost if the offsets
               have to be data.

The confound this deliberately removes: in a real render every one of those
ndarrays is ALREADY a view into the single ``ManualMemory`` byte arena, so the
two arms have identical locality anyway. The benchmark slices all arm arrays
out of one contiguous buffer for that reason -- what is measured is argument
passing and address arithmetic, nothing else.

Run one arm per process (``--arm``); the harness re-execs itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Geometry of the synthetic kernel. NF/NI bracket the real megakernels:
# sheet_resolve_shade takes 49 ndarrays, wavefront_shade 41,
# wavefront_traverse_events 34.
# Overridable so the same body can be re-shaped: a single body shape cannot
# tell a systematic indirection cost apart from the backend's scheduling
# lottery, which moves these kernels by ~10% on its own.
NF = int(os.environ.get("AB_NF", 40))  # f32 array parameters
NI = int(os.environ.get("AB_NI", 8))  # i32 array parameters
SZ = 1 << 15  # elements per f32 array  (128 KiB)
ISZ = 1 << 13  # elements per i32 array  (32 KiB)
ROUNDS = int(os.environ.get("AB_ROUNDS", 3))  # unrolled passes per thread
NTHREADS = 1 << 20


ALIAS = True  # emit the arena store + read-back block (set by --variant)


def _body(read, iread, store, load_back):
    """Statement list shared by every arm.

    ``read(k, idx)`` / ``iread(k, idx)`` / ``store(idx, val)`` / ``load_back``
    are the per-arm accessor spellings. Everything else -- the arithmetic, the
    index chain, the number and order of memory operations -- is identical, so
    a timing difference between arms is an argument-passing difference.
    """
    lines = []
    add = lines.append
    add("    for i in range(n):")
    add("        idx = i & MASK")
    add("        iidx = i & IMASK")
    add("        acc = 0.0")
    add("        iacc = 0")
    for r in range(ROUNDS):
        for k in range(NF):
            add(f"        idx = (idx * 1103515245 + 12345 + {k}) & MASK")
            add(f"        acc = acc * 0.5 + {read(k, 'idx')}")
        for k in range(NI):
            add(f"        iidx = (iidx * 1103515 + {k * 7 + 1}) & IMASK")
            add(f"        iacc += {iread(k, 'iidx')}")
        # A store into the same arena, then reads of arrays the store could
        # alias. With separate ndarray parameters a backend may prove no
        # aliasing; with one arena it cannot, which is the interesting risk.
        if ALIAS:
            add(f"        {store('i', 'acc + ti.cast(iacc & 1023, ti.f32)')}")
            add(f"        back = {load_back('i')}")
        else:
            add("        back = acc")
        for k in range(0, NF, 8):
            add(f"        acc += back * 1e-6 + {read(k, 'idx')}")
        add(f"        acc += {float(r)}")
    add("        out[i] = acc + ti.cast(iacc & 65535, ti.f32)")
    return lines


def build_split():
    params = ["n: ti.i32"]
    params += [f"a{k}: ti.types.ndarray()" for k in range(NF)]
    params += [f"b{k}: ti.types.ndarray()" for k in range(NI)]
    params += ["s: ti.types.ndarray()", "out: ti.types.ndarray()"]
    body = _body(
        read=lambda k, i: f"a{k}[{i}]",
        iread=lambda k, i: f"b{k}[{i}]",
        store=lambda i, v: f"s[{i}] = {v}",
        load_back=lambda i: f"s[{i}]",
    )
    head = ["@ti.kernel", "def k_split(", "        " + ", ".join(params) + "):"]
    return "\n".join(head + body), "k_split"


def build_arena(scalar_offsets):
    params = ["n: ti.i32", "af: ti.types.ndarray()", "ai: ti.types.ndarray()"]
    if scalar_offsets:
        params += [f"of{k}: ti.i32" for k in range(NF)]
        params += [f"oi{k}: ti.i32" for k in range(NI)]
        params += ["osc: ti.i32"]

        def off_f(k):
            return f"of{k}"

        def off_i(k):
            return f"oi{k}"

        off_s = "osc"
    else:
        params += ["off_f: ti.types.ndarray()", "off_i: ti.types.ndarray()"]

        def off_f(k):
            return f"off_f[{k}]"

        def off_i(k):
            return f"off_i[{k}]"

        off_s = "off_f[NF]"
    params += ["out: ti.types.ndarray()"]
    body = _body(
        read=lambda k, i: f"af[{off_f(k)} + {i}]",
        iread=lambda k, i: f"ai[{off_i(k)} + {i}]",
        store=lambda i, v: f"af[{off_s} + {i}] = {v}",
        load_back=lambda i: f"af[{off_s} + {i}]",
    )
    name = "k_arenak" if scalar_offsets else "k_arena"
    head = ["@ti.kernel", f"def {name}(", "        " + ", ".join(params) + "):"]
    return "\n".join(head + body), name


def run_arm(arm, cache_dir, reps):
    import taichi as ti
    import torch

    from algan.rendering.taichi_runtime import _sync_devices, taichi_init_kwargs

    kwargs = taichi_init_kwargs()
    kwargs["offline_cache_file_path"] = cache_dir
    kwargs["kernel_profiler"] = True
    ti.init(**kwargs)

    # ``_taichi_arch()`` can return a *list* of candidate archs; the arch that
    # actually came up is the only thing that says where the tensors belong.
    # Handing a CUDA-arch kernel CPU tensors makes Taichi stage every argument
    # through VRAM on every launch, which swamps the measurement entirely.
    arch = ti.lang.impl.current_cfg().arch
    dev = "cuda" if arch == ti.cuda else "cpu"

    # ONE contiguous buffer, sliced into the per-array views the split arm
    # takes -- exactly what ManualMemory hands the renderer today. Both arms
    # therefore touch byte-identical memory in byte-identical order.
    gen = torch.Generator(device=dev).manual_seed(1234)
    fbuf = torch.rand(
        NF * SZ + NTHREADS, device=dev, generator=gen, dtype=torch.float32
    )
    ibuf = torch.randint(
        0, 1000, (NI * ISZ,), device=dev, generator=gen, dtype=torch.int32
    )
    out = torch.zeros(NTHREADS, device=dev, dtype=torch.float32)

    if arm == "split":
        src, name = build_split()
    elif arm == "arena":
        src, name = build_arena(False)
    else:
        src, name = build_arena(True)
    # Taichi's front end re-reads a kernel's source with ``inspect``, so the
    # generated kernel has to live in a real file on disk rather than an exec'd
    # namespace. Stable path per arm keeps the offline-cache key stable too.
    mod_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_arena_arg_gen")
    os.makedirs(mod_dir, exist_ok=True)
    header = "\n".join(
        [
            "import taichi as ti",
            f"MASK = {SZ - 1}",
            f"IMASK = {ISZ - 1}",
            f"NF = {NF}",
            "",
            "",
        ]
    )
    tag = ("alias" if ALIAS else "plain") + f"_{NF}_{NI}_{ROUNDS}"
    mod_path = os.path.join(mod_dir, f"gen_{tag}_{arm}.py")
    with open(mod_path, "w", encoding="utf-8") as fh:
        fh.write(header + src + "\n")
    sys.path.insert(0, mod_dir)
    import importlib

    mod = importlib.import_module(f"gen_{tag}_{arm}")
    importlib.reload(mod)
    kernel = getattr(mod, name)

    if arm == "split":
        arrs = [fbuf[k * SZ : (k + 1) * SZ] for k in range(NF)]
        iarrs = [ibuf[k * ISZ : (k + 1) * ISZ] for k in range(NI)]
        scratch = fbuf[NF * SZ :]
        args = [NTHREADS, *arrs, *iarrs, scratch, out]
    elif arm == "arena":
        off_f = torch.tensor(
            [k * SZ for k in range(NF)] + [NF * SZ], device=dev, dtype=torch.int32
        )
        off_i = torch.tensor(
            [k * ISZ for k in range(NI)], device=dev, dtype=torch.int32
        )
        args = [NTHREADS, fbuf, ibuf, off_f, off_i, out]
    else:
        args = [NTHREADS, fbuf, ibuf]
        args += [k * SZ for k in range(NF)]
        args += [k * ISZ for k in range(NI)]
        args += [NF * SZ, out]

    # --- compile / cache-load time -------------------------------------
    _sync_devices()
    t0 = time.perf_counter()
    kernel(*args)
    _sync_devices()
    first_launch = time.perf_counter() - t0

    checksum = float(out.double().sum().item())

    # --- steady-state runtime ------------------------------------------
    for _ in range(3):
        kernel(*args)
    _sync_devices()
    ti.profiler.clear_kernel_profiler_info()
    t0 = time.perf_counter()
    for _ in range(reps):
        kernel(*args)
    _sync_devices()
    wall = (time.perf_counter() - t0) / reps
    device_ms = float("nan")
    try:
        ti.profiler.get_kernel_profiler_total_time()
        device_ms = float(ti.profiler.query_kernel_profiler_info(name).avg)
    except Exception as exc:  # noqa: BLE001
        print("profiler query failed:", exc)

    # --- per-launch host cost (1 thread: body ~free, args dominate) -----
    # Measured both on the production launch path (algan's cached fast
    # launcher, installed at ``import algan``) and with it routed off, because
    # the per-ndarray-argument Python cost is exactly what that patch exists to
    # cut -- an argument-count change moves the same quantity.
    from algan.utils import taichi_fast_launch as tfl

    def _time_launches(reps_):
        tiny = [1, *args[1:]]
        for _ in range(50):
            kernel(*tiny)
        _sync_devices()
        t0_ = time.perf_counter()
        for _ in range(reps_):
            kernel(*tiny)
        _sync_devices()
        return (time.perf_counter() - t0_) / reps_ * 1e6

    tfl.STATS["fast"] = tfl.STATS["slow"] = 0
    launch_us = _time_launches(2000)
    fast_hits = tfl.STATS["fast"]
    slow_hits = tfl.STATS["slow"]

    prev_enabled = tfl.ENABLED
    tfl.ENABLED = False
    launch_slow_us = _time_launches(500)
    tfl.ENABLED = prev_enabled

    return {
        "arm": arm,
        "arch": str(arch),
        "n_kernel_args": len(args),
        "first_launch_s": first_launch,
        "checksum": checksum,
        "wall_per_launch_ms": wall * 1e3,
        "device_ms": device_ms,
        "launch_overhead_us": launch_us,
        "launch_overhead_noFL_us": launch_slow_us,
        "fast_launch_hits": fast_hits,
        "fast_launch_misses": slow_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["split", "arena", "arenak"])
    ap.add_argument("--cache", default="")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", choices=["alias", "plain"], default="alias")
    args = ap.parse_args()

    global ALIAS
    ALIAS = args.variant == "alias"
    if args.arm:
        rec = run_arm(args.arm, args.cache, args.reps)
        rec["variant"] = args.variant
        print("RESULT " + json.dumps(rec))
        return

    import tempfile

    root = args.out or tempfile.mkdtemp(prefix="arena_ab_")
    os.makedirs(root, exist_ok=True)
    rows = []
    print(f"[shape NF={NF} NI={NI} ROUNDS={ROUNDS} variant={args.variant}]")
    for phase in ("cold", "warm"):
        for arm in ("split", "arena", "arenak"):
            cache = os.path.join(root, f"cache_{args.variant}_{NF}_{ROUNDS}_{arm}")
            os.makedirs(cache, exist_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--arm",
                    arm,
                    "--cache",
                    cache,
                    "--reps",
                    str(args.reps),
                    "--variant",
                    args.variant,
                ],
                capture_output=True,
                text=True,
            )
            line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print(proc.stdout[-4000:])
                print(proc.stderr[-4000:])
                raise SystemExit(f"{arm}/{phase} failed")
            rec = json.loads(line[0][7:])
            rec["phase"] = phase
            rows.append(rec)
            print(
                f"{phase:5s} {arm:7s} args={rec['n_kernel_args']:3d} "
                f"first_launch={rec['first_launch_s']:8.3f}s "
                f"device={rec['device_ms']:8.3f}ms "
                f"wall={rec['wall_per_launch_ms']:8.3f}ms "
                f"launch={rec['launch_overhead_us']:7.1f}us "
                f"(noFL {rec['launch_overhead_noFL_us']:7.1f}us, "
                f"fl {rec['fast_launch_hits']}/{rec['fast_launch_misses']}) "
                f"chk={rec['checksum']:.6e}"
            )
    with open(os.path.join(root, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nresults: " + os.path.join(root, "results.json"))


if __name__ == "__main__":
    main()
