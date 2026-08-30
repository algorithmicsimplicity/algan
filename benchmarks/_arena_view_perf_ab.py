"""A/B: ndarray parameters vs arena + in-kernel ``View`` objects.

Follow-up to ``_arena_arg_packing_ab.py``. That one measured raw flat-offset
indexing; this one measures the design that actually preserves the existing
``ti.func`` cascade -- a Python ``View`` object built in kernel scope, bound to
a local name with ``ti.static``, and passed to funcs whose array parameters are
already ``ti.template()``. Feasibility (reads, stores, ``.shape``, a real algan
func, runtime shapes) is established by ``_arena_view_feasibility.py``.

Arms, identical work over identical memory (all arrays sliced from one buffer):

* ``split`` -- NF f32 + NI i32 ndarray parameters, indexed ``a[i, c]``. Taichi
  emits a runtime-shape stride multiply, because an ndarray's shape is runtime
  data.
* ``view``  -- 1 f32 arena + 1 i32 arena + offset/shape tables; each array is a
  ``View`` bound at the top of the kernel and indexed ``a[i, c]`` -- the SAME
  spelling in the body, which is the whole point.

Run one arm per process (``--arm``); the harness re-execs itself.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NF = int(os.environ.get("AB_NF", 40))
NI = int(os.environ.get("AB_NI", 8))
COLS = 4
ROWS = 1 << 13
IROWS = 1 << 11
ROUNDS = int(os.environ.get("AB_ROUNDS", 3))
NTHREADS = 1 << 20

VIEW_SRC = '''
from taichi.lang import impl as _ti_impl


class View(tuple):
    """A window into a flat arena, indexed exactly like the ndarray it replaces.

    Subclasses ``tuple`` so ``ti.static`` accepts it and passes it through --
    that is what lets the view be bound to a local NAME in kernel scope
    (Taichi's assignment builder otherwise tries to make a Taichi local of type
    ``View`` and fails). Shape entries may be Taichi Exprs, so nothing about
    the geometry is baked into the compiled kernel.
    """

    def __new__(cls, buf, base, shape):
        return super().__new__(cls, (buf, base, tuple(shape)))

    @property
    def buf(self):
        return tuple.__getitem__(self, 0)

    @property
    def base(self):
        return tuple.__getitem__(self, 1)

    @property
    def shape(self):
        return tuple.__getitem__(self, 2)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = self.shape
        flat = idx[0]
        for d in range(1, len(idx)):
            flat = flat * shape[d] + idx[d]
        # Python scope here, so go through Taichi's own subscript builder. The
        # IndexExpression it returns is an lvalue, so stores work too.
        return _ti_impl.subscript(None, self.buf, self.base + flat)
'''


def _body(fname, iname):
    """Shared statement list. Identical in both arms -- only the names differ."""
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
            add(f"        acc = acc * 0.5 + {fname(k)}[idx, {k % COLS}]")
        for k in range(NI):
            add(f"        iidx = (iidx * 1103515 + {k * 7 + 1}) & IMASK")
            add(f"        iacc += {iname(k)}[iidx, {k % COLS}]")
        add(f"        {fname(0)}[idx, 3] = acc + ti.cast(iacc & 1023, ti.f32)")
        add(f"        back = {fname(0)}[idx, 3]")
        for k in range(0, NF, 8):
            add(f"        acc += back * 1e-6 + {fname(k)}[idx, 2]")
        add(f"        acc += {float(r)}")
    add("        out[i] = acc + ti.cast(iacc & 65535, ti.f32)")
    return lines


def build_split():
    params = ["n: ti.i32"]
    params += [f"a{k}: ti.types.ndarray()" for k in range(NF)]
    params += [f"b{k}: ti.types.ndarray()" for k in range(NI)]
    params += ["out: ti.types.ndarray()"]
    body = _body(lambda k: f"a{k}", lambda k: f"b{k}")
    head = ["@ti.kernel", "def k_split(", "        " + ", ".join(params) + "):"]
    return "\n".join(head + body), "k_split"


def build_view():
    params = [
        "n: ti.i32",
        "af: ti.types.ndarray()",
        "ai: ti.types.ndarray()",
        "off_f: ti.types.ndarray()",
        "off_i: ti.types.ndarray()",
        "shp: ti.types.ndarray()",
        "out: ti.types.ndarray()",
    ]
    head = ["@ti.kernel", "def k_view(", "        " + ", ".join(params) + "):"]
    # One binding line per array, at the top of the kernel -- the entire diff a
    # real port would make to a megakernel body.
    binds = []
    for k in range(NF):
        binds.append(f"    a{k} = ti.static(View(af, off_f[{k}], (shp[0], shp[1])))")
    for k in range(NI):
        binds.append(f"    b{k} = ti.static(View(ai, off_i[{k}], (shp[2], shp[3])))")
    body = _body(lambda k: f"a{k}", lambda k: f"b{k}")
    return "\n".join(head + binds + body), "k_view"


def run_arm(arm, cache_dir, reps):
    import taichi as ti
    import torch

    from algan.rendering.taichi_runtime import _sync_devices, taichi_init_kwargs

    kwargs = taichi_init_kwargs()
    kwargs["offline_cache_file_path"] = cache_dir
    kwargs["kernel_profiler"] = True
    ti.init(**kwargs)

    arch = ti.lang.impl.current_cfg().arch
    dev = "cuda" if arch == ti.cuda else "cpu"

    gen = torch.Generator(device=dev).manual_seed(1234)
    fbuf = torch.rand(NF * ROWS * COLS, device=dev, generator=gen, dtype=torch.float32)
    ibuf = torch.randint(
        0, 1000, (NI * IROWS * COLS,), device=dev, generator=gen, dtype=torch.int32
    )
    out = torch.zeros(NTHREADS, device=dev, dtype=torch.float32)

    src, name = build_split() if arm == "split" else build_view()
    mod_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_arena_arg_gen")
    os.makedirs(mod_dir, exist_ok=True)
    header = "\n".join(
        [
            "import taichi as ti",
            f"MASK = {ROWS - 1}",
            f"IMASK = {IROWS - 1}",
            VIEW_SRC if arm == "view" else "",
            "",
        ]
    )
    tag = f"v_{NF}_{NI}_{ROUNDS}_{arm}"
    with open(os.path.join(mod_dir, f"gen_{tag}.py"), "w", encoding="utf-8") as fh:
        fh.write(header + src + "\n")
    sys.path.insert(0, mod_dir)
    import importlib

    mod = importlib.import_module(f"gen_{tag}")
    importlib.reload(mod)
    kernel = getattr(mod, name)

    if arm == "split":
        arrs = [
            fbuf[k * ROWS * COLS : (k + 1) * ROWS * COLS].view(ROWS, COLS)
            for k in range(NF)
        ]
        iarrs = [
            ibuf[k * IROWS * COLS : (k + 1) * IROWS * COLS].view(IROWS, COLS)
            for k in range(NI)
        ]
        args = [NTHREADS, *arrs, *iarrs, out]
    else:
        off_f = torch.tensor(
            [k * ROWS * COLS for k in range(NF)], device=dev, dtype=torch.int32
        )
        off_i = torch.tensor(
            [k * IROWS * COLS for k in range(NI)], device=dev, dtype=torch.int32
        )
        shp = torch.tensor([ROWS, COLS, IROWS, COLS], device=dev, dtype=torch.int32)
        args = [NTHREADS, fbuf, ibuf, off_f, off_i, shp, out]

    _sync_devices()
    t0 = time.perf_counter()
    kernel(*args)
    _sync_devices()
    first_launch = time.perf_counter() - t0
    checksum = float(out.double().sum().item())

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

    from algan.utils import taichi_fast_launch as tfl

    def _time_launches(reps_):
        tiny = [1, *args[1:]]
        for _ in range(50):
            kernel(*tiny)
        _sync_devices()
        t1 = time.perf_counter()
        for _ in range(reps_):
            kernel(*tiny)
        _sync_devices()
        return (time.perf_counter() - t1) / reps_ * 1e6

    tfl.STATS["fast"] = tfl.STATS["slow"] = 0
    launch_us = _time_launches(2000)
    fast_hits, slow_hits = tfl.STATS["fast"], tfl.STATS["slow"]

    return {
        "arm": arm,
        "arch": str(arch),
        "n_kernel_args": len(args),
        "first_launch_s": first_launch,
        "checksum": checksum,
        "wall_per_launch_ms": wall * 1e3,
        "device_ms": device_ms,
        "launch_overhead_us": launch_us,
        "fast_launch_hits": fast_hits,
        "fast_launch_misses": slow_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["split", "view"])
    ap.add_argument("--cache", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm:
        print("RESULT " + json.dumps(run_arm(args.arm, args.cache, args.reps)))
        return

    import tempfile

    root = args.out or tempfile.mkdtemp(prefix="arena_view_ab_")
    os.makedirs(root, exist_ok=True)
    print(f"[shape NF={NF} NI={NI} COLS={COLS} ROUNDS={ROUNDS}]")
    for phase in ("cold", "warm"):
        for arm in ("split", "view"):
            cache = os.path.join(root, f"cache_{NF}_{ROUNDS}_{arm}")
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
            print(
                f"{phase:5s} {arm:6s} args={rec['n_kernel_args']:3d} "
                f"first_launch={rec['first_launch_s']:7.3f}s "
                f"device={rec['device_ms']:8.3f}ms "
                f"launch={rec['launch_overhead_us']:7.1f}us "
                f"(fl {rec['fast_launch_hits']}/{rec['fast_launch_misses']}) "
                f"chk={rec['checksum']:.9e}"
            )


if __name__ == "__main__":
    main()
