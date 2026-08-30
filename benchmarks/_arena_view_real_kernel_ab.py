"""A/B the REAL ``sheet_resolve_shade`` against an arena/``View`` variant.

**Superseded.** The measurement this made is what put the arena convention into
the renderer: ``sheet_resolve_shade`` now IS the ``keep:raystate`` arm
(`algan/rendering/raytracing/arena_args_taichi.py`), so the "ref" arm this
harness generates its variants from no longer exists in that form and the script
does not run against current master. Kept for the record -- its numbers, and the
mechanism behind them, are written up in `DESIGN_taichi_argument_loads.md`. To
re-run it, check out a commit from before the conversion.

Replays one captured launch (``_arena_view_real_capture.py``) through:

* **ref**   -- the shipped kernel, 47 ndarray parameters.
* **arena** -- a variant generated from the SAME source file. The only edits
  are (a) the signature, where every ``ti.types.ndarray()`` parameter is
  replaced by one arena ndarray per dtype plus an offset table and a shape
  table, and (b) a binding prologue that rebinds each original parameter name
  to a ``View`` over its arena. The 1000-line body is copied verbatim, and not
  one of the ~200 ``ti.func``s it calls is touched -- their array parameters
  are already ``ti.template()``, which inlines whatever object they are given.

Reports compile time (cold and warm offline cache), device kernel time, and
host launch cost, and checks every array the kernel writes for bit-identity.

Usage:
  ... _arena_view_real_kernel_ab.py --capture <cap.pt> [--arm ref|arena]
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KERNEL_NAME = "sheet_resolve_shade"
SRC_REL = "algan/rendering/raytracing/sheet_resolve_taichi.py"

VIEW_SRC = '''

from taichi.lang import impl as _ti_impl


class View(tuple):
    """A window into a flat arena, indexed exactly like the ndarray it replaces.

    Subclasses ``tuple`` so ``ti.static`` accepts it and passes it through --
    that is what lets a view be bound to a local NAME in kernel scope (Taichi's
    assignment builder otherwise tries to create a Taichi local of type
    ``View`` and fails). Shape entries are Taichi Exprs read from a runtime
    table, so no geometry size is baked into the compiled kernel.

    ``__getitem__`` returns the arena's own IndexExpression, which is an
    lvalue, so stores through a view work like stores through the array.
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
        # Python scope here, so build the subscript through Taichi's own
        # builder rather than AnyArray.__getitem__ (which does not exist).
        return _ti_impl.subscript(None, self.buf, self.base + flat)

'''

DT_TAG = {
    "torch.float32": "f32",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.uint8": "u8",
    "torch.float16": "f16",
}


def generate_variant(repo_root, layout, out_path, consts=None, suffix="_arena"):
    """Emit the arena variant module from the shipped kernel source.

    ``layout`` maps each ndarray parameter name to (dtype_tag, ndim, off_slot,
    shape_slot) -- everything the binding prologue needs. The body is copied
    byte-for-byte.
    """
    src_path = os.path.join(repo_root, SRC_REL)
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.split("\n")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == KERNEL_NAME
    )

    header_end = min(d.lineno for d in fn.decorator_list) - 1
    header = "\n".join(lines[:header_end])

    params = [
        (a.arg, ast.unparse(a.annotation) if a.annotation else None)
        for a in fn.args.args
    ]
    nd_all = [n for n, ann in params if ann == "ti.types.ndarray()"]
    nd_names = [n for n in nd_all if n in layout]
    kept = [n for n in nd_all if n not in layout]

    keep = [
        f"{n}: {ann}" if ann else n
        for n, ann in params
        if ann != "ti.types.ndarray()" or n in kept
    ]
    tags = sorted({layout[n][0] for n in nd_names})
    arenas = [f"arena_{t}: ti.types.ndarray()" for t in tags]
    new_params = (
        keep + arenas + ["aoff: ti.types.ndarray()", "ashp: ti.types.ndarray()"]
    )

    binds = []
    for name in nd_names:
        tag, ndim, off_slot, shp_slot = layout[name]
        if consts is None:
            base = f"aoff[{off_slot}]"
            dims = ", ".join(f"ashp[{shp_slot + d}]" for d in range(ndim))
        else:
            # Diagnostic arm only: baking the layout in as literals would make
            # the kernel recompile for every scene, which is a non-starter at
            # minutes per cold compile. It exists to say how much of the
            # arena arm's cost is the per-access aoff/ashp global loads.
            base_i, shape_t = consts[name]
            base = str(base_i)
            dims = ", ".join(str(int(x)) for x in shape_t)
        if ndim == 1:
            dims += ","
        binds.append(f"    {name} = ti.static(View(arena_{tag}, {base}, ({dims})))")

    body_start = fn.body[0].lineno - 1  # docstring line
    body = lines[body_start : fn.end_lineno]

    out = [
        header,
        VIEW_SRC,
        "",
        "@ti.kernel",
        f"def {KERNEL_NAME}{suffix}(",
        "        " + ",\n        ".join(new_params) + "):",
    ]
    # Docstring first (it is fn.body[0]), then the bindings, then the rest of
    # the body verbatim.
    doc_end = fn.body[0].end_lineno - fn.body[0].lineno + 1
    out += body[:doc_end]
    out += binds
    out += body[doc_end:]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return nd_names, tags


# Arrays this kernel (or any ti.func it reaches) STORES into, derived by
# walking every ``*_taichi.py`` for a subscript-assignment or atomic on a
# parameter of these names. Over-inclusive is safe: it only makes the
# read-only arena smaller.
WRITTEN = {
    "sheet_memo",
    "sheet_accept",
    "event_pos",
    "event_snrm",
    "event_fnrm",
    "event_frame",
    "event_msk",
    "event_dp",
    "event_toff",
    "shadow_vis",
    "rs_ro",
    "rs_rd",
    "rs_acc",
    "rs_sca",
    "rs_int",
    "rs_pix",
    "pix_accum",
    "rs_alloc",
    "dump_out",
}

# Hot read-only arrays worth their own arena under the "fine" policy: the
# per-hit geometry and material fetches, and the per-pixel sheet records.
HOT = {
    "tri_pos": "geo",
    "tri_norm": "geo",
    "tri_mat": "mat",
    "tri_extra": "mat",
    "tri_mat_id": "mat",
    "sheet_offsets": "sheet",
    "sheet_key": "sheet",
    "sheet_ref": "sheet",
    "sheet_ab": "sheet",
    "sheet_cov": "sheet",
    "sheet_msk": "sheet",
    "sheet_cap": "sheet",
}


# Arrays read in the innermost loops: the per-sheet record fields, the
# per-hit geometry/material fetches, and the ray-state rows. Ordered most-hit
# first -- the "hybrid" policies keep this many as real ndarray parameters and
# arena everything else.
HOTTEST = [
    "sheet_key",
    "sheet_ref",
    "sheet_ab",
    "sheet_cov",
    "sheet_msk",
    "sheet_cap",
    "sheet_offsets",
    "tri_pos",
    "tri_norm",
    "tri_mat",
    "tri_mat_id",
    "covered_idx",
    "pix_accum",
    "rs_sca",
    "rs_int",
    "rs_ro",
    "rs_rd",
    "rs_acc",
    "rs_pix",
    "circuit_meta",
]


#: Per-ray state rows plus the per-pixel accumulator: the arrays indexed by
#: SLOT (up to ~4.6M rows) rather than by primitive or sheet.
RAY_STATE = ["rs_ro", "rs_rd", "rs_acc", "rs_sca", "rs_int", "rs_pix", "pix_accum"]


def keep_set(policy):
    """Names that stay real ndarray parameters under this policy.

    ``hybridN``      -- the N hottest, by ``HOTTEST`` order.
    ``keep:a+b+c``   -- exactly these names.
    """
    if policy.startswith("hybrid"):
        return set(HOTTEST[: int(policy[len("hybrid") :])])
    if policy.startswith("keep:"):
        spec = policy[len("keep:") :]
        out = set()
        for tok in spec.split("+"):
            if tok == "raystate":
                out |= set(RAY_STATE)
            elif tok:
                out.add(tok)
        return out
    return set()


def arena_tag(name, dtype_tag, policy):
    """Which arena an array lands in.

    ``dtype`` -- one arena per dtype (3). The minimum a single-buffer port can
    do, since a Taichi ndarray has one element type.
    ``role``  -- additionally separates arrays the kernel WRITES from the ones
    it only reads, so a store can never alias a geometry/material load.
    ``fine``  -- role, plus its own arena for each hot read-only family.
    """
    if policy.startswith(("hybrid", "keep:")):
        policy = "role"
    if policy == "dtype":
        return dtype_tag
    role = "rw" if name in WRITTEN else "ro"
    if policy == "rsown":
        # Every earlier grouping -- role and fine included -- left all seven
        # slot-indexed arrays sharing one read-write arena, so none of them
        # ever tested whether the penalty is those seven ALIASING EACH OTHER.
        # This one gives each its own buffer while keeping the arena calling
        # convention (base still comes from the offset table), which is the
        # only pair that separates sharing from indirection.
        if name in RAY_STATE:
            return f"{dtype_tag}_{name}"
        return f"{dtype_tag}_{role}"
    if policy == "role":
        return f"{dtype_tag}_{role}"
    group = HOT.get(name, role)
    return f"{dtype_tag}_{group}"


def build_layout(names, args, policy="dtype"):
    """Assign each ndarray argument an arena, an offset slot and shape slots.

    Names in the policy's keep-set get no arena slot -- they stay ordinary
    ndarray parameters.
    """
    keep = keep_set(policy)
    layout = {}
    off_slot = 0
    shp_slot = 0
    for name, (kind, val) in zip(names, args):
        if kind != "tensor" or name in keep:
            continue
        tag = arena_tag(name, DT_TAG[str(val.dtype)], policy)
        layout[name] = (tag, val.dim(), off_slot, shp_slot)
        off_slot += 1
        shp_slot += val.dim()
    return layout


#: Byte boundary each array's start is padded up to inside its arena. 0 packs
#: back-to-back, which is what the first round of this experiment did -- and
#: what put every one of the seven slot-indexed arrays at a base that is NOT a
#: multiple of the 128-byte transaction, while the shipped arm's separate torch
#: allocations are all 512-byte aligned. That is a memory-coalescing difference,
#: not an argument-passing one, so it has to be controllable to be excluded.
ALIGN_BYTES = 0


def pack(names, args, layout, device):
    """Concatenate every captured tensor into one flat arena per dtype."""
    import torch

    order = [n for n, (k, _) in zip(names, args) if k == "tensor" and n in layout]
    by_name = {n: v for n, (k, v) in zip(names, args) if k == "tensor"}
    bufs = {}
    offs = [0] * len(order)
    shps = []
    cursors = {}
    for name in order:
        tag, ndim, off_slot, _ = layout[name]
        t = by_name[name]
        cursors.setdefault(tag, 0)
        if ALIGN_BYTES:
            step = max(1, ALIGN_BYTES // t.element_size())
            cursors[tag] = -(-cursors[tag] // step) * step
        offs[off_slot] = cursors[tag]
        cursors[tag] += t.numel()
        shps.extend(list(t.shape))
    for tag, total in cursors.items():
        dt = {
            "f32": torch.float32,
            "i32": torch.int32,
            "i64": torch.int64,
            "u8": torch.uint8,
        }[tag.split("_")[0]]
        bufs[tag] = torch.empty(total, dtype=dt, device=device)
    for name in order:
        tag, ndim, off_slot, _ = layout[name]
        t = by_name[name].to(device)
        bufs[tag][offs[off_slot] : offs[off_slot] + t.numel()] = t.reshape(-1)
    aoff = torch.tensor(offs, dtype=torch.int32, device=device)
    ashp = torch.tensor(shps, dtype=torch.int32, device=device)
    return bufs, aoff, ashp, order


def unpack(bufs, layout, names, args, device):
    """Read each array back out of the arenas, for the bit-identity check."""
    out = {}
    for name, (kind, val) in zip(names, args):
        if kind != "tensor" or name not in layout:
            continue
        tag, ndim, off_slot, _ = layout[name]
        n = val.numel()
        base = _OFFS[off_slot]
        out[name] = bufs[tag][base : base + n].reshape(val.shape).clone()
    return out


_OFFS = None


def run_both(cap_path, cache_dir, reps, blocks=6, policies=("dtype", "role", "fine")):
    """Warm, in-process, alternating A/B: shipped kernel vs each arena policy.

    Cross-process wall clock swung ~17% between two runs of the SAME arm on
    this box, so the device-time comparison has to happen inside one process
    with the kernels interleaved. Compile time still has to be measured across
    processes (that is what a cold cache means) -- ``run_arm`` does that.
    """
    import importlib

    import taichi as ti
    import torch

    from algan.rendering.taichi_runtime import _sync_devices, taichi_init_kwargs

    kwargs = taichi_init_kwargs()
    kwargs["offline_cache_file_path"] = cache_dir
    kwargs["kernel_profiler"] = True
    ti.init(**kwargs)
    arch = ti.lang.impl.current_cfg().arch
    device = "cuda" if arch == ti.cuda else "cpu"

    cap = torch.load(cap_path, weights_only=False)
    names, args = cap["names"], cap["args"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gen_dir = os.path.join(repo_root, "benchmarks", "_arena_arg_gen")
    os.makedirs(gen_dir, exist_ok=True)
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)

    from algan.rendering.raytracing.sheet_resolve_taichi import (
        sheet_resolve_shade as k_ref,
    )

    ref_args = [v.to(device) if k == "tensor" else v for k, v in args]
    ref_pristine = [t.clone() if isinstance(t, torch.Tensor) else t for t in ref_args]

    def restore_ref():
        for cur, orig in zip(ref_args, ref_pristine):
            if isinstance(cur, torch.Tensor):
                cur.copy_(orig)

    arms = []
    for pol in policies:
        # "<policy>/const" is the diagnostic arm: same arenas, same shared
        # buffers, same aliasing -- but the offset and shape tables are gone,
        # baked in as literals. It is not shippable (the layout changes with
        # every scene, and each layout would be a fresh cold compile), it is
        # the only way to price the tables apart from the indirection.
        base_pol, _, mode = pol.partition("/")
        layout = build_layout(names, args, base_pol)
        kept = keep_set(base_pol)
        # "keep:a+b" is not an identifier; the module and kernel need one.
        slug = "".join(c if c.isalnum() else "_" for c in pol)
        mod_name = f"sheet_resolve_arena_{slug}"
        bufs, aoff, ashp, _ = pack(names, args, layout, device)
        consts = None
        if mode == "const":
            off_list = aoff.tolist()
            by_name = {n: v for n, (k, v) in zip(names, args) if k == "tensor"}
            consts = {
                nm: (off_list[layout[nm][2]], tuple(by_name[nm].shape)) for nm in layout
            }
        generate_variant(
            repo_root,
            layout,
            os.path.join(gen_dir, mod_name + ".py"),
            consts=consts,
            suffix=f"_{slug}",
        )
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        kernel = getattr(mod, KERNEL_NAME + f"_{slug}")
        tags = sorted({layout[n][0] for n in layout})
        # Signature order: every non-ndarray parameter and every KEPT ndarray
        # stays in its original position (that is how generate_variant builds
        # the parameter list), then the arenas, then the offset/shape tables.
        # Kept arrays get their own copies so an arm cannot clobber ref's.
        own = {}
        head = []
        for nm, (kind, val) in zip(names, args):
            if kind != "tensor":
                head.append(val)
            elif nm in kept:
                own[nm] = val.to(device).clone()
                head.append(own[nm])
        own_pristine = {nm: t.clone() for nm, t in own.items()}
        call = head + [bufs[t] for t in tags] + [aoff, ashp]
        pristine = {t: b.clone() for t, b in bufs.items()}
        arms.append(
            {
                "policy": pol,
                "kernel": kernel,
                "args": call,
                "bufs": bufs,
                "pristine": pristine,
                "layout": layout,
                "aoff": aoff.tolist(),
                "n_arenas": len(tags),
                "tags": tags,
                "own": own,
                "own_pristine": own_pristine,
                "kept": kept,
            }
        )

    def restore(arm):
        for t, b in arm["bufs"].items():
            b.copy_(arm["pristine"][t])
        for nm, t in arm["own"].items():
            t.copy_(arm["own_pristine"][nm])

    # Warm every specialization before any timing.
    for _ in range(2):
        restore_ref()
        k_ref(*ref_args)
        for arm in arms:
            restore(arm)
            arm["kernel"](*arm["args"])
    _sync_devices()

    # --- correctness: one clean launch per arm, compared to the shipped one
    restore_ref()
    k_ref(*ref_args)
    _sync_devices()
    ref_out = {
        n: v.clone() for n, v in zip(names, ref_args) if isinstance(v, torch.Tensor)
    }
    for arm in arms:
        restore(arm)
        arm["kernel"](*arm["args"])
        _sync_devices()
        diff = []
        for name, (kind, val) in zip(names, args):
            if kind != "tensor":
                continue
            if name in arm["kept"]:
                got = arm["own"][name]
            else:
                tag, _nd, off_slot, _ss = arm["layout"][name]
                base = arm["aoff"][off_slot]
                got = arm["bufs"][tag][base : base + val.numel()].reshape(val.shape)
            if not torch.equal(got, ref_out[name]):
                diff.append(name)
        arm["differing"] = diff

    # --- device time: ``query_kernel_profiler_info`` is unusable here (it
    # matches by name PREFIX and averages over OFFLOADED TASKS, not launches),
    # so take the profiler's TOTAL time per block. Only our launches happen
    # inside the window -- the restores are torch, not Taichi.
    ref_tot = 0.0
    for arm in arms:
        arm["tot"] = 0.0
    for _ in range(blocks):
        ti.profiler.clear_kernel_profiler_info()
        for _ in range(reps):
            restore_ref()
            _sync_devices()
            k_ref(*ref_args)
        _sync_devices()
        ref_tot += ti.profiler.get_kernel_profiler_total_time()
        for arm in arms:
            ti.profiler.clear_kernel_profiler_info()
            for _ in range(reps):
                restore(arm)
                _sync_devices()
                arm["kernel"](*arm["args"])
            _sync_devices()
            arm["tot"] += ti.profiler.get_kernel_profiler_total_time()

    n = blocks * reps
    ref_ms = ref_tot / n * 1000.0

    def launch_us(kernel, kargs):
        tiny = list(kargs)
        tiny[0] = 0  # num_covered = 0: times argument binding only
        for _ in range(20):
            kernel(*tiny)
        _sync_devices()
        t0 = time.perf_counter()
        for _ in range(500):
            kernel(*tiny)
        _sync_devices()
        return (time.perf_counter() - t0) / 500 * 1e6

    lu_ref = launch_us(k_ref, ref_args)
    for arm in arms:
        arm["launch_us"] = launch_us(arm["kernel"], arm["args"])
    lu_ref2 = launch_us(k_ref, ref_args)

    print()
    print(f"align_bytes={ALIGN_BYTES}")
    for arm in arms:
        bad = []
        for nm in RAY_STATE:
            if nm not in arm["layout"]:
                continue
            _t, _nd, off_slot, _ss = arm["layout"][nm]
            esz = 8 if _t.startswith("i64") else 4
            bad.append(f"{nm}:{arm['aoff'][off_slot] * esz % 128}")
        print(
            f"  {arm['policy']:>12s} ray-state base offsets mod 128B: "
            f"{' '.join(bad) or '(kept as params)'}"
        )
    print(
        f"{'arm':>10s} {'arenas':>7s} {'ndarr':>6s} "
        f"{'device ms':>10s} {'delta':>8s} {'launch us':>10s}  differing"
    )
    print(
        f"{'ref':>10s} {'-':>7s} "
        f"{sum(1 for a in ref_args if isinstance(a, torch.Tensor)):6d} "
        f"{ref_ms:10.3f} {'--':>8s} {lu_ref:10.0f}  "
        f"(recheck {lu_ref2:.0f} us)"
    )
    out = {
        "arch": str(arch),
        "launches_each": n,
        "ref_device_ms": ref_ms,
        "ref_launch_us": lu_ref,
        "ref_launch_us_again": lu_ref2,
        "arms": [],
    }
    for arm in arms:
        ms = arm["tot"] / n * 1000.0
        d = (ms - ref_ms) / ref_ms * 100.0
        nd = sum(1 for a in arm["args"] if isinstance(a, torch.Tensor))
        print(
            f"{arm['policy']:>10s} {arm['n_arenas']:7d} {nd:6d} "
            f"{ms:10.3f} {d:+7.1f}% {arm['launch_us']:10.0f}  "
            f"{arm['differing'] or 'none'}"
        )
        out["arms"].append(
            {
                "policy": arm["policy"],
                "n_arenas": arm["n_arenas"],
                "n_ndarray_args": nd,
                "device_ms": ms,
                "delta_pct": d,
                "launch_us": arm["launch_us"],
                "differing": arm["differing"],
                "tags": arm["tags"],
            }
        )
    return out


def run_arm(arm, cap_path, cache_dir, reps, tag="", policy="dtype"):
    global _OFFS
    import taichi as ti
    import torch

    from algan.rendering.taichi_runtime import _sync_devices, taichi_init_kwargs

    kwargs = taichi_init_kwargs()
    kwargs["offline_cache_file_path"] = cache_dir
    kwargs["kernel_profiler"] = True
    ti.init(**kwargs)
    arch = ti.lang.impl.current_cfg().arch
    device = "cuda" if arch == ti.cuda else "cpu"

    cap = torch.load(cap_path, weights_only=False)
    names, args = cap["names"], cap["args"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout = build_layout(names, args, policy)
    kept = keep_set(policy)
    slug = "".join(c if c.isalnum() else "_" for c in policy)

    if arm == "ref":
        from algan.rendering.raytracing.sheet_resolve_taichi import (
            sheet_resolve_shade as kernel,
        )

        name = KERNEL_NAME
        dev_args = [v.to(device) if k == "tensor" else v for k, v in args]
        pristine = [t.clone() if isinstance(t, torch.Tensor) else t for t in dev_args]

        def restore():
            for cur, orig in zip(dev_args, pristine):
                if isinstance(cur, torch.Tensor):
                    cur.copy_(orig)

        call_args = dev_args
    else:
        gen_dir = os.path.join(repo_root, "benchmarks", "_arena_arg_gen")
        os.makedirs(gen_dir, exist_ok=True)
        mod_path = os.path.join(gen_dir, f"sheet_resolve_arena_{slug}.py")
        nd_names, tags = generate_variant(
            repo_root, layout, mod_path, suffix=f"_{slug}"
        )
        sys.path.insert(0, gen_dir)
        import importlib

        mod = importlib.import_module(f"sheet_resolve_arena_{slug}")
        importlib.reload(mod)
        kernel = getattr(mod, KERNEL_NAME + f"_{slug}")
        name = KERNEL_NAME + f"_{slug}"

        bufs, aoff, ashp, order = pack(names, args, layout, device)
        _OFFS = aoff.tolist()
        own = {}
        head = []
        for nm, (kind, val) in zip(names, args):
            if kind != "tensor":
                head.append(val)
            elif nm in kept:
                own[nm] = val.to(device)
                head.append(own[nm])
        own_pristine = {nm: t.clone() for nm, t in own.items()}
        call_args = head + [bufs[t] for t in tags] + [aoff, ashp]
        pristine_bufs = {t: b.clone() for t, b in bufs.items()}

        def restore():
            for t, b in bufs.items():
                b.copy_(pristine_bufs[t])
            for nm, t in own.items():
                t.copy_(own_pristine[nm])

    n_nd = sum(1 for a in call_args if isinstance(a, torch.Tensor))

    # --- compile / cache-load time -------------------------------------
    restore()
    _sync_devices()
    t0 = time.perf_counter()
    kernel(*call_args)
    _sync_devices()
    first_launch = time.perf_counter() - t0

    # --- correctness: digest every array the kernel may have written ----
    if arm == "ref":
        result = {
            n: v.clone() for n, v in zip(names, dev_args) if isinstance(v, torch.Tensor)
        }
    else:
        result = unpack(bufs, layout, names, args, device)
        result.update({nm: t.clone() for nm, t in own.items()})
    digests = {n: float(v.double().abs().sum().item()) for n, v in result.items()}

    # --- steady-state device time --------------------------------------
    for _ in range(2):
        restore()
        kernel(*call_args)
    _sync_devices()
    ti.profiler.clear_kernel_profiler_info()
    for _ in range(reps):
        restore()
        _sync_devices()
        kernel(*call_args)
    _sync_devices()
    device_ms = float("nan")
    try:
        ti.profiler.get_kernel_profiler_total_time()
        device_ms = float(ti.profiler.query_kernel_profiler_info(name).avg)
    except Exception as exc:  # noqa: BLE001
        print("profiler query failed:", exc)

    # --- host launch cost ----------------------------------------------
    from algan.utils import taichi_fast_launch as tfl

    tfl.STATS["fast"] = tfl.STATS["slow"] = 0
    # num_covered = 0 makes the body a no-op, so this times argument binding.
    tiny = list(call_args)
    tiny[0] = 0
    for _ in range(20):
        kernel(*tiny)
    _sync_devices()
    t0 = time.perf_counter()
    for _ in range(500):
        kernel(*tiny)
    _sync_devices()
    launch_us = (time.perf_counter() - t0) / 500 * 1e6

    out_dir = os.path.join(repo_root, "benchmarks", "_arena_real_out")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {n: v.cpu() for n, v in result.items()},
        os.path.join(out_dir, f"result_{arm}{tag}.pt"),
    )

    return {
        "arm": arm,
        "arch": str(arch),
        "n_ndarray_args": n_nd,
        "n_total_args": len(call_args),
        "first_launch_s": first_launch,
        "device_ms": device_ms,
        "launch_overhead_us": launch_us,
        "fast_launch": [tfl.STATS["fast"], tfl.STATS["slow"]],
        "digests": digests,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--arm", choices=["ref", "arena"])
    ap.add_argument("--policies", default="dtype,role,fine")
    ap.add_argument(
        "--policy",
        default="dtype",
        help="arena grouping for the cross-process compile matrix",
    )
    ap.add_argument(
        "--both",
        action="store_true",
        help="warm in-process alternating A/B (device time)",
    )
    ap.add_argument("--cache", default="")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--align",
        type=int,
        default=0,
        help="pad each array's arena base to N bytes (0 = pack "
        "back-to-back, the original behaviour)",
    )
    a = ap.parse_args()

    global ALIGN_BYTES
    ALIGN_BYTES = a.align

    if a.both:
        rec = run_both(
            a.capture, a.cache, a.reps, policies=tuple(a.policies.split(","))
        )
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(rec, fh, indent=2)
        return
    if a.arm:
        print(
            "RESULT "
            + json.dumps(run_arm(a.arm, a.capture, a.cache, a.reps, a.tag, a.policy))
        )
        return

    import tempfile

    root = a.out or tempfile.mkdtemp(prefix="arena_real_")
    os.makedirs(root, exist_ok=True)
    recs = {}
    for phase in ("cold", "warm"):
        for arm in ("ref", "arena"):
            cache = os.path.join(root, "cache_" + arm)
            os.makedirs(cache, exist_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--capture",
                    a.capture,
                    "--arm",
                    arm,
                    "--cache",
                    cache,
                    "--reps",
                    str(a.reps),
                    "--tag",
                    f"_{phase}",
                    "--policy",
                    a.policy,
                    "--align",
                    str(a.align),
                ],
                capture_output=True,
                text=True,
                # A warm daemon would serve these from a process that may hold
                # another arm's monkey-patched module, and its ti.static gates
                # are already resolved. One fresh process per arm.
                env={**os.environ, "ALGAN_USE_DAEMON": "0", "ALGAN_AUTO_DAEMON": "0"},
            )
            line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print(proc.stdout[-6000:])
                print(proc.stderr[-6000:])
                raise SystemExit(f"{arm}/{phase} failed")
            rec = json.loads(line[0][7:])
            recs[(phase, arm)] = rec
            print(
                f"{phase:5s} {arm:6s} ndarrays={rec['n_ndarray_args']:3d} "
                f"total_args={rec['n_total_args']:3d} "
                f"first_launch={rec['first_launch_s']:8.3f}s "
                f"device={rec['device_ms']:8.3f}ms "
                f"launch={rec['launch_overhead_us']:8.1f}us"
            )

    d1 = recs[("warm", "ref")]["digests"]
    d2 = recs[("warm", "arena")]["digests"]
    bad = [k for k in d1 if d1[k] != d2.get(k)]
    print("\narrays with differing digests:", bad or "none (all identical)")


if __name__ == "__main__":
    main()
