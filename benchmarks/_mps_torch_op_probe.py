"""Which torch ops the renderer relies on are wrong on MPS.

The sheet compaction (``algan/rendering/raytracing/sheets.py``) is almost
entirely torch: a lexsort, a ``unique``, a couple of prefix sums and a lot of
segmented ``scatter_reduce_``. So when a Metal render comes back with 128
sheets where the CPU finds 40956 -- with MPS-friendly mode proven byte-identical
to normal mode ON THE CPU, so the substitutions are not the difference -- the
suspect is a torch op that answers differently on the Apple GPU, not a Taichi
kernel and not the mode.

This asks that question directly, and it is the right shape of question because
the answer is checkable in one process: run the op on MPS and on the CPU over
the SAME input and compare. No cross-arm log diffing, no render in the path, no
Taichi at all -- which matters, because a wrong answer here would otherwise
arrive as a wrong picture forty minutes downstream.

Every case is modelled on a real call site: same dtype, same shape scale, and a
value distribution that matches what the renderer actually feeds it (packed
``pixel << 32 | depth_bits`` keys, surface-major group keys, coverage in
[0, 1]). An op that is only wrong on realistic data is exactly the one a
synthetic probe misses.

    uv run python benchmarks/_mps_torch_op_probe.py

Prints one line per case and exits non-zero if any of them disagrees. On a
machine with no MPS it reports that and exits 0 -- it is a probe, and the
absence of an Apple GPU is not a failure.
"""

from __future__ import annotations

import os
import sys

# A silent CPU fallback would answer every question with "the CPU agrees with
# the CPU", which is the one answer this script must not be able to give.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch  # noqa: E402

#: Fragment-stream scale for the smoke scene, rounded up. Several of the ops
#: below are only wrong past some size, so probing at n = 10 proves nothing.
N = 60000
#: Deliberately larger than the smoke scene: a 4K frame runs to millions of
#: fragments, and an op that is right at 60 k and wrong at 1 M is a defect that
#: only appears on real content.
N_BIG = 1_000_000

_FAILURES: list[str] = []


def _report(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(name)


def _check(name, cpu, mps, exact=True, tol=0.0):
    """Compare one op's CPU and MPS results, reporting how they differ."""
    cpu_cmp, mps_cmp = cpu, mps.cpu()
    if cpu_cmp.dtype != mps_cmp.dtype:
        _report(name, False, f"dtype {cpu_cmp.dtype} vs {mps_cmp.dtype}")
        return
    if cpu_cmp.shape != mps_cmp.shape:
        _report(name, False, f"shape {tuple(cpu_cmp.shape)} vs {tuple(mps_cmp.shape)}")
        return
    if exact:
        bad = cpu_cmp != mps_cmp
    else:
        bad = (cpu_cmp.to(torch.float64) - mps_cmp.to(torch.float64)).abs() > tol
    count = int(bad.sum())
    if count == 0:
        _report(name, True)
        return
    where = int(bad.nonzero()[0][0])
    _report(
        name,
        False,
        f"{count}/{cpu_cmp.numel()} differ, first at {where}: "
        f"cpu {cpu_cmp.reshape(-1)[where]!s} vs mps {mps_cmp.reshape(-1)[where]!s}",
    )


def _stream(n, device, seed=0):
    """A fragment stream shaped like the rasteriser's own.

    ``frag_key`` is ``pixel << 32 | float_bits(depth)``, which is what makes
    the compaction's keys int64 and its shifts 32-bit -- the detail a probe
    written from memory rather than from the call site would get wrong.
    """
    g = torch.Generator().manual_seed(seed)
    pixel = torch.randint(0, 864 * 486, (n,), generator=g, dtype=torch.int64)
    depth = torch.rand(n, generator=g, dtype=torch.float32) * 8.0 + 1.0
    key = (pixel << 32) | depth.view(torch.int32).to(torch.int64)
    # Surface-major, like ``gkey = sid * 2 + facing``: a handful of distinct
    # values over the whole stream, which is what makes the second lexsort pass
    # a near-constant key.
    gkey = torch.randint(0, 6, (n,), generator=g, dtype=torch.int64)
    cov = torch.rand(n, generator=g, dtype=torch.float32).clamp_min(0.001)
    return (
        key.to(device),
        depth.to(device),
        gkey.to(device),
        cov.to(device),
        pixel.to(device),
    )


def _lexsort(*keys):
    """``sheets._lexsort``, verbatim, so the probe tests the real composition."""
    order = None
    for key in reversed(keys):
        k = key if order is None else key.index_select(0, order)
        o = torch.argsort(k, stable=True)
        order = o if order is None else order.index_select(0, o)
    return order


def probe_bit_packing(device):
    """The int64 key unpacking every stage downstream is built on."""
    print("\nint64 key packing (sheets.py:949-950)")
    key_c, depth_c, _, _, pixel_c = _stream(N, "cpu")
    key_m = key_c.to(device)
    _check("frag_key >> 32", key_c >> 32, key_m >> 32)
    _check("frag_key & 0xFFFFFFFF", key_c & 0xFFFFFFFF, key_m & 0xFFFFFFFF)
    # The renderer reads the depth back out by bit-casting the low word, so a
    # shift that is right and a view that is not still loses the sort key.
    low_c = (key_c & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    low_m = (key_m & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    _check("depth = low word viewed as f32", low_c, low_m)
    _check("recovered pixel == original", pixel_c, (key_m >> 32).cpu())
    _check("recovered depth == original", depth_c, low_m)


def probe_sort(device):
    """``_lexsort`` -- the compaction's P1, and everything after it."""
    print("\nstable argsort and the lexsort built on it (sheets.py:288-298)")
    for n in (N, N_BIG):
        key_c, depth_c, gkey_c, _, pixel_c = _stream(n, "cpu")
        key_m, depth_m, gkey_m = key_c.to(device), depth_c.to(device), gkey_c.to(device)
        _check(
            f"argsort(int64, stable) n={n}",
            torch.argsort(key_c, stable=True),
            torch.argsort(key_m, stable=True),
        )
        _check(
            f"argsort(float32, stable) n={n}",
            torch.argsort(depth_c, stable=True),
            torch.argsort(depth_m, stable=True),
        )
        # The near-constant key: six distinct values over a million elements is
        # the hardest case for a stable sort and the one _lexsort actually
        # feeds it.
        _check(
            f"argsort(int64 few-valued, stable) n={n}",
            torch.argsort(gkey_c, stable=True),
            torch.argsort(gkey_m, stable=True),
        )
        order_c = _lexsort(pixel_c, gkey_c, depth_c)
        order_m = _lexsort(pixel_c.to(device), gkey_m, depth_m)
        _check(f"_lexsort(pix, gkey, t) n={n}", order_c, order_m)
        # Even where the permutation differs, the SORTED SEQUENCE must not --
        # that separates "stability differs" from "the sort is wrong", which
        # are very different defects.
        _check(
            f"_lexsort sorted pixels agree n={n}",
            pixel_c.index_select(0, order_c),
            pixel_c.to(device).index_select(0, order_m),
        )


def probe_unique(device):
    """``torch.unique`` -- what turns band ids into the sheet count."""
    print("\nunique (sheets.py:1111 -- this is what num_sheets counts)")
    for n in (N, N_BIG):
        g = torch.Generator().manual_seed(1)
        # ``cid = band_id * 16 + rank``: band ids run to roughly the fragment
        # count, and the low four bits are the conflict rank.
        band_c = torch.arange(n, dtype=torch.int64) // 2
        rank_c = torch.randint(0, 3, (n,), generator=g, dtype=torch.int64)
        cid_c = band_c * 16 + rank_c
        cid_m = cid_c.to(device)
        u_c, inv_c = torch.unique(cid_c, sorted=True, return_inverse=True)
        u_m, inv_m = torch.unique(cid_m, sorted=True, return_inverse=True)
        _report(
            f"unique count n={n}",
            int(u_c.numel()) == int(u_m.numel()),
            f"cpu {int(u_c.numel())} vs mps {int(u_m.numel())}",
        )
        if int(u_c.numel()) == int(u_m.numel()):
            _check(f"unique values n={n}", u_c, u_m)
            _check(f"unique inverse n={n}", inv_c, inv_m)


def probe_scans(device):
    """The prefix sums that turn per-element flags into segment ids."""
    print("\ncumsum (sheets.py:1074, 1178, 1607, 1700)")
    g = torch.Generator().manual_seed(2)
    for n in (N, N_BIG):
        start_c = torch.rand(n, generator=g) < 0.7
        start_m = start_c.to(device)
        _check(
            f"cumsum(bool -> int64) n={n}",
            torch.cumsum(start_c.to(torch.int64), 0),
            torch.cumsum(start_m.to(torch.int64), 0),
        )
        _check(
            f"cumsum(int32) n={n}",
            torch.cumsum(start_c.to(torch.int32), 0),
            torch.cumsum(start_m.to(torch.int32), 0),
        )
        area_c = torch.rand(n, generator=g, dtype=torch.float32)
        _check(
            f"cumsum(float32) n={n}",
            torch.cumsum(area_c, 0),
            torch.cumsum(area_c.to(device), 0),
            exact=False,
            # A parallel scan reassociates, so this is a tolerance rather than
            # an equality -- but it must still track, and 1e-2 over a sum that
            # reaches n/2 is a very loose leash.
            tol=1e-2,
        )


def probe_segmented(device):
    """The segmented reductions the band aggregates are made of."""
    print("\nscatter_add_ / scatter_reduce_ (sheets.py:1619, 1698 and P2)")
    g = torch.Generator().manual_seed(3)
    n, segments = N, N // 3
    idx_c = torch.randint(0, segments, (n,), generator=g, dtype=torch.int64)
    val_c = torch.rand(n, generator=g, dtype=torch.float32)
    idx_m, val_m = idx_c.to(device), val_c.to(device)

    def add(idx, val, dev):
        out = torch.zeros(segments, dtype=torch.float32, device=dev)
        out.scatter_add_(0, idx, val)
        return out

    _check(
        "scatter_add_(float32)",
        add(idx_c, val_c, "cpu"),
        add(idx_m, val_m, device),
        exact=False,
        tol=1e-4,
    )

    def add_i(idx, dev, dtype):
        out = torch.zeros(segments, dtype=dtype, device=dev)
        out.scatter_add_(0, idx, torch.ones_like(idx, dtype=dtype))
        return out

    _check(
        "scatter_add_(int64 counts)",
        add_i(idx_c, "cpu", torch.int64),
        add_i(idx_m, device, torch.int64),
    )
    _check(
        "scatter_add_(int32 counts)",
        add_i(idx_c, "cpu", torch.int32),
        add_i(idx_m, device, torch.int32),
    )

    for reduce_op, fill in (("amin", float("inf")), ("amax", float("-inf"))):

        def red(idx, val, dev, reduce_op=reduce_op, fill=fill):
            out = torch.full((segments,), fill, dtype=torch.float32, device=dev)
            out.scatter_reduce_(0, idx, val, reduce=reduce_op, include_self=True)
            return out

        _check(
            f"scatter_reduce_(float32, {reduce_op})",
            red(idx_c, val_c, "cpu"),
            red(idx_m, val_m, device),
        )

    # int32 rather than int64 because that is what MPS-friendly mode narrows
    # these to (`mps_compat.reduction_index_dtype`); the int64 spelling is the
    # one Metal has no atomic for.
    pos_c = torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int32)
    pos_m = pos_c.to(device)
    for reduce_op, fill in (("amin", 2147483647), ("amax", -2147483648)):

        def red_i(idx, val, dev, reduce_op=reduce_op, fill=fill):
            out = torch.full((segments,), fill, dtype=torch.int32, device=dev)
            out.scatter_reduce_(0, idx, val, reduce=reduce_op, include_self=True)
            return out

        _check(
            f"scatter_reduce_(int32, {reduce_op})",
            red_i(idx_c, pos_c, "cpu"),
            red_i(idx_m, pos_m, device),
        )


def probe_lookup(device):
    """``searchsorted`` and the gathers -- the CSR the resolve indexes with."""
    print("\nsearchsorted / index_select (sheets.py:1631, 1697)")
    g = torch.Generator().manual_seed(4)
    sorted_c = torch.unique(
        torch.randint(0, 1 << 20, (N,), generator=g).to(torch.int64)
    )
    query_c = torch.randint(0, 1 << 20, (N,), generator=g, dtype=torch.int64)
    _check(
        "searchsorted(int64)",
        torch.searchsorted(sorted_c, query_c),
        torch.searchsorted(sorted_c.to(device), query_c.to(device)),
    )
    src_c = torch.rand(N, generator=g, dtype=torch.float32)
    pick_c = torch.randint(0, N, (N,), generator=g, dtype=torch.int64)
    _check(
        "index_select(float32, int64 index)",
        src_c.index_select(0, pick_c),
        src_c.to(device).index_select(0, pick_c.to(device)),
    )
    big_c = torch.randint(-(1 << 62), 1 << 62, (N,), generator=g, dtype=torch.int64)
    _check(
        "index_select(int64 values)",
        big_c.index_select(0, pick_c),
        big_c.to(device).index_select(0, pick_c.to(device)),
    )


def probe_int64_arithmetic(device):
    """Plain int64 arithmetic, because Metal has no native 64-bit integer.

    The compaction builds keys well past 2**32 (``pix * K + sid``, ``band_id *
    16 + rank``, ``pk * AA_NUM_SAMPLES + lane``), so a backend that silently
    computes these in 32 bits produces collisions rather than an error -- and a
    collision reads downstream as two sheets fused into one.
    """
    print("\nint64 arithmetic past 2**32 (sheets.py:1109, 1143, 1589)")
    g = torch.Generator().manual_seed(5)
    a_c = torch.randint(0, 1 << 40, (N,), generator=g, dtype=torch.int64)
    b_c = torch.randint(0, 1 << 20, (N,), generator=g, dtype=torch.int64)
    a_m, b_m = a_c.to(device), b_c.to(device)
    _check("int64 a * 16 + b", a_c * 16 + b_c, a_m * 16 + b_m)
    _check("int64 a + b", a_c + b_c, a_m + b_m)
    _check("int64 a // 16", a_c // 16, a_m // 16)
    _check("int64 round trip", a_c, a_m)
    _check("int64 comparison", (a_c[1:] != a_c[:-1]), (a_m[1:] != a_m[:-1]))
    _check("int64 -> float32 cast", a_c.to(torch.float32), a_m.to(torch.float32))


def main() -> int:
    print(f"torch {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("no MPS on this machine -- nothing to probe")
        return 0
    device = torch.device("mps")
    print(f"probing {device} against cpu, fallback disabled\n")

    for probe in (
        probe_bit_packing,
        probe_int64_arithmetic,
        probe_sort,
        probe_unique,
        probe_scans,
        probe_segmented,
        probe_lookup,
    ):
        try:
            probe(device)
        except Exception as exc:  # noqa: BLE001
            # An op MPS has not implemented raises with the fallback off, and
            # that is a result rather than a crash: record it and keep going,
            # because the rest of the table is what says how much else works.
            _report(f"{probe.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} op(s) disagree with the CPU:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("every probed op agrees with the CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
