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

    # The WIDE key, which is the one the shading-class split actually builds:
    # ``skey = band * _SHADE_CLASS_BASE + cls`` with ``_SHADE_CLASS_BASE =
    # 1 << 25`` (sheets.py:120, 1267). For the smoke scene that reaches ~2**40,
    # against the ~2**20 the case above covers -- and the case above passes on
    # MPS while the render collapses 40956 sheets to 128, so the untested
    # magnitude is the interesting one. int64 SURVIVES a round trip at 2**40
    # (see the arithmetic probe), so this asks specifically whether ``unique``
    # -- which sorts internally -- keeps the low bits that distinguish two
    # bands.
    print("\n  unique over the shading-class key (sheets.py:1267, ~2**40)")
    shade_base = 1 << 25
    for bands in (4096, 40956):
        g = torch.Generator().manual_seed(7)
        band_c = torch.arange(bands, dtype=torch.int64).repeat_interleave(2)
        cls_c = torch.randint(0, 3, (band_c.numel(),), generator=g, dtype=torch.int64)
        skey_c = band_c * shade_base + cls_c
        u_c = torch.unique(skey_c, sorted=True)
        u_m = torch.unique(skey_c.to(device), sorted=True)
        ok = int(u_c.numel()) == int(u_m.numel())
        _report(
            f"unique(band * 2**25 + cls) bands={bands} max=2**{int(skey_c.max()).bit_length() - 1}",
            ok,
            f"cpu {int(u_c.numel())} distinct vs mps {int(u_m.numel())}",
        )
        if ok:
            _check(f"unique wide-key values bands={bands}", u_c, u_m)


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
            # an equality. It has to be RELATIVE: the running sum reaches n * 0.5,
            # and float32 carries ~7 significant digits, so an absolute leash
            # that is generous at n = 60 k is below one ulp at n = 1 M -- which
            # is exactly how this case first reported a failure (23475.7715 vs
            # 23475.7598, a 5e-7 relative difference) that was the probe's bug
            # and not the GPU's.
            tol=1e-4 * n,
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


def probe_gather_isolated(device):
    """Exactly which gather, on exactly which dtype, at exactly which width.

    The rest of this file compares ``f(cpu_inputs)`` against ``f(mps_inputs)``
    where the MPS inputs were themselves derived on the device -- fine for
    catching a difference, useless for attributing one. The earlier
    ``index_select(int32 ...)`` case built its operand with an on-device ``&``
    and ``.to(torch.int32)``, so a failure there could have been the mask, the
    narrowing cast, or the gather.

    This isolates the gather and nothing else:

    1. build the values on the CPU and move them over;
    2. **prove the move was exact** -- if the round trip already lost bits the
       gather is not what to blame, and the comparison below means nothing;
    3. gather with an index also built on the CPU;
    4. compare against the CPU's gather of the same values.

    It also asks whether the defect is ``index_select`` specifically or every
    gather, because that decides how much of the renderer is exposed: a
    ``t[idx]`` or a ``torch.gather`` that is exact would be a workaround, and
    one that is not says the whole class is affected.
    """
    print("\nisolated gathers: is the VALUE dtype or the gather at fault?")
    g = torch.Generator().manual_seed(11)
    n = 4096
    index_c = torch.randint(0, n, (n,), generator=g, dtype=torch.int64)
    index_m = index_c.to(device)

    cases = []
    for bits in (16, 20, 23, 24, 25, 30):
        cases.append((f"int32 2**{bits}", torch.int32, bits))
    for bits in (24, 25, 40, 62):
        cases.append((f"int64 2**{bits}", torch.int64, bits))

    for name, dtype, bits in cases:
        low = 1 << (bits - 1)
        high = (1 << bits) - 1
        values_c = torch.randint(low, high, (n,), generator=g, dtype=torch.int64)
        values_c = values_c.to(dtype)
        values_m = values_c.to(device)
        # Step 2: the move must be exact, or nothing below is attributable.
        if not torch.equal(values_c, values_m.cpu()):
            _report(f"{name}: round trip", False, "the values changed on the way")
            continue
        want = values_c.index_select(0, index_c)
        got = values_m.index_select(0, index_m).cpu()
        bad = int((want != got).sum())
        if bad == 0:
            _report(f"{name} index_select", True)
            continue
        where = int((want != got).nonzero()[0][0])
        _report(
            f"{name} index_select",
            False,
            f"{bad}/{n} differ, first cpu {int(want[where])} vs mps {int(got[where])}",
        )

    # Is it index_select, or every gather? Asked at one width known to fail.
    values_c = torch.randint(1 << 40, (1 << 41) - 1, (n,), generator=g)
    values_m = values_c.to(device)
    want = values_c.index_select(0, index_c)
    for label, run in (
        ("index_select", lambda v, i: v.index_select(0, i)),
        ("torch.gather", lambda v, i: torch.gather(v, 0, i)),
        ("advanced indexing v[i]", lambda v, i: v[i]),
        ("torch.take", lambda v, i: torch.take(v, i)),
        ("v.repeat_interleave(2)[::2]", lambda v, i: v.repeat_interleave(2)[::2]),
    ):
        try:
            got = run(values_m, index_m).cpu()
        except Exception as exc:  # noqa: BLE001
            _report(f"int64 2**40 via {label}", False, f"{type(exc).__name__}: {exc}")
            continue
        reference = want if label != "v.repeat_interleave(2)[::2]" else values_c
        _report(f"int64 2**40 via {label}", bool(torch.equal(reference, got)))


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
    # The packed key at its real magnitude, and then the SPLIT gather that
    # `mps_compat.gather_packed_key` replaces it with. The first is the defect
    # -- a 25-bit gather of a 2**50 key masks the low word with 0xFC000000, so
    # every depth in [4, 8) reads back as exactly 2.0, which is what the Apple
    # GPU produced. The second is the fix, and every op in it is 32 bits wide
    # or narrower.
    key_c, depth_c, _, _, _ = _stream(N, "cpu")
    key_m = key_c.to(device)
    pick_m = pick_c.to(device)
    _check(
        "index_select(packed pixel<<32|depth key)",
        key_c.index_select(0, pick_c),
        key_m.index_select(0, pick_m),
    )

    def halves_gather(key, pick):
        """The FIRST attempt: two 32-bit halves. Kept because it fails.

        A 32-bit half still reaches 2**32, above the 2**24 ceiling the ladder
        measures, so this rounds -- less than the full-width gather, enough to
        leave a render wrong. Probing it keeps that fact attached to the code
        that replaced it.
        """
        high = (key >> 32).to(torch.int32).index_select(0, pick)
        low = (key & 0xFFFFFFFF).to(torch.int32).index_select(0, pick)
        return (high.to(torch.int64) << 32) | (low.to(torch.int64) & 0xFFFFFFFF)

    def lane_gather(key, pick):
        """What ``mps_compat.gather_packed_key`` does: four 16-bit lanes."""
        out = None
        for shift in range(0, 64, 16):
            lane = ((key >> shift) & 0xFFFF).to(torch.int32).index_select(0, pick)
            part = lane.to(torch.int64) << shift
            out = part if out is None else (out | part)
        return out

    _check(
        "two-half gather of the same key (rejected: still over 2**24)",
        key_c.index_select(0, pick_c),
        halves_gather(key_m, pick_m),
    )
    _check(
        "four-lane gather of the same key (correct, superseded by v[i])",
        key_c.index_select(0, pick_c),
        lane_gather(key_m, pick_m),
    )
    _check(
        "four-lane gather recovers the depths",
        depth_c.index_select(0, pick_c),
        (lane_gather(key_m, pick_m) & 0xFFFFFFFF).to(torch.int32).view(torch.float32),
    )
    _check(
        "v[i] gather of the same key (THE SHIPPED FIX)",
        key_c.index_select(0, pick_c),
        key_m[pick_m],
    )
    _check(
        "index_select(int32 values near 2**30)",
        (key_c & 0xFFFFFFFF).to(torch.int32).index_select(0, pick_c),
        (key_m & 0xFFFFFFFF).to(torch.int32).index_select(0, pick_m),
    )
    _check(
        "index_select(int32 values under 2**16)",
        (key_c & 0xFFFF).to(torch.int32).index_select(0, pick_c),
        (key_m & 0xFFFF).to(torch.int32).index_select(0, pick_m),
    )
    _check(
        "int64 (hi << 32) | lo repack",
        key_c,
        ((key_m >> 32) << 32) | (key_m & 0xFFFFFFFF),
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

    # WHERE the int64 ops stop being exact, which is the actionable half. The
    # two that fail above -- floor division and a gather of full-width values --
    # look like a float round trip rather than true 64-bit integer work, and if
    # that is what they are then the ceiling is a mantissa width and every value
    # below it is safe. That distinction decides whether the renderer is
    # affected at all: its int64s are a packed `pixel << 32 | depth` key
    # (~2**50, but only ever shifted and masked) and a pile of small composite
    # ids (~2**21, divided and gathered constantly).
    print("\n  where int64 exactness ends (bits -> first failing op)")
    for bits in (20, 24, 30, 40, 50, 62):
        v_c = torch.randint(1 << (bits - 1), 1 << bits, (4096,), generator=g)
        v_m = v_c.to(device)
        pick_c = torch.randint(0, 4096, (4096,), generator=g, dtype=torch.int64)
        broken = []
        if not torch.equal(v_c // 16, (v_m // 16).cpu()):
            broken.append("//")
        if not torch.equal(
            v_c.index_select(0, pick_c),
            v_m.index_select(0, pick_c.to(device)).cpu(),
        ):
            broken.append("index_select")
        if not torch.equal(v_c, v_m.cpu()):
            broken.append("round trip")
        if not torch.equal(v_c * 3 + 1, (v_m * 3 + 1).cpu()):
            broken.append("mul/add")
        print(f"    2**{bits}: {', '.join(broken) if broken else 'all exact'}")


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
        probe_gather_isolated,
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
