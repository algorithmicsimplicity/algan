"""Where MPS's ``torch.unique`` merges distinct integers, and which ops do not.

``DESIGN_mps_support.md`` §4.10 recorded ``torch.unique`` on torch 2.13.0
returning 2,010,925 of 2,091,233 distinct sorted ids (run 42329686918) while
``unique_consecutive`` over the same input was exact. This asks what that one
reading leaves open: which forms of the op merge (``sorted``,
``return_inverse``, ``return_counts``, ``dim=``), at which dtypes and from
which magnitude and size, and whether the ops an exact replacement would be
built from -- ``sort``, ``argsort``, ``unique_consecutive``, a compare, a scan
and advanced indexing -- are exact over the same inputs.

Every case builds its input on the host, checks the copy to MPS is exact, runs
the op on both devices and compares, so a failure is the op's alone. One line
per case, printed as it runs.

    .venv/bin/python benchmarks/_mps_unique_probe.py [section ...]

Sections: ``repro``, ``ladder``, ``sizes``, ``forms``, ``dtypes``, ``sort``,
``consecutive``, ``exact``, ``blocks`` (all by default). Exits 1 when an op
disagrees with the CPU, and 0 on a machine without MPS, where there is nothing
to probe.
"""

from __future__ import annotations

import os
import sys
import time

# A silent CPU fallback would answer every question with the CPU's own answer.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch  # noqa: E402

_T0 = time.time()
_FAILS: list[str] = []
DEVICE = torch.device("mps")


def _say(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def _report(name, ok, detail=""):
    _say(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


def _dev(x):
    y = x.to(DEVICE)
    if not torch.equal(y.cpu(), x):
        raise AssertionError("the copy to the device is not exact")
    return y


def _f32(x):
    """``x`` rounded through a float32: the gather ceiling's signature."""
    return x.to(torch.float32).to(x.dtype)


def _mag(value):
    """``2**k``, or ``2**k - d`` for a start just under one."""
    value = int(value)
    if value <= 0:
        return str(value)
    k = value.bit_length()
    if value & (value - 1) == 0:
        return f"2**{k - 1}"
    return f"2**{k} - {2**k - value}" if 2**k - value <= 2 ** (k - 8) else str(value)


def _consecutive(start, n, dtype):
    return start + torch.arange(n, dtype=torch.int64).to(dtype)


def _shuffled(x, seed=0):
    g = torch.Generator().manual_seed(seed)
    return x[torch.randperm(x.numel(), generator=g)]


def _with_repeats(start, n, dtype, seed=0):
    """``n`` values from ``start`` on, about half of them repeated, shuffled."""
    g = torch.Generator().manual_seed(seed)
    x = start + torch.randint(0, max(1, n // 2), (n,), generator=g, dtype=torch.int64)
    return x.to(dtype)


def _describe_unique(x, values):
    """What MPS's unique values are, against the hypotheses that separate them."""
    cpu = torch.unique(x)
    if torch.equal(values, cpu):
        return "exact"
    rounded = torch.unique(_f32(x))
    low_words = torch.unique(x.to(torch.int32).to(torch.float32)).to(x.dtype)
    tags = [f"{values.numel()} of {cpu.numel()} distinct"]
    if torch.equal(values, rounded):
        tags.append("= unique(float32(x))")
    elif torch.equal(values, low_words):
        tags.append("= unique(float32(int32(x)))")
    elif bool(torch.isin(values, x).all()):
        tags.append(f"values all from x (float32 groups: {rounded.numel()})")
    else:
        tags.append("values not all from x")
    missing = cpu[~torch.isin(cpu, values)]
    if missing.numel():
        tags.append(f"first lost {int(missing[0])}")
    return ", ".join(tags)


def _check_unique(name, x, **kw):
    """``torch.unique(x, **kw)`` on MPS against the CPU, every output."""
    want = torch.unique(x, **kw)
    try:
        got = torch.unique(_dev(x), **kw)
    except Exception as exc:  # noqa: BLE001 -- a raise is a reading
        _report(name, False, f"raised {type(exc).__name__}: {str(exc)[:160]}")
        return False
    want = want if isinstance(want, tuple) else (want,)
    got = tuple(g.cpu() for g in (got if isinstance(got, tuple) else (got,)))
    values = got[0]
    ok = all(torch.equal(w, g) for w, g in zip(want, got))
    detail = "" if ok else _describe_unique(x.reshape(-1), values.reshape(-1))
    if not ok and kw.get("return_inverse") and kw.get("dim") is None:
        inverse = got[1]
        # Is the device's own answer at least self-consistent?
        detail += f"; values[inverse] == x: {torch.equal(values[inverse], x)}"
    _report(name, ok, detail)
    return ok


def probe_repro():
    """The smallest input that shows it, for an upstream report::

        x = torch.arange(2**24, 2**24 + 8, device="mps")
        torch.unique(x, return_inverse=True)

    Each form is printed beside the CPU's answer. Run 42351444222 had the
    values-only form exact and the inverse form merging.
    """
    _say("repro: x = arange(2**24, 2**24 + 8), int64")
    x = _consecutive(2**24, 8, torch.int64)
    for kw in ({}, {"return_inverse": True}):
        want = torch.unique(x, **kw)
        got = torch.unique(_dev(x), **kw)
        want = want if isinstance(want, tuple) else (want,)
        got = tuple(g.cpu() for g in (got if isinstance(got, tuple) else (got,)))
        ok = all(torch.equal(w, g) for w, g in zip(want, got))
        args = "".join(f", {k}={v}" for k, v in kw.items())
        _report(
            f"unique(x{args})",
            ok,
            f"mps {[g.tolist() for g in got]}, cpu {[w.tolist() for w in want]}",
        )


def probe_ladder():
    """Magnitude ladder: 8192 consecutive values from each start, both widths."""
    _say("ladder: unique(sorted=True, return_inverse, return_counts) by magnitude")
    n = 8192
    for dtype, starts in (
        (
            torch.int32,
            (0, 2**20, 2**23, 2**24 - n // 2, 2**24, 2**25, 2**26, 2**30, 2**31 - n),
        ),
        (
            torch.int64,
            (0, 2**24 - n // 2, 2**24, 2**25, 2**30, 2**31, 2**32, 2**40, 2**53, 2**62),
        ),
    ):
        for start in starts:
            x = _consecutive(start, n, dtype)
            for order, xs in (("sorted", x), ("shuffled", _shuffled(x))):
                _check_unique(
                    f"{dtype} from {_mag(start)} ({start}) n={n} {order}",
                    xs,
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )


def probe_sizes():
    """Where in the input size it starts, at a magnitude that merges."""
    _say("sizes: unique(return_inverse) over consecutive int64s from 2**25, by count")
    for n in (
        2,
        3,
        4,
        8,
        16,
        64,
        256,
        1024,
        2047,
        2048,
        4095,
        4096,
        4097,
        8192,
        2**16,
        2**20,
    ):
        _check_unique(
            f"n={n}", _consecutive(2**25, n, torch.int64), return_inverse=True
        )


def probe_forms():
    """Which spellings merge, over 8192 consecutive int64s from 2**24."""
    _say("forms: 8192 consecutive int64s from 2**24")
    x = _consecutive(2**24, 8192, torch.int64)
    xs = _shuffled(x)
    cases = [
        ("unique(x)", xs, {}),
        ("unique(x, sorted=False)", xs, {"sorted": False}),
        ("unique(x, return_inverse=True)", xs, {"return_inverse": True}),
        ("unique(x, return_counts=True)", xs, {"return_counts": True}),
        (
            "unique(x, inverse, counts)",
            xs,
            {"return_inverse": True, "return_counts": True},
        ),
        ("unique(x, dim=0)", xs, {"dim": 0}),
        (
            "unique(x, dim=0, inverse, counts)",
            xs,
            {"dim": 0, "return_inverse": True, "return_counts": True},
        ),
        ("unique(rows [x, x % 3], dim=0)", torch.stack((xs, xs % 3), 1), {"dim": 0}),
        ("unique(rows [x % 3, x], dim=0)", torch.stack((xs % 3, xs), 1), {"dim": 0}),
        ("unique(columns, dim=1)", torch.stack((xs, xs % 3), 0), {"dim": 1}),
        ("unique(2-D, flattened)", xs.reshape(64, 128), {}),
    ]
    for name, inp, kw in cases:
        if kw.get("sorted") is False:
            # The CPU's own order for sorted=False is unspecified: compare sets.
            got = torch.unique(_dev(inp), sorted=False).cpu()
            ok = torch.equal(got.sort().values, torch.unique(inp))
            _report(name, ok, "" if ok else _describe_unique(inp, got.sort().values))
            continue
        _check_unique(name, inp, **kw)
    _say("forms: Tensor.unique and unique_consecutive over the same values")
    got = _dev(xs).unique().cpu()
    _report(
        "Tensor.unique()",
        torch.equal(got, x),
        "" if torch.equal(got, x) else _describe_unique(xs, got),
    )
    for kw in ({}, {"return_inverse": True}, {"return_counts": True}, {"dim": 0}):
        want = torch.unique_consecutive(x, **kw)
        got = torch.unique_consecutive(_dev(x), **kw)
        want = want if isinstance(want, tuple) else (want,)
        got = tuple(g.cpu() for g in (got if isinstance(got, tuple) else (got,)))
        _report(
            f"unique_consecutive(sorted x, {kw})",
            all(torch.equal(w, g) for w, g in zip(want, got)),
            f"{got[0].numel()} of {want[0].numel()}",
        )


def probe_dtypes():
    """Every dtype, at a range where its values are distinct and large."""
    _say("dtypes")
    cases = [
        ("uint8 all values", torch.arange(256, dtype=torch.int64).to(torch.uint8)),
        ("int8 all values", torch.arange(-128, 128, dtype=torch.int64).to(torch.int8)),
        (
            "int16 all values",
            torch.arange(-(2**15), 2**15, dtype=torch.int64).to(torch.int16),
        ),
        ("int32 from 2**25", _consecutive(2**25, 8192, torch.int32)),
        ("int32 negative to -2**25", -_consecutive(2**25, 8192, torch.int32)),
        ("int64 from 2**25", _consecutive(2**25, 8192, torch.int64)),
        ("int64 negative to -2**25", -_consecutive(2**25, 8192, torch.int64)),
        ("float16 distinct", torch.arange(2048, dtype=torch.float32).to(torch.float16)),
        (
            "float32 2**30 + 128 k",
            (2**30 + 128 * torch.arange(8192, dtype=torch.float64)).to(torch.float32),
        ),
        (
            "float32 random",
            torch.rand(8192, generator=torch.Generator().manual_seed(3)),
        ),
        ("bool", torch.arange(8192) % 2 == 0),
    ]
    for name, x in cases:
        _check_unique(name, _shuffled(x), return_inverse=True, return_counts=True)


def _check_sort(name, x):
    """sort / argsort, plain and stable, ascending and descending."""
    for label, fn in (
        ("sort", lambda v: torch.sort(v)),
        ("sort stable", lambda v: torch.sort(v, stable=True)),
        (
            "sort stable descending",
            lambda v: torch.sort(v, stable=True, descending=True),
        ),
        ("argsort", lambda v: torch.argsort(v)),
        ("argsort stable", lambda v: torch.argsort(v, stable=True)),
    ):
        want = fn(x)
        try:
            got = fn(_dev(x))
        except Exception as exc:  # noqa: BLE001
            _report(
                f"{label} {name}",
                False,
                f"raised {type(exc).__name__}: {str(exc)[:160]}",
            )
            continue
        if isinstance(want, torch.Tensor):  # argsort: indices only
            idx = got.cpu()
            values_ok = torch.equal(x[idx], x[want])
            index_ok = torch.equal(idx, want) if "stable" in label else values_ok
            _report(
                f"{label} {name}",
                values_ok and index_ok,
                "" if values_ok else "order wrong",
            )
            continue
        values, idx = got[0].cpu(), got[1].cpu()
        values_ok = torch.equal(values, want[0])
        perm_ok = torch.equal(x[idx], values) and torch.equal(
            idx.sort().values, torch.arange(x.numel())
        )
        index_ok = torch.equal(idx, want[1]) if "stable" in label else True
        detail = []
        if not values_ok:
            bad = int((values != want[0]).sum())
            detail.append(
                f"{bad} values differ, first {int(values[(values != want[0]).nonzero()[0]])} vs {int(want[0][(values != want[0]).nonzero()[0]])}"
            )
        if not perm_ok:
            detail.append("indices do not reproduce the values")
        if not index_ok:
            detail.append("stable indices differ")
        _report(
            f"{label} {name}", values_ok and perm_ok and index_ok, "; ".join(detail)
        )


def probe_sort():
    """Is the sort a replacement would stand on exact at the same sizes?"""
    _say("sort / argsort over values past 2**24, with repeats, shuffled")
    for dtype, starts in (
        (torch.int32, (2**20, 2**24, 2**25, 2**30)),
        (torch.int64, (2**24, 2**25, 2**30, 2**40, 2**62)),
    ):
        for start in starts:
            for n in (8192, 2**20):
                _check_sort(
                    f"{dtype} from {_mag(start)} n={n}", _with_repeats(start, n, dtype)
                )


def probe_consecutive():
    """``unique_consecutive`` over sorted runs, the ladder and the sizes."""
    _say("unique_consecutive(return_inverse, return_counts) over sorted repeats")
    for dtype, starts in (
        (torch.int32, (2**24, 2**25, 2**30)),
        (torch.int64, (2**24, 2**25, 2**30, 2**31, 2**32, 2**40, 2**62)),
    ):
        for start in starts:
            for n in (16, 4096, 8192, 2**20, 2**22):
                x = _with_repeats(start, n, dtype).sort().values
                want = torch.unique_consecutive(
                    x, return_inverse=True, return_counts=True
                )
                got = torch.unique_consecutive(
                    _dev(x), return_inverse=True, return_counts=True
                )
                got = tuple(g.cpu() for g in got)
                ok = all(torch.equal(w, g) for w, g in zip(want, got))
                detail = ""
                if not ok:
                    wrong = [
                        name
                        for name, w, g in zip(
                            ("values", "inverse", "counts"), want, got
                        )
                        if not torch.equal(w, g)
                    ]
                    detail = f"{got[0].numel()} of {want[0].numel()} groups; wrong: {', '.join(wrong)}"
                    if "values" in wrong and got[0].shape == want[0].shape:
                        low = want[0].to(torch.int32).to(torch.int64)
                        detail += f"; values = int32(x): {torch.equal(got[0], low)}"
                _report(f"{dtype} from {_mag(start)} n={n}", ok, detail)


def _unique_by_sort(x):
    """The candidate exact route: a stable sort, compares, a scan, a scatter."""
    values, order = torch.sort(x, stable=True)
    starts = torch.ones(x.numel(), dtype=torch.bool, device=x.device)
    starts[1:] = values[1:] != values[:-1]
    group = torch.cumsum(starts, 0, dtype=torch.int32).to(torch.int64) - 1
    first = torch.nonzero(starts).flatten()
    unique = values[first]
    inverse = torch.empty_like(group).scatter_(0, order, group)
    ends = torch.cat((first[1:], first.new_tensor([x.numel()])))
    return unique, inverse, ends - first


def probe_exact():
    """The candidate replacement against the CPU's ``unique``, everywhere."""
    _say("exact: sort + compare + cumsum + scatter against the CPU's unique")
    for dtype, starts in (
        (torch.int32, (0, 2**24, 2**25, 2**30)),
        (torch.int64, (0, 2**24, 2**25, 2**40, 2**62)),
    ):
        for start in starts:
            for n in (2, 4096, 8192, 2**20, 2**22):
                x = _with_repeats(start, n, dtype)
                want = torch.unique(
                    x, sorted=True, return_inverse=True, return_counts=True
                )
                got = _unique_by_sort(_dev(x))
                ok = all(torch.equal(w, g.cpu()) for w, g in zip(want, got))
                _report(f"{dtype} from {_mag(start)} n={n}", ok)


def probe_blocks():
    """The building blocks past 2**24 in their OUTPUT, not only their input.

    A grouping's own products -- group ids, sort positions, nonzero indices,
    running counts -- pass 2**24 once a stream holds that many rows, so each op
    is asked here over 2**25 rows: a scan, a nonzero, a stable argsort's
    indices, a scatter and an indexed store of values past 2**24, and the
    counts-only ``repeat_interleave``.
    """
    n = 2**25 + 4099
    _say(f"blocks: ops whose outputs pass 2**24, over {n} rows")
    g = torch.Generator().manual_seed(9)
    ones = torch.ones(n, dtype=torch.bool)
    for dtype in (torch.int32, torch.int64):
        want = torch.cumsum(ones, 0, dtype=dtype)
        got = torch.cumsum(_dev(ones), 0, dtype=dtype).cpu()
        bad = int((got != want).sum())
        _report(
            f"cumsum(bool, dtype={dtype}) to {n}",
            bad == 0,
            f"{bad} wrong" if bad else "",
        )
        del want, got
    mask = torch.rand(n, generator=g) < 0.5
    want = torch.nonzero(mask).flatten()
    got = torch.nonzero(_dev(mask)).flatten().cpu()
    _report(
        f"nonzero over {n} bools",
        torch.equal(got, want),
        "" if torch.equal(got, want) else f"{got.numel()} of {want.numel()}",
    )
    del mask, want, got
    perm = torch.randperm(n, generator=g)
    for dtype in (torch.int32, torch.int64):
        keys = (perm % 1000).to(dtype)
        want = torch.argsort(keys, stable=True)
        got = torch.argsort(_dev(keys), stable=True).cpu()
        bad = int((got != want).sum())
        _report(
            f"argsort stable {dtype} indices to {n}",
            bad == 0,
            f"{bad} wrong" if bad else "",
        )
        del keys, want, got
    values = (2**24 + torch.arange(n, dtype=torch.int64)).flip(0)
    for dtype in (torch.int32, torch.int64):
        src = values.to(dtype)
        want = torch.empty_like(src).scatter_(0, perm, src)
        got = (
            torch.empty(n, dtype=dtype, device=DEVICE)
            .scatter_(0, _dev(perm), _dev(src))
            .cpu()
        )
        bad = int((got != want).sum())
        _report(
            f"scatter_ of {dtype} values past 2**24",
            bad == 0,
            f"{bad} wrong" if bad else "",
        )
        want = torch.empty_like(src)
        want[perm] = src
        out = torch.empty(n, dtype=dtype, device=DEVICE)
        out[_dev(perm)] = _dev(src)
        bad = int((out.cpu() != want).sum())
        _report(
            f"indexed store v[i] = x of {dtype} past 2**24",
            bad == 0,
            f"{bad} wrong" if bad else "",
        )
        del src, want, got, out
    counts = torch.randint(0, 3, (n // 2,), generator=g)
    want = torch.repeat_interleave(counts)
    got = torch.repeat_interleave(_dev(counts)).cpu()
    ok = torch.equal(got, want)
    if ok or got.shape != want.shape:
        detail = "" if ok else f"{got.numel()} rows, not {want.numel()}"
    else:
        detail = f"{int((got != want).sum())} wrong"
    _report(f"repeat_interleave(counts) to {int(want[-1])}", ok, detail)


PROBES = {
    "repro": probe_repro,
    "ladder": probe_ladder,
    "sizes": probe_sizes,
    "forms": probe_forms,
    "dtypes": probe_dtypes,
    "sort": probe_sort,
    "consecutive": probe_consecutive,
    "exact": probe_exact,
    "blocks": probe_blocks,
}


def main(names=()):
    """Run the named sections, or all of them.

    ``--cpu`` runs the script's own plumbing against the CPU, where every case
    must agree.
    """
    global DEVICE
    names = list(names)
    print(f"torch {torch.__version__}", flush=True)
    if "--cpu" in names:
        names.remove("--cpu")
        DEVICE = torch.device("cpu")
    elif not torch.backends.mps.is_available():
        print("no MPS on this machine -- nothing to probe")
        return 0
    unknown = set(names) - set(PROBES)
    if unknown:
        print(f"no such section: {', '.join(sorted(unknown))}")
        return 2
    for name in names or PROBES:
        try:
            PROBES[name]()
        except Exception as exc:  # noqa: BLE001 -- keep the rest of the table
            _report(f"section {name} raised", False, f"{type(exc).__name__}: {exc}")
    print(
        f"\n{len(_FAILS)} case(s) disagree with the CPU"
        if _FAILS
        else "\nevery case agrees with the CPU"
    )
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
