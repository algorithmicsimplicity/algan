"""Diagnostics-only counters for the sheet resolve's continuation births.

This is the host half of the spawn-counting build described in the ``rs_alloc``
diagnostics block (``wavefront_kernels_taichi``): a profiling script calls
:func:`enable` before rendering, ``shade_sparse_raster_coverage`` passes the
counting gate to ``sheet_resolve_shade`` and drains the per-tile counters into
here after each launch, and :func:`report` prints the classification table.

Nothing reads this module on a normal render. The counting gate is a
``ti.template()``, so a counting launch compiles a *different kernel variant*
from the shipping one; none of its wall time is a shipping number, and with
the gate off the atomic adds compile out entirely (the only residue is the
widened, zeroed and unread ``rs_alloc`` tail).

Two instruments live here:

* **Birth classification** — every bounce-0 continuation the sheet resolve
  emits is counted by kind (prefilter glossy row / pool reflection /
  refraction / primary that became its own bounce), bucketed by the weight it
  carries and by how much of its pixel was still unclaimed at birth.
* **Tail-ray recorder** — once the drain loop's active set gets small, the
  surviving rays' full state rows are stashed each iteration so a stuck ray
  (one that rides every iteration to the loop cap) can be diagnosed offline
  from its trajectory instead of inferred from launch counts.
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing.wavefront_kernels_taichi import (
    ALLOC_SC0,
    ALLOC_SC_N,
    SC_M_GLOSS_CLAIMS,
    SC_M_PIXELS,
    SC_M_RETIRE_FARCLIP,
    SC_M_RETIRE_WALK,
    SC_M_RETIRE_WEIGHT,
    SC_N_KINDS,
    SC_S_BUCKETS,
    SC_W_BUCKETS,
)

#: Kind order, matching the kernel's histogram indexing.
KINDS = ("prefilter_row", "reflection_pool", "refraction", "inplace_bounce")
KIND_PREFILTER = 0
KIND_REFLECTION = 1
KIND_REFRACTION = 2
KIND_INPLACE = 3

#: Weight-bucket edges: bucket i holds wt in [edge[i], edge[i+1]); the last
#: bucket is unbounded above. Edges are MIN_WEIGHT (1e-3) doubling, so a
#: bucket index is roughly log2 of the weight.
_W_EDGES = tuple(1e-3 * (2.0 ** k) for k in range(SC_W_BUCKETS - 1)) + (float("inf"),)

#: At-birth visibility buckets: how much of the pixel's samples were still
#: unclaimed when the continuation was born.
_S_EDGES = (0.05, 0.5, 0.95)

_MISC_NAMES = (
    "pixels_walked",
    "retired_primaries",
    "retired_weight",
    "retired_farclip",
    "gloss_claims",
)

_totals: torch.Tensor | None = None
_device = None

# Tail-recorder state: one entry per recorded drain iteration.
tail_records: list[dict] = []
_TAIL_MAX_ACTIVE = 256


def enabled() -> bool:
    """Whether the counting build is active (a profiling script opted in)."""
    return _totals is not None


def enable(device="cuda"):
    """Zero the totals and start collecting. Only for diagnostic scripts."""
    global _totals, _device
    _totals = torch.zeros(ALLOC_SC_N, dtype=torch.long)
    _device = device
    tail_records.clear()


def disable():
    global _totals, _device
    _totals = None
    _device = None
    tail_records.clear()


def counting_arg(device=None) -> int:
    """The value passed for the resolve kernel's ``counting`` template."""
    return 1 if _totals is not None else 0


@torch.no_grad()
def accumulate_after_resolve(rs_alloc):
    """Drain one tile's counter words into the totals and zero them.

    Called after each shading resolve launch (modes 0 and 2; mode 1 spawns
    nothing). This syncs the device -- which is exactly why it must never be
    reached outside a counting build.
    """
    global _totals
    if _totals is None:
        return
    block = rs_alloc[ALLOC_SC0 : ALLOC_SC0 + ALLOC_SC_N].detach().to("cpu")
    _totals = _totals + block.to(torch.long)
    rs_alloc[ALLOC_SC0 : ALLOC_SC0 + ALLOC_SC_N].zero_()


@torch.no_grad()
def record_tail_state(active, state, rs_pix, iteration):
    """Stash the surviving rays' state rows once the tail gets small.

    Called by the sparse drain loop after each compaction while a counting
    build is active. Records only while the active set is small enough that
    per-ray trajectories are readable.
    """
    if _totals is None:
        return
    na = int(active.numel())
    if na == 0 or na > _TAIL_MAX_ACTIVE:
        return
    idx = active.to(torch.long)
    rs_ro, rs_rd, rs_acc, rs_sca, rs_int = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
    )
    tail_records.append(
        {
            "iteration": iteration,
            "rays": na,
            "ro": rs_ro[idx].detach().to("cpu").clone(),
            "rd": rs_rd[idx].detach().to("cpu").clone(),
            "sca": rs_sca[idx].detach().to("cpu").clone(),
            "int": rs_int[idx].detach().to("cpu").clone(),
            "pix": rs_pix[idx].detach().to("cpu").clone(),
        }
    )


def _bucket_of(w):
    for i, edge in enumerate(_W_EDGES):
        if w <= edge:
            return i
    return len(_W_EDGES) - 1


def report(title="Sheet-resolve continuation births"):
    """Format the accumulated table. Returns the text; never raises."""
    if _totals is None:
        return f"{title}: counting build not enabled"
    t = _totals.tolist()

    def kind_base(kind):
        # The totals tensor is the counter block only: its index 0 is
        # ``rs_alloc[ALLOC_SC0]``, so all offsets here are block-relative.
        return kind * SC_W_BUCKETS

    def svis_base(kind):
        return SC_N_KINDS * SC_W_BUCKETS + kind * SC_S_BUCKETS

    lines = [title]
    header = "kind              total      " + " ".join(f"w>{_W_EDGES[i]:.1e}" for i in range(SC_W_BUCKETS))
    lines.append(header)
    grand = 0
    for k, name in enumerate(KINDS):
        row = t[kind_base(k) : kind_base(k) + SC_W_BUCKETS]
        total = sum(row)
        grand += total
        cells = " ".join(f"{v:>8}" for v in row)
        lines.append(f"{name:<18}{total:>9,}  {cells}")
        srow = t[svis_base(k) : svis_base(k) + SC_S_BUCKETS]
        if total:
            shares = [f"{v / total:.3f}" for v in srow]
            lines.append(
                f"    svis at birth <{_S_EDGES[0]} / <{_S_EDGES[1]} / "
                f"<{_S_EDGES[2]} / >=: {' / '.join(shares)}"
            )
    misc = {
        name: t[SC_M_PIXELS - ALLOC_SC0 + i]
        for i, name in enumerate(_MISC_NAMES)
    }
    lines.append(f"continuations born: {grand:,}")
    lines.append(
        "misc: " + ", ".join(f"{k}={v:,}" for k, v in misc.items())
    )
    return "\n".join(lines)


def tail_summary(max_print=8):
    """Summarise recorded tail iterations: sizes plus the first rays' motion."""
    if not tail_records:
        return "no tail records"
    lines = [f"tail iterations recorded: {len(tail_records)}"]
    counts = ", ".join(f"it{r['iteration']}:{r['rays']}" for r in tail_records[:12])
    lines.append("active counts: " + counts)
    first = tail_records[0]
    n = min(max_print, first["rays"])
    for i in range(n):
        ro = first["ro"][i].tolist()
        rd = first["rd"][i].tolist()
        w = max(first["sca"][i][0], first["sca"][i][5], first["sca"][i][6])
        bl = int(first["int"][i][0])
        pr = int(first["int"][i][1])
        pix = int(first["pix"][i])
        lines.append(
            f"ray {i}: pix={pix} bounces_left={bl} processed={pr} w={w:.3g} "
            f"ro=({ro[0]:.4f},{ro[1]:.4f},{ro[2]:.4f}) "
            f"rd=({rd[0]:.4f},{rd[1]:.4f},{rd[2]:.4f})"
        )
    return "\n".join(lines)
