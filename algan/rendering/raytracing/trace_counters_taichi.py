"""Ray and node-visit counters for the software BVH traversal.

A diagnostic, off by default and compiled out entirely when off: every
counting site in the traversal sits behind ``ti.static(counting())``, so the
kernels a normal render runs are the kernels it ran before these existed.
``ALGAN_TRACE_COUNTERS`` (read at import) turns them on:

* ``1`` -- count.
* ``verify`` -- count, and behind every opaque-first shadow answer
  (shadow mode 5, ``shadow_dispatch``) also run the ordered reference march and
  count the rays whose occlusion differs from it in any bit. The image keeps
  the opaque-first answer, so it is the image the plain ``1`` arm renders.

The counts are what turn a traversal estimate into arithmetic: how many shadow
rays a render traces, how each was answered, and how many sibling-group tests
(one 64-byte node fetch each under the default f16 blocks) and primitive tests
every class of walk spent. A walk is one traversal query: one closest-hit
step of the shadow march, one any-hit pass, one k-buffer gather.

Storage is one ``i64`` Taichi field, which is why this is a diagnostic and not
a statistic the renderer keeps: Algan otherwise holds no field (see
``taichi_runtime.ensure_taichi_for_render``), and a field read after a re-init
of the program segfaults. :func:`on_program_start` therefore makes a fresh one
every time ``taichi_runtime.start_program`` brings a program up, before any
kernel of that program compiles, and nothing ever reads an older one. Each
counter is spread over :data:`LANES` slots indexed by the thread, so a GPU
render does not serialize on a handful of addresses; walks accumulate in
registers and add once at their end. Counting kernels reference the field, so
their source keys are poisoned (they pay the full frontend, with one warning
each) -- another reason to keep a counting run in a process of its own.

CPU and CUDA only: the counters use 64-bit atomics.
"""

from algan.environment import env_str
from algan.taichi_compat import ti

_MODE = env_str("ALGAN_TRACE_COUNTERS", "").strip().lower()
if _MODE in ("", "0", "off", "false", "no"):
    TRACE_COUNTING = 0
elif _MODE == "verify":
    TRACE_COUNTING = 2
else:
    TRACE_COUNTING = 1

#: Slots each counter is spread over (a power of two).
LANES = 64

# Ray-level shadow counters.
SHADOW_RAYS = 0
SHADOW_OF_BLOCKED = 1
SHADOW_OF_CLEAR = 2
SHADOW_OF_MARCH = 3
SHADOW_OF_MISMATCH = 4
SHADOW_MARCH_CAPPED = 5

# Walk classes. Shadow walks are the ones that reject non-casting leaves
# (``nocast``); "trace" walks are every other ray: camera, bounce, aux passes.
WALK_SHADOW_NEAREST = 0
WALK_SHADOW_ANYHIT = 1
WALK_SHADOW_GATHER = 2
WALK_TRACE_NEAREST = 3
WALK_TRACE_GATHER = 4
_WALK_BASE = 8
_WALK_NAMES = ("shadow_nearest", "shadow_anyhit", "shadow_gather",
               "trace_nearest", "trace_gather")
# Per walk class: walks, sibling-group (node) tests, triangle tests, circuit
# tests.
WALKS = 0
NODES = 1
TRI_TESTS = 2
BEZ_TESTS = 3
NUM_SLOTS = _WALK_BASE + 4 * len(_WALK_NAMES)

_RAY_NAMES = {
    SHADOW_RAYS: "shadow_rays",
    SHADOW_OF_BLOCKED: "shadow_opaque_first_blocked",
    SHADOW_OF_CLEAR: "shadow_opaque_first_clear",
    SHADOW_OF_MARCH: "shadow_opaque_first_march",
    SHADOW_OF_MISMATCH: "shadow_opaque_first_mismatch",
    SHADOW_MARCH_CAPPED: "shadow_march_capped",
}
_STAT_NAMES = ("walks", "nodes", "tri_tests", "bez_tests")

_COUNTS = None
_BASELINE = None


def counting():
    """Whether the counting sites compile in (read under ``ti.static``)."""
    return TRACE_COUNTING != 0


def verifying():
    """Whether opaque-first shadow answers are re-derived by the march."""
    return TRACE_COUNTING == 2


def _field():
    return _COUNTS


def walk_slot(walk_class, stat):
    """Slot of ``stat`` (``WALKS``/``NODES``/...) for one walk class."""
    return _WALK_BASE + 4 * walk_class + stat


@ti.func
def bump(slot: ti.template(), n):
    """Add ``n`` to counter ``slot``; nothing at all unless counting."""
    if ti.static(counting()):
        lane = ti.lang.expr.Expr(ti.global_thread_idx()) & (LANES - 1)
        ti.atomic_add(ti.static(_field())[slot * LANES + lane],
                      ti.cast(n, ti.i64))


@ti.func
def tick(n):
    """``n + 1`` while counting and ``n`` otherwise: a walk's register tally."""
    out = n
    if ti.static(counting()):
        out = n + 1
    return out


@ti.func
def count_query(walk_class: ti.template()):
    """One traversal query of ``walk_class`` (both trees count as one)."""
    if ti.static(counting()):
        bump(ti.static(walk_slot(walk_class, WALKS)), 1)


@ti.func
def count_visits(walk_class: ti.template(), nodes, tri_tests, bez_tests):
    """What one tree walk of ``walk_class`` spent, added once at its end."""
    if ti.static(counting()):
        bump(ti.static(walk_slot(walk_class, NODES)), nodes)
        bump(ti.static(walk_slot(walk_class, TRI_TESTS)), tri_tests)
        bump(ti.static(walk_slot(walk_class, BEZ_TESTS)), bez_tests)


def on_program_start():
    """Give a freshly initialized program its own counter field.

    Called by ``taichi_runtime.start_program`` right after ``ti.init``; a no-op
    unless counting. The previous program's field is dropped, never read.
    """
    global _COUNTS, _BASELINE
    _COUNTS = None
    _BASELINE = None
    if counting():
        _COUNTS = ti.field(ti.i64, shape=(NUM_SLOTS * LANES,))


def _raw():
    import numpy as np

    if _COUNTS is None:
        return np.zeros(NUM_SLOTS, dtype=np.int64)
    return _COUNTS.to_numpy().reshape(NUM_SLOTS, LANES).sum(axis=1)


def reset():
    """Start a new reading: :func:`read` counts from here."""
    global _BASELINE
    _BASELINE = _raw()


def read():
    """Counts since :func:`reset` (or since the program started), by name.

    Synchronizes the device. Empty when counting is off.
    """
    if not counting():
        return {}
    raw = _raw()
    if _BASELINE is not None:
        raw = raw - _BASELINE
    out = {name: int(raw[slot]) for slot, name in _RAY_NAMES.items()}
    for w, wname in enumerate(_WALK_NAMES):
        for s, sname in enumerate(_STAT_NAMES):
            out[f"{wname}_{sname}"] = int(raw[walk_slot(w, s)])
    return out


def format_counts(counts):
    """A readable table of :func:`read`'s result, with per-walk averages."""
    if not counts:
        return "trace counters are off (ALGAN_TRACE_COUNTERS unset)"
    lines = []
    for name in _RAY_NAMES.values():
        lines.append(f"  {name:32s} {counts.get(name, 0):>16,d}")
    lines.append(f"  {'walk class':16s} {'walks':>14s} {'nodes':>16s} "
                 f"{'nodes/walk':>11s} {'tri tests':>15s} {'bez tests':>13s}")
    for wname in _WALK_NAMES:
        walks = counts.get(f"{wname}_walks", 0)
        nodes = counts.get(f"{wname}_nodes", 0)
        per = nodes / walks if walks else 0.0
        lines.append(
            f"  {wname:16s} {walks:>14,d} {nodes:>16,d} {per:>11.2f} "
            f"{counts.get(f'{wname}_tri_tests', 0):>15,d} "
            f"{counts.get(f'{wname}_bez_tests', 0):>13,d}")
    return "\n".join(lines)
