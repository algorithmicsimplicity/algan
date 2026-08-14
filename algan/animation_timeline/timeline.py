from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch

from algan.utils.tensor_utils import cast_to_tensor

#: Torch -> numpy dtypes for :func:`_sparsely_written_zeros`. Only the dtypes an
#: attribute buffer actually uses; anything else falls back to ``torch.zeros``.
_NUMPY_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}

#: Name of the special kwarg that animated functions receive the per-frame
#: elapsed time through (see :meth:`AnimationTimeline.set_state_to_times`).
TIME_PARAMETER_NAME = "time_elapsed"


#: Sentinel timestamp for "not yet spawned/despawned".
def _never():
    return -1


#: Timestamp used as the end of an updater that was never removed.
UPDATER_FOREVER = 1e12

#: Global structure version, bumped whenever the mob hierarchy or any
#: attribute timeline's row allocation changes. Version-checked caches of
#: concatenated descendant row indexes (see
#: :meth:`~algan.animatable_base.animatable.Animatable._get_attr_inds`) are
#: invalidated by comparing against it, so they never have to be cleared
#: explicitly.
STRUCTURE_VERSION = [0]

#: Global hierarchy version, bumped only when the mob hierarchy (any mob's
#: ``children``) changes -- NOT on attribute-buffer row allocation. The
#: descendant-*list* cache (:meth:`~algan.mobs.mob.Mob.get_descendants`)
#: depends only on the hierarchy, so keying it on this version lets it survive
#: the row allocations that happen constantly during construction (every mob's
#: first attribute write allocates rows and bumps STRUCTURE_VERSION). Every
#: hierarchy mutation bumps both versions (via :func:`bump_hierarchy_version`),
#: so the row-index caches stay correct too.
HIERARCHY_VERSION = [0]

#: Global spawn version, bumped whenever any mob's spawn time is (re)assigned
#: (:class:`Lifespan.start`). Together with :data:`HIERARCHY_VERSION` it keys
#: the subtree-spawn cache of
#: :meth:`~algan.animatable_base.animatable.Animatable.is_spawned_in_subtree`,
#: whose answer can only change when the hierarchy changes or something spawns.
SPAWN_VERSION = [0]

#: Global timing version, bumped whenever a TimelineSpan is rescaled or a
#: lifespan endpoint is reassigned. Opacity timelines use it to retain their
#: materialized spawn/despawn tensor until one of the live TimelineEvent
#: values it contains can actually have changed.
TIMING_VERSION = [0]


def bump_structure_version():
    STRUCTURE_VERSION[0] += 1


def bump_spawn_version():
    SPAWN_VERSION[0] += 1


def bump_timing_version():
    TIMING_VERSION[0] += 1


def bump_hierarchy_version():
    """Bump both the hierarchy and structure versions. Call this (instead of
    :func:`bump_structure_version`) for any change to a mob's ``children``: it
    invalidates both the descendant-list cache and the descendant-row-index
    caches, since a hierarchy change alters both.
    """
    HIERARCHY_VERSION[0] += 1
    STRUCTURE_VERSION[0] += 1


_OPT_DISABLED = None


def _opt_disabled(name):
    """Bisect aid: ALGAN_OPT_DISABLE=fastpath,ranges,desccache,windows,torchquery,
    timeslice,lazyzeros disables individual animation-prep optimizations (read
    once, first use). ``benchmarks/_prep_timeslice_ab.py`` and its s05 companion
    A/B the last two through this.
    """
    global _OPT_DISABLED
    if _OPT_DISABLED is None:
        import os

        _OPT_DISABLED = frozenset(os.environ.get("ALGAN_OPT_DISABLE", "").split(","))
    return name in _OPT_DISABLED


#: Event counts at or below which a plain scan beats the interval index, whose
#: tensor build is dominated by fixed per-op overhead at small sizes.
_SMALL_EVENT_SCAN = 64


def _event_interval_index(starts, ends):
    """Index ``[start, end)`` events so a window query need not scan them all.

    Both event lookups in :class:`FunctionTimeline` used to test every event the
    scene ever recorded against every queried time. That is O(scene) per batch
    and there are O(scene) batches, so O(n^2) over a render -- the same shape
    ``_by_caller`` exists to avoid, and the same one that makes the cost
    invisible on a short scene and dominant on a long one.

    Two bounds, both implied by the activity test rather than heuristics:

    * **From above**, via starts sorted ascending: an event that starts after
      the last queried time cannot contain any of them.
    * **From below**, via the running maximum end over that same order: it is
      non-decreasing, so at any position where it is still at or below the
      first queried time, *every* event up to there has already finished.

    So the survivors are a superset of the answer and the exact test still
    decides -- which keeps the result identical for unsorted ``times`` too,
    since only the min and max are used here.

    Kept in float64: the bounds are float32 and widening is exact, so neither
    comparison can round a boundary the wrong way and drop a live event.
    """
    order = torch.argsort(starts, stable=True)
    return (
        order,
        starts[order].double(),
        torch.cummax(ends[order], 0).values.double(),
    )


def _events_overlapping(index, times):
    """Event indices that may be active at ``times``, **ascending**.

    The ascending sort is load-bearing, not tidiness: replay re-executes
    functions in the order they were recorded, and this index works in
    start-sorted order.
    """
    order, sorted_starts, running_max_end = index
    hi = int(torch.searchsorted(sorted_starts, times.max().double(), right=True))
    lo = int(torch.searchsorted(running_max_end, times.min().double(), right=True))
    if lo >= hi:
        return order[:0]
    return order[lo:hi]


def _contiguous_time_selector(inds):
    """``inds`` as a ``slice`` when it selects one contiguous run of frames.

    :meth:`AttributeTimeline.get` / :meth:`~AttributeTimeline.modify` read the
    frame axis through ``active_time_inds``, and both already branch on it: a
    ``slice`` reads a view and writes a slice-assign, a tensor pays an
    advanced-index gather and a scatter. Outside replay the selector is
    ``slice(None)``; the tensor path exists only while
    :meth:`AnimationTimeline.set_state_to_times` is replaying recorded
    functions, which is where the great majority of the calls are.

    A replay window is the interval ``[start, replay_end)`` over the batch's
    ascending frame times, so the selected indices are contiguous. That is
    *tested* here rather than assumed, so a public caller passing unsorted
    ``times`` -- or an updater's clone-grouped frame set, which is genuinely
    scattered -- still gets the tensor path and the same answer.

    ``inds`` must be **ascending**, which is what makes the span test exact: a
    permutation of a contiguous run (``[0, 2, 1, 3]``) has the same first, last
    and length as the run itself and would convert wrongly. Both call sites
    satisfy it -- ``Tensor.nonzero`` returns indices in ascending order, and the
    updater's per-signature groups are appended while iterating those same
    ascending indices. Anything else plumbed in here needs checking against
    that, not against the shape.
    """
    if inds.shape[0] == 0:
        return inds
    first = int(inds[0])
    last = int(inds[-1])
    if last - first + 1 == inds.shape[0]:
        return slice(first, last + 1)
    return inds


class RowRanges:
    """A set of attribute-buffer rows stored as ordered [begin, end) runs.

    Every mob's rows are allocated as one contiguous block
    (:meth:`AttributeTimeline.add`), so the union of a subtree's rows is
    usually a handful of runs. Storing runs instead of one index per row keeps
    the per-mob descendant caches O(runs) rather than O(rows), and a
    single-run set lets the attribute buffers be read and written through
    plain slices instead of index gathers/scatters (see
    :meth:`AttributeTimeline.get` / :meth:`AttributeTimeline.modify`).

    ``pairs`` may be None for an uncompressible index set, in which case the
    materialized ``tensor()`` is the only representation. Adjacent runs are
    merged by the builders; duplicate or out-of-order runs are kept verbatim
    so the materialized tensor always equals the uncompressed concatenation.
    """

    __slots__ = ("pairs", "numel", "_tensor")

    def __init__(self, pairs, tensor=None):
        self.pairs = pairs
        self.numel = (
            sum(e - b for b, e in pairs) if pairs is not None else tensor.numel()
        )
        self._tensor = tensor

    def tensor(self):
        if self._tensor is None:
            if len(self.pairs) == 1:
                b, e = self.pairs[0]
                self._tensor = torch.arange(b, e)
            else:
                self._tensor = torch.cat([torch.arange(b, e) for b, e in self.pairs])
        return self._tensor

    @staticmethod
    def from_contiguous_blocks(inds_list):
        """Build from per-mob index tensors, each of which is a contiguous
        arange (how ``AttributeTimeline.add`` allocates). Returns None when a
        block is not contiguous (caller falls back to plain concatenation).
        """
        if len(inds_list) == 1:
            inds = inds_list[0]
            n = inds.numel()
            if n == 0:
                return RowRanges([])
            b = int(inds[0])
            e = int(inds[-1]) + 1
            if e - b != n:
                return None
            return RowRanges([(b, e)])
        pairs = []
        inds_list = sorted(inds_list, key=lambda x: x[0])
        for inds in inds_list:
            n = inds.numel()
            if n == 0:
                continue
            b = int(inds[0])
            e = int(inds[-1]) + 1
            if e - b != n:
                return None
            if pairs and pairs[-1][1] == b:
                pairs[-1] = (pairs[-1][0], e)
            else:
                pairs.append((b, e))
        return RowRanges(pairs)

    @staticmethod
    def from_runs(runs):
        """Build from a list of already-known ``(begin, end)`` integer runs
        (e.g. each mob's cached single-run range). Coalesces adjacent runs,
        equivalent to :meth:`from_contiguous_blocks` but without any per-row
        tensor indexing/conversion. ``runs`` must contain no empty runs.
        """
        if len(runs) == 1:
            return RowRanges([runs[0]])
        pairs = []
        for b, e in sorted(runs):
            if pairs and pairs[-1][1] == b:
                pairs[-1] = (pairs[-1][0], e)
            else:
                pairs.append((b, e))
        return RowRanges(pairs)


class Lifespan:
    """The [spawn, despawn) interval of one mob on its Scene timeline.

    ``start`` and ``end`` are zero-argument callables returning the spawn /
    despawn timestamp in seconds, or -1 when the mob has not (yet) spawned /
    despawned. They are stored as callables (:class:`TimelineEvent` s) because
    animation contexts rescale timestamps retroactively, so the final value is
    only known at render time.

    Lifespans are owned by the :class:`AnimationTimeline`
    (:meth:`AnimationTimeline.get_lifespan`), keyed by mob id.
    """

    __slots__ = ("_start", "_end")

    def __init__(self):
        self._start = _never
        self._end = _never

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        # Spawning (and un-spawning, see Mob.refresh_history) invalidates every
        # cached "is anything in this subtree spawned?" answer, so every
        # assignment -- not just the ones going through
        # TimelineManager.register_spawn -- bumps the global spawn version.
        self._start = value
        bump_spawn_version()
        bump_timing_version()

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value
        bump_timing_version()


class EditRecord:
    """One recorded modification of an attribute's buffer rows.

    Stores the rows' *pre-modification* values together with the end time of
    the animation that made the modification: materialization reconstructs a
    row's base state at time ``t`` as the pre-value of the earliest-executed
    edit still unfinished at ``t``, then re-applies the functions active at
    ``t`` on top of it, in execution order.

    ``seq`` is the global execution (recording) order across all attributes,
    and ``event`` is the :class:`FunctionApplicationEvent` whose function made
    the modification (None for edits recorded outside animated functions).
    ``replay_end`` is the edit's *effective* end used for base selection: its
    own end time, extended over the replay windows of every earlier-executed
    edit that overlaps it in time on shared rows (see
    :meth:`AnimationTimeline._resolve_replay_windows`). It is resolved to a
    float at render time, once context rescaling is final.
    """

    __slots__ = ("indexes", "values", "time", "seq", "event", "replay_end")

    def __init__(self, indexes, values, time, seq=0, event=None):
        self.indexes = indexes
        self.values = values
        self.time = time
        self.seq = seq
        self.event = event
        self.replay_end = None


def _replay_window_end(f):
    """End of a function application's replay window: its context end time,
    extended to its edits' resolved ``replay_end`` when they overlap
    earlier-executed edits (never shrunk below the context end).
    """
    end = f.time.end
    if f.replay_end is None:
        return end
    return max(f.replay_end, end)


class EditQueryIndex:
    """Search structure over one attribute's edit log, on ``times``' device.

    ``head``/``sorted_edit_ids``/``sorted_values`` are the CSR form of the edit
    table: the edits touching row ``j`` occupy ``[head[j], head[j+1])``, in
    execution order (and therefore in non-decreasing ``edit_timestamps`` order,
    which :meth:`AttributeTimeline.prepare_for_queries` guarantees).
    Materializing a row's state at time ``t`` is an upper-bound search for
    ``t`` inside that row's segment.

    ``keys`` linearizes those per-row searches into a single globally sorted
    array so one :func:`torch.searchsorted` answers all of them:
    ``keys[m] = row(m) * n_ranks + rank(timestamp(m))``, where ``rank`` indexes
    the sorted unique timestamps. Integer ranks (rather than the timestamps
    themselves) keep the composite key exact, and the key is sorted because
    ``row`` is non-decreasing across the table and the rank is non-decreasing
    within every row.

    The edit log is immutable while frames are rendered, so this is built once
    per attribute per render job and cached (:meth:`AttributeTimeline._prepared_queries`).
    """

    __slots__ = (
        "head",
        "sorted_edit_ids",
        "edit_timestamps",
        "sorted_values",
        "unique_timestamps",
        "keys",
    )

    def __init__(
        self,
        head,
        sorted_edit_ids,
        edit_timestamps,
        sorted_values,
        unique_timestamps,
        keys,
    ):
        self.head = head
        self.sorted_edit_ids = sorted_edit_ids
        self.edit_timestamps = edit_timestamps
        self.sorted_values = sorted_values
        self.unique_timestamps = unique_timestamps
        self.keys = keys


class _EndpointLayout:
    """Mutable, over-allocated row ownership table for lifespan endpoints."""

    __slots__ = ("lifespans", "rows", "owners", "blocks", "used")

    def __init__(self, lifespans, rows, owners, blocks, used):
        self.lifespans = lifespans
        self.rows = rows
        self.owners = owners
        self.blocks = blocks
        self.used = used


def _prepare_array_state_queries(times, N, edits):
    """Build the CSR representation of an attribute's edits, plus the sorted
    composite search key described on :class:`EditQueryIndex`.

    The edit log is immutable while frames are rendered.  Keeping this work
    separate from the actual time query lets :class:`AttributeTimeline` cache
    the concatenation, stable sort and row-boundary construction across frame
    batches.
    """
    device = times.device
    edit_timestamps = torch.tensor(
        [edit["timestamp"] for edit in edits], dtype=times.dtype, device=device
    )
    edit_sizes = torch.tensor(
        [edit["indexes"].shape[0] for edit in edits], dtype=torch.int64, device=device
    )
    flat_indices = torch.cat([edit["indexes"].to(device) for edit in edits])
    flat_values = torch.cat([edit["values"].to(device) for edit in edits])
    flat_edit_ids = torch.repeat_interleave(edit_sizes).to(torch.int32)
    perm = torch.argsort(flat_indices, stable=True)
    sorted_indices = flat_indices[perm]
    sorted_edit_ids = flat_edit_ids[perm]
    sorted_values = flat_values[perm]
    grid = torch.arange(N + 1, dtype=torch.int64, device=device)
    head = torch.searchsorted(sorted_indices, grid)

    unique_timestamps = torch.unique(edit_timestamps)
    # rank(timestamp) is exact: every edit timestamp is present in the sorted
    # unique values, so searchsorted returns its index there.
    edit_ranks = torch.searchsorted(unique_timestamps, edit_timestamps)
    row_of = torch.repeat_interleave(head[1:] - head[:-1])
    keys = row_of.mul_(unique_timestamps.shape[0]).add_(
        edit_ranks[sorted_edit_ids.to(torch.int64)]
    )
    return EditQueryIndex(
        head,
        sorted_edit_ids,
        edit_timestamps,
        sorted_values,
        unique_timestamps,
        keys,
    )


#: Cap on the temporary index buffer one :func:`_query_row_states` chunk builds
#: (bytes). Frame windows are split so the composite-key gather stays cache
#: friendly and its peak is bounded independently of the window size.
_QUERY_CHUNK_BYTES = 32 << 20


def _sparsely_written_zeros(shape, dtype, device):
    """Zeros for a buffer only a minority of which will be written.

    ``torch.zeros`` is ``empty`` + an explicit ``memset`` over the whole
    allocation. The rematerialization buffer is ``[T, N, D]`` where ``N`` is
    every row the scene ever allocated, and a render window writes ~30% of them
    -- so the memset touches (and resident-faults) three times the pages the
    batch will ever look at. ``np.zeros`` is ``calloc``, and an allocation this
    size comes straight from the OS page allocator already zeroed, so only the
    pages actually touched are ever charged. Measured on the reference scene's
    shapes (``benchmarks/_p1_zerofill_ab.py``): the fill goes from 8-77 ms to
    ~0.1 ms, and fill + scatter together 1.14-1.31x.

    Byte-identical -- both produce zeros. CPU only: on CUDA the fill is a
    device memset and numpy has nothing to offer. Falls back to ``torch.zeros``
    for any dtype numpy cannot express (``bfloat16``), so the caller never has
    to know which it got.
    """
    if device.type != "cpu" or _opt_disabled("lazyzeros"):
        return torch.zeros(shape, dtype=dtype, device=device)
    try:
        return torch.from_numpy(np.zeros(shape, dtype=_NUMPY_DTYPES[dtype]))
    except KeyError:
        return torch.zeros(shape, dtype=dtype, device=device)


def _query_row_states(times, index, rows=None):
    """Materialize ``rows`` (all rows when None) of one attribute at ``times``.

    Returns the compact ``[T, R, D]`` result; the caller places it in whatever
    layout it needs. See :class:`EditQueryIndex` for the search key.
    """
    head = index.head
    keys = index.keys
    sorted_values = index.sorted_values
    T = times.shape[0]
    N = head.shape[0] - 1
    U = keys.shape[0]
    D = sorted_values.shape[1]
    R = N if rows is None else rows.shape[0]

    if U == 0 or R == 0 or T == 0:
        return torch.zeros((T, R, D), dtype=sorted_values.dtype, device=head.device)

    if rows is None:
        ends = head[1:]
        bases = torch.arange(N, dtype=torch.int64, device=head.device)
    else:
        ends = head[rows + 1]
        bases = rows
    # Row j's keys live in [j * n_ranks, (j + 1) * n_ranks), so searching for
    # j * n_ranks + rank lands inside row j's segment (or on its end).
    bases = bases * index.unique_timestamps.shape[0]
    # Number of distinct timestamps <= t; an edit outlives t exactly when its
    # rank is >= this, which is what the composite-key search finds.
    query_ranks = torch.searchsorted(
        index.unique_timestamps, times.contiguous(), right=True
    )

    # A frame's answer depends on its time ONLY through that rank, so two
    # frames of the window with no edit boundary between them read exactly the
    # same rows of ``sorted_values``. A render window is a few seconds of a
    # scene whose edits are authored in blocks, so it usually covers far fewer
    # distinct boundaries than it has frames -- do the search and the (random
    # access) value gather once per distinct rank and expand the result back
    # over the frames, which is a contiguous per-frame copy. Bit-identical by
    # construction: equal ranks select equal ``low``.
    unique_ranks, inverse = torch.unique(query_ranks, return_inverse=True)
    ranks = query_ranks
    if int(unique_ranks.shape[0]) < T:
        ranks = unique_ranks
    else:
        inverse = None
    S = int(ranks.shape[0])
    # Every element is written by the loop below.
    out = torch.empty((S, R, D), dtype=sorted_values.dtype, device=head.device)

    chunk = max(1, min(S, _QUERY_CHUNK_BYTES // max(1, R * 8)))
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        low = torch.searchsorted(
            keys, (bases.unsqueeze(0) + ranks[start:stop].unsqueeze(1)).view(-1)
        ).view(stop - start, R)
        empty = low >= ends
        values = sorted_values[low.clamp_(max=U - 1)]
        # Rows with no edit still live at t read as zero, exactly as the
        # kernel wrote them (masked_fill, not a multiply, so non-finite
        # recorded values cannot leak in as NaN).
        values.masked_fill_(empty.unsqueeze(-1), 0.0)
        out[start:stop] = values
    if inverse is not None:
        out = out.index_select(0, inverse)
    return out


def generate_array_states(times, N, edits, *, active_rows=None, prepared=None):
    """
    Generates the state of an array given its history of edits.

    Parameters
    ----------
    times : torch.Tensor
        Shape [T], the inquiry times.
    N : int
        Length of the output vector.
    edits : list of dict
        Each dict contains 'indexes' (tensor of shape [M_i], values in
        [0, N-1]), 'values' (tensor of shape [M_i, D]) and 'timestamp'
        (float scalar). Timestamps must be non-decreasing along every row
        (i.e. among the edits containing any given index) — the per-row
        upper-bound search relies on it. prepare_for_queries guarantees this
        by passing edits in execution order with their replay-extended end
        times.
    """
    device = times.device
    T = times.shape[0]

    if len(edits) == 0:
        return torch.zeros((T, N, 1), dtype=torch.float32, device=device)

    if prepared is None:
        prepared = _prepare_array_state_queries(times, N, edits)
    D = prepared.sorted_values.shape[1]
    dtype = prepared.sorted_values.dtype

    if active_rows is not None:
        active_rows = active_rows.to(device=device, dtype=torch.int64)

    if _opt_disabled("torchquery"):
        return _generate_array_states_taichi(
            times, N, prepared, active_rows, T, D, dtype, device
        )

    if active_rows is None:
        return _query_row_states(times, prepared)
    # Keep the full global row layout for animated-function replay. Rows
    # outside this window's working set stay zero and are never consumed by
    # primitive preparation for this batch.
    out = _sparsely_written_zeros((T, N, D), dtype, device)
    if active_rows.numel():
        out.index_copy_(1, active_rows, _query_row_states(times, prepared, active_rows))
    return out


def _generate_array_states_taichi(times, N, prepared, active_rows, T, D, dtype, device):
    """Original Taichi implementation of :func:`generate_array_states`, kept as
    the A/B reference for the torch query (``ALGAN_OPT_DISABLE=torchquery``).

    Not used by default: Taichi's arch is the *render* device, so launching
    these kernels with the CPU animation tensors makes Taichi stage every
    argument -- including the whole ``[T, N, D]`` result -- through VRAM, on
    the batch-prep worker thread that is deliberately kept off the GPU. The
    import is deferred so the animation timeline no longer pulls Taichi in.
    """
    from algan.animation_timeline.utils_taichi import (
        _query_selected_state_from_edits,
        _query_state_from_edits,
    )

    if active_rows is None:
        out = torch.empty((T, N, D), dtype=dtype, device=device)
        _query_state_from_edits(
            times,
            prepared.head,
            prepared.sorted_edit_ids,
            prepared.edit_timestamps,
            prepared.sorted_values,
            out,
        )
    else:
        out = torch.zeros((T, N, D), dtype=dtype, device=device)
        if active_rows.numel():
            _query_selected_state_from_edits(
                times,
                active_rows,
                prepared.head,
                prepared.sorted_edit_ids,
                prepared.edit_timestamps,
                prepared.sorted_values,
                out,
            )
    return out


class AttributeTimeline:
    """
    A Scene-owned timeline recording every Mob row for one attribute.

    Edits (:class:`EditRecord` s) are kept in execution order; materialization
    sets each row's base state at time ``t`` to the pre-modification value of
    the row's earliest-executed edit whose (replay-extended) end is after
    ``t``, over which :meth:`AnimationTimeline.set_state_to_times` re-applies
    the functions whose replay windows cover ``t``, in execution order. This
    makes edits that overlap in time (including edits ending at the same
    time) rematerialize to the same chain of states they produced when
    recorded.
    """

    def __init__(
        self, channels, buffer_size=256, attr_name=None, record_end_points=False
    ):
        self.attr_name = attr_name
        self.record_end_points = record_end_points
        self.current_state = torch.empty(
            (1, buffer_size, channels)
        )  # latest state after all edits.
        self.active_state = (
            self.current_state
        )  # pointer to active state used to fulfil get requests.
        # Global row -> column in ``active_state``, or -1 for a row this window
        # did not materialize. ``None`` means active_state is in global row
        # order and no translation is needed (current_state, and the all-rows
        # materialization public callers get). See _set_active_row_map.
        self._active_row_map = None
        # Backing store for that map, grown in place across batches so a
        # window costs O(active rows) rather than O(every row ever allocated).
        self._row_map_buffer = None
        # Zero-copy numpy view of that buffer for scalar reads; see
        # _set_active_row_map. None when the map is not on the CPU.
        self._row_map_np = None
        # The rows currently written into the map, so the next window can clear
        # just those instead of rewriting the whole buffer.
        self._mapped_rows = None
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        self.pointer = 0
        self.edits = []
        self._is_ready_for_queries = False
        self._prepared_edit_count = 0
        self._edits_sorted = []
        self._query_cache = {}
        self.mob_id_to_inds = {}
        self.mob_id_to_ranges = {}
        self.mob_id_to_starts = {}
        self.mob_id_to_ends = {}
        self._endpoint_layout_revision = 0
        self._endpoint_layout_cache = None
        self._pending_start_endpoints = []
        self._pending_end_endpoints = []
        self._dirty_endpoint_rows = set()
        self._dirty_start_endpoints = set()
        self._dirty_end_endpoints = set()
        self._end_points = None
        self._end_points_version = None

    def set_start_point(self, mob, starts):
        replacing = mob.id in self.mob_id_to_starts
        self.mob_id_to_starts[mob.id] = starts
        if replacing:
            self._dirty_start_endpoints.add(mob.id)
            self._invalidate_endpoint_values()
        else:
            self._pending_start_endpoints.append((mob.id, starts))
            self._invalidate_endpoint_values()

    def set_end_point(self, mob, ends):
        replacing = mob.id in self.mob_id_to_ends
        self.mob_id_to_ends[mob.id] = ends
        if replacing:
            self._dirty_end_endpoints.add(mob.id)
            self._invalidate_endpoint_values()
        else:
            self._pending_end_endpoints.append((mob.id, ends))
            self._invalidate_endpoint_values()

    def invalidate_prepared_queries(self, *, retain_edit_prefix=False):
        """Drop data derived from the recorded edit log.

        The temporary ``active_state`` produced while rendering is deliberately
        not part of this cache.  Normal recording-time writes, row allocation,
        topology migration, and rollback of transient replay-window timestamps
        call this method; merely returning to ``current_state`` does not.

        Ordinary recording and row growth are append-only, so they can retain
        the already-normalized edit dictionaries. Topology migration and
        replay-window rollback mutate earlier records and request a full reset.
        """
        self._is_ready_for_queries = False
        self._query_cache.clear()
        if not retain_edit_prefix:
            self._prepared_edit_count = 0
            self._edits_sorted.clear()

    def _invalidate_endpoint_layout(self):
        self._endpoint_layout_revision += 1
        self._endpoint_layout_cache = None
        self._pending_start_endpoints.clear()
        self._pending_end_endpoints.clear()
        self._dirty_endpoint_rows.clear()
        self._dirty_start_endpoints.clear()
        self._dirty_end_endpoints.clear()
        self._end_points_version = None

    def _invalidate_endpoint_values(self):
        self._endpoint_layout_revision += 1
        self._end_points_version = None

    def get_current_values(self):
        return self.current_state[:, : self.pointer]

    def _set_active_row_map(self, active_rows):
        """Point global rows at their columns in a compact ``active_state``.

        ``active_state`` holds only the window's live rows, so every reader has
        to go through this. The map itself is the one thing that stays
        full-width -- but at 8 bytes a row against ``T * D * 4``, it is ~150x
        smaller than the buffer it replaces on the reference scene, and it is
        grown in place and rewritten only where it changed, so a batch costs
        O(active rows) rather than O(every row ever allocated).

        ``active_rows`` must be ascending (``rows_for_mob_ids`` coalesces runs
        and the ``extra_rows`` union sorts), which is what makes the map
        monotone -- and monotonicity is what lets :meth:`_compact_span` decide a
        whole contiguous range from its two endpoints.
        """
        width = self.pointer + 1
        rows = active_rows.to(dtype=torch.int64)
        buffer = self._row_map_buffer
        if (
            buffer is None
            or buffer.shape[0] < width
            or buffer.device != rows.device
        ):
            buffer = torch.full(
                (max(width, 1 if buffer is None else buffer.shape[0] * 2),),
                -1,
                dtype=torch.int64,
                device=rows.device,
            )
            self._mapped_rows = None
        if self._mapped_rows is not None and self._mapped_rows.numel():
            buffer[self._mapped_rows] = -1
        if rows.numel():
            buffer[rows] = torch.arange(
                rows.numel(), dtype=torch.int64, device=rows.device
            )
        self._row_map_buffer = buffer
        self._mapped_rows = rows
        self._active_row_map = buffer
        # A zero-copy numpy view of the same memory, purely so _compact_span
        # can read two scalars per accessor call without paying torch's
        # ~10 us-per-element scalar extraction. That cost is invisible on the
        # reference scene and dominates a suite of small scenes: it put the
        # fast suite from 130 s to 184 s before this went in. Rebuilt here
        # because a reallocation above invalidates the view; in-place writes
        # (materialize_additional_rows) stay visible through it.
        self._row_map_np = buffer.numpy() if buffer.device.type == "cpu" else None

    def _clear_active_row_map(self):
        """Return to global row order (``active_state`` indexed by row id)."""
        if self._mapped_rows is not None and self._mapped_rows.numel():
            self._row_map_buffer[self._mapped_rows] = -1
        self._mapped_rows = None
        self._active_row_map = None
        self._row_map_np = None

    def _compact_span(self, b, e):
        """Columns for global rows ``[b, e)``, or ``None`` if not one run.

        Endpoints are enough: the map is the rank of each row within the
        ascending ``active_rows``, so if row ``b`` is live and row ``e - 1``
        sits exactly ``e - 1 - b`` columns later, every row between them is
        live and consecutive too.
        """
        row_map = self._active_row_map
        if row_map is None:
            return b, e
        if e <= b:
            return 0, 0
        if e > row_map.shape[0]:
            # Rows allocated after this window was materialized. The full-width
            # buffer answered these with a silently truncated slice; the gather
            # path below reads them as zero, which is what an unmaterialized
            # row means everywhere else here.
            return None
        scalars = self._row_map_np if self._row_map_np is not None else row_map
        first = int(scalars[b])
        if first < 0 or int(scalars[e - 1]) != first + (e - 1 - b):
            return None
        return first, first + (e - b)

    def _compact_index(self, rows):
        """``(columns, live_mask)`` for arbitrary global ``rows``.

        ``live_mask`` is ``None`` when every row is live (the common case), and
        otherwise marks the rows this window did not materialize -- which read
        as zero, exactly as they did when the buffer was full width and those
        rows were left zeroed.
        """
        row_map = self._active_row_map
        if row_map is None:
            return rows, None
        try:
            mapped = row_map[rows]
        except IndexError:
            # See _compact_span: rows allocated after this materialization are
            # not in the map at all, and read as zero rather than raising.
            # Handled on the exception rather than by a bounds check, so the
            # ordinary path does not pay a reduction over `rows` every call.
            width = row_map.shape[0]
            mapped = torch.where(
                rows < width,
                row_map[rows.clamp(max=width - 1)],
                torch.full_like(rows, -1),
            )
        live = mapped >= 0
        if bool(live.all()):
            return mapped, None
        # Left unclamped on purpose: every caller selects the live entries
        # before indexing, and a clamped -1 would silently read column 0 --
        # which is not even a valid index when the window materialized no rows
        # at all and the buffer is [T, 0, D].
        return mapped, live

    def get(self, key, copy=True):
        if isinstance(key, RowRanges):
            if (
                not _opt_disabled("ranges")
                and key.pairs is not None
                and len(key.pairs) == 1
            ):
                # Contiguous rows: slice instead of index-gather. The clone
                # keeps the copy semantics of advanced indexing (callers may
                # mutate the result in place); read-only callers that only feed
                # the value into out-of-place arithmetic pass copy=False to skip
                # it (the dominant cost during mob construction).
                b, e = key.pairs[0]
                span = self._compact_span(b, e)
                if span is not None:
                    block = self.active_state[:, span[0] : span[1]]
                    t = self.active_time_inds
                    if isinstance(t, slice):
                        return block[t].clone() if copy else block[t]
                    return block[t.view(-1)]
            key = key.tensor()
        columns, live = self._compact_index(key)
        t = self.active_time_inds
        if live is None:
            return self.active_state[t, columns]
        # Rows this window did not materialize read as zero, exactly as they
        # did when the buffer was full width and those rows were left zeroed.
        # Built as zeros and filled, rather than gathered and masked, because
        # a window whose working set is empty leaves a [T, 0, D] buffer with no
        # column to gather from at all.
        n_times = self.active_state[t].shape[0] if isinstance(t, slice) else t.numel()
        out = self.active_state.new_zeros(
            (n_times, live.shape[0], self.active_state.shape[2])
        )
        kept = live.nonzero().view(-1)
        if kept.numel():
            out[:, kept] = self.active_state[t, columns[kept]]
        return out

    def modify(self, key, value):
        # Replaying an animation modifies the temporary materialized buffer,
        # not the recorded edit log/current state.  Invalidating the prepared
        # CSR query data here made every frame batch rebuild and re-sort the
        # identical edit history. Recording-time writes still invalidate it.
        if self.active_state is self.current_state:
            self.invalidate_prepared_queries(retain_edit_prefix=True)
        if isinstance(key, RowRanges):
            if (
                not _opt_disabled("ranges")
                and key.pairs is not None
                and len(key.pairs) == 1
            ):
                # Contiguous rows: slice-assign instead of index-scatter.
                b, e = key.pairs[0]
                span = self._compact_span(b, e)
                if span is not None:
                    t = self.active_time_inds
                    if isinstance(t, slice):
                        self.active_state[t, span[0] : span[1]] = value
                    else:
                        self.active_state[t.view(-1), span[0] : span[1]] = value
                    return self
            key = key.tensor()
        columns, live = self._compact_index(key)
        if live is None:
            self.active_state[self.active_time_inds, columns] = value
            return self
        # Rows this window did not materialize have no column to write to.
        # Dropping those writes matches the full-width buffer, where they
        # landed in a row nothing would read: primitive preparation consumes
        # only the working set, and a row discovered late comes back through
        # materialize_additional_rows, which gives it a column first.
        kept = live.nonzero().view(-1)
        if kept.numel() == 0:
            return self
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[-2] == live.shape[0]:
            # Per-row values: drop the rows whose writes are being discarded.
            # A broadcast value (scalar, or a singleton row axis) passes through.
            value = value.index_select(-2, kept)
        self.active_state[self.active_time_inds, columns[kept]] = value
        return self

    def reassign_inds(self, mob_id, inds):
        """Point ``mob_id`` at ``inds``, invalidating the cached
        :class:`RowRanges` (:meth:`ranges_for`) so callers don't read stale
        rows. Use this for any direct row hand-over that bypasses :meth:`add`
        (e.g. :meth:`~algan.animatable_base.mob.Mob.detach_history`'s history
        swap). Bumps the structure
        version so per-mob descendant-row caches (``Mob._attr_inds_cache``) that
        may reference the old ownership are rebuilt.
        """
        self.mob_id_to_inds[mob_id] = inds
        self.mob_id_to_ranges.pop(mob_id, None)
        self._dirty_endpoint_rows.add(mob_id)
        self._invalidate_endpoint_values()
        bump_structure_version()

    def drop_mob(self, mob_id):
        """Forget ``mob_id``'s rows and cached ranges."""
        self.mob_id_to_inds.pop(mob_id, None)
        self.mob_id_to_ranges.pop(mob_id, None)
        self._dirty_endpoint_rows.add(mob_id)
        self._invalidate_endpoint_values()
        bump_structure_version()

    def ranges_for(self, mob_id):
        """The mob's own rows as a (cached) single-run :class:`RowRanges`."""
        ranges = self.mob_id_to_ranges.get(mob_id)
        if ranges is None:
            inds = self.mob_id_to_inds[mob_id]
            ranges = RowRanges.from_contiguous_blocks([inds])
            if ranges is None:  # non-contiguous (defensive; add() never does this)
                ranges = RowRanges(None, tensor=inds)
            self.mob_id_to_ranges[mob_id] = ranges
        return ranges

    def record(self, key, value, time, seq=0, event=None):
        old_value = self.get(key)
        # Edits store materialized index tensors: the replay machinery
        # (prepare_for_queries, _resolve_replay_windows) consumes them
        # directly, and recording happens once per edit, not once per batch.
        indexes = key.tensor() if isinstance(key, RowRanges) else key
        edit = EditRecord(indexes, old_value, time, seq, event)
        self.edits.append(edit)
        self.modify(key, value)
        return edit

    def add(self, mob, values, overwrite=False):
        mob_id = mob.id
        replacing_rows = mob_id in self.mob_id_to_inds
        if (not overwrite) and replacing_rows:
            return

        self.invalidate_prepared_queries(retain_edit_prefix=True)

        # New (or re-allocated) rows invalidate cached concatenated row
        # indexes.
        bump_structure_version()
        values = cast_to_tensor(values)
        n = values.shape[-2]
        new_pointer = self.pointer + n
        buffer_size = self.current_state.shape[-2]
        if new_pointer >= buffer_size:
            while new_pointer >= buffer_size:
                buffer_size *= 2
            new_buffer = torch.empty((1, buffer_size, self.current_state.shape[-1]))
            new_buffer[:, : self.pointer] = self.get_current_values()
            self.current_state = new_buffer
            self.active_state = self.current_state
        self.current_state[:, self.pointer : new_pointer] = values
        inds = torch.arange(self.pointer, new_pointer)
        self.mob_id_to_inds[mob_id] = inds
        # The block is contiguous by construction, so cache its single-run
        # RowRanges directly instead of re-deriving it (with tensor->int
        # conversions) on the first ranges_for() query.
        self.mob_id_to_ranges[mob_id] = RowRanges([(self.pointer, new_pointer)])
        self.pointer = new_pointer
        if replacing_rows or (
            mob_id in self.mob_id_to_starts or mob_id in self.mob_id_to_ends
        ):
            self._dirty_endpoint_rows.add(mob_id)
        # Existing endpoint ownership is unchanged for an ordinary new Mob;
        # register_spawn appends it to the start layout later. Until then the
        # new rows retain the default "not visible" bounds.
        self._invalidate_endpoint_values()
        return inds

    def prepare_for_queries(self):
        if self._is_ready_for_queries:
            return self
        self._is_ready_for_queries = True

        # Edits are kept in execution order, timestamped with their
        # replay-extended end times (AnimationTimeline._resolve_replay_windows
        # guarantees these are non-decreasing along every buffer row), so the
        # per-row binary search in _query_state_from_edits stays valid even
        # when edits overlap in time. When several edits on a row end at the
        # same (extended) time, the search returns the earliest-executed one,
        # whose pre-modification value is the correct base for re-applying
        # all of them in execution order.
        # Normal authoring only appends edits. Retain the normalized prefix
        # (including its flattened index/value views) and normalize only the
        # suffix recorded since the previous still. The old final-state
        # sentinel is the one entry after ``_prepared_edit_count``.
        if self._prepared_edit_count > len(self.edits):
            self._prepared_edit_count = 0
            self._edits_sorted.clear()
        del self._edits_sorted[self._prepared_edit_count :]
        self._edits_sorted.extend(
            {
                "indexes": e.indexes.view(-1),
                "values": e.values.squeeze(0),
                "timestamp": e.replay_end if e.replay_end is not None else e.time.end,
            }
            for e in self.edits[self._prepared_edit_count :]
        )
        self._prepared_edit_count = len(self.edits)
        self._edits_sorted.append(
            {
                "indexes": torch.arange(self.pointer),
                "values": self.current_state[:, : self.pointer].squeeze(0),
                "timestamp": math.inf,
            }
        )
        self._query_cache.clear()

        return self

    @staticmethod
    def _build_endpoint_layout(endpoint_map, mob_id_to_inds):
        lifespans = []
        row_blocks = []
        row_counts = []
        blocks = {}
        row_start = 0
        for mob_id, lifespan in endpoint_map.items():
            inds = mob_id_to_inds.get(mob_id)
            if inds is None or inds.numel() == 0:
                continue
            lifespans.append(lifespan)
            row_blocks.append(inds.view(-1))
            row_counts.append(inds.numel())
            blocks[mob_id] = (
                len(lifespans) - 1,
                row_start,
                row_start + inds.numel(),
            )
            row_start += inds.numel()
        if not row_blocks:
            empty = torch.empty((0,), dtype=torch.long)
            return _EndpointLayout(lifespans, empty, empty, blocks, 0)
        flat_rows = torch.cat(row_blocks)
        flat_owners = torch.repeat_interleave(
            torch.arange(len(row_blocks), dtype=torch.long),
            torch.tensor(row_counts, dtype=torch.long),
        )
        used = flat_rows.numel()
        capacity = max(16, 1 << (used - 1).bit_length())
        rows = torch.empty((capacity,), dtype=torch.long)
        owners = torch.empty((capacity,), dtype=torch.long)
        rows[:used] = flat_rows
        owners[:used] = flat_owners
        return _EndpointLayout(lifespans, rows, owners, blocks, used)

    def _endpoint_layouts(self):
        cached = self._endpoint_layout_cache
        if (
            cached is not None
            and not self._pending_start_endpoints
            and not self._pending_end_endpoints
            and not self._dirty_endpoint_rows
            and not self._dirty_start_endpoints
            and not self._dirty_end_endpoints
        ):
            return cached
        if cached is None:
            starts = self._build_endpoint_layout(
                self.mob_id_to_starts, self.mob_id_to_inds
            )
            ends = self._build_endpoint_layout(self.mob_id_to_ends, self.mob_id_to_inds)
        else:
            starts, ends = cached
            starts = self._patch_endpoint_layout(
                starts,
                self.mob_id_to_starts,
                self._pending_start_endpoints,
                self._dirty_start_endpoints,
                self._dirty_endpoint_rows,
                self.mob_id_to_inds,
            )
            ends = self._patch_endpoint_layout(
                ends,
                self.mob_id_to_ends,
                self._pending_end_endpoints,
                self._dirty_end_endpoints,
                self._dirty_endpoint_rows,
                self.mob_id_to_inds,
            )
            if starts is None or ends is None:
                starts = self._build_endpoint_layout(
                    self.mob_id_to_starts, self.mob_id_to_inds
                )
                ends = self._build_endpoint_layout(
                    self.mob_id_to_ends, self.mob_id_to_inds
                )
        self._pending_start_endpoints.clear()
        self._pending_end_endpoints.clear()
        self._dirty_endpoint_rows.clear()
        self._dirty_start_endpoints.clear()
        self._dirty_end_endpoints.clear()
        if cached is None or starts is not cached[0] or ends is not cached[1]:
            self._endpoint_layout_cache = (starts, ends)
        return starts, ends

    @classmethod
    def _patch_endpoint_layout(
        cls,
        layout,
        endpoint_map,
        pending,
        dirty_values,
        dirty_rows,
        mob_id_to_inds,
    ):
        """Apply same-sized history row swaps to a cached endpoint layout.

        ``detach_history`` changes thousands of Mob-to-row mappings between
        stills, but almost all are swaps of equal-sized blocks. Patching those
        blocks lazily avoids rebuilding every unchanged Mob's layout. A true
        row-count change returns ``None`` and lets the caller use the safe full
        rebuild path.
        """
        lifespans = layout.lifespans
        rows = layout.rows
        blocks = layout.blocks
        additions = dict(pending)
        for mob_id in dirty_values:
            block = blocks.get(mob_id)
            lifespan = endpoint_map.get(mob_id)
            if block is None:
                if lifespan is not None:
                    additions[mob_id] = lifespan
            elif lifespan is None:
                return None
            else:
                lifespans[block[0]] = lifespan

        for mob_id in dirty_rows:
            block = blocks.get(mob_id)
            inds = mob_id_to_inds.get(mob_id)
            lifespan = endpoint_map.get(mob_id)
            if block is None:
                if lifespan is not None and inds is not None and inds.numel():
                    additions[mob_id] = lifespan
                continue
            _, start, stop = block
            if inds is None or inds.numel() != stop - start:
                return None
            rows[start:stop] = inds.view(-1)

        return cls._extend_endpoint_layout(layout, additions.items(), mob_id_to_inds)

    @staticmethod
    def _extend_endpoint_layout(layout, pending, mob_id_to_inds):
        entries = []
        lifespans = []
        row_blocks = []
        row_counts = []
        for mob_id, lifespan in pending:
            block = layout.blocks.get(mob_id)
            if block is not None:
                layout.lifespans[block[0]] = lifespan
                continue
            inds = mob_id_to_inds.get(mob_id)
            if inds is None or inds.numel() == 0:
                continue
            entries.append((mob_id, lifespan))
            lifespans.append(lifespan)
            row_blocks.append(inds.view(-1))
            row_counts.append(inds.numel())
        if not row_blocks:
            return layout

        new_rows = torch.cat(row_blocks)
        old_lifespan_count = len(layout.lifespans)
        new_owners = torch.repeat_interleave(
            torch.arange(
                old_lifespan_count,
                old_lifespan_count + len(lifespans),
                dtype=torch.long,
            ),
            torch.tensor(row_counts, dtype=torch.long),
        )
        new_used = layout.used + new_rows.numel()
        if new_used > layout.rows.numel():
            capacity = max(16, 1 << (new_used - 1).bit_length())
            rows = torch.empty((capacity,), dtype=torch.long)
            owners = torch.empty((capacity,), dtype=torch.long)
            rows[: layout.used] = layout.rows[: layout.used]
            owners[: layout.used] = layout.owners[: layout.used]
            layout.rows = rows
            layout.owners = owners
        layout.rows[layout.used : new_used] = new_rows
        layout.owners[layout.used : new_used] = new_owners

        row_start = layout.used
        for offset, ((mob_id, _lifespan), row_count) in enumerate(
            zip(entries, row_counts)
        ):
            layout.blocks[mob_id] = (
                old_lifespan_count + offset,
                row_start,
                row_start + row_count,
            )
            row_start += row_count
        layout.lifespans.extend(lifespans)
        layout.used = new_used
        return layout

    def _refresh_end_points(self):
        """Materialize live spawn/despawn bounds for the next state query.

        Lifespans contain :class:`TimelineEvent` callables whose values can be
        rescaled after a still is rendered from inside an open animation
        context. A global timing revision catches that rescaling; a local
        layout revision catches row ownership and endpoint-map changes. The
        expensive row expansion is cached and each column is written in one
        vectorized operation rather than one indexed assignment per Mob.
        """
        version = (TIMING_VERSION[0], self._endpoint_layout_revision)
        if self._end_points_version == version:
            return self

        self._end_points = torch.full((1, self.pointer + 1, 2), 1e12)
        starts, ends = self._endpoint_layouts()
        for column, endpoint_name, layout in (
            (0, "start", starts),
            (1, "end", ends),
        ):
            if layout.used == 0:
                continue
            lifespans = layout.lifespans
            rows = layout.rows[: layout.used]
            owners = layout.owners[: layout.used]
            values = torch.tensor(
                [getattr(lifespan, endpoint_name)() for lifespan in lifespans],
                dtype=self._end_points.dtype,
            )
            self._end_points[0, rows, column] = values[owners]
        self._end_points_version = version
        return self

    def rows_for_mob_ids(self, mob_ids):
        """Return the global rows owned by ``mob_ids`` as coalesced runs."""
        runs = []
        loose = []
        for mob_id in mob_ids:
            ranges = self.mob_id_to_ranges.get(mob_id)
            if ranges is None and mob_id in self.mob_id_to_inds:
                ranges = self.ranges_for(mob_id)
            if ranges is None:
                continue
            if ranges.pairs is None:
                # Structural rebatching can defensively produce an
                # uncompressible set. Preserve it verbatim.
                loose.append(ranges.tensor())
            else:
                runs.extend(ranges.pairs)
        if not runs and not loose:
            return torch.empty((0,), dtype=torch.long)
        compressed = (
            RowRanges.from_runs(runs).tensor()
            if runs
            else torch.empty((0,), dtype=torch.long)
        )
        if loose:
            return torch.unique(torch.cat([compressed, *loose]), sorted=True)
        return compressed

    def _prepared_queries(self, times):
        key = (str(times.device), times.dtype)
        prepared = self._query_cache.get(key)
        if prepared is None:
            prepared = _prepare_array_state_queries(
                times, self.pointer + 1, self._edits_sorted
            )
            self._query_cache[key] = prepared
        return prepared

    def rematerialize_state_at_times(self, times, active_mob_ids=None, extra_rows=None):
        self.prepare_for_queries()
        if self.record_end_points:
            self._refresh_end_points()
        active_rows = (
            None if active_mob_ids is None else self.rows_for_mob_ids(active_mob_ids)
        )
        if active_rows is not None and extra_rows is not None:
            extra_rows = extra_rows.view(-1)
            if extra_rows.numel():
                active_rows = torch.unique(
                    torch.cat((active_rows, extra_rows)), sorted=True
                )
        compact = active_rows is not None and not _opt_disabled("compactstate")
        if compact:
            # Materialize only the window's live rows. The full-width
            # [T, N, D] buffer this replaces was ~70% dead weight, and its
            # commit is O(every row the scene ever allocated) per attribute per
            # batch -- O(n^2) across a render even after the zero-fill itself
            # stopped costing time. _set_active_row_map is what lets every
            # reader keep addressing rows by global id.
            active_rows = active_rows.to(device=times.device, dtype=torch.int64)
            self.active_state = _query_row_states(
                times, self._prepared_queries(times), active_rows
            )
            self._set_active_row_map(active_rows)
        else:
            self.active_state = generate_array_states(
                times,
                self.pointer + 1,
                self._edits_sorted,
                active_rows=active_rows,
                prepared=self._prepared_queries(times),
            )
            self._clear_active_row_map()
        if self.record_end_points:
            t = times.view(-1, 1)
            if active_rows is None:
                self.active_state *= (
                    (self._end_points[..., 0] <= t) & (t < self._end_points[..., 1])
                ).unsqueeze(-1)
            elif active_rows.numel():
                rows = active_rows.to(self.active_state.device)
                endpoint = self._end_points[:, rows].to(self.active_state.device)
                mask = ((endpoint[..., 0] <= t) & (t < endpoint[..., 1])).unsqueeze(-1)
                if compact:
                    # The compact buffer is already in active_rows order, so
                    # the mask lines up with it column for column -- no
                    # gather-modify-scatter through the global layout.
                    self.active_state *= mask
                else:
                    self.active_state[:, rows] *= mask
        self.rematerialized_times = times
        self.active_time_inds = slice(None, None, None)
        return self

    def set_active_time_inds(self, time_inds):
        self.active_time_inds = time_inds

    def materialize_additional_rows(self, times, rows):
        """Fill selected rows in an already materialized sparse state.

        Updater dependency tracing can discover a previously unseen mob while
        the updater is executing. Materializing those rows lazily keeps that
        first frame correct; the dependency is retained on the updater so
        subsequent frame windows include it in their initial working set.
        """
        rows = rows.view(-1)
        if rows.numel() == 0 or self.active_state is self.current_state:
            return self
        self.prepare_for_queries()
        device_rows = rows.to(device=self.active_state.device, dtype=torch.long)
        if _opt_disabled("torchquery"):
            queried = generate_array_states(
                times,
                self.pointer + 1,
                self._edits_sorted,
                active_rows=rows,
                prepared=self._prepared_queries(times),
            )[:, device_rows]
        else:
            # Only these rows are wanted, so query them compactly instead of
            # materializing (and discarding) the whole global-row buffer.
            queried = _query_row_states(
                times, self._prepared_queries(times), device_rows
            )
        if self.record_end_points:
            t = times.view(-1, 1)
            endpoint = self._end_points[:, rows].to(self.active_state.device)
            mask = ((endpoint[..., 0] <= t) & (t < endpoint[..., 1])).unsqueeze(-1)
            queried = queried * mask
        if self._active_row_map is None:
            self.active_state[:, device_rows] = queried
            return self
        # Compact buffer: a row discovered here has no column yet. Append the
        # ones that are new (this is the lazily-traced updater dependency path,
        # so it is rare and small) and point the map at them; rows that already
        # have a column are overwritten in place.
        columns = self._active_row_map[device_rows]
        existing = (columns >= 0).nonzero().view(-1)
        if existing.numel():
            self.active_state[:, columns[existing]] = queried[:, existing]
        fresh = (columns < 0).nonzero().view(-1)
        if fresh.numel():
            width = int(self.active_state.shape[1])
            new_rows = device_rows[fresh]
            self.active_state = torch.cat(
                (self.active_state, queried[:, fresh]), dim=1
            )
            self._active_row_map[new_rows] = torch.arange(
                width, width + int(fresh.numel()), dtype=torch.int64
            )
            self._mapped_rows = torch.cat((self._mapped_rows, new_rows))
        return self

    def clear_buffers(self):
        self.active_state = self.current_state
        self._clear_active_row_map()
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        return self


class TimelineEvent:
    """One timestamp on the timeline, resolved through its span's rescaling.

    The resolved value is memoized against the global timing revision. A
    render reads a mob's spawn and despawn timestamps for every actor of every
    frame batch (and again for every attribute's endpoint table), and the value
    cannot move unless something rescales a span -- which nothing does once the
    scene is authored. Every writer of a span's timings bumps that revision, so
    a stale entry is not reachable.
    """

    def __init__(self, time, span):
        self.span = span
        self._time = time
        self._cached_version = -1
        self._cached_value = 0.0

    def __call__(self):
        return self.time

    @property
    def end(self):
        return self.time

    @property
    def time(self):
        version = TIMING_VERSION[0]
        if self._cached_version == version:
            return self._cached_value
        value = self.span.get_rescaled_time(self._time)
        self._cached_value = value
        self._cached_version = version
        return value

    @time.setter
    def time(self, time):
        self._time = time
        self._cached_version = -1


class TimelineSpan:
    def __init__(self, start_time=0, end_time=0, current_time=0):
        self._rescaled_start = start_time
        self._rescaled_end = end_time
        self._original_start = start_time
        self._original_end = end_time
        self.current_time = current_time

    # The four timing fields are properties rather than plain attributes so
    # that every write is caught by the global timing revision: TimelineEvent
    # memoizes its resolved timestamp against that revision, and the original
    # bounds are part of the rescaling it resolves through. Writes that do not
    # change the value do not bump, so the ratio-of-one rescale every replayed
    # animation context performs leaves derived caches valid.
    @property
    def original_start(self):
        return self._original_start

    @original_start.setter
    def original_start(self, value):
        if value != self._original_start:
            self._original_start = value
            bump_timing_version()

    @property
    def original_end(self):
        return self._original_end

    @original_end.setter
    def original_end(self, value):
        if value != self._original_end:
            self._original_end = value
            bump_timing_version()

    def __call__(self):
        return self.original_start

    def rescale(self, new_start, ratio):
        self.start = (self.start - new_start) * ratio + new_start
        self.end = (self.end - new_start) * ratio + new_start

    def get_rescaled_time(self, t):
        a = (t - self.original_start) / max(
            self.original_end - self.original_start, 1e-6
        )
        return self.start + (self.end - self.start) * a

    def get_time(self, time):
        return TimelineEvent(time, self)

    def get_current_time(self):
        return self.get_time(self.current_time)

    @property
    def start(self):
        return self._rescaled_start

    @start.setter
    def start(self, value):
        # Only a change of value can invalidate anything derived from it, and
        # rescaling by a ratio of one (what every context entered during
        # animation replay does) writes the value back unchanged. Bumping
        # regardless made the derived caches -- notably the per-attribute
        # spawn/despawn endpoint table -- miss on every frame batch of a
        # render, rebuilding a table that had not moved since the scene was
        # authored.
        if value != self._rescaled_start:
            self._rescaled_start = value
            bump_timing_version()

    @property
    def end(self):
        return self._rescaled_end

    @end.setter
    def end(self, value):
        if value != self._rescaled_end:
            self._rescaled_end = value
            bump_timing_version()


class FunctionApplicationEvent:
    def __init__(
        self,
        function,
        caller,
        animated_args=None,
        kwargs=None,
        rate_func=None,
        time=None,
    ):
        self.function = function
        self.caller = caller
        self.animated_args = animated_args
        self.kwargs = kwargs
        self.rate_func = rate_func
        self.time = time
        # Resolved replay-window end (see
        # AnimationTimeline._resolve_replay_windows); None until resolved or
        # when the function recorded no attribute edits.
        self.replay_end = None
        # Exact attribute rows modified while this function was recorded, in
        # execution order. A mob can be structurally rebatched later, so
        # replay must not rediscover its rows from the current hierarchy.
        self.recorded_edits = []
        # The EditRecord behind each of those entries, positionally aligned.
        # Only a topology split reads it (see Surface's boundary capture); it
        # is kept here so that split does not have to find them by searching
        # the attribute's entire edit log.
        self.recorded_edit_records = []


class UpdaterSpan:
    """The [added, removed) interval of one updater. ``start``/``end`` expose
    the (lazily rescaled) timestamps as numbers, matching the protocol of the
    context timespans carried by ordinary :class:`FunctionApplicationEvent` s.
    An updater that was never removed ends at :data:`UPDATER_FOREVER`.
    """

    __slots__ = ("start_event", "end_event")

    def __init__(self, start_event):
        self.start_event = start_event
        self.end_event = None

    @property
    def start(self):
        return self.start_event.time

    @property
    def end(self):
        return UPDATER_FOREVER if self.end_event is None else self.end_event.time


class UpdaterEvent:
    """An updater: ``function(mob, time_elapsed, *args, **kwargs)`` is applied
    at every frame in ``time.start <= t < time.end``, with ``time_elapsed``
    equal to ``t - time.start``.
    """

    __slots__ = (
        "function",
        "caller",
        "args",
        "kwargs",
        "time",
        "dependency_mob_ids",
        "_history_clones",
        "_invocation_cache",
    )

    def __init__(self, function, caller, args, kwargs, time):
        self.function = function
        self.caller = caller
        self.args = args
        self.kwargs = kwargs
        self.time = time
        self.dependency_mob_ids = set()
        # ``Mob.detach_history`` leaves the live Python object on the newest
        # timeline rows and hands each earlier interval to a clone. Persistent
        # updaters still need to address whichever incarnation is visible in a
        # queried frame, including replacements nested below their caller.
        self._history_clones = {}
        self._invocation_cache = {}

    def register_history_clone(self, original, clone):
        entry = self._history_clones.setdefault(id(original), (original, []))
        if not any(existing is clone for existing in entry[1]):
            entry[1].append(clone)
            self._invocation_cache.clear()
        self.dependency_mob_ids.add(clone.id)

    def replacement_signature_at(self, time):
        """Return ``(original, visible historical clone)`` pairs at ``time``."""
        replacements = []
        for original, clones in self._history_clones.values():
            replacement = next(
                (
                    clone
                    for clone in clones
                    if clone.lifespan.start() <= time < clone.lifespan.end()
                ),
                None,
            )
            if replacement is not None:
                replacements.append((original, replacement))
        replacements.sort(key=lambda pair: id(pair[0]))
        return tuple(replacements)

    def invocation_for_signature(self, signature):
        """Build a view of the caller graph containing historical Mobs.

        The view shares every untouched Mob's timeline id. Only references to
        objects replaced by ``detach_history`` point at the independent clone,
        so ordinary updater code can keep navigating user attributes and child
        lists without knowing that ownership changed.
        """
        if not signature:
            return self.caller, self.args, self.kwargs
        key = tuple((id(original), id(clone)) for original, clone in signature)
        cached = self._invocation_cache.get(key)
        if cached is not None:
            return cached
        memo = {
            "___copy_add_to_scene___": False,
            "___copy_spawn___": False,
            "___copy_animate_creation___": False,
            "___copy_recursive___": True,
            "___clone_data___": False,
            **{id(original): clone for original, clone in signature},
        }
        invocation = (
            copy.deepcopy(self.caller, memo),
            copy.deepcopy(self.args, memo),
            copy.deepcopy(self.kwargs, memo),
        )
        self._invocation_cache[key] = invocation
        return invocation


class FunctionTimeline:
    def __init__(self):
        self.function_applications = []
        self.updaters = []
        # (num_events, starts [E], replay ends [E]): the resolved window of
        # every recorded event, rebuilt when events are added or replay
        # windows re-resolved (see AnimationTimeline._resolve_replay_windows).
        # Event windows are lazy (contexts rescale timestamps retroactively),
        # so evaluating them for every event on every frame batch is a
        # per-batch O(events) property-call cost otherwise.
        self._window_cache = None
        self._updater_window_cache = None
        # id(caller) -> the events recorded against it. A topology split
        # (Mob.detach_history) has to re-target every event belonging to the
        # subtree it replaces, and finding them by scanning the whole log is
        # quadratic in the length of the scene -- one wave_color over a coarsely
        # sampled mob splits once per part, and each split then walked every
        # event authored so far. Keys stay live by construction: a bucket is
        # dropped as soon as it empties, and while it is non-empty its events
        # hold the caller they are keyed by.
        self._by_caller = {}

    def add(self, function_application):
        # Its position in ``function_applications``, so the replay-window
        # resolver can say *which* events it touched instead of invalidating
        # the whole bounds cache (see :meth:`invalidate_window_cache`).
        function_application._window_slot = len(self.function_applications)
        self.function_applications.append(function_application)
        self._by_caller.setdefault(id(function_application.caller), []).append(
            function_application
        )

    def events_for_caller(self, caller):
        """The recorded events whose caller *is* ``caller`` (identity)."""
        return self._by_caller.get(id(caller), ())

    def retarget_caller(self, event, new_caller):
        """Hand one recorded event to a different caller, keeping the index."""
        bucket = self._by_caller.get(id(event.caller))
        if bucket is not None:
            for index, existing in enumerate(bucket):
                if existing is event:
                    del bucket[index]
                    break
            if not bucket:
                del self._by_caller[id(event.caller)]
        event.caller = new_caller
        self._by_caller.setdefault(id(new_caller), []).append(event)

    def add_updater(self, updater):
        self.updaters.append(updater)

    def invalidate_window_cache(self, from_index=0):
        """Drop cached event bounds from ``from_index`` on.

        ``from_index`` is the lowest position in ``function_applications``
        whose window may have moved; everything before it keeps its bounds and
        :meth:`_windows` rebuilds only the tail. ``None`` means no function
        window changed at all.

        This exists because the resolver runs on *every* batch of a render (a
        render records edits of its own, which un-resolves the timeline), so a
        full invalidation meant re-walking every recorded event's span every
        batch -- 58 ms a batch at 26 000 events, which was the entire cost of
        the function lookup and swamped the query it feeds.

        The updater cache is dropped wholesale regardless: the resolver does
        not touch updater spans, but a context exit can rescale them without
        changing their count, and the resolver running is the signal that
        authoring happened. Rebuilding it is O(updaters), which is why this
        stays cheap in practice and would want the same treatment on a scene
        with very many updaters.
        """
        if from_index is not None:
            cache = self._window_cache
            if from_index <= 0 or cache is None or cache[0] <= from_index:
                if from_index <= 0:
                    self._window_cache = None
            else:
                # Keep the verified prefix; index set to None forces a rebuild.
                self._window_cache = (
                    from_index,
                    cache[1][:from_index],
                    cache[2][:from_index],
                    None,
                )
        self._updater_window_cache = None

    def _windows(self):
        # Length plus explicit invalidation, NOT the timing revision. Keying on
        # TIMING_VERSION looks safer and is a trap: it is bumped whenever any
        # timespan is configured, including for the transient mobs a render
        # builds, so it changes *during* a render and rebuilt this on every
        # batch. Recording invalidates through _resolve_replay_windows instead.
        #
        # Grown incrementally, because a render *does* record new events as it
        # goes (the reference scene appends ~30-100 per batch). Rebuilding from
        # scratch means walking every recorded event's TimelineEvent to resolve
        # its span, which measured 53 ms a batch at 26 000 events and was the
        # real cost of this lookup all along -- far more than the window test
        # it was feeding. Only the appended tail is walked now; the tensor
        # concat and the index rebuild over the whole array are vectorized and
        # cost ~0.5 ms at that size.
        n = len(self.function_applications)
        cache = self._window_cache
        if cache is not None and cache[0] == n and cache[3] is not None:
            return cache[1], cache[2], cache[3]
        # float32 to match the dtype the per-event scalar comparisons
        # used (python-float scalars compare in the tensor's dtype).
        if cache is None or cache[0] > n:
            appended = self.function_applications
            prefix_starts = prefix_ends = None
        else:
            appended = self.function_applications[cache[0] :]
            prefix_starts, prefix_ends = cache[1], cache[2]
        starts = torch.tensor(
            [f.time.start for f in appended], dtype=torch.float32
        )
        ends = torch.tensor(
            [_replay_window_end(f) for f in appended], dtype=torch.float32
        )
        if prefix_starts is not None:
            starts = torch.cat((prefix_starts, starts))
            ends = torch.cat((prefix_ends, ends))
        cache = self._window_cache = (
            n,
            starts,
            ends,
            _event_interval_index(starts, ends),
        )
        return cache[1], cache[2], cache[3]

    def _updater_windows(self):
        # float32 like _windows, and for the same reason: the per-event scalar
        # comparison this replaces promoted each python-float bound into the
        # queried times' dtype, which is float32 for every frame window the
        # render loop builds. A caller passing float64 times now compares
        # against a float32-rounded bound, exactly as the function lookup has
        # always done.
        #
        # Same validity rule as _windows (length + invalidate_window_cache),
        # and see the trap documented there about TIMING_VERSION.
        key = len(self.updaters)
        cache = self._updater_window_cache
        if cache is None or cache[0] != key:
            starts = torch.tensor(
                [f.time.start for f in self.updaters], dtype=torch.float32
            )
            ends = torch.tensor(
                [f.time.end for f in self.updaters], dtype=torch.float32
            )
            cache = self._updater_window_cache = (
                key,
                starts,
                ends,
                _event_interval_index(starts, ends),
            )
        return cache[1], cache[2], cache[3]

    @staticmethod
    def _active_in_window(events, starts, ends, times, index):
        """The members of ``events`` active at ``times``, in recorded order.

        ``index`` narrows the exact test to the events that can possibly be
        active (see :func:`_event_interval_index`); ``None`` runs it over all of
        them, which is what ``ALGAN_OPT_DISABLE=eventindex`` restores.
        """
        t = times.view(1, -1)
        if index is None:
            active = ((starts.view(-1, 1) <= t) & (t < ends.view(-1, 1))).any(1)
            selected = active.nonzero().view(-1)
        else:
            candidates = _events_overlapping(index, times)
            if candidates.numel() == 0:
                return []
            active = (
                (starts[candidates].view(-1, 1) <= t)
                & (t < ends[candidates].view(-1, 1))
            ).any(1)
            selected = candidates[active].sort().values
        return [events[i] for i in selected.tolist()]

    def get_functions_for_times(self, times):
        if not self.function_applications:
            return []
        if _opt_disabled("windows"):
            return [
                f
                for f in self.function_applications
                if ((f.time.start <= times) & (times < _replay_window_end(f))).any()
            ]
        starts, ends, index = self._windows()
        return self._active_in_window(
            self.function_applications,
            starts,
            ends,
            times,
            None if _opt_disabled("eventindex") else index,
        )

    def get_updaters_for_times(self, times):
        if len(self.updaters) <= _SMALL_EVENT_SCAN or _opt_disabled("windows"):
            # Below the threshold the index costs more than it saves: building
            # and sorting its tensors is dominated by fixed per-op overhead,
            # and most scenes have a handful of updaters or none. Measured at
            # one updater: 0.45 ms a batch indexed against 0.14 ms scanned.
            return [
                f
                for f in self.updaters
                if ((f.time.start <= times) & (times < f.time.end)).any()
            ]
        starts, ends, index = self._updater_windows()
        return self._active_in_window(
            self.updaters,
            starts,
            ends,
            times,
            None if _opt_disabled("eventindex") else index,
        )


class AnimationTimeline:
    def __init__(self):
        self.attr_to_timeline = {}
        self.function_timeline = FunctionTimeline()
        self.mob_id_to_lifespan = {}
        # Edit attribution state: a global execution counter for edits, the
        # function application the currently-executing animated function was
        # recorded as (edits made while it runs attach to it), and the most
        # recently recorded function application (consumed by the
        # animated_function wrapper to scope the former).
        self._edit_seq = 0
        # Global edit order, maintained as edits are recorded.  Attribute
        # timelines keep their own per-attribute logs for state queries, but
        # replay-window resolution needs the interleaved execution order.  A
        # central list avoids rebuilding and sorting that order for every
        # still render.
        self._edits_in_order = []
        self._edit_order_dirty = False
        # A resolved prefix is immutable once all contexts containing it have
        # exited: later edits depend on its per-row ends, but can never change
        # an earlier edit's replay window.  Reusing this checkpoint turns a
        # sequence of save_frame calls from a repeated full-history scan into
        # an incremental suffix scan.
        self._resolved_prefix_count = 0
        self._resolved_prefix_seq = -1
        self._resolved_row_ends = {}
        # Backing storage for _resolved_row_ends, kept across resolves so the
        # checkpoint is grown in place instead of reallocated and copied on
        # every batch. The dict above holds views into these. Whenever
        # _resolved_row_ends is replaced from outside this class' incremental
        # path, this MUST be cleared alongside it, or the views and their
        # backing store disagree.
        self._row_ends_capacity = {}
        self._active_edit_event = None
        self.last_recorded_event = None
        self._replay_windows_resolved = True
        self._active_replay_event = None
        self._active_replay_edit_index = 0
        self._active_updater_trace = None
        self._active_updater_write_capture = None
        self._materialization_times = None
        self._materialized_mob_ids = None
        self._updater_history_clones = {}

    def set_active_edit_event(self, event):
        """Set the function application that subsequently recorded attribute
        edits are attributed to, returning the previous one (so callers can
        restore it).
        """
        previous = self._active_edit_event
        self._active_edit_event = event
        return previous

    def begin_updater_dependency_trace(self, event):
        """Trace Mob state read or written by ``event`` until restored."""
        previous = self._active_updater_trace
        self._active_updater_trace = event
        return previous

    def end_updater_dependency_trace(self, previous):
        self._active_updater_trace = previous

    def begin_updater_write_capture(self, event):
        """Capture attribute writes made by one updater during replay."""
        previous = self._active_updater_write_capture
        writes = []
        self._active_updater_write_capture = (event, writes)
        return previous, writes

    def end_updater_write_capture(self, previous):
        self._active_updater_write_capture = previous

    def capture_updater_write(self, attr_name, indexes, value):
        """Save one replayed write when its updater is being finalized."""
        capture = self._active_updater_write_capture
        if capture is None or capture[0] is not self._active_updater_trace:
            return
        if isinstance(indexes, RowRanges):
            indexes = indexes.tensor()
        capture[1].append((attr_name, indexes.detach().clone(), value.detach().clone()))

    def trace_updater_mob_access(self, mob, include_descendants=False):
        """Record one updater Mob access and materialize newly seen rows.

        The initial updater invocation records the common dependency set at
        authoring time. The same trace remains active during materialization,
        so time-dependent branches can safely discover extra Mobs on demand.
        """
        event = self._active_updater_trace
        if event is None:
            return
        mob_ids = self._collect_mob_ids((mob,)) if include_descendants else {mob.id}
        event.dependency_mob_ids.update(mob_ids)
        mob_ids.update(self._register_known_history_clones(event, mob_ids))

        if self._materialized_mob_ids is None or self._materialization_times is None:
            return
        missing = mob_ids.difference(self._materialized_mob_ids)
        if not missing:
            return
        for timeline in self.attr_to_timeline.values():
            rows = timeline.rows_for_mob_ids(missing)
            timeline.materialize_additional_rows(self._materialization_times, rows)
        self._materialized_mob_ids.update(missing)

    def _register_known_history_clones(self, event, mob_ids):
        # Registering a clone adds its id to the same dependency set.
        registered_ids = set()
        for mob_id in tuple(mob_ids):
            for original, clone in self._updater_history_clones.get(mob_id, ()):
                event.register_history_clone(original, clone)
                registered_ids.add(clone.id)
        return registered_ids

    def register_updater_history_split(self, descendant_map):
        """Teach persistent updaters about rows handed to historical clones."""
        for original, clone in descendant_map.items():
            entries = self._updater_history_clones.setdefault(original.id, [])
            if not any(
                known_original is original and known_clone is clone
                for known_original, known_clone in entries
            ):
                entries.append((original, clone))
        for event in self.function_timeline.updaters:
            self._register_known_history_clones(event, event.dependency_mob_ids)

    def get_lifespan(self, mob_id):
        """The :class:`Lifespan` of the mob with the given id, created on
        first access (start = end = "never").
        """
        lifespan = self.mob_id_to_lifespan.get(mob_id)
        if lifespan is None:
            lifespan = self.mob_id_to_lifespan[mob_id] = Lifespan()
        return lifespan

    def register_spawn(self, mob, lifespan):
        # Visibility masking: the opacity timeline zeroes a mob's opacity
        # outside its [spawn, despawn) interval when materializing state.
        timeline = self.attr_to_timeline.get("opacity")
        if timeline is not None:
            timeline.set_start_point(mob, lifespan)

    def register_despawn(self, mob, lifespan):
        timeline = self.attr_to_timeline.get("opacity")
        if timeline is not None:
            timeline.set_end_point(mob, lifespan)

    def add_mob_attr(self, mob, attr, value, add_mob=True):
        if attr not in self.attr_to_timeline:
            self.attr_to_timeline[attr] = AttributeTimeline(
                value.shape[-1], attr_name=attr, record_end_points=attr == "opacity"
            )
        if not add_mob:
            return
        timeline = self.attr_to_timeline[attr]
        # if mob.id not in timeline.mob_id_to_inds:
        timeline.add(mob, value)
        return

    def get_inds(self, attr, mob, value=None):
        if attr not in self.attr_to_timeline:
            if value is None:
                raise AttributeError
            else:
                self.attr_to_timeline[attr] = AttributeTimeline(value.shape[-1])
        timeline = self.attr_to_timeline[attr]
        if mob.id not in timeline.mob_id_to_inds:
            if value is None:
                raise AttributeError
            else:
                timeline.add(mob, value)
        return self.attr_to_timeline[attr].mob_id_to_inds[mob.id]

    def get_attr(self, attr, inds, copy=True):
        timeline = self.attr_to_timeline[attr]
        return timeline.get(inds, copy=copy)

    def record_function(
        self, function, caller, animated_args, kwargs, animation_context
    ):
        c = animation_context
        self.last_recorded_event = None
        if c.run_time_unit <= 0 or not c.record_funcs:
            return kwargs
        rate_func = c.rate_func
        rate_func_compose = c.rate_func_compose
        rf = rate_func
        if rate_func_compose is not None:

            def rf(x, rf=rate_func, rfc=rate_func_compose):
                return rf(rfc(x))

        event = FunctionApplicationEvent(
            function, caller, animated_args, kwargs, rf, c.timespan
        )
        self.function_timeline.add(event)
        self.last_recorded_event = event
        return kwargs

    def record_updater(self, function, caller, args, kwargs, animation_context):
        """Register an updater starting at the context's current time and
        lasting until :meth:`end_updater` (or forever). Returns its id.
        """
        span = UpdaterSpan(animation_context.timespan.get_current_time())
        event = UpdaterEvent(function, caller, args, kwargs, span)
        event.dependency_mob_ids.update(self._collect_mob_ids((caller, args, kwargs)))
        self._register_known_history_clones(event, event.dependency_mob_ids)
        self.function_timeline.add_updater(event)
        return len(self.function_timeline.updaters) - 1

    def end_updater(self, updater_id, animation_context):
        span = self.function_timeline.updaters[updater_id].time
        span.end_event = animation_context.timespan.get_current_time()
        return span

    def get_timeline_inds(self, mob, new_value, attr_name):
        timeline = self.attr_to_timeline[attr_name]
        inds = None
        if mob.id not in timeline.mob_id_to_inds:
            inds = timeline.add(mob, new_value)
        return timeline, inds

    def modify_attribute_and_record(
        self, attr_name, mob_id, include_descendants, mob_inds, new_value, time
    ):
        timeline = self.attr_to_timeline[attr_name]
        self._edit_seq += 1
        event = self._active_edit_event
        edit = timeline.record(mob_inds, new_value, time, self._edit_seq, event)
        self._edits_in_order.append((edit.seq, attr_name, edit))
        if event is not None:
            event.recorded_edits.append(
                (attr_name, mob_id, include_descendants, edit.indexes)
            )
            # The EditRecord each entry came from, positionally aligned with
            # recorded_edits. Recovering it by searching the attribute's whole
            # edit log (which is what a topology split used to do, once per
            # entry) is quadratic in how much the scene has recorded.
            event.recorded_edit_records.append(edit)
        self._replay_windows_resolved = False
        return self

    def register_migrated_edit(
        self, attr_name, attr_timeline, source_edit, migrated_edit
    ):
        """Register an edit created by a historical-topology migration.

        Surface resolution changes can split an already-recorded edit and add
        a replacement carrying the same execution sequence.  That is the only
        path that inserts into (rather than appends to) global edit order, so
        it marks the order dirty and invalidates a resolved checkpoint only
        when the changed source lies inside that checkpoint.
        """
        attr_timeline.edits.append(migrated_edit)
        self._edits_in_order.append((migrated_edit.seq, attr_name, migrated_edit))
        self._edit_order_dirty = True
        if source_edit.seq <= self._resolved_prefix_seq:
            self._resolved_prefix_count = 0
            self._resolved_prefix_seq = -1
            self._resolved_row_ends = {}
            self._row_ends_capacity = {}
        self._replay_windows_resolved = False

    def replay_inds(self, attr_name, mob_id, include_descendants, consume=False):
        """Return the next recorded row set while replaying one function."""
        event = self._active_replay_event
        if event is None:
            return None
        index = self._active_replay_edit_index
        if index >= len(event.recorded_edits):
            return None
        edit_attr, edit_mob_id, edit_recursive, inds = event.recorded_edits[index]
        if (
            edit_attr != attr_name
            or edit_mob_id != mob_id
            or edit_recursive != include_descendants
        ):
            return None
        if consume:
            self._active_replay_edit_index += 1
        return inds

    def peek_replay_inds(self, attr_name, mob_id, include_descendants):
        """Rows a *read* must use while replaying one function.

        A replayed function has to read the same rows it will write, or the
        value it computes is indexed differently from the buffer slots it
        lands in. :meth:`replay_inds` only answers for the edit at the cursor,
        which is right for writes -- they consume the recorded edits in
        order -- but wrong for reads, because a function need not read its
        attributes in the order it writes them.
        :meth:`~algan.animatable_base.mob.Mob._apply_basis_change` reads the
        recursive ``basis`` before writing the recursive ``location``, so an
        at-the-cursor match misses and the read silently falls back to the
        *current* hierarchy's rows.

        That fallback is only harmless while the two agree. They stop
        agreeing as soon as anything reallocates a descendant's rows after the
        function was recorded -- ``detach_history`` (and so every
        :meth:`~.Mob.wave_color` auto-resolution restore) hands the old rows
        to a clone and appends fresh ones at the end of the buffer, which
        reorders the sorted descendant union. The read then returns some other
        Mob's values, and the write scatters them across unrelated rows.

        So search forward from the cursor for the edit this read pairs with,
        without consuming it. Returns ``None`` when the function records no
        matching write, in which case the caller's live-topology rows are the
        correct answer.
        """
        event = self._active_replay_event
        if event is None:
            return None
        recorded_edits = event.recorded_edits
        for index in range(self._active_replay_edit_index, len(recorded_edits)):
            edit_attr, edit_mob_id, edit_recursive, inds = recorded_edits[index]
            if (
                edit_attr == attr_name
                and edit_mob_id == mob_id
                and edit_recursive == include_descendants
            ):
                return inds
        return None

    def modify_attribute(self, attr_name, mob_inds, new_value):
        timeline = self.attr_to_timeline[attr_name]
        timeline.modify(mob_inds, new_value)
        return self

    def _resolve_replay_windows(self):
        """Resolve the effective replay window of every edit and function
        application, making materialization robust to multiple edits of the
        same attribute overlapping in time.

        An edit's recorded pre-modification snapshot is only a valid base
        state once every *earlier-executed* edit of the same rows has fully
        finished (until then the base must stay at the earliest-executed
        unfinished edit's pre-values, and the later edits' functions must be
        re-applied on top, at their final parameters once past their own end,
        to rebuild the recorded chain of states). To that effect each edit's
        ``replay_end`` is its own end time extended over the replay windows of
        all earlier-executed edits that overlap it on any row, propagated
        transitively; all edits recorded by one function application share one
        window (the function is re-executed as a whole), which is also stored
        on the event to extend its replay interval in
        :meth:`set_state_to_times`.

        By construction ``replay_end`` is non-decreasing along every buffer
        row in execution order, which is what lets
        :meth:`AttributeTimeline.prepare_for_queries` keep edits in execution
        order for the per-row binary search. For edits that do not overlap,
        ``replay_end`` equals the edit's own end time and behavior is
        unchanged.
        """
        if self._replay_windows_resolved:
            return
        self._replay_windows_resolved = True
        # Replay-window ends feed the cached event-window tensors, but this
        # runs on every batch of a render and resumes from _resolved_prefix_count
        # -- so it only ever assigns windows to events at the tail. The lowest
        # slot it touches is collected below and handed to the cache, which
        # keeps the (much larger) verified prefix.
        touched_slot = None

        if self._edit_order_dirty:
            # Migration records reuse their source edit's sequence number and
            # therefore have to be inserted beside it. Python's stable sort
            # preserves source-before-migration order for the tie, matching
            # the former per-attribute gather-and-sort implementation.
            self._edits_in_order.sort(key=lambda x: x[0])
            self._edit_order_dirty = False
        all_edits = self._edits_in_order

        # Latest replay-window end per buffer row, per attribute (float64 so
        # timestamps round-trip exactly).
        #
        # Grown in place rather than cloned. This runs on every batch of a
        # render, and cloning the checkpoint allocated one float64 buffer per
        # attribute sized by that attribute's *total* row count every time --
        # 7.2 MB a batch on the reference scene across 9 attributes, and O(n^2)
        # over a render, since both the row count and the batch count grow with
        # the scene. Capacity doubles, so growth is amortized O(1) per row
        # added and a steady-state batch allocates nothing at all.
        #
        # Mutating the checkpoint in place is safe because the update is a
        # monotone max-assign: re-running a group over rows that already hold
        # its result recomputes the same end, so a partially-applied suffix is
        # simply re-applied. What it is *not* safe against is
        # preserving_authoring_state's rollback, which snapshots this dict --
        # that snapshot now copies (see there), moving one O(N) copy from every
        # batch to once per render.
        row_ends = {}
        capacity = self._row_ends_capacity
        for attr, timeline in self.attr_to_timeline.items():
            pointer = timeline.pointer
            previous = self._resolved_row_ends.get(attr)
            used = 0 if previous is None else previous.shape[0]
            rows = capacity.get(attr)
            if rows is None:
                # No backing store: first resolve for this attribute, or a
                # rollback dropped it and left the checkpoint values behind.
                rows = torch.full(
                    (max(pointer, used, 1),), -math.inf, dtype=torch.float64
                )
                if used:
                    rows[:used] = previous
            elif rows.shape[0] < pointer:
                grown = torch.full(
                    (max(pointer, rows.shape[0] * 2),), -math.inf, dtype=torch.float64
                )
                grown[:used] = rows[:used]
                rows = grown
            if used < pointer:
                rows[used:pointer] = -math.inf
            capacity[attr] = rows
            row_ends[attr] = rows[:pointer]

        i = min(self._resolved_prefix_count, len(all_edits))
        while i < len(all_edits):
            # Edits recorded by one function application are consecutive in
            # execution order; group them so they share one window.
            event = all_edits[i][2].event
            j = i + 1
            while (
                event is not None
                and j < len(all_edits)
                and all_edits[j][2].event is event
            ):
                j += 1
            group = all_edits[i:j]

            end = max(e.time.end for _, _, e in group)
            for _, attr, e in group:
                rows = e.indexes.view(-1)
                if rows.numel():
                    end = max(end, row_ends[attr][rows].max().item())
            for _, attr, e in group:
                e.replay_end = end
                row_ends[attr][e.indexes.view(-1)] = end
            if event is not None:
                event.replay_end = end
                slot = getattr(event, "_window_slot", None)
                if slot is None:
                    # Predates _window_slot (or was built outside add()): fall
                    # back to invalidating everything rather than guessing.
                    touched_slot = 0
                elif touched_slot is None or slot < touched_slot:
                    touched_slot = slot
            i = j

        self.function_timeline.invalidate_window_cache(touched_slot)
        self._resolved_prefix_count = len(all_edits)
        self._resolved_prefix_seq = all_edits[-1][0] if all_edits else -1
        self._resolved_row_ends = row_ends

    @contextlib.contextmanager
    def preserving_authoring_state(self, preserve_replay_resolution=True):
        """Render frames without leaving derived state on the timeline.

        Materializing frames resolves replay windows
        (:meth:`_resolve_replay_windows`), which freezes every edit's and
        event's context-rescaled end time into a plain ``replay_end`` float.
        That is correct once authoring is finished, but a render started from
        *inside* an unfinished context bakes in timestamps the enclosing
        contexts have not rescaled yet (a ``run_time`` rescales its block
        retroactively, on exit). Nothing invalidates those floats afterwards --
        only recording a new edit does -- so the stale, too-early ends survive
        into the next render, where :meth:`AttributeTimeline.prepare_for_queries`
        uses them verbatim as edit timestamps and the affected animations stop
        advancing early.

        Every render that leaves the Scene re-renderable wraps itself in this:
        :meth:`~algan.scene.Scene.save_frame`,
        :meth:`~algan.scene.Scene.show_frame`, and
        :meth:`~algan.scene.Scene.save_video` with ``reset=False``. The frames
        come out of the timeline as it stands, and the Scene carries nothing
        away from the render, so authoring can continue and render again --
        including from inside a block that has not finished yet. Lifespans
        created for the transient mobs a render builds are dropped for the same
        reason.
        """
        resolved = self._replay_windows_resolved
        edit_windows = []
        event_windows = []
        prefix_state = None
        if preserve_replay_resolution:
            edit_windows = [
                (edit, edit.replay_end)
                for timeline in self.attr_to_timeline.values()
                for edit in timeline.edits
            ]
            event_windows = [
                (event, event.replay_end)
                for event in self.function_timeline.function_applications
            ]
            prefix_state = (
                self._resolved_prefix_count,
                self._resolved_prefix_seq,
                # Copied, not aliased: _resolve_replay_windows grows this
                # checkpoint in place across the render's batches, so keeping
                # the dict alone would hand back views the render has since
                # overwritten. This is the one O(rows) copy that pays for
                # dropping the per-batch one -- once per render, not per batch.
                {attr: rows.clone() for attr, rows in self._resolved_row_ends.items()},
            )
        known_mob_ids = frozenset(self.mob_id_to_lifespan)
        try:
            yield self
        finally:
            if preserve_replay_resolution:
                for edit, replay_end in edit_windows:
                    edit.replay_end = replay_end
                for event, replay_end in event_windows:
                    event.replay_end = replay_end
                self._replay_windows_resolved = resolved
                (
                    self._resolved_prefix_count,
                    self._resolved_prefix_seq,
                    self._resolved_row_ends,
                ) = prefix_state
                # The restored checkpoint is a set of standalone copies, not
                # views into the backing store the render just mutated, so that
                # store has to go with it. The next resolve rebuilds it from
                # these values.
                self._row_ends_capacity = {}
                # Prepared edit dictionaries and CSR indexes embed the
                # transient replay_end values resolved by this render.  They
                # must not survive after those values are rolled back; the
                # next render will rebuild them from the final rescaled times.
                for timeline in self.attr_to_timeline.values():
                    timeline.invalidate_prepared_queries()
                # Any edit or event recorded during the render (there should
                # be none) is outside the snapshot, so drop cached windows
                # rather than trust them.
                self.function_timeline.invalidate_window_cache()
            for mob_id in [
                mob_id
                for mob_id in self.mob_id_to_lifespan
                if mob_id not in known_mob_ids
            ]:
                del self.mob_id_to_lifespan[mob_id]

    @staticmethod
    def _collect_mob_ids(values):
        """Collect mob ids reachable from roots/event arguments.

        Only ordinary Python containers are traversed; tensors and arbitrary
        iterables are deliberately treated as leaves. This keeps dependency
        discovery cheap even when an animated argument contains large data.
        """
        mob_ids = set()
        seen = set()
        stack = list(values)
        while stack:
            value = stack.pop()
            oid = id(value)
            if oid in seen:
                continue
            seen.add(oid)
            if hasattr(value, "lifespan") and hasattr(value, "id"):
                mob_ids.add(value.id)
                children = getattr(value, "children", None)
                if children is not None:
                    stack.extend(children)
                else:
                    get_descendants = getattr(value, "get_descendants", None)
                    if get_descendants is None:
                        continue
                    with contextlib.suppress(AttributeError, TypeError):
                        stack.extend(get_descendants(include_self=False))
                continue
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                stack.extend(value)
        return mob_ids

    def _active_mob_ids(self, active_mobs, functions, updaters):
        """Resolve a conservative working set for one frame window.

        Built-in animation dependencies come from their caller and arguments.
        Updaters additionally retain the Mob reads/writes observed by their
        dependency trace, including dependencies discovered lazily in a
        time-dependent branch.
        """
        if active_mobs is None:
            return None
        custom_entry_points = {"animate_function", "animate_function_of_time"}
        for event in functions:
            fn = event.function
            if getattr(fn, "__name__", "") in custom_entry_points or not getattr(
                fn, "__module__", ""
            ).startswith("algan."):
                return None
        roots = list(active_mobs)
        for event in functions:
            roots.extend((event.caller, event.kwargs, event.animated_args))
        mob_ids = self._collect_mob_ids(roots)
        for event in updaters:
            mob_ids.update(event.dependency_mob_ids)
        return mob_ids

    def set_state_to_times(self, times, active_mobs=None):
        """Materialize animated state at ``times``.

        ``active_mobs`` is the render window's conservative actor working set.
        When supplied, built-in animations query only rows reachable from that
        set while keeping the full global-row buffer layout used by replay.
        Omitting it preserves the original all-row behavior for public callers.

        Replay runs inside a non-recording context, because it calls each
        recorded function's *undecorated* body: the ``record_funcs=False`` wrap
        that ``animated_function`` normally applies is absent, so a recorded
        function whose body calls another animated function
        (``Cylinder.set_start_point`` -> ``_move_between_points`` ->
        ``move_to``) would record a **new** event on every replay -- growing the
        timeline without bound as batches are prepared, and re-resolving replay
        windows every time. A render never saw this because its batch loop runs
        inside the same context (:meth:`~algan.render_loop.RenderLoopMixin
        .batch_prep_context`); doing it here means every caller is safe,
        including the benchmarks and probes that drive prep directly.

        The context matches the render's exactly, so entering it inside a
        render is inert -- the values are already set and inherited.
        """
        # Deferred: animation_contexts imports TimelineSpan from this module.
        from algan.animation_timeline.animation_contexts import Off

        manager = None
        for mob in active_mobs or ():
            manager = getattr(mob, "animation_manager", None)
            if manager is not None:
                break
        with Off(
            record_attr_modifications=False,
            record_funcs=False,
            priority_level=math.inf,
            animation_manager=manager,
        ):
            return self._replay_state_to_times(times, active_mobs)

    def _replay_state_to_times(self, times, active_mobs=None):
        """:meth:`set_state_to_times`' body, run under its non-recording context."""
        self._resolve_replay_windows()
        functions = self.function_timeline.get_functions_for_times(times)
        updaters = self.function_timeline.get_updaters_for_times(times)
        active_mob_ids = self._active_mob_ids(active_mobs, functions, updaters)
        self._materialization_times = times
        self._materialized_mob_ids = (
            None if active_mob_ids is None else set(active_mob_ids)
        )
        replay_rows = {}
        if active_mob_ids is not None:
            for function in functions:
                for attr_name, _, _, inds in function.recorded_edits:
                    replay_rows.setdefault(attr_name, []).append(inds.view(-1))
        for attr_name, timeline in self.attr_to_timeline.items():
            rows = replay_rows.get(attr_name)
            extra_rows = torch.cat(rows) if rows else None
            timeline.rematerialize_state_at_times(
                times, active_mob_ids, extra_rows=extra_rows
            )

        for f in functions:
            s = f.time.start
            e = f.time.end
            replay_end = _replay_window_end(f)
            active_time_inds = ((s <= times) & (times < replay_end)).nonzero()
            if active_time_inds.numel() == 0:
                continue
            time_selector = (
                active_time_inds
                if _opt_disabled("timeslice")
                else _contiguous_time_selector(active_time_inds)
            )
            for timeline in self.attr_to_timeline.values():
                timeline.set_active_time_inds(time_selector)

            elapsed = times[active_time_inds.squeeze(-1)] - s
            a = (elapsed / (e - s + 1e-6)).view(-1, 1, 1)
            if replay_end > e:
                # Frames past the function's own end (reachable only while an
                # earlier-executed animation overlapping this one's rows is
                # still running) replay it at its final parameters, keeping
                # its finished contribution in the rebuilt state.
                duration = e - s
                a = torch.where(
                    elapsed.view(-1, 1, 1) >= duration, torch.ones_like(a), a
                )
                elapsed = elapsed.clamp(max=duration)
            a = f.rate_func(a)

            kwargs = dict(f.kwargs.items())
            for k in f.animated_args:
                kwargs[k] = torch.lerp(
                    cast_to_tensor(f.animated_args[k]), f.kwargs[k], a
                )
            if TIME_PARAMETER_NAME in kwargs:
                # Functions of time (animate_function_of_time) receive the
                # per-frame elapsed seconds instead of an interpolated value.
                kwargs[TIME_PARAMETER_NAME] = elapsed.view(-1, 1, 1)

            previous_event = self._active_replay_event
            previous_edit_index = self._active_replay_edit_index
            self._active_replay_event = f
            self._active_replay_edit_index = 0
            try:
                f.function(f.caller, **kwargs)
            finally:
                self._active_replay_event = previous_event
                self._active_replay_edit_index = previous_edit_index

        for f in updaters:
            active_time_inds = (
                (f.time.start <= times) & (times < f.time.end)
            ).nonzero()
            if active_time_inds.numel() == 0:
                continue
            groups = [((), active_time_inds)]
            if f._history_clones:
                grouped = {}
                flat_inds = active_time_inds.squeeze(-1)
                queried_times = times[flat_inds].detach().cpu().tolist()
                for index, time in zip(flat_inds.tolist(), queried_times):
                    signature = f.replacement_signature_at(time)
                    key = tuple(
                        (id(original), id(clone)) for original, clone in signature
                    )
                    grouped.setdefault(key, [signature, []])[1].append(index)
                groups = [
                    (
                        signature,
                        torch.tensor(
                            indexes,
                            dtype=torch.long,
                            device=active_time_inds.device,
                        ).unsqueeze(-1),
                    )
                    for signature, indexes in grouped.values()
                ]

            for signature, group_time_inds in groups:
                group_selector = (
                    group_time_inds
                    if _opt_disabled("timeslice")
                    else _contiguous_time_selector(group_time_inds)
                )
                for timeline in self.attr_to_timeline.values():
                    timeline.set_active_time_inds(group_selector)
                elapsed = times[group_time_inds.squeeze(-1)] - f.time.start
                caller, args, kwargs = f.invocation_for_signature(signature)
                previous_trace = self.begin_updater_dependency_trace(f)
                try:
                    f.function(
                        caller,
                        elapsed.view(-1, 1, 1),
                        *args,
                        **kwargs,
                    )
                finally:
                    self.end_updater_dependency_trace(previous_trace)

        for timeline in self.attr_to_timeline.values():
            timeline.set_active_time_inds(slice(None, None, None))
        self._materialization_times = None
        self._materialized_mob_ids = None
        return self

    def clear_buffers(self):
        for t in self.attr_to_timeline.values():
            t.clear_buffers()


class TimelineManager(AnimationTimeline):
    """Per-scene animation timeline.

    Unlike the historical singleton accessor, this is an ordinary class. Each
    :class:`~algan.scene.Scene` constructs and owns one instance.
    """

    pass
