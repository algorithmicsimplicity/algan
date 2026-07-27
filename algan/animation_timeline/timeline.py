import math

import torch

from algan.animation_timeline.utils_taichi import (
    _query_selected_state_from_edits,
    _query_state_from_edits,
)
from algan.utils.tensor_utils import cast_to_tensor


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
#: :meth:`~algan.animation.animatable.Animatable.get_attr_inds`) are
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


def bump_structure_version():
    STRUCTURE_VERSION[0] += 1


def bump_hierarchy_version():
    """Bump both the hierarchy and structure versions. Call this (instead of
    :func:`bump_structure_version`) for any change to a mob's ``children``: it
    invalidates both the descendant-list cache and the descendant-row-index
    caches, since a hierarchy change alters both."""
    HIERARCHY_VERSION[0] += 1
    STRUCTURE_VERSION[0] += 1


_OPT_DISABLED = None


def _opt_disabled(name):
    """Bisect aid: ALGAN_OPT_DISABLE=fastpath,ranges,desccache,windows disables
    individual animation-prep optimizations (read once, first use)."""
    global _OPT_DISABLED
    if _OPT_DISABLED is None:
        import os
        _OPT_DISABLED = frozenset(
            os.environ.get("ALGAN_OPT_DISABLE", "").split(","))
    return name in _OPT_DISABLED


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
        self.numel = (sum(e - b for b, e in pairs) if pairs is not None
                      else tensor.numel())
        self._tensor = tensor

    def tensor(self):
        if self._tensor is None:
            if len(self.pairs) == 1:
                b, e = self.pairs[0]
                self._tensor = torch.arange(b, e)
            else:
                self._tensor = torch.cat(
                    [torch.arange(b, e) for b, e in self.pairs])
        return self._tensor

    @staticmethod
    def from_contiguous_blocks(inds_list):
        """Build from per-mob index tensors, each of which is a contiguous
        arange (how ``AttributeTimeline.add`` allocates). Returns None when a
        block is not contiguous (caller falls back to plain concatenation)."""
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
        tensor indexing/conversion. ``runs`` must contain no empty runs."""
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

    __slots__ = ("start", "end")

    def __init__(self):
        self.start = _never
        self.end = _never


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
    earlier-executed edits (never shrunk below the context end)."""
    end = f.time.end
    if f.replay_end is None:
        return end
    return max(f.replay_end, end)


def _prepare_array_state_queries(times, N, edits):
    """Build the device-side CSR representation of an attribute's edits.

    The edit log is immutable while frames are rendered.  Keeping this work
    separate from the actual time query lets :class:`AttributeTimeline` cache
    the concatenation, stable sort and row-boundary construction across frame
    batches.
    """
    device = times.device
    edit_timestamps = torch.tensor(
        [edit['timestamp'] for edit in edits], dtype=times.dtype, device=device)
    edit_sizes = torch.tensor(
        [edit['indexes'].shape[0] for edit in edits],
        dtype=torch.int64, device=device)
    flat_indices = torch.cat([edit['indexes'].to(device) for edit in edits])
    flat_values = torch.cat([edit['values'].to(device) for edit in edits])
    flat_edit_ids = torch.repeat_interleave(edit_sizes).to(torch.int32)
    perm = torch.argsort(flat_indices, stable=True)
    sorted_indices = flat_indices[perm]
    sorted_edit_ids = flat_edit_ids[perm]
    sorted_values = flat_values[perm]
    grid = torch.arange(N + 1, dtype=torch.int64, device=device)
    head = torch.searchsorted(sorted_indices, grid)
    return head, sorted_edit_ids, edit_timestamps, sorted_values


def generate_array_states_taichi(times, N, edits, *, active_rows=None,
                                 prepared=None):
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
        binary search in _query_state_from_edits relies on it.
        prepare_for_queries guarantees this by passing edits in execution
        order with their replay-extended end times.
    """
    device = times.device
    T = times.shape[0]

    if len(edits) == 0:
        return torch.zeros((T, N, 1), dtype=torch.float32, device=device)

    if prepared is None:
        prepared = _prepare_array_state_queries(times, N, edits)
    head, sorted_edit_ids, edit_timestamps, sorted_values = prepared
    D = sorted_values.shape[1]
    dtype = sorted_values.dtype

    if active_rows is None:
        out = torch.empty((T, N, D), dtype=dtype, device=device)
        _query_state_from_edits(
            times, head, sorted_edit_ids, edit_timestamps, sorted_values, out)
    else:
        # Keep the full global row layout for animated-function replay. Rows
        # outside this window's working set stay zero and are never consumed by
        # primitive preparation for this batch.
        out = torch.zeros((T, N, D), dtype=dtype, device=device)
        active_rows = active_rows.to(device=device, dtype=torch.int64)
        if active_rows.numel():
            _query_selected_state_from_edits(
                times, active_rows, head, sorted_edit_ids, edit_timestamps,
                sorted_values, out)

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

    def __init__(self, channels, buffer_size=256, attr_name=None, record_end_points=False):
        self.attr_name = attr_name
        self.record_end_points = record_end_points
        self.current_state = torch.empty((1, buffer_size, channels)) # latest state after all edits.
        self.active_state = self.current_state # pointer to active state used to fulfil get requests.
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        self.pointer = 0
        self.edits = []
        self._is_ready_for_queries = False
        self._query_cache = {}
        self.mob_id_to_inds = dict()
        self.mob_id_to_ranges = dict()
        self.mob_id_to_starts = dict()
        self.mob_id_to_ends = dict()

    def set_start_point(self, mob, starts):
        self.mob_id_to_starts[mob.id] = starts

    def set_end_point(self, mob, ends):
        self.mob_id_to_ends[mob.id] = ends

    def get_current_values(self):
        return self.current_state[:, :self.pointer]

    def get(self, key, copy=True):
        if isinstance(key, RowRanges):
            if (not _opt_disabled("ranges")
                    and key.pairs is not None and len(key.pairs) == 1):
                # Contiguous rows: slice instead of index-gather. The clone
                # keeps the copy semantics of advanced indexing (callers may
                # mutate the result in place); read-only callers that only feed
                # the value into out-of-place arithmetic pass copy=False to skip
                # it (the dominant cost during mob construction).
                b, e = key.pairs[0]
                block = self.active_state[:, b:e]
                t = self.active_time_inds
                if isinstance(t, slice):
                    return block[t].clone() if copy else block[t]
                return block[t.view(-1)]
            key = key.tensor()
        return self.active_state[self.active_time_inds, key]

    def modify(self, key, value):
        # Replaying an animation modifies the temporary materialized buffer,
        # not the recorded edit log/current state.  Invalidating the prepared
        # CSR query data here made every frame batch rebuild and re-sort the
        # identical edit history. Recording-time writes still invalidate it.
        if self.active_state is self.current_state:
            self._is_ready_for_queries = False
            self._query_cache.clear()
        if isinstance(key, RowRanges):
            if (not _opt_disabled("ranges")
                    and key.pairs is not None and len(key.pairs) == 1):
                # Contiguous rows: slice-assign instead of index-scatter.
                b, e = key.pairs[0]
                t = self.active_time_inds
                if isinstance(t, slice):
                    self.active_state[t, b:e] = value
                else:
                    self.active_state[t.view(-1), b:e] = value
                return self
            key = key.tensor()
        self.active_state[self.active_time_inds, key] = value
        return self

    def reassign_inds(self, mob_id, inds):
        """Point ``mob_id`` at ``inds``, invalidating the cached
        :class:`RowRanges` (:meth:`ranges_for`) so callers don't read stale
        rows. Use this for any direct row hand-over that bypasses :meth:`add`
        (e.g. :meth:`Mob.detach_history`'s history swap). Bumps the structure
        version so per-mob descendant-row caches (``Mob._attr_inds_cache``) that
        may reference the old ownership are rebuilt."""
        self.mob_id_to_inds[mob_id] = inds
        self.mob_id_to_ranges.pop(mob_id, None)
        bump_structure_version()

    def drop_mob(self, mob_id):
        """Forget ``mob_id``'s rows and cached ranges."""
        self.mob_id_to_inds.pop(mob_id, None)
        self.mob_id_to_ranges.pop(mob_id, None)
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
        if (not overwrite) and (mob_id in self.mob_id_to_inds):
            return

        self._is_ready_for_queries = False
        self._query_cache.clear()

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
            new_buffer[:, :self.pointer] = self.get_current_values()
            self.current_state = new_buffer
            self.active_state = self.current_state
        self.current_state[:, self.pointer:new_pointer] = values
        inds = torch.arange(self.pointer, new_pointer)
        self.mob_id_to_inds[mob_id] = inds
        # The block is contiguous by construction, so cache its single-run
        # RowRanges directly instead of re-deriving it (with tensor->int
        # conversions) on the first ranges_for() query.
        self.mob_id_to_ranges[mob_id] = RowRanges([(self.pointer, new_pointer)])
        self.pointer = new_pointer
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
        self._edits_sorted = [{'indexes': e.indexes.view(-1), 'values': e.values.squeeze(0),
                               'timestamp': e.replay_end if e.replay_end is not None else e.time.end}
                              for e in self.edits]
        self._edits_sorted.append({'indexes': torch.arange(self.pointer),
                                   'values': self.current_state[:, :self.pointer].squeeze(0),
                                   'timestamp': math.inf})
        self._query_cache.clear()

        if not self.record_end_points:
            return self
        self._end_points = torch.full((1, self.pointer + 1, 2), 1e12)
        for mob_id in self.mob_id_to_starts:
            inds = self.mob_id_to_inds[mob_id]
            self._end_points[:,inds,0] = self.mob_id_to_starts[mob_id].start()
        for mob_id in self.mob_id_to_ends:
            inds = self.mob_id_to_inds[mob_id]
            self._end_points[:,inds,1] = self.mob_id_to_ends[mob_id].end()
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
        compressed = (RowRanges.from_runs(runs).tensor() if runs
                      else torch.empty((0,), dtype=torch.long))
        if loose:
            return torch.unique(torch.cat([compressed, *loose]), sorted=True)
        return compressed

    def _prepared_queries(self, times):
        key = (str(times.device), times.dtype)
        prepared = self._query_cache.get(key)
        if prepared is None:
            prepared = _prepare_array_state_queries(
                times, self.pointer + 1, self._edits_sorted)
            self._query_cache[key] = prepared
        return prepared

    def rematerialize_state_at_times(self, times, active_mob_ids=None,
                                     extra_rows=None):
        self.prepare_for_queries()
        active_rows = (None if active_mob_ids is None
                       else self.rows_for_mob_ids(active_mob_ids))
        if active_rows is not None and extra_rows is not None:
            extra_rows = extra_rows.view(-1)
            if extra_rows.numel():
                active_rows = torch.unique(
                    torch.cat((active_rows, extra_rows)), sorted=True)
        self.active_state = generate_array_states_taichi(
            times, self.pointer + 1, self._edits_sorted,
            active_rows=active_rows,
            prepared=self._prepared_queries(times))
        if self.record_end_points:
            t = times.view(-1,1)
            if active_rows is None:
                self.active_state *= (
                    (self._end_points[..., 0] <= t)
                    & (t < self._end_points[..., 1])
                ).unsqueeze(-1)
            elif active_rows.numel():
                rows = active_rows.to(self.active_state.device)
                endpoint = self._end_points[:, rows].to(
                    self.active_state.device)
                mask = ((endpoint[..., 0] <= t)
                        & (t < endpoint[..., 1])).unsqueeze(-1)
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
        queried = generate_array_states_taichi(
            times,
            self.pointer + 1,
            self._edits_sorted,
            active_rows=rows,
            prepared=self._prepared_queries(times),
        )
        device_rows = rows.to(self.active_state.device)
        self.active_state[:, device_rows] = queried[:, device_rows]
        if self.record_end_points:
            t = times.view(-1, 1)
            endpoint = self._end_points[:, rows].to(self.active_state.device)
            mask = ((endpoint[..., 0] <= t) & (t < endpoint[..., 1])).unsqueeze(-1)
            self.active_state[:, device_rows] *= mask
        return self

    def clear_buffers(self):
        self.active_state = self.current_state
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        self._is_ready_for_queries = False
        self._query_cache.clear()
        return self

class TimelineEvent:
    def __init__(self, time, span):
        self.span = span
        self._time = time

    def __call__(self):
        return self.time

    @property
    def end(self):
        return self.time

    @property
    def time(self):
        return self.span.get_rescaled_time(self._time)

    @time.setter
    def time(self, time):
        self._time = time


class TimelineSpan:
    def __init__(self, start_time=0, end_time=0, current_time=0):
        self._rescaled_start = start_time
        self._rescaled_end = end_time
        self.original_start = start_time
        self.original_end = end_time
        self.current_time = current_time

    def __call__(self):
        return self.original_start

    def rescale(self, new_start, ratio):
        self.start = (self.start - new_start) * ratio + new_start
        self.end = (self.end - new_start) * ratio + new_start

    def get_rescaled_time(self, t):
        a = (t - self.original_start) / max(self.original_end - self.original_start, 1e-6)
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
        self._rescaled_start = value

    @property
    def end(self):
        return self._rescaled_end

    @end.setter
    def end(self, value):
        self._rescaled_end = value

class FunctionApplicationEvent:
    def __init__(self, function, caller, animated_args=None, kwargs=None, rate_func=None, time=None):
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


class UpdaterSpan:
    """The [added, removed) interval of one updater. ``start``/``end`` expose
    the (lazily rescaled) timestamps as numbers, matching the protocol of the
    context timespans carried by ordinary :class:`FunctionApplicationEvent` s.
    An updater that was never removed ends at :data:`UPDATER_FOREVER`."""

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
    equal to ``t - time.start``."""

    __slots__ = (
        "function",
        "caller",
        "args",
        "kwargs",
        "time",
        "dependency_mob_ids",
    )

    def __init__(self, function, caller, args, kwargs, time):
        self.function = function
        self.caller = caller
        self.args = args
        self.kwargs = kwargs
        self.time = time
        self.dependency_mob_ids = set()


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

    def add(self, function_application):
        self.function_applications.append(function_application)

    def add_updater(self, updater):
        self.updaters.append(updater)

    def invalidate_window_cache(self):
        self._window_cache = None

    def _windows(self):
        cache = self._window_cache
        if cache is None or cache[0] != len(self.function_applications):
            # float32 to match the dtype the per-event scalar comparisons
            # used (python-float scalars compare in the tensor's dtype).
            starts = torch.tensor(
                [f.time.start for f in self.function_applications],
                dtype=torch.float32)
            ends = torch.tensor(
                [_replay_window_end(f) for f in self.function_applications],
                dtype=torch.float32)
            cache = self._window_cache = (
                len(self.function_applications), starts, ends)
        return cache[1], cache[2]

    def get_functions_for_times(self, times):
        if not self.function_applications:
            return []
        if _opt_disabled("windows"):
            return [f for f in self.function_applications
                    if ((f.time.start <= times)
                        & (times < _replay_window_end(f))).any()]
        starts, ends = self._windows()
        t = times.view(1, -1)
        active = ((starts.view(-1, 1) <= t) & (t < ends.view(-1, 1))).any(1)
        return [self.function_applications[i]
                for i in active.nonzero().view(-1).tolist()]

    def get_updaters_for_times(self, times):
        return [f for f in self.updaters if ((f.time.start <= times) &
                (times < f.time.end)).any()]


class AnimationTimeline:
    def __init__(self):
        self.attr_to_timeline = dict()
        self.function_timeline = FunctionTimeline()
        self.mob_id_to_lifespan = dict()
        # Edit attribution state: a global execution counter for edits, the
        # function application the currently-executing animated function was
        # recorded as (edits made while it runs attach to it), and the most
        # recently recorded function application (consumed by the
        # animated_function wrapper to scope the former).
        self._edit_seq = 0
        self._active_edit_event = None
        self.last_recorded_event = None
        self._replay_windows_resolved = True
        self._active_replay_event = None
        self._active_replay_edit_index = 0
        self._active_updater_trace = None
        self._materialization_times = None
        self._materialized_mob_ids = None

    def set_active_edit_event(self, event):
        """Set the function application that subsequently recorded attribute
        edits are attributed to, returning the previous one (so callers can
        restore it)."""
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

    def trace_updater_mob_access(self, mob, include_descendants=False):
        """Record one updater Mob access and materialize newly seen rows.

        The initial updater invocation records the common dependency set at
        authoring time. The same trace remains active during materialization,
        so time-dependent branches can safely discover extra Mobs on demand.
        """
        event = self._active_updater_trace
        if event is None:
            return
        if include_descendants:
            mob_ids = self._collect_mob_ids((mob,))
        else:
            mob_ids = {mob.id}
        event.dependency_mob_ids.update(mob_ids)

        if self._materialized_mob_ids is None or self._materialization_times is None:
            return
        missing = mob_ids.difference(self._materialized_mob_ids)
        if not missing:
            return
        for timeline in self.attr_to_timeline.values():
            rows = timeline.rows_for_mob_ids(missing)
            timeline.materialize_additional_rows(self._materialization_times, rows)
        self._materialized_mob_ids.update(missing)

    def get_lifespan(self, mob_id):
        """The :class:`Lifespan` of the mob with the given id, created on
        first access (start = end = "never")."""
        lifespan = self.mob_id_to_lifespan.get(mob_id)
        if lifespan is None:
            lifespan = self.mob_id_to_lifespan[mob_id] = Lifespan()
        return lifespan

    def register_spawn(self, mob, lifespan):
        # Visibility masking: the opacity timeline zeroes a mob's opacity
        # outside its [spawn, despawn) interval when materializing state.
        timeline = self.attr_to_timeline.get('opacity')
        if timeline is not None:
            timeline.set_start_point(mob, lifespan)

    def register_despawn(self, mob, lifespan):
        timeline = self.attr_to_timeline.get('opacity')
        if timeline is not None:
            timeline.set_end_point(mob, lifespan)

    def add_mob_attr(self, mob, attr, value, add_mob=True):
        if attr not in self.attr_to_timeline:
            self.attr_to_timeline[attr] = AttributeTimeline(value.shape[-1],
                                                            attr_name=attr,
                                                            record_end_points=attr=='opacity')
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

    def record_function(self, function, caller, animated_args, kwargs, animation_context):
        c = animation_context
        self.last_recorded_event = None
        if c.run_time_unit <= 0 or not c.record_funcs:
            return kwargs
        rate_func = c.rate_func
        rate_func_compose = c.rate_func_compose
        rf = rate_func
        if rate_func_compose is not None:
            rf = lambda x, rf=rate_func, rfc=rate_func_compose: rf(rfc(x))
        event = FunctionApplicationEvent(
            function, caller, animated_args, kwargs, rf, c.timespan)
        self.function_timeline.add(event)
        self.last_recorded_event = event
        return kwargs

    def record_updater(self, function, caller, args, kwargs, animation_context):
        """Register an updater starting at the context's current time and
        lasting until :meth:`end_updater` (or forever). Returns its id."""
        span = UpdaterSpan(animation_context.timespan.get_current_time())
        event = UpdaterEvent(function, caller, args, kwargs, span)
        event.dependency_mob_ids.update(
            self._collect_mob_ids((caller, args, kwargs))
        )
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

    def modify_attribute_and_record(self, attr_name, mob_id,
                                    include_descendants, mob_inds,
                                    new_value, time):
        timeline = self.attr_to_timeline[attr_name]
        self._edit_seq += 1
        event = self._active_edit_event
        edit = timeline.record(mob_inds, new_value, time, self._edit_seq, event)
        if event is not None:
            event.recorded_edits.append(
                (attr_name, mob_id, include_descendants, edit.indexes))
        self._replay_windows_resolved = False
        return self

    def replay_inds(self, attr_name, mob_id, include_descendants,
                    consume=False):
        """Return the next recorded row set while replaying one function."""
        event = self._active_replay_event
        if event is None:
            return None
        index = self._active_replay_edit_index
        if index >= len(event.recorded_edits):
            return None
        edit_attr, edit_mob_id, edit_recursive, inds = event.recorded_edits[index]
        if (edit_attr != attr_name or edit_mob_id != mob_id
                or edit_recursive != include_descendants):
            return None
        if consume:
            self._active_replay_edit_index += 1
        return inds

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
        # Replay-window ends feed the cached event-window tensors.
        self.function_timeline.invalidate_window_cache()

        all_edits = []
        for attr, timeline in self.attr_to_timeline.items():
            all_edits.extend((e.seq, attr, e) for e in timeline.edits)
        all_edits.sort(key=lambda x: x[0])

        # Latest replay-window end per buffer row, per attribute (float64 so
        # timestamps round-trip exactly).
        row_ends = {attr: torch.full((timeline.pointer,), -math.inf, dtype=torch.float64)
                    for attr, timeline in self.attr_to_timeline.items()}

        i = 0
        while i < len(all_edits):
            # Edits recorded by one function application are consecutive in
            # execution order; group them so they share one window.
            event = all_edits[i][2].event
            j = i + 1
            while event is not None and j < len(all_edits) and all_edits[j][2].event is event:
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
            i = j

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
                get_descendants = getattr(value, "get_descendants", None)
                if get_descendants is not None:
                    try:
                        stack.extend(get_descendants(include_self=False))
                    except (AttributeError, TypeError):
                        pass
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
            if (getattr(fn, "__name__", "") in custom_entry_points
                    or not getattr(fn, "__module__", "").startswith("algan.")):
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
        """
        self._resolve_replay_windows()
        functions = self.function_timeline.get_functions_for_times(times)
        updaters = self.function_timeline.get_updaters_for_times(times)
        active_mob_ids = self._active_mob_ids(
            active_mobs, functions, updaters)
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
                times, active_mob_ids, extra_rows=extra_rows)

        for f in functions:
            s = f.time.start
            e = f.time.end
            replay_end = _replay_window_end(f)
            active_time_inds = ((s <= times) & (times < replay_end)).nonzero()
            if active_time_inds.numel() == 0:
                continue
            for timeline in self.attr_to_timeline.values():
                timeline.set_active_time_inds(active_time_inds)

            elapsed = times[active_time_inds.squeeze(-1)] - s
            a = (elapsed / (e - s + 1e-6)).view(-1, 1, 1)
            if replay_end > e:
                # Frames past the function's own end (reachable only while an
                # earlier-executed animation overlapping this one's rows is
                # still running) replay it at its final parameters, keeping
                # its finished contribution in the rebuilt state.
                duration = e - s
                a = torch.where(elapsed.view(-1, 1, 1) >= duration, torch.ones_like(a), a)
                elapsed = elapsed.clamp(max=duration)
            a = f.rate_func(a)

            kwargs = {k: v for k, v in f.kwargs.items()}
            for k in f.animated_args:
                kwargs[k] = torch.lerp(cast_to_tensor(f.animated_args[k]), f.kwargs[k], a)
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
            active_time_inds = ((f.time.start <= times) & (times < f.time.end)).nonzero()
            if active_time_inds.numel() == 0:
                continue
            for timeline in self.attr_to_timeline.values():
                timeline.set_active_time_inds(active_time_inds)
            elapsed = times[active_time_inds.squeeze(-1)] - f.time.start
            previous_trace = self.begin_updater_dependency_trace(f)
            try:
                f.function(
                    f.caller,
                    elapsed.view(-1, 1, 1),
                    *f.args,
                    **f.kwargs,
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
