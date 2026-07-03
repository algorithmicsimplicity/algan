"""Global batched animation state.

This module replaces the per-Mob Python-object storage of animated attributes
with one global, row-allocated buffer per attribute plus a single global
modification log, so that

* every Mob's animated attributes live in rows of a shared tensor (allocated
  once at import time from a pre-allocated arena),
* a hierarchy-wide attribute write (e.g. ``group.move(UP)``) is one batched
  tensor operation over the row indices of all descendants instead of a
  Python recursion through the Mob hierarchy, and
* render-time state materialization is one batched binary search
  (``torch.searchsorted``) over the sorted modification log per attribute,
  instead of a Python walk over every Mob's private history.

Layout
------
``GlobalAnimationState`` owns one ``AttributeBuffer`` per (attribute name,
width).  A Mob allocates a contiguous block of rows per attribute at creation
(``RowBlock``); rows can grow later (batch expansion) by appending extra
ranges, so writes accept a list of (start, end) ranges.

Every attribute modification appends one entry to the buffer's log: the
affected row indices, the *previous* values of those rows, and the (lazily
evaluated) timestamp callables of the enclosing animation context.  This
mirrors the old per-Mob ``ModificationHistory.attribute_modifications`` --
which also stored the pre-write value per modification -- just globally and
row-indexed.

At render time, ``ensure_window(start, end)`` materializes the state of all
modified rows for a range of frames in one batched pass per attribute
(``MaterializedWindow``).  Rows with no recorded modification are constant
over all time and are served as broadcast views of the live buffer, so the
dense per-frame tensors only cover rows that actually animate.  Animated
functions are then re-executed on top of this window with interpolated
parameters (unchanged protocol, see ``Animatable.set_state_full``), writing
through the same row-indexed accessors.
"""

from __future__ import annotations

import os

import torch


def _pow2_at_least(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class RowBlock:
    """The set of rows in one :class:`AttributeBuffer` owned by one mob's data.

    Usually a single contiguous range.  Batch growth appends further ranges
    instead of relocating (relocation would orphan the row indices recorded in
    the modification log), so all consumers accept a list of ranges.
    """

    __slots__ = ("buffer", "ranges", "_indices")

    def __init__(self, buffer: "AttributeBuffer", start: int, end: int):
        self.buffer = buffer
        self.ranges = [(start, end)]
        self._indices = None

    @property
    def size(self) -> int:
        return sum(e - s for s, e in self.ranges)

    @property
    def contiguous(self) -> bool:
        return len(self.ranges) == 1

    @property
    def indices(self) -> torch.Tensor:
        if self._indices is None:
            self._indices = torch.cat(
                [torch.arange(s, e) for s, e in self.ranges]
            )
        return self._indices

    def append_range(self, start: int, end: int):
        self.ranges.append((start, end))
        self._indices = None

    def read(self, snapshot: bool = False) -> torch.Tensor:
        """Current values of these rows, shape [size, W].

        With ``snapshot=True`` the result is guaranteed to be an independent
        copy: attribute reads must not retro-actively change when the buffer
        rows are written later (callers capture a value, write something else,
        then use the captured value -- e.g. the spawn fade-in).
        """
        if len(self.ranges) == 1:
            s, e = self.ranges[0]
            value = self.buffer.values[s:e]
            return value.clone() if snapshot else value
        return self.buffer.values[self.indices]

    def write(self, value: torch.Tensor):
        self.buffer.write_rows(self, value)


class AttributeBuffer:
    """Global storage plus modification log for one animated attribute."""

    def __init__(self, state: "GlobalAnimationState", name: str, width: int):
        self.state = state
        self.name = name
        self.width = width
        self.values = state._arena_alloc(state.initial_rows, width)
        self.capacity = self.values.shape[0]
        self.size = 0
        # Modification log, columnar. Each entry is one batched write:
        # the affected row indices, the values those rows held *before* the
        # write, and the context's start/end time callables (evaluated only
        # when a render session begins, since animation contexts rescale
        # timestamps retroactively).
        self.mod_rows: list[torch.Tensor] = []
        self.mod_values: list[torch.Tensor] = []
        self.mod_starts: list = []
        self.mod_ends: list = []
        # Sorted-log cache for the current render session.
        self._session_token = -1
        self._sorted = None

    # -- row allocation ----------------------------------------------------

    def alloc(self, n: int) -> RowBlock:
        start = self._alloc_range(n)
        return RowBlock(self, start, start + n)

    def grow(self, block: RowBlock, n_extra: int, init: torch.Tensor,
             inherit_history_from_row: int | None = None):
        """Append ``n_extra`` rows to ``block``, initialised to ``init``
        ([n_extra, W] or broadcastable).  If ``inherit_history_from_row`` is
        given, the new rows replay that row's modification history (this
        matches the old pad-with-last batch expansion, where the expanded
        columns shared the source column's recorded history)."""
        start = self._alloc_range(n_extra)
        self.values[start:start + n_extra] = init
        if inherit_history_from_row is not None:
            self._duplicate_row_history(
                inherit_history_from_row, torch.arange(start, start + n_extra)
            )
        block.append_range(start, start + n_extra)

    def _alloc_range(self, n: int) -> int:
        if self.size + n > self.capacity:
            new_capacity = max(self.capacity * 2, self.size + n)
            new_values = self.state._arena_alloc(new_capacity, self.width)
            new_values[: self.size] = self.values[: self.size]
            self.values = new_values
            self.capacity = new_values.shape[0]
        start = self.size
        self.size += n
        return start

    # -- reads / writes ----------------------------------------------------

    def write_rows(self, rows, value: torch.Tensor):
        """Write ``value`` to ``rows`` (RowBlock or index tensor) in place."""
        value = value.reshape(-1, value.shape[-1]) if value.dim() > 2 else value
        if isinstance(value, torch.Tensor) and value.device != self.values.device:
            value = value.to(self.values.device)
        # Writing a slice of this very buffer back into it (self-assignment)
        # must not alias mid-copy.
        if (
            isinstance(value, torch.Tensor)
            and value.untyped_storage().data_ptr()
            == self.values.untyped_storage().data_ptr()
        ):
            value = value.clone()
        if isinstance(rows, RowBlock):
            if len(rows.ranges) == 1:
                s, e = rows.ranges[0]
                self.values[s:e] = value
                return
            rows = rows.indices
        self.values[rows] = value

    def read_rows(self, rows) -> torch.Tensor:
        if isinstance(rows, RowBlock):
            return rows.read()
        return self.values[rows]

    # -- modification log ----------------------------------------------------

    def record(self, rows, old_values: torch.Tensor, start_fn, end_fn):
        """Record one batched modification: ``rows`` held ``old_values`` until
        ``end_fn()``."""
        if isinstance(rows, RowBlock):
            rows = rows.indices
        self.mod_rows.append(rows)
        self.mod_values.append(old_values)
        self.mod_starts.append(start_fn)
        self.mod_ends.append(end_fn)
        # Invalidate the sorted-log cache (but keep the stale copy: row
        # history duplication during a render session reads from it).
        self._session_token = -1

    def copy_row_history(self, src_rows: torch.Tensor, dst_rows: torch.Tensor):
        """Replay all modifications of ``src_rows`` onto ``dst_rows``
        (position-wise src->dst).  Used by clones that share their source's
        history."""
        if len(self.mod_rows) == 0 or src_rows.numel() == 0:
            return
        src_rows = src_rows.contiguous()
        n = src_rows.numel()
        # Map buffer row index -> position in src_rows (-1 = not involved).
        max_row = int(src_rows.max())
        pos = torch.full((max_row + 1,), -1, dtype=torch.long)
        pos[src_rows] = torch.arange(n)
        for i in range(len(self.mod_rows)):
            rows = self.mod_rows[i]
            in_range = rows < max_row + 1
            p = torch.where(in_range, pos[rows.clamp(max=max_row)], torch.tensor(-1))
            hit = (p >= 0).nonzero().view(-1)
            if hit.numel() == 0:
                continue
            self.record(
                dst_rows[p[hit]],
                self.mod_values[i][hit],
                self.mod_starts[i],
                self.mod_ends[i],
            )

    def _duplicate_row_history(self, src_row: int, dst_rows: torch.Tensor):
        # Fast path: inside a render session the sorted log is built and its
        # timestamps are final, so the source row's instances are one
        # searchsorted away (the linear scan below is O(total log size) and
        # was quadratic over a whole scene's worth of batch expansions).
        # The sorted copy may be missing entries appended by *other*
        # duplications this session, which only matters for a row that was
        # itself added by an earlier expansion this session (a double batch
        # expansion of one mob mid-render).
        srt = self._sorted
        if srt is not None and srt.get("built_token") == self.state.session_token:
            rows, times, vals = srt["rows"], srt["times"], srt["values"]
            lo = int(torch.searchsorted(rows, src_row))
            hi = int(torch.searchsorted(rows, src_row, right=True))
            n = dst_rows.numel()
            for j in range(lo, hi):
                t = float(times[j])
                self.record(
                    dst_rows,
                    vals[j:j + 1].expand(n, -1).clone(),
                    (lambda t=t: t),
                    (lambda t=t: t),
                )
            return
        for i in range(len(self.mod_rows)):
            hit = (self.mod_rows[i] == src_row).nonzero().view(-1)
            if hit.numel() == 0:
                continue
            src_val = self.mod_values[i][hit[-1]:hit[-1] + 1]
            self.record(
                dst_rows,
                src_val.expand(dst_rows.numel(), -1).clone(),
                self.mod_starts[i],
                self.mod_ends[i],
            )

    def rows_have_history(self, rows) -> bool:
        srt = self._get_sorted_log()
        if srt is None:
            return False
        if isinstance(rows, RowBlock):
            rows = rows.indices
        srows = srt["rows"]
        idx = torch.searchsorted(srows, rows)
        idx = idx.clamp(max=srows.numel() - 1)
        return bool((srows[idx] == rows).any())

    # -- render-session sorted log ------------------------------------------

    def _get_sorted_log(self):
        """The modification log flattened per affected row and sorted by
        (row, end-time, insertion order), with timestamp callables evaluated.
        Built once per render session."""
        token = self.state.session_token
        if self._session_token == token:
            return self._sorted
        self._session_token = token
        if len(self.mod_rows) == 0:
            self._sorted = None
            return None
        starts = torch.tensor([float(s()) for s in self.mod_starts])
        ends = torch.tensor([float(e()) for e in self.mod_ends])
        # A modification that ends before it starts records nothing (matches
        # the old per-mob history filter).
        keep = (ends >= starts).nonzero().view(-1)
        if keep.numel() == 0:
            self._sorted = None
            return None
        counts = torch.tensor([self.mod_rows[i].numel() for i in keep])
        rows = torch.cat([self.mod_rows[i] for i in keep])
        vals = torch.cat([self.mod_values[i] for i in keep])
        times = torch.repeat_interleave(
            ends[keep].to(torch.get_default_dtype()), counts
        )
        # Stable two-pass argsort = lexsort by (row, time, insertion order).
        order = torch.argsort(times, stable=True)
        order = order[torch.argsort(rows[order], stable=True)]
        rows = rows[order]
        times = times[order]
        vals = vals[order]
        self._sorted = dict(
            rows=rows,
            times=times,
            values=vals,
            dense_rows=torch.unique_consecutive(rows),
            built_token=token,
        )
        return self._sorted

    def materialize_rows(self, t: torch.Tensor, query_rows: torch.Tensor | None = None):
        """State of every modified row (or of ``query_rows``) at each time in
        ``t`` ([T] float32 seconds), as (dense_rows [D], dense [T, D, W]).
        Returns (None, None) when nothing was ever modified (and no explicit
        rows were requested).

        For each (row, t) the value is the logged pre-write value of the
        first modification of that row with end-time > t, and the row's
        current (live buffer) value when no such modification exists --
        identical to the old per-mob ``(t >= end_times).sum`` gather, just
        batched over all rows via one searchsorted on the sorted log.
        """
        srt = self._get_sorted_log()
        if srt is None:
            if query_rows is None:
                return None, None
            current = self.values[query_rows]
            return query_rows, current.unsqueeze(0).expand(t.numel(), -1, -1)
        rows, times, vals = srt["rows"], srt["times"], srt["values"]
        dense_rows = srt["dense_rows"] if query_rows is None else query_rows
        K = rows.numel()
        D = dense_rows.numel()
        T = t.numel()
        # Exact integer ranking of the (float32) time domain so comparisons
        # match the old code's float32 `t >= end_time` bit-for-bit.
        domain = torch.cat([times, t]).unique(sorted=True)
        u = domain.numel() + 1
        inst_rank = torch.searchsorted(domain, times)
        q_rank = torch.searchsorted(domain, t)
        keys = rows * u + inst_rank
        q = dense_rows.view(1, -1) * u + q_rank.view(-1, 1)  # [T, D]
        idx = torch.searchsorted(keys, q.reshape(-1), right=True).view(T, D)
        in_row = (idx < K) & (
            rows[idx.clamp(max=K - 1)] == dense_rows.view(1, -1)
        )
        # Rows past their last modification take the live (current) value.
        current = self.values[dense_rows]
        vals_ext = torch.cat([vals, current])
        flat = torch.where(in_row, idx, torch.arange(K, K+D, device=idx.device).view(1, -1))
        dense = vals_ext[flat.reshape(-1)].view(T, D, self.width)
        return dense_rows, dense


class MaterializedWindow:
    """Per-attribute dense state over one range of frames [start, end).

    ``dense`` covers only rows with recorded modifications (plus rows
    densified on demand, e.g. spawn-opacity masking or render-time writes to
    otherwise-constant rows); all other rows are constant and read straight
    from the live buffer.
    """

    def __init__(self, state, start: int, end: int, fps: float):
        self.state = state
        self.start = int(start)
        self.end = int(end)
        self.fps = fps
        self.T = self.end - self.start
        self.time_inds = torch.arange(self.start, self.end)
        # Matches Animatable.set_state_pre_function_applications: t in seconds.
        self.t = (self.time_inds / fps).unsqueeze(-1)
        self._attrs: dict[int, dict] = {}

    def _entry(self, buffer: AttributeBuffer):
        entry = self._attrs.get(id(buffer))
        if entry is None:
            dense_rows, dense = buffer.materialize_rows(self.t.view(-1))
            pos = torch.full((buffer.size,), -1, dtype=torch.long)
            if dense_rows is not None:
                pos[dense_rows] = torch.arange(dense_rows.numel())
                used = dense_rows.numel()
            else:
                dense = torch.empty(
                    (self.T, 0, buffer.width), dtype=buffer.values.dtype
                )
                used = 0
            entry = dict(buffer=buffer, dense=dense, pos=pos, used=used)
            self._attrs[id(buffer)] = entry
        return entry

    def _positions(self, entry, rows: torch.Tensor) -> torch.Tensor:
        pos = entry["pos"]
        # Rows allocated after this window was created are constant here.
        safe = rows.clamp(max=pos.numel() - 1)
        p = pos[safe]
        p[rows >= pos.numel()] = -1
        return p

    def is_dense(self, buffer: AttributeBuffer, rows: torch.Tensor):
        entry = self._entry(buffer)
        return self._positions(entry, rows) >= 0

    def any_dense(self, buffer: AttributeBuffer, rows: torch.Tensor) -> bool:
        return bool((self.is_dense(buffer, rows)).any())

    def ensure_dense(self, buffer: AttributeBuffer, rows: torch.Tensor):
        """Densify ``rows`` (copy their constant value across all T frames)
        so they can be written per-frame."""
        entry = self._entry(buffer)
        p = self._positions(entry, rows)
        missing = rows[p < 0]
        if missing.numel() == 0:
            return entry
        missing = missing.unique()
        n = missing.numel()
        used = entry["used"]
        dense = entry["dense"]
        if used + n > dense.shape[1]:
            new_cap = max(dense.shape[1] * 2, used + n, 8)
            new_dense = torch.empty(
                (self.T, new_cap, buffer.width), dtype=dense.dtype
            )
            new_dense[:, :used] = dense[:, :used]
            entry["dense"] = new_dense
            dense = new_dense
        dense[:, used:used + n] = buffer.values[missing].unsqueeze(0)
        pos = entry["pos"]
        if buffer.size > pos.numel():
            new_pos = torch.full((buffer.size,), -1, dtype=torch.long)
            new_pos[: pos.numel()] = pos
            entry["pos"] = pos = new_pos
        pos[missing] = torch.arange(used, used + n)
        entry["used"] = used + n
        return entry

    def read(self, buffer: AttributeBuffer, rows: torch.Tensor,
             time_sel=None) -> torch.Tensor:
        """Read [T_sel, n, W] for ``rows``; ``time_sel`` is a tensor of
        window-relative frame indices (or None for all frames).  If none of
        the rows are dense the constant value is returned as [1, n, W]."""
        entry = self._entry(buffer)
        p = self._positions(entry, rows)
        if not bool((p >= 0).any()):
            return buffer.values[rows].unsqueeze(0)
        dense = entry["dense"]
        if bool((p >= 0).all()):
            if time_sel is None:
                return dense[:, p]
            return dense[time_sel.view(-1, 1), p.view(1, -1)]
        # Mixed constant/dense rows: overlay dense rows onto the constants.
        n_t = self.T if time_sel is None else time_sel.numel()
        out = (
            buffer.values[rows]
            .unsqueeze(0)
            .expand(n_t, -1, -1)
            .clone()
        )
        d_mask = p >= 0
        d_pos = p[d_mask]
        if time_sel is None:
            out[:, d_mask] = dense[:, d_pos]
        else:
            out[:, d_mask] = dense[time_sel.view(-1, 1), d_pos.view(1, -1)]
        return out

    def write(self, buffer: AttributeBuffer, rows: torch.Tensor, value,
              time_sel=None):
        """Write ``value`` ([T_sel, n, W] or broadcastable) to ``rows`` at
        ``time_sel`` (window-relative frame indices, None = all frames)."""
        entry = self.ensure_dense(buffer, rows)
        p = self._positions(entry, rows)
        dense = entry["dense"]
        if time_sel is None:
            dense[:, p] = value
        else:
            dense[time_sel.view(-1, 1), p.view(1, -1)] = value

    def clone_row_state(self, buffer: AttributeBuffer, src_row: int,
                        dst_rows: torch.Tensor):
        """Give ``dst_rows`` the same per-frame state ``src_row`` currently
        has in this window.  Used by render-time batch expansion: the old
        per-mob code broadcast the materialized (including
        already-applied-function) trajectory of the last column into the new
        columns."""
        entry = self._attrs.get(id(buffer))
        if entry is None:
            return
        p_src = self._positions(entry, torch.tensor([src_row]))[0]
        if p_src < 0:
            # Source row is constant in this window; the new rows already
            # hold its constant value in the live buffer.
            return
        entry = self.ensure_dense(buffer, dst_rows)
        p_dst = self._positions(entry, dst_rows)
        p_src = self._positions(entry, torch.tensor([src_row]))[0]
        entry["dense"][:, p_dst] = entry["dense"][:, p_src:p_src + 1]

    def rematerialize_rows(self, buffer: AttributeBuffer, rows: torch.Tensor):
        """Restore the pristine pre-function state of ``rows`` in this window.

        Needed when a mob is reset and later re-bound to the *same* window
        (the camera/screen/lights are re-materialized once per batch): its
        animated functions re-execute on binding, and they must not stack on
        top of the values their previous execution wrote into the shared
        dense tensors."""
        entry = self._attrs.get(id(buffer))
        if entry is None:
            return
        p = self._positions(entry, rows)
        sel = p >= 0
        if not bool(sel.any()):
            return
        rows_d = rows[sel]
        _, fresh = buffer.materialize_rows(self.t.view(-1), query_rows=rows_d)
        entry["dense"][:, p[sel]] = fresh

    def zero_time_range(self, buffer: AttributeBuffer, rows: torch.Tensor,
                        t0: int, t1: int):
        """Zero rows over window-relative frames [t0, t1) (opacity spawn
        masking)."""
        t0 = max(t0, 0)
        t1 = min(t1, self.T)
        if t1 <= t0:
            return
        entry = self.ensure_dense(buffer, rows)
        p = self._positions(entry, rows)
        entry["dense"][t0:t1, p] = 0


class GlobalAnimationState:
    """Singleton owning all attribute buffers, the arena backing them, the
    global modification log and render-time materialization windows."""

    _instance = None

    def __init__(self):
        raise RuntimeError(
            "Call GlobalAnimationState.instance() instead of GlobalAnimationState()."
        )

    @classmethod
    def instance(cls) -> "GlobalAnimationState":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._init()
        return cls._instance

    @classmethod
    def reset(cls):
        """Drop all rows, logs and windows (new scene).  The arena tensor is
        kept and re-used."""
        if cls._instance is not None:
            cls._instance._reset_keep_arena()

    def _init(self):
        # Pre-allocate the arena at import time (the plan's up-front global
        # buffer).  Committed lazily by the OS, so a large default is cheap
        # until actually written.
        mb = float(os.environ.get("ALGAN_ANIMATION_ARENA_MB", "1024"))
        self.arena_numel = int(mb * (2**20) // 4)
        self.arena = torch.empty((self.arena_numel,), dtype=torch.get_default_dtype())
        self.arena_used = 0
        self.initial_rows = int(os.environ.get("ALGAN_ANIMATION_INITIAL_ROWS", "4096"))
        self.buffers: dict[tuple[str, int], AttributeBuffer] = {}
        self.session_token = 0
        self.topology_version = 0
        self._windows: dict[tuple[int, int], MaterializedWindow] = {}
        self._session_fps = None

    def _reset_keep_arena(self):
        self.arena_used = 0
        self.buffers = {}
        self.session_token += 1
        self.topology_version += 1
        self._windows = {}
        self._session_fps = None

    # -- arena ---------------------------------------------------------------

    def _arena_alloc(self, rows: int, width: int) -> torch.Tensor:
        numel = rows * width
        if self.arena_used + numel <= self.arena_numel:
            out = self.arena[self.arena_used:self.arena_used + numel].view(
                rows, width
            )
            self.arena_used += numel
            return out
        # Arena exhausted: fall back to a plain allocation.
        return torch.empty((rows, width), dtype=self.arena.dtype)

    # -- buffers ---------------------------------------------------------------

    def get_buffer(self, name: str, width: int) -> AttributeBuffer:
        key = (name, width)
        buf = self.buffers.get(key)
        if buf is None:
            buf = AttributeBuffer(self, name, width)
            self.buffers[key] = buf
        return buf

    # -- render sessions / windows ---------------------------------------------

    def begin_session(self, fps: float):
        """Called at the start of every render (timestamps are final for its
        duration).  Invalidates sorted-log caches and windows."""
        self.session_token += 1
        self._windows = {}
        self._session_fps = fps

    def ensure_window(self, start: int, end: int, fps: float) -> MaterializedWindow:
        if self._session_fps is None:
            self._session_fps = fps
        key = (int(start), int(end))
        window = self._windows.get(key)
        if window is None:
            window = MaterializedWindow(self, start, end, fps)
            self._windows[key] = window
            # Windows from previous batches are unreachable through the cache
            # but stay alive while any mob's data is still bound to them.
            if len(self._windows) > 8:
                oldest = next(iter(self._windows))
                if oldest != key:
                    del self._windows[oldest]
        return window

    def bump_topology(self):
        self.topology_version += 1
