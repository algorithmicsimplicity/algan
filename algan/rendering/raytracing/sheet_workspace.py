"""Short-lived, explicitly owned scratch for sheet-compaction stages."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


class CompactionWorkspace:
    """Reuse forward-arena scratch between stages, with an allocator fallback.

    Only call ``tensor`` / ``copy`` inside ``stage`` and do not let their views
    escape that stage. Long-lived reduction results must use caller-owned
    outputs. Nested stages restore the previous scratch region even on failure;
    they never rewind the reverse end, where final resolver records live.

    ``peak_bytes`` counts this workspace's overlapping arena allocations,
    including alignment. It excludes library sort/scan workspace, external
    result arrays, and allocations made directly through ``memory``. Discovery
    adds it to its existing record/permutation budget, not to a claim of total
    device peak memory. No tensor references are retained by the accounting.
    """

    def __init__(self, memory=None, *, device=None):
        if memory is None and device is None:
            raise ValueError("a compaction workspace needs an arena or a device")
        self.memory = memory
        self.device = memory.data.device if memory is not None else torch.device(device)
        if device is not None and torch.device(device) != self.device:
            raise ValueError("compaction workspace and inputs must share a device")
        self.peak_bytes = 0
        self._live_bytes = 0
        self._depth = 0

    @contextmanager
    def stage(self):
        """Release this stage's scratch while keeping its caller's inputs live."""
        live = self._live_bytes
        self._depth += 1
        context = self.memory.temp() if self.memory is not None else nullcontext()
        try:
            with context:
                yield self
        finally:
            self._live_bytes = live
            self._depth -= 1

    def tensor(self, shape, dtype, fill=None):
        if self._depth == 0:
            raise RuntimeError("compaction scratch must be allocated inside a stage")
        if self.memory is None:
            result = torch.empty(shape, dtype=dtype, device=self.device)
        else:
            start = self.memory.current_pointer
            result = self.memory.get_tensor(shape, dtype)
            self._live_bytes += self.memory.current_pointer - start
            self.peak_bytes = max(self.peak_bytes, self._live_bytes)
        if fill is not None:
            result.fill_(fill)
        return result

    def copy(self, value, dtype=None):
        """Always copy, including same-dtype casts used as kernel scratch."""
        result = self.tensor(value.shape, value.dtype if dtype is None else dtype)
        result.copy_(value)
        return result

    def gather(self, value, indices):
        """Gather exact rows into this stage, without an allocator-owned result."""
        from algan.rendering.raytracing.array_ops import gather_rows

        result = self.tensor((indices.numel(), *value.shape[1:]), value.dtype)
        return gather_rows(value, indices, out=result)
