"""Stable ordering the device does, in place of torch's own sort.

The renderer's raster and compaction stages are a chain of stable sorts over
the fragment stream, and on most backends torch's ``argsort`` already *is* a
radix sort -- CUDA dispatches to CUB, the CPU to a parallel merge sort -- so
there is nothing to win by writing another one. On **MPS** it is not: an
``argsort`` of 2.9M int64 keys goes through MPSGraph and takes ~0.11 s where a
radix sort of the same keys is a few milliseconds, and a warm UHD chunk spends
more time inside those sorts and the gathers that compose them than it spends
in every ray-tracing kernel put together
(``benchmarks/performance/reports/mac_2026_09/SHARED_QUEUE.md`` §4).

So this module routes those sorts to
:func:`~algan.rendering.raytracing.radix_sort_taichi.argsort_pairs`, the
device-wide LSB radix sort Quadrants publishes, whenever the running compiler
has it and a kernel can be launched against the tensors without staging them
(:func:`~algan.rendering.taichi_runtime.taichi_launch_is_local`). What it
replaces is not only the sort: an LSD multi-key order in torch is one
``argsort`` and one ``index_select`` per key, and the kernel form does the
gather inside the seed loop, so the ``index_select`` chain disappears with it.

**It is OFF by default, and the measurement is why.** Isolated, it is worth
1.3-2.1x on Metal (and a 3.5-4.5x *loss* on CUDA, where torch dispatches to
CUB). But the sort it would have replaced in the compaction's hot path is gone:
``sheet_pixel_sort`` orders each pixel's run in place instead, at 4.1 ms
against torch's 215 ms, and what is left for a global sort does not recoup this
one's cost. Measured ABBA on the Mac runner
(``reports/mac_2026_09/DEVICE_SORT.md``): warm ``nn_scene_UHD`` at **43.7 s
with the run sort alone and 47.5 s with this on top**, +8.7%, with the
kernel's own time only +0.19 s over 17 chunks -- so what it costs is the torch
work around it, not the sorting.

``ALGAN_DEVICE_RADIX_SORT=1`` turns it on, which is how both columns above were
taken and how the next candidate site gets priced. It is read per call, never
bound at import, so a render can be A/B'd without a fresh process.

**Small inputs stay on torch.** The emitted sort is a fixed chain of ~30 (32-bit
keys) or ~60 (64-bit keys) offloaded launches whatever ``n`` is, so under a few
tens of thousands of elements the chain costs more than the sort saves.
``ALGAN_DEVICE_RADIX_SORT_MIN`` is that floor.

**The order is int32.** ``torch.argsort`` returns int64; these return int32,
which is what the kernel carries and what the next stage of an LSD chain wants
back. A fragment stream cannot approach 2**31 -- the arrays would not fit --
and ``index_select`` takes an ``IntTensor`` as readily as a ``LongTensor``, so
the narrower permutation is a straight saving in traffic. Call sites that hand
the order to something needing int64 convert at that boundary.
"""

from __future__ import annotations

import torch

from algan.environment import env_flag, env_int

#: Bits the sort must cover per key dtype -- and, with them, the pass count,
#: which is one per byte. Only these three dtypes reach it: an integer key the
#: renderer packs (int32 where the range is known to fit, int64 for a
#: pixel/depth composite) and a raw float32 depth.
#:
#: Every one of them is a full width rather than the key's measured range,
#: because the sort maps a signed or float key to a monotone unsigned order by
#: a twiddle that **sets the top bit** -- so a small nonnegative key still has
#: work in its highest byte. Narrowing the *dtype* is the way to spend fewer
#: passes, and ``sheets._narrow_sort_key`` already does exactly that.
_END_BITS = {torch.int32: 32, torch.int64: 64, torch.float32: 32}


def radix_sort_enabled() -> bool:
    """Whether device sorting is wanted here, before asking whether it is possible.

    Off unless asked for; the module docstring has the measurement that decided
    that. Read per call rather than bound at import, so an A/B can flip it
    between renders in one process.
    """
    return env_flag("ALGAN_DEVICE_RADIX_SORT", False)


def _minimum_elements() -> int:
    return max(1, env_int("ALGAN_DEVICE_RADIX_SORT_MIN", 1 << 16))


def radix_sort_available(keys) -> bool:
    """Whether :func:`stable_argsort` would take the kernel for ``keys``.

    Every condition that decides it, in one place, so a call site asks once and
    a diagnostic can ask the same question:

    * the mode is on (:func:`radix_sort_enabled`);
    * the compiler is Quadrants and publishes ``algorithms.sort`` -- the Taichi
      arm has only the deprecated ``parallel_sort``, an odd-even merge sort
      that is slower than what it would replace;
    * a program is up, it is **not** on the CPU arch, and a launch against this
      tensor stages nothing;
    * the key dtype is one the sort orders, and there are enough of them to pay
      for its launch chain.

    The CPU arch is excluded by name rather than left to the staging test,
    which passes there: the sort is built out of ``block.sync`` and
    ``block.radix_rank_match_atomic_or``, which the LLVM backend does not
    implement, so the kernel does not compile at all. Nothing is lost --
    torch's CPU sort is a parallel merge sort with nothing to gain from this.
    """
    if keys.dtype not in _END_BITS:
        return False
    n = int(keys.numel())
    if n < _minimum_elements() or n >= 2**31:
        return False
    if not keys.is_contiguous() or not radix_sort_enabled():
        return False
    from algan.taichi_compat import BACKEND, ti

    if BACKEND != "quadrants":
        return False
    algorithms = getattr(ti, "algorithms", None)
    if algorithms is None or not hasattr(algorithms, "sort"):
        return False
    from algan.rendering.taichi_runtime import (
        _live_arch,
        taichi_arch_is_cpu,
        taichi_launch_is_local,
    )

    if _live_arch() is None or taichi_arch_is_cpu():
        return False
    return taichi_launch_is_local(keys.device)


def _scan_depth(n: int) -> int:
    """The sort's ``log256_max_n``, from the smallest set that covers ``n``.

    The depth fixes the emitted launch topology (``2D - 1`` scan levels per
    digit pass), so it is a specialization key: choosing the true minimum per
    call would compile a fresh sort for every input size. Three values cover
    every fragment stream that fits in memory, and a stream is nearly always in
    the third.
    """
    if n <= 1 << 16:
        return 2
    if n <= 1 << 24:
        return 3
    return 4


def _key_dtype(keys):
    from algan.taichi_compat import ti

    return {
        torch.int32: ti.i32,
        torch.int64: ti.i64,
        torch.float32: ti.f32,
    }[keys.dtype]


def _end_bit(keys) -> int:
    return _END_BITS[keys.dtype]


def stable_argsort(
    keys, *, perm=None, keys_are_permuted=False, out=None, workspace=None
):
    """``torch.argsort(keys, stable=True)``, on the device, as an int32 order.

    ``perm`` composes this sort with one that already ran, which is how an LSD
    multi-key order is built and the whole reason the gather lives in the
    kernel:

    * ``perm=None`` -- a fresh order over ``keys``.
    * ``perm`` given, ``keys_are_permuted=True`` -- ``keys[i]`` is already the
      key of element ``perm[i]``, so only the permutation composes. This is the
      shape of a second sort whose key was *derived* in the first sort's order.
    * ``perm`` given, ``keys_are_permuted=False`` -- ``keys`` is in the
      original order and the kernel reads ``keys[perm[i]]``: one more
      significant pass of an LSD chain, with no torch gather between the
      passes.

    ``out`` may supply the contiguous int32 destination. It must be disjoint
    from the input keys and permutation. ``workspace`` supplies short-lived
    sort buffers; the default result always lives outside its scratch stage.
    Index values remain the caller's responsibility, without device readback.

    Returns None where the kernel does not apply, without writing the output,
    so a caller can fall through to its torch arm.
    """
    if not radix_sort_available(keys):
        return None
    n = int(keys.numel())
    if keys.shape != (n,):
        raise ValueError("sort keys must be a one-dimensional vector")
    if perm is not None and int(perm.numel()) != n:
        # The kernel subscripts one array by the other with no bound of its
        # own, so a mismatch is an out-of-range read rather than an error.
        raise ValueError(
            f"perm has {int(perm.numel())} entries for {n} keys; a composed "
            "sort permutes the same stream it orders"
        )
    if perm is not None and (
        perm.shape != (n,)
        or perm.dtype not in (torch.int32, torch.int64)
        or perm.device != keys.device
    ):
        raise ValueError("perm must be an integer vector on the keys device")
    if perm is None:
        mode = 0
    elif keys_are_permuted:
        mode = 1
    else:
        mode = 2
    device = keys.device
    from algan.rendering.raytracing.array_ops import require_tensor_outputs
    from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace

    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("sort workspace and keys must share a device")
    values = torch.empty(n, dtype=torch.int32, device=device) if out is None else out
    require_tensor_outputs(
        (values,),
        (((n,), torch.int32),),
        device=device,
        inputs=(keys,) if perm is None else (keys, perm),
    )
    # With nothing to compose, the argument the kernel never reads aliases the
    # output rather than costing an allocation and a specialization of its own.
    from algan.rendering.raytracing.radix_sort_taichi import argsort_pairs
    from algan.taichi_compat import ti

    depth = _scan_depth(n)
    scratch_slots = int(ti.algorithms.sort_scratch_slots(n, depth))
    with workspace.stage():
        if perm is None:
            perm = values
        elif perm.dtype != torch.int32 or not perm.is_contiguous():
            perm = workspace.copy(perm, torch.int32)
        work_keys = workspace.tensor((n,), keys.dtype)
        tmp_keys = workspace.tensor((n,), keys.dtype)
        tmp_values = workspace.tensor((n,), torch.int32)
        scratch = workspace.tensor((scratch_slots,), torch.int32)
        count_buf = workspace.tensor((), torch.int32)
        argsort_pairs(
            keys,
            perm,
            work_keys,
            tmp_keys,
            values,
            tmp_values,
            scratch,
            count_buf,
            n,
            _key_dtype(keys),
            _end_bit(keys),
            depth,
            mode,
        )
    return values


def stable_lexsort(*keys, out=None, workspace=None):
    """The stable order of ``keys`` in priority order, or None for the torch arm.

    The same composition ``sheets._lexsort`` does -- least significant key
    first, each pass reordering the previous permutation -- with the gather
    that composes them folded into the sort kernel, so a three-key order is
    three launches and no ``index_select`` at all.

    All-or-nothing: if any key would fall through to torch, so does the whole
    order, because a half-device chain would pay the gathers it exists to
    remove.
    """
    if not keys or not all(radix_sort_available(key) for key in keys):
        return None
    from algan.rendering.raytracing.array_ops import require_tensor_outputs
    from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace

    n, device = keys[0].numel(), keys[0].device
    if any(key.shape != (n,) or key.device != device for key in keys):
        raise ValueError("sort keys must be equal-length vectors on one device")
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("sort workspace and keys must share a device")
    result = torch.empty(n, dtype=torch.int32, device=device) if out is None else out
    require_tensor_outputs(
        (result,), (((n,), torch.int32),), device=device, inputs=keys
    )
    with workspace.stage():
        other = workspace.tensor((n,), torch.int32) if len(keys) > 1 else result
        order = None
        destination = result
        for key in reversed(keys):
            stable_argsort(key, perm=order, out=destination, workspace=workspace)
            order = destination
            destination = other if destination is result else result
        if order is not result:
            result.copy_(order)
    return result
