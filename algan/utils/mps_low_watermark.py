"""Start torch's MPS allocator with a low watermark the render's hold can rely on.

torch's MPS allocator puts a request of 10 MiB to 512 MiB that finds no room
in an existing heap into a new 1 GiB heap, but only while the device's total
is under the allocator's low watermark; past it torch counts the device as
under memory pressure and gives each such request a heap of its own size,
rounded up to 2 MiB (``createHeapBlock`` in torch 2.13's
aten/src/ATen/mps/MPSAllocator.h). The render's hold
(``memory_utils.hold_mps_allocator_to_the_working_set``) is a limit on the
device's total that torch checks against the request, not the heap, so under
torch's default low watermark, 1.4 working sets, a request that passes the
check just under the limit can still open a 1 GiB heap and take the device up
to 1 GiB past it, which is far enough for Metal to fail a command buffer
without a word (``memory_utils._MPS_RENDER_HIGH_WATERMARK`` has the
measurements). With the low watermark at :data:`LOW_WATERMARK` the last 1 GiB
heap opens below it and ends at most 1 GiB above it, every request after that
gets a heap of its own size, and the hold keeps the device short of the
working set.

torch reads ``PYTORCH_MPS_LOW_WATERMARK_RATIO`` once, when its allocator
starts, and has no call that changes it afterwards. The allocator starts at
the first MPS allocation or the first ``torch.mps`` memory call
(``recommended_max_memory`` and ``set_per_process_memory_fraction``
included); ``import torch`` and ``torch.backends.mps.is_available()`` do not
start it. So ``algan/__init__.py`` calls :func:`set_before_torch_starts`
before it imports torch, and a process that started MPS before importing
Algan keeps torch's default.

Nothing here imports torch: it runs before torch is imported.
"""

from __future__ import annotations

import os
import sys

#: The low watermark Algan starts torch's MPS allocator with, as a share of
#: Metal's working set (``recommendedMaxWorkingSetSize``).
#:
#: Low enough that the last 1 GiB heap, opened just under it, ends at least
#: 64 MiB under the working set: 0.75 working sets + 1 GiB does that for any
#: working set of 4.25 GiB or more. The 7 GB Mac runner has 4.67 GiB, and an
#: 8 GB Apple-silicon Mac about 5.3.
#:
#: High enough to cost nothing in an ordinary render. Past it torch runs its
#: pressure path: heaps of the request's size instead of shared 1 GiB ones,
#: cached buffers garbage-collected when a request misses the cache, and each
#: MPSGraph op's command buffer committed as soon as it is encoded. On the
#: runner the HD and UHD reference renders peak at 0.51 to 0.53 and 0.69 to
#: 0.71 working sets and never reached 0.75 in runs 42385375749 and
#: 42388655058, so they ran as under torch's default, and their frames
#: matched. Even with the low watermark at 0.5, which the UHD render was past
#: for nearly all of its time, it was no slower and its frames matched.
LOW_WATERMARK = 0.75

_LOW_VARIABLE = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
_HIGH_VARIABLE = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"

#: Whether torch's MPS allocator in this process runs, or will run, with a low
#: watermark of :data:`LOW_WATERMARK` or less, as :func:`set_before_torch_starts`
#: found it.
_IN_EFFECT = False


def set_before_torch_starts(environ=None, platform=None) -> bool:
    """Put :data:`LOW_WATERMARK` in the environment for torch's MPS allocator.

    Only on macOS, only while the allocator has not started (torch is not
    imported yet, or it is and its allocator says it has not started), and
    never over the user's own choice:

    * ``PYTORCH_MPS_LOW_WATERMARK_RATIO`` already set, by the user or by a
      parent Algan process, stands, and counts as in effect when it is above
      0 and at most :data:`LOW_WATERMARK` (0 turns torch's pressure path off).
    * ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` set means the render does not hold
      the allocator at all, so there is nothing for a low watermark to serve;
      and torch refuses to start with a low watermark above the high one.

    ``environ`` and ``platform`` default to the process's own; tests pass
    their own. Returns whether the low watermark is in effect.
    """
    global _IN_EFFECT
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    _IN_EFFECT = False
    if platform != "darwin":
        return False
    own = environ.get(_LOW_VARIABLE)
    if own is not None:
        _IN_EFFECT = _at_most_ours(own)
        return _IN_EFFECT
    if environ.get(_HIGH_VARIABLE) is not None or _allocator_started():
        return False
    environ[_LOW_VARIABLE] = str(LOW_WATERMARK)
    _IN_EFFECT = True
    return True


def in_effect() -> bool:
    """Whether :func:`set_before_torch_starts` found the low watermark in effect."""
    return _IN_EFFECT


def _at_most_ours(raw: str) -> bool:
    try:
        ratio = float(raw)
    except ValueError:
        return False
    return 0.0 < ratio <= LOW_WATERMARK


def _allocator_started() -> bool:
    """Whether torch's MPS allocator has already read its environment.

    False while torch is not imported, the usual case at ``import algan``. A
    torch without ``torch._C._accelerator_isAllocatorInitialized`` (2.7 has
    none; 2.13, the Mac arm's, has it) cannot say, and is taken not to have
    started: setting the variable then changes nothing in a process whose
    allocator did start.
    """
    torch = sys.modules.get("torch")
    probe = getattr(
        getattr(torch, "_C", None), "_accelerator_isAllocatorInitialized", None
    )
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001  (no accelerator: nothing to start)
        return False
