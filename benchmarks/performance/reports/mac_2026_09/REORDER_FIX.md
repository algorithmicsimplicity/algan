# Mac ray queue corruption: cause and fix

Investigated 2026-09-08 against `claude/eager-cori-h61xc8` at
`a2073e5475e8e469dc53e2f199a3f7eff4877017`, PyTorch 2.7.1 and the patched
Quadrants wheel from build 33850787142.

## Confirmed defect

`_ArenaRayCompactor.reorder` used
`torch.index_select(active, 0, perm, out=self.spare[:n])`.
On this MPS stack the output view's storage offset is ignored: the result
lands at the allocation base, overwriting the render arena's prefix, while
the intended destination is unchanged. This is separate from the previously
documented packed-int64 gather problem.

A seven-element reproduction on disposable storage confirms the failure:

```python
import torch

arena = torch.full((64,), -123, dtype=torch.int32, device="mps")
arena[8:15].copy_(torch.arange(23, 30, dtype=torch.int32, device="mps"))
indices = torch.tensor([4, 1, 6, 0, 5, 2, 3], device="mps")
torch.index_select(arena[8:15], 0, indices, out=arena[24:31])
print(arena.cpu()[:7])       # [27, 24, 29, 23, 28, 25, 26] -- corrupted prefix
print(arena.cpu()[24:31])    # [-123, -123, -123, -123, -123, -123, -123]
```

The upstream implementation agrees with the reproduction:
[`index_select_out_mps`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mps/operations/Indexing.mm)
creates its output Placeholder with both `gatherTensorData=false` and
`useStridedAPI=false`.
[`Placeholder::Placeholder`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mps/OperationUtils.mm)
then wraps the base Metal buffer without the view offset; its offset-aware
NDArray path is skipped.

In the failing UHD render, all 336,331 active-ray indices became zero after
sorting. Compaction itself matched a CPU reference. The real active rays never
retired, and chunks repeated traversal 275 times, reaching the iteration limit.
Thus the earlier launch counts measured a correctness failure as well as
submission overhead. Arena-prefix corruption also provides a concrete trigger
for layout-dependent shader faults.

## Fix

For MPS only, gather the permutation into the spare arena view with
`reorder_ray_slots`, a small Quadrants kernel. Imported ndarray views preserve
their offsets. Keep buffer swapping, capacity, sort behavior, CPU/CUDA code,
memory defaults and synchronization boundaries unchanged.

Regression tests check the complete arena byte-for-byte around empty, small
and larger gathers, including nonzero offsets, padding, source and unused
destination bytes. A second test applies successive permutations across the
compactor's buffer swaps and exercises the MPS branch when run on a Mac GPU.

## Runner validation

| Experiment | Result |
| --- | --- |
| [67: clean uncapped UHD](https://github.com/algorithmicsimplicity/algan/actions/runs/34176537370) | All 18 chunks; 277.3 s; 1720 MiB arena |
| [68: two uncapped UHD renders in one process](https://github.com/algorithmicsimplicity/algan/actions/runs/34176792316) | 288.4 s cold, 175.6 s warm; 18 chunks each; 1720 MiB arena throughout |
| [69: lossless frame parity](https://github.com/algorithmicsimplicity/algan/actions/runs/34176867370) | Fixed sorted MPS and unsorted MPS are byte-identical, 704×396 RGB |

All three clean UHD renders use 4–5 traversal calls and 59–62 converted
kernel launches per chunk. The corrupted uncapped baseline used 275 traversal
calls and 1,141 launches per chunk. Runs 67 and 68 exit normally and their
captured system logs contain no GPU timeout/restart events.

This removes the demonstrated corruption and prevents the reproduced stalls
in these tests. It does not prove every historical timeout had the same cause,
or guarantee arbitrary arena sizes. The previous failing system logs showed
approximately ten-second GPU command timeouts followed by unsuccessful
AppleParavirtGPU recovery, with Python blocked in a native GPU wait. Changing
only dispatch size did not solve that failure.

The parity run's CPU image has mean absolute channel difference 0.00534/255
from MPS; 99.9% of channels differ by at most 1, but the maximum is 42 and
0.055% of channels exceed 2. Therefore cross-backend strict pixel parity is
not established; the direct sorted/unsorted MPS control is exact. The full
images are retained in the run artifact.

## Controlled CPU/MPS timing after the repair

[Run 70](https://github.com/algorithmicsimplicity/algan/actions/runs/34177523689)
runs both backends sequentially on one VM, in fresh processes, with exactly
1720 MiB arenas, the same UHD scene, two renders per process, software
encoding, and Torch compilation disabled for both. Each render completes
18 chunks in two primitive batches. It omits the external stack sampling and
extra synchronization-timing wrappers used in runs 67–68.

| Backend | First render | Second render |
| --- | ---: | ---: |
| CPU | 94.9 s | 73.2 s |
| Repaired MPS | 150.9 s | 84.5 s |

MPS is 15.4% slower warm in this controlled run. By the start of chunk 3,
8.6 seconds of the total 11.3-second warm gap has already accumulated
(CPU +11.0 s, MPS +19.6 s). The remaining 16 chunks and finalization add
only 2.7 seconds to that gap. This localizes much of the residual cost to
early render work; it does not isolate a virtualization penalty.
Cold shader compilation costs also differ substantially.

The former 1010.3-second MPS result used a smaller arena and a corrupted
queue, so it is not a matched baseline for claiming a precise speedup.
Five clean repaired uncapped UHD renders now complete across runs 67, 68
and 70. The conservative default arena cap remains in place.

## Hardware and performance interpretation

The runner has hardware GPU acceleration through Apple's paravirtualized
Metal interface. It is a VM with a virtual GPU device, not CPU-only software
rendering. [Apple describes host GPU acceleration for macOS guests](https://developer.apple.com/videos/play/wwdc2022/10002/?time=644);
the runner enumerates `Apple Paravirtual device` and loads the corresponding
driver. Its earlier FP32 benchmark reaches 1.293 TFLOP/s on MPS versus
0.452 TFLOP/s on its CPU.

Neither that throughput nor the repaired render measures the penalty relative
to a physical Mac. The old report's two-microsecond comparison was a CPU
operation, and its import counter counted arguments rather than launches.
Cold shader compilation, CPU preparation, fine-grained MPS operations and
cross-queue synchronization still matter after the correctness fix. Do not
remove required synchronization based on aggregate wait times.
