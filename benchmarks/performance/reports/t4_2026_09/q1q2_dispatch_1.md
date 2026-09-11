# Audit Q1/Q2 device dispatch: does it speed the renderer up?

**Answer: no. It is slower, on both presets, in every pairing.**
UHD **+5.1%**, PREVIEW **+2.2%** on the cleaner pairing. The rendered frames are
byte-identical, so the correctness claim holds; the performance claim — which
the branch never made — does not.

| | `ALGAN_DEVICE_DISPATCH=0` | `=1` | |
| --- | --- | --- | --- |
| UHD warm, three arms each | 9.96 / 9.81 / **9.82** | 10.32 / 10.84 / **10.32** | **+5.1%** (medians) |
| PREVIEW warm, two arms each | 4.53 / **4.48** | 4.97 / **4.58** | **+2.2%** (best of each) |

Kaggle T4 (Tesla T4, driver 580.159.04, `ALGAN_DEVICE cuda`), one session,
arms interleaved ABBA, `codex/q1q2-device-dispatch-20260911` @ `445bc7e`,
`ALGAN_VIDEO_ENCODER=software`. Warm RUN 2 only. Full transcript:
`q1q2-dispatch-1.txt`.

Every `on` run is slower than every `off` run at UHD — 10.32 against 9.96 at
the closest — so the result does not depend on which pairing you read. That
matters here because a single UHD gap of 5% is only just outside this box's
~3% within-session band; the **separation** is the signal, not the margin.

> The absolute level is not comparable to this directory's README, which
> records `nn_ablation base UHD` at 17.63–17.75 s. The renderer has moved a
> long way since that round and the same fixture now runs in ~9.8 s. Only the
> within-session off/on comparison above is a reading.

## The fixture, because the obvious one measures nothing

`nn_scene_UHD.py` sets `shadows=False`. Q1/Q2's only production integration is
the deterministic sheet renderer's **primary shadow events**, so on that
fixture the entire path compiles out and both arms are the same program. The
arms above run `nn_ablation.py base`, which is the same scene with shadows on.
`nn_scene_PREVIEW.py` also has shadows on; `nn_ablation base PREVIEW` is used
here so both presets share one script.

## Where the time goes

Median arms, RUN 2 inclusive seconds (call counts in parentheses):

| stage | off | on | Δ |
| --- | ---: | ---: | ---: |
| end-to-end | 9.82 | 10.32 | **+0.50** |
| `ray traced render total` | 6.519 (11) | 6.976 (16) | +0.457 |
| `raster: sparse resolve` | 0.902 (11) | 1.186 (16) | +0.284 |
| `kernel: raster_shadow_trace_arena` | 0.808 (61) | 0.965 (86) | +0.157 |
| `raster: sparse discovery` | 2.422 (11) | 2.518 (16) | +0.096 |
| `raster: - compact_sheets` | 1.376 (11) | 1.383 (16) | ~0 |
| the five new dispatch kernels | — | 0.041 (96) | +0.041 |

Ninety-one percent of the regression is inside the ray-traced render. Note
the **call counts**: identical scene, identical two batches of 23 and 7 frames,
but the `on` arm ran **16 render chunks where the `off` arm ran 11**.

## The mechanism is arena footprint, not the new kernels

The new kernels are cheap and are not the problem. Across all three `on` arms
they are flat to the millisecond:

```
pack_primary_shadow_events   16   0.025
count_selected               16   0.008
scan_dispatch_blocks         32   0.004
reset_dispatch_header        16   0.002
add_dispatch_carries         16   0.002
                                  0.041 s total
```

That is 0.4% of the render. Replacing `nonzero` + seven `index_select` calls
with a hand-written block scan costs nothing measurable, even though
`count_selected` and `scan_dispatch_blocks` parallelize over 256-element
blocks — about 780 threads for a 200k-sheet window, on a 2560-core card.

What does cost is that **Q1 allocates its queues from the arena at full source
capacity**. `prepare_primary_shadow_dispatch` sizes every output buffer by the
slice's *source sheet* count, not its accepted-event count:

* `pos`, `snrm`, `fnrm` — 12 B each
* `frame`, `mask`, `source` — 4 B each
* `dp` (24 B) and `toff` (12 B) when footprint/terminator are live
* plus `rank`, the scan levels and the header

That is 52–88 bytes per source sheet. The reference path instead allocates
Torch temporaries sized by the *accepted* count, which this scene measures at
41.5% — so the dispatch arm reserves roughly six times the bytes for the same
events. On a UHD render whose arena is already the binding constraint, the
render-chunk preflight prices that in and hands back shorter chunks: 16 instead
of 11. More chunks means more per-chunk fixed cost, and that is the +0.50 s.

PREVIEW is the control that isolates it. There the arena is not pressured, so
**the chunk count is 4 in all four arms**, and:

| stage | off | on |
| --- | ---: | ---: |
| `kernel: raster_shadow_trace_arena` | 0.074 / 0.074 (20) | 0.073 / 0.072 (20) |
| `raster: sparse resolve` | 0.116 / 0.105 | 0.144 / 0.129 |

The trace kernel is **unchanged** — same launches, same time — and the only
residue is +0.026 s of host-side dispatch work in the resolve. So when Q1 does
not perturb chunking it is roughly free but still not a win; the UHD
regression is the chunking, not the queues themselves.

This is the tradeoff the audit's own Rank 1 warns about in the opposite
direction: "Do not assume the largest possible chunk is fastest." Here the
chunk got *smaller*, and it cost 5%. It also sits awkwardly beside Rank 1's
actual goal, which is to *reduce* persistent per-fragment arena storage — Q1 as
implemented adds 52–88 B per source sheet to it.

## Queue geometry (`nn_dispatch_queues.py UHD`)

```
window 0: sheets=1105597 accepted=455257  rate=0.4118 overlaunch=2.43x
window 1: sheets=2259804 accepted=933225  rate=0.4130 overlaunch=2.42x
window 2: sheets=3329118 accepted=1388012 rate=0.4169 overlaunch=2.40x
acceptance rate=0.4147  trace over-launch=2.41x
```

The two arms do not launch the same grid. The reference compacts on the host
and launches `num_events * num_lights`; the device-counted arm never learns the
count host-side and launches `capacity * num_lights`. At 41.5% acceptance that
is a 2.41x grid of threads, most of which read the header, fail the
`idx < count` guard and return.

**That over-launch appears to be nearly free**, which was not the expectation.
PREVIEW carries the same 2.4x factor and shows no change in trace time at all.
It is not separately measured at UHD — there the trace total rose 19% while its
launch count rose 41%, so per launch it actually fell (13.2 ms to 11.2 ms), and
the chunking change confounds any attempt to attribute the rest. If Q1 is
pursued, sizing the queue to a device-published count rather than to source
capacity would address the footprint and the grid together.

## A second behavioural difference, unmeasured

The `off` arm sorts accepted events by source triangle
(`_order_primary_shadow_events`, CUDA, ≥8192 events — both true at UHD) for
traversal coherence. The dispatch arm packs in stable sheet order and does not
sort. `agent_guidance/device_dispatch.md` describes this as changing
"scheduling only", which is true of the output and not of the time: the
repository keeps that sort because it was measured to pay. This round does not
separate its contribution from the chunking, because the two move together.

## Output parity holds

sha256 of the rendered mp4s, both arms, both presets:

```
UHD      1c89e4c1712a4cf6   dd0 == dd1
PREVIEW  994adb9379c268ac   dd0 == dd1
```

Byte-identical. Whatever else is true of Q1/Q2, it does preserve the image on
real CUDA hardware, which the branch's own validation could only assert on CPU.

## What was not measured

* **Whether the branch regresses with the toggle off.** With
  `device_dispatch=False`, `ManualMemory.get_tensor` still reads
  `SETTINGS.raytracing.device_dispatch` on every arena allocation and
  `current_pointer` is now a property with a tracker check on every write, so
  the `off` arm is not master. A master control is in `q1q2_dispatch_2.md`.
* **Metal and CPU.** CUDA only.
* **Scenes with a lower or higher acceptance rate**, which would move the
  footprint penalty directly.
* Per-kernel GPU time: `nn_ablation.py` passes `kernel_profiler=False`, so the
  stage numbers are device-synced wall time. The whole-render figure is the
  verdict; the per-stage split is directional.

## Reproducing

```
uv run python scripts/kaggle/make_notebook.py --tag q1q2-dispatch \
    --branch codex/q1q2-device-dispatch-20260911 \
    --step "uhd_off_1:ALGAN_DEVICE_DISPATCH=0 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "uhd_on_1:ALGAN_DEVICE_DISPATCH=1 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "uhd_on_2:ALGAN_DEVICE_DISPATCH=1 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "uhd_off_2:ALGAN_DEVICE_DISPATCH=0 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "uhd_off_3:ALGAN_DEVICE_DISPATCH=0 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "uhd_on_3:ALGAN_DEVICE_DISPATCH=1 python benchmarks/performance/nn_ablation.py base UHD" \
    --step "prev_off_1:ALGAN_DEVICE_DISPATCH=0 python benchmarks/performance/nn_ablation.py base PREVIEW" \
    --step "prev_on_1:ALGAN_DEVICE_DISPATCH=1 python benchmarks/performance/nn_ablation.py base PREVIEW" \
    --step "prev_on_2:ALGAN_DEVICE_DISPATCH=1 python benchmarks/performance/nn_ablation.py base PREVIEW" \
    --step "prev_off_2:ALGAN_DEVICE_DISPATCH=0 python benchmarks/performance/nn_ablation.py base PREVIEW" \
    --step "queues_uhd:python benchmarks/performance/nn_dispatch_queues.py UHD" \
    --env ALGAN_VIDEO_ENCODER=software --step-timeout 600 --out /tmp/nb.py
```

then `save_notebook` per `agent_guidance/gpu_harnesses.md`.
