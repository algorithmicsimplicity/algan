# Overlapping the arena preflight (`prefetch_gpu_prep`): a real but modest win

**UHD −1.8%, PREVIEW −5.4%, byte-identical output, faster in all six pairings.**
Worth turning on. It does not recover the preflight's full cost, and the stage
table says why.

| | `base` (default) | `ovl` (`prefetch_gpu_prep=True`) | |
| --- | --- | --- | --- |
| UHD warm, three arms each | 9.47 / 9.36 / **9.36** | 9.35 / 9.10 / **9.19** | **−1.8%** |
| PREVIEW warm, three arms each | 4.42 / 4.49 / **4.42** | 4.38 / 4.18 / **4.18** | **−5.4%** |

(medians in bold). Kaggle T4, one session, arms interleaved,
`codex/q1q2-device-dispatch-20260911` @ `e95f589`, warm RUN 2,
`ALGAN_VIDEO_ENCODER=software`. Transcript: `prefetch-overlap-1.txt`.

Unlike the Q1/Q2 result, the arms do **not** fully separate — `uhd_ovl_1` at
9.35 is slower than `uhd_base_3` at 9.30. What carries the reading is direction:
every adjacent pairing favours `ovl`, three at UHD (−1.3%, −2.8%, −1.2%) and
three at PREVIEW (−0.9%, −6.9%, −4.4%). That is the same standard the
shadow-visibility payload result in this directory was accepted on: each
individual gap sits inside the ~3% band, the direction does not.

> Do not compare these absolutes to `q1q2_dispatch_1.md`. Its `off` arm measured
> 9.81–9.96 for the same scene; this session's `base` measures 9.30–9.47. That
> is cross-session drift, which is exactly why both arms of an A/B go in one
> session.

## Every arm proves it did what it claims

`nn_prefetch_overlap.py` counts batches actually prepared on the worker and
prints the count. All six `base` arms report **0**; all six `ovl` arms report
**2** — one per render, two renders per `profile_scene(runs=2)`.

This is not ceremony. The overlap is gated on GPU projection *and* merge both
being active, it is skipped for a render's first batch by design, and an
exception inside it is caught and downgraded to render-thread preparation with
a DEBUG-level log. Each of those is invisible in a timing and would turn the
arm into a null measurement that reads as "neutral".

The count of 2 also states the ceiling: this scene renders in **two batches**
(23 frames then 7), the first never overlaps, so only the second — the smaller
one — can be hidden. A render with more batches has more to gain.

## Where the time went, and why it isn't more

Medians, RUN 2 inclusive seconds:

| stage | UHD base | UHD ovl | Δ | PREVIEW base | PREVIEW ovl | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| end-to-end | 9.36 | 9.19 | **−0.17** | 4.42 | 4.18 | **−0.24** |
| `arena preflight (batch)` | 1.624 | 1.054 | **−0.570** | 1.566 | 0.809 | **−0.757** |
| `merge collections + build BVHs` | 0.876 | 1.816 | **+0.940** | 0.943 | 0.974 | +0.031 |
| `- project_to_screen (prewarm)` | 0.737 | 1.258 | **+0.521** | 0.614 | 0.575 | −0.039 |
| `ray traced render total` | 6.213 | 6.651 | **+0.438** | 1.240 | 1.193 | −0.047 |

The preflight shrinks in both, as designed — by a third at UHD and by half at
PREVIEW. The two presets then diverge completely.

**At PREVIEW the overlap is nearly free.** Merge, projection and the render are
all unchanged, so the 0.757 s that leaves the preflight is genuinely hidden.
The render here is only 1.24 s of 4.42 s, so there is little GPU work for the
worker to contend with.

**At UHD it mostly is not.** The preflight drops 0.570 s, but merge grows
0.940 s and projection 0.521 s — those builds now run *beside* a live render and
share one T4, so their wall time inflates — and `ray traced render total` itself
grows 0.438 s for the same reason. The 1.8% that survives is what is left after
the contention. This is the same phenomenon the Mac report recorded from the
other side ("a readback on the worker waits out the whole queued render").

So the honest reading of the 16.8% the preflight costs at UHD is: roughly a
third of it can be moved off the render thread, and most of *that* comes back as
contention. Overlap is not a way to make preparation cheap; it is a way to hide
the part of it that the GPU is not already busy with.

`overlap_pool_headroom_fraction` (0.6) did not bite here — the render chunk
count is 11 at UHD and 4 at PREVIEW in **both** arms, so unlike Q1/Q2 the
derated worker headroom did not move window sizing.

## Output parity

```
UHD      1c89e4c1712a4cf6   base == ovl
PREVIEW  994adb9379c268ac   base == ovl
```

Byte-identical, and the same digests the Q1/Q2 and master-control sessions
produced. The docstring's "output-identical by construction" holds in practice.

## Recommendation

Turn it on. It is free in correctness terms, never lost a pairing here, and
helps most exactly where the profile is worst (PREVIEW, where preparation is
~68% of the render). Two caveats before making it the default:

* **One scene, one GPU, two batches.** A render with more batches should gain
  more; a render whose GPU is already saturated should gain less, and could
  lose. A multi-batch and a geometry-light scene both want measuring.
* **It does not address the underlying cost.** Scene preparation is still
  ~27% of a warm UHD render and ~68% of a PREVIEW one. Making it cheaper — the
  dependency-separated cache the audit files under Rank 11 — is the larger
  prize, and overlap does not substitute for it.

## Reproducing

```
uv run python scripts/kaggle/make_notebook.py --tag pf-overlap \
    --branch codex/q1q2-device-dispatch-20260911 \
    --step "uhd_base_1:python benchmarks/performance/nn_prefetch_overlap.py base UHD" \
    --step "uhd_ovl_1:python benchmarks/performance/nn_prefetch_overlap.py ovl UHD" \
    --step "uhd_ovl_2:python benchmarks/performance/nn_prefetch_overlap.py ovl UHD" \
    --step "uhd_base_2:python benchmarks/performance/nn_prefetch_overlap.py base UHD" \
    --step "uhd_base_3:python benchmarks/performance/nn_prefetch_overlap.py base UHD" \
    --step "uhd_ovl_3:python benchmarks/performance/nn_prefetch_overlap.py ovl UHD" \
    (and the same six at PREVIEW) \
    --env ALGAN_VIDEO_ENCODER=software --step-timeout 600 --out /tmp/nb.py
```
