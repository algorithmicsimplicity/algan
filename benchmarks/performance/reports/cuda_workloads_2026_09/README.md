# CUDA workload optimization, 13 September 2026

This work imports the explainer and graphics workloads and optimization work
from `claude/peaceful-carson-px98kh` at
`ebe2e3d1d249eb4d77e3b43fbcffbd3a3e213f14`, on top of master
`40aab2d7b110d2988e3b8095ebdb745a0208e854`. The implementation and reproducible
benchmarks are on `codex/gpu-workload-optimization`.

## Results

Warm medians from four measured samples per arm, beyond the imported branch's
optimizations. These are reductions in elapsed time, not increases in FPS.

| GPU / workload | Actual frames | Reference render | Candidate render | Time reduction | Whole run, including authoring |
| --- | ---: | ---: | ---: | ---: | ---: |
| T4 graphics UHD, coherent shadows | 55 | 88.357 s | 69.585 s | **21.2%** | 92.113 -> 73.329 s (**20.4%**) |
| T4 explainer UHD, pinned transfers | 30 | 3.395 s | 2.842 s | **16.3%** | 7.163 -> 6.636 s (**7.4%**) |
| GTX 1050 explainer UHD, pinned transfers | 30 | 9.473 s | 9.004 s | 5.0%, noisy | 18.289 -> 17.260 s (5.6%) |

A separate final-default run, including both enabled optimizations and the
small-queue fallback, confirmed **32.785 -> 25.486 s (22.3% less)** for 22 UHD
graphics frames and **3.499 -> 2.951 s (15.6% less)** for 30 UHD explainer
frames. Each arm had two warmups and two measured samples in mirrored order.
Whole-run times were 36.746 -> 29.477 s and 7.346 -> 6.870 s respectively.
The explainer remained byte-identical; the graphics comparison found 27
differing channels, maximum two. See `t4_final_*.json`.

The T4 graphics samples were tightly grouped: reference 88.09-88.66 s,
candidate 69.45-69.84 s. Its lossless video comparison found 25 differing
channels across 55 UHD frames, maximum difference two; the explainer was
byte-identical. Local UHD graphics differed in only three channels across 22
frames, maximum one; local MD graphics and both explainer experiments were
byte-identical. All satisfy the repository's channel tolerance of two.

The GTX 1050 MD graphics run also encountered memory-pressure compiler resets
inside its nominal warm phase. Its 60.57 -> 41.02 s render medians therefore
**are not evidence of a 32% steady-state speedup**. The full readings are
retained in `gtx1050_graphics_md.json`; the local conclusion instead uses
matched device-kernel measurements and pixel parity. The UHD diagnostic was
stopped after both arms completed two warmups, before the measured phase.

The T4's matched graphics queues compared **6,073,032 visibility scalars
exactly**. The sum of per-queue median device times was 24.179 -> 20.785 ms
(14.0% less). The large first reflection queue was 4.321 -> 3.166 ms (26.7%
less). The legacy full-fan fixture compared another 77,865 scalars exactly.
Small queues did not consistently improve, so the final default uses the new
schedule only with multiple lights and at least **16,384 events**. The
candidate readings above precede this conservative dispatch threshold.

On the GTX 1050, the matched primary queue improved 47.734 -> 39.249 ms
(17.8%), and the first reflection queue improved 24.335 -> 17.596 ms (27.7%).
The sum of per-queue medians was 122.218 -> 97.080 ms (20.6%). All 6,072,936
graphics visibility scalars and 77,865 legacy-fan scalars matched exactly.
The local probe likewise found
regressions in the smallest queues, supporting the size threshold. These
are device-kernel measurements from identical live inputs, independent of the
compiler-reset overhead in the whole-render diagnostic.

The separate T4 UHD profile reports shadow device time of 54.803 -> 35.604 s,
consistent with the workload speedup. Those profiles came from separate
sessions and serve only to attribute the gain; the alternating A/B above is
the performance evidence.

Pinned transfers and coherent scheduling are enabled by default. Fused sheet
gathers remain off: T4 explainer render medians were 3.438 s with fusion alone,
and 2.890 s with fusion plus the other changes, against 2.842 s for pinned
transfers alone. The more complex parallel shadow queue also remains off.

Machine-readable readings are in the adjacent `t4_*.json` and
`gtx1050_*.json` files. Raw run logs and videos remain in the local output
directories and the linked Kaggle notebooks.

## Changes under comparison

* **Coherent shadow scheduling:** adjacent CUDA lanes trace adjacent shading
  events for the same light. Every event/light cell retains its serial sample
  loop and reduction order. No rays, samples, bounces, or lighting terms are
  removed. Primary and deferred reflection/refraction queues both use it.
  `SETTINGS.raytracing.experimental.shadow_light_major` controls the schedule;
  small queues, single lights, and non-CUDA devices retain the reference.
* **Pinned frame transfers:** completed CUDA frames copy into owned page-locked
  host buffers. The current CUDA stream is synchronized before returning, so
  consumers receive ready data that survives render-arena reuse. Individual
  buffers are limited to 256 MiB, and pinning allocation failures fall back to
  pageable storage. `pinned_frame_readback` controls this path.
* **Fused sheet gathers:** the upstream `sheet_fused_stream` experiment is
  measured separately and together with the above changes. Its arithmetic and
  sorting order are unchanged.
* **Parallel shadow queue:** the upstream `shadow_ray_parallel` experiment
  remains opt-in. Its matched-input visibility check passed on T4, but the
  whole PREVIEW frame median was 5.14 s versus 5.12 s for the serial fan.

The upstream soft-shadow budget (16 primary rays per light, one ray per
secondary light row) is identical in all workload arms. It is a quality/cost
choice inherited from that branch, separate from the new scheduling and
transfer changes. Those new changes also have a legacy full-fan parity probe.

## Measurement method

`benchmarks/performance/workload_ab.py` warms each arm twice, then runs mirrored
orders twice (four measured samples per arm). Authoring and rendering are
recorded separately. Video encoding is software x264, lossless, with no added
fade-out. Final videos are decoded and compared across all frames, with a
maximum allowed channel difference of two. Warm render time includes the
normal render preparation, post-processing, frame transfer, and encoder drain.

`benchmarks/performance/shadow_schedule_ab.py` additionally replays every live
shadow queue from a real frame under both schedules. It checks each float
visibility value exactly, checks arena-pointer restoration, and records
alternating Taichi device-profiler times. These kernel times exclude authoring
and synchronization overhead; they are not whole-render speedups.

**Frame-count caveat:** the imported graphics storyboard appends animations
after its outer synchronized block. Its `--frames 30` render is actually 55
frames, and `--frames 12` is 22. The comparisons retain that authored scene
unchanged and report decoded frame counts. The explainer's requested and
actual counts match.

The local GPU is a 4 GiB NVIDIA GeForce GTX 1050, Windows/WDDM, driver 576.52,
PyTorch 2.7.1+cu128, with the diagnostic
`quadrants 1.3.1.dev0+gab9a58ab5.d20260905` build. Kaggle sessions explicitly
request and verify `NvidiaTeslaT4`, render through CUDA, and use the published
`algan-quadrants==1.3.0.post2` distribution and PyTorch 2.10.0+cu128. No Mac
measurements are used.

The local UHD graphics diagnostic repeatedly reset the compiler under memory
pressure; cold or cache-reloading samples must not be treated as steady-state
speedups. The repeated local graphics comparison therefore uses MD.

## Reproduction

Run the venv interpreter directly, one local render process at a time:

```powershell
.venv/Scripts/python.exe benchmarks/performance/workload_ab.py --scene graphics --quality MD --frames 12 --pairs 2 --arms reference,coherent --tag gtx1050_schedule_md
.venv/Scripts/python.exe benchmarks/performance/workload_ab.py --scene explainer --quality UHD --frames 30 --pairs 2 --arms reference,pinned,fused,combined --tag explainer_candidates
.venv/Scripts/python.exe benchmarks/performance/shadow_schedule_ab.py --scene graphics --pairs 3 --tag shadow_schedule
.venv/Scripts/python.exe benchmarks/performance/shadow_schedule_ab.py --scene soft --width 256 --height 144 --budget 0 --pairs 2 --tag shadow_legacy
.venv/Scripts/python.exe -m pytest -q --fast
```

Kaggle was submitted through MCP using `scripts/kaggle/make_notebook.py` and
`scripts/kaggle/runner.py`, with software encoding. The notebooks clone a
published branch; they contain no injected patch payload.

The main workload A/B ran commit
`a7daa677c6491e103252afdcd6778bfc67e76cc0`; the matched-queue probe ran
`28607bf3e7b929ad27c7607fb0d47d48692043cb`; the final-default check ran
`3ed3fdb2453c9a248e8c3d163efcca9f18b61425`. Compiler distributions are recorded
in the workload JSON, rather than inferred from the import banner.

* [Upstream profile and experiments](https://www.kaggle.com/code/algorithmicsimp/algan-cuda-workloads-0913-probe)
* [UHD workload comparisons](https://www.kaggle.com/code/algorithmicsimp/algan-cuda-workloads-0913-candidates)
* [Matched shadow queues and legacy fan](https://www.kaggle.com/code/algorithmicsimp/algan-cuda-workloads-0913-queues)
* [Final defaults, including the small-queue fallback](https://www.kaggle.com/code/algorithmicsimp/algan-cuda-workloads-0913-final)

Per the user's request, validation uses the fast suite and the focused
benchmarks above; the full test suite is not run.

Final fast-suite result on the GTX 1050: **607 passed, 3,806 deselected, two
warnings**, in 185.79 s. The run exceeded the suite's 75 s timing budget on
this host; no tests were added to or removed from its opt-in membership.
The warnings concern Quadrants template-mapper caching. Ruff on the changed
source/benchmark/test files and `git diff --check` both passed. The new
unmarked feature tests were collected but not executed by the fast-only run;
the focused benchmarks performed the visibility and decoded-frame assertions.
