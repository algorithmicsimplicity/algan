# MPS ndarray launch plans

The Quadrants branch of `algan/utils/taichi_fast_launch.py` accepts both torch
CPU/CUDA tensors and Quadrants `Ndarray` instances, including the
`ExternalMetalNdarray` instances created by `mps_zero_copy.py`. It uses the
existing `quadrants_patches/0001-metal-zero-copy-ndarray.patch`; there is no new
compiler patch, wheel requirement or scalar-value context cache.

## Cache boundary and binding

A plan stores only the compiler's materialized and compiled kernel handles.
Argument annotations are classified once per kernel. A hit constructs a
**fresh** launch context, binds the current runtime arguments, and launches the
already compiled specialization. A plan never retains arrays, scalar values,
buffer handles, byte offsets, or runtime array extents.

A Quadrants ndarray contributes a tagged tuple of the compiler mapper's
features: element type (the mapper's primitive-type identity optimization
included), rank, gradient requirement, annotation boundary behavior and
`_qd_layout`. The representation tag prevents torch and native ndarrays from
sharing a plan. Array identity, extents and byte offset are deliberately not
key features. Vector/matrix element shape is part of the element type. Template
arguments keep the existing dispatcher rules; non-template scalar values are
rebound, not keyed. In particular, changing a float, a large integer, or the
sign of zero does not require a new plan.

On every hit, native ndarray `(slot, current_array.arr)` bindings are sent to
`LaunchContextBuilder.set_args_ndarray` in one batch. The patched C++ setter
records each imported slice's byte offset and byte size. It is incorrect to
bind an MPS tensor's `data_ptr()` as a host/CUDA address. A **CPU** torch tensor
still uses the same external-array setter as Quadrants' original path; this
is a supported binding, not a claim of zero transfer cost on Metal. Mixed
bindings are checked separately from zero-copy coverage.

First calls retain the original compiler's full validation and compilation.
A different element type, rank, layout or template configuration misses the
plan and takes that original path. Kernels with unsupported annotations,
return values, autodiff, printing, graphs or checkpoints, keyword calls, and
gradient-bearing/noncontiguous/foreign-device tensors retain their fallbacks.
Gradient-bearing and reset native ndarrays also fall back. No public kernel
signature changes. The Taichi backend's dispatch implementation is unchanged.

## Ordering, lifetime and resets

The wrapper order is unchanged: render-arch guard, zero-copy conversion and
fences, fast dispatcher, original compiler launcher. The zero-copy wrapper
continues to own buffer imports, storage lifetime and Torch/Quadrants queue
ordering. Cache hits do not bypass that wrapper.

Each launch has its own argument lists and launch context. Per-kernel plan
bookkeeping is published once with `dict.setdefault`, retaining the existing
Python/GIL concurrency model; this is not a new promise that cold compilation
can be raced or that runtime resets can run concurrently with rendering.
`Kernel.reset` drops the compiled plans. No launch context or previous array
binding is reused across calls or program resets.

## Telemetry

Detailed accounting is opt-in and is intended to be switched between render
jobs, not concurrently with worker launches:

```python
from algan.utils import taichi_fast_launch as fast

fast.set_telemetry_enabled(True)
fast.launch_report(reset=True)  # clears counters, not cached plans
# Render the workload, then wait for its worker threads to finish.
report = fast.launch_report()
fast.set_telemetry_enabled(False)
```

`report` contains `kernels` and `totals`. Each row identifies the function and
compiler arch and counts these mutually exclusive completed-call outcomes:

- `fast`: a successful Algan plan hit.
- `quadrants_cache`: an actual successful
  `LaunchContextBufferCache.populate_launch_ctx_from_cache` return on the
  original path; eligibility alone is never counted as a hit.
- `cold`: an original call that successfully recorded a new Algan plan. One
  call can both record a plan and hit the compiler's context cache -- arguments
  launched on the original path first (an off arm, a fallback) leave that cache
  warm for them -- and it is counted `cold`, because a run that re-records plans
  is the thing this accounting exists to expose.
- `fallback`: another original call without a context-cache hit. `reasons`
  identifies disabled dispatch, unsupported arguments, failed plan recording,
  and other deliberate fallback conditions.
- `error`: a call that raised, including invalid arguments and verify failures.

The observer uses a thread-local stack for nested/worker calls and a lock for
report aggregation. Reports retain names and counters only. Enabling the
observer wraps the compiler cache once; while disabled the observer returns
the original result with only a boolean guard. No per-kernel accounting or
report locks are used on ordinary fast hits when telemetry is disabled.

The existing global `STATS` remain engagement counters. On **Quadrants**,
`slow` now counts every dispatch to the original path (including original
cache hits and calls that raise), not just cold plans. `fast` counts attempted
plan hits. Use the detailed report for complete outcome accounting. The
Taichi backend retains its previous counter semantics.

A supported render is covered when every eligible launch after warming its
**fast-plan key** is a hit. A compiler specialization can have separate torch
and native-array fast-plan keys. Genuine new template configurations during
later frames are cold plans, not repeated misses. Every unsupported launch
must be listed with its reason; a zero slow count must not be inferred from
partial engagement counters.

## Validation

Portable feature tests live in
`tests/unit_tests/test_quadrants_ndarray_launch.py`, alongside the existing
fast-launch, zero-copy and launch-pairing tests. They compare actual results,
not only specialization keys: rebinding buffers and extents, changing scalar
values/templates, vector/matrix elements, layouts, mixed array
representations, gradients, object lifetime, reset, actual original-cache
hits, rejected arguments, and concurrent warmed launches. Metal-only tests
exercise slice offsets and the original/fast A/B paths on the same GPU.

`ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1` (or runtime `fast.VERIFY = True`) recomputes
the compiler specialization on fast hits. This guards plan selection, **not**
binding correctness; direct-output tests remain necessary.

Run the same-job acceptance/performance probe on the Mac harness:

```text
.venv/bin/python -m pytest -q tests/unit_tests/test_quadrants_ndarray_launch.py tests/unit_tests/test_taichi_fast_launch.py tests/unit_tests/test_mps_zero_copy.py tests/unit_tests/test_taichi_launch_pairing.py
.venv/bin/python benchmarks/_mps_launch_cache_check.py --scenes all
```

Set arms to `mac-mps,linux-cpu`, `taichi_wheel_run_id="none"`, `latex=true` and
`ALGAN_VIDEO_ENCODER=software`. Read `agent_guidance/gpu_harnesses.md` for the
request-file transport and device verification. The probe checks the resolved
MPS/Metal device rather than trusting the arm name. It emits results after
each workload and saves JSON and PNGs under `algan_outputs/launch-cache/`.

The `all` probe samples two times from the fast scene, every full-render scene,
and every path-traced scene; it is **not** every-frame baseline coverage. Its
same-process off/on reference comparisons use PNGs (before lossy H.264), keep
zero-copy enabled in both arms, record complete launch outcomes plus staged
and host arguments, and fail on unexpected fallbacks or staged MPS arrays.
Detailed verification and telemetry are disabled for timed alternating arms.
The microbenchmark reports both native-ndarray dispatch and the complete
Torch-wrapper path, and separates host enqueue from synchronized completion.
No speedup or complete renderer coverage should be claimed from CPU-only
results or a non-engaged fast path.
