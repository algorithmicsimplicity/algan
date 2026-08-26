# Brief: apply and verify the recovered sheet-chain patch

`scratch_perf/r2/patches/ox_sheet_host_chain.patch` is a round-2 deliverable
that was never verified or landed (the round ended; the patch was recovered
from the dev box). It adds three kernel replacements for host passes in the
sheet chain — `RASTER_PAIR_EXPAND_KERNEL` (`pair_expand_count`/`_write`, the
kernel arm of `_class_pairs_flat`), `SHEET_BAND_STATS_KERNEL`
(`band_stats_reduce`/`_rep_orig`) and `SHEET_SHELL_CEILING_KERNEL`
(`solid_shell_ceiling`) — plus the `compact_sheets` restructuring that calls
them and an extension of `benchmarks/_sheet_kernel_check.py` with cases for
all three. Your job: apply it, verify every claim it embodies, fix what
fails, and report. Read `CLAUDE.md` (Taichi gotchas, linting) and
`scratch_perf/r3/ox/REPORT_window_pairs.md` §1 and §5 first — that audit's
acceptance contract governs the pair-expand kernel (row order is
load-bearing through two stable sorts).

You are in `/home/user/algan`, branch `claude/algan-t4-optimization-b5a10b`.
Do not commit, do not push. `ALGAN_USE_DAEMON=0` everywhere. No GPU here —
the Taichi CPU backend is real verification for integer host-pass kernels.

## Steps

1. `git apply scratch_perf/r2/patches/ox_sheet_host_chain.patch` (it checked
   clean against HEAD). Read the full diff afterward as a reviewer: the
   patch was written against a slightly older tree, so check every hunk
   landed where it thinks it did (grep the call sites; do not trust context
   lines alone).
2. **Run the patch's own harness**: `benchmarks/_sheet_kernel_check.py` —
   every case must pass bitwise. Confirm the new `pairs_case` compares
   values AND row order AND dtype; strengthen it if it does not.
3. **Kernel-compile unit test.** Add a test under `tests/unit_tests/` that
   launches each of the five new kernels on small real inputs and asserts
   equality with the torch arm. A host-side check cannot see a Taichi
   scoping/compile error (a local first assigned inside an `if` does not
   exist outside it — that exact defect shipped once). Break a kernel
   deliberately, confirm the test fails, restore it, say so.
4. **A/B renders, one toggle at a time, separate processes** (the toggles
   are import-time module globals — an in-process flip does nothing): arm
   off vs arm on, nn scene at PREVIEW,
   `SETTINGS.computing.available_memory_override` pinned identically,
   `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]`,
   `benchmarks/_video_diff.py`, 0 differing pixels required per toggle.
5. **Prove each kernel actually ran in its A/B** — a pass on a path never
   taken is not a pass. Count kernel launches (profiler hooks name every
   Taichi kernel) or instrument in the arm's env. In particular:
   the shell-ceiling path runs only when a **closed solid at opacity < 1**
   is on screen, which the nn scene likely lacks — check, and if so run that
   toggle's A/B on a probe scene with a translucent `Sphere`/`Cube` (put it
   under `scratch_perf/r3/probes/`), and additionally run
   `benchmarks/_opacity_alpha_check.py` if its harness applies. State for
   every toggle which code path its A/B exercised and how you know.
6. `uv run -m pytest -q --fast` twice (cold kernel compiles inflate run 1;
   report both timing lines). Then
   `uv run -m pytest -q tests/unit_tests/test_raster*.py tests/unit_tests/test_sheet*.py`
   plus files covering anything you touched, plus your new test. Not the
   whole suite in one process (RAM).
7. `uv run ruff check --no-fix` and `uv run ruff format --check` on touched
   files (`*_taichi.py` is excluded from format — leave it).
8. Re-run `scratch_perf/r3/probes/count_window_pairs_dispatches.py` (copy
   and adapt if needed — do not edit it in place) with the pair-expand
   kernel active: report dispatched ops + syncs per `_window_pairs` call,
   before vs after.

If a kernel fails verification and the fix is small and clearly within the
patch's own intent, fix it and note it; if a failure looks structural,
leave that toggle default OFF, make the default path byte-identical to
pre-patch, and say exactly what fails and how.

## Traps

- `tests/full_renders` baselines are per-machine and FAIL here —
  pre-existing, do not chase.
- The Taichi offline cache does not invalidate on `@ti.func` edits — clear
  it (`clear_cache(taichi_kernels=True)`) after editing a kernel before
  re-verifying.
- Never edit a `*_taichi.py` while one of your renders is running.

## Report

`scratch_perf/r3/ox/REPORT_sheet_chain_verify.md`: hunk-placement review
findings; each verification's output verbatim; the per-toggle
path-exercised proof; the dispatch census before/after; anything you fixed
(with the defect named precisely); everything you did NOT verify — T4
timings belong there (no GPU here; the operator runs the T4 A/B after).
