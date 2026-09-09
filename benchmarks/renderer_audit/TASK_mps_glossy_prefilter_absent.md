# Closed investigation: missing MPS glossy reflection on an older branch

**Status: closed as a current-master task, source checked 2026-09-09.**
The glossy-prefilter test is no longer in `tests/mps_known_failures.py`.
The remaining entry concerns `ti.real_func` early return, not a missing render.

## What the evidence established

The older branch `claude/awesome-fermi-lrapoq` at `9583b4a` produced no reflected
signal in the calibration window. Its test reported `nan` for the reflection
spread because the signal sum was zero. CPU and MPS-friendly-on-CPU control
renders passed, but that comparison did not establish a defect on master.

[DESIGN_mps_support.md §4.5](../../algan/rendering/DESIGN_mps_support.md#45-d-the-prefiltered-reflection--it-never-reproduced-on-master)
records the run/commit table. Master already passed the ordinary test at
`0eacd92` after the Mac integration, before the xfail entry arrived. Subsequent
strict unexpected passes showed the entry was stale. Removing it restored a
normal regression test; no particular source fix was isolated for the old
branch's missing reflection.

The shared-command-queue hypothesis was tested with that route disabled and
did not explain the difference. Do not attribute the result to that optimization
or claim the branch-only failure still blocks MPS support.

## When to reopen

Reproduce the failure on an exact current commit, using the committed
`scenes/calib_glossy.json` fixture and
`tests/unit_tests/test_glossy_prefilter.py::test_prefiltered_reflection_is_substantially_wider`.
Use [the GPU harness guide](../../agent_guidance/gpu_harnesses.md) for runner
setup. Keep CPU, MPS-friendly CPU and MPS arms on the same source/settings and
verify that a passing result is a rendered comparison, not a skip.

If the reflected signal is genuinely absent, inspect `gl_main`'s sigma markers,
the successive `gl_pyr` levels and the final composite. Those separate missing
glossy writes from a pyramid or sampling failure. Do not lower the width-ratio
threshold to hide a zero-signal failure, and do not add a new expected failure
based only on a different branch's result.

The longer original investigation brief remains in Git history. The implementation
record and measurements remain in `DESIGN_mps_support.md`; there is no separate
open work queue in this file.
