# Integration into master through PR 115

PR 115 consolidates the Mac work from `claude/eager-cori-h61xc8` and the
16 `codex/mac-*` measurement branches. Its base is `master`, including the
changes through `1b4683a52221c7ed99ad2c0659a5fee6afdd1e9e`.

## What is retained

- The original Mac bring-up, arena budgeting and Metal interoperability work.
- The validated MPS row-reorder fix and bloom kernel/fallback corrections
  from PR 112, which had previously merged only into the Claude working branch.
- Depth-buffer reuse and sibling-weight kernels, their tests, and their A/B
  harness. **Both candidates remain disabled by default:** their whole-render
  measurements regressed despite faster individual helpers.
- Live benchmark output, completion markers, bounded teardown and timeout
  diagnostics, plus the result reports in this directory.
- All unique diagnostic scripts from the measurement branches, together with
  their original Git history. `DIAGNOSTIC_SOURCES.json` records which commit
  supplied each script before integration formatting.

The old `_mac_globals_probe.py` and `_mps_reorder_probe.py` are named
`_mac_globals_probe_taichi.py` and `_mps_reorder_probe_taichi.py` in the integrated
tree. They contain runtime kernel annotations and must follow the repository's
kernel-specific lint/format rules. The diagnostic imports use the new names.
Some historical probes deliberately reproduce the broken operation or stop
after capturing a few launches; they are opt-in investigation tools, not
production paths or general acceptance tests. Original request files and
script spellings remain available at the recorded commits.

## Conflict resolution

The overlapping edits were in `sheets.py`. Master's newer CUDA packed-key,
local pixel-sort and group-reuse paths are preserved. Safe int32 narrowing is
applied only after the CUDA/local fast paths decline the operation, so their
required int64 IDs remain intact. Sorted MPS groups retain consecutive unique;
CPU/CUDA keep master's existing optimization gates. Added tests cover stable
tie order, unchanged inputs, safe-bound fallback, dispatch to the existing
packed path, and sparse/repeated group IDs on the active render device.

Master's other renderer and authoring changes, including rough dielectric
shading, premultiplied output and newer CUDA queue optimizations, are retained.

## Measurements and their limits

The runner exposes hardware-accelerated, paravirtualized Metal. No matched
physical Mac was measured, so the virtualization penalty is not isolated.
The repaired GPU remained slower than CPU in the matched tests; unrelated
runner jobs must not be combined into one timing comparison.

| Measurement | Result on that runner |
| --- | --- |
| Bloom, corrected fallback versus kernels | 193.219 to 127.246 seconds, 34.1% less total time |
| Matched CPU/GPU, run 79 | 61.976 / 66.557 seconds; GPU 7.4% slower |
| Matched CPU/GPU, run 80 | 85.913 / 106.700 seconds; GPU 24.2% slower |
| Depth reuse, warm GPU A/B | Candidate 31.2% slower; disabled |
| Sibling weights, warm GPU A/B | Candidate 14.9% slower; disabled |

Nested host timers include queued work and synchronization. Their values
overlap and are not additive device-kernel timings. See `REORDER_FIX.md`,
`BLOOM_FIX.md`, `DEPTH_BUFFER_REUSE.md` and `SIBLING_WEIGHTS.md` for methods,
parity checks, individual samples and evidence links. `FINDINGS.md` and
`README.md` retain the earlier chronology and hypotheses.

## Validation and branch cleanup

Completed local checks: 209 focused renderer/integration tests (20 skipped),
550 fast tests, 14 compiler-setting tests, and 9 cleanup tests, including a
real Git atomic-deletion race. Ruff and formatting checks passed. At the
user's explicit request, the remaining full local suite was stopped and the
integration was merged without waiting for additional validation. Its partial
run is not a full-suite pass. CUDA hardware was unavailable locally.

The configured Metal request exercises arena offsets, canaries, reordering,
bloom and sheet helpers, then renders UHD twice with both rejected candidates
disabled. It runs separately from this requested merge, as do the standard CI
checks; their results must be read from GitHub. The render child has a 15-minute
work deadline and a separate 60-second teardown deadline. Older report counts
describe their original candidate, not this integration.

The integration commit retains every recorded investigation head as an
ancestor, including probe-only histories whose obsolete request files are
superseded in the final tree. Merge PR 115 with a **merge commit** to keep that
history; squash/rebase would defeat this preservation and the cleanup guard.

`INTEGRATION_BRANCHES.json` records the exact heads of the 16 measurement
branches and the original Claude working branch. After PR 115 is merged into
master, the scoped cleanup workflow removes only entries whose heads are
unchanged, whose history is in master, and which no open PR uses as head or
base. It resolves PR 115's final integration head from the merged PR itself.
An atomic push with an explicit lease per ref prevents deleting a branch that
changes during cleanup. Unrelated branches are outside the manifest.

The cleanup script defaults to a dry run; the master-only workflow applies it
and uploads its decision log. If a recorded branch is still active or has new
commits, it is preserved for a later decision. Removing branch names does not
remove the retained commits, diagnostic sources, or benchmark provenance.
