# UX audit validation record — 10 September 2026

Companion to [the UX audit](UX_AUDIT_2026-09-10.md). This is a snapshot of verified results, not a claim that every configured test or backend passed.

## Source and scope

- Repository: `algorithmicsimplicity/algan`.
- Audited base: `d055e77f10dc28e3908329158b2b41d80826e9da`.
- Branch: `codex/ux-audit-2026-09-10`.
- Code commit for the eleven fixes, including the hierarchy retry: `277285a8c7457457d16001017164826b739cb2b4`.
- The temporary branch publication workflows remove themselves; the net diff does not retain them.
- A later review pass folded the audit's regression tests into `tests/unit_tests/test_ux_regressions.py`, renamed `_reject_negative_runtime`, and added tilde expansion to the Project frame path. The **Review follow-ups** section below records why and what was re-run; every figure in this document is from that reviewed state unless it names a GitHub run.

## Confirmed checks

| Check | Result | Qualification |
| --- | --- | --- |
| Targeted package regressions: the audit's own tests | **50 passed**, 0.67 seconds | Full locked package environment on GitHub; before the final JavaScript-only hierarchy retry fix. They lived in `tests/unit_tests/test_ux_audit.py` at the time and are now in `test_ux_regressions.py`; see **Review follow-ups**. |
| Curated cross-cutting suite: `pytest -q --fast` | **604 passed**, 3,466 deselected, 2 warnings; 84.90 seconds | Includes the existing pixel-compared fast render. Before the final JavaScript-only fix; no later Python production changes. |
| Final viewer regressions: `node --test tests/viewer/test_viewer_async.cjs` | **10 passed** | Includes B11. Passed locally and in the second GitHub validation's patch/test step. Deterministic DOM/network doubles, not a real browser. |
| Changed Python files: Ruff 0.12.4 lint/fix and formatting | **Passed** | Repository-pinned version, run before publishing the Python changes. |
| Python syntax, JavaScript syntax, whitespace/error checks | **Passed** | Python compilation, `node --check`, and `git diff --check`. |
| Source-isolated local Python checks | **15 passed** | Real source and Torch with unavailable renderer/registration boundaries replaced. Not package integration coverage. |

These counts overlap; they must not be added together as a count of distinct tests. The original nine viewer regressions failed against the original client. The tenth test separately reproduced failed hierarchy expansion before its fix.

The complete first validation is GitHub Actions run **34427555338**, job **102715903142**. Its `ux-audit-validation` artifact contains the targeted and fast-suite logs. The runner used Ubuntu 24.04, Python 3.11.16, CPU rendering, `ALGAN_USE_DAEMON=0`, and `uv sync --locked --all-extras --dev`.

### Fast-suite timing

The initial 84.90-second fast run exceeded the repository's 75-second budget. First-use compilation/cache work is included; no warm-time conclusion or performance improvement is inferred from it. The two warnings report disabled Quadrants template-mapper caching for objects that cannot be weak-referenced. No tests were removed from the suite and no baselines were changed to hide timing or image differences.

## The portable gate and the warm repeat

Second GitHub Actions run **34427955688**, job **102717106299**, successfully applied and tested B11, published code commit `277285a8c7457457d16001017164826b739cb2b4`, removed its temporary workflow, and installed the locked environment. Its wider `pytest -q tests/unit_tests tests/fast --durations=10` step and the warm `pytest -q --fast` step after it never reported, so for a time this record carried no confirmed result for either. Both have since been run to completion on the reviewed branch; the results are in **Review follow-ups** below. A workflow's pending status was neither a pass nor a diagnosed failure, and the first-run results above were not substituted for it.

## Review follow-ups

A review of the branch found one failure the audit's own runs could not have seen, because the check that catches it is not in `--fast` and the second GitHub run never reached it:

- **`tests/unit_tests/test_fast_suite_curation.py::test_the_membership_table_matches_the_markers` failed.** `test_ux_audit.py` carried 43 `fast` markers with no row in the membership table in `tests/README.md`, which that audit requires of any file joining the suite. CI runs `tests/unit_tests tests/fast`, so this would have merged red. **Fixed** by folding the tests into `tests/unit_tests/test_ux_regressions.py` — already in the table, already marked per test, and already the file for the authoring surface these tests cover — and deleting `test_ux_audit.py`. No test was dropped and no marker changed: the same 43 stay in the fast suite.

Two follow-ups were taken from the same review:

- `_reject_negative_runtime` is now `_reject_invalid_runtime`. B03 widened it to reject non-finite and non-numeric values, which left the name describing a third of what it does.
- `_ProjectSceneRun.prepare_frame_path` now calls `expanduser()` before its directory probe, as `_resolve_output_destination` already did. `Path("~/stills").is_dir()` is False however real the directory is, so B08's fix would not have recognised a tilde path as a directory. One test covers it, and it fails without the change.

### Re-run on the reviewed branch

Local, on the full locked environment: Linux, Python 3.11.15, CPU rendering, Quadrants 1.3.0. These are the portable gate and warm repeat the second GitHub run never reported.

| Check | Result |
| --- | --- |
| Portable gate, what CI runs: `pytest -q tests/unit_tests tests/fast` | **3,881 passed, 178 skipped**, 0 failed; 583.92 seconds |
| Warm repeat: `pytest -q --fast` | **604 passed**, 3,467 deselected; 32.88 seconds, 44% of the 75-second budget |
| Viewer regressions: `node --test tests/viewer/test_viewer_async.cjs` | **10 passed** |
| Whole package: Ruff 0.12.4 `check` and `format --check` | **Passed** |
| Sphinx structural build: `docs/make_and_open_docs.py --skip-examples --no-open` | **Succeeded** |

Before the fold, that same gate was **3,879 passed, 178 skipped, 1 failed** — the curation audit above. The two added passes are that audit and the new tilde test; the fast count is unchanged at 604, which is the check that the fold moved all 43 markers and invented none.

The fast suite also ran **113 seconds cold** on this machine, over the 75-second budget, before settling at the 33 seconds above. That is the first-use Taichi compile the suite's own documentation warns about, not a cost this branch adds: its tests account for about 0.6 seconds. The same effect shortened the portable gate from 27 minutes on a cold cache to the 9:43 recorded here. No conclusion about performance is drawn from either.

Each of the 50 tests the audit added was also run against the audited base `d055e77` in a detached worktree: **42 failed**. The 8 that passed are the six `-inf` runtime cases and the one `-inf` wait — already caught by the negative guard B03 replaced — and the bare-name Project case B08 was required not to disturb. Every test therefore either reproduces the defect it names or is a control for one.

## Uncovered or blocked checks

While the audit itself was being written, the local `pytest -q --fast` attempt stopped during conftest import with `ModuleNotFoundError: mapbox_earcut`, and other full-environment dependencies were unavailable too, which is why the results in the table above came from GitHub rather than from that machine. The re-run recorded under **Review follow-ups** was not blocked in that way; it used a complete environment.

The heavy full-render/path-traced baseline suites, GPU backends, real-browser interaction, keyboard/assistive-technology behavior, and manual rendered visual review are still **not** validated. The passing fast render provides narrower pixel-comparison coverage and does not establish those broader guarantees. This audit does not claim physical-rendering correctness or performance improvements.

The heavy suites are the gap worth naming precisely, because two of the fixes change layout arithmetic. Every scene in `tests/full_renders` calls `arrange_in_line` with a unit `RIGHT`, no `start_at_first` and no `align_to`, and none calls `arrange_between_points` — so B04 and B05 are inert for those baselines by inspection of the call sites, and the fast suite's pixel-compared render (which does call `arrange_in_line(RIGHT, ...)`) is unchanged. That is an argument plus one render, not a run of those suites, and no baseline was updated.
