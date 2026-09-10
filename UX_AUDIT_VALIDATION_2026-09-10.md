# UX audit validation record — 10 September 2026

Companion to [the UX audit](UX_AUDIT_2026-09-10.md). This is a snapshot of verified results, not a claim that every configured test or backend passed.

## Source and scope

- Repository: `algorithmicsimplicity/algan`.
- Audited base: `d055e77f10dc28e3908329158b2b41d80826e9da`.
- Branch: `codex/ux-audit-2026-09-10`.
- Final code commit, including the hierarchy retry fix: `277285a8c7457457d16001017164826b739cb2b4`.
- Later audit/validation commits change documentation only. The temporary branch publication workflows remove themselves; the final net diff does not retain them.

## Confirmed checks

| Check | Result | Qualification |
| --- | --- | --- |
| Targeted package regressions: `pytest -q tests/unit_tests/test_ux_audit.py tests/unit_tests/test_viewer_client.py` | **50 passed**, 0.67 seconds | Full locked package environment on GitHub; before the final JavaScript-only hierarchy retry fix. |
| Curated cross-cutting suite: `pytest -q --fast` | **604 passed**, 3,466 deselected, 2 warnings; 84.90 seconds | Includes the existing pixel-compared fast render. Before the final JavaScript-only fix; no later Python production changes. |
| Final viewer regressions: `node --test tests/viewer/test_viewer_async.cjs` | **10 passed** | Includes B11. Passed locally and in the second GitHub validation's patch/test step. Deterministic DOM/network doubles, not a real browser. |
| Changed Python files: Ruff 0.12.4 lint/fix and formatting | **Passed** | Repository-pinned version, run before publishing the Python changes. |
| Python syntax, JavaScript syntax, whitespace/error checks | **Passed** | Python compilation, `node --check`, and `git diff --check`. |
| Source-isolated local Python checks | **15 passed** | Real source and Torch with unavailable renderer/registration boundaries replaced. Not package integration coverage. |

These counts overlap; they must not be added together as a count of distinct tests. The original nine viewer regressions failed against the original client. The tenth test separately reproduced failed hierarchy expansion before its fix.

The complete first validation is GitHub Actions run **34427555338**, job **102715903142**. Its `ux-audit-validation` artifact contains the targeted and fast-suite logs. The runner used Ubuntu 24.04, Python 3.11.16, CPU rendering, `ALGAN_USE_DAEMON=0`, and `uv sync --locked --all-extras --dev`.

### Fast-suite timing

The initial 84.90-second fast run exceeded the repository's 75-second budget. First-use compilation/cache work is included; no warm-time conclusion or performance improvement is inferred from it. The two warnings report disabled Quadrants template-mapper caching for objects that cannot be weak-referenced. No tests were removed from the suite and no baselines were changed to hide timing or image differences.

## Wider validation status at this snapshot

Second GitHub Actions run **34427955688**, job **102717106299**, successfully applied and tested B11, published code commit `277285a8c7457457d16001017164826b739cb2b4`, removed its temporary workflow, and installed the locked environment.

At the last verified status, its wider `pytest -q tests/unit_tests tests/fast --durations=10` step was **still in progress**. The subsequent warm `pytest -q --fast` step and final evidence artifact were **pending**. There is therefore **no confirmed result for the wider portable gate or warm repeat in this record**. A workflow's pending status is neither a pass nor a diagnosed failure, and the confirmed first-run results above are not substituted for it.

## Uncovered or blocked checks

The normal local `pytest -q --fast` attempt stopped during conftest import with `ModuleNotFoundError: mapbox_earcut`; other full-environment dependencies were also unavailable locally. The passing package results above came from GitHub, not from that blocked local invocation.

The heavy full-render/path-traced baseline suites, GPU backends, real-browser interaction, keyboard/assistive-technology behavior, and manual rendered visual review were not validated. The passing fast render provides narrower pixel-comparison coverage and does not establish those broader guarantees. This audit does not claim physical-rendering correctness or performance improvements.
