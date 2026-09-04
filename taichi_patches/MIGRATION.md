# The Quadrants migration, as executed

What `PLAN.md` proposed, this is what actually happened when it was run —
2026-09-04, in one session, from the fact-finding gate through to Quadrants
being the default compiler with its own patch set. `PLAN.md` remains the
design and the reasoning; this is the record: what was measured, what the
measurements changed, what shipped, and what is still not true.

Read `PLAN.md` §6.1 for the gate's own numbers and
`../quadrants_patches/README.md` for the patches. This file is the index over
both, plus the things neither of them is the right home for — the corrections,
the ledger of verified-versus-not, and how to re-run any of it.

**Status in one line: the base decision is made and executed on CPU and CUDA;
the Apple path is fixed and verified on the fast scene and on the heaviest
scene in the suite, the crash that blocked the latter having turned out to be
a leak in Algan's own MPS import cache and been fixed on master; the seven
Quadrants patches apply and compile; what remains unverified is everything
that needs a CUDA device (0003 on sm_61, 0005–0007 in PTX and pixels).**
§10 is the current ledger; §6 and §9 are kept as the record of how it got
there.

---

## 1. The decision

The gate passed on all four criteria, three by a wider margin than they asked.
`PLAN.md` §6.1 carries it; the short form:

| criterion | result |
| --- | --- |
| macOS build green | **green for Quadrants, red for Taichi** — on the same runner image, Taichi 1.7.4 + `taichi_patches/` no longer builds at all |
| `tests/fast` pixel deltas ≤ 2 | **zero** — byte-identical mp4, 0 of 37,635,840 channel samples |
| the three upstream repros | do not distinguish the bases **except in Quadrants' favour** (#8745) |
| no user on Python 3.9 / macOS < 13 | met; Intel Macs *refuted* as a loss, cp39 the only real one, and the maintainer confirmed no 3.9 users |

The one criterion that needed correcting after the fact was the second, and it
is the most important paragraph in this file: see §4.

## 2. What shipped

Twenty-eight commits on `claude/taichi-patches-rebase-decision-0zvnir`,
89 files, +6,950/−1,580. In dependency order rather than commit order:

**The gate itself.** `taichi_patches/0003-literal-operator-whitespace.patch`
(gate step 1); `benchmarks/_upstream_repro_874{4,5}.py`, `_8794.py`;
`scripts/gate/{quadrants,taichi}_macos_build.sh`, `quadrants_linux_build.sh`,
`backend_pixel_ab.py`; `PLAN.md` §6.1.

**The warm-start port.** `algan/utils/taichi_warmstart.py` now memoizes on
either compiler — `get_pos_info` on both, plus `get_source_info_and_src` and the
per-line `textwrap.fill` behind `get_tree_and_ctx` on Quadrants. No Quadrants
*release* carries the upstream memo (it landed two days after v1.3.0 and is in
no release tag), so this was port-or-pin and the maintainer's policy is to track
releases. `algan check` now reports when the memoization is off, because the
version gate silently refusing to fire is what cost ~25 s per render for the
length of the evaluation. Covered by `tests/unit_tests/test_taichi_warmstart.py`
(34 tests, one marked `fast`) and `benchmarks/_taichi_warmstart_check.py`.

**The flip.** `quadrants>=1.3.0,<1.4` is a runtime dependency and `BACKENDS[0]`;
`taichi` moved to an extra and stays a first-class arm. Python floor 3.10, CI
matrix, docs and `uv.lock` to match. It exposed four backend bugs, every one
shaped like a silent wrong answer: an unrecognised `@ti.func` background falling
back to the Python path, `mps_zero_copy._ndarray_positions` returning an empty
map through a bare `except`, three `get_runtime().prog` reads, and the stdout
banner. It also caught two workflows that would have installed the patched
*Taichi* wheel and then compiled with Quadrants.

**The patches.** `quadrants_patches/` with 0001 (Metal zero-copy, ported), 0002
(`ContinueStmt` + diagnostics, minus the hunk Quadrants already fixes) and 0003
(pre-Volta CUDA, new). Two hunks disappeared because the newer base already
carries their fixes, which is the shrinking patch set the plan predicted.

**The clean-up the flip implied.** 19 `benchmarks/` files no longer open a
second compiler behind Algan's back (10 routed, 8 now refuse to run under the
wrong backend rather than reporting zeros, 1 commented); the vendored Manim
shim moved to the 3.10 floor by regenerating it, not by hand.

## 3. Every number this session produced

**CPU, Linux x86-64.** `tests/fast` byte-identical (md5 match, 0 of 37,635,840
channel samples). Full suite: Quadrants 4 failed / 2,960 passed, Taichi 2,964
passed / 0 failed — the four are `tests/full_renders`, see §4.

**Warm render cost** (one `Square` frame, 22 kernels, `_taichi_warmstart_check.py`):

| | off | on | |
| --- | --- | --- | --- |
| Quadrants | 29.0 s | **12.5 s** | 2.32× |
| Taichi | 15.1 s | **7.2 s** | 2.11× |

`--fast` on Quadrants 106 s → 78 s. The residual gap to Taichi is Quadrants
building every kernel AST twice — every frontend counter is exactly 2.00× — which
the memo does not touch and which remains the one known frontend item left.

**Builds**, `macos-latest` (macOS 26, Apple clang 21) and `ubuntu-latest`:

| | result |
| --- | --- |
| stock Quadrants v1.3.0, macOS | PASS, 748 s, 23.3 MB wheel, `smoke_metal=ok`, **zero** `-W` diagnostics |
| Quadrants + patches, macOS | PASS, 781 s, `qd.init(metal)=ok` |
| Quadrants + patches, Linux, CUDA on | PASS, 1041 s, `runtime_cuda.bc` present |
| Taichi 1.7.4 + patches, macOS | **FAIL** — 0003 clears `-Wdeprecated-literal-operator`, then `-Wnontrivial-memcall` at `linalg.h:245` stops it at object 15 of 589 |

**Metal, `shapes_and_timeline`, MPS against the same machine's CPU:**

| | px over tolerance (of 83,913,984) | max delta | worst frame | mean brightness MPS vs CPU |
| --- | --- | --- | --- | --- |
| Taichi (reference) | 11,527 | 221 | 174 | 47.62 vs 47.62 |
| Quadrants, before the fix | 79,914,286 | 255 | 4 | 16.41 vs 47.62 |
| Quadrants, after | **11,526** | 221 | 174 | equal |

One pixel from the reference in 83.9 million, same maximum, same frame. Zero-copy
engagement on that run: `converted=29 launches (155 args), passthrough=0, 0
staged, 0 host` — every argument took the path.

**The upstream repros:** #8744 and #8794 reproduce identically on both compilers
(#8744 is bounded by `advanced_optimization=False`, already Algan's shipped
config; #8794 is a 512-entry array Algan cannot reach without `ti.field`).
#8745 reproduces on Taichi and is **clean on Quadrants**.

## 4. Corrections

Measurement falsified nine claims this session, four of them mine. They are
listed because each one was believed on reasonable grounds.

**In `PLAN.md`** (all now fixed in place): `gpu_max_reg` raises no `KeyError` on
the 1.3.0 wheel; `.prog` → `._prog` needs no change in `algan/`;
`QD_WITH_VULKAN=OFF` does *not* drop MoltenVK (`build.py` fetches the LunarG SDK
regardless, so their shipped wheel carries it and the 26.7 MB-vs-50.4 MB size
comparison is not like-for-like); no Quadrants *release* contains the
`get_pos_info` memo; #8794 is not a Metal bug, so the pending-launch valve
cannot fix it; `7a9b6cb23` (#384) is a **must-copy** under Track A, not a
"copy if"; the pre-Volta reduction gate is `cap >= 75`, not 70, because the
constraint is the PTX ISA version and sm_70/sm_72 are broken too; and the
source-level `__scoped_atomic_*` fix that section proposed **does not work** —
the runtime bitcode is compiled with no `-target`, so the scope is dropped and
the re-scoping has to happen in IR.

**In §6.1, mine.** I wrote that "LLVM 15 → 22 costs no re-baseline on x86-64"
off `tests/fast` and one CUDA render. Running the *full* suite on both backends
on one box says otherwise: Taichi passes all six `tests/full_renders` scenes
here, Quadrants moves four of them — by 3, 12, **100** and **158** channel
values. The two that survive are exactly the two the suite names as portable;
the four that move carry PN surfaces, shadows, refraction or glTF, which is
that comment's own `fast_math`-flips-tessellation mechanism. **Track B carries a
real `tests/full_renders` re-baseline on CPU**, inspected scene by scene, plus
the release-asset repackaging `tests/README.md` requires. The baselines are
deliberately untouched: which pixels are acceptable is the maintainer's call,
and a rebaseline that updates the tree without the tarballs is worse than a
failing test.

**Two hypotheses of mine that the instruments killed.** The first Metal reading
was nearly written up as the black frame patch 0001 exists to prevent; the arm
was dimmer, not blank, and only a per-arm brightness number said so. The
follow-up theory — one colour channel of three surviving — died on per-channel
means: `(16.7, 15.8, 16.7)` against `(46.1, 48.1, 48.6)` is uniform dimming, not
channel loss. Both were killed by diagnostics added *because* the first number
could not distinguish them.

**And one about method.** The Metal probe compared MPS against CPU on Quadrants
and had nothing to compare that to; the same reading had never been taken on
Taichi, on either compiler, because there are no committed macOS baselines and
`tests/full_renders` skips every comparison on a Mac. One unattributed number
cannot say whether a port is wrong. The control is what turned it into a
finding.

## 5. The two Metal defects

Both are in `quadrants_patches/0001`, both were found after it compiled cleanly
and reported `zero_copy_available() == True`, and both are the same class of
mistake: **the port was reviewed against the patch it came from rather than the
backend it landed on.** `../quadrants_patches/README.md` has the full account.

1. **The offset went where this backend never reads it.** Quadrants' Metal device
   advertises `spirv_has_physical_storage_buffer`; Taichi's does not. Under that
   capability an ndarray is addressed by a raw 64-bit GPU address out of the args
   buffer, not through its `ExtArr` descriptor — so the descriptor bind site the
   patch faithfully ported is *dead code on Apple silicon*, and every imported
   slice was bound at the base of its arena. Fixed by offsetting the published
   address in `host_to_device`. **Verified on hardware** (§3).
   `PLAN.md` row 28 already recorded that Quadrants took the physical-storage-buffer
   route for the 31-buffer limit; nobody connected it to the patch.
2. **The launch-context cache drops the offsets.** Quadrants caches launch
   contexts keyed on argument identity, and a hit skips `set_args_ndarray` — but
   `LaunchContextBuilder::copy` replays five members and not the two 0001 itself
   added. So the first launch of a kernel binds correctly and **every launch after
   it binds at the base of the arena**. Reachable only where Algan is: an
   `Ndarray` argument is cacheable where a torch tensor is not. Fixed by replaying
   both maps. **Not yet verified on hardware** — see §6.

## 6. Verified, and not

| claim | status |
| --- | --- |
| Quadrants renders CPU pixels identical to Taichi | **verified**, `tests/fast`, byte-identical |
| Quadrants renders CUDA pixels identical | **verified** by the maintainer, Windows, before this session |
| Both patch sets compile | **verified**, two runner images, one leg each |
| The Apple GPU renders correctly on the patched Quadrants wheel | **verified on one scene** (`shapes_and_timeline`), one virtualized M1 — and since on `fast` and `materials_and_lighting`, §10 |
| …on a scene using bloom, glow, surfaces or glossy prefilter | was **NOT verified** at the time (the scene picked for it crashed on Metal on *both* compilers, see below); **verified since**, §10 |
| Defect 2's fix works | was **NOT verified on hardware**; **verified since**, §10 |
| pre-Volta CUDA works on sm_61 | **NOT verified** — compile-only; needs the maintainer's GTX 1050 |
| `tests/full_renders` on CPU | **known to differ**; re-baselined in the second session (§9), re-checked in the third (§10) |

**The crash, attributed: it is not the port, and it is not new.** Rendering
`materials_and_lighting` on Metal dies at frame 119 of 179 with
`Trace/BPT trap: 5`, printing nothing at all — no assert text, no traceback, no
Metal error. **Taichi 1.7.4 with its own patched wheel fails identically**: same
scene, same hardware, the same `Trace/BPT trap: 5`, at the same frame 119 of
179. The CPU arm completes on both.

So this is a **pre-existing defect in Algan's Apple path**, independent of the
migration, and it had never been seen because nothing renders that scene on
Metal: `tests/full_renders` skips every comparison on a Mac, and no probe before
this session rendered anything heavier than a moving square there. Both arms log
`Prepared batch does not fit the render arena; binary-searching the largest
fitting runtime` twice before dying, so memory pressure on the 7 GB virtualized
runner is the first place to look; it stops at the same frame on both compilers,
which says deterministic rather than a race. No `.ips` report was written, which
points at the process being killed rather than faulting.

It is filed in `../algan/rendering/DESIGN_mps_support.md` beside the other Metal
failure modes, because that is where a Metal bug belongs and not here.

## 7. What to do next, in order

*As written at the end of the first session. Items 1–3 are done (§10 has the
runs); the live list is §10's.*

1. **Fix the pre-existing Metal crash** on `materials_and_lighting` — attributed
   now: Taichi fails identically at the same frame, so it is Algan's Apple path,
   not the port. It is the reason "the Apple path works" is still a one-scene
   claim, and it means **no Mac user can render that scene today** on either
   compiler. `DESIGN_mps_support.md` has what is known.
2. **Verify defect 2's fix on hardware**, on a scene that reaches the
   always-cacheable kernels *and does not trip item 1* — `solids_and_camera` is
   the next candidate, being lighter than `materials_and_lighting` while still
   using surfaces and materials. The kernels to reach are — `apply_glow_and_opacity`, `gloss_pyramid_level`,
   `bloom_conv1d_f32`, `bloom_upsample_bilinear_f32`,
   `grid_normals_sides_crosses`.
3. **Re-baseline `tests/full_renders`** on CPU, scene by scene, then
   `scripts/package_baselines.py --tag …` and upload. Maintainer's call.
4. **Run 0003 on the GTX 1050.** Look for `atom.gpu.cas.b64` and no remaining
   `atom.sys` (`quadrants_patches/PORTING-NOTES.md` §7).
5. **The frontend's remaining 1.7×**: Quadrants builds every kernel AST twice.
6. **Upstream what deserves it**: 0003's two CUDA edits (defensible on their own
   merits), 0002's `ContinueStmt` fix to taichi-dev, and both Metal defects to
   Quadrants — defect 1 is arguably *their* bug, since `Ndarray::buffer_offset`
   is theirs and the physical-storage path ignores it.

## 8. Re-running any of it

All through `.github/workflows/run_on_mac.yaml`
(`agent_guidance/gpu_harnesses.md` is the operating manual):

```
# Quadrants from source on macOS, patches optional
command: bash scripts/gate/quadrants_macos_build.sh
arms: mac-cpu    env: GATE_QD_PATCHES=1

# the CUDA patch, compile-only (Quadrants forces CUDA off on Apple)
command: bash scripts/gate/quadrants_linux_build.sh
arms: linux-cpu

# build a patched wheel, then render on the real Apple GPU
command: bash scripts/gate/mps_probe_quadrants.sh
arms: mac-cpu    env: ALGAN_TAICHI_BACKEND=quadrants
                      GATE_SCENE=<scene>
taichi_wheel_run_id: none    latex: true

# the control: Metal against CPU on whatever compiler is installed
command: bash scripts/gate/mps_vs_cpu_ab.sh
arms: mac-mps    env: GATE_SCENE=<scene>
```

Locally: `scripts/gate/backend_pixel_ab.py --both --workdir DIR` compares the
two compilers on one box, and `benchmarks/_taichi_warmstart_check.py` audits the
memoization (three arms, one of them recomputing every memoized value the
original way).

Two arm choices are deliberate and look wrong: the probe runs on **`mac-cpu`**
and selects MPS itself, because on `mac-mps` the harness's own `algan check`
runs before the command and refuses MPS for a wheel the script has not built
yet; the control runs on **`mac-mps`** because it installs nothing and needs MPS
selectable up front.

---

## 9. Second session (2026-09-04, later): the plan's remaining steps, as far as they got

Executed on `claude/rebase-quadrants-plan-on6ojl`, in parallel worktrees, and
**cut short when the maintainer ran out of usage**: five of the seven work
packages were stopped before their own verification finished. Everything is
committed so nothing is lost; the table says what each package is, and what it
is not. Read the "not verified" column before trusting a row.

| package | what landed | verified | not verified |
| --- | --- | --- | --- |
| Track C step 1 (§7.1) | stale "cache ignores `@ti.func` edits" claims corrected (probe-verified: a func-body edit misses the cache on both compilers); nested-tuple claim corrected; `ALGAN_GPU_MAX_REG` deleted; `ALGAN_TI_FULL_TRACEBACK`; `TI_SKIP_VERSION_CHECK` on the taichi arm; concrete arch + `enable_fallback=False`; stale `ticache.lock`/`qdcache.lock`/`ptxcache.lock` sweep before `init`; eviction comment; 64-argument note; `_inside_class` made cheap; `CUDA_CACHE_PATH` under Algan's cache | the agent's own `--fast` and unit runs before it was stopped; this branch's `--fast` | `benchmarks/_cfg_optimization_ab.py` was written but never run — no cfg_optimization numbers |
| Track B step 3, fast launch (§7.3) | `taichi_fast_launch.py` now carries a Quadrants dispatcher too; `benchmarks/_quadrants_launch_overhead.py` measured a warm 20-ndarray launch at ~300 µs on either compiler, ~260 µs of it Python above `prog.launch_kernel`, ~90 µs on a plan hit; `tests/unit_tests/test_taichi_fast_launch.py`; `algan check` reports its gate | fast suite pixel-identical on Quadrants with it on (agent's run) | the taichi-arm parity run and the full unit suite were still running when stopped |
| item 1, source-keyed cache index | `algan/utils/taichi_source_key.py` (1,180 lines: Algan key over kernel source, transitively visited funcs, closure/global walk, config + caps, template values; reuses Quadrants' `load_fast_cache` / `src_hasher` store; verify mode; STATS), `tests/unit_tests/test_taichi_source_key.py`, `benchmarks/_taichi_source_key_check.py`; **default off**, `ALGAN_TAICHI_SOURCE_KEY=1` opts in | imports; `algan check` reports it off | **nothing else** — its tests were never run, no hit rate, no timing, no verify pass. Treat as a draft |
| item 20, early return in inlined funcs | `algan/utils/taichi_early_return.py` (840 lines), installed at import, `ALGAN_TAICHI_EARLY_RETURN=0` turns it off | imports; this branch's `--fast` (no Algan func has an early return, so the rewrite is inert there) | its own tests were never written or run; no early-return func has been compiled through it |
| step 5, wheel CI | `quadrants_build.yaml` builds macOS arm64 (Metal) / manylinux x86_64 (CUDA) / Windows x64 (CUDA) and publishes a release when `release_tag` is set; helper scripts under `.github/workflows/scripts/`; `quadrants_patches/README.md` "Getting a wheel"; the Mac harnesses accept a Quadrants wheel | YAML + actionlint; the source script against the pristine tree | never dispatched; Windows is transcribed from Quadrants' `scripts_new/win` and untested |
| patches 0005–0007 (rows 14, 15, 18) | `0005-cuda-max-reg`, `0006-cuda-readonly-ndarray-ldg`, `0007-cuda-fast-expf`, `verify_cuda_patches.py` | strict `git apply --check` in order after 0001–0004 on pristine v1.3.0 | **not compiled**, not clang-formatted, no README/PORTING-NOTES sections; build with `quadrants_build.yaml` before believing any of them |
| step 6, CPU re-baseline | the four `tests/full_renders` scenes LLVM 22 moves (3 / 13 / 100 / 158) re-baselined after frame-by-frame inspection; pointer digest refreshed, tag still null | `pytest tests/full_renders` on this box | the release-asset upload (`tests/README.md`) |
| Metal on the fast scene | `backend_pixel_ab.py --scenes fast`; `run_on_mac.yaml` dispatched with `GATE_SCENE=fast` on the patched Quadrants wheel | — | the run had not reported when the session ended; read it in the Actions tab |

Attribution of the `materials_and_lighting` Metal crash (§6, §7 step 1): the
control run on **Taichi 1.7.4** dies at the same frame 119 of 179, so it is the
Apple path on the heaviest scene, not the Quadrants port. The maintainer is
investigating it separately.

Not started: row 21 (stage contract v2 / `algan.shading` helpers / user buffers
/ seed) — a user-facing API design with no spec in this plan; and upstreaming.

Order to finish, cheapest first: run `tests/unit_tests/test_taichi_source_key.py`
and `benchmarks/_taichi_source_key_check.py`; write and run the early-return
tests; dispatch `quadrants_build.yaml` on this branch (compile-checks 0005–0007
and exercises the release legs); then the full suite.

---

## 10. Third session (2026-09-04, later still): consolidation, and what the verification found

The second session's branch was merged onto `master` (`d9e89a9`), which had
moved in the meantime: the maintainer had fixed the `materials_and_lighting`
Metal crash (a leak in Algan's MPS import cache, §1.4 of
`../algan/rendering/DESIGN_mps_support.md`), built the three-platform wheel
workflow independently and run it green on all three legs, and re-baselined
the CUDA renders. This section is the ledger after the merge: what was
decided, what each unverified row of §9 turned into when it was actually run,
and what is still open.

### 10.1 The merge

Two conflicts, both where the branch and `master` had built the same thing
twice, and both resolved for `master`:

* **`quadrants_build.yaml`.** `master`'s version had been dispatched and passed
  on all three platforms (run `33850787142`, cp311) and comes with a driver
  (`scripts/build_quadrants_wheels.py`), a matrix resolver and tests. The
  branch's rewrite (`1ac25a6`: a manylinux container build and a release job
  with a `.postN` version pin) was never dispatched and is dropped, with its
  two helper scripts. One step was added to the Linux leg: the 0005–0007 gates
  from Python, and `verify_cuda_patches.py` (whose PTX arms skip without a
  CUDA device).
* **`quadrants_patches/README.md` "Getting a patched wheel".** `master`'s, plus
  a paragraph on the Mac harnesses' `quadrants_wheel` input, whose artifact
  name now follows `master`'s workflow.

Everything else merged clean. `--fast` on the merge: 531 passed, pixel-identical.

### 10.2 What the runs settled

| §9 row | run | result |
| --- | --- | --- |
| Metal on the fast scene | `run_on_mac.yaml` `33847294165` (dispatched in session 2, read here) | **PASS in substance**: MPS vs CPU on the patched Quadrants wheel, mean brightness 39.23 on both arms, per-channel means identical, 1,059 of 12,545,280 channel samples over tolerance (0.008 %), max delta 24 — the float32-accumulator class, the same as Taichi's own MPS-vs-CPU reading. Cold smoke 67 s, warm 3.05 s. |
| The dense scene on Metal (§7 items 1–2) | `33926483875` | **materials_and_lighting renders on the Apple GPU on the patched Quadrants wheel**: 179 frames, 420 s on MPS vs 209 s CPU, means 55.33 vs 55.34, per-channel `(53.1, 57.5, 55.4)` on both, 76,866 of 49,902,336 over tolerance (0.15 %), max 131 at frame 42. This scene reaches every unconditionally cacheable kernel, so it is also **defect 2's fix, verified on hardware**: a cache hit binding at the arena base would look like defect 1 (95 % of pixels, a third of the brightness), not an edge residual with identical means. |
| …and its control | `33927559059`, Taichi 1.7.4 patched wheel, same scene, same harness | **Taichi's own Metal-vs-CPU is the same reading**: 76,983 over tolerance, max 131, worst frame 42, means 55.33 vs 55.34, per-channel identical. The difference of differences is 117 pixels in 49.9 million. So the residual is Algan's Apple path (float32 accumulators, glossy prefilter, edges), not the port; the port's Metal picture on the heaviest scene is as good as the reference compiler's. |
| Patches 0005–0007 | `quadrants_build.yaml` `33926192036` | **apply strictly in order and compile with CUDA on**; the wheel installs and the 0004 checks still pass on it. A hunk-by-hunk review (recorded per patch in `../quadrants_patches/README.md`) found one soundness hole (a store rooted in `ExternalTensorBasePtrStmt` was invisible to 0006's "written" analysis) and one policy error (0006 defaulted **on**, so it would have engaged on every CUDA render with no opt-in); both fixed in `165914d`, rebuilt in `33927637278` <<PENDING-REBUILD>>. Not verified: anything on a CUDA device — `.maxnreg`, `ld.global.nc` and `__nv_fast_expf` in PTX/IR, and pixels. 0007 is the one that is *live* on a CUDA render, since Algan runs `fast_math=True`; expect a last-bit change wherever f32 `exp` is called. |
| Item 20, early return | 41 tests, both compilers | <<PENDING-EARLY-RETURN>> |
| Item 1, source-keyed cache index | | <<PENDING-SOURCE-KEY>> |
| Fast launcher, taichi arm; `cfg_optimization` A/B; full unit suite | | <<PENDING-PARITY>> |
| `tests/full_renders` on CPU | this box, after the merge | <<PENDING-FULL-RENDERS>> |

### 10.3 Two harness defects found on the way

* **`uv run` reverts a patched Quadrants wheel.** The `quadrants_wheel` input
  installed the wheel and lost it two lines later: the sync `uv run` performs
  first put stock `quadrants==1.3.0` back, because the patched wheel's
  `1.3.1.dev0+g…` does not satisfy the lockfile. The Taichi wheel survives the
  same `uv run` only because its version matches the lock exactly, which is
  why the pattern looked safe to copy. Both Mac workflows now check with
  `.venv/bin/python` and export `UV_NO_SYNC=1` for the rest of the job
  (`8415711`; `agent_guidance/gpu_harnesses.md` has the rule).
* **The compiler cannot read a kernel out of `python -c`.** The first gate step
  defined its probe kernel inline and died in `inspect`; it now writes a file.

### 10.4 Still not verified, and the order to do it

1. **A CUDA device for 0005–0007** — `verify_cuda_patches.py on/off/--compare`
   on the T4 harness (`agent_guidance/gpu_harnesses.md`) against the Linux
   wheel from `33927637278`, then `tests/full_renders` on CUDA with the wheel
   installed, inspecting 0007's last-bit change before re-baselining.
2. **0003 on sm_61**, the maintainer's GTX 1050, unchanged from §7.
3. **The release-asset upload of the CPU baselines** (`tests/README.md`),
   still the maintainer's step.
4. Row 21 (stage contract v2) still has no spec; upstreaming is unstarted.
