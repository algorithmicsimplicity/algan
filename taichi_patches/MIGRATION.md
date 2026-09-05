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
Quadrants patches apply, compile, and — since 2026-09-05 — 0003 and 0005–0007
are verified in PTX on real sm_61 hardware; and the source-keyed cache index
is on by default, its verify arm clean on both arches.** §11 is the newest
ledger, §10 the one before it; §6 and §9 are kept as the record of how it got
there.

What is still unverified needs hardware nobody here has: sm_70/sm_72 for
0003's second defect, and Metal for 0002. `tests/full_renders` on CPU cannot
be read on the Windows box either — those baselines are Linux's.

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
| Patches 0005–0007 | `quadrants_build.yaml` `33926192036` | **apply strictly in order and compile with CUDA on**; the wheel installs and the 0004 checks still pass on it. A hunk-by-hunk review (recorded per patch in `../quadrants_patches/README.md`) found one soundness hole (a store rooted in `ExternalTensorBasePtrStmt` was invisible to 0006's "written" analysis) and one policy error (0006 defaulted **on**, so it would have engaged on every CUDA render with no opt-in); both fixed in `165914d` and **rebuilt green in `33927637278`** (apply, compile, install, the 0004 checks, and the new gate step: `qd.init(gpu_max_reg=48, readonly_ndarray_ldg=True)` reads back and a `loop_config(max_reg=32)` kernel runs on CPU). Not verified: anything on a CUDA device — `.maxnreg`, `ld.global.nc` and `__nv_fast_expf` in PTX/IR, and pixels. 0007 is the one that is *live* on a CUDA render, since Algan runs `fast_math=True`; expect a last-bit change wherever f32 `exp` is called. |
| Item 20, early return | `tests/unit_tests/test_taichi_early_return.py`, 45 tests, both compilers | **Tested and kept installed by default.** Each early-return func is compiled beside a hand-written single-exit twin and both must agree with Python: 45/45 on quadrants 1.3.0 and on taichi 1.7.4; `--fast` green on both with the hook on; a real render leaves the rewrite counter at 0 (104 funcs seen; 255 across `algan/` statically), so no shipped kernel's IR or cache key moves. The tests found three module bugs (an i32 initialiser truncating every float answer, `-1` taken for a typed expression, a decline leaving the body half-rewritten) and one design error: a `return` inside a func's outermost runtime `for` was body-guarded, which *compiles*, and at a kernel's top level that loop is the parallel one — a 65,536-element search with matches at 1000 and 50000 returned **both** across 20 launches on Quadrants. That case is now **refused** with a message naming the `while` spelling (which stays serial and is tested at a kernel's top level); the pass cannot see call sites, so the refusal is lexical and deliberately costs the legitimate inlined-inside-a-loop case too. One test is marked `fast`. |
| Item 1, source-keyed cache index | `test_taichi_source_key.py` (28 on Quadrants; 12 + 16 skipped on Taichi, where the feature stands down by spec), `benchmarks/_taichi_source_key_check.py` twice per backend | **Works, and had two holes.** Over the 22 kernels a `Square` frame materializes: 22 hits, 0 poisoned, the verify arm re-deriving all 22 with **zero mismatches**, one frame digest across every arm and both backends; frontend seconds **13.0 → 0.48** (~27×), whole process 20.7 s → 7.5 s; `--fast` green with it on. Adversarial probing found two ways to a hit on stale IR, both reproduced as a wrong picture: a class read in kernel scope was keyed by its body source, not its attributes (the `ti.static(ArenaView(...))` shape Algan uses), and a `@ti.func` in a class body was skipped by the class walk. Both fixed by hashing every class member by value; schema bumped; config exclusions aligned to the compiler's own. Looked for and not found: globals/closures/attribute chains, aliases and `getattr` (they poison), mutable globals, same-repr template values, nested func tuples, config fields (all 98 audited), non-`ALGAN_` env, dataclasses (poison), kernel defaults (the compiler rejects them). **Default stays off**: the module's own bar is a clean verify arm over `tests/full_renders` on CPU and CUDA, and both holes were invisible to a one-frame render. |
| Fast launcher, taichi arm | `benchmarks/_taichi_fast_launch_check.py`, twice on Taichi, once on Quadrants | **PASS on both**: identical frames across the off / on / verify arms (one digest, `717c4690…`, on both compilers too), the on arm taking the fast path for 96 of 119 launches. No render-time speedup is measurable on CPU (0.85–1.04×, cross-process, a shared box); the launcher's ~200 µs per launch is invisible behind a CPU frame, which is the expected shape — it exists for the GPU arms. |
| `cfg_optimization` A/B | `benchmarks/_cfg_optimization_ab.py`, both compilers | **The docstring's 2.1× compile saving did not reproduce**: backend compile 20.9 s on / 21.2 s off on Quadrants, 18.4 / 19.2 on Taichi, frames byte-identical, warm render inside noise. Both arms of this run beat the earlier run's *off* arm, so its 63.4 s "on" arm was contention. On Algan's kernels the pass costs nothing measurable and buys nothing measurable; the default stays. Recorded in the script's docstring beside the first run. |
| Full unit suite | `tests/unit_tests`, Quadrants | **3168 passed, 139 skipped, 2 failed**, both fixed here: `test_baseline_store` was red because `master`'s CUDA re-baseline (`5d558f1`) never refreshed `tests/baselines.json` (pointer refreshed, tag still null, upload still the maintainer's); `test_taichi_launch_pairing` asserted the "unknown device matches any GPU arch" rule that Track C step 1 replaced with "an unknown device is served by the CPU arch" — the test now states the new rule. Taichi arm: see the note below. |
| `tests/full_renders` on CPU | this box, after the merge | **4 passed, 3 differ by 13 / 5 / 8** (`materials_and_lighting` frame 63, `solids_and_camera` frame 130, `text_and_media` frame 129) against the second session's re-baseline. **Not the merge's doing:** the second session's own commit (`5846c87`), rendered in this same container, fails `solids_and_camera` by the identical 5 at the identical frame 130. What differs is the container: `torch.compile` fails here (`InductorError: AssertionError: …/distutils/core.py`, four times per run) and the fused triangle projection runs eagerly, so its rounding is not the rounding the baselines were made with. The baselines stand; `pytest tests/full_renders` must be read on a box where `torch.compile` works, and the release-asset upload is still outstanding. |

The Taichi arm of the unit suite (`ALGAN_TAICHI_BACKEND=taichi`): **3201
passed, 155 skipped, 1 failed** — the same launch-pairing test, collected
before its fix landed; it passes on Taichi in the re-run after the fix (23
passed in that file), and the baseline-pointer test is backend-independent.
This box's `torch.compile` failure (§10.2, full renders) is visible in this
run too as `AlganWarning: torch.compile failed … runs eagerly`; no test
depends on it.

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

*Items 1 and 2 were run on 2026-09-05 and are settled; §11 is the record.*

1. **A CUDA device for 0005–0007** — `verify_cuda_patches.py on/off/--compare`
   on the T4 harness (`agent_guidance/gpu_harnesses.md`) against the Linux
   wheel from `33927637278`, then `tests/full_renders` on CUDA with the wheel
   installed, inspecting 0007's last-bit change before re-baselining.
   **Done on sm_61 instead of the T4** (§11.2), and the 0007 re-baseline turns
   out to have happened already — see the end of §11.2.
2. **0003 on sm_61**, the maintainer's GTX 1050, unchanged from §7. **Done**
   (§11.1).
3. **The release-asset upload of the CPU baselines** (`tests/README.md`),
   still the maintainer's step.
4. Row 21 (stage contract v2) still has no spec; upstreaming is unstarted.

---

## 11. Fourth session (2026-09-05): the GTX 1050 run

The maintainer's own box, Windows 10, GTX 1050 (sm_61, 4 GB, driver 576.52),
against the locally-built patched wheel
`quadrants-1.3.1.dev0+gab9a58ab5.d20260905-cp311-cp311-win_amd64`. This is the
hardware §10.4 items 1 and 2 were waiting for.

**Read §11.0 first.** The session opened by reproducing a "the patches are not
working" report that was an artefact of the harness, and the same trap is one
`uv run` away from anyone.

### 11.0 `uv run` had silently reverted the wheel

The reported symptom was that `debug/debug.py` still took ~100 s on a fresh
process with the source-keyed index supposedly shipped. Two things were true at
once and neither was the port:

* **The index is opt-in** and the script sets no environment, so it was never
  installed — `skipped_reason()` says exactly that, and `algan check` prints it.
* **`uv run python` had uninstalled the patched wheel**, twice, before anything
  was measured. This is §10.3's defect, unchanged, on the maintainer's box
  rather than in a workflow: `uv run` syncs the lockfile first, and
  `1.3.1.dev0+g…` does not satisfy it, so stock `quadrants==1.3.0` goes back.
  On sm_61 the tell is loud — `qd.init` dies in `cuModuleLoadDataEx` with
  `CUDA_ERROR_NOT_SUPPORTED`, which is defect (a) of §7 with 0003 gone — but on
  any post-Volta box it would be silent, and would read as a performance
  regression in whatever was being tested.

`CLAUDE.md` now says to run `.venv/Scripts/python.exe` directly and carries the
`--reinstall-package quadrants` recipe. **That accidental uninstall is also the
negative control** the sm_61 verification below would otherwise have lacked: the
same script, same box, same hour, pristine v1.3.0 fails at `Program()` and the
patched wheel does not.

A **per-patch marker check** was run on the restored wheel, because "the wheel is
patched" had been assumed rather than checked. Four of the seven leave a
Python-visible symbol and all four are present: 0001
(`quadrants.lang._ndarray.ExternalMetalNdarray`,
`Program.create_ndarray_from_metal_buffer`), 0004
(`CompileConfig.invariant_arg_loads`), 0005 (`ASTBuilder.max_reg`,
`loop_config(max_reg=)`) and 0006 (`CompileConfig.readonly_ndarray_ldg`). Note
that `CompileConfig.gpu_max_reg` is **not** a marker — pristine v1.3.0 has it,
which is §8's point about 0005 having two halves. 0002, 0003 and 0007 are
behavioural; 0003 and 0007 are settled below, 0002 remains Metal-only.

### 11.1 0003 on sm_61 — verified, exactly as §7 predicted

`qd.init(arch=qd.cuda)` comes up, kernels compile and run, and the answers are
right. Dumping the module PTX (`print_kernel_asm=True`) gives, across 56 atomic
instructions in the runtime module:

* **`atom.sys`: 0.**
* **`atom.gpu`: 1** — `atom.gpu.cas.b64`, inside `runtime_eval_adstack_max_reduce`,
  which is the exact instruction and the exact function §7 names as the one
  that took the whole runtime module down.
* Every other atomic is unscoped (`atom.exch.b32`, `atom.global.add.u64`), i.e.
  untouched by the patch, as intended.

The header reads `.version 5.0 / .target sm_61`, which is
`getMinPTXVersionForSM(61) == 50` — the outside confirmation §7 wanted for
defect (b)'s "the bound is PTX ISA version, not SM" correction. No
`Cannot select: intrinsic %llvm.nvvm.activemask` abort occurred, so change 3's
`kMinComputeCapabilityForWarpReduction = 75` gate is doing its job.

What this does **not** settle, still: sm_70/sm_72 (§7 item 4), and whether the
driver's refusal of `atom.sys` tracks compute capability or something else
(§7 item 5). One sm_61 box remains one sm_61 box.

### 11.2 0005–0007 on a CUDA device — verified, after fixing the verifier

`verify_cuda_patches.py` **failed on its first run**, and the failure was the
script's, not a patch's:

    the off arm emitted ld.global.nc; readonly_ndarray_ldg=False does not gate

The script's docstring claimed that leaving `invariant_arg_loads` (0004) at its
default "isolates 0005-0007". It does not. 0004's default is **True**, and NVPTX
lowers a global load carrying `!invariant.load` to `ld.global.nc` on its own —
so 0004 emits the very instruction the script attributes to 0006. A 2×2 sweep,
one process per cell, on the script's own probe kernel:

| `invariant_arg_loads` | `readonly_ndarray_ldg` | `ld.global.nc` | `ld.global` (plain) |
| --- | --- | --- | --- |
| False | False | **0** | 6 |
| False | True | 2 | 6 |
| True | False | 4 | 2 |
| True | True | 6 | 2 |

The four in row 3 are 0004 hoisting base pointers and shape dims; the two 0006
adds are the two read-only ndarrays. **0006 gates correctly** — with 0004 out of
the way, off means zero. Both arms now pin `invariant_arg_loads=False`, the
config read-back records it, and the failure message names 0004 when an arm did
not pin it, so the next reader is not sent after the wrong patch. With that:

    maxnreg_directives   off 0   on 1  (['64'])
    ld_global_nc         off 0   on 2
    ld_global_plain      off 6   on 6
    fast_expf_calls      off 0   on 2
    expf_calls           off 2   on 0
    PASS

So on sm_61: 0005's per-kernel `.maxnreg` lands, 0006's `ld.global.nc` engages
only when asked and leaves the read-and-written array on a plain load, and 0007
picks `__nv_fast_expf` only under `fast_math`.

**Two of the seven are inert on every Algan render**, which the patches' own
notes imply but nothing said outright: nothing in `algan/` sets
`readonly_ndarray_ldg` (0006, and the compiler defaults it off) or
`gpu_max_reg` / `loop_config(max_reg=)` (0005). 0004 defaults on and Algan sets
`fast_math=True`, so those two, and only those two, are live. The stale
paragraph in `algan/rendering/taichi_runtime.py` claiming `gpu_max_reg` "never
reached ptxas on either compiler" is corrected: with 0005 it does.

**0007's re-baseline turns out to be already done.** The worry was that the CUDA
baselines were made with `__nv_expf` while this wheel renders `__nv_fast_expf`.
They were not: `expected_outputs_cuda/` was regenerated on 2026-09-04 21:11-21:28
(`5d558f1`), and the first patched wheel (`d20260904`) was built at 20:02 the
same evening — so the committed CUDA baselines already carry 0007. Confirmed
from the other side by `materials_and_lighting` passing pixel-wise on the
`d20260905` wheel, which is the densest scene and the one 0007's f32 `exp` sites
most affect.

### 11.3 The source-keyed cache index is now on by default

§10.2 left it a working feature with an explicit bar: *"the module's own bar is
a clean verify arm over `tests/full_renders` on CPU and CUDA, and both holes
were invisible to a one-frame render"*. That bar is met, and the default is
flipped (`ALGAN_TAICHI_SOURCE_KEY=0` is now the opt-*out*).

**What the report that started this session actually was.** "A simple script
still takes ~100 s on a fresh process." Two causes, neither the index being
broken: the script sets no environment and the index was opt-in, so it was
never installed; and `uv run` had reverted the patched wheel (§11.0). Worth
recording because "the feature shipped" and "the feature runs" were two
different things and nothing in a render said so — which is why `algan check`
now prints an off index as a **WARNING** rather than as INFO.

**The measurements.** `benchmarks/_taichi_source_key_check.py`, twice, warm
(the first run pays cold compilation and is not the number):

| arm | process s | render s | frontend s | hits | miss | verified | frame digest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off | 43.66 | 34.55 | 23.81 | 0 | 0 | 0 | `e0a4f46bf40a7a67` |
| warm | 18.96 | 9.15 | 5.78 | 22 | 0 | 0 | `e0a4f46bf40a7a67` |
| on | 20.85 | 10.01 | 6.13 | 22 | 0 | 0 | `e0a4f46bf40a7a67` |
| verify | 44.01 | 34.47 | 23.65 | 0 | 0 | 22 | `e0a4f46bf40a7a67` |

One digest across all four arms. Frontend 23.8 s → 6.1 s; the whole process
43.7 s → 20.9 s on a scene that is one `Square`.

**The bar, on `tests/full_renders`:**

| | keyed | verified | poisoned | misses | mismatches |
| --- | --- | --- | --- | --- | --- |
| CUDA | 39 | **39** | 0 | 0 | **0** |
| CPU (`x64`) | 40 | **40** | 0 | 0 | **0** |

Each arch needs **two** passes and the reason is in the hook: a miss returns
`None` and the full compile stores, so the first pass over a new arch is all
first sightings (CUDA 39 misses, CPU 40) and verifies nothing. The second pass
is the one that verifies. A run that reports `verified=0` has not tested
anything, which is easy to mistake for a pass.

**And on CUDA, all six scenes came out byte-identical** between an index-on run
and a full-transform run — not "within the suite's ±2", identical. That is the
strongest form of the claim and it is what makes a third CPU pass unnecessary:
restoring an artifact once the key matches is arch-independent, and what *is*
arch-dependent — whether the key captures the CPU compile config — is exactly
what the verify arm re-derives.

**Two things the CPU arm needs a note about.** Its pixel comparisons all fail
on this box, by 26–94 channel values, because `expected_outputs_cpu/` was
generated on Linux and this is Windows; the signal read there was the
verified/poisoned counts and the absence of a raise. And key computation costs
**6.2 s** on the CPU arm's first pass against 0.95 s on CUDA, dropping to 1.1 s
once the index is warm — the walk is the same, the box is not.

**Cost of being wrong, and the control.** An unsound key serves a stale
*kernel*, so it shows up as a wrong picture rather than as an error.
`ALGAN_TAICHI_SOURCE_KEY=0` is the control arm and is the first thing to try if
a render's output is ever in question.

### 11.4 Two defects found on the way, unrelated to the port

* **`Color.__new__` had no NumPy branch** (`algan/constants/color.py`). Every
  colour from the Manim layer arrives as one — `ManimColor.to_rgba()` is a
  NumPy array and a `VMobject`'s `fill_opacity` is an array per submobject — so
  an `ndarray` fell through every `isinstance` branch and was splatted into the
  five-channel tuple unconverted. It survived only on NumPy's size-1-array-to-
  scalar coercion, deprecated since 1.25 and **raising in 2.4**, where the whole
  Manim import path dies in `Axes(...)`. Fixed by coercing `glow`/`opacity`
  before the branches (the string branch compares `opacity == 1`, which on an
  array is an array) and widening the tensor branch to `np.ndarray` via a
  zero-copy `torch.as_tensor`. A second latent defect fell out: a 4- or 5-wide
  `ndarray` had been producing a **six**- or seven-channel colour. 13 tests in
  `tests/unit_tests/test_color_array_inputs.py`, with `DeprecationWarning`
  promoted to an error so the NumPy version installed decides nothing; the
  three Manim-touching full-render scenes are byte-identical across the fix.
* **`verify_cuda_patches.py` blamed the wrong patch** — §11.2.
