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
the Apple path is fixed and verified on one scene, and one heavier scene
currently crashes on Metal on a compiler that has never rendered it.**

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
| The Apple GPU renders correctly on the patched Quadrants wheel | **verified on one scene** (`shapes_and_timeline`), one virtualized M1 |
| …on a scene using bloom, glow, surfaces or glossy prefilter | **NOT verified** — `materials_and_lighting` currently crashes on Metal, see below |
| Defect 2's fix works | **NOT verified on hardware** — the run that would have shown it is the one that crashed |
| pre-Volta CUDA works on sm_61 | **NOT verified** — compile-only; needs the maintainer's GTX 1050 |
| `tests/full_renders` on CPU | **known to differ**, re-baseline outstanding (§4) |

**The open crash.** Rendering `materials_and_lighting` on Metal dies at frame
119 of 179 with `Trace/BPT trap: 5` and prints nothing at all — no assert text,
no traceback, no Metal error. The CPU arm of the same scene completes in 198 s.
It is not obviously the new asserts in `copy()`: those check the destination
context is fresh, so they would fire on the first cache hit in the opening
frames, not after thousands of launches. Both Metal scripts now capture the
macOS `.ips` crash report, and the attributing experiment — the same scene on
**Taichi's** patched wheel, which has never rendered it either — was in flight
when this was written. If Taichi also dies there, this is a pre-existing limit
of Algan's Apple path on the heaviest scene (7 GB, and the log shows the arena
binary-searching for a fitting batch twice before the trap), not the port.

## 7. What to do next, in order

1. **Attribute the `materials_and_lighting` crash** (run queued; the Taichi arm
   is the control). Until it is attributed, "the Apple path works" is a
   one-scene claim.
2. **Verify defect 2's fix on hardware**, on a scene that reaches the
   always-cacheable kernels — `apply_glow_and_opacity`, `gloss_pyramid_level`,
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
