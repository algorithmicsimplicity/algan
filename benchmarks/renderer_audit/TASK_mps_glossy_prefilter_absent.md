# Task: the prefiltered glossy reflection is absent on an Apple GPU

**Status: OPEN.** One of the two tests the required macOS MPS arm of
`.github/workflows/test.yaml` cannot pass. The arm is otherwise green — 3564
passed, 217 skipped, 2 xfailed — so this is one of the last two things between
Algan and a fully green Apple-GPU gate.

**You do not need to read anything else to work on this.** Everything measured
so far is in this file. `../../algan/rendering/DESIGN_mps_support.md` §4.5 is
the same finding written for a reader of the whole port, and §2.3b/§2.3f are
the two defects that turned out to be *next door* to this one and are already
fixed — worth reading only if the leads in §5 run out.

---

## 0. The one line

`tests/unit_tests/test_glossy_prefilter.py::test_prefiltered_reflection_is_substantially_wider`
fails on MPS and only on MPS. In the window where the emitter's mirror image
should land, the rendered frame has **no reflected signal at all**.

Fixing it means: make that render produce the reflection, delete the test's
line from `tests/mps_known_failures.py`, and record what it was in §4.5. §7 is
the full definition of done.

---

## 1. Read this first: you almost certainly have no Apple GPU

The development box has no Metal device. Two consequences shape everything
below.

**The cheap loop is local and it is a control, not a reproduction.** Algan's
MPS-friendly mode is settable on any device, so you can run every substitution
this port makes over a CPU render device:

```bash
ALGAN_MPS_FRIENDLY=1 .venv/bin/python -m pytest -q \
  "tests/unit_tests/test_glossy_prefilter.py::test_prefiltered_reflection_is_substantially_wider"
```

~85 s. It **passes** (§3), which is exactly why it is useful: it is green
today, so if a change you make turns it red you have broken the renderer for
everyone, and you will know in a minute and a half instead of an hour. Run it
before every Apple round.

**The real loop is a GitHub Actions round on `macos-latest`, and it costs
15–60 minutes.** Budget accordingly: get several questions into each round.
Edit `.github/gpu-run/mac.json` and push — that file is a push trigger for
`.github/workflows/run_on_mac.yaml`, which is how a branch whose workflow file
is not on `master` reaches the Apple runner:

```json
{
  "command": "uv run python benchmarks/_mps_render_smoke.py --verify-torch-ops || true",
  "arms": ["mac-mps"],
  "env": {},
  "latex": false,
  "taichi_wheel_run_id": "none",
  "timeout_minutes": 45
}
```

Four things about that file that each cost a wasted round to learn:

* **`taichi_wheel_run_id: "none"` is required.** The default installs a patched
  *Taichi* wheel and flips `ALGAN_TAICHI_BACKEND=taichi`, so you measure a
  different compiler from the one CI and users run. `"none"` gives you the
  plain locked `algan-quadrants`, which is the supported path and does render
  on the Apple GPU.
* **The run step is `bash -e`.** Chain commands with `|| true` or the first
  non-zero exit — including a probe that *reports* by exiting 1 — takes the
  rest of the round with it.
* **`latex: true` costs ~4 minutes** and this test does not need it.
* Read the result with
  `mcp__github__actions_list method=list_workflow_runs resource_id=run_on_mac.yaml`
  then `mcp__github__get_job_logs`. Poll at 5-minute intervals; nothing
  notifies you. `agent_guidance/gpu_harnesses.md` is the operating manual.

---

## 2. The failure, exactly

The test renders the audit tree's calibration scene twice — the glossy route
off and on — and asserts the reflection gets wider:

```python
ghost = _reflection_spread(calib_arms("gl_off_pf1", glossy=False, prefilter=True))
glow  = _reflection_spread(calib_arms("gl_on_pf1",  glossy=True,  prefilter=True))
assert glow[0] > 3.0 * ghost[0], (ghost, glow)      # test_glossy_prefilter.py:452
```

On MPS it reads **`nan > 3.0 * 74.9`**, with

```
RuntimeWarning: invalid value encountered in scalar divide
  cy = (sig * ys).sum() / total          # test_glossy_prefilter.py:395
```

**The `nan` is the finding, not noise.** `_reflection_spread` subtracts the
window's own 10th percentile and divides by what is left:

```python
sig   = np.clip(win - np.percentile(win, 10), 0.0, None)
total = sig.sum()
cy    = (sig * ys).sum() / total
```

A `nan` there means `total == 0` — every pixel in the window sits at or below
the window's own 10th percentile. The window is **flat**.

So this is categorical, not a tolerance. The reflection did not come out
narrow, or dim, or displaced. On the CPU the same measurement gives a glow of
**61.6 px rms** against the throttled arm's **7.7** (8.0x, asserted at 3.0x);
on MPS there is nothing above the wall's own level anywhere in the window.

**Do not chase the ratio. Chase the zero.**

---

## 3. What is measured, and what it rules out

Four arms, one variable:

| arm | result |
| --- | --- |
| macOS **CPU** — the same runner, the same torch build, run 34102515789 | **passes** |
| Linux 3.10 and 3.13, `auto` → CPU | **passes** |
| Linux CPU with `ALGAN_MPS_FRIENDLY=1` — the control | **passes** (99.7 s off the mode, 84.6 s on it) |
| macOS **MPS** | **fails** |

The third row is the one that carries weight, and it is the discriminator
`DESIGN_mps_support.md` §1.2c established for exactly this purpose. Forcing the
mode on over a CPU render device exercises **every** substitution the port
makes — the float32 accumulators, the int32 reductions, the log-step scan that
replaces `cummax`, `clamp_floor`, `gather_exact` — with no Apple GPU in the
picture. It is green.

**Therefore it is not:**

* MPS-friendly mode's numerics. The control arm runs all of them and passes.
* The renderer's own arithmetic. Three CPU arms on two operating systems pass.
* The fragment stream. This test's sibling
  `test_a_creases_siblings_share_the_pixels_prefiltered_claim` had the same
  shape of failure and now **passes**, since the acceptance-mask gather fix
  (§2.3f) made the Apple GPU's fragment stream identical to the CPU's — same
  fragment count, same pixel range, same depth range, verified on the runner.
  Whatever this is, it is downstream of the stream.
* A skipped test misread as a pass. `benchmarks/renderer_audit/scenes/calib_glossy.json`
  is committed, so `_needs_audit_tree` does not skip and the green arms above
  are real renders. Check this again if you move the scene.

**What is left is Metal, or torch's MPS backend.**

---

## 4. What is NOT established

State these honestly if you write anything up.

* **CUDA is untested.** Every green arm above resolves to a *CPU* render
  device. They say "not the CPU path" and nothing more. Nothing in any run read
  so far exercises the glossy prefilter on a CUDA device, so if you find a
  genuine renderer bug here, check whether CUDA has it too before calling it
  Apple-specific.
* **The mechanism is unknown.** §3 is elimination, not diagnosis. No one has
  yet looked at what `gl_main` or `gl_pyr` actually contain on the hardware.
* **Whether it is one defect or two.** The sibling test passing is suggestive
  but not conclusive — it asserts a different property of the same route.

---

## 5. Where to look, in order

The glossy route (`glossy_reflection_mode() == 3`) adds three kernels over the
plain one, in `algan/rendering/raytracing/glossy_prefilter_taichi.py`, driven
from `tracer.py`'s tile loop and `_gloss_finish_frame`:

1. **`gloss_scatter`** — moves a drained tile's glossy pixels into the frame's
   `gl_main` / `gl_pyr` buffers. Runs per tile, per frame-part.
2. **`gloss_pyramid_level`** — builds the mip pyramid bottom-up, one call per
   level, in `_gloss_finish_frame`.
3. **`gloss_composite`** — each glossy pixel fetches the pyramid trilinearly at
   the level matching its own blur radius, and overwrites what the tile
   composite wrote for it.

**The first thing to instrument is `gl_main`'s sigma column**, and here is why
it is the highest-value probe rather than a guess:

```python
# tracer.py, _gloss_clear
gl_main.zero_()
gl_main[:, GL_MAIN_SIGMA] = -1.0      # GL_MAIN_SIGMA = 7, GL_MAIN_WIDTH = 8
```

That column is initialised **negative on purpose** — it is what marks a pixel
as having a prefiltered glossy branch at all, because zero is a legal blur
radius (a reflection in contact with its reflector). So a `gloss_scatter` that
fails to write it leaves every pixel unmarked, `gloss_composite` finds nothing
to fetch, and the window is flat. **That is precisely the symptom.**

Concretely, for one Apple round:

* After the render, count `(gl_main[:, GL_MAIN_SIGMA] >= 0).sum()` and compare
  against the CPU. If it is 0 on MPS and non-zero on CPU, the defect is in
  `gloss_scatter` or in what it is handed, and you are done searching.
* If it is non-zero, dump `gl_pyr`'s level 0 sum, then each level's, and find
  where the energy disappears — that separates `gloss_pyramid_level` from
  `gloss_composite`.
* `_mps_render_smoke.py --verify-torch-ops` already wraps the suspect torch
  ops and prints the **Algan caller** of any that disagrees with the CPU
  (that is how §2.3f's `raster_pipeline.py:1903` was found). Note it renders
  its own scene, not this one — point it at the calib scene, or add the same
  wrapper to a probe that renders this one.

Two priors worth carrying, from the defects already fixed on this port:

* **A torch op that answers *wrongly* rather than failing** is this backend's
  signature. Two of the three MPS defects found so far were that
  (`index_select` past 2**24, `clamp_min`'s float16 bound). If a value looks
  plausible but wrong, check the op, not the algorithm.
* **A Taichi kernel that compiles and does nothing** is the other. Metal
  refuses some constructs by handing Taichi a nil pipeline; §1.2c was a
  `continue` under a `ti.static` gate emitting invalid SPIR-V that LLVM
  executed happily. `gloss_pyramid_level` is a loop over levels with static
  bounds — read it for that shape.

The Vulkan reproduction loop §1.2c recommends (llvmpipe on Linux reproducing
SPIR-V codegen bugs) **no longer works**: the published Quadrants wheel is
built without the Vulkan backend. Check with
`python -c "from quadrants._lib import core; print(core.with_vulkan())"` — it
prints `False`. So a codegen question costs an Apple round unless someone
builds a Vulkan-enabled wheel.

---

## 6. Traps

* **Do not use bare `uv run`** if a locally-built compiler wheel is installed —
  it syncs the lockfile first and silently replaces it. `.venv/bin/python`
  directly, or `UV_NO_SYNC=1`. On the CI runner `uv run` is correct, because
  there the locked wheel *is* what you want.
* **The xfail is strict.** The moment your fix works, the arm goes **red** with
  `XPASS(strict)` naming the test. That is the mechanism working, not a new
  failure — delete the entry (§7).
* **macOS concurrency is 5 jobs across the whole account**, so do not fan out
  rounds. Runner minutes are free; slots are not.
* **This test is expensive.** It renders the calibration scene twice through a
  module-scoped fixture that caches arms by suffix. Do not convert it to a
  function-scoped fixture to debug it.
* `tests/full_renders` and `tests/path_traced` baselines live as release assets
  and a suite that cannot resolve them **skips**. Check for skips before
  believing a green run.

---

## 7. Definition of done

1. The fix, with the mechanism understood — not a threshold moved and not a
   value clamped into range. §2 says the window is flat; a fix that makes the
   assertion pass without explaining the zero has not found the defect.
2. `ALGAN_MPS_FRIENDLY=1` control still green locally, and the full
   `tests/unit_tests tests/fast` green on Linux (`.venv/bin/python -m pytest -q
   tests/unit_tests tests/fast`, ~35 min on a CPU box).
3. **Delete this test's line from `tests/mps_known_failures.py`.** The arm is
   red until you do. `tests/unit_tests/test_mps_known_failures.py` guards the
   list's other direction — an entry whose test was renamed marks nothing.
4. Update `DESIGN_mps_support.md` §4.5 with what it actually was, and §4.1's
   table row for cause D. If the mechanism is a torch-MPS defect, add it to
   §2.3's family and give it a case in `benchmarks/_mps_torch_op_probe.py` —
   that file is the standing record of what this backend gets wrong.
5. One Apple round on the full gate scope to confirm. The arithmetic, so you
   can tell a fix from a coincidence: the arm stands at **3564 passed, 217
   skipped, 2 xfailed, 0 failed**. Your fix should give **3565 passed, 217
   skipped, 1 xfailed, 0 failed** — one test moves from the xfailed column to
   the passed one and nothing else changes. If `failed` is not 0 you left the
   entry in (§7.3); if `passed` moved by more than one, you changed something
   else too and should find out what.

---

## 8. The other open one, so you do not trip over it

`tests/unit_tests/test_taichi_early_return.py::test_a_real_function_is_not_rewritten`
is the second and last entry. It is unrelated: `@ti.real_func` with an early
`return`, launched, dies in Quadrants' SPIR-V builder with `Value "tmp6" does
not yet exist`. A compiler defect a layer below Algan; no renderer kernel uses
`ti.real_func`, so it blocks a test rather than a picture. Do not bundle the
two.
