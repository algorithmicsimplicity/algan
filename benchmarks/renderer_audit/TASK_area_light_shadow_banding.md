# Task: an area light casts K hard shadows instead of one penumbra

**Status: DONE.** Each emitter row now integrates visibility over its own cell
of the rectangle, in the light's own plane, instead of testing the cell's centre
point. `shadow_band_probe.py` finds no `k/K` grid at any emitter count, flatness
falls 0.87 → 0.73 at the shipped `samples = 4` and → 0.54 at 16 against the path
tracer's 0.49, and the profile is monotone into the umbra and back out.
`benchmarks/_area_light_shadow_check.py` is the acceptance harness;
`ALGAN_AREA_LIGHT_SOFT_SHADOWS=0` restores the old behaviour.

Three things this brief got wrong, or could not have known, recorded because the
next reader will hit them:

* **§6's "the umbra stays dark" is the wrong acceptance test**, and a fix that
  passes it is wrong. The umbra *should* lift, 0.009 → 0.039, because a `k × k`
  grid of point emitters spans only `(1 - 1/k)` of the rectangle: at
  `samples = 4` the old renderer shadowed from an emitter **half** the authored
  width and height, and a too-small emitter casts a too-large, too-dark umbra.
  The honest test is that the umbra **converges** (0.039 → 0.035 → 0.032 as
  `samples` rises) rather than keeps climbing, and that an unshadowed render is
  byte-identical between the arms so nothing reached the radiance term.
* **§5(a)'s "raise the sample count" was never a route.** `MAX_SHADOW_LIGHTS` is
  16 and each emitter sample spends one slot, so at `samples = 64` the surplus
  rows are lit but cast no shadow and the shadow washes out entirely (scanline
  minimum 0.73). Raising `samples` only shrinks the steps until the cap erases
  the shadow.
* **§5's "leaves the light half-corrected"** stands, and deliberately so: this
  is the visibility half only. The falloff (`REPORT.md` §6.7) is untouched
  because fixing it redefines `intensity` for every existing `RectAreaLight`.
  The two are mechanically independent — visibility lives in the two shadow
  fans, radiance in `_light_eval` — so neither needs redoing for the other.

What follows is the original brief, unedited.

---

**Status:** open, diagnosed, not fixed.

`REPORT.md` §6.7 is the same mechanism seen in the light's *falloff*, and it
already noted this shadow symptom in passing — "at `samples = 4` the penumbra is
four discrete overlapping copies rather than a gradient, which is visible in the
frame as banded ellipses". What the audit's fourth run added is the
**measurement** and, more importantly, **a reference that can answer it**: the
three.js *rasterizer* casts no rect-area shadow at all, so until the path tracer
was pointed at this scene there was nothing to compare against.

**You do not need to read `REPORT.md` to work on this.** Everything needed is
here. Do read §6.7 before choosing a fix, though — see §5.

---

## 1. The defect in one paragraph

`RectAreaLight` is expanded at render time into a fixed grid of `samples` point
emitters, each carrying `1/samples` of the power. Every one of those emitters
casts its own **hard** shadow. So a sphere in front of an area light does not
get a penumbra — it gets `K` offset hard shadows whose union is a staircase, and
whose overlapping ellipses read as a fan of separate shadows. At the default
`samples = 4` the shadowed brightness can only take five values (0, ¼, ½, ¾, 1),
and a horizontal scanline through the fan crosses **eight** step edges, which is
what makes it look like eight shadows. The reference casts one smooth penumbra.

The class docstring already promises the right behaviour — "Both the lighting
and — with ray-traced shadows enabled — the penumbras are therefore smooth, with
a smoothness set by `samples`" — which is true only asymptotically. At the
shipped default it is visibly a stack of hard shadows.

## 2. The measurement that defines it

Scene: `scenes/calib_lights.json`, the port of
`tests/full_renders/scenes/materials_and_lighting.py` **act 3** — one neutral
probe per light type in front of a wall. The relevant object is `probe_rect`,
lit by the `rect_area` light (width 1.8, height 1.0, no `samples` given, so the
default 4).

A horizontal scanline two sphere-radii below the probe's centre, normalised to
its own maximum, in **linear** light:

| | min | flatness | plateau levels below 0.8 | reading |
| --- | --- | --- | --- | --- |
| **algan** | 0.009 | **0.87** | **[0.01, 0.25, 0.52, 0.74]** | sits on a k/**4** grid — 4 emitters at ¼ power each |
| three_pathtrace | 0.000 | 0.49 | [0.0, 0.33, 0.56, 0.78] | fits no small-integer grid — a continuous penumbra |
| three_raster | 0.583 | 0.98 | — | **no shadow at all** |

The 24-bucket profiles make it plain:

```
algan            [0.91 0.94 0.95 0.97 0.99 0.78 0.58 0.52 0.30 0.06 0.01 0.01
                  0.19 0.43 0.51 0.66 0.86 0.94 0.92 0.90 0.88 0.85 0.83 0.80]
three_pathtrace  [0.73 0.79 0.82 0.87 0.81 0.70 0.56 0.41 0.32 0.09 0.01 0.00
                  0.19 0.38 0.49 0.61 0.75 0.77 0.79 0.74 0.72 0.64 0.60 0.51]
```

Algan steps 0.99 → 0.78 → 0.58 → 0.52 → 0.30 → 0.06; the reference ramps
0.87 → 0.81 → 0.70 → 0.56 → 0.41 → 0.32 → 0.09 monotonically.

**The other two lights in the scene are correct**, which is what isolates the
cause. Rendered one light at a time, `point` and `spot` each cast exactly one
clean shadow in Algan; only `rect_area` produces the fan.

### Which reference to use, and why it matters here

**Use the path tracer.** `three.js`'s rasterizer **cannot cast a shadow from a
`RectAreaLight` at all** (its scanline above never darkens — min 0.583 is just
the light's own falloff). `three-gpu-pathtracer` treats it as a real area light
and integrates it, so it is the only reference that answers this question. Two
further reference limits to know before comparing the *whole* scene rather than
this one light:

* the path tracer **silently drops `HemisphereLight`** — `getLights()` in
  `three-gpu-pathtracer/src/core/utils/sceneUpdateUtils.js` collects only
  rectArea / spot / point / directional — and it drops `AmbientLight` too. Of
  `calib_lights.json`'s four lights it renders three.
* so full-frame comparisons on this scene compare against a reference missing a
  light. **Render one light at a time** (§3.4) and the problem disappears.

## 3. Reproduce it

### 3.1 Environment

Algan renders need the project venv (the only place the path is written down is
`CLAUDE.md`):

| Platform | `<venv-python>` |
| --- | --- |
| Linux / macOS | `.venv/bin/python` |
| Windows | `.venv\Scripts\python.exe` |

The Three.js reference needs a scratch npm project and a browser:

```bash
mkdir -p /tmp/three && cd /tmp/three && npm init -y
npm install three three-gpu-pathtracer playwright
npx playwright install chromium
export AUDIT_THREE_NODE_MODULES=/tmp/three/node_modules
```

Pin `three@0.185.1` and `three-gpu-pathtracer@0.0.24` — the versions every
number above was measured against.

### 3.2 Split the scene into one light each — do this first

**The full four-light scene will not show you the defect.** Its other three
lights fill the rect-area shadow, and the path tracer is missing one of them
besides, so the probe reads a washed-out `[0.05, 0.12, 0.2, 0.27, 0.33, 0.37]`
and fits no grid — an honest reading of a contaminated measurement. Isolate:

```python
import json, pathlib
src = pathlib.Path("benchmarks/renderer_audit/scenes/calib_lights.json")
base = json.loads(src.read_text())
for light in base["lights"]:
    d = dict(base, name=f"cl_{light['type']}", lights=[light])
    pathlib.Path(f"cl_{light['type']}.json").write_text(json.dumps(d, indent=1))
```

That writes `cl_point.json`, `cl_spot.json`, `cl_rect_area.json` and
`cl_hemisphere.json`. `cl_rect_area` is the subject; `cl_point` and `cl_spot`
are the controls.

### 3.3 The two renders

```bash
<venv-python> benchmarks/renderer_audit/algan_render.py cl_rect_area.json --out out --no-tonemap
```

```bash
node benchmarks/renderer_audit/three_render.mjs cl_rect_area.json --out out --mode both --samples 64 --gl hardware
```

`--gl hardware` needs a GPU; without one expect ~35 s per sample rather than
30 ms. Leave `--tiles` at its hardware default of 4.

Note `algan_render.py` renders with `SETTINGS.raytracing.set(shadows=True)` —
Algan defaults shadows **off**, and the harness turns them on because three.js
has them on.

### 3.4 The measurement

```bash
<venv-python> benchmarks/renderer_audit/shadow_band_probe.py cl_rect_area.json --object probe_rect \
    --images out/cl_rect_area.algan.png out/cl_rect_area.three_pathtrace.png out/cl_rect_area.three_raster.png \
    --labels algan three_pathtrace three_raster
```

It locates the scanline by projecting the scene spec through the spec's own
camera, then reports the profile, the plateau levels, and the smallest `k/K`
grid those levels sit on. A staircase names its own emitter count; a penumbra
fits none. On an unmodified checkout this reproduces §2's table:

```
algan:            min 0.009  flatness 0.87  levels [0.01, 0.25, 0.52, 0.74]
  -> those sit on a k/4 grid (error 0.075): consistent with 4 point emitters
     each carrying 1/4 of the power
three_pathtrace:  min 0.000  flatness 0.49  levels [0.0, 0.33, 0.56, 0.78]
  -> no small-integer k/K grid fits these levels: a continuous penumbra
three_raster:     min 0.583  flatness 0.98
  -> (never darkens: three.js's rasterizer casts no RectAreaLight shadow at all)
```

Then run the same probe on `cl_point.json --object probe_point` as a control: one
light, one hard shadow, no intermediate levels. That is the acceptance test.

## 4. Where the code is

The expansion is host-side and entirely readable — no kernel work is needed to
understand it.

* **`algan/rendering/lights.py`**, `class RectAreaLight` — the declaration.
  `samples=4` by default. `num_samples()` rounds up to the next square number
  ("the emitters are laid out on a square grid"), so 4 → a 2×2 grid.
  `get_sample_positions(location)` returns `[T, K, 3]`, "the centres of a square
  grid covering the rectangle". `_rect_axes` builds the rectangle's basis.

* **`algan/render_loop.py`**, around line 2439 — where an extended light becomes
  a list of point lights:

  ```python
  pos_rows = light.get_sample_positions(loc_f)  # [T, K, 3]
  k = pos_rows.shape[-2]
  col_rows = (col_f / k if k > 1 else col_f).unsqueeze(-2).expand(-1, k, -1)
  ```

  The comment above it states the design plainly: "Area lights expand into K
  samples, each carrying 1/K of the light's power."

* **`algan/rendering/raytracing/scene_builder.py`**, around line 2022 — packs
  those K rows into the flat light arrays the kernel consumes, one row per
  emitter. From here down, nothing knows the K rows were ever one light.

* **`algan/rendering/raytracing/settings.py`** — `SHADOWS` and
  `SOFT_SHADOW_SAMPLES`. The comment there describes the intended design:
  "Lights with a non-zero `shadow_radius` / `shadow_angle` (and area lights) get
  *soft* shadows: a fixed deterministic fan of SOFT_SHADOW_SAMPLES rays is
  traced across the emitter instead of a single ray." Establish early whether
  the rect-area light actually reaches that fan, or whether being expanded into
  K independent point lights means each emitter takes the single-ray path — the
  measured 4 levels say the latter, but confirm it in the code rather than
  inferring it from the image. `SOFT_SHADOW_SAMPLES` is **initialization-only**
  (`ALGAN_SOFT_SHADOW_SAMPLES`, set before `import algan`) and is baked into the
  shade kernel at compile time.

## 5. Two directions, and the trade

**(a) Raise the sample count.** Cheapest, and the design already anticipates it:
`RectAreaLight(..., samples=N)`. It converges — but cost is linear in emitters
for *both* lighting and shadow rays, and it never removes the banding, only
makes the steps smaller. Worth measuring to know the shape of the trade: probe
the scene at `samples` 4, 9, 16, 25 and see where the `k/K` grid stops being
detectable.

**(b) Make the shadow term an integral rather than a sum of hard tests.** The
real fix, and the same one `REPORT.md` §6.7 specifies for the falloff: give each
emitter sample the rectangle's area element and let the visibility term be
averaged over the emitter rather than resolved per point light. §6.7 is worth
reading because it is the *same* mechanism seen in the lighting rather than the
shadows, and it says why it was not done with the third run's fixes:
**it redefines what `intensity` means for every existing `RectAreaLight`.**

Whichever route: the two are related, and a fix that addresses the shadow
staircase while leaving §6.7's falloff wrong (or vice versa) leaves the light
half-corrected. Read §6.7 before choosing.

## 6. What "fixed" looks like

* `shadow_band_probe.py` on `probe_rect` reports **no small-integer `k/K` grid**
  for Algan, and flatness drops from **0.87** toward the reference's **0.49**.
* The scanline profile becomes monotone into the umbra and back out, like the
  reference's `0.87 → 0.81 → 0.70 → 0.56 → 0.41 → 0.32 → 0.09`.
* The umbra stays dark: Algan's min is currently 0.009 against the reference's
  0.000. A "fix" that smooths the staircase by lifting the umbra is wrong.
* **The controls must not move.** `cl_point` and `cl_spot` already cast one
  clean shadow each; re-probe them and check they still report no intermediate
  levels.
* If you change what `intensity` means (route b), say so explicitly — it is a
  user-visible change to every existing `RectAreaLight`, and `REPORT.md` §6.7
  already flags it as the reason this was deferred.

## 7. Hazards specific to this codebase

Read `CLAUDE.md` before changing kernel code. The ones that bite here:

* **`SOFT_SHADOW_SAMPLES` is baked into the shade kernel at compile time** and
  is initialization-only: set `ALGAN_SOFT_SHADOW_SAMPLES` **before**
  `import algan`, and expect a full kernel recompile when you do.
* **A `ti.static` gate is resolved when the kernel compiles.** Flipping a
  setting behind one mid-process does nothing — the second arm silently reuses
  the first arm's code and reports its numbers as its own. **One process per
  arm.**
* **The offline kernel cache does not invalidate on `@ti.func` edits.** Clear it
  with `clear_cache(taichi_kernels=True)` before any kernel A/B.
* **Never edit a `*_taichi.py` file while a render is running.**
* `algan_render.py` sets `ALGAN_USE_DAEMON=0`; keep it. A warm daemon does not
  adopt import-time environment variables and carries the previous run's state.
* On Windows, run render work **one process at a time**.
* Shadow cost is per light *row*, so raising `samples` raises shadow-ray count
  linearly. Time it on a real scene, not a synthetic one, and remember the first
  timing after a kernel edit is a compile, not a measurement.

## 8. Landing it

This changes rendered output, so:

* `<venv-python> -m pytest -q --fast` after every change; the full suite before
  pushing.
* `<venv-python> -m pytest -q tests/full_renders` — `materials_and_lighting.py`
  is the suite scene this audit scene was ported from, and act 3 is exactly this
  light rig, so that is where a change will show.
* Re-baseline only deliberately (`ALGAN_UPDATE_FAST_BASELINE=1` /
  `ALGAN_UPDATE_FULL_RENDER_BASELINES=1`), **look at the diff videos** in the
  suite's `output_errors/` first, and say in the commit why output moved. CPU and
  CUDA baselines are separate committed sets and CUDA needs a CUDA machine.
* Update `REPORT.md` §6.7, which currently records only the falloff half of this
  mechanism, and the §6.7 row of the summary table.
