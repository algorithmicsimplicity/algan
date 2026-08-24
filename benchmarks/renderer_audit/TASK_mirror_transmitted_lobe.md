# Task: a mirror's image of glass is the transmitted lobe, not the reflected one

**Status: FIXED 2026-08-24 for the transmissive case — but read §0 first, because
two of the numbers this brief asks you to move turned out to be measuring
something else.** The cause, the fix and the corrected measurements are in
`REPORT.md` §9.3.1; the rest of this file is preserved as written, so §0 is
where it is contradicted.

**You do not need to read `REPORT.md` to work on this.** Everything needed is
here; `REPORT.md` §9.3 is the same finding written for a reader of the whole
audit, and §4.2 is the earlier Fresnel fix this one sits next to.

---

## 0. What was actually wrong, and what this brief got wrong

**The cause.** The reflected share `R` a hit splits off is traced as a
continuation **ray**, and a ray can only find light that has geometry to hit. A
directional or point light is a delta: no continuation will ever land on one. So
the reflected lobe's response to the direct lights exists only as the analytic
GGX term the material stages evaluate — and that term rides inside the shaded
colour, which the scatter sites weight by `1 - R - trans_share`, the share that
is explicitly *not* reflected. For clear glass `trans_share = 1 - R` exactly, so
the weight collapses to `R * (1 - _mirror_share(roughness))`: **1.2% at
roughness 0.05.** The ball's own reflection is annihilated, and all a mirror can
then show of it is what lies *behind* it, tinted once entering and once leaving.
That albedo² is what §2's table measures.

`_material_reflectance` was innocent, as §4 suspected it might be. The defect is
in how its output is spent, exactly as §4 warned.

**Correction 1 — §2's reference column is mostly the MIRROR's own highlight.**
Render the `mirror` sphere alone on a black background, with nothing in the
scene for it to reflect, and three-gpu-pathtracer still returns
**(4.66, 4.66, 4.66)** over its disc at 99.8% concentration — the directional
light's specular reflection off the mirror itself, which three renders because
it clamps roughness and Algan does not (Algan's GGX `alpha` floor at roughness 0
is 1e-4, a lobe far too narrow to find). That is 82% of the 5.67 the reference
returns with the glass present. **The "small untinted specular highlight" §1
attributes to the glass ball is mostly not the glass ball.** Always render the
mirror-only control before trusting a mirror-disc number.

**Correction 2 — §6's headline target is unreachable and was the wrong metric.**
The mirror-disc `g/r` sits at 1.77 before the fix and 1.77 after, for two
measured reasons: the confound above, and the fact that the ball's image in a
convex mirror is ~4% of its own screen area, so its restored highlight lands in
a handful of pixels. Integrating the GGX lobe over the sphere analytically, the
ball's highlight toward the mirror is **0.77x** its highlight toward the camera;
Algan now measures 1.1x where it measured ~0, while the reference's apparent 7x
is a clipping artifact (its 15 added pixels saturate at 255).

**Measure the transmissive surface's own disc instead.** It is what the defect
is about, and it is not diluted:

| | total (linear R, G, B) | g/r | top-2% |
| --- | --- | --- | --- |
| algan, before | (0.034, 0.063, 0.013) | **1.87** | 94.5% |
| algan, after | (1.129, 1.159, 1.108) | **1.03** | 98.9% |
| three path tracer | (2.815, 3.008, 2.656) | 1.07 | 99.2% |

```bash
<venv-python> benchmarks/renderer_audit/mirror_tint_probe.py <scene> \
    --images <images> --labels <labels> --mirror glass --source glass
```

(`--mirror` names the disc to measure, so pointing it at the glass ball works.)

§6's other four requirements all hold: `calib_mirror` 0.9004 and `calib_glass`
0.9134 are unchanged to four decimals, the opaque control does not move, and
total energy does not collapse.

**Still open: the opaque half.** The same defect throttles an opaque rough
metal's direct-light highlight the same way. Extending the fix there moves
`tests/fast` by 25 channel values, which is the CORRECT restoration and not a
surprise: the weight in the default split-sum arm is `_material_env_brdf`, the
lobe's exact directional albedo, and for the fast scene's
`MeshStandardMaterial(roughness=0.35, metalness=0.4)` that is **25x**
`R * _mirror_share(0.35)` (0.344 against 0.0138 in red). The identity
`share + R = 1` holds in that arm too. What blocks the extension is not the
arithmetic but the baselines: it moves `tests/fast` and every full-render
scene, and the CUDA set cannot be regenerated on a CPU-only machine.
`OX_DIRECT_SPECULAR_REVIEW.md` closes the arithmetic independently, by
dumping the merged batch to confirm every argument the add-back and the
shading stage share carries the same value at the same hit, and predicting
+27 bytes against the +25 measured.

**And the transmitted lobe is NOT carrying too much energy — checked against
theory, not against the reference.** `calib_transmittance` /
`calib_transmittance_tinted` + `transmittance_probe.py` force the answer with
an unlit backdrop, no lights and normal incidence: the centre must transmit
`(1-F)^2` times the base colour once per crossing. Algan measures 0.9216
against 0.9216 white (0.00% error) and (0.3140, 0.5333, 0.1651) against
(0.3144, 0.5331, 0.1649) tinted (0.13%). Exact, and the tinted row pins the
tint's *order* too.

Do not try to establish this by differencing the reference's mirror disc. Two
traps, both hit on the way here: the feature is ~15 px in a 4100 px disc, so at
32 samples the reference's blob is pure noise (it triples in red and octuples
in blue at 256 samples); and the path tracer drops `AmbientLight` entirely, so
its sources are several times dimmer and any absolute cross-engine ratio is
meaningless. Normalised by each engine's own source at 256 samples, Algan's
blob is (0.57, 0.84, 0.55) of the reference's — below it, not above.

---

## 1. The defect in one paragraph

Put a perfect mirror and a transmissive glass sphere side by side. Algan's
mirror shows the glass ball as a **broad patch tinted by the glass's own
colour, brighter than the brightest pixel of the ball it is reflecting**. The
physically-based reference (`three-gpu-pathtracer`) shows a **small, untinted
specular highlight** instead. The two carry comparable total energy, so Algan is
not inventing light — it is putting it in the *transmitted* lobe where the
reference puts it in the *reflected* one. At the grazing angles a mirror sees a
sphere's limb through, Fresnel reflectance approaches 1 and reflection should
dominate, which is why the reference's reading is the one to move toward.

## 2. The measurement that defines it

Scene: `scenes/matlight_pbr_subset.json`. The `mirror` sphere (metalness 1,
roughness 0) reflects the `glass` sphere (transmission 1, ior 1.5, albedo a pale
green). Measured over the mirror's whole disc, **on a black-background copy of
the scene** so the numbers are purely reflected geometry (see §5 for why):

| glass sphere variant | mirror-disc total (linear R, G, B) | g/r | g/b |
| --- | --- | --- | --- |
| green, transmission 1 — as authored | (4.81, 8.53, 3.14) | **1.77** | **2.72** |
| green, transmission **0** | (68.5, 88.0, 48.9) | 1.28 | 1.80 |
| **white**, transmission 1 | (10.2, 13.7, 14.1) | 1.34 | 0.97 |
| three-gpu-pathtracer, as authored | (7.64, 7.27, 5.49) | **0.95** | 1.32 |

The glass albedo in linear light is (0.584, 0.761, 0.423), so:

* **one tinting** (albedo) predicts g/r **1.30**, g/b **1.80**
* **two tintings** (albedo²) predicts g/r **1.70**, g/b **3.23**
* an **untinted specular reflection** predicts g/r **1.00**, g/b **1.00**

(`mirror_tint_probe.py` prints those three predictions for you, so you never
have to work them out by hand.)

Algan sits at **1.77** — the tint applied about twice, i.e. light that crossed
into the tinted medium and back out. Make the ball opaque and it drops to one
albedo exactly (1.28 vs 1.30 predicted). Make it white and the tint vanishes.
The reference sits at **0.95**, i.e. untinted.

Concentration says the same thing from the other side: the brightest 2% of the
mirror's pixels hold **95%** of the reference's energy and only **81%** of
Algan's. The reference puts it in a highlight; Algan spreads it.

In the as-authored (grey background) scene the same thing reads as:

| | the glass sphere itself | its reflection in the mirror |
| --- | --- | --- |
| algan | mean (18, 25, 11), max **(31, 40, 34)** | brightest-40 mean **(49, 79, 46)** |
| three | mean (16, 24, 10) | brightest-40 mean (20, 37, 21) |

## 3. Reproduce it

### 3.1 Environment

Algan renders need the project venv (this is the only place the path is
written down; see `CLAUDE.md`):

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

Pin `three@0.185.1` and `three-gpu-pathtracer@0.0.24` — those are the versions
every number above was measured against.

### 3.2 Build the black-background scene first

**Do this before rendering anything.** On the scene as authored the backdrop
swamps the measurement: Algan returns the background colour to an escaped
secondary ray, so the mirror renders the backdrop and the probe reads
`total (68.7, 72.7, 66.9) g/r 1.06 top-2% 6.6% mean_u8 (34, 35, 34)` — 34 being
the background's own 8-bit value. That is a *different*, deliberate convention
(§5, item 1), and it hides this defect completely. On black it goes away.

```python
import json, pathlib
src = pathlib.Path("benchmarks/renderer_audit/scenes/matlight_pbr_subset.json")
d = json.loads(src.read_text())
d["name"] = "mirror_task_black"
d["render"]["background"] = [0.0, 0.0, 0.0]
pathlib.Path("mirror_task_black.json").write_text(json.dumps(d, indent=1))
```

For the two controls in §2's table, copy that file again and set, on the object
named `glass`, either `material.transmission = 0.0` or
`material.color = [1, 1, 1]`.

### 3.3 The two renders

From the repo root:

```bash
<venv-python> benchmarks/renderer_audit/algan_render.py mirror_task_black.json --out out --no-tonemap
```

```bash
node benchmarks/renderer_audit/three_render.mjs mirror_task_black.json --out out --mode pathtrace --samples 64 --gl hardware
```

`--gl hardware` needs a GPU; drop it on a machine without one and expect about
35 s per sample instead of 30 ms. On a GPU, leave `--tiles` at its default (4) —
at 1 a single draw call can outrun the display driver's watchdog and you get a
silently blank frame (the back end now throws instead, but do not fight it).

### 3.4 The measurement

```bash
<venv-python> benchmarks/renderer_audit/mirror_tint_probe.py mirror_task_black.json \
    --images out/mirror_task_black.algan.png out/mirror_task_black.three_pathtrace.png \
    --labels algan three_pathtrace --mirror mirror --source glass
```

It prints the total linear energy, the `g/r` and `g/b` tint ratios, the
albedo / albedo² predictions to read them against, and the top-2%
concentration. On an unmodified checkout this reproduces §2's table exactly:

```
algan            total (   4.813,   8.527,   3.137)  g/r  1.77  g/b  2.72  top-2% holds  80.8%
three_pathtrace  total (   7.642,   7.268,   5.486)  g/r  0.95  g/b  1.32  top-2% holds  95.0%
```

That is the whole acceptance test.

## 4. Where the code is

The reflection of a *secondary* ray is decided in the wavefront bounce loop.

* **`algan/rendering/raytracing/wavefront_kernels_taichi.py`**
  * `_material_reflectance(...)` — the Fresnel split. Returns `(R, diel_pass)`,
    where `R` is the metal-blended reflectance and
    `diel_pass = (1-m)(1-r_diel)` is the share that enters the dielectric
    interior. Its docstring is long and worth reading in full: it already
    implements the KHR_materials_volume side test (Schlick at the incident angle
    entering, at the air-side partner angle leaving, `F = 1` past the critical
    angle) that §4.2 of the audit fixed. **This function looked correct under
    inspection** — the defect is more likely in how its output is spent than in
    the value it returns, but verify rather than assume.
  * `_scatter_impl(...)` — spends it. The four shares a hit splits into are
    written out in a comment there: `alpha*(1-R-trans_share)` shaded here,
    `alpha*R` reflected, `alpha*trans_share` transmitted (albedo-tinted),
    `1-alpha` missed, with `trans_share = diel_pass * T`. **`tint` is applied to
    the transmitted share and to the metal share of the Fresnel lobe; the
    dielectric reflection is documented as staying achromatic.** Check that this
    holds on the path a *continuation* ray takes, not only on a camera ray.
  * Note the documented scope limit right there: "**Roughness does not fade the
    bounce here**... A mirror reflecting a rough metal shows that metal's
    reflection unfaded as well as unblurred". The glass here is roughness 0.05,
    so this is adjacent to, but not obviously the cause of, the concentration
    difference.

* Three things that are **not** the mechanism, already checked -- see §5.

## 5. Already ruled out — do not re-derive these

Each was ruled out by measurement, not by reading code.

1. **Not the background.** Algan returns the background colour to an
   escaped secondary ray, so a mirror can render the backdrop and vanish into it
   (`REPORT.md` §9.4 — a separate, deliberate convention). That is *not* this:
   re-rendered on a black background the mirror disc collapses from 34 to 1.6 in
   8-bit, while the green patch does not move — (49, 79, 46) → (48, 77, 44).
   Work on a black background anyway; it removes a constant that dilutes every
   ratio, which is what §3.2 is for.

2. **Not a difference between Algan's two shading paths.** Primary visibility
   resolves through the sheet route and a reflection through the wavefront bounce
   loop, so they were compared head-on: with `ALGAN_ANALYTIC_AA=0` forcing the
   whole frame through the classic wavefront, the glass disc and the reflected
   patch both reproduce to within **1.5/255**. The two routes agree, so this is
   not a sheet-versus-wavefront discrepancy.

   ```bash
   ALGAN_ANALYTIC_AA=0 <venv-python> benchmarks/renderer_audit/algan_render.py \
       <scene> --out out --suffix wavefront --aa 1 --no-tonemap
   ```

   (Use `--aa 1`: the wavefront route supersamples, and AA=3 at 960×540 does not
   fit in 4 GB of VRAM.)

3. **Not the closed-shell coverage ceiling.** Algan caps a closed shell's
   cumulative coverage so a translucent solid composites once rather than twice
   — but a surface whose material *transmits* is folded back to open at pack
   time. See `declare_closed_shell` in
   `algan/rendering/raytracing/primitives.py`: "A surface whose material
   transmits is folded back to open at pack time (`_rt_tri_closed`): refraction
   visits both shells as physical transport, and the ceiling would eat the
   second one." The ceiling never applies to this sphere.

## 6. What "fixed" looks like

* The mirror-disc `g/r` moves from **1.77** toward **1.00–1.30** — an untinted
  or singly-tinted reflection rather than a doubly-tinted transmission. Getting
  to exactly the reference's 0.95 is not required; getting off albedo² is.
* **Total energy must not collapse.** This is a redistribution between lobes, so
  a "fix" that simply darkens the mirror is wrong. Algan's disc total is
  (4.81, 8.53, 3.14) against the reference's (7.64, 7.27, 5.49); stay in that
  range.
* The concentration (top-2% energy share) should rise from 0.81 toward the
  reference's 0.95, but treat that as a signal rather than a target — it is also
  sensitive to the roughness-does-not-blur-the-bounce scope limit noted in §4.
* The **opaque control must not move**: with `transmission = 0` the same scene
  measures g/r 1.28 against the albedo's 1.30, which is already correct. If that
  number changes, the fix has reached surfaces it should not.
* `calib_mirror` must stay right — it has no transmissive geometry and is the
  control for reflection itself. `metrics.py` takes no arguments and reads
  whatever is in `out/`:

  ```bash
  <venv-python> benchmarks/renderer_audit/metrics.py
  ```

  On an unmodified checkout that reports `calib_mirror` reflection efficiency
  **0.9004 for Algan against 0.8904 for the path tracer** (and 0.0 for the
  rasterizer, which cannot reflect without an environment map — the reason the
  path tracer is the reference here at all). `calib_glass` transmission
  efficiency is **0.9134 against 0.9376**. Both should survive the change.

## 7. Hazards specific to this codebase

Read `CLAUDE.md` in full before changing kernel code. The ones that bite here:

* **Never edit a `*_taichi.py` file while a render is running** — the JIT reads
  sources at first launch and can compile half-edited code.
* **The offline kernel cache does not invalidate on `@ti.func` edits.** Clear it
  with `clear_cache(taichi_kernels=True)` before A/B-benchmarking a change to
  `_material_reflectance` or `_scatter_impl`, or you will measure the old code.
* **A `ti.static` gate is resolved at compile time**, so flipping the setting
  behind one mid-process does nothing and the second arm silently reuses the
  first arm's code. **Run one process per arm.**
* `algan_render.py` already sets `ALGAN_USE_DAEMON=0`; keep it that way, since a
  warm daemon carries the previous run's adaptive state and will not adopt
  import-time environment variables.
* On Windows, run render work **one process at a time**.
* Cold kernel compilation takes minutes. The first timing after a kernel edit is
  not a measurement.

## 8. Landing it

This changes rendered output, so:

* Run `<venv-python> -m pytest -q --fast` after every change, and the full suite
  before pushing.
* Run `<venv-python> -m pytest -q tests/full_renders` — a transport change is
  **invisible to `--fast`**, whose one scene has no transmissive solid.
* If frames legitimately move, re-baseline with `ALGAN_UPDATE_FAST_BASELINE=1` /
  `ALGAN_UPDATE_FULL_RENDER_BASELINES=1`, **look at the diff videos** in the
  suite's `output_errors/` before committing, and say in the commit message why
  output moved. CPU and CUDA baselines are separate committed sets and CUDA
  needs a CUDA machine.
* Update `REPORT.md` §9.3 and the summary table row for §9.3, and strike item 6
  from §7's follow-up list.
