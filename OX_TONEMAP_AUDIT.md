# Tonemapping path audit (read-only)

Repo root `/home/user/algan`, audited at the current checkout. No files were
modified; the only arithmetic executed was the closed-form table in §5,
evaluated in plain Python from the transcribed source.

---

## 1. Every place a tonemap curve is applied to pixel colour

The list of three sites is **complete** for `algan/` (grep for `tonemap`,
`aces`, `reinhard`, `filmic`, curve constants `6.25`/`0.76`; nothing else
applies a curve to pixels). But it is really **two implementations**, not
three: site 2 imports the exact ti.funcs of site 1.

### Site A — in-composite Taichi: `finalize_pixel_color`
- `algan/rendering/raytracing/raytrace_kernels_taichi.py:1882`
  (`pbr_neutral_tonemap` :1855, `agx_tonemap` :1822 +
  `agx_default_contrast_approx` :1815).
- **Selected by** the compile-time template arg `tonemapping` ∈ {0,1,2,3},
  which every caller passes as `_get_tonemap_t_val()`:
  - Monte Carlo: `finalize_samples` (`raytrace_kernels_taichi.py:3200`,
    call at :3216).
  - Wavefront composites: `wf_composite` (`wavefront_kernels_taichi.py:1348`),
    `wf_composite_accum` (:1447), `wf_composite_accum_sparse` (:1490),
    `wf_finalize_uncovered` (:1529), `wf_finalize_aa` (:1581).
- Runs **only when `POST_PROCESS_TONEMAP=False`** (t_val ≠ 3); see §2.
- **Input range**: byte scale, 0–255 (+HDR headroom above 255 possible).
  Accumulators are multiplied ×255 at composite time
  (`wavefront_kernels_taichi.py:1347,1446,1489`; `raytrace_kernels_taichi.py:3212-3215`),
  then `finalize_pixel_color` itself does `/255` inside the curve argument:
  `pbr_neutral_tonemap(color_hdr * (tonemap_exposure / 255.0)) * 255.0`
  (`raytrace_kernels_taichi.py:1885`, agx :1887). So the curve sees
  exposure-scaled 0–1 values and its result is rescaled to 0–255.

### Site B — standalone post-process Taichi kernel: `tonemap_to_u8`
- `algan/rendering/post_processing/tonemap_kernels_taichi.py:28`; reuses
  `pbr_neutral_tonemap`/`agx_tonemap` imported from site A (:21-24).
- **Selected by** `_finalize_on_device`
  (`post_process.py:206-210`): requires `tonemap_enabled`
  (= `POST_PROCESS_TONEMAP`), a non-uint8 frame, and
  `is_post_tonemap_kernel_enabled()` (`settings.py:2128`). `method_id`:
  0 = clamp-only, 1 = neutral, 2 = AgX (`post_process.py:215`).
- **Input range**: linear 0–1 (+HDR >1). The frame was already divided by
  255 in `post_process_frames` (`post_process.py:321`) before finalize;
  exposure is multiplied inside the kernel (`tonemap_kernels_taichi.py:45,47`).
  Quantises with `*255 + 0.5`, clamp, `ti.cast(..., ti.u8)` (:50-52).
  Glow channel dropped; alpha (channel 4, never normalised) clamped only (:53-55).

### Site C — torch post-process: `_neutral_tonemap` / `_agx_tonemap`
- `algan/rendering/post_processing/post_process.py:23` and :95.
- **Selected by** the same `_finalize_on_device`, when the site-B gate fails
  (`POST_TONEMAP_KERNEL=0`) or the frame arrived as uint8 while
  `POST_PROCESS_TONEMAP=True`: dispatch at `post_process.py:234-240`.
- **Input range**: same 0–1 linear (+HDR) tensor; if the frame was uint8 it is
  cast and divided at :228-230 ("a caller handed over bytes" case). Exposure
  applied inside the functions (:26, :98). Rescale/quantise at :245-247:
  `*255, +0.5, clamp(0,255)`, then `copy_` into a uint8 tensor (truncation).

Not curves, for completeness: the TONEMAPPING-off clamp paths inside all
three sites (§6), bloom (`bloom.py`), and FXAA/SMAA. `manim_defaults.py:221`
merely sets the flag off.

## 2. Which runs by default

Defaults (`algan/rendering/raytracing/settings.py`):

| flag | default | line |
| --- | --- | --- |
| `TONEMAPPING` | `True` | :63 |
| `TONEMAP_EXPOSURE` | `1.0` | :64 |
| `TONEMAP_METHOD` | `"neutral"` | :65 |
| `POST_PROCESS_TONEMAP` | env-flag default `True` | :74 |
| `POST_TONEMAP_KERNEL` | env-flag default `True` | :2114 |

Exact default chain:

1. `_get_tonemap_t_val()` returns **3 immediately** because
   `POST_PROCESS_TONEMAP` short-circuits first (`settings.py:2137-2139`);
   `TONEMAPPING`/`TONEMAP_METHOD` are not even consulted in-kernel.
2. Frame buffer dtype = `hdr_frame_dtype()` = float32 (float16 only under
   `ALGAN_HDR_BUFFER_F16=1`; `settings.py:2095-2111`), chosen at
   `render_loop.py:1050-1052` and `tracer.py:1443-1454`.
3. Route split (all write **linear, un-tonemapped** values under t_val=3):
   - Default route (SPP=1 per `settings.py:32`, sheet route accepted):
     sheet resolve writes `ec[k] * 255.0` (`sheet_resolve_taichi.py:943`),
     covered pixels composited linearly by `wf_composite_accum_sparse`
     (`wavefront_kernels_taichi.py:1490-1496`), untouched pixels keep the
     prefilled background (`scene_builder.py:2054`, byte-scale into the
     float buffer).
   - Fallback wavefront (route vetoed, `tracer.py:442-535`): linear writes in
     `wf_composite_accum` / `wf_finalize_aa`.
   - SPP>1 Monte Carlo: `finalize_samples` averages and writes linear floats
     (`raytrace_kernels_taichi.py:3212-3221`, t_val=3 branch skips the u8 cast).
4. `post_process_frames` (`post_process.py:264`): `hdr=True` (:280) → AA
   downsample stays float → colour+glow channels `/255` once, in place (:312-321)
   → user post-processes/bloom run on linear HDR → `_finalize_on_device`
   with `tonemap_enabled=True, tonemapping=True, method="neutral",
   exposure=1.0` (:340-348).
5. Because `POST_TONEMAP_KERNEL=True` and the frame is float, the **site B**
   kernel runs with `method_id=1` → `pbr_neutral_tonemap(c * 1.0)`
   (`tonemap_kernels_taichi.py:44-45`).

**So the default implementation is the Taichi `pbr_neutral_tonemap` via
`tonemap_to_u8`.** The torch `_neutral_tonemap` (site C) runs only under
`ALGAN_POST_TONEMAP_KERNEL=0`; site A's curves run only under
`ALGAN_POST_PROCESS_TONEMAP=0`. Setting `SETTINGS.raytracing.tonemapping`
alone (as `use_manim_defaults()` does, `manim_defaults.py:221`) keeps t_val=3
and just switches the post stage's `method_id` to 0 / the torch clamp branch.

## 3. Do the implementations agree numerically?

### neutral: torch `_neutral_tonemap` vs Taichi `pbr_neutral_tonemap`

Line-by-line, algebraically **identical**, including both strictness choices:

| step | torch (`post_process.py`) | taichi (`raytrace_kernels_taichi.py`) |
| --- | --- | --- |
| min channel | `amin` :30 | `ti.min(min())` :1859 |
| low branch test | `< 0.08` strict :32 | `x < 0.08` strict :1861 |
| offset | `min²·6.25`, `min − that`, else `0.04` :34-39 | `x − 6.25·x²`, else `0.04` :1860-1862 |
| subtract | `.sub_(offset)` :42 | `color - offset` :1864 |
| peak / gate | `amax`, `>= 0.76` :44-46 | `max`, `peak >= startCompression` :1866-1868 |
| newPeak | `(peak+0.24)−0.76` :52, `0.0576/that`, `1−…` :54-55 — same association | `d=0.24`; `1 − d²/(peak+d−sc)` :1869-1870 |
| scale | `new_peak/peak`, where-gate to 1.0 :57-60 | `newPeak/peak` :1871 |
| desat g | `(peak−newPeak)·0.15+1`, reciprocate, `1−…` :64-66 | identical form :1873 |
| mix toward newPeak | `(newPeak−c)·g + c` :69-70 | `mix(c, newPeak, g)` :1874-1876 |
| final clamp | `clamp_(0,1)` :72 | `clamp(out,0,1)` :1878 |

Residual divergences (all last-ulp class, none structural):

1. **mix rounding order**: `ti.math.mix(x,y,a)` computes `x*(1−a)+a*y`
   (taichi `math/mathimpl.py:104-107`); torch computes `(y−x)*a+x`
   (`post_process.py:69-70`). Same value in exact arithmetic, ≤1 ulp apart
   in floating point.
2. **fast_math**: Taichi initialises with `fast_math: True`
   (`taichi_runtime.py:389`), so mul+add pairs in the kernel can contract to
   FMA and divisions may be approximate; torch ops round individually. The
   codebase documents exactly this class of divergence elsewhere
   (`settings.py:1827-1828`).
3. **dtype**: the torch pipeline runs in the buffer's dtype — float16
   throughout if `ALGAN_HDR_BUFFER_F16=1` (the f16 frame passes unconverted
   through `post_process.py:232` and every scratch inherits
   `color_exposed.dtype` :29ff); the Taichi kernel always computes f32
   (`tonemap_kernels_taichi.py:10-11`). Under the default (f32) they match.
4. **Exposure association**: `finalize_pixel_color` multiplies
   `c * (exposure/255)` (`raytrace_kernels_taichi.py:1885`); the post paths do
   `(c/255) * exposure` (`post_process.py:321` then :26/:45). Differs by ≤1 ulp
   for exposure ≠ 1. Irrelevant by default (exposure 1.0).

### agx: torch `_agx_tonemap` vs Taichi `agx_tonemap`

Algebraically **identical**: Rec.2020 matrix rows match
(`post_process.py:106-108` vs `raytrace_kernels_taichi.py:1823-1825`), inset
matrix :116-118 vs :1827-1829, log encode :124-125 vs :1831-1837 (torch uses
`log2_()`; taichi `ti.log(x)/ln(2)` — same function, potentially different
libm rounding, plus fast_math), the contrast polynomial term-for-term with the
same multiplication association (:77-92 vs :1818), inverse matrices
(:136-141, :146-149 vs :1843-1849), final clamp :152 vs :1851. Divergence
risks are the same four as neutral (mix n/a here; log2 and FMA matter most).

I did not execute either implementation against each other (no render allowed);
"algebraically identical, possibly ≠ by ULPs" is a source-level conclusion.

## 4. sRGB ↔ linear anywhere?

**Confirmed: no.** Grepping the whole package excluding
`algan/external_libraries/` finds zero sRGB transfer functions, gamma
expansions/compressions, or `pow(x, 1/2.2)`-style code. The only hits for
"srgb" are variable names inside AgX's *output gamut matrix*
(`r_srgb/g_srgb/b_srgb`, `raytrace_kernels_taichi.py:1847-1849`, same matrix
in `post_process.py:146-148`) — a linear-to-linear colour transform misnamed
"srgb", not an encode. Timeline `linearizes` (`timeline.py:399`) is unrelated.

So the colour path treats display-encoded values as if they were linear end
to end:

- Authored colours: integer tuples parsed with plain `/255`
  (`constants/color.py:95-104`); `RED=(255,0,0)` becomes `1.0` and reaches the
  tonemap as-is. Confirmed.
- Textures: `/255` without decode (`utils/file_utils.py:53`,
  `mobs/image_mob.py:65`, `mobs/three_d_models/model_mob.py:122`,
  `gltf_loader.py:52,66`).
- Backgrounds: solid colours arrive `[0,1]` and are scaled ×255 at prefill
  (`scene_builder.py:2054`); the HDR normalisation is a plain `/255`
  (`post_process.py:321`).

If linearisation existed it would need (a) an sRGB→linear decode at
ingestion — `constants/color.py` parsing and the texture loaders above — and
(b) a linear→sRGB encode after the tonemap, before quantisation, in **each**
of the three output sites (inside/after the curve in `finalize_pixel_color`,
in `tonemap_to_u8`, and at the `scaled` step `post_process.py:245-247`), since
any of them can produce final bytes. Note decoding at ingestion alone would
also change all lighting/shading math, which currently operates on encoded
values.

## 5. Closed-form mapping of the default curve (neutral, exposure 1)

Per pixel, with `x = min(rgb)`, `peak = max(rgb − offset)`:

```
offset = x < 0.08 ? x − 6.25·x² : 0.04          # raytrace_kernels_taichi.py:1860-1862
co     = rgb − offset                            # :1864
if peak ≥ 0.76:                                  # :1868
    newPeak = 1 − 0.0576/(peak − 0.52)
    co      *= newPeak/peak
    g       = 1 − 1/(0.15·(peak − newPeak) + 1)
    co      = mix(co, newPeak, g)                # every channel pulled toward newPeak
out    = clamp(co, 0, 1)                         # :1878 ; then *255+0.5, truncate
```

For greys below the compression onset (`v − offset < 0.76 ⇔ v < 0.80`) the map
collapses to the **uniform shift `v − 0.04`**, i.e. −10.2/255 for every
uncompressed grey ≥ 0.08 — the CLAUDE.md "darkens every flat fill by a uniform
10/255". Compression onset sits at v = 0.80 (v=0.79→0.750, v=0.80→0.760).

Table (grey inputs, 0–1 units; u8 = round-half-up of out×255):

| v in | v×255 | out | u8 out | Δu8 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.000000 | 0 | 0 |
| 0.02 | 5.10 | 0.002500 | 1 | −4 |
| 0.04 | 10.20 | 0.010000 | 3 | −7 |
| 0.08 | 20.40 | 0.040000 | 10 | −10 |
| 0.20 | 51.00 | 0.160000 | 41 | −10 |
| 0.50 | 127.50 | 0.460000 | 117 | −11 |
| 0.76 | 193.80 | 0.720000 | 184 | −10 |
| **1.00** | **255.00** | **0.869091** | **222** | **−33** |
| 2.00 | 510.00 | 0.960000 | 245 | −10 |
| 4.00 | 1020.00 | 0.983256 | 251 | −4 |

At exactly 1.0: the offset colour is 0.96 ≥ 0.76, so white compresses hard —
**255 lands on 222** (−33), the largest shift on the table. Saturated authored
colours behave differently from grey 1.0 because their peak after offset is
larger: `RED → (224, 4, 4)` (R darkened 31, G/B lifted off zero by the
desaturation mix), `WHITE(255,255,255) → 222` grey. The output never exceeds
1.0 (newPeak < 1 always for peak > 0.52), so the final clamp is inert except
for NaN safety; HDR inputs (2.0, 4.0) roll off toward 255 without reaching it.

## 6. The `TONEMAPPING=False` path

Key point: with everything else default, `TONEMAPPING=False` does **not**
change what the kernels do — `_get_tonemap_t_val()` still returns 3
(`settings.py:2138-2139`), the render still writes linear floats, and only the
post stage changes. Exact arithmetic per implementation:

- **Site B kernel** (`method_id=0`, `tonemap_kernels_taichi.py:48-52`):
  `c = clamp(c, 0, 1)`; each channel
  `u8 = ti.cast(clamp(c·255 + 0.5, 0, 255), ti.u8)`. Clamps twice (before
  scaling, and after +0.5); the **+0.5 is added after the ×255**; `ti.cast`
  truncates toward zero, so this is round-half-up. Alpha clamped to [0,255]
  (:53-55).
- **Site C torch** (`post_process.py:241-247`):
  `rgb_tonemapped = clone(rgb); clamp_(0,1)` (:242-243) → `×255, +0.5,
  clamp(0,255)` (:245-246) → `copy_` into the uint8 output, which truncates.
  Same numbers as the kernel. Alpha clamped and copied (:248-255).
- **Sites A in-composite** (reached only when `POST_PROCESS_TONEMAP=False`
  too, giving t_val=0): `finalize_pixel_color` else-branch
  (`raytrace_kernels_taichi.py:1890-1902`):
  `clamp(csum·inv_samples, 0, 255)` — note the input is byte-scale and the
  **+0.5 is added to all four lanes including glow** before the final clamp
  and u8 cast. The sparse sheet route additionally launches
  `wf_finalize_uncovered` so empty pixels get `finalize(bg)`
  (`tracer.py:2582-2605`, `wavefront_kernels_taichi.py:1504-1535`). The
  torch finalize takes its byte fast-path and strips only
  (`post_process.py:182-183`).

Net effect: `TONEMAPPING=False` yields `clamp(v,0,1)·255` rounded half-up —
byte-faithful for authored 0–255 colours (up to f32 `v/255·255` rounding).

## 7. Tests and baselines sensitive to a neutral-curve change

Grep of `tests/` for tonemapping references: only three files touch it.

Would move (render pixels with the **default** neutral tonemap):

- `tests/fast/test_fast_render.py:175` — renders `tests/fast/scene.py` with
  untouched settings (scene sets only a background colour,
  `tests/fast/scene.py:58`), compares every frame against
  `expected_outputs_{cpu,cuda}/fast.mp4` with tolerance 2
  (`tests/conftest.py:77`). Any curve change moves it.
- `tests/full_renders/test_full_renders.py` — all six scenes render with the
  default tonemap; the only scene-level renderer setting in the suite is
  `shadows=True` (`scenes/materials_and_lighting.py:22`; the word
  "tonemapping" at :241 is a figure caption). All six baselines in
  `expected_outputs_{cpu,cuda}/` would move.

Would NOT move:

- `tests/unit_tests/test_deterministic_shadow_opacity.py:73` — renders, but
  with `tonemapping=False` and loose relative-luminance assertions; immune to
  curve shape.
- `tests/unit_tests/test_manim_defaults.py:91-96` — behavioural assertion that
  the flag flips; independent of curve values.

Reminder from CLAUDE.md: baselines are per machine/device; regenerating the
CPU set here does not refresh the CUDA set.

## Caveats

- §3's comparison is source reading, not execution; ULP-level claims
  (FMA/fast_math, `log2` libm differences, mix order) are reasoned, not
  measured.
- I did not exhaustively read every composite variant's full body — the five
  `finalize_pixel_color` call sites were checked at their finalize lines, and
  the sheet-resolve shade path confirmed curve-free by grep (it writes linear
  colour only, `sheet_resolve_taichi.py:943` is the sole ×255 store).
