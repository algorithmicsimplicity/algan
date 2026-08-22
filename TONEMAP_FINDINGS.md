# Default tonemapping: what it does, and why

Written 2026-08-22. The question asked was whether the default tonemap is
working correctly, on the stated belief that "tonemapping should only affect
HDR values, but that isn't the case at the moment".

**The belief is correct, and it is measurable.** The default tonemap alters
every non-black colour in the frame, whether or not that colour was ever
outside the display range. What follows is the measurement, the mechanism, and
what can and cannot be done about it.

`OX_TONEMAP_AUDIT.md` is the companion code-path map -- every site that
applies a curve, which one runs by default, and the flag chain that selects it.

Three scripts reproduce everything here:

    <venv-python> benchmarks/_tonemap_transfer_probe.py    # the curve, in isolation
    <venv-python> benchmarks/_tonemap_render_check.py      # the same thing, on real pixels
    <venv-python> benchmarks/_tonemap_hdr_occupancy.py     # how much HDR there actually is

## 1. What the default does to authored colour

`benchmarks/_tonemap_render_check.py` renders a flat fill covering the whole
frame, once with `tonemapping=True` (the default) and once off, and reads the
centre pixel back out of the encoded PNG. The tonemap-off column is the
control: it reproduces the authored bytes **exactly** for all nine colours, so
the difference between the columns is the tonemap and nothing else.

|  colour |         authored |       tonemap ON |      tonemap OFF |  ON − authored |
| ------: | ---------------: | ---------------: | ---------------: | -------------: |
|   white |  (255, 255, 255) |  (222, 222, 222) |  (255, 255, 255) | (−33, −33, −33)|
|  grey75 |  (191, 191, 191) |  (181, 181, 181) |  (191, 191, 191) | (−10, −10, −10)|
|  grey50 |  (128, 128, 128) |  (118, 118, 118) |  (128, 128, 128) | (−10, −10, −10)|
|  grey25 |     (64, 64, 64) |     (54, 54, 54) |     (64, 64, 64) | (−10, −10, −10)|
|  grey10 |     (26, 26, 26) |     (16, 16, 16) |     (26, 26, 26) | (−10, −10, −10)|
|     red |      (255, 0, 0) |      (224, 4, 4) |      (255, 0, 0) |   (−31, +4, +4)|
|   green |      (0, 255, 0) |      (4, 224, 4) |      (0, 255, 0) |   (+4, −31, +4)|
|    blue |      (0, 0, 255) |      (4, 4, 224) |      (0, 0, 255) |   (+4, +4, −31)|
|  yellow |    (255, 255, 0) |    (224, 224, 4) |    (255, 255, 0) | (−31, −31, +4)|

Three separate errors are visible there, and only the first was on record:

1. **A flat −10/255 pedestal on every mid-tone.** Independent of the value.
2. **White is not white.** An authored `255` lands on `222`. That is −33, not
   −10, and it is the most visible of the three: in
   `algan_outputs/tonemap_check/scene_on.png` the white square and the white
   circle borders render as light grey.
3. **Saturated colours desaturate.** Pure red `(255, 0, 0)` comes back as
   `(224, 4, 4)` — the red channel drops 31 *and* the two zero channels are
   lifted to 4. A primary is no longer primary.

None of these inputs was ever above the display range. Every one of them was
altered.

## 2. The curve, measured in isolation

`benchmarks/_tonemap_transfer_probe.py` hands the post stage a synthetic
linear-HDR ramp, so the input can go above 1.0 where a flat authored fill
cannot. `ON` and `OFF` are the encoded byte for each input:

| linear in | ideal u8 | tonemap ON | tonemap OFF | ON − OFF |
| --------: | -------: | ---------: | ----------: | -------: |
|     0.000 |        0 |          0 |           0 |       +0 |
|     0.040 |       10 |          3 |          10 |       −7 |
|     0.080 |       20 |         10 |          20 |      −10 |
|     0.250 |       64 |         54 |          64 |      −10 |
|     0.500 |      128 |        117 |         128 |      −11 |
|     0.750 |      191 |        181 |         191 |      −10 |
|     0.800 |      204 |        194 |         204 |      −10 |
|     0.950 |      242 |        217 |         242 |      −25 |
|     1.000 |      255 |        222 |         255 |      −33 |
|     1.500 |      255 |        239 |         255 |      −16 |
|     2.000 |      255 |        245 |         255 |      −10 |
|     4.000 |      255 |        251 |         255 |       −4 |
|     8.000 |      255 |        253 |         255 |       −2 |

**The only input the default curve leaves alone is exactly 0.0.** Everything
else moves. The rows above 1.0 are the ones the tonemap exists for, and there
it is doing real work — 1.5, 2.0, 4.0 and 8.0 are four distinguishable bytes
under the curve and one flat `255` without it.

The probe also confirms the two shipping implementations agree: the Taichi
post-process kernel (`post_tonemap_kernel=True`, the default) and the torch
pipeline it replaced produce byte-identical output on every row.

## 3. Where it comes from

The curve is the **Khronos PBR Neutral** tonemapper, and Algan's transcription
of it is faithful — `pbr_neutral_tonemap` in
`algan/rendering/raytracing/raytrace_kernels_taichi.py:1855` matches the
reference term for term, including `startCompression = 0.76` (the reference's
`0.8 - 0.04`). This is not a transcription bug. Two properties of that
reference curve produce the three errors above:

* **The pedestal.** The reference computes
  `offset = x < 0.08 ? x - 6.25 * x * x : 0.04` on the *darkest* channel and
  subtracts it from all three. For any colour whose darkest channel is at or
  above 0.08 that is a flat `−0.04`, which is `−10.2/255`. That is error 1
  exactly, and it is also the `+4` lift on the zero channels of a primary:
  a pure red has `x = 0`, so `offset = 0`, and the `−10` never happens —
  instead the desaturation term at the end mixes the channels toward the peak.
* **Compression starting below display white.** The `0.76` threshold is
  compared against the peak *after* the pedestal, so in input terms
  compression begins at `v = 0.80` and everything from there to infinity is
  squeezed into `[0.76, 1.0)`. An input of 1.0 leaves at 0.869 — error 2. The
  output never actually reaches 1.0 for any finite input, which is why no
  authored colour can render as 255 with the curve on.

The mismatch is that Khronos PBR Neutral is designed for **scene-linear**
input, where 1.0 is a reference exposure level rather than display white, and
is normally followed by an sRGB transfer function. Algan has neither: there is
no sRGB↔linear conversion anywhere in the colour path, so an authored
`RED = (255, 0, 0)` is a *display-referred* value that reaches the curve as
1.0 with nothing applied afterwards. Under that mismatch the curve's
headroom-reserving behaviour lands raw on the output byte.

## 4. Why "only affect HDR values" cannot be taken literally

It is worth stating the constraint plainly, because it decides the fix.

A tonemap is a monotone map `f` onto `[0, 1]`. If `f` is the identity on
`[0, 1]` then `f(1) = 1`, and since `f(x) ≤ 1` for all `x`, every input above
1.0 must also map to 1.0. That is a clamp. So:

> **Preserving the display range exactly and preserving detail above it are
> mutually exclusive.** Display white and HDR headroom compete for the same
> top byte, and you can have exactly one.

Any curve that keeps `2.0` and `4.0` distinguishable *must* move something
inside `[0, 1]` to make room. The question is not whether the default tonemap
should touch SDR values — it is whether the HDR detail it buys is worth what
it costs, and where the cost should fall.

## 5. How much HDR is there, actually?

`benchmarks/_tonemap_hdr_occupancy.py` answers that by measurement rather than
by argument. It intercepts the linear-HDR frame at the point the post stage
hands it to the tonemap and histograms it, over the repo's six dense
full-render scenes — including `materials_and_lighting`, the only one with
glow (up to 2.5), refraction and shadows.

Over all six scenes, 1117 frames:

| bucket | share of colour channels |
| --- | ---: |
| exactly 0 (background) | 0.03% |
| `0 < x <= 1.0` — already inside the display range | **98.52%** |
| `x > 1.0` — what the tonemap exists for | **1.45%** |
| `x > 1.05` — over by more than quantization noise | 1.37% |

    channels the tonemap exists for (> 1.0)   : 13,554,843
    channels it alters anyway (0 < x <= 1.0)  : 920,335,589
    ratio                                     : 1 : 68

**For every channel the default tonemap exists to rescue, it puts a colour
error on 68 that were already correct.** The peak value seen anywhere across
the six scenes was 3625.0, so real HDR does exist — it is just very rare, and
concentrated in the glow and specular highlights of one scene.

(`text_and_media`, the scene most like typical Algan output — text and flat
2-D — was 99.59% in-range and 0.40% above 1.0. The per-scene rows for the
other five were lost to a stdout interleaving in the recorded run; the totals
above are the run's own and are self-consistent. Re-run the script for a full
per-scene table.)

### The background is tonemapped too

Not only geometry. With no geometry in the frame at all, the background colour
takes the same shift:

| background | authored | tonemap ON | tonemap OFF |
| --- | ---: | ---: | ---: |
| grey | (128, 128, 128) | (118, 118, 118) | (128, 128, 128) |
| navy | (16, 24, 64) | (6, 14, 54) | (16, 24, 64) |
| white | (255, 255, 255) | (222, 222, 222) | (255, 255, 255) |

## 6. What can be done

There is no fix that makes the curve "HDR only" while keeping it a curve --
§4 rules that out. What is available is a choice about where the cost falls.

### Option A -- default `tonemapping=False` **(chosen, and implemented)**

Authored colour lands on the pixel it names: white is white, a primary stays
primary, mid-tones are exact. Values above 1.0 clip.

The reason clipping is acceptable *here specifically* is that *bloom already
runs before the tonemap, on the unclamped linear-HDR buffer*. Over-range energy
has therefore already been spread into a visible halo by the time anything
clamps, and a clipped core plus a halo is how a bright source reads anyway.
Both halves of that are checked rather than assumed: bloom is in the default
post-process tuple (`post_processes=(bloom_filter,)`, `render_loop.py:2465`,
`:2508`, `:3058`), and the ordering is deliberate -- "this is the
physically-correct order: bloom/glow" before the curve
(`raytracing/settings.py:67`).
The tonemap's marginal contribution is the roll-off detail inside the core --
paid for with a colour error on every other pixel in the frame.

The occupancy measurement is what makes this concrete rather than a matter of
taste: the curve is spending a colour error on 68 already-correct channels for
every one it rescues.

This is also what the repo has already concluded once, in a narrower context:
`Scene.use_manim_defaults()` turns tonemapping off, and
`tests/unit_tests/test_manim_defaults.py:91` pins it, with the comment "Algan's
tonemap darkens every fill by about 10/255, which reads as a colour error
rather than a roll-off". Option A generalises that finding from Manim parity to
the default.

*Cost*: every pixel baseline moves -- `tests/fast` and `tests/full_renders`,
and each has a separate committed CPU and CUDA set. The CUDA set cannot be
regenerated in a cloud session.

**This is the option taken.** By explicit decision, *neither* baseline set is
regenerated here: both are left stale, and the suites are expected to fail
until someone on a CUDA machine regenerates the CPU and CUDA sets together. A
half-regenerated pair (CPU fresh, CUDA stale) was judged worse than a clean
pair of stale ones. `tests/fast` and all six `tests/full_renders`
scenes are the suites affected; `tests/unit_tests` is not, since nothing
there compares pixels under the default tonemap.

Route-neutral, which is what makes it a low-risk change: `_get_tonemap_t_val()`
already returns 3 whenever `post_process_tonemap` is on (the default), so the
render kernels never consulted `TONEMAPPING` in the first place. Only the post
stage's `method_id` changes, from 1 (neutral curve) to 0 (clamp). The sheet
resolve serves "non-default tonemaps" either way
(`raytracing/settings.py:927`), so no batch changes route.

### Option B -- keep the curve on, drop the pedestal *(not taken)*

Remove the `offset` subtraction and move `startCompression` back to 0.8. The
curve becomes the identity on `[0, 0.8]`, which removes the flat −10 on darks
and mid-tones. White still lands at 229 rather than 255, and a primary still
desaturates, because those come from the compression and the desaturation mix
rather than from the pedestal.

*Cost*: same baseline churn as A, and `tonemap_method="neutral"` would no
longer be Khronos PBR Neutral, so the name would have to change. Fixes one of
the three errors.

### Option C -- tonemap lit geometry only *(not taken)*

Flat unlit fills (2-D shapes, text) bypass the curve; lit surfaces keep it.
This is how a game renders its HUD -- the 3-D scene is tonemapped, the UI is
composited afterwards -- and it fits an engine whose frames are mostly flat
diagrams and text with 3-D solids as the accent. Text and diagrams would land
exactly on the authored colour while genuine HDR still rolls off.

*Cost*: much larger change (a per-primitive flag through the material block,
like `_MAT_ONE_SIDED`), and it introduces a discontinuity: a flat fill and a
lit surface authored the same colour would no longer match.

## 7. A separate bug found on the way: AgX maps grey to pink

The alternative curve, `tonemap_method="agx"`, is **broken outright** — not
aggressive, wrong. The same nine authored fills under it:

| authored | agx renders |
| ---: | ---: |
| white (255, 255, 255) | **(255, 89, 208)** |
| grey75 (191, 191, 191) | (255, 84, 197) |
| grey50 (128, 128, 128) | **(255, 77, 180)** |
| grey25 (64, 64, 64) | (215, 63, 147) |
| green (0, 255, 0) | (235, 117, 115) |

Neutral grey comes out saturated magenta, and grey50 comes out *brighter* than
it was authored. No tonemapper does this.

**Cause: the final colour-space matrix is applied transposed.** `agx_tonemap`
ends by converting linear Rec.2020 to linear Rec.709/sRGB
(`raytrace_kernels_taichi.py:1847-1849`, and the same numbers again in
`post_process.py:143-148`). A conversion between two spaces sharing a white
point must map white to white, so each row must sum to 1. As applied, the rows
sum to **1.5177 / 0.4447 / 1.0376** — a fixed gain of +52% red, −56% green,
+4% blue on any neutral, which is exactly the magenta cast measured. The
transpose sums to 1.0001 / 1.0000 / 0.9999 and is the correct matrix.

This was confirmed rather than inferred: an independent Python implementation
of the whole AgX chain using the matrix *as written* reproduces the rendered
bytes **exactly** for every colour tested — `(255,89,208)`, `(255,77,180)`,
`(215,63,147)`, `(235,117,115)`, `(255,5,77)`. With the matrix transposed, greys
stay neutral (255 → 201, 128 → 173, 64 → 142) and colours keep their hue.

Nothing in `tests/` uses AgX, so fixing it moves no committed baseline. The
other three matrices in the function (Rec.709→Rec.2020 in, and the AgX inset
and outset) all have rows summing to 1.0 and are correct.

## 8. A second, independent defect

`tonemap_exposure` is **silently ignored whenever `tonemapping=False`**.
Measured on both implementations:

    tonemapping=True   exposure=0.5 -> [22, 54, 85]
    tonemapping=True   exposure=1.0 -> [54, 117, 181]
    tonemapping=True   exposure=2.0 -> [117, 222, 239]
    tonemapping=False  exposure=0.5 -> [64, 128, 191]
    tonemapping=False  exposure=1.0 -> [64, 128, 191]
    tonemapping=False  exposure=2.0 -> [64, 128, 191]

Both the Taichi kernel (`tonemap_kernels_taichi.py:48`, the `method == 0` arm)
and the torch path (`post_process.py:242`) clamp without multiplying by
exposure. The documentation advertises it without that caveat:
"``tonemap_exposure`` is the right control for 'the whole scene is too dark' --
reach for it before you start raising every light's intensity"
(`docs/source/advanced_user_tutorials/backgrounds_and_post_processing.rst:222`).

This matters most under Option A: the moment the default curve is off, the
documented brightness control stops working. It should be fixed either way, and
it is a small, self-contained change that moves no baseline while the default
stays as it is.

Related, smaller: `set_tonemap_exposure`'s docstring
(`algan/rendering/raytracing/settings.py:2068`) still says "the ACES Filmic
Tonemapper". There is no ACES curve in the codebase --- `set_tonemapping`'s own
docstring already corrects this ("not ACES, whatever the old docstring said").


## 9. What turning it off costs, measured after the fact

The flip was decided on §5's occupancy figures. Rendering `tests/fast/scene.py`
both ways afterwards showed a cost those figures do not capture, and it is
worth stating plainly because it is visible rather than subtle.

Compare `algan_outputs/tonemap_check/fast_f30_curve_{on,off}.png`:

* **The flat content gets better, exactly as intended.** The title text and the
  white circle border render white instead of grey; the 2-D fills land on their
  authored colours.
* **The lit 3-D solids blow out.** The orange cube's front face becomes a flat
  yellow-white and the purple cube's becomes near-white. Peak difference
  between the two renders is 125 channel values, at a pixel over range in two
  channels: the curve gives `(242, 130, 64)`, clamping gives `(255, 255, 95)`.
  2.51% of that frame's pixels differ by more than 33.

The mechanism matters, because it is not really about the tonemap. The curve
divides *all three* channels by `peak / newPeak`, so when one channel is over
range the others come down with it and the colour keeps its hue. A clamp
truncates each channel independently, so an over-range saturated colour loses
its hue and slides toward white. That is why the blowout reads as a different
colour rather than merely a brighter one.

And the reason anything is over range at all: **the default light intensities
were implicitly calibrated with the tonemap in the loop.** The fast scene peaks
at 2.397 linear, with 0.55% of channels above 1.0 and 0.11% above 2.0 -- lit
surfaces were relying on the curve to compress an overshoot the lighting
creates. Removing the curve does not create that overshoot, it exposes it.

So the change as it stands is right for flat 2-D and text -- which is most of
what Algan renders, and what prompted the investigation -- and is a regression
for lit 3-D. Three ways to close that gap, in increasing order of effort:

1. **Re-tune the default light intensities** so a fully lit diffuse surface
   lands at or just below 1.0. This is the honest fix: it makes the lighting
   independent of the encoder, which it should always have been. It moves every
   baseline again, so it wants doing in the same pass as the re-baselining.
2. **Ship a default `tonemap_exposure` below 1.0.** Cheap, but it darkens the
   flat content too, giving back some of what the flip just won.
3. **Option C from §6** -- tonemap lit geometry only, leave flat fills alone.
   The only one that gets both, and much the largest change.

**Followed up in §11**, and none of the three turned out to be the fix.

## 10. Baseline state

Deliberately not regenerated, per §6. For whoever picks that up:

* `tests/fast` on this cloud container fails at **115 channel values, frame
  32** (tolerance 2).
* There is a **pre-existing failure of 32 channel values at frame 6** on this
  container, reproduced on a completely clean tree with every change of this
  work stashed. It is not caused by anything here -- an earlier session
  recorded the same frame at 40 -- and it means this container's CPU baseline
  was not generated on hardware matching it. Do not read the 115 as purely the
  tonemap: 32 of it was already there.
* `tests/unit_tests` is unaffected: 1541 passed, 93 skipped, 0 failed.
* `tests/full_renders` skips itself under `CI`, and was not run here.

## 11. Bounding the lit colour, and why it was not a light-intensity change

§9's follow-up was "re-tune the default light intensities". Measuring first
showed that premise was wrong in two ways, and the real fix is neither of the
three options listed there.

`benchmarks/_light_ldr_probe.py` renders a white Lambert cube and a white flat
fill, adding one light at a time, and reports the peak linear value the post
stage receives:

| lights on the scene | peak | % > 1.0 |
| --- | ---: | ---: |
| Algan's own default rig (one white `PointLight`) | **1.000** | 0.000% |
| + `AmbientLight(0.45)` | 1.325 | 5.13% |
| + `DirectionalLight(0.85)` | 1.744 | 5.42% |
| + `PointLight(0.6)` -- this is `tests/fast`'s rig | 2.154 | 5.42% |

**The default lighting was already exactly LDR.** One default light lands a
fully lit white surface on 1.000 and not a fraction over. There was no default
intensity that needed turning down.

**And there is no such knob anyway.** `SETTINGS.raytracing.light_intensity`
(the `LIGHT_INTENSITY = pi` at `raytracing/settings.py:94`) is refused by the
settings layer, which says so itself:

    'light_intensity' is not read by any renderer this build can launch (only
    by the unwired physical-mode Monte Carlo kernel), so setting it would
    silently do nothing. Scale a light with its own intensity= instead.

What actually breaks the invariant is that **light contributions accumulate
without normalisation**. Each light adds its diffuse, ambient and specular
terms to the running colour, so any scene with more than one light drives a
fully lit surface past 1.0 even when every individual light is at or below unit
intensity. Nothing bounded that: `_run_frag_pipeline`
(`raytracing/shading_taichi.py`) ended with its clamp **commented out** --
correct while the tonemap was compressing the overshoot, and a hole the moment
it stopped.

The fix bounds the shaded colour at that one point, and in the torch vertex
path's `_recombine` (`shaders/material_shaders.py`) to match:

    out = max(out, 0)
    if peak(out) > 1:  out /= peak(out)

Scaling by the peak rather than clamping per channel is the whole point. A
clamp truncates each channel independently, so an over-range orange
`(2.0, 1.0, 0.4)` becomes `(1.0, 1.0, 0.4)` -- a different, yellower colour.
Scaling gives `(1.0, 0.5, 0.2)`, the same colour at the brightness that fits.
That is what turns the blown yellow-white cube face of §9 back into an orange
one.

It is **the identity below 1.0**, so everything already in range is
bit-identical and only pixels that were going to clip anyway move. And `glow`
is returned untouched, so it stays the one route to above-1.0 output.

Measured after the change, with the probe unchanged:

| lights on the scene | peak | % > 1.0 |
| --- | ---: | ---: |
| default rig | 1.000 | 0.000% |
| + Ambient | **1.000** | 0.000% |
| + Ambient + Directional | **1.000** | 0.000% |
| + Ambient + Directional + Point | **1.000** | 0.000% |

and glow still does what it is for:

| | peak |
| --- | ---: |
| `glow = 0.0` | 1.000 |
| `glow = 0.5` | 5.593 |
| `glow = 1.5` | 124.999 |

So the invariant holds exactly as asked: **a surface with `glow == 0` never
exceeds the display range however many lights are on it, and only `glow > 0`
produces HDR.**

`algan_outputs/tonemap_check/fast_f30_ldr_bound.png` is the same frame as §9's
pair. The flat content keeps everything the tonemap flip won -- white title,
white borders -- and the lit solids keep their hue.

### Checked: secondary rays

`OX_LIGHTING_AUDIT.md` is the companion code map for this section, and it
raised the one caveat worth testing: the bound sits on each *shading event*,
but a pixel carrying reflection or refraction composites several of them, so
the sum could in principle exceed 1.0 even though every term is in range.

Measured, on a mirror-metal sphere and a transmissive one over a white
backdrop under the same three-light rig: **peak 1.000, 0.000% over**. The
composite is a weighted blend rather than an unweighted sum, so bounded events
stay bounded through it. That is one scene rather than a proof, but the
specific concern does not reproduce.

That audit also independently confirms three things this section asserts: that
`light_intensity` and `ambient_light` reach only a kernel no renderer launches
(and are already declared inert, pinned by
`tests/unit_tests/test_inert_settings.py`), that the live paths hard-code
`light_intensity == ambient == 1` at `primitives.py:620-621`, and that the two
shading paths genuinely disagreed before this change -- the torch side
truncating per channel at every lit vertex while the kernel side never bounded
at all.

### What this does not do

It bounds the result, it does not make the lighting energy-conserving. A three-light rig still saturates a white surface
where a physically-based one would not; it now saturates without changing hue.
Normalising the light accumulation itself would be the deeper fix and would
change every multi-light scene's shading, not just its clipped pixels.
