# Linear working colour space: diagnosis, design, and what to check

Working document for the change that gives Algan a linear working space.
`TONEMAP_FINDINGS.md` is the history that leads here — read its §11 and §12
first, because this change removes both of the compensations they added, and
says why.

## 1. The diagnosis, measured

Algan has no sRGB↔linear conversion anywhere. Authored colours are
display-referred, every shading and compositing operation runs on those encoded
numbers, and the float→byte write is a bare `clamp(c) * 255`
(`post_processing/tonemap_kernels_taichi.py:53-56`).

`benchmarks/_linear_color_check.py` measures which space the shading arithmetic
actually happens in, rather than arguing it from the source. A white Lambert
cube face-on under one directional light, sweeping the light's intensity:

| intensity | byte | byte/255 | srgb_to_linear(byte/255) |
| --------: | ---: | -------: | -----------------------: |
|      0.10 |   51 |   0.2000 |                   0.0331 |
|      0.20 |   77 |   0.3020 |                   0.0742 |
|      0.30 |  102 |   0.4000 |                   0.1329 |
|      0.40 |  128 |   0.5020 |                   0.2159 |
|      0.50 |  153 |   0.6000 |                   0.3185 |
|      0.60 |  179 |   0.7020 |                   0.4508 |
|      0.70 |  204 |   0.8000 |                   0.6038 |
|      0.80 |  230 |   0.9020 |                   0.7913 |
|      0.90 |  255 |   1.0000 |                   1.0000 |

Fit a line to each candidate model:

* **gamma**: `byte/255 = 0.1009 + 1.0000 * i` — max residual **0.0011**
* **linear**: `srgb_to_linear(byte/255) = -0.1974 + 1.1993 * i` — max residual 0.118

The gamma fit is essentially exact, and its intercept recovers
`AMBIENT_STRENGTH = 0.1` to three decimals. **The arithmetic is provably
display-referred.** That is the whole defect: reflected radiance is proportional
to intensity in the *encoded* space, where it should be proportional in linear
light.

The consequence for addition is the sharp one. sRGB encoding is concave, so
`encode(a) + encode(b) >> encode(a + b)`:

| | linear-correct | Algan's space |
| --- | ---: | ---: |
| one light putting white at byte 137 | 0.25 → **137** | 0.537 → **137** |
| a second identical light | 0.50 → **188** | 1.074 → **255**, clipped |

Summing in a gamma space overshoots by construction. That overshoot is what
`_energy_scale` (TONEMAP_FINDINGS §12) was introduced to normalise away.

## 2. Why the normalisation has to go with it

The same harness measures the response to rising total intensity. On the tree
before this change, at exposure 0.4:

    0.60 -> 71,  0.80 -> 92,  1.20 -> 102,  1.50 -> 102,  1.80 -> 102

**Past a total of 1.0, extra light changes nothing.** `_energy_scale`'s
`1 / max(budget, 1)` pins every over-unity rig to exactly albedo.

The harness is **deterministic**: two independent runs on this container
produced byte-identical tables on every row. That is what makes "the `off` arm
must reproduce the previous numbers exactly" a legitimate regression gate for
D2 rather than a tolerance judgement.

Note what this does to a naive test of additivity. "N lights at intensity i must
render as one light at N*i" holds *perfectly* here — split and single agreed in
every case above — because a rule that depends only on the total satisfies it
whether it sums or normalises. The invariant that catches the defect is
**monotonicity**: adding light must keep making the surface brighter. Both are
in the harness; only the second one has teeth.

## 3. The design

Decode authored colour at the render boundary, do all arithmetic in linear
light, apply the sRGB OETF at the final byte write. This is what three.js does:
`ColorManagement` decodes inputs into a `LinearSRGBColorSpace` working space,
`lights_fragment_begin.glsl.js` accumulates lights as a plain unnormalised sum,
and the shader tail is `tonemapping_fragment` then `colorspace_fragment` —
tonemap (default `NoToneMapping`), then the OETF, unconditionally.

* **D1** Exact piecewise sRGB, not gamma 2.2. Torch pair in
  `algan/utils/color_space.py`, Taichi `@ti.func` pair beside the kernels.
  Clamp at 0 before the power in both directions — a linear value can go
  slightly negative through interpolation and `pow` would return NaN.
* **D2** `ALGAN_LINEAR_COLOR`, default on, declared in `algan/environment.py`'s
  `_IMPORT_TIME_VARIABLES`; module global + setter in
  `rendering/raytracing/settings.py`; surfaced as
  `SETTINGS.raytracing.linear_color_space`. **Off must reproduce the previous
  tree byte for byte** — every conversion is gated, and the harness's `off` arm
  is the regression gate.
* **D3** Decode at the render boundary, **not** in `Color` and not in the
  timeline. `Color` (`algan/constants/color.py`) is a `torch.Tensor` subclass
  that flows through the animation timeline, so decoding there would change what
  `mob.color` reads back and would make colour tweens interpolate in linear
  light. Authored colour stays display-referred until it is packed for the
  renderer. This is a deliberate deviation from three.js, whose `Color` stores
  linear and lerps there.
* **D4** Encode once, at the display transform:
  `exposure → tonemap (optional) → sRGB OETF → quantize`.
* **D5** Remove `_energy_scale` and the peak-scale bound at the tail of
  `_run_frag_pipeline` (plus their torch twins in `material_shaders.py`), behind
  the gate. Under the linear arm the lighting is plainly additive.
* **D6** FXAA must run *after* the OETF — canonical FXAA computes luma on
  sRGB-encoded RGB and its thresholds are tuned for that space. Bloom stays
  where it is; it is correct on linear HDR.

Decode colours only. glow, opacity/alpha, roughness, metalness, IOR, light
`intensity` and `AMBIENT_STRENGTH` are not colours.

## 4. Semantic changes that are correct but must be said out loud

* **`AMBIENT_STRENGTH` is 0.01 under the linear space, not 0.1.** *(Resolved:
  0.1 was a display-referred coefficient, and carrying it across unchanged made
  the fill nearly nine times brighter — 0.1 of linear light encodes to byte 89
  where 0.1 of an encoded value is byte 26. `srgb_to_linear(0.1) = 0.01003`, so
  0.01 delivers the same fill: measured, a fully shadowed surface reads byte 25
  under linear against 26 under display-referred. The number changed because
  the units changed, not because the look was retuned.)*
* **Mid-tones lift across the board**: a surface at half illumination goes from
  byte 128 to byte 188. Everything gets brighter, so *more* pixels sit near 1.0
  and clipping becomes more likely — the opposite of what "removing the
  normalisation" sounds like it should do.
* **Unlit flat 2-D content is unchanged.** Decode-then-encode with no arithmetic
  between is the identity, which is the acceptance gate. It also preserves the
  documented Manim-parity claim that a flat fill is byte-identical to Manim's;
  what diverges from Manim is antialiased edges (linear vs gamma compositing)
  and 3-D shading.
* **User post-processes now receive linear values**, SMAA included, and bloom's
  threshold semantics shift for the same reason.

## 5. Pre-existing state on this container, before any of this

Branch `claude/algan-light-accumulation-adyxt1` at `1a4c9d2`, `git status` clean:

* **`tests/fast` already fails: 118 channel values, worst at frame 30**
  (tolerance 2). The branch's earlier commits moved output and deliberately did
  not regenerate the committed CPU baseline — TONEMAP_FINDINGS §10 records that
  choice. The baseline was already stale before this work started, so the fast
  suite's delta after this change cannot be read as this change's alone.
* `tests/unit_tests` was reported at 1541 passed / 93 skipped / 0 failed in §10.

## 6. What landed, and what it measured

All three acceptance invariants pass (`benchmarks/_linear_color_check.py`,
one process per arm):

| | before | after |
| --- | --- | --- |
| which space the shading happens in | gamma (residual 0.0011) | **linear** (residual 0.0029) |
| authored flat fill round-trip | exact | **exact** |
| response to total intensity 1.2 / 1.5 / 1.8 | 102 / 102 / 102 | **191 / 209 / 226** |

The middle row is the acceptance gate and the bottom row is the point of the
exercise: lights add.

Conversions landed at these sites, all gated on `linear_color_space`:

* `render_loop.py` — light colour, decoded **upstream of alpha, opacity and
  intensity**, which are linear scalars. Decoding at `_pack_lights` (the
  obvious choke point) would have been wrong, because the colour is
  intensity-premultiplied there and `srgb_to_linear(c * i) != srgb_to_linear(c) * i`.
* `lights.py` — a hemisphere's ground colour, before its intensity multiply.
* `scene_builder.py` — `tri_colors` / `circuit_colors` / `circuit_border_colors`
  once per batch, and the background at the prefill.
* `tonemap_kernels_taichi.py` + `post_process.py` — the OETF, last, after
  exposure and after any curve. The transparent route unpremultiplies before
  encoding and re-premultiplies after, because encoding a premultiplied value
  would store `encode(0.5) = 188` where the correct answer is
  `0.5 * encode(1.0) = 127`.
* `shading_taichi.py` + `material_shaders.py` — the illumination budget and the
  peak bound are off, and `AMBIENT_STRENGTH` becomes 0.01.
* `tracer.py` — refuses `linear_color_space` with `post_process_tonemap` off,
  because that route's frame buffer is uint8 and linear 0.033 quantises to
  byte 8.

### The trap that cost the most time, twice

**The shading kernels resolve their gate with `ti.static`, which Taichi
evaluates when it compiles the kernel and then caches.** A second arm in the
same process silently reuses the first arm's code and reports its numbers as
its own. This caught the acceptance harness (its two-arm output was
untrustworthy for anything the shading stages touch) and then caught the shadow
probe, where the ambient change appeared to do nothing at all — the floor sat
at byte 89, which is exactly `encode(0.1)`, because the kernel was still the
display arm's. Clearing the Taichi offline cache does not help; it was never
the cause.

**One process per arm is the only reliable way to measure anything these
kernels gate.** Both harnesses now enforce it, and `ALGAN_LINEAR_COLOR` is in
`_IMPORT_TIME_VARIABLES` for the same reason.

### What it looks like

`algan_outputs/fast_ab/` has the frames. Flat 2-D is pixel-identical. The lit
solids in `tests/fast` blow out — an albedo-0.5 surface reaches 255 under that
scene's 1.9 of total light — and hue slides toward white on clipped pixels
because the peak bound that preserved it is gone, which is the price of lights
adding. The fixture's rig was authored against a normalising renderer and is
what wants retuning; it deliberately has not been, so the baseline keeps
showing it.

## 7. Review checklist

Traps specific to this change, written before reading any diff:

1. **Double decode** — a colour decoded at the primitive pack *and* again at the
   material-parameter pack, or at both the background prefill and the composite.
2. **Decoding a non-colour** — grep every call to the decode helper and name what
   each argument is.
3. **Encoding premultiplied colour** — the composite writes
   `csum[ci] = rs_acc[r,ci]*255 + weight[ci]*bg` alongside a separate alpha. If
   that RGB is premultiplied, the OETF must not be applied to it as-is. Check the
   transparent-background / alpha-channel video route specifically.
4. **f16** — `ALGAN_HDR_BUFFER_F16=1` makes the torch buffer float16, and
   `pow(x, 1/2.4)` in f16 loses real precision. Compute the OETF in f32
   regardless of buffer dtype.
5. **Negative input to `pow`** — NaN in both directions without a clamp.
6. **A route left un-encoded** — the sheet route, the classic supersampled
   wavefront and the Monte Carlo (SPP>1) route may not agree on scale. A missed
   route renders that path twice as dark.
7. **A missed torch/Taichi twin** — five kernel stages against five torch twins;
   `tonemap_to_u8` against `_finalize_on_device`; `_run_frag_pipeline`'s bound
   against `_recombine`.
8. **The gate is not total** — anything unconditional breaks D2.
9. **`ti.static` gating** — a runtime `if` where a compile-time gate belongs;
   and per CLAUDE.md, never collapse `if ti.static(gate): if cond:` into one
   `and`.
10. **Shadowed locals / wrong-arg plumbing** — read every changed signature and
    call site, not just the logic. This is the defect class that has twice got
    past an adversarial re-read on this repo.

## 8. What the checklist missed: two ingest routes, found by re-running the audit

Item 6 above says "a route left un-**encoded**". The defect that shipped was a
route left un-**decoded**, and there were two of them. Found on 2026-08-22 by
re-running `benchmarks/renderer_audit`, not by any test:

* **Colour texture maps.** Constant-property promotion (`PROMOTE_CONSTANTS`, on
  by default) renders a mob whose colour and material are uniform from a shared
  1×1 colour map in `scene["textures"]` instead of per-vertex `tri_colors`. §6's
  list names `tri_colors` / `circuit_colors` / `circuit_border_colors`; the
  promoted map is none of them, and neither is a real `ImageMob` texture. So
  **most** geometry — every plainly-coloured 3-D mob — reached the kernel
  display-referred and was then encoded on the way out.
* **The material parameter block's colour slots** — `emissive`, `specular`,
  `specular_color`, `sheen_color`. Same shape of miss: the block is not a colour
  array, so it was not on the list.

Measured: an unlit slab authored 0.5 grey rendered **188** (`encode(0.5)`), and
`ALGAN_PROMOTE_CONSTANTS=0` rendered the same slab at **128**, which is the A/B
that names the cause without any instrumentation. An authored emissive of
(0.5, 0.25, 0.75) rendered (188, 137, 225) instead of (128, 64, 191).

### Why the acceptance gate passed anyway

§6's table says "authored flat fill round-trip: **exact**", and it was — for the
thing it measured. `_flat_fill` builds a `Square`, which is a bezier circuit, and
`circuit_colors` was on the decode list. The gate tested one of three ingest
routes and generalised to all of them.

The harness now runs the round trip through **two** routes, a circuit and a
triangle mesh (`_flat_mesh_fill`), and `tests/unit_tests/test_color_decode_boundary.py`
pins the rendered round trip for both a promoted uniform colour and an emissive.
Both fail on the pre-fix tree with exactly the numbers above, which is the check
that the guard is real.

**The general lesson, worth carrying to the next pipeline change:** an invariant
of the form "X survives the pipeline" is only as strong as the number of ways X
can *enter* the pipeline, and that count is a property of the engine's plumbing
rather than of the invariant. Enumerate the ingest routes first, then write one
case per route. A single case passing tells you one route works, and says
nothing whatever about the others.
