# Read-only audit: texture and material-parameter transport in the renderer

Scope: structural (memory-traffic / dedup / upload-cost) audit of texture and
per-corner material transport on the default deterministic path (samples <= 1,
HYBRID_RASTER + ANALYTIC_AA + SHEET_RESOLVE + ANALYTIC_AA_RUN all default on,
`algan/rendering/raytracing/settings.py:870,1103,1369,1488`). All statements
are from source; no renders were run, nothing was modified. Line numbers refer
to the working tree as of this audit.

---

## Claim verdicts

### Claim 1 — "A texture image used by N primitives across T frames is stored once per batch in the merged upload" — **REFUTED**

It is stored once per (textured primitive × frame), i.e. N×T times, not once.

* The merge appends one texel block per textured collection:
  `for p in textured_triangles: color_meta = _append_texture(p._rt_texture_map,
  is_color=True)` (`algan/rendering/raytracing/scene_builder.py:1593-1594`,
  plus material at :1596 and normal maps at :1597). There is no content or
  identity dedup anywhere in `_append_texture`
  (`scene_builder.py:1327-1367`).
* Every textured primitive is its own singleton collection — "Textured
  primitives are batched one per collection: a collection carries a single
  texture map set … Their geometry is still merged into one kernel launch
  downstream" (`algan/render_loop.py:2329-2338`); inside
  `TrianglePrimitive.__init__` the first member's `texture_map` simply wins
  (`algan/rendering/primitives/triangle_primitive.py:147-157`). So two Mobs
  sharing an image are two collections and two appends.
* Within one primitive's map the time axis is dense: `_append_texture` stores
  `tex.reshape(tex.shape[0], -1, 5)` — all T frame slices
  (`scene_builder.py:1364`) — and `scene["textures"]` is never passed through
  `_dedup_time`: the `MERGE_DEDUP_TIME` collapse lists only
  `tri_norm/tri_mat_id/tri_mat/tri_colors/tri_extra/tri_uvs`
  (`scene_builder.py:1639-1648`) and the circuit tables
  (`scene_builder.py:1829-1836`); the assembly at
  `scene_builder.py:1721-1722` does not dedup.

**What is duplicated and by what factor:** a W×H image carried by N textured
collections over a T-frame batch occupies N·T·W·H·5 f32 texels
(= 20·N·T·W·H bytes) in `scene["textures"]`, versus the 20·W·H bytes of unique
information for a static image — factor N·T (T alone if one primitive). The
one-per-batch part that *is* true: the merged scene is built and uploaded once
per batch (`_rt_merged_scene` cache, `scene_builder.py:1250-1253`; arena upload
`algan/rendering/raytracing/tracer.py:1146-1165`).

Side note: the *bezier* texture transport (circuit fill/border color grids)
does have the time dedup that ImageMob textures lack —
`circuit_colors`/`circuit_border_colors` go through `_dedup_time`
(`scene_builder.py:1829-1836`).

### Claim 2 — "Two Mobs built from the same image file/tensor share one texture in the merge" — **REFUTED**

No dedup exists; the key is **none**.

* `get_image` re-reads and re-decodes the file on every call with no cache
  (`algan/utils/file_utils.py:44-54`), so even the source tensors are distinct.
* Each Surface registers its texture as its own animatable attribute row and
  builds its own primitive carrying its own premultiplied copy
  (`algan/mobs/surfaces/surface.py:1389-1438`, `:3114-3139`);
  `_stash_texture_maps` then makes `.float().contiguous()` per primitive
  (`algan/rendering/raytracing/primitives.py:1094-1098`).
* The merge appends per collection with no sharing (Claim 1). The only
  value-based texture sharing in the builder is the constant-promotion path,
  which groups plain triangles' *1×1* colour/material maps by value with
  `torch.unique` over corner-0 colour + material across all frames
  (`scene_builder.py:589-634`), and the legacy `WF_TEXTURED` promoter's
  identical grouping (`scene_builder.py:1085-1110`). Neither path applies to
  image textures; a textured triangle always lands in `textured_triangles`
  (`scene_builder.py:1393-1398`) and is never promoted.

### Claim 3 — "Texture texels are uploaded and sampled as f32; no u8/f16 texture storage exists on the default path" — **CONFIRMED**

Every stage converts to or preserves float32:

* Load: `torchvision.io.read_image(...).float() / 255`
  (`file_utils.py:49-53`); manim path also `/255` after `.float()`
  (`algan/mobs/image_mob.py:69-72`); padded to 5 channels in f32
  (`algan/constants/color.py:240-245`).
* Timeline storage buffers are f32 (`row_bytes = channels * 4`,
  `algan/animation_timeline/timeline.py:786`, buffer alloc `:812-814`).
* Pack: `.float().contiguous()` (`primitives.py:1094-1098`); merge decode
  `srgb_to_linear(tex[..., :3].float())` + same-dtype zero pad to 5 channels
  (`scene_builder.py:1356-1362`).
* Sampling reads raw f32 values out of the ndarray:
  `textures[tc, abs_idx, ci]` (`wavefront_kernels_taichi.py:321-322`,
  `raytrace_kernels_taichi.py:1553-1557`).

The only uint8/float16 in the renderer are unrelated: output framebuffer /
tonemapped writes (`tracer.py:1578`, `wavefront_kernels_taichi.py:1688`, the
opt-in frame-buffer dtype at `settings.py:2687-2713`), masks and arena byte
views (`scene_builder.py:371-372`), BVH kind tags and compressed BVH blocks
(`refit_bvh.py:506`). No u8/f16 *texture* format exists on any path.

### Claim 4 — "An animated texture is stored one image per frame of the batch, with no dedup of identical consecutive frames" — **CONFIRMED**

* The attribute timeline materializes state densely over the batch window:
  `active_state = generate_array_states(times, ...)` shaped `[T, rows, D]`
  (`timeline.py:1543-1549`; reader returns `[t-slice, columns]`,
  `timeline.py:1019-1056`). One row of the colour-texture attribute is one
  whole image, so the window is one image per frame ("a whole image per frame
  per row", `timeline.py:754-755`).
* The primitive build reshapes that window to `[T, H, W, 5]`
  (`surface.py:3120-3139`) and the merge stores all T slices
  (`primitives.py:1094-1098`, `scene_builder.py:1364`).
* No consecutive-frame or all-batch dedup touches `scene["textures"]`
  (Claim 1); kernels index time as `f % textures.shape[0]`
  (`raytrace_kernels_taichi.py:1537`, `wavefront_kernels_taichi.py:307`),
  which is exactly why a collapsed axis would still read correctly — the
  mechanism used for the tri/circuit tables is simply not applied here.

Refinement worth more than the claim itself: this holds for **static**
textures too — a texture assigned once and never animated still materializes
as T identical images and ships T copies per batch. And the windows are freed
only after primitives are built (`release_wide_windows`,
`render_loop.py:2270-2274`, `timeline.py:2955-2967`), so the T-image window
and the primitive's premultiplied copy coexist through batch prep.

### Claim 5 — "Per-corner attributes are carried per (frame, corner) even when constant over the batch's frames — the merged time axis is dense for them" — **REFUTED as stated (true pre-merge and conditionally; collapsed on the default path)**

The arrays exist exactly as described, per (frame, prim, corner), all float32:

| array | shape | dtype | width | packed at |
| --- | --- | --- | --- | --- |
| `tri_colors` | `[T, N, 3, 5]` | f32 | RGBA+glow, 5 | `primitives.py:1141` |
| `tri_extra` | `[T, N, 15]` | f32 | `_EXTRA_W=15` (cols 0-5 refl/rough interleaved per corner, 6-8 IOR, 9-11 transmission, 12-14 sigma) | `primitives.py:1140`, layout `raytrace_kernels_taichi.py:375-378`, pack `primitives.py:974-1004` |
| `tri_norm` | `[T, N, 9]` | f32 | 3 corners × vec3 | `primitives.py:1137-1139` |
| `tri_mat` | `[Tm, N, MAT_W]` | f32 | `MAT_W=34` (`shading_taichi.py:78`); per-primitive, narrower blocks zero-padded by `_cat_mat_blocks` (`scene_builder.py` import, `utils.py:63-83`) | `primitives.py:807-835+` |
| `tri_mat_id` | `[1, N]` | int32 | 1 | `primitives.py:833-835` |
| `tri_uvs` | `[Tu, N, 6]` | f32 | 3 corners × uv2 | `surface.py:3102-3106`, `primitives.py:1200` |

But at merge time `MERGE_DEDUP_TIME` — **default ON**
(`settings.py:481-490`) — collapses every table whose values are identical
across all batch frames to a single frame, via `_dedup_time`
(`scene_builder.py:525-533`, applied at `:1639-1648`):

```python
if x.shape[0] > 1 and bool((x == x[:1]).all()):
    return x[:1].contiguous()
```

So on the default path a temporally-constant table is stored **once**, not T
times ("a batch whose materials/normals/colours do not animate stores one row
instead of T -- tri_mat alone is [T, N, 26], tens of MB of identical frames",
`scene_builder.py:1632-1638` — width now 34). Two conditions restore the dense
axis the claim describes:

1. `ALGAN_MERGE_DEDUP_TIME=0`, or
2. **any** primitive in the batch animates that particular table — the test is
   all-or-nothing per table, so one animating mob keeps the whole table
   (every static mob included) dense-T.

Also: `tri_pos` is deliberately never collapsed (rigid motion lives there,
`scene_builder.py:1636-1638`), and the collapse does not apply to textures
(Claim 4) or to the per-primitive pre-merge arrays, which are dense-T from the
materialized window regardless.

### Claim 6 — "`_sample_tex_vec5` bilinear-samples with no LOD, sheet resolve shades one dominant fragment per sheet → minified texture ≈ point-sampled per sheet; the sheet record carries exact covered area usable for mip selection" — **CONFIRMED**

Call path (default route, `sheet_resolve_shade` launched from
`raster_pipeline.py:2184,2298,2321`):

1. One shade per sheet at the dominant fragment: module docstring "shades ONCE
   per sheet at its dominant fragment" (`sheet_resolve_taichi.py:3-6`); the
   dominant fragment is chosen host-side as "largest exact area, first on
   ties" and stored in `sheet_ref`/`sheet_ab` (`sheets.py:878-880`), read at
   `sheet_resolve_taichi.py:274-277`.
2. Colour fetch: `_tri_color_g(0, f, prim, w0, a, b, tri_colors, col_row,
   tri_uvs, tri_tex_meta, textures, num_colored_triangles)`
   (`sheet_resolve_taichi.py:427-429`) → baseline `_flat_triangle_color`
   (`raytrace_kernels_taichi.py:1566`, imported at
   `wavefront_kernels_taichi.py:52`) or mem-trim `_flat_triangle_color_trim`
   (`wavefront_kernels_taichi.py:510`).
3. Sampler: `_sample_texture` (`raytrace_kernels_taichi.py:1517-1562`) /
   `_sample_tex_vec5` (`wavefront_kernels_taichi.py:287-325`) — both a fixed
   4-tap bilinear at `u*(W-1), v*(H-1)` with clamped indices. Neither takes
   derivatives, ray differentials, footprint, or a level argument; the texel
   grid is always full resolution. This is exactly RENDERER_WORK_QUEUE.md item
   4 ("plain bilinear tap with no level of detail … the region is shaded at
   its dominant fragment and whatever texel that lands on wins the pixel",
   `RENDERER_WORK_QUEUE.md:190-199`).
4. Sheet record fields: `sheet_key, sheet_ref, sheet_ab, sheet_cov, sheet_msk,
   sheet_cap` (`sheet_resolve_taichi.py:113-116`), where **`sheet_cov` is the
   sheet's exact area**: "float64 sum of its fragments' `frag_cov`, rounded to
   float32. NOT clamped to 1" (`sheets.py:881-885`), accumulated via a
   float64 `scatter_add_` (`sheets.py:409-448`), consumed in-kernel as
   `cov = ti.abs(sheet_cov[idx]); area = ti.min(cov, 1.0)`
   (`sheet_resolve_taichi.py:278-280,314`).

So yes: an exact screen-space covered area already rides in every sheet
record, and RENDERER_WORK_QUEUE.md itself notes it "is a screen-space
footprint the LOD could be derived from without derivatives"
(`RENDERER_WORK_QUEUE.md:210-213`). Caveat for whoever builds it: `sheet_cov`
measures *screen* coverage; converting it to a mip level also needs the
texture-space scale of the UV mapping (available per-hit from `tri_uvs` +
`tri_tex_meta` w/h, e.g. `wavefront_kernels_taichi.py:274-283,521-522`) —
the area alone suffices only up to that per-primitive factor.

---

## Question A — life of one texture (ImageMob colour map), every copy/move/conversion

Labels: **[construction]** once when the Mob is built; **[batch]** once per
frame-window batch (= the chunk unit of `render_loop`'s fetch/materialize
loop); **[frame]** once per frame of the batch (inside a [batch] step).

1. **[construction]** File decode: u8 tensor → `.float()/255` on the animation
   device (`file_utils.py:46-53`) — copy #1 + dtype conversion u8→f32.
   `Color.add_defaults` pads 4→5 channels (fresh cat, `color.py:240-245`);
   `ImageMob` transposes/flips to `[W,H,5]` contiguous (`image_mob.py:79`).
2. **[construction]** Authoring store: squashed into the Mob's animatable
   attribute `color_texture_{W*H}` inside the AttributeTimeline f32 buffer
   (`surface.py:1414-1437`, `timeline.py:812-814`). No further copy until
   render.
3. **[batch]** Materialization: the timeline queries the edit log and writes
   the dense `[T, rows, W·H·5]` window (`timeline.py:1543-1549`) — copy #2
   ([frame] writes; replay of an animating assignment additionally lerps over
   all T rows first, `timeline.py:755-758`). On CUDA/MPS the whole window is
   moved to the render device because a texture exceeds
   `WIDE_ATTR_MIN_CHANNELS = 2^16` (`timeline.py:746,770-776`, move at
   `:1551-1556`) — device move; the edit log itself was moved once per render
   job (`:1494-1504`).
4. **[batch]** Primitive build (`Surface._build_render_primitive`): view to
   `[T,H,W,5]`, optional wrap-pad copy on closed axes only (never for a quad;
   `surface.py:403-433,3127-3136`), then `.mult_opacity(opacity)` →
   `prep_set` does `self.data.clone()` — copy #3 ([frame]-sized: full T
   images) + arithmetic pass (`color.py:205-226`, `surface.py:3120-3139`).
5. **[batch]** Pack: `_stash_texture_maps` `.float().contiguous()` — copy #4
   only if the clone above is non-contiguous (usually no-op);
   `_pack_frame_visibility` reduces texture alpha (read-only,
   `primitives.py:1027-1058,1199-1200`).
6. **[batch]** GPU merge input staging: `_upload_primitive_inputs` moves every
   `_rt_*` tensor including `_rt_texture_map` to the merge device
   (`scene_builder.py:150-159,1266-1274`) — device move (CPU→GPU) of copy #3's
   bytes. CPU merge keeps them where projection built them.
7. **[batch]** Merge append: `_append_texture` — device move if needed
   (`scene_builder.py:1348-1353`), then under default `LINEAR_COLOR_SPACE=True`
   (`settings.py:78`) `srgb_to_linear` decodes channels 0:3 into a fresh
   tensor re-cat with 3: — copy #5, the sRGB→linear conversion
   (`scene_builder.py:1354-1359`); zero-pad to 5 channels only when C<5
   (`:1360-1362`); reshape into the flat texel list is a view (`:1364`).
8. **[batch]** Assembly: `scene["textures"] = _cat_collections(...)`. With ≥2
   textured collections this is `torch.cat` after `_unify_time` expands each
   map's time axis to the max T (`scene_builder.py:1721-1722`,
   `utils.py:51-60`) — copy #6 (+ expansion copies for any map shorter than
   T). With exactly one collection it passes through uncopied
   (`utils.py:57-58`).
9. **[batch]** Arena upload: `copy_merged_scene_to_arena` byte-copies every
   unique storage into the render arena (or host→device here),
   transactional per storage (`tracer.py:1165`,
   `scene_builder.py:459-522`) — copy #7. Cached: a re-render of the same
   batch reuses `_rt_merged_scene`/`_rt_device_scene`
   (`scene_builder.py:1250-1253`, `tracer.py:1144-1165`).
10. **[frame, in-kernel]** Sampling: `sheet_resolve_shade` → `_tri_color_g` →
    `_flat_triangle_color(_trim)` → `_sample_texture`/`_sample_tex_vec5`,
    reading `textures[f % T, offset + cx·h + cy, 0:5]`
    (`sheet_resolve_taichi.py:427`, `wavefront_kernels_taichi.py:307-322`) —
    reads only, no further copies. Env maps, if present, were appended to the
    same buffer before upload (`tracer.py:1156-1164,858-883`).
11. **[batch, cleanup]** After primitives are built the materialized window is
    dropped (`render_loop.py:2270-2274`, `timeline.py:2955-2967`); the
    primitive's own copies die with the collection after the merge
    (`scene_builder.py:1952-1959`).

Net: ~4 full-size copies of every texel beyond the buffer the kernel reads
(materialize write, opacity clone, sRGB decode, arena upload; +cat when ≥2
textured collections), plus up to two device moves, all [batch]-scoped, all
f32.

## Question B — largest byte multiplier, from the code

Per-frame texel payload is 20 B/texel (5 channels × f32) throughout.

**(i) Static textured quad, B-frame batch.** Unique information is one image
(20·W·H B; 4·W·H in the source PNG's u8). The merged upload carries B
identical copies (`scene["textures"]` never time-collapsed, Claim 4/1), so the
dominant multiplier is the **frame axis: ×B pure waste** (B=30 → 30×). Next in
order: **×5 from f32-5-channel vs the file's u8-RGBA**; **×N** if N Mobs share
the image (no dedup, Claim 2); transient prep peak adds ~2-3 more concurrent
image sets (window + premultiplied copy + decoded map + merged buffer, steps
3-9 above). All of ×B is removable with the same `f % shape[0]` trick the
tri/circuit tables already use (`_dedup_time`, `scene_builder.py:525-533`).

**(ii) Animated ~1774×887 ImageMob, 30-frame batch.** Genuine data is 30
distinct images: 1774·887·5·4 ≈ **31.5 MB/frame → ≈944 MB** that must reach
the GPU (matches the in-code "whole image per frame … T x 31 MB pass"
estimate, `timeline.py:755-763`, and "a second 30 MB texture",
`surface.py:1426-1431`). Here the frame axis is real work, so the largest
*avoidable* multipliers are:

* **dtype/channel width: ×5** vs 8-bit RGBA source (u8→f32 ×4, 4→5 channels
  ×1.25); the sampler consumes plain lerps of stored values, so nothing in it
  requires f32 storage (Claim 3).
* **copy chain: ×~3-4 transient/concurrent**, not additive traffic but peak:
  during merge/upload up to four near-full representations coexist
  (materialized window + premultiplied clone + decoded linear map + merged/
  arena buffer; steps 3-9), i.e. roughly 3-4 GB touched for ~944 MB of
  retained data on this size, each pass [batch]-scoped.
* **×N** again if several Mobs share the asset.

Ranking for structural work: (1) time-collapse `scene["textures"]` (fixes the
entire ×B waste of case (i) with existing machinery), (2) cross-collection
texture dedup keyed on content/tensor identity (case (i) ×N and repeated
stickers), (3) lower-precision texel storage/sampling (×4-5 on everything),
(4) shortening the copy chain (decode-in-place / premultiply-on-upload).

## What I did not verify

* **Nothing was measured.** All multipliers are derived from shapes/dtypes in
  code; no render, benchmark, or memory profile was run (brief forbids it;
  container is CPU-only, so device-move costs are described, not timed).
* Whether `_stash_texture_maps`' `.contiguous()` actually copies in practice
  (depends on whether `Color.prep_set`'s `broadcast(...).contiguous()` output
  is already contiguous — almost certainly yes, making step 5 a no-op; I did
  not execute it).
* The exact value of `active_time_inds` left behind by the last replayed
  function in a window (`timeline.py:1584-1585`, set inside the replay loop):
  whether a texture read can observe fewer than the batch's T frames when some
  other function replayed last. The dense-T statement holds for the common
  case (no overlapping replay narrowing); I did not trace every interleaving.
* CUDA/MPS behaviour end-to-end (this container has no GPU): the
  `merge_on_gpu`/`project_on_gpu`/wide-attr-device defaults are read from
  settings code, not exercised.
* Whether the Monte Carlo path (SAMPLES_PER_PIXEL > 1) samples textures
  differently — out of scope (default deterministic path only); its kernels
  were not audited for LOD.
* `wrap_pad_texture` growth on closed-axis surfaces (Sphere/Torus) adds one
  row/column per frame; I confirmed the mechanism (`surface.py:430-441`) but
  did not compute its cost contribution.
* Bezier circuit-grid transport (`circuit_colors`/`circuit_border_colors`)
  was only sketched (padding + time-dedup, `scene_builder.py:51-71,1829-1836`)
  to contrast with `scene["textures"]`; a full audit of the `[Tf,C,P,5]`
  pipeline was not done.
