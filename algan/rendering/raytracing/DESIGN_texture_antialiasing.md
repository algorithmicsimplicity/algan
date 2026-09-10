# UV texture anti-aliasing

Implements `RENDERER_WORK_QUEUE.md` item 4. Enabled by default with
`SETTINGS.raytracing.texture_antialiasing`; set it to `False` to rebuild the next
batch with the legacy bilinear sampler. The switch is data-driven, not a
captured `ti.static` global. This is independent of `glossy_prefilter`.

## Footprint rather than coverage

A sheet's covered area alone does not tell us how many texels land in a pixel:
a fully covered distant checkerboard and a magnified image both have area one.
Instead, derive the world-space gradients of U and V from the hit triangle's
edges and UVs, then project the circular ray footprint onto its geometric plane.
For a unit geometric normal `n`, direction `d`, and a tangent gradient `g`, the
projected gradient is `g - n * dot(g, d) / dot(n, d)`. Its length times the cone
width gives a UV width. This handles nonuniform UV scale and grazing incidence
without dividing both texture axes indiscriminately by the grazing cosine.

The cone width is the camera's pixel angular size times **accumulated camera
path length**, using the distance already carried in `rs_sca`. A reflected ray
therefore does not restart its texture footprint at zero on the mirror. No
extra rays, ray differentials, per-ray storage, traversal launches, or arena
arguments are needed. Untextured hits and promoted 1x1 maps skip the geometry
calculation. The sheet resolver computes the same camera scale from its existing
camera vectors; classic wavefront and path tracing reuse `pixel_world_scale`.

For each map separately, `rho = max(width * dU, height * dV)` and
`lod = clamp(log2(max(rho, 1)), 0, last_level)`. Colour, material and normal maps
can have different sizes. Adjacent levels are bilinearly sampled and blended.
Full-resolution texels are not fetched at `lod >= 1`; at the last level there
is no second level fetch. The cost is at most eight texel taps per map lookup, rather
than extra surface shading or additional traced rays.

## Bank layout and construction

The existing vec5 bank holds the original level zero unchanged (including
packed RGBA and endpoint stacks). Metadata columns 18, 19 and 20 are the colour,
material and normal mip-directory row offsets, or -1 for absent directories.
No other metadata columns change. Old 18-column metadata remains level-zero
only. Each directory row is five **int32 bit patterns** in the float bank:
`(offset, width, height, time_length, last_level)`. This preserves integer row
offsets above 2^24. Row zero supplies wrap flags in its first lane, level count and base dimensions;
levels one onward refer to ordinary flattened `[T, W, H, 5]` data.

Pyramids are constructed once per distinct map placement **per merged batch**.
They reuse level-zero content dedup; endpoint timing is part of the reuse key.
No hidden persistent cache retains wide tensors outside the memory model. The
existing arena accounting includes the extra rows; GPU merge transient peaks
are learned by the existing ratio model, with its OOM/window-shrink retry as
the backstop for a growing animated pyramid; ray tile
sizing and kernel argument layouts are unchanged.

Each level halves both dimensions, rounding down (but never below one). This
keeps the last level reachable by `log2(rho)` for NPOT maps just above powers
of two. Even axes use a two-sample box;
odd axes integrate equally sized bins over the original pixel intervals. Unlike
padding/duplicating the edge texel, this preserves the image's mean through a
non-power-of-two pyramid. One-pixel axes stay one pixel. Reduced grids and the
enabled base sampler use texel-centred, clamp-to-edge coordinates. Disabling the
feature restores the old align-corners bilinear convention too.

Closed Surface axes carry their wrap declaration through primitive collection.
The duplicated closing texels are excluded from mip reduction: they must not
bias the mean or form a clamped seam. Coarse levels use periodic addressing and
a half-base-texel phase correction to match the existing `i/N` convention of
wrapped level zero. Open axes still clamp. No content probes infer wrapping.

RGB is decoded to working linear light **before** filtering (unless the legacy
nonlinear working-space option was explicitly selected). Colour's RGB/glow are
premultiplied by coverage for both bilinear and mip filtering, then divided by
filtered coverage on return to the existing straight-colour surface shader.
Thus transparent coloured texels do not leak halos into opaque neighbours.
The separate per-frame mob opacity is applied after the filtered sample.
Material maps are averaged as data; normal maps are averaged in their stored
vector representation and normalized by the existing normal-map shader.

An animated endpoint map must interpolate in authored space, then decode, then
filter. Interpolating already-decoded endpoint mips would change the animation.
The host reduces one interpolated frame at a time and retains only its reduced
levels, never a dense full-resolution frame window. This has a real cost:
animated endpoint maps retain reduced data for each frame, not just endpoints.
Static maps remain single-frame, including opacity-only animation. Reduced
levels follow the same flat-time/shared-time layout as level zero, so the
legacy shared-time switch does not duplicate an animation quadratically. Packed level
zero remains packed; reduced levels are float32 to avoid quantization artifacts.

## Scope and tradeoffs

The three production entry points are sheet resolve (including deferred-shadow
mode/memo reuse), classic hybrid wavefront shading (including trimmed layouts),
and path-tracer shading. All current-hit colour/material/normal queries receive
the same footprint. Secondary reflection/refraction and alpha pass-throughs use
the existing accumulated distance. Environment maps, shadow visibility queries
and path-tracer emitter-connection queries retain their own level-zero sampling;
they must not accidentally use the receiver's footprint.

This is an **isotropic pixel-cone approximation**, not anisotropic filtering or
full ray differentials. It can overblur oblique textures; the on-axis camera
angle is conservative away from the image centre. Curved-mirror magnification,
refractive focusing and stochastic-lobe differential growth are not tracked.
Roughness still comes from the existing glossy/Monte Carlo transport, not an
extra ad-hoc mip bias. Geometry-edge coverage remains the analytic rasterizer's
job. Procedural effects not represented by UV maps are outside this feature.

For square power-of-two float maps, reduced levels add approximately one third
the base texel count, plus a small directory. This is **not** a one-third memory
claim for packed RGBA: each reduced vec5 texel is 20 bytes while a packed base
texel is 4 bytes. Very thin images approach one extra base-sized row of texels.
The implementation favours inexpensive lookup and fidelity over quantizing the
coarse levels or re-expanding a packed base.

## Validation

`tests/unit_tests/test_texture_antialiasing_taichi.py` runs numerical oracles
through the compiled production samplers: odd dimensions and means, all data
channels, alpha/glow, closed seams, nonlinear endpoint timing, packed endpoints, opacity,
fractional levels, clamped coarsest levels, runtime disable and UV footprints.
`benchmarks/_texture_antialiasing.py` is the asset-free primary/mirror A/B for
both renderers, writing images and alternating warm-run metrics under
`algan_outputs`. `tests/unit_tests/test_texture_antialiasing.py` compares real
primary/mirror renders with an independently authored constant-mean oracle,
including the raster-disabled route and a subpixel-motion regression.
See the PR validation notes for the actual backend and measured results;
CPU measurements are not claims about GPU performance.
