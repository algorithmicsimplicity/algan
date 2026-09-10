# Physical `RectAreaLight` geometry and emission

**Status: implemented for path tracing.** Geometry integration landed
2026-09-07; receiver-independent emission and the physical API contract landed
2026-09-08. Verified against `master` at
`f10cc230a108863d02980fc27079254473ae7de3` (2026-09-09). This records the
current integration, not a proposal to add another BVH or a camera-visibility
switch.

## Visibility and authoring contract

A path-traced `RectAreaLight` is a rectangular physical surface: camera,
reflection and transmission rays can hit it. Its front emits; its back is
black. Both sides are opaque and can occlude other geometry. One-sided emission
does **not** mean that the back face is transparent.

There is no public camera-invisibility toggle. The experimental
`pt_area_light_quads=False` setting retains the analytic-row arm for comparisons;
it is not a separate visibility property on the light.

## One merge and one acceleration-structure build

`scene_builder._merge_scene` calls `area_light_quads.build_area_light_quads`
using the batch's immutable light snapshot, before the ordinary BVH build and
arena upload. The helper appends two triangles per supported rectangular light,
extends the material/geometry arrays and BVH inputs, and records the synthetic
emitter rows in `pt_quad_rows` and related metadata.

The helper handles the supported batch layouts conservatively. A light that
cannot be represented by this geometry path keeps its analytic rows. Do not
reintroduce the old prototype's per-window scene widening, secondary BVH build
or unaccounted persistent copies in `path_tracer.py`.

## Sampling and radiometry

Physical next-event estimation samples the emitter surface. The corresponding
analytic cell rows are removed from that estimator to avoid counting the light
twice. Authored fragment pipelines retain their direct-light rows, since their
appearance model is not simply the physical BSDF estimator.

An authored diffuse continuation uses `prev_pdf=-2` to suppress its immediate
synthetic-emitter contribution without making the panel transparent. Camera,
delta and positive-density MIS states remain distinct. Straight pass-throughs
preserve the marker; a new scatter replaces it.

Emitted radiance is distance-independent: `Le = linear_colour * intensity /
(width * height)`, including the existing glow and opacity scaling. Emitter hits
and next-event samples read that same value, and packed cells obey the same law.
`RectAreaLight` defaults to and only accepts `decay=2, distance=0`; other values
raise on construction and on later assignment or `set`, rather than reinstating
a nonphysical distance law. The per-quad falloff table, its two emission
multipliers and its light-tree exponent override are gone, so the packed cells,
authored materials, the light tree and the panel all agree on physical geometric
falloff.

## Radiometry and migration

`intensity` retains the physical normalization already used with `decay=2,
distance=0`: one-sided integrated flux is `pi * linear_colour * intensity`.
At fixed intensity a larger panel has lower radiance, not larger total power.
These are scene-linear units, not calibrated watts or lumens. `samples` changes
the row quadrature, not power or the panel's radiance.

New code should omit the two falloff keywords. Previously physical scenes keep
their light normalization. Scenes relying on the old default `decay=0` need their
intensity retuned; no single conversion can preserve the old distance law at
all receivers. Arbitrary decay/range still belongs to point/spot lights, not
emitting surfaces. No camera-visibility switch was added.

## Regression requirements

Keep coverage for camera-visible fronts and opaque backs, reflected/refracted
hits, finite visibility windows, emitter sampling/MIS, self-intersection at the
sampled endpoint, and agreement between emitter-hit and sampled-light radiance.
The area-light tests in `tests/unit_tests/test_path_tracer.py` exercise the
integration end to end, and `test_physical_area_emission.py` guards canonical
packing, normalization, distance-independent geometry, cloning and invalid
writes. The distance-dependent row/quad fixture was replaced with physical
comparisons at two light distances. A geometry change must preserve the normal
merge's frame layout, material widths, BVH leaf bounds and arena accounting.
