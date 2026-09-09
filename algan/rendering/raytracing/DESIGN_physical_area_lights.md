# Area lights as visible emitter geometry

**Status: implemented for path tracing.** Verified against `master` at
`f10cc230a108863d02980fc27079254473ae7de3` (2026-09-09). This records the current
integration, not a proposal to add another BVH or a camera-visibility switch.

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

`pt_quad_falloff` preserves the existing `decay`/`distance` law for both emitter
hits and next-event samples. **`decay=2, distance=0` gives distance-independent
emitter radiance.** The API default remains `decay=0`, a legacy no-falloff
lighting convention, so geometry integration alone does not make every authored
area light radiometrically physical. A consistent public radiance contract is
remaining work, tracked in the repository's `TODO.md`.

## Regression requirements

Keep coverage for camera-visible fronts and opaque backs, reflected/refracted
hits, finite visibility windows, emitter sampling/MIS, self-intersection at the
sampled endpoint, and agreement between emitter-hit and sampled-light falloff.
The area-light tests in `tests/unit_tests/test_path_tracer.py` exercise this
integration. A geometry
change must preserve the normal merge's frame layout, material widths, BVH
leaf bounds and arena accounting.
