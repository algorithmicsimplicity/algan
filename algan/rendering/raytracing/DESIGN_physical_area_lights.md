# Physical `RectAreaLight` geometry — future work

## Decision

In the path tracer, `RectAreaLight` should represent a real rectangular emissive surface.

There is no camera-visibility toggle and no camera-invisible BVH flag. If a camera ray hits the emitting side of the rectangle, the light is visible. The same geometry is visible to reflection/refraction/BSDF continuation rays and is sampled by next-event estimation (NEE).

This deliberately prefers physically coherent path-traced behavior over compatibility with the older analytic-light convention where light objects themselves are invisible.

## Required behavior

A path-traced `RectAreaLight` is two one-sided emissive triangles with these semantics:

- **Camera-visible.** Direct camera hits return the emitter radiance.
- **Indirectly visible.** Reflection, refraction and ordinary BSDF continuation rays can hit the same geometry.
- **Opaque geometry.** Rays do not peel through the rectangle merely because it is a light.
- **Geometric occluder.** The rectangle can block visibility to other emitters like any other opaque surface. A shadow/visibility ray aimed at this emitter must terminate at the sampled emitter endpoint rather than treating the emitter as an intervening blocker.
- **One transport model.** Direct NEE and BSDF-hit emission use the same emitter radiance and MIS probability model.

The path tracer should not add a public setting such as `visible=False`; physical visibility is the definition of `RectAreaLight` on this render path.

## Why change the current prototype

The current area-light-quad prototype appends two emissive triangles to a private copy of an already merged scene. Because the triangle BVH has already been built, the widened primitive set requires reconstructing triangle bounds and rebuilding the BVH. The synthetic quads are also marked non-opaque so camera rays can peel through them, which disables opaque/closest-hit optimizations for the batch.

Those costs exist only to preserve camera invisibility. With the physical-emitter decision, they are unnecessary.

## Implementation direction

Integrate area-light geometry **before the path tracer's normal triangle acceleration build**, rather than widening an already merged scene.

1. During path-traced scene preparation (`samples_per_pixel > 1`), convert every `RectAreaLight` to two triangles before triangle BVH construction.
2. Append those triangles through the same merged triangle tables as ordinary geometry: position, normal, material/emission, object identity, frame-validity and texture metadata.
3. Mark the triangles one-sided and opaque. Do not add a camera-invisible flag and do not force `all_visible_opaque = False` merely because area lights exist.
4. Build the triangle BVH once over the complete primitive set, including area-light triangles. Remove the path-tracer-only post-merge BVH rebuild.
5. Keep the emitter metadata required by NEE/light-tree sampling, but make it refer to the same triangle primitives used by BSDF hits. Remove the packed analytic area-light rows from the path tracer's direct-light table so the light is not counted twice.
6. Preserve the existing `RectAreaLight` intensity/color/size/orientation and distance-falloff semantics when deriving emitted radiance. Direct-hit emission and NEE must evaluate the same radiometric model.
7. Treat the rectangle as ordinary opaque geometry for visibility rays. NEE visibility should use a segment whose endpoint is the sampled light point (with the existing robust ray-offset/tmax rules), so the target emitter does not self-occlude while still allowing the panel to occlude unrelated paths.
8. Delete the camera-segment special case, fake non-opacity, private scene widening and secondary triangle-BVH rebuild once the integrated path is complete.

This integration may remain path-tracer-specific at merge time so the deterministic renderer can keep its existing analytic `RectAreaLight` implementation until/unless its semantics are deliberately changed separately.

## Acceptance tests

The implementation is complete when all of the following hold:

- A `RectAreaLight` directly in front of the camera appears as an emissive rectangle.
- The same light appears consistently in a mirror and through refractive transport where physically visible.
- An opaque object behind the light is hidden by a direct camera ray through the panel.
- The panel can occlude illumination from another emitter.
- NEE aimed at the panel is not rejected as self-occlusion.
- Direct-hit and NEE estimates agree with the same emitter radiance/MIS model.
- Adding a `RectAreaLight` does not force the batch onto the non-opaque traversal path solely because it is a light.
- Scene preparation performs one triangle-BVH build for the path-traced merged scene; no post-merge area-light BVH rebuild remains.

## Non-goals

This work does not add a visibility toggle, a camera-invisible leaf bit, a separate `AreaEmitter` public API, or a requirement to change the deterministic renderer's current light-object visibility semantics.