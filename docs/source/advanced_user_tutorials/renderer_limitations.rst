.. _renderer-limitations:

====================
Renderer Limitations
====================

Algan's renderer is a hybrid: it resolves primary visibility with an exact
analytic rasterizer and traces rays for shadows, reflection and refraction. That
combination is fast and noise-free, and it buys those properties by not doing
some things a full path tracer does. This page is the complete list of what it
does not do, why, and what to reach for instead.

It is a reference rather than a tutorial. The companion pages --
:doc:`lighting_and_shadows`, :doc:`reflections_and_glass`,
:doc:`shaders_and_materials`, :doc:`images_and_textures`,
:doc:`performance_and_quality` -- describe the features themselves.

.. note::

   Everything below describes the renderer at its default settings. Where a
   limitation can be lifted by a setting, the setting is named. Names under
   ``SETTINGS.raytracing.experimental`` are explicitly *not* part of Algan's
   supported surface: they track the kernels and can change between releases.


Feature matrix
==============

Every renderer feature, and where it is available. "Analytic" is the
deterministic renderer's default path; "supersampled" is its fallback (both are
``samples_per_pixel == 1``); "Path tracer" is ``samples_per_pixel > 1``. Each
row links to the section that explains it.

.. list-table::
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - Feature
     - Analytic
     - Supersampled
     - Path tracer
     - Notes
   * - Analytic (exact-coverage) anti-aliasing
     - Yes
     - No
     - No
     - `Anti-aliasing`_
   * - Supersampling (``supersampling``)
     - Ignored
     - Yes
     - Ignored (jittered samples instead)
     - `Anti-aliasing`_
   * - Per-fragment materials
     - Triangles only
     - Triangles only
     - Triangles only
     - `What is lit, and how`_
   * - Extended lights (all but ``PointLight``)
     - Triangles only
     - Triangles only
     - Triangles only
     - `What is lit, and how`_
   * - Ray-traced shadows
     - Triangles only
     - Triangles only
     - Triangles only
     - `Shadows`_
   * - Soft shadows
     - Yes (8-ray fan)
     - Yes (8-ray fan)
     - Yes (sampled per path)
     - `Shadows`_
   * - Color / material / normal maps
     - Triangles only
     - Triangles only
     - Triangles only
     - `Texture maps`_
   * - Mip-mapped texture minification
     - **No**
     - **No**
     - **No**
     - `Texture maps`_
   * - Environment map (skybox + reflections)
     - Yes
     - Yes
     - Yes
     - `Texture maps`_
   * - Environment lighting (image-based)
     - Order-1 SH
     - Order-1 SH
     - Full map, importance-sampled
     - `Texture maps`_
   * - Mirror reflection
     - Yes
     - Yes
     - Yes
     - `Reflection, refraction and transmission`_
   * - Blurred (glossy) reflection
     - Opt-in, screen-space prefilter
     - **No** (the setting is inert here)
     - Yes
     - `Reflection, refraction and transmission`_
   * - Refraction (glass)
     - Yes, single medium
     - Yes, single medium
     - Yes, nested media
     - `Reflection, refraction and transmission`_
   * - Nested media (glass in glass)
     - **No**
     - **No**
     - Yes
     - `Reflection, refraction and transmission`_
   * - Transmission through a 2-D shape
     - Thin pane, no bending
     - Thin pane, no bending
     - Thin pane, no bending
     - `Reflection, refraction and transmission`_
   * - Custom fragment-shader pipelines
     - Yes
     - Yes
     - Yes
     - `What is lit, and how`_
   * - Custom ray scatter (bounce override)
     - **Falls back**
     - Yes
     - Yes, as a delta lobe
     - `Which renderer runs your scene`_
   * - Near clipping (``camera.near``)
     - **Falls back**
     - Yes
     - Yes
     - `Camera`_
   * - Far clipping (``camera.far``)
     - Yes
     - Yes
     - Yes
     - `Camera`_
   * - Transparent background
     - Yes (not with an env map)
     - Yes
     - Yes
     - `Which renderer runs your scene`_
   * - Glow / bloom
     - Yes
     - Yes
     - Yes
     - `Anti-aliasing`_
   * - FXAA
     - Yes
     - Yes
     - Yes
     - `Anti-aliasing`_
   * - Tonemapping (neutral / AgX)
     - Yes
     - Yes
     - Yes
     - —
   * - Global illumination, emissive surfaces as lights
     - **No**
     - **No**
     - Yes
     - `Not implemented at all`_
   * - Denoising (``denoise``, default on)
     - Not applicable (noise-free)
     - Not applicable (noise-free)
     - Yes
     - `Which renderer runs your scene`_
   * - True orthographic projection
     - **No**
     - **No**
     - **No**
     - `Camera`_
   * - Depth of field, motion blur
     - **No**
     - **No**
     - **No**
     - `Camera`_
   * - Volumetrics, ambient occlusion, displacement
     - **No**
     - **No**
     - **No**
     - `Not implemented at all`_
   * - Auxiliary passes (depth / normal / ID)
     - **No**
     - **No**
     - **No**
     - `Not implemented at all`_

"Triangles only" means the feature applies to triangle geometry and not to
Bezier circuits -- see :ref:`limits-lit`. "Falls back" means the batch is routed
off the analytic path onto the supersampled one. Nothing in this table is
refused: where a renderer cannot honour a feature it says so here rather than
dropping it silently, and if that ever changes Algan raises
:class:`~algan.errors.UnsupportedFeatureError` naming the feature rather than
rendering a wrong frame.

"Yes, as a delta lobe" is how the path tracer takes a **custom ray scatter**.
Your function picks the direction; the path continues along the branch it
returns with weight 1 and no MIS coverage, exactly as refraction and a tinted
pane already do. The one limitation is the flip side of that: such a surface is
not covered by next-event estimation, so light reaching it arrives only through
the sampled continuation -- a scatter surface facing a small bright light is
noisier than a Lambert one in the same place, and needs more samples rather
than a different renderer.


Which renderer runs your scene
==============================

``samples_per_pixel``: two renderers
------------------------------------

``SETTINGS.raytracing.samples_per_pixel`` selects the renderer, not a quality
dial. ``1`` (the default) is the deterministic renderer; anything above it is
the path tracer, and the path tracer **refuses nothing**: every feature the
deterministic renderer accepts renders there too, custom scatter overrides
included (as a delta continuation -- see :ref:`renderer-capabilities` for the
full table). That is deliberate: it is the fallback, so a feature it rejected
would leave that scene with no renderer at all.

The path tracer is the **fallback** for the scenes the deterministic renderer
cannot render: more lights than its shadow cap (below), reflective or
transparent geometry whose ray splitting exhausts render memory, and anything
needing global illumination. A failure of either kind names the switch; the
setting to reach for is a modest sample count with a short bounce budget,
``SETTINGS.raytracing.set(samples_per_pixel=16, max_bounces=2)``, raised from
there only if the scene needs indirect light or the denoised result is still
noisy. See :ref:`renderer-settings` in the performance guide.

Five further consequences of the split, not covered there:

* The path tracer shades **per fragment**, like the deterministic renderer's
  fragment route. Direct light comes from sampling one entry of a
  power-weighted table per lit surface point -- delta and area lights,
  emissive triangles and the environment map together -- while the
  direction-less ambient and hemisphere lights keep their deterministic fill.
  What it does not reproduce is the deterministic renderer's screen-space
  glossy prefilter: real sampled glossy transport replaces it.
* **Lit surfaces are not as bright here, and they answer to one BSDF.** The
  path tracer evaluates every light with the same physically-normalised
  response its own rays sample -- ``albedo / pi`` diffuse, GGX with the exact
  Smith masking-shadowing term, Fresnel and multiple-scattering compensation
  -- where the deterministic renderer uses its stage formulas. So a Lambert
  surface under a light is about ``pi`` times dimmer than its
  ``samples_per_pixel = 1`` render, before whatever indirect light the scene
  bounces back into it. This is deliberate: the path tracer is the fallback
  for scenes the other renderer cannot do, and one response is what makes an
  area light and an emissive quad of the same radiance light a surface
  identically. Two consequences worth knowing:

  * :class:`~.MeshPhongMaterial` has **no Blinn-Phong highlight** under the
    path tracer. Its ``specular`` colour and ``shininess`` are converted to a
    GGX lobe (``alpha = sqrt(2 / (shininess + 2))``, F0 = ``specular``), so
    the material still has a highlight -- it is a slightly different shape
    and it sits in a slightly different place. Nothing is dropped; nothing
    matches the deterministic renderer pixel for pixel either.
  * The ambient and hemisphere fill reaches **diffuse only**. A constant
    radiance arriving from every direction, integrated over the diffuse
    lobe, is exactly what the fill contributes; the specular equivalent is
    real indirect transport, which this renderer has and the other one does
    not.

  If you are comparing the two renderers side by side, expect to adjust
  ``intensity``. If you reached for the path tracer because the other one
  could not render your scene, there is nothing to compare against.
* **A** :class:`~.RectAreaLight` **is real geometry here.** The deterministic
  renderer expands one into a grid of ``samples`` point emitters; the path
  tracer instead treats it as an emissive rectangle, which is what it
  physically is. Three visible consequences: a mirror or a polished metal
  **shows the light's reflection**, which the deterministic renderer cannot
  draw at all; the panel itself is still **invisible to the camera**, so
  putting a light in shot does not put a white rectangle in the frame; and it
  still **casts no shadow**, so you can place one between the camera and your
  subject. Its ``decay`` and ``distance`` mean exactly what they do in the
  other renderer -- ``decay = 0``, the default, really is no falloff, even
  though a physical emitter of that size would fade with distance. A
  ``samples = 16`` area light also costs the sampler two emitters here rather
  than sixteen, so raising ``samples`` for the deterministic renderer's sake
  no longer makes path-traced renders slower.
* Its raw output is stochastic, so low sample counts are visibly noisy. By
  default it is **denoised** (``SETTINGS.raytracing.denoise``) with the Open
  Image Denoise RT filter re-implemented in torch, guided by albedo and
  normal information the render accumulates alongside the image. The weights
  (about 2 MB) are fetched once into the cache directory on first use; a
  machine that cannot fetch them renders without denoising after one warning.
  Flat 2-D content composites deterministically inside the path tracer (zero
  variance), so vector graphics and text stay exact with or without it.
* Path-traced output is *stochastic*. Two renders of the same scene converge
  to the same image but need not be identical frame for frame, and the
  renderer makes no byte-identity promise. Raise ``samples_per_pixel`` (or
  leave the denoiser on) if residual noise is visible;
  ``SETTINGS.raytracing.experimental.pt_seed`` re-rolls the noise without
  changing what the render converges to.

Within the deterministic renderer: two paths
--------------------------------------------

The deterministic renderer has an **analytic-coverage path** (the default, and
the one every example in these docs uses) and a **supersampled fallback**. The
fallback renders the frame at ``supersampling`` (``ssaa``) times
the output resolution and box-filters it back down, casting one primary ray per
sub-pixel sample.

A batch falls back when any of the following holds:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Condition
     - Why
   * - ``samples_per_pixel > 1``
     - A different renderer entirely.
   * - ``camera.near > 0``
     - Near clipping is only implemented in the ray-traversal path.
   * - A mob whose fragment shader overrides ray bouncing
     - Custom scatter is only implemented in the ray-traversal path.
   * - A transparent background **together with** an environment map
     - The environment prefill would fill the alpha the background owes.
   * - The batch contains no triangles and no Bezier circuits
     - Nothing for the rasterizer to emit.
   * - Any of the analytic-AA or sparse-coverage experimental switches off
     - They are preconditions of the analytic path, not independent options.

The fallback is a genuine quality *and* cost change, so it is worth knowing when
you are on it:

* Analytic coverage is off. Edge quality is whatever
  ``supersampling`` (default ``2``) buys.
* The frame buffer is ``supersampling ** 2`` times larger, so
  batches shrink by the same factor and the render takes correspondingly longer.
  At the default that is 4x.
* Per-fragment shading, shadows, reflection and refraction all still work. Only
  the way primary visibility and coverage are resolved changes.

The two paths do **not** produce identical images
-------------------------------------------------

This is deliberate, and the differences are small but enumerable:

* **Anti-aliasing.** Analytic coverage against box-filtered supersampling.
* **Shading rate.** The analytic path evaluates a material **once per
  same-surface region per pixel** (see :ref:`limits-shading-rate`); the fallback
  evaluates it once per ray hit.
* **Intersection.** The analytic path uses an exact fixed-point screen-space
  fill rule; the fallback uses a watertight ray/triangle test. They agree to
  floating-point epsilon on where a surface is, and can disagree about which of
  two adjacent triangles owns a sample exactly on their shared edge.
* **Shadow query points** are reconstructed from rasterizer barycentrics on the
  analytic path (agreement with the traversal path measured at ~5e-5 world
  units), so a shadow *boundary* can sit about a pixel differently.
* **Coplanar decals.** On the analytic path a fragment whose depth and layer key
  exactly equal an opaque winner's is culled; see
  :ref:`limits-coplanar`.

Do not treat one path as the reference for the other.


.. _limits-lit:

What is lit, and how
====================

Only triangle geometry is lit
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Object family
     - Lit
     - Receives shadows
     - Casts shadows
     - Reflects / transmits
     - Image textures
   * - Anything made of **triangles**: :class:`~algan.mobs.surfaces.surface.Surface` and its subclasses
       (:class:`~algan.mobs.shapes_3d.Sphere`, :class:`~algan.mobs.shapes_3d.Cylinder`, :class:`~algan.mobs.shapes_3d.Cone`,
       :class:`~algan.mobs.shapes_3d.Torus`, :class:`~.ImageMob`, :class:`~algan.mobs.shapes_3d.Dot3D`, point clouds,
       …), :class:`~algan.mobs.shapes_3d.Polyhedron`, imported 3-D models, and
       :class:`~.TriangulatedBezierCircuit`
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Anything made of **Bezier circuits**: :class:`~algan.mobs.shapes_2d.Line`,
       :class:`~algan.mobs.shapes_2d.Polygon`, :class:`~algan.mobs.shapes_2d.Circle`, :class:`~algan.mobs.shapes_2d.Square`,
       :class:`~algan.mobs.shapes_2d.Dot`, :class:`~algan.mobs.text.Text`, :class:`~algan.mobs.text.Tex`, Manim vector mobs
     - **No**
     - **No**
     - Yes
     - Yes
     - Color grid only

A Bezier circuit is drawn with the color it was authored with. No light source
touches it, no shadow falls on it, and a normal map or material-property map has
nothing to perturb. It *is* a full participant in ray transport in the other
direction: it occludes shadow rays according to its own opacity, and it can be
made reflective or transmissive with a material, so a mirror will show it.

This is the intended behaviour for flat 2-D content, and it is what keeps text
legible under any lighting rig. It is a limitation only if you wanted a lit 2-D
shape. Then use :class:`~.TriangulatedBezierCircuit`, which triangulates a
bezier outline into a real mesh precisely so that its interior can carry
per-fragment shading, a texture and 3-D lighting -- at the cost of the analytic
outline the circuit path gives you for free.

Materials: which are shaded per fragment
----------------------------------------

A material is shaded **per fragment**, in the render kernel, only if it has an
in-kernel implementation. Everything else is baked into vertex colors before
the frame is rendered.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Material
     - Shading
     - Consequences
   * - No material set (Algan's default), :class:`~.MeshLambertMaterial`,
       :class:`~.MeshPhongMaterial`, :class:`~.MeshStandardMaterial`,
       :class:`~.MeshPhysicalMaterial`, :class:`~.MeshToonMaterial`,
       :class:`~.MeshNormalMaterial`, :class:`~.MeshMatcapMaterial`,
       :class:`~.MeshDepthMaterial`
     - Per fragment
     - Full behaviour: every light type, shadows, environment lighting, normal
       maps, material-property maps.
   * - A custom pipeline from
       :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
     - Per fragment
     - Same, and it can read shadow visibility itself.
   * - :class:`~.MeshBasicMaterial` / :class:`~.UnlitMaterial`
     - Unlit by design
     - Passes its color through. Not a limitation — it is the point.
   * - A custom **per-vertex** shader from
       :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`
       (a plain function, not a
       :class:`~algan.rendering.shaders.fragment_shaders.FragmentStage`)
     - **Per vertex only**
     - See below.

Custom per-vertex shading is the sharp edge here. Because such a shader runs at
the mesh's vertices before rendering:

* It sees **only** :class:`~.PointLight`. Directional, ambient, hemisphere,
  spot and rect-area lights are skipped entirely, as is an environment map's
  diffuse contribution.
* It **never receives shadows**, whatever ``shadows`` is set to.
* Its shading resolution is the mesh's resolution: lighting is interpolated
  between corners instead of evaluated per fragment.

None of that is silent. Combining a vertex-baked shader with a lighting rig that
asks for more than the bake delivers -- any light beyond a plain
:class:`~.PointLight`, ``shadows=True``, or an environment map -- emits a warning
naming what is being dropped. It fires where
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` or
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material` is
called, against the lights that exist by then, and again once per render over
the whole scene, which is what catches the usual authoring order of choosing the
shader before spawning the lights.

Two of the built-in materials keep their own documented approximations even
though they now shade per fragment:
:class:`~.MeshMatcapMaterial` never samples a matcap image; it uses a
view-facing approximation tinted by the base color.
:class:`~.MeshToonMaterial` never samples a gradient map; its band count comes
from the Algan-specific ``bands`` argument.

Three.js material properties that are accepted and ignored
----------------------------------------------------------

The material classes mirror Three.js's API, and some of that API has no
implementation behind it. Setting one of these emits a warning and has no
effect:

* Every image slot on a :class:`~.Material`: ``map``, ``alpha_map``, ``ao_map``,
  ``env_map``, ``light_map``, ``bump_map``, ``normal_map``,
  ``displacement_map``, ``roughness_map``, ``metalness_map``, ``emissive_map``,
  ``specular_map``, ``gradient_map``, ``matcap``, ``clearcoat_map``,
  ``clearcoat_normal_map``, ``sheen_color_map``, ``transmission_map``,
  ``thickness_map``, ``iridescence_map``, ``specular_intensity_map``,
  ``specular_color_map``, ``normal_scale``, ``displacement_scale``,
  ``displacement_bias``.
* ``wireframe``, ``vertex_colors``, and any non-default ``side``
  (``BackSide`` / ``DoubleSide``). Algan renders all faces; whether a
  back-facing hit is lit from the viewer's side is decided by the geometry
  through ``Mob.two_sided``, not by the material.

Textures do work — through a different door. See the next section.

:class:`~.MeshPhysicalMaterial`'s ``attenuation_color`` and
``attenuation_distance`` *are* honoured: light crossing a transmissive solid is
absorbed along the path it actually travels, following
``KHR_materials_volume``'s ``attenuation_color ** (d / attenuation_distance)``,
so a thick piece of colored glass comes out deeper than a thin one. Two limits
are worth knowing. The path length is measured from the surface the ray last
crossed, which is exact for a single convex solid and an approximation for
nested transmissive media. And ``thickness`` is stored for API parity and unused
— three.js's rasterizer needs it because it has no ray to measure, and Algan
does not.

Shadow rays carry color through the same medium. A shadow ray holds an RGB
payload rather than one scalar per light, so a transmissive surface tints the
light it passes by its albedo -- the same treatment the refracted ray gets --
and a transmissive *solid* also absorbs over the chord the ray spends inside
it, from the same ``attenuation_color`` / ``attenuation_distance`` the view ray
uses above: a bigger piece of colored glass casts a deeper shadow, not just a
colored one. Circuits are flat zero-thickness panes with no interior, so they
tint but never absorb.

Two limits stand. There is still no refraction: the shadow ray travels
straight through the glass, so there is no caustic core, and the umbra comes
out uniformly tinted where a path tracer concentrates light into a bright
centre. And the entry/exit pairing behind the chord is exact for a single
convex solid and approximate where solids nest or overlap. Both behaviours are
compiled in behind ``ALGAN_RGB_SHADOW_TINT`` (default on); set it to ``0``
before ``import algan`` to restore achromatic shadows.


Texture maps
============

Algan samples exactly three maps per triangle, bilinearly, in the render kernel.
They live on the geometry. A material forwards ``map``, ``normal_map``,
``roughness_map`` and ``metalness_map`` onto it (see
:doc:`shaders_and_materials`), but only the geometry's own arguments reach every
channel, and only they can be animated.

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Map
     - How to set it
     - Notes
   * - Color (RGB + glow + alpha)
     - :class:`~algan.mobs.surfaces.surface.Surface`'s ``color_texture``; :class:`~.ImageMob`; a glTF/FBX
       base-color texture
     - Drives albedo and the glow lane. Alpha is honoured, including by shadow
       rays.
   * - Material properties
     - :class:`~algan.mobs.surfaces.surface.Surface`'s ``reflectivity_texture``, ``roughness_texture``,
       ``refractive_index_texture``
     - Three channels only. There is **no way to author a transmission map**
       even though the kernel has a channel for one.
   * - Tangent-space normal
     - :class:`~algan.mobs.surfaces.surface.Surface`'s ``normal_texture``; a glTF/FBX normal texture
     - Tangents are derived per hit from positions and UVs, so a mesh with
       degenerate UVs falls back to the unperturbed normal.

Everything else about texturing:

* **There is no mip chain and no anisotropic filtering.** A minified texture --
  a detailed image on a small or steeply angled surface -- aliases and crawls as
  the camera moves. Bilinear magnification is fine; minification is not
  filtered at all. Pre-downsample the image to roughly the size it will occupy
  on screen if this bites.
* **Bezier circuits carry a color grid, not a UV-mapped image.** A 2-D shape's
  ``grid_width`` x ``grid_height`` grid of color samples is
  laid over the shape's own frame. It is not an image sampler and it takes no
  normal or material map. :class:`~.ImageMob` is a :class:`~algan.mobs.surfaces.surface.Surface`, so it is
  the way to put a real image on screen.
* **Imported models collapse two maps to a constant.** glTF base-color and
  normal maps are sampled per fragment; a packed **metallic-roughness** map and
  an **emissive** map are reduced to their *mean* and applied as per-primitive
  constants. Occlusion maps are ignored.
* **Environment maps are resampled to at most 2048 pixels wide**, and on the
  deterministic renderer their diffuse (image-based-lighting) contribution is
  an **order-1 spherical harmonic** -- four coefficients. That is enough for a
  directional tint and no more: a map with a small bright sun lights the scene
  as though the sun were smeared across the sky. The map's *specular*
  contribution -- the sky itself, and what a mirror or a lens shows of it -- is
  sampled from the full (resampled) image, so only the diffuse term is
  band-limited. The **path tracer** has no such band limit: it
  importance-samples the full map through a luminance table at every lit
  surface point, so a small bright sun lights the scene as a sun, with the
  correct sharp-soft shadows.


Shadows
=======

Shadows are off by default (``SETTINGS.raytracing.set(shadows=True)``).

Who casts and who receives
--------------------------

* **Casters: everything.** Triangle geometry and Bezier circuits alike are
  traversed by shadow rays, and a partially transparent occluder attenuates the
  light by its opacity rather than blocking it. Stacked occluders multiply.
  Texture alpha counts.
* **Receivers: fragment-shaded triangle geometry only.** Bezier circuits never
  receive shadows (see :ref:`limits-lit`), and neither do the four per-vertex
  materials or :class:`~.MeshBasicMaterial`.

Limits and approximations
-------------------------

* **At most 16 lights are shadowed** (``ALGAN_MAX_SHADOW_LIGHTS``, set before
  the first render). Lights past the cap are still lit, just never shadowed, and
  **each emitter sample of a** :class:`~.RectAreaLight` **counts as one slot**, so
  a single 4x4 area light fills the default cap on its own. A render that goes
  over the cap warns and reports the surplus (:ref:`limits-truncation`). The
  deterministic renderer's cost also grows with every light, shadowed or not.
  A scene with more lights than the cap is what the path tracer is for: it
  samples lights instead of summing them, so every light casts a shadow and
  the cost per shading point does not depend on how many there are
  (``SETTINGS.raytracing.set(samples_per_pixel=16, max_bounces=2)``).
* **One shadow query point per same-surface region per pixel.** The query is
  taken at the region's largest fragment and, by default, at four sub-pixel
  positions around it
  (``SETTINGS.raytracing.experimental.analytic_aa_secondary_samples``). Shadow edges are
  therefore resolved at four positions per pixel, not analytically -- they are
  the softest edges in an otherwise exactly-antialiased frame.
* **Soft shadows use a fixed fan of 8 rays** per light per shaded point
  (``ALGAN_SOFT_SHADOW_SAMPLES``, baked into the kernels, so it must be set
  before ``import algan``). A wide emitter with 8 samples bands rather than
  blurs.
* **Contact shadows have a world-space floor.** A shadow ray starts 1e-3 world
  units off the surface along its face normal, and stops 2e-3 short of the
  light. An object resting on a plane loses its shadow within about that
  distance of the contact. Both offsets are absolute, so what they cost you
  depends on your scene's scale: see :ref:`limits-scale`.
* **A curved surface's shadow terminator is corrected, and a flat one's is not
  in need of it.** A sphere, cylinder, cone, torus or parametric
  :class:`~algan.mobs.surfaces.surface.Surface` is diced to flat triangles under
  a smooth normal field, so each facet is a chord *below* the surface it stands
  for and a shadow ray leaving it near the terminator can strike a neighbouring
  facet that rises above it -- speckled false self-shadow, which no acceptance
  epsilon can reject. The origin is therefore displaced onto the smooth surface
  the vertex normals imply before the ray is traced (Hanika, *Ray Tracing Gems
  II* ch. 4). A flat-shaded mesh has no smooth surface to be displaced onto --
  a :class:`~algan.mobs.shapes_3d.Polyhedron` carries no vertex normals at all,
  an imported flat mesh carries the same normal at every corner of a face --
  so its displacement is exactly zero and nothing about it changes. Turn it off
  with
  ``SETTINGS.raytracing.experimental.set(shadow_terminator=False)`` if you need
  to compare against the old behaviour.
* **No refractive shadow transport.** Light is not bent as it passes through
  glass, so there are no caustics: a glass object's shadow keeps its sharp
  silhouette, and everything its interior does to the light crossing it --
  opacity, albedo tint, absorption over the chord -- happens along a straight
  line. The path tracer's shadow rays travel the same straight line, so this
  holds under both renderers; see `Not implemented at all`_.


Reflection, refraction and transmission
=======================================

Reflection
----------

* ``SETTINGS.raytracing.max_bounces`` (default ``8``) caps reflection and
  refraction depth. Beyond it, a ray stops and contributes its remaining
  throughput to whatever is behind it.
* **Roughness does not blur a reflection by default.** A single continuation
  ray can only honestly stand for a narrow lobe, so a rough reflector's mirror
  ray carries only the share of its specular lobe that fits inside a cone one
  direction can represent -- 100% at roughness 0, about 83% at 0.10, 50% at
  0.15, 3% at 0.35 -- and the remainder goes back to the material's own
  roughness-correct highlight and ambient term. The result reads as a rough
  metal; it is not a blurred mirror image.

  ``SETTINGS.raytracing.set(glossy_reflection=True)`` replaces that throttle
  with the **split-sum approximation**, which is what a real-time renderer uses
  to get a wide lobe out of one deterministic ray. The lobe's energy becomes
  analytic -- the environment-BRDF term, exact and ray-free -- and its shape
  comes from tracing one mirror ray per pixel into a reflection buffer, blurring
  that buffer by the lobe's screen footprint, and compositing. It is still
  opt-in, and it has two limits of its own:

  * **It is screen space.** The reflection can only show what the frame
    contains. A reflector pointed at something behind the camera, or off the
    edge of the frame, reflects the background instead. An environment map
    covers exactly that gap and is the right pairing.
  * **A rough metal gets darker, correctly.** With the throttle, a metal keeps
    its ambient fill in place of the reflection it declines to draw. With
    split-sum that energy is spent on the reflection, which is as bright as the
    surroundings actually are -- dark, in a dark room.

  Blur radius, contact hardening and the mip prefilter are described in
  ``algan/rendering/raytracing/DESIGN_glossy_prefilter.md``. The older four-tap
  lobe fan remains reachable as
  ``set(glossy_reflection=True, prefilter=False)``; it is **not** recommended,
  because four taps cannot integrate a wide lobe -- with the screen-space
  rotation on it resolves a glossy gradient into a handful of levels that crawl
  as geometry moves, and with the rotation off the taps land as discrete ghost
  copies of the reflected image.
* **A reflected or refracted image is not analytically antialiased.** Coverage
  resolves a mirror's own outline exactly, but what the mirror shows is sampled
  by continuation rays -- four sub-pixel positions at best, and only when the
  branch carries at least 0.12 of the pixel's energy
  (``SETTINGS.raytracing.experimental.analytic_aa_secondary_min_energy``). Below
  that threshold it takes a single ray. A minified reflected image therefore
  aliases where the surface holding it does not.

Refraction
----------

* **Nested media are modelled, up to four deep.** A ray carries the stack of
  media it is inside, so each interface refracts with the relative index of the
  two media it separates: glass inside glass, a sphere inside a box, a bubble
  in a liquid. Only a ray that enters a **fifth** medium without leaving one
  loses track. Three limits stand: Fresnel reflectance uses the material's own
  index rather than the relative one, a scene carrying a custom fragment
  scatter gets no nesting at all (every interface there still assumes air
  outside), and the camera is assumed to start in air.
  ``SETTINGS.raytracing.experimental.set(nested_ior=False)`` returns to the
  air-outside assumption, which is worth knowing about for one reason: with the
  stack on, a ray grazing a shared edge of an *un-nested* solid is no longer
  bent a second time as though re-entering, so a tenth of a percent of pixels
  at edges and grazing silhouettes differ from older renders. That is the
  physically right answer at a hit where there is no interface, but it is a
  difference.
* **A Bezier circuit transmits as a thin pane**: light passes through tinted,
  but is not bent. Only triangle geometry refracts.
* **No dispersion.** Every wavelength takes the same index of refraction, so no
  colored fringing at a prism. Absorption over distance *is* modelled:
  transmitted light is attenuated along its actual path through the medium by
  the material's ``attenuation_color`` / ``attenuation_distance`` described
  above, and the shadow ray applies the same coefficient over its own chord.
* Refraction needs both ``transmission > 0`` and ``ior > 1``. In practice that
  means a :class:`~.MeshPhysicalMaterial`, a :class:`~algan.mobs.surfaces.surface.Surface` with a
  ``refractive_index_texture``, or an imported model whose material carries
  them. It routes the batch through the splitting ray path, which is the most
  expensive configuration Algan has.

The depth budget, and what happens at the end of it
---------------------------------------------------

A primary ray composites **at most 256 surfaces**. Beyond that the ray stops and
the background shows through the remainder of the stack. This is a real ceiling
only for pathological geometry -- 256 stacked translucent sheets in one pixel --
and reaching it now warns, naming the ceiling and how many rays hit it (see
:ref:`limits-truncation` for what is counted and how to read it back).

Continuation rays for reflection and refraction are allocated from a shared pool
sized from an estimate of how many the batch will need. Exceeding the estimate
costs a discarded and re-rendered tile, which shows up as render time rather
than as an error. A **single pixel** whose ray tree exceeds the whole pool
raises ``OutOfRenderMemory``; lowering ``max_bounces`` is the fix.


Anti-aliasing
=============

.. _limits-shading-rate:

Shading is evaluated once per surface region per pixel
------------------------------------------------------

The analytic path groups a pixel's fragments into **maximal same-surface
regions** and evaluates the material once per region, at the region's largest
fragment. That is what makes analytic coverage affordable, and it is exact
wherever shading varies smoothly across the region. Where it does not:

* A **hard crease** -- two flat-shaded faces of one solid meeting inside a
  pixel -- is split so each face shades with its own normal
  (``SETTINGS.raytracing.experimental.sheet_shade_split``, on by default).
  This case is handled.
* A **high-frequency texture** minified into one pixel is not. The region is
  shaded at one point, so a checkerboard smaller than a pixel resolves to
  whichever texel that point lands on. This is the same missing mip chain as
  above, seen from the shading side.

What analytic coverage does and does not resolve
------------------------------------------------

* **Exact:** a primitive's own outline, and the way several fragments of one
  surface tile a pixel between them.
* **Sampled at 8 sub-pixel positions:** occlusion *between different* surfaces.
  Silhouette-against-silhouette error is bounded by the contrast divided by 8.
* **Sampled at 4 sub-pixel positions:** shadow edges, reflected images,
  refracted images.
* **Not resolved:** texture minification.

Other anti-aliasing notes
-------------------------

* ``supersampling`` is **ignored** on the analytic path, which
  always renders at output resolution. It applies only on the supersampled
  fallback.
* FXAA is available (``video_settings.fxaa``). It runs on linear HDR values
  before tonemapping, where its luma-based edge detection is not the one it was
  designed around.
* An SMAA implementation exists in the source tree but is not connected to
  anything and cannot be enabled.
* A filled shape's drawn region is given a minimum half-width of 0.3 output
  pixels so that hairlines, thin glyph stems and degenerate zero-area fills
  survive at all
  (``SETTINGS.raytracing.experimental.analytic_aa_bez_min_half_width``).
  Sub-pixel strokes are therefore slightly heavier than their geometry.


Camera
======

* **True orthographic projection is not implemented.**
  :meth:`~.Camera.set_near_orthographic` is the only spelling, and it says what
  it does: an ordinary perspective camera moved 1e5 units back from its screen.
  It looks orthographic and is not:
  geometry spanning a large depth range still converges slightly, and the
  extreme camera distance puts every world-space epsilon in
  :ref:`limits-scale` a long way from the geometry it is meant to separate.
* **No depth of field**, no aperture, no focus distance. Everything is in focus.
* **No motion blur.** Frames are instantaneous samples of the timeline.
* **No lens distortion, no fisheye, no panoramic projection.**
* ``camera.near > 0`` forces the supersampled fallback path for the whole batch.
  Leave it at ``0`` unless you specifically need near clipping.
* Geometry crossing the camera plane is handled **exactly** -- such a primitive
  is intersected by ray casting per sub-pixel sample rather than projected --
  but a primitive whose bounding box *contains the camera origin* cannot be
  bounded on screen at all and is tested against the whole frame. A camera
  flying through the middle of a scene puts many primitives in that state at
  once, and it is the usual cause of a fly-through running out of render memory
  where the same scene renders fine from outside.


.. _limits-nonplanar:

Bezier outlines that are not flat
=================================

A Bezier circuit is resolved by intersecting a camera ray with the circuit's own
plane and deciding coverage analytically in that plane, which is what keeps a
circle exactly round and a glyph crisp at any zoom. Geometry that does not lie in
a plane cannot be resolved that way, so Algan classifies every circuit once, when
you construct it, and gives the non-planar ones real 3-D geometry instead:

* **Filled** -- each closed sub-path becomes curved patches, the same primitive
  :class:`~algan.mobs.surfaces.surface.Surface` produces. This is how a Manim ``Sphere`` imports.
* **Unfilled** -- the path is split into near-straight runs, each drawn as its
  own circuit facing the camera, so a 3-D curve keeps its position in space and
  its stroke keeps a constant width on screen.

Both produce ordinary geometry, so shadows, reflections and refraction see
exactly what the camera sees. What to know:

* The decision is made **at construction and does not change**, exactly as the
  circuit's plane does not. A flat shape that you later
  :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` into a
  non-flat one stays on the flat path (and keeps its original plane), and the
  reverse holds too.
* A non-planar **filled** circuit's holes are filled. The even-odd rule that
  carves a counter out of a glyph has no equivalent once each sub-path is its
  own patch group. Manim's 3-D tiles have no holes, so this is only reachable by
  hand-building one.
* A non-planar circuit's texture grid collapses to one color per shape, so
  :meth:`~.BezierCircuitCubic.set_color_by_function` and color waves across it
  come out flat. The grid is laid out across a circuit's plane frame, which
  these no longer have.
* Neighbouring patches share corner *positions* exactly, so the surface is
  watertight, but their corner *normals* are each estimated from one patch's own
  boundary -- about 2.5 degrees apart on a stock ``manim.Sphere()``, which is a
  sub-pixel seam at 1080p.

Set ``ALGAN_NONPLANAR_CIRCUITS=0`` to turn the whole thing off and flatten every
circuit onto a plane, which is what Algan did before this existed.

.. _limits-coplanar:

Ordering, coplanar geometry and z-fighting
==========================================

* Hits within **1e-4 world units** of each other along a ray are treated as
  coplanar and ordered by an internal layer index rather than by depth. That
  index puts **all Bezier circuits behind all triangle geometry**, and within
  each kind orders by position in the merged scene, which follows construction
  order but is not a documented contract.
* :attr:`BezierCircuitCubic.z_index <.BezierCircuitCubic.z_index>` is the
  supported way to break such a tie for 2-D shapes: it nudges the circuit
  toward the camera by one tie-bin (1e-4 world units) per unit of ``z_index``.
  **Only Bezier circuits have it.** There is no equivalent for 3-D geometry;
  move it.
* On the analytic path, a fragment whose depth *and* layer key exactly equal an
  opaque winner's is culled. A decal placed exactly on an opaque surface
  therefore disappears unless it sorts in front of it -- which, for a 2-D shape
  on a 3-D surface, it never does without a ``z_index``.

.. _limits-scale:

The renderer assumes a roughly unit-scale scene
-----------------------------------------------

Several of the renderer's tolerances are **absolute world-space constants**, not
fractions of the scene's own size:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Constant
     - Value
     - What it decides
   * - Minimum hit distance
     - 1e-4
     - Self-intersection rejection for bounced and shadow rays.
   * - Depth-tie epsilon
     - 1e-4
     - When two hits count as coplanar; also one ``z_index`` step.
   * - Triangle edge epsilon
     - 2e-4
     - When two hits on a shared mesh edge are merged into one.
   * - Shadow-ray origin offset
     - 1e-3
     - How far off a surface a shadow ray starts (and it stops 2e-3 short of
       the light).

Algan's default camera sits 7 units back and frames about 7 world units of
height at the origin, so all four are far below a pixel at ordinary scales. A
scene authored a thousand times larger will show z-fighting and merged surfaces;
one authored a thousand times smaller will lose contact shadows and
self-shadowing. **Scale the scene, not the camera** -- and note that
:meth:`~.Camera.set_near_orthographic` moves the camera 1e5 units out, which is
the same problem arriving from the other direction.


.. _limits-hard:

Hard limits
===========

.. list-table::
   :header-rows: 1
   :widths: 42 18 40

   * - Limit
     - Value
     - What happens if exceeded
   * - Surfaces composited along one primary ray
     - 256
     - The ray stops; the background shows through the rest. **Warns**
       (:ref:`limits-truncation`).
   * - Reflection / refraction bounces
     - 8 (``max_bounces``)
     - The branch stops and contributes its remaining throughput.
   * - Shadowed lights
     - 16 (``ALGAN_MAX_SHADOW_LIGHTS``)
     - Further lights are lit but never shadowed. **Warns**
       (:ref:`limits-truncation`). The path tracer has no cap: it samples
       lights instead of summing them.
   * - Overlapping layers of one surface in one pixel
     - 16
     - Further layers merge into the last, and attenuate once between them
       instead of once each. **Warns** (:ref:`limits-truncation`).
   * - Nested translucent closed-shell solids along one path-traced camera ray
     - 4
     - The surplus shell attenuates once per crossing instead of once per
       entry/exit pair, rendering slightly too opaque. **Warns**
       (:ref:`limits-truncation`).
   * - Frames in one render batch
     - 32767
     - Raises. Not reachable in practice -- memory bounds the batch far below
       this.
   * - Bezier circuits in one render batch
     - 8 388 607
     - Raises with a clear message.
   * - Triangles in one render batch
     - ~1.07e9
     - Not reachable; memory bounds it far below.
   * - Environment map width
     - 2048
     - Silently resampled down.
   * - :class:`~algan.mobs.surfaces.surface.Surface` construction grid
     - 200 vertices per axis
     - ``max_grid_resolution`` clamps the automatic search at construction.
       Render-time dicing is a separate budget, below.
   * - Subdivision level of one curved patch
     - 8
     - The dice stops refining that patch and **warns**
       (``RuntimeWarning``: "tessellation reached its safety cap before meeting
       render_tolerance_pixels for every patch").
   * - Diced triangles in one frame
     - 2 000 000
     - The level search refuses further promotions and warns, as above. The
       budget is per frame, not per batch, so a mesh does not pop at batch
       boundaries.
   * - Polyline samples per Bezier segment
     - 512
     - The flattening search stops refining; a very long curve viewed very close
       can show flattening facets.

Where a limit is marked *silent*, nothing is printed and no exception is raised.
Everything marked **warns** logs one ``WARNING`` naming the ceiling the first
time a render reaches it.

.. _limits-truncation:

Reading back what a render truncated
------------------------------------

Three of the ceilings above degrade the image rather than raising, and a render
that reaches one says so once, at ``WARNING``, naming the ceiling and what it
cost. They are warnings rather than the renderer's usual ``PERF`` budget
messages because they change the picture: a batch split or a ray-pool retry is
the memory model working as intended, but a truncated ray is transport that
never reached the pixel.

The counts are also on the render's :class:`~.RenderPlan`, so a script can
check without reading logs::

    result = Scene.save_video("scene")
    truncations = result.render_plan.truncations

    assert not truncations, truncations.as_dict()

:class:`~.TruncationCounts` has one field per ceiling --
``surfaces_per_ray``, ``shadow_lights``, ``sheet_layers``,
``dropped_continuations`` and ``closed_shell_ring`` -- plus ``total``. The
counts are cumulative over the whole render, except ``shadow_lights``, which is
a property of the scene rather than a tally of events and reports the worst
batch.

Every counter is unconditional, so **a zero is a measurement**: it says the
ceiling was watched and never reached, not that nothing was looking.
``dropped_continuations`` in particular should always read zero on the shipped
renderer -- every path that can lose a continuation ray retries its tile
instead -- and is counted so that a future change which breaks that cannot do
it quietly.

The two tessellation budgets in the table warn through Python's ``warnings``
module instead (a ``RuntimeWarning``), because they are decided while geometry
is built rather than while a frame is composited.


Not implemented at all
======================

Neither renderer does any of these, at any setting:

* **Global illumination** on the deterministic path. Color bleeding and indirect
  light need ``samples_per_pixel > 1``.
* **Caustics.**
* **Ambient occlusion**, in any form -- no SSAO pass, no AO map.
* **Volumetrics**: fog, god rays, participating media, smoke, subsurface
  scattering.
* **Displacement mapping** or height-map tessellation. Geometry comes from the
  mob; a texture never moves a vertex.
* **Wireframe rendering.**
* **Auxiliary output passes.** There is no depth buffer, normal buffer, object
  ID buffer, motion-vector buffer or cryptomatte to write out -- only the shaded
  RGB(A) frame.
* **Temporal anti-aliasing** or temporal accumulation. Denoising exists, but
  only for the path tracer (``denoise``; see
  `Which renderer runs your scene`_) -- the deterministic renderer has no
  noise to remove.
* **A "physical" light-transport mode.** The unwired physical-mode Monte Carlo
  kernel and the two settings only it read (``light_intensity`` and
  ``ambient_light``) have been deleted, as has ``indirect_bounce_strength``,
  whose color-bleed hack belonged to the replaced Monte Carlo megakernel.
  Scale a light with its own ``intensity=`` and add an
  :class:`~algan.rendering.lights.AmbientLight` for ambient.

For the path tracer's share of this list -- caustics, adaptive sampling,
temporal stability, volumes and subsurface scattering -- the engineering side
(why each is absent, and what implementing it would take under the renderer's
reproducibility and kernel contracts) is written up in
``algan/rendering/raytracing/DESIGN_path_tracer_roadmap.md``, which is the
plan of record for that remaining scope.


Determinism and reproducibility
===============================

This section is about the **deterministic** renderer (``samples_per_pixel ==
1``). The path tracer is a Monte Carlo estimator and promises only that it
converges to the right image -- not that two runs of it agree. Nothing below
applies to it.

The deterministic renderer is designed to render the same frame the same way
every time, and it does on the paths the project measures. Two caveats:

* **Across machines and devices, no.** Frames rendered on CPU and on CUDA
  differ, and frames rendered on two different CPUs have been measured to differ
  as well. Curved surfaces are the sensitive part: their tessellation level is
  chosen per patch per frame from a projected error, so a patch sitting on a
  level boundary can round either way depending on the hardware evaluating it,
  and one level change moves every microtriangle in that patch. Do not diff a
  render against one produced elsewhere and expect byte-identity; the project's
  own pixel baselines are kept per device for this reason.
* **Across batch windows, approximately.** Rendering the same scene in different
  frame-batch sizes -- which happens automatically when available memory
  changes -- can move a pixel by a channel value or two, because rate functions
  are evaluated over different windows.


See Also
========

- :ref:`renderer-capabilities` -- the deterministic/path-tracer feature table.
- :doc:`lighting_and_shadows` -- the light types and how shadows are enabled.
- :doc:`reflections_and_glass` -- setting up mirrors, metals and glass.
- :doc:`shaders_and_materials` -- the material classes in full.
- :doc:`images_and_textures` -- how to get a texture onto a mob.
- :doc:`cameras` -- the projection model, and the near-orthographic
  approximation named above.
- :doc:`backgrounds_and_post_processing` -- the anti-aliasing settings this page
  bounds.
- :doc:`settings` -- where the settings named on this page live, and what
  ``experimental`` means.
- :doc:`performance_and_quality` -- what each of these features costs.
