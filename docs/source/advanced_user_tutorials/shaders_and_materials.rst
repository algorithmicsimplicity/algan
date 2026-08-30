=====================
Shaders and Materials
=====================

A *shader* decides how an object's brightness and color change when light falls
on it. Algan gives you three levels of control:

1. **Materials** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`) -- Three.js-style material objects.
   Start here; this is the documented workflow.
2. **Vertex shaders** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`) -- a PyTorch function evaluated at
   each vertex.
3. **Fragment shaders** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`) -- a Taichi pipeline
   evaluated at each rendered fragment, in-kernel.

.. important::

    Shading applies to **3-D objects** only. Flat 2-D shapes and text are drawn
    in their own color and do not interact with lighting.

    All three of ``set_material``, ``set_shader`` and ``set_fragment_shader`` must
    be called **before** the Mob is spawned.

Materials
=========

A material bundles a lighting model with its parameter values, so
``MeshStandardMaterial(metalness=1.0, roughness=0.2)`` gives you polished metal
without your having to know which shader that is.

What an unconfigured Mob gets
-----------------------------

A 3-D Mob that sets no material of its own is not unshaded: it renders as
:class:`~algan.rendering.shaders.materials.DiffuseMaterial`, Algan's default
material (Lambert diffuse plus emissive). That material is installed at import
as ``SETTINGS.style.default_material``, and
``SETTINGS.style.set(default_material=MeshStandardMaterial(roughness=0.3))``
replaces it scene-wide -- its parameter values then apply to every
material-less Mob, exactly as if each had been given the material explicitly.
Flat 2-D content (shapes, text, images) is drawn unlit and never consults the
setting; see :meth:`~algan.scene.Scene.use_manim_defaults` for the
Manim-compatible variant.

.. algan:: MaterialsSetMaterial

    from algan import *

    with Sync():
        metal = Sphere().move(LEFT * 2).set_material(
            MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.2)).spawn()
        plastic = Sphere().move(RIGHT * 2).set_material(
            MeshPhongMaterial(color=BLUE, shininess=80)).spawn()

    with Seq(duration_unit=5):
        metal.roughness = 1.0
        metal.metalness = 0.0

    Scene.save_video()

Applying a material registers its numeric and color properties as **animatable
attributes** on the Mob, so ``metal.roughness = 1.0`` above animates exactly like
``metal.color`` or ``metal.location`` would. That is the whole point of the material
workflow: you configure once and then animate the properties by name.

Available materials
-------------------

All of the Three.js mesh materials are provided, with matching default settings:

.. list-table::
    :header-rows: 1

    * - Material
      - Lighting
      - Key properties (defaults)
    * - :class:`~algan.rendering.shaders.materials.MeshBasicMaterial`
      - Unlit (flat color)
      - ``color``
    * - :class:`~algan.rendering.shaders.materials.MeshLambertMaterial`
      - Diffuse (Lambert)
      - ``color``, ``emissive`` (0x000000), ``emissiveIntensity`` (1)
    * - :class:`~algan.rendering.shaders.materials.MeshPhongMaterial`
      - Blinn-Phong specular
      - ``specular`` (0x111111), ``shininess`` (30), ``emissive``
    * - :class:`~algan.rendering.shaders.materials.MeshStandardMaterial`
      - PBR metalness/roughness
      - ``roughness`` (1), ``metalness`` (0), ``emissive``, ``envMapIntensity`` (1)
    * - :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial`
      - PBR + clearcoat/sheen/transmission
      - adds ``clearcoat`` (0), ``ior`` (1.5), ``specularIntensity`` (1),
        ``sheen`` (0), ``transmission`` (0), ...
    * - :class:`~algan.rendering.shaders.materials.MeshToonMaterial`
      - Cel / banded diffuse
      - ``color``, ``bands`` (3), ``emissive``
    * - :class:`~algan.rendering.shaders.materials.MeshNormalMaterial`
      - Normal-as-color
      - ``flatShading`` (False)
    * - :class:`~algan.rendering.shaders.materials.MeshMatcapMaterial`
      - Material capture (approx.)
      - ``color``
    * - :class:`~algan.rendering.shaders.materials.MeshDepthMaterial`
      - Camera-distance grayscale
      - ``near`` (0.1), ``far`` (100)

Only :class:`~algan.rendering.shaders.materials.MeshStandardMaterial` and
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` are true PBR
materials, and they are the two that drive ray transport -- reflections come from
``metalness`` / ``roughness``, and refraction from a transmissive
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial`'s ``transmission``
and ``ior``. There are no separate
Mob-level reflectivity or refractive-index setters; the material is the single
source of these. See :doc:`reflections_and_glass`.

Material presets
----------------

Common surfaces are available as ready-to-use constants: ``WOOD``, ``GLASS``,
``PLASTIC``, ``RUBBER``, ``CERAMIC``, ``STONE``, ``MIRROR``,
``BRUSHED_METAL``, ``CHROME`` and ``COPPER``. Apply them exactly like a material
you constructed yourself:

.. algan:: MaterialsPresets

    from algan import *

    with Off():
        desk = Prism(width=7, height=0.3, depth=4, color=GREY).move(DOWN * 1.6)
        desk.set_material(WOOD).spawn()
        lens = Sphere(radius=0.9, color=BLUE_A).set_material(GLASS).spawn()

    lens.move(RIGHT * 2)

    Scene.save_video()

Neutral presets preserve the Mob's existing color, while naturally colored
presets such as ``WOOD`` and ``COPPER`` supply a representative base color.
They configure PBR surface response and a flat base color only; they do not add
wood grain, stone detail or other texture maps.

Colors and naming
-----------------

Colors accept hex ints (``0xff0000``), hex strings (``"#ff0000"``), Algan color
constants (``RED``), or RGB tuples.

.. note::

    Algan deliberately deviates from Three.js in one place: a material's ``color``
    defaults to ``None``, meaning "leave the Mob's own color alone", where Three.js
    would default it to white and silently repaint the Mob. Pass ``color`` explicitly
    when you want the material to set it.

.. note::

    The animatable attribute names on the Mob use Python ``snake_case``
    (``mob.emissive_intensity``, ``mob.metalness``), while the material
    constructors accept the Three.js ``camelCase`` names
    (``MeshStandardMaterial(emissiveIntensity=2)``).

Vertex shading vs. fragment shading
-----------------------------------

By default, materials are evaluated **per vertex**: lighting is computed at each
triangle corner and the resulting color interpolated across the face. That is fast
and looks good on the finely-tessellated curved surfaces typical of Algan scenes.

For lighting that varies smoothly *within* a face -- crisp specular highlights, or
smooth shading on a coarse mesh -- use per-fragment shading, which the
deterministic renderer evaluates in-kernel at every ray hit. It is on by default
(``SETTINGS.raytracing.experimental.fragment_shading``), and
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader` forces it on for any scene the Mob appears in.

Every built-in material class shades per fragment in the render kernel, so all
of them respond to the full lighting rig -- every light type, shadows and an
environment map's diffuse contribution. Only a **custom per-vertex shader**
(:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` with
a plain function rather than a fragment stage) is still baked into vertex
colors before the frame renders. That bake sees only a plain
:class:`~.PointLight` and never receives a shadow, so using one in a scene that
also has a directional, ambient, hemisphere, spot or rect-area light, an
environment map, or ``shadows=True``, warns and names what is being dropped --
at the ``set_shader`` / ``set_material`` call, and again once per render for
the lights spawned after it. See :doc:`renderer_limitations`.

For full physically-based light transport -- true global illumination rather than
direct lighting plus deterministic bounces -- switch to the path tracer
by raising the sample count:

.. code-block:: python

    from algan import *

    SETTINGS.raytracing.set(samples_per_pixel=64)

    Sphere().set_material(MeshStandardMaterial(metalness=1.0, roughness=0.2)).spawn()
    Scene.save_video()

Under path tracing the Mob's color is treated as raw *albedo* and all illumination
comes from the scene's lights, emissive materials and the environment map, so the
result differs from the default preview -- that is the point. It is also
dramatically slower; see :doc:`performance_and_quality`.

Texture maps
------------

:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`
forwards the four image slots the renderer has a sampler for -- ``map``,
``normal_map``, ``roughness_map`` and ``metalness_map`` -- onto the geometry,
which is where Algan's texture pipeline lives. Each takes a file path or an
``[H, W, C]`` image, and is sampled bilinearly per fragment in the ray tracing
kernel:

.. code-block:: python

    Sphere().set_material(MeshStandardMaterial(map="earth.png",
                                               roughness_map="ocean_gloss.png"))

Following Three.js, ``roughness_map`` is read from the image's **green** channel
and ``metalness_map`` from its **blue** one, so a single packed
occlusion/roughness/metalness image drives both; a single-channel image is used
as-is.

.. important::

    A forwarded map is **static**. Unlike the scalar properties a material
    installs -- ``mob.roughness``, ``mob.metalness`` and the rest, which are
    animatable attributes -- a material property map is fixed once the Mob
    spawns, and setting one warns to that effect. The exception is ``map`` on a
    :class:`~.Surface`, which lands on the animatable
    :attr:`~algan.mobs.surfaces.surface.Surface.color_texture` and so warns not
    at all.

Sampling needs per-vertex UVs, which means a :class:`~.Surface` (and its
subclasses -- :class:`~.Sphere`, :class:`~.Cylinder`, :class:`~.Torus`,
:class:`~.ImageMob`, ...) or a
:class:`~algan.mobs.three_d_models.mesh.TriangleMesh` built with ``uvs``. On
anything else -- a :class:`~.Cube`, a :class:`~.Polyhedron` -- the maps are
ignored with a warning, and ignored *wholesale*: a Cube's faces cannot be
textured even though the decorative dot at each of its corners is a Sphere that
could be, and texturing the corners instead would be worse than refusing.

The remaining image slots (``env_map``, ``matcap``, ``gradient_map``,
``ao_map``, ``transmission_map``, ...) have no channel in the renderer. They are
still accepted so a Three.js material transcribes without edits, then dropped
with a warning naming them. ``wireframe``, ``vertexColors`` and non-default
``side`` are likewise unsupported, and the matcap, normal and depth materials use
documented approximations (matcap has no image; normals are world-space rather
than view-space).

Going through the geometry directly gives you more: :class:`~.Surface` takes
``color_texture``, ``roughness_texture``, ``reflectivity_texture``,
``refractive_index_texture``, ``normal_texture`` and ``glow_texture``, in the
``[W, H, C]`` ``(u, v)`` layout rather than as images. See
:doc:`images_and_textures`.

Vertex Shaders
==============

A shader is just a function, and
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` installs one.
Algan ships
:func:`~algan.rendering.shaders.pbr_shaders.basic_pbr_shader`,
which adds ``smoothness`` and ``metallicness`` to the lighting model. (What an
unconfigured 3-D Mob gets is decided by materials, not here: it renders as
:class:`~algan.rendering.shaders.materials.DiffuseMaterial`, Algan's default
material -- see `What an unconfigured Mob gets`_.)

Every shader follows one calling convention, whose reference is the named
constant ``SHADER_FIXED_PARAM_COUNT`` in
``algan/rendering/shaders/material_shaders.py``: a signature opens with nine
fixed parameters, and any parameters after those are the material's animatable
properties. ``set_shader`` inspects the function's signature, sees which
parameters come *after* the fixed nine, and registers those as animatable
attributes on
the Mob. So installing :func:`~algan.rendering.shaders.pbr_shaders.basic_pbr_shader` gives you ``mob.smoothness`` and
``mob.metallicness`` to animate, without either being a predeclared Mob attribute.

To write your own, declare the fixed parameters and append your own:

.. code-block:: python

    def my_shader(memory, vertex_location, vertex_normal, albedo_color,
                  camera_location, light_origin, light_color,
                  light_intensity, ambient_light_intensity,
                  banding=4.0):
        # ... torch operations returning a color per vertex ...
        return color

    mob.set_shader(my_shader)   # before spawning
    mob.banding = 8.0           # now animatable

Every fixed parameter must be declared even if you ignore it.
Read the source of ``basic_material_shader`` (in
``algan/rendering/shaders/material_shaders.py``) and
:func:`~algan.rendering.shaders.pbr_shaders.basic_pbr_shader` for
working implementations.

.. note::

    Mobs with different shaders are batched separately at render time. Reuse the
    same function object where you can -- defining the shader once and applying it
    to many Mobs batches much better than defining an equivalent function per Mob.

Fragment Shaders
================

The PyTorch shaders above run per vertex, before upload. The deterministic ray
tracer can instead shade **per fragment**, in-kernel: each ray hit evaluates a
pipeline of Taichi stages, so specular highlights stay crisp and coarse meshes
shade smoothly.

:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader` accepts a built-in material shader, a
:class:`~algan.rendering.shaders.fragment_shaders.FragmentStage`, or a **list** of these forming a pipeline run left to
right -- each stage receives the previous stage's output color:

.. algan:: MaterialsFragmentPipeline

    from algan import *
    from algan.rendering.shaders.fragment_shaders import cosine_color
    from algan.rendering.shaders.material_shaders import phong_shader

    ball = Sphere(radius=1.4, color=BLUE)
    ball.set_fragment_shader([cosine_color, phong_shader])
    ball.spawn()

    ball.rotate(180, UP)

    Scene.save_video()

That recolors each fragment with a cosine wave and then lights the result with
Blinn-Phong. As with vertex shaders, each stage's parameters become animatable
attributes on the Mob (duplicate names across stages are suffixed).

A custom stage is a Taichi ``@ti.func`` plus its parameter specs -- see
``cosine_color`` in ``algan/rendering/shaders/fragment_shaders.py`` for the
template.

A pipeline is compiled into the shade kernel, so the first render that uses one
pays a kernel compile for it (cached afterwards, like every other Algan kernel).
That cost is scoped to the scenes that actually use the shader: a scene rendered
from the same script without it compiles the ordinary kernel and reuses the
cached one.

The Built-in Lighting Stages
----------------------------

The lighting models themselves are stages too. Passing a vertex-shader function
such as ``phong_shader`` to
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
resolves it to the matching in-kernel stage, so you rarely name these directly --
but they are exported by ``from algan import *``, and you need them when a stage
has to sit in the *middle* of a pipeline rather than at the start:

=====================  ========================================================
Stage                  Lighting model
=====================  ========================================================
``STAGE_MANIM``        Manim's default 3-D lighting -- per light, a
                       ``0.5 * (n . to_light) ** 3`` offset, halved when the
                       surface faces away from the light and scaled by the
                       light's color. Resolved from ``manim_shader``;
                       installed as the default 3-D shading by
                       :meth:`~algan.scene.Scene.use_manim_defaults`.
``STAGE_UNLIT``        No lighting: the fragment keeps its own color. Resolved
                       from ``null_shader`` and ``basic_material_shader``, and
                       what :class:`~.MeshBasicMaterial` maps to.
``STAGE_LAMBERT``      Diffuse only. Resolved from ``lambert_shader``.
``STAGE_PHONG``        Blinn-Phong diffuse plus specular. Resolved from
                       ``phong_shader``.
``STAGE_STANDARD``     Metallic/roughness PBR. Resolved from
                       ``standard_shader``.
``STAGE_PHYSICAL``     PBR plus clearcoat, sheen and transmission. Resolved
                       from ``physical_shader``; it declares a wider parameter
                       block than the other five.
=====================  ========================================================

So these two lines mean the same thing:

.. code-block:: python

    mob.set_fragment_shader(phong_shader)
    mob.set_fragment_shader(STAGE_PHONG)

and naming the stage is what lets you put a recoloring stage before the light:

.. code-block:: python

    mob.set_fragment_shader([cosine_color, STAGE_STANDARD, fresnel_rim])

Shipped Stage Looks
-------------------

``algan.rendering.shaders.fragment_stage_library`` collects ready-made stages,
exported by ``from algan import *``. They are **additive** -- each adds to the
color it is handed rather than replacing it -- so they layer over a lit base:

===============  ==============================================================
Stage            What it adds
===============  ==============================================================
``fresnel_rim``  A rim light: ``rim_color * rim_gain * (1 - |N.V|) ** rim_power``.
                 ``rim_power`` sets how tightly it hugs the silhouette.
``glass_ball``   The studio glass-ball edge -- two Fresnel lobes, a Gaussian
                 silhouette ring and two screen-space specular blobs.
===============  ==============================================================

.. algan:: MaterialsFresnelRim

    from algan import *

    ball = Sphere(radius=1, color=BLUE_E)
    ball.set_fragment_shader([standard_shader, fresnel_rim])
    ball.rim_color = (0.40, 0.90, 1.00)
    ball.rim_power = 3.0
    ball.spawn()

    ball.rotate(180, UP)

    Scene.save_video()

Stage parameters are plain numbers and tuples of the width the stage declares --
``rim_color`` is a width-3 RGB triple. Assigning a five-channel Algan color
constant such as ``TEAL_A`` to it raises; use ``TEAL_A[..., :3]`` if you want to
derive one from the palette.

A rim light is an authoring control rather than a BSDF term, which is why it
lives here and not on
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` -- Three.js,
Unreal, Unity and Blender all keep it out of the physically-based material and
expose it through a shader graph or custom shader instead. Note that a
*physically* lit glass ball does not need these at all: give it
``MeshPhysicalMaterial(transmission=...)`` plus an environment map that is bright
in the directions the camera cannot see, and the bright rim falls out of
refraction (see :meth:`~algan.scene.Scene.set_environment_map`). The stages are
for when you want the look without authoring an environment, or want it stylised.

Custom Ray Bouncing (Scatter Stages)
------------------------------------

Each material pipeline also owns a **scatter function**, which decides how a ray
continues after shading a surface: pass through (transparency), mirror-bounce, or
split into a reflected and a refracted ray (glass). The default scatter implements
the standard opacity / metalness / Fresnel-glass behaviour; attach your own to any
stage to customise it:

.. code-block:: python

    FragmentStage(my_stage_func, my_param_specs, scatter=my_scatter_func)

See ``forced_mirror_scatter`` in ``fragment_shaders.py`` for a complete example, and
the scatter contract documented in
``algan/rendering/raytracing/shading_taichi.py``.

.. note::

    Custom scatter and normal-mapped lighting run inside the deterministic ray
    tracer's monolithic shade kernel, which is the only supported deterministic
    shade path. An older *sorted* material-dispatch pipeline (one GPU kernel per
    material, as in Blender Cycles) is no longer maintained and no longer works:
    ``SETTINGS.raytracing.experimental.set(wavefront_sort_materials=True)`` raises
    :class:`~algan.errors.UnsupportedFeatureError`.

    Custom fragment-shader pipelines are also a deterministic-renderer feature --
    see :ref:`renderer-capabilities`.

See Also
========

- :doc:`../new_user_tutorials/three_d_basics` -- the gentler introduction to
  materials.
- :doc:`lighting_and_shadows` -- the lights these materials respond to.
- :doc:`reflections_and_glass` -- what ``metalness``, ``roughness``,
  ``transmission`` and ``ior`` actually do to rays.
- :doc:`images_and_textures` -- per-texel material properties.
- :doc:`renderer_limitations` -- which materials are shaded per fragment, and
  which Three.js properties are accepted and ignored.
- :doc:`performance_and_quality` -- what distinct materials and shaders cost.
- :doc:`extending_algan` -- adding new render primitives.
