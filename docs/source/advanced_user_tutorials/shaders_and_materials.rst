=====================
Shaders and Materials
=====================

A *shader* decides how an object's brightness and colour change when light falls
on it. Algan gives you three levels of control:

1. **Materials** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`) -- Three.js-style material objects.
   Start here; this is the documented workflow.
2. **Vertex shaders** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`) -- a PyTorch function evaluated at
   each vertex.
3. **Fragment shaders** (:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`) -- a Taichi pipeline
   evaluated at each rendered fragment, in-kernel.

.. important::

    Shading applies to **3-D objects** only. Flat 2-D shapes and text are drawn
    in their own colour and do not interact with lighting.

    All three of ``set_material``, ``set_shader`` and ``set_fragment_shader`` must
    be called **before** the Mob is spawned.

Materials
=========

A material bundles a lighting model with its parameter values, so
``MeshStandardMaterial(metalness=1.0, roughness=0.2)`` gives you polished metal
without your having to know which shader that is.

.. algan:: MaterialsSetMaterial

    from algan import *

    with Sync():
        mob1 = Sphere().move(LEFT * 2).set_material(
            MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.2)).spawn()
        mob2 = Sphere().move(RIGHT * 2).set_material(
            MeshPhongMaterial(color=BLUE, shininess=80)).spawn()

    with Seq(run_time_unit=5):
        mob1.roughness = 1.0
        mob1.metalness = 0.0

    Scene.save_video()

Applying a material registers its numeric and colour properties as **animatable
attributes** on the Mob, so ``mob1.roughness = 1.0`` above animates exactly like
``mob1.color`` or ``mob1.location`` would. That is the whole point of the material
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
      - Unlit (flat colour)
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
      - Normal-as-colour
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
you constructed yourself::

    desk = Prism().set_material(WOOD).spawn()
    lens = Sphere(color=BLUE_A).set_material(GLASS).spawn()

Neutral presets preserve the Mob's existing colour, while naturally coloured
presets such as ``WOOD`` and ``COPPER`` supply a representative base colour.
They configure PBR surface response and a flat base colour only; they do not add
wood grain, stone detail or other texture maps.

Colours and naming
------------------

Colours accept hex ints (``0xff0000``), hex strings (``"#ff0000"``), Algan colour
constants (``RED``), or RGB tuples.

.. note::

    Algan deliberately deviates from Three.js in one place: a material's ``color``
    defaults to ``None``, meaning "leave the Mob's own colour alone", where Three.js
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
triangle corner and the resulting colour interpolated across the face. That is fast
and looks good on the finely-tessellated curved surfaces typical of Algan scenes.

For lighting that varies smoothly *within* a face -- crisp specular highlights, or
smooth shading on a coarse mesh -- use per-fragment shading, which the
deterministic renderer evaluates in-kernel at every ray hit. It is on by default
(``SETTINGS.raytracing.experimental.fragment_shading``), and
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader` forces it on for any scene the Mob appears in.

Four materials have no in-kernel implementation at all and are therefore always
baked into vertex colours before the frame renders:
:class:`~algan.rendering.shaders.materials.MeshToonMaterial`,
:class:`~algan.rendering.shaders.materials.MeshNormalMaterial`,
:class:`~algan.rendering.shaders.materials.MeshMatcapMaterial` and
:class:`~algan.rendering.shaders.materials.MeshDepthMaterial`. That bake sees
only a plain :class:`~.PointLight` and never receives a shadow, so applying one
in a scene that also has a directional, ambient, hemisphere, spot or rect-area
light, an environment map, or ``shadows=True``, warns and names what is being
dropped -- at the ``set_material`` call, and again once per render for the
lights spawned after it. See :doc:`renderer_limitations`.

For full physically-based light transport -- true global illumination rather than
direct lighting plus deterministic bounces -- switch to the Monte Carlo path tracer
by raising the sample count:

.. code-block:: python

    from algan import *

    SETTINGS.raytracing.set(samples_per_pixel=64)

    Sphere().set_material(MeshStandardMaterial(metalness=1.0, roughness=0.2)).spawn()
    Scene.save_video()

Under path tracing the Mob's colour is treated as raw *albedo* and all illumination
comes from the scene's lights, emissive materials and the environment map, so the
result differs from the default preview -- that is the point. It is also
dramatically slower; see :doc:`performance_and_quality`.

Texture maps
------------

The material classes accept Three.js's image-based property slots (``map``,
``normalMap``, ``roughnessMap``, ``envMap``, ``matcap``, ``gradientMap``, ...) for
API parity, but **do not sample them** -- a warning is emitted when one is set.
``wireframe``, ``vertexColors`` and non-default ``side`` are likewise unsupported,
and the matcap, normal and depth materials use documented approximations (matcap
has no image; normals are world-space rather than view-space).

Texturing in Algan goes through :class:`~.Surface` instead, which takes
``color_texture``, ``roughness_texture``, ``reflectivity_texture``,
``refractive_index_texture``, ``normal_texture`` and ``glow_texture`` and samples
them bilinearly per fragment inside the ray tracing kernel. See
:doc:`images_and_textures`.

Vertex Shaders
==============

A shader is just a function, and
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` installs one.
Algan ships
:func:`~algan.rendering.shaders.pbr_shaders.default_shader` (a simplified diffuse
model with no material properties, which is what an unconfigured Mob uses) and
:func:`~algan.rendering.shaders.pbr_shaders.basic_pbr_shader`,
which adds ``smoothness`` and ``metallicness``.

The interesting part is how parameters are handled. ``set_shader`` inspects the
function's signature, sees which parameters come *after* the ones
:func:`~algan.rendering.shaders.pbr_shaders.default_shader` declares, and registers those as animatable attributes on
the Mob. So installing :func:`~algan.rendering.shaders.pbr_shaders.basic_pbr_shader` gives you ``mob.smoothness`` and
``mob.metallicness`` to animate, without either being a predeclared Mob attribute.

To write your own, match :func:`~algan.rendering.shaders.pbr_shaders.default_shader`'s signature and append your own
parameters:

.. code-block:: python

    def my_shader(memory, vertex_location, vertex_normal, albedo_color,
                  camera_location, light_origin, light_color,
                  light_intensity, ambient_light_intensity,
                  banding=4.0):
        # ... torch operations returning a colour per vertex ...
        return color

    mob.set_shader(my_shader)   # before spawning
    mob.banding = 8.0           # now animatable

Every parameter of :func:`~algan.rendering.shaders.pbr_shaders.default_shader` must be declared even if you ignore it.
Read the source of :func:`~algan.rendering.shaders.pbr_shaders.default_shader` and
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
right -- each stage receives the previous stage's output colour:

.. code-block:: python

    from algan import *
    from algan.rendering.shaders.fragment_shaders import cosine_color
    from algan.rendering.shaders.material_shaders import phong_shader

    mob.set_fragment_shader([cosine_color, phong_shader])

That recolours each fragment with a cosine wave and then lights the result with
Blinn-Phong. As with vertex shaders, each stage's parameters become animatable
attributes on the Mob (duplicate names across stages are suffixed).

A custom stage is a Taichi ``@ti.func`` plus its parameter specs -- see
``cosine_color`` in ``algan/rendering/shaders/fragment_shaders.py`` for the
template.

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
``STAGE_DEFAULT``      Algan's built-in shading -- what a Mob uses when you set
                       no material. Resolved from ``default_shader``.
``STAGE_UNLIT``        No lighting: the fragment keeps its own colour. Resolved
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

and naming the stage is what lets you put a recolouring stage before the light:

.. code-block:: python

    mob.set_fragment_shader([cosine_color, STAGE_STANDARD, fresnel_rim])

Shipped Stage Looks
-------------------

``algan.rendering.shaders.fragment_stage_library`` collects ready-made stages,
exported by ``from algan import *``. They are **additive** -- each adds to the
colour it is handed rather than replacing it -- so they layer over a lit base:

===============  ==============================================================
Stage            What it adds
===============  ==============================================================
``fresnel_rim``  A rim light: ``rim_color * rim_gain * (1 - |N.V|) ** rim_power``.
                 ``rim_power`` sets how tightly it hugs the silhouette.
``glass_ball``   The studio glass-ball edge -- two Fresnel lobes, a Gaussian
                 silhouette ring and two screen-space specular blobs.
===============  ==============================================================

.. code-block:: python

    from algan import *

    ball = Sphere(radius=1, color=BLUE_E)
    ball.set_fragment_shader([standard_shader, fresnel_rim])
    ball.rim_color = (0.40, 0.90, 1.00)
    ball.rim_power = 3.0
    ball.spawn()

Stage parameters are plain numbers and tuples of the width the stage declares --
``rim_color`` is a width-3 RGB triple. Assigning a five-channel Algan colour
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

- :doc:`lighting_and_shadows` -- the lights these materials respond to.
- :doc:`reflections_and_glass` -- what ``metalness``, ``roughness``,
  ``transmission`` and ``ior`` actually do to rays.
- :doc:`images_and_textures` -- per-texel material properties.
- :doc:`extending_algan` -- adding new render primitives.
