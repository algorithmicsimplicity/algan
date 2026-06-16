=====================
Shaders and Materials
=====================

In rendering pipelines, the *shader* is responsible for determining
how the brightness and color of objects changes when light is cast on them.

.. important::

    Currently shaders are only implemented for 3-D objects! 2-D objects
    do not interact with lighting.

By default, Algan will use the :func:`.default_shader` function to shade Mobs.
This function implements a simplified diffusion shader, which does not depend
on any material properties. This means that all objects will interact
with light in the same way.

If you want to get more realistic lighting, you can use more sophisticated
shaders that take into consideration the material properties of the object.

Using a Physics-based Rendering Shader
======================================

Algan also provides a basic implementation of a physics-based shader
in the :func:`.basic_pbr_shader` function. This function takes an additional
2 parameters as input compared to the default shader: metallicness and
smoothness. This function simulates how light interacts with a surface made
of metal vs non-metal and smooth vs rough texture to compute diffuse
and specular lighting effects.

You can make a mob use this shader with the :meth:`~.Mob.set_shader` method.
Let's look at an example.

.. algan:: SetShader

    from algan import *
    from algan.rendering.shaders.pbr_shaders import basic_pbr_shader

    with Sync():
        mob1 = Sphere().move(LEFT*2).spawn()
        mob2 = Sphere().move(RIGHT*2).set_shader(basic_pbr_shader).spawn()

    with Seq(run_time_unit=5):
        mob2.smoothness = 0
        mob2.metallicness = 1
        mob2.smoothness = 1
        mob2.metallicness = 0

    render_to_file()

.. important::

    You must use :meth:`~.Mob.set_shader` before spawning the mob! Once spawned,
    the shader cannot be changed.

In this example, the first mob (left) uses the default shader, and the second (right)
uses the PBR shader,
with a range of different material properties. Note that the `smoothness` and `metallicness`
attributes are not properties of the :class:`.Mob` class. When we called the :meth:`~.Mob.set_shader`
method, it read the the function signature of the shader and realised that there were
2 additional arguments named smoothness and metallicness, so it automatically
added those as animatable attributes to our mob.

Materials (Three.js-style)
==========================

For a more comprehensive workflow, Algan provides a set of **material** classes
that mirror the `Three.js <https://threejs.org/>`_ mesh materials -- the same
material types, property names, and default settings. Instead of picking a shader
function and animating loose parameters, you configure a material object and apply
it with :meth:`~.Mob.set_material`.

.. algan:: SetMaterial

    from algan import *

    with Sync():
        mob1 = Sphere().move(LEFT*2).set_material(
            MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.2)).spawn()
        mob2 = Sphere().move(RIGHT*2).set_material(
            MeshPhongMaterial(color=BLUE, shininess=80)).spawn()

    with Seq(run_time_unit=5):
        mob1.roughness = 1.0
        mob1.metalness = 0.0

    render_to_file()

As with :meth:`~.Mob.set_shader`, :meth:`~.Mob.set_material` **must** be called
before the mob is spawned. Applying a material registers the material's
numeric/colour properties as animatable attributes on the mob, so you can animate
them afterwards (``mob1.roughness = 1.0`` above), exactly like the built-in
attributes.

Available materials
-------------------

All of the Three.js mesh materials are provided, with matching default settings:

.. list-table::
    :header-rows: 1

    * - Material
      - Lighting
      - Key properties (defaults)
    * - :class:`MeshBasicMaterial`
      - Unlit (flat colour)
      - ``color``
    * - :class:`MeshLambertMaterial`
      - Diffuse (Lambert)
      - ``color``, ``emissive`` (0x000000), ``emissiveIntensity`` (1)
    * - :class:`MeshPhongMaterial`
      - Blinn-Phong specular
      - ``specular`` (0x111111), ``shininess`` (30), ``emissive``
    * - :class:`MeshStandardMaterial`
      - PBR metalness/roughness
      - ``roughness`` (1), ``metalness`` (0), ``emissive``, ``envMapIntensity`` (1)
    * - :class:`MeshPhysicalMaterial`
      - PBR + clearcoat/sheen
      - adds ``clearcoat`` (0), ``ior`` (1.5), ``specularIntensity`` (1),
        ``sheen`` (0), ``transmission`` (0), ...
    * - :class:`MeshToonMaterial`
      - Cel / banded diffuse
      - ``color``, ``bands`` (3), ``emissive``
    * - :class:`MeshNormalMaterial`
      - Normal-as-colour
      - ``flatShading`` (False)
    * - :class:`MeshMatcapMaterial`
      - Material capture (approx.)
      - ``color``
    * - :class:`MeshDepthMaterial`
      - Camera-distance grayscale
      - ``near`` (0.1), ``far`` (100)

Colours accept hex ints (``0xff0000``), hex strings (``"#ff0000"``), Algan colour
constants (``RED``), or RGB tuples. Following Three.js, a material's ``color``
default is white and drives the mesh's base colour (overriding a shape's own
default colour).

.. note::

    The animatable attribute names on the mob use Python ``snake_case`` (e.g.
    ``mob.emissive_intensity``, ``mob.metalness``), while the material
    constructors accept the Three.js ``camelCase`` names (e.g.
    ``MeshStandardMaterial(emissiveIntensity=2)``).

.. important::

    **Limitations.** Algan shades *per vertex* and has no UV / image-sampling
    pipeline, so every texture / image-based property (``map``, ``normalMap``,
    ``roughnessMap``, ``envMap``, ``matcap``, ``gradientMap``, ...) is accepted
    for API parity but **not sampled** -- a warning is emitted when one is set.
    ``wireframe``, ``vertexColors`` and non-default ``side`` are likewise
    unsupported. The matcap, normal and depth materials use approximations
    (matcap has no image; normals are world-space rather than view-space). For
    the best results, light the scene with a single point light.

Writing Custom Shaders
======================

If you want to make your own shader, all you need to do is implement the function for it.
Take a look at the source code for :func:`.default_shader` and :func:`.basic_pbr_shader` functions
to see how this can be done in Pytorch. If you make your own shader function,
it must have the same signature as the default shader, plus any additional shader
parameters you require. Even if you don't use them, your function signature must
still declare the default parameters. Any new parameters you introduce beyond those
in the :func:`.default_shader` will be automatically added as animatable attributes to your mobs
when you set this function as their shader.

Once you've defined your shader function, simply use :meth:`~.Mob.set_shader` as in the above example.
You can then animate any shader parameters just as you would any of the built in
animatable attributes.

.. note::

    During rendering, mobs with different shaders will be batched separately.
    This means you should reuse the same function definition where possible,
    as it will allow mobs to be batched more effectively.
