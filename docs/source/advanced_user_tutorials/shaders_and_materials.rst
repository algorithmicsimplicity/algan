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
attributes are not methods of the :class:`.Mob` class. When we called the :meth:`~.Mob.set_shader`
method, it read the the function signature of the shader and realised that there were
2 additional arguments named smoothness and metallicness, so it automatically
added those as animatable attributes to our mob.

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
