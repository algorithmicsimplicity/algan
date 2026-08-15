======================
Importing 3-D Models
======================

:class:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob` loads a 3-D model file -- geometry, UVs, textures, PBR
materials, node hierarchy and rigid animation -- and gives you an ordinary Algan
:class:`~algan.animatable_base.mob.Mob`.

.. algan-doc-check: skip -- needs dragon.glb, which does not ship with the docs

.. code-block:: python

    from algan import *

    model = ThreeDModelMob('dragon.glb', normalize=True).scale(3).spawn()
    with Seq(run_time=4, rate_func=rate_funcs.identity):
        model.rotate(360, UP)

    Scene.save_video()

File Formats
============

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Formats
     - Requirements
   * - ``.glb``, ``.gltf``, ``.obj``, ``.ply``, ``.stl``, ``.dae``, ``.off``
     - Loaded through ``trimesh``, which is pure Python. Works with a standard
       Algan install.
   * - ``.fbx``
     - Loaded through ``pyassimp``, which needs the **native** ``assimp``
       library as well as the Python bindings.

glTF / glB is the format to prefer: it is the best supported, needs no extra
install, and carries PBR materials and embedded textures.

To use FBX, install the extra and the native library:

.. code-block:: bash

    pip install "algan[fbx]"

``pyassimp`` is only a ``ctypes`` wrapper, so the native ``assimp`` library must be
installed separately -- ``conda install -c conda-forge assimp``,
``apt install libassimp5``, ``brew install assimp``, or an ``assimp*.dll`` on
``PATH`` on Windows. Algan raises an error naming the missing piece if it cannot
find it.

Model paths are resolved against the working directory and then your script's
directory, like every other Algan asset.

Scale and Position
==================

Model files use wildly inconsistent unit scales -- one file's "1" is a metre,
another's is a centimetre. ``normalize=True`` recentres the model and uniformly
scales it to fit a box of ``normalize_size`` (default 2), which makes an unfamiliar
asset usable immediately:

.. code-block:: python

    # Predictable size regardless of the file's units.
    model = ThreeDModelMob('asset.glb', normalize=True, normalize_size=3).spawn()

    # Or take the file's own scale and adjust by hand.
    model = ThreeDModelMob('asset.glb').scale(0.01).spawn()

After that it is a normal Mob: :meth:`~algan.animatable_base.mob.Mob.scale`,
:meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`,
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` and
:meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.scale_to_height` all behave as usual.

Materials and Textures
======================

By default Algan applies each imported material's PBR parameters -- metalness,
roughness, emissive -- as a
:class:`~algan.rendering.shaders.materials.MeshStandardMaterial`, so imported meshes shade
with Cook-Torrance GGX and respond correctly to your lighting. Diffuse texture maps
and tangent-space normal maps are loaded and applied too.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Argument
     - Effect
   * - ``pbr_materials``
     - Apply each material's PBR parameters. Default True; False keeps Algan's
       default lit shader.
   * - ``load_textures``
     - Load diffuse texture maps. Default True; False (or a failed load) falls
       back to the material's flat base colour.
   * - ``normal_maps``
     - Apply tangent-space normal maps. Default True. Requires per-vertex UVs.
   * - ``smooth_normals``
     - Use the mesh's authored per-vertex normals. Default True; False derives
       flat per-face normals at render time, for a low-poly look.

.. note::

    Batches carrying a normal map are routed automatically to the general wavefront
    tracer, which supports per-fragment normal perturbation. That is a performance
    consideration, not something you have to configure -- but see
    :doc:`performance_and_quality` if an imported model renders more slowly than you
    expect.

Because the materials land on the meshes as ordinary Algan materials, their
properties are animatable attributes like any other:

.. code-block:: python

    model = ThreeDModelMob('robot.glb', normalize=True).spawn()
    with Seq(run_time=3):
        model.roughness = 0.1      # polish the whole model

Working With Parts
==================

An imported model keeps its node hierarchy, so you can reach in and animate a single
part:

.. code-block:: python

    model = ThreeDModelMob('robot.glb', normalize=True).spawn()

    print(model.node_names)              # what's in the file

    arm = model.get_part('LeftArm')      # one TriangleMesh, or a list of them
    with Seq(run_time=2):
        arm.rotate(45, OUT)
        arm.color = RED

:attr:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob.node_names` lists the nodes that carry geometry, and
:meth:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob.get_part` returns the mesh Mob (or list of them) for a named
node. A part is a normal Mob, so everything in
:doc:`../new_user_tutorials/basic_animations` applies -- and because it is a child of
the model, moving the model still carries it along (see
:doc:`../new_user_tutorials/child_mobs`).

Playing Baked Animations
========================

If the file carries animation clips, :meth:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob.play_animation` records
one onto Algan's timeline:

.. code-block:: python

    model = ThreeDModelMob('walking.glb', normalize=True).spawn()

    print(model.animation_names)         # available clips

    model.play_animation('Walk', run_time=4, loop=2)

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Meaning
   * - ``name``
     - Which clip. Defaults to the first one.
   * - ``run_time``
     - Seconds per loop. Defaults to the clip's authored duration.
   * - ``loop``
     - How many times to repeat it.
   * - ``fps``
     - Sampling rate used when baking. Higher is smoother for fast rotation,
       because poses are interpolated linearly in between. Default 30.
   * - ``rate_func``
     - Easing. Defaults to ``rate_funcs.identity``, which is what you want --
       a walk cycle should not ease in and out.

Because the clip is recorded on the timeline like any other animation, it composes
with animation contexts -- you can play a walk cycle while simultaneously moving the
model across the frame:

.. code-block:: python

    with Sync(run_time=4):
        model.play_animation('Walk', run_time=4)
        model.move(RIGHT * 6)

.. important::

    **Rigid node animation only.** Algan bakes each clip by evaluating the animated
    node transforms and composing them down the hierarchy, so a part that
    translates, rotates or scales plays back correctly. Skeletal *skinning* -- where
    vertices are weighted to several bones and deform between them -- is not applied.
    A model animated by moving rigid parts works; one animated by deforming a
    continuous skinned mesh will move at the node level only.

:meth:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob.bake_animation` exposes the same computation without
touching the scene, returning the sample times and per-frame geometry, if you want
to inspect or post-process the poses yourself.

Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Likely cause
   * - Nothing appears
     - The model's units. Try ``normalize=True``.
   * - Faceted where it should be smooth
     - The file has no authored normals; they are being derived per face.
   * - Untextured / flat colour
     - A texture failed to load, or the mesh has no UVs.
   * - Black or very dark
     - A metallic material with nothing to reflect. Add an environment map --
       see :doc:`lighting_and_shadows`.
   * - Very slow
     - Triangle count, or normal maps forcing the wavefront tracer. See
       :doc:`performance_and_quality`.
   * - An FBX raises on load
     - The native ``assimp`` library is missing. The error message names what to
       install.

See Also
========

- :doc:`images_and_textures` -- texturing surfaces you build yourself.
- :doc:`shaders_and_materials` -- overriding an imported model's materials.
- :doc:`lighting_and_shadows` -- lighting an imported asset.
