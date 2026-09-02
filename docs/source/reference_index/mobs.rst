====
Mobs
====

.. autosummary::
   :toctree: ../reference

   ~algan.mobs.group
   ~algan.mobs.text
   ~algan.mobs.image_mob
   ~algan.mobs.image_compat
   ~algan.mobs.manim_compat
   ~algan.mobs.opengl_compat
   ~algan.mobs.bezier_circuit
   ~algan.mobs.triangulated_bezier_circuit
   ~algan.mobs.shapes_2d
   ~algan.mobs.shapes_3d
   ~algan.mobs.point_cloud
   ~algan.mobs.surfaces.surface
   ~algan.mobs.surfaces.procedural_textures
   ~algan.mobs.three_d_models
   ~algan.mobs.three_d_models.mesh
   ~algan.mobs.three_d_models.model_mob
   ~algan.mobs.three_d_models.assimp_loader
   ~algan.mobs.three_d_models.gltf_loader
   ~algan.mobs.three_d_models.scene_data
   ~algan.mobs.manim_mob
   ~algan.mobs.numeric_display

The Manim Compatibility Layer
-----------------------------

``algan.mobs.manim_compat`` and its siblings above are the *implementation*. The public face
is :mod:`algan.manim`, reached as ``import algan.manim as mn`` and deliberately left out of
``from algan import *``: a name in ``mn.`` follows Manim's conventions where the same name at
the root follows Algan's (degrees vs radians, and Algan's stroke width vs Manim's, which is
twice it). Where Algan has no native implementation of a Manim class, a curated subset gets a
root spelling that converts and delegates -- see
:doc:`/manim_migration_guide` and :doc:`/advanced_user_tutorials/importing_from_manim`.

This is also why several class names below appear twice, once under ``manim_compat`` and once
under a native module: they are two different classes under two conventions, not one class
documented twice.

Delegated Manim Methods
-----------------------

Compatibility Mobs forward methods that are provided by the backing Manim
object. The most commonly used example is ``Axes.plot``:

.. _reference-manim-axes-plot:

.. py:method:: algan.mobs.manim_compat.Axes.plot

   Delegate ``plot`` to the backing Manim :class:`~algan.mobs.manim_compat.Axes` and convert the returned
   geometry into an Algan Mob.

SceneData Fields
----------------

The importer IR is a dataclass, so its fields are constructor parameters rather
than ordinary class members in autodoc. Keep stable anchors for the fields used
by the loader documentation.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.meshes

   The imported mesh records.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.materials

   The imported material records.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.nodes

   The imported node hierarchy.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.animations

   The imported node and skeletal animation records.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.unit_scale

   The source model's unit scale.

.. py:attribute:: algan.mobs.three_d_models.scene_data.SceneData.source_path

   The source model path.
