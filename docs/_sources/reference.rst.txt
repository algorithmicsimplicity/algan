Reference Manual
================

This reference manual details modules, functions, and variables included in
Algan, describing what they are and what they do.  For learning how to use
Algan, see :doc:`tutorials/index`.  For a list of changes since the last release, see
the :doc:`changelog`.

.. warning:: The pages linked to here are currently a work in progress.

Inheritance Graphs
------------------

Animation
**********

Animations
**********

.. inheritance-diagram::
   algan.animation.animatable
   algan.mobs.mob
   :parts: 1
   :top-classes: algan.animation.animatable.Animatable

Cameras
*******

.. inheritance-diagram::
   algan.rendering.camera
   :parts: 1
   :top-classes: algan.rendering.camera.Camera

Mobs
********

.. inheritance-diagram::
   algan.mobs.surfaces
   algan.mobs.text
   algan.mobs.shapes_2d
   algan.mobs.bezier_circuit
   :parts: 1
   :top-classes: algan.mobs.mob.Mob

Scenes
******

.. inheritance-diagram::
   algan.scene
   :parts: 1
   :top-classes: algan.scene.Scene


Module Index
------------

.. toctree::
   :maxdepth: 3

   reference_index/mobs
   reference_index/animation
   reference_index/rendering
   reference_index/scenes
   reference_index/utils
