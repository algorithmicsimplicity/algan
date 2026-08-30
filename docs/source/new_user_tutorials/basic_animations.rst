================
Basic Animations
================

In Algan you build animations by creating :class:`~algan.animatable_base.mob.Mob` s
and then changing them. By default, changes
you make to a mob are animated over a one second period. More complex
animation behaviour will be covered in the later :doc:`combining_animations` tutorial.

Changing Animatable Attributes
==============================

Every :class:`~algan.animatable_base.mob.Mob` has these animatable attributes:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Attribute
     - Shape
     - Meaning
   * - :attr:`~algan.animatable_base.mob.Mob.location`
     - 3 floats
     - Where the Mob sits in 3-D space. New Mobs start at ``ORIGIN``.
   * - :attr:`~algan.animatable_base.mob.Mob.basis`
     - 9 floats
     - Orientation *and* scale, as three basis vectors. Change it with
       :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` /
       :meth:`~algan.animatable_base.mob.Mob.scale`, not by hand.
   * - :ref:`color <reference-mob-color>`
     - :class:`~algan.constants.color.Color`
     - The Mob's main color.
   * - :ref:`glow <reference-mob-glow>`
     - float
     - How strongly the mob glows. ``0`` is off.
   * - :ref:`opacity <reference-mob-opacity>`
     - float
     - How see-through the mob is. ``1`` is opaque, ``0`` is invisible.

These attributes are special: **assigning to one performs a 1-second animation**
that interpolates from the old value to the new value.

.. algan:: BasicChangingAttributes

    from algan import *

    circle = Circle().spawn()

    circle.location = circle.location + UP * 2
    circle.color = GREEN
    circle.location = circle.location + DOWN + RIGHT * 2
    circle.glow = 0.5
    circle.opacity = 0.0

    Scene.save_video()

.. important::

    **Reading an animatable attribute gives you a copy.** ``circle.location``
    hands back a copy of the value on the timeline, so editing that copy in place
    (e.g. ``circle.location[0] = 1`` ) changes nothing and records nothing. The
    write goes into the copy and is thrown away silently.

    Assignment is what Algan records, so both ``circle.location = circle.location
    + UP`` and ``circle.location += UP`` work.

A note on colors
----------------

An Algan :class:`~algan.constants.color.Color` carries five components: red, green, blue, glow and
opacity, in that order. So glow and opacity can be set either on the Mob or baked
into the color you assign:

.. algan:: BasicColorComponents

    from algan import *

    circle = Circle().spawn()

    circle.color = BLUE                     # leaves glow and opacity alone
    circle.opacity = 0.5                    # ... or set them separately

    circle.color = GREEN.set_glow(0.5)   # ... or together, in one animation

    Scene.save_video()

Algan ships the full Manim color palette: ``RED``, ``BLUE_E``, ``TEAL_A``, ``GOLD`` and so
on, plus ``WHITE``, ``BLACK`` and ``TRANSPARENT``.

.. seealso::

    * :doc:`../advanced_user_tutorials/images_and_textures` -- painting a
      gradient or an image across a shape instead of one flat color.
    * :doc:`../advanced_user_tutorials/backgrounds_and_post_processing` -- what
      ``glow`` actually does, and the bloom pass that makes it visible.

Mob Methods
===========

:class:`~algan.animatable_base.mob.Mob` s also have a bunch of common operations built in to them as methods.
Most of the time, these are what you will use.

.. algan:: BasicMobMethods

    from algan import *

    pentagon = RegularPolygon(5).spawn()
    pentagon.move(RIGHT * 2)
    pentagon.rotate(360, OUT)
    pentagon.rotate(360, UP)
    pentagon.rotate(360, OUT, about=ORIGIN)
    pentagon.scale(2)
    pentagon = pentagon.become(Circle(add_to_scene=False))
    pentagon.wait(2)

    Scene.save_video()

The methods used above:

* :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move` translates the Mob by a vector: ``pentagon.move(RIGHT)`` slides it
  one unit right. To move to an absolute point instead, use
  :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`.
* :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` turns the Mob about an axis through its own centre.
  Passing ``about`` turns it about *that* point instead, which sweeps the
  Mob around in an arc: ``pentagon.rotate(180, OUT, about=ORIGIN)`` swings it
  half way around the origin.
* :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` morphs the Mob into a different Mob. It returns the
  resulting Mob, so assign it back: ``pentagon = pentagon.become(Circle(add_to_scene=False))``.
  Build the target with ``add_to_scene=False``: it is only there to say what shape
  to become, and is never itself drawn on screen. Without that flag Algan registers it as a
  Mob you meant to show, and warns that you never spawned it.
* :meth:`~algan.animatable_base.mob.Mob.scale` grows or shrinks the Mob: ``pentagon.scale(2)`` doubles its size.
* :meth:`~algan.animatable_base.animatable.Animatable.wait` holds the Mob still: ``pentagon.wait(2)`` leaves two
  seconds of nothing happening. ``Scene.wait(2)`` also does the same thing.

The :class:`~algan.animatable_base.mob.Mob` reference lists every method.

.. seealso::

    * :doc:`../advanced_user_tutorials/positioning_and_layout` -- every
      placement and sizing method, and when to reach for which.
    * :doc:`../galleries/built_in_animations` -- ready-made animations for
      drawing attention, moving along a path and transforming a diagram.
    * :doc:`../advanced_user_tutorials/custom_animations` -- writing an
      animation of your own with
      :func:`~algan.animatable_base.animatable.animated_function`,
      for anything the built-ins do not cover.
