================
Basic Animations
================

In Algan you build animations by creating :class:`.Mob` s -- the objects that
appear on screen -- and then changing them. Every change you make is *recorded*
as an animation rather than applied instantly, so a script reads like a
description of what happens, in order, from top to bottom.

There are three ways to change a Mob, in increasing order of how much work they
ask of you:

1. **Assign to an animatable attribute** (``mob.color = BLUE``). Best for
   appearance.
2. **Call a Mob method** (``mob.move(RIGHT)``, ``mob.rotate(90, OUT)``). Best for
   motion and geometry -- these are the workhorses.
3. **Write an animated function** (:func:`~.animated_function`). For the rare
   animation Algan has no method for.

This tutorial covers all three.

Changing Animatable Attributes
------------------------------

Every :class:`.Mob` has these animatable attributes:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Attribute
     - Shape
     - Meaning
   * - :attr:`~.Mob.location`
     - 3 floats
     - Where the Mob sits in 3-D space. New Mobs start at ``ORIGIN``.
   * - :attr:`~.Mob.basis`
     - 9 floats
     - Orientation *and* scale, as three basis vectors. Change it with
       :meth:`~.Mob.rotate` / :meth:`~.Mob.scale`, not by hand.
   * - :attr:`~.Mob.color`
     - :class:`~.Color`
     - The Mob's main colour.
   * - :attr:`~.Mob.glow`
     - float
     - How much light the Mob bleeds into surrounding pixels. ``0`` is off.
   * - :attr:`~.Mob.opacity`
     - float
     - ``1`` is solid, ``0`` is invisible.

These attributes are special: **assigning to one records a 1-second animation**
that interpolates from the old value to the new one. Nothing else in your script
has to change.

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

    **Only out-of-place assignment is animated.** ``circle.location += UP`` and
    ``circle.location[0] = 1`` mutate the underlying tensor in place, which Algan
    cannot see and cannot record. Always write
    ``circle.location = circle.location + UP``.

    In practice you will reach for :meth:`~.Mob.move` instead, which does the
    arithmetic for you.

A note on colours
=================

An Algan :class:`~.Color` carries five components: red, green, blue, glow and
opacity, in that order. So glow and opacity can be set either on the Mob or baked
into the colour you assign:

.. code-block:: python

    circle.color = BLUE            # leaves glow and opacity alone
    circle.glow = 0.5              # ... or set them separately
    circle.opacity = 0.5

    circle.color = BLUE.set_opacity(0.5)   # ... or together, in one animation

A colour with a non-zero glow component emits light into nearby pixels (see
:doc:`../advanced_user_tutorials/backgrounds_and_post_processing`). Algan ships
the full Manim colour palette -- ``RED``, ``BLUE_E``, ``TEAL_A``, ``GOLD`` and so
on -- plus ``WHITE``, ``BLACK`` and ``TRANSPARENT``.

Mob Methods
-----------

Most animation is done with methods rather than raw attribute assignment,
because they say what you mean:

.. algan:: BasicMobMethods

    from algan import *

    mob = RegularPolygon(5).spawn()
    mob.move(RIGHT * 2)
    mob.rotate(360, OUT)
    mob.rotate(360, UP)
    mob.rotate(360, OUT, about_point=ORIGIN)
    mob = mob.become(Circle())

    Scene.save_video()

The methods used above:

* :meth:`.Mob.move` translates the Mob by a vector: ``mob.move(RIGHT)`` slides it
  one unit right. To move to an absolute point instead, use
  :meth:`~.Mob.move_to`.
* :meth:`.Mob.rotate` turns the Mob about an axis through its own centre.
  Passing ``about_point`` turns it about *that* point instead, which sweeps the
  Mob around in an arc -- ``mob.rotate(180, OUT, about_point=ORIGIN)`` swings it
  half way around the origin.
* :meth:`.Mob.become` morphs the Mob into a different Mob. It returns the
  resulting Mob, so assign it back: ``mob = mob.become(Circle())``.

Two more you will use constantly:

* :meth:`.Mob.scale` grows or shrinks the Mob: ``mob.scale(2)`` doubles its size.
* :meth:`.Animatable.wait` holds the Mob still: ``mob.wait(2)`` leaves two
  seconds of nothing happening. ``Scene.wait(2)`` does the same for the whole
  scene.

:doc:`positioning_and_layout` covers the placement and sizing methods in full,
and the :class:`~.Mob` reference lists every method.

Spawning and despawning
-----------------------

A Mob does not appear -- and cannot be animated -- until it is *spawned*:

.. code-block:: python

    square = Square()        # created, but not on screen
    square.spawn()           # fades in over 1 second, now animatable

    square.despawn()         # fades out and stops being animatable

:meth:`~.Animatable.spawn` returns the Mob, so it chains:
``square = Square().spawn()``. Everything before ``spawn()`` happens instantly
and costs no time on the timeline, which makes it the right place to do setup:

.. code-block:: python

    # Position and size it first, then bring it on screen.
    square = Square(color=BLUE).scale(0.5).move(LEFT * 3).spawn()

.. important::

    Before a Mob is spawned, its animations are turned **off** regardless of
    the surrounding animation context. That is why the chain above takes no time
    at all.

.. _animated-functions:

Animated Functions
------------------

Attribute changes and Mob methods cover the overwhelming majority of what you
will want. For an animation Algan has no method for, write your own with the
:func:`~.animated_function` decorator.

.. algan:: BasicAnimatedFunction

    from algan import *
    import numpy as np

    # A function mapping a scalar parameter t to a point in space.
    def path_func(t):
        return UP * np.sin(t) + RIGHT * (t - PI)

    # An animated_function that moves our mob along that path.
    @animated_function(animated_args={'t': 0})
    def move_along_path(mob, t):
        mob.location = path_func(t)

    square = Square().spawn()
    square.location = path_func(0)   # Jump to the starting point.
    move_along_path(square, 2 * PI)

    Scene.save_video()

``animated_args`` maps each animated parameter to its value at the *start* of the
animation. Algan then interpolates from there to whatever you called the function
with, evaluating the body at every frame. Above, ``t`` starts at ``0`` and the
call passes ``2 * PI``, so the animation sweeps ``t`` from ``0`` to ``2 * PI``
over one second.

.. important::

    An :func:`~.animated_function` must take a :class:`.Mob` as its first
    argument, and every name listed in ``animated_args`` must be a float.

.. note::

    Inside an :func:`~.animated_function`, attribute assignment is *not*
    separately animated -- the function body describes a single frame, and the
    decorator does the animating. Write the body vectorized over torch tensors;
    it is evaluated once per frame for the whole Mob.

Where to next
-------------

* :doc:`mob_gallery` -- what Mobs are available.
* :doc:`positioning_and_layout` -- putting Mobs exactly where you want them.
* :doc:`controlling_animations` -- controlling *when* animations happen and how
  long they take.
