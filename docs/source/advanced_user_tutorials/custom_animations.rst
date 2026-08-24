.. _animated-functions:

=================
Custom Animations
=================

Combining attribute changes and mob methods in animation contexts
will cover most of your every-day needs.
But if you need an animation which can't be created by combining the
existing ones, then you can make your own animations using the
:func:`~algan.animatable_base.animatable.animated_function` decorator.

Before writing one, check that none of these is what you actually want:

* a combination of existing Mob methods inside
  :doc:`animation contexts <../new_user_tutorials/combining_animations>`;
* one of the ready-made animations in :doc:`../galleries/built_in_animations`
  -- there is already an ``ApplyMatrix``, a ``Homotopy`` and a ``PhaseFlow``;
* an :doc:`updater <../new_user_tutorials/updaters>`, if the rule should hold
  continuously rather than for a fixed duration;
* :doc:`animating_out_of_order`, if what you need is not a new animation but a
  different *time* to write existing ones to.

Animated functions
==================

An animated function describes **one frame**: given a parameter, it puts the Mob
where it should be for that parameter's value. The decorator does the animating,
by sweeping the parameter across the animation and calling the body at every
frame.

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

    An :func:`~algan.animatable_base.animatable.animated_function` must take a
    :class:`~algan.animatable_base.mob.Mob` as its first
    argument, and every name listed in ``animated_args`` must be a float.

.. note::

    Inside an :func:`~algan.animatable_base.animatable.animated_function`,
    attribute assignment is *not*
    separately animated; the function body describes a single frame, and the
    decorator does the animating.

An animated function is an animation like any other, so it obeys the surrounding
:doc:`animation context <../new_user_tutorials/combining_animations>`: wrap it in
``Seq(run_time=...)`` to set its duration, put it inside ``Sync()`` to run it
alongside other animations, and give it a ``rate_func`` to change its easing.

More than one animated argument
-------------------------------

Any number of parameters can be animated at once, and each gets its own start
value:

.. algan:: CustomTwoAnimatedArgs

    from algan import *
    import numpy as np

    @animated_function(animated_args={'turns': 0, 'radius': 0})
    def spiral_out(mob, turns, radius):
        angle = turns * 2 * PI
        mob.location = (RIGHT * np.cos(angle) + UP * np.sin(angle)) * radius

    dot = Dot(color=YELLOW).scale(2).spawn()
    with Seq(run_time=3, rate_func=rate_funcs.identity):
        spiral_out(dot, turns=2.5, radius=3.0)

    Scene.save_video()

See Also
========

* :doc:`../galleries/built_in_animations` -- the animations Algan already
  provides, several of which are animated functions themselves.
* :doc:`../new_user_tutorials/combining_animations` -- the contexts an animated
  function composes with.
* :doc:`../new_user_tutorials/updaters` -- the other way to write behaviour of
  your own, for rules that hold continuously.
* :doc:`extending_algan` -- packaging animations, Mobs and primitives of your
  own for reuse.
