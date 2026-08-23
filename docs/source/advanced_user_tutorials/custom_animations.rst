=================
Custom Animations
=================

Combining attribute changes and mob methods in animation contexts
will cover most of your every-day needs.
But if you need an animation which can't be created by combining the
existing ones, then you can make your own animations using the
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

    An :func:`~.animated_function` must take a :class:`~algan.animatable_base.mob.Mob` as its first
    argument, and every name listed in ``animated_args`` must be a float.

.. note::

    Inside an :func:`~.animated_function`, attribute assignment is *not*
    separately animated; the function body describes a single frame, and the
    decorator does the animating.
