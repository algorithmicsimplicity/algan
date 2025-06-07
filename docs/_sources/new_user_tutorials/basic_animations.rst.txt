================
Basic Animations
================

In Algan you create animations by controlling :class:`.Mob` s,
which are objects that will appear on screen.
Algan provides a range of basic :class:`.Mob` s covering basic 2-D and 3-D
shapes such as circles, rectangles, spheres, cubes, as well as
:class:`.Text` :class:`.Mob` s for displaying text and LaTex.
You can also import mobs from Manim.

Changing Animatable Attributes
------------------------------

All :class:`.Mob` s have the following animatable attributes: :attr:`~.Mob.location`, :attr:`~.Mob.basis`,
:attr:`~.Mob.color`, :attr:`~.Mob.glow`, :attr:`~.Mob.opacity`. These attributes
are special in that any modifications made to them will automatically be animated.
Specifically, when a new value is assigned to an animatable attribute, that modification will take place over a 1 second period,
during which the attribute is linearly interpolated from the old value to the new value.

:attr:`~.Mob.location` is a vector length 3 which specifies where in 3-D space a mob is located. By default, new Mobs are created at the ORIGIN (0, 0, 0).

.. note::

    By default the camera is located at OUT*7 (0, 0, 7) and looks towards the ORIGIN.

:attr:`~.Mob.basis` is a vector of length 9 that specifies the orientation and scale of a Mob. It is not recommended to modify
basis directly and instead use the helper methods like :meth:`.Mob.rotate` and :meth:`.Mob.scale`.

:attr:`~.Mob.color` is a :class:`~.Color` object which specifies the main color of the Mob.

.. note::

    In Algan colors have red, green, blue components, as well as glow and opacity (and internally are stored
    as a vector of 5 components in that order). Colors with a non-zero glow component will 'glow', emitting light
    into nearby pixels.
    :attr:`~.Mob.glow` and :attr:`~.Mob.opacity` can optionally be set in the :class:`~.constants.color.Color` object, or as properties of the Mob itself.

.. algan:: BasicChangingAttributes

    from algan import *

    circle = Circle().spawn()

    circle.location = circle.location + UP
    circle.location = circle.location + DOWN + RIGHT
    circle.color = GREEN
    circle.glow = 0.5
    circle.opacity = 0.5

    render_to_file()

.. important::

    Only non-inplace assignments are animated! That means that, for example, ``circle.location += UP * 0.5`` will
    not be animated. You should NEVER assign animated attributes inplace!


Animated Functions
------------------

You will be able to get pretty far just modifying the animatable attributes, but there are some cases were linear
interpolation just isn't enough. For example, you may wish to move a Mob along a curved path. This animation
cannot be expressed by linearly interpolating the Mob's location. For such a case, you can create an :func:`~.animated_function`.

.. algan:: BasicAnimatedFunction

    from algan import *
    import numpy as np

    # Define a function mapping a scalar parameter t to a point on the circle.
    def path_func(t):
        return (UP * np.sin(t) + RIGHT * np.cos(t))

    # Create an animated_function which will move our mob along this path.
    @animated_function(animated_args={'t': 0})
    def move_along_path(mob, t):
        mob.location = path_func(t)

    square = Square().spawn()
    square.location = path_func(0) # Move to starting point.
    move_along_path(square, 2*PI)

    render_to_file()

The :func:`~.animated_function` decorator specifies that a function should be animated. This decorator accepts a parameter
``animated_args``, which must be a dictionary mapping the names of animated arguments to their initial values when the animation
begins. Like with attribute modification, the animation will take place over a 1 second period. The animation is created by linearly interpolating the ``animated_args`` from their initial values given in the dict,
to the value the function is called with. In this example, we specify that parameter ``t`` has an initial value
of 0, and we call the function with ``t=2*PI``, so the animation will range from ``t=0`` to ``t=2*PI``.

.. important::

    An :func:`~.animated_function` must accept at least one argument, and the first argument must be a :class:`.Mob`. Any arguments
    marked as ``animated_args`` must be floats.

.. note::

    Inside of an :func:`~.animated_function`, the default animations created by modifying animatable attributes are disabled.

Mob Methods
-----------

Finally, the :class:`.Mob` class comes with a bunch of animated functions built in as methods, which should serve
most of your common needs. Here are some examples:

.. algan:: BasicMobMethods

    from algan import *

    mob = RegularPolygon(5).spawn()
    mob.move(RIGHT)
    mob.rotate(360, OUT)
    mob.rotate(360, UP)
    mob.rotate_around_point(ORIGIN, 180, OUT)
    mob = mob.become(Circle())

    render_to_file()

Here's a brief explanation of the methods shown in the example:

* :meth:`.Mob.move`: This method is used to translate (move) the :class:`.Mob` by a specified vector. For example,
  `mob.move(RIGHT)` moves the object to the right one unit.
* :meth:`.Mob.rotate`: This method rotates the :class:`.Mob` by a given angle around a specified axis. For instance, `mob.rotate(360, OUT)` rotates the object 360 degrees around the outward-pointing Z-axis.
* :meth:`.Mob.rotate_around_point`: This method rotates the :class:`.Mob` around a specific point in space. `mob.rotate_around_point(ORIGIN, 180, OUT)` rotates the object 180 degrees around the origin (0,0,0) along the Z-axis.
* :meth:`.Mob.become`: This method smoothly transforms the current :class:`.Mob` into the shape of another :class:`.Mob` provided as an argument. For example, `mob = mob.become(Circle())` transforms the existing mob into a circle.