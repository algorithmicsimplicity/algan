====================
Combining Animations
====================

So far every animation has taken exactly one second and happened strictly after
the one before it. Real videos need more control than that: things happening at the
same time, at different speeds, with different easing.

Algan handles all of this with :class:`.AnimationContext` s: ``with`` blocks that
change *when* the animations inside them happen and *how long* they take.

Here's a basic example, playing two animations at the same time:

.. algan:: ControllingSync

    from algan import *

    square = Square().spawn()
    with Sync():
        square.move(RIGHT * 2)
        square.rotate(90, OUT)

    Scene.save_video()

Everything inside a :class:`.Sync` block
plays *simultaneously*, so the square above slides and turns at once.

The Four Contexts
=================

The four basic contexts are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Context
     - Behaviour
   * - :class:`.Seq`
     - with animations sequenced -- each animation starts when the previous one
       finishes. This is the default behaviour when not in any context.
   * - :class:`.Sync`
     - with animations synchronized -- everything starts together at the same time.
   * - :class:`.Lag`
     - with animations lagged -- ``Lag(r)`` starts the next animation when the current one
       is a fraction ``r`` of the way through.
   * - :class:`.Off`
     - with animations off -- Instant, changes apply in a single frame, taking no time at all.

.. algan:: ControllingContexts

    from algan import *

    with Off():
        square = Square().spawn()
        circle = Circle().spawn()
        square.move(LEFT)
        circle.move(RIGHT)

    with Sync():
        square.rotate(360, OUT)
        circle.move(RIGHT)

    with Seq():
        square.move(RIGHT*2)
        circle.move(UP)

    with Lag(0.5):
        square.move_to(ORIGIN)
        circle.move_to(ORIGIN)

    Scene.save_video()

:class:`.Off` is useful for doing scene setup, use it to put
everything in position for where it should first appear.

.. important::

    Before a Mob is spawned, its animations are turned **Off** regardless of
    the surrounding animation context.

.. note::

    Sync is equivalent to Lag(0) and Seq is equivalent to Lag(1).

Timing
======

Two arguments control how long a context takes:

* ``duration`` -- the total duration of the whole block, in seconds. The
  animations inside are rescaled to fit.
* ``duration_unit`` -- the duration of each individual animation inside.

.. algan:: ControllingTiming

    from algan import *

    circle = Circle().spawn()

    with Seq(duration=1):
        circle.move(LEFT)
        circle.move(UP)
        circle.move(RIGHT * 2)
        circle.move(DOWN)

    with Seq(duration_unit=5):
        circle.rotate(360, UP)
        circle.move_to(ORIGIN)

    Scene.save_video()

The first block squeezes four moves into one second total; the second gives each
of its two animations five seconds. If you set both, ``duration`` overrides ``duration_unit``.

Nesting Contexts
================

Contexts nest, and this is where they get really useful. A nested context is treated
by its parent as a *single* animation, so you can build up a complex piece of
choreography out of small, readable (and reusable!) blocks. A nested context also inherits every
parameter you did not set (``duration_unit``, ``lag_ratio``, ``rate_func``)
so you can set a house style on the outside and only override the exceptions.

.. algan:: ControllingNesting

    from algan import *

    circle = Circle().spawn()
    square = Square().spawn()

    with Sync():
        with Seq():
            with Sync():
                circle.move(LEFT * 3)
                circle.rotate(180, UP)
            with Sync():
                circle.move(UP)
                circle.color = YELLOW_A
            with Sync():
                circle.move(RIGHT * 3)
                circle.glow = 0.5

        with Seq():
            with Sync():
                square.move(RIGHT * 3)
                square.rotate(180, OUT)
            with Sync():
                square.move(DOWN)
                square.color = GREEN_E
            with Sync():
                square.move(LEFT * 3)
                square.glow = 0.5
    Scene.wait()

    Scene.save_video()

The outer :class:`.Sync` sees two things, the circle's three-step routine and
the square's, and so plays them together, even though each is internally a
sequence of pairs. Wrapping either routine in a function would let you reuse the
whole choreography as one animation.

Easing With Rate Functions
==========================

A ``rate_func`` maps progress through an animation (``0`` to ``1``) to how far
along the change should be at that moment. It is what makes motion feel like it
accelerates and settles rather than snapping between states.

Algan's default is ``rate_funcs.smooth``: a gentle ease in and out.
You can pass a different one to any context:

.. algan:: ControllingRateFuncs

    from algan import *

    with Off():
        squares = [Square(color=c).scale(0.4).move((i-1)*DOWN*1.5 + LEFT*3).spawn()
                        for i, c in enumerate((BLUE, GREEN, YELLOW))]

    funcs = (rate_funcs.identity, rate_funcs.smooth, rate_funcs.ease_out_quintic)
    with Sync(duration=2):
        for square, func in zip(squares, funcs):
            with Seq(rate_func=func):
                square.move(RIGHT * 6)

    Scene.save_video()

The three squares cover the same distance in the same time but arrive
differently. Here are some useful rate functions:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Function
     - Feel
   * - ``rate_funcs.smooth``
     - Ease in and out. The default, and the right choice most of the time.
   * - ``rate_funcs.identity``
     - Constant speed. Use it for anything that should look mechanical (a
       rotating turntable, a camera orbit, a clock hand).
   * - ``rate_funcs.ease_out_quintic``
     - Fast start, long settle. Good for things arriving on screen.
   * - ``rate_funcs.ease_in_expo`` / ``rate_funcs.ease_out_expo``
     - Sharp acceleration / deceleration.

The :mod:`~algan.constants.rate_funcs` reference lists the whole catalogue.

Writing your own rate function
------------------------------

A rate function is just a function from a tensor in ``[0, 1]`` to a tensor in
``[0, 1]``, so you can write your own:

.. algan:: ControllingCustomRateFunc

    from algan import *

    def bounce_out(t):
        return 1 - (1 - t) ** 2

    square = Square(color=BLUE).spawn()
    with Seq(rate_func=bounce_out, duration=2):
        square.move(DOWN * 2)

    Scene.save_video()

``rate_funcs.inversed(f)`` gives you the time-reversed version of any rate
function, and passing ``rate_func_compose`` instead of ``rate_func`` composes
with the parent context's easing rather than replacing it.

.. note::

    A context that uses ``rate_func`` applies it across the *whole block*. If
    you want a long orbit to run at constant speed, put ``rate_func`` on the
    context that owns the orbit, not on an enclosing one that also holds other
    animations.

.. seealso::

    * :doc:`../advanced_user_tutorials/audio_and_speech` --
      :class:`~algan.animation_timeline.animation_contexts.Audio` and
      :class:`~algan.animation_timeline.animation_contexts.Speech` are contexts
      too, and they take their duration from a sound file rather than from
      ``duration``.
    * :doc:`../advanced_user_tutorials/animating_out_of_order` -- writing
      animations to a point on the timeline of your own choosing, for when each
      Mob's start time is a function of something about that Mob.
