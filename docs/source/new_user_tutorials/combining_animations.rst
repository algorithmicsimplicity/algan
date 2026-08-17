====================
Combining Animations
====================

So far every animation has taken exactly one second and happened strictly after
the one before it. Real videos need more control than that: things happening at the
same time, overlapping, at different speeds, with different easing.

Algan handles all of this with :class:`.AnimationContext` s: ``with`` blocks that
change *when* the animations inside them happen and *how long* they take. You
never write a ``play`` call or pass a duration to an individual animation; you
put the animations in the right block.

The Four Contexts
=================

.. algan:: ControllingSync

    from algan import *

    mob = Square().spawn()
    with Sync():
        mob.move(RIGHT * 2)
        mob.rotate(90, OUT)

    Scene.save_video()

Everything inside a :class:`.Sync` block
plays *simultaneously*, so the square above slides and turns at once. The four basic contexts are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Context
     - Behaviour
   * - :class:`.Seq`
     - with animations sequenced -- each animation starts when the previous one finishes. This
       is the default behaviour when not in any context.
   * - :class:`.Sync`
     - with animations synchronized -- everything starts together at the same time.
   * - :class:`.Lag`
     - with animations lagged -- ``Lag(r)`` starts the next animation when the current one
       is a fraction ``r`` of the way through.
   * - :class:`.Off`
     - with animations off -- Instant, changes apply in a single frame, taking no time at all.

.. algan:: ControllingContexts

    from algan import *

    mob1 = Square().spawn()
    mob2 = Circle().spawn()

    with Sync():
        mob1.rotate(360, OUT)
        mob2.move(RIGHT)

    with Seq():
        mob1.move(RIGHT)
        mob2.move(UP)

    with Lag(0.5):
        mob1.move_to(ORIGIN)
        mob2.move_to(ORIGIN)

    with Off():
        mob1.move(LEFT)
        mob2.move(RIGHT)

    Scene.save_video()

:class:`.Off` is how you do setup. Positioning, spawning and configuring things
inside ``with Off():`` costs no timeline, so your video starts where you want it
to:

.. important::

    Before a Mob is spawned, its animations are turned **Off** regardless of
    the surrounding animation context.

.. note::

    Sync is equivalent to Lag(0) and Seq is equivalent to Lag(1).

.. code-block:: python

    with Off():
        # Build and place the whole scene instantly.
        Scene.clear_light_sources()
        DirectionalLight(location=UP * 8, target=ORIGIN).spawn()
        diagram.scale(0.8).move_to_edge(LEFT).spawn()

    # Now the video begins.
    diagram.rotate(360, UP)

Timing
======

Two arguments control how long a context takes:

* ``run_time`` -- the total duration of the whole block, in seconds. The
  animations inside are rescaled to fit.
* ``run_time_unit`` -- the duration of each individual animation inside.

.. algan:: ControllingTiming

    from algan import *

    mob1 = Circle().spawn()

    with Seq(run_time=1):
        mob1.move(LEFT)
        mob1.move(UP)
        mob1.move(RIGHT * 2)
        mob1.move(DOWN)

    with Seq(run_time_unit=5):
        mob1.rotate(360, UP)
        mob1.move_to(ORIGIN)

    Scene.save_video()

The first block squeezes four moves into one second total; the second gives each
of its two animations five seconds. If you set both, ``run_time`` overrides ``run_time_unit``.

To leave a deliberate pause, use :meth:`~.Animatable.wait`:

.. code-block:: python

    mob.wait(2)        # two seconds of this mob doing nothing
    Scene.wait(2)      # exactly equivalent to mob.wait(2)


Nesting Contexts
================

Contexts nest, and this is where they get really useful. A nested context is treated
by its parent as a *single* animation, so you can build up a complex piece of
choreography out of small, readable (and reusable!) blocks. A nested context also inherits every
parameter you did not set (``run_time_unit``, ``lag_ratio``, ``rate_func``)
so you can set a house style on the outside and only override the exceptions.

.. algan:: ControllingNesting

    from algan import *

    mob1 = Circle().spawn()
    mob2 = Square().spawn()

    with Sync():
        with Seq():
            with Sync():
                mob1.move(LEFT * 3)
                mob1.rotate(180, UP)
            with Sync():
                mob1.move(UP)
                mob1.color = YELLOW_A
            with Sync():
                mob1.move(RIGHT * 3)
                mob1.glow = 0.5

        with Seq():
            with Sync():
                mob2.move(RIGHT * 3)
                mob2.rotate(180, OUT)
            with Sync():
                mob2.move(DOWN)
                mob2.color = GREEN_E
            with Sync():
                mob2.move(LEFT * 3)
                mob2.glow = 0.5
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

Algan's default is ``rate_funcs.smooth``: a gentle ease in and out. Pass a
different one to any context:

.. algan:: ControllingRateFuncs

    from algan import *

    mobs = [Square(color=c).scale(0.4) for c in (BLUE, GREEN, YELLOW)]
    group = Group(mobs)
    group.arrange_in_line(DOWN, buffer=0.8).move(LEFT * 3).spawn()

    funcs = (rate_funcs.identity, rate_funcs.smooth, rate_funcs.ease_out_quintic)
    with Sync(run_time=2):
        for mob, func in zip(mobs, funcs):
            with Seq(rate_func=func):
                mob.move(RIGHT * 6)

    Scene.save_video()

The three squares cover the same distance in the same time but arrive
differently. The ones you will reach for most:

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
   * - ``rate_funcs.delay_fade``, ``rate_funcs.pulse_fade``
     - Shaped fades, used by Algan's own spawn animations.

A rate function is just a function from a tensor in ``[0, 1]`` to a tensor in
``[0, 1]``, so you can write your own:

.. code-block:: python

    def bounce_out(t):
        return 1 - (1 - t) ** 2

    with Seq(rate_func=bounce_out):
        mob.move(DOWN * 2)

``rate_funcs.inversed(f)`` gives you the time-reversed version of any rate
function, and passing ``rate_func_compose`` instead of ``rate_func`` composes
with the parent context's easing rather than replacing it.

.. note::

    A context that uses ``rate_func`` applies it across the *whole block*. If
    you want a long orbit to run at constant speed, put ``rate_func`` on the
    context that owns the orbit, not on an enclosing one that also holds other
    animations.

Timing recipes
==============

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - You want
     - Write
   * - Two things at once
     - ``with Sync(): ...``
   * - A block to last exactly 3 seconds
     - ``with Seq(run_time=3): ...``
   * - Each step to last 3 seconds
     - ``with Seq(run_time_unit=3): ...``
   * - A staggered ripple across a list
     - ``with Lag(0.2): for m in mobs: ...``
   * - Instant, untimed setup
     - ``with Off(): ...``
   * - Constant speed, no easing
     - ``with Seq(rate_func=rate_funcs.identity): ...``
   * - Every step stretched to the longest one
     - ``with Sync(same_run_time=True): ...``
   * - A pause
     - ``Scene.wait(2)``

Where to next
-------------

* :doc:`built_in_animations` -- ready-made animations to put inside these
  contexts.
* :doc:`updaters` -- animations that run indefinitely rather than for a fixed
  time.
* :doc:`../advanced_user_tutorials/animating_out_of_order` -- writing animations
  to arbitrary points on the timeline.
* :doc:`../advanced_user_tutorials/audio_and_speech` -- contexts whose duration
  comes from a sound file or a line of narration.
