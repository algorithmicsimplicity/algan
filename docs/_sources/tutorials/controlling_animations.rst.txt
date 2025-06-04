====================
Controlling Animations
====================

So far we've seen how to create basic animations by modifying animatable attributes and using animated_functions.
But in all of these examples, the animations play for 1 second each, and take place one after the other.
What if we want more control over how the animations should happen? Not to worry,
Algan is specifically designed to make orchestrating complex animations easy, and to this end it provides
:class:`.AnimationContext` s.

Animation Contexts
------------------

In Algan, you can control how animations should be played by placing them within an appropriate context.
Let's dive in with an example.

.. algan:: CombiningSync

    from algan import *

    mob = Square().spawn()
    with Sync():
        mob.move(RIGHT)
        mob.rotate(90, OUT)

    render_to_file()

Here we use the :class:`.Sync` context (read as "with animations synchronized") to specify
the animations should be synchronized. All animations that take place within the with Sync(): clause will be
played at the same time.

In addition to Sync, Algan also provides the :class:`.Seq` ("with animations sequenced"), :class:`.Lag` ("with animations lagged"),
and :class:`.Off` ("with animations off") contexts.


.. algan:: CombiningContextExamples

    from algan import *

    mob1 = Circle().spawn()
    mob2 = Square().spawn()

    with Sync():
        mob1.move(RIGHT)
        mob2.rotate(360, OUT)

    with Seq():
        mob1.move(UP)
        mob2.move(RIGHT)

    with Lag(0.5):
        mob1.move_to(ORIGIN)
        mob2.move_to(ORIGIN)

    with Off():
        mob1.move(RIGHT)
        mob2.move(LEFT)

    render_to_file()

Seq() will play animations sequentially one after the other (note that this is the default behaviour when not in any context),
Lag(lag_ratio=r) will play animations sequentially lagged by a factor of r. e.g. Lag(0.5) will begin the next
animation once the current one is 50% finished.

.. note::

    Lag(0) is equivalent to Sync() and Lag(1) is equivalent to Seq().

Finally Off() disables animations that take place within its context, all changes will be instant (1 frame).

:class:`~.AnimationContexts` s can also be given a number of parameters to change their
behaviour. Most notably, the length of animations that take place within a context can be
controlled with run_time and run_time_unit.

.. algan:: CombiningContextExamples

    from algan import *

    mob1 = Circle().spawn()

    with Seq(run_time=1):
        mob1.move(LEFT)
        mob1.move(UP)
        mob1.move(RIGHT*2)
        mob1.move(DOWN)

    with Seq(run_time_unit=5):
        mob1.rotate(360, UP)
        mob1.move_to(ORIGIN)

    render_to_file()

The run_time parameter specifies the total amount of time that the context should take place over, individual animations
will be rescaled (sped up or slowed down) so that their total time is equal run_time.
The run_time_unit parameter specifies how long each individual animation should be played for.

.. note::

    If both parameters are set, run_time overrides run_time_unit.

Nesting Contexts
----------------

The real power of animation contexts is that they can be nested seamlessly. When one context is created within
another, the sub-context will be treated as a single cohesive animation by the parent. This way, you can think of
each context as combining the animations that take place within it into a single new animation.
This makes specifying complex animations and designing modular animation code a breeze.


.. algan:: CombiningContextExamples

    from algan import *

    mob1 = Circle().spawn()
    mob2 = Square().spawn()

    with Sync():
        with Seq():
            with Sync():
                mob1.move(LEFT*3)
                mob1.rotate(180, UP)
            with Sync():
                mob1.move(UP)
                mob1.color = YELLOW_A
            with Sync():
                mob1.move(RIGHT*3)
                mob1.glow = 0.5

        with Seq():
            with Sync():
                mob2.move(RIGHT*3)
                mob2.rotate(180, OUT)
            with Sync():
                mob2.move(DOWN)
                mob2.color = GREEN_E
            with Sync():
                mob2.move(LEFT*3)
                mob2.glow = 0.5

    render_to_file()
