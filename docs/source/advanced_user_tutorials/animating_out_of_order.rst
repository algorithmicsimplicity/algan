======================
Animating Out of Order
======================

In Algan, when you change an animatable attribute, or run an animated function, Algan
does not actually perform that animation immediately. Instead, Algan makes a record
of the fact that this animation took place, and the times at which the animation
begins and ends. Algan stores this information on the owning Scene's
:class:`~algan.animation_timeline.timeline.AnimationTimeline`, accessed through
``mob.scene.timeline_manager``. Each Scene has a separate timeline. The time at
which the animation takes place is controlled by that Scene's AnimationContexts. For example,
in a Seq context, once an animation is done, the context will write the animation
to the current time, then increment the current time by 1. So the next animation
will be written to one second later on the timeline, and so on.

Once a command to render is given, as in :meth:`~algan.scene.Scene.save_video`, Algan reads through
the rendered Scene's timeline and performs the interpolations needed to compute
animated states.

Most of the time, you do not need to worry about this and you can just let the
animation contexts handle the writing of animations to the timeline. But if you want to,
you can take manual control of the animation writing, to write animations anywhere
in the timeline, at any point in the code. And in some situations,
this makes animation code much simpler.

The two handles you need are on the context object itself, which you get by naming
it with ``as``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Meaning
   * - ``context.current_time``
     - The point on the timeline the context is writing to. Assign to it to move
       where subsequent animations land.
   * - ``context.end_time``
     - Where the context finishes. Assign ``current_time = end_time`` when you are
       done jumping around, so ordinary sequential animation resumes cleanly
       afterwards.

.. important::

    Timeline events must be recorded against a context that is *entered* -- that is
    what ``with ... as context:`` gives you. Times written outside any entered
    context all collapse to zero.

Animating a wave effect
***********************

Suppose that we have a bunch of mobs

.. code-block:: python

    n = 10
    mobs = Group([Square(color=BLUE) for _ in range(n*n)]).arrange_in_grid(n).scale(0.25).spawn()

and we want to animate the effect of a wave passing through them, from the top-left of the
screen to the
bottom-right. When the wave hits a Mob, we will change its color to RED briefly.
This is quite a difficult animation to orchestrate normally. You need to sort
mobs in the order in which the wave hits them, and then calculate how much time
there is between the wave hitting one mob and the next.
Instead, it is much simpler to specify for each Mob when its animation should start.

.. code-block:: python

    wave_direction = F.normalize(RIGHT + DOWN, p=2, dim=-1)
    mob_dots = [(mob.location * wave_direction).sum().item() for mob in mobs]
    min_dot = min(mob_dots)
    max_dot = max(mob_dots)

We now have a list of the times at which each mob should start playing
its animation. And we can use out of order animation to implement the animations.

.. algan:: AOOWave1

    from algan import *
    import torch.nn.functional as F

    n = 10
    mobs = Group([Square(color=BLUE) for _ in range(n*n)]).arrange_in_grid(n).scale(0.25).spawn()

    wave_direction = F.normalize(RIGHT + DOWN, p=2, dim=-1)
    mob_dots = [(mob.location * wave_direction).sum().item() for mob in mobs]
    min_dot = min(mob_dots)
    max_dot = max(mob_dots)

    with Seq() as context:
        # Get the current point in the timeline which this context is writing to.
        animation_start_time = context.current_time
        for i in range(len(mobs)):
            # rescale to [0, 5], so the wave takes 5 seconds to propagate.
            mob_start_time = 5 * (mob_dots[i] - min_dot) / (max_dot - min_dot)

            # Set the current time we write animations to,
            # to the point in time when this mob should start
            context.current_time = animation_start_time+mob_start_time

            # Write the animation to this point on the timeline.
            with Seq(run_time=2):
                original_color = mobs[i].color
                mobs[i].color = RED
                mobs[i].color = original_color

        # Now that we are done writing the animations, jump to the end of the context to
        # continue animating in order.
        context.current_time = context.end_time

    Scene.save_video()

When to reach for this
**********************

Out-of-order writing is the right tool when **each Mob's start time is a function
of something about that Mob** -- its position, its value, its index in a sorted
order. The wave above is the canonical case: the start time comes from a dot
product, so computing it directly is far simpler than working out the lag ratios
that would produce the same effect.

For anything simpler, prefer the ordinary contexts:

* A fixed stagger across a list is :class:`~.Lag` (see
  :doc:`../new_user_tutorials/combining_animations`).
* A rule that holds continuously, rather than a set of scheduled animations, is an
  updater (see :doc:`../new_user_tutorials/updaters`).

See Also
********

* :doc:`../new_user_tutorials/combining_animations` -- the contexts this bypasses.
* :doc:`../developer_tutorials/index` -- how the timeline materializes state.
