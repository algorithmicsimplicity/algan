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

    Timeline events can *not* be recorded against the default context,
    you must record them inside a ``with ... as context:`` block.

Animating a wave effect
=======================

Suppose that we have a grid of squares

.. code-block:: python

    n = 10
    squares = Group([Square(color=BLUE) for _ in range(n*n)]).arrange_in_grid(n).scale(0.25).spawn()

and we want to animate the effect of a wave passing through them, from the top-left of the
screen to the
bottom-right. When the wave hits a Mob, we will change its color to RED briefly.
This is quite a difficult animation to orchestrate normally. You need to sort
mobs in the order in which the wave hits them, and then calculate how much time
there is between the wave hitting one mob and the next.
Instead, it is much simpler to specify for each Mob when its animation should start.

.. code-block:: python

    wave_direction = F.normalize(RIGHT + DOWN, p=2, dim=-1)
    square_dots = [(square.location * wave_direction).sum().item() for square in squares]
    min_dot = min(square_dots)
    max_dot = max(square_dots)

We now have a list of the times at which each mob should start playing
its animation. And we can use out of order animation to implement the animations.

.. algan:: AOOWave1

    from algan import *
    import torch.nn.functional as F

    n = 10
    squares = Group([Square(color=BLUE) for _ in range(n*n)]).arrange_in_grid(n).scale(0.25).spawn()

    # Calculate wave arrival time for each square based on its position
    wave_direction = F.normalize(RIGHT + DOWN, p=2, dim=-1)
    square_dots = [(square.location * wave_direction).sum().item() for square in squares]
    min_dot = min(square_dots)
    max_dot = max(square_dots)

    with Seq() as context:
        start_time = context.current_time

        for i in range(len(squares)):
            # rescale to [0, 5], so the wave takes 5 seconds to propagate.
            square_start_time = 5 * (square_dots[i] - min_dot) / (max_dot - min_dot)

            # Jump the timeline pointer to when this square should animate
            context.current_time = start_time + square_start_time

            # Write the animation to this point on the timeline.
            with Seq(runtime=2):
                original_color = squares[i].color
                squares[i].color = RED
                squares[i].color = BLUE

        # Jump to the end of the context so future animations continue in order
        context.current_time = context.end_time

    Scene.save_video()

See Also
========

* :doc:`../new_user_tutorials/combining_animations` -- ``Seq``, ``Sync``, and ``Lag`` contexts.
* :doc:`../new_user_tutorials/updaters` -- continuous frame updates.
* :doc:`custom_animations` -- creating custom animation functions.
