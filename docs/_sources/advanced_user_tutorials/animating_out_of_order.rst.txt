======================
Animating Out of Order
======================

In Algan, when you change an animatable attribute, or run an animated function, Algan
does not actually perform that animation immediately. Instead, Algan makes a record
of the fact that this animation took place, and the times at which the animation
begins and ends. Algan stores this information inside of the :class:`.Mob` s :attr:`~.Animatable.data`
attribute, which is an :class:`.AnimatableData` object. The time at which the animation
takes place is controlled by the AnimationContexts. For example,
in a Seq context, once an animation is done, the context will write the animation
to the current time, then increment the current time by 1. So the next animation
will be written to one second later on the timeline, and so on.

Once a command to render is given, as in :func:`.render_to_file`, Algan reads through
all of the Mob's animation timelines and actually performs the interpolations
to compute animated states.

Most of the time, you do not need to worry about this and you can just let the
animation contexts handle the writing of animations to the timeline. But if you want to,
you can take manual control of the animation writing, to write animations anywhere
in the timeline, at any point in the code. And in some situations,
this makes animation code much simpler.

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

    render_to_file()