========
Updaters
========

So far, every animation has had a fixed duration on the timeline: you describe
what should change, and Algan interpolates the frames in between. But sometimes
you want a rule that holds *continuously*, for as long as it is needed, without
knowing in advance how long that will be.

That is what an updater is for: a function attached to a :class:`~algan.animatable_base.mob.Mob`
that Algan runs once per frame, from the moment you add it until you take it away.

Updaters are great for making passive/idle animations that always play.
They also give you more freedom to define relations between mobs,
compared to the parent-child relation (see :doc:`child_mobs`). For example, you
can use an updater to make one mob follow another *without* also changing its
orientation.

Here's a basic example: keeping a triangle spinning continuously while other
animations play:

.. algan:: UpdatersBasic

    from algan import *

    triangle = Triangle(color=BLUE).spawn()

    # Spin at 180 degrees per second, indefinitely.
    spin = triangle.add_updater(lambda mob, t: mob.rotate(t * 180, OUT))

    # Ordinary animations carry on as usual; the updater runs alongside them.
    triangle.move(RIGHT * 2)
    triangle.color = GREEN
    triangle.move(LEFT * 2)

    # And it stops when you say so.
    triangle.remove_updater(spin)
    triangle.wait(1)

    Scene.save_video()

Let's break down how this works:

* :meth:`~algan.animatable_base.animatable.Animatable.add_updater` attaches a function to the Mob and runs it on every frame.
* The updater function takes two parameters: the **Mob** itself and the **elapsed time** ``t`` in seconds since the updater was added.
* :meth:`~algan.animatable_base.animatable.Animatable.add_updater` returns an integer ID,
  which you can pass to :meth:`~algan.animatable_base.animatable.Animatable.remove_updater` to stop the updater later.

Two important things to notice here:

* The triangle keeps spinning right through the other
  animations. Updaters and standard timeline animations coexist: the timeline drives
  your animations, and updaters are applied on top of the result at every frame.
* Removing an updater does not undo it: the Mob keeps whatever state the last frame
  left it in, which is why the triangle above holds whatever angle it had reached
  rather than snapping back upright.

.. important::

    The second parameter must appear in the updater's signature -- ``(mob, t)``
    -- even if you do not use ``t`` in the function body.

Elapsed Time and Periodic Motion
================================

The second argument, ``t``, is the elapsed time in seconds since the updater was
attached. It makes periodic and open-ended motion easy:

.. algan:: UpdatersPeriodic

    from algan import *
    import torch

    with Off():
        ball = Circle(color=YELLOW).scale(0.5).move(LEFT * 3).spawn()
        label = Text("ball", font_size=32).spawn()

    start_pos = ball.location

    # Bob up and down once per second.
    ball.add_updater(lambda mob, t: mob.move_to(start_pos + UP * 0.8 * torch.sin(t * 2 * PI)))

    # Follow the ball without taking its orientation.
    label.add_updater(lambda mob, t: mob.move_next_to(ball, DOWN))

    with Seq(run_time=4, rate_func=rate_funcs.identity):
        ball.move(RIGHT * 6)

    Scene.save_video()

Notice that these updaters set the Mob's state as a function of total elapsed time
``t`` from a fixed reference point, rather than accumulating small increments
each frame. Always write updaters as functions of ``t``: because Algan evaluates
frames in parallel batches, incremental accumulation gives inconsistent results.

Notice too that the bobbing composes with the ordinary ``move`` animation
underneath it rather than fighting it: the timeline decides where the ball is,
and the updater is applied on top of that result.

.. important::

    ``t`` is a **torch tensor**, not a Python float, and it carries a whole batch of
    frames at once, with shape ``[frames, 1, 1]``. Use ``torch`` functions on it
    (``import torch``, ``torch.sin``, ``torch.exp``, ...) instead of Python's standard
    ``math`` module. The ``math`` module will fail at render time with
    ``ValueError: only one element tensors can be converted to Python scalars``, a
    long way away from the line you actually wrote.

.. seealso::

    * :doc:`../advanced_user_tutorials/cameras` -- an updater on the camera is
      how you follow a subject whose path you do not know in advance.
    * :doc:`../advanced_user_tutorials/animating_out_of_order` -- the other way
      to escape the "one animation after another" model: writing animations to
      a point on the timeline you choose yourself.
    * :doc:`../advanced_user_tutorials/custom_animations` -- when what you want
      *is* a fixed-duration animation, but not one the built-ins provide.
