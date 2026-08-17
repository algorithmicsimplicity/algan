========
Updaters
========

Every animation so far has had a start and an end: you said what should change
and Algan worked out the frames in between. Sometimes you want a rule that holds
*continuously*, for as long as it is needed, without knowing in advance how long that is.

That is what an updater is for: a function Algan runs once per frame, from the moment
you add it until you take it away.

Reach for an updater when you want:

* something to spin, pulse or drift indefinitely in the background.
* one Mob to stay attached to another however that other one moves.
* a label to track a value that other animations are changing.
* an idle animation that plays continuously.

The Basics
==========

:meth:`~.Animatable.add_updater` takes a function of two arguments, the Mob
itself and the elapsed time in seconds since the updater was added,
and applies that function on every frame while the updater is active. :meth:`~.Animatable.add_updater`
returns an id you can use to stop the updater later.

.. algan:: UpdatersRotating

    from algan import *

    square = Square().spawn()
    # Set square rotating at 180 degrees per second, indefinitely.
    # add_updater returns the new updater's id; hang onto it if you
    # want to stop the animation later.
    updater_id_1 = square.add_updater(lambda self, t: self.rotate(t * 180, OUT))

    square2 = Square(color=BLUE).move(RIGHT * 1.5).spawn()
    # Make square2 track square's right-hand direction.
    # Note that even though we don't use t here, we still have to
    # declare it in the signature.
    updater_id_2 = square2.add_updater(lambda self, t: self.move_to(square.location +
                                                    square.get_right_direction() * 1.5))

    # Now carry on animating as usual; the updaters persist alongside.
    square.wait(2)
    square.color = GREEN
    square.wait(2)

    # And stop them whenever you like.
    square2.remove_updater(updater_id_2)
    square.wait(2)

    Scene.save_video()

.. important::

    The second parameter must appear in the updater's signature even if you never use it.

Note that the updaters keep running through the ``square.color = GREEN``
animation. Updaters and ordinary animations coexist: the timeline drives the
animations, and the updaters run on top of the result at every frame.

Stopping an updater
===================

Use :meth:`~.Animatable.remove_updater` or :meth:`~.Animatable.remove_all_updaters`

.. code-block:: python

    updater_id = mob.add_updater(...)

    mob.remove_updater(updater_id)   # stop this one
    mob.remove_all_updaters()        # stop all of this Mob's updaters

Removing an updater does not undo what it did, the Mob keeps whatever state it
was last left in.

Attaching One Mob to Another
============================

The most common use is keeping something pinned to something else. Because the
updater re-reads the target's state every frame, it survives *any* way the target
moves (e.g. an animation, another earlier applied updater, a camera change):

.. algan:: UpdatersTracking

    from algan import *

    anchor = Dot(color=YELLOW).spawn()
    follower = Square(color=BLUE).scale(0.3).spawn()

    follower.add_updater(lambda self, t: self.move_to(anchor.location + UP * 1.2))

    with Seq(run_time=3):
        anchor.move(RIGHT * 2)
        anchor.move(DOWN * 1.5 + LEFT * 3)

    Scene.save_video()

Compare this with :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_next_to`, which resolves the target's
position *once*, when you call it. Use ``move_next_to`` for a one-off placement
and an updater when the relationship has to hold over time.

.. note::

    Adding the same Mob as a child of another (see :doc:`child_mobs`) is a third
    option, and usually the best one when two Mobs should move together rigidly,
    however the parent-child relation can be interefered with when animations
    directly target a child. A link created with an updater will not break
    no matter what. Updaters can also express more general relations
    (like an offset that dependd on the target's orientation) that parent-child
    relations can not.

Using the Elapsed Time
======================

The second argument is the time since the updater was added, in seconds. It makes
periodic and open-ended motion easy, and it does not depend on the frame rate:

.. code-block:: python

    import torch

    # Bob up and down forever, once per second.
    mob.add_updater(lambda self, t: self.move_to(start + UP * 0.3 * torch.sin(t * 2 * PI)))

    # Spin at a constant 90 degrees per second.
    mob.add_updater(lambda self, t: self.rotate(t * 90, UP))

Note the second one: it sets the *total* rotation as a function of ``t`` rather
than adding a bit each frame. Write updaters as a function of ``t`` from a fixed
starting state rather than as incremental steps, as Algan may materialize frames
in a different order than they are played, so an updater that accumulates gives
inconsistent results.

.. important::

    ``t`` is a **torch tensor**, not a Python float. Algan evaluates an updater
    for a whole batch of frames at once, so ``t`` arrives with shape
    ``[frames, 1, 1]``. Use ``torch`` functions on it (``torch.sin``,
    ``torch.exp``, ...); the ``math`` module raises ``ValueError: only one element
    tensors can be converted to Python scalars`` as soon as the batch holds more
    than one frame, and the error surfaces at render time rather than where you
    wrote the updater.

Longer updaters
===============

A lambda is convenient but not required; any function with the right signature
works, which is what you want as soon as the rule needs more than one line:

.. code-block:: python

    def orbit_and_face(self, t):
        angle = t * 60
        self.move_to(ORIGIN + RIGHT * 3)
        self.orbit(angle, UP, about_point=ORIGIN)
        self.look_at(Scene.camera)

    moon.add_updater(orbit_and_face)

Where to next
-------------

* :doc:`child_mobs` -- rigid parent/child attachment, the simpler alternative.
* :doc:`combining_animations` -- fixed-duration animations, which is what you
  want for most things.
* :doc:`../advanced_user_tutorials/animating_out_of_order` -- how the timeline
  works underneath, and how to write to it directly.
