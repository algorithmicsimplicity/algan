========
Updaters
========

So far we have been specifying animations by saying what time they start,
and how long they should run for. Sometimes, you want to be able to play
an animation indefinitely, without knowing how long it will play for.
In such cases, you can use *Updaters*.

An updater applied to a mob will continuously update a mob's state
every frame, until it is explicitly stopped.

Example: Perpetually Rotating
-----------------------------

In this example we apply 2 updaters, one to make the inner square rotate,
and another to make the outer square follow its rotation.

.. algan:: BasicChangingAttributes

    from algan import *

    square = Square().spawn()
    # Set square rotating at a rate of 180 degrees per second, indefinitely.
    # The add_updater function returns the ID the updater it creates,
    # we will need to hang onto this ID if we want to stop the animation later.
    updater_id_1 = square.add_updater(lambda self, t: self.rotate(t*180, OUT))

    square2 = Square(color=BLUE).move(RIGHT*1.5).spawn()
    # Make square2 track square's right direction.
    # Note that even though we don't use the t parameter here,
    # we still must declare it in the function signature.
    updater_id_2 = square2.add_updater(lambda self, t: self.move_to(square.location +
                                                    square.get_right_direction()*1.5))

    # Now we can continue animating as usual, the updaters will persist.
    square.wait(2)
    square.color = GREEN
    square.wait(2)

    # And we can stop the updaters when we want.
    square2.remove_updater(updater_id_2)
    square.wait(2)

    render_to_file()

The :meth:`~.Animatable.add_updater` function takes as input a function, and applies
that function at every subsequent frame. The function must take a Mob as its first argument
(which is set to the mob itself), and a float as its second argument. During animation,
at each frame the function will be called with the second parameter set to the current
elapsed time (the number of seconds since the updater was first added) for that frame.

.. important::

    The second parameter must be specified in the function signature even if you don't use it!
