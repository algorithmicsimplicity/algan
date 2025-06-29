Manim User Quickstart
=====================

This guide is meant for users who are already familiar with Manim, and only explains the key
differences between Manim and Algam so you can get started as quickly as possible.

Algan Basics
------------

In Algan, just like Manim, you create animations by controlling "Mobs", which are objects that will appear on screen.
Algan provides a range of basic mobs covering basic 2d and 3d shapes such as circles, rectangles, spheres,
cubes, as well as Text mobs for displaying text. You can also import mobs from Manim.
You can create animations by modifying mob attributes such as location, scale,
orientation, color using the "set" method, or with specialized methods such as rotate and move.

Basic Mobs
----------

.. algan:: MQSBasicMobs

    from algan import *

    circle = Circle()
    circle.set(color=BLUE, opacity=0.5, border_color=BLUE_E,  border_width=4)

    square = Square()

    square.spawn()
    square.wait()

    square = square.become(circle)
    square.wait()
    square.rotate(180, UP)
    square.move(0.3 * RIGHT)
    square.scale(0.5)
    square.move_to(LEFT+UP*2)

    text = Text("I am a Text Mob.").spawn()

    render_to_file()


Animation Contexts
------------------

In Manim, it is necessary to specify when changes should be animated by using the self.play function
and mob.animate.method animation builder. In Algan, Mob modifications are animated by default once a mob has been spawned.
Prior to spawning, modifications made to a mob will not be animated (and the mob will not appear on screen).
Once it has been spawned, it will begin to appear on screen, and from then on any changes made (e.g. to location, color or basis)
will be animated over a 1 second period. Changes are animated in sequence, in the order that changes are made in code.
In order to override this default behaviour you can make use of AnimationContexts.
AnimationContexts control how all changes made within its contexts should be animated.

.. algan:: MQSAnimationContexts

    from algan import *

    square = Square().spawn()
    circle = Circle().move(OUT*0.01).spawn() # Slight offset towards IN so they don't intersect

    with AnimationContext(run_time_unit=2.5):
        # Each modifications within this context will be animated over 2.5 seconds,
        # instead of the default 1 second.
        square.move(RIGHT)
        circle.move(LEFT)
        # This context is animated over a total of 5 seconds.

    with AnimationContext(run_time=1):
        # The total run time of all animations in this context will be 1 second,
        # i.e. each modification is animated over a period of 1/n seconds where n is the number of
        # modifications made.
        square.move(LEFT)
        circle.move(RIGHT)

    with Sync():
        # All modifications within this context will be animated at the same time (and take 1 second).
        square.move(UP)
        circle.move(DOWN)

    with Seq():
        # Modifications within this context are animated one after the other (note
        # this is the default behaviour when not in any context.
        square.move(DOWN)
        circle.move(UP)

    with Lag(0.6):
        # Modifications within this context are animated sequentially but instead of waiting
        # for one animation to finish completely before starting the next, the next animation
        # is started when the current animation of 60% of the way done.
        # Note that Synchronized() is equivalent to Lagged(0) and Sequenced() is equivalent to Lagged(1).
        square.move(LEFT)
        circle.move(RIGHT)

    with Off():
        # Modifications within this context will not be animated (i.e. they will be applied instantly,
        # taking place in 1 frame).
        square.move_to(ORIGIN)
        circle.move_to(ORIGIN)

    square.wait(1) # wait one second without any changes.

    render_to_file()


Nested Animation Contexts
-------------------------

The real power of AnimationContexts is that they can be nested seamlessly. This makes orchestrating
complex animations involving many Mobs straightforward. When nesting contexts, each sub-context
is treated as a single animation for rules of the parent context.

.. algan:: MQSNestedAnimationContexts

    from algan import *

    square = Square().spawn()
    circle = Circle().spawn()

    with Sync(run_time=3):
        # In this context there are 2 modifications, the square (modified over 1 second),
        # and the circle (modified over 2 seconds). Since this context has run_time=3,
        # each of these 2 animations will be scaled to 3 seconds. i.e. the square
        # animation will be slowed down by a factor of 3, and the circle will be
        # slowed by a factor of 3/2.
        square.move(RIGHT)
        with Seq():
            # This Sequenced context is treated as one single modification, which first moves
            # the circle left over 1 second, then up over 1 second.
            circle.move(LEFT)
            circle.move(UP)

    # This context animates the square and circle moving and rotating at the same time.
    # The square and circle both first move and rotate about the OUT axis at the same time, over 1 second.
    # Then the square and circle both move and rotate about the UP axis at the same time, over 1 second.
    with Sync():
        with Seq():
            with Sync():
                square.move(LEFT)
                square.rotate(180, OUT)
            with Sync():
                square.move(DOWN)
                square.rotate(180, UP)
        with Seq():
            with Sync():
                circle.move(RIGHT)
                circle.rotate(-180, OUT)
            with Sync():
                circle.move(UP)
                circle.rotate(-180, UP)

    render_to_file()
