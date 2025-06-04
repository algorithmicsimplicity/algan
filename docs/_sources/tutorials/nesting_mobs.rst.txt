============
Nesting Mobs
============

So far we have been using simple mobs. To make more complex mobs we can combine together multiple mobs into one.
This is done by adding other mobs as children to one another.

.. algan:: NMBasic

    from algan import *

    parent_mob = Square(color=BLUE)
    children_mobs = [Square(location=loc) for loc in [LEFT*2.5, UP*2.5, RIGHT*2.5, DOWN*2.5]]

    parent_mob.add_children(children_mobs) # this is the crucial step

    # Now, any change that we make to the parent mob will be propagated to the children mobs (including spawning).
    parent_mob.scale(0.75).spawn()
    parent_mob.rotate(90, OUT)
    parent_mob.move(RIGHT*0.5)
    with Seq(run_time=5):
        parent_mob.rotate_around_point(ORIGIN, 360, OUT)

    parent_mob.wait()
    # We can even apply animations to the parent and the children at the same time.
    with Sync(run_time=5):
        parent_mob.rotate(90, OUT)
        children_mobs[0].rotate(180, UP)

    render_to_file()
