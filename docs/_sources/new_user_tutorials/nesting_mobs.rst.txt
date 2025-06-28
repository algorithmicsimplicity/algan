============
Nesting Mobs
============

So far we have been using simple :class:`.Mob` s. To make more complex :class:`.Mob` s we can combine together multiple mobs into one.
This is done by adding other mobs as children to one another with :meth:`~.Animatable.add_children` .

.. algan:: NMBasic

    from algan import *

    parent_mob = Square(color=BLUE)
    children_mobs = [Square(location=loc) for loc in [LEFT*2.5, UP*2.5, RIGHT*2.5, DOWN*2.5]]

    parent_mob.add_children(children_mobs) # this is the crucial step

    # Now, any change that we make to the parent mob will be propagated to the
    # children mobs (including spawning).
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

Once mobs are added as a child to another, any changes applied to the parent will be propagated
to the child mobs. The way that the change is propagated depends on the attribute changed.
When a parent's location is changed, the child mob's location is moved by the same amount.
When a parent's basis is rotated, the child's basis is rotated by the same amount and the child's
relative location to the parent is preserved. Specifically, the child's location expressed in coordinates
of the parent's basis is unchanged.
The result of this is that the child behaves as if it is attached to the parent by a solid pole.

Changes made directly to a child will ignore the parent relation.

.. note::

    You can get a list of all of a mob's children with :attr:`~.Mob.children` , but this variable
    is read only! You should always use :meth:`~.Animatable.add_children` for adding children.
    You can also get a list of all descendent mobs (children, and grand-children, and so-on) with :meth:`~.Mob.get_descendants`

Grouping Mobs
=============

Mobs can also be grouped together into one :class:`~.Mob` using :class:`~.Group` . The :class:`~.Group`
class provides the :meth:`~.Group.arrange_in_line` and :meth:`~.Group.arrange_in_grid` methods
which are useful for arranging Mobs.

.. algan:: NMGroup

    from algan import *

    mobs = [Square() for _ in range(9)]
    group = Group(mobs)
    group.scale(1/3).spawn()
    group.arrange_in_line(RIGHT)
    group.wait()
    with Sync():
        group.scale(3)
        group.arrange_in_grid(3)
    group.wait()

    render_to_file()


.. note::

    Internally, the way that :class:`~.Group` works is by creating a new empty mob
    at the center of the provided mobs, then adding all of the provided mobs as children to it.
