=============
Grouping Mobs
=============

So far we've been applying animations to individual mobs. But often it is
more convenient to apply animations to **Groups** of mobs.
Algan handles this defining a parent-child relation between two mobs.
Once such a relationship is established, any animation applied to the parent
will be propagated to its children (and those children will then propagate
the animation to *their* children, recursively).
As such, complex objects can be built up of many simple individual mob components,
and animated as one entity.

The parent-child relation is established by calling
:meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children`,
which attaches the parameter mob as a child of the calling mob.
Algan also provides the :class:`~.Group` class which takes a collection of
mobs and attaches them to a new invisible parent ``Group`` mob. ``Group``
also provides various layout helpers.

Let's see an example.

.. algan:: ChildMobsBasic

    from algan import *

    center_square = Square(color=BLUE)
    outer_squares = [Square(location=loc) for loc in [LEFT * 2.5, UP * 2.5, RIGHT * 2.5, DOWN * 2.5]]

    center_square.add_children(outer_squares)  # this is the crucial step

    # Now any change to the parent propagates to the children,
    # including spawning.
    center_square.scale(0.75).spawn()
    center_square.rotate(90, OUT)
    center_square.move(RIGHT * 1)
    with Seq(duration=5):
        center_square.rotate(360, OUT, about=ORIGIN)

    center_square.wait()
    # You can even animate the parent and a child at the same time.
    with Sync(duration=5):
        center_square.rotate(90, OUT)
        outer_squares[0].rotate(180, UP)

    Scene.save_video()

How changes propagate
=====================

What "propagate" means depends on the attribute:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Change to the parent
     - Effect on each child
   * - :attr:`~algan.animatable_base.mob.Mob.location`
     - The child moves by the same displacement.
   * - :attr:`~algan.animatable_base.mob.Mob.basis` (via :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` / :meth:`~algan.animatable_base.mob.Mob.scale`)
     - The child's basis is rotated or scaled the same way, *and* its position
       relative to the parent is preserved.
   * - :ref:`color <reference-mob-color>`, :ref:`opacity <reference-mob-opacity>`, :ref:`glow <reference-mob-glow>`
     - The child gets the same change.
   * - :meth:`~algan.animatable_base.animatable.Animatable.spawn` / :meth:`~algan.animatable_base.animatable.Animatable.despawn`
     - The child spawns or despawns too.

The upshot for geometry is that a child behaves as though bolted to the parent by
a rigid pole: rotate the parent and the children swing around with it, keeping
their orientation relative to the parent fixed.

Changes made **directly to a child** ignore the parent relationship entirely, so
you can animate a child independently, as in the last block of the example above.

.. seealso::

    :doc:`updaters` -- for a relation the parent-child rule cannot express, such
    as one mob following another *without* also taking its orientation.

Changing the hierarchy mid-scene
================================

The hierarchy is read when an animation is **recorded**, not when it plays. So
attaching, detaching and re-parenting all take effect immediately, and only
affect the animations you record after them:

.. code-block:: python

    first.add_children(square)
    first.move(RIGHT * 3)       # square travels with first

    first.remove_child(square)
    second.add_children(square)
    second.move(UP * 4)         # square travels with second
    first.move(RIGHT * 10)      # square stays where it is

Play that back and the square follows ``first`` over the first second and
``second`` over the next -- re-parenting does not reach backwards and rewrite an
animation that was already recorded.

Re-parenting never moves the mob. Algan keeps positions in world space rather
than relative to a parent, so a mob keeps exactly where it is when it changes
hands, and simply starts taking the new parent's changes instead of the old
one's.

A mob can have **more than one parent**, in which case it accumulates every
parent's changes. That is what lets overlapping ``Group``\ s each arrange the
same member::

    Group(a, b).arrange_in_line(RIGHT)
    Group(b, c).arrange_in_line(UP)     # b takes both

Bear in mind that :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children`
does not detach the mob from any parent it already has, so an intended *move*
from one parent to another is two calls -- ``remove_child`` (or
:meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_parent`,
which is the same detachment from the other side) and then ``add_children``.
Leave the first one out and the mob is driven by both.

.. warning::

    An :doc:`updater <updaters>` is the exception: it is re-run for every frame
    it covers, against the hierarchy as it stands at the end of the script. Change
    the hierarchy while an updater is live and the change applies to every frame
    that updater covers, including frames before the change. Add and remove
    updaters around a hierarchy edit rather than across one.

    Algan says so when it happens, with a
    :class:`~algan.errors.HierarchyChangedDuringUpdaterWarning` naming the
    updater and the Mob whose children changed, at the line that changed them::

        square.add_updater(spin)
        Scene.wait(1)
        parent.remove_child(square)   # <- warns: spin is still running
        Scene.wait(1)

    The fix is to bracket the edit instead::

        updater_id = square.add_updater(spin)
        Scene.wait(1)
        square.remove_updater(updater_id)
        parent.remove_child(square)
        square.add_updater(spin)
        Scene.wait(1)

Inspecting the hierarchy
========================

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Accessor
     - Returns
   * - :ref:`children <reference-mob-children>`
     - This Mob's direct children. **Read-only** -- always add through
       :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children`.
   * - :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.get_descendants`
     - Children, grandchildren and so on, plus this Mob.
   * - :ref:`parents <reference-mob-parents>`
     - The Mobs whose changes this one follows -- there can be several.
       **Read-only** -- always add through
       :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_parent`.
   * - :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_child`, :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_parent`
     - Detach one link, from either side. Both drop it in both directions.
   * - :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_parent`, :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.replace_children`
     - Attach from the child's side, or swap the whole child list.

Groups
======

:class:`~algan.mobs.group.Group` wraps a collection of Mobs so you can treat them as one. It
creates an empty Mob at the centre of the collection and adds everything in the collection as its
children, so all of the propagation rules above apply:
rotating the Group turns each member about the Group's centre, and setting the
Group's color sets every member's.

.. algan:: ChildMobsGroup

    from algan import *

    group = Group([Square().scale(0.35).move(RIGHT * x) for x in (-1, 0, 1)])
    group.spawn()

    with Seq():
        group.rotate(180, OUT)   # the whole row turns about its centre
        group.color = BLUE       # every member changes
        group.move(UP * 0.5)

    Scene.save_video()

Arranging
=========

:meth:`~algan.mobs.group.Group.arrange_in_line` spreads the members along a direction;
:meth:`~algan.mobs.group.Group.arrange_in_grid` lays them out in rows and columns. Both are
ordinary animations, so members slide into place rather than jumping.

.. algan:: ChildMobsArrange

    from algan import *

    group = Group([Square(color=BLUE).scale(0.3) for _ in range(9)]).spawn()

    group.arrange_in_line(RIGHT)                    # a row
    group.arrange_in_line(DOWN, buffer=0.2)         # a tight column
    group.arrange_in_grid(3)                        # 3 rows
    group.arrange_in_grid(3, row_buffer=1.0)        # 3 rows, generously spaced

    Scene.save_video()

``arrange_in_line`` also takes ``align_to`` to line the members up on
an edge rather than their centres, and ``equal_widths`` to space centres
evenly instead of leaving equal gaps. ``arrange_in_grid`` takes
``row_direction`` / ``column_direction`` to control which way the grid fills.

.. seealso::

    :doc:`../advanced_user_tutorials/positioning_and_layout` -- the rest of the
    positioning methods, including screen-relative placement and
    :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen`,
    which is what you usually want after arranging a group.

Indexing
========

Groups are indexable and iterable, so you can reach individual members without
keeping a separate list:

.. algan:: ChildMobsGroupIndexing

    from algan import *

    group = Group([Circle(color=BLUE).scale(0.4) for _ in range(6)])
    group.arrange_in_line(RIGHT, buffer=0.3).spawn()

    with Lag(0.4):
        for circle in group:
            circle.color = YELLOW

    group[0].move(UP)

    group.arrange_in_grid()

    Scene.save_video()
