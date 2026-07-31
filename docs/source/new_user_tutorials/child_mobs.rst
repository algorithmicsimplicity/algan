==========================
Child Mobs and Groups
==========================

Complex objects are built by combining simple ones. Algan gives you two ways to
do that, and they are the same mechanism underneath:

* :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children` attaches Mobs to a **parent** Mob, so
  changes to the parent propagate to them.
* :class:`~.Group` collects Mobs into a new, invisible parent, and adds layout
  helpers.

Use ``add_children`` when one of your Mobs is naturally the main body of the
thing; use ``Group`` when they are peers.

Parents and Children
====================

.. algan:: ChildMobsBasic

    from algan import *

    parent_mob = Square(color=BLUE)
    children_mobs = [Square(location=loc) for loc in [LEFT * 2.5, UP * 2.5, RIGHT * 2.5, DOWN * 2.5]]

    parent_mob.add_children(children_mobs)  # this is the crucial step

    # Now any change to the parent propagates to the children,
    # including spawning.
    parent_mob.scale(0.75).spawn()
    parent_mob.rotate(90, OUT)
    parent_mob.move(RIGHT * 1)
    with Seq(run_time=5):
        parent_mob.rotate(360, OUT, about_point=ORIGIN)

    parent_mob.wait()
    # You can even animate the parent and a child at the same time.
    with Sync(run_time=5):
        parent_mob.rotate(90, OUT)
        children_mobs[0].rotate(180, UP)

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
       relative to the parent is preserved -- expressed in the parent's basis,
       the child's location does not change.
   * - :ref:`color <reference-mob-color>`, :ref:`opacity <reference-mob-opacity>`, :ref:`glow <reference-mob-glow>`
     - The child gets the same change.
   * - :meth:`~algan.animatable_base.animatable.Animatable.spawn` / :meth:`~algan.animatable_base.animatable.Animatable.despawn`
     - The child spawns or despawns too.

The upshot for geometry is that a child behaves as though bolted to the parent by
a rigid pole: rotate the parent and the children swing around with it, keeping
their orientation relative to it.

Changes made **directly to a child** ignore the parent relationship entirely, so
you can animate a child independently, as in the last block of the example above.

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
   * - :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_child`, :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_parent`
     - Detach a link.
   * - :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.set_parent_to`, :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.replace_children`
     - Re-parent, or swap the whole child list.

Groups
======

:class:`~algan.mobs.group.Group` wraps a collection of Mobs so you can treat them as one. It
creates an empty Mob at the centre of the collection and adds everything as its
children, so all of the propagation rules above apply.

.. algan:: ChildMobsGroup

    from algan import *

    mobs = [Square() for _ in range(9)]
    group = Group(mobs)
    group.scale(1 / 3).spawn()
    group.arrange_in_line(RIGHT)
    group.wait()
    with Sync():
        group.scale(2)
        group.arrange_in_grid(3)
    group.wait()

    Scene.save_video()

Groups are indexable and iterable, so you can reach individual members without
keeping a separate list:

.. algan:: ChildMobsGroupIndexing

    from algan import *

    group = Group([Circle(color=BLUE).scale(0.4) for _ in range(6)])
    group.arrange_in_line(RIGHT, buffer=0.3).spawn()

    with Lag(0.4):
        for mob in group:
            mob.color = YELLOW

    group[0].move(UP)

    Scene.save_video()

Arranging
=========

:meth:`~algan.mobs.group.Group.arrange_in_line` spreads the members along a direction;
:meth:`~algan.mobs.group.Group.arrange_in_grid` lays them out in rows and columns. Both are
ordinary animations, so members slide into place rather than jumping.

.. code-block:: python

    group.arrange_in_line(RIGHT)                    # a row
    group.arrange_in_line(DOWN, buffer=0.2)         # a tight column
    group.arrange_in_grid(3)                        # 3 rows
    group.arrange_in_grid(3, buffer=1.0)            # 3 rows, generously spaced

.. important::

    Both arrangements use a uniform cell size, taken from the largest member. If
    one Mob is much bigger than the rest -- or bigger than the frame -- the whole
    layout inflates to match it and can end up off-screen. Give your Mobs
    comparable sizes before arranging, or :meth:`~algan.animatable_base.mob.Mob.scale` the group
    afterwards.

    A common trap: :class:`~.Torus` defaults to ``major_radius=3``, which is
    wider than the visible frame.

``arrange_in_line`` also takes ``alignment_direction`` to line the members up on
an edge rather than their centres, and ``equal_displacement`` to space centres
evenly instead of leaving equal gaps. ``arrange_in_grid`` takes
``row_direction`` / ``column_direction`` to control which way the grid fills.

To place a whole group on screen as a unit, :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen_rectangle`
scales and moves it in one call -- see :doc:`positioning_and_layout`.

Sub-Mobs
========

Indexing a Mob that has internal structure gives you a view onto part of it.
:ref:`character_mobs <reference-text-character-mobs>` is the most useful case (see :doc:`text_and_math`),
and multi-part :class:`~.Tex` exposes its pieces as ``children``.

.. note::

    A sub-Mob obtained by indexing shares its source's identity, and therefore
    its lifespan: it is spawned and despawned with the whole. A
    :meth:`~algan.animatable_base.animatable.Animatable.clone` is independent.
