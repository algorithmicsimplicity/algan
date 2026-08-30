======================
Positioning and Layout
======================

Getting things where you want them on screen is most of the work in an
explanatory animation. Algan gives you three levels of control, and you should
reach for the highest one that does the job:

1. **Relative to another Mob** -- ``a.move_next_to(b, RIGHT)``. Survives changes
   to the other Mob's size and position.
2. **Relative to the screen** -- ``mob.move_to_screen_edge(UP)``,
   ``mob.move_to_screen_position(0.9, 0.1)``. Survives resolution changes.
3. **Absolute world coordinates** -- ``mob.move_to(UP * 2 + LEFT * 3)``. Precise,
   but you have to know the numbers.

Every method below is a normal animation: it takes one second by default and
obeys the surrounding animation context. Wrap it in ``with Off():`` to place
something instantly, or ``with Sync():`` to run several placements at once --
:doc:`../new_user_tutorials/combining_animations` covers those contexts in full, and
:doc:`../new_user_tutorials/child_mobs` covers the :class:`~algan.mobs.group.Group` used at the end of
this page.

The Coordinate System
=====================

Algan uses a right-handed 3-D coordinate system with six unit direction
constants:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Constant
     - Vector
     - Direction
   * - ``RIGHT`` / ``LEFT``
     - ``±x``
     - Across the screen
   * - ``UP`` / ``DOWN``
     - ``±y``
     - Up and down the screen
   * - ``IN`` / ``OUT``
     - ``±z``
     - ``OUT`` is towards the viewer (out of the screen), ``IN`` is away (into the screen).
   * - ``ORIGIN``
     - ``(0, 0, 0)``
     - The centre of the world, where new Mobs start

Because they are unit vectors, you compose positions by arithmetic:
``UP * 2 + RIGHT * 3`` is two units up and three to the right.

The default camera sits at ``OUT * 7`` looking at the ``ORIGIN``. With the
default settings that makes the visible area at the origin plane roughly
**12.4 units wide by 7 units tall** -- so ``x`` runs about ``-6.2`` to ``6.2``
and ``y`` about ``-3.5`` to ``3.5``:

.. algan:: PositioningExtent

    from algan import *

    Dot(color=YELLOW).move_to(RIGHT * 6.2 + UP * 3.4).spawn()
    Dot(color=YELLOW).move_to(LEFT * 6.2 + DOWN * 3.4).spawn()
    Dot(color=RED).move_to_screen_corner((UP, LEFT)).spawn()
    Dot(color=RED).move_to_screen_corner((DOWN, RIGHT)).spawn()
    Scene.wait()

    Scene.save_video()

Those numbers change if you move the camera, change its field of view, or change
the aspect ratio -- which is exactly why the screen-relative methods below are
usually the better choice.

Absolute Placement
==================

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it does
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move`
     - Move *by* a displacement.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`
     - Move *to* an absolute point. ``arc_angle`` curves the path.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_between`
     - Move to the midpoint of two points or Mobs.
   * - :attr:`~algan.animatable_base.mob_layout.MobLayoutMixin.x`, :attr:`~algan.animatable_base.mob_layout.MobLayoutMixin.y`, :attr:`~algan.animatable_base.mob_layout.MobLayoutMixin.z`
     - Change one axis, leaving the others alone.

.. algan:: PositioningAbsolute

    from algan import *

    square = Square(color=BLUE).scale(0.5).spawn()

    square.move_to(UP * 2 + LEFT * 3)
    square.move_to(RIGHT * 3, arc_angle=120)   # swing round instead of sliding
    square.y = 0                                    # drop back to the middle row

    Scene.save_video()

Screen-Relative Placement
=========================

These resolve against the current camera, so you say where the *viewer* should
see the Mob:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it does
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_edge`
     - Rest against one screen edge, ``buffer`` away from it.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_corner`
     - Rest in a corner, named by its two edges.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_position`
     - Place at fractional screen coordinates: ``(0, 0)`` bottom-left,
       ``(1, 1)`` top-right.
   * - :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen`
     - Scale *and* move so the Mob fills a rectangle of the screen.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_off_screen`
     - Slide off-screen entirely (and despawn there, by default).

.. algan:: PositioningScreen

    from algan import *

    label = Text("center", font_size=36).spawn()
    box = Square(color=BLUE).scale(0.5).spawn()

    box.move_to(UP * 2 + LEFT * 3)
    box.move_to_screen_edge(RIGHT)
    box.move_to_screen_corner((DOWN, LEFT))
    box.move_next_to(label, UP)
    box.move_to_screen_position(0.9, 0.1)

    Scene.save_video()

The gap left by :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_edge`, :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_corner` and :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_next_to` is measured from the Mob's *boundary*, not its centre,
so a big shape and a small shape both end up equally inset. It defaults to
``SETTINGS.style.buffer`` (``0.6`` world units) and every one of those methods
takes a ``buffer`` argument to override it.

:meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen` is the quickest way to say "put this
diagram in the left half of the frame":

.. algan:: PositioningFitToScreen

    from algan import *

    diagram = Group([Square(color=BLUE).scale(0.3).move(RIGHT * x + UP * y)
                     for x in (-1, 0, 1) for y in (-1, 1)]).spawn()

    diagram.fit_to_screen((0.0, 0.0), (0.5, 1.0))   # left half
    diagram.fit_to_screen()                          # whole screen

    Scene.save_video()

It works on the bounding box of the whole hierarchy, so calling it on a
:class:`~algan.mobs.group.Group` lays out the entire collection at once and preserves the
members' relative positions.

Relative Placement
==================

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it does
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_next_to`
     - Sit just beside another Mob, edge to edge.
   * - :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.align_with`
     - Line two Mobs up along one axis. ``anchor='center'`` lines their centres
       up, ``'boundary'`` their ``direction``-side edges, and ``'edge'`` brings
       this Mob's far side up against the other's near side.

``move_next_to`` also takes ``align_edge``, which adds a secondary alignment --
so two Mobs placed side by side can additionally share a bottom edge:

.. algan:: PositioningAlignEdge

    from algan import *

    with Off():
        chart = Rectangle(width=3, height=2, color=BLUE).spawn()
        caption = Text("caption", font_size=32).spawn()

    caption.move_next_to(chart, RIGHT)                    # centres line up
    caption.move_next_to(chart, RIGHT, align_edge=DOWN)   # bottoms line up

    Scene.save_video()

Sizing
======

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it does
   * - :meth:`~algan.animatable_base.mob.Mob.scale`
     - Multiply the Mob's size by a factor.
   * - :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.scale_to_height`, :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.scale_to_width`
     - Scale uniformly until one dimension matches a target.
   * - :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_width`, :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_height`, :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_depth`
     - Measure the Mob, in world units.
   * - :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_center`, :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_bounding_box`
     - Where the Mob is and how far it extends.

Because the measurement methods return plain values, you can use one Mob to size
another:

.. algan:: PositioningSizing

    from algan import *

    square = Square(color=BLUE).spawn()
    circle = Circle(color=YELLOW).move(RIGHT * 3).spawn()

    square.scale_to_height(2.5)
    circle.scale_to_width(square.get_width())
    circle.move_next_to(square, RIGHT, buffer=0.5)

    Scene.save_video()

.. note::

    ``get_width`` and friends are read at the moment you call them, on the Mob's
    *current* state. If you need a Mob to keep tracking another one as it
    changes, use an updater instead -- see :doc:`../new_user_tutorials/updaters`.

Orientation
===========

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it does
   * - :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`
     - Turn about an axis, optionally about another point.
   * - :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.orbit`
     - Swing around a point *without* turning.
   * - :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at`
     - Turn to face a point.
   * - :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.reset_basis`
     - Return to the default orientation and scale.
   * - :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_right_direction`, :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_up_direction`, :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.get_forward_direction`
     - The Mob's own axes, as unit vectors.

The difference between ``rotate(..., about=p)`` and ``orbit(..., about=p)``
is whether the Mob turns as it travels: ``rotate`` carries the orientation
around with it (like the Moon, always showing the same face inward), ``orbit``
keeps the orientation fixed (like a carousel horse that stays upright).

Arranging Several Mobs
======================

For collections, put them in a :class:`~algan.mobs.group.Group` and let it do the arithmetic:

.. algan:: PositioningArrange

    from algan import *

    squares = [Square() for _ in range(9)]
    group = Group(squares).spawn()
    with Sync():
        group.arrange_in_line(RIGHT)
        group.fit_to_screen()
    group.wait()
    with Sync():
        group.arrange_in_grid(3)
        group.fit_to_screen()
    group.wait()

    Scene.save_video()

:meth:`~algan.mobs.group.Group.arrange_in_line` and :meth:`~algan.mobs.group.Group.arrange_in_grid` are covered
in :doc:`../new_user_tutorials/child_mobs` along with the rest of the Group and parent/child machinery.

See Also
========

* :doc:`text_and_math` -- putting labels and formulae where you just learned to
  put shapes.
* :doc:`../new_user_tutorials/combining_animations` -- the ``Off()`` and ``Sync()`` contexts used
  above, in full.
* :doc:`../new_user_tutorials/child_mobs` -- Groups, parent/child propagation, and the rest of the
  layout methods.
* :doc:`../new_user_tutorials/updaters` -- keeping one Mob tracking another as it
  changes, which none of these one-shot methods does.
* :doc:`cameras` -- moving the frame instead of the Mobs, and what the
  screen-relative methods resolve against.
* :doc:`../galleries/mob_gallery` -- the Mobs to position.
