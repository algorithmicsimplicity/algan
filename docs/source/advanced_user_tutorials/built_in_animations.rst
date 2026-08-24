====================
Built-in Animations
====================

Beyond the Mob methods, Algan ships a set of ready-made animations for the
things explanatory videos do over and over: drawing attention to something,
moving along a path, and applying a mathematical transformation to a whole
diagram.

They are all plain functions that take a Mob and record their animation on the
timeline, so they compose with :doc:`animation contexts <combining_animations>`
exactly like anything else:

.. code-block:: python

    with Sync():          # both at once
        Indicate(circle)
        Indicate(square)

    with Lag(0.3):        # a ripple down a list
        for mob in mobs:
            Indicate(mob)

Most of them take their own ``run_time``, which overrides the enclosing
context's timing for that animation.

Drawing Attention
=================

.. algan:: AnimationsIndication

    from algan import *

    grid = Group([Square(color=BLUE).scale(0.35) for _ in range(9)])
    grid.arrange_in_grid(3, buffer=0.5).spawn()

    target = grid[4]
    Indicate(target)
    Circumscribe(target)
    Flash(target)
    FocusOn(target)

    Scene.save_video()

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Animation
     - What it does
   * - :func:`~.Indicate`
     - Briefly scales the Mob up and tints it. The default "look here".
       ``scale_factor``, ``color``.
   * - :func:`~.Circumscribe`
     - Traces a shape around the Mob. ``shape``, ``buff``, ``fade_in``,
       ``fade_out``.
   * - :func:`~.Flash`
     - Fires a burst of short lines outward from a point or Mob.
       ``num_lines``, ``line_length``, ``flash_radius``.
   * - :func:`~.FocusOn`
     - Dims everything else and closes in on a point. ``opacity``, ``color``.
   * - :func:`~.Wiggle`
     - Rocks the Mob back and forth. ``n_wiggles``, ``rotation_angle``.
   * - :func:`~.Blink`
     - Flicks the Mob's visibility. ``blinks``, ``time_on``, ``time_off``,
       ``hide_at_end``.

.. algan:: AnimationsWiggleBlink

    from algan import *

    a = Square(color=BLUE).scale(0.6).move(LEFT * 2).spawn()
    b = Circle(color=YELLOW).scale(0.6).move(RIGHT * 2).spawn()
    with Sync():
        Wiggle(a)
        Blink(b, blinks=2)

    Scene.save_video()

Highlighting an Outline
=======================

:func:`~.ShowPassingFlash` sends a bright segment travelling along a Mob's
outline -- the standard way to trace a path or emphasise the boundary of a
region:

.. algan:: AnimationsPassingFlash

    from algan import *

    outline = Circle(radius=2, color=BLUE, border_width=6).spawn()
    ShowPassingFlash(outline, run_time=2)
    ShowPassingFlash(outline, run_time=2)

    Scene.save_video()

``time_width`` controls how long the travelling segment is, as a fraction of the
whole outline. :func:`~.ShowPassingFlashWithThinningStrokeWidth` does the same
with a tapering stroke.

To draw a shape on as if by hand, use :func:`~.draw_border_then_fill` -- it
traces the outline and then floods the fill:

.. code-block:: python

    draw_border_then_fill([circle, square], run_time=2)

For text, :meth:`~algan.mobs.text.Tex.write` is the glyph-wise shorthand for the same effect
(see :doc:`text_and_math`).

An indefinitely repeating version of the same idea is
:class:`~.AnimatedBoundary`, which keeps redrawing an outline around a Mob for as
long as you leave it there. Unlike everything else on this page it is a Mob rather
than a function, so spawn it, and call :meth:`~.AnimatedBoundary.stop` to freeze
it:

.. algan:: AnimationsAnimatedBoundary

    from algan import *

    with Off():
        square = Square(color=TRANSPARENT, border_width=0).scale(1.5).spawn()
        boundary = AnimatedBoundary(square, max_stroke_width=10, cycle_rate=1.0).spawn()

    square.wait(3)

    Scene.save_video()

Give the source Mob no border of its own (as above), or the travelling highlight
is drawn over the top of it and you will not see it. ``cycle_rate`` sets how fast
the outline is traced and ``colors`` the palette it cycles through.

Moving Along a Path
===================

:func:`~.MoveAlongPath` walks a Mob along another Mob's outline. Any Bezier Mob
works as the path -- a circle, a polygon, a hand-drawn curve:

.. algan:: AnimationsMoveAlongPath

    from algan import *

    path = Circle(radius=2, color=GREY).spawn()
    dot = Dot(color=YELLOW).spawn()
    MoveAlongPath(dot, path, run_time=3)

    Scene.save_video()

If you want the path invisible, spawn it inside ``with Off():`` and set its
opacity to zero, or simply never spawn it -- the geometry is read from the Mob,
not from the screen.

Transforming Whole Diagrams
===========================

These apply a mathematical map to every point of a Mob, which is how you show a
linear transformation, a change of coordinates, or a flow.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Animation
     - What it does
   * - :func:`~.ApplyMatrix`
     - Applies a 2×2 or 3×3 matrix. The clearest way to show a linear map.
   * - :func:`~.ApplyPointwiseFunction`
     - Applies any point-to-point function.
   * - :func:`~.ApplyComplexFunction`
     - Treats the xy-plane as the complex plane and applies a complex function.
   * - :func:`~.Homotopy`
     - A time-dependent deformation ``(x, y, z, t) -> (x, y, z)``.
   * - :func:`~.ComplexHomotopy`
     - The same, in complex coordinates.
   * - :func:`~.PhaseFlow`
     - Integrates a vector field, flowing points along it.
   * - :func:`~.ApplyWave`
     - Sends a ripple across the Mob.

.. algan:: AnimationsApplyMatrix

    from algan import *
    import torch

    grid = Group([Square(color=BLUE).scale(0.45) for _ in range(16)])
    grid.arrange_in_grid(4, buffer=0.1).spawn()

    ApplyMatrix(grid, torch.tensor([[1.0, 0.6], [0.0, 1.0]]), run_time=2)

    Scene.save_video()

A homotopy receives the coordinates *and* the animation's progress, so it can
deform continuously rather than just interpolate between two states:

.. algan:: AnimationsHomotopy

    from algan import *
    import torch

    grid = Group([Square(color=BLUE).scale(0.3) for _ in range(16)])
    grid.arrange_in_grid(4, buffer=0.15).spawn()

    def swirl(x, y, z, t):
        angle = t * 1.5 * torch.exp(-(x ** 2 + y ** 2) / 6)
        return (x * torch.cos(angle) - y * torch.sin(angle),
                x * torch.sin(angle) + y * torch.cos(angle),
                z)

    Homotopy(grid, swirl, run_time=3)

    Scene.save_video()

:func:`~.PhaseFlow` takes a vector field instead and integrates it, which is the
natural way to visualise a differential equation:

.. algan:: AnimationsPhaseFlow

    from algan import *
    import torch

    dots = Group([Dot(color=YELLOW).scale(1.5) for _ in range(25)])
    dots.arrange_in_grid(5, buffer=1.0).spawn()

    def rotation_field(points):
        return torch.stack((-points[..., 1], points[..., 0],
                            torch.zeros_like(points[..., 2])), -1) * 0.5

    PhaseFlow(dots, rotation_field, run_time=3, virtual_time=2.0)

    Scene.save_video()

``virtual_time`` is how much of the field's own time to integrate over, and
``integration_steps`` how finely. Both are independent of ``run_time``, which is
only how long the viewer watches it.

.. important::

    All of these functions receive batched torch tensors, not individual points,
    and must return tensors of the same shape. Write them with torch operations
    (``torch.cos``, ``torch.exp``, ...) rather than the ``math`` module, and do
    not loop over points.

:func:`~.ApplyWave` is the simplest of the family and needs no function at all:

.. algan:: AnimationsApplyWave

    from algan import *

    text = Text("wave me", font_size=72).spawn()
    ApplyWave(text, run_time=2)

    Scene.save_video()

Writing Your Own
================

If none of these fit, an :func:`~.animated_function` gets you the same
capabilities -- see :ref:`Animated Functions <animated-functions>` in
:doc:`basic_animations`. For animations that run indefinitely rather than for a
fixed duration, use :doc:`updaters`.

Where to next
=============

* :doc:`updaters` -- animations that run indefinitely rather than for a fixed
  duration.
* :doc:`child_mobs` -- applying an animation to a whole hierarchy at once.
* :doc:`combining_animations` -- the contexts that decide when these run.
* :doc:`../advanced_user_tutorials/extending_algan` -- writing your own
  reusable animation.
