===================
Built-in Animations
===================

Beyond standard Mob methods, Algan ships with a collection of ready-made
animations for common video patterns: drawing attention to elements, moving
objects along paths, and applying mathematical transformations across entire
diagrams.

These are standard functions that take a Mob and record animations on the
timeline, so they compose seamlessly with :doc:`animation contexts
<../new_user_tutorials/combining_animations>`:

.. algan:: AnimationsCompose

    from algan import *

    circle = Circle(color=BLUE).scale(0.6).move(LEFT * 2).spawn()
    square = Square(color=YELLOW).scale(0.6).move(RIGHT * 2).spawn()

    with Sync():          # Both at the same time
        Indicate(circle)
        Indicate(square)

    with Lag(0.3):        # Cascading ripple across elements
        for shape in (circle, square):
            Indicate(shape)

    Scene.save_video()

Most animation functions take an optional ``runtime`` argument, which overrides
the enclosing context's timing for that specific animation.

Drawing Attention
=================

Here are the built-in ways to draw the viewer's eye to a specific Mob:

.. algan:: AnimationsIndication

    from algan import *

    grid = Group([Square(color=BLUE).scale(0.35) for _ in range(9)])
    grid.arrange_in_grid(3, row_buffer=0.5).spawn()

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
     - Briefly scales the Mob up and tints its color. The standard "look here" effect.
   * - :func:`~.Circumscribe`
     - Draws an animated bounding outline around the Mob.
   * - :func:`~.Flash`
     - Emits a quick radial burst of short rays from a point or Mob.
   * - :func:`~.FocusOn`
     - Dims the rest of the scene and targets a specific point.
   * - :func:`~.Wiggle`
     - Wiggles the Mob back and forth.
   * - :func:`~.Blink`
     - Rapidly toggles the Mob's visibility.

.. algan:: AnimationsWiggleBlink

    from algan import *

    square = Square(color=BLUE).scale(0.6).move(LEFT * 2).spawn()
    circle = Circle(color=YELLOW).scale(0.6).move(RIGHT * 2).spawn()
    with Sync():
        Wiggle(square)
        Blink(circle, blinks=2)

    Scene.save_video()

Highlighting an Outline
=======================

:func:`~.ShowPassingFlash` sends a bright segment travelling along a Mob's
outline; the standard way to trace a path or emphasise the boundary of a
region:

.. algan:: AnimationsPassingFlash

    from algan import *

    outline = Circle(radius=2, color=BLUE, stroke_width=6).spawn()
    ShowPassingFlash(outline, runtime=2)
    ShowPassingFlash(outline, runtime=2)

    Scene.save_video()

``time_width`` controls how long the travelling segment is, as a fraction of the
whole outline. :func:`~.ShowPassingFlashWithThinningStrokeWidth` does the same
with a tapering stroke.

To draw a shape on screen as if by hand, use :func:`~.DrawBorderThenFill`. It
first traces the outer border and then animates the fill:

.. algan:: AnimationsDrawBorderThenFill

    from algan import *

    circle = Circle(color=BLUE).scale(0.8).move(LEFT * 2).spawn(False)
    square = Square(color=YELLOW).scale(0.8).move(RIGHT * 2).spawn(False)

    DrawBorderThenFill([circle, square], runtime=2)

    Scene.save_video()

Notice that we called ``spawn(False)`` so the shapes don't play their default
fade-in before being drawn.

For text and LaTeX, :meth:`~algan.mobs.text.Tex.write` provides the convenient
glyph-by-glyph handwriting equivalent (see
:doc:`../advanced_user_tutorials/text_and_math`).

An indefinitely repeating version of the same idea is
:class:`~.AnimatedBoundary`, which keeps redrawing an outline around a Mob for as
long as you leave it there. Unlike everything else on this page it is a Mob rather
than a function, so spawn it, and call :meth:`~.AnimatedBoundary.stop` to freeze
it:

.. algan:: AnimationsAnimatedBoundary

    from algan import *

    with Off():
        square = Square(color=TRANSPARENT, stroke_width=0).scale(1.5).spawn()
        boundary = AnimatedBoundary(square, max_stroke_width=5, cycle_rate=1.0).spawn()

    square.wait(3)

    Scene.save_video()

Give the source Mob no border of its own (as above), or the travelling highlight
is drawn over the top of it and you will not see it. ``cycle_rate`` sets how fast
the outline is traced and ``colors`` the palette it cycles through.

Moving Along a Path
===================

:func:`~.MoveAlongPath` moves a Mob along the trajectory of another Mob's
outline. Any curve, line, or polygon can serve as the path:

.. algan:: AnimationsMoveAlongPath

    from algan import *

    path = Circle(radius=2, color=GREY).spawn()
    dot = Dot(color=YELLOW).spawn()
    MoveAlongPath(dot, path, runtime=3)

    Scene.save_video()

If you want the path itself to be invisible, you can leave it unspawned, the
geometry is read directly from the Mob object.

Transforming Whole Diagrams
===========================

These functions apply spatial or mathematical mappings across all points of a
Mob, making it easy to illustrate linear transformations, coordinate changes, or
vector flows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Animation
     - What it does
   * - :func:`~.ApplyMatrix`
     - Applies a 2×2 or 3×3 transformation matrix.
   * - :func:`~.ApplyPointwiseFunction`
     - Applies any point-to-point function.
   * - :func:`~.ApplyComplexFunction`
     - Treats the xy-plane as the complex plane and applies a complex function.
   * - :func:`~.Homotopy`
     - A continuous time-dependent deformation ``(x, y, z, t) -> (x, y, z)``.
   * - :func:`~.ComplexHomotopy`
     - Time-dependent deformation on the complex plane.
   * - :func:`~.PhaseFlow`
     - Integrates a vector field to flow points along it over time.
   * - :func:`~.ApplyWave`
     - Propagates a wave distortion across a Mob.

.. algan:: AnimationsApplyMatrix

    from algan import *
    import torch

    grid = Group([Square(color=BLUE).scale(0.45) for _ in range(16)])
    grid.arrange_in_grid(4, row_buffer=0.1).spawn()

    ApplyMatrix(grid, torch.tensor([[1.0, 0.6], [0.0, 1.0]]), runtime=2)

    Scene.save_video()

A homotopy receives the coordinates *and* the animation's progress, so it can
deform continuously rather than just interpolate between two states:

.. algan:: AnimationsHomotopy

    from algan import *
    import torch

    grid = Group([Square(color=BLUE).scale(0.3) for _ in range(16)])
    grid.arrange_in_grid(4, row_buffer=0.15).spawn()

    def swirl(x, y, z, t):
        angle = t * 1.5 * torch.exp(-(x ** 2 + y ** 2) / 6)
        return (x * torch.cos(angle) - y * torch.sin(angle),
                x * torch.sin(angle) + y * torch.cos(angle),
                z)

    Homotopy(grid, swirl, runtime=3)

    Scene.save_video()

:func:`~.PhaseFlow` takes a vector field instead and integrates it, which is the
natural way to visualise a differential equation:

.. algan:: AnimationsPhaseFlow

    from algan import *
    import torch

    dots = Group([Dot(color=YELLOW).scale(1.5) for _ in range(25)])
    dots.arrange_in_grid(5, row_buffer=1.0).spawn()

    def rotation_field(points):
        return torch.stack((-points[..., 1], points[..., 0],
                            torch.zeros_like(points[..., 2])), -1) * 0.5

    PhaseFlow(dots, rotation_field, runtime=3, virtual_time=2.0)

    Scene.save_video()

``virtual_time`` is how much of the field's own time to integrate over, and
``integration_steps`` how finely. Both are independent of ``runtime``, which is
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
    ApplyWave(text, runtime=2)

    Scene.save_video()

Custom Animations
=================

If you need an animation that isn't covered here, you can write your own using
:func:`~algan.animatable_base.animatable.animated_function` (see
:doc:`../advanced_user_tutorials/custom_animations`), or attach a continuous
updater (see :doc:`../new_user_tutorials/updaters`).

Where To Next
=============

* :doc:`../new_user_tutorials/updaters` -- animations that run indefinitely rather than for a fixed
  runtime.
* :doc:`../new_user_tutorials/child_mobs` -- applying an animation to a whole hierarchy at once.
* :doc:`../new_user_tutorials/combining_animations` -- the contexts that decide when these run.
* :doc:`../advanced_user_tutorials/custom_animations` -- writing an animation of
  your own with :func:`~.animated_function`.
* :doc:`../advanced_user_tutorials/animating_out_of_order` -- scheduling these
  animations at times you compute yourself.
* :doc:`mob_gallery` -- the Mobs to apply them to.
