"""Attention animations: drawing the viewer's eye to something.

These are the "look here" effects. :func:`Indicate` swells a Mob and flashes its
color; :func:`Wiggle` shakes it; :func:`Blink` flickers it; :func:`FocusOn`
dims everything else and closes a spotlight onto a point; the passing-flash
family runs a highlight along an outline.

Each is an animated function returning a Mob, so it composes with the animation
contexts like any other change -- put several in a :class:`~.Sync` to indicate
things together, or a :class:`~.Lag` to sweep across them.

:func:`there_and_back` and :func:`wiggle` are the rate functions behind them, and
are useful on their own as ``easing`` arguments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animation_timeline.animation_contexts import (
    Off,
    Seq,
    Sync,
    animation_manager_for,
)
from algan.animations.movement import Homotopy
from algan.constants import easings
from algan.constants.color import GRAY, YELLOW
from algan.constants.math import RADIANS_TO_DEGREES
from algan.constants.spatial import OUTWARD, UP
from algan.geometry.geometry import get_rotation_around_axis
from algan.utils.tensor_utils import cast_to_tensor, squish, unsquish


def there_and_back(t, inflection: float = 10.0):
    """Rate function that eases out to 1 and back to 0.

    Progress rises over the first half and returns over the second, which turns any
    animation into a there-and-back gesture.

    Parameters
    ----------
    t
        Animation progress, ``0`` to ``1``.
    inflection
        Steepness of the easing at the midpoint; larger is snappier. Defaults to
        ``10.0``.

    Returns
    -------
    torch.Tensor
        Adjusted progress, ``0`` at both ends and ``1`` in the middle.
    """
    t = cast_to_tensor(t)
    new_t = torch.where(t < 0.5, 2.0 * t, 2.0 * (1.0 - t))
    return easings.smooth(new_t, inflection)


def wiggle(t, wiggles: int = 2):
    """Rate function that oscillates, with the swing fading in and out.

    A sine wave scaled by :func:`there_and_back`, so the oscillation grows towards
    the middle and settles by the end rather than stopping mid-swing.

    Parameters
    ----------
    t
        Animation progress, ``0`` to ``1``.
    wiggles
        How many half-oscillations to perform. Defaults to ``2``.

    Returns
    -------
    torch.Tensor
        Signed oscillation, starting and ending at ``0``.
    """
    t = cast_to_tensor(t)
    val = torch.sin(wiggles * math.pi * t)
    return there_and_back(t) * val


@animated_function(animated_args={"interpolation": 0.0})
def _indicate_scale_step(mobject, scale_factor, interpolation=1.0):
    """Apply one frame of Indicate's relative there-and-back scale pulse."""
    interpolation = cast_to_tensor(interpolation)
    scale_factor = cast_to_tensor(scale_factor).to(interpolation)
    pulse = 1.0 - (2.0 * interpolation - 1.0).abs()
    # Go through Mob.scale rather than writing the derived scale_coefficient
    # timeline directly. The scale property ultimately updates basis (the
    # renderer's source of truth) and carries descendant locations with it.
    mobject.scale(1.0 + (scale_factor - 1.0) * pulse)
    return mobject


def Indicate(mobject, scale_factor: float = 1.2, color=YELLOW, duration: float = 1.0):
    """Draw attention to a Mob by briefly growing and recoloring it.

    The Mob swells and flashes color, then returns to exactly how it was -- the
    standard "look here" gesture.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. The scale pulse is relative to each part's own size, so a
    composite Mob whose parts were scaled separately keeps its proportions.

    Parameters
    ----------
    mobject
        The Mob to indicate.
    scale_factor
        How large the Mob grows at the peak, as a multiple of its current size.
        Defaults to ``1.2``.
    color
        Color to flash. Defaults to ``YELLOW``.
    duration
        Duration of the whole gesture, in seconds. Defaults to ``1.0``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    color = cast_to_tensor(color)
    scale_factor = cast_to_tensor(scale_factor)
    with Sync(duration=duration, animation_manager=animation_manager_for(mobject)):
        mobject.pulse_color(color)
        _indicate_scale_step(mobject, scale_factor)
    return mobject


@animated_function(
    animated_args={"t": 0.0},
    unique_args=[
        "basis_0",
        "location_0",
        "scale_value",
        "rotation_angle",
        "n_wiggles",
        "scale_about_point",
        "rotate_about_point",
    ],
)
def wiggle_step(
    mob,
    t,
    basis_0,
    location_0,
    scale_value,
    rotation_angle,
    n_wiggles,
    scale_about_point,
    rotate_about_point,
):
    """Internal: place a Mob at one instant of a :func:`Wiggle`.

    Rebuilds the Mob's basis and location from the values it had before the wiggle
    started, so the effect leaves no drift. Use :func:`Wiggle`.

    Parameters
    ----------
    mob
        The Mob being wiggled.
    t
        Animation progress, ``0`` to ``1``, supplied per frame.
    basis_0
        The Mob's basis before the wiggle began.
    location_0
        The Mob's location before the wiggle began.
    scale_value
        Peak scale, as a multiple of the original size.
    rotation_angle
        Peak rotation, **in radians** (converted internally to degrees).
    n_wiggles
        Number of half-oscillations.
    scale_about_point
        Point to scale around, or ``None`` to scale in place.
    rotate_about_point
        Point to rotate around, or ``None`` to rotate in place.
    """
    t = cast_to_tensor(t)
    s_val = 1.0 + (scale_value - 1.0) * there_and_back(t)
    rot_deg = wiggle(t, n_wiggles) * rotation_angle * RADIANS_TO_DEGREES

    R = get_rotation_around_axis(rot_deg, OUTWARD, dim=-1)
    R_basis = R.view(-1, 1, 3, 3)

    new_basis = squish(
        unsquish(basis_0, -1, 3) @ R_basis * s_val.view(-1, 1, 1, 1), -2, -1
    )

    loc = location_0.clone()
    if rotate_about_point is not None:
        rp = cast_to_tensor(rotate_about_point)
        R_loc = R.view(-1, 1, 3, 3)
        loc = rp + ((loc - rp).unsqueeze(-2) @ R_loc).squeeze(-2)
    if scale_about_point is not None:
        sp = cast_to_tensor(scale_about_point)
        loc = sp + (loc - sp) * s_val.view(-1, 1, 1)

    mob.basis = new_basis
    mob.location = loc


def Wiggle(
    mobject,
    scale_value: float = 1.1,
    rotation_angle: float = 0.01 * math.pi * 2,
    n_wiggles: int = 6,
    scale_about_point=None,
    rotate_about_point=None,
    duration: float = 2.0,
):
    """Shake a Mob back and forth, as if jostled.

    The Mob rocks a few degrees either way while swelling slightly, then settles
    exactly where it started. Reads as "this thing is trying to get your attention"
    without moving it anywhere.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. Position and orientation are rebuilt each frame from the
    pre-wiggle state, so nothing accumulates.

    Parameters
    ----------
    mobject
        The Mob to wiggle.
    scale_value
        Peak size during the wiggle, as a multiple of the current size. Defaults to
        ``1.1``.
    rotation_angle
        Peak rocking angle, **in radians** -- unusually for Algan, since this
        mirrors Manim's signature. Defaults to ``0.02 * pi`` (about 3.6 degrees).
    n_wiggles
        How many times the Mob rocks. Defaults to ``6``.
    scale_about_point
        Point to scale around, shape ``(*, 3)``. Defaults to ``None``, meaning scale
        in place.
    rotate_about_point
        Point to rock around, shape ``(*, 3)``. Defaults to ``None``, meaning rock in
        place.
    duration
        Duration of the whole wiggle, in seconds. Defaults to ``2.0``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    basis_0 = mobject.basis.clone()
    location_0 = mobject.location.clone()
    with Sync(duration=duration, animation_manager=animation_manager_for(mobject)):
        mobject.animate_function(
            wiggle_step,
            basis_0=basis_0,
            location_0=location_0,
            scale_value=scale_value,
            rotation_angle=rotation_angle,
            n_wiggles=n_wiggles,
            scale_about_point=scale_about_point,
            rotate_about_point=rotate_about_point,
        )
    return mobject


def Blink(
    mobject,
    time_on: float = 0.5,
    time_off: float = 0.5,
    blinks: int = 1,
    hide_at_end: bool = False,
):
    """Flash a Mob on and off.

    Visibility is switched instantly rather than faded, so the Mob blinks crisply.
    Each cycle costs ``time_on + time_off`` seconds.

    Animation
    ---------
    Recorded as an animation whose duration comes from these parameters rather than
    the enclosing context. Opacity is driven through the Mob's color, so parts with
    their own colors keep them.

    Parameters
    ----------
    mobject
        The Mob to blink.
    time_on
        Seconds visible per cycle. Defaults to ``0.5``.
    time_off
        Seconds hidden per cycle. Defaults to ``0.5``.
    blinks
        How many on/off cycles to perform. Defaults to ``1``.
    hide_at_end
        Whether to leave the Mob hidden when the blinking stops. Defaults to False,
        which leaves it visible.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    with Seq(animation_manager=animation_manager_for(mobject)):
        for _ in range(blinks):
            with Off(animation_manager=animation_manager_for(mobject)):
                mobject.set_opacity_via_color(1.0)
            mobject.wait(time_on)
            with Off(animation_manager=animation_manager_for(mobject)):
                mobject.set_opacity_via_color(0.0)
            mobject.wait(time_off)
        if not hide_at_end:
            with Off(animation_manager=animation_manager_for(mobject)):
                mobject.set_opacity_via_color(1.0)
            mobject.wait(time_on)
    return mobject


def FocusOn(focus_point, opacity: float = 0.2, color=GRAY, duration: float = 2.0):
    """Contract a large translucent disc onto a point, like a closing spotlight.

    Draws the eye to one spot by shrinking a tinted circle down to nothing there. The
    circle is created and removed for you.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. The spotlight is spawned and despawned instantly around it,
    so it costs no extra video time.

    Parameters
    ----------
    focus_point
        A Mob to focus on -- its center is used -- or a point of shape ``(*, 3)``.
    opacity
        Peak opacity of the disc as it closes in, ``0`` to ``1``. Defaults to ``0.2``.
    color
        Color of the disc. Defaults to ``GRAY``.
    duration
        Duration of the contraction, in seconds. Defaults to ``2.0``.

    Returns
    -------
    :class:`~algan.mobs.shapes_2d.Circle`
        The spotlight Mob, already despawned.
    """
    from algan.animatable_base.mob import Mob
    from algan.mobs.shapes_2d import Circle

    animation_manager = animation_manager_for(focus_point)
    if isinstance(focus_point, Mob):
        focus_point = focus_point.get_center()
    else:
        focus_point = cast_to_tensor(focus_point)
    with Seq(animation_manager=animation_manager):
        with Off(animation_manager=animation_manager):
            spotlight = Circle(
                scene=animation_manager.scene,
                radius=10.0,
                color=color,
                opacity=0.0,
                location=focus_point,
            ).spawn()
        with Sync(duration=duration, animation_manager=animation_manager):
            spotlight.scale(1e-4)
            spotlight.opacity = opacity
        with Off(animation_manager=animation_manager):
            spotlight.despawn(animate=False)
    return spotlight


@animated_function(
    animated_args={"t": 0.0}, unique_args=["time_width", "full_control_points"]
)
def passing_flash_step(mob, t, time_width, full_control_points):
    """Internal: show one instant of a :func:`ShowPassingFlash`.

    Reveals a moving window of the curve, sliding from start to end.

    Parameters
    ----------
    mob
        The bezier circuit being flashed.
    t
        Animation progress, ``0`` to ``1``, supplied per frame.
    time_width
        Width of the visible window as a fraction of the curve.
    full_control_points
        The circuit's complete control points, captured before the flash.
    """
    t = cast_to_tensor(t)
    upper = t * (1.0 + time_width)
    lower = upper - time_width
    upper = torch.clamp(upper, 0.0, 1.0)
    lower = torch.clamp(lower, 0.0, 1.0)
    mob._set_control_points_to_partial(full_control_points, lower, upper)


@animated_function(animated_args={"t": 0.0}, unique_args=["full_control_points"])
def draw_step(mob, t, full_control_points):
    """Internal: show one instant of a curve being drawn in.

    Reveals the curve from its start up to progress ``t``.

    Parameters
    ----------
    mob
        The bezier circuit being drawn.
    t
        Animation progress, ``0`` to ``1``, supplied per frame.
    full_control_points
        The circuit's complete control points, captured before the animation.
    """
    t = cast_to_tensor(t)
    mob._set_control_points_to_partial(full_control_points, 0.0, t)


@animated_function(animated_args={"t": 0.0}, unique_args=["full_control_points"])
def undraw_step(mob, t, full_control_points):
    """Internal: show one instant of a curve being erased.

    The reverse of :func:`draw_step`: the curve retreats towards its start.

    Parameters
    ----------
    mob
        The bezier circuit being erased.
    t
        Animation progress, ``0`` to ``1``, supplied per frame.
    full_control_points
        The circuit's complete control points, captured before the animation.
    """
    t = cast_to_tensor(t)
    mob._set_control_points_to_partial(full_control_points, 0.0, 1.0 - t)


def _show_passing_flash_on_bezier(
    mobject, time_width: float, duration: float, *, clone_source: bool
):
    animation_manager = animation_manager_for(mobject)
    with Seq(animation_manager=animation_manager):
        with Off(animation_manager=animation_manager):
            source_uses_fill = mobject.filled
            source_color = mobject.color.clone()
            flash = mobject.clone(spawn=False) if clone_source else mobject
            flash.filled = False
            if source_uses_fill and torch.any(source_color[..., -1] > 0):
                flash.stroke_color = source_color

            full_pts = flash.control_points.location.clone()
            flash._set_control_points_to_partial(full_pts, 0.0, 0.0)
            flash.spawn()
            flash.set_non_recursive(opacity=torch.ones_like(flash.opacity))

            source_is_visible = (
                clone_source and mobject.is_spawned() and not mobject.is_despawned()
            )
            if source_is_visible:
                source_opacity = mobject.opacity.clone()
                mobject.set_non_recursive(opacity=torch.zeros_like(mobject.opacity))
        with Sync(duration=duration, animation_manager=animation_manager):
            flash.animate_function(
                passing_flash_step,
                time_width=time_width,
                full_control_points=full_pts,
            )
        with Off(animation_manager=animation_manager):
            flash.despawn(animate=False)
            if source_is_visible:
                mobject.set_non_recursive(opacity=source_opacity)


def ShowPassingFlash(mobject, time_width: float = 0.1, duration: float = 1.0):
    """Run a bright segment along a curve, like a spark following a wire.

    A short piece of the curve is visible at a time and travels from one end to the
    other, leaving nothing behind. Works on any Mob built from curves; for a
    composite, every curve in it flashes at once.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. A transient stroke-only clone is spawned and despawned
    around the flash, so the Mob does not need to be spawned beforehand. A visible
    source Mob is hidden during the traversal and restored exactly afterwards.

    Parameters
    ----------
    mobject
        The curve, or a Mob containing curves, to flash along.
    time_width
        Length of the travelling segment as a fraction of the curve. Defaults to
        ``0.1``; smaller values look like a sharper spark.
    duration
        Duration of the traversal, in seconds. Defaults to ``1.0``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    from algan.mobs.bezier_circuit import BezierCircuitCubic

    if isinstance(mobject, BezierCircuitCubic):
        _show_passing_flash_on_bezier(
            mobject,
            time_width=time_width,
            duration=duration,
            clone_source=True,
        )
    else:
        beziers = [
            d for d in mobject.get_descendants() if isinstance(d, BezierCircuitCubic)
        ]
        with Sync(duration=duration, animation_manager=animation_manager_for(mobject)):
            for b in beziers:
                ShowPassingFlash(b, time_width=time_width, duration=duration)
    return mobject


def ShowPassingFlashWithThinningStrokeWidth(
    vmobject, n_segments: int = 10, time_width: float = 0.1, duration: float = 1.0
):
    """Run a tapering flash along a curve, like a comet with a tail.

    Several passing flashes are layered, each thinner and longer than the last, so
    the travelling spark trails off behind itself instead of ending abruptly.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. The layers are clones created instantly beforehand; the
    original Mob is untouched.

    Parameters
    ----------
    vmobject
        The curve to flash along. Its ``stroke_width`` sets the thickest layer.
    n_segments
        How many layers to draw. Defaults to ``10``; more is smoother and slower.
    time_width
        Length of the leading segment as a fraction of the curve. Defaults to
        ``0.1``.
    duration
        Duration of the traversal, in seconds. Defaults to ``1.0``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    widest = getattr(vmobject, "stroke_width", 5.0)
    if isinstance(widest, torch.Tensor):
        widest = widest.item()
    clones = []
    with Off(animation_manager=animation_manager_for(vmobject)):
        for i in range(n_segments):
            factor = i / (n_segments - 1) if n_segments > 1 else 1.0
            stroke_w = factor * widest
            time_w = (1.0 - factor) * time_width
            clone = vmobject.clone(spawn=False)
            clone.stroke_width = stroke_w
            clones.append((clone, time_w))
    with Sync(duration=duration, animation_manager=animation_manager_for(vmobject)):
        for clone, time_w in clones:
            _show_passing_flash_on_bezier(
                clone,
                time_width=time_w,
                duration=duration,
                clone_source=False,
            )
    return vmobject


def Flash(
    point_or_mobject,
    line_length: float = 0.2,
    num_lines: int = 12,
    flash_radius: float = 0.1,
    line_stroke_width: float = 3,
    color=YELLOW,
    time_width: float = 1.0,
    duration: float = 1.0,
):
    """Burst short lines outwards from a point, like a spark or a ping.

    Lines are arranged radially in the screen plane and each one flashes outwards,
    marking a moment at a location.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. The lines are created instantly beforehand and removed by
    their own flashes.

    Parameters
    ----------
    point_or_mobject
        A Mob to flash around -- its center is used -- or a point of shape
        ``(*, 3)``.
    line_length
        Length of each line, in world units. Defaults to ``0.2``.
    num_lines
        How many lines to arrange around the point. Defaults to ``12``.
    flash_radius
        Distance from the point at which the lines begin, in world units. Defaults to
        ``0.1``.
    line_stroke_width
        Thickness of each line. Defaults to ``3``.
    color
        Color of the lines. Defaults to ``YELLOW``.
    time_width
        Fraction of each line visible at a time. Defaults to ``1.0``, i.e. the whole
        line.
    duration
        Duration of the burst, in seconds. Defaults to ``1.0``.

    Returns
    -------
    :class:`~.Mob` or torch.Tensor
        Whatever was passed in.
    """
    from algan.animatable_base.mob import Mob
    from algan.mobs.shapes_2d import Line

    animation_manager = animation_manager_for(point_or_mobject)
    if isinstance(point_or_mobject, Mob):
        center = point_or_mobject.get_center()
    else:
        center = cast_to_tensor(point_or_mobject)
    lines = []
    with Off(animation_manager=animation_manager):
        for i in range(num_lines):
            angle = i * (2 * math.pi / num_lines)
            direction = torch.tensor(
                [math.cos(angle), math.sin(angle), 0.0], device=center.device
            )
            start = center + flash_radius * direction
            end = start + line_length * direction
            line = Line(
                start,
                end,
                scene=animation_manager.scene,
                stroke_color=color,
                stroke_width=line_stroke_width,
            )
            lines.append(line)
    with Sync(duration=duration, animation_manager=animation_manager):
        for line in lines:
            _show_passing_flash_on_bezier(
                line,
                time_width=time_width,
                duration=duration,
                clone_source=False,
            )
    return point_or_mobject


def Circumscribe(
    mobject,
    shape=None,
    fade_in: bool = False,
    fade_out: bool = False,
    time_width: float = 0.3,
    buff: float = 0.2,
    color=YELLOW,
    duration: float = 1.0,
    stroke_width: float = 3,
):
    """Trace an outline around a Mob to single it out.

    A frame is drawn around the Mob and then removed, so the highlight is a gesture
    rather than a lasting box. ``fade_in`` and ``fade_out`` choose between four
    behaviours: neither traces a passing flash around the frame, both fades the whole
    frame in and out, and either one alone combines a fade with drawing or erasing the
    outline.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration. The frame is created and despawned around it, so nothing is
    left in the scene.

    Parameters
    ----------
    mobject
        The Mob to circumscribe.
    shape
        Outline shape: ``Rectangle``, ``Square`` or ``Circle`` (the classes
        themselves, not instances). Defaults to ``None``, meaning a rectangle around
        the Mob's bounding box.
    fade_in
        Whether the frame fades in. Defaults to False.
    fade_out
        Whether the frame fades out. Defaults to False.
    time_width
        Length of the travelling segment when neither fade is used, as a fraction of
        the outline. Defaults to ``0.3``.
    buff
        Gap between the Mob and the outline, in world units. Defaults to ``0.2``.
    color
        Color of the outline. Defaults to ``YELLOW``.
    duration
        Duration of the whole gesture, in seconds. Defaults to ``1.0``.
    stroke_width
        Thickness of the outline. Defaults to ``3``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.

    Raises
    ------
    ValueError
        If ``shape`` is not ``Rectangle``, ``Square``, ``Circle`` or ``None``.
    """
    from algan.mobs.shapes_2d import Circle, Rectangle, Square, SurroundingRectangle

    animation_manager = animation_manager_for(mobject)
    if shape is None or shape in (Rectangle, Square):
        frame = SurroundingRectangle(
            mobject,
            scene=mobject.scene,
            color=color,
            buffer=buff,
            stroke_width=stroke_width,
            filled=False,
        )
    elif shape == Circle:
        bbox = mobject.get_bounding_box()
        mn = bbox.amin(-2)
        mx = bbox.amax(-2)
        center = (mn + mx) * 0.5
        width = mx[..., 0] - mn[..., 0]
        height = mx[..., 1] - mn[..., 1]
        radius = 0.5 * torch.sqrt(width**2 + height**2) + buff
        frame = Circle(
            scene=mobject.scene,
            radius=radius,
            stroke_color=color,
            stroke_width=stroke_width,
            location=center,
            filled=False,
        )
    else:
        raise ValueError("shape should be either Rectangle or Circle.")

    if fade_in and fade_out:
        with Seq(animation_manager=animation_manager):
            with Off(animation_manager=animation_manager):
                frame.spawn()
                frame.opacity = 0.0
            with Seq(duration=duration, animation_manager=animation_manager):
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.opacity = 1.0
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.opacity = 0.0
            with Off(animation_manager=animation_manager):
                frame.despawn(animate=False)
    elif fade_in:
        with Seq(animation_manager=animation_manager):
            with Off(animation_manager=animation_manager):
                frame.opacity = 0.0
                full_pts = frame.control_points.location.clone()
                frame.spawn()
            with Seq(duration=duration, animation_manager=animation_manager):
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.opacity = 1.0
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.animate_function(undraw_step, full_control_points=full_pts)
            with Off(animation_manager=animation_manager):
                frame.despawn(animate=False)
    elif fade_out:
        with Seq(animation_manager=animation_manager):
            with Off(animation_manager=animation_manager):
                frame.opacity = 1.0
                full_pts = frame.control_points.location.clone()
                frame._set_control_points_to_partial(full_pts, 0.0, 0.0)
                frame.spawn()
            with Seq(duration=duration, animation_manager=animation_manager):
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.animate_function(draw_step, full_control_points=full_pts)
                with Sync(duration=duration / 2, animation_manager=animation_manager):
                    frame.opacity = 0.0
            with Off(animation_manager=animation_manager):
                frame.despawn(animate=False)
    else:
        _show_passing_flash_on_bezier(
            frame,
            time_width=time_width,
            duration=duration,
            clone_source=False,
        )
    return mobject


def ApplyWave(
    mobject,
    direction=UP,
    amplitude: float = 0.2,
    ripples: int = 1,
    time_width: float = 1.0,
    duration: float = 2.0,
    wave_func=easings.smooth,
):
    """Ripple a wave across a Mob's geometry, left to right.

    The Mob's own points are displaced, so it flexes like a flag rather than moving as
    a rigid body, and it returns to its original shape at the end.

    Animation
    ---------
    Recorded as an animation of ``duration`` seconds, regardless of the enclosing
    context's duration.

    Parameters
    ----------
    mobject
        The Mob to ripple.
    direction
        Direction points are displaced in, shape ``(*, 3)``. Defaults to ``UP``. The
        wave always *travels* along x; this is which way the Mob bulges.
    amplitude
        Peak displacement, in world units. Defaults to ``0.2``.
    ripples
        How many oscillations pass through the Mob. Defaults to ``1``.
    time_width
        Width of the wave as a fraction of the Mob. Defaults to ``1.0``.
    duration
        Duration of the ripple, in seconds. Defaults to ``2.0``.
    wave_func
        Easing applied to the wave's shape. Defaults to ``easings.smooth``.

    Returns
    -------
    :class:`~.Mob`
        The Mob that was passed in.
    """
    direction = cast_to_tensor(direction)
    vect = amplitude * F.normalize(direction, p=2, dim=-1)
    bbox = mobject.get_bounding_box()
    x_min = bbox.amin(-2)[..., 0].min()
    x_max = bbox.amax(-2)[..., 0].max()

    def wave(t):
        t = 1.0 - t
        mask = (t >= 0.0) & (t <= 1.0)
        phases = ripples * 2
        phase = torch.floor(t * phases).long()

        val_0 = wave_func(t * phases)
        t_last = t - (phases - 1) / phases
        val_last = (1.0 - wave_func(t_last * phases)) * (2.0 * (ripples % 2) - 1.0)

        phase_rel = torch.floor((phase - 1.0).float() / 2.0)
        t_rel = t - (2.0 * phase_rel + 1.0) / phases
        val_mid = (1.0 - 2.0 * wave_func(t_rel * ripples)) * (
            1.0 - 2.0 * (phase_rel % 2)
        )

        res = torch.where(
            phase == 0,
            val_0,
            torch.where(phase == phases - 1, val_last, val_mid),
        )
        return torch.where(mask, res, torch.zeros_like(t))

    def wave_homotopy(points, t):
        x = points[..., 0]
        t = t.reshape(points.shape[0], -1)[:, :1]
        upper = (1.0 + time_width) * t
        lower = upper - time_width
        relative_x = (x - x_min) / (x_max - x_min + 1e-8)
        wave_phase = (relative_x - lower) / (time_width + 1e-8)
        w = wave(wave_phase)
        nudge = w.unsqueeze(-1) * vect.to(points)
        return points + nudge

    return Homotopy(mobject, wave_homotopy, duration=duration)
