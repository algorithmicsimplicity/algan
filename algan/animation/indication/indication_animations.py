from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animation.animatable import animated_function
from algan.animation.animation_contexts import Off, Seq, Sync
from algan.animation.movement import Homotopy
from algan.constants import rate_funcs
from algan.constants.color import GRAY, YELLOW
from algan.constants.math import RADIANS_TO_DEGREES
from algan.constants.spatial import OUT, UP
from algan.geometry.geometry import get_rotation_around_axis
from algan.utils.tensor_utils import cast_to_tensor, squish, unsquish


def there_and_back(t, inflection=10.0):
    t = cast_to_tensor(t)
    new_t = torch.where(t < 0.5, 2.0 * t, 2.0 * (1.0 - t))
    return rate_funcs.smooth(new_t, inflection)


def wiggle(t, wiggles=2):
    t = cast_to_tensor(t)
    val = torch.sin(wiggles * math.pi * t)
    return there_and_back(t) * val


def Indicate(mobject, scale_factor=1.2, color=YELLOW, run_time=1.0):
    color = cast_to_tensor(color)
    scale_factor = cast_to_tensor(scale_factor)
    current_scale = mobject.scale_coefficient
    with Sync(run_time=run_time):
        mobject.pulse_color(color)
        mobject.apply_absolute_change_two(
            "scale_coefficient", current_scale * scale_factor, current_scale
        )
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
    t = cast_to_tensor(t)
    s_val = 1.0 + (scale_value - 1.0) * there_and_back(t)
    rot_deg = wiggle(t, n_wiggles) * rotation_angle * RADIANS_TO_DEGREES

    R = get_rotation_around_axis(rot_deg, OUT, dim=-1)
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
    scale_value=1.1,
    rotation_angle=0.01 * math.pi * 2,
    n_wiggles=6,
    scale_about_point=None,
    rotate_about_point=None,
    run_time=2.0,
):
    basis_0 = mobject.basis.clone()
    location_0 = mobject.location.clone()
    with Sync(run_time=run_time):
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


def Blink(mobject, time_on=0.5, time_off=0.5, blinks=1, hide_at_end=False):
    with Seq():
        for _ in range(blinks):
            with Off():
                mobject.set_opacity_via_color(1.0)
            mobject.wait(time_on)
            with Off():
                mobject.set_opacity_via_color(0.0)
            mobject.wait(time_off)
        if not hide_at_end:
            with Off():
                mobject.set_opacity_via_color(1.0)
            mobject.wait(time_on)
    return mobject


def FocusOn(focus_point, opacity=0.2, color=GRAY, run_time=2.0):
    from algan.mobs.mob import Mob
    from algan.mobs.shapes_2d import Circle

    if isinstance(focus_point, Mob):
        focus_point = focus_point.get_center()
    else:
        focus_point = cast_to_tensor(focus_point)
    with Seq():
        with Off():
            spotlight = Circle(
                radius=10.0, color=color, opacity=0.0, location=focus_point
            ).spawn()
        with Sync(run_time=run_time):
            spotlight.scale(1e-4)
            spotlight.opacity = opacity
        with Off():
            spotlight.despawn(animate=False)
    return spotlight


@animated_function(
    animated_args={"t": 0.0}, unique_args=["time_width", "full_control_points"]
)
def passing_flash_step(mob, t, time_width, full_control_points):
    t = cast_to_tensor(t)
    upper = t * (1.0 + time_width)
    lower = upper - time_width
    upper = torch.clamp(upper, 0.0, 1.0)
    lower = torch.clamp(lower, 0.0, 1.0)
    mob.set_control_points_to_partial(full_control_points, lower, upper)


@animated_function(animated_args={"t": 0.0}, unique_args=["full_control_points"])
def draw_step(mob, t, full_control_points):
    t = cast_to_tensor(t)
    mob.set_control_points_to_partial(full_control_points, 0.0, t)


@animated_function(animated_args={"t": 0.0}, unique_args=["full_control_points"])
def undraw_step(mob, t, full_control_points):
    t = cast_to_tensor(t)
    mob.set_control_points_to_partial(full_control_points, 0.0, 1.0 - t)


def ShowPassingFlash(mobject, time_width=0.1, run_time=1.0):
    from algan.mobs.bezier_circuit import BezierCircuitCubic

    if isinstance(mobject, BezierCircuitCubic):
        with Seq():
            with Off():
                full_pts = mobject.control_points.location.clone()
                mobject.set_control_points_to_partial(full_pts, 0.0, 0.0)
                mobject.spawn()
            with Sync(run_time=run_time):
                mobject.animate_function(
                    passing_flash_step,
                    time_width=time_width,
                    full_control_points=full_pts,
                )
            with Off():
                mobject.despawn(animate=False)
    else:
        beziers = [
            d for d in mobject.get_descendants() if isinstance(d, BezierCircuitCubic)
        ]
        with Sync(run_time=run_time):
            for b in beziers:
                ShowPassingFlash(b, time_width=time_width, run_time=run_time)
    return mobject


def ShowPassingFlashWithThinningStrokeWidth(
    vmobject, n_segments=10, time_width=0.1, run_time=1.0
):
    max_stroke_width = getattr(vmobject, "border_width", 5.0)
    if isinstance(max_stroke_width, torch.Tensor):
        max_stroke_width = max_stroke_width.item()
    clones = []
    with Off():
        for i in range(n_segments):
            factor = i / (n_segments - 1) if n_segments > 1 else 1.0
            stroke_w = factor * max_stroke_width
            time_w = (1.0 - factor) * time_width
            clone = vmobject.clone(spawn=False)
            clone.border_width = stroke_w
            clones.append((clone, time_w))
    with Sync(run_time=run_time):
        for clone, time_w in clones:
            ShowPassingFlash(clone, time_width=time_w, run_time=run_time)
    return vmobject


def Flash(
    point_or_mobject,
    line_length=0.2,
    num_lines=12,
    flash_radius=0.1,
    line_stroke_width=3,
    color=YELLOW,
    time_width=1.0,
    run_time=1.0,
):
    from algan.mobs.mob import Mob
    from algan.mobs.shapes_2d import Line

    if isinstance(point_or_mobject, Mob):
        center = point_or_mobject.get_center()
    else:
        center = cast_to_tensor(point_or_mobject)
    lines = []
    with Off():
        for i in range(num_lines):
            angle = i * (2 * math.pi / num_lines)
            direction = torch.tensor(
                [math.cos(angle), math.sin(angle), 0.0], device=center.device
            )
            start = center + flash_radius * direction
            end = start + line_length * direction
            line = Line(start, end, border_color=color, border_width=line_stroke_width)
            lines.append(line)
    with Sync(run_time=run_time):
        for line in lines:
            ShowPassingFlash(line, time_width=time_width, run_time=run_time)
    return point_or_mobject


def Circumscribe(
    mobject,
    shape=None,
    fade_in=False,
    fade_out=False,
    time_width=0.3,
    buff=0.2,
    color=YELLOW,
    run_time=1.0,
    stroke_width=3,
):
    from algan.mobs.shapes_2d import Circle, Rectangle, Square, SurroundingRectangle

    if shape is None or shape in (Rectangle, Square):
        frame = SurroundingRectangle(
            mobject,
            color=color,
            buffer=buff,
            border_width=stroke_width,
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
            radius=radius,
            border_color=color,
            border_width=stroke_width,
            location=center,
            filled=False,
        )
    else:
        raise ValueError("shape should be either Rectangle or Circle.")

    if fade_in and fade_out:
        with Seq():
            with Off():
                frame.spawn()
                frame.opacity = 0.0
            with Seq(run_time=run_time):
                with Sync(run_time=run_time / 2):
                    frame.opacity = 1.0
                with Sync(run_time=run_time / 2):
                    frame.opacity = 0.0
            with Off():
                frame.despawn(animate=False)
    elif fade_in:
        with Seq():
            with Off():
                frame.opacity = 0.0
                frame.portion_of_curve_drawn = 1.0
                full_pts = frame.control_points.location.clone()
                frame.spawn()
            with Seq(run_time=run_time):
                with Sync(run_time=run_time / 2):
                    frame.opacity = 1.0
                with Sync(run_time=run_time / 2):
                    frame.animate_function(undraw_step, full_control_points=full_pts)
            with Off():
                frame.despawn(animate=False)
    elif fade_out:
        with Seq():
            with Off():
                frame.opacity = 1.0
                full_pts = frame.control_points.location.clone()
                frame.set_control_points_to_partial(full_pts, 0.0, 0.0)
                frame.spawn()
            with Seq(run_time=run_time):
                with Sync(run_time=run_time / 2):
                    frame.animate_function(draw_step, full_control_points=full_pts)
                with Sync(run_time=run_time / 2):
                    frame.opacity = 0.0
            with Off():
                frame.despawn(animate=False)
    else:
        ShowPassingFlash(frame, time_width=time_width, run_time=run_time)
    return mobject


def ApplyWave(
    mobject,
    direction=UP,
    amplitude=0.2,
    ripples=1,
    time_width=1.0,
    run_time=2.0,
    wave_func=rate_funcs.smooth,
):
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
        upper = (1.0 + time_width) * t
        lower = upper - time_width
        relative_x = (x - x_min) / (x_max - x_min + 1e-8)
        wave_phase = (relative_x - lower) / (time_width + 1e-8)
        w = wave(wave_phase)
        nudge = w.unsqueeze(-1) * vect.unsqueeze(0).unsqueeze(0)
        return points + nudge

    return Homotopy(mobject, wave_homotopy, run_time=run_time)
