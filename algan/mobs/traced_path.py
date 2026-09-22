"""Motion trails reconstructed from samples of the owning Scene's timeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import active_scene_for_new_mob
from algan.constants.color import WHITE, Color
from algan.errors import AlganConfigurationError
from algan.mobs.bezier_circuit import _stroke_width_in_render_pixels
from algan.mobs.nonplanar_circuit import NonPlanarPlan, camera_eye, run_planes
from algan.settings.renderer_settings import RENDERER_REGISTRY
from algan.utils.tensor_utils import cast_to_tensor

__all__ = ["TracedPath"]


def _positive_seconds(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlganConfigurationError(
            f"{name} must be positive finite seconds"
        ) from exc
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise AlganConfigurationError(f"{name} must be positive finite seconds")
    return result


class _PointSource:
    """Cloning a trail keeps its source, without cloning an external Mob."""

    def __init__(self, function):
        self.function = function

    def __deepcopy__(self, memo):
        return self

    def points(self, frames):
        try:
            points = cast_to_tensor(self.function())
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(
                "TracedPath's callable must return one 3-D point per sampled time"
            ) from exc
        if points.shape[-1:] != (3,) or points.numel() not in (3, frames * 3):
            raise AlganConfigurationError(
                "TracedPath's callable must return one 3-D point per sampled time; "
                "use shape (3,), (T, 3) or (T, 1, 3), not a pack of points"
            )
        if not bool(torch.isfinite(points).all()):
            raise AlganConfigurationError("TracedPath points must be finite")
        return points.reshape(-1, 3).expand(frames, 3).clone()


class TracedPath(Mob):
    """A path drawn by a moving point, with an optional finite trail lifetime.

    Pass a Mob to follow its bounding-box center, or a zero-argument callable
    such as ``lambda: dot.location`` to follow a particular world-space point.
    The trace consists of straight segments with round ends and joins, and
    retains depth for motion in three dimensions. Sampling depends on timeline
    time, so seeking backwards, rendering isolated frames and changing batch
    size produce the same trail.

    Animation
    ---------
    Construction is immediate. Spawn the source and the trail before animating
    the source; tracing starts at the trail's spawn time. ``spawn(animate=False)``
    starts it without spending a second on its entrance. Recorded animations and
    source updaters both contribute to the path. Color, opacity, stroke width
    and transforms animate normally, over 1 second by default. Transforming the
    trail changes the displayed path without moving its source. Sampling options
    are fixed at construction. A cloned trail follows the same source.

    Parameters
    ----------
    traced_point_func
        Mob or zero-argument callable returning one world-space point. A callable
        must be deterministic and must not modify the scene. Algan batches its
        state reads: return shape ``(T, 1, 3)`` or ``(T, 3)`` for T sampled times;
        a constant point of shape ``(3,)`` also works. Source Mobs must belong to
        the same Scene. Reading another TracedPath from the callable is unsupported.
    stroke_width
        Stroke width in Algan's reference pixels, matching Line. Defaults to 2.
    stroke_color
        Initial stroke color. Defaults to WHITE. The ``color`` Mob keyword can
        also select it, and ``trail.color`` animates it afterwards.
    dissipating_time
        Positive trail lifetime in seconds. Defaults to None, keeping all motion
        since spawn. With a finite lifetime, older portions disappear even while
        the source is stationary.
    sample_interval
        Positive spacing of samples in seconds, independent of video frame rate.
        Defaults to 1/60. Smaller values describe fast curves more accurately and
        use more geometry. Current endpoints are sampled at their exact times.
    **kwargs
        Passed to :class:`~.Mob`, including ``scene``, ``color`` and ``opacity``.
        The Scene defaults to a passed Mob's Scene, a bound method's Mob's Scene,
        or the active Scene for other callables. The initial location is the
        source's current point; transforms use that point as the trail's anchor.

    Attributes
    ----------
    stroke_width
        Animatable stroke width in Algan's reference pixels.

    Raises
    ------
    :class:`.AlganConfigurationError`
        If the source, point shape, timing options or Scene ownership is invalid.

    Examples
    --------
    Retain an orbit and show a shorter trail alongside it:

    .. algan:: Example1TracedPath

        from algan import *

        dot = Dot(RIGHT * 2, color=YELLOW).spawn(animate=False)
        TracedPath(dot.get_center, stroke_color=BLUE).spawn(animate=False)
        TracedPath(dot, stroke_color=YELLOW, stroke_width=4,
                   dissipating_time=0.6).spawn(animate=False)
        with Seq(runtime=4):
            dot.orbit(300, OUT, about=ORIGIN)
        Scene.wait(1)
        Scene.save_video()
    """

    def __init__(
        self,
        traced_point_func: Mob | Callable[[], torch.Tensor],
        stroke_width: float = 2,
        stroke_color: Color | torch.Tensor | tuple | list = WHITE,
        dissipating_time: float | None = None,
        *,
        sample_interval: float = 1 / 60,
        **kwargs: Any,
    ) -> None:
        sample_interval = _positive_seconds(sample_interval, "sample_interval")
        if dissipating_time is not None:
            dissipating_time = _positive_seconds(dissipating_time, "dissipating_time")
        source_mob = (
            traced_point_func
            if isinstance(traced_point_func, Mob)
            else getattr(traced_point_func, "__self__", None)
        )
        scene = kwargs.get("scene")
        if scene is None:
            scene = (
                source_mob.scene
                if isinstance(source_mob, Mob)
                else active_scene_for_new_mob()
            )
        kwargs["scene"] = scene
        if isinstance(source_mob, Mob) and source_mob.scene is not scene:
            raise AlganConfigurationError("TracedPath source must belong to its Scene")
        if isinstance(traced_point_func, Mob):
            traced_point_func = traced_point_func.get_center
        if not callable(traced_point_func):
            raise AlganConfigurationError("TracedPath needs a Mob or a point callable")
        self._point_source = _PointSource(traced_point_func)
        self._trace_anchor = self._point_source.points(1).reshape(1, 1, 3)
        self._trace_points = None
        self._sample_interval = sample_interval
        self._dissipating_time = dissipating_time
        kwargs.setdefault("location", self._trace_anchor)
        kwargs.setdefault("color", stroke_color)
        super().__init__(**kwargs)
        self.register_attrs_as_animatable(["stroke_width"], TracedPath)
        self.stroke_width = cast_to_tensor(stroke_width)
        self.is_primitive = True
        self.scene.timeline_manager._traced_paths[self.id] = self

    def __deepcopy__(self, memo):
        clone = super().__deepcopy__(memo)
        if clone.id != self.id:
            clone._trace_points = None
            clone.scene.timeline_manager._traced_paths[clone.id] = clone
        return clone

    def _after_repack(self):
        raise AlganConfigurationError(
            "TracedPath objects cannot be packed; use a Group"
        )

    def _sample_times(self, times):
        start, end = self.lifespan.start(), self.lifespan.end()
        if start < 0 or not bool(
            ((times >= start) & ((times < end) | (end < 0))).any()
        ):
            return None
        # Float64 host arithmetic keeps the sample grid independent of which
        # frame window was requested, and is also supported on MPS hosts.
        upper = times.detach().to(device="cpu", dtype=torch.float64).clamp(min=start)
        if end >= 0:
            upper = upper.clamp(max=end)
        lower = torch.full_like(upper, start)
        interval = self._sample_interval
        if self._dissipating_time is None:
            count = max(1, math.ceil((float(upper.max()) - start) / interval))
            first = torch.zeros_like(upper)
        else:
            lower = (upper - self._dissipating_time).clamp(min=start)
            first = torch.floor((lower - start) / interval)
            count = max(1, math.ceil(self._dissipating_time / interval) + 1)
        grid = (
            start + (first[:, None] + torch.arange(count + 1, device="cpu")) * interval
        )
        return grid.clamp(min=lower[:, None], max=upper[:, None]).to(times)

    def _display_points(self):
        points = (
            self._trace_anchor if self._trace_points is None else self._trace_points
        )
        relative = points - self._trace_anchor.to(points)
        basis = self.basis.reshape(-1, 3, 3).to(points)
        return relative @ basis + self.location.to(points)

    def get_boundary_points(self) -> torch.Tensor:
        """Get the sampled points of the currently displayed trail.

        Animation
        ---------
        Immediate and not recorded. During rendering, return the trail at the
        requested times. Outside rendering, return its transformed initial
        point; the full path is reconstructed only when frames are requested.

        Returns
        -------
        torch.Tensor
            World-space points with shape ``(T, N, 3)`` for T requested times
            and N samples. Repeated points can pad shorter trails in a batch.
        """
        if self.scene.timeline_manager._sampling_traced_paths:
            raise AlganConfigurationError(
                "A TracedPath point callable cannot read another TracedPath"
            )
        return self._display_points()

    def get_render_primitives(self):
        """Internal: build camera-facing stroke segments for the sampled path."""
        if self._trace_points is None:
            return None
        points = self._display_points()
        a, b = points[:, :-1], points[:, 1:]
        live = (b - a).norm(dim=-1, keepdim=True) > 1e-8
        keep = live.any(dim=0).squeeze(-1)
        if not bool(keep.any()):
            return None
        a, b, live = a[:, keep], b[:, keep], live[:, keep]
        x = torch.stack((a, torch.lerp(a, b, 1 / 3), torch.lerp(a, b, 2 / 3), b), dim=2)
        frames, segments = a.shape[:2]
        plan = NonPlanarPlan(
            "stroke",
            torch.arange(segments, device=x.device),
            torch.ones(segments, device=x.device, dtype=torch.long),
        )
        centre, first, second, normal = run_planes(x, plan, camera_eye(self))
        # A segment outside this frame's trail has coincident endpoints. Keep
        # its invisible plane nonsingular while sharing topology across frames.
        first = torch.where(live, first, x.new_tensor([1, 0, 0]))
        second = torch.where(live, second, x.new_tensor([0, 1, 0]))
        normal = torch.where(live, normal, x.new_tensor([0, 0, 1]))

        def per_segment(value):
            return (
                value.to(x).reshape(-1, 1, value.shape[-1]).expand(frames, segments, -1)
            )

        color = per_segment(self.get_animated_attribute("color")).unsqueeze(-2)
        grid = torch.ones((1, segments, 1), dtype=torch.int32, device=x.device)
        width = _stroke_width_in_render_pixels(
            self.stroke_width,
            getattr(self.scene, "_geometry_view", self.scene).video_settings,
        )
        primitive = RENDERER_REGISTRY.bezier_circuit_primitive(
            corners=x,
            next_segment_inds=torch.zeros(
                (1, segments, 1, 1), dtype=torch.long, device=x.device
            ),
            num_segments_per_circuit=plan.run_counts,
            colors=color,
            stroke_color=color,
            opacity=per_segment(self.opacity) * live,
            normals=normal,
            stroke_width=per_segment(width),
            mob_center=centre,
            grid_width=grid,
            grid_height=grid,
            first_basis=first,
            second_basis=second,
            glow=per_segment(self.glow),
            filled=False,
        )
        primitive.declare_shadow_flags(*self._resolved_shadow_flags())
        return primitive

    def _get_memory_used_per_timestep(self):
        duration = max(
            0,
            self.scene._recorded_end_time_for_render() - max(0, self.lifespan.start()),
        )
        if self._dissipating_time is not None:
            duration = min(duration, self._dissipating_time)
        # Sampling indices, points, cubics, run-plane scratch, colors and the
        # primitive output coexist during preparation. Charge the longest
        # authored trail, not the short geometry in an early render window.
        return (math.ceil(duration / self._sample_interval) + 2) * 768
