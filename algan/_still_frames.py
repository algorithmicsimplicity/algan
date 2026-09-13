"""Immutable-at-capture still requests and bounded sparse-frame output.

Project owns deferred batches; no queue survives a project run on a Scene or
in global settings. Keeping this separate also avoids the Scene/output import
cycle (RenderResult is imported only when a request is rendered).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import torch

from algan.errors import AlganConfigurationError
from algan.logging.logger import get_logger
from algan.settings import SETTINGS
from algan.settings.video_settings import VideoSettings

logger = get_logger()


def _background_key(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return type(value), value
    # An in-place edit changes the key; equal but independent tensors need not
    # be compared (which would synchronize a render device).
    if torch.is_tensor(value) and value.is_inference():
        # Inference tensors have no version counter. Keep separate captures
        # separate rather than coalescing potentially different snapshots.
        return object()
    return id(value), getattr(value, "_version", None)


def _snapshot(value):
    return value.detach().clone() if torch.is_tensor(value) else value


@dataclass(frozen=True)
class _StillTarget:
    path: Path
    timestamp: float
    given: object = None
    anchor: object = None

    def resolved_time(self):
        return self.timestamp + (self.anchor() if self.anchor is not None else 0.0)


@dataclass
class _StillBatch:
    scene: object
    targets: list[_StillTarget]
    video_settings: VideoSettings
    background_state: tuple
    background_override: object
    background_key: tuple
    post_processes: tuple | None
    overwrite: bool
    raytracing: dict
    environment: tuple

    @classmethod
    def capture(
        cls,
        scene,
        targets,
        video_settings,
        background,
        post_processes,
        overwrite,
        *,
        deferred=False,
    ):
        from algan.render_loop import _check_post_processes

        settings = scene._resolve_video_settings(video_settings)
        if not isinstance(settings, VideoSettings):
            raise AlganConfigurationError(
                "video_settings must be a VideoSettings instance or preset "
                f"(for example HD or PREVIEW), got {type(settings).__name__}"
            )
        settings = settings.as_preset()
        if post_processes is not None and not callable(post_processes):
            post_processes = tuple(post_processes)
        _check_post_processes(post_processes)
        context = scene.animation_manager.context
        cursor = context.timespan.current_time
        captured = []
        for path, given in targets:
            value = 1.5 / settings.frames_per_second if given is None else float(given)
            relative = given is None or value < 0
            resolved = cursor + value if relative else value
            if not math.isfinite(resolved) or resolved < 0:
                # Use Scene's diagnostic, including the original negative input.
                scene._frame_index_for_timestamp(resolved, given)
            anchor = None
            if relative and deferred and context.prev_context is not None:
                # Snapshot the cursor, not the mutable current_time field. Its
                # event follows a Speech/Seq context's eventual rescaling;
                # offsets stay seconds (and the default stays 1.5 frames).
                anchor = context.timespan.get_current_time()
            captured.append(
                _StillTarget(path, value if anchor else resolved, given, anchor)
            )
        background_state = (
            _snapshot(scene.background_frame),
            scene.background,
            scene.background_is_set,
        )
        return cls(
            scene,
            captured,
            settings,
            background_state,
            _snapshot(background),
            (_background_key(background), _background_key(scene.background_frame)),
            post_processes,
            overwrite,
            SETTINGS.raytracing.to_dict(),
            (
                scene.environment_map,
                scene.environment_intensity,
                scene.environment_ambient,
                scene.premultiplied_over,
            ),
        )

    def compatible_with(self, other):
        """Only coalesce adjacent calls with identical render-affecting options."""
        return (
            self.scene is other.scene
            and self.video_settings == other.video_settings
            and self.background_key == other.background_key
            and self.post_processes == other.post_processes
            and self.overwrite == other.overwrite
            and self.raytracing == other.raytracing
            and self.environment[0] is other.environment[0]
            and self.environment[1:] == other.environment[1:]
        )

    def deferred_results(self):
        from algan.utils.algan_utils import RenderResult

        return [RenderResult("deferred", target.path) for target in self.targets]

    def render(self):
        scene = self.scene
        previous_settings = scene.video_settings
        previous_explicit = scene._video_settings_explicit
        previous_background = (
            scene.background_frame,
            scene.background,
            scene.background_is_set,
        )
        previous_raytracing = SETTINGS.raytracing.to_dict()
        previous_environment = (
            scene.environment_map,
            scene.environment_intensity,
            scene.environment_ambient,
            scene.premultiplied_over,
        )
        try:
            scene.set_video_settings(self.video_settings)
            SETTINGS.raytracing._restore(self.raytracing)
            (
                scene.environment_map,
                scene.environment_intensity,
                scene.environment_ambient,
                scene.premultiplied_over,
            ) = self.environment
            (scene.background_frame, scene.background, scene.background_is_set) = (
                self.background_state
            )
            if self.background_override is not None:
                scene.set_background(self.background_override)
            with scene.timeline_manager.preserving_authoring_state(
                preserve_replay_resolution=scene.animation_manager.context.prev_context
                is not None
            ):
                return self._write_frames()
        finally:
            scene.set_video_settings(previous_settings, _explicit=previous_explicit)
            SETTINGS.raytracing._restore(previous_raytracing)
            (
                scene.environment_map,
                scene.environment_intensity,
                scene.environment_ambient,
                scene.premultiplied_over,
            ) = previous_environment
            (scene.background_frame, scene.background, scene.background_is_set) = (
                previous_background
            )

    def _write_frames(self):
        from PIL import Image

        from algan.utils.algan_utils import RenderResult

        results = [None] * len(self.targets)
        by_frame = defaultdict(list)
        reserved = set()
        for position, target in enumerate(self.targets):
            if not self.overwrite and (target.path.exists() or target.path in reserved):
                results[position] = RenderResult("skipped", target.path)
                continue
            reserved.add(target.path)
            index = self.scene._frame_index_for_timestamp(
                target.resolved_time(), target.given
            )
            by_frame[index].append(position)
        indices = sorted(by_frame)
        if not indices:
            return results

        extra = (
            {}
            if self.post_processes is None
            else {"post_processes": self.post_processes}
        )
        # Keep the one-frame route's established get_frames contract. Multiple
        # frames share one job, even when their timeline indices are far apart.
        if len(indices) == 1:
            frames = self.scene.get_frames(indices[0], indices[0] + 1, **extra)
        else:
            frames = self.scene.get_frames(
                0, len(indices), frame_indices=indices, **extra
            )
        started = time.perf_counter()
        emitted = 0
        rendered = []
        last_write = {}
        with torch.no_grad(), closing(frames):
            for batch in frames:
                for frame in batch:
                    if emitted >= len(indices):
                        raise RuntimeError("More frames were produced than requested")
                    positions = by_frame[indices[emitted]]
                    image = Image.fromarray(frame.contiguous().numpy())
                    for position in positions:
                        target = self.targets[position]
                        # Sorting times must not reverse last-write-wins when
                        # different requests intentionally share a destination.
                        if position > last_write.get(target.path, -1):
                            image.save(str(target.path))
                            last_write[target.path] = position
                        rendered.append(
                            (position, getattr(self.scene, "last_render_plan", None))
                        )
                    emitted += 1
        if emitted != len(indices):
            raise RuntimeError("Not all requested still frames were produced")
        # A shared render has no independent per-file duration. Attribute its
        # wall time evenly, so summing result times still gives the job's cost.
        walltime = (time.perf_counter() - started) / len(rendered)
        for position, plan in rendered:
            target = self.targets[position]
            results[position] = RenderResult("rendered", target.path, walltime, plan)
            logger.info("Finished rendering %s in %.1f s", target.path, walltime)
        return results
