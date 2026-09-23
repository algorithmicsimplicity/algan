"""Bounded, sequential render-to-texture passes on one recorded Scene.

Each pass owns its arena and projection caches. Finish each pass (including its
prep worker) before the next materializes the shared timeline. Only a bounded
window of host textures survives into the main camera's pass.
"""

from __future__ import annotations

import contextlib

import torch

from algan.settings import SETTINGS
from algan.utils.color_space import linear_to_srgb
from algan.utils.memory_utils import get_num_available_bytes


def _live_views(scene):
    if getattr(scene, "_camera_view_pass", False):
        return []
    return [
        actor
        for actor in getattr(scene, "actors", ())
        if getattr(actor, "_is_camera_view", False) and actor.lifespan.start() >= 0
    ]


def _render_pass(scene, camera, actors, resolution):
    # Scene returns itself from copy/deepcopy. This is instead a private,
    # non-owning render context: no constructor, SceneManager push, timeline
    # registration, Mob cloning or authored camera replacement.
    result = object.__new__(type(scene))
    result.__dict__.update(scene.__dict__)
    result.camera = camera
    result.actors = actors
    result.memory = None
    result._actor_window_cache = None
    result.__dict__.pop("_bezier_geometry_cache", None)
    result._camera_view_pass = True
    result._camera_view_captures = {}
    result.set_video_settings(scene.video_settings.set(resolution=resolution))
    return result


@contextlib.contextmanager
def _geometry_view(scene, render_pass):
    # Camera-facing 3-D strokes need the capture eye, while recorded updaters
    # and screen-layout operations must still refer to the authored main camera.
    previous = getattr(scene, "_geometry_view", None)
    scene._geometry_view = render_pass
    try:
        yield
    finally:
        if previous is None:
            del scene._geometry_view
        else:
            scene._geometry_view = previous


def _texture_from_capture(frames):
    # Captures arrive top-down in working-space RGB, glow, and optional alpha
    # (alpha is still in byte units). Encode just RGB for the ordinary image
    # ingestion boundary, which decodes it before shading. Exposure, tone mapping
    # and quantization run only on the final composed frame.
    texture = torch.empty((*frames.shape[:-1], 5), dtype=torch.float32, device="cpu")
    rgb = frames[..., :3].float()
    glow = frames[..., 3].float()
    alpha = frames[..., 4] / 255 if frames.shape[-1] == 5 else torch.ones_like(glow)
    if frames.shape[-1] == 5:
        # The render buffer is premultiplied, but image textures carry straight
        # color. Coverage is applied by the texture sampler during composition.
        safe_alpha = alpha.clamp_min(1e-6)
        rgb = rgb / safe_alpha.unsqueeze(-1)
        glow = glow / safe_alpha
    texture[..., :3] = (
        linear_to_srgb(rgb) if SETTINGS.raytracing.linear_color_space else rgb
    )
    texture[..., 3] = glow
    texture[..., 4] = alpha
    return texture.transpose(-3, -2).flip(-2).contiguous()


def _window_size(views):
    bytes_per_frame = sum(
        w * h * 5 * 4 for w, h in (view.capture_resolution for view in views)
    )
    # Leave room for readback, conversion, packing and prefetched geometry.
    # Normal per-pass preflight and OOM retries still size the render arena.
    budget = min(
        256 * 1024 * 1024, int(get_num_available_bytes(torch.device("cpu")) * 0.02)
    )
    return max(
        1,
        min(
            SETTINGS.computing.max_animation_batch_size,
            budget // max(1, bytes_per_frame * 4),
        ),
    )


def _render_with_camera_views(
    scene,
    views,
    start,
    end,
    *,
    background,
    post_processes,
    manual_memory,
    frame_indices=None,
):
    excluded_displays = {
        id(mob)
        for actor in scene.actors
        if getattr(actor, "_is_camera_view", False)
        for mob in actor.get_descendants()
    }
    step = _window_size(views)
    for offset in range(start, end, step):
        stop = min(offset + step, end)
        indices = (
            tuple(frame_indices[offset:stop])
            if frame_indices is not None
            else tuple(range(offset, stop))
        )
        captures = {}
        for view in views:
            spawn, despawn = view.lifespan.start(), view.lifespan.end()
            times = [index / scene.frames_per_second for index in indices]
            if not any(t >= spawn and (despawn < 0 or t <= despawn) for t in times):
                continue
            excluded = excluded_displays | {
                id(mob)
                for root in view.capture_exclude
                for mob in root.get_descendants()
            }
            capture = _render_pass(
                scene,
                view.camera,
                [actor for actor in scene.actors if id(actor) not in excluded],
                view.capture_resolution,
            )
            capture._linear_output = True
            capture.light_sources = [
                light for light in scene.light_sources if id(light) not in excluded
            ]
            capture.premultiplied_over = False
            # Even the legacy in-composite tone mapping mode needs an HDR
            # intermediate here. Restore the setting before the main pass.
            with (
                _geometry_view(scene, capture),
                SETTINGS.raytracing.experimental.override(post_process_tonemap=True),
                contextlib.closing(
                    capture.get_frames(
                        0,
                        len(indices),
                        background=background,
                        post_processes=(),
                        manual_memory=manual_memory,
                        frame_indices=indices,
                    )
                ) as stream,
            ):
                textures = [_texture_from_capture(batch) for batch in stream]
            captures[id(view)] = (indices, torch.cat(textures, dim=0))
            del textures, capture
        main = _render_pass(
            scene, scene.camera, scene.actors, scene.video_settings.resolution
        )
        main._camera_view_captures = captures
        with contextlib.closing(
            main.get_frames(
                0,
                len(indices),
                background=background,
                post_processes=post_processes,
                manual_memory=manual_memory,
                frame_indices=indices,
            )
        ) as stream:
            for frames in stream:
                if hasattr(main, "last_render_plan"):
                    scene.last_render_plan = main.last_render_plan
                yield frames
        del main, captures


def _view_primitive(view, capture, time_indices):
    indices, texture = capture
    positions = {index: position for position, index in enumerate(indices)}
    selection = torch.tensor(
        [positions[int(index)] for index in time_indices],
        dtype=torch.long,
        device=texture.device,
    )
    return view._get_camera_view_primitives(texture.index_select(0, selection))
