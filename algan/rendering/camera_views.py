"""Bounded, sequential render-to-texture passes on one recorded Scene.

Each pass owns its projection caches. Finish each pass (including its prep
work) before the next materializes the shared timeline. Only a bounded window
of host textures survives into the main camera's pass.

What a pass would otherwise rebuild for every window -- the render arena, the
measured chunk-peak and peak-ratio models, the batch-prep worker -- belongs to
the job instead (:class:`_JobPassResources`): one arena and one worker for all
of its passes, and one set of models per pass (each capture's, and the main
camera's), so only the first window probes.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor

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


class _PassModels:
    """The batch-sizing models one pass carries from window to window."""

    def __init__(self, chunk, merge_peak_ratio, project_peak_ratio):
        self.chunk = chunk
        self.merge_peak_ratio = merge_peak_ratio
        self.project_peak_ratio = project_peak_ratio


class _JobPassResources:
    """What every window of one camera-view render job shares.

    The passes run strictly one after another, and each finishes its batch
    preparation before returning, so one arena and one prep worker serve them
    all; the arena is reset at the start of each pass. The chunk-peak and
    peak-ratio models are measurements of one pass's geometry at one
    resolution, so they are kept per pass, keyed by the capturing view (or
    ``None`` for the main camera). Before this a render with a live view made
    each of these afresh for every pass of every window, so every window
    probed with single-frame chunks, faulted a new arena in and could not
    prefetch past its own end.

    Used from the render loop (``RenderLoopMixin._get_frames_impl``), which
    finds it on the pass as ``_camera_view_resources``; :meth:`close` releases
    it when the job's last window is done.
    """

    def __init__(self):
        self.memory = None
        self._executor = None
        self._models = {}

    def arena(self, factory):
        if self.memory is None:
            self.memory = factory()
        else:
            self.memory.reset()
        return self.memory

    def models(self, key, factory):
        if key not in self._models:
            self._models[key] = factory()
        return self._models[key]

    def executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="algan-batch-prep"
            )
        return self._executor

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.memory is not None:
            self.memory.data = None
            self.memory = None
        self._models.clear()


def _render_pass(scene, camera, actors, resolution, resources=None, key=None):
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
    if resources is not None:
        result._camera_view_resources = resources
        result._camera_view_pass_key = key
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
    aux_passes=False,
    aux_sink=None,
    independent_frames=False,
):
    # Aux passes belong to the main camera's frames only: the capture passes
    # below are textures, and never trace them.
    aux_kwargs = {"aux_passes": True, "aux_sink": aux_sink} if aux_passes else {}
    # Stills keep one frame per batch in every pass (see RenderLoopMixin.get_frames).
    independent = {"_independent_frames": True} if independent_frames else {}
    excluded_displays = {
        id(mob)
        for actor in scene.actors
        if getattr(actor, "_is_camera_view", False)
        for mob in actor.get_descendants()
    }
    step = _window_size(views)
    resources = _JobPassResources()
    try:
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
                    resources,
                    id(view),
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
                    SETTINGS.raytracing.experimental.override(
                        post_process_tonemap=True
                    ),
                    contextlib.closing(
                        capture.get_frames(
                            0,
                            len(indices),
                            background=background,
                            post_processes=(),
                            manual_memory=manual_memory,
                            frame_indices=indices,
                            **independent,
                        )
                    ) as stream,
                ):
                    textures = [_texture_from_capture(batch) for batch in stream]
                captures[id(view)] = (indices, torch.cat(textures, dim=0))
                del textures, capture
            main = _render_pass(
                scene,
                scene.camera,
                scene.actors,
                scene.video_settings.resolution,
                resources,
                None,
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
                    **aux_kwargs,
                    **independent,
                )
            ) as stream:
                for frames in stream:
                    if hasattr(main, "last_render_plan"):
                        scene.last_render_plan = main.last_render_plan
                    yield frames
            del main, captures

    finally:
        resources.close()


def _view_primitive(view, capture, time_indices):
    indices, texture = capture
    positions = {index: position for position, index in enumerate(indices)}
    selection = torch.tensor(
        [positions[int(index)] for index in time_indices],
        dtype=torch.long,
        device=texture.device,
    )
    return view._get_camera_view_primitives(texture.index_select(0, selection))
