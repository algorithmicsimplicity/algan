import inspect
import math
import time

from algan.settings import SETTINGS

import torch.nn.functional as F

from algan.errors import AlganConfigurationError

from algan.constants.spatial import *

from algan.animation_timeline.animation_contexts import (
    Seq,
    Sync,
    AnimationManager,
    animation_manager_context,
)
from algan.animation_timeline.timeline import TimelineManager
from algan.sound.audio_effect import AudioManager

# EmptySceneWarning and write_frames_from_queue moved to render_loop.py;
# re-exported here for backwards compatibility.
from algan.render_loop import (  # noqa: F401
    EmptySceneWarning,
    RenderLoopMixin,
    write_frames_from_queue,
)
from algan.utils.file_utils import get_image
from algan.scene_manager import SceneManager

from functools import wraps


class active_scene_method:
    """Bind to an instance, or resolve the active Scene when called on Scene.

    ``scene.save_video(...)`` renders ``scene``; ``Scene.save_video(...)``
    renders whichever Scene is currently active.  Class-level access reports
    the signature with ``self`` removed so ``help()``, IDE tooltips and Sphinx
    autodoc show the real parameters instead of ``(*args, **kwargs)``.
    """

    def __init__(self, function):
        self.function = function
        wraps(function)(self)
        signature = inspect.signature(function)
        self._unbound_signature = signature.replace(
            parameters=list(signature.parameters.values())[1:]
        )
        self._class_accessor = None

    def __get__(self, instance, owner):
        if instance is not None:
            return self.function.__get__(instance, owner)

        if self._class_accessor is None:
            function = self.function

            @wraps(function)
            def call_on_active_scene(*args, **kwargs):
                scene = SceneManager.instance().current_scene
                return function(scene, *args, **kwargs)

            # ``wraps`` sets __wrapped__, which would otherwise make
            # inspect.signature report the bound-method ``self`` parameter.
            call_on_active_scene.__signature__ = self._unbound_signature
            self._class_accessor = call_on_active_scene
        return self._class_accessor


class Scene(RenderLoopMixin):
    """The container that turns recorded animations into rendered video.

    A Scene owns its actor registry, camera, lights, timeline, animation
    contexts, audio state, and render loop. Creating one pushes it onto the
    process-global :class:`~algan.scene_manager.SceneManager` stack, making it
    the destination for mobs constructed without an explicit ``scene``.

    Rendering (:meth:`get_frames`, from
    :class:`~algan.render_loop.RenderLoopMixin`) proceeds in batches of frames
    sized to the memory budget: for each batch this scene's timeline
    materializes every actor's animated state at the batch's frame times,
    actors produce render primitives, and the ray tracer renders and
    post-processes the frames, which are streamed to the video writer. Batch
    preparation for the next batch runs concurrently on a worker thread
    (``ALGAN_PREFETCH_BATCHES=0`` disables).

    Parameters
    ----------
    video_settings
        Resolution / fps / quality settings (see
        :mod:`algan.settings.video_settings`).
    background_frame
        Background color/image or procedural callable. A Taichi ``@ti.func``
        uses the scalar ``(x, y, time) -> color`` contract. Python callables
        passed through the render APIs receive broadcastable Torch tensors;
        the direct constructor retains its legacy coordinate-grid callback.
    memory
        Optional :class:`~algan.utils.memory_utils.ManualMemory` render arena.
    scene_initializer
        Callable run on (re)creation; the default spawns the camera and a
        point light.
    """

    def __init__(
        self,
        video_settings=None,
        background_frame=None,
        memory=None,
        scene_initializer=None,
    ):
        if video_settings is None:
            video_settings = SETTINGS.video
        if background_frame is None:
            background_frame = SETTINGS.style.frame
        self.set_video_settings(video_settings)
        self.current_time = 0
        self.min_time = 0
        self.max_time = 0
        self.background_is_set = False
        # Preserve the legacy direct-Scene constructor callback while leaving
        # a Taichi func deferred: a @ti.func can only be called from a kernel.
        if (
            callable(background_frame)
            and not getattr(background_frame, "_is_taichi_function", False)
        ):
            background_frame = background_frame(
                torch.stack(
                    (
                        torch.arange(self.num_pixels_screen_height)
                        .view(-1, 1)
                        .expand([-1, self.num_pixels_screen_width]),
                        torch.arange(self.num_pixels_screen_width)
                        .view(1, -1)
                        .expand([self.num_pixels_screen_height, -1]),
                    ),
                    -1,
                )
            )
        self.background_frame = background_frame
        self._initial_background_frame = background_frame
        self.background_color = background_frame
        self.actors = [[]]
        self.effects = []
        self.camera = None
        self.light_sources = []
        self.scene_times = [[self.current_time, self.current_time]]
        depth_source = (
            SETTINGS.style.frame if callable(background_frame) else background_frame
        )
        self.background_depths = torch.full_like(
            depth_source[..., :1],
            dtype=torch.get_default_dtype(),
            fill_value=1e12,
        )
        self.animation_off = False
        self.context_max_time = 0
        self.environment_map = None
        self.environment_intensity = 1.0
        self.environment_ambient = True
        self.priority = 0
        self.id_count = 0
        self.allow_new_actors = True
        self.animate_scene_clear = False
        self.memory = memory

        # Every Scene is a self-contained authoring universe.  Mobs always use
        # these references after construction rather than consulting globals.
        self.timeline_manager = TimelineManager()
        self.animation_manager = AnimationManager(scene=self)
        self.audio_manager = AudioManager(scene=self)

        manager = SceneManager.instance()
        if scene_initializer is None:
            scene_initializer = (
                type(manager)._scene_initializer or (lambda scene: scene)
            )
        self.scene_initializer = scene_initializer
        self._terminated = False
        self._context_depth = 0
        manager.push(self)
        try:
            self.reset_scene()
        except Exception:
            manager.terminate(self)
            raise

    def __enter__(self):
        SceneManager.instance().push(self)
        self._context_depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._context_depth = max(0, self._context_depth - 1)
        if self._context_depth == 0:
            self.terminate()
        return False

    def terminate(self):
        """Pop this scene from the active-scene stack and return it."""
        SceneManager.instance().terminate(self)
        return self

    @active_scene_method
    def wait(self, time=1):
        self.animation_manager.wait(time)
        return self

    @staticmethod
    def instance():
        """Compatibility accessor for the current active scene."""
        return SceneManager.instance().current_scene

    @staticmethod
    def current():
        """Alias for instance."""
        return Scene.instance()

    @active_scene_method
    def get_camera(self):
        return self.camera

    @active_scene_method
    def get_light_sources(self):
        return self.light_sources

    @active_scene_method
    def add_light_source(self, light_source):
        if not hasattr(self, "light_sources"):
            self.light_sources = []
        if not any(light is light_source for light in self.light_sources):
            self.light_sources.append(light_source)
        return light_source

    @active_scene_method
    def remove_light_source(self, light_source):
        """Remove a light from this scene and return the light."""
        self.light_sources[:] = [
            light for light in self.light_sources if light is not light_source
        ]
        return light_source

    @active_scene_method
    def clear_light_sources(self):
        """Remove every registered light and return this scene."""
        self.light_sources.clear()
        return self

    add_light = add_light_source
    remove_light = remove_light_source

    @active_scene_method
    def set_environment_map(self, source, intensity=1.0, ambient=True):
        """Set an equirectangular environment map for this scene."""
        if source is None:
            self.environment_map = None
            return self
        env = source
        if isinstance(env, str):
            import cv2

            img = cv2.imread(env, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Could not read environment map image: {env}"
                )
            env = torch.from_numpy(img[..., ::-1].copy())  # BGR -> RGB
        if not torch.is_tensor(env):
            env = torch.tensor(env)
        env = env.float()
        if env.dim() != 3 or env.shape[-1] < 3:
            raise ValueError(
                "Environment map must have shape [height, width, >=3], got "
                f"{tuple(env.shape)}"
            )
        if env.max() > 1.5:
            env = env / 255.0
        self.environment_map = env[..., :3].contiguous()
        self.environment_intensity = float(intensity)
        self.environment_ambient = bool(ambient)
        return self

    def length_to_num_pixels(self, length):
        return length * 0.5 * self.num_pixels_screen_height

    def num_pixels_to_length(self, length):
        return length / (0.5 * self.num_pixels_screen_height)

    def set_current_time(self, t):
        self.current_time = t
        self.update_max_time(self.current_time)
        return self

    def increment_current_time(self, t):
        self.set_current_time(self.current_time + t)
        return self

    def update_max_time(self, t):
        self.context_max_time = max(self.context_max_time, t)
        self.max_time = max(self.max_time, t)
        return self

    def set_time_to_latest(self):
        self.current_time = self.max_time
        return self

    def add_actor(self, actor):
        if self.allow_new_actors:
            self.actors[-1].append(actor)
        return self

    def add_effect(self, effect):
        self.effects.append(effect)
        return self

    def initialize_frames(self):
        self.num_frames = int((self.max_time - self.min_time) * self.frames_per_second)
        return

    def clear(self):
        self.clear_scene()
        return self

    def despawn_scene(self, **kwargs):
        with Sync(animation_manager=self.animation_manager):
            for actor in list(
                sorted(self.actors[-1], key=lambda x: x.anchor_priority, reverse=True)
            ):
                if actor.is_spawned():
                    actor.despawn(**kwargs)

    def clear_scene(self, **kwargs):
        with Seq(run_time=0.5, animation_manager=self.animation_manager):
            self.despawn_scene(**kwargs)
        self.actors[-1] = [
            _ for _ in self.actors[-1] if (_.is_spawned() and _.is_despawned())
        ]

    def render_audio_to_file(self, file_path, frames_per_second=44100, codec='pcm_s32le', nbytes=4):
        if len(self.effects) == 0:
            return None

        clips_to_compose = []
        start_time = self.scene_times[-1][0] / self.video_settings.frames_per_second
        for audio_effect in self.effects:
            timed_clip = audio_effect.audio_clip.with_start(
                audio_effect.start_time_func() - start_time
            )
            clips_to_compose.append(timed_clip)

        from moviepy import CompositeAudioClip  # deferred: ~0.3 s of import algan

        audio_clip = CompositeAudioClip(clips_to_compose)
        audio_clip.duration = self.animation_manager.context.timespan.original_end
        audio_clip.write_audiofile(file_path, fps=frames_per_second, codec=codec, nbytes=nbytes)
        audio_clip.close()
        return file_path

    def reset_scene(self):
        self.actors = [[]]
        self.effects = []
        self.camera = None
        self.light_sources = []
        with (
            SceneManager.instance().activating(self),
            animation_manager_context(self.animation_manager),
        ):
            self.scene_initializer(self)

    def reset(self):
        """Reset only this scene's authoring state.

        Enclosing or sibling scenes on the SceneManager stack are untouched.
        Existing mob references from the old timeline should be considered
        invalid, matching the historical post-render reset contract.
        """
        self.current_time = 0
        self.min_time = 0
        self.max_time = 0
        self.context_max_time = 0
        self.id_count = 0
        self.scene_times = [[0, 0]]
        self.background_frame = self._initial_background_frame
        self.background_color = self._initial_background_frame
        self.background_is_set = False
        depth_source = (
            SETTINGS.style.frame
            if callable(self.background_frame)
            else self.background_frame
        )
        self.background_depths = torch.full_like(
            depth_source[..., :1],
            dtype=torch.get_default_dtype(),
            fill_value=1e12,
        )
        self.environment_map = None
        self.environment_intensity = 1.0
        self.environment_ambient = True
        self.animation_off = False
        self.priority = 0
        self.allow_new_actors = True
        self.animate_scene_clear = False
        self.timeline_manager = TimelineManager()
        self.animation_manager = AnimationManager(scene=self)
        self.audio_manager = AudioManager(scene=self)
        self.reset_scene()
        return self

    @active_scene_method
    def set_video_settings(self, video_settings):
        """Set this Scene's resolution, frame rate and anti-aliasing.

        ``video_settings`` is a :class:`~algan.settings.video_settings.VideoSettings`
        instance, usually one of the built-in presets (``PREVIEW``, ``LD``,
        ``MD``, ``HD``, ``PRODUCTION``, ``UHD``).

        Most scripts do not need this: pass ``video_settings`` to
        :meth:`save_video` / :meth:`save_frame` for a one-off render, or set
        ``SETTINGS.video`` for a process-wide default.
        """
        if not hasattr(video_settings, "resolution"):
            raise AlganConfigurationError(
                "video_settings must be a VideoSettings instance or preset "
                f"(for example HD or PREVIEW), got {type(video_settings).__name__}"
            )
        self.video_settings = (
            video_settings.as_preset()
            if hasattr(video_settings, "as_preset")
            else video_settings
        )
        video_settings = self.video_settings
        self.num_pixels_screen_width, self.num_pixels_screen_height = (
            video_settings.resolution
        )
        self.frame_size = torch.tensor(
            (self.num_pixels_screen_height, self.num_pixels_screen_width)
        )
        self.frames_per_second = video_settings.frames_per_second
        self.num_pixels = self.frame_size.prod()
        self.size = self.num_pixels_screen_width, self.num_pixels_screen_height
        return self

    def background_is_transparent(self):
        if hasattr(self.background_frame, '__call__'):
            return False
        return (self.background_frame[..., -1].min() < (1-(0.5/255))).item()

    def get_pixel_format(self):
        return "rgba" if self.background_is_transparent() else "rgb"

    def show_frame(self, time_stamp=None):
        from algan.utils.plotting_utils import plot_tensor
        if time_stamp is None:
            time_stamp = (
                self.animation_manager.context.current_time
                + 1.5 / self.video_settings.frames_per_second
            )
        time_ind = self._frame_index_for_timestamp(time_stamp)
        frames = []
        for frame in self.get_frames(time_ind, time_ind + 1):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        for frame in frames:
            plot_tensor(frame)

        return frames

    def _frame_index_for_timestamp(self, time_stamp):
        time_stamp = float(time_stamp)
        if not math.isfinite(time_stamp) or time_stamp < 0:
            raise AlganConfigurationError(
                f"Frame timestamp must be finite and non-negative, got {time_stamp}"
            )
        return round(time_stamp * self.video_settings.frames_per_second)

    def _render_still(self, destination, time_stamp):
        """Render one frame at ``time_stamp`` and write it to ``destination``."""
        if time_stamp is None:
            time_stamp = (
                self.animation_manager.context.timespan.current_time
                + 1.5 / self.video_settings.frames_per_second
            )
        time_ind = self._frame_index_for_timestamp(time_stamp)
        frames = []
        with torch.inference_mode():
            for frame in self.get_frames(time_ind, time_ind + 1):
                frame = frame.float() / 255
                frames.append(frame.squeeze(0).permute(-1, 0, 1))
        if not frames:
            raise RuntimeError("No frame was produced for the requested timestamp")
        import torchvision.utils  # deferred: ~0.2 s of import algan

        torchvision.utils.save_image(frames[-1], str(destination))
        return destination

    @active_scene_method
    def save_frame(
        self,
        file_path=None,
        video_settings=None,
        at=None,
        *,
        overwrite=True,
        background_color=None,
    ):
        """Render one or more still frames from this Scene.

        Unlike :meth:`save_video` this never modifies the Scene: nothing is
        despawned, the timeline is left as authored, and any temporary video
        settings or background are restored before returning. Call it as often
        as you like while building a scene.

        Parameters
        ----------
        file_path
            Where to write the image. A bare filename is placed in Algan's
            output directory; a path with a parent directory is used as given.
            A missing extension defaults to ``.png``.
        video_settings
            Resolution and anti-aliasing for this still only, normally a
            preset such as ``HD``. Defaults to the Scene's current settings.
        at
            Timestamp in seconds to capture, or a sequence of timestamps to
            capture several stills in one call. Each timestamp must be finite
            and non-negative. When omitted, captures just after the current
            authoring time.
        overwrite
            When False, existing files are left alone and reported as
            ``"skipped"``.
        background_color
            A color, image, or procedural callable, applied to this still only.

        Returns
        -------
        RenderResult or list of RenderResult
            One result per still. A list is returned only when ``at`` is a
            sequence, matching the shape of the input.

        Examples
        --------
        .. code-block:: python

            Scene.save_frame("thumbnail", HD)
            Scene.save_frame("shot.png", at=2.5)
            Scene.save_frame("contact_sheet", at=[0, 1, 2])
        """
        # Import lazily to avoid the Scene/algan_utils import cycle during
        # package initialization while sharing video output's exact resolver.
        from algan.utils.algan_utils import RenderResult, _resolve_output_destination

        destination = _resolve_output_destination(file_path, ".png")
        if at is None or not hasattr(at, "__len__"):
            targets = [(destination, at)]
            returns_list = False
        else:
            suffix = destination.suffix
            stem = destination.with_suffix("")
            targets = [
                (stem.with_name(f"{stem.name}_{time_stamp}{suffix}"), time_stamp)
                for time_stamp in at
            ]
            returns_list = True

        previous_settings = self.video_settings
        previous_background = (
            self.background_frame,
            self.background_color,
            self.background_is_set,
        )
        results = []
        try:
            if video_settings is not None:
                self.set_video_settings(video_settings)
            if background_color is not None:
                self.set_background_color(background_color)
            for target, time_stamp in targets:
                if target.exists() and not overwrite:
                    results.append(RenderResult("skipped", target))
                    continue
                started = time.perf_counter()
                self._render_still(target, time_stamp)
                results.append(
                    RenderResult(
                        "rendered", target, time.perf_counter() - started
                    )
                )
        finally:
            # set_video_settings restores every derived cache (dimensions,
            # fps, frame size, pixel count), not merely the settings reference.
            if self.video_settings is not previous_settings:
                self.set_video_settings(previous_settings)
            (
                self.background_frame,
                self.background_color,
                self.background_is_set,
            ) = previous_background

        return results if returns_list else results[0]

    @active_scene_method
    def set_background_color(self, background_color, overwrite=True):
        if (background_color is None) or (self.background_is_set and not overwrite):
            return self
        if isinstance(background_color, str):
            a = self.video_settings.anti_alias_level
            background_color = F.interpolate(get_image(background_color).transpose(0,-1).unsqueeze(0), [_*a for _ in tuple(self.frame_size)],
                                             mode='bilinear', antialias='bilinear').squeeze(0).permute(1,2,0).unsqueeze(0)
        self.background_frame = self.background_color = background_color
        self.background_is_set = True
        return self

    @active_scene_method
    def get_background_color(self):
        return self.background_color

    def get_new_id(self):
        self.id_count += 1
        return self.id_count - 1

    @active_scene_method
    def save_video(
        self,
        file_path=None,
        video_settings=None,
        *,
        overwrite=True,
        reset=False,
        background_color=None,
        animate_fade_out=None,
        post_processes=None,
        codec=None,
        audio_codec=None,
        ffmpeg_params=None,
    ):
        """Render everything recorded on this Scene to a video file.

        Parameters
        ----------
        file_path
            Where to write the video. A bare filename such as ``"my_video"``
            is placed in Algan's output directory; a path with a parent
            directory, relative or absolute, is used exactly as given. If the
            name has no extension Algan appends ``.mp4``, or ``.mov`` when the
            background is transparent. Defaults to
            ``SETTINGS.paths.output_filename``.
        video_settings
            Resolution, frame rate and anti-aliasing for this render, normally
            one of the presets (``PREVIEW``, ``LD``, ``MD``, ``HD``,
            ``PRODUCTION``, ``UHD``). Applies to this render only; the Scene's
            own settings are restored afterwards. Defaults to
            ``SETTINGS.video``.
        overwrite
            When False and the destination already exists, skip rendering and
            return a ``"skipped"`` result instead of replacing the file.
        reset
            When True, discard this Scene's recorded animation after
            rendering, despawn its mobs and rebuild its timeline, animation
            and audio managers. Mobs created before the render become unusable.
            The default leaves the Scene exactly as authored, so you can keep
            animating and render again.
        background_color
            A color, image, or procedural callable ``(x, y, time) -> color``.
            Python callables receive broadcastable Torch tensors. A Taichi
            ``@ti.func`` receives scalar normalized coordinates and time and
            must return a color vector; it is evaluated for the whole render
            batch by one Taichi kernel writing directly into the output buffer.
        animate_fade_out
            Whether to fade every spawned mob out at the end of the video.
            Recorded on the timeline, so it persists even when ``reset`` is
            False. Defaults to ``SETTINGS.style.fade_out_on_scene_end``.
        post_processes
            Post-processing passes to apply to each frame. Defaults to bloom.
        codec, audio_codec, ffmpeg_params
            Encoder overrides passed through to FFmpeg. Algan picks sensible
            defaults from the background's transparency.

        Returns
        -------
        RenderResult
            Metadata with ``status`` (``"rendered"`` or ``"skipped"``),
            ``output_path``, ``duration_seconds`` and the resolved
            ``render_plan``.

        Examples
        --------
        .. code-block:: python

            Scene.save_video("my_video")            # LD into algan_outputs/
            Scene.save_video("my_video", HD)        # one-off quality override
            Scene.save_video("renders/final.mov")   # explicit directory
        """
        from algan.utils.algan_utils import _render_scene_to_file

        # render_to_video owns the post-processing default, so only forward an
        # explicit choice rather than restating it here.
        extra = {} if post_processes is None else {"post_processes": post_processes}
        with (
            SceneManager.instance().activating(self),
            animation_manager_context(self.animation_manager),
        ):
            return _render_scene_to_file(
                self,
                file_path=file_path,
                video_settings=video_settings,
                overwrite=overwrite,
                reset=reset,
                codec=codec,
                audio_codec=audio_codec,
                background_color=background_color,
                ffmpeg_params=ffmpeg_params,
                animate_fade_out=animate_fade_out,
                **extra,
            )

    @staticmethod
    def render_all_funcs(*args, **kwargs):
        """Render discovered scene functions in isolated Scene contexts."""
        from algan.utils.algan_utils import render_all_funcs

        return render_all_funcs(*args, **kwargs)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
