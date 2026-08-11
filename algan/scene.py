from __future__ import annotations

import inspect
import math
import time
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn.functional as F

from algan.animation_timeline.animation_contexts import (
    AnimationManager,
    Seq,
    Sync,
    animation_manager_context,
)
from algan.animation_timeline.timeline import TimelineManager
from algan.constants.spatial import *
from algan.errors import AlganConfigurationError

# EmptySceneWarning and write_frames_from_queue moved to render_loop.py;
# re-exported here for backwards compatibility.
from algan.render_loop import (  # noqa: F401
    EmptySceneWarning,
    RenderLoopMixin,
    write_frames_from_queue,
)
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.settings.video_settings import VideoSettings
from algan.sound.audio_effect import AudioManager
from algan.utils.file_utils import get_image

if TYPE_CHECKING:  # algan_utils imports Scene, so only for annotations.
    from algan.utils.algan_utils import RenderResult


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

    Rendering (:meth:`~algan.render_loop.RenderLoopMixin.get_frames`, from
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
        if callable(background_frame) and not getattr(
            background_frame, "_is_taichi_function", False
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
        self.actors = []
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
        self._project_run = None
        self._suppress_automatic_transcript = False

        # Every Scene is a self-contained authoring universe.  Mobs always use
        # these references after construction rather than consulting globals.
        self.timeline_manager = TimelineManager()
        self.animation_manager = AnimationManager(scene=self)
        self.audio_manager = AudioManager(scene=self)

        manager = SceneManager.instance()
        if scene_initializer is None:
            scene_initializer = type(manager)._scene_initializer or (
                lambda scene: scene
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
    def wait(self, time: float = 1):
        """Hold the scene still for a while.

        Advances time without changing anything, leaving a pause in the video --
        room for narration, or a beat before the next animation.

        Animation
        ---------
        Recorded on the timeline: it consumes video time and nothing else.

        Parameters
        ----------
        time
            How long to wait, in seconds. Defaults to ``1``.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.animation_manager.wait(time)
        return self

    @staticmethod
    def instance():
        """Get the Scene currently being authored.

        Creates the default Scene on first use, so this never returns ``None``.

        Returns
        -------
        :class:`~.Scene`
            The active Scene.
        """
        return SceneManager.instance().current_scene

    @staticmethod
    def current():
        """Get the Scene currently being authored; an alias of :meth:`~.Scene.instance`.

        Returns
        -------
        :class:`~.Scene`
            The active Scene.
        """
        return Scene.instance()

    @active_scene_method
    def get_camera(self):
        """Get this Scene's camera.

        Returns
        -------
        :class:`~.Camera`
            The camera, or ``None`` if the Scene has not been initialized with one.
        """
        return self.camera

    @active_scene_method
    def get_light_sources(self):
        """Get this Scene's lights.

        Returns
        -------
        list[:class:`~.Light`]
            The live list of registered lights; mutating it changes the Scene.
        """
        return self.light_sources

    @active_scene_method
    def add_light_source(self, light_source):
        """Add a light to this Scene.

        Lights only affect Mobs whose material responds to light -- a
        :class:`~algan.rendering.shaders.materials.MeshBasicMaterial` looks the
        same however the scene is lit.
        Adding the same light twice does nothing.

        Animation
        ---------
        Not animated: the light exists from this point in the timeline onwards.

        Parameters
        ----------
        light_source
            The light to add.

        Returns
        -------
        :class:`~.Light`
            The light that was added, so it can be kept and animated.
        """
        if not hasattr(self, "light_sources"):
            self.light_sources = []
        if not any(light is light_source for light in self.light_sources):
            self.light_sources.append(light_source)
        return light_source

    @active_scene_method
    def remove_light_source(self, light_source):
        """Remove a light from this Scene.

        Removing a light that is not registered does nothing.

        Animation
        ---------
        Not animated: the light stops contributing from this point in the timeline
        onwards.

        Parameters
        ----------
        light_source
            The light to remove.

        Returns
        -------
        :class:`~.Light`
            The light that was passed in.
        """
        self.light_sources[:] = [
            light for light in self.light_sources if light is not light_source
        ]
        return light_source

    @active_scene_method
    def clear_light_sources(self):
        """Remove every light from this Scene.

        Lit materials go black afterwards unless a new light or an environment map
        is added.

        Animation
        ---------
        Not animated: the lights stop contributing from this point in the timeline
        onwards.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.light_sources.clear()
        return self

    add_light = add_light_source
    remove_light = remove_light_source

    @active_scene_method
    def set_environment_map(self, source, intensity: float = 1.0, ambient: bool = True):
        """Light the Scene with an environment map, and show it as a backdrop.

        An equirectangular image surrounds the scene, so reflective and metallic
        materials pick up their surroundings instead of reflecting a void -- the
        cheapest way to make metal look like metal.

        Animation
        ---------
        Not animated: the map applies from this point in the timeline onwards.

        Parameters
        ----------
        source
            Path to an image file, or an image tensor of shape
            ``[height, width, >=3]``. Values above ``1.5`` are treated as 0-255 and
            scaled down. ``None`` removes the current map.
        intensity
            Brightness multiplier for the map's contribution. Defaults to ``1.0``.
        ambient
            Whether the map also provides ambient light to non-reflective surfaces,
            rather than only appearing in reflections. Defaults to True.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.

        Raises
        ------
        FileNotFoundError
            If ``source`` is a path that cannot be read.
        ValueError
            If the image is not shaped ``[height, width, >=3]``.
        """
        if source is None:
            self.environment_map = None
            return self
        env = source
        if isinstance(env, str):
            import cv2

            img = cv2.imread(env, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read environment map image: {env}")
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

    def length_to_num_pixels(self, length: float) -> float:
        """Convert a world-space length to a length in rendered pixels.

        Parameters
        ----------
        length
            Length in world units.

        Returns
        -------
        float
            The equivalent number of pixels at the Scene's current resolution.
        """
        return length * 0.5 * self.num_pixels_screen_height

    def num_pixels_to_length(self, length: float) -> float:
        """Convert a length in rendered pixels to a world-space length.

        Parameters
        ----------
        length
            Length in pixels.

        Returns
        -------
        float
            The equivalent length in world units at the Scene's current resolution.
        """
        return length / (0.5 * self.num_pixels_screen_height)

    def set_current_time(self, t: float):
        """Internal: move the authoring cursor to an absolute time.

        Parameters
        ----------
        t
            New authoring time, in seconds.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.current_time = t
        self.update_max_time(self.current_time)
        return self

    def increment_current_time(self, t: float):
        """Internal: advance the authoring cursor by an interval.

        Parameters
        ----------
        t
            How far to advance, in seconds.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.set_current_time(self.current_time + t)
        return self

    def update_max_time(self, t: float):
        """Internal: extend the recorded end of the animation to include a time.

        The video's length is the largest time any recording reached, which is what
        this tracks.

        Parameters
        ----------
        t
            Time in seconds that must fall within the animation.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.context_max_time = max(self.context_max_time, t)
        self.max_time = max(self.max_time, t)
        return self

    def set_time_to_latest(self):
        """Move the authoring cursor to the end of everything recorded so far.

        Use it after animations that ran in parallel, to carry on from the end of the
        longest one.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.current_time = self.max_time
        return self

    def add_actor(self, actor):
        """Register a Mob with this Scene so it takes part in rendering.

        Mob constructors call this for you; you only need it for a Mob built with
        ``add_to_scene=False`` that you later decide to render.

        Parameters
        ----------
        actor
            The Mob to register. Ignored if the Scene is no longer accepting actors
            (during a render).

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        if self.allow_new_actors:
            self.actors.append(actor)
        return self

    def add_effect(self, effect):
        """Register an audio effect with this Scene.

        The :class:`~.Audio` and :class:`~.Speech` contexts use this; the effect's own
        start time decides where it lands in the finished video.

        Parameters
        ----------
        effect
            The :class:`~.AudioEffect` to add.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.effects.append(effect)
        return self

    def initialize_frames(self):
        """Internal: work out how many frames the recorded animation needs.

        Derives the frame count from the recorded duration and the Scene's frame
        rate. Called by the render loop before rendering.
        """
        self.num_frames = int((self.max_time - self.min_time) * self.frames_per_second)
        return

    def clear(self):
        """Despawn everything in the Scene; an alias of :meth:`~.Scene.clear_scene`.

        Animation
        ---------
        Recorded as an animation: every spawned Mob fades out together over 0.5
        seconds.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        self.clear_scene()
        return self

    def despawn_scene(self, **kwargs):
        """Despawn every spawned Mob in the Scene.

        Parents are despawned before their children, so composite Mobs disappear as a
        unit rather than in pieces.

        Animation
        ---------
        Recorded as an animation: all the despawns run together inside a
        :class:`~.Sync`, over the current context's duration (1 second by default).

        Parameters
        ----------
        **kwargs
            Passed to each :meth:`~.Animatable.despawn` -- notably
            ``animate=False`` to remove everything without fading.
        """
        with Sync(animation_manager=self.animation_manager):
            for actor in sorted(
                self.actors, key=lambda x: x.anchor_priority, reverse=True
            ):
                if actor.is_spawned():
                    actor.despawn(**kwargs)

    def clear_scene(self, **kwargs):
        """Despawn everything and retain the Mobs with recorded history.

        Like :meth:`~.Scene.despawn_scene`, then keeps the fully despawned actors
        whose earlier lifespan still has to render. Actors which never acquired a
        complete lifespan are discarded.

        Animation
        ---------
        Recorded as an animation: everything fades out together over **0.5 seconds**,
        regardless of the current context's duration.

        Parameters
        ----------
        **kwargs
            Passed to each :meth:`~.Animatable.despawn` -- notably ``animate=False``.
        """
        with Seq(run_time=0.5, animation_manager=self.animation_manager):
            self.despawn_scene(**kwargs)
        self.actors = [_ for _ in self.actors if (_.is_spawned() and _.is_despawned())]

    def render_audio_to_file(
        self,
        file_path: str | Path,
        frames_per_second: int = 44100,
        codec: str = "pcm_s32le",
        nbytes: int = 4,
    ):
        """Mix this Scene's audio effects down to an audio file.

        Every registered effect is placed at its recorded start time and the result is
        written out. :meth:`~.Scene.save_video` does this for you; call it directly
        only when you want the audio on its own.

        Parameters
        ----------
        file_path
            Where to write the audio.
        frames_per_second
            Sample rate in Hz. Defaults to ``44100``.
        codec
            FFmpeg audio codec. Defaults to ``'pcm_s32le'`` (uncompressed).
        nbytes
            Bytes per sample. Defaults to ``4``.

        Returns
        -------
        str or pathlib.Path or None
            The path written, or ``None`` if the Scene has no audio effects.
        """
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
        audio_clip.write_audiofile(
            file_path, fps=frames_per_second, codec=codec, nbytes=nbytes
        )
        audio_clip.close()
        return file_path

    def reset_scene(self):
        """Rebuild the Scene's contents from its initializer.

        Drops all actors, audio effects, the camera and the lights, then re-runs the
        Scene initializer, which puts the default camera and lighting back. The
        timeline is **not** cleared -- use :meth:`~.Scene.reset` for that.

        Animation
        ---------
        Not animated: everything is discarded rather than despawned, so nothing fades
        out.
        """
        self.actors = []
        self.effects = []
        self.camera = None
        self.light_sources = []
        with (
            SceneManager.instance().activating(self),
            animation_manager_context(self.animation_manager),
        ):
            self.scene_initializer(self)

    def reset(self):
        """Empty the Scene completely and start over.

        Time returns to zero and the timeline, animation and audio managers are
        rebuilt, so nothing recorded so far survives. **Mob references from before
        the reset are invalid** and must not be reused. Other Scenes on the
        SceneManager stack are untouched.

        Animation
        ---------
        Not animated, and destructive: this discards the recording rather than
        animating anything out.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
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

    def background_is_transparent(self) -> bool:
        """Whether the Scene's background has any transparency.

        This decides the output format: a transparent background makes
        :meth:`~.Scene.save_video` write ``.mov`` with an alpha channel instead of
        ``.mp4``. A procedural background is always treated as opaque.

        Returns
        -------
        bool
            Whether any background pixel is less than fully opaque.
        """
        if callable(self.background_frame):
            return False
        return (self.background_frame[..., -1].min() < (1 - (0.5 / 255))).item()

    def get_pixel_format(self) -> str:
        """Get the pixel format the Scene's frames should be encoded in.

        Returns
        -------
        str
            ``"rgba"`` if the background is transparent, otherwise ``"rgb"``.
        """
        return "rgba" if self.background_is_transparent() else "rgb"

    def show_frame(self, time_stamp: float | None = None):
        """Render one frame and display it, for interactive work.

        Meant for a notebook or REPL: it plots the frame rather than writing a file.
        Use :meth:`~.Scene.save_frame` to save one instead.

        Animation
        ---------
        Not animated and non-destructive: rendering a frame leaves the Scene as
        authored.

        Parameters
        ----------
        time_stamp
            Time to render, in seconds. Defaults to ``None``, meaning just after the
            current authoring time -- i.e. the scene as it stands.

        Returns
        -------
        list[torch.Tensor]
            The frame(s) that were plotted, as ``(channels, height, width)`` tensors
            with values in ``[0, 1]``.
        """
        from algan.utils.plotting_utils import plot_tensor

        if time_stamp is None:
            time_stamp = (
                self.animation_manager.context.current_time
                + 1.5 / self.video_settings.frames_per_second
            )
        time_ind = self._frame_index_for_timestamp(time_stamp)
        frames = []
        # See Scene.save_frame: a render that leaves the Scene re-renderable
        # must not leave its replay-window resolution behind for the next one.
        with self.timeline_manager.preserving_authoring_state(
            preserve_replay_resolution=self.animation_manager.context.prev_context
            is not None
        ):
            for frame in self.get_frames(time_ind, time_ind + 1):
                frame = frame.float() / 255
                frames.append(frame.squeeze(0).permute(-1, 0, 1))
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

    def _resolve_still_timestamp(self, time_stamp):
        """Resolve a still timestamp against the current authoring cursor."""
        if time_stamp is None:
            return (
                self.animation_manager.context.timespan.current_time
                + 1.5 / self.video_settings.frames_per_second
            )
        time_stamp = float(time_stamp)
        if time_stamp < 0:
            time_stamp += self.animation_manager.context.timespan.current_time
        return time_stamp

    def _render_still(self, destination, time_stamp):
        """Render one frame at ``time_stamp`` and write it to ``destination``."""
        time_stamp = self._resolve_still_timestamp(time_stamp)
        time_ind = self._frame_index_for_timestamp(time_stamp)
        frame = None
        with torch.inference_mode():
            for batch in self.get_frames(time_ind, time_ind + 1):
                if batch.shape[0]:
                    frame = batch[-1]
        if frame is None:
            raise RuntimeError("No frame was produced for the requested timestamp")
        # The render loop already returns a CPU uint8 HWC image.  Passing it
        # through torchvision converted it to float CHW and, on first use,
        # imported torchvision's datasets/models/ops packages solely to reach
        # ``save_image``.  Pillow can write the finished pixels directly.
        from PIL import Image

        Image.fromarray(frame.contiguous().numpy()).save(str(destination))
        return destination

    @active_scene_method
    def save_frame(
        self,
        file_path: str | Path | None = None,
        video_settings: VideoSettings | None = None,
        at: float | Sequence[float] | None = None,
        *,
        overwrite: bool = True,
        background_color=None,
    ) -> RenderResult | list[RenderResult]:
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
            A missing extension defaults to ``.png``. Defaults to ``None``,
            meaning ``SETTINGS.paths.output_filename``.
        video_settings
            Resolution and anti-aliasing for this still only, normally a
            preset such as ``HD``. Defaults to ``None``, meaning the Scene's
            current settings.
        at
            Timestamp in seconds to capture, or a sequence of timestamps to
            capture several stills in one call. A negative timestamp is an
            offset backwards from the current authoring time, so ``-0.5``
            captures half a second before the current context's cursor.
            Resolved timestamps must be finite and non-negative. Defaults to
            ``None``, capturing just after the current authoring time -- i.e.
            the scene as it stands.
        overwrite
            Whether an existing file at the destination is replaced. Defaults to
            True; False leaves it alone and reports ``"skipped"``.
        background_color
            A color, image, or procedural callable, applied to this still only.
            Defaults to ``None``, meaning keep the Scene's background.

        Returns
        -------
        RenderResult or list of RenderResult
            One result per still, with ``status`` (``"rendered"`` or
            ``"skipped"``), ``output_path`` and ``duration_seconds``. A list is
            returned only when ``at`` is a sequence, matching the shape of the
            input.

        Examples
        --------
        .. code-block:: python

            Scene.save_frame("thumbnail", HD)
            Scene.save_frame("shot.png", at=2.5)
            Scene.save_frame("previous.png", at=-0.5)
            Scene.save_frame("contact_sheet", at=[0, 1, 2])
        """
        project_run = self._project_run
        if project_run is None:
            from algan.project import _get_active_project_run

            project_run = _get_active_project_run()
        if project_run is not None:
            file_path = project_run.prepare_frame_path(file_path)
            # False in video mode, and for a frame the project run's selection
            # has filtered out.
            if not project_run.should_render_frame():
                return []
        if SETTINGS.skip_save_frame:
            return []
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
            # Rendering resolves replay windows against the timings as they
            # stand. Mid-authoring those are not final -- an enclosing context
            # with a run_time rescales its block when it exits -- so the
            # resolution is restored rather than left on the timeline for the
            # next render to reuse.
            with self.timeline_manager.preserving_authoring_state(
                preserve_replay_resolution=self.animation_manager.context.prev_context
                is not None
            ):
                for target, time_stamp in targets:
                    if target.exists() and not overwrite:
                        results.append(RenderResult("skipped", target))
                        continue
                    started = time.perf_counter()
                    self._render_still(target, time_stamp)
                    results.append(
                        RenderResult("rendered", target, time.perf_counter() - started)
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

        result = results if returns_list else results[0]
        if project_run is not None:
            project_run.record_frame_results(result)
        return result

    @active_scene_method
    def set_background_color(self, background_color, overwrite: bool = True):
        """Set what the Scene is drawn against.

        Animation
        ---------
        Not animated: the background changes for the whole video, not from this point
        onwards, since it is Scene state rather than timeline state. For a one-off
        render, pass ``background_color`` to :meth:`~.Scene.save_video` instead.

        Parameters
        ----------
        background_color
            A colour, a path to an image (scaled to the frame), or a procedural
            callable ``(x, y, time) -> color``. A colour with alpha below 1 makes the
            output transparent. ``None`` leaves the background unchanged.
        overwrite
            Whether to replace a background that has already been set. Defaults to
            True; False makes the call a no-op once a background exists, which is how
            defaults are applied without stomping a user's choice.

        Returns
        -------
        :class:`~.Scene`
            This Scene, so calls can be chained.
        """
        if (background_color is None) or (self.background_is_set and not overwrite):
            return self
        if isinstance(background_color, str):
            a = self.video_settings.anti_alias_level
            # get_image returns [height, width, channels]; interpolate wants
            # [1, channels, height, width]. (``transpose(0, -1)`` here swapped
            # the image's rows and columns instead, rendering it transposed.)
            image = get_image(background_color).permute(2, 0, 1).unsqueeze(0)
            image = (
                F.interpolate(
                    image,
                    [_ * a for _ in tuple(self.frame_size)],
                    mode="bilinear",
                    antialias="bilinear",
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )
            # Frame buffers are bottom-up (post_process_frames flips them on
            # the way out, matching the tracer's py = height-1-row), so the
            # background rows have to be stored bottom-up too -- as the
            # procedural background path already produces them.
            background_color = image.flip(0).unsqueeze(0)
        self.background_frame = self.background_color = background_color
        self.background_is_set = True
        return self

    @active_scene_method
    def get_background_color(self):
        """Get the Scene's current background.

        Returns
        -------
        :class:`~.Color` or torch.Tensor or Callable
            Whatever the background was set to: a colour, an image tensor, or a
            procedural callable.
        """
        return self.background_color

    def get_new_id(self) -> int:
        """Internal: allocate the next Mob id for this Scene.

        Ids key a Mob's rows on the Scene timeline. Called during Mob construction.

        Returns
        -------
        int
            A fresh id, unique within this Scene.
        """
        self.id_count += 1
        return self.id_count - 1

    @active_scene_method
    def save_video(
        self,
        file_path: str | Path | None = None,
        video_settings: VideoSettings | None = None,
        *,
        overwrite: bool = True,
        reset: bool = False,
        background_color=None,
        animate_fade_out: bool | None = None,
        post_processes=None,
        codec: str | None = None,
        audio_codec: str | None = None,
        ffmpeg_params: list[str] | None = None,
    ) -> RenderResult:
        """Render everything recorded on this Scene to a video file.

        Parameters
        ----------
        file_path
            Where to write the video. A bare filename such as ``"my_video"``
            is placed in Algan's output directory; a path with a parent
            directory, relative or absolute, is used exactly as given. If the
            name has no extension Algan appends ``.mp4``, or ``.mov`` when the
            background is transparent. Defaults to ``None``, meaning
            ``SETTINGS.paths.output_filename``.
        video_settings
            Resolution, frame rate and anti-aliasing for this render, normally
            one of the presets (``PREVIEW``, ``LD``, ``MD``, ``HD``,
            ``PRODUCTION``, ``UHD``). Applies to this render only; the Scene's
            own settings are restored afterwards. Defaults to ``None``, meaning
            ``SETTINGS.video``.
        overwrite
            Whether an existing file at the destination is replaced. Defaults to
            True; False skips rendering and returns a ``"skipped"`` result.
        reset
            Whether to tear the Scene down after rendering: discard its recorded
            animation, despawn its mobs and rebuild its timeline, animation and
            audio managers. Mobs created before the render become unusable.
            Defaults to False, which leaves the Scene exactly as authored, so you
            can keep animating and render again -- including from inside a
            ``with`` block that has not finished yet. A mid-block render covers
            everything recorded so far and changes nothing, so the final render
            is the same as if the preview had never happened.
        background_color
            A color, image, or procedural callable ``(x, y, time) -> color``.
            Python callables receive broadcastable Torch tensors. A Taichi
            ``@ti.func`` receives scalar normalized coordinates and time and
            must return a color vector; it is evaluated for the whole render
            batch by one Taichi kernel writing directly into the output buffer.
            Defaults to ``None``, meaning keep the Scene's background.
        animate_fade_out
            Whether to fade every spawned mob out at the end of the video.
            Recorded on the timeline, so it persists even when ``reset`` is
            False. Defaults to ``None``, meaning
            ``SETTINGS.style.fade_out_on_scene_end`` (``False``).
        post_processes
            Post-processing passes to apply to each frame. Defaults to ``None``,
            meaning bloom.
        codec, audio_codec, ffmpeg_params
            Encoder overrides passed through to FFmpeg. Each defaults to
            ``None``, letting Algan pick from the background's transparency.

        Returns
        -------
        RenderResult
            Metadata with ``status`` (``"rendered"`` or ``"skipped"``),
            ``output_path``, ``duration_seconds`` and the resolved
            ``render_plan``.

        Examples
        --------
        .. code-block:: python

            Scene.save_video("my_video")  # LD into algan_outputs/
            Scene.save_video("my_video", HD)  # one-off quality override
            Scene.save_video("renders/final.mov")  # explicit directory
        """
        project_run = self._project_run
        if project_run is None:
            from algan.project import _get_active_project_run

            project_run = _get_active_project_run()
        if project_run is not None and not project_run.allow_video_render:
            from algan.utils.algan_utils import (
                RenderResult,
                _resolve_output_destination,
            )

            default_extension = ".mov" if self.background_is_transparent() else ".mp4"
            destination = _resolve_output_destination(file_path, default_extension)
            return RenderResult("skipped", destination)

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
