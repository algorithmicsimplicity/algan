import torch
import torch.nn.functional as F
import torchvision.utils
from moviepy import CompositeAudioClip

from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS

from algan.constants.color import *
from algan.constants.spatial import *

from algan.animation.animation_contexts import Seq, Sync, AnimationManager

# EmptySceneWarning and write_frames_from_queue moved to render_loop.py;
# re-exported here for backwards compatibility.
from algan.render_loop import (  # noqa: F401
    EmptySceneWarning,
    RenderLoopMixin,
    write_frames_from_queue,
)
from algan.utils.file_utils import get_image
from algan.scene_manager import SceneManager


class Scene(RenderLoopMixin):
    """The container that turns recorded animations into rendered video.

    A Scene owns the registry of every :class:`~.Animatable` created while it
    is active (``actors``), the camera and light sources, and the render loop.
    User scripts rarely construct one: importing ``algan`` configures the
    :class:`~algan.scene_manager.SceneManager` singleton, which creates the
    scene lazily on first use, and ``render_to_file()`` drives it.

    Rendering (:meth:`get_frames`, from
    :class:`~algan.render_loop.RenderLoopMixin`) proceeds in batches of frames
    sized to the memory budget: for each batch the global timeline
    materializes every actor's animated state at the batch's frame times,
    actors produce render primitives, and the ray tracer renders and
    post-processes the frames, which are streamed to the video writer. Batch
    preparation for the next batch runs concurrently on a worker thread
    (``ALGAN_PREFETCH_BATCHES=0`` disables).

    Parameters
    ----------
    background_frame
        Background color/image, or a callable ``(y, x) -> color`` evaluated
        per pixel.
    output_path
        Base path for output files.
    memory
        Optional :class:`~algan.utils.memory_utils.ManualMemory` render arena.
    render_settings
        Resolution / fps / quality settings (see
        :mod:`algan.settings.render_settings`).
    scene_initializer
        Callable run on (re)creation; the default spawns the camera and a
        point light.
    """

    def __init__(
        self,
        background_frame=STYLE_DEFAULTS.frame,
        output_path="output",
        memory=None,
        render_settings=RENDERING_DEFAULTS.settings,
        scene_initializer=lambda x: x,
    ):
        self.set_render_settings(render_settings)
        self.current_time = 0
        self.min_time = 0
        self.max_time = 0
        self.background_is_set = False
        if hasattr(background_frame, "__call__"):
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
        self.actors = [[]]
        self.effects = []
        self.scene_times = [[self.current_time, self.current_time]]
        self.background_depths = torch.full_like(
            self.background_frame[..., :1],
            dtype=torch.get_default_dtype(),
            fill_value=1e12,
        )
        self.animation_off = False
        self.output_path = output_path
        self.priority = 0
        self.id_count = 0
        self.scene_initializer = scene_initializer
        self.reset_scene()
        self.allow_new_actors = True
        self.animate_scene_clear = False

        self.memory = memory

    @staticmethod
    def wait(time=1):
        return AnimationManager.wait(time)

    @staticmethod
    def instance():
        return SceneManager.instance()

    @staticmethod
    def get_camera():
        return SceneManager.instance().camera

    @staticmethod
    def get_light_sources():
        return SceneManager.instance().light_sources

    @staticmethod
    def add_light_source(light_source):
        SceneManager.instance().light_sources.append(light_source)

    @staticmethod
    def set_environment_map(source, intensity=1.0, ambient=True):
        """Set an equirectangular environment map for the scene.

        The map is used as a skybox (rays that leave the scene show the map,
        including in reflections and refractions) and -- when ``ambient`` is
        True -- as diffuse image-based lighting: every lit surface receives
        the map's irradiance (an order-1 spherical-harmonics approximation)
        in addition to the scene's explicit lights.

        Supported by the deterministic (single-sample) ray tracer.

        Parameters
        ----------
        source
            Path to an image file, or a ``[H, W, >=3]`` tensor/array holding
            an equirectangular (longitude x latitude, sky at the top row)
            RGB image. Values may be 0-255 or 0-1. Pass ``None`` to remove
            the environment map.
        intensity
            Brightness multiplier applied to the map.
        ambient
            Whether the map also lights surfaces (image-based lighting), or
            is only visible as a background/in reflections.
        """
        scene = SceneManager.instance()
        if source is None:
            scene.environment_map = None
            return
        env = source
        if isinstance(env, str):
            import cv2

            img = cv2.imread(env, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Could not read environment map image: {env}")
            env = torch.from_numpy(img[..., ::-1].copy())  # BGR -> RGB
        if not torch.is_tensor(env):
            env = torch.tensor(env)
        env = env.float()
        if env.dim() != 3 or env.shape[-1] < 3:
            raise ValueError(
                "Environment map must have shape [height, width, >=3], got "
                f"{tuple(env.shape)}")
        if env.max() > 1.5:
            env = env / 255.0
        scene.environment_map = env[..., :3].contiguous()
        scene.environment_intensity = float(intensity)
        scene.environment_ambient = bool(ambient)

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

    @staticmethod
    def clear():
        SceneManager.instance().clear_scene()

    def despawn_scene(self, **kwargs):
        with Sync():
            for actor in list(
                sorted(self.actors[-1], key=lambda x: x.anchor_priority, reverse=True)
            ):
                if actor.is_spawned():
                    actor.despawn(**kwargs)

    def clear_scene(self, **kwargs):
        with Seq(run_time=0.5):
            self.despawn_scene(**kwargs)
        self.actors[-1] = [
            _ for _ in self.actors[-1] if (_.is_spawned() and _.is_despawned())
        ]

    def render_audio_to_file(self, file_path, frames_per_second=44100, codec='pcm_s32le', nbytes=4):
        if len(self.effects) == 0:
            return None

        clips_to_compose = []
        start_time = self.scene_times[-1][0] / self.render_settings.frames_per_second
        for audio_effect in self.effects:
            timed_clip = audio_effect.audio_clip.with_start(
                audio_effect.start_time_func() - start_time
            )
            clips_to_compose.append(timed_clip)

        audio_clip = CompositeAudioClip(clips_to_compose)
        audio_clip.duration = AnimationManager.instance().context.timespan.original_end
        audio_clip.write_audiofile(file_path, fps=frames_per_second, codec=codec, nbytes=nbytes)
        audio_clip.close()
        return file_path

    def reset_scene(self):
        self.actors = [[]]
        self.effects = []
        self.scene_initializer(self)

    def set_render_settings(self, render_settings):
        self.render_settings = render_settings
        self.num_pixels_screen_width, self.num_pixels_screen_height = (
            render_settings.resolution
        )
        self.frame_size = torch.tensor(
            (self.num_pixels_screen_height, self.num_pixels_screen_width)
        )
        self.frames_per_second = render_settings.frames_per_second
        self.num_pixels = self.frame_size.prod()
        self.size = self.num_pixels_screen_width, self.num_pixels_screen_height

    def background_is_transparent(self):
        if hasattr(self.background_frame, '__call__'):
            return False
        return (self.background_frame[..., -1].min() < (1-(0.5/255))).item()

    def get_pixel_format(self):
        return "rgba" if self.background_is_transparent() else "rgb"

    def show_frame(self, time_stamp=None):
        from algan.utils.plotting_utils import plot_tensor
        if time_stamp is None:
            time_stamp = AnimationManager.instance().context.current_time + 1.5/self.render_settings.frames_per_second
        time_ind = round(time_stamp * self.render_settings.frames_per_second)
        frames = []
        for frame in self.get_frames(time_ind-1, time_ind):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        for frame in frames:
            plot_tensor(frame)

        return frames

    def save_frame(self, filename, time_stamp=None):
        if not COMPUTING_DEFAULTS.allow_save_frame:
            return

        if time_stamp is None:
            time_stamp = AnimationManager.instance().context.timespan.current_time + 1.5/self.render_settings.frames_per_second
        time_ind = round(time_stamp * self.render_settings.frames_per_second)
        frames = []
        for frame in self.get_frames(time_ind-1, time_ind):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        torchvision.utils.save_image(frames[-1], filename)
        return frames

    def save_frames(self, filename, time_stamps=None):
        if not hasattr(time_stamps, '__len__'):
            time_stamps = [time_stamps]
        return [self.save_frame(f'{".".join(filename.split(".")[:-1])}_{t}.{filename.split(".")[-1]}',
                                t) for t in time_stamps]

    def set_background_color(self, background_color, overwrite=False):
        if self.background_is_set and not overwrite:
            return self
        if isinstance(background_color, str):
            a = self.render_settings.anti_alias_level
            background_color = F.interpolate(get_image(background_color).transpose(0,-1).unsqueeze(0), [_*a for _ in tuple(self.frame_size)],
                                             mode='bilinear', antialias='bilinear').squeeze(0).permute(1,2,0).unsqueeze(0)
        self.background_frame = self.background_color = background_color
        self.background_is_set = True
        return self

    def get_background_color(self):
        return self.background_color

    def get_new_id(self):
        self.id_count += 1
        return self.id_count - 1

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
