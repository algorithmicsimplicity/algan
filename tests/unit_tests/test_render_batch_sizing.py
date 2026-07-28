from types import SimpleNamespace

import torch

import algan.render_loop as render_loop_module
from algan.render_loop import (
    RenderLoopMixin,
    _max_duration_that_fits,
    _max_render_duration,
    _prepare_background_for_chunk,
    _raytrace_frame_buffers_end,
    _raytrace_persistent_input_end,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.scene_builder import _prefill_background


def test_fixed_render_memory_is_paid_once_not_per_frame():
    # 100 bytes/frame plus tile state that saturates at 400 bytes.
    def fixed(n):
        return min(n * 200, 400)

    assert _max_render_duration(900, 20, 100, fixed) == 5


def test_undersized_arena_preserves_single_frame_oom_path():
    assert _max_render_duration(50, 20, 100, lambda _n: 400) == 1


def test_animation_duration_search_uses_the_true_maximum():
    # Repeated halving returned 500 here even though every duration through 999
    # fits.  The animation-device budget should not discard that headroom.
    assert _max_duration_that_fits(1000, lambda n: n <= 999) == 999


def test_animation_duration_search_preserves_single_frame_failure_path():
    assert _max_duration_that_fits(1000, lambda _n: False) == 1


def test_raytrace_frame_buffers_include_mc_alignment_and_accumulator():
    # A transparent uint8 pixel ends at byte 6.  The five-float MC
    # accumulator aligns to byte 8 and ends at byte 28.
    assert _raytrace_frame_buffers_end(
        1, 1, 1, 1, 5, torch.uint8, samples=2
    ) == 28

    # Post-process-tonemap output is itself float32 and starts aligned.
    assert _raytrace_frame_buffers_end(
        1, 1, 1, 1, 5, torch.float32, samples=1
    ) == 24


def test_raytrace_persistent_inputs_cover_full_batch_camera_and_lights(
    monkeypatch,
):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 1)
    monkeypatch.setattr(rt_settings, "FRAGMENT_SHADING", True)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    num_frames = 3
    lights = [
        SimpleNamespace(
            origin=torch.zeros((num_frames, 1, 3)), _render_aux=None
        ),
        SimpleNamespace(
            origin=torch.zeros((num_frames, 1, 3)), _render_aux=None
        ),
    ]

    end = _raytrace_persistent_input_end(
        0,
        num_frames,
        lights,
        {},
        environment_map=None,
        environment_ambient=True,
    )

    # Five camera arrays: 13 f32/frame. Two compact lights each have a
    # three-float position and three-float color per frame.
    assert end == num_frames * (13 + 2 * 6) * 4


def test_monte_carlo_persistent_inputs_do_not_reserve_light_arrays(monkeypatch):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 4)
    num_frames = 3

    end = _raytrace_persistent_input_end(
        0,
        num_frames,
        [SimpleNamespace(
            origin=torch.zeros((num_frames, 1, 3)), _render_aux=None
        )],
        {},
        environment_map=None,
        environment_ambient=True,
    )

    assert end == num_frames * 13 * 4


def test_render_batch_sizes_postprocess_for_each_candidate(monkeypatch):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 1)
    monkeypatch.setattr(rt_settings, "FRAGMENT_SHADING", False)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    monkeypatch.setattr(rt_settings, "POST_PROCESS_TONEMAP", False)

    merged = {
        "num_frames": 8,
        "textures": torch.zeros((1, 1, 5)),
    }
    monkeypatch.setattr(
        "algan.rendering.raytracing.scene_builder._merge_scene",
        lambda _primitives: merged,
    )
    monkeypatch.setattr(
        "algan.rendering.raytracing.scene_builder.copy_merged_scene_to_arena",
        lambda scene, _memory, persist=True: scene,
    )
    monkeypatch.setattr(
        "algan.rendering.raytracing.tracer.get_wavefront_memory_required",
        lambda *_args, **_kwargs: 0,
    )

    sized_candidates = []

    def postprocess_size(**kwargs):
        num_frames = kwargs["frame_shape"][0]
        sized_candidates.append(num_frames)
        return 100 * num_frames * num_frames

    monkeypatch.setattr(
        render_loop_module, "_postprocess_memory_used", postprocess_size
    )

    rendered_durations = []

    class Primitive:
        _rt_projected = True
        memory = None

        @staticmethod
        def get_fixed_memory_used(_num_frames):
            return 0

        def render(self, *args, **_kwargs):
            duration = args[6] - args[5]
            rendered_durations.append(duration)
            return torch.zeros((duration, 1, 1, 3), dtype=torch.uint8)

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.camera = SimpleNamespace()
    scene.light_sources = []
    scene.num_pixels_screen_width = 1
    scene.num_pixels_screen_height = 1
    scene.video_settings = SimpleNamespace(anti_alias_level=1, fxaa=False)
    scene.memory = render_loop_module.ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=1000
    )
    scene.background_frame = torch.ones(4)

    render_state = {
        "ray_origin": torch.zeros((8, 1, 3)),
        "screen_point": torch.zeros((8, 1, 3)),
        "screen_basis": torch.eye(3).expand(8, -1, -1),
        "lights": [],
    }
    frames = list(scene.render_primitive_batch(
        [Primitive()], 0, 8, post_processes=(), render_state=render_state
    ))

    # 440 bytes of full-batch camera/light inputs + 4 bytes/frame of output +
    # 100*T**2 post bytes fits exactly two frames, but not three. Crucially the
    # nonlinear estimator was queried at binary-search candidates, not once at
    # T=1 and multiplied.
    assert rendered_durations == [2, 2, 2, 2]
    assert [len(frame) for frame in frames] == rendered_durations
    assert max(sized_candidates) > 1


def test_background_callback_streams_one_frame_at_a_time_on_render_device():
    callback_devices = []
    callback_batch_sizes = []

    def background(x, y, time):
        callback_devices.extend((x.device, y.device, time.device))
        callback_batch_sizes.append(time.shape[0])
        values = x + y + time
        return values.expand(-1, -1, -1, 4)

    deferred = _prepare_background_for_chunk(
        background,
        screen_width=3,
        screen_height=2,
        anti_alias_level=1,
        current_ind=0,
        new_ind=19,
        frames_per_second=4,
        device=torch.device("cpu"),
    )
    assert callback_devices == []

    callback_result = torch.empty((19, 6, 4), dtype=torch.uint8)
    _prefill_background(
        callback_result, deferred, frame_offset=0, device=torch.device("cpu")
    )

    x = torch.arange(3).view(1, -1, 1) / 3
    y = torch.arange(2).view(-1, 1, 1) / 2
    time = torch.arange(19).view(-1, 1, 1, 1) / 4
    raw = (x + y + time).expand(-1, -1, -1, 4)
    expected = raw.reshape(-1, 4)
    expected = torch.add(0.5, expected, alpha=255).clamp_(0, 255).to(
        torch.uint8)
    expected = expected.view(19, 6, 4)

    assert all(device.type == "cpu" for device in callback_devices)
    assert callback_batch_sizes == [1] * 19
    assert callback_result.device.type == "cpu"
    assert torch.equal(callback_result, expected)


def test_background_image_quantization_stays_on_requested_device():
    image_device = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.linspace(
        0.0, 1.0, 2 * 3 * 4, device=image_device
    ).view(1, 2, 3, 4)
    image_result = _prepare_background_for_chunk(
        image,
        screen_width=3,
        screen_height=2,
        anti_alias_level=1,
        current_ind=0,
        new_ind=2,
        frames_per_second=4,
        device=torch.device(image_device),
    )
    image_expected = image.expand(2, -1, -1, -1).contiguous().view(-1, 4)
    image_expected = torch.cat((image_expected[:1], image_expected))
    image_expected = ((image_expected + (0.5 / 255)) * 255).to(
        torch.uint8
    ).clamp_max_(255)

    assert image_result.device.type == image_device
    assert torch.equal(image_result, image_expected)


def test_failed_render_retry_resets_arena_with_full_gc(monkeypatch):
    cache_calls = []
    monkeypatch.setattr(
        render_loop_module,
        "empty_cache",
        lambda force_gc=False: cache_calls.append(force_gc),
    )

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    old_memory = render_loop_module.ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=128
    )
    scene.memory = old_memory
    old_memory.get_tensor((8,), dtype=torch.uint8)
    old_memory.get_tensor((16,), dtype=torch.uint8, persist=True)

    scene._reset_render_arena_after_failure()

    assert scene.memory is old_memory
    assert scene.memory.managed
    assert len(scene.memory) == 128
    assert scene.memory.get_pointers() == (0, 128)
    assert scene.memory.max_pointer == 0
    assert scene.memory.stack == []
    assert cache_calls == [True]


def test_unmanaged_render_bypasses_custom_postprocess_sizing(monkeypatch):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 1)
    monkeypatch.setattr(rt_settings, "FRAGMENT_SHADING", False)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    monkeypatch.setattr(rt_settings, "POST_PROCESS_TONEMAP", False)

    merged = {
        "num_frames": 4,
        "textures": torch.zeros((1, 1, 5)),
    }
    monkeypatch.setattr(
        "algan.rendering.raytracing.scene_builder._merge_scene",
        lambda _primitives: merged,
    )
    monkeypatch.setattr(
        "algan.rendering.raytracing.scene_builder.copy_merged_scene_to_arena",
        lambda scene, _memory, persist=True: scene,
    )
    monkeypatch.setattr(
        "algan.rendering.raytracing.tracer.get_wavefront_memory_required",
        lambda *_args, **_kwargs: 0,
    )

    def unexpected_estimator(**_kwargs):
        raise AssertionError("unmanaged rendering must not size post-processes")

    monkeypatch.setattr(
        render_loop_module, "_postprocess_memory_used", unexpected_estimator
    )
    rendered_durations = []

    class Primitive:
        _rt_projected = True
        memory = None

        def render(self, *args, **_kwargs):
            duration = args[6] - args[5]
            rendered_durations.append(duration)
            return torch.zeros((duration, 1, 1, 3), dtype=torch.uint8)

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.camera = SimpleNamespace()
    scene.light_sources = []
    scene.num_pixels_screen_width = 1
    scene.num_pixels_screen_height = 1
    scene.frames_per_second = 1
    scene.video_settings = SimpleNamespace(anti_alias_level=1, fxaa=False)
    scene.memory = render_loop_module.ManualMemory(
        0, device=torch.device("cpu"), managed=False
    )
    scene.background_frame = torch.ones(4)
    render_state = {
        "ray_origin": torch.zeros((4, 1, 3)),
        "screen_point": torch.zeros((4, 1, 3)),
        "screen_basis": torch.eye(3).expand(4, -1, -1),
        "lights": [],
    }

    def custom_postprocess(frames, memory):
        return frames

    frames = list(scene.render_primitive_batch(
        [Primitive()], 0, 4,
        post_processes=(custom_postprocess,), render_state=render_state,
    ))

    assert rendered_durations == [4]
    assert [len(frame) for frame in frames] == [4]


def test_outer_preflight_retry_renders_first_fitting_halved_duration(
    monkeypatch,
):
    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", "0")
    monkeypatch.setattr(render_loop_module, "_sync_devices", lambda: None)
    monkeypatch.setattr(
        render_loop_module, "empty_cache", lambda force_gc=False: None
    )
    monkeypatch.setattr(
        render_loop_module, "get_num_available_bytes", lambda _device: 1000
    )

    class NullOff:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(render_loop_module, "Off", NullOff)

    rendered_durations = []

    class Primitive:
        _rt_device_scene = None
        _rt_prepared_host_scene = None

    class Scene(RenderLoopMixin):
        def background_is_transparent(self):
            return False

        def get_batch_of_primitives(
            self, start_ind, end_ind, _actors, _max_memory
        ):
            primitive = Primitive()
            primitive.duration = end_ind - start_ind
            return [primitive], end_ind, {"lights": []}

        def _prewarm_render_batch(self, _primitives, _render_state):
            pass

        def _prepared_batch_fits_render_arena(
            self, primitives, *_args, **_kwargs
        ):
            return primitives[0].duration <= 5

        def render_primitive_batch(
            self, _primitives, start_ind, end_ind, *_args, **_kwargs
        ):
            duration = end_ind - start_ind
            rendered_durations.append(duration)
            yield torch.zeros((duration, 1, 1, 3), dtype=torch.uint8)

    scene = Scene.__new__(Scene)
    scene.background_frame = torch.ones(4)
    scene.memory = None
    scene.light_sources = []
    scene.camera = SimpleNamespace(screen=SimpleNamespace())
    scene.timeline_manager = SimpleNamespace(clear_buffers=lambda: None)
    scene.animation_manager = SimpleNamespace()
    scene.actors = [[]]
    scene.frames_per_second = 1

    frames = list(scene.get_frames(
        0, 8, post_processes=(), manual_memory=False
    ))

    # Eight frames fail preflight, so the retry renders the fitting half
    # immediately. The remaining four frames form the next batch.
    assert rendered_durations == [4, 4]
    assert [len(frame) for frame in frames] == rendered_durations
