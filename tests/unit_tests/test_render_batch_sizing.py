from types import SimpleNamespace

import torch

import algan.render_loop as render_loop_module
from algan.render_loop import (
    RenderLoopMixin,
    _max_duration_that_fits,
    _prepare_background_for_chunk,
)
from algan.rendering.raytracing.scene_builder import _prefill_background


def test_animation_duration_search_uses_the_true_maximum():
    # Repeated halving returned 500 here even though every duration through 999
    # fits.  The animation-device budget should not discard that headroom.
    assert _max_duration_that_fits(1000, lambda n: n <= 999) == 999


def test_animation_duration_search_preserves_single_frame_failure_path():
    assert _max_duration_that_fits(1000, lambda _n: False) == 1


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
