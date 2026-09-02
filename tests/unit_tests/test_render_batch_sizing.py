from types import SimpleNamespace

import pytest
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
    expected = torch.add(0.5, expected, alpha=255).clamp_(0, 255).to(torch.uint8)
    expected = expected.view(19, 6, 4)

    assert all(device.type == "cpu" for device in callback_devices)
    assert callback_batch_sizes == [1] * 19
    assert callback_result.device.type == "cpu"
    assert torch.equal(callback_result, expected)


def test_background_image_quantization_stays_on_requested_device():
    image_device = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.linspace(0.0, 1.0, 2 * 3 * 4, device=image_device).view(1, 2, 3, 4)
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
    image_expected = (
        ((image_expected + (0.5 / 255)) * 255).to(torch.uint8).clamp_max_(255)
    )

    assert image_result.device.type == image_device
    assert torch.equal(image_result, image_expected)


def test_failed_render_retry_resets_arena_with_full_gc(monkeypatch):
    cache_calls = []
    monkeypatch.setattr(
        render_loop_module,
        "release_torch_memory",
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


def test_chunk_peak_is_credited_to_the_window_that_actually_rendered():
    # A chunk the tracer had to sub-divide (out of memory, or the Monte Carlo
    # path budget) peaks at its largest sub-window, not at the count that was
    # planned. Crediting the planned count halves the apparent per-frame cost,
    # so the model plans the same over-large chunk again -- and again, because
    # the split is recovered inside the tracer and never reaches the loop's
    # own render-failure path.
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.memory = SimpleNamespace(last_launch_frames=None)

    assert scene._observed_chunk_frames(64) == 64

    scene.memory.last_launch_frames = 16
    assert scene._observed_chunk_frames(64) == 16

    # A chunk that rendered whole (or in windows no smaller than it) is its own
    # measurement; nothing may inflate the frame count a peak is charged to.
    scene.memory.last_launch_frames = 64
    assert scene._observed_chunk_frames(64) == 64
    scene.memory.last_launch_frames = 0
    assert scene._observed_chunk_frames(64) == 64


def test_probe_duration_lets_a_capacity_estimate_shortcut_the_halving():
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)

    # No measurement: plain binary search over [low, high].
    scene._begin_batch_cost_measurement()
    assert scene._next_probe_duration(1, 99) == 50

    def capacity(frames):
        # One observation reads as a pure per-frame cost, so the budget picks
        # the capacity directly.
        scene._begin_batch_cost_measurement()
        scene._note_batch_cost("term", 10, 100, 10 * frames)

    # An estimate below the halving point is taken: one probe instead of
    # several.
    capacity(12)
    assert scene._next_probe_duration(1, 99) == 12

    # An estimate at or above it is not: the measured batch overflowed, so the
    # estimate overshoots by the batch's frame-independent bytes, and the
    # search must keep its guaranteed halving progress.
    capacity(90)
    assert scene._next_probe_duration(1, 99) == 50

    # Always inside the bracket: an estimate below a duration already known to
    # fit is clamped up to it.
    capacity(3)
    assert scene._next_probe_duration(20, 99) == 20

    # ...and a caller that says the measurement no longer describes what it is
    # about to build gets the plain halving back.
    assert scene._next_probe_duration(1, 99, use_hint=False) == 50


def test_slicing_across_spawns_is_on_with_a_kill_switch(monkeypatch):
    # A prefix of a fetched window can carry mobs that have not spawned by the
    # time it ends. They are never drawn (opacity is zeroed outside a lifespan
    # and the primitive gets empty per-frame bounds), but carrying them
    # reorders the merged arrays and the STBVH, so edge tie-breaks land
    # differently -- correct, not byte-identical. Refusing to slice instead
    # costs a full rematerialization per batch.
    monkeypatch.delenv("ALGAN_SLICE_ACROSS_SPAWNS", raising=False)
    assert RenderLoopMixin._may_slice_across_spawns()
    monkeypatch.setenv("ALGAN_SLICE_ACROSS_SPAWNS", "0")
    assert not RenderLoopMixin._may_slice_across_spawns()
    monkeypatch.setenv("ALGAN_SLICE_ACROSS_SPAWNS", "1")
    assert RenderLoopMixin._may_slice_across_spawns()


def test_actor_share_and_frame_capacity_split_a_batch_cost():
    # A batch's cost is a + b*frames: what the frame count buys, and what its
    # actor set costs regardless. Read as if it all scaled, the two are
    # indistinguishable -- and only the first is fixable by shortening the
    # window, which is how a render ended up rendering single frames.
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene._begin_batch_cost_measurement()
    assert scene._batch_frame_capacity() is None
    assert scene._batch_actor_share() is None

    # 900 MB of actor set + 1 MB/frame, against a 1000 MB budget.
    scene._note_batch_cost("arena", 10, 910_000_000, 1_000_000_000)
    scene._note_batch_cost("arena", 20, 920_000_000, 1_000_000_000)
    assert scene._batch_frame_capacity() == 100
    assert scene._batch_actor_share() == pytest.approx(900 / 920, rel=1e-6)

    # The two questions are answered by different terms, and each takes the
    # term that binds it: the tightest frame limit (here a term whose fixed
    # part alone overruns its budget, so no window is short enough), and the
    # most actor-bound cost.
    scene._note_batch_cost("merge", 10, 1_100_000_000, 1_000_000_000)
    scene._note_batch_cost("merge", 20, 1_200_000_000, 1_000_000_000)
    assert scene._batch_frame_capacity() == 0
    assert scene._batch_actor_share() == pytest.approx(900 / 920, rel=1e-6)


def test_spawn_boundary_is_where_shortening_drops_an_actor():
    # Batch prep selects actors that have spawned by the window's end, so only
    # a window that reaches back past a spawn carries fewer of them. Inside a
    # stretch with no spawn, shortening drops nothing -- which is why a frame
    # search cannot relieve an actor-bound batch.
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.frames_per_second = 10  # frames 10:30 -> seconds 1.0:3.0

    def actor(spawn_time):
        return SimpleNamespace(
            lifespan=SimpleNamespace(start=lambda: spawn_time),
            get_render_primitives=lambda: None,
        )

    # Spawns at frames 15 and 22; retreat behind the later one.
    boundary = scene._previous_spawn_boundary(
        [actor(0.5), actor(1.5), actor(2.2)], 10, 30
    )
    assert boundary == 21
    assert boundary / scene.frames_per_second < 2.2

    # No spawn inside the window: nothing to retreat behind.
    assert scene._previous_spawn_boundary([actor(0.5), actor(9.0)], 10, 30) is None
    # A spawn at the window's own start is already admitted.
    assert scene._previous_spawn_boundary([actor(1.0)], 10, 30) is None


def test_stable_actor_set_sees_only_spawns_inside_the_window():
    # The kill switch's predicate: only a spawn strictly inside the window
    # makes a prefix differ from a freshly fetched one.
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.frames_per_second = 10  # frames 10:20 -> seconds 1.0:2.0

    def actor(spawn_time):
        return SimpleNamespace(
            lifespan=SimpleNamespace(start=lambda: spawn_time),
            get_render_primitives=lambda: None,
        )

    assert scene._fetched_window_has_stable_actor_set(
        [actor(0.0), actor(1.0), actor(5.0)], 10, 20
    )
    assert not scene._fetched_window_has_stable_actor_set([actor(1.5)], 10, 20)
    # Actors with no geometry are irrelevant either way.
    assert scene._fetched_window_has_stable_actor_set(
        [SimpleNamespace(lifespan=SimpleNamespace(start=lambda: 1.5))], 10, 20
    )


def _make_preflight_scene(
    monkeypatch, preflight, rendered_durations, requested_windows=None
):
    """A Scene whose batching loop is driven by ``preflight``."""
    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", "0")
    monkeypatch.setattr(render_loop_module, "_sync_devices", lambda: None)
    monkeypatch.setattr(
        render_loop_module, "release_torch_memory", lambda force_gc=False: None
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

    class Primitive:
        _rt_device_scene = None
        _rt_prepared_host_scene = None
        _rt_merged_scene = None

    class Scene(RenderLoopMixin):
        def background_is_transparent(self):
            return False

        def _get_batch_of_primitives(self, start_ind, end_ind, _actors, _max_memory):
            if requested_windows is not None:
                requested_windows.append(end_ind - start_ind)
            primitive = Primitive()
            primitive.duration = end_ind - start_ind
            return [primitive], end_ind, {"lights": []}

        def _prewarm_render_batch(self, _primitives, _render_state):
            pass

        def _prepared_batch_fits_render_arena(
            self, primitives, *_args, require_estimates_fit=True, **_kwargs
        ):
            return preflight(primitives[0].duration, require_estimates_fit)

        def _render_primitive_batch(
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
    return scene


def test_single_frame_batch_renders_when_only_an_estimate_rejects_it(monkeypatch):
    # The preflight's frame cost is modelled, and a contaminated model can put
    # it above the whole arena. Once the window is down to one frame there is
    # nothing smaller to retreat to, so acting on that estimate aborts the
    # render -- which is how a scene that renders fine ended up raising
    # OutOfRenderMemory mid-job. Estimates lose their vote at one frame; the
    # render's own out-of-memory retry stays the backstop.
    rendered_durations = []
    scene = _make_preflight_scene(
        monkeypatch,
        lambda _duration, require_estimates_fit: not require_estimates_fit,
        rendered_durations,
    )

    frames = list(scene.get_frames(0, 3, post_processes=(), manual_memory=False))

    assert rendered_durations == [1, 1, 1]
    assert [len(frame) for frame in frames] == rendered_durations


def test_single_frame_batch_that_exactly_overflows_still_reports_out_of_memory(
    monkeypatch,
):
    # The exact terms (the scene's own arena bytes, an actual OOM from the
    # merge) keep their vote: a scene that genuinely cannot be rendered must
    # still say so rather than looping.
    rendered_durations = []
    scene = _make_preflight_scene(
        monkeypatch,
        lambda _duration, _require_estimates_fit: False,
        rendered_durations,
    )

    with pytest.raises(render_loop_module.OutOfRenderMemory):
        list(scene.get_frames(0, 3, post_processes=(), manual_memory=False))
    assert rendered_durations == []


def test_fetch_window_carries_the_arena_verdict_into_the_next_batch(monkeypatch):
    # Materializing a batch is the expensive half of preparing one, and the
    # animation-device budget it is sized by has no idea what the render arena
    # can hold. Nothing used to carry the arena's answer from one batch to the
    # next, so *every* batch of a long render fetched a window the arena
    # rejected and then rematerialized a smaller one -- the same search, over
    # and over, for the whole job.
    rendered_durations = []
    requested_windows = []
    scene = _make_preflight_scene(
        monkeypatch,
        lambda duration, _require_estimates_fit: duration <= 5,
        rendered_durations,
        requested_windows,
    )

    frames = list(scene.get_frames(0, 20, post_processes=(), manual_memory=False))

    assert rendered_durations == [5, 5, 5, 5]
    assert [len(frame) for frame in frames] == rendered_durations
    # Only the first batch pays for the search; every later fetch asks for the
    # window that was just shown to fit.
    assert requested_windows == [20, 10, 5, 5, 5, 5]


def test_outer_preflight_retry_renders_first_fitting_halved_duration(
    monkeypatch,
):
    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", "0")
    monkeypatch.setattr(render_loop_module, "_sync_devices", lambda: None)
    monkeypatch.setattr(
        render_loop_module, "release_torch_memory", lambda force_gc=False: None
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

        def _get_batch_of_primitives(self, start_ind, end_ind, _actors, _max_memory):
            primitive = Primitive()
            primitive.duration = end_ind - start_ind
            return [primitive], end_ind, {"lights": []}

        def _prewarm_render_batch(self, _primitives, _render_state):
            pass

        def _prepared_batch_fits_render_arena(self, primitives, *_args, **_kwargs):
            return primitives[0].duration <= 5

        def _render_primitive_batch(
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

    frames = list(scene.get_frames(0, 8, post_processes=(), manual_memory=False))

    # Eight frames fail preflight, so the retry renders the fitting half
    # immediately. The remaining four frames form the next batch.
    assert rendered_durations == [4, 4]
    assert [len(frame) for frame in frames] == rendered_durations
