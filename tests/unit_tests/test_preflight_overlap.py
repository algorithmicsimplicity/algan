"""The prefetch-worker overlap of the arena preflight.

These drive the scheduling logic directly -- fake predictor warmth, a fake
worker, forced rejection -- because the overlapped builds themselves are GPU
builds: ``project_on_gpu_active`` / ``merge_on_gpu_active`` hard-require a CUDA
render device (see ``raytracing.settings``), so no CPU render can reach
``_prepare_batch_on_worker`` through ``_get_frames_impl``. The end-to-end tests
here patch exactly that device gate and nothing else, which lets a CPU box
exercise the handover, the preflight's skip branches and the fallback-to-
serial path against the real batching loop.
"""

import copy
import threading
from types import SimpleNamespace

import pytest
import torch

import algan.render_loop as render_loop_module
import algan.rendering.raytracing.settings as rt_module
from algan.errors import AlganConfigurationError, AlganWarning
from algan.render_loop import RenderLoopMixin
from algan.rendering.memory_model import PeakRatioModel, memory_model_history
from algan.settings import SETTINGS


@pytest.fixture
def restore_computing_settings():
    saved = {
        name: getattr(SETTINGS.computing, name)
        for name in ("prefetch_gpu_prep", "overlap_pool_headroom_fraction")
    }
    yield
    for name, value in saved.items():
        SETTINGS.computing.set(**{name: value})


def test_overlap_settings_default_off_with_derate_and_validation():
    from algan.settings.computing_settings import ComputingSettings

    defaults = ComputingSettings()
    assert defaults.prefetch_gpu_prep is False
    assert defaults.overlap_pool_headroom_fraction == pytest.approx(0.6)

    clamped = ComputingSettings(overlap_pool_headroom_fraction=1)
    assert clamped.overlap_pool_headroom_fraction == pytest.approx(1.0)

    with pytest.raises(AlganConfigurationError):
        ComputingSettings(prefetch_gpu_prep="yes")
    with pytest.raises(AlganConfigurationError):
        ComputingSettings(overlap_pool_headroom_fraction=0.0)
    with pytest.raises(AlganConfigurationError):
        ComputingSettings(overlap_pool_headroom_fraction=1.5)


def test_overlap_env_names_are_declared():
    # The registry must know both names, or `import algan` warns about a
    # variable the package itself reads.
    from algan.environment import ALGAN_ENVIRONMENT_VARIABLES

    assert "ALGAN_PREFETCH_GPU_PREP" in ALGAN_ENVIRONMENT_VARIABLES
    assert "ALGAN_OVERLAP_HEADROOM_FRACTION" in ALGAN_ENVIRONMENT_VARIABLES


def test_overlap_headroom_fraction_reads_live_and_guards_range(monkeypatch):
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    monkeypatch.delenv("ALGAN_OVERLAP_HEADROOM_FRACTION", raising=False)

    assert scene._overlap_headroom_fraction() == pytest.approx(
        SETTINGS.computing.overlap_pool_headroom_fraction
    )

    monkeypatch.setenv("ALGAN_OVERLAP_HEADROOM_FRACTION", "0.75")
    assert scene._overlap_headroom_fraction() == pytest.approx(0.75)

    # An out-of-range override must not silently drop the derate.
    monkeypatch.setenv("ALGAN_OVERLAP_HEADROOM_FRACTION", "4")
    with pytest.warns(AlganWarning):
        assert scene._overlap_headroom_fraction() == pytest.approx(
            SETTINGS.computing.overlap_pool_headroom_fraction
        )


def test_overlap_gate_requires_setting_cuda_builds_and_worker_thread(monkeypatch):
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)

    def active_in_thread():
        seen = []
        thread = threading.Thread(
            target=lambda: seen.append(scene._overlap_gpu_prep_active()),
            name="algan-batch-prep_0",
        )
        thread.start()
        thread.join()
        return seen[0]

    # Setting off (the default): closed everywhere, whatever the builds say.
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    assert scene._overlap_gpu_prep_active() is False
    assert active_in_thread() is False

    # Setting on, but the calling thread is the render thread (a synchronous /
    # retry fetch): no render to hide behind, so no overlap.
    SETTINGS.computing.set(prefetch_gpu_prep=True)
    try:
        assert scene._overlap_gpu_prep_active() is False
        assert active_in_thread() is True

        # Both GPU builds must be active: on this CPU box they are not, so the
        # gate stays closed even on the worker.
        monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: False)
        assert active_in_thread() is False
        monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
        monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: False)
        assert active_in_thread() is False
    finally:
        SETTINGS.computing.set(prefetch_gpu_prep=False)


class _FakePrimitive:
    def __init__(self):
        self._rt_projected = False
        self._rt_prep_overlapped = False
        self._rt_merged_scene = None
        self._rt_prepared_host_scene = None


@pytest.fixture
def overlap_scene(monkeypatch):
    """A Scene stub whose GPU-build seams record instead of build."""
    from algan.rendering.raytracing import primitives as rt_primitives

    monkeypatch.setattr(rt_primitives, "RayTracedTrianglePrimitive", _FakePrimitive)
    monkeypatch.setattr(
        rt_primitives, "RayTracedBezierCircuitPrimitive", _FakePrimitive
    )

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.builds = []
    scene.memory = SimpleNamespace(device=torch.device("cpu"), data=None)

    def _prewarm_render_batch(primitives, _render_state):
        scene.builds.append("project")
        for primitive in primitives:
            primitive._rt_projected = True

    def _prepare_merged_host_scene(
        primitive_batch, *, render_state=None, track_peak=None
    ):
        scene.builds.append(("merge", track_peak))
        return {"num_triangles": 0}, None

    scene._prewarm_render_batch = _prewarm_render_batch
    scene._prepare_merged_host_scene = _prepare_merged_host_scene
    monkeypatch.setattr(
        Scene, "_gpu_merge_headroom_bytes", lambda self: 1_000_000, raising=False
    )

    # Real models, really calibrated: one observation each is enough for
    # is_calibrated(), which is the gate the first batch of a job fails.
    scene._project_peak_ratio = PeakRatioModel(8.0)
    scene._merge_peak_ratio = PeakRatioModel(6.0)
    return scene


def _run_worker(fn):
    error = []

    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=run, name="algan-batch-prep_0")
    thread.start()
    thread.join()
    if error:
        raise error[0]


def test_prepare_batch_on_worker_needs_calibrated_predictors(
    overlap_scene, monkeypatch
):
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True, raising=False)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )

    # The first batch(es) of a job stay on the render thread exactly as
    # today: no build ran, nothing was stamped.
    assert overlap_scene.builds == []
    assert primitives[0]._rt_prep_overlapped is False
    assert primitives[0]._rt_projected is False


def test_prepare_batch_on_worker_runs_both_builds_and_stamps(
    overlap_scene, monkeypatch
):
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    overlap_scene._project_peak_ratio.observe(10, 20)
    overlap_scene._merge_peak_ratio.observe(10, 20)

    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )

    assert overlap_scene.builds[0] == "project"
    assert overlap_scene.builds[1] == ("merge", False)
    assert all(p._rt_projected for p in primitives)
    assert primitives[0]._rt_prep_overlapped is True


def test_prepare_batch_on_worker_derates_headroom_for_the_render(
    overlap_scene, monkeypatch
):
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    overlap_scene._project_peak_ratio.observe(10, 20)
    overlap_scene._merge_peak_ratio.observe(10, 20)

    predictions = {}
    monkeypatch.setattr(
        PeakRatioModel,
        "predict",
        lambda self, inputs: predictions.setdefault(
            "project" if self is overlap_scene._project_peak_ratio else "merge", 0
        ),
    )

    # Headroom 1_000_000 derated by the default 0.6 -> 600_000. Just above it
    # the projection declines and leaves the whole batch to the render thread.
    predictions["project"] = 600_001
    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )
    assert overlap_scene.builds == []
    assert primitives[0]._rt_prep_overlapped is False

    # At (or below) the derated headroom the projection runs.
    overlap_scene.builds.clear()
    predictions["project"] = 600_000
    predictions["merge"] = 600_001
    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )
    assert overlap_scene.builds == ["project"]
    assert primitives[0]._rt_prep_overlapped is False

    # The merge gets the same derate after projecting cleanly.
    overlap_scene.builds.clear()
    predictions["merge"] = 600_000
    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )
    assert overlap_scene.builds == ["project", ("merge", False)]
    assert primitives[0]._rt_prep_overlapped is True


@pytest.mark.parametrize(
    "exc",
    [render_loop_module.InsufficientMemoryException(), RuntimeError("out of memory")],
)
def test_prepare_batch_on_worker_merge_oom_defers_to_render_thread(
    overlap_scene, monkeypatch, exc
):
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    overlap_scene._project_peak_ratio.observe(10, 20)
    overlap_scene._merge_peak_ratio.observe(10, 20)

    cache_calls = []
    monkeypatch.setattr(
        render_loop_module,
        "release_torch_memory",
        lambda force_gc=False: cache_calls.append(True),
    )

    def failing_merge(primitive_batch, *, render_state=None, track_peak=None):
        raise exc

    overlap_scene._prepare_merged_host_scene = failing_merge

    primitives = [_FakePrimitive()]
    _run_worker(
        lambda: overlap_scene._prepare_batch_on_worker(primitives, {"lights": []})
    )

    # Partial merge state dropped, projection kept, nothing stamped: the
    # render thread re-runs the merge under its own full-headroom estimates.
    assert overlap_scene.builds == ["project"]
    assert primitives[0]._rt_merged_scene is None
    assert primitives[0]._rt_prepared_host_scene is None
    assert primitives[0]._rt_prep_overlapped is False
    assert primitives[0]._rt_projected is True
    assert cache_calls == [True]


def test_prepare_batch_on_worker_reraises_real_merge_errors(overlap_scene, monkeypatch):
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    overlap_scene._project_peak_ratio.observe(10, 20)
    overlap_scene._merge_peak_ratio.observe(10, 20)

    def broken_merge(primitive_batch, *, render_state=None, track_peak=None):
        raise ValueError("not a memory failure")

    overlap_scene._prepare_merged_host_scene = broken_merge

    with pytest.raises(ValueError):
        _run_worker(
            lambda: overlap_scene._prepare_batch_on_worker(
                [_FakePrimitive()], {"lights": []}
            )
        )


class _PreflightPrimitive:
    """Just enough tensor surface for the input-byte walks."""

    def __init__(self, projected):
        self.corners = torch.zeros(3)
        self._rt_tri_pos = torch.zeros(4)
        self._rt_projected = projected
        self._rt_prep_overlapped = False
        self._rt_merged_scene = None
        self._rt_prepared_host_scene = None
        self._rt_device_scene = None


def _make_preflight_driver(monkeypatch, overlapped):
    """A Scene whose real preflight runs against faked seams."""
    from algan.rendering.raytracing import primitives as rt_primitives

    monkeypatch.setattr(
        rt_primitives, "RayTracedTrianglePrimitive", _PreflightPrimitive
    )
    monkeypatch.setattr(
        rt_primitives, "RayTracedBezierCircuitPrimitive", _PreflightPrimitive
    )
    # Pretend both GPU builds route is active so their branches execute; the
    # peak token pair is faked too, since this box has no CUDA counters.
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: True)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)
    monkeypatch.setattr(render_loop_module, "begin_cuda_peak", lambda _device: object())
    monkeypatch.setattr(render_loop_module, "end_cuda_peak", lambda _token: 12345)

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.prewarm_calls = []
    scene.memory = render_loop_module.ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=1 << 16
    )
    scene.video_settings = SimpleNamespace(supersampling=1)
    scene.num_pixels_screen_width = 4
    scene.num_pixels_screen_height = 4
    scene.camera = SimpleNamespace(near=0.0, far=100.0)
    monkeypatch.setattr(Scene, "_gpu_merge_headroom_bytes", lambda self: 1 << 30)

    def _prewarm_render_batch(primitives, _render_state):
        scene.prewarm_calls.append(len(primitives))
        # Like the real one: successfully projected primitives are marked so
        # the preflight's all-projected gate lets the merge proceed.
        for primitive in primitives:
            primitive._rt_projected = True

    def _prepare_merged_host_scene(
        primitive_batch, *, render_state=None, track_peak=None
    ):
        # A measured merge peak rides the dict; whether anybody reads it is
        # what the assertions below decide. The tensor gives the exact arena
        # accounting something nonzero to weigh.
        return {
            "num_triangles": 0,
            "num_circuits": 0,
            "_gpu_merge_peak_bytes": 777,
            "tri_pos": torch.zeros(16),
        }, None

    scene._prewarm_render_batch = _prewarm_render_batch
    scene._prepare_merged_host_scene = _prepare_merged_host_scene
    scene._chunk_memory_model = render_loop_module.ChunkMemoryModel()
    scene._begin_batch_cost_measurement()
    scene._project_peak_ratio = PeakRatioModel(8.0)
    scene._merge_peak_ratio = PeakRatioModel(6.0)
    scene._arena_unmodeled_bytes = 0
    scene._last_arena_preflight = None

    observations = {"project": [], "merge": []}
    real_observe = PeakRatioModel.observe

    def recording_observe(self, input_bytes, peak_bytes):
        key = "project" if self is scene._project_peak_ratio else "merge"
        observations[key].append((input_bytes, peak_bytes))
        return real_observe(self, input_bytes, peak_bytes)

    monkeypatch.setattr(PeakRatioModel, "observe", recording_observe)
    return scene, observations


def test_preflight_of_unoverlapped_batch_measures_and_estimates_as_today(
    monkeypatch,
):
    scene, observations = _make_preflight_driver(monkeypatch, overlapped=False)
    primitives = [_PreflightPrimitive(projected=False)]

    fits = scene._prepared_batch_fits_render_arena(
        primitives, {"lights": []}, (), False, require_estimates_fit=True, num_frames=3
    )

    # Today's path, unchanged: the prewarm ran, both builds' peaks were fed
    # to their predictors, and every cost term was noted.
    assert fits is True
    assert scene.prewarm_calls == [1]
    assert len(observations["project"]) == 1
    assert len(observations["merge"]) == 1
    assert set(scene._batch_costs) == {"projection", "merge", "arena"}
    assert scene._last_arena_preflight is not None


def test_preflight_of_overlapped_batch_skips_prewarm_peaks_and_estimates(
    monkeypatch,
):
    scene, observations = _make_preflight_driver(monkeypatch, overlapped=True)
    primitives = [_PreflightPrimitive(projected=True) for _ in range(2)]
    primitives[0]._rt_prep_overlapped = True

    fits = scene._prepared_batch_fits_render_arena(
        primitives, {"lights": []}, (), False, require_estimates_fit=True, num_frames=3
    )

    # Nothing was measured beside a (hypothetical) render and no moot
    # estimate was consulted -- but the exact arena accounting still ran and
    # still decides.
    assert fits is True
    assert scene.prewarm_calls == []
    assert observations["project"] == []
    assert observations["merge"] == []
    assert set(scene._batch_costs) == {"arena"}
    assert scene._last_arena_preflight is not None


def _release_like_projection(primitive):
    """What project_to_screen leaves behind: source tensors dropped, shader
    parameter values nulled but their names kept, one stray non-``_rt_``
    tensor still attached.
    """
    primitive.corners = None
    primitive.shader_param_names = ["emissive"]
    primitive.shader_param_values = None
    primitive.texture_opacity = torch.zeros(5)
    primitive._rt_projected = True
    return primitive


def test_project_input_bytes_count_projected_primitives_as_zero():
    from algan.rendering.raytracing.scene_builder import gpu_project_input_bytes

    pending = _PreflightPrimitive(projected=False)
    pending.shader_param_names = ["emissive"]
    pending.shader_param_values = [torch.zeros(2)]
    released = _release_like_projection(_PreflightPrimitive(projected=False))

    # corners (3 x f32) + the one shader parameter (2 x f32).
    assert gpu_project_input_bytes([pending]) == 20
    # Used to raise TypeError zipping names against the released None.
    assert gpu_project_input_bytes([released]) == 0
    assert gpu_project_input_bytes([released, pending]) == 20

    # Released values with no projection mark are tolerated too.
    released._rt_projected = False
    assert gpu_project_input_bytes([released]) == 20


def test_preflight_of_worker_projected_unmerged_batch_builds_only_the_merge(
    monkeypatch,
):
    # The worker projected the whole batch, then declined the merge against
    # its derated headroom, so the batch arrives projected but not stamped
    # overlapped. The preflight used to re-sum the released source geometry
    # and crash at HD; it must merge on the render thread and feed the merge
    # predictor, with no projection left to estimate or measure.
    scene, observations = _make_preflight_driver(monkeypatch, overlapped=False)
    primitives = [
        _release_like_projection(_PreflightPrimitive(projected=True)) for _ in range(2)
    ]

    fits = scene._prepared_batch_fits_render_arena(
        primitives, {"lights": []}, (), False, require_estimates_fit=True, num_frames=3
    )

    assert fits is True
    assert observations["project"] == []
    assert observations["merge"] == [(2 * 16, 777)]
    assert set(scene._batch_costs) == {"merge", "arena"}


def test_preflight_of_partly_projected_batch_sizes_only_the_remainder(monkeypatch):
    # A worker projection that failed part-way leaves the projected prefix
    # marked; the render thread projects the rest and measures only that.
    scene, observations = _make_preflight_driver(monkeypatch, overlapped=False)
    primitives = [
        _release_like_projection(_PreflightPrimitive(projected=True)),
        _PreflightPrimitive(projected=False),
    ]

    fits = scene._prepared_batch_fits_render_arena(
        primitives, {"lights": []}, (), False, require_estimates_fit=True, num_frames=3
    )

    assert fits is True
    assert observations["project"] == [(12, 12345)]
    assert len(observations["merge"]) == 1


def _make_loop_scene(monkeypatch, *, overlap_enabled):
    """Drive the real batching loop with prefetching and a faked overlap."""
    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", "1")
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

        def __init__(self):
            self._rt_projected = False
            self._rt_prep_overlapped = False

    records = {
        "worker_threads": [],
        "overlapped_flags": [],
        "durations": [],
        "rejected_overlapped": 0,
        "calibrated": False,
    }

    class Scene(RenderLoopMixin):
        def background_is_transparent(self):
            return False

        def _get_batch_of_primitives(self, start_ind, end_ind, _actors, _max_memory):
            primitive = Primitive()
            duration = min(2, end_ind - start_ind)
            primitive.duration = duration
            return [primitive], start_ind + duration, {"lights": []}

        def _overlap_gpu_prep_active(self):
            # The real method minus the CUDA requirement, which is what keeps
            # this CPU box from reaching the overlap through a real render.
            if not overlap_enabled:
                return False
            return threading.current_thread().name.startswith("algan-batch-prep")

        def _prepare_batch_on_worker(self, primitive_batch, _render_state):
            records["worker_threads"].append(threading.current_thread().name)
            if not records["calibrated"]:
                # Stand-in for the predictor-calibration gate: the first
                # batch of a job prepares on the render thread.
                return
            for primitive in primitive_batch:
                primitive._rt_projected = True
            primitive_batch[0]._rt_prep_overlapped = True

        def _prewarm_render_batch(self, _primitives, _render_state):
            pass

        def _prepared_batch_fits_render_arena(
            self, primitives, *_args, require_estimates_fit=True, **_kwargs
        ):
            overlapped = bool(getattr(primitives[0], "_rt_prep_overlapped", False))
            records["overlapped_flags"].append(overlapped)
            if overlapped and reject_first_overlapped[0]:
                reject_first_overlapped[0] = False
                records["rejected_overlapped"] += 1
                return False
            if not overlapped:
                # First batch measured on the render thread -> predictors warm.
                records["calibrated"] = True
            return True

        def _render_primitive_batch(
            self, _primitives, start_ind, end_ind, *_args, **_kwargs
        ):
            duration = end_ind - start_ind
            records["durations"].append(duration)
            yield torch.zeros((duration, 2, 2, 3), dtype=torch.uint8)

    reject_first_overlapped = [False]
    scene = Scene.__new__(Scene)
    scene.background_frame = torch.ones(4)
    scene.memory = None
    scene.light_sources = []
    scene.camera = SimpleNamespace(screen=SimpleNamespace())
    scene.timeline_manager = SimpleNamespace(clear_buffers=lambda: None)
    scene.animation_manager = SimpleNamespace()
    scene.actors = [[]]
    scene.frames_per_second = 1
    return scene, records, reject_first_overlapped


def test_worker_overlaps_every_batch_after_the_first(monkeypatch):
    scene, records, _ = _make_loop_scene(monkeypatch, overlap_enabled=True)

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    # Windows are decided before the overlap ever runs, so they match the
    # serial schedule exactly.
    assert records["durations"] == [2, 2, 2]
    assert [len(frame) for frame in frames] == [2, 2, 2]
    # Batch 1 was fetched synchronously (nothing to hide behind); every
    # successor was prepared on the worker and arrived stamped.
    assert records["worker_threads"], "expected prefetched successors"
    assert all(
        name.startswith("algan-batch-prep") for name in records["worker_threads"]
    )
    assert records["overlapped_flags"] == [False, True, True]


def test_serial_arm_stamps_nothing(monkeypatch):
    scene, records, _ = _make_loop_scene(monkeypatch, overlap_enabled=False)

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    assert records["durations"] == [2, 2, 2]
    assert [len(frame) for frame in frames] == [2, 2, 2]
    assert records["worker_threads"] == []
    assert records["overlapped_flags"] == [False, False, False]


def test_rejected_overlapped_window_discards_work_and_finishes_serially(
    monkeypatch,
):
    scene, records, reject_first_overlapped = _make_loop_scene(
        monkeypatch, overlap_enabled=True
    )
    reject_first_overlapped[0] = True

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    # The worker's window was thrown away and the loop fell back to today's
    # serial path for that stretch: a shorter refetch, then single-frame
    # batches while the cap carries the smaller verdict. Every frame is still
    # rendered exactly once.
    assert records["rejected_overlapped"] == 1
    assert sum(records["durations"]) == 6
    assert records["durations"][0] == 2
    assert 1 in records["durations"]
    assert len(frames) == len(records["durations"])
    assert all(
        name.startswith("algan-batch-prep") for name in records["worker_threads"]
    )


class _ExplodingPrimitive:
    """Projects cleanly unless told to fail part-way through."""

    def __init__(self, fail=False):
        self.fail = fail
        self.memory = None
        self.corners = torch.zeros(3)

    def project_to_screen(self, _camera, _lights):
        # Stands in for the in-place work a real projection does before it can
        # fail (vertex shading accumulates into ``colors``).
        self.corners += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")


def test_prewarm_marks_a_projection_that_fails_part_way(monkeypatch):
    from algan.rendering.raytracing import primitives as rt_primitives

    monkeypatch.setattr(
        rt_primitives, "RayTracedTrianglePrimitive", _ExplodingPrimitive
    )
    monkeypatch.setattr(
        rt_primitives, "RayTracedBezierCircuitPrimitive", _ExplodingPrimitive
    )
    monkeypatch.setattr(rt_module, "project_on_gpu_active", lambda: False)
    monkeypatch.setattr(rt_module, "merge_on_gpu_active", lambda: True)

    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene.video_settings = SimpleNamespace(supersampling=1)
    scene.camera = SimpleNamespace(near=0.0)
    scene.num_pixels_screen_width = 4
    scene.num_pixels_screen_height = 4
    render_state = {
        "ray_origin": torch.zeros(1, 3),
        "screen_point": torch.zeros(1, 3),
        "screen_basis": torch.zeros(1, 3, 3),
        "lights": [],
    }
    primitives = [
        _ExplodingPrimitive(),
        _ExplodingPrimitive(fail=True),
        _ExplodingPrimitive(),
    ]

    with pytest.raises(RuntimeError):
        scene._prewarm_render_batch(primitives, render_state)

    assert primitives[0]._rt_projected is True
    assert primitives[0]._rt_projection_interrupted is False
    # Half-transformed: shaded once, not projected, and says so.
    assert getattr(primitives[1], "_rt_projected", False) is False
    assert primitives[1]._rt_projection_interrupted is True
    # Never reached: pristine and unmarked.
    assert not hasattr(primitives[2], "_rt_projection_interrupted")
    assert torch.equal(primitives[2].corners, torch.zeros(3))


def test_interrupted_worker_projection_rebuilds_the_window(monkeypatch):
    scene, records, _ = _make_loop_scene(monkeypatch, overlap_enabled=True)

    fetches = []
    real_fetch = scene._get_batch_of_primitives

    def recording_fetch(start_ind, end_ind, actors, max_memory):
        batch = real_fetch(start_ind, end_ind, actors, max_memory)
        fetches.append((start_ind, batch[1]))
        return batch

    interrupted = []

    def interrupted_worker(primitive_batch, _render_state):
        if interrupted:
            return
        # Dies mid-projection, as a worker build can beside a live render.
        primitive_batch[0]._rt_projection_interrupted = True
        interrupted.append(primitive_batch[0])
        raise RuntimeError("CUDA out of memory")

    preflighted = []
    real_fits = scene._prepared_batch_fits_render_arena

    def recording_fits(primitives, *args, **kwargs):
        preflighted.append(primitives[0])
        return real_fits(primitives, *args, **kwargs)

    scene._get_batch_of_primitives = recording_fetch
    scene._prepare_batch_on_worker = interrupted_worker
    scene._prepared_batch_fits_render_arena = recording_fits

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    # Same windows as the serial schedule, every frame rendered once.
    assert records["durations"] == [2, 2, 2]
    assert [len(frame) for frame in frames] == [2, 2, 2]
    # The interrupted window was materialized twice at the same bounds, and
    # only the fresh copy reached the render thread.
    assert len(interrupted) == 1
    assert fetches.count((2, 4)) == 2
    assert interrupted[0] not in preflighted
    assert not any(getattr(p, "_rt_projection_interrupted", False) for p in preflighted)


def test_prefix_search_skips_a_runtime_already_known_to_fail(monkeypatch):
    class Scene(RenderLoopMixin):
        pass

    scene = Scene.__new__(Scene)
    scene._begin_batch_cost_measurement()
    probes = []
    monkeypatch.setattr(Scene, "_gpu_merge_headroom_bytes", lambda self: 1 << 30)
    scene._can_slice_fetched_batch = lambda batch, total: True
    scene._slice_fetched_batch = lambda batch, state, duration, total: (
        [SimpleNamespace(duration=duration)],
        state,
    )
    scene._release_preflight_candidate = lambda batch: None

    def fits(candidate, _state, *_args, num_frames=None, **_kwargs):
        probes.append(num_frames)
        return num_frames <= 3

    scene._prepared_batch_fits_render_arena = fits

    baseline = scene._select_largest_fitting_fetched_prefix(
        [object()], {}, 8, (), False
    )
    assert probes[0] == 8
    assert baseline[1] <= 3
    baseline_probes = probes[1:]

    # Told that 8 already failed, the search starts where it would have
    # continued after that probe -- and makes the same choices.
    probes.clear()
    selected = scene._select_largest_fitting_fetched_prefix(
        [object()], {}, 8, (), False, known_failing=8
    )
    assert selected[1] == baseline[1]
    assert probes == baseline_probes

    # A known failure above the projection's own bound changes nothing.
    probes.clear()
    scene._select_largest_fitting_fetched_prefix(
        [object()], {}, 8, (), False, known_failing=9
    )
    assert probes[0] == 8


def _make_sliceable_loop_scene(monkeypatch, *, calibrate=True):
    """The batching loop with sliceable fake batches and a recording planner."""
    scene, records, reject_first_overlapped = _make_loop_scene(
        monkeypatch, overlap_enabled=True
    )
    if not calibrate:
        # Predictors stay cold: the worker builds nothing.
        real_fits = scene._prepared_batch_fits_render_arena

        def fits_without_calibrating(primitives, *args, **kwargs):
            ok = real_fits(primitives, *args, **kwargs)
            records["calibrated"] = False
            return ok

        scene._prepared_batch_fits_render_arena = fits_without_calibrating
    fetched = []
    real_fetch = scene._get_batch_of_primitives

    def recording_fetch(start_ind, end_ind, actors, max_memory):
        batch = real_fetch(start_ind, end_ind, actors, max_memory)
        fetched.append((start_ind, batch[0][0]))
        return batch

    def slice_batch(primitive_batch, render_state, duration, total_frames):
        copies = []
        for primitive in primitive_batch:
            piece = copy.copy(primitive)
            piece.source = primitive
            piece.duration = duration
            piece._rt_projected = False
            piece._rt_prep_overlapped = False
            copies.append(piece)
        return copies, render_state

    selections = []

    def select(
        primitive_batch,
        render_state,
        total_frames,
        post_processes,
        transparent_background,
        *,
        known_failing=None,
    ):
        selections.append((primitive_batch[0], total_frames, known_failing))
        runtime = total_frames if known_failing is None else total_frames - 1
        candidate, state = slice_batch(
            primitive_batch, render_state, runtime, total_frames
        )
        # The planner preflights what it returns, as the real one does.
        scene._prepared_batch_fits_render_arena(
            candidate, state, post_processes, transparent_background
        )
        return candidate, runtime, state

    released = []
    scene._get_batch_of_primitives = recording_fetch
    scene._can_slice_fetched_batch = lambda batch, total: total > 1 and not any(
        p._rt_projected for p in batch
    )
    scene._slice_fetched_batch = slice_batch
    scene._select_largest_fitting_fetched_prefix = select
    scene._may_slice_across_spawns = lambda: True
    scene._release_preflight_candidate = lambda batch: released.append(batch[0])
    return scene, records, reject_first_overlapped, fetched, selections, released


def test_worker_builds_a_copy_and_a_rejected_window_is_trimmed_not_refetched(
    monkeypatch,
):
    scene, records, reject_first_overlapped, fetched, selections, released = (
        _make_sliceable_loop_scene(monkeypatch)
    )
    reject_first_overlapped[0] = True

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    assert sum(records["durations"]) == 6
    assert len(frames) == len(records["durations"])
    assert records["rejected_overlapped"] == 1
    # No window was fetched twice: the rejected one was trimmed from its
    # pristine fetch, which the worker's build never touched.
    starts = [start for start, _ in fetched]
    assert len(starts) == len(set(starts))
    assert fetched[1][1]._rt_projected is False
    (rejected,) = released
    assert rejected.source is fetched[1][1]
    trimmed = [s for s in selections if s[2] is not None]
    assert trimmed == [(fetched[1][1], 2, 2)]
    assert records["durations"][1] == 1


def test_worker_hands_on_the_pristine_batch_when_it_builds_nothing(monkeypatch):
    scene, records, _, fetched, selections, released = _make_sliceable_loop_scene(
        monkeypatch, calibrate=False
    )

    list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    # Every batch reached the planner as the object that was fetched: the
    # worker's unbuilt copy was dropped, not handed on.
    assert [s[0] for s in selections] == [primitive for _, primitive in fetched]
    assert all(s[2] is None for s in selections)
    assert released == []
    assert records["durations"] == [2, 2, 2]


def test_peak_ratio_model_survives_concurrent_observers_and_readers():
    # With the overlap on, the worker predicts while the render thread
    # observes; the deque swap must not corrupt either side.
    model = PeakRatioModel(8.0)
    errors = []
    stop = threading.Event()

    def observe_many():
        try:
            for i in range(5000):
                model.observe(i + 1, (i + 1) * 2)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def read_forever():
        try:
            while not stop.is_set():
                model.predict(1000)
                model.describe()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    readers = [threading.Thread(target=read_forever) for _ in range(2)]
    writers = [threading.Thread(target=observe_many) for _ in range(3)]
    for thread in readers:
        thread.start()
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()

    assert errors == []
    assert len(model._samples) == memory_model_history
    assert model.is_calibrated()
