from __future__ import annotations

from types import SimpleNamespace

import pytest

from algan.utils import profiling_utils as profiler


def test_temporary_hooks_restore_previous_state_on_interruption():
    first = SimpleNamespace(call=lambda: "first")
    second = SimpleNamespace(call=lambda: "second")
    original_first, original_second = first.call, second.call
    with profiler._temporary_instrumentation():
        profiler.TIMERS.wrap_function(first, "call", "outer")
        prior_hook = first.call

        def interrupted_scope():
            with profiler._temporary_instrumentation():
                profiler.TIMERS.wrap_function(first, "call", "already wrapped")
                profiler.TIMERS.wrap_function(second, "call", "inner")
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            interrupted_scope()
        assert first.call is prior_hook
        assert second.call is original_second
    assert first.call is original_first


def test_kernel_materialization_is_not_charged_twice(monkeypatch):
    timers = profiler.StageTimers()
    monkeypatch.setattr(profiler, "TIMERS", timers)
    monkeypatch.setattr(profiler, "_sync_devices", lambda: None)
    clock = [0.0]
    monkeypatch.setattr(profiler.time, "perf_counter", lambda: clock[0])

    def kernel():
        with timers.stage("compile"):
            clock[0] += 2
        clock[0] += 1

    with timers.stage("render"):
        profiler._make_kernel_wrapper(kernel, "example")()
    assert timers.times["render"] == 3
    assert timers.exclusive_times["render"] == 0
    assert timers.exclusive_times["kernel: example"] == 1
    assert timers.exclusive_times["compile"] == 2


@pytest.mark.parametrize("runs", [0, -1, True, 1.5])
def test_invalid_pass_count_fails_before_rendering(runs, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        profiler.profile_scene(
            lambda: pytest.fail("authored"), None, runs=runs, output_directory=tmp_path
        )


@pytest.mark.parametrize("status", ["rendered", "skipped", "exception"])
def test_profile_metadata_and_cleanup(monkeypatch, tmp_path, status):
    events = []
    scene = SimpleNamespace(
        set_video_settings=lambda _: None, _recorded_end_time_for_render=lambda: 3.0
    )
    monkeypatch.setattr(profiler.SceneManager, "reset", lambda: scene)
    monkeypatch.setattr(profiler, "KERNEL_PROFILER", False)
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiler, "_sync_devices", lambda: None)
    monkeypatch.setattr(profiler, "_collect_taichi_kernel_gpu", lambda: {})

    class Sampler:
        def start(self):
            events.append("start")
            return self

        def stop(self):
            events.append("stop")

        def summary(self):
            return {}

    monkeypatch.setattr(profiler, "GpuTelemetrySampler", Sampler)

    def save(path, **options):
        assert str(path).startswith(str(tmp_path))
        assert options["reset"] is True
        if status == "exception":
            raise RuntimeError("encoder broke")
        return SimpleNamespace(status=status, output_path=path)

    monkeypatch.setattr(profiler.Scene, "save_video", save)
    if status == "rendered":
        result = profiler.run_once(lambda: None, None, output_directory=tmp_path)
        assert result["scene_seconds"] == 3.0
        assert result["authoring_seconds"] >= 0
        assert str(tmp_path) in result["output_path"]
    else:
        with pytest.raises(RuntimeError):
            profiler.run_once(lambda: None, None, output_directory=tmp_path)
    assert events == ["start", "stop"]
