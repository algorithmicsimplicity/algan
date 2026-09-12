"""Cancellation must discard compiler state before the next warm-daemon job."""

import io
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from algan import daemon
from algan.rendering import taichi_runtime as runtime


def test_cancel_is_authenticated_idle_aware_and_idempotent(monkeypatch):
    interrupts = []
    monkeypatch.setattr(daemon._thread, "interrupt_main", lambda: interrupts.append(1))
    server = SimpleNamespace(
        state=SimpleNamespace(token="secret"),
        busy=threading.Event(),
        cancel_lock=threading.Lock(),
        cancel_pending=False,
    )
    handler = object.__new__(daemon._TriggerHandler)
    handler.server = server
    handler.wfile = io.BytesIO()
    handler._handle_cancel("wrong")
    assert handler.wfile.getvalue() == b"err: bad token\n"
    handler._handle_cancel("secret")
    assert handler.wfile.getvalue().endswith(b"idle\n")
    server.busy.set()
    handler._handle_cancel("secret")
    handler._handle_cancel("secret")
    assert interrupts == [1]
    assert server.cancel_pending


def test_recovery_resets_even_a_partly_initialized_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.ti, "reset", lambda: calls.append("reset"))
    monkeypatch.setattr(runtime, "_already_initialized", lambda: False)
    monkeypatch.setattr(runtime, "_ARCH_READY_FOR", "stale")
    monkeypatch.setattr(runtime, "_BUILT_A_SPECIALIZATION", True)
    monkeypatch.setattr(runtime, "_PRESSURE_RESET_PENDING", True)
    monkeypatch.setattr(runtime, "_COMPILED_IN_SETTINGS", ("stale",))
    runtime._reset_after_interruption()
    assert calls == ["reset"]
    assert runtime._ARCH_READY_FOR is None
    assert runtime._COMPILED_IN_SETTINGS is None
    assert not runtime._BUILT_A_SPECIALIZATION
    assert not runtime._PRESSURE_RESET_PENDING


def test_recovery_refuses_to_reset_under_live_workers(monkeypatch):
    monkeypatch.setattr(runtime, "_RENDER_JOBS_ACTIVE", 1)
    with pytest.raises(RuntimeError, match="render is active"):
        runtime._reset_after_interruption()


@pytest.mark.parametrize("recover", [False, True])
def test_interrupted_render_then_retry_in_same_daemon(tmp_path, recover):
    """Fault-inject the orphaned FieldsBuilder observed after cancellation.

    The first job renders a real batch before being interrupted. The second
    runs in the same daemon and exports a real video. The control arm pins
    the original KeyError; the recovered arm must publish a decodable video.
    """
    script = tmp_path / "daemon_retry.py"
    script.write_text(
        textwrap.dedent("""
        from algan import *
        from algan import daemon as d
        from algan.rendering import taichi_runtime as rt
        from algan.taichi_compat import submodule, ti
        from pathlib import Path
        import sys

        folder = Path(__file__).parent
        events = None
        attempts = 0
        def stdin(queue):
            global events
            events = queue
        d._start_stdin = stdin
        # Memory-pressure recovery also resets the runtime and could mask the
        # injected cancellation fault on a busy host. Isolate that mechanism.
        rt.reset_quadrants_for_memory_pressure = lambda: False
        if sys.argv[1] == "control":
            rt._reset_after_interruption = lambda: None

        def run(path, run_name):
            global attempts
            attempts += 1
            if attempts == 2:
                events.put(("quit", "test finished"))
                rt.init_taichi()
                # Force the next lazy materialization to consult the root
                # builder before a device/settings change could mask it.
                submodule("lang.impl").get_runtime().materialize_root_fb(True)
            Scene.set_video_settings(SMOKE_TEST)
            with Off():
                Circle().spawn()
            Scene.wait(0.5)
            original = Scene._render_primitive_batch
            if attempts == 1:
                def interrupt(scene, *args, **kwargs):
                    original(scene, *args, **kwargs)
                    impl = submodule("lang.impl")
                    impl._root_fb = ti.FieldsBuilder()
                    impl.get_runtime().unfinalized_fields_builder.pop(impl._root_fb)
                    events.put(("render", "retry after cancellation"))
                    raise KeyboardInterrupt
                Scene._render_primitive_batch = interrupt
            try:
                result = Scene.save_video(folder / ("cancelled.mp4" if attempts == 1 else "recovered.mp4"))
                assert result.status == "rendered"
            finally:
                Scene._render_primitive_batch = original
        d.runpy.run_path = run
        d.main([str(folder / "scene.py"), "--no-serve"])
        assert attempts == 2
    """),
        encoding="utf-8",
    )
    (tmp_path / "scene.py").write_text("# served by the test controller\n")
    env = {
        **os.environ,
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_DAEMON_CHILD": "1",
        "ALGAN_PROGRESS": "none",
        "ALGAN_DAEMON_RELEASE_MEMORY": "0",
    }
    result = subprocess.run(
        [sys.executable, str(script), "recover" if recover else "control"],
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    destination = tmp_path / "recovered.mp4"
    if recover:
        assert destination.exists(), output
        from moviepy import VideoFileClip

        with VideoFileClip(str(destination)) as clip:
            assert clip.get_frame(0).shape == (32, 32, 3)
        assert "KeyError" not in output
    else:
        assert "KeyError" in output, output
        assert not destination.exists()
