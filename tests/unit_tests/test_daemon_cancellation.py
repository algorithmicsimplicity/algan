"""Cancellation must discard compiler state before the next warm-daemon job."""

import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from algan import daemon
from algan.rendering import taichi_runtime as runtime


def test_early_ctrl_c_waits_for_acceptance_and_restores_signal_handler(monkeypatch):
    from algan import daemon_client

    sent = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sendall(self, message):
            sent.append(message)

    monkeypatch.setattr(
        daemon_client.socket, "create_connection", lambda *_args: Connection()
    )
    previous = signal.getsignal(signal.SIGINT)
    accepted, restore = daemon_client._install_cancel_handler(
        {"port": 1234, "token": "secret"}, "queued-job"
    )
    try:
        signal.raise_signal(signal.SIGINT)
        assert sent == []
        accepted()
        accepted()
        assert sent == [b"cancel secret queued-job\n"]
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    finally:
        restore()
    assert signal.getsignal(signal.SIGINT) is previous


def test_scoped_cancel_does_not_interrupt_a_different_active_job(monkeypatch):
    interrupts = []
    monkeypatch.setattr(daemon._thread, "interrupt_main", lambda: interrupts.append(1))
    active = daemon._RunJob({"script": "a.py", "request_id": "a"}, io.BytesIO())
    queued = daemon._RunJob({"script": "b.py", "request_id": "b"}, io.BytesIO())
    server = SimpleNamespace(
        state=SimpleNamespace(token="secret"),
        busy=threading.Event(),
        cancel_lock=threading.Lock(),
        cancel_pending=False,
        jobs={"a": active, "b": queued},
        active_job=active,
    )
    server.busy.set()
    handler = object.__new__(daemon._TriggerHandler)
    handler.server, handler.wfile = server, io.BytesIO()
    handler._handle_cancel("wrong", "b")
    assert not queued.done.is_set()
    handler._handle_cancel("secret", "b")
    handler._handle_cancel("secret", "b")
    handler._handle_cancel("secret", "unknown")
    assert queued.cancelled
    assert queued.done.is_set()
    assert interrupts == []
    assert not active.cancelled
    handler._handle_cancel("secret", "a")
    handler._handle_cancel("secret", "a")
    assert interrupts == [1]


@pytest.mark.skipif(
    sys.platform == "win32", reason="SIGINT process delivery is POSIX-only"
)
def test_cancelling_queued_remote_client_preserves_active_job(tmp_path):
    """Real daemon, production clients, private socket and owned processes."""
    state_dir = tmp_path / "daemon-home"
    env = {
        **os.environ,
        "ALGAN_HOME": str(state_dir),
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_DAEMON_CHILD": "1",
        "ALGAN_PRECOMPILE_JOBS": "0",
    }
    a = tmp_path / "a.py"
    a.write_text(
        "from pathlib import Path\nimport time\n"
        "folder = Path(__file__).parent\n"
        "(folder / 'a-started').touch()\n"
        "while not (folder / 'release-a').exists(): time.sleep(0.02)\n"
        "(folder / 'a-finished').touch()\n"
    )
    b = tmp_path / "b.py"
    b.write_text(
        "from pathlib import Path\nPath(__file__).with_suffix('.ran').touch()\n"
    )
    client_code = (
        "import json, sys\nfrom algan.daemon_client import run_remote\n"
        "with open(sys.argv[1]) as f: state = json.load(f)\n"
        "sys.exit(run_remote(state, sys.argv[2], argv=[]))\n"
    )
    processes = []

    def wait_until(predicate):
        deadline = time.monotonic() + 40
        while not predicate():
            assert time.monotonic() < deadline, (
                "daemon/client did not reach the expected state"
            )
            time.sleep(0.025)

    with (
        (tmp_path / "daemon.log").open("w") as daemon_log,
        (tmp_path / "b.log").open("w") as b_log,
    ):
        try:
            server = subprocess.Popen(
                [sys.executable, "-m", "algan.daemon", "--port", "0"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=daemon_log,
                stderr=daemon_log,
            )
            processes.append(server)
            state_file = state_dir / "daemon.json"
            wait_until(state_file.exists)
            # Verify that setup published an actual ephemeral socket.
            assert json.loads(state_file.read_text())["port"] > 0
            active = subprocess.Popen(
                [sys.executable, "-c", client_code, str(state_file), str(a)],
                env=env,
                stdout=daemon_log,
                stderr=daemon_log,
            )
            processes.append(active)
            wait_until((tmp_path / "a-started").exists)
            queued = subprocess.Popen(
                [sys.executable, "-c", client_code, str(state_file), str(b)],
                env=env,
                stdout=b_log,
                stderr=b_log,
            )
            processes.append(queued)
            wait_until(lambda: "queued behind" in (tmp_path / "b.log").read_text())
            queued.send_signal(signal.SIGINT)
            assert queued.wait(timeout=15) == 130
            assert active.poll() is None
            (tmp_path / "release-a").touch()
            assert active.wait(timeout=15) == 0
            assert (tmp_path / "a-finished").exists()
            assert not b.with_suffix(".ran").exists()
        finally:
            (tmp_path / "release-a").touch()
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


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
