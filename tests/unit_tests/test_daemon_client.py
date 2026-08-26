"""Handoff client: gating, framing, and the fallback contract.

These never start a real daemon (that would cost a render). The wire is
exercised against a fake server so the framing and the fallback rules are
pinned without one.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading

import pytest

from algan import daemon_client as dc

# --------------------------------------------------------------------------
# should_try: the gate that keeps unrelated processes out of the daemon
# --------------------------------------------------------------------------


class _Main:
    def __init__(self, file):
        self.__file__ = file


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "scene.py"
    path.write_text("import algan\n", encoding="utf-8")
    return _Main(str(path))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ALGAN_DAEMON_CHILD", "ALGAN_USE_DAEMON", "ALGAN_AUTO_DAEMON"):
        monkeypatch.delenv(name, raising=False)


def _hide_test_runner(monkeypatch):
    """Make this process look like a plain ``python scene.py``.

    ``should_try`` refuses under a test runner -- which is exactly where these
    tests run -- so both markers have to go for the *other* conditions to be
    what a test measures. It must happen in the test body: pytest re-sets
    ``PYTEST_CURRENT_TEST`` for the call phase, after fixtures.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(dc.sys.modules, "pytest", raising=False)
    # CI runs this suite under coverage, whose trace function ``debugger_name``
    # reports on purpose (see its docstring). Blank it so each test below
    # measures the condition it names rather than how the suite was launched.
    monkeypatch.setattr(dc, "debugger_name", lambda: None)


def test_hands_off_for_a_plain_script_run(script, monkeypatch):
    _hide_test_runner(monkeypatch)
    assert dc.should_try(script) is True


def test_declines_when_disabled(script, monkeypatch):
    _hide_test_runner(monkeypatch)
    monkeypatch.setenv("ALGAN_USE_DAEMON", "0")
    assert dc.should_try(script) is False


def test_declines_inside_the_daemons_own_run(script, monkeypatch):
    """The daemon sets this; without it the handoff would recurse."""
    _hide_test_runner(monkeypatch)
    monkeypatch.setenv("ALGAN_DAEMON_CHILD", "1")
    assert dc.should_try(script) is False


def test_declines_under_a_test_runner(script, monkeypatch):
    _hide_test_runner(monkeypatch)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "something::test")
    assert dc.should_try(script) is False, "the env marker alone must be enough"
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.setitem(dc.sys.modules, "pytest", pytest)
    assert dc.should_try(script) is False, "an imported pytest must be enough"


def test_declines_a_dash_m_invocation(script, monkeypatch):
    """``python -m pkg`` is not a scene script, and it looks like one.

    ``__main__`` is then the package's own ``__main__.py``: it ends in .py and
    exists on disk, so the path checks alone let it through. That handed
    ``python -m sphinx`` -- whose conf.py imports algan -- to a render daemon,
    and the documentation build ran inside it. Only a -m invocation gives
    ``__main__`` a module spec.
    """
    _hide_test_runner(monkeypatch)
    assert dc.should_try(script) is True  # the same module, minus the spec
    script.__spec__ = object()
    assert dc.should_try(script) is False


def test_declines_without_a_main_script(monkeypatch):
    _hide_test_runner(monkeypatch)
    assert dc.should_try(_Main(None)) is False
    assert dc.should_try(object()) is False


def test_declines_for_a_non_python_main(tmp_path, monkeypatch):
    _hide_test_runner(monkeypatch)
    other = tmp_path / "thing.txt"
    other.write_text("", encoding="utf-8")
    assert dc.should_try(_Main(str(other))) is False


# --------------------------------------------------------------------------
# Debuggers: the daemon's process is not the one with the breakpoints in it
# --------------------------------------------------------------------------


def _no_debugger_modules(monkeypatch):
    for name, _label in dc._DEBUGGER_MODULES:
        monkeypatch.delitem(dc.sys.modules, name, raising=False)


@pytest.mark.parametrize(("module", "label"), dc._DEBUGGER_MODULES)
def test_each_known_debugger_is_named(module, label, monkeypatch):
    _no_debugger_modules(monkeypatch)
    monkeypatch.setitem(dc.sys.modules, module, object())
    assert dc.debugger_name() == label


def test_any_trace_function_counts(monkeypatch):
    """pdb, coverage, anything: it is watching *these* frames, not the daemon's."""
    _no_debugger_modules(monkeypatch)
    monkeypatch.setattr(dc.sys, "gettrace", lambda: object())
    assert dc.debugger_name() == "a tracing tool (sys.gettrace)"


def test_an_undebugged_process_has_no_debugger(monkeypatch):
    _no_debugger_modules(monkeypatch)
    monkeypatch.setattr(dc.sys, "gettrace", lambda: None)
    assert dc.debugger_name() is None


def test_detection_never_raises(monkeypatch):
    """A broken probe must cost one cold start, not an ImportError at import."""

    def boom():
        raise RuntimeError("something replaced sys.gettrace")

    _no_debugger_modules(monkeypatch)
    monkeypatch.setattr(dc.sys, "gettrace", boom)
    assert dc.debugger_name() is None


def test_declines_under_a_debugger(script, monkeypatch, capsys):
    """The handoff would run the script where the breakpoints are not."""
    _hide_test_runner(monkeypatch)
    monkeypatch.setattr(dc, "debugger_name", lambda: "pydevd (PyCharm / PyDev)")
    assert dc.should_try(script) is False
    told = capsys.readouterr().err
    assert "pydevd" in told, "a silent cold start is the thing to avoid"
    assert "ALGAN_USE_DAEMON=1" in told, "the override has to be discoverable"


def test_an_explicit_opt_in_overrides_the_debugger_check(script, monkeypatch, capsys):
    """For the warm *and* debuggable arrangement: a debugged daemon.

    An unset variable is not an opt-in -- the daemon is on by default, so only
    a value that was actually written down can mean "yes, even here".
    """
    _hide_test_runner(monkeypatch)
    monkeypatch.setattr(dc, "debugger_name", lambda: "debugpy (VS Code)")
    assert dc.should_try(script) is False
    monkeypatch.setenv("ALGAN_USE_DAEMON", "1")
    assert dc.should_try(script) is True
    assert "breakpoints" in capsys.readouterr().err, "both paths must say so"


def test_a_debugged_non_candidate_is_told_nothing(script, monkeypatch, capsys):
    """Only a process that would otherwise hand off gets the explanation.

    This one is a debugged pytest run: it was never going to reach the daemon,
    so warning it about breakpoints would be noise.
    """
    monkeypatch.setattr(dc, "debugger_name", lambda: "pydevd (PyCharm / PyDev)")
    assert dc.should_try(script) is False
    assert capsys.readouterr().err == ""


def test_nothing_is_said_when_there_was_no_handoff_to_lose(
    script, tmp_path, monkeypatch, capsys
):
    """With auto-start off and no daemon running, the run was cold regardless."""
    _hide_test_runner(monkeypatch)
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    monkeypatch.setenv("ALGAN_AUTO_DAEMON", "0")
    monkeypatch.setattr(dc, "debugger_name", lambda: "pydevd (PyCharm / PyDev)")
    assert dc.should_try(script) is False
    assert capsys.readouterr().err == ""


def test_no_daemon_is_started_under_a_debugger(tmp_path, monkeypatch, capsys):
    """Declining must also mean not spawning one in the background."""
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    _hide_test_runner(monkeypatch)
    monkeypatch.setattr(dc, "debugger_name", lambda: "pydevd (PyCharm / PyDev)")
    monkeypatch.setattr(
        dc, "_spawn_daemon", lambda: pytest.fail("must not start a daemon")
    )
    monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("must not exit"))
    main = tmp_path / "scene.py"
    main.write_text("import algan\n", encoding="utf-8")
    monkeypatch.setitem(dc.sys.modules, "__main__", _Main(str(main)))
    assert dc.maybe_handoff() is None
    assert "pydevd" in capsys.readouterr().err


# --------------------------------------------------------------------------
# State file discovery -- absence must be the cheap, silent path
# --------------------------------------------------------------------------


def test_no_state_file_means_no_daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    assert dc.read_state() is None


def test_malformed_state_file_is_treated_as_no_daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    (tmp_path / "daemon.json").write_text("{not json", encoding="utf-8")
    assert dc.read_state() is None
    (tmp_path / "daemon.json").write_text('{"port": 1}', encoding="utf-8")
    assert dc.read_state() is None, "a state file without a token is unusable"


# --------------------------------------------------------------------------
# Startup env: the daemon cannot adopt these, so a mismatch must be caught
# --------------------------------------------------------------------------


def test_matching_startup_env_is_no_mismatch():
    env = dict.fromkeys(dc.STARTUP_ENV, "x")
    assert dc.describe_env_mismatch(env, env) is None


def test_unset_and_empty_are_the_same_value():
    assert dc.describe_env_mismatch({}, dict.fromkeys(dc.STARTUP_ENV, "")) is None


def test_render_device_mismatch_is_reported():
    report = dc.describe_env_mismatch(
        {"ALGAN_RENDER_DEVICE": "cpu"}, {"ALGAN_RENDER_DEVICE": "cuda"}
    )
    assert report is not None
    assert "ALGAN_RENDER_DEVICE" in report
    assert "'cpu'" in report
    assert "'cuda'" in report
    assert "ALGAN_USE_DAEMON=0" in report, "the report must say how to proceed"


# --------------------------------------------------------------------------
# Import-time env: read into module defaults when the daemon started, so a
# client wanting different values has to run cold too
# --------------------------------------------------------------------------


def test_matching_import_env_is_no_mismatch():
    env = dict.fromkeys(dc.IMPORT_TIME_ENV, "x")
    assert dc.describe_import_env_mismatch(env, env) is None


def test_an_unset_import_variable_matches_an_empty_one():
    assert (
        dc.describe_import_env_mismatch({}, dict.fromkeys(dc.IMPORT_TIME_ENV, ""))
        is None
    )


def test_a_toggle_the_daemon_never_saw_is_reported():
    report = dc.describe_import_env_mismatch(
        {"ALGAN_SHEET_RESOLVE": "0"}, {"ALGAN_SHEET_RESOLVE": "1"}
    )
    assert report is not None
    assert "ALGAN_SHEET_RESOLVE" in report
    assert "'0'" in report
    assert "'1'" in report
    assert "fresh process" in report, "the report must say what happens next"


def test_a_live_variable_is_not_grounds_for_refusal():
    """The A/B case that must keep working warm: flipping an arm mid-script.

    A variable read at the point of use is picked up by the very next read, on
    the daemon exactly as in a fresh process, so a difference in one is not a
    reason to refuse a run.
    """
    assert (
        dc.describe_import_env_mismatch(
            {"ALGAN_PREFETCH_BATCHES": "0"}, {"ALGAN_PREFETCH_BATCHES": "1"}
        )
        is None
    )


def test_the_transport_variables_are_exempt():
    """They configure the handoff that has already happened by this point."""
    for name in dc.IMPORT_TIME_ENV_EXEMPT:
        assert dc.describe_import_env_mismatch({name: "1"}, {name: "2"}) is None


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def test_frame_round_trip(tmp_path):
    path = tmp_path / "frames.bin"
    with open(path, "wb") as fh:
        dc.write_frame(fh, dc.FRAME_STDOUT, b"hello")
        dc.write_frame(fh, dc.FRAME_INFO, "queued")
        dc.write_frame(fh, dc.FRAME_START)
        dc.write_frame(fh, dc.FRAME_EXIT, struct.pack("!i", 3))
    with open(path, "rb") as fh:
        assert dc.read_frame(fh) == (dc.FRAME_STDOUT, b"hello")
        assert dc.read_frame(fh) == (dc.FRAME_INFO, b"queued")
        assert dc.read_frame(fh) == (dc.FRAME_START, b"")
        kind, payload = dc.read_frame(fh)
        assert kind == dc.FRAME_EXIT
        assert struct.unpack("!i", payload) == (3,)
        assert dc.read_frame(fh) == (None, b"")


def test_truncated_frame_reads_as_eof(tmp_path):
    path = tmp_path / "cut.bin"
    path.write_bytes(dc.FRAME_STDOUT + struct.pack("!I", 100) + b"short")
    with open(path, "rb") as fh:
        assert dc.read_frame(fh) == (None, b"")


# --------------------------------------------------------------------------
# The wire, against a fake daemon
# --------------------------------------------------------------------------


class _FakeDaemon:
    """Accepts one connection and replies with a scripted frame sequence."""

    def __init__(self, reply):
        self._reply = reply
        self.request = None
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self.token = "t0ken"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def state(self):
        return {"port": self.port, "token": self.token}

    def _serve(self):
        conn, _ = self._sock.accept()
        with conn:
            stream = conn.makefile("rwb")
            stream.readline()  # the "run" command line
            (length,) = struct.unpack("!I", stream.read(4))
            self.request = json.loads(stream.read(length).decode("utf-8"))
            self._reply(stream)
            stream.flush()
        self._sock.close()

    def join(self):
        self._thread.join(timeout=5)


class _Sink:
    def __init__(self):
        self.data = b""

    def write(self, chunk):
        self.data += chunk

    def flush(self):
        pass


def test_run_remote_streams_output_and_returns_the_exit_code(tmp_path):
    def reply(stream):
        dc.write_frame(stream, dc.FRAME_START)
        dc.write_frame(stream, dc.FRAME_STDOUT, b"rendering\n")
        dc.write_frame(stream, dc.FRAME_STDERR, b"100%\n")
        dc.write_frame(stream, dc.FRAME_EXIT, struct.pack("!i", 7))

    daemon = _FakeDaemon(reply)
    out, err = _Sink(), _Sink()
    code = dc.run_remote(
        daemon.state(),
        str(tmp_path / "s.py"),
        argv=["--flag"],
        cwd=str(tmp_path),
        out=out,
        err=err,
    )
    daemon.join()
    assert code == 7
    assert out.data == b"rendering\n"
    assert b"100%\n" in err.data
    assert daemon.request["script"].endswith("s.py")
    assert daemon.request["argv"] == ["--flag"]
    assert daemon.request["token"] == "t0ken"
    assert daemon.request["protocol"] == dc.PROTOCOL_VERSION
    assert set(daemon.request["env"]) == set(dc.STARTUP_ENV)


def test_refusal_before_start_is_recoverable(tmp_path):
    def reply(stream):
        dc.write_frame(stream, dc.FRAME_REFUSE, "wrong device")

    daemon = _FakeDaemon(reply)
    with pytest.raises(dc.DaemonUnavailable, match="wrong device"):
        dc.run_remote(daemon.state(), str(tmp_path / "s.py"), out=_Sink(), err=_Sink())
    daemon.join()


def test_death_after_start_is_not_recoverable(tmp_path):
    """Once the script is running, re-running locally could duplicate effects."""

    def reply(stream):
        dc.write_frame(stream, dc.FRAME_START)
        dc.write_frame(stream, dc.FRAME_STDOUT, b"half a render\n")
        # then the connection closes with no exit frame

    daemon = _FakeDaemon(reply)
    with pytest.raises(dc.DaemonRunFailed):
        dc.run_remote(daemon.state(), str(tmp_path / "s.py"), out=_Sink(), err=_Sink())
    daemon.join()


def test_death_before_start_falls_back(tmp_path):
    daemon = _FakeDaemon(lambda stream: None)
    with pytest.raises(dc.DaemonUnavailable):
        dc.run_remote(daemon.state(), str(tmp_path / "s.py"), out=_Sink(), err=_Sink())
    daemon.join()


def test_nothing_listening_falls_back(tmp_path):
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    with pytest.raises(dc.DaemonUnavailable):
        dc.run_remote(
            {"port": port, "token": "x"},
            str(tmp_path / "s.py"),
            out=_Sink(),
            err=_Sink(),
        )


def test_maybe_handoff_is_a_noop_without_a_daemon(tmp_path, monkeypatch):
    """The common case: no daemon, so the import must simply continue."""
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("must not exit"))
    assert dc.maybe_handoff() is None


def test_the_full_environment_is_shipped(tmp_path, monkeypatch):
    """Without it a script reads the daemon's variables, not its caller's."""
    monkeypatch.setenv("ALGAN_TEST_PROBE_VAR", "from-the-client")

    def reply(stream):
        dc.write_frame(stream, dc.FRAME_START)
        dc.write_frame(stream, dc.FRAME_EXIT, struct.pack("!i", 0))

    daemon = _FakeDaemon(reply)
    dc.run_remote(daemon.state(), str(tmp_path / "s.py"), out=_Sink(), err=_Sink())
    daemon.join()
    assert daemon.request["env_full"]["ALGAN_TEST_PROBE_VAR"] == "from-the-client"


# --------------------------------------------------------------------------
# Unreachable vs refused: only one of them invalidates the registration
# --------------------------------------------------------------------------


def test_nothing_listening_reads_as_unreachable(tmp_path):
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    with pytest.raises(dc.DaemonUnreachable):
        dc.run_remote(
            {"port": port, "token": "x"},
            str(tmp_path / "s.py"),
            out=_Sink(),
            err=_Sink(),
        )


def test_a_refusal_is_not_unreachable(tmp_path):
    """A daemon that answers and declines is alive; its registration stands."""

    def reply(stream):
        dc.write_frame(stream, dc.FRAME_REFUSE, "algan sources changed")

    daemon = _FakeDaemon(reply)
    with pytest.raises(dc.DaemonUnavailable) as caught:
        dc.run_remote(daemon.state(), str(tmp_path / "s.py"), out=_Sink(), err=_Sink())
    daemon.join()
    assert not isinstance(caught.value, dc.DaemonUnreachable)


# --------------------------------------------------------------------------
# Auto-start
# --------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    return tmp_path


def _no_spawn(monkeypatch):
    monkeypatch.setattr(
        dc, "_spawn_daemon", lambda: pytest.fail("must not start a daemon")
    )


def test_no_daemon_is_started_when_disabled(home, monkeypatch):
    monkeypatch.setenv("ALGAN_AUTO_DAEMON", "0")
    _no_spawn(monkeypatch)
    assert dc._dispatch("scene.py") is None


def test_no_daemon_is_started_under_a_test_runner(home, monkeypatch):
    """should_try already refuses here, which is what keeps the suite clean."""
    _no_spawn(monkeypatch)
    monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("must not exit"))
    assert dc.maybe_handoff() is None


def test_a_dead_registration_is_removed_and_replaced(home, monkeypatch):
    """A hard-killed daemon leaves its state file behind.

    Left alone it would defeat auto-start forever: every later run finds a
    state file, fails to connect, and falls back cold without ever starting a
    replacement.
    """
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    state_file = home / "daemon.json"
    state_file.write_text(
        json.dumps({"port": port, "token": "dead", "pid": 999999}), encoding="utf-8"
    )
    monkeypatch.setenv("ALGAN_AUTO_DAEMON", "0")  # stop after the cleanup

    assert dc._dispatch("scene.py") is None
    assert not state_file.exists()


def test_a_newer_registration_survives_the_cleanup(home):
    """Only the daemon we actually failed to reach gets de-registered."""
    state_file = home / "daemon.json"
    state_file.write_text(
        json.dumps({"port": 1, "token": "new", "pid": 2}), encoding="utf-8"
    )
    dc._clear_stale_state({"port": 1, "token": "old", "pid": 1})
    assert state_file.exists()


def test_the_daemon_log_is_trimmed_when_it_grows(home, monkeypatch):
    monkeypatch.setenv("ALGAN_DAEMON_LOG_MAX_BYTES", "100")
    log = home / "daemon.log"
    log.write_bytes(b"x" * 500)
    with dc._open_log() as handle:
        handle.write(b"fresh\n")
    assert log.read_bytes() == b"fresh\n"
    assert (home / "daemon.log.old").read_bytes() == b"x" * 500
