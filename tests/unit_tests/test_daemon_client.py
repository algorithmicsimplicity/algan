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


#: A script the render parse (``script_may_render``) recognises as rendering.
_RENDERING_SCRIPT = "from algan import *\n\nSquare().spawn()\nScene.save_video()\n"


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "scene.py"
    path.write_text(_RENDERING_SCRIPT, encoding="utf-8")
    return _Main(str(path))


@pytest.fixture
def renderless_script(tmp_path):
    """Imports algan to compute something and never renders (issue 8's repro)."""
    path = tmp_path / "numbers.py"
    path.write_text(
        "import os\nprint('[before import]', os.getpid())\n"
        "from algan import *\nimport torch\n"
        "print(float(torch.tensor([1.0, 2.0]).view(2, 1).sum()), RIGHT)\n",
        encoding="utf-8",
    )
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
    main.write_text(_RENDERING_SCRIPT, encoding="utf-8")
    monkeypatch.setitem(dc.sys.modules, "__main__", _Main(str(main)))
    assert dc.maybe_handoff() is None
    assert "pydevd" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Scripts that never render are not handed off (issue 8)
# --------------------------------------------------------------------------


def test_declines_a_script_that_never_renders(renderless_script, monkeypatch):
    """Nothing for a warm renderer to do, and its pre-import code would run twice."""
    _hide_test_runner(monkeypatch)
    assert dc.should_try(renderless_script) is False


def test_an_explicit_opt_in_hands_off_a_script_the_parse_calls_renderless(
    renderless_script, monkeypatch
):
    """For a render the parse cannot see, e.g. through a module elsewhere."""
    _hide_test_runner(monkeypatch)
    monkeypatch.setenv("ALGAN_USE_DAEMON", "1")
    assert dc.should_try(renderless_script) is True


def test_a_debugged_renderless_script_is_told_nothing(
    renderless_script, monkeypatch, capsys
):
    """It was never going to be handed off, so there are no breakpoints to save."""
    _hide_test_runner(monkeypatch)
    monkeypatch.setattr(dc, "debugger_name", lambda: "pydevd (PyCharm / PyDev)")
    assert dc.should_try(renderless_script) is False
    assert capsys.readouterr().err == ""


def test_no_daemon_is_started_for_a_script_that_never_renders(
    renderless_script, tmp_path, monkeypatch, capsys
):
    """The report's repro: no handoff, no background daemon, nothing said."""
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    _hide_test_runner(monkeypatch)
    monkeypatch.setattr(
        dc, "_spawn_daemon", lambda: pytest.fail("must not start a daemon")
    )
    monkeypatch.setattr(dc, "run_remote", lambda *a, **k: pytest.fail("no handoff"))
    monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("must not exit"))
    monkeypatch.setitem(dc.sys.modules, "__main__", renderless_script)
    assert dc.maybe_handoff() is None
    assert capsys.readouterr().err == ""


def _may_render(tmp_path, source, search_path=(), **files):
    """``script_may_render`` for a script with ``source`` and sibling ``files``.

    ``search_path`` stands in for ``sys.path``, which in a test process holds
    the repository and the test folders.
    """
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    main = tmp_path / "main_script.py"
    main.write_text(source, encoding="utf-8")
    return dc.script_may_render(str(main), search_path=list(search_path))


@pytest.mark.parametrize(
    "source",
    [
        "Scene.save_video()",
        "scene.save_frame('x.png', at=1)",
        "Scene.show_frame()",
        "Scene.view()",
        "scene.view(PREVIEW)",
        "project.view(scenes=[0])",
        "Project([intro]).render_video()",
        "p.render_screenshots()",
        "project.run_cli()",
        "project.profile()",
        "from algan.utils.algan_utils import render_all_funcs",
        "getattr(Scene, 'save_video')()",
        "def main():\n    Scene.save_video()\n",  # named, if never called
        "viewer = scene.view",
        # Code the parse cannot follow might render.
        "import runpy\nrunpy.run_path('other.py')",
        "import importlib\nimportlib.import_module(name)",
        "exec(open('other.py').read())",
        # So might code from a folder the script puts on the import path.
        "import sys\nsys.path.insert(0, '..')\nfrom common import go\ngo()",
        "import sys\nsys.path.append(str(ROOT))",
        "import sys\nsys.path[:0] = [ROOT]",
        "import site\nsite.addsitedir(ROOT)",
    ],
)
def test_a_script_naming_a_render_may_render(tmp_path, source):
    assert _may_render(tmp_path, "from algan import *\n" + source + "\n")


@pytest.mark.parametrize(
    "source",
    [
        "print(RIGHT, BLUE)",
        "import torch\nx = torch.zeros(4).view(2, 2).view(-1)",
        "import numpy as np\ny = np.zeros(4).view(np.int32)",
        "# Scene.save_video() would go here\n",
        '"""Call Scene.save_video() to render."""\n',
        "label = 'save the video'",
        "import os, sys\nimport torch\n",
    ],
)
def test_a_script_naming_no_render_never_renders(tmp_path, source):
    assert not _may_render(tmp_path, "from algan import *\n" + source + "\n")


def test_a_render_in_a_sibling_module_counts(tmp_path):
    files = {"helpers.py": "from algan import *\ndef out():\n    Scene.save_video()\n"}
    assert _may_render(tmp_path, "import helpers\nhelpers.out()\n", **files)
    assert _may_render(tmp_path, "from helpers import out\nout()\n", **files)


def test_a_renderless_sibling_module_does_not(tmp_path):
    files = {"helpers.py": "import math\nTAU = 2 * math.pi\n"}
    assert not _may_render(tmp_path, "from helpers import TAU\nprint(TAU)\n", **files)


def test_packages_are_followed_through_init_and_relative_imports(tmp_path):
    files = {
        "scenes/__init__.py": "from .intro import make\n",
        "scenes/intro.py": "from algan import *\ndef make():\n    Scene.save_frame()\n",
    }
    assert _may_render(tmp_path, "from scenes import make\nmake()\n", **files)
    assert _may_render(tmp_path, "import scenes.intro\n", **files)


def test_imports_are_followed_transitively(tmp_path):
    files = {
        "a.py": "import b\n",
        "b.py": "from algan import *\nScene.save_video()\n",
    }
    assert _may_render(tmp_path, "import a\n", **files)


def test_project_folders_on_the_import_path_are_followed(tmp_path):
    """``PYTHONPATH``, an editable install, a folder added before the import."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "common.py").write_text(
        "from algan import *\ndef go():\n    Scene.save_video()\n", encoding="utf-8"
    )
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    script = "from common import go\ngo()\n"
    assert not _may_render(scenes, script)
    assert _may_render(scenes, script, search_path=[str(shared)])


def test_installed_libraries_are_not_read(tmp_path):
    """site-packages names render methods too (algan's own, for one)."""
    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    roots = dc._project_roots(str(tmp_path), [purelib, str(tmp_path)])
    assert roots == [os.path.normcase(os.path.realpath(tmp_path))]


def test_algan_beside_the_script_is_not_read_as_the_scripts_code(tmp_path):
    """A source checkout: algan's own files name every render method."""
    files = {"algan/__init__.py": "def save_video():\n    pass\n"}
    assert not _may_render(tmp_path, "from algan import *\nprint(RIGHT)\n", **files)


def test_an_unreadable_script_is_treated_as_rendering(tmp_path):
    """Anything the parse cannot read gets the handoff it always got."""
    assert dc.script_may_render(str(tmp_path / "missing.py"))
    files = {"broken.py": "def (:\n"}
    assert _may_render(tmp_path, "import broken\n", **files)


def test_too_many_local_modules_is_treated_as_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "_MAX_SCANNED_FILES", 2)
    files = {f"m{i}.py": "x = 1\n" for i in range(3)}
    assert _may_render(tmp_path, "import m0, m1, m2\n", **files)


def test_an_auto_started_daemon_is_told_about_the_run_it_is_for(home, monkeypatch):
    """It stands down if that run turns out to render nothing (algan.daemon)."""
    import subprocess

    launched = []

    class _Popen:
        def __init__(self, argv, **kwargs):
            launched.append(argv)

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    assert dc._spawn_daemon() is not None
    assert launched
    assert "--exit-if-first-run-renders-nothing" in launched[0]


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
# Startup env: the daemon cannot adopt these, so a mismatch must be caught --
# except for STARTUP_ENV_ADOPTED, which it applies per run instead
# --------------------------------------------------------------------------


def test_matching_startup_env_is_no_mismatch():
    env = dict.fromkeys(dc.STARTUP_ENV, "x")
    assert dc.describe_env_mismatch(env, env) is None


def test_unset_and_empty_are_the_same_value():
    assert dc.describe_env_mismatch({}, dict.fromkeys(dc.STARTUP_ENV, "")) is None


def test_animation_device_mismatch_is_reported():
    report = dc.describe_env_mismatch(
        {"ALGAN_ANIMATION_DEVICE": "cpu"}, {"ALGAN_ANIMATION_DEVICE": "cuda"}
    )
    assert report is not None
    assert "ALGAN_ANIMATION_DEVICE" in report
    assert "'cpu'" in report
    assert "'cuda'" in report
    assert "ALGAN_USE_DAEMON=0" in report, "the report must say how to proceed"


def test_a_render_device_mismatch_is_adopted_rather_than_refused():
    """The one startup variable a warm daemon takes from its client.

    It only seeds ``SETTINGS.computing.render_device``, which the daemon
    re-applies per run (``algan.daemon._adopt_render_device``) and which every
    render re-reads when it selects Taichi's arch. Refusing it would send the
    script to a cold process to reach a device the warm one can just switch to.
    """
    assert "ALGAN_RENDER_DEVICE" in dc.STARTUP_ENV_ADOPTED
    assert (
        dc.describe_env_mismatch(
            {"ALGAN_RENDER_DEVICE": "cpu"}, {"ALGAN_RENDER_DEVICE": "cuda"}
        )
        is None
    )


def test_an_adopted_variable_does_not_mask_a_real_mismatch():
    report = dc.describe_env_mismatch(
        {"ALGAN_RENDER_DEVICE": "cpu", "ALGAN_ANIMATION_DEVICE": "cpu"},
        {"ALGAN_RENDER_DEVICE": "cuda", "ALGAN_ANIMATION_DEVICE": "cuda"},
    )
    assert report is not None
    assert "ALGAN_ANIMATION_DEVICE" in report
    assert "ALGAN_RENDER_DEVICE" not in report


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


def test_the_transport_variables_are_not_grounds_for_refusal():
    """They configure the handoff that has already happened by this point.

    They used to need a named exemption from the import-time comparison; both
    are now read at the point of use, so they are declared live and never
    reach it.
    """
    for name in ("ALGAN_DAEMON_PORT", "ALGAN_DAEMON_TIMEOUT"):
        assert name not in dc.IMPORT_TIME_ENV
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


def _identity_state(port, **overrides):
    state = {"port": port, "token": "t", "pid": 1, **dc.interpreter_identity()}
    state.update(overrides)
    return state


def test_a_matching_interpreter_is_no_mismatch():
    assert dc.describe_interpreter_mismatch(_identity_state(1)) is None


def test_a_state_file_from_an_older_daemon_carries_no_identity():
    """Nothing to compare is not a mismatch; the protocol check catches it."""
    assert dc.describe_interpreter_mismatch({"port": 1, "token": "t"}) is None


@pytest.mark.parametrize("field", ["python", "prefix", "algan_path", "algan_version"])
def test_each_identity_field_is_compared(field):
    message = dc.describe_interpreter_mismatch(
        _identity_state(1, **{field: "/somewhere/else"})
    )
    assert message is not None
    assert field in message


def test_a_mismatched_interpreter_names_both(monkeypatch):
    message = dc.describe_interpreter_mismatch(
        _identity_state(1, python="/other/venv/bin/python")
    )
    assert "/other/venv/bin/python" in message
    assert dc.sys.executable in message


def test_a_live_daemon_from_another_virtualenv_is_not_used(home, monkeypatch):
    """It would execute this script against the wrong site-packages."""
    daemon = _PingableDaemon()
    (home / "daemon.json").write_text(
        json.dumps(_identity_state(daemon.port, python="/other/venv/bin/python")),
        encoding="utf-8",
    )
    _no_spawn(monkeypatch)
    monkeypatch.setattr(
        dc, "run_remote", lambda *a, **k: pytest.fail("must not hand off")
    )
    try:
        assert dc._dispatch("scene.py") is None
    finally:
        daemon.close()
    assert (home / "daemon.json").exists(), "the live daemon keeps its registration"


def test_a_dead_registration_from_another_virtualenv_is_replaced(home, monkeypatch):
    """Nothing owns it, so it must not block auto-start for ever."""
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    state_file = home / "daemon.json"
    state_file.write_text(
        json.dumps(_identity_state(port, python="/other/venv/bin/python")),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALGAN_AUTO_DAEMON", "0")  # stop after the cleanup

    assert dc._dispatch("scene.py") is None
    assert not state_file.exists()


class _PingableDaemon:
    """A socket that answers one ``ping <token>`` with ``pong``."""

    def __init__(self):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.received = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        with conn:
            self.received = conn.recv(128)
            conn.sendall(b"pong\n")

    def close(self):
        self.thread.join(timeout=5)
        self.server.close()


def test_reachability_is_a_token_carrying_ping(home):
    daemon = _PingableDaemon()
    try:
        assert dc.is_reachable({"port": daemon.port, "token": "tok"})
    finally:
        daemon.close()
    assert daemon.received == b"ping tok\n"


def test_nothing_listening_is_not_reachable():
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    assert not dc.is_reachable({"port": port, "token": "tok"})


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
