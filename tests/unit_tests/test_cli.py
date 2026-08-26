"""The ``algan`` command line: what each flag actually does to a run.

Every flag here is one a user can see in ``--help``, so the rule these tests
enforce is that none of them is decorative. ``-q`` and ``-o`` reach the script
through ``SETTINGS`` in the process it runs in; ``--no-daemon`` reaches it as
the environment variable the handoff client reads; ``daemon --stop`` reaches a
running daemon through its trigger socket. Nothing renders: the scripts these
tests run report their settings and exit.
"""

from __future__ import annotations

import json
import socket
import sys
import threading

import pytest

import algan
from algan import SETTINGS, cli

# --------------------------------------------------------------------------
# Scripts that report what the CLI did to them, instead of rendering
# --------------------------------------------------------------------------


@pytest.fixture
def reporting_script(tmp_path):
    """A scene script that dumps the settings and argv it was given."""
    report = tmp_path / "report.json"
    script = tmp_path / "scene.py"
    script.write_text(
        "import json, sys\n"
        "from algan import SETTINGS\n"
        "json.dump(\n"
        "    {\n"
        '        "resolution": list(SETTINGS.video.resolution),\n'
        '        "output_root": SETTINGS.paths.output_root,\n'
        '        "output_directory": SETTINGS.paths.output_directory,\n'
        '        "output_filename": SETTINGS.paths.output_filename,\n'
        '        "argv": sys.argv,\n'
        "    },\n"
        f"    open({str(report)!r}, 'w'),\n"
        ")\n",
        encoding="utf-8",
    )
    assert report.parent == script.parent  # _report_of relies on this
    return script


def _report_of(script):
    return script.with_name("report.json")


def _reported(script):
    return json.loads(_report_of(script).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# -q: the preset the script renders at, when it does not name one
# --------------------------------------------------------------------------


def test_quality_reaches_the_script(reporting_script):
    # Not LD: that is the default preset, so a -q that did nothing at all would
    # still leave the script reporting it.
    assert algan.HD.resolution != SETTINGS.video.resolution
    assert cli.main(["render", str(reporting_script), "-q", "hd"]) == 0
    assert _reported(reporting_script)["resolution"] == list(algan.HD.resolution)


def test_preview_renders_at_the_preview_preset(reporting_script):
    """``algan preview`` used to raise AttributeError before rendering at all."""
    assert cli.main(["preview", str(reporting_script)]) == 0
    assert _reported(reporting_script)["resolution"] == list(algan.PREVIEW.resolution)


def test_every_advertised_quality_names_a_real_preset():
    """``--help`` lists these; each has to be a preset Algan actually has."""
    for name in cli.QUALITY_PRESETS:
        assert hasattr(algan, name.upper()), name


# --------------------------------------------------------------------------
# -o: where output goes, for the part of the path the script leaves open
# --------------------------------------------------------------------------


def test_a_directory_output_holds_for_a_script_that_names_its_video(tmp_path):
    """The whole point of setting the root: ``save_video("intro")`` lands in it."""
    settings = cli._output_settings(str(tmp_path / "renders"))
    assert settings == {
        "output_root": str(tmp_path / "renders"),
        "output_directory": "",
    }


def test_a_file_output_supplies_the_name_too(tmp_path):
    settings = cli._output_settings(str(tmp_path / "renders" / "final.mp4"))
    assert settings["output_root"] == str(tmp_path / "renders")
    assert settings["output_directory"] == ""
    assert settings["output_filename"] == "final.mp4", "the suffix is Algan's to honour"


def test_a_trailing_separator_is_a_directory(tmp_path):
    assert "output_filename" not in cli._output_settings(f"{tmp_path}/out.d/")


def test_output_reaches_the_script(reporting_script, tmp_path):
    destination = tmp_path / "renders" / "final.mp4"
    assert cli.main(["render", str(reporting_script), "-o", str(destination)]) == 0
    reported = _reported(reporting_script)
    assert reported["output_root"] == str(destination.parent)
    assert reported["output_filename"] == "final.mp4"


# --------------------------------------------------------------------------
# What a script gets when the CLI runs it in this process
# --------------------------------------------------------------------------


def test_output_defaults_follow_the_scene_not_the_cli(reporting_script):
    """Without this the defaults are the console script's own directory.

    ``SETTINGS.paths`` resolves its defaults from ``__main__`` at import, which
    for the ``algan`` command is the console-script wrapper in the environment's
    bin directory -- so a script that says ``Scene.save_video()`` would write
    its video in there.
    """
    assert cli.main(["render", str(reporting_script), "-q", "ld"]) == 0
    reported = _reported(reporting_script)
    assert reported["output_root"] == str(reporting_script.parent)
    assert reported["output_filename"] == reporting_script.stem


def test_the_script_gets_its_own_argv(reporting_script):
    assert cli.main(["render", str(reporting_script), "-q", "ld", "--", "-x", "7"]) == 0
    argv = _reported(reporting_script)["argv"]
    assert argv[0] == str(reporting_script)
    assert argv[-2:] == ["-x", "7"]


def test_the_cli_leaves_this_process_as_it_found_it(reporting_script):
    argv, path = list(sys.argv), list(sys.path)
    assert cli.main(["render", str(reporting_script), "-q", "ld"]) == 0
    assert sys.argv == argv
    assert sys.path == path


def test_a_scripts_exit_code_is_the_cli_s(tmp_path):
    script = tmp_path / "quits.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    assert cli.main(["render", str(script), "-q", "ld"]) == 3


def test_settings_are_not_left_behind_by_a_failed_run(tmp_path):
    """A script that raises must not take the CLI's cleanup down with it."""
    script = tmp_path / "boom.py"
    script.write_text("raise RuntimeError('scene is broken')\n", encoding="utf-8")
    argv = list(sys.argv)
    with pytest.raises(RuntimeError):
        cli.main(["render", str(script), "-q", "ld"])
    assert sys.argv == argv


def test_a_missing_script_is_an_error(tmp_path, capsys):
    assert cli.main(["render", str(tmp_path / "nope.py")]) == 1
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --no-daemon, and the plain run that keeps the daemon
# --------------------------------------------------------------------------


@pytest.fixture
def spawned(monkeypatch):
    """Capture the subprocess a flagless run launches, without running it."""
    calls = []

    def fake_call(cmd, env=None, **kwargs):
        calls.append({"cmd": cmd, "env": env or {}})
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    return calls


def test_a_plain_run_launches_the_script_as_its_own_process(reporting_script, spawned):
    """Which is what lets the render daemon serve it warm."""
    assert cli.main(["render", str(reporting_script)]) == 0
    assert spawned[0]["cmd"] == [sys.executable, str(reporting_script)]
    assert "ALGAN_USE_DAEMON" not in spawned[0]["env"]
    assert not _report_of(reporting_script).exists(), "the script must not run here"


def test_no_daemon_sets_the_variable_the_client_reads(reporting_script, spawned):
    assert cli.main(["render", str(reporting_script), "--no-daemon"]) == 0
    assert spawned[0]["env"]["ALGAN_USE_DAEMON"] == "0"


def test_the_no_daemon_variable_is_one_algan_declares():
    """A name nothing reads is the bug this flag had; keep it wired to the list."""
    from algan.environment import ALGAN_ENVIRONMENT_VARIABLES

    assert "ALGAN_USE_DAEMON" in ALGAN_ENVIRONMENT_VARIABLES


def test_extra_arguments_are_forwarded(reporting_script, spawned):
    assert cli.main(["render", str(reporting_script), "--", "-x", "7"]) == 0
    assert spawned[0]["cmd"][-2:] == ["-x", "7"]


def test_a_bare_script_argument_means_render(reporting_script, spawned):
    assert cli.main([str(reporting_script)]) == 0
    assert spawned[0]["cmd"] == [sys.executable, str(reporting_script)]


def test_a_bare_script_argument_still_takes_flags(reporting_script, spawned):
    assert cli.main([str(reporting_script), "--no-daemon"]) == 0
    assert spawned[0]["env"]["ALGAN_USE_DAEMON"] == "0"


# --------------------------------------------------------------------------
# The script's own command line -- Project.run_cli() reads one
# --------------------------------------------------------------------------


def test_unrecognized_arguments_go_to_the_script(reporting_script, spawned):
    """``algan render project.py --render-video intro`` has to keep working."""
    assert cli.main(["render", str(reporting_script), "--render-video", "0"]) == 0
    assert spawned[0]["cmd"][-2:] == ["--render-video", "0"]


def test_they_go_to_the_script_alongside_our_own_flags(reporting_script):
    """One command line, split between the two parsers that read it."""
    argv = ["render", str(reporting_script), "-q", "hd", "--render-video", "0"]
    assert cli.main(argv) == 0
    reported = _reported(reporting_script)
    assert reported["argv"][1:] == ["--render-video", "0"]
    assert reported["resolution"] == list(algan.HD.resolution), "ours applied too"


def test_a_flag_spelled_like_ours_gets_through_after_a_separator(
    reporting_script, spawned
):
    """``--`` is what a script whose own flag is ``-o`` needs."""
    assert cli.main(["render", str(reporting_script), "--", "-o", "mine"]) == 0
    assert spawned[0]["cmd"][-2:] == ["-o", "mine"], "the script's, not ours"


def test_an_abbreviation_of_our_flag_is_not_stolen_from_the_script(
    reporting_script, spawned
):
    """argparse would read --out as --output; a script may want its own."""
    assert cli.main(["render", str(reporting_script), "--out", "mine"]) == 0
    assert spawned[0]["cmd"][-2:] == ["--out", "mine"]


def test_a_typo_is_still_an_error_where_no_script_runs(tmp_path):
    """Only render and preview forward; everywhere else it is a mistake."""
    with pytest.raises(SystemExit) as exiting:
        cli.main(["new", str(tmp_path / "x.py"), "--frce"])
    assert exiting.value.code == 2


# --------------------------------------------------------------------------
# daemon --stop
# --------------------------------------------------------------------------


class _FakeDaemon:
    """A socket that accepts one line and remembers it."""

    def __init__(self):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.received = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.server.accept()
        with conn:
            self.received = conn.recv(64)
            conn.sendall(b"ok\n")

    def close(self):
        self.thread.join(timeout=5)
        self.server.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    return tmp_path


def _register(home, port, token="t"):
    (home / "daemon.json").write_text(
        json.dumps({"port": port, "token": token, "pid": 4242}), encoding="utf-8"
    )


def test_stop_says_so_when_nothing_is_running(home, capsys):
    assert cli.main(["daemon", "--stop"]) == 0
    assert "No Algan daemon is running." in capsys.readouterr().out


def test_stop_asks_a_live_daemon_to_quit(home, capsys):
    daemon = _FakeDaemon()
    _register(home, daemon.port)
    try:
        assert cli.main(["daemon", "--stop"]) == 0
    finally:
        daemon.close()
    assert daemon.received == b"quit\n"
    assert "4242" in capsys.readouterr().out, "say which process was stopped"


def test_stop_clears_a_dead_registration(home, capsys):
    """A killed daemon leaves one behind, and it defeats auto-start forever."""
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    _register(home, port)

    assert cli.main(["daemon", "--stop"]) == 0
    assert not (home / "daemon.json").exists()
    assert "cleared its registration" in capsys.readouterr().out


def test_the_daemon_command_does_not_pass_it_our_own_arguments(monkeypatch):
    """``daemon.main(None)`` would read "daemon" as a script to render."""
    from algan import daemon

    seen = []
    monkeypatch.setattr(daemon, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["daemon"]) == 0
    assert seen == [[]]


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------


def test_version_reports_the_installed_version(capsys):
    with pytest.raises(SystemExit) as exiting:
        cli.main(["--version"])
    assert exiting.value.code == 0
    assert algan.__version__ in capsys.readouterr().out


def test_new_scaffolds_a_runnable_script(tmp_path, capsys):
    target = tmp_path / "my_scene.py"
    assert cli.main(["new", str(target)]) == 0
    compile(target.read_text(encoding="utf-8"), str(target), "exec")


def test_check_reports_the_environment(capsys):
    assert cli.main(["check"]) == 0
    printed = capsys.readouterr().out
    assert "PyTorch" in printed
    assert "Taichi" in printed


def test_settings_survive_a_cli_run(reporting_script):
    """The CLI writes SETTINGS to steer a run; the run must not outlive it.

    (``tests/conftest.py`` restores SETTINGS around every test, so this asserts
    against the snapshot rather than the import-time values.)
    """
    before = SETTINGS.snapshot()
    assert cli.main(["render", str(reporting_script), "-q", "uhd"]) == 0
    assert SETTINGS.video.resolution == algan.UHD.resolution
    SETTINGS.restore(before)
    assert SETTINGS.video.resolution != algan.UHD.resolution
