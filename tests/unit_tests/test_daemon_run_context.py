"""Run parity: a script served by the daemon must behave as its own process.

With auto-start, even a user's first ``python scene.py`` runs on a daemon, so
anything the daemon fails to reproduce becomes a silent behaviour change rather
than a known cost of opting in. What is reproduced -- and what deliberately is
not -- is pinned here; see ``DESIGN_daemon_lifecycle.md`` §5.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PRE_EXISTING_CHILD_FLAG = os.environ.get("ALGAN_DAEMON_CHILD")
from algan import daemon as d  # noqa: E402
from algan import daemon_client as dc  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_child_flag(monkeypatch):
    if _PRE_EXISTING_CHILD_FLAG is None:
        monkeypatch.delenv("ALGAN_DAEMON_CHILD", raising=False)
    else:
        monkeypatch.setenv("ALGAN_DAEMON_CHILD", _PRE_EXISTING_CHILD_FLAG)


class _FakeJob:
    """Stands in for a client's connection, collecting what it would be sent."""

    def __init__(self, env=None, isatty_out=False, isatty_err=False):
        self.env = env
        self.isatty_out = isatty_out
        self.isatty_err = isatty_err
        self.frames = []

    def send(self, kind, payload=b""):
        self.frames.append((kind, payload))

    def text(self, kind):
        return "".join(p for k, p in self.frames if k == kind)


def test_python_level_output_reaches_the_client():
    job = _FakeJob()
    with d._run_context(job):
        print("from print")
        print("to stderr", file=sys.stderr)
    assert "from print" in job.text(dc.FRAME_STDOUT)
    assert "to stderr" in job.text(dc.FRAME_STDERR)


def test_descriptor_level_output_reaches_the_client():
    """The bug this replaced: only ``sys.stdout`` used to be redirected."""
    job = _FakeJob()
    with d._run_context(job):
        os.write(1, b"raw fd 1\n")
        os.write(2, b"raw fd 2\n")
    assert "raw fd 1" in job.text(dc.FRAME_STDOUT)
    assert "raw fd 2" in job.text(dc.FRAME_STDERR)


def test_subprocess_output_reaches_the_client():
    """ffmpeg, via moviepy, is the case that actually matters."""
    job = _FakeJob()
    with d._run_context(job):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print('child out'); print('child err', file=sys.stderr)",
            ],
            check=True,
        )
    assert "child out" in job.text(dc.FRAME_STDOUT)
    assert "child err" in job.text(dc.FRAME_STDERR)


def test_multibyte_output_survives_chunk_boundaries():
    job = _FakeJob()
    payload = "π≈3.14159 " * 5000
    with d._run_context(job):
        sys.stdout.write(payload)
        sys.stdout.flush()
    assert job.text(dc.FRAME_STDOUT).count("π≈3.14159") == 5000


@pytest.mark.parametrize("isatty", [True, False])
def test_the_clients_tty_ness_is_what_the_script_sees(isatty):
    """A pipe is not a tty; tqdm's progress bars depend on this lie."""
    job = _FakeJob(isatty_out=isatty, isatty_err=isatty)
    with d._run_context(job):
        seen_out = sys.stdout.isatty()
        seen_err = sys.stderr.isatty()
    assert (seen_out, seen_err) == (isatty, isatty)


def test_streams_are_restored_afterwards():
    before = (sys.stdout, sys.stderr, sys.stdin)
    with d._run_context(_FakeJob()):
        pass
    assert (sys.stdout, sys.stderr, sys.stdin) == before


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def test_the_clients_environment_is_applied_and_restored(monkeypatch):
    monkeypatch.setenv("ALGAN_TEST_DAEMON_OWN", "daemon-value")
    job = _FakeJob(env={"ALGAN_TEST_FROM_CLIENT": "client-value"})
    with d._run_context(job):
        assert os.environ.get("ALGAN_TEST_FROM_CLIENT") == "client-value"
        assert "ALGAN_TEST_DAEMON_OWN" not in os.environ
    assert os.environ.get("ALGAN_TEST_DAEMON_OWN") == "daemon-value"
    assert "ALGAN_TEST_FROM_CLIENT" not in os.environ


def test_the_child_marker_survives_the_environment_swap():
    """A python subprocess started by the script must not hand itself to us.

    It would queue behind the very run that spawned it, and wait forever.
    """
    job = _FakeJob(env={"PATH": os.environ.get("PATH", "")})
    with d._run_context(job):
        assert os.environ["ALGAN_DAEMON_CHILD"] == "1"


def test_a_local_run_keeps_the_daemons_environment(monkeypatch):
    """``None`` means an Enter-triggered re-run: there is no client to adopt."""
    monkeypatch.setenv("ALGAN_TEST_DAEMON_OWN", "daemon-value")
    with d._run_context(None):
        assert os.environ.get("ALGAN_TEST_DAEMON_OWN") == "daemon-value"


# --------------------------------------------------------------------------
# Where the render lands
# --------------------------------------------------------------------------


def test_output_defaults_follow_the_script_being_run(tmp_path):
    """A daemon resolves these once, at its own startup, with no user script.

    Left alone that sends every client's video to the daemon's own directory,
    named ``algan_render_output`` rather than after the script -- the most
    visible way a daemon-served run can differ from its own process.
    """
    from algan.settings.path_settings import output_filename_for, output_root_for

    script = tmp_path / "my_scene.py"
    assert output_root_for(str(script)) == str(tmp_path)
    assert output_filename_for(str(script)) == "my_scene"


def test_output_defaults_fall_back_without_a_script():
    from algan.settings.path_settings import output_filename_for, output_root_for

    assert output_root_for(None) == os.getcwd()
    assert output_filename_for(None) == "algan_render_output"


# --------------------------------------------------------------------------
# stdin, which is deliberately *not* reproduced
# --------------------------------------------------------------------------


def test_stdin_is_isolated_from_the_trigger():
    """The daemon's own stdin is its re-render trigger, so a run cannot have it.

    A script reading stdin sees EOF, which is what ``stdin=DEVNULL`` gives a
    subprocess -- not the daemon's next trigger line.
    """
    for job in (_FakeJob(), None):
        with d._run_context(job):
            assert sys.stdin.read() == ""
