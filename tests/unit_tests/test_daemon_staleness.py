"""What makes the daemon stand down: stale sources, and a run it was not needed for.

The daemon refuses to serve a run once algan's sources on disk no longer match
the modules it imported, so a render can never come out of stale code (see
``DESIGN_daemon_lifecycle.md``). These tests pin the detector itself against a
temporary tree; the refusal it drives is exercised in ``test_daemon_client.py``.

An auto-started daemon also exits after the run it was started for, if that
run finished without rendering anything; the last section pins that rule and
runs it once through a real daemon and client.
"""

from __future__ import annotations

import os

import pytest

# Importing the daemon marks this process as one -- it sets ALGAN_DAEMON_CHILD
# at import so that neither it nor the scripts it runs hand themselves to
# another daemon. Undone below so the flag cannot leak into other tests.
_PRE_EXISTING_CHILD_FLAG = os.environ.get("ALGAN_DAEMON_CHILD")
from algan import daemon as d  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_child_flag(monkeypatch):
    if _PRE_EXISTING_CHILD_FLAG is None:
        monkeypatch.delenv("ALGAN_DAEMON_CHILD", raising=False)
    else:
        monkeypatch.setenv("ALGAN_DAEMON_CHILD", _PRE_EXISTING_CHILD_FLAG)


@pytest.fixture
def tree(tmp_path):
    """A miniature source tree standing in for the algan package."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not source\n", encoding="utf-8")
    return tmp_path


def capture(tree):
    return d._SourceDigest.capture(str(tree))


def test_an_unchanged_tree_reads_as_unchanged(tree):
    assert capture(tree).changed_since(capture(tree)) == []


def test_only_python_files_are_fingerprinted(tree):
    assert set(capture(tree).files) == {"a.py", "pkg/b.py"}


def test_edited_content_is_detected(tree):
    before = capture(tree)
    (tree / "pkg" / "b.py").write_text("y = 3\n", encoding="utf-8")
    assert capture(tree).changed_since(before) == ["pkg/b.py"]


def test_a_new_file_is_detected(tree):
    before = capture(tree)
    (tree / "pkg" / "c.py").write_text("z = 4\n", encoding="utf-8")
    assert capture(tree).changed_since(before) == ["pkg/c.py"]


def test_a_deleted_file_is_detected(tree):
    before = capture(tree)
    os.remove(tree / "a.py")
    assert capture(tree).changed_since(before) == ["a.py"]


def test_a_touched_but_identical_file_is_not_a_change(tree):
    """The reason this hashes content instead of stat'ing mtimes.

    ``git checkout``/``stash``/``rebase`` rewrite mtimes wholesale without
    changing a byte. An mtime-based gate would shut the daemon down and force a
    cold restart on every branch switch, including switching away and back.
    """
    before = capture(tree)
    target = tree / "a.py"
    stat = os.stat(target)
    os.utime(target, ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))
    assert os.stat(target).st_mtime_ns != stat.st_mtime_ns
    assert capture(tree).changed_since(before) == []


def test_bytecode_caches_are_ignored(tree):
    before = capture(tree)
    cache = tree / "pkg" / "__pycache__"
    cache.mkdir()
    (cache / "b.cpython-311.py").write_text("compiled\n", encoding="utf-8")
    assert capture(tree).changed_since(before) == []


def test_an_unreadable_file_errs_toward_restarting(tree, monkeypatch):
    """A file we cannot hash must not silently read as unchanged."""
    before = capture(tree)
    real_open = open

    def explode(path, *args, **kwargs):
        if str(path).endswith("a.py"):
            raise OSError(13, "denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", explode)
    assert capture(tree).changed_since(before) == ["a.py"]


# --------------------------------------------------------------------------
# The refusal text, which is what the user actually sees
# --------------------------------------------------------------------------


def test_the_message_names_the_changed_files():
    message = d._stale_message(["mobs/text.py"])
    assert "mobs/text.py" in message
    assert "fresh process" in message


def test_long_lists_are_summarised():
    message = d._stale_message([f"m{i}.py" for i in range(9)])
    assert "(+4 more)" in message


def test_a_kernel_edit_warns_about_the_recompile():
    plain = d._stale_message(["scene.py"])
    kernel = d._stale_message(["rendering/raytracing/raster_taichi.py"])
    assert "recompile" in kernel
    assert "recompile" not in plain


# --------------------------------------------------------------------------
# An auto-started daemon whose first run rendered nothing (issue 8)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_count", "code", "rendered", "waiting", "exits"),
    [
        (1, 0, False, 0, True),  # the run it was started for needed no renderer
        (1, 0, True, 0, False),  # it rendered: stay warm for the next one
        (1, 1, False, 0, False),  # it failed: the fixed retry is what it is for
        (1, 130, False, 0, False),  # cancelled
        (1, 2, False, 0, False),  # e.g. an argparse error, to be re-run
        (1, None, False, 0, False),  # cancelled before it started
        (1, 0, False, 1, False),  # another client is already waiting
        (2, 0, False, 0, False),  # it has served others: it is wanted
    ],
)
def test_when_an_auto_started_daemon_stands_down(
    run_count, code, rendered, waiting, exits
):
    assert d._started_for_nothing(run_count, code, rendered, waiting) is exits


def test_a_daemon_started_for_a_renderless_run_exits_after_it(tmp_path):
    """The real loop, a real client: the daemon leaves no process behind.

    The script stands in for one the client's parse took for rendering (it
    names ``save_video``) and that then did not render -- a ``--help``, a dry
    run. Without the flag the daemon would idle for ``--idle-timeout``.
    """
    import json
    import subprocess
    import sys
    import time

    home = tmp_path / "home"
    env = {
        **os.environ,
        "ALGAN_HOME": str(home),
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_DAEMON_CHILD": "1",
        "ALGAN_PRECOMPILE_JOBS": "0",
    }
    script = tmp_path / "dry_run.py"
    script.write_text(
        "import sys\nif '--render' in sys.argv:\n"
        "    from algan import Scene\n    Scene.save_video()\n"
        "print('dry run')\n",
        encoding="utf-8",
    )
    client = (
        "import json, sys\nfrom algan.daemon_client import run_remote\n"
        "state = json.load(open(sys.argv[1]))\n"
        "sys.exit(run_remote(state, sys.argv[2], argv=[]))\n"
    )
    with (tmp_path / "daemon.log").open("w") as log:
        daemon = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "algan.daemon",
                "--port",
                "0",
                "--idle-timeout",
                "600",
                "--exit-if-first-run-renders-nothing",
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        try:
            state_file = home / "daemon.json"
            deadline = time.monotonic() + 120
            while not state_file.exists():
                assert daemon.poll() is None, (tmp_path / "daemon.log").read_text()
                assert time.monotonic() < deadline, "the daemon never came up"
                time.sleep(0.05)
            assert json.loads(state_file.read_text())["port"] > 0
            ran = subprocess.run(
                [sys.executable, "-c", client, str(state_file), str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert ran.returncode == 0, ran.stderr
            assert "dry run" in ran.stdout
            assert daemon.wait(timeout=60) == 0
        finally:
            if daemon.poll() is None:
                daemon.kill()
                daemon.wait(timeout=10)
    assert not state_file.exists(), "a daemon that exits takes its registration"
    assert "rendered nothing" in (tmp_path / "daemon.log").read_text()
