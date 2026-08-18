"""Source-staleness detection: what makes the daemon stand down.

The daemon refuses to serve a run once algan's sources on disk no longer match
the modules it imported, so a render can never come out of stale code (see
``DESIGN_daemon_lifecycle.md``). These tests pin the detector itself against a
temporary tree; the refusal it drives is exercised in ``test_daemon_client.py``.
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
