"""What the daemon may evict from ``sys.modules`` between runs.

The daemon reloads the user's own modules so an edit is picked up by the next
run. Installed packages are not the user's modules, and the distinction is not
cosmetic: the installation tutorial builds the virtual environment *inside* the
project folder, so ``site-packages`` sits under the script directory. Evicting
torch from there poisons the daemon permanently -- torch cannot be imported
twice in one process -- and every later run died with an exit code and no
output at all.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import runpy
import sys
import sysconfig
import types

import pytest

from algan import daemon as d


def _fake_module(name, path):
    module = types.ModuleType(name)
    module.__file__ = path
    return module


def test_user_module_under_script_dir_is_evicted(tmp_path, monkeypatch):
    script_dir = tmp_path / "alganimations"
    script_dir.mkdir()
    helper = script_dir / "helper.py"
    helper.write_text("")
    monkeypatch.setitem(
        sys.modules, "algan_test_helper", _fake_module("algan_test_helper", str(helper))
    )
    assert "algan_test_helper" in d._user_modules(str(script_dir))


def test_site_packages_under_script_dir_is_not_evicted(monkeypatch):
    """The tutorial's layout: ``.venv`` inside the project folder."""
    site_packages = sysconfig.get_paths()["purelib"]
    # The project folder any script in it would report -- an ancestor of
    # site-packages exactly as `mkdir proj && cd proj && python -m venv .venv`
    # leaves it.
    script_dir = os.path.dirname(os.path.dirname(sys.prefix))
    installed = os.path.join(site_packages, "torch", "__init__.py")
    monkeypatch.setitem(
        sys.modules, "algan_test_torch", _fake_module("algan_test_torch", installed)
    )
    assert "algan_test_torch" not in d._user_modules(script_dir)


def test_real_torch_survives_a_reset_from_the_venv_parent():
    """The exact module the poisoned daemon evicted, at its real path."""
    if "torch" not in sys.modules:
        pytest.skip("torch is not imported in this process")
    script_dir = os.path.dirname(os.path.dirname(sys.prefix))
    assert "torch" not in d._user_modules(script_dir)


def test_stdlib_is_never_evicted():
    stdlib = sysconfig.get_paths()["stdlib"]
    names = d._user_modules(os.path.dirname(stdlib))
    assert "os" not in names
    assert "json" not in names


def test_external_helpers_and_circular_imports_are_fresh_after_rapid_edit(
    tmp_path, monkeypatch
):
    """Same-size edits in the same timestamp tick must bypass stale .pyc files."""
    script_dir = tmp_path / "video"
    helpers = tmp_path / "shared"
    script_dir.mkdir()
    helpers.mkdir()
    common = helpers / "algan_test_common.py"
    common.write_text("VALUE = 1\nimport algan_test_cycle\n", encoding="utf-8")
    (helpers / "algan_test_cycle.py").write_text(
        "from algan_test_common import VALUE\n", encoding="utf-8"
    )
    script = script_dir / "main.py"
    script.write_text("from algan_test_cycle import VALUE\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(helpers))
    protected = frozenset(sys.modules)
    before = common.stat()
    try:
        assert runpy.run_path(str(script))["VALUE"] == 1
        evicted = d._evict_user_modules(protected)
        assert {"algan_test_common", "algan_test_cycle"} <= set(evicted)
        common.write_text("VALUE = 2\nimport algan_test_cycle\n", encoding="utf-8")
        os.utime(common, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert runpy.run_path(str(script))["VALUE"] == 2
        assert sys.modules["algan"] is d.algan
    finally:
        d._evict_user_modules(protected)


def test_editable_startup_dependencies_and_their_lazy_modules_stay_loaded(
    tmp_path, monkeypatch
):
    name = "algan_test_compiler"
    monkeypatch.setitem(
        sys.modules, name, _fake_module(name, str(tmp_path / "__init__.py"))
    )
    protected = frozenset(sys.modules)
    monkeypatch.setitem(
        sys.modules,
        name + ".lazy",
        _fake_module(name + ".lazy", str(tmp_path / "lazy.py")),
    )
    assert name + ".lazy" not in d._user_modules(protected=protected)


def test_algan_aliases_and_extension_modules_are_not_evicted(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "algan_test_alias", _fake_module("algan_test_alias", d.__file__)
    )
    monkeypatch.setitem(
        sys.modules,
        "algan_test_native",
        _fake_module("algan_test_native", str(tmp_path / "native.pyd")),
    )
    assert "algan_test_alias" not in d._user_modules()
    assert "algan_test_native" not in d._user_modules()


def test_retained_namespace_package_does_not_keep_stale_child(tmp_path, monkeypatch):
    package = tmp_path / "algan_test_namespace"
    package.mkdir()
    (package / "common.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    protected = frozenset(sys.modules)
    namespace = importlib.import_module("algan_test_namespace")
    try:
        old = importlib.import_module("algan_test_namespace.common")
        d._evict_user_modules(protected)
        assert not hasattr(namespace, "common")
        new = importlib.import_module("algan_test_namespace.common")
        assert new is not old
    finally:
        # A namespace deliberately protected by this test should be retained,
        # while a real daemon owns and releases its user namespaces too.
        sys.modules.pop("algan_test_namespace.common", None)
        sys.modules.pop("algan_test_namespace", None)


def test_daemon_runs_external_helpers_fresh_and_watches_them(tmp_path, monkeypatch):
    script_dir = tmp_path / "video"
    helpers = tmp_path / "shared"
    script_dir.mkdir()
    helpers.mkdir()
    helper = helpers / "algan_test_external.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    script = script_dir / "main.py"
    script.write_text("from algan_test_external import VALUE\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(helpers))
    watched, values = [], []
    events = None

    class Watcher:
        def __init__(self, queue):
            pass

        def set_paths(self, paths):
            watched.append(paths)

    def stdin(queue):
        nonlocal events
        events = queue

    original = runpy.run_path

    def run(path, **kwargs):
        result = original(path, **kwargs)
        values.append(result["VALUE"])
        if len(values) == 1:
            helper.write_text("VALUE = 2\n", encoding="utf-8")
            events.put(("render", "test edit"))
        else:
            events.put(("quit", "test complete"))
        return result

    monkeypatch.setattr(d, "_start_stdin", stdin)
    monkeypatch.setattr(d, "_Watcher", Watcher)
    monkeypatch.setattr(d, "_release_run_memory", lambda: None)
    monkeypatch.setattr(d._StateFile, "remove", lambda self: None)
    monkeypatch.setattr(d.runpy, "run_path", run)
    assert d.main([str(script), "--no-serve", "--watch"]) == 0
    assert values == [1, 2]
    assert all(str(helper) in paths for paths in watched)
    assert "algan_test_external" not in sys.modules


def test_packages_with_compiled_extensions_are_kept_whole(tmp_path, monkeypatch):
    """Re-running a package's Python half against its old extension is unsafe."""
    package = tmp_path / "algan_test_mixed"
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    for name, file in (
        ("algan_test_mixed", package / "__init__.py"),
        ("algan_test_mixed.helpers", package / "helpers.py"),
        ("algan_test_mixed._native", package / f"_native{suffix}"),
    ):
        monkeypatch.setitem(sys.modules, name, _fake_module(name, str(file)))
    helper = tmp_path / "algan_test_plain.py"
    monkeypatch.setitem(
        sys.modules, "algan_test_plain", _fake_module("algan_test_plain", str(helper))
    )
    kept = set()
    names = d._user_modules(kept=kept)
    assert "algan_test_plain" in names
    assert not [name for name in names if name.startswith("algan_test_mixed")]
    assert "algan_test_mixed" in kept
