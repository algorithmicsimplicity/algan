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

import os
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
