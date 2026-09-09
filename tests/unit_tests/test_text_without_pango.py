"""``Text`` must reach its LaTeX fallback when manimpango is absent.

On Linux the ``pango`` extra is opt-in, and both the installation guide and
``algan check`` promise that ``Text`` still typesets -- through LaTeX text mode
-- without it. It did not: ``slant`` and ``weight`` have string defaults, so
every construction normalized them against Pango's name list and died on the
import before any fallback could run.
"""

from __future__ import annotations

import builtins

import pytest

from algan.mobs import text as text_module


@pytest.fixture
def no_manimpango(monkeypatch):
    """Make ``import manimpango`` fail, whether or not it is installed."""
    monkeypatch.setattr(text_module, "_PANGO_NAMES", {})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "manimpango" or name.startswith("manimpango."):
            raise ImportError("No module named 'manimpango'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield
    text_module._PANGO_NAMES.clear()


def test_pango_names_are_empty_without_manimpango(no_manimpango):
    assert text_module._pango_names("slant") == ()
    assert text_module._pango_names("weight") == ()


def test_defaults_do_not_raise_without_manimpango(no_manimpango):
    assert text_module._pango_style("slant", "NORMAL") == "NORMAL"
    assert text_module._pango_style("weight", "NORMAL") == "NORMAL"


def test_case_is_still_normalized_without_manimpango(no_manimpango):
    assert text_module._pango_style("weight", "bold") == "BOLD"


def test_unvalidatable_name_is_accepted_without_manimpango(no_manimpango):
    """Nothing can honour it in LaTeX text mode, and nothing knows it is wrong."""
    assert text_module._pango_style("weight", "BOLDER") == "BOLDER"


def test_bad_name_still_rejected_when_manimpango_is_present():
    pytest.importorskip("manimpango")
    from algan.errors import AlganConfigurationError

    with pytest.raises(AlganConfigurationError):
        text_module._pango_style("weight", "BOLDER")
