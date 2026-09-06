"""The compiler shim hides only known-safe Quadrants warnings."""

import warnings

import algan.taichi_compat as taichi_compat


def test_quadrants_warning_filter_is_narrow(monkeypatch):
    monkeypatch.setattr(taichi_compat, "BACKEND", "quadrants")

    with warnings.catch_warnings():
        warnings.resetwarnings()
        taichi_compat._install_backend_warning_filters()
        entry = warnings.filters[0]

    assert entry[0] == "ignore"
    assert entry[1].pattern == taichi_compat._QUADRANTS_BENIGN_TEMPLATE_CACHE_WARNING
    assert entry[2] is UserWarning
    assert entry[3].pattern == r"quadrants\._test_tools\.warnings_helper"


def test_taichi_backend_does_not_install_quadrants_filter(monkeypatch):
    monkeypatch.setattr(taichi_compat, "BACKEND", "taichi")

    with warnings.catch_warnings():
        warnings.resetwarnings()
        taichi_compat._install_backend_warning_filters()
        assert not warnings.filters
