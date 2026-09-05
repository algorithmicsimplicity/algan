"""The compiler shim hides only Quadrants warnings Algan knowingly triggers."""

import warnings

import algan.taichi_compat as taichi_compat

_WARNING_MODULE = "quadrants._test_tools.warnings_helper"
_DTYPE_WARNING = (
    "cannot create weak reference to 'DataTypeCxx' object. "
    "Template mapper caching disabled."
)
_TUPLE_WARNING = (
    "cannot create weak reference to 'tuple' object. "
    "Template mapper caching disabled."
)


def _warn_explicit(message, *, module=_WARNING_MODULE):
    warnings.warn_explicit(
        message,
        UserWarning,
        filename="quadrants/_test_tools/warnings_helper.py",
        lineno=11,
        module=module,
    )


def test_quadrants_filter_hides_only_the_two_known_template_cache_warnings(monkeypatch):
    monkeypatch.setattr(taichi_compat, "BACKEND", "quadrants")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        taichi_compat._install_backend_warning_filters()

        _warn_explicit(_DTYPE_WARNING)
        _warn_explicit(_TUPLE_WARNING)
        _warn_explicit(
            "cannot create weak reference to 'list' object. "
            "Template mapper caching disabled."
        )
        _warn_explicit(_DTYPE_WARNING, module="some_other_module")

    assert [str(item.message) for item in caught] == [
        "cannot create weak reference to 'list' object. Template mapper caching disabled.",
        _DTYPE_WARNING,
    ]


def test_taichi_backend_does_not_install_the_quadrants_filter(monkeypatch):
    monkeypatch.setattr(taichi_compat, "BACKEND", "taichi")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        taichi_compat._install_backend_warning_filters()
        _warn_explicit(_DTYPE_WARNING)

    assert [str(item.message) for item in caught] == [_DTYPE_WARNING]
