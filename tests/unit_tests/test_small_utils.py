"""Fast pure-Python tests for small helper modules under ``algan/utils``."""

from __future__ import annotations

import json
import sys
import types
import warnings

import pytest

from algan.errors import AlganConfigurationError, ApproximationWarning
from algan.utils import _launch_telemetry as telemetry
from algan.utils.api_renames import (
    _format_number,
    _looks_like_radians,
    _reject_renamed_keywords,
    _renamed_keywords,
    _warn_if_angle_looks_like_radians,
)
from algan.utils.lazy_import import LazyModule, isinstance_if_loaded
from algan.utils.singleton import Singleton

# --------------------------------------------------------------------------
# api_renames
# --------------------------------------------------------------------------


def test_reject_renamed_keywords_names_new_spelling():
    with pytest.raises(AlganConfigurationError, match="Algan spells it `mob`"):
        _reject_renamed_keywords("Create", {"mobject": 1})


def test_reject_renamed_keywords_passes_clean_kwargs():
    assert _reject_renamed_keywords("Create", {"mob": 1}) is None


def test_reject_renamed_keywords_manim_alternative_and_hint():
    with pytest.raises(AlganConfigurationError) as info:
        _reject_renamed_keywords(
            "Table",
            {"old": 1},
            {"old": "new"},
            manim_alternative="Table",
            hints={"old": "Extra hint."},
        )
    message = str(info.value)
    assert "algan.manim.Table" in message
    assert message.endswith("Extra hint.")


def test_reject_renamed_keywords_without_alternative_has_no_pointer():
    with pytest.raises(AlganConfigurationError) as info:
        _reject_renamed_keywords("f", {"element_to_mobject": 1})
    assert "algan.manim" not in str(info.value)


def test_renamed_keywords_decorator():
    @_renamed_keywords(colour="color")
    def paint(color=None):
        """Doc."""
        return color

    assert paint(color="red") == "red"
    assert paint.__name__ == "paint"
    assert paint.__doc__ == "Doc."
    with pytest.raises(AlganConfigurationError, match="paint\\(\\) has no `colour`"):
        paint(colour="red")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5707963, True),
        (-3.14159, True),
        (6.28, True),
        (6.3, False),
        (5, False),
        (5.0, False),
        (0.0, False),
        (True, False),
        (float("nan"), False),
        (float("inf"), False),
        ("1.5", False),
        (None, False),
    ],
)
def test_looks_like_radians(value, expected):
    assert _looks_like_radians(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(90.0, "90"), (89.9999999, "90"), (12.5, "12.5"), (-45.0, "-45")],
)
def test_format_number(value, expected):
    assert _format_number(value) == expected


def test_warn_if_angle_looks_like_radians_warns_with_degrees():
    import math

    with pytest.warns(ApproximationWarning, match="did you mean `angle=90`"):
        _warn_if_angle_looks_like_radians("angle", math.pi / 2)


def test_warn_if_angle_whole_number_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_angle_looks_like_radians("angle", 5)


# --------------------------------------------------------------------------
# lazy_import
# --------------------------------------------------------------------------


@pytest.fixture
def fake_module(monkeypatch):
    name = "_algan_fake_lazy_target"
    module = types.ModuleType(name)

    class Thing:
        pass

    module.Thing = Thing
    module.value = 42
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_lazy_module_defers_import(monkeypatch):
    lazy = LazyModule("_algan_module_that_does_not_exist")
    assert lazy._module is None
    with pytest.raises(ModuleNotFoundError):
        _ = lazy.anything


def test_lazy_module_loads_on_attribute_access(fake_module):
    lazy = LazyModule(fake_module.__name__)
    assert lazy.value == 42
    assert lazy._module is fake_module


def test_lazy_module_imports_extras(fake_module, monkeypatch):
    extra = types.ModuleType("_algan_fake_extra")
    monkeypatch.setitem(sys.modules, "_algan_fake_extra", extra)
    calls = []
    import importlib

    real = importlib.import_module

    def spy(name, *args, **kwargs):
        calls.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", spy)
    lazy = LazyModule(fake_module.__name__, extras=["_algan_fake_extra"])
    _ = lazy.value
    _ = lazy.value
    assert calls == [fake_module.__name__, "_algan_fake_extra"]


def test_lazy_module_missing_attribute(fake_module):
    lazy = LazyModule(fake_module.__name__)
    with pytest.raises(AttributeError):
        _ = lazy.missing


def test_isinstance_if_loaded_unloaded_is_false():
    lazy = LazyModule("_algan_module_that_does_not_exist")
    assert isinstance_if_loaded(object(), lazy, "Thing") is False
    assert lazy._module is None


def test_isinstance_if_loaded_in_sys_modules(fake_module):
    lazy = LazyModule(fake_module.__name__)
    assert isinstance_if_loaded(fake_module.Thing(), lazy, "Thing") is True
    assert isinstance_if_loaded(object(), lazy, "Thing") is False


# --------------------------------------------------------------------------
# singleton
# --------------------------------------------------------------------------


def _make_singleton():
    class Service(Singleton):
        @classmethod
        def _create(cls):
            return object.__new__(cls)

    return Service


def test_singleton_direct_construction_raises():
    service = _make_singleton()
    with pytest.raises(RuntimeError, match=r"Call Service.instance\(\)"):
        service()


def test_singleton_instance_is_cached_and_reset():
    service = _make_singleton()
    first = service.instance()
    assert service.instance() is first
    service.reset()
    assert service.instance() is not first


def test_singleton_subclasses_do_not_share_instances():
    a = _make_singleton()
    b = _make_singleton()
    assert a.instance() is not b.instance()
    assert isinstance(b.instance(), b)


def test_singleton_base_create_not_implemented():
    class Bare(Singleton):
        pass

    with pytest.raises(NotImplementedError):
        Bare.instance()


# --------------------------------------------------------------------------
# _launch_telemetry
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_telemetry(monkeypatch):
    monkeypatch.setattr(telemetry, "_ROWS", {})
    monkeypatch.setattr(telemetry, "ENABLED", False)
    monkeypatch.setattr(telemetry, "_CACHE_CLASS", None)
    monkeypatch.setattr(telemetry, "_LOCAL", types.SimpleNamespace())
    return telemetry


class _FakeCache:
    def __init__(self, hit=False):
        self.hit = hit

    def populate_launch_ctx_from_cache(self, *args, **kwargs):
        return self.hit


def _kernel_func():
    pass


def _make_kernel(cache=None, arch="cpu"):
    return types.SimpleNamespace(
        func=_kernel_func,
        runtime=types.SimpleNamespace(_arch=arch),
        launch_context_buffer_cache=cache,
    )


def _make_cache_class():
    class Cache(_FakeCache):
        pass

    return Cache


def test_set_enabled_requires_configured_cache(fresh_telemetry):
    with pytest.raises(RuntimeError, match="fast dispatcher"):
        fresh_telemetry.set_enabled(True)


def test_set_enabled_false_without_cache_is_allowed(fresh_telemetry):
    fresh_telemetry.set_enabled(False)
    assert fresh_telemetry.ENABLED is False


def test_set_enabled_installs_observer_once(fresh_telemetry):
    cache_class = _make_cache_class()
    fresh_telemetry.configure(cache_class)
    fresh_telemetry.set_enabled(True)
    wrapped = cache_class.populate_launch_ctx_from_cache
    fresh_telemetry.set_enabled(True)
    assert cache_class._algan_observed is True
    assert cache_class.populate_launch_ctx_from_cache is wrapped
    assert fresh_telemetry.ENABLED is True


def test_report_empty(fresh_telemetry):
    result = fresh_telemetry.report()
    assert result["kernels"] == []
    assert result["totals"] == dict.fromkeys(fresh_telemetry._OUTCOMES, 0)


def test_fast_and_error_counts(fresh_telemetry):
    kernel = _make_kernel()
    fresh_telemetry.fast(kernel)
    fresh_telemetry.fast(kernel)
    fresh_telemetry.fast(kernel, error=True)
    (row,) = fresh_telemetry.report()["kernels"]
    assert row["kernel"].endswith("_kernel_func")
    assert row["arch"] == "cpu"
    assert row["fast"] == 2
    assert row["error"] == 1
    assert row["reasons"] == {"fast_launch": 1}


def test_original_defaults_to_fallback_with_reason(fresh_telemetry):
    kernel = _make_kernel(cache=object())
    with fresh_telemetry.original(kernel, "unsupported"):
        pass
    result = fresh_telemetry.report()
    assert result["totals"]["fallback"] == 1
    assert result["kernels"][0]["reasons"] == {"unsupported": 1}


def test_original_cache_hit_becomes_quadrants_cache(fresh_telemetry):
    cache_class = _make_cache_class()
    fresh_telemetry.configure(cache_class)
    fresh_telemetry.set_enabled(True)
    cache = cache_class(hit=True)
    kernel = _make_kernel(cache=cache)
    with fresh_telemetry.original(kernel, "first"):
        assert cache.populate_launch_ctx_from_cache() is True
    row = fresh_telemetry.report()["kernels"][0]
    assert row["quadrants_cache"] == 1
    assert row["fallback"] == 0
    assert row["reasons"] == {}


def test_original_cold_outcome_wins_over_cache_hit(fresh_telemetry):
    cache_class = _make_cache_class()
    fresh_telemetry.configure(cache_class)
    fresh_telemetry.set_enabled(True)
    cache = cache_class(hit=True)
    kernel = _make_kernel(cache=cache)
    with fresh_telemetry.original(kernel, "first") as event:
        cache.populate_launch_ctx_from_cache()
        event["outcome"] = "cold"
    row = fresh_telemetry.report()["kernels"][0]
    assert row["cold"] == 1
    assert row["quadrants_cache"] == 0


def test_original_hit_ignored_when_disabled(fresh_telemetry):
    cache_class = _make_cache_class()
    fresh_telemetry.configure(cache_class)
    fresh_telemetry.set_enabled(True)
    fresh_telemetry.set_enabled(False)
    cache = cache_class(hit=True)
    kernel = _make_kernel(cache=cache)
    with fresh_telemetry.original(kernel, "r"):
        cache.populate_launch_ctx_from_cache()
    assert fresh_telemetry.report()["totals"]["fallback"] == 1


def test_original_exception_records_error(fresh_telemetry):
    kernel = _make_kernel(cache=object())

    def run():
        with fresh_telemetry.original(kernel, "why"):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run()
    row = fresh_telemetry.report()["kernels"][0]
    assert row["error"] == 1
    assert row["reasons"] == {"why": 1}
    assert fresh_telemetry._LOCAL.stack == []


def test_report_reset_clears_and_is_json(fresh_telemetry):
    fresh_telemetry.fast(_make_kernel(arch="cuda"))
    fresh_telemetry.fast(_make_kernel(arch="cpu"))
    result = fresh_telemetry.report(reset=True)
    json.dumps(result)
    assert [row["arch"] for row in result["kernels"]] == ["cpu", "cuda"]
    assert result["totals"]["fast"] == 2
    assert fresh_telemetry.report()["kernels"] == []


# --------------------------------------------------------------------------
# plotting_utils
# --------------------------------------------------------------------------


class _FakeAxes:
    def __init__(self, log):
        self.log = log

    def set_axis_off(self):
        pass

    def imshow(self, img, **kwargs):
        self.log.append((tuple(img.shape), kwargs))


class _FakeFigure:
    def __init__(self, log):
        self.log = log

    def add_axes(self, rect):
        return _FakeAxes(self.log)


@pytest.fixture
def fake_plt(monkeypatch):
    from algan.utils import plotting_utils

    log = []
    fake = types.SimpleNamespace(
        figure=lambda figsize: log.append(("figsize", figsize)) or _FakeFigure(log),
        savefig=lambda f, **kwargs: open(f, "w").close(),
        clf=lambda: None,
        close=lambda: None,
    )
    monkeypatch.setattr(plotting_utils, "plt", fake)
    return plotting_utils.plot_tensor, log


def test_plot_tensor_saves_with_unique_suffix(tmp_path, fake_plt):
    torch = pytest.importorskip("torch")
    plot_tensor, log = fake_plt
    base = tmp_path / "img"
    x = torch.rand(3, 8, 4)
    plot_tensor(x, save_file=str(base))
    plot_tensor(x[0], save_file=str(base))
    plot_tensor(x.unsqueeze(0), save_file=str(base))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["img.pdf", "img_0.pdf", "img_1.pdf"]
    assert log[0] == ("figsize", (8 * 3 / 100, 4 * 3 / 100))
    assert log[1][0] == (8, 4, 3)
    assert log[3][0] == (8, 4)


def test_plot_tensor_normalises_to_unit_range(tmp_path, fake_plt):
    torch = pytest.importorskip("torch")
    plot_tensor, log = fake_plt
    seen = []
    x = torch.tensor([[[2.0, 4.0], [6.0, 10.0]]])
    original = _FakeAxes.imshow
    try:
        _FakeAxes.imshow = lambda self, img, **kw: seen.append(img.clone())
        plot_tensor(x, save_file=str(tmp_path / "n"))
    finally:
        _FakeAxes.imshow = original
    assert float(seen[0].min()) == 0.0
    assert float(seen[0].max()) == 1.0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: norm=True divides by amax() of a constant image, producing NaN",
)
def test_plot_tensor_constant_image_not_nan(tmp_path, fake_plt):
    torch = pytest.importorskip("torch")
    plot_tensor, _ = fake_plt
    seen = []
    original = _FakeAxes.imshow
    try:
        _FakeAxes.imshow = lambda self, img, **kw: seen.append(img.clone())
        plot_tensor(torch.ones(1, 2, 2), save_file=str(tmp_path / "c"))
    finally:
        _FakeAxes.imshow = original
    assert not bool(torch.isnan(seen[0]).any())
