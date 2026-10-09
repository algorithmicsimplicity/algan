"""Quadrants' template-mapper warnings: what Algan avoids, and what it hides.

The mapper caches a launch's specialization key against weak references to its
arguments and warns "Template mapper caching disabled" for any it cannot weakly
reference. Algan passes dtypes through ``taichi_compat.template_dtype`` so they
can be, and filters the one message it accepts -- the fragment-pipeline tuple.
"""

import re
import warnings
import weakref

import pytest

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


def _mapper_message(value):
    """The text Quadrants' mapper warns with when it cannot weakly reference ``value``.

    Built the way ``_template_mapper.TemplateMapper.lookup`` builds it, from the
    real ``TypeError`` -- the first version of the filter was written against a
    guessed message and never matched the dtype one, whose type name is
    module-qualified.
    """
    try:
        weakref.ref(value)
    except TypeError as exc:
        return f"{exc}. Template mapper caching disabled."
    pytest.fail(f"{value!r} is weakly referenceable")


def _is_hidden(message):
    pattern = re.compile(taichi_compat._QUADRANTS_BENIGN_TEMPLATE_CACHE_WARNING, re.I)
    return pattern.match(message) is not None


@pytest.mark.skipif(taichi_compat.BACKEND != "quadrants", reason="Quadrants only")
def test_the_filter_hides_the_tuple_message_and_nothing_else():
    assert _is_hidden(_mapper_message(()))
    assert not _is_hidden(_mapper_message(taichi_compat.ti.f64)), (
        "a raw dtype is avoidable (template_dtype), so it must stay visible"
    )
    assert not _is_hidden(_mapper_message(None))


@pytest.mark.skipif(taichi_compat.BACKEND != "quadrants", reason="Quadrants only")
def test_the_filter_is_in_force_in_a_test():
    """pytest runs each test under its own filters, not the ones set at import.

    ``tests/conftest.py`` hands it the same filter. Without that, the import-time
    filter was gone by the time any test rendered, and the tuple warning showed
    in the fast suite's output.
    """
    assert any(
        entry[0] == "ignore"
        and entry[1] is not None
        and entry[1].pattern == taichi_compat._QUADRANTS_BENIGN_TEMPLATE_CACHE_WARNING
        for entry in warnings.filters
    )


@pytest.mark.skipif(taichi_compat.BACKEND != "quadrants", reason="Quadrants only")
@pytest.mark.parametrize("name", ["f64", "f32", "i64", "i32", "u8"])
def test_a_template_dtype_is_the_same_dtype_and_weakly_referenceable(name):
    from algan.rendering.kernel_precompile import _decode, _encode
    from algan.utils.taichi_fast_launch import _template_key_supported
    from algan.utils.taichi_source_key import _dtype_name

    raw = getattr(taichi_compat.ti, name)
    handle = taichi_compat.template_dtype(raw)

    assert weakref.ref(handle)() is handle
    # Everything that keys a specialization on the dtype keys it the same way.
    assert handle == raw
    assert hash(handle) == hash(raw)
    assert handle.to_string() == raw.to_string() == name
    assert (1, handle) == (1, raw), "the mapper's key is a tuple of these"
    assert _dtype_name(handle) == _dtype_name(raw), "the source key"
    assert _encode(handle) == _encode(raw), "the precompile spec"
    assert _decode(_encode(raw)) is handle
    assert _template_key_supported(handle, nested=True), "the fast launcher's key"
    # One handle per dtype for the process: the mapper's entry is keyed by id.
    assert taichi_compat.template_dtype(raw) is handle
    assert taichi_compat.template_dtype(handle) is handle


def test_template_dtype_leaves_other_values_alone():
    for value in (None, 3, "f64", (1, 2)):
        assert taichi_compat.template_dtype(value) is value


def test_the_accumulate_and_index_dtypes_are_template_dtypes():
    from algan.rendering.mps_compat import (
        taichi_accumulate_dtype,
        taichi_reduction_index_dtype,
    )

    for dtype in (taichi_accumulate_dtype(), taichi_reduction_index_dtype()):
        assert taichi_compat.template_dtype(dtype) is dtype
        if taichi_compat.BACKEND == "quadrants":
            weakref.ref(dtype)


@pytest.mark.skipif(taichi_compat.BACKEND != "quadrants", reason="Quadrants only")
def test_a_3d_mesh_render_gives_the_template_mapper_nothing_to_warn_about(
    tmp_path, monkeypatch
):
    """Renders a ``Cube`` with every launch going through the template mapper.

    The fast launcher is switched off so the mapper sees every launch rather
    than each variant's first, and Quadrants' warn-once memory is cleared, so
    the outcome does not depend on what ran earlier in the process. Every
    template argument has to be weakly referenceable except the fragment
    pipeline tuple, and no "caching disabled" warning may get past Algan's
    filters.
    """
    import algan
    from algan.utils import taichi_fast_launch
    from algan.utils.taichi_source_key import _dtype_name

    mapper_module = taichi_compat.submodule("lang._template_mapper")
    helper = taichi_compat.submodule("_test_tools.warnings_helper")
    original = mapper_module.TemplateMapper.lookup
    primitive = mapper_module._primitive_types
    dtype_args = []
    unreferenceable = set()

    def lookup(self, raise_on_templated_floats, args):
        for arg in args:
            if type(arg) in primitive:
                continue
            if _dtype_name(arg) is not None:
                dtype_args.append(type(arg).__name__)
            try:
                weakref.ref(arg)
            except TypeError:
                unreferenceable.add(type(arg).__name__)
        return original(self, raise_on_templated_floats, args)

    monkeypatch.setattr(mapper_module.TemplateMapper, "lookup", lookup)
    monkeypatch.setattr(taichi_fast_launch, "ENABLED", False)
    monkeypatch.setattr(helper, "_seen", set())

    tiny = algan.VideoSettings((32, 32), 2, supersampling=1)
    with (
        warnings.catch_warnings(record=True) as caught,
        algan.Scene(video_settings=tiny) as scene,
    ):
        algan.Cube().spawn()
        scene.save_frame(str(tmp_path / "cube.png"))

    assert dtype_args, "no dtype template argument reached the mapper; test is vacuous"
    assert unreferenceable <= {"tuple"}, unreferenceable
    shown = [
        str(w.message)
        for w in caught
        if "Template mapper caching disabled" in str(w.message)
    ]
    assert not shown, shown
