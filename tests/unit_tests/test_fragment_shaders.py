"""Functional tests for custom & composable fragment shaders
(:meth:`algan.mobs.mob.Mob.set_fragment_shader`).

These assert registry bookkeeping, parameter registration, pipeline
composition, and the set-after-spawn guard. Visual execution of a composed
pipeline is covered by ``tests/full_renders/scenes/media_and_shaders.py``.

    .venv/Scripts/python.exe -m pytest tests/unit_tests/test_fragment_shaders.py
"""

import pytest

import algan
from algan import SceneManager, Sphere, cosine_color, phong_shader
from algan.animatable_base.mob import ModifiedProtectedAttributeError
from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE
from algan.rendering.shaders.fragment_shaders import (
    STAGE_PHONG,
    build_frag_pipelines,
    build_fragment_pipeline,
    resolve_stage,
)

#: TEMPORARY. Registering a fragment pipeline appends to
#: ``fragment_shaders._PIPELINE_LIST``, which is **process-global and
#: append-only**, and ``build_frag_pipelines()`` hands that whole list to the
#: shade kernel as a ``ti.template()`` tuple. Taichi specialises on it, so an
#: empty tuple and a non-empty one are different kernels: once any test in a
#: process has registered one pipeline, every later render in that process --
#: including scenes with no custom shader at all -- compiles a bigger kernel
#: that inlines all of them, and that variant is in nobody's offline cache.
#: Measured: ``pytest -q tests/fast`` alone takes 37 s, and the same test run
#: after ``tests/unit_tests`` in one process spends ~6 minutes in
#: ``timed_compile_kernel`` on ``shade_sparse_raster_coverage`` before it
#: starts. These are skipped while that is being fixed; grep this name to find
#: all of them (there is a twin in ``test_ux_regressions.py``).
_LEAKS_A_PIPELINE = (
    "TEMPORARY: registers a fragment pipeline into the process-global registry, "
    "which specialises every later render's shade kernel in the same process."
)


def test_builtin_fragment_pipeline_is_available_to_star_imports():
    assert {"cosine_color", "phong_shader"} <= set(algan.__all__)


def test_resolve_builtin_shader_to_stage():
    # A built-in PyTorch material shader resolves to its ti.func stage port.
    assert resolve_stage(phong_shader) is STAGE_PHONG
    # A FragmentStage resolves to itself.
    assert resolve_stage(cosine_color) is cosine_color
    with pytest.raises(TypeError):
        resolve_stage(lambda: None)


@pytest.mark.skip(reason=_LEAKS_A_PIPELINE)
def test_registry_single_composed_and_dedup():
    m1, specs1 = build_fragment_pipeline(cosine_color)
    assert m1._frag_pipeline_id >= _USER_PIPELINE_BASE
    assert m1._frag_total_width == 2  # frequency + phase
    assert [n for n, _ in specs1] == ["frequency", "phase"]

    m2, specs2 = build_fragment_pipeline([cosine_color, phong_shader])
    assert m2._frag_pipeline_id != m1._frag_pipeline_id
    assert m2._frag_total_width == 14  # 2 (cosine) + 12 (phong canonical)

    # Identical pipelines reuse the same id and composed func.
    m3, _ = build_fragment_pipeline(cosine_color)
    assert m3._frag_pipeline_id == m1._frag_pipeline_id
    assert len(build_frag_pipelines()) >= 2


@pytest.mark.skip(reason=_LEAKS_A_PIPELINE)
def test_set_fragment_shader_registers_animatable_params():
    SceneManager.reset()
    s = Sphere().set_fragment_shader([cosine_color, phong_shader])
    # Marker shader carries the pipeline metadata used by the ray tracer.
    assert getattr(s.shader, "_frag_pipeline_id", None) is not None
    # Stage params are exposed as (animatable) attributes.
    assert hasattr(s, "frequency")
    assert hasattr(s, "shininess")


@pytest.mark.skip(reason=_LEAKS_A_PIPELINE)
def test_set_fragment_shader_after_spawn_raises():
    SceneManager.reset()
    s = Sphere().spawn()
    with pytest.raises(ModifiedProtectedAttributeError):
        s.set_fragment_shader(phong_shader)
