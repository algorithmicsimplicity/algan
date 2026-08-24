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


def test_builtin_fragment_pipeline_is_available_to_star_imports():
    assert {"cosine_color", "phong_shader"} <= set(algan.__all__)


def test_resolve_builtin_shader_to_stage():
    # A built-in PyTorch material shader resolves to its ti.func stage port.
    assert resolve_stage(phong_shader) is STAGE_PHONG
    # A FragmentStage resolves to itself.
    assert resolve_stage(cosine_color) is cosine_color
    with pytest.raises(TypeError):
        resolve_stage(lambda: None)


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


def test_set_fragment_shader_registers_animatable_params():
    SceneManager.reset()
    s = Sphere().set_fragment_shader([cosine_color, phong_shader])
    # Marker shader carries the pipeline metadata used by the ray tracer.
    assert getattr(s.shader, "_frag_pipeline_id", None) is not None
    # Stage params are exposed as (animatable) attributes.
    assert hasattr(s, "frequency")
    assert hasattr(s, "shininess")


def test_set_fragment_shader_after_spawn_raises():
    SceneManager.reset()
    s = Sphere().spawn()
    with pytest.raises(ModifiedProtectedAttributeError):
        s.set_fragment_shader(phong_shader)


# --------------------------------------------------------------------------
# Batch narrowing of the injected pipeline tuple.
#
# The registry is process-global and append-only, and Taichi specialises the
# shade kernels on the injected tuple, so a render that handed over the whole
# registry compiled -- and cache-missed on -- a kernel variant carrying every
# pipeline the *process* had ever registered. That made one custom shader
# anywhere in a script slow every render in it, and turned the test suite from
# 3 minutes into 37 (most of it in ``timed_compile_kernel``). These assert the
# narrowing that closes it, and the safety rule it must never break.
# --------------------------------------------------------------------------
def test_a_batch_without_a_user_pipeline_injects_nothing():
    """The whole point: a scene with no custom shader compiles the ordinary
    shade kernel however many pipelines the process has registered.
    """
    build_fragment_pipeline(cosine_color)
    assert build_frag_pipelines() != ()  # the registry is not empty
    assert build_frag_pipelines(frozenset()) == ()


def test_a_batch_injects_only_the_pipelines_it_uses():
    first, _ = build_fragment_pipeline(cosine_color)
    second, _ = build_fragment_pipeline([cosine_color, phong_shader])

    only_second = build_frag_pipelines((second._frag_pipeline_id,))
    assert len(only_second) == 1
    assert only_second[0] is not None

    both = build_frag_pipelines((first._frag_pipeline_id, second._frag_pipeline_id))
    assert len(both) == 2
    assert None not in both
    assert both[0] is not both[1]


def test_the_injected_tuple_does_not_depend_on_the_registrys_history():
    """REGRESSION, and the reason the ids are renumbered per batch.

    A pipeline's id is its position in a process-global append-only registry,
    and the injected tuple used to be indexed by that id -- padded with
    ``None`` up to it. So one pipeline was ``(fn,)`` if it registered first and
    ``(None, None, fn)`` if two others beat it to the registry, Taichi
    specialised the shade kernel on the difference, and the same custom shader
    compiled a fresh uncached variant per process. The merge now renumbers a
    batch's ids into a dense range, so the shape depends on the batch alone.

    (This is also what made the old assertion order-dependent: run after
    another module had registered the same two pipelines in the other order,
    it indexed past the end of its own narrowed tuple.)
    """
    early, _ = build_fragment_pipeline(cosine_color)
    for _ in range(3):  # push the registry along
        build_fragment_pipeline([cosine_color, phong_shader, cosine_color])
    late, _ = build_fragment_pipeline([phong_shader, cosine_color])

    assert late._frag_pipeline_id > early._frag_pipeline_id
    assert len(build_frag_pipelines((early._frag_pipeline_id,))) == 1
    assert len(build_frag_pipelines((late._frag_pipeline_id,))) == 1


def test_solo_dispatch_never_calls_an_uninjected_pipeline():
    """``solo_pid`` promises the kernel it may call that one stage with no id
    fetch at all, so it must not name a slot the batch narrowed away.
    """
    from algan.rendering.raytracing.shading_taichi import solo_pid

    mask = 1 << (_USER_PIPELINE_BASE + 1)
    assert solo_pid(mask, (object(), object())) == _USER_PIPELINE_BASE + 1
    assert solo_pid(mask, (object(), None)) == -1


def test_unenumerable_ids_fall_back_to_the_whole_registry():
    """Narrowing is only safe where the merged scene can list its material ids.
    A geometry type present without one keeps every pipeline compiled in, for
    the same reason ``_frag_pid_mask`` falls back to ``ALL_PIDS``: dropping a
    pipeline the kernel still dispatches to would shade that surface with no
    material at all.
    """
    import torch

    from algan.rendering.raytracing.tracer import _batch_user_pipeline_ids

    build_fragment_pipeline(cosine_color)
    ids = torch.zeros((1, 4), dtype=torch.int32)
    assert _batch_user_pipeline_ids({"tri_mat_id": ids}) is None
    assert build_frag_pipelines(None) != ()

    # Enumerable: the merge publishes the batch's GLOBAL ids in the slot order
    # it renumbered them into, and that list is what the narrowing reads.
    merged = {"frag_pipeline_ids": (_USER_PIPELINE_BASE + 1,)}
    assert _batch_user_pipeline_ids(merged) == (_USER_PIPELINE_BASE + 1,)
    # A batch with no user pipeline at all is enumerable and empty -- NOT the
    # unenumerable fallback, which would compile the whole registry in.
    assert _batch_user_pipeline_ids({"frag_pipeline_ids": ()}) == ()


def test_a_render_after_registering_a_pipeline_injects_nothing(tmp_path, monkeypatch):
    """The wiring, end to end: the pathology was a *render* -- one with no
    custom shader in it at all -- picking up the registry and compiling its own
    shade kernel variant. Registering a pipeline and then rendering a plain
    scene is the shape that regressed, so it is the shape that guards it.
    """
    from algan.rendering.shaders import fragment_shaders
    from algan.settings.video_settings import SMOKE_TEST

    build_fragment_pipeline(cosine_color)
    assert fragment_shaders.build_frag_pipelines() != ()

    injected = []
    original = fragment_shaders.build_frag_pipelines

    def _spy(pids=None):
        result = original(pids)
        injected.append(result)
        return result

    monkeypatch.setattr(fragment_shaders, "build_frag_pipelines", _spy)

    SceneManager.reset()
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(SMOKE_TEST)
    Sphere().spawn()
    scene.save_frame(str(tmp_path / "plain"))

    assert injected, "the render never reached the fragment-pipeline dispatch"
    assert all(pipelines == () for pipelines in injected), injected
