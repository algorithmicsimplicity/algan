"""Functional tests for custom & composable fragment shaders
(:meth:`algan.mobs.mob.Mob.set_fragment_shader`).

These assert behaviour (registry bookkeeping, param registration, the
set-after-spawn guard, and a render smoke that the composition is visible and a
single-stage user pipeline reproduces the built-in material) rather than
frame-comparing against stored expected outputs, so they don't need baked
reference frames.

    .venv/Scripts/python.exe -m pytest tests/test_files/test_fragment_shaders.py
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from algan import (
    Sync, Sphere, SceneManager, LEFT, BLUE,
    MeshPhongMaterial, cosine_color, )
from algan.animatable_base.mob import ModifiedProtectedAttributeError
from algan.rendering.shaders.material_shaders import phong_shader
import algan.rendering.raytracing.primitives as rtp
from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE
from algan.rendering.shaders.fragment_shaders import (
    build_fragment_pipeline, build_frag_pipelines, resolve_stage, STAGE_PHONG,
)


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
    assert m1._frag_total_width == 2          # frequency + phase
    assert [n for n, _ in specs1] == ["frequency", "phase"]

    m2, specs2 = build_fragment_pipeline([cosine_color, phong_shader])
    assert m2._frag_pipeline_id != m1._frag_pipeline_id
    assert m2._frag_total_width == 14         # 2 (cosine) + 12 (phong canonical)

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
    assert hasattr(s, "frequency") and hasattr(s, "shininess")


def test_set_fragment_shader_after_spawn_raises():
    SceneManager.reset()
    s = Sphere().spawn()
    with pytest.raises(ModifiedProtectedAttributeError):
        s.set_fragment_shader(phong_shader)


def _render(kind):
    SceneManager.reset()
    with Sync():
        s = Sphere().scale(1.3).move(LEFT * 0.0)
        if kind == "builtin":
            s.set_material(MeshPhongMaterial(color=BLUE))
        elif kind == "user_phong":
            s.set_color(BLUE)
            s.set_fragment_shader(phong_shader)
        else:  # cosine_phong
            s.set_color(BLUE)
            s.set_fragment_shader([cosine_color, phong_shader])
        s.spawn()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        out = tf.name
    try:
        frames = SceneManager.instance().save_frame(out)
    finally:
        if os.path.exists(out):
            os.remove(out)
    return (frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0)


@pytest.mark.parametrize("wavefront", [False, True])
def test_render_custom_and_composed(wavefront):
    with torch.inference_mode():
        rtp.set_wavefront(wavefront)
        try:
            builtin = _render("builtin")
            user = _render("user_phong")
            cosine = _render("cosine_phong")
        finally:
            rtp.set_wavefront(False)

    # A single-stage user pipeline reproduces the built-in material stage.
    assert np.abs(builtin.astype(np.float64)
                  - user.astype(np.float64)).max() <= 2.0
    # Composition changes the image (the cosine recolour is visible), and the
    # sphere is still lit + non-empty (phong ran on the recoloured albedo).
    assert np.abs(user.astype(np.float64)
                  - cosine.astype(np.float64)).mean() > 2.0
    assert cosine[..., :3].max() > 40.0
