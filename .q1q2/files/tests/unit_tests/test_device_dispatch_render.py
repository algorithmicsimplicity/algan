"""Real shadowed parity with both Q1 and Q2 enabled and observed."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from algan import (
    BLACK, OUT, RIGHT, SMOKE_TEST, UP, WHITE, Cube, MeshLambertMaterial,
    MeshPhysicalMaterial, Off, PointLight, Prism, Scene,
)
from algan.rendering.raytracing import shadow_dispatch
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS


@pytest.mark.parametrize("reflective", [False, True])
def test_shadow_render_toggle_parity(tmp_path, monkeypatch, reflective):
    saved = SETTINGS.snapshot()
    seen = []
    real = shadow_dispatch.PrimaryShadowDispatch.run
    def observe(self, *args, **kwargs):
        seen.append(self.extent.capacity)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(shadow_dispatch.PrimaryShadowDispatch, "run", observe)
    frames = []
    try:
        SETTINGS.raytracing.set(samples_per_pixel=1, shadows=True, analytic_aa=True)
        for enabled in (False, True, False):
            SETTINGS.raytracing.experimental.device_dispatch = enabled
            SceneManager.reset()
            with Scene(video_settings=SMOKE_TEST) as scene:
                scene.set_background(BLACK)
                with Off():
                    Scene.clear_lights()
                    PointLight(location=UP * 3 + RIGHT * 2, color=WHITE,
                               intensity=1).spawn(animate=False)
                    material = (MeshPhysicalMaterial(color=WHITE, roughness=0.12, ior=5)
                                if reflective else MeshLambertMaterial(color=WHITE))
                    Cube(size=1, fill_opacity=0.65).set_material(material).move(
                        OUT * 2).spawn(animate=False)
                    Prism(width=6, height=6, depth=0.1).set_material(
                        MeshLambertMaterial(color=WHITE)).spawn(animate=False)
                before = len(seen)
                result = scene.save_frame(tmp_path / f"{reflective}_{len(frames)}.png",
                                          video_settings=SMOKE_TEST, overwrite=True)
                assert (len(seen) > before) == enabled
                frames.append(np.asarray(Image.open(result.output_path)).astype(np.int16))
        assert np.max(np.abs(frames[1] - frames[0])) <= 2
        assert np.max(np.abs(frames[2] - frames[0])) <= 2
        assert any(c > 0 for c in seen)
    finally:
        SceneManager.reset()
        SETTINGS.restore(saved)
