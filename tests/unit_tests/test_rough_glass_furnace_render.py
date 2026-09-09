"""Unclamped, unsaturated render-level equilibrium check for very rough glass."""

from __future__ import annotations

import pytest
import torch

from algan import (
    ORIGIN,
    OUT,
    SMOKE_TEST,
    WHITE,
    MeshPhysicalMaterial,
    Prism,
    Scene,
)
from tests.unit_tests.test_path_tracer import _render_scene_exp


@pytest.mark.parametrize("roughness", [0.35, 0.65, 1.0])
def test_closed_rough_glass_in_a_white_furnace(tmp_path, roughness):
    def build(scene):
        Scene.clear_lights()
        scene.set_environment_map(torch.full((4, 8, 3), 0.125))
        glass = Prism(width=8, height=8, depth=1)
        glass.set_material(
            MeshPhysicalMaterial(
                color=WHITE, transmission=1, roughness=roughness, ior=1.5
            )
        )
        glass.spawn(animate=False)
        camera = scene.get_camera()
        camera.move_to(OUT * 7)
        camera.look_at(ORIGIN)
        camera.set_fov(5)

    image = _render_scene_exp(
        tmp_path,
        f"rough_glass_furnace_{roughness}.png",
        build,
        512,
        video=SMOKE_TEST.set(resolution=(24, 24)),
        max_bounces=32,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_firefly_clamp": 0,
            "pt_error_target": 0,
            "pt_rr_start_bounce": 4,
            # Exercise actual refracted paths, not straight shadow connections
            # through the other surface of the slab (not a caustic estimator).
            "pt_env_nee": False,
        },
    )
    patch = image[4:20, 4:20, :3].float()
    assert float(patch.mean()) == pytest.approx(255 * 0.125, abs=1.0)
    assert float(patch.max()) < 64, "display saturation must not hide energy errors"
