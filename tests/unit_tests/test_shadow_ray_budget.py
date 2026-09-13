"""The shadow-ray budget: what the scene builder packs into a light row.

``rt_settings.shadow_ray_budget`` is spent host-side, in
``scene_builder._soft_fan_sizes``: each packed light row carries the number
of soft-shadow rays its fan fires at a primary hit (aux 13, packed column
16) and at a secondary one (aux 14, packed 17), and the kernels read those
two numbers. Zero means the compile-time fan with no jitter -- the
pre-budget renderer -- which is what makes ``ALGAN_SHADOW_RAY_BUDGET=0`` the
byte-identical kill switch. These tests pin that arithmetic without a
render; ``test_area_light_soft_shadow.py``'s render tests compile both fans.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.lights import LIGHT_AUX_COLS
from algan.rendering.raytracing import scene_builder
from algan.rendering.raytracing import settings as rt_settings
from algan.settings import SETTINGS
from algan.settings._startup import _SOFT_SHADOW_SAMPLES


def _aux(ltype, radius, frames=2):
    aux = torch.zeros((frames, LIGHT_AUX_COLS))
    aux[:, 0] = ltype
    aux[:, 8] = radius
    return aux


@pytest.fixture
def budget():
    previous = (rt_settings.shadow_ray_budget, rt_settings.shadow_bounce_rays)

    def set_budget(rays, bounce=1):
        SETTINGS.raytracing.experimental.set(
            shadow_ray_budget=rays, shadow_bounce_rays=bounce
        )

    yield set_budget
    rt_settings.shadow_ray_budget, rt_settings.shadow_bounce_rays = previous


def test_the_settings_surface_on_the_experimental_section(budget):
    budget(12, 2)
    assert rt_settings.shadow_ray_budget == 12
    assert rt_settings.shadow_bounce_rays == 2
    assert SETTINGS.raytracing.experimental.shadow_ray_budget == 12


def test_an_area_light_splits_the_budget_over_its_cells(budget):
    budget(16, 1)
    primary, secondary = scene_builder._soft_fan_sizes(_aux(5, 0.4), num_rows=16)
    assert primary.tolist() == [1.0, 1.0]
    assert secondary.tolist() == [1.0, 1.0]
    primary, _ = scene_builder._soft_fan_sizes(_aux(5, 0.4), num_rows=9)
    assert primary.tolist() == [2.0, 2.0]  # ceil(16 / 9)
    primary, _ = scene_builder._soft_fan_sizes(_aux(5, 0.4), num_rows=64)
    assert primary.tolist() == [1.0, 1.0]  # never below one ray per cell


def test_a_single_row_soft_light_never_exceeds_the_fixed_fan(budget):
    budget(16, 1)
    primary, secondary = scene_builder._soft_fan_sizes(_aux(1, 0.05), num_rows=1)
    assert primary.tolist() == [float(min(_SOFT_SHADOW_SAMPLES, 16))] * 2
    assert secondary.tolist() == [1.0, 1.0]
    budget(3, 0)
    primary, secondary = scene_builder._soft_fan_sizes(_aux(0, 0.05), num_rows=1)
    assert primary.tolist() == [3.0, 3.0]
    # A zero bounce cap leaves secondary hits on the primary budget.
    assert secondary.tolist() == [3.0, 3.0]


def test_hard_rows_and_the_off_switch_pack_zero(budget):
    budget(16, 1)
    primary, secondary = scene_builder._soft_fan_sizes(_aux(0, 0.0), num_rows=1)
    assert not primary.any()
    assert not secondary.any()
    budget(0, 1)
    primary, secondary = scene_builder._soft_fan_sizes(_aux(5, 0.4), num_rows=16)
    assert not primary.any()
    assert not secondary.any()


def test_packed_rows_carry_the_two_fan_columns(budget):
    """End to end through ``_pack_lights``: an area light's rows are 18 wide
    and columns 16/17 hold the split budget; a plain point light sharing the
    pack carries zeros there.
    """
    budget(16, 1)

    class _Light:
        def __init__(self, origin, color, aux):
            self.origin = origin
            self.light_color = color
            self._render_aux = aux

    frames, rows = 2, 4
    area = _Light(
        torch.zeros((frames, rows, 3)),
        torch.ones((frames, rows, 4)),
        _aux(5, 0.3, frames).unsqueeze(1).expand(frames, rows, LIGHT_AUX_COLS),
    )
    point = _Light(torch.zeros((frames, 1, 3)), torch.ones((frames, 1, 4)), None)
    _pos, col, num = scene_builder._pack_lights([area, point], frames, "cpu")
    assert num == rows + 1
    assert col.shape == (frames, rows + 1, 3 + LIGHT_AUX_COLS)
    assert col[:, :rows, 16].tolist() == [[4.0] * rows] * frames  # ceil(16 / 4)
    assert col[:, :rows, 17].tolist() == [[1.0] * rows] * frames
    assert not col[:, rows, 16:].any()
    assert col[:, rows, 15].tolist() == [1.0] * frames  # power fraction kept
