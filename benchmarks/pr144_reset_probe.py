"""Repeat the real triangle probe across compiled-setting program resets."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "unit_tests"))

import torch
import test_watertight_triangle as cases

from algan.settings import SETTINGS
from algan.rendering.taichi_runtime import ensure_taichi_for_render

snapshot = SETTINGS.snapshot()
try:
    for cycle in range(60):
        SETTINGS.raytracing.linear_color_space = bool(cycle % 2)
        ensure_taichi_for_render()
        cases.test_a_ray_exactly_on_a_shared_edge_hits_exactly_one_neighbour()
        cases.test_rays_well_inside_one_triangle_hit_only_that_one()
        cases.test_a_ray_missing_both_triangles_hits_neither()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        print("RESET_PROBE_PASS", cycle, flush=True)
finally:
    SETTINGS.restore(snapshot)
