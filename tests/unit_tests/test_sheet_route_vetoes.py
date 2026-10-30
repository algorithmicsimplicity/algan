"""Which batches take the sheet route, and which fall back to supersampling.

``tracer.analytic_raster_route_active`` is the one host-side decision between
the analytic-coverage path and the supersampled fallback
(renderer_limitations.rst, "Within the deterministic renderer: two paths").
Every veto in it is silent. A batch that falls back renders its edges at
``supersampling`` -- 1 at PREVIEW, so no anti-aliasing at all -- and nothing
says so. A veto that stops firing sends a batch down a path that cannot draw
it; one that starts firing quietly takes a whole class of scenes off analytic
coverage; and nothing else in the CI group would notice either.

So the documented table is pinned here, together with the veto it leaves out
(``wf_mem_trim``), each next to the change one short of it that keeps the
route. Host-built batches carry only the keys the decision reads: no render,
no kernel.

Not marked fast: a change elsewhere cannot break these without changing the
route decision itself.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer

#: The experimental switches that are "preconditions of the analytic path, not
#: independent options" (renderer_limitations.rst).
PRECONDITIONS = (
    "hybrid_raster",
    "analytic_aa",
    "sheet_resolve",
    "analytic_aa_run",
    "raster_sparse_coverage",
    "raster_empty_skip",
    "raster_covered_shade",
)

#: Any environment map; the decision only asks whether there is one.
ENV = torch.zeros(4, 8, 3)


class Batch(NamedTuple):
    """One case: settings to change, keys to set on the merged batch, and the
    route's keyword arguments.
    """

    settings: dict = {}
    merged: dict = {}
    route: dict = {}


VETOES = {
    "samples_per_pixel > 1": Batch(settings={"samples_per_pixel": 2}),
    "a fragment shader that overrides ray bouncing": Batch(
        merged={"has_user_pipeline": True, "has_custom_scatter": True}
    ),
    "no triangles and no circuits": Batch(
        merged={"num_triangles": 0, "num_circuits": 0}
    ),
    **{f"{name} off": Batch(settings={name: False}) for name in PRECONDITIONS},
    "triangle coverage off, with triangles": Batch(settings={"analytic_aa_tri": False}),
    "circuit coverage off, with circuits": Batch(
        settings={"analytic_aa_bez": False},
        merged={"num_triangles": 0, "num_circuits": 1},
    ),
    "wf_mem_trim on a batch that splits no ray": Batch(
        settings={"wf_mem_trim": True}, merged={"mem_trim_active": True}
    ),
}

#: One change short of a veto: these keep the route.
NEAR_MISSES = {
    "camera.near == 0": Batch(route={"near_clip": 0.0}),
    # The emission cuts geometry at the near plane.
    "camera.near > 0": Batch(route={"near_clip": 0.1}),
    "a user pipeline that only shades": Batch(merged={"has_user_pipeline": True}),
    "a transparent background alone": Batch(route={"transparent_background": True}),
    "an environment map alone": Batch(route={"environment_map": ENV}),
    # The env prefill writes the alpha opaque, as every route does.
    "a transparent background with an environment map": Batch(
        route={"transparent_background": True, "environment_map": ENV}
    ),
    "circuits alone": Batch(merged={"num_triangles": 0, "num_circuits": 1}),
    "triangle coverage off, circuits alone": Batch(
        settings={"analytic_aa_tri": False},
        merged={"num_triangles": 0, "num_circuits": 1},
    ),
    "circuit coverage off, triangles alone": Batch(settings={"analytic_aa_bez": False}),
    "wf_mem_trim with shadows": Batch(
        settings={"wf_mem_trim": True, "shadows": True},
        merged={"mem_trim_active": True},
    ),
    "wf_mem_trim with refraction": Batch(
        settings={"wf_mem_trim": True},
        merged={"mem_trim_active": True, "has_refractive": True},
    ),
}


@pytest.fixture(autouse=True)
def _shipped_settings(monkeypatch):
    """Every switch the decision reads at its shipped value, whatever the
    environment's ``ALGAN_*`` variables say.
    """
    for name in PRECONDITIONS:
        monkeypatch.setattr(rt_settings, name, True)
    monkeypatch.setattr(rt_settings, "analytic_aa_tri", True)
    monkeypatch.setattr(rt_settings, "analytic_aa_bez", True)
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 1)
    monkeypatch.setattr(rt_settings, "shadows", False)
    monkeypatch.setattr(rt_settings, "fragment_shading", True)
    monkeypatch.setattr(rt_settings, "wf_mem_trim", False)


def _merged(**keys):
    """A one-triangle batch as far as the decision reads one."""
    return {
        "num_triangles": 1,
        "num_circuits": 0,
        "tri_frame_valid": torch.ones(1, 1, dtype=torch.bool),
        "has_user_pipeline": False,
        "has_custom_scatter": False,
        **keys,
    }


def _route_active(monkeypatch, case=Batch()):
    for name, value in case.settings.items():
        monkeypatch.setattr(rt_settings, name, value)
    return tracer.analytic_raster_route_active(_merged(**case.merged), **case.route)


def test_a_plain_batch_takes_the_route(monkeypatch):
    assert _route_active(monkeypatch)


@pytest.mark.parametrize("name", VETOES)
def test_each_veto_falls_back(monkeypatch, name):
    assert not _route_active(monkeypatch, VETOES[name])


@pytest.mark.parametrize("name", NEAR_MISSES)
def test_one_change_short_of_a_veto_keeps_the_route(monkeypatch, name):
    assert _route_active(monkeypatch, NEAR_MISSES[name])


def test_the_decision_is_the_supersample_level(monkeypatch):
    """What the route changes for the batch: analytic coverage at output
    resolution, or the requested supersampling.
    """
    scatters = _merged(has_user_pipeline=True, has_custom_scatter=True)
    assert tracer.effective_anti_alias_level(_merged(), 2) == 1
    assert tracer.effective_anti_alias_level(scatters, 2) == 2
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 4)
    # The path tracer's jittered samples are its anti-aliasing.
    assert tracer.effective_anti_alias_level(_merged(), 2) == 1
