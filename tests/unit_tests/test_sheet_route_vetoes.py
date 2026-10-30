"""Which batches take the sheet route, which fall back, and what they say.

``tracer.analytic_raster_route_veto`` is the one host-side decision between
the analytic-coverage path and the supersampled fallback
(renderer_limitations.rst, "Within the deterministic renderer: two paths"):
``None`` keeps a batch on the route, and anything else is the first reason, in
``ROUTE_VETOES`` order, that sends it to the classic wavefront (or, above one
sample per pixel, the path tracer). ``analytic_raster_route_active`` is the
same decision as a bool. A batch that falls back renders its edges at
``supersampling`` -- 1 at PREVIEW, so no anti-aliasing at all -- so the reason
rides on the batch's ``RenderPlan`` and is logged once per render job. A veto
that stops firing sends a batch down a path that cannot draw it; one that
starts firing quietly takes a whole class of scenes off analytic coverage; and
nothing else in the CI group would notice either.

So every reason is pinned here, each next to the change one short of it that
keeps the route. Host-built batches carry only the keys the decision reads: no
render, no kernel.

Not marked fast: a change elsewhere cannot break these without changing the
route decision itself.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pytest
import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.tracer import ROUTE_VETOES, RenderPlan

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


#: One batch for each reason the route gives, keyed by the reason, in the order
#: the decision checks them.
VETOES = {
    "path_tracer": Batch(settings={"samples_per_pixel": 2}),
    **{f"{name}_off": Batch(settings={name: False}) for name in PRECONDITIONS},
    "transparent_background_env_map": Batch(
        route={"transparent_background": True, "environment_map": ENV}
    ),
    "near_clip": Batch(route={"near_clip": 0.1}),
    "no_geometry": Batch(merged={"num_triangles": 0, "num_circuits": 0}),
    "analytic_aa_tri_off": Batch(settings={"analytic_aa_tri": False}),
    "raster_ss_off": Batch(settings={"raster_ss": False}),
    "analytic_aa_bez_off": Batch(
        settings={"analytic_aa_bez": False}, merged={"num_circuits": 1}
    ),
    "custom_scatter": Batch(
        merged={"has_user_pipeline": True, "has_custom_scatter": True}
    ),
    "wf_mem_trim": Batch(
        settings={"wf_mem_trim": True}, merged={"mem_trim_active": True}
    ),
}

#: One change short of a veto: these keep the route.
NEAR_MISSES = {
    "camera.near == 0": Batch(route={"near_clip": 0.0}),
    "a user pipeline that only shades": Batch(merged={"has_user_pipeline": True}),
    "a transparent background alone": Batch(route={"transparent_background": True}),
    "an environment map alone": Batch(route={"environment_map": ENV}),
    "circuits alone": Batch(merged={"num_triangles": 0, "num_circuits": 1}),
    "triangle coverage off, circuits alone": Batch(
        settings={"analytic_aa_tri": False},
        merged={"num_triangles": 0, "num_circuits": 1},
    ),
    "raster_ss off, circuits alone": Batch(
        settings={"raster_ss": False},
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
    environment's ``ALGAN_*`` variables say, and the once-per-reason INFO line
    armed.
    """
    for name in PRECONDITIONS:
        monkeypatch.setattr(rt_settings, name, True)
    monkeypatch.setattr(rt_settings, "analytic_aa_tri", True)
    monkeypatch.setattr(rt_settings, "analytic_aa_bez", True)
    monkeypatch.setattr(rt_settings, "raster_ss", True)
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 1)
    monkeypatch.setattr(rt_settings, "shadows", False)
    monkeypatch.setattr(rt_settings, "fragment_shading", True)
    monkeypatch.setattr(rt_settings, "wf_mem_trim", False)
    tracer.reset_route_veto_log()
    yield
    tracer.reset_route_veto_log()


def _merged(**keys):
    """A one-triangle batch as far as the decision reads one."""
    return {
        "num_triangles": 1,
        "num_circuits": 0,
        "has_user_pipeline": False,
        "has_custom_scatter": False,
        **keys,
    }


def _apply(monkeypatch, case):
    """Change the case's settings; return its batch and route keywords."""
    for name, value in case.settings.items():
        monkeypatch.setattr(rt_settings, name, value)
    return _merged(**case.merged), case.route


def _route_active(monkeypatch, case=Batch()):
    merged, route = _apply(monkeypatch, case)
    return tracer.analytic_raster_route_active(merged, **route)


def test_a_plain_batch_takes_the_route(monkeypatch):
    assert _route_active(monkeypatch)
    assert tracer.analytic_raster_route_veto(_merged()) is None


def test_every_reason_has_a_batch():
    assert list(VETOES) == list(ROUTE_VETOES)


@pytest.mark.parametrize("reason", VETOES)
def test_each_veto_falls_back_and_says_why(monkeypatch, reason):
    merged, route = _apply(monkeypatch, VETOES[reason])
    assert tracer.analytic_raster_route_veto(merged, **route) == reason
    # The bool form is the same decision.
    assert tracer.analytic_raster_route_active(merged, **route) is False
    # Off the route a batch keeps its supersampling; the path tracer never
    # supersamples.
    expected = 1 if reason == "path_tracer" else 3
    assert tracer.effective_anti_alias_level(merged, 3, **route) == expected


@pytest.mark.parametrize("name", NEAR_MISSES)
def test_one_change_short_of_a_veto_keeps_the_route(monkeypatch, name):
    assert _route_active(monkeypatch, NEAR_MISSES[name])


def test_the_first_reason_in_table_order_wins(monkeypatch):
    """Applied last to first, each batch's reason outranks every one applied
    before it, so ``ROUTE_VETOES`` lists the reasons in the order the decision
    checks them: what "the first reason" means to a reader of either.
    """
    merged, route = {}, {}
    for reason in reversed(VETOES):
        case = VETOES[reason]
        for name, value in case.settings.items():
            monkeypatch.setattr(rt_settings, name, value)
        merged.update(case.merged)
        route.update(case.route)
        assert tracer.analytic_raster_route_veto(_merged(**merged), **route) == reason


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


def _plan(samples_per_pixel=1):
    return RenderPlan(
        backend="path_tracer" if samples_per_pixel > 1 else "deterministic_wavefront",
        samples_per_pixel=samples_per_pixel,
        requested_features=(),
    )


def test_the_plan_names_the_route_and_why(monkeypatch):
    validation = _plan()
    assert validation.route is None
    assert validation.route_veto is None

    plan, aa = tracer._choose_route(validation, _merged(), 2)
    assert (plan.route, plan.route_veto, aa) == ("analytic", None, 1)
    assert plan.as_dict()["route"] == "analytic"
    assert plan.as_dict()["route_veto"] is None

    monkeypatch.setattr(rt_settings, "analytic_aa", False)
    plan, aa = tracer._choose_route(validation, _merged(), 2)
    assert (plan.route, plan.route_veto, aa) == ("supersampled", "analytic_aa_off", 2)
    assert plan.as_dict()["route_veto"] == "analytic_aa_off"
    # Only the route fields are grafted on.
    assert plan.backend == validation.backend
    assert plan.requested_features == validation.requested_features

    monkeypatch.setattr(rt_settings, "analytic_aa", True)
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 4)
    plan, aa = tracer._choose_route(_plan(4), _merged(), 2)
    assert (plan.route, plan.route_veto, aa) == ("path_tracer", "path_tracer", 1)


@pytest.fixture
def algan_logs(caplog):
    """``caplog`` on Algan's logger, which does not propagate to root."""
    algan_logger = logging.getLogger("algan")
    previous_level = algan_logger.level
    caplog.set_level(logging.DEBUG)
    caplog.handler.setLevel(logging.DEBUG)
    algan_logger.addHandler(caplog.handler)
    algan_logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        algan_logger.removeHandler(caplog.handler)
        algan_logger.setLevel(previous_level)


def _route_lines(records):
    return [
        r.getMessage()
        for r in records
        if r.levelno == logging.INFO and "analytic anti-aliasing" in r.getMessage()
    ]


def test_a_vetoed_batch_says_why_once_per_reason(monkeypatch, algan_logs):
    monkeypatch.setattr(rt_settings, "analytic_aa", False)
    for _ in range(3):
        tracer._choose_route(_plan(), _merged(), 2)
    (line,) = _route_lines(algan_logs.records)
    assert "analytic_aa_off" in line
    assert ROUTE_VETOES["analytic_aa_off"] in line
    assert "supersampling=2" in line

    # A second reason in the same render gets its own line, once.
    monkeypatch.setattr(rt_settings, "analytic_aa", True)
    monkeypatch.setattr(rt_settings, "sheet_resolve", False)
    for _ in range(2):
        tracer._choose_route(_plan(), _merged(), 2)
    lines = _route_lines(algan_logs.records)
    assert len(lines) == 2
    assert "sheet_resolve_off" in lines[1]

    # A new render job re-arms it.
    tracer.reset_route_veto_log()
    tracer._choose_route(_plan(), _merged(), 2)
    assert len(_route_lines(algan_logs.records)) == 3


def test_at_supersampling_1_the_line_says_there_is_no_anti_aliasing(
    monkeypatch, algan_logs
):
    """PREVIEW and SMOKE_TEST render at supersampling=1, where the fallback
    is one ray per pixel and nothing else.
    """
    monkeypatch.setattr(rt_settings, "hybrid_raster", False)
    tracer._choose_route(_plan(), _merged(), 1)
    (line,) = _route_lines(algan_logs.records)
    assert "supersampling=1" in line
    assert "no anti-aliasing at all" in line


def test_the_route_and_the_path_tracer_say_nothing(monkeypatch, algan_logs):
    tracer._choose_route(_plan(), _merged(), 2)
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 4)
    tracer._choose_route(_plan(4), _merged(), 2)
    assert _route_lines(algan_logs.records) == []
