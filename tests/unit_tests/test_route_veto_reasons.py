"""Which batches leave the sheet route, and what they say about it.

``tracer.analytic_raster_route_veto`` is the single host-side decision behind
analytic coverage: ``None`` keeps a batch on the sheet route, and anything else
is the first reason, in ``ROUTE_VETOES`` order, that sends it to the classic
supersampled wavefront (or, above one sample per pixel, the path tracer). The
reason rides on the batch's ``RenderPlan`` and is logged once per render, so a
render that silently lost its anti-aliasing -- at ``supersampling=1``, PREVIEW
and SMOKE_TEST lose all of it -- says so.

Host-only: a stub ``merged`` dict and live settings. Nothing compiles or
renders.
"""

from __future__ import annotations

import logging

import pytest

from algan import SETTINGS
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.tracer import (
    ROUTE_VETOES,
    RenderPlan,
    analytic_raster_route_active,
    analytic_raster_route_veto,
    effective_anti_alias_level,
    reset_route_veto_log,
)

#: Every switch the route reads, at the value that keeps it on.
_ON_ROUTE_SETTINGS = {
    "samples_per_pixel": 1,
    "shadows": False,
    "fragment_shading": False,
    "wf_mem_trim": False,
    "hybrid_raster": True,
    "analytic_aa": True,
    "sheet_resolve": True,
    "analytic_aa_run": True,
    "raster_sparse_coverage": True,
    "raster_empty_skip": True,
    "raster_covered_shade": True,
    "analytic_aa_tri": True,
    "analytic_aa_bez": True,
    "raster_ss": True,
}


def _batch(**facts):
    """A merged batch the route takes: triangles and circuits, nothing else."""
    merged = {
        "num_triangles": 4,
        "num_circuits": 2,
        "has_user_pipeline": False,
        "has_custom_scatter": False,
    }
    merged.update(facts)
    return merged


def _set(**fields):
    SETTINGS.raytracing.experimental.set(**fields)


#: One way to trigger each reason from an on-route batch: (settings, batch
#: facts, route-call keywords). Keyed by reason so a veto that is retired drops
#: one row.
_TRIGGERS = {
    "path_tracer": ({"samples_per_pixel": 4}, {}, {}),
    "hybrid_raster_off": ({"hybrid_raster": False}, {}, {}),
    "analytic_aa_off": ({"analytic_aa": False}, {}, {}),
    "sheet_resolve_off": ({"sheet_resolve": False}, {}, {}),
    "analytic_aa_run_off": ({"analytic_aa_run": False}, {}, {}),
    "raster_sparse_coverage_off": ({"raster_sparse_coverage": False}, {}, {}),
    "raster_empty_skip_off": ({"raster_empty_skip": False}, {}, {}),
    "raster_covered_shade_off": ({"raster_covered_shade": False}, {}, {}),
    "transparent_background_env_map": (
        {},
        {},
        {"transparent_background": True, "environment_map": object()},
    ),
    "near_clip": ({}, {}, {"near_clip": 0.1}),
    "no_geometry": ({}, {"num_triangles": 0, "num_circuits": 0}, {}),
    "analytic_aa_tri_off": ({"analytic_aa_tri": False}, {}, {}),
    "raster_ss_off": ({"raster_ss": False}, {}, {}),
    "analytic_aa_bez_off": ({"analytic_aa_bez": False}, {}, {}),
    "custom_scatter": (
        {},
        {"has_user_pipeline": True, "has_custom_scatter": True},
        {},
    ),
    "wf_mem_trim": ({"wf_mem_trim": True}, {"mem_trim_active": True}, {}),
}


@pytest.fixture(autouse=True)
def _on_route():
    """Start every test on the route, with the INFO line armed.

    The suite's own fixture restores the settings afterwards.
    """
    _set(**_ON_ROUTE_SETTINGS)
    reset_route_veto_log()
    yield
    reset_route_veto_log()


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


def test_an_eligible_batch_takes_the_route():
    merged = _batch()
    assert analytic_raster_route_veto(merged) is None
    assert analytic_raster_route_active(merged)
    assert effective_anti_alias_level(merged, 3) == 1


def test_every_reason_has_a_trigger():
    assert list(_TRIGGERS) == list(ROUTE_VETOES)


@pytest.mark.parametrize("reason", list(_TRIGGERS))
def test_each_reason_is_reported_by_name(reason):
    settings, facts, kwargs = _TRIGGERS[reason]
    _set(**settings)
    merged = _batch(**facts)

    veto = analytic_raster_route_veto(merged, **kwargs)
    assert veto == reason
    assert veto in ROUTE_VETOES
    # The bool form is the same decision.
    assert analytic_raster_route_active(merged, **kwargs) is False
    # Off the route a batch keeps its supersampling; the path tracer never
    # supersamples.
    expected = 1 if reason == "path_tracer" else 3
    assert effective_anti_alias_level(merged, 3, **kwargs) == expected


def test_the_first_reason_in_table_order_wins():
    """Triggered last-to-first, each new reason outranks every one before it.

    Pins that ``ROUTE_VETOES`` lists the reasons in the order the veto checks
    them, which is what "the first reason" means to a reader of either.
    """
    settings, facts, kwargs = {}, {}, {}
    for reason in reversed(list(_TRIGGERS)):
        more_settings, more_facts, more_kwargs = _TRIGGERS[reason]
        settings.update(more_settings)
        facts.update(more_facts)
        kwargs.update(more_kwargs)
        _set(**settings)
        assert analytic_raster_route_veto(_batch(**facts), **kwargs) == reason


@pytest.mark.parametrize(
    ("switch", "kind"),
    [
        ("analytic_aa_tri", "num_triangles"),
        ("raster_ss", "num_triangles"),
        ("analytic_aa_bez", "num_circuits"),
    ],
)
def test_a_per_kind_switch_vetoes_only_a_batch_holding_that_kind(switch, kind):
    _set(**{switch: False})
    assert analytic_raster_route_veto(_batch(**{kind: 0})) is None


def test_the_memory_trim_vetoes_only_where_it_would_engage():
    """The trim stays off a shadowed batch, so that batch keeps the route."""
    _set(wf_mem_trim=True, shadows=True)
    assert analytic_raster_route_veto(_batch(mem_trim_active=True)) is None


def _plan(samples_per_pixel=1):
    return RenderPlan(
        backend="path_tracer" if samples_per_pixel > 1 else "deterministic_wavefront",
        samples_per_pixel=samples_per_pixel,
        requested_features=(),
    )


def test_the_plan_names_the_route_and_why():
    validation = _plan()
    assert validation.route is None
    assert validation.route_veto is None

    plan, aa = tracer._choose_route(validation, _batch(), 2)
    assert (plan.route, plan.route_veto, aa) == ("analytic", None, 1)
    assert plan.as_dict()["route"] == "analytic"
    assert plan.as_dict()["route_veto"] is None

    _set(analytic_aa=False)
    plan, aa = tracer._choose_route(validation, _batch(), 2)
    assert (plan.route, plan.route_veto, aa) == ("supersampled", "analytic_aa_off", 2)
    assert plan.as_dict()["route_veto"] == "analytic_aa_off"
    # Only the route fields are grafted on.
    assert plan.backend == validation.backend
    assert plan.requested_features == validation.requested_features

    _set(analytic_aa=True, samples_per_pixel=4)
    plan, aa = tracer._choose_route(_plan(4), _batch(), 2)
    assert (plan.route, plan.route_veto, aa) == ("path_tracer", "path_tracer", 1)


def test_a_vetoed_batch_says_why_once_per_reason(algan_logs):
    _set(analytic_aa=False)
    for _ in range(3):
        tracer._choose_route(_plan(), _batch(), 2)
    (line,) = _route_lines(algan_logs.records)
    assert "analytic_aa_off" in line
    assert ROUTE_VETOES["analytic_aa_off"] in line
    assert "supersampling=2" in line

    # A second reason in the same render gets its own line, once.
    _set(analytic_aa=True, sheet_resolve=False)
    for _ in range(2):
        tracer._choose_route(_plan(), _batch(), 2)
    lines = _route_lines(algan_logs.records)
    assert len(lines) == 2
    assert "sheet_resolve_off" in lines[1]

    # A new render job re-arms it.
    reset_route_veto_log()
    tracer._choose_route(_plan(), _batch(), 2)
    assert len(_route_lines(algan_logs.records)) == 3


def test_at_supersampling_1_the_line_says_there_is_no_anti_aliasing(algan_logs):
    """PREVIEW and SMOKE_TEST render at supersampling=1, where the fallback
    is one ray per pixel and nothing else.
    """
    _set(hybrid_raster=False)
    tracer._choose_route(_plan(), _batch(), 1)
    (line,) = _route_lines(algan_logs.records)
    assert "supersampling=1" in line
    assert "no anti-aliasing at all" in line


def test_the_route_and_the_path_tracer_say_nothing(algan_logs):
    tracer._choose_route(_plan(), _batch(), 2)
    _set(samples_per_pixel=4)
    tracer._choose_route(_plan(4), _batch(), 2)
    assert _route_lines(algan_logs.records) == []
