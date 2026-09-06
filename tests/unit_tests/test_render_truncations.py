"""The instrument on the render path's four silent truncations.

``RENDERER_WORK_QUEUE.md`` item 1. Each of the four ceilings degrades the image
when it binds and used to say nothing at all, so what is under test here is
that they now *report*: a WARNING naming the ceiling, and a count on the
``RenderPlan`` a script can assert on.

The four are not equally reachable, and the tests say so rather than pretending
otherwise:

* ``shadow_lights`` and ``sheet_layers`` are exercised by a real render of a
  scene built to exceed them, which is the acceptance criterion the queue item
  states.
* ``surfaces_per_ray`` needs 257 surfaces stacked in one pixel, each thin
  enough that the ray's throughput has not already fallen under ``min_weight``
  by the 256th. That is a real render but a slow one, so it is checked here at
  the level of the counter and left to the manual probe in the commit message
  as a scene.
* ``dropped_continuations`` has **no reachable scene**: every kernel branch
  that reserves a pool slot is compiled in only when ``refraction_flag != 0``,
  and every condition that sets that flag also drives ``pool_ratio`` above 1,
  where the host discards and retries the tile instead of losing the branch.
  The counter is there for the case where the host's flags and the kernel's
  runtime test stop agreeing -- which is precisely the failure that would
  otherwise be invisible -- so what is tested is the host arithmetic that would
  catch it.

Not in the fast suite: a change elsewhere in the codebase does not break these,
only a change to the instrument itself.
"""

from __future__ import annotations

import logging

import pytest

from algan.constants.color import BLUE
from algan.logging.logger import PERF
from algan.mobs.shapes_3d import Polyhedron
from algan.rendering.lights import PointLight
from algan.rendering.raytracing.shading_taichi import max_shadow_lights
from algan.rendering.raytracing.sheets import SHEET_RANK_LIMIT
from algan.rendering.raytracing.tracer import (
    ALLOC_NEXT,
    ALLOC_TRUNC_SURFACES,
    ALLOC_WIDTH,
    RenderPlan,
    _record_tile_truncations,
)
from algan.rendering.raytracing.truncation import (
    TruncationCounts,
    _TruncationRecorder,
    record_truncation,
    report_truncations,
    reset_truncations,
    restore_truncations,
    snapshot_truncations,
)
from algan.scene import Scene
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.settings.video_settings import SMOKE_TEST


@pytest.fixture
def recorder():
    """A recorder of its own, so a test cannot see the process-wide one."""
    return _TruncationRecorder()


@pytest.fixture
def algan_logs(caplog):
    """``caplog``, wired to Algan's logger.

    ``logging.getLogger("algan")`` sets ``propagate = False`` on purpose -- so
    an application that configured the root logger does not double-print every
    render message -- and that is also why ``caplog`` alone sees nothing: its
    handler sits on root. Attach it to Algan's logger instead, at DEBUG so the
    PERF-level escalations are visible too.
    """
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


def _warnings(records):
    return [record for record in records if record.levelno >= logging.WARNING]


@pytest.fixture
def clean_counters():
    """Zero the process-wide counters around a test that uses them."""
    reset_truncations()
    try:
        yield
    finally:
        reset_truncations()


# ---------------------------------------------------------------------------
# The counts themselves
# ---------------------------------------------------------------------------


def test_a_render_that_truncated_nothing_reports_zero_rather_than_nothing():
    """§Y's rule: an instrument that reports zero may not be looking.

    So the empty value is a real all-zero reading with every ceiling named,
    not an absent one.
    """
    counts = TruncationCounts()

    assert counts.total == 0
    assert not counts
    assert counts.as_dict() == {
        "surfaces_per_ray": 0,
        "shadow_lights": 0,
        "sheet_layers": 0,
        "dropped_continuations": 0,
        "closed_shell_ring": 0,
    }


def test_total_covers_every_ceiling():
    counts = TruncationCounts(
        surfaces_per_ray=1,
        shadow_lights=2,
        sheet_layers=4,
        dropped_continuations=8,
        closed_shell_ring=16,
    )

    assert counts.total == 31
    assert counts


def test_event_ceilings_add_up_across_a_render(recorder):
    """A ray truncated in batch 2 is a different ray from one in batch 1."""
    recorder.record("surfaces_per_ray", 10, cap=256)
    recorder.record("surfaces_per_ray", 7, cap=256)

    assert recorder.snapshot().surfaces_per_ray == 17


def test_the_shadow_light_ceiling_reports_the_worst_batch_not_the_sum(recorder):
    """The five lights over the cap in every batch are the same five lights.

    Adding them would report a three-batch render of one over-lit scene as
    fifteen unshadowed lights, which is a number that describes nothing.
    """
    for _ in range(3):
        recorder.record("shadow_lights", 5, cap=16)

    assert recorder.snapshot().shadow_lights == 5


def test_the_shadow_light_warning_names_the_path_tracer_switch(recorder, algan_logs):
    """The 16-light cap is one of the three failures the path tracer exists
    for (DESIGN_path_tracer_roadmap.md section 0.3), so the warning at the
    failure names the switch rather than leaving the user to find the docs.
    """
    from algan.render_loop import PATH_TRACER_FALLBACK_SPELLING

    recorder.record("shadow_lights", 5, cap=16)
    recorder.report()

    message = algan_logs.records[-1].message
    assert "path tracer" in message
    assert PATH_TRACER_FALLBACK_SPELLING in message


def test_the_one_frame_oom_message_names_the_switch_only_for_the_deterministic_renderer():
    """The hint is for a user whose deterministic render did not fit; the
    path tracer's own out-of-memory must not tell them to switch to it.
    """
    from algan.render_loop import (
        PATH_TRACER_FALLBACK_SPELLING,
        _one_frame_does_not_fit_message,
    )
    from algan.settings import SETTINGS

    with SETTINGS.raytracing.override(samples_per_pixel=1):
        assert PATH_TRACER_FALLBACK_SPELLING in _one_frame_does_not_fit_message()
    with SETTINGS.raytracing.override(samples_per_pixel=4):
        assert PATH_TRACER_FALLBACK_SPELLING not in _one_frame_does_not_fit_message()


def test_reset_zeroes_the_counts_and_rearms_the_warning(recorder, algan_logs):
    recorder.record("sheet_layers", 3, cap=16)
    recorder.report()
    assert recorder.snapshot().sheet_layers == 3

    recorder.reset()
    assert recorder.snapshot().total == 0

    recorder.record("sheet_layers", 3, cap=16)
    algan_logs.clear()
    recorder.report()

    assert [r.levelno for r in algan_logs.records] == [logging.WARNING]


# ---------------------------------------------------------------------------
# What gets logged, and how loudly
# ---------------------------------------------------------------------------


def test_the_first_batch_to_truncate_warns_and_names_the_ceiling(recorder, algan_logs):
    """WARNING, not PERF: these move the image, unlike the pool retries and
    batch splits that PERF exists for.
    """
    recorder.record("surfaces_per_ray", 12, cap=256)

    recorder.report()

    assert len(algan_logs.records) == 1
    record = algan_logs.records[0]
    assert record.levelno == logging.WARNING
    assert "12" in record.message
    assert "256" in record.message
    assert "max_surfaces_per_ray" in record.message


def test_later_batches_escalate_the_total_below_info_instead_of_warning_again(
    recorder, algan_logs
):
    """One scene's one defect is one warning, however many batches meet it."""
    recorder.record("surfaces_per_ray", 12, cap=256)
    recorder.report()
    recorder.record("surfaces_per_ray", 30, cap=256)
    recorder.report()

    assert [r.levelno for r in algan_logs.records] == [logging.WARNING, PERF]
    assert "42" in algan_logs.records[-1].message


def test_a_batch_that_truncated_nothing_new_says_nothing_at_all(recorder, algan_logs):
    recorder.record("sheet_layers", 5, cap=16)
    recorder.report()
    algan_logs.clear()
    recorder.report()
    recorder.report()

    assert algan_logs.records == []


def test_a_ceiling_that_never_bound_is_never_logged(recorder, algan_logs):
    recorder.record("sheet_layers", 1, cap=16)
    recorder.report()

    assert len(algan_logs.records) == 1
    assert "overlapped" in algan_logs.records[0].message


# ---------------------------------------------------------------------------
# The out-of-memory chunk retry
# ---------------------------------------------------------------------------


def test_a_discarded_chunk_does_not_leave_its_counts_behind(recorder):
    """The render loop halves a chunk and re-renders it after an OOM.

    The discarded attempt's truncations describe frames that are about to be
    rendered again, so they roll back with the arena pointers.
    """
    recorder.record("surfaces_per_ray", 100, cap=256)
    entry = recorder.snapshot()

    recorder.record("surfaces_per_ray", 400, cap=256)  # the attempt that OOMed
    recorder.restore(entry)

    assert recorder.snapshot().surfaces_per_ray == 100


def test_the_retry_after_a_rollback_can_still_report_its_own_total(
    recorder, algan_logs
):
    """Rolling back must not leave the retry's real count looking like one
    that has already been reported.
    """
    recorder.record("surfaces_per_ray", 400, cap=256)
    recorder.report()
    entry = TruncationCounts()
    recorder.restore(entry)
    algan_logs.clear()
    recorder.record("surfaces_per_ray", 400, cap=256)
    recorder.report()

    assert [r.levelno for r in algan_logs.records] == [PERF]
    assert "400" in algan_logs.records[0].message


# ---------------------------------------------------------------------------
# The host-side arithmetic on a tile's allocator words
# ---------------------------------------------------------------------------


def _alloc(next_slot, overflow=0, truncated_surfaces=0):
    words = [0] * ALLOC_WIDTH
    words[ALLOC_NEXT] = next_slot
    words[ALLOC_TRUNC_SURFACES] = truncated_surfaces
    return words


def test_a_tile_that_stayed_inside_its_pool_drops_nothing(clean_counters):
    """``rs_alloc[ALLOC_NEXT]`` ends at the capacity when the last slot was
    taken but none failed -- the boundary that must not read as a drop.
    """
    _record_tile_truncations(_alloc(next_slot=512), pool=512)

    assert snapshot_truncations().dropped_continuations == 0


def test_reservations_past_the_pool_are_counted_as_dropped(clean_counters):
    """A failed reservation still does its atomic increment, so the surplus
    over the capacity is exactly how many continuations found no slot.
    """
    _record_tile_truncations(_alloc(next_slot=519), pool=512)

    assert snapshot_truncations().dropped_continuations == 7


def test_a_tile_folds_in_the_kernel_s_surface_ceiling_counter(clean_counters):
    _record_tile_truncations(_alloc(next_slot=64, truncated_surfaces=9), pool=64)

    counts = snapshot_truncations()
    assert counts.surfaces_per_ray == 9
    assert counts.dropped_continuations == 0


# ---------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------


def test_the_render_plan_carries_the_counts():
    plan = RenderPlan(
        backend="deterministic_wavefront",
        samples_per_pixel=1,
        requested_features=(),
        truncations=TruncationCounts(sheet_layers=4),
    )

    assert plan.truncations.sheet_layers == 4
    assert plan.as_dict()["truncations"]["sheet_layers"] == 4


def test_a_plan_built_without_them_still_answers_the_question():
    """The plan is built during validation, before anything has rendered, so
    the field has to default rather than be absent.
    """
    plan = RenderPlan(
        backend="deterministic_wavefront",
        samples_per_pixel=1,
        requested_features=(),
    )

    assert plan.truncations == TruncationCounts()
    assert plan.truncations.total == 0


def test_the_module_api_drives_the_process_wide_recorder(clean_counters):
    record_truncation("sheet_layers", 6, cap=16)
    assert snapshot_truncations().sheet_layers == 6

    restore_truncations(TruncationCounts())
    assert snapshot_truncations().total == 0

    record_truncation("sheet_layers", 2, cap=16)
    assert report_truncations().sheet_layers == 2


# ---------------------------------------------------------------------------
# End to end: a scene built to exceed a ceiling reports it
# ---------------------------------------------------------------------------


def _stacked_faces(copies):
    """One polyhedron of ``copies`` quads stacked 1e-4 apart along the view.

    A ``Polyhedron`` declares all of its faces as ONE surface, and 1e-4 keeps
    them inside a single depth band, so every pixel they share sees ``copies``
    overlapping layers of the same surface -- which is what the conflict rank
    counts and what its four bits of the sheet key cannot hold past 16.
    """
    vertices, faces = [], []
    for i in range(copies):
        z = i * 1e-4
        base = len(vertices)
        vertices += [[-2, -2, z], [2, -2, z], [2, 2, z], [-2, 2, z]]
        faces.append([base, base + 1, base + 2, base + 3])
    return vertices, faces


def test_a_stack_of_overlapping_faces_reports_the_sheet_layer_ceiling(
    tmp_path, algan_logs
):
    copies = SHEET_RANK_LIMIT + 9
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            vertices, faces = _stacked_faces(copies)
            Polyhedron(vertices, faces, color=BLUE).set_opacity(0.3).spawn(
                animate=False
            )
            result = scene.save_frame(
                str(tmp_path / "layers.png"),
                video_settings=SMOKE_TEST,
                overwrite=True,
            )
    finally:
        SceneManager.reset()

    assert result.render_plan.truncations.sheet_layers > 0
    assert any(
        record.levelno == logging.WARNING and "overlapped" in record.message
        for record in algan_logs.records
    ), [r.message for r in algan_logs.records]


def test_more_lights_than_the_shadow_cap_reports_the_surplus(tmp_path, algan_logs):
    surplus = 5
    SETTINGS.raytracing.set(shadows=True)
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            Polyhedron(*_stacked_faces(1), color=BLUE).spawn(animate=False)
            # The Scene already carries one light of its own, so spawn one
            # fewer than the surplus the cap should end up reporting.
            for i in range(max_shadow_lights + surplus - 1):
                PointLight(location=(i * 0.1, 0.0, 3.0)).spawn(animate=False)
            result = scene.save_frame(
                str(tmp_path / "lights.png"),
                video_settings=SMOKE_TEST,
                overwrite=True,
            )
    finally:
        SceneManager.reset()

    assert result.render_plan.truncations.shadow_lights == surplus
    assert any(
        record.levelno == logging.WARNING and "shadow cap" in record.message
        for record in algan_logs.records
    ), [r.message for r in algan_logs.records]


def test_an_ordinary_scene_neither_truncates_nor_warns(tmp_path, algan_logs):
    """The counters have to be quiet on a scene that trips nothing, or the
    warning stops meaning anything.
    """
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            Polyhedron(*_stacked_faces(1), color=BLUE).spawn(animate=False)
            result = scene.save_frame(
                str(tmp_path / "plain.png"),
                video_settings=SMOKE_TEST,
                overwrite=True,
            )
    finally:
        SceneManager.reset()

    assert result.render_plan is not None
    assert result.render_plan.truncations.total == 0
    assert _warnings(algan_logs.records) == []
