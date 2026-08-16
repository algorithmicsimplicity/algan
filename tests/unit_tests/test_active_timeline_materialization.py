import math

import pytest
import torch

from algan import Mob, Scene
from algan.animation_timeline.animation_contexts import Off, Seq
from algan.animation_timeline.timeline import AttributeTimeline, Lifespan, TimelineSpan
from algan.constants import rate_funcs
from algan.constants.spatial import OUT, RIGHT
from algan.scene_manager import SceneManager

# In the fast suite: materializing actor state at a set of frame times is the
# step between "the user recorded something" and "the renderer sees geometry".
pytestmark = pytest.mark.fast


def test_selected_materialization_matches_full_state_for_active_mob():
    scene = SceneManager.reset()
    active = Mob().spawn(animate=False)
    start = scene.animation_manager.context.timespan.current_time
    with Seq(rate_func=rate_funcs.identity):
        active.move(RIGHT * 2)

    # A later mob contributes rows and edits to the global timeline but cannot
    # affect the queried render window.
    scene.wait(4)
    future = Mob().spawn(animate=False)
    with Seq(rate_func=rate_funcs.identity):
        future.move(RIGHT * 3)

    times = torch.tensor([start + 0.25, start + 0.75])
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times)
        expected = active.location.clone()
        timeline.clear_buffers()

        timeline.set_state_to_times(times, active_mobs=[active])
        actual = active.location.clone()

    assert torch.equal(actual, expected)
    location_timeline = timeline.attr_to_timeline["location"]
    selected = location_timeline.rows_for_mob_ids({active.id})
    assert selected.numel() < location_timeline.pointer


def test_updaters_contribute_traced_mobs_to_active_materialization():
    scene = SceneManager.reset()
    mob = Mob().spawn(animate=False)
    updater_id = mob.add_updater(lambda _m, _dt: None)
    timeline = scene.timeline_manager
    times = torch.tensor([0.25])
    functions = timeline.function_timeline.get_functions_for_times(times)
    updaters = timeline.function_timeline.get_updaters_for_times(times)
    assert timeline._active_mob_ids([mob], functions, updaters) == {mob.id}
    mob.remove_updater(updater_id)


def test_removing_dependent_updater_holds_boundary_state_without_reversal():
    scene = SceneManager.reset()
    hub = Mob().spawn(animate=False)
    satellite = Mob().spawn(animate=False)

    with Seq():
        spin_id = hub.add_updater(
            lambda mob, time_elapsed: mob.rotate(time_elapsed * 90, OUT)
        )
        orbit_id = satellite.add_updater(
            lambda mob, _time_elapsed: mob.move_to(
                hub.get_center() + hub.get_right_direction()
            )
        )
        Scene.wait(1)
        removal_time = scene.animation_manager.context.timespan.current_time

        # The dependent updater is deliberately removed first, matching the
        # shapes_and_timeline scene that exposed the one-frame reversal.
        satellite.remove_updater(orbit_id)
        hub.remove_updater(spin_id)
        Scene.wait(0.1)

    times = torch.tensor([removal_time - 0.01, removal_time, removal_time + 0.01])
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times)
        satellite_locations = satellite.location[:, 0].clone()
        hub_directions = hub.get_right_direction()[:, 0].clone()
        timeline.clear_buffers()

    expected_boundary = torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(satellite_locations[1], expected_boundary)
    torch.testing.assert_close(satellite_locations[1], hub_directions[1])
    torch.testing.assert_close(satellite_locations[2], satellite_locations[1])
    assert (
        torch.linalg.vector_norm(satellite_locations[1] - satellite_locations[0]) < 0.02
    )


def test_active_mob_collection_walks_each_hierarchy_edge_once():
    SceneManager.reset()
    root = Mob()
    child = Mob()
    grandchild = Mob()
    child.add_children(grandchild)
    root.add_children(child)

    assert root.scene.timeline_manager._collect_mob_ids([root]) == {
        root.id,
        child.id,
        grandchild.id,
    }


def test_endpoint_materialization_reuses_layout_and_tracks_live_timing():
    class StubMob:
        def __init__(self, mob_id):
            self.id = mob_id

    timeline = AttributeTimeline(1, record_end_points=True)
    first = StubMob(1)
    second = StubMob(2)
    first_rows = timeline.add(first, torch.ones((3, 1)))
    second_rows = timeline.add(second, torch.ones((2, 1)))

    first_span = TimelineSpan(0, 2)
    first_lifespan = Lifespan()
    first_lifespan.start = first_span.get_time(0.5)
    first_lifespan.end = first_span.get_time(1.5)
    second_span = TimelineSpan(0, 4)
    second_lifespan = Lifespan()
    second_lifespan.start = second_span.get_time(2)
    timeline.set_start_point(first, first_lifespan)
    timeline.set_end_point(first, first_lifespan)
    timeline.set_start_point(second, second_lifespan)

    timeline._refresh_end_points()
    initial = timeline._end_points
    assert torch.all(initial[0, first_rows, 0] == 0.5)
    assert torch.all(initial[0, first_rows, 1] == 1.5)
    assert torch.all(initial[0, second_rows, 0] == 2)
    assert torch.all(initial[0, second_rows, 1] == 1e12)

    timeline._refresh_end_points()
    assert timeline._end_points is initial

    # A new Mob extends only the pending part of the row-owner layout. The
    # already-expanded prefix remains identical, including uneven row counts.
    old_layout = timeline._endpoint_layout_cache
    old_start_rows = old_layout[0].rows[: old_layout[0].used].clone()
    old_start_owners = old_layout[0].owners[: old_layout[0].used].clone()
    third = StubMob(3)
    third_rows = timeline.add(third, torch.ones((4, 1)))
    third_lifespan = Lifespan()
    third_lifespan.start = TimelineSpan(0, 6).get_time(3)
    timeline.set_start_point(third, third_lifespan)
    timeline._refresh_end_points()
    extended_layout = timeline._endpoint_layout_cache
    assert extended_layout is old_layout
    assert torch.equal(
        extended_layout[0].rows[: old_start_rows.numel()], old_start_rows
    )
    assert torch.equal(
        extended_layout[0].owners[: old_start_owners.numel()], old_start_owners
    )
    assert torch.all(timeline._end_points[0, third_rows, 0] == 3)

    # TimelineEvents are live: rescaling their span invalidates values but not
    # the cached row-owner layout.
    layout = timeline._endpoint_layout_cache
    first_span.end = 4
    timeline._refresh_end_points()
    assert timeline._end_points is not initial
    assert timeline._endpoint_layout_cache is layout
    assert torch.all(timeline._end_points[0, first_rows, 0] == 1)
    assert torch.all(timeline._end_points[0, first_rows, 1] == 3)

    # Replacing a lifespan or handing a Mob an equal-sized history block can
    # patch the cached layout in place.
    replacement_lifespan = Lifespan()
    replacement_lifespan.start = TimelineSpan(0, 5).get_time(2)
    layout = timeline._endpoint_layout_cache
    timeline.set_start_point(first, replacement_lifespan)
    timeline._refresh_end_points()
    assert timeline._endpoint_layout_cache is layout
    assert torch.all(timeline._end_points[0, first_rows, 0] == 2)

    spare = StubMob(4)
    spare_rows = timeline.add(spare, torch.ones((3, 1)))
    timeline.reassign_inds(first.id, spare_rows)
    timeline._refresh_end_points()
    assert timeline._endpoint_layout_cache is layout
    assert torch.all(timeline._end_points[0, spare_rows, 0] == 2)
    assert torch.all(timeline._end_points[0, first_rows, 0] == 1e12)

    # Reallocation is not append-only: overwrite moves the Mob to new rows and
    # therefore has to discard, then rebuild, the full cached layout.
    layout = timeline._endpoint_layout_cache
    replacement_rows = timeline.add(first, torch.ones((1, 1)), overwrite=True)
    assert timeline._endpoint_layout_cache is layout
    timeline._refresh_end_points()
    assert timeline._endpoint_layout_cache is not layout
    assert torch.all(timeline._end_points[0, replacement_rows, 0] == 2)
    assert torch.all(timeline._end_points[0, first_rows, 0] == 1e12)
