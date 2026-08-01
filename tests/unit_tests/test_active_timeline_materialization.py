import math

import torch

from algan import Mob
from algan.animation_timeline.animation_contexts import Off, Seq
from algan.constants import rate_funcs
from algan.constants.spatial import RIGHT
from algan.scene_manager import SceneManager


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
