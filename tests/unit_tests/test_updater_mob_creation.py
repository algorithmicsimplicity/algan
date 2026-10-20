"""Mobs built by an updater while a render is materializing frames.

An updater runs once per frame of every batch, so anything it constructs is
constructed again on every frame. Building a Mob there used to fail the render
outright -- ``IndexError: index N is out of bounds for dimension 0 with size
N``, raised from ``materialize_additional_rows`` and naming neither the updater
nor the Mob. It is reachable from ordinary code: the Manim compatibility layer
rebuilds its Mob tree on every ``set_value``, so a counting number does it on
every frame.

These Mobs are now *ephemeral*: they read and write their own constructed state
while the updater runs, and their rows are released when the replay ends, so a
render leaves the Scene exactly as the script wrote it. What an updater still
cannot do is reshape a Mob that existed before the render -- the batch's window
was materialized against the rows it had -- and that now says so.

Such a Mob holds one pose, so an updater that builds one is replayed one frame
at a time; replayed over a batch, a Mob built from per-frame values held the
batch's last frame on every frame, and ``become`` into it failed.
"""

from __future__ import annotations

import pytest
import torch

import algan
import algan.manim as mn
from algan.errors import AlganConfigurationError, UnsupportedFeatureError
from algan.settings.video_settings import SMOKE_TEST

FRAMES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])


def _row_counts(scene):
    return {
        name: timeline.pointer
        for name, timeline in scene.timeline_manager.attr_to_timeline.items()
    }


def test_constructing_a_mob_inside_an_updater_renders(tmp_path):
    with algan.Scene(video_settings=SMOKE_TEST):
        square = algan.Square(color=algan.BLUE).spawn()
        square.add_updater(lambda mob, t: algan.Circle(radius=0.1, color=algan.RED))
        algan.Scene.wait(1)
        result = algan.Scene.save_video(str(tmp_path / "built.mp4"), SMOKE_TEST)

    assert result.status == "rendered"
    assert result.output_path.exists()


def test_spawning_a_mob_inside_an_updater_renders(tmp_path):
    with algan.Scene(video_settings=SMOKE_TEST):
        square = algan.Square(color=algan.BLUE).spawn()
        square.add_updater(
            lambda mob, t: algan.Circle(radius=0.1, color=algan.RED).spawn()
        )
        algan.Scene.wait(1)
        result = algan.Scene.save_video(str(tmp_path / "spawned.mp4"), SMOKE_TEST)

    assert result.status == "rendered"


def test_a_render_leaves_the_scene_as_it_was_authored(tmp_path):
    """The point of making them ephemeral: rendering is not authoring.

    Without the rollback an updater that builds a Mob claims a fresh block of
    timeline rows on every frame of the render, and registers a fresh actor
    with it -- so the same script rendered twice would not be the same scene.
    """
    with algan.Scene(video_settings=SMOKE_TEST) as scene:
        square = algan.Square(color=algan.BLUE).spawn()
        square.add_updater(lambda mob, t: algan.Circle(radius=0.1, color=algan.RED))
        algan.Scene.wait(1)

        actors_before = len(scene.actors)
        rows_before = _row_counts(scene)

        algan.Scene.save_video(str(tmp_path / "first.mp4"), SMOKE_TEST)

        assert len(scene.actors) == actors_before
        assert _row_counts(scene) == rows_before


def test_an_updater_reads_the_mob_it_just_built():
    """An ephemeral Mob answers with its constructed state, not with zeros.

    Its rows sit past the end of the batch's materialized window, where every
    other reader is told to answer zero. That is right for a row the window
    chose not to materialize and wrong for one that did not exist when the
    window was built, and the difference is the whole value of building a Mob
    in an updater: you build it to read something off it.
    """
    seen = []

    with algan.Scene(video_settings=SMOKE_TEST):
        square = algan.Square(color=algan.BLUE).spawn()

        def follow(mob, t):
            probe = algan.Circle(radius=0.5, color=algan.RED).move_to(algan.RIGHT * 2)
            location = probe.location.reshape(-1, 3)
            seen.append(float(location[0, 0]))
            mob.set_location(probe.location)

        square.add_updater(follow)

    assert seen
    assert all(x == pytest.approx(2.0) for x in seen)


def test_reshaping_a_pre_existing_mob_inside_an_updater_is_explained(tmp_path):
    """A Manim-compat number rebuilds its glyphs, which needs different rows.

    Row layout is decided when the scene is authored, so this cannot work
    inside a render -- but the failure used to be a bare ``IndexError`` from
    the timeline, and the message now names the updater and the alternative
    that does work.
    """
    with algan.Scene(video_settings=SMOKE_TEST):
        tracker = mn.ValueTracker(0).spawn()
        number = mn.DecimalNumber(0).spawn()
        number.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))

        tracker.set_value(5)
        with pytest.raises(UnsupportedFeatureError, match="inside an updater"):
            algan.Scene.save_video(str(tmp_path / "counter.mp4"), SMOKE_TEST)


def test_numeric_display_counts_inside_an_updater(tmp_path):
    """The alternative the error above points at has to actually work."""
    with algan.Scene(video_settings=SMOKE_TEST):
        tracker = mn.ValueTracker(0).spawn()
        display = algan.DecimalNumber(0.0).spawn()
        display.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
        tracker.set_value(5)
        result = algan.Scene.save_video(str(tmp_path / "numeric.mp4"), SMOKE_TEST)

    assert result.status == "rendered"


def test_set_value_leaves_a_manim_number_visible(tmp_path):
    """``set_value`` used to make the number disappear, updater or not.

    ``_sync_manim_node_from_algan`` pushed the Algan side's style onto every
    Manim node including the point-less root, whose colour and opacity rows are
    placeholders. Manim treats a point-less node's style as a template:
    ``DecimalNumber.set_value`` rebuilds its glyphs and calls ``init_colors()``,
    which broadcast that placeholder over the whole family at opacity 0.
    """
    import cv2

    def brightest(value):
        with algan.Scene(video_settings=SMOKE_TEST):
            number = mn.DecimalNumber(0.0).spawn()
            if value is not None:
                number.set_value(value)
            result = algan.Scene.save_frame(str(tmp_path / f"number_{value}.png"))
        return cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED).max()

    assert brightest(None) > 0
    assert brightest(3.0) > 0


def test_a_mob_built_in_an_updater_can_be_read_outside_a_render():
    """Materializing without a render's working set reads the whole buffer.

    The rows of a Mob built during the replay lie past the end of that buffer,
    and reading them raised ``IndexError`` -- from ``set_state_to_times``, the
    interactive viewer, and ``remove_updater``, which all materialize this way.
    """
    with algan.Scene() as scene:
        dot = algan.Dot().spawn(animate=False)

        def follow(mob, t):
            marker = algan.Dot()
            marker.location = algan.RIGHT * 2
            mob.move_to(marker.get_center() + algan.UP)

        dot.add_updater(follow)
        algan.Scene.wait(1)

        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5]))
        assert dot.location.reshape(-1, 3).tolist() == [[2, 1, 0], [2, 1, 0]]


def test_removing_an_updater_that_builds_a_mob():
    """The writes kept at the removal boundary must not name the built Mob.

    It is discarded once the boundary is materialized, and a write to its rows
    sent the next materialization past the end of the buffer.
    """
    with algan.Scene() as scene:
        dot = algan.Dot().spawn(animate=False)

        def follow(mob, t):
            marker = algan.Dot()
            marker.location = algan.RIGHT * 2
            mob.move_to(marker.get_center() + algan.UP)

        dot.add_updater(follow)
        algan.Scene.wait(1)
        dot.remove_updater(-1)
        algan.Scene.wait(1)

        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5]))
        assert dot.location.reshape(-1, 3).tolist() == [[2, 1, 0], [2, 1, 0]]


def test_a_number_outgrowing_its_digits_in_an_updater_is_explained(tmp_path):
    """Its new digit slot would never be drawn, and left a dead Mob behind.

    The batch's Mobs are gathered before its updaters run, so a slot added
    there is not drawn, and the slot is discarded with the batch while the
    number goes on using it -- the next batch failed reading it. Room for the
    digits up front is what works.
    """
    with algan.Scene(video_settings=SMOKE_TEST):
        display = algan.DecimalNumber(0.0, decimal_places=1).spawn()
        display.add_updater(lambda mob, t: mob.set_value(t * 20))
        algan.Scene.wait(2)
        with pytest.raises(UnsupportedFeatureError, match="integer_places=2"):
            algan.Scene.save_video(str(tmp_path / "grown.mp4"), SMOKE_TEST)

    with algan.Scene(video_settings=SMOKE_TEST):
        display = algan.DecimalNumber(0.0, decimal_places=1, integer_places=2)
        display.spawn().add_updater(lambda mob, t: mob.set_value(t * 20))
        algan.Scene.wait(2)
        result = algan.Scene.save_video(str(tmp_path / "roomy.mp4"), SMOKE_TEST)

    assert result.status == "rendered"


def _dots_moving_apart(scene):
    a, b = algan.Dot(algan.LEFT * 2), algan.Dot(algan.RIGHT * 2)
    algan.Group(a, b).spawn(animate=False)
    return a, b


def test_a_mob_an_updater_builds_holds_each_frames_values():
    with algan.Scene() as scene:
        a, b = _dots_moving_apart(scene)
        middle = algan.Dot().spawn(animate=False)
        middle.add_updater(
            lambda m, t: m.move_to(
                algan.Line(a.get_center(), b.get_center()).get_center()
            )
        )
        with algan.Seq(runtime=2):
            a.move(algan.UP * 2)

        scene.timeline_manager.set_state_to_times(FRAMES)
        expected = (a.get_center() + b.get_center()) / 2
        assert len(set(expected[..., 1].reshape(-1).tolist())) == len(FRAMES)
        torch.testing.assert_close(middle.get_center(), expected)


def test_become_inside_an_updater_morphs_the_mob_in_place(tmp_path):
    """The Manim habit of redrawing a Mob every frame."""
    with algan.Scene() as scene:
        a, b = _dots_moving_apart(scene)
        line = algan.Line(algan.LEFT, algan.RIGHT).spawn(animate=False)
        line.add_updater(
            lambda m, t: m.become(algan.Line(a.get_center(), b.get_center()))
        )
        with algan.Seq(runtime=2):
            a.move(algan.UP * 2)

        scene.timeline_manager.set_state_to_times(FRAMES)
        corners = line.get_bounding_box()
        ends = torch.cat([a.get_center(), b.get_center()], -2)
        torch.testing.assert_close(corners.amin(-2), ends.amin(-2))
        torch.testing.assert_close(corners.amax(-2), ends.amax(-2))
        scene.timeline_manager.clear_buffers()

        result = scene.save_video(tmp_path / "redrawn.mp4", SMOKE_TEST)

    assert result.status == "rendered"


def test_become_cannot_hand_the_mob_away_inside_an_updater():
    """The replacement would take the screen; the updater would stay behind."""
    with algan.Scene():
        line = algan.Line(algan.LEFT, algan.RIGHT).spawn(animate=False)
        with pytest.raises(AlganConfigurationError, match="inside an updater"):
            line.add_updater(
                lambda m, t: m.become(
                    algan.Line(algan.UP, algan.DOWN), detach_history=True
                )
            )


def test_an_updater_that_writes_before_building_a_mob_writes_once():
    """Switching it to one frame at a time mid-call would redo the write."""
    with algan.Scene() as scene:
        dot = algan.Dot().spawn(animate=False)

        def drift(mob, t):
            mob.move(algan.RIGHT * t)
            algan.Dot()

        dot.add_updater(drift)
        algan.Scene.wait(2)

        # The first batch replays it over every frame, the next one at a time.
        for _ in range(2):
            scene.timeline_manager.set_state_to_times(FRAMES)
            torch.testing.assert_close(
                dot.location[..., 0].reshape(-1), FRAMES, atol=1e-5, rtol=0
            )
