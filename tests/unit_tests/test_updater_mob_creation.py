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
"""

from __future__ import annotations

import pytest

import algan
import algan.manim as mn
from algan.errors import UnsupportedFeatureError
from algan.settings.video_settings import SMOKE_TEST


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
        display = algan.NumericDisplay(0.0).spawn()
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
