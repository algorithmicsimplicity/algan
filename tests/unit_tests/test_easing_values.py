"""An animation context checks its easing where it is given.

An easing is first called when its animations are replayed for a render, so a
name, a number or a function that cannot take the progress failed there,
inside the timeline -- ``'str' object is not callable`` -- naming neither the
context nor the easing.
"""

from __future__ import annotations

from functools import partial

import pytest

from algan import RIGHT, SMOKE_TEST, ComposedEasing, Lag, Scene, Seq, Square, Sync
from algan.animation_timeline.animation_contexts import AnimationContext
from algan.constants import easings
from algan.errors import AlganConfigurationError


@pytest.mark.parametrize("context", [Seq, Sync, Lag, AnimationContext])
def test_an_easing_given_by_name_is_refused_with_the_function_to_pass(context):
    with pytest.raises(
        AlganConfigurationError,
        match=rf"^{context.__name__}'s easing is 'linear' \(str\), but an easing is "
        r"a function of the progress t, such as easings\.smooth\. Pass the "
        r"function itself: easings\.linear\.$",
    ):
        context(easing="linear")


@pytest.mark.parametrize(
    ("easing", "described"),
    [
        ("linnear", r"'linnear' \(str\)"),
        (3, r"3 \(int\)"),
        ([0, 1], r"\[0, 1\] \(list\)"),
    ],
)
def test_an_easing_that_is_no_function_is_refused(easing, described):
    with pytest.raises(
        AlganConfigurationError,
        match=rf"^Seq's easing is {described}, but an easing is a function of the "
        r"progress t, such as easings\.smooth\.$",
    ):
        Seq(easing=easing)


def _two(t, k):
    return t


@pytest.mark.parametrize(
    ("easing", "name", "problem"),
    [
        (lambda: 1, "<lambda>", "too many positional arguments"),
        (_two, "_two", "missing a required argument: 'k'"),
    ],
)
def test_an_easing_that_cannot_take_the_progress_is_refused(easing, name, problem):
    with pytest.raises(
        AlganConfigurationError,
        match=rf"^Seq's easing, {name}, cannot be called as easing\(t\): {problem}\. "
        r"Algan calls an easing with the progress t",
    ):
        Seq(easing=easing)


def test_a_composed_easing_is_checked_too():
    with pytest.raises(
        AlganConfigurationError,
        match=r"^ComposedEasing's composed_easing is 'ease_out_bounce' \(str\), .+ "
        r"Pass the "
        r"function itself: easings\.ease_out_bounce\.$",
    ):
        ComposedEasing("ease_out_bounce")


def test_a_bad_easing_is_refused_before_anything_is_recorded():
    with Scene(video_settings=SMOKE_TEST) as scene:
        square = Square(scene=scene).spawn()
        recorded = len(scene.timeline_manager.function_timeline.function_applications)
        with pytest.raises(AlganConfigurationError), Seq(easing="linear"):
            square.move(RIGHT)
        assert (
            len(scene.timeline_manager.function_timeline.function_applications)
            == recorded
        )


class _Eased:
    def __call__(self, t):
        return t


@pytest.mark.parametrize(
    "easing",
    [
        *(getattr(easings, name) for name in easings.__all__),
        lambda t: t**2,
        lambda t, *rest: t,
        partial(easings.smooth, inflection=5.0),
        _Eased(),
        None,
    ],
)
def test_every_function_of_the_progress_is_taken(easing):
    assert Seq(easing=easing).easing is easing
    assert ComposedEasing(easing).composed_easing is easing
