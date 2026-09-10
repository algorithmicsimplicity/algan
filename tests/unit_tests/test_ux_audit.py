"""Regression coverage for the September 2026 public-authoring UX audit."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from algan import (
    BLACK,
    ORIGIN,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    Color,
    Group,
    Scene,
    Square,
)
from algan.animation_timeline.animation_contexts import Lag, Seq, Sync
from algan.errors import AlganConfigurationError
from algan.project import _ProjectScene, _ProjectSceneRun
from algan.utils.python_utils import traverse


@pytest.mark.fast
@pytest.mark.parametrize("background", ["red", "#ff0000", RED])
def test_constructor_background_is_normalized_and_survives_defaults_and_reset(
    background,
):
    expected = background if torch.is_tensor(background) else Color(background)
    scene = Scene(SMOKE_TEST, background=background)
    assert torch.equal(scene.background, expected)
    scene.set_background(BLACK, overwrite=False)
    assert torch.equal(scene.background, expected)
    scene.reset()
    scene.set_background(BLACK, overwrite=False)
    assert torch.equal(scene.background, expected)


def test_constructor_background_image_uses_the_setter_path(tmp_path):
    from PIL import Image

    path = tmp_path / "background.png"
    Image.new("RGB", (3, 2), "red").save(path)
    scene = Scene(SMOKE_TEST, background=str(path))
    expected = scene.background.clone()
    assert expected.shape[-3:] == (32, 32, 5)
    scene.set_background(BLACK, overwrite=False)
    assert torch.equal(scene.background, expected)
    scene.reset()
    assert torch.equal(scene.background, expected)
    assert scene.background_is_set


@pytest.mark.fast
@pytest.mark.parametrize("context", [Seq, Sync, Lag])
@pytest.mark.parametrize("field", ["runtime", "runtime_per_part"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "one"])
def test_invalid_runtime_is_rejected_before_entering_context(context, field, value):
    with pytest.raises(AlganConfigurationError, match=field):
        context(**{field: value})


@pytest.mark.fast
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_wait_does_not_change_authoring_cursor(value):
    scene = Scene(SMOKE_TEST)
    context = scene.animation_manager.context
    before = (context.current_time, context.end_time)
    with pytest.raises(AlganConfigurationError, match="time"):
        scene.wait(value)
    assert (context.current_time, context.end_time) == before


@pytest.mark.fast
@pytest.mark.parametrize("magnitude", [3, 3e-20, 3e20])
def test_line_layout_uses_direction_not_vector_magnitude(magnitude):
    group = Group(Square(), Square())
    group.arrange_in_line(RIGHT, buffer=0.25)
    expected = [child.get_center().clone() for child in group]
    group.arrange_in_line(magnitude * RIGHT, buffer=0.25)
    for child, center in zip(group, expected):
        torch.testing.assert_close(child.get_center(), center)


@pytest.mark.fast
@pytest.mark.parametrize("method", ["arrange_in_line", "arrange_in_grid"])
def test_zero_layout_direction_is_rejected_without_moving_members(method):
    group = Group(Square(), Square())
    before = [child.location.clone() for child in group]
    kwargs = {"direction" if method == "arrange_in_line" else "row_direction": ORIGIN}
    with pytest.raises(AlganConfigurationError, match="direction"):
        getattr(group, method)(**kwargs)
    for child, location in zip(group, before):
        torch.testing.assert_close(child.location, location)


def _offset_anchor_member():
    member = Group(Square().move(2 * RIGHT))
    member.set_non_recursive(location=ORIGIN)
    return member


@pytest.mark.fast
def test_line_start_at_first_keeps_first_visual_center():
    first = _offset_anchor_member()
    group = Group(first, Square())
    center = first.get_center().clone()
    group.arrange_in_line(start_at_first=True)
    torch.testing.assert_close(first.get_center(), center)


@pytest.mark.fast
def test_arrange_between_points_spaces_centers_not_anchors():
    group = Group(_offset_anchor_member(), Square())
    group.arrange_between_points(-3 * RIGHT, 3 * RIGHT)
    torch.testing.assert_close(group[0].get_center(), -RIGHT.reshape(1, 1, 3))
    torch.testing.assert_close(group[1].get_center(), RIGHT.reshape(1, 1, 3))


@pytest.mark.fast
def test_component_filter_is_preserved_when_getting_grandchildren():
    component, ordinary = Square(), Square()
    child = Group(component, ordinary)
    child.components.append(component)
    parent = Group(child)
    assert parent.get_children(generation=1, include_components=False) == [ordinary]
    assert parent.get_children(generation=1) == [component, ordinary]


def test_traverse_treats_strings_and_bytes_as_atomic():
    class Label(str):
        pass

    label = Label("member")
    assert list(traverse(["abc", [b"abc", label], 3])) == ["abc", b"abc", label, 3]


@pytest.mark.fast
@pytest.mark.parametrize("value", ["Square", b"Square", None, 4, object()])
def test_group_invalid_members_raise_an_actionable_error(value):
    scene = Scene(SMOKE_TEST)
    actors = list(scene.actors)
    with pytest.raises(TypeError, match="Animatable"):
        Group(value)
    assert scene.actors == actors


@pytest.mark.parametrize("form", ["existing", "trailing", "pathlike"])
def test_project_frame_directory_keeps_the_file_inside_the_directory(tmp_path, form):
    SETTINGS.paths.set(output_filename="frame")
    directory = tmp_path / "stills"
    if form != "trailing":
        directory.mkdir()
    path = str(directory) + os.sep if form == "trailing" else directory
    if form == "existing":
        path = str(path)
    project = SimpleNamespace(screenshot_directory=tmp_path / "project-stills")
    scene = _ProjectScene(2, "example", lambda: None)
    run = _ProjectSceneRun(project, scene, "screenshots")
    assert run.prepare_frame_path(path) == directory / "s2_f0_frame.png"
    assert run.should_render_frame()


def test_project_bare_frame_name_still_uses_project_directory(tmp_path):
    project = SimpleNamespace(screenshot_directory=tmp_path / "project-stills")
    scene = _ProjectScene(2, "example", lambda: None)
    run = _ProjectSceneRun(project, scene, "screenshots")
    assert run.prepare_frame_path("detail") == (
        Path(project.screenshot_directory) / "s2_f0_detail.png"
    )
