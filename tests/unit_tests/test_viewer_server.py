"""The viewer's HTTP surface, driven the way the page drives it.

The viewer is a local web app, so the whole of it below the browser is testable
here: a real server on a real port, answering real requests. That is the point of
these -- nothing needs a display, and the parts a browser would exercise (the
tree, the attribute panel, the frame images) are just routes.

Rendering is the slow half, so the scenes here are tiny and everything that can
share one server does. None of this is marked ``fast``: it renders, and scene
isolation (``fresh_scene``) is opt-in in this suite, so every test asks for it.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from algan import PREVIEW, RIGHT, Scene, Square

#: Small enough that a frame costs almost nothing, big enough to have an inside.
TINY = PREVIEW.set(resolution=(48, 27))


@pytest.fixture
def viewer(fresh_scene):
    """A running viewer over a two-second scene, stopped afterwards."""
    square = Square().spawn()
    square.move(RIGHT * 0.5)
    handle = Scene.view(TINY, block=False, open_browser=False)
    try:
        yield handle
    finally:
        handle.stop()


def fetch(handle, path, *, raw=False, timeout=300):
    with urllib.request.urlopen(handle.url.rstrip("/") + path, timeout=timeout) as r:
        return r.read() if raw else json.load(r)


def post(handle, path, *, timeout=300):
    request = urllib.request.Request(
        handle.url.rstrip("/") + path, data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return json.load(r)


def test_state_describes_the_scene(viewer):
    state = fetch(viewer, "/api/state")
    assert (state["width"], state["height"]) == TINY.resolution
    assert state["fps"] == TINY.frames_per_second
    # A spawn and a move are one second each, and the frame count follows the
    # duration at the viewer's frame rate.
    assert state["duration"] == pytest.approx(2.0, abs=1e-6)
    assert state["total_frames"] == round(state["duration"] * state["fps"])
    assert state["error"] is None


def test_frame_route_returns_a_png_of_the_right_size(viewer):
    from PIL import Image

    data = fetch(viewer, "/frame/0.png", raw=True)
    assert data[:4] == b"\x89PNG"
    assert Image.open(io.BytesIO(data)).size == TINY.resolution


def test_frames_can_be_fetched_out_of_order(viewer):
    """A seek is a jump, not a scan: a late frame must not need the early ones."""
    last = fetch(viewer, "/api/state")["total_frames"] - 1
    assert fetch(viewer, f"/frame/{last}.png", raw=True)[:4] == b"\x89PNG"


def test_hierarchy_lists_roots_and_the_scene_furniture(viewer):
    roots = fetch(viewer, "/api/hierarchy")["roots"]
    kinds = {row["kind"] for row in roots}
    # The camera and the lights are not in ``scene.actors``, so a viewer that
    # simply listed the actors would show neither.
    assert {"mob", "camera", "light"} <= kinds
    square = next(r for r in roots if r["type"] == "Square")
    assert square["spawned"] is True
    # Components are hidden by default, so a Square is a leaf rather than a
    # node with a control point under every corner.
    assert square["has_children"] is False


def test_components_are_available_when_asked_for(viewer):
    roots = fetch(viewer, "/api/hierarchy")["roots"]
    square = next(r for r in roots if r["type"] == "Square")
    hidden = fetch(viewer, f"/api/children?node={square['node']}")["children"]
    shown = fetch(viewer, f"/api/children?node={square['node']}&components=1")[
        "children"
    ]
    assert hidden == []
    assert shown, "the component mobs should still be reachable on request"


def test_attributes_report_values_and_explain_absences(viewer):
    roots = fetch(viewer, "/api/hierarchy")["roots"]
    square = next(r for r in roots if r["type"] == "Square")
    payload = fetch(viewer, f"/api/attrs?node={square['node']}&frame=0")
    rows = {row["name"]: row for row in payload["attributes"]}
    assert rows["location"]["value"] is not None
    assert rows["color"]["channels"] == ["r", "g", "b", "glow", "opacity"]
    # ``scale_coefficient`` is registered but owns no timeline rows -- it is read
    # back out of ``basis``. It is listed with a reason rather than dropped.
    assert rows["scale_coefficient"]["value"] is None
    assert rows["scale_coefficient"]["note"] == "derived"


def test_attributes_follow_the_playhead(viewer):
    """The panel shows the frame on screen, not the authoring cursor."""
    roots = fetch(viewer, "/api/hierarchy")["roots"]
    square = next(r for r in roots if r["type"] == "Square")
    last = fetch(viewer, "/api/state")["total_frames"] - 1

    def location(frame):
        payload = fetch(viewer, f"/api/attrs?node={square['node']}&frame={frame}")
        return next(r for r in payload["attributes"] if r["name"] == "location")[
            "value"
        ]

    assert location(0)[0] < location(last)[0], "the square moves right over the scene"


def test_pixel_answers_without_holding_the_request_open(viewer):
    """A slow inspection must be polled for, never waited out on one socket.

    The first inspection of a session compiles a Taichi kernel variant for the
    capture-armed render path -- measured in tens of seconds, and far worse
    while frames are still rendering. Holding one HTTP request open for that
    long is what made the browser abandon it and report ``Failed to fetch`` to
    the page, throwing away an answer that was still coming. So the route
    answers quickly with ``pending`` instead, and the page asks again.
    """
    import time

    deadline = time.monotonic() + 300
    payload = {"pending": True}
    while payload.get("pending") and time.monotonic() < deadline:
        started = time.monotonic()
        payload = fetch(viewer, "/api/pixel?frame=0&x=10&y=10")
        # Whatever the answer, it has to come back promptly. The wait is 3s;
        # allow generously for a loaded CI box without allowing a real block.
        assert time.monotonic() - started < 60, "the pixel route blocked"
    assert not payload.get("pending"), "the inspection never resolved"
    assert payload["x"] == 10
    assert payload["y"] == 10
    # Answered once, it is cached, so asking again is free rather than another
    # capture-armed render.
    again = time.monotonic()
    repeat = fetch(viewer, "/api/pixel?frame=0&x=10&y=10")
    assert time.monotonic() - again < 30
    assert repeat == payload


def test_pixel_outside_the_frame_is_refused_not_rendered(viewer):
    payload = fetch(viewer, "/api/pixel?frame=0&x=9999&y=9999")
    assert payload["available"] is False
    assert "outside" in payload["reason"]


def test_resolution_options_list_the_presets_and_the_two_named_ones(viewer):
    """The picker offers every built-in size, plus the Scene's and view()'s."""
    state = fetch(viewer, "/api/state")
    rows = {row["name"]: row for row in state["resolution_options"]}
    assert {"SMOKE_TEST", "PREVIEW", "HD", "UHD", "SCENE", "VIEW"} <= set(rows)
    # This viewer was opened with TINY, so that is the option it starts on.
    assert state["resolution_name"] == "VIEW"
    width, height = TINY.resolution
    # Labelled (height, width), which is the reverse of the (width, height) that
    # ``VideoSettings.resolution`` stores. Pinned because it is easy to
    # "helpfully" flip back while editing.
    assert rows["VIEW"]["label"] == f"View: ({height}, {width})"
    assert rows["HD"]["label"] == "HD: (1080, 1920)"


def test_changing_resolution_re_renders_at_the_new_size(viewer):
    from PIL import Image

    before = fetch(viewer, "/api/state")
    first = Image.open(io.BytesIO(fetch(viewer, "/frame/0.png", raw=True)))
    assert first.size == TINY.resolution

    after = post(viewer, "/api/resolution?name=SMOKE_TEST")
    assert after["resolution_name"] == "SMOKE_TEST"
    # The epoch is what tells the page its decoded frames -- and the browser's
    # own HTTP cache, which is told frames are immutable -- to let go.
    assert after["epoch"] > before["epoch"]
    # A preset brings its resolution, not its clock. SMOKE_TEST is 2 fps and
    # this viewer opened on TINY's 10; adopting the preset's rate would renumber
    # every frame the viewer reports, so the picker changes size only. This
    # caught a real bug: presets were built with the *Scene's* rate, so the
    # first switch away from the ``view()`` option silently re-timed the video.
    assert after["fps"] == before["fps"]
    assert after["total_frames"] == before["total_frames"]

    now = Image.open(io.BytesIO(fetch(viewer, "/frame/0.png", raw=True)))
    assert now.size == (32, 32)


def test_unknown_resolution_is_refused(viewer):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(viewer, "/api/resolution?name=NOPE")
    assert caught.value.code == 404


def test_unknown_node_and_route_are_reported_not_raised(viewer):
    for path in ("/api/children?node=1", "/api/attrs?node=1", "/nope"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            fetch(viewer, path)
        assert caught.value.code == 404
        assert "error" in json.load(caught.value)


def test_static_files_cannot_escape_the_package(viewer):
    """The static route serves this package's own files and nothing else."""
    assert fetch(viewer, "/static/viewer.js", raw=True).startswith(b"/*")
    for path in ("/static/../session.py", "/static/%2e%2e/session.py"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            fetch(viewer, path)
        assert caught.value.code == 404


def test_view_leaves_the_scene_authorable(fresh_scene):
    """A viewer is a look, not a render: nothing about the Scene may change.

    Mirrors ``test_default_render_keeps_the_scene_authorable`` -- the same
    contract ``save_video(reset=False)`` has, for the same reason. A viewer that
    left resolved replay windows behind would make every later render stop its
    animations early.
    """
    square = Square().spawn()
    square.move(RIGHT * 0.5)
    scene = Scene.instance()
    timeline, animations = scene.timeline_manager, scene.animation_manager
    end_before = scene._recorded_end_time_for_render()

    handle = Scene.view(TINY, block=False, open_browser=False)
    try:
        fetch(handle, "/frame/0.png", raw=True)
    finally:
        handle.stop()

    assert scene.timeline_manager is timeline
    assert scene.animation_manager is animations
    assert scene._recorded_end_time_for_render() == end_before
    assert square.is_spawned()
    # Still authorable, and the new animation still lands on the timeline.
    square.move(RIGHT * 0.5)
    assert scene._recorded_end_time_for_render() > end_before


def test_view_does_not_change_the_scenes_video_settings(fresh_scene):
    """The viewer renders small; the Scene must not be left that way."""
    Square().spawn()
    scene = Scene.instance()
    before = scene.video_settings
    handle = Scene.view(TINY, block=False, open_browser=False)
    try:
        fetch(handle, "/frame/0.png", raw=True)
    finally:
        handle.stop()
    # Equality, not identity: ``set_video_settings`` stores ``as_preset()`` of
    # what it is given, so a restore is always an equal new instance -- the same
    # for ``save_frame`` as for the viewer.
    assert scene.video_settings == before


def test_default_settings_keep_the_scenes_frame_rate(fresh_scene):
    """Frame indices the viewer reports must be the video's own.

    The PREVIEW preset is 10 fps. Adopting that wholesale would renumber a 30
    fps scene's frames, so the viewer takes the preset's size and the Scene's
    clock.
    """
    Square().spawn()
    scene = Scene.instance()
    scene.set_video_settings(PREVIEW.set(frames_per_second=30))
    handle = Scene.view(block=False, open_browser=False)
    try:
        assert handle.session.fps == 30
        assert tuple(handle.session.video_settings.resolution) == PREVIEW.resolution
    finally:
        handle.stop()
