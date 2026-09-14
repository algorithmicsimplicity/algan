"""Project authoring and the tabbed viewer's real HTTP/session boundary.

Frames are deterministic tiny tensors: these tests exercise session ownership,
requests, handoff and cleanup without paying for kernel compilation.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from PIL import Image

from algan import (
    PREVIEW,
    SETTINGS,
    Circle,
    Project,
    Scene,
    SceneManager,
    Speech,
    Square,
)
from algan.errors import AlganConfigurationError
from algan.project import _get_active_project_run
from algan.viewer.session import ViewerSession

TINY = PREVIEW.set(resolution=(24, 16), frames_per_second=4)


def get(handle, path, *, raw=False, method="GET"):
    request = urllib.request.Request(handle.url_for(path), method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read() if raw else json.load(response)


@pytest.fixture
def fake_frames(monkeypatch):
    def render(session, start, end, *, store, **kwargs):
        assert Scene.current() is session.scene
        value = 40 if session.scene.actors[0].__class__.__name__ == "Square" else 180
        if store:
            for index in range(start, end):
                session._store(
                    index,
                    torch.full(
                        (session.height, session.width, 3), value, dtype=torch.uint8
                    ),
                )

    monkeypatch.setattr(ViewerSession, "_render_range", render)


@pytest.fixture
def project_viewer(fresh_scene, tmp_path, fake_frames):
    calls = []

    class Clip:
        duration = 1

    def intro():
        calls.append("intro")
        Square().spawn(animate=False)
        with Speech("First scene.", wait_at_end=0):
            Scene.wait(1)
        assert Scene.save_frame("ignored") == []
        assert Scene.save_video("ignored").status == "skipped"
        assert Scene.view() is None

    def outro():
        calls.append("outro")
        Scene.set_video_settings(TINY.set(frames_per_second=8))
        Circle().spawn(animate=False)
        with Speech("Second scene.", wait_at_end=0):
            Scene.wait(1)
        Scene.wait(1)

    project = Project(
        [intro, outro],
        video_settings=TINY,
        file_path=tmp_path / "project.mp4",
        transcript_directory=tmp_path / "transcripts",
        video_directory=tmp_path / "videos",
        screenshot_directory=tmp_path / "screenshots",
        speech_source=lambda _: Clip(),
    )
    before = SceneManager.instance().scene_stack
    handle = project.view(block=False, open_browser=False)
    try:
        yield project, handle, calls
    finally:
        handle.stop()
    assert SceneManager.instance().scene_stack == before


def test_project_view_authors_once_and_exports_nothing(project_viewer):
    project, handle, calls = project_viewer
    assert calls == ["intro", "outro"]
    assert not project.global_transcript_path.exists()
    assert not project.file_path.exists()
    assert not project.screenshot_directory.exists()
    assert _get_active_project_run() is None
    for scene_id in (1, 0, 1):
        get(handle, f"/api/scene?id={scene_id}", method="POST")
    assert calls == ["intro", "outro"]


def test_tabs_keep_stable_ids_names_and_scene_fps(project_viewer):
    _, handle, _ = project_viewer
    first = get(handle, "/api/state")
    assert first["scenes"] == [
        {"id": 0, "name": "0_intro"},
        {"id": 1, "name": "1_outro"},
    ]
    assert first["scene_id"] == 0
    assert first["fps"] == 4
    assert first["runtime"] == pytest.approx(1)
    assert first["resolution_name"] == "PREVIEW"
    second = get(handle, "/api/scene?id=1", method="POST")
    assert second["scene_id"] == 1
    assert second["fps"] == 8
    assert second["runtime"] == pytest.approx(2)
    assert second["scene_version"] > first["scene_version"]


def test_frames_hierarchy_and_transcript_belong_to_selected_scene(project_viewer):
    _, handle, _ = project_viewer
    for scene_id, kind, word, value in [
        (0, "Square", "First", 40),
        (1, "Circle", "Second", 180),
    ]:
        state = get(handle, f"/api/scene?id={scene_id}", method="POST")
        version = state["scene_version"]
        roots = get(handle, f"/api/hierarchy?s={version}")["roots"]
        assert any(row["type"] == kind for row in roots)
        transcript = get(handle, f"/api/transcript?s={version}")
        assert word in transcript["blocks"][0]["text"]
        png = get(handle, f"/frame/0.png?s={version}", raw=True)
        assert Image.open(io.BytesIO(png)).getpixel((0, 0)) == (value,) * 3


def test_stale_scene_requests_are_rejected_even_after_returning(project_viewer):
    _, handle, _ = project_viewer
    first = get(handle, "/api/state")
    get(handle, "/api/scene?id=1", method="POST")
    get(handle, "/api/scene?id=0", method="POST")
    for path, method in [
        ("/api/hierarchy", "GET"),
        ("/frame/0.png", "GET"),
        ("/api/resolution?name=SMOKE_TEST", "POST"),
    ]:
        separator = "&" if "?" in path else "?"
        with pytest.raises(urllib.error.HTTPError) as error:
            get(handle, f"{path}{separator}s={first['scene_version']}", method=method)
        assert error.value.code == 400


def test_invalid_and_repeated_selections_do_not_replace_worker(project_viewer):
    _, handle, _ = project_viewer
    original = handle.session.session_for_request()
    state = get(handle, "/api/state")
    same = get(handle, "/api/scene?id=0", method="POST")
    assert same["scene_version"] == state["scene_version"]
    for selector, status in [("99", 404), ("bad", 400), ("-1", 404)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(handle, f"/api/scene?id={selector}", method="POST")
        assert error.value.code == status
    assert handle.session.session_for_request() is original


def test_selection_requires_session_token(project_viewer):
    _, handle, _ = project_viewer
    request = urllib.request.Request(
        handle._server.origin + "/api/scene?id=1", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 403
    assert handle.session.state()["scene_id"] == 0


def test_selection_closes_previous_worker_and_handle_closes_last(project_viewer):
    _, handle, _ = project_viewer
    old = handle.session.session_for_request()
    worker = old._worker
    get(handle, "/api/scene?id=1", method="POST")
    assert old._closed
    assert not worker.is_alive()
    assert old._worker is None
    active = handle.session.session_for_request()
    handle.stop()
    assert active._closed
    assert active._worker is None


def test_scene_subset_and_explicit_settings(fresh_scene, tmp_path, fake_frames):
    def intro():
        Square().spawn(animate=False)

    def outro():
        Circle().spawn(animate=False)

    project = Project([intro, outro], file_path=tmp_path / "out.mp4")
    with project.view(
        ["1_outro", 1], video_settings=TINY, block=False, open_browser=False
    ) as handle:
        state = get(handle, "/api/state")
        assert state["scenes"] == [{"id": 1, "name": "1_outro"}]
        assert state["scene_id"] == 1
        assert state["total_frames"] == 1
        assert state["fps"] == TINY.frames_per_second
        assert state["resolution"] == list(TINY.resolution)
    with pytest.raises(AlganConfigurationError, match="at least one"):
        project.view([])
    with pytest.raises(AlganConfigurationError, match="Unknown"):
        project.view("missing")


def test_authoring_failure_restores_context_and_never_launches(
    monkeypatch, fresh_scene, tmp_path
):
    import algan.viewer.viewer as launch

    before = SceneManager.instance().scene_stack

    def broken():
        raise ValueError("author failed")

    monkeypatch.setattr(
        launch, "_view_project", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(ValueError, match="author failed") as error:
        Project([broken], file_path=tmp_path / "out.mp4").view()
    assert SceneManager.instance().scene_stack == before
    assert _get_active_project_run() is None
    if hasattr(error.value, "__notes__"):
        assert "0_broken" in error.value.__notes__[0]


def test_project_render_settings_are_snapshotted_per_scene(
    monkeypatch, fresh_scene, tmp_path
):
    seen = []

    def intro():
        SETTINGS.raytracing.shadows = False
        Square().spawn(animate=False)

    def outro():
        SETTINGS.raytracing.shadows = True
        Circle().spawn(animate=False)

    def render(session, start, end, *, store, **kwargs):
        seen.append(
            (type(session.scene.actors[0]).__name__, SETTINGS.raytracing.shadows)
        )
        if store:
            for index in range(start, end):
                session._store(index, torch.zeros((16, 24, 3), dtype=torch.uint8))

    monkeypatch.setattr(ViewerSession, "_render_range", render)
    project = Project([intro, outro], file_path=tmp_path / "out.mp4")
    with project.view(video_settings=TINY, block=False, open_browser=False) as handle:
        get(handle, "/frame/0.png", raw=True)
        get(handle, "/api/scene?id=1", method="POST")
        get(handle, "/frame/0.png", raw=True)
    assert ("Square", False) in seen
    assert ("Circle", True) in seen
    assert SETTINGS.raytracing.shadows is True


def test_handoff_waits_for_inflight_scene_access(project_viewer):
    _, handle, _ = project_viewer
    previous = handle.session.session_for_request()
    entered = threading.Event()
    release = threading.Event()
    switched = threading.Event()

    def hold_scene():
        with previous._scene():
            entered.set()
            assert release.wait(5)

    def switch_scene():
        handle.session.select_scene(1)
        switched.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(hold_scene)
        assert entered.wait(5)
        switcher = pool.submit(switch_scene)
        try:
            assert not switched.wait(0.05)
        finally:
            release.set()
        holder.result(timeout=5)
        switcher.result(timeout=5)
    with pytest.raises(RuntimeError, match="closed"), previous._scene():
        pytest.fail("closed scene was accessed")


def test_closed_session_releases_waiting_frame_requests(monkeypatch, fresh_scene):
    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    session = ViewerSession(Scene.current(), TINY)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(session.frame, 0)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            result.result(timeout=5)


def test_failed_server_bind_cleans_up_worker(monkeypatch, fresh_scene):
    import algan.viewer.viewer as launch

    sessions = []

    def fail_server(session, **kwargs):
        sessions.append(session)
        raise OSError("port in use")

    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    monkeypatch.setattr(launch, "ViewerServer", fail_server)
    with pytest.raises(OSError, match="port in use"):
        Scene.view(TINY, open_browser=False, block=False)
    assert sessions[0]._closed
    assert sessions[0]._worker is None


def test_real_project_frames_survive_scene_switch_and_return(fresh_scene, tmp_path):
    from PIL import ImageChops

    def intro():
        Square().spawn(animate=False)
        Scene.wait(0.25)

    def outro():
        Circle().spawn(animate=False)
        Scene.wait(0.25)

    project = Project([intro, outro], file_path=tmp_path / "out.mp4")
    with project.view(video_settings=TINY, block=False, open_browser=False) as handle:
        # A cold compiler can take longer than ordinary HTTP test requests.
        def frame():
            with urllib.request.urlopen(
                handle.url_for("/frame/0.png"), timeout=300
            ) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")

        first = frame()
        get(handle, "/api/scene?id=1", method="POST")
        second = frame()
        get(handle, "/api/scene?id=0", method="POST")
        repeated = frame()
    assert ImageChops.difference(first, second).getbbox() is not None
    assert all(
        high <= 2 for low, high in ImageChops.difference(first, repeated).getextrema()
    )
