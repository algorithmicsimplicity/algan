"""The viewer's per-pixel inspector: what is behind a pixel, and in what order.

These pin the three things the inspector cannot get subtly wrong without lying to
whoever is reading it:

* **which pixel** a query lands on. The renderer numbers rows from the bottom and
  an image numbers them from the top, so a viewer that guessed would report the
  fragments of the pixel mirrored about the middle -- plausible-looking, and
  wrong. ``test_pixel_rows_are_not_flipped`` is that assertion.
* **what order** fragments come in, and that a nearer surface really is nearer.
* **whose** surface it is: a mesh id is only useful if it names the Mob back.

All of these render, so none is marked ``fast``. Every one takes ``fresh_scene``:
scene isolation is opt-in in this suite, and two of these count the surfaces in a
frame, which a mob left behind by the previous test would quietly change.
"""

from __future__ import annotations

import pytest
import torch

from algan import BLUE, OUT, PREVIEW, RED, UP, Cube, Scene, Square
from algan.rendering import fragment_capture
from algan.viewer.pixels import PixelRecord

TINY = PREVIEW.set(resolution=(64, 36))


def last_frame(scene, settings=TINY):
    """The final frame of the scene as authored.

    Always used instead of an arbitrary index, because a ``spawn()`` is itself a
    one-second animation: a second mob spawned after a first does not exist until
    a second in, and a test that looked at frame 5 of a 10 fps scene would be
    asking about a frame the mob is not in yet.
    """
    duration = scene._recorded_end_time_for_render()
    return max(0, round(duration * settings.frames_per_second) - 1)


def capture_frame(scene, frame=None, settings=TINY):
    """Render one frame with the inspector armed; return its record and pixels."""
    if frame is None:
        frame = last_frame(scene, settings)
    previous = scene.video_settings
    scene.set_video_settings(settings)
    fragment_capture.arm()
    image = None
    try:
        with (
            torch.inference_mode(),
            scene.timeline_manager.preserving_authoring_state(
                preserve_replay_resolution=False
            ),
        ):
            for batch in scene.get_frames(frame, frame + 1):
                if batch.shape[0]:
                    image = batch[-1]
    finally:
        captures = fragment_capture.disarm()
        scene.set_video_settings(previous)
    return captures, image


def test_capture_is_off_unless_armed(fresh_scene):
    """The hook must cost nothing on a render nobody is inspecting."""
    assert fragment_capture.is_armed() is False
    Cube().spawn().rotate(45, UP)
    captures, image = capture_frame(Scene.current(), 5)
    assert captures, "an armed render should produce a record"
    assert fragment_capture.is_armed() is False, "disarm must leave it off"


def test_fragments_are_sorted_nearest_first_and_name_their_mob(fresh_scene):
    cube = Cube().set_color(BLUE).spawn()
    cube.rotate(60, UP)
    scene = Scene.current()
    captures, image = capture_frame(scene)
    record = PixelRecord(captures[0], {cube.id: cube})

    deepest = None
    for y in range(record.height):
        for x in range(record.width):
            found = record.fragments(x, y)
            if len(found) >= 2:
                deepest = found
                break
        if deepest:
            break
    assert deepest, "a rotated cube should show front and back faces somewhere"

    depths = [f["depth"] for f in deepest]
    assert depths == sorted(depths), "fragments must come out nearest first"
    assert all(f["kind"] == "triangle" for f in deepest)
    # A Cube declares one mesh key for all twelve triangles, so every face of it
    # is one surface -- and that surface names the Mob that authored it.
    assert {f["mesh_id"] for f in deepest} == {0}
    assert {f["mob"] for f in deepest} == {f"Cube #{cube.id}"}
    assert {f["mob_id"] for f in deepest} == {cube.id}


def test_two_mobs_are_told_apart(fresh_scene):
    """Overlapping mobs must not be merged into one surface."""
    back = Cube().set_color(BLUE).spawn()
    front = Cube().set_color(RED)
    front.location = front.location + OUT * 1.2
    front.scale_coefficient = front.scale_coefficient * 0.4
    front.spawn()
    scene = Scene.current()
    captures, image = capture_frame(scene)
    record = PixelRecord(captures[0], {back.id: back, front.id: front})

    overlapped = None
    for y in range(record.height):
        for x in range(record.width):
            found = record.fragments(x, y)
            if len({f["mob_id"] for f in found if f["mob_id"] is not None}) >= 2:
                overlapped = found
                break
        if overlapped:
            break
    assert overlapped, "the small cube should sit in front of the large one"
    assert len({f["mesh_id"] for f in overlapped}) >= 2
    # Nearest first means the front cube's fragment comes before the back one's.
    first = next(f for f in overlapped if f["mob_id"] is not None)
    assert first["mob_id"] == front.id


def test_pixel_rows_are_not_flipped(fresh_scene):
    """A pixel's fragments must belong to the pixel the image shows there.

    The kernel numbers rows from the bottom of the frame and an image numbers
    them from the top. Getting that backwards mirrors every lookup about the
    horizontal midline, which is invisible on a symmetric scene -- so this one
    is deliberately not symmetric: the cube sits high in frame, and the rows the
    image finds it in are the rows that must have fragments.
    """
    cube = Cube().set_color(BLUE)
    cube.location = cube.location + UP * 1.6
    cube.spawn()
    cube.rotate(45, UP)
    scene = Scene.current()
    captures, image = capture_frame(scene)
    record = PixelRecord(captures[0], {cube.id: cube})

    lit = (image.sum(-1) > 0).nonzero()
    assert lit.numel(), "the cube should be visible"
    rows = lit[:, 0]
    assert int(rows.max()) < record.height // 2, (
        "the cube was placed high, so it must occupy the image's top half"
    )

    covered = [
        (int(y), int(x)) for y, x in lit.tolist() if record.fragments(int(x), int(y))
    ]
    assert covered, "every lit pixel should have fragments behind it"
    # And the mirrored rows -- the empty bottom half -- must have none.
    for y, x in lit.tolist()[:40]:
        mirrored = record.height - 1 - int(y)
        assert not record.fragments(int(x), mirrored), (
            f"row {mirrored} is empty in the image but reported fragments; "
            "the row convention is inverted"
        )


def test_circuit_fragments_decode_their_circuit_and_border(fresh_scene):
    """A 2-D mob is a Bezier circuit, and reports what it can rather than a
    triangle's fields.

    A circuit carries no ``tri_obj`` entry, so it has no mesh id and -- until
    the 2-D mob builders stamp mesh keys too -- no Mob either. What it does have
    is its circuit index and the border weight folded into the same ref, and
    both must come back out with the packing's own arithmetic.
    """
    Square().set_color(RED).spawn()
    scene = Scene.current()
    captures, image = capture_frame(scene)
    record = PixelRecord(captures[0], {})

    found = None
    for y in range(record.height):
        for x in range(record.width):
            hits = [f for f in record.fragments(x, y) if f["kind"] == "circuit"]
            if hits:
                found = hits[0]
                break
        if found:
            break
    assert found, "a spawned Square should put a circuit fragment somewhere"
    assert found["mesh_id"] is None, "a circuit has no triangle surface id"
    assert found["circuit"] >= 0
    assert 0.0 <= found["border"] <= 1.0
    # The packing is ``-((circuit << 8) + round(border*255) + 1)``; decoding it
    # any other way lands on a plausible-looking wrong circuit.
    assert found["primitive"] == -(
        (found["circuit"] << 8) + round(found["border"] * 255) + 1
    )


def test_uncovered_pixel_reports_no_fragments(fresh_scene):
    """Background is a real answer, not a missing one."""
    cube = Cube().spawn()
    cube.rotate(45, UP)
    scene = Scene.current()
    captures, image = capture_frame(scene)
    record = PixelRecord(captures[0], {cube.id: cube})
    dark = (image.sum(-1) == 0).nonzero()
    assert dark.numel(), "the scene should not fill the whole frame"
    y, x = (int(v) for v in dark[0])
    assert record.fragments(x, y) == []


def test_albedo_matches_the_authored_colour(fresh_scene):
    """A flat-coloured mob's albedo is the colour the script asked for.

    The renderer works in linear light, so the number to compare against the
    authored colour is the sRGB-encoded one.
    """
    cube = Cube().set_color(BLUE).spawn()
    cube.rotate(45, UP)
    scene = Scene.current()
    captures, _ = capture_frame(scene)
    record = PixelRecord(captures[0], {cube.id: cube})

    fragment = None
    for y in range(record.height):
        for x in range(record.width):
            found = record.fragments(x, y)
            if found and found[0]["rgb_srgb"]:
                fragment = found[0]
                break
        if fragment:
            break
    assert fragment, "a visible cube should have a fragment with a colour"
    expected = [float(v) for v in BLUE.reshape(-1)[:3]]
    assert fragment["rgb_srgb"] == pytest.approx(expected, abs=2e-3)
    assert fragment["albedo_source"] in {"vertex", "texture"}


def test_view_reports_fragments_for_a_clicked_pixel(fresh_scene):
    """The session route the page actually calls, end to end."""
    cube = Cube().set_color(BLUE).spawn()
    cube.rotate(45, UP)
    handle = Scene.view(TINY, block=False, open_browser=False)
    try:
        session = handle.session
        image_frame = session.total_frames - 1
        found = None
        for y in range(0, session.height, 2):
            for x in range(0, session.width, 2):
                answer = session.pixel(image_frame, x, y)
                if answer["available"] and answer["fragments"]:
                    found = answer
                    break
            if found:
                break
        assert found, "the cube should be inspectable somewhere in frame"
        assert found["raw_fragments"] >= len(found["fragments"])
        first = found["fragments"][0]
        assert first["mob"] == f"Cube #{cube.id}"
        # Opacity comes from the Mob's timeline, not from the colour fetch.
        assert first["opacity"] == pytest.approx(1.0, abs=1e-6)
    finally:
        handle.stop()


def test_pixel_outside_the_frame_is_refused_not_guessed(fresh_scene):
    Square().spawn()
    handle = Scene.view(TINY, block=False, open_browser=False)
    try:
        answer = handle.session.pixel(0, 9999, 9999)
        assert answer["available"] is False
        assert "outside" in answer["reason"]
    finally:
        handle.stop()
