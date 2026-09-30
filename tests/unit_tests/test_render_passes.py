"""Auxiliary render passes: from ``save_video``/``save_frame`` to files on disk.

``test_aux_passes.py`` checks what the renderer measures per pixel; this module
checks what becomes of it:

* the ``passes`` argument is validated early and canonically ordered;
* each pass's encoding round-trips exactly through the FFmpeg encoder that
  writes it (decoded back with FFmpeg, never Pillow, which reads 16-bit RGB
  PNG as 8-bit);
* a video's sequences hold exactly one image per video frame -- held frames
  included -- are published only after the video, replace an earlier render's
  frames without touching unrelated files, and come with a sidecar that names
  every object id;
* stills write one file per pass and a per-still sidecar.

Outside the fast suite: nothing here is liable to break from a change
elsewhere without ``test_aux_passes.py`` breaking first.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algan import (
    BLUE,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    Circle,
    Group,
    Off,
    Scene,
    SceneManager,
    Seq,
    Square,
    Text,
)
from algan._render_passes import (
    DEPTH_BACKGROUND,
    PASS_NAMES,
    PassEncoder,
    VideoPassJob,
    _ffmpeg_binary,
    encode_depth,
    encode_normal,
    normalize_passes,
)
from algan.errors import AlganConfigurationError

VIDEO = SMOKE_TEST.set(resolution=(48, 32), frames_per_second=10)


@pytest.fixture
def fresh_scene():
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        yield
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)


def _decode(path, pix_fmt, width, height, dtype, channels):
    """Every frame of an image (or ``%05d`` sequence) as raw pixels, via FFmpeg."""
    raw = subprocess.run(
        [_ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", pix_fmt, "-"],
        capture_output=True,
        check=True,
    ).stdout  # fmt: skip
    shape = (-1, height, width) + ((channels,) if channels > 1 else ())
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# The argument
# ---------------------------------------------------------------------------


def test_passes_argument_is_canonical_and_validated():
    assert normalize_passes(None) == ()
    assert normalize_passes(()) == ()
    assert normalize_passes("depth") == ("depth",)
    assert normalize_passes(["object_id", "depth", "depth"]) == ("depth", "object_id")
    assert normalize_passes(PASS_NAMES) == PASS_NAMES
    with pytest.raises(AlganConfigurationError, match="Unknown render pass 'z'"):
        normalize_passes(("depth", "z"))
    with pytest.raises(AlganConfigurationError):
        normalize_passes(3)


def test_an_unknown_pass_fails_before_rendering(fresh_scene, tmp_path):
    with Scene(video_settings=VIDEO) as scene:
        Square().spawn()
        with pytest.raises(AlganConfigurationError, match="Unknown render pass"):
            scene.save_video(tmp_path / "v.mp4", passes=("depth", "zdepth"))
        with pytest.raises(AlganConfigurationError, match="Unknown render pass"):
            scene.save_frame(tmp_path / "f.png", passes="normals")
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


def test_depth_and_normal_encodings():
    depth = torch.tensor([[[1.5, float("inf")]]])
    encoded = encode_depth(depth)
    assert encoded.dtype == np.dtype("<f4")
    assert encoded.tolist() == [[[1.5, DEPTH_BACKGROUND]]]

    normal = torch.tensor([[[[0.0, 0.0, 1.0], [0.3, 0.4, 0.5]]]])
    hit = torch.tensor([[[True, False]]])
    rgb = encode_normal(normal, hit)
    assert rgb.dtype == np.dtype("<u2")
    assert rgb[0, 0, 0].tolist() == [32768, 32768, 65535]
    assert rgb[0, 0, 1].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("name", "pix_fmt", "dtype", "channels"),
    [
        ("depth", "grayf32le", "<f4", 1),
        ("normal", "rgb48le", "<u2", 3),
        ("object_id", "rgb24", "u1", 3),
    ],
)
def test_each_pass_round_trips_exactly(tmp_path, name, pix_fmt, dtype, channels):
    width, height, frames = 7, 5, 3  # odd sizes on purpose
    rng = np.random.default_rng(0)
    if dtype == "<f4":
        data = rng.uniform(0.1, 1e4, (frames, height, width)).astype(dtype)
    else:
        top = 65535 if dtype == "<u2" else 255
        data = rng.integers(0, top + 1, (frames, height, width, channels)).astype(dtype)
    ext = "exr" if name == "depth" else "png"
    pattern = tmp_path / f"seq.%05d.{ext}"
    encoder = PassEncoder(
        _ffmpeg_binary(), name, width, height, 10, pattern, still=False
    )
    for frame in data:
        encoder.write_frame(frame)
    encoder.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"seq.{i:05d}.{ext}" for i in range(frames)
    ]
    back = _decode(pattern, pix_fmt, width, height, dtype, channels)
    assert np.array_equal(back, data)

    still = tmp_path / f"still.{ext}"
    encoder = PassEncoder(_ffmpeg_binary(), name, width, height, 10, still, still=True)
    encoder.write_frame(data[1])
    encoder.close()
    assert np.array_equal(
        _decode(still, pix_fmt, width, height, dtype, channels)[0], data[1]
    )


def test_an_encoder_failure_names_the_pass(tmp_path):
    encoder = PassEncoder(
        _ffmpeg_binary(),
        "normal",
        4,
        4,
        10,
        tmp_path / "missing" / "x.%05d.png",  # FFmpeg cannot open this
        still=False,
    )

    def write_until_closed():
        for _ in range(64):
            encoder.write_frame(np.zeros((4, 4, 3), "<u2"))
        encoder.close()

    with pytest.raises(RuntimeError, match="normal pass") as failure:
        write_until_closed()
    assert "exit code" in str(failure.value)


# ---------------------------------------------------------------------------
# Video sequences
# ---------------------------------------------------------------------------


def _fake_job(tmp_path, names, frames):
    from algan.render_loop import _VideoWriter

    job = VideoPassJob.create(names, tmp_path / "clip.mp4")
    scene = SimpleNamespace(
        num_pixels_screen_width=6, num_pixels_screen_height=4, frames_per_second=10
    )
    job.open(scene, _VideoWriter)
    depth = torch.full((frames, 4, 6), 3.0)
    aux = {
        "depth": depth,
        "normal": torch.zeros((frames, 4, 6, 3)),
        "mob_id": torch.full((frames, 4, 6), -2, dtype=torch.int32),
    }
    job.sink(aux)
    encoded = job.take(frames)
    for i in range(frames):
        job.put(encoded, i, copies=2 if i == 0 else 1)
    job.finish()
    return job, scene


def test_video_sequences_publish_one_image_per_frame(tmp_path):
    job, scene = _fake_job(tmp_path, ("depth", "normal"), frames=3)
    # Nothing is visible until the video is published.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "clip.depth_temp",
        "clip.normal_temp",
    ]
    (tmp_path / "clip.depth").mkdir()
    for stale in ("clip.depth.00007.exr", "clip.depth.00012.exr"):
        (tmp_path / "clip.depth" / stale).write_bytes(b"old")
    (tmp_path / "clip.depth" / "notes.txt").write_text("mine")

    paths = job.publish()
    job.release(scene)

    assert paths == {
        "depth": tmp_path / "clip.depth",
        "normal": tmp_path / "clip.normal",
    }
    # A held frame is written once per repeat: 2 + 1 + 1 images.
    assert sorted(p.name for p in (tmp_path / "clip.depth").iterdir()) == [
        "clip.depth.00000.exr",
        "clip.depth.00001.exr",
        "clip.depth.00002.exr",
        "clip.depth.00003.exr",
        "notes.txt",
    ]
    assert len(list((tmp_path / "clip.normal").iterdir())) == 4
    assert not (tmp_path / "clip.depth_temp").exists()
    meta = json.loads((tmp_path / "clip.passes.json").read_text())
    assert meta["frame_count"] == 4
    assert meta["resolution"] == [6, 4]
    assert meta["output"] == "clip.mp4"
    assert meta["passes"]["depth"]["path"] == "clip.depth"
    assert meta["passes"]["depth"]["background"] == DEPTH_BACKGROUND
    assert set(meta["passes"]) == {"depth", "normal"}


def test_a_mismatched_aux_batch_is_an_error(tmp_path):
    from algan.render_loop import _VideoWriter

    job = VideoPassJob.create("depth", tmp_path / "clip.mp4")
    scene = SimpleNamespace(
        num_pixels_screen_width=2, num_pixels_screen_height=2, frames_per_second=10
    )
    job.open(scene, _VideoWriter)
    try:
        with pytest.raises(RuntimeError, match="without their passes"):
            job.take(1)
        job.sink({"depth": torch.zeros((2, 2, 2))})
        with pytest.raises(RuntimeError, match="for 2 frames alongside 1"):
            job.take(1)
    finally:
        job.abort()
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _object_colours(meta):
    return {
        tuple(entry["rgb"]): entry for entry in meta["passes"]["object_id"]["objects"]
    }


def test_save_video_writes_every_pass_in_step_with_the_video(fresh_scene, tmp_path):
    from algan.rendering.pass_identity import pass_id_from_color

    with Scene(video_settings=VIDEO) as scene:
        square = Square(size=1.0, color=RED).move_to(LEFT * 2).spawn(animate=False)
        with Seq(runtime=1):
            square.move(OUT * 4)
        Scene.wait(1)  # a hold: static frames may be reused, passes must not drift
        result = scene.save_video(
            tmp_path / "clip.mp4", passes=("depth", "normal", "object_id")
        )
    assert result.rendered
    frames = int(round(2 * VIDEO.frames_per_second)) + 1
    for name, directory in result.passes.items():
        count = len(list(directory.iterdir()))
        assert count >= frames - 1, (name, count)
    counts = {len(list(d.iterdir())) for d in result.passes.values()}
    assert len(counts) == 1, "the passes disagree on the frame count"
    meta = json.loads((tmp_path / "clip.passes.json").read_text())
    assert meta["frame_count"] == counts.pop()

    w, h = VIDEO.resolution
    depth = _decode(
        result.passes["depth"] / "clip.depth.%05d.exr", "grayf32le", w, h, "<f4", 1
    )
    centre = depth[:, h // 2, w // 2 - int(w * 2 / 8)]
    # The square moves toward the camera, so its depth falls, then holds.
    assert centre[0] > centre[len(centre) // 2] >= centre[-1]
    assert depth[0, 0, 0] == DEPTH_BACKGROUND

    ids = _decode(
        result.passes["object_id"] / "clip.object_id.%05d.png", "rgb24", w, h, "u1", 3
    )
    colours = _object_colours(meta)
    hit = ids[-1, h // 2, w // 2 - int(w * 2 / 8)]
    assert tuple(hit) in colours
    entry = colours[tuple(hit)]
    assert entry["id"] == pass_id_from_color(tuple(int(c) for c in hit))
    assert entry["mobs"][0]["class"] == "Square"
    assert ids[0, 0, 0].tolist() == [0, 0, 0]


def test_object_ids_follow_owners_and_pass_index(fresh_scene, tmp_path):
    from algan.rendering.pass_identity import AUTO_ID_BASE

    with Scene(video_settings=VIDEO.set(resolution=(96, 48))) as scene:
        with Off():
            title = Text("ab", color=BLUE).move_to(LEFT * 3).spawn()
            pair = Group(
                Circle(radius=0.4, color=RED).move_to(RIGHT * 1),
                Circle(radius=0.4, color=RED).move_to(RIGHT * 3),
            ).spawn()
            marked = Square(size=0.8).move_to(LEFT * 0.5).spawn()
            marked.pass_index = 7
        result = scene.save_frame(tmp_path / "shot.png", passes="object_id")
    assert set(result.passes) == {"object_id"}
    meta = json.loads((tmp_path / "shot.passes.json").read_text())
    objects = meta["passes"]["object_id"]["objects"]
    by_class = {}
    for entry in objects:
        for mob in entry["mobs"]:
            by_class.setdefault(mob["class"], set()).add(entry["id"])
    # One object for the whole Text, one per Group member, and the explicit id.
    assert len(by_class["Text"]) == 1
    assert len(by_class["Circle"]) == 2
    assert by_class["Square"] == {7}
    assert all(i >= AUTO_ID_BASE for i in by_class["Text"] | by_class["Circle"])
    assert title.id + AUTO_ID_BASE in by_class["Text"]
    assert {c.id + AUTO_ID_BASE for c in pair.children} == by_class["Circle"]


def test_save_frame_writes_one_file_per_pass(fresh_scene, tmp_path):
    with Scene(video_settings=VIDEO) as scene:
        with Off():
            Square(size=1.5).spawn()
        results = scene.save_frame(
            tmp_path / "shot.png", at=[0.0, 0.0], passes=("normal", "depth")
        )
        skipped = scene.save_frame(
            tmp_path / "shot_0.0.png", overwrite=False, passes="depth"
        )
    assert skipped.status == "skipped"
    assert skipped.passes == {}
    for result in results:
        stem = result.output_path.with_suffix("")
        assert result.passes == {
            "depth": stem.with_name(stem.name + ".depth.exr"),
            "normal": stem.with_name(stem.name + ".normal.png"),
        }
        for path in result.passes.values():
            assert path.exists()
        w, h = VIDEO.resolution
        normal = _decode(result.passes["normal"], "rgb48le", w, h, "<u2", 3)[0]
        # A camera-facing square at the centre: n = (0, 0, 1).
        assert abs(int(normal[h // 2, w // 2, 2]) - 65535) <= 1
        assert abs(int(normal[h // 2, w // 2, 0]) - 32768) <= 64
        assert normal[0, 0].tolist() == [0, 0, 0]
        meta = json.loads(stem.with_name(stem.name + ".passes.json").read_text())
        assert meta["timestamp"] == 0.0
        assert set(meta["passes"]) == {"depth", "normal"}
