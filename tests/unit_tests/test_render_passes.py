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
        "clip.mp4.depth_temp",
        "clip.mp4.normal_temp",
    ]
    (tmp_path / "clip.mp4.depth").mkdir()
    for stale in ("clip.mp4.depth.00007.exr", "clip.mp4.depth.00012.exr"):
        (tmp_path / "clip.mp4.depth" / stale).write_bytes(b"old")
    (tmp_path / "clip.mp4.depth" / "notes.txt").write_text("mine")

    paths = job.publish()
    job.release(scene)

    assert paths == {
        "depth": tmp_path / "clip.mp4.depth",
        "normal": tmp_path / "clip.mp4.normal",
    }
    # A held frame is written once per repeat: 2 + 1 + 1 images.
    assert sorted(p.name for p in (tmp_path / "clip.mp4.depth").iterdir()) == [
        "clip.mp4.depth.00000.exr",
        "clip.mp4.depth.00001.exr",
        "clip.mp4.depth.00002.exr",
        "clip.mp4.depth.00003.exr",
        "notes.txt",
    ]
    assert len(list((tmp_path / "clip.mp4.normal").iterdir())) == 4
    assert not (tmp_path / "clip.mp4.depth_temp").exists()
    meta = json.loads((tmp_path / "clip.mp4.passes.json").read_text())
    assert meta["frame_count"] == 4
    assert meta["resolution"] == [6, 4]
    assert meta["output"] == "clip.mp4"
    assert meta["passes"]["depth"]["path"] == "clip.mp4.depth"
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


def test_a_file_in_the_way_of_a_pass_directory_fails_before_rendering(tmp_path):
    (tmp_path / "clip.mp4.depth").write_text("not a directory")
    with pytest.raises(AlganConfigurationError, match="in the way"):
        VideoPassJob.create("depth", tmp_path / "clip.mp4")


def test_a_failed_publish_keeps_the_encoded_frames(tmp_path, monkeypatch):
    import algan._render_passes as passes_module

    job, scene = _fake_job(tmp_path, ("depth", "normal"), frames=2)
    real_replace = passes_module.os.replace

    def locked(src, dst):
        if "normal" in str(src):
            raise PermissionError("locked by an editor")
        return real_replace(src, dst)

    monkeypatch.setattr(passes_module.os, "replace", locked)
    with pytest.raises(RuntimeError, match="clip.mp4.normal_temp"):
        job.publish()
    job.discard()  # what the render's cleanup does next
    job.release(scene)
    assert len(list((tmp_path / "clip.mp4.depth").iterdir())) == 3
    assert len(list((tmp_path / "clip.mp4.normal_temp").iterdir())) == 3


@pytest.mark.parametrize("kind", ["missing", "not_executable"])
def test_an_unrunnable_ffmpeg_fails_before_the_still_renders(
    fresh_scene, tmp_path, kind
):
    binary = tmp_path / "ffmpeg"
    if kind == "not_executable":
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o644)  # exists, but cannot be run
    SETTINGS.paths.set(ffmpeg_binary=str(binary))
    with Scene(video_settings=VIDEO) as scene:
        Square().spawn()
        with pytest.raises(AlganConfigurationError, match="Cannot run FFmpeg"):
            scene.save_frame(tmp_path / "x.png", passes="depth")
    assert not (tmp_path / "x.png").exists()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_still_and_a_video_of_one_stem_keep_their_own_sidecars(fresh_scene, tmp_path):
    with Scene(video_settings=VIDEO) as scene:
        with Off():
            Square().spawn()
        scene.save_frame(tmp_path / "intro.png", passes=("depth", "object_id"))
        scene.save_video(tmp_path / "intro.mp4", passes="depth")
    still = json.loads((tmp_path / "intro.png.passes.json").read_text())
    video = json.loads((tmp_path / "intro.mp4.passes.json").read_text())
    assert still["output"] == "intro.png"
    assert set(still["passes"]) == {"depth", "object_id"}
    assert video["output"] == "intro.mp4"
    assert set(video["passes"]) == {"depth"}


def test_a_percent_sign_in_the_path_survives_the_sequence_pattern(
    fresh_scene, tmp_path
):
    folder = tmp_path / "50%off"
    folder.mkdir()
    with Scene(video_settings=VIDEO) as scene:
        with Off():
            Square().spawn()
        result = scene.save_video(folder / "take%d.mp4", passes="depth")
    directory = result.passes["depth"]
    assert directory == folder / "take%d.mp4.depth"
    names = sorted(p.name for p in directory.iterdir())
    assert names
    assert names[0] == "take%d.mp4.depth.00000.exr"


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
    meta = json.loads((tmp_path / "clip.mp4.passes.json").read_text())
    assert meta["frame_count"] == counts.pop()

    w, h = VIDEO.resolution
    depth = _decode(
        result.passes["depth"] / "clip.mp4.depth.%05d.exr", "grayf32le", w, h, "<f4", 1
    )
    centre = depth[:, h // 2, w // 2 - int(w * 2 / 8)]
    # The square moves toward the camera, so its depth falls, then holds.
    assert centre[0] > centre[len(centre) // 2] >= centre[-1]
    assert depth[0, 0, 0] == DEPTH_BACKGROUND

    ids = _decode(
        result.passes["object_id"] / "clip.mp4.object_id.%05d.png",
        "rgb24",
        w,
        h,
        "u1",
        3,
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
    meta = json.loads((tmp_path / "shot.png.passes.json").read_text())
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


def _ids_per_frame(result, width, height):
    from algan.rendering.pass_identity import pass_ids_from_colors

    rgb = _decode(
        result.passes["object_id"] / f"{result.output_path.name}.object_id.%05d.png",
        "rgb24",
        width,
        height,
        "u1",
        3,
    )
    ids = pass_ids_from_colors(torch.from_numpy(rgb.copy()))
    return [set(frame.unique().tolist()) - {0} for frame in ids]


@pytest.mark.parametrize("case", ["untagged", "group_tag", "self_tag_cross_family"])
def test_an_object_keeps_its_id_across_become(fresh_scene, tmp_path, case):
    """``become`` hands the earlier frames to a hidden history clone and, for
    a cross-kind morph, renders through stand-ins and a target-class
    replacement. None of that is the author's business: the object keeps one
    ID -- and its ``pass_index`` -- on every frame.
    """
    from algan import Sphere

    video = VIDEO.set(frames_per_second=4)
    w, h = video.resolution
    with Scene(video_settings=video) as scene:
        with Off():
            square = Square(size=1.5).spawn()
            group = Group(square).spawn()
        if case == "group_tag":
            group.pass_index = 6
        elif case == "self_tag_cross_family":
            square.pass_index = 6
        square.wait(1)
        square.become(
            Sphere(radius=0.8, add_to_scene=False)
            if case.endswith("cross_family")
            else Circle(add_to_scene=False)
        )
        scene.wait(1)
        result = scene.save_video(tmp_path / "morph.mp4", passes="object_id")
    frames = _ids_per_frame(result, w, h)
    seen = set().union(*frames)
    assert len(seen) == 1, f"one object, several ids over the shot: {frames}"
    if case != "untagged":
        assert seen == {6}
    assert all(frame == seen for frame in frames), frames


def _ids_per_half(result, width, height):
    """Per frame, the object ids seen in the left and in the right half."""
    from algan.rendering.pass_identity import pass_ids_from_colors

    rgb = _decode(
        result.passes["object_id"] / f"{result.output_path.name}.object_id.%05d.png",
        "rgb24",
        width,
        height,
        "u1",
        3,
    )
    ids = pass_ids_from_colors(torch.from_numpy(rgb.copy()))
    half = width // 2
    return [
        (
            set(frame[:, :half].unique().tolist()) - {0},
            set(frame[:, half:].unique().tolist()) - {0},
        )
        for frame in ids
    ]


class _Molecule(Group):
    """A composite that is an object of its own (not a pure container)."""


@pytest.mark.parametrize(
    "case",
    ["group_tag", "child_tags", "composite", "both_tagged", "no_detach"],
)
def test_children_keep_their_ids_across_a_hierarchy_become(fresh_scene, tmp_path, case):
    """A Group become()s child by child: each child morphs through its own
    stand-ins and is spliced out for its replacement. Links are positional,
    so a tag on the group reaches every child on every frame, per-child tags
    stay per child, a composite stays one object, and the source's explicit
    tag wins over the target's.
    """
    from algan import Sphere

    video = VIDEO.set(resolution=(64, 32), frames_per_second=4)
    w, h = video.resolution
    with Scene(video_settings=video) as scene:
        with Off():
            a = Square(size=1.2).move_to(LEFT * 2)
            b = Square(size=1.2).move_to(RIGHT * 2)
            holder = (_Molecule if case == "composite" else Group)(a, b).spawn()
        if case == "group_tag":
            holder.pass_index = 6
        elif case in ("child_tags", "no_detach"):
            a.pass_index = 1
            b.pass_index = 2
        target_a = Sphere(radius=0.6, add_to_scene=False).move_to(LEFT * 2)
        target_b = (
            Square(size=1.2, add_to_scene=False)
            if case == "no_detach"
            else Sphere(radius=0.6, add_to_scene=False)
        ).move_to(RIGHT * 2)
        # A composite becomes a composite; a plain Group, a plain Group.
        target_group = (_Molecule if case == "composite" else Group)(
            target_a, target_b, add_to_scene=False
        )
        if case == "both_tagged":
            a.pass_index = 6
            target_a.pass_index = 9
        holder.wait(1)
        holder.become(target_group, detach_history=case != "no_detach")
        scene.wait(1)
        result = scene.save_video(tmp_path / "hier.mp4", passes="object_id")
    halves = _ids_per_half(result, w, h)
    lefts = set().union(*(left for left, _ in halves))
    rights = set().union(*(right for _, right in halves))
    if case == "group_tag":
        assert lefts == rights == {6}, halves
    elif case == "child_tags":
        assert (lefts, rights) == ({1}, {2}), halves
    elif case == "composite":
        assert len(lefts) == 1, halves
        assert lefts == rights, "a composite is one object"
    elif case == "both_tagged":
        assert lefts == {6}, "the source's explicit tag wins, as on the same-kind route"
    elif case == "no_detach":
        assert 1 in lefts, halves
        assert 2 in rights, halves
        assert not lefts & rights, "two children must not collapse into one id"


@pytest.mark.parametrize(
    "case",
    [
        "group_to_member",
        "group_to_member_tagged",
        "group_and_member_tagged",
        "leaf_kept_in_group",
        "nested_group_tag",
        "text_in_group_tag",
        "piecewise_dissolve",
        "piecewise_dissolve_tagged",
    ],
)
def test_become_hands_structure_ids_to_what_replaces_it(fresh_scene, tmp_path, case):
    """What a hierarchy ``become`` retires besides its primitives -- the root,
    an inner Group, a Text inside a Group -- hands its ID and ``pass_index``
    on to what took its place, and to nothing else: a Group that becomes one
    of its members does not fold the other member into it, a member's own tag
    outranks its group's, a leaf kept inside its result stays itself, and a
    piecewise dissolve pairs its parts. Left half: the object on the left;
    right half: the one on the right.
    """
    video = VIDEO.set(resolution=(64, 32), frames_per_second=4)
    w, h = video.resolution

    def circle(x, radius=0.5):
        return Circle(radius=radius, add_to_scene=False).move_to(RIGHT * x)

    with Scene(video_settings=video) as scene:
        with Off():
            a = Square(size=1.0).move_to(LEFT * 2)
            b = Square(size=1.0).move_to(RIGHT * 2)
        if case.startswith("group_"):
            with Off():
                group = Group(a, b).spawn()
            if case != "group_to_member":
                a.pass_index = 3
            if case == "group_and_member_tagged":
                group.pass_index = 5
            group.wait(1)
            result = group.become(circle(-2, 0.6))
            assert result is a
            assert a.pass_index == (None if case == "group_to_member" else 3)
        elif case == "leaf_kept_in_group":
            with Off():
                leaf = Circle(radius=0.6).move_to(LEFT * 2).spawn()
            leaf.wait(1)
            result = leaf.become(
                Group(
                    circle(-2, 0.6),
                    Square(size=1.0, add_to_scene=False).move_to(RIGHT * 2),
                    add_to_scene=False,
                )
            )
            assert result.children[0] is leaf
            result.children[0].pass_index = 7
            result.children[1].pass_index = 8
        elif case == "nested_group_tag":
            with Off():
                near = Square(size=0.6).move_to(LEFT * 1.2)
                far = Square(size=0.6).move_to(LEFT * 2.6)
                inner = Group(far, near)
                group = Group(inner, b).spawn()
            inner.pass_index = 7
            group.wait(1)
            group.become(
                Group(
                    Group(circle(-2.6, 0.3), circle(-1.2, 0.3), add_to_scene=False),
                    circle(2),
                    add_to_scene=False,
                )
            )
        elif case == "text_in_group_tag":
            with Off():
                text = Text("ab").scale(2).move_to(LEFT * 2)
                group = Group(text, b).spawn()
            text.pass_index = 6
            group.wait(1)
            group.become(
                Group(
                    Text("abc", add_to_scene=False).scale(2).move_to(LEFT * 2),
                    circle(2),
                    add_to_scene=False,
                )
            )
        else:
            with Off():
                group = Group(a, b).spawn()
            if case.endswith("tagged"):
                a.pass_index = 1
                b.pass_index = 2
            group.wait(1)
            group.become(
                Group(circle(-2), circle(2), add_to_scene=False), strategy="dissolve"
            )
        scene.wait(1)
        result = scene.save_video(tmp_path / "structure.mp4", passes="object_id")
    from algan.rendering.pass_identity import AUTO_ID_BASE

    halves = _ids_per_half(result, w, h)
    lefts = set().union(*(left for left, _ in halves))
    rights = set().union(*(right for _, right in halves))
    seen = lefts | rights
    first, last = halves[0], halves[-1]
    if case.startswith("group_"):
        # The surplus square shrinks toward the circle, into the left half, so
        # the check is per object: two ids from the first frame to the last,
        # the left square's the circle's throughout.
        left_id = 3 if case != "group_to_member" else AUTO_ID_BASE + a.id
        right_id = 5 if case == "group_and_member_tagged" else AUTO_ID_BASE + b.id
        assert first == ({left_id}, {right_id}), halves
        assert seen == {left_id, right_id}, halves
        assert last[0] == {left_id}, halves
    elif case == "leaf_kept_in_group":
        assert first[0] == {7}, halves
        assert last == ({7}, {8}), halves
        assert seen == {7, 8}, halves
    elif case == "nested_group_tag":
        assert (lefts, rights) == ({7}, {AUTO_ID_BASE + b.id}), halves
    elif case == "text_in_group_tag":
        assert (lefts, rights) == ({6}, {AUTO_ID_BASE + b.id}), halves
    elif case == "piecewise_dissolve_tagged":
        assert (lefts, rights) == ({1}, {2}), halves
    else:
        # Untagged, each part keeps one id through the dissolve.
        assert len(lefts) == len(rights) == 1, halves
        assert not lefts & rights, halves


@pytest.mark.parametrize(
    "case",
    [
        "outer_tag",
        "outer_tag_dissolve",
        "outer_composite",
        "dissolve_surplus_part",
        "nested_dissolve",
        "lone_member_dissolve",
        "leaf_tag_reaches_its_group",
        "no_detach_into_group",
        "text_tag_stays_on_text",
    ],
)
def test_become_keeps_ids_above_and_below_what_it_replaces(fresh_scene, tmp_path, case):
    """The members a ``become`` leaves behind keep reaching what is above the
    Mob it replaced -- an outer group's tag, an outer composite's id -- a
    dissolve pairs parts one to one at every depth, and a tag stays with the
    object it was set on: a leaf's reaches the Group it became, a Text's stays
    on the Text. Left half: the object on the left; right half: the right.
    """
    from algan.rendering.pass_identity import AUTO_ID_BASE

    video = VIDEO.set(resolution=(64, 32), frames_per_second=4)
    w, h = video.resolution

    def circle(x, radius=0.5):
        return Circle(radius=radius, add_to_scene=False).move_to(RIGHT * x)

    with Scene(video_settings=video) as scene:
        with Off():
            a = Square(size=0.8).move_to(LEFT * 2.4)
            b = Square(size=0.8).move_to(LEFT * 1.0)
            x = Square(size=0.8).move_to(RIGHT * 2)
        if case.startswith("outer_"):
            with Off():
                inner = Group(a, b)
                outer = (_Molecule if case == "outer_composite" else Group)(
                    inner, x
                ).spawn()
            if case != "outer_composite":
                outer.pass_index = 4
            outer.wait(1)
            inner.become(
                circle(-2.4),
                strategy="dissolve" if case.endswith("dissolve") else "auto",
            )
        elif case == "dissolve_surplus_part":
            with Off():
                group = Group(a, x).spawn()
            a.pass_index = 1
            x.pass_index = 2
            group.wait(1)
            group.become(Group(circle(-2.4), add_to_scene=False), strategy="dissolve")
        elif case == "nested_dissolve":
            with Off():
                group = Group(Group(a, b), x).spawn()
            a.pass_index = 1
            group.wait(1)
            group.become(
                Group(
                    Group(circle(-2.4, 0.3), circle(-1.0, 0.3), add_to_scene=False),
                    circle(2),
                    add_to_scene=False,
                ),
                strategy="dissolve",
            )
        elif case == "lone_member_dissolve":
            with Off():
                group = Group(a).spawn()
            group.wait(1)
            group.become(circle(-2.4), strategy="dissolve")
        elif case in ("leaf_tag_reaches_its_group", "no_detach_into_group"):
            with Off():
                leaf = Circle(radius=0.5).move_to(LEFT * 2.4).spawn()
            leaf.pass_index = 7
            leaf.wait(1)
            leaf.become(
                Group(
                    circle(-2.4),
                    Square(size=0.8, add_to_scene=False).move_to(RIGHT * 2),
                    add_to_scene=False,
                ),
                detach_history=case == "leaf_tag_reaches_its_group",
            )
        else:
            with Off():
                text = Text("ab").scale(1.5).move_to(LEFT * 2)
                group = Group(text).spawn()
            text.pass_index = 6
            group.pass_index = 5
            group.wait(1)
            group.become(
                Group(
                    Text("abc", add_to_scene=False).scale(1.5).move_to(LEFT * 2),
                    circle(2),
                    add_to_scene=False,
                )
            )
        scene.wait(1)
        result = scene.save_video(tmp_path / "around.mp4", passes="object_id")
    halves = _ids_per_half(result, w, h)
    lefts = set().union(*(left for left, _ in halves))
    rights = set().union(*(right for _, right in halves))
    seen = lefts | rights
    if case in ("outer_tag", "outer_tag_dissolve"):
        assert seen == {4}, halves
    elif case == "outer_composite":
        assert seen == {AUTO_ID_BASE + outer.id}, halves
    elif case == "dissolve_surplus_part":
        # The right square fades toward the one circle on the left: another
        # object, it keeps its own tag rather than the circle's.
        assert halves[0] == ({1}, {2}), halves
        assert seen == {1, 2}, halves
    elif case == "nested_dissolve":
        assert 1 in lefts, halves
        assert len(lefts) == 2, "a keeps its tag, b one id of its own"
        assert len(rights) == 1, halves
        assert not lefts & rights, halves
    elif case == "lone_member_dissolve":
        assert len(seen) == 1, halves
    elif case in ("leaf_tag_reaches_its_group", "no_detach_into_group"):
        assert seen == {7}, halves
    else:
        assert (lefts, rights) == ({6}, {5}), halves


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
        name = result.output_path.name
        assert result.passes == {
            "depth": result.output_path.with_name(name + ".depth.exr"),
            "normal": result.output_path.with_name(name + ".normal.png"),
        }
        for path in result.passes.values():
            assert path.exists()
        w, h = VIDEO.resolution
        normal = _decode(result.passes["normal"], "rgb48le", w, h, "<u2", 3)[0]
        # A camera-facing square at the centre: n = (0, 0, 1).
        assert abs(int(normal[h // 2, w // 2, 2]) - 65535) <= 1
        assert abs(int(normal[h // 2, w // 2, 0]) - 32768) <= 64
        assert normal[0, 0].tolist() == [0, 0, 0]
        meta = json.loads(
            result.output_path.with_name(
                result.output_path.name + ".passes.json"
            ).read_text()
        )
        assert meta["timestamp"] == 0.0
        assert set(meta["passes"]) == {"depth", "normal"}
