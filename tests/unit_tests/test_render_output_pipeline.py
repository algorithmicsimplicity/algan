"""The post-processing chain and the file the renderer actually writes.

``save_video(post_processes=...)`` is a public extension point: a user pass is
handed every frame and the arena it must allocate from. Nothing else in the
suite runs a user pass, and nothing describes its memory -- the render loop
measures the arena's high-water mark instead -- so the only way to know the
plumbing still works is to run one.

These render a 32x32 scene, which is cheap once the kernels are cached, but
they are GPU work and they cover one extension point rather than the engine's
core, so they sit outside the fast suite.
"""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS, SMOKE_TEST, Off, Scene, Square
from algan.constants.color import BLUE


@pytest.fixture
def render_scene(tmp_path):
    SETTINGS.paths.set(output_root=str(tmp_path), output_directory=".")
    SETTINGS.video.set(SMOKE_TEST)

    def build():
        scene = Scene()
        with scene:
            with Off():
                # Sized against the default camera's frame (8 world units
                # tall), not against a number that once looked big: the square
                # has to be most of a 32x32 frame for the channel assertions
                # below to be about the post-process rather than about how
                # libx264's chroma subsampling smears a handful of pixels.
                Square(size=5.0, color=BLUE).spawn()
            scene.wait(0.5)
        return scene

    return build


@pytest.fixture
def glowing_scene(tmp_path):
    """A scene whose mob actually glows, so bloom has something to do.

    ``bloom_filter`` short-circuits when no pixel carries glow, which would make
    "default chain" and "empty chain" indistinguishable.
    """
    SETTINGS.paths.set(output_root=str(tmp_path), output_directory=".")
    SETTINGS.video.set(SMOKE_TEST)

    def build():
        scene = Scene()
        with scene:
            with Off():
                Square(size=1.0, color=BLUE, glow=0.6).spawn()
            scene.wait(0.5)
        return scene

    return build


def _read_still(path):
    import cv2

    image = cv2.imread(str(path))
    assert image is not None, f"no image was written to {path}"
    return torch.from_numpy(image.astype("int16"))


def _read_frames(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_a_user_post_process_sees_every_frame_and_reaches_the_file(
    render_scene, tmp_path
):
    seen = []

    def paint_red(frames, memory):
        seen.append(tuple(frames.shape))
        out = memory.get_tensor(frames.shape, frames.dtype)
        out.copy_(frames)
        # Channels are (r, g, b, glow, [alpha]); drive green and blue to zero
        # so the written video is unmistakably the pass's own output.
        out[..., 1:3] = 0
        return out

    scene = render_scene()
    result = scene.save_video(
        tmp_path / "post_processed",
        video_settings=SMOKE_TEST,
        overwrite=True,
        post_processes=(paint_red,),
    )

    assert result.rendered
    assert seen, "the post-process was never called"
    frames = _read_frames(result.output_path)
    assert frames, "no frames were written"

    # OpenCV reads BGR. The pass zeroed green and blue, so red has to dominate
    # by far -- exactly is too strong a claim against a lossy codec.
    import numpy

    stacked = torch.from_numpy(numpy.stack(frames).astype("int16"))
    red, green, blue = (int(stacked[..., i].max()) for i in (2, 1, 0))
    assert red > 100, "the blue square's red channel never survived the pass"
    assert green < red // 3
    assert blue < red // 3

    # And the default chain, on the same scene, produces something different.
    control = render_scene().save_video(
        tmp_path / "control", video_settings=SMOKE_TEST, overwrite=True
    )
    control_frames = torch.from_numpy(
        numpy.stack(_read_frames(control.output_path)).astype("int16")
    )
    assert int((control_frames - stacked).abs().max()) > 20


def test_an_empty_post_process_chain_renders_without_bloom(render_scene, tmp_path):
    scene = render_scene()
    result = scene.save_video(
        tmp_path / "no_post",
        video_settings=SMOKE_TEST,
        overwrite=True,
        post_processes=(),
    )
    assert result.rendered
    assert _read_frames(result.output_path)


def test_the_default_chain_and_an_explicit_empty_one_differ_only_by_bloom(
    render_scene, tmp_path
):
    """A regression that dropped the default chain would otherwise be silent."""
    with_bloom = render_scene().save_video(
        tmp_path / "with_bloom", video_settings=SMOKE_TEST, overwrite=True
    )
    without = render_scene().save_video(
        tmp_path / "without_bloom",
        video_settings=SMOKE_TEST,
        overwrite=True,
        post_processes=(),
    )
    assert with_bloom.rendered
    assert without.rendered
    assert len(_read_frames(with_bloom.output_path)) == len(
        _read_frames(without.output_path)
    )


def test_save_frame_runs_a_user_post_process_and_writes_its_output(
    render_scene, tmp_path
):
    """``save_frame`` takes the same extension point as ``save_video``."""
    seen = []

    def paint_red(frames, memory):
        seen.append(tuple(frames.shape))
        out = memory.get_tensor(frames.shape, frames.dtype)
        out.copy_(frames)
        out[..., 1:3] = 0
        return out

    result = render_scene().save_frame(
        tmp_path / "still_post.png",
        video_settings=SMOKE_TEST,
        post_processes=(paint_red,),
    )

    assert result.rendered
    assert seen, "the post-process was never called"
    # OpenCV reads BGR; the pass zeroed green and blue.
    still = _read_still(result.output_path)
    red, green, blue = (int(still[..., i].max()) for i in (2, 1, 0))
    assert red > 100, "the blue square's red channel never survived the pass"
    assert green < red // 3
    assert blue < red // 3


def test_save_frame_defaults_to_bloom_and_honours_an_empty_chain(
    glowing_scene, tmp_path
):
    """Omitting the argument must still bloom; ``()`` must genuinely skip it."""
    default = glowing_scene().save_frame(
        tmp_path / "still_default.png", video_settings=SMOKE_TEST
    )
    without = glowing_scene().save_frame(
        tmp_path / "still_no_bloom.png", video_settings=SMOKE_TEST, post_processes=()
    )
    assert default.rendered
    assert without.rendered

    bloomed, plain = _read_still(default.output_path), _read_still(without.output_path)
    assert bloomed.shape == plain.shape
    # Bloom spreads the glowing square's light outwards, so the frame gets
    # strictly brighter overall.
    assert int(bloomed.sum()) > int(plain.sum())


def test_save_frame_applies_the_post_process_to_every_still_in_a_sequence(
    render_scene, tmp_path
):
    calls = []

    def counting_pass(frames, memory):
        calls.append(tuple(frames.shape))
        return frames

    results = render_scene().save_frame(
        tmp_path / "sheet.png",
        video_settings=SMOKE_TEST,
        at=[0, 0.25],
        post_processes=(counting_pass,),
    )

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(result.rendered for result in results)
    assert len(calls) == 2, f"expected one call per still, got {len(calls)}"


def test_fxaa_is_a_video_setting_the_render_honours(render_scene, tmp_path):
    scene = render_scene()
    result = scene.save_video(
        tmp_path / "fxaa",
        video_settings=SMOKE_TEST.set(fxaa=True),
        overwrite=True,
    )
    assert result.rendered
    assert _read_frames(result.output_path)


def test_save_frame_writes_a_png_at_the_requested_resolution(render_scene, tmp_path):
    from PIL import Image

    scene = render_scene()
    result = scene.save_frame(tmp_path / "still", SMOKE_TEST, at=0.0, overwrite=True)
    assert result.rendered
    with Image.open(result.output_path) as still:
        assert still.size == SMOKE_TEST.resolution
