"""Image backgrounds: how a file becomes the backdrop of every render.

An image background is kept at its own resolution and fitted to the frame of
each render, whatever that frame's size: the Scene's own settings, a preset
passed to one ``save_frame`` or ``save_video``, the viewer's resolution, or a
camera view's capture.

A feature test, so it sits outside the fast suite: it only breaks when image
backgrounds themselves change.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from algan import PREVIEW, CameraView, Color, Square
from algan.errors import AlganConfigurationError
from algan.render_loop import _prepare_background_for_chunk
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _prefill_background,
)
from algan.scene import Scene
from algan.scene_manager import SceneManager
from algan.settings.video_settings import SMOKE_TEST

STILL = SMOKE_TEST.set(resolution=(64, 36))
#: A color per quarter of the test image, as the file holds it (top-down).
QUADRANTS = {
    "top left": (200, 0, 0),
    "top right": (0, 200, 0),
    "bottom left": (0, 0, 200),
    "bottom right": (200, 200, 200),
}


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _write_test_image(tmp_path, height, width):
    """A PNG with no symmetry in either axis, so a transpose or a flip of the
    decoded background is detectable.
    """
    from PIL import Image

    rows = torch.arange(height, dtype=torch.float32).view(-1, 1) * 7
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    red = (rows + cols).clamp_max(255)
    image = torch.stack((red, torch.zeros_like(red), 255 - red), dim=0).to(torch.uint8)
    path = tmp_path / "background.png"
    Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB").save(path)
    return path, image.permute(1, 2, 0).float() / 255


def test_image_background_keeps_the_source_orientation(tmp_path):
    height, width = SMOKE_TEST.resolution[1], SMOKE_TEST.resolution[0]
    path, source = _write_test_image(tmp_path, height, width)

    scene = Scene(video_settings=SMOKE_TEST)
    scene.set_background(str(path))
    background = scene.background_frame

    # [1, height, width, channels] -- neither axis transposed.
    assert background.shape[:3] == (1, height, width)
    # Frame buffers are bottom-up (post_process_frames flips them on the way
    # out), so the stored rows run bottom-to-top: undoing that flip must
    # recover the image exactly as it sits in the file.
    recovered = background[0].flip(0)[..., :3].cpu()
    assert torch.allclose(recovered, source, atol=1.5 / 255)


def test_image_background_is_scaled_to_the_supersampled_frame(tmp_path):
    settings = SMOKE_TEST.set(supersampling=2)
    height, width = settings.resolution[1], settings.resolution[0]
    path, _ = _write_test_image(tmp_path, height // 2, width // 2)

    scene = Scene(video_settings=settings)
    scene.set_background(str(path))
    fitted = scene._background_fitted_to_frame(scene.background_frame)

    assert fitted.shape[:3] == (1, height * 2, width * 2)


def test_prefill_rejects_a_background_at_the_wrong_resolution():
    """A super-sampled background read at output stride used to scroll a
    different slice of itself into every frame (flickering image backgrounds
    on the analytic raster route, which renders at aa == 1).
    """
    aa, screen_height, screen_width, frames = 2, 2, 3, 3
    image = torch.rand(1, screen_height * aa, screen_width * aa, 4)
    background = _prepare_background_for_chunk(
        image,
        screen_width=screen_width,
        screen_height=screen_height,
        anti_alias_level=aa,
        current_ind=0,
        new_ind=frames,
        frames_per_second=1,
        device=torch.device("cpu"),
    )

    out = torch.empty((frames, screen_height * screen_width, 4), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="background resolution"):
        _prefill_background(out, background, 0, out.device, background_frames=frames)

    averaged = _downsample_background(
        background, aa, frames, screen_height, screen_width
    )
    _prefill_background(out, averaged, 0, out.device, background_frames=frames)

    # Every frame gets the same still image back, not a different slice.
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[0], out[2])


@pytest.fixture
def quadrants(tmp_path):
    """A 32 x 18 image of four flat quarters (``QUADRANTS``)."""
    image = np.zeros((18, 32, 3), np.uint8)
    image[:9, :16] = QUADRANTS["top left"]
    image[:9, 16:] = QUADRANTS["top right"]
    image[9:, :16] = QUADRANTS["bottom left"]
    image[9:, 16:] = QUADRANTS["bottom right"]
    path = tmp_path / "quadrants.png"
    Image.fromarray(image).save(path)
    return path


def _read(png):
    return np.asarray(Image.open(png).convert("RGB")).astype(np.int16)


def _assert_shows_the_quadrants(frame, tolerance=2):
    """``frame`` ([H, W, 3], top-down) shows each quarter in its place."""
    height, width = frame.shape[:2]
    middles = {
        "top left": frame[height // 4, width // 4],
        "top right": frame[height // 4, 3 * width // 4],
        "bottom left": frame[3 * height // 4, width // 4],
        "bottom right": frame[3 * height // 4, 3 * width // 4],
    }
    for name, color in middles.items():
        assert np.abs(color - QUADRANTS[name]).max() <= tolerance, (name, color)


def _small_square():
    """Something in the middle of the frame, clear of the quarters' middles."""
    return Square().scale(0.2).spawn()


def test_a_still_at_a_preset_of_its_own_fits_the_image(tmp_path, quadrants):
    """REGRESSION. The image was scaled to the frame once, when it was set, so
    a render at any other size -- a preset passed to ``save_frame`` -- failed
    with "background resolution does not match render output".
    """
    with Scene(video_settings=STILL) as scene:
        scene.set_background(str(quadrants))
        _small_square()
        renders = {
            (18, 32): scene.save_frame(
                tmp_path / "smaller.png", STILL.set(resolution=(32, 18))
            ),
            (54, 96): scene.save_frame(
                tmp_path / "larger.png",
                STILL.set(resolution=(96, 54), supersampling=2),
            ),
            (36, 64): scene.save_frame(tmp_path / "own.png"),
        }
        assert scene.background_frame.shape[1:3] == (18, 32)

    for size, result in renders.items():
        frame = _read(result.output_path)
        assert frame.shape[:2] == size
        _assert_shows_the_quadrants(frame)


def test_settings_changed_after_the_background_still_fit(tmp_path, quadrants):
    """REGRESSION. Setting the Scene's video settings after its background
    left the image at the old frame size, and the render failed.
    """
    with Scene(video_settings=STILL) as scene:
        scene.set_background(str(quadrants))
        scene.set_video_settings(STILL.set(resolution=(48, 27)))
        # Nothing spawned: the frames that draw only the background fit too.
        frame = _read(scene.save_frame(tmp_path / "later.png").output_path)

    assert frame.shape[:2] == (27, 48)
    _assert_shows_the_quadrants(frame)


def test_a_video_at_a_preset_of_its_own_fits_the_image(tmp_path, quadrants):
    """REGRESSION. As for a still: ``save_video`` with a preset failed."""
    cv2 = pytest.importorskip("cv2")
    with Scene(video_settings=STILL) as scene:
        scene.set_background(str(quadrants))
        _small_square()
        result = scene.save_video(tmp_path / "clip.mp4", STILL.set(resolution=(48, 28)))

    capture = cv2.VideoCapture(str(result.output_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert frame.shape[:2] == (28, 48)
    # Video is lossy and stores color at half resolution; the quarters' middles
    # are far enough from their edges to keep their own colors regardless.
    _assert_shows_the_quadrants(frame[..., ::-1].astype(np.int16), tolerance=24)


def test_the_viewer_fits_the_image_to_its_own_resolution(quadrants, monkeypatch):
    """REGRESSION. The viewer renders at the resolution its picker shows, not
    the Scene's, so every frame it drew over an image background failed.
    """
    from algan.viewer import session as viewer

    monkeypatch.setattr(viewer.ViewerSession, "_run", lambda self: None)
    with Scene(video_settings=STILL) as scene:
        scene.set_background(str(quadrants))
        _small_square()
        session = viewer.ViewerSession(scene, PREVIEW.set(resolution=(32, 18)))
        try:
            session._render_range(0, 1, store=True)
            png = session._cache[0]
        finally:
            session.close()

    frame = _read(io.BytesIO(png))
    assert frame.shape[:2] == (18, 32)
    _assert_shows_the_quadrants(frame)


def test_a_camera_view_captures_over_an_image_background(tmp_path, quadrants):
    """REGRESSION. A camera view captures at its own resolution, with the
    Scene's background behind it, and that capture failed.
    """
    with Scene(video_settings=STILL) as scene:
        scene.set_background(str(quadrants))
        CameraView(scene=scene, resolution=(30, 20)).spawn()
        result = scene.save_frame(tmp_path / "with_view.png")

    # The view sits in the middle of the frame, clear of the quarters' middles.
    frame = _read(result.output_path)
    assert frame.shape[:2] == (36, 64)
    _assert_shows_the_quadrants(frame)


def test_a_larger_render_resamples_the_file_not_a_smaller_copy(tmp_path):
    """Fitted when it is drawn, the image is resampled from the file itself:
    a Scene set up at a small size and rendered at the image's own size shows
    the image exactly, not one shrunk to the Scene's size and enlarged again.
    """
    checker = (np.indices((36, 64)).sum(0) % 2 * 255).astype(np.uint8)
    path = tmp_path / "checker.png"
    Image.fromarray(np.repeat(checker[..., None], 3, -1)).save(path)

    with Scene(video_settings=STILL.set(resolution=(16, 9))) as scene:
        scene.set_background(str(path))
        _small_square()
        frame = _read(scene.save_frame(tmp_path / "full.png", STILL).output_path)

    away_from_the_square = np.r_[0:12, 24:36]
    assert np.array_equal(
        frame[away_from_the_square, :, 0], checker[away_from_the_square]
    )


def test_a_relative_path_is_found_beside_the_running_script(tmp_path, monkeypatch):
    """REGRESSION. Every other asset path is looked for beside the script when
    the working directory does not have it. The background checked the path
    as given, so ``set_background("backdrop.png")`` said the file did not
    exist unless Python was started in the script's directory.
    """
    from algan.settings import path_settings

    project = tmp_path / "project"
    project.mkdir()
    script = project / "scene.py"
    script.write_text("")
    image = np.full((9, 16, 3), (40, 90, 200), np.uint8)
    Image.fromarray(image).save(project / "backdrop.png")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: str(script))

    with Scene(video_settings=STILL) as scene:
        scene.set_background("backdrop.png")
        torch.testing.assert_close(
            scene.background_frame[0, ..., :3],
            torch.from_numpy(image).float() / 255,
        )


def test_a_pathlib_path_is_an_image_path(tmp_path, quadrants):
    """REGRESSION. Only a ``str`` was read as a file: a :class:`~pathlib.Path`
    was kept as the background itself, and the render failed with "'PosixPath'
    object is not subscriptable".
    """
    with Scene(video_settings=STILL) as scene:
        scene.set_background(Path(quadrants))
        _small_square()
        frame = _read(scene.save_frame(tmp_path / "from_path.png").output_path)
        with pytest.raises(AlganConfigurationError, match="missing.png"):
            scene.set_background(tmp_path / "missing.png")

    _assert_shows_the_quadrants(frame)


#: One color, in each spelling a Mob's color takes.
SLATE = (0.1, 0.2, 0.3)
SLATE_SPELLINGS = {
    "tuple": SLATE,
    "list": list(SLATE),
    "hex int": 0x1A334D,
    "hex string": "#1A334D",
    "rgb tensor": torch.tensor(SLATE),
    "Color": Color(SLATE),
}


@pytest.mark.parametrize("spelling", SLATE_SPELLINGS)
def test_every_color_spelling_is_an_opaque_background(tmp_path, spelling):
    """REGRESSION. Only a string went through the color parser. A tuple, a
    list or a hex int failed the render with "tuple indices must be integers"
    (or "'int' object is not subscriptable"), and an RGB tensor's blue was read
    as its alpha: the output came out transparent, an RGBA PNG and a ``.mov``.
    """
    with Scene(video_settings=STILL) as scene:
        scene.set_background(SLATE_SPELLINGS[spelling])
        assert not scene.background_is_transparent()
        _small_square()
        result = scene.save_frame(tmp_path / "slate.png")

    with Image.open(result.output_path) as image:
        assert image.mode == "RGB"
        corner = image.getpixel((2, 2))
    assert np.abs(np.subtract(corner, (26, 51, 77))).max() <= 1


def test_an_rgb_tensor_background_makes_an_mp4(tmp_path):
    """REGRESSION. See above: the RGB tensor's video was a transparent ``.mov``."""
    with Scene(video_settings=STILL) as scene:
        scene.set_background(torch.tensor(SLATE))
        _small_square()
        result = scene.save_video(tmp_path / "slate")

    assert result.output_path.suffix == ".mp4"


def test_the_constructors_grid_callback_is_an_image_a_video_can_use(tmp_path):
    """REGRESSION. ``Scene(background=callable)`` calls the callable once with a
    grid of pixel indices, row 0 at the bottom, and keeps the image it returns.
    Stored as returned, ``[height, width, channels]``, a video read its rows as
    frames and failed on every batch of more than one frame; and its blue,
    being last, was read as alpha.
    """
    cv2 = pytest.importorskip("cv2")

    def bottom_blue_top_red(grid):
        rows = grid[..., :1].float() / (grid.shape[0] - 1)
        return torch.cat((rows, torch.zeros_like(rows), 1 - rows), -1)

    clip = STILL.set(frames_per_second=10)
    with Scene(clip, background=bottom_blue_top_red) as scene:
        assert not scene.background_is_transparent()
        square = _small_square()
        square.move((0.5, 0, 0))
        result = scene.save_video(tmp_path / "gradient")

    assert result.output_path.suffix == ".mp4"
    capture = cv2.VideoCapture(str(result.output_path))
    try:
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame[..., ::-1].astype(np.int16))
    finally:
        capture.release()
    assert len(frames) > 1
    # Red at the top and blue at the bottom of every frame: the image the
    # callback drew, the right way up, in each batch.
    for frame in frames:
        top, bottom = frame[1, 2], frame[-2, 2]
        assert top[0] > 200, top
        assert top[2] < 40, top
        assert bottom[2] > 200, bottom
        assert bottom[0] < 40, bottom
