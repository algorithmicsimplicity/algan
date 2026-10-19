"""What ``save_frame`` writes.

A write that failed part way left a broken file where the last good one had
been.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from algan import Scene, SceneManager, Square
from algan.constants.color import BLUE, TRANSPARENT
from algan.settings.video_settings import VideoSettings

STILL = VideoSettings((16, 12), 30, supersampling=1)


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _spy_on_frames(monkeypatch):
    """Replace the renderer with flat frames; return the calls it was given."""
    calls = []

    def frames(scene, start, end, **options):
        if "frame_indices" in options:
            indices = list(options["frame_indices"])[start:end]
        else:
            indices = list(range(start, end))
        settings = scene.video_settings
        calls.append((indices, settings.frames_per_second, settings.resolution))
        width, height = settings.resolution
        channels = 4 if scene.background_is_transparent() else 3
        yield torch.full((len(indices), height, width, channels), 9, dtype=torch.uint8)

    monkeypatch.setattr(Scene, "get_frames", frames)
    return calls


@pytest.mark.parametrize("suffix", [".png", ".webp", ".tiff"])
def test_a_transparent_still_keeps_its_alpha(tmp_path, suffix):
    with Scene(video_settings=STILL) as scene:
        Square(color=BLUE, size=4).spawn(animate=False)
        result = scene.save_frame(
            tmp_path / f"shot{suffix}", background=TRANSPARENT, post_processes=()
        )
    with Image.open(result.output_path) as image:
        assert image.mode == "RGBA"
        alpha = image.getchannel("A")
        assert alpha.getpixel((0, 0)) == 0
        assert alpha.getpixel((8, 6)) == 255


def test_a_failed_write_leaves_the_file_that_was_there(monkeypatch, tmp_path):
    _spy_on_frames(monkeypatch)
    existing = tmp_path / "shot.png"
    existing.write_bytes(b"an earlier render")

    def half_written(image, path, *args, **kwargs):
        with open(path, "wb") as out:
            out.write(b"\x89PNG half")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", half_written)
    with Scene(video_settings=STILL) as scene, pytest.raises(OSError, match="full"):
        scene.save_frame(existing)
    assert existing.read_bytes() == b"an earlier render"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shot.png"]
