import pytest
import torch

from algan.render_loop import _prepare_background_for_chunk
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _prefill_background,
)
from algan.scene import Scene
from algan.settings.video_settings import SMOKE_TEST


def _write_test_image(tmp_path, height, width):
    """A PNG with no symmetry in either axis, so a transpose or a flip of the
    decoded background is detectable.
    """
    import torchvision

    rows = torch.arange(height, dtype=torch.float32).view(-1, 1) * 7
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    red = (rows + cols).clamp_max(255)
    image = torch.stack((red, torch.zeros_like(red), 255 - red), dim=0).to(torch.uint8)
    path = tmp_path / "background.png"
    torchvision.io.write_png(image, str(path))
    return path, image.permute(1, 2, 0).float() / 255


def test_image_background_keeps_the_source_orientation(tmp_path):
    height, width = SMOKE_TEST.resolution[1], SMOKE_TEST.resolution[0]
    path, source = _write_test_image(tmp_path, height, width)

    scene = Scene(video_settings=SMOKE_TEST)
    scene.set_background_color(str(path))
    background = scene.background_frame

    # [1, height, width, channels] -- neither axis transposed.
    assert background.shape[:3] == (1, height, width)
    # Frame buffers are bottom-up (post_process_frames flips them on the way
    # out), so the stored rows run bottom-to-top: undoing that flip must
    # recover the image exactly as it sits in the file.
    recovered = background[0].flip(0)[..., :3].cpu()
    assert torch.allclose(recovered, source, atol=1.5 / 255)


def test_image_background_is_scaled_to_the_supersampled_frame(tmp_path):
    settings = SMOKE_TEST.set(anti_alias_level=2)
    height, width = settings.resolution[1], settings.resolution[0]
    path, _ = _write_test_image(tmp_path, height // 2, width // 2)

    scene = Scene(video_settings=settings)
    scene.set_background_color(str(path))

    assert scene.background_frame.shape[:3] == (1, height * 2, width * 2)


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
