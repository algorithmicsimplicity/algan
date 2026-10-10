"""Text carried to just in front of a distant camera renders as it did far away.

A HUD that rides a camera is shrunk and slid along the camera's rays until it
sits a unit in front of the eye, which leaves it looking exactly as it did:
the backpropagation video does this with a label panel beside a near-
orthographic view (camera distance 80, so the panel is scaled by about 1/107).
Its glyphs grew one-pixel dashed streaks there. Their outlines are cubics a few
ten-thousandths of a unit long, below the fixed world-unit size the flattening
treated as "degenerate", so those cubics were dropped; each gap left the even-
odd crossing parity open, and every pixel centre lined up with one flipped
inside/outside to the end of the glyph's candidate box.

The label is rendered twice in one frame: once where it is authored, and once
brought forward in front of the eye. The two must match. A quarter of a unit
rather than the video's one unit, because at this test's 180 rows the video's
gaps (up to a third of a pixel at 720) rarely meet a pixel centre; the defect
is the same one, and before the fix it drew 70 pixels here wrong by more than
64 levels.
"""

import shutil

import numpy as np
import pytest

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import WHITE
from algan.constants.spatial import DOWN, UP
from algan.mobs.text import Tex
from algan.scene import Scene
from algan.settings.video_settings import LD

pytestmark = pytest.mark.skipif(
    not shutil.which("latex") or not shutil.which("dvisvgm"),
    reason="Tex needs latex and dvisvgm",
)

# The camera's frame is 8 units tall at the origin, so at 180 rows each label's
# row band is exactly 4 units = 90 rows from the other's.
HEIGHT = 180
VIDEO = LD.set(resolution=(320, HEIGHT))
LABEL = r"E_{\rm new} - E_{\rm old}"
DEPTH = 0.25


def _bring_forward(camera, mob, depth):
    """Scale ``mob`` about the eye to ``depth`` in front of it, unchanged on screen.

    This is the backpropagation video's ``bring_forward``.
    """
    eye = camera.location.reshape(3)
    forward = camera.forward.reshape(3)
    current = float(((mob.get_center().reshape(3) - eye) * forward).sum())
    k = depth / current
    mob.scale(k)
    mob.move_to(eye + (mob.location.reshape(3) - eye) * k)


def test_text_brought_forward_to_a_distant_camera_renders_unchanged(tmp_path):
    path = tmp_path / "hud.png"
    with Scene() as scene:
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic(distance=80)
            far = Tex(LABEL, color=WHITE).scale(3).move_to(UP * 2)
            near = Tex(LABEL, color=WHITE).scale(3).move_to(DOWN * 2)
            _bring_forward(camera, near, DEPTH)
        eye = camera.location.reshape(3)
        # The premise: one label at the origin plane, one just in front of the
        # eye, a hundred-odd units out from the origin.
        assert float((eye - far.get_center().reshape(3)).norm()) > 100
        assert abs(float((eye - near.get_center().reshape(3))[2]) - DEPTH) < 1e-3
        far.spawn(animate=False)
        near.spawn(animate=False)
        scene.save_frame(str(path), video_settings=VIDEO)

    from PIL import Image

    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB")).astype(np.int16)
    half = HEIGHT // 2
    far_band, near_band = frame[:half], frame[half:]
    assert (far_band.max(-1) > 128).sum() > 200, "the label did not render"
    difference = np.abs(far_band - near_band).max(-1)
    # The two depths may round an antialiased edge pixel differently: by a
    # level or two on the CPU, and by 17 at one edge pixel on MPS. A parity
    # streak is a column of pixels wrong by up to 255 -- before the fix, 70
    # here past 64 -- so it fails both checks.
    assert int((difference > 64).sum()) == 0, (
        f"{int((difference > 64).sum())} pixels differ by more than 64 "
        f"(max {int(difference.max())})"
    )
    assert int((difference > 16).sum()) <= 2, (
        f"{int((difference > 16).sum())} pixels differ by more than 16 "
        f"(max {int(difference.max())})"
    )
