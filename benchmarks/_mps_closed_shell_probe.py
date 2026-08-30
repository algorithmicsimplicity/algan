"""Where the closed-shell frame's two routes disagree, pixel by pixel.

``test_closed_shell_attenuates_once_at_authored_opacity`` renders one
translucent emissive cube twice -- once path-traced (8 spp), once through the
deterministic sheet route (1 spp) -- and asserts the two agree to within 2
channel values over the centre 12x12 of a 64x64 frame. On an Apple GPU they
agree everywhere in that window **except its last column**, by 86
(``DESIGN_mps_support.md`` §1.2c); on the CPU they agree, in MPS-friendly mode
and out of it.

The assertion says how far apart they are and nothing about where or which one
moved, and those are the two facts that separate the readings:

* a **whole column of the image** that differs, running past the window, is a
  geometry edge the window happens to clip -- the two routes are allowed to
  disagree at an edge, and the finding would be that this window is only just
  clear of one rather than that Metal is wrong;
* a **12-pixel run that stops at the window's edge** is inside the flat
  interior, and then something really is wrong on this backend.

So this prints both frames over a window WIDER than the assertion's, plus every
column of the frame that disagrees within the window's own rows, and says which
route sits at the authored ``0.6 * 255`` and which one moved.

    uv run python benchmarks/_mps_closed_shell_probe.py

Renders on whatever ``ALGAN_RENDER_DEVICE`` selects, so the reading is taken by
running it once per device and comparing the two reports. It exits 0 whatever
it finds: it is a measurement, not a check -- the test is the check.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from algan import (  # noqa: E402
    BLACK,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    Off,
    Prism,
    Scene,
    SceneManager,
)

#: The test's own settings and scene, kept identical on purpose: a probe that
#: renders something slightly different is measuring something slightly
#: different.
SHELL_SETTINGS = SMOKE_TEST.set(resolution=(64, 64))
OPACITY = 0.6
_RAW_KW = {"linear_color_space": False, "tonemapping": False}
_RAW_EXP = {"post_process_tonemap": False}


def _emissive_shell_cube():
    """The test's cube: black albedo, white emission, one closed shell."""
    cube = Prism(width=2.0, height=2.0, depth=2.0)
    cube.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.0)
    )
    cube.set_opacity(OPACITY)
    cube.rotate(17, UP).rotate(9, RIGHT)
    return cube


def _render(out_dir, name, samples_per_pixel):
    """One frame, as an int32 HxWxC tensor on the host."""
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=samples_per_pixel, denoise=False)
        for key, value in _RAW_KW.items():
            SETTINGS.raytracing.set(**{key: value})
        for key, value in _RAW_EXP.items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        with Scene(video_settings=SHELL_SETTINGS) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                _emissive_shell_cube().spawn(animate=False)
            result = scene.save_frame(
                out_dir / name, video_settings=SHELL_SETTINGS, overwrite=True
            )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"unreadable frame at {result.output_path}")
    return torch.from_numpy(frame.astype(np.int32))


def _grid(values, label, first_col):
    """A numbered grid of one channel, so a column can be pointed at."""
    lines = [f"{label} (columns {first_col}..{first_col + values.shape[1] - 1}):"]
    lines.append(
        "      " + "".join(f"{first_col + i:5d}" for i in range(values.shape[1]))
    )
    for r in range(values.shape[0]):
        lines.append("      " + "".join(f"{int(v):5d}" for v in values[r]))
    return "\n".join(lines)


def main():
    from algan.rendering.mps_compat import mps_friendly

    print(f"render device    : {SETTINGS.computing.render_device}")
    print(f"mps friendly     : {mps_friendly()}")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        pt = _render(out_dir, "shell_pt.png", 8)
        det = _render(out_dir, "shell_det.png", 1)

    h, w = pt.shape[0], pt.shape[1]
    expected = OPACITY * 255.0
    err = (pt[..., :3] - det[..., :3]).abs().amax(-1).float()

    lo, hi = h // 2 - 6, h // 2 + 6
    clo, chi = w // 2 - 6, w // 2 + 6
    print(f"\nframe            : {tuple(pt.shape)}, authored interior {expected:.0f}")
    print(
        f"assertion window : rows {lo}..{hi - 1} cols {clo}..{chi - 1}, "
        f"max |pt-det| = {float(err[lo:hi, clo:chi].max()):.0f}"
    )

    # Whether the run of disagreement stops at the window's edge or runs past
    # it, which is the reading this probe exists to take. Across the window's
    # ROWS but the frame's whole WIDTH: the silhouette crosses other rows in
    # every render, so a full-frame column count says nothing, while these
    # twelve rows are the interior the test is asserting about.
    band = err[lo:hi]
    hot = [int(c) for c in torch.nonzero(band.amax(0) > 2).flatten().tolist()]
    print(f"columns differing by > 2 across the window's rows: {hot}")

    # Four columns either side of the window, so an edge shows as a run.
    wlo, whi = max(0, clo - 4), min(w, chi + 4)
    green = 1  # BGR from cv2; the channel is arbitrary, the frame is grey.
    print()
    print(_grid(pt[lo:hi, wlo:whi, green], "path traced (8 spp)", wlo))
    print()
    print(_grid(det[lo:hi, wlo:whi, green], "deterministic (1 spp)", wlo))
    print()
    print(_grid(err[lo:hi, wlo:whi], "|pt - det|", wlo))

    # Which route moved. The test's own oracle is the authored opacity, so the
    # frame sitting at 0.6*255 is the right one and the other is the finding.
    core_pt = pt[lo:hi, clo:chi, :3].float()
    core_det = det[lo:hi, clo:chi, :3].float()
    print(
        f"\npath traced   interior: mean {float(core_pt.mean()):7.2f} "
        f"min {float(core_pt.min()):5.0f} max {float(core_pt.max()):5.0f}"
    )
    print(
        f"deterministic interior: mean {float(core_det.mean()):7.2f} "
        f"min {float(core_det.min()):5.0f} max {float(core_det.max()):5.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
