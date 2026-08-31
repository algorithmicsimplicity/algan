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

So this prints the frames over a window WIDER than the assertion's, plus every
column of the frame that disagrees within the window's own rows, and says which
route sits at the authored ``0.6 * 255`` and which one moved.

It renders the deterministic route through **each arm** of the
solid-shell opacity ceiling: the fused Taichi kernel
(``sheet_compact_taichi.solid_shell_ceiling``) and the torch block it replaced.
The two are meant to be bit-identical and they share every input, so they split
the remaining question in half -- an arm that agrees with the path tracer while
the other does not puts the defect in that kernel; two arms that agree with each
other put it in what the ceiling was handed (the segment key, the facing bit,
the exclusive prefix).

And it renders one of those arms **twice**, which is the question that comes
first: MPS-friendly mode is documented non-deterministic, and §1.2's amendment
predicts this symptom in as many words -- "a ceiling that wobbles in its low
bits flipping borderline fragments in and out of being clipped". Two renders of
one configuration that disagree are that prediction; two that agree bit for bit
mean there is a fixed wrong answer to find.

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


def _render(out_dir, name, samples_per_pixel, shell_ceiling_kernel=None):
    """One frame, as an int32 HxWxC tensor, plus its truncation counters.

    ``shell_ceiling_kernel`` selects which arm applies the solid-shell opacity
    ceiling: the fused Taichi kernel (the default) or the torch block it
    replaced. They are meant to be bit-identical, and A/B-ing them is what
    separates "the ceiling's INPUTS are wrong on this device" from "the kernel
    is" -- the two arms share every input and nothing else.
    """
    from algan.rendering.raytracing import settings as rt_settings

    snapshot = SETTINGS.snapshot()
    previous_kernel = rt_settings.sheet_shell_ceiling_kernel
    SceneManager.reset()
    try:
        if shell_ceiling_kernel is not None:
            rt_settings.set_sheet_shell_ceiling_kernel(shell_ceiling_kernel)
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
        rt_settings.set_sheet_shell_ceiling_kernel(previous_kernel)
    frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"unreadable frame at {result.output_path}")
    truncations = getattr(getattr(result, "render_plan", None), "truncations", None)
    return torch.from_numpy(frame.astype(np.int32)), truncations


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
        pt, pt_trunc = _render(out_dir, "shell_pt.png", 8)
        det, det_trunc = _render(out_dir, "shell_det.png", 1, shell_ceiling_kernel=True)
        # The same arm again, same process, same settings. MPS-friendly mode is
        # documented non-deterministic -- f32 atomics replace the f64
        # accumulator, and DESIGN_mps_support.md 1.2's amendment predicts the
        # symptom as "a ceiling that wobbles in its low bits flipping
        # borderline fragments in and out of being clipped". If two renders of
        # ONE configuration disagree at the offending column, that prediction
        # is what this is and no amount of looking at the kernel will find a
        # bug; if they agree bit for bit, it is a real defect with a fixed
        # answer.
        det_again, _ = _render(
            out_dir, "shell_det_again.png", 1, shell_ceiling_kernel=True
        )
        # LAST, and allowed to fail. The torch arm of the ceiling calls
        # ``index_copy_``, which torch has not implemented for MPS
        # (`aten::index_copy.out`), so on an Apple GPU this arm raises and
        # takes every reading above with it if it runs first. It is the least
        # important of the three and it is the only one that can die.
        torch_det = torch_trunc = None
        try:
            torch_det, torch_trunc = _render(
                out_dir, "shell_det_torch.png", 1, shell_ceiling_kernel=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"det, ceiling torch    : unavailable -- {type(exc).__name__}: {exc}")

    h, w = pt.shape[0], pt.shape[1]
    expected = OPACITY * 255.0
    lo, hi = h // 2 - 6, h // 2 + 6
    clo, chi = w // 2 - 6, w // 2 + 6
    print(f"\nframe            : {tuple(pt.shape)}, authored interior {expected:.0f}")

    def summarize(label, frame):
        err = (pt[..., :3] - frame[..., :3]).abs().amax(-1).float()
        # Whether the run of disagreement stops at the window's edge or runs
        # past it, which is the reading this probe exists to take. Across the
        # window's ROWS but the frame's whole WIDTH: the silhouette crosses
        # other rows in every render, so a full-frame column count says
        # nothing, while these twelve rows are the interior the test asserts
        # about.
        hot = [int(c) for c in torch.nonzero(err[lo:hi].amax(0) > 2).flatten().tolist()]
        core = frame[lo:hi, clo:chi, :3].float()
        print(
            f"{label:22s}: max |pt - this| in window "
            f"{float(err[lo:hi, clo:chi].max()):3.0f}; columns > 2 across the "
            f"window's rows {hot}; interior mean {float(core.mean()):7.2f} "
            f"min {float(core.min()):5.0f} max {float(core.max()):5.0f}"
        )
        return err

    print(
        f"assertion window : rows {lo}..{hi - 1} cols {clo}..{chi - 1}, "
        f"authored {expected:.0f}"
    )
    core_pt = pt[lo:hi, clo:chi, :3].float()
    print(
        f"{'path traced (8 spp)':22s}: interior mean {float(core_pt.mean()):7.2f} "
        f"min {float(core_pt.min()):5.0f} max {float(core_pt.max()):5.0f} "
        "-- the oracle"
    )
    err = summarize("det, ceiling kernel", det)
    if torch_det is not None:
        summarize("det, ceiling torch", torch_det)

    # Reproducibility, which decides whether there is a defect to find at all.
    repeat = (det[..., :3] - det_again[..., :3]).abs().amax(-1).float()
    repeat_hot = [
        int(c) for c in torch.nonzero(repeat[lo:hi].amax(0) > 0).flatten().tolist()
    ]
    print(
        f"{'det vs det again':22s}: max |diff| whole frame "
        f"{float(repeat.max()):3.0f}; columns differing at all across the "
        f"window's rows {repeat_hot}"
    )

    for label, trunc in (
        ("path traced", pt_trunc),
        ("det, kernel", det_trunc),
        ("det, torch", torch_trunc),
    ):
        if trunc is None:
            continue
        hits = {
            name: getattr(trunc, name)
            for name in dir(trunc)
            if not name.startswith("_") and isinstance(getattr(trunc, name), int)
        }
        hits = {k: v for k, v in hits.items() if v}
        print(f"truncations {label:12s}: {hits or 'none'}")

    # Four columns either side of the window, so an edge shows as a run.
    wlo, whi = max(0, clo - 4), min(w, chi + 4)
    green = 1  # BGR from cv2; the channel is arbitrary, the frame is grey.
    print()
    print(_grid(pt[lo:hi, wlo:whi, green], "path traced (8 spp)", wlo))
    print()
    print(_grid(det[lo:hi, wlo:whi, green], "deterministic, ceiling kernel", wlo))
    if torch_det is not None:
        print()
        print(
            _grid(torch_det[lo:hi, wlo:whi, green], "deterministic, ceiling torch", wlo)
        )
    print()
    print(_grid(err[lo:hi, wlo:whi], "|pt - det(kernel)|", wlo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
