"""Why the glossy tile loop walks off the end of its frame table on MPS.

Thirteen of the failures the ordinary macOS MPS gate reports are one line:

    frame_end = gl_bounds[gl_frame + 1]
    IndexError: list index out of range

``tracer._gloss_frame_bounds`` returns ``num_frames + 1`` ordinals, one per
frame boundary, and the tile loop walks them in order; running past the end
means a boundary came back SMALLER than the number of covered pixels, so the
loop still had pixels to place after it had finished the last frame. That is a
wrong answer from a torch op, not a control-flow bug -- the same loop is green
on CPU and CUDA -- and this says which op and by how much.

It renders the smallest scene that reaches the glossy route and wraps
``_gloss_frame_bounds`` for the duration, printing, per call: the covered
ordinals' dtype, length, range and whether they are actually ascending; the
bounds the device computed; and the bounds the same inputs give on the host.
A disagreement there is the defect; agreement moves the question upstream to
``covered_idx`` itself, which is why its stats are printed rather than assumed.

    uv run python benchmarks/_mps_gloss_bounds_probe.py

Exits non-zero if the device's bounds differ from the host's, or if the
covered ordinals are not ascending (which would make *any* searchsorted answer
meaningless). On a machine with no Apple GPU it renders on the CPU and checks
the same invariants, which is how the probe itself is debugged.
"""

from __future__ import annotations

import os
import sys

# The point is to see MPS answer, not to watch torch quietly answer on the CPU
# for it. Set before torch is imported, like every other probe here.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch  # noqa: E402

from algan import SETTINGS, SMOKE_TEST, Off, Scene, Square  # noqa: E402
from algan.constants.color import BLUE  # noqa: E402
from algan.rendering.raytracing import tracer  # noqa: E402

_PROBLEMS: list[str] = []


def _describe(covered_idx, pixels_per_frame, num_frames, device_bounds):
    """Print one call's inputs and both answers; record any disagreement."""
    host = covered_idx.cpu()
    ascending = bool((host[1:] >= host[:-1]).all()) if host.numel() > 1 else True
    edges = torch.arange(num_frames + 1, dtype=host.dtype) * int(pixels_per_frame)
    host_bounds = torch.searchsorted(host, edges).tolist()

    print(
        f"  covered_idx: dtype={covered_idx.dtype} n={host.numel()} "
        f"min={int(host.min()) if host.numel() else '-'} "
        f"max={int(host.max()) if host.numel() else '-'} "
        f"ascending={ascending}"
    )
    print(f"  window     : {num_frames} frame(s) x {pixels_per_frame} px")
    print(f"  device     : {device_bounds}")
    print(f"  host       : {host_bounds}")
    if not ascending:
        _PROBLEMS.append("covered_idx is not ascending on the render device")
    if device_bounds != host_bounds:
        _PROBLEMS.append(f"bounds differ: device {device_bounds} vs host {host_bounds}")
    # The invariant the tile loop actually depends on, stated on its own so a
    # green run says which property was checked.
    if device_bounds and device_bounds[-1] != host.numel():
        _PROBLEMS.append(
            f"last bound {device_bounds[-1]} != {host.numel()} covered pixels, "
            "which is exactly what walks gl_frame off the end"
        )


def main() -> int:
    device = SETTINGS.computing.render_device
    print(f"torch {torch.__version__}, render device {device}")
    print(f"glossy_reflection_mode = {SETTINGS.raytracing.glossy_reflection_mode()}")

    original = tracer._gloss_frame_bounds
    calls = [0]

    def wrapped(covered_idx, pixels_per_frame, num_frames):
        bounds = original(covered_idx, pixels_per_frame, num_frames)
        calls[0] += 1
        print(f"\n_gloss_frame_bounds call {calls[0]}:")
        _describe(covered_idx, pixels_per_frame, num_frames, bounds)
        return bounds

    tracer._gloss_frame_bounds = wrapped
    try:
        SETTINGS.video.set(SMOKE_TEST)
        scene = Scene()
        with scene:
            with Off():
                Square(size=5.0, color=BLUE).spawn()
            scene.wait(0.5)
        scene.save_video("mps_gloss_bounds_probe", overwrite=True)
    finally:
        tracer._gloss_frame_bounds = original

    print()
    if calls[0] == 0:
        print("the glossy route never ran -- this scene does not reach it")
        return 1
    if _PROBLEMS:
        print(f"{len(_PROBLEMS)} problem(s):")
        for problem in _PROBLEMS:
            print(f"  - {problem}")
        return 1
    print(f"{calls[0]} call(s), every bound matches the host")
    return 0


if __name__ == "__main__":
    sys.exit(main())
