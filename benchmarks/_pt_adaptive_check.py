"""Acceptance harness for adaptive sampling (roadmap §2): what it may move.

``pt_error_target`` is a runtime setting (the pixel list is host-side and
``pt_reduce`` takes ``adaptive`` as a runtime word), so both arms run in ONE
process on ONE scene and the comparison is on the raw frames
``Scene.get_frames`` returns -- byte counts, not a lossy video.

Two scenes from ``benchmarks/performance/pt_baseline.py``, each rendered at
``--pt-error-target 0`` (uniform, every pixel gets the ceiling) and at the
shipped default, with the denoiser off and then on:

* ``lit`` -- an all-opaque lit scene. Every lit pixel is stochastic, so the
  gate in ``path_tracer._pt_active_pixels`` runs it to the ceiling and the
  frame must come out **byte-identical** with the denoiser off. Only the
  background stops early; its value is exact, and the sole thing that can
  differ is float summation order (4 samples scaled by 4 against 16 samples
  summed), which is sub-count. With the denoiser on, a sub-ULP input
  difference can move a count through the network, so that arm reports
  rather than asserts.
* ``text_2d`` -- unlit 2-D content. Interiors are zero-variance and stop at
  the floor with the same value; the only differences allowed are on
  geometry EDGES, where jittered anti-aliasing at ``pt_min_samples`` samples
  replaces jittered anti-aliasing at the ceiling. The harness counts how
  many differing pixels have a non-differing 4-neighbourhood -- an interior
  difference -- which must be zero.

Prints one ``RESULTS`` JSON line per scene. Exit code 0 only when the lit
scene is byte-identical with the denoiser off and the text scene moved no
interior pixel.

Usage::

    uv run python benchmarks/_pt_adaptive_check.py
    uv run python benchmarks/_pt_adaptive_check.py --resolution 1280x720 --scene lit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

FRAMES = 2


def _parse_resolution(text):
    w, h = (int(p) for p in text.lower().split("x"))
    return w, h


def _interior_differences(diff_mask, reference):
    """Differing pixels whose 3x3 neighbourhood in the UNIFORM reference is
    flat -- no edge there, so a difference is a change to an interior.

    Judged on the reference rather than on the difference mask: a one-pixel
    glyph stroke is all edge, and its differing pixels can have no differing
    neighbour at all, which a mask-based test would misread as interior.
    """
    import numpy as np

    ref = reference
    lo = ref.copy()
    hi = ref.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(ref, dy, axis=1), dx, axis=2)
            lo = np.minimum(lo, shifted)
            hi = np.maximum(hi, shifted)
    flat = (hi - lo).max(-1) == 0
    return int((diff_mask & flat).sum())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=_parse_resolution, default=(320, 180))
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--scene", choices=("lit", "text_2d", "both"), default="both")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    import algan as A
    from algan import SETTINGS
    from algan.scene_manager import SceneManager
    from algan.settings._startup import render_device
    from benchmarks.performance import pt_baseline as pb

    scenes = ("lit", "text_2d") if args.scene == "both" else (args.scene,)
    target_default = float(SETTINGS.raytracing.experimental.pt_error_target)
    ok = True
    for name in scenes:
        SceneManager.reset()
        SETTINGS.raytracing.set(samples_per_pixel=args.spp, max_bounces=4, shadows=True)
        q = A.SMOKE_TEST.set(resolution=args.resolution)
        SceneManager.instance().current_scene.set_video_settings(q)
        getattr(pb, f"scene_{name}")()
        scene = SceneManager.instance().current_scene
        summary = {
            "scene": name,
            "device": str(render_device()),
            "resolution": list(args.resolution),
            "spp": args.spp,
            "target": target_default,
        }
        for denoise in (False, True):
            SETTINGS.raytracing.set(denoise=denoise)
            frames = {}
            for target in (0.0, target_default):
                SETTINGS.raytracing.experimental.set(pt_error_target=target)
                frames[target] = (
                    torch.cat([f.cpu() for f in scene.get_frames(0, FRAMES)])
                    .numpy()
                    .astype(np.int32)
                )
                if not denoise:
                    key = "mean_spp_uniform" if target == 0.0 else "mean_spp_adaptive"
                    summary[key] = round(scene.last_render_plan.path_samples_mean, 3)
            diff = np.abs(frames[0.0] - frames[target_default])
            per_pixel = diff.max(-1) if diff.ndim == 4 else diff
            mask = per_pixel > 0
            arm = "denoise_on" if denoise else "denoise_off"
            summary[arm] = {
                "max": int(diff.max()),
                "mean": round(float(diff.mean()), 5),
                "pixels_differing": int(mask.sum()),
                "pixels_total": int(mask.size),
                "interior_differences": _interior_differences(mask, frames[0.0]),
            }
            print(
                f"{name} {arm}: max |diff| {summary[arm]['max']}, "
                f"mean {summary[arm]['mean']}, {summary[arm]['pixels_differing']} "
                f"of {summary[arm]['pixels_total']} pixels differ, "
                f"{summary[arm]['interior_differences']} of them interior"
            )
        SETTINGS.raytracing.experimental.set(pt_error_target=target_default)
        if name == "lit" and summary["denoise_off"]["max"] != 0:
            ok = False
            print("!! lit scene is not byte-identical with the denoiser off")
        if name == "text_2d" and summary["denoise_off"]["interior_differences"] != 0:
            ok = False
            print("!! text_2d moved an interior pixel")
        print("RESULTS " + json.dumps(summary))
    print("all arms agree" if ok else "ARMS DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
