"""Parity check for adaptive ("auto") gen-fused switching.

Renders the same short animated scene (PREVIEW, multi-batch: the animate
memory budget is squeezed so the job splits into several batches) three ways
in subprocesses:

  fused    ALGAN_WF_GEN_FUSED=1        every batch uses the fused kernels
  classic  ALGAN_WF_GEN_FUSED=0        every batch uses the classic kernels
  auto     (env unset) + tiny MIN_WIN  starts classic, must switch to fused
                                       mid-render (the mode's whole point)

and asserts all three videos decode to identical pixels. All arms use the
same squeezed budget so their batch windows -- and therefore the CPU
rate-function rounding (see CLAUDE.md) -- are identical, making byte-exact
comparison valid. Also asserts the auto arm actually logged the mid-render
switch, so the parity covers a batch boundary where the kernel set changes.

    .venv/Scripts/python.exe benchmarks/_wf_gen_fused_adaptive_ab.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

SWITCH_LOG = "Adaptive gen-fused"
# Squeeze the animate budget so the ~3 s scene splits into several batches.
ANIMATE_PORTION = float(os.environ.get("ALGAN_AB_ANIMATE_PORTION", "2e-4"))


def child(name):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algan import (  # noqa: E402
        BLUE,
        DOWN,
        GREEN,
        LEFT,
        PREVIEW,
        RED,
        RIGHT,
        UP,
        WHITE,
        Cylinder,
        MeshLambertMaterial,
        MeshPhongMaterial,
        MeshStandardMaterial,
        Sphere,
        Sync,
    )
    from algan.settings.defaults import COMPUTING_DEFAULTS  # noqa: E402
    from algan.utils.algan_utils import render_to_file  # noqa: E402

    COMPUTING_DEFAULTS.portion_of_memory_used_for_animating = ANIMATE_PORTION

    with Sync():
        a = (
            Sphere()
            .scale(0.8)
            .move(LEFT * 2.5)
            .set_material(MeshLambertMaterial(color=BLUE))
            .spawn()
        )
        b = (
            Sphere()
            .scale(0.8)
            .set_material(MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50))
            .spawn()
        )
        c = (
            Sphere()
            .scale(0.8)
            .move(RIGHT * 2.5)
            .set_material(
                MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3)
            )
            .spawn()
        )
        m = (
            Cylinder(radius=0.4, height=1.8)
            .move(DOWN * 2)
            .set_material(
                MeshStandardMaterial(color=WHITE, metalness=0.6, roughness=0.4)
            )
            .spawn()
        )
    for mob, d in ((a, UP), (b, DOWN), (c, UP), (m, RIGHT)):
        mob.move(d * 0.8)

    render_to_file(file_name=name, output_dir=OUT_DIR, render_settings=PREVIEW)


def decode(path):
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(frames)


def run(label, env_overrides, name):
    env = dict(os.environ)
    env.pop("ALGAN_WF_GEN_FUSED", None)
    # The "Fetching batch" / "Adaptive gen-fused" lines scraped below are DEBUG.
    env["ALGAN_LOG_LEVEL"] = "DEBUG"
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name],
        env=env,
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(log)
        raise SystemExit(f"child ({label}) failed: {proc.returncode}")
    batches = log.count("Fetching batch")
    switched = SWITCH_LOG in log
    print(f"{label:8s}: {batches} batch fetches, switched={switched}", flush=True)
    return switched, batches


def main():
    run("fused", {"ALGAN_WF_GEN_FUSED": "1"}, "genfused_on")
    run("classic", {"ALGAN_WF_GEN_FUSED": "0"}, "genfused_off")
    switched, batches = run(
        "auto", {"ALGAN_WF_GEN_FUSED_MIN_WIN": "0.0001"}, "genfused_auto"
    )

    vids = {
        n: decode(os.path.join(OUT_DIR, n + ".mp4"))
        for n in ("genfused_on", "genfused_off", "genfused_auto")
    }
    ref = vids["genfused_on"]
    ok = True
    for n, v in vids.items():
        if v.shape != ref.shape:
            print(f"FAIL: {n} shape {v.shape} != {ref.shape}")
            ok = False
            continue
        diff = int(np.abs(v.astype(np.int16) - ref.astype(np.int16)).max())
        print(f"{n}: max |diff| vs fused = {diff}")
        ok = ok and diff == 0
    if not switched:
        print(
            f"WARN: auto arm never switched (only {batches} batches); "
            "increase scene length or lower ALGAN_AB_ANIMATE_PORTION "
            "for the switch itself to be exercised."
        )
        ok = False
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
