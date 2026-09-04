"""Parity + timing check for the Taichi warm-start memoization.

Renders the material smoke scene (one save_frame through the real wavefront
pipeline) in two subprocesses -- ALGAN_TAICHI_WARMSTART=0 (patch disabled)
and =1 (default) -- and asserts the PNGs are byte-identical, reporting the
save_frame wall time of each. Run twice if the offline cache is cold; the
timing comparison is only meaningful on warm-cache runs.

    .venv/Scripts/python.exe benchmarks/_taichi_warmstart_check.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def child(out_png):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algan import (  # noqa: E402
        BLUE,
        LEFT,
        RED,
        RIGHT,
        UP,
        MeshNormalMaterial,
        MeshPhongMaterial,
        MeshStandardMaterial,
        SceneManager,
        Sphere,
        Sync,
    )

    with Sync():
        a = (
            Sphere()
            .move(LEFT * 3)
            .set_material(
                MeshPhongMaterial(
                    color=BLUE, specular=0xFFFFFF, shininess=60, emissive=0x330000
                )
            )
            .spawn()
        )
        Sphere().set_material(
            MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.25)
        ).spawn()
        Sphere().move(RIGHT * 3).set_material(MeshNormalMaterial()).spawn()
    a.move(UP * 0.5)

    t0 = time.perf_counter()
    SceneManager.instance().save_frame(out_png)
    print(f"SAVE_FRAME_SECONDS {time.perf_counter() - t0:.3f}", flush=True)


def run(warmstart, out_png, verify=False):
    env = dict(os.environ)
    env["ALGAN_TAICHI_WARMSTART"] = warmstart
    if verify:
        # Recomputes every get_pos_info with the original implementation and
        # raises in-process on any byte difference (fast path + memo keys).
        env["ALGAN_TAICHI_WARMSTART_VERIFY"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", out_png],
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"child (warmstart={warmstart}) failed: {proc.returncode}")
    sf = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("SAVE_FRAME_SECONDS")
    ]
    save_frame = float(sf[0].split()[1]) if sf else float("nan")
    with open(out_png, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    label = f"warmstart={warmstart}" + ("+verify" if verify else "")
    print(
        f"{label:20s}: save_frame {save_frame:6.2f} s  "
        f"total {wall:6.2f} s  sha256 {digest[:16]}",
        flush=True,
    )
    return digest


def main():
    d0 = run("0", os.path.join(OUT_DIR, "warmstart_off.png"))
    d1 = run("1", os.path.join(OUT_DIR, "warmstart_on.png"))
    dv = run("1", os.path.join(OUT_DIR, "warmstart_verify.png"), verify=True)
    if d0 != d1 or d0 != dv:
        raise SystemExit("FAIL: outputs differ")
    print("PASS: byte-identical output (incl. per-call verified strings)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
