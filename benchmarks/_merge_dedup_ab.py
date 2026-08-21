"""Byte-parity A/B for MERGE_DEDUP_TIME + OPAQUE_BVH_SKIP_DEAD.

Both features change only how the merged scene is stored (collapsed
time-constant tables; opaque trees aliased while no rollout walks them), so
on a single-batch render -- where the arena planner cannot re-window -- the
output must be byte-identical with them on and off. Renders the same mixed
scene (PBR + glass triangles, PN spheres, text circuits, spot + ranged
point lights, shadows) in subprocesses and compares mp4 SHA256.

    .venv/Scripts/python.exe benchmarks/_merge_dedup_ab.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

OUT_DIR = os.path.join(os.path.dirname(__file__), "_merge_dedup_out")
os.makedirs(OUT_DIR, exist_ok=True)


def child(out_name):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algan import (  # noqa: E402
        BLUE,
        DOWN,
        GREEN,
        LEFT,
        ORIGIN,
        OUT,
        PREVIEW,
        RED,
        RIGHT,
        UP,
        WHITE,
        MeshLambertMaterial,
        MeshStandardMaterial,
        PointLight,
        Scene,
        Sphere,
        SpotLight,
        Square,
        Sync,
        Text,
    )
    from algan.rendering.raytracing import (  # noqa: E402
        set_fragment_shading,
        set_ray_traced_shadows,
    )

    set_fragment_shading(True)
    set_ray_traced_shadows(True)
    with Sync():
        SpotLight(
            location=UP * 4 + LEFT * 2,
            target=ORIGIN,
            angle=0.35,
            penumbra=0.4,
            color=WHITE,
        ).spawn()
        PointLight(location=UP * 2 + RIGHT * 3, distance=4.0, color=WHITE).spawn()
        floor = Square(color=WHITE).scale(6).rotate(90, RIGHT).move(DOWN * 1.2)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn()
        s1 = Sphere(radius=0.5).set_material(
            MeshStandardMaterial(color=RED, roughness=0.3, metalness=0.8)
        )
        s1.move(LEFT * 1.5).spawn()
        s2 = Sphere(radius=0.5).set_material(MeshLambertMaterial(color=GREEN))
        s2.spawn()
        sq = Square(color=BLUE).scale(0.8).move(RIGHT * 1.6 + UP * 0.4)
        sq.opacity = 0.6
        sq.spawn()
        Text("dedup parity").scale(0.5).move(UP * 1.8).spawn()
    with Sync():
        s1.move(RIGHT * 0.8)
        sq.rotate(40, OUT)

    t0 = time.perf_counter()
    Scene.save_video(os.path.join(OUT_DIR, out_name), PREVIEW, overwrite=True)
    print(f"RENDER_SECONDS {time.perf_counter() - t0:.3f}", flush=True)


def run(name, env_extra):
    env = dict(os.environ)
    env["ALGAN_PREFETCH_BATCHES"] = "0"
    # The "Fetching batch" lines scraped below are DEBUG.
    env["ALGAN_LOG_LEVEL"] = "DEBUG"
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"child {name} failed: {proc.returncode}")
    # Algan's log handler writes to stderr, so stdout alone never matched.
    batches = [
        ln
        for ln in (proc.stdout + proc.stderr).splitlines()
        if ln.startswith("Fetching batch")
    ]
    with open(os.path.join(OUT_DIR, name + ".mp4"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{name:12s}: windows={batches}  sha256={digest[:16]}", flush=True)
    return digest, tuple(batches)


def main():
    d_on, w_on = run("features_on", {})
    d_off, w_off = run(
        "features_off",
        {"ALGAN_MERGE_DEDUP_TIME": "0", "ALGAN_OPAQUE_BVH_SKIP_DEAD": "0"},
    )
    if w_on != w_off:
        raise SystemExit(
            "batch windows differ -- scene is not single-batch, strict "
            "comparison invalid; shrink the scene"
        )
    if d_on != d_off:
        raise SystemExit("FAIL: outputs differ with identical windows")
    print(
        "PASS: byte-identical with dedup + opaque-skip on vs off "
        "(identical batch windows)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
