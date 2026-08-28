"""A/B the cost of the semi-transparent-PBR reflection split.

Same scene twice, differing only in the PBR ball's opacity:

  A: opacity 1.00 -> has_refl_transparent False -> split pool off (baseline)
  B: opacity 0.99 -> has_refl_transparent True  -> split pool on

B is the price of tracing both the reflection and the pass-through: the split
pool costs slots per pixel (smaller tiles) and disables wf_gen_fused. Runs
alternate in one process, since cross-process wall time swings with thermal
throttling (see the `algan-render-benchmarking` notes).
"""

from __future__ import annotations

import time

from algan import *


def build(opacity):
    Scene.instance().reset_scene()
    with Off():
        Square(color=GREEN).look(UP).move(DOWN).scale(10).spawn()
        Square(color=WHITE).look(LEFT + OUT * 0.5).move(RIGHT * 2).set_material(
            MeshStandardMaterial(color=WHITE, roughness=0.1, metalness=0.9)
        ).spawn()
        ball = Sphere(color=RED).move(LEFT * 2).set_material(MeshPhysicalMaterial())
        ball.set(opacity=opacity).spawn()


def run(opacity):
    build(opacity)
    t0 = time.perf_counter()
    Scene.instance().save_frame(f"_ab_{opacity}.png", render_settings=MD)
    return time.perf_counter() - t0


ITERS = 4
run(1.0)  # warm kernels / caches for both configs
run(0.99)
times = {1.0: [], 0.99: []}
for _ in range(ITERS):
    for op in (1.0, 0.99):
        times[op].append(run(op))

for op, ts in times.items():
    ts = sorted(ts)
    label = "split OFF (opaque)" if op == 1.0 else "split ON  (translucent)"
    print(f"{label}: median {ts[len(ts) // 2]:.3f}s  min {ts[0]:.3f}s")
