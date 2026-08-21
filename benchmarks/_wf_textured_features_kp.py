"""Marginal cost of each monolith feature added back to the lean textured
wavefront shade kernel, measured by **Taichi kernel-profiler device time** of
the shade kernel itself (per the project's benchmarking guidance that wall time
is too noisy on this GPU -- kernel-isolated device time is far cleaner).

Renders the same all-Surface scene through the per-vertex monolith and the
textured kernel with features compiled in cumulatively (lean -> +beziers ->
+scatter dispatch -> +shadows -> +normal maps), reporting the summed GPU time of
the shade kernel (``wavefront_shade`` for the monolith, ``wf_shade_textured``
for the textured configs) and of ``wavefront_traverse`` for reference. Configs
are interleaved with alternating order to cancel thermal drift; rep 0 is a
compile warm-up.

    .venv/Scripts/python.exe benchmarks/_wf_textured_features_kp.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import taichi as ti  # noqa: E402

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    IN,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Cylinder,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_reflectivity,
    set_refractive_index,
)
from algan.rendering.raytracing.settings import (  # noqa: E402
    WF_TEX_BEZ,
    WF_TEX_NORMALMAP,
    WF_TEX_SCATTER,
    WF_TEX_SHADOWS,
    set_textured_features,
    set_textured_wavefront,
)
from algan.utils.profiling_utils import (  # noqa: E402
    _collect_taichi_kernel_gpu,
    enable_taichi_kernel_profiler,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

enable_taichi_kernel_profiler()  # must run before the first render

_M = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshLambertMaterial(color=ORANGE),
]

B, S, SH, NM = WF_TEX_BEZ, WF_TEX_SCATTER, WF_TEX_SHADOWS, WF_TEX_NORMALMAP
STAGES = [
    ("monolith (per-vertex)", False, 0),
    ("textured lean", True, 0),
    ("+ beziers", True, B),
    ("+ scatter dispatch", True, B | S),
    ("+ shadows", True, B | S | SH),
    ("+ normal maps", True, B | S | SH | NM),
]


def build():
    with Sync():
        for i in range(12):
            row, col = divmod(i, 4)
            (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 1.5) * 1.5 + UP * (row - 1) * 1.6)
                .set_material(_M[i % 4]())
                .spawn()
            )
        m = (
            Cylinder(radius=0.35, height=2.2)
            .move(DOWN * 2.6 + LEFT * 3)
            .set_material(MeshLambertMaterial(color=WHITE))
        )
        set_reflectivity(m, 0.6)
        m.spawn()
        g = (
            Sphere()
            .scale(0.7)
            .move(DOWN * 2.4 + RIGHT * 2.5 + IN * 1.0)
            .set_material(MeshLambertMaterial(color=YELLOW))
        )
        set_refractive_index(g, 1.5)
        g.spawn()


def _kernel_ms(rows, name):
    return sum(r["total_ms"] for r in rows if r["name"] == name)


def render_stage(textured, feat, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_textured_wavefront(textured)
    if textured:
        set_textured_features(feat)
    build()
    ti.profiler.clear_kernel_profiler_info()
    SceneManager.instance().save_frame(os.path.join(OUT_DIR, f"kp_{tag}.png"))
    ti.sync()
    rows = _collect_taichi_kernel_gpu()
    shade = _kernel_ms(rows, "wf_shade_textured") + _kernel_ms(rows, "wavefront_shade")
    trav = _kernel_ms(rows, "wavefront_traverse")
    return shade, trav


def main():
    shade = {lbl: [] for lbl, _, _ in STAGES}
    trav = {lbl: [] for lbl, _, _ in STAGES}
    for rep in range(REPS + 1):
        order = STAGES if rep % 2 == 0 else list(reversed(STAGES))
        for lbl, textured, feat in order:
            s, t = render_stage(textured, feat, lbl.replace(" ", "_"))
            if rep > 0:
                shade[lbl].append(s)
                trav[lbl].append(t)
    ms = {lbl: statistics.median(v) for lbl, v in shade.items()}
    mt = {lbl: statistics.median(v) for lbl, v in trav.items()}
    lean = ms["textured lean"]
    mono = ms["monolith (per-vertex)"]
    print(
        f"\n{'stage':26s} {'shade GPU':>10s} {'vs prev':>8s} {'vs lean':>8s} "
        f"{'traverse':>9s}",
        flush=True,
    )
    prev = None
    for lbl, _, _ in STAGES:
        t = ms[lbl]
        vp = "" if prev is None else f"{(t - prev) / prev * 100:+.1f}%"
        vl = f"{t / lean:.2f}x"
        print(f"{lbl:26s} {t:8.3f}ms {vp:>8s} {vl:>8s} {mt[lbl]:7.3f}ms", flush=True)
        prev = t
    print(
        f"\nmonolith shade {mono:.3f}ms  vs textured-lean {lean:.3f}ms  "
        f"= {mono / lean:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
