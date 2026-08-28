"""Measure the marginal cost of adding each monolith feature back into the
lean textured wavefront shade kernel, one at a time.

Renders the same all-Surface scene through: the current per-vertex monolith
wavefront, then the textured kernel with features compiled in cumulatively --
lean -> +beziers -> +scatter dispatch -> +shadows -> +normal maps (see
settings.wf_textured_features). Reports the median wavefront render-stage time
(CUDA-synced around ``raytrace_render_wavefront``) at each stage, the marginal
hit vs the previous stage, and the speed vs the monolith. Configs are
interleaved within each rep to cancel thermal-throttle drift; the first rep is a
compile / clock warm-up and discarded.

    .venv/Scripts/python.exe benchmarks/_wf_textured_features_ab.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.tracer as tracer_mod  # noqa: E402
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
    set_wf_textured,
    set_wf_textured_features,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

_M = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshLambertMaterial(color=ORANGE),
]

B, S, SH, NM = WF_TEX_BEZ, WF_TEX_SCATTER, WF_TEX_SHADOWS, WF_TEX_NORMALMAP
# (label, textured?, feature mask) -- cumulative.
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


_wf_times = []
_orig = tracer_mod.raytrace_render_wavefront


def _timed(*a, **k):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = _orig(*a, **k)
    torch.cuda.synchronize()
    _wf_times.append(time.perf_counter() - t0)
    return r


tracer_mod.raytrace_render_wavefront = _timed


def render_stage(textured, feat, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_wf_textured(textured)
    if textured:
        set_wf_textured_features(feat)
    build()
    _wf_times.clear()
    SceneManager.instance().save_frame(os.path.join(OUT_DIR, f"feat_{tag}.png"))
    return sum(_wf_times)


def main():
    times = {lbl: [] for lbl, _, _ in STAGES}
    for rep in range(REPS + 1):  # rep 0 = warm-up (compile), discarded
        # Alternate the config order each rep so no config is always rendered
        # at the same (thermally-drifted) position within a rep.
        order = STAGES if rep % 2 == 0 else list(reversed(STAGES))
        for lbl, textured, feat in order:
            t = render_stage(textured, feat, lbl.replace(" ", "_"))
            if rep > 0:
                times[lbl].append(t)
    med = {lbl: statistics.median(v) for lbl, v in times.items()}
    mono = med["monolith (per-vertex)"]
    lean = med["textured lean"]
    print(
        f"\n{'stage':26s} {'wavefront':>10s}  {'vs prev':>8s}  {'vs lean':>8s}  "
        f"{'vs monolith':>11s}",
        flush=True,
    )
    prev = None
    for lbl, _, _ in STAGES:
        t = med[lbl]
        vp = (
            ""
            if prev is None
            else f"{t / prev:+.2f}x"
            if False
            else f"{(t - prev) / prev * 100:+.1f}%"
        )
        vl = f"{t / lean:.2f}x"
        vm = f"{mono / t:.2f}x"
        print(f"{lbl:26s} {t * 1e3:8.1f}ms  {vp:>8s}  {vl:>8s}  {vm:>11s}", flush=True)
        prev = t


if __name__ == "__main__":
    with torch.inference_mode():
        main()
