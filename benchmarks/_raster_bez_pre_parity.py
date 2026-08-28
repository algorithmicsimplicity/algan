"""Byte parity: hybrid-raster candidate emission, per-(tile, frame) vs the
batched once-per-window screen-bounds precomputes
(ALGAN_RASTER_BEZ_PRECOMPUTE + ALGAN_RASTER_TRI_PRECOMPUTE, default on).

The precomputes must reproduce ``_frame_bez_pairs`` / ``_frame_pairs``
byte-for-byte: identical candidate pair tensors (content *and* row order --
fragment sorting breaks ties by pre-sort position) feeding the same raster
kernels.  Each config is rendered twice -- precomputes off (legacy per-frame
emission) and on -- and the decoded videos must match pixel-exactly.

The wavefront tile is pinned small (ALGAN_WAVEFRONT_TILE) so every window
splits into many tiles with partial top/bottom row bands crossing frame
boundaries: the exact geometry the batched row-band clamp must reproduce.

Configs:
    shapes   filled + translucent + bordered moving circuits, one crossing
             the camera plane (straddler/behind classification)
    text     glyph circuits, static (dedup T=1 sources) + moving
    tri      triangle meshes only: opaque + translucent + mid-scene spawn
    mixed    circuits + a triangle mesh + a mid-scene spawn (validity)
    shadow   + fragment shading + hard shadows (sparse raster shadow queue)
    size0    a single scale-0 circuit (degenerate bounds, empty candidates)

Engagement is asserted per render: the ON render must call at least one
precompute and never a legacy per-frame path, and vice versa.

Run: .venv/Scripts/python.exe benchmarks/_raster_bez_pre_parity.py [configs...]
"""

from __future__ import annotations

import os
import sys

# Pin small fixed wavefront tiles BEFORE the algan import (the env var also
# disables auto tile sizing): many multi-frame tiles per window with partial
# row bands, the regime the batched pair emission must match exactly.
os.environ.setdefault("ALGAN_WAVEFRONT_TILE", "200000")
os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.raster_pipeline as rp  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    PREVIEW,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Off,
    Scene,
    SceneManager,
    Seq,
    Sphere,
    Square,
    Sync,
    Text,
    Triangle,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_shadows,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Engagement probes: the tracer imports these from the module per call, so
# rebinding the module attributes intercepts every use.
_counts = {"bez_pre": 0, "bez_legacy": 0, "tri_pre": 0, "tri_legacy": 0, "raster": 0}


def _probe(key, orig):
    def wrapper(*a, **k):
        _counts[key] += 1
        return orig(*a, **k)

    return wrapper


rp.precompute_circuit_screen_bounds = _probe(
    "bez_pre", rp.precompute_circuit_screen_bounds
)
rp.precompute_triangle_screen_bounds = _probe(
    "tri_pre", rp.precompute_triangle_screen_bounds
)
rp._frame_bez_pairs = _probe("bez_legacy", rp._frame_bez_pairs)
rp._frame_pairs = _probe("tri_legacy", rp._frame_pairs)
rp.raster_iteration_zero = _probe("raster", rp.raster_iteration_zero)
# The default route is the sparse covered-pixel lifecycle
# (raster_sparse_coverage); raster_iteration_zero is the dense fallback. Either
# one counts as the raster front-end having engaged.
rp.prepare_sparse_raster_coverage = _probe("raster", rp.prepare_sparse_raster_coverage)


def build_scene(cfg):
    if cfg == "size0":
        with Off():
            Triangle().scale(0).spawn()
        Scene.wait(2)
        return
    if cfg == "tri":
        with Off():
            s1 = Sphere().scale(1.0).move(LEFT * 1.6).set_color(BLUE)
            s1.spawn()
            s2 = Sphere().scale(0.7).move(RIGHT * 1.4 + UP * 0.5)
            s2.opacity = 0.5  # translucent triangles: fragment stream
            s2.spawn()
        with Sync():
            s1.move(RIGHT * 1.0)
            s2.move(DOWN * 0.8)
        with Seq():
            late = Sphere().scale(0.4).move(UP * 1.2)
            late.set_color(RED)
            late.spawn()  # mid-scene spawn: per-frame validity
            late.move(LEFT * 0.9)
        return
    with Off():
        sq = Square(color=RED).scale(0.9).move(LEFT * 1.6 + UP * 0.6)
        sq.spawn()
        ci = Circle(color=GREEN).scale(0.7).move(RIGHT * 1.2 + DOWN * 0.5)
        ci.opacity = 0.55  # translucent circuit: fragment stream
        ci.spawn()
        tr = Triangle(color=YELLOW).scale(0.8).move(DOWN * 1.4)
        tr.spawn()
        if cfg in ("text", "shadow"):
            Text("static").scale(0.4).move(UP * 1.7 + LEFT * 2.0).spawn()
            title = Text("Algan raster").scale(0.6).move(UP * 1.5)
            title.spawn()
        if cfg in ("mixed", "shadow"):
            sph = Sphere().scale(0.9).move(OUT * -1.0).set_color(BLUE)
            sph.spawn()
    with Sync():
        sq.rotate(35, OUT)
        ci.move(LEFT * 0.8)
        # Crosses the camera plane: exercises the straddler / behind-camera
        # classification (full-band fallback vs cull).
        tr.move(OUT * 9.0)
        if cfg in ("text", "shadow"):
            title.move(DOWN * 0.4)
        if cfg in ("mixed", "shadow"):
            sph.move(RIGHT * 1.0)
    if cfg == "mixed":
        with Seq():
            late = Circle(color=YELLOW).scale(0.5).move(UP * 0.9)
            late.spawn()  # mid-scene spawn: per-frame validity
            late.move(LEFT * 0.7)


def render_once(cfg, precompute, tag):
    SceneManager.reset()
    set_fragment_shading(cfg == "shadow")
    set_shadows(cfg == "shadow")
    rt_settings.set_raster_bez_precompute(precompute)
    rt_settings.set_raster_tri_precompute(precompute)
    build_scene(cfg)
    for k in _counts:
        _counts[k] = 0
    name = f"bezpre_{cfg}_{tag}"
    Scene.save_video(os.path.join(OUT_DIR, name), PREVIEW, reset=True)
    # size0's only mob is degenerate, so it is culled upstream and the raster
    # front-end is never reached -- the comparison still checks both routes
    # agree on an empty frame.
    if cfg != "size0":
        assert _counts["raster"] > 0, f"raster did not engage ({cfg}/{tag})"
    pre = _counts["bez_pre"] + _counts["tri_pre"]
    legacy = _counts["bez_legacy"] + _counts["tri_legacy"]
    if precompute:
        assert legacy == 0, f"legacy leaked through ({cfg}/{tag}): {_counts}"
        if cfg != "size0":
            assert pre > 0, f"precompute engagement failure ({cfg}/{tag}): {_counts}"
    else:
        assert pre == 0, f"precompute leaked through ({cfg}/{tag}): {_counts}"
        if cfg != "size0":
            assert legacy > 0, f"legacy engagement failure ({cfg}/{tag}): {_counts}"
    print(
        f"  {cfg}/{tag}: raster tiles={_counts['raster']} "
        f"bez pre/legacy={_counts['bez_pre']}/{_counts['bez_legacy']} "
        f"tri pre/legacy={_counts['tri_pre']}/{_counts['tri_legacy']}",
        flush=True,
    )
    rt_settings.set_raster_bez_precompute(True)
    rt_settings.set_raster_tri_precompute(True)
    set_fragment_shading(False)
    set_shadows(False)
    return os.path.join(OUT_DIR, name + ".mp4")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.int32))
    cap.release()
    return frames


def main():
    configs = sys.argv[1:] or ["shapes", "text", "tri", "mixed", "shadow", "size0"]
    all_ok = True
    for cfg in configs:
        legacy = render_once(cfg, False, "legacy")
        pre = render_once(cfg, True, "pre")
        fa, fb = read_frames(legacy), read_frames(pre)
        if len(fa) != len(fb) or not fa:
            print(f"[{cfg:6s}] FAIL: frame count {len(fa)} vs {len(fb)}")
            all_ok = False
            continue
        worst = max(int(np.abs(a - b).max()) for a, b in zip(fa, fb))
        ok = worst == 0
        all_ok = all_ok and ok
        print(
            f"[{cfg:6s}] frames={len(fa):3d}  max|d|={worst}  "
            f"{'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
    print("\nBEZ_PRE_PARITY_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
