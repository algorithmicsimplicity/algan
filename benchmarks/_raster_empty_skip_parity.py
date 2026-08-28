"""Byte parity: hybrid-raster empty-pixel fast path + host pair flags
(ALGAN_RASTER_EMPTY_SKIP + ALGAN_RASTER_PAIR_FLAGS, default on).

EMPTY_SKIP pre-fills every primary's ``pix_accum`` row with the
retired-empty result so ``raster_first_shade`` threads with nothing to
shade exit before ray generation (worked pixels store their leftover
weight instead of accumulating onto a zero base; bounced pixels zero the
pre-fill back out), tiles without any candidate pairs skip the resolve
and shadow-event launches entirely, AND such whole-tile-empty tiles are
composited by the lean ``empty`` variant of ``wf_composite_accum`` that
skips the dominant per-pixel ``pix_accum`` read (bare-background
``finalize(bg)``).  PAIR_FLAGS hoists a per-frame
(opaque, translucent) candidate-existence summary to the host once per
window so ``_window_pairs`` skips its tensor work -- and the synchronizing
``.nonzero()`` in ``_class_pairs_flat`` -- for provably-empty (tile,
class) combinations.  Both must be byte-identical: each config is rendered
with both toggles off and on, and the decoded videos must match
pixel-exactly.

The wavefront tile is pinned small (ALGAN_WAVEFRONT_TILE) so every window
splits into many tiles, most of which are empty on the sparse configs --
the regime the fast paths target.

Configs:
    size0      a single scale-0 triangle over a 2 s wait (the tiny-scene
               render-floor case: nearly every tile skips the resolve)
    spawnlate  empty lead-in/tail around a mid-scene spawn+despawn
               (whole frames without candidates)
    shapes     filled + translucent + bordered moving circuits, one
               crossing the camera plane
    text       glyph circuits, static + moving
    tri        opaque + translucent spheres + mid-scene spawn
    shadow     + fragment shading + hard shadows (sparse shadow queue
               beside the launch skip)
    refl       glass + semi-transparent metal (bounced/split resolve
               paths: the pre-fill must be zeroed back out)
    env        environment map (disables the launch skip: empty pixels
               still sample the map through the pre-filled state)

Engagement is asserted per config: with the fast path on, the sparse
configs must launch strictly fewer ``raster_first_shade`` kernels and make
strictly fewer ``_class_pairs_flat`` calls than the legacy run; the env
config must launch exactly as many resolves as legacy.

Run: .venv/Scripts/python.exe benchmarks/_raster_empty_skip_parity.py [configs...]
"""

from __future__ import annotations

import os
import sys

# Pin small fixed wavefront tiles BEFORE the algan import (the env var also
# disables auto tile sizing): many tiles per window, most empty on the
# sparse configs.
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
    render_to_file,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_shadows,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Engagement probes: raster_pipeline references these as module globals, so
# rebinding the attributes intercepts every use.  ``empty_composite`` counts
# tiles composited by the lean pix_accum-free variant (empty flag == 1).
_counts = {
    "raster": 0,
    "first_shade": 0,
    "class_pairs": 0,
    "empty_composite": 0,
    "covered_shade": 0,
    "covered_lt_total": 0,
    "composite_calls": 0,
    "composite_covered": 0,
}


def _probe(key, orig):
    def wrapper(*a, **k):
        _counts[key] += 1
        return orig(*a, **k)

    return wrapper


rp.raster_iteration_zero = _probe("raster", rp.raster_iteration_zero)
rp._class_pairs_flat = _probe("class_pairs", rp._class_pairs_flat)

# raster_first_shade positional args. Located by NAME from the kernel's own
# signature rather than hardcoded: the argument list has shifted three times
# while analytic AA was built, and a stale index reads a neighbouring flag and
# silently reports "no engagement" on a run that was in fact fine.
import inspect  # noqa: E402

_FS_PARAMS = list(
    inspect.signature(
        rp.raster_first_shade.__wrapped__
        if hasattr(rp.raster_first_shade, "__wrapped__")
        else rp.raster_first_shade
    ).parameters
)
_FS_NUM_PIXELS = _FS_PARAMS.index("num_pixels")
_FS_COVERED = _FS_PARAMS.index("covered")
_FS_NUM_COVERED = _FS_PARAMS.index("num_covered")
_orig_first_shade = rp.raster_first_shade


def _first_shade_probe(*a, **k):
    _counts["first_shade"] += 1
    if len(a) > _FS_NUM_COVERED and int(a[_FS_COVERED]) == 1:
        _counts["covered_shade"] += 1
        # A covered launch that resolves strictly fewer than all tile pixels
        # is proof the compaction removed empty-pixel threads.
        if int(a[_FS_NUM_COVERED]) < int(a[_FS_NUM_PIXELS]):
            _counts["covered_lt_total"] += 1
    return _orig_first_shade(*a, **k)


rp.raster_first_shade = _first_shade_probe

# wf_composite_accum positional args (see the call in tracer.py):
# [8] empty flag, [9] covered flag, [10] covered_idx, [11] num_covered.
# Under post-process tonemapping (default) empty tiles skip the composite
# entirely and covered tiles pass covered == 1; under in-composite tonemap
# empty tiles use the empty == 1 lean variant instead.
import algan.rendering.raytracing.tracer as _tr  # noqa: E402

_orig_composite = _tr.wf_composite_accum


def _composite_probe(*a, **k):
    _counts["composite_calls"] += 1
    if len(a) > 9:
        if int(a[8]) == 1:
            _counts["empty_composite"] += 1
        if int(a[9]) == 1:
            _counts["composite_covered"] += 1
    return _orig_composite(*a, **k)


_tr.wf_composite_accum = _composite_probe

# The composite compaction (whole-empty skip / covered compaction) only
# engages under post-process tonemapping, where the composite is a linear
# no-op on empty pixels. In-composite tonemap (ALGAN_POST_PROCESS_TONEMAP=0)
# instead uses the lean empty=1 variant, so its composite-engagement asserts
# differ; gate them on the live mode so the script is correct either way.
MODE3 = rt_settings.is_post_process_tonemap_enabled()

# Configs whose ON run must demonstrably skip work (sparse screens).
SPARSE = ("size0", "spawnlate")
# Configs with genuine partial coverage (a real object over empty regions),
# so the resolve must run COMPACTED over fewer-than-all pixels.  size0 is too
# degenerate for this -- its zero-coverage tiles take the whole-tile skip
# instead of a compacted launch -- so it is excluded.
COMPACTED = ("spawnlate",)


def build_scene(cfg):
    if cfg == "size0":
        with Off():
            Triangle().scale(0).spawn()
        Scene.wait(2)
        return
    if cfg == "spawnlate":
        Scene.wait(1)  # leading frames: zero candidates
        with Seq():
            s = Sphere().scale(0.6).move(UP * 0.5).set_color(RED)
            s.spawn()
            s.move(DOWN * 0.8)
            s.despawn()
        Scene.wait(1)  # trailing frames: zero candidates
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
            late.spawn()
            late.move(LEFT * 0.9)
        return
    if cfg == "refl":
        with Off():
            backdrop = Sphere().scale(6).set_color(GREEN).move(DOWN * 0.2 + UP * 4.0)
            backdrop.spawn()
            # Glass (solid refractive): is_glass in-place bounce + split.
            glass = Sphere().scale(0.7).move(LEFT * 1.3)
            glass.set_material(
                MeshPhysicalMaterial(transmission=0.9, roughness=0.05, ior=1.5)
            )
            glass.spawn()
            # Semi-transparent metal: refl-transparent split slot.
            refl = Sphere().scale(0.6).move(RIGHT * 1.3).set_color(RED)
            refl.set_material(MeshStandardMaterial(metalness=0.9, roughness=0.1))
            refl.opacity = 0.5
            refl.spawn()
        with Sync():
            glass.move(RIGHT * 0.4)
            refl.move(LEFT * 0.4 + UP * 0.2)
        return
    if cfg == "env":
        Scene.set_environment_map(
            os.path.join(os.path.dirname(__file__), "..", "world_map.jpg"),
            intensity=1.0,
            ambient=True,
        )
        with Off():
            s = Sphere().scale(0.8).move(LEFT * 1.0).set_color(BLUE)
            s.spawn()
        s.move(RIGHT * 1.2)
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
        if cfg == "shadow":
            sph = Sphere().scale(0.9).move(OUT * -1.0).set_color(BLUE)
            sph.spawn()
    with Sync():
        sq.rotate(35, OUT)
        ci.move(LEFT * 0.8)
        # Crosses the camera plane: straddler / behind-camera classification.
        tr.move(OUT * 9.0)
        if cfg in ("text", "shadow"):
            title.move(DOWN * 0.4)
        if cfg == "shadow":
            sph.move(RIGHT * 1.0)


def render_once(cfg, enabled, tag):
    SceneManager.reset()
    set_fragment_shading(cfg in ("shadow", "refl"))
    set_shadows(cfg == "shadow")
    # This harness isolates the older dense-tile empty/covered fast paths.
    # Exact sparse coverage has its own A/B in
    # ``_raster_sparse_coverage_parity.py`` and otherwise bypasses the probes
    # below by design.
    rt_settings.set_raster_sparse_coverage(False)
    rt_settings.set_raster_empty_skip(enabled)
    rt_settings.set_raster_pair_flags(enabled)
    rt_settings.set_raster_covered_shade(enabled)
    build_scene(cfg)
    for k in _counts:
        _counts[k] = 0
    name = f"emptyskip_{cfg}_{tag}"
    render_to_file(
        file_name=name,
        output_dir=OUT_DIR,
        output_path="",
        render_settings=PREVIEW,
        file_extension="mp4",
    )
    counts = dict(_counts)
    assert counts["raster"] > 0, f"raster did not engage ({cfg}/{tag})"
    if cfg == "env":
        Scene.set_environment_map(None)
    rt_settings.set_raster_empty_skip(True)
    rt_settings.set_raster_pair_flags(True)
    rt_settings.set_raster_covered_shade(True)
    rt_settings.set_raster_sparse_coverage(True)
    set_fragment_shading(False)
    set_shadows(False)
    print(
        f"  {cfg}/{tag}: tiles={counts['raster']} "
        f"first_shade={counts['first_shade']} "
        f"class_pairs={counts['class_pairs']} "
        f"covered={counts['covered_shade']}/"
        f"{counts['covered_lt_total']} "
        f"composite={counts['composite_calls']}"
        f"(cov={counts['composite_covered']},"
        f"empty={counts['empty_composite']})",
        flush=True,
    )
    return os.path.join(OUT_DIR, name + ".mp4"), counts


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
    configs = sys.argv[1:] or [
        "size0",
        "spawnlate",
        "shapes",
        "text",
        "tri",
        "shadow",
        "refl",
        "env",
    ]
    all_ok = True
    for cfg in configs:
        legacy, c_off = render_once(cfg, False, "legacy")
        fast, c_on = render_once(cfg, True, "fast")
        fa, fb = read_frames(legacy), read_frames(fast)
        if len(fa) != len(fb) or not fa:
            print(f"[{cfg:9s}] FAIL: frame count {len(fa)} vs {len(fb)}")
            all_ok = False
            continue
        worst = max(int(np.abs(a - b).max()) for a, b in zip(fa, fb))
        ok = worst == 0
        # Engagement: the fast path must actually skip launches/pair calls
        # on sparse screens, and must NOT skip resolves under an env map.
        if cfg in SPARSE:
            if not (c_on["first_shade"] < c_off["first_shade"]):
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: first_shade "
                    f"{c_on['first_shade']} !< {c_off['first_shade']}"
                )
                ok = False
            if not (c_on["class_pairs"] < c_off["class_pairs"]):
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: class_pairs "
                    f"{c_on['class_pairs']} !< {c_off['class_pairs']}"
                )
                ok = False
            # The fast path must do strictly less composite work on empty
            # tiles: under post-process tonemapping (default) whole-empty
            # tiles skip the composite launch entirely, so the fast run makes
            # fewer composite calls than the full-composite legacy run. Under
            # in-composite tonemap they instead use the lean empty=1 variant.
            if MODE3:
                cond = c_on["composite_calls"] < c_off["composite_calls"]
            else:
                cond = c_on["empty_composite"] > 0 and c_off["empty_composite"] == 0
            if not cond:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: composite empty-skip "
                    f"on={c_on['composite_calls']}/{c_on['empty_composite']}"
                    f" off={c_off['composite_calls']}/"
                    f"{c_off['empty_composite']}"
                )
                ok = False
            # Covered-shade must never engage with the fast path off.
            if c_off["covered_shade"] != 0:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: covered_shade leaked "
                    f"with fast path off ({c_off['covered_shade']})"
                )
                ok = False
        if cfg in COMPACTED:
            # A real object over empty regions must resolve compacted, over
            # strictly fewer than all tile pixels on at least one launch, and
            # the composite must compact over the same covered list.
            if not (c_on["covered_lt_total"] > 0):
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: covered_shade not "
                    f"compacted (on={c_on['covered_shade']}/"
                    f"{c_on['covered_lt_total']})"
                )
                ok = False
            if MODE3 and not (c_on["composite_covered"] > 0):
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: composite not compacted "
                    f"(composite_covered={c_on['composite_covered']})"
                )
                ok = False
        if cfg == "env":
            if c_on["first_shade"] != c_off["first_shade"]:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: env must not skip "
                    f"resolves ({c_on['first_shade']} vs "
                    f"{c_off['first_shade']})"
                )
                ok = False
            # Env active disables the whole-tile skip AND covered-shade, so
            # no compaction must engage (empty pixels sampled the env map),
            # and the composite runs full for every tile.
            if c_on["composite_covered"] != 0 or c_on["empty_composite"] != 0:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: env engaged composite "
                    f"compaction (cov={c_on['composite_covered']} "
                    f"empty={c_on['empty_composite']})"
                )
                ok = False
            if c_on["covered_shade"] != 0:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: env engaged covered "
                    f"shade ({c_on['covered_shade']})"
                )
                ok = False
            if c_on["composite_calls"] != c_off["composite_calls"]:
                print(
                    f"[{cfg:9s}] ENGAGEMENT FAIL: env composite calls "
                    f"differ ({c_on['composite_calls']} vs "
                    f"{c_off['composite_calls']})"
                )
                ok = False
        all_ok = all_ok and ok
        print(
            f"[{cfg:9s}] frames={len(fa):3d}  max|d|={worst}  "
            f"{'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
    print("\nEMPTY_SKIP_PARITY_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
