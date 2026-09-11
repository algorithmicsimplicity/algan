"""Do the opt-in tiled frontend's proofs certify anything on a given scene?

    python benchmarks/performance/tile_proof_sweep.py [scene ...]

Renders each ``tests/full_renders/scenes/*.py`` with ``raster_tile_binning``
and ``raster_simple_interiors`` on, and reports how far each proof got: how
many candidates could be occluders at all, how many were certified, how many
rejections that bought, and the same staged breakdown for the simple-interior
classifier.

This exists because a neutral A/B timing and an inert feature look identical.
The counters here are scene-deterministic and device-independent -- they are
integer outcomes of the proofs, not timings -- so CPU is a valid place to take
them, and they answer "is there anything to optimise here" without a GPU.

Mirrors ``tests/full_renders/test_full_renders.py``'s settings contract (the
pinned frame-window split and eager torch), so the geometry and the window
split are the ones that suite renders. Results:
``benchmarks/performance/reports/t4_2026_09/tiled_primary_ab_5.md``.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import tile_raster_taichi as kern  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_MASK_ALL,
    _AA_SLIVER_BIT,
)
from algan.scene_manager import SceneManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "tests" / "full_renders"
SCENES = sorted(p for p in (HERE / "scenes").glob("*.py") if not p.name.startswith("_"))
#: The full-render suite's pinned split, so a scene bins the same way it does
#: there -- free VRAM is not reproducible and a different split reorders the
#: merged arrays.
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024

T = {}


def bump(key, value=1):
    T[key] = T.get(key, 0) + int(value)


_real = {
    name: getattr(kern, name)
    for name in (
        "candidate_proofs",
        "opaque_material_proofs",
        "interior_fragment_proofs",
        "interior_pixels",
    )
}


def candidate_proofs(candidates, intervals, flags, screen, pos, cam, opaque, n, w, h, ts, cull):
    _real["candidate_proofs"](
        candidates, intervals, flags, screen, pos, cam, opaque, n, w, h, ts, cull
    )
    tri = candidates[:n, 6] == 1
    bump("cand", n)
    bump("cand_tri", int(tri.sum()))
    bump("cand_opaque_class", int((tri & (candidates[:n, 7] != 0)).sum()))
    bump("cand_outside", int((flags[:n] == 1).sum()))
    bump("cand_full_opaque", int((flags[:n] == 2).sum()))
    # Filling its tile is the precondition for an occluder certificate, tested
    # before any interval arithmetic runs. Counting it separates "could not be
    # an occluder" from "the proof failed".
    x0 = candidates[:n, 2] // kern.FINE_TILE * kern.FINE_TILE
    y0 = candidates[:n, 3] // kern.FINE_TILE * kern.FINE_TILE
    fills = (
        (candidates[:n, 2] == x0)
        & (candidates[:n, 3] == y0)
        & (candidates[:n, 4] == torch.clamp(x0 + kern.FINE_TILE, max=w) - 1)
        & (candidates[:n, 5] == torch.clamp(y0 + kern.FINE_TILE, max=h) - 1)
    )
    bump("cand_fills_tile", int((tri & fills).sum()))


def opaque_material_proofs(*a):
    _real["opaque_material_proofs"](*a)
    bump("mat_proven", int(a[6].sum()))
    bump("mat_total", a[6].numel())


def interior_fragment_proofs(keys, refs, cov, masks, screen, pos, camera, objects,
                             unc, closed, eligible, distances, surface, n, t0, w, h):
    _real["interior_fragment_proofs"](
        keys, refs, cov, masks, screen, pos, camera, objects, unc, closed,
        eligible, distances, surface, n, t0, w, h,
    )
    k = slice(0, n)
    is_tri = refs[k] >= 0
    cov1 = cov[k] == 1.0
    full = (masks[k] & _AA_MASK_ALL) == _AA_MASK_ALL
    nosliv = (masks[k] & _AA_SLIVER_BIT) == 0
    bump("frag", n)
    bump("frag_tri", int(is_tri.sum()))
    bump("frag_cov1", int(cov1.sum()))
    bump("frag_fullmask", int(full.sum()))
    bump("frag_pre", int((is_tri & cov1 & full & nosliv).sum()))
    bump("frag_eligible", int(eligible[k].sum()))


def interior_pixels(offsets, eligible, distances, surface, scratch, simple, npixels):
    _real["interior_pixels"](
        offsets, eligible, distances, surface, scratch, simple, npixels
    )
    bump("px", npixels)
    bump("px_simple", int(simple[:npixels].sum()))


kern.candidate_proofs = candidate_proofs
kern.opaque_material_proofs = opaque_material_proofs
kern.interior_fragment_proofs = interior_fragment_proofs
kern.interior_pixels = interior_pixels

import algan.rendering.raytracing.raster_pipeline as rp  # noqa: E402

_inner = rp.prepare_sparse_raster_coverage


def counted(merged, *a, **k):
    for name, label in (
        ("tri_frame_opaque", "geo_opaque"),
        ("tri_closed", "geo_closed"),
        ("tri_alpha_uncertain", "geo_uncertain"),
    ):
        v = merged.get(name)
        if v is not None:
            bump(label, float(v.to(float).sum()))
            bump(label + "_of", v.numel())
    bump("geo_tri", int(merged.get("num_triangles", 0)))
    bump("geo_circ", int(merged.get("num_circuits", 0)))
    bump("windows")
    out = _inner(merged, *a, **k)
    if isinstance(out, dict):
        for key in (
            "num_fragments", "num_covered", "num_sheets", "num_simple_pixels",
            "tile_candidates", "tile_bbox_rejected", "tile_occluded",
        ):
            if key in out:
                bump(key, out[key])
    return out


rp.prepare_sparse_raster_coverage = counted


def render(path):
    T.clear()
    snapshot = SETTINGS.snapshot()
    os.chdir(HERE)
    SETTINGS.paths.set(
        output_root=str(HERE),
        output_directory="algan_outputs",
        cache_directory=str(HERE / "algan_cache"),
    )
    SETTINGS.computing.set(
        available_memory_override=AVAILABLE_MEMORY_OVERRIDE, torch_compile=False
    )
    SETTINGS.raytracing.experimental.set(
        raster_tile_binning=True, raster_simple_interiors=True
    )
    SceneManager.reset()
    try:
        with Scene() as scene:
            name = f"_sweep_{path.stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
            scene.save_video(
                HERE / "algan_outputs" / f"sweep_{path.stem}.mp4",
                video_settings=PREVIEW,
                overwrite=True,
                animate_fade_out=True,
                codec="libx264rgb",
            )
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()
        os.chdir(ROOT)
    return dict(T)


def report(stem, t):
    def pct(a, b):
        return f"{100.0 * t.get(a, 0) / max(t.get(b, 0), 1):5.1f}%"

    print(f"\n### {stem}")
    print(f"  windows={t.get('windows', 0)} tri={t.get('geo_tri', 0)} "
          f"circ={t.get('geo_circ', 0)}")
    print(f"  geometry: frame_opaque {pct('geo_opaque', 'geo_opaque_of')}  "
          f"closed {pct('geo_closed', 'geo_closed_of')}  "
          f"alpha_uncertain {pct('geo_uncertain', 'geo_uncertain_of')}")
    print(f"  RANK 3: candidates={t.get('cand', 0)} tri={t.get('cand_tri', 0)} "
          f"opaque_class={t.get('cand_opaque_class', 0)} "
          f"fills_tile={t.get('cand_fills_tile', 0)} "
          f"material_proven={t.get('mat_proven', 0)}/{t.get('mat_total', 0)}")
    print(f"          outside={t.get('tile_bbox_rejected', 0)} "
          f"CERTIFIED_OCCLUDERS={t.get('cand_full_opaque', 0)} "
          f"OCCLUSION_REJECTED={t.get('tile_occluded', 0)}")
    print(f"  RANK 4: fragments={t.get('frag', 0)} tri={t.get('frag_tri', 0)} "
          f"cov==1={t.get('frag_cov1', 0)} fullmask={t.get('frag_fullmask', 0)} "
          f"pre_geometry={t.get('frag_pre', 0)} eligible={t.get('frag_eligible', 0)}")
    print(f"          pixels={t.get('px', 0)} SIMPLE_PIXELS={t.get('px_simple', 0)} "
          f"(coverage num_simple={t.get('num_simple_pixels', 0)} "
          f"of {t.get('num_covered', 0)})")


def main():
    wanted = sys.argv[1:] or [p.stem for p in SCENES]
    rows = {}
    for path in SCENES:
        if path.stem not in wanted:
            continue
        print(f"--- rendering {path.stem} ...", flush=True)
        rows[path.stem] = render(path)
        report(path.stem, rows[path.stem])
    print("\n\n===== SUMMARY =====")
    print(f"{'scene':28s} {'occluded':>10s} {'simple px':>10s} {'of covered':>12s}")
    for stem, t in rows.items():
        print(f"{stem:28s} {t.get('tile_occluded', 0):10d} "
              f"{t.get('num_simple_pixels', 0):10d} {t.get('num_covered', 0):12d}")


if __name__ == "__main__":
    main()
