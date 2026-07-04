"""Validation for constant-property -> 1x1-texture promotion.

Renders through the deterministic fragment-shading general wavefront (the only
path promotion applies to) and checks:

  1. const_pixels  -- a constant-colour Surface promoted (default) matches the
                      same render with promotion disabled to within 2 LSB
                      (sampling a 1x1 map reduces to the stored constant; the
                      <=1 ULP slack is the barycentric sum not being exactly 1).
  2. const_shrink  -- with promotion the merged tri_colors / tri_extra no longer
                      span the constant mob's triangles (memory reclaimed), and
                      the shared texel buffer gained the mob's 1x1 maps.
  3. refl_const    -- a constant *reflective* Surface (mirror) still reflects:
                      promoted vs per-vertex match within 2 LSB (reflectivity now
                      read from the material map, not the dropped vertex row).
  4. grad_identity -- a per-vertex-gradient Surface cannot be promoted, so
                      promotion ON vs OFF is byte-identical (no rounding drift).
  5. multi_mob     -- a constant mob batched with a gradient mob: only the
                      constant one is promoted; output matches promotion OFF.

    .venv/Scripts/python.exe benchmarks/_promote_constants_check.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import Sync, SceneManager, RIGHT, LEFT, UP, OUT, DOWN, RED, GREEN, BLUE  # noqa: E402
from algan.mobs.surfaces.surface import Surface  # noqa: E402
from algan.mobs.shapes_2d import TriangleTriangulated  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from algan.rendering.raytracing import (  # noqa: E402
    enable_ray_tracing, set_reflectivity)
from algan.rendering.raytracing import primitives as P  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_pc_out")
os.makedirs(OUT_DIR, exist_ok=True)

results = []


def check(name, ok, detail):
    results.append((name, bool(ok)))
    print(f"[{name:14s}] {'OK' if ok else 'FAIL'}  {detail}")


# Capture the merged scene of the last render so shrink can be inspected.
_captured = {}
_orig_merge = P._merge_scene


def _spy_merge(primitives):
    scene = _orig_merge(primitives)
    _captured["scene"] = scene
    return scene


P._merge_scene = _spy_merge


def render(tag, build, promote):
    SceneManager.reset()
    P.set_promote_constants(promote)
    enable_ray_tracing(1, fragment_shading=True)
    P.set_wavefront(True)
    with Sync():
        build()
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"pc_{tag}_{'on' if promote else 'off'}.png")
    frames = scene.save_frame(out)
    img = frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0
    return img, _captured.get("scene")


def diff(a, b):
    return np.abs(a.astype(np.float64) - b.astype(np.float64))


def const_surface():
    Surface(grid_height=16, grid_width=16, color=GREEN).scale(2.0).move(
        OUT * 0.2).spawn()


def refl_surface():
    s = Surface(grid_height=16, grid_width=16, color=GREEN).scale(2.0)
    set_reflectivity(s, 0.7)
    s.spawn()


def _grad_tri(offset=OUT * 0.2, scale=1.6):
    # A single triangle with three different vertex colours -> genuinely
    # per-vertex non-constant (and no UVs) -> cannot be promoted.
    pts = torch.stack((UP * 0.5,
                       F.normalize(RIGHT + DOWN, p=2, dim=-1) * 0.5,
                       F.normalize(LEFT + DOWN, p=2, dim=-1) * 0.5))
    return TriangleTriangulated(
        pts, color=torch.stack([RED, GREEN, BLUE])).scale(scale).move(offset)


def grad_surface():
    _grad_tri().spawn()


def multi_mob():
    const_surface()
    _grad_tri(offset=RIGHT * 1.2 + UP * 0.4, scale=1.0).spawn()


def main():
    with torch.inference_mode():
        # 1 + 2: constant colour surface.
        img_off, sc_off = render("const", const_surface, promote=False)
        img_on, sc_on = render("const", const_surface, promote=True)
        d = diff(img_off, img_on).max()
        check("const_pixels", d <= 2.0,
              f"promoted vs per-vertex max|d|={d:.2f}")

        nver_off = sc_off["tri_colors"].shape[1]
        nver_on = sc_on["tri_colors"].shape[1]
        ntri = sc_on["num_triangles"]
        tex_off = sc_off["textures"].shape[1]
        tex_on = sc_on["textures"].shape[1]
        check("const_shrink",
              nver_on < nver_off and nver_on <= 1 and tex_on > tex_off,
              f"tri_colors rows {nver_off}->{nver_on} (of {ntri} tris); "
              f"texels {tex_off}->{tex_on}; tri_extra rows "
              f"{sc_off['tri_extra'].shape[1]}->{sc_on['tri_extra'].shape[1]}")

        # 3: constant reflective surface.
        r_off, _ = render("refl", refl_surface, promote=False)
        r_on, _ = render("refl", refl_surface, promote=True)
        d = diff(r_off, r_on).max()
        check("refl_const", d <= 2.0,
              f"reflective promoted vs per-vertex max|d|={d:.2f}")

        # 4: gradient surface cannot promote -> byte-identical.
        g_off, gsc_off = render("grad", grad_surface, promote=False)
        g_on, gsc_on = render("grad", grad_surface, promote=True)
        d = diff(g_off, g_on).max()
        same_rows = (gsc_off["tri_colors"].shape[1]
                     == gsc_on["tri_colors"].shape[1])
        check("grad_identity", d == 0.0 and same_rows,
              f"gradient (not promotable) max|d|={d:.2f}; "
              f"tri_colors rows unchanged={same_rows}")

        # 5: mixed batch.
        m_off, msc_off = render("multi", multi_mob, promote=False)
        m_on, msc_on = render("multi", multi_mob, promote=True)
        d = diff(m_off, m_on).max()
        shrank = (msc_on["tri_colors"].shape[1] < msc_off["tri_colors"].shape[1])
        check("multi_mob", d <= 2.0 and shrank,
              f"mixed batch max|d|={d:.2f}; only-constant shrank tri_colors "
              f"{msc_off['tri_colors'].shape[1]}->{msc_on['tri_colors'].shape[1]}")

        ok = all(r[1] for r in results)
        print("\nPROMOTE_CONSTANTS_OK:", ok)
        return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
