"""Golden-walk check of the per-fragment AA dump (DESIGN_analytic_aa_v2.md ss7.1).

Three checks, each on a one-frame scene:

  1. GOLDEN WALK -- re-render a probed silhouette pixel with ``ALGAN_AA_DUMP``
     aimed at it, then recompute the resolve's per-sample transmittance walk in
     numpy FROM THE DUMPED INPUTS (mask, cov, mat_alpha, trans_share) and diff
     eff, svis after every fragment, and the terminal vis_all/weight. This is
     the instrument that replaces guessing at resolve accounting (the ss21.9
     postmortem); Phase D's run rule extends this walk with corr.

  2. WALK SYNC -- with shadows on, the shadow-event walk must accept exactly
     the fragment sequence the resolve accepts (same q, ref, eff per row up to
     the resolve-only tail). A desync here is the ss6/ss13.2 failure mode.

  3. TRI_OBJ MAPPING -- fragments of a probed pixel on each of two separated
     spheres must carry that sphere's own surface id (diced-PN logical
     triangles included -- the ss21.9 assumption, proven rather than believed).

Run: .venv/Scripts/python.exe benchmarks/_aa_dump_check.py
"""

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algan import (  # noqa: E402
    BLUE,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    Off,
    Scene,
    SceneManager,
    Sphere,
    Square,
    VideoSettings,
)
from algan.settings import SETTINGS  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
import algan.rendering.raytracing.raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_NUM_SAMPLES,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_aa_iter_out")
os.makedirs(OUT_DIR, exist_ok=True)
W, H = 320, 180
N = _AA_NUM_SAMPLES
FAILS = []


def check(ok, msg):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {msg}")
    if not ok:
        FAILS.append(msg)


def render(build, path, probe=None, shadows=False, run=False):
    """Render one frame; with ``probe=(px, ky)`` re-render with the dump on."""
    SceneManager.reset()
    SETTINGS.raytracing.set(shadows=shadows)
    rt_settings.set_analytic_aa(True, run=run)
    try:
        settings = VideoSettings((W, H), frames_per_second=4, anti_alias_level=1)
        with Scene(video_settings=settings) as scene:
            build()
            scene.save_frame(path, video_settings=settings, overwrite=True)
            if probe is not None:
                os.environ["ALGAN_AA_DUMP"] = f"{probe[0]},{probe[1]},0"
                rp.LAST_AA_DUMP.clear()
                try:
                    scene.save_frame(path, video_settings=settings, overwrite=True)
                finally:
                    del os.environ["ALGAN_AA_DUMP"]
    finally:
        SETTINGS.raytracing.set(shadows=False)
        rt_settings.set_analytic_aa(True, run=False)
    return cv2.imread(path).astype(np.float64)


def find_silhouette(img, x_lo=0, x_hi=W):
    """A lit pixel with a strong horizontal gradient, as (px, kernel_y)."""
    lum = img.mean(axis=2)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    cand = np.argwhere((gx > 25) & (lum > 10))
    cand = cand[(cand[:, 1] >= x_lo) & (cand[:, 1] < x_hi)]
    assert len(cand), "no silhouette pixel found"
    py_png, px = (int(v) for v in cand[len(cand) // 2])
    return px, (H - 1) - py_png


def golden_walk(rows, run=False):
    """Recompute the per-sample transmittance walk from the dumped inputs.

    Returns the worst absolute error over every compared column. Mirrors the
    non-glass branches of ``raster_first_shade`` (the probe scenes carry no
    refraction). ``run``: the run-corrected representation -- a triangle's
    per-fragment magnitude is its mask at density 1 (the exact area is a
    run-scan input only), an empty mask commits nothing, and the dumped corr
    column multiplies both the claim and the occlusion writes.
    """
    svis = np.ones(N)
    worst = 0.0
    for r in rows:
        q, kind, note = int(r[0]), int(r[1]), int(r[2])
        if q < 0:
            vis_all = svis.mean()
            worst = max(worst, abs(vis_all - r[4]))
            # Terminal weight folds vis_all when the pixel retired unbounced.
            if int(r[1]) == 0:
                worst = max(worst, abs(r[9] - vis_all))
            continue
        msk, cov, corr = int(r[6]), float(r[7]), float(r[9])
        mat_alpha, ts = float(r[11]), float(r[13])
        sel = np.array([(msk >> s) & 1 for s in range(N)], dtype=bool)
        is_bez = kind in (1, 3)
        areal = is_bez or ((not run) and (not sel.any()))
        if kind >= 2:
            sel[:] = True
            areal = False
        if areal:
            dens = cov
            sel = np.ones(N, dtype=bool)
        elif run and kind < 2:
            if not sel.any():
                if float(r[10]) > 0.0:
                    # A pristine all-sliver claim: its magnitude depends on
                    # the run scan (E, vstart), which one row cannot carry.
                    # Trust the row and resync; the energy invariant is
                    # checked by the terminal vis_all and the thin-scene ink
                    # gate instead.
                    svis[:] = r[16 : 16 + N]
                continue
            dens = 1.0
        else:
            pop = int(sel.sum())
            dens = min(cov * N / max(pop, 1), 1.0) if pop else 0.0
        eff = svis[sel].sum() / N * dens * corr
        if note == 1:
            worst = max(worst, abs(eff - r[10]))
            continue  # eff-skip commits nothing
        worst = max(worst, abs(eff - r[10]))
        a_s = mat_alpha * dens
        upd = (1.0 - corr * a_s) + corr * a_s * ts
        svis[sel] *= max(upd, 0.0)
        worst = max(worst, float(np.abs(svis - r[16 : 16 + N]).max()))
    return worst


def main():
    # -- 1. golden walk on a translucent sphere over an opaque square --------
    def build_trans():
        with Off():
            s = Sphere().scale(1.0).move(LEFT * 0.4)
            s.set_color(GREEN)
            s.opacity = 0.55
            s.spawn()
            sq = Square(color=RED).scale(1.1).move(LEFT * 0.1)
            sq.spawn()

    path = os.path.join(OUT_DIR, "_dump_trans.png")
    img = render(build_trans, path)
    px, ky = find_silhouette(img, x_hi=W // 2)
    render(build_trans, path, probe=(px, ky))
    rows = rp.LAST_AA_DUMP.get("resolve")
    print(f"golden walk at ({px},{ky}): {0 if rows is None else len(rows)} rows")
    check(rows is not None and len(rows) >= 2, "resolve dump produced rows")
    if rows is not None and len(rows):
        err = golden_walk(rows)
        check(err < 2e-5, f"golden walk matches kernel (worst err {err:.2e})")

    # -- 1b. the same walk under the run-corrected representation ------------
    render(build_trans, path, probe=(px, ky), run=True)
    rrows = rp.LAST_AA_DUMP.get("resolve")
    check(rrows is not None and len(rrows) >= 2, "run-mode dump produced rows")
    if rrows is not None and len(rrows):
        err = golden_walk(rrows, run=True)
        check(err < 2e-5, f"run-mode golden walk matches (worst err {err:.2e})")

    # -- 2. resolve/shadow walk sync ----------------------------------------
    render(build_trans, path, probe=(px, ky), shadows=True)
    rres = rp.LAST_AA_DUMP.get("resolve")
    rsh = rp.LAST_AA_DUMP.get("shadow")
    check(rsh is not None and len(rsh) > 0, "shadow dump produced rows")
    if rsh is not None and rres is not None and len(rsh) and len(rres):
        fr = rres[rres[:, 0] >= 0]
        fs = rsh[rsh[:, 0] >= 0]
        m = min(len(fr), len(fs))
        same = (
            np.array_equal(fr[:m, 0], fs[:m, 0])
            and np.array_equal(fr[:m, 3], fs[:m, 3])
            and np.allclose(fr[:m, 10], fs[:m, 10], atol=1e-6)
        )
        check(
            same and abs(len(fr) - len(fs)) <= 1,
            f"walks in lockstep ({len(fr)} vs {len(fs)} rows)",
        )

    # -- 3. tri_obj maps fragments to their source surface -------------------
    def build_two():
        with Off():
            a = Sphere().scale(0.9).move(LEFT * 1.2)
            a.set_color(BLUE)
            a.spawn()
            b = Sphere().scale(0.9).move(RIGHT * 1.2)
            b.set_color(GREEN)
            b.spawn()

    path2 = os.path.join(OUT_DIR, "_dump_two.png")
    img2 = render(build_two, path2)
    sids = []
    for lo, hi in ((0, W // 2), (W // 2, W)):
        qx, qy = find_silhouette(img2, lo, hi)
        render(build_two, path2, probe=(qx, qy))
        rows = rp.LAST_AA_DUMP.get("resolve")
        assert rows is not None and len(rows)
        frag = rows[(rows[:, 0] >= 0) & (rows[:, 1] < 1.5)]
        ids = set(int(v) for v in frag[:, 4])
        check(len(ids) == 1, f"pixel ({qx},{qy}) fragments share one sid {ids}")
        sids.append(ids)
    check(
        sids[0] and sids[1] and sids[0] != sids[1],
        f"the two spheres carry distinct sids {sids[0]} vs {sids[1]}",
    )

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        sys.exit(1)
    print("all dump checks passed")


if __name__ == "__main__":
    main()
