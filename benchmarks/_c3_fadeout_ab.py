"""Does ss6.7's arm still move shapes_and_timeline's fade-out at HEAD?

DESIGN_mesh_identity_open.md ssC.3 blocks the default flip on one unattributed
number: "shapes_and_timeline still moves by 31 channel values over 4,514 pixels
with the arm confined and the ss6.7.1 dense-path bug fixed". But
raster_pipeline.py's ``_aa_group_dense`` docstring attributes the SAME
31/4,514 move to the pre-fix out-of-bounds lane read, and by construction the
fixed dense path compiles identically (aa_grp 5, sentinel cap) whether
ALGAN_ANALYTIC_AA_RUN_EXACT is on or off. Those two claims cannot both be
right, so this script measures which one describes HEAD:

  arm OFF  -> render, diff against the committed CUDA baseline (sanity: this
              box owns that baseline; anything but ~0 means stop and think)
  arm ON   -> render, diff against the OFF arm frame by frame

While rendering it logs, per batch and in call order:
  * which path the batch took (sparse prepare vs dense raster_iteration_zero),
    with the batch's (time_start, time_end) -- time_start RESTARTS per render
    segment (ssA), so the tail of the call sequence is the fade-out segment,
    not any particular time_start value;
  * for sparse batches, how many host-reduction runs are >= 17 fragments long
    (the population the 16-budget scan cannot finish) and how many covered
    pixel-frames hold one -- the count the notch probe reports, recomputed
    here so dense-path blindness is explicit in the log rather than silent.

Run:  <venv-python> benchmarks/_c3_fadeout_ab.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

FULL_RENDERS = REPO / "tests" / "full_renders"
SCENE = FULL_RENDERS / "scenes" / "shapes_and_timeline.py"
OUT = FULL_RENDERS / "algan_outputs" / "_c3_ab"

#: (kind, time_start, time_end_or_None, n_frags, n_trunc_runs, n_trunc_px)
BATCH_LOG: list[tuple] = []


def _register_fonts():
    conftest = FULL_RENDERS.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_c3_ab_conftest", conftest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _sparse_trunc_stats(coverage, merged, time_start, width, height):
    """Replicate the host reduction's run segmentation over the compact CSR
    and count the runs a 16-fragment scan budget cannot finish.
    """
    counts = (coverage["run_offsets"][1:] - coverage["run_offsets"][:-1]).to(
        torch.int64
    )
    n = int(coverage["num_fragments"])
    if n == 0:
        return 0, 0
    device = counts.device
    pix = torch.repeat_interleave(coverage["covered_idx"].to(torch.int64), counts)
    ref = coverage["frag_ref"].to(torch.int64)
    msk = coverage["frag_msk"]
    tri_obj = merged["tri_obj"]
    ppf = int(width) * int(height)
    row = rp._tri_obj_row(pix, ppf, int(time_start), tri_obj.shape[0])
    sid = tri_obj[row, ref.clamp_min(0)].to(torch.int64)
    face = ((msk & rp.AA_BACKFACE_BIT) != 0).to(torch.int64)
    idx = torch.arange(n, dtype=torch.int64, device=device)
    key = torch.where(ref >= 0, sid * 2 + face, -(idx + 2))
    starts = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        starts[1:] = (key[1:] != key[:-1]) | (pix[1:] != pix[:-1])
    run_id = torch.cumsum(starts.to(torch.int64), 0) - 1
    lens = torch.bincount(run_id)
    trunc = lens >= 17
    n_trunc = int(trunc.sum())
    if n_trunc == 0:
        return 0, 0
    run_pix = pix[starts]
    n_px = int(torch.unique(run_pix[trunc]).numel())
    return n_trunc, n_px


class _PathSpy:
    """Log which resolve path every batch takes, without changing either."""

    def __enter__(self):
        self._sparse = rp.prepare_sparse_raster_coverage
        self._dense = rp.raster_iteration_zero
        spy = self

        def sparse(
            merged,
            tri_screen,
            tri_bounds,
            bez_bounds,
            memory,
            cam_origin,
            screen_point,
            pixel_basis_x,
            pixel_basis_y,
            pixel_world_scale,
            col_row_arr,
            time_start,
            time_end,
            width,
            height,
            half_w,
            half_h,
            layer_offset_triangles,
        ):
            cov = spy._sparse(
                merged,
                tri_screen,
                tri_bounds,
                bez_bounds,
                memory,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                pixel_world_scale,
                col_row_arr,
                time_start,
                time_end,
                width,
                height,
                half_w,
                half_h,
                layer_offset_triangles,
            )
            if cov is None:
                BATCH_LOG.append(
                    ("sparse-empty", int(time_start), int(time_end), 0, 0, 0)
                )
            else:
                n_tr, n_px = _sparse_trunc_stats(cov, merged, time_start, width, height)
                BATCH_LOG.append(
                    (
                        "sparse",
                        int(time_start),
                        int(time_end),
                        int(cov["num_fragments"]),
                        n_tr,
                        n_px,
                    )
                )
            return cov

        def dense(*args, **kwargs):
            # time_start is positional argument 21 of raster_iteration_zero.
            BATCH_LOG.append(("dense", int(args[21]), None, -1, -1, -1))
            return spy._dense(*args, **kwargs)

        rp.prepare_sparse_raster_coverage = sparse
        rp.raster_iteration_zero = dense
        return self

    def __exit__(self, *exc):
        rp.prepare_sparse_raster_coverage = self._sparse
        rp.raster_iteration_zero = self._dense
        return False


def _render(run_exact, out_name):
    BATCH_LOG.clear()
    rt_settings.set_analytic_aa(True, run_exact=run_exact)
    snapshot = SETTINGS.snapshot()
    cwd = os.getcwd()
    OUT.mkdir(parents=True, exist_ok=True)
    (FULL_RENDERS / "algan_cache").mkdir(parents=True, exist_ok=True)
    os.chdir(FULL_RENDERS)
    SETTINGS.paths.set(
        output_root=str(FULL_RENDERS),
        output_directory=str(OUT.relative_to(FULL_RENDERS)),
        cache_directory=str(FULL_RENDERS / "algan_cache"),
    )
    # Same pin the render suite and the notch probe use, so the frame-window
    # split is reproducible run to run.
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    SceneManager.reset()
    try:
        with Scene() as scene:
            name = f"_c3_ab_{SCENE.stem}"
            spec = importlib.util.spec_from_file_location(name, SCENE)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
            with _PathSpy():
                scene.save_video(
                    str(OUT / out_name),
                    video_settings=PREVIEW,
                    overwrite=True,
                    animate_fade_out=True,
                )
    finally:
        os.chdir(cwd)
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    print(f"\n== batches, arm {'ON' if run_exact else 'OFF'} (call order) ==")
    total_tr = total_px = 0
    dense_starts = []
    for kind, t0, t1, n_frags, n_tr, n_px in BATCH_LOG:
        if kind == "dense":
            dense_starts.append(t0)
            continue
        note = f" trunc_runs={n_tr} trunc_px={n_px}" if n_tr else ""
        print(f"  {kind:12s} t=[{t0},{t1})  frags={n_frags}{note}")
        total_tr += max(n_tr, 0)
        total_px += max(n_px, 0)
    if dense_starts:
        print(
            f"  dense raster_iteration_zero calls: {len(dense_starts)} "
            f"(time_starts {sorted(set(dense_starts))})"
        )
    print(f"  sparse totals: trunc_runs={total_tr} trunc_px={total_px}")
    return OUT / out_name


def _diff(path_a, path_b, label):
    import cv2
    import numpy as np

    a, b = cv2.VideoCapture(str(path_a)), cv2.VideoCapture(str(path_b))
    worst = worst_frame = frames = total_moved = 0
    per_frame = []
    while True:
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                print(f"{label}: FRAME COUNT MISMATCH at {frames}")
            break
        delta = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        d = int(delta.max())
        moved = int((delta.max(axis=2) > 2).sum())
        if d > worst:
            worst, worst_frame = d, frames
        if moved:
            per_frame.append((frames, d, moved))
        total_moved += moved
        frames += 1
    a.release()
    b.release()
    print(f"\n== {label} ==")
    if not per_frame:
        print(f"  byte-identical over {frames} frames (worst |d| {worst})")
        return
    print(
        f"  worst |d| {worst} at frame {worst_frame}; "
        f"{total_moved} moved pixel-frames over {len(per_frame)} frames "
        f"of {frames}"
    )
    for f, d, m in per_frame:
        print(f"    frame {f:4d}  max|d| {d:3d}  moved px {m}")


def main():
    print(f"cuda={torch.cuda.is_available()}")
    _register_fonts()
    off = _render(False, "shapes_off.mp4")
    baseline = FULL_RENDERS / "expected_outputs_cuda" / "shapes_and_timeline.mp4"
    _diff(off, baseline, "arm OFF vs committed CUDA baseline (sanity)")
    on = _render(True, "shapes_on.mp4")
    _diff(on, off, "arm ON vs arm OFF")


if __name__ == "__main__":
    main()
