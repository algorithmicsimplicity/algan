"""Per-pixel run census for the C.3 fade-out investigation.

Renders shapes_and_timeline once (arm OFF -- emission is identical in both
arms, verified by identical per-batch fragment counts) and records, for every
sparse batch, every run of the host segmentation: its pixel, its length, and
whether its fragments are triangles. Saved to an npz per batch for offline
joining against the moved-pixel masks of the A/B videos.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

FULL_RENDERS = REPO / "tests" / "full_renders"
SCENE = FULL_RENDERS / "scenes" / "shapes_and_timeline.py"
OUT = FULL_RENDERS / "algan_outputs" / "_c3_ab"

RECORDS = []


def _census(coverage, merged, time_start, width, height):
    n = int(coverage["num_fragments"])
    if n == 0:
        return None
    counts = (coverage["run_offsets"][1:] - coverage["run_offsets"][:-1]).to(
        torch.int64
    )
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
    run_pix = pix[starts]
    run_tri = (ref >= 0)[starts]
    # Per-pixel fragment count as well, for the walk-length question.
    return {
        "run_pix": run_pix.cpu().numpy(),
        "run_len": lens.cpu().numpy().astype(np.int32),
        "run_tri": run_tri.cpu().numpy(),
        "cov_pix": coverage["covered_idx"].cpu().numpy(),
        "cov_nfrag": counts.cpu().numpy().astype(np.int32),
    }


class _Spy:
    def __enter__(self):
        self._sparse = rp.prepare_sparse_raster_coverage
        spy = self

        def sparse(merged, tri_screen, tri_bounds, bez_bounds, memory,
                   cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                   pixel_world_scale, col_row_arr, time_start, time_end,
                   width, height, half_w, half_h, layer_offset_triangles):
            cov = spy._sparse(
                merged, tri_screen, tri_bounds, bez_bounds, memory,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                pixel_world_scale, col_row_arr, time_start, time_end,
                width, height, half_w, half_h, layer_offset_triangles)
            data = None
            if cov is not None:
                data = _census(cov, merged, time_start, width, height)
            RECORDS.append(
                (int(time_start), int(time_end), int(width), int(height), data))
            return cov

        rp.prepare_sparse_raster_coverage = sparse
        return self

    def __exit__(self, *exc):
        rp.prepare_sparse_raster_coverage = self._sparse
        return False


def main():
    snapshot = SETTINGS.snapshot()
    cwd = os.getcwd()
    conftest = FULL_RENDERS.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_c3_census_conftest", conftest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(FULL_RENDERS)
    SETTINGS.paths.set(
        output_root=str(FULL_RENDERS),
        output_directory=str(OUT.relative_to(FULL_RENDERS)),
        cache_directory=str(FULL_RENDERS / "algan_cache"),
    )
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    SceneManager.reset()
    try:
        with Scene() as scene:
            name = f"_c3_census_{SCENE.stem}"
            spec = importlib.util.spec_from_file_location(name, SCENE)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
            with _Spy():
                scene.save_video(
                    str(OUT / "shapes_census.mp4"),
                    video_settings=PREVIEW,
                    overwrite=True,
                    animate_fade_out=True,
                )
    finally:
        os.chdir(cwd)
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    payload = {}
    for i, (t0, t1, w, h, data) in enumerate(RECORDS):
        payload[f"b{i:03d}_meta"] = np.array([t0, t1, w, h])
        if data is not None:
            for k, v in data.items():
                payload[f"b{i:03d}_{k}"] = v
    np.savez_compressed(OUT / "run_census.npz", **payload)
    print(f"saved {len(RECORDS)} batches to {OUT / 'run_census.npz'}")


if __name__ == "__main__":
    main()
