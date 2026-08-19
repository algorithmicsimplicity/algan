"""Does the covered-slice partition alone move shapes_and_timeline's fade-out?

Arm OFF (shipped aa_grp 5, no exact-run lanes anywhere) rendered with a
different WAVEFRONT_TILE_RAYS, then diffed against the arm-OFF reference from
_c3_fadeout_ab.py. Slice boundaries are logged for both this render and for
reference. If the fade-out frames move here, the C.3 "arm" move is a
pre-existing slice-partition dependence that the lanes' 8 B/fragment arena
footprint merely exposes -- not anything the aa_grp 6 kernel computes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

FULL = REPO / "tests" / "full_renders"
OUT = FULL / "algan_outputs" / "_c3_ab"

SLICES = []


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    out_name = sys.argv[2] if len(sys.argv) > 2 else "shapes_off_tiles.mp4"
    conftest = FULL.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_cft", conftest)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    rt_settings.WAVEFRONT_TILE_RAYS = int(rt_settings.WAVEFRONT_TILE_RAYS * scale)
    print(f"WAVEFRONT_TILE_RAYS = {rt_settings.WAVEFRONT_TILE_RAYS}")

    real_shade = rp.shade_sparse_raster_coverage

    def shade(coverage, covered_start, covered_end, *args, **kwargs):
        SLICES.append((int(covered_start), int(covered_end)))
        return real_shade(coverage, covered_start, covered_end, *args, **kwargs)

    rp.shade_sparse_raster_coverage = shade

    os.chdir(FULL)
    SETTINGS.paths.set(
        output_root=str(FULL),
        output_directory="algan_outputs/_c3_ab",
        cache_directory=str(FULL / "algan_cache"),
    )
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    SceneManager.reset()
    with Scene() as scene:
        sp = importlib.util.spec_from_file_location(
            "_sc", FULL / "scenes" / "shapes_and_timeline.py"
        )
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        scene.save_video(
            str(OUT / out_name),
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
        )
    widths = sorted({b - a for a, b in SLICES})
    print(f"{len(SLICES)} shade slices; widths {widths[:8]}{'...' if len(widths) > 8 else ''}")
    print("last 12 slices:", SLICES[-12:])


if __name__ == "__main__":
    main()
