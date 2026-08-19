"""Golden-walk dump of one pixel under one C.3 arm (see _c3_fadeout_ab.py)."""

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

FULL = REPO / "tests" / "full_renders"
OUT = FULL / "algan_outputs" / "_c3_ab"


def main():
    run_exact = sys.argv[1] == "on"
    conftest = FULL.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_cft", conftest)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rt_settings.set_analytic_aa(True, run_exact=run_exact)
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
            str(OUT / f"shapes_dump_{sys.argv[1]}.mp4"),
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
        )


if __name__ == "__main__":
    main()
