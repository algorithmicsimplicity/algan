"""Kernel-level profile of one ALGAN_FRAG_PID_GATE arm.

Wall-clock A/B (``_frag_pid_gate_ab.py``) says whether a render got faster;
this says *where*, by reading the shade kernels' device time out of the
profiler report -- the gate can only move ``raster_first_shade`` /
``wavefront_shade``, so anything else moving is noise.

Run once per arm and diff the reports:

    .venv/Scripts/python.exe benchmarks/_frag_pid_gate_profile.py off raster
    ALGAN_FRAG_PID_GATE=1 .venv/Scripts/python.exe benchmarks/_frag_pid_gate_profile.py on raster

The second argument picks the scene shape: ``raster`` (hybrid raster
front-end, shading in ``raster_first_shade``), ``wavefront`` (front-end off,
shading in ``wavefront_shade``), each with the ``_frag_pid_gate_ab`` solo
material set; append ``_mixed`` for the three-material variant. The third
picks the quality preset (default MD -- at PREVIEW the shade kernel is too
small a share of the render to read).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _frag_pid_gate_ab import build_scene  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "arm"
    shape = sys.argv[2] if len(sys.argv) > 2 else "raster"
    quality = globals()[sys.argv[3]] if len(sys.argv) > 3 else MD
    mixed = shape.endswith("_mixed")
    raster = not shape.startswith("wavefront")
    SETTINGS.computing.set(available_memory_override=2_400_000_000)
    SETTINGS.raytracing.experimental.set(
        fragment_shading=True, hybrid_raster=bool(raster)
    )
    profile_scene(
        lambda: build_scene(mixed=mixed),
        quality,
        tag=f"_pidgate_{shape}_{tag}",
        telemetry=False,
        nvprof=False,
    )


if __name__ == "__main__":
    main()
