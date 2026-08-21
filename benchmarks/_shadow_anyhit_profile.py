"""Kernel-level profile of one anyhit A/B arm on the small opaque scene.

Run once per arm (the env var decides), then compare the reports'
raster_shadow_trace device/launch numbers:

    ALGAN_SHADOW_ANYHIT=0 .venv/Scripts/python.exe benchmarks/_shadow_anyhit_profile.py off
    ALGAN_SHADOW_ANYHIT=1 .venv/Scripts/python.exe benchmarks/_shadow_anyhit_profile.py on
    ALGAN_SHADOW_ANYHIT=gather .venv/Scripts/python.exe benchmarks/_shadow_anyhit_profile.py gather
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shadow_anyhit_ab import build_scene  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "arm"
    SETTINGS.computing.set(available_memory_override=2_400_000_000)
    profile_scene(
        lambda: build_scene(mixed=False),
        PREVIEW,
        tag=f"_anyhit_{tag}",
        telemetry=False,
        nvprof=False,
    )


if __name__ == "__main__":
    main()
