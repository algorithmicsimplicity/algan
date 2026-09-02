"""Stage profile of the PN-heavy geometry fixture (``_pn_geometry_scene.py``).

Drives ``profile_scene`` on the same recording the torch.compile A/B uses, so
the stage breakdown and the A/B wall times describe one scene.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runpy

from algan import PREVIEW  # noqa: F401
from algan.utils.profiling_utils import profile_scene

SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_pn_geometry_scene.py"
)


def build_scene():
    runpy.run_path(SCENE, run_name="_pn_geometry_scene")


if __name__ == "__main__":
    profile_scene(build_scene, PREVIEW, tag="_pn_geometry")
