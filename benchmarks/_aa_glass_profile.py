"""Where does analytic AA's time go on a REFRACTIVE scene?

``_aa_match_aa2.py`` shows the glass config costing analytic AA about 1.8x what
anti_alias_level=2 costs, which should not happen: analytic renders a quarter of
the primaries and spawns the same number of secondary rays per output pixel. And
it is not the secondary sampling -- the cost is there at one continuation per
pixel too. So this prints the per-kernel device breakdown of both arms.

Run: .venv/Scripts/python.exe benchmarks/_aa_glass_profile.py [analytic|aa2]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")
os.environ.setdefault("ALGAN_PROFILE_RUNS", "2")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    GREEN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    YELLOW,
    Off,
    RenderSettings,
    Sphere,
    Square,
)
from algan.rendering.raytracing import set_fragment_shading  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshPhysicalMaterial,
)
from algan.utils.profiling_utils import profile_scene  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "analytic"


def glass():
    set_fragment_shading(True)
    with Off():
        for i in range(3):
            bar = Square(color=(YELLOW, GREEN, BLUE)[i]).scale(0.5)
            bar.rotate(25 * i - 25, OUT)
            bar.move(UP * (0.9 - 0.9 * i) + LEFT * (0.9 - 0.9 * i) - OUT * 2.0)
            bar.spawn()
        g = Sphere().scale(1.2)
        g.set_material(MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5))
        g.spawn()
    g.move(RIGHT * 0.3)


if __name__ == "__main__":
    aa = 1
    if ARM == "analytic":
        rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    else:
        aa = 2
    print(f"arm: {ARM} (anti_alias_level={aa})")
    profile_scene(
        glass, RenderSettings((640, 360), 8, anti_alias_level=aa), f"aa_glass_{ARM}"
    )
