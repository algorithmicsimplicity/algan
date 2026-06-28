import torch.nn.functional as F

from algan.constants.spatial import *  # RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.rendering.raytracing import enable_ray_tracing
from algan.settings.render_settings import UHD
from algan.utils.algan_utils import render_all_funcs


get_mob = (
    lambda r=0: TriangleTriangulated(
        torch.stack(
            (
                UP * 0.5,
                F.normalize(RIGHT + DOWN, p=2, dim=-1) * 0.5,
                F.normalize(LEFT + DOWN, p=2, dim=-1) * 0.5,
            )
        ),
        color=torch.stack([PURE_RED for _ in range(3)]),
    )
    .spawn()
)


def test_glow():
    x = get_mob()
    x.glow = 100.0
    x.glow_radius = 100
    #x.glow_radius = 0.1
    x.wait()
    return


#enable_ray_tracing(1, fragment_shading=True)
render_all_funcs(__name__, start_index=0, max_rendered=-1)
