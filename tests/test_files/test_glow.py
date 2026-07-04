import torch.nn.functional as F

from algan.constants.spatial import *  # RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
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
        color=torch.stack([RED for _ in range(3)]),
    )
    .spawn()
)


def test_glow():
    #enable_ray_tracing(samples_per_pixel=1, tonemapping=False)
    x = get_mob()
    x.glow = 1.0
    x.wait()
    x.glow_radius = 1
    x.wait()
    return


def test_tonemapping():
    #enable_ray_tracing(samples_per_pixel=1, tonemapping=True, tonemap_exposure=1.5)
    x = get_mob()
    x.glow = 10
    x.glow_radius = 1.0
    x.wait()
    return


def test_post_process_tonemapping():
    #enable_ray_tracing(samples_per_pixel=1, tonemapping=True, tonemap_exposure=1.5, post_process_tonemap=True)
    x = get_mob()
    x.glow = 10.0
    x.glow_radius = 1.0
    x.wait()
    return


render_all_funcs(__name__, start_index=0, max_rendered=-1)

