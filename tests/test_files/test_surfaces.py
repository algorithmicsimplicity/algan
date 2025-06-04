import torch.nn.functional as F

from algan.animation.animation_contexts import Sync
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.group import Group
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.mobs.surfaces.surface import Surface
from algan.mobs.shapes_3d import Sphere, Cylinder
from algan.utils.algan_utils import render_all_funcs


get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_cylinder():
    xs = Group([Cylinder(num_grid_pieces=80).scale(0.1) for _ in range(3)]).arrange_in_line(RIGHT).spawn()
    #x = get_mob()
    #with Sequenced(run_time=4):
    #x.scale(torch.tensor((1,2,1)))
    #x.rotate(180, RIGHT)
    for i, x in enumerate(xs):
        x.move_between_points(LEFT + UP + DOWN * i, RIGHT)
    #x.move(LEFT)
    with Sync():
        for i, x in enumerate(xs):
            x.wave_color(PURE_BLUE, 0)


def test_sphere():
    x = Sphere(num_grid_pieces=80).spawn()
    #x = get_mob()
    #with Sequenced(run_time=4):
    x.rotate(180, RIGHT)
    x.move(LEFT)


def test_surface():
    x = Surface(lambda x: torch.cat((x, torch.zeros_like(x[...,:1])), -1), checkered_color=PURE_GREEN, num_grid_pieces=10).spawn()
    x.move(RIGHT)
    x.rotate(180, RIGHT)


render_all_funcs(__name__, start_index=0, max_rendered=1)
