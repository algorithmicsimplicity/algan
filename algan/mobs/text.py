import copy

import numpy
import torch.nn.functional as F
import algan.external_libraries.manim as mn
from svgelements import Path, Line, Move, Close
import pathlib

from algan.defaults.style_defaults import *
from algan.animation.animation_contexts import Sync, Off, AnimationContext, Lag
from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit, point_to_tensor2
from algan.constants.spatial import DOWN, RIGHT
from algan.mobs.group import Group
from algan.mobs.mob import Mob
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.python_utils import traverse


class Tex(Mob):
    """Mob for displaying tex.

    Parameters
    ----------
    text
        String containing the text to display.
    font_size
        Font size of the text.
    **kwargs
        Passed to :class:`~.Mob`.

    """
    def __init__(self, text:str, font_size:float=48, latex=True, debug=False, **kwargs):
        if 'preamble' in kwargs:
            kwargs['tex_template'] = mn.TexTemplate(preamble=_DEFAULT_PREAMBLE + '\n' + kwargs['preamble'])
            del kwargs['preamble']

        if 'color' not in kwargs:
            kwargs['color'] = DEFAULT_TEXT_COLOR

        kwargs2 = {k: v for k, v in kwargs.items()}
        if 'create' in kwargs2:
            del kwargs2['create']
        if 'init' in kwargs2:
            del kwargs2['init']
        super().__init__(**kwargs2, init=False)

        self.debug = debug
        self.kwargs = kwargs
        self.size = self.font_size = font_size
        self.text = text
        self.latex = latex
        self.create_character_mobs(text, **kwargs2)
        self.add_children(self.character_mobs)
        with Off():
            self.scale(self.convert_ratio)

    def __getitem__(self, item):
        return Group([self.character_mobs[item]])

    def __len__(self):
        return len(self.character_mobs)

    def default_color(self):
        return BLUE

    def highlight(self):
        self.orig_color = self.color
        with Sync():
            for _ in self.get_descendants():
                _.color = RED_A
        return self

    def highlight_off(self):
        with Sync():
            for _ in self.get_descendants():
                _.color = WHITE
        return self

    def on_create(self):
        tiles = list(traverse([c.children for c in self.children]))
        with AnimationContext(run_time_unit=2):
            animate_lagged_by_location(tiles, lambda m: m.spawn_from_random_direction(), F.normalize(RIGHT*1.5+DOWN, p=2, dim=-1))
        return self

    def on_destroy(self):
        tiles = list(traverse([c.children for c in self.children]))
        with AnimationContext(run_time_unit=2):
            animate_lagged_by_location(tiles, lambda m: m.despawn_from_random_direction(), F.normalize(RIGHT*1.5+DOWN, p=2, dim=-1))
        return self

    def set_fill_width(self, fill_portion):
        with Lag(0.5, run_time=1.0):
            for c in self.character_mobs:
                c.fill_portion = fill_portion
            self.fill_portion = fill_portion

    def set_color(self, color):
        with Sync():
            for c in self.character_mobs:
                c.color = color
            self.color = color
        return self

    def set_size(self, size):
        with Sync():
            for c in self.character_mobs:
                c.size = size
            self.size = size
        return self

    def set_text(self, text):
        self.children = set()
        self.create_character_mobs(text, **self.kwargs)
        self.add_children(self.character_mobs)
        return self

    def create_character_mobs(self, text, **kwargs):
        pathlib.Path('media/tex').mkdir(exist_ok=True, parents=True)
        #s = 0.105 * self.size / 100
        s = 0.02 * 45 / 100
        self.convert_ratio = (0.105 * self.font_size / 100) / s
        manim_kwargs = {k: v for k, v in kwargs.items()}
        if 'color' in manim_kwargs:
            del manim_kwargs['color']
        if 'scale' in manim_kwargs:
            del manim_kwargs['scale']
        if 'use_cache' in manim_kwargs:
            del manim_kwargs['use_cache']
        if 'add_to_scene' in manim_kwargs:
            del manim_kwargs['add_to_scene']
        if 'create' in manim_kwargs:
            del manim_kwargs['create']
        text = (mn.MathTex if self.latex else mn.Tex)(text, **manim_kwargs)

        def get_rect_as_path(ps):
            ps = ps[...,:2].astype(numpy.float32)
            ps = numpy.flip(ps, 0)
            vmob = mn.VMobjectFromSVGPath(Path(Move(ps[0]), Close(ps[0], ps[0]), *([(Line)(ps[i*4], ps[(i+1)*4-1]) for i in range(4)]), Move(ps[0])))
            vmob.needs_to_reverse = True
            return vmob

        svg_mobs = [[__ if isinstance(__, mn.VMobjectFromSVGPath) else get_rect_as_path(_.original_points[i]) for i, __ in enumerate(_.submobjects)] for _ in text.submobjects]
        svg_mobs = [x for l in svg_mobs for x in l]

        all_points = torch.cat([torch.stack([point_to_tensor2(_.end) for _ in c.path_obj], 0) for c in svg_mobs]).flip(-1)
        mx_point = all_points.amax(0)
        mn_point = all_points.amin(0)
        mean = (mx_point + mn_point) / 2

        def update_attr_mean(ele, m):
            for attr in ['start', 'end', 'control1', 'control2']:
                if hasattr(ele, attr) and ele.__getattribute__(attr) is not None:
                    ele.__getattribute__(attr).x = float((ele.__getattribute__(attr).x - m[1].item()) * s)
                    ele.__getattribute__(attr).y = float(-(ele.__getattribute__(attr).y - m[0].item()) * s)

        def normalize(_, m=mean):
            _ = copy.deepcopy(_)
            _[..., 0] = (_[...,0] - m[1].item()) * s
            _[..., 1] = -(_[..., 1] - m[0].item()) * s
            return _

        for c in svg_mobs:
            for element in c.path_obj:
                update_attr_mean(element, mean)

        all_points = torch.cat([torch.stack([point_to_tensor2(_.end) for _ in c.path_obj], 0).flip(-1) for c in svg_mobs])
        mx_point = all_points.amax(0)
        mn_point = all_points.amin(0)
        self.mn_point = torch.cat((torch.zeros_like(mn_point[..., :1]), mn_point), -1)
        self.mx_point = torch.cat((torch.zeros_like(mx_point[..., :1]), mx_point), -1)

        with Off():
            self.character_mobs = TriangulatedBezierCircuit([c.path_obj for c in svg_mobs], invert=True, hash_keys=None, reverse_points=hasattr(svg_mobs[0], 'needs_to_reverse'), init=False, **kwargs)

    def get_boundary_points_test(self):
        return torch.stack((self.mn_point,
                torch.stack((torch.zeros_like(self.mn_point[..., 0]), self.mn_point[...,1], self.mx_point[...,2]), -1),
                torch.stack((torch.zeros_like(self.mn_point[..., 0]), self.mx_point[...,1], self.mn_point[...,2]), -1),
                     self.mx_point), -2) + self.location.unsqueeze(-2)


from algan.external_libraries.manim.utils.tex import _DEFAULT_PREAMBLE


class Text(Tex):
    """Mob for displaying LaTeX.

    Parameters
    ----------
    text
        The LaTeX source that will be compiled.
    **kwargs
        Passed to :class:`~.Text`

    """
    def __init__(self, text, **kwargs):
        super().__init__(text, latex=False, **kwargs)