import copy

import numpy
import torch.nn.functional as F
from svgelements import Path, Line, Move, Close
import pathlib

# Deferred: manim's import chain (sympy/networkx/scipy/...) costs ~2 s of
# ``import algan`` and is only needed once a Text/Tex is constructed. The
# svg-cache module patches manim, so it must ride along on the first load.
from algan.utils.lazy_import import LazyModule

mn = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))
from algan.settings.defaults import *
from algan.settings.style_defaults import *
from algan.animation.animation_contexts import Sync, Off, AnimationContext, Lag, Seq
from algan.mobs.triangulated_bezier_circuit import (
    TriangulatedBezierCircuit,
    point_to_tensor2,
)
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.constants.spatial import DOWN, RIGHT
from algan.constants.color import *
from algan.mobs.group import Group
from algan.mobs.mob import Mob
from algan.mobs.image_mob import ImageMob
from algan.utils.animation_utils import animate_lagged_by_location
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import unsquish
from algan.utils.mob_utils import BatchedMobViewSequence


def make_manim_dir():
    """Create manim's tex/text output directories if they don't exist yet.

    Called lazily on first :class:`Tex` construction (manim errors if they are
    missing) rather than at ``import algan`` time, so importing the package
    doesn't write to disk.
    """
    for tex_dir in [mn.config.get_dir("tex_dir"), mn.config.get_dir("text_dir")]:
        if not tex_dir.exists():
            tex_dir.mkdir(parents=True)


class Tex(Mob):
    """LaTeX text rendered as one packed batch of cubic bezier glyphs.

    ``character_mobs`` provides lazy indexed views into the batch.
    """

    triangulated = False

    def __init__(self, text, font_size=24, latex=True, *args, **kwargs):
        make_manim_dir()
        if "preamble" in kwargs:
            kwargs["tex_template"] = mn.TexTemplate(
                preamble=_default_preamble() + "\n" + kwargs["preamble"]
            )
            del kwargs["preamble"]
        self.latex = latex
        # if not self.latex:
        #    text = f'\\text{{{text}}}'
        base_font_size = 48
        if isinstance(text, str):
            text = [text]
        t = (mn.MathTex if self.latex else mn.Text)(*text, font_size=base_font_size)

        def maybe_flip(submob):
            x = torch.from_numpy(submob.points).to(COMPUTING_DEFAULTS.animation_device)
            if (not latex) or (not isinstance(submob, mn.VMobjectFromSVGPath)):
                return x.flip(-2)
            return x

        if latex:
            sub_mobs = [_.submobjects for _ in t.submobjects]
            self.num_mobs_per_segment = torch.tensor([len(_) for _ in sub_mobs])
            self.segment_ends = self.num_mobs_per_segment.cumsum(0)
            self.segment_starts = self.segment_ends - self.num_mobs_per_segment
            chars = [x for l in sub_mobs for x in l]
        else:
            chars = t.submobjects
        triangulated_paths = [
            unsquish(maybe_flip(_), -2, 4).transpose(-3, -2)
            for _ in [_ for _ in chars if not isinstance(_, mn.ImageMobject)]
        ]
        bezier_paths = [
            unsquish(
                torch.from_numpy(_.points).to(
                    COMPUTING_DEFAULTS.animation_device
                ).float(),
                -2,
                4,
            )
            for _ in chars
            if not isinstance(_, mn.ImageMobject)
        ]
        with Off():
            paths = triangulated_paths if self.triangulated else bezier_paths
            # Both geometry types retain path boundaries in parent_batch_sizes.
            # Building once avoids a Mob hierarchy (and timeline rows) per glyph.
            if self.triangulated:
                character_batch = (
                    TriangulatedBezierCircuit(
                        paths,
                        *args,
                        invert=False,
                        hash_keys=paths,
                        reverse_points=False,
                        **kwargs,
                    )
                    if paths
                    else None
                )
            else:
                bezier_kwargs = dict(kwargs)
                bezier_kwargs.setdefault("color", WHITE)
                bezier_kwargs.setdefault("border_color", bezier_kwargs["color"])
                bezier_kwargs.setdefault("border_width", 0)
                character_batch = (
                    BezierCircuitCubic.from_batches(paths, *args, **bezier_kwargs)
                    if paths
                    else None
                )
            self._character_batch = character_batch
            self.character_mobs = BatchedMobViewSequence(
                self._character_batch, len(paths)
            )
            self.image_mobs = [
                ImageMob(_) for _ in chars if isinstance(_, mn.ImageMobject)
            ]
            super().__init__(*args, **kwargs)
            if self._character_batch is not None:
                self.add_children(self._character_batch)
            self.add_children(self.image_mobs)
            self.scale(font_size / base_font_size)

    def become(self, other_mob, *args, **kwargs):
        result = super().become(other_mob, *args, **kwargs)
        # ``detach_history`` returns a clone, and cubic morphing may expand the
        # packed glyph batch to match the target. Cached lightweight views from
        # the pre-morph object still carry the old size/data_sub_inds, so rebuild
        # the sequence against the returned batch owner.
        if isinstance(result, Tex) and result._character_batch is not None:
            result.character_mobs = BatchedMobViewSequence(
                result._character_batch,
                result._character_batch.location.shape[-2],
            )
        return result

    def get_segment(self, i):
        return self[self.segment_starts[i]:self.segment_ends[i]]

    def __getitem__(self, item):
        return Group([self.character_mobs[item]])

    def __len__(self):
        return len(self.character_mobs)

    def default_color(self):
        return BLUE

    def on_create(self):
        with Seq(run_time=1):
            with Off():  # Ensure initial state setting is not recorded as an animation
                opacity = self.opacity
                self.opacity = 0
            self._create_recursive(animate=False)  # Mark as created without immediate animation
            self.wave_color( None, direction=F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1), opacity=opacity)
            #self.opacity = 1
        return self
        tiles = list(traverse([c.children for c in self.children]))
        with AnimationContext(run_time_unit=2):
            animate_lagged_by_location(
                tiles,
                lambda m: m.spawn_from_random_direction(),
                F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
            )
        return self

    def on_destroy(self):
        # tiles = list(traverse([c.children for c in self.children]))
        # with AnimationContext(run_time_unit=2):
        #    animate_lagged_by_location(tiles, lambda m: m.despawn_from_random_direction(), F.normalize(RIGHT*1.5+DOWN, p=2, dim=-1))
        with Seq():
            self.wave_color(
                None, direction=F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1), opacity=0
            )
            old_ct = self.animation_manager.context.timespan.current_time
            self.animation_manager.context.timespan.current_time = (
                self.animation_manager.context.timespan.original_end
            )
            self._destroy_recursive(animate=False)
            self.animation_manager.context.timespan.current_time = old_ct
        return self


class OldTex(Mob):
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

    def __init__(
        self, text: str, font_size: float = 48, latex=True, debug=False, **kwargs
    ):
        if "preamble" in kwargs:
            kwargs["tex_template"] = mn.TexTemplate(
                preamble=_default_preamble() + "\n" + kwargs["preamble"]
            )
            del kwargs["preamble"]

        if "color" not in kwargs:
            kwargs["color"] = STYLE_DEFAULTS.text_color

        kwargs2 = {k: v for k, v in kwargs.items()}
        if "create" in kwargs2:
            del kwargs2["create"]
        if "init" in kwargs2:
            del kwargs2["init"]
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
            animate_lagged_by_location(
                tiles,
                lambda m: m.spawn_from_random_direction(),
                F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
            )
        return self

    def on_destroy(self):
        tiles = list(traverse([c.children for c in self.children]))
        with AnimationContext(run_time_unit=2):
            animate_lagged_by_location(
                tiles,
                lambda m: m.despawn_from_random_direction(),
                F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
            )
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
        make_manim_dir()
        pathlib.Path("media/tex").mkdir(exist_ok=True, parents=True)
        # s = 0.105 * self.size / 100
        s = 0.04 * 45 / 100
        self.convert_ratio = (0.105 * self.font_size / 100) / s
        manim_kwargs = {k: v for k, v in kwargs.items()}
        if "color" in manim_kwargs:
            del manim_kwargs["color"]
        if "scale" in manim_kwargs:
            del manim_kwargs["scale"]
        if "use_cache" in manim_kwargs:
            del manim_kwargs["use_cache"]
        if "add_to_scene" in manim_kwargs:
            del manim_kwargs["add_to_scene"]
        if "create" in manim_kwargs:
            del manim_kwargs["create"]
        text = (mn.MathTex if self.latex else mn.Tex)(text, **manim_kwargs)

        def get_rect_as_path(ps):
            ps = ps[..., :2].astype(numpy.float32)
            ps = numpy.flip(ps, 0)
            vmob = mn.VMobjectFromSVGPath(
                Path(
                    Move(ps[0]),
                    Close(ps[0], ps[0]),
                    *([(Line)(ps[i * 4], ps[(i + 1) * 4 - 1]) for i in range(4)]),
                    Move(ps[0]),
                )
            )
            vmob.needs_to_reverse = True
            return vmob

        svg_mobs = [
            [
                __
                if isinstance(__, mn.VMobjectFromSVGPath)
                else get_rect_as_path(_.original_points[i])
                for i, __ in enumerate(_.submobjects)
            ]
            for _ in text.submobjects
        ]
        svg_mobs = [x for l in svg_mobs for x in l]

        all_points = torch.cat(
            [
                torch.stack([point_to_tensor2(_.end) for _ in c.path_obj], 0)
                for c in svg_mobs
            ]
        ).flip(-1)
        mx_point = all_points.amax(0)
        mn_point = all_points.amin(0)
        mean = (mx_point + mn_point) / 2

        def update_attr_mean(ele, m):
            for attr in ["start", "end", "control1", "control2"]:
                if hasattr(ele, attr) and ele.__getattribute__(attr) is not None:
                    ele.__getattribute__(attr).x = float(
                        (ele.__getattribute__(attr).x - m[1].item()) * s
                    )
                    ele.__getattribute__(attr).y = float(
                        -(ele.__getattribute__(attr).y - m[0].item()) * s
                    )

        def normalize(_, m=mean):
            _ = copy.deepcopy(_)
            _[..., 0] = (_[..., 0] - m[1].item()) * s
            _[..., 1] = -(_[..., 1] - m[0].item()) * s
            return _

        for c in svg_mobs:
            for element in c.path_obj:
                update_attr_mean(element, mean)

        all_points = torch.cat(
            [
                torch.stack([point_to_tensor2(_.end) for _ in c.path_obj], 0).flip(-1)
                for c in svg_mobs
            ]
        )
        mx_point = all_points.amax(0)
        mn_point = all_points.amin(0)
        self.mn_point = torch.cat((torch.zeros_like(mn_point[..., :1]), mn_point), -1)
        self.mx_point = torch.cat((torch.zeros_like(mx_point[..., :1]), mx_point), -1)

        with Off():
            self.character_mobs = TriangulatedBezierCircuit(
                [c.path_obj for c in svg_mobs],
                invert=True,
                hash_keys=None,
                reverse_points=hasattr(svg_mobs[0], "needs_to_reverse"),
                init=False,
                **kwargs,
            )

    def get_boundary_points_test(self):
        return torch.stack(
            (
                self.mn_point,
                torch.stack(
                    (
                        torch.zeros_like(self.mn_point[..., 0]),
                        self.mn_point[..., 1],
                        self.mx_point[..., 2],
                    ),
                    -1,
                ),
                torch.stack(
                    (
                        torch.zeros_like(self.mn_point[..., 0]),
                        self.mx_point[..., 1],
                        self.mn_point[..., 2],
                    ),
                    -1,
                ),
                self.mx_point,
            ),
            -2,
        ) + self.location.unsqueeze(-2)


def _default_preamble():
    """Vendored manim's default LaTeX preamble, fetched on first use
    (deferred: importing ``algan.external_libraries.manim`` costs ~2 s of
    ``import algan`` and is only needed when a Tex has a custom preamble)."""
    from algan.external_libraries.manim.utils.tex import _DEFAULT_PREAMBLE

    return _DEFAULT_PREAMBLE


class Text(Tex):
    """Plain text rendered as one packed batch of cubic bezier glyphs.

    Parameters
    ----------
    text
        The text to display.
    **kwargs
        Passed to :class:`~.Tex`.

    """

    def __init__(self, text, **kwargs):
        super().__init__(text, latex=False, **kwargs)


class TexTriangulated(Tex):
    """LaTeX text rendered as one packed batch of triangulated glyphs."""

    triangulated = True


class TextTriangulated(TexTriangulated):
    """Plain text rendered as one packed batch of triangulated glyphs."""

    def __init__(self, text, **kwargs):
        super().__init__(text, latex=False, **kwargs)
