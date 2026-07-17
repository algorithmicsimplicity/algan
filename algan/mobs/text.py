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
from algan.constants.spatial import DOWN, LEFT, ORIGIN, RIGHT, UP
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

    Touching ``mn.config`` loads manim, and with it
    :mod:`algan.utils.manim_svg_cache`, which first redirects these
    directories into ``DIRECTORY_DEFAULTS.cache_directory``.

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

    def __init__(
        self,
        *tex_strings,
        arg_separator="",
        tex_environment="center",
        font_size=24,
        latex=True,
        **kwargs,
    ):
        """Build TeX from one or more strings using Manim's public API.

        ``latex`` is retained as an Algan extension: when false, the strings are
        treated as plain text.  Manim's ``arg_separator`` and
        ``tex_environment`` keyword arguments are accepted directly.
        """
        make_manim_dir()
        if "preamble" in kwargs:
            kwargs["tex_template"] = mn.TexTemplate(
                preamble=_default_preamble() + "\n" + kwargs.pop("preamble")
            )

        if len(tex_strings) == 1 and isinstance(tex_strings[0], (list, tuple)):
            tex_strings = tuple(tex_strings[0])
        if not tex_strings:
            tex_strings = ("",)
        self.tex_strings = tuple(str(part) for part in tex_strings)
        self.tex_environment = tex_environment
        self.arg_separator = arg_separator
        self.latex = latex

        base_font_size = 48
        if self.latex:
            t = mn.MathTex(
                *self.tex_strings,
                arg_separator=arg_separator,
                tex_environment=tex_environment,
                font_size=base_font_size,
            )
        else:
            if not hasattr(mn, "Text"):
                raise RuntimeError(
                    "Plain Text rendering requires Manim's optional Pango support; "
                    "use algan.Text, which provides a LaTeX fallback."
                )
            t = mn.Text(arg_separator.join(self.tex_strings), font_size=base_font_size)

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
            chars = [x for group in sub_mobs for x in group]
        else:
            chars = t.submobjects
            self.num_mobs_per_segment = torch.tensor([len(chars)])
            self.segment_ends = self.num_mobs_per_segment.cumsum(0)
            self.segment_starts = self.segment_ends - self.num_mobs_per_segment

        triangulated_paths = [
            unsquish(maybe_flip(char), -2, 4).transpose(-3, -2)
            for char in chars
            if not isinstance(char, mn.ImageMobject)
        ]
        bezier_paths = [
            unsquish(
                torch.from_numpy(char.points)
                .to(COMPUTING_DEFAULTS.animation_device)
                .float(),
                -2,
                4,
            )
            for char in chars
            if not isinstance(char, mn.ImageMobject)
        ]
        with Off():
            paths = triangulated_paths if self.triangulated else bezier_paths
            if self.triangulated:
                character_batch = (
                    TriangulatedBezierCircuit(
                        paths,
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
                    BezierCircuitCubic.from_batches(paths, **bezier_kwargs)
                    if paths
                    else None
                )
            self._character_batch = character_batch
            self.character_mobs = BatchedMobViewSequence(
                self._character_batch, len(paths)
            )
            self.image_mobs = [
                ImageMob(char, add_to_scene=False)
                for char in chars
                if isinstance(char, mn.ImageMobject)
            ]
            super().__init__(**kwargs)
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

    When Pango is unavailable, Algan renders the textual content through
    LaTeX text mode. Font-family and span-level styling arguments are accepted
    and retained as metadata, but cannot affect that fallback renderer.
    """

    def __init__(
        self,
        text,
        fill_opacity=1.0,
        stroke_width=0,
        color=None,
        font_size=48,
        line_spacing=-1,
        font="",
        slant="NORMAL",
        weight="NORMAL",
        t2c=None,
        t2f=None,
        t2g=None,
        t2s=None,
        t2w=None,
        gradient=None,
        tab_width=4,
        warn_missing_font=True,
        height=None,
        width=None,
        should_center=True,
        disable_ligatures=False,
        use_svg_cache=False,
        **kwargs,
    ):
        self.text = str(text).expandtabs(tab_width)
        self.font = font
        self.slant = slant
        self.weight = weight
        self.line_spacing = line_spacing
        self.t2c, self.t2f, self.t2g = t2c, t2f, t2g
        self.t2s, self.t2w = t2s, t2w
        self.gradient = gradient
        self.disable_ligatures = disable_ligatures
        self.use_svg_cache = use_svg_cache
        kwargs.setdefault("opacity", fill_opacity)
        kwargs.setdefault("border_width", stroke_width / 2)
        if color is not None:
            kwargs.setdefault("color", color)
            kwargs.setdefault("border_color", color)

        if hasattr(mn, "Text"):
            # This path is retained for vendored builds that opt into Pango.
            super().__init__(
                self.text,
                font_size=font_size,
                latex=False,
                **kwargs,
            )
        else:
            import re

            escaped = re.sub(r"([#$%&_{}])", r"\\\1", self.text)
            escaped = escaped.replace("~", r"\textasciitilde{}")
            escaped = escaped.replace("^", r"\textasciicircum{}")
            escaped = escaped.replace("\n", r"\\")
            super().__init__(
                rf"\text{{{escaped}}}",
                font_size=font_size,
                latex=True,
                **kwargs,
            )
            self.latex = False

        # Match Manim's post-construction size overrides.
        with Off():
            if height is not None:
                current = self.get_length_in_direction(UP)
                if float(current.reshape(-1)[0]) > 0:
                    self.scale(float(height) / float(current.reshape(-1)[0]))
            if width is not None:
                current = self.get_length_in_direction(RIGHT)
                if float(current.reshape(-1)[0]) > 0:
                    self.scale(float(width) / float(current.reshape(-1)[0]))
            if should_center:
                self.move_to(ORIGIN)


class TexTriangulated(Tex):
    """LaTeX text rendered as one packed batch of triangulated glyphs."""

    triangulated = True


class TextTriangulated(TexTriangulated):
    """Triangulated plain text; accepts the same arguments as :class:`Text`."""

    def __init__(self, text, **kwargs):
        # Reuse Text's fallback preprocessing, then construct the triangulated
        # TeX representation directly.
        import re

        font_size = kwargs.pop("font_size", 48)
        escaped = re.sub(r"([#$%&_{}])", r"\\\1", str(text))
        escaped = escaped.replace("~", r"\textasciitilde{}")
        escaped = escaped.replace("^", r"\textasciicircum{}")
        super().__init__(rf"\text{{{escaped}}}", font_size=font_size, **kwargs)
        self.text = str(text)
        self.latex = False


class MarkupText(Text):
    """Text accepting Manim/Pango markup syntax.

    Markup is stripped when the optional Pango renderer is unavailable; the
    resulting textual content, entities, and line breaks are preserved.
    """

    def __init__(
        self,
        text,
        fill_opacity=1,
        stroke_width=0,
        color=None,
        font_size=48,
        line_spacing=-1,
        font="",
        slant="NORMAL",
        weight="NORMAL",
        justify=False,
        gradient=None,
        tab_width=4,
        height=None,
        width=None,
        should_center=True,
        disable_ligatures=False,
        warn_missing_font=True,
        **kwargs,
    ):
        import html
        import re

        self.original_text = str(text)
        plain = re.sub(r"<br\s*/?>", "\n", self.original_text, flags=re.IGNORECASE)
        plain = re.sub(r"<[^>]+>", "", plain)
        self.justify = justify
        super().__init__(
            html.unescape(plain),
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            color=color,
            font_size=font_size,
            line_spacing=line_spacing,
            font=font,
            slant=slant,
            weight=weight,
            gradient=gradient,
            tab_width=tab_width,
            height=height,
            width=width,
            should_center=should_center,
            disable_ligatures=disable_ligatures,
            warn_missing_font=warn_missing_font,
            **kwargs,
        )


class Paragraph(Group):
    """A group of individually addressable text lines."""

    def __init__(self, *text, line_spacing=-1, alignment=None, **kwargs):
        add_to_scene = kwargs.pop("add_to_scene", True)
        lines = []
        for part in text:
            lines.extend(str(part).split("\n"))
        if not lines:
            lines = [""]
        mobs = [Text(line, add_to_scene=False, **kwargs) for line in lines]
        super().__init__(*mobs, add_to_scene=add_to_scene)
        if mobs:
            buffer = 0.2 if line_spacing == -1 else line_spacing
            align_direction = {
                "left": LEFT,
                "center": None,
                "right": RIGHT,
                None: None,
            }.get(alignment)
            if alignment not in {None, "left", "center", "right"}:
                raise ValueError("alignment must be 'left', 'center', 'right', or None")
            self.arrange_in_line(
                DOWN,
                buffer=buffer,
                alignment_direction=align_direction,
            )
        self.lines_text = lines
        self.chars = self.mobs

    def set_all_lines_alignments(self, alignment):
        replacement = Paragraph(
            *self.lines_text,
            alignment=alignment,
            add_to_scene=False,
        )
        return self.become(replacement, detach_history=False)


class Code(Group):
    """Source-code display with Manim 0.20.1 constructor names.

    The dependency-light implementation preserves line addressing, line
    numbers, and both rectangle/window background modes. Syntax tokens use the
    base text color rather than Pygments span colors.
    """

    def __init__(
        self,
        code_file=None,
        code_string=None,
        language=None,
        formatter_style="vim",
        tab_width=4,
        add_line_numbers=True,
        line_numbers_from=1,
        background="rectangle",
        background_config=None,
        paragraph_config=None,
        **kwargs,
    ):
        from algan.mobs.shapes_2d import Circle, Rectangle, SurroundingRectangle

        add_to_scene = kwargs.pop("add_to_scene", True)
        if code_string is None:
            if code_file is None:
                raise ValueError("either code_file or code_string must be provided")
            code_string = pathlib.Path(code_file).read_text(encoding="utf-8")
        source_lines = str(code_string).expandtabs(tab_width).splitlines() or [""]
        paragraph_config = dict(paragraph_config or {})
        paragraph_config.update(kwargs)
        self.code = Paragraph(
            *source_lines,
            alignment="left",
            add_to_scene=False,
            **paragraph_config,
        )
        mobs = [self.code]
        self.line_numbers = None
        if add_line_numbers:
            self.line_numbers = Paragraph(
                *(str(i) for i in range(line_numbers_from, line_numbers_from + len(source_lines))),
                alignment="right",
                add_to_scene=False,
                **paragraph_config,
            )
            with Off():
                self.line_numbers.move_next_to(self.code, LEFT, buffer=0.2)
            mobs.insert(0, self.line_numbers)
        super().__init__(*mobs, add_to_scene=add_to_scene)

        self.background_mobject = None
        background_config = dict(background_config or {})
        if background == "rectangle":
            self.background_mobject = SurroundingRectangle(
                self,
                add_to_scene=False,
                **background_config,
            )
        elif background == "window":
            frame = SurroundingRectangle(
                self,
                add_to_scene=False,
                **background_config,
            )
            dots = Group(
                *[
                    Circle(radius=0.04, add_to_scene=False)
                    for _ in range(3)
                ],
                add_to_scene=False,
            )
            with Off():
                dots.arrange_in_line(RIGHT, buffer=0.08)
                dots.move_next_to(frame.get_boundary_in_direction(UP), DOWN, buffer=0.08)
            self.background_mobject = Group(frame, dots, add_to_scene=False)
        elif background not in {None, False}:
            raise ValueError("background must be 'rectangle', 'window', or None")
        if self.background_mobject is not None:
            self.add(self.background_mobject)

        self.language = language
        self.formatter_style = formatter_style
        self.code_string = str(code_string)
        self.code_file = code_file
