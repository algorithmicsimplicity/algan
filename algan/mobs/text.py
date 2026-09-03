"""Text and LaTeX, rendered as packed batches of bezier glyphs.

:class:`Tex` compiles LaTeX through Algan's bundled Manim, converts the resulting
glyph outlines to cubic bezier circuits, and packs every glyph of the string into
a single batched Mob. :class:`Text` is the same machinery with Pango font
rendering instead of LaTeX, and :class:`MarkupText` accepts Pango markup.

Because the glyphs are outlines rather than bitmaps, text scales without
softening and morphs into other shapes like any 2-D Mob.

``character_mobs`` gives lazy indexed views onto individual glyphs in the packed
batch, which is what per-character animation works on. A multi-part :class:`Tex`
also exposes each of its source strings through
:meth:`Tex.get_segment` -- these are views, not ``children``, which hold the
single packed batch. :meth:`Tex.write` draws the string as though by hand.

The ``Triangulated`` variants build filled triangle meshes instead of bezier
circuits, for cases where a fragment-shaded interior is wanted.

:func:`make_manim_dir` prepares Manim's Tex/text scratch directories inside
Algan's cache so nothing is written beside the user's script.

See :doc:`/advanced_user_tutorials/text_and_math`.
"""

from __future__ import annotations

import pathlib

import torch.nn.functional as F

from algan.settings._startup import _ANIMATION_DEVICE

# Deferred: manim's import chain (sympy/networkx/scipy/...) costs ~2 s of
# ``import algan`` and is only needed once a Text/Tex is constructed. The
# svg-cache module patches manim, so it must ride along on the first load.
from algan.utils.lazy_import import LazyModule

mn = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))
# mn = LazyModule("algan.external_libraries.manim", extras=("algan.utils.manim_svg_cache",))
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    Off,
    Seq,
    active_scene_for_new_mob,
)
from algan.constants.color import *
from algan.constants.spatial import DOWN, LEFT, ORIGIN, RIGHT, UP
from algan.errors import AlganConfigurationError
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.mobs.image_mob import ImageMob
from algan.mobs.triangulated_bezier_circuit import (
    TriangulatedBezierCircuit,
)
from algan.utils.mob_utils import BatchedMobViewSequence
from algan.utils.tensor_utils import unsquish

#: The external programs LaTeX typesetting needs, and what each one does.
#: ``latex`` compiles the document, ``dvisvgm`` turns the DVI into the SVG
#: outlines Algan reads.
_LATEX_BINARIES = ("latex", "dvisvgm")


def _require_latex_toolchain():
    """Raise before anything is written when there is no TeX distribution.

    Most people who ``pip install algan`` have no TeX, and until this existed
    the first :class:`Tex` produced a ``rich``-formatted line from the vendored
    Manim and then a raw ``FileNotFoundError: 'latex'`` from deep inside it,
    with nothing to say which program was missing or that :class:`Text` needs
    none of it. Checked here rather than in the vendored code, and before
    :func:`make_manim_dir` writes a scratch directory for a run that cannot
    happen.
    """
    import shutil

    missing = [name for name in _LATEX_BINARIES if shutil.which(name) is None]
    if not missing:
        return
    names = " and ".join(missing)
    raise AlganConfigurationError(
        f"LaTeX typesetting needs {names} on PATH, and "
        f"{'they were' if len(missing) > 1 else 'it was'} not found.\n"
        "  Debian/Ubuntu: sudo apt install texlive-latex-base "
        "texlive-latex-extra dvisvgm\n"
        "  macOS:         brew install --cask basictex, then "
        "sudo tlmgr install standalone preview dvisvgm\n"
        "  Windows:       install MiKTeX (https://miktex.org) and let it "
        "install packages on the fly\n"
        "Text(...) needs none of this: it renders through Pango and is the "
        "right class for prose. Tex(..., latex=False) does the same."
    )


def make_manim_dir():
    """Create manim's tex/text output directories if they don't exist yet.

    Touching ``mn.config`` loads manim, and with it
    :mod:`algan.utils.manim_svg_cache`. The paths are resolved again here so a
    runtime change to ``SETTINGS.paths.cache_directory`` is honored and Manim
    never creates its default ``media`` tree beside the user's script.

    Called lazily on first :class:`Tex` construction (manim errors if they are
    missing) rather than at ``import algan`` time, so importing the package
    doesn't write to disk.
    """
    config = mn.config
    from algan.utils.manim_svg_cache import _configure_manim_dirs

    for tex_dir in _configure_manim_dirs(config):
        tex_dir.mkdir(parents=True, exist_ok=True)


class Tex(Mob):
    r"""LaTeX compiled to one packed batch of cubic bezier glyphs.

    The string is typeset in LaTeX's **math mode**, so ``Tex("x^2")`` is a
    squared x rather than the three literal characters. Pass ``latex=False`` for
    prose, or use :class:`~algan.mobs.text.Text`, which is this class wired to a
    Pango font renderer instead.

    The glyphs are outlines, not bitmaps: the text stays sharp at any scale and
    morphs into other 2-D shapes like any other bezier Mob. They arrive as a
    single packed Mob rather than one Mob per character, which is what makes a
    long string cheap. Index it to animate one glyph -- ``equation[3]`` is a
    view sharing the batch's rows, so moving it moves that glyph of the
    original, and it needs no spawning of its own. Several source strings become
    several *segments*, addressable with :meth:`get_segment`, which is the usual
    way to highlight one term of an equation.

    :class:`~algan.mobs.manim_compat.MathTex` renders LaTeX too and is a
    different object: it is the Manim-compatibility wrapper, for when a ported
    script needs ``tex_to_color_map`` or Manim method delegation. This class is
    the native one, and the only one with per-glyph indexing,
    :meth:`get_segment` and :meth:`write`.

    Animation
    ---------
    Constructing a ``Tex`` records nothing: LaTeX runs immediately and the Mob
    joins the active Scene unspawned. :meth:`~algan.animatable_base.animatable.Animatable.spawn`
    is what plays its entrance -- :meth:`on_create`, a diagonal fade running down
    and to the right so the words arrive in reading order, lasting 1 second
    regardless of the enclosing context. ``Tex(...).spawn(False)`` skips it, which
    is what :meth:`write` wants.

    Parameters
    ----------
    *tex_strings
        One or more LaTeX sources. Each becomes a segment retrievable with
        :meth:`get_segment`, and they are compiled as one document so a
        ``\left`` in one string can close in the next. A single list or tuple is
        unpacked, and no strings at all gives an empty ``Tex``.
    delimiter
        Inserted between consecutive ``tex_strings`` in the compiled source (and
        used to join them when ``latex=False``). Defaults to ``" "``, one space.
    tex_environment
        Name of the LaTeX environment to typeset in, such as ``"align*"`` or
        ``"gather*"``. Defaults to ``None``, meaning Manim's own default of
        ``align*``.
    font_size
        Glyph size in Manim's font-size units. The batch is always built at 48
        and then scaled by ``font_size / 48``, so this is a scale factor in
        disguise: ``48`` matches Manim's default text size and ``24`` is half of
        it. Defaults to ``24``.
    latex
        Whether to typeset through LaTeX. ``False`` treats the strings as plain
        text and routes them to Manim's Pango renderer instead -- an Algan
        extension, and how :class:`~algan.mobs.text.Text` is built. Defaults to
        ``True``.
    pango_kwargs
        Styling forwarded to the Pango renderer when ``latex=False``: ``font``,
        ``weight``, ``slant``, ``line_spacing``, ``color_map``, ``gradient`` and the
        rest of Manim's ``Text`` arguments. Ignored under LaTeX. Defaults to
        ``None`` (no styling). Prefer :class:`~algan.mobs.text.Text`, which
        exposes these as ordinary named arguments.
    pango_color_map
        Maps the hex strings in ``pango_kwargs`` back to the Algan
        :class:`~algan.constants.color.Color` objects they came from, so glow
        and opacity survive the round trip through Pango's SVG output (a hex
        string cannot carry either). Defaults to ``None``. Set for you by
        :class:`~algan.mobs.text.Text`.
    sync_stroke_color
        Whether a glyph colored by Pango styling also gets that color on its
        border. Only has an effect when a border is actually drawn
        (``stroke_width`` above 0). Defaults to ``True``; pass an explicit
        ``stroke_color`` to keep one outline color across styled glyphs.
    **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` and to the packed
        :class:`~algan.mobs.bezier_circuit.BezierCircuitCubic` -- notably
        ``color`` (defaults to ``WHITE``), ``stroke_color``, ``stroke_width``
        (defaults to ``0``, no outline), ``location`` and ``scene``. One extra
        keyword is consumed here: ``preamble``, a string of LaTeX appended to
        Manim's default preamble, for ``\usepackage`` lines a formula needs.

    Attributes
    ----------
    character_mobs
        Lazy per-glyph views into the packed batch, in typeset order. This is
        what ``tex[i]`` and :meth:`write` animate.
    tex_strings
        The source strings as given, after list/tuple unpacking, as a tuple.
    latex
        Whether this text was typeset by LaTeX rather than by Pango.

    Examples
    --------
    A formula, one segment per term, with the middle term picked out:

    .. algan:: Example1Tex
        :save_last_frame:

        from algan import *

        equation = Tex(r"e^{i\pi}", "+", "1", "=", "0", font_size=48).spawn()
        equation.get_segment(2).color = YELLOW

        Scene.save_video()

    ``latex=False`` for prose, and a larger ``font_size``:

    .. algan:: Example2Tex
        :save_last_frame:

        from algan import *

        Tex("Not a formula", latex=False, font_size=48, color=BLUE).spawn()

        Scene.save_video()

    Individual glyphs are views, so animating one animates the original:

    .. algan:: Example3Tex
        :save_last_frame:

        from algan import *

        word = Tex("ALGAN", font_size=48).spawn()
        with Sync():
            word[0].move(UP * 0.3)
            word[4].move(DOWN * 0.3)

        Scene.save_video()
    """

    triangulated = False

    def __init__(
        self,
        *tex_strings,
        delimiter=" ",
        tex_environment=None,
        font_size=24,
        latex=True,
        pango_kwargs=None,
        pango_color_map=None,
        sync_stroke_color=True,
        **kwargs,
    ):
        if latex:
            _require_latex_toolchain()
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
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
        self.delimiter = delimiter
        self.latex = latex

        base_font_size = 48
        if self.latex:
            tex_kwargs = {
                "arg_separator": delimiter,
                "font_size": base_font_size,
            }
            if tex_environment is not None:
                tex_kwargs["tex_environment"] = tex_environment
            t = mn.MathTex(*self.tex_strings, **tex_kwargs)
        else:
            if not hasattr(mn, "Text"):
                raise RuntimeError(
                    "Pango text rendering needs the optional `manimpango` "
                    'package: `pip install "algan[pango]"`. Or use algan.Text, '
                    "which typesets through LaTeX's text mode without it."
                )
            t = mn.Text(
                delimiter.join(self.tex_strings),
                font_size=base_font_size,
                **(pango_kwargs or {}),
            )

        def maybe_flip(submob):
            x = torch.from_numpy(submob.points).to(_ANIMATION_DEVICE)
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

        if latex:
            styled_fills = None
        else:
            # Pango styling (color_map/gradient_map/gradient) lands as per-submobject fill
            # colors on the manim Text; capture them (aligned with the
            # ImageMobject-filtered glyph list) to re-apply on the batch.
            styled_fills = [
                (char.fill_color.to_hex().upper(), float(char.fill_opacity))
                for char in chars
                if not isinstance(char, mn.ImageMobject)
            ]

        triangulated_paths = [
            unsquish(maybe_flip(char), -2, 4).transpose(-3, -2)
            for char in chars
            if not isinstance(char, mn.ImageMobject)
        ]
        bezier_paths = [
            unsquish(
                torch.from_numpy(char.points).to(_ANIMATION_DEVICE).float(),
                -2,
                4,
            )
            for char in chars
            if not isinstance(char, mn.ImageMobject)
        ]
        with Off(animation_manager=kwargs["scene"].animation_manager):
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
                bezier_kwargs.setdefault("stroke_color", bezier_kwargs["color"])
                bezier_kwargs.setdefault("stroke_width", 0)
                character_batch = (
                    BezierCircuitCubic.from_batches(paths, **bezier_kwargs)
                    if paths
                    else None
                )
            self._character_batch = character_batch
            self.character_mobs = BatchedMobViewSequence(
                self._character_batch, len(paths)
            )
            # Image characters (emoji) are added as children below, and Algan
            # renders registered actors rather than walking the hierarchy, so
            # they have to join the scene along with the rest of this Text.
            self.image_mobs = [
                ImageMob(
                    char,
                    scene=kwargs["scene"],
                    add_to_scene=kwargs.get("add_to_scene", True),
                )
                for char in chars
                if isinstance(char, mn.ImageMobject)
            ]
            # Outline and texture-grid settings belong to the packed Bezier
            # child, not the non-renderable Text container.  Passing them to Mob
            # leaks into Animatable.__init__ and breaks ordinary Text
            # construction.
            mob_kwargs = dict(kwargs)
            mob_kwargs.pop("stroke_width", None)
            mob_kwargs.pop("stroke_color", None)
            mob_kwargs.pop("grid_width", None)
            mob_kwargs.pop("grid_height", None)
            super().__init__(**mob_kwargs)
            if self._character_batch is not None:
                self.add_children(self._character_batch)
            self.add_children(self.image_mobs)
            if (
                styled_fills is not None
                and not self.triangulated
                and self._character_batch is not None
            ):
                color_map = {k.upper(): v for k, v in (pango_color_map or {}).items()}
                base_color = bezier_kwargs.get("color", WHITE)
                base_glow = (
                    float(base_color.reshape(-1)[3])
                    if isinstance(base_color, torch.Tensor) and base_color.numel() >= 5
                    else 0.0
                )
                set_stroke_color = sync_stroke_color and bool(
                    bezier_kwargs.get("stroke_width", 0)
                )
                for i, (hex_c, fill_op) in enumerate(styled_fills):
                    styled = color_map.get(hex_c)
                    if styled is None:
                        if hex_c == "#FFFFFF":
                            # Untouched by color_map/gradient: keep the base color.
                            continue
                        styled = Color(hex_c, glow=base_glow, opacity=fill_op)
                    view = self.character_mobs[i]
                    view.color = styled
                    if set_stroke_color:
                        view.stroke_color = styled
            self.scale(font_size / base_font_size)

    def become(self, other_mob, *args, **kwargs):
        """Morph this text into another Mob, keeping its glyph views usable.

        As :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`, with one
        addition: because a morph can expand the
        packed glyph batch, the per-character views are rebuilt against the result, so
        indexing (``text[0]``) still works afterwards.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second by
        default).

        Parameters
        ----------
        other_mob
            The Mob to morph into. Text-to-text and text-to-bezier morphs preserve
            the tightest correspondence; other primitive families use geometric
            conversion or a dissolve according to ``strategy``.
        *args, **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` -- notably
            ``minimize_movement=True``,
            which pairs each glyph fragment with its nearest counterpart and is
            usually what you want for text.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            The morphed Mob. With the default ``detach_history=True`` this can be a
            **different object** from the one you called the method on, so use the
            return value afterwards. Character views are rebuilt when the result is
            still text.
        """
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

    def get_segment(self, index: int):
        """Get one of the text's LaTeX segments as a Mob.

        Segments are the pieces the text was constructed from, so a ``Tex`` built from
        several strings can have each one animated separately -- the usual way to
        highlight one term of an equation.

        Parameters
        ----------
        index
            Index of the segment.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            A Group of the glyphs in that segment, sharing data with this text.
        """
        return self[self.segment_starts[index] : self.segment_ends[index]]

    def __getitem__(self, item):
        """Get individual glyphs by index or slice, so ``text[0]`` works.

        The result is a view sharing this text's data, so animating it animates those
        glyphs of the original. It needs no spawning of its own.

        Parameters
        ----------
        item
            Index of a glyph, or a slice selecting several.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            A Group of the selected glyphs.
        """
        return Group([self.character_mobs[item]], scene=self.scene)

    def __len__(self):
        """Get the number of glyphs, so ``len(text)`` works.

        Returns
        -------
        int
            How many glyphs the text was rendered into. Note this counts glyphs, not
            the characters of the source string -- LaTeX markup produces neither one
            glyph per character nor a predictable ratio.
        """
        return len(self.character_mobs)

    def write(self, *args, **kwargs):
        """Animate this text appearing as if it were being hand-written.

        Each glyph's outline is traced and then filled, one glyph after another. This
        is :func:`~algan.animations.manim_animations.DrawBorderThenFill` applied to
        this text's glyphs.

        Animation
        ---------
        Recorded as an animation. Its runtime comes from ``runtime`` and
        ``lag_ratio`` rather than the enclosing context, so a long string takes longer
        to write unless you set ``runtime``.

        Parameters
        ----------
        *args, **kwargs
            Passed to
            :func:`~algan.animations.manim_animations.DrawBorderThenFill`

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This text, so calls can be chained.

        Examples
        --------

        .. algan:: Example1TextWrite

            from algan import *

            Text('Hello World!').spawn(False).write()

            Scene.save_video()
        """
        # Imported here rather than at module scope: the animations package is
        # imported after the mobs package during algan's own initialization.
        from algan.animations.manim_animations import DrawBorderThenFill

        DrawBorderThenFill(self.character_mobs, *args, **kwargs)
        return self

    def on_create(self):
        """Play the text's entrance: a fade that sweeps across the glyphs.

        Instead of the plain fade a :class:`~algan.animatable_base.mob.Mob` uses, text
        fades in as a diagonal
        wave running down and to the right, so the words appear to arrive in reading
        order.

        Animation
        ---------
        Recorded as an animation lasting **1 second**, regardless of the enclosing
        context's runtime.

        Returns
        -------
        :class:`~.Tex`
            This text, so calls can be chained.
        """
        with Seq(runtime=1, animation_manager=self.animation_manager):
            with Off(
                animation_manager=self.animation_manager
            ):  # Ensure initial state setting is not recorded as an animation
                opacity = self.opacity
                self.opacity = 0
            self._create_recursive(
                animate=False
            )  # Mark as created without immediate animation
            self.wave_color(
                None,
                direction=F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
                opacity=opacity,
            )
            for im in self.image_mobs:
                im.opacity = opacity
        return self

    def on_destroy(self):
        """Play the text's exit: a fade that sweeps across the glyphs.

        The mirror of :meth:`~.Tex.on_create` -- the glyphs fade out as a diagonal wave
        rather than all at once.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second by
        default). The despawn itself is recorded at the end of the wave, so no glyph
        disappears before the wave reaches it.

        Returns
        -------
        :class:`~.Tex`
            This text, so calls can be chained.
        """
        with Seq(animation_manager=self.animation_manager):
            self.wave_color(
                None, direction=F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1), opacity=0
            )
            for im in self.image_mobs:
                im.opacity = 0
            old_ct = self.animation_manager.context.timespan.current_time
            self.animation_manager.context.timespan.current_time = (
                self.animation_manager.context.timespan.original_end
            )
            self._destroy_recursive(animate=False)
            self.animation_manager.context.timespan.current_time = old_ct
        return self


def _to_pango_hex(color, color_map):
    """Convert a color spec (algan Color/tensor, hex/named string, or manim
    color) to an RGB hex string that manim's Pango renderer accepts.

    Algan colors carry glow/opacity channels that a plain hex cannot express,
    so the original is recorded in ``color_map`` keyed by the hex; the Tex
    constructor uses that map to restore the full algan color on glyphs after
    the SVG round trip.  Two algan colors with identical RGB collide on one
    key (the last one wins).
    """
    if isinstance(color, torch.Tensor):
        flat = color.reshape(-1)
        rgb = [int(round(min(max(float(c), 0.0), 1.0) * 255)) for c in flat[:3]]
        hex_c = "#{:02X}{:02X}{:02X}".format(*rgb)
        if isinstance(color, Color):
            color_map[hex_c] = color
        else:
            color_map[hex_c] = Color(
                tuple(float(c) for c in flat[:3]),
                glow=float(flat[3]) if flat.numel() >= 5 else 0,
                opacity=float(flat[-1]) if flat.numel() >= 4 else 1,
            )
        return hex_c
    return mn.ManimColor(color).to_hex().upper()


def _default_preamble():
    """Vendored manim's default LaTeX preamble, fetched on first use
    (deferred: importing ``algan.external_libraries.manim`` costs ~2 s of
    ``import algan`` and is only needed when a Tex has a custom preamble).
    """
    from algan.external_libraries.manim.utils.tex import _DEFAULT_PREAMBLE

    return _DEFAULT_PREAMBLE


class Text(Tex):
    """Plain (non-LaTeX) text, rendered as one packed batch of cubic bezier glyphs.

    Use :class:`~algan.mobs.text.Tex` for mathematics and this for prose. Index it to
    get individual glyphs (``text[0]``), and see
    :meth:`~algan.mobs.text.Tex.write` for the hand-written entrance.

    When Pango is available (manim's optional ``Text`` support), the styling
    arguments -- ``font``, ``weight``, ``slant``, ``line_spacing``,
    ``disable_ligatures``, and the span-level
    ``color_map``/``font_map``/``slant_map``/``weight_map``/``gradient_map``/
    ``gradient`` -- are forwarded to the Pango renderer and fully
    affect the glyphs. Color values in ``color_map``/``gradient_map``/``gradient`` may be
    algan colors (glow and opacity are preserved), hex strings, or named
    manim colors. ``weight`` accepts Pango weight names (``"THIN"``, ``"LIGHT"``,
    ``"MEDIUM"``, ``"SEMIBOLD"``, ``"BOLD"``, ``"HEAVY"``, ...), ``slant`` accepts
    ``"NORMAL"``, ``"ITALIC"``, ``"OBLIQUE"``; both are matched case-insensitively.
    Note a ``color_map`` value of pure white is
    indistinguishable from unstyled text and falls back to the base color.

    When Pango is unavailable, Algan renders the textual content through
    LaTeX text mode. Font-family and span-level styling arguments are accepted
    and retained as metadata, but cannot affect that fallback renderer.

    Parameters
    ----------
    text
        The text to display. Cast to ``str``, and tabs are expanded to
        ``tab_width`` spaces.
    fill_opacity
        Opacity of the glyph interiors, 0 for invisible and 1 for solid. Manim's
        spelling of Algan's ``opacity``. Defaults to ``1.0``.
    stroke_width
        Width of the outline drawn around each glyph, in Algan's stroke units.
        Manim means twice this by the same number; ``mn.Text`` is the
        exact-parity spelling. Defaults to ``0``, no outline.
    color
        Color of the glyphs, and of their outline if one is drawn. Accepts an
        Algan :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to ``None``, meaning
        Algan's default text color (``WHITE``).
    font_size
        Glyph size in Manim's font-size units; the glyphs are built at 48 and
        scaled by ``font_size / 48``. Defaults to ``48``, so plain ``Text`` comes
        out twice the size of plain :class:`~algan.mobs.text.Tex`.
    line_spacing
        Distance between baselines of a multi-line string, in Pango's units.
        Defaults to ``None``, meaning Pango's own spacing for the font.
    font
        Font family name, as installed on the system. Defaults to ``""``,
        meaning Pango's default family.
    slant
        ``"NORMAL"``, ``"ITALIC"`` or ``"OBLIQUE"``. Defaults to ``"NORMAL"``.
    weight
        A Pango weight name -- ``"THIN"``, ``"LIGHT"``, ``"NORMAL"``,
        ``"MEDIUM"``, ``"SEMIBOLD"``, ``"BOLD"``, ``"HEAVY"``, and the rest.
        Defaults to ``"NORMAL"``.
    color_map
        Maps a substring to the color its glyphs take. Color
        values may be Algan colors (glow and opacity survive), hex strings, or
        named Manim colors. A value of pure white is indistinguishable from
        unstyled text and falls back to the base color. Defaults to ``None``.
    font_map
        Maps a substring to a font family. Defaults to ``None``.
    gradient_map
        Maps a substring to a tuple of colors to fade between across it.
        Defaults to ``None``.
    slant_map
        Maps a substring to a slant name. Defaults to ``None``.
    weight_map
        Maps a substring to a weight name. Defaults to ``None``.
    gradient
        A tuple of colors faded across the whole string. Defaults to ``None``,
        one flat color.
    tab_width
        How many spaces a tab in ``text`` expands to. Defaults to ``4``.
    warn_missing_font
        Whether to log a warning when ``font`` is not installed. Defaults to
        ``True``.
    height
        Scale the finished text uniformly so it is this tall, in world units.
        Defaults to ``None``, its natural size for ``font_size``.
    width
        Scale the finished text uniformly so it is this wide, in world units.
        Applied after ``height``, so passing both leaves the width matched and
        the height wherever the aspect ratio puts it. Defaults to ``None``.
    center
        Whether to move the finished text to the world origin. Defaults to
        ``True``; pass ``False`` to keep the position ``location`` gave it.
    disable_ligatures
        Whether to render each character separately rather than letting the font
        combine pairs such as "fi". Slower, but it makes ``text[i]`` line up
        with the i-th character. Defaults to ``False``.
    use_svg_cache
        Accepted for Manim parity and has no effect: Algan caches the glyph
        geometry itself, keyed on the source, whatever this is set to. Defaults
        to ``False``.
    **kwargs
        Passed to :class:`~algan.mobs.text.Tex` -- notably ``location``,
        ``stroke_color``, ``scene`` and ``add_to_scene``.

    Examples
    --------
    Plain prose, and the same words with one span colored:

    .. algan:: Example1Text
        :save_last_frame:

        from algan import *

        Text("Hello, world", font_size=36).move(UP * 0.5).spawn()
        Text("Hello, world", font_size=36,
             color_map={"world": BLUE}).move(DOWN * 0.5).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        text,
        fill_opacity=1.0,
        stroke_width=0,
        color=None,
        font_size=48,
        line_spacing=None,
        font="",
        slant="NORMAL",
        weight="NORMAL",
        color_map=None,
        font_map=None,
        gradient_map=None,
        slant_map=None,
        weight_map=None,
        gradient=None,
        tab_width=4,
        warn_missing_font=True,
        height=None,
        width=None,
        center=True,
        disable_ligatures=False,
        use_svg_cache=False,
        **kwargs,
    ):
        self.text = str(text).expandtabs(tab_width)
        self.font = font
        # Pango wants these upper-cased ("BOLD", "ITALIC"); accept whatever
        # case the caller wrote and normalize at the boundary rather than
        # changing what Pango is sent.
        slant = slant.upper() if isinstance(slant, str) else slant
        weight = weight.upper() if isinstance(weight, str) else weight
        self.slant = slant
        self.weight = weight
        self.line_spacing = line_spacing
        self.color_map, self.font_map = color_map, font_map
        self.gradient_map = gradient_map
        self.slant_map, self.weight_map = slant_map, weight_map
        self.gradient = gradient
        self.disable_ligatures = disable_ligatures
        self.use_svg_cache = use_svg_cache
        explicit_stroke_color = "stroke_color" in kwargs
        self._write_uses_default_pango_border = not explicit_stroke_color
        kwargs.setdefault("opacity", fill_opacity)
        kwargs.setdefault("stroke_width", stroke_width)
        if color is not None:
            kwargs.setdefault("color", color)
            kwargs.setdefault("stroke_color", color)

        if hasattr(mn, "Text"):
            pango_kwargs = {
                "font": font,
                "slant": slant,
                "weight": weight,
                # Pango spells "use the font's own spacing" as -1.
                "line_spacing": -1 if line_spacing is None else line_spacing,
                "warn_missing_font": warn_missing_font,
                "disable_ligatures": disable_ligatures,
            }
            # ``pango_colors`` is the hex lookup handed to Pango, not the
            # user's substring -> colour ``color_map``.
            pango_colors = {}
            if font_map:
                pango_kwargs["t2f"] = dict(font_map)
            if slant_map:
                pango_kwargs["t2s"] = {
                    k: v.upper() if isinstance(v, str) else v
                    for k, v in slant_map.items()
                }
            if weight_map:
                pango_kwargs["t2w"] = {
                    k: v.upper() if isinstance(v, str) else v
                    for k, v in weight_map.items()
                }
            if color_map:
                pango_kwargs["t2c"] = {
                    k: _to_pango_hex(v, pango_colors) for k, v in color_map.items()
                }
            if gradient_map:
                pango_kwargs["t2g"] = {
                    k: tuple(_to_pango_hex(c, pango_colors) for c in v)
                    for k, v in gradient_map.items()
                }
            if gradient:
                pango_kwargs["gradient"] = tuple(
                    _to_pango_hex(c, pango_colors) for c in gradient
                )
            super().__init__(
                self.text,
                font_size=font_size,
                latex=False,
                pango_kwargs=pango_kwargs,
                pango_color_map=pango_colors,
                sync_stroke_color=not explicit_stroke_color,
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
        with Off(animation_manager=self.animation_manager):
            if height is not None:
                current = self.get_length_in_direction(UP)
                if float(current.reshape(-1)[0]) > 0:
                    self.scale(float(height) / float(current.reshape(-1)[0]))
            if width is not None:
                current = self.get_length_in_direction(RIGHT)
                if float(current.reshape(-1)[0]) > 0:
                    self.scale(float(width) / float(current.reshape(-1)[0]))
            if center:
                self.move_to(ORIGIN)

    def write(self, *args, **kwargs):
        """Write this plain text with Manim's default Pango outline style.

        Manim's ``Text`` keeps a white stroke color when only its fill color is
        changed, so a stroke-free colored word is first traced in white. ``Tex``
        instead traces in its own color. An explicit Algan ``stroke_color`` keeps
        that custom outline behavior.

        Spawn the text without its ordinary entrance first:
        ``Text(...).spawn(False).write()``.
        """
        if self._write_uses_default_pango_border and "stroke_color" not in kwargs:
            kwargs["stroke_color"] = WHITE
        return super().write(*args, **kwargs)


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
    """Plain text from a Pango-markup source, with the markup stripped out.

    Manim's markup syntax is accepted so a ported script keeps running, but the
    tags never reach the glyph renderer: ``<br/>`` becomes a line break, HTML
    entities are unescaped, and every other tag is deleted before the text is
    typeset. This happens whether or not the optional Pango renderer is
    available, so a ``<span foreground='red'>`` span comes out in the Mob's own
    color, not red. The source you passed is kept on ``original_text``.

    To color or restyle part of a string, use :class:`~algan.mobs.text.Text`'s
    ``color_map`` / ``font_map`` / ``slant_map`` / ``weight_map`` /
    ``gradient_map`` arguments instead -- those do reach the renderer.

    Parameters
    ----------
    text
        The marked-up source. Tags are stripped, ``<br/>`` becomes a newline and
        entities such as ``&amp;`` are unescaped; what remains is typeset.
    justify
        Accepted for Manim parity and has no effect on the rendered text; it is
        stored on the Mob as ``justify``. Defaults to ``False``.
    fill_opacity, stroke_width, color, font_size, line_spacing
        As :class:`~algan.mobs.text.Text`, with the same defaults.
    font, slant, weight, gradient, disable_ligatures, warn_missing_font
        As :class:`~algan.mobs.text.Text`, with the same defaults.
    tab_width, height, width, center
        As :class:`~algan.mobs.text.Text`, with the same defaults. All of these
        are redeclared here only so this constructor's signature matches
        Manim's.
    **kwargs
        Passed to :class:`~algan.mobs.text.Text`.

    Attributes
    ----------
    original_text
        The markup source exactly as given, before stripping.

    Examples
    --------
    A markup string, and the ``Text`` spelling that actually colors the span:

    .. algan:: Example1MarkupText
        :save_last_frame:

        from algan import *

        MarkupText("<b>bold</b> markup is stripped",
                   font_size=32).move(UP * 0.5).spawn()
        Text("color_map is not", font_size=32,
             color_map={"not": BLUE}).move(DOWN * 0.5).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        text,
        fill_opacity=1,
        stroke_width=0,
        color=None,
        font_size=48,
        line_spacing=None,
        font="",
        slant="NORMAL",
        weight="NORMAL",
        justify=False,
        gradient=None,
        tab_width=4,
        height=None,
        width=None,
        center=True,
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
            center=center,
            disable_ligatures=disable_ligatures,
            warn_missing_font=warn_missing_font,
            **kwargs,
        )


class Paragraph(Group):
    """A group of individually addressable text lines."""

    def __init__(self, *text, line_spacing=None, alignment=None, **kwargs):
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        add_to_scene = kwargs.pop("add_to_scene", True)
        lines = []
        for part in text:
            lines.extend(str(part).split("\n"))
        if not lines:
            lines = [""]
        # The lines are this Paragraph's only geometry -- the Group itself has no
        # render primitives -- so they must join the scene whenever it does.
        mobs = [Text(line, add_to_scene=add_to_scene, **kwargs) for line in lines]
        super().__init__(*mobs, add_to_scene=add_to_scene)
        if mobs:
            buffer = 0.2 if line_spacing is None else line_spacing
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
                align_to=align_direction,
            )
        self.lines_text = lines
        self.chars = self.children

    def set_all_lines_alignments(self, alignment: str):
        """Re-align every line of the paragraph.

        The paragraph is rebuilt with the new alignment and morphed into, so the lines
        slide into their new positions.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second by
        default): the glyphs travel to their new positions.

        Parameters
        ----------
        alignment
            Alignment to apply to every line, e.g. ``"left"``, ``"center"``,
            ``"right"``.

        Returns
        -------
        :class:`~.Paragraph`
            The re-aligned paragraph.
        """
        replacement = Paragraph(
            *self.lines_text,
            scene=self.scene,
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
        from algan.mobs.shapes_2d import Circle, SurroundingRectangle

        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        add_to_scene = kwargs.pop("add_to_scene", True)
        if code_string is None:
            if code_file is None:
                raise ValueError("either code_file or code_string must be provided")
            code_string = pathlib.Path(code_file).read_text(encoding="utf-8")
        source_lines = str(code_string).expandtabs(tab_width).splitlines() or [""]
        paragraph_config = dict(paragraph_config or {})
        paragraph_config.update(kwargs)
        # Every part below is this Code Mob's visible geometry, and only
        # registered actors render, so each one joins the scene along with it.
        self.code = Paragraph(
            *source_lines,
            alignment="left",
            add_to_scene=add_to_scene,
            **paragraph_config,
        )
        mobs = [self.code]
        self.line_numbers = None
        if add_line_numbers:
            self.line_numbers = Paragraph(
                *(
                    str(i)
                    for i in range(
                        line_numbers_from, line_numbers_from + len(source_lines)
                    )
                ),
                alignment="right",
                add_to_scene=add_to_scene,
                **paragraph_config,
            )
            with Off(animation_manager=kwargs["scene"].animation_manager):
                self.line_numbers.move_next_to(self.code, LEFT, buffer=0.2)
            mobs.insert(0, self.line_numbers)
        super().__init__(*mobs, add_to_scene=add_to_scene)

        self.background_mobject = None
        background_config = dict(background_config or {})
        if background == "rectangle":
            self.background_mobject = SurroundingRectangle(
                self,
                scene=self.scene,
                add_to_scene=add_to_scene,
                **background_config,
            )
        elif background == "window":
            frame = SurroundingRectangle(
                self,
                scene=self.scene,
                add_to_scene=add_to_scene,
                **background_config,
            )
            dots = Group(
                *[
                    Circle(radius=0.04, scene=self.scene, add_to_scene=add_to_scene)
                    for _ in range(3)
                ],
                scene=self.scene,
                add_to_scene=add_to_scene,
            )
            with Off(animation_manager=self.animation_manager):
                dots.arrange_in_line(RIGHT, buffer=0.08)
                dots.move_next_to(frame.get_boundary_point(UP), DOWN, buffer=0.08)
            self.background_mobject = Group(
                frame, dots, scene=self.scene, add_to_scene=add_to_scene
            )
        elif background not in {None, False}:
            raise ValueError("background must be 'rectangle', 'window', or None")
        if self.background_mobject is not None:
            self.add(self.background_mobject)

        self.language = language
        self.formatter_style = formatter_style
        self.code_string = str(code_string)
        self.code_file = code_file
