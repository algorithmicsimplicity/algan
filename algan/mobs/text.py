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

import os
import threading
from typing import NamedTuple

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
_LATEX_TOOLCHAIN_FOUND = None


#: How :class:`Text`'s LaTeX fallback spells the characters TeX reads as
#: markup, or would set as something else. The template's text encoding (OT1)
#: has no ASCII ``~``, ``<``, ``>`` or ``|`` in its serif and sans faces --
#: ``<`` comes out as ``¡`` -- so those borrow symbol-font glyphs.
_LATEX_ESCAPES = {char: "\\" + char for char in "#$%&_{}"}
_LATEX_ESCAPES.update(
    {
        "\\": r"\textbackslash{}",
        "^": r"\textasciicircum{}",
        "~": r"$\sim$",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
        "|": r"\textbar{}",
    }
)
#: The typewriter face has every printable ASCII glyph in its ASCII slot, so
#: the characters TeX reserves are reached by slot number -- the trailing
#: space ends the number -- and the apostrophe and backtick by the straight
#: quote and grave the ``upquote`` package uses, rather than curly quotes. Its
#: ``~`` is drawn at accent height, and is lowered to where a monospace font
#: puts it.
_LATEX_TYPEWRITER_ESCAPES = {char: f"\\char{ord(char)} " for char in "\\{}#$%&_^"}
_LATEX_TYPEWRITER_ESCAPES.update(
    {"~": r"\raisebox{-2.9pt}{\char126}", "'": r"\char13 ", "`": r"\char18 "}
)
#: Adjacent pairs Computer Modern fuses into a different character (``--``
#: into an en dash, ``!`` and a backtick into ``¡``); always kept apart.
_LATEX_TEXT_LIGATURES = {("-", "-"), ("`", "`"), ("'", "'"), ("!", "`"), ("?", "`")}
#: Purely typographic ligatures, kept apart only for ``disable_ligatures``.
_LATEX_F_LIGATURES = {("f", "f"), ("f", "i"), ("f", "l")}


def _escape_plain_text(text, family="rm", ligatures=True):
    r"""LaTeX text-mode source that sets ``text``'s characters literally.

    ``family`` is ``"rm"``, ``"sf"`` or ``"tt"``. Every whitespace character
    becomes a control space, so runs of spaces (code indentation) keep their
    width. Line breaks are the caller's: a ``\\`` inside ``\text{}`` is an
    error, so see :func:`_latex_text_layout`.
    """
    escapes = _LATEX_TYPEWRITER_ESCAPES if family == "tt" else _LATEX_ESCAPES
    parts = []
    previous = ""
    for char in str(text):
        pair = (previous, char)
        if pair in _LATEX_TEXT_LIGATURES or (
            not ligatures and pair in _LATEX_F_LIGATURES
        ):
            parts.append("{}")
        # One pass: inserted TeX commands must never be escaped a second time.
        parts.append("\\ " if char.isspace() else escapes.get(char, char))
        previous = char
    return "".join(parts)


def _pango_available():
    """Whether :class:`Text` renders through Pango rather than LaTeX.

    The vendored Manim exports ``Text`` only when ``manimpango`` imports
    (``manim.PANGO_AVAILABLE``), so the export is the test -- never
    ``import manimpango``.
    """
    return hasattr(mn, "Text")


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
    global _LATEX_TOOLCHAIN_FOUND
    import shutil

    # ``shutil.which`` stats every PATH entry for every extension, ~15 ms, and
    # this runs for every Tex. A success is remembered for the PATH it was
    # found on; a failure is not, so a TeX installed mid-session is picked up.
    searched = (shutil.which, os.environ.get("PATH"))
    if searched == _LATEX_TOOLCHAIN_FOUND:
        return
    missing = [name for name in _LATEX_BINARIES if shutil.which(name) is None]
    if not missing:
        _LATEX_TOOLCHAIN_FOUND = searched
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
        "Text(...) renders prose through Pango and needs none of this -- but "
        "only where manimpango is installed, which is automatic on Windows and "
        'macOS and `pip install "algan[pango]"` on Linux. '
        "Tex(..., latex=False) does the same."
    )


class _TexGlyphs(NamedTuple):
    """What a typeset formula contributes to a :class:`Tex`: its outlines.

    ``tex_keys`` and ``segment_sizes`` are the typeset segments and how many
    glyphs each holds; ``points`` is each outline glyph's Manim point array
    and ``is_svg_path`` whether it came from an SVG path (which decides the
    triangulated path's orientation). ``styled_fills`` is each Pango glyph's
    fill color and opacity (None for LaTeX).
    """

    tex_keys: tuple | None
    segment_sizes: tuple
    points: tuple
    is_svg_path: tuple
    styled_fills: tuple | None


#: Typeset outlines by source, so a formula is typeset and parsed once per
#: process. Manim's MathTex rebuilds a glyph tree from the (disk-cached) SVG
#: each time, ~50 ms for a short formula, and a ``DecimalNumber`` typesets
#: ``"0123456789"`` once per digit slot.
_TEX_GLYPH_MEMO: dict = {}
_TEX_GLYPH_MEMO_SIZE = 4096
_TEX_GLYPH_MEMO_LOCK = threading.Lock()


def _tex_glyph_memo_key(
    tex_strings, delimiter, tex_environment, latex, pango_kwargs, kwargs
):
    """Everything the typeset outlines depend on, or None to typeset afresh.

    For LaTeX: the strings, the separator and environment they are joined
    with, and the TeX template MathTex falls back to
    (``config["tex_template"]``), which a script may replace or edit between
    two Tex. For Pango: the joined string and every Pango option, and the
    Pango backend's version.
    """
    if "tex_template" in kwargs:
        return None
    try:
        if latex:
            template = mn.config["tex_template"]
            style_key = (
                template.body,
                template.tex_compiler,
                template.output_format,
            )
        else:
            from algan.utils.manim_svg_cache import _manimpango_cache_version

            style_key = (
                _manimpango_cache_version(),
                _frozen_pango_options(pango_kwargs or {}),
            )
    except Exception:  # noqa: BLE001 - an unkeyable input is never memoized
        return None
    return (bool(latex), tuple(tex_strings), delimiter, tex_environment, style_key)


def _frozen_pango_options(value):
    """A hashable copy of Pango's options; raises on anything not plain data."""
    if isinstance(value, dict):
        return tuple(
            sorted((str(k), _frozen_pango_options(v)) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_pango_options(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unkeyable Pango option {type(value).__name__}")


def _remember_tex_glyphs(key, glyphs):
    with _TEX_GLYPH_MEMO_LOCK:
        if len(_TEX_GLYPH_MEMO) >= _TEX_GLYPH_MEMO_SIZE:
            # Drop the oldest half: insertion order is age order in a dict.
            for stale in list(_TEX_GLYPH_MEMO)[: _TEX_GLYPH_MEMO_SIZE // 2]:
                del _TEX_GLYPH_MEMO[stale]
        _TEX_GLYPH_MEMO[key] = glyphs


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
    from algan.utils.path_utils import _ensure_writable_directory

    for tex_dir in _configure_manim_dirs(config, create=False):
        # The write probe creates and deletes a file, ~2 ms a directory, and
        # this runs for every Tex. A directory that passed it once this
        # process and still exists is not probed again.
        key = str(tex_dir)
        if key in _VERIFIED_TEX_DIRS and os.path.isdir(key):
            continue
        _ensure_writable_directory(
            tex_dir,
            purpose="text/LaTeX cache",
            remedy="Set SETTINGS.paths.cache_directory to a writable directory "
            "or set ALGAN_CACHE_DIR before starting Python.",
        )
        _VERIFIED_TEX_DIRS.add(key)


_VERIFIED_TEX_DIRS: set = set()


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
        memo_key = _tex_glyph_memo_key(
            self.tex_strings, delimiter, tex_environment, latex, pango_kwargs, kwargs
        )
        glyphs = _TEX_GLYPH_MEMO.get(memo_key) if memo_key is not None else None
        if glyphs is not None:
            t = None
        elif self.latex:
            tex_kwargs = {
                "arg_separator": delimiter,
                "font_size": base_font_size,
            }
            if tex_environment is not None:
                tex_kwargs["tex_environment"] = tex_environment
            if "tex_template" in kwargs:
                tex_kwargs["tex_template"] = kwargs["tex_template"]
            t = mn.MathTex(*self.tex_strings, **tex_kwargs)
        else:
            if not _pango_available():
                raise RuntimeError(
                    "Pango text rendering needs the optional `manimpango` "
                    'package: `pip install "algan[pango]"`. Or use algan.Text, '
                    "which typesets through LaTeX's text mode without it."
                )
            t = mn.Text(
                delimiter.join(self.tex_strings),
                font_size=base_font_size,
                # Pango defaults this off. Use Algan's existing content-keyed
                # SVG recipe cache, just as MathTex does, for repeated labels.
                **{"use_svg_cache": True, **(pango_kwargs or {})},
            )

        if latex:
            if glyphs is None:
                sub_mobs = [_.submobjects for _ in t.submobjects]
                chars = [x for group in sub_mobs for x in group]
                outlines = [c for c in chars if not isinstance(c, mn.ImageMobject)]
                # Actual typeset segments can differ from constructor arguments
                # when double braces isolate terms inside a single source string.
                glyphs = _TexGlyphs(
                    tuple(part.tex_string for part in t.submobjects),
                    tuple(len(_) for _ in sub_mobs),
                    tuple(c.points.copy() for c in outlines),
                    tuple(isinstance(c, mn.VMobjectFromSVGPath) for c in outlines),
                    None,
                )
                if memo_key is not None and len(outlines) == len(chars):
                    _remember_tex_glyphs(memo_key, glyphs)
            else:
                # Only formulas without image glyphs are memoized.
                chars = []
            self._matching_tex_keys = glyphs.tex_keys
            self.num_mobs_per_segment = torch.tensor(list(glyphs.segment_sizes))
            self.segment_ends = self.num_mobs_per_segment.cumsum(0)
            self.segment_starts = self.segment_ends - self.num_mobs_per_segment
            # Copied: a memoized outline is shared by every later Tex of the
            # same source, and a path built on it must not alias it.
            outline_points = [points.copy() for points in glyphs.points]
            outline_is_svg_path = glyphs.is_svg_path
        else:
            if glyphs is None:
                chars = t.submobjects
                outlines = [c for c in chars if not isinstance(c, mn.ImageMobject)]
                glyphs = _TexGlyphs(
                    None,
                    (len(chars),),
                    tuple(c.points.copy() for c in outlines),
                    tuple(isinstance(c, mn.VMobjectFromSVGPath) for c in outlines),
                    # Pango styling (color_map/gradient_map/gradient) lands as
                    # per-submobject fill colors on the manim Text; capture them
                    # (aligned with the ImageMobject-filtered glyph list) to
                    # re-apply on the batch.
                    tuple(
                        (c.fill_color.to_hex().upper(), float(c.fill_opacity))
                        for c in outlines
                    ),
                )
                if memo_key is not None and len(outlines) == len(chars):
                    _remember_tex_glyphs(memo_key, glyphs)
            else:
                chars = []
            self.num_mobs_per_segment = torch.tensor(list(glyphs.segment_sizes))
            self.segment_ends = self.num_mobs_per_segment.cumsum(0)
            self.segment_starts = self.segment_ends - self.num_mobs_per_segment
            outline_points = [points.copy() for points in glyphs.points]
            outline_is_svg_path = glyphs.is_svg_path

        def maybe_flip(points, is_svg_path):
            x = torch.from_numpy(points).to(_ANIMATION_DEVICE)
            if (not latex) or (not is_svg_path):
                return x.flip(-2)
            return x

        styled_fills = None if latex else list(glyphs.styled_fills)

        triangulated_paths = [
            unsquish(maybe_flip(points, is_svg_path), -2, 4).transpose(-3, -2)
            for points, is_svg_path in zip(outline_points, outline_is_svg_path)
        ]
        bezier_paths = [
            unsquish(
                torch.from_numpy(points).to(_ANIMATION_DEVICE).float(),
                -2,
                4,
            )
            for points in outline_points
        ]
        # Templates configure typesetting, not Mob or glyph geometry. Keep the
        # keyword until after typesetting so any glyph-memo lookup can see it.
        kwargs.pop("tex_template", None)
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
        highlight one term of an equation. Calling ``spawn()`` on the segment
        reveals only its glyphs; other segments can be spawned later.

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

        Each glyph fades in to the opacity it had before the text spawned, as
        every Mob's entrance does, so a glyph hidden beforehand stays hidden.

        Animation
        ---------
        Recorded as an animation lasting **1 second**, regardless of the enclosing
        context's runtime.

        Returns
        -------
        :class:`~.Tex`
            This text, so calls can be chained.
        """
        entering = self.__dict__.pop("_entering_rows", None) or {}
        with Seq(runtime=1, animation_manager=self.animation_manager):
            with Off(
                animation_manager=self.animation_manager
            ):  # Ensure initial state setting is not recorded as an animation
                opacity = self.opacity
                # Read before the zeroing below: the wave used to fade every
                # glyph to the text's own opacity, which re-showed any glyph
                # hidden before the spawn.
                parts = self._wave_pulsed_parts()
                targets = {id(part): part.opacity for part in parts}
                image_targets = [im.opacity for im in self.image_mobs]
                self.opacity = 0
                # Glyphs a selection spawned earlier are on screen already:
                # only the rest enter (see _spawn_packed_rows below).
                for part in parts:
                    rows = entering.get(part.id)
                    target = targets[id(part)]
                    if rows is not None and rows.shape[0] == target.shape[-2]:
                        rows = rows.to(target.device).view(1, -1, 1)
                        part.set_non_recursive(opacity=torch.where(rows, 0, target))
            self._create_recursive(
                animate=False
            )  # Mark as created without immediate animation
            self.wave_color(
                None,
                direction=F.normalize(RIGHT * 1.5 + DOWN, p=2, dim=-1),
                opacity=lambda part: targets.get(id(part), opacity),
            )
            for im, target in zip(self.image_mobs, image_targets):
                im.opacity = target
        return self

    def _spawn_packed_rows(self, animate):
        """Hand glyphs still waiting on a partial spawn to this text's entrance.

        After a glyph selection has spawned, spawning the whole text reveals
        the remaining glyphs -- and :meth:`on_create` then fades every glyph
        in. Animating the reveal as well played two entrances over each other,
        so the reveal is made instant here and :meth:`on_create` fades in just
        the glyphs it revealed, leaving the ones already on screen alone.
        """
        if not animate or self.is_spawned() or self.__dict__.get("on_create"):
            return super()._spawn_packed_rows(animate)
        pending = self.scene.timeline_manager._pending_packed_spawns
        ids = {mob.id for mob in self.get_descendants()}
        self._entering_rows = {
            mob_id: remaining.clone()
            for mob_id, (_, remaining) in pending.items()
            if mob_id in ids
        }
        return super()._spawn_packed_rows(False)

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


#: The Pango style and weight names, as ``manimpango`` spells them. Read once
#: at first use rather than at import, since manimpango is imported lazily.
_PANGO_NAMES: dict[str, tuple[str, ...]] = {}


def _pango_names(kind: str) -> tuple[str, ...]:
    """Every name Pango accepts for ``slant`` or ``weight``.

    Empty when manimpango is not installed -- the caller then has no list to
    validate against and must let the value through.
    """
    if kind not in _PANGO_NAMES:
        try:
            import manimpango
        except ImportError:
            # Not an error: without Pango, Text typesets through LaTeX text
            # mode, where slant and weight are retained as metadata and
            # cannot reach a renderer at all. Raising here instead made the
            # documented fallback unreachable -- ``slant``/``weight`` have
            # string defaults, so every `Text(...)` normalized them and died
            # on the import before it could fall back to anything.
            return ()
        source = manimpango.Style if kind == "slant" else manimpango.Weight
        _PANGO_NAMES[kind] = tuple(
            sorted(n for n in dir(source) if not n.startswith("_"))
        )
    return _PANGO_NAMES[kind]


def _pango_style(kind: str, value):
    """Normalize a ``slant``/``weight`` name, rejecting one Pango does not know.

    Pango takes these as strings, and an unknown one is not an error there: it
    silently falls back to the default, so ``Text("hi", weight="BOLDER")``
    renders in the regular face and nothing anywhere says why. Case is the
    caller's to write however they like; the name has to be real.

    With no Pango installed there is no authority on which names are real, and
    nothing the value could affect; it is upper-cased and accepted.
    """
    if not isinstance(value, str):
        return value
    upper = value.upper()
    names = _pango_names(kind)
    if not names:
        return upper
    if upper not in names:
        import difflib

        close = difflib.get_close_matches(upper, names, n=1)
        did_you_mean = f" Did you mean {close[0]!r}?" if close else ""
        raise AlganConfigurationError(
            f"{kind} must be one of {', '.join(repr(n) for n in names)}; "
            f"got {value!r}.{did_you_mean}"
        )
    return upper


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


# --------------------------------------------------------------------------
# Text without Pango: LaTeX text mode, laid out to stand in for Pango's.
#
# The sizes are measured, at font_size 48. Pango (DejaVu, Linux's default
# family, whose serif, sans and mono capitals are all one height) sets a
# capital 0.4859 world units tall and puts baselines 0.5 * (1 + line_spacing)
# apart. LaTeX's 10 pt Computer Modern, through the same MathTex at 48, makes
# 1 pt 0.049785 world units -- a capital 0.3402, so Text was 70% of Pango's
# size. The fallback scales by the ratio of capital heights for its family.
# --------------------------------------------------------------------------

_PANGO_CAP_HEIGHT_AT_48 = 0.4859
_PANGO_LINE_PITCH_AT_48 = 0.5
#: ``line_spacing=None`` is passed to Pango as ``-1``, which Manim's Text
#: turns into 0.3.
_PANGO_DEFAULT_LINE_SPACING = 0.3
_LATEX_WORLD_UNITS_PER_PT_AT_48 = 0.049785
#: Capital height of each Computer Modern family at 10 pt, from the
#: cmr10/cmss10/cmtt10 font metrics.
_LATEX_CAP_HEIGHT_PT = {"rm": 6.83333, "sf": 6.94444, "tt": 6.11111}
#: ``align*``'s distance between baselines: ``\baselineskip`` plus ``\jot``.
_LATEX_ALIGN_PITCH_PT = 15.0

#: Font-name fragments the fallback reads as a family, monospace first so
#: "DejaVu Sans Mono" is typewriter rather than sans. Anything else -- the
#: default ``font=""`` included -- is serif.
_MONOSPACE_FONT_HINTS = (
    "mono",
    "courier",
    "consola",
    "menlo",
    "monaco",
    "inconsolata",
    "typewriter",
    "code",
    "console",
    "terminal",
    "fixed",
)
_SANS_FONT_HINTS = (
    "sans",
    "arial",
    "helvetica",
    "verdana",
    "tahoma",
    "calibri",
    "segoe",
    "roboto",
    "ubuntu",
    "futura",
    "lato",
    "trebuchet",
    "avenir",
    "myriad",
    "franklin",
    "gill",
)
_LATEX_FAMILY_DECLARATIONS = {"rm": "", "sf": r"\sffamily ", "tt": r"\ttfamily "}
_PANGO_BOLD_WEIGHTS = frozenset(
    {"SEMIBOLD", "BOLD", "ULTRABOLD", "HEAVY", "ULTRAHEAVY"}
)
_LATEX_SHAPE_DECLARATIONS = {"ITALIC": r"\itshape ", "OBLIQUE": r"\slshape "}

_LATEX_TEXT_FALLBACK_WARNED = False


def _latex_family(font):
    """The Computer Modern family (``rm``/``sf``/``tt``) nearest a font name."""
    name = str(font or "").lower()
    if any(hint in name for hint in _MONOSPACE_FONT_HINTS):
        return "tt"
    if any(hint in name for hint in _SANS_FONT_HINTS):
        return "sf"
    return "rm"


def _text_spans(key, text):
    """Where a ``color_map``-style key applies in ``text``, as Manim reads it.

    ``"[a:b]"`` is a slice of character indices (newlines and spaces count);
    anything else is a substring, matched at every occurrence.
    """
    import re

    sliced = re.match(r"\[([0-9\-]{0,}):([0-9\-]{0,})\]", key)
    if sliced:
        start = int(sliced.group(1)) if sliced.group(1) != "" else 0
        end = int(sliced.group(2)) if sliced.group(2) != "" else len(text)
        start = len(text) + start if start < 0 else start
        end = len(text) + end if end < 0 else end
        return [(start, end)]
    spans = []
    index = text.find(key) if key else -1
    while index != -1:
        spans.append((index, index + len(key)))
        index = text.find(key, index + len(key))
    return spans


class _LatexTextLayout(NamedTuple):
    """A :class:`Text` laid out for LaTeX: one segment per styled run."""

    tex_strings: tuple
    #: Per segment, the index into the caller's colors it takes, or None.
    segment_colors: tuple
    #: Multiplies ``font_size`` so capitals match Pango's height.
    scale: float


def _latex_text_layout(
    text,
    font="",
    slant="NORMAL",
    weight="NORMAL",
    line_spacing=None,
    disable_ligatures=False,
    font_map=None,
    slant_map=None,
    weight_map=None,
    char_colors=None,
):
    r"""LaTeX source standing in for Pango's layout of ``text``.

    Each line is one row of ``align*``, left-aligned and spaced as Pango
    spaces it; each run of one style within a line is one ``\text{}`` and one
    Tex segment, so the glyphs a style applies to are exactly that segment's,
    however many glyphs LaTeX makes of the characters. Whitespace joins the run
    around it and a line with no glyphs joins a neighboring segment, because a
    segment without glyphs has nothing to locate it in the typeset SVG.

    ``char_colors`` gives, per character of ``text``, an index into the
    caller's colors (or None for the base color); runs split where it changes.
    """
    n = len(text)
    base_family = _latex_family(font)
    family = [base_family] * n
    bold = [str(weight).upper() in _PANGO_BOLD_WEIGHTS] * n
    shape = [_LATEX_SHAPE_DECLARATIONS.get(str(slant).upper(), "")] * n
    for mapping, values, convert in (
        (font_map, family, _latex_family),
        (weight_map, bold, lambda w: str(w).upper() in _PANGO_BOLD_WEIGHTS),
        (slant_map, shape, lambda s: _LATEX_SHAPE_DECLARATIONS.get(str(s).upper(), "")),
    ):
        for key, value in (mapping or {}).items():
            for start, end in _text_spans(key, text):
                values[start:end] = [convert(value)] * len(values[start:end])
    colors = list(char_colors) if char_colors is not None else [None] * n

    scale = _PANGO_CAP_HEIGHT_AT_48 / (
        _LATEX_WORLD_UNITS_PER_PT_AT_48 * _LATEX_CAP_HEIGHT_PT[base_family]
    )
    spacing = (
        _PANGO_DEFAULT_LINE_SPACING
        if line_spacing is None or line_spacing == -1
        else line_spacing
    )
    pitch_pt = (
        _PANGO_LINE_PITCH_AT_48
        * (1 + float(spacing))
        / (_LATEX_WORLD_UNITS_PER_PT_AT_48 * scale)
    )
    row_break = rf"\\[{pitch_pt - _LATEX_ALIGN_PITCH_PT:.3f}pt]"
    lines = text.split("\n")

    def run_source(indices, style):
        run_family, run_bold, run_shape, _ = style
        declarations = (
            _LATEX_FAMILY_DECLARATIONS[run_family]
            + (r"\bfseries " if run_bold else "")
            + run_shape
        )
        content = _escape_plain_text(
            "".join(text[i] for i in indices),
            run_family,
            ligatures=not disable_ligatures,
        )
        return rf"\text{{{declarations}{content}}}"

    segments = []  # [source, color index]
    pending = ""  # glyph-less source waiting for the next segment
    start = 0
    for row, line in enumerate(lines):
        if len(lines) > 1:
            pending += "&" if row == 0 else row_break + "&"
        runs = []  # [style, character indices]
        leading = []
        for i in range(start, start + len(line)):
            if text[i].isspace():
                (runs[-1][1] if runs else leading).append(i)
                continue
            style = (family[i], bold[i], shape[i], colors[i])
            if runs and runs[-1][0] == style:
                runs[-1][1].append(i)
            else:
                runs.append([style, [i]])
        if runs:
            runs[0][1][:0] = leading
            for style, indices in runs:
                segments.append([pending + run_source(indices, style), style[3]])
                pending = ""
        elif leading:
            first = leading[0]
            style = (family[first], bold[first], shape[first], None)
            pending += run_source(leading, style)
        start += len(line) + 1
        if pending and segments:
            segments[-1][0] += pending
            pending = ""
    if not segments:
        segments.append([pending or r"\text{}", None])
    return _LatexTextLayout(
        tuple(source for source, _ in segments),
        tuple(color for _, color in segments),
        scale,
    )


def _latex_text_colors(text, color_map, gradient, gradient_map, base_color):
    """Per-character colors for the LaTeX fallback, as Pango would apply them.

    Returns ``(palette, char_colors)``: the Algan colors used, and for each
    character of ``text`` an index into them, or None for the base color.
    ``gradient`` fades across every character, ``gradient_map`` across each
    occurrence of its key, and ``color_map`` is applied last. As on the Pango
    path, a color goes through its hex spelling, and an Algan color with that
    exact hex comes back with its glow and opacity.
    """
    palette, char_colors, algan_colors = [], [None] * len(text), {}
    base_glow = (
        float(base_color.reshape(-1)[3])
        if isinstance(base_color, torch.Tensor) and base_color.numel() >= 5
        else 0.0
    )

    def resolved(hex_color):
        color = algan_colors.get(hex_color)
        palette.append(Color(hex_color, glow=base_glow) if color is None else color)
        return len(palette) - 1

    def fade(stops, start, end):
        if end <= start:
            return
        hexes = [_to_pango_hex(color, algan_colors) for color in stops]
        for i, color in enumerate(mn.color_gradient(hexes, end - start)):
            char_colors[start + i] = resolved(color.to_hex().upper())

    if gradient:
        fade(gradient, 0, len(text))
    for key, stops in (gradient_map or {}).items():
        for start, end in _text_spans(key, text):
            fade(stops, start, end)
    for key, color in (color_map or {}).items():
        index = resolved(_to_pango_hex(color, algan_colors))
        for start, end in _text_spans(key, text):
            char_colors[start:end] = [index] * len(char_colors[start:end])
    return palette, char_colors


def _warn_latex_text_fallback():
    """Say once per process that :class:`Text` is not using Pango, and why."""
    global _LATEX_TEXT_FALLBACK_WARNED
    if _LATEX_TEXT_FALLBACK_WARNED:
        return
    _LATEX_TEXT_FALLBACK_WARNED = True
    import warnings

    from algan.errors import UnsupportedFeatureWarning

    warnings.warn(
        "Text is typesetting through LaTeX's text mode, because the optional "
        "`manimpango` package is not installed. For your system's fonts, "
        'install it: `pip install "algan[pango]"` (on Linux this builds '
        "against Pango, so first e.g. `sudo apt install build-essential "
        "python3-dev libpango1.0-dev pkg-config`). Until then every font is "
        "Computer Modern: `font`/`font_map` choose only its serif, sans-serif "
        "or typewriter face (a monospace name such as 'DejaVu Sans Mono' "
        "selects typewriter), `weight` is regular or bold, and characters "
        "LaTeX's text fonts lack (most non-Latin scripts, emoji) cannot be "
        "typeset.",
        UnsupportedFeatureWarning,
        stacklevel=3,
    )


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

    When Pango is unavailable -- a Linux install without the ``algan[pango]``
    extra -- Algan typesets the text through LaTeX's text mode instead, and
    warns once. Everything is then Computer Modern: ``font`` and ``font_map``
    choose its serif, sans-serif or typewriter face from the font's name (a
    monospace name such as ``"DejaVu Sans Mono"`` selects typewriter), weights
    from ``"SEMIBOLD"`` up are bold, ``"ITALIC"``/``"OBLIQUE"`` are italic and
    slanted, and ``color_map``, ``gradient_map`` and ``gradient`` color the
    same characters they would under Pango. Lines, spacing and capital height
    match Pango's for the same ``font_size``. Characters LaTeX's text fonts lack
    (most non-Latin scripts, emoji) cannot be typeset there.

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
        ``"NORMAL"``, ``"ITALIC"`` or ``"OBLIQUE"``, in any case. Defaults to
        ``"NORMAL"``. A name Pango does not know raises, rather than silently
        rendering in the default face.
    weight
        A Pango weight name -- ``"THIN"``, ``"LIGHT"``, ``"NORMAL"``,
        ``"MEDIUM"``, ``"SEMIBOLD"``, ``"BOLD"``, ``"HEAVY"``, and the rest --
        in any case. Defaults to ``"NORMAL"``. As with ``slant``, an unknown
        name raises.
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
        slant = _pango_style("slant", slant)
        weight = _pango_style("weight", weight)
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

        if _pango_available():
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
                    k: _pango_style("slant", v) for k, v in slant_map.items()
                }
            if weight_map:
                pango_kwargs["t2w"] = {
                    k: _pango_style("weight", v) for k, v in weight_map.items()
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
            _warn_latex_text_fallback()
            palette, char_colors = _latex_text_colors(
                self.text, color_map, gradient, gradient_map, kwargs.get("color")
            )
            layout = _latex_text_layout(
                self.text,
                font=font,
                slant=slant,
                weight=weight,
                line_spacing=line_spacing,
                disable_ligatures=disable_ligatures,
                font_map=font_map,
                slant_map=slant_map,
                weight_map=weight_map,
                char_colors=char_colors,
            )
            super().__init__(
                *layout.tex_strings,
                delimiter="",
                font_size=font_size * layout.scale,
                latex=True,
                **kwargs,
            )
            self.latex = False
            self._color_latex_segments(
                layout.segment_colors,
                palette,
                stroke=not explicit_stroke_color and bool(kwargs.get("stroke_width")),
            )

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

    def _color_latex_segments(self, segment_colors, palette, stroke):
        """Color the LaTeX fallback's glyphs, one styled segment at a time."""
        if self._character_batch is None or all(c is None for c in segment_colors):
            return
        if len(segment_colors) != len(self.num_mobs_per_segment):
            # MathTex could not split the SVG by segment, so there is nothing
            # to say which glyphs a color belongs to.
            return
        with Off(animation_manager=self.animation_manager):
            for segment, color_index in enumerate(segment_colors):
                if color_index is None:
                    continue
                color = palette[color_index]
                first = int(self.segment_starts[segment])
                for i in range(first, int(self.segment_ends[segment])):
                    view = self.character_mobs[i]
                    view.color = color
                    if stroke:
                        view.stroke_color = color

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
        # Reuse Text's fallback layout (one row per line, characters set
        # literally), then construct the triangulated TeX representation.
        font_size = kwargs.pop("font_size", 48)
        layout = _latex_text_layout(str(text))
        super().__init__(
            *layout.tex_strings, delimiter="", font_size=font_size, **kwargs
        )
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
                raise AlganConfigurationError(
                    "alignment must be 'left', 'center', 'right', or None"
                )
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
