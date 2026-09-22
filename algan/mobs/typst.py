"""Optional Typst typesetting with selections over live Algan geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from algan.animatable_base.mob import Mob
from algan.constants.color import WHITE, Color, to_color
from algan.mobs.group import Group
from algan.mobs.manim_mob import ManimMob
from algan.settings import SETTINGS
from algan.utils.lazy_import import LazyModule
from algan.utils.mob_utils import BatchedMobViewSequence

_manim = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))


class _TypstSelection:
    """Share label views between the native and Manim-convention surfaces."""

    def _initialize_from_manim(self, source, *, batch=False, **kwargs):
        # ManimCompatMob's factory and constructor both pass through here.
        super()._initialize_from_manim(source, batch=batch, **kwargs)
        self._setup_typst_parts(source, batch=batch)

    def _sync_manim_from_algan(self):
        # A packed conversion retains unregistered construction-time children
        # in submobjects. Compatibility queries must read the live packed views.
        from algan.mobs.manim_compat import _sync_manim_node_from_algan

        for part, source in zip(self._typst_parts, self.manim_mobject.submobjects):
            _sync_manim_node_from_algan(part, source)
        return self.manim_mobject

    def _animate_to_manim(self, source, *, before_source=None):
        # Preserve the packing layout when a compatibility transform (scale,
        # move_to, ...) converts its target; otherwise old selections point at
        # rows belonging to a different grouping of the glyphs.
        from algan.mobs.manim_compat import _preserve_algan_state_unchanged_by_manim

        if before_source is None:
            before_source = self._sync_manim_from_algan().copy()
        batch = isinstance(self._typst_parts, BatchedMobViewSequence)
        target = ManimMob(source, batch=batch, scene=self.scene, add_to_scene=False)
        before = ManimMob(
            before_source, batch=batch, scene=self.scene, add_to_scene=False
        )
        _preserve_algan_state_unchanged_by_manim(self, before, target)
        self.manim_mobject = source
        return self.become(target, detach_history=False)

    def _setup_typst_parts(self, source, *, batch, parts=None):
        self._typst_parts = (
            BatchedMobViewSequence(
                self.get_non_component_children()[0], len(source.submobjects)
            )
            if batch and source.submobjects
            else self.submobjects
            if parts is None
            else parts
        )
        # Use positions rather than object ids so cloning keeps selections
        # attached to the clone's own paths and packed rows.
        indices = {id(part): i for i, part in enumerate(source.submobjects)}
        self._typst_labels = {
            key: tuple(indices[id(part)] for part in source.select(key))
            for key in source._label_aliases
        }
        self._typst_baselines = {
            i: (
                part._typst_reference_points.copy(),
                part._typst_reference_baseline_frame.copy(),
            )
            for i, part in enumerate(source.submobjects)
            if "_typst_reference_points" in vars(part)
        }

    def select(self, key: str | int) -> Group:
        """Select all paths in a labeled Typst passage or mathematical group.

        Use ``#box[content] <label>`` in :class:`Typst`, or
        ``{{ content : label }}`` in :class:`MathTypst`. Repeated labels select
        every occurrence. Unnamed ``{{ content }}`` groups are numbered from
        zero, independently of named groups.

        Animation
        ---------
        Selection is immediate and does not change the Scene. The returned
        view edits the original paths; it never needs spawning. Before the
        containing Mob is spawned, edits are immediate. Afterwards, moves and
        color changes record over the current context's runtime (1 second by
        default); use ``with Seq(runtime=2): ...`` to change it.

        Parameters
        ----------
        key
            Label string, or nonnegative index of an unnamed math group.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            A non-owning view of the selected paths, in drawing order.

        Raises
        ------
        KeyError
            If the label does not exist.
        IndexError
            If an unnamed group index does not exist.
        TypeError
            If the key is neither a string nor an integer.

        Examples
        --------
        Move just the numerator:

        .. algan:: Example1TypstSelect

            from algan import *

            equation = MathTypst("{{ a + b : num }} / {{ c : den }}").spawn()
            equation.select("num").color = YELLOW
            equation.select("num").move(UP)
            Scene.save_video()
        """
        if not isinstance(key, (str, int)):
            raise TypeError("Typst selection keys must be strings or integers")
        label = f"_grp-{key}" if isinstance(key, int) else key
        if label not in self._typst_labels:
            error = IndexError if isinstance(key, int) else KeyError
            raise error(
                f"No Typst group {key!r}. Available labels: {list(self._typst_labels)}"
            )
        return Group(
            [self._typst_parts[i] for i in self._typst_labels[label]],
            scene=self.scene,
            add_to_scene=False,
            link_children=False,
        )

    def get_baseline_frame(
        self, part: Mob
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a glyph's baseline origin and its right and up reference points.

        Parameters
        ----------
        part
            One path from this Mob's :meth:`select` result. Construct the
            containing Mob with ``track_baselines=True`` first.

        Returns
        -------
        tuple
            Three tensors, each of shape ``(3,)`` in world units: origin,
            right reference point, and up reference point. These are positions,
            not unit direction vectors, and follow the glyph's affine transforms.

        Raises
        ------
        ValueError
            If the path belongs to another Mob or has no tracked baseline.

        Examples
        --------
        Mark a symbol's baseline origin:

        .. algan:: Example1TypstBaseline

            from algan import *

            equation = MathTypst("{{ x }}", track_baselines=True).spawn()
            origin, right, up = equation.get_baseline_frame(equation.select(0)[0])
            Dot().move_to(origin).spawn()
            Scene.save_video()
        """
        for i, candidate in enumerate(self._typst_parts):
            if candidate is part and i in self._typst_baselines:
                return self._baseline_frame(i)
        raise ValueError(
            "No tracked Typst baseline for this path. Use a path selected from "
            "this Mob, constructed with track_baselines=True."
        )

    def _baseline_frame(self, index):
        reference_points, reference_frame = self._typst_baselines[index]
        points = self._typst_parts[index].control_points.location.reshape(-1, 3)
        if len(points) != len(reference_points):
            raise ValueError(
                "The path's topology changed; its Typst baseline is unavailable"
            )
        reference_xy = np.column_stack(
            (reference_points[:, :2], np.ones(len(reference_points)))
        )
        frame_xy = np.column_stack((reference_frame[:, :2], np.ones(3)))
        transform = np.linalg.lstsq(
            reference_xy, points.detach().cpu().numpy(), rcond=None
        )[0]
        return tuple(
            torch.as_tensor(p, dtype=points.dtype, device=points.device)
            for p in frame_xy @ transform
        )

    @property
    def baseline_frames(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Current world-space baseline frames for tracked paths, in drawing order.

        Each frame contains three shape-``(3,)`` tensors: origin, right
        reference point and up reference point. Returns an empty list unless
        ``track_baselines=True`` was requested. See :meth:`get_baseline_frame`
        for an example.
        """
        return [self._baseline_frame(i) for i in self._typst_baselines]


class Typst(_TypstSelection, ManimMob):
    """Typeset Typst markup as animatable cubic Bezier paths.

    Install the optional backend with ``pip install "algan[typst]"``. Its
    Python wheel includes the compiler. Labels such as
    ``#box[important words] <focus>`` make parts accessible with
    ``text.select("focus")``. Explicit colors in the markup are preserved.

    Animation
    ---------
    Construction typesets immediately. Call ``spawn()`` to display the Mob;
    later transforms and selected-part edits record ordinary Algan animations
    over the current context's runtime (1 second by default). Use
    ``with Seq(runtime=2): ...`` to change the duration or ``Off()`` for
    immediate edits. Set materials before spawning.

    Parameters
    ----------
    typst_code
        Typst markup, including ``$ ... $`` delimiters for embedded mathematics.
    font_size
        Font size in points. Defaults to 48. Must be positive; ``height``
        overrides the resulting geometric size when provided.
    typst_preamble
        Typst rules inserted before the content, for example
        ``#set text(font: "DejaVu Sans")``. Defaults to an empty string.
    color
        Default text color; an Algan Color, named constant, or value accepted
        by Color. Defaults to ``None``, meaning WHITE. Non-black colors
        explicitly set in the Typst markup take precedence.
    stroke_width
        Outline width in Algan stroke units. Defaults to ``None``, preserving
        the SVG's strokes, including fraction bars and underlines.
    font_paths
        Additional font directories for the compiler. Defaults to ``None``,
        using system fonts and the compiler's bundled fonts.
    track_baselines
        Whether to retain glyph baseline reference frames for
        :meth:`get_baseline_frame`. Defaults to False.
    should_center
        Whether to center the imported content at the origin. Defaults to True.
    height
        Desired total height in world units. Defaults to ``None``, letting
        ``font_size`` determine the size.
    batch
        Whether to pack paths into one renderable Mob. Defaults to False.
        Selected paths still support transforms and color changes when packed,
        but share one lifespan, so cannot spawn or despawn separately.
    **kwargs
        Passed to :class:`~.ManimMob`, including ``scene``, ``add_to_scene``,
        ``glow`` and ``glow_radius``.

    Raises
    ------
    ImportError
        If the optional Typst package is missing.
    ValueError
        If ``font_size`` is not positive.

    See Also
    --------
    :class:`MathTypst` : Typeset a mathematical expression without delimiters.
    :class:`~algan.mobs.text.Tex` : Typeset LaTeX.

    Examples
    --------
    Highlight a labeled passage:

    .. algan:: Example1Typst

        from algan import *

        text = Typst("A #box[selectable] <word> passage").spawn()
        text.select("word").color = BLUE
        Scene.save_video()
    """

    _typst_class_name = "Typst"
    # This empty SVG root is a container. Let become() replace the hierarchy
    # with the target's paths and selection metadata, rather than morphing an
    # empty cubic root in place and retaining the old labels.
    _morph_family = None

    # Publish the shared operations on the public class so autodoc documents
    # them here rather than only on the private compatibility mixin.
    select = _TypstSelection.select
    get_baseline_frame = _TypstSelection.get_baseline_frame
    baseline_frames = _TypstSelection.baseline_frames

    def __init__(
        self,
        typst_code: str,
        *,
        font_size: float = 48,
        typst_preamble: str = "",
        color: Color | str | int | tuple | None = None,
        stroke_width: float | None = None,
        font_paths: list[str | Path] | None = None,
        track_baselines: bool = False,
        should_center: bool = True,
        height: float | None = None,
        batch: bool = False,
        **kwargs: Any,
    ) -> None:
        if not np.isfinite(font_size) or font_size <= 0:
            raise ValueError("font_size must be positive and finite")
        from algan.mobs.manim_compat import to_manim

        source = getattr(_manim, self._typst_class_name)(
            typst_code,
            font_size=font_size,
            typst_preamble=typst_preamble,
            color=to_manim(to_color(WHITE if color is None else color)),
            stroke_width=(
                None
                if stroke_width is None
                else stroke_width * SETTINGS.style.manim_stroke_width_ratio
            ),
            font_paths=font_paths,
            track_baselines=track_baselines,
            should_center=should_center,
            height=height,
        )
        super().__init__(source, batch=batch, **kwargs)
        self._setup_typst_parts(source, batch=batch)


class MathTypst(Typst):
    """Typeset a Typst mathematical expression with selectable subexpressions.

    The expression is wrapped in math delimiters automatically. Use
    ``{{ content : label }}`` to name a part, or ``{{ content }}`` for a group
    accessible by its zero-based index. Repeated labels select all occurrences.
    The syntax is Typst mathematics, for example ``frac(a, b)`` and ``sqrt(x)``.

    Animation
    ---------
    Construction is immediate; call ``spawn()`` to display the equation. Edits
    after spawning record over the current context's runtime (1 second by
    default); ``with Seq(runtime=2): ...`` changes the duration. Selected parts
    edit the displayed equation without needing their own ``spawn()``.

    Parameters
    ----------
    math_expression
        Typst math source without the surrounding ``$`` delimiters.
    **kwargs
        Passed to :class:`Typst`, including ``font_size`` (48 points by
        default), ``typst_preamble``, ``color``, ``font_paths``, ``batch`` and
        ``track_baselines``.

    Raises
    ------
    ImportError
        If the optional Typst package is missing; install ``algan[typst]``.
    ValueError
        If ``font_size`` is not positive.

    Examples
    --------
    Color both sides of an equation:

    .. algan:: Example1MathTypst

        from algan import *

        equation = MathTypst("{{ a^2 + b^2 : lhs }} = {{ c^2 }}").spawn()
        with Sync():
            equation.select("lhs").color = BLUE
            equation.select(0).color = YELLOW
        Scene.save_video()
    """

    _typst_class_name = "MathTypst"

    def __init__(self, math_expression: str, **kwargs: Any) -> None:
        super().__init__(math_expression, **kwargs)


__all__ = ["Typst", "MathTypst"]
