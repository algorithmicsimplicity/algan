"""A number on screen that animates between values.

:class:`DecimalNumber` renders a number as text and makes its ``value`` an
animatable attribute, so ``counter.value = 100`` counts smoothly from wherever it
was over the surrounding context's runtime -- re-rendering the glyphs each frame
rather than interpolating their outlines.

``decimal_places`` and ``integer_places`` fix the format so the display
does not jitter in width as digits come and go. ``significant_figures`` opts
into a fixed number of significant digits instead.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import (
    _reports_refused_keywords,
    animated_function,
)
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    NoExtra,
    Off,
    Seq,
    Sync,
    active_scene_for_new_mob,
)
from algan.errors import UnsupportedFeatureError
from algan.geometry.geometry import (
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.mobs.text import Tex
from algan.utils.mob_utils import batch_mobs
from algan.utils.tensor_utils import cast_to_tensor, reject_non_finite


def _glyph_alphas(circuit):
    """The (fill, stroke) alpha a packed glyph circuit was built with."""
    return tuple(
        float(part.get_animated_attribute("color")[..., -1].reshape(-1)[0])
        for part in (circuit.grid, circuit.border_grid)
    )


def _show_glyphs(circuit, shown, alphas):
    """Draw only the ``shown`` members of a packed glyph circuit.

    ``shown`` holds one 0/1 entry per member, shape ``(*, members, 1)``, and
    ``alphas`` the (fill, stroke) alpha a shown member is drawn with. The
    choice is written into the alpha of the two colors the renderer draws a
    circuit from, its fill and stroke texture grids, which it multiplies by
    ``opacity`` -- so the choice never lives in ``opacity`` itself. A fade,
    recorded or replayed inside an animated function, then reaches every glyph
    without revealing the hidden ones, and a value change replayed during a
    fade never undoes the fade.
    """
    scene = circuit.scene
    # A deliberate alpha write, not a recolor that should keep the old alpha
    # (Mob._keep_explicit_alpha).
    previous = getattr(scene, "_writing_component_alpha", False)
    scene._writing_component_alpha = True
    try:
        for part, alpha in zip((circuit.grid, circuit.border_grid), alphas):
            colors = part.get_animated_attribute("color")
            alpha = (shown * alpha).to(device=colors.device, dtype=colors.dtype)
            sizes = part.parent_batch_sizes
            if sizes is not None and sizes.numel() == alpha.shape[-2]:
                # A member owns a block of texels (several once refined).
                alpha = alpha.repeat_interleave(sizes.to(alpha.device), dim=-2)
            rows = torch.broadcast_shapes(colors.shape[:-1], alpha.shape[:-1])
            colors = colors.expand(*rows, colors.shape[-1]).clone()
            colors[..., -1:] = alpha
            part.set_animated_attribute("color", colors, recursive=False)
    finally:
        scene._writing_component_alpha = previous


def _drawn_members(circuit):
    """Which members of a packed glyph circuit :func:`_show_glyphs` draws.

    One flag per member: whether the alpha of its fill or stroke color is
    above zero anywhere in its block of texels.
    """
    drawn = None
    for part in (circuit.grid, circuit.border_grid):
        alpha = part.get_animated_attribute("color")[..., -1]
        alpha = alpha.reshape(-1, alpha.shape[-1]).amax(0) > 0
        sizes = part.parent_batch_sizes
        if sizes is not None and int(sizes.sum()) == alpha.shape[-1]:
            alpha = torch.stack(
                [block.any() for block in torch.split(alpha, sizes.tolist())]
            )
        drawn = alpha if drawn is None else drawn | alpha
    return drawn


def _keep_glyph_selection(mobs):
    """Make color writes over ``mobs`` keep the glyph selection in their alpha.

    See :meth:`~algan.animatable_base.mob.Mob._keep_explicit_alpha`: a color
    whose alpha is 1, which includes every named color, then recolors the
    glyphs without showing the hidden ones.
    """
    for mob in mobs:
        if isinstance(mob, BezierCircuitCubic):
            mob._mark_explicit_alpha(stroke=False)
            mob._mark_explicit_alpha(stroke=True)


class DecimalNumber(Mob):
    """A number drawn on screen, which counts when you assign to it.

    :attr:`value` is an animatable attribute, so ``counter.value = 100``
    is recorded like a move or a color change and the display counts from
    wherever it was to 100 over the surrounding context's runtime. The glyphs
    are re-typeset each frame rather than morphed into one another, which is
    what makes ``7`` become ``8`` rather than bending into it.

    The number is laid out to a fixed width from ``decimal_places`` and
    ``integer_places``, so it does not jitter sideways as digits come and go.
    Leading-digit slots are added automatically when the value outgrows them.
    Alternatively, ``significant_figures=3`` shows three significant digits,
    including trailing zeros: ``12.0``, ``0.00120``, or ``1.20e+04``. This mode
    centers the formatted number on the Mob; its width can change with notation.
    ``group_with_commas=True`` separates thousands in either mode:
    ``1,234,567``.

    Animation
    ---------
    Constructing one records nothing; the Mob joins the active Scene unspawned,
    and :meth:`~algan.animatable_base.animatable.Animatable.spawn` makes it
    appear. Assigning to :attr:`value` afterwards is recorded and interpolates
    from the current value over the current context's runtime (1 second by
    default) -- ``with Seq(runtime=3): counter.value = 100`` to change that, or
    ``with Off():`` to jump. Before spawn, assignments take effect immediately
    without spending video time. A value change never spawns or revives the
    number, including when another animation or updater changes it. Each frame
    formats the interpolated value using the selected precision mode.

    Fade it through ``opacity`` like any other Mob -- while it counts, or from
    inside an :func:`~algan.animatable_base.animatable.animated_function`. Each
    digit slot holds every glyph it can show and draws one of them by giving
    the others zero color alpha, which the renderer multiplies by ``opacity``,
    so no fade can reveal them. Recoloring keeps that selection, but assigning
    a color whose own alpha is below 1 shows every glyph: make the number
    translucent through ``opacity`` (or the ``color`` it is built with).

    Parameters
    ----------
    value
        The number shown at construction, as a scalar or a tensor of shape
        ``(*, 1)``. Python scalars use the default Torch floating-point dtype.
    decimal_places
        Digits after the decimal point. Defaults to ``2``; ``0`` shows no
        decimal point at all. Ignored when ``significant_figures`` is supplied.
    integer_places
        Minimum digits before the decimal point. Defaults to ``None``, meaning
        take it from ``value`` (at least 1). More are allocated automatically
        when the value grows past the reserved width -- except by an updater,
        which cannot add digits while the video renders: give a number an
        updater sets room for the largest value it will show. Ignored when
        ``significant_figures`` is supplied.
    significant_figures
        Positive integer count of significant digits. Defaults to ``None``,
        preserving decimal-place formatting. Rounds the stored binary floating
        value to nearest, with exact ties going to the even last digit, just
        like Python's general (``g``) format. Signs do not count as digits;
        negative numbers round symmetrically. Trailing zeros are retained.
        Zero, including negative zero, is ``0`` followed by a decimal point
        and ``significant_figures - 1`` zeros (no point for one figure).
        Scientific notation is used when the rounded decimal exponent is less
        than ``-4`` or at least ``significant_figures``; otherwise fixed
        notation is used. Exponents have a sign and at least two digits,
        e.g. ``1.00e-05``. Commas are added only with ``group_with_commas``,
        and only in fixed notation: ``12,300``.
    group_with_commas
        Put a comma between each group of three integer digits, as in
        ``1,234,567``. Defaults to ``False``. A comma is drawn only once
        the number has a digit to its left, so ``999`` counting up to
        ``1,000`` gains its comma with the fourth digit.
    **kwargs
        Passed to :class:`~algan.mobs.text.Tex`, which typesets the digits --
        notably ``color`` and ``font_size`` -- and on to :class:`~.Mob`.

    Attributes
    ----------
    value
        The number displayed, shape ``(*, 1)``. Assigning to it records a
        counting animation.

    Raises
    ------
    TypeError
        If ``significant_figures`` is neither an integer nor ``None``.
        Booleans and integer-valued floats are rejected too.
    ValueError
        If ``significant_figures`` is less than one.
    :class:`~algan.errors.AlganConfigurationError`
        If a value is not finite: no glyph shows NaN or infinity.

    Examples
    --------
    A counter that runs from 0 to 50 over two seconds:

    .. algan:: Example1DecimalNumber

        from algan import *

        counter = DecimalNumber(0, decimal_places=1).spawn()
        with Seq(runtime=2):
            counter.value = 50

        Scene.save_video()

    A population count, with commas between thousands:

    .. algan:: Example3DecimalNumber

        from algan import *

        counter = DecimalNumber(0, decimal_places=0, group_with_commas=True)
        counter.spawn()
        with Seq(runtime=3):
            counter.value = 1234567

        Scene.save_video()

    Count using three significant figures, switching notation as needed:

    .. algan:: Example2DecimalNumber

        from algan import *

        counter = DecimalNumber(0, significant_figures=3).spawn()
        with Seq(runtime=3):
            counter.value = 12000

        Scene.save_video()
    """

    @_reports_refused_keywords
    def __init__(
        self,
        value: float | torch.Tensor,
        decimal_places: int = 2,
        integer_places: int | None = None,
        *,
        significant_figures: int | None = None,
        group_with_commas: bool = False,
        **kwargs: Any,
    ) -> None:
        if significant_figures is not None:
            if isinstance(significant_figures, bool) or not isinstance(
                significant_figures, Integral
            ):
                raise TypeError(
                    "significant_figures must be a positive integer or None"
                )
            if significant_figures < 1:
                raise ValueError("significant_figures must be at least 1")
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        value = cast_to_tensor(value)
        self.significant_figures = significant_figures
        self.decimal_places = decimal_places
        self.group_with_commas = bool(group_with_commas)
        if significant_figures is not None:
            value = value.to(dtype=torch.get_default_dtype())
        # Neither mode has a glyph for NaN or infinity, and decimal places
        # showed them as 0.00 and as the largest number its slots hold.
        reject_non_finite("value", value)
        if significant_figures is not None:
            self._init_significant_display(value, kwargs)
            return
        required_integer_places = self._required_integer_places(value)
        if integer_places is None:
            integer_places = required_integer_places
        else:
            integer_places = max(integer_places, required_integer_places)
        self.integer_places = integer_places
        num_i, num_d = integer_places, decimal_places
        animation_manager = kwargs["scene"].animation_manager
        with (
            Off(animation_manager=animation_manager),
            NoExtra(priority_level=1, animation_manager=animation_manager),
        ):
            # One character per glyph. Tex typesets math, where a bare comma
            # is punctuation followed by a space; "{,}" sets it between digits.
            placeholder_glyphs = (
                "-"
                + self._group_digits("0" * num_i)
                + ("." + "0" * num_d if num_d > 0 else "")
            )
            self.placeholder = Tex(placeholder_glyphs.replace(",", "{,}"), **kwargs)
            ct = self.placeholder.animation_manager.context.timespan.current_time
            self.placeholder.animation_manager.context.rewind(
                1 / self.placeholder.scene.frames_per_second + 1e-3
            )
            self.placeholder.opacity = 0
            self.placeholder.animation_manager.context.timespan.current_time = ct

            # Placeholder glyphs: [0]='-', then the integer digits (with a
            # comma between groups of three when grouping), '.', and the
            # decimal digits.
            digit_glyphs = [i for i, c in enumerate(placeholder_glyphs) if c == "0"]
            comma_glyphs = [i for i, c in enumerate(placeholder_glyphs) if c == ","]
            self.decimal = (
                self.placeholder[placeholder_glyphs.index(".")] if num_d > 0 else None
            )
            self.negative_sign = self.placeholder[0]
            # The point and sign are drawn at the digits' opacity; the sign is
            # shown and hidden through its alpha (_show_glyphs), as they are.
            for part in (self.decimal, self.negative_sign):
                if part is not None:
                    part.opacity = 1
            if num_i >= 2:
                digit_advance_points = (
                    self.placeholder[digit_glyphs[num_i - 2]].location,
                    self.placeholder[digit_glyphs[num_i - 1]].location,
                )
            elif num_d >= 2:
                digit_advance_points = (
                    self.placeholder[digit_glyphs[num_i]].location,
                    self.placeholder[digit_glyphs[num_i + 1]].location,
                )
            else:
                spacing_kwargs = dict(kwargs)
                spacing_kwargs["add_to_scene"] = False
                spacing_template = Tex("00", **spacing_kwargs)
                digit_advance_points = (
                    spacing_template[0].location,
                    spacing_template[1].location,
                )
            self.digit_mobs = []
            for _ in range(num_i + num_d):
                self.digit_mobs.append(Tex("0123456789", **kwargs))
            self._glyph_alphas = _glyph_alphas(self.digit_mobs[0]._character_batch)
            for i in range(len(self.digit_mobs)):
                location = self.placeholder[digit_glyphs[i]].location
                for d in self.digit_mobs[i].character_mobs:
                    d.location = location
            # separator_mobs[j] is the comma left of the (3j+3)-th integer
            # digit from the right, drawn once the number has that digit. The
            # template is never drawn: it is what more commas are cloned from
            # when the number grows.
            self.separator_mobs = []
            self._separator_template = None
            if self.group_with_commas:
                self._separator_template = Tex(",", **kwargs)
                for index in reversed(comma_glyphs):
                    separator = Tex(",", **kwargs)
                    for glyph in separator.character_mobs:
                        glyph.location = self.placeholder[index].location
                    self.separator_mobs.append(separator)
            super().__init__(**kwargs)
            self.register_attrs_as_animatable(["value"], DecimalNumber)
            self._setattr_without_record("value", value)

            parent_location, parent_basis = self.location, self.basis

            def to_local(point):
                return map_global_to_local_coords(parent_location, parent_basis, point)

            self._rightmost_integer_location = to_local(
                self.digit_mobs[num_i - 1].character_mobs[0].location
            )
            self._digit_advance = to_local(digit_advance_points[1]) - to_local(
                digit_advance_points[0]
            )
            # The extra advance a comma adds between two digits, and where a
            # comma sits from the digit on its left.
            self._separator_advance = torch.zeros_like(self._digit_advance)
            self._separator_offset = torch.zeros_like(self._digit_advance)
            if self.group_with_commas:
                spacing_kwargs = dict(kwargs)
                spacing_kwargs["add_to_scene"] = False
                spacing_template = Tex("0{,}0", **spacing_kwargs)
                left, comma, right = (
                    to_local(spacing_template[i].location) for i in range(3)
                )
                self._separator_advance = right - left - self._digit_advance
                self._separator_offset = comma - left
                self._place_separator(self._separator_template, 0)
            self._negative_sign_offset = to_local(
                self.negative_sign.location
            ) - self._integer_slot_location(num_i - 1)
            self.update_display(self.value)
        parts = [self.digit_mobs, self.separator_mobs]
        if self._separator_template is not None:
            parts.append(self._separator_template)
        if self.decimal is not None:
            parts.append(self.decimal)
        self.add_children(*parts, self.negative_sign)
        for c in self.children:
            c.on_create = lambda c=c: c
        _keep_glyph_selection(self.get_descendants())

    def __deepcopy__(self, memo):
        # The placeholder lays the number out at construction, and its pack
        # then draws the sign and point, which are views of it. A copy would
        # be a second sign and point, drawn where this number was cloned and
        # never updated; a clone's sign and point are copies of their own.
        placeholder = getattr(self, "placeholder", None)
        if placeholder is not None:
            memo.setdefault(id(placeholder), placeholder)
        return super().__deepcopy__(memo)

    def _morphs_through_stand_in(self):
        return True

    def _morph_stand_in(self):
        """The glyphs this number draws, packed as a Tex packs its own, for
        ``become`` to morph.

        Each digit slot holds every glyph it can show and draws one of them
        (:func:`_show_glyphs`), so a morph of the slots themselves pairs the
        target with glyphs that are never drawn and ends invisible.
        """
        glyphs = []
        for circuit in self.get_descendants():
            if not isinstance(circuit, BezierCircuitCubic):
                continue
            drawn = _drawn_members(circuit)
            if circuit.data_sub_inds is not None:
                # The sign or the point: a glyph of the placeholder.
                members = [circuit] if bool(drawn.any()) else []
            else:
                members = [circuit[int(k)] for k in torch.nonzero(drawn).flatten()]
            glyphs.extend(
                member.clone(add_to_scene=False, spawn=False) for member in members
            )
        if not glyphs:
            # Nothing drawn (a fully transparent color): the target grows
            # from nothing, as from an empty Group.
            return Group(scene=self.scene)
        # Left to right, as the glyphs of the text the number reads as.
        glyphs.sort(key=lambda glyph: float(glyph.get_center().reshape(-1)[0]))
        return batch_mobs(glyphs)

    @property
    def value(self):
        # Documented in the class docstring's Attributes section; a docstring
        # here would describe the same object twice in the API reference.
        return self.get_animated_attribute("value")

    @value.setter
    def value(self, value):
        timeline = self.scene.timeline_manager.attr_to_timeline.get("value")
        if timeline is None or self.id not in timeline.mob_id_to_inds:
            # A clone is filled from its source by assignment before it owns a
            # value of its own (Animatable.__deepcopy__), and there is nothing
            # to count from yet. Its glyphs are copies already showing it.
            self._setattr_without_record("value", cast_to_tensor(value))
            return
        self.set_value(value)

    def on_create(self):
        """Play this number's spawn-in animation: its glyphs fade in together.

        Called by :meth:`~.Animatable.spawn` when ``animate=True``.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default). Every glyph fades from transparent to its own opacity over
        the same span.
        """
        with Sync(animation_manager=self.animation_manager):
            for c in self.get_descendants():
                o = c.opacity
                with Seq(animation_manager=self.animation_manager):
                    with Off(animation_manager=self.animation_manager):
                        c.set_non_recursive(opacity=0)
                    c.set_non_recursive(opacity=o)

    def set_opacity_via_color(self, opacity: float | torch.Tensor) -> DecimalNumber:
        """Fade the number; for a DecimalNumber, the same as setting ``opacity``.

        Each digit slot picks the glyph it draws through the glyphs' color
        alpha, so writing alpha here, as
        :meth:`~algan.animatable_base.mob.Mob.set_opacity_via_color` does for
        other Mobs, would show every hidden glyph. ``opacity`` already fades
        the digits without disturbing their color, which is what the color
        route exists for. :func:`~algan.animations.indication.Blink` uses it.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default), exactly like assigning to ``opacity``.

        Parameters
        ----------
        opacity
            Target opacity, ``0`` for fully transparent to ``1`` for fully
            opaque.

        Returns
        -------
        :class:`~algan.mobs.numeric_display.DecimalNumber`
            This Mob, so calls can be chained.
        """
        self.opacity = opacity
        return self

    def get_value(self):
        """Get the number shown; the same as reading :attr:`value`.

        Returns
        -------
        torch.Tensor
            The value as of the animations recorded so far, shape ``(*, 1)``.
        """
        return self.value

    def set_value(
        self,
        value: float | torch.Tensor,
        interpolation: float | torch.Tensor = 1,
    ) -> DecimalNumber:
        """Count toward a new value using this number's precision mode.

        Animation
        ---------
        After spawn, recorded over the current context's runtime (1 second
        by default), with ``interpolation`` sweeping from zero to its target.
        Use ``with Seq(runtime=3):`` to change the timing or ``with Off():``
        to jump. Before spawn, changes are immediate and spend no video time.
        Updates select this number's glyphs without changing its lifespan.

        Parameters
        ----------
        value
            Target number, as a scalar or a tensor of shape ``(*, 1)``.
            Python scalars use the default Torch floating-point dtype.
        interpolation
            Fraction of the displayed change to apply; ``1`` shows the target.
            Accepts a scalar or a tensor of shape ``(*, 1)``. Defaults to ``1``.

        Returns
        -------
        :class:`~algan.mobs.numeric_display.DecimalNumber`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If ``value`` is not finite.
        :class:`~algan.errors.UnsupportedFeatureError`
            If an updater sets a value with more digits before the decimal
            point than the number has room for; see ``integer_places``.

        Examples
        --------
        Count from one to ten with three significant figures:

        .. algan:: Example1DecimalNumberSetValue

            from algan import *

            counter = DecimalNumber(1, significant_figures=3).spawn()
            counter.set_value(10)

            Scene.save_video()
        """
        value = cast_to_tensor(value)
        if self.significant_figures is not None:
            value = value.to(dtype=self.value.dtype)
        reject_non_finite("value", value)
        if self.significant_figures is None:
            self._ensure_integer_places(value)
        return self._set_value(value, interpolation=interpolation)

    @animated_function(animated_args={"interpolation": 0})
    def _set_value(self, value, interpolation=1):
        value = cast_to_tensor(value)
        old_value = self.value
        interpolated_value = old_value * (1 - interpolation) + interpolation * value
        self.update_display(interpolated_value)
        self._setattr_and_record_modification("value", value)
        return self

    def _format_significant_value(self, value):
        # Alternate general format retains significant trailing zeros. Remove
        # only its otherwise bare decimal point ("123." / "1.e+03"). Python
        # rounds the stored binary float to nearest, with ties to even.
        if value == 0:
            value = 0.0  # Normalize negative zero, as in the decimal-place mode.
        grouping = "," if self.group_with_commas else ""
        formatted = f"{value:#{grouping}.{self.significant_figures}g}"
        return formatted.replace(".e", "e").rstrip(".")

    def _init_significant_display(self, value, kwargs):
        # General notation needs at most p+7 characters: sign, p digits,
        # decimal point, 'e', exponent sign and three exponent digits. Fixed
        # notation uses at most four leading fractional zeros instead. Reserve
        # all slots now so replay never creates glyphs, even across zero or
        # under an easing curve that overshoots the authored endpoints.
        # Grouping adds at most one comma per three of the p integer digits
        # fixed notation can have.
        self._significant_characters = "0123456789.e+-" + (
            "," if self.group_with_commas else ""
        )
        commas = (self.significant_figures - 1) // 3 if self.group_with_commas else 0
        with (
            Off(animation_manager=kwargs["scene"].animation_manager),
            NoExtra(
                priority_level=1, animation_manager=kwargs["scene"].animation_manager
            ),
        ):
            super().__init__(**kwargs)
            self.register_attrs_as_animatable(["value"], DecimalNumber)
            self._setattr_without_record("value", value)
            spacing = Tex("00", **{**kwargs, "add_to_scene": False})
            self._digit_advance = map_global_to_local_coords(
                self.location, self.basis, spacing[1].location
            ) - map_global_to_local_coords(
                self.location, self.basis, spacing[0].location
            )
            # A comma's offset from the digit on its left, and the advance
            # between the digits either side of it, in digit advances.
            self._comma_advances = (1.0, 2.0)
            if self.group_with_commas:
                comma_spacing = Tex("0{,}0", **{**kwargs, "add_to_scene": False})
                left, comma, right = (
                    map_global_to_local_coords(
                        self.location, self.basis, comma_spacing[i].location
                    )
                    for i in range(3)
                )
                squared = float((self._digit_advance**2).sum())
                self._comma_advances = tuple(
                    float(((point - left) * self._digit_advance).sum()) / squared
                    for point in (comma, right)
                )
            self._significant_slots = [
                Tex(self._significant_characters, **kwargs)
                for _ in range(self.significant_figures + 7 + commas)
            ]
            template = self._significant_slots[0]
            self._glyph_alphas = _glyph_alphas(template._character_batch)
            self._significant_offsets = []
            for glyph in template.character_mobs:
                offset = map_global_to_local_coords(
                    self.location, self.basis, glyph.location
                ) - map_global_to_local_coords(
                    self.location, self.basis, template[0].location
                )
                offset[..., 0] = 0
                self._significant_offsets.append(offset)
            self.add_children(self._significant_slots)
            for slot in self._significant_slots:
                slot.on_create = lambda slot=slot: slot
            self.update_display(self.value)
        _keep_glyph_selection(self.get_descendants())

    def _update_significant_display(self, value):
        strings = [self._format_significant_value(float(v)) for v in value.reshape(-1)]
        positions = self.location.new_tensor(
            [self._significant_positions(s) for s in strings]
        )
        with Sync(animation_manager=self.animation_manager):
            for i, slot in enumerate(self._significant_slots):
                slot_location = self._digit_advance * positions[:, i].reshape(
                    value.shape
                )
                shown = []
                for character, glyph, offset in zip(
                    self._significant_characters,
                    slot.character_mobs,
                    self._significant_offsets,
                ):
                    shown.append(
                        value.new_tensor(
                            [i < len(s) and s[i] == character for s in strings]
                        ).reshape(value.shape)
                    )
                    glyph.set(
                        location=map_local_to_global_coords(
                            self.location, self.basis, slot_location + offset
                        ),
                    )
                _show_glyphs(
                    slot._character_batch, torch.cat(shown, -2), self._glyph_alphas
                )

    def _significant_positions(self, string):
        """Where each slot draws ``string``, in digit advances from its center.

        Every character takes one digit advance except a comma, which is set
        as tightly as between two digits. Slots past the string's end, which
        draw nothing, continue at one advance apiece.
        """
        positions = []
        for k, character in enumerate(string):
            if k == 0:
                position = 0.0
            elif character == ",":
                position = positions[-1] + self._comma_advances[0]
            elif string[k - 1] == ",":
                position = positions[-2] + self._comma_advances[1]
            else:
                position = positions[-1] + 1.0
            positions.append(position)
        center = (positions[0] + positions[-1]) / 2
        positions = [position - center for position in positions]
        positions += [
            positions[-1] + k
            for k in range(1, len(self._significant_slots) - len(string) + 1)
        ]
        return positions

    def _required_integer_places(self, value):
        """Return the slots needed by ``value`` after decimal rounding."""
        required = 1
        for item in cast_to_tensor(value).reshape(-1):
            scalar = abs(float(item.item()))
            if not math.isfinite(scalar):
                continue
            formatted = f"{scalar:.{self.decimal_places}f}"
            required = max(required, len(formatted.partition(".")[0]))
        return required

    def _ensure_integer_places(self, value):
        """Prepend enough integer slots for a newly assigned finite value."""
        required = self._required_integer_places(value)
        if required <= self.integer_places:
            return
        timeline_manager = self.scene.timeline_manager
        if timeline_manager.is_replaying():
            # An updater, or an animated function, run for a batch of frames. A
            # slot built now is never drawn -- the batch's Mobs were gathered
            # before it ran -- and is discarded with the batch while this
            # number still holds it.
            largest = max(
                abs(float(item))
                for item in cast_to_tensor(value).reshape(-1)
                if math.isfinite(float(item))
            )
            if timeline_manager.is_running_updater():
                writer, when = "An updater", "An updater"
            else:
                # Recorded with its final arguments, which needed fewer
                # digits than a value it passes through on the way.
                writer = "An animated function, replayed while the video renders,"
                when = "An animated function"
            raise UnsupportedFeatureError(
                f"{writer} set this DecimalNumber to "
                f"{largest:.{self.decimal_places}f}, which needs {required} digits "
                f"before the decimal point, and it was built with "
                f"{self.integer_places}. {when} cannot add digits while the video "
                "renders: build the number with room for the largest value it "
                f"will show, e.g. DecimalNumber(0, integer_places={required})."
            )

        old_integer_places = self.integer_places
        source_digit = self.digit_mobs[0]
        new_digits_near_to_far = []
        new_separators = []
        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=self.animation_manager,
        ):
            for place_from_right in range(old_integer_places, required):
                digit_mob = source_digit.clone(spawn=False)
                _show_glyphs(
                    digit_mob._character_batch,
                    torch.zeros(1, len(digit_mob.character_mobs), 1),
                    self._glyph_alphas,
                )
                slot_location = map_local_to_global_coords(
                    self.location,
                    self.basis,
                    self._integer_slot_location(place_from_right),
                )
                for digit in digit_mob.character_mobs:
                    digit.location = slot_location
                digit_mob.on_create = lambda digit_mob=digit_mob: digit_mob
                new_digits_near_to_far.append(digit_mob)
            if self.group_with_commas:
                for j in range(len(self.separator_mobs), (required - 1) // 3):
                    separator = self._separator_template.clone(spawn=False)
                    self._place_separator(separator, j)
                    separator.on_create = lambda separator=separator: separator
                    new_separators.append(separator)

        new_digits = list(reversed(new_digits_near_to_far))
        self.digit_mobs = (
            new_digits
            + self.digit_mobs[:old_integer_places]
            + self.digit_mobs[old_integer_places:]
        )
        self.integer_places = required
        self.separator_mobs = self.separator_mobs + new_separators
        self.add_children(new_digits, new_separators)
        _keep_glyph_selection(
            mob
            for slot in new_digits + new_separators
            for mob in slot.get_descendants()
        )

        if self.is_spawned() and not self.is_despawned():
            for slot in new_digits + new_separators:
                slot.spawn(animate=False)

    def _group_digits(self, digits):
        """``digits`` with a comma between groups of three, if grouping."""
        if not self.group_with_commas:
            return digits
        head = len(digits) % 3 or 3
        groups = [digits[:head]] + [
            digits[i : i + 3] for i in range(head, len(digits), 3)
        ]
        return ",".join(groups)

    def _integer_slot_location(self, place_from_right):
        """Local location of an integer digit slot, ``0`` being the units.

        ``place_from_right`` is an integer or a tensor of shape ``(*, 1)``.
        """
        return (
            self._rightmost_integer_location
            - self._digit_advance * place_from_right
            - self._separator_advance * (place_from_right // 3)
        )

    def _place_separator(self, separator, j):
        """Put ``separator`` where ``separator_mobs[j]`` is drawn."""
        location = map_local_to_global_coords(
            self.location,
            self.basis,
            self._integer_slot_location(3 * j + 3) + self._separator_offset,
        )
        for glyph in separator.character_mobs:
            glyph.location = location

    def update_display(self, value):
        """Internal: select and position glyphs for the current value."""
        value = cast_to_tensor(value)
        # Selection is written to color alpha only (_show_glyphs), never to
        # opacity: replay runs after lifespan masking, so a value driven by
        # another Mob's animation or updater before this number's spawn or
        # after its despawn stays hidden, and a concurrent fade is kept.
        if self.significant_figures is not None:
            self._update_significant_display(value)
            return
        negative = torch.where((value < 0), 1, 0)
        value = value.abs()
        num_i, num_d = self.integer_places, self.decimal_places
        # Largest value the digit slots can show; anything bigger is clamped
        # (also protects against rounding carrying into a non-existent digit).
        limit = (10**num_i) - ((10**-num_d) if num_d > 0 else 1)
        visible_integer_places = []

        def get_shown(value):
            all_shown = []
            for v in value:
                x = float(v.item())
                if x != x:  # NaN guard
                    x = 0.0
                x = min(x, limit)
                value_string = f"{x:.{num_d}f}"
                int_part, _, frac_part = value_string.partition(".")
                visible_integer_places.append(len(int_part))
                int_part = int_part.rjust(num_i, "0")
                # Blank out leading zeros, always keeping the last integer digit.
                num_leading = min(len(int_part) - len(int_part.lstrip("0")), num_i - 1)
                for k, digit in enumerate(int_part + frac_part):
                    if k < num_leading:
                        all_shown.append(torch.zeros(10, 1, dtype=torch.long))
                    else:
                        all_shown.append(
                            F.one_hot(torch.tensor((int(digit),)), 10).transpose(0, 1)
                        )
            return torch.stack(all_shown)

        all_shown = torch.stack([get_shown(_) for _ in value], -3)
        visible_integer_places = torch.tensor(
            visible_integer_places,
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape)
        sign_location = map_local_to_global_coords(
            self.location,
            self.basis,
            self._integer_slot_location(visible_integer_places - 1)
            + self._negative_sign_offset,
        )

        with Sync(animation_manager=self.animation_manager):
            self.negative_sign.set(location=sign_location)
            (sign_glyph,) = self.negative_sign.children
            _show_glyphs(sign_glyph, negative, self._glyph_alphas)
            for i, digit_mob in enumerate(self.digit_mobs):
                _show_glyphs(
                    digit_mob._character_batch, all_shown[i], self._glyph_alphas
                )
            for j, separator in enumerate(self.separator_mobs):
                _show_glyphs(
                    separator._character_batch,
                    torch.where(visible_integer_places > 3 * j + 3, 1, 0),
                    self._glyph_alphas,
                )
            if self._separator_template is not None:
                _show_glyphs(
                    self._separator_template._character_batch,
                    torch.zeros_like(negative),
                    self._glyph_alphas,
                )
