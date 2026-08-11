from __future__ import annotations

from algan.animation_timeline.animation_contexts import *


def _with_opacity(color, opacity):
    """Set a color's alpha whether it is a ``Color`` or a plain tensor.

    ``Color`` keeps opacity in its last channel and exposes ``set_opacity``,
    but a border color assigned as a raw tensor (as ``Square`` and friends do)
    has no such method, so write that channel directly.
    """
    set_opacity = getattr(color, "set_opacity", None)
    if set_opacity is not None:
        return set_opacity(opacity)
    color = color.clone()
    color[..., -1:] = opacity
    return color


def draw_border_then_fill(
    mobs,
    run_time=None,
    lag_ratio=None,
    border_width=1,
    border_color=None,
    rate_func=rate_funcs.identity,
    reverse=False,
):
    """Animate mobs appearing as if hand-drawn: outline first, then fill.

    Each mob's border is traced out, then its fill fades in. Mobs are animated
    in iteration order, each starting slightly before the previous one
    finishes, which reads as a hand moving across the screen.

    Parameters
    ----------
    mobs
        Any iterable of Mobs: the glyphs of a :class:`~.Text`, a
        :class:`~.Group`'s children, or a list you assembled yourself. Drawn in
        iteration order.
    run_time
        Total seconds for the whole sequence. Defaults to 1 second, or 2 for
        more than 15 mobs.
    lag_ratio
        Fraction of one mob's animation that elapses before the next begins.
        Defaults to a value that keeps the whole sequence legible.
    border_width
        Temporary outline width while the border is drawn. The original widths are
        restored as the fills appear. If None, the existing widths are used.
        Defaults to 1, equivalent to Manim's stroke width of 2.
    border_color
        Temporary outline color. Defaults to each Mob's existing border color.
        :meth:`~algan.mobs.text.Text.write` supplies white for an ordinary
        stroke-free ``Text``, matching Manim's Pango text style.
    rate_func
        Easing applied to each glyph. Defaults to linear timing, as Manim's
        :class:`~manim.animation.creation.Write` does.
    reverse
        Draw the Mobs in reverse iteration order. Defaults to False.

    Returns
    -------
    list of :class:`~.Mob`
        The mobs that were animated, in the order they were drawn.

    Examples
    --------

    .. algan:: Example1MAnimationsDrawBorderThenFill

        from algan import *

        squares = Group([Square() for _ in range(3)]).arrange_in_line(RIGHT).spawn()
        draw_border_then_fill(squares.children)

        Scene.save_video()

    See Also
    --------
    :meth:`~algan.mobs.text.Tex.write` : the same animation over a text's glyphs.
    """
    mobs = list(mobs)
    if reverse:
        mobs.reverse()
    if not mobs:
        return mobs

    animation_manager = mobs[0].animation_manager
    length = len(mobs)
    if run_time is None:
        run_time = 1 if length < 15 else 2
    if lag_ratio is None:
        lag_ratio = min(4.0 / max(1.0, length), 0.2)

    original_styles = []
    with Off(animation_manager=animation_manager):
        for mob in mobs:
            colors = [
                (descendant, descendant.color.clone())
                for descendant in mob.get_descendants()
            ]
            original_border_width = mob.border_width.clone()
            original_border_color = mob.border_color.clone()
            outline_color = (
                original_border_color if border_color is None else border_color
            )
            original_styles.append((colors, original_border_width, outline_color))

            for descendant, color in colors:
                descendant.set_non_recursive(color=_with_opacity(color, 0))
            mob.border_color = _with_opacity(outline_color, 0)
            if border_width is not None:
                mob.border_width = border_width

    with Lag(
        lag_ratio,
        run_time=run_time,
        rate_func=rate_func,
        animation_manager=animation_manager,
    ):
        for mob, (colors, original_border_width, outline_color) in zip(
            mobs, original_styles
        ):
            with Seq(animation_manager=animation_manager):
                with Off(animation_manager=animation_manager):
                    mob.border_color = outline_color
                mob.draw(1.0)
                with Sync(animation_manager=animation_manager):
                    for descendant, color in colors:
                        descendant.set_non_recursive(color=color)
                    mob.border_width = original_border_width

    return mobs
