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


def draw_border_then_fill(mobs, run_time=None, lag_ratio=None, border_width=2):
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
        Sets the Mobs' border_widths to this value prior to animation.
        If None, no change to border_widths is made. Defaults to 5.

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
    if not mobs:
        return mobs

    animation_manager = mobs[0].animation_manager
    length = len(mobs)
    if run_time is None:
        run_time = 1 if length < 15 else 2
    if lag_ratio is None:
        lag_ratio = min(4.0 / max(1.0, length), 0.2)

    with Off(animation_manager=animation_manager):
        for mob in mobs:
            mob.set_opacity_via_color(0)
            mob.border_color = _with_opacity(mob.border_color, 0)
            if border_width is not None:
                mob.border_width = border_width

    with Lag(
        lag_ratio,
        run_time=run_time,
        rate_func=rate_funcs.identity,
        animation_manager=animation_manager,
    ):
        for mob in mobs:
            with Seq(animation_manager=animation_manager):
                with Off(animation_manager=animation_manager):
                    mob.border_color = _with_opacity(mob.border_color, 1)
                mob.draw(1.0)
                mob.set_opacity_via_color(1)

    return mobs
