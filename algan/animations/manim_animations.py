from algan.animation_timeline.animation_contexts import *


def write(bezier_mob, border_width: float = 1, run_time=None, lag_ratio=None):
    """Plays an animation of the bezier_mob spawning as if being hand-drawn.

    Parameters
    ----------
    bezier_mob
        A mob created by ManimMob(mn.Text("some text")).
    border_width
        The width to set the border to for the drawing animation. If set to None the mob's original
        border_width will be used.

    Returns
    =======
    :class:`~.Mob`
        The Mob instance itself, allowing for method chaining.

    Examples
    ---------

    .. algan:: Example1MAnimationsWrite

        from algan import *

        x = ManimMob(mn.Text('Hello'))
        write(x)

        Scene.save_video()

    """
    length = len(bezier_mob.children[2])
    if run_time is None:
        run_time = 1 if length < 15 else 2
    if lag_ratio is None:
        lag_ratio = min(4.0 / max(1.0, length), 0.2)

    with Off(animation_manager=bezier_mob.animation_manager):
        bezier_mob.set_opacity_via_color(0)
        for character in bezier_mob.children[2]:
            if border_width is not None:
                character.border_width = border_width
            character.border_color = character.border_color.set_opacity(0)

    with Lag(lag_ratio, run_time=run_time, rate_func=rate_funcs.identity, animation_manager=bezier_mob.animation_manager):
        for character in bezier_mob.children[2]:
            with Seq(animation_manager=bezier_mob.animation_manager):
                with Off(animation_manager=bezier_mob.animation_manager):
                    character.border_color = character.border_color.set_opacity(1)
                character.draw(1.0)
                character.set_opacity_via_color(1)

    return bezier_mob

