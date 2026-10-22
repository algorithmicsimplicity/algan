"""Helpers for building animations over collections of Mobs.

``map_mob_over_inputs`` applies a function across a sequence of inputs to produce
one animation per input, and ``animate_lagged_by_location`` staggers a set of
animations by where each Mob sits in space, so a change sweeps across the screen
rather than happening everywhere at once.

These compose with the animation contexts rather than replacing them: the lag is
expressed as timing within the surrounding context.
"""

from __future__ import annotations

import torch

from algan.animation_timeline.animation_contexts import (
    ComposedEasing,
    Off,
    Seq,
    Sync,
    animation_manager_for,
)
from algan.utils.tensor_utils import dot_product


def map_mob_over_inputs(mob, animation_func, inputs, percent_shown=0.1):
    """Animate ``mob`` stepping across a sequence of input Mobs.

    Only the start and end of the sequence are shown: ``mob`` visits the first
    ``int(len(inputs) * percent_shown)`` inputs one after another, then jumps
    (un-animated) towards the end of the sequence and visits the last
    ``num_shown - 1`` inputs. At each input it is moved to that input's
    location plus its initial offset from ``inputs[0]``, and
    ``animation_func(mob, input)`` is played. Steps run in sequence.

    Parameters
    ----------
    mob : Mob
        The Mob that moves over the inputs.
    animation_func : callable
        Called as ``animation_func(mob, input)`` at each visited input.
    inputs : sequence of Mob
        The inputs to step over, in order.
    percent_shown : float, default 0.1
        Fraction of ``inputs`` visited at each end of the sequence.
    """
    num_shown = int(len(inputs) * percent_shown)
    d = mob.location - inputs[0].location
    with Seq(animation_manager=animation_manager_for(mob, inputs)):
        for i in range(num_shown):
            inp = inputs[i]
            mob.location = inp.location + d
            with Sync(animation_manager=animation_manager_for(mob, inputs)):
                animation_func(mob, inp)
        with Off(animation_manager=animation_manager_for(mob, inputs)):
            mob.location = inputs[-num_shown]
        for i in range(-num_shown, -1):
            inp = inputs[i + 1]
            mob.location = inp.location + d
            with Sync(animation_manager=animation_manager_for(mob, inputs)):
                animation_func(mob, inp)


def rfd(x, start_portion, runtime, lag_time):
    """Return the lagged progress of one element of a staggered animation.

    Maps overall progress ``x`` in ``[0, 1]`` across a span of
    ``runtime + lag_time`` to this element's own progress: the element starts
    after ``start_portion * lag_time`` and then runs for ``runtime``. The
    result is clamped to ``[0, 1]``. Used as the easing of each Mob by
    :func:`animate_lagged_by_location`.

    Parameters
    ----------
    x : torch.Tensor
        Overall progress in ``[0, 1]``.
    start_portion : torch.Tensor or float
        Position of the element in the stagger, in ``[0, 1]``.
    runtime : float
        Duration of each element's own animation.
    lag_time : float
        Delay between the first and last element starting.

    Returns
    -------
    torch.Tensor
        The element's progress in ``[0, 1]``.
    """
    x = x * (runtime + lag_time)
    return ((x - (start_portion * lag_time)).clamp_(min=0) / runtime).clamp_(max=1)


def animate_lagged_by_location(mobs, animation_func, direction, lag_duration=1):
    """Play ``animation_func`` on each Mob, staggered by position along ``direction``.

    Each Mob's start is delayed in proportion to where its location projects
    onto ``direction``, normalized so the lowest projection starts first and
    the highest starts ``lag_duration`` later; the change therefore sweeps
    across the Mobs as a wave. Each Mob animates for the surrounding
    context's runtime per part. The enclosing block is extended to cover the
    whole wave. Does nothing when ``mobs`` is empty.

    Parameters
    ----------
    mobs : sequence of Mob
        The Mobs to animate.
    animation_func : callable
        Called as ``animation_func(mob)`` for each Mob; the animations it
        records are retimed by the stagger.
    direction : torch.Tensor
        Direction of the sweep, last dim 3. Its length does not change the
        timing, since projections are normalized.
    lag_duration : float, default 1
        Time in seconds between the first and last Mob starting.
    """
    if not mobs:
        # Nothing to stagger. Reachable from ordinary authoring: Text("") and
        # Text("   ") produce no glyphs, and their entrance wave used to die
        # here on ``torch.cat`` of an empty list -- a torch error for a string
        # that simply has nothing in it.
        return
    dots = [dot_product(direction, mob.location, dim=-1, keepdim=True) for mob in mobs]
    dotsc = torch.cat(dots, -2)
    min_dot, max_dot = dotsc.amin(-2, keepdim=True), dotsc.amax(-2, keepdim=True)

    amc = mobs[0].animation_manager.context
    ts = [((_ - min_dot) / (max_dot - min_dot).clamp_(min=1e-8)) for _ in dots]
    runtime = amc.runtime_per_part  # max(amc.runtime_per_part - lag_duration, 0)
    start_time = amc.timespan.current_time
    for i in range(len(mobs)):
        # ``rf`` already delays every attribute row by its normalized spatial
        # position.  Starting each primitive at its own minimum position would
        # apply that offset a second time, so a composite wave would travel
        # faster across one wide primitive than across many small ones (for
        # example, a Code panel versus its separately represented glyphs).
        amc.timespan.current_time = start_time

        def rf(x, t=ts[i], r=runtime, lag=lag_duration):
            return rfd(
                x, t, r, lag
            )  # ((x - t).clamp_(min=0) / lag_duration).clamp_(max=1)

        with ComposedEasing(
            rf,
            runtime=runtime + lag_duration,
            animation_manager=animation_manager_for(mobs),
        ):
            animation_func(mobs[i])
    # The wave covers one lagged runtime from where it started, and the block
    # containing it has to be at least that long. Written the way every other
    # recorded animation extends its context -- against the span's current end,
    # so a wave can only ever lengthen the block, never cut short what was
    # recorded inside it. (This used to assign ``original_end_time``, which is
    # not a field of TimelineSpan: the attribute stuck to the span object, no
    # reader ever looked for it, and the extension simply did not happen.)
    amc.timespan.original_end = max(
        amc.timespan.original_end, start_time + (runtime + lag_duration)
    )
    amc.timespan.current_time = start_time + amc.lag_ratio * (runtime + lag_duration)
