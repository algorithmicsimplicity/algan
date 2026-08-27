"""Helpers for building animations over collections of Mobs.

``map_mob_over_inputs`` applies a function across a sequence of inputs to produce
one animation per input, and ``animate_lagged_by_location`` staggers a set of
animations by where each Mob sits in space, so a change sweeps across the screen
rather than happening everywhere at once.

These compose with the animation contexts rather than replacing them: the lag is
expressed as timing within the surrounding context.
"""

# from camera import Sequential, Synchronized, Off
from __future__ import annotations

import torch

from algan.animation_timeline.animation_contexts import (
    ComposeRateFunc,
    Off,
    Seq,
    Sync,
    animation_manager_for,
)
from algan.utils.tensor_utils import dot_product


def map_mob_over_inputs(mob, animation_func, inputs, percent_shown=0.1):
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


def rfd(x, start_portion, run_time, lag_time):
    # return x
    x = x * (run_time + lag_time)
    # t = t.unsqueeze(-2)
    # x = x.unsqueeze(-1)#unsqueeze_right(x, t)
    return ((x - (start_portion * lag_time)).clamp_(min=0) / run_time).clamp_(max=1)


def animate_lagged_by_location(mobs, animation_func, direction, lag_duration=1):
    if not mobs:
        # Nothing to stagger. Reachable from ordinary authoring: Text("") and
        # Text("   ") produce no glyphs, and their entrance wave used to die
        # here on ``torch.cat`` of an empty list -- a torch error for a string
        # that simply has nothing in it.
        return
    # dots = dot_product(direction, torch.cat([mob.location for mob in mobs]), dim=-1, keepdim=True)
    dots = [dot_product(direction, mob.location, dim=-1, keepdim=True) for mob in mobs]
    dotsc = torch.cat(dots, -2)
    min_dot, max_dot = dotsc.amin(-2, keepdim=True), dotsc.amax(-2, keepdim=True)

    amc = mobs[0].animation_manager.context
    ts = [((_ - min_dot) / (max_dot - min_dot).clamp_(min=1e-8)) for _ in dots]
    # t = t * lag_duration

    run_time = amc.run_time_unit  # max(amc.run_time_unit - lag_duration, 0)
    # lag_duration = min(lag_duration, amc.run_time_unit - run_time)
    start_time = amc.timespan.current_time
    old_max_time = amc.timespan.original_end
    # amc.max_max_time = max(amc.max_time, start_time + (run_time + lag_duration))
    for i in range(len(mobs)):
        # ``rf`` already delays every attribute row by its normalized spatial
        # position.  Starting each primitive at its own minimum position would
        # apply that offset a second time, so a composite wave would travel
        # faster across one wide primitive than across many small ones (for
        # example, a Code panel versus its separately represented glyphs).
        amc.timespan.current_time = start_time

        def rf(x, t=ts[i], r=run_time, lag=lag_duration):
            return rfd(
                x, t, r, lag
            )  # ((x - t).clamp_(min=0) / lag_duration).clamp_(max=1)

        with ComposeRateFunc(
            rf,
            run_time=run_time + lag_duration,
            animation_manager=animation_manager_for(mobs),
        ):
            animation_func(mobs[i])
    amc.timespan.original_end_time = max(
        old_max_time, start_time + (run_time + lag_duration)
    )
    amc.timespan.current_time = start_time + amc.lag_ratio * (run_time + lag_duration)
