#from camera import Sequential, Synchronized, Off
import torch

from algan.animation.animation_contexts import Seq, Sync, Off, ComposeRateFunc
from algan.utils.tensor_utils import dot_product


def map_mob_over_inputs(mob, animation_func, inputs, percent_shown=0.1):
    num_shown = int(len(inputs) * percent_shown)
    d = mob.location - inputs[0].location
    with Seq():
        for i in range(num_shown):
            inp = inputs[i]
            mob.location = inp.location + d
            with Sync():
                animation_func(mob, inp)
        with Off():
            mob.location = inputs[-num_shown]
        for i in range(-num_shown, -1):
            inp = inputs[i+1]
            mob.location = inp.location + d
            with Sync():
                animation_func(mob, inp)


def rfd(x, start_portion, run_time, lag_time):
    #return x
    x = x * (run_time+lag_time)
    #t = t.unsqueeze(-2)
    #x = x.unsqueeze(-1)#unsqueeze_right(x, t)
    return ((x - (start_portion*lag_time)).clamp_(min=0) / run_time).clamp_(max=1)


def animate_lagged_by_location(mobs, animation_func, direction, lag_duration=1):
    #dots = dot_product(direction, torch.cat([mob.location for mob in mobs]), dim=-1, keepdim=True)
    dots = [dot_product(direction, mob.location, dim=-1, keepdim=True) for mob in mobs]
    dotsc = torch.cat(dots, -2)
    min_dot, max_dot = dotsc.amin(-2, keepdim=True), dotsc.amax(-2, keepdim=True)
    ts = [(_ - min_dot) / (max_dot - min_dot).clamp_(min=1e-8) for _ in dots]
    #t = t * lag_duration

    amc = mobs[0].animation_manager.context
    run_time = max(amc.run_time_unit - lag_duration, 0)
    lag_duration = min(lag_duration, amc.run_time_unit - run_time)
    start_time = amc.current_time
    old_max_time = amc.end_time
    #amc.max_max_time = max(amc.max_time, start_time + (run_time + lag_duration))
    for i in range(len(mobs)):
        amc.current_time = (start_time + ts[i].amin()).item()
        rf = lambda x, t=ts[i], r=run_time, l=(lag_duration): rfd(x, t, r, l)#((x - t).clamp_(min=0) / lag_duration).clamp_(max=1)
        with ComposeRateFunc(rf, run_time=run_time+lag_duration):
            animation_func(mobs[i])
    amc.end_time = max(old_max_time, start_time + (run_time + lag_duration))
    amc.current_time = start_time + amc.lag_ratio * (run_time + lag_duration)
