import math
from collections import defaultdict

import torch

from algan import cast_to_tensor
from algan.animation.utils_taichi import _query_state_from_edits


def generate_array_states_taichi(times, N, edits):
    """
    Generates the state of an array given its history of edits.

    Args:
        times (Tensor): Shape [T], the inquiry times.
        N (int): Length of the output vector.
        edits (list of dicts): Each dict contains:
            - 'indexes': Tensor of shape [M_i] (values in [0, N-1])
            - 'values': Tensor of shape [M_i, D]
            - 'timestamp': float scalar
        future_only (bool): If True, finds the smallest timestamp > t.
                            If False, finds the largest timestamp <= t.
    """
    device = times.device
    T = times.shape[0]

    if len(edits) == 0:
        return torch.zeros((T, N, 1), dtype=torch.float32, device=device)

    D = edits[0]['values'].shape[1]
    dtype = edits[0]['values'].dtype

    # 1. Sort the edits chronologically in Python first.
    # Sorting a python list of size E (where E is the number of edits) takes negligible time.

    # 2. Extract timestamps and sizes of each edit
    edit_timestamps = torch.tensor([edit['timestamp'] for edit in edits], dtype=times.dtype, device=device)
    edit_sizes = torch.tensor([edit['indexes'].shape[0] for edit in edits], dtype=torch.int64, device=device)

    # 3. Flatten only the indices and values (no floating-point timestamp arrays are repeated)
    flat_indices = torch.cat([edit['indexes'].to(device) for edit in edits])
    flat_values = torch.cat([edit['values'].to(device) for edit in edits])

    # 4. Generate the edit IDs via PyTorch's native C++ repeat_interleave
    # We cast to int32 to optimize memory usage (halving the index footprint compared to int64)
    flat_edit_ids = torch.repeat_interleave(edit_sizes).to(torch.int32)

    # 5. Perform a single stable sort on flat_indices.
    # Because flat_edit_ids is already ascending, the stable sort preserves chronological order.
    perm = torch.argsort(flat_indices, stable=True)

    sorted_indices = flat_indices[perm]
    sorted_edit_ids = flat_edit_ids[perm]
    sorted_values = flat_values[perm]

    # 6. Build the CSR index boundaries
    grid = torch.arange(N + 1, dtype=torch.int64, device=device)
    head = torch.searchsorted(sorted_indices, grid)

    # 7. Execute the Taichi parallel kernel
    out = torch.zeros((T, N, D), dtype=dtype, device=device)
    _query_state_from_edits(times, head, sorted_edit_ids, edit_timestamps, sorted_values, out)

    return out


class AttributeTimeline:
    """
    A global timeline recording state and edit history of all Mobs for a particular attribute.
    """

    def __init__(self, channels, buffer_size=256, attr_name=None, record_end_points=False):
        self.attr_name = attr_name
        self.record_end_points = record_end_points
        self.current_state = torch.empty((1, buffer_size, channels)) # latest state after all edits.
        self.active_state = self.current_state # pointer to active state used to fulfil get requests.
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        self.pointer = 0
        self.edits = []
        self._is_ready_for_queries = False
        self.mob_id_to_inds = dict()
        self.mob_id_to_starts = dict()
        self.mob_id_to_ends = dict()

    def set_start_point(self, mob, starts):
        self.mob_id_to_starts[mob.id] = starts
        #self.end_points[:,inds,0] = starts

    def set_end_point(self, mob, ends):
        self.mob_id_to_ends[mob.id] = ends
        #inds = self.mob_id_to_inds[mob]
        #self.end_points[:,inds,1] = ends

    def get_current_values(self):
        return self.current_state[:, :self.pointer]

    def get(self, key):
        try:
            return self.active_state[self.active_time_inds, key]
        except:
            print('debug')

    def modify(self, key, value):
        try:
            self.active_state[self.active_time_inds, key] = value
        except:
            print('debug')
        self._is_ready_for_queries = False
        return self

    def record(self, key, value, time):
        old_value = self.get(key)
        self.edits.append([key, old_value, time])
        self.modify(key, value)
        return self

    def add(self, mob, values, overwrite=False):
        mob_id = mob.id
        if (not overwrite) and (mob_id in self.mob_id_to_inds):
            return

        values = cast_to_tensor(values)
        n = values.shape[-2]
        new_pointer = self.pointer + n
        buffer_size = self.current_state.shape[-2]
        if new_pointer >= buffer_size:
            while new_pointer >= buffer_size:
                buffer_size *= 2
            new_buffer = torch.empty((1, buffer_size, self.current_state.shape[-1]))
            new_buffer[:, :self.pointer] = self.get_current_values()
            self.current_state = new_buffer
            self.active_state = self.current_state
            if self.record_end_points:
                new_buffer = torch.empty((1, buffer_size, 2))
                new_buffer[:, :self.pointer] = self.end_points[:,:self.pointer]
                self.end_points = new_buffer
        self.current_state[:, self.pointer:new_pointer] = values
        inds = torch.arange(self.pointer, new_pointer)
        self.mob_id_to_inds[mob_id] = inds
        self.pointer = new_pointer
        return inds

    def prepare_for_queries(self):
        if self._is_ready_for_queries:
            return self
        self._is_ready_for_queries = True

        self._edits_sorted = [*(sorted([[k, v, time.end] for k, v, time in self.edits],
                                         key=lambda x: x[-1])),
                              (torch.arange(self.pointer), self.current_state[:,:self.pointer], math.inf)]
        self._edits_sorted = [{'indexes': k.view(-1), 'values': v.squeeze(0),
                               'timestamp': t} for k, v, t, in self._edits_sorted]

        if not self.record_end_points:
            return self
        self.end_points = torch.full((1, self.pointer + 1, 2), 1e12)
        for mob_id in self.mob_id_to_starts:
            inds = self.mob_id_to_inds[mob_id]
            self.end_points[:,inds,0] = self.mob_id_to_starts[mob_id].start()
        for mob_id in self.mob_id_to_ends:
            inds = self.mob_id_to_inds[mob_id]
            self.end_points[:,inds,1] = self.mob_id_to_ends[mob_id].end()
        return self

    def rematerialize_state_at_times(self, times):
        self.prepare_for_queries()
        self.active_state = generate_array_states_taichi(times, self.pointer+1, self._edits_sorted)
        if self.record_end_points:
            t = times.view(-1,1)
            self.active_state *= ((self.end_points[...,0] <= t) & (t < self.end_points[...,1])).unsqueeze(-1)
        self.rematerialized_times = times
        self.active_time_inds = slice(None, None, None)
        return self

    def set_active_time_inds(self, time_inds):
        self.active_time_inds = time_inds

    def clear_buffers(self):
        self.active_state = self.current_state
        self.active_time_inds = slice(None, None, None)
        self.rematerialized_times = None
        self._is_ready_for_queries = False
        return self

class TimelineEvent:
    def __init__(self, time, span):
        self.span = span
        self._time = time

    def __call__(self):
        return self.time

    @property
    def end(self):
        return self.time

    @property
    def time(self):
        return self.span.get_rescaled_time(self._time)

    @time.setter
    def time(self, time):
        self._time = time


class TimelineSpan:
    def __init__(self, start_time=0, end_time=0, current_time=0):
        self._rescaled_start = start_time
        self._rescaled_end = end_time
        self.original_start = start_time
        self.original_end = end_time
        self.current_time = current_time

    def __call__(self):
        return self.original_start

    def rescale(self, new_start, ratio):
        self.start = (self.start - new_start) * ratio + new_start
        self.end = (self.end - new_start) * ratio + new_start

    def get_rescaled_time(self, t):
        a = (t - self.original_start) / max(self.original_end - self.original_start, 1e-6)
        return self.start + (self.end - self.start) * a

    def get_time(self, time):
        return TimelineEvent(time, self)

    def get_current_time(self):
        return self.get_time(self.current_time)

    @property
    def start(self):
        return self._rescaled_start

    @start.setter
    def start(self, value):
        self._rescaled_start = value

    @property
    def end(self):
        return self._rescaled_end

    @end.setter
    def end(self, value):
        self._rescaled_end = value

class FunctionApplicationEvent:
    def __init__(self, function, caller, animated_args, kwargs, rate_func, time):
        self.function = function
        self.caller = caller
        self.animated_args = animated_args
        self.kwargs = kwargs
        self.rate_func = rate_func
        self.time = time

class FunctionTimeline:
    def __init__(self):
        self.function_applications = []

    def add(self, function_application):
        self.function_applications.append(function_application)

    def get_functions_for_times(self, times):
        return [f for f in self.function_applications if ((f.time.start <= times) &
                (times < f.time.end)).any()]

    def apply_functions_at_times(self, times):
        for f in self.get_functions_for_times(times):
            f.function(f.caller, *f.animated_args, *f.other_args)


class AnimationTimeline:
    def __init__(self):
        self.attr_to_timeline = dict()#defaultdict(AttributeTimeline)
        self.function_timeline = FunctionTimeline()

    def register_spawn(self, mob, time):
        self.attr_to_timeline['opacity'].set_start_point(mob, time)

    def register_despawn(self, mob, time):
        self.attr_to_timeline['opacity'].set_end_point(mob, time)

    def add_mob_attr(self, mob, attr, value, add_mob=True):
        if attr not in self.attr_to_timeline:
            self.attr_to_timeline[attr] = AttributeTimeline(value.shape[-1],
                                                            record_end_points=attr=='opacity')
        if not add_mob:
            return
        timeline = self.attr_to_timeline[attr]
        # if mob.id not in timeline.mob_id_to_inds:
        timeline.add(mob, value)
        return

    def get_inds(self, attr, mob, value=None):
        if attr not in self.attr_to_timeline:
            if value is None:
                raise AttributeError
            else:
                self.attr_to_timeline[attr] = AttributeTimeline(value.shape[-1])
        timeline = self.attr_to_timeline[attr]
        if mob.id not in timeline.mob_id_to_inds:
            if value is None:
                raise AttributeError
            else:
                timeline.add(mob, value)
        return self.attr_to_timeline[attr].mob_id_to_inds[mob.id]

    def get_attr(self, attr, inds):
        timeline = self.attr_to_timeline[attr]
        return timeline.get(inds)

    def record_function(self, function, caller, animated_args, kwargs, animation_context):
        c = animation_context
        if c.run_time_unit <= 0 or not c.record_funcs:
            return kwargs
        rate_func = c.rate_func
        rate_func_compose = c.rate_func_compose
        rf = rate_func
        if rate_func_compose is not None:
            rf = lambda x, rf=rate_func, rfc=rate_func_compose: rf(rfc(x))
        self.function_timeline.add(FunctionApplicationEvent(
            function, caller, animated_args, kwargs, rf, c.timespan))
        return kwargs

    def get_timeline_inds(self, mob, new_value, attr_name):
        try:
            timeline = self.attr_to_timeline[attr_name]
        except:
            print('debug')
        inds = None
        if mob.id not in timeline.mob_id_to_inds:
            inds = timeline.add(mob, new_value)
        return timeline, inds

    def modify_attribute_and_record(self, attr_name, mob_inds, new_value, time):
        timeline = self.attr_to_timeline[attr_name]
        timeline.record(mob_inds, new_value, time)
        return self

    def modify_attribute(self, attr_name, mob_inds, new_value):
        timeline = self.attr_to_timeline[attr_name]
        timeline.modify(mob_inds, new_value)
        return self

    def set_state_to_times(self, times):
        for attr, timeline in self.attr_to_timeline.items():
            if attr == 'location':
                print('debug')
            timeline.rematerialize_state_at_times(times)

        for f in self.function_timeline.get_functions_for_times(times):
            active_time_inds = ((f.time.start <= times) & (times < f.time.end)).nonzero()
            if active_time_inds.numel() == 0:
                continue
            for timeline in self.attr_to_timeline.values():
                timeline.set_active_time_inds(active_time_inds)

            s = f.time.start
            e = f.time.end
            a = (times[active_time_inds.squeeze(-1)] - s) / (e - s + 1e-6)
            a = a.view(-1,1,1)
            try:
                a = f.rate_func(a)
            except:
                print('debug')

            kwargs = {k: v for k, v in f.kwargs.items()}
            for k in f.animated_args:
                kwargs[k] = torch.lerp(cast_to_tensor(f.animated_args[k]), f.kwargs[k], a)

            f.function(f.caller, **kwargs)
        for timeline in self.attr_to_timeline.values():
            timeline.set_active_time_inds(slice(None, None, None))
        return self

    def clear_buffers(self):
        for t in self.attr_to_timeline.values():
            t.clear_buffers()



class TimelineManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call TimelineManager.instance() instead of TimelineManager().")

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = AnimationTimeline()
        return cls._instance
