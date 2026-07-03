from collections import defaultdict
import copy
import functools
from functools import wraps
import inspect
from typing import Dict

import torch
import torch.nn.functional as F

from algan.scene import Scene
from algan.animation.animation_contexts import (
    Sync,
    AnimationManager,
    AnimationContext,
    Off,
)
from algan.animation.global_state import GlobalAnimationState, RowBlock
from algan.constants.color import BLACK
from algan.utils.tensor_utils import (
    broadcast_all,
    robust_concat,
    concat_dicts,
    prepare_kwargs,
    HANDLED_FUNCTIONS, wait_for_cuda,
)
from algan import SceneManager, compiled
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import (
    broadcast_gather,
    cast_to_tensor,
    cast_to_tensor_single,
    unsqueeze_dims,
)


TIME_PARAMETER_NAME = "time_elapsed"


class TimeInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class ModificationHistory:
    """A record of every animated_function applied to a particular Mob, and the
    timestamps those changes occur over.  At render time this history is used
    to re-execute the functions with interpolated parameters.

    Attribute-value modifications are no longer stored here: they are recorded
    in the :class:`~.GlobalAnimationState` modification log (row-indexed into
    the global attribute buffers), which materializes the pre-function state of
    every mob in one batched pass.  ``attribute_modifications`` remains as a
    lightweight marker dict (attr name -> list of end-time callables) so
    callers can still ask *whether* an attribute was ever modified.
    """

    def __init__(self):
        self.function_applications = (
            dict()
        )  # contains all animated_functions applied to the mob.
        self.attribute_modifications = defaultdict(
            list
        )  # marker only: which attrs were modified, and when.
        self.attribute_overwrites = (
            dict()
        )  # overwrites are not supported at the moment.
        self.most_recent_function_added = None
        self.cached_history = None

    def overwrite_attr_history(self, attr, end_time):
        if attr not in self.attribute_overwrites:
            self.attribute_overwrites[attr] = []
        self.attribute_overwrites[attr].append(end_time)

    def insert_function_application(
        self, func_name, func, animated_args, kwargs, animation_context
    ):
        if animation_context.run_time_unit <= 0 or not animation_context.record_funcs:
            return
        start_time = animation_context.get_current_time()
        end_time = animation_context.get_current_end_time()
        rate_func = animation_context.rate_func
        rate_func_compose = animation_context.rate_func_compose
        prio = 1 if (animation_context.updater is not None and animation_context.updater) else 0
        func_name = f'{prio}_{func_name}'

        if func_name not in self.function_applications:
            self.function_applications[func_name] = [func, [], [], -1]
        self.function_applications[func_name][2].append(
            [
                animated_args,
                kwargs,
                start_time,
                end_time,
                end_time,
                (rate_func, rate_func_compose),
            ]
        )
        self.function_applications[func_name][3] = (
            AnimationManager.get_execution_count() + prio * 1e12
        )
        self.most_recent_function_added = self.function_applications[func_name][2][-1]
        self.cached_history = None

    def get_func_history(self, animatable):
        """The animated_function history, batched and split into
        non-overlapping groups, ready for re-execution at render time.
        (Attribute-value history is materialized globally, see
        :class:`~.GlobalAnimationState`.)"""
        if self.cached_history is not None:
            return self.cached_history

        funcs = []

        # For overlapping animations, if one ends in the middle of another, we need to extend
        # the end time to match the end of the longest animation, otherwise the remaining portion
        # of the time won't be animated.
        # TODO rewrite this into non-awful code.
        for func_name_outer, func_data_outer in sorted(
            self.function_applications.items(),
            key=lambda _: (
                0 if "setattr_" in _[0] else (1 if "update_relative" in _[0] else 2)
            ),
        ):
            for func_name, func_data in self.function_applications.items():
                for func_application_outer in func_data_outer[2]:
                    for func_application in func_data[2]:
                        if (
                            func_application[2]()
                            < func_application_outer[2]()
                            < func_application_outer[4]()
                            < func_application[3]()
                        ):
                            func_application_outer[4] = func_application[3]

        for func_name, (func, modified_attrs, arg_list, execution_order) in sorted(
            self.function_applications.items(),
            key=lambda _: (
                0 if "setattr_" in _[0] else (1 if "update_relative" in _[0] else (2 if _[0][0] == '0' else 3))
            ),
        ):  # , _[1][-1])):
            (
                animated_args,
                kwargs,
                start_time,
                end_time,
                extended_end_time,
                rate_funcs,
            ) = zip(*arg_list)
            kwargs = concat_dicts(kwargs)
            animated_args = concat_dicts(animated_args)
            animated_args = {
                k: unsqueeze_dims(v, kwargs[k], 1) for k, v in animated_args.items()
            }  # TODO shouldn't the unsqueeze dim be 0?
            start_time, end_time, extended_end_time = (
                torch.tensor([_() for _ in start_time]),
                torch.tensor([_() for _ in end_time]),
                torch.tensor([_() for _ in extended_end_time]),
            )

            # In the event where 2 or more functions take place at the same time (overlap), we manually split them up.
            # into separate funcs.
            overlaps = (
                ~(
                    (start_time > extended_end_time.unsqueeze(-1))
                    | (extended_end_time < start_time.unsqueeze(-1))
                )
            ).float()
            orders = (overlaps * torch.triu(torch.ones_like(overlaps), diagonal=1)).sum(
                -2
            )
            unique_orders, uinds = torch.unique(orders, return_inverse=True)
            for order in unique_orders:  # [-1:]:

                def g(x):
                    if not isinstance(x, torch.Tensor):
                        return x
                    return x[uinds == order]

                sub_rate_funcs = [rate_funcs[i] for i in (uinds == order).nonzero()]
                funcs.append(
                    (
                        func,
                        {k: g(v) for k, v in animated_args.items()},
                        {k: g(v) for k, v in kwargs.items()},
                        g(start_time),
                        g(end_time),
                        g(extended_end_time),
                        sub_rate_funcs,
                    )
                )

        self.cached_history = funcs
        return funcs


def animated_function(
    function=None, *, animated_args: Dict[str, float] = dict(), unique_args=list()
):
    """Decorator that turns a function into an animated function. The animation is created by interpolating
    all args named in the animated_args dict from the value provided in this dict the value passed as an actual argument
    when the function is called. Most commonly, animated_args will just be {'t': 0}, and the function
    will be called with t=1.

    Parameters
    ----------
    function
        The function to be decorated. It MUST accept a :class:`~.Mob` as its first argument, and any arguments
        given in `animated_args` or `unique_args` must also be arguments of this function.

    animated_args
        A dictionary with strings as keys and floats as values. The strings are names of arguments which will
        be animated. The arguments will be animated by linearly interpolating their values from the corresponding
        value provided in the animated_args dict to the value they have when the function is called.

    unique_args
        A list of strings. This is only for batching, when the function is called with different values for a unique
        argument, they will be batched as two entirely separate functions. Any arguments named in `unique_args` MUST
        only accept string values.
    """

    def _decorate(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            if self.animation_manager.context.trace_mode:
                return func(self, *args, **kwargs)
            if not self.is_animating():
                with AnimationContext(record_funcs=False):
                    return func(self, *args, **kwargs)
            with AnimationContext():
                kwargs = prepare_kwargs(
                    self, func, args, kwargs, animated_args, unique_args
                )
                with AnimationContext(record_funcs=False):
                    out = func(self, **kwargs)
            self.animation_manager.context.increment_times()
            return out

        return wrapper_func

    if function:
        return _decorate(function)
    return _decorate


class _ActiveView:
    """Dict-like view of a mob's *current* attribute values, backed by the
    global attribute buffers.  ``view[attr]`` reads the mob's rows as a
    ``[1, B, W]`` tensor; ``view[attr] = value`` writes the rows in place
    (allocating or resizing them as needed) without recording a modification,
    exactly like the old direct ``data_dict_active[attr] = value`` assignment.
    Values that cannot live in the buffers (non-tensors, non-float dtypes)
    fall back to a per-mob side dict."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __contains__(self, key):
        return key in self.data.rows or key in self.data.side

    def __getitem__(self, key):
        d = self.data
        if key in d.side:
            return d.side[key]
        rows = d.rows.get(key)
        if rows is None:
            raise KeyError(key)
        return d.read_current(key, rows)

    def __setitem__(self, key, value):
        self.data.write_current(key, value)

    def keys(self):
        seen = list(self.data.rows.keys())
        return seen + [k for k in self.data.side.keys() if k not in self.data.rows]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def get(self, key, default=None):
        return self[key] if key in self else default


class _WindowView:
    """Dict-like view of a mob's *materialized* (per-frame) attribute values,
    backed by the bound :class:`~.MaterializedWindow`.  Reads return
    ``[T, B, W]`` for attributes with recorded modifications and the constant
    ``[1, B, W]`` current value otherwise (the caller broadcasts constants
    over time, as before).  Membership additionally mirrors the old
    materialized-dict quirk: an attribute counts as present only once it has
    per-frame (dense) state in the window."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __contains__(self, key):
        d = self.data
        if key in d.side:
            return True
        rows = d.rows.get(key)
        if rows is None or d.window is None:
            return False
        return d.window.any_dense(rows.buffer, rows.indices)

    def __getitem__(self, key):
        d = self.data
        if key in d.side:
            return d.side[key]
        rows = d.rows.get(key)
        if rows is None or d.window is None:
            raise KeyError(key)
        value = d.window.read(rows.buffer, rows.indices)
        cls = d.attr_cls.get(key)
        if cls is not None:
            value = value.as_subclass(cls)
        return value

    def __setitem__(self, key, value):
        d = self.data
        rows = d.rows.get(key)
        if rows is None or d.window is None or not isinstance(value, torch.Tensor):
            d.write_current(key, value)
            return
        # Write over all frames of the window.
        value = cast_to_tensor(value)
        d.window.write(rows.buffer, rows.indices, value)

    def keys(self):
        return self.data.data_dict_active.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def values(self):
        return [self.data.data_dict_active[k] for k in self.keys()]

    def items(self):
        return [(k, self.data.data_dict_active[k]) for k in self.keys()]

    def get(self, key, default=None):
        return self[key] if key in self else default


class AnimatableData:
    """A container for all of the animation-relevant data for a mob.

    The mob's animated attribute values live in rows of the global attribute
    buffers (:class:`~.GlobalAnimationState`); this object holds the row
    allocations (``rows``: attr name -> :class:`~.RowBlock`), the per-mob
    function history, and the binding to the current render-time
    materialization window.

    ``data_dict_active`` / ``data_dict_materialized`` keep their old roles as
    the current-value and per-frame views of the data, but are now adapters
    over the global buffers / the materialized window rather than plain dicts.

    Parameters
    ----------
    animatable
        The mob for which we are recording data.
    history
        The ModificationHistory object to which function applications will be recorded.
    time_inds_materialized
        The time-inds materialized by the bound window.
    time_inds_active
        The currently active subset of the materialized time-inds.
    spawn_time
        (function which yields) the timestamp at which the mob spawned.
    despawn_time
        (function which yields) the timestamp at which the mob despawned.
    """

    def __init__(
        self,
        animatable,
        data_dict_active=None,
        data_dict_materialized=None,
        history=None,
        time_inds_materialized=None,
        time_inds_active=None,
        spawn_time=lambda: -1,
        despawn_time=lambda: -1,
    ):
        self.animatable = animatable
        if history is None:
            history = ModificationHistory()
        self.rows = dict()
        self.side = dict()
        self.attr_cls = dict()
        self.window = None
        self.stale_window = None
        self.data_dict_active = _ActiveView(self)
        self.data_dict_materialized = _WindowView(self)
        if data_dict_active:
            for k, v in data_dict_active.items():
                self.data_dict_active[k] = v
        self.history = history
        self.time_inds_active = time_inds_active
        self.time_inds_materialized = time_inds_materialized
        self.data_dict = self.data_dict_active
        self.spawn_time = spawn_time
        self.despawn_time = despawn_time
        self.set_pre_function_application = False

    # -- global-buffer plumbing --------------------------------------------

    def _buffer_for(self, key, width):
        return GlobalAnimationState.instance().get_buffer(key, width)

    def read_current(self, key, rows=None):
        """Current value of ``key`` as a [1, B, W] snapshot of the global
        buffer rows (a copy: later writes must not mutate captured reads)."""
        if rows is None:
            rows = self.rows[key]
        value = rows.read(snapshot=True).unsqueeze(0)
        cls = self.attr_cls.get(key)
        if cls is not None:
            value = value.as_subclass(cls)
        return value

    def write_current(self, key, value):
        """Set the current value of ``key`` without recording a modification
        (the old direct dict-assignment semantics).  Allocates rows on first
        write; re-allocates on batch-size or width change."""
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.get_default_dtype()
            or value.dim() > 3
        ):
            if key not in self.side or key in self.rows:
                GlobalAnimationState.instance().bump_topology()
            self.side[key] = value
            self.rows.pop(key, None)
            return
        self.side.pop(key, None)
        if type(value) is not torch.Tensor:
            self.attr_cls[key] = type(value)
        while value.dim() < 3:
            value = value.unsqueeze(0)
        value = value[-1]  # [B, W]; current value keeps the latest timestep.
        n, width = value.shape
        rows = self.rows.get(key)
        if rows is None or rows.size != n or rows.buffer.width != width:
            buffer = self._buffer_for(key, width)
            old_rows = rows
            rows = buffer.alloc(n)
            if old_rows is not None and old_rows.buffer is buffer:
                # Batch-size change of an existing attribute (become / batch
                # expansion): carry the recorded history over to the new rows
                # so pre-existing animation still materializes. Rows beyond
                # the old batch inherit the last old row's history (the old
                # pad-with-last broadcast semantics).
                m = min(old_rows.size, n)
                buffer.copy_row_history(
                    old_rows.indices[:m], rows.indices[:m]
                )
                if n > m:
                    buffer._duplicate_row_history(
                        int(old_rows.indices[-1]), rows.indices[m:]
                    )
            self.rows[key] = rows
            GlobalAnimationState.instance().bump_topology()
        rows.write(value)

    def alloc_rows_like(self, key, value_2d):
        """Allocate rows for ``key`` initialised to ``value_2d`` [n, W]."""
        buffer = self._buffer_for(key, value_2d.shape[-1])
        rows = buffer.alloc(value_2d.shape[0])
        rows.write(value_2d)
        self.rows[key] = rows
        GlobalAnimationState.instance().bump_topology()
        return rows

    def grow_rows(self, key, n_total):
        """Grow ``key``'s rows to ``n_total`` by padding with the last row's
        value; the new rows inherit the last row's modification history (this
        matches the old pad-with-last batch expansion)."""
        rows = self.rows[key]
        n_extra = n_total - rows.size
        if n_extra <= 0:
            return rows
        last_row = int(rows.indices[-1])
        init = rows.buffer.values[last_row:last_row + 1].expand(n_extra, -1)
        rows.buffer.grow(
            rows, n_extra, init.clone(), inherit_history_from_row=last_row
        )
        GlobalAnimationState.instance().bump_topology()
        return rows

    def record_modification(self, key, rows_idx, old_values, animation_context):
        """Record a batched modification (the values ``rows_idx`` held before
        this write) into the global modification log."""
        if not animation_context.record_attr_modifications:
            return
        start_time = animation_context.get_current_time()
        end_time = animation_context.get_current_end_time()
        animation_context.end_time = max(
            animation_context.end_time,
            animation_context.current_time + animation_context.run_time_unit,
        )
        rows = self.rows[key]
        rows.buffer.record(rows_idx, old_values, start_time, end_time)
        self.history.attribute_modifications[key].append(end_time)
        self.history.cached_history = None

    def __deepcopy__(self, memo):
        # Fresh row allocations holding a copy of the current values; the
        # caller (Animatable.__deepcopy__) decides about history sharing and
        # re-assigns animatable/history/spawn times.
        clone = AnimatableData(self.animatable)
        for key, rows in self.rows.items():
            clone.alloc_rows_like(key, rows.read().clone())
        clone.side = copy.deepcopy(self.side, memo)
        clone.attr_cls = dict(self.attr_cls)
        clone.history = self.history
        return clone

    def copy_history_from(self, source):
        """Replay the recorded modifications of ``source``'s rows onto this
        data's rows (used by clones that share their source's history)."""
        for key, rows in self.rows.items():
            src = source.rows.get(key)
            if src is None or src.size != rows.size:
                continue
            rows.buffer.copy_row_history(src.indices, rows.indices)


class Animatable:
    """Base class for anything that needs animation.

    Parameters
    ----------
    scene
        The Scene to which this animatable should (possibly) be added.
    add_to_scene
        Whether this animatable should be added to the scene.
    name
        The name of this animatable.
    init
        Whether this animatable should be initialized.
    animation_manager
        The AnimationManager that will control animations applied to this animatable.
    data
        The AnimatableData which will record animatable attribute values for this animatable.
    data_sub_inds
        Specifies which indexes in data dictionaries this animatable will read and write from.
        Used to implement multiple sub-mobs which all share the same underlying data tensors, for batching purposes.
    parent_batch_sizes
        If this animatable's parent is batched, parent_batch_sizes specifies how the parent's attribute modifications
        will be expanded for this animatable's attributes.
    is_primitive
        Whether this animatable is a rendering primitive, i.e. needs to be kept around at rendering time.

    Attributes
    ----------

    animatable_attrs : Set[String]
        A set of attribute names which will be treated as animatable. When ever an animatable attribute is modified,
        it will be treated as applying an animated function to this mob.
    """

    def __init__(
        self,
        scene: Scene | None = None,
        add_to_scene: bool = True,
        name: str = "_",
        init: bool = True,
        animation_manager: AnimationManager | None = None,
        data: AnimatableData | None = None,
        data_sub_inds: torch.Tensor | None = None,
        parent_batch_sizes: torch.Tensor | None = None,
        is_primitive: bool = False,
    ):
        if not hasattr(self, "animatable_attrs"):
            self.animatable_attrs = []

        self.generate_animatable_attr_set_get_methods()

        if scene is None:
            scene = SceneManager.instance()
        self.scene = scene
        self.id = self.scene.get_new_id()
        if add_to_scene:
            self.scene.add_actor(self)

        if animation_manager is None:
            animation_manager = AnimationManager.instance()
        self.animation_manager = animation_manager
        if add_to_scene:
            animation_manager.context.add_mob(self)
        self.name = name

        if data is None:
            data = AnimatableData(self)
        self.data = data

        self.anchor_priority = 0

        self.children = []
        self.components = []
        self.parents = []
        self.traversable = True
        self.parent_batch_sizes = parent_batch_sizes

        self.data_sub_inds = data_sub_inds
        self.batch_size = max(
            [1, *[_.shape[-2] for _ in self.data.data_dict_active.values()]]
        )
        self.is_primitive = is_primitive

        for attr in self.animatable_attrs:
            uattr = f"_{attr}"
            if hasattr(self, uattr):
                self.data.data_dict_active[attr] = self.__getattribute__(uattr)
                delattr(self, uattr)
        # setup_getters(self)
        self.previous_retroactive_time = 0
        self.reset_state()
        self.passive_animations = []
        self._passive_animation_functions = []
        self.ignore_wave_animations = False

        if init:
            self.init()

    def to(self, device):
        # Attribute values live in the global animation buffers (which stay on
        # the animation device); only side-stored tensors are moved.
        for attr in self.animatable_attrs:
            if attr in self.data.side and isinstance(
                self.data.side[attr], torch.Tensor
            ):
                self.data.side[attr] = self.data.side[attr].to(device)
        return self

    def _set_dependant_mobs_time_inds_to_self_then_run_function(self, function):
        with AnimationContext(trace_mode=True) as context:
            function()
            dependent_mobs = [
                _
                for _ in (
                    sorted(
                        context.traced_mobs,
                        key=lambda x: x.anchor_priority,
                        reverse=True,
                    )
                )
                if _ != self
            ]

        materialized_mobs = []
        for mob in dependent_mobs:
            # mob.anchor_priority = max(mob.anchor_priority, self.anchor_priority + 1)
            if hasattr(self, "raw_s"):
                if mob.set_state_full(self.raw_s, self.raw_e):
                    materialized_mobs.append(mob)
            mob.set_time_inds_to(self)
        function()

    @animated_function(animated_args={"t": 0}, unique_args=["function"])
    def animate_function(self, function, t=1, *args, **kwargs):
        """Animates the application of function, interpolating its animated parameter from 0 to t.

        Parameters
        ----------
        function
            The function to animate. It must accept a mob as its first parameter, and a float as its second parameter.
            During the animation, the second parameter will be interpolated from 0 to t.
        t
            The final value that the animated parameter will have at the end of the animation.
        *args, **kwargs
            Passed to function.

        """
        self._set_dependant_mobs_time_inds_to_self_then_run_function(
            lambda: function(self, t, *args, **kwargs)
        )
        return self

    @animated_function(unique_args=["function"])
    def animate_function_of_time(self, function, time_elapsed=0, *args, **kwargs):
        """Same as :meth:`~.Animatable.animate_function` but the animation parameter is equal
        to time elapsed since starting the animation, instead of interpolating 0 to t over the animation duration.
        This formulation can be useful when you don't know how long an animation will play for,
        and you want it to play indefinitely.

        Parameters
        ----------
        function
            The function to animate. It must accept a mob as its first parameter, and a float as its second parameter.
            During the animation, the second parameter will range from 0 to the duration
            of the animation (in seconds).
        time_elapsed
            Dummy parameter. No matter what value you give it, it will be overwritten with
            the time elapsed since the animation beginning.
        *args, **kwargs
            Passed to function.

        """
        start_time = self.animation_manager.context.current_time
        end_time = self.animation_manager.context.end_time
        self._set_dependant_mobs_time_inds_to_self_then_run_function(
            lambda: function(self, cast_to_tensor(time_elapsed), *args, **kwargs)
        )
        self.animation_manager.context.current_time = start_time
        self.animation_manager.context.end_time = end_time
        return self

    def add_updater(self, update_function, *args, **kwargs):
        """Adds a function to this Mob's collection of updaters. During animation, at every
        frame all of the Mob's updaters are executed, with the time elapsed since being added (in seconds)
        passed as the second parameter to each updater. Useful for implementing permanent or 'idle' animations.

        Parameters
        ----------
        update_function
            The function which will be called every frame. It must accept a Mob as its first parameter and
            a float as its second parameter. During animation it will be called with this mob
            as the first parameter and the time elapsed (in seconds) as the second parameter.
        *args, **kwargs
            Passed to update_function.

        Returns
        -------
            An integer ID identifying the updater that was added. This ID can be used to remove
            the updater at a later time, using :meth:`~.Animatable.remove_updater` .
             If it is never ignored, the updater will continue forever.

        """
        start_pointer = self.animation_manager.context.get_current_time()
        start_time = self.animation_manager.context.current_time
        end_time = self.animation_manager.context.end_time
        with AnimationContext(record_funcs=True, updater=True):
            self.animate_function_of_time(update_function, *args, **kwargs)
        self.passive_animations.append(self.data.history.most_recent_function_added)
        self.animation_manager.context.current_time = start_time
        self.animation_manager.context.end_time = end_time
        self._passive_animation_functions.append(
            lambda mob, t: update_function(mob, cast_to_tensor(t), *args, **kwargs)
        )
        self.passive_animations[-1][2] = start_pointer
        self.passive_animations[-1][3] = (
            lambda: 1e13 + 1
        )  # last forever, unless remove_updater is called to set it to an earlier time.
        self.passive_animations[-1][4] = lambda: 1e13 + 1
        return len(self.passive_animations) - 1

    def remove_updater(self, updater_id):
        """Removes the specified updater from this mobs updater, leaving the mob with whatever state the updater left
        it with at the time-stamp when it was removed.

        Parameters
        ----------
        updater_id
            The identifier of the updater to be removed. Can be given a value of -1 to remove the most
            recently added updater.

        """
        i = updater_id
        if self.passive_animations[i][3]() < 1e13:
            # Make sure that we only remove it the first time.
            return
        self.passive_animations[i][3] = (
            self.animation_manager.context.get_current_time()
        )
        self.passive_animations[i][4] = (
            self.animation_manager.context.get_current_time()
        )
        # self.set_state_to_time_t(self.passive_animations[i][3])
        with Off(record_funcs=False):
            self._passive_animation_functions[i](
                self, self.passive_animations[i][3]() - self.passive_animations[i][2]()
            )

    def remove_all_passive_animations(self):
        for i in range(len(self.passive_animations) - 1, -1, -1):
            self.remove_passive_animation(i)

    def set_to_retroactive(self):
        prt = self.animation_manager.context.current_time
        self.animation_manager.context.current_time = self.previous_retroactive_time
        self.previous_retroactive_time = prt

    def set_to_current(self):
        self.animation_manager.context.current_time = self.previous_retroactive_time

    @property
    def animation_manager(self):
        if not hasattr(self, "_animation_manager"):
            return AnimationManager.instance()
        return self._animation_manager

    @animation_manager.setter
    def animation_manager(self, a):
        self._animation_manager = a

    def is_animating(self):
        if not (hasattr(self, "animation_manager") and hasattr(self, "data")):
            return False
        return (
            self.animation_manager.context.record_funcs and self.data.spawn_time() >= 0
        )

    def generate_animatable_attr_set_get_methods(self):
        for attr in self.animatable_attrs:

            def setattr_general(value, attr=attr, self=self, recursive=True):
                if recursive:
                    self.__setattr__(attr, value)
                else:
                    self.setattr_non_recursive(attr, value)
                return self

            super().__setattr__(f"set_{attr}", setattr_general)
            super().__setattr__(
                f"get_{attr}", lambda attr=attr: self.__getattribute__(attr)
            )

    def setattr_and_record_modification(self, key, value):
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs.add(self)
            return
        d = self.data
        if not isinstance(value, torch.Tensor):
            d.side[key] = value
            return
        self.batch_size = max(self.batch_size, value.shape[-2])
        n2 = self.batch_size
        at_render = d.time_inds_materialized is not None
        rows = d.rows.get(key)
        if rows is None:
            if not at_render:
                # This is the first time this attr's value has been set.
                old_value = self.getattribute_animated_full(key)
                if old_value is None:
                    old_value = torch.zeros_like(value[:1, :1])
                else:
                    old_value = unsqueeze_dims(old_value, value)
                if old_value.shape[1] < n2:
                    old_value = old_value.expand(-1, n2, -1)
                old_value = old_value[-1].contiguous()  # [n2, W]
                rows = d.alloc_rows_like(key, old_value)
                if type(value) is not torch.Tensor:
                    d.attr_cls[key] = type(value)
                d.record_modification(
                    key, rows.indices, old_value.clone(),
                    self.animation_manager.context,
                )
                target = (
                    rows.indices
                    if self.data_sub_inds is None
                    else rows.indices[self.data_sub_inds]
                )
                rows.buffer.write_rows(target, value[-1])
            else:
                # We are at render time and this attr has never been modified:
                # snapshot the current value across the window (the incoming
                # value is dropped, matching the previous behavior).
                current = self.__getattribute__(key)[-1:]
                rows = d.alloc_rows_like(key, current[-1])
                d.window.ensure_dense(rows.buffer, rows.indices)
            return
        # Batch expansion: pad with the last element (the new rows inherit the
        # last row's history), exactly like the old pad-with-last dict growth.
        if rows.size < n2:
            if rows.size > 1 and value.shape[1] < n2:
                value = torch.cat(
                    (value, value[:, -1:].expand(-1, n2 - value.shape[1], -1)), 1
                )
            old_size = rows.size
            src_row = int(rows.indices[-1])
            d.grow_rows(key, n2)
            if at_render and d.window is not None:
                # The new columns must start from the source column's
                # materialized per-frame trajectory (the old code broadcast
                # the private dense tensor), not from the live buffer value.
                d.window.clone_row_state(
                    rows.buffer, src_row, rows.indices[old_size:]
                )
        target = (
            rows.indices
            if self.data_sub_inds is None
            else rows.indices[self.data_sub_inds]
        )
        if not at_render:
            if self.data.spawn_time() >= 0:
                d.record_modification(
                    key, rows.indices, rows.read().clone(),
                    self.animation_manager.context,
                )
            rows.buffer.write_rows(target, value[-1])
        else:
            if not d.window.any_dense(rows.buffer, rows.indices):
                # First render-time write to a never-modified attr: snapshot
                # the current value and drop the write (previous behavior).
                d.window.ensure_dense(rows.buffer, rows.indices)
                return
            time_sel = d.time_inds_active
            d.window.write(rows.buffer, target, value, time_sel=time_sel)

    def getattribute_animated_full(self, key):
        if key not in self.data.data_dict:
            if key not in self.data.data_dict_active:
                return super().__getattribute__(key)
            return self.data.data_dict_active[key]
        return self.data.data_dict[key]

    def getattribute_animated(self, key):
        if self.animation_manager.context.trace_mode:
            self.animation_manager.context.traced_mobs.add(self)
        d = self.data
        rows = d.rows.get(key)
        at_render = d.time_inds_materialized is not None
        if rows is not None:
            if not at_render:
                value = d.read_current(key, rows)  # [1, B, W] buffer view
                if value.shape[1] == 1 or self.data_sub_inds is None:
                    return value
                return value[:, self.data_sub_inds]
            # Render time: gather the active frames of this mob's rows from
            # the materialization window in one indexing op.  Rows without
            # any recorded modification are constant and served as a
            # broadcast view.
            time_sel = d.time_inds_active
            value = d.window.read(rows.buffer, rows.indices, time_sel=time_sel)
            if value.shape[0] == 1 and time_sel is not None:
                n_t = (
                    time_sel.numel()
                    if isinstance(time_sel, torch.Tensor)
                    else len(d.time_inds_materialized)
                )
                # Materialize (the old per-mob dense state was always a
                # private copy; callers mutate reads in place).
                value = value.expand(n_t, -1, -1).clone()
            cls = d.attr_cls.get(key)
            if cls is not None:
                value = value.as_subclass(cls)
            if value.shape[1] == 1 or self.data_sub_inds is None:
                return value
            return value[:, self.data_sub_inds]
        if key not in d.side:
            raise AttributeError
        value = d.side[key]
        if not isinstance(value, torch.Tensor):
            return value
        while value.dim() < 3:
            value = value.unsqueeze(0)
        if at_render and value.shape[0] == 1:
            value = value.expand(d.time_inds_active.amax() + 1, -1, -1)
        data_inds = (
            self.data_sub_inds if self.data_sub_inds is not None else slice(None)
        )
        time_inds = d.time_inds_active if at_render else slice(None)
        if value.shape[1] == 1 and self.data_sub_inds is not None:
            return value[time_inds]
        return value[time_inds][:, data_inds]

    def wait(self, *args, **kwargs):
        """An animated function that does nothing for one second!"""
        self.animation_manager.context.wait(*args, **kwargs)

    def get_default_color(self):
        return BLACK

    def on_init(self):
        return self

    def on_create(self):
        return self

    def on_destroy(self):
        return self

    def identity(self):
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        if "___copy_add_to_scene___" not in memo:
            memo["___copy_add_to_scene___"] = self.copy_add_to_scene
        if "___copy_spawn___" not in memo:
            memo["___copy_spawn___"] = self.copy_spawn
        if "___copy_animate_creation___" not in memo:
            memo["___copy_animate_creation___"] = self.copy_animate_creation
        if "___copy_recursive___" not in memo:
            memo["___copy_recursive___"] = self.copy_recursive
        if "___clone_data___" not in memo:
            memo["___clone_data___"] = self.clone_data
        if "___reset_history___" not in memo:
            memo["___reset_history___"] = self.reset_history
        add_to_scene = memo["___copy_add_to_scene___"]
        spawn = memo["___copy_spawn___"]
        animate_creation = memo["___copy_animate_creation___"]
        copy_recursive = memo["___copy_recursive___"]
        clone_data = memo["___clone_data___"]
        reset_history = memo["___reset_history___"]
        cls = self.__class__
        clone = cls.__new__(cls)
        clone.parents = []
        clone.anchor_priority = 0
        memo[id(self)] = clone
        object.__setattr__(clone, "scene", self.scene)
        object.__setattr__(clone, "_animation_manager", self.animation_manager)
        object.__setattr__(clone, "animatable_attrs", self.animatable_attrs)

        if clone_data:
            oa = self.data.animatable
            oh = self.data.history
            self.data.animatable = None
            self.data.history = None
            ti = copy.deepcopy(self.data)
            ti.animatable = clone
            ti.history = ModificationHistory() if reset_history else oh
            ti.spawn_time = lambda: -1
            ti.despawn_time = lambda: -1
            self.data.animatable = oa
            self.data.history = oh
            if not reset_history:
                # The clone shares its source's (function) history; replay the
                # source rows' recorded value modifications onto the clone's
                # rows so its attribute history matches too.
                ti.copy_history_from(self.data)
        else:
            ti = self.data

        object.__setattr__(clone, "data", ti)
        if add_to_scene:
            self.scene.add_actor(clone)
            self.animation_manager.context.add_mob(clone)
        clone.id = self.scene.get_new_id()
        children = (
            list(object.__getattribute__(self, "children"))
            if (hasattr(self, "children") and copy_recursive)
            else []
        )
        children_clones = (
            [copy.deepcopy(c, memo) for c in children] if copy_recursive else []
        )
        component_clones = [
            copy.deepcopy(c, memo) for c in object.__getattribute__(self, "components")
        ]

        child_to_id = {c: i for i, c in enumerate(children)}
        id_to_child = {i: c for i, c in enumerate(children_clones)}

        for k, v in self.__dict__.items():
            if k in [
                "video",
                "id",
                "created",
                "destroyed",
                "spawn_time",
                "despawn_time",
                "animation_manager",
                "_animation_manager",
                "time_inds",
                "history",
                "_hier_plan_cache",
            ]:
                continue
            if k == "data":  # and not clone_data:
                continue
            if k in ["parents"]:
                object.__setattr__(clone, k, [])
                continue
            if isinstance(v, Animatable) and v in children:
                object.__setattr__(clone, k, id_to_child[child_to_id[v]])
                continue
            if k in ["children", "components"]:
                v = []
            if k in ["anchors"]:
                v = defaultdict(list)
            object.__setattr__(clone, k, copy.deepcopy(v, memo))

        clone.generate_animatable_attr_set_get_methods()
        if self.data.spawn_time() >= 0 and spawn:
            clone.spawn(animate_creation)
        if copy_recursive:
            clone.add_children(*children_clones)
        clone.components = component_clones
        return clone

    def clone(
        self,
        add_to_scene=True,
        spawn=True,
        animate_creation=False,
        recursive=True,
        clone_data=True,
        reset_history=True,
    ):
        self.copy_add_to_scene = add_to_scene
        self.copy_spawn = spawn
        self.copy_animate_creation = animate_creation
        self.copy_recursive = recursive
        self.clone_data = clone_data
        self.reset_history = reset_history
        c = copy.deepcopy(self)

        if clone_data:
            c.batch_size = 1
            for d in c.get_descendants():
                for attr in ["location", "opacity", "basis", "color"]:
                    dloc = d.__getattribute__(attr)
                    d.data.data_dict_active[f"{attr}"] = dloc
                    if dloc is None:
                        continue
                    d.batch_size = max(c.batch_size, dloc.shape[-2])
                d.data_sub_inds = None
        return c

    def spawn(self, animate: bool = True):
        """Spawns the mob, introducing it into the video. Prior to spawning, a Mob will not appear on
        screen and any changes made to its animatable attributes will not be animated. After spawning,
        changes made to the Mob are animated by default.

        Parameters
        ----------
        animate
            Whether a spawn-in animation should be played. By default, the spawn-in animation is
            a simple fade-in. Defaults to True.

        """
        if (self.data.spawn_time() >= 0) or self.animation_manager.context.spawn_at_end:
            return self
        self._create_recursive(animate)
        self.animation_manager.context.on_create(self)
        return self

    def _create_recursive(self, animate=True):
        with Sync():
            if self.data.spawn_time() < 0:
                self.data.spawn_time = self.animation_manager.context.get_current_time()
                if animate:
                    self.on_create()
            for c in self.children:
                c._create_recursive(animate)
        return self

    def despawn(self, animate=True):
        if self.data.despawn_time() >= 0:  # or (self.data.spawn_time() < 0):
            return self
        self._destroy_recursive(animate)
        self.animation_manager.context.on_destroy(self)

        return self

    def _destroy_recursive(self, animate=True):
        with Sync():
            if self.data.despawn_time() < 0:
                if animate:
                    self.on_destroy()
                self.data.despawn_time = self.animation_manager.context.get_end_time()
                # self.remove_all_passive_animations()
            for c in self.children:
                c._destroy_recursive(animate)
        return self

    def init(self):
        self.on_init()
        self.animation_manager.context.on_init(self)
        return self

    def delete(self):
        return self.despawn()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Animatable)) for t in types
        ):
            args = [a.location if hasattr(a, "location") else a for a in args]
            return func(*args, **kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def get_memory_used_per_timestep(self):
        return 0

    def set_state_to_time_t(self, time_inds):
        if (self.data.time_inds_materialized is not None
            and len(self.data.time_inds_materialized) == len(time_inds)
            and self.data.time_inds_materialized[0] == time_inds[0]
            and self.data.time_inds_materialized[-1] == time_inds[-1]):
            self.data.time_inds_active = torch.arange(len(time_inds), device=time_inds.device)
        else:
            self.data.time_inds_active = (
                (self.data.time_inds_materialized.view(-1, 1) == time_inds.view(1, -1))
                .sum(1)
                .nonzero()
                .view(-1)
            )
        return self

    def set_state_to_time_all(self):
        return self.set_state_to_time_t(self.time_inds.time_inds_materialized)

    def reset_state(self, make_new_state=False):
        if make_new_state:
            self.data = AnimatableData(self)
            self.data.spawn_time = self.animation_manager.context.get_current_time()
        # Remember the window this data was bound to: if it is re-bound to the
        # very same window later (camera/screen/lights are re-materialized per
        # batch), its rows must be restored to their pre-function state first.
        if self.data.window is not None:
            self.data.stale_window = self.data.window
        self.data.window = None
        self.data.time_inds_materialized = None
        self.data.time_inds_active = None
        self.data.data_dict = self.data.data_dict_active
        self.data.set_pre_function_application = False
        self.already_set_state = False

    def set_state_pre_function_applications(self, spawn_ind, despawn_ind):
        """Binds this mob to the batched materialization window covering
        frames [spawn_ind, despawn_ind).  The window holds the pre-function
        state of *every* modified row (one batched bisect over the global
        modification log per attribute), so no per-mob history walk happens
        here; this method only applies the mob's spawn/despawn opacity masking
        and prepares its function history for re-execution."""
        fps = self.scene.frames_per_second
        window = GlobalAnimationState.instance().ensure_window(
            spawn_ind, despawn_ind, fps
        )
        d = self.data
        if getattr(d, "stale_window", None) is window:
            # Re-binding to a window whose dense rows this mob's functions
            # already wrote: restore its rows to pristine pre-function state.
            for rows in d.rows.values():
                window.rematerialize_rows(rows.buffer, rows.indices)
        d.stale_window = None
        d.window = window
        self.spawn_ind = int(d.spawn_time() * fps)
        if d.despawn_time() < 0:
            self.despawn_ind = despawn_ind+1
        else:
            # Hide the mob from its own despawn frame onwards. We must use the
            # actor's despawn_time here (not the render batch's end) so that mobs
            # despawned without an animation (despawn(animate=False), as used by
            # become()/detach_history()) actually disappear instead of lingering
            # at full opacity until the end of the current render batch.
            self.despawn_ind = max(int(d.despawn_time() * fps), self.spawn_ind + 1)

        self.t = window.t
        self.func_history = d.history.get_func_history(self)

        # Zero opacity outside the [spawn, despawn) frame range. Rows are only
        # densified when the mask actually cuts into this window.
        opacity_rows = d.rows.get("opacity")
        if opacity_rows is not None:
            T = window.T
            z1 = self.spawn_ind - spawn_ind
            z2 = self.despawn_ind - spawn_ind
            if z1 > 0:
                window.zero_time_range(
                    opacity_rows.buffer, opacity_rows.indices, 0, min(z1, T)
                )
            if z2 < T:
                window.zero_time_range(
                    opacity_rows.buffer, opacity_rows.indices, max(z2, 0), T
                )
        d.data_dict = d.data_dict_materialized
        d.time_inds_active = torch.arange(window.T)
        d.time_inds_materialized = window.time_inds
        d.set_pre_function_application = True

    #@compiled
    def set_state_full(self, s, e):
        """Sets all animatable attribute values to their final values after animated_functions have been applied."""
        if self.already_set_state:
            return False
        self.raw_s = s
        self.raw_e = e
        #_sync_devices()
        if not self.data.set_pre_function_application:
            self.set_state_pre_function_applications(s, e)
        if not self.func_history:
            self.already_set_state = True
            return True
        t = self.t
        animating_inds = [torch.zeros((1,), dtype=torch.long)]
        for (
            func,
            animated_args,
            kwargs,
            start_times,
            end_times,
            extended_end_times,
            rate_funcs,
        ) in self.func_history:
            (func, caller) = func
            found = ((start_times <= t) & (t < extended_end_times)).type(t.dtype)
            if found.nonzero().numel() == 0:
                continue

            found_inds = found.sum(1).nonzero().squeeze(-1)
            animating_inds.append(found_inds)

            fa = (
                found[found_inds]
                + (torch.arange(found.shape[-1]) / (2 * found.shape[-1]))
            ).argmax(-1, keepdim=True)
            s, e, ee = (
                broadcast_gather(_.unsqueeze(0), 1, fa, keepdim=True)
                for _ in (start_times, end_times, extended_end_times)
            )
            elapsed_time = t[found_inds] - s
            a = elapsed_time / (e - s)
            a = a.clamp_(max=1)

            a = a.unsqueeze(-2)
            ar = torch.stack(
                broadcast_all(
                    [rf(rfc(a)) if rfc is not None else rf(a) for rf, rfc in rate_funcs]
                ),
                -1,
            )
            z = broadcast_gather(
                ar, -1, unsqueeze_dims(fa, ar, -1), keepdim=False
            ).clamp_(min=0, max=1)
            if caller.parent_batch_sizes is not None and z.shape[1] == len(
                caller.parent_batch_sizes
            ):
                z = torch.repeat_interleave(z, caller.parent_batch_sizes, 1)

            def select_kwargs(kwargs):
                return {
                    key: broadcast_gather(
                        value, 0, unsqueeze_dims(fa, value, 1), keepdim=True
                    )
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in kwargs.items()
                }

            animated_args = select_kwargs(animated_args)
            caller.data.time_inds_active = torch.arange(
                len(caller.data.time_inds_materialized)
            )[found_inds]
            kwargs2 = {
                key: (animated_args[key] * (1 - z) + z * value)
                if key in animated_args
                else value
                for key, value in select_kwargs(kwargs).items()
            }

            if TIME_PARAMETER_NAME in kwargs2:
                kwargs2[TIME_PARAMETER_NAME] = elapsed_time.view(-1, 1, 1)

            func(caller, **kwargs2)

        self.already_set_state = True
        #_sync_devices()
        return True

    def update_gather_scatter_inds(self, n):
        all_inds = torch.arange(n)
        idle_gather_inds = []
        non_idle_inds = self.non_idle_inds
        for i, noni_ind in enumerate(non_idle_inds):
            next_i = non_idle_inds[i + 1] if i < len(non_idle_inds) - 1 else n
            idle_gather_inds.append(torch.tensor((i,)).repeat((next_i - noni_ind) - 1))

        if len(idle_gather_inds) == 0:
            return
        idle_gather_inds = torch.cat(idle_gather_inds)

        super().__setattr__("idle_gather_inds", idle_gather_inds)

        m = (all_inds.unsqueeze(-1) == self.idle_inds).sum(-1)
        ni = (all_inds.unsqueeze(-1) > self.non_idle_inds).sum(-1)
        ii = (all_inds.unsqueeze(-1) > self.idle_inds).sum(-1) + len(self.non_idle_inds)
        idle_scatter_inds = ni * (1 - m) + m * ii
        super().__setattr__("idle_scatter_inds", idle_scatter_inds)

    def getattr_at_time_t(self, attr, t):
        self.set_state_to_time_t(t)
        return self.__getattribute__(attr)

    def set_parent_to(self, other_mob):
        self.parents.append(other_mob)
        return self

    def add_children(self, *mobs):
        """Adds a collection of mobs to this mob's list of children, thereby
        propagating attribute changes made to the parent to the children.

        Parameters
        ----------
        *mobs : Iterable[:class:`~.Mob`}
            A (possibly nested) iterable of :class:`~.Mob` s to be added as children.

        Returns
        -------
        :class:`~.Animatable`
            The Animatable instance itself, allowing for method chaining.

        """
        for mob in traverse(mobs):
            self.children.append(mob)
            mob.set_parent_to(self)
            self.anchor_priority = max(self.anchor_priority, 1 + mob.anchor_priority)
        GlobalAnimationState.instance().bump_topology()
        return self

    def remove_child(self, mob):
        self.children.remove(mob)
        GlobalAnimationState.instance().bump_topology()
        return self
