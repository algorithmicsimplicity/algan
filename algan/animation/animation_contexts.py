import copy
from typing import Any, Callable, Optional

from algan.constants import rate_funcs
from dataclasses import dataclass, field

from algan.utils.python_utils import traverse

DEFAULT_RUN_TIME = 2
DEFAULT_RATE_FUNC = rate_funcs.smooth


class AnimationManager:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call AnimationManager.instance() instead of AnimationManager().')

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.context = Seq(run_time_unit=1.0, priority_level=0, rate_func=rate_funcs.smooth,
                                        record_funcs=True, record_attr_modifications=True, spawn_at_end=False)
            cls._instance.context.begin_time = 0
            cls._instance.context.current_time = 0
            cls._instance.context.end_time = 0
        return cls._instance

    @classmethod
    def wait(cls, t=None):
        am = AnimationManager.instance()
        am.context.wait(t)


class RateFuncWrapper:
    def __init__(self, rf):
        self.rf = rf
        self.time_set = False

    def set_part_time(self, sf, ef):
        self.s_part = sf
        self.e_part = ef

    def set_full_time(self, sf, ef):
        self.s_full = sf
        self.e_full = ef

    def __call__(self, t):
        return self.rf(t)


@dataclass(kw_only=True)
class AnimationContext:
    """An AnimationContext is a context manager that defines how animated_functions that occur within its context
    should be combined and scaled when creating the final animation timeline. Roughly speaking, an AnimationContext
    combines all of the :func:`~.animated_function` s that take place within its context into a single :func:`~.animated_function` that
    plays all of the component animations one after the other.

    AnimationContexts are designed to be nested, in order to make defining complex animation behaviours easy.
    When creating a new AnimationContext inside of an existing one, any parameters which are None will inherit
    their value from the parent context. Parameters with a non-None value will override the parent's value for the
    new context.

    Parameters
    ----------
    run_time
        If not None, then this context will have its duration be rescaled to run_time, otherwise run_time
        is defined by the component animations that take place in this context.
    run_time_unit
        The duration that each component animation within this context will run for.
        Uf `run_time` is not None, then `run_time` overrides `run_time_unit`.
    same_run_time
        If True, rescale all component animations to have the same run_time
        (equal to the longest component run time).
    lag_ratio
        The portion of `run_time_unit` that will be waited for before starting the next component animation.
        When lag_ratio=0, all component animations are played at the same time, when lag_ratio=1, component animations
        are played one after the other immediately after the previous finishes.
    priority_level
        The priority level of this context. AnimationContexts can only be overridden by new AnimationContexts
        of equal or higher priority.
    rate_func
        The rate function defines the rate at which time progresses for each component animation. Defaults to smooth.
        Setting this parameter overrides the parent context's `rate_func` to be equal to this value.
    rate_func_compose
        Setting this parameter sets the rate_func to be the composition of the parent context's `rate_func` with this
        `rate_func_compose`.
    record_funcs
        Whether :func:`~.animated_function` s within this context should be recorded in :class:`~.ModificationHistory` s.
    record_attr_modifications
        Whether changes to `animatable_attributes` within this context should be recorded in :class:`~.ModificationHistory` s.
    prev_context : :class:`~.AnimationContext`
        The parent context in which this AnimationContext was created.
    spawn_at_end
        If True, all new :class:`~.Mob` s created in this context will be prevented from spawning, until the end of this
        context where they will all be spawned.

    """

    run_time: float|None = None
    run_time_unit: float|None = None
    same_run_time: bool|None = None
    lag_ratio: float|None = None
    priority_level: float|None = None
    rate_func: Callable[[],float]|None = None
    rate_func_compose: Callable[[],float]|None = None
    combine_rate_func: bool = False
    record_funcs: bool|None = None
    record_attr_modifications: bool|None = None
    prev_context: Optional["AnimationContext"] = None
    spawn_at_end: bool|None = None
    new_animation: bool|None = False
    finished: bool = False
    trace_mode: bool|None = None
    traced_mobs: set = field(default_factory=set)
    new_mobs: list = field(default_factory=list)
    child_contexts: list = field(default_factory=list)
    kwargs: Any = field(default_factory=dict)

    def __enter__(self):
        am = AnimationManager.instance()
        if self.priority_level is None:
            self.priority_level = am.context.priority_level
        if am.context.priority_level > self.priority_level:
            self.ignored = True
            return am.context

        self.ignored = False
        self.prev_context = am.context
        am.context = self
        self.prev_context.add_child_context(self)

        def inherit_missing_value(attr):
            if self.__getattribute__(attr) is None:
                self.__setattr__(attr, self.prev_context.__getattribute__(attr))

        [inherit_missing_value(attr) for attr in ['run_time_unit', 'lag_ratio',
                                                  'priority_level', 'rate_func', 'rate_func_compose', 'record_funcs', 'record_attr_modifications',
                                                  'spawn_at_end', 'trace_mode']]

        if self.rate_func is not None and not isinstance(self.rate_func, RateFuncWrapper):
            self.rate_func = RateFuncWrapper(self.rate_func)
        self.rate_func = copy.deepcopy(self.rate_func)
        if self.rate_func_compose is not None and not isinstance(self.rate_func_compose, RateFuncWrapper):
            self.rate_func_compose = RateFuncWrapper(self.rate_func_compose)
        self.rate_func_compose = copy.deepcopy(self.rate_func_compose)
        new_kwargs = self.kwargs
        self.kwargs = self.prev_context.kwargs | new_kwargs

        self.begin_time = self.prev_context.current_time
        self.current_time = self.begin_time
        self.end_time = self.begin_time
        self.rescaler = lambda x: x
        return self

    def add_child_context(self, c):
        self.child_contexts.append(c)

    def get_rescaling_time(self, t):
        return lambda t=t: self.get_rescaled(t)

    def get_current_time(self):
        return self.get_rescaling_time(self.current_time)

    def get_end_time(self):
        return self.get_rescaling_time(self.end_time)

    def get_current_end_time(self):
        return self.get_rescaling_time(self.current_time + self.run_time_unit)

    def get_rescaled(self, x):
        if not self.finished:
            return x
        my_run_time = max(self.end_time - self.begin_time, 1e-6)
        parent_run_time = max(self.end_time_r - self.begin_time_r, 1e-6)
        return self.begin_time_r + (x-self.begin_time) * (parent_run_time / my_run_time)

    def add_mob(self, mob):
        self.new_mobs.append(mob)
        if self.prev_context is not None:
            self.prev_context.add_mob(mob)
        return self

    def get_descendants(self, include_self=True):
        return list(traverse([*([self] if include_self else []), [c.get_descendants() for c in self.child_contexts]]))

    def rewind(self, num_frames):
        self.current_time = self.current_time - num_frames

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for c in self.child_contexts:
            self.traced_mobs = self.traced_mobs.union(c.traced_mobs)
        if exc_type is not None:
            return False
            #raise exc_value
        if self.ignored:
            return False

        def rescale(x, b=self.begin_time, s=1):
            return (x - b) * s + b

        def rescale_run_time(context, s):
            for c in context.get_descendants(include_self=True):
                c.begin_time_r = rescale(c.begin_time_r, s=s)
                c.end_time_r = rescale(c.end_time_r, s=s)
            return False

        if self.same_run_time:
            max_run_time = 0
            for c in self.child_contexts:
                max_run_time = max(max_run_time, c.end_time_r - c.begin_time_r)
            for c in self.child_contexts:
                rescale_run_time(c, max_run_time / (c.end_time_r - c.begin_time_r))

        if self.run_time is not None:
            my_run_time = max(self.end_time - self.begin_time, 1e-6)
            s = self.run_time / my_run_time

            for c in self.get_descendants(include_self=False):
                c.begin_time_r = rescale(c.begin_time_r, s=s)
                c.end_time_r = rescale(c.end_time_r, s=s)
            self.begin_time_r = rescale(self.begin_time, s=s)
            self.end_time_r = rescale(self.end_time, s=s)
            self.current_time = rescale(self.current_time, s=s)
        else:
            self.begin_time_r = self.begin_time
            self.end_time_r = self.end_time
        if self.combine_rate_func:
            def wrap(rate_func):
                if rate_func is None:
                    return rate_func
                rate_func.set_full_time(lambda s=self: s.begin_time_r, lambda s=self: s.end_time_r)
                rate_func.time_set = True
                return rate_func
            for c in self.get_descendants(include_self=False):
                wrap(c.rate_func)
                wrap(c.rate_func_compose)
        self.finished = True
        am = AnimationManager.instance()
        am.context = self.prev_context
        am.context.end_time = max(am.context.end_time, self.end_time_r)
        if self.new_animation:
            am.context.current_time = self.begin_time_r + (self.end_time_r - self.begin_time_r) * am.context.lag_ratio
        else:
            am.context.current_time = self.current_time

        if not (self.spawn_at_end and not am.context.spawn_at_end):
            return False
        with Sync():
            for mob in self.new_mobs:
                mob.spawn()
        return False

    def increment_times(self):
        self.end_time = max(self.end_time, self.current_time + self.run_time_unit)
        self.current_time = self.current_time + self.run_time_unit * self.lag_ratio

    def wait(self, t=None):
        if t is None:
            t = self.run_time_unit * 0.5
        self.end_time = max(self.end_time, self.current_time + t)
        self.current_time = self.current_time + t * self.lag_ratio

    def on_create_extra(self, animatable):
        return self

    def on_destroy_extra(self, animatable):
        return self

    def on_init_extra(self, animatable):
        return self

    def on_init(self, animatable):
        self.on_init_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_init(animatable)

    def on_create(self, animatable):
        self.on_create_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_create(animatable)

    def on_destroy(self, animatable):
        self.on_destroy_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_destroy(animatable)


class NoExtra(AnimationContext):
    def on_create(self, animatable):
        return self

    def on_destroy(self, animatable):
        return self


class Off(AnimationContext):
    """Disables animations within its context.
    """
    def __init__(self, priority_level=1, **kwargs):
        if 'record_funcs' not in kwargs:
            kwargs['record_funcs'] = False
        super().__init__(lag_ratio=1, run_time_unit=0, run_time=0, priority_level=priority_level, new_animation=True, **kwargs)


class Lag(AnimationContext):
    """Plays component animations sequentially lagged by a factor `lag_ratio`.

    Parameters
    ----------
    lag_ratio
        The portion of run_time to wait before playing the next animation. For example, lag_ratio=0.1
        would wait 10% of the `run_time_unit` for one animation before starting the next.

    """
    def __init__(self, lag_ratio:float, **kwargs):
        super().__init__(lag_ratio=lag_ratio, new_animation=True, **kwargs)


class Sync(Lag):
    """Plays all component animations synchronously."""
    def __init__(self, **kwargs):
        super().__init__(lag_ratio=0, **kwargs)


class Seq(Lag):
    """Plays all component animations sequentially, with the next starting as soon as the current one finishes."""
    def __init__(self, **kwargs):
        super().__init__(lag_ratio=1, **kwargs)


class SlideShow(Seq):
    def on_create_extra(self, animatable):
        animatable.wait(1)

    def on_destroy_extra(self, animatable):
        animatable.wait(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        with Sync():
            for mob in self.created_mobs:
                mob.despawn()
        self.wait()


class OnInit(AnimationContext):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def on_init_extra(self, animatable):
        self.func(animatable)


class ComposeRateFunc(AnimationContext):
    def __init__(self, rfunc, **kwargs):
        kwargs['rate_func_compose'] = rfunc
        super().__init__(**kwargs)
