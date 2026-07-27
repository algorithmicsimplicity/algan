import copy
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional


from algan.scene_manager import SceneManager
from algan.animation_timeline.timeline import TimelineSpan
from algan.sound.audio_effect import AudioEffect
from algan.constants import rate_funcs
from dataclasses import dataclass

from algan.utils.python_utils import traverse

DEFAULT_RUN_TIME = 2
DEFAULT_RATE_FUNC = rate_funcs.smooth


class AnimationManager:
    """Per-scene owner of the active animation-context stack."""

    def __init__(self, scene=None):
        self.scene = scene
        self.context = Seq(
            run_time_unit=1.0,
            priority_level=0,
            rate_func=rate_funcs.smooth,
            record_funcs=True,
            record_attr_modifications=True,
            spawn_at_end=False,
            animation_manager=self,
        )
        self.execution_count = 0

    def get_execution_count(self):
        count = self.execution_count
        self.execution_count += 1
        return count

    def wait(self, t=None):
        self.context.wait(t)
        return self


_ANIMATION_MANAGER_OVERRIDE = ContextVar(
    "algan_animation_manager_override", default=None
)


@contextmanager
def animation_manager_context(animation_manager):
    """Temporarily bind implicit AnimationContexts to one scene manager."""
    token = _ANIMATION_MANAGER_OVERRIDE.set(animation_manager)
    try:
        yield animation_manager
    finally:
        _ANIMATION_MANAGER_OVERRIDE.reset(token)


def animation_manager_bound(function):
    """Run an initialized Animatable method with its Scene manager bound."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        scene = getattr(self, "scene", None)
        manager = getattr(scene, "animation_manager", None)
        if manager is None:
            # Some constructors call animated helpers before Animatable.__init__.
            # Those fast-path calls have no scene state to bind yet.
            return function(self, *args, **kwargs)
        with animation_manager_context(manager):
            return function(self, *args, **kwargs)

    return wrapped


def _active_animation_manager():
    override = _ANIMATION_MANAGER_OVERRIDE.get()
    if override is not None:
        return override
    return SceneManager.instance().current_scene.animation_manager


def active_scene_for_new_mob():
    """Return the SceneManager's current active scene."""
    return SceneManager.instance().current_scene


def animation_manager_for(*owners):
    """Resolve the one animation manager associated with ``owners``.

    Owners may be scenes, mobs, or nested Python collections containing them.
    A context cannot coherently combine mobs from different scenes, so mixed
    managers are rejected instead of silently recording into whichever scene
    happens to be globally active.
    """
    managers = []

    def collect(owner):
        if owner is None:
            return
        manager = getattr(owner, "animation_manager", None)
        if manager is not None:
            if not any(existing is manager for existing in managers):
                managers.append(manager)
            return
        scene = getattr(owner, "scene", None)
        manager = getattr(scene, "animation_manager", None)
        if manager is not None:
            if not any(existing is manager for existing in managers):
                managers.append(manager)
            return
        if isinstance(owner, dict):
            for value in owner.values():
                collect(value)
        elif isinstance(owner, (list, tuple, set, frozenset)):
            for value in owner:
                collect(value)

    for owner in owners:
        collect(owner)
    if len(managers) > 1:
        raise ValueError("One animation context cannot span multiple Scenes")
    return managers[0] if managers else _active_animation_manager()


class RateFuncWrapper:
    def __init__(self, rf):
        self.rf = rf
        self.time_set = False

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
        If `run_time` is not None, then `run_time` overrides `run_time_unit`.
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
        Whether animated functions in this context are recorded on the owning Scene timeline.
    record_attr_modifications
        Whether animatable-attribute changes are recorded on the owning Scene timeline.
    prev_context : :class:`~.AnimationContext`
        The parent context in which this AnimationContext was created.
    spawn_at_end
        If True, all new :class:`~.Mob` s created in this context will be prevented from spawning, until the end of this
        context where they will all be spawned.

    """

    run_time: float | None = None
    run_time_unit: float | None = None
    same_run_time: bool | None = None
    lag_ratio: float | None = None
    priority_level: float | None = None
    rate_func: Callable[[], float] | None = None
    rate_func_compose: Callable[[], float] | None = None
    combine_rate_func: bool = False
    record_funcs: bool | None = None
    record_attr_modifications: bool | None = None
    prev_context: Optional["AnimationContext"] = None
    spawn_at_end: bool | None = None
    new_animation: bool | None = False
    finished: bool = False
    new_mobs: list | None = None
    child_contexts: list | None = None
    kwargs: Any = None
    animation_manager: Any = None

    def __post_init__(self):
        if self.new_mobs is None:
            self.new_mobs = list()
        if self.child_contexts is None:
            self.child_contexts = list()
        if self.kwargs is None:
            self.kwargs = dict()
        self.timespan = TimelineSpan()

    def __enter__(self):
        am = self.animation_manager or _active_animation_manager()
        self.animation_manager = am
        if self.priority_level is None:
            self.priority_level = am.context.priority_level
        if am.context.priority_level > self.priority_level:
            self.ignored = True
            self._manager_override_token = _ANIMATION_MANAGER_OVERRIDE.set(am)
            return am.context

        self.ignored = False
        self.prev_context = am.context
        am.context = self
        self.prev_context.add_child_context(self)

        def inherit_missing_value(attr):
            if self.__getattribute__(attr) is None:
                self.__setattr__(attr, self.prev_context.__getattribute__(attr))

        for attr in [
            "run_time_unit",
            "lag_ratio",
            "priority_level",
            "rate_func",
            "rate_func_compose",
            "record_funcs",
            "record_attr_modifications",
            "spawn_at_end",
        ]:
            inherit_missing_value(attr)

        if self.rate_func is not None and not isinstance(
            self.rate_func, RateFuncWrapper
        ):
            self.rate_func = RateFuncWrapper(self.rate_func)
        self.rate_func = copy.deepcopy(self.rate_func)
        if self.rate_func_compose is not None and not isinstance(
            self.rate_func_compose, RateFuncWrapper
        ):
            self.rate_func_compose = RateFuncWrapper(self.rate_func_compose)
        self.rate_func_compose = copy.deepcopy(self.rate_func_compose)
        new_kwargs = self.kwargs
        self.kwargs = self.prev_context.kwargs | new_kwargs

        t = self.prev_context.timespan.current_time
        self.timespan = TimelineSpan(t, t, t)
        self._manager_override_token = _ANIMATION_MANAGER_OVERRIDE.set(am)
        return self

    def add_child_context(self, c):
        self.child_contexts.append(c)

    def get_timespan(self):
        return self.timespan

    def get_end_time(self):
        return self.timespan.get_time(self.timespan.original_end)

    def get_current_time(self):
        return self.timespan.get_current_time()

    def get_current_end_time(self):
        return self.timespan.current_time + self.run_time_unit

    def add_mob(self, mob):
        self.new_mobs.append(mob)
        if self.prev_context is not None:
            self.prev_context.add_mob(mob)
        return self

    def get_descendants(self, include_self=True):
        return list(
            traverse(
                [
                    *([self] if include_self else []),
                    [c.get_descendants() for c in self.child_contexts],
                ]
            )
        )

    def rewind(self, num_frames):
        self.timespan.current_time = self.timespan.current_time - num_frames

    def __exit__(self, exc_type, exc_value, exc_traceback):
        token = getattr(self, "_manager_override_token", None)
        if self.ignored:
            if token is not None:
                _ANIMATION_MANAGER_OVERRIDE.reset(token)
                self._manager_override_token = None
            return False

        am = self.animation_manager or _active_animation_manager()
        try:
            if exc_type is not None:
                # The failed context must not participate in later rescaling.
                if (
                    self.prev_context is not None
                    and self in self.prev_context.child_contexts
                ):
                    self.prev_context.child_contexts.remove(self)
                return False

            def rescale(x, b=self.timespan.original_start, s=1):
                return (x - b) * s + b

            def rescale_run_time(context, scale):
                for child in context.get_descendants(include_self=True):
                    child.timespan.start = rescale(child.timespan.start, s=scale)
                    child.timespan.end = rescale(child.timespan.end, s=scale)
                return False

            if self.same_run_time:
                durations = [
                    child.timespan.end - child.timespan.start
                    for child in self.child_contexts
                ]
                positive_durations = [duration for duration in durations if duration > 0]
                max_run_time = max(positive_durations, default=0)
                for child, duration in zip(self.child_contexts, durations):
                    # Empty / Off child contexts already have the desired zero
                    # duration and must not cause a division by zero.
                    if duration > 0:
                        rescale_run_time(child, max_run_time / duration)

            if self.run_time is not None:
                my_run_time = max(
                    self.timespan.original_end - self.timespan.original_start,
                    1e-6,
                )
                scale = self.run_time / my_run_time

                for child in self.get_descendants(include_self=False):
                    child.timespan.start = rescale(child.timespan.start, s=scale)
                    child.timespan.end = rescale(child.timespan.end, s=scale)
                self.timespan.start = rescale(self.timespan.original_start, s=scale)
                self.timespan.end = rescale(self.timespan.original_end, s=scale)
                self.timespan.current_time = rescale(
                    self.timespan.current_time, s=scale
                )
            else:
                self.timespan.start = self.timespan.original_start
                self.timespan.end = self.timespan.original_end

            if self.combine_rate_func:
                def wrap(rate_func):
                    if rate_func is None:
                        return rate_func
                    rate_func.set_full_time(
                        lambda context=self: context.timespan.start,
                        lambda context=self: context.timespan.end,
                    )
                    rate_func.time_set = True
                    return rate_func

                for child in self.get_descendants(include_self=False):
                    wrap(child.rate_func)
                    wrap(child.rate_func_compose)

            self.finished = True
            # Subsequent parent updates and spawn-at-end work must execute in
            # the parent context, just as normal authoring after the with block.
            am.context = self.prev_context
            if self.record_funcs:
                am.context.timespan.original_end = max(
                    am.context.timespan.original_end, self.timespan.end
                )
                if self.new_animation:
                    am.context.timespan.current_time = (
                        self.timespan.start
                        + (self.timespan.end - self.timespan.start)
                        * am.context.lag_ratio
                    )
                else:
                    am.context.timespan.current_time = self.timespan.current_time

            if self.spawn_at_end and not am.context.spawn_at_end:
                with Sync(animation_manager=am):
                    for mob in sorted(
                        self.new_mobs, key=lambda item: -item.anchor_priority
                    ):
                        mob.spawn()
            return False
        finally:
            # Context-stack restoration is unconditional.  In particular, an
            # exception raised by user code or by finalization must never leave
            # this context installed globally after the with statement exits.
            if am.context is self:
                am.context = self.prev_context
            if token is not None:
                _ANIMATION_MANAGER_OVERRIDE.reset(token)
                self._manager_override_token = None

    def increment_times(self):
        #self.end_time = max(self.end_time, self.current_time + self.run_time_unit)
        self.timespan.original_end = max(self.timespan.original_end, self.timespan.current_time + self.run_time_unit)
        self.timespan.current_time = self.timespan.current_time + self.run_time_unit * self.lag_ratio

    def wait(self, t=None):
        if t is None:
            t = self.run_time_unit
        self.timespan.original_end = max(self.timespan.original_end, self.timespan.current_time + t)
        self.timespan.current_time = self.timespan.current_time + t * self.lag_ratio

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
    """Disables animations within its context."""

    def __init__(self, priority_level=1, **kwargs):
        if "record_funcs" not in kwargs:
            kwargs["record_funcs"] = False
        super().__init__(
            lag_ratio=1,
            run_time_unit=0,
            run_time=0,
            priority_level=priority_level,
            new_animation=True,
            **kwargs,
        )


class Lag(AnimationContext):
    """Plays component animations sequentially lagged by a factor `lag_ratio`.

    Parameters
    ----------
    lag_ratio
        The portion of run_time to wait before playing the next animation. For example, lag_ratio=0.1
        would wait 10% of the `run_time_unit` for one animation before starting the next.

    """

    def __init__(self, lag_ratio: float, *args, **kwargs):
        super().__init__(*args, lag_ratio=lag_ratio, new_animation=True, **kwargs)


class Sync(Lag):
    """Plays all component animations synchronously."""

    def __init__(self, *args, **kwargs):
        super().__init__(lag_ratio=0, *args, **kwargs)


class Seq(Lag):
    """Plays all component animations sequentially, with the next starting as soon as the current one finishes."""

    def __init__(self, *args, **kwargs):
        super().__init__(lag_ratio=1, *args, **kwargs)


class Audio(AnimationContext):
    """Plays audio sound from a file.
    This context's run_time will automatically be set to the duration of the played audio.

    Parameters
    ----------
    file_path
        Path to the file which contains the audio.

    """

    def __init__(self, file_path_or_clip: str, wait_at_end=0, **kwargs):
        audio_clip = file_path_or_clip
        if isinstance(file_path_or_clip, str):
            from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

            audio_clip = AudioFileClip(file_path_or_clip)
        kwargs["run_time"] = audio_clip.duration + wait_at_end
        super().__init__(**kwargs)
        self.audio_clip = audio_clip

    def __enter__(self):
        context = super().__enter__()
        if self.prev_context.run_time_unit > 0:
            self.animation_manager.scene.add_effect(
                AudioEffect(self.audio_clip, self.get_current_time())
            )
        return context


class Speech(Audio):
    """Plays audio sound assosciated the given script over the course of this context.
    This context's run_time will automatically be set to the duration of the played audio.

    Parameters
    ----------
    script
        The segment of script identifying which portion of the audio source to play during this context.

    """

    def __init__(self, script, wait_at_end=1, *args, **kwargs):
        animation_manager = kwargs.get("animation_manager") or _active_animation_manager()
        kwargs["animation_manager"] = animation_manager
        audio_manager = animation_manager.scene.audio_manager
        audio_manager.append_script(script)
        super().__init__(audio_manager.get_speech(script), wait_at_end, *args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.wait(1)


class SlideShow(Seq):
    def on_create_extra(self, animatable):
        animatable.wait(1)

    def on_destroy_extra(self, animatable):
        animatable.wait(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        with Sync(animation_manager=self.animation_manager):
            for mob in self.new_mobs:
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
        kwargs["rate_func_compose"] = rfunc
        super().__init__(**kwargs)
