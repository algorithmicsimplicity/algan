"""Animation contexts -- the ``with`` blocks that decide *when* things happen.

Recording an animation says what changes; the surrounding context says how the
changes are laid out in time. There are four:

:class:`Seq`
    One after another. The default outside any context.
:class:`Sync`
    All at once.
:class:`Lag`
    Overlapping -- each change starts when the previous is ``ratio`` of the way
    through. ``Lag(0)`` is :class:`Sync`, ``Lag(1)`` is :class:`Seq`.
:class:`Off`
    Instantly, in a single frame, recording no animation.

Contexts nest, and a nested context counts as a single animation to its parent,
which is what makes complex multi-Mob choreography readable. Unset parameters are
inherited from the enclosing context. ``run_time`` sets a context's total
duration and is applied **retroactively** on ``__exit__``, by rescaling every
child timestamp -- which is why an event recorded outside any entered-and-exited
context evaluates to time zero.

:class:`~algan.animation_timeline.animation_contexts.Audio` and
:class:`~algan.animation_timeline.animation_contexts.Speech` are the same
mechanism with their duration taken from a sound clip instead of a number.

:class:`AnimationManager` is the per-Scene owner of the context stack.

See :doc:`/new_user_tutorials/combining_animations`.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from algan.animation_timeline.timeline import TimelineSpan
from algan.constants import rate_funcs
from algan.scene_manager import SceneManager
from algan.sound.audio_effect import AudioEffect

DEFAULT_RUN_TIME = 1
DEFAULT_RATE_FUNC = rate_funcs.smooth

# AnimationContext parameters -- not method arguments. Manim spells timing per
# call (``mob.shift(RIGHT, run_time=2)``); Algan spells it with a with-block, so
# one of these names arriving at a Mob method is always the same mistake. The
# value is the context class to point the user at.
_CONTEXT_ONLY_PARAMS = {
    "run_time": "Seq",
    "run_time_unit": "Seq",
    "same_run_time": "Sync",
    "lag_ratio": "Lag",
    "rate_func": "Seq",
    "rate_func_compose": "ComposeRateFunc",
    "combine_rate_func": "Seq",
    "priority_level": "Off",
    "record_funcs": "Off",
    "record_attr_modifications": "Off",
    "spawn_at_end": "Seq",
}


def _reject_context_kwargs(kwargs):
    """Raise a useful TypeError if ``kwargs`` carries animation-context timing.

    Mob methods do not take ``run_time`` and friends; the surrounding
    :class:`AnimationContext` does. Without this, such a call dies inside a
    generated closure with a traceback that names neither the method the user
    wrote nor the thing they should have written instead.
    """
    for name in kwargs:
        context = _CONTEXT_ONLY_PARAMS.get(name)
        if context is None:
            continue
        value = kwargs[name] if isinstance(kwargs, dict) else None
        # Lag takes its ratio positionally; Seq/Sync hard-code it, so
        # ``with Seq(lag_ratio=...)`` would itself raise.
        if name == "lag_ratio":
            call = f"Lag({value!r})" if value is not None else "Lag(0.5)"
        elif name == "run_time" and value == 0:
            call = "Off()"
        elif value is not None:
            call = f"{context}({name}={value!r})"
        else:
            call = f"{context}({name}=...)"
        raise TypeError(
            f"'{name}' sets the timing of an animation context, not of a "
            f"single call. Wrap the call instead:\n\n"
            f"    with {call}:\n"
            f"        ...\n"
        )


class AnimationManager:
    """Per-Scene owner of the active animation-context stack.

    Each Scene has one. It holds whichever :class:`~.AnimationContext` is currently
    in effect, which is what decides how the next recorded animation is timed. The
    outermost context is a :class:`~.Seq` with a one-second unit, which is why
    top-level statements in a script play one after another for a second each.

    Parameters
    ----------
    scene
        The Scene that owns this manager. Defaults to ``None``.
    """

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

    def get_execution_count(self) -> int:
        """Get the next recording sequence number, and advance the counter.

        Recorded events carry this so replay can reproduce the order they were
        authored in, which is what makes overlapping edits resolve consistently.

        Returns
        -------
        int
            The count before incrementing.
        """
        count = self.execution_count
        self.execution_count += 1
        return count

    def wait(self, t: float | None = None):
        """Hold still in the current context, leaving a pause.

        Animation
        ---------
        Recorded on the timeline: it consumes video time and nothing else.

        Parameters
        ----------
        t
            How long to wait, in seconds. Defaults to ``None``, meaning one
            animation's duration (1 second by default).

        Returns
        -------
        :class:`~.AnimationManager`
            This manager, so calls can be chained.
        """
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


@dataclass(kw_only=True)
class AnimationContext:
    """Base class for the ``with`` blocks that control animation timing.

    A context decides *when* the animations recorded inside it happen: together,
    one after another, overlapping, or not at all. Use the subclasses
    rather than this class directly: :class:`~.Sync`, :class:`~.Seq`,
    :class:`~.Lag`, :class:`~.Off`, :class:`~.Audio`, :class:`~.Speech`.

    Contexts nest, and a nested context inherits every parameter left as ``None``
    from its parent, overriding only what it sets. On exit, a context with a
    ``run_time`` retroactively rescales all the timestamps recorded inside it, which
    is how you give a whole block a fixed duration without timing its parts.

    Parameters
    ----------
    run_time
        Total duration of this context, in seconds; the animations inside are
        rescaled to fit. Defaults to ``None``, meaning the duration follows from the
        animations themselves.
    run_time_unit
        Duration of each individual animation inside, in seconds. Defaults to
        ``None``, meaning inherit from the parent context (``1.0`` at the top
        level). ``run_time`` overrides this.
    same_run_time
        Whether to stretch every animation inside to the duration of the longest one.
        Defaults to ``None`` (inherited; effectively False).
    lag_ratio
        Fraction of one animation's duration to wait before starting the next: ``0``
        plays them together, ``1`` strictly in sequence, in between overlaps them.
        Defaults to ``None``, meaning inherit from the parent.
    priority_level
        Priority of this context. A context can only be overridden by one of equal or
        higher priority, which is what lets :class:`~.Off` resist an enclosing
        ``run_time``. Defaults to ``None``, meaning inherit (``0`` at the top level).
    rate_func
        Easing function mapping progress in ``[0, 1]`` to adjusted progress, e.g.
        ``rate_funcs.identity``. Replaces the parent's. Defaults to ``None``, meaning
        inherit (``rate_funcs.smooth`` at the top level).
    rate_func_compose
        Easing function to compose *with* the parent's rather than replace it. Defaults
        to ``None``. See :class:`~.ComposeRateFunc`.
    combine_rate_func
        Whether component rate functions are combined into the context's own.
        Defaults to False.
    record_funcs
        Whether animated functions inside are recorded on the Scene timeline.
        Defaults to ``None``, meaning inherit (True at the top level;
        :class:`~.Off` sets it False).
    record_attr_modifications
        Whether attribute changes inside are recorded on the Scene timeline. Defaults
        to ``None``, meaning inherit (True at the top level).
    prev_context
        The context this one was created inside. Defaults to ``None``; it is filled in
        on entry.
    spawn_at_end
        Whether Mobs created inside are held back and spawned together when the block
        ends. Defaults to ``None``, meaning inherit (False at the top level).
    animation_manager
        The :class:`~.AnimationManager` to record against. Defaults to ``None``,
        meaning the active Scene's -- pass one explicitly when authoring a Scene that
        is not currently active.
    """

    run_time: float | None = field(default=None, kw_only=False)
    run_time_unit: float | None = None
    same_run_time: bool | None = None
    lag_ratio: float | None = None
    priority_level: float | None = None
    rate_func: Callable[[], float] | None = None
    rate_func_compose: Callable[[], float] | None = None
    combine_rate_func: bool = False
    record_funcs: bool | None = None
    record_attr_modifications: bool | None = None
    prev_context: AnimationContext | None = None
    spawn_at_end: bool | None = None
    new_animation: bool | None = False
    finished: bool = False
    new_mobs: list | None = None
    child_contexts: list | None = None
    kwargs: Any = None
    animation_manager: Any = None

    def __post_init__(self):
        if self.new_mobs is None:
            self.new_mobs = []
        if self.child_contexts is None:
            self.child_contexts = []
        if self.kwargs is None:
            self.kwargs = {}
        self.exit_callbacks = []
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

        self.rate_func = copy.deepcopy(self.rate_func)
        self.rate_func_compose = copy.deepcopy(self.rate_func_compose)
        new_kwargs = self.kwargs
        self.kwargs = self.prev_context.kwargs | new_kwargs

        t = self.prev_context.timespan.current_time
        self.timespan = TimelineSpan(t, t, t)
        self._manager_override_token = _ANIMATION_MANAGER_OVERRIDE.set(am)
        return self

    @property
    def current_time(self):
        """Where the authoring cursor sits within this context, in seconds.

        The time the next animation recorded in this context will start at.
        """
        return self.timespan.current_time

    @current_time.setter
    def current_time(self, value):
        self.timespan.current_time = value

    @property
    def end_time(self):
        """The furthest time anything recorded in this context reaches, in seconds."""
        return self.timespan.original_end

    @end_time.setter
    def end_time(self, value):
        self.timespan.original_end = value

    def add_exit_callback(self, callback):
        """Register work to run once this block has finished, at its end time.

        The callback runs after everything inside the block has been recorded
        and rescaled, with the authoring cursor placed at this context's end, so
        anything the callback records lands after the whole block. Callbacks run
        in reverse registration order, so nested set-up unwinds the way it was
        applied.

        This exists for changes that have to outlive the statement that asked for
        them: :meth:`~.Mob.wave_color` refines a Mob's sampling so its colour
        wave renders smoothly, and can only drop that resolution again once
        nothing further will be recorded alongside the wave.

        Parameters
        ----------
        callback
            Zero-argument callable.
        """
        self.exit_callbacks.append(callback)

    def add_child_context(self, c):
        """Register a nested context, so this one's rescaling reaches it.

        Called when a context is entered inside this one.

        Parameters
        ----------
        c
            The child :class:`~.AnimationContext`.
        """
        self.child_contexts.append(c)

    def get_timespan(self):
        """Get this context's timespan object.

        Returns
        -------
        :class:`~algan.animation_timeline.timeline.TimelineSpan`
            The span holding this context's start, cursor and end times.
        """
        return self.timespan

    def get_end_time(self):
        """Get the time this context ends, in seconds, after any rescaling.

        Returns
        -------
        float
            End time on the Scene timeline.
        """
        return self.timespan.get_time(self.timespan.original_end)

    def get_current_time(self):
        """Get the current authoring time, in seconds, after any rescaling.

        Returns
        -------
        float
            The time the next recorded animation starts at.
        """
        return self.timespan.get_current_time()

    def get_current_end_time(self):
        """Get the time an animation started now would finish, in seconds.

        Returns
        -------
        float
            The current time plus one animation's duration.
        """
        return self.timespan.current_time + self.run_time_unit

    def add_mob(self, mob):
        """Record that a Mob was created in this context.

        Bubbles up to enclosing contexts too, which is how :class:`~.SlideShow` knows
        what to clear at the end.

        Parameters
        ----------
        mob
            The Mob that was created.

        Returns
        -------
        :class:`~.AnimationContext`
            This context, so calls can be chained.
        """
        self.new_mobs.append(mob)
        if self.prev_context is not None:
            self.prev_context.add_mob(mob)
        return self

    def get_descendants(self, include_self: bool = True):
        """Get this context and every context nested inside it, flattened.

        Parameters
        ----------
        include_self
            Whether this context is the first element. Defaults to True.

        Returns
        -------
        list[:class:`~.AnimationContext`]
            This context (unless excluded) followed by all nested contexts.
        """
        # Context trees can be both broad and deeply nested (a neural-network
        # activation records tens of thousands of small contexts).  Building
        # one recursively nested list per child and then feeding it through
        # the generic ``traverse`` helper made each query visit the tree twice
        # and perform an ``Iterable``/tensor check for every intermediate
        # list.  A stack gives the same depth-first, authoring-order result in
        # one pass without recursion or temporary trees.
        descendants = []
        stack = [self] if include_self else list(reversed(self.child_contexts))
        while stack:
            context = stack.pop()
            descendants.append(context)
            stack.extend(reversed(context.child_contexts))
        return descendants

    def rewind(self, num_frames: float):
        """Move the authoring cursor backwards, so what follows is recorded earlier.

        Animation
        ---------
        Not animated: this moves the cursor, not any Mob. Animation recorded after
        this call overlaps what was recorded before it.

        Parameters
        ----------
        num_frames
            How far back to move the cursor, in the context's own time units.
        """
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
                positive_durations = [
                    duration for duration in durations if duration > 0
                ]
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
            if self.exit_callbacks:
                self._run_exit_callbacks(am.context)
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

    def _run_exit_callbacks(self, parent):
        """Run the callbacks registered with :meth:`add_exit_callback`.

        They run in the parent context (this one has already been popped) with
        its cursor moved to this block's end, so whatever they record follows
        everything recorded inside the block. The cursor is put back afterwards
        so the callbacks do not disturb the parent's own timing.
        """
        callbacks, self.exit_callbacks = self.exit_callbacks, []
        cursor = parent.timespan.current_time
        parent.timespan.current_time = max(cursor, self.timespan.end)
        try:
            for callback in reversed(callbacks):
                callback()
        finally:
            parent.timespan.current_time = cursor

    def increment_times(self):
        """Advance the cursor after recording one animation.

        How much it moves is what distinguishes the contexts: ``lag_ratio`` of one
        animation's duration, so :class:`~.Sync` (0) leaves the cursor put and
        :class:`~.Seq` (1) moves it past the whole animation. Called by the
        ``animated_function`` machinery; you do not call it yourself.
        """
        # self.end_time = max(self.end_time, self.current_time + self.run_time_unit)
        self.timespan.original_end = max(
            self.timespan.original_end, self.timespan.current_time + self.run_time_unit
        )
        self.timespan.current_time = (
            self.timespan.current_time + self.run_time_unit * self.lag_ratio
        )

    def wait(self, t: float | None = None):
        """Hold still, leaving a pause before the next animation.

        Animation
        ---------
        Recorded on the timeline: it consumes video time and nothing else. As with
        any animation in this context, the cursor advances by ``lag_ratio`` of the
        wait, so a wait inside a :class:`~.Sync` extends the block without pushing
        later animations back.

        Parameters
        ----------
        t
            How long to wait, in seconds. Defaults to ``None``, meaning one
            animation's duration (``run_time_unit``, 1 second by default).
        """
        if t is None:
            t = self.run_time_unit
        self.timespan.original_end = max(
            self.timespan.original_end, self.timespan.current_time + t
        )
        self.timespan.current_time = self.timespan.current_time + t * self.lag_ratio

    def on_create_extra(self, animatable):
        """Hook for behaviour to add whenever a Mob spawns in this context.

        Does nothing by default. :class:`~.SlideShow` overrides it to pause after each
        spawn; override it in your own context subclass for similar effects.

        Parameters
        ----------
        animatable
            The Mob being spawned.

        Returns
        -------
        :class:`~.AnimationContext`
            This context.
        """
        return self

    def on_destroy_extra(self, animatable):
        """Hook for behaviour to add whenever a Mob despawns in this context.

        Does nothing by default.

        Parameters
        ----------
        animatable
            The Mob being despawned.

        Returns
        -------
        :class:`~.AnimationContext`
            This context.
        """
        return self

    def on_init_extra(self, animatable):
        """Hook for behaviour to add whenever a Mob is constructed in this context.

        Does nothing by default; :class:`~.OnInit` overrides it to run a function of
        your choosing.

        Parameters
        ----------
        animatable
            The Mob being constructed.

        Returns
        -------
        :class:`~.AnimationContext`
            This context.
        """
        return self

    def on_init(self, animatable):
        """Run construction hooks for this context and every enclosing one.

        Parameters
        ----------
        animatable
            The Mob being constructed.
        """
        self.on_init_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_init(animatable)

    def on_create(self, animatable):
        """Run spawn hooks for this context and every enclosing one.

        Parameters
        ----------
        animatable
            The Mob being spawned.
        """
        self.on_create_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_create(animatable)

    def on_destroy(self, animatable):
        """Run despawn hooks for this context and every enclosing one.

        Parameters
        ----------
        animatable
            The Mob being despawned.
        """
        self.on_destroy_extra(animatable)
        if self.prev_context is not None:
            self.prev_context.on_destroy(animatable)


class NoExtra(AnimationContext):
    """Suppress enclosing contexts' spawn/despawn extras inside this block.

    Some contexts add behaviour to every spawn or despawn -- :class:`~.SlideShow`
    inserts a pause around each one, for instance. Wrapping code in ``NoExtra``
    stops those additions from applying to spawns inside the block, while leaving
    the timing behaviour of the enclosing context intact.

    Examples
    --------
    .. code-block:: python

        with SlideShow():
            title.spawn()  # gets the slideshow pauses
            with NoExtra(priority_level=1):
                helper.spawn()  # does not
    """

    def on_create(self, animatable):
        """Do nothing, dropping enclosing contexts' spawn extras.

        Parameters
        ----------
        animatable
            The Mob being spawned.

        Returns
        -------
        :class:`~.NoExtra`
            This context.
        """
        return self

    def on_destroy(self, animatable):
        """Do nothing, dropping enclosing contexts' despawn extras.

        Parameters
        ----------
        animatable
            The Mob being despawned.

        Returns
        -------
        :class:`~.NoExtra`
            This context.
        """
        return self


class Off(AnimationContext):
    """Apply changes instantly, with no animation and no video time.

    Everything inside the block takes effect at once: the scene jumps straight to
    the new state, and the video is no longer for it. This is how you set a scene
    up (position, colour, materials, etc) without the viewer watching things slide
    into place, and how you make a cut rather than a transition.

    ``Off`` takes priority over enclosing contexts by default, so a ``run_time``
    further out cannot stretch it back into an animation.

    Parameters
    ----------
    priority_level
        Priority of this context; contexts can only be overridden by ones of equal
        or higher priority. Defaults to ``1``, above the ordinary default of ``0``,
        which is what keeps enclosing timing from re-animating this block.
    **kwargs
        Passed to :class:`~.AnimationContext`. ``record_funcs`` defaults to False
        here, so nothing inside is recorded as a timed event.

    Examples
    --------
    .. code-block:: python

        with Off():
            square.move(RIGHT * 3)  # teleports, takes no video time
            square.color = BLUE

    See Also
    --------
    :class:`~.Sync` : Play the block's animations simultaneously.
    :class:`~.Seq` : Play them one after another.
    """

    def __init__(self, priority_level: float = 1, **kwargs):
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
    """Overlap the block's animations, each starting partway into the last.

    Each animation begins after a fraction of the previous one has
    played, giving a cascade or ripple. Animating a list of Mobs inside
    ``Lag(0.1)`` is the usual way to make them arrive in a wave rather than
    together.

    Parameters
    ----------
    lag_ratio
        Fraction of one animation's duration to wait before starting the next.
        ``0`` is fully simultaneous, ``1`` fully sequential, ``0.1`` starts each
        animation when the previous one is a tenth done. Can be larger than 1,
        in which case it introduces a pause (wait) after each animation finishes.
    run_time
        Passed to :class:`~.AnimationContext` .
    **kwargs
        Passed to :class:`~.AnimationContext` .

    Examples
    --------
    .. code-block:: python

        with Lag(0.2):
            for square in squares:
                square.move(UP)

    See Also
    --------
    :class:`~.Sync` : ``lag_ratio=0``.
    :class:`~.Seq` : ``lag_ratio=1``.
    """

    def __init__(self, lag_ratio: float, run_time: float | None = None, **kwargs):
        super().__init__(run_time, lag_ratio=lag_ratio, new_animation=True, **kwargs)


class Sync(Lag):
    """Play the block's animations all at the same time.

    Everything inside starts together, so the block takes as long as the longest
    animation component animation rather than the sum of them.

    Parameters
    ----------
    run_time
        Passed to :class:`~.AnimationContext` .
    **kwargs
        Passed to :class:`~.Lag` with ``lag_ratio=0`` .

    Examples
    --------
    .. code-block:: python

        with Sync():  # both happen over one second
            square.rotate(90, OUT)
            square.color = BLUE
    """

    def __init__(self, run_time: float | None = None, **kwargs):
        super().__init__(lag_ratio=0, run_time=run_time, **kwargs)


class Seq(Lag):
    """Play the block's animations one after another.

    Each animation starts as the previous one finishes, so the block lasts as long
    as all of them put together. This is what statements at the top level of a
    script already do; reach for ``Seq`` when you need to give a run of animations a
    combined duration, or to sequence inside a :class:`~.Sync`.

    Parameters
    ----------
    run_time
        Passed to :class:`~.AnimationContext` .
    **kwargs
        Passed to :class:`~.Lag` with ``lag_ratio=1``.

    Examples
    --------
    .. code-block:: python

        with Seq(run_time=2):  # three moves squeezed into two seconds
            square.move(RIGHT)
            square.move(UP)
            square.move(LEFT)
    """

    def __init__(self, run_time: float | None = None, **kwargs):
        super().__init__(lag_ratio=1, run_time=run_time, **kwargs)


class Audio(AnimationContext):
    """Play a sound, and fit the block's animations to its length.

    The context's duration is taken from the audio, so whatever you animate inside
    is stretched or squeezed to finish with the sound. That inversion is the point:
    you describe what should happen while a clip plays, rather than timing the clip
    against the animation.

    Parameters
    ----------
    file_path_or_clip
        Path to an audio file, or an already-loaded moviepy audio clip.
    wait_at_end
        Extra seconds to hold after the audio finishes, before the block ends.
        Defaults to ``0``.
    **kwargs
        Passed to :class:`~.AnimationContext`. Note ``run_time`` is set from the
        audio and should not be passed.

    Examples
    --------
    .. code-block:: python

        with Audio("whoosh.wav"):
            square.move(RIGHT * 5)  # takes exactly as long as the clip

    See Also
    --------
    :class:`~.Speech` : The same, driven by a line of narration script.
    """

    def __init__(self, file_path_or_clip: str, wait_at_end: float = 0, **kwargs):
        audio_clip = file_path_or_clip
        if isinstance(file_path_or_clip, str):
            from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

            audio_clip = AudioFileClip(file_path_or_clip)
        kwargs["run_time"] = audio_clip.duration + wait_at_end
        super().__init__(**kwargs)
        self.audio_clip = audio_clip

    def __enter__(self):
        context = super().__enter__()
        if self.prev_context.run_time_unit > 0 and (
            self.prev_context.run_time is None or self.prev_context.run_time > 0
        ):
            self.animation_manager.scene.add_effect(
                AudioEffect(self.audio_clip, self.get_current_time())
            )
        return context


class Speech(Audio):
    """Narrate a line, and fit the block's animations to how long it takes to say.

    The script text is looked up in the Scene's audio manager, and the block runs
    for exactly as long as that narration. Write the sentence, animate what should
    happen while it is spoken, and the timing follows -- which is how a whole
    explanatory video stays in sync with its voiceover.

    Parameters
    ----------
    script
        The line of narration to play. It is appended to the Scene's script and used
        to select the matching audio.
    wait_at_end
        Extra seconds to hold after the line finishes. Defaults to ``1``, leaving a
        beat between sentences.
    *args, **kwargs
        Passed to :class:`~.Audio`.

    Examples
    --------
    .. code-block:: python

        with Speech("Now watch what happens when we double it."):
            square.scale(2)
    """

    def __init__(self, script, wait_at_end: float = 1, *args, **kwargs):
        animation_manager = (
            kwargs.get("animation_manager") or _active_animation_manager()
        )
        kwargs["animation_manager"] = animation_manager
        audio_manager = animation_manager.scene.audio_manager
        audio_manager.append_script(script)
        super().__init__(audio_manager.get_speech(script), wait_at_end, *args, **kwargs)


class SlideShow(Seq):
    """Present Mobs one at a time, pausing on each, and clear up at the end.

    A :class:`~.Seq` with presentation manners: every Mob spawned inside gets a
    one-second pause after it appears, so the viewer has time to read it, and when
    the block ends everything spawned inside is despawned together. Good for
    stepping through a list of points.

    Examples
    --------
    .. code-block:: python

        with SlideShow():
            Tex("First point").spawn()
            Tex("Second point").move_to(DOWN).spawn()
        # both fade out here
    """

    def on_create_extra(self, animatable):
        """Pause for a second after a Mob spawns, so it can be read.

        Parameters
        ----------
        animatable
            The Mob that was just spawned.
        """
        animatable.wait(1)

    def on_destroy_extra(self, animatable):
        """Pause for a second after a Mob despawns.

        Parameters
        ----------
        animatable
            The Mob that was just despawned.
        """
        animatable.wait(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Despawn everything spawned inside the block, then pause.

        The despawns run together inside a :class:`~.Sync`, so the slide clears in one
        beat rather than item by item.
        """
        super().__exit__(exc_type, exc_val, exc_tb)
        with Sync(animation_manager=self.animation_manager):
            for mob in self.new_mobs:
                mob.despawn()
        self.wait()


class OnInit(AnimationContext):
    """Run a function over every Mob constructed inside the block.

    A hook for applying the same setup to a batch of Mobs without repeating it --
    a shared material, a starting position, a shader.

    Parameters
    ----------
    func
        Callable taking one Mob, called as each Mob inside the block is
        constructed.
    **kwargs
        Passed to :class:`~.AnimationContext`.

    Examples
    --------
    .. code-block:: python

        with OnInit(lambda mob: mob.set_material(MeshStandardMaterial())):
            shapes = [Sphere() for _ in range(3)]
    """

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def on_init_extra(self, animatable):
        """Apply this context's function to a newly constructed Mob.

        Parameters
        ----------
        animatable
            The Mob that was just constructed.
        """
        self.func(animatable)


class ComposeRateFunc(AnimationContext):
    """Bend the timing of the block's animations through an extra rate function.

    The given function is composed with whatever rate function is already in effect,
    rather than replacing it, so an enclosing ease is kept and this one layers on
    top. Use it to make a run of animations rush, dawdle or bounce without
    restating the base easing.

    Parameters
    ----------
    rfunc
        Rate function mapping progress in ``[0, 1]`` to adjusted progress. Composed
        with the enclosing context's rate function.
    **kwargs
        Passed to :class:`~.AnimationContext`.

    Examples
    --------
    .. code-block:: python

        with ComposeRateFunc(rate_funcs.there_and_back):
            square.move(RIGHT)  # goes out and comes back
    """

    def __init__(self, rfunc, **kwargs):
        kwargs["rate_func_compose"] = rfunc
        super().__init__(**kwargs)
