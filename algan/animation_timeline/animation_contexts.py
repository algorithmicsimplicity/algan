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
inherited from the enclosing context. ``runtime`` sets a context's total
runtime and is applied **retroactively** on ``__exit__``, by rescaling every
child timestamp -- which is why an event recorded outside any entered-and-exited
context evaluates to time zero.

:class:`~algan.animation_timeline.animation_contexts.Audio` and
:class:`~algan.animation_timeline.animation_contexts.Speech` are the same
mechanism with their runtime taken from a sound clip instead of a number.

:class:`AnimationManager` is the per-Scene owner of the context stack.

See :doc:`/new_user_tutorials/combining_animations`.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from algan.animation_timeline.timeline import TimelineSpan
from algan.constants import easings
from algan.errors import AlganConfigurationError, ContextReuseError
from algan.scene_manager import SceneManager
from algan.sound.audio_effect import AudioEffect

DEFAULT_RUNTIME = 1
DEFAULT_EASING = easings.smooth


class _TimingParameterError(AlganConfigurationError, TypeError):
    """Raised when a timing parameter reaches a call that cannot take it.

    Both bases carry weight. It is an :class:`~algan.errors.AlganConfigurationError`
    like every other bad-argument error Algan raises, and a ``TypeError``
    because an unexpected keyword argument has always been one -- so code that
    catches either still catches this.
    """

    code = "ALGAN_TIMING_PARAMETER"


# AnimationContext parameters -- not method arguments. Manim spells timing per
# call (``mob.shift(RIGHT, run_time=2)``); Algan spells it with a with-block, so
# one of these names arriving at a Mob method is always the same mistake. The
# value is the context class to point the user at.
_CONTEXT_ONLY_PARAMS = {
    "runtime": "Seq",
    "runtime_per_part": "Seq",
    "equalize_runtimes": "Sync",
    "ratio": "Lag",
    "easing": "Seq",
    "composed_easing": "ComposedEasing",
    "combine_easing": "Seq",
    "priority_level": "Off",
    "record_funcs": "Off",
    "record_attr_modifications": "Off",
    "spawn_at_end": "Seq",
}

#: Manim's spelling of a context parameter -> Algan's. A reader arriving from
#: Manim writes ``run_time`` and ``lag_ratio``, so the same mistake has to be
#: caught under those names and answered with the name Algan actually takes.
_MANIM_CONTEXT_PARAM_SPELLINGS = {
    "run_time": "runtime",
    "lag_ratio": "ratio",
}

#: Names Algan used to spell differently, or that a reader guesses from Manim
#: or from a video library, mapped to the one Algan name for the same thing.
#: These are rejected *everywhere* -- on a context, on ``Scene.wait`` and on an
#: animated Mob method -- because none of them is a parameter of anything, so
#: leaving one through means the value is silently ignored or the call dies
#: several frames down in a setter the user never typed.
_LEGACY_PARAM_SPELLINGS = {
    "duration": "runtime",
    "run_time": "runtime",
    "rate_func": "easing",
    "rate_functions": "easings",
}


#: Every name the guards below answer for, as one set. Checked first, with a
#: single ``isdisjoint``, because ``animated_function`` and every
#: ``AnimationContext`` construction run this on the authoring hot path --
#: mob construction performs thousands of both.
_GUARDED_PARAM_NAMES = frozenset(
    (*_CONTEXT_ONLY_PARAMS, *_MANIM_CONTEXT_PARAM_SPELLINGS, *_LEGACY_PARAM_SPELLINGS)
)

_LEGACY_PARAM_NAMES = frozenset(_LEGACY_PARAM_SPELLINGS)


def _show_value(value):
    """A short literal for ``value``, for quoting back in an error message."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return "..."
    return repr(value)


def _reject_legacy_timing_kwargs(kwargs, *, context_name=None):
    """Raise if ``kwargs`` uses a name Algan does not have for a timing thing.

    ``duration``/``run_time``/``rate_func`` are the natural wrong guesses --
    the README teaches ``runtime`` and ``easing`` -- and every one of them
    otherwise surfaces as an internal traceback. ``context_name`` words the
    suggestion as a constructor call when the mistake was made on a context.
    """
    if _LEGACY_PARAM_NAMES.isdisjoint(kwargs):
        return
    for name in kwargs:
        replacement = _LEGACY_PARAM_SPELLINGS.get(name)
        if replacement is None:
            continue
        shown = _show_value(kwargs[name])
        if context_name is None:
            fix = f"    with Seq({replacement}={shown}):\n        ...\n"
        else:
            fix = f"    with {context_name}({replacement}={shown}):\n        ...\n"
        raise _TimingParameterError(
            f"'{name}' is not an Algan parameter; Algan spells it "
            f"'{replacement}'. Write:\n\n{fix}"
        )


def _reject_context_kwargs(kwargs):
    """Raise a useful error if ``kwargs`` carries animation-context timing.

    Mob methods do not take ``runtime`` and friends; the surrounding
    :class:`AnimationContext` does. Without this, such a call dies inside a
    generated closure with a traceback that names neither the method the user
    wrote nor the thing they should have written instead.
    """
    if _GUARDED_PARAM_NAMES.isdisjoint(kwargs):
        return
    _reject_legacy_timing_kwargs(kwargs)
    for name in kwargs:
        # ``run_time``/``lag_ratio`` are Manim's spellings; they are caught here
        # and answered with Algan's name, not echoed back.
        algan_name = _MANIM_CONTEXT_PARAM_SPELLINGS.get(name, name)
        context = _CONTEXT_ONLY_PARAMS.get(algan_name)
        if context is None:
            continue
        value = kwargs[name] if isinstance(kwargs, dict) else None
        # Lag takes its ratio positionally; Seq/Sync hard-code it, so
        # ``with Seq(ratio=...)`` would itself raise.
        if algan_name == "ratio":
            call = f"Lag({value!r})" if value is not None else "Lag(0.5)"
        elif algan_name == "runtime" and value == 0:
            call = "Off()"
        elif value is not None:
            call = f"{context}({algan_name}={value!r})"
        else:
            call = f"{context}({algan_name}=...)"
        raise _TimingParameterError(
            f"'{name}' sets the timing of an animation context, not of a "
            f"single call. Wrap the call instead:\n\n"
            f"    with {call}:\n"
            f"        ...\n"
        )


def _reject_negative_runtime(name, value):
    """Raise if ``value`` is a negative number of seconds.

    Time only moves forward. A negative runtime rewinds the scene clock, which
    then silently shortens or empties the render rather than failing -- and
    ``scene.wait(target - now)`` coming out negative is an easy thing to write.
    """
    if value is None:
        return value
    try:
        negative = value < 0
    except TypeError:
        return value
    if negative:
        raise AlganConfigurationError(
            f"{name}={value!r} is negative, and a runtime is a number of "
            f"seconds, so it must be zero or more. Time in Algan only moves "
            f"forward: to animate something backwards, reverse the animation "
            f"itself rather than its runtime."
        )
    return value


def _guard_context_init(cls):
    """Make ``cls``'s generated ``__init__`` reject legacy timing spellings.

    ``AnimationContext`` is a dataclass, so its ``__init__`` is generated and
    cannot simply be written by hand; wrapping it after the fact is what lets
    ``Sync(duration=1)`` answer with ``runtime`` instead of dying on an
    unexpected keyword argument. Every subclass funnels through ``super()``,
    so guarding the base guards all of them.
    """
    original = cls.__init__

    @wraps(original)
    def __init__(self, *args, **kwargs):
        if kwargs and not _LEGACY_PARAM_NAMES.isdisjoint(kwargs):
            _reject_legacy_timing_kwargs(kwargs, context_name=type(self).__name__)
        original(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


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
            runtime_per_part=1.0,
            priority_level=0,
            easing=easings.smooth,
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
            How long to wait, in seconds. Must be zero or more. Defaults to
            ``None``, meaning one animation's runtime (1 second by default).

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


# Plain, rather than `@dataclass(kw_only=True)` with `runtime` opted back out of
# it. `kw_only` is 3.10, on both the decorator and `field()`, and it is evaluated
# when the class body runs -- so it made `import algan` a TypeError on 3.9, which
# `requires-python` claims to support. Every field here has a default, so nothing
# but the keyword-only constraint is lost, and that constraint only ever applied
# to this base class: `Sync`, `Lag`, `Seq`, `Off` and the rest declare their own
# `__init__` and take `runtime` positionally through it, which is the spelling
# users actually write.
@_guard_context_init
@dataclass
class AnimationContext:
    """Base class for the ``with`` blocks that control animation timing.

    A context decides *when* the animations recorded inside it happen: together,
    one after another, overlapping, or not at all. Use the subclasses
    rather than this class directly: :class:`~.Sync`, :class:`~.Seq`,
    :class:`~.Lag`, :class:`~.Off`, :class:`~.Audio`, :class:`~.Speech`.

    Contexts nest, and a nested context inherits every parameter left as ``None``
    from its parent, overriding only what it sets. On exit, a context with a
    ``runtime`` retroactively rescales all the timestamps recorded inside it, which
    is how you give a whole block a fixed runtime without timing its parts.

    Parameters
    ----------
    runtime
        Total runtime of this context, in seconds; the animations inside are
        rescaled to fit. Must be zero or more. Defaults to ``None``, meaning the
        runtime follows from the animations themselves.
    runtime_per_part
        Runtime of each individual animation inside, in seconds. Must be zero or
        more. Defaults to ``None``, meaning inherit from the parent context
        (``1.0`` at the top level). ``runtime`` overrides this.
    equalize_runtimes
        Whether to stretch every animation inside to the runtime of the longest one.
        Defaults to ``None`` (inherited; effectively False).
    lag_ratio
        Fraction of one animation's runtime to wait before starting the next: ``0``
        plays them together, ``1`` strictly in sequence, in between overlaps them.
        Defaults to ``None``, meaning inherit from the parent.
    priority_level
        Priority of this context. A context can only be overridden by one of equal or
        higher priority, which is what lets :class:`~.Off` resist an enclosing
        ``runtime``. Defaults to ``None``, meaning inherit (``0`` at the top level).
    easing
        Easing function mapping progress in ``[0, 1]`` to adjusted progress, e.g.
        ``easings.identity``. Replaces the parent's. Defaults to ``None``, meaning
        inherit (``easings.smooth`` at the top level).
    composed_easing
        Easing function to compose *with* the parent's rather than replace it. Defaults
        to ``None``. See :class:`~.ComposedEasing`.
    combine_easing
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

    Raises
    ------
    :class:`.AlganConfigurationError`
        If ``runtime`` or ``runtime_per_part`` is negative, or a parameter is
        spelled the way Manim or an older Algan spelled it (``duration``,
        ``run_time``, ``rate_func``).
    :class:`.ContextReuseError`
        If the same context object is entered by a second ``with`` block.
    """

    runtime: float | None = None
    runtime_per_part: float | None = None
    equalize_runtimes: bool | None = None
    lag_ratio: float | None = None
    priority_level: float | None = None
    easing: Callable[[], float] | None = None
    composed_easing: Callable[[], float] | None = None
    combine_easing: bool = False
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
        _reject_negative_runtime("runtime", self.runtime)
        _reject_negative_runtime("runtime_per_part", self.runtime_per_part)
        if self.new_mobs is None:
            self.new_mobs = []
        if self.child_contexts is None:
            self.child_contexts = []
        if self.kwargs is None:
            self.kwargs = {}
        self.exit_callbacks = []
        self.timespan = TimelineSpan()
        # One context object, one ``with`` block.  See _reject_reuse.
        self._entered = False
        self._exited = False

    def _reject_reuse(self):
        """Refuse a second ``with`` on this object.

        A context is single-use: its timespan, its child list and its
        ``ContextVar`` reset token all describe one block. Entering the same
        object twice overwrites the token, so the first entry's override is
        never undone and every later context in the process resolves against
        the wrong :class:`~.AnimationManager` -- which silently turns every
        subsequent ``Sync`` and ``Lag`` into a sequence. Nesting an object in
        itself additionally makes ``prev_context`` self-referential, so the
        stack can never unwind.

        Contexts are cheap, so the fix is always to construct a new one.
        """
        if not (self._entered or self._exited):
            return
        name = type(self).__name__
        detail = (
            "is already being used by an enclosing 'with' block"
            if self._entered
            else "has already been used by a 'with' block that finished"
        )
        raise ContextReuseError(
            f"This {name} {detail}, and an animation context can only be "
            f"entered once. Construct a new one for each block:\n\n"
            f"    with {name}(...):\n"
            f"        ...\n"
            f"    with {name}(...):   # a second object, not the same one\n"
            f"        ...\n\n"
            f"A context's timing describes one block, so reusing the object "
            f"would carry the first block's cursor and children into the second."
        )

    def __enter__(self):
        self._reject_reuse()
        self._entered = True
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
            "runtime_per_part",
            "lag_ratio",
            "priority_level",
            "easing",
            "composed_easing",
            "record_funcs",
            "record_attr_modifications",
            "spawn_at_end",
        ]:
            inherit_missing_value(attr)

        self.easing = copy.deepcopy(self.easing)
        self.composed_easing = copy.deepcopy(self.composed_easing)
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
        them: :meth:`~.Mob.wave_color` refines a Mob's sampling so its color
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
            The current time plus one animation's runtime.
        """
        return self.timespan.current_time + self.runtime_per_part

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
        self._entered = False
        self._exited = True
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

            def rescale_runtime(context, scale):
                for child in context.get_descendants(include_self=True):
                    child.timespan.start = rescale(child.timespan.start, s=scale)
                    child.timespan.end = rescale(child.timespan.end, s=scale)
                return False

            if self.equalize_runtimes:
                runtimes = [
                    child.timespan.end - child.timespan.start
                    for child in self.child_contexts
                ]
                positive_runtimes = [runtime for runtime in runtimes if runtime > 0]
                max_runtime = max(positive_runtimes, default=0)
                for child, runtime in zip(self.child_contexts, runtimes):
                    # Empty / Off child contexts already have the desired zero
                    # runtime and must not cause a division by zero.
                    if runtime > 0:
                        rescale_runtime(child, max_runtime / runtime)

            if self.runtime is not None:
                my_runtime = max(
                    self.timespan.original_end - self.timespan.original_start,
                    1e-6,
                )
                scale = self.runtime / my_runtime

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

            if self.combine_easing:

                def wrap(easing):
                    if easing is None:
                        return easing
                    easing.set_full_time(
                        lambda context=self: context.timespan.start,
                        lambda context=self: context.timespan.end,
                    )
                    easing.time_set = True
                    return easing

                for child in self.get_descendants(include_self=False):
                    wrap(child.easing)
                    wrap(child.composed_easing)

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
        animation's runtime, so :class:`~.Sync` (0) leaves the cursor put and
        :class:`~.Seq` (1) moves it past the whole animation. Called by the
        ``animated_function`` machinery; you do not call it yourself.
        """
        # self.end_time = max(self.end_time, self.current_time + self.runtime_per_part)
        self.timespan.original_end = max(
            self.timespan.original_end,
            self.timespan.current_time + self.runtime_per_part,
        )
        self.timespan.current_time = (
            self.timespan.current_time + self.runtime_per_part * self.lag_ratio
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
            How long to wait, in seconds. Must be zero or more. Defaults to
            ``None``, meaning one animation's runtime (``runtime_per_part``,
            1 second by default).

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``t`` is negative.
        """
        if t is None:
            t = self.runtime_per_part
        _reject_negative_runtime("t", t)
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
    up (position, color, materials, etc) without the viewer watching things slide
    into place, and how you make a cut rather than a transition.

    ``Off`` takes priority over enclosing contexts by default, so a ``runtime``
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
            runtime_per_part=0,
            runtime=0,
            priority_level=priority_level,
            new_animation=True,
            **kwargs,
        )


def _reject_fixed_lag_ratio(name, fixed, kwargs):
    """Explain that ``Sync`` and ``Seq`` are ``Lag`` with the ratio decided.

    Both fix their own ratio when they call ``Lag``, so a caller's keyword
    collided with it and Python reported "got multiple values for keyword
    argument" against ``Lag.__init__`` -- an internal class the caller never
    mentioned. ``lag_ratio`` is caught alongside ``ratio`` because it is what a
    reader arriving from Manim writes.
    """
    for spelling in ("ratio", "lag_ratio"):
        if spelling in kwargs:
            given = kwargs[spelling]
            break
    else:
        return
    raise TypeError(
        f"{name} is Lag with ratio={fixed}, so it takes no {spelling} of "
        f"its own. Use Lag({given!r}) for that overlap, or {name}() for "
        f"ratio={fixed}."
    )


class Lag(AnimationContext):
    """Overlap the block's animations, each starting partway into the last.

    Each animation begins after a fraction of the previous one has
    played, giving a cascade or ripple. Animating a list of Mobs inside
    ``Lag(0.1)`` is the usual way to make them arrive in a wave rather than
    together.

    Parameters
    ----------
    ratio
        Fraction of one animation's runtime to wait before starting the next.
        ``0`` is fully simultaneous, ``1`` fully sequential, ``0.1`` starts each
        animation when the previous one is a tenth done. Can be larger than 1,
        in which case it introduces a pause (wait) after each animation finishes.
        Defaults to ``0.5``.
    runtime
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
    :class:`~.Sync` : ``Lag(0)``.
    :class:`~.Seq` : ``Lag(1)``.
    """

    def __init__(self, ratio: float = 0.5, runtime: float | None = None, **kwargs):
        super().__init__(runtime, lag_ratio=ratio, new_animation=True, **kwargs)


class Sync(Lag):
    """Play the block's animations all at the same time.

    Everything inside starts together, so the block takes as long as the longest
    animation component animation rather than the sum of them.

    Parameters
    ----------
    runtime
        Passed to :class:`~.AnimationContext` .
    **kwargs
        Passed to :class:`~.Lag` with ``ratio=0`` .

    Examples
    --------
    .. code-block:: python

        with Sync():  # both happen over one second
            square.rotate(90, OUT)
            square.color = BLUE
    """

    def __init__(self, runtime: float | None = None, **kwargs):
        _reject_fixed_lag_ratio("Sync", 0, kwargs)
        super().__init__(ratio=0, runtime=runtime, **kwargs)


class Seq(Lag):
    """Play the block's animations one after another.

    Each animation starts as the previous one finishes, so the block lasts as long
    as all of them put together. This is what statements at the top level of a
    script already do; reach for ``Seq`` when you need to give a run of animations a
    combined runtime, or to sequence inside a :class:`~.Sync`.

    Parameters
    ----------
    runtime
        Passed to :class:`~.AnimationContext` .
    **kwargs
        Passed to :class:`~.Lag` with ``lag_ratio=1``.

    Examples
    --------
    .. code-block:: python

        with Seq(runtime=2):  # three moves squeezed into two seconds
            square.move(RIGHT)
            square.move(UP)
            square.move(LEFT)
    """

    def __init__(self, runtime: float | None = None, **kwargs):
        _reject_fixed_lag_ratio("Seq", 1, kwargs)
        super().__init__(ratio=1, runtime=runtime, **kwargs)


class Audio(AnimationContext):
    """Play a sound, and fit the block's animations to its length.

    The context's runtime is taken from the audio, so whatever you animate inside
    is stretched or squeezed to finish with the sound. That inversion is the point:
    you describe what should happen while a clip plays, rather than timing the clip
    against the animation.

    Parameters
    ----------
    source
        Path to an audio file, or an already-loaded moviepy audio clip.
    wait_at_end
        Extra seconds to hold after the audio finishes, before the block ends.
        Defaults to ``0``.
    **kwargs
        Passed to :class:`~.AnimationContext`. Note ``runtime`` is set from the
        audio and should not be passed.

    Examples
    --------
    .. code-block:: python

        with Audio("whoosh.wav"):
            square.move(RIGHT * 5)  # takes exactly as long as the clip

    See Also
    --------
    :class:`~.Speech` : The same, driven by a line of narration transcript.
    """

    def __init__(self, source: str, *, wait_at_end: float = 0.0, **kwargs):
        audio_clip = source
        if isinstance(source, str):
            from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

            audio_clip = AudioFileClip(source)
        # ``duration`` is moviepy's attribute name -- not Algan's ``runtime``.
        kwargs["runtime"] = audio_clip.duration + wait_at_end
        super().__init__(**kwargs)
        self.audio_clip = audio_clip

    def __enter__(self):
        context = super().__enter__()
        if self.prev_context.runtime_per_part > 0 and (
            self.prev_context.runtime is None or self.prev_context.runtime > 0
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
    transcript
        The line of narration to play. It is appended to the Scene's transcript and
        used to select the matching audio.
    wait_at_end
        Extra seconds to hold after the line finishes. Defaults to ``1``, leaving a
        beat between sentences.
    **kwargs
        Passed to :class:`~.Audio`.

    Examples
    --------
    .. code-block:: python

        with Speech("Now watch what happens when we double it."):
            square.scale(2)
    """

    def __init__(self, transcript, *, wait_at_end: float = 1.0, **kwargs):
        animation_manager = (
            kwargs.get("animation_manager") or _active_animation_manager()
        )
        kwargs["animation_manager"] = animation_manager
        audio_manager = animation_manager.scene.audio_manager
        audio_manager.append_script(transcript)
        super().__init__(
            audio_manager.get_speech(transcript), wait_at_end=wait_at_end, **kwargs
        )


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


class ComposedEasing(AnimationContext):
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

        with ComposedEasing(easings.there_and_back):
            square.move(RIGHT)  # goes out and comes back
    """

    def __init__(self, rfunc, **kwargs):
        kwargs["composed_easing"] = rfunc
        super().__init__(**kwargs)
