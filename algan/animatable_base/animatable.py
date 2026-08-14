from __future__ import annotations

import copy
import inspect
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import torch

from algan.animation_timeline.animation_contexts import (
    AnimationContext,
    AnimationManager,
    Off,
    Sync,
    _reject_context_kwargs,
    active_scene_for_new_mob,
    animation_manager_bound,
)

# Re-exported for backwards compatibility (it used to be defined here).
from algan.animation_timeline.timeline import (  # noqa: F401
    HIERARCHY_VERSION,
    SPAWN_VERSION,
    STRUCTURE_VERSION,
    TIME_PARAMETER_NAME,
    RowRanges,
    TimelineManager,
    _opt_disabled,
)
from algan.constants.color import BLACK, Color
from algan.scene import Scene
from algan.utils.tensor_utils import HANDLED_FUNCTIONS, cast_to_tensor

#: Bumped whenever an animatable property is attached to a class, so per-class
#: views of "what can this be told to set" can be cached across Mobs.
ANIMATABLE_PROPERTY_VERSION = [0]


def prepare_kwargs(self, func, args, kwargs, initial_args, unique_args):
    """Combine args and kwargs and record the call on this mob's timeline."""
    params = inspect.signature(func).parameters
    arg_names = list(params.keys())[1:]
    kwargs.update({arg_names[i]: args[i] for i in range(len(args))})
    default_kwargs = {
        param.name: param.default
        for param in params.values()
        if param.default is not inspect._empty
    }
    default_kwargs.update(kwargs)
    kwargs = {
        k: cast_to_tensor(v) if k in initial_args else v
        for k, v in default_kwargs.items()
    }
    timeline = self.scene.timeline_manager
    timeline.record_function(
        func, self, initial_args, kwargs, self.animation_manager.context
    )
    return kwargs


def animated_function(function=None, *, animated_args=None, unique_args=()):
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
    if animated_args is None:
        animated_args = {}

    def _decorate(func):
        @animation_manager_bound
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            if not self.is_animating():
                # Mobs with nothing spawned in their subtree record nothing:
                # no function events (this branch), and attribute writes go
                # through the un-recorded path of
                # _setattr_and_record_modification, which never touches the
                # context. The throwaway context below is then pure overhead
                # -- and mob construction performs thousands of these calls.
                if not _opt_disabled("fastpath") and not (
                    hasattr(self, "id") and self.is_spawned_in_subtree()
                ):
                    return func(self, *args, **kwargs)
                # Spawned but non-recording (e.g. inside Off(record_funcs=
                # False)): attribute edits still record timestamps against
                # the current context, so keep the context wrap.
                with AnimationContext(
                    record_funcs=False, animation_manager=self.animation_manager
                ):
                    return func(self, *args, **kwargs)
            else:
                with AnimationContext(animation_manager=self.animation_manager):
                    kwargs = prepare_kwargs(
                        self, func, args, kwargs, animated_args, unique_args
                    )
                    # Attribute edits made while func runs are attributed to
                    # the function application recorded by prepare_kwargs, so
                    # overlapping edits' replay windows can be resolved (see
                    # AnimationTimeline._resolve_replay_windows).
                    timeline = self.scene.timeline_manager
                    previous_event = timeline.set_active_edit_event(
                        timeline.last_recorded_event
                    )
                    try:
                        with AnimationContext(
                            record_funcs=False, animation_manager=self.animation_manager
                        ):
                            out = func(self, **kwargs)
                    finally:
                        timeline.set_active_edit_event(previous_event)
                    self.animation_manager.context.increment_times()
            return out

        return wrapper_func

    if function:
        return _decorate(function)
    return _decorate


class Animatable:
    """Base class for anything that needs animation.

    All animation state lives on this mob's Scene-owned timeline
    (:class:`~algan.animation_timeline.timeline.AnimationTimeline`, via
    :class:`~algan.animation_timeline.timeline.TimelineManager`): each animatable
    attribute occupies rows of a per-attribute buffer keyed by this object's
    ``id``, attribute modifications and animated-function applications are
    recorded as timeline events, and the mob's
    :class:`~algan.animation_timeline.timeline.Lifespan` (spawn / despawn
    interval) is likewise owned by the timeline
    (:attr:`~algan.animatable_base.animatable.Animatable.lifespan`).

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
    data_sub_inds
        Specifies which indexes in the shared attribute rows this animatable will read and write from.
        Used to implement multiple sub-mobs which all share the same underlying data tensors, for batching purposes.
    parent_batch_sizes
        If this animatable's parent is batched, parent_batch_sizes specifies how the parent's attribute modifications
        will be expanded for this animatable's attributes.
    is_primitive
        Whether this animatable is a rendering primitive, i.e. needs to be kept around at rendering time.

    Attributes
    ----------

    animatable_attrs : list[str]
        Attribute names which will be treated as animatable. Whenever an animatable attribute is modified,
        the modification is recorded on this mob's Scene timeline for replay at render time.
    """

    def __init__(
        self,
        scene: Scene | None = None,
        add_to_scene: bool = True,
        name: str = "_",
        init: bool = True,
        animation_manager: AnimationManager | None = None,
        data_sub_inds: torch.Tensor | None = None,
        parent_batch_sizes: torch.Tensor | None = None,
        is_primitive: bool = False,
    ):
        if not hasattr(self, "animatable_attrs"):
            self.animatable_attrs = []

        self.generate_animatable_attr_set_get_methods()

        if scene is None:
            scene = active_scene_for_new_mob()
        self.scene = scene
        self.id = self.scene.get_new_id()
        if add_to_scene:
            self.scene.add_actor(self)

        if animation_manager is None:
            animation_manager = scene.animation_manager
        self.animation_manager = animation_manager
        if add_to_scene:
            animation_manager.context.add_mob(self)
        self.name = name

        self.anchor_priority = 0

        self.children = []
        self.components = []
        self.parents = []
        self.traversable = False
        self.parent_batch_sizes = parent_batch_sizes

        self.data_sub_inds = data_sub_inds
        self.batch_size = 1
        self.is_primitive = is_primitive

        self.previous_retroactive_time = 0
        self.ignore_wave_animations = False

        if init:
            self.init()

    def register_attrs_as_animatable(self, attrs: list[str], my_class=None):
        """Make attributes animatable, so writing them animates.

        Registered attributes get property getters and setters wired into the
        animation system: assigning to one records an interpolated change instead of
        overwriting a value. This is how a custom shader's parameters become things
        you can animate, and how a subclass exposes its own animatable state.

        Animation
        ---------
        Not animated: registration is setup. Call it in a subclass's ``__init__``
        before assigning the attributes -- registering after the first write leaves
        the value outside the animation system.

        Parameters
        ----------
        attrs
            Attribute name, or list of names, to register.
        my_class
            Class to attach the properties to. Defaults to ``None``, meaning this
            object's own class.
        """
        if isinstance(attrs, str):
            attrs = {
                attrs,
            }
        if not hasattr(self, "animatable_attrs"):
            self.animatable_attrs = []
        if my_class is None:
            my_class = self.__class__
        # Attaching a property to a class changes what that class can be told
        # to set, which is cached per class (Mob._settable_property_names).
        ANIMATABLE_PROPERTY_VERSION[0] += 1
        for attr in attrs:
            self._add_property_getter_and_setter(attr, my_class)
        self.animatable_attrs.extend(
            [_ for _ in attrs if _ not in self.animatable_attrs]
        )

    def _add_property_getter_and_setter(
        self, property_name: str, class_to_attach_to=None
    ):
        """Dynamically adds a property with a getter and setter for a given attribute name.

        The getter retrieves the current (potentially animated) value from
        this mob's Scene-owned attribute timeline; the setter writes the
        value to the timeline, recording the modification so it can be
        replayed at render time.

        Parameters
        ----------
        property_name
            The name of the property to create (e.g., 'location', 'color').
        class_to_attach_to : (type, optional)
            The class to which this property
            will be added. Defaults to the instance's own class.

        """
        if class_to_attach_to is None:
            class_to_attach_to = self.__class__
        if hasattr(class_to_attach_to, property_name):
            return

        tensor_subclass = Color if property_name == "color" else torch.Tensor

        @property
        def prop(self):
            return self.get_animated_attribute(property_name).as_subclass(
                tensor_subclass
            )

        @prop.setter
        def prop(self, value):
            return self.set_animated_attribute(property_name, value)

        setattr(class_to_attach_to, property_name, prop)

    @property
    def lifespan(self):
        """This mob's [spawn, despawn) interval on its Scene timeline (a
        :class:`~algan.animation_timeline.timeline.Lifespan`). Sub-mobs created by
        indexing share their source's id, and therefore its lifespan.
        """
        return self.scene.timeline_manager.get_lifespan(self.id)

    @animated_function(animated_args={"t": 0}, unique_args=["function"])
    def animate_function(self, function, t=1, *args, **kwargs):
        """Animate an arbitrary function of your own over this Mob.

        The function is called once per frame with a progress value that ramps up
        over the animation, which lets you drive any state you like -- the escape
        hatch for effects Algan has no built-in method for.

        Animation
        ---------
        Recorded as an animation: the function's second argument sweeps from 0 to
        ``t`` over the current context's duration (1 second by default). The
        function body must be vectorized over the Mob's batch; it runs on tensors,
        not on one part at a time.

        Parameters
        ----------
        function
            Callable taking ``(mob, t, *args, **kwargs)``. Its second argument is
            the interpolated value.
        t
            Value the second argument reaches at the end of the animation.
            Defaults to ``1``.
        *args, **kwargs
            Passed to ``function`` after the interpolated value.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.

        See Also
        --------
        :meth:`~.Animatable.animate_function_of_time` : Drive the function by elapsed seconds instead.
        :meth:`~.Animatable.add_updater` : Keep running every frame indefinitely.
        """
        function(self, t, *args, **kwargs)
        return self

    @animated_function(unique_args=["function"])
    def animate_function_of_time(self, function, time_elapsed=0, *args, **kwargs):
        """Animate a function of your own, driven by elapsed seconds.

        Like :meth:`~.Animatable.animate_function`, but the function receives the
        seconds elapsed rather than a 0-to-1 progress value. Use it when the effect
        is defined in real time -- a rotation of 90 degrees per second, say -- and
        you would rather not know in advance how long the animation runs.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). The function's second argument runs from 0 to that duration, in
        seconds. The body must be vectorized over the Mob's batch.

        Parameters
        ----------
        function
            Callable taking ``(mob, time_elapsed, *args, **kwargs)``, where
            ``time_elapsed`` is in seconds.
        time_elapsed
            Placeholder, overwritten per frame with the elapsed time. Defaults to
            ``0``; whatever you pass is ignored.
        *args, **kwargs
            Passed to ``function`` after the elapsed time.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        function(self, cast_to_tensor(time_elapsed), *args, **kwargs)
        return self

    def add_updater(self, update_function, *args, **kwargs) -> int:
        """Attach a function that runs every frame from now on.

        Updaters are how you get behaviour that persists rather than a one-off
        animation: a Mob that always faces the camera, a label that tracks a moving
        dot, an idle bobbing motion. The function is called on every frame, with the
        seconds elapsed since it was added, and keeps running until it is removed.

        Animation
        ---------
        Runs every frame for as long as it is attached, so it is unaffected by the
        current context's duration. It is applied once immediately at zero elapsed
        time, so the scene reflects it right away. Updaters are applied after
        recorded animations each frame, so an updater writing an attribute wins over
        an animation of that attribute.

        Parameters
        ----------
        update_function
            Callable taking ``(mob, time_elapsed, *args, **kwargs)``, where
            ``time_elapsed`` is in seconds. It must be vectorized over the Mob's
            batch.
        *args, **kwargs
            Passed to ``update_function`` after the elapsed time.

        Returns
        -------
        int
            Id of the updater, for passing to
            :meth:`~.Animatable.remove_updater`. An updater that is never removed
            runs for the rest of the video.

        Examples
        --------
        .. algan:: Example1AnimatableAddUpdater

            from algan import *

            square = Square().spawn()
            square.add_updater(lambda mob, t: mob.set(location=RIGHT * t))
            Scene.wait(2)

            Scene.save_video()
        """
        timeline = self.scene.timeline_manager
        # The span must be recorded on an *entered* context: only contexts
        # that enter and exit get their rescaled timestamps synced, so events
        # on the top-level context's timespan would all evaluate to time 0.
        with Off(
            record_funcs=False, animation_manager=self.animation_manager
        ) as context:
            updater_id = timeline.record_updater(
                update_function, self, args, kwargs, context
            )
            # Apply once at elapsed = 0 so the scene-definition state reflects
            # the updater immediately (e.g. a tracker snaps to its target),
            # while tracing every Mob state it reads or writes.
            event = timeline.function_timeline.updaters[updater_id]
            previous_trace = timeline.begin_updater_dependency_trace(event)
            try:
                update_function(self, cast_to_tensor(0.0), *args, **kwargs)
            finally:
                timeline.end_updater_dependency_trace(previous_trace)
        return updater_id

    def remove_updater(self, updater_id):
        """Stop an updater from running any further.

        The Mob keeps whatever state the updater left it in at this moment, rather
        than snapping back, so removing a "follow the dot" updater leaves the Mob
        where the dot last was.

        Animation
        ---------
        Not animated. Takes effect at the current timestamp: the updater runs on
        frames before this point and not after. Removing an already-removed updater
        does nothing.

        Parameters
        ----------
        updater_id
            Id returned by :meth:`~.Animatable.add_updater`. ``-1`` removes the
            most recently added updater.
        """
        timeline = self.scene.timeline_manager
        event = timeline.function_timeline.updaters[updater_id]
        if event.time.end_event is not None:
            # Already removed.
            return
        # See add_updater: the end event must live on an entered context.
        with Off(
            record_funcs=False, animation_manager=self.animation_manager
        ) as context:
            # Materialize the whole updater chain at the removal boundary before
            # ending this event. A removed updater may read state written by an
            # earlier updater (a satellite following a rotating hub); replaying
            # only this event against authoring-time state would make it read the
            # hub's stale initial orientation and snap backwards on the last frame.
            previous_capture, writes = timeline.begin_updater_write_capture(event)
            try:
                with Off(
                    record_attr_modifications=False,
                    record_funcs=False,
                    priority_level=float("inf"),
                    animation_manager=self.animation_manager,
                ):
                    timeline.set_state_to_times(
                        cast_to_tensor(context.timespan.current_time).reshape(1)
                    )
            finally:
                timeline.end_updater_write_capture(previous_capture)
                timeline.clear_buffers()

            span = timeline.end_updater(updater_id, context)
            # Preserve only the removed updater's boundary writes. Other active
            # updaters were materialized so dependency reads were current, but
            # recording their state here would make later frames apply them on
            # top of an already-advanced base state.
            for attr_name, indexes, value in writes:
                timeline.modify_attribute_and_record(
                    attr_name,
                    event.caller.id,
                    False,
                    indexes,
                    value,
                    span.end_event,
                )

    def remove_all_updaters(self):
        """Remove every updater attached to this Mob.

        Each one is removed as by :meth:`~.Animatable.remove_updater`, so the Mob
        keeps whatever state the updaters left it in.

        Animation
        ---------
        Not animated. Takes effect at the current timestamp: frames before it keep
        the updaters, frames after do not.
        """
        timeline = self.scene.timeline_manager
        for i, event in enumerate(timeline.function_timeline.updaters):
            if event.caller is self:
                self.remove_updater(i)

    def set_to_retroactive(self):
        """Rewind authoring time to this Mob's retroactive timestamp.

        Animation recorded after this call is inserted *earlier* in the video,
        which is how you go back and add something you only realised was needed
        later. Prefer the :meth:`~.Animatable.retroactive` context manager, which
        restores the timestamp even if the block raises.

        Animation
        ---------
        Not animated: this moves the authoring cursor, not the Mob.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        timespan = self.animation_manager.context.timespan
        previous_time = timespan.current_time
        timespan.current_time = self.previous_retroactive_time
        self.previous_retroactive_time = previous_time
        return self

    def set_to_current(self):
        """Return authoring time to where it was before rewinding.

        The counterpart of :meth:`~.Animatable.set_to_retroactive`.

        Animation
        ---------
        Not animated: this moves the authoring cursor, not the Mob.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        timespan = self.animation_manager.context.timespan
        timespan.current_time = self.previous_retroactive_time
        return self

    @contextmanager
    def retroactive(self):
        """Record animation back at this Mob's retroactive timestamp.

        Inside the ``with`` block, authoring time is rewound, so anything recorded
        is inserted earlier in the video; the timestamp is restored on exit even if
        the block raises.

        Animation
        ---------
        The block's own animations are recorded normally -- only *when* they happen
        changes.

        Examples
        --------
        .. code-block:: python

            with mob.retroactive():
                mob.color = BLUE  # happens earlier in the video
        """
        self.set_to_retroactive()
        try:
            yield self
        finally:
            self.set_to_current()

    @property
    def animation_manager(self):
        """This mob's scene-owned animation manager."""
        return self.scene.animation_manager

    @animation_manager.setter
    def animation_manager(self, manager):
        if manager is not self.scene.animation_manager:
            raise ValueError("A Mob must use the AnimationManager owned by its Scene")

    def is_animating(self) -> bool:
        """Whether changes to this Mob would currently be recorded as animation.

        True when the Mob is on screen (itself or through a child) *and* the current
        context records functions -- so it is False inside :class:`~.Off`, and False
        before the Mob has spawned, which is why setup done pre-spawn costs no video
        time.

        Returns
        -------
        bool
            Whether edits are being recorded right now.
        """
        if not hasattr(self, "id"):
            # Not yet fully constructed (e.g. mid-clone).
            return False
        return (
            self.animation_manager.context.record_funcs and self.is_spawned_in_subtree()
        )

    def generate_animatable_attr_set_get_methods(self):
        """Internal: install ``set_<attr>`` / ``get_<attr>`` helpers on this object.

        Gives every animatable attribute a matching pair of accessors, so a
        registered ``roughness`` attribute also answers to ``mob.set_roughness(...)``
        and ``mob.get_roughness()``. Called during construction; you do not call
        this yourself.
        """
        for attr in self.animatable_attrs:

            def setattr_general(value, attr=attr, self=self, recursive=True, **kwargs):
                if kwargs:
                    # ``mob.move(RIGHT, run_time=2)`` lands here via
                    # move -> move_to -> set_location. Catch the Manim timing
                    # idiom with a message that names the fix; re-raise anything
                    # else so genuine typos keep failing.
                    _reject_context_kwargs(kwargs)
                    raise TypeError(
                        f"set_{attr}() got an unexpected keyword argument "
                        f"'{next(iter(kwargs))}'"
                    )
                if recursive:
                    self.__setattr__(attr, value)
                else:
                    self.set_non_recursive(**{attr: value})
                return self

            super().__setattr__(f"set_{attr}", setattr_general)
            super().__setattr__(
                f"get_{attr}", lambda attr=attr: self.__getattribute__(attr)
            )

    def _try_add_to_timeline(self, key, value):
        timeline = self.scene.timeline_manager
        timeline.add_mob_attr(self, key, value)
        return self

    def _setattr_without_record(self, key, value, include_descendants: bool = False):
        """Internal: write an animatable attribute without recording an animation.

        The value changes for every frame, past and future, because nothing is
        recorded on the timeline. For ordinary authoring use plain assignment, or
        wrap it in :class:`~.Off` for an instant-but-recorded change.

        Parameters
        ----------
        key
            Name of the animatable attribute.
        value
            New value.
        include_descendants
            Whether descendants' rows are written too. Defaults to False.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        self._try_add_to_timeline(key, value)
        inds = self._get_attr_ranges(key, include_descendants=include_descendants)
        timeline = self.scene.timeline_manager
        timeline.modify_attribute(key, inds, value)
        return self

    def _setattr_and_rebatch_without_record(self, key, value):
        """Internal: overwrite an attribute, re-allocating rows if the shape changed.

        Unlike :meth:`~.Animatable._setattr_without_record`, this copes with a value
        whose batch size differs from the current one. Recorded history stays with
        the old rows, so it is only valid for structural rewrites (the batch
        expansions in
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`) on a Mob
        whose history is fresh -- see
        :meth:`~algan.animatable_base.mob.Mob.detach_history`.

        Parameters
        ----------
        key
            Name of the animatable attribute.
        value
            New value; may have a different batch size from the current one.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        value = cast_to_tensor(value)
        timeline = self.scene.timeline_manager
        timeline.add_mob_attr(self, key, value)
        attr_timeline = timeline.attr_to_timeline[key]
        inds = attr_timeline.mob_id_to_inds[self.id]
        if inds.shape[0] == value.shape[-2]:
            attr_timeline.modify(inds, value)
        else:
            attr_timeline.add(self, value, overwrite=True)
        self.batch_size = max(self.batch_size, value.shape[-2])
        return self

    def is_spawned(self) -> bool:
        """Whether this Mob has been spawned into the video.

        Stays True after :meth:`~.Animatable.despawn` -- it reports that the Mob has
        a spawn time, not that it is on screen right now. Pair it with
        :meth:`~.Animatable.is_despawned` for that.

        Returns
        -------
        bool
            Whether the Mob has a recorded spawn time.
        """
        return self.lifespan.start() >= 0

    def is_spawned_in_subtree(self) -> bool:
        """Whether this Mob or anything below it in the hierarchy has spawned.

        Containers are routinely left unspawned while their contents are
        spawned individually (``for mob in group: mob.spawn()``), and Group
        views (``group[1:3]``) never spawn at all. Such a container is on
        screen through its children, so modifications made to it must animate
        exactly as they do for a spawned mob: gating on :meth:`is_spawned`
        alone applies them instantly *and* records no timeline event, so the
        animation also contributes no time to the rendered video.

        The answer only changes when the hierarchy changes or something
        spawns, so it is cached against those two global versions.

        Returns
        -------
        bool
            Whether this Mob or any descendant has a recorded spawn time.
        """
        if self.is_spawned():
            return True
        children = getattr(self, "children", None)
        if not children:
            return False
        version = (HIERARCHY_VERSION[0], SPAWN_VERSION[0])
        cache = getattr(self, "_subtree_spawn_cache", None)
        if cache is not None and cache[0] == version:
            return cache[1]
        spawned = any(child.is_spawned_in_subtree() for child in children)
        object.__setattr__(self, "_subtree_spawn_cache", (version, spawned))
        return spawned

    def is_despawned(self) -> bool:
        """Whether this Mob has been despawned.

        Returns
        -------
        bool
            Whether the Mob has a recorded despawn time. A Mob that was never
            spawned reports False here as well as from
            :meth:`~.Animatable.is_spawned`.
        """
        return self.lifespan.end() >= 0

    def _setattr_and_record_modification(
        self, key, value, include_descendants: bool = False
    ):
        """Internal: write an animatable attribute and record it on the timeline.

        The write that ordinary attribute assignment goes through. Whether it
        becomes an animation depends on context: a Mob that is on screen and inside
        a recording context gets a timed edit, otherwise the value is written
        directly with no video time spent.

        Parameters
        ----------
        key
            Name of the animatable attribute.
        value
            New value.
        include_descendants
            Whether descendants' rows are written too. Defaults to False.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        timeline = self.scene.timeline_manager
        replay_inds = timeline.replay_inds(key, self.id, include_descendants)
        inds = (
            replay_inds
            if replay_inds is not None
            else self._get_attr_ranges(key, include_descendants=include_descendants)
        )
        timeline.capture_updater_write(key, inds, value)

        context = self.animation_manager.context
        if (
            replay_inds is not None
            or not self.is_spawned_in_subtree()
            or not context.record_attr_modifications
        ):
            timeline.modify_attribute(key, inds, value)
            if replay_inds is not None:
                timeline.replay_inds(key, self.id, include_descendants, consume=True)
            return self
        ts = context.timespan
        # Reached only for mobs that are on screen (themselves or through a
        # descendant), so the edit always spans the context's animation.
        nt = ts.current_time + context.run_time_unit
        ts.original_end = max(ts.original_end, nt)
        timeline.modify_attribute_and_record(
            key, self.id, include_descendants, inds, value, ts.get_time(nt)
        )
        return self

    def _get_attr_ranges(self, key, include_descendants=False, value=None):
        """This mob's rows of ``key``'s attribute buffer as a
        :class:`~algan.animation_timeline.timeline.RowRanges` (compressed [begin, end)
        runs; usually a single run, which the buffer reads/writes as a slice).

        The descendant union is re-read for every recorded function replay of
        every frame batch, so it is cached against the global structure
        version (bumped on any hierarchy / row-allocation change).
        """
        timeline = self.scene.timeline_manager
        inds = timeline.get_inds(key, self, value)
        # Trace after row allocation so a lazily introduced animatable
        # attribute can be materialized immediately on first updater access.
        timeline.trace_updater_mob_access(self, include_descendants)
        attr_timeline = timeline.attr_to_timeline[key]

        def ranges_for_mob(mob):
            mob_inds = attr_timeline.mob_id_to_inds[mob.id]
            sub_inds = getattr(mob, "data_sub_inds", None)
            if sub_inds is None:
                return attr_timeline.ranges_for(mob.id)
            # Packed geometry components commonly store only one row for
            # attributes they do not consume (for example control-point
            # colors/bases).  Their location rows are still independently
            # indexed, but a singleton attribute remains a shared broadcast
            # row instead of being duplicated for every geometry point.
            if mob_inds.numel() == 1:
                return attr_timeline.ranges_for(mob.id)
            selected = mob_inds[sub_inds]
            ranges = RowRanges.from_contiguous_blocks([selected])
            return ranges if ranges is not None else RowRanges(None, tensor=selected)

        if not include_descendants:
            return ranges_for_mob(self)
        cache = getattr(self, "_attr_inds_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_attr_inds_cache", cache)
        hit = cache.get(key)
        if (
            hit is not None
            and hit[0] == STRUCTURE_VERSION[0]
            and not _opt_disabled("desccache")
        ):
            return hit[1]
        if inds is None:
            return inds
        descendants = self.get_descendants(include_self=True)
        # A structural child can opt out of inheriting selected attributes
        # from its ancestors while remaining in the hierarchy for transforms,
        # lifespan, cloning, and direct animation.  Bezier border texture
        # points use this for ``color``: changing a circuit's fill color must
        # not overwrite its independently-authored border texture.
        descendants = [
            mob
            for mob in descendants
            if mob is self or key not in getattr(mob, "_excluded_from_parent_attrs", ())
        ]
        # Each mob's own rows are a cached single-run RowRanges; merge those
        # integer runs directly (RowRanges.from_runs) instead of re-deriving
        # each run from its index tensor. Fall back to concatenation only if a
        # mob's rows are non-contiguous (defensive; add() never does this).
        runs = []
        contiguous = True
        for m in descendants:
            timeline.get_inds(key, m, value)  # ensure rows are allocated
            r = ranges_for_mob(m)
            if r.pairs is None:
                contiguous = False
                break
            runs.extend(r.pairs)
        if contiguous:
            ranges = RowRanges.from_runs(runs) if runs else RowRanges([])
        else:
            inds_list = [ranges_for_mob(m).tensor() for m in descendants]
            ranges = RowRanges(None, tensor=torch.cat(inds_list))
        # Read the version after computing: get_inds may have allocated rows
        # (bumping it) when ``value`` is provided.
        cache[key] = (STRUCTURE_VERSION[0], ranges)
        return ranges

    def _get_attr_inds(self, key, include_descendants: bool = False, value=None):
        """Internal: get this Mob's row indices in an attribute's shared buffer.

        Parameters
        ----------
        key
            Name of the animatable attribute.
        include_descendants
            Whether to include descendants' rows. Defaults to False.
        value
            Value to size a first-time row allocation from. Defaults to ``None``,
            meaning do not allocate.

        Returns
        -------
        torch.Tensor or None
            The row indices, or ``None`` if the Mob has no rows for this attribute.
        """
        ranges = self._get_attr_ranges(
            key, include_descendants=include_descendants, value=value
        )
        return None if ranges is None else ranges.tensor()

    def get_animated_attribute(
        self, key, include_descendants: bool = False, default=None, copy: bool = True
    ):
        """Get an animatable attribute's current authoring value.

        This is the value as the scene is being authored -- the state the Mob has
        reached at the current point in the timeline -- not the value at any
        particular rendered frame. Plain attribute access
        (``mob.location``) goes through this.

        Parameters
        ----------
        key
            Name of the animatable attribute.
        include_descendants
            Whether to return descendants' values as well, stacked into the batch
            dimension. Defaults to False.
        default
            Value used to seed the attribute if the Mob has none yet. Defaults to
            ``None``, meaning do not create one.
        copy
            Whether to return a copy. Defaults to True; pass False only for a
            read you will not retain, since the underlying buffer is reused.

        Returns
        -------
        torch.Tensor
            The attribute's current value.
        """
        timeline = self.scene.timeline_manager
        # A read must land on the rows the replayed function will write, which
        # are not necessarily the ones at the replay cursor -- see
        # AnimationTimeline.peek_replay_inds.
        replay_inds = timeline.peek_replay_inds(key, self.id, include_descendants)
        if default is not None and replay_inds is None:
            self._prepare_buffers(key, default)
        inds = (
            replay_inds
            if replay_inds is not None
            else self._get_attr_ranges(
                key, include_descendants=include_descendants, value=default
            )
        )
        return timeline.get_attr(key, inds, copy=copy)

    def wait(self, *args, **kwargs):
        """Hold still for a while before the next animation.

        Advances authoring time without changing anything, which leaves a pause in
        the video. Same as :meth:`~.Scene.wait`, reachable from a Mob so it can be
        chained between animations.

        Animation
        ---------
        Recorded on the timeline: it consumes video time and nothing else.

        Parameters
        ----------
        *args, **kwargs
            Passed to the current context's ``wait`` -- notably the duration in
            seconds, which defaults to 1.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        self.animation_manager.context.wait(*args, **kwargs)
        return self

    def get_default_color(self):
        """Get the colour this Mob uses when none was given.

        Override in a subclass to give a shape its own default; the built-in shapes
        do exactly that.

        Returns
        -------
        :class:`~.Color`
            ``BLACK`` for the base class.
        """
        return BLACK

    def on_init(self):
        """Hook called once when the Mob is constructed.

        Does nothing by default. Override it to run setup that must happen before
        spawning, e.g. building child Mobs.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        return self

    def on_create(self):
        """Hook called by :meth:`~.Animatable.spawn` to play an entrance animation.

        Does nothing at this level; :class:`~.Mob` overrides it with a fade-in.
        Override it to give a subclass its own entrance.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        return self

    def on_destroy(self):
        """Hook called by :meth:`~.Animatable.despawn` to play an exit animation.

        Does nothing at this level; :class:`~.Mob` overrides it with a fade-out.
        Override it to give a subclass its own exit.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        return self

    def identity(self):
        """Return this object unchanged.

        A do-nothing placeholder for APIs that expect a transform function.

        Returns
        -------
        :class:`~.Animatable`
            This object.
        """
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Clone policy travels in deepcopy's private memo instead of being
        # written temporarily onto the source Mob (which was non-reentrant and
        # left implementation attributes visible on user objects).
        memo.setdefault("___copy_add_to_scene___", False)
        memo.setdefault("___copy_spawn___", False)
        memo.setdefault("___copy_animate_creation___", False)
        memo.setdefault("___copy_recursive___", True)
        memo.setdefault("___clone_data___", True)
        add_to_scene = memo["___copy_add_to_scene___"]
        spawn = memo["___copy_spawn___"]
        animate_creation = memo["___copy_animate_creation___"]
        copy_recursive = memo["___copy_recursive___"]
        clone_data = memo["___clone_data___"]
        cls = self.__class__
        clone = cls.__new__(cls)
        clone.parents = []
        clone.anchor_priority = 0
        memo[id(self)] = clone
        object.__setattr__(clone, "scene", self.scene)
        object.__setattr__(clone, "animatable_attrs", self.animatable_attrs)

        if clone_data:
            # A new id gives the clone fresh timeline rows (filled from the
            # original's current values by the attribute copy below) and a
            # fresh lifespan.
            clone.id = self.scene.get_new_id()
        else:
            # Share the original's timeline rows and lifespan.
            clone.id = self.id
        if add_to_scene:
            self.scene.add_actor(clone)
            self.animation_manager.context.add_mob(clone)
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
        id_to_child = dict(enumerate(children_clones))

        # The per-attribute set_*/get_* accessors are instance-level closures
        # over *this* mob, and the clone regenerates its own below. Copying
        # them was two wasted deepcopy dispatches per animatable attribute on
        # every cloned mob -- and a clone that somehow kept one would drive the
        # original.
        generated_accessors = frozenset(
            name
            for attr in self.animatable_attrs
            for name in (f"set_{attr}", f"get_{attr}")
        )

        for k, v in self.__dict__.items():
            if k in generated_accessors:
                continue
            if k in [
                "id",
                "animation_manager",
                "_animation_manager",
                # Version-checked caches; clones rebuild their own lazily.
                "_attr_inds_cache",
                "_descendants_cache",
                "_subtree_spawn_cache",
            ]:
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
        if clone_data:
            for attr in self.animatable_attrs:
                setattr(clone, attr, getattr(self, attr, None))
        if self.is_spawned() and spawn:
            clone.spawn(animate_creation)
        if copy_recursive:
            clone.add_children(*children_clones)
        clone.components = component_clones
        return clone

    def clone(
        self,
        add_to_scene: bool = True,
        spawn: bool = True,
        animate_creation: bool = False,
        recursive: bool = True,
        clone_data: bool = True,
    ):
        """Make a copy of this Mob, by default spawned into the scene.

        The copy starts out identical -- same position, colour, material, children --
        but is independent from then on: animating one does not affect the other. Use
        it to stamp out repeated shapes, or to get a second version of something you
        are about to change.

        The copy has its own animation history rather than inheriting this Mob's, so
        it does not replay the original's past animations.

        Animation
        ---------
        Not animated by default: the copy appears instantly, already in place. Pass
        ``animate_creation=True`` for it to fade in over the current context's
        duration (1 second by default).

        Parameters
        ----------
        add_to_scene
            Whether the copy is added to the scene. Defaults to True; False makes a
            detached Mob, useful purely as a source of values (this is what
            :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` does
            with its target).
        spawn
            Whether the copy is spawned, i.e. visible. Defaults to True; pass False
            to configure it -- including anything that must happen before spawning,
            such as
            :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`
            -- and spawn it yourself later.
        animate_creation
            Whether spawning the copy plays its entrance animation. Defaults to
            False, so the copy simply appears.
        recursive
            Whether children are copied too. Defaults to True; False copies this Mob
            alone, leaving it childless.
        clone_data
            Whether the copy gets its own animation data. Defaults to True. False
            produces a **view** that shares this Mob's data and identity, so
            animating the view animates the original -- what indexing (``mob[0]``)
            uses.

        Returns
        -------
        :class:`~.Animatable`
            The new copy. When ``clone_data`` is False this is a view of this Mob
            rather than an independent object.

        Examples
        --------
        .. algan:: Example1AnimatableClone

            from algan import *

            square = Square(color=BLUE).spawn()
            copy = square.clone()
            copy.move(RIGHT * 2)
            copy.color = RED

            Scene.save_video()
        """
        memo = {
            "___copy_add_to_scene___": bool(add_to_scene),
            "___copy_spawn___": bool(spawn),
            "___copy_animate_creation___": bool(animate_creation),
            "___copy_recursive___": bool(recursive),
            "___clone_data___": bool(clone_data),
        }
        c = copy.deepcopy(self, memo)

        if clone_data:
            c.batch_size = 1
            for d in c.get_descendants():
                for attr in ["location", "opacity", "basis", "color"]:
                    dloc = d.__getattribute__(attr)
                    if dloc is None:
                        continue
                    d.batch_size = max(c.batch_size, dloc.shape[-2])
                d.data_sub_inds = None
        return c

    def spawn(self, animate: bool = True):
        """Bring the Mob into the video.

        The mob does not appear on screen until it is spawned.
        Changes made before spawning are not animated.
        After spawning, changes to the Mob animate by
        default and are controlled by
        :class:`~algan.animation_timeline.animation_contexts.AnimationContext`.

        Spawning is recursive: children spawn with their parent. Spawning a Mob that
        is already spawned does nothing.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default): the Mob fades in. ``spawn(animate=False)`` makes it appear
        immediately, no animation.

        Parameters
        ----------
        animate
            Whether to play the entrance animation, by default a fade-in (see
            :meth:`~.Mob.on_create`). Defaults to True.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained -- which is what makes
            ``square = Square().spawn()`` work.

        See Also
        --------
        :meth:`~.Animatable.despawn` : Remove the Mob from the video again.
        """
        if self.is_spawned() or self.animation_manager.context.spawn_at_end:
            return self
        self._create_recursive(animate)
        self.animation_manager.context.on_create(self)
        return self

    def _create_recursive(self, animate=True):
        with Sync(animation_manager=self.animation_manager):
            lifespan = self.lifespan
            if lifespan.start() < 0:
                lifespan.start = self.animation_manager.context.get_current_time()
                self.scene.timeline_manager.register_spawn(self, lifespan)
                if animate:
                    self.on_create()
            for c in self.children:
                c._create_recursive(animate)
        return self

    def despawn(self, animate: bool = True):
        """Remove the Mob from the video.

        The Mob stops being drawn from this point on, but the animation already
        recorded for it is untouched -- everything it did before still plays. A
        despawned Mob cannot be brought back; clone it before despawning if you need
        it again later.

        Despawning is recursive: children despawn with their parent. Despawning a Mob
        that is already despawned does nothing.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default): the Mob fades out. ``despawn(animate=False)`` removes it instantly,
        which is what you want for something already off-screen.

        Parameters
        ----------
        animate
            Whether to play the exit animation, by default a fade-out (see
            :meth:`~.Mob.on_destroy`). Defaults to True.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        if self.is_despawned():
            return self
        self._destroy_recursive(animate)
        self.animation_manager.context.on_destroy(self)

        return self

    def _destroy_recursive(self, animate=True):
        with Sync(animation_manager=self.animation_manager):
            lifespan = self.lifespan
            # An unspawned container may still own independently spawned
            # children. Do not let its direct opacity write erase their
            # recorded pre-despawn state; recurse and destroy those children.
            if lifespan.start() >= 0 and lifespan.end() < 0:
                if animate:
                    self.on_destroy()
                lifespan.end = self.animation_manager.context.get_end_time()
                self.scene.timeline_manager.register_despawn(self, lifespan)
            for c in self.children:
                c._destroy_recursive(animate)
        return self

    def init(self):
        """Run this Mob's initialization hooks.

        Calls :meth:`~.Animatable.on_init` and lets the current context do its own
        setup. Constructors call this for you unless they were passed
        ``init=False``.

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
        self.on_init()
        self.animation_manager.context.on_init(self)
        return self

    def delete(self):
        """Remove the Mob from the video; an alias for :meth:`~.Animatable.despawn`.

        Animation
        ---------
        Recorded as an animation: the Mob fades out over the current context's
        duration (1 second by default).

        Returns
        -------
        :class:`~.Animatable`
            This object, so calls can be chained.
        """
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

    def _get_memory_used_per_timestep(self) -> int:
        """Get this Mob's render memory cost for one frame, in bytes.

        Used by the render loop to size frame batches. The base class holds no
        render data and so reports ``0``; renderable Mobs override this.

        Returns
        -------
        int
            Bytes needed per frame, ``0`` for a Mob with no render primitives.
        """
        return 0
