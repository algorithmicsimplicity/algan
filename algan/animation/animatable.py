from collections import defaultdict
import copy
from functools import wraps
import inspect

import torch

# Re-exported for backwards compatibility (it used to be defined here).
from algan.animation.timeline import TIME_PARAMETER_NAME, TimelineManager  # noqa: F401
from algan.animation.timeline import (
    STRUCTURE_VERSION,
    RowRanges,
    _opt_disabled,
    bump_structure_version,
)
from algan.scene import Scene
from algan.animation.animation_contexts import (
    Sync,
    AnimationManager,
    AnimationContext,
    Off,
)
from algan.constants.color import BLACK
from algan.utils.tensor_utils import HANDLED_FUNCTIONS, cast_to_tensor
from algan.scene_manager import SceneManager
from algan.utils.python_utils import traverse


def prepare_kwargs(self, func, args, kwargs, initial_args, unique_args):
    """Combine args and kwargs into one dict, using default values where arg is missing,
    and record the function application on the global timeline."""
    params = inspect.signature(func).parameters
    arg_names = list(params.keys())[1:]
    kwargs.update({arg_names[i]: args[i] for i in range(len(args))})
    default_kwargs = {
        param.name: param.default
        for param in params.values()
        if not (param.default is inspect._empty)
    }
    default_kwargs.update(kwargs)
    kwargs = {
        k: cast_to_tensor(v) if k in initial_args else v
        for k, v in default_kwargs.items()
    }
    timeline = TimelineManager.instance()
    timeline.record_function(
        func, self, initial_args, kwargs, self.animation_manager.context
    )
    return kwargs


def animated_function(
    function=None, *, animated_args=None, unique_args=()
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
    if animated_args is None:
        animated_args = {}

    def _decorate(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            if not self.is_animating():
                # Unspawned mobs record nothing: no function events (this
                # branch), and attribute writes go through the un-recorded
                # path of setattr_and_record_modification, which never touches
                # the context. The throwaway context below is then pure
                # overhead -- and mob construction performs thousands of
                # these calls.
                if not _opt_disabled("fastpath") and not (
                        hasattr(self, "id") and self.is_spawned()):
                    return func(self, *args, **kwargs)
                # Spawned but non-recording (e.g. inside Off(record_funcs=
                # False)): attribute edits still record timestamps against
                # the current context, so keep the context wrap.
                with AnimationContext(record_funcs=False):
                    return func(self, *args, **kwargs)
            else:
                with AnimationContext():
                    kwargs = prepare_kwargs(
                        self, func, args, kwargs, animated_args, unique_args
                    )
                    # Attribute edits made while func runs are attributed to
                    # the function application recorded by prepare_kwargs, so
                    # overlapping edits' replay windows can be resolved (see
                    # AnimationTimeline._resolve_replay_windows).
                    timeline = TimelineManager.instance()
                    previous_event = timeline.set_active_edit_event(
                        timeline.last_recorded_event
                    )
                    try:
                        with AnimationContext(record_funcs=False):
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

    All animation state lives on the global timeline
    (:class:`~algan.animation.timeline.AnimationTimeline`, via
    :class:`~algan.animation.timeline.TimelineManager`): each animatable
    attribute occupies rows of a per-attribute buffer keyed by this object's
    ``id``, attribute modifications and animated-function applications are
    recorded as timeline events, and the mob's :class:`~.Lifespan` (spawn /
    despawn interval) is likewise owned by the timeline
    (:attr:`~.Animatable.lifespan`).

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
        the modification is recorded on the global timeline so it can be replayed at render time.
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

    @property
    def lifespan(self):
        """This mob's [spawn, despawn) interval on the global timeline (a
        :class:`~algan.animation.timeline.Lifespan`). Sub-mobs created by
        indexing share their source's id, and therefore its lifespan."""
        return TimelineManager.instance().get_lifespan(self.id)

    def to(self, device):
        # Attribute values live in the global attribute timelines (which stay
        # on the animation device); nothing per-mob to move.
        return self

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
        function(self, t, *args, **kwargs)
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
            the time elapsed since the animation beginning (see
            :meth:`~algan.animation.timeline.AnimationTimeline.set_state_to_times`).
        *args, **kwargs
            Passed to function.

        """
        function(self, cast_to_tensor(time_elapsed), *args, **kwargs)
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
        int
            An ID identifying the updater that was added. This ID can be
            used to remove the updater at a later time, using
            :meth:`~.Animatable.remove_updater` . If it is never removed,
            the updater will continue forever.

        """
        timeline = TimelineManager.instance()
        # The span must be recorded on an *entered* context: only contexts
        # that enter and exit get their rescaled timestamps synced, so events
        # on the top-level context's timespan would all evaluate to time 0.
        with Off(record_funcs=False) as context:
            updater_id = timeline.record_updater(
                update_function, self, args, kwargs, context
            )
            # Apply once at elapsed = 0 so the scene-definition state reflects
            # the updater immediately (e.g. a tracker snaps to its target).
            update_function(self, cast_to_tensor(0.0), *args, **kwargs)
        return updater_id

    def remove_updater(self, updater_id):
        """Removes the specified updater from this mob's updaters, leaving the mob with whatever state the updater left
        it with at the time-stamp when it was removed.

        Parameters
        ----------
        updater_id
            The identifier of the updater to be removed. Can be given a value of -1 to remove the most
            recently added updater.

        """
        timeline = TimelineManager.instance()
        event = timeline.function_timeline.updaters[updater_id]
        if event.time.end_event is not None:
            # Already removed.
            return
        # See add_updater: the end event must live on an entered context.
        with Off(record_funcs=False) as context:
            timeline.end_updater(updater_id, context)
            # Record the updater's final state as an ordinary attribute
            # modification at the removal time, so the mob keeps it afterwards.
            elapsed = event.time.end_event() - event.time.start_event()
            event.function(
                event.caller, cast_to_tensor(elapsed), *event.args, **event.kwargs
            )

    def remove_all_updaters(self):
        timeline = TimelineManager.instance()
        for i, event in enumerate(timeline.function_timeline.updaters):
            if event.caller is self:
                self.remove_updater(i)

    def set_to_retroactive(self):
        timespan = self.animation_manager.context.timespan
        previous_time = timespan.current_time
        timespan.current_time = self.previous_retroactive_time
        self.previous_retroactive_time = previous_time

    def set_to_current(self):
        timespan = self.animation_manager.context.timespan
        timespan.current_time = self.previous_retroactive_time

    @property
    def animation_manager(self):
        if not hasattr(self, "_animation_manager"):
            return AnimationManager.instance()
        return self._animation_manager

    @animation_manager.setter
    def animation_manager(self, a):
        self._animation_manager = a

    def is_animating(self):
        if not hasattr(self, "id"):
            # Not yet fully constructed (e.g. mid-clone).
            return False
        return self.animation_manager.context.record_funcs and self.is_spawned()

    def generate_animatable_attr_set_get_methods(self):
        for attr in self.animatable_attrs:

            def setattr_general(value, attr=attr, self=self, recursive=True):
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
        timeline = TimelineManager.instance()
        timeline.add_mob_attr(self, key, value)
        return self

    def setattr_without_record(self, key, value, include_descendants=False):
        self._try_add_to_timeline(key, value)
        inds = self._get_attr_ranges(key, include_descendants=include_descendants)
        timeline = TimelineManager.instance()
        timeline.modify_attribute(key, inds, value)
        return self

    def setattr_and_rebatch_without_record(self, key, value):
        """Overwrites this mob's current value for ``key`` without recording a
        modification, re-allocating the mob's rows in the global attribute
        timeline when the batch size of ``value`` differs from the current
        rows. Past recorded modifications stay with the old rows, so this must
        only be used for structural rewrites (e.g. the batch expansions in
        :meth:`~.Mob.become`) on mobs whose history is fresh."""
        value = cast_to_tensor(value)
        timeline = TimelineManager.instance()
        timeline.add_mob_attr(self, key, value)
        attr_timeline = timeline.attr_to_timeline[key]
        inds = attr_timeline.mob_id_to_inds[self.id]
        if inds.shape[0] == value.shape[-2]:
            attr_timeline.modify(inds, value)
        else:
            attr_timeline.add(self, value, overwrite=True)
        self.batch_size = max(self.batch_size, value.shape[-2])
        return self

    def is_spawned(self):
        return self.lifespan.start() >= 0

    def is_despawned(self):
        return self.lifespan.end() >= 0

    def setattr_and_record_modification(self, key, value, include_descendants=False):
        inds = self._get_attr_ranges(key, include_descendants=include_descendants)
        timeline = TimelineManager.instance()

        context = self.animation_manager.context
        if (not self.is_spawned()) or (not context.record_attr_modifications):
            timeline.modify_attribute(key, inds, value)
            return self
        ts = context.timespan
        nt = ts.current_time + (context.run_time_unit if self.is_spawned() else 0)
        ts.original_end = max(ts.original_end, nt)
        timeline.modify_attribute_and_record(key, inds, value, ts.get_time(nt))
        return self

    def _get_attr_ranges(self, key, include_descendants=False, value=None):
        """This mob's rows of ``key``'s attribute buffer as a
        :class:`~algan.animation.timeline.RowRanges` (compressed [begin, end)
        runs; usually a single run, which the buffer reads/writes as a slice).

        The descendant union is re-read for every recorded function replay of
        every frame batch, so it is cached against the global structure
        version (bumped on any hierarchy / row-allocation change).
        """
        timeline = TimelineManager.instance()
        inds = timeline.get_inds(key, self, value)
        if not include_descendants:
            return timeline.attr_to_timeline[key].ranges_for(self.id)
        cache = getattr(self, "_attr_inds_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_attr_inds_cache", cache)
        hit = cache.get(key)
        if (hit is not None and hit[0] == STRUCTURE_VERSION[0]
                and not _opt_disabled("desccache")):
            return hit[1]
        if inds is None:
            return inds
        inds_list = [timeline.get_inds(key, m, value)
                     for m in self.get_descendants(include_self=True)]
        ranges = RowRanges.from_contiguous_blocks(inds_list)
        if ranges is None:  # non-contiguous block (defensive)
            ranges = RowRanges(None, tensor=torch.cat(inds_list))
        # Read the version after computing: get_inds may have allocated rows
        # (bumping it) when ``value`` is provided.
        cache[key] = (STRUCTURE_VERSION[0], ranges)
        return ranges

    def get_attr_inds(self, key, include_descendants=False, value=None):
        ranges = self._get_attr_ranges(
            key, include_descendants=include_descendants, value=value)
        return None if ranges is None else ranges.tensor()

    def get_animated_attribute(self, key, include_descendants=False, default=None):
        if default is not None:
            self._prepare_buffers(key, default)
        inds = self._get_attr_ranges(key, include_descendants=include_descendants, value=default)
        timeline = TimelineManager.instance()
        return timeline.get_attr(key, inds)

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
        object.__setattr__(clone, "_animation_manager", self.animation_manager)
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
        id_to_child = {i: c for i, c in enumerate(children_clones)}

        for k, v in self.__dict__.items():
            if k in [
                "id",
                "animation_manager",
                "_animation_manager",
                # Version-checked caches; clones rebuild their own lazily.
                "_attr_inds_cache",
                "_descendants_cache",
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
        add_to_scene=True,
        spawn=True,
        animate_creation=False,
        recursive=True,
        clone_data=True,
        reset_history=True,
    ):
        # reset_history is kept for backwards compatibility but no longer does
        # anything: a clone's history is fresh by construction (its new id has
        # no recorded timeline events); history hand-over is done explicitly by
        # Mob.detach_history.
        self.copy_add_to_scene = add_to_scene
        self.copy_spawn = spawn
        self.copy_animate_creation = animate_creation
        self.copy_recursive = recursive
        self.clone_data = clone_data
        c = copy.deepcopy(self)

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
        """Spawns the mob, introducing it into the video. Prior to spawning, a Mob will not appear on
        screen and any changes made to its animatable attributes will not be animated. After spawning,
        changes made to the Mob are animated by default.

        Parameters
        ----------
        animate
            Whether a spawn-in animation should be played. By default, the spawn-in animation is
            a simple fade-in. Defaults to True.

        """
        if self.is_spawned() or self.animation_manager.context.spawn_at_end:
            return self
        self._create_recursive(animate)
        self.animation_manager.context.on_create(self)
        return self

    def _create_recursive(self, animate=True):
        with Sync():
            lifespan = self.lifespan
            if lifespan.start() < 0:
                lifespan.start = self.animation_manager.context.get_current_time()
                TimelineManager.instance().register_spawn(self, lifespan)
                if animate:
                    self.on_create()
            for c in self.children:
                c._create_recursive(animate)
        return self

    def despawn(self, animate=True):
        if self.is_despawned():
            return self
        self._destroy_recursive(animate)
        self.animation_manager.context.on_destroy(self)

        return self

    def _destroy_recursive(self, animate=True):
        with Sync():
            lifespan = self.lifespan
            if lifespan.end() < 0:
                if animate:
                    self.on_destroy()
                lifespan.end = self.animation_manager.context.get_end_time()
                TimelineManager.instance().register_despawn(self, lifespan)
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

    def set_parent_to(self, other_mob):
        self.parents.append(other_mob)
        return self

    def add_children(self, *mobs):
        """Adds a collection of mobs to this mob's list of children, thereby
        propagating attribute changes made to the parent to the children.

        Parameters
        ----------
        *mobs : Iterable[:class:`~.Mob`]
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
        bump_structure_version()
        return self

    def remove_child(self, mob):
        self.children.remove(mob)
        bump_structure_version()
        return self
