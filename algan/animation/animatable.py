from collections import defaultdict
import copy
import functools
from functools import wraps
import inspect
from typing import Dict

import torch
import torch.nn.functional as F

from algan.scene import Scene
from algan.animation.animation_contexts import Sync, AnimationManager, AnimationContext
from algan.constants.color import BLACK
from algan.utils.tensor_utils import broadcast_all, robust_concat, concat_dicts, prepare_kwargs, HANDLED_FUNCTIONS
from algan.scene_tracker import SceneTracker
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import broadcast_gather, cast_to_tensor, cast_to_tensor_single, unsqueeze_dims


class ModificationHistory:
    """A complete record of every change applied to a particular Mob, and the timestamps those changes occur over.
    At render time, this history will be used to construct the animated (interpolated) attributes of the mob for all timesteps.
    """

    def __init__(self):
        self.function_applications = dict() # contains all animated_functions applied to the mob.
        self.attribute_modifications = defaultdict(list) # contains every instance that one of the mob's animatable attributes is changed.
        self.attribute_overwrites = dict() # overwrites are not supported at the moment.

    def overwrite_attr_history(self, attr, end_time):
        if attr not in self.attribute_overwrites:
            self.attribute_overwrites[attr] = []
        self.attribute_overwrites[attr].append(end_time)

    def insert_attr_modification(self, attr, new_value, animation_context):
        if not animation_context.record_attr_modifications:
            return
        start_time = animation_context.get_current_time()
        end_time = animation_context.get_current_end_time()
        animation_context.end_time = max(animation_context.end_time, animation_context.current_time + animation_context.run_time_unit)
        self.attribute_modifications[attr].append([new_value, start_time, end_time])
        return self

    def insert_function_application(self, func_name, func, animated_args, kwargs, animation_context):
        if animation_context.run_time_unit <= 0 or not animation_context.record_funcs:
            return
        start_time = animation_context.get_current_time()
        end_time = animation_context.get_current_end_time()
        rate_func = animation_context.rate_func
        rate_func_compose = animation_context.rate_func_compose

        if func_name not in self.function_applications:
            self.function_applications[func_name] = (func, [], [])
        self.function_applications[func_name][-1].append([animated_args, kwargs, start_time, end_time, (rate_func, rate_func_compose)])

    def get_history(self, animatable):
        attrs = []

        for attr in self.attribute_modifications:
            v = animatable.data.animatable.__getattribute__(attr) # current (i.e. ending) value
            history = [(v, e) for v, s, e in self.attribute_modifications[attr]] + [(v, lambda: float('inf'))]
            values, times = zip(*history)
            # Note that times are stored as functions so they can be retroactively changed by animation contexts, here we evaluate them to get the actual (float) times.
            attrs.append((attr, robust_concat(values), torch.tensor([_() for _ in times])))

        funcs = []

        for func_name, (func, modified_attrs, arg_list) in sorted(self.function_applications.items(), key=lambda _: 0 if 'setattr_' in _[0] else (1 if 'update_relative' in _[0] else 2)):
            animated_args, kwargs, start_time, end_time, rate_funcs = zip(*arg_list)
            kwargs = concat_dicts(kwargs)
            animated_args = concat_dicts(animated_args)
            animated_args = {k: unsqueeze_dims(v, kwargs[k], 1) for k, v in animated_args.items()} #TODO shouldn't the unsqueeze dim be 0?
            start_time, end_time = torch.tensor([_() for _ in start_time]), torch.tensor([_() for _ in end_time])

            # In the event where 2 or more functions take place at the same time (overlap), we manually split them up.
            # into separate funcs.
            overlaps = (~((start_time > end_time.unsqueeze(-1)) | (end_time < start_time.unsqueeze(-1)))).float()
            orders = (overlaps * torch.triu(torch.ones_like(overlaps), diagonal=1)).sum(-2)
            unique_orders, uinds = torch.unique(orders, return_inverse=True)
            for order in unique_orders:#[-1:]:
                def g(x):
                    if not isinstance(x, torch.Tensor):
                        return x
                    return x[uinds == order]
                sub_rate_funcs = [rate_funcs[i] for i in (uinds == order).nonzero()]
                funcs.append((func, {k: g(v) for k, v in animated_args.items()}, {k: g(v) for k, v in kwargs.items()},
                              g(start_time), g(end_time), sub_rate_funcs))
        return attrs, funcs


def animated_function(function=None, *, animated_args:Dict[str, float]={'t': 0}, unique_args=list()):
    """Decorator that turns a function into an animated function. The animation is created by interpolating
    all args named in the animated_args dict from the value provided in this dict the value passed as an actual argument
    when the function is called. Most commonly, animated_args will just be {'t': 0}, and the function
    will be called with t=1.

    Parameters
    ----------
    function
        The function to be decorated. It MUST accept a Mob as its first argument, and any arguments
        given in animated_args or unique_args must also be arguments of this function.

    animated_args
        A dictionary with strings as keys and floats as values. The strings are names of arguments which will
        be animated. The arguments will be animated by linearly interpolating their values from the corresponding
        value provided in the animated_args dict to the value they have when the function is called.
        
    unique_args
        A list of strings. This is only for batching, when the function is called with different values for a unique
        argument, they will be batched as two entirely separate functions. Any arguments named in unique_args MUST
        only accept string values.
    """
    def _decorate(func):
        @wraps(func)
        def wrapper_func(self, *args, **kwargs):
            if not self.is_animating():
                with AnimationContext(record_funcs=False):
                    return func(self, *args, **kwargs)
            with AnimationContext():
                kwargs = prepare_kwargs(self, func, args, kwargs, animated_args, unique_args)
                with AnimationContext(record_funcs=False):
                    out = func(self, **kwargs)
            self.animation_manager.context.increment_times()
            return out
        return wrapper_func

    if function:
        return _decorate(function)
    return _decorate


class AnimatableData:
    """A container for all of the animation-relevant data for a mob, including its ModificationHistory.

    Parameters
    ----------
    animatable
        The mob for which we are recording data.
    data_dict_active
        The dictionary used for storing active animatable attribute values. At animate time this is simply the current value,
        at render time it is all values for the current batch of timesteps being rendered.
    data_dict_materialized
        The dictionary used for storing all materialized animatable attribute values. At animate time this isn't used,
        at render time this contains all values for a (larger) batch of timesteps being materialized.
    history
        The ModificationHistory object to which modifications will be recorded.
    time_inds_materialized
        A list of all time-inds which have been materialized (i.e. are in data_dict_materialized).
    time_inds_active
        A list of all time-inds which are currently active (i.e. are in data_dict_active).
    spawn_time
        (function which yields) the timestamp at which the mob spawned.
    despawn_time
        (function which yields) the timestamp at which the mob despawned.
    """
    def __init__(self, animatable, data_dict_active=None, data_dict_materialized=None, history=None,
                 time_inds_materialized=None, time_inds_active=None, spawn_time=lambda: -1, despawn_time=lambda: -1):
        self.animatable = animatable
        if data_dict_active is None:
            data_dict_active = dict()
        if data_dict_materialized is None:
            data_dict_materialized = dict()
        if history is None:
            history = ModificationHistory()
        self.data_dict_active = data_dict_active
        self.data_dict_materialized = data_dict_materialized
        self.history = history
        self.time_inds_active = time_inds_active
        self.time_inds_materialized = time_inds_materialized
        self.data_dict = self.data_dict_active
        self.spawn_time = spawn_time
        self.despawn_time = despawn_time
        self.set_pre_function_application = False


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
    def __init__(self,
                 scene: Scene | None=None,
                 add_to_scene: bool=True,
                 name: str='_',
                 init: bool=True,
                 animation_manager: AnimationManager | None=None,
                 data: AnimatableData | None=None,
                 data_sub_inds: torch.Tensor | None=None,
                 parent_batch_sizes: torch.Tensor | None=None,
                 is_primitive: bool=False):

        if not hasattr(self, 'animatable_attrs'):
            self.animatable_attrs = set()

        self.generate_animatable_attr_set_get_methods()

        if scene is None:
            scene = SceneTracker.instance()
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
        self.parents = []
        self.traversable = True
        self.parent_batch_sizes = parent_batch_sizes

        self.data_sub_inds = data_sub_inds
        self.batch_size = max([1, *[_.shape[-2] for _ in self.data.data_dict_active.values()]])
        self.is_primitive = is_primitive

        for attr in self.animatable_attrs:
            uattr = f'_{attr}'
            if hasattr(self, uattr):
                self.data.data_dict_active[attr] = self.__getattribute__(uattr)
                delattr(self, uattr)
        #setup_getters(self)
        self.previous_retroactive_time = 0
        self.reset_state()

        if init:
            self.init()

    def set_to_retroactive(self):
        prt = self.animation_manager.context.current_time
        self.animation_manager.context.current_time = self.previous_retroactive_time
        self.previous_retroactive_time = prt

    def set_to_current(self):
        self.animation_manager.context.current_time = self.previous_retroactive_time

    @property
    def animation_manager(self):
        if not hasattr(self, '_animation_manager'):
            return AnimationManager.instance()
        return self._animation_manager

    @animation_manager.setter
    def animation_manager(self, a):
        self._animation_manager = a

    def is_animating(self):
        if not (hasattr(self, 'animation_manager') and hasattr(self, 'data')):
            return False
        return self.animation_manager.context.record_funcs and self.data.spawn_time() >= 0

    def generate_animatable_attr_set_get_methods(self):
        for attr in self.animatable_attrs:
            def setattr_general(value, attr=attr, self=self, recursive=True):
                if recursive:
                    self.__setattr__(attr, value)
                else:
                    self.setattr_non_recursive(attr, value)
                return self
            super().__setattr__(f'set_{attr}', setattr_general)
            super().__setattr__(f'get_{attr}', lambda attr=attr: self.__getattribute__(attr))

    def setattr_and_record_modification(self, key, value):
        dd = self.data.data_dict
        self.batch_size = max(self.batch_size, value.shape[-2])
        n1, n2 = 1 if self.data.time_inds_materialized is None else len(self.data.time_inds_materialized), self.batch_size
        data_inds = self.data_sub_inds if self.data_sub_inds is not None else slice(None)
        data_inds = torch.arange(n2)[data_inds]
        time_inds = self.data.time_inds_active if self.data.time_inds_materialized is not None else slice(None)
        time_inds = torch.arange(n1)[time_inds].unsqueeze(-1)
        if key not in dd:
            if self.data.time_inds_materialized is None:
                # This is the first time this attr's value has been set.
                old_value = self.getattribute_animated_full(key)
                if old_value is None:
                    old_value = torch.zeros_like(value[:1, :1])
                else:
                    old_value = unsqueeze_dims(old_value, value)
                if (old_value.shape[0] < n1) or (old_value.shape[1] < n2):
                    old_value = old_value.expand(n1, n2, -1).contiguous()
                new_value = old_value.clone()
                new_value[time_inds, data_inds] = value
                dd[key] = new_value
                self.data.history.insert_attr_modification(key, old_value, self.animation_manager.context)
            else:
                # We are at render time and this attr has never been modified, just fill with current value.
                dd[key] = self.__getattribute__(key).expand(len(self.data.time_inds_materialized), -1, -1)
            return
        if (dd[key].shape[0] < n1) or (dd[key].shape[1] < n2):
            dd[key] = dd[key].expand(n1, n2, -1).contiguous()
        old_value = dd[key]
        new_value = old_value.clone()
        new_value[time_inds, data_inds] = value
        dd[key] = new_value
        if self.data.spawn_time() >= 0:
            self.data.history.insert_attr_modification(key, old_value, self.animation_manager.context)

    def getattribute_animated_full(self, key):
        if key not in self.data.data_dict:
            if key not in self.data.data_dict_active:
                return super().__getattribute__(key)
            return self.data.data_dict_active[key]
        return self.data.data_dict[key]

    def getattribute_animated(self, key):
        if key not in self.data.data_dict:
            if key not in self.data.data_dict_active:
                raise AttributeError
            value = self.data.data_dict_active[key]
        else:
            value = self.data.data_dict[key]

        if not isinstance(value, torch.Tensor):
            return value
        while value.dim() < 3:
            value = value.unsqueeze(0)
        if self.data.time_inds_materialized is not None and value.shape[0] == 1:
            value = value.expand(self.data.time_inds_active.amax()+1, -1, -1)
        data_inds = self.data_sub_inds if self.data_sub_inds is not None else slice(None)
        time_inds = self.data.time_inds_active if self.data.time_inds_materialized is not None else slice(None)
        if value.shape[1] == 1 and self.data_sub_inds is not None:
            return value[time_inds]
        return value[time_inds][:, data_inds]

    def wait(self, *args, **kwargs):
        """An animated function that does nothing for one second!
        """
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
        if '___copy_add_to_scene___' not in memo:
            memo['___copy_add_to_scene___'] = self.copy_add_to_scene
        if '___copy_animate_creation___' not in memo:
            memo['___copy_animate_creation___'] = self.copy_animate_creation
        if '___copy_recursive___' not in memo:
            memo['___copy_recursive___'] = self.copy_recursive
        if '___clone_data___' not in memo:
            memo['___clone_data___'] = self.clone_data
        if '___reset_history___' not in memo:
            memo['___reset_history___'] = self.reset_history
        add_to_scene = memo['___copy_add_to_scene___']
        animate_creation = memo['___copy_animate_creation___']
        copy_recursive = memo['___copy_recursive___']
        clone_data = memo['___clone_data___']
        reset_history = memo['___reset_history___']
        cls = self.__class__
        clone = cls.__new__(cls)
        clone.parents = []
        clone.anchor_priority = 0
        memo[id(self)] = clone
        object.__setattr__(clone, 'scene', self.scene)
        object.__setattr__(clone, '_animation_manager', self.animation_manager)
        object.__setattr__(clone, 'animatable_attrs', self.animatable_attrs)

        if clone_data:
            oa = self.data.animatable
            self.data.animatable = None
            ti = copy.deepcopy(self.data, memo)
            ti.animatable = clone
            if reset_history:
                ti.history = ModificationHistory()
            self.data.animatable = oa
        else:
            ti = self.data

        object.__setattr__(clone, 'data', ti)
        if not clone_data:
            object.__setattr__(clone, 'data', self.data)
        if add_to_scene:
            self.scene.add_actor(clone)
            self.animation_manager.context.add_mob(clone)
        clone.id = self.scene.get_new_id()
        children = list(object.__getattribute__(self, 'children')) if (hasattr(self, 'children') and copy_recursive) else []
        children_clones = [copy.deepcopy(c, memo) for c in children] if copy_recursive else []

        child_to_id = {c: i for i, c in enumerate(children)}
        id_to_child = {i: c for i, c in enumerate(children_clones)}

        for k, v in self.__dict__.items():
            if k in ['video', 'id', 'created', 'destroyed', 'spawn_time', 'despawn_time', 'animation_manager', '_animation_manager', 'time_inds', 'history']:
                continue
            if k == 'data' and not clone_data:
                continue
            if k in ['parents']:
                object.__setattr__(clone, k, [])
                continue
            if isinstance(v, Animatable) and v in children:
                object.__setattr__(clone, k, id_to_child[child_to_id[v]])
                continue
            if k in ['children']:
                v = []
            if k in ['anchors']:
                v = defaultdict(list)
            object.__setattr__(clone, k, copy.deepcopy(v, memo))

        clone.generate_animatable_attr_set_get_methods()
        clone.spawn(animate_creation)
        if copy_recursive:
            clone.add_children(*children_clones)
        return clone

    def clone(self, add_to_scene=True, animate_creation=False, recursive=True, clone_data=True, reset_history=True):
        self.copy_add_to_scene = add_to_scene
        self.copy_animate_creation = animate_creation
        self.copy_recursive = recursive
        self.clone_data = clone_data
        self.reset_history = reset_history
        c = copy.deepcopy(self)

        if clone_data:
            c.batch_size = 1
            for d in c.get_descendants():
                for attr in ['location', 'opacity', 'basis', 'color']:
                    dloc = d.__getattribute__(attr)
                    d.data.data_dict_active[f'{attr}'] = dloc
                    if dloc is None:
                        continue
                    d.batch_size = max(c.batch_size, dloc.shape[-2])
                d.data_sub_inds = None
        return c

    def spawn(self, animate=True):
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
        if (self.data.despawn_time() >= 0) or (self.data.spawn_time() < 0):
            return self
        self._destroy_recursive(animate)
        self.animation_manager.context.on_destroy(self)
        return self

    def _destroy_recursive(self, animate=True):
        with Sync():
            if self.data.despawn_time() < 0:
                if animate:
                    self.on_destroy()
                self.data.despawn_time = self.animation_manager.context.get_current_end_time()
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
        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, Animatable)) for t in types):
            args = [a.location if hasattr(a, 'location') else a for a in args]
            return func(*args, **kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def set_state_to_time_t(self, time_inds):
        self.data.time_inds_active = (self.data.time_inds_materialized.view(-1,1) == time_inds.view(1,-1)).sum(1).nonzero().view(-1)
        return self

    def set_state_to_time_all(self):
        return self.set_state_to_time_t(self.time_inds.time_inds_materialized)

    def reset_state(self, make_new_state=False):
        if make_new_state:
            self.data = AnimatableData(self)
            self.data.spawn_time = self.animation_manager.context.get_current_time()
        self.data.data_dict_materialized = dict()
        self.data.time_inds_materialized = None
        self.data.time_inds_active = None
        self.data.data_dict = self.data.data_dict_active
        self.data.set_pre_function_application = False

    def set_state_pre_function_applications(self, spawn_ind, despawn_ind):
        """Sets all animatable attribute values to the values they had before any animated_function applications take place.
        """
        fps = self.scene.frames_per_second
        time_inds = torch.arange(spawn_ind, despawn_ind)
        self.despawn_ind = int(self.data.despawn_time() * fps)
        self.spawn_ind = int(self.data.spawn_time() * fps)
        self.despawn_ind = max(self.despawn_ind, self.spawn_ind+1)

        t = ((time_inds / fps))
        t = t.unsqueeze(-1)
        attr_to_values = dict()
        self.t = t

        attr_history, func_history = self.data.history.get_history(self)
        self.func_history = func_history
        for attr, new_values, end_times in attr_history:
            if hasattr(self.__class__, attr) and isinstance(getattr(self.__class__, attr), property):
                attr = f'{attr}'
            i = (t >= end_times).sum(-1, keepdim=True).clamp_max_(end_times.shape[-1]-1)
            attr_to_values[attr] = broadcast_gather(new_values, 0, i.unsqueeze(-1), keepdim=True)

        if 'opacity' not in attr_to_values:
            attr_to_values['opacity'] = self.opacity.expand(despawn_ind-spawn_ind,-1,-1).clone()
        attr_to_values['opacity'][:max(self.spawn_ind-(spawn_ind), 0)] = 0
        attr_to_values['opacity'][max(self.despawn_ind - spawn_ind, 0):] = 0
        self.data.data_dict_materialized = attr_to_values
        self.data.data_dict = self.data.data_dict_materialized
        self.data.time_inds_active = torch.arange(len(time_inds))
        self.data.time_inds_materialized = time_inds
        self.data.set_pre_function_application = True

    def set_state_full(self, s, e):
        """Sets all animatable attribute values to their final values after animated_functions have been applied.
        """
        if not self.data.set_pre_function_application:
            self.set_state_pre_function_applications(s, e)
        t = self.t
        animating_inds = [torch.zeros((1,), dtype=torch.long)]
        for func, animated_args, kwargs, start_times, end_times, rate_funcs in self.func_history:
            (func, caller) = func
            found = ((start_times < t) & (t < end_times)).type(t.dtype)
            if found.nonzero().numel() == 0:
                continue

            found_inds = found.sum(1).nonzero().squeeze(-1)
            animating_inds.append(found_inds)

            fa = (found[found_inds] + (torch.arange(found.shape[-1]) / (2*found.shape[-1]))).argmax(-1, keepdim=True)
            s, e = (broadcast_gather(_.unsqueeze(0), 1, fa, keepdim=True) for _ in (start_times, end_times))
            a = (t[found_inds] - s) / (e - s)

            a = a.unsqueeze(-2)
            ar = torch.stack(broadcast_all([rf(rfc(a)) if rfc is not None else rf(a) for rf, rfc in rate_funcs]), -1)
            z = broadcast_gather(ar, -1, unsqueeze_dims(fa, ar, -1), keepdim=False).clamp_(min=0, max=1)
            if self.parent_batch_sizes is not None and z.shape[1] == len(self.parent_batch_sizes):
                z = torch.repeat_interleave(z, self.parent_batch_sizes, 1)

            def select_kwargs(kwargs):
                return {key: broadcast_gather(value, 0, unsqueeze_dims(fa, value, 1), keepdim=True) if
                      isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}

            animated_args = select_kwargs(animated_args)
            caller.data.time_inds_active = torch.arange(len(caller.data.time_inds_materialized))[found_inds]
            kwargs2 = {key: (animated_args[key] * (1 - z) + z * value) if key in animated_args else value for key, value in select_kwargs(kwargs).items()}

            func(caller, **kwargs2)

        return self

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

        super().__setattr__('idle_gather_inds', idle_gather_inds)

        m = (all_inds.unsqueeze(-1) == self.idle_inds).sum(-1)
        ni = (all_inds.unsqueeze(-1) > self.non_idle_inds).sum(-1)
        ii = (all_inds.unsqueeze(-1) > self.idle_inds).sum(-1) + len(self.non_idle_inds)
        idle_scatter_inds = ni * (1 - m) + m * ii
        super().__setattr__('idle_scatter_inds', idle_scatter_inds)

    def getattr_at_time_t(self, attr, t):
        self.set_state_to_time_t(t)
        return self.__getattribute__(attr)

    def set_parent_to(self, other_mob):
        self.parents.append(other_mob)
        return self

    def add_children(self, *mobs):
        for mob in traverse(mobs):
            self.children.append(mob)
            mob.set_parent_to(self)
            self.anchor_priority = max(self.anchor_priority, 1 + mob.anchor_priority)
        return self

    def remove_child(self, mob):
        self.children.remove(mob)
        return self
