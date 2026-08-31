"""The Scene's mobs, as a tree a viewer can walk, with their animatable state.

Two things make this less obvious than listing ``scene.actors``:

* ``actors`` is flat and holds every Animatable ever constructed, including the
  component mobs a shape builds itself out of, and in construction order -- so a
  child can appear before its parent. The tree is rebuilt from the roots
  (``not mob.parents``) downwards instead. Parents are a *list*: the hierarchy is
  a graph, and a mob reached by two paths appears under both.
* The camera and the lights are not in ``actors`` at all, so they get their own
  sections rather than being silently missing.

Mob state does not live on the Mob. Every animatable attribute is a row in the
Scene's timeline, so reading one at a chosen time means materializing the whole
timeline at that time, reading, and putting it back -- which is what
:func:`attributes_at` does, and why the caller must hold the session lock.
"""

from __future__ import annotations

import contextlib
import math

import torch

from algan.animation_timeline.animation_contexts import Off
from algan.animation_timeline.timeline import WIDE_ATTR_MIN_CHANNELS
from algan.viewer.pixels import mob_label

#: Channel names for the attributes whose layout is fixed and worth naming, so
#: the panel shows ``opacity 1.0`` rather than the fifth number of a bare list.
CHANNEL_NAMES = {
    "location": ("x", "y", "z"),
    "color": ("r", "g", "b", "glow", "opacity"),
    "scale_coefficient": ("x", "y", "z"),
}

#: Never render more than this many numbers for one attribute. A basis is 9, a
#: colour 5; a packed mob's rows multiply that by its member count.
MAX_VALUES = 64


def node_id(mob) -> int:
    """The viewer's identity for a mob.

    Deliberately ``id(mob)`` and not ``mob.id``: the latter is the timeline row
    key, and a packed mob's ``mob[i]`` views and its clones share it, so two
    different tree nodes would collide.
    """
    return id(mob)


def _describe(mob, *, kind="mob"):
    children = _children_of(mob)
    lifespan = getattr(mob, "lifespan", None)
    start = end = None
    if lifespan is not None:
        # Both ends are callables, not numbers: a context rescales its block
        # retroactively, so a timestamp is only resolved when asked for.
        try:
            start, end = float(lifespan.start()), float(lifespan.end())
        except Exception:
            start = end = None
    return {
        "node": node_id(mob),
        "label": mob_label(mob),
        "type": type(mob).__name__,
        "kind": kind,
        "spawned": bool(_is_spawned(mob)),
        "lifespan": [start, end],
        "has_children": bool(children),
        "num_children": len(children),
    }


def _is_spawned(mob):
    for name in ("is_spawned_in_subtree", "is_spawned"):
        probe = getattr(mob, name, None)
        if probe is None:
            continue
        try:
            return bool(probe())
        except Exception:
            continue
    return False


def _children_of(mob, include_components=False):
    """A mob's children, components hidden unless asked for.

    Components are the sub-parts a shape builds itself from -- a circuit's
    control points, a surface's vertex grid. They are real children and real
    actors, and showing them by default buries the two mobs a user wrote.
    """
    if include_components:
        return list(getattr(mob, "children", ()) or ())
    getter = getattr(mob, "get_non_component_children", None)
    if getter is None:
        return list(getattr(mob, "children", ()) or ())
    try:
        return list(getter())
    except Exception:
        return list(getattr(mob, "children", ()) or ())


def roots(scene):
    """The Scene's top-level nodes: its root mobs, its camera, its lights."""
    out = []
    for actor in scene.actors:
        if not getattr(actor, "parents", None):
            out.append(_describe(actor))
    camera = getattr(scene, "camera", None)
    if camera is not None:
        out.append(_describe(camera, kind="camera"))
    for light in getattr(scene, "light_sources", ()) or ():
        out.append(_describe(light, kind="light"))
    return out


def index(scene):
    """Every node the tree can reach, keyed by :func:`node_id`.

    Built once per request rather than cached: a script may keep authoring while
    the viewer is open, and a stale index would hide the new mobs.
    """
    found = {}
    stack = list(scene.actors)
    camera = getattr(scene, "camera", None)
    if camera is not None:
        stack.append(camera)
    stack.extend(getattr(scene, "light_sources", ()) or ())
    while stack:
        mob = stack.pop()
        key = node_id(mob)
        if key in found:
            continue
        found[key] = mob
        stack.extend(getattr(mob, "children", ()) or ())
    return found


def children(mob, include_components=False):
    """The child rows under one node."""
    return [_describe(child) for child in _children_of(mob, include_components)]


def mob_by_timeline_id(scene):
    """``Mob.id`` -> mob, for naming the surface behind a rendered fragment.

    ``Mob.id`` is not unique across views and clones, so the first actor holding
    an id wins; that is the one that owns the timeline rows the render read.
    """
    out = {}
    for actor in scene.actors:
        out.setdefault(getattr(actor, "id", None), actor)
    return out


@contextlib.contextmanager
def materialized(scene, time_seconds):
    """Hold the Scene's timeline at ``time_seconds`` for the block, then undo it.

    Materializing points every attribute's storage at a buffer of values for the
    requested times, which is what makes an ordinary ``mob.location`` read
    return the frame's value instead of the authored one. It is global mutation
    of the Scene, so the caller must hold the viewer's lock, and it must be
    undone however the block exits.

    ``time_seconds`` of ``None`` is a no-op, leaving the authoring state in
    place -- the scene as the script last left it.
    """
    if time_seconds is None:
        yield
        return
    times = torch.tensor([float(time_seconds)], dtype=torch.float32)
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        scene.timeline_manager.set_state_to_times(times)
        try:
            yield
        finally:
            scene.timeline_manager.clear_buffers()


def attributes_of(scene, mob):
    """A mob's animatable attributes, read from the timeline as it stands.

    Wrap the call in :func:`materialized` to read them at a chosen time instead.
    """
    manager = scene.timeline_manager
    names = list(dict.fromkeys(getattr(mob, "animatable_attrs", ()) or ()))
    return [_read(manager, mob, name) for name in names]


def attributes_at(scene, mob, time_seconds=None):
    """A mob's animatable attributes, optionally as of ``time_seconds``.

    Convenience for the single-mob case; reading several mobs at one time should
    share a single :func:`materialized` block rather than repeat it per mob.
    """
    with materialized(scene, time_seconds):
        return attributes_of(scene, mob)


def _read(manager, mob, name):
    """One attribute row: its value if it has one, or why it has none."""
    timeline = manager.attr_to_timeline.get(name)
    row = {"name": name, "value": None, "channels": None, "note": None}
    if timeline is None or getattr(mob, "id", None) not in timeline.mob_id_to_inds:
        # Registered on the class but owning no timeline rows: a derived
        # property such as ``scale_coefficient``, which is read back out of
        # ``basis`` rather than stored. Shown, so its absence is explained.
        row["note"] = "derived"
        return row
    width = int(timeline.current_state.shape[-1])
    if width >= WIDE_ATTR_MIN_CHANNELS:
        # A Surface's colour texture is one row millions of channels wide.
        # Reading it to print it would pull the whole image off the render
        # device for a panel that cannot show it.
        row["note"] = f"{width} channels (too wide to display)"
        return row
    try:
        value = mob.get_animated_attribute(name, include_descendants=False, copy=True)
    except AttributeError:
        # Registered but never assigned -- the timeline has no rows to hand back.
        row["note"] = "unset"
        return row
    except Exception as exc:  # noqa: BLE001
        row["note"] = f"unreadable ({type(exc).__name__})"
        return row
    flat = value.detach().reshape(-1).tolist()
    row["shape"] = list(value.shape)
    if len(flat) > MAX_VALUES:
        row["value"] = flat[:MAX_VALUES]
        row["note"] = f"first {MAX_VALUES} of {len(flat)} values"
    else:
        row["value"] = flat
    names = CHANNEL_NAMES.get(name)
    if names and len(flat) == len(names):
        row["channels"] = list(names)
    return row
