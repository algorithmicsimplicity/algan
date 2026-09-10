"""Region-checked packed arguments, without allocation-wide alias promises.

Cold views retain one binding per scalar dtype. A hull can include unrelated
mutable bytes: never mark it readonly or noalias. Only layout is hoisted.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from algan.rendering.arena_regions import TypedArenaView, validate_regions

_TABLES = OrderedDict()
_MAX_TABLES = 64


def pack_regions(spec, tensors):
    """Narrow bases and export exact int32 offsets/shapes from checked regions."""
    from algan.rendering.raytracing.arena_args_taichi import DTYPE_TAGS

    if len(spec) != len(tensors) or not spec:
        raise ValueError("Region pack requires a nonempty matching binding specification")
    dtypes = {tag: dtype for dtype, tag in DTYPE_TAGS}
    views = tuple(TypedArenaView.bind(t, name=name, dtype=dtypes[tag], ndim=ndim)
                  for (name, tag, ndim), t in zip(spec, tensors))
    validate_regions(views)
    if len({v.tensor.device for v in views}) != 1:
        raise ValueError("Region bindings must share one device")
    tags = tuple(tag for _, tag in DTYPE_TAGS if any(s[1] == tag for s in spec))
    groups = {tag: [v for s, v in zip(spec, views) if s[1] == tag] for tag in tags}
    arenas, origins = [], {}
    for tag, group in groups.items():
        if len({v.layout.storage for v in group}) != 1:
            raise ValueError(f"{tag}: region arguments must share one storage")
        begin = min(v.layout.byte_offset for v in group)
        end = max(v.layout.byte_end for v in group)
        sample = group[0].tensor
        size = sample.element_size()
        arena = torch.empty(0, dtype=sample.dtype, device=sample.device)
        arena.set_(sample.untyped_storage(), begin // size, ((end - begin) // size,))
        arenas.append(arena)
        origins[tag] = begin
    key = (tuple(spec), tuple(v.layout for v in views))
    table = _TABLES.get(key)
    if table is None:
        offsets = [(v.layout.byte_offset - origins[s[1]]) // v.tensor.element_size()
                   for s, v in zip(spec, views)]
        shapes = [dim for v in views for dim in v.layout.shape]
        table = torch.tensor(offsets + shapes, dtype=torch.int32, device=arenas[0].device)
        _TABLES[key] = table
        if len(_TABLES) > _MAX_TABLES:
            _TABLES.popitem(last=False)
    else:
        _TABLES.move_to_end(key)
    return (*arenas, table[:len(spec)], table[len(spec):])


def checked_launch_bindings(call_params, args, access):
    """Include ordinary hot arguments in the complete launch alias proof."""
    views = []
    for name, value in zip(call_params, args):
        if isinstance(value, torch.Tensor):
            if name not in access:
                raise ValueError(f"No region access contract for {name}")
            views.append(TypedArenaView.bind(value, name=name, access=access[name]))
    return tuple(views)


def clear_region_pack_cache():
    _TABLES.clear()
