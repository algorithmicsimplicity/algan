"""Small pure-Python helpers with no torch dependency.

``traverse`` flattens arbitrarily nested iterables into a flat sequence, which is
what lets Algan's constructors accept ``Group([a, [b, c], d])`` as readily as a
flat list. ``downsample_nested_list`` thins such a structure while preserving its
shape, and ``get_factors`` returns integer factor pairs, used when choosing grid
dimensions.

``arithmetic_operators``, ``binary_operators`` and ``other_operators`` name the
dunder methods Algan forwards when it makes a wrapper behave like the value it
wraps.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isqrt

import torch

#: Leaf types that are provably not tensors, carry no ``traversable`` flag and
#: are not iterable, so :func:`traverse` can yield them without asking.
_ATOMIC_TYPES = frozenset({float, int, bool, type(None)})


def traverse(nested_iterable):
    # Dispatch on the exact type first. Authoring a scene calls this millions
    # of times, and the general test below costs two attribute probes plus an
    # abstract-base ``isinstance`` (which goes through ABCMeta) per node. A
    # plain list or tuple is never a tensor, never carries ``traversable`` and
    # is always iterable, so it can recurse immediately; the scalar types are
    # the reverse and can be yielded immediately. Subclasses miss both fast
    # paths and take the general one, so behaviour is unchanged.
    node_type = type(nested_iterable)
    if node_type is list or node_type is tuple:
        for _ in nested_iterable:
            yield from traverse(_)
        return
    if node_type in _ATOMIC_TYPES:
        yield nested_iterable
        return
    if (
        isinstance(nested_iterable, torch.Tensor)
        or (hasattr(nested_iterable, "traversable") and not nested_iterable.traversable)
        or not isinstance(nested_iterable, Iterable)
    ):
        yield nested_iterable
    else:
        for _ in nested_iterable:
            yield from traverse(_)


binary_operators = [
    "add",
    "sub",
    "mul",
    "matmul",
    "truediv",
    "floordiv",
    "mod",
    "divmod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "or",
    "xor",
]
other_operators = ["neg", "pos", "abs", "invert", "lt", "le", "eq", "ne", "gt", "ge"]
arithmetic_operators = [f"__{_}__" for _ in binary_operators + other_operators]


def downsample_nested_list(lists, factor=2):
    out = []
    for i in range(0, len(lists), factor):
        out.append([lists[i][j] for j in range(0, len(lists[i]), factor)])
    return out


def get_factors(x):
    return [i for i in range(1, isqrt(x)) if (x % i) == 0]
