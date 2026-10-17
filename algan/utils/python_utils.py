"""Small pure-Python helpers with no torch dependency.

``traverse`` flattens arbitrarily nested iterables into a flat sequence, which is
what lets Algan's constructors accept ``Group([a, [b, c], d])`` as readily as a
flat list. ``downsample_nested_list`` thins such a structure while preserving its
shape, and ``get_factors`` returns integer factor pairs, used when choosing grid
dimensions.

``callable_signature`` reads a function's parameters the way
``inspect.signature`` does, without evaluating its annotations.

``arithmetic_operators``, ``binary_operators`` and ``other_operators`` name the
dunder methods Algan forwards when it makes a wrapper behave like the value it
wraps.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from math import isqrt

import torch

#: Scalar leaves. Strings and bytes stay whole: recursively flattening a
#: one-character string would recurse forever instead of reaching a leaf.
_ATOMIC_TYPES = frozenset({float, int, bool, type(None), str, bytes})


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
        isinstance(nested_iterable, (str, bytes, torch.Tensor))
        or (hasattr(nested_iterable, "traversable") and not nested_iterable.traversable)
        or not isinstance(nested_iterable, Iterable)
    ):
        yield nested_iterable
    else:
        for _ in nested_iterable:
            yield from traverse(_)


def callable_signature(function, *, follow_wrapped=True):
    """``inspect.signature(function)``, without evaluating its annotations.

    Algan reads a callback's signature to check how it can be called, which
    its annotations play no part in. From Python 3.14 annotations are
    evaluated when they are read, so ``inspect.signature`` raised
    ``NameError`` for one naming a type imported only under
    ``TYPE_CHECKING``, though calling the function never evaluates it. Such
    an annotation now reads as a forward reference. Raises ``TypeError`` or
    ``ValueError`` where ``inspect.signature`` does, for a callable with no
    signature Python can read.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        return inspect.signature(
            function,
            follow_wrapped=follow_wrapped,
            annotation_format=annotationlib.Format.FORWARDREF,
        )
    return inspect.signature(function, follow_wrapped=follow_wrapped)


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
