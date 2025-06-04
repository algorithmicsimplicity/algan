from collections.abc import Iterable
from math import isqrt

import torch


def traverse(nested_iterable):
    if isinstance(nested_iterable, torch.Tensor) or (hasattr(nested_iterable, 'traversable') and not nested_iterable.traversable) or not isinstance(nested_iterable, Iterable):
        yield nested_iterable
    else:
        for _ in nested_iterable:
            yield from traverse(_)


binary_operators = ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'or', 'xor']
other_operators = ['neg', 'pos', 'abs', 'invert', 'lt', 'le', 'eq', 'ne', 'gt', 'ge']
arithmetic_operators = [f'__{_}__' for _ in binary_operators + other_operators]


def downsample_nested_list(lists, factor=2):
    out = []
    for i in range(0, len(lists), factor):
        out.append([lists[i][j] for j in range(0,len(lists[i]), factor)])
    return out


def get_factors(x):
    return [i for i in range(1, isqrt(x)) if (x % i) == 0]
