import functools
import inspect
from collections import defaultdict
from math import sqrt
from string import ascii_lowercase
from typing import List, Optional
import re

import torch
import torch.nn.functional as F
import torch.nn as nn

from algan.defaults.device_defaults import DEFAULT_DEVICE

try:
    from torch_scatter import scatter_max as scatter_max_op
except ModuleNotFoundError:
    scatter_max_op = None


def packed_reorder(x, counts, ids):
    #packed_counts = scatter_add(counts, ids, 0)
    cc = counts.cumsum(0)
    cc = torch.cat((torch.zeros_like(cc[:1]), cc))
    id_to_chunks = defaultdict(list)
    for i in range(len(cc)-1):
        j = ids[i].item()
        if cc[i]==cc[i+1]:
            continue
            raise NotImplementedError('packing 0 size chunks')
        id_to_chunks[j].append(x[cc[i]:cc[i+1]])
        #outs[cc[i]:cc[i+1]] = x[]
    chunks = [torch.cat(id_to_chunks[k]) for k in sorted(id_to_chunks.keys())]
    return torch.cat(chunks), torch.tensor([len(_) for _ in chunks])
    ids_rep = torch.repeat_interleave(ids, counts, 0)
    s = ids.argsort(0)
    cum_counts = counts[s]
    cum_counts = (cum_counts.cumsum(0) - cum_counts[0])[ids]
    cc_rep = torch.repeat_interleave(cum_counts, counts, 0)
    s = ids_rep.argsort(0)
    scatter_inds = cc_rep + (torch.arange(len(s)) - cc_rep[s])[ids_rep]
    return ids

def _cast_to_tensor_recursive(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return torch.stack([_cast_to_tensor_recursive(_) for _ in x], 0)
    else:
        return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1)

def cast_to_tensor(x):
    """
    Converts scalars or lists of scalars into tensors, and combines lists of tensors into a single tensor.
    All other input types are returned unchanged.
    Returned tensors are always of shape [1,N,D] where D is dimension and N is the number of tensors combined.
    """
    if isinstance(x, torch.Tensor):
        while x.dim() < 3:
            x = x.unsqueeze(0)
        return x#.view(-1,x.shape[-1])
    if isinstance(x, list) or isinstance(x, tuple):
        x = _cast_to_tensor_recursive(x).squeeze(-1)
        return x.view(1,-1,x.shape[-1])
    if x is None:
        x = 0
    try:
        return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1,1,1)
    except:
        return x


def cast_to_tensor_single(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return torch.stack(x, -2)#[cast_to_tensor(_) for _ in x]
    if isinstance(x, torch.Tensor):
        return x#.view(-1,x.shape[-1])
    if x is None:
        return torch.tensor((0,), dtype=torch.get_default_dtype()).view(1)
    return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1)


def make_grid(height, width=None, min_coord=-1, max_coord=1, min_coord2=None, max_coord2=None):
    if width is None:
        width = height
    if min_coord2 is None:
        min_coord2 = min_coord
    if max_coord2 is None:
        max_coord2 = max_coord

    return torch.stack((torch.zeros((height, width), device=DEFAULT_DEVICE),
           torch.linspace(min_coord, max_coord, height, device=DEFAULT_DEVICE).view(-1,  1).expand(-1, width),
           torch.linspace(min_coord2, max_coord2, width, device=DEFAULT_DEVICE).view(1, -1).expand(height, -1)), -1)


def add_dummy_dims_right(x, y):
    return x.view(list(x.shape) + [1] * (len(y.shape)))


def add_dummy_dims_left(x, y):
    return x.view([1] * (len(y.shape)) + list(x.shape))


def unsqueeze_right(x, y, offset:int=0):
    if not isinstance(x, torch.Tensor):
        return x
    return x.view([1] * offset + list(x.shape) + [1] * (len(y.shape) - (len(x.shape) + offset)))


def unsqueeze_left(x, y, offset:int=0):
    if not isinstance(x, torch.Tensor):
        return x
    return x.view(*([1] * (len(y.shape) - (len(x.shape) + offset))), *([1] * offset + list(x.shape)))


def unsqueeze_dims(x, y, insert_dim=0):
    if not isinstance(x, torch.Tensor):
        return x
    while x.dim() < y.dim():
        x = x.unsqueeze(insert_dim)
    return x


def expand_as_left(x, y, offset:int=0):
    n = y.dim() - x.dim()
    x = unsqueeze_left(x, y)
    return x.expand([-1 if x.shape[i] != 1 else y.shape[i] for i in range(x.dim())])
    return x.expand(list(y.shape[:n]) + ([-1] * (x.dim()-n)))


def expand_as_right(x, y, offset:int=0):
    n = x.dim()
    x = unsqueeze_right(x, y)
    return x.expand(([-1] * (n)) + list(y.shape[n:]))


def broadcast(x, y, ignored_dims:List[int]):
    return x.expand([y.shape[i] if (x.shape[i] == 1 and i not in ignored_dims) else -1 for i in range(len(x.shape))])


def broadcast_both(x, y, ignored_dims:List[int]):
    x = unsqueeze_right(x, y)
    x = broadcast(x, y, ignored_dims=ignored_dims)
    y = broadcast(y, x, ignored_dims=ignored_dims)
    return x, y


def broadcast_both_left(x, y, ignored_dims:List[int]):
    x = unsqueeze_left(x, y)
    y = unsqueeze_left(y, x)
    x = broadcast(x, y, ignored_dims=ignored_dims)
    y = broadcast(y, x, ignored_dims=ignored_dims)
    return x, y

def unsqueeze_until_dim(x, dim, insert_dim=0):
    while x.dim() < dim:
        x = x.unsqueeze(insert_dim)
    return x

def broadcast_all(xs, ignored_dims=[]):
    max_dim = max([_.dim() if hasattr(_, 'dim') else 0 for _ in xs])
    ignored_dims = [_ if _ >= 0 else _ + max_dim for _ in ignored_dims]
    xs = [unsqueeze_until_dim(x, max_dim) if isinstance(x, torch.Tensor) else x for x in xs]
    max_shapes = [max([x.shape[i] if isinstance(x, torch.Tensor) else 0 for x in xs]) if i not in ignored_dims else -1 for i in range(max_dim)]
    return [x.expand(*max_shapes) if isinstance(x, torch.Tensor) else x for x in xs]

def broadcast_gather(src, dim:int, ind, keepdim=False, **kwargs):
    ind, src = broadcast_both_left(ind, src, ignored_dims=[dim if dim >= 0 else len(src.shape)+dim])
    out = torch.gather(src, dim, ind, **kwargs)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def broadcast_scatter(input, dim, ind, src, **kwargs):
    input, ind, src = broadcast_all([input, ind, src], ignored_dims=[dim if dim >= 0 else len(src.shape)+dim])
    try:
        return input.scatter_reduce(dim, ind, src, **kwargs)
    except:
        return input.scatter_reduce(dim, ind, src, **kwargs)


def offset(x):
    return torch.cat((x[1:], x[:1]))


def shuffle(x):
    dim=0
    perm = torch.randperm(x.shape[dim])
    return x[perm]


def squish(x, start:int = 0, end:int = 1):
    if end < 0:
        end = end + len(x.shape)
    return x.reshape(list(x.shape[:start]) + [-1] + list(x.shape[end+1:]))


def unsquish(x, dim: int = 0, factor:Optional[int] = None):
    if dim < 0:
        dim = len(x.shape) + dim
    if factor is None:
        factor = int(sqrt(x.shape[dim]))
    if factor < 0:
        new_d = -factor
        new_d2 = x.shape[dim] // new_d
    else:
        new_d = x.shape[dim] // factor
        new_d2 = factor
    return x.reshape(list(x.shape[:dim]) + [new_d, new_d2] + list(x.shape[dim+1:]))


def interpolate(x, y, a):
    return x * (1-a) + (a) * y


def pad_dim_left(x, num_dims):
    if x.dim() < num_dims:
        x = x.view(*([1] * (num_dims-x.dim())), *x.shape)
    return x


def pad_dim_right(x, num_dims):
    if x.dim() < num_dims:
        x = x.view(*x.shape, *([1] * (num_dims-x.dim())))
    return x


def broadcast_cross_product(x, y, dim=-1):
    x, y = broadcast_both_left(x, y, ignored_dims=[dim])#[dim if dim >= 0 else len(y.shape) + dim])
    return torch.cross(x, y, dim=dim)


def dot_product_in_place(x, y, dim=-1):
    x *= y
    return x.sum(dim, keepdim=True)


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def _get_empty_tensor_of_broadcasted_shape(x, y):
    if x.dim() < y.dim():
        x = unsqueeze_left(x, y)
    if y.dim() < x.dim():
        y = unsqueeze_left(y, x)
    return torch.empty([max(x_, y_) for x_, y_ in zip(x.shape[:-1], y.shape[:-1])] + [1], device=x.device)


def _dot_product_low_dim(x, y, out=None):
    if out is None:
        out = _get_empty_tensor_of_broadcasted_shape(x, y)
    out[:] = 0
    for i in range(x.shape[-1]):
        torch.addcmul(out, x[...,i].unsqueeze(-1), y[...,i].unsqueeze(-1), out=out)
    return out


def dot_product(x, y, dim=-1, keepdim=True, out=None):
    if dim==-1 or dim==(max(x.dim(), y.dim())-1):
        if x.shape[-1] <= 3:
            out = _dot_product_low_dim(x, y, out=out)
            if not keepdim:
                out = out.squeeze(-1)
            return out
        if out is not None:
            out = out.unsqueeze(-1)
        out = torch.matmul(x.unsqueeze(-2), y.unsqueeze(-1), out=out)
        out = out.squeeze(-1)
        if not keepdim:
            out = out.squeeze(-1)
        return out

    if not hasattr(dim, '__iter__'):
        dim = [dim]

    if x.dim() < y.dim():
        x = unsqueeze_left(x, y)
    if y.dim() < x.dim():
        y = unsqueeze_left(y, x)

    a = ascii_lowercase[:x.dim()]
    r = [a[i] for i in dim]
    ar = re.sub(f'[{"|".join(r)}]', '', a)
    out = torch.einsum(f'{a},{a}->{ar}', x, y)
    dim = [d if d >= 0 else d + max(x.dim(), y.dim()) for d in dim]
    if keepdim:
        for d in sorted(dim):
            out = out.unsqueeze(d)
    return out


def broadcast_interleave(x, counts, inds, dim=-2):
    x = broadcast_gather(x, dim, unsqueeze_right(inds.clamp_max(x.shape[dim] - 1), x, dim), keepdim=True)
    return torch.repeat_interleave(x, counts, dim)


def pad_dims(xs, unsqueeze_dim=-2):
    max_dim = max([_.dim() for _ in xs])
    outs = []
    for x in xs:
        while x.dim() < max_dim:
            x = x.unsqueeze(unsqueeze_dim)
        outs.append(x)
    return outs


def pack_tensor(x, packing):
    if packing is None:
        return x
    counts, num_dims = packing
    d = x.dim() - (num_dims+1)
    if (d < 0) or (x.shape[d] == 1):
        return x
    return torch.repeat_interleave(x, counts, d)


def unsqueeze_pack_tensors(xs, packing):
    if packing is None:
        return xs
    xs = pad_dims(xs)
    return [pack_tensor(x, packing) for x in xs]


def unpack_tensor(x, packing):
    if packing is None:
        return x
    counts, num_dims = packing
    d = x.dim() - (num_dims + 1)
    return torch.split(x, [_.item() for _ in counts], d)


def reduce_max_score(x, scores, dim=-1):
    max_score, max_score_ind = scores.max(dim, keepdim=True)
    x = broadcast_gather(x, dim, max_score_ind, keepdim=True)
    return x, max_score


def robust_concat(xs):
    """
    Concatenates multiple tensors together while broadcasting as necessary to ensure shapes match.
    """
    xs = [cast_to_tensor_single(x) for x in xs]
    max_dim = max([x.dim() for x in xs])

    def unsqueeze_left_to_max_dim(x):
        if len(x) == max_dim:
            return x
        return torch.cat((torch.ones((max_dim - len(x),), dtype=x.dtype), x))

    max_shape = torch.stack([unsqueeze_left_to_max_dim(torch.tensor(x.shape)) for x in xs]).amax(0)
    xs = [x.view(*([1] * (max_dim - x.dim())), *x.shape) for x in xs]
    return torch.cat([x.expand([_ if x.shape[i] == 1 else -1 for i, _ in enumerate(max_shape)]) for x in xs])


def concat_dicts(kwargs):
    """
    Concatenates a list of dicts sharing the same keys, the resulting dictionary has
    the same keys and concatenated values.
    """
    kwargs = [{k: cast_to_tensor(v) for k, v in d.items()} for d in kwargs]
    return {a: (robust_concat([kwargs[i][a] for i in range(len(kwargs))])
                if (isinstance(kwargs[0][a], torch.Tensor)) else kwargs[0][a]) for a in kwargs[0]}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


HANDLED_FUNCTIONS = {}


def prepare_kwargs(self, func, args, kwargs, initial_args, unique_args):
    """Combine args and kwargs into one dict, using default values where arg is missing
    """
    params = inspect.signature(func).parameters
    arg_names = list(params.keys())[1:]
    kwargs.update({arg_names[i]: args[i] for i in range(len(args))})
    default_kwargs = {param.name: param.default for param in params.values() if not (param.default is inspect._empty)}
    default_kwargs.update(kwargs)
    kwargs = {k: cast_to_tensor(v) if k in initial_args else v for k, v in default_kwargs.items()}
    # func_name needs to be a unique identifier, as all funcs with the same func_name will be put in the same batch.
    # This is why unique_args are part of the name.
    func_name = f'{func.__name__}_{"_".join([str(kwargs[a]) for a in unique_args])}_{id(self)}'
    self.data.history.insert_function_application(func_name, (func, self), initial_args, kwargs, self.animation_manager.context)
    return kwargs


def scatter_arg_max(x, inds, dim=-1, dim_size=None):
    if scatter_max_op is not None:
        return scatter_max_op(x, inds, -1, dim_size=dim_size)
    x = x.view(-1)
    out_dims = [*x.shape]
    out_dims[dim] = dim_size if dim_size is not None else inds.amax()
    out = torch.zeros(out_dims, device=x.device)
    max_vals = torch.scatter_reduce(out, dim, inds, x, 'amax', include_self=False)
    max_vals_gathered = broadcast_gather(max_vals, dim, inds)
    m = (x == max_vals_gathered)
    inds[~m] = 0
    prev_inds = inds.cummax(-1)[0].roll(1,-1)
    prev_inds[...,0] = -1
    argmax_inds = ((inds != prev_inds) & m).nonzero()
    max_vals = broadcast_gather(x, -1, argmax_inds)
    return max_vals, argmax_inds
