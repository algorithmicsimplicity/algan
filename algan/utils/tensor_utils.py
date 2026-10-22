"""Broadcasting, reshaping and interpolation helpers for batched tensors.

Algan's animation state is uniformly shaped ``[time, mob, ...]``, and almost
every operation has to line up operands that disagree about how many batch
dimensions they carry. This module is the vocabulary for doing that:
``broadcast*`` for aligning operands, ``unsqueeze_*`` / ``add_dummy_dims_*`` for
inserting axes on either side, ``squish`` / ``unsquish`` for flattening and
restoring axis pairs, and ``cast_to_tensor`` for accepting the Python lists,
floats and tuples a user naturally writes.

``interpolate`` is the blend used when materializing an animated attribute
between two recorded states.

These are internal helpers and deliberately excluded from
``from algan import *``, whose names are chosen not to collide with a user's own.
"""

from __future__ import annotations

import functools
import math
import re
from collections import defaultdict
from math import sqrt
from string import ascii_lowercase

import numpy as np
import torch

from algan.errors import AlganConfigurationError
from algan.settings._startup import _ANIMATION_DEVICE


def texture_u8_provenance(texture):
    """Whether a ``[..., 5]`` color texture is exactly 8-bit with zero glow.

    True iff every channel value is exactly ``k / 255`` for an integer ``k``
    in ``[0, 255]`` (the values :func:`~algan.utils.file_utils.get_image`
    produces from an 8-bit file) AND the glow channel (index 3) is
    identically zero. Proved by the round trip itself --
    ``round(x * 255) / 255 == x`` elementwise in f32 -- so a map this
    accepts quantizes to u8 and back bit-identically, which is the admission
    rule for texture_u8_storage's packed color-map layout.

    Checked ONCE at authoring (the ``color_texture`` setter, a mesh
    constructor) rather than at the merge: the merge runs on the prefetch
    worker, where an elementwise probe's device sync waits out the whole
    queued chunk. NaNs fail the round trip, so they fail the check.
    """
    tex = texture.detach().float()
    if tex.shape[-1] < 5:
        return False
    q = torch.round(tex * 255.0)
    return bool(
        torch.equal(q.clamp(0.0, 255.0) / 255.0, tex)
        and not bool((tex[..., 3] != 0).any())
    )


def packed_reorder(x, counts, ids):
    cc = counts.cumsum(0)
    cc = torch.cat((torch.zeros_like(cc[:1]), cc))
    id_to_chunks = defaultdict(list)
    for i in range(len(cc) - 1):
        if cc[i] == cc[i + 1]:
            continue
        id_to_chunks[ids[i].item()].append(x[cc[i] : cc[i + 1]])
    chunks = [torch.cat(id_to_chunks[k]) for k in sorted(id_to_chunks.keys())]
    return torch.cat(chunks), torch.tensor([len(_) for _ in chunks])


def _cast_to_tensor_recursive(x):
    if isinstance(x, (list, tuple)):
        return torch.stack([_cast_to_tensor_recursive(_) for _ in x], 0)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        # A vector inside a list -- [ORIGIN, RIGHT, UP] -- stands where a list
        # of its numbers would, each of which is cast to shape [1].
        return x.to(dtype=torch.get_default_dtype()).unsqueeze(-1)
    return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1)


#: ``Tensor.__torch_function__``'s implementation: a subclass that keeps it
#: (``Color``) only re-types each op's result, so an op on its plain view
#: followed by the same re-typing is the same op.
_DEFAULT_TORCH_FUNCTION = torch.Tensor.__torch_function__.__func__


def _with_leading_dims(x):
    """``x`` with size-1 dimensions prepended until it has three.

    One ``x[None, ...]`` index rather than one ``unsqueeze(0)`` per missing
    dimension: indexing inserts each dimension with ``unsqueeze``, so the
    shape, strides and storage are the ones the loop produced. A ``Color``
    is indexed as its plain view and re-typed once, which is what its
    ``__torch_function__`` does to every op's result anyway, rather than
    paying that dispatch once per dimension.
    """
    cls = type(x)
    if cls is torch.Tensor:
        missing = 3 - x.dim()
        return x[(None,) * missing] if missing > 0 else x
    if getattr(cls.__torch_function__, "__func__", None) is _DEFAULT_TORCH_FUNCTION:
        plain = x.as_subclass(torch.Tensor)
        missing = 3 - plain.dim()
        if missing <= 0:
            return x
        return plain[(None,) * missing].as_subclass(cls)
    while x.dim() < 3:
        x = x.unsqueeze(0)
    return x


def cast_to_tensor(x):
    """
    Converts scalars or lists of scalars into tensors, and combines lists of tensors into a single tensor.
    All other input types are returned unchanged.
    Returned tensors are always of shape [1,N,D] where D is dimension and N is the number of tensors combined.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    if isinstance(x, torch.Tensor):
        return _with_leading_dims(x)
    if type(x) is float:
        dtype = torch.get_default_dtype()
        if abs(x) <= torch.finfo(dtype).max:
            # The common scalar, in one op. ``full`` converts a float fill
            # value exactly as ``tensor`` converts a float element (a C cast
            # of the double), so this is the tensor the general path below
            # builds. ``full`` refuses a value beyond the dtype's range where
            # ``tensor`` rounds it to an infinity, so those -- and NaN -- take
            # the general path.
            return torch.full((1, 1, 1), x, dtype=dtype)
    if isinstance(x, (list, tuple)):
        if x and all(type(item) is float or type(item) is int for item in x):
            # A flat run of numbers: every element is stored by the same
            # per-element conversion the recursive path applies one at a
            # time, so one ``tensor`` call builds the same values.
            x = torch.tensor(list(x), dtype=torch.get_default_dtype())
        else:
            x = _cast_to_tensor_recursive(x).squeeze(-1)
        return x.view(1, -1, x.shape[-1])
    if x is None:
        x = 0
    try:
        return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1, 1, 1)
    except (TypeError, ValueError, RuntimeError):
        # Non-numeric values (strings, arbitrary objects) pass through
        # unchanged by design; anything else should surface.
        return x


#: Sentinel for "the caller did not say what ``value`` was cast from".
_NO_SOURCE = object()


def _plain_tensor(value):
    """``value`` viewed as a plain :class:`torch.Tensor`.

    A :class:`~algan.constants.color.Color` routes every op through
    ``Tensor.__torch_function__``, which costs several times the op itself on
    the tiny tensors validation sees. The view shares the data, so a check on
    it reads exactly the same values.
    """
    return value if type(value) is torch.Tensor else value.as_subclass(torch.Tensor)


def _number_casts_to_finite(source):
    """Whether ``source`` is plain Python numbers whose cast is finite.

    True only for an ``int`` or ``float`` -- or a flat, non-empty tuple or list
    of them, which ``cast_to_tensor`` casts element by element -- (exactly
    those types, so ``bool``, NumPy scalars, tensors and nested sequences take
    the tensor check) whose magnitudes fit the default floating dtype:
    ``cast_to_tensor`` builds that dtype, and a finite float beyond its range
    -- ``1e39`` in float32 -- casts to an infinity, which must still be
    rejected. NaN and the infinities fail the comparison.
    """
    kind = type(source)
    if kind is tuple or kind is list:
        if not source:
            return False
        limit = torch.finfo(torch.get_default_dtype()).max
        for item in source:
            kind = type(item)
            if (kind is not float and kind is not int) or not abs(item) <= limit:
                return False
        return True
    if kind is not float and kind is not int:
        return False
    return abs(source) <= torch.finfo(torch.get_default_dtype()).max


def _all_finite(value):
    """Whether every element of the floating tensor ``value`` is finite.

    One reduction and one scalar read on the path every valid value takes: a
    sum is finite only if every term is (a NaN or an infinity anywhere makes
    it NaN or infinite). The converse can fail -- finite values large enough
    to overflow the sum -- so a non-finite sum falls back to the exact
    elementwise test before anything is reported.
    """
    value = _plain_tensor(value)
    if math.isfinite(value.sum().item()):
        return True
    return bool(torch.isfinite(value).all())


def reject_non_finite(name, value, given=None, *, cast_from=_NO_SOURCE):
    """Raise if ``value`` holds a NaN or an infinity.

    A NaN reaching an animatable attribute is not caught anywhere downstream:
    it propagates through the basis, the projection and the rasterizer, and
    the symptom is a black frame (or a mob that silently disappears) with
    nothing in it pointing back at the line that wrote it. Checked once, where
    the value is authored, rather than per replayed frame.

    Parameters
    ----------
    name
        The user-facing parameter or attribute name, used in the message.
    value
        The already-cast tensor to check. Non-tensors pass through.
    given
        What the user actually wrote, when it reads better than the tensor.
    cast_from
        What ``value`` was cast from with :func:`cast_to_tensor`, when the
        caller knows. A plain Python number that casts to a finite value
        needs no look at the tensor, which is most of what authoring writes;
        anything else is checked on ``value`` as usual.

    Returns
    -------
    The value, unchanged, so this can wrap a cast.

    Raises
    ------
    AlganConfigurationError
        If any element is NaN or infinite.
    """
    if cast_from is not _NO_SOURCE and _number_casts_to_finite(cast_from):
        return value
    if torch.is_tensor(value):
        plain = _plain_tensor(value)
        if plain.is_floating_point() and not _all_finite(plain):
            what = "NaN" if bool(torch.isnan(plain).any()) else "infinite"
            shown = value if given is None else given
            raise AlganConfigurationError(
                f"{name} must be finite; got a {what} value: {shown!r}. A "
                f"non-finite value propagates through every later frame and "
                f"renders as a blank or missing shape rather than raising."
            )
    return value


def _within_unit_interval(value):
    """Whether the floating tensor ``value`` is finite and inside ``[0, 1]``.

    The one-reduction answer for the common, valid case. ``aminmax``
    propagates a NaN into both ends, and an infinity is outside the interval
    anyway, so a ``True`` here needs no separate finiteness check. ``False``
    says only "look closer": the caller then runs the exact checks, which
    decide which error -- if any -- to raise.

    The lower end must be strictly positive. With denormals flushed (a thread
    whose floating-point mode some library set), the reduction can return a
    negative denormal as ``-0.0`` while the exact comparison still sees it
    below zero, so a zero minimum is left to that comparison.
    """
    value = _plain_tensor(value)
    if not value.is_floating_point() or value.numel() == 0:
        return False
    # Both ends in one host read.
    low, high = torch.stack(torch.aminmax(value)).tolist()
    return low > 0.0 and high <= 1.0


def reject_outside_unit_interval(name, value, given=None, *, cast_from=_NO_SOURCE):
    """Raise if ``value`` falls outside the closed unit interval ``[0, 1]``.

    For the attributes documented as fractions -- opacity above all -- where a
    value outside the interval has no meaning the renderer could honour. It is
    not clamped silently: ``opacity=1.5`` and ``opacity=-0.2`` are authoring
    mistakes, and the frame they produce looks like a bug in Algan rather than
    a bug in the script.

    Only for values that are *set*. A value that is *scaled* stays unchecked on
    purpose, on the same reasoning that leaves ``scale(0)`` and a negative
    scale legal: a multiply is a transform, and the interval is a property of
    what the attribute means, not of every arithmetic step towards it.

    Non-finite values are rejected first, by :func:`reject_non_finite`, so a
    NaN reports as a NaN rather than as an out-of-range value.

    Parameters
    ----------
    name
        The user-facing parameter or attribute name, used in the message.
    value
        The value to check. A tensor is checked element-wise; a Python number
        is checked directly, so the plain ``set_opacity(1.5)`` spelling is
        caught as well as the tensor one. Anything else passes through.
    given
        What the user actually wrote, when it reads better than the tensor.
    cast_from
        What ``value`` was cast from, as for :func:`reject_non_finite`: a plain
        Python number inside the interval needs no look at the tensor.

    Returns
    -------
    The value, unchanged, so this can wrap a cast.

    Raises
    ------
    AlganConfigurationError
        If any element is NaN, infinite, below 0 or above 1.
    """
    if cast_from is not _NO_SOURCE:
        kind = type(cast_from)
        if (kind is float or kind is int) and 0.0 <= cast_from <= 1.0:
            # Rounding to any floating dtype keeps a value of [0, 1] inside it.
            return value
    if torch.is_tensor(value) and _within_unit_interval(value):
        return value
    # Anything the fast answers above could not clear takes the exact checks,
    # which decide which error -- if any -- applies.
    reject_non_finite(name, value, given=given)
    if torch.is_tensor(value):
        outside = bool(((value < 0) | (value > 1)).any())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # Python numbers never reach reject_non_finite's tensor guard, so a
        # bare NaN lands here; `not (0 <= nan <= 1)` is True, which is the
        # answer we want anyway.
        outside = not 0.0 <= float(value) <= 1.0
    else:
        outside = False
    if outside:
        shown = value if given is None else given
        raise AlganConfigurationError(
            f"{name} must be between 0 and 1 inclusive; got {shown!r}. It is a "
            f"fraction, so a value outside that range has no rendered meaning; "
            f"it is rejected at the line that wrote it rather than clamped "
            f"silently at materialization."
        )
    return value


def cast_to_direction(name, x):
    """Cast a user-supplied 3-D vector argument to a ``[1, N, 3]`` tensor.

    Like :func:`cast_to_tensor`, but rejects anything whose last dimension is
    not 3. A bare scalar casts to shape ``[1, 1, 1]``, which broadcasts across
    all three axes rather than failing, so without this ``mob.move(1)`` moves
    diagonally to ``(1, 1, 1)`` instead of telling the user it wanted a vector.

    Parameters
    ----------
    name
        The user-facing parameter name, used in the error message.
    x
        The value to cast.

    Raises
    ------
    AlganConfigurationError
        If ``x`` is not a 3-D vector.
    """
    value = cast_to_tensor(x)
    if not isinstance(value, torch.Tensor) or value.shape[-1] != 3:
        raise AlganConfigurationError(
            f"{name} must be a 3-D vector of shape (*, 3), such as RIGHT, "
            f"UP * 2 or [1, 0, 0]; got {x!r}"
        )
    return reject_non_finite(name, value, given=x)


def make_grid(
    height, width=None, min_coord=-1, max_coord=1, min_coord2=None, max_coord2=None
):
    if width is None:
        width = height
    if min_coord2 is None:
        min_coord2 = min_coord
    if max_coord2 is None:
        max_coord2 = max_coord

    device = _ANIMATION_DEVICE
    return torch.stack(
        (
            torch.zeros((height, width), device=device),
            torch.linspace(min_coord, max_coord, height, device=device)
            .view(-1, 1)
            .expand(torch.Size([-1, width])),
            torch.linspace(min_coord2, max_coord2, width, device=device)
            .view(1, -1)
            .expand(torch.Size([height, -1])),
        ),
        -1,
    )


def unsqueeze_right(x, y, offset: int = 0):
    if not isinstance(x, torch.Tensor):
        return x
    return x.view(
        [1] * offset + list(x.shape) + [1] * (len(y.shape) - (len(x.shape) + offset))
    )


def unsqueeze_left(x, y, offset: int = 0):
    if not isinstance(x, torch.Tensor):
        return x
    return x.view(
        *([1] * (len(y.shape) - (len(x.shape) + offset))),
        *([1] * offset + list(x.shape)),
    )


def broadcast(x, y, ignored_dims: list[int]):
    return x.expand(
        [
            y.shape[i] if (x.shape[i] == 1 and i not in ignored_dims) else -1
            for i in range(len(x.shape))
        ]
    )


def broadcast_both_left(x, y, ignored_dims: list[int]):
    x = unsqueeze_left(x, y)
    y = unsqueeze_left(y, x)
    x = broadcast(x, y, ignored_dims=ignored_dims)
    y = broadcast(y, x, ignored_dims=ignored_dims)
    return x, y


def unsqueeze_until_dim(x, dim, insert_dim=0):
    while x.dim() < dim:
        x = x.unsqueeze(insert_dim)
    return x


def broadcast_all(xs, ignored_dims=()):
    max_dim = max([_.dim() if hasattr(_, "dim") else 0 for _ in xs])
    ignored_dims = [_ if _ >= 0 else _ + max_dim for _ in ignored_dims]
    xs = [
        unsqueeze_until_dim(x, max_dim) if isinstance(x, torch.Tensor) else x
        for x in xs
    ]
    max_shapes = [
        max([x.shape[i] if isinstance(x, torch.Tensor) else 0 for x in xs])
        if i not in ignored_dims
        else -1
        for i in range(max_dim)
    ]
    return [
        x.expand(torch.Size(max_shapes)) if isinstance(x, torch.Tensor) else x
        for x in xs
    ]


def broadcast_gather(src, dim: int, ind, keepdim=True, out=None, **kwargs):
    src_shape = src.shape
    ind_shape = ind.shape
    if src.dim() < ind.dim():
        src_shape = [*([1] * (ind.dim() - src.dim())), *src.shape]
    elif ind.dim() < src.dim():
        ind_shape = [*([1] * (src.dim() - ind.dim())), *ind.shape]
    if hasattr(out, "get_tensor"):
        out_shape = [max(s, i) for s, i in zip(src_shape, ind_shape)]
        out_shape[dim] = ind_shape[dim]
        out = out.get_tensor(out_shape, dtype=src.dtype)
    if ind.squeeze().dim() <= 1 and ind_shape[dim] == max(ind.shape):
        out = torch.index_select(src, dim, ind.view(-1), out=out)
    else:
        ind, src = broadcast_both_left(
            ind, src, ignored_dims=[dim if dim >= 0 else len(src.shape) + dim]
        )
        out = torch.gather(src, dim, ind, out=out, **kwargs)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def offset(x):
    return torch.cat((x[1:], x[:1]))


def shuffle(x):
    dim = 0
    perm = torch.randperm(x.shape[dim])
    return x[perm]


def squish(x, start: int = 0, end: int = 1):
    """Flatten dimensions ``start`` through ``end`` (inclusive) of ``x`` into one.

    E.g. ``squish(x, 1, 2)`` reshapes ``[A, B, C, D]`` to ``[A, B*C, D]``.
    The inverse of :func:`unsquish`.
    """
    if end < 0:
        end = end + len(x.shape)
    return x.reshape(list(x.shape[:start]) + [-1] + list(x.shape[end + 1 :]))


def unsquish(x, dim: int = 0, factor: int | None = None):
    """Split dimension ``dim`` of ``x`` into two dimensions.

    ``factor`` sets the size of the *second* new dimension (``dim`` becomes
    ``[size/factor, factor]``); a negative ``factor`` sets the size of the
    *first* instead (``[-factor, size/-factor]``); ``None`` splits into a
    square (``[sqrt(size), sqrt(size)]``). The inverse of :func:`squish`.
    """
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
    return x.reshape(list(x.shape[:dim]) + [new_d, new_d2] + list(x.shape[dim + 1 :]))


def interpolate(x, y, a):
    return x * (1 - a) + (a) * y


def broadcast_cross_product(x, y, dim=-1):
    x, y = broadcast_both_left(
        x, y, ignored_dims=[dim]
    )  # [dim if dim >= 0 else len(y.shape) + dim])
    return torch.cross(x, y, dim=dim)


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def mid_point(x, dim=-1, keepdim=True):
    return (x.amin(dim, keepdim=keepdim) + x.amax(dim, keepdim=keepdim)) * 0.5


def _get_broadcasted_shape(x, y):
    if x.dim() < y.dim():
        x = unsqueeze_left(x, y)
    if y.dim() < x.dim():
        y = unsqueeze_left(y, x)
    return [max(x_, y_) for x_, y_ in zip(x.shape[:-1], y.shape[:-1])] + [1]


def _dot_product_low_dim(x, y, out=None):
    if out is None:
        out = torch.empty(_get_broadcasted_shape(x, y), device=x.device)
    elif hasattr(out, "get_tensor"):
        out = out.get_tensor(_get_broadcasted_shape(x, y))
    out = out.squeeze(-1)
    torch.mul(x[..., 0], y[..., 0], out=out)
    for i in range(1, x.shape[-1]):
        torch.addcmul(out, x[..., i], y[..., i], out=out)
    return out.unsqueeze(-1)


def dot_product(x, y, dim=-1, keepdim=True, out=None):
    if dim == -1 or dim == (max(x.dim(), y.dim()) - 1):
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

    if not hasattr(dim, "__iter__"):
        dim = [dim]

    if x.dim() < y.dim():
        x = unsqueeze_left(x, y)
    if y.dim() < x.dim():
        y = unsqueeze_left(y, x)

    a = ascii_lowercase[: x.dim()]
    r = [a[i] for i in dim]
    ar = re.sub(f"[{'|'.join(r)}]", "", a)
    out = torch.einsum(f"{a},{a}->{ar}", x, y)
    dim = [d if d >= 0 else d + max(x.dim(), y.dim()) for d in dim]
    if keepdim:
        for d in sorted(dim):
            out = out.unsqueeze(d)
    return out


def pack_tensor(x, packing):
    if packing is None:
        return x
    counts, num_dims = packing
    d = x.dim() - (num_dims + 1)
    if (d < 0) or (x.shape[d] == 1):
        return x
    return torch.repeat_interleave(x, counts, d)


def unpack_tensor(x, packing):
    if packing is None:
        return x
    counts, num_dims = packing
    d = x.dim() - (num_dims + 1)
    return torch.split(x, [_.item() for _ in counts], d)


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


HANDLED_FUNCTIONS = {}
