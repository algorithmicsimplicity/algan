"""What the MPS zero-copy conversion will and will not import.

The conversion itself needs an Apple GPU and the patched Taichi
(``taichi_patches/``), so almost nothing about it can be exercised here. Its
*decision* can be, and that decision is where the bug was: the module read a
kernel's annotations to pick which arguments to import, and it excluded every
vector-element ndarray -- which is exactly the BVH node array that every
ray-tracing kernel takes, so the widest arrays in the renderer stayed on
Taichi's host-staging path with nothing saying so.

That is the shape of defect this file exists to catch: a *silent* fallback.
A staged argument costs four copies and an MPS stream sync per launch and
renders an identical frame, so a Mac cannot tell you it regressed and a green
suite there would not either. Reading the annotations is pure Python, so it
can be checked on any machine, which is why it is checked here.

No ``from __future__ import annotations`` here, deliberately and for the same
reason ``*_taichi.py`` files carry ``I002`` off: it turns a kernel's runtime
annotations into strings, and Taichi reads them raw
(``Invalid type annotation (argument 0) of Taichi kernel: ti.i32``).
"""

import torch

from algan.rendering.mps_zero_copy import _ndarray_positions, import_tensor
from algan.taichi_compat import kernel_arguments, ti


def _kernel_of(wrapped):
    """The ``Kernel`` behind a ``@ti.kernel``-decorated function.

    ``ti.kernel`` returns a wrapper; the object that carries ``arguments`` --
    and the object whose ``__call__`` the conversion is installed in front of
    -- is ``_primal``. Reached the same way ``taichi_fast_launch`` reaches it.
    """
    return wrapped._primal


@ti.kernel
def _scalar_args(
    n: ti.i32,
    plain: ti.types.ndarray(),
    typed: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(n):
        plain[i] = typed[i]


@ti.kernel
def _vector_arg(
    n: ti.i32,
    nodes: ti.types.ndarray(dtype=ti.types.vector(4, ti.f16), ndim=2),
    out: ti.types.ndarray(),
):
    for i in range(n):
        out[i] = ti.cast(nodes[i, 0][0], ti.f32)


def test_scalar_ndarray_arguments_are_importable_with_no_element_shape():
    positions = _ndarray_positions(_kernel_of(_scalar_args))
    # The i32 is not an ndarray and must not be in the table at all; both
    # ndarrays are, with a scalar element.
    assert positions == {1: (), 2: ()}


def test_a_vector_element_annotation_reports_its_element_shape():
    """The regression this module had: these used to be excluded outright.

    An excluded argument is not an error, it is a silent fall back to Taichi's
    staging path -- so the assertion is on the element shape being *carried*,
    which is what lets the import build the vector-element ndarray Taichi
    type-checks for.
    """
    positions = _ndarray_positions(_kernel_of(_vector_arg))
    assert positions == {1: (4,), 2: ()}


def test_the_bvh_node_argument_of_a_real_kernel_is_importable():
    """The one that matters, asked of the renderer's own kernel.

    ``NODE_ARG`` is built from ``bvh_arity`` and from whether the block is
    f16, both of which are settings, so this reads the extent off the same
    annotation the kernel was declared with rather than restating 4.
    """
    from algan.rendering.raytracing import raster_taichi
    from algan.rendering.raytracing import raytrace_kernels_taichi as rk

    kernel = _kernel_of(raster_taichi.raster_shadow_trace_arena)
    node_positions = [
        index
        for index, argument in enumerate(kernel_arguments(kernel))
        if argument.annotation is rk.NODE_ARG
    ]
    assert node_positions, "this kernel no longer takes a NODE_ARG"
    positions = _ndarray_positions(kernel)
    expected = tuple(rk.NODE_ARG.dtype.get_shape())
    for index in node_positions:
        assert positions.get(index) == expected


def test_a_host_tensor_is_never_imported():
    """A None is the "let Taichi have it" answer, and it has to stay one.

    Nothing about a CPU tensor can be adopted as an ``MTLBuffer``; the import
    has to decline rather than raise, because it runs in front of every launch
    on every device.
    """
    assert import_tensor(torch.zeros(4, 4)) is None
    assert import_tensor(torch.zeros(4, 4, 4), (4,)) is None
    assert import_tensor("not a tensor") is None
