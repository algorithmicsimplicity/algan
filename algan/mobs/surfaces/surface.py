"""Parametric surfaces -- the curved 3-D geometry primitive.

A :class:`Surface` samples a uniform grid of ``(u, v)`` points from the unit
square, maps each through a coordinate function into 3-D space, and tiles the
result with triangles. Every curved shape in Algan is one of these: a sphere is a
:class:`Surface` whose coordinate function is a sphere.

The grid resolution is chosen for you, **one axis at a time**: ``geometry_tolerance``
bounds how far the PN-triangle approximation may stray from the analytic surface,
and the search sizes each parameter axis against its own contribution to that
error, so an axis the surface is straight along (a cylinder's length, a cone's
slant) costs the minimum however curved the other axis is. The search is cached
per subclass and geometry configuration.
``render_tolerance_pixels`` then bounds the on-screen error when each PN
triangle is diced into flat render triangles -- per triangle, per frame, so
detail is spent only where the surface is near the camera. It is an absolute
pixel count at the renderer's reference frame height, scaled down in proportion
on shorter frames.

Surfaces carry the texture-map API: ``color_texture`` plus per-texel
reflectivity, roughness, refractive-index, normal and glow maps, all sampled in
the ray-tracing kernel. :meth:`Surface.set_color_by_function` colors by ``(u, v)``,
:meth:`Surface.set_color_by_image` paints an image across the domain, and
:meth:`Surface.set_shape_to` morphs one surface into another.

See :doc:`/advanced_user_tutorials/images_and_textures`.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    Off,
    Sync,
    active_scene_for_new_mob,
)
from algan.animation_timeline.timeline import EditRecord
from algan.constants.color import *
from algan.geometry.geometry import (
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.rendering.logical_pn import (
    evaluate_logical_pn,
    evaluate_logical_pn_per_patch,
    logical_pn_control_points,
    mean_patch_edge_length,
    normalize_pixel_tolerance,
)
from algan.settings.shape_style_profiles import _manim_shape_style_for
from algan.utils.file_utils import get_image
from algan.utils.mob_utils import pack_animatable_rows, pack_member_rows
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    cast_to_tensor,
    dot_product,
    squish,
    texture_u8_provenance,
    unsqueeze_left,
    unsquish,
)


def _as_member_colors(colors, count):
    """Cast per-member colors to Algan's ``[N, 5]`` RGB+glow+opacity form.

    Accepts RGB, RGBA or the full five channels, matching what
    :class:`~algan.constants.color.Color` stores, so a point cloud's RGBA array
    and a list of named colors both work.
    """
    colors = cast_to_tensor(colors).reshape(-1, cast_to_tensor(colors).shape[-1])
    if len(colors) == 1 and count > 1:
        colors = colors.expand(count, -1)
    if len(colors) != count:
        raise ValueError(
            f"expected {count} colors to match {count} centers, got {len(colors)}"
        )
    channels = colors.shape[-1]
    if channels == 5:
        return colors.contiguous()
    ones = torch.ones_like(colors[..., :1])
    zeros = torch.zeros_like(colors[..., :1])
    if channels == 4:
        # RGBA: glow sits between the color and the opacity.
        return torch.cat((colors[..., :3], zeros, colors[..., 3:4]), -1).contiguous()
    if channels == 3:
        return torch.cat((colors, zeros, ones), -1).contiguous()
    raise ValueError(
        f"colors must have 3 (RGB), 4 (RGBA) or 5 channels, got {channels}"
    )


# Ceiling on the grid resolution ``wave_color`` will refine a surface to. A
# tighter wave than this can afford is drawn as smoothly as the budget allows
# rather than tessellating the surface into millions of triangles.
_MAX_WAVE_GRID_RESOLUTION = 64

# Budget for the nearest-surface-point search that turns the general geometry
# metric from a same-parameter comparison into a true distance (see
# ``Surface._refine_geometry_deviation``). The search only ever runs on samples
# that are still above tolerance, and it runs in batches of
# ``_PROJECTION_BATCH`` so its cost stays bounded no matter how fine the probed
# grid is. ``_PROJECTION_BOX_CELLS`` limits how far, in grid cells, a sample may
# travel in parameter space: the metric wants the distance to the piece of
# surface the patch is meant to approximate, not to whatever fold of the shape
# happens to pass nearby.
_PROJECTION_BATCH = 1 << 16
_PROJECTION_STEPS = 12
_PROJECTION_BOX_CELLS = 2.0
# Backtracking ladder tried at every step. A Gauss-Newton step overshoots badly
# where the parameterization is strongly nonlinear (its Jacobian collapses at a
# pole and grows again within the same cell), and letting the damping alone
# walk the step back down costs one iteration per rejection. All scales are
# evaluated in one batch instead, so each iteration lands on the best of them.
_PROJECTION_SCALES = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)

# How many texels ``Surface.get_texture_locations`` resolves at a time. Every
# texel carries ten intermediate PN control points, so a 4K map evaluated in one
# go would want gigabytes; this bounds it to tens of megabytes regardless.
_TEXTURE_LOCATION_CHUNK_TEXELS = 1 << 18


#: Hysteresis band for the (currently unreachable) runtime resolution search:
#: a smaller grid is adopted only if it cuts triangle count by more than this.
#: Was a ``resolution_shrink_margin`` constructor argument, removed because the
#: search it feeds has been disabled since the logical PN system landed.
_RESOLUTION_SHRINK_MARGIN = 0.1


def _surface_resolution_pair(resolution):
    if isinstance(resolution, int):
        return int(resolution), int(resolution)
    u_resolution, v_resolution = resolution
    return int(u_resolution), int(v_resolution)


def _resolution_cache_callable_key(function):
    """Return a stable in-process identity for a geometry callable."""
    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    if code is not None:
        return ("python", code)
    return (
        "callable",
        type(function),
        getattr(function, "__module__", None),
        getattr(function, "__qualname__", None),
        id(function),
    )


def _freeze_resolution_cache_value(value):
    """Make common subclass construction state suitable for a cache key."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return (
            "tensor",
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("ndarray", str(array.dtype), array.shape, array.tobytes())
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        repr(key),
                        _freeze_resolution_cache_value(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_freeze_resolution_cache_value(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(
                sorted(
                    (_freeze_resolution_cache_value(item) for item in value),
                    key=repr,
                )
            ),
        )
    if callable(value):
        return _resolution_cache_callable_key(value)
    try:
        hash(value)
    except TypeError:
        return (type(value), repr(value))
    return value


_grid_triangle_indices_cache = {}


def get_grid_to_triangle_indices(
    grid_width: int, grid_height: int, device, weld=(False, False, False)
):
    """Internal: get the vertex indices that split a grid into triangles.

    Each grid cell becomes two triangles. The result is cached per grid shape and
    device, since surfaces of the same resolution all share it.

    Parameters
    ----------
    grid_width
        Number of grid points across.
    grid_height
        Number of grid points down.
    device
        Torch device the indices should live on.
    weld
        ``(wrap_x, pole_lo, pole_hi)`` from :func:`surface_weld_flags`. Each
        welds one boundary of a closed grid into shared vertices instead of
        coincident duplicates (DESIGN_mesh_identity.md ss3.1). Defaults to no
        welding, which is the shipped topology exactly.

    Returns
    -------
    torch.Tensor
        Vertex indices into the flattened grid, shape
        ``[(W-1) * (H-1) * 2, 3]`` -- minus ``W-1`` triangles per welded pole.
    """
    cache_key = (grid_width, grid_height, device, weld)
    if cache_key not in _grid_triangle_indices_cache:
        W, H = grid_width, grid_height
        wrap_x, pole_lo, pole_hi = weld
        i_indices = torch.arange(W - 1, device=device).unsqueeze(1).expand(-1, H - 1)
        j_indices = torch.arange(H - 1, device=device).unsqueeze(0).expand(W - 1, -1)

        i_next = i_indices + 1
        if wrap_x:
            # The wrap cell indexes column 0 instead of the duplicate column
            # W-1, so the seam becomes a SHARED edge rather than two copies
            # 1.7e-7 apart. Column W-1 is then simply never gathered; the
            # triangle count is unchanged.
            i_next = torch.where(i_next == W - 1, torch.zeros_like(i_next), i_next)

        def vertex_id(col, row):
            # A collapsed pole row is one vertex, not W coincident copies.
            if pole_lo:
                col = torch.where(row == 0, torch.zeros_like(col), col)
            if pole_hi:
                col = torch.where(row == H - 1, torch.zeros_like(col), col)
            return col * H + row

        j_next = j_indices + 1
        idx00 = vertex_id(i_indices, j_indices)
        idx01 = vertex_id(i_indices, j_next)
        idx10 = vertex_id(i_next, j_indices)
        idx11 = vertex_id(i_next, j_next)

        # The index order is the original one, and it is a SCREEN-space
        # contract: the renderer's backface bit is the projected winding, and
        # mirroring the world and the camera together (which is what moving
        # OUTWARD to +z did) leaves that projection unchanged. The world-space
        # cross of these three did flip with everything else, which is why the
        # outward normal is -cross here -- see ``_flat_corner_normals`` and
        # ``test_normal_orientation.py``.
        t1 = torch.stack((idx00, idx01, idx10), dim=-1)
        t2 = torch.stack((idx10, idx01, idx11), dim=-1)
        stacked = torch.stack((t1, t2), dim=-2)
        if pole_lo or pole_hi:
            # Collapsing a pole makes exactly one triangle of each adjacent
            # cell degenerate: t1 at the low pole and t2 at the high one, each
            # left with the pole vertex twice (which of the three slots it lands
            # in follows the winding above, and does not matter here -- the
            # degeneracy is a property of the vertex SET). Dropping them is what
            # turns the fan into
            # a proper cone of triangles, and it is why welding a pole changes
            # the triangle COUNT while welding the seam does not.
            keep = torch.ones((W - 1, H - 1, 2), dtype=torch.bool, device=device)
            if pole_lo:
                keep[:, 0, 0] = False
            if pole_hi:
                keep[:, H - 2, 1] = False
            stacked = stacked[keep]
        _grid_triangle_indices_cache[cache_key] = stacked.reshape(-1)
    return _grid_triangle_indices_cache[cache_key]


#: Tolerance for deciding a grid wraps or collapses. The same 1e-4 the normal
#: merge has always used (``compute_grid_vertex_normals``), and it stays a
#: tolerance for the same reason: whether a parametrization closes is a property
#: of the coordinates, not something the topology can be asked.
_WELD_TOLERANCE = 1e-4


def surface_weld_flags(grid):
    """Which of a grid's boundaries can be welded: ``(wrap_x, pole_lo, pole_hi)``.

    ``wrap_x`` when column ``W-1`` duplicates column 0 (a closed surface of
    revolution's u-seam), ``pole_lo``/``pole_hi`` when every column of row 0 /
    row ``H-1`` coincides (a collapsed pole, e.g. a ``Sphere``'s or a ``Cone``'s
    tip). All three are ``False`` unless ``ALGAN_WELD_SURFACE_SEAMS`` is on.

    Returns Python bools, so this synchronises with the device once per
    primitive build. That is deliberate: the result selects a cached index
    tensor, which cannot be a device value.
    """
    from algan.rendering.raytracing import settings as rt_settings

    if not rt_settings.weld_surface_seams or grid.dim() < 3:
        return (False, False, False)
    tol = _WELD_TOLERANCE
    wrap_x = bool((grid[..., 0, :, :] - grid[..., -1, :, :]).abs().lt(tol).all().item())
    pole_lo = bool(
        (grid[..., :, 0, :] - grid[..., :1, 0, :]).abs().lt(tol).all().item()
    )
    pole_hi = bool(
        (grid[..., :, -1, :] - grid[..., :1, -1, :]).abs().lt(tol).all().item()
    )
    return (wrap_x, pole_lo, pole_hi)


def surface_closed_axes(grid):
    """Which parameter axes of a materialized grid close on themselves.

    Returns ``(closed_u, closed_v)``: ``closed_u`` when column ``W-1``
    coincides with column 0 (a surface of revolution's u-seam -- a
    :class:`~algan.mobs.shapes_3d.Sphere`, a :class:`~algan.mobs.shapes_3d.Cylinder`, a :class:`~algan.mobs.shapes_3d.Cone`), ``closed_v``
    when row ``H-1`` coincides with row 0 (a :class:`~algan.mobs.shapes_3d.Torus` closes on both).

    This is :func:`surface_weld_flags`'s ``wrap_x`` test on both axes, but it
    answers a different question -- how a *texture* must be sampled, not how
    triangles are indexed -- so it is deliberately not gated on
    ``ALGAN_WELD_SURFACE_SEAMS``: a closed surface's texture has to wrap
    whether or not its seam vertices are shared.

    Returns Python bools, so this synchronises with the device once per
    textured primitive build. The result changes a tensor's *shape* (see
    :func:`wrap_pad_texture`), which a device value cannot do.
    """
    if grid.dim() < 3:
        return (False, False)
    tol = _WELD_TOLERANCE
    # A single-sample axis has one column playing both edges, which the test
    # below cannot tell from a wraparound.
    closed_u = grid.shape[-3] > 1 and bool(
        (grid[..., 0, :, :] - grid[..., -1, :, :]).abs().lt(tol).all().item()
    )
    closed_v = grid.shape[-2] > 1 and bool(
        (grid[..., :, 0, :] - grid[..., :, -1, :]).abs().lt(tol).all().item()
    )
    return (closed_u, closed_v)


def wrap_pad_texture(texture, closed_axes):
    """Repeat a texture's first row/column at its far edge on closed axes.

    The renderer addresses a ``[W, H]`` map as ``u * (W - 1)`` and clamps, so
    texel 0 sits at ``u == 0`` and texel ``W-1`` at ``u == 1``. On a surface
    whose u axis closes those are the *same place*, and the sampler has no way
    to blend the last texel back into the first: the map lands on the surface
    stretched by ``W / (W - 1)`` and cut by a hard seam wherever column 0
    disagrees with column ``W-1``.

    Appending a copy of column 0 as column ``W`` fixes both at once. Texel
    ``i`` then sits at ``u == i / W``, so every column spans the same
    ``1 / W`` of the way around, and the wrap cell interpolates column ``W-1``
    into column 0 exactly as an interior cell interpolates its neighbours.

    Parameters
    ----------
    texture
        A texture map ``[T, W, H, C]`` (u along ``W``, v along ``H``), or None.
    closed_axes
        ``(closed_u, closed_v)`` from :func:`surface_closed_axes`.

    Returns
    -------
    torch.Tensor
        The padded map, or ``texture`` unchanged when neither axis closes.
    """
    if texture is None:
        return None
    closed_u, closed_v = closed_axes
    if closed_u and texture.shape[-3] > 1:
        texture = torch.cat((texture, texture[..., :1, :, :]), -3)
    if closed_v and texture.shape[-2] > 1:
        texture = torch.cat((texture, texture[..., :, :1, :]), -2)
    return texture


def grid_to_triangle_vertices(grid, weld=(False, False, False)):
    """Internal: gather a per-grid-point quantity into per-triangle-vertex form.

    Turns values laid out on the surface grid into the flat triangle-vertex layout the
    renderer consumes. Works for positions, normals and colors alike.

    Parameters
    ----------
    grid
        Grid-shaped values, ``[..., W, H, C]``. A 1-D input is returned unchanged.

    Returns
    -------
    torch.Tensor
        The same values gathered per triangle vertex.
    """
    if grid.dim() == 1:
        return grid
    W, H = grid.shape[-3], grid.shape[-2]
    flat_grid = grid.reshape(*grid.shape[:-3], W * H, grid.shape[-1])
    indices = get_grid_to_triangle_indices(W, H, grid.device, weld)
    fused = _gather_triangles_on_cpu(flat_grid, indices)
    if fused is not None:
        return fused
    return flat_grid[..., indices, :]


def _cpu_prep_kernel(name):
    """Whether to dispatch the CPU batch-prep kernel called ``name``.

    Imported lazily: this module is on the animation side and must not pull
    Taichi in at import, the same reason ``timeline`` defers its own kernel
    import.
    """
    from algan.rendering.taichi_runtime import cpu_prep_kernel_enabled

    return cpu_prep_kernel_enabled(name)


def _gather_triangles_on_cpu(flat_grid, indices):
    """``flat_grid[..., indices, :]`` through the kernel, or None to decline.

    Declines on anything the kernel does not cover -- a non-CPU arch, a
    non-float32 or non-contiguous grid, an index table that is not the flat
    ``[triangles * 3]`` one ``get_grid_to_triangle_indices`` builds -- so the
    caller falls back to the advanced index. Byte-identical when it does run,
    since both paths copy the same elements.
    """
    if not _cpu_prep_kernel("cpugather"):
        return None
    if (
        flat_grid.dtype != torch.float32
        or flat_grid.device.type != "cpu"
        or flat_grid.dim() < 2
        or flat_grid.numel() == 0
        or not flat_grid.is_contiguous()
        or indices.dim() != 1
        or indices.numel() == 0
        or indices.dtype != torch.int64
        or not indices.is_contiguous()
    ):
        # Empty declines rather than being handled: Taichi has no use for a
        # zero-extent ndarray and torch's advanced index already does the right
        # thing with one.
        return None

    from algan.mobs.surfaces.surface_kernels_taichi import gather_grid_to_triangles

    points, channels = flat_grid.shape[-2], flat_grid.shape[-1]
    leading = flat_grid.shape[:-2]
    gathered = indices.shape[0]
    batches = flat_grid.numel() // (points * channels)
    out = torch.empty(
        (batches, gathered, channels), dtype=flat_grid.dtype, device=flat_grid.device
    )
    # Flattened views on both sides: the kernel indexes with flat offsets, which
    # measured 1.7x faster than the multi-dimensional form (see its comment).
    gather_grid_to_triangles(
        flat_grid.reshape(-1), indices, out.view(-1), points, channels
    )
    return out.reshape(*leading, gathered, channels)


def _sides_and_crosses_on_cpu(grid):
    """The fused sides + crosses + accumulate pass, or None to decline.

    Not bit-identical to the torch block it replaces (``surface_kernels_taichi``
    records why, and why the difference cannot open a seam); every other reason
    to decline is a shape or dtype the kernel does not cover.
    """
    if not _cpu_prep_kernel("cpunormals"):
        return None
    if (
        grid.dtype != torch.float32
        or grid.device.type != "cpu"
        or grid.dim() < 3
        or grid.shape[-1] != 3
        or grid.numel() == 0
        or not grid.is_contiguous()
    ):
        return None

    from algan.mobs.surfaces.surface_kernels_taichi import grid_normals_sides_crosses

    W, H = grid.shape[-3], grid.shape[-2]
    batched = grid.reshape(-1, W, H, 3)
    out = torch.empty_like(batched)
    grid_normals_sides_crosses(batched, out)
    return out.reshape(grid.shape)


def _grid_normals_paired():
    """Whether to build the per-triangle sides pairwise (see below).

    Gated through the same ``ALGAN_OPT_DISABLE`` bisect switch as the other
    byte-identical prep optimizations, under the name ``gridnormals``, so the
    A/B script can run both arms in one process.
    """
    from algan.animation_timeline.timeline import _opt_disabled

    return not _opt_disabled("gridnormals")


def _wrapped_difference(grid, axis, shift):
    """``grid.roll(shift, axis) - grid``, without materializing the roll.

    ``roll`` allocates and fills a whole copy of the grid, which the very next
    subtraction then reads once and throws away: two full-size writes where one
    will do. Writing the difference straight into one output buffer, in the two
    pieces the wrap-around splits it into, halves the traffic of every side.

    Bit-identical, and not by an argument about associativity: every element is
    the same subtraction of the same two elements, just written to a different
    place. That holds for NaN and inf operands as much as for finite ones, and
    for a degenerate axis of length 1, where the interior slice is empty and the
    wrap piece is the element minus itself -- exactly what ``roll`` gives there.

    ``axis`` is ``-3`` (the grid's x axis) or ``-2`` (its y axis); ``shift``
    matches ``Tensor.roll``'s, so ``+1`` reads the previous neighbour and ``-1``
    the next one.

    Measured on the shape the batched build passes, ``[120, 50, 24, 12, 3]``:
    **1.44x** over roll-then-subtract for the four sides (and 1.33x over the
    whole sides-and-crosses block once the accumulation below is in place too).
    Small grids are dispatch-bound and show nothing -- read the large rows, the
    same caveat P11 records.
    """

    def along(piece):
        return (
            (Ellipsis, piece, slice(None), slice(None))
            if axis == -3
            else (Ellipsis, slice(None), piece, slice(None))
        )

    interior, wrap = (
        ((slice(None, -1), slice(1, None)), (slice(-1, None), slice(None, 1)))
        if shift == 1
        else ((slice(1, None), slice(None, -1)), (slice(None, 1), slice(-1, None)))
    )
    neighbour, here = interior
    wrap_neighbour, wrap_here = wrap
    out = torch.empty_like(grid)
    torch.sub(grid[along(neighbour)], grid[along(here)], out=out[along(here)])
    torch.sub(
        grid[along(wrap_neighbour)],
        grid[along(wrap_here)],
        out=out[along(wrap_here)],
    )
    return out


def compute_grid_vertex_normals(grid):
    """Area-weighted vertex normals for a surface grid ``[..., W, H, 3]``,
    with closed-seam and pole merging. All computations broadcast over any
    leading dims (time, or a stack of same-shaped surfaces), which lets
    :func:`get_render_primitives_batched` run this once for many surfaces.
    """
    fused = _sides_and_crosses_on_cpu(grid)
    if fused is not None:
        # One stencil pass instead of the nine full-size intermediates the torch
        # block below materializes. Everything after this point -- the seam
        # merges, the pole fans, the normalize -- is shared, which is what keeps
        # closed seams and poles watertight regardless of which arm ran.
        unnormalized_normals = fused
    elif _grid_normals_paired():
        # The four triangles around a vertex use each neighbour twice, as the
        # second side of one and the first side of the next. The stacked form
        # below made that literal: eight copies of the grid in one
        # [..., W, H, 8, 3] tensor, the grid subtracted from every one of them,
        # then two stride-2 views sliced back out to cross, and finally a
        # [..., W, H, 4, 3] tensor reduced over its triangle axis. Differencing
        # each neighbour once, crossing the four pairs directly and adding them
        # evaluates exactly the same products and the same sum from the same
        # values: every operation is elementwise, and a length-4 reduction over
        # a contiguous axis is the sequential order these adds take, so the
        # result is bit-identical (asserted, including NaN payloads and signed
        # zeros, by benchmarks/_grid_normals_ab.py). It moves roughly half the
        # bytes, which matters because this function is the largest single item
        # in a render (see DESIGN_optimization_targets.md).
        #
        # Two later passes over the same block took the traffic down again, on
        # the same bit-identical terms: the sides are written straight into
        # their output instead of through a materialized roll
        # (_wrapped_difference), and the four are accumulated in place rather
        # than through three temporaries. 1.33x over the shipped form on the
        # shape the batched build passes; see P11b in
        # DESIGN_optimization_targets.md.
        side_x_minus = _wrapped_difference(grid, -3, 1)
        side_y_minus = _wrapped_difference(grid, -2, 1)
        side_x_plus = _wrapped_difference(grid, -3, -1)
        side_y_plus = _wrapped_difference(grid, -2, -1)
        normals_xm_ym = broadcast_cross_product(side_x_minus, side_y_minus)
        normals_ym_xp = broadcast_cross_product(side_y_minus, side_x_plus)
        normals_xp_yp = broadcast_cross_product(side_x_plus, side_y_plus)
        normals_yp_xm = broadcast_cross_product(side_y_plus, side_x_minus)
        # The same four boundary masks as below, addressed per triangle instead
        # of through a fancy index on the stacked axis. Corners are written by
        # two of these; both write zero, so the order does not matter.
        normals_xm_ym[..., 0, :, :] = 0
        normals_yp_xm[..., 0, :, :] = 0
        normals_ym_xp[..., -1, :, :] = 0
        normals_xp_yp[..., -1, :, :] = 0
        normals_xm_ym[..., :, 0, :] = 0
        normals_ym_xp[..., :, 0, :] = 0
        normals_xp_yp[..., :, -1, :] = 0
        normals_yp_xm[..., :, -1, :] = 0
        # ``a + b + c + d`` allocates and fills three whole tensors to reach one
        # answer. Accumulating in place after the first add makes the same three
        # additions in the same left-to-right order on the same values -- so it
        # is bit-identical, associativity never enters -- and materializes one.
        unnormalized_normals = normals_xm_ym + normals_ym_xp
        unnormalized_normals += normals_xp_yp
        unnormalized_normals += normals_yp_xm
    else:
        grid_x_plus_1 = grid.roll(-1, -3)
        grid_x_minus_1 = grid.roll(1, -3)
        grid_y_plus_1 = grid.roll(-1, -2)
        grid_y_minus_1 = grid.roll(1, -2)
        triangle_sides = unsquish(
            torch.stack(
                (
                    grid_x_minus_1,
                    grid_y_minus_1,
                    grid_y_minus_1,
                    grid_x_plus_1,
                    grid_x_plus_1,
                    grid_y_plus_1,
                    grid_y_plus_1,
                    grid_x_minus_1,
                ),
                -2,
            )
            - grid.unsqueeze(-2),
            -2,
            2,
        )
        triangle_normals = broadcast_cross_product(
            triangle_sides[..., 0, :], triangle_sides[..., 1, :]
        )
        triangle_normals[..., 0, :, [0, 3], :] = 0
        triangle_normals[..., -1, :, [1, 2], :] = 0
        triangle_normals[..., :, 0, [0, 1], :] = 0
        triangle_normals[..., :, -1, [2, 3], :] = 0
        unnormalized_normals = triangle_normals.sum(-2)

    # Merge unnormalized normals along closed seams using vectorized masking to avoid CPU-GPU synchronization.
    is_closed_x = torch.all(
        (grid[..., 0, :, :] - grid[..., -1, :, :]).abs() < 1e-4, dim=(-1, -2)
    )
    mask_x = is_closed_x.view(*is_closed_x.shape, 1, 1)
    closed_normals_x = (
        unnormalized_normals[..., 0, :, :] + unnormalized_normals[..., -1, :, :]
    )
    unnormalized_normals[..., 0, :, :] = torch.where(
        mask_x, closed_normals_x, unnormalized_normals[..., 0, :, :]
    )
    unnormalized_normals[..., -1, :, :] = torch.where(
        mask_x, closed_normals_x, unnormalized_normals[..., -1, :, :]
    )

    is_closed_y = torch.all(
        (grid[..., :, 0, :] - grid[..., :, -1, :]).abs() < 1e-4, dim=(-1, -2)
    )
    mask_y = is_closed_y.view(*is_closed_y.shape, 1, 1)
    closed_normals_y = (
        unnormalized_normals[..., :, 0, :] + unnormalized_normals[..., :, -1, :]
    )
    unnormalized_normals[..., :, 0, :] = torch.where(
        mask_y, closed_normals_y, unnormalized_normals[..., :, 0, :]
    )
    unnormalized_normals[..., :, -1, :] = torch.where(
        mask_y, closed_normals_y, unnormalized_normals[..., :, -1, :]
    )

    # Rebuild the normals at singular poles (e.g. Sphere poles, Cone tip).
    # A collapsed pole column accumulates nothing usable of its own: every
    # triangle side along the grid's x axis is degenerate there, so the
    # accumulated pole normals are built entirely from sub-epsilon
    # differences between coincident points. On the raw parametrization
    # those differences still carry the pole's direction, but once the mob
    # has been transformed the O(1) world coordinates round them away
    # completely, and summing them yields a direction that is tens of
    # degrees off and that swings as the mob moves -- a bright blob that
    # slides over the pole while the surface animates.
    #
    # Instead take the pole normal from the fan of faces that actually meet
    # it: pole vertex to each edge of the adjacent ring (column 1 / -2).
    # That is the same area-weighted vertex normal the accumulation is meant
    # to produce, but computed from non-degenerate geometry. The fan winding
    # follows the grid's own, which reverses between the two poles, hence
    # the swapped cross-product order. As a backstop the result is still
    # oriented into the ring's hemisphere: a pole normal from the wrong
    # hemisphere makes the patches touching the pole interpolate from an
    # inward normal at the pole to the (correct) outward normals on the
    # ring, sweeping the shading normal through the lit hemisphere on the
    # way -- a bright ring around an otherwise unlit pole.
    def _orient_to_ring(pole_normal, ring_normal):
        dot = (pole_normal * ring_normal).sum(-1, keepdim=True)
        return torch.where(dot < 0, -pole_normal, pole_normal)

    def _merged_pole_normal(pole_index, ring_index, reverse):
        pole = grid[..., :1, pole_index, :]
        ring = grid[..., :, ring_index, :]
        first = ring[..., :-1, :] - pole
        second = ring[..., 1:, :] - pole
        if reverse:
            first, second = second, first
        pole_normal = broadcast_cross_product(first, second).sum(-2, keepdim=True)
        accumulated = unnormalized_normals[..., :, pole_index, :]
        is_pole = torch.all(
            (grid[..., :, pole_index, :] - pole).abs() < 1e-4, dim=(-1, -2)
        )
        # A fan too thin to trust (single-column grid, or a ring collapsed
        # onto the pole itself) leaves the accumulated normals alone.
        usable = pole_normal.norm(p=2, dim=-1, keepdim=True) > 1e-12
        replace = is_pole.view(*is_pole.shape, 1, 1) & usable
        ring_normal = unnormalized_normals[..., :, ring_index, :].sum(-2, keepdim=True)
        return torch.where(
            replace,
            _orient_to_ring(pole_normal, ring_normal),
            accumulated,
        )

    if grid.shape[-2] > 1:
        # Both poles read the untouched accumulation, so a two-row grid
        # cannot have the first merge feed the second.
        merged_poles = [
            (pole_index, _merged_pole_normal(pole_index, ring_index, reverse))
            for pole_index, ring_index, reverse in ((0, 1, True), (-1, -2, False))
        ]
        for pole_index, merged in merged_poles:
            unnormalized_normals[..., :, pole_index, :] = merged

    # Unnegated: a mirror reverses a grid's orientation, so cross(du, dv) comes
    # out along the surface's outward normal now that OUTWARD is +z. The
    # negation this used to carry was the compensation for the old -z world.
    return F.normalize(unnormalized_normals, p=2, dim=-1)


def get_render_primitives_batched(surfaces):
    """Build render primitives for N surfaces that share a grid shape and
    frame count, running the geometry pipeline (normal computation and
    triangle-vertex gathers) once on a ``[N, T, W, H, 3]`` stack instead of
    once per surface. Numerically identical to calling
    :meth:`~algan.mobs.surfaces.surface.Surface.get_render_primitives` on each
    surface (all ops are
    elementwise or reduce over non-batch dims), but with N times fewer
    Python/torch dispatches. Callers must ensure every surface uses the stock
    ``Surface.get_render_primitives``, has no ``color_texture``, has
    ``ignore_normals`` False, and has identical grid dimensions and
    ``grid.location`` shape.
    """
    # Read uncopied: the result only feeds ``reshape`` and ``torch.stack``,
    # both out-of-place, so the property getter's defensive copy would be a
    # second full-grid copy per surface per batch for nothing.
    grids = torch.stack(
        [
            s._reshape_grid_for_render(
                s.grid.get_animated_attribute("location", copy=False)
            )
            for s in surfaces
        ]
    )
    weld = surface_weld_flags(grids)
    vertex_normals = grid_to_triangle_vertices(compute_grid_vertex_normals(grids), weld)
    corners = grid_to_triangle_vertices(grids, weld)
    return [
        s._build_render_primitive(
            grids[i], vertex_normals[i], precomputed_corners=corners[i], weld=weld
        )
        for i, s in enumerate(surfaces)
    ]


class Surface(Mob):
    """A smooth 2-D surface, embedded in 3-D space, A.K.A a manifold.
    The surface is implemented by sampling a uniform grid of 2-D points
    from the unit square (known as intrinsic coordinates, or "UV coordinates"),
    tiling this grid with triangles, and then mapping the triangle corners
    to 3-D world coordinates as defined by the manifold function.

    Parameters
    ----------
    coord_function
        The function mapping 2-D intrinsic coordinates (ranging from [0,1]), to 3-D world coordinates,
        which defines the manifold's shape.
    grid_height
        Number of sampled points along the ``v`` axis. Defaults to ``None``,
        meaning the resolution is chosen automatically to meet
        ``geometry_tolerance``; giving either grid size turns that search off.
    grid_width
        Number of sampled points along the ``u`` axis. Defaults to ``None``, as
        ``grid_height`` does; giving only one of the two uses it for both.
    checkered_color
        Second color, applied to alternating vertices of the grid to give the
        surface a checkerboard. Accepts anything ``color`` does. Defaults to
        ``None``, meaning the surface is a single flat ``color``.
    ignore_normals
        Whether to draw the surface as flat, faceted triangles instead of
        smoothing it. Defaults to ``False``, meaning smooth vertex normals are
        computed from the sampled grid and the triangles are curved into PN
        patches. ``True`` skips both, so each triangle is shaded flat and the
        surface reads as a low-poly facet mesh.
    geometry_tolerance
        Maximum sampled world-space distance between the analytic surface and its
        PN-triangle approximation at construction time, in world units. It measures
        how far the approximation strays *off* the surface, not how the surface is
        parameterized, so a coordinate function that stretches or collapses its
        parameters (a polar cap, a superellipse meridian) costs no extra vertices.
        Shapes with a known exact surface-distance expression use it directly;
        general parametric surfaces search for each sample's nearest analytic
        point.
        The selected grid is cached per concrete Surface subclass and geometry
        configuration, so constructing the same shape again does not repeat
        the resolution search.
    render_tolerance_pixels
        Maximum sampled deviation between a PN triangle and the flat render
        triangles it is dynamically diced into, in output pixels. Defaults to
        ``0.5`` -- half a pixel, so the dice is invisible; ``None`` removes the
        bound entirely. Lower it for a sharper silhouette on a surface that
        fills the frame, raise it to buy back the triangles.

        Every PN triangle is diced only as finely as it itself needs, in every
        frame, so detail spent where the surface is close to the camera is not
        spent on the rest of the mesh or on the frames where it is far away.

        The budget is stated at the renderer's reference frame height (1000 px)
        and scaled down in proportion on anything shorter, since a
        low-resolution frame needs finer dicing than its pixel count alone
        suggests -- the analytic-coverage antialiasing computes a pixel's
        coverage from the microtriangles crossing it, and at ``PREVIEW`` each
        pixel covers far more of the object. So the default is worth 0.5 px from
        1080p up and 0.2 px at ``PREVIEW``, and halving it halves both.
    min_grid_resolution, max_grid_resolution
        Bounds for automatic grid sizing, measured in vertices per axis. Default
        to ``2`` and ``200``. The floor is two vertices -- one cell -- because a
        surface that is straight along an axis needs no more than that, and the
        search measures rather than assumes it.
    u_range, v_range
        The interval of each parameter the surface is built over, as
        ``(start, end)``. Default to ``(0, 1)``, the whole unit square. These
        are stored for a subclass's ``coord_function`` to read -- the base class
        does not itself remap ``(u, v)`` -- which is how
        :class:`~algan.mobs.shapes_3d.Sphere` and its siblings cut open shells
        from partial sweeps. Those classes take theirs as angles, in degrees
        like every other angle in Algan.
    resolution
        Grid size as ``(u_patches, v_patches)``, or one int for both. Counts
        *patches*, one less than the vertices ``grid_width`` / ``grid_height``
        count, matching Manim. Defaults to ``None``. Ignored if either grid size
        is given.
    color_texture
        Optional color texture map ``[W, H, 5]`` -- one image, sampled
        bilinearly in-kernel by the ray tracer. It is an ordinary animatable
        attribute: assign a new map of the same shape to animate it.
    reflectivity_texture, roughness_texture, refractive_index_texture
        Optional per-texel material property maps, each ``[W, H, 1]`` (or
        ``[W, H]``). Like ``color_texture`` they are
        sampled bilinearly per fragment inside the ray tracing kernel (only
        the general wavefront tracer implements this; batches containing such
        maps are routed to it automatically, for both flat and curved PN
        triangles). Properties without a map keep the per-vertex system. Maps
        of different resolutions are resampled to a common resolution.
    normal_texture
        Optional tangent-space normal map ``[W, H, 3]``, with components in
        ``[-1, 1]``: x along increasing ``u``, y along
        increasing ``v``, z along the smooth surface normal (``(0, 0, 1)`` =
        unperturbed). Perturbs the shading normal per fragment in-kernel.
        Note: under the default vertex-shaded pipeline lighting is baked at
        the vertices, so a normal map only affects effects evaluated per
        fragment (mirror reflections, refraction, ray traced shadows, and
        fragment shading when enabled).
    glow_texture
        Optional glow strength/radius maps, each ``[W, H, 1]`` (or ``[W, H]``).
        These are consumed per-vertex by the glow accumulator, so they are
        baked to the surface grid resolution (raise ``grid_width``/
        ``grid_height`` for more detail).
    *args, **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` -- notably ``color``,
        ``opacity`` and ``location``.

    Attributes
    ----------
    grid : :class:`~algan.animatable_base.mob.Mob`
        The surface's vertices, as a child Mob, and the way to reach anything
        that varies from vertex to vertex. ``grid.location`` holds their 3-D
        world positions, shape ``(*, grid_width * grid_height, 3)`` row-major
        over the sample grid, and ``grid.color`` their colors -- both animatable
        like any other Mob attribute, so writing either records an animation.

        These colors are the surface's albedo, interpolated across each triangle
        from its corners. Setting a :attr:`color_texture` replaces them as the
        albedo source, which is then sampled bilinearly from the texture's texels
        instead; shading itself is per-fragment either way. The grid's resolution
        is fixed at construction and a texture's is not, which is why the two are
        kept separate.

        :attr:`vertices` is shorthand for ``grid.location``.

    See Also
    --------
    :class:`~algan.mobs.shapes_3d.Sphere` : And ``Cylinder`` / ``Cone`` / ``Torus``, the built-in surfaces.
    :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_function` : Color it by a function of ``(u, v)``.
    :meth:`~algan.mobs.surfaces.surface.Surface.set_location_by_function` : Reshape it after construction.

    Examples
    --------
    A saddle, from its parametric equation. The coordinate function takes the
    whole ``(u, v)`` grid at once and returns a 3-D point per sample:

    .. algan:: Example1Surface
        :save_last_frame:

        from algan import *
        import torch

        def saddle(uv):
            x = uv[..., :1] * 4 - 2
            y = uv[..., 1:] * 4 - 2
            return torch.cat((x, y, (x ** 2 - y ** 2) * 0.4), -1)

        Surface(saddle, checkered_color=BLUE).rotate(60, RIGHT).spawn()

        Scene.save_video()
    """

    _morph_family = "grid"

    def _adopt_structural_attrs(self, target):
        """Take the target's image at the end of a morph.

        A surface's texture is stored under an attribute name encoding its own
        ``W * H``, so two surfaces with differently-sized textures share no
        attribute for the same-kind morph's ``animatable_attrs`` intersection
        to copy -- and the result kept the SOURCE's picture. A 4x4 red texture
        becoming an 8x4 blue one ended red.

        Assigned through the property rather than the generic
        ``_MORPH_ADOPTED_ATTRS`` list because the setter is what detaches
        history when the two resolutions cannot be interpolated. It reads the
        target through the uncopied row and folds that back into the ``[W, H,
        5]`` image the setter wants, rather than through the public getter,
        which would clone the widest attribute in the engine for nothing.
        """
        super()._adopt_structural_attrs(target)
        if getattr(target, "_color_texture_attr", None) is None:
            if getattr(self, "_color_texture_attr", None) is not None:
                self.color_texture = None
            return self
        self.color_texture = target._as_texture_image(target._color_texture_uncopied())
        return self

    #: Handedness of this surface's ``(u, v)`` parameterization: ``1`` when
    #: ``du x dv`` already points out of the solid, ``-1`` when it points in.
    #: A ``-1`` surface has its v axis reversed on the way to the renderer
    #: (:meth:`_reshape_grid_for_render`), which flips its vertex normals and
    #: its triangle winding without moving a single vertex -- so the shape's own
    #: ``coord_function`` keeps whatever parameterization it is written to
    #: match (Manim's, for the built-ins) while the geometry the renderer sees
    #: faces outward. Shading reads this through
    #: :attr:`~algan.animatable_base.mob.Mob.two_sided`: a surface whose
    #: normals face out does not need the renderer to guess a side.
    _grid_orientation = 1

    # Every concrete Surface type gets its own dictionary via
    # ``__init_subclass__``. Entries are keyed by both the fitting policy and a
    # compact signature of the construction-time geometry, so parameterized
    # subclasses (Sphere(radius=...), user-defined shapes, etc.) do not share
    # incompatible resolutions.
    _geometry_resolution_cache = {}
    _geometry_resolution_warning_keys = set()
    _geometry_resolution_cache_lock = threading.RLock()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._geometry_resolution_cache = {}
        cls._geometry_resolution_warning_keys = set()
        cls._geometry_resolution_cache_lock = threading.RLock()

    @classmethod
    def clear_geometry_resolution_cache(cls):
        """Clear construction-time resolution entries for this exact class."""
        with cls._geometry_resolution_cache_lock:
            cls._geometry_resolution_cache.clear()
            cls._geometry_resolution_warning_keys.clear()

    def __init__(
        self,
        coord_function=None,
        grid_height=None,
        grid_width=None,
        checkered_color=None,
        color_texture=None,
        reflectivity_texture=None,
        roughness_texture=None,
        refractive_index_texture=None,
        normal_texture=None,
        glow_texture=None,
        ignore_normals=False,
        geometry_tolerance=0.0005,
        render_tolerance_pixels=0.5,
        min_grid_resolution=2,
        max_grid_resolution=200,
        *args,
        u_range=None,
        v_range=None,
        resolution=None,
        **kwargs,
    ):
        # User-defined subclasses conventionally store their geometry
        # parameters before calling ``super().__init__``. Preserve that small
        # pre-Surface state for the cache identity before Mob construction adds
        # scene/timeline bookkeeping that differs for every instance.
        resolution_cache_state = dict(self.__dict__)

        self.u_range = (0, 1) if u_range is None else tuple(u_range)
        self.v_range = (0, 1) if v_range is None else tuple(v_range)

        self.resolution = resolution
        if resolution is not None and grid_width is None and grid_height is None:
            u_resolution, v_resolution = _surface_resolution_pair(resolution)
            grid_width = u_resolution + 1
            grid_height = v_resolution + 1

        if coord_function is None:
            coord_function = self.coord_function

        self.coord_function_active = coord_function
        self.ignore_normals = ignore_normals
        self._color_texture_attr = None

        self._geometry_auto_resolution_enabled = (
            grid_height is None and grid_width is None
        )
        # Compatibility flag retained for older introspection. Runtime
        # topology changes are deliberately disabled by the logical PN system.
        self._auto_resolution_enabled = False
        self._geometry_tolerance = float(geometry_tolerance)
        self._render_tolerance_pixels = normalize_pixel_tolerance(
            render_tolerance_pixels
        )
        self._resolution_tolerance = self._geometry_tolerance
        self._min_grid_resolution = int(min_grid_resolution)
        self._max_grid_resolution = int(max_grid_resolution)
        self._pending_auto_resolution = None
        self._resolution_update_in_progress = True
        if not np.isfinite(self._geometry_tolerance):
            raise ValueError("geometry_tolerance must be finite")
        if self._geometry_tolerance <= 0:
            raise ValueError("geometry_tolerance must be greater than zero")
        if self._min_grid_resolution < 2:
            raise ValueError("min_grid_resolution must be at least 2")
        if self._max_grid_resolution < self._min_grid_resolution:
            raise ValueError(
                "max_grid_resolution must be greater than or equal to "
                "min_grid_resolution"
            )
        # Opt-in Manim shape profile: a mapped shape adopts Manim's
        # constructor fill (and checkerboard pair) unless the caller passed a
        # color of its own -- ``Sphere(checkerboard_colors=[a, b])`` arrives
        # here already translated to ``color``/``checkered_color`` by
        # ``shapes_3d._surface_resolution_kwargs``. Injected
        # here so both the Mob's own color attribute and the grid child built
        # below carry it.
        if "color" not in kwargs:
            style = _manim_shape_style_for(type(self))
            if style is not None and style["color"] is not None:
                kwargs["color"] = style["color"]
                if checkered_color is None and style.get("checker_color") is not None:
                    checkered_color = style["checker_color"]
        super().__init__(*args, **kwargs)
        kwargs["scene"] = self.scene
        # Texture timelines are keyed by texel count. AttributeTimeline fixes
        # its channel width at first creation, so differently sized textures
        # must never share the generic ``color_texture`` key.
        self.color_texture = color_texture

        # Compile a stable logical PN topology once. Geometry tolerance is
        # measured against the surface's construction-time world geometry;
        # later transforms and deformations never rewrite this topology.
        if self._geometry_auto_resolution_enabled:
            initial_location = self.location.clone()

            def initial_surface_function(uv):
                return coord_function(uv.clone()) + initial_location

            grid_width, grid_height = self._get_cached_geometry_resolution(
                initial_surface_function,
                coord_function,
                resolution_cache_state,
            )
        else:
            if grid_width is None:
                grid_width = grid_height
            if grid_height is None:
                grid_height = grid_width

        self.grid_height, self.grid_width = grid_height, grid_width

        # Optional texture-mapped material properties. Reflectivity/roughness/
        # refractive-index maps are combined into one 5-channel "material
        # texture" (plus a bitmask of which channels are texture-driven) that
        # the general wavefront kernel samples per fragment; the normal map is
        # kept separate. Glow maps are baked to per-vertex grid values (the
        # glow accumulator interpolates triangle corners, so per-vertex is its
        # native resolution).
        self.material_texture = None
        self.material_texture_flags = 0
        self.normal_texture = None
        material_prop_textures = {
            "reflectivity": reflectivity_texture,
            "roughness": roughness_texture,
            "refractive_index": refractive_index_texture,
        }
        # Kept so a material applied later (set_material -> a roughness_map or
        # metalness_map) merges into these rather than replacing them: the
        # three share one packed texture, so a new channel means rebuilding it
        # from every source map, not just the new one.
        self._material_prop_textures = {
            k: v for k, v in material_prop_textures.items() if v is not None
        }
        if self._material_prop_textures:
            self._rebuild_material_texture()
        if normal_texture is not None:
            self.normal_texture = self._normalize_texture_shape(normal_texture, 3).to(
                self.location.device
            )
        if glow_texture is not None:
            kwargs["glow"] = self._bake_texture_to_grid(glow_texture)

        base_grid = self.get_base_grid()
        grid_points = squish(coord_function(base_grid), -3, -2) + self.location

        # ``geometry_tolerance`` bounds a world-space distance measured on
        # exactly this geometry, so remember it as a fraction of how big the
        # geometry's patches were. A surface that is scaled (or deformed)
        # afterwards can then carry the bound forward to the renderer instead of
        # quoting a stale absolute length, and a PN soup converted from it
        # arrives at the identical number from the identical triangles.
        #
        # The weld has to be the render path's, not the default: welding a pole
        # DROPS the degenerate triangle of every cell touching it, and those are
        # exactly the short ones, so an unwelded reference quotes a scale the
        # renderer never measures (a Sphere's came out 3.7% low).
        reference_grid = unsquish(grid_points, -2, self.grid_height).reshape(
            -1, self.grid_width, self.grid_height, 3
        )
        reference = float(
            mean_patch_edge_length(
                grid_to_triangle_vertices(
                    reference_grid, surface_weld_flags(reference_grid)
                ).reshape(1, -1, 3, 3)
            ).mean()
        )
        self._geometry_slack_ratio = (
            self._geometry_tolerance / reference if reference > 0 else 0.0
        )

        # Parsed here because the vertex grid below indexes and assigns into a
        # color tensor, so a hex string or an RGB tuple has to be a color
        # before it reaches Mob.__init__.
        color = to_color(
            kwargs["color"] if "color" in kwargs else self.get_default_color()
        )
        checkered_color = to_color(checkered_color)
        if checkered_color is None:
            checkered_color = color
        else:
            checkered_color = unsqueeze_left(checkered_color, color)

        if color_texture is not None:
            tex = color_texture
            if tex.dim() == 3:  # [W, H, 5]
                tex_temp = tex.unsqueeze(0).permute(0, 3, 1, 2)
                tex_temp = F.interpolate(
                    tex_temp,
                    size=(grid_width, grid_height),
                    mode="bilinear",
                    align_corners=True,
                )
                vertex_color_texture = tex_temp.permute(0, 2, 3, 1).squeeze(0)
            elif tex.dim() == 4:  # [T, W, H, 5]
                tex_temp = tex.permute(0, 3, 1, 2)
                tex_temp = F.interpolate(
                    tex_temp,
                    size=(grid_width, grid_height),
                    mode="bilinear",
                    align_corners=True,
                )
                vertex_color_texture = tex_temp.permute(0, 2, 3, 1)
            else:
                vertex_color_texture = tex
            color = squish(vertex_color_texture, -3, -2)
        else:
            color_grid = (
                (BLACK * 0)
                .view(1, -1)
                .expand((self.grid_width * self.grid_height, -1))
                .contiguous()
            )
            color_grid[::2] = color
            color_grid[1::2] = checkered_color
            color_grid = color_grid.view(self.grid_height, self.grid_width, 5)
            color = squish(color_grid, -3, -2)
        # color = grid_to_triangle_vertices(color)
        kwargs["color"] = color
        kwargs["location"] = grid_points
        self.grid = Mob(**kwargs)
        self.add_children(self.grid)
        self.components = [self.grid]
        self.grid.is_primitive = True
        self.is_primitive = True
        self.ignore_wave_animations = True
        self._resolution_update_in_progress = False

    @property
    def vertices(self) -> torch.Tensor:
        """The surface's vertex positions, shape ``(*, grid_width * grid_height, 3)``.

        Shorthand for ``surface.grid.location``: the live tensor the renderer
        tessellates from, laid out row-major over the ``grid_height`` x
        ``grid_width`` sample grid. Writing it moves the surface's vertices,
        which is the lowest-level way to deform a shape --
        :meth:`set_location_by_function` is the usual one. Reach for
        :attr:`grid` itself when you want the vertices' *colors*
        (``grid.color``) rather than their positions.

        The assigned value must carry the same number of vertices the surface
        already has: the grid resolution is chosen once, at construction, and
        nothing here can change it. Positions are absolute, in world units, not
        offsets from the surface's location.

        Animation
        ---------
        Assignment is recorded, so the vertices travel to their new positions
        over the current context's duration (1 second by default). Wrap the
        write in ``Off()`` to move them instantly.
        """
        return self.grid.location

    @vertices.setter
    def vertices(self, value):
        self.grid.location = value

    @classmethod
    def from_batches(cls, centers, *args, colors=None, **kwargs):
        """Build many independently indexable surfaces without per-surface mobs.

        One packed Mob covers every centre in ``centers``: it is a single Scene
        actor whose vertex grids are concatenated into one tensor, so the whole
        collection costs one construction and one
        :meth:`get_render_primitives` call per frame batch rather than one each.
        Index it (``spheres[3]``) for a view onto a single member, which shares
        the pack's timeline rows.

        Every member has the same shape, resolution and material -- only its
        centre and color vary. Anything else that differs needs separate Mobs.

        Parameters
        ----------
        centers
            World-space centre of each surface, shape ``(N, 3)`` in world units.
            Any nested sequence is cast to a tensor and reshaped.
        colors
            Per-member color, shape ``(N, 3)`` as RGB, ``(N, 4)`` as RGBA or
            ``(N, 5)`` as Algan's RGB+glow+opacity. Defaults to None, giving
            every member the ``color`` passed in ``kwargs``.
        *args, **kwargs
            Passed to the ordinary constructor, which builds one representative
            member -- so ``radius``, ``resolution``, ``color`` and the texture
            maps all mean what they usually do, and apply to the whole pack.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            The packed Mob, of whichever subclass this was called on.

        Animation
        ---------
        Not animated: this only constructs. Animating the pack moves every
        member; animating ``pack[i]`` moves just that one. All members share one
        lifespan, so they spawn and despawn together -- stagger an entrance with
        opacity rather than with separate spawns.

        Examples
        --------
        A lattice of spheres as one Mob:

        .. algan:: Example1SurfaceFromBatches
            :save_last_frame:

            from algan import *
            import torch

            grid = torch.linspace(-2, 2, 6)
            centers = torch.cartesian_prod(grid, grid, torch.zeros(1))
            Sphere.from_batches(centers, radius=0.15, color=BLUE).spawn()

            Scene.save_video()
        """
        centers = cast_to_tensor(centers).reshape(-1, 3)
        count = len(centers)
        if count == 0:
            raise ValueError("from_batches requires at least one centre")

        # One representative member through the ordinary constructor, at the
        # first centre. The resolution search, color grid, textures and
        # materials are then exactly what a lone member would have got, and the
        # packing below only widens rows -- which is what keeps a pack
        # bit-identical to batch_mobs over separately constructed members.
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        kwargs["location"] = centers[:1]
        # Construction is instantaneous by definition, and the packing below
        # re-allocates timeline rows, which is only valid while the Mob's
        # history is fresh. A subclass constructor that repositions itself
        # (Dot3D calls move_to) would otherwise record an animation here.
        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=kwargs["scene"].animation_manager,
        ):
            mob = cls(*args, **kwargs)

        points_per_grid = mob.grid_width * mob.grid_height
        # Rebuilt from the same expression __init__ uses, rather than by
        # subtracting the representative's centre back off its grid, so member
        # i's rows are bit-identical to a separately constructed member's.
        relative_grid = squish(
            mob.coord_function_active(mob.get_base_grid().clone()), -3, -2
        )
        grid_overrides = {
            "location": (relative_grid.unsqueeze(-3) + centers.unsqueeze(-2)).reshape(
                1, count * points_per_grid, 3
            )
        }
        surface_overrides = {"location": centers.unsqueeze(0)}

        if colors is not None:
            if kwargs.get("checkered_color") is not None:
                raise ValueError(
                    "from_batches cannot combine per-member colors with a "
                    "shared checkered_color"
                )
            if mob._has_color_texture:
                raise ValueError(
                    "from_batches cannot combine per-member colors with a "
                    "color_texture, which the whole pack shares"
                )
            member_colors = _as_member_colors(colors, count)
            grid_color = mob.grid.get_animated_attribute("color")
            packed = (
                grid_color.repeat(1, count, 1)
                .contiguous()
                .view(1, count, points_per_grid, grid_color.shape[-1])
            )
            # __init__ lays a grid out as alternating color / checkered-color
            # rows. Substitute per-member values into that same layout instead
            # of rebuilding it, so the two constructions cannot drift apart.
            packed[:, :, ::2] = member_colors.view(1, count, 1, -1)
            packed[:, :, 1::2] = member_colors.view(1, count, 1, -1)
            grid_overrides["color"] = packed.view(1, count * points_per_grid, -1)
            surface_overrides["color"] = member_colors.unsqueeze(0)

        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=mob.animation_manager,
        ):
            pack_member_rows(mob.grid, count, points_per_grid, overrides=grid_overrides)
            pack_animatable_rows(mob, count, overrides=surface_overrides)
        return mob

    @property
    def geometry_tolerance(self) -> float:
        """How far this surface's mesh may sit from the exact shape, in world units.

        Fixed when the surface is constructed -- it is what chose the grid
        resolution -- and read-only thereafter. Set it in the constructor to
        trade vertices against accuracy.

        See Also
        --------
        :attr:`~algan.mobs.surfaces.surface.Surface.render_tolerance_pixels` : The per-frame budget, in screen terms.
        """
        return self._geometry_tolerance

    @property
    def render_tolerance_pixels(self) -> float:
        """How far a drawn triangle may sit from the true surface, in pixels.

        The budget at the renderer's reference frame height, scaled down in
        proportion on shorter frames. Fixed at construction and read-only.
        Unlike :attr:`geometry_tolerance` it is spent afresh every frame, on
        whichever parts of the surface are near the camera. ``inf`` when the
        surface declares no bound at all.
        """
        return self._render_tolerance_pixels

    @property
    def color_texture(self):
        """An image painted across the surface, as an ``[W, H, 5]`` RGBA+glow tensor.

        Assigning an image maps it over the surface's parameter domain, so it follows
        the surface as it deforms. ``None`` means the surface is drawn from its
        per-vertex colors instead. Assigning a texture whose resolution differs from
        the current one detaches history, since the two cannot be interpolated.

        Reading it back gives the image in that same ``[W, H, 5]`` layout -- not the
        flat ``W * H * 5`` row the timeline stores it as -- so arithmetic on it can
        be assigned straight back::

            surface.color_texture = surface.color_texture * 0.5  # half brightness

        The value is a :class:`~algan.constants.color.Color`, one per texel, so the
        color API applies to a whole map at once::

            surface.color_texture = surface.color_texture.mult_opacity(0.5)

        Its five channels are ``(R, G, B, glow, alpha)``, which is why the plain
        multiplication above dims the alpha and the glow along with the color.
        Reach for one of them through ``.rgb``, ``.glow`` or ``.opacity`` and assign
        the result back -- the read is a copy, so writing into it alone changes
        nothing::

            texels = surface.color_texture
            texels.rgb = texels.rgb * 0.5
            surface.color_texture = texels

        On an axis where the surface closes on itself -- ``u`` on a
        :class:`~algan.mobs.shapes_3d.Sphere`, both axes on a
        :class:`~algan.mobs.shapes_3d.Torus` -- the image wraps: its last column of
        texels neighbours its first, each column spanning ``1 / W`` of the way
        around, so a map whose edges join draws no seam. On an open axis the first
        and last texels sit on the two edges.

        Animation
        ---------
        An ordinary animatable attribute: assigning a new image is recorded as an
        animation, interpolating texel by texel from the old texture to the new one
        over the current context's duration (1 second by default). Assign inside
        ``Off()`` to swap it instantly. The new image must match the current
        resolution to interpolate -- see the note above about detaching history.
        """
        attr = getattr(self, "_color_texture_attr", None)
        if attr is None:
            return None
        # A texel is a color, so hand the map back as one: .rgb / .glow /
        # .opacity and mult_opacity then apply to the whole image. Only on the
        # public read -- the primitive build goes through
        # _color_texture_uncopied, which keeps its cat and mult_opacity off
        # Color's __torch_function__.
        return self._as_texture_image(self.get_animated_attribute(attr)).as_subclass(
            Color
        )

    def _as_texture_image(self, row):
        """A stored ``[..., W*H*5]`` texture row as its ``[W, H, 5]`` image.

        The timeline stores an attribute as a flat channel vector per row, which
        for a texture is the whole picture in one row. The public shape is the
        image, so the row is folded back here -- and the leading dims go with it
        when there is exactly one row, since a texture belongs to the surface
        rather than to a row of it.
        """
        leading = tuple(row.shape[:-1])
        height, width = int(self.texture_height), int(self.texture_width)
        image = row.reshape(*leading, height, width, 5)
        if all(d == 1 for d in leading):
            image = image.reshape(height, width, 5)
        return image

    @property
    def _has_color_texture(self):
        """Whether a color texture is set, without reading it.

        ``color_texture is not None`` answers the same question by materializing
        the whole image out of the timeline and cloning it. The render loop and
        the primitive build ask it several times per frame batch, where the
        texture is the widest attribute in the engine, so presence tests go
        through this instead.
        """
        return getattr(self, "_color_texture_attr", None) is not None

    def _color_texture_uncopied(self):
        """The color texture as a read-only view, or None.

        Same texels as :attr:`color_texture`, but as the flat ``[..., W*H*5]``
        row the timeline stores and without the defensive clone the public
        property makes. Only for callers that feed the result straight into
        out-of-place arithmetic -- mutating it corrupts the timeline's
        materialized state.
        """
        attr = getattr(self, "_color_texture_attr", None)
        if attr is None:
            return None
        return self.get_animated_attribute(attr, copy=False)

    @color_texture.setter
    def color_texture(self, texture):
        previous_attr = getattr(self, "_color_texture_attr", None)
        if texture is None:
            if previous_attr is not None and self.is_spawned():
                self.detach_history()
            self._color_texture_attr = None
            return self

        texture = torch.as_tensor(texture)
        # A texture belongs to the surface, not to a row of it, so a leading
        # dim of 1 is a batch axis some other tensor carried along rather than
        # a picture axis -- drop it before the shape is judged.
        while texture.dim() > 3 and texture.shape[0] == 1:
            texture = texture[0]
        if texture.dim() == 4:
            # A texture is an ordinary animatable attribute: animate it by
            # assigning a new image, not by handing over a sequence of them.
            # This used to be accepted here and then fail deep in
            # materialization with an unrelated tensor-size error.
            raise ValueError(
                "color_texture must be a single image [W, H, 5], not a sequence "
                f"of them; got {tuple(texture.shape)}. To animate it, assign a "
                "new [W, H, 5] image of the same resolution and Algan will "
                "interpolate to it."
            )
        if texture.dim() != 3 or texture.shape[-1] != 5:
            current = (
                f" This surface's is [{int(self.texture_height)}, "
                f"{int(self.texture_width)}, 5]."
                if self._has_color_texture
                else ""
            )
            raise ValueError(
                f"color_texture must have shape [W, H, 5], got "
                f"{tuple(texture.shape)}.{current}"
            )
        texture_height, texture_width = texture.shape[-3:-1]
        attr = f"color_texture_{texture_height * texture_width}"

        if previous_attr is not None and previous_attr != attr and self.is_spawned():
            # Keep the old texture topology on a frozen historical clone. The
            # live surface then receives a fresh timeline with the new width.
            self.detach_history()

        self._color_texture_attr = attr
        self.texture_height = int(texture_height)
        self.texture_width = int(texture_width)
        # u8 provenance, proved here at authoring where a full-image pass is
        # cheap and sync-free, and trusted by the merge (texture_u8_storage
        # must not probe texels on the prefetch worker). Any later assignment
        # re-runs the proof, so arithmetic on the texels (``tex * 0.5``)
        # clears it exactly when it stops holding.
        u8_ok = texture_u8_provenance(texture)
        self._color_texture_u8_ok = u8_ok
        # ...and the AND over every map assigned at this resolution, because a
        # frame window can show ANY of them: a batch before an animated
        # reassignment carries the previous map, and admitting it to u8
        # storage on the latest map's proof would round texels that are not
        # k/255. Windows described as segments prove their own endpoints
        # instead (see _segment_stack_u8_ok).
        self._color_texture_u8_ok_all = bool(
            u8_ok
            and (
                previous_attr != attr or getattr(self, "_color_texture_u8_ok_all", True)
            )
        )
        self.register_attrs_as_animatable([attr])
        # Opt the attribute into the timeline's segment-window description
        # (texture_time_lerp): an animated reassignment's frame window then
        # reaches get_render_primitives as endpoint images plus per-frame
        # weights instead of one materialized image per frame.
        self.scene.timeline_manager.enable_segment_windows(attr)
        # The surface's own row is the only one anything reads: the grid child
        # shades from the map through the surface, never from a row of its own.
        # A recursive write (the default for every attribute) would hand the
        # grid a row too, and a row of this attribute is a whole image -- so
        # every frame of every batch would materialize, lerp and copy a second
        # 30 MB texture that no code path consumes.
        prs = self._prevent_recursive_sets
        will_record = self.is_animating()
        flat = squish(texture, -3, -1)
        self._prevent_recursive_sets = True
        try:
            setattr(self, attr, flat)
        finally:
            self._prevent_recursive_sets = prs
        if will_record:
            # Stamp the AUTHORED map on the edit the assignment just
            # recorded: the stored state is ``pre + (map - pre)``, an ulp
            # off per texel, which is enough to void k/255 provenance for
            # the segment-window endpoint stack (texture_time_lerp +
            # texture_u8_storage). The description uses this as its lerp
            # TARGET only; the stored states stay what constant frames read.
            event = self.scene.timeline_manager.last_recorded_event
            if event is not None and len(event.recorded_edit_records) == 1:
                edit = event.recorded_edit_records[0]
                rows = event.recorded_edits[0][3].view(-1)
                mine = self.scene.timeline_manager.attr_to_timeline[
                    attr
                ].mob_id_to_inds.get(self.id)
                if (
                    event.recorded_edits[0][0] == attr
                    and mine is not None
                    and rows.numel() == mine.view(-1).numel()
                    and bool((rows == mine.view(-1)).all())
                ):
                    edit.authored_target = (
                        flat.detach().reshape(1, rows.numel(), -1).clone()
                    )
        return self

    def _get_u_values_and_v_values(self):
        """Return Manim-compatible sample coordinates for the UV domain."""
        resolution = self.resolution
        if resolution is None:
            resolution = (self.grid_width - 1, self.grid_height - 1)
        u_resolution, v_resolution = _surface_resolution_pair(resolution)
        return (
            np.linspace(*self.u_range, u_resolution + 1),
            np.linspace(*self.v_range, v_resolution + 1),
        )

    def get_unit_normals(self):
        """Return one smooth unit normal for each sampled surface vertex.

        Oriented the way the renderer orients them (outward, for the built-in
        shapes), because both go through :meth:`_reshape_grid_for_render`. The
        v axis that reorientation reverses is put back before flattening, so
        row i is still the normal of ``grid.location``'s row i.
        """
        grid = self._reshape_grid_for_render(self.grid.location)
        normals = compute_grid_vertex_normals(grid)
        if self._grid_orientation < 0:
            normals = normals.flip(-2)
        return normals.reshape(*grid.shape[:-3], -1, 3)

    def set_checkerboard_colors(self, *colors, opacity=None) -> Surface:
        """Paint the surface in a checkerboard of alternating colors.

        Colors are assigned to grid vertices in rotation along both axes, so two
        colors give the usual checkerboard and three or more give diagonal
        stripes. The pattern is laid over the surface's ``(u, v)`` grid, so it
        follows the shape as it deforms, and its resolution is the grid's --
        raise ``grid_width`` / ``grid_height`` for finer squares.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default): the vertices cross-fade to their new colors. Wrap the call
        in ``Off()`` to apply it instantly.

        Parameters
        ----------
        *colors
            Two or more colors to alternate between. Each is an Algan
            :class:`~algan.constants.color.Color`, a named constant such as
            ``BLUE``, or anything ``Color()`` accepts. Passing none leaves the
            surface unchanged.
        opacity
            Opacity to apply to the whole surface alongside the colors, from
            ``0`` to ``1``. Defaults to ``None``, leaving it as it is.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_function` : Color it by an arbitrary function of ``(u, v)``.

        Examples
        --------
        .. algan:: Example1SurfaceSetCheckerboardColors
            :save_last_frame:

            from algan import *

            Sphere(grid_width=17, grid_height=17).set_checkerboard_colors(
                BLUE, YELLOW
            ).rotate(20, RIGHT).spawn()

            Scene.save_video()
        """
        if not colors:
            return self
        converted = [
            torch.as_tensor(
                color, device=self.grid.color.device, dtype=self.grid.color.dtype
            ).reshape(-1, self.grid.color.shape[-1])[0]
            for color in colors
        ]
        palette = torch.stack(converted)
        u_indices = torch.arange(self.grid_width, device=palette.device).unsqueeze(1)
        v_indices = torch.arange(self.grid_height, device=palette.device).unsqueeze(0)
        color_grid = palette[(u_indices + v_indices) % len(palette)]
        self.grid.color = color_grid.reshape(-1, color_grid.shape[-1])
        if opacity is not None:
            self.grid.opacity = opacity
        return self

    def set_color_by_axis(
        self, axes=None, colorscale=None, axis: int = 2, **kwargs
    ) -> Surface:
        """Color the surface by how far along one axis each point sits.

        The classic height map: low points take the first color of the scale,
        high points the last, and everything between is interpolated. Because
        the colors are assigned per grid vertex they travel with the surface, so
        a shape colored by height keeps those colors when it is later moved --
        recolor it in an updater if the map should stay locked to world space.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default), so the surface cross-fades into the new coloring. Wrap the
        call in ``Off()`` to apply it instantly.

        Parameters
        ----------
        axes
            A plot axes object supplying the value range to spread the scale
            over, read from its ``x_range`` / ``y_range`` / ``z_range``.
            Defaults to ``None``, meaning the range is taken from the surface's
            own extent along ``axis``.
        colorscale
            The colors to interpolate between, either as a plain sequence
            (spread evenly over the range) or as ``(color, value)`` pairs
            pinning each color to a coordinate. Defaults to ``None``, which
            leaves the surface unchanged.
        axis
            Which world axis the value is read along: ``0`` for x, ``1`` for y,
            ``2`` for z. Defaults to ``2``, colouring by height.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        Raises
        ------
        ValueError
            If an unrecognized keyword argument is passed.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_function` : Color it by ``(u, v)`` rather than by position.

        Examples
        --------
        .. algan:: Example1SurfaceSetColorByAxis
            :save_last_frame:

            from algan import *
            import torch

            def saddle(uv):
                x = uv[..., :1] * 4 - 2
                y = uv[..., 1:] * 4 - 2
                return torch.cat((x, y, (x ** 2 - y ** 2) * 0.4), -1)

            Surface(saddle).set_color_by_axis(
                colorscale=[BLUE, GREEN, YELLOW]
            ).rotate(60, RIGHT).spawn()

            Scene.save_video()
        """
        if colorscale is None and "colors" in kwargs:
            colorscale = kwargs.pop("colors")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported keyword argument(s): {unsupported}")
        if colorscale is None:
            return self

        values = self.grid.location.reshape(-1, 3)[..., axis]
        entries = list(colorscale)
        if not entries:
            return self
        if isinstance(entries[0], tuple) and len(entries[0]) == 2:
            colors, pivots = zip(*entries)
            pivots = torch.as_tensor(pivots, device=values.device, dtype=values.dtype)
        else:
            colors = entries
            axis_range = getattr(axes, ("x_range", "y_range", "z_range")[axis], None)
            if axis_range is None:
                pivot_min, pivot_max = values.min(), values.max()
            else:
                pivot_min, pivot_max = axis_range[:2]
            pivots = torch.linspace(
                float(pivot_min),
                float(pivot_max),
                len(colors),
                device=values.device,
                dtype=values.dtype,
            )

        palette = torch.stack(
            [
                torch.as_tensor(
                    color, device=self.grid.color.device, dtype=self.grid.color.dtype
                ).reshape(-1, self.grid.color.shape[-1])[0]
                for color in colors
            ]
        )
        upper = torch.searchsorted(pivots, values).clamp(1, len(pivots) - 1)
        lower = upper - 1
        denominator = (pivots[upper] - pivots[lower]).clamp_min(1e-12)
        alpha = ((values - pivots[lower]) / denominator).clamp(0, 1).unsqueeze(-1)
        vertex_colors = palette[lower] * (1 - alpha) + palette[upper] * alpha
        vertex_colors[values <= pivots[0]] = palette[0]
        vertex_colors[values >= pivots[-1]] = palette[-1]
        self.grid.color = vertex_colors
        return self

    def _prepare_auto_resolution_translation(self, displacement):
        if not self._can_update_resolution():
            return False
        current_function = self._current_surface_function()

        def target_function(uv):
            return current_function(uv) + displacement

        target_width, target_height = self._find_screen_space_resolution(
            target_function
        )
        self._pending_auto_resolution = (
            target_width,
            target_height,
            target_function,
        )
        pre_width = max(self.grid_width, target_width)
        pre_height = max(self.grid_height, target_height)
        if (pre_width, pre_height) != (self.grid_width, self.grid_height):
            self._change_resolution(pre_width, pre_height, current_function)
        return True

    def _prepare_auto_resolution_basis_change(
        self, transform_location, current_basis, target_basis
    ):
        if not self._can_update_resolution():
            return False
        current_function = self._current_surface_function()

        def target_function(uv):
            points = current_function(uv)
            local_points = map_global_to_local_coords(
                transform_location, current_basis, points
            )
            return map_local_to_global_coords(
                transform_location, target_basis, local_points
            )

        target_width, target_height = self._find_screen_space_resolution(
            target_function
        )
        self._pending_auto_resolution = (
            target_width,
            target_height,
            target_function,
        )
        pre_width = max(self.grid_width, target_width)
        pre_height = max(self.grid_height, target_height)
        if (pre_width, pre_height) != (self.grid_width, self.grid_height):
            self._change_resolution(pre_width, pre_height, current_function)
        return True

    def _finalize_auto_resolution_change(self):
        prepared = self._pending_auto_resolution
        self._pending_auto_resolution = None
        if prepared is None:
            return self._update_resolution_for_current_shape(allow_upsample=False)
        required_width, required_height, target_function = prepared
        width, height = self._select_auto_resolution(required_width, required_height)
        if (width, height) == (self.grid_width, self.grid_height):
            return self
        return self._change_resolution(width, height, target_function)

    def _select_auto_resolution(
        self, required_width, required_height, allow_upsample=True
    ):
        """Apply asymmetric hysteresis to a required grid resolution.

        Any required growth is retained immediately. A smaller dimension is
        adopted only when the complete required grid would reduce triangle
        count by more than ``_RESOLUTION_SHRINK_MARGIN``; otherwise that
        dimension stays at its current size.

        Unreachable in the current engine: its only caller is gated on
        ``_can_update_resolution``, and the logical PN system fixes topology at
        construction, so ``_auto_resolution_enabled`` is always False. Kept
        against that gate being reopened.
        """
        current_width = self.grid_width
        current_height = self.grid_height
        required_width = int(required_width)
        required_height = int(required_height)
        if not allow_upsample:
            required_width = min(required_width, current_width)
            required_height = min(required_height, current_height)

        target_width = max(current_width, required_width)
        target_height = max(current_height, required_height)
        current_work = max(current_width - 1, 1) * max(current_height - 1, 1)
        required_work = max(required_width - 1, 1) * max(required_height - 1, 1)
        shrink_boundary = current_work * (1.0 - _RESOLUTION_SHRINK_MARGIN)
        if required_work < shrink_boundary:
            return required_width, required_height
        return target_width, target_height

    def _can_update_resolution(self):
        if not getattr(self, "_auto_resolution_enabled", False):
            return False
        if getattr(self, "_resolution_update_in_progress", False):
            return False
        if not hasattr(self, "grid"):
            return False
        timeline = self.scene.timeline_manager
        return not any(
            attr_timeline.active_state is not attr_timeline.current_state
            for attr_timeline in timeline.attr_to_timeline.values()
        )

    def _current_surface_function(self):
        """Return a continuous evaluator for the current world-space surface.

        **Only valid while this Mob's transform is unchanged.** The evaluator is
        half snapshot, half live read: ``affine`` is fitted once, here, against
        the Mob's state at this moment, while ``coord_function_active`` is called
        afresh every time and reads live state -- a Cylinder's is built from
        ``radius``, ``height`` and the Mob's own basis vectors. Move, rotate or
        rescale the Mob in between and the canonical points come from the new
        transform while the correction mapping them into the world still
        describes the old one, so the result is wrong by the difference.

        Not a theoretical concern: :meth:`_change_resolution` builds the
        evaluator before ``detach_history`` and calls it afterwards, so a bug
        that let ``detach_history`` inflate a Cylinder's basis (see
        :func:`~algan.geometry.geometry.get_rotation_between_bases`) surfaced
        here as a grid scaled by that inflation. It is a punishing failure to
        diagnose, because the fit is self-consistent by construction: the
        residual measured *here* is zero no matter how wrong the later
        evaluation turns out to be. ``detach_history`` preserving the transform
        is what closes that gap today, and
        ``test_detached_history_preserves_the_surface_function`` guards it.
        """
        base_grid = self.get_base_grid()
        canonical = self.coord_function_active(base_grid.clone()).reshape(-1, 3)
        current = self.grid.location.reshape(-1, self.grid_width * self.grid_height, 3)[
            0
        ]
        design = torch.cat((canonical, torch.ones_like(canonical[..., :1])), dim=-1)
        # The design matrix is rank deficient whenever the canonical surface is
        # flat in some direction -- a plane spans only three of its four columns
        # -- which is exactly the shape a color wave most often has to refine.
        # ``lstsq``'s CUDA driver assumes full rank and returns garbage there
        # without raising, so take the pseudo-inverse's minimum-norm solution:
        # identical where the fit is determined, and correct on the surface's own
        # affine span where it is not.
        affine = torch.linalg.pinv(design) @ current

        def current_function(uv):
            points = self.coord_function_active(uv.clone())
            homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
            return homogeneous @ affine

        return current_function

    def _project_points_to_pixels(self, points):
        camera = getattr(self.scene, "camera", None)
        if camera is None:
            return None, None

        camera_location = camera.location.reshape(-1, 3)[0]
        forward = camera.get_forward_direction().reshape(-1, 3)[0]
        right = camera.get_right_direction().reshape(-1, 3)[0]
        upwards = camera.get_up_direction().reshape(-1, 3)[0]
        relative = points - camera_location
        depth = (relative * forward).sum(dim=-1)

        screen_vector = camera.screen.location.reshape(-1, 3)[0] - camera_location
        screen_distance = (screen_vector * forward).sum().abs().clamp_min(1e-8)
        pixel_scale = self.scene.video_settings.resolution[1] / (
            2.0 * float(camera.screen_half_height)
        )
        safe_depth = depth.clamp_min(1e-8)
        x = (relative * right).sum(dim=-1) * screen_distance / safe_depth
        y = (relative * upwards).sum(dim=-1) * screen_distance / safe_depth
        return torch.stack((x, y), dim=-1) * pixel_scale, depth

    def _screen_space_error(self, approximated, exact):
        approximated_pixels, approximated_depth = self._project_points_to_pixels(
            approximated
        )
        exact_pixels, exact_depth = self._project_points_to_pixels(exact)
        if approximated_pixels is None:
            # Without a camera there is no meaningful pixel-space metric.
            # Conservatively force the automatic search to its configured
            # maximum rather than silently reverting to world-space error.
            return torch.tensor(
                float("inf"),
                device=approximated.device,
                dtype=approximated.dtype,
            )

        near = max(float(getattr(self.scene.camera, "near", 0.0)), 1e-6)
        approximated_visible = approximated_depth > near
        exact_visible = exact_depth > near
        if torch.any(approximated_visible != exact_visible):
            return torch.tensor(
                float("inf"), device=approximated.device, dtype=approximated.dtype
            )
        visible = approximated_visible & exact_visible
        if not torch.any(visible):
            return torch.zeros((), device=approximated.device, dtype=approximated.dtype)
        return (
            (approximated_pixels[visible] - exact_pixels[visible])
            .norm(p=2, dim=-1)
            .max()
        )

    def _compute_pn_geometry_error(self, coord_function, width, height):
        """Sample construction-time world error of a logical PN grid.

        Exact at the decision that uses it: the result exceeds
        ``geometry_tolerance`` if and only if some sampled point really is
        further than that from the surface. Below the tolerance it is only an
        upper bound, because samples that are already inside it are not worth
        the cost of measuring precisely (see
        :meth:`_refine_geometry_deviation`).
        """
        device = self.location.device
        dtype = self.location.dtype
        grid_u = torch.linspace(0, 1, width, device=device, dtype=dtype)
        grid_v = torch.linspace(0, 1, height, device=device, dtype=dtype)
        grid_uu, grid_vv = torch.meshgrid(grid_u, grid_v, indexing="ij")
        base_grid = torch.stack((grid_uu, grid_vv), dim=-1)
        grid_points = coord_function(base_grid.clone())
        vertex_normals = compute_grid_vertex_normals(grid_points)

        triangle_uvs = grid_to_triangle_vertices(base_grid).reshape(-1, 3, 2)
        triangle_corners = grid_to_triangle_vertices(grid_points).reshape(-1, 3, 3)
        triangle_normals = grid_to_triangle_vertices(vertex_normals).reshape(-1, 3, 3)
        control_points = logical_pn_control_points(triangle_corners, triangle_normals)
        sample_uv = torch.tensor(
            [
                [1 / 3, 1 / 3],
                [1 / 2, 0.0],
                [0.0, 1 / 2],
                [1 / 2, 1 / 2],
                [1 / 6, 1 / 6],
                [1 / 6, 2 / 3],
                [2 / 3, 1 / 6],
                [1 / 4, 1 / 4],
                [1 / 4, 1 / 2],
                [1 / 2, 1 / 4],
            ],
            device=device,
            dtype=dtype,
        )
        pn_points = evaluate_logical_pn(control_points, sample_uv)
        barycentric = torch.stack(
            (
                1.0 - sample_uv[:, 0] - sample_uv[:, 1],
                sample_uv[:, 0],
                sample_uv[:, 1],
            ),
            dim=-1,
        )
        analytic_uv = torch.einsum("sk,pka->psa", barycentric, triangle_uvs)
        analytic_points = coord_function(analytic_uv.clone())
        deviation = self._pn_geometry_deviation(
            pn_points,
            analytic_points,
            analytic_uv,
        )
        if self._geometry_deviation_is_same_parameter():
            deviation = self._refine_geometry_deviation(
                coord_function,
                pn_points,
                analytic_uv,
                deviation,
                width,
                height,
            )
        return deviation.max()

    def _pn_geometry_deviation(self, pn_points, analytic_points, analytic_uv):
        """Return sampled distance from PN points to the analytic surface.

        The general parametric surface has no inverse mapping or implicit
        distance function, so its fallback compares points at the same
        parameter coordinates. That is only an *upper bound* on the geometric
        distance; :meth:`_refine_geometry_deviation` tightens it to the real
        one. Shapes with an exact surface-distance expression override this
        hook instead, and are then used as-is.
        """
        return (pn_points - analytic_points).norm(dim=-1)

    @classmethod
    def _geometry_deviation_is_same_parameter(cls):
        """True when this shape uses the generic same-parameter fallback."""
        return cls._pn_geometry_deviation is Surface._pn_geometry_deviation

    def _refine_geometry_deviation(
        self, coord_function, pn_points, analytic_uv, deviation, width, height
    ):
        """Turn same-parameter deviations into true surface distances.

        Comparing points at matching parameters counts tangential
        reparameterization as geometric error. Wherever the parameterization
        stretches -- a polar cap, a superellipse meridian, any pole or crease
        -- that term dominates and decays only as fast as the parameter
        spacing, so no attainable grid meets an absolute world tolerance even
        though the mesh itself sits on the surface to within microns. That is
        what drove arbitrary shapes to ``max_grid_resolution``.

        Measure instead the distance from each PN sample to the *nearest*
        analytic point, found by a local search seeded at the matching
        parameter. Only samples still above tolerance can change the outcome,
        so the rest keep their (upper-bound) same-parameter value, and the
        search stops as soon as one refined sample is still out of tolerance.
        """
        tolerance = self._geometry_tolerance
        flat_deviation = deviation.reshape(-1)
        candidates = (flat_deviation > tolerance).nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            return deviation

        # Worst first, so the common "this grid is nowhere near good enough"
        # verdict is reached in the first batch.
        candidates = candidates[flat_deviation[candidates].argsort(descending=True)]
        targets = pn_points.reshape(-1, 3)
        seeds = analytic_uv.reshape(-1, 2)
        box = torch.tensor(
            (
                _PROJECTION_BOX_CELLS / max(width - 1, 1),
                _PROJECTION_BOX_CELLS / max(height - 1, 1),
            ),
            device=seeds.device,
            dtype=seeds.dtype,
        )

        refined = flat_deviation.clone()
        for start in range(0, candidates.numel(), _PROJECTION_BATCH):
            batch = candidates[start : start + _PROJECTION_BATCH]
            distances = self._nearest_surface_distance(
                coord_function, targets[batch], seeds[batch], box, tolerance
            )
            refined[batch] = distances
            if bool((distances > tolerance).any()):
                break
        return refined.reshape(deviation.shape)

    @staticmethod
    def _nearest_surface_distance(coord_function, targets, uv, box, tolerance):
        """Distance from each target to the closest point on the surface.

        ``uv`` seeds a damped Gauss-Newton search for the parameters
        minimizing ``|coord_function(uv) - target|``, confined to ``box``
        (per-axis half-widths in parameter units) around each seed and to the
        unit parameter square. Derivatives are finite differences, since the
        package runs under ``inference_mode`` and user coordinate functions
        need not be differentiable.

        A step is kept only where it reduces the distance, so the result is
        never larger than the seed's. The refined metric is therefore never
        *looser* than the same-parameter one, and a search that fails to
        converge -- a degenerate Jacobian at a pole, a non-smooth coordinate
        function -- degrades to the old conservative answer instead of an
        optimistic one.

        Samples drop out of the iteration as soon as they are within
        ``tolerance``: the caller only needs to know which side of it the
        worst sample lands on, and on a stretched parameterization the great
        majority of samples get there in the first two or three steps.
        """
        eps = torch.finfo(targets.dtype).eps
        difference_step = (box * 0.05).clamp_min(1e-5)

        def evaluate(parameters):
            # Surface functions built by ``_find_geometry_resolution`` add the
            # mob's ``[1, 1, 3]`` location, which broadcasts a flat ``[N, 2]``
            # parameter list up to ``[1, N, 3]``. Keep the sample axis aligned
            # with the caller's so every quantity below stays ``[N, ...]``.
            values = coord_function(parameters.clone())
            return values.reshape(*parameters.shape[:-1], values.shape[-1])

        points = evaluate(uv)
        result = (points - targets).norm(dim=-1)

        active = (result > tolerance).nonzero(as_tuple=False).squeeze(-1)
        current = uv[active]
        points = points[active]
        targets = targets[active]
        distance = result[active]
        lower = torch.clamp(current - box, 0.0, 1.0)
        upper = torch.clamp(current + box, 0.0, 1.0)
        damping = torch.full_like(distance, 1e-3)

        for _ in range(_PROJECTION_STEPS):
            if active.numel() == 0:
                break
            residual = points - targets
            jacobian = []
            for axis in (0, 1):
                offset = torch.zeros_like(current)
                offset[..., axis] = difference_step[axis]
                plus = (current + offset).clamp(0.0, 1.0)
                minus = (current - offset).clamp(0.0, 1.0)
                spacing = (plus[..., axis] - minus[..., axis]).clamp_min(eps)
                jacobian.append(
                    (evaluate(plus) - evaluate(minus)) / spacing.unsqueeze(-1)
                )
            du, dv = jacobian

            # Levenberg normal equations for the 2 unknowns. The damping is
            # *added* to the diagonal rather than scaling it: a parameter
            # derivative that vanishes -- the collapsed axis of a polar cap,
            # any axis at the edge of a domain it approaches with zero speed --
            # leaves a zero on the diagonal, and scaling it leaves the system
            # singular, which would freeze the well-conditioned direction too.
            u_norm = (du * du).sum(-1)
            v_norm = (dv * dv).sum(-1)
            regularizer = damping * torch.maximum(u_norm, v_norm).clamp_min(eps)
            uu = u_norm + regularizer
            uv_dot = (du * dv).sum(-1)
            vv = v_norm + regularizer
            gradient_u = (du * residual).sum(-1)
            gradient_v = (dv * residual).sum(-1)
            # The system is a Gram matrix plus a positive diagonal, so its
            # determinant is non-negative and only vanishes when the two
            # parameter directions become collinear. Test that *relatively*:
            # a stretched parameterization has a small Jacobian and hence a
            # determinant of order ``|J|^4``, which an absolute epsilon rejects
            # as unsolvable even though the system is perfectly conditioned.
            determinant = uu * vv - uv_dot * uv_dot
            reference = (uu * vv).clamp_min(torch.finfo(targets.dtype).tiny)
            solvable = determinant > reference * eps
            safe = torch.where(solvable, determinant, determinant.new_ones(()))
            step = torch.stack(
                (
                    (uv_dot * gradient_v - vv * gradient_u) / safe,
                    (uv_dot * gradient_u - uu * gradient_v) / safe,
                ),
                dim=-1,
            )
            step = torch.where(solvable.unsqueeze(-1), step, torch.zeros_like(step))
            step = step.clamp(-box, box)

            # Backtracking line search, all scales in one evaluation.
            scales = step.new_tensor(_PROJECTION_SCALES).view(1, -1, 1)
            ladder = torch.minimum(
                torch.maximum(
                    current.unsqueeze(-2) + step.unsqueeze(-2) * scales,
                    lower.unsqueeze(-2),
                ),
                upper.unsqueeze(-2),
            )
            ladder_points = evaluate(ladder)
            ladder_distance = (ladder_points - targets.unsqueeze(-2)).norm(dim=-1)
            best_distance, best = ladder_distance.min(dim=-1)
            best_index = best.unsqueeze(-1).unsqueeze(-1)

            improved = best_distance < distance
            if not bool(improved.any()):
                break
            keep = improved.unsqueeze(-1)
            current = torch.where(
                keep,
                ladder.gather(-2, best_index.expand(-1, -1, 2)).squeeze(-2),
                current,
            )
            points = torch.where(
                keep,
                ladder_points.gather(-2, best_index.expand(-1, -1, 3)).squeeze(-2),
                points,
            )
            distance = torch.where(improved, best_distance, distance)
            result[active] = distance

            # Retire everything that has come inside tolerance, and trust the
            # linearization more where it worked, less where it did not.
            damping = torch.where(
                improved, (damping * 0.5).clamp_min(1e-6), damping * 4.0
            )
            unresolved = distance > tolerance
            if not bool(unresolved.all()):
                active = active[unresolved]
                current = current[unresolved]
                points = points[unresolved]
                targets = targets[unresolved]
                distance = distance[unresolved]
                damping = damping[unresolved]
                lower = lower[unresolved]
                upper = upper[unresolved]
        return result

    def _find_geometry_resolution(self, surface_function):
        """Choose the stable construction-time logical PN grid dimensions."""
        minimum = self._min_grid_resolution
        maximum = self._max_grid_resolution
        tolerance = self._geometry_tolerance
        measured = {}

        def acceptable(width, height):
            # The per-axis searches, the trim pass and the final check all
            # revisit grids their neighbours already measured, and measuring
            # one is far more expensive than remembering it.
            key = (width, height)
            if key in measured:
                return measured[key]
            try:
                error = self._compute_pn_geometry_error(surface_function, width, height)
                verdict = bool(
                    torch.isfinite(error).item() and error.item() <= tolerance
                )
            except Exception:
                verdict = False
            measured[key] = verdict
            return verdict

        def first_acceptable(other, vary_width):
            low, high = minimum, maximum
            best = maximum
            while low <= high:
                middle = (low + high) // 2
                width, height = (middle, other) if vary_width else (other, middle)
                if acceptable(width, height):
                    best = middle
                    high = middle - 1
                else:
                    low = middle + 1
            return best

        def trim(width, height):
            """Shrink each axis as far as the *joint* measurement allows.

            Each axis was sized on its own with the other axis at
            ``max_grid_resolution``, and the growth loop raises both when the
            pair falls short. Both leave slack: an axis measured against a
            near-exact partner is asked to carry error that the partner will in
            fact share. Re-searching each axis against its real partner
            recovers it, and since every probe measures the grid that would
            actually be built, the pair stays within tolerance throughout.
            """
            if not acceptable(width, height):
                return width, height
            for vary_width in (True, False):
                low = minimum
                high = width if vary_width else height
                while low < high:
                    middle = (low + high) // 2
                    candidate = (middle, height) if vary_width else (width, middle)
                    if acceptable(*candidate):
                        high = middle
                    else:
                        low = middle + 1
                if vary_width:
                    width = high
                else:
                    height = high
            return width, height

        width = first_acceptable(maximum, vary_width=True)
        height = first_acceptable(maximum, vary_width=False)
        while not acceptable(width, height) and (width < maximum or height < maximum):
            if width < maximum:
                width = min(maximum, max(width + 1, int(width * 1.25)))
            if height < maximum:
                height = min(maximum, max(height + 1, int(height * 1.25)))
        result = trim(width, height)

        self._geometry_resolution_limit_reached = not acceptable(*result)
        if self._geometry_resolution_limit_reached:
            warnings.warn(
                "Logical PN construction reached max_grid_resolution before "
                "meeting geometry_tolerance.",
                RuntimeWarning,
                stacklevel=3,
            )
        return result

    def _geometry_resolution_fingerprint(self, coord_function):
        """Cheaply fingerprint geometry without repeating the sizing search."""
        device = self.location.device
        dtype = self.location.dtype
        # Include domain boundaries plus nonuniform interior points. The
        # instance's pre-Surface state is also part of the key; this sample
        # catches closure, global, class-attribute, and Mob-basis inputs that
        # are not represented there.
        sample_uv = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.2113248654, 0.7886751346],
                [0.7886751346, 0.2113248654],
                [0.1270166538, 0.3819660113],
                [0.6180339887, 0.8729833462],
            ],
            device=device,
            dtype=dtype,
        )
        try:
            points = coord_function(sample_uv.clone())
            points = torch.as_tensor(points, device=device, dtype=dtype)
            points = points.detach().contiguous().cpu()
        except Exception:
            # Callable identity and explicit subclass state still provide a
            # conservative key. Unsupported functions will take the normal
            # search path on their first distinct key.
            return None
        return (
            str(points.dtype),
            tuple(points.shape),
            points.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )

    def _geometry_resolution_cache_key(
        self,
        coord_function,
        resolution_cache_state,
    ):
        """Build the cache key; subclasses may override for custom state."""
        geometry_fingerprint = self._geometry_resolution_fingerprint(coord_function)
        coord_function_key = _resolution_cache_callable_key(coord_function)
        if geometry_fingerprint is None:
            # If the cheap probe is incompatible with an unusual coordinate
            # function, prefer a harmless cache miss over sharing a result for
            # geometry that could not be identified.
            coord_function_key = (coord_function_key, id(self))
        return (
            self._geometry_tolerance,
            self._min_grid_resolution,
            self._max_grid_resolution,
            str(self.location.device),
            str(self.location.dtype),
            _freeze_resolution_cache_value(self.u_range),
            _freeze_resolution_cache_value(self.v_range),
            _freeze_resolution_cache_value(resolution_cache_state),
            coord_function_key,
            _resolution_cache_callable_key(self._compute_pn_geometry_error),
            _resolution_cache_callable_key(self._pn_geometry_deviation),
            geometry_fingerprint,
        )

    def _get_cached_geometry_resolution(
        self,
        surface_function,
        coord_function,
        resolution_cache_state,
    ):
        """Return this geometry's cached grid or run and memoize the search."""
        cache_key = self._geometry_resolution_cache_key(
            coord_function,
            resolution_cache_state,
        )
        cls = type(self)
        with cls._geometry_resolution_cache_lock:
            cached = cls._geometry_resolution_cache.get(cache_key)
            if cached is not None:
                if cache_key in cls._geometry_resolution_warning_keys:
                    warnings.warn(
                        "Logical PN construction reached max_grid_resolution before "
                        "meeting geometry_tolerance.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                return cached

            self._geometry_resolution_limit_reached = False
            result = tuple(self._find_geometry_resolution(surface_function))
            cls._geometry_resolution_cache[cache_key] = result
            if self._geometry_resolution_limit_reached:
                cls._geometry_resolution_warning_keys.add(cache_key)
            return result

    def _find_screen_space_resolution(self, surface_function):
        minimum = self._min_grid_resolution
        maximum = self._max_grid_resolution
        tolerance = self._resolution_tolerance

        def acceptable(width, height):
            try:
                error = self._compute_error(surface_function, width, height)
            except Exception:
                return False
            return bool(torch.isfinite(error).item() and error.item() <= tolerance)

        def first_acceptable(other, vary_width):
            low, high = minimum, maximum
            best = maximum
            while low <= high:
                middle = (low + high) // 2
                width, height = (middle, other) if vary_width else (other, middle)
                if acceptable(width, height):
                    best = middle
                    high = middle - 1
                else:
                    low = middle + 1
            return best

        width = first_acceptable(maximum, vary_width=True)
        height = first_acceptable(maximum, vary_width=False)
        while not acceptable(width, height) and (width < maximum or height < maximum):
            if width < maximum:
                width = min(maximum, max(width + 1, int(width * 1.25)))
            if height < maximum:
                height = min(maximum, max(height + 1, int(height * 1.25)))
        return width, height

    @staticmethod
    def _resample_grid_value(value, old_width, old_height, new_width, new_height):
        leading_shape = value.shape[:-2]
        channels = value.shape[-1]
        image = value.reshape(-1, old_width, old_height, channels).permute(0, 3, 1, 2)
        resized = F.interpolate(
            image,
            size=(new_width, new_height),
            mode="bilinear",
            align_corners=True,
        )
        return resized.permute(0, 2, 3, 1).reshape(
            *leading_shape, new_width * new_height, channels
        )

    def _capture_resolution_boundary_events(self):
        """Capture transform events that begin at a topology split.

        A transform recorded earlier in the same ``Sync`` starts at the
        instant ``detach_history`` replaces this surface. It must migrate to
        the replacement topology; moving it to the historical clone makes it
        invisible because that clone despawns at the same instant. The
        transform may be owned by this surface or by an ancestor whose
        recursive edit also contains rows belonging to other mobs.
        """
        timeline = self.scene.timeline_manager
        detach_time = self.animation_manager.context.timespan.current_time
        descendants = self.get_descendants()
        captured = []

        for event in timeline.function_timeline.function_applications:
            if event.time.start < detach_time:
                continue

            edits = []
            touches_surface = False
            # The EditRecord behind each entry is carried alongside it (see
            # FunctionApplicationEvent.recorded_edit_records), so it does not
            # have to be found by searching the attribute's whole edit log --
            # which made a topology split cost the length of the scene so far,
            # once per recorded edit of every event at the boundary.
            records = event.recorded_edit_records
            aligned = len(records) == len(event.recorded_edits)
            for position, (attr, mob_id, recursive, indexes) in enumerate(
                event.recorded_edits
            ):
                attr_timeline = timeline.attr_to_timeline[attr]
                if aligned:
                    source = records[position]
                else:
                    source = next(
                        (
                            edit
                            for edit in attr_timeline.edits
                            if edit.event is event and edit.indexes is indexes
                        ),
                        None,
                    )
                if source is None:
                    edits = None
                    break
                touches_surface = touches_surface or any(
                    mob.id in attr_timeline.mob_id_to_inds
                    and torch.any(
                        torch.isin(
                            source.indexes,
                            attr_timeline.mob_id_to_inds[mob.id],
                        )
                    )
                    for mob in descendants
                )
                edits.append(
                    {
                        "attr": attr,
                        "mob_id": mob_id,
                        "recursive": recursive,
                        "source": source,
                        "indexes": indexes.clone(),
                        "values": source.values.clone(),
                        "time": source.time,
                        "seq": source.seq,
                    }
                )
            if edits is not None and touches_surface:
                captured.append(
                    {
                        "event": event,
                        "caller": event.caller,
                        "edits": edits,
                    }
                )

        attrs = {
            edit["attr"]
            for captured_event in captured
            for edit in captured_event["edits"]
        }
        owner_rows = {
            (attr, mob.id): timeline.attr_to_timeline[attr]
            .mob_id_to_inds[mob.id]
            .clone()
            for attr in attrs
            for mob in descendants
            if mob.id in timeline.attr_to_timeline[attr].mob_id_to_inds
        }
        current_location = self.location.clone()
        current_basis = self.basis.clone()
        for captured_event in captured:
            captured_event["current_location"] = current_location
            captured_event["current_basis"] = current_basis
            captured_event["pre_location"] = current_location
            captured_event["pre_basis"] = current_basis
            for edit in captured_event["edits"]:
                if edit["attr"] not in ("location", "basis"):
                    continue
                surface_rows = owner_rows.get((edit["attr"], self.id))
                if surface_rows is None:
                    continue
                surface_mask = torch.isin(edit["indexes"], surface_rows)
                if not torch.any(surface_mask):
                    continue
                captured_event[f"pre_{edit['attr']}"] = edit["values"][:, surface_mask]
        return captured, owner_rows

    def _map_resolution_boundary_block(
        self,
        attr,
        owner,
        values,
        old_width,
        old_height,
        new_width,
        new_height,
        new_surface_points,
        captured_event,
    ):
        if owner is not self.grid:
            return values
        if values.shape[-2] != old_width * old_height:
            # Shared grid attributes such as ``basis`` retain a singleton row
            # regardless of the number of sampled vertices.
            return values

        if attr == "location":
            # Resolution changes accompany affine Mob transforms.  Map the new
            # analytic grid through the exact parent transform captured by the
            # event, avoiding a numerically unstable least-squares fit.
            new_points = new_surface_points.to(values)
            local_points = map_global_to_local_coords(
                captured_event["current_location"].to(values),
                captured_event["current_basis"].to(values),
                new_points,
            )
            return map_local_to_global_coords(
                captured_event["pre_location"].to(values),
                captured_event["pre_basis"].to(values),
                local_points,
            )

        return self._resample_grid_value(
            values, old_width, old_height, new_width, new_height
        )

    def _migrate_resolution_boundary_events(
        self,
        captured,
        owner_rows,
        old_width,
        old_height,
        new_surface_points,
    ):
        timeline = self.scene.timeline_manager
        surface_owners = self.get_descendants()

        for captured_event in captured:
            event = captured_event["event"]
            caller = captured_event["caller"]
            caller_descendants = (
                caller.get_descendants()
                if hasattr(caller, "get_descendants")
                else [caller]
            )
            id_to_target = {mob.id: mob for mob in caller_descendants}
            migrated_edits = []
            pending_records = []
            try:
                for edit in captured_event["edits"]:
                    attr = edit["attr"]
                    target = id_to_target[edit["mob_id"]]
                    attr_timeline = timeline.attr_to_timeline[attr]
                    new_indexes = target._get_attr_ranges(
                        attr, include_descendants=edit["recursive"]
                    ).tensor()
                    index_blocks = []
                    value_blocks = []
                    internal_old_mask = torch.zeros_like(
                        edit["indexes"], dtype=torch.bool
                    )

                    for owner in surface_owners:
                        old_owner_rows = owner_rows.get((attr, owner.id))
                        new_owner_rows = attr_timeline.mob_id_to_inds.get(owner.id)
                        if old_owner_rows is None or new_owner_rows is None:
                            continue
                        old_mask = torch.isin(edit["indexes"], old_owner_rows)
                        if not torch.any(old_mask):
                            continue
                        internal_old_mask |= old_mask
                        new_mask = torch.isin(new_indexes, new_owner_rows)
                        if not torch.any(new_mask):
                            raise ValueError(
                                "could not find replacement animation rows"
                            )
                        block = self._map_resolution_boundary_block(
                            attr,
                            owner,
                            edit["values"][:, old_mask],
                            old_width,
                            old_height,
                            self.grid_width,
                            self.grid_height,
                            new_surface_points,
                            captured_event,
                        )
                        owner_new_indexes = new_indexes[new_mask]
                        if block.shape[-2] != owner_new_indexes.numel():
                            raise ValueError(
                                "could not map animation rows to new surface resolution"
                            )
                        index_blocks.append(owner_new_indexes)
                        value_blocks.append(block)

                    if not torch.any(internal_old_mask):
                        migrated_edits.append(
                            (
                                attr,
                                edit["mob_id"],
                                edit["recursive"],
                                edit["source"].indexes,
                            )
                        )
                        continue

                    external_mask = ~internal_old_mask
                    if torch.any(external_mask):
                        index_blocks.append(edit["indexes"][external_mask])
                        value_blocks.append(edit["values"][:, external_mask])

                    combined_indexes = torch.cat(index_blocks)
                    combined_values = torch.cat(value_blocks, dim=-2)
                    order = torch.argsort(combined_indexes)
                    combined_indexes = combined_indexes[order]
                    combined_values = combined_values[:, order]
                    if not torch.equal(combined_indexes, new_indexes):
                        raise ValueError(
                            "incomplete animation-row migration for surface resolution"
                        )
                    record = EditRecord(
                        new_indexes,
                        combined_values,
                        edit["time"],
                        edit["seq"],
                        event,
                    )
                    pending_records.append(
                        (
                            attr,
                            attr_timeline,
                            edit["source"],
                            edit["indexes"][internal_old_mask],
                            edit["values"][:, internal_old_mask],
                            record,
                        )
                    )
                    migrated_edits.append(
                        (
                            attr,
                            edit["mob_id"],
                            edit["recursive"],
                            new_indexes,
                        )
                    )
            except (KeyError, RuntimeError, ValueError):
                # Leave an unsupported callback on the historical topology
                # rather than partially migrating its row history.
                continue

            for (
                attr,
                attr_timeline,
                source,
                historical_indexes,
                historical_values,
                record,
            ) in pending_records:
                # The old surface remains visible before the topology split,
                # so retain its portion of the original edit for that
                # historical clone. External rows move exclusively into the
                # replacement record to avoid duplicating an ancestor edit.
                source.indexes = historical_indexes
                source.values = historical_values
                source.replay_end = None
                timeline.register_migrated_edit(attr, attr_timeline, source, record)
                attr_timeline.invalidate_prepared_queries()
            event.recorded_edits = migrated_edits
            # Through the timeline, not by assignment: the caller index has to
            # follow the move (see FunctionTimeline.retarget_caller).
            timeline.function_timeline.retarget_caller(event, captured_event["caller"])
            event.replay_end = None

    def _change_resolution(self, grid_width, grid_height, surface_function=None):
        grid_width = int(grid_width)
        grid_height = int(grid_height)
        if (grid_width, grid_height) == (self.grid_width, self.grid_height):
            return self
        if surface_function is None:
            # Built here and evaluated below, on the far side of
            # detach_history. That is only sound because nothing in between
            # touches this Mob's transform, which the evaluator reads live --
            # see _current_surface_function.
            surface_function = self._current_surface_function()

        old_width, old_height = self.grid_width, self.grid_height
        old_count = old_width * old_height
        boundary_events = []
        boundary_owner_rows = {}
        old_values = {}
        for attr in dict.fromkeys(self.grid.animatable_attrs):
            try:
                old_values[attr] = getattr(self.grid, attr).clone()
            except AttributeError:
                continue

        self._resolution_update_in_progress = True
        try:
            # Unspawned mobs have no interpolation history to detach. Once a
            # surface has spawned (including one that was later despawned),
            # detach_history transfers the old topology to a frozen clone and
            # leaves this surface with fresh timeline rows for the new shape.
            if self.is_spawned():
                boundary_events, boundary_owner_rows = (
                    self._capture_resolution_boundary_events()
                )
                self.detach_history()
            self.grid_width = grid_width
            self.grid_height = grid_height
            self.resolution = (grid_width - 1, grid_height - 1)
            self.__dict__.pop("_cached_base_grid", None)
            self.__dict__.pop("_cached_base_grid_key", None)

            base_grid = self.get_base_grid()
            new_location = squish(surface_function(base_grid), -3, -2)
            if new_location.dim() == 2:
                new_location = new_location.unsqueeze(0)
            self.grid._setattr_and_rebatch_without_record("location", new_location)

            for attr, value in old_values.items():
                if attr == "location":
                    continue
                if value.shape[-2] == old_count:
                    value = self._resample_grid_value(
                        value,
                        old_width,
                        old_height,
                        grid_width,
                        grid_height,
                    )
                self.grid._setattr_and_rebatch_without_record(attr, value)

            self.grid.batch_size = grid_width * grid_height
            self._memory_per_timestep_cache = None
            if boundary_events:
                new_surface_points = surface_function(base_grid).reshape(-1, 3)
                self._migrate_resolution_boundary_events(
                    boundary_events,
                    boundary_owner_rows,
                    old_width,
                    old_height,
                    new_surface_points,
                )
        finally:
            self._resolution_update_in_progress = False
        return self

    def _update_resolution_for_current_shape(self, allow_upsample=True):
        if not self._can_update_resolution():
            return self
        surface_function = self._current_surface_function()
        required_width, required_height = self._find_screen_space_resolution(
            surface_function
        )
        width, height = self._select_auto_resolution(
            required_width,
            required_height,
            allow_upsample=allow_upsample,
        )
        if (width, height) == (self.grid_width, self.grid_height):
            return self
        return self._change_resolution(width, height, surface_function)

    def _refine_sampling_for_color_wave(self, direction, max_spacing, pulsed_attrs):
        """Refine the grid so a travelling color wave crosses it smoothly.

        A surface's grid resolution is chosen to fit its *shape*, so a surface
        that is flat along one axis is sampled only at the ends of it -- enough
        to draw the geometry, far too coarse to draw a color band moving across
        it. Both color and opacity are carried per grid vertex, so either kind
        of wave benefits. Re-sample the parameter grid finely enough that
        neighbouring vertices are no further than ``max_spacing`` apart along the
        wave, by running ``coord_function`` over a denser ``(u, v)`` grid (see
        :meth:`~.Mob._refine_sampling_for_color_wave`).
        """
        if (
            not hasattr(self, "grid")
            or self._resolution_update_in_progress
            or self.data_sub_inds is not None
            or self.grid.data_sub_inds is not None
        ):
            return None
        if self._has_color_texture:
            # Textured surfaces shade from the texture map rather than from
            # vertex colors, so a finer grid would buy geometry and no pixels.
            return None
        width, height = self.grid_width, self.grid_height
        location = self.grid.location
        if location.shape[-2] != width * height:
            # A grid that has been rebatched away from its parameter domain
            # (imported meshes, indexed views) cannot be regenerated from uv.
            return None
        projected = dot_product(direction, location, dim=-1, keepdim=False).reshape(
            -1, width, height
        )[0]

        maximum = min(self._max_grid_resolution, _MAX_WAVE_GRID_RESOLUTION)

        def refined(size, gaps):
            if size < 2:
                return size
            spacing = gaps.abs().amax().item()
            if spacing <= max_spacing:
                return size
            required = int(np.ceil((size - 1) * spacing / max_spacing)) + 1
            return max(size, min(required, maximum))

        new_width = refined(width, projected[1:, :] - projected[:-1, :])
        new_height = refined(height, projected[:, 1:] - projected[:, :-1])
        if (new_width, new_height) == (width, height):
            return None
        self._change_resolution(new_width, new_height)

        def restore():
            self._change_resolution(width, height)

        return restore

    def _compute_error(self, coord_function, W, H):
        """Max screen-space deviation in pixels for a ``W x H`` grid.

        A surface is drawn as flat triangles (logical PN patches dice into them
        per frame), so the resolution must make the *flat* mesh approximate the
        surface to within tolerance.
        """
        return self._compute_flat_error(coord_function, W, H)

    def _compute_flat_error(self, coord_function, W, H):
        """Max deviation between the flat-triangle mesh and the true surface,
        sampled at a fixed set of barycentric coordinates per triangle.

        Each triangle is approximated by the flat plane through its three
        corners (linear interpolation of the corner positions).
        """
        device = self.location.device
        grid_u = torch.linspace(0, 1, W, device=device)
        grid_v = torch.linspace(0, 1, H, device=device)
        grid_uu, grid_vv = torch.meshgrid(grid_u, grid_v, indexing="ij")
        base_grid = torch.stack([grid_uu, grid_vv], dim=-1)

        grid_points = coord_function(base_grid.clone())

        triangle_uvs = grid_to_triangle_vertices(base_grid)
        triangle_corners = grid_to_triangle_vertices(grid_points)

        corners_3d = triangle_corners.reshape(-1, 3, 3)
        uvs_2d = triangle_uvs.reshape(-1, 3, 2)

        bary_coords = torch.tensor(
            [
                [1 / 3, 1 / 3],
                [1 / 2, 0.0],
                [0.0, 1 / 2],
                [1 / 2, 1 / 2],
                [1 / 6, 1 / 6],
                [1 / 6, 2 / 3],
                [2 / 3, 1 / 6],
            ],
            device=device,
        )

        u = bary_coords[:, 0].view(1, -1, 1)
        v = bary_coords[:, 1].view(1, -1, 1)
        w = 1.0 - u - v

        # Flat triangle: linear (barycentric) interpolation of the corners.
        p0 = corners_3d[:, 0, :].unsqueeze(1)
        p1 = corners_3d[:, 1, :].unsqueeze(1)
        p2 = corners_3d[:, 2, :].unsqueeze(1)
        S_points = w * p0 + u * p1 + v * p2

        uv0 = uvs_2d[:, 0, :].unsqueeze(1)
        uv1 = uvs_2d[:, 1, :].unsqueeze(1)
        uv2 = uvs_2d[:, 2, :].unsqueeze(1)
        uv_true = w * uv0 + u * uv1 + v * uv2
        P_true = coord_function(uv_true.clone())

        return self._screen_space_error(S_points, P_true)

    @staticmethod
    def _normalize_texture_shape(tex, channels):
        """Normalize a user-supplied texture to ``[T, W, H, channels]``. See
        :func:`~algan.rendering.shaders.materials._as_texture_stack`, which is
        shared with the maps a material forwards.
        """
        from algan.rendering.shaders.materials import _as_texture_stack

        return _as_texture_stack(tex, channels)

    def _build_material_texture(self, textures_dict):
        """Combine per-property maps into one ``[T, W, H, 5]`` material
        texture plus its channel bitmask, on this surface's device. See
        :func:`~algan.rendering.shaders.materials._pack_material_texture`,
        which is shared with the maps a material forwards.
        """
        from algan.rendering.shaders.materials import _pack_material_texture

        return _pack_material_texture(textures_dict, self.location.device)

    def _rebuild_material_texture(self):
        """Repack ``self._material_prop_textures`` into the material texture."""
        self.material_texture, self.material_texture_flags = (
            self._build_material_texture(self._material_prop_textures)
        )

    def _can_accept_material_textures(self):
        """A surface generates UVs from its parameter grid, so it can always
        sample a map.
        """
        return True

    def _accept_material_textures(self, maps):
        """Take a material's texture maps onto this surface's own map slots.

        See
        :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin._accept_material_textures`.
        A surface generates UVs from its parameter grid, so every map a
        material can forward has a home here: ``map`` becomes
        :attr:`color_texture`, ``normal_map`` becomes ``normal_texture``, and
        the property maps join the packed material texture.
        """
        from algan.rendering.shaders.materials import _MAP_SLOT_PROPERTIES

        device = self.location.device
        # color_texture is a real animatable attribute; the other two are plain
        # fields read at primitive build, hence the animatable flag per slot.
        applied = {}
        if "map" in maps:
            self.color_texture = maps["map"].to(device)
            applied["map"] = True
        if "normal_map" in maps:
            self.normal_texture = self._normalize_texture_shape(
                maps["normal_map"], 3
            ).to(device)
            applied["normal_map"] = False
        properties = {
            name: maps[slot]
            for slot, (name, _) in _MAP_SLOT_PROPERTIES.items()
            if slot in maps
        }
        if properties:
            # Rebound rather than mutated in place, so a subclass that never
            # reached Surface.__init__ still merges correctly instead of
            # raising (and never shares one dict between instances).
            merged = dict(getattr(self, "_material_prop_textures", {}))
            merged.update(properties)
            self._material_prop_textures = merged
            self._rebuild_material_texture()
            applied.update(
                {slot: False for slot in _MAP_SLOT_PROPERTIES if slot in maps}
            )
        return applied

    def _bake_texture_to_grid(self, tex, channels=1):
        """Resample a texture to the surface grid resolution and flatten it to
        per-vertex values ``[W*H, channels]`` (the same bake the color path
        applies to grid-resolution textures).
        """
        t = self._normalize_texture_shape(tex, channels).to(self.location.device)
        if t.shape[0] != 1:
            raise ValueError(
                "glow textures must be static (no time "
                f"dimension), got {tuple(t.shape)}"
            )
        t = F.interpolate(
            t.permute(0, 3, 1, 2),
            size=(self.grid_width, self.grid_height),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        return squish(t, -3, -2).squeeze(0)

    def _get_memory_used_per_timestep(self) -> int:
        """Get this surface's render memory cost for one frame, in bytes.

        Grows with the surface's grid resolution, and is what the render loop uses to
        decide how many frames fit in a batch -- a high-resolution surface therefore
        renders in smaller batches.

        Returns
        -------
        int
            Estimated bytes needed per frame.
        """
        # Called once per surface per render batch just to size batches; the
        # result only depends on grid size and shader-param widths (fixed per
        # material), so cache it. Reading the shader params goes through the
        # animated-attribute machinery, which is the expensive part.
        names = tuple(getattr(self, "shader_specific_param_names", ()))
        n_grid = self.grid.location.shape[-2]
        packed_grid_count = self._packed_grid_count()
        rendered_grid_count = 1 if packed_grid_count is None else packed_grid_count
        # Computed before the cache test, and compared as part of the key: a
        # texture can be assigned (or its resolution changed) after this surface
        # was first priced, and a key that does not carry it would keep serving
        # the texture-free estimate forever.
        texture_bytes = self._color_texture_bytes_per_timestep()
        key = (n_grid, rendered_grid_count, names, texture_bytes)
        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        n_tri = (
            rendered_grid_count
            * 2
            * max(self.grid_height - 1, 1)
            * max(self.grid_width - 1, 1)
        )
        n_v = n_tri * 3
        # Grid animation state: location(3*4) + color(5*4) = 32 bytes per grid point.
        # Normal computation intermediates (grid rolls, stack, cross products):
        # ~150 bytes per grid point peak.
        animation_and_intermediates = n_grid * 182
        # Source-device primitive output retained through scene preparation:
        # corners(3*3*4=36) + colors(3*5*4=60, cloned) + normals(3*3*4=36) = 132 bytes
        # per triangle, plus RT frame bounds ~8 bytes/vertex.
        primitive_bytes = n_v * 52
        # BVH and packed render geometry are no longer estimated here: the
        # finished unique storages are copied into and charged to ManualMemory.
        # Shader params broadcast to vertices.
        shader_bytes = 0
        for _ in self.get_shader_params().values():
            shader_bytes += n_v * _.shape[-1] * 4
        result = int(
            animation_and_intermediates + primitive_bytes + shader_bytes + texture_bytes
        )
        self._memory_per_timestep_cache = (key, result)
        return result

    def _segment_stack_u8_ok(self, seg):
        """u8 provenance of a segment window's endpoint stack.

        The endpoints are authored maps (edit-log snapshots up to the ulp the
        recorded write's round trip costs), so the proof usually holds -- but
        it is proved on the ACTUAL stack, because any of the maps ever
        assigned can appear as an endpoint and the per-assignment stamp only
        describes the latest. One image pass per stack, on the animation
        device (the edit log lives there), memoized against the stack's
        cache key -- a window that runs past the last edit reads the mutable
        current state and is proved fresh each batch.
        """
        memo = getattr(self, "_texture_lerp_u8_memo", None)
        if seg.cache_key is not None and memo is not None and memo[0] == seg.cache_key:
            return memo[1]
        ok = texture_u8_provenance(
            seg.endpoints.view(
                seg.endpoints.shape[0],
                int(self.texture_height),
                int(self.texture_width),
                5,
            )
        )
        if seg.cache_key is not None:
            self._texture_lerp_u8_memo = (seg.cache_key, ok)
        return ok

    def _color_texture_bytes_per_timestep(self) -> int:
        """This surface's color-texture cost for one frame, in bytes.

        ``color_texture`` is an ordinary animated attribute whose channel width
        is the whole flattened image (``H * W * 5``), so the timeline
        materializes a full copy of it for every frame of a batch, and the
        legacy premultiply arm of :meth:`get_render_primitives` keeps one more
        copy of the same width (under texture_opacity_in_kernel, the default,
        the opacity rides the primitive as scalars and that copy does not
        exist). Every other term in
        :meth:`_get_memory_used_per_timestep` is per grid point or per triangle,
        and a textured Surface has almost none of either -- so without this the
        batch sizer prices a 1774x887 image at a few kilobytes per frame and
        puts an entire video in a single batch.

        A closed surface's map is wrap-padded (:func:`wrap_pad_texture`) on its
        way to the renderer, which is a third copy, live while the premultiply
        clones off it. Whether a surface closes is a property of its animated
        geometry, so it is read off the previous primitive build -- the first
        batch of a job prices a globe as if it were a plane, and the
        out-of-render-memory retry is what covers that.

        Rows orphaned by :meth:`~algan.animatable_base.mob.Mob.detach_history`
        are materialized too but belong to no live Mob, so they are not
        attributed here; the animation memory fraction absorbs them.

        Returns
        -------
        int
            Estimated bytes needed per frame, or 0 when untextured.
        """
        attr = getattr(self, "_color_texture_attr", None)
        if attr is None:
            return 0
        timeline = self.scene.timeline_manager.attr_to_timeline.get(attr)
        if timeline is not None and timeline.materialize_device is not None:
            # The window lives on the render device; priced there instead
            # (_get_render_device_memory_used_per_timestep). What stays on
            # the animation device is the edit log, which is per render job,
            # not per frame.
            return 0
        inds = None if timeline is None else timeline.mob_id_to_inds.get(self.id)
        rows = 1 if inds is None else int(inds.numel())
        texels = int(self.texture_height) * int(self.texture_width) * 5
        from algan.rendering.raytracing import settings as _rts

        if _rts.texture_opacity_in_kernel_active():
            # No premultiply copy on this path (texture_opacity_in_kernel):
            # per frame the animation device holds the materialized window
            # plus, on a closed surface, the wrap pad's copy of it.
            copies = 2 if getattr(self, "_texture_is_wrap_padded", False) else 1
        else:
            copies = 3 if getattr(self, "_texture_is_wrap_padded", False) else 2
        if getattr(self, "_texture_window_collapsed", False) or getattr(
            self, "_texture_window_lerp", False
        ):
            # The previous build proved the window constant and collapsed it
            # (texture_window_collapse), so the premultiply/pad copies are per
            # batch rather than per frame; only the materialized window itself
            # still scales with the frame count. Priced off the previous
            # build, like ``_texture_is_wrap_padded`` above -- a texture that
            # STARTS animating re-prices dense one batch late, covered by the
            # out-of-render-memory retry. Under texture_opacity_in_kernel the
            # collapse fires on texel constancy alone, so a FADE of a static
            # image prices at the window too -- which is what lets its batch
            # windows lengthen (the point of the in-sampler multiply). A
            # segment-described window (texture_time_lerp) never materializes
            # at all and its endpoint/pad/merge copies are per batch, so it
            # prices the same envelope.
            copies = 1
        return rows * texels * 4 * copies

    def _get_render_device_memory_used_per_timestep(self) -> int:
        """This surface's color-texture cost for one frame on the render device.

        Zero unless the texture's frame window materializes there (see
        :meth:`_color_texture_bytes_per_timestep`, which then prices it at
        zero on the animation device). Per frame, per row, the render device
        holds the materialized window itself; the replayed assignment's lerp
        (``base + change * a``, two transients at its peak); the premultiplied
        map :meth:`get_render_primitives` builds; the linear-light decode and
        the concatenation the scene merge makes of it; and the arena copy the
        render reads -- so the estimate counts six images, which is the peak of
        that sequence with the transients released between steps and a margin
        for the wrap padding a closed surface adds.
        """
        attr = getattr(self, "_color_texture_attr", None)
        if attr is None:
            return 0
        timeline = self.scene.timeline_manager.attr_to_timeline.get(attr)
        if timeline is None or timeline.materialize_device is None:
            return 0
        inds = timeline.mob_id_to_inds.get(self.id)
        rows = 1 if inds is None else int(inds.numel())
        texels = int(self.texture_height) * int(self.texture_width) * 5
        from algan.rendering.raytracing import settings as _rts

        # One image of the six-image chain is the host premultiply, absent
        # under texture_opacity_in_kernel (the opacity rides the bank as
        # per-frame scalars instead).
        factor = 5 if _rts.texture_opacity_in_kernel_active() else 6
        if getattr(self, "_texture_window_collapsed", False):
            # The previous build collapsed the window's downstream copies to
            # one frame (texture_window_collapse): the per-frame residue is
            # the materialized window itself, plus margin for the handful of
            # per-batch images (premultiplied map, decode, merge concat,
            # arena copy) the collapse amortizes. Same read-off-the-previous-
            # build contract as the animation-device estimate. Under
            # texture_opacity_in_kernel a fade of a static image lands here
            # too (the collapse keys on texel constancy alone), which is what
            # lengthens its batch windows.
            factor = 2
        if getattr(self, "_texture_window_lerp", False):
            # A segment-described window (texture_time_lerp): its rows are
            # excluded from materialization, so the render device holds NO
            # per-frame image at all -- only the per-batch endpoint stack and
            # its decode/concat/arena copies, priced as the same one-image
            # margin the collapse keeps. The out-of-render-memory retry
            # backstops a window whose endpoint count grows unusually large.
            factor = 1
        return rows * texels * 4 * factor

    def _packed_grid_count(self):
        """Number of independent grids concatenated into ``self.grid``.

        ``batch_mobs`` keeps child ownership in ``parent_batch_sizes`` while
        concatenating the child's animatable rows.  For a packed collection of
        surfaces (point-cloud spheres are the main example), every entry must
        therefore describe exactly one complete surface grid.  Treating the
        concatenation as a single wider grid creates triangles between adjacent
        objects.
        """
        batch_sizes = self.grid.parent_batch_sizes
        if batch_sizes is None:
            return None

        points_per_grid = self.grid_width * self.grid_height
        flat_sizes = batch_sizes.detach().cpu().reshape(-1)
        if flat_sizes.numel() == 0:
            raise ValueError("packed surface grid has no parent batch sizes")
        if bool((flat_sizes != points_per_grid).any()):
            raise ValueError(
                "packed surface grid entries must each contain exactly "
                f"grid_width * grid_height = {points_per_grid} points, got "
                f"{flat_sizes.tolist()}"
            )
        if int(flat_sizes.sum()) != self.grid.location.shape[-2]:
            raise ValueError(
                "packed surface grid sizes do not match the materialized grid: "
                f"{int(flat_sizes.sum())} != {self.grid.location.shape[-2]}"
            )
        return flat_sizes.numel()

    def _reshape_grid_for_render(self, values):
        """Restore flat grid rows to one or more independent surface grids.

        Also applies :attr:`_grid_orientation`, which is why every consumer of
        the grid's *orientation* -- vertex normals, triangle winding, the morph
        soup -- reaches it through here.
        """
        packed_grid_count = self._packed_grid_count()
        if packed_grid_count is None:
            grid = unsquish(values, -2, self.grid_height)
        else:
            grid = values.reshape(
                *values.shape[:-2],
                packed_grid_count,
                self.grid_width,
                self.grid_height,
                values.shape[-1],
            )
        if self._grid_orientation < 0:
            # Reverse the v axis: the same sampled points in the opposite
            # order, which is the whole point -- it costs no geometry and
            # flips the handedness of (u, v), hence of every normal and
            # every triangle's winding.
            grid = grid.flip(-2)
        return grid

    def _flatten_packed_triangle_vertices(self, values):
        """Merge the packed-grid and triangle-vertex axes after gathering."""
        if self._packed_grid_count() is None:
            return values
        return values.flatten(-3, -2)

    def get_render_primitives(self):
        """Build the triangles the renderer draws this surface from.

        Called once per render batch. Vertex normals are computed from the grid unless
        ``ignore_normals`` is set, which is what makes a deformed surface light
        correctly without the author supplying normals.

        Returns
        -------
        :class:`~algan.rendering.primitives.triangle_primitive.TrianglePrimitive`
            The surface's triangles for every frame of the batch.
        """
        # Read uncopied: the grid only feeds reshapes and out-of-place
        # gathers below (see get_render_primitives_batched for the same
        # read on the batched path).
        grid = self._reshape_grid_for_render(
            self.grid.get_animated_attribute("location", copy=False)
        )
        weld = surface_weld_flags(grid)
        if not self.ignore_normals:
            vertex_normals = grid_to_triangle_vertices(
                compute_grid_vertex_normals(grid), weld
            )
        else:
            vertex_normals = None
        return self._build_render_primitive(grid, vertex_normals, weld=weld)

    def _build_render_primitive(
        self, grid, vertex_normals, precomputed_corners=None, weld=None
    ):
        """Assemble the
        :class:`~algan.rendering.primitives.triangle_primitive.TrianglePrimitive`
        for this surface from an already-materialized grid ``[T, W, H, 3]``
        and (triangle-gathered)
        vertex normals. ``precomputed_corners`` lets the batched path pass in
        corners gathered on the whole surface stack at once.
        """
        weld = weld or (False, False, False)

        corners = self._flatten_packed_triangle_vertices(
            grid_to_triangle_vertices(grid, weld)
            if precomputed_corners is None
            else precomputed_corners
        )
        if corners.shape[-2] == 0:
            # A surface with no extent tessellates to no triangles at all --
            # ``Sphere(radius=0)``, a radius that a calculation drove to zero,
            # a degenerate ``u_range``. There is nothing to draw, and the
            # callers already treat ``None`` as "this actor contributes no
            # geometry". Built anyway, the empty primitive reached
            # ``broadcast_all`` against one row of color and failed the whole
            # render with a tensor-shape error.
            return None

        def expand_grid_to_verts(x):
            # Same weld as the corners: a pole weld drops triangles, so every
            # per-vertex attribute has to be gathered through the same index
            # list or the primitive's arrays disagree on length.
            if x.shape[-2] == 1:
                # A per-surface constant -- every material parameter arrives
                # this way -- reads the same at every triangle vertex, so the
                # expand-then-gather below only copied it W*H-fold and then
                # vertex-fold. Broadcast it straight to the vertex layout
                # instead: bit-identical (a gather of a broadcast is the
                # broadcast), and it removes the ~1 ms gather per parameter
                # per surface that was the largest item of a batch's primitive
                # build on the reference neural-net scene.
                return x.expand(*x.shape[:-2], corners.shape[-2], x.shape[-1])
            x = self._reshape_grid_for_render(x)
            return self._flatten_packed_triangle_vertices(
                grid_to_triangle_vertices(x, weld)
            )

        def compute_grid_color():
            # Plain tensor, not the Color subclass the public property returns:
            # geometry building only does arithmetic on these numbers, and a
            # subclass sends every operation through __torch_function__ (see
            # the matching note in BezierCircuitCubic.get_render_primitives).
            # Read uncopied and take the one defensive copy here: the values
            # are mutated in place just below, and the public read's own clone
            # would make it two.
            grid_color = self.grid.get_animated_attribute("color", copy=False).clone()
            grid_color[..., -1:] *= self.grid.opacity
            grid_color[..., -2:-1] += self.grid.glow
            return grid_color

        uvs = None
        texture_map = None
        texture_opacity = None
        closed_axes = (False, False)
        material_texture_map = getattr(self, "material_texture", None)
        material_texture_flags = getattr(self, "material_texture_flags", 0)
        normal_texture_map = getattr(self, "normal_texture", None)
        has_color_texture = self._has_color_texture
        if (
            has_color_texture
            or material_texture_map is not None
            or normal_texture_map is not None
        ):
            # Generate UV coordinates for the triangle corners from the base grid.
            # The POLE welds apply (they change the triangle list, so uvs must
            # match it), but the u-seam wrap deliberately does NOT: wrapping it
            # would give the last cell column u = 0 where the texture needs
            # u = 1, running the map backwards across that column. The duplicate
            # uv column exists precisely to carry that discontinuity, and it is
            # the one thing the position weld must not take away.
            base_grid = self.get_base_grid()
            uv_weld = (False, weld[1], weld[2])
            uvs = grid_to_triangle_vertices(base_grid, uv_weld)
            packed_grid_count = self._packed_grid_count()
            if packed_grid_count is not None:
                uvs = uvs.unsqueeze(0).expand(packed_grid_count, -1, -1).flatten(0, 1)
            uvs = uvs.unsqueeze(0)  # [1, num_triangles * 3, 2]
            # u = 0 and u = 1 are the same place on a closed surface, so every
            # map sampled against these uvs has to wrap there rather than
            # clamp. wrap_pad_texture makes the clamping sampler do that.
            closed_axes = surface_closed_axes(grid)
            self._texture_is_wrap_padded = any(closed_axes)
            material_texture_map = wrap_pad_texture(material_texture_map, closed_axes)
            normal_texture_map = wrap_pad_texture(normal_texture_map, closed_axes)
        if has_color_texture:
            from algan.rendering.raytracing import settings as _rts

            # A window described as segments (texture_time_lerp): K endpoint
            # images plus per-frame (i0, i1, w) stand in for the dense
            # window, which the timeline then never materialized. Keyed on
            # the DESCRIPTION's presence, not the setting, so a mid-batch
            # toggle cannot desynchronize this build from the
            # materialization that fed it.
            seg = self.scene.timeline_manager.segment_window_for(
                self._color_texture_attr, self.id
            )
            opacity = self.opacity.unsqueeze(-2)
            texture_lerp = None
            seg_u8 = False
            if seg is not None:
                texels = seg.endpoints.unsqueeze(1)  # [K, 1, W*H*5]
                # The gate only runs while the in-sampler opacity multiply is
                # active (texture_time_lerp_active), so this build hands the
                # opacity to the sampler like any other window of the batch.
                op_in_kernel = True
                collapsed = texels.shape[0] == 1
                seg_u8 = self._segment_stack_u8_ok(seg)
                if not collapsed:
                    texture_lerp = torch.stack(
                        (
                            seg.index0.to(torch.float32),
                            seg.index1.to(torch.float32),
                            seg.weights.float(),
                        ),
                        -1,
                    )
                self._texture_window_collapsed = collapsed
            else:
                # Read the texels once, uncopied: this is the widest attribute
                # in the engine, and mult_opacity is out-of-place
                # (Color.prep_set clones), so the materialized state is never
                # written through. Pad as a plain tensor, before mult_opacity,
                # so the cat stays off Color's __torch_function__.
                texels = self._color_texture_uncopied()
                # The window may live on the render device (a wide attribute,
                # see AttributeTimeline.materialize_device) while opacity is
                # an ordinary animation-device attribute.
                if opacity.device != texels.device:
                    opacity = opacity.to(texels.device)
                # The timeline materializes the window dense -- one image per
                # frame whether or not anything edited it -- and every copy
                # below (wrap pad, premultiply, and the merge's
                # decode/concat/upload downstream) used to be made per frame.
                # When the window and the opacity are byte-identical across
                # the batch, one frame carries it: every consumer reads
                # texture time as ``f % shape[0]``
                # (rt_settings.texture_window_collapse kills this for
                # byte-level A/B). Opacity first -- it is a handful of floats
                # against a full image pass.
                #
                # With the in-sampler multiply the opacity never touches the
                # texels, so the collapse keys on texel constancy ALONE -- a
                # fade of a static image keeps its one-frame map, which is
                # the point of texture_opacity_in_kernel (the premultiply
                # welded the fade into the widest attribute in the engine).
                op_in_kernel = _rts.texture_opacity_in_kernel_active()
                collapsed = (
                    _rts.texture_window_collapse
                    and texels.shape[0] > 1
                    and (
                        op_in_kernel
                        or opacity.shape[0] == 1
                        or bool((opacity[1:] == opacity[:1]).all())
                    )
                    and bool((texels[1:] == texels[:1]).all())
                )
                if collapsed:
                    texels = texels[:1]
                    if not op_in_kernel:
                        opacity = opacity[:1]
                # Observed constancy, read by the batch sizer for the NEXT
                # batch (the same read-off-the-previous-build pattern as
                # ``_texture_is_wrap_padded``): a collapsed window's
                # premultiply / pad / decode / merge copies are per batch, not
                # per frame, so the texture prices at roughly the materialized
                # window alone. Gated on the toggle so
                # texture_window_collapse=0 restores the legacy per-frame
                # pricing along with the legacy copies.
                self._texture_window_collapsed = bool(
                    _rts.texture_window_collapse
                ) and (collapsed or texels.shape[0] == 1)
            # Observed by the batch sizer, like _texture_window_collapsed: a
            # described window prices at its endpoints, not at one image per
            # frame.
            self._texture_window_lerp = texture_lerp is not None
            texture_map = wrap_pad_texture(
                texels.view(
                    texels.shape[0],
                    self.texture_height,
                    self.texture_width,
                    5,
                ),
                closed_axes,
            )
            if texture_lerp is not None:
                # [1, K, H, W, 5]: the leading singleton declares the stack
                # frame-static to everything that treats axis 0 as batch time
                # (slice_time_window would otherwise slice the endpoint axis
                # whenever K happened to equal the batch's frame count); the
                # K endpoints are addressed through texture_lerp instead.
                texture_map = texture_map.unsqueeze(0)
            if op_in_kernel:
                # The opacity rides the primitive as per-frame scalars (the
                # merge appends them as a tiny bank region the sampler
                # reads); the map itself stays the authored texels, which is
                # also what keeps them k/255 for texture_u8_storage.
                # mult_opacity's out-of-place copy used to decouple the
                # primitive from the timeline window; without it a collapsed
                # frame would be a VIEW pinning the whole T-frame window past
                # release_wide_windows, so clone the one frame (1/T of the
                # copy this path removes). An uncollapsed (animating) window
                # is deliberately left aliased: cloning it would be the very
                # T-frame copy this path exists to avoid.
                # A segment window's endpoint stack is already a standalone
                # copy (the gate stacks it out of the edit log), so only the
                # dense path needs the decoupling clone.
                if (
                    seg is None
                    and collapsed
                    and (
                        texture_map.untyped_storage().data_ptr()
                        == texels.untyped_storage().data_ptr()
                    )
                ):
                    texture_map = texture_map.clone()
                texture_opacity = opacity.reshape(opacity.shape[0], -1)[:, :1].reshape(
                    -1
                )
            else:
                texture_map = texture_map.as_subclass(Color).mult_opacity(opacity)
                texture_opacity = None

        colors = expand_grid_to_verts(compute_grid_color())
        normals = (
            None
            if vertex_normals is None
            else self._flatten_packed_triangle_vertices(vertex_normals)
        )

        from algan.rendering.raytracing.primitives import (
            LogicalPNTrianglePrimitive,
        )

        primitive = LogicalPNTrianglePrimitive(
            corners=corners,
            colors=colors,
            normals=normals,
            glow=colors[..., -2:-1].as_subclass(torch.Tensor),
            shader=self.shader,
            uvs=uvs,
            texture_map=texture_map,
            material_texture_map=material_texture_map,
            material_texture_flags=material_texture_flags,
            normal_texture_map=normal_texture_map,
            render_tolerance_pixels=self._render_tolerance_pixels,
            geometry_slack_ratio=self._geometry_slack_ratio,
            **{
                k: expand_grid_to_verts(v)
                for k, v in self.grid.get_shader_params().items()
            },
        )
        if texture_opacity is not None:
            # In-sampler opacity + u8 provenance for the color map; see the
            # texture block above. Post-construction assignment, like
            # ``mesh_ids`` below (the collection wrapper picks both up from
            # the member carrying the map). The provenance only holds while
            # the map is NOT premultiplied, which texture_opacity's presence
            # certifies.
            primitive.texture_opacity = texture_opacity
            # A described window proves provenance on its OWN endpoints (any
            # of the maps ever assigned can appear in a window); the dense
            # path uses the AND over every assignment at this resolution, for
            # the same reason.
            primitive.texture_u8_ok = (
                seg_u8
                if seg is not None
                else bool(getattr(self, "_color_texture_u8_ok_all", False))
            )
            primitive.texture_lerp = texture_lerp
        # A packed collection (point-cloud spheres are the main case) flattens
        # several INDEPENDENT grids into one primitive, so "one member = one
        # surface" would union every packed sphere into a single surface and let
        # the analytic-AA run rule sum coverage across objects that merely
        # overlap. Declare one shell per packed grid; the flatten keeps each
        # grid's triangles contiguous, so this is a repeat_interleave.
        packed = self._packed_grid_count()
        if packed is not None and packed > 1:
            per_grid = (corners.shape[1] // 3) // packed
            primitive.mesh_ids = torch.arange(
                packed, dtype=torch.int32, device=corners.device
            ).repeat_interleave(per_grid)
        # A solid built from several Surfaces -- a capped Cylinder is a tube
        # plus two discs -- says so by giving every part the same ``_mesh_key``,
        # which merges them into one surface for the analytic-AA run rule
        # instead of leaving each joint a boundary between two (see
        # ``primitives._mesh_ids_from_collection``). Only consecutive members
        # merge, which the authored draw order provides: it walks each tree
        # parent-first, so a part's own caps follow it.
        if getattr(self, "_mesh_key", None) is not None:
            primitive.mesh_key = self._mesh_key
        # A plain Surface is a two-sided sheet; the shapes of revolution built
        # on it declare an outside (Mob.two_sided). A solid among them
        # (:class:`~algan.mobs.shapes_3d.Sphere` with full ranges, a capped
        # :class:`~algan.mobs.shapes_3d.Cylinder`, ...) also declares its
        # triangles a closed shell, which is what lets ``opacity`` composite
        # once instead of once per shell crossing; the base Surface stays open,
        # as does any partial sweep that cuts the shell.
        primitive.declare_one_sided(not self.two_sided)
        primitive.declare_closed_shell(bool(getattr(self, "closed_shell", False)))
        primitive.declare_shadow_flags(*self._resolved_shadow_flags())
        return primitive

    def coord_function(self, uv: torch.Tensor):
        """Map the surface's ``(u, v)`` parameters to positions in space.

        This is what defines the surface's shape, and what each 3-D shape class
        overrides: :class:`~algan.mobs.shapes_3d.Sphere` maps the unit square onto a sphere,
        :class:`~algan.mobs.shapes_3d.Torus` onto a torus, and so on. The base implementation gives a flat
        plane spanning ``[-1, 1]`` on both axes.

        Parameters
        ----------
        uv
            Parameter coordinates to map, shape ``(*, 2)``, with both components in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Positions relative to the surface's location, shape ``(*, 3)``.
        """
        return torch.cat(((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1)

    def get_base_grid(self) -> torch.Tensor:
        """Get the surface's parameter grid, the ``(u, v)`` domain it is built from.

        Values run from 0 to 1 along both axes. This is the input the
        ``set_*_by_function`` methods evaluate their functions over, so it is what to
        write those functions in terms of.

        Returns
        -------
        torch.Tensor
            The ``(u, v)`` coordinates, shape ``[W, H, 2]``.
        """
        device = (
            self.grid.get_animated_attribute("location", copy=False).device
            if hasattr(self, "grid") and hasattr(self.grid, "location")
            else None
        )
        cache_key = (self.grid_width, self.grid_height, device)
        if getattr(self, "_cached_base_grid_key", None) != cache_key:
            grid = torch.stack(
                (
                    torch.linspace(0, 1, self.grid_width, device=device)
                    .view(-1, 1)
                    .expand(-1, self.grid_height),
                    torch.linspace(0, 1, self.grid_height, device=device)
                    .view(1, -1)
                    .expand(self.grid_width, -1),
                ),
                -1,
            )
            self._cached_base_grid = grid
            self._cached_base_grid_key = cache_key
        return self._cached_base_grid

    def _surface_points_at(self, u, v, grid, normals):
        """World positions of the parameter lattice ``u`` x ``v`` on the mesh.

        ``u`` and ``v`` are 1-D tensors in ``[0, 1]``, ``grid`` is
        ``[..., grid_width, grid_height, 3]`` and ``normals`` its vertex
        normals; the result is ``[..., len(u), len(v), 3]``.

        This reproduces what the renderer does with a UV coordinate, which is
        the only reason it is not simply ``coord_function(uv)``: the kernel
        interpolates the triangle corners' UVs barycentrically and the geometry
        under them is the logical PN patch, so a point's position is that patch
        evaluated at the barycentric coordinate -- not the analytic surface at
        the same parameter. The two differ tangentially by a fixed fraction of
        a grid cell (~12% on a stock Sphere), which is what would otherwise
        scallop a color boundary once the texture out-resolves the grid.
        """
        width, height = self.grid_width, self.grid_height
        fu = (u.clamp(0, 1) * (width - 1)).view(-1, 1)
        fv = (v.clamp(0, 1) * (height - 1)).view(1, -1)
        i = fu.floor().clamp(0, width - 2)
        j = fv.floor().clamp(0, height - 2)
        s = fu - i
        t = fv - j
        i = i.long().expand(-1, v.shape[0])
        j = j.long().expand(u.shape[0], -1)

        # get_grid_to_triangle_indices splits every cell into t1 = (00, 01, 10)
        # and t2 = (10, 01, 11); the diagonal between them is s + t == 1.
        lower = (s + t) <= 1.0
        pick = lower.unsqueeze(-1)

        def corners_of(field):
            return torch.stack(
                (
                    torch.where(pick, field[..., i, j, :], field[..., i + 1, j, :]),
                    field[..., i, j + 1, :],
                    torch.where(
                        pick, field[..., i + 1, j, :], field[..., i + 1, j + 1, :]
                    ),
                ),
                -2,
            )

        # Barycentric weights of (s, t) in whichever triangle contains it, in
        # the (corner 1, corner 2) form the PN evaluator takes.
        barycentric = torch.stack(
            (
                torch.where(lower, t, 1 - s),
                torch.where(lower, s, s + t - 1),
            ),
            -1,
        )
        return evaluate_logical_pn_per_patch(
            logical_pn_control_points(corners_of(grid), corners_of(normals)),
            barycentric,
        )

    def get_texture_locations(
        self, resolution: int | tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Get where in the world each texel of this surface's texture maps sits.

        Textures are addressed in the surface's own ``(u, v)`` coordinates, which
        makes a map easy to write in terms of the surface's parameters and awkward
        to write in terms of *space*. This is the bridge: it hands back the world
        position of every texel, laid out exactly like a texture map, so a map can
        be built from arithmetic on 3-D coordinates.

        .. code-block:: python

            xyz = surface.get_texture_locations()
            surface.color_texture = WHITE.mult_opacity(xyz[..., 1:2])

        The positions are read from the surface's current mesh, so they are right
        whether the shape came from its coordinate function, from a deformation,
        or from writing ``surface.grid.location`` yourself. They describe the
        surface *now*: a texture is carried in ``(u, v)``, so colors derived from
        world position travel with the surface when it later moves. Recompute them
        in an :meth:`~algan.animatable_base.animatable.Animatable.add_updater`
        callback for a texture that stays locked to world space.

        Animation
        ---------
        A query, not a change: nothing is recorded and the surface is untouched.
        The positions are those of the surface's current state, so call it after
        the transforms you want reflected in it.

        Parameters
        ----------
        resolution
            Texel counts ``(W, H)`` along ``u`` and ``v``, or one int for a square
            map. Defaults to ``None``, meaning the resolution of the surface's
            current :attr:`color_texture`, or its grid resolution when it has no
            texture yet. Pass the resolution explicitly to build a map for one of
            the material texture arguments instead.

        Returns
        -------
        torch.Tensor
            World-space positions, shape ``[W, H, 3]`` -- the layout
            :attr:`color_texture` takes, one row per texel of the map that will
            be sampled there. Called where the surface's state spans several
            frames (inside an updater), it keeps a leading frame axis,
            ``[F, W, H, 3]``.

        Raises
        ------
        ValueError
            If this is a packed surface built by
            :meth:`~algan.mobs.surfaces.surface.Surface.from_batches`, whose
            members share one texture and therefore have no single answer; if the
            grid has fewer than two points on an axis, so it has no triangles to
            sit on; or if ``resolution`` is not positive on both axes.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.get_base_grid`
            The ``(u, v)`` domain these positions correspond to.
        :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_function`
            Color the surface's vertices by ``(u, v)`` instead.

        Examples
        --------
        Paint a sphere's northern half, cutting the boundary in world space
        rather than along a parameter line:

        .. algan:: Example1SurfaceGetTextureLocations
            :save_last_frame:

            from algan import *

            globe = Sphere(radius=1.5)
            height = globe.get_texture_locations((128, 128))[..., 1:2]
            globe.color_texture = BLUE.mult_opacity((height > 0).float())
            globe.spawn()

            Scene.save_video()
        """
        if self._packed_grid_count() is not None:
            raise ValueError(
                "get_texture_locations is not defined for a packed surface: the "
                "members built by from_batches share one texture, so a texel "
                "has one position per member. Build the members separately to "
                "texture them by world position."
            )
        if self.grid_width < 2 or self.grid_height < 2:
            raise ValueError(
                "get_texture_locations needs at least 2 grid points on each "
                f"axis, got grid_width={self.grid_width}, "
                f"grid_height={self.grid_height}"
            )

        if resolution is None:
            if self._has_color_texture:
                # texture_height / texture_width are the u / v axis lengths, in
                # that order -- the names are inverted with respect to the
                # public [W, H, 5] contract, the axes are not.
                resolution = (self.texture_height, self.texture_width)
            else:
                resolution = (self.grid_width, self.grid_height)
        if isinstance(resolution, (tuple, list)):
            width, height = int(resolution[0]), int(resolution[1])
        else:
            width = height = int(resolution)
        if width < 1 or height < 1:
            raise ValueError(
                f"resolution must be positive on both axes, got {(width, height)}"
            )

        grid = self._reshape_grid_for_render(self.grid.location)
        # The renderer builds its PN patches from these same normals, and
        # ignore_normals leaves it without any: zero normals collapse the patch
        # onto the flat triangle, which is then exactly what is drawn.
        normals = (
            torch.zeros_like(grid)
            if self.ignore_normals
            else compute_grid_vertex_normals(grid)
        )

        # A closed axis is wrap-padded at render time, putting texel i at
        # i / W rather than i / (W - 1) (see wrap_pad_texture).
        closed_u, closed_v = surface_closed_axes(grid)
        u = torch.arange(width, device=grid.device, dtype=grid.dtype) / (
            width if closed_u and width > 1 else max(width - 1, 1)
        )
        v = torch.arange(height, device=grid.device, dtype=grid.dtype) / (
            height if closed_v and height > 1 else max(height - 1, 1)
        )

        rows = max(1, _TEXTURE_LOCATION_CHUNK_TEXELS // height)
        locations = torch.cat(
            [
                self._surface_points_at(u[start : start + rows], v, grid, normals)
                for start in range(0, width, rows)
            ],
            -3,
        )
        # The grid carries a leading frame axis, which is a single frame in
        # every case but a query made mid-materialization.
        return locations[0] if locations.shape[0] == 1 else locations

    def set_shape_to(self, other_surface: Surface) -> Surface:
        """Reshape this surface into the shape of another one.

        Takes ``other_surface``'s
        :meth:`~algan.mobs.surfaces.surface.Surface.coord_function` and applies
        it to this surface's own grid, which is how one parametric shape morphs
        into another. Any grid axis coarser than ``other_surface``'s is refined
        to match first, so the target shape is not under-sampled. The other
        surface is left untouched.

        Animation
        ---------
        Recorded as an animation: this surface's vertices travel to their new
        positions over the current context's duration (1 second by default).
        Wrap the call in ``Off()`` to reshape instantly.

        Parameters
        ----------
        other_surface
            The surface whose shape to take.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.set_location_by_function` : Shape it by a function you write.
        :meth:`~algan.animatable_base.mob.Mob.become` : Morph into a Mob of any kind, not just another Surface.
        """
        grid_width = max(self.grid_width, other_surface.grid_width)
        grid_height = max(self.grid_height, other_surface.grid_height)
        if (grid_width, grid_height) != (self.grid_width, self.grid_height):
            self._change_resolution(grid_width, grid_height)

        with Sync(animation_manager=self.animation_manager):
            self.set_location_by_function(other_surface.coord_function)
        # Normals need no transfer: they are recomputed from the grid every
        # render batch, so they follow the new shape on their own.
        return self

    def set_location_by_function(self, function):
        """Shape the surface by a function of its ``(u, v)`` parameters.

        The function maps each point of the parameter grid to a 3-D offset from the
        surface's location, which is how a sphere, a torus or an arbitrary parametric
        surface is defined. Animating between two such shapes is what makes a surface
        morph.

        Animation
        ---------
        Recorded as an animation: the grid points travel to their new positions over the
        current context's duration (1 second by default), so calling this on a spawned
        surface deforms it smoothly. Changing the grid *resolution* is a different
        matter -- that needs
        :meth:`~algan.animatable_base.mob.Mob.detach_history` first.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]`` and returning
            offsets of shape ``[..., 3]``. It must be vectorized -- it is called once
            on the whole grid, not per point.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.
        """

        def target_function(uv):
            # The location read is uncopied: it only feeds this out-of-place
            # add, and this runs per grid per shape rebuild.
            return function(uv.clone()) + self.get_animated_attribute(
                "location", copy=False
            )

        self.coord_function_active = function
        new_loc = target_function(squish(self.get_base_grid(), -3, -2).unsqueeze(0))
        self.grid.location = new_loc
        return self

    def get_default_color(self):
        """Get the color a Surface uses when none was given.

        Returns
        -------
        :class:`~algan.constants.color.Color`
            ``GREEN``.
        """
        return GREEN

    def set_color_by_function(self, function):
        """Color the surface by a function of its ``(u, v)`` parameters.

        Gives each point of the grid its own color, for gradients, heat maps or
        anything where color carries data. The colors travel with the surface as it
        deforms.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default), so the colors cross-fade smoothly.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]`` and returning
            colors of shape ``[..., 3]`` (RGB), ``[..., 4]`` (RGBA) or ``[..., 5]``
            (RGB, glow, alpha -- Algan's internal channel order). Channels are in
            ``[0, 1]``; a missing alpha defaults to 1 and a missing glow to 0.
            Must be vectorized over the whole grid.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`
            Paint an image on instead.
        """
        new_color = function(squish(self.get_base_grid(), -3, -2).unsqueeze(0))
        # Colors are stored five-channel (RGB + glow + alpha). Accept the
        # three- and four-channel forms a caller naturally writes.
        self.grid.color = Color.add_defaults(new_color)
        return self

    def set_color_by_image(self, rgba_array_or_file_path):
        """Paint an image across the surface.

        The image is mapped over the surface's ``(u, v)`` domain at its own
        resolution and sampled per fragment by the renderer, so it stays sharp
        however coarse the surface's grid is, and it follows the surface as it
        deforms. This sets
        :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`, which you can
        assign directly when you already hold the image as a tensor.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default): the surface cross-fades, texel by texel, to the image.

        Parameters
        ----------
        rgba_array_or_file_path
            Path to an image file, or an RGBA array. Paths resolve relative to the
            working directory and then the main script's directory, so an image beside
            your script is found either way.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        See Also
        --------
        :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`
            The attribute this writes, for images already loaded as tensors.
        """
        texture_image = get_image(rgba_array_or_file_path)
        # A color texture is indexed by the surface's own axes, (u, v); a
        # loaded image is [row, column] with rows running down the picture.
        # Same transposition ImageMob applies to the array it is built from.
        surface_texture = texture_image.transpose(-3, -2).flip(-2).contiguous()

        texture_shape = tuple(surface_texture.shape[-3:-1])
        if self.is_spawned() and (
            not self._has_color_texture
            or (self.texture_height, self.texture_width) != texture_shape
        ):
            # A texture attribute that does not exist yet -- or one whose
            # resolution is about to change, which detaches history -- has no
            # earlier value to interpolate from, so the assignment below would
            # pop. Seeding it with what the surface looks like now makes the
            # assignment the cross-fade this method documents.
            with Off(animation_manager=self.animation_manager):
                self.color_texture = self._current_appearance_as_texture(texture_shape)

        # The texture and the per-vertex bake are one visual change, so they
        # have to be recorded as simultaneous edits: written plainly, an
        # enclosing ``Seq`` would play them one after the other, each over half
        # its window. The bake keeps the vertex colors in step with the
        # texture -- shading reads the texture, but the glow accumulator
        # interpolates triangle corners, and this is the same bake
        # ``Surface(color_texture=...)`` does at construction.
        with Sync(animation_manager=self.animation_manager):
            self.color_texture = surface_texture
            self.grid.color = squish(
                F.interpolate(
                    texture_image.permute(2, 0, 1).unsqueeze(0),
                    (self.grid_height, self.grid_width),
                    mode="bilinear",
                    antialias=True,
                )
                .squeeze(0)
                .permute(2, 1, 0)
                .flip(-2),
                -3,
                -2,
            ).unsqueeze(0)
        return self

    def _current_appearance_as_texture(self, texture_shape):
        """This surface's current colors as a ``[W, H, 5]`` image of the given
        resolution: its color texture if it has one, its per-vertex grid colors
        otherwise. Used as the value a freshly created color texture
        interpolates *from*.
        """
        if self._has_color_texture:
            current = (
                self._color_texture_uncopied()
                .as_subclass(torch.Tensor)
                .view(-1, self.texture_height, self.texture_width, 5)[-1]
            )
        else:
            current = self.grid.get_animated_attribute("color").as_subclass(
                torch.Tensor
            )
            if current.shape[-2] == 1:
                current = current.expand(
                    *current.shape[:-2], self.grid_width * self.grid_height, -1
                )
            current = current.reshape(
                -1, self.grid_width, self.grid_height, current.shape[-1]
            )[-1]
        if tuple(current.shape[-3:-1]) == tuple(texture_shape):
            return current.contiguous()
        return (
            F.interpolate(
                current.permute(2, 0, 1).unsqueeze(0),
                size=tuple(texture_shape),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
        )
