from __future__ import annotations

import inspect
import threading
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Sync
from algan.animation_timeline.timeline import EditRecord
from algan.constants.color import *
from algan.constants.spatial import OUT
from algan.geometry.geometry import (
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.rendering.logical_pn import (
    evaluate_logical_pn,
    logical_pn_control_points,
)
from algan.settings.renderer_settings import RENDERER_REGISTRY
from algan.utils.file_utils import get_image
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    dot_product,
    squish,
    unsqueeze_left,
    unsquish,
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


def _call_parametric_function(function, u, v):
    """Evaluate a Manim-style ``func(u, v)`` on a tensor UV grid.

    Functions written with NumPy/scalar operations are evaluated point by
    point; torch-vectorized functions stay on the active device.
    """
    try:
        result = function(u, v)
        if isinstance(result, torch.Tensor):
            result = result.to(device=u.device, dtype=u.dtype)
            if result.shape[-1:] == (3,):
                return result
            if result.shape[:1] == (3,) and result.shape[1:] == u.shape:
                return result.movedim(0, -1)
        array = np.asarray(result)
        if array.shape[-1:] == (3,) and array.shape[:-1] == tuple(u.shape):
            return torch.as_tensor(array, device=u.device, dtype=u.dtype)
    except (TypeError, ValueError, RuntimeError):
        pass

    flat_u = u.detach().cpu().reshape(-1).numpy()
    flat_v = v.detach().cpu().reshape(-1).numpy()
    points = [
        np.asarray(function(float(uu), float(vv)), dtype=float)
        for uu, vv in zip(flat_u, flat_v)
    ]
    return torch.as_tensor(
        np.asarray(points).reshape(*u.shape, 3),
        device=u.device,
        dtype=u.dtype,
    )


def _looks_like_manim_surface_function(function):
    if function is None:
        return False
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional) >= 2 or any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )


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


def get_grid_to_triangle_indices(grid_width: int, grid_height: int, device):
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

    Returns
    -------
    torch.Tensor
        Vertex indices into the flattened grid, shape
        ``[(W-1) * (H-1) * 2, 3]``.
    """
    cache_key = (grid_width, grid_height, device)
    if cache_key not in _grid_triangle_indices_cache:
        W, H = grid_width, grid_height
        i_indices = torch.arange(W - 1, device=device).unsqueeze(1).expand(-1, H - 1)
        j_indices = torch.arange(H - 1, device=device).unsqueeze(0).expand(W - 1, -1)

        idx00 = i_indices * H + j_indices
        idx01 = i_indices * H + (j_indices + 1)
        idx10 = (i_indices + 1) * H + j_indices
        idx11 = (i_indices + 1) * H + (j_indices + 1)

        t1 = torch.stack((idx00, idx01, idx10), dim=-1)
        t2 = torch.stack((idx10, idx01, idx11), dim=-1)
        stacked = torch.stack((t1, t2), dim=-2)
        _grid_triangle_indices_cache[cache_key] = stacked.reshape(-1)
    return _grid_triangle_indices_cache[cache_key]


def grid_to_triangle_vertices(grid):
    """Internal: gather a per-grid-point quantity into per-triangle-vertex form.

    Turns values laid out on the surface grid into the flat triangle-vertex layout the
    renderer consumes. Works for positions, normals and colours alike.

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
    indices = get_grid_to_triangle_indices(W, H, grid.device)
    return flat_grid[..., indices, :]


def compute_grid_vertex_normals(grid):
    """Area-weighted vertex normals for a surface grid ``[..., W, H, 3]``,
    with closed-seam and pole merging. All computations broadcast over any
    leading dims (time, or a stack of same-shaped surfaces), which lets
    :func:`get_render_primitives_batched` run this once for many surfaces.
    """
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

    return -F.normalize(unnormalized_normals, p=2, dim=-1)


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
    grids = torch.stack([s._reshape_grid_for_render(s.grid.location) for s in surfaces])
    vertex_normals = grid_to_triangle_vertices(compute_grid_vertex_normals(grids))
    corners = grid_to_triangle_vertices(grids)
    return [
        s._build_render_primitive(
            grids[i], vertex_normals[i], precomputed_corners=corners[i]
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
    normal_function
        Function mapping 3-D world coordinates to their normal vectors (i.e. vectors pointing directly out
        of the surface), used for lighting.
    grid_height
        Height of the grid from which intrinsic coordinates are sampled.
    grid_width
        Width of the grid from which intrinsic coordinates are sampled.
    grid_aspect_ratio
        If not None, set the grid_height to be equal to grid_width * grid_aspect_ratio.
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
    render_tolerance
        Maximum sampled output-pixel deviation (as percentage of screen height) used
        when each PN triangle is dynamically diced into ordinary flat render triangles.
        Every PN triangle is diced only as finely as it itself needs, in every frame,
        so detail spent where the surface is close to the camera is not spent on the
        rest of the mesh or on the frames where it is far away.
    min_grid_resolution, max_grid_resolution
        Bounds for automatic grid sizing, measured in vertices per axis.
    resolution_shrink_margin
        Deprecated compatibility argument. Logical PN topology is fixed at
        construction and is never resized during animation.
    color_texture
        Optional color texture map ``[W, H, 5]`` (or ``[T, W, H, 5]`` for an
        animated map), sampled bilinearly in-kernel by the ray tracer.
    reflectivity_texture, roughness_texture, refractive_index_texture
        Optional per-texel material property maps, each ``[W, H, 1]`` (or
        ``[W, H]``, or ``[T, W, H, 1]``). Like ``color_texture`` they are
        sampled bilinearly per fragment inside the ray tracing kernel (only
        the general wavefront tracer implements this; batches containing such
        maps are routed to it automatically, for both flat and curved PN
        triangles). Properties without a map keep the per-vertex system. Maps
        of different resolutions are resampled to a common resolution.
    normal_texture
        Optional tangent-space normal map ``[W, H, 3]`` (or ``[T, W, H, 3]``),
        with components in ``[-1, 1]``: x along increasing ``u``, y along
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
        ``grid_height`` for more detail). Static only (no time dimension).
    *args, **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob`

    """

    _morph_family = "grid"

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
        normal_function=None,
        grid_height=None,
        grid_width=None,
        grid_aspect_ratio=None,
        checkered_color=None,
        color_texture=None,
        reflectivity_texture=None,
        roughness_texture=None,
        refractive_index_texture=None,
        normal_texture=None,
        glow_texture=None,
        ignore_normals=False,
        geometry_tolerance=0.001,
        render_tolerance=0.001,
        min_grid_resolution=4,
        max_grid_resolution=200,
        resolution_shrink_margin=0.1,
        *args,
        func=None,
        u_range=None,
        v_range=None,
        resolution=None,
        surface_piece_config=None,
        fill_color=None,
        fill_opacity=None,
        checkerboard_colors=None,
        stroke_color=None,
        stroke_width=None,
        should_make_jagged=False,
        pre_function_handle_to_anchor_scale_factor=1e-5,
        **kwargs,
    ):
        # User-defined subclasses conventionally store their geometry
        # parameters before calling ``super().__init__``. Preserve that small
        # pre-Surface state for the cache identity before Mob construction adds
        # scene/timeline bookkeeping that differs for every instance.
        resolution_cache_state = dict(self.__dict__)

        # ``Surface`` predates this compatibility layer in Algan and accepts a
        # vectorized ``coord_function(uv)``.  Manim instead accepts
        # ``func(u, v)``.  Support both forms without weakening the native API.
        manim_function = func
        if manim_function is None and _looks_like_manim_surface_function(
            coord_function
        ):
            manim_function = coord_function

        self._func = manim_function
        self.u_range = (0, 1) if u_range is None else tuple(u_range)
        self.v_range = (0, 1) if v_range is None else tuple(v_range)
        self.surface_piece_config = (
            {} if surface_piece_config is None else dict(surface_piece_config)
        )
        self.should_make_jagged = should_make_jagged
        self.pre_function_handle_to_anchor_scale_factor = (
            pre_function_handle_to_anchor_scale_factor
        )
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width

        if manim_function is not None:

            def mapped_coord_function(uv):
                u = self.u_range[0] + uv[..., 0] * (self.u_range[1] - self.u_range[0])
                v = self.v_range[0] + uv[..., 1] * (self.v_range[1] - self.v_range[0])
                return _call_parametric_function(manim_function, u, v)

            coord_function = mapped_coord_function
            if resolution is not None:
                u_resolution, v_resolution = _surface_resolution_pair(resolution)
                grid_width = u_resolution + 1
                grid_height = v_resolution + 1
            self.resolution = resolution

            if fill_color is None:
                fill_color = BLUE_D
            if fill_opacity is None:
                fill_opacity = 1.0
            kwargs.setdefault("color", fill_color)
            kwargs.setdefault("opacity", fill_opacity)

            if checkerboard_colors is None:
                checkerboard_colors = [BLUE_D, BLUE_E]
            self.checkerboard_colors = checkerboard_colors
            if checkerboard_colors is not False:
                checkerboard_colors = list(checkerboard_colors)
                if checkerboard_colors:
                    kwargs["color"] = checkerboard_colors[0]
                if len(checkerboard_colors) > 1:
                    checkered_color = checkerboard_colors[1]
        else:
            self.resolution = resolution
            if resolution is not None and grid_width is None and grid_height is None:
                u_resolution, v_resolution = _surface_resolution_pair(resolution)
                grid_width = u_resolution + 1
                grid_height = v_resolution + 1
            self.checkerboard_colors = checkerboard_colors
            if fill_color is not None:
                kwargs.setdefault("color", fill_color)
            if fill_opacity is not None:
                kwargs.setdefault("opacity", fill_opacity)

        if coord_function is None:
            coord_function = self.coord_function
        if normal_function is None:
            normal_function = self.normal_function

        self.coord_function_active = coord_function
        self.normal_function_active = normal_function
        self.ignore_normals = ignore_normals
        self._color_texture_attr = None

        self._geometry_auto_resolution_enabled = (
            grid_height is None and grid_width is None
        )
        # Compatibility flag retained for older introspection. Runtime
        # topology changes are deliberately disabled by the logical PN system.
        self._auto_resolution_enabled = False
        self._geometry_tolerance = float(geometry_tolerance)
        self._render_tolerance = float(render_tolerance)
        self._resolution_tolerance = self._geometry_tolerance
        self._min_grid_resolution = int(min_grid_resolution)
        self._max_grid_resolution = int(max_grid_resolution)
        self._resolution_shrink_margin = float(resolution_shrink_margin)
        self._grid_aspect_ratio = grid_aspect_ratio
        self._pending_auto_resolution = None
        self._resolution_update_in_progress = True
        if not np.isfinite(self._geometry_tolerance):
            raise ValueError("geometry_tolerance must be finite")
        if self._geometry_tolerance <= 0:
            raise ValueError("geometry_tolerance must be greater than zero")
        if not np.isfinite(self._render_tolerance):
            raise ValueError("render_tolerance must be finite")
        if self._render_tolerance <= 0:
            raise ValueError("render_tolerance must be greater than zero")
        if self._min_grid_resolution < 2:
            raise ValueError("min_grid_resolution must be at least 2")
        if self._max_grid_resolution < self._min_grid_resolution:
            raise ValueError(
                "max_grid_resolution must be greater than or equal to "
                "min_grid_resolution"
            )
        if not 0 <= self._resolution_shrink_margin < 1:
            raise ValueError("resolution_shrink_margin must be in [0, 1)")
        # triangle_normals = grid_to_triangle_vertices(F.normalize(normal_function(base_grid), p=2, dim=-1)) if not ignore_normals else None
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
            if grid_aspect_ratio is not None:
                grid_height = int(grid_width * grid_aspect_ratio)

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
        material_prop_textures = {
            k: v for k, v in material_prop_textures.items() if v is not None
        }
        if material_prop_textures:
            self.material_texture, self.material_texture_flags = (
                self._build_material_texture(material_prop_textures)
            )
        if normal_texture is not None:
            self.normal_texture = self._normalize_texture_shape(normal_texture, 3).to(
                self.location.device
            )
        if glow_texture is not None:
            kwargs["glow"] = self._bake_texture_to_grid(glow_texture)

        base_grid = self.get_base_grid()
        grid_points = squish(coord_function(base_grid), -3, -2) + self.location

        color = kwargs["color"] if "color" in kwargs else self.get_default_color()
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
    def geometry_tolerance(self):
        """Construction-time absolute world-space PN fitting tolerance."""
        return self._geometry_tolerance

    @property
    def render_tolerance(self):
        """Per-frame output-pixel flat-triangle tessellation tolerance."""
        return self._render_tolerance

    def func(self, u, v):
        """Evaluate the original Manim-style parametric function."""
        if self._func is None:
            raise AttributeError(
                "this Surface was constructed with coord_function, not func"
            )
        return self._func(u, v)

    @property
    def color_texture(self):
        """An image painted across the surface, as an ``[W, H, 5]`` RGBA+glow tensor.

        Assigning an image maps it over the surface's parameter domain, so it follows
        the surface as it deforms. ``None`` means the surface is drawn from its
        per-vertex colours instead. Assigning a texture whose resolution differs from
        the current one detaches history, since the two cannot be interpolated.

        A time dimension is allowed (``[T, W, H, 5]``) for a texture that changes
        during the animation.
        """
        attr = getattr(self, "_color_texture_attr", None)
        if attr is None:
            return None
        return self.get_animated_attribute(attr)

    @color_texture.setter
    def color_texture(self, texture):
        previous_attr = getattr(self, "_color_texture_attr", None)
        if texture is None:
            if previous_attr is not None and self.is_spawned():
                self.detach_history()
            self._color_texture_attr = None
            return self

        texture = torch.as_tensor(texture)
        if texture.dim() not in (3, 4) or texture.shape[-1] != 5:
            raise ValueError("color_texture must have shape [W, H, 5] or [T, W, H, 5]")
        texture_height, texture_width = texture.shape[-3:-1]
        attr = f"color_texture_{texture_height * texture_width}"

        if previous_attr is not None and previous_attr != attr and self.is_spawned():
            # Keep the old texture topology on a frozen historical clone. The
            # live surface then receives a fresh timeline with the new width.
            self.detach_history()

        self._color_texture_attr = attr
        self.texture_height = int(texture_height)
        self.texture_width = int(texture_width)
        self.register_attrs_as_animatable([attr])
        setattr(self, attr, squish(texture, -3, -1))
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
        """Return one smooth unit normal for each sampled surface vertex."""
        grid = unsquish(self.grid.location, -2, self.grid_height)
        return compute_grid_vertex_normals(grid).reshape(*grid.shape[:-3], -1, 3)

    def set_fill_by_checkerboard(self, *colors, opacity=None):
        """Apply an alternating vertex-color pattern and return this surface."""
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
        self.checkerboard_colors = list(colors)
        return self

    def set_fill_by_value(self, axes, colorscale=None, axis=2, **kwargs):
        """Color sampled vertices by their coordinate value along an axis.

        This is the renderer-independent counterpart of Manim's per-face
        coloring. Algan interpolates these vertex colors over its triangle mesh.
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
        count by more than ``resolution_shrink_margin``; otherwise that
        dimension stays at its current size.
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
        shrink_boundary = current_work * (1.0 - self._resolution_shrink_margin)
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
        # -- which is exactly the shape a colour wave most often has to refine.
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
        upwards = camera.get_upwards_direction().reshape(-1, 3)[0]
        relative = points - camera_location
        depth = (relative * forward).sum(dim=-1)

        screen_vector = camera.screen.location.reshape(-1, 3)[0] - camera_location
        screen_distance = (screen_vector * forward).sum().abs().clamp_min(1e-8)
        pixel_scale = self.scene.video_settings.resolution[1] / (
            2.0 * float(camera.screen_scale_factor)
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

        if self._grid_aspect_ratio is not None:
            ratio = float(self._grid_aspect_ratio)
            low, high = minimum, maximum
            best_width = maximum
            while low <= high:
                width = (low + high) // 2
                height = min(
                    maximum,
                    max(minimum, int(round(width * ratio))),
                )
                if acceptable(width, height):
                    best_width = width
                    high = width - 1
                else:
                    low = width + 1
            result = (
                best_width,
                min(
                    maximum,
                    max(minimum, int(round(best_width * ratio))),
                ),
            )
        else:
            width = first_acceptable(maximum, vary_width=True)
            height = first_acceptable(maximum, vary_width=False)
            while not acceptable(width, height) and (
                width < maximum or height < maximum
            ):
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
            _freeze_resolution_cache_value(self._grid_aspect_ratio),
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

        if self._grid_aspect_ratio is not None:
            ratio = float(self._grid_aspect_ratio)
            low, high = minimum, maximum
            best_width = maximum
            while low <= high:
                width = (low + high) // 2
                height = min(maximum, max(minimum, int(round(width * ratio))))
                if acceptable(width, height):
                    best_width = width
                    high = width - 1
                else:
                    low = width + 1
            return best_width, min(
                maximum, max(minimum, int(round(best_width * ratio)))
            )

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
        """Refine the grid so a travelling colour wave crosses it smoothly.

        A surface's grid resolution is chosen to fit its *shape*, so a surface
        that is flat along one axis is sampled only at the ends of it -- enough
        to draw the geometry, far too coarse to draw a colour band moving across
        it. Both colour and opacity are carried per grid vertex, so either kind
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
        if self.color_texture is not None:
            # Textured surfaces shade from the texture map rather than from
            # vertex colours, so a finer grid would buy geometry and no pixels.
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

    def _uses_pn_triangles(self):
        """True if newly created surfaces render as curved point-normal (PN)
        triangles, i.e. ``enable_ray_tracing(pn_triangles=True)`` is active.

        ``enable_ray_tracing`` rebinds this module's ``TrianglePrimitive`` name
        to the PN primitive class in that case (and to a flat triangle class
        otherwise), so checking the currently bound class tells us how the mesh
        will actually be rendered.
        """
        try:
            from algan.rendering.raytracing.primitives import (
                RayTracedPNTrianglePrimitive,
            )
        except Exception:
            return False
        return isinstance(RENDERER_REGISTRY.triangle_primitive, type) and issubclass(
            RENDERER_REGISTRY.triangle_primitive, RayTracedPNTrianglePrimitive
        )

    def _compute_error(self, coord_function, W, H):
        """Max screen-space deviation in pixels for a ``W x H`` grid.

        With PN (curved) triangles active each triangle is bent to a quadratic
        patch, so we measure against that patch. Otherwise the surface is drawn
        as flat triangles, so the resolution must instead make the *flat* mesh
        approximate the surface to within tolerance.
        """
        if self._uses_pn_triangles():
            return self._compute_pn_error(coord_function, W, H)
        return self._compute_flat_error(coord_function, W, H)

    def _compute_flat_error(self, coord_function, W, H):
        """Max deviation between the flat-triangle mesh and the true surface,
        sampled at a fixed set of barycentric coordinates per triangle.

        Mirrors :meth:`_compute_pn_error` but approximates each triangle by the
        flat plane through its three corners (linear interpolation of the corner
        positions) instead of a curved PN patch.
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

    def _compute_pn_error(self, coord_function, W, H):
        device = self.location.device
        grid_u = torch.linspace(0, 1, W, device=device)
        grid_v = torch.linspace(0, 1, H, device=device)
        grid_uu, grid_vv = torch.meshgrid(grid_u, grid_v, indexing="ij")
        base_grid = torch.stack([grid_uu, grid_vv], dim=-1)

        grid_points = coord_function(base_grid.clone())

        # Same normals the renderer will build for this grid -- including the
        # seam and pole merges -- so the tessellation error this estimates is
        # the error of the patches actually rendered.
        vertex_normals = compute_grid_vertex_normals(grid_points)

        triangle_uvs = grid_to_triangle_vertices(base_grid)
        triangle_corners = grid_to_triangle_vertices(grid_points)
        triangle_normals = grid_to_triangle_vertices(vertex_normals)

        corners_3d = triangle_corners.reshape(-1, 3, 3)
        normals_3d = triangle_normals.reshape(-1, 3, 3)
        uvs_2d = triangle_uvs.reshape(-1, 3, 2)

        from algan.rendering.raytracing.pn_patch import (
            evaluate_pn_patch,
            pn_control_points,
            pn_patch_coefficients,
        )

        control_points = pn_control_points(corners_3d, normals_3d)
        coefficients = pn_patch_coefficients(control_points)

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

        coefs = coefficients.unsqueeze(1)
        u = bary_coords[:, 0].unsqueeze(0)
        v = bary_coords[:, 1].unsqueeze(0)

        S_points = evaluate_pn_patch(coefs, u, v)

        uv0 = uvs_2d[:, 0, :].unsqueeze(1)
        uv1 = uvs_2d[:, 1, :].unsqueeze(1)
        uv2 = uvs_2d[:, 2, :].unsqueeze(1)

        u_expanded = u.unsqueeze(-1)
        v_expanded = v.unsqueeze(-1)
        w_expanded = 1.0 - u_expanded - v_expanded

        uv_true = w_expanded * uv0 + u_expanded * uv1 + v_expanded * uv2
        P_true = coord_function(uv_true.clone())

        return self._screen_space_error(S_points, P_true)

    @staticmethod
    def _normalize_texture_shape(tex, channels):
        """Normalize a user-supplied texture to ``[T, W, H, channels]``.
        Accepts ``[W, H]`` (single-channel maps only), ``[W, H, channels]``
        or ``[T, W, H, channels]``; ``W`` is the ``u`` axis, ``H`` the ``v``
        axis of the surface's intrinsic coordinates.
        """
        tex = torch.as_tensor(tex).float()
        if tex.dim() == 2:
            if channels != 1:
                raise ValueError(
                    f"a 2-D texture is only valid for single-channel "
                    f"properties, expected {channels} channels"
                )
            tex = tex.unsqueeze(-1)
        if tex.dim() == 3:
            tex = tex.unsqueeze(0)
        if tex.dim() != 4 or tex.shape[-1] != channels:
            raise ValueError(
                f"texture must have shape [W, H, {channels}] or "
                f"[T, W, H, {channels}], got {tuple(tex.shape)}"
            )
        return tex

    def _build_material_texture(self, textures_dict):
        """Combine per-property maps into one ``[T, W, H, 5]`` material
        texture (channels: reflectivity, roughness, refractive index, and two
        reserved) at the finest common resolution, plus the bitmask of which
        channels are texture-driven (bit i = channel i has a map; unset
        channels keep the per-vertex value in-kernel).
        """
        channel_slots = {"reflectivity": 0, "roughness": 1, "refractive_index": 2}
        device = self.location.device
        texs = {
            k: self._normalize_texture_shape(v, 1).to(device)
            for k, v in textures_dict.items()
        }
        T = max(t.shape[0] for t in texs.values())
        W = max(t.shape[1] for t in texs.values())
        H = max(t.shape[2] for t in texs.values())
        combined = torch.zeros((T, W, H, 5), device=device)
        flags = 0
        for name, t in texs.items():
            if t.shape[1:3] != (W, H):
                t = F.interpolate(
                    t.permute(0, 3, 1, 2),
                    size=(W, H),
                    mode="bilinear",
                    align_corners=True,
                ).permute(0, 2, 3, 1)
            slot = channel_slots[name]
            combined[..., slot] = t.expand(T, W, H, 1)[..., 0]
            flags |= 1 << slot
        return combined, flags

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
        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == (n_grid, rendered_grid_count, names):
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
        result = int(animation_and_intermediates + primitive_bytes + shader_bytes)
        self._memory_per_timestep_cache = (
            (n_grid, rendered_grid_count, names),
            result,
        )
        return result

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
        """Restore flat grid rows to one or more independent surface grids."""
        packed_grid_count = self._packed_grid_count()
        if packed_grid_count is None:
            return unsquish(values, -2, self.grid_height)
        return values.reshape(
            *values.shape[:-2],
            packed_grid_count,
            self.grid_width,
            self.grid_height,
            values.shape[-1],
        )

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
        grid = self._reshape_grid_for_render(self.grid.location)
        if not self.ignore_normals:
            vertex_normals = grid_to_triangle_vertices(
                compute_grid_vertex_normals(grid)
            )
        else:
            vertex_normals = None
        return self._build_render_primitive(grid, vertex_normals)

    def _build_render_primitive(self, grid, vertex_normals, precomputed_corners=None):
        """Assemble the
        :class:`~algan.rendering.primitives.triangle_primitive.TrianglePrimitive`
        for this surface from an already-materialized grid ``[T, W, H, 3]``
        and (triangle-gathered)
        vertex normals. ``precomputed_corners`` lets the batched path pass in
        corners gathered on the whole surface stack at once.
        """

        def expand_grid_to_verts(x):
            if x.shape[-2] == 1:
                x = x.expand(*x.shape[:-2], self.grid.location.shape[-2], -1)
            x = self._reshape_grid_for_render(x)
            return self._flatten_packed_triangle_vertices(grid_to_triangle_vertices(x))

        def compute_grid_color():
            # Plain tensor, not the Color subclass the public property returns:
            # geometry building only does arithmetic on these numbers, and a
            # subclass sends every operation through __torch_function__ (see
            # the matching note in BezierCircuitCubic.get_render_primitives).
            grid_color = self.grid.get_animated_attribute("color").clone()
            grid_color[..., -1:] *= self.grid.opacity
            grid_color[..., -2:-1] += self.grid.glow
            return grid_color

        uvs = None
        texture_map = None
        material_texture_map = getattr(self, "material_texture", None)
        material_texture_flags = getattr(self, "material_texture_flags", 0)
        normal_texture_map = getattr(self, "normal_texture", None)
        if (
            self.color_texture is not None
            or material_texture_map is not None
            or normal_texture_map is not None
        ):
            # Generate UV coordinates for the triangle corners from the base grid
            base_grid = self.get_base_grid()
            uvs = grid_to_triangle_vertices(base_grid)
            packed_grid_count = self._packed_grid_count()
            if packed_grid_count is not None:
                uvs = uvs.unsqueeze(0).expand(packed_grid_count, -1, -1).flatten(0, 1)
            uvs = uvs.unsqueeze(0)  # [1, num_triangles * 3, 2]
        if self.color_texture is not None:
            texture_map = (
                (self.color_texture)
                .view(
                    self.color_texture.shape[0],
                    self.texture_height,
                    self.texture_width,
                    5,
                )
                .as_subclass(Color)
                .mult_opacity(self.opacity.unsqueeze(-2))
            )

        colors = expand_grid_to_verts(compute_grid_color())
        corners = self._flatten_packed_triangle_vertices(
            grid_to_triangle_vertices(grid)
            if precomputed_corners is None
            else precomputed_corners
        )
        normals = (
            None
            if vertex_normals is None
            else self._flatten_packed_triangle_vertices(vertex_normals)
        )

        from algan.rendering.raytracing.primitives import (
            LogicalPNTrianglePrimitive,
        )

        return LogicalPNTrianglePrimitive(
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
            render_tolerance=self._render_tolerance,
            **{
                k: expand_grid_to_verts(v)
                for k, v in self.grid.get_shader_params().items()
            },
        )

    def coord_function(self, uv: torch.Tensor):
        """Map the surface's ``(u, v)`` parameters to positions in space.

        This is what defines the surface's shape, and what each 3-D shape class
        overrides: :class:`~.Sphere` maps the unit square onto a sphere,
        :class:`~.Torus` onto a torus, and so on. The base implementation gives a flat
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

    def normal_function(self, uv):
        """Map the surface's ``(u, v)`` parameters to shading normals.

        Overridden by the 3-D shape classes alongside
        :meth:`~algan.mobs.surfaces.surface.Surface.coord_function`, so each shape
        lights correctly. The base
        implementation returns ``OUT``, the normal of a flat plane facing the viewer.

        Parameters
        ----------
        uv
            Parameter coordinates to map, shape ``(*, 2)``, with both components in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Unit normals, shape ``(*, 3)``, or a single vector broadcast over the grid.
        """
        return OUT

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
            self.grid.location.device
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

    def set_shape_to(self, other_surface: Surface):
        """Changes this surface's shape to the shape defined by another surface's
        :meth:`~algan.mobs.surfaces.surface.Surface.coord_function`. Any lower-resolution
        grid axis is
        refined to match ``other_surface`` before the shape change.

        Parameters
        ----------
        other_surface
            The surface from which to get coord_function.

        """
        grid_width = max(self.grid_width, other_surface.grid_width)
        grid_height = max(self.grid_height, other_surface.grid_height)
        if (grid_width, grid_height) != (self.grid_width, self.grid_height):
            self._change_resolution(grid_width, grid_height)

        with Sync(animation_manager=self.animation_manager):
            self.set_location_by_function(other_surface.coord_function)
            # TODO setting normals currently doesn't work, implement it.
            # self.set_normal_by_function(other_surface.normal_function)

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
            return function(uv.clone()) + self.location

        self.coord_function_active = function
        new_loc = target_function(squish(self.get_base_grid(), -3, -2).unsqueeze(0))
        self.grid.location = new_loc
        return self

    def set_normal_by_function(self, function):
        """Set the surface's shading normals from a function of ``(u, v)``.

        Overrides the normals computed from the geometry, which lets a flat surface
        light as though it were curved, or a faceted one look smooth. The geometry
        itself is unchanged.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default), so the shading shifts smoothly rather than snapping.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]`` and returning
            normals of shape ``[..., 3]``. Must be vectorized over the whole grid.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.
        """
        new_normals = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(
            unsquish(self.triangles.corners.location, -2, 3),
            scene=self.scene,
            normals=new_normals,
            add_to_scene=False,
        )
        with Sync(animation_manager=self.animation_manager):
            self.triangles.basis = new_triangles.basis
            self.triangles.corners.basis = new_triangles.corners.basis
        return self

    def get_default_color(self):
        """Get the colour a Surface uses when none was given.

        Returns
        -------
        :class:`~algan.constants.color.Color`
            ``GREEN``.
        """
        return GREEN

    def set_color_by_function(self, function):
        """Colour the surface by a function of its ``(u, v)`` parameters.

        Gives each point of the grid its own colour, for gradients, heat maps or
        anything where colour carries data. The colours travel with the surface as it
        deforms.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default), so the colours cross-fade smoothly.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]`` and returning
            colours of shape ``[..., 4]`` or ``[..., 5]`` (RGBA, optionally with glow).
            Must be vectorized over the whole grid.

        Returns
        -------
        :class:`~algan.mobs.surfaces.surface.Surface`
            This surface, so calls can be chained.

        See Also
        --------
        :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_texture`
            Paint an image on instead.
        """
        new_color = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(
            unsquish(self.triangles.corners.location, -2, 3),
            scene=self.scene,
            color=new_color,
            normals=None,
            add_to_scene=False,
        )
        with Sync(animation_manager=self.animation_manager):
            self.triangles.color = new_triangles.color
            self.triangles.corners.color = new_triangles.corners.color
        return self

    def set_color_by_texture(self, rgba_array_or_file_path):
        """Paint an image across the surface.

        The image is resampled to the surface's grid resolution and baked into its
        per-vertex colours, so detail finer than the grid is lost -- raise the
        surface's resolution, or use
        :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`, if you need the
        image sharp.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default): the colours cross-fade to the image.

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
        """
        texture_image = get_image(rgba_array_or_file_path)
        texture_image = (
            F.interpolate(
                texture_image.permute(2, 0, 1).unsqueeze(0),
                (self.grid_height, self.grid_width),
                mode="bilinear",
                antialias=True,
            )
            .squeeze(0)
            .permute(2, 1, 0)
            .flip(-2)
        )
        self.triangles.corners.color = squish(
            grid_to_triangle_vertices(texture_image), -3, -2
        )
        return self
