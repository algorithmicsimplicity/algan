import torch
import torch.nn.functional as F

from algan.mobs.renderable import Renderable
from algan.settings.renderer_settings import RENDERER_SETTINGS
from algan.utils.tensor_utils import broadcast_cross_product
from algan.animation.animation_contexts import Sync
from algan.constants.color import *
from algan.constants.spatial import OUT
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.file_utils import get_image
from algan.utils.tensor_utils import unsqueeze_left, squish, unsquish


_grid_triangle_indices_cache = {}

def get_grid_to_triangle_indices(grid_width, grid_height, device):
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
    :func:`get_render_primitives_batched` run this once for many surfaces."""
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
    is_closed_x = torch.all((grid[..., 0, :, :] - grid[..., -1, :, :]).abs() < 1e-4, dim=(-1, -2))
    mask_x = is_closed_x.view(*is_closed_x.shape, 1, 1)
    closed_normals_x = unnormalized_normals[..., 0, :, :] + unnormalized_normals[..., -1, :, :]
    unnormalized_normals[..., 0, :, :] = torch.where(mask_x, closed_normals_x, unnormalized_normals[..., 0, :, :])
    unnormalized_normals[..., -1, :, :] = torch.where(mask_x, closed_normals_x, unnormalized_normals[..., -1, :, :])

    is_closed_y = torch.all((grid[..., :, 0, :] - grid[..., :, -1, :]).abs() < 1e-4, dim=(-1, -2))
    mask_y = is_closed_y.view(*is_closed_y.shape, 1, 1)
    closed_normals_y = unnormalized_normals[..., :, 0, :] + unnormalized_normals[..., :, -1, :]
    unnormalized_normals[..., :, 0, :] = torch.where(mask_y, closed_normals_y, unnormalized_normals[..., :, 0, :])
    unnormalized_normals[..., :, -1, :] = torch.where(mask_y, closed_normals_y, unnormalized_normals[..., :, -1, :])

    # Merge unnormalized normals at singular poles (e.g. Sphere poles, Cone tip).
    # The fan of triangles around a collapsed pole column can sum to a
    # normal pointing *inward* (the degenerate pole faces carry the
    # opposite winding sign from the rest of the grid). An inverted pole
    # normal makes the patches touching the pole interpolate from an
    # inward normal at the pole to the (correct) outward normals on the
    # neighbouring ring, sweeping the shading normal through the lit
    # hemisphere on the way -- a bright ring around an otherwise unlit
    # pole. Orient each pole normal into the same hemisphere as its
    # adjacent ring (column 1 / -2), which is reliably outward.
    def _orient_to_ring(pole_normal, ring_normal):
        dot = (pole_normal * ring_normal).sum(-1, keepdim=True)
        return torch.where(dot < 0, -pole_normal, pole_normal)

    is_south_pole = torch.all((grid[..., :, 0, :] - grid[..., :1, 0, :]).abs() < 1e-4, dim=(-1, -2))
    mask_sp = is_south_pole.view(*is_south_pole.shape, 1, 1)
    pole_normal_sp = unnormalized_normals[..., :, 0, :].sum(-2, keepdim=True)
    ring_normal_sp = unnormalized_normals[..., :, 1, :].sum(-2, keepdim=True)
    pole_normal_sp = _orient_to_ring(pole_normal_sp, ring_normal_sp)
    unnormalized_normals[..., :, 0, :] = torch.where(mask_sp, pole_normal_sp, unnormalized_normals[..., :, 0, :])

    is_north_pole = torch.all((grid[..., :, -1, :] - grid[..., :1, -1, :]).abs() < 1e-4, dim=(-1, -2))
    mask_np = is_north_pole.view(*is_north_pole.shape, 1, 1)
    pole_normal_np = unnormalized_normals[..., :, -1, :].sum(-2, keepdim=True)
    ring_normal_np = unnormalized_normals[..., :, -2, :].sum(-2, keepdim=True)
    pole_normal_np = _orient_to_ring(pole_normal_np, ring_normal_np)
    unnormalized_normals[..., :, -1, :] = torch.where(mask_np, pole_normal_np, unnormalized_normals[..., :, -1, :])

    return -F.normalize(unnormalized_normals, p=2, dim=-1)


def get_render_primitives_batched(surfaces):
    """Build render primitives for N surfaces that share a grid shape and
    frame count, running the geometry pipeline (normal computation and
    triangle-vertex gathers) once on a ``[N, T, W, H, 3]`` stack instead of
    once per surface. Numerically identical to calling
    :meth:`Surface.get_render_primitives` on each surface (all ops are
    elementwise or reduce over non-batch dims), but with N times fewer
    Python/torch dispatches. Callers must ensure every surface uses the stock
    ``Surface.get_render_primitives``, has no ``color_texture``, has
    ``ignore_normals`` False, and has identical grid dimensions and
    ``grid.location`` shape."""
    grids = torch.stack(
        [unsquish(s.grid.location, -2, s.grid_height) for s in surfaces]
    )
    vertex_normals = grid_to_triangle_vertices(compute_grid_vertex_normals(grids))
    corners = grid_to_triangle_vertices(grids)
    return [
        s._build_render_primitive(grids[i], vertex_normals[i], precomputed_corners=corners[i])
        for i, s in enumerate(surfaces)
    ]


class Surface(Renderable):
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
    glow_texture, glow_radius_texture
        Optional glow strength/radius maps, each ``[W, H, 1]`` (or ``[W, H]``).
        These are consumed per-vertex by the glow accumulator, so they are
        baked to the surface grid resolution (raise ``grid_width``/
        ``grid_height`` for more detail). Static only (no time dimension).
    *args, **kwargs
        Passed to :class:`~.Mob`

    """

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
        glow_radius_texture=None,
        ignore_normals=False,
        tolerance=0.005,
            min_grid_resolution=4,
        *args,
        **kwargs,
    ):
        if coord_function is None:
            coord_function = self.coord_function
        if normal_function is None:
            normal_function = self.normal_function

        self.coord_function_active = coord_function
        self.normal_function_active = normal_function
        self.ignore_normals = ignore_normals
        # triangle_normals = grid_to_triangle_vertices(F.normalize(normal_function(base_grid), p=2, dim=-1)) if not ignore_normals else None
        super().__init__(*args, **kwargs)
        # A surface with an explicit texture map keeps it as an animatable
        # attribute; without one (color_texture=None) the per-vertex colours are
        # built from `color` in the else-branch below, so leave it None (the
        # downstream texture path is guarded by `self.color_texture is not None`).
        if color_texture is not None:
            self.register_attrs_as_animatable(['color_texture'])
            self.color_texture = squish(color_texture, -3, -1)
            self.texture_height, self.texture_width = color_texture.shape[-3:-1]
        else:
            self.color_texture = None

        # Auto-tuning grid resolution
        if grid_height is None and grid_width is None:
            device = self.location.device
            # We sample the true surface on a fine grid to determine the bounding box diagonal scale.
            sample_u = torch.linspace(0, 1, 100, device=device)
            sample_v = torch.linspace(0, 1, 100, device=device)
            grid_u, grid_v = torch.meshgrid(sample_u, sample_v, indexing='ij')
            sample_uv = torch.stack([grid_u, grid_v], dim=-1)
            
            # Since coord_function may modify uv in-place (e.g. Cylinder/Cone), pass a clone
            sample_points = coord_function(sample_uv.clone())
            
            min_coords = sample_points.min(dim=0).values.min(dim=0).values
            max_coords = sample_points.max(dim=0).values.max(dim=0).values
            scale = (max_coords - min_coords).norm()
            if scale < 1e-8:
                scale = torch.tensor(1.0, device=device)

            if grid_aspect_ratio is not None:
                # Fixed aspect ratio: search for a single resolution parameter
                low = min_grid_resolution
                high = 200
                best_N = high
                while low <= high:
                    mid = (low + high) // 2
                    W = mid
                    H = max(min_grid_resolution, int(mid * grid_aspect_ratio))
                    try:
                        error = self._compute_error(coord_function, W, H)
                        if error < tolerance * scale:
                            best_N = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                grid_width = best_N
                grid_height = max(min_grid_resolution, int(best_N * grid_aspect_ratio))
            else:
                # Independent rectangular search
                # 1. Search for best grid_width (W) with grid_height (H) set to a high resolution (200)
                low = min_grid_resolution
                high = 200
                best_W = high
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        error = self._compute_error(coord_function, mid, 200)
                        if error < tolerance * scale:
                            best_W = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                
                # 2. Search for best grid_height (H) with grid_width (W) set to a high resolution (200)
                low = min_grid_resolution
                high = 200
                best_H = high
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        error = self._compute_error(coord_function, 200, mid)
                        if error < tolerance * scale:
                            best_H = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                
                # Joint error correction loop
                try:
                    joint_error = self._compute_error(coord_function, best_W, best_H)
                    while joint_error > tolerance * scale and (best_W < 200 or best_H < 200):
                        ratio = joint_error / (tolerance * scale)
                        factor = max(1.15, float(torch.sqrt(ratio).item()))
                        if best_W < 200:
                            best_W = min(200, int(best_W * factor) + 1)
                        if best_H < 200:
                            best_H = min(200, int(best_H * factor) + 1)
                        joint_error = self._compute_error(coord_function, best_W, best_H)
                except Exception:
                    pass
                
                grid_width = best_W
                grid_height = best_H
        else:
            # Fall back to specified manual resolution
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
            'reflectivity': reflectivity_texture,
            'roughness': roughness_texture,
            'refractive_index': refractive_index_texture,
        }
        material_prop_textures = {
            k: v for k, v in material_prop_textures.items() if v is not None
        }
        if material_prop_textures:
            self.material_texture, self.material_texture_flags = (
                self._build_material_texture(material_prop_textures))
        if normal_texture is not None:
            self.normal_texture = self._normalize_texture_shape(
                normal_texture, 3).to(self.location.device)
        if glow_texture is not None:
            kwargs['glow'] = self._bake_texture_to_grid(glow_texture)
        if glow_radius_texture is not None:
            kwargs['glow_radius'] = self._bake_texture_to_grid(glow_radius_texture)

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
                tex_temp = F.interpolate(tex_temp, size=(grid_width, grid_height), mode='bilinear', align_corners=True)
                vertex_color_texture = tex_temp.permute(0, 2, 3, 1).squeeze(0)
            elif tex.dim() == 4:  # [T, W, H, 5]
                tex_temp = tex.permute(0, 3, 1, 2)
                tex_temp = F.interpolate(tex_temp, size=(grid_width, grid_height), mode='bilinear', align_corners=True)
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
        self.grid = Renderable(**kwargs)
        self.add_children(self.grid)
        self.components = [self.grid]
        self.grid.is_primitive = True
        self.is_primitive = True
        self.ignore_wave_animations = True

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
        return isinstance(RENDERER_SETTINGS.triangle_primitive, type) and issubclass(
            RENDERER_SETTINGS.triangle_primitive, RayTracedPNTrianglePrimitive
        )

    def _compute_error(self, coord_function, W, H):
        """Max deviation between the rendered mesh and the true surface for a
        ``W x H`` grid, used to drive auto-resolution.

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
        grid_uu, grid_vv = torch.meshgrid(grid_u, grid_v, indexing='ij')
        base_grid = torch.stack([grid_uu, grid_vv], dim=-1)

        grid_points = coord_function(base_grid.clone())

        triangle_uvs = grid_to_triangle_vertices(base_grid)
        triangle_corners = grid_to_triangle_vertices(grid_points)

        corners_3d = triangle_corners.reshape(-1, 3, 3)
        uvs_2d = triangle_uvs.reshape(-1, 3, 2)

        bary_coords = torch.tensor([
            [1/3, 1/3],
            [1/2, 0.0],
            [0.0, 1/2],
            [1/2, 1/2],
            [1/6, 1/6],
            [1/6, 2/3],
            [2/3, 1/6]
        ], device=device)

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

        errors = (S_points - P_true).norm(p=2, dim=-1)
        return errors.max()

    def _compute_pn_error(self, coord_function, W, H):
        device = self.location.device
        grid_u = torch.linspace(0, 1, W, device=device)
        grid_v = torch.linspace(0, 1, H, device=device)
        grid_uu, grid_vv = torch.meshgrid(grid_u, grid_v, indexing='ij')
        base_grid = torch.stack([grid_uu, grid_vv], dim=-1)

        grid_points = coord_function(base_grid.clone())

        grid_x_plus_1 = grid_points.roll(-1, -3)
        grid_x_minus_1 = grid_points.roll(1, -3)
        grid_y_plus_1 = grid_points.roll(-1, -2)
        grid_y_minus_1 = grid_points.roll(1, -2)
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
            - grid_points.unsqueeze(-2),
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

        # Merge unnormalized normals along closed seams
        is_closed_x = torch.allclose(grid_points[..., 0, :, :], grid_points[..., -1, :, :], atol=1e-4, rtol=1e-4)
        if is_closed_x:
            closed_normals = unnormalized_normals[..., 0, :, :] + unnormalized_normals[..., -1, :, :]
            unnormalized_normals[..., 0, :, :] = closed_normals
            unnormalized_normals[..., -1, :, :] = closed_normals

        is_closed_y = torch.allclose(grid_points[..., :, 0, :], grid_points[..., :, -1, :], atol=1e-4, rtol=1e-4)
        if is_closed_y:
            closed_normals = unnormalized_normals[..., :, 0, :] + unnormalized_normals[..., :, -1, :]
            unnormalized_normals[..., :, 0, :] = closed_normals
            unnormalized_normals[..., :, -1, :] = closed_normals

        def _orient_to_ring(pole_normal, ring_normal):
            dot = (pole_normal * ring_normal).sum(-1, keepdim=True)
            return torch.where(dot < 0, -pole_normal, pole_normal)

        is_south_pole = torch.allclose(grid_points[..., :, 0, :], grid_points[..., :1, 0, :], atol=1e-4, rtol=1e-4)
        if is_south_pole:
            pole_normal = unnormalized_normals[..., :, 0, :].sum(-2, keepdim=True)
            ring_normal = unnormalized_normals[..., :, 1, :].sum(-2, keepdim=True)
            pole_normal = _orient_to_ring(pole_normal, ring_normal)
            unnormalized_normals[..., :, 0, :] = pole_normal

        is_north_pole = torch.allclose(grid_points[..., :, -1, :], grid_points[..., :1, -1, :], atol=1e-4, rtol=1e-4)
        if is_north_pole:
            pole_normal = unnormalized_normals[..., :, -1, :].sum(-2, keepdim=True)
            ring_normal = unnormalized_normals[..., :, -2, :].sum(-2, keepdim=True)
            pole_normal = _orient_to_ring(pole_normal, ring_normal)
            unnormalized_normals[..., :, -1, :] = pole_normal

        vertex_normals = -F.normalize(unnormalized_normals, p=2, dim=-1)



        triangle_uvs = grid_to_triangle_vertices(base_grid)
        triangle_corners = grid_to_triangle_vertices(grid_points)
        triangle_normals = grid_to_triangle_vertices(vertex_normals)

        corners_3d = triangle_corners.reshape(-1, 3, 3)
        normals_3d = triangle_normals.reshape(-1, 3, 3)
        uvs_2d = triangle_uvs.reshape(-1, 3, 2)

        from algan.rendering.raytracing.pn_patch import pn_control_points, pn_patch_coefficients, evaluate_pn_patch
        control_points = pn_control_points(corners_3d, normals_3d)
        coefficients = pn_patch_coefficients(control_points)

        bary_coords = torch.tensor([
            [1/3, 1/3],
            [1/2, 0.0],
            [0.0, 1/2],
            [1/2, 1/2],
            [1/6, 1/6],
            [1/6, 2/3],
            [2/3, 1/6]
        ], device=device)

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

        errors = (S_points - P_true).norm(p=2, dim=-1)
        return errors.max()

    @staticmethod
    def _normalize_texture_shape(tex, channels):
        """Normalize a user-supplied texture to ``[T, W, H, channels]``.
        Accepts ``[W, H]`` (single-channel maps only), ``[W, H, channels]``
        or ``[T, W, H, channels]``; ``W`` is the ``u`` axis, ``H`` the ``v``
        axis of the surface's intrinsic coordinates."""
        tex = torch.as_tensor(tex).float()
        if tex.dim() == 2:
            if channels != 1:
                raise ValueError(
                    f"a 2-D texture is only valid for single-channel "
                    f"properties, expected {channels} channels")
            tex = tex.unsqueeze(-1)
        if tex.dim() == 3:
            tex = tex.unsqueeze(0)
        if tex.dim() != 4 or tex.shape[-1] != channels:
            raise ValueError(
                f"texture must have shape [W, H, {channels}] or "
                f"[T, W, H, {channels}], got {tuple(tex.shape)}")
        return tex

    def _build_material_texture(self, textures_dict):
        """Combine per-property maps into one ``[T, W, H, 5]`` material
        texture (channels: reflectivity, roughness, refractive index, and two
        reserved) at the finest common resolution, plus the bitmask of which
        channels are texture-driven (bit i = channel i has a map; unset
        channels keep the per-vertex value in-kernel)."""
        channel_slots = {'reflectivity': 0, 'roughness': 1,
                         'refractive_index': 2}
        device = self.location.device
        texs = {k: self._normalize_texture_shape(v, 1).to(device)
                for k, v in textures_dict.items()}
        T = max(t.shape[0] for t in texs.values())
        W = max(t.shape[1] for t in texs.values())
        H = max(t.shape[2] for t in texs.values())
        combined = torch.zeros((T, W, H, 5), device=device)
        flags = 0
        for name, t in texs.items():
            if t.shape[1:3] != (W, H):
                t = F.interpolate(t.permute(0, 3, 1, 2), size=(W, H),
                                  mode='bilinear', align_corners=True
                                  ).permute(0, 2, 3, 1)
            slot = channel_slots[name]
            combined[..., slot] = t.expand(T, W, H, 1)[..., 0]
            flags |= 1 << slot
        return combined, flags

    def _bake_texture_to_grid(self, tex, channels=1):
        """Resample a texture to the surface grid resolution and flatten it to
        per-vertex values ``[W*H, channels]`` (the same bake the color path
        applies to grid-resolution textures)."""
        t = self._normalize_texture_shape(tex, channels).to(self.location.device)
        if t.shape[0] != 1:
            raise ValueError(
                "glow/glow_radius textures must be static (no time "
                f"dimension), got {tuple(t.shape)}")
        t = F.interpolate(t.permute(0, 3, 1, 2),
                          size=(self.grid_width, self.grid_height),
                          mode='bilinear', align_corners=True
                          ).permute(0, 2, 3, 1)
        return squish(t, -3, -2).squeeze(0)

    def get_memory_used_per_timestep(self):
        # Called once per surface per render batch just to size batches; the
        # result only depends on grid size and shader-param widths (fixed per
        # material), so cache it. Reading the shader params goes through the
        # animated-attribute machinery, which is the expensive part.
        names = tuple(getattr(self, "shader_specific_param_names", ()))
        n_grid = self.grid.location.shape[-2]
        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == (n_grid, names):
            return cache[1]
        n_tri = 2 * max(self.grid_height - 1, 1) * max(self.grid_width - 1, 1)
        n_v = n_tri * 3
        # Grid animation state: location(3*4) + color(5*4) = 32 bytes per grid point.
        # Normal computation intermediates (grid rolls, stack, cross products):
        # ~150 bytes per grid point peak.
        animation_and_intermediates = n_grid * 182
        # Primitive output that persists through rendering:
        # corners(3*3*4=36) + colors(3*5*4=60, cloned) + normals(3*3*4=36) = 132 bytes
        # per triangle, plus RT frame bounds ~8 bytes/vertex.
        primitive_bytes = n_v * 52
        # BVH: ~64 bytes per triangle per timestep.
        bvh_bytes = n_tri * 64
        # Shader params broadcast to vertices.
        shader_bytes = 0
        for _ in self.get_shader_params().values():
            shader_bytes += n_v * _.shape[-1] * 4
        result = int(animation_and_intermediates + primitive_bytes + bvh_bytes + shader_bytes)
        self._memory_per_timestep_cache = ((n_grid, names), result)
        return result

    def get_render_primitives(self):
        grid = unsquish(self.grid.location, -2, self.grid_height)
        if not self.ignore_normals:
            vertex_normals = grid_to_triangle_vertices(compute_grid_vertex_normals(grid))
        else:
            vertex_normals = None
        return self._build_render_primitive(grid, vertex_normals)

    def _build_render_primitive(self, grid, vertex_normals, precomputed_corners=None):
        """Assemble the :class:`TrianglePrimitive` for this surface from an
        already-materialized grid ``[T, W, H, 3]`` and (triangle-gathered)
        vertex normals. ``precomputed_corners`` lets the batched path pass in
        corners gathered on the whole surface stack at once."""
        def expand_grid_to_verts(x):
            if x.shape[-2] == 1:
                x = x.expand(
                    [*[-1 for _ in x.shape[:-2]], grid.shape[-2] * grid.shape[-3], -1]
                )
            x = unsquish(x, -2, self.grid_height)
            return grid_to_triangle_vertices(x)

        def compute_grid_color():
            grid_color = self.grid.color.clone()
            grid_color[..., -1:] *= self.grid.opacity
            grid_color[..., -2:-1] += self.grid.glow
            return grid_color

        uvs = None
        texture_map = None
        material_texture_map = getattr(self, 'material_texture', None)
        material_texture_flags = getattr(self, 'material_texture_flags', 0)
        normal_texture_map = getattr(self, 'normal_texture', None)
        if (self.color_texture is not None or material_texture_map is not None
                or normal_texture_map is not None):
            # Generate UV coordinates for the triangle corners from the base grid
            base_grid = self.get_base_grid()
            uvs = grid_to_triangle_vertices(base_grid).unsqueeze(0)  # [1, num_triangles * 3, 2]
        if self.color_texture is not None:
            texture_map = (self.color_texture
                          ).view(self.color_texture.shape[0], self.texture_height, self.texture_width,
                                 5).as_subclass(Color).mult_opacity(self.opacity.unsqueeze(-2))

        colors = expand_grid_to_verts(compute_grid_color())
        corners = (
            grid_to_triangle_vertices(grid)
            if precomputed_corners is None
            else precomputed_corners
        )
        normals = vertex_normals

        return RENDERER_SETTINGS.triangle_primitive(
            corners=corners,
            colors=colors,
            normals=normals,
            glow=colors[..., -2:-1].as_subclass(torch.Tensor),
            glow_radius=self.grid.glow_radius,
            shader=self.shader,
            uvs=uvs,
            texture_map=texture_map,
            material_texture_map=material_texture_map,
            material_texture_flags=material_texture_flags,
            normal_texture_map=normal_texture_map,
            **{
                k: get_cached_expanded_param(k, lambda k=k, v=v: expand_grid_to_verts(v))
                for k, v in self.grid.get_shader_params().items()
            },
        )

    def coord_function(self, uv: torch.Tensor):
        """Default function used to map intrinsic coordinates to world space to define
        manifold shape. This method is overwritten by subclasses to define new shapes.

        Parameters
        ----------
        uv : torch.Tensor[*, 2]
            Collection of 2-D coordinates to be mapped.

        """
        return torch.cat(((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1)

    def normal_function(self, uv):
        """Default function used to map intrinsic coordinates to world space normals to define
        manifold normal directions. This method is overwritten by subclasses to define new shapes.

        Parameters
        ----------
        uv : torch.Tensor[*, 2]
            Collection of 2-D coordinates to be mapped.

        """
        return OUT

    def get_base_grid(self):
        device = self.grid.location.device if hasattr(self, "grid") and hasattr(self.grid, "location") else None
        if not hasattr(self, "_cached_base_grid") or self._cached_base_grid.device != device:
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
        return self._cached_base_grid

    def set_shape_to(self, other_surface: "Surface"):
        """Changes this surface's shape to the shape defined by another surface's :meth:`~.Surface.coord_function` .

        Parameters
        ----------
        other_surface
            The surface from which to get coord_function.

        """
        with Sync():
            self.set_location_by_function(other_surface.coord_function)
            # TODO setting normals currently doesn't work, implement it.
            # self.set_normal_by_function(other_surface.normal_function)

    def set_location_by_function(self, function):
        new_loc = function(squish(self.get_base_grid(), -3, -2).unsqueeze(0)) + self.location
        self.grid.location = new_loc
        return self

    def set_normal_by_function(self, function):
        new_normals = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(
            unsquish(self.triangles.corners.location, -2, 3),
            normals=new_normals,
            add_to_scene=False,
        )
        with Sync():
            self.triangles.basis = new_triangles.basis
            self.triangles.corners.basis = new_triangles.corners.basis
        return self

    def get_default_color(self):
        return GREEN

    def set_color_by_function(self, function):
        new_color = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(
            unsquish(self.triangles.corners.location, -2, 3),
            color=new_color,
            normals=None,
            add_to_scene=False,
        )
        with Sync():
            self.triangles.color = new_triangles.color
            self.triangles.corners.color = new_triangles.corners.color
        return self

    def set_color_by_texture(self, rgba_array_or_file_path):
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
