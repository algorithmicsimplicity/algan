import math

import torch
import torch.nn.functional as F

from algan.mobs.renderable import Renderable
from algan.utils.tensor_utils import broadcast_cross_product
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.animation.animation_contexts import Sync, Off
from algan.constants.color import *
from algan.constants.spatial import ORIGIN, OUT
from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.file_utils import get_image
from algan.utils.tensor_utils import unsqueeze_left, squish, unsquish, cast_to_tensor


def grid_to_triangle_vertices(grid):
    if grid.dim() == 1:
        return grid
    transformed_grid = grid

    triangle_corners = torch.stack(
        (
            torch.stack(
                (
                    transformed_grid[..., :-1, :-1, :],
                    transformed_grid[..., :-1, 1:, :],
                    transformed_grid[..., 1:, :-1, :],
                ),
                -2,
            ),
            torch.stack(
                (
                    transformed_grid[..., 1:, :-1, :],
                    transformed_grid[..., :-1, 1:, :],
                    transformed_grid[..., 1:, 1:, :],
                ),
                -2,
            ),
        ),
        -3,
    )
    return triangle_corners.reshape(
        *grid.shape[:-3], -1, transformed_grid.shape[-1]
    )  # unsquish(triangle_corners, -2, 3)


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
        ignore_normals=False,
        tolerance=0.01,
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
                low = 4
                high = 200
                best_N = high
                while low <= high:
                    mid = (low + high) // 2
                    W = mid
                    H = max(4, int(mid * grid_aspect_ratio))
                    try:
                        error = self._compute_pn_error(coord_function, W, H)
                        if error < tolerance * scale:
                            best_N = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                grid_width = best_N
                grid_height = max(4, int(best_N * grid_aspect_ratio))
            else:
                # Independent rectangular search
                # 1. Search for best grid_width (W) with grid_height (H) set to a high resolution (200)
                low = 4
                high = 200
                best_W = high
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        error = self._compute_pn_error(coord_function, mid, 200)
                        if error < tolerance * scale:
                            best_W = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                
                # 2. Search for best grid_height (H) with grid_width (W) set to a high resolution (200)
                low = 4
                high = 200
                best_H = high
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        error = self._compute_pn_error(coord_function, 200, mid)
                        if error < tolerance * scale:
                            best_H = mid
                            high = mid - 1
                        else:
                            low = mid + 1
                    except Exception:
                        low = mid + 1
                
                # Joint error correction loop
                try:
                    joint_error = self._compute_pn_error(coord_function, best_W, best_H)
                    while joint_error > tolerance * scale and (best_W < 200 or best_H < 200):
                        ratio = joint_error / (tolerance * scale)
                        factor = max(1.15, float(torch.sqrt(ratio).item()))
                        if best_W < 200:
                            best_W = min(200, int(best_W * factor) + 1)
                        if best_H < 200:
                            best_H = min(200, int(best_H * factor) + 1)
                        joint_error = self._compute_pn_error(coord_function, best_W, best_H)
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

    def get_memory_used_per_timestep(self):
        n_grid = self.grid.location.shape[-2]
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
        return int(animation_and_intermediates + primitive_bytes + bvh_bytes + shader_bytes)

    def get_render_primitives(self):
        self.grid.set_time_inds_to(self)
        grid = unsquish(self.grid.location, -2, self.grid_height)
        if not self.ignore_normals:
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

            # Merge unnormalized normals along closed seams
            is_closed_x = torch.allclose(grid[..., 0, :, :], grid[..., -1, :, :], atol=1e-4, rtol=1e-4)
            if is_closed_x:
                closed_normals = unnormalized_normals[..., 0, :, :] + unnormalized_normals[..., -1, :, :]
                unnormalized_normals[..., 0, :, :] = closed_normals
                unnormalized_normals[..., -1, :, :] = closed_normals

            is_closed_y = torch.allclose(grid[..., :, 0, :], grid[..., :, -1, :], atol=1e-4, rtol=1e-4)
            if is_closed_y:
                closed_normals = unnormalized_normals[..., :, 0, :] + unnormalized_normals[..., :, -1, :]
                unnormalized_normals[..., :, 0, :] = closed_normals
                unnormalized_normals[..., :, -1, :] = closed_normals

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

            is_south_pole = torch.allclose(grid[..., :, 0, :], grid[..., :1, 0, :], atol=1e-4, rtol=1e-4)
            if is_south_pole:
                pole_normal = unnormalized_normals[..., :, 0, :].sum(-2, keepdim=True)
                ring_normal = unnormalized_normals[..., :, 1, :].sum(-2, keepdim=True)
                pole_normal = _orient_to_ring(pole_normal, ring_normal)
                unnormalized_normals[..., :, 0, :] = pole_normal

            is_north_pole = torch.allclose(grid[..., :, -1, :], grid[..., :1, -1, :], atol=1e-4, rtol=1e-4)
            if is_north_pole:
                pole_normal = unnormalized_normals[..., :, -1, :].sum(-2, keepdim=True)
                ring_normal = unnormalized_normals[..., :, -2, :].sum(-2, keepdim=True)
                pole_normal = _orient_to_ring(pole_normal, ring_normal)
                unnormalized_normals[..., :, -1, :] = pole_normal

            vertex_normals = -F.normalize(unnormalized_normals, p=2, dim=-1)
            vertex_normals = grid_to_triangle_vertices(vertex_normals)
        else:
            vertex_normals = None

        def expand_grid_to_verts(x):
            if x.shape[-2] == 1:
                x = x.expand(
                    [*[-1 for _ in x.shape[:-2]], grid.shape[-2] * grid.shape[-3], -1]
                )
            x = unsquish(x, -2, self.grid_height)
            return grid_to_triangle_vertices(x)

        grid_color = self.grid.color.clone()
        grid_color[..., -1:] *= self.grid.opacity
        grid_color[..., -2:-1] += self.grid.glow
        uvs = None
        texture_map = None
        if self.color_texture is not None:
            # Generate UV coordinates for the triangle corners from the base grid
            base_grid = self.get_base_grid()
            uvs = grid_to_triangle_vertices(base_grid).unsqueeze(0)  # [1, num_triangles * 3, 2]
            texture_map = (self.color_texture
                          ).view(self.color_texture.shape[0], self.texture_height, self.texture_width,
                                 5).as_subclass(Color).mult_opacity(self.opacity.unsqueeze(-2))

        colors = expand_grid_to_verts(grid_color)
        return TrianglePrimitive(
            corners=grid_to_triangle_vertices(grid),
            colors=colors,
            normals=vertex_normals,
            glow=colors[..., -2:-1].as_subclass(torch.Tensor),
            glow_radius=self.grid.glow_radius,
            shader=self.shader,
            uvs=uvs,
            texture_map=texture_map,
            **{
                k: expand_grid_to_verts(v)
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
        grid = torch.stack(
            (
                torch.linspace(0, 1, self.grid_width)
                .view(-1, 1)
                .expand(-1, self.grid_height),
                torch.linspace(0, 1, self.grid_height)
                .view(1, -1)
                .expand(self.grid_width, -1),
            ),
            -1,
        )
        return grid

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
