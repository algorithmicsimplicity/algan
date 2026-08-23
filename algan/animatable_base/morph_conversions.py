"""Primitive-family adapters for :meth:`Mob.become`.

Cross-family geometry is converted to an internal cubic-PN triangle soup.  The
registry is intentionally internal so new primitive families can participate
without adding pairwise conversion code to ``mob_morph``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import torch
import torch.nn.functional as F

from algan.constants.color import Color
from algan.mobs.pn_mesh import PNMesh
from algan.utils.tensor_utils import cast_to_tensor, unsquish


class MorphConversionError(RuntimeError):
    """A registered family could not be converted to the PN morph medium."""


@dataclass(frozen=True)
class MorphConversion:
    to_pn_soup: Callable
    pre_animate: Callable | None = None
    post_animate: Callable | None = None


_MORPH_CONVERSIONS: dict[str, MorphConversion] = {}


def register_morph_conversion(
    family,
    *,
    to_pn_soup,
    pre_animate=None,
    post_animate=None,
):
    """Register one primitive family's PN-soup adapter and choreography hooks."""
    _MORPH_CONVERSIONS[family] = MorphConversion(
        to_pn_soup=to_pn_soup,
        pre_animate=pre_animate,
        post_animate=post_animate,
    )


def get_morph_conversion(family):
    return _MORPH_CONVERSIONS.get(family)


def convert_to_pn_soup(mob, *, add_to_scene=False):
    conversion = get_morph_conversion(mob._morph_family)
    if conversion is None:
        raise NotImplementedError(
            f"No PN morph conversion is registered for family {mob._morph_family!r}"
        )
    try:
        return conversion.to_pn_soup(mob, add_to_scene=add_to_scene)
    except MorphConversionError:
        raise
    except Exception as exc:
        raise MorphConversionError(
            f"Could not convert {type(mob).__name__} to the PN morph medium"
        ) from exc


def _expand_rows(value, row_count):
    value = cast_to_tensor(value)
    if value.shape[-2] == row_count:
        return value
    if value.shape[-2] == 1:
        return value.expand(*value.shape[:-2], row_count, value.shape[-1])
    if row_count % value.shape[-2] == 0:
        return value.repeat_interleave(row_count // value.shape[-2], dim=-2)
    raise MorphConversionError(
        f"Cannot broadcast {value.shape[-2]} rows over {row_count} PN corners"
    )


def _flat_corner_normals(corners):
    triangles = unsquish(corners, -2, 3)
    face_normals = F.normalize(
        torch.cross(
            triangles[..., 1, :] - triangles[..., 0, :],
            triangles[..., 2, :] - triangles[..., 0, :],
            dim=-1,
        ),
        p=2,
        dim=-1,
    )
    return (
        face_normals.unsqueeze(-2)
        .expand(*face_normals.shape[:-1], 3, 3)
        .reshape(corners.shape)
    )


def _carrier_values(carrier, corners):
    rows = corners.shape[-2]
    return {
        "color": _expand_rows(carrier.color, rows).as_subclass(Color),
        "opacity": _expand_rows(carrier.opacity, rows),
        "glow": _expand_rows(carrier.glow, rows),
        "shader": carrier.shader,
        "shader_params": {
            name: _expand_rows(value, rows)
            for name, value in carrier.get_shader_params().items()
        },
    }


def _grid_to_pn_soup(surface, *, add_to_scene=False):
    from algan.mobs.surfaces.surface import (
        compute_grid_vertex_normals,
        grid_to_triangle_vertices,
        surface_weld_flags,
    )

    grid = surface._reshape_grid_for_render(surface.grid.location)
    # THE WELD FLAGS BELONG TO THE GRID, NOT TO THE CALLER. Both this path and
    # Surface.get_render_primitives turn the same grid into triangles through
    # grid_to_triangle_vertices, and the weld decides how many triangles that
    # is -- a Sphere's two pole fans collapse and its u-seam does not. This path
    # used to omit the argument and take the default, so with
    # ALGAN_WELD_SURFACE_SEAMS on a Sphere MORPHED from one triangulation and
    # RENDERED as another. Asking surface_weld_flags here is what keeps the two
    # answers the same question. (Off by default, so this is inert until the
    # gate flips; DESIGN_mesh_identity.md ss3.1.)
    weld = surface_weld_flags(grid)
    corners = surface._flatten_packed_triangle_vertices(
        grid_to_triangle_vertices(grid, weld)
    )
    if surface.ignore_normals:
        normals = torch.zeros_like(corners)
    else:
        normals = surface._flatten_packed_triangle_vertices(
            grid_to_triangle_vertices(compute_grid_vertex_normals(grid), weld)
        )

    def gather(value):
        value = _expand_rows(value, surface.grid.location.shape[-2])
        return surface._flatten_packed_triangle_vertices(
            grid_to_triangle_vertices(surface._reshape_grid_for_render(value), weld)
        )

    return PNMesh(
        corners,
        normals,
        color=gather(surface.grid.color).as_subclass(Color),
        opacity=gather(surface.grid.opacity),
        glow=gather(surface.grid.glow),
        shader=surface.shader,
        shader_params={
            name: gather(value)
            for name, value in surface.grid.get_shader_params().items()
        },
        render_tolerance=surface._render_tolerance,
        render_tolerance_pixels=surface._render_tolerance_pixels,
        # The soup approximates the analytic surface exactly as well as the
        # patches it is made of do, so it inherits their accuracy and dices the
        # same way (``test_surface_and_pn_conversion_render_pixel_identically``).
        geometry_slack_ratio=surface._geometry_slack_ratio,
        scene=surface.scene,
        add_to_scene=add_to_scene,
    )


def _mesh_carriers(mob):
    from algan.mobs.shapes_2d import TriangleTriangulated, TriangleVertices
    from algan.mobs.shapes_3d import Polyhedron
    from algan.mobs.three_d_models.mesh import TriangleMesh

    if isinstance(mob, TriangleMesh):
        return [mob.grid]
    if isinstance(mob, TriangleVertices):
        return [mob]
    if isinstance(mob, TriangleTriangulated):
        return [mob.corners]
    if isinstance(mob, Polyhedron):
        return list(mob._face_primitive_mobs())
    grid = getattr(mob, "grid", None)
    if grid is not None and grid.location.shape[-2] % 3 == 0:
        return [grid]
    if mob.location.shape[-2] % 3 == 0:
        return [mob]
    raise MorphConversionError(f"Unsupported mesh Mob {type(mob).__name__}")


def _mesh_to_pn_soup(mob, *, add_to_scene=False):
    carriers = _mesh_carriers(mob)
    if not carriers:
        raise MorphConversionError("Mesh contains no triangle geometry")
    corners = torch.cat([carrier.location for carrier in carriers], dim=-2)
    if corners.shape[-2] == 0 or corners.shape[-2] % 3:
        raise MorphConversionError("Mesh corner rows do not form complete triangles")
    normals = _flat_corner_normals(corners)

    colors = []
    opacities = []
    glows = []
    for carrier in carriers:
        rows = carrier.location.shape[-2]
        colors.append(_expand_rows(carrier.color, rows))
        opacities.append(_expand_rows(carrier.opacity, rows))
        glows.append(_expand_rows(carrier.glow, rows))
    carrier = carriers[0]
    return PNMesh(
        corners,
        normals,
        color=torch.cat(colors, dim=-2).as_subclass(Color),
        opacity=torch.cat(opacities, dim=-2),
        glow=torch.cat(glows, dim=-2),
        shader=carrier.shader,
        shader_params={
            name: _expand_rows(value, corners.shape[-2])
            for name, value in carrier.get_shader_params().items()
        },
        scene=mob.scene,
        add_to_scene=add_to_scene,
    )


def _sampled_signed_area(params):
    from algan.mobs.triangulated_bezier_circuit import (
        get_points_along_cubic_bezier,
    )

    # Sum each disconnected loop separately so jumps between glyph holes do not
    # contribute a fictitious shoelace edge.
    segments = params.transpose(-3, -2)
    starts = [0]
    for index in range(1, len(segments)):
        if not torch.allclose(segments[index - 1, -1], segments[index, 0], atol=1e-5):
            starts.append(index)
    starts.append(len(segments))
    area = params.new_zeros(())
    for start, end in zip(starts, starts[1:]):
        sampled = get_points_along_cubic_bezier(segments[start:end].transpose(-3, -2))[
            0
        ].reshape(-1, 2)
        if len(sampled) < 3:
            continue
        area = (
            area
            + 0.5
            * (
                sampled[:, 0] * sampled.roll(-1, 0)[:, 1]
                - sampled[:, 1] * sampled.roll(-1, 0)[:, 0]
            ).sum()
        )
    return area


def _circuit_batches(circuit):
    segments = unsquish(circuit.control_points.location[0], -2, 4)
    sizes = circuit.control_points.parent_batch_sizes
    if sizes is None:
        batches = [segments]
    else:
        if bool((sizes % 4 != 0).any()):
            raise MorphConversionError("Cubic parent batches are not divisible by four")
        batches = list(segments.split((sizes // 4).tolist(), dim=-3))
    count = len(batches)
    locations = _expand_rows(circuit.location, count)[0]
    bases = _expand_rows(circuit.basis, count)[0].reshape(count, 3, 3)
    return batches, locations, bases


def _bezier_to_pn_soup(circuit, *, add_to_scene=False):
    from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit

    triangle_budget = 2048
    batches, locations, bases = _circuit_batches(circuit)
    all_corners = []
    all_normals = []
    all_colors = []
    all_opacities = []
    all_glows = []
    batch_corner_counts = []

    for index, (control_points, location, basis) in enumerate(
        zip(batches, locations, bases)
    ):
        scale = basis[0].norm().clamp_min(1e-6)
        e0 = F.normalize(basis[0], p=2, dim=-1)
        e1 = F.normalize(basis[1], p=2, dim=-1)
        plane_normal = F.normalize(basis[2], p=2, dim=-1)
        relative = control_points - location
        projected = torch.stack(
            (
                (relative * e0).sum(-1) / scale,
                (relative * e1).sum(-1) / scale,
            ),
            dim=-1,
        )
        params = projected.transpose(-3, -2).contiguous()
        if _sampled_signed_area(params) < 0:
            params = params.flip(0).flip(-2).contiguous()

        extent = (projected.amax((-3, -2)) - projected.amin((-3, -2))).clamp_min(1e-4)
        tile_size = float(extent.max() / sqrt(triangle_budget / 2))
        try:
            triangulated = TriangulatedBezierCircuit(
                params,
                invert=False,
                border_width=0,
                tile_size=max(tile_size, 1e-4),
                hash_keys=params,
                use_cache=True,
                reverse_points=False,
                scene=circuit.scene,
                add_to_scene=False,
            )
        except RuntimeError as exc:
            # A PATH THAT ENCLOSES NOTHING IS AN EMPTY FILL, NOT A FAILURE.
            # Two crossed lines, a Cross, a DashedLine, an Axes, the rule of a
            # MathTex: the tiler emits no tiles and its packing step cannot
            # concatenate an empty list, so the whole conversion used to raise
            # and `Axes().become(Sphere())` was simply unavailable. What such a
            # circuit draws is its stroke, and the soup zeroes an unfilled
            # circuit's opacity a few lines below in any case -- so stand one
            # degenerate triangle at the path's centroid, which contributes the
            # rows the morph interpolates and no visible area. Matched on the
            # message so an unrelated RuntimeError still reaches the caller.
            if "non-empty list of Tensors" not in str(exc):
                raise
            local_2d = projected.reshape(-1, 2).mean(0).expand(1, 3, 2)
        else:
            triangle_root = triangulated.tiles.children[0]
            local = triangle_root.corners.location[0].reshape(-1, 3, 3)
            local_2d = local[..., :2]
        world = (
            location + local_2d[..., 0:1] * e0 * scale + local_2d[..., 1:2] * e1 * scale
        ).reshape(1, -1, 3)

        # Use one sign for the whole tiling. Per-triangle normal flips would
        # make formerly shared PN boundary curves disagree and open cracks.
        triangles_world = world.reshape(-1, 3, 3)
        winding = torch.cross(
            triangles_world[:, 1] - triangles_world[:, 0],
            triangles_world[:, 2] - triangles_world[:, 0],
            dim=-1,
        ).sum(0)
        if (winding * plane_normal).sum() < 0:
            plane_normal = -plane_normal
        normals = plane_normal.view(1, 1, 3).expand_as(world)

        rows = world.shape[-2]
        batch_corner_counts.append(rows)
        all_corners.append(world)
        all_normals.append(normals)
        color = _expand_rows(circuit.color, len(batches))[:, index : index + 1]
        opacity = _expand_rows(circuit.opacity, len(batches))[:, index : index + 1]
        glow = _expand_rows(circuit.glow, len(batches))[:, index : index + 1]
        if not getattr(circuit, "filled", True) or getattr(circuit, "empty", False):
            opacity = torch.zeros_like(opacity)
        all_colors.append(color.expand(-1, rows, -1))
        all_opacities.append(opacity.expand(-1, rows, -1))
        all_glows.append(glow.expand(-1, rows, -1))

    corners = torch.cat(all_corners, dim=-2)
    normals = torch.cat(all_normals, dim=-2)
    params = {}
    for name, value in circuit.get_shader_params().items():
        value = _expand_rows(value, len(batch_corner_counts))
        params[name] = torch.cat(
            [
                value[:, index : index + 1].expand(-1, rows, -1)
                for index, rows in enumerate(batch_corner_counts)
            ],
            dim=-2,
        )
    return PNMesh(
        corners,
        normals,
        color=torch.cat(all_colors, dim=-2).as_subclass(Color),
        opacity=torch.cat(all_opacities, dim=-2),
        glow=torch.cat(all_glows, dim=-2),
        shader=circuit.shader,
        shader_params=params,
        scene=circuit.scene,
        add_to_scene=add_to_scene,
    )


def _pn_soup_identity(mob, *, add_to_scene=False):
    return PNMesh(
        mob.location.clone(),
        mob.normals.clone(),
        color=mob.color.clone(),
        opacity=mob.opacity.clone(),
        glow=mob.glow.clone(),
        shader=mob.shader,
        shader_params={k: v.clone() for k, v in mob.get_shader_params().items()},
        render_tolerance=mob.render_tolerance,
        render_tolerance_pixels=mob.render_tolerance_pixels,
        scene=mob.scene,
        add_to_scene=add_to_scene,
    )


def _border_to_zero(circuit, target=None):
    circuit.set_non_recursive(border_width=torch.zeros_like(circuit.border_width))


def _border_to_target(circuit, target):
    circuit.set_non_recursive(border_width=target.border_width)


register_morph_conversion("grid", to_pn_soup=_grid_to_pn_soup)
register_morph_conversion("mesh", to_pn_soup=_mesh_to_pn_soup)
register_morph_conversion(
    "bezier",
    to_pn_soup=_bezier_to_pn_soup,
    pre_animate=_border_to_zero,
    post_animate=_border_to_target,
)
register_morph_conversion("pn_soup", to_pn_soup=_pn_soup_identity)
