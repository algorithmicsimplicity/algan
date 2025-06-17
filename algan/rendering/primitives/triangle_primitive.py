import torch

import algan
from algan.constants.color import BLUE
from algan.defaults.device_defaults import DEFAULT_RENDER_DEVICE, DEFAULT_DEVICE
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.utils.tensor_utils import broadcast_all
from algan.utils.tensor_utils import dot_product, squish, broadcast_gather, expand_as_left, unsquish, cast_to_tensor


def get_bary_coordinates(triangle_corners, fragment_x, fragment_y, aa_offsets):
    cs = triangle_corners
    y23 = (cs[..., 1, 1] - cs[..., 2, 1]).unsqueeze(-1)
    x13 = (cs[..., 0, 0] - cs[..., 2, 0]).unsqueeze(-1)
    x32 = (cs[..., 2, 0] - cs[..., 1, 0]).unsqueeze(-1)
    y13 = (cs[..., 0, 1] - cs[..., 2, 1]).unsqueeze(-1)
    y31 = (cs[..., 2, 1] - cs[..., 0, 1]).unsqueeze(-1)
    denom = (y23 * x13 + x32 * y13)
    inv_denom = 1 / denom

    def get_coords(anti_alias_offset, out=None):
        px3 = (fragment_x - cs[..., 2, 0].unsqueeze(-1)) + anti_alias_offset[0]
        py3 = (fragment_y - cs[..., 2, 1].unsqueeze(-1)) + anti_alias_offset[1]

        w1 = (((x32 * py3) + y23 * px3) * inv_denom).nan_to_num_(nan=-1.0)
        w2 = (((x13 * py3) + y31 * px3) * inv_denom).nan_to_num_(nan=-1.0)
        w3 = (1 - (w1 + w2))
        return torch.stack((w1, w2, w3), -2)

    return torch.stack([get_coords(_) for _ in aa_offsets])


def interpolate_triangle_corners(self, interpolation_coord, property):
    ws = interpolation_coord
    x = property
    out = self.get_tensor([*x.shape[:-2], x.shape[-1]])
    out[:] = 0
    for i in range(ws.shape[-2]):
        torch.addcmul(out, x[..., i, :], ws[..., i, :], out=out)
    return out


class TrianglePrimitive(RenderPrimitive):
    def __init__(self, corners=None, colors=BLUE, opacity=1, normals=None, perimeter_points=None,
                 reverse_perimeter=False, triangle_collection=None, glow=0, shader=None, **shader_kwargs):
        glow = cast_to_tensor(glow).to(DEFAULT_DEVICE)
        opacity = cast_to_tensor(opacity).to(DEFAULT_DEVICE)
        """
        corners: Tensor[batch[*], num_corners[3], corner_locations[3]]
            Location of triangle vertices/corners in 3d world space.
        constants: Tensor[batch[*], num_corners[3], RGBA[3|4]]
            RGBA constants values for each vertex. A value is optional, if not provided then transparency parameter will be used.
        transparency: Tensor[batch[*], num_corners[3], 1]
            Transparency value for each vertex. Only used if A is not provided in constants value.
        triangle_collection: Iterable[TrianglePrimitive]
            Collection of n Triangles, if provided then this instance will be a batch containing the corners and constants
            of all n triangles in one. If provided, all other parameters are ignored.
        """
        self.reverse_perimeter = reverse_perimeter
        self.min_interpolation_coord = 0
        if triangle_collection is not None:
            self.shader = triangle_collection[0].shader
            self.corners, self.colors, self.normals, *self.shader_param_values = (
            unsquish(torch.cat(_, 1), -2, 3).to(DEFAULT_RENDER_DEVICE, non_blocking=True) for _ in
            zip(*(broadcast_all([triangle.corners, triangle.colors, triangle.normals,
                                 *triangle.shader_param_values], ignored_dims=[-1]) for triangle in triangle_collection)))
            return
        self.corners = corners
        if normals is None:
            normals = torch.zeros_like(corners)
        colors, opacity, glow = broadcast_all([colors, opacity, glow], ignored_dims=[-1])
        self.colors = colors.clone()
        self.colors[...,-2:-1] += glow
        self.colors[..., -1:] *= opacity
        self.normals = normals
        self.shader_param_values = broadcast_all([colors, *shader_kwargs.values()], ignored_dims=[-1])[1:]

        if shader is None:
            shader = algan.defaults.render_defaults.DEFAULT_SHADER
        self.shader = shader

    def get_batch_identifier(self):
        return f'{self.__class__}_{id(self.shader)}'

    def get_interpolation_coordinates(self, vertex_corners, fragment_x, fragment_y, aa_offsets, repeats_inds):
        return get_bary_coordinates(self.expand_verts_to_frags(vertex_corners, repeats_inds.unsqueeze(-1), -3), fragment_x, fragment_y, aa_offsets)

    def interpolate_property(self, interpolation_coord, property, repeats_inds):
        return interpolate_triangle_corners(self, interpolation_coord, self.expand_verts_to_frags(property, repeats_inds.unsqueeze(-1), -3))


def get_tangents(x):
    return torch.cat((x[:, 1:] - x[:, -1:], x[:, :1] - x[:, -1:]), 1), torch.cat((x[:, -1:] - x[:, :1], x[:, -1:] - x[:, 1:]), 1)
