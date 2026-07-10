import torch
from torch.export.dynamic_shapes import Dim

from algan.constants.color import BLUE
from algan.settings.defaults import *
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.utils.tensor_utils import broadcast_all
from algan.utils.tensor_utils import (
    dot_product,
    squish,
    broadcast_gather,
    expand_as_left,
    unsquish,
    cast_to_tensor,
)


#@compiled
t = 2
num_frag = 100000
#@exported(example_inputs=(torch.randn((t, num_frag, 3, 2)), torch.randn((t, num_frag, 1)), torch.randn((t, num_frag, 1))),
#          dynamic_shapes=[(Dim.AUTO, Dim.AUTO, Dim.STATIC, Dim.STATIC), (Dim.AUTO, Dim.AUTO, Dim.STATIC), (Dim.AUTO, Dim.AUTO, Dim.STATIC)])
def _get_bary_coordinates_part_1(triangle_corners, fragment_x, fragment_y):
    #stream = torch.cuda.Stream()
    #with torch.cuda.stream(stream):
    cs = triangle_corners
    fragment_x -= cs[..., 2, 0].unsqueeze(-1)
    fragment_y -= cs[..., 2, 1].unsqueeze(-1)
    # y23 = (cs[..., 1, 1] - cs[..., 2, 1]).unsqueeze(-1)
    y23 = torch.subtract(cs[..., 1, 1], cs[..., 2, 1], out=cs[..., 1, 1]).unsqueeze(-1)
    # x13 = (cs[..., 0, 0] - cs[..., 2, 0]).unsqueeze(-1)
    x13 = torch.subtract(cs[..., 0, 0], cs[..., 2, 0], out=cs[..., 0, 0]).unsqueeze(-1)
    # x32 = (cs[..., 2, 0] - cs[..., 1, 0]).unsqueeze(-1)
    x32 = torch.subtract(cs[..., 2, 0], cs[..., 1, 0], out=cs[..., 2, 0]).unsqueeze(-1)
    # y13 = (cs[..., 0, 1] - cs[..., 2, 1]).unsqueeze(-1)
    y13 = torch.subtract(cs[..., 0, 1], cs[..., 2, 1], out=cs[..., 1, 0]).unsqueeze(-1)
    # y31 = (cs[..., 2, 1] - cs[..., 0, 1]).unsqueeze(-1)
    y31 = torch.subtract(cs[..., 2, 1], cs[..., 0, 1], out=cs[..., 0, 1]).unsqueeze(-1)
    # denom = (y23 * x13 + x32 * y13)
    # inv_denom = 1 / denom
    denom = torch.mul(y23, x13, out=cs[..., 2, 1].unsqueeze(-1))
    denom = torch.addcmul(denom, x32, y13, value=1, out=denom)
    inv_denom = torch.div(1, denom, out=denom)
    return denom, inv_denom, x13, x32, y13, y31, y23


def _get_bary_coordinates_part_2(fragment_x, fragment_y, inv_denom, x13, x32, y13, y31, y23):
    px3 = fragment_x  # + anti_alias_offset[0]
    py3 = fragment_y  # + anti_alias_offset[1]

    # w2 = (((x13 * py3) + y31 * px3) * inv_denom).nan_to_num_(nan=-1.0)
    w2 = torch.mul(y31, px3, out=y31)
    w2 = torch.addcmul(w2, x13, py3, out=w2)
    w2 *= inv_denom
    # w1 = (((x32 * py3) + y23 * px3) * inv_denom).nan_to_num_(nan=-1.0)
    w1 = torch.mul(x32, py3, out=x13)
    w1 = torch.addcmul(w1, y23, px3, out=w1)
    w1 *= inv_denom
    # w3 = (1 - (w1 + w2))
    w3 = torch.add(w1, w2, out=y13)
    w3 *= -1
    w3 += 1


def get_bary_coordinates(triangle_corners, fragment_x, fragment_y):
    denom, inv_denom, x13, x32, y13, y31, y23 = _get_bary_coordinates_part_1(triangle_corners, fragment_x, fragment_y)
    _get_bary_coordinates_part_2(fragment_x, fragment_y, inv_denom, x13, x32, y13, y31, y23)


    # We carefully wrote w1, w2, w3 into the first 3 positions of cs, so we can just return that and save ourselves a stack.
    return triangle_corners.view(*triangle_corners.shape[:-2], -1)[..., :3].unsqueeze(-1)
        # return torch.stack((w1, w2, w3), -2)


#@compiled
def interpolate_triangle_corners(self, interpolation_coord, property):
    ws = interpolation_coord
    x = property
    out = self.get_tensor([*x.shape[:-2], x.shape[-1]], persist=True)
    torch.mul(x[..., 0, :], ws[..., 0, :], out=out)
    for i in range(1, ws.shape[-2]):
        torch.addcmul(out, x[..., i, :], ws[..., i, :], out=out)
    return out


class TrianglePrimitive(RenderPrimitive):
    def __init__(
        self,
        corners=None,
        colors=BLUE,
        opacity=1,
        normals=None,
        perimeter_points=None,
        reverse_perimeter=False,
        triangle_collection=None,
        glow=0,
        glow_radius=0.2,
        shader=None,
        uvs=None,
        texture_map=None,
        material_texture_map=None,
        material_texture_flags=0,
        normal_texture_map=None,
        **shader_kwargs,
    ):
        device = COMPUTING_DEFAULTS.animation_device
        glow = cast_to_tensor(glow).to(device)
        glow_radius = cast_to_tensor(glow_radius).to(device)
        opacity = cast_to_tensor(opacity).to(device)
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
        self.uvs = None
        self.texture_map = None
        self.material_texture_map = None
        self.material_texture_flags = 0
        self.normal_texture_map = None

        if triangle_collection is not None:
            self.shader = triangle_collection[0].shader
            # Names of the positional shader_param_values, in the same order
            # (kept so the ray tracer can map them to its material slots).
            self.shader_param_names = getattr(
                triangle_collection[0], "shader_param_names", [])
            self.corners, self.colors, self.normals, self.glow_radius, *self.shader_param_values = (
                unsquish(torch.cat(_, 1), -2, 3).to(COMPUTING_DEFAULTS.render_device)
                for _ in zip(
                    *(
                        broadcast_all(
                            [
                                triangle.corners,
                                triangle.colors,
                                triangle.normals,
                                triangle.glow_radius,
                                *triangle.shader_param_values,
                            ],
                            ignored_dims=[-1],
                        )
                        for triangle in triangle_collection
                    )
                )
            )

            # Check if any triangle in the collection has uvs or texture_map
            has_uvs = any(getattr(t, "uvs", None) is not None for t in triangle_collection)
            if has_uvs:
                uv_list = []
                for triangle in triangle_collection:
                    uv = getattr(triangle, "uvs", None)
                    if uv is None:
                        uv = torch.zeros((*triangle.corners.shape[:-1], 2), device=triangle.corners.device)
                    else:
                        if uv.dim() == 4:
                            uv = squish(uv, -3, -2)
                        uv = uv.to(triangle.corners.device)
                    uv_list.append(uv)
                merged_uvs = []
                for i, triangle in enumerate(triangle_collection):
                    cor, uv = broadcast_all([triangle.corners, uv_list[i]], ignored_dims=[-1])
                    merged_uvs.append(uv)
                self.uvs = unsquish(torch.cat(merged_uvs, 1), -2, 3)

            for triangle in triangle_collection:
                tex = getattr(triangle, "texture_map", None)
                if tex is not None:
                    self.texture_map = tex.to(self.corners.device)
                    break

            # Texture maps cannot be concatenated across primitives (each map
            # keeps its own resolution), so like texture_map above they are
            # taken from the first primitive that has one. The scene batcher
            # puts every textured primitive in its own singleton collection,
            # which makes this exact.
            for triangle in triangle_collection:
                tex = getattr(triangle, "material_texture_map", None)
                if tex is not None:
                    self.material_texture_map = tex.to(self.corners.device)
                    self.material_texture_flags = getattr(
                        triangle, "material_texture_flags", 0)
                    break
            for triangle in triangle_collection:
                tex = getattr(triangle, "normal_texture_map", None)
                if tex is not None:
                    self.normal_texture_map = tex.to(self.corners.device)
                    break

            self.padding = 1
            return

        self.corners = corners
        if normals is None:
            normals = torch.zeros_like(corners)
        colors, opacity, glow, glow_radius = broadcast_all(
            [colors, opacity, glow, glow_radius], ignored_dims=[-1]
        )
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow
        self.colors[..., -1:] *= opacity
        self.glow = glow
        self.glow_radius = glow_radius
        self.normals = normals
        self.shader_param_names = list(shader_kwargs.keys())
        self.shader_param_values = broadcast_all(
            [colors, *shader_kwargs.values()], ignored_dims=[-1]
        )[1:]

        if uvs is not None:
            if uvs.dim() == 3:
                uvs = unsquish(uvs, -2, 3)
            self.uvs = uvs.to(self.corners.device)
        self.texture_map = texture_map.to(self.corners.device) if texture_map is not None else None
        self.material_texture_map = (material_texture_map.to(self.corners.device)
                                     if material_texture_map is not None else None)
        self.material_texture_flags = material_texture_flags
        self.normal_texture_map = (normal_texture_map.to(self.corners.device)
                                   if normal_texture_map is not None else None)

        if shader is None:
            shader = RENDERING_DEFAULTS.shader
        self.shader = shader

    def get_batch_identifier(self):
        return f"{self.__class__}_{id(self.shader)}"

    def get_interpolation_coordinates(
        self, vertex_corners, fragment_x, fragment_y, aa_offsets
    ):

        out = get_bary_coordinates(vertex_corners, fragment_x, fragment_y)
        return out

    def interpolate_property(self, interpolation_coord, property, repeats_inds):
        out = interpolate_triangle_corners(
            self,
            interpolation_coord,
            self.expand_verts_to_frags(property, repeats_inds.unsqueeze(-1), -3, persist=True),
        )
        return out


def get_tangents(x):
    return torch.cat((x[:, 1:] - x[:, -1:], x[:, :1] - x[:, -1:]), 1), torch.cat(
        (x[:, -1:] - x[:, :1], x[:, -1:] - x[:, 1:]), 1
    )
