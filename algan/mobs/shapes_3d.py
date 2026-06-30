import torch

from algan import *
from algan import PI
from algan.geometry.geometry import project_onto_basis, get_orthonormal_vector
from algan.mobs.surfaces.surface import Surface
from algan.mobs.shapes_2d import Circle


class Sphere(Surface):
    """A 3-D sphere.

    Parameters
    ----------
    radius
        Sphere radius.
    *args, **kwargs
        Passed to :class:`~.Surface`

    """

    def __init__(self, radius=1, *args, **kwargs):
        self.radius = radius
        super().__init__(*args, **kwargs)

    def coord_function(self, coords_2d):
        x = coords_2d[..., 0]
        y = coords_2d[..., 1]

        longitude = -torch.pi * (1 - x) + x * torch.pi
        latitude = -torch.pi * 0.5 * (1 - y) + y * torch.pi * 0.5

        X = torch.cos(latitude) * torch.cos(longitude)
        Y = torch.sin(latitude)
        Z = torch.cos(latitude) * torch.sin(longitude)

        coords_3d = torch.stack([X, Y, Z], dim=-1)
        return coords_3d * self.radius

    def normal_function(self, uv):
        return self.coord_function(uv)


class Cone(Surface):
    """A 3-D cone.

    Parameters
    ----------
    radius
        base radius.
    height
        cone height.
    *args, **kwargs
        Passed to :class:`~.Surface`

    """

    def __init__(self, radius=1, height=1, closed=False, *args, **kwargs):
        self.radius = radius
        self.height = height
        if "grid_aspect_ratio" not in kwargs:
            kwargs["grid_aspect_ratio"] = 1 / PI
        super().__init__(*args, **kwargs)
        if closed:
            self.cap = Circle(radius=radius).move(DOWN * height * 0.5)
            self.add_children(self.cap)

    def coord_function(self, uv):
        uv[..., 1:] /= uv[..., 1:].amax()
        u = -uv[..., :1]
        v = uv[..., 1:]
        return torch.cat(
            (
                (u * torch.pi * 2).sin() * self.radius * v,
                (v - 0.5) * self.height,
                (u * torch.pi * 2).cos() * self.radius * v,
            ),
            -1,
        )

    def normal_function(self, uv):
        xyz = self.coord_function(uv)
        xyz[..., 1] = 0
        return xyz


class Arrow3D(Mob):
    def __init__(
        self,
        start_point,
        end_point,
        thickness: float = 0.02,
        height: float = 0.3,
        base_radius: float = 0.08,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.tail = Cylinder(
            radius=thickness, closed=True, *args, **kwargs
        ).move_between_points(start_point, end_point)
        self.head = (
            Cone(base_radius, height, closed=True, *args, **kwargs)
            .move_to(self.tail)
            .move((end_point - start_point) * 0.5)
        )


class Cylinder(Surface):
    """A 3-D cylinder.

    Parameters
    ----------
    radius
        Cylinder radius.
    height
        Cylinder height.
    *args, **kwargs
        Passed to :class:`~.Surface`

    """

    def __init__(self, radius=1, height=1, closed=False, *args, **kwargs):
        self.radius = radius
        self.height = height
        if "grid_aspect_ratio" not in kwargs and "grid_height" not in kwargs:
            kwargs["grid_aspect_ratio"] = 1 / PI
        super().__init__(*args, **kwargs)
        if closed:
            self.bottom_cap = Circle(radius=radius).move(DOWN * height * 0.5)
            self.top_cap = Circle(radius=radius).move(UP * height * 0.5)
            self.add_children(self.bottom_cap, self.top_cap)

    def coord_function(self, uv):
        uv[..., 1:] /= uv[..., 1:].amax()
        u = -uv[..., :1]
        v = uv[..., 1:]
        return ((u * torch.pi * 2).sin() * self.radius * self.get_right_basis() +
                (v - 0.5) * self.height * self.get_upwards_basis() +
                (u * torch.pi * 2).cos() * -self.get_forward_basis()
            )

    def normal_function(self, uv):
        xyz = self.coord_function(uv)
        return project_onto_basis(xyz, [self.get_right_direction(), self.get_forward_direction()])
        xyz[..., 1] = 0
        return xyz

    @animated_function(animated_args={"interpolation": 0})
    def set_start_point(self, point, interpolation=1):
        offset = (
            self.get_upwards_basis()
            * 0.5
        )
        current_end = self.location + offset
        current_start = self.location - offset
        point = current_start * (1 - interpolation) + interpolation * point
        self._move_between_points(point, current_end)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def move_between_points(self, start, end, interpolation=1):
        offset = (
            self.get_upwards_direction()
            * self.scale_coefficient[..., 1].unsqueeze(-1)
            * 0.5
        )
        current_end = self.location + offset
        current_start = self.location - offset
        start = current_start * (1 - interpolation) + interpolation * start
        end = current_end * (1 - interpolation) + interpolation * end
        self._move_between_points(start, end)
        return self

    def _move_between_points(self, start, end):
        with Sync():
            s = torch.ones_like(self.scale_coefficient)
            s[..., 1] = (end - start).norm(p=2, dim=-1) / self.scale_coefficient[..., 1]
            self.move_to((start + end) * 0.5)
            up_b = F.normalize(end - start, p=2, dim=-1)
            right_b = get_orthonormal_vector(up_b)
            forward_b = get_orthonormal_vector(up_b, right_b)
            self.setattr_and_record_modification('basis', torch.cat((right_b * self.scale_coefficient[...,:1],
                                    end - start,
                                    forward_b * self.scale_coefficient[..., 2:],
                                    ), -1))
            self.set_location_by_function(self.coord_function)
            #self.look_and_scale(F.normalize(end - start, p=2, dim=-1), s, axis=1)
            #self.set_normal_by_function(self.normal_function)
        return self
