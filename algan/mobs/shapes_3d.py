from algan import *
from algan.mobs.surfaces.surface import Surface


class Sphere(Surface):
    def __init__(self, radius=1, **kwargs):
        l = 0 if 'location' not in kwargs else kwargs['location']
        def coords(u):
            u = u[:, :int(len(u) / (PI))]
            u[..., 1:] /= u[..., 1:].amax()
            x = u[..., :1]
            y = u[..., 1:]
            r = (y - 0.5).abs() * 2
            r = (1 - r.square()).sqrt()
            return torch.cat(((x*math.pi*2).sin() * r, (y-0.5)*2, (x*math.pi*2).cos() * r), -1) * radius / 2 + l

        def normals(x):
            x = x - l
            return map_global_to_local_coords(ORIGIN, squish(torch.eye(3)), F.normalize(x, p=2, dim=-1) + x)

        super().__init__(coords, normals, **kwargs)


class Cylinder(Surface):
    def __init__(self, radius=1, **kwargs):
        def coords(u):
            u = u[:, :int(len(u) / (math.pi))]
            u[..., 1:] /= u[..., 1:].amax()
            x = u[..., :1]
            y = u[..., 1:]
            r = (y - 0.5).abs() * 2
            r = (1 - r.square()).sqrt()
            return torch.cat(((x*math.pi*2).sin(), (y-0.5)*2, (x*math.pi*2).cos()), -1) * radius / 2

        def normals(x):
            return map_global_to_local_coords(ORIGIN, squish(torch.eye(3)), F.normalize(x, p=2, dim=-1) + x)

        super().__init__(coords, normals, **kwargs)

    @animated_function(animated_args={"interpolation": 0})
    def set_start_point(self, point, interpolation=1):
        offset = self.get_upwards_direction() * self.scale_coefficient[...,1].unsqueeze(-1) * 0.5
        current_end = self.location + offset
        current_start = self.location - offset
        point = current_start * (1-interpolation) + interpolation * point
        self._move_between_points(point, current_end)
        return self

    @animated_function(animated_args={"interpolation": 0})
    def move_between_points(self, start, end, interpolation=1):
        offset = self.get_upwards_direction() * self.scale_coefficient[..., 1].unsqueeze(-1) * 0.5
        current_end = self.location + offset
        current_start = self.location - offset
        start = current_start * (1 - interpolation) + interpolation * start
        end = current_end * (1 - interpolation) + interpolation * end
        self._move_between_points(start, end)
        return self

    def _move_between_points(self, start, end):
        with Sync():
            s = torch.ones_like(self.scale_coefficient)
            s[...,1] = ((end-start).norm(p=2,dim=-1) / self.scale_coefficient[...,1])
            self.move_to((start + end) * 0.5)
            self.look(F.normalize(end-start, p=2, dim=-1), axis=1)
            self.scale(s)
        return self
