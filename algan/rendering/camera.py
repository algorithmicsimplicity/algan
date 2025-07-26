from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.animation.animation_contexts import Off, Sync
from algan.constants.spatial import *  # CAMERA_ORIGIN
from algan.mobs.mob import Mob
from algan.utils.tensor_utils import (
    expand_as_left,
    broadcast_gather,
    squish,
    unsquish,
    dot_product,
)
from algan.geometry.geometry import (
    intersect_line_with_plane,
    intersect_line_with_plane_colinear,
)
import torch.nn.functional as F


class Camera(Mob):
    def __init__(
        self, orthographic=False, screen_distance=5, screen_scale=2.5, *args, **kwargs
    ):
        super().__init__(add_to_scene=False, init=False, *args, **kwargs)
        self.animatable_attrs.remove("color")
        with Off():
            self.orthographic = orthographic
            # self.rotate(180, UP)
            self.screen = Mob(
                location=self.location + screen_distance * self.get_forward_direction(),
                add_to_scene=False,
                init=False,
            )
            self.screen.scale(torch.tensor((1 / screen_scale, 1 / screen_scale, 1)))
            self.screen.is_primitive = True
            self.is_primitive = True
            self.add_children(self.screen)
            # self.look_at(ORIGIN)
            # self.light_color[-2] = 0.0#1
            coord2_range = (
                self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height
            )
            self.screen_distance = 5
            # self.screen = make_grid(self.video.num_pixels_screen_height, self.video.num_pixels_screen_width,
            #                        min_coord2=-coord2_range, max_coord2=coord2_range)# + self.screen_distance * IN
            # self.pixel_inds = torch.arange(self.screen.shape[0]*self.screen.shape[1]).view(self.screen.shape[0], self.screen.shape[1])
            # self.timestep_inds = torch.arange(10000) * (self.screen.shape[0]*self.screen.shape[1])
            # self.corner_pixels = torch.stack((self.screen[0,0], self.screen[0,-1], self.screen[-1,-1], self.screen[-1,0]))
            self.pixel_height = 2 / self.scene.num_pixels_screen_height
            self.pixel_width = self.pixel_height
            # if self.orthographic:
            #    self.focal_point = self.screen - IN
            self.rays_outdated = True
            self.rays = None
            self.in_subview_mode = False
            s = 1  # self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height
            self.corner_x_coords = torch.tensor([-s, -s, s, s]).view(-1, 1, 1, 1)
            self.corner_y_coords = torch.tensor([-1, 1, 1, -1]).view(-1, 1, 1, 1)
            # self.animatable_attrs.update({'light_source_location'})
            self.spawn(animate=False)

    def set_euler_angles(self, angle_1, angle_2, angle_3):
        with Sync():
            self.orbit_around_line(ORIGIN, RIGHT, num_degrees=angle_1)
            self.orbit_around_line(ORIGIN, UP, num_degrees=angle_2)
            self.orbit_around_line(ORIGIN, OUT, num_degrees=angle_3)

    def set_state_to_time_t(self, time_inds):
        super().set_state_to_time_t(time_inds)
        device = COMPUTING_DEFAULTS.render_device
        self.ray_origin = self.location.unsqueeze(-2).to(device)
        self.screen_point = self.screen.location.unsqueeze(-2).to(device)
        self.screen_basis = unsquish(self.screen.basis, -1, 3).to(device)

    def retroactive_center(self, mob, **kwargs):
        self.set_to_retroactive()
        self.move_to_make_mob_center_of_view(mob, **kwargs)
        self.set_to_current()

    def move_to_make_mob_center_of_view(self, mob, buffer_portion=0.7):
        f = self.get_forward_direction()
        r = self.get_right_direction()
        u = self.get_upwards_direction()
        # mob_boundary_points = [mob.get_boundary_in_direction(_) + _ * buffer for _ in [u, r, -u, -r]]
        mob_boundary_points = [mob.get_boundary_in_direction(_) for _ in [-r, u, r, -u]]
        mob_boundary_points = torch.stack(mob_boundary_points)
        # mobl = sum(mob_boundary_points) / len(mob_boundary_points)
        mobl = 0.5 * ((mob_boundary_points).amax(0) + (mob_boundary_points).amin(0))
        mob_boundary_points = (mob_boundary_points - mobl) * (1 + buffer_portion) + mobl
        selfl = self.location

        with Sync():
            self.move_to(mobl - f * dot_product(mobl - selfl, f))
            selfl = self.location

            corner_rays = F.normalize(self.get_corner_pixels() - selfl, dim=-1, p=2)
            edge_plane_rays = torch.stack(
                (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1]))), 1
            )
            up_plane = edge_plane_rays[1]
            right_plane = edge_plane_rays[0]

            vertical_move, vertical_dist = intersect_line_with_plane_colinear(
                -f, selfl, up_plane[0], up_plane[1], mob_boundary_points[1]
            )
            horizontal_move, horizontal_dist = intersect_line_with_plane_colinear(
                -f, selfl, right_plane[0], right_plane[1], mob_boundary_points[0]
            )

            return self.move(
                (mob_boundary_points[1] - vertical_move)
                if vertical_dist <= horizontal_dist
                else (mob_boundary_points[0] - horizontal_move)
            )

    def coord_to_pixel(self, coord):
        normalized_coord = coord * 0.5 * self.scene.num_pixels_screen_height  # *0.5+0.5
        normalized_coord += 0.5 * torch.tensor(
            (self.scene.num_pixels_screen_height, self.scene.num_pixels_screen_width)
        )
        return normalized_coord.long()

    def pixel_to_coord(self, pixel):
        return pixel * 2 / self.scene.num_pixels_screen_height - torch.tensor(
            (
                1,
                self.scene.num_pixels_screen_width
                / self.scene.num_pixels_screen_height,
            )
        )

    def screen_offset(self, p):
        return p + self.location + self.get_forward_direction() * self.screen_distance

    def project_onto_screen(self, point):
        # point.shape: [3,*]
        starts = self.location.unsqueeze(-1)
        """if self.orthographic:
            focal_parallel1 = F.normalize(self.focal_point[1, 0] - self.focal_point[0, 0], p=2, dim=-1)
            focal_parallel2 = F.normalize(self.focal_point[0, 1] - self.focal_point[0, 0], p=2, dim=-1)
            b1 = (bounding_box * focal_parallel1).sum(-1, keepdim=True) * focal_parallel1
            b2 = (bounding_box * focal_parallel2).sum(-1, keepdim=True) * focal_parallel2
            starts = b1 + b2"""
        # point -= starts
        # TODO change this from IN to self.forward_direction()
        hits = (
            intersect_line_with_plane(
                point - starts,
                self.screen_offset(self.screen[:1, :1]).unsqueeze(-1),
                self.get_forward_direction().unsqueeze(-1),
                starts,
                dim=-2,
            )[0]
            - starts
        )
        return torch.cat(
            (
                dot_product(
                    hits,
                    self.get_upwards_direction().unsqueeze(-1),
                    dim=-2,
                    keepdim=True,
                ),
                dot_product(
                    hits, self.get_right_direction().unsqueeze(-1), dim=-2, keepdim=True
                ),
            ),
            -2,
        )
        # return [..., 1:, :]

    def set_view_to_bounding_box(self, actor):  # , sub_ts, spawn_t):
        rect_corners = actor.get_rectangle_outline_for_camera(self)
        """if False:#bounding_box.dim() == 2:
            bounding_box = bounding_box.unsqueeze(0).expand(sub_ts.shape[0], -1, -1).clone()
        if bounding_box is None:
            self.in_subview_mode = False
            return"""

        def get_extreme_coord(coords):
            coords.clamp_(min=torch.tensor((0,)), max=self.scene.frame_size)
            return coords.unsqueeze(1)

        min_coord = get_extreme_coord(
            self.coord_to_pixel(rect_corners.amin(-1)) - 1
        )  # .unsqueeze(1) # TODO why is -1, +1 here?
        max_coord = get_extreme_coord(
            self.coord_to_pixel(rect_corners.amax(-1)) + 1
        )  # .unsqueeze(1)
        sizes = max_coord - min_coord
        self.subframe_sizes = sizes
        ray_counts = (sizes).prod(-1)
        max_num_pixels = ray_counts.amax((0, 1))
        pixel_inds = (
            squish(self.pixel_inds)[:max_num_pixels]
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        # start = spawn_t
        self.inds = (
            (pixel_inds % sizes[..., 1:])
            + (pixel_inds // sizes[..., 1:]) * self.pixel_inds.shape[1]
            + (min_coord[..., 1:] + min_coord[..., :1] * self.pixel_inds.shape[1])
        ).clamp_(max=self.pixel_inds.shape[0] * self.pixel_inds.shape[1] - 1)

        # pixel_coords = self.screen_offset(squish(self.screen))#[..., 1:]
        # pixel_coords = torch.cat((dot_product(pixel_coords, self.get_upwards_direction(), dim=-1),
        #                          dot_product(pixel_coords, self.get_right_direction(), dim=-1)), -1)
        pixel_coords = (
            squish(self.screen)[..., 1:]
            .unsqueeze(0)
            .expand(self.location.shape[0], -1, -1)
        )

        box_points = broadcast_gather(
            pixel_coords.unsqueeze(1).unsqueeze(1),
            -2,
            self.inds.unsqueeze(-1),
            keepdim=False,
        )
        rect_corners = rect_corners.unsqueeze(1)

        def in_range(p1, p2):
            l = p2 - p1
            d = dot_product(box_points - p1, l, dim=-1)
            return (0 <= d) & (d <= l.norm(p=2, dim=-1, keepdim=True).square_())

        rect_points = in_range(rect_corners[..., 0], rect_corners[..., 1]) & in_range(
            rect_corners[..., 1], rect_corners[..., 2]
        )
        num_pixels = rect_points.sum(1)
        self.subframe_sizes = sizes
        si = self.inds  # , 0, 1)
        sr = rect_points  # , 0, 1)
        self.inds = unsquish(
            torch.nn.utils.rnn.pad_sequence(
                [si[i, sr[i]] for i in range(sr.shape[0])], batch_first=True
            ),
            -1,
            self.inds.shape[-2],
        ).unsqueeze(-1)
        self.num_pixels = num_pixels
        self.num_non_idle_timesteps = self.inds.shape[0]
        ##self.inds = torch.cat((self.inds, broadcast_gather(self.inds, 0, unsqueeze_right(actor.idle_gather_inds, self.inds), keepdim=True)), 0)
        return self

    def project_point_onto_screen_border(self, point, direction):
        corner_rays = F.normalize(
            (self.get_corner_pixels()) - self.location, dim=-1, p=2
        )
        edge_plane_rays = torch.stack(
            (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1])))
        )
        intersection_points, intersection_distances = (
            torch.stack(_)
            for _ in zip(
                *[
                    intersect_line_with_plane_colinear(
                        direction,
                        self.location,
                        edge_plane_rays[0, i],
                        edge_plane_rays[1, i],
                        point,
                    )
                    for i in range(edge_plane_rays.shape[1])
                ]
            )
        )
        intersection_distances = intersection_distances.nan_to_num(
            nan=1e12, posinf=1e12, neginf=1e12
        )
        m = (
            intersection_distances.sign()
            == torch.cat(
                (intersection_distances[2:], intersection_distances[:2])
            ).sign()
        ).float()
        diff_sign_intersection_distances = intersection_distances.clone()
        diff_sign_intersection_distances[diff_sign_intersection_distances < 0] = 1e12
        intersection_distances = (
            intersection_distances.abs() * m
            + (1 - m) * diff_sign_intersection_distances
        )
        closest_ind = intersection_distances.argmin(0, keepdim=True)
        closest_point = broadcast_gather(
            intersection_points, 0, closest_ind, keepdim=False
        )
        return closest_point

    def get_corner_pixels(self):
        b = unsquish(self.screen.basis, -1, 3)
        b = b / b.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        aspect_ratio = (
            self.scene.render_settings.resolution[0]
            / self.scene.render_settings.resolution[1]
        )
        return (
            self.screen.location
            + b[..., 0, :] * self.corner_x_coords * aspect_ratio
            + b[..., 1, :] * self.corner_y_coords
        )
        self.location, camera.screen.location, camera.screen.basis
        return (
            self.corner_pixels
            + self.location
            + self.get_forward_direction() * self.screen_distance
        )

    def get_screen(self):
        return unsquish(
            squish(self.screen_offsets, 0, 1)
            + self.location
            + self.get_forward_direction() * self.screen_distance,
            1,
            self.screen_offsets.shape[1],
        )
