from algan.animation.animation_contexts import Off, Sync
from algan.constants.color import *#WHITE
from algan.constants.spatial import *#CAMERA_ORIGIN
from algan.scene_tracker import SceneTracker
from algan.mobs.mob import Mob
from algan.utils.tensor_utils import expand_as_left, broadcast_gather, squish, unsquish, dot_product
from algan.geometry.geometry import intersect_line_with_plane, intersect_line_with_plane_colinear
import torch.nn.functional as F


class Camera(Mob):
    def __init__(self, orthographic=False, screen_distance=5, screen_scale=2.5, *args, **kwargs):
        super().__init__(add_to_scene=False, init=False, *args, **kwargs)
        self.animatable_attrs.remove('color')
        with Off():
            self.orthographic = orthographic
            self.location = CAMERA_ORIGIN
            #self.rotate(180, UP)
            self.screen = Mob(location=self.location + screen_distance * self.get_forward_direction(), add_to_scene=False, init=False)
            self.screen.scale(torch.tensor((1/screen_scale, 1/screen_scale, 1)))
            self.screen.is_primitive = True
            self.is_primitive = True
            self.add_children(self.screen)
            #self.look_at(ORIGIN)
            self.light_source_location = self.location + UP * 1 + RIGHT*5 + OUT*1
            self.light_color = WHITE.clone()
            #self.light_color[-2] = 0.0#1
            coord2_range = self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height
            self.screen_distance = 5
            #self.screen = make_grid(self.video.num_pixels_screen_height, self.video.num_pixels_screen_width,
            #                        min_coord2=-coord2_range, max_coord2=coord2_range)# + self.screen_distance * IN
            #self.pixel_inds = torch.arange(self.screen.shape[0]*self.screen.shape[1]).view(self.screen.shape[0], self.screen.shape[1])
            #self.timestep_inds = torch.arange(10000) * (self.screen.shape[0]*self.screen.shape[1])
            #self.corner_pixels = torch.stack((self.screen[0,0], self.screen[0,-1], self.screen[-1,-1], self.screen[-1,0]))
            self.pixel_height = 2 / self.scene.num_pixels_screen_height
            self.pixel_width = self.pixel_height
            #if self.orthographic:
            #    self.focal_point = self.screen - IN
            self.rays_outdated = True
            self.rays = None
            self.in_subview_mode = False
            s = self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height
            self.corner_x_coords = torch.tensor([-s, -s, s, s]).view(-1, 1, 1, 1)
            self.corner_y_coords = torch.tensor([-1, 1, 1, -1]).view(-1, 1, 1, 1)
            self.animatable_attrs.update({'light_source_location'})
            self.spawn(animate=False)

    def retroactive_center(self, mob, **kwargs):
        self.set_to_retroactive()
        self.move_to_make_mob_center_of_view(mob, **kwargs)
        self.set_to_current()

    def move_to_make_mob_center_of_view(self, mob, buffer_portion=0.7):
        f = self.get_forward_direction()
        r = self.get_right_direction()
        u = self.get_upwards_direction()
        #mob_boundary_points = [mob.get_boundary_in_direction(_) + _ * buffer for _ in [u, r, -u, -r]]
        mob_boundary_points = [mob.get_boundary_in_direction(_) for _ in [-r, u, r, -u]]
        mob_boundary_points = torch.stack(mob_boundary_points)
        #mobl = sum(mob_boundary_points) / len(mob_boundary_points)
        mobl = 0.5 * ((mob_boundary_points).amax(0) + (mob_boundary_points).amin(0))
        mob_boundary_points = (mob_boundary_points - mobl) * (1+buffer_portion) + mobl
        selfl = self.location

        with Sync():
            self.move_to(mobl - f * dot_product(mobl - selfl, f))
            selfl = self.location

            corner_rays = F.normalize(self.get_corner_pixels() - selfl, dim=-1, p=2)
            edge_plane_rays = torch.stack((corner_rays, torch.cat((corner_rays[1:], corner_rays[:1]))), 1)
            up_plane = edge_plane_rays[1]
            right_plane = edge_plane_rays[0]

            vertical_move, vertical_dist = intersect_line_with_plane_colinear(-f, selfl, up_plane[0], up_plane[1], mob_boundary_points[1])
            horizontal_move, horizontal_dist = intersect_line_with_plane_colinear(-f, selfl, right_plane[0], right_plane[1], mob_boundary_points[0])

            return self.move((mob_boundary_points[1] - vertical_move) if vertical_dist <= horizontal_dist else (mob_boundary_points[0] - horizontal_move))

    def coord_to_pixel(self, coord):
        normalized_coord = coord * 0.5 * self.scene.num_pixels_screen_height#*0.5+0.5
        normalized_coord += 0.5*torch.tensor((self.scene.num_pixels_screen_height, self.scene.num_pixels_screen_width))
        return normalized_coord.long()

    def pixel_to_coord(self, pixel):
        return pixel * 2 / self.scene.num_pixels_screen_height - torch.tensor((1, self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height))

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
        #point -= starts
        #TODO change this from IN to self.forward_direction()
        hits = (intersect_line_with_plane(point - starts, self.screen_offset(self.screen[:1, :1]).unsqueeze(-1), self.get_forward_direction().unsqueeze(-1), starts, dim=-2)[0] - starts)
        return torch.cat((dot_product(hits, self.get_upwards_direction().unsqueeze(-1), dim=-2, keepdim=True),
                          dot_product(hits, self.get_right_direction().unsqueeze(-1), dim=-2, keepdim=True)), -2)
        #return [..., 1:, :]

    def set_view_to_bounding_box(self, actor):#, sub_ts, spawn_t):
        rect_corners = actor.get_rectangle_outline_for_camera(self)
        """if False:#bounding_box.dim() == 2:
            bounding_box = bounding_box.unsqueeze(0).expand(sub_ts.shape[0], -1, -1).clone()
        if bounding_box is None:
            self.in_subview_mode = False
            return"""

        def get_extreme_coord(coords):
            coords.clamp_(min=torch.tensor((0,)), max=self.scene.frame_size)
            return coords.unsqueeze(1)

        min_coord = get_extreme_coord(self.coord_to_pixel(rect_corners.amin(-1))-1)#.unsqueeze(1) # TODO why is -1, +1 here?
        max_coord = get_extreme_coord(self.coord_to_pixel(rect_corners.amax(-1))+1)#.unsqueeze(1)
        sizes = max_coord - min_coord
        self.subframe_sizes = sizes
        ray_counts = (sizes).prod(-1)
        max_num_pixels = ray_counts.amax((0, 1))
        pixel_inds = squish(self.pixel_inds)[:max_num_pixels].unsqueeze(1).unsqueeze(1).unsqueeze(0)
        # start = spawn_t
        self.inds = ((((pixel_inds % sizes[...,1:]) + (pixel_inds // sizes[..., 1:]) * self.pixel_inds.shape[1] + (min_coord[...,1:] +
                    min_coord[...,:1] * self.pixel_inds.shape[1])).clamp_(max=self.pixel_inds.shape[0] * self.pixel_inds.shape[1] - 1)))

        #pixel_coords = self.screen_offset(squish(self.screen))#[..., 1:]
        #pixel_coords = torch.cat((dot_product(pixel_coords, self.get_upwards_direction(), dim=-1),
        #                          dot_product(pixel_coords, self.get_right_direction(), dim=-1)), -1)
        pixel_coords = squish(self.screen)[..., 1:].unsqueeze(0).expand(self.location.shape[0], -1, -1)

        box_points = broadcast_gather(pixel_coords.unsqueeze(1).unsqueeze(1), -2, self.inds.unsqueeze(-1), keepdim=False)
        rect_corners = rect_corners.unsqueeze(1)

        def in_range(p1, p2):
            l = p2-p1
            d = dot_product(box_points - p1, l, dim=-1)
            return (0 <= d) & (d <= l.norm(p=2, dim=-1, keepdim=True).square_())

        rect_points = ((in_range(rect_corners[..., 0], rect_corners[..., 1]) & in_range(rect_corners[..., 1], rect_corners[..., 2])))
        num_pixels = rect_points.sum(1)
        self.subframe_sizes = sizes
        si = self.inds#, 0, 1)
        sr = rect_points#, 0, 1)
        self.inds = unsquish(torch.nn.utils.rnn.pad_sequence([si[i, sr[i]] for i in range(sr.shape[0])], batch_first=True), -1, self.inds.shape[-2]).unsqueeze(-1)
        self.num_pixels = num_pixels
        self.num_non_idle_timesteps = self.inds.shape[0]
        ##self.inds = torch.cat((self.inds, broadcast_gather(self.inds, 0, unsqueeze_right(actor.idle_gather_inds, self.inds), keepdim=True)), 0)
        return self
        return subrays, subray_counts
        return self.get_rays(actor, sub_ts, spawn_t)

    def get_ray_iterator(self):
        target_mem = torch.cuda.get_device_properties(0).total_memory - int(1e9)
        batch_size = 1
        current_ind = 0
        prev_alloc_mem = None
        prev_batch_size = 1
        while current_ind < self.inds.shape[1]:
            torch.cuda.reset_peak_memory_stats()
            yield self.get_subrays(self.inds[:, current_ind:current_ind+batch_size], self.num_pixels)
            reserved_mem = torch.cuda.max_memory_reserved()
            alloc_mem = torch.cuda.max_memory_allocated()
            if prev_alloc_mem is None:
                prev_alloc_mem = alloc_mem
            current_ind += batch_size
            #torch.cuda.empty_cache()
            if batch_size == prev_batch_size:
                batch_size += 1
                continue
            mem_per_pixel = ((alloc_mem - prev_alloc_mem) // max(batch_size - prev_batch_size, 1))
            batch_size = max(batch_size + 1 + (target_mem - reserved_mem) // max(mem_per_pixel, 1), 1)
        return
        #prev_used_mem = torch.cuda.memory_allocated(0)  # torch.cuda.memory_reserved(0)
        #torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        yield self.get_subrays(self.inds[:1], self.num_pixels)
        if self.inds.shape[0] <= 1:
            return
        peak_1 = torch.cuda.max_memory_allocated()
        #prev_used_mem = torch.cuda.memory_allocated(0)  # torch.cuda.memory_reserved(0)
        #torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        yield self.get_subrays(self.inds[1:2], self.num_pixels)
        if self.inds.shape[0] <= 1:
            return
        peak_1 = torch.cuda.max_memory_allocated()
        # prev_used_mem = torch.cuda.memory_allocated(0)  # torch.cuda.memory_reserved(0)
        #torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        j = 2
        yield self.get_subrays(self.inds[2:j+2], self.num_pixels)
        if self.inds.shape[0] <= j+2:
            return
        peak_2 = torch.cuda.max_memory_allocated()
        mem_per_pixel = max((peak_2 - peak_1), 1) / (j-1)
        batch_size = int(max((total_mem - torch.cuda.max_memory_reserved()) / max(mem_per_pixel, 1), 1))
        #print(f'p1: {peak_1 / 1e9}, p2: {peak_2 / 1e9}, mpp: {mem_per_pixel / 1e9}, bs: {batch_size}')
        current_ind = j+2
        b = batch_size
        while current_ind < len(self.inds):
            #for i in range(j+1, len(self.inds), batch_size):
            torch.cuda.reset_peak_memory_stats()
            yield self.get_subrays(self.inds[current_ind:current_ind+b], self.num_pixels)
            peak_3 = torch.cuda.max_memory_reserved()
            current_ind = current_ind + b
            b = max(b + int((total_mem - torch.cuda.max_memory_reserved()) / mem_per_pixel), 1)
            #print(f'b: {b}')
            #print(peak_3 / 1e9)

    def get_subframe(self, frame, depths, inds, transpose=True):
        if frame is None:# or not self.in_subview_mode:
            return frame

        if transpose:
            inds = inds.transpose(0, 1)

        def gather_subset(x):
            return broadcast_gather(squish(x, 0, 1).unsqueeze(-2), 0, inds, keepdim=True)
        return (gather_subset(_) for _ in (frame, depths))

    def update_frame(self, frame, depths, subframe, subdepths, inds):
        num_frames = frame.shape[-1]
        if frame is None:
            return subframe
        if subframe is None:
            return frame
        #frame = frame.clone()

        one_frame = False#self.inds.amax() < self.screen.shape[0]*self.screen.shape[1]
        if one_frame:
            raise Exception('only one frame?')
            #frame = frame[:1]
        inds = inds.expand(-1, -1, -1, frame.shape[-1])#,-1, -1)
        #inds = (inds + torch.arange(inds.shape[-1]) * (frame.shape[0]*frame.shape[1])).view(-1).unsqueeze(1).expand(-1, frame.shape[2])#,-1)

        def update_frame(frame, subframe):
            fs = frame.shape
            #frame = squish(frame.transpose(-2,-1), 0, 2)
            #frame = frame.transpose(-2,-1).view(-1,frame.shape[-2])
            #fs2 = frame.shape
            frame = squish(frame, 0, 1)
            #subframe = squish(subframe.permute(0,2,3,1), 0, 2)
            #sf = squish(subframe, -2, -1)
            for i in range(inds.shape[-2]):
                frame = frame.scatter_(0, inds[..., i, :frame.shape[-1]], subframe[...,i, :])#torch.full_like(subframe, 255))
            #frame = frame.reshape(fs2).transpos
            return frame

        frame = update_frame(frame, subframe)
        depths = update_frame(depths, subdepths)
        if one_frame:
            frame = frame.expand(-1, -1, -1, num_frames)
        return frame, depths

        frame.scatter()
        frame[self.min_coord[0]:self.max_coord[0], self.min_coord[1]:self.max_coord[1]] = subframe.clone()
        return frame

    def project_point_onto_screen_border(self, point, direction):
        corner_rays = F.normalize((self.get_corner_pixels()) - self.location, dim=-1, p=2)
        edge_plane_rays = torch.stack((corner_rays, torch.cat((corner_rays[1:], corner_rays[:1]))))
        intersection_points, intersection_distances = (torch.stack(_) for _ in zip(*[intersect_line_with_plane_colinear(
            direction, self.location, edge_plane_rays[0, i], edge_plane_rays[1, i], point) for i in range(edge_plane_rays.shape[1])]))
        intersection_distances = intersection_distances.nan_to_num(nan=1e12, posinf=1e12, neginf=1e12)
        m = (intersection_distances.sign() == torch.cat((intersection_distances[2:], intersection_distances[:2])).sign()).float()
        diff_sign_intersection_distances = intersection_distances.clone()
        diff_sign_intersection_distances[diff_sign_intersection_distances < 0] = 1e12
        intersection_distances = intersection_distances.abs() * m + (1-m) * diff_sign_intersection_distances
        closest_ind = intersection_distances.argmin(0, keepdim=True)
        closest_point = broadcast_gather(intersection_points, 0, closest_ind, keepdim=False)
        return closest_point

    def get_corner_pixels(self):
        b = unsquish(self.screen.basis, -1, 3)
        b = b / b.norm(p=2,dim=-1,keepdim=True).square().clamp_min(1e-6)
        return self.screen.location + b[..., 0, :] * self.corner_x_coords + b[..., 1, :] * self.corner_y_coords
        self.location, camera.screen.location, camera.screen.basis
        return self.corner_pixels + self.location + self.get_forward_direction() * self.screen_distance

    def get_screen(self):
        return unsquish(squish(self.screen_offsets, 0, 1) + self.location + self.get_forward_direction() * self.screen_distance, 1, self.screen_offsets.shape[1])

    def get_subrays(self, inds, ray_counts):
        if self.rays_outdated or len(self.location) != len(self.rays):
            rays = F.normalize(unsquish(self.screen_offset(squish(self.screen, 0, 1)) - self.location, 1, self.screen.shape[1]), dim=-1, p=2)
            if len(rays) == 0:
                return None, None, None
            #rays = F.normalize(self.screen.unsqueeze(0).expand(len(self.location), -1, -1, -1), dim=-1, p=2)
            self.rays = torch.cat((rays, expand_as_left(self.location.unsqueeze(-2), rays)), -1)
            #self.rays = torch.cat(((self.screen - self.focal_point), expand_as_left(self.focal_point, self.screen)), -1)
            #self.rays = self.screen - self.focal_point
            self.np = (squish(self.rays, 1, 2)).shape[1]
            #self.rays = self.rays.permute(-1, 0, 1)
            self.rays_outdated = False
        if False:#not self.in_subview_mode:
            return self.rays
        np = self.np

        max_num_pixels = torch.tensor((inds.shape[1],))#ray_counts.amax(0)
        full_inds = inds
        inds = full_inds[:self.num_non_idle_timesteps]
        ray_counts = max_num_pixels.view(1,1).expand(inds.shape[0], inds.shape[1]).unsqueeze(-1).unsqueeze(-1)
        if max_num_pixels <= 0:
            return None, ray_counts, full_inds
        subrays = broadcast_gather(squish(self.rays, 1, 2), 1, inds.reshape(len(inds), -1, 1) % np).reshape(inds.shape[0], inds.shape[1], inds.shape[2], -1)#self.rays.shape[0], -1, inds.shape[1], inds.shape[2])
        subrays = subrays.transpose(0,1)
        #ray_counts = (self.max_coord - self.min_coord).prod(-1)
        #subrays = [torch.nn.utils.rnn.pad_sequence(torch.split(_, list(ray_counts)), padding_value=1-i) for i, _ in enumerate(torch.split(subrays, 3, -1))]
        #subrays = [unsquish(_, 0, -self.min_coord.shape[0]).transpose(0,1) for i, _ in enumerate(torch.split(subrays, 3, -1))]
        subrays = torch.split(subrays, 3, -1)

        ##self.inds = torch.cat((broadcast_gather(self.inds, 1, actor.non_idle_inds.unsqueeze(0), keepdim=True),
        ##                       broadcast_gather(self.inds, 1, actor.idle_inds.unsqueeze(0), keepdim=True)), 1)
        return subrays, ray_counts, full_inds
        return self.rays[self.min_coord[...,0]:self.max_coord[...,0], self.min_coord[...,1]:self.max_coord[...,1]]


scene = SceneTracker.instance()
if scene.camera is None:
    scene.camera = Camera(False)
