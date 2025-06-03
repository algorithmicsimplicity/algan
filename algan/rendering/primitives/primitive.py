import torch
import torchvision
import torch.nn.functional as F
from torch_scatter import scatter_max

from algan.constants.color import BLUE, BLACK
from algan.geometry.geometry import intersect_line_with_plane
from algan.utils.memory_utils import InsufficientMemoryException
from algan.utils.tensor_utils import dot_product, squish, broadcast_gather, unsquish, unsqueeze_right


class RenderPrimitive:
    def __init__(self, corners=None, colors=BLUE, opacity=0, normals=None, perimeter_points=None, reverse_perimeter=False, triangle_collection=None, glow=0):
        self.corners = corners
        self.colors = colors
        self.normals = normals

    def get_batch_identifier(self):
        return f'{self.__class__}'

    def render(self, primitives, scene, save_image, screen_width, screen_height, background_color, *args, **kwargs):
        screen_width *= kwargs['anti_alias_level']
        screen_height *= kwargs['anti_alias_level']
        window = (0, 0, screen_width, screen_height)
        kwargs['screen_width'] = screen_width
        kwargs['screen_height'] = screen_height
        frames = self.render_window(primitives, scene, window, save_image, 0, self.corners.shape[0], 0, 1, background_color, False, *args, **kwargs)

    def save_frames(self, frames, save_image, scene):
        if not save_image:
            for frame in frames:
                scene.file_writer.write(frame)
        else:
            for frame in frames:
                torchvision.utils.save_image(torch.from_numpy(frame).permute(-1, 0, 1) / 255, scene.file_path)

    def render_window(self, primitives, scene, window, save_image, time_start, time_end, object_start, object_end, background_color, return_frags=False, *args, **kwargs):
        self.memory = kwargs['memory']
        original_pointer = self.memory.current_pointer
        try:
            chunks = [p.render_(time_start, time_end, object_start, object_end, *args, **kwargs, window_coords=window) for p in primitives]
            chunks = [_ for _ in chunks if _ is not None]
            if return_frags:
                return chunks
            out = self.get_tensor_from_memory((((window[2]-window[0])*(window[3]-window[1])), 4), torch.uint8)
            if len(chunks) == 0:
                frames = (next(scene.get_frames_from_fragments(None, window, out, anti_alias_level=kwargs['anti_alias_level'])) for _ in range(time_end - time_start))
            else:
                colors, dists, inds = [torch.cat(_) for _ in zip(*chunks)]
                frags = self.blend_frags_to_pixels(colors, dists, inds, background_color, time_end-time_start, kwargs['screen_width'], kwargs['screen_height'])
                frames = scene.get_frames_from_fragments(frags, window, out, anti_alias_level=kwargs['anti_alias_level'])
            if (window[2]-window[0]) == kwargs['screen_width'] and (window[3]-window[1]) == kwargs['screen_height']:
                self.save_frames(frames, save_image, scene)
        except (InsufficientMemoryException, torch.cuda.OutOfMemoryError):
            self.memory.current_pointer = original_pointer
            torch.cuda.empty_cache()
            if (time_end - time_start) > 1:
                m = time_start + (time_end - time_start)//2
                self.render_window(primitives, scene, window, save_image, time_start, m, object_start, object_end, background_color, False, *args, **kwargs)
                self.render_window(primitives, scene, window, save_image, m, time_end, object_start, object_end, background_color, False, *args, **kwargs)
                return

            m = object_start + (object_end - object_start)/2
            chunks1 = self.render_window(primitives, scene, window, save_image, time_start, time_end, object_start, m, background_color, True, *args, **kwargs)
            chunks2 = self.render_window(primitives, scene, window, save_image, time_start, time_end, m, object_end, background_color, True, *args, **kwargs)
            chunks = chunks1 + chunks2
            if return_frags:
                return chunks

            out = self.get_tensor_from_memory((((window[2] - window[0]) * (window[3] - window[1])), 4), torch.uint8)
            if len(chunks) == 0:
                frames = (next(scene.get_frames_from_fragments(None, window, out, anti_alias_level=kwargs['anti_alias_level'])) for _ in range(time_end - time_start))
            else:
                colors, dists, inds = [torch.cat(_) for _ in zip(*chunks)]
                frags = self.blend_frags_to_pixels(colors, dists, inds, background_color, time_end-time_start, kwargs['screen_width'], kwargs['scren_height'])
                frames = scene.get_frames_from_fragments(frags, window, out, anti_alias_level=kwargs['anti_alias_level'])

            self.save_frames(frames, save_image, scene)
        finally:
            self.memory.current_pointer = original_pointer
        return frames

    def get_tensor_from_memory(self, *args, **kwargs):
        return self.memory.get_tensor(*args, **kwargs)

    def get_tensor(self, *args, **kwargs):
        return self.get_tensor_from_memory(*args, **kwargs)

    def expand_verts_to_frags(self, x, repeats_inds, dim=-2, out=None):
        if out is None:
            xshape = [_ for _ in x.shape]
            xshape[dim] = repeats_inds.shape[dim]
            out = self.get_tensor(xshape, x.dtype)
        return broadcast_gather(x, dim, repeats_inds, out=out)

    def blend_frags_to_pixels(self, colors, dists, inds, background_color, num_frames, screen_width, screen_height):
        unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        current_frags = self.get_tensor((len(unique_inds), colors.shape[-1] - 1), torch.float)
        self.memory.save_pointer()

        if unique_counts.numel() == 0:
            max_buffer_depth = 1
        else:
            max_buffer_depth = unique_counts.amax()


        out = current_frags
        out[..., :] = background_color[...,:out.shape[-1]]
        out[..., -1] = 0

        # TODO make it so that if opacity is 0, that pixel is removed entirely (instead of just painting background constants), this will save us having to
        # render invisble objects.

        def blend_colors(dists, inds, colors, out):
            for i in range(max_buffer_depth):
                max_dist, max_ind = scatter_max(dists, inds, -1, dim_size=out.shape[-2])
                mask = ((0 < max_dist) & (max_dist < 1e12)).unsqueeze(-1)

                dists.scatter_(-1, max_ind, -1.0)

                def do_write(out):
                    inds_selected = broadcast_gather(inds, -1, max_ind, keepdim=True)
                    c_write = broadcast_gather(colors, -2, max_ind.unsqueeze(-1), keepdim=True)
                    ie = inds_selected.unsqueeze(-1).expand(-1, out.shape[-1])
                    c_read = broadcast_gather(out, -2, ie, keepdim=True)
                    a = c_write[..., -1:]
                    write = c_read * (1 - a) + a * (c_write[..., :-1])
                    write = write * mask + (~mask) * c_read
                    out.scatter_(-2, ie, write)
                    return out

                out = do_write(out)
            return out

        def blend_colors_layerwise(dists, inds, colors, out):
            while True:
                max_dist, max_ind = scatter_max(dists, inds, -1, dim_size=out.shape[-2])

                def apply_mask(max_ind):
                    remaining_inds = torch.arange(inds.shape[-1], device=inds.device)
                    dump_mask = (max_ind < inds.shape[-1]) & (max_dist < 1e12) & (0 < max_dist)
                    max_ind = max_ind[dump_mask]
                    return remaining_inds, max_ind

                remaining_inds, max_ind = apply_mask(max_ind)

                def do_write(out):
                    inds_selected = broadcast_gather(inds, -1, max_ind, keepdim=True)
                    c_write = broadcast_gather(colors, -2, max_ind.unsqueeze(-1), keepdim=True)
                    ie = inds_selected.unsqueeze(-1).expand(-1, 3)
                    c_read = broadcast_gather(out, -2, ie, keepdim=True)
                    a = c_write[..., -1:]
                    out.scatter_(-2, ie, c_read * (1 - a) + a * c_write[..., :-1])
                    return out

                out = do_write(out)

                def get_rem_inds(remaining_inds):
                    mu = max_ind.clamp_max_(inds.shape[-1] - 1).unique()
                    combined = torch.cat((mu, remaining_inds), -1)
                    uniques, counts = combined.unique(return_counts=True)
                    remaining_inds = uniques[counts == 1]
                    return remaining_inds

                remaining_inds = get_rem_inds(remaining_inds)

                if remaining_inds.shape[-1] <= 1:
                    break
                dists = torch.gather(dists, -1, remaining_inds)
                inds = torch.gather(inds, -1, remaining_inds)
                colors = torch.gather(colors, -2, remaining_inds.unsqueeze(-1).expand(-1, colors.shape[-1]))
            return out

        out = blend_colors(dists, unique_inds_inverse, colors, out)
        self.memory.reset_pointer()

        out_inds = unique_inds.scatter_(0, unique_inds_inverse, inds)
        ind_counts = torch.histc(out_inds.float(), num_frames, min=0, max=(screen_width * screen_height * num_frames)).long()
        return out, out_inds, ind_counts

    def render_(self, time_start, time_end, object_start, object_end, ray_origin, screen_point, screen_basis,
               background_color=BLACK, anti_alias=False, anti_alias_offset=[0.5, 0.5], anti_alias_level=1,
               light_origin=None, light_color=None, screen_width=2000, screen_height=2000, window_coords=None, memory=None,
                primitive_type=None):
        ray_origin = ray_origin.unsqueeze(-2)
        screen_point = screen_point.unsqueeze(-2)
        screen_basis = unsquish(screen_basis, -1, 3)

        def select_time(x):
            x = x if len(x) == 1 else x[time_start:time_end]
            x = x if x.shape[1] == 1 else x[:, int(x.shape[1]*object_start):int(x.shape[1]*object_end)]
            return x

        corners = select_time(self.corners)
        normals = select_time(self.normals)
        colors = select_time(self.colors)
        screen_point = select_time(screen_point)
        screen_basis = select_time(screen_basis)
        ray_origin = select_time(ray_origin)

        if window_coords is None:
            window_coords = 0, 0, screen_width, screen_height

        window_width = window_coords[-2] - window_coords[0]
        start_x, start_y, end_x, end_y = window_coords

        def project_onto_screen(x):
            rays = F.normalize(x - ray_origin, p=2, dim=-1)
            projected_corners, _ = intersect_line_with_plane(rays, screen_point, screen_basis[..., -1:, :], ray_origin)
            projected_corners.nan_to_num_()
            projected_distances = (x - ray_origin).norm(p=2, dim=-1, keepdim=True)
            projected_corners -= screen_point
            corners_2d = dot_product(projected_corners.unsqueeze(-2), screen_basis[...,:-1, :].unsqueeze(-3), -1, keepdim=False)
            corners_2d.nan_to_num_()
            corners_2d = corners_2d * (screen_height//2)
            corners_2d[..., 0] += screen_width // 2
            corners_2d[..., 1] += screen_height // 2
            corners_locs = corners_2d
            corners_inds = corners_locs.long()
            return corners_locs, corners_inds, projected_distances



        corners_locs, corners_inds, projected_distances = project_onto_screen(corners)
        bounding_corners = torch.stack(((corners_inds.amin(-2)-1), (corners_inds.amax(-2)+1)), -2).clamp_(min=torch.tensor((start_x, start_y), device=corners_locs.device), max=torch.tensor((end_x, end_y), device=corners_locs.device))

        bounding_box_sizes = (bounding_corners[..., 1, :] - bounding_corners[..., 0, :])

        bbss = bounding_box_sizes.prod(-1, keepdim=True)
        bounding_box_num_pixels = bbss.amax(0)
        num_fragments = bounding_box_num_pixels.sum() * bbss.shape[0]
        mem_per_fragment = 256
        total_mem_required = num_fragments * mem_per_fragment

        free_mem = memory.get_num_bytes_remaining()

        if free_mem < total_mem_required:
            raise InsufficientMemoryException


        repeats = bounding_box_num_pixels.view(-1)
        num_frags = repeats.sum()
        repeats_inds = torch.repeat_interleave(torch.arange(len(repeats), device=repeats.device), repeats, -1, output_size=num_frags).unsqueeze(-1)

        offsets = self.expand_verts_to_frags(bounding_box_num_pixels.cumsum(-2) - bounding_box_num_pixels, repeats_inds, -2)
        fragment_inds = torch.arange(offsets.shape[-2], device=offsets.device).view(-1,1) - offsets
        bounding_box_widths = self.expand_verts_to_frags(bounding_box_sizes[...,:1], repeats_inds, -2).clamp_min_(1)

        bounding_corners_rep = self.expand_verts_to_frags(bounding_corners[...,0,:], repeats_inds, -2)
        fragment_x = self.get_tensor(bounding_box_widths.shape, torch.long)
        fragment_x[:] = (fragment_inds % bounding_box_widths) + bounding_corners_rep[...,:1]
        fragment_y = self.get_tensor(bounding_box_widths.shape, torch.long)
        fragment_y[:] = (fragment_inds // bounding_box_widths) + bounding_corners_rep[...,1:]

        aa_offsets = torch.linspace(0, 1, anti_alias_level * 2 + 1, device=fragment_x.device)[1:-1:2]
        aa_offsets = squish(torch.stack((aa_offsets.view(-1, 1).expand(-1, len(aa_offsets)), aa_offsets.view(1, -1).expand(len(aa_offsets), -1)), -1))
        all_ws = self.get_interpolation_coordinates(corners_locs, fragment_x, fragment_y, aa_offsets, repeats_inds)

        all_mask = (all_ws.amin(-2) >= self.min_interpolation_coord).any(0)

        inds = fragment_x + (fragment_y) * window_width
        screen_size = screen_width * screen_height

        m = (inds < (screen_size)) & all_mask
        m = m.reshape(-1)
        inds = inds + unsqueeze_right(torch.arange(inds.shape[0], device=inds.device) * screen_size, inds)
        inds = inds.view(-1)
        inds = inds[m]
        unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)
        if light_origin is not None:
            #cent = self.corners.mean(-2)
            #normals = F.normalize(broadcast_cross_product(self.corners[...,0,:] - cent, self.corners[...,1,:] - cent).unsqueeze(-2), p=2, dim=-1)
            #normals = self.normals
            #views = F.normalize(ray_origin - self.corners, p=2, dim=-1)
            incidences = F.normalize(corners - light_origin, p=2, dim=-1)
            #reflects = F.normalize(incidences - 2 * normals * (dot_product(normals, incidences)), p=2, dim=-1)
            #diffuse_factor = dot_product(views, reflects).relu_().pow_(0.5)
            #diffuse_factor = (dot_product(-incidences, normals) * ((dot_product(views, normals) < 0).float()*2-1)).abs().relu_().pow_(10)
            diffuse_factor = (dot_product(-incidences, normals)).relu_().pow_(5)
            diffuse_factor = diffuse_factor * 0.5
            self_colors = colors.clone()
            self_colors[...,:-1] = self_colors[...,:-1] * (1-diffuse_factor) + diffuse_factor * light_color[:-1]
        else:
            self_colors = colors

        output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        output_frags[:] = 0
        current_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))

        if unique_counts.numel() == 0:
            max_buffer_depth = 1
        else:
            max_buffer_depth = unique_counts.amax()

        def get_frags(ws):

            def interpolate(x):
                return self.interpolate_property(ws, x, repeats_inds)

            def get_colors():
                colors = interpolate(self_colors)
                colors[..., -1:] *= (ws.amin(-2) >= self.min_interpolation_coord)
                colors = colors.reshape(-1, colors.shape[-1])
                colors = colors[m]
                return colors

            def get_dists():
                dists = interpolate(projected_distances)
                dists = dists.reshape(-1)
                dists = dists[m]
                return dists

            colors, dists = get_colors(), get_dists()
            return colors, dists

        colors, dists = get_frags(all_ws[0])
        return colors, dists, inds


class RenderPrimitive2D(RenderPrimitive):
    def raycast_onto_plane(self, ray_origins, ray_directions, plane_point, plane_basis):
        dists = -dot_product(ray_origins - plane_point, plane_basis) / dot_product(ray_directions, plane_basis)
        dists.nan_to_num_()
        return dists