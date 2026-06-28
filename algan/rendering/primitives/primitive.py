import torch
import torchvision
from torch.export.dynamic_shapes import Dim
import torch.nn.functional as F
import sys
import traceback
import gc

from algan.rendering.primitives.fragment_blender_taichi import blend_packed_power_of_two_fragment_list
from algan.rendering.primitives.triangle_rasterize_taichi import rasterize_triangle_taichi
from algan import compiled, exported, not_compiled
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.constants.color import BLUE, BLACK, WHITE, RED, GREEN
from algan.geometry.geometry import intersect_line_with_plane, distance, normalize
from algan.rendering.post_processing.anti_aliasing.fxaa import fxaa
from algan.utils.memory_utils import InsufficientMemoryException, empty_cache
from algan.utils.tensor_utils import (
    dot_product,
    squish,
    broadcast_gather,
    unsquish,
    unsqueeze_right,
)


try:
    from torch_scatter import scatter_max as scatter_max_op
except ModuleNotFoundError:
    scatter_max_op = None


class OutOfRenderMemory(Exception):
    pass


#@torch.compiler.disable(recursive=True)
def scatter_arg_max(x, inds, dim=-1, dim_size=None):
    if len(inds) == 0:
        return None, None
    if scatter_max_op is not None and COMPUTING_DEFAULTS.use_torch_scatter:
        return scatter_max_op(x, inds, -1, dim_size=dim_size)
    inds = inds.clone()
    x = x.view(-1)
    out_dims = [*x.shape]
    out_dims[dim] = dim_size if dim_size is not None else inds.amax() + 1
    out = torch.zeros(out_dims, device=x.device)
    max_vals = torch.scatter_reduce(out, dim, inds, x, "amax", include_self=False)
    max_vals_gathered = broadcast_gather(max_vals, dim, inds)
    m = x >= max_vals_gathered - 1e-6
    inds[~m] = -1
    #inds = torch.where(m, inds, -1)

    sorted_inds, sorted_indices = torch.sort(inds)
    is_new_mask = torch.cat(
        [torch.tensor([True], device=x.device), torch.diff(sorted_inds) != 0]
    )

    argmax_inds = sorted_indices[is_new_mask]
    if sorted_inds[0] == -1:
        if len(argmax_inds) == 1:
            argmax_inds = argmax_inds[:0]
        elif len(argmax_inds) > 1:
            argmax_inds = argmax_inds[1:]  # sorted_indices[is_new_mask]

    max_vals = broadcast_gather(x, -1, argmax_inds)
    return max_vals, argmax_inds


dummy_dists = torch.randn((100, 1))
num_frags_in_layer = 20
example_inputs = (dummy_dists, torch.randint_like(dummy_dists, 0, 1, dtype=torch.long), torch.randn((100,5)),
                          torch.randn((200,5)), torch.randn(num_frags_in_layer, 1), torch.randint(0,1,(num_frags_in_layer, 1)),
                          torch.randint(0,1,(num_frags_in_layer, 1)), torch.randn(num_frags_in_layer, 1), torch.randn(num_frags_in_layer, 1))
dynamic_shapes = [
    (Dim.AUTO, Dim.STATIC) for _ in range(len(example_inputs))
]


def do_write(out, c_write, max_dist, dists, max_ind, c_read, ie, memory):
    with memory.temp():
        mask1 = torch.lt(torch.tensor((0.0,), device=c_write.device), max_dist, out=memory.get_tensor(max_dist.shape, torch.bool))
        mask2 = torch.lt(max_dist, 1e12, out=memory.get_tensor(max_dist.shape, torch.bool))
        mask = torch.logical_and(mask1, mask2, out=mask1).unsqueeze(-1)
        #mask = ((0 < max_dist) & (max_dist < 1e12)).unsqueeze(-1)

    dists.scatter_(-1, max_ind, -1.0)
    a = c_write[..., -1:]
    """if False:#transparent_output:
        af = a
        ab = c_read[..., -1:]
        a_out = af + ab * (1 - af)

        # If the resulting alpha is 0, the color is black with 0 alpha (fully transparent)
        # m = a_out > 1e-3

        rgb_f = c_write[..., :-1]
        rgb_b = c_read[..., :-1]
        # Calculate the resulting RGB components
        rgb_out = (
                          rgb_f * af + (1 - af) * ab * rgb_b
                  ) / a_out.clamp_min(1e-3)
        # rgg_out = torch.where(m, rgb_out, torch.zeros((1,), device=m.device), out=c_write[...,:-1])
        c_write[..., -1:].copy_(a_out)
    else:"""
    c_write = c_write[..., :-1]
    torch.lerp(c_read, c_write, a, out=c_write)
    # if True:#not (i == 0 and transparent_output):
    # write = c_read * (1 - a) + a * (c_write)
    #torch.mul(c_write, a, out=c_write)
    #torch.mul(a, -1, out=a)
    #torch.add(a, 1, out=a)
    #torch.addcmul(c_write, c_read, a, out=c_write)

    # write = write * mask + (~mask) * c_read
    write = torch.where(mask, c_write, c_read, out=c_write)

    out.scatter_(-2, ie, write)
    return out


#@exported(example_inputs=example_inputs, dynamic_shapes=dynamic_shapes)
def blend_one_layer_of_fragments(dists, inds, colors, out, max_dist, max_ind, inds_selected_, c_write_, c_read_, memory):
    inds_selected = broadcast_gather(
        inds,
        -1,
        max_ind,
        out=inds_selected_[: len(max_ind)],
        keepdim=True,
    )
    c_write = broadcast_gather(
        colors,
        -2,
        max_ind.unsqueeze(-1),
        out=c_write_[: len(max_ind)],
        keepdim=True,
    )
    ie = inds_selected.unsqueeze(-1).expand([-1, out.shape[-1]])
    c_read = broadcast_gather(
        out, -2, ie, out=c_read_[: len(max_ind)], keepdim=True
    )

    out = do_write(out, c_write, max_dist, dists, max_ind, c_read, ie, memory)
    return out

"""
normal:
c[0] * a[0],
c[0] * a[0] * (1-a[1]) + a[1] * c[1]

pre_mult:
c[0] * a[0],
c[0] * a[0] * (1-a[1]) + (c[1]*a[1]*(1-a[0])
"""

def blend_seq(data):
    data = data.cpu()
    outs = torch.zeros_like(data)
    blend = torch.zeros_like(data[0,:-1])
    for i, frag in enumerate(data):
        if frag[-1].abs() > 0.5:
            blend[:] = 0
        a = frag[-2]
        #blend = blend * (1-a) + a * frag[:-1]
        blend = blend * (1-a) + frag[:-1]
        outs[i, :-1] = blend
    return outs.cuda()


def blend_with_scan(data):
    out = blend_packed_power_of_two_fragment_list(data)
    return out


def plot_inds(inds, width, height, t=0):
    from algan.utils.plotting_utils import plot_tensor
    out = torch.zeros((height, width, 3), device=inds.device)
    n = t * width * height
    inds = inds[(inds >= n) & (inds < ((t+1)*width*height))]
    inds = inds.view(-1,1).expand(-1,out.shape[-1])
    inds = inds % n
    x = torch.scatter(squish(out), 0, inds, torch.ones_like(inds, dtype=out.dtype)).view(out.shape).permute(-1,0,1)
    plot_tensor(x)
    return x


class RenderPrimitive:
    def __init__(
        self,
        corners=None,
        colors=BLUE,
        opacity=0,
        normals=None,
        perimeter_points=None,
        reverse_perimeter=False,
        triangle_collection=None,
        glow=0,
        glow_radius=0.2,
    ):
        self.corners = corners
        self.colors = colors
        self.normals = normals
        self.glow_radius = glow_radius
        self.padding = 1

    def get_batch_identifier(self):
        return f"{self.__class__}"

    def get_memory_used_per_timestep(self):
        return self.num_fragments_per_frame * (128)

    def get_memory_used_for_blending(self, start_ind, end_ind):
        mem_used_for_blending = self.num_fragments_per_frame * (9 * 4 + 8) * 2  # * 3 for buffers
        return mem_used_for_blending * (end_ind - start_ind)

    def get_memory_used(self, start_ind, end_ind):
        # The blending process uses, for each fragment, 1 4-channel color and 1 5-channel color (9 floats), and one index (long), so 9*4+1*8 bytes.
        return self.get_memory_used_per_timestep() * (end_ind - start_ind)

    def render(
        self,
        primitives,
        scene,
        save_image,
        screen_width,
        screen_height,
        time_start,
        time_end,
        background_color,
        transparent_background=False,
        *args,
        **kwargs,
    ):
        screen_width *= kwargs["anti_alias_level"]
        screen_height *= kwargs["anti_alias_level"]
        window = (0, 0, screen_width, screen_height)
        kwargs["screen_width"] = screen_width
        kwargs["screen_height"] = screen_height
        return self.render_window(
            primitives,
            scene,
            window,
            save_image,
            time_start,
            time_end,
            0,
            1,
            background_color,
            False,
            transparent_background,
            *args,
            **kwargs,
        )

    def post_process_frames(self, frames, anti_alias_level, post_processes=[]):
        self.pre_post_pointers = self.memory.get_pointers()
        frame_out = frames
        if anti_alias_level > 1:
            aa_frame_out = self.get_tensor([frame_out.shape[0],
                                            frame_out.shape[1] // anti_alias_level, frame_out.shape[2] // anti_alias_level,
                                            frame_out.shape[3]], dtype=torch.uint8)
            with self.memory.temp():
                frame_temp = self.get_tensor([frame_out.shape[0],
                                            frame_out.shape[1] // anti_alias_level, frame_out.shape[2] // anti_alias_level,
                                              frame_out.shape[3]])
                frame_temp[:] = frame_out[:, ::anti_alias_level, ::anti_alias_level]
                for i in range(anti_alias_level):
                    for j in range(anti_alias_level):
                        if i == j == 0:
                            continue
                        frame_temp[:] += frame_out[:, i::anti_alias_level, j::anti_alias_level]
                frame_temp /= (anti_alias_level * anti_alias_level)
                aa_frame_out[:] = frame_temp
            frame_out = aa_frame_out
        if self.memory.scene.render_settings.fxaa:
            frame_out = (fxaa(frame_out.float().permute(0,-1,1,2)).permute(0, 2,3,1)).to(torch.uint8)
        num_channels = frame_out.shape[-1]
        for p in post_processes:
            frame_out = p(frame_out, memory=self.memory)
        if num_channels == frame_out.shape[-1]:
            if num_channels == 5:
                frame_out = frame_out[..., [*range(num_channels - 2), -1]]
            else:
                frame_out = frame_out[..., :-1]
        frame_out = frame_out.cpu().flip(-3)
        # frame_out = frame_out.transpose(0, 1)
        # frame_out[...,:3] = frame_out[...,:3].flip(-1)
        return frame_out


    def save_frames(self, frames, save_image, scene, **kwargs):
        #frames = (self.post_process_frames(frame, **kwargs) for frame in frames)
        frames = self.post_process_frames(frames, **kwargs)
        if not save_image:
            for frame in frames:
                scene.frame_queue.put(frame)
                self.memory.set_pointers(self.pre_post_pointers)
        else:
            for frame in frames:
                torchvision.utils.save_image(
                    torch.from_numpy(frame).permute(-1, 0, 1) / 255, scene.file_path
                )

    def mem_cat(self, xs, extra_buffer=0, persist=False):
        dim = 0
        x_shape = xs[0].shape
        concatenated_size = sum([x.shape[dim] for x in xs]) + extra_buffer
        out_shape = [*x_shape[:dim], concatenated_size, *x_shape[dim:][1:]]
        out = self.get_tensor(out_shape, dtype=xs[0].dtype, persist=persist)
        if xs[0].device != out.device:
            out = out.to(xs[0].device)
        torch.cat(xs, out=out[:out.shape[0]-extra_buffer])
        return out
        '''i = 0
        for x in xs:
            out[i : i + x.shape[dim]] = x
            i += x.shape[dim]
        return out'''

    def render_window(
        self,
        primitives,
        scene,
        window,
        save_image,
        time_start,
        time_end,
        object_start,
        object_end,
        background_color,
        return_frags=False,
        transparent_output=False,
        *args,
        **kwargs,
    ):
        self.memory = kwargs["memory"]
        post_processes = kwargs["post_processes"]
        kwargs2 = {k: v for k, v in kwargs.items()}
        del kwargs2["post_processes"]
        original_pointers = self.memory.get_pointers()
        try:
            screen_width = (window[2] - window[0])
            screen_height = (window[3] - window[1])
            num_frames = time_end - time_start
            num_pixels_per_frame = screen_width * screen_height
            num_pixels = num_pixels_per_frame * num_frames
            out = self.memory.get_tensor(
                (num_pixels + 1, 5 if transparent_output else 4), torch.uint8)
            out_pointers = self.memory.get_pointers()
            empty_cache()

            chunks = []
            for p in primitives:
                pointers = self.memory.get_pointers()
                p.memory = self.memory
                chunk = p.render_(
                    time_start,
                    time_end,
                    object_start,
                    object_end,
                    *args,
                    **kwargs2,
                    window_coords=window,
                )
                if chunk is not None:
                    chunks.append(chunk)
                else:
                    self.memory.set_pointers(pointers)
            # [_.clone() for _ in chunk])
                # self.memory.current_pointer = original_pointer
            if return_frags:
                return chunks

            if len(chunks) == 0:
                frames = (
                    next(
                        scene.get_frames_from_fragments(
                            None,
                            window,
                            out,
                            anti_alias_level=kwargs["anti_alias_level"],
                        )
                    )
                    for _ in range(time_end - time_start)
                )
            else:
                # colors, dists, inds = [torch.cat(_) for _ in zip(*chunks)]
                extra_buffer = 0
                if background_color.dim() > 1 and background_color.shape[0] > 1:
                    with self.memory.temp(clear_persist=True):
                        unique_inds = (self.mem_cat(list(zip(*chunks))[-1]) if len(chunks) > 1 else chunks[0][-1]).unique()
                    extra_buffer = len(unique_inds)
                    colors, dists, inds = [self.mem_cat(_, extra_buffer).clone() for _ in zip(*chunks)]
                    bg_inds = unique_inds
                    bg_cols = broadcast_gather(background_color.to(bg_inds.device), 0, bg_inds.view(-1, 1))
                    colors[-extra_buffer:] = bg_cols
                    colors[-extra_buffer:] /= 255
                    dists[-extra_buffer:] = dists[:-extra_buffer].amax(0) + 1e-2
                    inds[-extra_buffer:] = bg_inds
                else:
                    colors, dists, inds = [self.mem_cat(_, extra_buffer).clone() for _ in zip(*chunks)] if len(chunks) > 1 else chunks[0]
                frames = self.blend_frags_to_pixels(
                    colors,
                    dists,
                    inds,
                    background_color,
                    time_end - time_start,
                    kwargs["screen_width"],
                    kwargs["screen_height"],
                    num_pixels,
                    transparent_output,
                    out
                )
                frames = frames[:-1].view(num_frames, screen_height, screen_width, frames.shape[-1])
                #frames = scene.get_frames_from_fragments(
                #    frags, window, out, anti_alias_level=kwargs["anti_alias_level"]
                #)
            self.memory.set_pointers(out_pointers)
            if (window[2] - window[0]) == kwargs["screen_width"] and (
                window[3] - window[1]
            ) == kwargs["screen_height"]:
                frames = self.post_process_frames(frames, anti_alias_level=kwargs["anti_alias_level"], post_processes=post_processes)
                self.memory.set_pointers(original_pointers)
            else:
                self.memory.set_pointers(out_pointers)
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            print(f'splitting to t={(time_end - time_start)//2}, frame={(window[0] + window[2]) // 2},'
                  f' {(window[1] + window[3]) // 2}')
            self.memory.set_pointers(original_pointers)
            # All this stuff is necessary to free local variables assigned during the previous render attempt.
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.clear_frames(exc_traceback)
            # traceback.print_tb(exc_traceback)
            # exc_traceback.tb_next.tb_frame.clear()

            if (time_end - time_start) > 1:
                m = time_start + (time_end - time_start) // 2
                if background_color.dim() > 1 and background_color.shape[0] > 1:
                    bg0 = background_color[:1]
                    background_color = unsquish(background_color[1:], 0, -(time_end-time_start))
                    bg1 = torch.cat((bg0, squish(background_color[:m], 0, 1)))
                    bg2 = torch.cat((bg0, squish(background_color[m:], 0, 1)))
                else:
                    bg1 = bg2 = background_color
                return self.mem_cat([self.render_window(
                    primitives,
                    scene,
                    window,
                    save_image,
                    time_start,
                    m,
                    object_start,
                    object_end,
                    bg1,
                    False,
                    transparent_output,
                    *args,
                    **kwargs,
                ),
                self.render_window(
                    primitives,
                    scene,
                    window,
                    save_image,
                    m,
                    time_end,
                    object_start,
                    object_end,
                    bg2,
                    False,
                    transparent_output,
                    *args,
                    **kwargs,
                )])
            else:
                window_size = (window[2] - window[0]) * (window[3] - window[1])
                if window_size < 100 * 100:
                    raise OutOfRenderMemory(
                        "Rendering process ran out of memory. Please reduce the number of objects in the scene."
                    )
                xm = (window[0] + window[2]) // 2
                ym = (window[1] + window[3]) // 2

                frames = [
                    next(
                        self.render_window(
                            primitives,
                            scene,
                            w,
                            save_image,
                            time_start,
                            time_end,
                            object_start,
                            object_end,
                            background_color,
                            False,
                            transparent_output,
                            *args,
                            **kwargs,
                        )
                    )
                    for w in [
                        (window[0], window[1], xm, ym),
                        (xm, window[1], window[2], ym),
                        (window[0], ym, xm, window[3]),
                        (xm, ym, window[2], window[3]),
                    ]
                ]

                gc.collect()
                empty_cache()
                # frames = torch.cat((torch.cat((frames[0], frames[1]), 1), torch.cat((frames[2], frames[3]), 1)), 0)
                top_row = self.get_tensor(
                    [
                        frames[0].shape[0],
                        frames[0].shape[1] + frames[1].shape[1],
                        frames[0].shape[2],
                    ],
                    dtype=frames[0].dtype,
                )
                torch.cat((frames[0], frames[1]), 1, out=top_row)
                bottom_row = self.get_tensor(
                    [
                        frames[2].shape[0],
                        frames[2].shape[1] + frames[3].shape[1],
                        frames[2].shape[2],
                    ],
                    dtype=frames[2].dtype,
                )
                torch.cat((frames[2], frames[3]), 1, out=bottom_row)
                frames = self.get_tensor(
                    [
                        top_row.shape[0] + bottom_row.shape[0],
                        top_row.shape[1],
                        top_row.shape[2],
                    ],
                    dtype=top_row.dtype,
                )
                torch.cat((top_row, bottom_row), 0, out=frames)
                frame_shape = frames.shape
                frames = (_ for _ in [frames])
                if (
                    frame_shape[1] == kwargs["screen_width"]
                    and frame_shape[0] == kwargs["screen_height"]
                ):
                    self.save_frames(
                        frames,
                        save_image,
                        scene,
                        anti_alias_level=kwargs["anti_alias_level"],
                        post_processes=post_processes,
                    )
                    self.memory.set_pointers(original_pointers)
                    return None
                else:
                    return frames
        return frames

    def get_tensor_from_memory(self, *args, **kwargs):
        return self.memory.get_tensor(*args, **kwargs)

    def get_tensor(self, *args, **kwargs):
        return self.get_tensor_from_memory(*args, **kwargs)

    def expand_verts_to_frags(self, x, repeats_inds, dim=-2, out=None, persist=False):
        if out is None:
            xshape = [_ for _ in x.shape]
            xshape[dim] = repeats_inds.shape[dim]
            out = self.get_tensor(xshape, x.dtype, persist=persist)
        return broadcast_gather(x, dim, repeats_inds, out=out)

    def blend_frags_to_pixels_old(
        self,
        colors,
        dists,
        inds,
        background_color,
        num_frames,
        screen_width,
        screen_height,
        transparent_output=False,
    ):
        colors[..., -1].clamp_(min=0, max=1)
        unique_inds, unique_inds_inverse, unique_counts = inds.unique(
            return_inverse=True, return_counts=True
        )

        current_frags = self.get_tensor(
            (len(unique_inds), colors.shape[-1] - (0 if transparent_output else 1)),
            torch.float,
        )
        out_pointer = self.memory.current_pointer
        self.memory.save_pointer()

        if unique_counts.numel() == 0:
            max_buffer_depth = 1
        else:
            max_buffer_depth = unique_counts.amax()

        out = current_frags
        out[..., :] = background_color[..., : out.shape[-1]]
        # out[..., -1] = 0

        # TODO make it so that if opacity is 0, that pixel is removed entirely (instead of just painting background constants), this will save us having to
        # render invisible objects.

        original_pointer = self.memory.current_pointer
        inds_selected_ = self.get_tensor(unique_inds_inverse.shape, unique_inds_inverse.dtype)
        c_write_ = self.get_tensor(colors.shape, colors.dtype)
        c_read_ = self.get_tensor(out.shape, out.dtype)

        #@compiled
        @not_compiled
        def blend_colors(dists, inds, colors, out):
            for i in range(max_buffer_depth):
                max_dist, max_ind = scatter_arg_max(
                    dists, inds, -1, dim_size=out.shape[-2]
                )
                if max_ind is None:
                    break

                blend_one_layer_of_fragments(dists, inds, colors, out, max_dist, max_ind, inds_selected_, c_write_, c_read_, self.memory)
            self.memory.current_pointer = original_pointer
            return out

        out = blend_colors(dists, unique_inds_inverse, colors, out)
        self.memory.reset_pointer()

        out_inds = unique_inds.scatter_(0, unique_inds_inverse, inds)
        # ind_counts = torch.histc(out_inds.float(), num_frames, min=0, max=(screen_width * screen_height * num_frames)).long()
        float_tensor = self.get_tensor(out_inds.shape, dtype=torch.float)
        float_tensor.copy_(out_inds)
        histc_result = torch.histc(
            float_tensor,
            num_frames,
            min=0,
            max=(screen_width * screen_height * num_frames),
        )
        ind_counts = histc_result.long()
        self.memory.current_pointer = out_pointer
        return out, out_inds, ind_counts

    def blend_frags_to_pixels(
            self,
            colors,
            dists,
            inds,
            background_color,
            num_frames,
            screen_width,
            screen_height,
            num_pixels,
            transparent_output=False,
            out=None,
    ):
        colors[..., -1].clamp_(min=0, max=1)
        #pre-multiply
        colors[..., :-1] *= colors[..., -1:]

        with self.memory.temp():
            dists = self.memory.cast(dists, torch.double)
            dists -= dists.view(-1).amin(0)
            fragment_order = torch.add(inds, dists, alpha=1/(dists.view(-1).amax(0) + 1), out=dists)
            sorted_frgment_order, s = torch.sort(fragment_order, stable=True, dim=0, descending=True,
                              out=(self.get_tensor(fragment_order.shape, torch.double),
                                   self.get_tensor(fragment_order.shape, torch.long)))
            N = colors.shape[0]
            padded_N = 1 << (N).bit_length() if N > 1 else N
            buffer = self.memory.get_tensor((padded_N, colors.shape[1]+1))
            buffer[:] = 0
            sorted_colors = broadcast_gather(colors, 0, s.view(-1,1), out=buffer[:N,:-1], keepdim=True)
            sorted_frgment_order.floor_()
            #sorted_frgment_order = sorted_frgment_order.float()# remove distance component, now we just have fragment ID in sorted order.
            with self.memory.temp():
                difs = torch.diff(sorted_frgment_order, dim=0, out=self.memory.get_tensor(sorted_frgment_order.shape),
                                  prepend=torch.tensor((1e12,), device=sorted_frgment_order.device))
                mask = torch.lt(difs, -0.5, out=buffer[:N,-1])
            mask_out = self.memory.get_tensor([mask.shape[0] + 1], torch.bool)
            mask_out[-1] = True
            mask = torch.gt(mask, 0.5, out=mask_out[:-1])
            mask = mask_out[1:]

            #buffer2 = blend_seq(buffer)
            #sorted_colors = buffer2[:N, :-1]
            blend_with_scan(buffer[:padded_N])
            sorted_colors = buffer[1:N+1, :-1]
            out[..., :] = background_color[..., : out.shape[-1]]
            sorted_frgment_order = self.memory.cast(sorted_frgment_order, torch.long)
            fragment_to_pixel_ind = sorted_frgment_order
            #fragment_to_pixel_ind = torch.where(mask, sorted_frgment_order,
            #                                    torch.tensor((num_pixels,), device=out.device
            #                                                 ), out=sorted_frgment_order)
            frags = sorted_colors[:, :out.shape[-1]]
            frags *= 255
            frags.clamp_(min=0, max=255)
            frags = self.memory.cast(frags, torch.uint8)
            out = torch.scatter(out, 0, fragment_to_pixel_ind[mask].unsqueeze(1).expand(-1,out.shape[-1]), frags[mask], out=out)
            return out
            return out[:-1].view(num_frames, screen_width, screen_height, out.shape[-1])



            dists += inds

        #TODO rest

        unique_inds, unique_inds_inverse, unique_counts = inds.unique(
            return_inverse=True, return_counts=True
        )

        current_frags = self.get_tensor(
            (len(unique_inds), colors.shape[-1] - (0 if transparent_output else 1)),
            torch.float,
        )
        out_pointer = self.memory.current_pointer
        self.memory.save_pointer()

        if unique_counts.numel() == 0:
            max_buffer_depth = 1
        else:
            max_buffer_depth = unique_counts.amax()

        out = current_frags
        out[..., :] = background_color[..., : out.shape[-1]]
        # out[..., -1] = 0

        # TODO make it so that if opacity is 0, that pixel is removed entirely (instead of just painting background constants), this will save us having to
        # render invisible objects.

        original_pointer = self.memory.current_pointer
        inds_selected_ = self.get_tensor(unique_inds_inverse.shape, unique_inds_inverse.dtype)
        c_write_ = self.get_tensor(colors.shape, colors.dtype)
        c_read_ = self.get_tensor(out.shape, out.dtype)

        # @compiled
        @not_compiled
        def blend_colors(dists, inds, colors, out):
            for i in range(max_buffer_depth):
                max_dist, max_ind = scatter_arg_max(
                    dists, inds, -1, dim_size=out.shape[-2]
                )
                if max_ind is None:
                    break

                blend_one_layer_of_fragments(dists, inds, colors, out, max_dist, max_ind, inds_selected_, c_write_,
                                             c_read_, self.memory)
            self.memory.current_pointer = original_pointer
            return out

        out = blend_colors(dists, unique_inds_inverse, colors, out)
        self.memory.reset_pointer()

        out_inds = unique_inds.scatter_(0, unique_inds_inverse, inds)
        # ind_counts = torch.histc(out_inds.float(), num_frames, min=0, max=(screen_width * screen_height * num_frames)).long()
        float_tensor = self.get_tensor(out_inds.shape, dtype=torch.float)
        float_tensor.copy_(out_inds)
        histc_result = torch.histc(
            float_tensor,
            num_frames,
            min=0,
            max=(screen_width * screen_height * num_frames),
        )
        ind_counts = histc_result.long()
        self.memory.current_pointer = out_pointer
        return out, out_inds, ind_counts

    def get_windowed_bounding_boxes(
        self, bounding_corners, screen_width, screen_height, window_coords=None
    ):
        if window_coords is None:
            window_coords = (0, 0, screen_width, screen_height)
        start_x, start_y, end_x, end_y = window_coords
        #end_x = end_x
        #end_y = end_y
        # bounding_corners = bounding_corners.clamp(
        #     min=torch.tensor((start_x, start_y), device=bounding_corners.device),
        #     max=torch.tensor((end_x, end_y), device=bounding_corners.device))
        #min_tensor = self.get_tensor([2], dtype=bounding_corners.dtype)
        #min_tensor[0] = start_x
        #min_tensor[1] = start_y
        min_tensor = torch.tensor((start_x, start_y), device=bounding_corners.device, dtype=bounding_corners.dtype).view(2)
        #max_tensor = self.get_tensor([2], dtype=bounding_corners.dtype)
        #max_tensor[0] = end_x
        #max_tensor[1] = end_y
        max_tensor = torch.tensor((end_x, end_y), device=bounding_corners.device, dtype=bounding_corners.dtype).view(2)
        bounding_corners = torch.clamp(
            bounding_corners,
            min=min_tensor,
            max=max_tensor,
            out=self.get_tensor(bounding_corners.shape, bounding_corners.dtype),
        )
        # bounding_box_sizes = (bounding_corners[..., 1, :] - bounding_corners[..., 0, :])
        bounding_box_sizes = self.get_tensor(
            bounding_corners[..., 1, :].shape, dtype=bounding_corners.dtype
        )
        torch.subtract(
            bounding_corners[..., 1, :],
            bounding_corners[..., 0, :],
            out=bounding_box_sizes,
        )
        # bbss = bounding_box_sizes.prod(-1, keepdim=True)
        bbss = self.get_tensor(
            [*bounding_box_sizes.shape[:-1], 1], dtype=bounding_box_sizes.dtype
        )
        torch.prod(bounding_box_sizes, -1, keepdim=True, out=bbss)
        # num_fragments_per_object = bbss.amax(0)
        num_fragments_per_object = self.get_tensor(bbss.shape[1:], dtype=bbss.dtype)
        torch.amax(bbss, 0, out=num_fragments_per_object)
        # num_fragments_per_frame = num_fragments_per_object.sum()
        num_fragments_per_frame = torch.sum(num_fragments_per_object)
        num_fragments = num_fragments_per_frame * bbss.shape[0]

        return (
            bounding_corners,
            bounding_box_sizes,
            bbss,
            num_fragments_per_object,
            num_fragments_per_frame,
            num_fragments,
            None,
        )

    def project_and_get_bounding_boxes(
        self,
        x,
        ray_origin,
        screen_point,
        screen_basis,
        screen_width,
        screen_height,
        window_coords=None,
            memory=None,
    ):

        original_persist_pointer = memory.current_reverse_pointer
        with memory.temp():
            rays = torch.sub(x, ray_origin, out=memory.get_tensor(x.shape, persist=True))
            rays = normalize(rays, p=2, dim=-1, memory=memory)
            projected_corners, _ = intersect_line_with_plane(
                rays, screen_point, screen_basis[..., -1:, :], ray_origin, memory=memory
            )
        projected_corners.nan_to_num_()
        projected_distances = distance(x, ray_origin, memory=memory)
        projected_corners -= screen_point
        corners_2d = dot_product(
            projected_corners.unsqueeze(-2),
            screen_basis[..., :-1, :].unsqueeze(-3),
            -1,
            keepdim=False, out=memory
        )
        memory.current_reverse_pointer = original_persist_pointer
        corners_2d.nan_to_num_()

        corners_2d *= screen_height // 2
        corners_2d[..., 0] += screen_width // 2
        corners_2d[..., 1] += screen_height // 2

        corners = corners_2d
        corners_int = memory.cast(corners, torch.int)

        # bounding_corners = torch.stack(((corners_int.amin(-2) - self.padding),
        #                                 (corners_int.amax(-2) + self.padding)),
        #                                -2)
        bounding_corners = self.get_tensor(
            [*corners_int.shape[:-2], 2, corners_int.shape[-1]], dtype=corners_int.dtype
        )
        min_corners = torch.amin(corners_int, -2, out=bounding_corners[..., 0, :])
        max_corners = torch.amax(corners_int, -2, out=bounding_corners[..., 1, :])
        min_corners -= self.padding
        max_corners += self.padding

        (
            bounding_corners,
            bounding_box_sizes,
            bbss,
            num_fragments_per_object,
            num_fragments_per_frame,
            num_fragments,
            _,
        ) = self.get_windowed_bounding_boxes(
            bounding_corners, screen_width, screen_height, window_coords
        )

        return (
            corners,
            corners_int,
            projected_distances,
            bounding_corners,
            bounding_box_sizes,
            bbss,
            num_fragments_per_object,
            num_fragments_per_frame,
            num_fragments,
            _,
        )

    def project_to_screen(self, camera, light_sources):
        ray_origin = camera.ray_origin
        screen_point = camera.screen_point
        screen_basis = camera.screen_basis
        screen_width = camera.screen_width
        screen_height = camera.screen_height

        light_intensity = 1
        ambient_light_intensity = 1
        d = -1
        if hasattr(self, "shader") and self.shader is not None:
            for light_source in light_sources:
                with self.memory.temp():
                    self.colors[..., :d] = self.shader(self.memory,
                        self.corners,
                        self.normals,
                        self.colors[..., :d],
                        ray_origin,
                        light_source.origin,
                        light_source.light_color,
                        light_intensity,
                        ambient_light_intensity,
                        *self.shader_param_values,
                    )

        self.first_projection = True
        (
            self.corners,
            self.corners_int,
            self.projected_distances,
            self.bounding_corners,
            self.bounding_box_sizes,
            self.bbss,
            self.num_fragments_per_object,
            self.num_fragments_per_frame,
            self.num_fragments,
            _,
        ) = self.project_and_get_bounding_boxes(
            self.corners,
            ray_origin,
            screen_point,
            screen_basis,
            screen_width,
            screen_height,
            memory=self.memory
        )
        self.first_projection = False
        return self

    #@compiled
    def render_(
        self,
        time_start,
        time_end,
        object_start,
        object_end,
        ray_origin,
        screen_point,
        screen_basis,
        background_color=BLACK,
        anti_alias=False,
        anti_alias_offset=[0.5, 0.5],
        anti_alias_level=1,
        light_sources=[],
        screen_width=2000,
        screen_height=2000,
        window_coords=None,
        memory=None,
        primitive_type=None,
    ):
        def select_time(x):
            x = x if len(x) == 1 else x[time_start:time_end]
            x = (
                x
                if x.shape[1] == 1
                else x[:, int(x.shape[1] * object_start) : int(x.shape[1] * object_end)]
            )
            return x

        corners = select_time(self.corners)
        corners_int = select_time(self.corners_int)
        projected_distances = select_time(self.projected_distances)
        colors = select_time(self.colors)

        if window_coords is None:
            window_coords = 0, 0, screen_width, screen_height

        window_width = window_coords[2] - window_coords[0]
        window_height = window_coords[3] - window_coords[1]

        fragment_count = torch.zeros((1,), device=corners.device, dtype=torch.long)
        pointer = memory.current_pointer
        num_outputs = 8
        bytes_per_output = 4 * num_outputs
        out_n_total = (memory.current_reverse_pointer - (memory.current_pointer+1))//bytes_per_output
        out_n_per_var = out_n_total // num_outputs
        out_buffer = memory.get_tensor((out_n_per_var, 6), torch.float)
        out_ind_buffer = memory.get_tensor((out_n_per_var,), torch.long)
        rasterize_triangle_taichi(squish(corners), squish(colors), squish(projected_distances),
                                      out_buffer, out_ind_buffer,
                                      window_width, window_height, fragment_count, corners.shape[1])
        memory.current_pointer = pointer
        if fragment_count > out_buffer.shape[0]:
            raise InsufficientMemoryException
        out = memory.get_tensor((fragment_count, num_outputs-2), torch.float)
        inds_ = memory.clone(out_ind_buffer[:fragment_count])
        colors_ = out[...,:5]
        dists_ = out[...,5]
        #inds_ = out[...,6]
        #inds_ = memory.cast(inds_, torch.long, persist=True)
        return colors_, dists_, inds_

        start_x, start_y, end_x, end_y = window_coords

        corners_locs, corners_inds, projected_distances = (
            corners,
            corners_int,
            projected_distances,
        )
        bounding_box_num_pixels = self.num_fragments_per_object
        bounding_box_sizes = select_time(self.bounding_box_sizes)
        bounding_corners = select_time(self.bounding_corners)

        if window_width < screen_width or window_height < screen_height:
            (
                bounding_corners,
                bounding_box_sizes,
                bbss,
                bounding_box_num_pixels,
                num_fragments_per_frame,
                num_fragments,
                _,
            ) = self.get_windowed_bounding_boxes(
                bounding_corners, screen_width, screen_height, window_coords
            )

        original_pointers = self.memory.get_pointers()

        repeats = bounding_box_num_pixels.view(-1)
        num_frags = repeats.sum().item()
        inds_shape = [*bounding_box_sizes.shape[:-2], num_frags, 1]

        if num_frags == 0:
            return None

        fragment_inds = self.get_tensor([num_frags], dtype=torch.long)
        memory = self.memory
        with memory.temp():
            # repeats_inds = torch.repeat_interleave(torch.arange(len(repeats), device=repeats.device), repeats, -1, output_size=num_frags).unsqueeze(-1)
            arange_tensor = self.get_tensor([len(repeats)], dtype=torch.long)
            torch.arange(len(repeats), device=arange_tensor.device, out=arange_tensor)
            repeats_inds = torch.repeat_interleave(
                arange_tensor, repeats, -1, output_size=num_frags
            ).unsqueeze(-1)

            # offsets = self.expand_verts_to_frags(bounding_box_num_pixels.cumsum(-2) - bounding_box_num_pixels, repeats_inds, -2)
            offsets = self.get_tensor([num_frags, 1], dtype=torch.long)
            cumsum_tensor = self.get_tensor(bounding_box_num_pixels.shape, dtype=torch.long)
            torch.cumsum(bounding_box_num_pixels, -2, out=cumsum_tensor)
            cumsum_tensor -= bounding_box_num_pixels
            offsets = self.expand_verts_to_frags(
                cumsum_tensor, repeats_inds, -2, out=offsets
            )
            # Free cumsum_tensor as it's no longer needed
            # fragment_inds = torch.arange(offsets.shape[-2], device=offsets.device).view(-1,1) - offsets
            fragment_inds = torch.arange(
                offsets.shape[-2], device=fragment_inds.device, out=fragment_inds
            ).view(-1, 1)
            fragment_inds -= offsets

        corners_locs = self.expand_verts_to_frags(
            corners_locs, repeats_inds.unsqueeze(-1), -3, persist=True
        )
        """bisector_lengths = self.get_tensor([*corners_locs.shape[:-1], 1])
        with memory.temp():
            mid_point_locs = self.get_tensor(corners_locs.shape)
            mid_point_locs[..., 0, :] = torch.add(
                corners_locs[..., 1, :],
                corners_locs[..., 2, :],
                out=mid_point_locs[..., 0, :],
            )
            mid_point_locs[..., 1, :] = torch.add(
                corners_locs[..., 0, :],
                corners_locs[..., 2, :],
                out=mid_point_locs[..., 1, :],
            )
            mid_point_locs[..., 2, :] = torch.add(
                corners_locs[..., 0, :],
                corners_locs[..., 1, :],
                out=mid_point_locs[..., 2, :],
            )
            mid_point_locs *= 0.5
            bisector_segments = torch.subtract(
                corners_locs, mid_point_locs, out=mid_point_locs
            )
            torch.norm(bisector_segments, p=2, dim=-1, keepdim=True, out=bisector_lengths)"""

        pointer = self.memory.current_pointer
        bounding_box_widths = self.expand_verts_to_frags(
            bounding_box_sizes[..., :1], repeats_inds, -2
        )  # .clamp_min_(1)
        bounding_box_widths.clamp_min_(1)
        fragment_inds_int = self.get_tensor(bounding_box_widths.shape, torch.int)
        fragment_x = self.get_tensor(bounding_box_widths.shape, torch.int)
        fragment_y = self.get_tensor(bounding_box_widths.shape, torch.int)
        bounding_corners_rep = self.expand_verts_to_frags(
            bounding_corners[..., 0, :], repeats_inds, -2
        )
        # fragment_x = self.get_tensor(bounding_box_widths.shape, torch.long)
        # fragment_x[:] = (fragment_inds % bounding_box_widths) + bounding_corners_rep[...,:1]
        torch.remainder(
            fragment_inds,
            bounding_box_widths,
            out=fragment_x,
        )
        torch.add(fragment_x, bounding_corners_rep[..., :1], out=fragment_x)
        # fragment_y = self.get_tensor(bounding_box_widths.shape, torch.long)
        # fragment_y[:] = (fragment_inds // bounding_box_widths) + bounding_corners_rep[...,1:]
        torch.div(
            fragment_inds, bounding_box_widths, rounding_mode="floor", out=fragment_y
        )
        torch.add(fragment_y, bounding_corners_rep[..., 1:], out=fragment_y)

        # aa_offsets = torch.linspace(0, 1, anti_alias_level * 2 + 1, device=fragment_x.device)[1:-1:2]
        # aa_offsets = squish(torch.stack((aa_offsets.view(-1, 1).expand([-1, len(aa_offsets)]), aa_offsets.view(1, -1).expand([len(aa_offsets), -1])), -1))
        # inds = (fragment_x - start_x) + (fragment_y - start_y) * window_width
        torch.mul(fragment_y, window_width, out=fragment_inds_int)
        fragment_inds_int += fragment_x
        fragment_inds_int -= start_y * window_width + start_x
        with memory.temp():
            fragment_x = memory.cast(fragment_x, torch.float)
            fragment_y = memory.cast(fragment_y, torch.float)
            all_ws = self.get_interpolation_coordinates(
                corners_locs, fragment_x, fragment_y, None
            )
        inds = memory.cast(fragment_inds_int, torch.long, persist=True)
        self.memory.current_pointer = pointer

        #
        #bisector_lengths *= all_ws
        ###distance_to_border = torch.amin(bisector_lengths, -2, out=bisector_lengths[..., 0, :])
        # all_mask = (min_w >= self.min_interpolation_coord).any(0)
        all_mask = self.get_tensor([*all_ws.shape[:-2], 1], dtype=torch.bool, persist=True)
        pointer = self.memory.current_pointer
        min_w = self.get_tensor([*all_ws.shape[:-2], 1])
        distance_to_border = torch.amin(all_ws, -2, out=min_w)
        ###distance_to_border += 0.5 + 1e-1
        torch.greater_equal(
            distance_to_border, self.min_interpolation_coord, out=all_mask
        )
        self.memory.current_pointer = original_pointers[0]
        # distance_to_border += 0.5
        ###anti_alias_mask = torch.clamp(distance_to_border, 0, 1, out=distance_to_border)
        ###anti_alias_mask.nan_to_num_(0, 1, 0)

        # TODO subtract window start from fragment x and y
        window_size = window_width * window_height

        with memory.temp():
            m = torch.lt(inds, window_size, out=memory.get_tensor(inds.shape, torch.bool))
            m = torch.logical_and(m, all_mask, out=all_mask).view(-1)

        with memory.temp():
            arange_window = self.get_tensor([inds.shape[0]], dtype=torch.long)
            torch.arange(inds.shape[0], device=arange_window.device, out=arange_window)
            torch.mul(arange_window, window_size, out=arange_window)
            unsqueezed_arange = unsqueeze_right(arange_window, inds)
            torch.add(inds, unsqueezed_arange, out=inds)
            inds = inds.view(-1)
        num_masked_frags = m.sum()

        # unique_inds, unique_inds_inverse, unique_counts = inds.unique(return_inverse=True, return_counts=True)

        self_colors = colors
        # output_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))
        # output_frags[:] = 0
        ##current_frags = self.get_tensor((len(unique_inds), colors.shape[-1]-1))

        def get_frags(ws):
            def interpolate(x):
                return self.interpolate_property(ws, x, repeats_inds)

            def get_colors():
                persis_pointer = self.memory.current_reverse_pointer
                colors = interpolate(self_colors)
                # colors[..., -1:] *= anti_alias_mask
                colors = colors.view(-1, colors.shape[-1])
                colors = torch.masked_select(colors, m.unsqueeze(-1), out=memory.get_tensor((num_masked_frags * colors.shape[-1],))).view(-1,colors.shape[-1])
                self.memory.current_reverse_pointer = persis_pointer
                return colors

            def get_dists():
                persis_pointer = self.memory.current_reverse_pointer
                dists = interpolate(projected_distances)
                dists = dists.view(-1)
                #dists = dists[m]
                dists = torch.masked_select(dists, m, out=memory.get_tensor((num_masked_frags,)))
                self.memory.current_reverse_pointer = persis_pointer
                return dists

            colors, dists = get_colors(), get_dists()
            return colors, dists

        colors, dists = get_frags(all_ws)

        inds = torch.masked_select(inds, m, out=memory.get_tensor((num_masked_frags,), torch.long))
        self.memory.current_reverse_pointer = original_pointers[1]
        return colors, dists, inds


class RenderPrimitive2D(RenderPrimitive):
    def raycast_onto_plane(self, ray_origins, ray_directions, plane_point, plane_basis):
        dists = -dot_product(ray_origins - plane_point, plane_basis) / dot_product(
            ray_directions, plane_basis
        )
        dists.nan_to_num_()
        return dists
