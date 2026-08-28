from __future__ import annotations

import torch

from algan.rendering.post_processing.anti_aliasing.fxaa import fxaa
from algan.settings import SETTINGS
from algan.utils.color_space import linear_to_srgb

__all__ = [
    "post_process_frames",
]


def _linear_rgb(src, dst, coefficients, scratch):
    """Apply a 3x3 color transform with the source expression's operation order."""
    for output_channel, row in enumerate(coefficients):
        torch.mul(src[:, 0], row[0], out=dst[:, output_channel])
        torch.mul(src[:, 1], row[1], out=scratch)
        dst[:, output_channel].add_(scratch)
        torch.mul(src[:, 2], row[2], out=scratch)
        dst[:, output_channel].add_(scratch)


def _neutral_tonemap(rgb, exposure, memory):
    flat = rgb.reshape(-1, 3)
    color_exposed = memory.clone(flat)
    color_exposed.mul_(exposure)
    pixels = flat.shape[0]

    minimum = memory.get_tensor((pixels, 1), color_exposed.dtype)
    torch.amin(color_exposed, dim=1, keepdim=True, out=minimum)
    low_mask = memory.get_tensor((pixels, 1), torch.bool)
    torch.lt(minimum, 0.08, out=low_mask)
    offset = memory.get_tensor((pixels, 1), color_exposed.dtype)
    torch.mul(minimum, minimum, out=offset)
    offset.mul_(6.25)
    torch.sub(minimum, offset, out=offset)
    constant = memory.get_tensor((1,), color_exposed.dtype)
    constant.fill_(0.04)
    torch.where(low_mask, offset, constant, out=offset)

    color_offset = memory.clone(color_exposed)
    color_offset.sub_(offset)
    peak = memory.get_tensor((pixels, 1), color_exposed.dtype)
    torch.amax(color_offset, dim=1, keepdim=True, out=peak)
    compress_mask = memory.get_tensor((pixels, 1), torch.bool)
    torch.ge(peak, 0.76, out=compress_mask)

    # Compute the compression formula for every row, then select only the
    # active rows.  This has a deterministic, data-independent arena size and
    # is algebraically identical to the old boolean-indexed branch.
    new_peak = memory.clone(peak)
    new_peak.add_(0.24).sub_(0.76)
    scratch_scalar = memory.get_tensor((pixels, 1), color_exposed.dtype)
    scratch_scalar.fill_(0.24 * 0.24).div_(new_peak)
    new_peak.fill_(1.0).sub_(scratch_scalar)

    scale = memory.get_tensor((pixels, 1), color_exposed.dtype)
    torch.div(new_peak, peak, out=scale)
    constant.fill_(1.0)
    torch.where(compress_mask, scale, constant, out=scale)
    color_c = memory.clone(color_offset)
    color_c.mul_(scale)

    torch.sub(peak, new_peak, out=scratch_scalar)
    scratch_scalar.mul_(0.15).add_(1.0).reciprocal_()
    scratch_scalar.neg_().add_(1.0)  # g

    compressed = memory.get_tensor(color_c.shape, color_c.dtype)
    torch.sub(new_peak, color_c, out=compressed)
    compressed.mul_(scratch_scalar).add_(color_c)
    torch.where(compress_mask, compressed, color_offset, out=color_offset)
    color_offset.clamp_(0.0, 1.0)
    return color_offset.reshape(rgb.shape)


def _agx_curve(src, dst, x2, x4, scratch):
    torch.mul(src, src, out=x2)
    torch.mul(x2, x2, out=x4)
    torch.mul(x4, 15.5, out=dst)
    dst.mul_(x2)
    torch.mul(x4, 40.14, out=scratch)
    scratch.mul_(src)
    dst.sub_(scratch)
    torch.mul(x4, 31.96, out=scratch)
    dst.add_(scratch)
    torch.mul(x2, 6.868, out=scratch)
    scratch.mul_(src)
    dst.sub_(scratch)
    torch.mul(x2, 0.4298, out=scratch)
    dst.add_(scratch)
    torch.mul(src, 0.1191, out=scratch)
    dst.add_(scratch).sub_(0.00232)


def _agx_tonemap(rgb, exposure, memory):
    shape = rgb.shape
    a = memory.clone(rgb.reshape(-1, 3))
    a.mul_(exposure)
    b = memory.get_tensor(a.shape, a.dtype)
    scratch = memory.get_tensor((a.shape[0],), a.dtype)

    _linear_rgb(
        a,
        b,
        (
            (0.627409, 0.329282, 0.043309),
            (0.069055, 0.919540, 0.011405),
            (0.016390, 0.088013, 0.895597),
        ),
        scratch,
    )
    _linear_rgb(
        b,
        a,
        (
            (0.856627153315983, 0.0951212405381588, 0.0482516061458583),
            (0.137318972929847, 0.761241990602591, 0.101439036467562),
            (0.11189821299995, 0.0767994186031903, 0.811302368396859),
        ),
        scratch,
    )

    for channel in range(3):
        a[:, channel].clamp_(min=1e-10).log2_().clamp_(-12.47393, 4.026069)
        a[:, channel].sub_(-12.47393).div_(4.026069 - (-12.47393))

    x2 = memory.get_tensor((a.shape[0],), a.dtype)
    x4 = memory.get_tensor((a.shape[0],), a.dtype)
    for channel in range(3):
        _agx_curve(a[:, channel], b[:, channel], x2, x4, scratch)

    _linear_rgb(
        b,
        a,
        (
            (1.1271005818144368, -0.11060664309660323, -0.016493938717834573),
            (-0.1413297634984383, 1.157823702216272, -0.016493938717834257),
            (-0.14132976349843826, -0.11060664309660294, 1.2519364065950405),
        ),
        scratch,
    )
    # Linear Rec.2020 -> linear Rec.709. Both spaces share the D65 white point,
    # so this must map white to white and every row must therefore sum to 1.
    # It was written transposed until 2026-08-22, giving row sums of
    # 1.5177 / 0.4447 / 1.0376 -- a fixed +52% red, -56% green on any neutral,
    # which rendered authored grey (128,128,128) as magenta (255,77,180).
    _linear_rgb(
        a,
        b,
        (
            (1.6605, -0.5876, -0.0728),
            (-0.1246, 1.1329, -0.0083),
            (-0.0182, -0.1006, 1.1187),
        ),
        scratch,
    )
    b.clamp_(0.0, 1.0)
    return b.reshape(shape)


def _strip_aux_channel(frame, original_num_channels, memory):
    if original_num_channels != frame.shape[-1]:
        return frame
    if original_num_channels == 5:
        stripped = memory.get_tensor((*frame.shape[:-1], 4), frame.dtype)
        stripped[..., 0].copy_(frame[..., 0])
        stripped[..., 1].copy_(frame[..., 1])
        stripped[..., 2].copy_(frame[..., 2])
        stripped[..., 3].copy_(frame[..., 4])
        return stripped
    return frame[..., :-1]


def _finalize_on_device(
    frame,
    original_num_channels,
    memory,
    *,
    tonemap_enabled,
    tonemapping,
    tonemap_method,
    exposure,
):
    """Strip render-only channels and return arena-owned uint8 frames."""
    # Fast path: the input is already byte output.  Only the transparent
    # five-channel layout needs a copied/reordered result.
    if not tonemap_enabled and frame.dtype == torch.uint8:
        return _strip_aux_channel(frame, original_num_channels, memory)

    stripped_channels = (
        4
        if original_num_channels == frame.shape[-1] == 5
        else (
            frame.shape[-1] - 1
            if original_num_channels == frame.shape[-1]
            else frame.shape[-1]
        )
    )
    output_channels = (
        (4 if stripped_channels == 4 else 3) if tonemap_enabled else stripped_channels
    )
    output = memory.get_tensor((*frame.shape[:-1], output_channels), torch.uint8)

    # Standalone Taichi tonemap (post_tonemap_kernel): under post-process
    # tonemapping the frame arrives as a linear-HDR float buffer bloom has
    # already run on. Tonemap + quantize in one f32 Taichi pass (reusing the
    # in-composite tonemap ti.funcs) instead of the ~20-op/pixel torch
    # pipeline -- it reads RGB (0-2), drops the glow channel and picks up any
    # alpha (channel 4) itself, so it needs no torch strip.
    _rt = SETTINGS.raytracing
    if (
        tonemap_enabled
        and frame.dtype != torch.uint8
        and _rt.is_post_tonemap_kernel_enabled()
    ):
        from algan.rendering.post_processing.tonemap_kernels_taichi import (
            tonemap_to_u8,
        )

        method_id = 0 if not tonemapping else (1 if tonemap_method == "neutral" else 2)
        if tonemapping and tonemap_method not in ("neutral", "agx"):
            raise ValueError(f"Unknown tonemapping method: {tonemap_method}")
        tonemap_to_u8(
            frame,
            output,
            method_id,
            float(exposure),
            1 if frame.shape[-1] == 5 else 0,
            1 if _rt.linear_color_space else 0,
        )
        return output

    with memory.temp():
        stripped = _strip_aux_channel(frame, original_num_channels, memory)

        if tonemap_enabled:
            rgb_source = stripped[..., :3]
            if rgb_source.dtype == torch.uint8:
                rgb = memory.cast(rgb_source, torch.float32)
                rgb.div_(255.0)
            else:
                rgb = rgb_source

            if tonemapping:
                if tonemap_method == "neutral":
                    rgb_tonemapped = _neutral_tonemap(rgb, exposure, memory)
                elif tonemap_method == "agx":
                    rgb_tonemapped = _agx_tonemap(rgb, exposure, memory)
                else:
                    raise ValueError(f"Unknown tonemapping method: {tonemap_method}")
            else:
                # Exposure applies here too, not just under a curve: it is the
                # documented "the whole scene is too dark" control, and with
                # tonemapping off (the default) it is the only one. Exact at
                # the default exposure of 1.0, so this moves no pixel by
                # itself.
                rgb_tonemapped = memory.clone(rgb)
                if exposure != 1.0:
                    rgb_tonemapped.mul_(exposure)
                rgb_tonemapped.clamp_(0.0, 1.0)

            if _rt.linear_color_space:
                # Twin of the OETF in ``tonemap_to_u8``; kept in step with it.
                # Applied last, after exposure and after any curve.
                if stripped.shape[-1] == 4:
                    # RGB is premultiplied by coverage with alpha carried
                    # separately, and the transfer function is not linear, so
                    # the premultiplied value cannot be encoded directly --
                    # unpremultiply, encode, re-premultiply. See the same
                    # reasoning in ``tonemap_to_u8``.
                    a = stripped[..., 3:].float().div(255.0).clamp_(0.0, 1.0)
                    safe = a.clamp_min(1e-6)
                    rgb_tonemapped = linear_to_srgb(rgb_tonemapped / safe) * a
                else:
                    rgb_tonemapped = linear_to_srgb(rgb_tonemapped)

            scaled = memory.clone(rgb_tonemapped)
            scaled.mul_(255.0).add_(0.5).clamp_(0.0, 255.0)
            output[..., :3].copy_(scaled)
            if stripped.shape[-1] == 4:
                alpha = stripped[..., 3:]
                if alpha.dtype == torch.uint8:
                    output[..., 3:].copy_(alpha)
                else:
                    alpha_u8 = memory.clone(alpha)
                    alpha_u8.clamp_(0.0, 255.0)
                    output[..., 3:].copy_(alpha_u8)
        else:
            scaled = memory.clone(stripped)
            scaled.mul_(255).clamp_max_(255)
            output.copy_(scaled)

    return output


def post_process_frames(
    self, frames, anti_alias_level, post_processes=(), apply_fxaa=False
):
    """Downsample, anti-alias, run the post-process chain and tonemap.

    ``self`` is the render arena. Nothing here declares how much memory the
    pipeline needs: the render loop measures the arena's high-water mark over a
    whole chunk, so an arbitrary user-supplied post-process is accounted for by
    running it, not by describing it.
    """
    rt_settings = SETTINGS.raytracing
    # Byte frames are never linear-HDR, whatever the toggle says: the render
    # loop picks the float buffer from the same setting, but a caller that
    # hands over uint8 frames (or a scene rendered before the toggle flipped)
    # must keep the in-composite-tonemap behaviour -- _finalize_on_device makes
    # the same dtype check -- rather than dividing a byte tensor by 255.
    hdr = rt_settings.is_post_process_tonemap_enabled() and frames.dtype != torch.uint8

    self.pre_post_pointers = self.get_pointers()
    frame_out = frames
    if anti_alias_level > 1:
        # Downsample in the frame's own dtype: uint8 in the in-composite
        # tonemap mode (as before), but float16 under post-process tonemapping
        # so the supersample average stays in linear HDR instead of clamping
        # to 0-255 before bloom.
        aa_frame_out = self.get_tensor(
            [
                frame_out.shape[0],
                frame_out.shape[1] // anti_alias_level,
                frame_out.shape[2] // anti_alias_level,
                frame_out.shape[3],
            ],
            dtype=frame_out.dtype,
        )
        with self.temp():
            frame_temp = self.get_tensor(aa_frame_out.shape, torch.float32)
            frame_temp.copy_(frame_out[:, ::anti_alias_level, ::anti_alias_level])
            for i in range(anti_alias_level):
                for j in range(anti_alias_level):
                    if i == j == 0:
                        continue
                    frame_temp.add_(
                        frame_out[:, i::anti_alias_level, j::anti_alias_level]
                    )
            frame_temp.div_(anti_alias_level * anti_alias_level)
            aa_frame_out.copy_(frame_temp)
        frame_out = aa_frame_out

    if hdr:
        # The render/composite and background pre-fill carry byte-range values
        # (0-255, with HDR headroom above 255 preserved by the f16 buffer).
        # Bloom and the post-process tonemap expect linear 0-1, so normalise
        # the color + glow channels (0-3) in place -- once, in f16 -- leaving
        # any alpha channel (4) in byte range for the finalize step. Bloom then
        # runs on unclamped linear HDR and tonemapping is applied last. In
        # place (no extra frame-sized buffer): frame_out here is either the
        # arena downsample target or the soon-discarded render buffer.
        frame_out[..., :4].div_(255.0)

    if apply_fxaa:
        # Result kept in the frame's dtype (uint8 normally, f16 under HDR so
        # the normalised 0-1 values are not truncated) while the float FXAA
        # scratch is released.
        fxaa_out = self.get_tensor(frame_out.shape, frame_out.dtype)
        with self.temp():
            fxaa_input = self.cast(frame_out, torch.float32)
            fxaa_float = fxaa(fxaa_input.permute(0, 3, 1, 2), memory=self).permute(
                0, 2, 3, 1
            )
            fxaa_out.copy_(fxaa_float)
        frame_out = fxaa_out

    num_channels = frame_out.shape[-1]
    for process in post_processes:
        frame_out = process(frame_out, memory=self)

    frame_out = _finalize_on_device(
        frame_out,
        num_channels,
        self,
        tonemap_enabled=rt_settings.is_post_process_tonemap_enabled(),
        tonemapping=rt_settings.tonemapping,
        tonemap_method=rt_settings.tonemap_method,
        exposure=rt_settings.tonemap_exposure,
    )

    return _frames_to_host(frame_out)


def _frames_to_host(frame_out):
    """Hand finished frames back as host tensors, in top-down row order.

    This is the intentional ownership boundary: the video writer needs the data
    after the render arena is reset, so the device-to-host copy is deliberately
    not a render-device tensor allocation.

    The row flip happens before the transfer, not after: the composite writes
    frames bottom-up and the writer wants them top-down, and reversing the rows
    on the render device leaves the transfer itself one contiguous copy instead
    of a strided host-side re-read of everything that just arrived. Identical
    bytes either way -- a flip only reorders whole rows.
    """
    return frame_out.flip(-3).cpu()
