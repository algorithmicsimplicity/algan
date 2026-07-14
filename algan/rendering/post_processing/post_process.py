from __future__ import annotations

import torch

from algan.rendering.post_processing.anti_aliasing.fxaa import fxaa
from algan.rendering.post_processing.memory_estimator import (
    PostProcessMemorySizer,
    get_post_process_memory_required,
)

__all__ = [
    "PostProcessMemorySizer",
    "get_post_process_memory_required",
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

    _linear_rgb(a, b, (
        (0.627409, 0.329282, 0.043309),
        (0.069055, 0.919540, 0.011405),
        (0.016390, 0.088013, 0.895597),
    ), scratch)
    _linear_rgb(b, a, (
        (0.856627153315983, 0.0951212405381588, 0.0482516061458583),
        (0.137318972929847, 0.761241990602591, 0.101439036467562),
        (0.11189821299995, 0.0767994186031903, 0.811302368396859),
    ), scratch)

    for channel in range(3):
        a[:, channel].clamp_(min=1e-10).log2_().clamp_(-12.47393, 4.026069)
        a[:, channel].sub_(-12.47393).div_(4.026069 - (-12.47393))

    x2 = memory.get_tensor((a.shape[0],), a.dtype)
    x4 = memory.get_tensor((a.shape[0],), a.dtype)
    for channel in range(3):
        _agx_curve(a[:, channel], b[:, channel], x2, x4, scratch)

    _linear_rgb(b, a, (
        (1.1271005818144368, -0.11060664309660323, -0.016493938717834573),
        (-0.1413297634984383, 1.157823702216272, -0.016493938717834257),
        (-0.14132976349843826, -0.11060664309660294, 1.2519364065950405),
    ), scratch)
    _linear_rgb(a, b, (
        (1.6605, -0.1246, -0.0182),
        (-0.5876, 1.1329, -0.1006),
        (-0.0728, -0.0083, 1.1187),
    ), scratch)
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


def _finalize_on_device(frame, original_num_channels, memory, *,
                        tonemap_enabled, tonemapping, tonemap_method,
                        exposure):
    """Strip render-only channels and return arena-owned uint8 frames."""
    # Fast path: the input is already byte output.  Only the transparent
    # five-channel layout needs a copied/reordered result.
    if not tonemap_enabled and frame.dtype == torch.uint8:
        return _strip_aux_channel(frame, original_num_channels, memory)

    stripped_channels = (
        4 if original_num_channels == frame.shape[-1] == 5
        else (frame.shape[-1] - 1
              if original_num_channels == frame.shape[-1] else frame.shape[-1])
    )
    output_channels = (
        (4 if stripped_channels == 4 else 3)
        if tonemap_enabled else stripped_channels
    )
    output = memory.get_tensor((*frame.shape[:-1], output_channels), torch.uint8)

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
                rgb_tonemapped = memory.clone(rgb)
                rgb_tonemapped.clamp_(0.0, 1.0)

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


def post_process_frames(self, frames, anti_alias_level, post_processes=(), apply_fxaa=False):
    self.pre_post_pointers = self.get_pointers()
    frame_out = frames
    if anti_alias_level > 1:
        aa_frame_out = self.get_tensor([
            frame_out.shape[0], frame_out.shape[1] // anti_alias_level,
            frame_out.shape[2] // anti_alias_level, frame_out.shape[3]
        ], dtype=torch.uint8)
        with self.temp():
            frame_temp = self.get_tensor(aa_frame_out.shape, torch.float32)
            frame_temp.copy_(frame_out[:, ::anti_alias_level, ::anti_alias_level])
            for i in range(anti_alias_level):
                for j in range(anti_alias_level):
                    if i == j == 0:
                        continue
                    frame_temp.add_(frame_out[:, i::anti_alias_level, j::anti_alias_level])
            frame_temp.div_(anti_alias_level * anti_alias_level)
            aa_frame_out.copy_(frame_temp)
        frame_out = aa_frame_out

    if apply_fxaa:
        # The byte result survives while all float FXAA tensors are released.
        fxaa_u8 = self.get_tensor(frame_out.shape, torch.uint8)
        with self.temp():
            fxaa_input = self.cast(frame_out, torch.float32)
            fxaa_float = fxaa(
                fxaa_input.permute(0, 3, 1, 2), memory=self
            ).permute(0, 2, 3, 1)
            fxaa_u8.copy_(fxaa_float)
        frame_out = fxaa_u8

    num_channels = frame_out.shape[-1]
    for process in post_processes:
        frame_out = process(frame_out, memory=self)

    from algan.rendering.raytracing import settings as rt_settings
    frame_out = _finalize_on_device(
        frame_out, num_channels, self,
        tonemap_enabled=rt_settings.is_post_process_tonemap_enabled(),
        tonemapping=rt_settings.TONEMAPPING,
        tonemap_method=rt_settings.TONEMAP_METHOD,
        exposure=rt_settings.TONEMAP_EXPOSURE,
    )

    # This is the intentional ownership boundary: the video writer needs data
    # after the render arena is reset, so the final device-to-host copy is not a
    # render-device tensor allocation.
    return frame_out.cpu().flip(-3)
