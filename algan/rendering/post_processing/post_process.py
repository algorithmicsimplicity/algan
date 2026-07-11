import torch

from algan.rendering.post_processing.anti_aliasing.fxaa import fxaa


def post_process_frames(self, frames, anti_alias_level, post_processes=(), apply_fxaa=False):
    self.pre_post_pointers = self.get_pointers()
    frame_out = frames
    if anti_alias_level > 1:
        aa_frame_out = self.get_tensor([frame_out.shape[0],
                                        frame_out.shape[1] // anti_alias_level, frame_out.shape[2] // anti_alias_level,
                                        frame_out.shape[3]], dtype=torch.uint8)
        with self.temp():
            frame_temp = self.get_tensor([frame_out.shape[0],
                                          frame_out.shape[1] // anti_alias_level,
                                          frame_out.shape[2] // anti_alias_level,
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
    if apply_fxaa:
        frame_out = (fxaa(frame_out.float().permute(0, -1, 1, 2)).permute(0, 2, 3, 1)).to(torch.uint8)
    num_channels = frame_out.shape[-1]
    for p in post_processes:
        frame_out = p(frame_out, memory=self)

    if num_channels == frame_out.shape[-1]:
        if num_channels == 5:
            frame_out = frame_out[..., [*range(num_channels - 2), -1]]
        else:
            frame_out = frame_out[..., :-1]

    # Check if the post process tonemap flag is set.
    # The tonemap settings are mutable module globals (set_tonemap_* setters);
    # read them live from the settings module -- importing them by value
    # (especially through primitives, which star-imports settings) freezes
    # them at import time.
    from algan.rendering.raytracing.settings import is_post_process_tonemap_enabled
    from algan.rendering.raytracing import settings as rt_settings
    TONEMAP_EXPOSURE = rt_settings.TONEMAP_EXPOSURE
    TONEMAP_METHOD = rt_settings.TONEMAP_METHOD
    TONEMAPPING = rt_settings.TONEMAPPING
    if is_post_process_tonemap_enabled():
        rgb = frame_out[..., :3]
        if frame_out.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        if TONEMAPPING:
            color_exposed = rgb * TONEMAP_EXPOSURE
            if TONEMAP_METHOD == "neutral":
                # Khronos PBR Neutral
                orig_shape = color_exposed.shape
                color_exposed_flat = color_exposed.reshape(-1, 3)

                x, _ = torch.min(color_exposed_flat, dim=1, keepdim=True)
                offset = torch.where(x < 0.08, x - 6.25 * x * x, torch.tensor(0.04, device=x.device))
                color_offset = color_exposed_flat - offset

                peak, _ = torch.max(color_offset, dim=1, keepdim=True)
                mask_compress = (peak >= 0.76).squeeze(1)

                if mask_compress.any():
                    color_c = color_offset[mask_compress]
                    peak_c = peak[mask_compress]

                    d = 0.24
                    newPeak = 1.0 - d * d / (peak_c + d - 0.76)
                    color_c *= newPeak / peak_c

                    g = 1.0 - 1.0 / (0.15 * (peak_c - newPeak) + 1.0)
                    color_offset[mask_compress] = color_c + g * (newPeak - color_c)

                rgb_tonemapped = torch.clamp(color_offset, 0.0, 1.0).reshape(orig_shape)
            elif TONEMAP_METHOD == "agx":
                # AgX
                orig_shape = color_exposed.shape
                color_exposed_flat = color_exposed.reshape(-1, 3)

                r_rec2020 = 0.627409 * color_exposed_flat[:, 0] + 0.329282 * color_exposed_flat[:,
                                                                             1] + 0.043309 * color_exposed_flat[:, 2]
                g_rec2020 = 0.069055 * color_exposed_flat[:, 0] + 0.919540 * color_exposed_flat[:,
                                                                             1] + 0.011405 * color_exposed_flat[:, 2]
                b_rec2020 = 0.016390 * color_exposed_flat[:, 0] + 0.088013 * color_exposed_flat[:,
                                                                             1] + 0.895597 * color_exposed_flat[:, 2]

                r_inset = 0.856627153315983 * r_rec2020 + 0.0951212405381588 * g_rec2020 + 0.0482516061458583 * b_rec2020
                g_inset = 0.137318972929847 * r_rec2020 + 0.761241990602591 * g_rec2020 + 0.101439036467562 * b_rec2020
                b_inset = 0.11189821299995 * r_rec2020 + 0.0767994186031903 * g_rec2020 + 0.811302368396859 * b_rec2020

                r_log = torch.clamp(torch.log2(torch.clamp(r_inset, min=1e-10)), -12.47393, 4.026069)
                g_log = torch.clamp(torch.log2(torch.clamp(g_inset, min=1e-10)), -12.47393, 4.026069)
                b_log = torch.clamp(torch.log2(torch.clamp(b_inset, min=1e-10)), -12.47393, 4.026069)

                r_norm = (r_log - (-12.47393)) / (4.026069 - (-12.47393))
                g_norm = (g_log - (-12.47393)) / (4.026069 - (-12.47393))
                b_norm = (b_log - (-12.47393)) / (4.026069 - (-12.47393))

                def agx_curve(x):
                    x2 = x * x
                    x4 = x2 * x2
                    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232

                r_curve = agx_curve(r_norm)
                g_curve = agx_curve(g_norm)
                b_curve = agx_curve(b_norm)

                r_out = 1.1271005818144368 * r_curve - 0.11060664309660323 * g_curve - 0.016493938717834573 * b_curve
                g_out = -0.1413297634984383 * r_curve + 1.157823702216272 * g_curve - 0.016493938717834257 * b_curve
                b_out = -0.14132976349843826 * r_curve - 0.11060664309660294 * g_curve + 1.2519364065950405 * b_curve

                r_srgb = 1.6605 * r_out - 0.1246 * g_out - 0.0182 * b_out
                g_srgb = -0.5876 * r_out + 1.1329 * g_out - 0.1006 * b_out
                b_srgb = -0.0728 * r_out - 0.0083 * g_out + 1.1187 * b_out

                rgb_tonemapped_flat = torch.empty_like(color_exposed_flat)
                rgb_tonemapped_flat[:, 0] = torch.clamp(r_srgb, 0.0, 1.0)
                rgb_tonemapped_flat[:, 1] = torch.clamp(g_srgb, 0.0, 1.0)
                rgb_tonemapped_flat[:, 2] = torch.clamp(b_srgb, 0.0, 1.0)
                rgb_tonemapped = rgb_tonemapped_flat.reshape(orig_shape)
            else:
                raise ValueError(f"Unknown tonemapping method: {TONEMAP_METHOD}")
        else:
            rgb_tonemapped = torch.clamp(rgb, 0.0, 1.0)

        # Convert RGB back to uint8 [0, 255]
        rgb_uint8 = (rgb_tonemapped * 255.0 + 0.5).clamp(0.0, 255.0).to(torch.uint8)

        if frame_out.shape[-1] == 4:
            alpha = frame_out[..., 3:]
            if alpha.dtype != torch.uint8:
                alpha = alpha.clamp(0.0, 255.0).to(torch.uint8)
            frame_out = torch.cat([rgb_uint8, alpha], dim=-1)
        else:
            frame_out = rgb_uint8
    else:
        if frame_out.dtype != torch.uint8:
            frame_out = (frame_out * 255).clamp_max_(255).to(torch.uint8)

    frame_out = frame_out.cpu().flip(-3)
    return frame_out