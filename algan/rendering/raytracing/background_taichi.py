"""Taichi kernels for procedural render backgrounds."""

from algan.rendering.raytracing.color_space_taichi import srgb_to_linear_f
from algan.taichi_compat import ti


@ti.kernel
def fill_background_from_func(
    out: ti.types.ndarray(),
    background_func: ti.template(),
    width: ti.i32,
    height: ti.i32,
    anti_alias_level: ti.i32,
    first_frame: ti.i32,
    frame_offset: ti.i32,
    frames_per_second: ti.f32,
    decode: ti.template(),
):
    """Evaluate ``background_func(x, y, time)`` into the whole output batch.

    ``width`` and ``height`` describe the supersampled background. When the
    renderer uses in-kernel anti-aliasing, ``out`` is smaller by
    ``anti_alias_level`` in each dimension; average the procedural background
    over those subpixels without allocating a supersampled intermediate.

    ``decode`` is a compile-time flag: under the linear working space the
    colour channels are decoded and accumulated in linear light, and nothing is
    put on the byte grid (the destination is the float HDR buffer, and a
    quantized linear value is what crushes a gradient's darks). Off, the samples
    are quantized exactly as they are written out, which is what a
    display-referred byte buffer holds.

    A callback that returns fewer than five components is not padded with its
    own last one. Only a colour channel can stand in for a colour channel: glow
    from an RGB background is 0, not its blue -- filled with blue, bloom lit up
    every pixel the background covered -- and its opacity is opaque.
    """
    full_pixels = width * height
    for frame, pixel in ti.ndrange(out.shape[0], out.shape[1]):
        sample_level = 1
        row = pixel // width
        column = pixel - row * width
        if out.shape[1] != full_pixels:
            sample_level = anti_alias_level
            base_width = width // anti_alias_level
            row = (pixel // base_width) * anti_alias_level
            column = (
                pixel - (pixel // base_width) * base_width
            ) * anti_alias_level

        time = ti.cast(first_frame + frame_offset + frame, ti.f32)
        time /= frames_per_second
        channel_sum = ti.Vector.zero(ti.f32, 5)
        for sample_y, sample_x in ti.ndrange(sample_level, sample_level):
            x = ti.cast(column + sample_x, ti.f32) / ti.cast(width, ti.f32)
            y = ti.cast(row + sample_y, ti.f32) / ti.cast(height, ti.f32)
            color = background_func(x, y, time)
            for channel in ti.static(range(5)):
                if channel < out.shape[2]:
                    value = 0.0
                    if ti.static(channel < color.n):
                        value = ti.min(1.0, ti.max(0.0, color[channel]))
                    elif ti.static(channel < 3 or color.n >= 4):
                        # Below three channels the last one is the whole
                        # colour, so it is the colour of the channels not
                        # given; at four it is the alpha the source does carry.
                        value = ti.min(1.0, ti.max(0.0, color[color.n - 1]))
                    elif ti.static(channel == 4):
                        value = 1.0  # opaque; glow (3) stays at 0.0
                    if ti.static(decode and channel < 3):
                        value = srgb_to_linear_f(value)
                    if ti.static(decode):
                        channel_sum[channel] += value * 255.0
                    else:
                        channel_sum[channel] += ti.floor(value * 255.0 + 0.5)

        num_samples = ti.cast(sample_level * sample_level, ti.f32)
        for channel in ti.static(range(5)):
            if channel < out.shape[2]:
                if ti.static(decode):
                    out[frame, pixel, channel] = channel_sum[channel] / num_samples
                else:
                    out[frame, pixel, channel] = ti.floor(
                        channel_sum[channel] / num_samples + 0.5
                    )
