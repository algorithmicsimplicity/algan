"""Taichi kernels for procedural render backgrounds."""

import taichi as ti


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
):
    """Evaluate ``background_func(x, y, time)`` into the whole output batch.

    ``width`` and ``height`` describe the supersampled background. When the
    renderer uses in-kernel anti-aliasing, ``out`` is smaller by
    ``anti_alias_level`` in each dimension; average the procedural background
    over those subpixels without allocating a supersampled intermediate.
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
        byte_sum = ti.Vector.zero(ti.f32, 5)
        for sample_y, sample_x in ti.ndrange(sample_level, sample_level):
            x = ti.cast(column + sample_x, ti.f32) / ti.cast(width, ti.f32)
            y = ti.cast(row + sample_y, ti.f32) / ti.cast(height, ti.f32)
            color = background_func(x, y, time)
            for channel in ti.static(range(5)):
                if channel < out.shape[2]:
                    source_channel = ti.static(min(channel, color.n - 1))
                    value = ti.min(1.0, ti.max(0.0, color[source_channel]))
                    byte_sum[channel] += ti.floor(value * 255.0 + 0.5)

        num_samples = ti.cast(sample_level * sample_level, ti.f32)
        for channel in ti.static(range(5)):
            if channel < out.shape[2]:
                out[frame, pixel, channel] = ti.floor(
                    byte_sum[channel] / num_samples + 0.5
                )
