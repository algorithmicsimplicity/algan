"""Workspace-free separable bloom kernels."""

import taichi as ti
import torch

from algan.rendering.taichi_runtime import init_taichi

init_taichi()


@ti.kernel
def bloom_conv1d_f32(
        input_tensor: ti.types.ndarray(dtype=ti.f32, ndim=4),
        kernel: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=4),
        dim: ti.i32):
    """Convolve NCHW data along height (2) or width (3), zero-padded."""
    radius = kernel.shape[0] // 2
    for batch, channel, y, x in output:
        value = 0.0
        if dim == 3:
            for tap in range(kernel.shape[0]):
                source_x = x + radius - tap
                if 0 <= source_x < input_tensor.shape[3]:
                    value += input_tensor[batch, channel, y, source_x] * kernel[tap]
        else:
            for tap in range(kernel.shape[0]):
                source_y = y + radius - tap
                if 0 <= source_y < input_tensor.shape[2]:
                    value += input_tensor[batch, channel, source_y, x] * kernel[tap]
        output[batch, channel, y, x] = value


@ti.kernel
def bloom_downsample_bilinear_aa_f32(
        input_hwc: ti.types.ndarray(dtype=ti.f32, ndim=4),
        output: ti.types.ndarray(dtype=ti.f32, ndim=4),
        source_scale: ti.f32):
    """PyTorch-compatible antialiased bilinear downsample, NHWC to NCHW."""
    scale_y = source_scale
    scale_x = source_scale
    for batch, channel, y, x in output:
        center_y = (y + 0.5) * scale_y - 0.5
        center_x = (x + 0.5) * scale_x - 0.5
        first_y = ti.max(0, ti.cast(ti.ceil(center_y - scale_y), ti.i32))
        last_y = ti.min(
            input_hwc.shape[1] - 1,
            ti.cast(ti.floor(center_y + scale_y), ti.i32),
        )
        first_x = ti.max(0, ti.cast(ti.ceil(center_x - scale_x), ti.i32))
        last_x = ti.min(
            input_hwc.shape[2] - 1,
            ti.cast(ti.floor(center_x + scale_x), ti.i32),
        )
        weight_sum_y = 0.0
        weight_sum_x = 0.0
        for source_y in range(first_y, last_y + 1):
            weight_sum_y += ti.max(
                0.0, 1.0 - ti.abs(source_y - center_y) / scale_y
            )
        for source_x in range(first_x, last_x + 1):
            weight_sum_x += ti.max(
                0.0, 1.0 - ti.abs(source_x - center_x) / scale_x
            )
        value = 0.0
        for source_y in range(first_y, last_y + 1):
            weight_y = ti.max(
                0.0, 1.0 - ti.abs(source_y - center_y) / scale_y
            ) / weight_sum_y
            for source_x in range(first_x, last_x + 1):
                weight_x = ti.max(
                    0.0, 1.0 - ti.abs(source_x - center_x) / scale_x
                ) / weight_sum_x
                value += input_hwc[
                    batch, source_y, source_x, channel
                ] * weight_y * weight_x
        output[batch, channel, y, x] = value


@ti.kernel
def bloom_upsample_bilinear_f32(
        input_tensor: ti.types.ndarray(dtype=ti.f32, ndim=4),
        output: ti.types.ndarray(dtype=ti.f32, ndim=4)):
    """align_corners=False bilinear upsample with border clamping."""
    scale_y = input_tensor.shape[2] / output.shape[2]
    scale_x = input_tensor.shape[3] / output.shape[3]
    for batch, channel, y, x in output:
        source_y = (y + 0.5) * scale_y - 0.5
        source_x = (x + 0.5) * scale_x - 0.5
        source_y = ti.min(ti.max(source_y, 0.0), input_tensor.shape[2] - 1.0)
        source_x = ti.min(ti.max(source_x, 0.0), input_tensor.shape[3] - 1.0)
        y0 = ti.cast(ti.floor(source_y), ti.i32)
        x0 = ti.cast(ti.floor(source_x), ti.i32)
        y1 = ti.min(y0 + 1, input_tensor.shape[2] - 1)
        x1 = ti.min(x0 + 1, input_tensor.shape[3] - 1)
        weight_y = source_y - y0
        weight_x = source_x - x0
        top = (
            input_tensor[batch, channel, y0, x0] * (1.0 - weight_x)
            + input_tensor[batch, channel, y0, x1] * weight_x
        )
        bottom = (
            input_tensor[batch, channel, y1, x0] * (1.0 - weight_x)
            + input_tensor[batch, channel, y1, x1] * weight_x
        )
        output[batch, channel, y, x] = (
            top * (1.0 - weight_y) + bottom * weight_y
        )


def can_use_bloom_taichi(device):
    """Whether the active Taichi backend can directly import this device."""
    device = torch.device(device)
    try:
        arch = ti.lang.impl.current_cfg().arch
    except Exception:
        return False
    if device.type == "cuda":
        return arch == ti.cuda
    if device.type == "cpu":
        return arch == ti.cpu
    return False
