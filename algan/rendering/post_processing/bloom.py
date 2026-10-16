from __future__ import annotations

import math

import torch
import torch.fft

from algan.environment import env_flag
from algan.utils.memory_utils import ManualMemory

# Round the bloom blur's transform length up to a length cuFFT has a native
# factorization for, instead of transforming at exactly ``L + K - 1``. That
# exact length is whatever the frame size and Gaussian radius happen to
# produce, and it regularly lands on a large prime factor -- 778 = 2 * 389 for
# the wide blur's vertical pass at 486 rows -- where the backend falls back to
# Bluestein's algorithm and runs several times slower than the slightly longer
# smooth transform beside it. Measured at 1.32x over the four blur passes of a
# frame batch (486x864, 16 frames; the 778-point pass alone 56.3 -> 42.3 ms).
#
# The extra zero padding cannot change the linear convolution the crop below
# extracts -- the transform is already long enough to be wrap-free -- only the
# order the arithmetic happens in: against the same input the two transforms
# agree to 9e-07 relative, four parts per billion of an 8-bit output code.
#
# The pixel baselines still had to be regenerated for it, because the longer
# transform also changes the arena footprint of the blur scratch, which
# re-sizes render chunks; a re-windowed render moves silhouette pixels by more
# than the suites' tolerance whatever the arithmetic does (see
# settings.available_memory_override). ALGAN_BLOOM_FFT_SMOOTH=0 restores the
# exact-length transform for A/B against a pre-regeneration baseline.
_SMOOTH_FACTORS = (2, 3, 5, 7)


def _fft_length(exact):
    """Transform length to use for a convolution needing ``exact`` samples."""
    if not env_flag("ALGAN_BLOOM_FFT_SMOOTH", True):
        return exact

    def smooth(n):
        for factor in _SMOOTH_FACTORS:
            while n % factor == 0:
                n //= factor
        return n == 1

    length = exact
    while not smooth(length):
        length += 1
    return length


def fft_conv1d(
    input_tensor, kernel, dim=-1, padding="same", memory=None, out=None, scratch=None
):
    """Perform 1D convolution using FFT.

    Args:
        input_tensor: Input tensor of shape (..., L, ...)
        kernel: Convolution kernel of shape (Kh,) or same ndim as input_tensor
        dim: Dimension along which to convolve
        padding: Padding mode ('same' or 'valid')
        memory: ManualMemory arena (required)
        out: Pre-allocated output tensor (optional)
        scratch: Unused parameter to match direct_conv1d signature (optional)

    Returns:
        Convolved tensor
    """
    if memory is None:
        raise ValueError("fft_conv1d requires a ManualMemory arena")
    if padding != "same" and padding != "valid":
        raise ValueError("fft_conv1d supports 'same' or 'valid' padding only")

    dim = dim % input_tensor.ndim
    L = input_tensor.shape[dim]

    if kernel.ndim == 1:
        L_k = kernel.shape[0]
        # Reshape 1D kernel to be broadcastable along target dim
        kernel_reshaped_shape = [1] * input_tensor.ndim
        kernel_reshaped_shape[dim] = L_k
        kernel_for_fft = kernel.view(kernel_reshaped_shape)
    else:
        L_k = kernel.shape[dim]
        kernel_for_fft = kernel
        if kernel.ndim >= 3:
            C = input_tensor.shape[-3]
            C_k = kernel.shape[-3]
            assert C_k == C, "Number of channels must match between input and kernel"

    # Calculate output size
    out_l = L if padding == "same" else L - L_k + 1

    # Calculate FFT size (rounded up to a natively-factorable length; the
    # extra zero padding leaves the extracted convolution unchanged).
    fft_l = _fft_length(L + L_k - 1)

    fft_shape = list(input_tensor.shape)
    fft_shape[dim] = fft_l // 2 + 1

    result_shape = list(input_tensor.shape)
    result_shape[dim] = fft_l

    if out is None:
        out = memory.get_tensor(input_tensor.shape, input_tensor.dtype)

    with memory.temp():
        result = memory.get_tensor(result_shape, input_tensor.dtype)
        input_fft = memory.get_tensor(fft_shape, dtype=torch.complex64)
        kernel_fft_shape = list(kernel_for_fft.shape)
        kernel_fft_shape[dim] = fft_l // 2 + 1
        kernel_fft = memory.get_tensor(kernel_fft_shape, dtype=torch.complex64)

        torch.fft.rfft(input_tensor, n=fft_l, dim=dim, out=input_fft)
        torch.fft.rfft(kernel_for_fft, n=fft_l, dim=dim, out=kernel_fft)

        # Element-wise multiplication in frequency domain and inverse FFT.
        torch.mul(input_fft, kernel_fft, out=input_fft)
        torch.fft.irfft(input_fft, n=fft_l, dim=dim, out=result)

        # Extract the valid convolution result
        if padding == "same":
            # Center crop to original size
            pad_left = (L_k - 1) // 2
            valid_slice = torch.ops.aten.slice(result, dim, pad_left, pad_left + L, 1)
        else:  # 'valid'
            valid_slice = torch.ops.aten.slice(result, dim, 0, out_l, 1)

        out.copy_(valid_slice)

    return out


def direct_conv1d(
    input_tensor, kernel, dim=-1, padding="same", memory=None, out=None, scratch=None
):
    """Zero-padded 1D convolution with no backend workspace allocation.

    The default bloom path used to call cuFFT.  Even with an arena-owned
    ``out=`` tensor, cuFFT creates a device-dependent temporary work area
    outside :class:`ManualMemory`.  This direct separable implementation uses
    only pointwise operations with explicit arena outputs.  ``kernel`` is a
    one-dimensional, symmetric Gaussian and ``dim`` is one of the two image
    axes.
    """
    if memory is None:
        raise ValueError("direct_conv1d requires a ManualMemory arena")
    if padding != "same":
        raise ValueError("direct_conv1d currently supports padding='same' only")
    dim %= input_tensor.ndim
    if dim not in (input_tensor.ndim - 2, input_tensor.ndim - 1):
        raise ValueError("direct_conv1d only supports the image height/width axes")
    if kernel.ndim != 1:
        raise ValueError("direct_conv1d expects a one-dimensional kernel")

    if out is None:
        out = memory.get_tensor(input_tensor.shape, input_tensor.dtype)

    from algan.rendering.post_processing.bloom_kernels_taichi import (
        bloom_conv1d_f32,
        can_use_bloom_taichi,
    )

    if (
        input_tensor.dtype == torch.float32
        and kernel.dtype == torch.float32
        and out.dtype == torch.float32
        and can_use_bloom_taichi(input_tensor.device)
    ):
        bloom_conv1d_f32(input_tensor, kernel, out, dim)
        return out

    out.zero_()

    owns_scratch = scratch is None
    scratch_context = memory.temp() if owns_scratch else None
    if scratch_context is not None:
        scratch_context.__enter__()
        scratch = memory.get_tensor(input_tensor.shape, input_tensor.dtype)
    try:
        length = input_tensor.shape[dim]
        radius = kernel.shape[0] // 2
        for tap in range(kernel.shape[0]):
            delta = radius - tap
            dst_start = max(0, -delta)
            dst_end = min(length, length - delta)
            if dst_start >= dst_end:
                continue
            src_start = dst_start + delta
            src_end = dst_end + delta
            dst_index = [slice(None)] * input_tensor.ndim
            src_index = [slice(None)] * input_tensor.ndim
            dst_index[dim] = slice(dst_start, dst_end)
            src_index[dim] = slice(src_start, src_end)
            dst_index = tuple(dst_index)
            src_index = tuple(src_index)
            torch.mul(input_tensor[src_index], kernel[tap], out=scratch[dst_index])
            out[dst_index].add_(scratch[dst_index])
    finally:
        if scratch_context is not None:
            scratch_context.__exit__(None, None, None)
    return out


def _fill_gaussian_filter(filter_1d, radius, sigma):
    """Fill an arena filter without a render-device reduction workspace."""
    values = [math.exp(-0.5 * ((x / sigma) ** 2)) for x in range(-radius, radius + 1)]
    total = sum(values)
    values = [value / total for value in values]
    if filter_1d.device.type == "cpu":
        # Scalar stores do not create a separate CPU tensor/storage.
        for index, value in enumerate(values):
            filter_1d[index] = value
    else:
        # The temporary is host-owned; only the destination storage lives on
        # the rendering device, and that destination belongs to ManualMemory.
        host_filter = torch.tensor(
            values, dtype=filter_1d.dtype, device=filter_1d.device
        )
        filter_1d.copy_(host_filter)


def _axis_weights(input_size, output_size, output_index, antialias, source_scale=None):
    scale = input_size / output_size if source_scale is None else source_scale
    center = (output_index + 0.5) * scale - 0.5
    support = scale if antialias and scale > 1.0 else 1.0
    first = max(0, int(math.ceil(center - support)))
    last = min(input_size - 1, int(math.floor(center + support)))
    weights = [
        max(0.0, 1.0 - abs(source - center) / support)
        for source in range(first, last + 1)
    ]
    total = sum(weights)
    return first, [weight / total for weight in weights]


def _downsample_bloom(input_hwc, output, memory, scale_factor):
    if (
        input_hwc.shape[-3] == output.shape[-2]
        and input_hwc.shape[-2] == output.shape[-1]
    ):
        output.copy_(input_hwc.permute(0, 3, 1, 2))
        return
    from algan.rendering.post_processing.bloom_kernels_taichi import (
        bloom_downsample_bilinear_aa_f32,
        can_use_bloom_taichi,
    )

    if input_hwc.dtype == output.dtype == torch.float32 and can_use_bloom_taichi(
        input_hwc.device
    ):
        if input_hwc.is_contiguous():
            bloom_downsample_bilinear_aa_f32(input_hwc, output, scale_factor)
        else:
            # Taichi's ndarray interop rejects the RGB view of an RGBA input.
            # Pack that view into temporary arena storage instead of calling
            # ``contiguous()``, whose storage PyTorch would own externally.
            with memory.temp():
                packed_input = memory.get_tensor(input_hwc.shape, input_hwc.dtype)
                packed_input.copy_(input_hwc)
                bloom_downsample_bilinear_aa_f32(packed_input, output, scale_factor)
        return

    # Separable CPU fallback with the same widened triangular filter.
    with memory.temp():
        horizontal = memory.get_tensor(
            (
                input_hwc.shape[0],
                input_hwc.shape[-1],
                input_hwc.shape[-3],
                output.shape[-1],
            ),
            input_hwc.dtype,
        )
        for x in range(output.shape[-1]):
            first, weights = _axis_weights(
                input_hwc.shape[-2], output.shape[-1], x, True, scale_factor
            )
            horizontal[..., x].zero_()
            for offset, weight in enumerate(weights):
                horizontal[..., x].add_(
                    input_hwc[:, :, first + offset, :].permute(0, 2, 1),
                    alpha=weight,
                )
        for y in range(output.shape[-2]):
            first, weights = _axis_weights(
                input_hwc.shape[-3], output.shape[-2], y, True, scale_factor
            )
            output[..., y, :].zero_()
            for offset, weight in enumerate(weights):
                output[..., y, :].add_(horizontal[..., first + offset, :], alpha=weight)


def _upsample_bloom(input_tensor, output, memory):
    if input_tensor.shape[-2:] == output.shape[-2:]:
        output.copy_(input_tensor)
        return
    from algan.rendering.post_processing.bloom_kernels_taichi import (
        bloom_upsample_bilinear_f32,
        can_use_bloom_taichi,
    )

    if input_tensor.dtype == output.dtype == torch.float32 and can_use_bloom_taichi(
        input_tensor.device
    ):
        bloom_upsample_bilinear_f32(
            input_tensor,
            output,
            input_tensor.shape[-2] / output.shape[-2],
            input_tensor.shape[-1] / output.shape[-1],
        )
        return

    # CPU/mismatched-backend fallback: separable lerp with one arena scratch.
    with memory.temp():
        horizontal = memory.get_tensor(
            (*input_tensor.shape[:-1], output.shape[-1]), input_tensor.dtype
        )
        scale_x = input_tensor.shape[-1] / output.shape[-1]
        for x in range(output.shape[-1]):
            source_x = max(
                0.0, min((x + 0.5) * scale_x - 0.5, input_tensor.shape[-1] - 1.0)
            )
            x0 = int(math.floor(source_x))
            x1 = min(x0 + 1, input_tensor.shape[-1] - 1)
            # Torch 2.7.1 MPS lerp(weight=0, out=...) mistakes distinct
            # views of one arena for an in-place no-op. Border samples must
            # be copied explicitly or retain unrelated scratch contents.
            if input_tensor.device.type == "mps" and source_x == x0:
                horizontal[..., x].copy_(input_tensor[..., x0])
                continue
            torch.lerp(
                input_tensor[..., x0],
                input_tensor[..., x1],
                source_x - x0,
                out=horizontal[..., x],
            )
        scale_y = input_tensor.shape[-2] / output.shape[-2]
        for y in range(output.shape[-2]):
            source_y = max(
                0.0, min((y + 0.5) * scale_y - 0.5, input_tensor.shape[-2] - 1.0)
            )
            y0 = int(math.floor(source_y))
            y1 = min(y0 + 1, input_tensor.shape[-2] - 1)
            if input_tensor.device.type == "mps" and source_y == y0:
                output[..., y, :].copy_(horizontal[..., y0, :])
                continue
            torch.lerp(
                horizontal[..., y0, :],
                horizontal[..., y1, :],
                source_y - y0,
                out=output[..., y, :],
            )


def bloom_filter(
    x: torch.Tensor,
    num_iterations: int = 1,
    kernel_size: int = 256,
    strength: float = 30,
    scale_factor: float = 8,
    glow_spread: float = 0.10,
    rim_frac: float = 0.004,
    tail_weight: float = 0.6,
    memory: ManualMemory | None = None,
    *,
    premultiplied_over: bool = False,
) -> torch.Tensor:
    """Add a soft halo around glowing parts of the rendered Scene.

    A single Gaussian (``exp(-r^2)`` tail) plummets and leaves a hard,
    shell-like halo border. Instead the glow source is blurred at two scales
    and the (area-normalized) results are summed:

    * a tight *rim* (``rim_frac``) that, being area-normalized, dominates the
      peak and saturates to a thin bright halo hugging the source outline, and
    * a faint, wide *tail* (``glow_spread``) at low ``tail_weight`` that decays
      gently far from the source.

    The result is the natural look of real glow -- a sharp bright outline, a
    sudden drop to a much fainter level, then a long gradual falloff -- rather
    than a uniform blurred disk with a hard edge.

    Animation
    ---------
    Applied at render time to finished frames, not recorded as an animation.
    Pass a tuned version through ``Scene.save_video(post_processes=...)``;
    mobs must be spawned to contribute to the image.

    Parameters
    ----------
    x
        Frame batch, shape ``(N, H, W, C)``: RGB, glow, and optional alpha.
        Float input carries linear RGB/glow in 0-1 with HDR headroom, and alpha
        in 0-255. Byte input carries all channels in 0-255.
    num_iterations
        Blur iterations at each scale. Defaults to 1.
    kernel_size
        Retained parameter; the Gaussian sizes are computed from the frame
        height and spread settings. Defaults to 256 and has no effect.
    strength
        Dimensionless multiplier for bloom brightness. Defaults to 30.
    scale_factor
        Downsampling factor at a frame height of 2160 pixels, scaled with
        resolution and clamped to at least 1. Defaults to 8.
    glow_spread
        Wide Gaussian sigma as a fraction of downsampled frame height.
        Defaults to 0.10.
    rim_frac
        Tight Gaussian sigma as a fraction of downsampled frame height.
        Defaults to 0.004; sigma is at least one downsampled pixel.
    tail_weight
        Wide halo brightness relative to the tight rim. Defaults to 0.6.
    memory
        Working storage supplied by the renderer when this is a post-process.
        Defaults to None; direct calls must supply a ManualMemory arena.
    premultiplied_over
        Whether bloom adds only to RGB, keeping coverage alpha unchanged.
        Defaults to False. The Scene export mode supplies True automatically
        for this pass and tuned partials of it.

    Returns
    -------
    torch.Tensor
        Bloomed frames with the input shape. Byte input is converted to
        normalized float32; float input retains its dtype and channel scales.
        If bloom is bypassed or there is no glow, returns the input directly.

    Raises
    ------
    ValueError
        If working storage is not supplied and bloom is not bypassed.

    Examples
    --------
    Reduce the halo brightness:

    .. algan:: Example1BloomFilter

        from algan import *
        from functools import partial
        from algan.rendering.post_processing.bloom import bloom_filter

        Circle(glow=0.6).spawn()
        Scene.save_video(post_processes=(partial(bloom_filter, strength=12),))
    """
    if memory is None:
        raise ValueError("bloom_filter requires a ManualMemory arena")
    with memory.temp():
        glow_max = memory.get_tensor((), x.dtype)
        torch.amax(x[..., 3:4], out=glow_max)
        if glow_max.item() <= 1e-5:
            return x
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype
    work_dtype = torch.float32 if xdtype == torch.uint8 else xdtype
    out = memory.get_tensor(x.shape, work_dtype)
    out.copy_(x)
    if xdtype == torch.uint8:
        out.div_(255)

    with memory.temp(clear_persist=True):
        if xdtype == torch.uint8:
            x_work = memory.cast(x, torch.float32)
            x_work.div_(255)
        else:
            x_work = x

        alpha_bloom = x.shape[-1] == 5 and not premultiplied_over
        channels = 4 if alpha_bloom else 3
        if alpha_bloom:
            color_hwc = memory.get_tensor((*x.shape[:-1], channels), work_dtype)
            color_hwc[..., 0].copy_(x_work[..., 0])
            color_hwc[..., 1].copy_(x_work[..., 1])
            color_hwc[..., 2].copy_(x_work[..., 2])
            color_hwc[..., 3].copy_(x_work[..., 4])
        elif premultiplied_over:
            color_hwc = memory.clone(x_work[..., :3])
        else:
            color_hwc = x_work[..., :3]
        glow = (
            memory.clone(x_work[..., 3:4]) if premultiplied_over else x_work[..., 3:4]
        )
        glow.pow_(3)
        color_hwc.mul_(glow).mul_(strength)

        orig_shape = color_hwc.shape[-3:-1]
        out_h = int(orig_shape[0] / scale_factor)
        out_w = int(orig_shape[1] / scale_factor)
        color = memory.get_tensor(
            (color_hwc.shape[0], channels, out_h, out_w), work_dtype
        )
        _downsample_bloom(color_hwc, color, memory, scale_factor)

        height = color.shape[-2]
        sigma_rim = max(rim_frac * height, 1.0)
        sigma_tail = max(glow_spread * height, sigma_rim * 1.5)
        components = ((sigma_rim, 1.0), (sigma_tail, tail_weight))

        acc = memory.get_tensor(color.shape, work_dtype)
        acc.zero_()

        for sigma, weight in components:
            with memory.temp(clear_persist=True):
                radius = max(1, int(math.ceil(3.0 * sigma)))
                filter_size = 2 * radius + 1
                filter_1d = memory.get_tensor((filter_size,), work_dtype)
                _fill_gaussian_filter(filter_1d, radius, sigma)
                blurred = color
                if num_iterations > 0:
                    horizontal = memory.get_tensor(color.shape, work_dtype)
                    vertical = memory.get_tensor(color.shape, work_dtype)
                    for _ in range(num_iterations):
                        fft_conv1d(
                            blurred,
                            filter_1d,
                            padding="same",
                            dim=-1,
                            memory=memory,
                            out=horizontal,
                        )
                        fft_conv1d(
                            horizontal,
                            filter_1d,
                            padding="same",
                            dim=-2,
                            memory=memory,
                            out=vertical,
                        )
                        blurred = vertical
                acc.add_(blurred, alpha=weight)

        upsampled = memory.get_tensor(
            (color.shape[0], channels, *orig_shape), work_dtype
        )
        _upsample_bloom(acc, upsampled, memory)
        upsampled = upsampled.permute(0, 2, 3, 1)

        if alpha_bloom:
            out[..., 0].add_(upsampled[..., 0])
            out[..., 1].add_(upsampled[..., 1])
            out[..., 2].add_(upsampled[..., 2])
            out[..., 4].add_(upsampled[..., 3])
        else:
            out[..., :3].add_(upsampled)

    return out
