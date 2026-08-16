from __future__ import annotations

import math

import torch
import torch.fft
import torch.nn.functional as F

from algan.environment import env_flag

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


def _should_bypass_bloom():
    try:
        from algan.rendering.raytracing import (
            is_ray_tracing_enabled,
            is_raytraced_glow_enabled,
        )

        return is_ray_tracing_enabled() and is_raytraced_glow_enabled()
    except ImportError:
        return False


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
        bloom_upsample_bilinear_f32(input_tensor, output)
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
            torch.lerp(
                horizontal[..., y0, :],
                horizontal[..., y1, :],
                source_y - y0,
                out=output[..., y, :],
            )


def fft_conv2d(input_tensor, kernel, padding="same", num_iterations=1):
    """
    Perform 2D convolution using FFT for better performance with large kernels.

    Args:
        input_tensor: Input tensor of shape (C, H, W)
        kernel: Convolution kernel of shape (C, Kh, Kw)
        padding: Padding mode ('same' or 'valid')

    Returns:
        Convolved tensor
    """
    C, H, W = input_tensor.shape
    C_k, Kh, Kw = kernel.shape

    assert C_k == C, "Number of channels must match between input and kernel"

    # Calculate output size and padding
    if padding == "same":
        Kh // 2
        Kw // 2
        out_h, out_w = H, W
    else:  # 'valid'
        out_h, out_w = H - Kh + 1, W - Kw + 1

    # Calculate FFT size (next power of 2 for efficiency)
    fft_h = 1 << (H + Kh - 1).bit_length()
    fft_w = 1 << (W + Kw - 1).bit_length()

    # Pad input and kernel to FFT size
    input_padded = F.pad(input_tensor, (0, fft_w - W, 0, fft_h - H))
    kernel_padded = F.pad(kernel, (0, fft_w - Kw, 0, fft_h - Kh))

    # Perform FFT
    input_fft = torch.fft.fft2(input_padded, dim=(-2, -1))
    kernel_fft = torch.fft.fft2(kernel_padded, dim=(-2, -1))

    # Element-wise multiplication in frequency domain
    result_fft = torch.mul(input_fft, kernel_fft, out=input_fft)

    # Inverse FFT
    result = torch.fft.ifft2(result_fft, dim=(-2, -1), out=result_fft).real

    # Extract the valid convolution result
    if padding == "same":
        # Center crop to original size
        pad_top = (Kh - 1) // 2
        pad_left = (Kw - 1) // 2
        result = result[:, pad_top : pad_top + H, pad_left : pad_left + W]
    else:  # 'valid'
        result = result[:, :out_h, :out_w]

    return result


def gaussian_kernel_2d(kernel_size, sigma, device):
    """
    Create a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel (odd number)
        sigma: Standard deviation of the Gaussian
        device: Device to create the kernel on

    Returns:
        2D Gaussian kernel tensor
    """
    # Create 1D Gaussian
    x = torch.linspace(
        -(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size, device=device
    )
    x /= sigma
    x.square_()
    x *= -0.5
    x = x.exp_()
    x /= x.sum()

    # Create 2D kernel by outer product
    kernel_2d = x[:, None] * x[None, :]

    return kernel_2d


# TODO fix up this code
def bloom_filter_old(
    x,
    blur_width=0.01 * 0.0005,
    num_iterations=3,
    kernel_size=31,
    strength=10,
    scale_factor=8,
):
    if _should_bypass_bloom():
        return x
    # def bloom_filter(x, blur_width=0.01*0.0005, num_iterations=3, kernel_size=11, strength=10, scale_factor=8):
    # kernel_size = int(kernel_size * x.shape[-3] / 2160)
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype
    x = x.to(torch.float) / 255
    x[..., -1] = (x[..., -1]) * strength
    xb = torch.cat((x[..., :-1].clamp(min=1 / 255) * x[..., -1:], x[..., -1:]), -1)

    # xb = torch.cat((x[...,:-1].clamp(min=1/255) * (1-x[...,-1:]).clamp_(min=0, max=1) + x[...,-1:].clamp(min=0, max=1) * torch.ones_like(x[...,:-1].clamp(min=1/255)), x[...,-1:]), -1)
    # xb = torch.cat((x[...,:-1].clamp(min=1/255), x[...,-1:]), -1)
    # d = kernel_size / (min(x.shape[0], x.shape[1])/scale_factor)
    # filter = torch.exp(-1*(torch.linspace(-d, d, kernel_size, device=x.device)**2) * 2 / blur_width)
    d = 1
    kernel_filter = torch.exp(
        -1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2)
    )
    kernel_filter /= kernel_filter.sum()
    kernel_filter *= 1
    # filter /= filter.amax()
    filter_horizontal = kernel_filter.view(1, 1, 1, kernel_size).expand(
        xb.shape[-1], -1, -1, -1
    )
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)
    # counter_horizontal = torch.ones_like(filter_horizontal)
    # counter_horizontal = counter_horizontal / counter_horizontal.numel()
    # counter_vertical = torch.ones_like(filter_vertical)
    # counter_vertical = counter_vertical / counter_vertical.numel()
    # count = x[...,-1:].expand(-1,-1,3)/255
    p = (kernel_size - 1) // 2
    xb = xb.permute(-1, 0, 1)
    orig_shape = xb.shape[-2:]
    xb = F.interpolate(
        xb.unsqueeze(0), scale_factor=1 / scale_factor, mode="bilinear"
    ).squeeze(0)
    dists = torch.stack(
        (
            torch.linspace(-1, 1, kernel_size, device=x.device)
            .view(-1, 1)
            .expand(-1, kernel_size),
            torch.linspace(-1, 1, kernel_size, device=x.device)
            .view(1, -1)
            .expand(kernel_size, -1),
        ),
        -1,
    )
    dists = dists.square().sum(-1, keepdim=True).unsqueeze(-1)

    k = 1  # kernel_size * kernel_size * 0.01
    # count = count.permute(-1,0,1)
    for _i in range(num_iterations):
        """xbu = F.unfold(xb.unsqueeze(0), (kernel_size, kernel_size), padding=(p, p)).squeeze(0)
        xbu = unsquish(unsquish(unsquish(xbu, 0, -xb.shape[0]), -1, xb.shape[-1]), 1, kernel_size)
        #a = torch.exp(-dists) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists*2) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / (xbu[-1:]).clamp(min=1e-5)) * (1 - (dists)).clamp(min=0,max=1) * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / (xbu[-1:]).clamp(min=1e-5)) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        #a = torch.exp(-dists / 2) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        a = torch.exp(-dists / 2) * ((1 - (dists)) > 0).float() * (xbu[-1:])#.clamp(min=1e-5))
        t = a.clamp(min=0, max=1)
        a[:,p,p] += k#kernel_size*kernel_size*0.3
        n = a.sum((1,2), keepdim=True).clamp(min=1e-5)
        a = a / n
        a[:,p,p] = 0
        q = (1.2*a.sum((1,2))).clamp(min=0, max=1)
        #xb = torch.cat((((xbu[:-1] * (1-t) + t * torch.ones_like(xbu[:-1])) * a).sum((1,2)), (n).sum((1,2))), 0)
        xb = torch.cat((((xbu[:-1]) * a).sum((1,2)), (n).sum((1,2))), 0)
        xb[:-1] = xb[:-1]*(1-q) + (q * torch.ones_like(xb[:-1]))
        continue"""
        xb = F.conv2d(xb, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        xb = F.conv2d(xb, filter_vertical, padding=(p, 0), groups=xb.shape[0])
        # count = F.conv2d(count, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        # count = F.conv2d(count, filter_vertical, padding=(p, 0), groups=xb.shape[0])
        # n2 = xb[-1:] + k
        # if (xb[-1:].amax() <= 1):
        #    break

    xb = F.interpolate(xb.unsqueeze(0), size=orig_shape, mode="bilinear").squeeze(0)
    xb = xb.permute(1, 2, 0)

    xb[..., -1:].clamp(min=0, max=1)
    (xb[..., -1:] * 0.5).clamp(min=0, max=1)
    ((xb[..., -1:] - x[..., -1:]) >= 0).float()
    torch.zeros_like((xb[..., -1:] - x[..., -1:]).clamp(min=0, max=1))
    # a5 = ((1/r)*((xb[...,-1:] +1).log() - r)).clamp(min=0, max=1)
    a5 = ((xb[..., -1:] + 1).log() / 3).clamp(min=0, max=1)
    # a5 = (((xb[...,-1:] / x[...,-1:].clamp(min=1e-5)))).clamp(min=0, max=1)
    # a5 = (1-((xb[...,-1:] - x[...,-1:]))).clamp(min=0, max=1)
    # a3 = (((xb[...,-1:] - x[...,-1:]) * 0.5) + 0.5).clamp(min=0,max=1)

    """xb[...,:-1] = xb[...,:-1] * (1-a2) + a2 * torch.ones_like(xb[...,:-1])
    xb = xb.permute(-1, 0, 1)
    xb = F.interpolate(xb.unsqueeze(0), scale_factor=1 / scale_factor, mode='bilinear').squeeze(0)
    for i in range(num_iterations):
        xb = F.conv2d(xb, filter_horizontal, padding=(0, p), groups=xb.shape[0])
        xb = F.conv2d(xb, filter_vertical, padding=(p, 0), groups=xb.shape[0])
    xb = F.interpolate(xb.unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze(0)
    xb = xb.permute(1,2,0)"""

    xb = (
        (xb[..., :-1] + x[..., :-1] * (x[..., -1:] + k))
        / (xb[..., -1:] + x[..., -1:] + k)
    ) * (1 - a5) + a5 * torch.ones_like(xb[..., :-1])
    # xb = (xb[...,:-1] + x[...,:-1] * (x[...,-1:]+k)) / (xb[...,-1:]+x[...,-1:]+k)# * m + (1-m) * (xb[...,-1:])# + a4 * torch.ones_like(xb[...,:-1]))
    # xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1] * (x[...,-1:]+k))# * m + (1-m) * (xb[...,-1:])# + a4 * torch.ones_like(xb[...,:-1]))
    # xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1]) * (1-a3) + a3 * torch.ones_like(xb[...,:-1])
    # xb = (xb[...,:-1] * (a) + (1-a) * x[...,:-1])

    # count = count.permute(1,2,0) + 1

    xb[..., -1:] + k
    # xb = (xb[...,:-1] + x[...,:-1] * k) / n3# / xb[...,-1:].clamp(min=1e-5)#*strength
    # xb = xb[...,:-1] + (x[...,:-1] * (k + x[...,-1:])) / xb[...,-1:].clamp(min=1e-5)
    # glow = (glow / glow.amax().clamp_(min=255)) * 255
    # glow = glow.clamp(max=255)
    (xb[..., -1:]).clamp(min=0, max=1)
    # glow = glow * (1-a) + a * (torch.ones_like(glow))# * 0.4 + glow * 0.6)
    # out = (x[...,:-1] * (1-a) + a * glow)# / (1+xb[...,-1:])
    (xb[..., -1:] * 1).clamp(min=0, max=1)
    out = xb  # + (x[..., :-1])# + glow)  # / (1+xb[...,-1:])
    # out = (x[..., :-1] + glow)  # / (1+xb[...,-1:])
    # out = (x[...,:-1] * (1-a) + (a2) * glow)# / (1+xb[...,-1:])
    # out = (x[...,:-1] *(1-a) + a * glow)# / (1+xb[...,-1:])
    return ((out * 255).clamp_(max=255)).to(xdtype)
    # return ((out / out.amax().clamp_(min=255)) * 255).to(xdtype)
    # return (x[...,:-1] + xb*strength).clamp_(max=255).to(xdtype)


def bloom_filter_premultiply(
    x, num_iterations=3, kernel_size=31, strength=10, scale_factor=8, memory=None
):
    if _should_bypass_bloom():
        return x
    if x.shape[-1] < 5:
        raise ValueError(
            "bloom_filter_premultiply only works for scenes with transparent backgrounds, please set"
            "background_color=TRANSPARENT when rendering."
        )
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype

    x = x.to(torch.float) / 255
    color = x[..., :3]
    glow = x[..., 3:4]

    color = color * glow * strength

    d = 3
    kernel_filter = torch.exp(
        -1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2)
    )
    kernel_filter /= kernel_filter.sum()
    filter_horizontal = kernel_filter.view(1, 1, 1, kernel_size).expand(
        color.shape[-1], -1, -1, -1
    )
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)

    p = (kernel_size - 1) // 2

    color = color.permute(-1, 0, 1)
    orig_shape = color.shape[-2:]
    color = F.interpolate(
        color.unsqueeze(0), scale_factor=1 / scale_factor, mode="bilinear"
    ).squeeze(0)

    for _i in range(num_iterations):
        color = F.conv2d(
            color, filter_horizontal, padding=(0, p), groups=color.shape[0]
        )
        color = F.conv2d(color, filter_vertical, padding=(p, 0), groups=color.shape[0])

    color = F.interpolate(color.unsqueeze(0), size=orig_shape, mode="bilinear").squeeze(
        0
    )
    color = color.permute(1, 2, 0)

    out = torch.cat((x[..., :3] * x[..., 4:5] + color, x[..., 4:5]), -1)
    return (out * 255).clamp_(min=0, max=255).to(xdtype)


def bloom_filter_conv(x, num_iterations=3, kernel_size=31, strength=10, scale_factor=8):
    if _should_bypass_bloom():
        return x
    # return x
    if x[..., 3:4].amax() <= 1e-5:
        return x
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype

    x = x.to(torch.float) / 255
    color_channels = [*range(3), 4] if x.shape[-1] == 5 else [*range(3)]
    color = x[..., color_channels]
    glow = x[..., 3:4]

    color = color * glow * strength

    d = 3
    kernel_filter = torch.exp(
        -1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2)
    )
    kernel_filter /= kernel_filter.sum()
    filter_horizontal = kernel_filter.view(1, 1, 1, kernel_size).expand(
        color.shape[-1], -1, -1, -1
    )
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)

    p = (kernel_size - 1) // 2

    color = color.permute(-1, 0, 1)
    orig_shape = color.shape[-2:]

    # Downsample for computational efficiency.
    color = F.interpolate(
        color.unsqueeze(0), scale_factor=1 / scale_factor, mode="bilinear"
    ).squeeze(0)

    # Apply the gaussian blur convolutional filter num_iteration times
    for _i in range(num_iterations):
        color = F.conv2d(
            color, filter_horizontal, padding=(0, p), groups=color.shape[0]
        )
    for _i in range(num_iterations):
        color = F.conv2d(color, filter_vertical, padding=(p, 0), groups=color.shape[0])

    color = F.interpolate(color.unsqueeze(0), size=orig_shape, mode="bilinear").squeeze(
        0
    )

    color = color.permute(1, 2, 0)

    out = x.clone()
    out[..., color_channels] += color
    # if x.shape[-1] == 5:
    #    out.clamp_(min=0, max=1)
    #    out[...,:3] /= out[...,-1:].clamp_min_(1e-3)
    # out = torch.cat((x[..., :3] * x[...,4:5] + color, x[...,4:5]), -1)
    return (out * 255).clamp_(min=0, max=255).to(xdtype)

    color = color[..., :-1]

    s = color.shape[-1] // 2

    color = color + torch.cat((x[..., :-1], 1 - x[..., :-1]), -1)

    # color = torch.maximum(color[...,:s], 1-color[...,s:])
    # inverse_color = color[...,s:]
    # Take average of both color and inverse color.
    color = color[..., :s]
    # m = color[...,:s].norm(p=2,dim=-1,keepdim=True) > color[...,s:].norm(p=2,dim=-1,keepdim=True)
    # color = (color[...,:s] * m + (~m) * (1-color[...,s:]))
    # color = (1-color[...,s:])
    # out = x[...,:-1] + color
    out = color
    # Interpolate original color and bloomed color based on
    # how much glow was accumulated.
    # w = 1/(1+glow)
    # out = x[...,:-1] * w + (1-w) * color

    return (out * 255).clamp_(min=0, max=255).to(xdtype)


def bloom_filter(
    x,
    num_iterations=1,
    kernel_size=256,
    strength=30,
    scale_factor=8,
    glow_spread=0.10,
    rim_frac=0.004,
    tail_weight=0.6,
    memory=None,
):
    """FFT-based bloom filter producing a soft, natural glow.

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

    Args:
        x: Input image tensor (..., H, W, C); channel 3 is the glow intensity.
        strength: Glow intensity multiplier (also sets how far the bright rim
            saturates beyond the source outline).
        scale_factor: Downsampling factor for efficiency (scaled by resolution).
        glow_spread: Sigma of the wide tail blur as a fraction of the
            (downsampled) frame height. Larger -> the faint glow reaches further.
        rim_frac: Sigma of the tight rim blur as a fraction of frame height.
        tail_weight: Weight of the wide tail relative to the rim (small -> the
            tail is much fainter than the rim, giving the sharp drop-off).

    Returns:
        Bloomed image tensor with same shape as input.
    """
    if _should_bypass_bloom():
        return x
    if memory is None:
        raise ValueError("bloom_filter requires a ManualMemory arena")
    # Avoid the expensive multi-scale blur for the overwhelmingly common
    # no-glow case.  The reduction output itself belongs to the arena; the
    # estimator still reserves the active worst-case because pixel values are
    # not available when batch sizes are chosen.
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

        channels = 4 if x.shape[-1] == 5 else 3
        if x.shape[-1] == 5:
            color_hwc = memory.get_tensor((*x.shape[:-1], channels), work_dtype)
            color_hwc[..., 0].copy_(x_work[..., 0])
            color_hwc[..., 1].copy_(x_work[..., 1])
            color_hwc[..., 2].copy_(x_work[..., 2])
            color_hwc[..., 3].copy_(x_work[..., 4])
        else:
            color_hwc = x_work[..., :3]
        glow = x_work[..., 3:4]
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

        if x.shape[-1] == 5:
            out[..., 0].add_(upsampled[..., 0])
            out[..., 1].add_(upsampled[..., 1])
            out[..., 2].add_(upsampled[..., 2])
            out[..., 4].add_(upsampled[..., 3])
        else:
            out[..., :3].add_(upsampled)

    return out
