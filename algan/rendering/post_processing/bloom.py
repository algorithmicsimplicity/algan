import torch
import torch.nn.functional as F
import torch.fft


def fft_conv1d(input_tensor, kernel, dim=-1, padding="same", num_iterations=1, memory=None):
    """
    Perform 2D convolution using FFT for better performance with large kernels.

    Args:
        input_tensor: Input tensor of shape (C, H, W)
        kernel: Convolution kernel of shape (C, Kh, Kw)
        padding: Padding mode ('same' or 'valid')

    Returns:
        Convolved tensor
    """

    C = input_tensor.shape[-3]
    L = input_tensor.shape[dim]
    C_k = kernel.shape[-3]
    L_k = kernel.shape[dim]

    assert C == C_k, "Number of channels must match between input and kernel"

    # Calculate output size and padding
    if padding == "same":
        pad = L_k // 2
        out_l = L
    else:  # 'valid'
        pad_h = pad_w = 0
        out_l = L - L_k + 1

    # Calculate FFT size (next power of 2 for efficiency)
    fft_l = L + L_k - 1#1 << (L + L_k - 1).bit_length()

    fft_shape = [_ for _ in input_tensor.shape]
    fft_shape[dim] = (fft_l // 2 + 1)
    # Perform FFT
    input_fft = torch.fft.rfft(input_tensor, n=fft_l, dim=dim, out=memory.get_tensor(fft_shape, dtype=torch.complex64))
    kernel_fft = torch.fft.rfft(kernel, n=fft_l, dim=dim)

    # Element-wise multiplication in frequency domain
    result_fft = torch.mul(input_fft, kernel_fft, out=input_fft)

    # Inverse FFT
    fft_shape = [_ for _ in input_tensor.shape]
    fft_shape[dim] = fft_l
    result = torch.fft.irfft(result_fft, n=fft_l, dim=dim, out=memory.get_tensor(fft_shape, persist=True))#, out=result_fft)

    # Extract the valid convolution result
    if padding == "same":
        # Center crop to original size
        pad_left = (L_k - 1) // 2
        result = torch.torch.ops.aten.slice(result, dim, pad_left, pad_left + L, 1)
    else:  # 'valid'
        result = torch.torch.ops.aten.slice(result, dim, 0, out_l, 1)

    return result


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

    assert C == C_k, "Number of channels must match between input and kernel"

    # Calculate output size and padding
    if padding == "same":
        pad_h = Kh // 2
        pad_w = Kw // 2
        out_h, out_w = H, W
    else:  # 'valid'
        pad_h = pad_w = 0
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
    filter = torch.exp(-1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2))
    filter /= filter.sum()
    filter *= 1
    # filter /= filter.amax()
    filter_horizontal = filter.view(1, 1, 1, kernel_size).expand(
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
    for i in range(num_iterations):
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

    a = xb[..., -1:].clamp(min=0, max=1)
    a2 = (xb[..., -1:] * 0.5).clamp(min=0, max=1)
    m = ((xb[..., -1:] - x[..., -1:]) >= 0).float()
    a4 = torch.zeros_like((xb[..., -1:] - x[..., -1:]).clamp(min=0, max=1))
    r = 0.5
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

    n3 = xb[..., -1:] + k
    # xb = (xb[...,:-1] + x[...,:-1] * k) / n3# / xb[...,-1:].clamp(min=1e-5)#*strength
    # xb = xb[...,:-1] + (x[...,:-1] * (k + x[...,-1:])) / xb[...,-1:].clamp(min=1e-5)
    # glow = (glow / glow.amax().clamp_(min=255)) * 255
    # glow = glow.clamp(max=255)
    a = (xb[..., -1:]).clamp(min=0, max=1)
    # glow = glow * (1-a) + a * (torch.ones_like(glow))# * 0.4 + glow * 0.6)
    # out = (x[...,:-1] * (1-a) + a * glow)# / (1+xb[...,-1:])
    a2 = (xb[..., -1:] * 1).clamp(min=0, max=1)
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
    filter = torch.exp(-1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2))
    filter /= filter.sum()
    filter_horizontal = filter.view(1, 1, 1, kernel_size).expand(
        color.shape[-1], -1, -1, -1
    )
    filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)

    p = (kernel_size - 1) // 2

    color = color.permute(-1, 0, 1)
    orig_shape = color.shape[-2:]
    color = F.interpolate(
        color.unsqueeze(0), scale_factor=1 / scale_factor, mode="bilinear"
    ).squeeze(0)

    for i in range(num_iterations):
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
    #return x
    if x[...,3:4].amax() <= 1e-5:
        return x
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)

    xdtype = x.dtype

    x = x.to(torch.float) / 255
    color_channels = [*range(3), 4] if x.shape[-1] == 5 else [*range(3)]
    color = x[..., color_channels]
    glow = x[..., 3:4]

    color = color * glow * strength

    d = 3
    filter = torch.exp(-1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2))
    filter /= filter.sum()
    filter_horizontal = filter.view(1, 1, 1, kernel_size).expand(
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
    for i in range(num_iterations):
        color = F.conv2d(
            color, filter_horizontal, padding=(0, p), groups=color.shape[0]
        )
    for i in range(num_iterations):
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


def bloom_filter(x, num_iterations=1, kernel_size=256, strength=30, scale_factor=8, memory=None):
    """
    FFT-based bloom filter for better performance with large kernel sizes.

    Args:
        x: Input image tensor with shape (..., H, W, C) where C includes glow channel
        num_iterations: Number of blur iterations
        kernel_size: Size of the Gaussian blur kernel
        strength: Glow intensity multiplier
        scale_factor: Downsampling factor for efficiency

    Returns:
        Bloomed image tensor with same shape as input
    """
    if x[...,3:4].amax() <= 1e-5:
        return x
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)
    if (kernel_size % 2) == 0:
        kernel_size = kernel_size + 1

    xdtype = x.dtype

    x = x.to(torch.float)
    x /= 255
    out = memory.clone(x)
    color_channels = [*range(3), 4] if x.shape[-1] == 5 else slice(0,3)
    color = x[..., color_channels]
    glow = x[..., 3:4]

    color *= glow
    color *= strength

    # Create 2D Gaussian kernel
    #sigma = kernel_size / 4.0
    #kernel_2d = gaussian_kernel_2d(kernel_size, sigma, x.device)

    d = 3

    channels = color.shape[-1]
    def get_filters(kernel_size):
        filter = torch.exp(-1 * (torch.linspace(-d, d, kernel_size, device=x.device) ** 2))
        filter /= filter.sum()
        filter_horizontal = filter.view(1, 1, 1, kernel_size).expand(
            channels, -1, -1, -1
        )
        filter_vertical = filter_horizontal.squeeze(-2).unsqueeze(-1)
        return filter_horizontal, filter_vertical

    # Prepare for convolution: (C, H, W) format
    color = color.permute(0, -1, 1, 2)
    orig_shape = color.shape[-2:]

    # Downsample for computational efficiency
    color = F.interpolate(
        color, scale_factor=1 / scale_factor, mode="bilinear", antialias=True
    )

    # Expand kernel for each channel
    #kernel_expanded = kernel_2d.unsqueeze(0).expand(color.shape[0], -1, -1)

    # Apply FFT-based convolution num_iterations times
    #p = (kernel_size - 1) // 2
    #color = F.conv2d(color, kernel_expanded.unsqueeze(1), padding=(p, p), groups=color.shape[0])
    with memory.temp(clear_persist=True):
        _color = memory.clone(color)
        def convolve(color, filter_horizontal, filter_vertical):
            for i in range(num_iterations):
                with memory.temp():
                    color = fft_conv1d(color, filter_horizontal.squeeze(1), padding="same", num_iterations=1, dim=-1, memory=memory)
                with memory.temp():
                    color = fft_conv1d(color, filter_vertical.squeeze(1), padding="same", num_iterations=1, dim=-2, memory=memory)
            return color
        with memory.temp(clear_persist=True):
            color1 = convolve(memory.clone(color), *get_filters(kernel_size))
        color1 = memory.clone(color1)
        with memory.temp(clear_persist=True):
            color2 = convolve(memory.clone(color), *get_filters(max(kernel_size // 32, 1)))
            color1 = torch.lerp(color1, color2, 0.4, out=color1)
        with memory.temp(clear_persist=True):
            color3 = convolve(color, *get_filters(max(kernel_size // 128, 1)))
        color = torch.lerp(color1, color3, 0.25, out=color)

    # Upsample back to original resolution
    color = F.interpolate(color, size=orig_shape, mode="bilinear", antialias=True)

    # Convert back to (H, W, C) format
    color = color.permute(0, 2, 3, 1)

    # Combine with original image
    out[..., color_channels] += color
    out *= 255
    out.clamp_(min=0, max=255)

    return out.to(xdtype)
