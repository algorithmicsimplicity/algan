from __future__ import annotations

import torch


def _where(condition, a, b, out):
    """``torch.where`` with an arena-owned output."""
    return torch.where(condition, a, b, out=out)


def rgb_to_luma(image, memory, out=None):
    """Convert RGB to luma without allocating outside ``memory``."""
    if out is None:
        out = memory.get_tensor((image.shape[0], 1, image.shape[2], image.shape[3]), image.dtype)
    product = memory.get_tensor(image[:, :3].shape, image.dtype)
    weights = memory.get_tensor((1, 3, 1, 1), image.dtype)
    weights[0, 0, 0, 0] = 0.299
    weights[0, 1, 0, 0] = 0.587
    weights[0, 2, 0, 0] = 0.114
    torch.mul(image[:, :3], weights, out=product)
    torch.sum(product, dim=1, keepdim=True, out=out)
    return out


def fxaa(images, edge_threshold=0.125, edge_threshold_min=0.0625,
         subpixel_quality=0.75, memory=None):
    """Apply FXAA using only tensors backed by ``ManualMemory``.

    ``images`` is ``[B, C, H, W]`` float data.  The returned tensor is a
    persistent arena allocation; every intermediate is released when this
    function returns.
    """
    if memory is None:
        raise ValueError("fxaa requires a ManualMemory arena")

    B, C, H, W = images.shape
    antialiased = memory.get_tensor(images.shape, images.dtype)

    # The three-channel product doubles as three full-resolution scratch
    # planes once luma has been padded.  Reusing it keeps FXAA's live arena
    # footprint bounded without changing the operation order of the filter.
    with memory.temp(clear_persist=True):
        product = memory.get_tensor((B, 3, H, W), images.dtype)
        luma = memory.get_tensor((B, 1, H, W), images.dtype)
        weights = memory.get_tensor((1, 3, 1, 1), images.dtype)
        weights[0, 0, 0, 0] = 0.299
        weights[0, 1, 0, 0] = 0.587
        weights[0, 2, 0, 0] = 0.114
        torch.mul(images[:, :3], weights, out=product)
        torch.sum(product, dim=1, keepdim=True, out=luma)

        luma_padded = memory.get_tensor((B, 1, H + 2, W + 2), images.dtype)
        torch.ops.aten.replication_pad2d.out(
            luma, [1, 1, 1, 1], out=luma_padded
        )

        luma_c = luma_padded[:, :, 1:H + 1, 1:W + 1]
        luma_n = luma_padded[:, :, 0:H, 1:W + 1]
        luma_s = luma_padded[:, :, 2:H + 2, 1:W + 1]
        luma_e = luma_padded[:, :, 1:H + 1, 2:W + 2]
        luma_w = luma_padded[:, :, 1:H + 1, 0:W]
        luma_nw = luma_padded[:, :, 0:H, 0:W]
        luma_ne = luma_padded[:, :, 0:H, 2:W + 2]
        luma_sw = luma_padded[:, :, 2:H + 2, 0:W]
        luma_se = luma_padded[:, :, 2:H + 2, 2:W + 2]

        s0 = product[:, 0:1]
        s1 = product[:, 1:2]
        s2 = product[:, 2:3]

        # Local minimum, maximum and contrast.  ``luma`` is no longer needed
        # after padding, so its storage becomes the contrast plane.
        torch.minimum(luma_n, luma_s, out=s0)
        torch.minimum(luma_e, luma_w, out=s1)
        torch.minimum(s0, s1, out=s0)
        torch.minimum(luma_c, s0, out=s0)
        torch.maximum(luma_n, luma_s, out=s1)
        torch.maximum(luma_e, luma_w, out=s2)
        torch.maximum(s1, s2, out=s1)
        torch.maximum(luma_c, s1, out=s1)
        torch.sub(s1, s0, out=luma)

        edge_mask = memory.get_tensor((B, 1, H, W), torch.bool)
        s0.fill_(edge_threshold_min)
        s1.mul_(edge_threshold)
        torch.maximum(s0, s1, out=s1)
        torch.gt(luma, s1, out=edge_mask)

        # Horizontal Sobel response.
        torch.mul(luma_nw, -1.0, out=s0)
        s0.add_(luma_ne)
        torch.mul(luma_w, -2.0, out=s1)
        s1.add_(luma_e, alpha=2.0)
        s0.add_(s1)
        torch.mul(luma_sw, -1.0, out=s1)
        s1.add_(luma_se)
        s0.add_(s1)
        s0.abs_()

        # Vertical Sobel response.
        torch.mul(luma_nw, -1.0, out=s1)
        s1.add_(luma_n, alpha=-2.0)
        s1.add_(luma_ne, alpha=-1.0)
        torch.mul(luma_sw, 1.0, out=s2)
        s2.add_(luma_s, alpha=2.0)
        s2.add_(luma_se)
        s1.add_(s2)
        s1.abs_()

        is_horizontal = memory.get_tensor((B, 1, H, W), torch.bool)
        torch.ge(s0, s1, out=is_horizontal)

        # Positive/negative edge gradients.  The luma selections can be
        # overwritten as soon as their difference has been taken.
        _where(is_horizontal, luma_s, luma_e, out=s0)
        _where(is_horizontal, luma_n, luma_w, out=s1)
        s0.sub_(luma_c).abs_()
        s1.sub_(luma_c).abs_()
        is_negative_dir = memory.get_tensor((B, 1, H, W), torch.bool)
        torch.ge(s1, s0, out=is_negative_dir)

        # Average luma, normalized subpixel offset and smoothstep.
        torch.add(luma_nw, luma_ne, out=s0)
        s0.add_(luma_sw).add_(luma_se).mul_(0.25).mul_(0.5)
        torch.add(luma_n, luma_s, out=s1)
        s1.add_(luma_e).add_(luma_w).mul_(0.25)
        s0.add_(s1)
        s0.sub_(luma_c).abs_().div_(luma).clamp_(0.0, 1.0)
        torch.mul(s0, s0, out=s1)
        torch.mul(s0, 2.0, out=s2)
        s2.neg_().add_(3.0)
        s1.mul_(s2).mul_(subpixel_quality)

        # Pixel step and final x/y offsets.  The tiny weight allocation is
        # reused for broadcast constants so no scalar device tensors escape
        # the arena.
        one = weights[:, 0:1]
        neg_one = weights[:, 1:2]
        zero = weights[:, 2:3]
        one.fill_(1.0)
        neg_one.fill_(-1.0)
        zero.zero_()
        _where(is_negative_dir, neg_one, one, out=s0)
        _where(is_horizontal, zero, s0, out=luma)
        luma.div_(W).mul_(s1)
        _where(is_horizontal, s0, zero, out=s2)
        s2.div_(H).mul_(s1)

        # Build the sampling grid directly in its arena destination.
        grid = memory.get_tensor((B, H, W, 2), images.dtype)
        grid_x = memory.get_tensor((W,), images.dtype)
        grid_y = memory.get_tensor((H,), images.dtype)
        torch.linspace(-1, 1, W, device=images.device, dtype=images.dtype, out=grid_x)
        torch.linspace(-1, 1, H, device=images.device, dtype=images.dtype, out=grid_y)
        grid[..., 0].copy_(grid_x.view(1, 1, W).expand(B, H, W))
        grid[..., 1].copy_(grid_y.view(1, H, 1).expand(B, H, W))

        # Match the original two multiply operations before each in-place grid
        # addition (important for byte parity).
        torch.mul(luma, 2.0, out=s0)
        torch.mul(s0, edge_mask, out=s0)
        grid[..., 0].add_(s0[:, 0])
        torch.mul(s2, 2.0, out=s0)
        torch.mul(s0, edge_mask, out=s0)
        grid[..., 1].add_(s0[:, 0])

        torch.ops.aten.grid_sampler_2d.out(
            images, grid, 0, 1, False, out=antialiased
        )

    return antialiased


def smoothstep(x):
    """Compatibility helper retained for callers outside the render path."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
