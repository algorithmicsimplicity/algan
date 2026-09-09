"""Batch-local UV mip pyramids, independent of the screen-space glossy cache.

The bank keeps the original (possibly packed/endpoint) level zero. Only reduced
levels are stored here, as linear-light, coverage-premultiplied vec5 rows for
colour and ordinary vec5 rows for data maps. Descriptor integers are bitcast,
not converted to floats: bank offsets must remain exact above 2**24 rows.
"""

from __future__ import annotations

import torch

from algan.utils.color_space import srgb_to_linear


def _area_axis(image: torch.Tensor, axis: int) -> torch.Tensor:
    """Halve one axis, preserving its integral, including odd/one-pixel sizes."""
    n = image.shape[axis]
    if n == 1:
        return image
    x = image.movedim(axis, 0)
    # Round down like a UV mip chain. Rounding up adds a spurious extra
    # level just above powers of two (257 -> ... -> 2 -> 1): log2(257)
    # would then fail to reach the mean even for a whole-image footprint.
    size = n // 2
    if n % 2 == 0:
        out = (x[0::2] + x[1::2]) * 0.5
    else:
        # Integrate a piecewise constant signal at equal-width bin boundaries.
        # Replicating the last texel, or adaptive pooling's overlapping bins,
        # would bias the DC level of non-power-of-two images.
        scale = n / size
        left = torch.arange(size, device=x.device, dtype=x.dtype) * scale
        right = left + scale
        first = left.long()
        out = torch.zeros_like(x[:size])
        # For n=2*k+1, each equal-width bin overlaps three source texels
        # (including the 3 -> 1 case). No full-image prefix sum or cancellation.
        for tap in range(3):
            idx = first + tap
            weight = (
                torch.minimum(right, idx + 1) - torch.maximum(left, idx)
            ).clamp_min(0)
            weight = weight.reshape((-1,) + (1,) * (x.ndim - 1))
            out += x.index_select(0, idx.clamp(max=n - 1)) * weight
        out *= 1 / scale
    return out.movedim(0, axis)


def _reduce(image: torch.Tensor) -> torch.Tensor:
    return _area_axis(_area_axis(image, -3), -2)


def build_mip_levels(
    texture: torch.Tensor,
    *,
    color: bool = False,
    linear: bool = False,
    lerp: torch.Tensor | None = None,
    wrap: tuple[bool, bool] = (False, False),
) -> list[torch.Tensor]:
    """Return reduced ``[T, W, H, 5]`` levels; never modify the authored tensor.

    Endpoint interpolation precedes the nonlinear colour decode and spatial
    filtering. Process one endpoint-blended frame at a time: no dense full-size
    time window is retained. Only the reduced frames survive into the bank.
    """
    # Surface's legacy level zero duplicates the first texel at each closed
    # edge. Exclude those duplicates from the integral, and use periodic
    # addressing for the reduced grids (the authored texels are untouched).
    if wrap[0]:
        texture = texture[..., :-1, :, :]
    if wrap[1]:
        texture = texture[..., :-1, :]
    w, h = texture.shape[-3:-1]
    if max(w, h) <= 1:
        return []

    def prepare(image):
        image = image.float()
        c = image.shape[-1]
        if c < 5:
            image = torch.cat((image, image.new_zeros((*image.shape[:-1], 5 - c))), -1)
        if color:
            rgb = srgb_to_linear(image[..., :3]) if linear else image[..., :3]
            return torch.cat(
                (
                    torch.cat((rgb, image[..., 3:4]), -1) * image[..., 4:5],
                    image[..., 4:5],
                ),
                -1,
            )
        return image

    if lerp is not None:
        endpoints = texture.reshape(-1, w, h, texture.shape[-1])
        frames = []
        for row in lerp.reshape(-1, 3).to(texture.device):
            # Tensor indexing stays on device; do not .item()/sync in prefetch.
            a = endpoints.index_select(0, row[0:1].long())
            b = endpoints.index_select(0, row[1:2].long())
            frames.append(_reduce(prepare(a + row[2] * (b - a))))
        level = torch.cat(frames, 0)
    else:
        level = _reduce(prepare(texture.reshape(-1, w, h, texture.shape[-1])))
    levels = [level]
    while max(level.shape[-3:-1]) > 1:
        level = _reduce(level)
        levels.append(level)
    return levels


def append_mip_pyramid(
    parts: list[torch.Tensor],
    offset: list[int],
    texture: torch.Tensor,
    *,
    color: bool,
    linear: bool,
    lerp: torch.Tensor | None,
    wrap: tuple[bool, bool] = (False, False),
    time_flat: bool = True,
) -> int:
    """Append reduced levels and an integer-bitcast directory; return its row."""
    levels = build_mip_levels(texture, color=color, linear=linear, lerp=lerp, wrap=wrap)
    if not levels:
        return -1
    flags = int(wrap[0]) | (int(wrap[1]) << 1)
    descriptors = [[flags, texture.shape[-3], texture.shape[-2], 1, len(levels)]]
    for level in levels:
        t, w, h, _ = level.shape
        # Match the base bank's time layout. Flattening T into rows while
        # retaining its leading axis would replicate coarse animation T times
        # when the legacy bank is assembled (quadratic memory growth).
        descriptors.append([offset[0], w, h, t if time_flat else 1, len(levels)])
        flat = level.reshape(1 if time_flat else t, -1, 5).contiguous()
        parts.append(flat)
        offset[0] += flat.shape[1]
    directory = torch.tensor(descriptors, dtype=torch.int32, device=texture.device)
    base = offset[0]
    parts.append(directory.view(torch.float32).unsqueeze(0))
    offset[0] += len(descriptors)
    return base
