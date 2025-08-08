import torch
import torch.nn.functional as F


def rgb_to_luma(image):
    """Convert RGB image to luminance (grayscale)."""
    # Standard luminance weights for RGB
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def fxaa(images, edge_threshold=0.125, edge_threshold_min=0.0625, subpixel_quality=0.75):
    """
    Fast Approximate Anti-aliasing (FXAA) implementation in PyTorch.

    Args:
        images: Input tensor of shape (B, C, H, W) with values in range [0, 1]
        edge_threshold: Minimum local contrast required to apply AA (0.125 is good default)
        edge_threshold_min: Minimum edge threshold for darker areas (0.0625 is good default)
        subpixel_quality: Controls subpixel anti-aliasing quality (0.75 is good default)

    Returns:
        Anti-aliased images tensor of shape (B, C, H, W)
    """
    B, C, H, W = images.shape
    device = images.device

    # Convert to luminance for edge detection
    luma = rgb_to_luma(images[:,:3])

    # Create padded version for neighbor sampling
    luma_padded = F.pad(luma, (1, 1, 1, 1), mode='replicate')

    # Sample neighboring pixels (3x3 kernel)
    # Center pixel
    luma_c = luma_padded[:, :, 1:H + 1, 1:W + 1]
    # Direct neighbors
    luma_n = luma_padded[:, :, 0:H, 1:W + 1]  # North
    luma_s = luma_padded[:, :, 2:H + 2, 1:W + 1]  # South
    luma_e = luma_padded[:, :, 1:H + 1, 2:W + 2]  # East
    luma_w = luma_padded[:, :, 1:H + 1, 0:W]  # West
    # Diagonal neighbors
    luma_nw = luma_padded[:, :, 0:H, 0:W]  # Northwest
    luma_ne = luma_padded[:, :, 0:H, 2:W + 2]  # Northeast
    luma_sw = luma_padded[:, :, 2:H + 2, 0:W]  # Southwest
    luma_se = luma_padded[:, :, 2:H + 2, 2:W + 2]  # Southeast

    # Calculate local contrast
    luma_min = torch.min(luma_c, torch.min(torch.min(luma_n, luma_s), torch.min(luma_e, luma_w)))
    luma_max = torch.max(luma_c, torch.max(torch.max(luma_n, luma_s), torch.max(luma_e, luma_w)))

    # Local contrast
    luma_range = luma_max - luma_min

    # Skip pixels with low contrast (no aliasing detected)
    edge_mask = luma_range > torch.max(
        torch.full_like(luma_range, edge_threshold_min),
        luma_max * edge_threshold
    )

    # Calculate gradient direction
    # Horizontal gradient
    grad_h = abs((-1.0 * luma_nw + 1.0 * luma_ne) +
                 (-2.0 * luma_w + 2.0 * luma_e) +
                 (-1.0 * luma_sw + 1.0 * luma_se))

    # Vertical gradient
    grad_v = abs((-1.0 * luma_nw - 2.0 * luma_n - 1.0 * luma_ne) +
                 (1.0 * luma_sw + 2.0 * luma_s + 1.0 * luma_se))

    # Determine if edge is more horizontal or vertical
    is_horizontal = grad_h >= grad_v

    # Calculate step size based on edge direction
    step_x = torch.where(is_horizontal, torch.zeros_like(grad_h), torch.ones_like(grad_h) / W)
    step_y = torch.where(is_horizontal, torch.ones_like(grad_v) / H, torch.zeros_like(grad_v))

    # Calculate gradient along the edge
    luma_pos = torch.where(is_horizontal, luma_s, luma_e)
    luma_neg = torch.where(is_horizontal, luma_n, luma_w)

    grad_pos = abs(luma_pos - luma_c)
    grad_neg = abs(luma_neg - luma_c)

    # Determine direction along edge
    is_negative_dir = grad_neg >= grad_pos

    # Calculate pixel offset for sampling
    pixel_step = torch.where(is_negative_dir, -1.0, 1.0)
    offset_x = torch.where(is_horizontal, 0.0, pixel_step / W)
    offset_y = torch.where(is_horizontal, pixel_step / H, 0.0)

    # Subpixel anti-aliasing
    # Calculate average luminance in the 3x3 neighborhood
    luma_avg = (luma_nw + luma_ne + luma_sw + luma_se) * 0.25 * 0.5 + \
               (luma_n + luma_s + luma_e + luma_w) * 0.25

    # Calculate subpixel offset
    subpixel_offset = torch.clamp(
        abs(luma_avg - luma_c) / luma_range,
        0.0, 1.0
    )
    subpixel_offset = smoothstep(subpixel_offset) * subpixel_quality

    # Final offset calculation
    final_offset_x = offset_x * subpixel_offset
    final_offset_y = offset_y * subpixel_offset

    # Apply anti-aliasing by sampling with offset
    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Apply offset to sampling grid (only where edges are detected)
    offset_grid = grid.clone()
    for b in range(B):
        offset_grid[b, :, :, 0] += final_offset_x[b, 0, :, :] * 2.0 * edge_mask[b, 0, :, :]
        offset_grid[b, :, :, 1] += final_offset_y[b, 0, :, :] * 2.0 * edge_mask[b, 0, :, :]

    # Sample using grid_sample
    antialiased = F.grid_sample(images, offset_grid, mode='bilinear',
                                padding_mode='border', align_corners=False)

    return antialiased


def smoothstep(x):
    """Smooth interpolation function."""
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# Example usage and testing
if __name__ == "__main__":
    # Create a test image with aliasing (checkerboard pattern)
    def create_test_image(size=256, square_size=8):
        """Create a checkerboard pattern that exhibits aliasing."""
        img = torch.zeros(1, 3, size, size)
        for i in range(0, size, square_size * 2):
            for j in range(0, size, square_size * 2):
                img[:, :, i:i + square_size, j:j + square_size] = 1.0
                img[:, :, i + square_size:i + 2 * square_size, j + square_size:j + 2 * square_size] = 1.0
        return img


    # Create test batch
    batch_size = 2
    test_images = torch.cat([create_test_image() for _ in range(batch_size)], dim=0)

    # Add some noise to make it more realistic
    test_images += torch.randn_like(test_images) * 0.05
    test_images = torch.clamp(test_images, 0, 1)

    # Apply FXAA
    antialiased_images = fxaa(test_images)

    print(f"Input shape: {test_images.shape}")
    print(f"Output shape: {antialiased_images.shape}")
    print(f"Input range: [{test_images.min():.3f}, {test_images.max():.3f}]")
    print(f"Output range: [{antialiased_images.min():.3f}, {antialiased_images.max():.3f}]")

    # Calculate difference to verify anti-aliasing is applied
    diff = torch.abs(antialiased_images - test_images).mean()
    print(f"Average pixel difference: {diff:.4f}")

    # Visualization code (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Show first image from batch
        axes[0, 0].imshow(test_images[0].permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title("Original (with aliasing)")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(antialiased_images[0].permute(1, 2, 0).cpu().numpy())
        axes[0, 1].set_title("After FXAA")
        axes[0, 1].axis('off')

        # Show difference map
        diff_map = torch.abs(antialiased_images[0] - test_images[0]).mean(dim=0)
        im = axes[1, 0].imshow(diff_map.cpu().numpy(), cmap='hot')
        axes[1, 0].set_title("Difference Map")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])

        # Zoom in on a region to see the effect better
        zoom_region = test_images[0, :, 100:150, 100:150]
        zoom_aa = antialiased_images[0, :, 100:150, 100:150]

        axes[1, 1].imshow(torch.cat([zoom_region, zoom_aa], dim=2).permute(1, 2, 0).cpu().numpy())
        axes[1, 1].set_title("Zoomed: Original (left) vs FXAA (right)")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")