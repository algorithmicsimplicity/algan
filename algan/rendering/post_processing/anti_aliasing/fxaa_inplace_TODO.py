import torch
import torch.nn.functional as F


class FXAAOptimized:
    """Ultra-optimized FXAA with minimal memory allocation and maximum in-place operations."""

    def __init__(self, max_batch_size=32, max_height=2048, max_width=2048, device='cuda'):
        """
        Initialize FXAA with pre-allocated buffers for all operations.

        Args:
            max_batch_size: Maximum batch size to support
            max_height: Maximum image height to support  
            max_width: Maximum image width to support
            device: Device to allocate tensors on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_batch_size = max_batch_size
        self.max_height = max_height
        self.max_width = max_width

        # Pre-allocate RGB to luminance weights
        self.luma_weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(1, 3, 1, 1)

        # Pre-allocate ALL working buffers to avoid any allocation during processing
        # Main buffers
        self.luma = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device)
        self.luma_padded = torch.zeros(max_batch_size, 1, max_height + 2, max_width + 2, device=self.device)

        # Working buffers for calculations (reused for multiple purposes)
        self.work1 = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device)
        self.work2 = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device)
        self.work3 = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device)
        self.work4 = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device)

        # Boolean mask
        self.edge_mask = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device, dtype=torch.bool)
        self.bool_work = torch.zeros(max_batch_size, 1, max_height, max_width, device=self.device, dtype=torch.bool)

        # Grid for sampling
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, max_height, device=self.device),
            torch.linspace(-1, 1, max_width, device=self.device),
            indexing='ij'
        )
        self.base_grid_x = grid_x.unsqueeze(0).unsqueeze(3)
        self.base_grid_y = grid_y.unsqueeze(0).unsqueeze(3)
        self.sampling_grid = torch.zeros(max_batch_size, max_height, max_width, 2, device=self.device)

        # Constants
        self.eps = 1e-8

    def apply(self, images, edge_threshold=0.125, edge_threshold_min=0.0625, subpixel_quality=0.75):
        """
        Apply FXAA with minimal memory allocation.

        Args:
            images: Input tensor of shape (B, C, H, W) with values in range [0, 1]
            edge_threshold: Minimum local contrast required to apply AA
            edge_threshold_min: Minimum edge threshold for darker areas
            subpixel_quality: Controls subpixel anti-aliasing quality

        Returns:
            Anti-aliased images tensor of shape (B, C, H, W)
        """
        B, C, H, W = images.shape

        # Get views of pre-allocated buffers for current size
        luma = self.luma[:B, :, :H, :W]
        luma_padded = self.luma_padded[:B, :, :H + 2, :W + 2]

        # Working buffers
        luma_min = self.work1[:B, :, :H, :W]
        luma_max = self.work2[:B, :, :H, :W]
        luma_range = self.work3[:B, :, :H, :W]
        grad_h = self.work4[:B, :, :H, :W]

        # Reusable buffers (will be reassigned for different purposes)
        grad_v = luma_min  # Reuse after luma_min is no longer needed
        offset_x = luma_min  # Further reuse
        offset_y = luma_max  # Further reuse

        edge_mask = self.edge_mask[:B, :, :H, :W]
        is_horizontal = self.bool_work[:B, :, :H, :W]

        # Convert to luminance (in-place into luma buffer)
        torch.sum(images * self.luma_weights, dim=1, keepdim=True, out=luma)

        # Pad luminance (copy into padded buffer)
        luma_padded[:, :, 1:H + 1, 1:W + 1] = luma
        # Replicate padding
        luma_padded[:, :, 0, 1:W + 1] = luma[:, :, 0, :]
        luma_padded[:, :, H + 1, 1:W + 1] = luma[:, :, H - 1, :]
        luma_padded[:, :, :, 0] = luma_padded[:, :, :, 1]
        luma_padded[:, :, :, W + 1] = luma_padded[:, :, :, W]

        # Create views for neighbors (no memory allocation, just views!)
        luma_c = luma_padded[:, :, 1:H + 1, 1:W + 1]
        luma_n = luma_padded[:, :, 0:H, 1:W + 1]
        luma_s = luma_padded[:, :, 2:H + 2, 1:W + 1]
        luma_e = luma_padded[:, :, 1:H + 1, 2:W + 2]
        luma_w = luma_padded[:, :, 1:H + 1, 0:W]
        luma_nw = luma_padded[:, :, 0:H, 0:W]
        luma_ne = luma_padded[:, :, 0:H, 2:W + 2]
        luma_sw = luma_padded[:, :, 2:H + 2, 0:W]
        luma_se = luma_padded[:, :, 2:H + 2, 2:W + 2]

        # Calculate local min (in-place operations)
        torch.minimum(luma_n, luma_s, out=luma_min)
        torch.minimum(luma_min, luma_e, out=luma_min)
        torch.minimum(luma_min, luma_w, out=luma_min)
        torch.minimum(luma_min, luma_c, out=luma_min)

        # Calculate local max (in-place operations)
        torch.maximum(luma_n, luma_s, out=luma_max)
        torch.maximum(luma_max, luma_e, out=luma_max)
        torch.maximum(luma_max, luma_w, out=luma_max)
        torch.maximum(luma_max, luma_c, out=luma_max)

        # Calculate range (in-place)
        torch.sub(luma_max, luma_min, out=luma_range)

        # Calculate edge threshold (in-place)
        # First calculate luma_max * edge_threshold
        luma_max.mul_(edge_threshold)
        # Take maximum with edge_threshold_min
        luma_max.clamp_min_(edge_threshold_min)
        # Now luma_max contains the threshold

        # Edge mask (in-place)
        torch.gt(luma_range, luma_max, out=edge_mask)

        if not edge_mask.any():
            return images

        # Now luma_min and luma_max are free to reuse
        # Calculate gradients (reuse buffers)

        # Horizontal gradient (store in grad_h)
        grad_h.zero_()
        grad_h.add_(luma_ne, alpha=-1.0)
        grad_h.add_(luma_nw, alpha=1.0)
        grad_h.add_(luma_e, alpha=2.0)
        grad_h.add_(luma_w, alpha=-2.0)
        grad_h.add_(luma_se, alpha=1.0)
        grad_h.add_(luma_sw, alpha=-1.0)
        grad_h.abs_()

        # Vertical gradient (reuse luma_min as grad_v)
        grad_v.zero_()
        grad_v.add_(luma_nw, alpha=-1.0)
        grad_v.add_(luma_n, alpha=-2.0)
        grad_v.add_(luma_ne, alpha=-1.0)
        grad_v.add_(luma_sw, alpha=1.0)
        grad_v.add_(luma_s, alpha=2.0)
        grad_v.add_(luma_se, alpha=1.0)
        grad_v.abs_()

        # Determine if horizontal (in-place)
        torch.ge(grad_h, grad_v, out=is_horizontal)

        # Calculate gradient along edge direction
        # Reuse grad_h for luma_pos, grad_v for luma_neg
        luma_pos = grad_h
        luma_neg = grad_v

        torch.where(is_horizontal, luma_s, luma_e, out=luma_pos)
        torch.where(is_horizontal, luma_n, luma_w, out=luma_neg)

        # Calculate gradients (in-place)
        luma_pos.sub_(luma_c).abs_()
        luma_neg.sub_(luma_c).abs_()

        # Determine negative direction (reuse is_horizontal as is_negative_dir)
        is_negative_dir = is_horizontal
        torch.ge(luma_neg, luma_pos, out=is_negative_dir)

        # Calculate offsets (reuse luma_min as offset_x, luma_max as offset_y)
        offset_x.zero_()
        offset_y.zero_()

        # Calculate pixel step and apply to offsets
        # For horizontal edges: offset_y = pixel_step / H, offset_x = 0
        # For vertical edges: offset_x = pixel_step / W, offset_y = 0

        # We need to be clever here to avoid allocation
        # Use luma_pos as temporary for pixel_step
        pixel_step = luma_pos
        torch.where(is_negative_dir, torch.tensor(-1.0, device=self.device),
                    torch.tensor(1.0, device=self.device), out=pixel_step)

        # Apply to offsets based on edge direction
        # First handle vertical edges (not horizontal)
        not_horizontal_mask = ~is_horizontal
        offset_x[not_horizontal_mask] = pixel_step[not_horizontal_mask] / W

        # Handle horizontal edges  
        horizontal_mask = is_horizontal
        offset_y[horizontal_mask] = pixel_step[horizontal_mask] / H

        # Subpixel anti-aliasing
        # Calculate average luminance (reuse luma_neg as luma_avg)
        luma_avg = luma_neg
        luma_avg.zero_()
        luma_avg.add_(luma_nw, alpha=0.25 * 0.5)
        luma_avg.add_(luma_ne, alpha=0.25 * 0.5)
        luma_avg.add_(luma_sw, alpha=0.25 * 0.5)
        luma_avg.add_(luma_se, alpha=0.25 * 0.5)
        luma_avg.add_(luma_n, alpha=0.25)
        luma_avg.add_(luma_s, alpha=0.25)
        luma_avg.add_(luma_e, alpha=0.25)
        luma_avg.add_(luma_w, alpha=0.25)

        # Calculate subpixel offset (reuse grad_h as subpixel_offset)
        subpixel_offset = grad_h
        subpixel_offset.copy_(luma_avg)
        subpixel_offset.sub_(luma_c).abs_()
        luma_range.add_(self.eps)  # Avoid division by zero
        subpixel_offset.div_(luma_range)
        subpixel_offset.clamp_(0.0, 1.0)

        # Apply smoothstep (in-place)
        # smoothstep(x) = x * x * (3 - 2 * x)
        temp = luma_range  # Reuse as temporary
        temp.copy_(subpixel_offset)
        temp.mul_(-2.0).add_(3.0)
        subpixel_offset.mul_(subpixel_offset).mul_(temp).mul_(subpixel_quality)

        # Apply final offsets (in-place)
        offset_x.mul_(subpixel_offset)
        offset_y.mul_(subpixel_offset)

        # Apply edge mask and scale
        edge_mask_float = edge_mask.float()
        offset_x.mul_(edge_mask_float).mul_(2.0)
        offset_y.mul_(edge_mask_float).mul_(2.0)

        # Create sampling grid (use pre-allocated grid)
        grid = self.sampling_grid[:B, :H, :W]
        grid[:, :, :, 0] = self.base_grid_x[:B, :H, :W, 0]
        grid[:, :, :, 1] = self.base_grid_y[:B, :H, :W, 0]

        # Add offsets (in-place)
        grid[:, :, :, 0].add_(offset_x.squeeze(1))
        grid[:, :, :, 1].add_(offset_y.squeeze(1))

        # Apply sampling
        return F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def __call__(self, images, edge_threshold=0.125, edge_threshold_min=0.0625, subpixel_quality=0.75):
        return self.apply(images, edge_threshold, edge_threshold_min, subpixel_quality)


batch_size = 8
height, width = 512, 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


fxaa_processor = FXAAOptimized(
        max_batch_size=batch_size,
        max_height=height,
        max_width=width,
        device=device
)


def fxaa(images, edge_threshold=0.125, edge_threshold_min=0.0625, subpixel_quality=0.75):
    return fxaa_processor(images, edge_threshold, edge_threshold_min, subpixel_quality)


def fxaa_minimal_memory(images, edge_threshold=0.125, edge_threshold_min=0.0625, subpixel_quality=0.75):
    """
    Standalone FXAA with minimal memory allocation using maximum in-place operations.

    Args:
        images: Input tensor of shape (B, C, H, W) with values in range [0, 1]
        edge_threshold: Minimum local contrast required to apply AA
        edge_threshold_min: Minimum edge threshold for darker areas
        subpixel_quality: Controls subpixel anti-aliasing quality

    Returns:
        Anti-aliased images tensor of shape (B, C, H, W)
    """
    B, C, H, W = images.shape
    device = images.device

    # Allocate minimal working buffers
    luma = torch.empty(B, 1, H, W, device=device)
    luma_padded = torch.empty(B, 1, H + 2, W + 2, device=device)
    work1 = torch.empty(B, 1, H, W, device=device)
    work2 = torch.empty(B, 1, H, W, device=device)
    edge_mask = torch.empty(B, 1, H, W, device=device, dtype=torch.bool)

    # RGB to luminance
    luma_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    torch.sum(images * luma_weights, dim=1, keepdim=True, out=luma)

    # Pad (manual padding to control memory)
    luma_padded[:, :, 1:H + 1, 1:W + 1] = luma
    luma_padded[:, :, 0, 1:W + 1] = luma[:, :, 0, :]
    luma_padded[:, :, H + 1, 1:W + 1] = luma[:, :, H - 1, :]
    luma_padded[:, :, :, 0] = luma_padded[:, :, :, 1]
    luma_padded[:, :, :, W + 1] = luma_padded[:, :, :, W]

    # Neighbor views (no allocation)
    luma_c = luma_padded[:, :, 1:H + 1, 1:W + 1]
    luma_n = luma_padded[:, :, 0:H, 1:W + 1]
    luma_s = luma_padded[:, :, 2:H + 2, 1:W + 1]
    luma_e = luma_padded[:, :, 1:H + 1, 2:W + 2]
    luma_w = luma_padded[:, :, 1:H + 1, 0:W]
    luma_nw = luma_padded[:, :, 0:H, 0:W]
    luma_ne = luma_padded[:, :, 0:H, 2:W + 2]
    luma_sw = luma_padded[:, :, 2:H + 2, 0:W]
    luma_se = luma_padded[:, :, 2:H + 2, 2:W + 2]

    # Local contrast
    torch.minimum(torch.minimum(luma_n, luma_s), torch.minimum(luma_e, luma_w), out=work1)
    torch.minimum(work1, luma_c, out=work1)  # work1 = luma_min

    torch.maximum(torch.maximum(luma_n, luma_s), torch.maximum(luma_e, luma_w), out=work2)
    torch.maximum(work2, luma_c, out=work2)  # work2 = luma_max

    luma_range = work2.sub_(work1)  # In-place subtraction, work2 now contains range

    # Edge detection
    threshold = torch.maximum(
        torch.tensor(edge_threshold_min, device=device),
        (work1 + luma_range) * edge_threshold  # Reconstruct luma_max
    )
    torch.gt(luma_range, threshold, out=edge_mask)

    if not edge_mask.any():
        return images

    # Gradient calculation (reuse work1 and work2)
    grad_h = work1.zero_()
    grad_h.add_(luma_ne, alpha=1.0).add_(luma_nw, alpha=-1.0)
    grad_h.add_(luma_e, alpha=2.0).add_(luma_w, alpha=-2.0)
    grad_h.add_(luma_se, alpha=1.0).add_(luma_sw, alpha=-1.0)
    grad_h.abs_()

    grad_v = luma.zero_()  # Reuse luma buffer since we don't need it anymore
    grad_v.add_(luma_sw, alpha=1.0).add_(luma_nw, alpha=-1.0)
    grad_v.add_(luma_s, alpha=2.0).add_(luma_n, alpha=-2.0)
    grad_v.add_(luma_se, alpha=1.0).add_(luma_ne, alpha=-1.0)
    grad_v.abs_()

    is_horizontal = grad_h >= grad_v

    # Edge direction analysis
    luma_pos = torch.where(is_horizontal, luma_s, luma_e)
    luma_neg = torch.where(is_horizontal, luma_n, luma_w)

    grad_pos = (luma_pos - luma_c).abs_()
    grad_neg = (luma_neg - luma_c).abs_()

    pixel_step = torch.where(grad_neg >= grad_pos, -1.0, 1.0)

    offset_x = torch.where(is_horizontal, 0.0, pixel_step / W)
    offset_y = torch.where(is_horizontal, pixel_step / H, 0.0)

    # Subpixel offset
    luma_avg = (luma_nw + luma_ne + luma_sw + luma_se) * 0.125 + \
               (luma_n + luma_s + luma_e + luma_w) * 0.25

    subpixel_offset = ((luma_avg - luma_c).abs_() / (luma_range + 1e-8)).clamp_(0.0, 1.0)
    subpixel_offset.mul_(subpixel_offset).mul_(3.0 - 2.0 * subpixel_offset).mul_(subpixel_quality)

    offset_x.mul_(subpixel_offset).mul_(edge_mask.float()).mul_(2.0)
    offset_y.mul_(subpixel_offset).mul_(edge_mask.float()).mul_(2.0)

    # Grid sampling
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    grid[..., 0].add_(offset_x.squeeze(1))
    grid[..., 1].add_(offset_y.squeeze(1))

    return F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)


# Testing and benchmarking
if __name__ == "__main__":
    import time

    # Test parameters
    batch_size = 8
    height, width = 512, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test images
    test_images = torch.rand(batch_size, 3, height, width, device=device)

    # Initialize optimized FXAA
    fxaa_processor = FXAAOptimized(
        max_batch_size=batch_size,
        max_height=height,
        max_width=width,
        device=device
    )

    # Warmup
    for _ in range(5):
        _ = fxaa_processor(test_images)
        _ = fxaa_minimal_memory(test_images)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Memory before
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # Benchmark class version
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        result_class = fxaa_processor(test_images)
    if device == 'cuda':
        torch.cuda.synchronize()
    class_time = time.time() - start

    if device == 'cuda':
        class_peak_mem = torch.cuda.max_memory_allocated() - mem_before
        torch.cuda.reset_peak_memory_stats()

    # Benchmark standalone version
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        result_standalone = fxaa_minimal_memory(test_images)
    if device == 'cuda':
        torch.cuda.synchronize()
    standalone_time = time.time() - start

    if device == 'cuda':
        standalone_peak_mem = torch.cuda.max_memory_allocated() - mem_before

    print(f"Device: {device}")
    print(f"Input shape: {test_images.shape}")
    print(f"\nPerformance (100 iterations):")
    print(f"  Class-based (pre-allocated): {class_time:.3f}s")
    print(f"  Standalone minimal memory: {standalone_time:.3f}s")
    print(f"  Speedup: {standalone_time / class_time:.2f}x")

    if device == 'cuda':
        print(f"\nMemory usage:")
        print(f"  Class-based peak: {class_peak_mem / 1024 ** 2:.2f} MB")
        print(f"  Standalone peak: {standalone_peak_mem / 1024 ** 2:.2f} MB")
        print(f"  Memory saved: {(standalone_peak_mem - class_peak_mem) / 1024 ** 2:.2f} MB")

    # Verify outputs are similar
    diff = (result_class - result_standalone).abs().mean()
    print(f"\nAccuracy check - difference: {diff:.6f}")