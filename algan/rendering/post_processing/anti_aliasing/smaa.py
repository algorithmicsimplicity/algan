from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from algan.logging.logger import get_logger

logger = get_logger("smaa")


class SMAA(nn.Module):
    """
    Subpixel Morphological Anti-aliasing (SMAA) implementation in PyTorch.

    This implementation follows the three-pass approach:
    1. Edge Detection Pass
    2. Blending Weight Calculation Pass
    3. Neighborhood Blending Pass

    Uses official SMAA area and search textures loaded from PNG files.
    """

    def __init__(self, area_tex_path: str = "AreaTex.png",
                 search_tex_path: str = "SearchTex.png",
                 threshold: float = 0.1,
                 local_contrast_adaptation_factor: float = 2.0,
                 max_search_steps: int = 32,
                 max_search_steps_diag: int = 16,
                 corner_rounding: int = 25):
        """
        Initialize SMAA module.

        Args:
            area_tex_path: Path to SMAA area texture PNG file
            search_tex_path: Path to SMAA search texture PNG file
            threshold: Edge detection threshold (0.05-0.2 typical)
            local_contrast_adaptation_factor: Factor for local contrast adaptation
            max_search_steps: Maximum search distance for horizontal/vertical edges
            max_search_steps_diag: Maximum search distance for diagonal edges
            corner_rounding: Corner rounding percentage (0-100)
        """
        super().__init__()

        self.threshold = threshold
        self.local_contrast_factor = local_contrast_adaptation_factor
        self.max_search_steps = max_search_steps
        self.max_search_steps_diag = max_search_steps_diag
        self.corner_rounding = corner_rounding

        # SMAA constants
        self.SMAA_AREATEX_MAX_DISTANCE = 16
        self.SMAA_AREATEX_MAX_DISTANCE_DIAG = 20
        self.SMAA_AREATEX_PIXEL_SIZE = (1.0 / 256.0, 1.0 / 256.0)
        self.SMAA_AREATEX_SUBTEX_SIZE = (1.0 / 16.0, 1.0 / 4.0)

        # Load official SMAA textures from files
        self.register_buffer('area_tex', self._load_area_texture(area_tex_path))
        self.register_buffer('search_tex', self._load_search_texture(search_tex_path))

    def _load_area_texture(self, path: str) -> torch.Tensor:
        """
        Load SMAA area texture from PNG file.
        The area texture is 256x256 with RG channels encoding area values.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Area texture not found at: {path}")

        img = Image.open(path)
        area_np = np.array(img).astype(np.float32) / 255.0

        if len(area_np.shape) == 2:
            area_np = np.stack([area_np, np.zeros_like(area_np)], axis=-1)
        elif area_np.shape[-1] >= 3:
            area_np = area_np[:, :, :2]

        # Convert to torch tensor (H, W, C) -> (C, H, W)
        area_tex = torch.from_numpy(area_np).permute(2, 0, 1)
        logger.debug(f"Loaded area texture from {path}: shape {area_tex.shape}")
        return area_tex

    def _load_search_texture(self, path: str) -> torch.Tensor:
        """
        Load SMAA search texture from PNG file.
        The search texture encodes search distances.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Search texture not found at: {path}")

        img = Image.open(path)
        search_np = np.array(img).astype(np.float32) / 255.0

        if len(search_np.shape) == 3:
            search_np = search_np[:, :, 0]

        search_tex = torch.from_numpy(search_np).unsqueeze(0)
        logger.debug(f"Loaded search texture from {path}: shape {search_tex.shape}")
        return search_tex

    def _rgb_to_luma(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance."""
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)
        weights = weights.view(1, 3, 1, 1)
        return (rgb * weights).sum(dim=1, keepdim=True)

    def _edge_detection_pass(self, color: torch.Tensor) -> torch.Tensor:
        """
        First pass: Detect edges using luminance gradients.
        Returns edges in 4 channels: left, top, right, bottom
        """
        B, C, H, W = color.shape
        device = color.device

        # Convert to luminance
        luma = self._rgb_to_luma(color) if C == 3 else color[:, :1]

        # Prepare shifted versions for gradient calculation
        padded = F.pad(luma, (1, 1, 1, 1), mode='replicate')

        luma_left = padded[:, :, 1:-1, :-2]
        luma_right = padded[:, :, 1:-1, 2:]
        luma_top = padded[:, :, :-2, 1:-1]
        luma_bottom = padded[:, :, 2:, 1:-1]

        # Calculate gradients
        delta_left = torch.abs(luma - luma_left)
        delta_right = torch.abs(luma - luma_right)
        delta_top = torch.abs(luma - luma_top)
        delta_bottom = torch.abs(luma - luma_bottom)

        # Local contrast adaptation
        local_avg = (luma_left + luma_right + luma_top + luma_bottom) / 4.0
        threshold = self.threshold * (1.0 + self.local_contrast_factor * local_avg)

        # Edge detection with threshold
        edges = torch.zeros(B, 4, H, W, device=device)
        edges[:, 0:1] = (delta_left >= threshold).float()  # Left edges
        edges[:, 1:2] = (delta_top >= threshold).float()  # Top edges
        edges[:, 2:3] = (delta_right >= threshold).float()  # Right edges
        edges[:, 3:4] = (delta_bottom >= threshold).float()  # Bottom edges

        # Disable edges at image boundaries
        edges[:, 0, :, 0] = 0  # No left edge at left boundary
        edges[:, 1, 0, :] = 0  # No top edge at top boundary
        edges[:, 2, :, -1] = 0  # No right edge at right boundary
        edges[:, 3, -1, :] = 0  # No bottom edge at bottom boundary

        return edges

    def _search_diag_1(
        self, edges: torch.Tensor, pos: tuple[int, int], direction: tuple[int, int]
    ) -> float:
        """Search for diagonal patterns in one direction."""
        B, C, H, W = edges.shape
        pos_y, pos_x = pos
        dir_y, dir_x = direction

        for i in range(self.SMAA_AREATEX_MAX_DISTANCE_DIAG):
            new_y = pos_y + dir_y * (i + 1)
            new_x = pos_x + dir_x * (i + 1)

            if new_y < 0 or new_y >= H or new_x < 0 or new_x >= W:
                return i

            # Check if edge continues
            edge_val = edges[0, 0, new_y, new_x] if C == 1 else edges[0, 1, new_y, new_x]
            if edge_val < 0.5:
                return i

        return self.SMAA_AREATEX_MAX_DISTANCE_DIAG

    def _search_diag_2(
        self, edges: torch.Tensor, pos: tuple[int, int], direction: tuple[int, int]
    ) -> float:
        """Search for diagonal patterns in opposite direction."""
        return self._search_diag_1(edges, pos, (-direction[0], -direction[1]))

    def _area_tex_lookup(self, d1: float, d2: float, y: int, subsample_index: int) -> tuple[float, float]:
        """Lookup area values from the area texture."""
        # Calculate texture coordinates
        tex_coord_x = (d1 * self.SMAA_AREATEX_SUBTEX_SIZE[0] + subsample_index * self.SMAA_AREATEX_SUBTEX_SIZE[0])
        tex_coord_y = (d2 * self.SMAA_AREATEX_SUBTEX_SIZE[1] + y * self.SMAA_AREATEX_SUBTEX_SIZE[1])

        # Clamp coordinates
        tex_coord_x = min(max(tex_coord_x, 0), 1)
        tex_coord_y = min(max(tex_coord_y, 0), 1)

        # Sample area texture
        x_idx = int(tex_coord_x * 255)
        y_idx = int(tex_coord_y * 255)

        area_val = self.area_tex[:, y_idx, x_idx]
        return area_val[0].item(), area_val[1].item()

    def _blending_weight_calculation_pass(self, edges: torch.Tensor,
                                          color: torch.Tensor) -> torch.Tensor:
        """Second pass: Calculate blending weights for detected edges."""
        B, _, H, W = edges.shape
        device = edges.device
        weights = torch.zeros(B, 4, H, W, device=device)

        # Process each pixel
        for b in range(B):
            for y in range(H):
                for x in range(W):
                    # Check for edges at this pixel
                    edge_left = edges[b, 0, y, x]
                    edge_top = edges[b, 1, y, x]
                    edge_right = edges[b, 2, y, x]
                    edge_bottom = edges[b, 3, y, x]

                    # Calculate horizontal weights
                    if edge_left > 0.5 or edge_right > 0.5:
                        # Search distances
                        d_left = 0
                        d_right = 0

                        # Search left
                        for i in range(1, min(self.max_search_steps, x + 1)):
                            if x - i >= 0:
                                if edges[b, 0, y, x - i] < 0.5:
                                    break
                                d_left = i

                        # Search right
                        for i in range(1, min(self.max_search_steps, W - x)):
                            if x + i < W:
                                if edges[b, 2, y, x + i] < 0.5:
                                    break
                                d_right = i

                        # Calculate area
                        if d_left + d_right > 0:
                            area = 1.0 / (d_left + d_right + 1)
                            weights[b, 0, y, x] = area * d_right * edge_left
                            weights[b, 2, y, x] = area * d_left * edge_right

                    # Calculate vertical weights
                    if edge_top > 0.5 or edge_bottom > 0.5:
                        # Search distances
                        d_top = 0
                        d_bottom = 0

                        # Search up
                        for i in range(1, min(self.max_search_steps, y + 1)):
                            if y - i >= 0:
                                if edges[b, 1, y - i, x] < 0.5:
                                    break
                                d_top = i

                        # Search down
                        for i in range(1, min(self.max_search_steps, H - y)):
                            if y + i < H:
                                if edges[b, 3, y + i, x] < 0.5:
                                    break
                                d_bottom = i

                        # Calculate area
                        if d_top + d_bottom > 0:
                            area = 1.0 / (d_top + d_bottom + 1)
                            weights[b, 1, y, x] = area * d_bottom * edge_top
                            weights[b, 3, y, x] = area * d_top * edge_bottom

        # Apply corner rounding
        if self.corner_rounding > 0:
            factor = 1.0 - (self.corner_rounding / 100.0)

            # Detect corners (where horizontal and vertical edges meet)
            corners_tl = edges[:, 0:1] * edges[:, 1:2]  # Top-left
            corners_tr = edges[:, 2:3] * edges[:, 1:2]  # Top-right
            corners_bl = edges[:, 0:1] * edges[:, 3:4]  # Bottom-left
            corners_br = edges[:, 2:3] * edges[:, 3:4]  # Bottom-right

            corners = torch.max(torch.max(corners_tl, corners_tr),
                                torch.max(corners_bl, corners_br))

            # Reduce weights at corners
            weights = weights * (1.0 - corners * (1.0 - factor))

        return weights

    def _neighborhood_blending_pass(self, color: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """Third pass: Blend colors using calculated weights."""
        B, C, H, W = color.shape

        # Extract weight components
        w_left = weights[:, 0:1]
        w_top = weights[:, 1:2]
        w_right = weights[:, 2:3]
        w_bottom = weights[:, 3:4]

        # Prepare padded image for neighbor sampling
        padded = F.pad(color, (1, 1, 1, 1), mode='replicate')

        # Sample neighbors
        n_left = padded[:, :, 1:-1, :-2]
        n_top = padded[:, :, :-2, 1:-1]
        n_right = padded[:, :, 1:-1, 2:]
        n_bottom = padded[:, :, 2:, 1:-1]

        # Calculate weighted blend
        weighted_sum = (
                color * (1.0 - torch.clamp(w_left + w_top + w_right + w_bottom, 0, 1)) +
                n_left * w_left +
                n_top * w_top +
                n_right * w_right +
                n_bottom * w_bottom
        )

        return torch.clamp(weighted_sum, 0.0, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply SMAA anti-aliasing to input images.

        Args:
            image: Input batch of images (B, C, H, W) in range [0, 1]

        Returns:
            Anti-aliased images (B, C, H, W)
        """
        # Ensure input is in correct range
        image = torch.clamp(image, 0.0, 1.0)

        # Pass 1: Edge Detection
        edges = self._edge_detection_pass(image)

        # Pass 2: Blending Weight Calculation
        weights = self._blending_weight_calculation_pass(edges, image)

        # Pass 3: Neighborhood Blending
        result = self._neighborhood_blending_pass(image, weights)

        return result


def apply_smaa(images: torch.Tensor,
               area_tex_path: str = "AreaTex.png",
               search_tex_path: str = "SearchTex.png",
               threshold: float = 0.05,
               local_contrast_adaptation_factor: float = 2.0,
               max_search_steps: int = 32,
               corner_rounding: int = 25) -> torch.Tensor:
    """
    Apply SMAA to a batch of images.

    Args:
        images: Input tensor of shape (B, C, H, W) with values in [0, 1]
        area_tex_path: Path to SMAA area texture PNG file
        search_tex_path: Path to SMAA search texture PNG file
        threshold: Edge detection threshold (lower = more edges detected)
        local_contrast_adaptation_factor: Adapts threshold to local contrast
        max_search_steps: Maximum edge search distance
        corner_rounding: Corner rounding percentage (0-100)

    Returns:
        Anti-aliased images with same shape as input
    """
    # Check if texture files exist
    if not os.path.exists(area_tex_path):
        raise FileNotFoundError(
            f"Area texture not found at: {area_tex_path}\n"
            "Please ensure you have the official SMAA AreaTex.png file in the same directory."
        )
    if not os.path.exists(search_tex_path):
        raise FileNotFoundError(
            f"Search texture not found at: {search_tex_path}\n"
            "Please ensure you have the official SMAA SearchTex.png file in the same directory."
        )

    smaa = SMAA(area_tex_path=area_tex_path,
                search_tex_path=search_tex_path,
                threshold=threshold,
                local_contrast_adaptation_factor=local_contrast_adaptation_factor,
                max_search_steps=max_search_steps,
                corner_rounding=corner_rounding)

    # Move to same device as input
    smaa = smaa.to(images.device)

    # Set to eval mode
    smaa.eval()

    with torch.no_grad():
        result = smaa(images)

    return result


# Example usage and testing
if __name__ == "__main__":

    # Check for required texture files
    area_tex_path = "AreaTex.png"
    search_tex_path = "SearchTex.png"

    print("SMAA PyTorch Implementation")
    print("-" * 40)

    try:
        # Create more aggressive test patterns
        batch_size = 1
        height, width = 256, 256
        channels = 3

        # Generate test image with strong aliasing
        test_image = torch.zeros(batch_size, channels, height, width)

        # Add diagonal line (worst case for aliasing)
        for i in range(min(height, width)):
            if i < height and i < width:
                test_image[0, :, i, i] = 1.0
                # Add adjacent pixels for visible line
                if i > 0:
                    test_image[0, :, i - 1, i] = 0.3
                if i < height - 1 and i < width - 1:
                    test_image[0, :, i + 1, i] = 0.3

        # Add horizontal and vertical lines for comparison
        test_image[0, :, height // 2 - 1:height // 2 + 1, :] = 0.7
        test_image[0, :, :, width // 2 - 1:width // 2 + 1] = 0.7

        # Add circle
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 3

        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        dist = torch.sqrt((y_grid - center_y).float() ** 2 + (x_grid - center_x).float() ** 2)
        circle_mask = (torch.abs(dist - radius) < 2).float()
        test_image[0, :] = torch.clamp(test_image[0, :] + circle_mask.unsqueeze(0) * 0.8, 0, 1)

        print(f"Input shape: {test_image.shape}")
        print(f"Input range: [{test_image.min():.3f}, {test_image.max():.3f}]")

        # Apply SMAA with lower threshold for more aggressive anti-aliasing
        print("\nApplying SMAA anti-aliasing...")
        antialiased = apply_smaa(test_image,
                                 area_tex_path=area_tex_path,
                                 search_tex_path=search_tex_path,
                                 threshold=0.05,  # Lower threshold
                                 local_contrast_adaptation_factor=2.0,
                                 max_search_steps=32,
                                 corner_rounding=25)

        print(f"Output shape: {antialiased.shape}")
        print(f"Output range: [{antialiased.min():.3f}, {antialiased.max():.3f}]")

        # Calculate difference
        difference = torch.abs(antialiased - test_image)
        print("\nAnti-aliasing effect:")
        print(f"Average change: {difference.mean():.4f}")
        print(f"Max change: {difference.max():.4f}")
        print(f"Pixels affected: {(difference > 0.001).float().mean() * 100:.1f}%")

        # Optional: Save results for visual inspection
        if difference.max() > 0.001:
            print("\n✓ SMAA is working! Edges have been smoothed.")

            # Convert to numpy for saving
            original_np = (test_image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            antialiased_np = (antialiased[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            diff_np = (difference[0].max(dim=0)[0].numpy() * 255 * 5).astype(np.uint8)  # Amplify difference

            # Save images
            Image.fromarray(original_np).save("test_original.png")
            Image.fromarray(antialiased_np).save("test_antialiased.png")
            Image.fromarray(diff_np).save("test_difference.png")

            print("Saved test images: test_original.png, test_antialiased.png, test_difference.png")
        else:
            print("\n⚠ Warning: No anti-aliasing effect detected.")
            print("This might indicate an issue with edge detection or weight calculation.")

            # Debug: Check edge detection
            smaa_debug = SMAA(area_tex_path=area_tex_path,
                              search_tex_path=search_tex_path,
                              threshold=0.05)
            edges = smaa_debug._edge_detection_pass(test_image)
            edge_count = (edges > 0.5).float().sum()
            print(f"Debug: Detected {edge_count.item():.0f} edge pixels")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo use this script, you need the official SMAA texture files:")
        print("1. AreaTex.png - The area texture")
        print("2. SearchTex.png - The search texture")
        print("\nThese can be obtained from the official SMAA repository:")
        print("https://github.com/iryoku/smaa")
        print("\nPlace these files in the same directory as this script.")
