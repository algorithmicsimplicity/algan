import torch
import taichi as ti
import numpy as np
from typing import Tuple, Optional

# Initialize Taichi
ti.init(arch=ti.gpu)  # Use GPU if available, fallback to CPU


@ti.data_oriented
class TriangleRasterizer3D:
    def __init__(self, width: int = 800, height: int = 600, max_fragments: int = 1000000):
        """
        Initialize the 3D triangle rasterizer with alpha blending using per-pixel linked lists.

        Args:
            width: Frame buffer width
            height: Frame buffer height
            max_fragments: Maximum number of fragments to store
        """
        self.width = width
        self.height = height
        self.max_fragments = max_fragments

        # Taichi fields for triangle data (will be allocated dynamically)
        self.triangles = None
        self.colors = None

        # Projection matrix (4x4)
        self.projection_matrix = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())

        # Fragment storage
        self.fragment_coords = ti.Vector.field(2, dtype=ti.i32, shape=max_fragments)
        self.fragment_colors = ti.Vector.field(4, dtype=ti.f32, shape=max_fragments)
        self.fragment_depths = ti.field(dtype=ti.f32, shape=max_fragments)
        self.fragment_count = ti.field(dtype=ti.i32, shape=())

        # Linked list structure for fragments
        self.fragment_next = ti.field(dtype=ti.i32, shape=max_fragments)  # Next fragment in linked list
        self.pixel_head = ti.field(dtype=ti.i32, shape=(width, height))  # Head of linked list for each pixel

        # Spinlock for each pixel to ensure atomic linked list operations
        self.pixel_lock = ti.field(dtype=ti.i32, shape=(width, height))  # 0 = unlocked, >0 = locked

        # Frame buffer for visualization (RGBA)
        self.framebuffer = ti.Vector.field(4, dtype=ti.f32, shape=(width, height))

        # Initialize projection matrix
        self.set_projection_matrix()

    def allocate_triangles(self, n_triangles: int):
        """Allocate memory for triangle data."""
        self.triangles = ti.Vector.field(3, dtype=ti.f32, shape=(n_triangles, 3))
        self.colors = ti.Vector.field(4, dtype=ti.f32, shape=(n_triangles, 3))

    def set_projection_matrix(self, fov: float = 60.0, near: float = 0.1, far: float = 100.0):
        """
        Set up a perspective projection matrix.

        Args:
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        aspect = self.width / self.height
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        self.projection_matrix[None] = proj

    def set_orthographic_projection(self, left: float = -1, right: float = 1,
                                    bottom: float = -1, top: float = 1,
                                    near: float = -1, far: float = 1):
        """
        Set up an orthographic projection matrix.
        """
        proj = np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.projection_matrix[None] = proj

    @ti.func
    def project_vertex(self, vertex: ti.math.vec3) -> ti.math.vec4:
        """Project 3D vertex to normalized device coordinates."""
        v_homo = ti.Vector([vertex[0], vertex[1], vertex[2], 1.0])
        v_proj = self.projection_matrix[None] @ v_homo

        if ti.abs(v_proj[3]) > 1e-6:
            v_proj = v_proj / v_proj[3]

        x = (v_proj[0] + 1.0) * 0.5 * self.width
        y = (v_proj[1] + 1.0) * 0.5 * self.height
        z = v_proj[2]
        w = v_proj[3]

        return ti.Vector([x, y, z, w])

    @ti.func
    def edge_function(self, a: ti.math.vec2, b: ti.math.vec2, c: ti.math.vec2) -> ti.f32:
        """Compute edge function for point c with respect to edge (a, b)."""
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    @ti.func
    def compute_barycentric(self, p: ti.math.vec2, v0: ti.math.vec2,
                            v1: ti.math.vec2, v2: ti.math.vec2) -> ti.math.vec3:
        """Compute barycentric coordinates of point p in triangle (v0, v1, v2)."""
        area = self.edge_function(v0, v1, v2)
        result = ti.Vector([0.0, 0.0, 0.0])
        if ti.abs(area) >= 1e-6:
            w0 = self.edge_function(v1, v2, p) / area
            w1 = self.edge_function(v2, v0, p) / area
            w2 = self.edge_function(v0, v1, p) / area
            result = ti.Vector([w0, w1, w2])
        return result

    @ti.kernel
    def rasterize(self, n_triangles: ti.i32):
        """
        Rasterize all triangles and generate fragments with linked lists per pixel.

        Args:
            n_triangles: Number of triangles to rasterize
        """
        # Reset fragment counter
        self.fragment_count[None] = 0

        # Initialize pixel heads to -1 (empty list) and locks to 0 (unlocked)
        for i, j in self.pixel_head:
            self.pixel_head[i, j] = -1
            self.pixel_lock[i, j] = 0

        # Initialize fragment next pointers to -1
        for i in range(self.max_fragments):
            self.fragment_next[i] = -1

        # Clear framebuffer
        for i, j in self.framebuffer:
            self.framebuffer[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        # Process each triangle
        for tri_idx in range(n_triangles):
            # Get triangle vertices in 3D
            v0_3d = self.triangles[tri_idx, 0]
            v1_3d = self.triangles[tri_idx, 1]
            v2_3d = self.triangles[tri_idx, 2]

            # Project vertices to screen space
            v0_proj = self.project_vertex(v0_3d)
            v1_proj = self.project_vertex(v1_3d)
            v2_proj = self.project_vertex(v2_3d)

            # Extract 2D positions for rasterization
            v0 = ti.Vector([v0_proj[0], v0_proj[1]])
            v1 = ti.Vector([v1_proj[0], v1_proj[1]])
            v2 = ti.Vector([v2_proj[0], v2_proj[1]])

            # Extract depths
            d0 = v0_proj[2]
            d1 = v1_proj[2]
            d2 = v2_proj[2]

            # Get vertex colors (RGBA)
            c0 = self.colors[tri_idx, 0]
            c1 = self.colors[tri_idx, 1]
            c2 = self.colors[tri_idx, 2]

            # Compute bounding box
            min_x = ti.max(0, ti.cast(ti.min(v0[0], v1[0], v2[0]), ti.i32))
            max_x = ti.min(self.width - 1, ti.cast(ti.max(v0[0], v1[0], v2[0]), ti.i32) + 1)
            min_y = ti.max(0, ti.cast(ti.min(v0[1], v1[1], v2[1]), ti.i32))
            max_y = ti.min(self.height - 1, ti.cast(ti.max(v0[1], v1[1], v2[1]), ti.i32) + 1)

            # Rasterize pixels in bounding box
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    p = ti.Vector([x + 0.5, y + 0.5])  # Pixel center

                    # Compute barycentric coordinates
                    bary = self.compute_barycentric(p, v0, v1, v2)

                    # Check if point is inside triangle
                    if bary[0] >= 0 and bary[1] >= 0 and bary[2] >= 0:
                        # Interpolate color (RGBA)
                        color = bary[0] * c0 + bary[1] * c1 + bary[2] * c2

                        # Interpolate depth
                        depth = bary[0] * d0 + bary[1] * d1 + bary[2] * d2

                        # Allocate a new fragment
                        frag_idx = ti.atomic_add(self.fragment_count[None], 1)

                        if frag_idx < self.max_fragments:
                            # Store fragment data
                            self.fragment_coords[frag_idx] = ti.Vector([x, y])
                            self.fragment_colors[frag_idx] = color
                            self.fragment_depths[frag_idx] = depth

                            # Use spinlock to safely add fragment to pixel's linked list
                            # The spinlock works as follows:
                            # - Lock is 0 when free
                            # - atomic_add(1) returns the OLD value
                            # - If old value was 0, we got the lock (it's now 1)
                            # - To release, atomic_add(-1) to decrement back to 0

                            lock_acquired = False
                            max_attempts = 1000

                            for attempt in range(max_attempts):
                                # Try to acquire lock
                                old_val = ti.atomic_add(self.pixel_lock[x, y], 1)

                                if old_val == 0:
                                    # We got the lock! (changed from 0 to 1)
                                    # Add fragment to linked list
                                    old_head = self.pixel_head[x, y]
                                    self.fragment_next[frag_idx] = old_head
                                    self.pixel_head[x, y] = frag_idx

                                    # Release lock by decrementing
                                    ti.atomic_add(self.pixel_lock[x, y], -1)
                                    lock_acquired = True
                                    break
                                else:
                                    # Someone else has the lock, undo our add
                                    ti.atomic_add(self.pixel_lock[x, y], -1)
                                    # Small busy wait to reduce contention
                                    # (Taichi doesn't have a yield, so we just continue)

    @ti.kernel
    def blend_fragments(self):
        """
        Sort and blend fragments for each pixel using linked lists.
        """
        # Process each pixel
        for x, y in self.framebuffer:
            # Get head of linked list for this pixel
            head = self.pixel_head[x, y]

            if head >= 0:  # If there are fragments at this pixel
                # Count fragments in the list
                count = 0
                current = head
                while current >= 0 and count < 100:  # Limit to prevent infinite loops
                    count += 1
                    current = self.fragment_next[current]

                # Collect fragments and their depths (using local arrays)
                # Note: Taichi doesn't support dynamic arrays, so we use a fixed size
                local_indices = ti.Vector([0 for _ in range(32)], dt=ti.i32)
                local_depths = ti.Vector([0.0 for _ in range(32)], dt=ti.f32)

                # Collect fragments
                actual_count = ti.min(count, 32)
                current = head
                for i in range(actual_count):
                    local_indices[i] = current
                    local_depths[i] = self.fragment_depths[current]
                    current = self.fragment_next[current]

                # Sort fragments by depth (bubble sort - simple and works for small lists)
                for i in range(actual_count):
                    for j in range(actual_count - 1 - i):
                        if local_depths[j] < local_depths[j + 1]:  # Sort back to front
                            # Swap depths
                            temp_depth = local_depths[j]
                            local_depths[j] = local_depths[j + 1]
                            local_depths[j + 1] = temp_depth
                            # Swap indices
                            temp_idx = local_indices[j]
                            local_indices[j] = local_indices[j + 1]
                            local_indices[j + 1] = temp_idx

                # Blend fragments in sorted order (back to front)
                final_color = ti.Vector([0.0, 0.0, 0.0, 0.0])
                for i in range(actual_count):
                    frag_idx = local_indices[i]
                    src_color = self.fragment_colors[frag_idx]

                    # Standard alpha blending
                    src_alpha = src_color[3]
                    dst_alpha = final_color[3]

                    out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

                    if out_alpha > 1e-6:
                        final_color[0] = (src_color[0] * src_alpha + final_color[0] * dst_alpha * (
                                    1 - src_alpha)) / out_alpha
                        final_color[1] = (src_color[1] * src_alpha + final_color[1] * dst_alpha * (
                                    1 - src_alpha)) / out_alpha
                        final_color[2] = (src_color[2] * src_alpha + final_color[2] * dst_alpha * (
                                    1 - src_alpha)) / out_alpha
                        final_color[3] = out_alpha

                # Write final color to framebuffer
                self.framebuffer[x, y] = final_color
            else:
                # No fragments at this pixel
                self.framebuffer[x, y] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    @ti.kernel
    def copy_triangles_to_fields(self, triangles_np: ti.types.ndarray(),
                                 colors_np: ti.types.ndarray(), n_triangles: ti.i32):
        """Copy numpy arrays to Taichi fields using a kernel."""
        for i, j in ti.ndrange(n_triangles, 3):
            for k in ti.static(range(3)):
                self.triangles[i, j][k] = triangles_np[i, j, k]
            for k in ti.static(range(4)):
                self.colors[i, j][k] = colors_np[i, j, k]

    def process_batch(self, triangles_tensor: torch.Tensor,
                      colors_tensor: torch.Tensor,
                      sort_and_blend: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a batch of 3D triangles with RGBA colors.

        Args:
            triangles_tensor: Tensor of shape (n_triangles, 3, 3) with 3D vertex positions
            colors_tensor: Tensor of shape (n_triangles, 3, 4) with RGBA vertex colors
            sort_and_blend: Whether to sort fragments by depth and apply alpha blending

        Returns:
            fragment_coords: Tensor of shape (n_fragments, 2) with pixel coordinates
            fragment_colors: Tensor of shape (n_fragments, 4) with RGBA colors
            fragment_depths: Tensor of shape (n_fragments,) with depth values
        """
        n_triangles = triangles_tensor.shape[0]

        # Allocate memory for triangles
        self.allocate_triangles(n_triangles)

        # Copy data to Taichi fields
        triangles_np = triangles_tensor.cpu().numpy()
        colors_np = colors_tensor.cpu().numpy()

        # Use kernel to copy data efficiently
        self.copy_triangles_to_fields(triangles_np, colors_np, n_triangles)

        # Rasterize (builds linked lists)
        self.rasterize(n_triangles)

        # Blend fragments if requested
        if sort_and_blend:
            self.blend_fragments()

        # Get fragment count
        n_fragments = self.fragment_count[None]

        if n_fragments > 0:
            # Get fragment data
            coords_np = self.fragment_coords.to_numpy()[:n_fragments]
            colors_np = self.fragment_colors.to_numpy()[:n_fragments]
            depths_np = self.fragment_depths.to_numpy()[:n_fragments]

            fragment_coords = torch.from_numpy(coords_np).to(triangles_tensor.device)
            fragment_colors = torch.from_numpy(colors_np).to(triangles_tensor.device)
            fragment_depths = torch.from_numpy(depths_np).to(triangles_tensor.device)
        else:
            fragment_coords = torch.empty((0, 2), dtype=torch.int32, device=triangles_tensor.device)
            fragment_colors = torch.empty((0, 4), dtype=torch.float32, device=triangles_tensor.device)
            fragment_depths = torch.empty((0,), dtype=torch.float32, device=triangles_tensor.device)

        return fragment_coords, fragment_colors, fragment_depths

    def get_framebuffer(self) -> np.ndarray:
        """Get the framebuffer as a numpy array for visualization."""
        return self.framebuffer.to_numpy()


def create_test_triangles_3d(n_triangles: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random test triangles in 3D space with RGBA colors.

    Returns:
        triangles: Tensor of shape (n_triangles, 3, 3) - 3D coordinates
        colors: Tensor of shape (n_triangles, 3, 4) - RGBA colors
    """
    triangles = torch.rand(n_triangles, 3, 3) * 2 - 1  # Range [-1, 1]
    triangles[:, :, 2] = triangles[:, :, 2] * 0.5 + 2  # Z in [1.5, 2.5]

    colors = torch.rand(n_triangles, 3, 4)

    return triangles, colors


def create_overlapping_triangles() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create overlapping triangles to demonstrate alpha blending and depth sorting.
    """
    triangles = torch.tensor([
        # Red triangle (back) - largest
        [[-0.8, -0.8, 3.0], [0.8, -0.8, 3.0], [0.0, 0.6, 3.0]],
        # Green triangle (middle) - medium
        [[-0.6, -0.4, 2.5], [0.6, -0.4, 2.5], [0.0, 0.5, 2.5]],
        # Blue triangle (front) - smallest
        [[-0.4, 0.0, 2.0], [0.4, 0.0, 2.0], [0.0, 0.6, 2.0]]
    ], dtype=torch.float32)

    colors = torch.tensor([
        # Red triangle with 0.5 alpha
        [[1.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5]],
        # Green triangle with 0.5 alpha
        [[0.0, 1.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5]],
        # Blue triangle with 0.5 alpha
        [[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 0.5]]
    ], dtype=torch.float32)

    return triangles, colors


def main():
    """Example usage of the 3D triangle rasterizer with alpha blending."""

    print("=== 3D Triangle Rasterization with Per-Pixel Linked Lists ===\n")

    # Create rasterizer
    width, height = 800, 600
    rasterizer = TriangleRasterizer3D(width=width, height=height)

    # Use perspective projection
    rasterizer.set_projection_matrix(fov=60, near=0.1, far=10.0)

    # Test 1: Random triangles
    print("Test 1: Random 3D triangles")
    n_triangles = 10
    triangles, colors = create_test_triangles_3d(n_triangles)

    print(f"Input triangles shape: {triangles.shape}")
    print(f"Input colors shape: {colors.shape}")

    # Rasterize with sorting and blending
    fragment_coords, fragment_colors, fragment_depths = rasterizer.process_batch(
        triangles, colors, sort_and_blend=True
    )

    print(f"\nOutput fragments:")
    print(f"Fragment coordinates shape: {fragment_coords.shape}")
    print(f"Fragment colors shape: {fragment_colors.shape}")
    print(f"Fragment depths shape: {fragment_depths.shape}")
    print(f"Total fragments generated: {fragment_coords.shape[0]}")

    if fragment_depths.shape[0] > 0:
        print(f"Depth range: [{fragment_depths.min():.3f}, {fragment_depths.max():.3f}]")

    # Test 2: Overlapping triangles with transparency
    print("\n\nTest 2: Overlapping triangles with transparency")
    triangles2, colors2 = create_overlapping_triangles()

    # Use orthographic projection for clearer overlap visualization
    rasterizer.set_orthographic_projection(left=-1.5, right=1.5, bottom=-1.5, top=1.5, near=1.0, far=5.0)

    fragment_coords2, fragment_colors2, fragment_depths2 = rasterizer.process_batch(
        triangles2, colors2, sort_and_blend=True
    )

    print(f"Generated {fragment_coords2.shape[0]} fragments")
    print(f"Using per-pixel linked lists for correct depth sorting")

    # Visualization
    try:
        import matplotlib.pyplot as plt

        framebuffer = rasterizer.get_framebuffer()

        # Convert RGBA to RGB for display
        rgb_buffer = framebuffer[:, :, :3]
        alpha_buffer = framebuffer[:, :, 3:4]

        # Create a checkerboard background to show transparency
        checkerboard = np.zeros((width, height, 3))
        checker_size = 20
        for i in range(0, width, checker_size):
            for j in range(0, height, checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    checkerboard[i:i + checker_size, j:j + checker_size] = 0.8
                else:
                    checkerboard[i:i + checker_size, j:j + checker_size] = 0.95

        # Composite over checkerboard background
        composited = rgb_buffer * alpha_buffer + checkerboard * (1 - alpha_buffer)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(composited)
        plt.title("3D Triangles with Linked-List Alpha Blending")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(framebuffer[:, :, 3], cmap='gray', vmin=0, vmax=1)
        plt.title("Alpha Channel")
        plt.axis('off')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(rgb_buffer)
        plt.title("RGB (without background)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('rasterized_3d_triangles.png', dpi=150, bbox_inches='tight')
        print("\nFramebuffer saved as 'rasterized_3d_triangles.png'")
        print("Linked list approach ensures correct ordering without race conditions!")
        plt.show()

    except ImportError:
        print("\nMatplotlib not available for visualization")

    return fragment_coords2, fragment_colors2, fragment_depths2


if __name__ == "__main__":
    fragment_coords, fragment_colors, fragment_depths = main()

    print("\n\n=== Bonus: Cube rendering ===")


    def create_cube_triangles():
        """Create a cube from triangles."""
        vertices = torch.tensor([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],  # Back
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]  # Front
        ], dtype=torch.float32)

        vertices[:, 2] += 3.0

        indices = [
            [4, 5, 6], [4, 6, 7],  # Front face
            [1, 0, 3], [1, 3, 2],  # Back face
            [7, 6, 2], [7, 2, 3],  # Top face
            [0, 1, 5], [0, 5, 4],  # Bottom face
            [5, 1, 2], [5, 2, 6],  # Right face
            [0, 4, 7], [0, 7, 3]  # Left face
        ]

        triangles = torch.stack([vertices[idx] for idx in indices])

        face_colors = torch.tensor([
            [1.0, 0.0, 0.0, 0.8],  # Red
            [0.0, 1.0, 0.0, 0.8],  # Green
            [0.0, 0.0, 1.0, 0.8],  # Blue
            [1.0, 1.0, 0.0, 0.8],  # Yellow
            [1.0, 0.0, 1.0, 0.8],  # Magenta
            [0.0, 1.0, 1.0, 0.8],  # Cyan
        ], dtype=torch.float32)

        colors = []
        for i in range(6):
            color = face_colors[i]
            colors.extend([[color, color, color], [color, color, color]])
        colors = torch.stack(colors)

        return triangles, colors


    rasterizer = TriangleRasterizer3D(width=800, height=600)
    cube_triangles, cube_colors = create_cube_triangles()

    print(f"Cube triangles shape: {cube_triangles.shape}")
    coords, colors, depths = rasterizer.process_batch(cube_triangles, cube_colors, sort_and_blend=True)
    print(f"Rendered {coords.shape[0]} fragments from cube")