import taichi as ti
import torch

ti.init(arch=ti.gpu)

@ti.func
def edge_function(a: ti.math.vec2, b: ti.math.vec2, c: ti.math.vec2) -> ti.f32:
    """Compute edge function for point c with respect to edge (a, b)."""
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

@ti.func
def compute_barycentric(p: ti.math.vec2, v0: ti.math.vec2,
                        v1: ti.math.vec2, v2: ti.math.vec2) -> ti.math.vec3:
    """Compute barycentric coordinates of point p in triangle (v0, v1, v2)."""
    area = edge_function(v0, v1, v2)
    result = ti.Vector([-1.0, -1.0, -1.0])
    if ti.abs(area) >= 1e-12:
        w0 = edge_function(v1, v2, p) / area
        w1 = edge_function(v2, v0, p) / area
        w2 = edge_function(v0, v1, p) / area
        result = ti.Vector([w0, w1, w2])
    return result

@ti.kernel
def rasterize_triangle_taichi(triangles_2d: ti.types.ndarray(), colors: ti.types.ndarray(), distances: ti.types.ndarray(),
              out_buffer: ti.types.ndarray(), out_ind_buffer: ti.types.ndarray(),
              width: int, height: int, fragment_count: ti.types.ndarray(), num_triangles: int):
    fragment_count[0] = 0

    # Process each triangle
    for tri_idx in range(triangles_2d.shape[0]):
        # Get triangle vertices in 3D
        v00 = triangles_2d[tri_idx, 0, 0]
        v10 = triangles_2d[tri_idx, 1, 0]
        v20 = triangles_2d[tri_idx, 2, 0]
        v01 = triangles_2d[tri_idx, 0, 1]
        v11 = triangles_2d[tri_idx, 1, 1]
        v21 = triangles_2d[tri_idx, 2, 1]
        d0 = distances[tri_idx, 0, 0]
        d1 = distances[tri_idx, 1, 0]
        d2 = distances[tri_idx, 2, 0]

        # Compute bounding box
        min_x = ti.max(0, ti.cast(ti.min(v00, v10, v20), ti.i32))
        max_x = ti.min(width - 1, ti.cast(ti.max(v00, v10, v20), ti.i32) + 1)
        min_y = ti.max(0, ti.cast(ti.min(v01, v11, v21), ti.i32))
        max_y = ti.min(height - 1, ti.cast(ti.max(v01, v11, v21), ti.i32) + 1)

        # Rasterize pixels in bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = ti.Vector([x + 0.5, y + 0.5])  # Pixel center

                # Compute barycentric coordinates
                bary = compute_barycentric(p, ti.Vector([v00, v01]), ti.Vector([v10, v11]), ti.Vector([v20, v21]))

                # Check if point is inside triangle. The small tolerance lets
                # adjacent triangles overlap on their shared edge instead of
                # excluding each other: with exact tests, floating-point noise
                # can make a pixel on the edge fail both triangles, leaving
                # crack pixels along mesh seams.
                if bary[0] >= -1e-5 and bary[1] >= -1e-5 and bary[2] >= -1e-5:
                    frag_idx = ti.atomic_add(fragment_count[0], 1)
                    if frag_idx < out_buffer.shape[0]:
                        #if frag_idx < out_buffer.shape[0]:
                        # Interpolate color (RGBA)
                        for ci in range(5):
                            color = bary[0] * colors[tri_idx, 0, ci] + bary[1] * colors[tri_idx, 1, ci] + bary[2] * colors[tri_idx, 2, ci]
                            out_buffer[frag_idx, ci] = color

                        out_buffer[frag_idx, 5] = bary[0] * d0 + bary[1] * d1 + bary[2] * d2
                        out_ind_buffer[frag_idx] = x + y * width + (tri_idx // num_triangles) * (width*height)


if __name__ == '__main__':
    output_size = int(1e9 * 0.1)
    num_triangles = int(output_size / 1000)
    d = torch.device('cuda')
    triangles_2d = torch.randn((num_triangles, 3, 2), device=d) * 10 + 400
    colors = torch.randn((num_triangles, 3, 5), device=d)
    distances = torch.randn((num_triangles, 3, 1), device=d)
    out_buffer = torch.randn((output_size, 7), device=d)
    width = height = 800
    for i in range(100):
        fragment_count = torch.zeros((1,), dtype=torch.int, device=d)
        rasterize(triangles_2d, colors, distances,
                  out_buffer,
                  width, height, fragment_count)
        print('done')
    print(fragment_count)