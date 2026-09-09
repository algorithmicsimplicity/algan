"""Shared, data-driven UV mip sampling and a stateless pixel-cone footprint."""

from algan.taichi_compat import ti


@ti.func
def _mip_table(meta: ti.template(), idx, slot):
    table = -1
    # Old synthetic/extension metadata remains a valid level-zero-only bank.
    if meta.shape[1] >= 21:
        table = meta[idx, 18 + slot]
    return table


@ti.func
def _mip_lod(table, width, height, du, dv, textures: ti.template()):
    lod = 0.0
    if table >= 0:
        last = ti.bit_cast(textures[0, table, 4], ti.i32)
        wrap = ti.bit_cast(textures[0, table, 0], ti.i32)
        rho = ti.max(ti.abs(du) * (width - (wrap & 1)),
                     ti.abs(dv) * (height - ((wrap >> 1) & 1)))
        lod = ti.min(ti.log(ti.max(rho, 1.0)) * 1.4426950408889634,
                     ti.cast(last, ti.f32))
    return lod


@ti.func
def _mip_bilinear(f, u, v, table, level, textures: ti.template()):
    row = table + level
    offset = ti.bit_cast(textures[0, row, 0], ti.i32)
    width = ti.bit_cast(textures[0, row, 1], ti.i32)
    height = ti.bit_cast(textures[0, row, 2], ti.i32)
    frames = ti.bit_cast(textures[0, row, 3], ti.i32)
    wrap = ti.bit_cast(textures[0, table, 0], ti.i32)
    # Pixel-centred reduced grids span the same unit-square image domain.
    x = ti.math.clamp(u, 0.0, 1.0) * width - 0.5
    y = ti.math.clamp(v, 0.0, 1.0) * height - 0.5
    # Wrapped base texels live at i/N, not (i+.5)/N. This half-base-texel
    # correction keeps their phase across levels; a mip has no seam padding.
    if (wrap & 1) != 0:
        base_w = ti.bit_cast(textures[0, table, 1], ti.i32) - 1
        x = (ti.math.clamp(u, 0.0, 1.0) + 0.5 / base_w) * width - 0.5
    if (wrap & 2) != 0:
        base_h = ti.bit_cast(textures[0, table, 2], ti.i32) - 1
        y = (ti.math.clamp(v, 0.0, 1.0) + 0.5 / base_h) * height - 0.5
    ix, iy = ti.cast(ti.floor(x), ti.i32), ti.cast(ti.floor(y), ti.i32)
    fx, fy = x - ti.floor(x), y - ti.floor(y)
    base = offset + (f % frames) * width * height
    out = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    for corner in ti.static(range(4)):
        cx = ti.math.clamp(ix + corner % 2, 0, width - 1)
        cy = ti.math.clamp(iy + corner // 2, 0, height - 1)
        if (wrap & 1) != 0:
            cx = (ix + corner % 2 + width) % width
        if (wrap & 2) != 0:
            cy = (iy + corner // 2 + height) % height
        weight = (fx if corner % 2 else 1.0 - fx) * (fy if corner // 2 else 1.0 - fy)
        for c in ti.static(range(5)):
            out[c] += weight * textures[f % textures.shape[0], base + cx * height + cy, c]
    return out


@ti.func
def _mip_blend(f, u, v, table, lod, base_value, textures: ti.template()):
    """Trilinear fetch; base_value is needed only while 0 <= lod < 1."""
    out = base_value
    if table >= 0 and lod > 0.0:
        low = ti.cast(ti.floor(lod), ti.i32)
        last = ti.bit_cast(textures[0, table, 4], ti.i32)
        high = ti.min(low + 1, last)
        if low > 0:
            out = _mip_bilinear(f, u, v, table, low, textures)
        if high != low:
            other = _mip_bilinear(f, u, v, table, high, textures)
            out += (lod - low) * (other - out)
    return out


@ti.func
def _triangle_uv_footprint(mem_trim: ti.template(), f, prim, rd, diameter,
                           positions: ti.template(), uvs: ti.template(),
                           meta: ti.template(), num_colored: ti.template()):
    """UV widths of a circular ray footprint projected onto the hit plane.

    Width = camera pixel angle * accumulated path length. A planar mirror
    therefore keeps the virtual-camera footprint without extra ray state.
    Isotropic/conservative: no anisotropic taps, curvature or refractive focus.
    """
    du, dv = 0.0, 0.0
    idx = prim - num_colored
    if ti.static(mem_trim != 0):
        idx = prim
    if meta.shape[1] >= 21:
        if idx >= 0:
            if meta[idx, 18] >= 0 or meta[idx, 19] >= 0 or meta[idx, 20] >= 0:
                tp, tu = f % positions.shape[0], f % uvs.shape[0]
                e1 = ti.math.vec3(positions[tp, prim, 3] - positions[tp, prim, 0],
                                 positions[tp, prim, 4] - positions[tp, prim, 1],
                                 positions[tp, prim, 5] - positions[tp, prim, 2])
                e2 = ti.math.vec3(positions[tp, prim, 6] - positions[tp, prim, 0],
                                 positions[tp, prim, 7] - positions[tp, prim, 1],
                                 positions[tp, prim, 8] - positions[tp, prim, 2])
                n = e1.cross(e2)
                n2 = n.dot(n)
                if n2 > 1e-30:
                    b1, b2 = e2.cross(n) / n2, n.cross(e1) / n2
                    gu = (uvs[tu, idx, 2] - uvs[tu, idx, 0]) * b1 + (uvs[tu, idx, 4] - uvs[tu, idx, 0]) * b2
                    gv = (uvs[tu, idx, 3] - uvs[tu, idx, 1]) * b1 + (uvs[tu, idx, 5] - uvs[tu, idx, 1]) * b2
                    n /= ti.sqrt(n2)
                    cosine = n.dot(rd)
                    safe_cos = ti.max(ti.abs(cosine), 1e-6)
                    if cosine < 0.0:
                        safe_cos = -safe_cos
                    gu -= n * (gu.dot(rd) / safe_cos)
                    gv -= n * (gv.dot(rd) / safe_cos)
                    du = ti.min(ti.max(diameter, 0.0) * gu.norm(), 1e10)
                    dv = ti.min(ti.max(diameter, 0.0) * gv.norm(), 1e10)
    return du, dv
