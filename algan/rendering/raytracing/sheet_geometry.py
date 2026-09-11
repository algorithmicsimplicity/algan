"""Block-local floating-point work for sheet shading classes and depth bands."""

from __future__ import annotations

import torch

from algan.rendering.mps_compat import clamp_floor


def _frame_rows(table, frames, workspace):
    rows = workspace.tensor(frames.shape, torch.int64)
    torch.remainder(frames, table.shape[0], out=rows)
    return workspace.gather(table, rows)


def shade_class_block(tri_norm, tri_pos, frames, quant, out, workspace):
    """Write a table block, retaining each original norm/rounding operation.

    The caller owns out and absolute frame ordinals. Normal classification and
    geometric-normal scratch have separate lifetimes; no scratch view escapes.
    """
    shape = out.shape
    with workspace.stage():
        declared = workspace.tensor(shape, torch.bool)
        geometric = workspace.tensor(shape, torch.bool)
        vertex = workspace.tensor((*shape, 3), tri_norm.dtype)
        with workspace.stage():
            nrm = _frame_rows(tri_norm, frames, workspace).reshape(*shape, 3, 3)
            mag = workspace.tensor((*shape, 3), nrm.dtype)
            torch.norm(nrm, dim=3, out=mag)
            denom = workspace.tensor(mag.shape, mag.dtype)
            clamp_floor(mag, 1e-12, out=denom, workspace=workspace)
            unit = workspace.tensor(nrm.shape, nrm.dtype)
            torch.div(nrm, denom.unsqueeze(3), out=unit)
            vertex.copy_(unit[:, :, 0])
            delta = workspace.tensor(vertex.shape, vertex.dtype)
            spread = workspace.tensor(shape, vertex.dtype)
            tmp = workspace.tensor(shape, vertex.dtype)
            torch.sub(unit[:, :, 1], vertex, out=delta)
            delta.abs_()
            torch.amax(delta, dim=2, out=spread)
            torch.sub(unit[:, :, 2], vertex, out=delta)
            delta.abs_()
            torch.amax(delta, dim=2, out=tmp)
            torch.maximum(spread, tmp, out=spread)
            torch.amin(mag, dim=2, out=tmp)
            torch.gt(tmp, 1e-6, out=declared)
            same = workspace.tensor(shape, torch.bool)
            torch.lt(spread, 1e-6, out=same)
            declared.logical_and_(same)
            torch.amax(mag, dim=2, out=tmp)
            torch.lt(tmp, 1e-6, out=geometric)

        face = workspace.tensor(
            vertex.shape, torch.promote_types(tri_norm.dtype, tri_pos.dtype)
        )
        with workspace.stage():
            pos = _frame_rows(tri_pos, frames, workspace)
            e1 = workspace.tensor(vertex.shape, pos.dtype)
            e2 = workspace.tensor(vertex.shape, pos.dtype)
            torch.sub(pos[..., 3:6], pos[..., 0:3], out=e1)
            torch.sub(pos[..., 6:9], pos[..., 0:3], out=e2)
            gn = workspace.tensor(vertex.shape, pos.dtype)
            torch.cross(e1, e2, dim=-1, out=gn)
            mag = workspace.tensor((*shape, 1), pos.dtype)
            torch.norm(gn, dim=-1, keepdim=True, out=mag)
            clamp_floor(mag, 1e-12, out=mag, workspace=workspace)
            gn.div_(mag)
            torch.where(geometric.unsqueeze(-1), gn, vertex, out=face)

        # Multiplication, rounding, integer conversion and clamping remain
        # separate: fusing or moving the cast can change a class at a bin edge.
        face.mul_(float(quant)).round_()
        q = workspace.copy(face, torch.int64)
        q.clamp_(-quant, quant).add_(quant)
        torch.bitwise_left_shift(q[..., 0], 16, out=out)
        middle = workspace.tensor(shape, torch.int64)
        torch.bitwise_left_shift(q[..., 1], 8, out=middle)
        out.bitwise_or_(middle).bitwise_or_(q[..., 2]).add_(1)
        declared.logical_or_(geometric).logical_not_()
        out.masked_fill_(declared, 0)


def depth_slope_block(tri_pos, cam_origin, tri_screen, frames, out, workspace):
    """Write one slope-table block without retaining world/screen temporaries."""
    shape = out.shape
    dtype = torch.promote_types(tri_pos.dtype, cam_origin.dtype)
    with workspace.stage():
        extent = workspace.tensor(shape, dtype)
        with workspace.stage():
            pos = _frame_rows(tri_pos, frames, workspace)
            ro = _frame_rows(cam_origin, frames, workspace).view(shape[0], 1, 3)
            delta = workspace.tensor((*shape, 3), dtype)
            distance = workspace.tensor(shape, dtype)
            nearest = workspace.tensor(shape, dtype)
            for k in range(3):
                torch.sub(pos[..., 3 * k : 3 * k + 3], ro, out=delta)
                # linalg.norm(out=) still allocates its vector result before copying.
                # Call that same underlying vector norm with its actual out API.
                torch.linalg.vector_norm(delta, dim=-1, out=distance)
                if k == 0:
                    nearest.copy_(distance)
                    extent.copy_(distance)
                else:
                    torch.minimum(nearest, distance, out=nearest)
                    torch.maximum(extent, distance, out=extent)
            extent.sub_(nearest)

        if tri_screen is None or tri_screen.shape[2] < 10:
            out.copy_(extent)
            return
        with workspace.stage():
            scr = _frame_rows(tri_screen, frames, workspace)
            span_x = workspace.tensor(shape, scr.dtype)
            span_y = workspace.tensor(shape, scr.dtype)
            minimum = workspace.tensor(shape, scr.dtype)
            torch.amax(scr[..., 0:3], dim=-1, out=span_x)
            torch.amin(scr[..., 0:3], dim=-1, out=minimum)
            span_x.sub_(minimum)
            torch.amax(scr[..., 3:6], dim=-1, out=span_y)
            torch.amin(scr[..., 3:6], dim=-1, out=minimum)
            span_y.sub_(minimum)
            torch.maximum(span_x, span_y, out=span_x)
            span_x.clamp_min_(1.0)
            valid = workspace.tensor(shape, torch.bool)
            torch.gt(scr[..., 9], 0.5, out=valid)
            slope = workspace.tensor(shape, torch.promote_types(dtype, scr.dtype))
            torch.div(extent, span_x, out=slope)
            torch.where(valid, slope, extent, out=slope)
            # The old assignment rounded here to the triangle-position dtype.
            out.copy_(slope)
