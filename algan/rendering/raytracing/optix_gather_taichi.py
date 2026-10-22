"""The Taichi half of the OptiX gather: merging in the circuits' hits.

The OptiX gather traces triangles only. In a batch with Bezier circuits the
circuits keep the software walk -- ``wavefront_traverse_events`` with the
triangle tree compiled out -- into a second k-buffer, and this kernel merges
the two: the first ``kbuf`` hits of their union in ``_comes_after`` order.

That is ``_collect_hits``' answer. Each buffer holds the first ``kbuf`` hits
of its own type (or all of them, when it is not full), so a hit missing from
one comes after everything that buffer kept, and the union's first ``kbuf``
can never reach past a full buffer's last hit. The count keeps the contract
the shade relies on: below ``kbuf`` only when both buffers were complete. A
buffer cut after an opaque hit is complete for every hit the shade can reach,
since nothing after an opaque hit is consumed.
"""
from algan.rendering.raytracing.raytrace_kernels_taichi import _comes_after, kbuf
from algan.taichi_compat import ti


@ti.kernel
def optix_gather_merge(
        active: ti.types.ndarray(), num_active: int,
        rs_int: ti.types.ndarray(), tri_count: ti.types.ndarray(),
        tri_f: ti.types.ndarray(), tri_i: ti.types.ndarray(),
        hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray()):
    """Merge the triangle k-buffer into the circuits' one.

    ``tri_*`` / ``tri_count`` hold the OptiX gather's triangle hits; ``hit_*``
    / ``rs_int[:, 3]`` hold what the software traversal gathered of the
    circuits, and receive the merged buffer.
    """
    for i in range(num_active):
        r = active[i]
        n_t = tri_count[i]
        n_c = rs_int[r, 3]
        # Gather both into one candidate list (at most 2 * kbuf).
        ct = ti.Vector([0.0] * (2 * kbuf))
        cl = ti.Vector([0.0] * (2 * kbuf))
        ca = ti.Vector([0.0] * (2 * kbuf))
        cb = ti.Vector([0.0] * (2 * kbuf))
        cp = ti.Vector([0] * (2 * kbuf))
        cf = ti.Vector([0] * (2 * kbuf))
        cv = ti.Vector([0] * (2 * kbuf))
        for q in ti.static(range(kbuf)):
            if q < n_t:
                ct[q] = tri_f[q, 0, i]
                cl[q] = tri_f[q, 1, i]
                ca[q] = tri_f[q, 2, i]
                cb[q] = tri_f[q, 3, i]
                cp[q] = tri_i[q, 0, i]
                cf[q] = tri_i[q, 1, i]
                cv[q] = 1
            if q < n_c:
                ct[kbuf + q] = hit_f[q, 0, i]
                cl[kbuf + q] = hit_f[q, 1, i]
                ca[kbuf + q] = hit_f[q, 2, i]
                cb[kbuf + q] = hit_f[q, 3, i]
                cp[kbuf + q] = hit_i[q, 0, i]
                cf[kbuf + q] = hit_i[q, 1, i]
                cv[kbuf + q] = 1
        count = 0
        out_t = ti.Vector([0.0] * kbuf)
        out_l = ti.Vector([0.0] * kbuf)
        out_a = ti.Vector([0.0] * kbuf)
        out_b = ti.Vector([0.0] * kbuf)
        out_p = ti.Vector([0] * kbuf)
        out_f = ti.Vector([0] * kbuf)
        for q in ti.static(range(kbuf)):
            # The next candidate in order: selected by value, never by a
            # dynamic vector index (that spills the vectors to local memory).
            sel_t = 0.0
            sel_l = 0.0
            sel_a = 0.0
            sel_b = 0.0
            sel_p = 0
            sel_f = 0
            found = 0
            for c in ti.static(range(2 * kbuf)):
                if cv[c] == 1:
                    if (found == 0) or _comes_after(sel_t, sel_l, ct[c], cl[c]):
                        sel_t = ct[c]
                        sel_l = cl[c]
                        sel_a = ca[c]
                        sel_b = cb[c]
                        sel_p = cp[c]
                        sel_f = cf[c]
                        found = 1
            if found == 1:
                for c in ti.static(range(2 * kbuf)):
                    if (cv[c] == 1) and (ct[c] == sel_t) and (cl[c] == sel_l):
                        cv[c] = 0
                out_t[q] = sel_t
                out_l[q] = sel_l
                out_a[q] = sel_a
                out_b[q] = sel_b
                out_p[q] = sel_p
                out_f[q] = sel_f
                count += 1
        rs_int[r, 3] = count
        if count > 0:
            for q in ti.static(range(kbuf)):
                hit_f[q, 0, i] = out_t[q]
                hit_f[q, 1, i] = out_l[q]
                hit_f[q, 2, i] = out_a[q]
                hit_f[q, 3, i] = out_b[q]
                hit_i[q, 0, i] = out_p[q]
                hit_i[q, 1, i] = out_f[q]
