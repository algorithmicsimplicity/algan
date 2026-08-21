"""The sheet resolve kernel (DESIGN_sheet_resolve.md P4-P6, Phase 2).

One thread per covered pixel composites its few depth-sorted SHEETS front to
back -- per-sample transmittance as a prefix product, per-sheet magnitude from
the sheet's own exact area -- and shades ONCE per sheet at its dominant
fragment. The aggregation already happened on the host (``sheets.py``), so
none of the fragment walk's machinery exists here: no run scan and no budget,
no seam de-duplication, no one-mesh cap or ink accounting, no engagement gate.
What remains is the per-sheet claim/occlusion arithmetic and the material
four-way split (shade / reflect / transmit / miss) with its continuations,
inherited from the fragment walk's ``raster_first_shade`` -- deleted in
Phase 4, so this kernel is the only copy.

Per-sheet semantics (kept in lockstep with ``sheets.resolve_pixel_reference``,
the sequential oracle, and pinned by the parity harness):

* AREAL sheets -- circuits, and donor-only sheets (empty sample union) --
  claim ``alpha * min(area, 1)`` uniformly over every sample.
* A FULL-union sheet takes ``corr = 1`` inside the ``_AA_FULL_DUST`` band
  (interior tilings stay bit-clean) and ``min(area, 1)`` outside it.
* A PARTIAL-union sheet takes ``corr = min(area, 1) / Q``; ``corr > 1``
  keeps the claim exact and redistributes the clamped occlusion residue onto
  the sheet's unowned samples immediately (the old rule B collapsed from walk
  state to per-record arithmetic -- ``_run_svis_write`` + ``_run_redistribute``
  with no pending flag and no cross-record feedback).

Launched only by the sparse covered-pixel path, with the same ray-state
contract as the walk -- retired pixels accumulate into ``pix_accum``, bounces
keep their slot, splits go to the shared pool through the same atomic
allocator (deterministic slot ids are P7). Shadows are two launches of this
one body (``mode`` 1 then 2, see the parameter comment): the event pass and
the shading pass share every line of transport, which is what retires the
old design's hand-maintained walk/shadow-walk lockstep.
"""

import taichi as ti

from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_ONE_MESH_BIT,
    _AA_SAMPLE_WEIGHT,
    _AA_SEC_JITTER,
    _AA_SLIVER_BIT,
    _AA_FULL_DUST,
    _GLOSSY_MIN_ROUGHNESS,
    _aa_dump_frag,
    _aa_dump_match,
    _aa_dump_terminal,
    _decode_bez_ref,
    _frag_t,
    _glossy_reflect,
    _glossy_rotation,
    _jittered_surface_sample,
    _pixel_footprint,
    _popcount_samples,
    _run_redistribute,
    _run_svis_write,
    _sec_positions,
    _spawn_pool_ray,
    _tri_shadow_normals,
    _tri_surface_point,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    _bezier_normal,
    _generate_ray,
    _sample_circuit_color_blend,
    _shade_tri_hit,
)
from algan.rendering.raytracing.shading_taichi import (
    _MID_UNLIT,
    _reflect_frame,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _material_reflectance,
    _mirror_share,
    _offset_transmitted_origin,
    _refract_ray,
    _sample_env_map,
    _tri_color_g,
    _tri_extra_g,
    _tri_ior_transmission_g,
    _tri_normal_g,
)


@ti.kernel
def sheet_resolve_shade(
        num_covered: int,
        sheet_offsets: ti.types.ndarray(),
        sheet_key: ti.types.ndarray(), sheet_ref: ti.types.ndarray(),
        sheet_ab: ti.types.ndarray(), sheet_cov: ti.types.ndarray(),
        sheet_msk: ti.types.ndarray(), sheet_cap: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        col_row: ti.types.ndarray(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        layer_offsets: ti.types.ndarray(),
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        tri_pids: ti.template(),
        refraction: ti.template(), skip_unlit_normal: ti.template(),
        has_bez: ti.template(),
        sec_aa: ti.template(), sec_min_energy: ti.f32,
        glossy: ti.template(),
        env_in_composite: ti.template(),
        # Shadow support (DESIGN_sheet_resolve.md §4.9). ``mode`` 0 is the
        # shadow-free resolve; 1 walks the IDENTICAL transport and writes one
        # candidate shadow event per accepted lit triangle sheet (no shading,
        # no spawns) into dense per-sheet tables — event identity is the
        # sheet index, so no counter and no atomics exist; 2 is the shading
        # resolve reading the traced per-event visibility. One kernel body
        # for all three is what makes a resolve/shadow desync structurally
        # impossible — the two fragment walks this replaces had to be kept
        # in lockstep by hand and grew a harness just to check it.
        mode: ti.template(),
        sheet_accept: ti.types.ndarray(),
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_msk: ti.types.ndarray(), event_dp: ti.types.ndarray(),
        sheet_event_id: ti.types.ndarray(), shadow_vis: ti.types.ndarray(),
        covered_idx: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(),
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(), rs_alloc: ti.types.ndarray(),
        dump: ti.template(), dump_out: ti.types.ndarray()):
    """Composite + shade each covered pixel's depth-sorted sheet list.

    Everything is indexed by covered ordinal ``t`` (the walk's ``compact``
    contract): the sheet CSR, ray state and accumulator rows are ``t``-local
    and ``covered_idx[t]`` is used only for the frame/pixel coordinate. The
    host zeroes ``pix_accum`` and pre-marks pool slots DONE, so retirement
    accumulates and only bounced pixels write ray state.
    """
    pixels_per_frame = width * height
    env_off = ti.cast(layer_offsets[1] + 0.5, ti.i32)
    env_w = ti.cast(layer_offsets[2] + 0.5, ti.i32)
    env_h = ti.cast(layer_offsets[3] + 0.5, ti.i32)
    env_intensity = layer_offsets[4]
    far_clip = layer_offsets[5]
    max_bounces = ti.cast(layer_offsets[6] + 0.5, ti.i32)
    for t in range(num_covered):
        r = t
        pixel = covered_idx[t]
        start = sheet_offsets[r]
        end = sheet_offsets[r + 1]
        total = end - start

        g = pixel
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                               gen_meta[2], gen_meta[3],
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        dmatch = False
        if ti.static(dump):
            dmatch = _aa_dump_match(dump_out, px, py, f)
        g_roff = 0.5
        g_aoff = 0.0
        if ti.static(glossy != 0):
            g_roff, g_aoff = _glossy_rotation(px, py, ti.static(glossy == 2))

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        # ONE-MESH ceiling bookkeeping: coverage this pixel's single mesh has
        # already committed. Only meaningful where the host flagged the pixel
        # (the _AA_ONE_MESH_BIT rides in every sheet's flags there).
        mesh_ink = 0.0
        # §4.4 band state: sheets of ONE band (the shading-class siblings the
        # compaction subdivides) claim against the same incoming visibility
        # and occlude ONCE, at the band's last sheet, by their summed
        # coverage factor. A NEGATIVE sheet coverage is the host's flag that
        # this band continues at the NEXT sheet of the walk, which is what
        # lets the sum ride in a register (sheets.py, ``_sibling_weights``).
        band_p = 0.0
        band_open = False
        base_dist = 0.0
        bounces_left = max_bounces
        processed = 0
        bounced = False
        done = False

        q = 0
        while q < total and processed < MAX_SURFACES_PER_RAY:
            idx = start + q
            t_hit = _frag_t(sheet_key[idx])
            prim_raw = sheet_ref[idx]
            a = sheet_ab[idx, 0]
            b = sheet_ab[idx, 1]
            cov = sheet_cov[idx]
            defer = cov < 0.0
            cov = ti.abs(cov)
            msk = sheet_msk[idx]
            in_border = 0.0
            q += 1
            if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, 0, 4, prim_raw, 0, 0,
                                      msk, cov, _popcount_samples(msk), 1.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, t_hit, svis)
                done = True
                break
            processed += 1

            is_bez = False
            if ti.static(has_bez):
                is_bez = prim_raw < 0
            if is_bez:
                circuit_decoded, border_decoded = _decode_bez_ref(prim_raw)
                prim_raw = circuit_decoded
                in_border = border_decoded
            d_kind = 0
            if ti.static(dump):
                if is_bez:
                    d_kind = 1

            # -- per-sheet coverage: claim from the exact area, position from
            # the sample union. corr rides in cfac exactly where the walk's
            # run correction did, so the downstream arithmetic (claim, tint
            # fraction, occlusion write) is the walk's with the run replaced
            # by one record.
            msk_low = msk & _AA_MASK_ALL
            sliver = (msk & _AA_SLIVER_BIT) != 0
            areal = is_bez or sliver or (msk_low == 0)
            area = ti.min(cov, 1.0)
            cfac = 1.0
            dens = 1.0
            nsm = _AA_NUM_SAMPLES
            slots = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
            if areal:
                dens = area
            else:
                pop = _popcount_samples(msk_low)
                nsm = pop
                if msk_low == _AA_MASK_ALL:
                    if ti.abs(1.0 - cov) > _AA_FULL_DUST:
                        cfac = area
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if ((msk_low >> s) & 1) == 0:
                            slots[s] = 0.0
                    cfac = area * ti.static(float(_AA_NUM_SAMPLES)) \
                        / ti.cast(pop, ti.f32)
            vis = 0.0
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                vis += slots[s] * svis[s]
            eff = vis * _AA_SAMPLE_WEIGHT * dens * cfac
            # ONE-MESH ceiling (kept from the walk as sheet DATA — see
            # sheets.py's docstring for the measurement that kept it): on a
            # single-opaque-mesh pixel the mesh may claim at most the larger
            # of its two sheets' exact areas in total, occlusion scaled with
            # the claim. 2.0 is the host's "no ceiling" sentinel.
            if (not is_bez) and ((msk & _AA_ONE_MESH_BIT) != 0) \
                    and (sheet_cap[idx] <= 1.0):
                room = ti.max(sheet_cap[idx] - mesh_ink, 0.0)
                if eff > room:
                    dens *= room / ti.max(eff, MIN_ALPHA)
                    eff = room
            # This sheet's contribution to its band's single occlusion write
            # (inert outside a band: there band_p IS the sheet's own factor).
            band_p += cfac * dens
            if eff <= MIN_ALPHA:
                if not defer:
                    # The band ends here with nothing left to claim -- its
                    # samples are already dark -- so drop the pending sum
                    # rather than carry it into the next band.
                    band_p = 0.0
                    band_open = False
                else:
                    band_open = True
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, d_kind, 1,
                                      prim_raw, 0, 0, msk, cov,
                                      _popcount_samples(msk), cfac, eff,
                                      0.0, 0.0, 0.0, 0.0, t_hit, svis)
                continue
            if not is_bez:
                mesh_ink += eff

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            rough = 0.0
            ior = 0.0
            T = 0.0
            albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
            prim = 0
            circuit = 0
            fetched_bez = False
            surf_pos = ro + t_hit * rd
            surf_rd = rd
            w0 = 1.0 - a - b
            if ti.static(has_bez):
                if is_bez:
                    fetched_bez = True
                    circuit = prim_raw
                    cm = f % circuit_meta.shape[0]
                    color, alpha = _sample_circuit_color_blend(
                        circuit, f, a, b, in_border, circuit_meta,
                        circuit_colors, circuit_border_colors)
                    albedo3 = ti.math.vec3(color[0], color[1], color[2])
                    reflectivity = circuit_meta[cm, circuit, _M_REFLECTIVITY]
                    rough = circuit_meta[cm, circuit, _M_ROUGHNESS]
                    ior = circuit_meta[cm, circuit, _M_IOR]
                    T = circuit_meta[cm, circuit, _M_TRANSMISSION]
            if not fetched_bez:
                prim = prim_raw
                surf_pos = _tri_surface_point(f, prim, w0, a, b, tri_pos)
                partial = msk_low != _AA_MASK_ALL
                if partial:
                    surf_rd = (surf_pos - ro).normalized()
                color, alpha = _tri_color_g(0, f, prim, w0, a, b, tri_colors,
                                            col_row, tri_uvs, tri_tex_meta,
                                            textures, num_colored_triangles)
                reflectivity, rough = _tri_extra_g(
                    0, f, prim, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                albedo3 = ti.math.vec3(color[0], color[1], color[2])
                if ti.static(frag_shading != 0 and mode != 1):
                    lvis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                    if ti.static(mode == 2):
                        event_id = sheet_event_id[idx]
                        if event_id >= 0:
                            for li in range(num_lights):
                                if li < MAX_SHADOW_LIGHTS:
                                    lvis[li] = shadow_vis[event_id, li]
                    sn = ti.math.vec3(0.0, 0.0, 0.0)
                    if ti.static(skip_unlit_normal != 0):
                        if tri_mat_id[f % tri_mat_id.shape[0], prim] \
                                != _MID_UNLIT:
                            sn = _tri_normal_g(
                                0, f, prim, w0, a, b, tri_norm, tri_pos,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                    else:
                        sn = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                    color = _shade_tri_hit(frag_pipelines, tri_pids,
                                           f, prim, a, b,
                                           surf_rd, surf_pos,
                                           tri_pos, sn,
                                           tri_mat_id, tri_mat,
                                           light_pos, light_col,
                                           num_lights, color,
                                           ti.static(1 if mode == 2 else 0),
                                           lvis)
                ior, T = _tri_ior_transmission_g(
                    0, f, prim, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                if ti.static(mode == 1):
                    # One candidate shadow event per accepted lit triangle
                    # sheet, at the sheet index — mirroring the fragment
                    # build's acceptance (reached the fetch, material not
                    # unlit) and its payload: position and normals at the
                    # dominant fragment, the 4-bit sub-pixel position mask
                    # with the material pipeline id above it, and the pixel
                    # footprint for soft sub-pixel sampling.
                    pid_e = tri_mat_id[f % tri_mat_id.shape[0], prim]
                    if pid_e != _MID_UNLIT:
                        snrm, fnrm = _tri_shadow_normals(
                            f, prim, a, b, surf_rd, tri_pos, tri_norm,
                            tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles)
                        sheet_accept[idx] = 1
                        for k in ti.static(range(3)):
                            event_pos[idx, k] = surf_pos[k]
                            event_snrm[idx, k] = snrm[k]
                            event_fnrm[idx, k] = fnrm[k]
                        event_frame[idx] = f
                        shadow_msk = 0xF
                        if (not sliver) and (msk_low != 0):
                            shadow_msk, _sn4 = _sec_positions(msk_low, 4)
                        event_msk[idx] = shadow_msk | (pid_e << 8)
                        if ti.static(sec_aa > 1):
                            dpx, dpy = _pixel_footprint(
                                f, px, py, gen_meta, surf_pos, fnrm,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y)
                            for k in ti.static(range(3)):
                                event_dp[idx, k] = dpx[k]
                                event_dp[idx, 3 + k] = dpy[k]

            mat_alpha = ti.math.clamp(alpha, 0.0, 1.0)
            alpha = ti.math.clamp(mat_alpha * eff, 0.0, 1.0)
            a_s = mat_alpha * dens
            # Inside a subdivided band the occlusion write is the BAND's, made
            # once at its last sheet: the summed coverage factor against the
            # material alpha. A lone sheet writes its own, unchanged.
            w_cfac = cfac
            w_a_s = a_s
            if defer or band_open:
                w_cfac = band_p
                w_a_s = mat_alpha
            band_open = defer
            T = ti.math.clamp(T, 0.0, 1.0)

            normal = ti.math.vec3(0.0, 0.0, 0.0)
            geo_normal = ti.math.vec3(0.0, 0.0, 0.0)
            if (reflectivity >= 0.0) or (T > 1e-4):
                if fetched_bez:
                    normal = _bezier_normal(
                        f, circuit, circuit_meta).normalized()
                    geo_normal = normal
                else:
                    normal = _tri_normal_g(
                        0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles
                    ).normalized()
                    gp = f % tri_pos.shape[0]
                    g0 = ti.math.vec3(tri_pos[gp, prim, 0],
                                      tri_pos[gp, prim, 1],
                                      tri_pos[gp, prim, 2])
                    g1 = ti.math.vec3(tri_pos[gp, prim, 3],
                                      tri_pos[gp, prim, 4],
                                      tri_pos[gp, prim, 5])
                    g2 = ti.math.vec3(tri_pos[gp, prim, 6],
                                      tri_pos[gp, prim, 7],
                                      tri_pos[gp, prim, 8])
                    geo_normal = (g1 - g0).cross(g2 - g0)

            R, diel_pass = _material_reflectance(surf_rd, normal,
                                                 reflectivity,
                                                 ior, albedo3)
            if ti.static(glossy == 0):
                # No lobe to spread the continuations over, so the mirror ray
                # keeps only the share of the lobe it can honestly stand for
                # and the remainder falls back to local shading (which the
                # ``share`` term below picks up). See ``_mirror_share``.
                R *= _mirror_share(rough)
            if bounces_left <= 0:
                R = ti.math.vec3(0.0, 0.0, 0.0)

            is_glass = False
            is_pane = False
            if ti.static(refraction != 0):
                if (T > 1e-4) and (bounces_left > 0) and (ior > 1.0 + 1e-4):
                    if fetched_bez:
                        is_pane = True
                    else:
                        is_glass = True

            one3 = ti.math.vec3(1.0, 1.0, 1.0)
            tint = ti.math.clamp(albedo3, 0.0, 1.0)
            trans_share = diel_pass * T
            r_glow = ti.max(R[0], ti.max(R[1], R[2]))
            w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
            share = (weight * alpha) * (one3 - R - trans_share)
            acc += ti.math.vec4(
                share[0], share[1], share[2],
                w_glow * alpha * (1.0 - r_glow - trans_share)) * color
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha

            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                        and (bounces_left > 0):
                    split_refl = True

            sec_pm = 0
            sec_n = 0
            if ti.static(sec_aa > 1):
                sec_pm, sec_n = _sec_positions(msk_low, sec_aa)

            # The redistribution target: the samples this sheet does NOT own.
            own_msk = msk_low
            if areal:
                own_msk = _AA_MASK_ALL

            if is_glass:
                wt = weight * trans_energy * tint
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    hp = surf_pos
                    tp = f % tri_pos.shape[0]
                    v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                      tri_pos[tp, prim, 1],
                                      tri_pos[tp, prim, 2])
                    v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                      tri_pos[tp, prim, 4],
                                      tri_pos[tp, prim, 5])
                    v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                      tri_pos[tp, prim, 7],
                                      tri_pos[tp, prim, 8])
                    face_normal = (v1 - v0).cross(v2 - v0)
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        wsub = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hp, normal,
                                        tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdt = _refract_ray(rdj, nj, ior)
                                if ti.static(mode != 1):
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc,
                                        _offset_transmitted_origin(
                                            hpj, rdt, face_normal, nj),
                                        rdt, wsub, base_dist + t_hit,
                                        bounces_left - 1, processed, pixel, r, 1)
                    else:
                        rdt = _refract_ray(surf_rd, normal, ior)
                        if ti.static(mode != 1):
                            _spawn_pool_ray(
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                rs_alloc,
                                _offset_transmitted_origin(
                                    hp, rdt, face_normal, normal),
                                rdt, wt, base_dist + t_hit,
                                bounces_left - 1, processed, pixel, r, 1)
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hit_point = surf_pos
                    if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                            and (sec_n > 1):
                        weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                        placed = False
                        jtap = 0
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hit_point,
                                        nref, tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                                if ti.static(glossy != 0):
                                    if rough > _GLOSSY_MIN_ROUGHNESS:
                                        rdr = _glossy_reflect(
                                            rdj, nj, rough, jtap, sec_n,
                                            g_roff, g_aoff)
                                jtap += 1
                                org = hpj + nj * (10.0 * MIN_HIT_DISTANCE)
                                if placed:
                                    if ti.static(mode != 1):
                                        _spawn_pool_ray(
                                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                            rs_pix, rs_alloc, org, rdr, weight,
                                            base_dist + t_hit, bounces_left - 1,
                                            processed, pixel, r, 1)
                                else:
                                    rd = rdr
                                    ro = org
                                    placed = True
                    else:
                        rd = refl_rd
                        ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                        weight *= refl_energy
                    base_dist += t_hit
                    bounces_left -= 1
                    bounced = True
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1,
                                          d_kind, 2,
                                          prim_raw, 0, 0, msk, cov,
                                          _popcount_samples(msk), cfac, eff,
                                          mat_alpha, alpha, trans_share,
                                          refl_max, t_hit, svis)
                    break
                else:
                    rwt = weight * refl_energy
                    rwt_max = ti.max(rwt[0], ti.max(rwt[1], rwt[2]))
                    if rwt_max > MIN_WEIGHT:
                        refl_rd, nref = _reflect_frame(surf_rd, normal,
                                                       geo_normal)
                        rhp = surf_pos
                        if ti.static(sec_aa > 1) \
                                and (rwt_max > sec_min_energy) and (sec_n > 1):
                            rwsub = rwt * (1.0 / ti.cast(sec_n, ti.f32))
                            jtap = 0
                            for s in ti.static(range(sec_aa)):
                                if (sec_pm >> s) & 1:
                                    rdj, hpj, nj, _b1, _b2 = \
                                        _jittered_surface_sample(
                                            f, px, py,
                                            ti.static(
                                                _AA_SEC_JITTER[sec_aa][s][0]),
                                            ti.static(
                                                _AA_SEC_JITTER[sec_aa][s][1]),
                                            gen_meta, fetched_bez, prim, rhp,
                                            nref, tri_pos, tri_norm, tri_uvs,
                                            tri_tex_meta, textures,
                                            num_colored_triangles,
                                            cam_origin, screen_point,
                                            pixel_basis_x, pixel_basis_y)
                                    rdr, nj = _reflect_frame(rdj, nj,
                                                             geo_normal)
                                    if ti.static(glossy != 0):
                                        if rough > _GLOSSY_MIN_ROUGHNESS:
                                            rdr = _glossy_reflect(
                                                rdj, nj, rough, jtap, sec_n,
                                                g_roff, g_aoff)
                                    jtap += 1
                                    if ti.static(mode != 1):
                                        _spawn_pool_ray(
                                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                            rs_pix, rs_alloc,
                                            hpj + nj * (10.0 * MIN_HIT_DISTANCE),
                                            rdr, rwsub, base_dist + t_hit,
                                            bounces_left - 1, processed, pixel, r,
                                            1)
                        else:
                            if ti.static(mode != 1):
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                    rs_alloc,
                                    rhp + nref * (10.0 * MIN_HIT_DISTANCE),
                                    refl_rd, rwt, base_dist + t_hit,
                                    bounces_left - 1, processed, pixel, r, 1)
                    if not defer:
                        rr = _run_svis_write(svis, slots, w_a_s, 0.0, w_cfac, 1)
                        _run_redistribute(svis, own_msk, rr)
                        band_p = 0.0
            elif is_pane or split_refl:
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hp = surf_pos
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        wsub = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        jtap = 0
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hp, nref,
                                        tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                                if ti.static(glossy != 0):
                                    if rough > _GLOSSY_MIN_ROUGHNESS:
                                        rdr = _glossy_reflect(
                                            rdj, nj, rough, jtap, sec_n,
                                            g_roff, g_aoff)
                                jtap += 1
                                if ti.static(mode != 1):
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc,
                                        hpj + nj * (10.0 * MIN_HIT_DISTANCE),
                                        rdr,
                                        wsub, base_dist + t_hit, bounces_left - 1,
                                        processed, pixel, r, 1)
                    else:
                        if ti.static(mode != 1):
                            _spawn_pool_ray(
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                rs_alloc,
                                hp + nref * (10.0 * MIN_HIT_DISTANCE),
                                refl_rd,
                                wt, base_dist + t_hit, bounces_left - 1,
                                processed, pixel, r, 1)
                ts_s = w_a_s * trans_share
                pm = (1.0 - w_a_s) + ts_s
                if not defer:
                    rr = _run_svis_write(svis, slots, w_a_s, trans_share,
                                         w_cfac, 1)
                    _run_redistribute(svis, own_msk, rr)
                    band_p = 0.0
                    if ts_s > 1e-6:
                        frac = w_cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - w_a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                hit_point = surf_pos
                if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                        and (sec_n > 1):
                    weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                    placed = False
                    jtap = 0
                    for s in ti.static(range(sec_aa)):
                        if (sec_pm >> s) & 1:
                            rdj, hpj, nj, _b1, _b2 = _jittered_surface_sample(
                                f, px, py,
                                ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                gen_meta, fetched_bez, prim, hit_point, nref,
                                tri_pos, tri_norm, tri_uvs, tri_tex_meta,
                                textures, num_colored_triangles,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y)
                            rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                            if ti.static(glossy != 0):
                                if rough > _GLOSSY_MIN_ROUGHNESS:
                                    rdr = _glossy_reflect(
                                        rdj, nj, rough, jtap, sec_n,
                                        g_roff, g_aoff)
                            jtap += 1
                            org = hpj + nj * (10.0 * MIN_HIT_DISTANCE)
                            if placed:
                                if ti.static(mode != 1):
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc, org, rdr, weight,
                                        base_dist + t_hit, bounces_left - 1,
                                        processed, pixel, r, 1)
                            else:
                                rd = rdr
                                ro = org
                                placed = True
                else:
                    rd = refl_rd
                    ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                    weight *= refl_energy
                base_dist += t_hit
                bounces_left -= 1
                bounced = True
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, d_kind, 2,
                                      prim_raw, 0, 0, msk, cov,
                                      _popcount_samples(msk), cfac, eff,
                                      mat_alpha, alpha, trans_share, refl_max,
                                      t_hit, svis)
                break
            else:
                ts_s = w_a_s * trans_share
                pm = (1.0 - w_a_s) + ts_s
                if not defer:
                    rr = _run_svis_write(svis, slots, w_a_s, trans_share,
                                         w_cfac, 1)
                    _run_redistribute(svis, own_msk, rr)
                    band_p = 0.0
                    if ts_s > 1e-6:
                        frac = w_cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - w_a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac

            if ti.static(dump):
                if dmatch:
                    _aa_dump_frag(dump_out, q - 1, d_kind, 0,
                                  prim_raw, 0, 0, msk, cov,
                                  _popcount_samples(msk), cfac, eff,
                                  mat_alpha, alpha, trans_share, refl_max,
                                  t_hit, svis)
            vis_all = 0.0
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                vis_all += svis[s]
            cur_w = weight * (vis_all * _AA_SAMPLE_WEIGHT)
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < MIN_WEIGHT:
                done = True
                break

        if not bounced:
            vis_all = 0.0
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                vis_all += svis[s]
            weight *= vis_all * _AA_SAMPLE_WEIGHT

        if processed >= MAX_SURFACES_PER_RAY:
            done = True

        if ti.static(dump):
            if dmatch:
                d_vis = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    d_vis += svis[s]
                _aa_dump_terminal(dump_out, bounced, done, processed,
                                  d_vis * _AA_SAMPLE_WEIGHT, acc, weight,
                                  svis)

        if ti.static(mode == 1):
            # The event pass owns no ray state and commits no pixels; its
            # writes were the per-sheet event tables above.
            continue
        if bounced and not done:
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            rs_sca[r, 0] = weight[0]
            rs_sca[r, 1] = 0.0
            rs_sca[r, 2] = 1e30
            rs_sca[r, 3] = -1e30
            rs_sca[r, 4] = base_dist
            rs_sca[r, 5] = weight[1]
            rs_sca[r, 6] = weight[2]
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _ACTIVE
            rs_int[r, 3] = 0
            rs_pix[r] = pixel
            rs_int[r, 4] = r
        else:
            # Background-as-final-sheet (§4.5): on the sparse route the frame
            # buffer is prefilled with the background — env map included, via
            # env_background_prefill — and the composite multiplies the
            # leftover weight by it. The primary retire direction IS the
            # prefill direction, so folding the env here as well would count
            # it twice; env_in_composite skips the fold and hands the weight
            # through. Bounced rays retire in wavefront_shade, which still
            # samples the env with THEIR direction (the pixel's background
            # would be the wrong ray).
            if ti.static(not env_in_composite):
                if (env_w > 0) and (ti.max(weight[0], ti.max(
                        weight[1], weight[2])) > 0.0):
                    ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                         env_intensity, textures)
                    for k in ti.static(range(3)):
                        acc[k] += weight[k] * ec[k]
                    weight = ti.math.vec3(0.0, 0.0, 0.0)
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[r, k], acc[k])
            for k in ti.static(range(3)):
                ti.atomic_add(pix_accum[r, 4 + k], weight[k])


@ti.kernel
def env_background_prefill(
        num_frames: int, width: int, height: int, start_frame: int,
        jx: ti.f32, jy: ti.f32, half_w: ti.f32, half_h: ti.f32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        env_off: int, env_w: int, env_h: int, env_intensity: ti.f32,
        textures: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """Prefill the frame buffer with the environment map, per (frame, pixel).

    Background-as-final-sheet (DESIGN_sheet_resolve.md §4.5): the resolve
    emits a leftover weight per covered pixel and the composite multiplies it
    by whatever the buffer holds — a flat color, an image plate, or this. An
    empty pixel is then already final with no resolve launch at all, which is
    what lets an env-mapped scene take the sparse covered-pixel route instead
    of forcing the dense one. Byte-scale to match ``_prefill_background``;
    the alpha channel is opaque (the sheet route excludes transparent
    backgrounds).
    """
    ppf = width * height
    for i in range(num_frames * ppf):
        f_rel = i // ppf
        p = i - f_rel * ppf
        f = start_frame + f_rel
        py = p // width
        px = p - py * width
        _ro, rd = _generate_ray(f, px, py, jx, jy, half_w, half_h,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y)
        ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                             env_intensity, textures)
        for k in ti.static(range(3)):
            out[f_rel, p, k] = ec[k] * 255.0
        if out.shape[2] > 3:
            # Column 3 is the GLOW lane, not alpha: the sky emits none, which
            # is also what the dense path's env retire deposits (acc[3] == 0,
            # weight zeroed). Writing 255 here bloomed every pixel white.
            out[f_rel, p, 3] = 0.0
