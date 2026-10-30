"""The sheet resolve kernel (DESIGN_sheet_resolve.md P4-P6, Phase 2).

One thread per covered pixel composites its few depth-sorted SHEETS front to
back -- per-sample transmittance as a prefix product, per-sheet magnitude from
the sheet's own exact area -- and shades ONCE per sheet at its dominant
fragment. The aggregation already happened on the host (``sheets.py``), so
the bounded fragment run scan and its lookahead budget are unnecessary here.
Compaction supplies the coverage, identity and closed-shell information; this
kernel applies per-sheet claims, occlusion and the material
four-way split (shade / reflect / transmit / miss) with its continuations,
inherited from the fragment walk's ``raster_first_shade`` -- deleted in
Phase 4, so this kernel is the only copy.

Per-sheet semantics (in lockstep with ``sheets.resolve_pixel_reference``, the
sequential oracle; tests/unit_tests/test_sheet_resolve_parity.py pins them):

* AREAL sheets -- circuits, and donor-only sheets (empty sample union) --
  claim ``alpha * min(area, 1)`` uniformly over every sample.
* A FULL-union sheet takes ``corr = 1`` inside the ``_AA_FULL_DUST`` band
  (interior tilings stay bit-clean) and ``min(area, 1)`` outside it.
* A PARTIAL-union sheet takes ``corr = min(area, 1) / Q``; ``corr > 1``
  keeps the claim exact, up to what the whole pixel still shows, and
  redistributes the clamped occlusion residue onto the sheet's unowned samples
  immediately (the old rule B collapsed from walk state to per-record
  arithmetic -- ``_occlusion_write`` -- with no pending flag and no
  cross-record feedback).
* The siblings of a §4.4 band claim against the visibility the band met and
  occlude once, where the band closes, by the sum of their coverage factors
  times their own alphas (``_band_write``). A band the walk leaves early
  still writes what its siblings covered.

Launched only by the sparse covered-pixel path, with the same ray-state
contract as the walk -- retired pixels accumulate into ``pix_accum``, bounces
keep their slot, splits go to the shared pool through the same atomic
allocator (deterministic slot ids are P7). Shadows are two launches of this
one body (``mode`` 1 then 2, see the parameter comment): the event pass and
the shading pass share every line of transport, which is what retires the
old design's hand-maintained walk/shadow-walk lockstep.
"""

from algan.rendering.raytracing.arena_args_taichi import (
    ArenaView,
    arena_packed,
)
from algan.rendering.raytracing.glossy_prefilter_taichi import (
    ACC_GEO,
    GL_ROW_DP,
    GL_ROW_MAG,
    GL_ROW_SIGMA_SCALE,
    GL_ROW_W,
    _reflector_magnification,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_BAND_CONT_BIT,
    _AA_FULL_DUST,
    _AA_LOSE_SHIFT,
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_ONE_MESH_BIT,
    _AA_SAMPLE_WEIGHT,
    _AA_SEC_JITTER,
    _AA_SLIVER_BIT,
    _aa_dump_frag,
    _aa_dump_match,
    _aa_dump_terminal,
    _decode_bez_ref,
    _footprint_on_triangle,
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
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    CameraProjection,
    _bezier_normal,
    _bezier_plane_point,
    _camera_point,
    _generate_ray,
    _projected,
    _sample_circuit_color_blend,
    _shade_tri_hit,
    _tri_surface_point,
    max_surfaces_per_ray,
    megakernel_register_cap,
    min_alpha,
    min_weight,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_NO_SHADOW_RECEIVE,
    _MAT_ONE_SIDED,
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
    SHADOW_VIS_CHANNELS,
    _reflect_frame,
    _shadow_terminator_delta,
    direct_specular_lobe,
    light_vis_index,
)
from algan.rendering.raytracing.texture_mips_taichi import _triangle_uv_footprint
from algan.rendering.raytracing.transport_taichi import (
    _offset_ray_origin,
    _transmission_normal,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _GLOSSY_MIN_ROUGHNESS,
    ALLOC_TRUNC_SURFACES,
    _det_deposit,
    _material_env_brdf,
    _material_reflectance,
    _mirror_share,
    _offset_transmitted_origin,
    _refract_ray,
    _relative_ior,
    _reset_shell_segment,
    _sample_env_map,
    _tri_color_g,
    _tri_material_g,
    _tri_normal_g,
)
from algan.taichi_compat import ti


@ti.func
def _sheet_slots(msk, areal):
    """The samples a sheet claims and occludes: 1.0 on each, 0.0 elsewhere.

    Every sample for an areal sheet; otherwise its mask's, less the ones
    sheet_sample_depth ceded to a strictly nearer surface
    (sheets.compact_sheets). Claiming a ceded sample painted an
    interpenetrating surface whole-pixel wrong; zeroed, it drops both the claim
    and this sheet's share of the occlusion there, and the winner claims that
    ink. Off, the host sets no lose bits and every owned sample stays.
    """
    slots = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
    if not areal:
        low = msk & _AA_MASK_ALL
        lose = (msk >> _AA_LOSE_SHIFT) & _AA_MASK_ALL
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            if (((low >> s) & 1) == 0) or (((lose >> s) & 1) != 0):
                slots[s] = 0.0
    return slots


@ti.func
def _occlusion_write(svis: ti.template(), slots, own_msk, a_s, trans_share,
                     cfac, tint, tinted):
    """A sheet's occlusion of what lies behind it, and its tint.

    Each claimed sample keeps ``(1 - ak) + ak * trans_share``, ``ak = cfac *
    a_s``, and rule B pushes what a factor clamped at zero would have taken
    onto the samples outside ``own_msk`` (``_run_svis_write``,
    ``_run_redistribute``). Returns what the throughput takes per channel:
    ``svis`` is one number per sample, so a coloured transmitter's tint rides
    in the throughput instead, as the ratio of the light the pixel keeps per
    channel (``trans_share`` through ``tint``) to what ``svis`` keeps after
    the write. Each channel's sum is left unclamped per sample, which is rule
    B for that channel: where ``corr > 1`` covers more than the sheet's own
    samples, what a channel's factor takes below zero comes off the rest of
    the pixel. That is exact for the pixel at any coverage. It used to blend
    toward the full-coverage tint linearly in the sheet's area: a green sheet
    passing half its light, over half a red pixel, let 0.375 of the red
    through, against the 0.5 its uncovered half alone shows. One where
    ``tinted`` is off.
    """
    kept_c = ti.math.vec3(0.0, 0.0, 0.0)
    if tinted:
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            ak = cfac * a_s * slots[s]
            kept_c += svis[s] * ((1.0 - ak) + (ak * trans_share) * tint)
    rr = _run_svis_write(svis, slots, a_s, trans_share, cfac, 1)
    _run_redistribute(svis, own_msk, rr)
    ratio = ti.math.vec3(1.0, 1.0, 1.0)
    if tinted:
        kept = 0.0
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            kept += svis[s]
        if kept > 1e-12:
            ratio = ti.max(kept_c, 0.0) / kept
    return ratio


@ti.func
def _band_write(svis: ti.template(), slots, own_msk, band_pa, band_pat,
                band_patc):
    """A §4.4 band's one occlusion write, from its siblings' sums.

    ``band_pa`` is the sum over the siblings of coverage factor times alpha,
    ``band_pat`` the part of it transmitted and ``band_patc`` that part per
    channel, so the band occludes as its siblings' area-weighted mix:
    ``_occlusion_write`` at a factor of 1, with the mix's transmitted share
    and tint.
    """
    ts = 0.0
    if band_pa > 0.0:
        ts = band_pat / band_pa
    tint = ti.math.vec3(1.0, 1.0, 1.0)
    if band_pat > 0.0:
        tint = band_patc / band_pat
    return _occlusion_write(svis, slots, own_msk, band_pa, ts, 1.0, tint,
                            band_pat > 1e-6)


@ti.kernel
def sheet_resolve_shade_arena(
        num_covered: int,
        num_colored_triangles: ti.i32,
        num_lights: int,
        # Light slots this variant's ``lvis`` payload carries -- what the batch
        # needs, not the ``max_shadow_lights`` cap (shading_taichi.
        # shadow_vis_slots).
        vis_lights: ti.template(),
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        tri_pids: ti.template(),
        refraction: ti.template(),
        # Nested-IOR media stack gate (wavefront_kernels_taichi module head).
        # This resolve handles PRIMARY rays only -- they regenerate in-kernel
        # from the camera and start in air -- so its whole stack duty is the
        # child stacks handed to _spawn_pool_ray below; every continuation is
        # drained (and nested) by wavefront_shade. In shadow mode 1 every
        # spawn compiles out, so the event pass needs no stack logic.
        ior_stack: ti.template(),
        skip_unlit_normal: ti.template(),
        has_bez: ti.template(),
        sec_aa: ti.template(), sec_min_energy: ti.f32,
        glossy: ti.template(),
        # Deliver the direct lights' share of the reflected specular lobe,
        # which the continuation this hit spawns cannot (rt_settings.
        # direct_specular_lobe). 0 restores the previous weighting exactly.
        direct_spec: ti.template(),
        # Shadow support (DESIGN_sheet_resolve.md §4.9). ``mode`` 0 is the
        # shadow-free resolve; 1 walks the IDENTICAL transport and writes one
        # candidate shadow event per accepted lit triangle sheet (no shading,
        # no spawns) into dense per-sheet tables — event identity is the
        # sheet index, so no counter and no atomics exist; 2 is the shading
        # resolve reading the traced per-event visibility. One kernel body
        # for all three means a resolve/shadow desync can no longer be
        # written into the source -- the two fragment walks this replaces had
        # to be kept in lockstep by hand and grew a harness just to check it.
        # It is not bit-exact, though: modes 1 and 2 are separate fast-math
        # compiles, so a value within rounding of a threshold (the
        # ``eff <= min_alpha`` skip, the far clip, the ``min_weight`` cut) can
        # land on different sides of it in the two walks. Only a sheet AT
        # such a threshold can flip -- one claiming about ``min_alpha``, one
        # reached with under ``min_weight`` left, or one at the far clip to
        # within rounding -- and a sheet mode 2 shades without an event is
        # lit unshadowed (``lvis`` keeps its 1.0).
        mode: ti.template(),
        # Shadow-terminator gate (rt_settings.shadow_terminator_mode()):
        # == 1 makes THIS pass compute the Hanika offset per accepted event
        # and store it in ``event_toff``; raster_shadow_trace relaxes its
        # face-normal horizon cull whenever the gate is != 0 but applies the
        # stored offset only at == 1 (0 keeps today's origin and today's
        # guard; 2 is the diagnostic relax-only arm and never reads
        # ``event_toff``).
        shadow_term: ti.template(),
        # One-sided back-face shadow cull (rt_settings.shadow_sided_cull):
        # the mode-1 event build keeps a one-sided surface's own orientation
        # for the shadow normals it stores, so the trace's light-facing cull
        # agrees with the side the shading lights.
        sided_cull: ti.template(),
        # Cross-pass material memoization (rt_settings.sheet_resolve_memo,
        # RENDERER_WORK_QUEUE.md item 9). != 0 makes the mode-1 event walk
        # STORE each processed triangle sheet's fetched material into
        # ``sheet_memo`` and the mode-2 shading walk READ it back instead of
        # re-fetching -- twelve floats against three barycentric-interpolated
        # (and possibly texture-sampled) fetches per sheet.
        #
        # Sound because the two walks process the same sheets: every
        # ``mode != 1`` gate in this body wraps a spawn, a shade, the
        # truncation counter or a pixel commit, and none of them touches the
        # loop-carried transport state (``weight``, ``svis``, the band sums,
        # ``bounces_left``, ``processed``, ``done``) or any break/continue
        # condition. The two skips that precede the fetch -- the far-clip
        # break and the ``eff <= min_alpha`` continue -- are transport-only
        # for the same reason. So a row read in mode 2 was written in mode 1,
        # except at the threshold flips the ``mode`` comment above describes:
        # there mode 2 reads a row mode 1 never wrote, whatever the arena
        # held. Values are copied verbatim through f32, hence byte-identical.
        memo: ti.template(),
        # 1 on a splitting batch: the pixel's first ``pix_accum`` deposits go
        # on the grid its pool rays add to later
        # (``wavefront_kernels_taichi._det_deposit``). Mode 1 commits nothing.
        det_sum: ti.template(),
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(), dump: ti.template(), dump_out: ti.types.ndarray(),
        projected: ti.template(),
        # The display-referred coverage resolve's per-sheet ceiling
        # (rt_settings.aa_display_resolve, DESIGN_sheet_resolve.md §4.8): the
        # largest linear value the display still tells apart, ``1 /
        # exposure``, or 0 for none. A partially covering sheet's local
        # radiance is clamped to it over the area the sheet claims. Runtime,
        # so the switch and the exposure compile nothing.
        display_cap: ti.f32,
        arena_f32: ti.types.ndarray(),
        arena_i32: ti.types.ndarray(),
        arena_i64: ti.types.ndarray(),
        aoff: ti.types.ndarray(),
        ashp: ti.types.ndarray()):
    """Composite + shade each covered pixel's depth-sorted sheet list.

    Everything is indexed by covered ordinal ``t`` (the walk's ``compact``
    contract): the sheet CSR, ray state and accumulator rows are ``t``-local
    and ``covered_idx[t]`` is used only for the frame/pixel coordinate. The
    host zeroes ``pix_accum`` and pre-marks pool slots DONE, so retirement
    accumulates and only bounced pixels write ray state.
    """
    # Arena-bound parameters (arena_args_taichi): each name is
    # rebound to a window into its dtype's buffer, at the offset
    # the host wrote into aoff. Order is _SHEET_RESOLVE_SHADE_ARENA's.
    sheet_offsets = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    sheet_key = ti.static(ArenaView(arena_i64, aoff[1], (ashp[1],)))
    sheet_ref = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    sheet_ab = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4])))
    sheet_cov = ti.static(ArenaView(arena_f32, aoff[4], (ashp[5],)))
    sheet_msk = ti.static(ArenaView(arena_i32, aoff[5], (ashp[6],)))
    sheet_cap = ti.static(ArenaView(arena_f32, aoff[6], (ashp[7],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[7], (ashp[8], ashp[9], ashp[10])))
    tri_norm = ti.static(ArenaView(arena_f32, aoff[8], (ashp[11], ashp[12], ashp[13])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[9], (ashp[14], ashp[15], ashp[16])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[10], (ashp[17], ashp[18], ashp[19], ashp[20])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[11], (ashp[21], ashp[22], ashp[23])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[12], (ashp[24], ashp[25])))
    textures = ti.static(ArenaView(arena_f32, aoff[13], (ashp[26], ashp[27], ashp[28])))
    col_row = ti.static(ArenaView(arena_i32, aoff[14], (ashp[29],)))
    tri_mat_id = ti.static(ArenaView(arena_i32, aoff[15], (ashp[30], ashp[31])))
    tri_mat = ti.static(ArenaView(arena_f32, aoff[16], (ashp[32], ashp[33], ashp[34])))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[17], (ashp[35], ashp[36], ashp[37])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[18], (ashp[38], ashp[39], ashp[40], ashp[41])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[19], (ashp[42], ashp[43], ashp[44], ashp[45])))
    light_pos = ti.static(ArenaView(
        arena_f32, aoff[20], (ashp[46], ashp[47], ashp[48])))
    light_col = ti.static(ArenaView(
        arena_f32, aoff[21], (ashp[49], ashp[50], ashp[51])))
    layer_offsets = ti.static(ArenaView(arena_f32, aoff[22], (ashp[52],)))
    sheet_memo = ti.static(ArenaView(arena_f32, aoff[23], (ashp[53], ashp[54])))
    sheet_accept = ti.static(ArenaView(arena_i32, aoff[24], (ashp[55],)))
    event_pos = ti.static(ArenaView(arena_f32, aoff[25], (ashp[56], ashp[57])))
    event_snrm = ti.static(ArenaView(arena_f32, aoff[26], (ashp[58], ashp[59])))
    event_fnrm = ti.static(ArenaView(arena_f32, aoff[27], (ashp[60], ashp[61])))
    event_frame = ti.static(ArenaView(arena_i32, aoff[28], (ashp[62],)))
    event_msk = ti.static(ArenaView(arena_i32, aoff[29], (ashp[63],)))
    event_dp = ti.static(ArenaView(arena_f32, aoff[30], (ashp[64], ashp[65])))
    # Per-event shadow-terminator displacement (vec3), written by the mode-1
    # build exactly when ``sheet_accept`` is set; uninitialised arena memory
    # otherwise, so nothing may read a row that was not written this frame.
    event_toff = ti.static(ArenaView(arena_f32, aoff[31], (ashp[66], ashp[67])))
    sheet_event_id = ti.static(ArenaView(arena_i32, aoff[32], (ashp[68],)))
    shadow_vis = ti.static(ArenaView(
        arena_f32, aoff[33], (ashp[69], ashp[70], ashp[71])))
    covered_idx = ti.static(ArenaView(arena_i32, aoff[34], (ashp[72],)))
    cam_origin = ti.static(ArenaView(arena_f32, aoff[35], (ashp[73], ashp[74])))
    screen_point = ti.static(ArenaView(arena_f32, aoff[36], (ashp[75], ashp[76])))
    pixel_basis_x = ti.static(ArenaView(arena_f32, aoff[37], (ashp[77], ashp[78])))
    pixel_basis_y = ti.static(ArenaView(arena_f32, aoff[38], (ashp[79], ashp[80])))
    gen_meta = ti.static(ArenaView(arena_f32, aoff[39], (ashp[81],)))
    rs_alloc = ti.static(ArenaView(arena_i32, aoff[40], (ashp[82],)))
    # Per-frame projection rows (algan.rendering.projection).
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[41], (ashp[83], ashp[84])))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    pixels_per_frame = width * height
    far_clip = layer_offsets[5]
    max_bounces = ti.cast(layer_offsets[6] + 0.5, ti.i32)
    # Taps the continuation fan site below unrolls (``for s in
    # ti.static(range(fan_taps))``). Each tap inlines a jittered surface
    # sample, a normal fetch and a pool spawn, so at ``sec_aa`` 4 the fans are
    # the largest part of this kernel's compile time. The event pass (mode 1)
    # compiles the site out: all it produces is the pool spawns, which mode 1
    # never makes, and the re-aimed ``ro``/``rd``, which nothing reads after
    # the ``break`` that follows. What the branches do to ask for a fan -- the
    # fan conditions, the ``weight`` scale by ``1 / sec_n``, ``bounces_left``,
    # ``bounced`` -- stays, so mode 1 walks the same transport as mode 2.
    # Past that, mode 1's output depends on ``sec_aa`` only through
    # ``sec_aa > 1`` (the ``event_dp`` footprint below), so the host passes it
    # 2 for any count above 1, and passes ``sec_aa`` 1 to the shading launches
    # of a batch with nothing a fan could reflect or refract through
    # (raster_pipeline's ``event_args`` and ``shade_taps``).
    fan_taps = ti.static(sec_aa if mode != 1 else 0)
    # Quadrants patch 0005's per-kernel register cap, off unless
    # rt_settings.megakernel_max_reg asks for one (see megakernel_register_cap).
    if ti.static(megakernel_register_cap()):
        ti.loop_config(max_reg=ti.static(megakernel_register_cap()))
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
                               pixel_basis_x, pixel_basis_y, cam_proj)
        dmatch = False
        if ti.static(dump):
            dmatch = _aa_dump_match(dump_out, px, py, f)
        # Only the tap-fan modes (1, 2) spread a ``sec_aa`` fan over the GGX
        # lobe: the rotation here and the four ``_glossy_reflect`` sites
        # below. Under the prefilter (3) the claiming sheet's one ray stands
        # for its lobe, and every other reflective sheet keeps the plain
        # throttled mirror fan its design promises (DESIGN_glossy_prefilter.md
        # §5). Spread as well, that fan drew stair-edged ghost copies of what
        # the sheet reflected, on top of the ``_mirror_share`` throttle that
        # already stands in for the lobe.
        g_roff = 0.5
        g_aoff = 0.0
        if ti.static(glossy == 1 or glossy == 2):
            g_roff, g_aoff = _glossy_rotation(px, py, ti.static(glossy == 2))

        # SPLIT-SUM PREFILTER (DESIGN_glossy_prefilter.md): pixels-per-radian
        # for this pixel's ray, which is what turns the lobe's angular width
        # into the blur radius its reflection buffer will be prefiltered at.
        #
        # ``pixel_basis_x`` is world-per-SCREEN-UNIT and a pixel is
        # ``1 / half_screen_h`` of one, so the world size of a pixel at unit
        # perpendicular depth is ``|pbx| / (half_h * focal)``. A ray parameter
        # is a SLANT range, so the conversion also carries the optical-axis
        # cosine, exactly as ``_axis_cos`` documents for every other width
        # derived from a pixel size. Computed here rather than taken as an
        # argument: this kernel is at 72 parameters and Taichi's ceiling is 64
        # runtime ones, which is why the env map's placement already rides
        # inside ``layer_offsets``.
        gl_px_per_rad = 0.0
        gl_taken = False
        # The prefiltered glossy event is a per-PIXEL resource -- one ``W``,
        # one radius, one reflection row (DESIGN_glossy_prefilter.md §2.2) --
        # so it goes to the sheet that reflects the most light, and every
        # other reflective sheet of the pixel falls back to the
        # ``_mirror_share`` throttle. The walk is front to back and cannot
        # know that in advance, so the first qualifying sheet claims it and a
        # later one that reflects more TAKES it: the claim's ray is not
        # spawned until the walk ends, and the sheet that loses it is
        # throttled after the fact (see ``prefilter_take`` below).
        #
        # First-come was the earlier rule, and it made the claim snap. A faint
        # sheet in front of a rough mirror -- a fading title layer, a
        # translucent label -- held the blur while the mirror behind it
        # reflected sharp, and the frame its weight dropped under
        # ``min_weight`` the mirror's reflection blurred all at once. Now the
        # handover happens where the two reflect equally, so what changes
        # hands is the weaker of two equal shares.
        #
        # A §4.4 BAND's siblings are not "later sheets". They are ONE surface's
        # coverage of the pixel, subdivided so each shades with its own normal,
        # and §4.4's contract is that the subdivision changes nothing the band
        # commits. It changed this: the throttle at roughness 0.35 is ~3% of
        # Schlick where the split-sum ``E`` is the lobe's whole directional
        # albedo, and the local term is ``alpha * (1 - R)`` -- so a crease
        # pixel's far sibling kept energy the interior of the same facet
        # spends on its reflection. ``E`` is per-channel and a metal's F0 is its
        # albedo, so on the fast suite's red Icosahedron that landed as a
        # brighter, REDDER line down every interior edge (+21 sRGB in red
        # against +1 in green, measured at PREVIEW).
        #
        # So the claim is shared by the band that made it: ``gl_band`` says
        # that band is still open, each sibling shades with ``E``, and their
        # energies sum into the one row they share -- one ray, in the claiming
        # sibling's mirror direction, at the band's total weight, which is the
        # weight the unsplit band would have spawned.
        gl_band = False
        gl_wt = ti.math.vec3(0.0, 0.0, 0.0)
        # The claim's ray, held until the walk ends: origin, direction, path
        # length, bounces and surfaces so far, and the claiming sibling's
        # roughness (its radius) and curvature (its magnification).
        gl_org = ti.math.vec3(0.0, 0.0, 0.0)
        gl_dir = ti.math.vec3(0.0, 0.0, 0.0)
        gl_dist = 0.0
        gl_bounces = 0
        gl_proc = 0
        gl_rough = 0.0
        gl_mag = 1.0
        # What the claiming band's local shading gave up for ``E``: the sum of
        # ``weight * alpha * E * color`` over its sheets, and the glow lane's
        # equivalent. A band that loses the claim takes back the part of it
        # the throttle would have left there.
        gl_back = ti.math.vec3(0.0, 0.0, 0.0)
        gl_glow = 0.0
        if ti.static(glossy == 3):
            sp_c = ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                                screen_point[f, 2])
            pbx_c = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                                 pixel_basis_x[f, 2])
            focal = (sp_c - ro).norm()
            axis_cos = rd.dot((sp_c - ro).normalized())
            px_per_depth = pbx_c.norm() / (gen_meta[3] * ti.max(focal, 1e-6))
            gl_px_per_rad = 1.0 / ti.max(px_per_depth * axis_cos, 1e-9)
            if ti.static(_projected(cam_proj)):
                if cam_proj[f, 1] != 0.0:
                    # Off the pinhole a pixel at camera path length s is
                    # ``B + C * s`` world units across (projection.py), and
                    # ``C`` is zero for an orthographic camera, so the scale
                    # stored is per unit of the constant part instead: one
                    # over ``B``, the axis cosine taken from the eye.
                    # gloss_scatter reads it with the frame's projection.
                    eye_c = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                                         cam_origin[f, 2])
                    axis_cos = rd.dot((sp_c - eye_c).normalized())
                    gl_px_per_rad = 1.0 / ti.max(
                        pbx_c.norm() * cam_proj[f, 1] / gen_meta[3] * axis_cos,
                        1e-12)
            if ti.static(mode != 1):
                # Published for the drain, NEGATED so the scatter still reads
                # the pixel as having no glossy branch: when no sheet below
                # claims the pixel's one prefiltered event, a pool ray that
                # meets a rough reflector further down the pixel's transport
                # (seen in a mirror, or through glass) claims it there, and
                # this is the scale its radius needs. A claim below overwrites
                # it. See ``wavefront_shade_arena``'s ``gl_take``.
                pix_accum[r + num_covered, GL_ROW_SIGMA_SCALE] = -gl_px_per_rad

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        # ONE-MESH ceiling bookkeeping: coverage this pixel's single mesh has
        # already committed. Only meaningful where the host flagged the pixel
        # (the _AA_ONE_MESH_BIT rides in every sheet's flags there).
        mesh_ink = 0.0
        # §4.4 band state: sheets of ONE band (the shading-class siblings the
        # compaction subdivides) claim against the same incoming visibility
        # and occlude ONCE, where the band closes. _AA_BAND_CONT_BIT is the
        # host's flag that a band continues at the NEXT sheet of the walk,
        # which is what lets the sums ride in registers (sheets.py,
        # ``_sibling_weights``). Per band: the summed coverage factor and
        # claim (for the over-claim cap), the summed factor times each
        # sibling's alpha, the part of that transmitted, and that part per
        # channel; and the last sibling's mask word, for a band the walk
        # leaves before it closes.
        band_open = False
        band_pc = 0.0
        band_claim = 0.0
        band_pa = 0.0
        band_pat = 0.0
        band_patc = ti.math.vec3(0.0, 0.0, 0.0)
        band_msk = 0
        base_dist = 0.0
        bounces_left = max_bounces
        processed = 0
        bounced = False
        done = False

        q = 0
        while q < total and processed < max_surfaces_per_ray:
            idx = start + q
            t_hit = _frag_t(sheet_key[idx])
            prim_raw = sheet_ref[idx]
            a = sheet_ab[idx, 0]
            b = sheet_ab[idx, 1]
            cov = sheet_cov[idx]
            # A NaN area claims nothing. Left alone it would claim the whole
            # pixel: ``ti.min(cov, 1.0)`` below returns 1.0 for a NaN on the
            # CPU backend, so one bad sheet would paint opaque. The test reads
            # the bits because fast math may fold ``cov != cov`` to false
            # (radix_sort_taichi's canonicalization has the history). The
            # oracle drops NaN areas the same way.
            if (ti.bit_cast(cov, ti.u32) & ti.u32(0x7FFFFFFF)) \
                    > ti.u32(0x7F800000):
                cov = 0.0
            cov = ti.max(cov, 0.0)
            msk = sheet_msk[idx]
            defer = (msk & _AA_BAND_CONT_BIT) != 0
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
            slots = _sheet_slots(msk, areal)
            if areal:
                dens = area
            else:
                pop = _popcount_samples(msk_low)
                if msk_low == _AA_MASK_ALL:
                    if ti.abs(1.0 - cov) > _AA_FULL_DUST:
                        cfac = area
                else:
                    # Normalized to the mask's own popcount, ceded samples
                    # included: their share of the area is the winner's ink.
                    cfac = area * ti.static(float(_AA_NUM_SAMPLES)) \
                        / ti.cast(pop, ti.f32)
            # The samples this sheet's occlusion residue may NOT go to (rule
            # B redistributes onto the rest): its own, ceded ones included.
            own_msk = msk_low
            if areal:
                own_msk = _AA_MASK_ALL
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
                    dens *= room / ti.max(eff, min_alpha)
                    eff = room
            # A coverage factor above 1 -- a sheet covering more of the pixel
            # than its samples' share, or a band whose siblings sum past it --
            # claims more than its own samples show, which is how corr > 1
            # keeps a claim exact. But it may not claim more than the WHOLE
            # pixel still shows: behind a nearer occluder the rest of its area
            # is hidden, not elsewhere, and the write below (rule B) already
            # stops there. The band's claim so far counts against it, as the
            # unsplit band's one claim would. Below 1 this cannot bind.
            band_pc += cfac * dens
            if band_pc > 1.0:
                shown = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    shown += svis[s]
                eff = ti.min(eff, ti.max(
                    shown * _AA_SAMPLE_WEIGHT - band_claim, 0.0))
            if eff <= min_alpha:
                if not defer:
                    if band_open:
                        # The band's last sibling claims nothing, but the
                        # ones before it did, and their occlusion is still
                        # owed: write it here, on this sheet's samples, which
                        # are the band's.
                        weight *= _band_write(svis, slots, own_msk, band_pa,
                                              band_pat, band_patc)
                    band_open = False
                    band_pc = 0.0
                    band_claim = 0.0
                    band_pa = 0.0
                    band_pat = 0.0
                    band_patc = ti.math.vec3(0.0, 0.0, 0.0)
                else:
                    band_open = True
                    band_msk = msk
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, d_kind, 1,
                                      prim_raw, 0, 0, msk, cov,
                                      _popcount_samples(msk), cfac, eff,
                                      0.0, 0.0, 0.0, 0.0, t_hit, svis)
                continue
            if not is_bez:
                mesh_ink += eff
            band_claim += eff

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            rough = 0.0
            ior = 0.0
            T = 0.0
            albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
            # Per-light shadow visibility, RGB and channel-major (see
            # shading_taichi.light_vis_index). Declared at the sheet's own
            # scope rather than inside the shading branch that fills it,
            # because the reflected lobe's direct-light add-back reads it
            # again further down, once the surface normal is known.
            lvis = ti.Vector([1.0] * (SHADOW_VIS_CHANNELS * vis_lights))
            prim = 0
            circuit = 0
            fetched_bez = False
            tex_du, tex_dv = 0.0, 0.0
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
                    # On the circuit's plane: the centre-ray point is
                    # rounded at the eye's magnitude, which a distant
                    # camera puts under the plane (_bezier_plane_point).
                    surf_pos = _bezier_plane_point(f, circuit, surf_pos,
                                                   circuit_meta)
            if not fetched_bez:
                prim = prim_raw
                # MEMO READ (mode 2). Column layout, shared with the mode-1
                # write below: 0-3 color, 4 alpha, 5 reflectivity,
                # 6 roughness, 7 IOR, 8 transmission, 9-11 surface point.
                # Read here, before ``partial``, because ``surf_rd`` is
                # derived from ``surf_pos`` and the shading below needs both.
                if ti.static(memo != 0 and mode == 2):
                    surf_pos = ti.math.vec3(sheet_memo[idx, 9],
                                            sheet_memo[idx, 10],
                                            sheet_memo[idx, 11])
                else:
                    surf_pos = _tri_surface_point(f, prim, w0, a, b, tri_pos)
                partial = msk_low != _AA_MASK_ALL
                if partial:
                    surf_rd = (surf_pos - ro).normalized()
                    if ti.static(_projected(cam_proj)):
                        # From where the camera ray reaching the point
                        # starts: an orthographic ray re-aimed from its own
                        # origin would tilt.
                        surf_rd = (surf_pos - _camera_point(
                            f, surf_pos, cam_origin, cam_proj)).normalized()
                # Avoid camera/footprint work for untextured and promoted
                # constant maps, including the feature-disabled bank.
                tex_idx = prim - num_colored_triangles
                if tri_tex_meta.shape[1] >= 21:
                    if tex_idx >= 0:
                        if (tri_tex_meta[tex_idx, 18] >= 0
                                or tri_tex_meta[tex_idx, 19] >= 0
                                or tri_tex_meta[tex_idx, 20] >= 0):
                            camera_screen = ti.math.vec3(screen_point[f, 0], screen_point[f, 1], screen_point[f, 2])
                            pixel_basis = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1], pixel_basis_y[f, 2])
                            pixel_angle = pixel_basis.norm() / ti.max(gen_meta[3] * (camera_screen - ro).norm(), 1e-12)
                            footprint = (surf_pos - ro).norm() * pixel_angle
                            if ti.static(_projected(cam_proj)):
                                if cam_proj[f, 1] != 0.0:
                                    # spread * D + persp * (ray length) pixel
                                    # widths at unit depth: measured from the
                                    # eye and the camera ray's own origin,
                                    # not from ``ro``, which an orthographic
                                    # ray starts off the axis (projection.py).
                                    eye = ti.math.vec3(cam_origin[f, 0],
                                                       cam_origin[f, 1],
                                                       cam_origin[f, 2])
                                    eye_dist = (camera_screen - eye).norm()
                                    pixel_angle = pixel_basis.norm() / ti.max(
                                        gen_meta[3] * eye_dist, 1e-12)
                                    footprint = pixel_angle * (
                                        cam_proj[f, 1] * eye_dist
                                        + cam_proj[f, 0] * (surf_pos - _camera_point(
                                            f, surf_pos, cam_origin, cam_proj)).norm())
                            tex_du, tex_dv = _triangle_uv_footprint(
                                0, f, prim, surf_rd, footprint,
                                tri_pos, tri_uvs, tri_tex_meta, num_colored_triangles)
                if ti.static(memo != 0 and mode == 2):
                    color = ti.math.vec4(sheet_memo[idx, 0], sheet_memo[idx, 1],
                                         sheet_memo[idx, 2], sheet_memo[idx, 3])
                    alpha = sheet_memo[idx, 4]
                    reflectivity = sheet_memo[idx, 5]
                    rough = sheet_memo[idx, 6]
                    ior = sheet_memo[idx, 7]
                    T = sheet_memo[idx, 8]
                else:
                    color, alpha = _tri_color_g(0, f, prim, w0, a, b, tri_colors,
                                                col_row, tri_uvs, tri_tex_meta,
                                                textures, num_colored_triangles, tex_du, tex_dv)
                    # All four in one call: a promoted constant material
                    # drives every property from its map, and separate
                    # reflectivity and IOR calls sampled the same texel twice.
                    reflectivity, rough, ior, T = _tri_material_g(
                        0, f, prim, w0, a, b, tri_extra, col_row, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
                albedo3 = ti.math.vec3(color[0], color[1], color[2])
                if ti.static(frag_shading != 0 and mode != 1):
                    if ti.static(mode == 2):
                        event_id = sheet_event_id[idx]
                        if event_id >= 0:
                            for li in range(num_lights):
                                if li < vis_lights:
                                    base = light_vis_index(li, 0)
                                    for c in ti.static(range(3)):
                                        lvis[base + c] = \
                                            shadow_vis[event_id, li, c]
                    sn = ti.math.vec3(0.0, 0.0, 0.0)
                    if ti.static(skip_unlit_normal != 0):
                        if tri_mat_id[f % tri_mat_id.shape[0], prim] \
                                != _MID_UNLIT:
                            sn = _tri_normal_g(
                                0, f, prim, w0, a, b, tri_norm, tri_pos,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles, tex_du, tex_dv)
                    else:
                        sn = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
                    color = _shade_tri_hit(frag_pipelines, tri_pids,
                                           f, prim, a, b,
                                           surf_rd, surf_pos,
                                           tri_pos, sn,
                                           tri_mat_id, tri_mat,
                                           light_pos, light_col,
                                           num_lights, color,
                                           ti.static(1 if mode == 2 else 0),
                                           lvis, cam_origin, cam_proj)
                # MEMO WRITE (mode 1). ``color`` is still the FETCHED color
                # here: _shade_tri_hit is inside the ``mode != 1`` branch
                # above, so the event walk never overwrites it. Written for
                # every processed triangle sheet, not only accepted events --
                # an unlit or shadow-opted-out sheet builds no event but mode
                # 2 still shades it.
                if ti.static(memo != 0 and mode == 1):
                    for k in ti.static(range(4)):
                        sheet_memo[idx, k] = color[k]
                    sheet_memo[idx, 4] = alpha
                    sheet_memo[idx, 5] = reflectivity
                    sheet_memo[idx, 6] = rough
                    sheet_memo[idx, 7] = ior
                    sheet_memo[idx, 8] = T
                    for k in ti.static(range(3)):
                        sheet_memo[idx, 9 + k] = surf_pos[k]
                if ti.static(mode == 1):
                    # One candidate shadow event per accepted lit triangle
                    # sheet, at the sheet index — mirroring the fragment
                    # build's acceptance (reached the fetch, material not
                    # unlit) and its payload: position and normals at the
                    # dominant fragment, the 4-bit sub-pixel position mask
                    # with the material pipeline id above it, and the pixel
                    # footprint for soft sub-pixel sampling.
                    pid_e = tri_mat_id[f % tri_mat_id.shape[0], prim]
                    # A mob that declared receives_shadows False is shaded as
                    # though every light reached it, so it builds no event and
                    # traces no shadow ray -- opting out is cheaper than the
                    # default rather than a query whose answer is discarded.
                    #
                    # Asked of BUILT-IN pipelines only, which is the same rule
                    # that keeps _run_frag_pipeline's one_sided read on the
                    # built-in branch, and it is a semantic guard rather than a
                    # bounds one: a custom fragment pipeline numbers its own
                    # slots from 0, so slot 33 of a block >= 34 wide is that
                    # pipeline's REAL parameter, and reading it here would let a
                    # user's unrelated value above 0.5 silently stop the mob
                    # receiving shadows. The width test then covers the narrow
                    # blocks, where the slot does not exist at all. So
                    # receives_shadows is inert for a custom-pipeline mob, the
                    # same way it is inert for 2-D geometry.
                    recv = 1
                    if pid_e < _USER_PIPELINE_BASE:
                        if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                            if tri_mat[f % tri_mat.shape[0], prim,
                                       _MAT_NO_SHADOW_RECEIVE] > 0.5:
                                recv = 0
                    if (pid_e != _MID_UNLIT) and (recv == 1):
                        # A one-sided surface's shadow normals keep its own
                        # orientation (see _orient_hit_normals_sided); the
                        # same built-in-pipeline guard as the receive flag.
                        one_sided_e = 0
                        if ti.static(sided_cull != 0):
                            if pid_e < _USER_PIPELINE_BASE:
                                if tri_mat.shape[2] > _MAT_ONE_SIDED:
                                    if tri_mat[f % tri_mat.shape[0], prim,
                                               _MAT_ONE_SIDED] > 0.5:
                                        one_sided_e = 1
                        snrm, fnrm = _tri_shadow_normals(
                            f, prim, a, b, surf_rd, tri_pos, tri_norm,
                            tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles, one_sided_e)
                        sheet_accept[idx] = 1
                        for k in ti.static(range(3)):
                            event_pos[idx, k] = surf_pos[k]
                            event_snrm[idx, k] = snrm[k]
                            event_fnrm[idx, k] = fnrm[k]
                        # Shadow-terminator displacement of the origin
                        # raster_shadow_trace starts from (RENDERER_WORK_
                        # QUEUE.md item 20). Written on exactly the
                        # sheet_accept condition above -- the arena tensor is
                        # uninitialised, so an accepted event without a write
                        # would trace from garbage. snrm here is the ORIENTED
                        # shading normal the sign rule in
                        # _shadow_terminator_delta asks for. Gate is == 1,
                        # NOT != 0: ``event_toff`` is full-size only in mode
                        # 1 (see raster_pipeline), so the relax-only arm --
                        # which never reads rows back -- must not write it.
                        if ti.static(shadow_term == 1):
                            delta = _shadow_terminator_delta(
                                f, prim, w0, a, b, surf_pos, snrm,
                                tri_pos, tri_norm)
                            for k in ti.static(range(3)):
                                event_toff[idx, k] = delta[k]
                        event_frame[idx] = f
                        shadow_msk = 0xF
                        if (not sliver) and (msk_low != 0):
                            shadow_msk, _sn4 = _sec_positions(msk_low, 4)
                        event_msk[idx] = shadow_msk | (pid_e << 8)
                        if ti.static(sec_aa > 1):
                            dpx, dpy = _pixel_footprint(
                                f, px, py, gen_meta, surf_pos, fnrm,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y, cam_proj)
                            dpx, dpy = _footprint_on_triangle(
                                f, prim, w0, a, b, dpx, dpy, shadow_msk,
                                tri_pos)
                            for k in ti.static(range(3)):
                                event_dp[idx, k] = dpx[k]
                                event_dp[idx, 3 + k] = dpy[k]

            mat_alpha = ti.math.clamp(alpha, 0.0, 1.0)
            alpha = ti.math.clamp(mat_alpha * eff, 0.0, 1.0)
            a_s = mat_alpha * dens
            cont_band = band_open
            band_open = defer
            if defer:
                band_msk = msk
            # This sheet may share the pixel's prefiltered glossy claim only if
            # it continues the band that made it (see ``gl_band`` above); the
            # claim moves to whichever band claims next, and dies with its own.
            gl_share = gl_band and cont_band
            gl_band = gl_share and defer
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
                    , tex_du, tex_dv).normalized()
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

            rel = ior
            interface_n = normal
            if (T > 1e-4) and (not fetched_bez):
                rel = _relative_ior(rs_sca, r, ior,
                                    surf_rd.dot(geo_normal) < 0.0, ior_stack)
                interface_n = _transmission_normal(surf_rd, normal, geo_normal)
            R, diel_pass = _material_reflectance(surf_rd, interface_n,
                                                 reflectivity,
                                                 ior, albedo3, T, rel)
            prefilter_take = False
            if ti.static(glossy == 3):
                # THE SPLIT-SUM SUBSTITUTION. One prefiltered glossy event per
                # pixel (the sheet reflecting the most light, and the rest of
                # ITS §4.4 band with it -- see ``gl_band``); for it, the lobe's
                # exact directional albedo replaces both the Schlick mirror
                # reflectance AND the ``_mirror_share`` throttle that stands in
                # for a sampled lobe. Everything else on the pixel -- a weaker
                # SURFACE's reflective sheet, a mirror below the roughness
                # threshold, glass -- keeps the throttle, unchanged.
                #
                # A sheet outside the claiming band takes the claim when it
                # would reflect more than that whole band has so far, weighed
                # with ``E`` on both sides. Ties stay with the earlier sheet.
                #
                # Made in BOTH shading modes and in the event-build mode: the
                # three walks have to agree about transport or the shadow
                # events would be built for a different image than the one
                # shaded. ``gl_band`` and ``gl_share`` are updated outside the
                # mode gates for the same reason, and the comparison reads
                # transport only (``weight``, ``alpha``, the fetched albedo).
                if (reflectivity >= 0.0) and (T <= 1e-4) \
                        and (rough > _GLOSSY_MIN_ROUGHNESS) \
                        and (bounces_left > 0):
                    e_brdf = _material_env_brdf(surf_rd, normal, reflectivity,
                                                ior, albedo3, rough)
                    if (not gl_taken) or gl_share:
                        prefilter_take = True
                    else:
                        e_wt = weight * (alpha * e_brdf)
                        if ti.max(e_wt[0], ti.max(e_wt[1], e_wt[2])) > ti.max(
                                gl_wt[0], ti.max(gl_wt[1], gl_wt[2])):
                            prefilter_take = True
                    if prefilter_take:
                        R = e_brdf
                    else:
                        R *= _mirror_share(rough)
                else:
                    R *= _mirror_share(rough)
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
                if (T > 1e-4) and (bounces_left > 0) and ((ior > 1.0 + 1e-4) or (ti.abs(rel - 1.0) > 1e-4)):
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
            lit = ti.math.vec4(
                share[0], share[1], share[2],
                w_glow * alpha * (1.0 - r_glow - trans_share)) * color
            # Display-referred resolve (``display_cap``, §4.8). The area this
            # sheet claims shows at most ``display_cap`` per channel, so that
            # is all its local shading may put in the pixel. The tracer sets
            # the cap only on an opaque batch, whose claims sit side by side
            # and partition the pixel, so the clamp gives exactly the area
            # average of the displayed colour -- over another sheet as much as
            # over the background, and across a crease. A reflected share
            # lands on the same area and stays out of the clamp, so a
            # reflector can come out brighter than that average, never
            # darker. Exempt: a glowing sheet, whose radiance bloom reads,
            # and a full claim, which the display clamps by itself.
            if (display_cap > 0.0) and (color[3] <= 0.0) \
                    and (alpha < 1.0 - _AA_FULL_DUST):
                ceiling = weight * (alpha * display_cap)
                lit = ti.math.vec4(ti.min(lit[0], ceiling[0]),
                                   ti.min(lit[1], ceiling[1]),
                                   ti.min(lit[2], ceiling[2]), lit[3])
            acc += lit
            # The direct lights' share of the REFLECTED lobe. The continuation
            # spawned below carries ``R`` of that lobe, but it is a ray and a
            # delta light has nothing for a ray to hit, so it can only return
            # the environment. Added here at exactly the weight the ray took,
            # which puts the lobe's total weight at 1 alongside ``share``.
            # See ``shading_taichi.direct_specular_lobe``. The 4th (glow)
            # lane takes nothing: glow is the authored bloom channel and a
            # Fresnel highlight is not authored emission, so every existing
            # bloom pass reads exactly what it read before.
            if ti.static(frag_shading != 0 and mode != 1 and direct_spec != 0):
                # Triangles/PN only. A bezier circuit is never material-
                # shaded -- its color is sampled, not lit, so it carries no
                # GGX term to restore -- and its ``prim`` indexes
                # ``circuit_meta``, not ``tri_mat``, so reading a material
                # block with it would silently pick another primitive's.
                if (not fetched_bez) and (reflectivity >= 0.0) and (T > 1e-4):
                    sa = (weight * alpha) * (R + trans_share) \
                        * direct_specular_lobe(
                            f, prim, surf_pos, -surf_rd, normal, geo_normal,
                            reflectivity, rough, ior, albedo3, tri_mat,
                            light_pos, light_col, num_lights,
                            ti.static(1 if mode == 2 else 0), lvis)
                    acc += ti.math.vec4(sa[0], sa[1], sa[2], 0.0)
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha

            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > min_alpha) and (cover_pass > min_alpha) \
                        and (bounces_left > 0):
                    split_refl = True

            sec_pm = 0
            sec_n = 0
            if ti.static(sec_aa > 1):
                sec_pm, sec_n = _sec_positions(msk_low, sec_aa)

            # The continuation fans the branches below ask for, which the one
            # fan site after them traces: at most the glass's refraction
            # (``fan_refract``, at ``fan_w0`` per tap across the face
            # ``fan_face_n``) and one reflection (``fan_reflect``, at
            # ``fan_w1`` per tap off ``fan_n1``), whose first tap re-aims the
            # primary when ``fan_to_primary``. A branch whose primary becomes
            # the reflection sets ``bounce_break``, and the walk breaks once
            # the fans are out.
            fan_refract = False
            fan_reflect = False
            fan_to_primary = False
            fan_w0 = ti.math.vec3(0.0, 0.0, 0.0)
            fan_w1 = ti.math.vec3(0.0, 0.0, 0.0)
            fan_n1 = ti.math.vec3(0.0, 0.0, 0.0)
            fan_face_n = ti.math.vec3(0.0, 0.0, 0.0)
            bounce_break = False

            if is_glass:
                wt = weight * trans_energy * tint
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > min_weight:
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
                        fan_refract = True
                        fan_w0 = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        fan_face_n = face_normal
                    else:
                        rdt = _refract_ray(surf_rd, interface_n, rel)
                        if ti.static(mode != 1):
                            # Refraction off a primary: push or pop the hit
                            # medium (see the fan site's refraction below).
                            _spawn_pool_ray(
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                rs_alloc,
                                _offset_transmitted_origin(
                                    hp, rdt, face_normal, normal),
                                rdt, wt, base_dist + t_hit,
                                bounces_left - 1, processed, pixel, r, r, 1,
                                ior_stack, 1, ior,
                                surf_rd.dot(face_normal) < 0.0)
                # The primary becomes the reflection only when nothing passes
                # beside the glass (``split_refl`` off, which for glass means
                # ``cover_pass <= min_alpha``). A partially covering sheet --
                # every silhouette pixel -- pools its reflection and keeps the
                # primary as the pass-through, as the opaque branch below
                # does. Choosing the stronger of the two here instead dropped
                # the pass-through whenever ``alpha * R >= 1 - alpha``: a dark
                # one-pixel rim round clear glass on a bright background.
                if (refl_max > min_alpha) and (not split_refl):
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hit_point = surf_pos
                    if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                            and (sec_n > 1):
                        weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                        fan_reflect = True
                        fan_to_primary = True
                        fan_w1 = weight
                        fan_n1 = nref
                    else:
                        rd = refl_rd
                        ro = _offset_ray_origin(hit_point, nref)
                        weight *= refl_energy
                    bounce_break = True
                else:
                    rwt = weight * refl_energy
                    rwt_max = ti.max(rwt[0], ti.max(rwt[1], rwt[2]))
                    if rwt_max > min_weight:
                        refl_rd, nref = _reflect_frame(surf_rd, normal,
                                                       geo_normal)
                        rhp = surf_pos
                        if ti.static(sec_aa > 1) \
                                and (rwt_max > sec_min_energy) and (sec_n > 1):
                            fan_reflect = True
                            fan_w1 = rwt * (1.0 / ti.cast(sec_n, ti.f32))
                            fan_n1 = nref
                        else:
                            if ti.static(mode != 1):
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                    rs_alloc,
                                    _offset_ray_origin(rhp, nref),
                                    refl_rd, rwt, base_dist + t_hit,
                                    bounces_left - 1, processed, pixel, r, r, 1,
                                    ior_stack, 0, 0.0, 0)
            elif prefilter_take:
                # The claim records ONE ray, the mirror direction, which the
                # walk spawns into the pixel's glossy row once it ends (below
                # the sheet loop). No ``sec_aa`` fan: the taps are exactly
                # what the prefilter replaces, and one deterministic ray in a
                # direction that is a smooth function of position is what stops
                # the reflection crawling under motion (REPORT.md §4.5).
                #
                # The primary does NOT become the reflection here. It carries
                # on down the sheet list as the pass-through, which is what
                # gives a partially covering reflector both its reflection and
                # what is behind it -- the same split ``split_refl`` performs
                # for a transparent one, now unconditional because the
                # reflection always has a pool slot of its own.
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > min_weight:
                    # Claimed here rather than on a successful spawn, so the
                    # event-build walk (mode 1, which spawns nothing) and the
                    # shading walk agree about which sheet took the pixel's one
                    # prefiltered event. They must: the substitution above
                    # changes ``share`` and therefore the running weight, and a
                    # walk that ran its weight out one sheet earlier than the
                    # other would build shadow events for a different image
                    # than the one being shaded.
                    if not gl_share:
                        if gl_taken:
                            # A TAKEOVER: the band holding the claim reflects
                            # less than this sheet, so it becomes what it would
                            # have been had it never claimed -- a throttled
                            # sharp reflection, the rest of its lobe shaded
                            # locally. Its ``share`` above was taken with ``E``,
                            # so the throttle's remainder goes back into
                            # ``acc`` now, and its ray leaves at the throttled
                            # weight into the pixel's own row. ``E`` stands in
                            # for its Schlick term, which is the difference
                            # left: the energy is the lobe's either way. One
                            # ray, not a ``sec_aa`` fan: the taps' surface
                            # points are gone.
                            gl_ms = _mirror_share(gl_rough)
                            acc += (1.0 - gl_ms) * ti.math.vec4(
                                gl_back[0], gl_back[1], gl_back[2], gl_glow)
                            if ti.static(mode != 1):
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc, gl_org, gl_dir,
                                    gl_wt * gl_ms, gl_dist, gl_bounces,
                                    gl_proc, pixel, r, r, 1,
                                    ior_stack, 0, 0.0, 0)
                        gl_taken = True
                        gl_wt = ti.math.vec3(0.0, 0.0, 0.0)
                        gl_back = ti.math.vec3(0.0, 0.0, 0.0)
                        gl_glow = 0.0
                        # The direction, radius and depth are the claiming
                        # sibling's: there is one row, and a lobe wide enough
                        # to be worth prefiltering at all spans a crease
                        # either way.
                        refl_rd, nref = _reflect_frame(surf_rd, normal,
                                                       geo_normal)
                        gl_org = _offset_ray_origin(surf_pos, nref)
                        gl_dir = refl_rd
                        gl_dist = base_dist + t_hit
                        gl_bounces = bounces_left - 1
                        gl_proc = processed
                        gl_rough = rough
                        if ti.static(mode != 1):
                            # A circuit is planar; a triangle may belong to a
                            # curved mesh, whose reflection the footprint
                            # must shrink by (DESIGN_glossy_prefilter.md §3).
                            gl_mag = 1.0
                            if not fetched_bez:
                                gl_mag = _reflector_magnification(
                                    f, prim, surf_rd, tri_pos, tri_norm,
                                    base_dist + t_hit)
                    # The claim now belongs to THIS sheet's band, and survives
                    # exactly as long as the band does.
                    gl_band = defer
                    # A band's siblings share one row, so ``W`` is their SUM --
                    # the weight the band would have spawned unsplit.
                    gl_wt += wt
                    gl_back += wt * ti.math.vec3(color[0], color[1], color[2])
                    gl_glow += w_glow * alpha * r_glow * color[3]
            elif is_pane or split_refl:
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > min_weight:
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hp = surf_pos
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        fan_reflect = True
                        fan_w1 = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        fan_n1 = nref
                    else:
                        if ti.static(mode != 1):
                            _spawn_pool_ray(
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                rs_alloc,
                                _offset_ray_origin(hp, nref),
                                refl_rd,
                                wt, base_dist + t_hit, bounces_left - 1,
                                processed, pixel, r, r, 1,
                                ior_stack, 0, 0.0, 0)
            elif (refl_max > min_alpha) and (refl_max >= cover_pass):
                refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                hit_point = surf_pos
                if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                        and (sec_n > 1):
                    weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                    fan_reflect = True
                    fan_to_primary = True
                    fan_w1 = weight
                    fan_n1 = nref
                else:
                    rd = refl_rd
                    ro = _offset_ray_origin(hit_point, nref)
                    weight *= refl_energy
                bounce_break = True

            # The occlusion write, reached by every branch above that does not
            # turn the primary into a reflection (``bounce_break``). Glass and
            # the prefiltered claim pass their light on through rays of their
            # own, so they occlude as opaque; everything else keeps
            # ``trans_share`` of what it covers, through its tint.
            #
            # A band's siblings write once, at the sheet that closes the band,
            # as the unsplit surface would have: each adds its coverage factor
            # times ITS OWN alpha, and what of that it transmits per channel,
            # to the band's sums. Taking the closing sibling's alpha for the
            # whole band, as this did, let a crease between faces of different
            # opacity occlude by whichever face the walk reached last.
            if not bounce_break:
                ts_w = trans_share
                if is_glass or prefilter_take:
                    ts_w = 0.0
                if defer or cont_band:
                    pa = cfac * a_s
                    band_pa += pa
                    band_pat += pa * ts_w
                    band_patc += (pa * ts_w) * tint
                if not defer:
                    if cont_band:
                        weight *= _band_write(svis, slots, own_msk, band_pa,
                                              band_pat, band_patc)
                    else:
                        weight *= _occlusion_write(
                            svis, slots, own_msk, a_s, ts_w, cfac, tint,
                            a_s * ts_w > 1e-6)
                    band_pc = 0.0
                    band_claim = 0.0
                    band_pa = 0.0
                    band_pat = 0.0
                    band_patc = ti.math.vec3(0.0, 0.0, 0.0)

            # THE CONTINUATION FANS: one site for the up to two the branches
            # above asked for, so the kernel unrolls ``fan_taps`` jittered
            # surface samples once, where it unrolled them in each of the five
            # branches that can fan. Job 0 is the glass's refraction, job 1 the
            # reflection; a runtime loop runs whichever were asked for, in that
            # order. The fans now follow the branch's single-ray spawns rather
            # than preceding them, which reorders a pixel's rays in the pool
            # and nothing else: a batch that can fan splits, and a splitting
            # batch commits on the order-independent grid (``det_sum``).
            if ti.static(sec_aa > 1 and fan_taps > 0):
                fan_first = 0 if fan_refract else 1
                fan_jobs = (1 if fan_refract else 0) + (1 if fan_reflect else 0)
                for fj in range(fan_jobs):
                    job = fan_first + fj
                    n_in = normal
                    fw = fan_w0
                    if job == 1:
                        n_in = fan_n1
                        fw = fan_w1
                    to_pool = (job == 0) or (not fan_to_primary)
                    jtap = 0
                    for s in ti.static(range(fan_taps)):
                        if (sec_pm >> s) & 1:
                            rdj, hpj, nj, _b1, _b2 = _jittered_surface_sample(
                                f, px, py,
                                ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                gen_meta, fetched_bez, prim, surf_pos, n_in,
                                tri_pos, tri_norm, tri_uvs, tri_tex_meta,
                                textures, num_colored_triangles,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y, cam_proj)
                            if job == 0:
                                # Refraction off a primary: push or pop the
                                # hit medium, the side read from the GEOMETRIC
                                # face normal (see _relative_ior). Not assumed
                                # to be an entry: a partially covering glass
                                # fragment lets the primary walk on to the
                                # solid's back face, where an unconditional
                                # push would record a medium the ray is
                                # leaving.
                                entering = rdj.dot(fan_face_n) < 0.0
                                nj = _transmission_normal(rdj, nj, fan_face_n)
                                rel_j = _relative_ior(rs_sca, r, ior, entering,
                                                      ior_stack)
                                rdt = _refract_ray(rdj, nj, rel_j)
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc,
                                    _offset_transmitted_origin(
                                        hpj, rdt, fan_face_n, nj),
                                    rdt, fw, base_dist + t_hit,
                                    bounces_left - 1, processed, pixel, r, r, 1,
                                    ior_stack, 1, ior, entering)
                            else:
                                rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                                if ti.static(glossy == 1 or glossy == 2):
                                    if rough > _GLOSSY_MIN_ROUGHNESS:
                                        rdr = _glossy_reflect(
                                            rdj, nj, rough, jtap, sec_n,
                                            g_roff, g_aoff)
                                jtap += 1
                                org = _offset_ray_origin(hpj, nj)
                                if to_pool:
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc, org, rdr, fw,
                                        base_dist + t_hit, bounces_left - 1,
                                        processed, pixel, r, r, 1,
                                        ior_stack, 0, 0.0, 0)
                                else:
                                    rd = rdr
                                    ro = org
                                    to_pool = True
            if bounce_break:
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
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < min_weight:
                done = True
                break

        if band_open and (not bounced):
            # The walk stopped inside a band -- at the far clip, or at the
            # surface cap -- before the sibling that would have written it.
            # The siblings it did reach claimed, so what they cover still
            # hides the background.
            b_low = band_msk & _AA_MASK_ALL
            b_areal = ((band_msk & _AA_SLIVER_BIT) != 0) or (b_low == 0)
            b_own = b_low
            if b_areal:
                b_own = _AA_MASK_ALL
            weight *= _band_write(svis, _sheet_slots(band_msk, b_areal), b_own,
                                  band_pa, band_pat, band_patc)

        if not bounced:
            vis_all = 0.0
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                vis_all += svis[s]
            weight *= vis_all * _AA_SAMPLE_WEIGHT

        if processed >= max_surfaces_per_ray:
            # Truncation, not completion (``truncation.py``): ``done`` is
            # already set for a walk that ended at the far clip or ran its
            # weight out, and ``q == total`` means every sheet was composited.
            # What is left -- sheets unwalked, or a bounce this retires -- is
            # transport the ceiling is dropping. Counted on the SHADING pass
            # only: a shadowed batch runs this body twice over the same sheets
            # (mode 1 builds events, mode 2 shades) and would count each ray
            # twice. The store is unchanged, so the frame is byte-identical.
            if ti.static(mode != 1):
                if (not done) and (bounced or (q < total)):
                    ti.atomic_add(rs_alloc[ALLOC_TRUNC_SURFACES], 1)
            done = True

        if ti.static(dump):
            if dmatch:
                d_vis = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    d_vis += svis[s]
                _aa_dump_terminal(dump_out, bounced, done, processed,
                                  d_vis * _AA_SAMPLE_WEIGHT, acc, weight,
                                  svis)

        # The prefiltered claim's ONE ray, spawned now that no later sheet can
        # take the claim: mirror direction, throughput 1, into this pixel's
        # GLOSSY row (``r + num_covered``) instead of its own, with the
        # energy, radius, depth and magnification the composite needs parked
        # in that row (DESIGN_glossy_prefilter.md §4.2). The radius is the
        # lobe's angular width in pixels, scaled at composite time by how far
        # past the reflector the reflected content turned out to be.
        if ti.static(glossy == 3 and mode != 1):
            if gl_taken:
                gl_row = r + num_covered
                if _spawn_pool_ray(
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                        rs_alloc, gl_org, gl_dir,
                        ti.math.vec3(1.0, 1.0, 1.0), gl_dist, gl_bounces,
                        gl_proc, pixel, r, gl_row, 1, ior_stack, 0, 0.0, 0):
                    for k in ti.static(range(3)):
                        pix_accum[gl_row, GL_ROW_W + k] = _det_deposit(
                            gl_wt[k], det_sum)
                    pix_accum[gl_row, GL_ROW_SIGMA_SCALE] = (
                        2.0 * gl_rough * gl_rough * gl_px_per_rad)
                    pix_accum[gl_row, GL_ROW_DP] = gl_dist
                    pix_accum[gl_row, GL_ROW_MAG] = gl_mag

        # The event pass owns no ray state and commits no pixels; its
        # writes were the per-sheet event tables above, so mode 1 ends
        # the sheet walk here.
        #
        # Gated rather than skipped with a ``continue``, and the
        # difference is not style: ``ti.static`` resolves at compile
        # time, so ``if ti.static(mode == 1): continue`` leaves a BARE
        # ContinueStmt in the loop body followed by every statement
        # below it. Taichi's SPIR-V codegen emits a ``continue`` as an
        # OpBranch and relies on the next IfStmt boundary to open a new
        # OpLabel; with no boundary to reach, everything after it lands
        # in a block that has already been terminated, and the module is
        # invalid SPIR-V (``Load must appear in a block``). LLVM does not
        # care, so CPU and CUDA rendered it correctly for as long as it
        # existed; Metal answers with a nil pipeline and Vulkan with
        # ``vkCreateComputePipelines failed``, which is what took six
        # shadow tests down on the Apple GPU (DESIGN_mps_support.md
        # 1.2c). taichi_patches/0002 fixes the codegen too, so a stray
        # ``continue`` cannot do this again; this shape is what the
        # kernel wants regardless.
        if ti.static(mode != 1):
            if bounced and not done:
                _reset_shell_segment(rs_sca, r)
                for k in ti.static(range(3)):
                    rs_ro[r, k] = ro[k]
                    rs_rd[r, k] = rd[k]
                for k in ti.static(range(4)):
                    rs_acc[r, k] = acc[k]
                # Columns 7+ (nested-IOR stack) stay untouched: a bounced PRIMARY
                # reflected off this hit and primaries start in air, so its stack
                # is still the host-zeroed empty one.
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
                # The GEOMETRIC share of the leftover, for the display-referred
                # coverage resolve (``ACC_GEO``). A primary that never bounced
                # only ever multiplied its weight by sample visibility, so what
                # is left of it is the fraction of the PIXEL AREA no sheet
                # covered -- an area, which the composite may clamp per part.
                # A bounced ray's leftover is reflected/transmitted throughput
                # through an area the geometry does cover, which must not be
                # clamped that way, so it deposits nothing here and the
                # composite falls back to the plain linear blend.
                if not bounced:
                    ti.atomic_add(pix_accum[r, ACC_GEO], weight[0])
                # Background-as-final-sheet (§4.5): the frame buffer is
                # prefilled with the background — env map included, via
                # env_background_prefill — and the composite multiplies the
                # leftover weight by it, so the weight is handed through
                # rather than folded with the env here. The primary retire
                # direction IS the prefill direction; sampling the env again
                # would count it twice. Bounced rays retire in wavefront_shade,
                # which still samples the env with THEIR direction (the pixel's
                # background would be the wrong ray).
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[r, k], _det_deposit(acc[k], det_sum))
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[r, 4 + k],
                                  _det_deposit(weight[k], det_sum))


#: What ``sheet_resolve_shade`` binds through the arena, in offset-table order:
#: ``aoff[i]`` is the i-th entry's element offset into its dtype's
#: buffer and ``ashp`` holds their shapes end to end. The kernel's
#: binding prologue reads those slots by literal index, so the two
#: are one edit apart -- ``tests/unit_tests/test_arena_args.py``
#: fails if they stop agreeing.
_SHEET_RESOLVE_SHADE_ARENA = (
    ("sheet_offsets", "i32", 1),
    ("sheet_key", "i64", 1),
    ("sheet_ref", "i32", 1),
    ("sheet_ab", "f32", 2),
    ("sheet_cov", "f32", 1),
    ("sheet_msk", "i32", 1),
    ("sheet_cap", "f32", 1),
    ("tri_pos", "f32", 3),
    ("tri_norm", "f32", 3),
    ("tri_extra", "f32", 3),
    ("tri_colors", "f32", 4),
    ("tri_uvs", "f32", 3),
    ("tri_tex_meta", "i32", 2),
    ("textures", "f32", 3),
    ("col_row", "i32", 1),
    ("tri_mat_id", "i32", 2),
    ("tri_mat", "f32", 3),
    ("circuit_meta", "f32", 3),
    ("circuit_colors", "f32", 4),
    ("circuit_border_colors", "f32", 4),
    ("light_pos", "f32", 3),
    ("light_col", "f32", 3),
    ("layer_offsets", "f32", 1),
    ("sheet_memo", "f32", 2),
    ("sheet_accept", "i32", 1),
    ("event_pos", "f32", 2),
    ("event_snrm", "f32", 2),
    ("event_fnrm", "f32", 2),
    ("event_frame", "i32", 1),
    ("event_msk", "i32", 1),
    ("event_dp", "f32", 2),
    ("event_toff", "f32", 2),
    ("sheet_event_id", "i32", 1),
    ("shadow_vis", "f32", 3),
    ("covered_idx", "i32", 1),
    ("cam_origin", "f32", 2),
    ("screen_point", "f32", 2),
    ("pixel_basis_x", "f32", 2),
    ("pixel_basis_y", "f32", 2),
    ("gen_meta", "f32", 1),
    ("rs_alloc", "i32", 1),
    ("cam_proj_table", "f32", 2),
)

#: The argument list every launch site passes. Unchanged by the
#: conversion -- that is the point of the wrapper below.
_SHEET_RESOLVE_SHADE_PARAMS = (
    "num_covered", "sheet_offsets", "sheet_key", "sheet_ref", "sheet_ab",
    "sheet_cov", "sheet_msk", "sheet_cap", "tri_pos", "tri_norm", "tri_extra",
    "tri_colors", "tri_uvs", "tri_tex_meta", "textures",
    "num_colored_triangles", "col_row", "tri_mat_id", "tri_mat",
    "circuit_meta", "circuit_colors", "circuit_border_colors", "light_pos",
    "light_col", "num_lights", "vis_lights", "layer_offsets", "frag_shading",
    "frag_pipelines", "tri_pids", "refraction", "ior_stack",
    "skip_unlit_normal", "has_bez", "sec_aa", "sec_min_energy", "glossy",
    "direct_spec", "mode", "shadow_term", "sided_cull",
    "memo", "det_sum",
    "sheet_memo", "sheet_accept", "event_pos", "event_snrm", "event_fnrm",
    "event_frame", "event_msk", "event_dp", "event_toff", "sheet_event_id",
    "shadow_vis", "covered_idx", "time_start", "width", "height",
    "cam_origin", "screen_point", "pixel_basis_x", "pixel_basis_y",
    "gen_meta", "rs_ro", "rs_rd", "rs_acc", "rs_sca", "rs_int", "rs_pix",
    "pix_accum", "rs_alloc", "dump", "dump_out", "cam_proj_table", "projected",
    "display_cap",
)

_sheet_resolve_shade_launch = arena_packed(
    __name__, "sheet_resolve_shade_arena",
    _SHEET_RESOLVE_SHADE_PARAMS, _SHEET_RESOLVE_SHADE_ARENA)


def sheet_resolve_shade(*args):
    """Pack the arena-bound arguments, then launch ``sheet_resolve_shade_arena``.

    Takes the argument list this kernel had before it was converted to
    the arena calling convention, so no launch site changed; see
    `arena_args_taichi`.
    """
    return _sheet_resolve_shade_launch(*args)


@ti.kernel
def env_background_prefill(
        num_frames: int, width: int, height: int, start_frame: int,
        jx: ti.f32, jy: ti.f32, half_w: ti.f32, half_h: ti.f32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        cam_proj_table: ti.types.ndarray(),
        projected: ti.template(),
        env_off: int, env_w: int, env_h: int, env_intensity: ti.f32,
        textures: ti.types.ndarray(),
        byte_out: ti.template(),
        out: ti.types.ndarray()):
    """Prefill the frame buffer with the environment map, per (frame, pixel).

    Background-as-final-sheet (DESIGN_sheet_resolve.md §4.5): the resolve
    emits a leftover weight per covered pixel and the composite multiplies it
    by whatever the buffer holds — a flat color, an image plate, or this. An
    empty pixel is then already final with no resolve launch at all, which is
    what lets an env-mapped scene take the sparse covered-pixel route instead
    of forcing the dense one. Byte-scale to match ``_prefill_background``,
    which also means rounded and clamped to ``[0, 255]`` when ``byte_out``
    says the buffer is uint8 (post-process tonemapping off): a map brighter
    than 1 would otherwise wrap, 1.2 storing as 50. The float HDR buffer keeps
    the unclamped value for the tonemap. Column 3, when present, is the glow
    lane and gets 0. A transparent background's alpha column (4) is written
    opaque: the map replaces the backdrop (``Scene.set_environment_map``),
    which is what the dense path's env retire and the path tracer's escape
    already give it.

    Every ray of an orthographic camera runs along the forward axis, so it
    sees the map in that one direction: a uniform backdrop, which is what
    parallel projection of an infinitely distant sphere is.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    ppf = width * height
    for i in range(num_frames * ppf):
        f_rel = i // ppf
        p = i - f_rel * ppf
        f = start_frame + f_rel
        py = p // width
        px = p - py * width
        _ro, rd = _generate_ray(f, px, py, jx, jy, half_w, half_h,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y, cam_proj)
        ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                             env_intensity, textures)
        for k in ti.static(range(3)):
            if ti.static(byte_out):
                out[f_rel, p, k] = ti.cast(
                    ti.math.clamp(ec[k] * 255.0 + 0.5, 0.0, 255.0), ti.u8)
            else:
                out[f_rel, p, k] = ec[k] * 255.0
        if out.shape[2] > 3:
            # Column 3 is the GLOW lane, not alpha: the sky emits none, which
            # is also what the dense path's env retire deposits (acc[3] == 0,
            # weight zeroed). Writing 255 here bloomed every pixel white.
            out[f_rel, p, 3] = 0.0
        if out.shape[2] > 4:
            # Column 4 is the alpha a transparent background brings, and the
            # map covers it: the composite's leftover weight then lands on an
            # opaque backdrop, as the dense path's env retire makes every
            # escaped ray (weight zeroed, so the coverage alpha is 255).
            out[f_rel, p, 4] = 255.0
