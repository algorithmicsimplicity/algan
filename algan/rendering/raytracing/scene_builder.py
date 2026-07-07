"""Collection of helper functions used to combine collections of primitives
into contiguous tensor data-structures, ready to be shipped to ray tracing kernels.
"""
import torch
import torch.nn.functional as F

from algan.utils.memory_utils import empty_cache
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.rendering.raytracing.primitives import RayTracedPNTrianglePrimitive, RayTracedTrianglePrimitive, \
    RayTracedBezierCircuitPrimitive
from algan.rendering.raytracing.settings import _constant_promotion_active
from algan.rendering.raytracing.shading_taichi import MAT_W, _MID_UNLIT
from algan.rendering.raytracing.stbvh import EMPTY_LO, EMPTY_HI, build_stbvh
from algan.rendering.raytracing.utils import _expand_frames, _cat_collections, _cat_mat_blocks, _flat_frames


def _dedup_time(x):
    """Collapse a leading (time) dimension that is constant across frames to
    length 1, so a temporally-constant map/colour is stored once instead of T
    times. The kernels index the time axis as ``f % shape[0]``, so a length-1
    axis is read by every frame."""
    if x.shape[0] > 1 and bool((x == x[:1]).all()):
        return x[:1].contiguous()
    return x


def _split_promotable(p, _append_texture, device, scene):
    """Partition a non-textured triangle primitive into the triangles that must
    stay per-vertex and the triangles whose colour + material are constant
    across their three corners and every frame (and are non-glowing). The
    constant triangles are grouped by value -- so a uniform mob is one group even
    when it was batched into a primitive alongside differently-coloured mobs --
    and each group is promoted to one shared 1x1 colour map + 1x1 material map
    (appended here to the shared texel buffer).

    Returns ``(keep_idx, promo_idx, promo_meta)``: ascending ``keep_idx`` selects
    the per-vertex triangles; ``promo_idx`` selects the promoted triangles
    grouped by value; ``promo_meta`` is the ``[len(promo_idx), 10]`` tex-meta
    (colour map cols 0-2, material map 3-5, no normal map 6-8 = -1, bitmask 9 =
    refl|rough|ior) aligned to ``promo_idx``. The kernel reads all three material
    properties from the material map, so promoted triangles need no per-vertex
    ``tri_colors``/``tri_extra`` row."""
    colors = p._rt_tri_colors           # [Tc, N, 3, 5]
    extra = p._rt_tri_extra             # [Te, N, 15]
    N = colors.shape[1]
    all_idx = torch.arange(N, device=device)
    if N == 0:
        empty = torch.zeros((0, 10), dtype=torch.int32, device=device)
        return all_idx, all_idx, empty
    # Per-triangle promotable: the three corners share one colour (all channels,
    # all frames) and one material (reflectivity 0/2/4, roughness 1/3/5, index of
    # refraction 6/7/8), and the triangle is non-glowing (glow magnitude cols
    # 9-11 zero; a nonzero default glow_radius in 12-14 is irrelevant once glow
    # is 0). Only such a triangle is fully described by a single 1x1 texel.
    color_eq = (colors == colors[:, :, :1, :]).all(-1).all(-1).all(0)      # [N]
    e = extra
    mat_eq = ((e[..., 0] == e[..., 2]) & (e[..., 0] == e[..., 4])
              & (e[..., 1] == e[..., 3]) & (e[..., 1] == e[..., 5])
              & (e[..., 6] == e[..., 7]) & (e[..., 6] == e[..., 8])).all(0)  # [N]
    nonglow = (e[..., 9:12] == 0).all(-1).all(0)                            # [N]
    promotable = color_eq & mat_eq & nonglow
    keep_idx = all_idx[~promotable]
    promo_all = all_idx[promotable]
    if promo_all.numel() == 0:
        empty = torch.zeros((0, 10), dtype=torch.int32, device=device)
        return keep_idx, promo_all, empty

    # Group promoted triangles by their (per-frame) constant colour + material
    # value, so identical mobs share one pair of maps. The key is the corner-0
    # colour [T,5] plus material (refl, rough, ior) [T,3] over all frames.
    Tc, Te = colors.shape[0], extra.shape[0]
    T = max(Tc, Te)
    col0 = _expand_frames(colors[:, :, 0, :], T)[:, promo_all, :]           # [T,P,5]
    mat3 = _expand_frames(
        torch.stack([extra[..., 0], extra[..., 1], extra[..., 6]], -1),
        T)[:, promo_all, :]                                                 # [T,P,3]
    key = torch.cat([col0, mat3], -1).permute(1, 0, 2).reshape(
        promo_all.numel(), -1)                                             # [P, 8T]
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)             # inv [P]
    order = torch.argsort(inv, stable=True)   # group identical values contiguously
    promo_idx = promo_all[order]
    inv_sorted = inv[order]

    # One colour + material map per distinct value; each promoted triangle's meta
    # row points at its group's maps.
    group_meta = []
    for gid in range(uniq.shape[0]):
        rep = int(promo_all[int((inv == gid).nonzero()[0])])
        cmap = _dedup_time(colors[:, rep:rep + 1, 0, :].contiguous())      # [T',1,5]
        color_meta = _append_texture(
            cmap.reshape(cmap.shape[0], 1, 1, 5).float().contiguous())
        e0 = extra[:, rep:rep + 1, :]
        z = torch.zeros_like(e0[..., 0])
        mmap = _dedup_time(torch.stack(
            [e0[..., 0], e0[..., 1], e0[..., 6], z, z], -1).contiguous())
        material_meta = _append_texture(
            mmap.reshape(mmap.shape[0], 1, 1, 5).float().contiguous())
        if bool((mmap[..., 2] > 1.0 + 1e-4).any()):
            scene["tex_has_refractive"] = True
        group_meta.append([*color_meta, *material_meta, -1, 0, 0, 1 | 2 | 4])
    group_meta = torch.tensor(group_meta, dtype=torch.int32, device=device)
    promo_meta = group_meta[inv_sorted]                                    # [P,10]
    return keep_idx, promo_idx, promo_meta


def _empty_scene_part(device):
    """Placeholder STBVH + arrays for an absent geometry type."""
    lo = torch.full((1, 1, 3), EMPTY_LO, device=device)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=device)
    return build_stbvh(lo, hi, num_frames=1)


def _build_mem_trim(scene, lo, hi, opaque, num_frames, device):
    """Build the 'Family A+B' memory-trim triangle arrays (see
    settings.WF_MEM_TRIM). Reorders prims into material-class bands -- band 0
    ``needs_mat`` (lit), band 1 ``needs_norm`` only (reflective / normal-mapped /
    promoted), band 2 bare (unlit matte) -- so that ``tri_norm`` and ``tri_mat``
    become compacted PREFIXES (needs_mat subset needs_norm, so both nest under a
    single permutation). ``tri_colors``/``tri_extra`` stay in their original
    (promotion-compacted) order, addressed by a per-prim remap ``col_row`` (-1 =
    promoted, colour/material from its 1x1 maps); ``tex_meta``/``uvs`` are widened
    to full band-order arrays indexed directly by prim. Byte-identical to the
    untrimmed path (only indexing/layout changes). Stores ``*_t`` variants +
    ``col_row`` + a band-reordered BVH; the wavefront picks them when engaged."""
    tri_pos = scene["tri_pos"].to(device)
    N = tri_pos.shape[1]
    if N == 0:
        scene["mem_trim_active"] = False
        return
    tri_norm = scene["tri_norm"].to(device)
    tri_mat = scene["tri_mat"].to(device)
    tri_mat_id = scene["tri_mat_id"].to(device)
    tri_extra = scene["tri_extra"].to(device)
    tri_uvs = scene["tri_uvs"].to(device)
    tri_tex_meta = scene["tri_tex_meta"].to(device)
    num_colored = int(scene["num_colored_triangles"])
    _UNLIT = 1
    Nc = tri_extra.shape[1]        # prims with a per-vertex colour/extra row

    lit = (tri_mat_id != _UNLIT).any(0)                       # [N]
    refl = torch.zeros(N, dtype=torch.bool, device=device)
    if Nc > 0:
        e = tri_extra
        refl[:Nc] = ((e[..., 0] > 0) | (e[..., 2] > 0)
                     | (e[..., 4] > 0)).any(0)
    promoted = torch.zeros(N, dtype=torch.bool, device=device)
    if Nc < N:
        promoted[Nc:] = True       # constant-material prims: value in 1x1 map
    normalmapped = torch.zeros(N, dtype=torch.bool, device=device)
    if tri_tex_meta.shape[0] > 0 and num_colored < N:
        nm = tri_tex_meta[:, 6] >= 0
        k = min(nm.shape[0], N - num_colored)
        normalmapped[num_colored:num_colored + k] = nm[:k]

    needs_mat = lit
    needs_norm = needs_mat | refl | promoted | normalmapped
    n_lit = int(needs_mat.sum().item())
    n_norm = int(needs_norm.sum().item())

    zeros = torch.zeros(N, dtype=torch.long, device=device)
    band = torch.where(needs_mat, zeros,
                       torch.where(needs_norm, zeros + 1, zeros + 2))
    perm = torch.argsort(band, stable=True)                   # band 0 first
    orig = perm                                               # orig idx of prim p

    tri_pos_t = tri_pos.index_select(1, perm).contiguous()
    tri_norm_t = tri_norm.index_select(
        1, perm)[:, :max(n_norm, 1)].contiguous()
    tri_mat_t = tri_mat.index_select(1, perm)[:, :max(n_lit, 1)].contiguous()
    tri_mat_id_t = tri_mat_id.index_select(1, perm).contiguous()
    col_row = torch.where(orig < Nc, orig,
                          torch.full_like(orig, -1)).to(torch.int32).contiguous()

    tex_meta_t = torch.zeros((N, 10), dtype=torch.int32, device=device)
    tex_meta_t[:, 0] = -1
    tex_meta_t[:, 3] = -1
    tex_meta_t[:, 6] = -1
    Tuv = tri_uvs.shape[0]
    tri_uvs_t = torch.zeros((Tuv, N, 6), dtype=tri_uvs.dtype, device=device)
    if tri_tex_meta.shape[0] > 0:
        has_meta = orig >= num_colored
        meta_src = (orig - num_colored).clamp(0, tri_tex_meta.shape[0] - 1)
        tex_meta_t = torch.where(has_meta.unsqueeze(1),
                                 tri_tex_meta.index_select(0, meta_src).int(),
                                 tex_meta_t)
        uv_src = (orig - num_colored).clamp(0, tri_uvs.shape[1] - 1)
        tri_uvs_t = (tri_uvs.index_select(1, uv_src)
                     * has_meta.view(1, N, 1).to(tri_uvs.dtype))

    tri_bvh_t = build_stbvh(
        lo.index_select(1, perm).contiguous(),
        hi.index_select(1, perm).contiguous(),
        num_frames=num_frames,
        tightness=RayTracedTrianglePrimitive.stbvh_tightness,
        opaque=opaque.index_select(1, perm).contiguous(), builder="split")

    scene["tri_pos_t"] = tri_pos_t
    scene["tri_norm_t"] = tri_norm_t
    scene["tri_mat_t"] = tri_mat_t
    scene["tri_mat_id_t"] = tri_mat_id_t
    scene["tri_uvs_t"] = tri_uvs_t
    scene["tri_tex_meta_t"] = tex_meta_t
    scene["tri_col_row"] = col_row
    scene["tri_bvh_t"] = tri_bvh_t
    scene["mem_trim_active"] = True


def _promote_property_group(cv, present, num_frames, device):
    """Promote one per-corner property group of a flat-triangle batch to a
    texture bank (see settings.WF_TEXTURED).

    ``cv`` is the per-corner value tensor ``[T, N, 3, C]`` (T frames, N
    triangles, 3 corners, C channels) and ``present`` a ``[N]`` bool mask of the
    triangles that actually carry this group (others get index -1 and sample
    nothing). A triangle whose three corners are equal in every frame is
    *constant across the surface* and is promoted to a shared 1x1 texture
    (grouped by value, so identical surfaces share one texel); one that varies
    per vertex gets its own 2x2 texture laid out ``[[v0, v1], [v2, v0]]`` so a
    bilinear lookup at the canonical corner UVs ``(0,0)/(1,0)/(0,1)``
    reproduces the corner values exactly and blends between them in the
    interior (an approximation of true barycentric interpolation).

    Returns ``(bank, meta, idx)``: ``bank`` is the flat texel buffer
    ``[Tb, num_texels, C]``, ``meta`` the ``[num_textures, 3]`` int32
    ``(offset, width, height)`` per texture and ``idx`` the ``[N]`` int32
    per-triangle texture index (-1 = absent).
    """
    T = num_frames
    cv = _expand_frames(cv, T).contiguous()
    N, C = cv.shape[1], cv.shape[3]
    idx = torch.full((N,), -1, dtype=torch.int32, device=device)
    tri_ids = torch.arange(N, device=device)
    # Constant across the surface: all three corners equal in every frame.
    const_mask = (cv == cv[:, :, :1, :]).all(3).all(2).all(0)  # [N]

    banks, metas = [], []
    texel_off = 0
    meta_base = 0

    def _emit(sel, texels_flat, per_tex_texels, w, h, inv):
        # texels_flat: [T, G * per_tex_texels, C]; inv: [len(sel)] group id.
        nonlocal texel_off, meta_base
        G = texels_flat.shape[1] // per_tex_texels
        banks.append(texels_flat)
        offs = texel_off + torch.arange(G, device=device,
                                        dtype=torch.int32) * per_tex_texels
        wv = torch.full((G,), w, dtype=torch.int32, device=device)
        hv = torch.full((G,), h, dtype=torch.int32, device=device)
        metas.append(torch.stack([offs, wv, hv], -1))
        idx[sel] = inv.to(torch.int32) + meta_base
        texel_off += G * per_tex_texels
        meta_base += G

    # Constant group -> one 1x1 texel per distinct value-over-time.
    cc = present & const_mask
    if bool(cc.any()):
        sel = tri_ids[cc]
        vals = cv[:, sel, 0, :]                                  # [T, nc, C]
        key = vals.permute(1, 0, 2).reshape(sel.numel(), T * C)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        G = uniq.shape[0]
        texels = uniq.reshape(G, T, C).permute(1, 0, 2).contiguous()  # [T,G,C]
        _emit(sel, texels, 1, 1, 1, inv)

    # Per-vertex group -> one 2x2 texture per distinct (v0, v1, v2)-over-time.
    cvary = present & ~const_mask
    if bool(cvary.any()):
        sel = tri_ids[cvary]
        vals = cv[:, sel, :, :]                                  # [T, nv, 3, C]
        key = vals.permute(1, 0, 2, 3).reshape(sel.numel(), T * 3 * C)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        G = uniq.shape[0]
        u = uniq.reshape(G, T, 3, C)
        v0, v1, v2 = u[:, :, 0, :], u[:, :, 1, :], u[:, :, 2, :]  # [G,T,C]
        # Column-major texel order (offset + cx*h + cy, h=2): texel(0,0)=v0,
        # texel(0,1)=v2, texel(1,0)=v1, texel(1,1)=v0 -> [[v0,v1],[v2,v0]].
        texs = torch.stack([v0, v2, v1, v0], 2)                  # [G,T,4,C]
        texels = texs.permute(1, 0, 2, 3).reshape(T, G * 4, C).contiguous()
        _emit(sel, texels, 4, 2, 2, inv)

    if banks:
        bank = _dedup_time(torch.cat(banks, 1).contiguous())
        meta = torch.cat(metas, 0).contiguous()
    else:  # nothing in this group carries a texture
        bank = torch.zeros((1, 1, C), device=device)
        meta = torch.zeros((1, 3), dtype=torch.int32, device=device)
    return bank, meta, idx


def _build_textured_scene(scene, num_frames, device):
    """Build the three per-triangle texture banks the textured wavefront shades
    from (see settings.WF_TEXTURED), from the full per-vertex merged arrays
    (constant-promotion is disabled for this path so they span every triangle).

    Groups, each promoted independently by :func:`_promote_property_group`:

    * **colour** -- RGBA + glow (``tri_colors`` 5 channels, per vertex).
    * **surface** -- reflectivity / roughness / index-of-refraction (from
      ``tri_extra``, per vertex) used for scatter; index -1 for a matte surface
      (no reflectivity, no refraction) so the kernel skips the lookup.
    * **material** -- the shading parameter block prefixed with the pipeline id
      (``tri_mat_id`` + ``tri_mat``, per primitive, hence always 1x1); index -1
      for an unlit surface (no shading, colour passes through).

    Every triangle is assigned the canonical corner UVs ``(0,0)/(1,0)/(0,1)``.
    """
    T = num_frames
    tc = _expand_frames(scene["tri_colors"].to(device), T)     # [T,N,3,5]
    te = _expand_frames(scene["tri_extra"].to(device), T)      # [T,N,15]
    tm = _expand_frames(scene["tri_mat"].to(device), T)[..., :MAT_W]  # [T,N,12]
    tmi = _expand_frames(scene["tri_mat_id"].to(device), T)    # [T,N]
    N = tc.shape[1]

    # Colour: every triangle carries a colour.
    present = torch.ones(N, dtype=torch.bool, device=device)
    col_bank, col_meta, col_idx = _promote_property_group(
        tc, present, T, device)

    # Surface (scatter): per-corner (reflectivity, roughness, index of
    # refraction) gathered from tri_extra cols {0,2,4}/{1,3,5}/{6,7,8}.
    c0 = torch.stack([te[..., 0], te[..., 1], te[..., 6]], -1)  # [T,N,3]
    c1 = torch.stack([te[..., 2], te[..., 3], te[..., 7]], -1)
    c2 = torch.stack([te[..., 4], te[..., 5], te[..., 8]], -1)
    surf_corner = torch.stack([c0, c1, c2], 2)                 # [T,N,3,3]
    refl = surf_corner[..., 0]
    ior = surf_corner[..., 2]
    surf_present = ((refl.abs() > 0).any(0).any(-1)
                    | (ior > 1.0 + 1e-4).any(0).any(-1))       # [N]
    surf_bank, surf_meta, surf_idx = _promote_property_group(
        surf_corner, surf_present, T, device)

    # Material (shading): [pipeline id | 12-slot param block], per primitive so
    # always constant across the corners -> promotes to 1x1. Fed as a degenerate
    # per-corner tensor (all three corners equal) so it shares the promoter.
    mat_vec = torch.cat([tmi.unsqueeze(-1).float(), tm], -1)   # [T,N,13]
    lit = (tmi != _MID_UNLIT).any(0)                           # [N]
    mat_corner = mat_vec.unsqueeze(2).expand(T, N, 3, 13)
    mat_bank, mat_meta, mat_idx = _promote_property_group(
        mat_corner, lit, T, device)

    scene["tx_color_bank"] = col_bank
    scene["tx_color_meta"] = col_meta
    scene["tx_color_idx"] = col_idx
    scene["tx_surf_bank"] = surf_bank
    scene["tx_surf_meta"] = surf_meta
    scene["tx_surf_idx"] = surf_idx
    scene["tx_mat_bank"] = mat_bank
    scene["tx_mat_meta"] = mat_meta
    scene["tx_mat_idx"] = mat_idx
    # Normal-map bank (feature): placeholder / index -1 for every triangle until
    # a Surface carries a normal map (the normal-map feature measures the
    # compiled-in cost; real maps would be promoted here like the colour bank).
    scene["tx_nmap_bank"] = torch.zeros((1, 1, 3), device=device)
    scene["tx_nmap_meta"] = torch.zeros((1, 3), dtype=torch.int32, device=device)
    scene["tx_nmap_idx"] = torch.full((N,), -1, dtype=torch.int32, device=device)
    # Canonical per-triangle corner UVs (shared, constant across frames).
    scene["tx_uv"] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 6).expand(
            1, N, 6).contiguous()


def _merge_scene(primitives):
    """Merge the batch's collections into one set per geometry type --
    triangles, PN patches and bezier circuits, each with a single STBVH
    over all frames -- cached for the batch.
    """
    first = primitives[0]
    cached = getattr(first, "_rt_merged_scene", None)
    if cached is not None:
        return cached

    empty_cache(force_gc=False)
    device = COMPUTING_DEFAULTS.render_device
    pn_patches = [p for p in primitives
                  if isinstance(p, RayTracedPNTrianglePrimitive)]
    triangles = [p for p in primitives
                 if isinstance(p, RayTracedTrianglePrimitive)
                 and not isinstance(p, RayTracedPNTrianglePrimitive)]
    beziers = [p for p in primitives
               if isinstance(p, RayTracedBezierCircuitPrimitive)]
    unknown = [p for p in primitives
               if p not in triangles and p not in pn_patches
               and p not in beziers]
    if unknown:
        raise TypeError(
            "The ray traced renderer can only draw ray traced primitives; "
            f"got {[type(p).__name__ for p in unknown]}. Was "
            "enable_ray_tracing() called before the mobs were created?")
    num_frames = max(p._rt_num_frames for p in primitives)

    # Any PN patch carrying a texture map forces the whole batch onto the
    # general wavefront tracer (the only kernel that samples PN textures); the
    # megakernel's PN path has no UVs. Flags PN color maps too (unlike flat
    # colour maps, which the megakernel can sample).
    has_pn_textures = any(
        getattr(p, "_rt_pn_uvs", None) is not None for p in pn_patches)

    scene = {}
    scene["has_pn_textures"] = has_pn_textures
    # Shared flat texel buffer for *all* texture maps, flat-triangle and
    # PN-patch alike (color / material / normal). Each map is appended once,
    # padded to 5 channels and flattened to [T, W*H, 5]; its placement is a
    # (offset, w, h) triplet recorded in the consuming geometry's metadata
    # (offset -1 = no map). Flat triangles key those triplets by tri_tex_meta;
    # PN patches fold them into pn_extra (no kernel-arg budget left). Assembled
    # into scene["textures"] once both geometry blocks below have appended.
    _texture_tensors = []
    _texel_offset = [0]

    def _append_texture(tex):
        if tex is None:
            return (-1, 0, 0)
        if tex.dim() == 3:  # [W, H, C]
            tex = tex.unsqueeze(0)  # [1, W, H, C]
        w, h, c = tex.shape[-3], tex.shape[-2], tex.shape[-1]
        if c < 5:
            tex = torch.cat(
                (tex, tex.new_zeros((*tex.shape[:-1], 5 - c))), -1)
        # Flatten W and H (dimensions 1 and 2).
        _texture_tensors.append(tex.reshape(tex.shape[0], -1, 5))
        o = _texel_offset[0]
        _texel_offset[0] += w * h
        return (o, w, h)

    scene["tex_has_refractive"] = False
    if triangles:
        # Constant-property promotion: triangles whose colour + material params
        # are constant across their corners (and frames) are rendered from a
        # shared 1x1 colour + material map instead of per-vertex tri_colors /
        # tri_extra rows (see _split_promotable). Detection is per triangle and
        # grouped by value, so a uniform mob is promoted even when it was batched
        # into one primitive alongside differently-coloured mobs. Promoted
        # triangles are ordered LAST (their prims sit past the shrunk arrays,
        # which the guarded kernel reads never index). With promotion inactive
        # every triangle is kept and this reduces byte-identically to the plain
        # per-vertex merge (see _sel: an all-keep selection returns the original
        # tensor, uncopied).
        from algan.rendering.raytracing import settings as _rts
        # The textured wavefront does its own (three-group) constant/per-vertex
        # promotion from the full per-vertex arrays, so the built-in single-map
        # promotion is turned off for it (it would shrink tri_colors/tri_extra
        # out from under the texture builder).
        promote = _constant_promotion_active() and not _rts.WF_TEXTURED
        plain_triangles = [p for p in triangles
                           if getattr(p, "_rt_tri_uvs", None) is None]
        textured_triangles = [p for p in triangles
                              if getattr(p, "_rt_tri_uvs", None) is not None]
        keep_idx, promo_idx, promo_meta = {}, {}, {}
        for p in plain_triangles:
            if promote:
                k, pr, meta = _split_promotable(p, _append_texture, device, scene)
            else:
                Np = p._rt_tri_pos.shape[1]
                k = torch.arange(Np, device=device)
                pr = torch.zeros((0,), dtype=torch.long, device=device)
                meta = torch.zeros((0, 10), dtype=torch.int32, device=device)
            keep_idx[id(p)] = k
            promo_idx[id(p)] = pr
            promo_meta[id(p)] = meta

        def _sel(arr, idx):
            # Index the primitive axis (dim 1) by ``idx``. Only an *identity*
            # selection (every prim, in order) may return the original tensor
            # uncopied -- that keeps the promotion-inactive path byte-identical.
            # ``promo_idx`` covers every prim too (when a whole primitive is
            # promoted) but is a *permutation* (grouped by value, see
            # _split_promotable), so it must still be applied: skipping it would
            # leave the geometry in source order while ``promo_meta`` is in
            # group order, pairing each triangle with another group's maps.
            if idx.numel() == arr.shape[1] and bool(
                    (idx == torch.arange(idx.numel(), device=idx.device)).all()):
                return arr
            return arr.index_select(1, idx.to(arr.device))

        def _geom(name):
            # Global order: kept triangles of the plain primitives, then the
            # whole textured primitives, then the promoted triangles. Empty
            # selections are dropped so the promotion-inactive path passes each
            # original tensor through _cat_collections uncopied.
            keep = [_sel(getattr(p, name), keep_idx[id(p)]) for p in plain_triangles
                    if keep_idx[id(p)].numel()]
            tex = [getattr(p, name) for p in textured_triangles]
            promo = [_sel(getattr(p, name), promo_idx[id(p)]) for p in plain_triangles
                     if promo_idx[id(p)].numel()]
            return keep + tex + promo

        num_colored = sum(int(keep_idx[id(p)].numel()) for p in plain_triangles)
        scene["num_colored_triangles"] = num_colored
        scene["tri_pos"] = _cat_collections(_geom("_rt_tri_pos"), 1, "triangle merge")
        scene["tri_norm"] = _cat_collections(_geom("_rt_tri_norm"), 1, "triangle merge")
        scene["tri_mat_id"] = _cat_collections(_geom("_rt_tri_mat_id"), 1,
                                               "triangle merge")
        scene["tri_mat"] = _cat_mat_blocks(_geom("_rt_tri_mat"), "triangle merge")
        lo = _cat_collections(_geom("_rt_frame_lo"), 1, "triangle merge")
        hi = _cat_collections(_geom("_rt_frame_hi"), 1, "triangle merge")
        opaque = _cat_collections(_geom("_rt_frame_opaque"), 1, "triangle merge")

        # tri_colors / tri_extra span only the kept per-vertex triangles + the
        # textured primitives (a textured primitive may carry only material /
        # normal maps and fall back to per-vertex colour, color-map offset -1).
        # Promoted triangles have no row here; guarded kernel reads keep their
        # (past-the-end) prims from ever indexing these.
        vcolors = ([_sel(p._rt_tri_colors, keep_idx[id(p)]) for p in plain_triangles
                    if keep_idx[id(p)].numel()]
                   + [p._rt_tri_colors for p in textured_triangles])
        vextra = ([_sel(p._rt_tri_extra, keep_idx[id(p)]) for p in plain_triangles
                   if keep_idx[id(p)].numel()]
                  + [p._rt_tri_extra for p in textured_triangles])
        if any(t.shape[1] for t in vcolors):
            scene["tri_colors"] = _cat_collections(vcolors, 1, "triangle merge")
            scene["tri_extra"] = _cat_collections(vextra, 1, "triangle merge")
        else:  # every triangle promoted -> minimal placeholder rows
            scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
            scene["tri_extra"] = torch.zeros((1, 1, 15), device=device)

        # Any promoted group synthesises material maps, so the batch carries
        # material textures -> it is routed to the general wavefront (the guarded
        # kernel), never the megakernel / lean path.
        has_promoted = any(promo_idx[id(p)].numel() for p in plain_triangles)
        scene["has_material_textures"] = bool(has_promoted) or any(
            getattr(p, "_rt_material_texture", None) is not None
            or getattr(p, "_rt_normal_texture", None) is not None
            for p in textured_triangles)

        # UVs + tex-meta cover the [textured ++ promoted] tiers, indexed by
        # ``prim - num_colored_triangles``. Meta layout: cols 0-2 color map, 3-5
        # material map (reflectivity, roughness, index of refraction), 6-8 normal
        # map, 9 bitmask of texture-driven material properties (offset -1 = no
        # map -> per-vertex fallback).
        meta_parts, uvs_parts = [], []
        for p in textured_triangles:
            color_meta = _append_texture(p._rt_texture_map)
            mtex = getattr(p, "_rt_material_texture", None)
            material_meta = _append_texture(mtex)
            normal_meta = _append_texture(getattr(p, "_rt_normal_texture", None))
            flags = int(getattr(p, "_rt_material_flags", 0) or 0)
            if (mtex is not None and (flags & 4)
                    and bool((mtex[..., 2] > 1.0 + 1e-4).any())):
                scene["tex_has_refractive"] = True
            meta_parts.append(torch.tensor(
                [*color_meta, *material_meta, *normal_meta, flags],
                dtype=torch.int32, device=device).view(1, 10).expand(
                    p._rt_tri_pos.shape[1], 10))
            uvs_parts.append(p._rt_tri_uvs)
        for p in plain_triangles:
            n = int(promo_idx[id(p)].numel())
            if n:
                # A 1x1 map ignores UVs (both texels clamp to index 0), so a
                # single-frame zero UV row per promoted triangle suffices.
                meta_parts.append(promo_meta[id(p)])
                uvs_parts.append(torch.zeros((1, n, 6), device=device))
        if meta_parts:
            scene["tri_tex_meta"] = torch.cat(meta_parts, 0).contiguous()
            scene["tri_uvs"] = _cat_collections(uvs_parts, 1, "triangle merge")
        else:
            scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
            scene["tri_tex_meta"] = torch.full((1, 10), -1, dtype=torch.int32,
                                               device=device)

        # Median-split ordering: ~25% faster traversal than Morton at ~0.2s
        # extra build per batch; byte-identical for triangles (the depth-peel
        # is arrangement-invariant). PN/bezier BVHs below stay Morton -- their
        # seam de-dup is discovery-order sensitive (see stbvh._BVH_BUILD).
        scene["tri_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=opaque, builder="split")
        if _rts.WF_MEM_TRIM:
            _build_mem_trim(scene, lo, hi, opaque, num_frames, device)
    else:
        scene["tri_pos"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_extra"] = torch.zeros((1, 1, 15), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
        scene["tri_tex_meta"] = torch.full((1, 10), -1, dtype=torch.int32, device=device)
        scene["num_colored_triangles"] = 0
        scene["has_material_textures"] = False
        scene["tri_mat_id"] = torch.zeros((1, 1), dtype=torch.int32,
                                          device=device)
        scene["tri_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)
    scene["num_triangles"] = scene["tri_pos"].shape[1] if triangles else 0

    # Temporal compression of triangle positions (knot representation). The BVH
    # is already built from the per-frame bounds (independent of tri_pos), so the
    # dense positions can be dropped once compressed -- the knot kernel
    # reconstructs each frame's geometry in-register. Only flat triangles are
    # wired up; reflective/fragment-shaded/shadowed batches keep the dense path.
    scene["tri_tc"] = None
    scene["tri_has_reflective"] = False

    if pn_patches:
        scene["pn_ctrl"] = _cat_collections(
            [p._rt_pn_ctrl for p in pn_patches], 1, "pn merge")
        scene["pn_obb"] = _cat_collections(
            [p._rt_pn_obb for p in pn_patches], 1, "pn merge")
        scene["pn_norm"] = _cat_collections(
            [p._rt_pn_norm for p in pn_patches], 1, "pn merge")
        # Fold per-patch UVs + texture metadata into the (cold, hit-only)
        # pn_extra array: PN has no kernel-arg budget for its own uv/meta/
        # texture arrays (the general wavefront shade kernel is at Taichi's
        # 64-arg cap), so it reads them from widened pn_extra. Layout appended
        # after the existing 15 material cols: cols 15-20 per-corner UV, 21-23
        # color map (offset, w, h) into the shared ``textures`` buffer, 24-26
        # material map, 27-29 normal map, 30 material bitmask. A color-map
        # offset of -1 means fall back to per-vertex pn_colors. The array is
        # widened unconditionally (even with no maps -> all -1) because the
        # default wavefront path shades every PN scene through this kernel, so
        # the texture-sampling code always executes and must find 31 columns.
        # Every patch keeps its slot (no colored/textured reorder -- the PN
        # morton BVH seam de-dup is discovery-order sensitive).
        pn_extra_list = []
        for p in pn_patches:
            extra = p._rt_pn_extra                # [Te, Np, 15]
            Np = extra.shape[1]
            uvs = getattr(p, "_rt_pn_uvs", None)
            if uvs is None:
                uvs = torch.zeros((1, Np, 6), device=device)
            if has_pn_textures:
                color_meta = _append_texture(getattr(p, "_rt_texture_map", None))
                mtex = getattr(p, "_rt_material_texture", None)
                material_meta = _append_texture(mtex)
                normal_meta = _append_texture(
                    getattr(p, "_rt_normal_texture", None))
                flags = int(getattr(p, "_rt_material_flags", 0) or 0)
                if (mtex is not None and (flags & 4)
                        and bool((mtex[..., 2] > 1.0 + 1e-4).any())):
                    scene["tex_has_refractive"] = True
                meta_vals = [*color_meta, *material_meta, *normal_meta, flags]
            else:
                meta_vals = [-1, 0, 0, -1, 0, 0, -1, 0, 0, 0]
            T = max(extra.shape[0], uvs.shape[0])
            # UVs inherit the (CPU) animation device from the per-mob build,
            # while extra/meta are on the render device -- unify before cat.
            extra_e = _expand_frames(extra, T).to(device)
            uvs_e = _expand_frames(uvs, T).to(device)
            meta_e = torch.tensor(
                meta_vals, dtype=torch.float32, device=device
            ).view(1, 1, 10).expand(T, Np, 10)
            pn_extra_list.append(torch.cat([extra_e, uvs_e, meta_e], -1))
        scene["pn_extra"] = _cat_collections(pn_extra_list, 1, "pn merge")
        scene["pn_colors"] = _cat_collections(
            [p._rt_pn_colors for p in pn_patches], 1, "pn merge")
        scene["pn_mat_id"] = _cat_collections(
            [p._rt_pn_mat_id for p in pn_patches], 1, "pn merge")
        scene["pn_mat"] = _cat_mat_blocks(
            [p._rt_pn_mat for p in pn_patches], "pn merge")
        lo = _cat_collections([p._rt_frame_lo for p in pn_patches], 1,
                              "pn merge")
        hi = _cat_collections([p._rt_frame_hi for p in pn_patches], 1,
                              "pn merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in pn_patches],
                                  1, "pn merge")
        scene["pn_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedPNTrianglePrimitive.stbvh_tightness,
            opaque=opaque)
    else:
        scene["pn_ctrl"] = torch.zeros((1, 1, 18), device=device)
        scene["pn_obb"] = torch.zeros((1, 1, 12), device=device)
        scene["pn_norm"] = torch.zeros((1, 1, 9), device=device)
        # 31 cols (15 material + 6 UV + 10 tex-meta) to match the real path, so
        # the wavefront's PN texture reads never run off the stub (see above).
        scene["pn_extra"] = torch.zeros((1, 1, 31), device=device)
        scene["pn_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["pn_mat_id"] = torch.zeros((1, 1), dtype=torch.int32,
                                         device=device)
        scene["pn_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["pn_bvh"] = _empty_scene_part(device)
    scene["num_pn"] = scene["pn_ctrl"].shape[1] if pn_patches else 0

    # Assemble the shared texel buffer now that both the flat-triangle and PN
    # blocks above have appended their maps (offsets recorded in tri_tex_meta /
    # pn_extra respectively).
    if _texture_tensors:
        scene["textures"] = _cat_collections(
            _texture_tensors, 1, "texture merge")
    else:
        scene["textures"] = torch.zeros((1, 1, 5), device=device)
    scene["has_pn_textures"] = has_pn_textures

    # Refraction is active iff some triangle/PN surface carries a meaningful
    # index of refraction (extra columns 6-8, per-corner; 0/1 = no bending).
    # Used to gate the wavefront's refraction template and to route refractive
    # batches to the general wavefront (the only path that refracts).
    def _extra_has_refractive(extra):
        return bool((extra[..., 6:9] > 1.0 + 1e-4).any())
    scene["has_refractive"] = (_extra_has_refractive(scene["tri_extra"])
                               or _extra_has_refractive(scene["pn_extra"])
                               or bool(scene.get("tex_has_refractive")))

    if beziers:
        scene["circuit_meta"] = _cat_collections(
            [p._rt_circuit_meta for p in beziers], 1, "bezier merge")
        scene["circuit_border_colors"] = _cat_collections(
            [p._rt_circuit_border_colors for p in beziers], 1, "bezier merge")
        max_points = max(p._rt_circuit_colors.shape[2] for p in beziers)
        padded = []
        for p in beziers:
            c = p._rt_circuit_colors
            if c.shape[2] < max_points:
                pad = torch.zeros((c.shape[0], c.shape[1],
                                   max_points - c.shape[2], c.shape[3]),
                                  device=c.device)
                c = torch.cat((c, pad), 2)
            padded.append(c)
        scene["circuit_colors"] = _cat_collections(padded, 1, "bezier merge")
        scene["edges_2d"] = _cat_collections(
            [p._rt_edges for p in beziers], 1, "bezier merge")
        offsets, shift = [torch.zeros((1,), dtype=torch.int32, device=device)], 0
        for p in beziers:
            offsets.append(p._rt_edge_offsets[1:].long() + shift)
            shift = shift + p._rt_edges.shape[1]
        scene["edge_offsets"] = torch.cat(
            [o.to(torch.int32) for o in offsets]).contiguous()
        lo = _cat_collections([p._rt_frame_lo for p in beziers], 1,
                              "bezier merge")
        hi = _cat_collections([p._rt_frame_hi for p in beziers], 1,
                              "bezier merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in beziers], 1,
                                  "bezier merge")
        scene["bez_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque)
        scene["num_circuits"] = scene["circuit_meta"].shape[1]
    else:
        scene["circuit_meta"] = torch.zeros((1, 1, 21), device=device)
        scene["circuit_colors"] = torch.zeros((1, 1, 1, 5), device=device)
        scene["circuit_border_colors"] = torch.zeros((1, 1, 5), device=device)
        scene["edges_2d"] = torch.zeros((1, 1, 5), device=device)
        scene["edge_offsets"] = torch.zeros((2,), dtype=torch.int32,
                                            device=device)
        scene["bez_bvh"] = _empty_scene_part(device)
        scene["num_circuits"] = 0

    scene["num_frames"] = num_frames

    # Experimental texture-lookup shading (Surface / flat-triangle scenes only:
    # no PN patches, no bezier circuits). Builds the three per-triangle texture
    # banks + indexes the textured wavefront kernel consumes.
    scene["textured_active"] = False
    from algan.rendering.raytracing import settings as _rts
    if (_rts.WF_TEXTURED and scene["num_triangles"] > 0
            and scene["num_pn"] == 0 and scene["num_circuits"] == 0):
        _build_textured_scene(scene, num_frames, device)
        scene["textured_active"] = True

    # The merged tensors replace the per-collection ones; release the
    # originals so peak GPU memory stays close to one copy of the scene.
    for p in triangles:
        p._rt_tri_pos = p._rt_tri_norm = None
        p._rt_tri_extra = p._rt_tri_colors = None
        p._rt_tri_mat_id = p._rt_tri_mat = None
        p._rt_tri_uvs = p._rt_texture_map = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in pn_patches:
        p._rt_pn_ctrl = p._rt_pn_norm = None
        p._rt_pn_obb = None
        p._rt_pn_extra = p._rt_pn_colors = None
        p._rt_pn_mat_id = p._rt_pn_mat = None
        p._rt_pn_uvs = p._rt_texture_map = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None

    empty_cache(force_gc=False)
    first._rt_merged_scene = scene
    return scene


def _pack_lights(light_sources, num_frames, device):
    """Per-frame packed light rows for the deterministic tracer's fragment
    lighting: positions ``[T, L, 3]`` and color rows ``[T, L, C]``.

    ``C == 3`` (the legacy compact packing: RGB radiance only) whenever every
    light is a plain point light -- keeping such scenes on the kernels'
    original point-light arithmetic. Any *extended* light (a non-point type,
    or falloff / soft-shadow parameters; see :mod:`algan.rendering.lights`)
    widens every row to ``C == 16``::

        0:3  RGB radiance (intensity premultiplied)   9  cos outer (spot)
        3    light type id                            10 cos inner (spot)
        4    decay exponent                           11 shadow softness
        5    range (0 = infinite)                     12:15 ground RGB / SH
        6:9  direction                                15 spare

    Area lights arrive pre-expanded into K emitter sample rows (see
    ``Scene._materialize_render_state``), each occupying its own light slot.
    """
    any_ext = any(getattr(light, "_render_aux", None) is not None
                  for light in (light_sources or ()))
    if not any_ext:
        positions, colors = [], []
        for light in light_sources or ():
            positions.append(_expand_frames(
                _flat_frames(light.origin, (3,)), num_frames))
            col = light.light_color.reshape(light.light_color.shape[0], -1)
            colors.append(_expand_frames(col[:, :3].float(), num_frames))
        if not positions:
            return (torch.zeros((1, 1, 3), device=device),
                    torch.zeros((1, 1, 3), device=device), 0)
        light_pos = torch.stack(positions, 1).to(device).contiguous()
        light_col = torch.stack(colors, 1).to(device).contiguous()
        return light_pos, light_col, light_pos.shape[1]

    positions, rows = [], []
    for light in light_sources or ():
        pos = light.origin                         # [T, K, 3]
        col = light.light_color                    # [T, K, >=3]
        aux = getattr(light, "_render_aux", None)  # [T, K, 13] or None
        num_samples = pos.shape[-2]
        pos = pos.reshape(pos.shape[0], num_samples, -1)[..., :3].float()
        col = col.reshape(col.shape[0], col.shape[-2], -1)[..., :3].float()
        for k in range(num_samples):
            positions.append(_expand_frames(pos[:, k], num_frames))
            c = _expand_frames(col[:, min(k, col.shape[1] - 1)], num_frames)
            if aux is None:
                a = torch.zeros((c.shape[0], 13), dtype=torch.float32)
            else:
                a = _expand_frames(aux[:, k].float(), num_frames)
            rows.append(torch.cat((c, a), -1))
    light_pos = torch.stack(positions, 1).to(device).contiguous()
    light_col = torch.stack(rows, 1).to(device).contiguous()
    return light_pos, light_col, light_pos.shape[1]


def _prefill_background(out, background_color, frame_offset, device):
    """Fill the output buffer with the background. Solid colors arrive as a
    float [channels] tensor in [0, 1]; animated/image backgrounds arrive as a
    uint8 row tensor [1 + frames * pixels, channels] (leading padding row).
    """
    num_frames, num_pixels, C_out = out.shape
    bg = background_color.to(device)
    if bg.dim() <= 1 or bg.shape[0] == 1:  # solid color (in [0, 1] floats)
        vals = (bg.float().flatten()[:5] * 255).round_().clamp_(0, 255)
        k = min(vals.shape[0], C_out)
        out[..., :k] = vals[:k].to(out.dtype)
        if C_out > k:
            # Alpha (and any missing channel) defaults to the background's
            # last channel, matching opaque-by-default behavior.
            out[..., k:] = vals[-1].to(out.dtype)
    else:
        rows = bg.reshape(-1, bg.shape[-1])[1:]
        rows = rows[frame_offset * num_pixels:
                    (frame_offset + num_frames) * num_pixels]
        rows = rows.view(num_frames, num_pixels, -1)
        k = min(rows.shape[-1], C_out)
        out[..., :k] = rows[..., :k].to(out.dtype)
        if C_out > k:
            out[..., k:] = rows[..., -1:].to(out.dtype)


def _downsample_background(background_color, aa, num_frames, screen_height,
                           screen_width):
    """Average a super-sampled animated/image background down to the output
    resolution (box filter, matching ``post_process_frames``), so the in-place
    anti-aliased renderer -- which samples the background once per output pixel
    -- gets a background at the right resolution.

    Solid colors (resolution-free) and backgrounds that are not super-sampled
    (row count not ``num_frames * (screen_height*aa) * (screen_width*aa)``) are
    returned unchanged.
    """
    bg = background_color
    if not torch.is_tensor(bg) or bg.dim() <= 1 or bg.shape[0] == 1:
        return bg  # solid color
    C = bg.shape[-1]
    body = bg.reshape(-1, C)[1:]  # drop the leading padding row
    h_aa, w_aa = screen_height * aa, screen_width * aa
    if body.shape[0] != num_frames * h_aa * w_aa:
        return bg  # not a super-sampled image background; leave as-is
    img = body.view(num_frames, h_aa, w_aa, C).float().permute(0, 3, 1, 2)
    ds = F.avg_pool2d(img, aa).permute(0, 2, 3, 1).reshape(-1, C)
    ds = (ds + 0.5).clamp_(0, 255).to(bg.dtype)
    return torch.cat((ds[:1], ds), 0)