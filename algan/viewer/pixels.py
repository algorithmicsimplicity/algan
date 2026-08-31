"""Turn a captured coverage record into the fragment list behind one pixel.

The renderer hands :mod:`algan.rendering.fragment_capture` a chunk's whole
per-pixel record. This module reads one pixel out of it and says, nearest first,
what is behind that pixel: how far away each surface is, which mesh and which Mob
it belongs to, how much of the pixel it claims, and what colour it is.

On colour, and what the numbers here do and do not mean:

* A fragment's **albedo** is the surface's own colour at the hit point, before
  any lighting. It is reproduced here from the same arrays the shading kernel
  fetches from -- per-vertex colours interpolated at the fragment's barycentrics,
  or the triangle's colour map sampled bilinearly -- so for those two cases the
  RGB matches what the kernel fetched, exactly.
* The **opacity** shown against a fragment is the owning Mob's authored opacity
  at that time, read off the timeline. The fifth lane the colour fetch returns is
  reported separately as ``material_alpha``: it is faithfully what the kernel
  reads, but the merge encodes it, so it is not the number a user authored.
* It is **not** the shaded result. The renderer keeps no per-fragment shaded
  colour, and adding one would mean editing a Taichi kernel. So a lit, glossy,
  refractive or shadowed surface contributes something other than its albedo to
  the finished pixel, and the two will differ.
* Colours here are **linear-light**, the space the renderer works in. The
  composited pixel a viewer reads off the image is display-referred sRGB. Both
  are reported, labelled, rather than one being quietly converted into the other.

Where a fragment's colour cannot be reproduced faithfully -- a Bezier circuit,
whose colour comes from a per-circuit grid, or a colour map built as interpolated
endpoints -- the fragment says so rather than reporting a number that looks
right and is not.

One gap worth knowing about: **only triangles name their Mob.** A mob that
builds triangles (``Cube``, ``Polyhedron``, ``Surface``, ``Model3D``) stamps a
mesh key on its primitive, and the merge now records those keys, so a triangle
fragment resolves back to the Mob that authored it. A Bezier circuit -- which is
what ``Square``, ``Circle`` and ``Text`` are made of -- carries no such key, so
its fragments report their circuit index and no Mob. Closing that gap means
stamping keys in the 2-D mob builders as well.
"""

from __future__ import annotations

import struct

import torch

#: Flag bits the compaction sets in a sheet's mask word, above the sample bits.
#: Mirrors ``raster_taichi``'s ``_AA_*_BIT`` definitions.
_BACKFACE_BIT = 1 << 16
_SLIVER_BIT = 1 << 17
_ONE_MESH_BIT = 1 << 18
_MAT_OPAQUE_BIT = 1 << 19

#: A Bezier circuit's fragment ref is negative and carries its border weight in
#: the low bits. Mirrors ``raster_taichi._pack_bez_ref``.
_BEZ_BORDER_LEVELS = 255
_BEZ_BORDER_BITS = 8


def decode_circuit_ref(ref):
    """``(circuit index, border fraction)`` from a negative fragment ref."""
    code = -int(ref) - 1
    return code >> _BEZ_BORDER_BITS, (code & _BEZ_BORDER_LEVELS) / _BEZ_BORDER_LEVELS


def depth_of_key(key) -> float:
    """The exact hit distance packed into the low 32 bits of a fragment key."""
    return struct.unpack("<f", struct.pack("<I", int(key) & 0xFFFFFFFF))[0]


def _srgb(value: float) -> float:
    """Encode one linear-light channel the way the frame's byte write does."""
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (max(value, 0.0) ** (1.0 / 2.4)) - 0.055


def _row(tensor, frame):
    """Index a ``[T, ...]`` merged array at a frame, the way the kernels do."""
    return int(frame) % int(tensor.shape[0])


class PixelRecord:
    """One render chunk's coverage, indexed by pixel.

    Wraps the dict :func:`algan.rendering.fragment_capture.capture` produced and
    resolves the two things a caller cannot work out for itself: where a pixel
    lives in the flat covered-cell list, and which Mob a surface id belongs to.
    """

    def __init__(self, capture, mob_by_id=None):
        self._c = capture
        self.width = int(capture["width"])
        self.height = int(capture["height"])
        self.time_start = int(capture["time_start"])
        self._mob_by_id = mob_by_id or {}
        self._surface_keys = self._build_surface_keys()
        covered = capture["covered_idx"]
        # ``covered_idx`` is sorted, so a pixel's slot is a binary search rather
        # than a scan of every covered pixel in the chunk.
        self._covered = covered
        self._pixels_per_frame = self.width * self.height

    def _build_surface_keys(self):
        """Global surface id -> the mesh key the Mob stamped on its primitive."""
        keys = {}
        for base, count, table in self._c.get("tri_obj_sources") or ():
            if not table:
                continue
            for local, key in enumerate(table):
                if local < count and key is not None:
                    keys[base + local] = key
        return keys

    @property
    def frames(self):
        """The frame indices this chunk covers, as the capture encodes them."""
        if self._covered.numel() == 0:
            return []
        rel = torch.div(
            self._covered, self._pixels_per_frame, rounding_mode="floor"
        ).unique()
        return [self.time_start + int(r) for r in rel]

    def _cell(self, x, y, frame_rel):
        """The flat covered-cell index for image pixel ``(x, y)``.

        The kernel's own decode is ``f_rel*W*H + py*W + px`` with ``py`` counted
        from the bottom, while an image's row ``y`` is counted from the top --
        so the two differ by ``height - 1 - y``. Pinned by
        ``test_pixel_rows_are_not_flipped`` in
        ``tests/unit_tests/test_viewer_fragments.py`` rather than guessed at
        run time.
        """
        py = self.height - 1 - int(y)
        return frame_rel * self._pixels_per_frame + py * self.width + int(x)

    def _slot(self, cell):
        """Index of ``cell`` in the covered list, or ``None`` if uncovered."""
        pos = int(torch.searchsorted(self._covered, torch.tensor(cell)))
        if pos < self._covered.numel() and int(self._covered[pos]) == cell:
            return pos
        # The covered list is built from a sorted stream, so the search above is
        # the answer. The scan is here because the failure it guards against is
        # silent: an unsorted list would make a covered pixel report an empty
        # fragment list, which reads as "nothing is there" rather than as a bug.
        # On a sorted list it never runs.
        match = (self._covered == cell).nonzero()
        return None if match.numel() == 0 else int(match[0])

    def fragments(self, x, y, frame_rel=0):
        """The depth-sorted fragment list behind image pixel ``(x, y)``.

        Nearest first, as the compaction already ordered them. Returns an empty
        list for a pixel nothing covers -- background shows through there, which
        is a real answer rather than a missing one.
        """
        if not (0 <= int(x) < self.width and 0 <= int(y) < self.height):
            return []
        slot = self._slot(self._cell(x, y, frame_rel))
        if slot is None:
            return []
        c = self._c
        start = int(c["sheet_offsets"][slot])
        end = int(c["sheet_offsets"][slot + 1])
        frame = self.time_start + frame_rel
        out = []
        for i in range(start, end):
            ref = int(c["sheet_ref"][i])
            mask = int(c["sheet_mask"][i])
            fragment = {
                "index": i - start,
                "depth": depth_of_key(c["sheet_key"][i]),
                "primitive": ref,
                "kind": "triangle" if ref >= 0 else "circuit",
                "weight": float(c["sheet_weight"][i]),
                "cap": float(c["sheet_cap"][i]),
                "backface": bool(mask & _BACKFACE_BIT),
                "sliver": bool(mask & _SLIVER_BIT),
                "one_mesh": bool(mask & _ONE_MESH_BIT),
                "opaque": bool(mask & _MAT_OPAQUE_BIT),
            }
            if ref < 0:
                circuit, border = decode_circuit_ref(ref)
                fragment["circuit"] = circuit
                fragment["border"] = border
            fragment["mesh_id"] = self._mesh_id(ref, frame)
            fragment.update(self._identify(fragment["mesh_id"]))
            fragment.update(
                self._albedo(
                    ref, float(c["sheet_ab"][i, 0]), float(c["sheet_ab"][i, 1]), frame
                )
            )
            out.append(fragment)
        return out

    def raw_fragment_count(self, x, y, frame_rel=0):
        """How many raw hits the sheets at this pixel were compacted from."""
        slot = self._slot(self._cell(x, y, frame_rel))
        if slot is None:
            return 0
        runs = self._c["run_offsets"]
        return int(runs[slot + 1]) - int(runs[slot])

    def _mesh_id(self, ref, frame):
        """The surface id behind a fragment, in the merge's global numbering.

        Triangles only. A Bezier circuit is its own surface and carries no entry
        in ``tri_obj``, so it reports no mesh id and gives its circuit index
        instead -- which is what "if applicable" means here.
        """
        tri_obj = self._c.get("tri_obj")
        if ref < 0 or tri_obj is None or ref >= tri_obj.shape[1]:
            return None
        return int(tri_obj[_row(tri_obj, frame), ref])

    def _identify(self, mesh_id):
        """Name the Mob a surface belongs to, when the merge recorded one."""
        key = self._surface_keys.get(mesh_id)
        if key is None:
            return {"mesh_key": None, "mob_id": None, "mob": None}
        # Every mob stamps ``(kind, mob.id)``; the kind says which builder made
        # the primitive and is worth showing, the id is what finds the Mob.
        mob_id = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        mob = self._mob_by_id.get(mob_id)
        return {
            "mesh_key": str(key),
            "mob_id": mob_id,
            "mob": None if mob is None else mob_label(mob),
        }

    def _albedo(self, ref, a, b, frame):
        """The surface's own colour at the hit point, before any lighting."""
        colour = None
        source = None
        if ref >= 0:
            colour, source = self._triangle_albedo(ref, a, b, frame)
        if colour is None:
            return {
                "rgb": None,
                "rgb_srgb": None,
                "glow": None,
                "material_alpha": None,
                "albedo_source": source,
            }
        r, g, bl, glow, alpha = colour
        return {
            "rgb": [r, g, bl],
            "rgb_srgb": [_srgb(r), _srgb(g), _srgb(bl)],
            "glow": glow,
            # The fifth lane of the fetch, reported as what it is: the value the
            # shading kernel's own colour fetch returns here. It is NOT the
            # mob's authored opacity -- the merge encodes that lane differently
            # -- so the opacity a user recognises is filled in beside it from
            # the owning Mob's timeline, and this stays for renderer debugging.
            "material_alpha": alpha,
            "albedo_source": source,
        }

    def _triangle_albedo(self, prim, a, b, frame):
        """Reproduce the kernel's colour fetch for one triangle hit.

        Mirrors ``_flat_triangle_color``: a triangle below
        ``num_colored_triangles``, or one whose colour map is absent, takes its
        per-vertex colours; everything else samples the map.
        """
        colors = self._c.get("tri_colors")
        meta = self._c.get("tri_tex_meta")
        num_colored = int(self._c.get("num_colored_triangles", 0))
        w0, w1, w2 = 1.0 - a - b, a, b
        mapped = prim - num_colored
        if (
            prim < num_colored
            or meta is None
            or mapped >= meta.shape[0]
            or int(meta[max(mapped, 0), 0]) < 0
        ):
            if colors is None or prim >= colors.shape[1]:
                return None, "unavailable"
            row = colors[_row(colors, frame), prim]
            value = w0 * row[0] + w1 * row[1] + w2 * row[2]
            return [float(v) for v in value], "vertex"
        return self._sample_colour_map(mapped, w0, w1, w2, frame)

    def _sample_colour_map(self, mapped, w0, w1, w2, frame):
        """Bilinear colour-map sample, mirroring ``_sample_texture``."""
        meta = self._c["tri_tex_meta"]
        textures = self._c.get("textures")
        uvs = self._c.get("tri_uvs")
        if textures is None or uvs is None:
            return None, "unavailable"
        row = meta[mapped]
        if int(row[16]) >= 0:
            # Endpoint-interpolated maps blend two authored images in the
            # texture bank. Reproducing that faithfully is more machinery than
            # an inspector earns; say so instead of reporting a wrong colour.
            return None, "endpoint_map"
        offset, width, height = int(row[0]), int(row[1]), int(row[2])
        if width <= 0 or height <= 0:
            return None, "unavailable"
        lut_base = int(row[15])
        tmap = max(int(row[10]), 1)
        uv_row = uvs[_row(uvs, frame), mapped]
        u = w0 * float(uv_row[0]) + w1 * float(uv_row[2]) + w2 * float(uv_row[4])
        v = w0 * float(uv_row[1]) + w1 * float(uv_row[3]) + w2 * float(uv_row[5])
        px = min(max(u * (width - 1.0), 0.0), max(width - 1.0, 0.0))
        py = min(max(v * (height - 1.0), 0.0), max(height - 1.0, 0.0))
        x_floor, y_floor = int(px // 1), int(py // 1)
        xr, yr = px - x_floor, py - y_floor
        base_row = offset + (frame % tmap) * width * height
        tc = _row(textures, frame)
        acc = [0.0] * 5
        total = 0.0
        for corner in range(4):
            cx = min(max(x_floor + (corner % 2), 0), width - 1)
            cy = min(max(y_floor + (corner // 2), 0), height - 1)
            weight = (xr if corner % 2 else 1.0 - xr) * (
                yr if corner // 2 else 1.0 - yr
            )
            texel = self._texel(tc, base_row, lut_base, cx * height + cy, textures)
            for k in range(5):
                acc[k] += weight * texel[k]
            total += weight
        total = max(total, 1e-6)
        acc = [v / total for v in acc]
        # In-sampler opacity: the mob's animated opacity rides the bank as its
        # own tiny region rather than being premultiplied into the map.
        op_off = int(row[13])
        if op_off >= 0:
            op_len = max(int(row[14]), 1)
            acc[4] *= float(textures[tc, op_off + (frame % op_len), 0])
        return acc, "texture"

    def _texel(self, tc, base_row, lut_base, texel_idx, textures):
        """One texel, in either of the bank's two layouts.

        Mirrors ``_color_map_texel``: a plain f32 map stores five channels per
        row, while a u8-packed map bit-packs RGBA into one lane and decodes it
        through the map's own 256-entry lookup table.
        """
        num_points = int(textures.shape[1])
        if lut_base < 0:
            idx = min(max(base_row + texel_idx, 0), num_points - 1)
            return [float(textures[tc, idx, k]) for k in range(5)]
        lane = base_row * 5 + texel_idx
        row = min(max(lane // 5, 0), num_points - 1)
        channel = lane - 5 * (lane // 5)
        bits = struct.unpack(
            "<I", struct.pack("<f", float(textures[tc, row, channel]))
        )[0]
        bytes_ = [
            bits & 0xFF,
            (bits >> 8) & 0xFF,
            (bits >> 16) & 0xFF,
            (bits >> 24) & 0xFF,
        ]
        return [
            float(textures[tc, lut_base + bytes_[0], 0]),
            float(textures[tc, lut_base + bytes_[1], 0]),
            float(textures[tc, lut_base + bytes_[2], 0]),
            0.0,
            float(textures[tc, lut_base + bytes_[3], 1]),
        ]


def mob_label(mob) -> str:
    """How a Mob is named in the viewer: its class, its id, its name if set."""
    name = getattr(mob, "name", None)
    base = f"{type(mob).__name__} #{getattr(mob, 'id', '?')}"
    return base if not name or name == "_" else f"{name} ({base})"
