"""Per-pixel fragment capture: the renderer's side of the GUI viewer's inspector.

The deterministic route already builds, on the host and before any kernel runs, a
per-pixel depth-sorted record of every surface covering that pixel: the sheet stream
that ``sheets.compact_sheets`` produces and ``prepare_sparse_raster_coverage``
returns. That record answers "what is behind this pixel, and in what order" exactly,
and it is discarded at the end of the render chunk that made it.

This module lets a caller ask for one copy of it. It is armed around a render,
the tracer hands it the coverage dict it just built, and it copies the arrays it
needs to the host before the arena reclaims them.

It follows the shape of the ``ALGAN_AA_DUMP`` diagnostic next door: the tracer's
call site is one ``is_armed()`` test, which is a module-global read and costs
nothing on a render nobody is inspecting, and no array here is arena-allocated, so
an armed capture cannot perturb the memory model's view of a chunk.

Deliberately free of Algan imports beyond torch, so the render path can import it
at module scope without a cycle.
"""

from __future__ import annotations

import threading

import torch

#: The pending capture request, or ``None``. Guarded by :data:`_LOCK` for arming
#: and disarming; the tracer's ``is_armed`` read is deliberately unlocked, being a
#: single attribute load on a hot path whose worst case is one wasted copy.
_PENDING: dict | None = None
_LOCK = threading.Lock()

#: Refuse a capture whose sheet stream is larger than this. A capture is a debug
#: convenience, not a reason to double a big render's host memory. At the viewer's
#: default resolution a frame's stream is a few hundred thousand sheets at most.
MAX_SHEETS = 1 << 23

#: Refuse to copy a texture bank larger than this many elements (~64 MB of
#: float32). Past it, mapped fragments report no colour instead.
MAX_TEXTURE_ELEMENTS = 1 << 24


def _bank(textures):
    """The texture bank, if copying it is proportionate."""
    if textures is None or textures.numel() > MAX_TEXTURE_ELEMENTS:
        return None
    return textures


def is_armed() -> bool:
    """Whether a caller is waiting for a coverage capture."""
    return _PENDING is not None


def arm() -> None:
    """Ask the next covered render chunk to hand over its coverage record."""
    global _PENDING
    with _LOCK:
        _PENDING = {"captures": []}


def disarm() -> list[dict]:
    """Stop capturing and return what was captured, one entry per render chunk."""
    global _PENDING
    with _LOCK:
        pending = _PENDING
        _PENDING = None
    return [] if pending is None else pending["captures"]


def capture(coverage, merged, time_start, width, height) -> None:
    """Copy one chunk's per-pixel sheet record to the host.

    Called by the tracer immediately after ``prepare_sparse_raster_coverage``, from
    inside the ``memory.temp`` block that owns the arrays -- they are arena
    tensors and stop being valid when that block exits, so everything kept here is
    copied rather than referenced.

    A chunk with no coverage, or one whose route produced no sheet stream, is
    skipped: the caller reads an empty capture list and reports the pixel as
    having no fragment data rather than being told a wrong answer.
    """
    pending = _PENDING
    if pending is None or coverage is None or not coverage.get("sheets"):
        return
    num_sheets = int(coverage.get("num_sheets", 0))
    num_covered = int(coverage.get("num_covered", 0))
    if num_covered <= 0 or num_sheets <= 0 or num_sheets > MAX_SHEETS:
        return

    def host(value, count=None):
        if value is None:
            return None
        tensor = value if count is None else value[:count]
        return tensor.detach().to("cpu", copy=True)

    entry = {
        "time_start": int(time_start),
        "width": int(width),
        "height": int(height),
        "num_covered": num_covered,
        "num_sheets": num_sheets,
        "covered_idx": host(coverage["covered_idx"], num_covered).to(torch.int64),
        "sheet_offsets": host(coverage["sheet_offsets"], num_covered + 1).to(
            torch.int64
        ),
        "sheet_key": host(coverage["sheet_key"], num_sheets),
        "sheet_ref": host(coverage["sheet_ref"], num_sheets),
        "sheet_ab": host(coverage["sheet_ab"], num_sheets),
        # NOTE: the pipeline overwrites these two with the COMPOSITING weights
        # (``sheet_wgt`` / ``sheet_wmsk``) before returning, so they are the
        # weights the resolve consumes, not the sheets' raw recorded areas. The
        # viewer labels them as weights for that reason.
        "sheet_weight": host(coverage["sheet_cov"], num_sheets),
        "sheet_mask": host(coverage["sheet_msk"], num_sheets),
        "sheet_cap": host(coverage["sheet_cap"], num_sheets),
        # The raw fragment stream the sheets were compacted from, kept so the
        # inspector can show what a sheet was made of.
        "num_fragments": int(coverage.get("num_fragments", 0)),
        "run_offsets": host(coverage["run_offsets"], num_covered + 1).to(torch.int64),
        "frag_key": host(coverage["frag_key"], int(coverage.get("num_fragments", 0))),
        "frag_ref": host(coverage["frag_ref"], int(coverage.get("num_fragments", 0))),
        "frag_ab": host(coverage["frag_ab"], int(coverage.get("num_fragments", 0))),
        # Everything needed to say WHAT a fragment is: the per-triangle surface
        # id, the vertex colours its albedo interpolates from, and the texture
        # table that says whether a triangle's colour comes from a map instead.
        "tri_obj": host(merged.get("tri_obj")),
        "tri_colors": host(merged.get("tri_colors")),
        "tri_tex_meta": host(merged.get("tri_tex_meta")),
        "tri_uvs": host(merged.get("tri_uvs")),
        # The texture bank, which is the one array here that can be large: a
        # scene with real image maps holds every texel in it. Copied only while
        # it stays small enough that an inspector is not doubling the render's
        # host memory; beyond that the colour of a mapped fragment is reported
        # as unavailable rather than paid for.
        "textures": host(_bank(merged.get("textures"))),
        "num_colored_triangles": int(merged.get("num_colored_triangles", 0)),
        # Plain Python, built by the scene merge: which primitive each block of
        # surface ids came from, and the mesh key each mob stamped on it.
        "tri_obj_sources": merged.get("tri_obj_sources"),
    }
    pending["captures"].append(entry)
