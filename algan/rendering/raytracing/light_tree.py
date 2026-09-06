"""The path tracer's light tree: Conty Estevez & Kulla 2018, host side.

Next-event estimation has to pick *one* emitter per shadow ray. The flat
power-weighted CDF this replaces (``path_tracer._build_nee_tables``, still
the ``pt_light_tree = False`` arm) picks purely by power, so it ignores both
distance and orientation: in a many-light scene most samples land on an
emitter that is far away or facing the wrong way, and the shadow ray is
spent for nothing. "Too many lights" is one of the three reasons the path
tracer exists (``DESIGN_path_tracer_roadmap.md``, top), so the selection
structure is a light **tree** -- the one production renderers build
(PBRT-v4's ``BVHLightSampler``, Cycles).

What is in the tree
-------------------

Every *finite* entry of the next-event table: point, spot and rect-area-cell
light rows, and every emissive lit triangle. Each carries

* a **power** (the same number the flat CDF weights by, so the two arms
  agree on what an emitter is worth),
* a world-space **bounding box** -- a point for a delta row, the cell
  rectangle for an area row, the three vertices for a triangle, and
* an **orientation cone** ``(axis, theta_o, theta_e)``: how far the emitting
  normals inside the node spread from ``axis`` (``theta_o``) and how far off
  its own normal an emitter still emits (``theta_e``). A point light is a
  full cone (``theta_o = pi``), a spot is its outer cone, a one-sided area
  cell or triangle is ``theta_o = 0, theta_e = pi/2``, a two-sided triangle
  ``theta_o = pi/2``.

**Directional rows and the environment entry stay out.** They are infinite
lights: they have no position, so no spatial structure can discriminate
between them, and a bounding box for them is meaningless. They keep a small
power-weighted flat list of their own, selected with a position-independent
probability. Because that list's share is ``P_inf / (P_inf + P_tree)`` and a
member's share inside it is ``power / P_inf``, an infinite entry's *effective*
selection probability is ``power / P_total`` -- exactly what the flat CDF gave
it. Only the split among the finite entries changes.

The build
---------

Top-down, one leaf per entry, splitting on the surface-area-orientation
heuristic (SAOH) of the paper: for each of the three axes, sort the entries
by centroid and score every prefix/suffix split by

    Kr * (E_L * A_L * M(Omega_L) + E_R * A_R * M(Omega_R))

with ``E`` the power sum, ``A`` the box surface area, ``M(Omega)`` the
paper's orientation measure and ``Kr = max_extent / extent[axis]`` the
regularization that makes the three axes comparable. Splitting a node whose
children face opposite ways is what the orientation term buys over a plain
SAH -- a tight box full of back-to-back emitters is a bad node for
*sampling* however good it is for traversal.

Two deliberate approximations, both conservative and both invisible to
correctness (the tree is a sampler; any positive importance is unbiased as
long as **both** MIS ends read the same tree):

* a node's bounding cone is taken about the normalized *mean* of its
  members' axes rather than by the paper's incremental pairwise union. It is
  a valid bound (``theta_o = max_i(angle(mean, axis_i) + theta_o_i)``),
  exact for a leaf, and it vectorizes -- the union does not, and a
  Python-level fold per node is what makes a host-side build expensive.
* the split search scores a prefix's cone from the same mean-axis bound
  evaluated on running sums, so all ``k - 1`` candidate splits of an axis
  are scored in one pass instead of one at a time.

Per frame
---------

``light_pos`` / ``light_col`` / ``tri_pos`` are per-frame tensors and lights
move, so the tree is built per frame of the chunk and indexed the way those
tensors are. Frames whose emitter geometry is byte-identical share one tree
(the common case: a static light rig under moving geometry collapses to a
single row), and a chunk whose distinct-frame count times entry count would
exceed ``PER_FRAME_BUILD_BUDGET`` falls back to **one** tree built over the
union of every frame's bounds and cones -- looser, still unbiased, and
bounded in build time.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

#: Node payload columns of the packed ``[rows, nodes, LT_F_WIDTH]`` tensor.
#: The cone half-angles are packed as their cosine and sine rather than as
#: angles: the kernel's importance evaluates ``cos(theta - theta_o -
#: theta_u)`` through the angle-subtraction identities (PBRT-v4's
#: ``CosSubClamped``), so it needs no inverse trigonometry at all. Measured on
#: a bare 32-light ring at 320x180, warm: the angle form cost ``pt_shade``
#: 844 ms of device time against the flat CDF's 504 ms, and this form
#: 674 ms -- half the descent's overhead was inverse trigonometry.
LT_F_WIDTH = 14
LT_BMIN = 0  # 3 columns: world-space bounds of the subtree ...
LT_BMAX = 3  # ... (a degenerate box for a delta row)
LT_AXIS = 6  # 3 columns: orientation-cone axis
LT_COS_THETA_O = 9  # cos / sin of the normal spread about the axis
LT_SIN_THETA_O = 10
LT_COS_THETA_E = 11  # cos of the emission spread about a normal
LT_POWER = 12  # summed emitted power of the subtree
#: Distance-falloff exponent the importance divides by (``d ** LT_DECAY``).
#: 2 for an emissive triangle -- the area-to-solid-angle Jacobian is
#: inverse-square whatever the author did -- and a light row's own authored
#: ``decay``, which in Algan **defaults to 0**: a `PointLight` does not fade
#: with distance unless asked to. Getting this from the emitter rather than
#: assuming inverse-square is load-bearing, not a nicety: a 1/d^2 importance
#: on decay-0 rows aims the sampler at the near lights while every light
#: contributes equally, and measured 1.34x WORSE mean squared error than the
#: flat CDF on the 32-light ring. A node takes the MINIMUM over its subtree
#: (the least distance discount), which is conservative in the direction
#: that keeps every emitter reachable.
LT_DECAY = 13

#: Link columns of the packed ``[rows, nodes, LT_I_WIDTH]`` int tensor.
#: ``LT_LEFT < 0`` marks a leaf, and then ``LT_RIGHT`` is its entry index.
LT_I_WIDTH = 3
LT_LEFT = 0
LT_RIGHT = 1
LT_PARENT = 2

#: Above ``distinct frames x entries`` the per-frame build collapses to one
#: union tree. The build is ~0.12 ms of host-side numpy per node and a tree
#: has ``2E - 1`` nodes, so this is a wall-clock budget -- about a quarter
#: second of build per render chunk at the ceiling, whatever E is -- and not
#: a correctness one: the union tree samples the same emitters through the
#: same code, just with bounds loose enough to cover every frame.
PER_FRAME_BUILD_BUDGET = 1024

#: Built trees, keyed by a digest of their inputs, so a static light rig is
#: built ONCE per render rather than once per chunk. The build is host-side
#: numpy at ~0.2 ms per node; on a Kaggle T4 the 64-light benchmark scene
#: spent 430 ms of a 2.1 s render rebuilding the same 127-node tree for each
#: of five single-frame chunks. The key is the bytes of every input the
#: build reads, so a hit is a byte-identical tree; the cache is small and
#: process-wide, and is cleared by ``clear_tree_cache`` (tests, and the
#: place a render job's setup could drop it if memory ever mattered -- it
#: does not: a tree is a few kilobytes).
_TREE_CACHE_SIZE = 16
_tree_cache: OrderedDict[bytes, tuple] = OrderedDict()


def clear_tree_cache():
    """Forget every memoized tree (tests)."""
    _tree_cache.clear()


def _tree_key(power, bmin, bmax, axis, theta_o, theta_e, decay):
    h = hashlib.sha1()
    for arr in (power, bmin, bmax, axis, theta_o, theta_e, decay):
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.digest()


_PI = float(np.pi)


def _surface_area(diag):
    """Surface area of an axis-aligned box from its diagonal."""
    return 2.0 * (
        diag[..., 0] * diag[..., 1]
        + diag[..., 1] * diag[..., 2]
        + diag[..., 2] * diag[..., 0]
    )


def _omega_measure(theta_o, theta_e):
    """The paper's orientation measure ``M(Omega)`` of a cone.

    Zero-width cones measure ~0 and a full sphere measures its solid angle,
    so a node that groups emitters facing every way is penalised exactly
    where a plain surface-area heuristic would be indifferent.
    """
    theta_w = np.minimum(theta_o + theta_e, _PI)
    return 2.0 * _PI * (1.0 - np.cos(theta_o)) + 0.5 * _PI * (
        2.0 * theta_w * np.sin(theta_o)
        - np.cos(theta_o - 2.0 * theta_w)
        - 2.0 * theta_o * np.sin(theta_o)
        + np.cos(theta_o)
    )


def _store_cone(node_f, idx, theta_o, theta_e):
    """Write a node's cone as the cosines and sine the kernel reads."""
    node_f[idx, LT_COS_THETA_O] = np.cos(theta_o)
    node_f[idx, LT_SIN_THETA_O] = np.sin(theta_o)
    node_f[idx, LT_COS_THETA_E] = np.cos(theta_e)


def _cone_bound(axis, theta_o):
    """A bounding cone of the ``(axis_i, theta_o_i)`` cones, mean-axis form.

    Exact for one member; conservative for several. Degenerate input (axes
    that cancel) falls back to the full sphere, which is always valid.
    """
    if axis.shape[0] == 1:
        return axis[0].copy(), float(min(theta_o[0], _PI))
    total = axis.sum(0)
    norm = float(np.linalg.norm(total))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0]), _PI
    mean = total / norm
    spread = np.arccos(np.clip(axis @ mean, -1.0, 1.0))
    return mean, float(min(_PI, float((spread + theta_o).max())))


def _running_cone(axis_sum, count, theta_o_max):
    """Cone spread of every prefix, from running sums (the split search).

    ``|sum(axis)| / count`` is 1 when the members agree and falls off as they
    spread, so its arc-cosine is a monotone stand-in for the exact bound.
    """
    mag = np.linalg.norm(axis_sum, axis=-1) / np.maximum(count, 1.0)
    return np.minimum(np.arccos(np.clip(mag, 0.0, 1.0)) + theta_o_max, _PI)


def _split(ids, lo, hi, power, cent, bmin, bmax, axis, theta_o, theta_e):
    """Best SAOH split of ``ids`` as ``(left ids, right ids)``.

    Every candidate of every axis is scored in one pass: the three centroid
    orders and their three reversals are stacked into one ``[6, k]`` index
    array, so the running bounds / power / cone that a prefix and a suffix
    need are six ``accumulate`` calls rather than thirty-odd. The build is
    dominated by host-side call count (the arrays are tiny), and this is
    what keeps a per-frame rebuild affordable.

    ``lo``/``hi`` are the node's own bounds, which the caller has already
    reduced. Falls back to a halving of the id order when no axis has any
    centroid spread (coincident emitters), which keeps the tree binary and
    finite rather than recursing forever on an unsplittable set.
    """
    k = ids.shape[0]
    diag = hi - lo
    max_ext = float(diag.max())
    col = cent[ids]
    spread = col.max(0) - col.min(0)
    if not (spread > 0.0).any():
        half = k // 2
        return ids[:half], ids[half:]
    order = ids[np.argsort(col, axis=0, kind="stable").T]  # [3, k]
    both = np.concatenate((order, order[:, ::-1]), axis=0)  # [6, k]
    cut = k - 1
    box_lo = np.minimum.accumulate(bmin[both], axis=1)[:, :cut]
    box_hi = np.maximum.accumulate(bmax[both], axis=1)[:, :cut]
    pw = np.cumsum(power[both], axis=1)[:, :cut]
    ax = np.cumsum(axis[both], axis=1)[:, :cut]
    to_max = np.maximum.accumulate(theta_o[both], axis=1)[:, :cut]
    te_max = np.maximum.accumulate(theta_e[both], axis=1)[:, :cut]
    cone = _running_cone(ax, np.arange(1.0, k), to_max)
    part = pw * _surface_area(box_hi - box_lo) * _omega_measure(cone, te_max)
    # part[3 + d, k - s - 1] is the suffix left over by a prefix of length s.
    kr = max_ext / np.maximum(diag, 1e-12)
    cost = np.where(
        spread[:, None] > 0.0,
        kr[:, None] * (part[:3] + part[3:][:, ::-1]),
        np.inf,
    )
    flat = int(np.argmin(cost))
    dim, j = divmod(flat, cut)
    sid = order[dim]
    return sid[: j + 1], sid[j + 1 :]


def build_light_tree(power, bmin, bmax, axis, theta_o, theta_e, decay):
    """Build one frame's tree over ``E`` entries.

    Every input is indexed by entry: ``power [E]``, ``bmin/bmax/axis [E, 3]``,
    ``theta_o/theta_e [E]`` in radians, ``decay [E]`` the falloff exponent.
    Returns
    ``(node_f [2E-1, LT_F_WIDTH] float32, node_i [2E-1, LT_I_WIDTH] int32,
    entry_leaf [E] int32)`` -- node 0 is the root, a leaf's ``LT_RIGHT`` is
    its entry index, and ``entry_leaf`` inverts that for the MIS pdf's upward
    walk.

    Deterministic: the traversal order, the stable sorts and the numpy
    reductions are all fixed, so two builds of one input are byte-identical
    -- which is what lets the result be memoized by its inputs' bytes (see
    ``_tree_cache``): the second chunk of a render with a static rig gets the
    first chunk's tree back without a build.
    """
    key = _tree_key(power, bmin, bmax, axis, theta_o, theta_e, decay)
    cached = _tree_cache.get(key)
    if cached is not None:
        _tree_cache.move_to_end(key)
        return tuple(a.copy() for a in cached)
    built = _build_light_tree(power, bmin, bmax, axis, theta_o, theta_e, decay)
    _tree_cache[key] = tuple(a.copy() for a in built)
    while len(_tree_cache) > _TREE_CACHE_SIZE:
        _tree_cache.popitem(last=False)
    return built


def _build_light_tree(power, bmin, bmax, axis, theta_o, theta_e, decay):
    """The build behind :func:`build_light_tree`, uncached."""
    power = np.ascontiguousarray(power, dtype=np.float64)
    bmin = np.ascontiguousarray(bmin, dtype=np.float64)
    bmax = np.ascontiguousarray(bmax, dtype=np.float64)
    axis = np.ascontiguousarray(axis, dtype=np.float64)
    theta_o = np.ascontiguousarray(theta_o, dtype=np.float64)
    theta_e = np.ascontiguousarray(theta_e, dtype=np.float64)
    decay = np.ascontiguousarray(decay, dtype=np.float64)
    entries = int(power.shape[0])
    nodes = max(2 * entries - 1, 1)
    node_f = np.zeros((nodes, LT_F_WIDTH), dtype=np.float64)
    node_i = np.full((nodes, LT_I_WIDTH), -1, dtype=np.int32)
    entry_leaf = np.zeros(max(entries, 1), dtype=np.int32)
    if entries == 0:
        return node_f.astype(np.float32), node_i, entry_leaf
    cent = 0.5 * (bmin + bmax)
    counter = 1
    # Explicit stack: a degenerate split chain is E deep, past Python's
    # recursion limit on a big emissive mesh.
    stack = [(0, np.arange(entries), -1)]
    while stack:
        idx, ids, parent = stack.pop()
        node_i[idx, LT_PARENT] = parent
        if ids.shape[0] == 1:
            # A leaf is its entry: no reductions, which halves the build's
            # host-side numpy calls (half a binary tree's nodes are leaves).
            j = int(ids[0])
            node_f[idx, LT_BMIN : LT_BMIN + 3] = bmin[j]
            node_f[idx, LT_BMAX : LT_BMAX + 3] = bmax[j]
            node_f[idx, LT_AXIS : LT_AXIS + 3] = axis[j]
            _store_cone(node_f, idx, min(float(theta_o[j]), _PI), float(theta_e[j]))
            node_f[idx, LT_POWER] = float(power[j])
            node_f[idx, LT_DECAY] = float(decay[j])
            node_i[idx, LT_RIGHT] = j
            entry_leaf[j] = idx
            continue
        lo = bmin[ids].min(0)
        hi = bmax[ids].max(0)
        node_f[idx, LT_BMIN : LT_BMIN + 3] = lo
        node_f[idx, LT_BMAX : LT_BMAX + 3] = hi
        cone_axis, cone_o = _cone_bound(axis[ids], theta_o[ids])
        node_f[idx, LT_AXIS : LT_AXIS + 3] = cone_axis
        _store_cone(node_f, idx, cone_o, float(theta_e[ids].max()))
        node_f[idx, LT_POWER] = float(power[ids].sum())
        node_f[idx, LT_DECAY] = float(decay[ids].min())
        left, right = _split(
            ids, lo, hi, power, cent, bmin, bmax, axis, theta_o, theta_e
        )
        node_i[idx, LT_LEFT] = counter
        node_i[idx, LT_RIGHT] = counter + 1
        stack.append((counter, left, idx))
        stack.append((counter + 1, right, idx))
        counter += 2
    return node_f.astype(np.float32), node_i, entry_leaf


def union_over_rows(bmin, bmax, axis, theta_o, theta_e):
    """Collapse per-frame emitter geometry ``[R, E, ...]`` to one row.

    The union box and a cone that bounds every frame's cone: the tree built
    from it is valid at every frame of the chunk, which is what makes the
    over-budget fallback correct rather than merely cheap.
    """
    ub_min = bmin.min(0)
    ub_max = bmax.max(0)
    total = axis.sum(0)
    norm = np.linalg.norm(total, axis=-1)
    mean = np.where(
        norm[:, None] > 1e-9,
        total / np.maximum(norm, 1e-9)[:, None],
        np.array([0.0, 0.0, 1.0]),
    )
    spread = np.arccos(np.clip(np.einsum("rec,ec->re", axis, mean), -1.0, 1.0))
    u_theta_o = np.minimum((spread + theta_o).max(0), _PI)
    u_theta_o = np.where(norm > 1e-9, u_theta_o, _PI)
    return ub_min, ub_max, mean, u_theta_o, theta_e.max(0)


def build_light_trees(power, bmin, bmax, axis, theta_o, theta_e, decay):
    """Build the chunk's trees and the frame -> tree-row map.

    ``power [E]`` is frame-constant (an emitter's power is what the flat
    table weights it by; only its geometry moves); the rest are ``[R, E, ...]``
    with one row per frame of the chunk. Returns
    ``(node_f [rows, nodes, LT_F_WIDTH], node_i [rows, nodes, LT_I_WIDTH],
    entry_leaf [rows, E], frame_row [R])``.

    Frames with identical emitter geometry share a row, so a static light rig
    collapses to one tree however long the chunk is.
    """
    rows = int(bmin.shape[0])
    entries = int(power.shape[0])
    flat = np.concatenate(
        (
            bmin.reshape(rows, -1),
            bmax.reshape(rows, -1),
            axis.reshape(rows, -1),
            theta_o.reshape(rows, -1),
            theta_e.reshape(rows, -1),
        ),
        axis=1,
    )
    uniq_rows, frame_row = np.unique(flat, axis=0, return_inverse=True)
    frame_row = np.asarray(frame_row, dtype=np.int32).reshape(rows)
    distinct = int(uniq_rows.shape[0])
    if distinct * max(entries, 1) > PER_FRAME_BUILD_BUDGET:
        u_min, u_max, u_axis, u_to, u_te = union_over_rows(
            bmin, bmax, axis, theta_o, theta_e
        )
        node_f, node_i, leaf = build_light_tree(
            power, u_min, u_max, u_axis, u_to, u_te, decay
        )
        return (
            node_f[None],
            node_i[None],
            leaf[None],
            np.zeros(rows, dtype=np.int32),
        )
    # One representative frame per distinct configuration.
    first = np.zeros(distinct, dtype=np.int64)
    first[frame_row[::-1]] = np.arange(rows - 1, -1, -1, dtype=np.int64)
    built_f = []
    built_i = []
    built_leaf = []
    for r in first:
        node_f, node_i, leaf = build_light_tree(
            power, bmin[r], bmax[r], axis[r], theta_o[r], theta_e[r], decay
        )
        built_f.append(node_f)
        built_i.append(node_i)
        built_leaf.append(leaf)
    return (
        np.stack(built_f),
        np.stack(built_i),
        np.stack(built_leaf),
        frame_row,
    )
