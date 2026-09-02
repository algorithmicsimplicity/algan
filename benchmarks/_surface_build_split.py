"""Probe: where the batched surface build's time goes, after P11.

``surfaces.get_render_primitives_batched`` was the largest single item in the
reference render (85.35 s, 21.9%). P11 halved its dominant callee and took the
stage to 56.62 s (15.8%) -- so **every share in the published split
(DESIGN_optimization_targets.md, P10, "Inside it: 60% is
compute_grid_vertex_normals") was measured before that halving and is stale**.
`RENDERER_WORK_QUEUE.md` item 12 asks for a re-split before anything inside
P10 is chosen; this is it.

Two tables:

1. ``get_render_primitives_batched`` by section -- the shared prefix (grid
   materialize + stack, weld flags, vertex normals, and the two whole-stack
   triangle-vertex gathers) against the per-surface tail
   (``Surface._build_render_primitive``), which is itself split into its
   timeline reads, its per-surface gathers, the shader parameters, the packed
   flatten and the primitive construction. The "prefix:" and "tail:" rows are
   the shared-prefix / per-surface-tail division the P10 table reports.
2. Inside ``compute_grid_vertex_normals`` -- sides and crosses, the two seam
   merges, the pole fans, the final normalize.

Both tables come from **instrumented copies** of the two functions, living in
this file, and the probe asserts on entry that each copy's output is
bit-identical to the shipped function's on the same input -- bit patterns, not
``==``, so a signed-zero flip or a changed NaN payload cannot pass, the same
standard ``benchmarks/_grid_normals_ab.py`` holds P11 to. It aborts if not.
Copies rather than wrappers because the interesting boundaries are *inside*
those function bodies, and `algan/` is not a place to leave timing hooks.

Timing discipline, because wall clock on a 4-vCPU CPU box invents results if
you let it (P11's first A/B read 0.93x and was pure noise):

* the first pass is discarded (allocator / cache warm-up);
* **shares are computed per pass and then medianed**, not derived from
  medianed seconds. Under this much jitter the per-row medians do not add up
  to the whole and a share-of-medians table silently does not sum to 100%;
* every row prints the min and max of its per-pass share. **Two rows whose
  ranges overlap cannot be ranked against each other** -- the table says so
  rather than pretending the order is meaningful;
* the residual is a row, so an incomplete split is visible.

Sections nest and the timer charges **exclusive** time, so a callee reached
from the per-surface tail is not also charged to the shared prefix. The one
wrapper left on a shared helper (``grid_to_triangle_vertices``) is gated on a
flag the tail sets, because an unscoped wrapper on a shared helper times the
whole program -- that mistake produced one wrong reading in this document's
history already.

    .venv/bin/python benchmarks/_surface_build_split.py [num_surfaces]

Prep only, no render.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _memory_cap import cap_process_memory  # noqa: E402

# Tensor sizes here come from parameters (surface count x frames x grid), not
# from a real scene, so the ceiling belongs on this process -- see
# benchmarks/_memory_cap.py. Prep only: no render arena is charged against it.
cap_process_memory(8)

# A warm daemon carries the previous run's adaptive renderer state, so a
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import algan.mobs.surfaces.surface as surface_mod  # noqa: E402
from algan.animatable_base.animatable import Animatable  # noqa: E402
from algan.animatable_base.mob_materials import MobMaterialsMixin  # noqa: E402
from algan.animation_timeline.animation_contexts import Off  # noqa: E402
from algan.constants.spatial import RIGHT  # noqa: E402
from algan.mobs.shapes_3d import Sphere  # noqa: E402
from algan.mobs.surfaces.surface import Surface  # noqa: E402
from algan.rendering.raytracing.primitives import (  # noqa: E402
    LogicalPNTrianglePrimitive,
)
from algan.scene import Scene  # noqa: E402
from algan.utils.tensor_utils import broadcast_cross_product  # noqa: E402

NUM_SURFACES = 220
#: Manim counts patches and Algan counts sampled vertices, so each value is
#: used as the grid dimension plus one: (23, 11) -> a 24 x 12 grid, the shape
#: the reference profile measured ([19, 50, 24, 12, 3]).
RESOLUTION = (23, 11)
WINDOW_FRAMES = 50
#: Surfaces per batched call in the reference profile (223 surfaces, 18.6 per
#: call). The natural grouping here puts all of them in one call, so the
#: forced split is measured too -- per-call dispatch overhead is one of the
#: things being ranked.
REFERENCE_PER_CALL = 19
PASSES = 9


class Sections:
    """Exclusive-time accumulator with a nesting stack."""

    def __init__(self):
        self.total = collections.defaultdict(float)
        self._stack = []
        self.enabled = False
        self.in_tail = False

    def reset(self):
        self.total.clear()
        self._stack.clear()

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        self._stack.append(0.0)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            child = self._stack.pop()
            self.total[name] += elapsed - child
            if self._stack:
                self._stack[-1] += elapsed


SECTIONS = Sections()


def bits(tensor):
    """Raw bit patterns, so a signed zero or a NaN payload cannot compare equal."""
    dtype = torch.int32 if tensor.element_size() == 4 else torch.int64
    return tensor.contiguous().view(dtype)


def assert_bit_identical(what, reference, candidate):
    if isinstance(reference, tuple):
        if reference != candidate:
            raise SystemExit(
                f"{what}: instrumented copy disagrees: {reference} vs {candidate}"
            )
        return
    if reference is None and candidate is None:
        return
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise SystemExit(f"{what}: instrumented copy changed shape/dtype")
    if not torch.equal(bits(reference), bits(candidate)):
        raise SystemExit(
            f"{what}: the instrumented copy is NOT bit-identical to the "
            "shipped function -- the table would be measuring different code"
        )


# --------------------------------------------------------------------------
# Instrumented copy of compute_grid_vertex_normals (Table 2).
#
# Verbatim apart from the timing markers, and from taking only the paired
# branch, which is the shipped default (ALGAN_OPT_DISABLE=gridnormals selects
# the other arm; this probe does not measure it).
# --------------------------------------------------------------------------
def instrumented_grid_vertex_normals(grid, sections):
    with sections("normals: sides + crosses"):
        grid_x_plus_1 = grid.roll(-1, -3)
        grid_x_minus_1 = grid.roll(1, -3)
        grid_y_plus_1 = grid.roll(-1, -2)
        grid_y_minus_1 = grid.roll(1, -2)
        side_x_minus = grid_x_minus_1 - grid
        side_y_minus = grid_y_minus_1 - grid
        side_x_plus = grid_x_plus_1 - grid
        side_y_plus = grid_y_plus_1 - grid
        normals_xm_ym = broadcast_cross_product(side_x_minus, side_y_minus)
        normals_ym_xp = broadcast_cross_product(side_y_minus, side_x_plus)
        normals_xp_yp = broadcast_cross_product(side_x_plus, side_y_plus)
        normals_yp_xm = broadcast_cross_product(side_y_plus, side_x_minus)
        normals_xm_ym[..., 0, :, :] = 0
        normals_yp_xm[..., 0, :, :] = 0
        normals_ym_xp[..., -1, :, :] = 0
        normals_xp_yp[..., -1, :, :] = 0
        normals_xm_ym[..., :, 0, :] = 0
        normals_ym_xp[..., :, 0, :] = 0
        normals_xp_yp[..., :, -1, :] = 0
        normals_yp_xm[..., :, -1, :] = 0
        unnormalized_normals = (
            normals_xm_ym + normals_ym_xp + normals_xp_yp + normals_yp_xm
        )

    with sections("normals: x seam merge"):
        is_closed_x = torch.all(
            (grid[..., 0, :, :] - grid[..., -1, :, :]).abs() < 1e-4, dim=(-1, -2)
        )
        mask_x = is_closed_x.view(*is_closed_x.shape, 1, 1)
        closed_normals_x = (
            unnormalized_normals[..., 0, :, :] + unnormalized_normals[..., -1, :, :]
        )
        unnormalized_normals[..., 0, :, :] = torch.where(
            mask_x, closed_normals_x, unnormalized_normals[..., 0, :, :]
        )
        unnormalized_normals[..., -1, :, :] = torch.where(
            mask_x, closed_normals_x, unnormalized_normals[..., -1, :, :]
        )

    with sections("normals: y seam merge"):
        is_closed_y = torch.all(
            (grid[..., :, 0, :] - grid[..., :, -1, :]).abs() < 1e-4, dim=(-1, -2)
        )
        mask_y = is_closed_y.view(*is_closed_y.shape, 1, 1)
        closed_normals_y = (
            unnormalized_normals[..., :, 0, :] + unnormalized_normals[..., :, -1, :]
        )
        unnormalized_normals[..., :, 0, :] = torch.where(
            mask_y, closed_normals_y, unnormalized_normals[..., :, 0, :]
        )
        unnormalized_normals[..., :, -1, :] = torch.where(
            mask_y, closed_normals_y, unnormalized_normals[..., :, -1, :]
        )

    def _orient_to_ring(pole_normal, ring_normal):
        dot = (pole_normal * ring_normal).sum(-1, keepdim=True)
        return torch.where(dot < 0, -pole_normal, pole_normal)

    def _merged_pole_normal(pole_index, ring_index, reverse):
        pole = grid[..., :1, pole_index, :]
        ring = grid[..., :, ring_index, :]
        first = ring[..., :-1, :] - pole
        second = ring[..., 1:, :] - pole
        if reverse:
            first, second = second, first
        pole_normal = broadcast_cross_product(first, second).sum(-2, keepdim=True)
        accumulated = unnormalized_normals[..., :, pole_index, :]
        is_pole = torch.all(
            (grid[..., :, pole_index, :] - pole).abs() < 1e-4, dim=(-1, -2)
        )
        usable = pole_normal.norm(p=2, dim=-1, keepdim=True) > 1e-12
        replace = is_pole.view(*is_pole.shape, 1, 1) & usable
        ring_normal = unnormalized_normals[..., :, ring_index, :].sum(-2, keepdim=True)
        return torch.where(
            replace,
            _orient_to_ring(pole_normal, ring_normal),
            accumulated,
        )

    with sections("normals: pole fans"):
        if grid.shape[-2] > 1:
            merged_poles = [
                (pole_index, _merged_pole_normal(pole_index, ring_index, reverse))
                for pole_index, ring_index, reverse in ((0, 1, True), (-1, -2, False))
            ]
            for pole_index, merged in merged_poles:
                unnormalized_normals[..., :, pole_index, :] = merged

    with sections("normals: final normalize"):
        return -F.normalize(unnormalized_normals, p=2, dim=-1)


# --------------------------------------------------------------------------
# Instrumented copy of get_render_primitives_batched (Table 1).
# --------------------------------------------------------------------------
def instrumented_batched(surfaces, sections):
    with sections("prefix: grid materialize + stack"):
        grids = torch.stack(
            [s._reshape_grid_for_render(s.grid.location) for s in surfaces]
        )
    with sections("prefix: surface_weld_flags"):
        weld = surface_mod.surface_weld_flags(grids)
    with sections("prefix: compute_grid_vertex_normals"):
        normals_grid = surface_mod.compute_grid_vertex_normals(grids)
    with sections("prefix: gather normals (whole stack)"):
        vertex_normals = surface_mod.grid_to_triangle_vertices(normals_grid, weld)
    with sections("prefix: gather corners (whole stack)"):
        corners = surface_mod.grid_to_triangle_vertices(grids, weld)
    out = []
    was = sections.in_tail
    sections.in_tail = True
    try:
        for i, s in enumerate(surfaces):
            with sections("tail: _build_render_primitive residual"):
                out.append(
                    s._build_render_primitive(
                        grids[i],
                        vertex_normals[i],
                        precomputed_corners=corners[i],
                        weld=weld,
                    )
                )
    finally:
        sections.in_tail = was
    return out


def install_tail_wrappers():
    """Time the helpers ``_build_render_primitive`` calls, scoped to the tail."""
    real_gtv = surface_mod.grid_to_triangle_vertices
    real_flatten = Surface._flatten_packed_triangle_vertices
    real_attr = Animatable.get_animated_attribute
    real_shader = MobMaterialsMixin.get_shader_params
    real_prim_init = LogicalPNTrianglePrimitive.__init__

    def gtv(grid, weld=(False, False, False)):
        if not SECTIONS.in_tail:
            return real_gtv(grid, weld)
        with SECTIONS("tail: grid_to_triangle_vertices (per surface)"):
            return real_gtv(grid, weld)

    def flatten(self, values):
        if not SECTIONS.in_tail:
            return real_flatten(self, values)
        with SECTIONS("tail: _flatten_packed_triangle_vertices"):
            return real_flatten(self, values)

    def attr(self, *args, **kwargs):
        if not SECTIONS.in_tail:
            return real_attr(self, *args, **kwargs)
        with SECTIONS("tail: timeline attribute reads"):
            return real_attr(self, *args, **kwargs)

    def shader(self):
        if not SECTIONS.in_tail:
            return real_shader(self)
        with SECTIONS("tail: get_shader_params"):
            return real_shader(self)

    def prim_init(self, *args, **kwargs):
        if not SECTIONS.in_tail:
            return real_prim_init(self, *args, **kwargs)
        with SECTIONS("tail: primitive construction"):
            return real_prim_init(self, *args, **kwargs)

    surface_mod.grid_to_triangle_vertices = gtv
    Surface._flatten_packed_triangle_vertices = flatten
    Animatable.get_animated_attribute = attr
    MobMaterialsMixin.get_shader_params = shader
    LogicalPNTrianglePrimitive.__init__ = prim_init


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def print_table(title, per_pass_rows, whole_samples):
    """``per_pass_rows`` is a list of ``{section: seconds}``, one per pass."""
    names = sorted({name for row in per_pass_rows for name in row})
    shares = collections.defaultdict(list)
    for row, whole in zip(per_pass_rows, whole_samples):
        accounted = 0.0
        for name in names:
            value = row.get(name, 0.0)
            accounted += value
            shares[name].append(value / whole)
        shares["RESIDUAL (unattributed)"].append((whole - accounted) / whole)

    print(f"\n== {title} ==")
    print(f"{'section':<48} {'share':>8}  {'share min..max':>16}  {'median':>10}")
    ranked = sorted(
        names + ["RESIDUAL (unattributed)"], key=lambda n: -median(shares[n])
    )
    bands = {}
    for name in ranked:
        values = shares[name]
        lo, hi = min(values), max(values)
        bands[name] = (lo, hi)
        seconds = (
            median([row.get(name, 0.0) for row in per_pass_rows])
            if name in names
            else float("nan")
        )
        print(
            f"{name:<48} {median(values):>7.1%}  {lo:>7.1%}..{hi:<7.1%}  "
            f"{seconds * 1e3:>8.2f}ms"
        )
    whole_median = median(whole_samples)
    print(
        f"{'WHOLE':<48} {1.0:>7.1%}  {'':>16}  {whole_median * 1e3:>8.2f}ms  "
        f"(min {min(whole_samples) * 1e3:.1f} max {max(whole_samples) * 1e3:.1f})"
    )
    overlapping = [
        (a, b)
        for i, a in enumerate(ranked)
        for b in ranked[i + 1 :]
        if bands[a][0] <= bands[b][1] and bands[b][0] <= bands[a][1]
    ]
    if overlapping:
        print(
            "  NOT SEPARABLE (share ranges overlap, so this order is not a "
            "ranking): " + "; ".join(f"{a} ~ {b}" for a, b in overlapping[:8])
        )


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SURFACES
    with Scene() as scene:
        with Off():
            spheres = [Sphere(radius=0.2, resolution=RESOLUTION) for _ in range(count)]
        for sphere in spheres:
            sphere.spawn(animate=False)
        # A real move, so the materialized grid varies over the window instead
        # of being one broadcast frame.
        for sphere in spheres:
            sphere.move(RIGHT * 0.1)

        first = spheres[0]
        print(
            f"surfaces: {count}  grid: {first.grid_width} x {first.grid_height}  "
            f"(resolution={RESOLUTION})"
        )

        scene.scene_times.append(
            [
                scene.scene_times[-1][0],
                round(scene._recorded_end_time_for_render() * scene.frames_per_second),
            ]
        )
        scene._initialize_frames()
        start_ind, end_ind = scene.scene_times[-1]
        end_ind = min(end_ind, start_ind + WINDOW_FRAMES)
        print(f"window: frames {start_ind}..{end_ind}")

        with scene._batch_prep_context():
            times = torch.arange(start_ind, end_ind)
            scene.timeline_manager.set_state_to_times(
                times / scene.frames_per_second, active_mobs=spheres
            )

            grid_shape = tuple(first.grid.location.shape)
            print(f"grid.location per surface: {grid_shape}")
            print(
                f"stacked grid: [{count}, {grid_shape[0]}, {first.grid_width}, "
                f"{first.grid_height}, 3]  (reference: [19, 50, 24, 12, 3])"
            )

            # --- the copies are checked against the shipped functions first ---
            grids = torch.stack(
                [s._reshape_grid_for_render(s.grid.location) for s in spheres]
            )
            assert_bit_identical(
                "compute_grid_vertex_normals",
                surface_mod.compute_grid_vertex_normals(grids),
                instrumented_grid_vertex_normals(grids, SECTIONS),
            )
            shipped = surface_mod.get_render_primitives_batched(spheres)
            copied = instrumented_batched(spheres, SECTIONS)
            for a, b in zip(shipped, copied):
                assert_bit_identical("primitive corners", a.corners, b.corners)
                assert_bit_identical("primitive normals", a.normals, b.normals)
                assert_bit_identical(
                    "primitive colors",
                    a.colors.as_subclass(torch.Tensor),
                    b.colors.as_subclass(torch.Tensor),
                )
            print(
                f"instrumented copies verified bit-identical over "
                f"{len(shipped)} primitives"
            )

            install_tail_wrappers()

            def one_pass(groups):
                SECTIONS.reset()
                SECTIONS.enabled = True
                start = time.perf_counter()
                for group in groups:
                    instrumented_batched(group, SECTIONS)
                whole = time.perf_counter() - start
                SECTIONS.enabled = False
                return whole, dict(SECTIONS.total)

            batchings = [
                (f"one call of {count} surfaces", [spheres]),
                (
                    f"{-(-count // REFERENCE_PER_CALL)} calls of "
                    f"{REFERENCE_PER_CALL} surfaces",
                    [
                        spheres[i : i + REFERENCE_PER_CALL]
                        for i in range(0, count, REFERENCE_PER_CALL)
                    ],
                ),
            ]
            for label, groups in batchings:
                one_pass(groups)  # discarded
                whole_samples, rows = [], []
                for _ in range(PASSES):
                    whole, totals = one_pass(groups)
                    whole_samples.append(whole)
                    rows.append(totals)
                print_table(
                    f"Table 1 -- get_render_primitives_batched, {label}",
                    rows,
                    whole_samples,
                )

            # --- Table 2 ---
            SECTIONS.in_tail = False
            instrumented_grid_vertex_normals(grids, SECTIONS)  # discarded
            whole_samples, rows = [], []
            for _ in range(PASSES):
                SECTIONS.reset()
                SECTIONS.enabled = True
                start = time.perf_counter()
                instrumented_grid_vertex_normals(grids, SECTIONS)
                whole_samples.append(time.perf_counter() - start)
                SECTIONS.enabled = False
                rows.append(dict(SECTIONS.total))
            print_table(
                "Table 2 -- inside compute_grid_vertex_normals "
                f"(one call, [{count}, {grid_shape[0]}, {first.grid_width}, "
                f"{first.grid_height}, 3])",
                rows,
                whole_samples,
            )

            print(f"\nload average (1/5/15 min): {os.getloadavg()}")
            print(f"torch threads: {torch.get_num_threads()}")


if __name__ == "__main__":
    main()
