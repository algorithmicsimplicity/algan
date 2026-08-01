"""Measure render memory and regenerate ``algan/rendering/mem_usage.py``.

Batch sizing used to be driven by hand-written byte formulas that mirrored the
allocation sequences in the tracer, the raster pipeline and post-processing.
Every kernel change required updating them by hand, and when they drifted the
failure surfaced as an out-of-memory error in somebody's render.

This module measures instead. Because
:meth:`~algan.utils.memory_utils.ManualMemory.get_tensor` is the arena's only
allocation entry point, arming its recorder captures every buffer -- including
ones added after this was written -- and the annotated scopes carry the shape
parameters each cost centre is driven by. Re-running this generator is the
whole update procedure.

Three kinds of model come out of it:

``trace``
    An ordered allocation stream whose element counts are exact and affine in
    the frame count. Replaying it reproduces the arena byte-for-byte. Used
    where the size is a pure function of input shapes.
``units``
    Bytes per unit of some driver (per pool slot, per primary ray, per
    frame-triangle) solved exactly from a sweep, with a zero-residual
    requirement. Used where the runtime chooses the driver itself, so a
    recorded peak would only measure the arena it was given.
``density``
    Structural bytes-per-unit as above, times a units-per-frame-pixel density
    that depends on scene *content* rather than shape. The density ships as a
    conservative corpus percentile and is raised in-job from observations.

Usage::

    .venv/Scripts/python.exe -m algan.utils.calibrate_memory            # write
    .venv/Scripts/python.exe -m algan.utils.calibrate_memory --verify   # check
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction

from algan.rendering.mem_usage_runtime import (
    ALLOC,
    TEMP_POP,
    TEMP_PUSH,
    is_monotone_in_frames,
    peak_bytes,
    replay,
    trace_from_events,
)

# --------------------------------------------------------------------------
# Scope models
# --------------------------------------------------------------------------

# Route keys are *derived*, not declared: a scope's key is every recorded
# parameter that is not the fit axis or a regressor. Adding a parameter to an
# annotation therefore extends the key automatically, and a parameter that
# turns out to matter shows up as a key collision rather than as silent
# averaging.


class TraceModel:
    """Replay-trace scope: exact, fitted along ``axis``."""

    kind = "trace"

    def __init__(self, axis="frames", merge_siblings=False):
        self.axis = axis
        # Some cost centres are annotated as several adjacent groups because
        # the allocations are interleaved with unrelated logic (the camera and
        # light copies, for instance, are ~120 lines apart). Those groups are
        # cumulative -- all of them are live at once -- so they must be summed
        # into one observation rather than treated as rival variants.
        self.merge_siblings = merge_siblings

    def key_params(self, params):
        return tuple(sorted(
            (name, value) for name, value in params.items()
            if name != self.axis))


class UnitsModel:
    """Unit-coefficient scope: ``bytes = sum(coef[r] * r) + fixed``."""

    kind = "units"

    def __init__(self, regressors, products=(), merge_siblings=False):
        self.regressors = tuple(regressors)
        # Derived regressors written as "a*b", evaluated from the params.
        self.products = tuple(products)
        self.merge_siblings = merge_siblings

    def all_regressors(self):
        return self.regressors + self.products

    def key_params(self, params):
        used = set(self.regressors)
        for product in self.products:
            used.update(product.split("*"))
        return tuple(sorted(
            (name, value) for name, value in params.items()
            if name not in used))

    def evaluate(self, params):
        row = [float(params.get(name, 0)) for name in self.regressors]
        for product in self.products:
            value = 1.0
            for factor in product.split("*"):
                value *= float(params.get(factor, 0))
            row.append(value)
        return row


class DensityModel(UnitsModel):
    """Value-dependent scope: structural units plus a learned density."""

    kind = "density"

    def __init__(self, regressors, density_of, density_per, products=()):
        super().__init__(regressors, products)
        self.density_of = density_of
        self.density_per = tuple(density_per)

    def key_params(self, params):
        # The density's numerator and denominators are measured quantities,
        # not route selectors. Leaving them in the key would split every
        # distinct frame count and resolution into its own single-sample
        # bucket, and a one-sample linear solve fits noise exactly.
        excluded = set(self.regressors) | {self.density_of}
        excluded.update(self.density_per)
        for product in self.products:
            excluded.update(product.split("*"))
        return tuple(sorted(
            (name, value) for name, value in params.items()
            if name not in excluded))


SCOPE_MODELS = {
    # Post-processing genuinely depends on resolution in non-linear ways
    # (``ceil`` on blur radii, FFT lengths rounded to fast sizes), so it is
    # keyed on resolution and replayed. Everything below scales linearly in
    # element counts, so it is fitted per unit and generalises to any
    # resolution -- including ones the corpus never measured.
    "postprocess": TraceModel(),
    "frame_buffers": UnitsModel(regressors=("out_cells",)),
    "frame_accum": UnitsModel(regressors=("accum_cells",)),
    "persistent_inputs": UnitsModel(
        regressors=("cam_frames", "light_pos_cells", "light_col_cells"),
        merge_siblings=True),
    "batch_metadata": UnitsModel(regressors=(), merge_siblings=True),
    "wavefront_state": UnitsModel(regressors=("pool", "primary")),
    # Drivers are reported by the precompute functions themselves: the table
    # frame count is the longest dynamic input rather than the batch's frame
    # count, and each table is emitted conditionally.
    "raster_precompute": UnitsModel(
        regressors=("tri_screen_cells", "tri_bounds_cells",
                    "bez_bounds_cells")),
    "sparse_discovery": DensityModel(
        regressors=("discovery_frags", "num_fragments", "num_covered",
                    "num_pairs"),
        density_of="num_fragments", density_per=("frames", "pixels")),
    # Measured for staleness detection only: its size is already known
    # exactly from get_merged_scene_arena_nbytes.
    "scene_upload": None,
}

DENSITY_PERCENTILE = 95.0

# Densities are seeded from the corpus and then only ever raised in-job, so an
# under-seeded density costs one OOM retry while an over-seeded one costs a
# slightly smaller batch. The multiplier reflects that asymmetry.
DENSITY_SAFETY = 1.15


class CalibrationError(RuntimeError):
    """A measurement contradicts the model form -- the table cannot be built.

    Always names the scope and what disagreed. This is the failure the design
    exists to produce: a loud generator error instead of a silent
    under-estimate that surfaces as an OOM in a user's render.
    """


# --------------------------------------------------------------------------
# Observation collection
# --------------------------------------------------------------------------

class Observation:
    """One recorded scope occurrence."""

    __slots__ = ("scope", "params", "events", "peak_forward", "peak_reverse",
                 "alloc_count", "entry_forward", "source")

    def __init__(self, record, source):
        self.scope = record.name
        self.params = dict(record.params)
        self.events = list(record.events)
        self.peak_forward = record.peak_forward
        self.peak_reverse = record.peak_reverse
        self.alloc_count = record.alloc_count
        self.entry_forward = record.entry_forward
        self.source = source

    def total_peak(self):
        return self.peak_forward + self.peak_reverse

    def alloc_bytes(self):
        """Sum of ``numel * itemsize`` over this scope's own allocations.

        Alignment-free by construction. Unit coefficients are fitted against
        this rather than a pointer delta because a scope that mixes item sizes
        -- the raster precompute tables interleave ``bool`` and ``float32`` --
        pads *between* allocations by an amount that varies with the element
        counts, which no linear model in those counts can reproduce.
        """
        return sum(event[4] * event[5]
                   for event in self.events if event[0] == "alloc")

    def normalized_peak(self):
        """Peak with entry-alignment padding removed.

        ``peak_forward`` is measured from wherever the arena happened to be,
        so it carries up to seven bytes of padding that vary between
        occurrences. That sawtooth makes an otherwise perfectly linear scope
        un-fittable, so unit coefficients are solved against the peak this
        exact allocation stream would reach from an aligned start. The
        discarded padding is carried separately as ``align_slack``.
        """
        concrete = []
        for event in self.events:
            if event[0] == "alloc":
                concrete.append((ALLOC, event[4], 0, event[5], event[3]))
            elif event[0] == "temp_push":
                concrete.append((TEMP_PUSH, event[1]))
            else:
                concrete.append((TEMP_POP,))
        result = replay(tuple(concrete), 1, 0)
        return result.peak + result.reverse_peak

    def max_itemsize(self):
        sizes = [event[5] for event in self.events if event[0] == "alloc"]
        return max(sizes) if sizes else 1

    def align_slack(self):
        """Padding the alignment-free fit does not account for.

        Measured from this scope's *own* allocations only. ``total_peak``
        includes any nested child scope, while ``alloc_bytes`` does not, so
        differencing those two would book an entire child scope as padding --
        which charged the Monte Carlo accumulator, hundreds of megabytes, as
        alignment slack on every HDR frame buffer.

        Internal padding (between allocations of differing item sizes) comes
        from replaying the stream; entry padding is bounded by the largest item
        size in it.
        """
        internal = max(0, self.normalized_peak() - self.alloc_bytes())
        return internal + max(0, self.max_itemsize() - 1)


def _merge_adjacent(records):
    """Fold runs of adjacent same-named siblings into one record.

    Only applied to scopes declared ``merge_siblings``. Adjacency matters: two
    ``persistent_inputs`` groups within one batch are cumulative and belong
    together, while the same scope in the *next* batch is a separate sample and
    is never adjacent to the previous batch's.
    """
    merged = []
    for record in records:
        model = SCOPE_MODELS.get(record.name)
        mergeable = getattr(model, "merge_siblings", False)
        if (mergeable and merged and merged[-1].name == record.name):
            previous = merged[-1]
            previous.events.extend(record.events)
            previous.children.extend(record.children)
            previous.alloc_count += record.alloc_count
            previous.exit_forward = record.exit_forward
            previous.exit_reverse = record.exit_reverse
            previous.peak_forward = max(
                previous.peak_forward,
                record.peak_forward + (record.entry_forward
                                       - previous.entry_forward))
            previous.peak_reverse = max(
                previous.peak_reverse,
                record.peak_reverse + (previous.entry_reverse
                                       - record.entry_reverse))
            previous.params.update(record.params)
            continue
        merged.append(record)
    return merged


def _walk(record, source, out):
    if record.name != "<root>":
        out.append(Observation(record, source))
    for child in _merge_adjacent(record.children):
        _walk(child, source, out)
    return out


def observations_from_recorders(recorders, source):
    """Flatten recorded scope trees into observations."""
    out = []
    for recorder in recorders:
        _walk(recorder.root, source, out)
    return out


# (input_bytes, measured_peak_bytes) pairs for the transient GPU merge, which
# builds out of place in pool headroom rather than in the arena. Populated by
# ``collect_from_render`` while MERGE_TRACK_PEAK is on.
_NONARENA_SAMPLES = {"merge": []}


def collect_from_render(scene_func, video_settings, source, *, reset=True):
    """Render a scene with the arena recorder armed; return observations."""
    from algan.rendering.raytracing import scene_builder as scb
    from algan.rendering.raytracing import settings as rts
    from algan.scene import Scene
    from algan.scene_manager import SceneManager
    from algan.utils import memory_utils as mu

    original_merge = scb._merge_scene
    saved_track = rts.MERGE_TRACK_PEAK

    def _measuring_merge(primitives, *args, **kwargs):
        # Read the input size *first*: merging nulls the per-collection _rt_*
        # arrays it is computed from, so measuring afterwards raises and the
        # sample is silently lost.
        try:
            inputs = int(scb.gpu_merge_input_bytes(primitives))
        except Exception:  # noqa: BLE001
            inputs = 0
        scene = original_merge(primitives, *args, **kwargs)
        try:
            measured = int(scene.get("_gpu_merge_peak_bytes", -1))
            if measured >= 0 and inputs > 0:
                _NONARENA_SAMPLES["merge"].append((inputs, measured))
        except Exception:  # noqa: BLE001
            pass
        return scene

    mu.clear_recorded_arenas()
    mu.set_auto_record(True)
    rts.MERGE_TRACK_PEAK = True
    scb._merge_scene = _measuring_merge
    try:
        SceneManager.reset()
        scene_func()
        Scene.save_video(
            f"_calibrate_{source}", video_settings, reset=reset,
            overwrite=True)
        for name, inputs, peak in mu.recorded_nonarena_peaks():
            _NONARENA_SAMPLES.setdefault(name, []).append((inputs, peak))
        arenas = mu.recorded_arenas()
        return observations_from_recorders(
            [arena._recorder for arena in arenas if arena._recorder], source)
    finally:
        scb._merge_scene = original_merge
        rts.MERGE_TRACK_PEAK = saved_track
        mu.set_auto_record(False)


def fit_nonarena():
    """Measured peak-to-input ratios for out-of-arena transient builds.

    These replace hand-guessed multipliers (the GPU merge's was 6.0). A ratio
    is a bound over the corpus, rounded up, not a mean: the headroom check it
    feeds is the thing standing between a large batch and a driver-level
    out-of-memory error, and torch's counters cannot see Taichi's separate
    pool at all.
    """
    import math

    fits = {}
    for name, samples in _NONARENA_SAMPLES.items():
        if not samples:
            continue
        ratios = [peak / inputs for inputs, peak in samples if inputs > 0]
        if not ratios:
            continue
        fits[name] = {
            "ratio": math.ceil(max(ratios) * 10) / 10,
            "ratio_p95": round(_percentile(ratios, 95.0), 3),
            "samples": len(ratios),
        }
    return fits


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _stream_shape(observation):
    """Structural signature of an allocation stream, ignoring element counts.

    Two observations share a shape when they made the same allocations of the
    same dtypes in the same temp nesting -- differing only in how big each one
    was. That is exactly the equivalence one replay trace can represent.
    """
    return tuple(
        (event[0], event[2], event[3], event[5]) if event[0] == "alloc"
        else (event[0],)
        for event in observation.events)


def fit_trace(scope, model, observations):
    """Build one replay trace per route key, verified against every sample.

    Requires at least two distinct values along the fit axis. Verification runs
    on *all* observations, including the two the trace was solved from, so an
    allocation that is not actually affine in the axis fails here rather than
    at render time.
    """
    by_key = {}
    for observation in observations:
        by_key.setdefault(model.key_params(observation.params), []).append(
            observation)

    traces = {}
    for key, samples in by_key.items():
        # A scope can legitimately take different allocation paths at one key
        # when the *content* differs -- bloom skips its whole blur chain on a
        # frame with no glow, for instance. The batcher must reserve for the
        # largest such variant, so fit that one and then prove it dominates
        # every other observation. A variant that is not dominated means the
        # size is not a function of the key at all, and that is a hard error.
        variants = {}
        for sample in samples:
            variants.setdefault(_stream_shape(sample), []).append(sample)

        def _variant_rank(shape):
            group = variants[shape]
            return (len({int(s.params.get(model.axis, 1)) for s in group}) > 1,
                    len(shape),
                    max(s.peak_forward for s in group))

        dominant_shape = max(variants, key=_variant_rank)
        dominant = variants[dominant_shape]
        dominant_ids = {id(sample) for sample in dominant}
        if len(variants) > 1:
            print(f"[calibrate] scope {scope!r} key {key}: "
                  f"{len(variants)} content-dependent allocation paths "
                  f"({sorted(len(s) for s in variants)} events); reserving "
                  f"for the largest.")

        by_axis = {}
        for sample in dominant:
            axis_value = int(sample.params.get(model.axis, 1))
            by_axis.setdefault(axis_value, []).append(sample)

        axis_values = sorted(by_axis)
        if len(axis_values) < 2:
            # Constant along the axis (batch metadata, for instance). Solve
            # with a zero slope from the single sample available.
            only = by_axis[axis_values[0]][0]
            trace = trace_from_events(
                only.events,
                ((axis_values[0], only.events),
                 (axis_values[0] + 1, only.events)),
            ) if only.events else ()
            traces[key] = trace
            continue

        low, high = axis_values[0], axis_values[-1]
        try:
            trace = trace_from_events(
                by_axis[low][0].events,
                ((low, by_axis[low][0].events),
                 (high, by_axis[high][0].events)),
            )
        except ValueError as exc:
            raise CalibrationError(f"scope {scope!r} at key {key}: {exc}")

        for sample in samples:
            axis_value = int(sample.params.get(model.axis, 1))
            predicted = peak_bytes(trace, axis_value, sample.entry_forward)
            if id(sample) in dominant_ids:
                if predicted != sample.peak_forward:
                    raise CalibrationError(
                        f"scope {scope!r} at key {key}: replay predicts "
                        f"{predicted} B at {model.axis}={axis_value} but "
                        f"{sample.peak_forward} B was measured "
                        f"(source {sample.source}). The allocation stream is "
                        f"not affine in {model.axis}.")
            elif predicted < sample.peak_forward:
                raise CalibrationError(
                    f"scope {scope!r} at key {key}: a content-dependent "
                    f"allocation path used {sample.peak_forward} B at "
                    f"{model.axis}={axis_value}, more than the "
                    f"{predicted} B reserved by the largest path "
                    f"(source {sample.source}). No single variant bounds the "
                    f"others, so this scope's size is not a function of its "
                    f"route key.")
        if not is_monotone_in_frames(trace):
            raise CalibrationError(
                f"scope {scope!r} at key {key}: replayed peak decreases as "
                f"the frame count grows. render_loop's chunk-size search "
                f"requires a monotone fit predicate.")
        traces[key] = trace
    return traces


def _collinear_regressors(rows, names):
    """Names whose columns are proportional across every sample."""
    groups = []
    for index, name in enumerate(names):
        column = [row[index] for row in rows]
        placed = False
        for group in groups:
            reference = [row[names.index(group[0])] for row in rows]
            ratios = {
                round(a / b, 9)
                for a, b in zip(column, reference) if b
            }
            if len(ratios) == 1 and all(
                    (b != 0) or (a == 0)
                    for a, b in zip(column, reference)):
                group.append(name)
                placed = True
                break
        if not placed:
            groups.append([name])
    return sorted(
        (tuple(group) for group in groups if len(group) > 1),
        key=len, reverse=True) or names


def _solve_exact(rows, targets):
    """Exact rational least-norm solve of ``rows @ x = targets``.

    Gaussian elimination over :class:`~fractions.Fraction`, so a system that is
    genuinely linear yields exact coefficients and one that is not is detected
    rather than smoothed over. Returns ``(solution, rank)`` or ``None`` when
    inconsistent.
    """
    width = len(rows[0])
    matrix = [[Fraction(value) for value in row] + [Fraction(target)]
              for row, target in zip(rows, targets)]
    pivots = []
    pivot_row = 0
    for column in range(width):
        pick = None
        for index in range(pivot_row, len(matrix)):
            if matrix[index][column] != 0:
                pick = index
                break
        if pick is None:
            continue
        matrix[pivot_row], matrix[pick] = matrix[pick], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for index in range(len(matrix)):
            if index != pivot_row and matrix[index][column] != 0:
                factor = matrix[index][column]
                matrix[index] = [
                    a - factor * b
                    for a, b in zip(matrix[index], matrix[pivot_row])
                ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    for row in matrix[pivot_row:]:
        if all(value == 0 for value in row[:width]) and row[width] != 0:
            return None, len(pivots)

    solution = [Fraction(0)] * width
    for index, column in enumerate(pivots):
        solution[column] = matrix[index][width]
    return solution, len(pivots)


def fit_units(scope, model, observations, *, strict=True):
    """Solve exact per-unit byte coefficients for each route key."""
    by_key = {}
    for observation in observations:
        by_key.setdefault(model.key_params(observation.params), []).append(
            observation)

    names = list(model.all_regressors()) + ["fixed"]
    fits = {}
    for key, samples in by_key.items():
        rows = [model.evaluate(sample.params) + [1.0] for sample in samples]
        # Alignment-free: see Observation.alloc_bytes. The padding this
        # discards is carried back as ``align_slack`` below.
        targets = [sample.alloc_bytes() for sample in samples]
        solution, rank = _solve_exact(rows, targets)
        dropped_fixed = False
        if solution is not None and rank == len(names) - 1:
            # Under-determined by exactly one degree of freedom. Try again
            # without the per-occurrence constant: some drivers are coupled by
            # construction and no corpus scene can separate them (the raster
            # projection table is always ``columns`` times the bounds table
            # whenever triangles exist, and ``columns`` is pinned by the route
            # key). Solving with the constant forced to zero is only accepted
            # if it still reproduces every measurement exactly, so this picks a
            # particular member of the solution set rather than guessing.
            reduced_rows = [row[:-1] for row in rows]
            reduced, reduced_rank = _solve_exact(reduced_rows, targets)
            if reduced is not None and reduced_rank == len(names) - 1:
                solution = list(reduced) + [Fraction(0)]
                rank = len(names)
                dropped_fixed = True
        if solution is None:
            if strict:
                raise CalibrationError(
                    f"scope {scope!r} at key {key}: no exact linear fit over "
                    f"{names} across {len(samples)} samples. The scope's size "
                    f"is not a linear function of its recorded drivers -- add "
                    f"the missing driver to the annotation, or reclassify the "
                    f"scope as a density model.")
            continue
        never_exercised = [
            name for index, name in enumerate(names)
            if name != "fixed" and all(row[index] == 0 for row in rows)
        ]
        if never_exercised and strict:
            # A driver that is zero in every sample has no measurable cost.
            # Solving anyway would assign it 0, which reads as "free" and is
            # wrong the first time production makes it non-zero.
            raise CalibrationError(
                f"scope {scope!r} at key {key}: driver(s) {never_exercised} "
                f"are zero in all {len(samples)} samples, so their cost was "
                f"never measured. Add a corpus scene that exercises them.")
        if rank < len(names) and strict:
            # Under-determined: some regressors never varied independently, so
            # the solve folds one's cost into another's coefficient. That is
            # numerically right for every sample measured and silently wrong
            # the moment production breaks the collinearity -- exactly the
            # class of bug this project exists to remove. Fail with the
            # offending pair named so the corpus can be extended.
            collinear = _collinear_regressors(rows, names)
            raise CalibrationError(
                f"scope {scope!r} at key {key}: the corpus never varies "
                f"{collinear} independently (rank {rank} < {len(names)} "
                f"unknowns), so their costs cannot be separated. Add a corpus "
                f"scene that breaks the correlation.")
        coefficients = {}
        fractional = [name for name, value in zip(names, solution)
                      if value.denominator != 1]
        if fractional:
            if strict:
                raise CalibrationError(
                    f"scope {scope!r} at key {key}: coefficient(s) "
                    f"{fractional} solved to non-integer values; byte counts "
                    f"must be whole, so the recorded drivers do not explain "
                    f"this scope's size.")
            continue
        for name, value in zip(names, solution):
            coefficients[name] = int(value)
        for row, target in zip(rows, targets):
            predicted = sum(
                coefficients[name] * value
                for name, value in zip(names, row))
            if predicted != target:
                raise CalibrationError(
                    f"scope {scope!r} at key {key}: fitted coefficients "
                    f"predict {predicted} B but {target} B was measured.")
        coefficients["align_slack"] = max(
            sample.align_slack() for sample in samples)
        # Padding is bounded by a few bytes per allocation. A large value means
        # something that is not padding has been booked as padding -- a nested
        # child scope's bytes, for instance -- and would silently inflate every
        # estimate on this route.
        worst_allocs = max(len(sample.events) for sample in samples)
        slack_ceiling = 64 + 8 * max(1, worst_allocs)
        if coefficients["align_slack"] > slack_ceiling:
            raise CalibrationError(
                f"scope {scope!r} at key {key}: alignment slack came out at "
                f"{coefficients['align_slack']} B against a ceiling of "
                f"{slack_ceiling} B for {worst_allocs} allocations. Something "
                f"other than padding is being attributed to it (a nested "
                f"scope's allocations are the usual cause).")
        if dropped_fixed:
            print(f"[calibrate] scope {scope!r} key {key}: drivers are coupled "
                  f"by construction; solved with no per-occurrence constant "
                  f"(verified exact on {len(samples)} samples).")
        fits[key] = {"coefficients": coefficients,
                     "regressors": names,
                     "samples": len(samples),
                     "rank": rank}
    return fits


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (percentile / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def fit_density(scope, model, observations):
    """Structural coefficients plus a conservative corpus density."""
    structural = fit_units(scope, model, observations, strict=False)
    densities = []
    for observation in observations:
        denominator = 1.0
        for name in model.density_per:
            denominator *= float(observation.params.get(name, 0) or 0)
        if denominator <= 0:
            continue
        numerator = float(observation.params.get(model.density_of, 0) or 0)
        densities.append(numerator / denominator)
    return {
        "structural": structural,
        "density_of": model.density_of,
        "density_per": list(model.density_per),
        "density": _percentile(densities, DENSITY_PERCENTILE),
        "density_max": max(densities) if densities else 0.0,
        "density_samples": len(densities),
    }


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------

def schema_fingerprint(tables):
    """Hash the fitted *model*, not the code that produced it.

    Invariant to line shifts, renames and moving an allocation between helper
    functions. Sensitive to adding or removing an allocation, changing a dtype
    or a shape formula, flipping ``persist``, changing ``temp`` nesting, and to
    a route key gaining a dimension. Caller qualnames are deliberately excluded
    -- they are kept beside each event for diagnostics only.
    """
    payload = []
    for scope in sorted(tables.get("traces", {})):
        entries = []
        for key, trace in sorted(
                tables["traces"][scope].items(), key=lambda kv: repr(kv[0])):
            entries.append([
                repr(key),
                [list(event) for event in trace],
            ])
        payload.append(["trace", scope, entries])
    for scope in sorted(tables.get("units", {})):
        entries = []
        for key, fit in sorted(
                tables["units"][scope].items(), key=lambda kv: repr(kv[0])):
            entries.append([repr(key), fit["regressors"],
                            [fit["coefficients"][name]
                             for name in fit["regressors"]]])
        payload.append(["units", scope, entries])
    for scope in sorted(tables.get("densities", {})):
        entry = tables["densities"][scope]
        structural = []
        for key, fit in sorted(
                entry["structural"].items(), key=lambda kv: repr(kv[0])):
            structural.append([repr(key), fit["regressors"],
                               [fit["coefficients"][name]
                                for name in fit["regressors"]]])
        payload.append([
            "density", scope, structural,
            entry["density_of"], entry["density_per"],
        ])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def environment_fingerprint():
    """Versions and startup-only settings the measured bytes depend on."""
    import torch

    from algan.rendering.raytracing.raytrace_kernels_taichi import KBUF

    try:
        import taichi

        taichi_version = taichi.__version__
    except Exception:
        taichi_version = None
    return {
        "torch": torch.__version__,
        "taichi": str(taichi_version),
        "kbuf": int(KBUF),
        "python": sys.version.split()[0],
    }


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

_HEADER = '''\
"""Measured render-memory tables. GENERATED -- do not edit by hand.

Regenerate with::

    .venv/Scripts/python.exe -m algan.utils.calibrate_memory

Every number here was measured by recording real allocations through
``ManualMemory``; none is hand-derived. ``mem_usage_runtime`` interprets them.
See ``algan/utils/calibrate_memory.py`` for what each table means.
"""
'''


def _format_key(key):
    return repr(tuple(key))


def render_module(tables, metadata):
    """Render the generated ``mem_usage`` module source."""
    lines = [_HEADER, ""]
    lines.append(f"ALGAN_VERSION = {metadata['algan_version']!r}")
    lines.append(f"GIT_COMMIT = {metadata['git_commit']!r}")
    lines.append(f"GENERATED_UTC = {metadata['generated_utc']!r}")
    lines.append(f"SCHEMA_FINGERPRINT = {metadata['fingerprint']!r}")
    lines.append(f"ENV_FINGERPRINT = {metadata['env']!r}")
    lines.append(f"CORPUS = {tuple(metadata['corpus'])!r}")
    lines.append(f"DENSITY_PERCENTILE = {DENSITY_PERCENTILE!r}")
    lines.append(f"DENSITY_SAFETY = {DENSITY_SAFETY!r}")
    lines.append("")
    lines.append("# scope -> route key -> ordered allocation trace.")
    lines.append("# Event forms: (\"A\", intercept, slope, itemsize, persist)")
    lines.append("#              (\"(\", clear_persist) / (\")\",)")
    lines.append("# An allocation's element count is intercept + slope*frames.")
    lines.append("TRACES = {")
    for scope in sorted(tables.get("traces", {})):
        lines.append(f"    {scope!r}: {{")
        for key, trace in sorted(
                tables["traces"][scope].items(), key=lambda kv: repr(kv[0])):
            lines.append(f"        {_format_key(key)}: (")
            for event in trace:
                lines.append(f"            {tuple(event)!r},")
            lines.append("        ),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("# scope -> route key -> exact bytes per unit of each driver.")
    lines.append("UNITS = {")
    for scope in sorted(tables.get("units", {})):
        lines.append(f"    {scope!r}: {{")
        for key, fit in sorted(
                tables["units"][scope].items(), key=lambda kv: repr(kv[0])):
            lines.append(f"        {_format_key(key)}: {fit['coefficients']!r},")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("# Value-dependent scopes: exact structural bytes per unit,")
    lines.append("# plus a conservative corpus density used as the in-job")
    lines.append("# learner's starting point (it is only ever raised).")
    lines.append("DENSITIES = {")
    for scope in sorted(tables.get("densities", {})):
        entry = tables["densities"][scope]
        lines.append(f"    {scope!r}: {{")
        lines.append(f"        'density_of': {entry['density_of']!r},")
        lines.append(f"        'density_per': {entry['density_per']!r},")
        lines.append(f"        'density': {entry['density']!r},")
        lines.append(f"        'density_max': {entry['density_max']!r},")
        lines.append(
            f"        'density_samples': {entry['density_samples']!r},")
        lines.append("        'structural': {")
        for key, fit in sorted(
                entry["structural"].items(), key=lambda kv: repr(kv[0])):
            lines.append(
                f"            {_format_key(key)}: {fit['coefficients']!r},")
        lines.append("        },")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("# Transient builds that run OUTSIDE the arena, in pool")
    lines.append("# headroom. 'ratio' bounds measured peak / packed input over")
    lines.append("# the corpus. Keep a margin: torch's counters cannot see")
    lines.append("# Taichi's separate CUDA pool, so this is not a hard bound.")
    lines.append(f"NONARENA = {tables.get('nonarena', {})!r}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_tables(observations, corpus):
    """Fit every scope model from a pool of observations."""
    by_scope = {}
    for observation in observations:
        by_scope.setdefault(observation.scope, []).append(observation)

    unknown = sorted(set(by_scope) - set(SCOPE_MODELS))
    if unknown:
        raise CalibrationError(
            f"unmodelled scope(s) {unknown} were recorded. A new scope must "
            f"be given a model in calibrate_memory.SCOPE_MODELS before the "
            f"table can be regenerated.")

    tables = {"traces": {}, "units": {}, "densities": {},
              "nonarena": fit_nonarena()}
    # Every scope is fitted before any failure is reported. Each failure names
    # a corpus gap that takes a full re-measurement to test a fix for, so
    # surfacing them one at a time turns a single afternoon into several.
    problems = []
    for scope, model in SCOPE_MODELS.items():
        samples = by_scope.get(scope)
        if model is None or not samples:
            continue
        try:
            if model.kind == "trace":
                tables["traces"][scope] = fit_trace(scope, model, samples)
            elif model.kind == "units":
                tables["units"][scope] = fit_units(scope, model, samples)
            else:
                tables["densities"][scope] = fit_density(scope, model, samples)
        except CalibrationError as exc:
            problems.append(str(exc))
    if problems:
        raise CalibrationError(
            f"{len(problems)} scope(s) could not be fitted:\n\n"
            + "\n\n".join(f"  * {problem}" for problem in problems))

    metadata = {
        "algan_version": _algan_version(),
        "git_commit": _git_commit(),
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": environment_fingerprint(),
        "corpus": corpus,
    }
    metadata["fingerprint"] = schema_fingerprint(tables)
    return tables, metadata


def _algan_version():
    try:
        from importlib.metadata import version

        return version("algan")
    except Exception:
        return "unknown"


def output_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rendering", "mem_usage.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true",
        help="re-measure and diff against the checked-in table instead of "
             "rewriting it")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from algan.utils.calibration_corpus import run_corpus

    observations, corpus = run_corpus()
    tables, metadata = build_tables(observations, corpus)
    source = render_module(tables, metadata)
    destination = args.out or output_path()

    if args.verify:
        try:
            from algan.rendering import mem_usage
        except ImportError:
            print("mem_usage.py is missing; run without --verify to create it.")
            return 1
        if mem_usage.SCHEMA_FINGERPRINT != metadata["fingerprint"]:
            print("STALE: mem_usage.py does not match a fresh measurement.")
            print(f"  checked in: {mem_usage.SCHEMA_FINGERPRINT}")
            print(f"  measured  : {metadata['fingerprint']}")
            return 1
        print(f"mem_usage.py is current ({metadata['fingerprint']}).")
        return 0

    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    print(f"wrote {destination}")
    print(f"  fingerprint : {metadata['fingerprint']}")
    print(f"  traces      : "
          f"{sum(len(v) for v in tables['traces'].values())} keys "
          f"over {len(tables['traces'])} scopes")
    print(f"  units       : "
          f"{sum(len(v) for v in tables['units'].values())} keys "
          f"over {len(tables['units'])} scopes")
    print(f"  densities   : {len(tables['densities'])} scopes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
