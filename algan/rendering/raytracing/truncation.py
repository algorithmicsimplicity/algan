"""Counters for the render path's four silent truncations.

Each of these is a fixed ceiling that degrades the *image* when it binds --
transport that should have reached the pixel does not -- and every one of them
used to pass without a word.  ``DESIGN_mesh_identity_open.md`` §Y states the
rule they broke: *an instrument that reports zero may not be looking*.  So the
counters here are unconditional; a zero is a measurement, not an absence of
one.

The ceilings, and what each costs when it binds:

``surfaces_per_ray``
    ``MAX_SURFACES_PER_RAY`` (256) surfaces composited along one primary ray.
    The walk stops and the ray's leftover weight is handed to the background,
    so a deep translucent stack goes see-through at the 257th layer.
``shadow_lights``
    ``MAX_SHADOW_LIGHTS`` (16, ``ALGAN_MAX_SHADOW_LIGHTS``) shadowed lights per
    fragment.  A compile-time vector length, so the surplus lights are still
    *lit* -- they simply never cast.  Each :class:`~.RectAreaLight` emitter
    sample spends one slot, so a 4x4 area light fills the default cap alone.
``sheet_layers``
    16 overlapping layers of one surface in one pixel (the conflict rank the
    sheet compaction packs into the sheet key).  Layers past the 16th merge
    into the last sub-band and attenuate once between them instead of once
    each, so a self-overlapping morph renders too light.
``dropped_continuations``
    A reflection/refraction continuation that could not reserve a slot in the
    tile's shared ray pool.  A *splitting* batch (``pool_ratio > 1``) discards
    and retries the tile, so it never loses one; a batch at ``pool_ratio == 1``
    has no spare slots at all and the reservation simply fails, dropping that
    branch's contribution.

Reporting.  Truncation is a correctness event, not a budgeting one, so the
first occurrence of each ceiling in a render is a ``WARNING`` -- unlike the
wavefront's pool retries and batch splits, which are the memory model working
as designed and log at :data:`~algan.logging.logger.PERF`.  Later batches of
the same render log their counts at ``PERF`` instead of repeating the warning,
because a scene that truncates one frame truncates all of them and a warning
per batch would be noise.  The running totals ride on
:class:`~algan.rendering.raytracing.tracer.RenderPlan` either way, so a script
can assert on them without parsing logs.

Scope: one *render job*.  :meth:`_TruncationRecorder.reset` is called by
``RenderLoopMixin.get_frames``, which is the boundary of a
``save_video`` / ``save_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from algan.logging.logger import PERF, get_logger

logger = get_logger("raytracing")


@dataclass(frozen=True)
class TruncationCounts:
    """How many times each render ceiling bound during a render.

    Attached to :class:`~algan.rendering.raytracing.tracer.RenderPlan` as its
    ``truncations`` field, and therefore reachable from a finished render as
    ``RenderResult.render_plan.truncations``.  Every count is cumulative over
    the render job, so the plan of the *last* batch carries the whole render's
    totals.

    A count of zero means the ceiling was watched and never bound.
    """

    #: Primary rays that composited ``MAX_SURFACES_PER_RAY`` surfaces and
    #: stopped with transport left to carry.
    surfaces_per_ray: int = 0
    #: Light slots past ``MAX_SHADOW_LIGHTS`` in the worst batch of the render.
    #: Those lights are lit but cast no shadow. The one counter here that is a
    #: *state* rather than a tally of events -- the same surplus lights are
    #: over the cap in every batch they are spawned for -- so it is reduced
    #: with a maximum and reads as "this many lights went unshadowed".
    shadow_lights: int = 0
    #: Fragments that were the 17th or later layer of their own surface in one
    #: pixel, and so merged into the 16th sub-band.
    sheet_layers: int = 0
    #: Reflection/refraction continuation rays that could not reserve a pool
    #: slot and were dropped.
    dropped_continuations: int = 0

    @property
    def total(self) -> int:
        """Sum of every counter -- zero exactly when nothing was truncated."""
        return sum(getattr(self, f.name) for f in fields(self))

    def __bool__(self) -> bool:
        return self.total > 0

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


#: Per-ceiling warning text. ``{count}`` is the running total for the render.
#: Written to say what moved in the image and what to do about it, because the
#: reader of a WARNING has not read this module.
_CEILING_MESSAGES = {
    "surfaces_per_ray": (
        "{count} primary ray(s) hit the {cap}-surface compositing ceiling "
        "(MAX_SURFACES_PER_RAY) and stopped early: the background shows "
        "through the rest of the stack. Reduce the number of overlapping "
        "transparent surfaces along the view direction."
    ),
    "shadow_lights": (
        "{count} light slot(s) past the shadow cap of {cap} are lit but cast "
        "no shadow (each RectAreaLight emitter sample spends one slot). Raise "
        "ALGAN_MAX_SHADOW_LIGHTS before 'import algan', or use fewer / "
        "coarser-sampled lights."
    ),
    "sheet_layers": (
        "{count} fragment(s) overlapped their own surface more than {cap} "
        "times within one pixel; the surplus layers merged into the last and "
        "attenuate once between them instead of once each, so that region "
        "renders too light."
    ),
    "dropped_continuations": (
        "{count} reflection/refraction continuation ray(s) could not reserve "
        "a slot in the tile's ray pool and were dropped, so those branches "
        "contribute nothing to the frame."
    ),
}

_CEILING_NAMES = tuple(f.name for f in fields(TruncationCounts))

#: How each ceiling combines across the batches of one render. Most are tallies
#: of independent events and add up; ``shadow_lights`` counts the light slots a
#: batch could not shadow, which is the same surplus in every batch that
#: carries those lights, so adding would report "15 unshadowed lights" for a
#: three-batch render of five. It takes the worst batch instead.
_CEILING_REDUCERS = dict.fromkeys(_CEILING_NAMES, int.__add__)
_CEILING_REDUCERS["shadow_lights"] = max


class _TruncationRecorder:
    """Render-job-scoped accumulator behind the module-level API below.

    Deliberately a plain object with plain ints: it is written from the render
    loop's inner iterations, and anything that had to be read back from the
    device or locked would cost more than the ceilings it watches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new render job: zero the counts and re-arm every warning."""
        self._counts = dict.fromkeys(_CEILING_NAMES, 0)
        self._reported = dict.fromkeys(_CEILING_NAMES, 0)
        self._caps = {}
        self._warned = set()

    def record(self, ceiling, count, cap=None):
        """Add ``count`` occurrences of ``ceiling``.

        ``cap`` is the ceiling's value at the time, carried only so the warning
        can name it (``MAX_SHADOW_LIGHTS`` is an env-var knob, so it is not a
        constant this module may bake in).
        """
        count = int(count)
        if count <= 0:
            return
        self._counts[ceiling] = _CEILING_REDUCERS[ceiling](self._counts[ceiling], count)
        if cap is not None:
            self._caps[ceiling] = int(cap)

    def snapshot(self) -> TruncationCounts:
        """The running totals, as the immutable value the plan carries."""
        return TruncationCounts(**self._counts)

    def restore(self, counts: TruncationCounts):
        """Roll the counts back to ``counts``.

        The render loop halves a chunk and re-renders it after an out-of-memory
        failure; the discarded attempt's counters are rolled back with the
        arena pointers so a retry does not double-count the same frames.
        Warnings already emitted stay emitted -- the truncation they reported
        did happen -- but the rolled-back counts are re-reportable, so the
        re-render's real totals are not mistaken for "already said that".
        """
        self._counts = dict(counts.as_dict())
        for ceiling in _CEILING_NAMES:
            self._reported[ceiling] = min(
                self._reported[ceiling], self._counts[ceiling]
            )

    def report(self):
        """Emit whatever the counts have earned since the last report.

        The batch that first trips a ceiling warns; a later batch that pushes
        the same count higher logs the new total at ``PERF``, and a batch that
        adds nothing to it says nothing at all -- otherwise a hundred-batch
        render would repeat itself a hundred times over one scene's single
        defect.  Returns the snapshot, so the caller can put the totals on the
        batch's ``RenderPlan`` in the same step.
        """
        for ceiling in _CEILING_NAMES:
            count = self._counts[ceiling]
            if count <= 0 or count == self._reported[ceiling]:
                continue
            self._reported[ceiling] = count
            message = _CEILING_MESSAGES[ceiling].format(
                count=count, cap=self._caps.get(ceiling, "?")
            )
            if ceiling in self._warned:
                logger.log(PERF, message)
            else:
                self._warned.add(ceiling)
                logger.warning(message)
        return self.snapshot()


_recorder = _TruncationRecorder()


def reset_truncations():
    """Begin a render job's truncation accounting.

    Called by ``RenderLoopMixin.get_frames``, so every ``save_video`` /
    ``save_frame`` reports its own render rather than inheriting the last one's
    counts and its already-spent warnings.
    """
    _recorder.reset()


def record_truncation(ceiling, count, cap=None):
    """Count ``count`` occurrences of one ceiling binding."""
    _recorder.record(ceiling, count, cap)


def snapshot_truncations() -> TruncationCounts:
    """The render job's truncation counts so far."""
    return _recorder.snapshot()


def restore_truncations(counts: TruncationCounts):
    """Roll the counts back to a snapshot (used by the OOM chunk retry)."""
    _recorder.restore(counts)


def report_truncations() -> TruncationCounts:
    """Log this batch's truncations and return the running totals."""
    return _recorder.report()


def attach_truncations(plan):
    """Return ``plan`` carrying the render job's truncation counts.

    ``RenderPlan`` is frozen and is built before the batch renders, so the
    counts are grafted on afterwards rather than mutated in place.
    """
    return replace(plan, truncations=snapshot_truncations())
