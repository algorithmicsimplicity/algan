"""Runtime model of how much render arena a chunk of frames needs.

Batch sizing used to be driven by hand-written byte formulas that mirrored the
allocation sequences in the tracer, the raster pipeline and post-processing.
Keeping them correct meant re-deriving them by hand after every change, and
when they drifted the failure surfaced as an out-of-memory error in somebody's
render.

This measures instead, and measures the only thing that actually matters: the
arena's own high-water mark. A chunk's peak is affine in its frame count --

    peak(n) = a + b * n

-- so rendering two short chunks and reading ``ManualMemory.max_pointer``
determines the whole line. Measured across the sample scenes, predictions from
n=1 and n=2 land within 0.5% at n=3, 5 and 8, and usually to the byte.

Two properties make this worth preferring to a model of the allocations:

* it needs no knowledge of *what* was allocated, so new render code -- a new
  primitive, a new tracer path, a user's own post-process -- is accounted for
  the moment it runs, with nothing to register and nothing to regenerate; and
* the frames it measures are frames the batch had to render anyway, so the
  measurement is close to free.

What it cannot do is see the future: the probe measures the first frames of a
batch, and a scene that grows denser later in that batch will exceed the line.
The out-of-memory retry therefore remains the backstop; this only makes it
rare.
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger("algan.memory_model")

# How many recent chunks inform the fit. The window exists so a single
# unusually dense chunk does not handicap the rest of the render: it raises the
# line while it is in view and ages out afterwards. Keeping a running maximum
# instead would let one heavy frame shrink every later batch for the whole job.
# Long enough to stay stable across ordinary variation, short enough that a
# transient spike is forgotten within a few chunks.
HISTORY = 8

# Multiplier on the fitted line. Alignment makes the peak *almost* affine --
# measurements land a few bytes either side -- and scene content can drift
# within a batch. Under-reserving costs a re-rendered chunk and a job-lifetime
# safety margin; over-reserving costs a slightly smaller batch. Hence a margin,
# and a deliberately asymmetric one.
DEFAULT_SAFETY = 1.15

# Margin while only one chunk has been measured. The first chunk of a job runs
# before kernel and allocator state has settled and comes in around a third
# cheaper per frame than steady state, so a line drawn through it alone
# under-reads. Widened until a second, larger chunk confirms the slope.
PROBE_SAFETY = 1.6

# Floor under the safety margin, for chunks small enough that a percentage is
# not worth having.
MINIMUM_PAD = 1 << 16

# How far a chunk may exceed the largest one already measured. The first chunk
# of a job is measurably cheaper than steady state -- kernel and allocator
# state has not settled -- so extrapolating straight from it to a full-size
# batch under-reads the per-frame cost by around a third. Growing
# geometrically instead reaches full size in a few chunks while never
# extrapolating more than this far beyond evidence.
PROBE_GROWTH = 8


class ChunkMemoryModel:
    """Affine fit of arena peak against chunk frame count.

    Lives for a render job and is carried across batches, so only the first
    batch pays for probing. Observations are kept per (route, resolution)
    signature -- a batch whose frame buffers changed shape has a different
    line.
    """

    __slots__ = ("_by_signature", "safety")

    def __init__(self, safety=DEFAULT_SAFETY):
        self._by_signature = {}
        self.safety = float(safety)

    # -- measurement -------------------------------------------------------

    def observe(self, signature, num_frames, peak_bytes):
        """Record that ``num_frames`` frames peaked at ``peak_bytes``."""
        num_frames = max(1, int(num_frames))
        peak_bytes = max(0, int(peak_bytes))
        window = self._by_signature.setdefault(signature, deque(maxlen=HISTORY))
        window.append((num_frames, peak_bytes))

    def _samples(self, signature):
        """Frame count -> worst peak seen at it, over the recent window.

        Worst *within the window* rather than ever: conservative against the
        chunks that just happened, without letting one dense stretch set the
        budget for the whole render.
        """
        window = self._by_signature.get(signature)
        if not window:
            return {}
        samples = {}
        for num_frames, peak in window:
            if peak > samples.get(num_frames, -1):
                samples[num_frames] = peak
        return samples

    @staticmethod
    def _per_frame_line(samples):
        """Conservative reading: charge the worst observed rate to every frame.

        Never under-reads any sample (``max(peak/n) * n_i >= peak_i`` for every
        i) and has no intercept to get wrong, which is what makes it the safe
        answer whenever the samples refuse to lie on one line.
        """
        return 0.0, max(peak / count for count, peak in samples.items())

    def _line(self, signature):
        """``(a, b)`` for a signature, or ``None`` while under-determined."""
        samples = self._samples(signature)
        if not samples:
            return None
        counts = sorted(samples)
        if len(counts) < 2:
            # One point fixes no slope. Charging the whole peak to every frame
            # is the conservative reading of a single sample, and is exact when
            # nothing is fixed (which is the common case: merged geometry is
            # [frames, ...], so it scales rather than sitting in the intercept).
            only = counts[0]
            return 0.0, samples[only] / only
        # The two largest chunks, not the smallest and largest: small chunks
        # carry the unsettled first-chunk cost and drawing the line through one
        # of them tilts the slope downwards, which is the unsafe direction.
        low, high = counts[-2], counts[-1]
        slope = (samples[high] - samples[low]) / (high - low)
        intercept = max(0.0, samples[low] - slope * low)
        if slope <= 0:
            # Peak did not grow with frames: the measurement is dominated by
            # something that does not scale. Fall back to the conservative
            # per-frame reading rather than planning an unbounded chunk.
            return self._per_frame_line(samples)
        if intercept > min(samples.values()):
            # The line claims a frame-independent cost larger than a chunk that
            # was measured in full -- which that chunk disproves. It happens
            # when the two largest counts come from batches of different
            # density (the signature buckets geometry, so a fit can mix them)
            # and both ran near the arena's capacity: two nearly equal peaks at
            # different frame counts define an almost flat line whose intercept
            # swallows the arena. Left alone it predicts that a *single* frame
            # needs more memory than the whole arena, which pins chunks to one
            # frame and makes the caller's preflight reject every batch it is
            # offered. The samples are simply not affine, so read them the
            # conservative way instead of trusting the fit.
            return self._per_frame_line(samples)
        return intercept, slope

    def is_calibrated(self, signature):
        """Whether anything has been measured for this signature yet.

        One sample is enough to plan from: ``_line`` reads it as a pure
        per-frame cost, which over-charges by the intercept and is therefore
        safe. Requiring two would deadlock -- planning would pin every chunk to
        one frame, and a second frame count would never be observed.
        """
        return bool(self._by_signature.get(signature))

    # -- planning ----------------------------------------------------------

    def _safety_for(self, signature):
        return self.safety if len(self._samples(signature)) >= 2 else PROBE_SAFETY

    def predict(self, signature, num_frames):
        """Bytes a chunk of ``num_frames`` is expected to need, with margin."""
        line = self._line(signature)
        if line is None:
            return None
        intercept, slope = line
        raw = intercept + slope * max(1, int(num_frames))
        return int(raw * self._safety_for(signature)) + MINIMUM_PAD

    def plan(self, signature, requested_frames, available_bytes):
        """Largest chunk expected to fit, or 1 while still probing.

        Probing deliberately renders single frames until the line is known.
        Those frames are part of the batch, so the cost is the loss of
        batching on them, not extra work.
        """
        requested_frames = max(1, int(requested_frames))
        if not self.is_calibrated(signature):
            return 1
        intercept, slope = self._line(signature)
        safety = self._safety_for(signature)
        usable = float(available_bytes) / safety - MINIMUM_PAD - intercept
        if usable <= 0 or slope <= 0:
            return 1
        planned = int(usable // slope)
        ceiling = PROBE_GROWTH * max(self._samples(signature))
        return max(1, min(requested_frames, planned, ceiling))

    def describe(self, signature):
        line = self._line(signature)
        if line is None:
            return "uncalibrated"
        intercept, slope = line
        return (
            f"{intercept / 1e6:.2f} MB + {slope / 1e6:.2f} MB/frame "
            f"from {sorted(self._samples(signature))}"
        )


class AffineFrameCost:
    """What one prepared batch costs, split into its actor and frame parts.

    A batch's cost is ``a + b * frames``. ``b * frames`` is what the frame
    count buys -- per-frame geometry, frame buffers -- and ``a`` is what the
    batch's *actor set* costs however few frames are rendered: textures,
    per-primitive metadata, kernel workspace.

    Reading the total as if it all scaled (divide the budget by the measured
    cost, multiply by the frame count) does two things wrong. It under-reads,
    because the fixed part is charged as though shrinking the window shrank it.
    And it cannot tell a batch with too many frames from one with too many
    actors -- which matters, because only the first is fixable by shortening
    the window. Batch preparation selects the actors that have spawned by the
    window's *end*, so the lever on ``a`` is the window's reach over the spawn
    schedule, not its length: a render that could not tell them apart shrank
    its window to a single frame and stayed there.

    One instance covers one prepared batch. The actor set is what fixes ``a``,
    so a fresh fetch is a fresh line; the arena preflight feeds it one
    observation per probe of that batch.
    """

    __slots__ = ("_points", "budget")

    def __init__(self):
        self._points = {}
        #: Bytes this cost is allowed, as of the most recent observation.
        self.budget = None

    def observe(self, frames, cost_bytes, budget_bytes=None):
        frames = max(1, int(frames))
        cost_bytes = max(0, int(cost_bytes))
        if cost_bytes > self._points.get(frames, -1):
            self._points[frames] = cost_bytes
        if budget_bytes is not None:
            self.budget = int(budget_bytes)

    def _line(self):
        """``(a, b)``, or ``None`` while nothing has been observed."""
        if not self._points:
            return None
        counts = sorted(self._points)
        if len(counts) < 2:
            # One point fixes no slope; charging it all per frame is the
            # conservative reading, and the one the caller had before this
            # model existed.
            only = counts[0]
            return 0.0, self._points[only] / only
        low, high = counts[-2], counts[-1]
        slope = max(0.0, (self._points[high] - self._points[low]) / (high - low))
        intercept = max(0.0, self._points[low] - slope * low)
        # A batch measured in full below the claimed fixed cost disproves it.
        return min(intercept, float(min(self._points.values()))), slope

    def fixed_bytes(self):
        """``a`` -- what this batch's actor set costs at any frame count."""
        line = self._line()
        return None if line is None else line[0]

    def actor_share(self):
        """Fraction of this batch's own cost that its actor set fixes.

        Measured against the batch's cost at the largest window observed, not
        against the budget: a budget is a prediction and can collapse (the peak
        models that supply some of them carry intercepts of their own), which
        would make every batch look actor-bound. What the caller needs to know
        is simply how much of *this* cost shortening the window can reach --
        near 1.0, almost none of it, and only a batch carrying fewer actors
        will be smaller.

        ``None`` until two frame counts have been measured; one sample cannot
        distinguish the two parts at all.
        """
        if len(self._points) < 2:
            return None
        intercept, slope = self._line()
        widest = max(self._points)
        total = intercept + slope * widest
        return None if total <= 0 else intercept / total

    def max_frames_for(self, budget_bytes=None):
        """Largest frame count expected to fit, or ``None`` if unmeasured.

        Zero means the fixed part alone overruns the budget: no window is
        short enough, and only a batch carrying fewer actors will fit.
        """
        budget = self.budget if budget_bytes is None else budget_bytes
        line = self._line()
        if line is None or budget is None:
            return None
        intercept, slope = line
        usable = float(budget) - intercept
        if usable <= 0:
            return 0
        if slope <= 0:
            # Nothing observed scales with the frame count; the budget does not
            # bound the window.
            return None
        return max(0, int(usable // slope))

    def describe(self):
        line = self._line()
        if line is None:
            return "unmeasured"
        intercept, slope = line
        return (
            f"{intercept / 1e6:.1f} MB actors + {slope / 1e6:.2f} MB/frame "
            f"from {sorted(self._points)}"
        )


class PeakRatioModel:
    """Measured bound on an out-of-arena transient build's peak.

    The scene merge and the vertex projection build out of place in pool
    headroom, before and outside the render arena, so the arena's high-water
    mark cannot see them and :class:`ChunkMemoryModel` does not cover them.
    Their peak does scale with the packed inputs they read, though, and that
    size *is* known before the build runs -- so the same measure-and-reuse
    approach applies, against input bytes instead of frame counts.

    The bound used to be a pure multiplier: worst recently observed
    ``peak / inputs``, seeded by a guess (6.0 for the merge, 8.0 for the
    projection). A ratio has no intercept, and these builds have a large one --
    kernel workspaces and allocator growth that a small build pays in full. A
    job's first merge is typically its smallest, so it measured ratios above
    20x, and every batch for the next :data:`HISTORY` builds was then throttled
    to a twentieth of the headroom it actually had. Hence the affine reading:
    the fixed part is charged once instead of to every byte.

    Under-reserving here is recovered -- the caller catches the build's
    out-of-memory and shrinks the window -- while over-reserving silently costs
    batch size for the rest of the job, so the fit deliberately leans on
    measurement rather than on worst-case extrapolation. The out-of-memory
    handler stays the backstop either way: torch's counters cannot see Taichi's
    separate pool, so no measurement here is a hard bound.
    """

    __slots__ = ("_samples", "seed", "safety")

    def __init__(self, seed, safety=1.25):
        self.seed = float(seed)
        self.safety = float(safety)
        self._samples = deque(maxlen=HISTORY)

    def observe(self, input_bytes, peak_bytes):
        input_bytes = int(input_bytes)
        peak_bytes = int(peak_bytes)
        if input_bytes > 0 and peak_bytes >= 0:
            self._samples.append((input_bytes, peak_bytes))

    def _points(self):
        """Input bytes -> worst peak seen at them, over the recent window."""
        points = {}
        for input_bytes, peak in self._samples:
            if peak > points.get(input_bytes, -1):
                points[input_bytes] = peak
        return points

    def _line(self):
        """``(a, b)`` with ``peak(n) = a + b * n``, or ``None`` if unmeasured."""
        points = self._points()
        if not points:
            return None
        sizes = sorted(points)
        if len(sizes) < 2:
            # One point fixes no slope. Charge the whole peak per byte, which
            # is the conservative reading of a single sample (and the only
            # reading available).
            only = sizes[0]
            return 0.0, points[only] / only
        # The two largest builds: small ones are dominated by the fixed part,
        # and drawing the rate through one of them is what produced the 20x
        # multipliers this model exists to avoid.
        low, high = sizes[-2], sizes[-1]
        slope = (points[high] - points[low]) / (high - low)
        if slope <= 0:
            # The peak did not grow with the inputs, so it is dominated by
            # something that does not scale with them. Read the largest build's
            # own rate per byte rather than letting a flat pair set no rate at
            # all.
            slope = points[high] / high
        intercept = max(0.0, points[low] - slope * low)
        # A build that peaked *below* the claimed fixed cost disproves it.
        # Without this an intercept fitted from two heavy builds can exceed the
        # whole headroom, and then every window is rejected however small --
        # right down to the single frame the caller renders on an estimate's
        # abstention. (Same guard as ChunkMemoryModel's, and the same failure.)
        return min(intercept, float(min(points.values()))), slope

    def predict(self, input_bytes):
        """Bytes a build reading ``input_bytes`` is expected to peak at."""
        input_bytes = max(0, int(input_bytes))
        line = self._line()
        if line is None:
            return int(self.seed * input_bytes)
        intercept, slope = line
        affine = (intercept + slope * input_bytes) * self.safety
        # The affine reading may only improve on the pure-ratio one, never
        # inflate it. The two fail in opposite directions: a ratio charges a
        # small build's fixed cost to every byte (20x multipliers, tiny
        # batches), while an intercept fitted from a run of large builds can
        # approach the whole headroom and reject every window however short --
        # both of which have pinned a render to single frames. Taking the
        # tighter is bounded by whichever is behaving.
        ratio = self._worst_ratio() * self.safety * input_bytes
        # Never below the inputs themselves: a build cannot peak at less than
        # what it has already materialised.
        return int(max(input_bytes, min(affine, ratio)))

    def fixed_for_test(self):
        """The fitted fixed part, for tests that pin the fit's shape."""
        line = self._line()
        return None if line is None else line[0]

    def _worst_ratio(self):
        points = self._points()
        return max(peak / size for size, peak in points.items()) if points else self.seed

    def max_inputs_for(self, budget_bytes):
        """Largest input size whose predicted peak fits ``budget_bytes``.

        The caller sizes frame windows, and input bytes scale with frames while
        the fixed part does not -- so a window cannot be scaled by the
        *prediction*, only by the part of it the window controls. Reading the
        budget back through the line is what separates the two.
        """
        budget = float(budget_bytes)
        line = self._line()
        if line is None:
            return budget / self.seed if self.seed > 0 else 0.0
        intercept, slope = line
        # The ratio reading bounds the affine one (see predict), so it also
        # sets a floor under the inputs the budget allows -- without it an
        # intercept approaching the budget reports that nothing fits.
        ratio = self._worst_ratio() * self.safety
        by_ratio = budget / ratio if ratio > 0 else 0.0
        usable = budget / self.safety - intercept
        by_line = usable / slope if (usable > 0 and slope > 0) else 0.0
        return max(by_line, by_ratio)

    def is_calibrated(self):
        return bool(self._samples)

    def describe(self):
        line = self._line()
        if line is None:
            return f"seed {self.seed:.1f}x (unmeasured)"
        intercept, slope = line
        return (
            f"{intercept / 1e6:.1f} MB + {slope:.2f}x inputs "
            f"from {len(self._samples)} builds"
        )


def chunk_signature(
    *, width, height, channels, dtype, samples_per_pixel, num_triangles, num_circuits
):
    """Key identifying batches whose peak lies on the same line.

    Resolution and buffer dtype change the per-frame cost; the primitive counts
    change it too, and a batch with different geometry is a different line.
    Geometry counts are bucketed logarithmically so ordinary scene variation
    does not discard a usable fit, while an order-of-magnitude change starts a
    fresh one.
    """

    def bucket(value):
        value = int(value or 0)
        return value.bit_length()

    return (
        int(width),
        int(height),
        int(channels),
        str(dtype),
        int(samples_per_pixel),
        bucket(num_triangles),
        bucket(num_circuits),
    )
