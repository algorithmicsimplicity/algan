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


class PeakRatioModel:
    """Measured multiplier bounding an out-of-arena transient build.

    The scene merge and the vertex projection build out of place in pool
    headroom, before and outside the render arena, so the arena's high-water
    mark cannot see them and :class:`ChunkMemoryModel` does not cover them.
    Their peak does scale with the packed inputs they read, though, and that
    size *is* known before the build runs -- so the same measure-and-reuse
    approach applies, with the ratio taking the place of the fitted line.

    These multipliers used to be guesses (6.0 for the merge, 8.0 for the
    projection). The guess now seeds the model and is superseded as soon as a
    real build has been observed. As with the chunk model, the window keeps one
    heavy batch from setting the budget for the whole render, and the
    out-of-memory handler stays the backstop -- torch's counters cannot see
    Taichi's separate pool, so no measurement here is a hard bound.
    """

    __slots__ = ("_ratios", "seed", "safety")

    def __init__(self, seed, safety=1.25):
        self.seed = float(seed)
        self.safety = float(safety)
        self._ratios = deque(maxlen=HISTORY)

    def observe(self, input_bytes, peak_bytes):
        input_bytes = int(input_bytes)
        peak_bytes = int(peak_bytes)
        if input_bytes > 0 and peak_bytes >= 0:
            self._ratios.append(peak_bytes / input_bytes)

    def factor(self):
        """Multiplier to apply to packed input bytes."""
        if not self._ratios:
            return self.seed
        # Worst recently observed, with margin. Never below 1: a build cannot
        # peak at less than the inputs it has already materialised.
        return max(1.0, max(self._ratios) * self.safety)

    def is_calibrated(self):
        return bool(self._ratios)

    def describe(self):
        if not self._ratios:
            return f"seed {self.seed:.1f}x (unmeasured)"
        return (
            f"{self.factor():.2f}x from {len(self._ratios)} builds "
            f"(worst {max(self._ratios):.2f}x)"
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
