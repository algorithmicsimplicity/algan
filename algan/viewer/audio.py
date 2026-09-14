"""Opening-time audio snapshot and lazy, in-memory PCM for the browser."""

from __future__ import annotations

import copy
import io
import math
import threading
import wave

import numpy as np


class SceneAudio:
    """Mix existing clips, never author speech or close Scene-owned readers.

    Resolve lazy starts before the viewer's render worker can materialize the
    timeline. Sampling uses a separate lock: a slow render must not delay audio,
    and simultaneous browser requests must not race MoviePy's file readers.
    """

    def __init__(self, effects, duration, sample_rate):
        self.duration = float(duration)
        self.sample_rate = int(sample_rate)
        self._clips = []
        for effect in effects:
            start = float(effect.start_time_func())
            clip = effect.audio_clip
            end = start + float(clip.duration)
            if math.isfinite(start) and start < self.duration and end > max(0, start):
                self._clips.append((copy.copy(clip), start, end))
        self.available = bool(self._clips)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._wav = None

    def wav(self):
        """One cached WAV, or None for a silent scene; no temporary files."""
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("The viewer audio is closed")
            if not self.available:
                return None
            if self._wav is None:
                self._wav = self._encode()
            return self._wav

    def _encode(self):
        if self.sample_rate <= 0 or not math.isfinite(self.duration):
            raise ValueError(
                "Viewer audio needs a positive sample rate and finite duration"
            )
        channels = max(int(clip.nchannels) for clip, _, _ in self._clips)
        count = math.ceil(self.duration * self.sample_rate)
        if count * channels * 2 > 0xFFFFFFFF - 36:
            raise ValueError("The scene audio is too long for an in-memory WAV")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            # Bound the working arrays even for a long scene. In particular,
            # normalise mono to (samples, 1) before broadcasting it to stereo;
            # multiplying a 1-D clip by a column mask creates a quadratic array.
            for first in range(0, count, 16384):
                if self._closed.is_set():
                    raise RuntimeError("The viewer audio is closed")
                times = np.arange(first, min(first + 16384, count)) / self.sample_rate
                mixed = np.zeros((len(times), channels), dtype=np.float64)
                for clip, start, end in self._clips:
                    active = (times >= start) & (times < end)
                    if not active.any():
                        continue
                    samples = np.asarray(clip.get_frame(times[active] - start))
                    if samples.ndim == 1 and clip.nchannels == 1:
                        samples = samples[:, None]
                    mixed[active] += samples
                np.nan_to_num(mixed, copy=False)
                np.clip(mixed, -1, 1 - 1 / 32768, out=mixed)
                wav.writeframesraw((mixed * 32768).astype("<i2").tobytes())
        return output.getvalue()

    def close(self):
        """Cancel mixing at its next chunk, then release only our references."""
        self._closed.set()
        with self._lock:
            self._clips.clear()
            self._wav = None
