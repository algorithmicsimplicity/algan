from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import pyttsx3

from algan.errors import AlganConfigurationError, AudioTranscriptMismatchError
from algan.settings import SETTINGS


class Timer:
    def __init__(self):
        self.time = 0


import re

import torch

from algan.logging.logger import get_logger

logger = get_logger("audio")
pattern = re.compile(r"[\W_]+", re.UNICODE)


# --- Configuration ---
# A larger chunk size is more efficient with the optimized torchaudio function.
CHUNK_DURATION_S = 1 * 60  # 1 minute
MODEL_ID = "facebook/wav2vec2-base-960h"
# Bump when the alignment algorithm or on-disk timestamp format changes.
_ALIGNMENT_CACHE_VERSION = 2


class Counter:
    def __init__(self):
        self.count = 0


def strip_nonchars(x):
    return pattern.sub("", x).upper()


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for length in lengths:
        ret.append(list_[i : i + length])
        i += length
    return ret


def align_large_audio_torchaudio_robust(
    audio_path, transcript_path, model_id=MODEL_ID, chunk_duration_s=CHUNK_DURATION_S
):
    """Internal: align recorded speech, failing if a chunk cannot advance."""
    if not math.isfinite(chunk_duration_s) or chunk_duration_s <= 0:
        raise AlganConfigurationError("chunk_duration_s must be finite and positive")
    try:
        import torchaudio
        import torchaudio.functional as F
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError as e:
        raise ImportError(
            "Synchronizing to a recorded speech audio file requires the optional"
            " audio dependencies (torchaudio and transformers). Install them with:"
            " pip install algan[audio]"
        ) from e

    logger.info("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    processor.tokenizer.encoder["*"] = len(processor.tokenizer.encoder)
    processor.tokenizer.decoder[processor.tokenizer.encoder["*"]] = "*"
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    logger.info("Loading audio file info...")
    audio_info = torchaudio.info(audio_path)
    audio_duration_s = audio_info.num_frames / audio_info.sample_rate

    with open(transcript_path, encoding="utf-8-sig") as f:
        full_transcript_text = f.read().upper()
        full_transcript_text = full_transcript_text.replace("-", " ")
        full_transcript_words = full_transcript_text.split()
        full_transcript_text = " ".join(full_transcript_words)

    if not full_transcript_words:
        return []

    total_chunks = int(
        torch.ceil(torch.tensor(audio_duration_s / chunk_duration_s)).item()
    )

    estimated_words_per_minute = 120
    estimated_words_per_chunk = estimated_words_per_minute * chunk_duration_s / 60
    logger.info(
        f"Audio duration: {audio_duration_s:.2f}s. Will be processed in {total_chunks} chunks."
    )

    all_word_segments = []
    transcript_cursor = 0

    chunk_start_s = 0
    final_chunk = False
    while chunk_start_s < audio_duration_s:
        chunk_end_s = min(chunk_start_s + chunk_duration_s, audio_duration_s)
        if chunk_start_s + chunk_duration_s >= audio_duration_s - 10:
            final_chunk = True

        logger.info(
            f"\n--- Processing Chunk {chunk_start_s}/{audio_duration_s} ({chunk_start_s:.2f}s to {chunk_end_s:.2f}s) ---"
        )

        logger.info("Loading audio chunk...")
        frame_offset = int(chunk_start_s * audio_info.sample_rate)
        num_frames = int((chunk_end_s - chunk_start_s) * audio_info.sample_rate)
        waveform, sr = torchaudio.load(
            audio_path, frame_offset=frame_offset, num_frames=num_frames
        )
        if sr != processor.feature_extractor.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=processor.feature_extractor.sampling_rate
            )(waveform)

        input_values = processor(
            waveform,
            return_tensors="pt",
            sampling_rate=processor.feature_extractor.sampling_rate,
        ).input_values.squeeze(0)

        logger.info("Running model encoder...")
        with torch.no_grad():
            logits = model(input_values.to(device)).logits[0]
            emissions = torch.log_softmax(logits, dim=-1)
            emissions = torch.cat((emissions, torch.zeros_like(emissions[..., :1])), -1)

        # Each audio window owns its result. In particular, exhausting retries
        # must never reuse the last successful window's word timestamps.
        chunk_word_segments = None
        max_retries = 10
        for _attempt in range(max_retries):
            remaining_transcript = full_transcript_words[transcript_cursor:]

            estimated_len = max(int(estimated_words_per_chunk), 3)

            transcript_words = (
                remaining_transcript[:estimated_len]
                if not final_chunk
                else remaining_transcript
            )
            transcript_segment_text = " ".join(transcript_words)
            pad = "*" if not final_chunk else ""

            tokenized_transcript = processor.tokenizer(
                strip_nonchars(transcript_segment_text) + pad
            ).input_ids

            # --- Perform the optimized alignment ---
            blank_id = processor.tokenizer.pad_token_id

            aligned_tokens, scores = torchaudio.functional.forced_align(
                emissions.unsqueeze(0),
                torch.tensor([tokenized_transcript], dtype=torch.int32, device=device),
                blank=blank_id,
            )
            token_spans = F.merge_tokens(aligned_tokens[0], scores[0])

            time_per_frame = 1

            word_spans = unflatten(
                token_spans,
                [len(strip_nonchars(word)) for word in transcript_words]
                + ([len(pad)] if len(pad) > 0 else []),
            )
            if not word_spans or any(not spans for spans in word_spans):
                raise AudioTranscriptMismatchError(
                    f"No alignable words in the audio window "
                    f"{chunk_start_s:.2f}s to {chunk_end_s:.2f}s of {audio_path!r}. "
                    "Check that the transcript contains words matching the recording."
                )
            transcript_words_star = transcript_words + ([pad] if len(pad) > 0 else [])
            word_spans = [
                [
                    transcript_words_star[i],
                    word_spans[i][0].start * time_per_frame,
                    word_spans[i][-1].end * time_per_frame,
                ]
                for i in range(len(word_spans))
            ]
            # Move results to CPU for analysis
            if word_spans[-1][1] >= emissions.shape[0] - 10 and not final_chunk:
                estimated_words_per_chunk *= 0.75
                logger.info(
                    "Transcript chunk was too long for audio chunk, retrying with shorter transcript chunk."
                )
                continue

            time_per_frame = (chunk_end_s - chunk_start_s) / emissions.shape[0]
            chunk_word_segments = [
                [
                    strip_nonchars(_[0]),
                    chunk_start_s + _[1] * time_per_frame,
                    chunk_start_s + _[2] * time_per_frame,
                ]
                for _ in (word_spans[:-1] if not final_chunk else word_spans)
            ]
            break

        if chunk_word_segments is None:
            raise AudioTranscriptMismatchError(
                f"Could not align the audio window {chunk_start_s:.2f}s to "
                f"{chunk_end_s:.2f}s of {audio_path!r} after {max_retries} attempts "
                f"(transcript word {transcript_cursor + 1}). "
                "Check that the transcript matches the recording."
            )

        confirmed_end = (
            chunk_word_segments[-1][-1] if chunk_word_segments else chunk_start_s
        )
        next_chunk_start = confirmed_end + time_per_frame * 0.5
        if (
            not math.isfinite(confirmed_end)
            or confirmed_end <= chunk_start_s
            or not math.isfinite(next_chunk_start)
            or next_chunk_start <= chunk_start_s
            or int(next_chunk_start * audio_info.sample_rate) <= frame_offset
        ):
            raise AudioTranscriptMismatchError(
                f"Speech alignment made no forward progress in the audio window "
                f"{chunk_start_s:.2f}s to {chunk_end_s:.2f}s of {audio_path!r} "
                f"(transcript word {transcript_cursor + 1}). "
                "Check that the transcript matches the recording."
            )

        estimated_words_per_chunk = (
            len(chunk_word_segments)
            * ((chunk_end_s - chunk_start_s) / (confirmed_end - chunk_start_s))
            * 0.9
        )

        all_word_segments.extend(chunk_word_segments)
        chunk_start_s = next_chunk_start

        confirmed_transcript_len = len(chunk_word_segments)
        transcript_cursor += confirmed_transcript_len
        if transcript_cursor >= len(full_transcript_words):
            break

    logger.info("\nAlignment process complete.")
    return all_word_segments


def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i : i + len(pattern)] == pattern:
            return i
    return -1


def _alignment_cache_key(audio_path, transcript_path):
    """Fingerprint the resolved sources, their contents, and alignment settings."""
    transcript = transcript_path.read_bytes()
    identity = {
        "version": _ALIGNMENT_CACHE_VERSION,
        "model": MODEL_ID,
        "chunk_duration_s": CHUNK_DURATION_S,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "transcript_sha256": hashlib.sha256(transcript).hexdigest(),
    }
    hasher = hashlib.sha256()
    hasher.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    with audio_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    words = [
        strip_nonchars(word)
        for word in transcript.decode("utf-8-sig").replace("-", " ").split()
    ]
    return hasher.hexdigest(), [word for word in words if word]


def _validated_alignment(rows, words, duration):
    """Return a complete, ordered alignment, or None for an unusable entry."""
    if not isinstance(rows, (list, tuple)) or len(rows) != len(words):
        return None
    result = []
    previous_end = 0.0
    for row, expected in zip(rows, words):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        word, start, end = row
        if (
            word != expected
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or isinstance(start, bool)
            or isinstance(end, bool)
        ):
            return None
        try:
            start, end = float(start), float(end)
        except OverflowError:
            return None
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end
            or end <= start
            # MoviePy's probed duration can be rounded while torchaudio counts
            # samples. Allow its last 50 ms, then clip to the actual reader.
            or end > duration + 0.05
            or start >= duration
        ):
            return None
        result.append([word, start, min(end, duration)])
        previous_end = end
    return result


def _write_alignment_cache(path, rows):
    """Publish a whole cache entry atomically; never trust a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as target:
            temporary = Path(target.name)
            json.dump(
                {"version": _ALIGNMENT_CACHE_VERSION, "words": rows},
                target,
                allow_nan=False,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def get_speech_generator_from_file(audio_file, transcript_file):
    from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

    audio_path = Path(audio_file).resolve(strict=True)
    transcript_path = Path(transcript_file).resolve(strict=True)
    cache_key, words = _alignment_cache_key(audio_path, transcript_path)
    time_stamp_file = (
        Path(SETTINGS.paths.cache_directory) / "audio" / f"{cache_key}.json"
    )
    full_ac = AudioFileClip(str(audio_path))
    try:
        word_time_stamps = None
        try:
            cached = json.loads(time_stamp_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("version") == _ALIGNMENT_CACHE_VERSION
            ):
                word_time_stamps = _validated_alignment(
                    cached.get("words"), words, full_ac.duration
                )
        except (OSError, ValueError):
            pass  # A missing, truncated or malformed cache is a cache miss.
        if word_time_stamps is None:
            aligned = align_large_audio_torchaudio_robust(
                str(audio_path),
                str(transcript_path),
                model_id=MODEL_ID,
                chunk_duration_s=CHUNK_DURATION_S,
            )
            word_time_stamps = _validated_alignment(aligned, words, full_ac.duration)
            if word_time_stamps is None:
                raise AudioTranscriptMismatchError(
                    f"Speech alignment for {str(audio_path)!r} did not produce "
                    "complete, ordered timestamps matching the transcript."
                )
            if _alignment_cache_key(audio_path, transcript_path)[0] != cache_key:
                raise AudioTranscriptMismatchError(
                    "The audio or transcript changed during speech alignment; retry "
                    "with the finished recording and transcript."
                )
            _write_alignment_cache(time_stamp_file, word_time_stamps)
    except BaseException:
        with suppress(Exception):
            full_ac.close()
        raise

    def generator(script):
        script = script.replace("-", " ")
        script_words = [strip_nonchars(_) for _ in script.split()]
        script_words = [_ for _ in script_words if len(_) > 0]

        script_start_ind = (
            subfinder([_[0] for _ in word_time_stamps], script_words)
            if script_words
            else -1
        )

        if (
            script_start_ind < 0
        ):  # script_words != [_[0] for _ in word_time_stamps[word_counter.count:word_counter.count+len(script_words)]]:
            # raise AudioTranscriptMismatchError(f'Error, the following text was not found in the recorded '
            #                                   f'transcript of the speech audio file:\n\n{script_words}')
            logger.warning(
                f"Warning: the following text was not found in the recorded transcript of the speech"
                f" audio file, and so this speech will be machine generated:\n\n{script_words}"
            )
            return get_pyttsx_speech_generator(script)

        audio_start = word_time_stamps[script_start_ind][1]
        audio_end = word_time_stamps[script_start_ind + len(script_words) - 1][2]
        if script_start_ind + len(script_words) - 1 < len(word_time_stamps) - 1:
            dif = word_time_stamps[script_start_ind + len(script_words)][1] - audio_end
            audio_end += min(dif * 0.5, 0.5)

        clip_start = max(audio_start - 0.05, 0)
        sub_ac = full_ac.subclipped(clip_start, min(audio_end + 0.05, full_ac.duration))
        # Keep alignment relative to the padded subclip, not the source file.
        # Speech later adds the effect's resolved Scene offset for the viewer.
        sub_ac.algan_word_timestamps = tuple(
            (word, start - clip_start, end - clip_start)
            for word, start, end in word_time_stamps[
                script_start_ind : script_start_ind + len(script_words)
            ]
        )
        return sub_ac

    return generator


def get_pyttsx_speech_generator(script):
    hasher = hashlib.sha256()
    hasher.update(script.encode())
    hash_bytes = hasher.hexdigest()[:32]
    file = os.path.join(SETTINGS.paths.cache_directory, "audio", f"{hash_bytes}.mp3")
    if not os.path.exists(file):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.save_to_file(script, file)
        engine.runAndWait()
        engine.stop()
    return _cached_audio_file_clip(file)


# Bump when the sidecar's fields or their meaning change.
_AUDIO_INFO_VERSION = 1


def _cached_audio_file_clip(file):
    """An ``AudioFileClip`` of ``file`` that opens no decoder until it is played.

    Authoring a Speech block needs only the clip's duration, but constructing an
    ``AudioFileClip`` runs ``ffmpeg -i`` to probe the file and then starts a
    second ffmpeg process and reads its first buffer: about 0.15 s per block,
    and a narrated video has one block per sentence. The probe's answer is
    stored in a sidecar next to the file, keyed by the file's size and
    modification time, and the returned clip starts its reader on first use --
    in practice when ``Scene.save_video`` mixes the audio track. Validating,
    screenshots and ``Scene.view`` then never decode the narration at all.
    """
    from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

    info_path = Path(f"{file}.info.json")
    try:
        stat = os.stat(file)
        key = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        return AudioFileClip(file)
    with suppress(Exception):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("version") == _AUDIO_INFO_VERSION and info.get("key") == key:
            return _lazy_audio_file_clip_class()(file, info)
    clip = AudioFileClip(file)
    info = {
        "version": _AUDIO_INFO_VERSION,
        "key": key,
        "duration": clip.duration,
        "nchannels": clip.nchannels,
        "buffersize": clip.buffersize,
        "fps": clip.fps,
    }
    temporary = None
    with suppress(OSError):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=info_path.parent,
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(info, target)
        os.replace(temporary, info_path)
        temporary = None
    if temporary is not None:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return clip


_LAZY_AUDIO_FILE_CLIP = None

#: moviepy releases whose ``FFMPEG_AudioReader.__init__`` sets exactly the
#: attributes :func:`_probed_audio_reader` reproduces. Any other release is
#: constructed normally; ``tests/unit_tests/test_audio_mixing.py`` compares
#: the two constructions attribute for attribute.
_PROBED_READER_MOVIEPY_VERSIONS = ("2.1.2",)


def _probed_audio_reader(reader_class, filename, fps, duration):
    """``reader_class(filename, decode_file=False, fps=fps, nbytes=2,
    buffersize=200000)`` without its ``ffmpeg -i`` probe.

    ``FFMPEG_AudioReader.__init__`` probes the file for the one value it needs
    from it, the duration, which the clip already holds from its sidecar (it
    *is* the reader's duration: ``AudioFileClip`` copies it from there). The
    probe is a process start plus a ``communicate`` -- tens of milliseconds a
    clip on Windows, once per narrated sentence of every render. Everything
    else is the constructor's own assignments and its first two calls, so the
    reader decodes the same bytes through the same ffmpeg command.
    """
    import moviepy

    from algan.animation_timeline.timeline import _opt_disabled

    version = getattr(moviepy, "__version__", None)
    if version not in _PROBED_READER_MOVIEPY_VERSIONS or _opt_disabled("audioprobe"):
        return reader_class(
            filename, decode_file=False, fps=fps, nbytes=2, buffersize=200000
        )
    nbytes = 2
    reader = reader_class.__new__(reader_class)
    reader.filename = filename
    reader.nbytes = nbytes
    reader.fps = fps
    reader.format = f"s{8 * nbytes}le"
    reader.codec = f"pcm_s{8 * nbytes}le"
    reader.nchannels = 2
    reader.duration = duration
    reader.bitrate = None
    reader.infos = None
    reader.proc = None
    reader.n_frames = int(reader.fps * reader.duration)
    reader.buffersize = min(reader.n_frames + 1, 200000)
    reader.buffer = None
    reader.buffer_startframe = 1
    reader.initialize()
    reader.buffer_around(1)
    return reader


def _lazy_audio_file_clip_class():
    """Build the lazy clip class on first use, so moviepy stays a deferred import."""
    global _LAZY_AUDIO_FILE_CLIP
    if _LAZY_AUDIO_FILE_CLIP is not None:
        return _LAZY_AUDIO_FILE_CLIP
    from moviepy import AudioFileClip
    from moviepy.audio.AudioClip import AudioClip
    from moviepy.audio.io.readers import FFMPEG_AudioReader

    class LazyAudioFileClip(AudioFileClip):
        """``AudioFileClip`` with its probed metadata supplied and its reader deferred.

        Exactly the attributes ``AudioFileClip.__init__`` sets, from the values
        the probe returned for this very file; ``reader`` is created on first
        access with ``AudioFileClip``'s own default arguments. Copies made by
        ``with_start`` and friends read through the original's reader, as
        ``AudioFileClip``'s own ``frame_function`` closure does.
        """

        def __init__(self, filename, info):
            AudioClip.__init__(self)
            self.filename = filename
            self._reader = None
            self.fps = info["fps"]
            self.duration = info["duration"]
            self.end = info["duration"]
            self.buffersize = info["buffersize"]
            self.frame_function = lambda t: self.reader.get_frame(t)
            self.nchannels = info["nchannels"]
            # Opens the decoder ``frame_function`` reads through -- this
            # original's, which copies share along with the closure itself.
            # write_composite_audio calls it ahead of the mix, on a worker.
            self.open_reader = lambda: self.reader

        @property
        def reader(self):
            if self._reader is None:
                self._reader = _probed_audio_reader(
                    FFMPEG_AudioReader, self.filename, self.fps, self.duration
                )
            return self._reader

        @reader.setter
        def reader(self, value):
            self._reader = value

        def close(self):
            if self._reader is not None:
                self._reader.close()
                self._reader = None

    _LAZY_AUDIO_FILE_CLIP = LazyAudioFileClip
    return LazyAudioFileClip


def write_composite_audio(composite, filename, fps, nbytes, codec, buffersize=2000):
    """Write a ``CompositeAudioClip`` as its ``write_audiofile`` would, faster.

    Produces the samples ``composite.write_audiofile(filename, fps=fps,
    nbytes=nbytes, codec=codec, buffersize=buffersize)`` produces -- the same
    chunk grid, the same ``get_frame`` calls on the same member clips in the
    same order, the same mixing and quantization -- and hands them to the same
    ffmpeg writer. Only moviepy's per-call overhead is gone: its decorators
    bind every argument through ``inspect.signature``, and the composite asks
    *every* member whether it is playing in *every* chunk, so a narrated scene
    of a few dozen sentences spent seconds mixing a minute of audio before its
    first frame could render. Here each chunk consults only the members whose
    span it overlaps, found with the float comparisons ``Clip.is_playing``
    makes, and the decoders of the next few members are opened on worker
    threads while earlier ones mix (a speech clip's ``open_reader``; opening
    one starts an ffmpeg process and waits for its first buffer, which is most
    of what the mix itself costs).

    Returns False without writing anything when the composite holds a member
    this cannot vouch for (one that overrides ``is_playing``), or a chunk grid
    moviepy itself would reject; the caller then writes it the moviepy way.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter
    from moviepy.Clip import Clip

    clips = list(composite.clips)
    if any(type(clip).is_playing is not Clip.is_playing for clip in clips):
        return False
    total_size = int(fps * composite.duration)
    nchunks = total_size // buffersize + 1
    positions = np.linspace(0, total_size, nchunks + 1, endpoint=True, dtype=int)
    if (np.diff(positions) <= 0).any():
        return False
    # Each chunk's ``t.min()`` and ``t.max()``: its times are ``(1 / fps) * k``
    # for increasing ``k``, so these are its first and last, computed by the
    # same product. A member is playing in chunk i unless ``tmin >= end`` or
    # ``tmax < start`` (Clip.is_playing), and both bounds are monotone in i, so
    # its chunks are one contiguous range found by bisection.
    step = 1.0 / fps
    chunk_tmin = step * positions[:-1]
    chunk_tmax = step * (positions[1:] - 1)
    ranges = []
    for clip in clips:
        first = int(np.searchsorted(chunk_tmax, clip.start, side="left"))
        stop = (
            nchunks
            if clip.end is None
            else int(np.searchsorted(chunk_tmin, clip.end, side="left"))
        )
        ranges.append((first, max(first, stop)))
    order = sorted(
        (k for k in range(len(clips)) if ranges[k][0] < ranges[k][1]),
        key=lambda k: (ranges[k][0], k),
    )

    # One opener per distinct decoder, in order of first use.
    openers = {}
    opener_of = {}
    for k in order:
        opener = getattr(clips[k], "open_reader", None)
        if callable(opener):
            openers.setdefault(id(opener), opener)
            opener_of[k] = id(opener)
    queue = list(openers)
    futures = {}
    prefetch = 3

    inttype = {1: "int8", 2: "int16", 4: "int32"}[nbytes]
    scale = 2 ** (8 * nbytes - 1)
    nchannels = composite.nchannels
    logger.debug("Mixing %d audio clips into %s", len(clips), filename)
    pool = ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="algan-audio")
    writer = FFMPEG_AudioWriter(filename, fps, nbytes, nchannels, codec=codec)
    try:

        def top_up():
            while queue and len(futures) < prefetch:
                key = queue.pop(0)
                futures[key] = pool.submit(openers[key])

        top_up()
        opened = set()
        active = []
        cursor = 0
        pending = []
        pending_samples = 0
        for i in range(nchunks):
            while cursor < len(order) and ranges[order[cursor]][0] <= i:
                active.append(order[cursor])
                cursor += 1
            active = sorted(k for k in active if ranges[k][1] > i)
            t = step * np.arange(positions[i], positions[i + 1])
            sounds = []
            for k in active:
                clip = clips[k]
                key = opener_of.get(k)
                if key is not None and key not in opened:
                    future = futures.pop(key, None)
                    if future is None:
                        # Not reached yet by the prefetch: open it here instead,
                        # and never on a worker afterwards.
                        queue.remove(key)
                        openers[key]()
                    else:
                        future.result()
                    opened.add(key)
                    top_up()
                start, end = clip.start, clip.end
                part = 1 * (t >= start)
                if end is not None:
                    part *= t <= end
                sounds.append(clip.get_frame(t - start) * np.array([part]).T)
            # CompositeAudioClip.frame_function and AudioClip.to_soundarray.
            frame = np.zeros((len(t), nchannels)) + sum(sounds)
            frame = np.maximum(-0.99, np.minimum(0.99, frame))
            pending.append((scale * frame).astype(inttype))
            pending_samples += len(t)
            if pending_samples >= 1 << 20:
                writer.write_frames(np.concatenate(pending))
                pending, pending_samples = [], 0
        if pending:
            writer.write_frames(np.concatenate(pending))
    finally:
        # Openers still queued or running are finished (not cancelled) so no
        # decoder is left half-built behind a clip the caller will close.
        pool.shutdown(wait=True, cancel_futures=True)
        writer.close()
    return True
