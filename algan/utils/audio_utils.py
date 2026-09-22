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
    from moviepy import AudioFileClip  # deferred: ~0.3 s of import algan

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
    return AudioFileClip(file)
