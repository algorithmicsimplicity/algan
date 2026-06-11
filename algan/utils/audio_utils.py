import bisect
import os
import hashlib
from pathlib import Path

from moviepy import AudioFileClip
import pyttsx3

from algan.settings.defaults import DIRECTORY_DEFAULTS

class Timer:
    def __init__(self):
        self.time = 0

import os
import torch
import torchaudio
#import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio.functional as F

import re, string
pattern = re.compile('[\W_]+', re.UNICODE)

# --- Configuration ---
# A larger chunk size is more efficient with the optimized torchaudio function.
CHUNK_DURATION_S = 1 * 60  # 45 minutes
MODEL_ID = "facebook/wav2vec2-base-960h"


class TranscriptAudioMismatchError(Exception):
    pass


class Counter():
    def __init__(self):
        self.count = 0


def strip_nonchars(x):
    return pattern.sub('', x).upper()
    for _ in "~`\"'][(),.?/><!@#$%^&*+-=-_\\|:;":
        x = x.replace(_, "")
    return x.upper()


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def align_large_audio_torchaudio_robust(audio_path, transcript_path, model_id=MODEL_ID, chunk_duration_s=CHUNK_DURATION_S):
    """
    Aligns a large audio file with its transcript using a robust, optimized
    CTC-based chunking strategy powered by torchaudio, including a retry guard.
    """
    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    processor.tokenizer.encoder['*'] = len(processor.tokenizer.encoder)
    processor.tokenizer.decoder[processor.tokenizer.encoder['*']] = '*'
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    print("Loading audio file info...")
    audio_info = torchaudio.info(audio_path)
    audio_duration_s = audio_info.num_frames / audio_info.sample_rate

    with open(transcript_path, 'r') as f:
        full_transcript_text = f.read().upper()
        full_transcript_text = full_transcript_text.replace('-', ' ')
        full_transcript_words = full_transcript_text.split()
        full_transcript_text = ' '.join(full_transcript_words)

    total_chunks = int(torch.ceil(torch.tensor(audio_duration_s / chunk_duration_s)).item())

    estimated_words_per_minute = 120
    estimated_words_per_chunk = estimated_words_per_minute * chunk_duration_s / 60
    print(f"Audio duration: {audio_duration_s:.2f}s. Will be processed in {total_chunks} chunks.")

    all_word_segments = []
    transcript_cursor = 0

    chunk_start_s = 0
    final_chunk = False
    while chunk_start_s < audio_duration_s:
        chunk_end_s = min(chunk_start_s + chunk_duration_s, audio_duration_s)
        if chunk_start_s + chunk_duration_s >= audio_duration_s-10:
            final_chunk = True
        if chunk_start_s > 1300:
            print(' ')

        print(f"\n--- Processing Chunk {chunk_start_s}/{audio_duration_s} ({chunk_start_s:.2f}s to {chunk_end_s:.2f}s) ---")

        print("Loading audio chunk...")
        frame_offset = int(chunk_start_s * audio_info.sample_rate)
        num_frames = int((chunk_end_s - chunk_start_s) * audio_info.sample_rate)
        waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        if sr != processor.feature_extractor.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.feature_extractor.sampling_rate)(
                waveform)

        input_values = processor(waveform, return_tensors="pt",
                                 sampling_rate=processor.feature_extractor.sampling_rate).input_values.squeeze(0)

        print("Running model encoder...")
        with torch.no_grad():
            logits = model(input_values.to(device)).logits[0]
            emissions = torch.log_softmax(logits, dim=-1)
            emissions = torch.cat((emissions, torch.zeros_like(emissions[...,:1])), -1)

        transcript_was_too_long = True
        retry_count = 0
        MAX_RETRIES = 10
        transcript_len_multiplier_base = 0.75
        transcript_len_multiplier = transcript_len_multiplier_base
        aligned_tokens, scores = None, None

        while transcript_was_too_long and retry_count < MAX_RETRIES:
            remaining_transcript = full_transcript_words[transcript_cursor:]

            audio_ratio = ((chunk_end_s - chunk_start_s) / (audio_duration_s - chunk_start_s) )if (
                                                        audio_duration_s - chunk_start_s) > 0 else 0
            estimated_len = max(int(estimated_words_per_chunk), 3)

            transcript_words = (remaining_transcript[:estimated_len] if not
                                       final_chunk else remaining_transcript)
            transcript_segment_text = ' '.join(transcript_words)
            pad = ('*' if not final_chunk else '')

            tokenized_transcript = processor.tokenizer(strip_nonchars(transcript_segment_text) +
                                                       pad).input_ids

            # --- Perform the optimized alignment ---
            blank_id = processor.tokenizer.pad_token_id
            delimiter_id = processor.tokenizer.word_delimiter_token_id
            try:
                aligned_tokens, scores = torchaudio.functional.forced_align(
                    emissions.unsqueeze(0),
                    torch.tensor([tokenized_transcript], dtype=torch.int32, device=device),
                    blank=blank_id
                )
            except:
                print('a')
            token_spans = F.merge_tokens(aligned_tokens[0], scores[0])

            time_per_frame = 1

            word_spans = unflatten(token_spans, [len(strip_nonchars(word)) for word in transcript_words] + ([len(pad)] if len(pad) > 0 else []))
            transcript_words_star = transcript_words + ([pad] if len(pad) > 0 else [])
            word_spans = [[transcript_words_star[i], word_spans[i][0].start * time_per_frame,
                           word_spans[i][-1].end * time_per_frame] for i in range(len(word_spans))]
            # Move results to CPU for analysis
            if word_spans[-1][1] >= emissions.shape[0]-10 and not final_chunk:
                transcript_was_too_long = True
                retry_count += 1
                estimated_words_per_chunk *= 0.75
                print("Transcript chunk was too long for audio chunk, retrying with shorter transcript chunk.")
                continue
            transcript_was_too_long = False

            time_per_frame = (chunk_end_s - chunk_start_s) / emissions.shape[0]
            chunk_word_segments = [[strip_nonchars(_[0]), chunk_start_s + _[1] * time_per_frame,
                                    chunk_start_s + _[2] * time_per_frame] for _ in (word_spans[:-1] if not final_chunk else word_spans)]

        try:
            estimated_words_per_chunk = len(chunk_word_segments) * ((chunk_end_s - chunk_start_s)
                                                                    / (chunk_word_segments[-1][-1] - chunk_start_s)) * 0.9
        except:
            print('b')
        all_word_segments.extend(chunk_word_segments)
        chunk_start_s = all_word_segments[-1][-1] + time_per_frame * 0.5

        confirmed_transcript_len = len(chunk_word_segments)
        transcript_cursor += confirmed_transcript_len
        if transcript_cursor >= len(full_transcript_words):
            break

    #df = pd.DataFrame(all_word_segments)
    #df.to_csv(output_filename, index=False)
    return all_word_segments

    print("\nAlignment process complete.")
    #print(f"Output saved to {output_filename}")

def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            return i
    return -1

def get_speech_generator_from_file(audio_file, transcript_file):
    full_ac = AudioFileClip(audio_file)

    hasher = hashlib.sha256()
    hasher.update((f'{audio_file}!!___!!{full_ac.duration}').encode())
    hash_bytes = hasher.hexdigest()[:64]
    time_stamp_file = os.path.join(DIRECTORY_DEFAULTS.cache_directory, 'audio', f"{hash_bytes}.csv")

    if os.path.exists(time_stamp_file):
        word_time_stamps = []
        with open(time_stamp_file, mode='r') as f:
            for line in f.readlines():
                word, start, end = [_.strip() for _ in line.split(',')]
                word_time_stamps.append([word, float(start), float(end)])
    else:
        word_time_stamps = align_large_audio_torchaudio_robust(audio_file, transcript_file)
        Path(time_stamp_file).parent.mkdir(parents=True, exist_ok=True)
        with open(time_stamp_file, mode='w') as f:
            for word, start, end in word_time_stamps:
                f.write(f"{word},{start},{end}\n")

    word_counter = Counter()
    def generator(script):
        script = script.replace('-', ' ')
        script_words = [strip_nonchars(_) for _ in script.split(' ')]
        script_words = [_ for _ in script_words if len(_) > 0]

        script_start_ind = subfinder([_[0] for _ in word_time_stamps], script_words)

        if script_start_ind < 0:#script_words != [_[0] for _ in word_time_stamps[word_counter.count:word_counter.count+len(script_words)]]:
            #raise TranscriptAudioMismatchError(f'Error, the following text was not found in the recorded '
            #                                   f'transcript of the speech audio file:\n\n{script_words}')
            print(f'Warning: the following text was not found in the recorded transcript of the speech'
                  f' audio file, and so this speech will be machine generated:\n\n{script_words}')
            return get_pyttsx_speech_generator(script)

        audio_start = word_time_stamps[script_start_ind][1]
        audio_end = word_time_stamps[script_start_ind+len(script_words)-1][2]
        if script_start_ind+len(script_words)-1 < len(word_time_stamps)-1:
            dif = word_time_stamps[script_start_ind+len(script_words)][1] - audio_end
            audio_end += min(dif*0.5, 0.5)

        sub_ac = full_ac.subclipped(max(audio_start - 0.05, 0),
                min(audio_end + 0.05, full_ac.duration))
        return sub_ac
    return generator


def get_pyttsx_speech_generator(script):
    hasher = hashlib.sha256()
    hasher.update(script.encode())
    hash_bytes = hasher.hexdigest()[:32]
    file = os.path.join(DIRECTORY_DEFAULTS.cache_directory, 'audio', f"{hash_bytes}.mp3")
    if not os.path.exists(file):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.save_to_file(script, file)
        engine.runAndWait()
        engine.stop()
    return AudioFileClip(file)
