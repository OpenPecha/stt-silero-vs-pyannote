import logging
import os

import librosa
import torch
import torchaudio
from pydub import AudioSegment

from silero_vs_pyannote.config import (
    AUDIO_SEG_LOWER_LIMIT,
    AUDIO_SEG_UPPER_LIMIT,
    MODEL_NAME,
    REPO,
)
from silero_vs_pyannote.utils import process_non_mute_segments, sec_to_millis


def initialize_silero_vad(repo, model_name):
    """
    Initialize the Silero VAD model and utilities.

    Parameters:
        repo (str): The repository from which to load the model.
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and utilities (functions) from the Silero VAD library.
    """
    try:
        logging.info("Initializing Silero VAD model...")

        # Load the Silero VAD model and its utilities
        model, utils = torch.hub.load(
            repo_or_dir=repo,
            model=model_name,
            force_reload=False,  # Change to True if you want to force re-download the model
        )

        logging.info("Silero VAD model successfully initialized.")

        return model, utils

    except Exception as e:
        logging.error(f"Failed to initialize Silero VAD model: {e}")
        raise


def get_split_audio_using_silero(
    audio_data,
    full_audio_id,
    lower_limit=AUDIO_SEG_LOWER_LIMIT,
    upper_limit=AUDIO_SEG_UPPER_LIMIT,
):
    logging.info(f"Splitting audio for {full_audio_id}")
    split_audio = {}
    temp_audio_file = "temp_audio_in_memory.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_data)
    # Initialize Silero VAD model
    model, utils = initialize_silero_vad(REPO, MODEL_NAME)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    wav = read_audio(temp_audio_file, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    original_audio_segment = AudioSegment.from_file(temp_audio_file)
    original_audio_ndarray, sampling_rate = torchaudio.load(temp_audio_file)
    original_audio_ndarray = original_audio_ndarray[0]

    counter = 1
    for ts in speech_timestamps:
        start_frame = ts["start"]
        end_frame = ts["end"]
        vad_span = type(
            "Timeline", (), {"start": start_frame / 16000, "end": end_frame / 16000}
        )()
        segment_duration = (end_frame - start_frame) / 16000  # Duration in seconds

        if lower_limit <= segment_duration <= upper_limit:
            start_ms = sec_to_millis(vad_span.start)
            end_ms = sec_to_millis(vad_span.end)
            segment = original_audio_segment[start_ms:end_ms]
            segment_key = (
                f"{full_audio_id}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}"  # noqa
            )
            split_audio[segment_key] = segment
            counter += 1
        elif segment_duration > upper_limit:
            non_mute_segment_splits = librosa.effects.split(
                original_audio_ndarray[start_frame:end_frame],
                top_db=30,
            )
            counter = process_non_mute_segments(
                non_mute_segment_splits,
                original_audio_segment,
                vad_span,
                16000,
                lower_limit,
                upper_limit,
                full_audio_id,
                counter,
                split_audio,
            )

    os.remove(temp_audio_file)
    logging.info(
        f"Finished splitting audio for {full_audio_id}. Total segments: {len(split_audio)}"
    )
    return split_audio
