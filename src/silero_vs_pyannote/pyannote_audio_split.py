import logging
import os

import librosa
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

from silero_vs_pyannote.config import (
    AUDIO_SEG_LOWER_LIMIT,
    AUDIO_SEG_UPPER_LIMIT,
    HYPER_PARAMETERS,
)
from silero_vs_pyannote.utils import (
    process_non_mute_segments,
    sec_to_frame,
    sec_to_millis,
)

# load the evnironment variable
load_dotenv()

USE_AUTH_TOKEN = os.getenv("use_auth_token")
# Call the setup_logging function at the beginning of your script


def initialize_vad_pipeline():
    """
    Initializes the Voice Activity Detection (VAD) pipeline using Pyannote.
    Returns:
        Pipeline: Initialized VAD pipeline
    """
    logging.info("Initializing Voice Activity Detection pipeline...")
    try:
        vad_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=USE_AUTH_TOKEN,
        )
    except Exception as e:
        logging.warning(f"Failed to load online model: {e}. Using local model.")
        vad_pipeline = Pipeline.from_pretrained(
            "tests/pyannote_vad_model",
            use_auth_token=False,
        )
    vad_pipeline.instantiate(HYPER_PARAMETERS)
    logging.info("VAD pipeline initialized successfully.")
    return vad_pipeline


def get_split_audio(
    audio_data,
    full_audio_id,
    lower_limit=AUDIO_SEG_LOWER_LIMIT,
    upper_limit=AUDIO_SEG_UPPER_LIMIT,
):
    """Splits audio into segments based on voice activity detection.

    Args:
        audio_data (bytes): Raw audio data
        lower_limit (float): Minimum segment duration in seconds
        upper_limit (_type_): Maximum segment duration in seconds
        full_audio_id (str):  Identifier for the full audio file

    Returns:
        dict: Mapping of segment IDs to raw audio data
    """

    logging.info(f"Splitting audio for {full_audio_id}")
    split_audio = {}
    temp_audio_file = "temp_audio_in_memory.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_data)

    # initialize vad pipeline
    pipeline = initialize_vad_pipeline()
    vad = pipeline(temp_audio_file)

    original_audio_segment = AudioSegment.from_file(temp_audio_file)
    original_audio_ndarray, sampling_rate = torchaudio.load(temp_audio_file)
    original_audio_ndarray = original_audio_ndarray[0]

    counter = 1
    for vad_span in vad.get_timeline().support():
        start_ms = sec_to_millis(vad_span.start)
        end_ms = sec_to_millis(vad_span.end)
        vad_segment = original_audio_segment[start_ms:end_ms]
        vad_span_length = vad_span.end - vad_span.start
        if lower_limit <= vad_span_length <= upper_limit:
            segment_key = f"{full_audio_id}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}"  # noqa: E231
            split_audio[segment_key] = vad_segment
            counter += 1
        elif vad_span_length > upper_limit:
            non_mute_segment_splits = librosa.effects.split(
                original_audio_ndarray[
                    int(
                        sec_to_frame(vad_span.start, sampling_rate)
                    ) : int(  # noqa: E203
                        sec_to_frame(vad_span.end, sampling_rate)
                    )
                ],
                top_db=30,
            )
            counter = process_non_mute_segments(
                non_mute_segment_splits,
                original_audio_segment,
                vad_span,
                sampling_rate,
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
