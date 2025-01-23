import io
import logging
from logging.handlers import RotatingFileHandler

import requests
from pydub import AudioSegment

from silero_vs_pyannote.config import AUDIO_HEADERS, BACKUP_COUNT, MAX_BYTES


# Configure logging
def setup_logging(filename):
    """This function sets up a logger with file handlers.

    Args:
        filename (str): The name of the log file to be created or appended to.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for a rotating log file
    file_handler = RotatingFileHandler(
        filename,
        MAX_BYTES,
        BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def sec_to_millis(seconds):
    return seconds * 1000


def frame_to_sec(frame, sampling_rate):
    return frame / sampling_rate


def sec_to_frame(sec, sr):
    return sec * sr


def convert_to_16K(audio_data):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio_16k = audio.set_frame_rate(16000).set_channels(1)
        output_buffer = io.BytesIO()
        audio_16k.export(output_buffer, format="wav")
        output_buffer.seek(0)
        return output_buffer.read()
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def chop_long_segment_duration(
    segment_split_duration,
    upper_limit,
    original_audio_segment,
    vad_span,
    split_start,
    sampling_rate,
    full_audio_id,
    split_audio,
    counter,
):
    """Splits an audio segment into smaller chunks if its duration exceeds the specified upper limit.


    Args:
        segment_split_duration (float): The duration of the segment to be split (in seconds).
        upper_limit (float): The maximum duration allowed for a segment (in seconds).
        original_audio_segment (AudioSegment): The original audio segment to be split.
        vad_span (Timeline): The Voice Activity Detection (VAD) span containing start and end times for the audio segment.
        split_start (float): The starting point for splitting the audio segment (in seconds).
        sampling_rate (int): The sampling rate of the audio (in Hz).
        full_audio_id (str): The unique identifier for the full audio file.
        split_audio (dict): A dictionary to store the resulting split audio segments with their IDs as keys.
        output_folder (str): The directory where the split segments should be saved.
        counter (int): The counter for naming the segment files.

    Returns:
        int: The updated counter after processing the split segments.
    """  # noqa: E501
    chop_length = segment_split_duration / 2
    while chop_length > upper_limit:
        chop_length = chop_length / 2
    for chop_index in range(int(segment_split_duration / chop_length)):
        start_ms = sec_to_millis(
            vad_span.start
            + frame_to_sec(split_start, sampling_rate)
            + chop_length * chop_index
        )
        end_ms = sec_to_millis(  # noqa: E203
            vad_span.start
            + frame_to_sec(split_start, sampling_rate)
            + chop_length * (chop_index + 1)
        )
        segment_split_chop = original_audio_segment[start_ms:end_ms]
        segment_key = (
            f"{full_audio_id}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}"  # noqa
        )
        split_audio[segment_key] = segment_split_chop
        counter += 1
    return counter


def process_non_mute_segments(
    non_mute_segment_splits,
    original_audio_segment,
    vad_span,
    sampling_rate,
    lower_limit,
    upper_limit,
    full_audio_id,
    counter,
    split_audio,
):
    """Processes non-mute segments by splitting them based on duration constraints and saving them as separate audio files.

    Args:
        non_mute_segment_splits (list of tuple): A list of tuples containing the start and end frame numbers for non-silent segments.
        original_audio_segment (AudioSegment): The original audio segment to be processed.
        vad_span (Timeline): The Voice Activity Detection (VAD) span containing start and end times for the audio segment.
        sampling_rate (int): The sampling rate of the audio (in Hz).
        lower_limit (float): The minimum duration allowed for a segment (in seconds).
        upper_limit (float): The maximum duration allowed for a segment (in seconds).
        full_audio_id (str): The unique identifier for the full audio file.
        output_folder (str): The directory where the segments should be saved.
        counter (int): The counter for naming the segment files.
        split_audio (dict): A dictionary to store the resulting split audio segments with their IDs as keys.

    Returns:
        int: The updated counter after processing the non-mute segments.
    """  # noqa: E501
    for split_start, split_end in non_mute_segment_splits:
        start_ms = sec_to_millis(
            vad_span.start + frame_to_sec(split_start, sampling_rate)
        )
        end_ms = sec_to_millis(  # noqa: E203
            vad_span.start + frame_to_sec(split_end, sampling_rate)
        )
        segment_split = original_audio_segment[start_ms:end_ms]
        segment_split_duration = (
            vad_span.start + frame_to_sec(split_end, sampling_rate)
        ) - (vad_span.start + frame_to_sec(split_start, sampling_rate))
        if lower_limit <= segment_split_duration <= upper_limit:
            segment_key = f"{full_audio_id}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}"  # noqa: E231
            split_audio[segment_key] = segment_split
            counter += 1
        elif segment_split_duration > upper_limit:
            counter = chop_long_segment_duration(
                segment_split_duration,
                upper_limit,
                original_audio_segment,
                vad_span,
                split_start,
                sampling_rate,
                full_audio_id,
                split_audio,
                counter,
            )
    return counter


def get_audio(audio_url):
    logging.info(f"Downloading audio from: {audio_url}")
    response = requests.get(audio_url, headers=AUDIO_HEADERS, stream=True)
    if response.status_code == 200:
        audio_data = response.content
        logging.info("Converting Audio to 16k")
        audio_data_16k = convert_to_16K(audio_data)
        logging.info("Audio downloaded and converted to 16kHz successfully")
        return audio_data_16k
    else:
        err_message = f"Failed to download audio from {audio_url}"
        logging.error(err_message)
        raise Exception(err_message)
