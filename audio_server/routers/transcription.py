from typing import Annotated, Literal, List
from fastapi import APIRouter, Form, HTTPException, Request
from ..dependencies import (
    WhisperModelManagerDependency,
    AudioFileDependency,
    ConfigDependency,
)
from ..config import get_enabled_models

router = APIRouter(tags=["Audio Transcription"])

ResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]
TimestampGranularity = Literal["word", "segment"]


def get_enabled_model_names():
    """Get list of enabled model names for validation"""
    return [model.id for model in get_enabled_models()]


@router.post(
    "/v1/audio/transcriptions",
    summary="Transcribe audio to text",
    description="Converts audio files to text using Whisper models. Compatible with OpenAI's transcription API.",
    response_description="The transcribed text from the audio file",
)
async def transcribe_file(
    request: Request,
    model_manager: WhisperModelManagerDependency,
    config: ConfigDependency,
    audio: AudioFileDependency,
    model: Annotated[str, Form(description="Model to use for transcription")] = "turbo",
    language: Annotated[
        str, Form(description="Language code (e.g. 'en', 'fr')")
    ] = "en",
    prompt: Annotated[
        str, Form(description="Optional prompt to guide transcription")
    ] = "",
    response_format: Annotated[
        ResponseFormat, Form(description="Output format")
    ] = "json",
    temperature: Annotated[
        float, Form(description="Sampling temperature (0-1)", ge=0, le=1)
    ] = 0.0,
    timestamp_granularities: Annotated[
        List[TimestampGranularity],
        Form(description="Timestamp granularities to include"),
    ] = ["segment"],
):
    """
    Transcribe audio file to text using Whisper models.

    - **file**: Audio file in supported format (MP3, WAV, M4A, FLAC, etc.)
    - **model**: Whisper model to use for transcription

    Returns the transcribed text in OpenAI-compatible format.
    """
    # Handle timestamp_granularities[] parameter for curl compatibility
    form_data = await request.form()
    timestamp_granularities_from_form = form_data.getlist("timestamp_granularities[]")
    if timestamp_granularities_from_form:
        timestamp_granularities = (
            timestamp_granularities_from_form  # Use array from curl
        )

    # Determine if word timestamps are needed
    needs_word_timestamps = "word" in timestamp_granularities

    # Validate model is enabled
    enabled_models = get_enabled_model_names()
    if model not in enabled_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is disabled. Available models: {enabled_models}",
        )

    try:
        whisper_model = model_manager.load_model(model)
        segments, transcription_info = whisper_model.transcribe(
            audio,
            language=language,
            initial_prompt=prompt if prompt else None,
            temperature=temperature,
            word_timestamps=needs_word_timestamps,
        )

        # Convert segments to list for multiple processing
        segments_list = list(segments)

        # Calculate duration for usage across all formats
        audio_duration = (
            transcription_info.duration
            if hasattr(transcription_info, "duration")
            else sum(segment.end - segment.start for segment in segments_list)
        )

        # Format response based on response_format
        if response_format == "text":
            text = "".join(segment.text for segment in segments_list).strip()
            return text
        elif response_format == "verbose_json":
            text = "".join(segment.text for segment in segments_list).strip()

            # Build OpenAI-compatible verbose response
            response = {
                "task": "transcribe",
                "language": transcription_info.language
                if hasattr(transcription_info, "language")
                else "english",
                "duration": audio_duration,
                "text": text,
                "usage": {"type": "duration", "seconds": round(audio_duration)},
            }

            # Always include segments for verbose_json
            response["segments"] = []
            for i, segment in enumerate(segments_list):
                segment_data = {
                    "id": i,
                    "seek": 0,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "temperature": temperature,
                }
                response["segments"].append(segment_data)

            # Add word-level timestamps if requested
            if needs_word_timestamps:
                all_words = []
                for segment in segments_list:
                    if hasattr(segment, "words") and segment.words:
                        for word in segment.words:
                            all_words.append(
                                {
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end,
                                }
                            )
                    else:
                        # If no word-level data, split segment text as approximation
                        words = segment.text.strip().split()
                        if words:
                            word_duration = (segment.end - segment.start) / len(words)
                            for j, word in enumerate(words):
                                word_start = segment.start + (j * word_duration)
                                word_end = word_start + word_duration
                                all_words.append(
                                    {"word": word, "start": word_start, "end": word_end}
                                )

                response["words"] = all_words

            return response
        elif response_format == "srt":
            srt_content = ""
            for i, segment in enumerate(segments_list, 1):
                start_time = format_timestamp_srt(segment.start)
                end_time = format_timestamp_srt(segment.end)
                srt_content += (
                    f"{i}\n{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
                )
            return srt_content.rstrip()
        elif response_format == "vtt":
            vtt_content = "WEBVTT\n\n"
            for segment in segments_list:
                start_time = format_timestamp_vtt(segment.start)
                end_time = format_timestamp_vtt(segment.end)
                vtt_content += (
                    f"{start_time} --> {end_time}\n{segment.text.strip()}\n\n"
                )
            return vtt_content.rstrip()
        else:  # json (default)
            text = "".join(segment.text for segment in segments_list).strip()
            return {
                "text": text,
                "usage": {"type": "duration", "seconds": round(audio_duration)},
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed with model '{model}': {str(e)}",
        )


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
