"""Wrappers and helpers for interacting with different ASR services."""

from __future__ import annotations

import json
import logging
import os
import time
from functools import partial
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

DEFAULT_MENLO_BASE_URL = os.getenv("MENLO_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MENLO_API_KEY = os.getenv("MENLO_API_KEY", "dummy")
DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://0.0.0.0:3349/v1")
DEFAULT_VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")

ServiceFunction = Callable[[str], str]
ServiceModelSpec = Tuple[str, str]


def _suppress_logs(*names: str, level: int = logging.WARNING) -> None:
    for name in names:
        logging.getLogger(name).setLevel(level)


def transcribe_menlo(audio_path: str, model: str = "large-v3") -> str:
    """Transcribe audio using a Menlo-hosted OpenAI-compatible endpoint."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The openai package is required for Menlo transcription") from exc

    client = OpenAI(base_url=DEFAULT_MENLO_BASE_URL, api_key=DEFAULT_MENLO_API_KEY)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
        )
    return _extract_openai_text(transcription)


def _extract_openai_text(payload) -> str:
    """Return transcription text from OpenAI-compatible responses."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if isinstance(payload, dict) and "text" in payload:
        return payload["text"]
    return str(payload)


def transcribe_vllm(audio_path: str, model: str = "large-v3") -> str:
    """Transcribe audio using a local vLLM Whisper service."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The openai package is required for vLLM transcription") from exc

    client = OpenAI(base_url=DEFAULT_VLLM_BASE_URL, api_key=DEFAULT_VLLM_API_KEY)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
        )
    return _extract_openai_text(transcription)


def transcribe_openai(audio_path: str, model: str = "gpt-4o-transcribe") -> str:
    """Transcribe audio using OpenAI's hosted models."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The openai package is required for OpenAI transcription") from exc

    client = OpenAI()
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model=model, file=audio_file)
    return transcription.text


def transcribe_groq(audio_path: str, model: str = "whisper-large-v3-turbo") -> str:
    """Transcribe audio using Groq's Whisper endpoints."""
    _suppress_logs("httpx", "urllib3", "requests", "groq", level=logging.ERROR)
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The groq package is required for Groq transcription") from exc

    client = Groq()
    with open(audio_path, "rb") as file_handle:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file_handle.read()),
            model=model,
            response_format="verbose_json",
        )
    return transcription.text


def transcribe_elevenlabs(audio_path: str, model_id: str = "scribe_v1") -> str:
    """Transcribe audio using ElevenLabs."""
    _suppress_logs("httpx", "elevenlabs", level=logging.WARNING)
    try:
        from elevenlabs import ElevenLabs
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The elevenlabs package is required for ElevenLabs transcription") from exc

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    with open(audio_path, "rb") as audio_file:
        transcript = client.speech_to_text.convert(
            file=audio_file,
            model_id=model_id,
            language_code="en",
        )
    return transcript.text


def transcribe_speechmatics(audio_path: str, language: str = "en") -> str:
    """Transcribe audio using Speechmatics batch API."""
    _suppress_logs("speechmatics", "httpx", level=logging.WARNING)
    try:
        from speechmatics.batch_client import BatchClient
        from speechmatics.models import BatchTranscriptionConfig, ConnectionSettings
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The speechmatics package is required for Speechmatics transcription") from exc

    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=os.getenv("SPEECHMATICS_API_KEY"),
    )

    with BatchClient(settings) as client:
        job_id = client.submit_job(audio_path, BatchTranscriptionConfig(language=language))
        transcript = client.wait_for_completion(job_id, transcription_format="txt")
    return transcript


def transcribe_gladia(audio_path: str) -> str:
    """Transcribe audio using Gladia's API."""
    _suppress_logs("requests", "urllib3", level=logging.WARNING)
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("The requests package is required for Gladia transcription") from exc

    api_key = os.getenv("GLADIA_API_KEY")
    headers = {"x-gladia-key": api_key}

    with open(audio_path, "rb") as file_handle:
        upload_resp = requests.post(
            "https://api.gladia.io/v2/upload",
            headers=headers,
            files={"audio": (os.path.basename(audio_path), file_handle, "audio/wav")},
            timeout=60,
        )
    upload_resp.raise_for_status()
    file_url = upload_resp.json()["audio_url"]

    job_resp = requests.post(
        "https://api.gladia.io/v2/pre-recorded",
        headers={**headers, "Content-Type": "application/json"},
        json={"audio_url": file_url},
        timeout=60,
    )
    job_resp.raise_for_status()
    result_url = job_resp.json()["result_url"]

    while True:
        poll_resp = requests.get(result_url, headers=headers, timeout=60)
        poll_resp.raise_for_status()
        payload = poll_resp.json()
        status = payload.get("status")
        if status == "done":
            return payload["result"]["transcription"]["full_transcript"]
        if status == "error":
            raise RuntimeError(f"Gladia transcription failed: {payload}")
        time.sleep(2)


DEFAULT_SERVICE_MODELS: Sequence[ServiceModelSpec] = (
    ("menlo", "large-v3"),
    ("vllm", "openai/whisper-large-v3"),
    ("openai", "gpt-4o-transcribe"),
    ("openai", "gpt-4o-mini-transcribe"),
    ("openai", "whisper-1"),
    ("groq", "whisper-large-v3"),
    ("groq", "whisper-large-v3-turbo"),
    ("elevenlabs", "scribe_v1"),
    ("speechmatics", "en"),
    ("gladia", "pre-recorded"),
)


def build_service_function_map(service_models: Iterable[ServiceModelSpec]) -> Dict[str, ServiceFunction]:
    """Return a mapping of service identifiers to callable transcription functions."""
    service_funcs: Dict[str, ServiceFunction] = {}
    for service, model in service_models:
        safe_model = model.replace("/", "-")
        key = f"{service}_{safe_model}"
        if service == "menlo":
            service_funcs[key] = partial(transcribe_menlo, model=model)
        elif service == "vllm":
            service_funcs[key] = partial(transcribe_vllm, model=model)
        elif service == "openai":
            service_funcs[key] = partial(transcribe_openai, model=model)
        elif service == "groq":
            service_funcs[key] = partial(transcribe_groq, model=model)
        elif service == "elevenlabs":
            service_funcs[key] = partial(transcribe_elevenlabs, model_id=model)
        elif service == "speechmatics":
            service_funcs[key] = partial(transcribe_speechmatics, language=model)
        elif service == "gladia":
            service_funcs[key] = transcribe_gladia
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported service '{service}'")
    return service_funcs


DEFAULT_SERVICE_FUNCTIONS: Dict[str, ServiceFunction] = build_service_function_map(DEFAULT_SERVICE_MODELS)
DEFAULT_SERVICES: List[str] = list(DEFAULT_SERVICE_FUNCTIONS.keys())


__all__ = [
    "ServiceFunction",
    "ServiceModelSpec",
    "transcribe_menlo",
    "transcribe_vllm",
    "transcribe_openai",
    "transcribe_groq",
    "transcribe_elevenlabs",
    "transcribe_speechmatics",
    "transcribe_gladia",
    "DEFAULT_SERVICE_MODELS",
    "DEFAULT_SERVICE_FUNCTIONS",
    "DEFAULT_SERVICES",
    "build_service_function_map",
]
