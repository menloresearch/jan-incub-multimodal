"""Wrappers and helpers for interacting with different ASR services."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_MENLO_BASE_URL = (
    os.getenv("MENLO_BASE_URL")
    or os.getenv("FASTERWHISPER_BASE_URL")
    or "http://localhost:8000/v1"
)
DEFAULT_MENLO_API_KEY = (
    os.getenv("MENLO_API_KEY") or os.getenv("FASTERWHISPER_API_KEY") or "dummy"
)
DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://0.0.0.0:3349/v1")
DEFAULT_VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")
DEFAULT_PARAKEET_BASE_URL = (
    os.getenv("PARAKEET_BASE_URL")
    or os.getenv("BASE_URL")
    or "http://localhost:8010/v1"
)

ServiceFunction = Callable[[str, Optional[str]], str]
ServiceModelSpec = Tuple[str, str]


ISO2_ALIASES = {
    "zh-cn": "zh",
    "zh_hans": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "zh_tw": "zh",
    "zh-tw": "zh",
    "zh-hk": "zh",
    "pt-br": "pt",
    "pt_br": "pt",
    "en-us": "en",
    "en-gb": "en",
    "ja-jp": "ja",
}

ELEVENLABS_LANG_MAP = {
    "en": "eng",
    "zh": "zho",
    "ja": "jpn",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "it": "ita",
    "pt": "por",
    "ko": "kor",
    "pl": "pol",
    "sv": "swe",
    "no": "nor",
    "nl": "nld",
    "hi": "hin",
    "ar": "ara",
}

SPEECHMATICS_LANG_MAP = {
    "en": {"language": "en"},
    "zh": {"language": "cmn"},
    "cmn": {"language": "cmn"},
}


def _normalize_iso2(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = language.lower().replace("_", "-")
    normalized = ISO2_ALIASES.get(normalized, normalized)
    if "-" in normalized:
        normalized = normalized.split("-")[0]
    if len(normalized) == 2:
        return normalized
    return None


def _suppress_logs(*names: str, level: int = logging.WARNING) -> None:
    for name in names:
        logging.getLogger(name).setLevel(level)


def transcribe_menlo(
    audio_path: str, model: str = "large-v3", language: Optional[str] = None
) -> str:
    """Transcribe audio using a Menlo-hosted OpenAI-compatible endpoint."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The openai package is required for Menlo transcription"
        ) from exc

    client = OpenAI(base_url=DEFAULT_MENLO_BASE_URL, api_key=DEFAULT_MENLO_API_KEY)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
            language=_normalize_iso2(language),
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


def transcribe_parakeet(
    audio_path: str, model: str = "parakeet-tdt-0.6b-v3", language: Optional[str] = None
) -> str:
    """Transcribe audio using a locally served Parakeet OpenAI-compatible endpoint."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The openai package is required for Parakeet transcription"
        ) from exc

    client = OpenAI(base_url=DEFAULT_PARAKEET_BASE_URL)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
            language=_normalize_iso2(language),
        )
    return _extract_openai_text(transcription)


def transcribe_vllm(
    audio_path: str, model: str = "large-v3", language: Optional[str] = None
) -> str:
    """Transcribe audio using a local vLLM Whisper service."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The openai package is required for vLLM transcription"
        ) from exc

    client = OpenAI(base_url=DEFAULT_VLLM_BASE_URL, api_key=DEFAULT_VLLM_API_KEY)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",
            language=_normalize_iso2(language),
        )
    return _extract_openai_text(transcription)


def transcribe_openai(
    audio_path: str, model: str = "gpt-4o-transcribe", language: Optional[str] = None
) -> str:
    """Transcribe audio using OpenAI's hosted models."""
    _suppress_logs("httpx", "openai", level=logging.WARNING)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The openai package is required for OpenAI transcription"
        ) from exc

    client = OpenAI()
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=_normalize_iso2(language),
        )
    return transcription.text


def transcribe_groq(
    audio_path: str,
    model: str = "whisper-large-v3-turbo",
    language: Optional[str] = None,
) -> str:
    """Transcribe audio using Groq's Whisper endpoints."""
    _suppress_logs("httpx", "urllib3", "requests", "groq", level=logging.ERROR)
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The groq package is required for Groq transcription"
        ) from exc

    client = Groq()
    with open(audio_path, "rb") as file_handle:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file_handle.read()),
            model=model,
            response_format="verbose_json",
            language=_normalize_iso2(language),
        )
    return transcription.text


def transcribe_elevenlabs(
    audio_path: str, model_id: str = "scribe_v1", language: Optional[str] = None
) -> str:
    """Transcribe audio using ElevenLabs."""
    _suppress_logs("httpx", "elevenlabs", level=logging.WARNING)
    try:
        from elevenlabs import ElevenLabs
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The elevenlabs package is required for ElevenLabs transcription"
        ) from exc

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    with open(audio_path, "rb") as audio_file:
        lang_code = None
        normalized = _normalize_iso2(language)
        if normalized:
            lang_code = ELEVENLABS_LANG_MAP.get(normalized)
        transcript = client.speech_to_text.convert(
            file=audio_file,
            model_id=model_id,
            language_code=lang_code,
        )
    return transcript.text


def transcribe_speechmatics(audio_path: str, language: str = "en") -> str:
    """Transcribe audio using Speechmatics batch API."""
    _suppress_logs("speechmatics", "httpx", level=logging.WARNING)
    try:
        from speechmatics.batch_client import BatchClient
        from speechmatics.models import BatchTranscriptionConfig, ConnectionSettings
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The speechmatics package is required for Speechmatics transcription"
        ) from exc

    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=os.getenv("SPEECHMATICS_API_KEY"),
    )

    normalized = _normalize_iso2(language) or language
    lang_config = SPEECHMATICS_LANG_MAP.get(normalized, {"language": normalized})

    # Merge operating point/domain hints if provided via environment variables
    operating_point = os.getenv("SPEECHMATICS_OPERATING_POINT")
    domain = os.getenv("SPEECHMATICS_DOMAIN")

    transcription_kwargs = dict(lang_config)
    if operating_point:
        transcription_kwargs["operating_point"] = operating_point
    if domain:
        transcription_kwargs["domain"] = domain

    with BatchClient(settings) as client:
        job_id = client.submit_job(
            audio_path,
            BatchTranscriptionConfig(**transcription_kwargs),
        )
        transcript = client.wait_for_completion(job_id, transcription_format="txt")
    return transcript


def transcribe_gladia(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using Gladia's API."""
    _suppress_logs("requests", "urllib3", level=logging.WARNING)
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "The requests package is required for Gladia transcription"
        ) from exc

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

    poll_interval = 0.5
    while True:
        poll_resp = requests.get(result_url, headers=headers, timeout=60)
        poll_resp.raise_for_status()
        payload = poll_resp.json()
        status = payload.get("status")
        if status == "done":
            return payload["result"]["transcription"]["full_transcript"]
        if status == "error":
            raise RuntimeError(f"Gladia transcription failed: {payload}")
        time.sleep(poll_interval)


DEFAULT_SERVICE_MODELS: Sequence[ServiceModelSpec] = (
    ("fasterwhisper", "whisper-large-v3"),
    ("nvidia", "parakeet-tdt-0.6b-v3"),
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


def build_service_function_map(
    service_models: Iterable[ServiceModelSpec],
) -> Dict[str, ServiceFunction]:
    """Return a mapping of service identifiers to callable transcription functions."""
    service_funcs: Dict[str, ServiceFunction] = {}
    for service, model in service_models:
        safe_model = model.replace("/", "-")
        key = f"{service}_{safe_model}"
        if service == "speechmatics" and model == "en":
            key = service
        elif service == "speechmatics":
            key = f"{service}_{safe_model}"

        if service == "fasterwhisper":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_menlo(
                    audio_path, model=_model, language=lang
                )
            )
        elif service == "menlo":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_menlo(
                    audio_path, model=_model, language=lang
                )
            )
        elif service == "nvidia":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_parakeet(
                    audio_path, model=_model, language=lang
                )
            )
        elif service == "vllm":
            service_funcs[key] = lambda audio_path, lang, _model=model: transcribe_vllm(
                audio_path, model=_model, language=lang
            )
        elif service == "openai":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_openai(
                    audio_path, model=_model, language=lang
                )
            )
        elif service == "groq":
            service_funcs[key] = lambda audio_path, lang, _model=model: transcribe_groq(
                audio_path, model=_model, language=lang
            )
        elif service == "elevenlabs":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_elevenlabs(
                    audio_path, model_id=_model, language=lang
                )
            )
        elif service == "speechmatics":
            service_funcs[key] = (
                lambda audio_path, lang, _model=model: transcribe_speechmatics(
                    audio_path, language=lang or _model
                )
            )
        elif service == "gladia":
            service_funcs[key] = lambda audio_path, lang: transcribe_gladia(
                audio_path, language=lang
            )
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported service '{service}'")
    return service_funcs


DEFAULT_SERVICE_FUNCTIONS: Dict[str, ServiceFunction] = build_service_function_map(
    DEFAULT_SERVICE_MODELS
)
DEFAULT_SERVICES: List[str] = list(DEFAULT_SERVICE_FUNCTIONS.keys())


__all__ = [
    "ServiceFunction",
    "ServiceModelSpec",
    "transcribe_menlo",
    "transcribe_parakeet",
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
