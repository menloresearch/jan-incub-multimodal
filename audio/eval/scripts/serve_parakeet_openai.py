#!/usr/bin/env python3
"""Serve NVIDIA Parakeet ASR via an OpenAI-compatible FastAPI server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from nemo.collections.asr.models import ASRModel

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_MODEL_FILENAME = "parakeet-tdt-0.6b-v3.nemo"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class ServerConfig:
    model_id: str
    model_filename: str
    model_path: Optional[Path]
    device: torch.device
    batch_size: int
    api_key: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight OpenAI-compatible server backed by NVIDIA's Parakeet ASR model."
        )
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("PARAKEET_MODEL_ID", DEFAULT_MODEL_ID),
        help="Hugging Face repo id or NeMo model name to download (default: %(default)s)",
    )
    parser.add_argument(
        "--model-filename",
        default=os.getenv("PARAKEET_MODEL_FILENAME", DEFAULT_MODEL_FILENAME),
        help="Checkpoint filename inside the repo when downloading (default: %(default)s)",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=os.getenv("PARAKEET_MODEL_FILE"),
        help="Local .nemo checkpoint path (skips download when provided)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("PARAKEET_DEVICE", "auto"),
        help="Torch device to run inference on (auto, cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("PARAKEET_BATCH_SIZE", "1")),
        help="Batch size used by NeMo's transcribe helper (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PARAKEET_API_KEY"),
        help="Optional API key to require via Bearer token (default: env PARAKEET_API_KEY)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("PARAKEET_HOST", DEFAULT_HOST),
        help="Interface to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PARAKEET_PORT", str(DEFAULT_PORT))),
        help="Port to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("PARAKEET_LOG_LEVEL", "info"),
        help="Logging level passed to uvicorn (default: %(default)s)",
    )
    return parser.parse_args()


def resolve_device(device_option: str) -> torch.device:
    device_option = (device_option or "auto").lower()
    if device_option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(device_option)
    except (TypeError, RuntimeError) as exc:  # pragma: no cover - defensive branch
        raise SystemExit(f"Unsupported device specification: {device_option}") from exc


def download_model(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    logging.info("Downloading %s from %s", filename, repo_id)
    download_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(download_path)


def load_model(config: ServerConfig) -> ASRModel:
    model_path = config.model_path or download_model(
        repo_id=config.model_id, filename=config.model_filename
    )
    if not model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {model_path}")

    logging.info("Loading NeMo checkpoint from %s", model_path)
    model = ASRModel.restore_from(
        restore_path=str(model_path), map_location=config.device
    )
    model = model.to(config.device)
    model.eval()
    model.freeze()
    logging.info("Model loaded on device %s", config.device)
    return model


def build_auth_dependency(config: ServerConfig):
    async def verify_api_key(request: Request) -> None:
        if not config.api_key:
            return
        header = request.headers.get("Authorization")
        if not header or not header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        token = header.split(" ", 1)[1].strip()
        if token != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return verify_api_key


def create_app(config: ServerConfig, model: ASRModel) -> FastAPI:
    app = FastAPI(title="Parakeet OpenAI Bridge", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    app.state.config = config
    app.state.model = model
    auth_dependency = Depends(build_auth_dependency(config))

    @app.get("/health")
    async def health() -> dict[str, str]:  # pragma: no cover - simple health endpoint
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[auth_dependency])
    async def list_models() -> dict[str, list[dict[str, str]]]:
        return {
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ]
        }

    @app.post("/v1/audio/transcriptions", dependencies=[auth_dependency])
    async def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        model_name: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
        response_format: str = Form("json"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(None),
    ):  # noqa: D401 - FastAPI handles docs
        del request, prompt, temperature  # not used but kept for API compatibility
        asr_model = app.state.model
        resolved_model = model_name or model
        if not resolved_model:
            raise HTTPException(status_code=400, detail="Missing model name")
        if resolved_model not in {config.model_id, Path(config.model_filename).stem}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model '{resolved_model}'. Available: {config.model_id}",
            )

        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp.flush()
            temp_path = tmp.name

        try:
            transcripts = await asyncio.to_thread(
                asr_model.transcribe,
                temp_path,
                batch_size=config.batch_size,
                return_hypotheses=True,
                verbose=False,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:  # pragma: no cover - best effort cleanup
                logging.warning("Could not delete temporary file %s", temp_path)

        if not transcripts:
            raise HTTPException(status_code=500, detail="No transcription returned")

        first_hypothesis = transcripts[0]
        if hasattr(first_hypothesis, "text"):
            text = str(first_hypothesis.text).strip()
        else:
            text = str(first_hypothesis).strip()
        response = {
            "id": f"transcription-{uuid.uuid4().hex}",
            "object": "transcription",
            "created": int(time.time()),
            "model": config.model_id,
            "text": text,
            "language": language,
        }

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format in {"json", "verbose_json"}:
            return JSONResponse(response)
        raise HTTPException(
            status_code=400,
            detail="Unsupported response_format; expected 'json', 'verbose_json', or 'text'",
        )

    return app


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = resolve_device(args.device)

    config = ServerConfig(
        model_id=args.model_id,
        model_filename=args.model_filename,
        model_path=Path(args.model_file).expanduser().resolve()
        if args.model_file
        else None,
        device=device,
        batch_size=args.batch_size,
        api_key=args.api_key,
    )

    model = load_model(config)
    app = create_app(config, model)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
