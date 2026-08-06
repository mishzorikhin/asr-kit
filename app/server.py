"""FastAPI application factory for the OpenAI-compatible ASR API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.config import resolve_compute_type, resolve_device
from app.errors import OpenAIAPIError, openai_error_handler, validation_error_handler
from app.logging_config import configure_logging
from app.model_registry import ModelRegistry
from app.routers import audio, health, models, realtime, ui
from app.services.asr import ASRService
from app.services.diarization import DiarizationService
from app.services.model_unloader import ModelUnloader

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and wire the FastAPI application.

    Returns:
        Configured ``FastAPI`` instance with routers and shared services.
    """
    configure_logging()

    model_registry = ModelRegistry()
    asr_service = ASRService(model_registry)
    diarization_service = DiarizationService(asr_service)
    model_unloader = ModelUnloader(asr_service, diarization_service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        device = resolve_device()
        compute_type = resolve_compute_type(device=device)
        logger.info("ASR runtime device=%s compute_type=%s", device, compute_type)
        model_unloader.start()
        try:
            yield
        finally:
            model_unloader.stop()

    app = FastAPI(
        title="OpenAI-compatible ASR API",
        description=(
            "Local OpenAI-compatible audio transcription API powered by "
            "faster-whisper, optional NVIDIA NeMo ASR, and optional pyannote "
            "diarization.\n\n"
            "Compatibility notes:\n"
            "- Use `base_url=http://<host>:<port>/v1` with the OpenAI Python SDK.\n"
            "- Implemented OpenAI-style endpoints: `GET /v1/models`, "
            "`GET /v1/models/{model}`, `POST /v1/audio/transcriptions`, and "
            "`WS /v1/realtime`.\n"
            "- `POST /v1/audio/translations` is not implemented.\n"
            "- Streaming transcription responses are not implemented; `stream=true` "
            "returns an OpenAI-style `unsupported_parameter` error.\n"
            "- `WS /v1/realtime` provides pseudo-realtime transcription over WebSocket "
            "(PCM16 mono @ 16 kHz, server-side VAD) with optional progressive "
            "detailization: word timestamps, provisional speaker labels, and "
            "optional session-final pyannote diarization triggered by `session.end`. "
            "Supports both `faster-whisper` and `nemo` backends; NeMo word timestamps "
            "are not filled yet (transcript text still works).\n"
            "- This is a local subset of the OpenAI Realtime API and does not include "
            "voice agents, TTS, or tools.\n"
            "- `include=logprobs` is accepted for validation compatibility but is not "
            "implemented by the ASR backends.\n"
            "- `diarized_json` and diarization controls are local extensions. Use a "
            "model with diarization capability, for example `*-diarize`.\n"
            "- Diarization-capable models currently return only `json`, `text`, and "
            "`diarized_json` response formats."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.model_registry = model_registry
    app.state.asr_service = asr_service
    app.state.diarization_service = diarization_service
    app.state.model_unloader = model_unloader

    app.add_exception_handler(OpenAIAPIError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(audio.router)
    app.include_router(realtime.router)
    app.include_router(ui.router)

    return app


app = create_app()
