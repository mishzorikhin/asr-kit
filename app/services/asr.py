"""ASR facade: dispatches to faster-whisper or NeMo by model backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import MODEL_IDLE_TTL_SECONDS
from app.errors import OpenAIAPIError
from app.model_registry import BACKEND_FASTER_WHISPER, BACKEND_NEMO, ModelRegistry
from app.services.whisper_asr import WhisperASRService

logger = logging.getLogger(__name__)


class ASRService:
    """Unified ASR entry point used by REST, realtime, and diarization."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._whisper = WhisperASRService(registry)
        self._nemo: Any | None = None

    def _nemo_service(self):
        if self._nemo is None:
            from app.services.nemo_asr import NEMO_AVAILABLE, NeMoASRService

            if not NEMO_AVAILABLE:
                raise OpenAIAPIError(
                    "NeMo backend requested but nemo_toolkit is not installed. "
                    "Install requirements.nemo.txt or use a faster-whisper model.",
                    status_code=500,
                    error_type="server_error",
                    param="model",
                    code="backend_unavailable",
                )
            self._nemo = NeMoASRService(self.registry)
        return self._nemo

    def _backend_for(self, model_id: str):
        configured = self.registry.get(model_id)
        backend = configured.get("backend", BACKEND_FASTER_WHISPER)
        if backend == BACKEND_FASTER_WHISPER:
            return self._whisper
        if backend == BACKEND_NEMO:
            return self._nemo_service()
        raise OpenAIAPIError(
            f"Unsupported ASR backend '{backend}' for model '{model_id}'.",
            param="model",
            code="unsupported_backend",
        )

    def transcribe(
        self,
        audio_path: str,
        *,
        model_id: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
        timestamp_granularities: list[str],
    ) -> dict[str, Any]:
        return self._backend_for(model_id).transcribe(
            audio_path,
            model_id=model_id,
            language=language,
            prompt=prompt,
            temperature=temperature,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            timestamp_granularities=timestamp_granularities,
        )

    def transcribe_array(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        model_id: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool = False,
        timestamp_granularities: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._backend_for(model_id).transcribe_array(
            audio,
            sample_rate=sample_rate,
            model_id=model_id,
            language=language,
            prompt=prompt,
            temperature=temperature,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            timestamp_granularities=timestamp_granularities,
        )

    def unload_idle_models(self, max_idle_seconds: int = MODEL_IDLE_TTL_SECONDS) -> int:
        unloaded = self._whisper.unload_idle_models(max_idle_seconds)
        if self._nemo is not None:
            unloaded += self._nemo.unload_idle_models(max_idle_seconds)
        return unloaded
