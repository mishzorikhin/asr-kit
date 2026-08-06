"""NVIDIA NeMo ASR backend.

Optional dependency: requires ``nemo_toolkit[asr]``. Whisper-only deployments
keep this module importable; construction fails clearly when NeMo is missing.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import (
    DEFAULT_LANGUAGE,
    MODEL_IDLE_TTL_SECONDS,
    MODEL_UNLOAD_AFTER_REQUEST,
    resolve_device,
)
from app.errors import OpenAIAPIError, gpu_memory_error, is_gpu_memory_error
from app.model_registry import ModelRegistry
from app.services.wav_io import audio_duration_seconds as _audio_duration_seconds
from app.services.wav_io import write_temp_wav as _write_temp_wav
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import ASRModel

    NEMO_AVAILABLE = True
except ImportError:  # whisper-only deployments
    ASRModel = Any  # type: ignore[misc, assignment]
    NEMO_AVAILABLE = False


@dataclass
class CachedNeMoModel:
    """In-memory cache entry for a loaded NeMo ASR model."""

    model: Any
    last_used_at: float
    active_uses: int = 0


def _hypothesis_text(hyp: Any) -> str:
    """Normalize a NeMo hypothesis object into plain text."""
    if hyp is None:
        return ""
    if isinstance(hyp, str):
        return hyp.strip()
    text = getattr(hyp, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(hyp, (list, tuple)) and hyp:
        return _hypothesis_text(hyp[0])
    return str(hyp).strip()


class NeMoASRService:
    def __init__(self, registry: ModelRegistry) -> None:
        if not NEMO_AVAILABLE:
            raise RuntimeError(
                "nemo_toolkit is not installed; cannot use backend=nemo. "
                "Install requirements.nemo.txt or remove nemo models from models.yaml."
            )
        self.registry = registry
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str], CachedNeMoModel] = {}

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def use_model(self, model_path: str, device: str, compute_type: str = "float32"):
        # compute_type is accepted for API compatibility with Whisper; NeMo uses torch dtype.
        del compute_type
        device = resolve_device(device)
        map_location = torch.device(device)
        key = (model_path, device)

        with self._lock:
            if key not in self._models:
                record_tool_call("asr.nemo.model.load", path=model_path, device=device)
                logger.info("Loading NeMo ASR model path=%s device=%s", model_path, device)
                try:
                    model = ASRModel.restore_from(
                        restore_path=model_path,
                        map_location=map_location,
                    )
                    model.eval()
                except Exception as exc:
                    if is_gpu_memory_error(exc):
                        self._clear_cuda_cache()
                        raise gpu_memory_error(exc) from exc
                    raise

                self._models[key] = CachedNeMoModel(
                    model=model,
                    last_used_at=time.monotonic(),
                )

            cached = self._models[key]
            record_tool_call(
                "asr.nemo.model.acquire",
                path=model_path,
                device=device,
                active_uses=cached.active_uses + 1,
            )
            cached.active_uses += 1
            cached.last_used_at = time.monotonic()

        try:
            yield cached.model
        finally:
            evicted: CachedNeMoModel | None = None
            with self._lock:
                cached.active_uses -= 1
                cached.last_used_at = time.monotonic()
                if (
                    MODEL_UNLOAD_AFTER_REQUEST
                    and cached.active_uses == 0
                    and self._models.get(key) is cached
                ):
                    evicted = self._models.pop(key)

            if evicted is not None:
                record_tool_call(
                    "asr.nemo.model.unload_after_request",
                    path=key[0],
                    device=key[1],
                )
                logger.info("Unloading NeMo ASR model after request path=%s device=%s", *key)
                del evicted.model
                self._clear_cuda_cache()

    def unload_idle_models(self, max_idle_seconds: int = MODEL_IDLE_TTL_SECONDS) -> int:
        if max_idle_seconds <= 0:
            return 0

        now = time.monotonic()
        evicted: list[tuple[tuple[str, str], CachedNeMoModel]] = []

        with self._lock:
            for key, cached in list(self._models.items()):
                if cached.active_uses == 0 and now - cached.last_used_at >= max_idle_seconds:
                    evicted.append((key, self._models.pop(key)))

        for key, cached in evicted:
            record_tool_call(
                "asr.nemo.model.unload_idle",
                path=key[0],
                device=key[1],
            )
            logger.info("Unloading idle NeMo ASR model path=%s device=%s", *key)
            del cached.model

        if evicted and torch.cuda.is_available():
            self._clear_cuda_cache()

        return len(evicted)

    def _run_transcribe(self, model: Any, audio_path: str) -> str:
        hypotheses = model.transcribe([audio_path], batch_size=1)
        if not hypotheses:
            return ""
        return _hypothesis_text(hypotheses[0])

    def _build_result(
        self,
        *,
        model_id: str,
        language: str | None,
        temperature: float,
        duration: float,
        text: str,
        include_text_field: bool = False,
    ) -> dict[str, Any]:
        text = text.strip()
        segment = {
            "id": 0,
            "start": 0.0,
            "end": duration,
            "text": text,
            "seek": 0,
            "tokens": [],
            "temperature": temperature,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
            "words": [],
        }
        result: dict[str, Any] = {
            "model": model_id,
            "language": language or DEFAULT_LANGUAGE,
            "language_probability": 1.0 if language else 0.0,
            "duration": duration,
            "segments": [segment] if text else [],
            "words": [],
        }
        if include_text_field:
            result["text"] = text
        return result

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
        configured_model = self.registry.get(model_id)
        if "word" in timestamp_granularities:
            logger.info(
                "NeMo ASR does not provide word timestamps yet; returning empty words model=%s",
                model_id,
            )

        if prompt or vad_filter:
            logger.info(
                "NeMo ASR ignores whisper-oriented knobs model=%s prompt=%s "
                "vad_filter=%s beam_size=%s compute_type=%s",
                model_id,
                bool(prompt),
                vad_filter,
                beam_size,
                compute_type,
            )

        logger.info(
            "Transcribing file with NeMo file=%s model=%s language=%s",
            audio_path,
            model_id,
            language,
        )

        try:
            record_tool_call(
                "asr.nemo.transcribe",
                model=model_id,
                language=language,
                device=device,
            )
            with self.use_model(configured_model["path"], device, compute_type) as model:
                text = self._run_transcribe(model, audio_path)
                duration = _audio_duration_seconds(audio_path)
        except OpenAIAPIError:
            raise
        except Exception as exc:
            if is_gpu_memory_error(exc):
                self._clear_cuda_cache()
                raise gpu_memory_error(exc) from exc
            raise OpenAIAPIError(
                f"Could not decode or transcribe audio file: {exc}",
                param="file",
                code="audio_decode_failed",
            ) from exc

        logger.info(
            "Transcribed file with NeMo file=%s model=%s duration=%s text_chars=%d",
            audio_path,
            model_id,
            duration,
            len(text),
        )
        return self._build_result(
            model_id=model_id,
            language=language,
            temperature=temperature,
            duration=duration,
            text=text,
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
        configured_model = self.registry.get(model_id)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)

        if samples.size == 0:
            raise OpenAIAPIError(
                "Audio buffer is empty",
                param="audio",
                code="empty_audio",
            )

        if sample_rate != 16000:
            raise OpenAIAPIError(
                f"Unsupported sample_rate {sample_rate}; realtime audio must be 16 kHz PCM16 mono.",
                param="sample_rate",
                code="unsupported_audio_format",
            )

        if timestamp_granularities and "word" in timestamp_granularities:
            logger.info(
                "NeMo ASR does not provide word timestamps yet; returning empty words model=%s",
                model_id,
            )

        if prompt or vad_filter:
            logger.info(
                "NeMo ASR ignores whisper-oriented knobs for array model=%s "
                "prompt=%s vad_filter=%s beam_size=%s",
                model_id,
                bool(prompt),
                vad_filter,
                beam_size,
            )

        duration = samples.size / float(sample_rate)
        tmp_path: str | None = None

        logger.info(
            "Transcribing array with NeMo model=%s language=%s samples=%d",
            model_id,
            language,
            samples.size,
        )

        try:
            record_tool_call(
                "asr.nemo.transcribe_array",
                model=model_id,
                language=language,
                device=device,
                samples=samples.size,
            )
            tmp_path = _write_temp_wav(samples, sample_rate)
            with self.use_model(configured_model["path"], device, compute_type) as model:
                text = self._run_transcribe(model, tmp_path)
        except OpenAIAPIError:
            raise
        except Exception as exc:
            if is_gpu_memory_error(exc):
                self._clear_cuda_cache()
                raise gpu_memory_error(exc) from exc
            raise OpenAIAPIError(
                f"Could not transcribe audio buffer: {exc}",
                param="audio",
                code="audio_decode_failed",
            ) from exc
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

        logger.info(
            "Transcribed array with NeMo model=%s duration=%s text_chars=%d",
            model_id,
            duration,
            len(text),
        )
        return self._build_result(
            model_id=model_id,
            language=language,
            temperature=temperature,
            duration=duration,
            text=text,
            include_text_field=True,
        )
