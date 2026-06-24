import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel

from app.config import (
    MODEL_DIR,
    MODEL_IDLE_TTL_SECONDS,
    MODEL_UNLOAD_AFTER_REQUEST,
    resolve_compute_type,
    resolve_device,
)
from app.errors import OpenAIAPIError, gpu_memory_error, is_gpu_memory_error
from app.model_registry import ModelRegistry, resolve_asr_model_path
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)


@dataclass
class CachedASRModel:
    model: WhisperModel
    last_used_at: float
    active_uses: int = 0


class ASRService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str, str], CachedASRModel] = {}

    def get_model(self, model_path: str, device: str, compute_type: str) -> WhisperModel:
        with self.use_model(model_path, device, compute_type) as model:
            return model

    @contextmanager
    def use_model(self, model_path: str, device: str, compute_type: str):
        device = resolve_device(device)
        compute_type = resolve_compute_type(compute_type, device=device)
        resolved_path = resolve_asr_model_path(model_path)
        key = (resolved_path, device, compute_type)

        with self._lock:
            if key not in self._models:
                record_tool_call(
                    "asr.model.load",
                    path=resolved_path,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info(
                    "Loading ASR model path=%s device=%s compute_type=%s",
                    resolved_path,
                    device,
                    compute_type,
                )
                try:
                    model = WhisperModel(
                        resolved_path,
                        device=device,
                        compute_type=compute_type,
                        download_root=str(MODEL_DIR),
                        local_files_only=True,
                    )
                except Exception as exc:
                    if is_gpu_memory_error(exc):
                        self._clear_cuda_cache()
                        raise gpu_memory_error(exc) from exc
                    raise

                self._models[key] = CachedASRModel(
                    model=model,
                    last_used_at=time.monotonic(),
                )

            cached = self._models[key]
            record_tool_call(
                "asr.model.acquire",
                path=resolved_path,
                device=device,
                compute_type=compute_type,
                active_uses=cached.active_uses + 1,
            )
            cached.active_uses += 1
            cached.last_used_at = time.monotonic()

        try:
            yield cached.model
        finally:
            evicted: CachedASRModel | None = None
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
                    "asr.model.unload_after_request",
                    path=key[0],
                    device=key[1],
                    compute_type=key[2],
                )
                logger.info(
                    "Unloading ASR model after request path=%s device=%s compute_type=%s",
                    *key,
                )
                del evicted.model
                self._clear_cuda_cache()

    def unload_idle_models(self, max_idle_seconds: int = MODEL_IDLE_TTL_SECONDS) -> int:
        if max_idle_seconds <= 0:
            return 0

        now = time.monotonic()
        evicted: list[tuple[tuple[str, str, str], CachedASRModel]] = []

        with self._lock:
            for key, cached in list(self._models.items()):
                if cached.active_uses == 0 and now - cached.last_used_at >= max_idle_seconds:
                    evicted.append((key, self._models.pop(key)))

        for key, cached in evicted:
            record_tool_call(
                "asr.model.unload_idle",
                path=key[0],
                device=key[1],
                compute_type=key[2],
            )
            logger.info("Unloading idle ASR model path=%s device=%s compute_type=%s", *key)
            del cached.model

        if evicted and torch.cuda.is_available():
            self._clear_cuda_cache()

        return len(evicted)

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        word_timestamps = "word" in timestamp_granularities

        logger.info(
            "Transcribing file=%s model=%s language=%s beam_size=%s vad_filter=%s word_timestamps=%s",
            audio_path,
            model_id,
            language,
            beam_size,
            vad_filter,
            word_timestamps,
        )

        try:
            record_tool_call(
                "asr.transcribe",
                model=model_id,
                language=language,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            with self.use_model(configured_model["path"], device, compute_type) as model:
                segments_iter, info = model.transcribe(
                    audio_path,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                )
                segments = []
                words = []

                for index, segment in enumerate(segments_iter):
                    segment_words = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                        for word in (getattr(segment, "words", None) or [])
                    ]
                    words.extend(segment_words)
                    segments.append(
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip(),
                            "seek": getattr(segment, "seek", 0),
                            "tokens": list(getattr(segment, "tokens", []) or []),
                            "temperature": temperature,
                            "avg_logprob": getattr(segment, "avg_logprob", 0.0),
                            "compression_ratio": getattr(segment, "compression_ratio", 0.0),
                            "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                            "words": segment_words,
                        }
                    )
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
            "Transcribed file=%s model=%s duration=%s segments=%d",
            audio_path,
            model_id,
            info.duration,
            len(segments),
        )
        return {
            "model": model_id,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
            "words": words,
        }

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

        logger.info(
            "Transcribing array model=%s language=%s samples=%d beam_size=%s vad_filter=%s",
            model_id,
            language,
            samples.size,
            beam_size,
            vad_filter,
        )

        try:
            record_tool_call(
                "asr.transcribe_array",
                model=model_id,
                language=language,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                samples=samples.size,
            )
            with self.use_model(configured_model["path"], device, compute_type) as model:
                segments_iter, info = model.transcribe(
                    samples,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=False,
                )
                segments = []
                for index, segment in enumerate(segments_iter):
                    segments.append(
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip(),
                        }
                    )
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

        text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
        logger.info(
            "Transcribed array model=%s duration=%s segments=%d text_chars=%d",
            model_id,
            info.duration,
            len(segments),
            len(text),
        )
        return {
            "model": model_id,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
            "text": text,
        }
