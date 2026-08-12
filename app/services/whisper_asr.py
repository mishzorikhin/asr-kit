import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from faster_whisper import WhisperModel

from app.config import (
    MODEL_DIR,
    MODEL_IDLE_TTL_SECONDS,
    MODEL_UNLOAD_AFTER_REQUEST,
    WHISPER_AUTOSCALE_ENABLED,
    WHISPER_MAX_REPLICAS,
    WHISPER_REPLICA_WAIT_SECONDS,
    resolve_compute_type,
    resolve_device,
)
from app.errors import OpenAIAPIError, gpu_memory_error, is_gpu_memory_error
from app.model_registry import ModelRegistry, resolve_asr_model_path
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)

PoolKey = tuple[str, str, str]


@dataclass
class CachedASRModel:
    model: WhisperModel
    last_used_at: float
    active_uses: int = 0
    replica_id: int = 0
    device_index: int = 0


class WhisperASRService:
    """faster-whisper backend with optional in-process replica autoscaling.

    When WHISPER_AUTOSCALE_ENABLED is on, each replica serves one inference at a
    time. A second concurrent request tries to spawn another WhisperModel (new
    GPU index when available) up to WHISPER_MAX_REPLICAS; otherwise it waits for
    a free replica.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pools: dict[PoolKey, list[CachedASRModel]] = {}
        self._pending_loads: dict[PoolKey, int] = {}
        self._next_replica_id = 0

    def get_model(self, model_path: str, device: str, compute_type: str) -> WhisperModel:
        with self.use_model(model_path, device, compute_type) as model:
            return model

    def replica_status(self) -> list[dict[str, Any]]:
        with self._lock:
            status: list[dict[str, Any]] = []
            for (path, device, compute_type), replicas in self._pools.items():
                for cached in replicas:
                    status.append(
                        {
                            "path": path,
                            "device": device,
                            "compute_type": compute_type,
                            "replica_id": cached.replica_id,
                            "device_index": cached.device_index,
                            "active_uses": cached.active_uses,
                            "busy": cached.active_uses > 0,
                        }
                    )
            return status

    def _cuda_device_count(self) -> int:
        if torch.cuda.is_available():
            return max(1, int(torch.cuda.device_count()))
        return 1

    def _pick_device_index(self, device: str, pool: list[CachedASRModel]) -> int:
        if device != "cuda":
            return 0
        gpu_count = self._cuda_device_count()
        counts = {index: 0 for index in range(gpu_count)}
        for replica in pool:
            counts[replica.device_index % gpu_count] += 1
        return min(counts, key=counts.get)

    def _can_scale(self, key: PoolKey, pool: list[CachedASRModel]) -> bool:
        if not WHISPER_AUTOSCALE_ENABLED:
            return len(pool) == 0 and self._pending_loads.get(key, 0) == 0
        current = len(pool) + self._pending_loads.get(key, 0)
        return current < WHISPER_MAX_REPLICAS

    def _find_idle(self, pool: list[CachedASRModel]) -> CachedASRModel | None:
        idle = [replica for replica in pool if replica.active_uses == 0]
        if not idle:
            return None
        idle.sort(key=lambda replica: replica.last_used_at)
        return idle[0]

    def _find_shareable(self, pool: list[CachedASRModel]) -> CachedASRModel | None:
        """Legacy path when autoscaling is off: share one model across requests."""
        if not pool:
            return None
        pool_sorted = sorted(pool, key=lambda replica: (replica.active_uses, replica.last_used_at))
        return pool_sorted[0]

    def _load_replica(
        self,
        *,
        resolved_path: str,
        device: str,
        compute_type: str,
        device_index: int,
        replica_id: int,
    ) -> CachedASRModel:
        load_kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "download_root": str(MODEL_DIR),
            "local_files_only": True,
        }
        if device == "cuda":
            load_kwargs["device_index"] = device_index

        record_tool_call(
            "asr.model.load",
            path=resolved_path,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
            replica_id=replica_id,
        )
        logger.info(
            "Loading ASR model path=%s device=%s compute_type=%s device_index=%s replica_id=%s",
            resolved_path,
            device,
            compute_type,
            device_index,
            replica_id,
        )
        model = WhisperModel(resolved_path, **load_kwargs)
        return CachedASRModel(
            model=model,
            last_used_at=time.monotonic(),
            replica_id=replica_id,
            device_index=device_index,
        )

    def _acquire_locked(self, cached: CachedASRModel, key: PoolKey) -> CachedASRModel:
        cached.active_uses += 1
        cached.last_used_at = time.monotonic()
        record_tool_call(
            "asr.model.acquire",
            path=key[0],
            device=key[1],
            compute_type=key[2],
            replica_id=cached.replica_id,
            device_index=cached.device_index,
            active_uses=cached.active_uses,
            replicas=len(self._pools.get(key, [])),
        )
        return cached

    def _release_locked(self, cached: CachedASRModel, key: PoolKey) -> CachedASRModel | None:
        cached.active_uses -= 1
        cached.last_used_at = time.monotonic()
        pool = self._pools.get(key)
        if pool is None:
            return None

        record_tool_call(
            "asr.model.release",
            path=key[0],
            device=key[1],
            compute_type=key[2],
            replica_id=cached.replica_id,
            active_uses=cached.active_uses,
            replicas=len(pool),
        )

        evicted: CachedASRModel | None = None
        if (
            MODEL_UNLOAD_AFTER_REQUEST
            and cached.active_uses == 0
            and cached in pool
        ):
            pool.remove(cached)
            evicted = cached
            if not pool:
                self._pools.pop(key, None)

        self._condition.notify_all()
        return evicted

    @contextmanager
    def use_model(self, model_path: str, device: str, compute_type: str) -> Iterator[WhisperModel]:
        device = resolve_device(device)
        compute_type = resolve_compute_type(compute_type, device=device)
        resolved_path = resolve_asr_model_path(model_path)
        key: PoolKey = (resolved_path, device, compute_type)

        cached: CachedASRModel | None = None
        scale_attempt: dict[str, Any] | None = None
        deadline = (
            None
            if WHISPER_REPLICA_WAIT_SECONDS <= 0
            else time.monotonic() + WHISPER_REPLICA_WAIT_SECONDS
        )

        with self._condition:
            while cached is None:
                pool = self._pools.setdefault(key, [])
                idle = self._find_idle(pool)
                if idle is not None:
                    cached = self._acquire_locked(idle, key)
                    break

                if self._can_scale(key, pool):
                    device_index = self._pick_device_index(device, pool)
                    replica_id = self._next_replica_id
                    self._next_replica_id += 1
                    self._pending_loads[key] = self._pending_loads.get(key, 0) + 1
                    scale_attempt = {
                        "device_index": device_index,
                        "replica_id": replica_id,
                    }
                    record_tool_call(
                        "asr.model.autoscale",
                        path=resolved_path,
                        device=device,
                        compute_type=compute_type,
                        device_index=device_index,
                        replica_id=replica_id,
                        replicas=len(pool),
                        max_replicas=WHISPER_MAX_REPLICAS,
                    )
                    break

                if not WHISPER_AUTOSCALE_ENABLED:
                    shareable = self._find_shareable(pool)
                    if shareable is not None:
                        cached = self._acquire_locked(shareable, key)
                        break

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OpenAIAPIError(
                            (
                                "All Whisper replicas are busy and no capacity remains "
                                f"(max_replicas={WHISPER_MAX_REPLICAS})."
                            ),
                            status_code=503,
                            error_type="server_error",
                            code="asr_replicas_busy",
                        )
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()

        if scale_attempt is not None:
            try:
                loaded = self._load_replica(
                    resolved_path=resolved_path,
                    device=device,
                    compute_type=compute_type,
                    device_index=scale_attempt["device_index"],
                    replica_id=scale_attempt["replica_id"],
                )
            except Exception as exc:
                with self._condition:
                    pending = self._pending_loads.get(key, 0) - 1
                    if pending > 0:
                        self._pending_loads[key] = pending
                    else:
                        self._pending_loads.pop(key, None)
                    self._condition.notify_all()

                if is_gpu_memory_error(exc):
                    self._clear_cuda_cache()
                    with self._condition:
                        pool = self._pools.get(key, [])
                        if pool:
                            # Fall back to waiting/sharing existing replicas.
                            while True:
                                pool = self._pools.get(key, [])
                                if not pool:
                                    raise gpu_memory_error(exc) from exc
                                idle = self._find_idle(pool)
                                if idle is not None:
                                    cached = self._acquire_locked(idle, key)
                                    break
                                if not WHISPER_AUTOSCALE_ENABLED:
                                    shareable = self._find_shareable(pool)
                                    if shareable is not None:
                                        cached = self._acquire_locked(shareable, key)
                                        break
                                if deadline is not None:
                                    remaining = deadline - time.monotonic()
                                    if remaining <= 0:
                                        raise gpu_memory_error(exc) from exc
                                    self._condition.wait(timeout=remaining)
                                else:
                                    self._condition.wait()
                        else:
                            raise gpu_memory_error(exc) from exc
                else:
                    raise
            else:
                with self._condition:
                    pending = self._pending_loads.get(key, 0) - 1
                    if pending > 0:
                        self._pending_loads[key] = pending
                    else:
                        self._pending_loads.pop(key, None)
                    pool = self._pools.setdefault(key, [])
                    pool.append(loaded)
                    cached = self._acquire_locked(loaded, key)

        assert cached is not None

        try:
            yield cached.model
        finally:
            evicted: CachedASRModel | None = None
            with self._condition:
                evicted = self._release_locked(cached, key)

            if evicted is not None:
                record_tool_call(
                    "asr.model.unload_after_request",
                    path=key[0],
                    device=key[1],
                    compute_type=key[2],
                    replica_id=evicted.replica_id,
                    device_index=evicted.device_index,
                )
                logger.info(
                    "Unloading ASR replica after request path=%s device=%s "
                    "compute_type=%s replica_id=%s",
                    key[0],
                    key[1],
                    key[2],
                    evicted.replica_id,
                )
                del evicted.model
                self._clear_cuda_cache()

    def unload_idle_models(self, max_idle_seconds: int = MODEL_IDLE_TTL_SECONDS) -> int:
        if max_idle_seconds <= 0:
            return 0

        now = time.monotonic()
        evicted: list[tuple[PoolKey, CachedASRModel]] = []

        with self._condition:
            for key, pool in list(self._pools.items()):
                remaining: list[CachedASRModel] = []
                for cached in pool:
                    if cached.active_uses == 0 and now - cached.last_used_at >= max_idle_seconds:
                        evicted.append((key, cached))
                    else:
                        remaining.append(cached)
                if remaining:
                    self._pools[key] = remaining
                else:
                    self._pools.pop(key, None)
            if evicted:
                self._condition.notify_all()

        for key, cached in evicted:
            record_tool_call(
                "asr.model.unload_idle",
                path=key[0],
                device=key[1],
                compute_type=key[2],
                replica_id=cached.replica_id,
                device_index=cached.device_index,
            )
            logger.info(
                "Unloading idle ASR replica path=%s device=%s compute_type=%s replica_id=%s",
                key[0],
                key[1],
                key[2],
                cached.replica_id,
            )
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
        timestamp_granularities: list[str] | None = None,
    ) -> dict[str, Any]:
        configured_model = self.registry.get(model_id)
        word_timestamps = bool(timestamp_granularities and "word" in timestamp_granularities)
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
            "Transcribing array model=%s language=%s samples=%d beam_size=%s vad_filter=%s word_timestamps=%s",
            model_id,
            language,
            samples.size,
            beam_size,
            vad_filter,
            word_timestamps,
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
            "words": words,
            "text": text,
        }
