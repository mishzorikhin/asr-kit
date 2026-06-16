import logging
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline

from app.config import DEFAULT_DEVICE, MODEL_IDLE_TTL_SECONDS, MODEL_UNLOAD_AFTER_REQUEST
from app.errors import OpenAIAPIError, gpu_memory_error, is_gpu_memory_error
from app.services.asr import ASRService
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)


@dataclass
class CachedDiarizationPipeline:
    pipeline: Pipeline
    last_used_at: float
    active_uses: int = 0


class DiarizationService:
    def __init__(self, asr_service: ASRService) -> None:
        self.asr_service = asr_service
        self._lock = threading.Lock()
        self._pipelines: dict[str, CachedDiarizationPipeline] = {}

    def get_pipeline(self, model_path: str) -> Pipeline:
        with self.use_pipeline(model_path) as pipeline:
            return pipeline

    @contextmanager
    def use_pipeline(self, model_path: str):
        with self._lock:
            if model_path not in self._pipelines:
                path = Path(model_path)

                if not (path / "config.yaml").exists():
                    raise OpenAIAPIError(
                        f"Diarization model path not found or invalid: {model_path}",
                        status_code=500,
                        error_type="server_error",
                        param="diarization_model",
                        code="diarization_model_not_found",
                    )

                logger.info("Loading diarization pipeline path=%s", model_path)
                record_tool_call("diarization.pipeline.load", path=model_path)
                try:
                    pipeline = Pipeline.from_pretrained(path)
                    pipeline.to(torch.device(DEFAULT_DEVICE))
                except Exception as exc:
                    if is_gpu_memory_error(exc):
                        self._clear_cuda_cache()
                        raise gpu_memory_error(exc) from exc
                    raise

                self._pipelines[model_path] = CachedDiarizationPipeline(
                    pipeline=pipeline,
                    last_used_at=time.monotonic(),
                )

            cached = self._pipelines[model_path]
            record_tool_call(
                "diarization.pipeline.acquire",
                path=model_path,
                active_uses=cached.active_uses + 1,
            )
            cached.active_uses += 1
            cached.last_used_at = time.monotonic()

        try:
            yield cached.pipeline
        finally:
            evicted: CachedDiarizationPipeline | None = None
            with self._lock:
                cached.active_uses -= 1
                cached.last_used_at = time.monotonic()
                if (
                    MODEL_UNLOAD_AFTER_REQUEST
                    and cached.active_uses == 0
                    and self._pipelines.get(model_path) is cached
                ):
                    evicted = self._pipelines.pop(model_path)

            if evicted is not None:
                record_tool_call("diarization.pipeline.unload_after_request", path=model_path)
                logger.info("Unloading diarization pipeline after request path=%s", model_path)
                del evicted.pipeline
                self._clear_cuda_cache()

    def unload_idle_pipelines(self, max_idle_seconds: int = MODEL_IDLE_TTL_SECONDS) -> int:
        if max_idle_seconds <= 0:
            return 0

        now = time.monotonic()
        evicted: list[tuple[str, CachedDiarizationPipeline]] = []

        with self._lock:
            for model_path, cached in list(self._pipelines.items()):
                if cached.active_uses == 0 and now - cached.last_used_at >= max_idle_seconds:
                    evicted.append((model_path, self._pipelines.pop(model_path)))

        for model_path, cached in evicted:
            record_tool_call("diarization.pipeline.unload_idle", path=model_path)
            logger.info("Unloading idle diarization pipeline path=%s", model_path)
            del cached.pipeline

        if evicted and torch.cuda.is_available():
            self._clear_cuda_cache()

        return len(evicted)

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def normalize_audio(self, input_path: str) -> str:
        record_tool_call("diarization.audio.normalize", input_path=input_path)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-vn",
                    "-f",
                    "wav",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            Path(output_path).unlink(missing_ok=True)
            raise OpenAIAPIError(
                f"Could not normalize audio for diarization: {exc}",
                param="file",
                code="audio_normalization_failed",
            ) from exc

        return output_path

    def diarize(
        self,
        audio_path: str,
        *,
        diarization_model: str,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
        use_exclusive: bool,
    ) -> list[dict[str, Any]]:
        diarization_kwargs = {}

        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                diarization_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_kwargs["max_speakers"] = max_speakers

        logger.info(
            "Diarizing file=%s model=%s kwargs=%s use_exclusive=%s",
            audio_path,
            diarization_model,
            diarization_kwargs,
            use_exclusive,
        )
        record_tool_call(
            "diarization.run",
            model=diarization_model,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
            use_exclusive=use_exclusive,
        )
        try:
            with self.use_pipeline(diarization_model) as pipeline:
                output = pipeline(audio_path, **diarization_kwargs)
        except OpenAIAPIError:
            raise
        except Exception as exc:
            if is_gpu_memory_error(exc):
                self._clear_cuda_cache()
                raise gpu_memory_error(exc) from exc
            raise

        annotation = output.speaker_diarization
        if use_exclusive and hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization

        turns = [
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        logger.info("Diarized file=%s turns=%d", audio_path, len(turns))
        return turns

    def transcribe_with_diarization(
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
        diarization_model: str,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
        use_exclusive: bool,
    ) -> dict[str, Any]:
        transcription = self.asr_service.transcribe(
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
        normalized_audio_path = self.normalize_audio(audio_path)

        try:
            diarization_turns = self.diarize(
                normalized_audio_path,
                diarization_model=diarization_model,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
                use_exclusive=use_exclusive,
            )
        finally:
            Path(normalized_audio_path).unlink(missing_ok=True)

        segments = []
        for segment in transcription["segments"]:
            segments.append(
                {
                    **segment,
                    "speaker": speaker_for_segment(
                        segment_start=segment["start"],
                        segment_end=segment["end"],
                        diarization_turns=diarization_turns,
                    ),
                }
            )

        return {
            **transcription,
            "diarization": diarization_turns,
            "segments": segments,
        }


def segment_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def speaker_for_segment(
    segment_start: float,
    segment_end: float,
    diarization_turns: list[dict[str, Any]],
) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn in diarization_turns:
        overlap = segment_overlap(
            segment_start,
            segment_end,
            turn["start"],
            turn["end"],
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker
