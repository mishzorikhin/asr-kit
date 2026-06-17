from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    REALTIME_BEAM_SIZE,
    REALTIME_INITIAL_PROMPT_CHARS,
    REALTIME_MAX_BUFFER_SEC,
    REALTIME_MIN_SEGMENT_MS,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION_MS,
    REALTIME_VAD_THRESHOLD,
)
from app.errors import OpenAIAPIError
from app.openai_realtime_events import (
    buffer_committed_event,
    default_session_config,
    error_event,
    session_updated_event,
    speech_started_event,
    speech_stopped_event,
    transcription_completed_event,
    transcription_delta_event,
)
from app.services.asr import ASRService
from app.services.audio_buffer import AudioBuffer, chunk_rms
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)

SendEvent = Callable[[dict[str, Any]], Awaitable[None]]


class RealtimeSession:
    def __init__(
        self,
        *,
        asr_service: ASRService,
        model_id: str,
        send_event: SendEvent,
        language: str = DEFAULT_LANGUAGE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        sample_rate: int = REALTIME_SAMPLE_RATE,
        max_buffer_sec: float = REALTIME_MAX_BUFFER_SEC,
        vad_threshold: float = REALTIME_VAD_THRESHOLD,
        silence_duration_ms: int = REALTIME_SILENCE_DURATION_MS,
        min_segment_ms: int = REALTIME_MIN_SEGMENT_MS,
        beam_size: int = REALTIME_BEAM_SIZE,
    ) -> None:
        self.asr_service = asr_service
        self.model_id = model_id
        self.send_event = send_event
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_segment_ms = min_segment_ms
        self.beam_size = beam_size
        self.input_audio_format = "pcm16"

        self.session_config = default_session_config(
            model_id=model_id,
            language=language,
            sample_rate=sample_rate,
            input_audio_format=self.input_audio_format,
            vad_threshold=vad_threshold,
            silence_duration_ms=silence_duration_ms,
        )
        self.buffer = AudioBuffer(sample_rate=sample_rate, max_duration_sec=max_buffer_sec)

        self._inference_lock = asyncio.Lock()
        self._closed = False
        self._is_speaking = False
        self._speech_start_sample = 0
        self._silence_ms = 0
        self._current_item_id: str | None = None
        self._transcript_tail = ""
        self._last_activity_at = time.monotonic()

    @property
    def closed(self) -> bool:
        return self._closed

    def touch(self) -> None:
        self._last_activity_at = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity_at

    def session_snapshot(self) -> dict[str, Any]:
        return dict(self.session_config)

    async def close(self) -> None:
        self._closed = True

    async def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.touch()

        if event_type == "session.update":
            await self._handle_session_update(payload)
            return
        if event_type == "input_audio_buffer.append":
            await self._handle_append(payload)
            return
        if event_type == "input_audio_buffer.commit":
            await self._handle_commit(force=True)
            return
        if event_type == "input_audio_buffer.clear":
            await self._handle_clear()
            return

        await self.send_event(
            error_event(
                f"Unsupported client event type: {event_type}",
                code="unsupported_event",
            )
        )

    async def _handle_session_update(self, payload: dict[str, Any]) -> None:
        session = payload.get("session")
        if not isinstance(session, dict):
            await self.send_event(
                error_event("session.update requires a session object", param="session")
            )
            return

        if "model" in session and session["model"]:
            self.model_id = str(session["model"])
            self.session_config["model"] = self.model_id

        transcription = session.get("input_audio_transcription")
        if isinstance(transcription, dict):
            if transcription.get("model"):
                self.model_id = str(transcription["model"])
                self.session_config["model"] = self.model_id
            if transcription.get("language"):
                self.language = str(transcription["language"])
                self.session_config.setdefault("input_audio_transcription", {})["language"] = self.language

        if session.get("input_audio_format"):
            audio_format = str(session["input_audio_format"])
            if audio_format != "pcm16":
                await self.send_event(
                    error_event(
                        (
                            f"Unsupported input_audio_format '{audio_format}'. "
                            "This server expects pcm16 mono at 16 kHz."
                        ),
                        param="input_audio_format",
                        code="unsupported_audio_format",
                    )
                )
                return
            self.input_audio_format = audio_format
            self.session_config["input_audio_format"] = audio_format

        turn_detection = session.get("turn_detection")
        if isinstance(turn_detection, dict):
            if turn_detection.get("threshold") is not None:
                self.vad_threshold = float(turn_detection["threshold"])
            if turn_detection.get("silence_duration_ms") is not None:
                self.silence_duration_ms = int(turn_detection["silence_duration_ms"])
            self.session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.silence_duration_ms,
                "sample_rate": self.sample_rate,
            }

        self.session_config["input_audio_transcription"] = {
            "model": self.model_id,
            "language": self.language,
        }
        await self.send_event(session_updated_event(self.session_snapshot()))

    async def _handle_append(self, payload: dict[str, Any]) -> None:
        audio_b64 = payload.get("audio")
        if not isinstance(audio_b64, str) or not audio_b64:
            await self.send_event(
                error_event("input_audio_buffer.append requires base64 audio", param="audio")
            )
            return

        try:
            chunk = self.buffer.append_pcm16_base64(audio_b64)
        except ValueError as exc:
            await self.send_event(error_event(str(exc), param="audio", code="invalid_audio"))
            return

        if chunk.size == 0:
            return

        chunk_ms = self.buffer.samples_to_ms(chunk.size)
        rms = chunk_rms(chunk)

        if rms >= self.vad_threshold:
            self._silence_ms = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_sample = len(self.buffer) - chunk.size
                self._current_item_id = f"item_{uuid.uuid4().hex[:16]}"
                await self.send_event(
                    speech_started_event(
                        self.buffer.samples_to_ms(self._speech_start_sample),
                        self._current_item_id,
                    )
                )
        elif self._is_speaking:
            self._silence_ms += chunk_ms
            if self._silence_ms >= self.silence_duration_ms:
                await self._finalize_speech(auto_commit=True)

    async def _handle_clear(self) -> None:
        self.buffer.clear()
        self._reset_speech_state()
        record_tool_call("realtime.buffer.clear", model=self.model_id)

    async def _handle_commit(self, *, force: bool) -> None:
        if self._is_speaking:
            await self._finalize_speech(auto_commit=True, force=force)
            return

        if len(self.buffer) == 0:
            await self.send_event(error_event("Audio buffer is empty", code="empty_audio"))
            return

        item_id = self._current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        self._current_item_id = item_id
        start = 0
        end = len(self.buffer)
        await self._transcribe_segment(start, end, item_id, force=force)

    async def _finalize_speech(self, *, auto_commit: bool, force: bool = False) -> None:
        if not self._is_speaking:
            return

        item_id = self._current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        end = len(self.buffer)
        start = self._speech_start_sample
        audio_end_ms = self.buffer.samples_to_ms(end)

        await self.send_event(speech_stopped_event(audio_end_ms, item_id))
        self._reset_speech_state()

        if auto_commit:
            await self._transcribe_segment(start, end, item_id, force=force)

    def _reset_speech_state(self) -> None:
        self._is_speaking = False
        self._silence_ms = 0
        self._speech_start_sample = 0

    async def _transcribe_segment(
        self,
        start: int,
        end: int,
        item_id: str,
        *,
        force: bool,
    ) -> None:
        segment = self.buffer.extract_range(start, end)
        duration_ms = int(segment.size * 1000 / self.sample_rate)

        if segment.size == 0:
            if force:
                await self.send_event(error_event("Audio buffer is empty", code="empty_audio"))
            return

        if duration_ms < self.min_segment_ms:
            logger.debug(
                "Skipping short realtime segment model=%s duration_ms=%d",
                self.model_id,
                duration_ms,
            )
            self.buffer.trim_prefix(end)
            return

        await self.send_event(buffer_committed_event(item_id))

        prompt = self._transcript_tail[-REALTIME_INITIAL_PROMPT_CHARS :] if self._transcript_tail else None

        async with self._inference_lock:
            try:
                record_tool_call(
                    "realtime.transcribe",
                    model=self.model_id,
                    duration_ms=duration_ms,
                    language=self.language,
                )
                result = await asyncio.to_thread(
                    self.asr_service.transcribe_array,
                    segment,
                    sample_rate=self.sample_rate,
                    model_id=self.model_id,
                    language=self.language,
                    prompt=prompt,
                    temperature=0.0,
                    device=self.device,
                    compute_type=self.compute_type,
                    beam_size=self.beam_size,
                    vad_filter=False,
                )
            except OpenAIAPIError as exc:
                await self.send_event(
                    error_event(
                        exc.message,
                        error_type=exc.error_type,
                        code=exc.code,
                        param=exc.param,
                    )
                )
                return
            except Exception as exc:
                logger.exception("Realtime transcription failed model=%s", self.model_id)
                await self.send_event(
                    error_event(
                        f"Transcription failed: {exc}",
                        error_type="server_error",
                        code="transcription_failed",
                    )
                )
                return

        text = str(result.get("text", "")).strip()
        self.buffer.trim_prefix(end)

        if not text:
            return

        await self.send_event(transcription_delta_event(item_id, text))
        await self.send_event(transcription_completed_event(item_id, text))

        if self._transcript_tail:
            self._transcript_tail = f"{self._transcript_tail} {text}".strip()
        else:
            self._transcript_tail = text

        if len(self._transcript_tail) > REALTIME_INITIAL_PROMPT_CHARS * 4:
            self._transcript_tail = self._transcript_tail[-REALTIME_INITIAL_PROMPT_CHARS * 4 :]
