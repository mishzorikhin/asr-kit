"""Realtime WebSocket transcription session orchestrator.

Coordinates server-side VAD, segment transcription, provisional speakers, and
optional session-final pyannote diarization. Heavy logic lives in sibling
modules under ``app.services.realtime_*``.
"""

from __future__ import annotations

import logging
import time
import uuid

from app.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_LANGUAGE,
    REALTIME_BEAM_SIZE,
    REALTIME_MAX_BUFFER_SEC,
    REALTIME_MIN_SEGMENT_MS,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION_MS,
    REALTIME_VAD_THRESHOLD,
)
from app.openai_realtime_events import (
    default_session_config,
    error_event,
    session_ended_event,
    speech_started_event,
    speech_stopped_event,
)
from app.services.asr import ASRService
from app.services.audio_buffer import AudioBuffer
from app.services.diarization import DiarizationService
from app.services.realtime_finalize import SessionDiarizationFinalizer
from app.services.realtime_session_update import (
    apply_session_update,
    ensure_realtime_backend,
)
from app.services.realtime_transcriber import SegmentTranscriber
from app.services.realtime_types import SendEvent, SpeakerDiarizationConfig
from app.services.realtime_vad import VadController
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)

# Keep full-session audio for finalize; VAD working buffer stays bounded.
_SESSION_AUDIO_MAX_SEC = 60 * 60


class RealtimeSession:
    """OpenAI-compatible pseudo-realtime transcription session."""

    def __init__(
        self,
        *,
        asr_service: ASRService,
        diarization_service: DiarizationService | None,
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
        diarization_capable: bool = False,
        default_diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    ) -> None:
        self.asr_service = asr_service
        self.diarization_service = diarization_service
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
        self.diarization_capable = diarization_capable
        self.default_diarization_model = default_diarization_model
        self.input_audio_format = "pcm16"
        self.timestamp_granularities: list[str] = []
        self.speaker_config = SpeakerDiarizationConfig(
            diarization_model=default_diarization_model,
        )

        self.session_config = default_session_config(
            model_id=model_id,
            language=language,
            sample_rate=sample_rate,
            input_audio_format=self.input_audio_format,
            vad_threshold=vad_threshold,
            silence_duration_ms=silence_duration_ms,
            timestamp_granularities=self.timestamp_granularities,
            speaker_diarization=self.speaker_config.snapshot(),
        )
        self.buffer = AudioBuffer(
            sample_rate=sample_rate,
            max_duration_sec=max_buffer_sec,
        )
        self._session_audio = AudioBuffer(
            sample_rate=sample_rate,
            max_duration_sec=_SESSION_AUDIO_MAX_SEC,
        )
        self._vad = VadController(
            buffer=self.buffer,
            vad_threshold=vad_threshold,
            silence_duration_ms=silence_duration_ms,
        )
        self._transcriber = SegmentTranscriber(
            asr_service=asr_service,
            send_event=send_event,
            speaker_config=self.speaker_config,
            sample_rate=sample_rate,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            min_segment_ms=min_segment_ms,
        )
        self._finalizer = SessionDiarizationFinalizer(
            diarization_service=diarization_service,
            speaker_config=self.speaker_config,
            sample_rate=sample_rate,
            model_id=model_id,
            send_event=send_event,
        )

        self._closed = False
        self._last_activity_at = time.monotonic()
        self._absolute_sample_offset = 0
        self._ended = False

    @property
    def closed(self) -> bool:
        """Whether the WebSocket session has been closed."""
        return self._closed

    @property
    def ended_via_protocol(self) -> bool:
        """Whether the client sent ``session.end``."""
        return self._ended

    @property
    def _transcript_items(self):
        """Compatibility alias used by tests and finalize."""
        return self._transcriber.transcript_items

    @property
    def _is_speaking(self) -> bool:
        return self._vad.is_speaking

    def touch(self) -> None:
        """Mark the session as recently active."""
        self._last_activity_at = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last client activity."""
        return time.monotonic() - self._last_activity_at

    def session_snapshot(self) -> dict:
        """Return a copy of the public session config."""
        return dict(self.session_config)

    async def close(self) -> None:
        """Mark the session closed without emitting protocol events."""
        if self._closed:
            return
        self._closed = True

    async def end_session(self) -> None:
        """Handle ``session.end``: flush audio, finalize diarization, close."""
        if self._ended:
            await self.send_event(
                error_event("Session already ended", code="session_already_ended")
            )
            return

        self._ended = True
        record_tool_call("realtime.session.end", model=self.model_id)

        if self._vad.is_speaking:
            await self._finalize_speech(auto_commit=True, force=True)
        elif len(self.buffer) > 0:
            await self._handle_commit(force=True)

        diarization_finalized = await self._run_finalize()
        await self.send_event(
            session_ended_event(
                item_count=len(self._transcript_items),
                diarization_finalized=diarization_finalized,
            )
        )
        self._closed = True

    async def finalize_session(self) -> None:
        """Best-effort finalize when the client disconnects without session.end."""
        if self._ended:
            return
        await self._run_finalize()

    async def _run_finalize(self) -> bool:
        return await self._finalizer.run(
            self._session_audio.extract_all(),
            self._transcript_items,
        )

    async def handle_event(self, event_type: str, payload: dict) -> None:
        """Dispatch a validated client event to the appropriate handler."""
        self.touch()

        if event_type == "session.end":
            await self.end_session()
            return

        if self._ended:
            await self.send_event(
                error_event("Session already ended", code="session_already_ended")
            )
            return

        if event_type == "session.update":
            await apply_session_update(self, payload)
            self._sync_vad_settings()
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

    def _sync_vad_settings(self) -> None:
        self._vad.vad_threshold = self.vad_threshold
        self._vad.silence_duration_ms = self.silence_duration_ms

    async def _ensure_realtime_backend(self, model_id: str) -> bool:
        """Validate realtime support for a model id (used by unit tests)."""
        return await ensure_realtime_backend(self, model_id)

    async def _handle_session_update(self, payload: dict) -> None:
        """Apply session.update (used by unit tests)."""
        await apply_session_update(self, payload)
        self._sync_vad_settings()

    async def _handle_append(self, payload: dict) -> None:
        audio_b64 = payload.get("audio")
        if not isinstance(audio_b64, str) or not audio_b64:
            await self.send_event(
                error_event(
                    "input_audio_buffer.append requires base64 audio",
                    param="audio",
                )
            )
            return

        try:
            chunk, dropped = self.buffer.append_pcm16_base64(audio_b64)
            self._session_audio.append_float32(chunk)
        except ValueError as exc:
            await self.send_event(
                error_event(str(exc), param="audio", code="invalid_audio")
            )
            return

        if dropped:
            self._absolute_sample_offset += dropped
            self._vad.note_truncation(dropped)

        started, stopped = self._vad.on_chunk(chunk)
        if started is not None:
            await self.send_event(
                speech_started_event(started.audio_start_ms, started.item_id)
            )
        if stopped is not None:
            await self.send_event(
                speech_stopped_event(stopped.audio_end_ms, stopped.item_id)
            )
            await self._transcribe_segment(
                stopped.start_sample,
                stopped.end_sample,
                stopped.item_id,
                force=False,
            )

    async def _handle_clear(self) -> None:
        self.buffer.clear()
        self._session_audio.clear()
        self._absolute_sample_offset = 0
        self._transcriber.clear()
        self._vad.reset()
        record_tool_call("realtime.buffer.clear", model=self.model_id)

    async def _handle_commit(self, *, force: bool) -> None:
        if self._vad.is_speaking:
            await self._finalize_speech(auto_commit=True, force=force)
            return

        if len(self.buffer) == 0:
            await self.send_event(
                error_event("Audio buffer is empty", code="empty_audio")
            )
            return

        item_id = self._vad.current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        self._vad.current_item_id = item_id
        await self._transcribe_segment(0, len(self.buffer), item_id, force=force)

    async def _finalize_speech(self, *, auto_commit: bool, force: bool = False) -> None:
        stopped = self._vad.begin_finalize()
        if stopped is None:
            return

        await self.send_event(
            speech_stopped_event(stopped.audio_end_ms, stopped.item_id)
        )
        if auto_commit:
            await self._transcribe_segment(
                stopped.start_sample,
                stopped.end_sample,
                stopped.item_id,
                force=force,
            )

    async def _transcribe_segment(
        self,
        start: int,
        end: int,
        item_id: str,
        *,
        force: bool,
    ) -> None:
        advanced = await self._transcriber.transcribe_segment(
            buffer=self.buffer,
            start=start,
            end=end,
            item_id=item_id,
            force=force,
            model_id=self.model_id,
            language=self.language,
            timestamp_granularities=self.timestamp_granularities,
            absolute_sample_offset=self._absolute_sample_offset,
        )
        self._absolute_sample_offset += advanced
