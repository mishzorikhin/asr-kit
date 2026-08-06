"""Segment transcription and provisional speaker assignment for realtime."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from app.config import (
    REALTIME_INITIAL_PROMPT_CHARS,
    REALTIME_SPEAKER_MIN_SEGMENT_SEC,
    REALTIME_SPEAKER_SIMILARITY_THRESHOLD,
)
from app.errors import OpenAIAPIError
from app.openai_realtime_events import (
    buffer_committed_event,
    error_event,
    speaker_assigned_event,
    transcription_completed_event,
    transcription_delta_event,
)
from app.services.asr import ASRService
from app.services.audio_buffer import AudioBuffer
from app.services.realtime_speaker_config import resolve_speaker_embedding_model
from app.services.realtime_speaker_tracker import RealtimeSpeakerTracker
from app.services.realtime_types import SendEvent, SpeakerDiarizationConfig, TranscriptItem
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)


def offset_words(
    words: list[dict[str, Any]],
    offset_sec: float,
) -> list[dict[str, Any]]:
    """Shift word timestamps by an absolute session offset.

    Args:
        words: Word dicts with ``start`` / ``end`` relative to a segment.
        offset_sec: Absolute start time of the segment in the session.

    Returns:
        New list of word dicts with absolute timestamps.
    """
    return [
        {
            "word": word["word"],
            "start": round(word["start"] + offset_sec, 3),
            "end": round(word["end"] + offset_sec, 3),
        }
        for word in words
    ]


class SegmentTranscriber:
    """Transcribes VAD segments and optionally assigns provisional speakers."""

    def __init__(
        self,
        *,
        asr_service: ASRService,
        send_event: SendEvent,
        speaker_config: SpeakerDiarizationConfig,
        sample_rate: int,
        device: str,
        compute_type: str,
        beam_size: int,
        min_segment_ms: int,
    ) -> None:
        self._asr_service = asr_service
        self._send_event = send_event
        self._speaker_config = speaker_config
        self._sample_rate = sample_rate
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._min_segment_ms = min_segment_ms
        self._inference_lock = asyncio.Lock()
        self._speaker_lock = asyncio.Lock()
        self._speaker_tracker: RealtimeSpeakerTracker | None = None
        self._transcript_tail = ""
        self.transcript_items: list[TranscriptItem] = []

    def clear(self) -> None:
        """Reset transcript state after buffer clear."""
        self.transcript_items.clear()
        self._transcript_tail = ""
        self._speaker_tracker = None

    def _ensure_speaker_tracker(self) -> RealtimeSpeakerTracker | None:
        if not self._speaker_config.enabled or self._speaker_config.mode != "provisional":
            return None
        if self._speaker_tracker is not None:
            return self._speaker_tracker

        embedding_model = resolve_speaker_embedding_model(
            self._speaker_config.diarization_model
        )
        if not embedding_model:
            logger.warning(
                "Realtime speaker diarization enabled but no embedding model "
                "resolved for %s",
                self._speaker_config.diarization_model,
            )
            return None

        self._speaker_tracker = RealtimeSpeakerTracker(
            embedding_model_path=embedding_model,
            similarity_threshold=REALTIME_SPEAKER_SIMILARITY_THRESHOLD,
            max_speakers=self._speaker_config.max_speakers,
            known_speaker_names=self._speaker_config.known_speaker_names,
            min_segment_sec=REALTIME_SPEAKER_MIN_SEGMENT_SEC,
        )
        return self._speaker_tracker

    async def transcribe_segment(
        self,
        *,
        buffer: AudioBuffer,
        start: int,
        end: int,
        item_id: str,
        force: bool,
        model_id: str,
        language: str,
        timestamp_granularities: list[str],
        absolute_sample_offset: int,
    ) -> int:
        """Transcribe ``buffer[start:end]`` and trim the committed prefix.

        Args:
            buffer: Working audio buffer for the current utterance window.
            start: Inclusive start sample index in ``buffer``.
            end: Exclusive end sample index in ``buffer``.
            item_id: OpenAI-style item id for emitted events.
            force: When True, emit empty-buffer errors.
            model_id: ASR model id.
            language: Language hint.
            timestamp_granularities: Optional word/segment granularities.
            absolute_sample_offset: Session-absolute sample offset of buffer[0].

        Returns:
            Number of samples trimmed from the buffer prefix (to advance
            the absolute session offset).
        """
        segment = buffer.extract_range(start, end)
        duration_ms = int(segment.size * 1000 / self._sample_rate)
        absolute_start = (absolute_sample_offset + start) / self._sample_rate
        absolute_end = (absolute_sample_offset + end) / self._sample_rate

        if segment.size == 0:
            if force:
                await self._send_event(
                    error_event("Audio buffer is empty", code="empty_audio")
                )
            return 0

        if duration_ms < self._min_segment_ms:
            logger.debug(
                "Skipping short realtime segment model=%s duration_ms=%d",
                model_id,
                duration_ms,
            )
            buffer.trim_prefix(end)
            return end

        await self._send_event(buffer_committed_event(item_id))

        prompt = (
            self._transcript_tail[-REALTIME_INITIAL_PROMPT_CHARS:]
            if self._transcript_tail
            else None
        )
        granularities = timestamp_granularities or None

        async with self._inference_lock:
            try:
                record_tool_call(
                    "realtime.transcribe",
                    model=model_id,
                    duration_ms=duration_ms,
                    language=language,
                )
                result = await asyncio.to_thread(
                    self._asr_service.transcribe_array,
                    segment,
                    sample_rate=self._sample_rate,
                    model_id=model_id,
                    language=language,
                    prompt=prompt,
                    temperature=0.0,
                    device=self._device,
                    compute_type=self._compute_type,
                    beam_size=self._beam_size,
                    vad_filter=False,
                    timestamp_granularities=granularities,
                )
            except OpenAIAPIError as exc:
                await self._send_event(
                    error_event(
                        exc.message,
                        error_type=exc.error_type,
                        code=exc.code,
                        param=exc.param,
                    )
                )
                return 0
            except Exception as exc:
                logger.exception("Realtime transcription failed model=%s", model_id)
                await self._send_event(
                    error_event(
                        f"Transcription failed: {exc}",
                        error_type="server_error",
                        code="transcription_failed",
                    )
                )
                return 0

        text = str(result.get("text", "")).strip()
        buffer.trim_prefix(end)

        if not text:
            return end

        words = offset_words(result.get("words", []), absolute_start)
        transcript_item = TranscriptItem(
            item_id=item_id,
            start_sec=absolute_start,
            end_sec=absolute_end,
            text=text,
        )
        self.transcript_items.append(transcript_item)

        await self._send_event(transcription_delta_event(item_id, text))
        await self._send_event(
            transcription_completed_event(
                item_id,
                text,
                words=words or None,
            )
        )
        self._append_prompt_tail(text)
        await self._assign_provisional_speaker(item_id, segment, transcript_item)
        return end

    def _append_prompt_tail(self, text: str) -> None:
        if self._transcript_tail:
            self._transcript_tail = f"{self._transcript_tail} {text}".strip()
        else:
            self._transcript_tail = text
        max_chars = REALTIME_INITIAL_PROMPT_CHARS * 4
        if len(self._transcript_tail) > max_chars:
            self._transcript_tail = self._transcript_tail[-max_chars:]

    async def _assign_provisional_speaker(
        self,
        item_id: str,
        segment: np.ndarray,
        transcript_item: TranscriptItem,
    ) -> None:
        tracker = self._ensure_speaker_tracker()
        if tracker is None:
            return

        async with self._speaker_lock:
            try:
                assignment = await asyncio.to_thread(
                    tracker.assign_speaker,
                    segment,
                    sample_rate=self._sample_rate,
                )
            except OpenAIAPIError as exc:
                await self._send_event(
                    error_event(
                        exc.message,
                        error_type=exc.error_type,
                        code=exc.code,
                        param=exc.param,
                    )
                )
                return

        if assignment is None:
            return

        speaker, confidence = assignment
        transcript_item.speaker = speaker
        transcript_item.speaker_provisional = True
        await self._send_event(
            speaker_assigned_event(
                item_id,
                speaker,
                confidence=confidence,
                provisional=True,
            )
        )
