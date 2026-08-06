"""Final pyannote diarization for a completed realtime session."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.errors import OpenAIAPIError
from app.openai_format import speaker_labels
from app.openai_realtime_events import (
    error_event,
    session_diarization_completed_event,
    speaker_updated_event,
)
from app.services.diarization import DiarizationService, speaker_for_segment
from app.services.realtime_types import SendEvent, SpeakerDiarizationConfig, TranscriptItem
from app.services.wav_io import write_temp_wav

logger = logging.getLogger(__name__)


class SessionDiarizationFinalizer:
    """Runs optional session-end pyannote diarization and emits update events."""

    def __init__(
        self,
        *,
        diarization_service: DiarizationService | None,
        speaker_config: SpeakerDiarizationConfig,
        sample_rate: int,
        model_id: str,
        send_event: SendEvent,
    ) -> None:
        self._diarization_service = diarization_service
        self._speaker_config = speaker_config
        self._sample_rate = sample_rate
        self._model_id = model_id
        self._send_event = send_event
        self._finalized = False
        self._diarization_completed = False

    @property
    def diarization_completed(self) -> bool:
        """Whether finalize has already produced speaker labels."""
        return self._diarization_completed

    async def run(
        self,
        audio: np.ndarray,
        transcript_items: list[TranscriptItem],
    ) -> bool:
        """Finalize speakers for the session when configured.

        Args:
            audio: Full session audio as float32 mono PCM.
            transcript_items: Committed transcript items to label.

        Returns:
            True when diarization completed successfully.
        """
        if self._finalized:
            return self._diarization_completed
        self._finalized = True

        if not self._speaker_config.enabled or not self._speaker_config.finalize:
            return False
        if self._diarization_service is None:
            return False
        if audio.size == 0 or not transcript_items:
            return False

        items_snapshot = list(transcript_items)
        try:
            final_items = await asyncio.to_thread(
                self._finalize_diarization,
                audio,
                items_snapshot,
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
            return False
        except Exception as exc:
            logger.exception(
                "Realtime session diarization finalize failed model=%s",
                self._model_id,
            )
            await self._send_event(
                error_event(
                    f"Session diarization failed: {exc}",
                    error_type="server_error",
                    code="diarization_failed",
                )
            )
            return False

        for item, final_speaker in final_items:
            if item.speaker and item.speaker != final_speaker:
                await self._send_event(
                    speaker_updated_event(
                        item.item_id,
                        final_speaker,
                        previous_speaker=item.speaker,
                        final=True,
                    )
                )
            elif not item.speaker:
                await self._send_event(
                    speaker_updated_event(
                        item.item_id,
                        final_speaker,
                        previous_speaker="UNKNOWN",
                        final=True,
                    )
                )
            item.speaker = final_speaker
            item.speaker_provisional = False

        await self._send_event(
            session_diarization_completed_event(
                items=[
                    {
                        "item_id": item.item_id,
                        "start": item.start_sec,
                        "end": item.end_sec,
                        "text": item.text,
                        "speaker": item.speaker,
                    }
                    for item in items_snapshot
                ],
                duration_sec=len(audio) / self._sample_rate,
            )
        )
        self._diarization_completed = True
        return True

    def _finalize_diarization(
        self,
        audio: np.ndarray,
        items: list[TranscriptItem],
    ) -> list[tuple[TranscriptItem, str]]:
        if self._diarization_service is None:
            raise OpenAIAPIError(
                "Diarization service is not available",
                status_code=500,
                error_type="server_error",
                code="diarization_unavailable",
            )

        temp_path = write_temp_wav(audio, self._sample_rate)
        try:
            diarization_turns = self._diarization_service.diarize(
                temp_path,
                diarization_model=self._speaker_config.diarization_model,
                min_speakers=self._speaker_config.min_speakers,
                max_speakers=self._speaker_config.max_speakers,
                num_speakers=self._speaker_config.num_speakers,
                use_exclusive=self._speaker_config.use_exclusive,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        internal_labels = speaker_labels(
            [{"speaker": turn["speaker"]} for turn in diarization_turns],
            self._speaker_config.known_speaker_names,
        )
        results: list[tuple[TranscriptItem, str]] = []
        for item in items:
            internal_speaker = speaker_for_segment(
                segment_start=item.start_sec,
                segment_end=item.end_sec,
                diarization_turns=diarization_turns,
            )
            final_speaker = internal_labels.get(internal_speaker, internal_speaker)
            results.append((item, final_speaker))
        return results
