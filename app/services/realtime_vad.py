"""Server-side VAD helpers for realtime audio append events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from app.services.audio_buffer import AudioBuffer, chunk_rms


@dataclass
class SpeechStart:
    """Result of detecting the beginning of speech in the working buffer."""

    item_id: str
    audio_start_ms: int


@dataclass
class SpeechStop:
    """Result of detecting end-of-speech silence."""

    item_id: str
    start_sample: int
    end_sample: int
    audio_end_ms: int


class VadController:
    """Tracks speaking state from RMS energy against a silence threshold."""

    def __init__(
        self,
        *,
        buffer: AudioBuffer,
        vad_threshold: float,
        silence_duration_ms: int,
    ) -> None:
        self._buffer = buffer
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.is_speaking = False
        self.speech_start_sample = 0
        self.silence_ms = 0
        self.current_item_id: str | None = None

    def reset(self) -> None:
        """Clear speaking state without touching the audio buffer."""
        self.is_speaking = False
        self.silence_ms = 0
        self.speech_start_sample = 0

    def note_truncation(self, dropped: int) -> None:
        """Adjust speech indices after the working buffer truncates from the front."""
        if not dropped:
            return
        if self.is_speaking:
            self.speech_start_sample = max(0, self.speech_start_sample - dropped)

    def on_chunk(self, chunk: np.ndarray) -> tuple[SpeechStart | None, SpeechStop | None]:
        """Update VAD state after a newly appended chunk.

        Args:
            chunk: Latest float32 mono audio chunk already appended to the buffer.

        Returns:
            Optional speech-start and speech-stop events for this chunk.
        """
        if chunk.size == 0:
            return None, None

        chunk_ms = self._buffer.samples_to_ms(chunk.size)
        rms = chunk_rms(chunk)
        started: SpeechStart | None = None
        stopped: SpeechStop | None = None

        if rms >= self.vad_threshold:
            self.silence_ms = 0
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_sample = len(self._buffer) - chunk.size
                self.current_item_id = f"item_{uuid.uuid4().hex[:16]}"
                started = SpeechStart(
                    item_id=self.current_item_id,
                    audio_start_ms=self._buffer.samples_to_ms(self.speech_start_sample),
                )
        elif self.is_speaking:
            self.silence_ms += chunk_ms
            if self.silence_ms >= self.silence_duration_ms:
                stopped = self.begin_finalize()

        return started, stopped

    def begin_finalize(self) -> SpeechStop | None:
        """Close the current speaking window and return its sample range."""
        if not self.is_speaking:
            return None

        item_id = self.current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        end = len(self._buffer)
        start = self.speech_start_sample
        audio_end_ms = self._buffer.samples_to_ms(end)
        self.reset()
        return SpeechStop(
            item_id=item_id,
            start_sample=start,
            end_sample=end,
            audio_end_ms=audio_end_ms,
        )
