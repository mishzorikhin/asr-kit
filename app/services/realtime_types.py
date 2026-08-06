"""Shared types for realtime WebSocket transcription sessions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import (
    DEFAULT_DIARIZATION_MODEL,
    REALTIME_SPEAKER_MAX_SPEAKERS,
)

SendEvent = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class TranscriptItem:
    """One committed transcript segment within a realtime session."""

    item_id: str
    start_sec: float
    end_sec: float
    text: str
    speaker: str | None = None
    speaker_provisional: bool = False


@dataclass
class SpeakerDiarizationConfig:
    """Speaker diarization options controlled via session.update."""

    enabled: bool = False
    finalize: bool = False
    mode: str = "provisional"
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    max_speakers: int = REALTIME_SPEAKER_MAX_SPEAKERS
    num_speakers: int | None = None
    min_speakers: int | None = None
    known_speaker_names: list[str] = field(default_factory=list)
    use_exclusive: bool = False

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the current config."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "finalize": self.finalize,
            "diarization_model": self.diarization_model,
            "max_speakers": self.max_speakers,
            "num_speakers": self.num_speakers,
            "min_speakers": self.min_speakers,
            "known_speaker_names": list(self.known_speaker_names),
            "use_exclusive": self.use_exclusive,
        }
