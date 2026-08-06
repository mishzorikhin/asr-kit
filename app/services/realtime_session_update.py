"""Apply OpenAI-style session.update payloads to a realtime session."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.errors import OpenAIAPIError
from app.model_registry import backend_supports_realtime
from app.openai_realtime_events import error_event, session_updated_event
from app.services.realtime_types import SendEvent, SpeakerDiarizationConfig

logger = logging.getLogger(__name__)


class SessionUpdateTarget(Protocol):
    """Mutable session fields touched by session.update."""

    model_id: str
    language: str
    input_audio_format: str
    timestamp_granularities: list[str]
    vad_threshold: float
    silence_duration_ms: int
    sample_rate: int
    diarization_capable: bool
    speaker_config: SpeakerDiarizationConfig
    session_config: dict[str, Any]
    asr_service: Any
    send_event: SendEvent

    def session_snapshot(self) -> dict[str, Any]:
        """Return the public session config snapshot."""


async def ensure_realtime_backend(session: SessionUpdateTarget, model_id: str) -> bool:
    """Validate that model_id is configured and supports realtime ASR.

    Args:
        session: Active realtime session.
        model_id: Requested model id.

    Returns:
        True when the model can be used for realtime transcription.
    """
    try:
        configured = session.asr_service.registry.get(model_id)
    except OpenAIAPIError as exc:
        await session.send_event(
            error_event(
                exc.message,
                error_type=exc.error_type,
                code=exc.code,
                param=exc.param,
            )
        )
        return False

    backend = configured.get("backend", "faster-whisper")
    if not backend_supports_realtime(backend):
        await session.send_event(
            error_event(
                (
                    f"Model '{model_id}' uses backend '{backend}', which is not "
                    "supported for WebSocket realtime yet. "
                    "Use POST /v1/audio/transcriptions instead."
                ),
                param="model",
                code="unsupported_backend",
            )
        )
        return False
    return True


async def apply_session_update(
    session: SessionUpdateTarget,
    payload: dict[str, Any],
) -> None:
    """Apply a session.update client event to session state.

    Args:
        session: Active realtime session.
        payload: Raw client event payload containing optional ``session`` object.
    """
    session_payload = payload.get("session")
    if not isinstance(session_payload, dict):
        await session.send_event(
            error_event("session.update requires a session object", param="session")
        )
        return

    if not await _apply_model_fields(session, session_payload):
        return
    await _apply_transcription_fields(session, session_payload)
    await _apply_speaker_diarization_fields(session, session_payload)
    if not await _apply_audio_format(session, session_payload):
        return
    _apply_turn_detection(session, session_payload)
    _refresh_session_config(session)
    await session.send_event(session_updated_event(session.session_snapshot()))


async def _apply_model_fields(
    session: SessionUpdateTarget,
    session_payload: dict[str, Any],
) -> bool:
    if "model" not in session_payload or not session_payload["model"]:
        return True
    next_model_id = str(session_payload["model"])
    if not await ensure_realtime_backend(session, next_model_id):
        return False
    session.model_id = next_model_id
    session.session_config["model"] = session.model_id
    return True


async def _apply_transcription_fields(
    session: SessionUpdateTarget,
    session_payload: dict[str, Any],
) -> None:
    transcription = session_payload.get("input_audio_transcription")
    if not isinstance(transcription, dict):
        return

    if transcription.get("model"):
        next_model_id = str(transcription["model"])
        if not await ensure_realtime_backend(session, next_model_id):
            return
        session.model_id = next_model_id
        session.session_config["model"] = session.model_id
    if transcription.get("language"):
        session.language = str(transcription["language"])
    granularities = transcription.get("timestamp_granularities")
    if isinstance(granularities, list):
        session.timestamp_granularities = [
            str(value) for value in granularities if isinstance(value, str)
        ]


async def _apply_speaker_diarization_fields(
    session: SessionUpdateTarget,
    session_payload: dict[str, Any],
) -> None:
    speaker_diarization = session_payload.get("speaker_diarization")
    if not isinstance(speaker_diarization, dict):
        return

    config = session.speaker_config
    if speaker_diarization.get("enabled") is not None:
        config.enabled = bool(speaker_diarization["enabled"])
    if speaker_diarization.get("finalize") is not None:
        config.finalize = bool(speaker_diarization["finalize"])
    if speaker_diarization.get("mode"):
        config.mode = str(speaker_diarization["mode"])
    if speaker_diarization.get("diarization_model"):
        config.diarization_model = str(speaker_diarization["diarization_model"])
    if speaker_diarization.get("max_speakers") is not None:
        config.max_speakers = int(speaker_diarization["max_speakers"])
    if speaker_diarization.get("num_speakers") is not None:
        config.num_speakers = int(speaker_diarization["num_speakers"])
    if speaker_diarization.get("min_speakers") is not None:
        config.min_speakers = int(speaker_diarization["min_speakers"])
    if speaker_diarization.get("use_exclusive") is not None:
        config.use_exclusive = bool(speaker_diarization["use_exclusive"])
    known_names = speaker_diarization.get("known_speaker_names")
    if isinstance(known_names, list):
        config.known_speaker_names = [
            str(name) for name in known_names if isinstance(name, str)
        ]

    if config.enabled and not session.diarization_capable:
        has_explicit_model = bool(speaker_diarization.get("diarization_model"))
        if not has_explicit_model:
            await session.send_event(
                error_event(
                    (
                        "speaker_diarization requires a model with diarization "
                        "capability or an explicit diarization_model path."
                    ),
                    param="speaker_diarization",
                    code="unsupported_capability",
                )
            )
            config.enabled = False
            config.finalize = False


async def _apply_audio_format(
    session: SessionUpdateTarget,
    session_payload: dict[str, Any],
) -> bool:
    if not session_payload.get("input_audio_format"):
        return True
    audio_format = str(session_payload["input_audio_format"])
    if audio_format != "pcm16":
        await session.send_event(
            error_event(
                (
                    f"Unsupported input_audio_format '{audio_format}'. "
                    "This server expects pcm16 mono at 16 kHz."
                ),
                param="input_audio_format",
                code="unsupported_audio_format",
            )
        )
        return False
    session.input_audio_format = audio_format
    session.session_config["input_audio_format"] = audio_format
    return True


def _apply_turn_detection(
    session: SessionUpdateTarget,
    session_payload: dict[str, Any],
) -> None:
    turn_detection = session_payload.get("turn_detection")
    if not isinstance(turn_detection, dict):
        return
    if turn_detection.get("threshold") is not None:
        session.vad_threshold = float(turn_detection["threshold"])
    if turn_detection.get("silence_duration_ms") is not None:
        session.silence_duration_ms = int(turn_detection["silence_duration_ms"])
    session.session_config["turn_detection"] = {
        "type": "server_vad",
        "threshold": session.vad_threshold,
        "silence_duration_ms": session.silence_duration_ms,
        "sample_rate": session.sample_rate,
    }


def _refresh_session_config(session: SessionUpdateTarget) -> None:
    transcription_config: dict[str, Any] = {
        "model": session.model_id,
        "language": session.language,
    }
    if session.timestamp_granularities:
        transcription_config["timestamp_granularities"] = (
            session.timestamp_granularities
        )
    session.session_config["input_audio_transcription"] = transcription_config
    session.session_config["speaker_diarization"] = session.speaker_config.snapshot()
