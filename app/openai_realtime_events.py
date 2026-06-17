from __future__ import annotations

import time
import uuid
from typing import Any


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def server_event(event_type: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "event_id": new_event_id(),
    }
    payload.update(fields)
    return payload


def error_event(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    return server_event(
        "error",
        error={
            "type": error_type,
            "code": code,
            "message": message,
            "param": param,
        },
    )


def session_created_event(session: dict[str, Any]) -> dict[str, Any]:
    return server_event("session.created", session=session)


def session_updated_event(session: dict[str, Any]) -> dict[str, Any]:
    return server_event("session.updated", session=session)


def speech_started_event(audio_start_ms: int, item_id: str) -> dict[str, Any]:
    return server_event(
        "input_audio_buffer.speech_started",
        audio_start_ms=audio_start_ms,
        item_id=item_id,
    )


def speech_stopped_event(audio_end_ms: int, item_id: str) -> dict[str, Any]:
    return server_event(
        "input_audio_buffer.speech_stopped",
        audio_end_ms=audio_end_ms,
        item_id=item_id,
    )


def buffer_committed_event(item_id: str) -> dict[str, Any]:
    return server_event("input_audio_buffer.committed", item_id=item_id)


def transcription_delta_event(item_id: str, delta: str, content_index: int = 0) -> dict[str, Any]:
    return server_event(
        "conversation.item.input_audio_transcription.delta",
        item_id=item_id,
        content_index=content_index,
        delta=delta,
    )


def transcription_completed_event(
    item_id: str,
    transcript: str,
    *,
    content_index: int = 0,
) -> dict[str, Any]:
    return server_event(
        "conversation.item.input_audio_transcription.completed",
        item_id=item_id,
        content_index=content_index,
        transcript=transcript,
    )


def default_session_config(
    *,
    model_id: str,
    language: str,
    sample_rate: int,
    input_audio_format: str = "pcm16",
    vad_threshold: float,
    silence_duration_ms: int,
) -> dict[str, Any]:
    return {
        "id": f"sess_{uuid.uuid4().hex[:24]}",
        "object": "realtime.session",
        "model": model_id,
        "modalities": ["text"],
        "input_audio_format": input_audio_format,
        "input_audio_transcription": {
            "model": model_id,
            "language": language,
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": vad_threshold,
            "silence_duration_ms": silence_duration_ms,
            "sample_rate": sample_rate,
        },
        "created_at": int(time.time()),
    }


def parse_client_event(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    event_type = raw.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("Missing or invalid event type")

    return event_type, raw
