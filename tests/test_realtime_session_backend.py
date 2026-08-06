from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.errors import OpenAIAPIError
from app.model_registry import BACKEND_FASTER_WHISPER, BACKEND_NEMO
from app.services.realtime_session import RealtimeSession


def _session_with_registry(models: dict[str, dict]) -> RealtimeSession:
    registry = MagicMock()

    def get(model_id: str) -> dict:
        if model_id not in models:
            raise OpenAIAPIError("missing", param="model", code="model_not_found")
        return models[model_id]

    registry.get.side_effect = get
    asr = MagicMock()
    asr.registry = registry
    diarization = MagicMock()
    send_event = AsyncMock()
    return RealtimeSession(
        asr_service=asr,
        diarization_service=diarization,
        model_id="w1",
        send_event=send_event,
    )


@pytest.mark.asyncio
async def test_ensure_realtime_backend_allows_nemo() -> None:
    session = _session_with_registry(
        {
            "n1": {
                "id": "n1",
                "backend": BACKEND_NEMO,
                "capabilities": {"transcription"},
            }
        }
    )
    assert await session._ensure_realtime_backend("n1") is True
    session.send_event.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_realtime_backend_allows_whisper() -> None:
    session = _session_with_registry(
        {
            "w1": {
                "id": "w1",
                "backend": BACKEND_FASTER_WHISPER,
                "capabilities": {"transcription"},
            }
        }
    )
    assert await session._ensure_realtime_backend("w1") is True


@pytest.mark.asyncio
async def test_ensure_realtime_backend_rejects_unknown() -> None:
    session = _session_with_registry(
        {
            "x1": {
                "id": "x1",
                "backend": "onnx",
                "capabilities": {"transcription"},
            }
        }
    )
    assert await session._ensure_realtime_backend("x1") is False
    event = session.send_event.await_args.args[0]
    assert event["error"]["code"] == "unsupported_backend"


@pytest.mark.asyncio
async def test_session_update_switches_to_nemo_model() -> None:
    session = _session_with_registry(
        {
            "w1": {
                "id": "w1",
                "backend": BACKEND_FASTER_WHISPER,
                "capabilities": {"transcription"},
            },
            "n1": {
                "id": "n1",
                "backend": BACKEND_NEMO,
                "capabilities": {"transcription"},
            },
        }
    )
    await session._handle_session_update({"session": {"model": "n1"}})
    assert session.model_id == "n1"


@pytest.mark.asyncio
async def test_session_update_rejects_unknown_backend_keeps_previous() -> None:
    session = _session_with_registry(
        {
            "w1": {
                "id": "w1",
                "backend": BACKEND_FASTER_WHISPER,
                "capabilities": {"transcription"},
            },
            "x1": {
                "id": "x1",
                "backend": "onnx",
                "capabilities": {"transcription"},
            },
        }
    )
    await session._handle_session_update({"session": {"model": "x1"}})
    assert session.model_id == "w1"
    event = session.send_event.await_args.args[0]
    assert event["error"]["code"] == "unsupported_backend"
