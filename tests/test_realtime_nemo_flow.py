from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.model_registry import BACKEND_NEMO
from app.services.realtime_session import RealtimeSession


@pytest.mark.asyncio
async def test_realtime_commit_uses_nemo_backend() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "n1",
        "backend": BACKEND_NEMO,
        "capabilities": {"transcription"},
        "path": "/tmp/m.nemo",
    }
    asr = MagicMock()
    asr.registry = registry
    asr.transcribe_array.return_value = {
        "model": "n1",
        "language": "ru",
        "duration": 0.2,
        "segments": [{"id": 0, "start": 0.0, "end": 0.2, "text": "тест", "words": []}],
        "words": [],
        "text": "тест",
    }

    send_event = AsyncMock()
    session = RealtimeSession(
        asr_service=asr,
        diarization_service=None,
        model_id="n1",
        send_event=send_event,
        device="cpu",
        compute_type="float16",
        min_segment_ms=50,
    )

    audio = np.ones(int(0.2 * session.sample_rate), dtype=np.float32) * 0.2
    session.buffer.append_float32(audio)

    await session._handle_commit(force=True)

    asr.transcribe_array.assert_called_once()
    kwargs = asr.transcribe_array.call_args.kwargs
    assert kwargs["model_id"] == "n1"
    assert kwargs["sample_rate"] == 16000

    completed = [
        call.args[0]
        for call in send_event.await_args_list
        if call.args[0].get("type")
        == "conversation.item.input_audio_transcription.completed"
    ]
    assert completed
    assert completed[0]["transcript"] == "тест"
