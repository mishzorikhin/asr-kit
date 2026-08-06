from __future__ import annotations

import pytest

from app.openai_realtime_events import (
    default_session_config,
    error_event,
    parse_client_event,
    session_created_event,
    transcription_completed_event,
)
from app.services.diarization import segment_overlap, speaker_for_segment
from app.tool_calls import list_tool_calls, new_request_id, record_tool_call


def test_error_event_shape() -> None:
    event = error_event("boom", code="unsupported_backend", param="model")
    assert event["type"] == "error"
    assert event["error"]["code"] == "unsupported_backend"
    assert event["error"]["param"] == "model"
    assert event["event_id"].startswith("evt_")


def test_session_created_and_default_config() -> None:
    config = default_session_config(
        model_id="nemo-demo",
        language="ru",
        sample_rate=16000,
        vad_threshold=0.01,
        silence_duration_ms=700,
        timestamp_granularities=["word"],
    )
    assert config["model"] == "nemo-demo"
    assert config["input_audio_transcription"]["model"] == "nemo-demo"
    assert config["turn_detection"]["sample_rate"] == 16000

    created = session_created_event(config)
    assert created["type"] == "session.created"
    assert created["session"]["model"] == "nemo-demo"


def test_transcription_completed_without_words() -> None:
    event = transcription_completed_event("item_1", "привет")
    assert event["type"] == "conversation.item.input_audio_transcription.completed"
    assert event["transcript"] == "привет"


def test_parse_client_event() -> None:
    event_type, raw = parse_client_event({"type": "session.update", "session": {}})
    assert event_type == "session.update"
    assert "session" in raw

    with pytest.raises(ValueError):
        parse_client_event({"session": {}})


def test_segment_overlap_and_speaker_for_segment() -> None:
    assert segment_overlap(0.0, 1.0, 0.5, 1.5) == pytest.approx(0.5)
    assert segment_overlap(0.0, 1.0, 2.0, 3.0) == 0.0

    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]
    assert speaker_for_segment(0.1, 0.9, diarization) == "A"
    assert speaker_for_segment(1.1, 1.8, diarization) == "B"


def test_tool_calls_record_and_list() -> None:
    request_id = new_request_id()
    assert len(request_id) == 12
    event = record_tool_call("test.unit.probe", status="ok", model="nemo-demo")
    assert event["name"] == "test.unit.probe"
    recent = list_tool_calls(limit=5)
    assert any(item["name"] == "test.unit.probe" for item in recent)
