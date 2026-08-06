from __future__ import annotations

from app.openai_format import (
    format_json,
    format_openai_response,
    format_srt,
    format_vtt,
    join_segment_text,
    speaker_labels,
)


def _nemo_like_transcription() -> dict:
    return {
        "model": "nemo-demo",
        "language": "ru",
        "language_probability": 1.0,
        "duration": 1.5,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.5,
                "text": "привет мир",
                "seek": 0,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
                "words": [],
            }
        ],
        "words": [],
    }


def test_join_segment_text() -> None:
    assert join_segment_text(_nemo_like_transcription()["segments"]) == "привет мир"
    assert join_segment_text([{"text": ""}, {"text": "a"}, {"text": "b"}]) == "a b"


def test_format_json_from_nemo_result() -> None:
    response = format_json(_nemo_like_transcription(), include=[])
    assert response["text"] == "привет мир"
    assert response["usage"]["seconds"] == 2  # rounded


def test_format_srt_and_vtt_from_nemo_result() -> None:
    segments = _nemo_like_transcription()["segments"]
    srt = format_srt(segments)
    assert "привет мир" in srt
    assert "00:00:00,000 --> 00:00:01,500" in srt

    vtt = format_vtt(segments)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_format_openai_response_text() -> None:
    body = format_openai_response(
        _nemo_like_transcription(),
        response_format="text",
        known_speaker_names=[],
        include=[],
    )
    assert body == "привет мир"


def test_format_openai_response_verbose_json() -> None:
    body = format_openai_response(
        _nemo_like_transcription(),
        response_format="verbose_json",
        known_speaker_names=[],
        include=[],
    )
    assert isinstance(body, dict)
    assert body["language"] == "ru"
    assert body["duration"] == 1.5
    assert body["segments"][0]["text"] == "привет мир"


def test_speaker_labels_with_known_names() -> None:
    segments = [
        {"speaker": "SPEAKER_00", "text": "a"},
        {"speaker": "SPEAKER_01", "text": "b"},
        {"speaker": "SPEAKER_00", "text": "c"},
    ]
    labels = speaker_labels(segments, ["agent", "customer"])
    assert labels["SPEAKER_00"] == "agent"
    assert labels["SPEAKER_01"] == "customer"
