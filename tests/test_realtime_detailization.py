import numpy as np

from app.openai_realtime_events import (
    speaker_assigned_event,
    transcription_completed_event,
)
from app.services.realtime_speaker_tracker import OnlineSpeakerCluster, cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vector, vector) == 1.0


def test_online_speaker_cluster_assigns_new_speaker() -> None:
    cluster = OnlineSpeakerCluster(similarity_threshold=0.9, known_speaker_names=["Alice"])
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    speaker_a, confidence_a = cluster.assign(first)
    speaker_b, confidence_b = cluster.assign(second)

    assert speaker_a == "Alice"
    assert confidence_a == 1.0
    assert speaker_b == "B"
    assert confidence_b == 1.0


def test_online_speaker_cluster_reuses_similar_embedding() -> None:
    cluster = OnlineSpeakerCluster(similarity_threshold=0.8)
    first = np.array([1.0, 0.1, 0.0], dtype=np.float32)
    second = np.array([0.95, 0.15, 0.0], dtype=np.float32)

    speaker_a, _ = cluster.assign(first)
    speaker_b, confidence = cluster.assign(second)

    assert speaker_a == speaker_b == "A"
    assert confidence >= 0.8


def test_transcription_completed_event_includes_words_and_speaker() -> None:
    event = transcription_completed_event(
        "item_1",
        "hello world",
        words=[{"word": "hello", "start": 0.0, "end": 0.4}],
        speaker="A",
        speaker_confidence=0.91,
        speaker_provisional=True,
    )

    assert event["type"] == "conversation.item.input_audio_transcription.completed"
    assert event["transcript"] == "hello world"
    assert event["words"][0]["word"] == "hello"
    assert event["speaker"] == "A"
    assert event["speaker_provisional"] is True


def test_speaker_assigned_event_shape() -> None:
    event = speaker_assigned_event("item_1", "B", confidence=0.77, provisional=True)

    assert event["type"] == "conversation.item.input_audio_transcription.speaker_assigned"
    assert event["speaker"] == "B"
    assert event["confidence"] == 0.77
    assert event["provisional"] is True
