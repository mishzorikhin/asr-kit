from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import OpenAIAPIError
from app.services import whisper_asr as whisper_mod
from app.services.whisper_asr import WhisperASRService


def _segment(
    *,
    text: str,
    start: float = 0.0,
    end: float = 1.0,
    words: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        seek=0,
        tokens=[1, 2],
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.01,
        words=words or [],
    )


def test_whisper_transcribe_builds_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    info = SimpleNamespace(language="ru", language_probability=0.99, duration=1.2)
    fake_model.transcribe.return_value = (
        iter(
            [
                _segment(
                    text=" привет ",
                    words=[SimpleNamespace(word="привет", start=0.0, end=0.5)],
                )
            ]
        ),
        info,
    )

    monkeypatch.setattr(
        whisper_mod,
        "WhisperModel",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(whisper_mod, "resolve_asr_model_path", lambda path: path)
    monkeypatch.setattr(whisper_mod, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(
        whisper_mod,
        "resolve_compute_type",
        lambda compute_type, device=None: "int8",
    )

    registry = MagicMock()
    registry.get.return_value = {
        "id": "w1",
        "path": "/tmp/whisper",
        "backend": "faster-whisper",
        "capabilities": {"transcription"},
    }
    service = WhisperASRService(registry)

    result = service.transcribe(
        "/tmp/a.wav",
        model_id="w1",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="int8",
        beam_size=5,
        vad_filter=False,
        timestamp_granularities=["word"],
    )

    assert result["language"] == "ru"
    assert result["duration"] == 1.2
    assert result["segments"][0]["text"] == "привет"
    assert result["words"][0]["word"] == "привет"


def test_whisper_transcribe_array_empty() -> None:
    service = WhisperASRService(MagicMock())
    service.registry.get.return_value = {
        "id": "w1",
        "path": "/tmp/whisper",
        "capabilities": {"transcription"},
    }
    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe_array(
            np.array([], dtype=np.float32),
            sample_rate=16000,
            model_id="w1",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="int8",
            beam_size=1,
        )
    assert exc_info.value.code == "empty_audio"


def test_whisper_unload_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_mod, "resolve_device", lambda device: "cpu")
    service = WhisperASRService(MagicMock())
    service._models[("/tmp/w", "cpu", "int8")] = whisper_mod.CachedASRModel(
        model=MagicMock(),
        last_used_at=0.0,
        active_uses=0,
    )
    assert service.unload_idle_models(max_idle_seconds=1) == 1
    assert service._models == {}
