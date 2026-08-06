from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import OpenAIAPIError
from app.services import nemo_asr as nemo_mod
from app.services.nemo_asr import NeMoASRService, _hypothesis_text


def test_hypothesis_text_variants() -> None:
    assert _hypothesis_text(None) == ""
    assert _hypothesis_text(" hello ") == "hello"
    assert _hypothesis_text(SimpleNamespace(text="  hi ")) == "hi"
    assert _hypothesis_text([SimpleNamespace(text="nested")]) == "nested"


def test_nemo_transcribe_builds_openai_shaped_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)

    fake_model = MagicMock()
    fake_model.transcribe.return_value = [SimpleNamespace(text="привет мир")]

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }

    service = NeMoASRService.__new__(NeMoASRService)
    service.registry = registry
    service._lock = __import__("threading").Lock()
    service._models = {}

    def fake_restore_from(*, restore_path: str, map_location: Any = None) -> Any:
        return fake_model

    monkeypatch.setattr(nemo_mod, "ASRModel", SimpleNamespace(restore_from=fake_restore_from))
    monkeypatch.setattr(nemo_mod, "_audio_duration_seconds", lambda _path: 1.5)

    result = NeMoASRService.transcribe(
        service,
        "/tmp/audio.wav",
        model_id="nemo-demo",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=5,
        vad_filter=False,
        timestamp_granularities=["segment"],
    )

    assert result["model"] == "nemo-demo"
    assert result["language"] == "ru"
    assert result["duration"] == 1.5
    assert result["segments"][0]["text"] == "привет мир"
    assert result["words"] == []
    fake_model.transcribe.assert_called_once()


def test_nemo_rejects_word_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)
    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = NeMoASRService.__new__(NeMoASRService)
    service.registry = registry
    service._lock = __import__("threading").Lock()
    service._models = {}

    with pytest.raises(OpenAIAPIError) as exc_info:
        NeMoASRService.transcribe(
            service,
            "/tmp/audio.wav",
            model_id="nemo-demo",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="float16",
            beam_size=5,
            vad_filter=False,
            timestamp_granularities=["word"],
        )

    assert exc_info.value.code == "unsupported_parameter"


def test_nemo_transcribe_array_empty() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = object.__new__(NeMoASRService)
    service.registry = registry

    with pytest.raises(OpenAIAPIError) as exc_info:
        NeMoASRService.transcribe_array(
            service,
            np.array([], dtype=np.float32),
            sample_rate=16000,
            model_id="nemo-demo",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="float16",
            beam_size=5,
        )

    assert exc_info.value.code == "empty_audio"


def test_nemo_transcribe_array_bad_sample_rate() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = object.__new__(NeMoASRService)
    service.registry = registry

    with pytest.raises(OpenAIAPIError) as exc_info:
        NeMoASRService.transcribe_array(
            service,
            np.zeros(1600, dtype=np.float32),
            sample_rate=8000,
            model_id="nemo-demo",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="float16",
            beam_size=5,
        )

    assert exc_info.value.code == "unsupported_audio_format"
