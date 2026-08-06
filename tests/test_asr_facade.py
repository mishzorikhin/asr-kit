from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import OpenAIAPIError
from app.model_registry import BACKEND_FASTER_WHISPER, BACKEND_NEMO
from app.services.asr import ASRService


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.transcribe_calls: list[dict[str, Any]] = []
        self.array_calls: list[dict[str, Any]] = []
        self.unload_calls = 0

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        self.transcribe_calls.append({"audio_path": audio_path, **kwargs})
        return {"model": kwargs["model_id"], "backend": self.name, "segments": []}

    def transcribe_array(self, audio: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        self.array_calls.append({"audio": audio, **kwargs})
        return {
            "model": kwargs["model_id"],
            "backend": self.name,
            "segments": [],
            "text": "ok",
        }

    def unload_idle_models(self, max_idle_seconds: int = 0) -> int:
        self.unload_calls += 1
        return 1


def _registry_with(models: dict[str, dict[str, Any]]) -> MagicMock:
    registry = MagicMock()

    def get(model_id: str) -> dict[str, Any]:
        if model_id not in models:
            raise OpenAIAPIError("missing", param="model", code="model_not_found")
        return models[model_id]

    registry.get.side_effect = get
    return registry


def test_facade_dispatches_whisper() -> None:
    registry = _registry_with(
        {
            "w1": {
                "id": "w1",
                "backend": BACKEND_FASTER_WHISPER,
                "path": "/tmp/w",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)
    whisper = _FakeBackend("whisper")
    service._whisper = whisper  # type: ignore[assignment]

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
        timestamp_granularities=["segment"],
    )

    assert result["backend"] == "whisper"
    assert whisper.transcribe_calls[0]["model_id"] == "w1"


def test_facade_dispatches_nemo() -> None:
    registry = _registry_with(
        {
            "n1": {
                "id": "n1",
                "backend": BACKEND_NEMO,
                "path": "/tmp/m.nemo",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)
    nemo = _FakeBackend("nemo")
    service._nemo = nemo

    result = service.transcribe(
        "/tmp/a.wav",
        model_id="n1",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=5,
        vad_filter=False,
        timestamp_granularities=["segment"],
    )

    assert result["backend"] == "nemo"
    assert nemo.transcribe_calls[0]["model_id"] == "n1"


def test_facade_dispatches_transcribe_array_nemo() -> None:
    registry = _registry_with(
        {
            "n1": {
                "id": "n1",
                "backend": BACKEND_NEMO,
                "path": "/tmp/m.nemo",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)
    nemo = _FakeBackend("nemo")
    service._nemo = nemo
    audio = np.zeros(100, dtype=np.float32)

    result = service.transcribe_array(
        audio,
        sample_rate=16000,
        model_id="n1",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=1,
    )

    assert result["backend"] == "nemo"
    assert result["text"] == "ok"
    assert nemo.array_calls[0]["sample_rate"] == 16000


def test_facade_defaults_missing_backend_to_whisper() -> None:
    registry = _registry_with(
        {
            "w1": {
                "id": "w1",
                "path": "/tmp/w",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)
    whisper = _FakeBackend("whisper")
    service._whisper = whisper  # type: ignore[assignment]

    service.transcribe(
        "/tmp/a.wav",
        model_id="w1",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="int8",
        beam_size=5,
        vad_filter=False,
        timestamp_granularities=["segment"],
    )
    assert whisper.transcribe_calls


def test_facade_unsupported_backend() -> None:
    registry = _registry_with(
        {
            "x1": {
                "id": "x1",
                "backend": "onnx",
                "path": "/tmp/x",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe(
            "/tmp/a.wav",
            model_id="x1",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="int8",
            beam_size=5,
            vad_filter=False,
            timestamp_granularities=["segment"],
        )

    assert exc_info.value.code == "unsupported_backend"


def test_facade_nemo_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry_with(
        {
            "n1": {
                "id": "n1",
                "backend": BACKEND_NEMO,
                "path": "/tmp/m.nemo",
                "capabilities": {"transcription"},
            }
        }
    )
    service = ASRService(registry)

    import app.services.nemo_asr as nemo_mod

    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", False)

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe(
            "/tmp/a.wav",
            model_id="n1",
            language="ru",
            prompt=None,
            temperature=0.0,
            device="cpu",
            compute_type="float16",
            beam_size=5,
            vad_filter=False,
            timestamp_granularities=["segment"],
        )

    assert exc_info.value.code == "backend_unavailable"


def test_facade_unload_idle_models_sums_backends() -> None:
    registry = _registry_with({})
    service = ASRService(registry)
    whisper = _FakeBackend("whisper")
    nemo = _FakeBackend("nemo")
    service._whisper = whisper  # type: ignore[assignment]
    service._nemo = nemo

    assert service.unload_idle_models(10) == 2
    assert whisper.unload_calls == 1
    assert nemo.unload_calls == 1


def test_facade_unload_without_nemo_only_whisper() -> None:
    registry = _registry_with({})
    service = ASRService(registry)
    whisper = _FakeBackend("whisper")
    service._whisper = whisper  # type: ignore[assignment]
    assert service._nemo is None
    assert service.unload_idle_models(10) == 1
    assert whisper.unload_calls == 1
