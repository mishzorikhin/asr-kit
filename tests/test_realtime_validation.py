from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.errors import OpenAIAPIError
from app.model_registry import BACKEND_FASTER_WHISPER, BACKEND_NEMO
from app.routers.realtime import _validate_realtime_model


def test_validate_realtime_allows_whisper() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "w1",
        "backend": BACKEND_FASTER_WHISPER,
        "capabilities": {"transcription"},
    }
    assert _validate_realtime_model(registry, "w1")["id"] == "w1"


def test_validate_realtime_allows_nemo() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "n1",
        "backend": BACKEND_NEMO,
        "capabilities": {"transcription"},
    }
    assert _validate_realtime_model(registry, "n1")["backend"] == BACKEND_NEMO


def test_validate_realtime_allows_diarize_capable_whisper() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "w-diarize",
        "backend": BACKEND_FASTER_WHISPER,
        "capabilities": {"transcription", "diarization"},
    }
    assert _validate_realtime_model(registry, "w-diarize")["id"] == "w-diarize"


def test_validate_realtime_rejects_unknown_backend() -> None:
    registry = MagicMock()
    registry.get.return_value = {
        "id": "x1",
        "backend": "onnx",
        "capabilities": {"transcription"},
    }
    with pytest.raises(OpenAIAPIError) as exc_info:
        _validate_realtime_model(registry, "x1")
    assert exc_info.value.code == "unsupported_backend"


def test_validate_realtime_propagates_model_not_found() -> None:
    registry = MagicMock()
    registry.get.side_effect = OpenAIAPIError(
        "missing",
        param="model",
        code="model_not_found",
    )
    with pytest.raises(OpenAIAPIError) as exc_info:
        _validate_realtime_model(registry, "nope")
    assert exc_info.value.code == "model_not_found"
