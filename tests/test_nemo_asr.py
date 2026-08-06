from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.errors import OpenAIAPIError
from app.services import nemo_asr as nemo_mod
from app.services.nemo_asr import (
    NeMoASRService,
    _audio_duration_seconds,
    _hypothesis_text,
    _write_temp_wav,
)


def _make_service(registry: MagicMock | None = None) -> NeMoASRService:
    service = NeMoASRService.__new__(NeMoASRService)
    service.registry = registry or MagicMock()
    service._lock = __import__("threading").Lock()
    service._models = {}
    return service


def _patch_restore(monkeypatch: pytest.MonkeyPatch, fake_model: Any) -> None:
    def fake_restore_from(*, restore_path: str, map_location: Any = None) -> Any:
        return fake_model

    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)
    monkeypatch.setattr(nemo_mod, "ASRModel", SimpleNamespace(restore_from=fake_restore_from))


def test_hypothesis_text_variants() -> None:
    assert _hypothesis_text(None) == ""
    assert _hypothesis_text(" hello ") == "hello"
    assert _hypothesis_text(SimpleNamespace(text="  hi ")) == "hi"
    assert _hypothesis_text([SimpleNamespace(text="nested")]) == "nested"
    assert _hypothesis_text([[SimpleNamespace(text="deep")]]) == "deep"
    assert _hypothesis_text(42) == "42"


def test_write_temp_wav_and_duration(tmp_path: Path) -> None:
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    path = _write_temp_wav(samples, 16000)
    try:
        assert Path(path).exists()
        with wave.open(path, "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16000
            assert handle.getnframes() == 16000
        assert _audio_duration_seconds(path) == pytest.approx(1.0, abs=0.01)
    finally:
        Path(path).unlink(missing_ok=True)


def test_audio_duration_unknown_file_returns_zero(tmp_path: Path) -> None:
    junk = tmp_path / "not-audio.bin"
    junk.write_bytes(b"not a wav")
    assert _audio_duration_seconds(str(junk)) == 0.0


def test_nemo_init_requires_toolkit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="nemo_toolkit"):
        NeMoASRService(MagicMock())


def test_nemo_transcribe_builds_openai_shaped_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [SimpleNamespace(text="привет мир")]
    _patch_restore(monkeypatch, fake_model)
    monkeypatch.setattr(nemo_mod, "_audio_duration_seconds", lambda _path: 1.5)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)

    result = service.transcribe(
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
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 1.5
    assert result["words"] == []
    fake_model.transcribe.assert_called_once_with(["/tmp/audio.wav"], batch_size=1)
    fake_model.eval.assert_called_once()


def test_nemo_word_timestamps_soft_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [SimpleNamespace(text="тест")]
    _patch_restore(monkeypatch, fake_model)
    monkeypatch.setattr(nemo_mod, "_audio_duration_seconds", lambda _path: 0.8)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)

    result = service.transcribe(
        "/tmp/audio.wav",
        model_id="nemo-demo",
        language="ru",
        prompt="ignored",
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=5,
        vad_filter=True,
        timestamp_granularities=["word"],
    )

    assert result["segments"][0]["text"] == "тест"
    assert result["words"] == []


def test_nemo_transcribe_empty_hypothesis(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [""]
    _patch_restore(monkeypatch, fake_model)
    monkeypatch.setattr(nemo_mod, "_audio_duration_seconds", lambda _path: 2.0)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)

    result = service.transcribe(
        "/tmp/audio.wav",
        model_id="nemo-demo",
        language=None,
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=5,
        vad_filter=False,
        timestamp_granularities=["segment"],
    )

    assert result["segments"] == []
    assert result["language_probability"] == 0.0


def test_nemo_transcribe_wraps_decode_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("bad audio")
    _patch_restore(monkeypatch, fake_model)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe(
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

    assert exc_info.value.code == "audio_decode_failed"


def test_nemo_transcribe_maps_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
    _patch_restore(monkeypatch, fake_model)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe(
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

    assert exc_info.value.code == "insufficient_gpu_memory"
    assert exc_info.value.status_code == 503


def test_nemo_transcribe_array_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [SimpleNamespace(text="из массива")]
    _patch_restore(monkeypatch, fake_model)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }
    service = _make_service(registry)
    audio = np.zeros(8000, dtype=np.float32)

    result = service.transcribe_array(
        audio,
        sample_rate=16000,
        model_id="nemo-demo",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="float16",
        beam_size=5,
        timestamp_granularities=["word"],
    )

    assert result["text"] == "из массива"
    assert result["duration"] == pytest.approx(0.5)
    assert result["words"] == []
    # temp wav path was passed into model.transcribe
    called_path = fake_model.transcribe.call_args[0][0][0]
    assert called_path.endswith(".wav")
    assert not Path(called_path).exists()


def test_nemo_transcribe_array_empty() -> None:
    service = _make_service()
    service.registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe_array(
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
    service = _make_service()
    service.registry.get.return_value = {
        "id": "nemo-demo",
        "path": "/tmp/model.nemo",
        "backend": "nemo",
        "capabilities": {"transcription"},
    }

    with pytest.raises(OpenAIAPIError) as exc_info:
        service.transcribe_array(
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


def test_nemo_unload_idle_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)
    service = _make_service()
    fake = MagicMock()
    service._models[("/tmp/model.nemo", "cpu")] = nemo_mod.CachedNeMoModel(
        model=fake,
        last_used_at=0.0,
        active_uses=0,
    )

    assert service.unload_idle_models(max_idle_seconds=1) == 1
    assert service._models == {}


def test_nemo_unload_idle_skips_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)
    service = _make_service()
    service._models[("/tmp/model.nemo", "cpu")] = nemo_mod.CachedNeMoModel(
        model=MagicMock(),
        last_used_at=0.0,
        active_uses=1,
    )

    assert service.unload_idle_models(max_idle_seconds=1) == 0
    assert len(service._models) == 1


def test_nemo_unload_idle_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemo_mod, "NEMO_AVAILABLE", True)
    service = _make_service()
    service._models[("/tmp/model.nemo", "cpu")] = nemo_mod.CachedNeMoModel(
        model=MagicMock(),
        last_used_at=0.0,
        active_uses=0,
    )
    assert service.unload_idle_models(max_idle_seconds=0) == 0
