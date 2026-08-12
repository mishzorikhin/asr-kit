from __future__ import annotations

import threading
import time
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


def _patch_common(monkeypatch: pytest.MonkeyPatch, fake_model_factory) -> None:
    monkeypatch.setattr(whisper_mod, "WhisperModel", fake_model_factory)
    monkeypatch.setattr(whisper_mod, "resolve_asr_model_path", lambda path: path)
    monkeypatch.setattr(whisper_mod, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(
        whisper_mod,
        "resolve_compute_type",
        lambda compute_type, device=None: "int8",
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

    _patch_common(monkeypatch, lambda *args, **kwargs: fake_model)

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
    service._pools[("/tmp/w", "cpu", "int8")] = [
        whisper_mod.CachedASRModel(
            model=MagicMock(),
            last_used_at=0.0,
            active_uses=0,
            replica_id=0,
        )
    ]
    assert service.unload_idle_models(max_idle_seconds=1) == 1
    assert service._pools == {}


def test_autoscale_spawns_second_replica_when_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_mod, "WHISPER_AUTOSCALE_ENABLED", True)
    monkeypatch.setattr(whisper_mod, "WHISPER_MAX_REPLICAS", 2)
    monkeypatch.setattr(whisper_mod, "WHISPER_REPLICA_WAIT_SECONDS", 5)
    monkeypatch.setattr(whisper_mod, "MODEL_UNLOAD_AFTER_REQUEST", False)

    created: list[dict[str, Any]] = []
    hold = threading.Event()
    second_started = threading.Event()

    def factory(*args: Any, **kwargs: Any) -> MagicMock:
        created.append(dict(kwargs))
        model = MagicMock()
        info = SimpleNamespace(language="ru", language_probability=1.0, duration=0.5)

        def transcribe(*_a: Any, **_k: Any):
            if len(created) == 1:
                second_started.wait(timeout=2)
                hold.wait(timeout=2)
            return (iter([_segment(text="ok")]), info)

        model.transcribe.side_effect = transcribe
        return model

    _patch_common(monkeypatch, factory)

    registry = MagicMock()
    registry.get.return_value = {
        "id": "w1",
        "path": "/tmp/whisper",
        "backend": "faster-whisper",
        "capabilities": {"transcription"},
    }
    service = WhisperASRService(registry)

    errors: list[BaseException] = []

    def run_first() -> None:
        try:
            service.transcribe(
                "/tmp/a.wav",
                model_id="w1",
                language="ru",
                prompt=None,
                temperature=0.0,
                device="cpu",
                compute_type="int8",
                beam_size=1,
                vad_filter=False,
                timestamp_granularities=["segment"],
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=run_first)
    first.start()

    # Wait until first replica is acquired/busy.
    for _ in range(50):
        status = service.replica_status()
        if status and status[0]["busy"]:
            break
        time.sleep(0.02)
    else:
        hold.set()
        first.join(timeout=1)
        pytest.fail("first replica did not become busy")

    second_started.set()
    service.transcribe(
        "/tmp/b.wav",
        model_id="w1",
        language="ru",
        prompt=None,
        temperature=0.0,
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=False,
        timestamp_granularities=["segment"],
    )
    hold.set()
    first.join(timeout=2)

    assert not errors
    assert len(created) == 2
    assert len(service.replica_status()) == 2


def test_autoscale_disabled_shares_single_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_mod, "WHISPER_AUTOSCALE_ENABLED", False)
    monkeypatch.setattr(whisper_mod, "WHISPER_MAX_REPLICAS", 4)
    monkeypatch.setattr(whisper_mod, "MODEL_UNLOAD_AFTER_REQUEST", False)

    created: list[MagicMock] = []

    def factory(*args: Any, **kwargs: Any) -> MagicMock:
        model = MagicMock()
        info = SimpleNamespace(language="ru", language_probability=1.0, duration=0.1)
        model.transcribe.return_value = (iter([_segment(text="x")]), info)
        created.append(model)
        return model

    _patch_common(monkeypatch, factory)
    registry = MagicMock()
    registry.get.return_value = {
        "id": "w1",
        "path": "/tmp/whisper",
        "capabilities": {"transcription"},
    }
    service = WhisperASRService(registry)

    with service.use_model("/tmp/whisper", "cpu", "int8") as _model1:
        with service.use_model("/tmp/whisper", "cpu", "int8") as _model2:
            status = service.replica_status()
            assert len(status) == 1
            assert status[0]["active_uses"] == 2

    assert len(created) == 1


def test_autoscale_busy_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_mod, "WHISPER_AUTOSCALE_ENABLED", True)
    monkeypatch.setattr(whisper_mod, "WHISPER_MAX_REPLICAS", 1)
    monkeypatch.setattr(whisper_mod, "WHISPER_REPLICA_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(whisper_mod, "MODEL_UNLOAD_AFTER_REQUEST", False)

    def factory(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock()

    _patch_common(monkeypatch, factory)
    service = WhisperASRService(MagicMock())

    with service.use_model("/tmp/whisper", "cpu", "int8"):
        with pytest.raises(OpenAIAPIError) as exc_info:
            with service.use_model("/tmp/whisper", "cpu", "int8"):
                pass
    assert exc_info.value.code == "asr_replicas_busy"
