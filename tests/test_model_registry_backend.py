from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.model_registry import (
    BACKEND_FASTER_WHISPER,
    BACKEND_NEMO,
    backend_supports_realtime,
    load_models_config,
    normalize_backend,
)


def test_normalize_backend_defaults_to_faster_whisper() -> None:
    assert normalize_backend(None, "m1") == BACKEND_FASTER_WHISPER
    assert normalize_backend("", "m1") == BACKEND_FASTER_WHISPER
    assert normalize_backend("NeMo", "m1") == BACKEND_NEMO


def test_normalize_backend_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="unsupported backend"):
        normalize_backend("onnx", "m1")


def test_backend_supports_realtime() -> None:
    assert backend_supports_realtime(BACKEND_FASTER_WHISPER) is True
    assert backend_supports_realtime(BACKEND_NEMO) is False


def test_realtime_rejects_nemo_backend_contract() -> None:
    """Mirrors app.routers.realtime._validate_realtime_model backend gate."""
    assert not backend_supports_realtime(BACKEND_NEMO)


def test_load_models_config_whisper_and_nemo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper_dir = tmp_path / "whisper-ct2"
    whisper_dir.mkdir()
    nemo_file = tmp_path / "model.nemo"
    nemo_file.write_bytes(b"fake-nemo")

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "whisper-demo",
                        "path": str(whisper_dir),
                        "capabilities": ["transcription"],
                    },
                    {
                        "id": "nemo-demo",
                        "backend": "nemo",
                        "path": str(nemo_file),
                        "capabilities": ["transcription"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    models = load_models_config()
    assert models["whisper-demo"]["backend"] == BACKEND_FASTER_WHISPER
    assert models["nemo-demo"]["backend"] == BACKEND_NEMO


def test_load_models_config_rejects_nemo_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nemo_dir = tmp_path / "nemo-dir"
    nemo_dir.mkdir()
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "nemo-bad",
                        "backend": "nemo",
                        "path": str(nemo_dir),
                        "capabilities": ["transcription"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="\\.nemo file"):
        load_models_config()


def test_load_models_config_rejects_nemo_wrong_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_file = tmp_path / "model.bin"
    bad_file.write_bytes(b"x")
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "nemo-bad",
                        "backend": "nemo",
                        "path": str(bad_file),
                        "capabilities": ["transcription"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="\\.nemo file"):
        load_models_config()
