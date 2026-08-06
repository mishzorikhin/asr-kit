from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.errors import OpenAIAPIError
from app.model_registry import (
    BACKEND_FASTER_WHISPER,
    BACKEND_NEMO,
    ModelRegistry,
    backend_supports_realtime,
    load_models_config,
    normalize_backend,
    resolve_asr_model_path,
    validate_local_asr_path,
    validate_local_nemo_path,
)


def test_normalize_backend_defaults_to_faster_whisper() -> None:
    assert normalize_backend(None, "m1") == BACKEND_FASTER_WHISPER
    assert normalize_backend("", "m1") == BACKEND_FASTER_WHISPER
    assert normalize_backend("NeMo", "m1") == BACKEND_NEMO
    assert normalize_backend("FASTER-WHISPER", "m1") == BACKEND_FASTER_WHISPER


def test_normalize_backend_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="unsupported backend"):
        normalize_backend("onnx", "m1")


def test_backend_supports_realtime() -> None:
    assert backend_supports_realtime(BACKEND_FASTER_WHISPER) is True
    assert backend_supports_realtime(BACKEND_NEMO) is True
    assert backend_supports_realtime("onnx") is False


def test_validate_local_asr_path_requires_absolute_existing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="absolute"):
        validate_local_asr_path("m", "relative/path")
    with pytest.raises(RuntimeError, match="does not exist"):
        validate_local_asr_path("m", str(missing))

    present = tmp_path / "model"
    present.mkdir()
    validate_local_asr_path("m", str(present))


def test_validate_local_nemo_path_rules(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="absolute"):
        validate_local_nemo_path("m", "x.nemo")

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="not a directory"):
        validate_local_nemo_path("m", str(directory))

    wrong = tmp_path / "model.bin"
    wrong.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="\\.nemo file"):
        validate_local_nemo_path("m", str(wrong))

    ok = tmp_path / "model.nemo"
    ok.write_bytes(b"nemo")
    validate_local_nemo_path("m", str(ok))


def test_resolve_asr_model_path_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "models--demo"
    snapshot = root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("abc123", encoding="utf-8")

    assert resolve_asr_model_path(str(root)) == str(snapshot)
    assert resolve_asr_model_path(str(tmp_path / "plain")) == str(tmp_path / "plain")


def _write_config(path: Path, models: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")


def test_load_models_config_whisper_and_nemo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper_dir = tmp_path / "whisper-ct2"
    whisper_dir.mkdir()
    nemo_file = tmp_path / "model.nemo"
    nemo_file.write_bytes(b"fake-nemo")

    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "whisper-demo",
                "path": str(whisper_dir),
                "capabilities": ["transcription"],
            },
            {
                "id": "nemo-demo",
                "backend": "nemo",
                "path": str(nemo_file),
                "owned_by": "nvidia-nemo",
                "capabilities": ["transcription"],
                "metadata": {"sample_rate": 16000},
            },
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    models = load_models_config()
    assert models["whisper-demo"]["backend"] == BACKEND_FASTER_WHISPER
    assert models["nemo-demo"]["backend"] == BACKEND_NEMO
    assert models["nemo-demo"]["owned_by"] == "nvidia-nemo"
    assert models["nemo-demo"]["metadata"]["sample_rate"] == 16000


def test_load_models_config_with_diarization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    diar_dir = tmp_path / "pyannote"
    diar_dir.mkdir()
    (diar_dir / "config.yaml").write_text("pipeline: demo\n", encoding="utf-8")

    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "whisper-diarize",
                "path": str(whisper_dir),
                "capabilities": ["transcription", "diarization"],
                "diarization_model": str(diar_dir),
            }
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    models = load_models_config()
    assert "diarization" in models["whisper-diarize"]["capabilities"]
    assert models["whisper-diarize"]["diarization_model"] == str(diar_dir)


def test_load_models_config_rejects_diarization_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    diar_dir = tmp_path / "pyannote"
    diar_dir.mkdir()

    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "bad-diarize",
                "path": str(whisper_dir),
                "capabilities": ["transcription", "diarization"],
                "diarization_model": str(diar_dir),
            }
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="config.yaml"):
        load_models_config()


def test_load_models_config_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {"id": "same", "path": str(whisper_dir), "capabilities": ["transcription"]},
            {"id": "same", "path": str(whisper_dir), "capabilities": ["transcription"]},
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="Duplicate model id"):
        load_models_config()


def test_load_models_config_rejects_nemo_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nemo_dir = tmp_path / "nemo-dir"
    nemo_dir.mkdir()
    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "nemo-bad",
                "backend": "nemo",
                "path": str(nemo_dir),
                "capabilities": ["transcription"],
            }
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="\\.nemo file"):
        load_models_config()


def test_load_models_config_rejects_unsupported_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "bad",
                "path": str(whisper_dir),
                "capabilities": ["transcription", "translation"],
            }
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    with pytest.raises(RuntimeError, match="unsupported capabilities"):
        load_models_config()


def test_model_registry_get_list_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    config_path = tmp_path / "models.yaml"
    _write_config(
        config_path,
        [
            {
                "id": "alpha",
                "path": str(whisper_dir),
                "owned_by": "local",
                "created": 10,
                "capabilities": ["transcription"],
            },
            {
                "id": "beta",
                "path": str(whisper_dir),
                "capabilities": ["transcription"],
            },
        ],
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config_path)

    registry = ModelRegistry()
    assert registry.get("alpha")["id"] == "alpha"
    listed = registry.list()
    assert [item["id"] for item in listed] == ["alpha", "beta"]
    assert listed[0]["object"] == "model"
    assert listed[0]["owned_by"] == "local"

    with pytest.raises(OpenAIAPIError) as exc_info:
        registry.get("missing")
    assert exc_info.value.code == "model_not_found"
