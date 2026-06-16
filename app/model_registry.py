import logging
from pathlib import Path
from typing import Any

import yaml

from app.config import DEFAULT_DIARIZATION_MODEL, MODELS_CONFIG_PATH
from app.errors import OpenAIAPIError

logger = logging.getLogger(__name__)


def validate_local_asr_path(model_id: str, model_path: str) -> None:
    path = Path(model_path)

    if not path.is_absolute():
        raise RuntimeError(f"Model '{model_id}' path must be an absolute local path: {model_path}")
    if not path.exists():
        raise RuntimeError(f"Model '{model_id}' path does not exist: {model_path}")


def resolve_asr_model_path(model_path: str) -> str:
    path = Path(model_path)
    refs_main = path / "refs" / "main"

    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = path / "snapshots" / snapshot_id

        if snapshot_path.exists():
            return str(snapshot_path)

    return model_path


def load_models_config() -> dict[str, dict[str, Any]]:
    if not MODELS_CONFIG_PATH.exists():
        raise RuntimeError(f"Models config not found: {MODELS_CONFIG_PATH}")

    try:
        raw_config = yaml.safe_load(MODELS_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid models config {MODELS_CONFIG_PATH}: {exc}") from exc

    models = {}
    for raw_model in raw_config.get("models", []):
        model_id = raw_model.get("id")
        model_path = raw_model.get("path")

        if not model_id or not model_path:
            raise RuntimeError("Each configured model must have id and path")
        if model_id in models:
            raise RuntimeError(f"Duplicate model id in config: {model_id}")

        capabilities = set(raw_model.get("capabilities", ["transcription"]))
        unsupported = capabilities - {"transcription", "diarization"}
        if unsupported:
            raise RuntimeError(
                f"Model '{model_id}' has unsupported capabilities: "
                + ", ".join(sorted(unsupported))
            )
        if "transcription" not in capabilities:
            raise RuntimeError(f"Model '{model_id}' must include transcription capability")

        validate_local_asr_path(model_id, model_path)
        diarization_model = raw_model.get("diarization_model", DEFAULT_DIARIZATION_MODEL)
        if "diarization" in capabilities:
            diarization_path = Path(diarization_model)
            if not diarization_path.is_absolute():
                raise RuntimeError(
                    f"Model '{model_id}' diarization_model must be an absolute local path: "
                    f"{diarization_model}"
                )
            if not (diarization_path / "config.yaml").exists():
                raise RuntimeError(
                    f"Model '{model_id}' diarization_model must point to a pyannote "
                    f"pipeline directory with config.yaml: {diarization_model}"
                )

        models[model_id] = {
            "id": model_id,
            "path": model_path,
            "owned_by": raw_model.get("owned_by", "local"),
            "created": int(raw_model.get("created", 0)),
            "capabilities": capabilities,
            "diarization_model": diarization_model,
            "metadata": raw_model.get("metadata", {}),
        }

    if not models:
        raise RuntimeError("Models config does not contain any models")

    logger.info("Loaded %d configured model(s) from %s", len(models), MODELS_CONFIG_PATH)
    return models


class ModelRegistry:
    def __init__(self) -> None:
        self.models = load_models_config()

    def get(self, model_id: str) -> dict[str, Any]:
        model = self.models.get(model_id)
        if model is None:
            raise OpenAIAPIError(
                f"The model '{model_id}' does not exist or is not available for transcriptions.",
                param="model",
                code="model_not_found",
            )

        return model

    def list(self) -> list[dict[str, Any]]:
        return [self.model_object(model_id) for model_id in sorted(self.models)]

    def model_object(self, model_id: str) -> dict[str, Any]:
        model = self.get(model_id)
        return {
            "id": model_id,
            "object": "model",
            "created": model["created"],
            "owned_by": model["owned_by"],
        }
