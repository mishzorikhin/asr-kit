from __future__ import annotations

import logging
from pathlib import Path

import yaml

from app.config import REALTIME_SPEAKER_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def resolve_speaker_embedding_model(diarization_model: str | None) -> str | None:
    if REALTIME_SPEAKER_EMBEDDING_MODEL:
        return REALTIME_SPEAKER_EMBEDDING_MODEL

    if not diarization_model:
        return None

    pipeline_dir = Path(diarization_model)
    sibling = pipeline_dir / "embedding"
    if sibling.is_dir():
        return str(sibling)

    config_path = pipeline_dir / "config.yaml"
    if not config_path.exists():
        return None

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not read diarization config for embedding model: %s", exc)
        return None

    embedding = config.get("pipeline", {}).get("params", {}).get("embedding")
    if isinstance(embedding, str):
        candidate = Path(embedding)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        nested = pipeline_dir / embedding
        if nested.exists():
            return str(nested)

    return None
