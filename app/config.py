import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_CUDA_DEVICE_ALIASES = {"cuda", "gpu"}
_CPU_COMPUTE_TYPE_FALLBACKS = {
    "float16": "int8",
}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE = os.getenv("DEFAULT_COMPUTE_TYPE", "float16")


def resolve_device(device: str | None = None) -> str:
    requested = (device or DEFAULT_DEVICE).strip().lower()
    if requested in _CUDA_DEVICE_ALIASES:
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "Requested device=%r but no CUDA-capable device is available; using cpu",
            device or DEFAULT_DEVICE,
        )
        return "cpu"
    return requested


def resolve_compute_type(
    compute_type: str | None = None,
    *,
    device: str | None = None,
) -> str:
    resolved_device = resolve_device(device)
    requested = (compute_type or DEFAULT_COMPUTE_TYPE).strip().lower()
    fallback = _CPU_COMPUTE_TYPE_FALLBACKS.get(requested)
    if resolved_device == "cpu" and fallback is not None:
        logger.warning(
            "compute_type=%r is not supported on cpu; using %r",
            requested,
            fallback,
        )
        return fallback
    return requested
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ru")
DEFAULT_DIARIZATION_MODEL = os.getenv(
    "DEFAULT_DIARIZATION_MODEL",
    "/workspace/models/pyannote/speaker-diarization-community-1",
)
MODELS_CONFIG_PATH = Path(os.getenv("MODELS_CONFIG_PATH", "/workspace/config/models.yaml"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_IDLE_TTL_SECONDS = int(os.getenv("MODEL_IDLE_TTL_SECONDS", str(10 * 60)))
MODEL_UNLOAD_INTERVAL_SECONDS = int(os.getenv("MODEL_UNLOAD_INTERVAL_SECONDS", "30"))
MODEL_UNLOAD_AFTER_REQUEST = env_bool("MODEL_UNLOAD_AFTER_REQUEST", False)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}

OPENAI_RESPONSE_FORMATS = {"diarized_json", "json", "text", "srt", "verbose_json", "vtt"}
DIARIZED_RESPONSE_FORMATS = {"diarized_json", "json", "text"}
TRANSCRIPTION_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}

# Realtime WebSocket transcription (PCM16 mono @ 16 kHz; OpenAI Realtime uses 24 kHz).
REALTIME_SAMPLE_RATE = int(os.getenv("REALTIME_SAMPLE_RATE", "16000"))
REALTIME_MAX_BUFFER_SEC = float(os.getenv("REALTIME_MAX_BUFFER_SEC", "60"))
REALTIME_MIN_SEGMENT_MS = int(os.getenv("REALTIME_MIN_SEGMENT_MS", "300"))
REALTIME_INITIAL_PROMPT_CHARS = int(os.getenv("REALTIME_INITIAL_PROMPT_CHARS", "200"))
REALTIME_VAD_THRESHOLD = float(os.getenv("REALTIME_VAD_THRESHOLD", "0.012"))
REALTIME_SILENCE_DURATION_MS = int(os.getenv("REALTIME_SILENCE_DURATION_MS", "700"))
REALTIME_BEAM_SIZE = int(os.getenv("REALTIME_BEAM_SIZE", "1"))
REALTIME_WS_IDLE_TIMEOUT_SEC = float(os.getenv("REALTIME_WS_IDLE_TIMEOUT_SEC", "300"))
