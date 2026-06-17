import os
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE = os.getenv("DEFAULT_COMPUTE_TYPE", "float16")
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
