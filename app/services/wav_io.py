"""Helpers for writing temporary mono WAV files used by ASR backends."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np


def write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
    """Write float32 mono audio to a temporary 16-bit PCM WAV file.

    Args:
        audio: Mono float32 samples in approximately [-1.0, 1.0].
        sample_rate: Sample rate in Hz.

    Returns:
        Absolute path to the temporary WAV file. Caller must delete it.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return tmp_path


def audio_duration_seconds(audio_path: str) -> float:
    """Return duration of a local audio file in seconds.

    Prefers the WAV header; falls back to soundfile when available.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Duration in seconds, or 0.0 when duration cannot be determined.
    """
    path = Path(audio_path)
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except wave.Error:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return 0.0
