from __future__ import annotations

import base64
import logging

import numpy as np

logger = logging.getLogger(__name__)


def pcm16_base64_to_float32(audio_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio payload: {exc}") from exc

    if len(raw) % 2 != 0:
        raise ValueError("PCM16 payload length must be even")

    if not raw:
        return np.array([], dtype=np.float32)

    pcm16 = np.frombuffer(raw, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def chunk_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class AudioBuffer:
    """Accumulates mono PCM float32 samples with a fixed maximum duration."""

    def __init__(self, *, sample_rate: int, max_duration_sec: float) -> None:
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(sample_rate * max_duration_sec))
        self._samples = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._samples.size)

    @property
    def duration_sec(self) -> float:
        return len(self) / self.sample_rate

    def clear(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)

    def append_float32(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return

        normalized = np.asarray(chunk, dtype=np.float32).reshape(-1)
        combined = np.concatenate([self._samples, normalized])
        if combined.size > self.max_samples:
            dropped = combined.size - self.max_samples
            logger.debug("Audio buffer truncated %d samples (%.2fs)", dropped, dropped / self.sample_rate)
            combined = combined[-self.max_samples :]
        self._samples = combined

    def append_pcm16_base64(self, audio_b64: str) -> np.ndarray:
        chunk = pcm16_base64_to_float32(audio_b64)
        self.append_float32(chunk)
        return chunk

    def extract_range(self, start: int, end: int) -> np.ndarray:
        start = max(0, min(start, len(self)))
        end = max(start, min(end, len(self)))
        return self._samples[start:end].copy()

    def extract_all(self) -> np.ndarray:
        return self._samples.copy()

    def trim_prefix(self, sample_count: int) -> None:
        sample_count = max(0, min(sample_count, len(self)))
        if sample_count <= 0:
            return
        self._samples = self._samples[sample_count:]

    def ms_to_samples(self, ms: int) -> int:
        return int(self.sample_rate * ms / 1000)

    def samples_to_ms(self, samples: int) -> int:
        return int(samples * 1000 / self.sample_rate)
