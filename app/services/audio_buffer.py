"""PCM16 float32 audio buffering helpers for realtime transcription."""

from __future__ import annotations

import base64
import logging

import numpy as np

logger = logging.getLogger(__name__)


def pcm16_base64_to_float32(audio_b64: str) -> np.ndarray:
    """Decode a base64 PCM16 mono payload into float32 samples.

    Args:
        audio_b64: Base64-encoded little-endian int16 PCM.

    Returns:
        Mono float32 samples normalized to approximately [-1.0, 1.0].

    Raises:
        ValueError: If the payload is not valid base64 PCM16.
    """
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
    """Return RMS energy of a float32 audio chunk.

    Args:
        audio: Mono float32 samples.

    Returns:
        Root-mean-square amplitude, or 0.0 for an empty chunk.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class AudioBuffer:
    """Accumulates mono PCM float32 samples with an optional max duration."""

    def __init__(
        self,
        *,
        sample_rate: int,
        max_duration_sec: float | None,
    ) -> None:
        """Create a buffer.

        Args:
            sample_rate: Sample rate in Hz.
            max_duration_sec: Maximum retained duration, or None for unbounded.
        """
        self.sample_rate = sample_rate
        if max_duration_sec is None:
            self.max_samples: int | None = None
        else:
            self.max_samples = max(1, int(sample_rate * max_duration_sec))
        self._samples = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._samples.size)

    @property
    def duration_sec(self) -> float:
        """Current buffered duration in seconds."""
        return len(self) / self.sample_rate

    def clear(self) -> None:
        """Drop all buffered samples."""
        self._samples = np.zeros(0, dtype=np.float32)

    def append_float32(self, chunk: np.ndarray) -> int:
        """Append float32 samples, optionally truncating from the front.

        Args:
            chunk: Mono float32 samples to append.

        Returns:
            Number of samples dropped from the front due to max duration.
        """
        if chunk.size == 0:
            return 0

        normalized = np.asarray(chunk, dtype=np.float32).reshape(-1)
        combined = np.concatenate([self._samples, normalized])
        dropped = 0
        if self.max_samples is not None and combined.size > self.max_samples:
            dropped = combined.size - self.max_samples
            logger.debug(
                "Audio buffer truncated %d samples (%.2fs)",
                dropped,
                dropped / self.sample_rate,
            )
            combined = combined[-self.max_samples :]
        self._samples = combined
        return dropped

    def append_pcm16_base64(self, audio_b64: str) -> tuple[np.ndarray, int]:
        """Decode and append a PCM16 base64 chunk.

        Args:
            audio_b64: Base64-encoded PCM16 mono payload.

        Returns:
            Tuple of (decoded float32 chunk, samples dropped from front).
        """
        chunk = pcm16_base64_to_float32(audio_b64)
        dropped = self.append_float32(chunk)
        return chunk, dropped

    def extract_range(self, start: int, end: int) -> np.ndarray:
        """Return a copy of samples in ``[start, end)``."""
        start = max(0, min(start, len(self)))
        end = max(start, min(end, len(self)))
        return self._samples[start:end].copy()

    def extract_all(self) -> np.ndarray:
        """Return a copy of the full buffer."""
        return self._samples.copy()

    def trim_prefix(self, sample_count: int) -> None:
        """Remove the first ``sample_count`` samples from the buffer."""
        sample_count = max(0, min(sample_count, len(self)))
        if sample_count <= 0:
            return
        self._samples = self._samples[sample_count:]

    def ms_to_samples(self, ms: int) -> int:
        """Convert milliseconds to sample count at this buffer's sample rate."""
        return int(self.sample_rate * ms / 1000)

    def samples_to_ms(self, samples: int) -> int:
        """Convert sample count to milliseconds at this buffer's sample rate."""
        return int(samples * 1000 / self.sample_rate)
