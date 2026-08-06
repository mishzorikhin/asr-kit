from __future__ import annotations

import base64

import numpy as np
import pytest

from app.services.audio_buffer import AudioBuffer, chunk_rms, pcm16_base64_to_float32


def test_chunk_rms() -> None:
    silence = np.zeros(100, dtype=np.float32)
    signal = np.ones(100, dtype=np.float32)
    assert chunk_rms(silence) == 0.0
    assert chunk_rms(signal) == pytest.approx(1.0)
    assert chunk_rms(np.array([], dtype=np.float32)) == 0.0


def test_pcm16_base64_roundtrip() -> None:
    pcm = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    encoded = base64.b64encode(pcm).decode("ascii")
    samples = pcm16_base64_to_float32(encoded)
    assert samples.dtype == np.float32
    assert samples.shape == (3,)
    assert samples[0] == 0.0


def test_pcm16_base64_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid base64"):
        pcm16_base64_to_float32("%%%not-base64%%%")
    with pytest.raises(ValueError, match="even"):
        pcm16_base64_to_float32(base64.b64encode(b"abc").decode("ascii"))


def test_audio_buffer_append_and_duration() -> None:
    buffer = AudioBuffer(sample_rate=16000, max_duration_sec=2.0)
    pcm = np.zeros(1600, dtype=np.int16).tobytes()
    encoded = base64.b64encode(pcm).decode("ascii")

    chunk, dropped = buffer.append_pcm16_base64(encoded)
    assert chunk.size == 1600
    assert dropped == 0
    assert len(buffer) == 1600
    assert buffer.duration_sec == pytest.approx(0.1)


def test_audio_buffer_trim_and_clear() -> None:
    buffer = AudioBuffer(sample_rate=16000, max_duration_sec=2.0)
    buffer.append_float32(np.ones(3200, dtype=np.float32))
    buffer.trim_prefix(1600)
    assert len(buffer) == 1600
    buffer.clear()
    assert len(buffer) == 0


def test_audio_buffer_truncation_reports_dropped_samples() -> None:
    buffer = AudioBuffer(sample_rate=16000, max_duration_sec=1.0)
    dropped = buffer.append_float32(np.ones(24000, dtype=np.float32))
    assert dropped == 8000
    assert len(buffer) == 16000


def test_audio_buffer_unbounded_when_max_duration_none() -> None:
    buffer = AudioBuffer(sample_rate=16000, max_duration_sec=None)
    dropped = buffer.append_float32(np.ones(48000, dtype=np.float32))
    assert dropped == 0
    assert len(buffer) == 48000
