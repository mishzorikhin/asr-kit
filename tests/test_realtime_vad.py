from __future__ import annotations

import numpy as np

from app.services.audio_buffer import AudioBuffer
from app.services.realtime_vad import VadController


def test_vad_detects_speech_start_and_silence_stop() -> None:
    buffer = AudioBuffer(sample_rate=16000, max_duration_sec=5.0)
    vad = VadController(
        buffer=buffer,
        vad_threshold=0.1,
        silence_duration_ms=100,
    )

    loud = np.ones(1600, dtype=np.float32)  # 100 ms
    buffer.append_float32(loud)
    started, stopped = vad.on_chunk(loud)
    assert started is not None
    assert stopped is None
    assert vad.is_speaking

    quiet = np.zeros(1600, dtype=np.float32)
    buffer.append_float32(quiet)
    started, stopped = vad.on_chunk(quiet)
    assert started is None
    assert stopped is not None
    assert not vad.is_speaking
    assert stopped.end_sample == len(buffer)
