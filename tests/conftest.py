"""Shared fixtures. Mock heavy optional deps so unit tests import cleanly."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _ensure_fake_module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    sys.modules[name] = module
    return module


# pyannote.audio pulls torch/lightning stacks; unit tests mock the surface API.
if "pyannote" not in sys.modules:
    pyannote = _ensure_fake_module("pyannote")
    pyannote_audio = _ensure_fake_module("pyannote.audio")
    pyannote.audio = pyannote_audio  # type: ignore[attr-defined]
    pyannote_audio.Pipeline = MagicMock(name="Pipeline")  # type: ignore[attr-defined]
    pyannote_audio.Inference = MagicMock(name="Inference")  # type: ignore[attr-defined]
    pyannote_audio.Model = MagicMock(name="Model")  # type: ignore[attr-defined]


# faster-whisper is heavy; unit tests mock WhisperModel at the module boundary.
if "faster_whisper" not in sys.modules:
    faster_whisper = _ensure_fake_module("faster_whisper")
    faster_whisper.WhisperModel = MagicMock(name="WhisperModel")  # type: ignore[attr-defined]
