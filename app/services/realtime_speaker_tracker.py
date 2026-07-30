from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.errors import OpenAIAPIError

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER_SIMILARITY_THRESHOLD = 0.75


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def speaker_label_for_index(index: int, known_speaker_names: list[str]) -> str:
    if index < len(known_speaker_names):
        return known_speaker_names[index]
    return chr(ord("A") + index)


@dataclass
class SpeakerCentroid:
    label: str
    embedding: np.ndarray
    count: int = 1

    def update(self, embedding: np.ndarray) -> None:
        total = self.count + 1
        self.embedding = ((self.embedding * self.count) + embedding) / total
        self.count = total


@dataclass
class OnlineSpeakerCluster:
    similarity_threshold: float = DEFAULT_SPEAKER_SIMILARITY_THRESHOLD
    max_speakers: int = 8
    known_speaker_names: list[str] = field(default_factory=list)
    centroids: list[SpeakerCentroid] = field(default_factory=list)

    def assign(self, embedding: np.ndarray) -> tuple[str, float]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return "UNKNOWN", 0.0

        if not self.centroids:
            label = speaker_label_for_index(0, self.known_speaker_names)
            self.centroids.append(SpeakerCentroid(label=label, embedding=vector.copy()))
            return label, 1.0

        scores = [cosine_similarity(vector, centroid.embedding) for centroid in self.centroids]
        best_index = int(np.argmax(scores))
        best_score = scores[best_index]

        if best_score >= self.similarity_threshold:
            self.centroids[best_index].update(vector)
            return self.centroids[best_index].label, best_score

        if len(self.centroids) >= self.max_speakers:
            self.centroids[best_index].update(vector)
            return self.centroids[best_index].label, best_score

        label = speaker_label_for_index(len(self.centroids), self.known_speaker_names)
        self.centroids.append(SpeakerCentroid(label=label, embedding=vector.copy()))
        return label, best_score


class RealtimeSpeakerTracker:
    """Provisional speaker labels via lightweight embedding + online clustering."""

    def __init__(
        self,
        *,
        embedding_model_path: str,
        similarity_threshold: float = DEFAULT_SPEAKER_SIMILARITY_THRESHOLD,
        max_speakers: int = 8,
        known_speaker_names: list[str] | None = None,
        min_segment_sec: float = 0.5,
    ) -> None:
        self.embedding_model_path = embedding_model_path
        self.min_segment_sec = min_segment_sec
        self._cluster = OnlineSpeakerCluster(
            similarity_threshold=similarity_threshold,
            max_speakers=max_speakers,
            known_speaker_names=list(known_speaker_names or []),
        )
        self._lock = threading.Lock()
        self._inference: Any | None = None

    def _get_inference(self) -> Any:
        if self._inference is not None:
            return self._inference

        with self._lock:
            if self._inference is not None:
                return self._inference

            try:
                import torch
                from pyannote.audio import Inference, Model
            except ImportError as exc:
                raise OpenAIAPIError(
                    "Speaker diarization requires pyannote.audio, which is not installed.",
                    status_code=500,
                    error_type="server_error",
                    code="speaker_diarization_unavailable",
                ) from exc

            from app.config import resolve_device

            device = torch.device(resolve_device())
            logger.info("Loading realtime speaker embedding model path=%s", self.embedding_model_path)
            model = Model.from_pretrained(self.embedding_model_path)
            self._inference = Inference(model, window="whole", device=device)
            return self._inference

    def assign_speaker(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[str, float] | None:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None

        duration_sec = samples.size / sample_rate
        if duration_sec < self.min_segment_sec:
            return None

        try:
            import torch

            inference = self._get_inference()
            waveform = torch.from_numpy(samples).unsqueeze(0)
            embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
            embedding_np = np.asarray(embedding).reshape(-1)
        except OpenAIAPIError:
            raise
        except Exception as exc:
            logger.warning("Realtime speaker embedding failed: %s", exc)
            return None

        label, confidence = self._cluster.assign(embedding_np)
        return label, confidence
