from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_LANGUAGE,
    REALTIME_BEAM_SIZE,
    REALTIME_INITIAL_PROMPT_CHARS,
    REALTIME_MAX_BUFFER_SEC,
    REALTIME_MIN_SEGMENT_MS,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION_MS,
    REALTIME_SPEAKER_MAX_SPEAKERS,
    REALTIME_SPEAKER_MIN_SEGMENT_SEC,
    REALTIME_SPEAKER_SIMILARITY_THRESHOLD,
    REALTIME_VAD_THRESHOLD,
)
from app.errors import OpenAIAPIError
from app.openai_format import speaker_labels
from app.openai_realtime_events import (
    buffer_committed_event,
    default_session_config,
    error_event,
    session_diarization_completed_event,
    session_updated_event,
    speaker_assigned_event,
    speaker_updated_event,
    speech_started_event,
    speech_stopped_event,
    transcription_completed_event,
    transcription_delta_event,
)
from app.services.asr import ASRService
from app.services.audio_buffer import AudioBuffer, chunk_rms
from app.services.diarization import DiarizationService, speaker_for_segment
from app.services.realtime_speaker_config import resolve_speaker_embedding_model
from app.services.realtime_speaker_tracker import RealtimeSpeakerTracker
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)

SendEvent = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class TranscriptItem:
    item_id: str
    start_sec: float
    end_sec: float
    text: str
    speaker: str | None = None
    speaker_provisional: bool = False


@dataclass
class SpeakerDiarizationConfig:
    enabled: bool = False
    finalize: bool = False
    mode: str = "provisional"
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    max_speakers: int = REALTIME_SPEAKER_MAX_SPEAKERS
    num_speakers: int | None = None
    min_speakers: int | None = None
    known_speaker_names: list[str] = field(default_factory=list)
    use_exclusive: bool = False


class RealtimeSession:
    def __init__(
        self,
        *,
        asr_service: ASRService,
        diarization_service: DiarizationService | None,
        model_id: str,
        send_event: SendEvent,
        language: str = DEFAULT_LANGUAGE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        sample_rate: int = REALTIME_SAMPLE_RATE,
        max_buffer_sec: float = REALTIME_MAX_BUFFER_SEC,
        vad_threshold: float = REALTIME_VAD_THRESHOLD,
        silence_duration_ms: int = REALTIME_SILENCE_DURATION_MS,
        min_segment_ms: int = REALTIME_MIN_SEGMENT_MS,
        beam_size: int = REALTIME_BEAM_SIZE,
        diarization_capable: bool = False,
        default_diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    ) -> None:
        self.asr_service = asr_service
        self.diarization_service = diarization_service
        self.model_id = model_id
        self.send_event = send_event
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_segment_ms = min_segment_ms
        self.beam_size = beam_size
        self.diarization_capable = diarization_capable
        self.default_diarization_model = default_diarization_model
        self.input_audio_format = "pcm16"
        self.timestamp_granularities: list[str] = []
        self.speaker_config = SpeakerDiarizationConfig(
            diarization_model=default_diarization_model,
        )

        self.session_config = default_session_config(
            model_id=model_id,
            language=language,
            sample_rate=sample_rate,
            input_audio_format=self.input_audio_format,
            vad_threshold=vad_threshold,
            silence_duration_ms=silence_duration_ms,
            timestamp_granularities=self.timestamp_granularities,
            speaker_diarization=self._speaker_config_snapshot(),
        )
        self.buffer = AudioBuffer(sample_rate=sample_rate, max_duration_sec=max_buffer_sec)
        self._session_audio = AudioBuffer(sample_rate=sample_rate, max_duration_sec=max_buffer_sec)

        self._inference_lock = asyncio.Lock()
        self._speaker_lock = asyncio.Lock()
        self._closed = False
        self._is_speaking = False
        self._speech_start_sample = 0
        self._silence_ms = 0
        self._current_item_id: str | None = None
        self._transcript_tail = ""
        self._last_activity_at = time.monotonic()
        self._absolute_sample_offset = 0
        self._transcript_items: list[TranscriptItem] = []
        self._speaker_tracker: RealtimeSpeakerTracker | None = None
        self._finalized = False

    @property
    def closed(self) -> bool:
        return self._closed

    def touch(self) -> None:
        self._last_activity_at = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity_at

    def session_snapshot(self) -> dict[str, Any]:
        return dict(self.session_config)

    def _speaker_config_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.speaker_config.enabled,
            "mode": self.speaker_config.mode,
            "finalize": self.speaker_config.finalize,
            "diarization_model": self.speaker_config.diarization_model,
            "max_speakers": self.speaker_config.max_speakers,
            "num_speakers": self.speaker_config.num_speakers,
            "min_speakers": self.speaker_config.min_speakers,
            "known_speaker_names": list(self.speaker_config.known_speaker_names),
            "use_exclusive": self.speaker_config.use_exclusive,
        }

    def _ensure_speaker_tracker(self) -> RealtimeSpeakerTracker | None:
        if not self.speaker_config.enabled or self.speaker_config.mode != "provisional":
            return None
        if self._speaker_tracker is not None:
            return self._speaker_tracker

        embedding_model = resolve_speaker_embedding_model(self.speaker_config.diarization_model)
        if not embedding_model:
            logger.warning(
                "Realtime speaker diarization enabled but no embedding model resolved for %s",
                self.speaker_config.diarization_model,
            )
            return None

        self._speaker_tracker = RealtimeSpeakerTracker(
            embedding_model_path=embedding_model,
            similarity_threshold=REALTIME_SPEAKER_SIMILARITY_THRESHOLD,
            max_speakers=self.speaker_config.max_speakers,
            known_speaker_names=self.speaker_config.known_speaker_names,
            min_segment_sec=REALTIME_SPEAKER_MIN_SEGMENT_SEC,
        )
        return self._speaker_tracker

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    async def finalize_session(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self.speaker_config.enabled or not self.speaker_config.finalize:
            return
        if self.diarization_service is None:
            return
        if len(self._session_audio) == 0 or not self._transcript_items:
            return

        audio = self._session_audio.extract_all()
        items_snapshot = list(self._transcript_items)

        try:
            final_items = await asyncio.to_thread(
                self._finalize_diarization,
                audio,
                items_snapshot,
            )
        except OpenAIAPIError as exc:
            await self.send_event(
                error_event(
                    exc.message,
                    error_type=exc.error_type,
                    code=exc.code,
                    param=exc.param,
                )
            )
            return
        except Exception as exc:
            logger.exception("Realtime session diarization finalize failed model=%s", self.model_id)
            await self.send_event(
                error_event(
                    f"Session diarization failed: {exc}",
                    error_type="server_error",
                    code="diarization_failed",
                )
            )
            return

        for item, final_speaker in final_items:
            if item.speaker and item.speaker != final_speaker:
                await self.send_event(
                    speaker_updated_event(
                        item.item_id,
                        final_speaker,
                        previous_speaker=item.speaker,
                        final=True,
                    )
                )
            item.speaker = final_speaker
            item.speaker_provisional = False

        await self.send_event(
            session_diarization_completed_event(
                items=[
                    {
                        "item_id": item.item_id,
                        "start": item.start_sec,
                        "end": item.end_sec,
                        "text": item.text,
                        "speaker": item.speaker,
                    }
                    for item in items_snapshot
                ],
                duration_sec=len(audio) / self.sample_rate,
            )
        )

    def _finalize_diarization(
        self,
        audio: np.ndarray,
        items: list[TranscriptItem],
    ) -> list[tuple[TranscriptItem, str]]:
        assert self.diarization_service is not None

        temp_path = self._write_temp_wav(audio)
        try:
            diarization_turns = self.diarization_service.diarize(
                temp_path,
                diarization_model=self.speaker_config.diarization_model,
                min_speakers=self.speaker_config.min_speakers,
                max_speakers=self.speaker_config.max_speakers,
                num_speakers=self.speaker_config.num_speakers,
                use_exclusive=self.speaker_config.use_exclusive,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        internal_labels = speaker_labels(
            [{"speaker": turn["speaker"]} for turn in diarization_turns],
            self.speaker_config.known_speaker_names,
        )
        results: list[tuple[TranscriptItem, str]] = []

        for item in items:
            internal_speaker = speaker_for_segment(
                segment_start=item.start_sec,
                segment_end=item.end_sec,
                diarization_turns=diarization_turns,
            )
            final_speaker = internal_labels.get(internal_speaker, internal_speaker)
            results.append((item, final_speaker))

        return results

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return temp_path

    async def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.touch()

        if event_type == "session.update":
            await self._handle_session_update(payload)
            return
        if event_type == "input_audio_buffer.append":
            await self._handle_append(payload)
            return
        if event_type == "input_audio_buffer.commit":
            await self._handle_commit(force=True)
            return
        if event_type == "input_audio_buffer.clear":
            await self._handle_clear()
            return

        await self.send_event(
            error_event(
                f"Unsupported client event type: {event_type}",
                code="unsupported_event",
            )
        )

    async def _handle_session_update(self, payload: dict[str, Any]) -> None:
        session = payload.get("session")
        if not isinstance(session, dict):
            await self.send_event(
                error_event("session.update requires a session object", param="session")
            )
            return

        if "model" in session and session["model"]:
            self.model_id = str(session["model"])
            self.session_config["model"] = self.model_id

        transcription = session.get("input_audio_transcription")
        if isinstance(transcription, dict):
            if transcription.get("model"):
                self.model_id = str(transcription["model"])
                self.session_config["model"] = self.model_id
            if transcription.get("language"):
                self.language = str(transcription["language"])
            granularities = transcription.get("timestamp_granularities")
            if isinstance(granularities, list):
                self.timestamp_granularities = [
                    str(value) for value in granularities if isinstance(value, str)
                ]

        speaker_diarization = session.get("speaker_diarization")
        if isinstance(speaker_diarization, dict):
            if speaker_diarization.get("enabled") is not None:
                self.speaker_config.enabled = bool(speaker_diarization["enabled"])
            if speaker_diarization.get("finalize") is not None:
                self.speaker_config.finalize = bool(speaker_diarization["finalize"])
            if speaker_diarization.get("mode"):
                self.speaker_config.mode = str(speaker_diarization["mode"])
            if speaker_diarization.get("diarization_model"):
                self.speaker_config.diarization_model = str(
                    speaker_diarization["diarization_model"]
                )
            if speaker_diarization.get("max_speakers") is not None:
                self.speaker_config.max_speakers = int(speaker_diarization["max_speakers"])
            if speaker_diarization.get("num_speakers") is not None:
                self.speaker_config.num_speakers = int(speaker_diarization["num_speakers"])
            if speaker_diarization.get("min_speakers") is not None:
                self.speaker_config.min_speakers = int(speaker_diarization["min_speakers"])
            if speaker_diarization.get("use_exclusive") is not None:
                self.speaker_config.use_exclusive = bool(speaker_diarization["use_exclusive"])
            known_names = speaker_diarization.get("known_speaker_names")
            if isinstance(known_names, list):
                self.speaker_config.known_speaker_names = [
                    str(name) for name in known_names if isinstance(name, str)
                ]

            if self.speaker_config.enabled and not self.diarization_capable:
                has_explicit_model = bool(speaker_diarization.get("diarization_model"))
                if not has_explicit_model:
                    await self.send_event(
                        error_event(
                            (
                                "speaker_diarization requires a model with diarization capability "
                                "or an explicit diarization_model path."
                            ),
                            param="speaker_diarization",
                            code="unsupported_capability",
                        )
                    )
                    self.speaker_config.enabled = False
                    self.speaker_config.finalize = False

        if session.get("input_audio_format"):
            audio_format = str(session["input_audio_format"])
            if audio_format != "pcm16":
                await self.send_event(
                    error_event(
                        (
                            f"Unsupported input_audio_format '{audio_format}'. "
                            "This server expects pcm16 mono at 16 kHz."
                        ),
                        param="input_audio_format",
                        code="unsupported_audio_format",
                    )
                )
                return
            self.input_audio_format = audio_format
            self.session_config["input_audio_format"] = audio_format

        turn_detection = session.get("turn_detection")
        if isinstance(turn_detection, dict):
            if turn_detection.get("threshold") is not None:
                self.vad_threshold = float(turn_detection["threshold"])
            if turn_detection.get("silence_duration_ms") is not None:
                self.silence_duration_ms = int(turn_detection["silence_duration_ms"])
            self.session_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.silence_duration_ms,
                "sample_rate": self.sample_rate,
            }

        transcription_config: dict[str, Any] = {
            "model": self.model_id,
            "language": self.language,
        }
        if self.timestamp_granularities:
            transcription_config["timestamp_granularities"] = self.timestamp_granularities
        self.session_config["input_audio_transcription"] = transcription_config
        self.session_config["speaker_diarization"] = self._speaker_config_snapshot()
        await self.send_event(session_updated_event(self.session_snapshot()))

    async def _handle_append(self, payload: dict[str, Any]) -> None:
        audio_b64 = payload.get("audio")
        if not isinstance(audio_b64, str) or not audio_b64:
            await self.send_event(
                error_event("input_audio_buffer.append requires base64 audio", param="audio")
            )
            return

        try:
            chunk = self.buffer.append_pcm16_base64(audio_b64)
            self._session_audio.append_float32(chunk)
        except ValueError as exc:
            await self.send_event(error_event(str(exc), param="audio", code="invalid_audio"))
            return

        if chunk.size == 0:
            return

        chunk_ms = self.buffer.samples_to_ms(chunk.size)
        rms = chunk_rms(chunk)

        if rms >= self.vad_threshold:
            self._silence_ms = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_sample = len(self.buffer) - chunk.size
                self._current_item_id = f"item_{uuid.uuid4().hex[:16]}"
                await self.send_event(
                    speech_started_event(
                        self.buffer.samples_to_ms(self._speech_start_sample),
                        self._current_item_id,
                    )
                )
        elif self._is_speaking:
            self._silence_ms += chunk_ms
            if self._silence_ms >= self.silence_duration_ms:
                await self._finalize_speech(auto_commit=True)

    async def _handle_clear(self) -> None:
        self.buffer.clear()
        self._session_audio.clear()
        self._absolute_sample_offset = 0
        self._transcript_items.clear()
        self._reset_speech_state()
        record_tool_call("realtime.buffer.clear", model=self.model_id)

    async def _handle_commit(self, *, force: bool) -> None:
        if self._is_speaking:
            await self._finalize_speech(auto_commit=True, force=force)
            return

        if len(self.buffer) == 0:
            await self.send_event(error_event("Audio buffer is empty", code="empty_audio"))
            return

        item_id = self._current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        self._current_item_id = item_id
        start = 0
        end = len(self.buffer)
        await self._transcribe_segment(start, end, item_id, force=force)

    async def _finalize_speech(self, *, auto_commit: bool, force: bool = False) -> None:
        if not self._is_speaking:
            return

        item_id = self._current_item_id or f"item_{uuid.uuid4().hex[:16]}"
        end = len(self.buffer)
        start = self._speech_start_sample
        audio_end_ms = self.buffer.samples_to_ms(end)

        await self.send_event(speech_stopped_event(audio_end_ms, item_id))
        self._reset_speech_state()

        if auto_commit:
            await self._transcribe_segment(start, end, item_id, force=force)

    def _reset_speech_state(self) -> None:
        self._is_speaking = False
        self._silence_ms = 0
        self._speech_start_sample = 0

    def _absolute_range(self, start: int, end: int) -> tuple[float, float]:
        absolute_start = (self._absolute_sample_offset + start) / self.sample_rate
        absolute_end = (self._absolute_sample_offset + end) / self.sample_rate
        return absolute_start, absolute_end

    def _offset_words(self, words: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
        return [
            {
                "word": word["word"],
                "start": round(word["start"] + offset_sec, 3),
                "end": round(word["end"] + offset_sec, 3),
            }
            for word in words
        ]

    async def _transcribe_segment(
        self,
        start: int,
        end: int,
        item_id: str,
        *,
        force: bool,
    ) -> None:
        segment = self.buffer.extract_range(start, end)
        duration_ms = int(segment.size * 1000 / self.sample_rate)
        absolute_start, absolute_end = self._absolute_range(start, end)

        if segment.size == 0:
            if force:
                await self.send_event(error_event("Audio buffer is empty", code="empty_audio"))
            return

        if duration_ms < self.min_segment_ms:
            logger.debug(
                "Skipping short realtime segment model=%s duration_ms=%d",
                self.model_id,
                duration_ms,
            )
            self.buffer.trim_prefix(end)
            self._absolute_sample_offset += end
            return

        await self.send_event(buffer_committed_event(item_id))

        prompt = self._transcript_tail[-REALTIME_INITIAL_PROMPT_CHARS :] if self._transcript_tail else None
        timestamp_granularities = self.timestamp_granularities or None

        async with self._inference_lock:
            try:
                record_tool_call(
                    "realtime.transcribe",
                    model=self.model_id,
                    duration_ms=duration_ms,
                    language=self.language,
                )
                result = await asyncio.to_thread(
                    self.asr_service.transcribe_array,
                    segment,
                    sample_rate=self.sample_rate,
                    model_id=self.model_id,
                    language=self.language,
                    prompt=prompt,
                    temperature=0.0,
                    device=self.device,
                    compute_type=self.compute_type,
                    beam_size=self.beam_size,
                    vad_filter=False,
                    timestamp_granularities=timestamp_granularities,
                )
            except OpenAIAPIError as exc:
                await self.send_event(
                    error_event(
                        exc.message,
                        error_type=exc.error_type,
                        code=exc.code,
                        param=exc.param,
                    )
                )
                return
            except Exception as exc:
                logger.exception("Realtime transcription failed model=%s", self.model_id)
                await self.send_event(
                    error_event(
                        f"Transcription failed: {exc}",
                        error_type="server_error",
                        code="transcription_failed",
                    )
                )
                return

        text = str(result.get("text", "")).strip()
        self.buffer.trim_prefix(end)
        self._absolute_sample_offset += end

        if not text:
            return

        words = self._offset_words(result.get("words", []), absolute_start)
        transcript_item = TranscriptItem(
            item_id=item_id,
            start_sec=absolute_start,
            end_sec=absolute_end,
            text=text,
        )
        self._transcript_items.append(transcript_item)

        await self.send_event(transcription_delta_event(item_id, text))
        await self.send_event(
            transcription_completed_event(
                item_id,
                text,
                words=words or None,
            )
        )

        if self._transcript_tail:
            self._transcript_tail = f"{self._transcript_tail} {text}".strip()
        else:
            self._transcript_tail = text

        if len(self._transcript_tail) > REALTIME_INITIAL_PROMPT_CHARS * 4:
            self._transcript_tail = self._transcript_tail[-REALTIME_INITIAL_PROMPT_CHARS * 4 :]

        await self._assign_provisional_speaker(item_id, segment, transcript_item)

    async def _assign_provisional_speaker(
        self,
        item_id: str,
        segment: np.ndarray,
        transcript_item: TranscriptItem,
    ) -> None:
        tracker = self._ensure_speaker_tracker()
        if tracker is None:
            return

        async with self._speaker_lock:
            try:
                assignment = await asyncio.to_thread(
                    tracker.assign_speaker,
                    segment,
                    sample_rate=self.sample_rate,
                )
            except OpenAIAPIError as exc:
                await self.send_event(
                    error_event(
                        exc.message,
                        error_type=exc.error_type,
                        code=exc.code,
                        param=exc.param,
                    )
                )
                return

        if assignment is None:
            return

        speaker, confidence = assignment
        transcript_item.speaker = speaker
        transcript_item.speaker_provisional = True

        await self.send_event(
            speaker_assigned_event(
                item_id,
                speaker,
                confidence=confidence,
                provisional=True,
            )
        )
