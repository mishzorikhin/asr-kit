import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse

from app.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_LANGUAGE,
    DIARIZED_RESPONSE_FORMATS,
    OPENAI_RESPONSE_FORMATS,
    TRANSCRIPTION_RESPONSE_FORMATS,
)
from app.errors import OpenAIAPIError
from app.openai_format import format_openai_response
from app.services.asr import ASRService
from app.services.diarization import DiarizationService
from app.tool_calls import current_request_id, new_request_id, record_tool_call
from app.upload import save_upload_to_temp_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/audio", tags=["audio"])

TRANSCRIPTION_DESCRIPTION = """
Creates an audio transcription from a multipart file upload.

OpenAI-compatible request fields:
- `file`: audio file. Supported extensions: `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`.
- `model`: model id from `GET /v1/models`.
- `language`: ISO-639-1 input language, for example `ru` or `en`.
- `prompt`: optional initial prompt passed to faster-whisper.
- `response_format`: `json`, `text`, `srt`, `verbose_json`, `vtt`, or local extension `diarized_json`.
- `temperature`: sampling temperature.
- `timestamp_granularities`: repeat the form field or use `timestamp_granularities[]`; supported values are `segment` and `word`.

Local compatibility limits:
- `stream=true` is rejected with `unsupported_parameter`; streaming is not implemented.
- `include=logprobs` is rejected with `unsupported_parameter`; faster-whisper does not provide OpenAI-style logprobs.
- `/v1/audio/translations` is not implemented by this server.
- Models with diarization capability currently support only `json`, `text`, and `diarized_json`.

Diarization extension:
- Use a diarization-capable model, for example `whisper-ru-turbo-diarize`.
- Set `response_format=diarized_json` to return speaker-labeled segments.
- `known_speaker_names` maps discovered speakers to stable labels in discovery order.
- `known_speaker_references` is accepted for request compatibility but local speaker matching against reference audio is not implemented.
- `min_speakers`, `max_speakers`, `num_speakers`, `diarization_model`, and `use_exclusive` are local pyannote controls.

OpenAI Python SDK example:

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://10.0.0.104:8000/v1")

with open("speech.mp3", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="whisper-ru-turbo",
        file=audio,
        response_format="verbose_json",
        language="ru",
        timestamp_granularities=["segment", "word"],
    )

with open("speech.mp3", "rb") as audio:
    diarized = client.audio.transcriptions.create(
        model="whisper-ru-turbo-diarize",
        file=audio,
        response_format="diarized_json",
        language="ru",
        extra_body={
            "known_speaker_names": ["agent", "customer"],
            "min_speakers": 2,
            "max_speakers": 6,
        },
    )
```
"""

TRANSCRIPTION_RESPONSES = {
    200: {
        "description": (
            "Transcription result. `json`, `verbose_json`, and `diarized_json` return JSON. "
            "`text`, `srt`, and `vtt` return plain text."
        ),
        "content": {
            "application/json": {
                "examples": {
                    "json": {
                        "summary": "response_format=json",
                        "value": {
                            "text": "Репка посадил дед репку.",
                            "usage": {"type": "duration", "seconds": 114},
                        },
                    },
                    "verbose_json": {
                        "summary": "response_format=verbose_json",
                        "value": {
                            "task": "transcribe",
                            "language": "ru",
                            "duration": 114.0,
                            "text": "Репка посадил дед репку.",
                            "segments": [
                                {
                                    "id": 0,
                                    "seek": 0,
                                    "start": 0.0,
                                    "end": 4.31,
                                    "text": "Репка посадил дед репку.",
                                    "tokens": [50369, 6325],
                                    "temperature": 0.0,
                                    "avg_logprob": -0.22,
                                    "compression_ratio": 2.09,
                                    "no_speech_prob": 0.00001,
                                }
                            ],
                            "words": [
                                {"word": " Репка", "start": 0.0, "end": 0.56}
                            ],
                            "usage": {"type": "duration", "seconds": 114},
                        },
                    },
                    "diarized_json": {
                        "summary": "response_format=diarized_json",
                        "value": {
                            "task": "transcribe",
                            "duration": 114.0,
                            "text": "agent: Добрый день.\ncustomer: Здравствуйте.",
                            "segments": [
                                {
                                    "type": "transcript.text.segment",
                                    "id": "seg_001",
                                    "start": 0.08,
                                    "end": 2.4,
                                    "text": "Добрый день.",
                                    "speaker": "agent",
                                }
                            ],
                            "usage": {"type": "duration", "seconds": 114},
                        },
                    },
                }
            },
            "text/plain": {
                "examples": {
                    "text": {
                        "summary": "response_format=text",
                        "value": "Репка посадил дед репку.",
                    },
                    "srt": {
                        "summary": "response_format=srt",
                        "value": "1\n00:00:00,080 --> 00:00:04,250\nРепка посадил дед репку.\n",
                    },
                }
            },
            "text/vtt": {
                "example": "WEBVTT\n\n00:00:00.080 --> 00:00:04.250\nРепка посадил дед репку.\n"
            },
        },
    },
    400: {
        "description": "OpenAI-style invalid request error.",
        "content": {
            "application/json": {
                "examples": {
                    "unsupported_stream": {
                        "summary": "stream=true is not implemented",
                        "value": {
                            "error": {
                                "message": "Streaming transcription responses are not supported by this local server.",
                                "type": "invalid_request_error",
                                "param": "stream",
                                "code": "unsupported_parameter",
                            }
                        },
                    },
                    "empty_file": {
                        "summary": "Uploaded file is empty",
                        "value": {
                            "error": {
                                "message": "Uploaded audio file is empty",
                                "type": "invalid_request_error",
                                "param": "file",
                                "code": "empty_file",
                            }
                        },
                    },
                }
            }
        },
    },
    507: {
        "description": "Server error when GPU memory is insufficient.",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "message": "Not enough GPU memory to process this request.",
                        "type": "server_error",
                        "param": None,
                        "code": "insufficient_gpu_memory",
                    }
                }
            }
        },
    },
}

TRANSCRIPTION_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "examples": {
                    "json": {
                        "summary": "Plain transcription",
                        "value": {
                            "file": "(binary audio file)",
                            "model": "whisper-ru-turbo",
                            "language": "ru",
                            "response_format": "json",
                        },
                    },
                    "verbose_json_words": {
                        "summary": "Verbose JSON with word timestamps",
                        "value": {
                            "file": "(binary audio file)",
                            "model": "whisper-ru-turbo",
                            "language": "ru",
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": ["segment", "word"],
                        },
                    },
                    "diarized_json": {
                        "summary": "Diarized transcription",
                        "value": {
                            "file": "(binary audio file)",
                            "model": "whisper-ru-turbo-diarize",
                            "language": "ru",
                            "response_format": "diarized_json",
                            "known_speaker_names[]": ["agent", "customer"],
                            "min_speakers": 2,
                            "max_speakers": 6,
                        },
                    },
                }
            }
        }
    }
}


def form_list(form: Any, name: str) -> list[str]:
    values = []
    for key in (name, f"{name}[]"):
        values.extend(form.getlist(key))
    return [str(value) for value in values if str(value)]


def parse_bool(value: str | bool | None, default: bool, param: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise OpenAIAPIError(
        f"Invalid boolean value: {value}",
        param=param,
        code="invalid_boolean",
    )


def validate_transcription_request(
    *,
    configured_model: dict[str, Any],
    model: str,
    response_format: str,
    stream: bool,
    include: list[str],
    timestamp_granularities: list[str],
) -> None:
    if response_format not in OPENAI_RESPONSE_FORMATS:
        raise OpenAIAPIError(
            (
                f"Unsupported response_format '{response_format}'. Expected one of: "
                + ", ".join(sorted(OPENAI_RESPONSE_FORMATS))
            ),
            param="response_format",
            code="unsupported_response_format",
        )
    if "diarization" in configured_model["capabilities"]:
        if response_format not in DIARIZED_RESPONSE_FORMATS:
            raise OpenAIAPIError(
                f"{model} supports only json, text, and diarized_json response formats.",
                param="response_format",
                code="unsupported_response_format",
            )
    elif response_format == "diarized_json":
        raise OpenAIAPIError(
            "diarized_json requires a model with diarization capability.",
            param="response_format",
            code="unsupported_response_format",
        )
    elif response_format not in TRANSCRIPTION_RESPONSE_FORMATS:
        raise OpenAIAPIError(
            (
                f"Model {model} supports only: "
                + ", ".join(sorted(TRANSCRIPTION_RESPONSE_FORMATS))
            ),
            param="response_format",
            code="unsupported_response_format",
        )
    if stream:
        raise OpenAIAPIError(
            "Streaming transcription responses are not supported by this local server.",
            param="stream",
            code="unsupported_parameter",
        )
    unsupported_includes = sorted(set(include) - {"logprobs"})
    if unsupported_includes:
        raise OpenAIAPIError(
            f"Unsupported include values: {', '.join(unsupported_includes)}",
            param="include",
            code="unsupported_parameter",
        )
    if "logprobs" in include and response_format != "json":
        raise OpenAIAPIError(
            "include=logprobs requires response_format=json.",
            param="include",
            code="unsupported_parameter",
        )
    if "logprobs" in include:
        raise OpenAIAPIError(
            "include=logprobs is not supported by this local faster-whisper server.",
            param="include",
            code="unsupported_parameter",
        )
    unsupported_granularities = sorted(set(timestamp_granularities) - {"segment", "word"})
    if unsupported_granularities:
        raise OpenAIAPIError(
            f"Unsupported timestamp granularities: {', '.join(unsupported_granularities)}",
            param="timestamp_granularities",
            code="unsupported_parameter",
        )


@router.post(
    "/transcriptions",
    response_model=None,
    summary="Create transcription",
    description=TRANSCRIPTION_DESCRIPTION,
    responses=TRANSCRIPTION_RESPONSES,
    openapi_extra=TRANSCRIPTION_OPENAPI_EXTRA,
)
async def create_transcription(
    request: Request,
    file: UploadFile = File(
        ...,
        description=(
            "Audio file object. Supported extensions: flac, mp3, mp4, mpeg, "
            "mpga, m4a, ogg, wav, webm."
        ),
    ),
    model: str = Form(
        ...,
        description=(
            "Configured model id from GET /v1/models. Use a diarization-capable "
            "model only when response_format=diarized_json is needed."
        ),
        examples=["whisper-ru-turbo"],
    ),
    language: str | None = Form(
        DEFAULT_LANGUAGE,
        description="Input language in ISO-639-1 format, for example ru or en.",
        examples=["ru"],
    ),
    prompt: str | None = Form(None, description="Optional initial prompt passed to faster-whisper."),
    response_format: str = Form(
        "json",
        description=(
            "json, text, srt, verbose_json, vtt, or local extension diarized_json. "
            "Diarization-capable models currently support only json, text, and diarized_json."
        ),
        examples=["json"],
    ),
    temperature: float = Form(0.0, description="Sampling temperature passed to faster-whisper."),
    stream: str | bool | None = Form(
        False,
        description=(
            "Accepted for OpenAI compatibility, but streaming is not implemented. "
            "Any true value returns unsupported_parameter."
        ),
    ),
    include: list[str] | None = Form(
        None,
        description=(
            "Repeatable OpenAI field. include=logprobs is recognized for validation "
            "but not supported by faster-whisper."
        ),
    ),
    timestamp_granularities: list[str] | None = Form(
        None,
        description=(
            "Repeatable OpenAI field. Supported values: segment, word. Use "
            "timestamp_granularities[]=word with curl if needed."
        ),
    ),
    known_speaker_names: list[str] | None = Form(
        None,
        description=(
            "Repeatable diarization extension. Labels discovered speakers by order, "
            "for example agent, customer."
        ),
    ),
    known_speaker_references: list[str] | None = Form(
        None,
        description=(
            "Repeatable compatibility field. Accepted only together with "
            "known_speaker_names, but local speaker matching against reference audio "
            "is not implemented."
        ),
    ),
    chunking_strategy: str | None = Form(
        None,
        description="Accepted for OpenAI compatibility. Use vad_filter for local VAD behavior.",
    ),
    device: str = Form(DEFAULT_DEVICE, description="Local extension: faster-whisper device."),
    compute_type: str = Form(DEFAULT_COMPUTE_TYPE, description="Local extension: faster-whisper compute_type."),
    beam_size: int = Form(5, description="Local extension: faster-whisper beam_size."),
    vad_filter: bool = Form(True, description="Local extension: faster-whisper VAD filter."),
    diarization_model: str = Form(DEFAULT_DIARIZATION_MODEL, description="Local extension: pyannote pipeline path."),
    min_speakers: int | None = Form(None, description="Local extension: minimum speaker count for pyannote."),
    max_speakers: int | None = Form(None, description="Local extension: maximum speaker count for pyannote."),
    num_speakers: int | None = Form(None, description="Local extension: exact speaker count for pyannote."),
    use_exclusive: bool = Form(True, description="Local extension: use pyannote exclusive diarization when available."),
) -> dict[str, Any] | PlainTextResponse:
    request_id = new_request_id()
    request_token = current_request_id.set(request_id)
    record_tool_call("audio.transcriptions.request", model=model, filename=file.filename)
    _ = chunking_strategy, include, timestamp_granularities
    _ = known_speaker_names, known_speaker_references
    try:
        form = await request.form()
        include_values = form_list(form, "include")
        known_speaker_name_values = form_list(form, "known_speaker_names")
        known_speaker_reference_values = form_list(form, "known_speaker_references")
        timestamp_granularity_values = form_list(form, "timestamp_granularities") or ["segment"]
        stream_enabled = parse_bool(stream, False, "stream")

        registry = request.app.state.model_registry
        configured_model = registry.get(model)
        validate_transcription_request(
            configured_model=configured_model,
            model=model,
            response_format=response_format,
            stream=stream_enabled,
            include=include_values,
            timestamp_granularities=timestamp_granularity_values,
        )

        if known_speaker_reference_values and not known_speaker_name_values:
            raise OpenAIAPIError(
                "known_speaker_references requires known_speaker_names.",
                param="known_speaker_names",
                code="missing_required_parameter",
            )

        record_tool_call(
            "audio.transcriptions.validate",
            model=model,
            response_format=response_format,
            diarization="diarization" in configured_model["capabilities"],
        )
        tmp_path = await save_upload_to_temp_file(file)
        record_tool_call("audio.upload.save", path=tmp_path)
        resolved_diarization_model = (
            diarization_model
            if diarization_model != DEFAULT_DIARIZATION_MODEL
            else configured_model["diarization_model"]
        )

        logger.info(
            "Received transcription request file=%s model=%s response_format=%s language=%s",
            file.filename,
            model,
            response_format,
            language,
        )
        try:
            if "diarization" in configured_model["capabilities"]:
                diarization_service: DiarizationService = request.app.state.diarization_service
                transcription = await asyncio.to_thread(
                    diarization_service.transcribe_with_diarization,
                    tmp_path,
                    model_id=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    device=device,
                    compute_type=compute_type,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    timestamp_granularities=timestamp_granularity_values,
                    diarization_model=resolved_diarization_model,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=num_speakers,
                    use_exclusive=use_exclusive,
                )
            else:
                asr_service: ASRService = request.app.state.asr_service
                transcription = await asyncio.to_thread(
                    asr_service.transcribe,
                    tmp_path,
                    model_id=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    device=device,
                    compute_type=compute_type,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    timestamp_granularities=timestamp_granularity_values,
                )

            record_tool_call("audio.response.format", response_format=response_format)
            formatted = format_openai_response(
                transcription,
                response_format=response_format,
                known_speaker_names=known_speaker_name_values,
                include=include_values,
            )

            record_tool_call("audio.transcriptions.response", response_format=response_format)
            if isinstance(formatted, str):
                media_type = "text/vtt" if response_format == "vtt" else "text/plain"
                return PlainTextResponse(formatted, media_type=media_type)

            return formatted
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            record_tool_call("audio.upload.cleanup", path=tmp_path)
    except Exception as exc:
        record_tool_call("audio.transcriptions.error", status="error", error=str(exc))
        raise
    finally:
        current_request_id.reset(request_token)
