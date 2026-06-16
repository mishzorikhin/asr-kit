from typing import Any

from app.errors import OpenAIAPIError


def usage_for_duration(duration: float) -> dict[str, Any]:
    return {
        "type": "duration",
        "seconds": round(duration),
    }


def join_segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(segment["text"] for segment in segments if segment["text"]).strip()


def format_timestamp(seconds: float, separator: str) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def format_srt(segments: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                str(index),
                (
                    f"{format_timestamp(segment['start'], ',')} --> "
                    f"{format_timestamp(segment['end'], ',')}"
                ),
                segment["text"],
            ]
        )
        for index, segment in enumerate(segments, start=1)
    )


def format_vtt(segments: list[dict[str, Any]]) -> str:
    return "WEBVTT\n\n" + "\n\n".join(
        "\n".join(
            [
                (
                    f"{format_timestamp(segment['start'], '.')} --> "
                    f"{format_timestamp(segment['end'], '.')}"
                ),
                segment["text"],
            ]
        )
        for segment in segments
    )


def speaker_labels(
    segments: list[dict[str, Any]],
    known_speaker_names: list[str],
) -> dict[str, str]:
    labels = {}

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        if speaker not in labels:
            if len(labels) < len(known_speaker_names):
                labels[speaker] = known_speaker_names[len(labels)]
            else:
                labels[speaker] = chr(ord("A") + len(labels))

    return labels


def format_json(transcription: dict[str, Any], include: list[str]) -> dict[str, Any]:
    response = {
        "text": join_segment_text(transcription["segments"]),
        "usage": usage_for_duration(transcription["duration"]),
    }

    if "logprobs" in include:
        response["logprobs"] = []

    return response


def format_verbose_json(transcription: dict[str, Any]) -> dict[str, Any]:
    response = {
        "task": "transcribe",
        "language": transcription["language"],
        "duration": transcription["duration"],
        "text": join_segment_text(transcription["segments"]),
        "segments": [
            {
                "id": segment["id"],
                "seek": segment["seek"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "tokens": segment["tokens"],
                "temperature": segment["temperature"],
                "avg_logprob": segment["avg_logprob"],
                "compression_ratio": segment["compression_ratio"],
                "no_speech_prob": segment["no_speech_prob"],
            }
            for segment in transcription["segments"]
        ],
        "usage": usage_for_duration(transcription["duration"]),
    }

    if transcription["words"]:
        response["words"] = transcription["words"]

    return response


def format_diarized_json(
    transcription: dict[str, Any],
    known_speaker_names: list[str],
) -> dict[str, Any]:
    labels = speaker_labels(transcription["segments"], known_speaker_names)
    segments = [
        {
            "type": "transcript.text.segment",
            "id": f"seg_{index:03d}",
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": labels.get(segment.get("speaker", "UNKNOWN"), "A"),
        }
        for index, segment in enumerate(transcription["segments"], start=1)
    ]
    text = "\n".join(
        f"{segment['speaker']}: {segment['text']}"
        for segment in segments
        if segment["text"]
    )

    return {
        "task": "transcribe",
        "duration": transcription["duration"],
        "text": text,
        "segments": segments,
        "usage": usage_for_duration(transcription["duration"]),
    }


def format_openai_response(
    transcription: dict[str, Any],
    *,
    response_format: str,
    known_speaker_names: list[str],
    include: list[str],
) -> dict[str, Any] | str:
    if response_format == "json":
        return format_json(transcription, include)
    if response_format == "text":
        return join_segment_text(transcription["segments"])
    if response_format == "srt":
        return format_srt(transcription["segments"])
    if response_format == "verbose_json":
        return format_verbose_json(transcription)
    if response_format == "vtt":
        return format_vtt(transcription["segments"])
    if response_format == "diarized_json":
        return format_diarized_json(transcription, known_speaker_names)

    raise OpenAIAPIError(
        f"Unsupported response_format: {response_format}",
        param="response_format",
        code="unsupported_response_format",
    )
