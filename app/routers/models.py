from typing import Any

from fastapi import APIRouter, Request

from app.errors import OpenAIAPIError
from app.model_registry import ModelRegistry

router = APIRouter(prefix="/v1", tags=["models"])

MODEL_OBJECT_EXAMPLE = {
    "id": "whisper-ru-turbo",
    "object": "model",
    "created": 0,
    "owned_by": "local",
}

MODEL_LIST_EXAMPLE = {
    "object": "list",
    "data": [
        MODEL_OBJECT_EXAMPLE,
        {
            "id": "whisper-ru-turbo-diarize",
            "object": "model",
            "created": 0,
            "owned_by": "local",
        },
    ],
}


def registry_from_request(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


@router.get(
    "/models",
    summary="List models",
    description=(
        "Lists models configured for this server in OpenAI-compatible shape. "
        "Use a regular ASR model for `json`, `text`, `srt`, `verbose_json`, and `vtt`. "
        "Use a diarization-capable model, usually with a `-diarize` suffix, for "
        "`response_format=diarized_json`."
    ),
    responses={
        200: {
            "description": "OpenAI-compatible model list.",
            "content": {"application/json": {"example": MODEL_LIST_EXAMPLE}},
        }
    },
)
def list_models(request: Request) -> dict[str, Any]:
    registry = registry_from_request(request)
    return {
        "object": "list",
        "data": registry.list(),
    }


@router.get(
    "/models/{model_id}",
    summary="Retrieve model",
    description=(
        "Retrieves a configured model by id. The response intentionally follows "
        "OpenAI's compact model object and does not expose local filesystem paths."
    ),
    responses={
        200: {
            "description": "OpenAI-compatible model object.",
            "content": {"application/json": {"example": MODEL_OBJECT_EXAMPLE}},
        },
        404: {
            "description": "OpenAI-style model_not_found error.",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "message": "The model 'missing-model' does not exist.",
                            "type": "invalid_request_error",
                            "param": "model",
                            "code": "model_not_found",
                        }
                    }
                }
            },
        },
    },
)
def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
    registry = registry_from_request(request)
    if model_id not in registry.models:
        raise OpenAIAPIError(
            f"The model '{model_id}' does not exist.",
            status_code=404,
            param="model",
            code="model_not_found",
        )

    return registry.model_object(model_id)
