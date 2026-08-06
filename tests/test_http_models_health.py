from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.errors import OpenAIAPIError, openai_error_handler
from app.model_registry import ModelRegistry
from app.routers import health, models


def test_health_endpoint() -> None:
    app = FastAPI()
    app.include_router(health.router)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_list_endpoint(tmp_path, monkeypatch) -> None:
    import yaml

    whisper_dir = tmp_path / "w"
    whisper_dir.mkdir()
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "demo",
                        "path": str(whisper_dir),
                        "owned_by": "local",
                        "created": 1,
                        "capabilities": ["transcription"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.model_registry.MODELS_CONFIG_PATH", config)

    app = FastAPI()
    app.add_exception_handler(OpenAIAPIError, openai_error_handler)
    app.state.model_registry = ModelRegistry()
    app.include_router(models.router)
    client = TestClient(app)

    listed = client.get("/v1/models")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "demo"

    one = client.get("/v1/models/demo")
    assert one.status_code == 200
    assert one.json()["id"] == "demo"

    missing = client.get("/v1/models/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
