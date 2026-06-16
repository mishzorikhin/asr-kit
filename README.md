# ASR Kit

Локальный OpenAI-compatible сервер транскрибации: `faster-whisper` + опциональная диаризация `pyannote.audio`.

Сервис не скачивает модели во время работы. Модели описываются в YAML-конфиге и должны быть заранее доступны в локальной директории, подключенной в контейнер.

## API

```text
GET  /health
GET  /v1/models
GET  /v1/models/{model}
POST /v1/audio/transcriptions
GET  /docs
```

## Конфиг моделей

Модели, которые видны в `/v1/models`, задаются в `config/models.yaml`.

```yaml
models:
  - id: bond005-whisper-podlodka-turbo
    path: /workspace/models/models--bond005--whisper-podlodka-turbo-ct2
    owned_by: local
    capabilities:
      - transcription

  - id: bond005-whisper-podlodka-turbo-diarize
    path: /workspace/models/models--bond005--whisper-podlodka-turbo-ct2
    owned_by: local
    capabilities:
      - transcription
      - diarization
    diarization_model: /workspace/models/pyannote/speaker-diarization-community-1
```

`path` может указывать на HF cache root вида `models--...`; сервер сам развернёт его в `snapshots/<hash>`. `diarization_model` должен указывать на директорию pyannote pipeline с `config.yaml`.

После изменений `config/models.yaml` перезапустите сервис:

```bash
docker compose restart asr-api
```

## Код

```text
app/server.py              # сборка FastAPI app
app/config.py              # env/default settings
app/model_registry.py      # загрузка и валидация config/models.yaml
app/routers/               # /health, /v1/models, /v1/audio/transcriptions
app/services/              # faster-whisper и pyannote
app/openai_format.py       # OpenAI-compatible responses
```

## Запуск

Требования:

- Docker и Docker Compose
- NVIDIA Container Toolkit для запуска на GPU
- локальная директория с моделями, подключенная в контейнер как `/workspace/models`

Перед запуском проверьте пути в `docker-compose.yml`:

```yaml
volumes:
  - ./config/models.yaml:/workspace/config/models.yaml:ro
  - /path/to/models:/workspace/models
  - /path/to/hf-cache:/workspace/hf-cache
```

Можно скопировать пример переменных и поправить пути под свою машину:

```bash
cp .env.example .env
```

Запуск:

```bash
docker compose up -d
curl http://localhost:8000/v1/models
```

Проверка GPU внутри контейнера:

```bash
docker exec -it faster-whisper-api nvidia-smi
```

## Автовыгрузка моделей

Загруженные `faster-whisper` модели и `pyannote` pipelines автоматически выгружаются из памяти, если не используются.

Настройки через env:

```bash
MODEL_IDLE_TTL_SECONDS=600        # сколько секунд модель может простаивать
MODEL_UNLOAD_INTERVAL_SECONDS=30  # как часто проверять простаивающие модели
MODEL_UNLOAD_AFTER_REQUEST=false  # выгружать сразу после обработки запроса
```

`MODEL_IDLE_TTL_SECONDS=0` отключает фоновую автовыгрузку по простою. Если `MODEL_UNLOAD_AFTER_REQUEST=true`, модель выгружается сразу после обработки последнего активного запроса. Модель не выгружается, пока по ней выполняется активный запрос.

## Переменные окружения

```text
MODELS_CONFIG_PATH=/workspace/config/models.yaml
MODEL_DIR=/workspace/models
LOG_LEVEL=INFO

DEFAULT_DEVICE=cuda
DEFAULT_COMPUTE_TYPE=float16
DEFAULT_LANGUAGE=ru
DEFAULT_DIARIZATION_MODEL=/workspace/models/pyannote/speaker-diarization-community-1

HF_HOME=/workspace/hf-cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

## Python SDK

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://localhost:8000/v1")

with open("speech.mp3", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="bond005-whisper-podlodka-turbo",
        file=audio,
        response_format="verbose_json",
        language="ru",
        timestamp_granularities=["segment", "word"],
    )

print(transcript.text)
```

С диаризацией:

```python
with open("speech.mp3", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="bond005-whisper-podlodka-turbo-diarize",
        file=audio,
        response_format="diarized_json",
        language="ru",
        known_speaker_names=["agent", "customer"],
        extra_body={"min_speakers": 2, "max_speakers": 6},
    )

for segment in transcript.segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
```

## Curl

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@speech.mp3" \
  -F "model=bond005-whisper-podlodka-turbo" \
  -F "language=ru" \
  -F "response_format=json"
```

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@speech.mp3" \
  -F "model=bond005-whisper-podlodka-turbo-diarize" \
  -F "language=ru" \
  -F "response_format=diarized_json" \
  -F "min_speakers=2" \
  -F "max_speakers=6"
```
