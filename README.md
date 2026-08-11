# ASR Kit

Локальный OpenAI-compatible сервер транскрибации: `faster-whisper` + опциональный **NVIDIA NeMo** + опциональная диаризация `pyannote.audio`.

Сервис не скачивает модели во время работы. Модели описываются в YAML-конфиге и должны быть заранее доступны в локальной директории, подключенной в контейнер.

## API

```text
GET  /health
GET  /v1/status
GET  /v1/models
GET  /v1/models/{model}
POST /v1/audio/transcriptions
WS   /v1/realtime?model=<model_id>
GET  /realtime-demo
GET  /docs
```

## Конфиг моделей

Модели, которые видны в `/v1/models`, задаются в `config/models.yaml`.

Поле `backend` выбирает ASR-движок:

| `backend` | Движок | `path` |
|-----------|--------|--------|
| `faster-whisper` (по умолчанию) | faster-whisper / CTranslate2 | директория CT2 или HF cache `models--...` |
| `nemo` | NVIDIA NeMo ASR | файл `*.nemo` |

```yaml
models:
  - id: bond005-whisper-podlodka-turbo
    backend: faster-whisper
    path: /workspace/models/models--bond005--whisper-podlodka-turbo-ct2
    owned_by: local
    capabilities:
      - transcription

  - id: bond005-whisper-podlodka-turbo-diarize
    backend: faster-whisper
    path: /workspace/models/models--bond005--whisper-podlodka-turbo-ct2
    owned_by: local
    capabilities:
      - transcription
      - diarization
    diarization_model: /workspace/models/pyannote/speaker-diarization-community-1

  # Требует nemo_toolkit (requirements.nemo.txt / Dockerfile.nemo)
  - id: nemo-stt-ru-conformer-ctc
    backend: nemo
    path: /workspace/models/nemo/stt_ru_conformer_ctc_large.nemo
    owned_by: nvidia-nemo
    capabilities:
      - transcription
```

Для `faster-whisper` `path` может указывать на HF cache root вида `models--...`; сервер сам развернёт его в `snapshots/<hash>`. Для NeMo `path` должен быть абсолютным путём к локальному файлу `.nemo` (без скачивания с NGC в рантайме). `diarization_model` должен указывать на директорию pyannote pipeline с `config.yaml` — диаризация общая для обоих ASR-бэкендов.

После изменений `config/models.yaml` перезапустите сервис:

```bash
docker compose restart asr-api
```

## NVIDIA NeMo

NeMo — отдельный опциональный модуль (`app/services/nemo_asr.py`). Дефолтный образ без `nemo_toolkit` не меняется.

1. Положите `.nemo` чекпоинт в volume моделей, например `/workspace/models/nemo/...`.
2. Добавьте запись с `backend: nemo` в `config/models.yaml`.
3. Соберите образ с NeMo:

```bash
docker compose -f docker-compose.nemo.yml up -d --build
```

Или локально:

```bash
pip install -r requirements.txt
pip install -r requirements.nemo.txt
```

Ограничения MVP:

- REST `POST /v1/audio/transcriptions` — да
- `WS /v1/realtime` — да для `faster-whisper` и `nemo` (тот же VAD + chunked `transcribe_array`)
- Word timestamps для NeMo — пока пустые (`words: []`), текст фразы возвращается
- Параметры `beam_size` / `vad_filter` / `prompt` / `compute_type` ориентированы на faster-whisper и для NeMo игнорируются

## Код

```text
app/server.py                 # сборка FastAPI app
app/config.py                 # env/default settings
app/model_registry.py         # загрузка и валидация config/models.yaml
app/routers/                  # /health, /v1/models, /v1/audio/transcriptions, /v1/realtime
app/services/asr.py           # фасад ASR (dispatch по backend)
app/services/whisper_asr.py   # faster-whisper
app/services/nemo_asr.py      # NVIDIA NeMo (optional import)
app/services/diarization.py   # pyannote
app/openai_format.py          # OpenAI-compatible responses
app/openai_realtime_events.py # Realtime WebSocket event helpers
```

## Realtime WebSocket transcription

Псевдо-реалтайм транскрибация через WebSocket — локальное подмножество [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime). Только транскрибация: без voice agent, TTS и tools. Работает с моделями `backend: faster-whisper` и `backend: nemo`. Поддерживается прогрессивная детализация: word timestamps (для Whisper), provisional speaker labels и финальная pyannote-диаризация по завершении сессии.

**Подключение:** `WS /v1/realtime?model=<model_id>`

**Формат аудио:** PCM16, mono, **16 kHz** (не 24 kHz как у OpenAI GA). Клиент кодирует чанки в base64.

### Клиент → сервер

| Событие | Описание |
|---------|----------|
| `session.update` | Модель, язык, `input_audio_format`, порог VAD, детализация |
| `input_audio_buffer.append` | Base64 PCM16 чанк |
| `input_audio_buffer.commit` | Принудительно обработать остаток буфера (одна фраза) |
| `input_audio_buffer.clear` | Сбросить буфер |
| `session.end` | **Аудио закончилось**: flush буфера, финальная диаризация, `session.ended` |

### Жизненный цикл сессии

Три разных момента, которые не стоит путать:

| Момент | Кто инициирует | Что происходит |
|--------|----------------|----------------|
| Конец **фразы** | Сервер (VAD) | `completed` + опционально `words`, `speaker_assigned` |
| Flush **хвоста** | Клиент (`commit`) | Обработка незавершённой фразы без паузы VAD |
| Конец **сессии** | Клиент (`session.end`) | Flush + finalize + `session.ended` |

Рекомендуемая последовательность:

```text
1. WS connect + session.update
2. input_audio_buffer.append (много раз)
3. session.end                         ← «аудио закончилось, жду финал»
4. дождаться session.diarization.completed (если finalize=true)
5. дождаться session.ended
6. ws.close()
```

`session.end` автоматически делает то, что раньше требовало отдельного `commit`: сбрасывает остаток буфера, ждёт транскрипцию последней фразы, запускает pyannote (если `finalize: true`), затем шлёт `session.ended`. **Не закрывайте WebSocket до `session.ended`.**

Пример завершения сессии:

```json
{ "type": "session.end" }
```

Ответ сервера (порядок):

```text
... completed / speaker_assigned для последней фразы ...
session.diarization.completed   ← только если finalize=true
session.ended                   ← сигнал «можно закрывать соединение»
```

`session.ended` всегда приходит последним:

```json
{
  "type": "session.ended",
  "item_count": 12,
  "diarization_finalized": true
}
```

Если клиент оборвёт соединение без `session.end`, сервер попытается сделать finalize в best-effort режиме, но ответ может не дойти.

Пример `session.update` с детализацией:

```json
{
  "type": "session.update",
  "session": {
    "input_audio_transcription": {
      "model": "bond005-whisper-podlodka-turbo-diarize",
      "language": "ru",
      "timestamp_granularities": ["word"]
    },
    "speaker_diarization": {
      "enabled": true,
      "mode": "provisional",
      "finalize": true,
      "max_speakers": 4
    }
  }
}
```

### Сервер → клиент

| Событие | Описание |
|---------|----------|
| `session.created` | Сессия открыта |
| `session.updated` | Настройки применены |
| `input_audio_buffer.speech_started` | Начало фразы (server VAD) |
| `input_audio_buffer.speech_stopped` | Конец фразы |
| `input_audio_buffer.committed` | Буфер отправлен на ASR |
| `conversation.item.input_audio_transcription.delta` | Частичный/сегментный текст (v1: полный сегмент) |
| `conversation.item.input_audio_transcription.completed` | Финальный текст сегмента; опционально `words` |
| `conversation.item.input_audio_transcription.speaker_assigned` | Provisional speaker label для сегмента |
| `conversation.item.input_audio_transcription.speaker_updated` | Уточнение speaker label (в т.ч. после finalize) |
| `session.diarization.completed` | Финальная диаризация всей сессии (перед `session.ended`) |
| `session.ended` | Сессия завершена; можно закрывать WebSocket |
| `error` | Ошибка в стиле OpenAI |

### Поведение

- Server-side VAD по RMS: порог и длительность тишины настраиваются в `session.update.turn_detection`.
- Сегменты короче ~300 ms не транскрибируются.
- `initial_prompt` для continuity берётся из хвоста предыдущего текста (~200 символов).
- Один ASR inference на сессию в момент времени (`beam_size=1`).
- Speaker embedding выполняется отдельно и не блокирует ASR.
- Provisional speaker labels могут уточняться; при `speaker_diarization.finalize=true` в конце сессии запускается полный pyannote pass.
- Кольцевой буфер до ~60 с (настраивается через env).

### Прогрессивная детализация

1. Сразу после паузы: `delta` + `completed` с текстом.
2. В том же `completed` (если включено): `words` с абсолютными таймкодами сессии.
3. Через ~0.5–1 с: `speaker_assigned` с provisional label (`A`, `B`, …).
4. После `session.end` (если `finalize=true`): `session.diarization.completed`, затем `session.ended`.

### Демо

Откройте в браузере:

```text
http://localhost:8000/realtime-demo?model=<model_id>&words=1&speakers=1
```

Микрофон → resample до 16 kHz → WebSocket → живой текст, слова и спикеры.

### Ожидания по задержке

Задержка складывается из: накопления аудио до конца фразы (VAD), времени inference ASR (`faster-whisper` или NeMo) на GPU/CPU и размера сегмента. Это **не** true streaming ASR token-by-token; типично сотни миллисекунд — несколько секунд после паузы в речи. Speaker labels приходят ещё с небольшой дополнительной задержкой.

### Отличия от OpenAI Realtime

- Аудио 16 kHz, не 24 kHz
- Нет `response.create`, tools, TTS, conversation items кроме транскрипции
- `delta` в v1 дублирует полный сегмент (нет посимвольного стриминга)
- `speaker_diarization` и `session.diarization.completed` — локальные расширения
- `POST /v1/audio/transcriptions` с `stream=true` по-прежнему не поддерживается

### Env (realtime)

```text
REALTIME_SAMPLE_RATE=16000
REALTIME_MAX_BUFFER_SEC=60
REALTIME_MIN_SEGMENT_MS=300
REALTIME_VAD_THRESHOLD=0.012
REALTIME_SILENCE_DURATION_MS=700
REALTIME_WS_IDLE_TIMEOUT_SEC=300
REALTIME_SPEAKER_EMBEDDING_MODEL=
REALTIME_SPEAKER_SIMILARITY_THRESHOLD=0.75
REALTIME_SPEAKER_MIN_SEGMENT_SEC=0.5
REALTIME_SPEAKER_MAX_SPEAKERS=8
```

`REALTIME_SPEAKER_EMBEDDING_MODEL` — опциональный абсолютный путь к pyannote embedding model. Если пусто, сервер пытается найти embedding рядом с `diarization_model`.

## CI/CD

При push в `master` или теге `v*` GitHub Actions собирает образы и публикует их в [GitHub Container Registry](https://github.com/mishzorikhin/asr-kit/pkgs/container/asr-kit):

| Тег | Dockerfile | Платформа | Назначение |
|-----|------------|-----------|------------|
| `cuda12`, `latest` | `Dockerfile` | `linux/amd64` | GPU (CUDA 12.8, NVIDIA) |
| `cpu` | `Dockerfile.cpu` | `linux/amd64` | CPU (AMD Ryzen/EPYC, Intel и др.) |
| `arm64` | `Dockerfile.arm64` | `linux/arm64` | CPU (Apple Silicon и др.) |

Пример запуска готового образа:

```bash
ASR_IMAGE=ghcr.io/mishzorikhin/asr-kit:cuda12 docker compose up -d
```

Для CPU на AMD/Intel (linux/amd64, без NVIDIA GPU):

```bash
docker compose -f docker-compose.cpu.yml up -d
# в .env: ASR_IMAGE=ghcr.io/mishzorikhin/asr-kit:cpu
```

Для ARM:

```bash
docker compose -f docker-compose.arm.yml up -d
# в .env: ASR_IMAGE=ghcr.io/mishzorikhin/asr-kit:arm64
```

Пакет в GHCR по умолчанию может быть приватным. Чтобы сделать его публичным: **GitHub → Packages → asr-kit → Package settings → Change visibility**.

Pull request'ы только проверяют сборку, без публикации в registry.

### Если push падает с `permission_denied: write_package`

Workflow permissions уже на **Read and write** — значит, проблема в доступе к пакету GHCR, а не в YAML.

**Вариант A — привязать репозиторий к пакету (рекомендуется):**

1. Откройте https://github.com/users/mishzorikhin/packages/container/asr-kit/settings  
   (если пакета ещё нет — сначала создайте пустой push вручную или перейдите к варианту B)
2. **Manage Actions access** → **Add repository** → `mishzorikhin/asr-kit` → роль **Write**
3. **Connect repository** (если есть) → выберите `mishzorikhin/asr-kit`
4. Перезапустите workflow: **Actions → Build and push container → Re-run all jobs**

**Вариант B — PAT (если вариант A не помог):**

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. Создайте токен со scope `write:packages` (и `read:packages`)
3. В репозитории: **Settings → Secrets and variables → Actions → New repository secret**
4. Имя: `GHCR_TOKEN`, значение: токен
5. Перезапустите workflow — он использует `GHCR_TOKEN` вместо `GITHUB_TOKEN`

## Запуск

Требования:

- Docker и Docker Compose
- NVIDIA Container Toolkit — только для `docker-compose.yml` (GPU)
- локальная директория с моделями, подключенная в контейнер как `/workspace/models`

### GPU (NVIDIA)

```bash
docker compose up -d
curl http://localhost:8000/v1/models
```

Проверка GPU внутри контейнера:

```bash
docker exec -it asr-kit nvidia-smi
```

### CPU на AMD / Intel (linux/amd64)

Образ `cpu` — PyTorch CPU + faster-whisper с `int8`. Подходит для машин без NVIDIA GPU (AMD Ryzen, EPYC, Intel Xeon и т.д.). NVIDIA Container Toolkit не нужен.

```bash
docker compose -f docker-compose.cpu.yml up -d
curl http://localhost:8000/v1/models
```

По умолчанию `DEFAULT_DEVICE=cpu`, `DEFAULT_COMPUTE_TYPE=int8`. Для ускорения на многоядерных AMD можно задать `OMP_NUM_THREADS` (0 = все ядра).

### Apple Silicon / ARM64

```bash
docker compose -f docker-compose.arm.yml up -d
```

### Общие настройки

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

## Автовыгрузка моделей

Загруженные `faster-whisper` модели и `pyannote` pipelines автоматически выгружаются из памяти, если не используются.

Настройки через env:

```bash
MODEL_IDLE_TTL_SECONDS=600        # сколько секунд модель может простаивать
MODEL_UNLOAD_INTERVAL_SECONDS=30  # как часто проверять простаивающие модели
MODEL_UNLOAD_AFTER_REQUEST=false  # выгружать сразу после обработки запроса

# Realtime WebSocket (см. раздел Realtime WebSocket transcription)
REALTIME_SAMPLE_RATE=16000
REALTIME_MAX_BUFFER_SEC=60
REALTIME_VAD_THRESHOLD=0.012
REALTIME_SILENCE_DURATION_MS=700
```

`MODEL_IDLE_TTL_SECONDS=0` отключает фоновую автовыгрузку по простою. Если `MODEL_UNLOAD_AFTER_REQUEST=true`, модель выгружается сразу после обработки последнего активного запроса. Модель не выгружается, пока по ней выполняется активный запрос.

## Автоскейлинг реплик Whisper

Whisper (`faster-whisper`) живёт **внутри** API-процесса. При включённом автоскейле каждый загруженный экземпляр модели — реплика, которая обслуживает один inference за раз.

Если приходит второй запрос, а единственная реплика занята:

1. Сервис проверяет лимит `WHISPER_MAX_REPLICAS` и пытается поднять ещё один `WhisperModel`.
2. На CUDA новая реплика по возможности садится на менее загруженный `device_index` (другой GPU).
3. Запрос уходит на новую реплику. Если поднять нельзя (лимит / OOM), запрос ждёт свободную реплику до `WHISPER_REPLICA_WAIT_SECONDS`.

```bash
WHISPER_AUTOSCALE_ENABLED=true
WHISPER_MAX_REPLICAS=2            # по умолчанию: число CUDA GPU или 2 на CPU
WHISPER_REPLICA_WAIT_SECONDS=300  # 0 = ждать бесконечно
```

Состояние пула: `GET /v1/status`.

Пока автоскейл только для Whisper. NeMo и отдельные Docker-реплики не затрагиваются. Учитывайте VRAM: каждая реплика держит свою копию весов.

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
