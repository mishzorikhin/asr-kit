FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/asr-kit-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    python3-venv \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/python" -m pip install --upgrade pip setuptools wheel


FROM base AS deps

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install -r /workspace/requirements.txt


FROM deps AS runtime

LABEL org.opencontainers.image.source=https://github.com/mishzorikhin/asr-kit

COPY app /workspace/app
COPY config /workspace/config

ENV HF_HOME=/workspace/hf-cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENV MODEL_DIR=/workspace/models
ENV DEFAULT_DIARIZATION_MODEL=/workspace/models/pyannote/speaker-diarization-community-1
ENV DEFAULT_DEVICE=cuda
ENV DEFAULT_COMPUTE_TYPE=float16
ENV DEFAULT_LANGUAGE=ru

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
