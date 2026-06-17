from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.tool_calls import list_tool_calls

router = APIRouter(tags=["ui"])


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def tool_calls_ui() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASR tool calls</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #18202a; }
    header { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #d9dee7; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 18px 20px 32px; }
    .toolbar { display: flex; align-items: center; gap: 10px; }
    button { height: 34px; border: 1px solid #c8d0db; border-radius: 6px; background: #ffffff; color: #18202a; padding: 0 12px; font: inherit; cursor: pointer; }
    button:hover { background: #eef2f7; }
    .status { font-size: 13px; color: #5a6675; min-width: 120px; text-align: right; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e7ebf0; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #eef2f7; color: #394557; font-weight: 650; }
    tr:last-child td { border-bottom: 0; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { display: inline-flex; align-items: center; min-height: 22px; border-radius: 999px; padding: 0 8px; background: #e9f6ef; color: #17633a; font-weight: 650; }
    .pill.error { background: #fdecec; color: #a33131; }
    .empty { padding: 24px; text-align: center; color: #5a6675; background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      .status { text-align: left; }
      th:nth-child(1), td:nth-child(1), th:nth-child(3), td:nth-child(3) { display: none; }
      main { padding: 12px; }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111820; color: #e8edf3; }
      header, table, .empty, button { background: #17212b; color: #e8edf3; border-color: #2a3745; }
      th { background: #202c38; color: #cbd5e1; }
      td { border-color: #263342; }
      button:hover { background: #223040; }
      .status { color: #9aa8b7; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Вызовы инструментов ASR</h1>
    <div class="toolbar">
      <button id="refresh" type="button">Обновить</button>
      <button id="pause" type="button">Пауза</button>
      <span class="status" id="status">Загрузка...</span>
    </div>
  </header>
  <main id="app"></main>
  <script>
    const app = document.querySelector("#app");
    const statusEl = document.querySelector("#status");
    const pauseBtn = document.querySelector("#pause");
    let paused = false;

    function formatTime(seconds) {
      return new Date(seconds * 1000).toLocaleTimeString();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function render(events) {
      if (!events.length) {
        app.innerHTML = '<div class="empty">Пока нет вызовов. Выполните запрос на транскрибацию.</div>';
        return;
      }
      const rows = events.slice().reverse().map((event) => {
        const details = escapeHtml(JSON.stringify(event.details ?? {}, null, 2));
        const statusClass = event.status === "error" ? "pill error" : "pill";
        return `<tr>
          <td><code>${event.id}</code></td>
          <td>${formatTime(event.time)}</td>
          <td><code>${escapeHtml(event.request_id ?? "-")}</code></td>
          <td><code>${escapeHtml(event.name)}</code></td>
          <td><span class="${statusClass}">${escapeHtml(event.status)}</span></td>
          <td><code>${details}</code></td>
        </tr>`;
      }).join("");
      app.innerHTML = `<table>
        <thead><tr><th>ID</th><th>Время</th><th>Запрос</th><th>Инструмент</th><th>Статус</th><th>Детали</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }

    async function loadEvents() {
      if (paused) return;
      const response = await fetch("/v1/tool-calls?limit=200", { cache: "no-store" });
      const payload = await response.json();
      render(payload.data);
      statusEl.textContent = `Событий: ${payload.data.length}`;
    }

    document.querySelector("#refresh").addEventListener("click", loadEvents);
    pauseBtn.addEventListener("click", () => {
      paused = !paused;
      pauseBtn.textContent = paused ? "Продолжить" : "Пауза";
      if (!paused) loadEvents();
    });
    loadEvents();
    setInterval(loadEvents, 2000);
  </script>
</body>
</html>"""


@router.get("/v1/tool-calls", summary="List internal tool calls")
def get_tool_calls(limit: int = Query(200, ge=1, le=500)) -> dict[str, list[dict[str, Any]]]:
    return {"data": list_tool_calls(limit)}


@router.get("/realtime-demo", response_class=HTMLResponse, include_in_schema=False)
def realtime_demo(
    model: str = Query("bond005-whisper-podlodka-turbo", description="Model id for the demo"),
) -> str:
    safe_model = model.replace("\\", "\\\\").replace('"', '\\"')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realtime transcription demo</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #18202a; }}
    header {{ padding: 16px 20px; background: #fff; border-bottom: 1px solid #d9dee7; }}
    main {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    p {{ margin: 0; color: #5a6675; font-size: 14px; }}
    .panel {{ margin-top: 16px; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 16px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
    button {{ height: 36px; border: 1px solid #c8d0db; border-radius: 6px; background: #fff; padding: 0 14px; font: inherit; cursor: pointer; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    button.primary {{ background: #17633a; color: #fff; border-color: #17633a; }}
    .status {{ font-size: 13px; color: #5a6675; margin-bottom: 10px; }}
    #transcript {{ min-height: 180px; white-space: pre-wrap; line-height: 1.5; font-size: 15px; }}
    #log {{ margin-top: 12px; max-height: 180px; overflow: auto; font-family: ui-monospace, monospace; font-size: 12px; color: #5a6675; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #111820; color: #e8edf3; }}
      header, .panel, button {{ background: #17212b; color: #e8edf3; border-color: #2a3745; }}
      button.primary {{ background: #2f8f57; border-color: #2f8f57; }}
      p, .status, #log {{ color: #9aa8b7; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Realtime transcription demo</h1>
    <p>Microphone → WebSocket <code>/v1/realtime</code> → live text. Audio is resampled to PCM16 mono 16 kHz.</p>
  </header>
  <main>
    <div class="panel">
      <div class="toolbar">
        <button id="start" class="primary" type="button">Start</button>
        <button id="stop" type="button" disabled>Stop</button>
        <button id="clear" type="button" disabled>Clear buffer</button>
      </div>
      <div class="status" id="status">Disconnected</div>
      <div id="transcript"></div>
      <pre id="log"></pre>
    </div>
  </main>
  <script>
    const MODEL_ID = "{safe_model}";
    const TARGET_RATE = 16000;
    const startBtn = document.querySelector("#start");
    const stopBtn = document.querySelector("#stop");
    const clearBtn = document.querySelector("#clear");
    const statusEl = document.querySelector("#status");
    const transcriptEl = document.querySelector("#transcript");
    const logEl = document.querySelector("#log");

    let ws = null;
    let audioContext = null;
    let mediaStream = null;
    let processor = null;
    let running = false;

    function log(line) {{
      logEl.textContent = `${{new Date().toLocaleTimeString()}} ${{line}}\\n` + logEl.textContent;
    }}

    function wsUrl() {{
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      return `${{protocol}}//${{location.host}}/v1/realtime?model=${{encodeURIComponent(MODEL_ID)}}`;
    }}

    function floatToPcm16Base64(float32Array) {{
      const pcm = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {{
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }}
      const bytes = new Uint8Array(pcm.buffer);
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {{
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
      }}
      return btoa(binary);
    }}

    function downsample(buffer, inputRate, outputRate) {{
      if (outputRate === inputRate) return buffer;
      const ratio = inputRate / outputRate;
      const newLength = Math.round(buffer.length / ratio);
      const result = new Float32Array(newLength);
      for (let i = 0; i < newLength; i++) {{
        const idx = i * ratio;
        const idxFloor = Math.floor(idx);
        const idxCeil = Math.min(idxFloor + 1, buffer.length - 1);
        const weight = idx - idxFloor;
        result[i] = buffer[idxFloor] * (1 - weight) + buffer[idxCeil] * weight;
      }}
      return result;
    }}

    function sendEvent(event) {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify(event));
    }}

    async function start() {{
      transcriptEl.textContent = "";
      logEl.textContent = "";
      statusEl.textContent = "Connecting...";
      ws = new WebSocket(wsUrl());

      ws.onopen = async () => {{
        statusEl.textContent = "Connected. Starting microphone...";
        sendEvent({{
          type: "session.update",
          session: {{
            input_audio_format: "pcm16",
            input_audio_transcription: {{ model: MODEL_ID, language: "ru" }},
            turn_detection: {{ type: "server_vad", threshold: 0.012, silence_duration_ms: 700 }},
          }},
        }});

        mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {{
          if (!running) return;
          const input = event.inputBuffer.getChannelData(0);
          const resampled = downsample(input, audioContext.sampleRate, TARGET_RATE);
          sendEvent({{
            type: "input_audio_buffer.append",
            audio: floatToPcm16Base64(resampled),
          }});
        }};
        source.connect(processor);
        processor.connect(audioContext.destination);
        running = true;
        statusEl.textContent = `Listening (${{audioContext.sampleRate}} Hz → ${{TARGET_RATE}} Hz)`;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        clearBtn.disabled = false;
        log("WebSocket connected");
      }};

      ws.onmessage = (message) => {{
        const event = JSON.parse(message.data);
        if (event.type === "conversation.item.input_audio_transcription.completed") {{
          const line = event.transcript || "";
          transcriptEl.textContent += (transcriptEl.textContent ? "\\n" : "") + line;
        }} else if (event.type === "error") {{
          log(`ERROR: ${{event.error?.message || "unknown"}}`);
        }} else {{
          log(event.type);
        }}
      }};

      ws.onclose = () => {{
        statusEl.textContent = "Disconnected";
        cleanupAudio();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        clearBtn.disabled = true;
      }};

      ws.onerror = () => log("WebSocket error");
    }}

    function cleanupAudio() {{
      running = false;
      if (processor) {{
        processor.disconnect();
        processor = null;
      }}
      if (audioContext) {{
        audioContext.close();
        audioContext = null;
      }}
      if (mediaStream) {{
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
      }}
    }}

    function stop() {{
      if (ws && ws.readyState === WebSocket.OPEN) {{
        sendEvent({{ type: "input_audio_buffer.commit" }});
        ws.close();
      }}
      cleanupAudio();
    }}

    startBtn.addEventListener("click", start);
    stopBtn.addEventListener("click", stop);
    clearBtn.addEventListener("click", () => sendEvent({{ type: "input_audio_buffer.clear" }}));
  </script>
</body>
</html>"""
