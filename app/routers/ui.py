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
