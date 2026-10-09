/* Логика веб-интерфейса «Локальный LLM-чат» (День 27).
   Общается с бэкендом FastAPI: статус, список моделей, чат (SSE), команды. */

"use strict";

const els = {
  chat: document.getElementById("chat"),
  input: document.getElementById("input"),
  send: document.getElementById("btn-send"),
  clear: document.getElementById("btn-clear"),
  export: document.getElementById("btn-export"),
  help: document.getElementById("btn-help"),
  modelSelect: document.getElementById("model-select"),
  modelHint: document.getElementById("model-hint"),
  dot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  metaHistory: document.getElementById("meta-history"),
  metaModel: document.getElementById("meta-model"),
  metaUptime: document.getElementById("meta-uptime"),
};

let liveBubble = null; // «живой» пузырь ассистента во время стриминга
let busy = false;       // идёт ли сейчас запрос

/* ------------------------------------------------------------------ */
/* Утилиты отрисовки                                                    */
/* ------------------------------------------------------------------ */
function createBubble(role, text, { id } = {}) {
  const div = document.createElement("div");
  div.className = "bubble " + role;
  if (id) div.id = id;

  const label = document.createElement("span");
  label.className = "role";
  label.textContent = roleLabel(role);
  div.appendChild(label);

  const content = document.createElement("span");
  content.className = "content";
  content.textContent = text || "";
  div.appendChild(content);

  els.chat.appendChild(div);
  scrollToBottom();
  return div;
}

function roleLabel(role) {
  return {
    user: "Пользователь",
    assistant: "Ассистент",
    system: "Система",
    command: "Команда",
    error: "Ошибка",
  }[role] || role;
}

function bubbleContent(bubble) {
  return bubble ? bubble.querySelector(".content") : null;
}

function addMessage(role, text) {
  return createBubble(role, text);
}

function setLiveText(bubble, text) {
  const c = bubbleContent(bubble);
  if (c) c.textContent = text;
  scrollToBottom();
}

function appendLiveText(bubble, piece) {
  const c = bubbleContent(bubble);
  if (c) c.textContent += piece;
  scrollToBottom();
}

function clearChat() {
  els.chat.innerHTML = "";
  liveBubble = null;
}

function scrollToBottom() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function showHint(text) {
  els.modelHint.textContent = text || "";
}

/* ------------------------------------------------------------------ */
/* Работа с API                                                         */
/* ------------------------------------------------------------------ */
async function getJSON(path) {
  const resp = await fetch(path, { headers: { Accept: "application/json" } });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || ("HTTP " + resp.status));
  return data;
}

async function postJSON(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.error || ("HTTP " + resp.status));
    err.payload = data;
    throw err;
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* Статус и модели                                                      */
/* ------------------------------------------------------------------ */
async function refreshStatus() {
  try {
    const s = await getJSON("/api/status");
    els.dot.className = "dot " + (s.ollama_available ? "ok" : "bad");
    els.statusText.textContent = s.ollama_available
      ? `Ollama доступна${s.ollama_version ? " (v" + s.ollama_version + ")" : ""}`
      : "Ollama не запущена";
    els.metaHistory.textContent = "История: " + s.history_length + " сообщений";
    els.metaModel.textContent = "Модель: " + s.current_model;
    els.metaUptime.textContent = "Аптайм: " + formatUptime(s.uptime_seconds);
  } catch (e) {
    els.dot.className = "dot bad";
    els.statusText.textContent = "Ошибка связи с бэкендом";
  }
}

async function loadModels() {
  try {
    const data = await getJSON("/api/models");
    els.modelSelect.innerHTML = "";
    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name + (m.size_human ? "  (" + m.size_human + ")" : "");
      if (m.current) opt.selected = true;
      els.modelSelect.appendChild(opt);
    });
    if (data.models.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "(нет моделей)";
      opt.disabled = true;
      els.modelSelect.appendChild(opt);
    }
  } catch (e) {
    els.modelSelect.innerHTML = "";
    const opt = document.createElement("option");
    opt.textContent = "(Ollama недоступна)";
    opt.disabled = true;
    els.modelSelect.appendChild(opt);
  }
}

function formatUptime(seconds) {
  let s = Math.floor(seconds || 0);
  const d = Math.floor(s / 86400); s -= d * 86400;
  const h = Math.floor(s / 3600); s -= h * 3600;
  const m = Math.floor(s / 60); s -= m * 60;
  const parts = [];
  if (d) parts.push(d + " д");
  if (h) parts.push(h + " ч");
  if (m) parts.push(m + " мин");
  parts.push(s + " с");
  return parts.join(" ");
}

/* ------------------------------------------------------------------ */
/* Отправка сообщения и чтение SSE-потока                               */
/* ------------------------------------------------------------------ */
async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || busy) return;

  const isCmd = text.startsWith("/");
  addMessage(isCmd ? "command" : "user", text);
  els.input.value = "";
  setBusy(true);

  liveBubble = null;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message: text, stream: true }),
    });

    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.error || ("HTTP " + resp.status));
    }

    await readSSE(resp.body, handleEvent);
  } catch (e) {
    finishLive();
    addMessage("error", e.message || String(e));
  } finally {
    setBusy(false);
    await refreshStatus();
  }
}

async function readSSE(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        let evt;
        try { evt = JSON.parse(json); } catch (_) { continue; }
        onEvent(evt);
      }
    }
  }
}

function handleEvent(evt) {
  if (evt.type === "token") {
    if (!liveBubble) liveBubble = createBubble("assistant", "", { id: "live" });
    appendLiveText(liveBubble, evt.content);
  } else if (evt.type === "done") {
    finishLive();
    if (evt.history_length !== undefined) {
      els.metaHistory.textContent = "История: " + evt.history_length + " сообщений";
    }
    handleAction(evt.data || {}, evt);
  } else if (evt.type === "error") {
    finishLive(true);
    addMessage("error", evt.message || "Неизвестная ошибка");
  }
}

function finishLive(discardEmpty) {
  if (!liveBubble) return;
  const c = bubbleContent(liveBubble);
  if (discardEmpty && c && c.textContent.trim() === "") {
    liveBubble.remove();
  } else {
    liveBubble.classList.remove("streaming");
  }
  liveBubble = null;
}

/* ------------------------------------------------------------------ */
/* Реакция на подсказки с бэкенда (data.action)                          */
/* ------------------------------------------------------------------ */
function handleAction(data, evt) {
  const action = data.action;
  if (action === "clear") {
    clearChat();
    addMessage("system", "Чат очищен.");
  } else if (action === "reset") {
    clearChat();
    const model = data.model || (evt && evt.model);
    if (model) selectModelValue(model);
    addMessage("system", "Выполнен сброс к настройкам по умолчанию.");
    loadModels();
  } else if (action === "model") {
    const model = data.model || (evt && evt.model);
    if (model) selectModelValue(model);
  } else if (action === "export") {
    downloadExport();
  }
}

function selectModelValue(model) {
  for (const opt of els.modelSelect.options) {
    if (opt.value === model) { opt.selected = true; return; }
  }
}

function downloadExport() {
  const a = document.createElement("a");
  a.href = "/api/export";
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ------------------------------------------------------------------ */
/* Переключение модели, очистка, прочие действия                        */
/* ------------------------------------------------------------------ */
async function changeModel(name) {
  if (!name) return;
  showHint("Загрузка модели…");
  setBusy(true);
  try {
    const r = await postJSON("/api/model", { name });
    showHint("");
    addMessage("system", r.message || ("Модель переключена: " + name));
    els.metaModel.textContent = "Модель: " + (r.current_model || name);
  } catch (e) {
    showHint("");
    addMessage("error", e.message || String(e));
  } finally {
    setBusy(false);
    await loadModels();
    await refreshStatus();
  }
}

async function clearChatServer() {
  try {
    const r = await postJSON("/api/clear", {});
    clearChat();
    addMessage("system", r.message || "История очищена.");
    await refreshStatus();
  } catch (e) {
    addMessage("error", e.message || String(e));
  }
}

function setBusy(value) {
  busy = value;
  els.send.disabled = value;
  els.clear.disabled = value;
  els.input.disabled = value;
}

/* ------------------------------------------------------------------ */
/* Инициализация и обработчики событий                                  */
/* ------------------------------------------------------------------ */
function wireEvents() {
  els.send.addEventListener("click", sendMessage);

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  els.clear.addEventListener("click", clearChatServer);
  els.export.addEventListener("click", downloadExport);
  els.help.addEventListener("click", () => {
    els.input.value = "/help";
    sendMessage();
  });

  els.modelSelect.addEventListener("change", (e) => changeModel(e.target.value));
}

async function init() {
  wireEvents();
  await refreshStatus();
  await loadModels();
  addMessage("system", "Готово. Введите сообщение или команду /help (кнопка «Справка»).");
  setInterval(refreshStatus, 10000);
  els.input.focus();
}

document.addEventListener("DOMContentLoaded", init);
