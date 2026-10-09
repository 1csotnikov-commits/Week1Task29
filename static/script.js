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
  // --- День 29: панель параметров ---
  panel: document.getElementById("panel"),
  panelToggle: document.getElementById("btn-panel"),
  profileLabel: document.getElementById("profile-label"),
  temperature: document.getElementById("p-temperature"),
  topP: document.getElementById("p-top-p"),
  topK: document.getElementById("p-top-k"),
  repeat: document.getElementById("p-repeat"),
  numPredict: document.getElementById("p-num-predict"),
  numCtx: document.getElementById("p-num-ctx"),
  seed: document.getElementById("p-seed"),
  vTemperature: document.getElementById("v-temperature"),
  vTopP: document.getElementById("v-top_p"),
  paramsReset: document.getElementById("btn-params-reset"),
  systemPrompt: document.getElementById("system-prompt"),
  systemSave: document.getElementById("btn-system-save"),
  systemReset: document.getElementById("btn-system-reset"),
  profileName: document.getElementById("profile-name"),
  profileSelect: document.getElementById("profile-select"),
  profileSave: document.getElementById("btn-profile-save"),
  profileLoad: document.getElementById("btn-profile-load"),
  profileDelete: document.getElementById("btn-profile-delete"),
};

let liveBubble = null; // «живой» пузырь ассистента во время стриминга
let busy = false;       // идёт ли сейчас запрос
let currentParams = {}; // текущие параметры генерации с бэкенда

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
/* День 29: параметры генерации                                         */
/* ------------------------------------------------------------------ */
function setInputValue(el, value) {
  if (!el) return;
  el.value = (value === null || value === undefined) ? "" : value;
}

function updateParamLabels() {
  if (els.vTemperature) els.vTemperature.textContent = els.temperature.value;
  if (els.vTopP) els.vTopP.textContent = els.topP.value;
}

function fillParams(params, profileName) {
  params = params || {};
  currentParams = params;
  setInputValue(els.temperature, params.temperature);
  setInputValue(els.topP, params.top_p);
  setInputValue(els.topK, params.top_k);
  setInputValue(els.repeat, params.repeat_penalty);
  setInputValue(els.numPredict, params.num_predict);
  setInputValue(els.numCtx, params.num_ctx);
  setInputValue(els.seed, params.seed);
  updateParamLabels();
  if (profileName !== undefined) {
    els.profileLabel.textContent = profileName ? "· профиль: " + profileName : "";
  }
}

function gatherParams() {
  return {
    temperature: parseFloat(els.temperature.value),
    top_p: parseFloat(els.topP.value),
    top_k: parseInt(els.topK.value, 10),
    repeat_penalty: parseFloat(els.repeat.value),
    num_predict: parseInt(els.numPredict.value, 10),
    num_ctx: parseInt(els.numCtx.value, 10),
    seed: els.seed.value === "" ? null : parseInt(els.seed.value, 10),
  };
}

async function loadParams() {
  try {
    const data = await getJSON("/api/params");
    fillParams(data.params, data.profile);
  } catch (e) {
    addMessage("error", "Не удалось загрузить параметры: " + e.message);
  }
}

async function applyParams() {
  try {
    const data = await postJSON("/api/params", gatherParams());
    fillParams(data.params, undefined);
    addMessage("system", "Параметры применены: temperature=" + data.params.temperature +
      ", num_predict=" + data.params.num_predict + ", num_ctx=" + data.params.num_ctx + ".");
  } catch (e) {
    addMessage("error", "Ошибка применения параметров: " + e.message);
  }
}

async function resetParams() {
  try {
    const r = await postJSON("/api/chat", { message: "/params reset", stream: false });
    if (r.data && r.data.params) fillParams(r.data.params, undefined);
    addMessage("system", "Параметры сброшены к дефолтам профиля.");
  } catch (e) {
    addMessage("error", "Ошибка сброса параметров: " + e.message);
  }
}

/* ------------------------------------------------------------------ */
/* День 29: системный промпт и профили                                  */
/* ------------------------------------------------------------------ */
async function loadSystemPrompt() {
  try {
    const data = await getJSON("/api/system");
    els.systemPrompt.value = data.system_prompt || "";
  } catch (e) { /* не критично */ }
}

async function saveSystemPrompt() {
  try {
    const r = await postJSON("/api/system", { prompt: els.systemPrompt.value });
    addMessage("system", "Системный промпт сохранён." + (r.hint ? "\n\n" + r.hint : ""));
  } catch (e) {
    addMessage("error", "Ошибка сохранения промпта: " + e.message);
  }
}

async function resetSystemPrompt() {
  try {
    const r = await postJSON("/api/chat", { message: "/system reset", stream: false });
    if (r.data && r.data.system !== undefined) els.systemPrompt.value = r.data.system;
    addMessage("system", r.text || "Системный промпт сброшен к дефолту профиля.");
  } catch (e) {
    addMessage("error", "Ошибка сброса промпта: " + e.message);
  }
}

async function loadProfiles() {
  try {
    const data = await getJSON("/api/profiles");
    els.profileSelect.innerHTML = "";
    (data.profiles || []).forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name + (p.builtin ? " (встроенный)" : "") + (p.model ? " · " + p.model : "");
      els.profileSelect.appendChild(opt);
    });
  } catch (e) { /* не критично */ }
}

async function saveProfile() {
  const name = (els.profileName.value || "").trim();
  if (!name) { addMessage("error", "Укажите имя профиля."); return; }
  try {
    const r = await postJSON("/api/profiles", {
      name,
      params: gatherParams(),
      system: els.systemPrompt.value,
    });
    addMessage("system", r.message || ("Профиль «" + name + "» сохранён."));
    await loadProfiles();
    els.profileSelect.value = name;
    els.profileName.value = "";
  } catch (e) {
    addMessage("error", e.message);
  }
}

async function loadProfile() {
  const name = els.profileSelect.value;
  if (!name) return;
  try {
    const r = await postJSON("/api/profiles/load", { name });
    if (r.params) fillParams(r.params, name);
    if (r.system_prompt !== undefined) els.systemPrompt.value = r.system_prompt;
    if (r.model) selectModelValue(r.model);
    addMessage("system", (r.message || ("Профиль «" + name + "» загружен.")) +
      (r.hint ? "\n\n" + r.hint : ""));
    await loadModels();
    await refreshStatus();
  } catch (e) {
    addMessage("error", e.message);
  }
}

async function deleteProfile() {
  const name = els.profileSelect.value;
  if (!name) return;
  try {
    const r = await postJSON("/api/profiles/delete", { name });
    addMessage("system", r.message || ("Профиль «" + name + "» удалён."));
    await loadProfiles();
  } catch (e) {
    addMessage("error", e.message);
  }
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
      body: JSON.stringify({ message: text, stream: true, params: gatherParams() }),
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
  } else if (action === "params") {
    if (data.params) fillParams(data.params, undefined);
  } else if (action === "profile") {
    if (data.params) fillParams(data.params, data.profile);
    if (data.system !== undefined) els.systemPrompt.value = data.system;
    if (data.model) selectModelValue(data.model);
  } else if (action === "system") {
    if (data.system !== undefined) els.systemPrompt.value = data.system;
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

  // --- День 29: панель параметров, промпт, профили ---
  if (els.panelToggle) {
    els.panelToggle.addEventListener("click", () => els.panel.classList.toggle("hidden"));
  }
  els.temperature.addEventListener("input", updateParamLabels);
  els.topP.addEventListener("input", updateParamLabels);
  els.temperature.addEventListener("change", applyParams);
  els.topP.addEventListener("change", applyParams);
  [els.topK, els.repeat, els.numPredict, els.numCtx, els.seed].forEach((el) => {
    el.addEventListener("change", applyParams);
  });
  els.paramsReset.addEventListener("click", resetParams);
  els.systemSave.addEventListener("click", saveSystemPrompt);
  els.systemReset.addEventListener("click", resetSystemPrompt);
  els.profileSave.addEventListener("click", saveProfile);
  els.profileLoad.addEventListener("click", loadProfile);
  els.profileDelete.addEventListener("click", deleteProfile);
}

async function init() {
  wireEvents();
  await refreshStatus();
  await loadModels();
  await loadParams();
  await loadProfiles();
  await loadSystemPrompt();
  addMessage("system", "Готово. Введите сообщение или команду /help. Параметры генерации — на панели справа.");
  setInterval(refreshStatus, 10000);
  els.input.focus();
}

document.addEventListener("DOMContentLoaded", init);
