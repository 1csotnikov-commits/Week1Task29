# -*- coding: utf-8 -*-
"""
FastAPI-приложение «Локальный LLM-чат» (День 27).

Роуты:
  GET  /                 — веб-интерфейс (static/index.html)
  GET  /api/status       — статус: доступность Ollama, модель, история, аптайм
  GET  /api/models       — список моделей Ollama + текущая
  POST /api/model        — переключить модель (старая выгружается из VRAM)
  POST /api/unload       — выгрузить текущую модель из памяти
  GET  /api/system       — текущий системный промпт
  POST /api/system       — задать системный промпт
  POST /api/chat         — чат (SSE-стриминг или JSON), здесь же команды
  POST /api/clear        — очистить историю
  POST /api/reset        — полный сброс к дефолтам
  GET  /api/export       — скачать историю в JSON
  GET  /api/help         — справка по командам (из реестра)
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import commands as cmd
from . import config as cfg
from .commands import _human_size
from .config import GenerationParams
from .llm_client import LLMError, ModelNotFoundError, OllamaClient
from .schemas import (
    ChatRequest,
    HelpCommandItem,
    HelpResponse,
    ModelInfo,
    ModelRequest,
    ModelsResponse,
    SimpleResponse,
    StatusResponse,
    SystemPromptRequest,
    SystemPromptResponse,
)
from .session import Session

log = logging.getLogger("app.main")


# ---------------------------------------------------------------------------
# Жизненный цикл приложения
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg.setup_logging()
    log.info("Запуск приложения. Ollama: %s | модель по умолчанию: %s",
             cfg.OLLAMA_BASE_URL, cfg.DEFAULT_MODEL)

    app.state.client = OllamaClient()
    app.state.session = Session()
    app.state.started_at = time.time()

    if await app.state.client.is_available():
        log.info("Ollama доступна. Версия: %s", await app.state.client.version())
    else:
        log.warning("Ollama недоступна на старте: %s", cfg.OLLAMA_START_HINT)

    try:
        yield
    finally:
        await app.state.client.aclose()
        log.info("Приложение остановлено.")


app = FastAPI(title="Локальный LLM-чат (День 27)", version="1.0.0", lifespan=lifespan)

# CORS: на Дне 27 открыт всем. На Дне 30 заменить список на конкретные источники.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика (CSS/JS).
app.mount("/static", StaticFiles(directory=str(cfg.STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Не пропускает трейсбэк на фронт: логируем, отдаём понятный русский текст."""
    log.error("Необработанная ошибка на %s: %s", request.url.path, exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Внутренняя ошибка сервера. Подробности — в logs/app.log.",
        },
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def get_session(request: Request) -> Session:
    return request.app.state.session


def get_client(request: Request) -> OllamaClient:
    return request.app.state.client


def sse(payload: dict[str, Any]) -> str:
    """Форматирует событие SSE."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def error_json(message: str, detail: str | None = None, status_code: int = 400) -> JSONResponse:
    """Единый формат ошибки API (без трейсбэка)."""
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": message, "detail": detail},
    )


async def friendly_error(exc: LLMError, client: OllamaClient) -> str:
    """Обогащает ошибку понятным русским текстом (в т.ч. список моделей)."""
    if isinstance(exc, ModelNotFoundError) and not exc.available:
        try:
            available = await client.list_models()
        except LLMError:
            available = []
        if available:
            return ModelNotFoundError(exc.model, available).message
    return exc.message


async def run_command(request: Request, text: str) -> cmd.CommandResult:
    """Выполняет команду через общий реестр."""
    ctx = cmd.CommandContext(session=get_session(request), client=get_client(request))
    return await cmd.dispatch(text, ctx)


# ---------------------------------------------------------------------------
# Веб-интерфейс
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(cfg.STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Статус и модели
# ---------------------------------------------------------------------------
@app.get("/api/status", response_model=StatusResponse)
async def api_status(request: Request) -> StatusResponse:
    session = get_session(request)
    client = get_client(request)
    available = await client.is_available()
    version = await client.version() if available else None
    return StatusResponse(
        ollama_available=available,
        current_model=session.model,
        history_length=session.history_length(),
        system_prompt=session.system_prompt,
        uptime_seconds=session.uptime(),
        ollama_version=version,
        error=None if available else cfg.OLLAMA_START_HINT,
    )


@app.get("/api/models", response_model=ModelsResponse)
async def api_models(request: Request) -> ModelsResponse:
    session = get_session(request)
    client = get_client(request)
    try:
        raw = await client.list_models_raw()
    except LLMError as exc:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"ok": False, "error": exc.message, "detail": exc.detail},
        )
    loaded = {m.get("name") for m in await client.ps()}
    models = []
    for m in raw:
        details = m.get("details", {}) or {}
        models.append(
            ModelInfo(
                name=m.get("name", ""),
                size=m.get("size"),
                size_human=_human_size(m.get("size")),
                parameter_size=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
                current=(m.get("name") == session.model),
                loaded=(m.get("name") in loaded),
            )
        )
    return ModelsResponse(models=models, current=session.model, count=len(models))


# ---------------------------------------------------------------------------
# Управление моделью
# ---------------------------------------------------------------------------
@app.post("/api/model", response_model=SimpleResponse)
async def api_set_model(request: Request, payload: ModelRequest):
    result = await run_command(request, f"/model {payload.name}")
    session = get_session(request)
    if not result.ok:
        return error_json(result.text)
    log.info("Модель переключена на %s", session.model)
    return SimpleResponse(
        ok=True,
        message=result.text,
        current_model=session.model,
        history_length=session.history_length(),
    )


@app.post("/api/unload", response_model=SimpleResponse)
async def api_unload(request: Request):
    result = await run_command(request, "/unload")
    session = get_session(request)
    if not result.ok:
        return error_json(result.text)
    return SimpleResponse(
        ok=True,
        message=result.text,
        current_model=session.model,
        history_length=session.history_length(),
    )


# ---------------------------------------------------------------------------
# Системный промпт
# ---------------------------------------------------------------------------
@app.get("/api/system", response_model=SystemPromptResponse)
async def api_get_system(request: Request) -> SystemPromptResponse:
    return SystemPromptResponse(system_prompt=get_session(request).system_prompt)


@app.post("/api/system", response_model=SystemPromptResponse)
async def api_set_system(request: Request, payload: SystemPromptRequest):
    session = get_session(request)
    session.system_prompt = payload.prompt
    log.info("Системный промпт обновлён (длина %s символов).", len(payload.prompt))
    return SystemPromptResponse(system_prompt=session.system_prompt)


# ---------------------------------------------------------------------------
# Очистка / сброс / экспорт / справка
# ---------------------------------------------------------------------------
@app.post("/api/clear", response_model=SimpleResponse)
async def api_clear(request: Request) -> SimpleResponse:
    result = await run_command(request, "/clear")
    session = get_session(request)
    return SimpleResponse(
        ok=True,
        message=result.text,
        current_model=session.model,
        history_length=session.history_length(),
    )


@app.post("/api/reset", response_model=SimpleResponse)
async def api_reset(request: Request) -> SimpleResponse:
    result = await run_command(request, "/reset")
    session = get_session(request)
    return SimpleResponse(
        ok=True,
        message=result.text,
        current_model=session.model,
        history_length=session.history_length(),
    )


@app.get("/api/export")
async def api_export(request: Request) -> Response:
    session = get_session(request)
    snapshot = session.snapshot()
    filename = "chat_export_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
    body = json.dumps(snapshot, ensure_ascii=False, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/help", response_model=HelpResponse)
async def api_help() -> HelpResponse:
    items = [
        HelpCommandItem(name=spec.name, args=spec.args, description=spec.description)
        for spec in cmd.visible_commands()
    ]
    return HelpResponse(commands=items, count=len(items))


# ---------------------------------------------------------------------------
# Чат
# ---------------------------------------------------------------------------
def _merge_params(overrides) -> GenerationParams:
    """Собирает параметры генерации: дефолты из конфига + переопределения.

    На Дне 27 UI переопределения не шлёт; на Дне 29 сюда придут значения
    temperature / max_tokens / num_ctx из интерфейса.
    """
    params = GenerationParams(
        temperature=cfg.DEFAULT_GENERATION.temperature,
        top_p=cfg.DEFAULT_GENERATION.top_p,
        max_tokens=cfg.DEFAULT_GENERATION.max_tokens,
        num_ctx=cfg.DEFAULT_GENERATION.num_ctx,
        keep_alive=cfg.DEFAULT_GENERATION.keep_alive,
    )
    if overrides is not None:
        if overrides.temperature is not None:
            params.temperature = overrides.temperature
        if overrides.top_p is not None:
            params.top_p = overrides.top_p
        if overrides.max_tokens is not None:
            params.max_tokens = overrides.max_tokens
        if overrides.num_ctx is not None:
            params.num_ctx = overrides.num_ctx
    return params


def _rollback_last_user(session: Session) -> None:
    """Удаляет последнее пользовательское сообщение (при ошибке запроса)."""
    if session.messages and session.messages[-1].role == "user":
        session.messages.pop()


async def _chat_sse(request: Request, payload: ChatRequest, params: GenerationParams):
    """Асинхронный генератор SSE-событий для чата."""
    session = get_session(request)
    client = get_client(request)
    message = payload.message

    # 1) Команда
    if cmd.is_command(message):
        result = await run_command(request, message)
        yield sse({"type": "token", "content": result.text})
        yield sse(
            {
                "type": "done",
                "ok": result.ok,
                "model": session.model,
                "history_length": session.history_length(),
                "data": result.data,
            }
        )
        return

    # 2) Обычное сообщение — идёт в LLM
    model = payload.model or session.model
    await session.add("user", message)
    try:
        collected: list[str] = []
        async for piece in client.chat_stream(session.to_llm_messages(), model, params):
            collected.append(piece)
            yield sse({"type": "token", "content": piece})
        answer = "".join(collected)
        await session.add("assistant", answer)
        yield sse(
            {
                "type": "done",
                "ok": True,
                "model": model,
                "history_length": session.history_length(),
                "data": {},
            }
        )
    except LLMError as exc:
        _rollback_last_user(session)
        friendly = await friendly_error(exc, client)
        log.error("Ошибка стриминга чата (model=%s): %s", model, friendly)
        yield sse({"type": "error", "message": friendly})


@app.post("/api/chat")
async def api_chat(request: Request, payload: ChatRequest):
    params = _merge_params(payload.params)

    if payload.stream:
        return StreamingResponse(
            _chat_sse(request, payload, params),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Нестриминговый режим
    session = get_session(request)
    client = get_client(request)
    message = payload.message

    if cmd.is_command(message):
        result = await run_command(request, message)
        return JSONResponse(
            {
                "ok": result.ok,
                "text": result.text,
                "model": session.model,
                "history_length": session.history_length(),
                "data": result.data,
            }
        )

    model = payload.model or session.model
    await session.add("user", message)
    try:
        answer = await client.chat(session.to_llm_messages(), model, params)
    except LLMError as exc:
        _rollback_last_user(session)
        friendly = await friendly_error(exc, client)
        log.error("Ошибка чата (model=%s): %s", model, friendly)
        return error_json(friendly, status_code=502)
    await session.add("assistant", answer)
    return JSONResponse(
        {
            "ok": True,
            "text": answer,
            "model": model,
            "history_length": session.history_length(),
        }
    )
