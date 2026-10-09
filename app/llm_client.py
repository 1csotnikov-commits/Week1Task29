# -*- coding: utf-8 -*-
"""
Клиент локальной LLM (Ollama).

Использует два интерфейса Ollama:
  * OpenAI-совместимый  http://localhost:11434/v1  — через библиотеку `openai`
    (чат, стриминг, параметры генерации);
  * нативный            http://localhost:11434/api — через `httpx`
    (список моделей, выгрузка из VRAM, статус, версия).

Все ошибки преобразуются в исключения с понятным русским текстом (без
трейсбэка) — их безопасно показывать пользователю. Полный трейсбэк пишется
в лог на уровне ERROR.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx2 as httpx  # httpx2 — HTTP-клиент, на который опирается openai 3.x (API совместим с httpx)
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
)

from . import config as cfg
from .config import GenerationParams

log = logging.getLogger("app.llm")


# ---------------------------------------------------------------------------
# Исключения с русскими сообщениями
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Базовая ошибка LLM. `message` — безопасный для показа текст."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class OllamaUnavailableError(LLMError):
    """Сервер Ollama недоступен (не запущен / connection refused)."""


class ModelNotFoundError(LLMError):
    """Запрошенная модель отсутствует в Ollama."""

    def __init__(self, model: str, available: list[str] | None = None) -> None:
        self.model = model
        self.available = available or []
        text = f"Модель {model} не найдена."
        if self.available:
            text += " Доступные: " + ", ".join(self.available)
        super().__init__(text)


class LLMTimeoutError(LLMError):
    """Превышено время ожидания ответа."""


class ContextLengthError(LLMError):
    """Превышен размер контекста модели."""


# Маркеры для распознавания текстовых ошибок Ollama
_CONTEXT_MARKERS = ("exceeds", "context size", "num_ctx", "context length", "too long")
_NOT_FOUND_MARKERS = ("not found", "no such model", "model not found", "pull model")
# Ошибки нехватки памяти при слишком большом num_ctx / модели.
_MEMORY_MARKERS = (
    "requires more", "insufficient memory", "out of memory", "not enough memory",
    "failed to allocate", "out of vram", "not enough vram",
)
_MEMORY_MESSAGE = (
    "Недостаточно памяти/VRAM для запрошенного размера контекста (num_ctx слишком велик). "
    "Уменьшите num_ctx на панели «Параметры» или выгрузите другие модели командой /unload."
)


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


# ---------------------------------------------------------------------------
# Клиент
# ---------------------------------------------------------------------------
class OllamaClient:
    """Асинхронный клиент к локальному серверу Ollama."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or cfg.OLLAMA_BASE_URL).rstrip("/")
        self.openai_base_url = self.base_url + "/v1"

        # Клиент OpenAI SDK: base_url указывает на локальный Ollama-совместимый API.
        # api_key для Ollama не требуется, но SDK его требует — подставляем заглушку.
        self._client = AsyncOpenAI(
            base_url=self.openai_base_url,
            api_key="ollama",
            timeout=httpx.Timeout(cfg.REQUEST_TIMEOUT, connect=cfg.CONNECT_TIMEOUT),
            max_retries=1,
        )
        # Клиент для нативных вызовов Ollama (/api/*).
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(cfg.REQUEST_TIMEOUT, connect=cfg.CONNECT_TIMEOUT),
        )

    async def aclose(self) -> None:
        """Закрывает соединения (вызывается при остановке приложения)."""
        await self._client.close()
        await self._http.aclose()

    # -- доступность и статус ---------------------------------------------
    async def is_available(self) -> bool:
        """Проверяет, отвечает ли сервер Ollama."""
        try:
            resp = await self._http.get("/api/version", timeout=cfg.CONNECT_TIMEOUT)
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def version(self) -> str | None:
        """Версия сервера Ollama (None, если недоступен)."""
        try:
            resp = await self._http.get("/api/version", timeout=cfg.CONNECT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("version")
        except (httpx.RequestError, ValueError):
            pass
        return None

    # -- список моделей ----------------------------------------------------
    async def list_models_raw(self) -> list[dict[str, Any]]:
        """Сырой список моделей из /api/tags."""
        try:
            resp = await self._http.get("/api/tags", timeout=cfg.LIST_TIMEOUT)
        except httpx.RequestError as exc:
            log.error("Ollama недоступна при запросе списка моделей: %s", exc)
            raise OllamaUnavailableError(cfg.OLLAMA_START_HINT, detail=str(exc)) from exc

        if resp.status_code != 200:
            log.error("Ollama вернула код %s при запросе /api/tags", resp.status_code)
            raise LLMError("Не удалось получить список моделей Ollama.", detail=resp.text)
        return resp.json().get("models", []) or []

    async def list_models(self) -> list[str]:
        """Список имён установленных моделей."""
        return [m.get("name", "") for m in await self.list_models_raw() if m.get("name")]

    async def ps(self) -> list[dict[str, Any]]:
        """Список моделей, загруженных сейчас в память (/api/ps)."""
        try:
            resp = await self._http.get("/api/ps", timeout=cfg.LIST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("models", []) or []
        except (httpx.RequestError, ValueError):
            pass
        return []

    async def list_running(self) -> list[str]:
        """Имена моделей, загруженных в память сейчас (для метрик ресурсов)."""
        return [m.get("name", "") for m in await self.ps() if m.get("name")]

    async def unload_all(self) -> list[str]:
        """Выгружает из VRAM все загруженные модели. Возвращает имена выгруженных.

        Используется перед замерами ресурсов, чтобы стартовать с «холодной» VRAM.
        """
        unloaded: list[str] = []
        for name in await self.list_running():
            try:
                await self.unload_model(name)
                unloaded.append(name)
            except LLMError as exc:
                log.warning("Не удалось выгрузить модель %s: %s", name, exc.message)
        return unloaded

    async def ensure_model(self, model: str) -> None:
        """Проверяет, что модель установлена; иначе — ModelNotFoundError."""
        available = await self.list_models()
        if model not in available:
            raise ModelNotFoundError(model, available)

    # -- выгрузка модели из VRAM ------------------------------------------
    async def unload_model(self, model: str) -> None:
        """Освобождает VRAM: POST /api/generate {model, keep_alive: 0}.

        Модель остаётся выбранной — при следующем запросе Ollama загрузит её снова.
        """
        try:
            resp = await self._http.post(
                "/api/generate",
                json={"model": model, "keep_alive": 0, "prompt": ""},
                timeout=cfg.UNLOAD_TIMEOUT,
            )
        except httpx.RequestError as exc:
            log.error("Ollama недоступна при выгрузке модели %s: %s", model, exc)
            raise OllamaUnavailableError(cfg.OLLAMA_START_HINT, detail=str(exc)) from exc

        if resp.status_code >= 400:
            log.error("Не удалось выгрузить модель %s: %s", model, resp.text)
            raise LLMError(f"Не удалось выгрузить модель {model} из памяти.", detail=resp.text)
        log.info("Модель %s выгружена из VRAM (keep_alive=0).", model)

    async def generate_raw(
        self,
        messages: list[dict[str, str]],
        model: str,
        params: GenerationParams | None = None,
    ) -> dict[str, Any]:
        """Нативный вызов /api/chat (stream=false) с полными метриками.

        Возвращает сырой ответ Ollama: текст ответа, счётчики токенов и
        тайминги (eval_count, eval_duration, load_duration, prompt_eval_*).
        Используется инструментами сравнения и сбора метрик (tools/*).
        """
        params = params or cfg.DEFAULT_GENERATION
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": params.to_ollama_options(),
            "keep_alive": params.keep_alive,
        }
        try:
            resp = await self._http.post("/api/chat", json=payload)
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(cfg.OLLAMA_START_HINT, detail=str(exc)) from exc

        if resp.status_code >= 400:
            text = resp.text
            if resp.status_code == 404 or _contains(text, _NOT_FOUND_MARKERS):
                raise ModelNotFoundError(model)
            if _contains(text, _MEMORY_MARKERS):
                raise ContextLengthError(_MEMORY_MESSAGE, detail=text)
            if _contains(text, _CONTEXT_MARKERS):
                raise ContextLengthError(
                    "Превышен размер контекста модели (num_ctx). Увеличьте num_ctx "
                    "(например, до 16384) или уменьшите объём запроса.",
                    detail=text,
                )
            raise LLMError(f"Ошибка Ollama при генерации ({model}): {text}", detail=text)
        return resp.json()

    # -- генерация ---------------------------------------------------------
    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        model: str,
        params: GenerationParams | None,
    ) -> dict[str, Any]:
        """Собирает аргументы для openai.chat.completions.create.

        num_ctx и keep_alive специфичны для Ollama и передаются через
        extra_body.options (проверено: OpenAI-эндпоинт Ollama их принимает).
        """
        params = params or cfg.DEFAULT_GENERATION
        # Все параметры Ollama передаём единым блоком options (top_k,
        # repeat_penalty, seed не входят в стандарт OpenAI API).
        options = params.to_ollama_options()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "extra_body": {"options": options, "keep_alive": params.keep_alive},
        }
        # num_predict > 0 ограничивает длину ответа; иначе — без ограничения.
        if params.num_predict and params.num_predict > 0:
            kwargs["max_tokens"] = params.num_predict
        return kwargs

    def _map_error(self, exc: Exception, model: str) -> LLMError:
        """Преобразует исключение SDK в понятную ошибку на русском."""
        text = str(exc)

        if isinstance(exc, APIConnectionError):
            return OllamaUnavailableError(cfg.OLLAMA_START_HINT, detail=text)
        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError(
                "Запрос превысил время ожидания. Модель не успела ответить — "
                "попробуйте ещё раз или сократите запрос.",
                detail=text,
            )
        if isinstance(exc, NotFoundError):
            return ModelNotFoundError(model)
        # Нехватка памяти (слишком большой num_ctx) — раньше остальных текстовых проверок.
        if _contains(text, _MEMORY_MARKERS):
            return ContextLengthError(_MEMORY_MESSAGE, detail=text)
        if isinstance(exc, BadRequestError):
            if _contains(text, _CONTEXT_MARKERS):
                return ContextLengthError(
                    "Превышен размер контекста модели (num_ctx). Увеличьте параметр "
                    "num_ctx (по умолчанию 4096 токенов) или очистите историю командой /clear.",
                    detail=text,
                )
            if _contains(text, _NOT_FOUND_MARKERS):
                return ModelNotFoundError(model)
        if isinstance(exc, APIStatusError) and exc.status_code == 404:
            return ModelNotFoundError(model)
        if _contains(text, _CONTEXT_MARKERS):
            return ContextLengthError(
                "Превышен размер контекста модели (num_ctx). Увеличьте параметр num_ctx "
                "или очистите историю командой /clear.",
                detail=text,
            )
        return LLMError(f"Ошибка при обращении к модели {model}: {text}", detail=text)

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        params: GenerationParams | None = None,
    ) -> str:
        """Обычный (нестриминговый) ответ модели."""
        kwargs = self._build_kwargs(messages, model, params)
        try:
            resp = await self._client.chat.completions.create(stream=False, **kwargs)
        except Exception as exc:  # noqa: BLE001 — преобразуем в доменную ошибку
            mapped = self._map_error(exc, model)
            log.error("Ошибка chat (model=%s): %s", model, mapped.message, exc_info=exc)
            raise mapped from exc
        return (resp.choices[0].message.content or "") if resp.choices else ""

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        params: GenerationParams | None = None,
    ) -> AsyncIterator[str]:
        """Поток токенов (SSE) от модели."""
        kwargs = self._build_kwargs(messages, model, params)
        try:
            stream = await self._client.chat.completions.create(stream=True, **kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    yield piece
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = self._map_error(exc, model)
            log.error("Ошибка chat_stream (model=%s): %s", model, mapped.message, exc_info=exc)
            raise mapped from exc
