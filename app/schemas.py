# -*- coding: utf-8 -*-
"""
Pydantic-модели запросов и ответов API.

Вынесены отдельно, чтобы структуру эндпоинтов было легко расширять
(Дни 29/30: параметры генерации, аутентификация, лимиты) без правки логики.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Запросы
# ---------------------------------------------------------------------------
class GenerationOverrides(BaseModel):
    """Необязательные переопределения параметров генерации.

    На Дне 27 UI их не отправляет, но эндпоинт уже принимает —
    на Дне 29 останется только добавить элементы управления на фронте.
    """

    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=-1)
    num_ctx: Optional[int] = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    """Запрос в чат. Если `message` начинается с '/', это команда."""

    message: str = Field(..., min_length=1, description="Текст сообщения или команда")
    stream: bool = Field(default=True, description="SSE-стриминг токенов")
    model: Optional[str] = Field(default=None, description="Переопределить модель разово")
    params: Optional[GenerationOverrides] = Field(default=None)


class ModelRequest(BaseModel):
    """Переключение модели."""

    name: str = Field(..., min_length=1, description="Имя модели Ollama")


class SystemPromptRequest(BaseModel):
    """Установка системного промпта."""

    prompt: str = Field(default="", description="Новый системный промпт")


# ---------------------------------------------------------------------------
# Ответы
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None
    size_human: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    current: bool = False
    loaded: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    current: str
    count: int


class StatusResponse(BaseModel):
    ollama_available: bool
    current_model: str
    history_length: int
    system_prompt: str
    uptime_seconds: float
    ollama_version: Optional[str] = None
    error: Optional[str] = None


class SystemPromptResponse(BaseModel):
    system_prompt: str


class SimpleResponse(BaseModel):
    """Универсальный ответ для команд переключения/очистки."""

    ok: bool = True
    message: str
    current_model: Optional[str] = None
    history_length: Optional[int] = None


class HelpCommandItem(BaseModel):
    name: str
    args: str = ""
    description: str


class HelpResponse(BaseModel):
    commands: list[HelpCommandItem]
    count: int


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None
