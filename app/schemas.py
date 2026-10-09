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
    """Параметры генерации (День 29: полный набор).

    Используется и в POST /api/chat (разовое переопределение), и в
    POST /api/params (обновление текущих параметров).
    """

    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=1000)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    num_predict: Optional[int] = Field(default=None, ge=-1, le=32768)
    num_ctx: Optional[int] = Field(default=None, ge=512, le=131072)
    seed: Optional[int] = Field(default=None)
    keep_alive: Optional[str] = Field(default=None)


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
    hint: Optional[str] = None


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


# ---------------------------------------------------------------------------
# День 29: параметры генерации и профили
# ---------------------------------------------------------------------------
class ParamsResponse(BaseModel):
    params: dict[str, Any]
    defaults: dict[str, Any]
    profile: str


class ProfileInfo(BaseModel):
    name: str
    description: str = ""
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    builtin: bool = False
    created_at: Optional[str] = None


class ProfilesResponse(BaseModel):
    profiles: list[ProfileInfo]
    count: int


class ProfileSaveRequest(BaseModel):
    """Сохранение текущего состояния как профиля."""

    name: str = Field(..., min_length=1)
    description: str = ""
    model: Optional[str] = None
    system: Optional[str] = None
    params: Optional[GenerationOverrides] = None


class ProfileNameRequest(BaseModel):
    """Загрузка/удаление профиля по имени."""

    name: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    """Ручной запуск сравнения из UI (опционально)."""

    profile: Optional[str] = None
    configs: list[str] = Field(default_factory=lambda: ["baseline", "optimized", "qat"])
    seed: int = 42
    judge: bool = True
