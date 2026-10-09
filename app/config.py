# -*- coding: utf-8 -*-
"""
Конфигурация приложения — единая точка изменения настроек.

Здесь собраны все «ручки», которые понадобятся на Днях 27/29/30:
  * День 27 — модель по умолчанию, системный промпт, адрес Ollama, хост/порт;
  * День 29 — параметры генерации (temperature, max_tokens, num_ctx, top_p) —
    уже принимаются клиентом LLM, останется добавить элементы управления в UI;
  * День 30 — CORS (пока ["*"]), место под аутентификацию и rate limit.

Настройки можно переопределять переменными окружения, не меняя код.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
STATIC_DIR: Path = BASE_DIR / "static"
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "app.log"
HISTORY_FILE: Path = LOG_DIR / "history.json"  # опциональное сохранение истории

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
# Базовый URL сервера Ollama. Переопределяется переменной OLLAMA_BASE_URL.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# OpenAI-совместимый префикс (используется библиотекой openai).
OLLAMA_OPENAI_BASE_URL: str = OLLAMA_BASE_URL + "/v1"

# Подсказка пользователю, если сервер не запущен.
OLLAMA_START_HINT: str = r"Локальный сервер Ollama не запущен. Запустите C:\LLM\Ollama\ollama.exe serve"

# ---------------------------------------------------------------------------
# Приложение (хост/порт веб-сервера)
# ---------------------------------------------------------------------------
APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

# ---------------------------------------------------------------------------
# Модель и промпт по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")

DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "Ты — полезный ассистент. Отвечай на русском языке, кратко и по делу.",
)

# ---------------------------------------------------------------------------
# Параметры генерации по умолчанию (задействуются в UI на Дне 29)
# ---------------------------------------------------------------------------
@dataclass
class GenerationParams:
    """Параметры генерации, передаваемые в Ollama.

    На Дне 27 в интерфейсе их нет, но клиент LLM уже умеет их принимать —
    на Дне 29 останется только добавить поля ввода и проброс значений.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    # num_predict = максимальное число генерируемых токенов (-1 = без ограничения)
    max_tokens: int = -1
    # Размер контекста. По умолчанию у Ollama 4096; поднимается на Дне 29/30.
    num_ctx: int = 4096
    # keep_alive: сколько держать модель в VRAM после запроса ("5m", 0 — выгрузить).
    keep_alive: str | int = "5m"

    def to_ollama_options(self) -> dict:
        """Приводит параметры к формату блока options нативного API Ollama."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }


# Значения по умолчанию (экземпляр — чтобы можно было менять в рантайме на Дне 29).
DEFAULT_GENERATION: GenerationParams = GenerationParams()

# ---------------------------------------------------------------------------
# Таймауты (секунды)
# ---------------------------------------------------------------------------
CONNECT_TIMEOUT: float = 5.0      # проверка доступности Ollama
REQUEST_TIMEOUT: float = 600.0    # генерация ответа (долгие «думающие» модели)
LIST_TIMEOUT: float = 10.0        # получение списка моделей
UNLOAD_TIMEOUT: float = 30.0      # выгрузка модели из памяти

# ---------------------------------------------------------------------------
# CORS — на Дне 27 открыт всем; на Дне 30 заменить на конкретные источники.
# ---------------------------------------------------------------------------
CORS_ALLOW_ORIGINS: list[str] = ["*"]

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.DEBUG if os.getenv("APP_DEBUG") else logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> logging.Logger:
    """Настраивает логирование в файл (logs/app.log) и в консоль.

    Идемпотентна: повторный вызов не добавляет дублирующие обработчики.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("app")
    root.setLevel(LOG_LEVEL)
    root.propagate = False

    if root.handlers:  # уже настроено
        return root

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return root
