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

# День 29: профили, отчёты и набор тестовых запросов
PROFILES_DIR: Path = BASE_DIR / "profiles"          # сохранённые JSON-профили
REPORTS_DIR: Path = BASE_DIR / "reports"            # отчёты сравнения «до/после»
MODELFILES_DIR: Path = BASE_DIR / "modelfiles"      # сгенерированные Modelfile
EVAL_QUERIES_FILE: Path = BASE_DIR / "evaluation_queries.json"

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
# Целевая задача оптимизации (День 29)
# ---------------------------------------------------------------------------
# Профиль целевой задачи можно менять без правки кода (TASK_PROFILE / env).
TASK_PROFILE: str = os.getenv("TASK_PROFILE", "1c_support")

# ---------------------------------------------------------------------------
# Модель и промпт по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")

# Системный промпт Дня 27 (используется как «baseline» в сравнении).
DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "Ты — полезный ассистент. Отвечай на русском языке, кратко и по делу.",
)

# Улучшенный системный промпт под задачу «ассистент техподдержки 1С».
SYSTEM_PROMPT_1C_SUPPORT: str = (
    "Ты — ассистент технической поддержки по платформе «1С:Предприятие 8.3» "
    "и встроенному языку 1С.\n\n"
    "Правила:\n"
    "— Отвечай на русском языке, по делу, без воды и вступлений.\n"
    "— Сначала дай короткий прямой ответ (1–3 предложения), затем детали.\n"
    "— Код на языке 1С оформляй блоком ```bsl и комментируй ключевые строки.\n"
    "— Для типовых ошибок указывай вероятные причины и как проверить.\n"
    "— Если данных не хватает — задай уточняющий вопрос, не домысливай.\n"
    "— Не выдумывай методы и свойства платформы; если не уверен — скажи об этом.\n\n"
    "Структура ответа: 1) краткий ответ; 2) пояснение/шаги; 3) пример кода (если уместно)."
)

# ---------------------------------------------------------------------------
# Параметры генерации (День 29: расширены top_k / repeat_penalty / seed)
# ---------------------------------------------------------------------------
@dataclass
class GenerationParams:
    """Параметры генерации, передаваемые в Ollama через блок options.

      * temperature     — «креативность» (0.0–1.5);
      * top_p           — nucleus sampling;
      * top_k           — отсечение по k лучших токенов;
      * repeat_penalty  — штраф за повторы;
      * num_predict     — максимум генерируемых токенов (-1 = без ограничения);
      * num_ctx         — размер контекстного окна (дефолт Ollama = 4096);
      * seed            — зерно генерации (для воспроизводимости сравнений);
      * keep_alive      — сколько держать модель в VRAM ("5m", 0 — выгрузить).
    """

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 512
    num_ctx: int = 8192
    seed: int | None = 42
    keep_alive: str | int = "5m"

    # Обратная совместимость с Днём 27 (там поле называлось max_tokens).
    @property
    def max_tokens(self) -> int:
        return self.num_predict

    def to_ollama_options(self) -> dict:
        """Приводит параметры к формату блока options нативного API Ollama."""
        options: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def to_dict(self) -> dict:
        """Сериализация (для API и JSON-профилей)."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
            "keep_alive": self.keep_alive,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GenerationParams":
        """Создаёт параметры из словаря, игнорируя неизвестные ключи."""
        data = data or {}
        known = {
            "temperature", "top_p", "top_k", "repeat_penalty",
            "num_predict", "num_ctx", "seed", "keep_alive",
        }
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})


# Значения по умолчанию (экземпляр — можно менять в рантайме).
DEFAULT_GENERATION: GenerationParams = GenerationParams()


# ---------------------------------------------------------------------------
# Определения профилей целевой задачи
# ---------------------------------------------------------------------------
@dataclass
class TaskProfileDef:
    """Профиль целевой задачи: промпт, параметры, базовые модели."""

    name: str
    description: str
    system_prompt: str
    params: GenerationParams
    baseline_model: str          # модель «до» (День 27)
    optimized_model: str         # рекомендованная модель «после»
    quantization_model: str      # вариант для сравнения квантования


TASK_PROFILES: dict[str, TaskProfileDef] = {
    "1c_support": TaskProfileDef(
        name="1c_support",
        description="Ассистент технической поддержки 1С:Предприятие 8.3",
        system_prompt=SYSTEM_PROMPT_1C_SUPPORT,
        # Параметры оптимизированного профиля: ниже температура (факты, а не
        # фантазии), ограниченная длина ответа, расширенный контекст.
        params=GenerationParams(
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=512,
            num_ctx=8192,
            seed=42,
        ),
        baseline_model="gpt-oss:20b",
        optimized_model="gemma3:12b",
        quantization_model="gemma3:12b-it-qat",
    ),
}


def get_task_profile(name: str | None = None) -> TaskProfileDef:
    """Возвращает профиль целевой задачи по имени (или профиль по умолчанию)."""
    key = name or TASK_PROFILE
    return TASK_PROFILES.get(key, TASK_PROFILES["1c_support"])

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
