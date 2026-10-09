# -*- coding: utf-8 -*-
"""
Хранение состояния диалога.

На Дне 27 история хранится в памяти процесса (с опциональным сохранением
в JSON-файл). Класс Session инкапсулирует:
  * список сообщений (роли: user / assistant / system);
  * текущую модель;
  * текущий системный промпт;
  * время старта (для расчёта аптайма).

Асинхронная блокировка защищает состояние от гонок при параллельных запросах
(пригодится на Дне 30 при работе нескольких клиентов).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config as cfg
from .config import GenerationParams

VALID_ROLES = ("system", "user", "assistant")


@dataclass
class Message:
    """Одно сообщение диалога."""

    role: str
    content: str
    ts: float = field(default_factory=time.time)


class Session:
    """Состояние одного чата (в памяти)."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self.started_at: float = time.time()
        profile = cfg.get_task_profile()
        self.model: str = model or cfg.DEFAULT_MODEL
        self.system_prompt: str = (
            system_prompt if system_prompt is not None else profile.system_prompt
        )
        # Текущие параметры генерации (День 29). Стартуем со значений профиля задачи.
        self.params: GenerationParams = GenerationParams(**profile.params.to_dict())
        self.messages: list[Message] = []

    # -- работа с историей -------------------------------------------------
    async def add(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю."""
        if role not in VALID_ROLES:
            raise ValueError(f"Недопустимая роль сообщения: {role}")
        async with self._lock:
            self.messages.append(Message(role=role, content=content))

    def history_length(self) -> int:
        """Число содержательных сообщений (без системных)."""
        return sum(1 for m in self.messages if m.role != "system")

    async def clear(self) -> None:
        """Очищает историю диалога (модель и промпт сохраняются)."""
        async with self._lock:
            self.messages.clear()

    async def reset(self) -> None:
        """Полный сброс к значениям по умолчанию (модель, промпт, параметры)."""
        profile = cfg.get_task_profile()
        async with self._lock:
            self.messages.clear()
            self.model = cfg.DEFAULT_MODEL
            self.system_prompt = profile.system_prompt
            self.params = GenerationParams(**profile.params.to_dict())

    # -- параметры генерации и профили (День 29) --------------------------
    def update_params(self, values: dict) -> GenerationParams:
        """Частично обновляет параметры генерации."""
        for key, value in (values or {}).items():
            if hasattr(self.params, key) and value is not None:
                setattr(self.params, key, value)
        return self.params

    def apply_profile(self, profile: dict) -> None:
        """Применяет профиль (модель + системный промпт + параметры) к сессии."""
        if profile.get("model"):
            self.model = profile["model"]
        if profile.get("system_prompt") is not None:
            self.system_prompt = profile["system_prompt"]
        if profile.get("params"):
            self.params = GenerationParams.from_dict(profile["params"])

    # -- формирование запроса к LLM ---------------------------------------
    def to_llm_messages(self) -> list[dict[str, str]]:
        """История в формате messages, ожидаемом API (/v1/chat/completions).

        Системный промпт добавляется первым сообщением, если он не пустой.
        """
        result: list[dict[str, str]] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for m in self.messages:
            result.append({"role": m.role, "content": m.content})
        return result

    # -- сериализация / экспорт -------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Снимок состояния для экспорта в JSON."""
        return {
            "exported_at": time.time(),
            "exported_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "model": self.model,
            "system_prompt": self.system_prompt,
            "params": self.params.to_dict(),
            "messages_count": self.history_length(),
            "messages": [asdict(m) for m in self.messages],
        }

    def uptime(self) -> float:
        """Аптайм приложения в секундах."""
        return round(time.time() - self.started_at, 1)

    # -- опциональное сохранение на диск ----------------------------------
    def save_to_file(self, path=None) -> None:
        """Сохраняет историю в JSON-файл (не критично для Дня 27)."""
        path = path or cfg.HISTORY_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass  # сохранение — необязательно, не должно ломать работу
