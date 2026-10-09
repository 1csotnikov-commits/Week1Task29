# -*- coding: utf-8 -*-
"""
Работа с профилями (JSON).

Профиль = сохранённый набор «модель + системный промпт + параметры генерации».
Профили бывают двух видов:
  * встроенные (built-in) — из app/config.py (TASK_PROFILES), только для чтения;
  * пользовательские — файлы profiles/<имя>.json, их можно создавать/удалять.

Профиль — это и есть «вынесенная в конфигурацию» часть, чтобы UI не хардкодил
параметры (задел на День 30: конфигурация читается из профилей).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from . import config as cfg

log = logging.getLogger("app.profiles")

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-. ]{1,64}$")


def _valid_name(name: str) -> bool:
    return bool(name) and bool(_NAME_RE.match(name))


def _profile_path(name: str) -> Path:
    return cfg.PROFILES_DIR / f"{name}.json"


# ---------------------------------------------------------------------------
# Встроенные профили
# ---------------------------------------------------------------------------
def builtin_profiles() -> list[dict[str, Any]]:
    """Встроенные профили целевой задачи (из config.TASK_PROFILES)."""
    result = []
    for prof in cfg.TASK_PROFILES.values():
        result.append(
            {
                "name": prof.name,
                "description": prof.description,
                "model": prof.optimized_model,
                "system_prompt": prof.system_prompt,
                "params": prof.params.to_dict(),
                "builtin": True,
                "created_at": None,
            }
        )
    return result


def get_builtin(name: str) -> dict[str, Any] | None:
    for p in builtin_profiles():
        if p["name"] == name:
            return p
    return None


# ---------------------------------------------------------------------------
# Пользовательские профили (файлы)
# ---------------------------------------------------------------------------
def list_user_profiles() -> list[dict[str, Any]]:
    """Список пользовательских профилей из каталога profiles/."""
    cfg.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for path in sorted(cfg.PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("name", path.stem)
            data["builtin"] = False
            result.append(data)
        except (OSError, ValueError) as exc:
            log.warning("Пропущен некорректный профиль %s: %s", path.name, exc)
    return result


def list_profiles() -> list[dict[str, Any]]:
    """Все профили: сначала пользовательские, затем встроенные."""
    return list_user_profiles() + builtin_profiles()


def get_profile(name: str) -> dict[str, Any] | None:
    """Находит профиль: сначала пользовательский, затем встроенный."""
    if _valid_name(name):
        path = _profile_path(name)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data.setdefault("name", name)
                data["builtin"] = False
                return data
            except (OSError, ValueError):
                pass
    return get_builtin(name)


def save_profile(
    name: str,
    *,
    model: str,
    system_prompt: str,
    params: dict[str, Any],
    description: str = "",
) -> dict[str, Any]:
    """Сохраняет профиль в profiles/<имя>.json."""
    if not _valid_name(name):
        raise ValueError(
            "Недопустимое имя профиля. Разрешены буквы, цифры, пробел, '_', '-', '.' (до 64 символов)."
        )
    cfg.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "description": description,
        "model": model,
        "system_prompt": system_prompt,
        "params": cfg.GenerationParams.from_dict(params).to_dict(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _profile_path(name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Профиль сохранён: %s", name)
    return data


def delete_profile(name: str) -> bool:
    """Удаляет пользовательский профиль. Встроенные удалить нельзя."""
    if not _valid_name(name):
        return False
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        log.info("Профиль удалён: %s", name)
        return True
    return False
