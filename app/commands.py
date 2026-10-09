# -*- coding: utf-8 -*-
"""
Реестр команд — единый источник описаний.

Из этого реестра формируется и справка /help, и обработка команд.
Чтобы добавить новую команду, достаточно зарегистрировать CommandSpec
в словаре COMMANDS в конце файла — она автоматически появится в /help.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import config as cfg
from . import profiles as profiles_mod
from .config import GenerationParams
from .llm_client import LLMError, ModelNotFoundError, OllamaUnavailableError
from .session import Session

log = logging.getLogger("app.commands")


# ---------------------------------------------------------------------------
# Контекст и результат выполнения команды
# ---------------------------------------------------------------------------
@dataclass
class CommandContext:
    """Данные, доступные обработчику команды."""

    session: Session
    client: Any  # OllamaClient (без импорта типа — чтобы избежать циклов)


@dataclass
class CommandResult:
    """Результат команды, который уходит на фронт.

    `data` может содержать подсказку для UI, например {"action": "clear"}.
    """

    text: str
    ok: bool = True
    data: dict[str, Any] = field(default_factory=dict)


CommandHandler = Callable[[CommandContext, str], Awaitable[CommandResult]]


@dataclass
class CommandSpec:
    """Описание одной команды."""

    name: str
    description: str
    args: str = ""
    handler: CommandHandler | None = None
    hidden: bool = False  # скрытые команды не показываются в /help


# ---------------------------------------------------------------------------
# Работа с реестром
# ---------------------------------------------------------------------------
COMMANDS: dict[str, CommandSpec] = {}
ALIASES: dict[str, str] = {"?": "help"}


def register(spec: CommandSpec) -> None:
    """Регистрирует команду в реестре."""
    COMMANDS[spec.name.lower()] = spec


def is_command(text: str) -> bool:
    """Проверяет, является ли текст командой (начинается с '/')."""
    return text.strip().startswith("/")


def resolve(name: str) -> CommandSpec | None:
    """Находит команду по имени (с учётом алиасов)."""
    key = name.lower()
    key = ALIASES.get(key, key)
    return COMMANDS.get(key)


def visible_commands() -> list[CommandSpec]:
    """Список команд, показываемых в справке."""
    seen: set[str] = set()
    result: list[CommandSpec] = []
    for spec in COMMANDS.values():
        if spec.hidden or spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


def help_text() -> str:
    """Формирует текст справки строго из реестра команд."""
    lines = ["Доступные команды:", ""]
    for spec in visible_commands():
        usage = "/" + spec.name + (f" {spec.args}" if spec.args else "")
        lines.append(f"  {usage:<32} — {spec.description}")
    lines.append("")
    lines.append("Любой другой текст отправляется модели как сообщение.")
    return "\n".join(lines)


def parse(text: str) -> tuple[str, str]:
    """Разбирает строку команды на имя и аргументы.

    '/model gemma3:12b' -> ('model', 'gemma3:12b')
    """
    body = text.strip()[1:]  # убираем ведущий '/'
    parts = body.split(maxsplit=1)
    name = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    return name, args.strip()


async def dispatch(text: str, ctx: CommandContext) -> CommandResult:
    """Выполняет команду по тексту. Возвращает CommandResult.

    Неизвестная команда и ошибки LLM превращаются в результат с ok=False
    и понятным русским текстом.
    """
    name, args = parse(text)
    spec = resolve(name)

    if spec is None:
        return CommandResult(
            text=f"Неизвестная команда: /{name}\n\n" + help_text(),
            ok=False,
        )
    if spec.handler is None:
        return CommandResult(text=f"Команда /{spec.name} пока не реализована.", ok=False)

    try:
        result = await spec.handler(ctx, args)
    except OllamaUnavailableError as exc:
        return CommandResult(text=exc.message, ok=False)
    except LLMError as exc:
        return CommandResult(text=exc.message, ok=False)
    except Exception as exc:  # noqa: BLE001 — не роняем сервер из-за команды
        log.error("Ошибка команды /%s: %s", spec.name, exc, exc_info=exc)
        return CommandResult(text=f"Ошибка выполнения команды /{spec.name}: {exc}", ok=False)
    return result


# ---------------------------------------------------------------------------
# Вспомогательные функции форматирования
# ---------------------------------------------------------------------------
def _human_size(num: int | None) -> str:
    """Человекочитаемый размер (например, 8.1 ГБ)."""
    if not num:
        return "?"
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    value = float(num)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"


def _fmt_uptime(seconds: float) -> str:
    """Форматирует аптайм в виде «1 ч 5 мин 3 с»."""
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days} д")
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    parts.append(f"{secs} с")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Обработчики команд
# ---------------------------------------------------------------------------
async def cmd_help(ctx: CommandContext, args: str) -> CommandResult:
    return CommandResult(help_text())


async def cmd_clear(ctx: CommandContext, args: str) -> CommandResult:
    await ctx.session.clear()
    return CommandResult("История диалога очищена.", data={"action": "clear"})


async def cmd_models(ctx: CommandContext, args: str) -> CommandResult:
    raw = await ctx.client.list_models_raw()
    current = ctx.session.model
    loaded = {m.get("name") for m in await ctx.client.ps()}

    if not raw:
        return CommandResult("Ollama не вернула ни одной установленной модели.")

    lines = ["Доступные модели Ollama:", ""]
    for m in raw:
        name = m.get("name", "")
        marks: list[str] = []
        if name == current:
            marks.append("текущая")
        if name in loaded:
            marks.append("в VRAM")
        suffix = f"   [{', '.join(marks)}]" if marks else ""
        lines.append(f"  • {name}{suffix}")
    lines.append("")
    lines.append(f"Всего моделей: {len(raw)}. Текущая: {current}")
    return CommandResult("\n".join(lines))


async def cmd_model(ctx: CommandContext, args: str) -> CommandResult:
    name = args.strip()
    if not name:
        return CommandResult(
            "Укажите модель: /model <имя>. Например: /model gemma3:12b-it-qat",
            ok=False,
        )

    available = await ctx.client.list_models()
    if name not in available:
        raise ModelNotFoundError(name, available)

    old = ctx.session.model
    unload_note = ""
    if old and old != name:
        try:
            await ctx.client.unload_model(old)
            unload_note = f" Предыдущая модель «{old}» выгружена из VRAM."
        except LLMError as exc:
            unload_note = f" Предыдущую модель «{old}» выгрузить не удалось: {exc.message}"
            log.warning("Выгрузка модели %s не удалась: %s", old, exc.message)

    ctx.session.model = name
    return CommandResult(
        f"Модель переключена на «{name}».{unload_note}",
        data={"action": "model", "model": name},
    )


async def cmd_unload(ctx: CommandContext, args: str) -> CommandResult:
    model = ctx.session.model
    await ctx.client.unload_model(model)
    return CommandResult(
        f"Модель «{model}» выгружена из VRAM. Следующий запрос загрузит её снова.",
        data={"action": "model", "model": model},
    )


async def cmd_system(ctx: CommandContext, args: str) -> CommandResult:
    args = args.strip()

    # Без аргументов — показать текущий промпт.
    if not args:
        current = ctx.session.system_prompt or "(пусто)"
        return CommandResult(f"Текущий системный промпт:\n\n{current}")

    head, _, tail = args.partition(" ")
    head = head.lower()
    name = tail.strip()

    # /system reset — вернуть дефолт из профиля задачи.
    if args.lower() == "reset":
        profile = cfg.get_task_profile()
        ctx.session.system_prompt = profile.system_prompt
        return CommandResult(
            "Системный промпт сброшен к дефолту профиля:\n\n" + profile.system_prompt,
            data={"action": "system", "system": profile.system_prompt},
        )

    # /system save <имя> — сохранить именованный промпт.
    if head == "save":
        if not name:
            return CommandResult("Использование: /system save <имя>", ok=False)
        profiles_mod.save_profile(
            name,
            model=ctx.session.model,
            system_prompt=ctx.session.system_prompt,
            params=ctx.session.params.to_dict(),
            description="Именованный промпт",
        )
        return CommandResult(f"Системный промпт сохранён как «{name}».")

    # /system load <имя> — загрузить именованный промпт.
    if head == "load":
        if not name:
            return CommandResult("Использование: /system load <имя>", ok=False)
        prof = profiles_mod.get_profile(name)
        if not prof:
            available = ", ".join(p["name"] for p in profiles_mod.list_profiles())
            return CommandResult(f"Профиль «{name}» не найден. Доступные: {available}", ok=False)
        ctx.session.system_prompt = prof.get("system_prompt") or ""
        return CommandResult(
            f"Системный промпт загружен из «{name}»:\n\n{ctx.session.system_prompt}",
            data={"action": "system", "system": ctx.session.system_prompt},
        )

    # Иначе — задать новый системный промпт.
    ctx.session.system_prompt = args
    return CommandResult(
        f"Системный промпт обновлён:\n\n{args}",
        data={"action": "system", "system": args},
    )


async def cmd_status(ctx: CommandContext, args: str) -> CommandResult:
    session = ctx.session
    available = await session_available(ctx)
    version = await ctx.client.version() if available else None
    ollama_state = "доступна"
    if available and version:
        ollama_state += f" (v{version})"
    if not available:
        ollama_state = "недоступна"

    lines = [
        "Статус приложения:",
        "",
        f"  Модель:              {session.model}",
        f"  Сообщений в истории: {session.history_length()}",
        f"  Ollama:              {ollama_state}",
        f"  Аптайм приложения:   {_fmt_uptime(session.uptime())}",
    ]
    return CommandResult("\n".join(lines))


async def session_available(ctx: CommandContext) -> bool:
    """Безопасная проверка доступности Ollama (без исключений)."""
    try:
        return await ctx.client.is_available()
    except Exception:  # noqa: BLE001
        return False


async def cmd_export(ctx: CommandContext, args: str) -> CommandResult:
    snap = ctx.session.snapshot()
    return CommandResult(
        f"История диалога готова к скачиванию: {snap['messages_count']} сообщений. "
        "Файл сохранится автоматически (или откройте GET /api/export).",
        data={"action": "export"},
    )


async def cmd_reset(ctx: CommandContext, args: str) -> CommandResult:
    await ctx.session.reset()
    return CommandResult(
        "Выполнен сброс к настройкам по умолчанию: история очищена, "
        f"модель — {cfg.DEFAULT_MODEL}, системный промпт восстановлен.",
        data={"action": "reset", "model": cfg.DEFAULT_MODEL},
    )


async def cmd_ping(ctx: CommandContext, args: str) -> CommandResult:
    """Тестовая команда: показывает, что /help формируется из реестра."""
    return CommandResult("pong — тестовая команда, зарегистрированная через реестр. Связь есть.")


# ---------------------------------------------------------------------------
# День 29: команды работы с параметрами генерации
# ---------------------------------------------------------------------------
PARAM_FLOAT_KEYS = ("temperature", "top_p", "repeat_penalty")
PARAM_INT_KEYS = ("top_k", "num_predict", "num_ctx", "seed")
PARAM_KEYS = PARAM_FLOAT_KEYS + PARAM_INT_KEYS

_PARAM_ORDER = (
    "temperature", "top_p", "top_k", "repeat_penalty",
    "num_predict", "num_ctx", "seed", "keep_alive",
)


def _format_params(params: GenerationParams) -> str:
    """Человекочитаемый список текущих параметров генерации."""
    d = params.to_dict()
    lines = ["Текущие параметры генерации:", ""]
    for key in _PARAM_ORDER:
        lines.append(f"  {key:<15} = {d.get(key)}")
    return "\n".join(lines)


def _coerce_param(key: str, raw: str):
    """Приводит строковое значение к типу параметра."""
    if key in PARAM_INT_KEYS:
        return int(raw)
    if key in PARAM_FLOAT_KEYS:
        return float(raw)
    return raw


async def cmd_params(ctx: CommandContext, args: str) -> CommandResult:
    args = args.strip()

    # Показать параметры.
    if not args or args.lower() == "show":
        return CommandResult(
            _format_params(ctx.session.params),
            data={"action": "params", "params": ctx.session.params.to_dict()},
        )

    parts = args.split()
    action = parts[0].lower()

    # Сброс к дефолтам профиля.
    if action == "reset":
        profile = cfg.get_task_profile()
        ctx.session.params = GenerationParams(**profile.params.to_dict())
        return CommandResult(
            "Параметры сброшены к дефолтам профиля.\n\n" + _format_params(ctx.session.params),
            data={"action": "params", "params": ctx.session.params.to_dict()},
        )

    # Разбор «set <ключ> <значение>» и сокращённой формы «<ключ> <значение>».
    if action == "set":
        if len(parts) < 3:
            return CommandResult(
                "Использование: /params set <ключ> <значение>. Пример: /params set temperature 0.2",
                ok=False,
            )
        key, raw = parts[1].lower(), " ".join(parts[2:])
    elif action in PARAM_KEYS and len(parts) >= 2:
        key, raw = action, " ".join(parts[1:])
    else:
        return CommandResult(
            "Использование: /params [set <ключ> <значение> | reset]. "
            "Доступные параметры: " + ", ".join(PARAM_KEYS),
            ok=False,
        )

    if key not in PARAM_KEYS:
        return CommandResult(
            f"Неизвестный параметр: {key}. Доступные: {', '.join(PARAM_KEYS)}", ok=False
        )
    try:
        value = _coerce_param(key, raw)
    except ValueError:
        return CommandResult(f"Некорректное значение для {key}: «{raw}»", ok=False)

    ctx.session.update_params({key: value})
    return CommandResult(
        f"Параметр {key} = {value}.\n\n" + _format_params(ctx.session.params),
        data={"action": "params", "params": ctx.session.params.to_dict()},
    )


async def cmd_profile(ctx: CommandContext, args: str) -> CommandResult:
    """Управление профилями: list / save / load / delete."""
    parts = args.split(maxsplit=1)
    action = parts[0].lower() if parts and parts[0] else "list"
    name = parts[1].strip() if len(parts) > 1 else ""

    if action == "list":
        profiles = profiles_mod.list_profiles()
        if not profiles:
            return CommandResult("Профили не найдены.")
        lines = ["Профили:", ""]
        for p in profiles:
            kind = "встроенный" if p.get("builtin") else "пользовательский"
            lines.append(f"  • {p['name']}   [{kind}, модель: {p.get('model') or '—'}]")
            if p.get("description"):
                lines.append(f"      {p['description']}")
        lines += ["", "Загрузить: /profile load <имя>, сохранить: /profile save <имя>, "
                      "удалить: /profile delete <имя>."]
        return CommandResult("\n".join(lines))

    if action == "save":
        if not name:
            return CommandResult("Использование: /profile save <имя>", ok=False)
        try:
            profiles_mod.save_profile(
                name,
                model=ctx.session.model,
                system_prompt=ctx.session.system_prompt,
                params=ctx.session.params.to_dict(),
            )
        except ValueError as exc:
            return CommandResult(str(exc), ok=False)
        return CommandResult(
            f"Профиль «{name}» сохранён (модель: {ctx.session.model})."
        )

    if action == "load":
        if not name:
            return CommandResult("Использование: /profile load <имя>", ok=False)
        prof = profiles_mod.get_profile(name)
        if not prof:
            available = ", ".join(p["name"] for p in profiles_mod.list_profiles())
            return CommandResult(f"Профиль «{name}» не найден. Доступные: {available}", ok=False)
        ctx.session.apply_profile(prof)
        return CommandResult(
            f"Профиль «{name}» загружен. Модель: {ctx.session.model}\n\n"
            + _format_params(ctx.session.params),
            data={
                "action": "profile",
                "profile": name,
                "model": ctx.session.model,
                "params": ctx.session.params.to_dict(),
                "system": ctx.session.system_prompt,
            },
        )

    if action == "delete":
        if not name:
            return CommandResult("Использование: /profile delete <имя>", ok=False)
        if profiles_mod.delete_profile(name):
            return CommandResult(f"Профиль «{name}» удалён.")
        return CommandResult(
            f"Профиль «{name}» не найден (встроенные профили удалить нельзя).", ok=False
        )

    return CommandResult(
        "Использование: /profile [list | save <имя> | load <имя> | delete <имя>]", ok=False
    )


# ---------------------------------------------------------------------------
# Регистрация команд в реестре
# ---------------------------------------------------------------------------
def _register_default_commands() -> None:
    """Регистрирует стандартные команды. /help строится из этого же списка."""
    register(CommandSpec("help", "Показать список команд (из реестра)", "", cmd_help))
    register(CommandSpec("clear", "Очистить историю диалога", "", cmd_clear))
    register(CommandSpec("models", "Показать список моделей Ollama и текущую", "", cmd_models))
    register(
        CommandSpec("model", "Переключить модель (старая выгружается из VRAM)", "<имя>", cmd_model)
    )
    register(CommandSpec("unload", "Выгрузить текущую модель из памяти Ollama", "", cmd_unload))
    register(
        CommandSpec(
            "system",
            "Показать/задать промпт: /system [текст | reset | save <имя> | load <имя>]",
            "[текст|reset|save|load]",
            cmd_system,
        )
    )
    register(
        CommandSpec(
            "params",
            "Параметры генерации: /params [set <ключ> <значение> | reset]",
            "[set|reset]",
            cmd_params,
        )
    )
    register(
        CommandSpec(
            "profile",
            "Профили: /profile [list | save <имя> | load <имя> | delete <имя>]",
            "[list|save|load|delete]",
            cmd_profile,
        )
    )
    register(
        CommandSpec(
            "status",
            "Текущий статус: модель, история, доступность Ollama, аптайм",
            "",
            cmd_status,
        )
    )
    register(CommandSpec("export", "Скачать историю диалога в JSON", "", cmd_export))
    register(CommandSpec("reset", "Сброс к настройкам по умолчанию", "", cmd_reset))
    # Тестовая команда: демонстрирует, что справка /help формируется из реестра.
    register(CommandSpec("ping", "Проверка связи (тестовая команда из реестра)", "", cmd_ping))


_register_default_commands()
