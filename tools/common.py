# -*- coding: utf-8 -*-
"""
Общие утилиты инструментов Дня 29.

Скрипты из tools/ запускаются из корня проекта, например:
    python tools/compare.py --profile 1c_support --configs baseline,optimized,qat

Здесь собраны: bootstrap sys.path (чтобы импортировать пакет app), синхронный
клиент Ollama для CLI-скриптов, описания экспериментов (baseline/optimized/qat)
и вспомогательные функции форматирования.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- Bootstrap: делаем пакет app импортируемым при запуске из tools/ ---------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from app import config as cfg  # noqa: E402

OLLAMA_EXE = r"C:\LLM\Ollama\ollama.exe"

# Скрипты пишут логи/отчёты в UTF-8 — принудительно задаём кодировку вывода,
# чтобы файлы были корректными независимо от кодовой страницы консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def timestamp() -> str:
    """Метка времени для имён файлов отчётов."""
    return time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Описание экспериментов (конфигураций) сравнения
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """Одна конфигурация: базовая модель + промпт + параметры."""

    key: str           # baseline | optimized | qat
    title: str
    base_model: str    # FROM в Modelfile
    model_name: str    # имя кастомной модели в Ollama
    system_prompt: str
    params: cfg.GenerationParams
    baseline: bool = False


def build_experiments(profile_name: str | None = None) -> dict[str, Experiment]:
    """Строит три конфигурации сравнения для профиля задачи."""
    prof = cfg.get_task_profile(profile_name)
    # Параметры «как в Дне 27»: генеричный промпт, высокая температура, узкий контекст.
    baseline_params = cfg.GenerationParams(
        temperature=0.7, top_p=1.0, top_k=40, repeat_penalty=1.1,
        num_predict=-1, num_ctx=4096, seed=42,
    )
    return {
        "baseline": Experiment(
            key="baseline",
            title="baseline — День 27: gpt-oss:20b, generic prompt, temp 0.7, ctx 4096",
            base_model=prof.baseline_model,
            model_name="week1task29-1c-baseline",
            system_prompt=cfg.DEFAULT_SYSTEM_PROMPT,
            params=baseline_params,
            baseline=True,
        ),
        "optimized": Experiment(
            key="optimized",
            title=f"optimized — {prof.optimized_model}, 1С-prompt, temp 0.2, ctx 8192",
            base_model=prof.optimized_model,
            model_name="week1task29-1c-optimized",
            system_prompt=prof.system_prompt,
            params=prof.params,
        ),
        "qat": Experiment(
            key="qat",
            title=f"qat — {prof.quantization_model} (QAT Q4_0), тот же prompt/параметры",
            base_model=prof.quantization_model,
            model_name="week1task29-1c-qat",
            system_prompt=prof.system_prompt,
            params=prof.params,
        ),
    }


def select_experiments(profile_name: str | None, keys: list[str]) -> list[Experiment]:
    """Выбирает эксперименты по списку ключей (baseline/optimized/qat)."""
    all_exp = build_experiments(profile_name)
    result: list[Experiment] = []
    for key in keys:
        key = key.strip().lower()
        if key in all_exp:
            result.append(all_exp[key])
        else:
            print(f"[!] неизвестная конфигурация: {key} (доступно: {', '.join(all_exp)})")
    return result


def model_available(name: str, available: list[str]) -> bool:
    """Проверяет наличие модели с учётом суффикса ':latest'.

    `ollama list` возвращает имена вида 'week1task29-1c-baseline:latest',
    а сравниваем мы с именем без тега.
    """
    if name in available:
        return True
    return any(item == f"{name}:latest" or item.startswith(f"{name}:") for item in available)


# ---------------------------------------------------------------------------
# Тестовые запросы
# ---------------------------------------------------------------------------
def load_queries(path: Path | None = None) -> list[dict[str, Any]]:
    """Загружает список тестовых запросов из evaluation_queries.json."""
    p = Path(path) if path else cfg.EVAL_QUERIES_FILE
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("queries", [])
    return data


# ---------------------------------------------------------------------------
# Синхронный клиент Ollama (удобен для CLI-инструментов)
# ---------------------------------------------------------------------------
class Ollama:
    """Минимальный синхронный клиент Ollama для инструментов сравнения."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base = (base_url or cfg.OLLAMA_BASE_URL).rstrip("/")
        self.client = httpx.Client(base_url=self.base, timeout=cfg.REQUEST_TIMEOUT)

    def close(self) -> None:
        self.client.close()

    def is_available(self) -> bool:
        try:
            return self.client.get("/api/version", timeout=cfg.CONNECT_TIMEOUT).status_code == 200
        except httpx.RequestError:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get("/api/tags", timeout=cfg.LIST_TIMEOUT)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except httpx.RequestError as exc:
            raise RuntimeError(cfg.OLLAMA_START_HINT + f" ({exc})") from exc

    def running(self) -> list[str]:
        try:
            resp = self.client.get("/api/ps", timeout=cfg.LIST_TIMEOUT)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except (httpx.RequestError, ValueError):
            pass
        return []

    def unload_all(self) -> list[str]:
        """Выгружает все модели из VRAM (перед замерами)."""
        unloaded: list[str] = []
        for name in self.running():
            try:
                self.client.post(
                    "/api/generate",
                    json={"model": name, "keep_alive": 0, "prompt": ""},
                    timeout=cfg.UNLOAD_TIMEOUT,
                )
                unloaded.append(name)
            except httpx.RequestError:
                pass
        return unloaded

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict | None = None,
        keep_alive: str | int = "5m",
    ) -> dict[str, Any]:
        """Синхронный вызов /api/chat (stream=false) с возвратом сырого JSON."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {},
            "keep_alive": keep_alive,
        }
        resp = self.client.post("/api/chat", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama вернула {resp.status_code}: {resp.text}")
        return resp.json()


def create_model(name: str, modelfile_path: Path) -> str:
    """Вызывает `ollama create <name> -f <Modelfile>` и возвращает вывод."""
    exe = Path(OLLAMA_EXE)
    cmd = [str(exe) if exe.exists() else "ollama", "create", name, "-f", str(modelfile_path)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"ollama create не удался:\n{output}")
    return output.strip()


# ---------------------------------------------------------------------------
# Извлечение метрик из ответа Ollama
# ---------------------------------------------------------------------------
def extract_metrics(resp: dict[str, Any], wall_time: float) -> dict[str, Any]:
    """Считает метрики по сырому ответу /api/chat."""
    answer = (resp.get("message") or {}).get("content", "") or ""
    eval_count = resp.get("eval_count", 0) or 0
    eval_duration = resp.get("eval_duration", 0) or 0
    tps = (eval_count / (eval_duration / 1e9)) if eval_duration else 0.0
    return {
        "answer": answer,
        "answer_chars": len(answer),
        "answer_words": len(answer.split()),
        "eval_count": eval_count,
        "prompt_eval_count": resp.get("prompt_eval_count", 0) or 0,
        "eval_duration_s": round(eval_duration / 1e9, 3),
        "load_duration_s": round((resp.get("load_duration", 0) or 0) / 1e9, 3),
        "total_duration_s": round((resp.get("total_duration", 0) or 0) / 1e9, 3),
        "wall_time_s": round(wall_time, 3),
        "tokens_per_sec": round(tps, 2),
    }


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

