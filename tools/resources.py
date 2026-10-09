#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сбор метрик ресурсов (VRAM/RAM) во время прогона моделей (День 29).

* VRAM — через nvidia-smi (опрос каждые 200 мс);
* RAM процесса Ollama — через Get-Process (WorkingSet64);
* потоковый сэмплер пиков, который можно запускать вокруг прогона.

Примеры:
    python tools/resources.py            # текущее потребление + выгрузка моделей
    python tools/resources.py --unload   # выгрузить все модели из VRAM
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Ollama  # noqa: E402

_NVIDIA_SMI_CANDIDATES = (
    Path(r"C:\Windows\System32\nvidia-smi.exe"),
    Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
)


def _nvidia_smi() -> str:
    for path in _NVIDIA_SMI_CANDIDATES:
        if path.exists():
            return str(path)
    return "nvidia-smi"  # возможно, есть в PATH


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def query_vram() -> dict:
    """Возвращает {'used_mb': int|None, 'total_mb': int|None}."""
    out = _run([_nvidia_smi(), "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits"])
    if not out:
        return {"used_mb": None, "total_mb": None}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {"used_mb": int(parts[0]), "total_mb": int(parts[1])}
    except (ValueError, IndexError):
        return {"used_mb": None, "total_mb": None}


def ollama_ram_mb() -> int | None:
    """Суммарный WorkingSet всех процессов ollama (МБ)."""
    cmd = ("(Get-Process ollama -ErrorAction SilentlyContinue | "
           "Measure-Object WorkingSet64 -Sum).Sum")
    out = _run(["powershell", "-NoProfile", "-Command", cmd])
    if not out:
        return None
    try:
        return int(float(out) / (1024 * 1024))
    except ValueError:
        return None


@dataclass
class ResourceSample:
    t: float
    vram_used_mb: int | None
    ram_mb: int | None


class ResourceSampler:
    """Фоновый сэмплер VRAM/RAM с фиксацией пиков."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: list[ResourceSample] = []
        self.vram_base_mb: int | None = None
        self.ram_base_mb: int | None = None
        self.vram_peak_mb: int = 0
        self.ram_peak_mb: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = query_vram().get("used_mb")
            ram = ollama_ram_mb()
            if self.vram_base_mb is None and used is not None:
                self.vram_base_mb = used
            if self.ram_base_mb is None and ram is not None:
                self.ram_base_mb = ram
            if used is not None:
                self.vram_peak_mb = max(self.vram_peak_mb, used)
            if ram is not None:
                self.ram_peak_mb = max(self.ram_peak_mb, ram)
            self.samples.append(ResourceSample(time.time(), used, ram))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        """Итог замеров: база, пик и прирост по VRAM/RAM."""
        vram_delta = (
            self.vram_peak_mb - self.vram_base_mb
            if self.vram_peak_mb and self.vram_base_mb is not None else None
        )
        ram_delta = (
            self.ram_peak_mb - self.ram_base_mb
            if self.ram_peak_mb and self.ram_base_mb is not None else None
        )
        return {
            "vram_base_mb": self.vram_base_mb,
            "vram_peak_mb": self.vram_peak_mb or None,
            "vram_delta_mb": vram_delta,
            "ram_base_mb": self.ram_base_mb,
            "ram_peak_mb": self.ram_peak_mb or None,
            "ram_delta_mb": ram_delta,
            "samples_count": len(self.samples),
        }


def unload_all_models() -> list[str]:
    """Выгружает все модели из VRAM (перед замером «холодного» старта)."""
    ollama = Ollama()
    try:
        return ollama.unload_all()
    finally:
        ollama.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Метрики ресурсов Ollama (День 29)")
    ap.add_argument("--unload", action="store_true", help="выгрузить все модели из VRAM")
    args = ap.parse_args()

    vram = query_vram()
    ram = ollama_ram_mb()
    print(f"VRAM: {vram['used_mb']} МБ из {vram['total_mb']} МБ")
    print(f"RAM процесса ollama: {ram} МБ")

    if args.unload:
        unloaded = unload_all_models()
        print("Выгружены модели:", ", ".join(unloaded) if unloaded else "(нет загруженных)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
