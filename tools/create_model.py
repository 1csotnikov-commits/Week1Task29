#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создание кастомных моделей Ollama через Modelfile (День 29).

Для каждой конфигурации (baseline / optimized / qat) генерируется Modelfile
с SYSTEM и PARAMETERS, затем вызывается `ollama create <имя> -f Modelfile`.
Имена моделей: week1task29-<профиль>-<конфигурация>.

Примеры:
    python tools/create_model.py
    python tools/create_model.py --configs baseline,optimized
    python tools/create_model.py --dry-run        # только сгенерировать Modelfile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Experiment,
    Ollama,
    cfg,
    create_model,
    model_available,
    select_experiments,
)


def build_modelfile(exp: Experiment) -> str:
    """Формирует текст Modelfile (FROM + SYSTEM + PARAMETER)."""
    p = exp.params
    lines = [
        f"# Modelfile для Дня 29: {exp.title}",
        f"FROM {exp.base_model}",
        "",
        'SYSTEM """',
        exp.system_prompt,
        '"""',
        "",
        f"PARAMETER temperature {p.temperature}",
        f"PARAMETER top_p {p.top_p}",
        f"PARAMETER top_k {p.top_k}",
        f"PARAMETER repeat_penalty {p.repeat_penalty}",
        f"PARAMETER num_predict {p.num_predict}",
        f"PARAMETER num_ctx {p.num_ctx}",
    ]
    if p.seed is not None:
        lines.append(f"PARAMETER seed {p.seed}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Создание кастомных моделей Ollama (День 29)")
    ap.add_argument("--profile", default=None, help="профиль задачи (по умолчанию из config)")
    ap.add_argument("--configs", default="baseline,optimized,qat",
                    help="список конфигураций через запятую")
    ap.add_argument("--out", default=str(cfg.MODELFILES_DIR), help="куда сохранять Modelfile")
    ap.add_argument("--dry-run", action="store_true", help="только сгенерировать Modelfile")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = select_experiments(args.profile, args.configs.split(","))
    if not experiments:
        print("Нет конфигураций для создания.")
        return 1

    created: list[str] = []
    for exp in experiments:
        modelfile_text = build_modelfile(exp)
        mf_path = out_dir / f"{exp.model_name}.Modelfile"
        mf_path.write_text(modelfile_text, encoding="utf-8")
        print(f"[+] Modelfile сохранён: {mf_path}")
        if args.dry_run:
            print(modelfile_text)
            continue
        print(f"    создаю модель «{exp.model_name}» (FROM {exp.base_model}) ...")
        try:
            create_model(exp.model_name, mf_path)
            created.append(exp.model_name)
            print("    OK")
        except Exception as exc:  # noqa: BLE001
            print(f"    ОШИБКА: {exc}")

    if args.dry_run:
        return 0

    # Проверка через ollama list.
    ollama = Ollama()
    try:
        available = ollama.list_models()
    finally:
        ollama.close()

    print("\nПроверка (ollama list):")
    for name in created:
        mark = "✔" if model_available(name, available) else "✘"
        print(f"  {mark} {name}")
    print(f"\nВсего моделей в Ollama: {len(available)}")
    return 0 if all(model_available(n, available) for n in created) else 2


if __name__ == "__main__":
    raise SystemExit(main())
