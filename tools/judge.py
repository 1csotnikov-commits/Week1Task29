#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge: оценка качества ответов локальной моделью (День 29).

Судья — локальная модель (по умолчанию gpt-oss:20b) с temperature=0.0 и
фиксированным seed. Рубрика: полнота, корректность, релевантность, формат
(каждое 0–5). Для честности порядок A/B рандомизируется: делается два прохода
с перестановкой, оценки усредняются.

Пример (ручная проверка):
    python tools/judge.py --query "Что такое ТекущаяДата?" --a "Дата сервера" --b "Не знаю"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Ollama  # noqa: E402

RUBRIC = ("completeness", "correctness", "relevance", "format")

JUDGE_SYSTEM = (
    "Ты — строгий, но справедливый эксперт по платформе «1С:Предприятие 8.3». "
    "Ты оцениваешь два ответа (A и B) ассистента техподдержки на один и тот же "
    "вопрос. Оценивай каждый ответ по рубрике, используя ключевые пункты эталона:\n"
    "  * completeness — полнота (все ли ключевые пункты раскрыты), 0–5;\n"
    "  * correctness  — корректность (нет ли фактических/технических ошибок), 0–5;\n"
    "  * relevance    — релевантность (по делу ли, без воды), 0–5;\n"
    "  * format       — формат/структура (понятность, код оформлен), 0–5.\n\n"
    "Отвечай СТРОГО одним JSON-объектом, без пояснений вне JSON:\n"
    '{"A": {"completeness": N, "correctness": N, "relevance": N, "format": N}, '
    '"B": {"completeness": N, "correctness": N, "relevance": N, "format": N}, '
    '"reason": "краткое обоснование (1–2 предложения)"}'
)


def _parse_json(text: str) -> dict | None:
    """Извлекает первый JSON-объект из ответа судьи."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


def _total(scores: dict) -> float:
    values = [float(scores.get(key, 0)) for key in RUBRIC]
    return round(sum(values) / len(values), 2)


def _norm_scores(scores: dict | None) -> dict:
    scores = scores or {}
    out = {key: float(scores.get(key, 0)) for key in RUBRIC}
    out["total"] = _total(out)
    return out


def _single_pass(
    ollama: Ollama,
    model: str,
    query: str,
    key_points: list[str],
    answer_a: str,
    answer_b: str,
    seed: int,
    num_ctx: int,
    num_predict: int,
) -> dict | None:
    """Один проход судьи: A и B в заданном порядке."""
    key_points_text = "\n".join(f"- {p}" for p in key_points) or "(не заданы)"
    user = (
        f"ВОПРОС:\n{query}\n\n"
        f"КЛЮЧЕВЫЕ ПУНКТЫ ЭТАЛОННОГО ОТВЕТА:\n{key_points_text}\n\n"
        f"ОТВЕТ A:\n{answer_a}\n\n"
        f"ОТВЕТ B:\n{answer_b}\n\n"
        "Оцени оба ответа и верни JSON по заданной схеме."
    )
    options = {
        "temperature": 0.0,
        "seed": seed,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "top_p": 1.0,
    }
    resp = ollama.chat(
        model,
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        options,
    )
    text = (resp.get("message") or {}).get("content", "") or ""
    return _parse_json(text)


def judge_pair(
    query: str,
    key_points: list[str],
    answer_a: str,
    answer_b: str,
    ollama: Ollama | None = None,
    model: str = "gpt-oss:20b",
    seed: int = 42,
    num_ctx: int = 8192,
    num_predict: int = 1024,
    swap: bool = True,
) -> dict[str, Any]:
    """Оценивает два ответа и возвращает усреднённые оценки A и B.

    При swap=True выполняется второй проход с перестановкой A/B — это снижает
    позиционное смещение судьи.
    """
    own = ollama is None
    client = ollama or Ollama()
    try:
        pass1 = _single_pass(client, model, query, key_points, answer_a, answer_b,
                             seed, num_ctx, num_predict)
        result = {
            "A": _norm_scores(pass1.get("A") if pass1 else None),
            "B": _norm_scores(pass1.get("B") if pass1 else None),
            "reason": (pass1 or {}).get("reason", ""),
            "passes": 1,
            "raw_pass1": pass1,
        }
        if swap:
            pass2 = _single_pass(client, model, query, key_points, answer_b, answer_a,
                                 seed, num_ctx, num_predict)
            if pass2:
                # Во втором проходе A и B переставлены — усредняем с учётом этого.
                p2_a = _norm_scores(pass2.get("B"))  # B второго прохода = A первого
                p2_b = _norm_scores(pass2.get("A"))
                for key in RUBRIC + ("total",):
                    result["A"][key] = round((result["A"][key] + p2_a[key]) / 2, 2)
                    result["B"][key] = round((result["B"][key] + p2_b[key]) / 2, 2)
                result["passes"] = 2
                result["raw_pass2"] = pass2
                if pass2.get("reason"):
                    result["reason"] = (result["reason"] + " | " + pass2["reason"]).strip(" |")
        return result
    finally:
        if own:
            client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge (День 29)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--a", required=True, help="ответ A")
    ap.add_argument("--b", required=True, help="ответ B")
    ap.add_argument("--key-points", default="", help="ключевые пункты через ';'")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key_points = [p.strip() for p in args.key_points.split(";") if p.strip()]
    result = judge_pair(args.query, key_points, args.a, args.b, model=args.model, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
