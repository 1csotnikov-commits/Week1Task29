#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение «до/после» для локальной LLM (День 29).

Прогоняет набор тестовых запросов (evaluation_queries.json) через несколько
конфигураций (baseline / optimized / qat), фиксирует метрики скорости и
ресурсов, оценивает качество через LLM-as-judge и формирует отчёты:
  * reports/raw_<timestamp>.json  — сырые результаты;
  * reports/REPORT_<timestamp>.md — сводный отчёт «до/после».

Пример:
    python tools/compare.py --profile 1c_support --configs baseline,optimized,qat --seed 42
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Experiment,
    Ollama,
    cfg,
    extract_metrics,
    load_queries,
    mean,
    model_available,
    save_json,
    select_experiments,
    timestamp,
)
from judge import RUBRIC, judge_pair  # noqa: E402
from resources import ResourceSampler  # noqa: E402


def _std(values: list[float]) -> float:
    """Выборочное стандартное отклонение."""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5


def _messages_for(exp: Experiment, query: str, use_custom: bool) -> list[dict[str, str]]:
    """Формирует список сообщений.

    Если кастомная модель создана (SYSTEM «зашит» в Modelfile), системное
    сообщение не передаём — иначе оно продублируется с Modelfile.
    """
    messages: list[dict[str, str]] = []
    if not use_custom:
        messages.append({"role": "system", "content": exp.system_prompt})
    messages.append({"role": "user", "content": query})
    return messages


def run_config(
    exp: Experiment,
    queries: list[dict],
    ollama: Ollama,
    seed: int,
    available: list[str],
    speed_repeat: int,
    log=print,
) -> dict:
    """Прогоняет одну конфигурацию по всем запросам + замер ресурсов/стабильности."""
    use_custom = model_available(exp.model_name, available)
    model = exp.model_name if use_custom else exp.base_model
    log(f"  Модель: {model} ({'кастомная через Modelfile' if use_custom else 'базовая'})")

    # Холодный старт: выгружаем всё из VRAM.
    ollama.unload_all()
    time.sleep(1)

    sampler = ResourceSampler(interval=0.2)
    sampler.start()
    results: dict[str, dict] = {}
    try:
        for item in queries:
            options = exp.params.to_ollama_options()
            options["seed"] = seed
            messages = _messages_for(exp, item["query"], use_custom)
            t0 = time.perf_counter()
            try:
                resp = ollama.chat(model, messages, options, keep_alive=exp.params.keep_alive)
                metrics = extract_metrics(resp, time.perf_counter() - t0)
                results[item["id"]] = {"ok": True, **metrics}
                log(f"    {exp.key} · {item['id']}: {metrics['eval_count']} ток, "
                    f"{metrics['tokens_per_sec']} ток/с, {metrics['wall_time_s']} с")
            except Exception as exc:  # noqa: BLE001
                results[item["id"]] = {"ok": False, "error": str(exc), "answer": ""}
                log(f"    {exp.key} · {item['id']}: ОШИБКА — {exc}")

        # Стабильность: несколько повторов первого (короткого) запроса.
        stability = None
        if speed_repeat and speed_repeat > 1:
            first = queries[0]
            options = exp.params.to_ollama_options()
            options["seed"] = seed
            messages = _messages_for(exp, first["query"], use_custom)
            tps_values: list[float] = []
            for _ in range(speed_repeat):
                try:
                    resp = ollama.chat(model, messages, options, keep_alive=exp.params.keep_alive)
                    tps_values.append(extract_metrics(resp, 0.0)["tokens_per_sec"])
                except Exception:  # noqa: BLE001
                    pass
            if tps_values:
                stability = {
                    "runs": len(tps_values),
                    "tokens_per_sec": tps_values,
                    "mean": mean(tps_values),
                    "min": min(tps_values),
                    "max": max(tps_values),
                    "std": round(_std(tps_values), 2),
                }
    finally:
        sampler.stop()

    return {
        "key": exp.key,
        "title": exp.title,
        "model": model,
        "used_custom": use_custom,
        "system_prompt": exp.system_prompt,
        "params": exp.params.to_dict(),
        "results": results,
        "resources": sampler.summary(),
        "stability": stability,
    }


def aggregate(config_run: dict) -> dict:
    """Сводные метрики одной конфигурации по всем успешным запросам."""
    rows = [r for r in config_run["results"].values() if r.get("ok")]
    total = len(config_run["results"])
    if not rows:
        return {"queries": 0, "errors": total}
    return {
        "queries": len(rows),
        "errors": total - len(rows),
        "tokens_total": sum(r["eval_count"] for r in rows),
        "tokens_mean": mean([r["eval_count"] for r in rows]),
        "tps_mean": mean([r["tokens_per_sec"] for r in rows]),
        "tps_min": min(r["tokens_per_sec"] for r in rows),
        "tps_max": max(r["tokens_per_sec"] for r in rows),
        "wall_mean": mean([r["wall_time_s"] for r in rows]),
        "wall_total": round(sum(r["wall_time_s"] for r in rows), 2),
        "load_max": round(max(r["load_duration_s"] for r in rows), 3),
        "chars_mean": mean([r["answer_chars"] for r in rows]),
    }


def run_judge(
    config_runs: list[dict],
    queries: list[dict],
    ollama: Ollama,
    judge_model: str,
    seed: int,
    log=print,
) -> dict:
    """Оценивает ответы конфигураций попарно через LLM-as-judge."""
    keys = [c["key"] for c in config_runs]
    by_key = {c["key"]: c for c in config_runs}
    scores: dict[str, list[float]] = {k: [] for k in keys}
    details: list[dict] = []

    for a_key, b_key in itertools.combinations(keys, 2):
        for q in queries:
            qid = q["id"]
            answer_a = by_key[a_key]["results"].get(qid, {}).get("answer", "")
            answer_b = by_key[b_key]["results"].get(qid, {}).get("answer", "")
            if not answer_a or not answer_b:
                continue
            try:
                res = judge_pair(
                    q["query"], q.get("key_points", []), answer_a, answer_b,
                    ollama=ollama, model=judge_model, seed=seed,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"    судья: ошибка {a_key} vs {b_key} / {qid}: {exc}")
                continue
            scores[a_key].append(res["A"]["total"])
            scores[b_key].append(res["B"]["total"])
            details.append({
                "pair": [a_key, b_key],
                "query": qid,
                "A": res["A"],
                "B": res["B"],
                "reason": res.get("reason", ""),
            })
            log(f"    судья {a_key} vs {b_key} · {qid}: "
                f"{res['A']['total']} : {res['B']['total']}")

    summary = {
        key: {"avg_total": mean(values), "queries": len(values)}
        for key, values in scores.items()
    }
    return {"summary": summary, "details": details}


def _trunc(text: str, limit: int = 1400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… (текст обрезан)"


def _rubric_avg(judge: dict, key: str) -> dict:
    """Средние оценки по каждой рубрике для конфигурации."""
    values: dict[str, list[float]] = {r: [] for r in RUBRIC}
    for detail in judge.get("details", []):
        a_key, b_key = detail["pair"]
        if key == a_key:
            scores = detail["A"]
        elif key == b_key:
            scores = detail["B"]
        else:
            continue
        for r in RUBRIC:
            values[r].append(float(scores.get(r, 0)))
    return {r: mean(v) for r, v in values.items()}


def build_report(raw: dict, out_path: Path) -> Path:
    """Формирует Markdown-отчёт «до/после»."""
    configs = raw["configs"]
    aggs = raw["aggregates"]
    judge = raw.get("judge") or {}
    queries = raw["queries"]
    lines: list[str] = []
    add = lines.append

    add("# Отчёт сравнения «до/после» — локальная LLM (День 29)")
    add("")
    add(f"- **Профиль задачи:** {raw['profile']}")
    add(f"- **Дата:** {raw['created_at']}")
    add(f"- **Seed:** {raw['seed']} (одинаковый для всех конфигураций — для воспроизводимости)")
    add(f"- **Судья (LLM-as-judge):** {raw.get('judge_model') or '—'}")
    add(f"- **Запросов:** {len(queries)}   **Конфигураций:** {len(configs)}")
    add("")

    # --- Конфигурации ---
    add("## 1. Конфигурации")
    add("")
    add("| Ключ | Модель | Тип | temp | num_predict | num_ctx | top_p | top_k | repeat_penalty |")
    add("|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        p = c["params"]
        kind = "Modelfile" if c["used_custom"] else "базовая"
        add(f"| {c['key']} | {c['model']} | {kind} | {p['temperature']} | "
            f"{p['num_predict']} | {p['num_ctx']} | {p['top_p']} | {p['top_k']} | {p['repeat_penalty']} |")
    add("")

    # --- Сводная таблица метрик ---
    add("## 2. Сводная таблица: конфигурация × метрика")
    add("")
    add("| Конфигурация | Запросов | Ср. токенов | Ср. ток/с | Мин ток/с | Макс ток/с | "
        "Ср. время, с | Загрузка, с | Ср. длина, симв. |")
    add("|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        a = aggs.get(c["key"], {})
        add(f"| {c['key']} | {a.get('queries', 0)} | {a.get('tokens_mean', '—')} | "
            f"{a.get('tps_mean', '—')} | {a.get('tps_min', '—')} | {a.get('tps_max', '—')} | "
            f"{a.get('wall_mean', '—')} | {a.get('load_max', '—')} | {a.get('chars_mean', '—')} |")
    add("")

    # --- Скорость и ресурсы ---
    add("## 3. Скорость и ресурсы")
    add("")
    add("| Конфигурация | VRAM пик, МБ | VRAM прирост, МБ | RAM пик, МБ | Стабильность ток/с (ср ± σ) |")
    add("|---|---|---|---|---|")
    for c in configs:
        res = c.get("resources", {})
        stab = c.get("stability") or {}
        stab_text = (f"{stab.get('mean')} ± {stab.get('std')} (n={stab.get('runs')})"
                     if stab else "—")
        add(f"| {c['key']} | {res.get('vram_peak_mb', '—')} | {res.get('vram_delta_mb', '—')} | "
            f"{res.get('ram_peak_mb', '—')} | {stab_text} |")
    add("")

    # --- Качество ---
    add("## 4. Качество (LLM-as-judge)")
    add("")
    if judge.get("summary"):
        add("| Конфигурация | Полнота | Корректность | Релевантность | Формат | Итоговая (0–5) |")
        add("|---|---|---|---|---|---|")
        for c in configs:
            key = c["key"]
            rub = _rubric_avg(judge, key)
            total = judge["summary"].get(key, {}).get("avg_total", "—")
            add(f"| {key} | {rub.get('completeness', '—')} | {rub.get('correctness', '—')} | "
                f"{rub.get('relevance', '—')} | {rub.get('format', '—')} | **{total}** |")
        add("")
        add("Попарное сравнение (оценка A : B):")
        add("")
        add("| Пара | Запрос | A | B | Обоснование |")
        add("|---|---|---|---|---|")
        for d in judge.get("details", []):
            a_key, b_key = d["pair"]
            reason = _trunc(d.get("reason", ""), 160).replace("\n", " ")
            add(f"| {a_key} vs {b_key} | {d['query']} | {d['A']['total']} | {d['B']['total']} | {reason} |")
    else:
        add("_(оценка судьи не выполнялась)_")
    add("")

    # --- Side-by-side ---
    add("## 5. Сравнение ответов (side-by-side)")
    add("")
    for q in queries:
        add(f"### {q['id']} — {q.get('category', '')}")
        add("")
        add(f"**Запрос:** {q['query']}")
        add("")
        if q.get("key_points"):
            add("_Ключевые пункты эталона:_ " + "; ".join(q["key_points"]))
            add("")
        for c in configs:
            r = c["results"].get(q["id"], {})
            add(f"<details><summary><b>{c['key']}</b> ({c['model']}) — "
                f"{r.get('eval_count', 0)} ток, {r.get('tokens_per_sec', 0)} ток/с, "
                f"{r.get('wall_time_s', 0)} с</summary>")
            add("")
            add(_trunc(r.get("answer", "") or (r.get("error") or "(нет ответа)")))
            add("")
            add("</details>")
            add("")

    # --- Выводы ---
    add("## 6. Выводы и рекомендации")
    add("")
    for line in _conclusions(raw):
        add(line)
    add("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _conclusions(raw: dict) -> list[str]:
    """Автоматические выводы на основе метрик."""
    configs = raw["configs"]
    aggs = raw["aggregates"]
    judge = (raw.get("judge") or {}).get("summary", {})
    notes: list[str] = []

    base = aggs.get("baseline", {})
    opt = aggs.get("optimized", {})

    if base and opt:
        if base.get("tps_mean") and opt.get("tps_mean"):
            diff = round((opt["tps_mean"] - base["tps_mean"]) / base["tps_mean"] * 100, 1)
            notes.append(f"- Скорость генерации (optimized vs baseline): {diff:+}% "
                         f"({base['tps_mean']} → {opt['tps_mean']} ток/с).")
        if base.get("wall_mean") and opt.get("wall_mean"):
            diff = round((opt["wall_mean"] - base["wall_mean"]) / base["wall_mean"] * 100, 1)
            notes.append(f"- Среднее время ответа: {diff:+}% "
                         f"({base['wall_mean']} → {opt['wall_mean']} с).")
        if base.get("chars_mean") and opt.get("chars_mean"):
            notes.append(f"- Средняя длина ответа: {base['chars_mean']} → {opt['chars_mean']} симв. "
                         "(короче = ближе к требованию «кратко и по делу»).")

    if judge:
        for key in ("baseline", "optimized", "qat"):
            if key in judge:
                notes.append(f"- Качество (судья) {key}: {judge[key].get('avg_total')} / 5.")
        if "baseline" in judge and "optimized" in judge:
            delta = round(judge["optimized"].get("avg_total", 0)
                          - judge["baseline"].get("avg_total", 0), 2)
            notes.append(f"- Прирост качества optimized vs baseline: {delta:+} балла (по 5-балльной шкале).")

    for c in configs:
        res = c.get("resources", {})
        if res.get("vram_peak_mb"):
            notes.append(f"- VRAM ({c['key']}): пик {res['vram_peak_mb']} МБ "
                         f"(прирост {res.get('vram_delta_mb')} МБ).")

    notes.append("")
    notes.append("**Рекомендации:**")
    notes.append("- Для целевой задачи (техподдержка 1С) использовать оптимизированную конфигурацию: "
                 "низкая температура (0.2) + структурированный системный промпт + ограничение длины ответа.")
    notes.append("- Квантование QAT (Q4_0) даёт меньший размер модели при близком качестве — "
                 "предпочтительно для экономии VRAM.")
    notes.append("- Для длинных диалогов поднимать num_ctx осознанно (растёт расход VRAM) — "
                 "задел на День 30 (лимиты и ресурсы).")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение «до/после» локальной LLM (День 29)")
    ap.add_argument("--profile", default=None, help="профиль задачи (по умолчанию из config)")
    ap.add_argument("--configs", default="baseline,optimized,qat",
                    help="конфигурации через запятую")
    ap.add_argument("--seed", type=int, default=42, help="зерно генерации")
    ap.add_argument("--queries", default=str(cfg.EVAL_QUERIES_FILE), help="файл с запросами")
    ap.add_argument("--judge-model", default="gpt-oss:20b", help="модель-судья")
    ap.add_argument("--no-judge", action="store_true", help="не запускать LLM-as-judge")
    ap.add_argument("--speed-repeat", type=int, default=3,
                    help="число повторов первого запроса для оценки стабильности")
    args = ap.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    profile_name = args.profile or cfg.TASK_PROFILE
    experiments = select_experiments(profile_name, args.configs.split(","))
    queries = load_queries(Path(args.queries))
    if not experiments or not queries:
        log("Нечего сравнивать: нет конфигураций или запросов.")
        return 1

    ollama = Ollama()
    if not ollama.is_available():
        log(cfg.OLLAMA_START_HINT)
        ollama.close()
        return 1
    available = ollama.list_models()

    log(f"Профиль: {profile_name}")
    log(f"Конфигурации: {', '.join(e.key for e in experiments)}")
    log(f"Запросов: {len(queries)}; seed={args.seed}; повтор для стабильности: {args.speed_repeat}")
    log(f"Доступные модели: {', '.join(available)}")

    config_runs: list[dict] = []
    for exp in experiments:
        log(f"\n=== Конфигурация: {exp.key} — {exp.title} ===")
        config_runs.append(
            run_config(exp, queries, ollama, args.seed, available, args.speed_repeat, log=log)
        )

    aggregates = {c["key"]: aggregate(c) for c in config_runs}

    judge_result = None
    if not args.no_judge:
        log("\n=== Оценка качества (LLM-as-judge) ===")
        ollama.unload_all()
        judge_result = run_judge(
            config_runs, queries, ollama, args.judge_model, args.seed, log=log
        )

    ollama.unload_all()
    ollama.close()

    raw = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile_name,
        "seed": args.seed,
        "judge_model": None if args.no_judge else args.judge_model,
        "queries": queries,
        "configs": config_runs,
        "aggregates": aggregates,
        "judge": judge_result,
    }

    ts = timestamp()
    raw_path = cfg.REPORTS_DIR / f"raw_{ts}.json"
    report_path = cfg.REPORTS_DIR / f"REPORT_{ts}.md"
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(raw_path, raw)
    build_report(raw, report_path)

    log(f"\nСырые данные: {raw_path}")
    log(f"Отчёт: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
