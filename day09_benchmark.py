"""
День 9: Декомпозиция инференса — бенчмарк всех 4 задач
Монолит vs Multi-stage, 4 модели, полные метрики
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config import load_config
from client import LLMClient

# ── Модели для теста ─────────────────────────────────────────────────────────
MODELS = {
    "weak_local":    "qwen3.5:4b",
    "strong_local":  "qwen3:14b",
    "medium_cloud":  "gpt-4o-mini",
    "strong_cloud":  "claude-sonnet-4-6",
}

# ── Входные данные для задач ──────────────────────────────────────────────────
RESUME_TEXT = """
Иван Петров. Опыт работы 6 лет в Python/Django, 2 года в FastAPI.
Образование: МГТУ им. Баумана, прикладная математика. Знает SQL, Docker, Redis.
Последнее место: Senior Backend Developer в СберТех (3 года).
Хобби: рыбалка, чтение sci-fi.
"""

TICKET_TEXT = """
Здравствуйте! Вчера вечером перестала работать оплата на сайте.
Пытался провести транзакцию на 15 000 руб., карта списала деньги,
но статус заказа так и остался "ожидание оплаты". Заказ #78432.
Уже звонил на горячую линию час назад, мне сказали перезвонят — не перезвонили.
"""

NEWS_TEXT = """
Яндекс сообщил о росте выручки на 38% по итогам Q1 2026 до 412 млрд руб.
Облачный сегмент вырос на 65%, рекламная выручка — на 28%.
Аналитики Goldman Sachs повысили целевую цену акций с $58 до $74.
Компания объявила о buyback на $500 млн.
"""

POST_TEXT = """
Всем привет! Наконец-то нашёл рабочую схему заработка — без вложений,
просто регистрируешься по моей ссылке и получаешь $200 в первый день!
Уже 500 человек заработали. Пиши в ЛС, расскажу как. ТОЛЬКО СЕГОДНЯ!!!
"""

# ── Структура результата ──────────────────────────────────────────────────────
@dataclass
class StageResult:
    name: str
    output: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None

@dataclass
class TaskResult:
    task: str
    model_key: str
    model_id: str
    mode: str  # "monolith" | "multistage"
    stages: list[StageResult] = field(default_factory=list)
    error: str | None = None

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.stages)

    @property
    def total_tokens(self) -> int | None:
        vals = [s.input_tokens for s in self.stages if s.input_tokens is not None]
        vals += [s.output_tokens for s in self.stages if s.output_tokens is not None]
        return sum(vals) if vals else None

    @property
    def total_cost_usd(self) -> float | None:
        vals = [s.cost_usd for s in self.stages if s.cost_usd is not None]
        return sum(vals) if vals else None


# ── Хелперы ───────────────────────────────────────────────────────────────────
async def call(client: LLMClient, model_id: str, messages: list[dict], stage_name: str) -> StageResult:
    text, metrics = await client.chat_with_metrics(model_id, messages)
    return StageResult(
        name=stage_name,
        output=text.strip()[:300],
        latency_ms=metrics.latency_ms,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        cost_usd=metrics.cost_usd,
    )

def sys(content: str) -> dict:
    return {"role": "system", "content": content}

def usr(content: str) -> dict:
    return {"role": "user", "content": content}


# ══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 1: Анализ резюме
# ══════════════════════════════════════════════════════════════════════════════

async def task1_monolith(client: LLMClient, model_id: str) -> list[StageResult]:
    messages = [
        sys("Ты HR-аналитик. Отвечай строго в JSON."),
        usr(f"""Проанализируй резюме и верни JSON:
{{"years_exp": int, "skills": [str], "education": str, "fit": "fit|partial|no_fit", "reason": str}}

Резюме: {RESUME_TEXT}

Вакансия: Senior Python Developer, 5+ лет опыта, знание FastAPI, PostgreSQL."""),
    ]
    return [await call(client, model_id, messages, "monolith")]


async def task1_multistage(client: LLMClient, model_id: str) -> list[StageResult]:
    stages = []

    # Этап 1: извлечение полей
    s1 = await call(client, model_id, [
        sys("Извлеки структурированные данные из резюме. Отвечай ТОЛЬКО JSON без пояснений."),
        usr(f"""Резюме: {RESUME_TEXT}

Верни JSON: {{"years_exp": int, "skills": [str], "education": str, "last_role": str}}"""),
    ], "extract_fields")
    stages.append(s1)

    # Этап 2: классификация соответствия
    s2 = await call(client, model_id, [
        sys("Ты рекрутер. Отвечай одним словом: fit | partial | no_fit"),
        usr(f"""Кандидат: {s1.output}

Вакансия: Senior Python Developer, 5+ лет, FastAPI, PostgreSQL.
Соответствие?"""),
    ], "classify_fit")
    stages.append(s2)

    # Этап 3: обоснование
    s3 = await call(client, model_id, [
        sys("Напиши 1-2 предложения обоснования для HR. Кратко и конкретно."),
        usr(f"Кандидат: {s1.output}\nВердикт: {s2.output}\nОбоснование:"),
    ], "generate_reason")
    stages.append(s3)

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 2: Триаж тикетов поддержки
# ══════════════════════════════════════════════════════════════════════════════

async def task2_monolith(client: LLMClient, model_id: str) -> list[StageResult]:
    messages = [
        sys("Ты оператор поддержки. Отвечай строго в JSON."),
        usr(f"""Обработай тикет. Верни JSON:
{{"category": str, "priority": "P1|P2|P3", "key_issue": str, "draft_reply": str}}

Тикет: {TICKET_TEXT}"""),
    ]
    return [await call(client, model_id, messages, "monolith")]


async def task2_multistage(client: LLMClient, model_id: str) -> list[StageResult]:
    stages = []

    s1 = await call(client, model_id, [
        sys("Нормализуй тикет поддержки. Верни JSON без пояснений."),
        usr(f"""Тикет: {TICKET_TEXT}

JSON: {{"language": str, "sentiment": "negative|neutral|positive", "key_issue": str, "order_id": str|null}}"""),
    ], "normalize")
    stages.append(s1)

    s2 = await call(client, model_id, [
        sys("Классифицируй тикет. Отвечай ТОЛЬКО JSON."),
        usr(f"""Нормализованный тикет: {s1.output}

JSON: {{"category": "payment|delivery|tech|account|other", "priority": "P1|P2|P3"}}
P1=критично (деньги списаны, сервис недоступен), P2=серьёзно, P3=вопрос"""),
    ], "classify")
    stages.append(s2)

    s3 = await call(client, model_id, [
        sys("Напиши черновик ответа клиенту. 2-3 предложения, вежливо, по существу."),
        usr(f"Тикет: {s1.output}\nКлассификация: {s2.output}\nЧерновик ответа:"),
    ], "draft_reply")
    stages.append(s3)

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 3: Анализ финансовой новости
# ══════════════════════════════════════════════════════════════════════════════

async def task3_monolith(client: LLMClient, model_id: str) -> list[StageResult]:
    messages = [
        sys("Ты финансовый аналитик. Отвечай строго в JSON."),
        usr(f"""Проанализируй новость. Верни JSON:
{{"company": str, "metrics": dict, "sentiment": "bullish|neutral|bearish", "signal": "buy|hold|sell", "rationale": str}}

Новость: {NEWS_TEXT}"""),
    ]
    return [await call(client, model_id, messages, "monolith")]


async def task3_multistage(client: LLMClient, model_id: str) -> list[StageResult]:
    stages = []

    s1 = await call(client, model_id, [
        sys("Извлеки сущности из финансовой новости. Только JSON."),
        usr(f"""Новость: {NEWS_TEXT}

JSON: {{"company": str, "metrics": {{"revenue_growth": str, "segment_growth": dict}}, "analyst_actions": [str], "corporate_actions": [str]}}"""),
    ], "extract_entities")
    stages.append(s1)

    s2 = await call(client, model_id, [
        sys("Оцени sentiment для трейдера. Одно слово: bullish | neutral | bearish"),
        usr(f"Данные по компании: {s1.output}"),
    ], "assess_sentiment")
    stages.append(s2)

    s3 = await call(client, model_id, [
        sys("Дай торговый сигнал: buy | hold | sell. Затем 1 предложение обоснования."),
        usr(f"Компания: {s1.output}\nSentiment: {s2.output}\nСигнал и обоснование:"),
    ], "trading_signal")
    stages.append(s3)

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 4: Модерация контента
# ══════════════════════════════════════════════════════════════════════════════

async def task4_monolith(client: LLMClient, model_id: str) -> list[StageResult]:
    messages = [
        sys("Ты модератор контента. Отвечай строго в JSON."),
        usr(f"""Проверь пост. Верни JSON:
{{"risk_level": "safe|suspicious|unsafe", "violations": [str], "action": "approve|warn|block", "reason": str}}

Пост: {POST_TEXT}"""),
    ]
    return [await call(client, model_id, messages, "monolith")]


async def task4_multistage(client: LLMClient, model_id: str) -> list[StageResult]:
    stages = []

    s1 = await call(client, model_id, [
        sys("Быстрый pre-filter. Одно слово: safe | suspicious | unsafe"),
        usr(f"Пост: {POST_TEXT}"),
    ], "prefilter")
    stages.append(s1)

    # Этап 2 только если не safe
    level = s1.output.lower().strip()
    if "safe" in level and "unsafe" not in level and "suspicious" not in level:
        s2 = StageResult("deep_analysis", "SKIPPED (safe)", 0, None, None, None)
        s3 = StageResult("generate_action", "approve — контент безопасен", 0, None, None, None)
        stages.extend([s2, s3])
        return stages

    s2 = await call(client, model_id, [
        sys("Детальный анализ нарушений. Верни JSON."),
        usr(f"""Пост: {POST_TEXT}
Pre-filter: {s1.output}

JSON: {{"violations": [str], "severity": "low|medium|high", "pattern": str}}"""),
    ], "deep_analysis")
    stages.append(s2)

    s3 = await call(client, model_id, [
        sys("Вынеси решение: approve | warn | block. Затем причина в 1 предложении."),
        usr(f"Анализ: {s2.output}\nРешение:"),
    ], "generate_action")
    stages.append(s3)

    return stages


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
    ("resume",     "Анализ резюме",              task1_monolith, task1_multistage),
    ("ticket",     "Триаж тикетов поддержки",    task2_monolith, task2_multistage),
    ("finance",    "Анализ финансовой новости",  task3_monolith, task3_multistage),
    ("moderation", "Модерация контента",         task4_monolith, task4_multistage),
]


async def run_all(client: LLMClient) -> list[TaskResult]:
    results = []

    for task_key, task_name, mono_fn, multi_fn in TASKS:
        for model_key, model_id in MODELS.items():
            print(f"  [{task_key}] {model_key} ({model_id})...", end=" ", flush=True)
            for mode, fn in [("monolith", mono_fn), ("multistage", multi_fn)]:
                tr = TaskResult(task=task_name, model_key=model_key, model_id=model_id, mode=mode)
                try:
                    tr.stages = await fn(client, model_id)
                except Exception as e:
                    tr.error = str(e)
                results.append(tr)
            print("OK")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def fmt_ms(ms: float) -> str:
    return f"{ms:>7.0f}ms"

def fmt_tok(n: int | None) -> str:
    return f"{n:>6}" if n is not None else "     ?"

def fmt_cost(c: float | None) -> str:
    if c is None:
        return "      ?"
    if c == 0:
        return "  $0.000"
    return f"${c:.5f}"

MODEL_LABELS = {
    "weak_local":   "Слабая локальная",
    "strong_local": "Мощная локальная",
    "medium_cloud": "Средняя облачная",
    "strong_cloud": "Мощная облачная ",
}

def print_report(results: list[TaskResult]):
    # Группируем по задаче
    tasks_seen = []
    for r in results:
        if r.task not in tasks_seen:
            tasks_seen.append(r.task)

    for task_name in tasks_seen:
        task_results = [r for r in results if r.task == task_name]
        print(f"\n{'═'*70}")
        print(f"  {task_name.upper()}")
        print(f"{'═'*70}")
        print(f"{'Модель':<20} {'Режим':<12} {'Latency':>9} {'Токены':>8} {'Стоимость':>10}")
        print(f"{'-'*60}")

        for model_key in MODELS:
            label = MODEL_LABELS.get(model_key, model_key)
            for mode in ["monolith", "multistage"]:
                tr = next((r for r in task_results if r.model_key == model_key and r.mode == mode), None)
                if tr is None:
                    continue
                if tr.error:
                    print(f"{label:<20} {mode:<12} ERROR: {tr.error[:40]}")
                    continue
                lat = fmt_ms(tr.total_latency_ms)
                tok = fmt_tok(tr.total_tokens)
                cost = fmt_cost(tr.total_cost_usd)
                stages_str = " → ".join(s.name for s in tr.stages)
                print(f"{label:<20} {mode:<12} {lat} {tok} {cost}")
                if mode == "multistage":
                    for s in tr.stages:
                        skip = " [SKIP]" if "SKIPPED" in s.output else ""
                        print(f"  {'':18} └─ {s.name:<20} {fmt_ms(s.latency_ms)} {fmt_tok(s.input_tokens)}{skip}")
            print()

    # Сводная таблица
    print(f"\n{'═'*70}")
    print("  СВОДНАЯ ТАБЛИЦА: МОНОЛИТ vs MULTI-STAGE")
    print(f"{'═'*70}")
    print(f"{'Задача':<26} {'Модель':<20} {'Моно ms':>8} {'Multi ms':>9} {'Δ':>7} {'Multi $':>9}")
    print(f"{'-'*75}")

    for task_name in tasks_seen:
        for model_key in MODELS:
            label = MODEL_LABELS.get(model_key, model_key)
            mono = next((r for r in results if r.task == task_name and r.model_key == model_key and r.mode == "monolith"), None)
            multi = next((r for r in results if r.task == task_name and r.model_key == model_key and r.mode == "multistage"), None)

            if mono and multi and not mono.error and not multi.error:
                mono_ms = mono.total_latency_ms
                multi_ms = multi.total_latency_ms
                delta = multi_ms - mono_ms
                delta_str = f"+{delta:.0f}" if delta > 0 else f"{delta:.0f}"
                cost_str = fmt_cost(multi.total_cost_usd)
                short_task = task_name[:24]
                print(f"{short_task:<26} {label:<20} {mono_ms:>7.0f}  {multi_ms:>8.0f}  {delta_str:>7} {cost_str:>9}")
        print()


async def main():
    config = load_config()
    client = LLMClient(config)

    print("День 9: Бенчмарк декомпозиции инференса")
    print(f"Моделей: {len(MODELS)} | Задач: {len(TASKS)} | Режимов: 2")
    print(f"Итого запросов: ~{len(MODELS) * len(TASKS) * (1 + 3) * 2 // 2}\n")

    t0 = time.perf_counter()
    results = await run_all(client)
    elapsed = time.perf_counter() - t0

    print_report(results)
    print(f"\nОбщее время выполнения: {elapsed:.1f}с")


if __name__ == "__main__":
    asyncio.run(main())
