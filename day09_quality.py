"""
День 9: Сравнение качества — Монолит vs Multi-stage
Захватываем полные выводы, судья — claude-sonnet-4-6
"""
import asyncio
import json
import textwrap

from config import load_config
from client import LLMClient

JUDGE_MODEL = "claude-opus-4-7"

MODELS = {
    "weak_local":    "qwen3.5:4b",
    "strong_local":  "qwen3:14b",
    "medium_cloud":  "gpt-4o-mini",
    "strong_cloud":  "claude-sonnet-4-6",
}
MODEL_LABELS = {
    "weak_local":   "Слабая локальная  (qwen3.5:4b)",
    "strong_local": "Мощная локальная  (qwen3:14b)",
    "medium_cloud": "Средняя облачная  (gpt-4o-mini)",
    "strong_cloud": "Мощная облачная   (claude-sonnet-4-6)",
}

# ── Входные данные ────────────────────────────────────────────────────────────
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

def usr(c): return {"role": "user", "content": c}
def sys(c): return {"role": "system", "content": c}

# ── Монолитные промпты ────────────────────────────────────────────────────────
async def mono_resume(client, model_id):
    return await client.chat(model_id, [
        sys("Ты HR-аналитик. Отвечай строго в JSON."),
        usr(f"""Проанализируй резюме и верни JSON:
{{"years_exp": int, "skills": [str], "education": str, "fit": "fit|partial|no_fit", "reason": str}}

Резюме: {RESUME_TEXT}
Вакансия: Senior Python Developer, 5+ лет опыта, знание FastAPI, PostgreSQL."""),
    ])

async def mono_ticket(client, model_id):
    return await client.chat(model_id, [
        sys("Ты оператор поддержки. Отвечай строго в JSON."),
        usr(f"""Обработай тикет. Верни JSON:
{{"category": str, "priority": "P1|P2|P3", "key_issue": str, "draft_reply": str}}

Тикет: {TICKET_TEXT}"""),
    ])

async def mono_finance(client, model_id):
    return await client.chat(model_id, [
        sys("Ты финансовый аналитик. Отвечай строго в JSON."),
        usr(f"""Проанализируй новость. Верни JSON:
{{"company": str, "metrics": dict, "sentiment": "bullish|neutral|bearish", "signal": "buy|hold|sell", "rationale": str}}

Новость: {NEWS_TEXT}"""),
    ])

async def mono_moderation(client, model_id):
    return await client.chat(model_id, [
        sys("Ты модератор контента. Отвечай строго в JSON."),
        usr(f"""Проверь пост. Верни JSON:
{{"risk_level": "safe|suspicious|unsafe", "violations": [str], "action": "approve|warn|block", "reason": str}}

Пост: {POST_TEXT}"""),
    ])

# ── Multi-stage пайплайны (возвращают финальный результат) ───────────────────
async def multi_resume(client, model_id):
    s1 = await client.chat(model_id, [
        sys("Извлеки структурированные данные из резюме. Отвечай ТОЛЬКО JSON без пояснений."),
        usr(f"Резюме: {RESUME_TEXT}\n\nJSON: {{\"years_exp\": int, \"skills\": [str], \"education\": str, \"last_role\": str}}"),
    ])
    s2 = await client.chat(model_id, [
        sys("Ты рекрутер. Отвечай одним словом: fit | partial | no_fit"),
        usr(f"Кандидат: {s1}\n\nВакансия: Senior Python Developer, 5+ лет, FastAPI, PostgreSQL.\nСоответствие?"),
    ])
    s3 = await client.chat(model_id, [
        sys("Напиши 1-2 предложения обоснования для HR. Кратко и конкретно."),
        usr(f"Кандидат: {s1}\nВердикт: {s2}\nОбоснование:"),
    ])
    return f"[Этап 1 — Извлечение]\n{s1}\n\n[Этап 2 — Классификация]\n{s2}\n\n[Этап 3 — Обоснование]\n{s3}"

async def multi_ticket(client, model_id):
    s1 = await client.chat(model_id, [
        sys("Нормализуй тикет поддержки. Верни JSON без пояснений."),
        usr(f'Тикет: {TICKET_TEXT}\n\nJSON: {{"language": str, "sentiment": str, "key_issue": str, "order_id": str}}'),
    ])
    s2 = await client.chat(model_id, [
        sys("Классифицируй тикет. Отвечай ТОЛЬКО JSON."),
        usr(f'Нормализованный тикет: {s1}\n\nJSON: {{"category": "payment|delivery|tech|account|other", "priority": "P1|P2|P3"}}\nP1=критично (деньги списаны), P2=серьёзно, P3=вопрос'),
    ])
    s3 = await client.chat(model_id, [
        sys("Напиши черновик ответа клиенту. 2-3 предложения, вежливо, по существу."),
        usr(f"Тикет: {s1}\nКлассификация: {s2}\nЧерновик ответа:"),
    ])
    return f"[Этап 1 — Нормализация]\n{s1}\n\n[Этап 2 — Классификация]\n{s2}\n\n[Этап 3 — Черновик]\n{s3}"

async def multi_finance(client, model_id):
    s1 = await client.chat(model_id, [
        sys("Извлеки сущности из финансовой новости. Только JSON."),
        usr(f'Новость: {NEWS_TEXT}\n\nJSON: {{"company": str, "metrics": dict, "analyst_actions": [str], "corporate_actions": [str]}}'),
    ])
    s2 = await client.chat(model_id, [
        sys("Оцени sentiment для трейдера. Одно слово: bullish | neutral | bearish"),
        usr(f"Данные по компании: {s1}"),
    ])
    s3 = await client.chat(model_id, [
        sys("Дай торговый сигнал: buy | hold | sell. Затем 1 предложение обоснования."),
        usr(f"Компания: {s1}\nSentiment: {s2}\nСигнал и обоснование:"),
    ])
    return f"[Этап 1 — Сущности]\n{s1}\n\n[Этап 2 — Sentiment]\n{s2}\n\n[Этап 3 — Сигнал]\n{s3}"

async def multi_moderation(client, model_id):
    s1 = await client.chat(model_id, [
        sys("Быстрый pre-filter. Одно слово: safe | suspicious | unsafe"),
        usr(f"Пост: {POST_TEXT}"),
    ])
    level = s1.lower().strip()
    if "safe" in level and "unsafe" not in level and "suspicious" not in level:
        return f"[Этап 1 — Pre-filter]\n{s1}\n\n[Этапы 2-3 — ПРОПУЩЕНЫ (safe)]\napprove — контент безопасен"
    s2 = await client.chat(model_id, [
        sys("Детальный анализ нарушений. Верни JSON."),
        usr(f'Пост: {POST_TEXT}\nPre-filter: {s1}\n\nJSON: {{"violations": [str], "severity": "low|medium|high", "pattern": str}}'),
    ])
    s3 = await client.chat(model_id, [
        sys("Вынеси решение: approve | warn | block. Затем причина в 1 предложении."),
        usr(f"Анализ: {s2}\nРешение:"),
    ])
    return f"[Этап 1 — Pre-filter]\n{s1}\n\n[Этап 2 — Детальный анализ]\n{s2}\n\n[Этап 3 — Решение]\n{s3}"

# ── LLM-судья ─────────────────────────────────────────────────────────────────
JUDGE_CRITERIA = """Оцени два ответа по 4 критериям (1-5 каждый):
1. Полнота — все ли требуемые поля/данные присутствуют?
2. Точность — насколько правильны факты и решения?
3. Формат — соблюдён ли требуемый формат (JSON, enum)?
4. Полезность — насколько ответ пригоден к использованию на практике?

Верни ТОЛЬКО JSON без пояснений:
{"monolith": {"completeness": 1-5, "accuracy": 1-5, "format": 1-5, "usefulness": 1-5, "verdict": str},
 "multistage": {"completeness": 1-5, "accuracy": 1-5, "format": 1-5, "usefulness": 1-5, "verdict": str},
 "winner": "monolith|multistage|tie",
 "key_difference": str}"""

async def judge(client, task_name, input_text, monolith_out, multistage_out):
    prompt = f"""Задача: {task_name}

Входные данные:
{input_text}

=== МОНОЛИТ ===
{monolith_out}

=== MULTI-STAGE ===
{multistage_out}

{JUDGE_CRITERIA}"""
    result = await client.chat(JUDGE_MODEL, [
        sys("Ты объективный судья качества LLM-ответов. Оценивай строго и честно."),
        usr(prompt),
    ])
    try:
        # Извлечь JSON даже если модель обернула его в ```
        import re
        m = re.search(r'\{.*\}', result, re.DOTALL)
        return json.loads(m.group()) if m else {"error": result[:200]}
    except Exception:
        return {"error": result[:200]}

# ── Главный цикл ──────────────────────────────────────────────────────────────
TASKS = [
    ("Анализ резюме",             RESUME_TEXT, mono_resume,      multi_resume),
    ("Триаж тикетов поддержки",   TICKET_TEXT, mono_ticket,      multi_ticket),
    ("Анализ финансовой новости",  NEWS_TEXT,  mono_finance,     multi_finance),
    ("Модерация контента",         POST_TEXT,  mono_moderation,  multi_moderation),
]

def wrap(text, width=72, indent="    "):
    lines = text.split("\n")
    result = []
    for line in lines:
        if len(line) <= width:
            result.append(indent + line)
        else:
            result.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent + "  "))
    return "\n".join(result)

async def main():
    config = load_config()
    client = LLMClient(config)

    all_scores = {}  # model_key -> {task -> {monolith_total, multi_total}}

    for model_key, model_id in MODELS.items():
        label = MODEL_LABELS[model_key]
        print(f"\n{'█'*70}")
        print(f"  МОДЕЛЬ: {label}")
        print(f"{'█'*70}")

        model_scores = {}

        for task_name, input_text, mono_fn, multi_fn in TASKS:
            print(f"\n  ── {task_name} ──")

            # Последовательно — Ollama не поддерживает параллельные запросы
            mono_out = await mono_fn(client, model_id)
            multi_out = await multi_fn(client, model_id)

            print(f"\n  [МОНОЛИТ]")
            print(wrap(mono_out.strip()))
            print(f"\n  [MULTI-STAGE]")
            print(wrap(multi_out.strip()))

            # Оцениваем судьёй
            print(f"\n  [ОЦЕНКА СУДЬИ]...")
            scores = await judge(client, task_name, input_text, mono_out, multi_out)

            if "error" in scores:
                print(f"  Ошибка судьи: {scores['error']}")
            else:
                m = scores.get("monolith", {})
                ms = scores.get("multistage", {})
                mono_total = sum([m.get("completeness",0), m.get("accuracy",0), m.get("format",0), m.get("usefulness",0)])
                multi_total = sum([ms.get("completeness",0), ms.get("accuracy",0), ms.get("format",0), ms.get("usefulness",0)])
                winner = scores.get("winner", "?")
                diff = scores.get("key_difference", "")

                print(f"  Монолит:    полнота={m.get('completeness')} точность={m.get('accuracy')} формат={m.get('format')} польза={m.get('usefulness')} → ИТОГО {mono_total}/20")
                print(f"  Multi-stage: полнота={ms.get('completeness')} точность={ms.get('accuracy')} формат={ms.get('format')} польза={ms.get('usefulness')} → ИТОГО {multi_total}/20")
                print(f"  Победитель: {winner.upper()}")
                print(f"  Ключевое различие: {diff}")

                model_scores[task_name] = {
                    "mono": mono_total, "multi": multi_total, "winner": winner
                }

        all_scores[model_key] = model_scores

    # Итоговая сводка
    print(f"\n\n{'═'*70}")
    print("  ИТОГОВАЯ СВОДКА КАЧЕСТВА")
    print(f"{'═'*70}")
    print(f"{'Модель':<28} {'Задача':<28} {'Моно':>5} {'Multi':>6} {'Победитель'}")
    print("-"*75)

    mono_wins = 0
    multi_wins = 0
    ties = 0

    for model_key, tasks in all_scores.items():
        label = MODEL_LABELS[model_key][:26]
        for task_name, s in tasks.items():
            short_task = task_name[:26]
            w = s["winner"]
            if w == "monolith": mono_wins += 1
            elif w == "multistage": multi_wins += 1
            else: ties += 1
            print(f"{label:<28} {short_task:<28} {s['mono']:>5} {s['multi']:>6}  {w}")
        print()

    total = mono_wins + multi_wins + ties
    print(f"\nМОНОЛИТ побед: {mono_wins}/{total}")
    print(f"MULTI-STAGE побед: {multi_wins}/{total}")
    print(f"НИЧЬИХ: {ties}/{total}")

if __name__ == "__main__":
    asyncio.run(main())
