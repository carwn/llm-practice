"""
Day 10: Micro-model first — двухуровневый инференс для классификации тикетов.

Level 1 — MicroClassifier:
  TF-IDF (1-2 gram) + LogisticRegression, обучен на train.jsonl (49 примеров)
  Latency: ~1-2 ms, нет API-вызовов, confidence через predict_proba
  Если confidence >= THRESHOLD → результат финальный (статус OK)
  Если confidence < THRESHOLD → статус UNSURE → уходит на Level 2

Level 2 — LLM fallback:
  gpt-4o-mini, вызывается только для пограничных случаев

Тест: 30 запросов (13 eval + 17 ручных: 6 простых / 6 пограничных / 5 сложных)
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from client import LLMClient
from config import load_config

# ── Config ──────────────────────────────────────────────────────────────────────

THRESHOLD      = 0.75
FALLBACK_MODEL = "gpt-4o-mini"
VALID_LABELS   = {"bug", "feature", "billing", "question", "other"}
TRAIN_PATH     = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
EVAL_PATH      = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"

_FALLBACK_SYSTEM = (
    "You are a support ticket classifier. "
    "Classify the ticket into exactly one of: bug, feature, billing, question, other.\n"
    "Return ONLY valid JSON with no other text: "
    '{"category": "<label>", "confidence": <0.0-1.0>}'
)

# ── Manual test cases ────────────────────────────────────────────────────────────
# 6 simple: очевидные → micro должна справиться
# 6 boundary: смешанные сигналы → micro может быть UNSURE
# 5 hard: сложные → скорее всего упадут на LLM

MANUAL_CASES: list[dict] = [
    # ── Simple (6) ──────────────────────────────────────────────────────────────
    {
        "ticket": (
            "Subject: Password reset broken\n\n"
            "The 'Forgot Password' link doesn't send any email. I checked spam. "
            "Tried 3 times today. Please fix."
        ),
        "expected": "bug",
        "label": "simple_bug_password",
    },
    {
        "ticket": (
            "Subject: Please add Slack notifications\n\n"
            "It would be great to get Slack alerts when a new order comes in. "
            "Would save us from constantly refreshing the dashboard."
        ),
        "expected": "feature",
        "label": "simple_feature_slack",
    },
    {
        "ticket": (
            "Subject: Double charged in April\n\n"
            "My credit card was charged twice on April 3rd — both charges are $29. "
            "Please refund the duplicate immediately."
        ),
        "expected": "billing",
        "label": "simple_billing_double",
    },
    {
        "ticket": (
            "Subject: Free tier API limit\n\n"
            "How many API calls can I make per month on the free plan? "
            "I couldn't find this in the docs."
        ),
        "expected": "question",
        "label": "simple_question_limit",
    },
    {
        "ticket": (
            "Subject: Enterprise training inquiry\n\n"
            "We're a team of 50 and interested in onboarding sessions for your platform. "
            "Do you offer corporate training packages?"
        ),
        "expected": "other",
        "label": "simple_other_training",
    },
    {
        "ticket": (
            "Subject: Images not loading in gallery\n\n"
            "All images show broken icons. Happens on Chrome/Mac, reproducible 100%. "
            "Started after yesterday's update."
        ),
        "expected": "bug",
        "label": "simple_bug_images",
    },
    # ── Boundary (6) ────────────────────────────────────────────────────────────
    {
        "ticket": (
            "Subject: Charged but account still expired\n\n"
            "You charged my card $49 last week but my account still shows as expired. "
            "I can't login. This is unacceptable."
        ),
        "expected": None,
        "label": "boundary_billing_bug",
    },
    {
        "ticket": (
            "Subject: How do I upgrade?\n\n"
            "I'd like to move from the free tier to Pro. "
            "What are the options and how much does it cost?"
        ),
        "expected": None,
        "label": "boundary_billing_question",
    },
    {
        "ticket": (
            "Subject: app crashin when upload foto!!\n\n"
            "hi pls help, my app keeeps crashing wenever i try upload foto from galery. "
            "tried restart stil same. plz fix asap!!!"
        ),
        "expected": "bug",
        "label": "boundary_noisy_typos",
    },
    {
        "ticket": (
            "Тема: Приложение постоянно зависает\n\n"
            "Добрый день, ваше приложение зависает при попытке открыть настройки. "
            "Пожалуйста, исправьте как можно скорее."
        ),
        "expected": "bug",
        "label": "boundary_russian",
    },
    {
        "ticket": (
            "Subject: CSV export broken + add Excel please\n\n"
            "The CSV export downloads but all date columns are empty. "
            "Also, while we're at it, could you add Excel (.xlsx) export? "
            "The CSV workaround is very painful."
        ),
        "expected": None,
        "label": "boundary_bug_feature_mix",
    },
    {
        "ticket": "It doesn't work.",
        "expected": None,
        "label": "boundary_ultra_terse",
    },
    # ── Hard (5) ────────────────────────────────────────────────────────────────
    {
        "ticket": (
            "Subject: Refund — feature never worked\n\n"
            "I signed up for Pro specifically for the AI summarisation feature. "
            "It has never worked — 'processing' forever. I've been charged 3 months. "
            "I want a refund for all three AND I expect the bug to be fixed."
        ),
        "expected": None,
        "label": "hard_refund_plus_bug",
    },
    {
        "ticket": (
            "Subject: Cancel my subscription\n\n"
            "I'd like to cancel immediately. "
            "Will I still have access until the end of the billing period, "
            "or does it cut off right away?"
        ),
        "expected": None,
        "label": "hard_cancel_question",
    },
    {
        "ticket": (
            "Subject: GDPR deletion — is there a fee?\n\n"
            "We need to submit a right-to-erasure request for an EU customer. "
            "Is there a charge for bulk data deletion? If so, how is it calculated?"
        ),
        "expected": None,
        "label": "hard_gdpr_billing_question",
    },
    {
        "ticket": (
            "Bonjour, j'ai un problème avec l'API.\n\n"
            "When I call POST /v2/documents with Content-Type: application/json "
            "I get error 422 Unprocessable Entity. Но в документации написано что "
            "этот endpoint принимает JSON. Ist das ein Bug oder mache ich etwas falsch?"
        ),
        "expected": None,
        "label": "hard_multilang_bug_question",
    },
    {
        "ticket": (
            "Subject: Great job breaking dark mode\n\n"
            "Congrats on the latest update! Dark mode now renders white text on white "
            "background in settings — very readable. Would love if you could not ship "
            "untested UI changes. Also please add a way to revert to the previous version."
        ),
        "expected": None,
        "label": "hard_sarcasm_bug_feature",
    },
]

# ── Data classes ─────────────────────────────────────────────────────────────────

@dataclass
class MicroResult:
    label: str
    confidence: float
    status: str        # "OK" | "UNSURE"
    latency_ms: float


@dataclass
class PipelineResult:
    ticket_label: str
    expected: str | None

    # Final answer
    category: str | None
    confidence: float | None
    level_used: int          # 1 = micro only, 2 = LLM fallback

    # Micro details
    micro_label: str
    micro_confidence: float
    micro_status: str

    # Timing
    micro_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float

    # Cost (micro = $0)
    llm_cost_usd: float

    # Accuracy (None if no ground truth)
    correct: bool | None


# ── MicroClassifier ──────────────────────────────────────────────────────────────

class MicroClassifier:
    def __init__(self, train_path: Path):
        texts, labels = self._load_jsonl(train_path)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
            ("lr",    LogisticRegression(max_iter=1000, C=1.5, class_weight="balanced")),
        ])
        self.pipeline.fit(texts, labels)
        self.classes_: list[str] = list(self.pipeline.classes_)
        self.train_size = len(texts)

    @staticmethod
    def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
        texts, labels = [], []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                msgs = ex["messages"]
                labels.append(next(m["content"] for m in msgs if m["role"] == "assistant"))
                texts.append(next(m["content"] for m in msgs if m["role"] == "user"))
        return texts, labels

    def predict(self, text: str) -> MicroResult:
        t0    = time.perf_counter()
        proba = self.pipeline.predict_proba([text])[0]
        lat   = (time.perf_counter() - t0) * 1000

        idx        = int(proba.argmax())
        label      = self.classes_[idx]
        confidence = float(proba[idx])
        status     = "OK" if confidence >= THRESHOLD else "UNSURE"
        return MicroResult(label=label, confidence=round(confidence, 4),
                           status=status, latency_ms=round(lat, 3))


# ── LLM fallback ─────────────────────────────────────────────────────────────────

async def llm_fallback(
    client: LLMClient, ticket: str
) -> tuple[str | None, float | None, float, float]:
    """Returns (label, confidence, latency_ms, cost_usd)."""
    messages = [
        {"role": "system", "content": _FALLBACK_SYSTEM},
        {"role": "user",   "content": ticket},
    ]
    text, m = await client.chat_with_metrics(
        FALLBACK_MODEL, messages,
        max_tokens=80,
        response_format={"type": "json_object"},
    )
    try:
        data  = json.loads(text)
        cat   = str(data.get("category", "")).strip().lower()
        conf  = float(data.get("confidence", 0.0))
        cat   = cat if cat in VALID_LABELS else None
    except Exception:
        cat, conf = None, 0.0

    return cat, round(conf, 4), m.latency_ms, m.cost_usd or 0.0


# ── Two-level pipeline ───────────────────────────────────────────────────────────

async def classify(
    micro:    MicroClassifier,
    client:   LLMClient,
    ticket:   str,
    expected: str | None,
    label:    str,
) -> PipelineResult:
    mr = micro.predict(ticket)

    if mr.status == "OK":
        return PipelineResult(
            ticket_label=label,
            expected=expected,
            category=mr.label,
            confidence=mr.confidence,
            level_used=1,
            micro_label=mr.label,
            micro_confidence=mr.confidence,
            micro_status=mr.status,
            micro_latency_ms=mr.latency_ms,
            llm_latency_ms=0.0,
            total_latency_ms=mr.latency_ms,
            llm_cost_usd=0.0,
            correct=(mr.label == expected) if expected is not None else None,
        )

    # Level 2 — LLM fallback
    llm_cat, llm_conf, llm_lat, llm_cost = await llm_fallback(client, ticket)
    final_cat  = llm_cat  if llm_cat  is not None else mr.label
    final_conf = llm_conf if llm_cat  is not None else mr.confidence

    return PipelineResult(
        ticket_label=label,
        expected=expected,
        category=final_cat,
        confidence=final_conf,
        level_used=2,
        micro_label=mr.label,
        micro_confidence=mr.confidence,
        micro_status=mr.status,
        micro_latency_ms=mr.latency_ms,
        llm_latency_ms=llm_lat,
        total_latency_ms=mr.latency_ms + llm_lat,
        llm_cost_usd=llm_cost,
        correct=(final_cat == expected) if expected is not None else None,
    )


# ── Data loaders ─────────────────────────────────────────────────────────────────

def load_eval_cases() -> list[dict]:
    cases = []
    with open(EVAL_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            ex    = json.loads(line)
            msgs  = ex["messages"]
            label = next(m["content"] for m in msgs if m["role"] == "assistant")
            text  = next(m["content"] for m in msgs if m["role"] == "user")
            cases.append({"ticket": text, "expected": label,
                          "label": f"eval_{label}_{len(cases)}"})
    return cases


# ── Report ───────────────────────────────────────────────────────────────────────

def print_report(results: list[PipelineResult], micro: MicroClassifier) -> None:
    W = 98
    n = len(results)

    at_micro = [r for r in results if r.level_used == 1]
    at_llm   = [r for r in results if r.level_used == 2]

    print("\n" + "═" * W)
    print("ДЕНЬ 10: MICRO-MODEL FIRST")
    print("═" * W)

    print(f"\nКонфигурация:")
    print(f"  Level 1  TF-IDF (1-2gram) + LogisticRegression  "
          f"(обучен на {micro.train_size} примерах из train.jsonl)")
    print(f"  Level 2  {FALLBACK_MODEL:<36} (LLM fallback)")
    print(f"  Порог уверенности: {THRESHOLD}")
    print(f"  Всего запросов:    {n}")

    # ── Distribution ──
    micro_pct = len(at_micro) / n * 100
    llm_pct   = len(at_llm)   / n * 100
    micro_lat_avg = (sum(r.total_latency_ms for r in at_micro) / len(at_micro)
                     if at_micro else 0.0)
    llm_lat_avg   = (sum(r.total_latency_ms for r in at_llm) / len(at_llm)
                     if at_llm else 0.0)
    all_lat_avg   = sum(r.total_latency_ms for r in results) / n

    print(f"\n{'─'*W}")
    print(f"  {'Уровень':<36} {'Обработано':>10} {'%':>6}  {'Avg lat(ms)':>12}  {'Cost($)':>10}")
    print(f"  {'─'*76}")
    print(f"  {'Level 1 (TF-IDF + LR)':<36} {len(at_micro):>10} {micro_pct:>5.1f}%"
          f"  {micro_lat_avg:>12.2f}  {'0.000000':>10}")
    print(f"  {'Level 2 (' + FALLBACK_MODEL + ')':<36} {len(at_llm):>10} {llm_pct:>5.1f}%"
          f"  {llm_lat_avg:>12.0f}  {sum(r.llm_cost_usd for r in at_llm):>10.6f}")
    print(f"  {'─'*76}")
    print(f"  {'ИТОГО':<36} {n:>10} {'100%':>6}  {all_lat_avg:>12.0f}  "
          f"{sum(r.llm_cost_usd for r in results):>10.6f}")

    # ── Cost savings vs always-LLM ──
    llm_cost_total = sum(r.llm_cost_usd for r in results)
    # Estimate: hypothetical cost if all 30 went to LLM
    # Use actual per-ticket LLM cost avg from at_llm cases
    if at_llm:
        avg_llm_cost = sum(r.llm_cost_usd for r in at_llm) / len(at_llm)
        baseline = avg_llm_cost * n
        savings  = (1 - llm_cost_total / baseline) * 100 if baseline else 0
        print(f"\n── Стоимость vs всегда-LLM ──────────────────────────────────────────────────")
        print(f"  LLM (baseline, все {n} запросов):  ${baseline:.6f}")
        print(f"  Фактически:                       ${llm_cost_total:.6f}")
        print(f"  Экономия:                          {savings:.1f}%  "
              f"({len(at_micro)}/{n} обработано без LLM)")

    # ── Accuracy ──
    with_gt = [r for r in results if r.expected is not None and r.category is not None]
    correct  = [r for r in with_gt if r.correct]

    gt_micro = [r for r in with_gt if r.level_used == 1]
    gt_llm   = [r for r in with_gt if r.level_used == 2]
    ok_micro = [r for r in gt_micro if r.correct]
    ok_llm   = [r for r in gt_llm   if r.correct]

    if with_gt:
        print(f"\n── Точность (ground truth: {len(with_gt)} из {n} тикетов) ───────────────────────")
        if gt_micro:
            print(f"  Level 1 (micro):  {len(ok_micro)}/{len(gt_micro)} "
                  f"({len(ok_micro)/len(gt_micro)*100:.0f}%)")
        if gt_llm:
            print(f"  Level 2 (LLM):    {len(ok_llm)}/{len(gt_llm)} "
                  f"({len(ok_llm)/len(gt_llm)*100:.0f}%)")
        print(f"  Итого:            {len(correct)}/{len(with_gt)} "
              f"({len(correct)/len(with_gt)*100:.0f}%)")

    # ── Micro confidence distribution ──
    confs_ok    = [r.micro_confidence for r in results if r.micro_status == "OK"]
    confs_unsure = [r.micro_confidence for r in results if r.micro_status == "UNSURE"]

    print(f"\n── Уверенность micro-model ─────────────────────────────────────────────────")
    print(f"  OK    ({len(confs_ok):>2} случаев): "
          f"avg={sum(confs_ok)/len(confs_ok):.3f}  "
          f"min={min(confs_ok):.3f}  max={max(confs_ok):.3f}" if confs_ok else "  OK: нет")
    if confs_unsure:
        print(f"  UNSURE ({len(confs_unsure):>2} случаев): "
              f"avg={sum(confs_unsure)/len(confs_unsure):.3f}  "
              f"min={min(confs_unsure):.3f}  max={max(confs_unsure):.3f}")

    # ── Per-ticket breakdown ──
    print(f"\n── Детализация по тикетам ──────────────────────────────────────────────────")
    print(f"  {'Label':<30} {'Lvl':>3}  {'Cat':<10} {'MicroConf':>9}  {'Lat(ms)':>8}  {'Correct':>7}")
    print(f"  {'─'*80}")

    # Sort: micro first, then LLM
    for r in sorted(results, key=lambda x: (x.level_used, x.ticket_label)):
        correct_s = ("✓" if r.correct else "✗") if r.correct is not None else "—"
        cat_s     = r.category or "—"
        print(f"  {r.ticket_label:<30} {'L' + str(r.level_used):>3}  {cat_s:<10} "
              f"{r.micro_confidence:>9.3f}  {r.total_latency_ms:>8.1f}  {correct_s:>7}")

    print("═" * W + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("День 10: Micro-model first")
    print("=" * 60)

    # Train micro-model
    print(f"\nОбучение MicroClassifier на {TRAIN_PATH.name}...")
    micro = MicroClassifier(TRAIN_PATH)
    print(f"  Классы: {micro.classes_}")
    print(f"  Примеров в train: {micro.train_size}")

    # Load all test cases
    eval_cases = load_eval_cases()
    all_cases  = eval_cases + MANUAL_CASES
    print(f"\nТест: {len(eval_cases)} eval + {len(MANUAL_CASES)} manual = {len(all_cases)} всего")
    print()

    config = load_config()
    client = LLMClient(config)

    results: list[PipelineResult] = []
    sem = asyncio.Semaphore(5)

    async def safe_classify(case: dict) -> PipelineResult:
        async with sem:
            return await classify(
                micro, client,
                case["ticket"],
                case.get("expected"),
                case["label"],
            )

    t0    = time.monotonic()
    tasks = [safe_classify(c) for c in all_cases]
    done  = 0

    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done += 1
        lvl_s  = f"L{r.level_used}"
        conf_s = f"{r.micro_confidence:.3f}"
        lat_s  = f"{r.total_latency_ms:.1f}ms"
        ok_s   = ("✓" if r.correct else "✗") if r.correct is not None else "—"
        print(f"  [{done:>2}/{len(all_cases)}] {r.ticket_label:<30} {lvl_s}  "
              f"micro={conf_s}  {lat_s}  {r.category or '?'}  {ok_s}")

    elapsed = time.monotonic() - t0
    print(f"\nВремя выполнения: {elapsed:.1f}с")

    print_report(results, micro)

    out_path = Path(__file__).resolve().parent.parent / "results" / "day10_micro_first.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
