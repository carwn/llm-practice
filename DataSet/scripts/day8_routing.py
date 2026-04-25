"""
Day 8: 3-level model routing with confidence-based escalation.

Routing chain:
  Tier 1: Local  (deepseek-r1:14b via Ollama)  — free, slowest
  Tier 2: Fast   (gpt-4o-mini)                 — cheap, fast
  Tier 3: Strong (claude-sonnet-4-6)            — expensive, best quality

Escalation heuristics:
  1. confidence < THRESHOLD  — self-reported score below 0.75
  2. parse_error             — model returned invalid JSON / unknown category
  3. short_response          — response under SHORT_THRESHOLD chars (garbage)

If Ollama is offline, routing starts at Tier 2.

Metrics reported:
  - latency per tier and total
  - token counts and cost per tier and total
  - cost vs "always-strong" baseline
  - escalation breakdown by reason and ticket type
"""

import asyncio
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from client import LLMClient
from config import load_config

# ── Configuration ──────────────────────────────────────────────────────────────

LOCAL_MODEL  = "deepseek-r1:14b"
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "claude-sonnet-4-6"

CONFIDENCE_THRESHOLD = 0.75
SHORT_THRESHOLD      = 5   # chars after stripping — treat as short_response

VALID_LABELS = {"bug", "feature", "billing", "question", "other"}

# Provider capabilities
JSON_MODE_SOURCES  = {"proxyapi-openai", "ollama"}
MAX_TOKENS_SOURCES = {"proxyapi-openai", "proxyapi-anthropic", "ollama"}
REASONING_SOURCES  = {"ollama"}

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a support ticket classifier. "
    "Classify the ticket into exactly one of: bug, feature, billing, question, other.\n"
    "Return ONLY valid JSON with no other text:\n"
    '{"category": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
)

# ── Edge cases ─────────────────────────────────────────────────────────────────

EDGE_CASES = [
    # ── original 6 ──────────────────────────────────────────────────────────────
    {
        "ticket": (
            "Subject: Charged but can't access my account\n\n"
            "You charged my credit card $49 last week but my account still shows as expired "
            "and I can't login. This is unacceptable."
        ),
        "expected": None,
        "label": "edge_billing_bug",
    },
    {
        "ticket": (
            "Subject: How do I upgrade my plan?\n\n"
            "I'd like to move from the free tier to a paid plan. "
            "What are the options and how much does it cost?"
        ),
        "expected": None,
        "label": "edge_billing_question",
    },
    {
        "ticket": (
            "Subject: app crashin when upload foto!!\n\n"
            "hi pls help, my app keeeps crashing wenever i try upload foto from galery. "
            "tried restart stil same. plz fix asap!!!"
        ),
        "expected": "bug",
        "label": "edge_noisy_typos",
    },
    {
        "ticket": (
            "Тема: Приложение постоянно зависает\n\n"
            "Добрый день, ваше приложение постоянно зависает при попытке открыть настройки "
            "аккаунта. Пожалуйста, исправьте это как можно скорее."
        ),
        "expected": "bug",
        "label": "edge_russian",
    },
    {
        "ticket": "Subject: \n\n",
        "expected": None,
        "label": "edge_empty",
    },
    {
        "ticket": "asdf qwer zxcv lkjh 1234 !!! @@@",
        "expected": None,
        "label": "edge_garbage",
    },
    # ── hard cases: designed to trigger low confidence ───────────────────────────
    {
        # bug report that transforms into a feature request mid-text
        "ticket": (
            "Subject: Export to CSV not working\n\n"
            "When I click Export → CSV the file downloads but all date columns are blank. "
            "Also, while we're at it, could you please add an Excel (.xlsx) export option? "
            "The CSV workaround is painful for our finance team."
        ),
        "expected": None,   # bug + feature equally present
        "label": "hard_bug_feature_mix",
    },
    {
        # stacktrace pasted inline — looks technical but user is asking a question
        "ticket": (
            "Subject: Getting 500 on /api/reports — is this expected?\n\n"
            "Hi, our monitoring started catching this error this morning:\n\n"
            "  TypeError: Cannot read properties of undefined (reading 'toFixed')\n"
            "      at ReportBuilder.format (report.js:142)\n\n"
            "Is this a known issue or are we doing something wrong in our integration? "
            "We haven't changed anything on our end."
        ),
        "expected": None,   # bug vs question — unclear who owns it
        "label": "hard_stacktrace_question",
    },
    {
        # refund + broken feature — billing or bug?
        "ticket": (
            "Subject: Refund request — feature never worked\n\n"
            "I signed up for the Pro plan specifically for the AI summarisation feature. "
            "It has never worked for me — I get 'processing' forever and nothing happens. "
            "I've been charged for 3 months. I want a refund for all three months "
            "and I expect the bug to be fixed too."
        ),
        "expected": None,   # billing + bug, equal weight
        "label": "hard_refund_bug",
    },
    {
        # ultra-terse, could be anything
        "ticket": "It doesn't work.",
        "expected": None,
        "label": "hard_ultra_terse",
    },
    {
        # sarcastic tone with mixed signals
        "ticket": (
            "Subject: Great job breaking the dark mode\n\n"
            "Congrats on the latest update! Dark mode now renders white text on white background "
            "in the settings panel — very readable. Would love it if you could *not* ship "
            "untested UI changes. Also, please add a way to revert to the previous version."
        ),
        "expected": None,   # bug + feature + sarcasm obscures category
        "label": "hard_sarcasm_bug_feature",
    },
    {
        # cancel request — billing or question?
        "ticket": (
            "Subject: Cancel my subscription\n\n"
            "I'd like to cancel my subscription effective immediately. "
            "Will I still have access until the end of the billing period, "
            "or does it cut off right away?"
        ),
        "expected": None,   # billing action + question about policy
        "label": "hard_cancel_question",
    },
    {
        # mixed languages + technical content
        "ticket": (
            "Bonjour, j'ai un problème avec l'API.\n\n"
            "When I call POST /v2/documents with Content-Type: application/json "
            "I get error 422 Unprocessable Entity. Но в документации написано что этот "
            "endpoint принимает JSON. Ist das ein Bug oder mache ich etwas falsch?"
        ),
        "expected": None,   # FR+EN+RU+DE, bug vs question
        "label": "hard_multilang_bug_question",
    },
    {
        # policy / compliance question that looks like billing
        "ticket": (
            "Subject: GDPR data deletion — do you charge for it?\n\n"
            "We need to submit a right-to-erasure request for one of our EU customers. "
            "I couldn't find pricing for bulk data deletion in your docs. "
            "Is there a fee, and if so how is it calculated?"
        ),
        "expected": None,   # billing vs question vs other (compliance)
        "label": "hard_gdpr_billing_question",
    },
]

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TierCall:
    model_id: str
    tier: int
    category: str | None
    confidence: float | None
    escalation_reason: str | None  # reason this tier triggered escalation (None = accepted)
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class RoutingResult:
    ticket_label: str
    expected: str | None

    tier_reached: int           # 1 | 2 | 3
    models_tried: list[str]
    category: str | None
    confidence: float | None
    escalated: bool
    escalation_reasons: list[str]  # chain of reasons that led to escalation

    # Per-tier latencies (ms)
    latency_tier1_ms: float
    latency_tier2_ms: float
    latency_tier3_ms: float
    total_latency_ms: float

    # Per-tier costs (USD)
    cost_tier1_usd: float
    cost_tier2_usd: float
    cost_tier3_usd: float
    total_cost_usd: float

    # Tokens
    input_tokens: int
    output_tokens: int

    calls_made: int

# ── Helpers ────────────────────────────────────────────────────────────────────

def check_ollama(host: str) -> bool:
    try:
        r = subprocess.run(
            ["curl", "-s", "--connect-timeout", "3", f"{host}/api/tags"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def extract_json(text: str) -> dict:
    text = strip_think(text)
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON in: {text[:200]!r}")


def load_eval_cases() -> list[dict]:
    path = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"
    cases = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            expected = next(m["content"] for m in msgs if m["role"] == "assistant")
            ticket   = next(m["content"] for m in msgs if m["role"] == "user")
            cases.append({"ticket": ticket, "expected": expected, "label": f"eval_{expected}"})
    return cases


def model_source(client: LLMClient, model_id: str) -> str:
    return client.registry.get(model_id).source

# ── Single-tier call ───────────────────────────────────────────────────────────

async def call_tier(client: LLMClient, model_id: str, ticket: str, tier: int) -> TierCall:
    source = model_source(client, model_id)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": ticket},
    ]
    kwargs: dict = {}
    if source in MAX_TOKENS_SOURCES:
        kwargs["max_tokens"] = 1500 if source in REASONING_SOURCES else 300
    if source in JSON_MODE_SOURCES:
        kwargs["response_format"] = {"type": "json_object"}

    text, m = await client.chat_with_metrics(model_id, messages, **kwargs)

    # Determine escalation reason
    raw_stripped = strip_think(text).strip()
    if len(raw_stripped) < SHORT_THRESHOLD:
        return TierCall(
            model_id=model_id, tier=tier,
            category=None, confidence=None,
            escalation_reason="short_response",
            latency_ms=m.latency_ms,
            input_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
            cost_usd=m.cost_usd or 0.0,
        )

    try:
        data = extract_json(text)
        cat  = str(data.get("category", "")).strip().lower()
        conf = float(data.get("confidence", 0.0))
        cat  = cat if cat in VALID_LABELS else None
    except Exception:
        return TierCall(
            model_id=model_id, tier=tier,
            category=None, confidence=None,
            escalation_reason="parse_error",
            latency_ms=m.latency_ms,
            input_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
            cost_usd=m.cost_usd or 0.0,
        )

    if cat is None:
        reason = "parse_error"
    elif conf < CONFIDENCE_THRESHOLD:
        reason = "low_confidence"
    else:
        reason = None  # accepted

    return TierCall(
        model_id=model_id, tier=tier,
        category=cat, confidence=round(conf, 3),
        escalation_reason=reason,
        latency_ms=m.latency_ms,
        input_tokens=m.input_tokens or 0,
        output_tokens=m.output_tokens or 0,
        cost_usd=m.cost_usd or 0.0,
    )

# ── Router ─────────────────────────────────────────────────────────────────────

async def route(
    client: LLMClient,
    ticket: str,
    expected: str | None,
    label: str,
    ollama_ok: bool,
) -> RoutingResult:
    tier_calls: list[TierCall] = []
    escalation_reasons: list[str] = []

    tiers: list[tuple[int, str]] = []
    if ollama_ok:
        tiers.append((1, LOCAL_MODEL))
    tiers.append((2, FAST_MODEL))
    tiers.append((3, STRONG_MODEL))

    final_call: TierCall | None = None

    for tier_num, model_id in tiers:
        tc = await call_tier(client, model_id, ticket, tier_num)
        tier_calls.append(tc)

        if tc.escalation_reason is not None:
            # Record why we escalate and continue to next tier
            escalation_reasons.append(tc.escalation_reason)
        else:
            # Accepted — stop here
            final_call = tc
            break

    if final_call is None:
        # All tiers exhausted — use last result regardless
        final_call = tier_calls[-1]

    # Aggregate metrics per tier
    def sum_tier(t: int) -> tuple[float, float, int, int]:
        calls = [c for c in tier_calls if c.tier == t]
        return (
            sum(c.latency_ms for c in calls),
            sum(c.cost_usd   for c in calls),
            sum(c.input_tokens  for c in calls),
            sum(c.output_tokens for c in calls),
        )

    lat1, cost1, in1, out1 = sum_tier(1)
    lat2, cost2, in2, out2 = sum_tier(2)
    lat3, cost3, in3, out3 = sum_tier(3)

    return RoutingResult(
        ticket_label=label,
        expected=expected,
        tier_reached=final_call.tier,
        models_tried=[c.model_id for c in tier_calls],
        category=final_call.category,
        confidence=final_call.confidence,
        escalated=len(tier_calls) > 1,
        escalation_reasons=escalation_reasons,
        latency_tier1_ms=round(lat1),
        latency_tier2_ms=round(lat2),
        latency_tier3_ms=round(lat3),
        total_latency_ms=round(lat1 + lat2 + lat3),
        cost_tier1_usd=round(cost1, 6),
        cost_tier2_usd=round(cost2, 6),
        cost_tier3_usd=round(cost3, 6),
        total_cost_usd=round(cost1 + cost2 + cost3, 6),
        input_tokens=in1 + in2 + in3,
        output_tokens=out1 + out2 + out3,
        calls_made=len(tier_calls),
    )

# ── Report ─────────────────────────────────────────────────────────────────────

def strong_only_cost(client: LLMClient, results: list[RoutingResult]) -> float:
    """Hypothetical cost if every request went straight to the strong model."""
    model = client.registry.get(STRONG_MODEL)
    total = 0.0
    for r in results:
        in_tok  = r.input_tokens  // r.calls_made if r.calls_made else 0
        out_tok = r.output_tokens // r.calls_made if r.calls_made else 0
        total += (in_tok * model.cost_input + out_tok * model.cost_output) / 1_000_000
    return round(total, 6)


def print_report(results: list[RoutingResult], client: LLMClient, ollama_ok: bool) -> None:
    W = 100
    total = len(results)
    accepted_at = Counter(r.tier_reached for r in results)
    escalated   = sum(1 for r in results if r.escalated)

    print("\n" + "═" * W)
    print("ДЕНЬ 8: ROUTING МЕЖДУ МОДЕЛЯМИ")
    print("═" * W)

    tiers_info = []
    if ollama_ok:
        tiers_info.append((1, LOCAL_MODEL,  "Локальная"))
    tiers_info.append((2, FAST_MODEL,   "Быстрая облачная"))
    tiers_info.append((3, STRONG_MODEL, "Мощная облачная"))

    print(f"\nЦепочка маршрутизации:")
    for tier, model, label in tiers_info:
        print(f"  Tier {tier}: {label:<20} {model}")
    print(f"  Порог уверенности: {CONFIDENCE_THRESHOLD}")

    # ── Tier distribution ──
    print(f"\n{'─'*W}")
    print(f"  {'Tier':<6} {'Модель':<26} {'Принято':>8} {'%':>6} {'Avg lat(ms)':>12} {'Cost($)':>10}")
    print(f"  {'─'*70}")
    for tier, model, label in tiers_info:
        at_tier  = [r for r in results if r.tier_reached == tier]
        n        = len(at_tier)
        pct      = n / total * 100 if total else 0
        lats     = [r.total_latency_ms for r in at_tier]
        avg_lat  = sum(lats) / len(lats) if lats else 0
        costs    = [r.total_cost_usd for r in at_tier]
        tot_cost = sum(costs)
        print(f"  {tier:<6} {model:<26} {n:>8} {pct:>5.1f}% {avg_lat:>12.0f} {tot_cost:>10.6f}")

    print(f"  {'─'*70}")
    print(f"  {'ИТОГО':<33} {total:>8} {'100%':>6} "
          f"{sum(r.total_latency_ms for r in results)/total:>12.0f} "
          f"{sum(r.total_cost_usd for r in results):>10.6f}")

    # ── Escalation reasons ──
    reason_counter: Counter = Counter()
    for r in results:
        for reason in r.escalation_reasons:
            reason_counter[reason] += 1

    if reason_counter:
        print(f"\n── Причины эскалации ──────────────────────────────────────────")
        for reason, cnt in reason_counter.most_common():
            print(f"  {reason:<20} {cnt:>4}  ({cnt/total*100:.1f}% запросов)")

    # ── Cost comparison ──
    routing_cost   = sum(r.total_cost_usd for r in results)
    baseline_cost  = strong_only_cost(client, results)
    savings        = (1 - routing_cost / baseline_cost) * 100 if baseline_cost else 0

    print(f"\n── Стоимость ──────────────────────────────────────────────────")
    print(f"  Routing (фактически):       ${routing_cost:.6f}")
    print(f"  Baseline (всегда сильная):  ${baseline_cost:.6f}")
    print(f"  Экономия:                   {savings:.1f}%")

    # ── Latency breakdown ──
    lat_t1 = [r.latency_tier1_ms for r in results if r.latency_tier1_ms > 0]
    lat_t2 = [r.latency_tier2_ms for r in results if r.latency_tier2_ms > 0]
    lat_t3 = [r.latency_tier3_ms for r in results if r.latency_tier3_ms > 0]
    lat_all = [r.total_latency_ms for r in results]

    print(f"\n── Задержка (мс) ──────────────────────────────────────────────")
    print(f"  {'Слой':<30} {'avg':>8} {'min':>8} {'max':>8}")
    print(f"  {'─'*58}")
    for label_s, lat_list in [
        (f"Tier 1 ({LOCAL_MODEL})",  lat_t1),
        (f"Tier 2 ({FAST_MODEL})",   lat_t2),
        (f"Tier 3 ({STRONG_MODEL})", lat_t3),
        ("Итого на запрос",          lat_all),
    ]:
        if not lat_list:
            continue
        print(f"  {label_s:<30} {sum(lat_list)/len(lat_list):>8.0f} "
              f"{min(lat_list):>8.0f} {max(lat_list):>8.0f}")

    # ── Accuracy ──
    with_gt = [r for r in results if r.expected is not None and r.category is not None]
    correct = [r for r in with_gt if r.category == r.expected]
    if with_gt:
        print(f"\n── Точность ────────────────────────────────────────────────────")
        print(f"  Всего с ground truth: {len(with_gt)}")
        print(f"  Верных:               {len(correct)} ({len(correct)/len(with_gt)*100:.1f}%)")

        for tier, model, _ in tiers_info:
            tier_gt   = [r for r in with_gt if r.tier_reached == tier]
            tier_ok   = [r for r in tier_gt  if r.category == r.expected]
            if tier_gt:
                print(f"    Tier {tier} ({model:<22}): {len(tier_ok)}/{len(tier_gt)} "
                      f"({len(tier_ok)/len(tier_gt)*100:.1f}%)")

    # ── Per-label breakdown ──
    print(f"\n── Детализация по тикетам ─────────────────────────────────────")
    print(f"  {'Label':<28} {'Tier':>4} {'Cat':<10} {'Conf':>5} {'Lat(ms)':>8} {'Cost':>10}  Эскалация")
    print(f"  {'─'*85}")
    for r in sorted(results, key=lambda x: (x.tier_reached, x.ticket_label)):
        esc_str = " → ".join(r.escalation_reasons) if r.escalation_reasons else "—"
        conf_s  = f"{r.confidence:.2f}" if r.confidence is not None else "—   "
        cat_s   = r.category or "—"
        print(f"  {r.ticket_label:<28} {r.tier_reached:>4} {cat_s:<10} {conf_s:>5} "
              f"{r.total_latency_ms:>8} {r.total_cost_usd:>10.6f}  {esc_str}")

    print("═" * W + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    config    = load_config()
    ollama_ok = check_ollama(config.ollama_host)

    print("=" * 60)
    print("День 8: Routing между моделями")
    print("=" * 60)
    print(f"Ollama: {'доступен ✓' if ollama_ok else 'недоступен — старт с Tier 2'}")
    print(f"Tier 1: {LOCAL_MODEL}  (local)")
    print(f"Tier 2: {FAST_MODEL}  (fast cloud)")
    print(f"Tier 3: {STRONG_MODEL}  (strong cloud)")
    print(f"Порог:  confidence < {CONFIDENCE_THRESHOLD}")
    print()

    eval_cases = load_eval_cases()
    all_cases  = eval_cases + EDGE_CASES
    print(f"Случаев: {len(eval_cases)} eval + {len(EDGE_CASES)} edge = {len(all_cases)}")
    print()

    client  = LLMClient(config)
    results: list[RoutingResult] = []

    sem = asyncio.Semaphore(4)

    async def safe_route(case: dict) -> RoutingResult:
        async with sem:
            try:
                return await route(
                    client,
                    case["ticket"],
                    case.get("expected"),
                    case.get("label", "?"),
                    ollama_ok,
                )
            except Exception as e:
                return RoutingResult(
                    ticket_label=case.get("label", "?"),
                    expected=case.get("expected"),
                    tier_reached=0, models_tried=[], category=None, confidence=None,
                    escalated=False, escalation_reasons=[f"error:{e}"],
                    latency_tier1_ms=0, latency_tier2_ms=0, latency_tier3_ms=0,
                    total_latency_ms=0,
                    cost_tier1_usd=0, cost_tier2_usd=0, cost_tier3_usd=0,
                    total_cost_usd=0, input_tokens=0, output_tokens=0, calls_made=0,
                )

    t0     = time.monotonic()
    tasks  = [safe_route(c) for c in all_cases]
    done   = 0

    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done += 1
        tier_s = f"T{r.tier_reached}" if r.tier_reached else "ERR"
        print(f"  [{done:>2}/{len(all_cases)}] {r.ticket_label:<28} {tier_s}  "
              f"conf={r.confidence or 0:.2f}  {r.total_latency_ms}ms  "
              f"${r.total_cost_usd:.6f}")

    elapsed = time.monotonic() - t0
    print(f"\nВсего: {elapsed:.1f}с")

    print_report(results, client, ollama_ok)

    out_path = Path(__file__).resolve().parent.parent / "results" / "day8_routing.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
