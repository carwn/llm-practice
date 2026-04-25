"""
Day 7: Confidence evaluation and inference quality control.

Approaches:
  1. Scoring   — one call, JSON with confidence score, retry if < threshold
  2. Redundancy — three parallel calls, majority vote

Models: gpt-4o-mini, gemini-2.0-flash, claude-haiku-4-5, deepseek-r1:14b (Ollama if online)
Data  : 13 eval.jsonl + 6 handcrafted edge cases
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

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_LABELS = {"bug", "feature", "billing", "question", "other"}
CONFIDENCE_THRESHOLD = 0.75
REDUNDANCY_N = 3
REDUNDANCY_TEMP = 0.4   # not applied to Google (requires config object)

MODELS = [
    "gpt-4o-mini",
    "gemini-2.0-flash",
    "claude-haiku-4-5",
    "deepseek-r1:14b",   # Ollama — skipped if offline
]

# Sources that accept response_format={"type":"json_object"}
JSON_MODE_SOURCES = {"proxyapi-openai", "ollama"}
# Sources that accept temperature as a flat kwarg
TEMP_SOURCES = {"proxyapi-openai", "proxyapi-anthropic", "ollama"}
# Sources that accept max_tokens as a flat kwarg (Google uses config object)
MAX_TOKENS_SOURCES = {"proxyapi-openai", "proxyapi-anthropic", "ollama"}
# Reasoning models need more tokens for thinking blocks
REASONING_SOURCES = {"ollama"}

# ── Edge cases ─────────────────────────────────────────────────────────────────

EDGE_CASES = [
    {
        "ticket": (
            "Subject: Charged but can't access my account\n\n"
            "You charged my credit card $49 last week but my account still shows as expired "
            "and I can't login. This is unacceptable."
        ),
        "expected": None,   # intentionally ambiguous: billing + bug
        "label": "edge_billing_bug",
    },
    {
        "ticket": (
            "Subject: How do I upgrade my plan?\n\n"
            "I'd like to move from the free tier to a paid plan. "
            "What are the options and how much does it cost?"
        ),
        "expected": None,   # billing vs question
        "label": "edge_billing_question",
    },
    {
        "ticket": (
            "Subject: app crashin when upload foto!!\n\n"
            "hi pls help, my app keeeps crashing wenever i try upload foto from galery. "
            "tried restart stil same. plz fix asap!!!"
        ),
        "expected": "bug",  # noisy but clear
        "label": "edge_noisy_typos",
    },
    {
        "ticket": (
            "Тема: Приложение постоянно зависает\n\n"
            "Добрый день, ваше приложение постоянно зависает при попытке открыть настройки "
            "аккаунта. Пожалуйста, исправьте это как можно скорее."
        ),
        "expected": "bug",  # Russian — model should still classify correctly
        "label": "edge_russian",
    },
    {
        "ticket": "Subject: \n\n",
        "expected": None,   # empty body
        "label": "edge_empty",
    },
    {
        "ticket": "asdf qwer zxcv lkjh 1234 !!! @@@",
        "expected": None,   # random garbage
        "label": "edge_garbage",
    },
]

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    model_id: str
    approach: str       # "scoring" | "redundancy"
    ticket_label: str
    expected: str | None
    category: str | None
    status: str         # "ACCEPT" | "UNSURE" | "FAIL"
    confidence: float | None    # scoring only
    agreement_pct: float | None # redundancy only
    calls_made: int
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    retried: bool = False

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
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def extract_json(text: str) -> dict:
    """Parse JSON from model response, tolerating markdown fences and think tags."""
    text = strip_think(text)
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group())
    raise ValueError(f"No JSON found in: {text[:200]!r}")


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
            ticket = next(m["content"] for m in msgs if m["role"] == "user")
            cases.append({
                "ticket": ticket,
                "expected": expected,
                "label": f"eval_{expected}",
            })
    return cases


def model_source(client: LLMClient, model_id: str) -> str:
    return client.registry.get(model_id).source

# ── Scoring approach ───────────────────────────────────────────────────────────

_SCORING_SYSTEM = (
    "You are a support ticket classifier. "
    "Classify the ticket into exactly one of: bug, feature, billing, question, other.\n"
    "Return ONLY valid JSON with no other text:\n"
    '{"category": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
)


async def _one_scoring_call(
    client: LLMClient, model_id: str, ticket: str, source: str
) -> tuple[str | None, float | None, float, int, int, float]:
    """Returns (category, confidence, latency_ms, in_tok, out_tok, cost_usd)."""
    messages = [
        {"role": "system", "content": _SCORING_SYSTEM},
        {"role": "user",   "content": ticket},
    ]
    kwargs: dict = {}
    if source in MAX_TOKENS_SOURCES:
        kwargs["max_tokens"] = 1500 if source in REASONING_SOURCES else 300
    if source in JSON_MODE_SOURCES:
        kwargs["response_format"] = {"type": "json_object"}

    text, m = await client.chat_with_metrics(model_id, messages, **kwargs)
    try:
        data = extract_json(text)
        cat = str(data.get("category", "")).strip().lower()
        conf = float(data.get("confidence", 0.0))
        cat = cat if cat in VALID_LABELS else None
    except Exception:
        cat, conf = None, None

    return cat, conf, m.latency_ms, m.input_tokens or 0, m.output_tokens or 0, m.cost_usd or 0.0


async def scoring_approach(
    client: LLMClient, model_id: str, ticket: str, expected: str | None, label: str
) -> InferenceResult:
    source = model_source(client, model_id)
    total_lat = 0.0
    total_in = total_out = 0
    total_cost = 0.0
    retried = False

    cat, conf, lat, in_t, out_t, cost = await _one_scoring_call(client, model_id, ticket, source)
    total_lat += lat; total_in += in_t; total_out += out_t; total_cost += cost
    calls = 1

    if cat is None or conf is None or conf < CONFIDENCE_THRESHOLD:
        retried = True
        cat2, conf2, lat2, in_t2, out_t2, cost2 = await _one_scoring_call(client, model_id, ticket, source)
        total_lat += lat2; total_in += in_t2; total_out += out_t2; total_cost += cost2
        calls = 2
        # Accept retry result if it is better
        if cat2 and conf2 is not None and conf2 >= CONFIDENCE_THRESHOLD:
            cat, conf = cat2, conf2
        elif conf2 is not None and (conf is None or conf2 > conf):
            cat, conf = cat2, conf2

    if cat and conf is not None and conf >= CONFIDENCE_THRESHOLD:
        status = "ACCEPT"
    elif cat and conf is not None and conf >= 0.5:
        status = "UNSURE"
    else:
        status = "FAIL"

    return InferenceResult(
        model_id=model_id, approach="scoring", ticket_label=label,
        expected=expected, category=cat, status=status,
        confidence=round(conf, 3) if conf is not None else None,
        agreement_pct=None,
        calls_made=calls, total_latency_ms=round(total_lat),
        input_tokens=total_in, output_tokens=total_out,
        cost_usd=round(total_cost, 6), retried=retried,
    )

# ── Redundancy approach ────────────────────────────────────────────────────────

_REDUNDANCY_SYSTEM = (
    "You are a support ticket classifier. "
    "Classify into exactly one of: bug, feature, billing, question, other. "
    "Reply with a single word — the category name only."
)


async def _one_redundancy_call(
    client: LLMClient, model_id: str, ticket: str, source: str
) -> tuple[str | None, float, int, int, float]:
    messages = [
        {"role": "system", "content": _REDUNDANCY_SYSTEM},
        {"role": "user",   "content": ticket},
    ]
    kwargs: dict = {}
    if source in MAX_TOKENS_SOURCES:
        kwargs["max_tokens"] = 1500 if source in REASONING_SOURCES else 10
    if source in TEMP_SOURCES:
        kwargs["temperature"] = REDUNDANCY_TEMP

    text, m = await client.chat_with_metrics(model_id, messages, **kwargs)
    text = strip_think(text)
    word = text.strip().lower().split()[0].strip(".,!?;:") if text.strip() else None
    word = word if word in VALID_LABELS else None
    return word, m.latency_ms, m.input_tokens or 0, m.output_tokens or 0, m.cost_usd or 0.0


async def redundancy_approach(
    client: LLMClient, model_id: str, ticket: str, expected: str | None, label: str
) -> InferenceResult:
    source = model_source(client, model_id)
    tasks = [_one_redundancy_call(client, model_id, ticket, source) for _ in range(REDUNDANCY_N)]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    total_lat = 0.0; total_in = total_out = 0; total_cost = 0.0
    votes: list[str] = []
    for r in raw:
        if isinstance(r, Exception):
            continue
        word, lat, in_t, out_t, cost = r
        total_lat += lat; total_in += in_t; total_out += out_t; total_cost += cost
        if word:
            votes.append(word)

    if not votes:
        return InferenceResult(
            model_id=model_id, approach="redundancy", ticket_label=label,
            expected=expected, category=None, status="FAIL",
            confidence=None, agreement_pct=0.0,
            calls_made=REDUNDANCY_N, total_latency_ms=round(total_lat),
            input_tokens=total_in, output_tokens=total_out,
            cost_usd=round(total_cost, 6),
        )

    counts = Counter(votes)
    winner, top_count = counts.most_common(1)[0]
    agreement_pct = round(top_count / REDUNDANCY_N * 100, 1)

    if top_count >= 2:   # 2/3 or 3/3 → accept
        status = "ACCEPT"
        category = winner
    else:                # all three different → fail
        status = "FAIL"
        category = None

    return InferenceResult(
        model_id=model_id, approach="redundancy", ticket_label=label,
        expected=expected, category=category, status=status,
        confidence=None, agreement_pct=agreement_pct,
        calls_made=REDUNDANCY_N, total_latency_ms=round(total_lat),
        input_tokens=total_in, output_tokens=total_out,
        cost_usd=round(total_cost, 6),
    )

# ── Run ────────────────────────────────────────────────────────────────────────

async def run_model(client: LLMClient, model_id: str, cases: list[dict]) -> list[InferenceResult]:
    sem = asyncio.Semaphore(4)   # max 4 concurrent outer tasks per model

    async def safe_scoring(case: dict) -> InferenceResult:
        async with sem:
            try:
                return await scoring_approach(
                    client, model_id, case["ticket"], case.get("expected"), case.get("label", "?")
                )
            except Exception as e:
                return InferenceResult(
                    model_id=model_id, approach="scoring", ticket_label=case.get("label", "?"),
                    expected=case.get("expected"), category=None, status="FAIL",
                    confidence=None, agreement_pct=None,
                    calls_made=0, total_latency_ms=0, input_tokens=0, output_tokens=0,
                    cost_usd=0.0,
                )

    async def safe_redundancy(case: dict) -> InferenceResult:
        async with sem:
            try:
                return await redundancy_approach(
                    client, model_id, case["ticket"], case.get("expected"), case.get("label", "?")
                )
            except Exception as e:
                return InferenceResult(
                    model_id=model_id, approach="redundancy", ticket_label=case.get("label", "?"),
                    expected=case.get("expected"), category=None, status="FAIL",
                    confidence=None, agreement_pct=None,
                    calls_made=0, total_latency_ms=0, input_tokens=0, output_tokens=0,
                    cost_usd=0.0,
                )

    scoring_tasks   = [safe_scoring(c)    for c in cases]
    redundancy_tasks = [safe_redundancy(c) for c in cases]
    results = await asyncio.gather(*scoring_tasks, *redundancy_tasks)
    return list(results)

# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(all_results: list[InferenceResult]) -> None:
    groups: dict[tuple, list[InferenceResult]] = defaultdict(list)
    for r in all_results:
        groups[(r.model_id, r.approach)].append(r)

    W = 116
    print("\n" + "═" * W)
    print(
        f"{'Model':<22} {'Approach':<11} {'Total':>5} {'Accept':>6} "
        f"{'Reject':>6} {'Retry':>5} {'Accuracy':>8} {'Lat(ms)':>8} "
        f"{'In tok':>7} {'Out tok':>7} {'Cost($)':>9}"
    )
    print("─" * W)

    sum_total = sum_accept = sum_reject = sum_retry = 0
    sum_in = sum_out = 0
    sum_cost = sum_lat = 0.0

    for (model_id, approach), results in sorted(groups.items()):
        total   = len(results)
        accepted = [r for r in results if r.status == "ACCEPT"]
        rejected = [r for r in results if r.status != "ACCEPT"]
        retried  = sum(1 for r in results if r.retried)

        with_gt  = [r for r in accepted if r.expected is not None and r.category is not None]
        accuracy = (sum(1 for r in with_gt if r.category == r.expected) / len(with_gt) * 100
                    if with_gt else float("nan"))

        avg_lat  = sum(r.total_latency_ms for r in results) / total
        tot_in   = sum(r.input_tokens  for r in results)
        tot_out  = sum(r.output_tokens for r in results)
        cost     = sum(r.cost_usd      for r in results)

        acc_str  = f"{accuracy:.1f}%" if not isnan(accuracy) else "—"
        print(
            f"{model_id:<22} {approach:<11} {total:>5} {len(accepted):>6} "
            f"{len(rejected):>6} {retried:>5} {acc_str:>8} {avg_lat:>8.0f} "
            f"{tot_in:>7} {tot_out:>7} {cost:>9.5f}"
        )

        sum_total += total; sum_accept += len(accepted); sum_reject += len(rejected)
        sum_retry += retried; sum_in += tot_in; sum_out += tot_out
        sum_cost  += cost;    sum_lat += sum(r.total_latency_ms for r in results)

    avg_lat_all = sum_lat / sum_total if sum_total else 0
    print("─" * W)
    print(
        f"{'TOTAL':<22} {'':<11} {sum_total:>5} {sum_accept:>6} "
        f"{sum_reject:>6} {sum_retry:>5} {'':>8} {avg_lat_all:>8.0f} "
        f"{sum_in:>7} {sum_out:>7} {sum_cost:>9.5f}"
    )
    print("═" * W)

    # Per-approach summary
    print("\n── Детализация по подходам ───────────────────────────────────────────────────")
    for approach in ("scoring", "redundancy"):
        rows = [r for r in all_results if r.approach == approach]
        if not rows:
            continue
        accept = [r for r in rows if r.status == "ACCEPT"]
        fail   = [r for r in rows if r.status == "FAIL"]
        unsure = [r for r in rows if r.status == "UNSURE"]
        gt_ok  = [r for r in accept if r.expected and r.category == r.expected]
        gt_all = [r for r in accept if r.expected]
        acc    = len(gt_ok) / len(gt_all) * 100 if gt_all else float("nan")
        acc_s  = f"{acc:.1f}%" if not isnan(acc) else "—"

        print(f"\n  {approach.upper()}")
        print(f"    Принято  : {len(accept)}/{len(rows)} ({len(accept)/len(rows)*100:.1f}%)")
        print(f"    UNSURE   : {len(unsure)}")
        print(f"    FAIL     : {len(fail)}")
        print(f"    Accuracy (на принятых с GT): {acc_s}")
        if approach == "scoring":
            retried = [r for r in rows if r.retried]
            print(f"    Ретраев  : {len(retried)}")
            confs = [r.confidence for r in rows if r.confidence is not None]
            if confs:
                print(f"    Confidence avg/min: {sum(confs)/len(confs):.2f} / {min(confs):.2f}")
        else:
            agree = [r.agreement_pct for r in rows if r.agreement_pct is not None]
            if agree:
                print(f"    Agreement avg: {sum(agree)/len(agree):.1f}%")
    print()


def print_edge_details(all_results: list[InferenceResult]) -> None:
    edge = [r for r in all_results if r.ticket_label.startswith("edge_")]
    if not edge:
        return
    labels = sorted({r.ticket_label for r in edge})
    print("── Edge cases ────────────────────────────────────────────────────────────────")
    for lbl in labels:
        rows = [r for r in edge if r.ticket_label == lbl]
        print(f"\n  {lbl}")
        for r in sorted(rows, key=lambda x: (x.model_id, x.approach)):
            cat  = r.category or "—"
            conf = f"conf={r.confidence:.2f}" if r.confidence is not None else ""
            agr  = f"agree={r.agreement_pct:.0f}%" if r.agreement_pct is not None else ""
            print(f"    {r.model_id:<22} {r.approach:<11} [{r.status:<6}] {cat:<10} {conf} {agr}")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    config = load_config()
    ollama_ok = check_ollama(config.ollama_host)
    active_models = [m for m in MODELS if m != "deepseek-r1:14b" or ollama_ok]

    print("=" * 60)
    print("День 7: оценка уверенности инференса")
    print("=" * 60)
    print(f"Ollama: {'доступен ✓' if ollama_ok else 'недоступен — deepseek пропущен'}")
    print(f"Модели : {', '.join(active_models)}")

    eval_cases  = load_eval_cases()
    all_cases   = eval_cases + EDGE_CASES
    total_calls = len(active_models) * len(all_cases) * (2 + REDUNDANCY_N)  # max with retries
    print(f"Случаев: {len(eval_cases)} eval + {len(EDGE_CASES)} edge = {len(all_cases)}")
    print(f"Вызовов API (макс.): ~{total_calls}")
    print()

    client = LLMClient(config)
    all_results: list[InferenceResult] = []

    for model_id in active_models:
        print(f"▶ {model_id} ...", end="", flush=True)
        t0 = time.monotonic()
        results = await run_model(client, model_id, all_cases)
        elapsed = time.monotonic() - t0
        n_accept = sum(1 for r in results if r.status == "ACCEPT")
        print(f" {elapsed:.1f}с   принято {n_accept}/{len(results)} (оба подхода)")
        all_results.extend(results)

    print_report(all_results)
    print_edge_details(all_results)

    out_path = Path(__file__).resolve().parent.parent / "results" / "day7_confidence.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
