"""Run 10 eval examples through gpt-4o-mini and record baseline results."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from config import load_config
from client import LLMClient


async def classify(client: LLMClient, example: dict) -> dict:
    messages = [m for m in example["messages"] if m["role"] != "assistant"]
    expected = next(m["content"] for m in example["messages"] if m["role"] == "assistant")

    start = time.monotonic()
    try:
        response = await client.chat("gpt-4o-mini", messages)
        latency_ms = (time.monotonic() - start) * 1000
        predicted = response.strip().lower().split()[0] if response.strip() else ""
        # Normalize: strip punctuation
        predicted = predicted.strip(".,!?;:")
    except Exception as e:
        latency_ms = 0
        predicted = f"ERROR: {e}"

    return {
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "latency_ms": round(latency_ms),
        "user_content": next(m["content"] for m in messages if m["role"] == "user"),
    }


async def main():
    eval_path = Path("DataSet/data/eval.jsonl")
    out_path = Path("DataSet/results/baseline.json")
    out_path.parent.mkdir(exist_ok=True)

    with open(eval_path) as f:
        all_eval = [json.loads(line) for line in f if line.strip()]

    # Take 10 examples — 2 per category for coverage
    from collections import defaultdict
    by_label: dict[str, list] = defaultdict(list)
    for ex in all_eval:
        label = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        by_label[label].append(ex)

    sample = []
    for label, items in by_label.items():
        sample.extend(items[:2])
    sample = sample[:10]

    print(f"Running baseline on {len(sample)} examples with gpt-4o-mini...")
    config = load_config()
    client = LLMClient(config)

    tasks = [classify(client, ex) for ex in sample]
    results = await asyncio.gather(*tasks)

    correct = sum(r["correct"] for r in results)
    accuracy = correct / len(results) * 100

    output = {
        "model": "gpt-4o-mini",
        "total": len(results),
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results)),
        "criteria": {
            "primary": "Exact match: predicted label == expected label (one of: bug, feature, billing, question, other)",
            "secondary": "Format compliance: single word response, no extra text",
            "target_after_finetune": "Accuracy ≥ 90%, format compliance 100%",
        },
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nBaseline results:")
    print(f"  Accuracy : {accuracy:.1f}% ({correct}/{len(results)})")
    print(f"  Avg latency: {output['avg_latency_ms']} ms")
    print(f"\nPer-example results:")
    for r in results:
        status = "OK" if r["correct"] else "WRONG"
        print(f"  [{status}] expected={r['expected']:10s} predicted={r['predicted']}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
