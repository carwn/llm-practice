"""Generate synthetic support ticket examples via claude-sonnet-4-6."""
import asyncio
import json
import sys
import random

sys.path.insert(0, ".")
from config import load_config
from client import LLMClient

SYSTEM_PROMPT = (
    "You are a support ticket classifier. "
    "Classify the ticket into exactly one category: bug, feature, billing, question, or other. "
    "Reply with a single word — the category name."
)

GENERATOR_SYSTEM = "You generate realistic customer support ticket examples for a software company."

CATEGORIES = {
    "bug": (
        "Generate a realistic customer support ticket about a software bug or crash. "
        "The customer describes an unexpected error, broken functionality, or system outage. "
        "Return JSON with fields: subject (short title) and body (2-4 sentences describing the issue). "
        "Make it sound like a real frustrated customer. Vary the product/context each time."
    ),
    "feature": (
        "Generate a realistic customer support ticket requesting a new feature or product improvement. "
        "The customer suggests something they'd like to see added. "
        "Return JSON with fields: subject (short title) and body (2-4 sentences). "
        "Vary the product/context each time."
    ),
    "billing": (
        "Generate a realistic customer support ticket about a billing issue: overcharge, wrong invoice, "
        "refund request, subscription confusion, payment failure, or pricing question. "
        "Return JSON with fields: subject (short title) and body (2-4 sentences). "
        "Vary the situation each time."
    ),
    "question": (
        "Generate a realistic customer support ticket where the customer asks a technical or usage question. "
        "Not a bug — the product works, they just need help understanding how to use it. "
        "Return JSON with fields: subject (short title) and body (2-4 sentences). "
        "Vary the topic each time."
    ),
    "other": (
        "Generate a realistic customer support ticket that doesn't fit bug/feature/billing/question. "
        "Could be: account deletion request, general feedback, compliment, partnership inquiry, "
        "legal question, or something unusual. "
        "Return JSON with fields: subject (short title) and body (2-4 sentences). "
        "Vary the type each time."
    ),
}

WANT_PER_CATEGORY = {"bug": 10, "feature": 10, "billing": 10, "question": 8, "other": 7}


async def generate_one(client: LLMClient, category: str, idx: int) -> dict | None:
    prompt = CATEGORIES[category]
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": prompt + f"\n\nAttempt #{idx+1} — make it different from previous ones."},
    ]
    try:
        response = await client.chat("claude-sonnet-4-6", messages)
        # Extract JSON from response
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            print(f"  [warn] no JSON in response for {category}#{idx}")
            return None
        data = json.loads(text[start:end])
        subject = data.get("subject", "").strip()
        body = data.get("body", "").strip()
        if not subject or not body:
            return None
        user_content = f"Subject: {subject}\n\n{body}"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": category},
            ]
        }
    except Exception as e:
        print(f"  [error] {category}#{idx}: {e}")
        return None


async def main():
    config = load_config()
    client = LLMClient(config)

    all_examples = []
    for category, count in WANT_PER_CATEGORY.items():
        print(f"Generating {count} examples for '{category}'...")
        tasks = [generate_one(client, category, i) for i in range(count)]
        results = await asyncio.gather(*tasks)
        good = [r for r in results if r is not None]
        print(f"  Got {len(good)}/{count}")
        all_examples.extend(good)

    random.seed(99)
    random.shuffle(all_examples)

    out_path = "DataSet/data/synthetic.jsonl"
    with open(out_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_examples)} synthetic examples to {out_path}")
    from collections import Counter
    labels = [ex["messages"][2]["content"] for ex in all_examples]
    print("Distribution:", dict(Counter(labels)))


if __name__ == "__main__":
    asyncio.run(main())
