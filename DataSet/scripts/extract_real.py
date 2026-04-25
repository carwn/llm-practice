"""Extract real labeled examples from HuggingFace dataset."""
import json
import random
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a support ticket classifier. "
    "Classify the ticket into exactly one category: bug, feature, billing, question, or other. "
    "Reply with a single word — the category name."
)

TAG_TO_LABEL = {
    "Bug": "bug",
    "Crash": "bug",
    "Outage": "bug",
    "Feature": "feature",
    "Billing": "billing",
    "Technical": "question",
    "Performance": "question",
    "Network": "question",
    "Security": "question",
    "Login": "question",
    "Hardware": "question",
    "Disruption": "question",
    "Feedback": "other",
    "Product": "other",
    "Documentation": "other",
    "Customer": "other",
    "Marketing": "other",
    "Sales": "other",
    "IT": "other",
    "Customer Support": "other",
}

WANT_PER_LABEL = {"bug": 4, "feature": 4, "billing": 4, "question": 3, "other": 3}


def build_user_message(row):
    subject = row["subject"].strip()
    body = row["body"].strip().replace("\\n", "\n")
    # Trim body to ~300 chars to keep examples concise
    if len(body) > 300:
        body = body[:297] + "..."
    return f"Subject: {subject}\n\n{body}"


def main():
    print("Loading dataset...")
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    en = [x for x in ds if x.get("language") == "en"]
    print(f"English rows: {len(en)}")

    by_label: dict[str, list] = {k: [] for k in WANT_PER_LABEL}
    for row in en:
        label = TAG_TO_LABEL.get(row.get("tag_1", ""))
        if label and len(by_label[label]) < WANT_PER_LABEL[label] * 5:
            by_label[label].append(row)

    random.seed(42)
    examples = []
    for label, rows in by_label.items():
        selected = random.sample(rows, min(WANT_PER_LABEL[label], len(rows)))
        for row in selected:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(row)},
                    {"role": "assistant", "content": label},
                ]
            })

    random.shuffle(examples)
    out_path = "DataSet/data/real.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} real examples to {out_path}")
    from collections import Counter
    labels = [ex["messages"][2]["content"] for ex in examples]
    print("Distribution:", dict(Counter(labels)))


if __name__ == "__main__":
    main()
