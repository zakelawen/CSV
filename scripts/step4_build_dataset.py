"""
Step 4: Build the gold-vs-distractor classification dataset.

Inputs:
  data/annotated/{nq,triviaqa}_annotated.json

Outputs:
  data/final/train.json
  data/final/eval.json
  data/final/stats.json

This dataset keeps records where the retrieved top-1 passage was annotated as
distracting and pairs it with the original DPR positive passage.
"""

import json
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import ANNOTATED_DIR, DATASETS, FINAL_DIR, TRAIN_RATIO


def load_and_filter(dataset_name: str) -> list:
    """Load annotated records and keep samples with a distracting retrieved passage."""
    path = os.path.join(ANNOTATED_DIR, f"{dataset_name}_annotated.json")
    if not os.path.exists(path):
        print(f"[skip] Missing {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for item in data:
        ann = item.get("annotation", {})
        has_gold_doc = item.get("relevant_doc", {}).get("text", "").strip()
        if ann.get("label") == "distracting" and has_gold_doc:
            filtered.append(
                {
                    "question": item["question"],
                    "answers": item["answers"],
                    "relevant_doc": item["relevant_doc"],
                    "distracting_doc": item["retrieved_top1"],
                    "annotation": ann,
                    "source_dataset": dataset_name,
                }
            )

    return filtered


def main() -> None:
    os.makedirs(FINAL_DIR, exist_ok=True)

    all_samples = []
    stats = {"per_dataset": {}}

    for dataset_name in DATASETS:
        path = os.path.join(ANNOTATED_DIR, f"{dataset_name}_annotated.json")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        filtered = load_and_filter(dataset_name)
        all_samples.extend(filtered)

        label_counts = {}
        for item in raw:
            lbl = item.get("annotation", {}).get("label", "unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        stats["per_dataset"][dataset_name] = {
            "total_queries": len(raw),
            "label_distribution": label_counts,
            "after_filter": len(filtered),
        }

        print(f"[{dataset_name}] label distribution: {label_counts}")
        print(f"[{dataset_name}] after filter: {len(filtered)} records")

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_set = all_samples[:split_idx]
    eval_set = all_samples[split_idx:]

    stats["total_filtered"] = len(all_samples)
    stats["train_size"] = len(train_set)
    stats["eval_size"] = len(eval_set)

    for name, data in [("train", train_set), ("eval", eval_set), ("stats", stats)]:
        path = os.path.join(FINAL_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[save] {path}")

    print("\n===== Final statistics =====")
    print(f"Filtered total: {len(all_samples)}")
    print(f"Train size: {len(train_set)}")
    print(f"Eval size: {len(eval_set)}")


if __name__ == "__main__":
    main()
