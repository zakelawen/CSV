"""
Step 1: Extract questions and gold evidence passages from DPR-format training data.

Inputs:
  data/dpr_download/nq-train.json
  data/dpr_download/trivia-train.json

Outputs:
  data/queries/nq_queries.json
  data/queries/triviaqa_queries.json

Each output record contains:
  - question
  - answers
  - relevant_doc: {text, title}, taken from DPR positive_ctxs[0]
"""

import json
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DPR_NQ_TRAIN, DPR_TRIVIA_TRAIN, QUERIES_DIR, QUERIES_PER_DATASET


DPR_FILES = {
    "nq": DPR_NQ_TRAIN,
    "triviaqa": DPR_TRIVIA_TRAIN,
}


def extract_queries(dpr_path: str, n: int, seed: int = 42) -> list:
    """Sample up to n DPR training records that contain at least one positive passage."""
    with open(dpr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = [item for item in data if item.get("positive_ctxs")]

    random.seed(seed)
    if len(valid) > n:
        valid = random.sample(valid, n)

    results = []
    for item in valid:
        pos = item["positive_ctxs"][0]
        results.append(
            {
                "question": item["question"],
                "answers": item["answers"],
                "relevant_doc": {
                    "text": pos["text"],
                    "title": pos.get("title", ""),
                },
            }
        )
    return results


def main() -> None:
    os.makedirs(QUERIES_DIR, exist_ok=True)

    for dataset_name, dpr_path in DPR_FILES.items():
        if not os.path.exists(dpr_path):
            print(f"[skip] Missing {dpr_path}. Download the DPR data first.")
            continue

        print(
            f"[process] {dataset_name}: sampling {QUERIES_PER_DATASET} queries "
            f"from {dpr_path} ..."
        )
        queries = extract_queries(dpr_path, QUERIES_PER_DATASET)

        out_path = os.path.join(QUERIES_DIR, f"{dataset_name}_queries.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)

        print(f"[done] Saved {len(queries)} records to {out_path}")


if __name__ == "__main__":
    main()
