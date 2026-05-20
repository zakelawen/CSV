"""
Step 5: Prepare retrieved passages for the downstream QA test sets.

Inputs:
  data/dpr_download/nq-test.csv
  data/dpr_download/trivia-test.csv
  data/dpr_download/wiki_dpr_nq/
  data/dpr_download/wiki_dpr_nq.faiss

Outputs:
  data/final/test_retrieval_{nq,triviaqa}.json

Prerequisite: Step 2's DPR/FAISS resources must be available locally.
"""

import csv
import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DPR_NQ_TEST, DPR_TRIVIA_TEST, FINAL_DIR, RETRIEVAL_TOP_K


TEST_QA_FILES = {
    "nq": DPR_NQ_TEST,
    "triviaqa": DPR_TRIVIA_TEST,
}


def load_test_queries(path: str) -> list:
    """Load a DPR-style test CSV with columns: question, answers."""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                question = row[0]
                try:
                    answers = json.loads(row[1])
                except json.JSONDecodeError:
                    answers = [row[1]]
                queries.append({"question": question, "answers": answers})
    return queries


def main() -> None:
    os.makedirs(FINAL_DIR, exist_ok=True)

    from step2_retrieve import DPRRetriever, load_wiki_dpr

    ds = load_wiki_dpr()
    retriever = DPRRetriever(ds)

    for dataset_name, qa_path in TEST_QA_FILES.items():
        if not os.path.exists(qa_path):
            print(f"[skip] Missing {qa_path}. Download the DPR test set first.")
            continue

        queries = load_test_queries(qa_path)
        print(f"[retrieve] {dataset_name} test set: {len(queries)} queries ...")

        results = []
        for item in tqdm(queries, desc=f"{dataset_name} test retrieval"):
            retrieved = retriever.retrieve(item["question"], top_k=RETRIEVAL_TOP_K)
            results.append(
                {
                    "question": item["question"],
                    "answers": item["answers"],
                    "retrieved_doc": retrieved[0],
                }
            )

        out_path = os.path.join(FINAL_DIR, f"test_retrieval_{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[done] Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
