"""
Step 2: Retrieve the top-1 DPR passage for each query.

Inputs:
  data/queries/{nq,triviaqa}_queries.json
  data/dpr_download/wiki_dpr_nq/
  data/dpr_download/wiki_dpr_nq.faiss

Outputs:
  data/retrieved/{nq,triviaqa}_retrieved.json
"""

import json
import os
import sys

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATASETS,
    DPR_DIR,
    DPR_QUESTION_ENCODER,
    QUERIES_DIR,
    RETRIEVAL_TOP_K,
    RETRIEVED_DIR,
)


WIKI_DPR_PATH = os.path.join(DPR_DIR, "wiki_dpr_nq")
FAISS_INDEX_PATH = os.path.join(DPR_DIR, "wiki_dpr_nq.faiss")


def load_wiki_dpr():
    """Load the local DPR passage dataset and its FAISS index."""
    print(f"[load] Dataset: {WIKI_DPR_PATH} ...")
    ds = load_from_disk(WIKI_DPR_PATH)

    print(f"[load] FAISS index: {FAISS_INDEX_PATH} ...")
    ds.load_faiss_index("embeddings", FAISS_INDEX_PATH)

    print(f"[done] Loaded {len(ds)} passages with a FAISS index.")
    return ds


class DPRRetriever:
    """Thin wrapper around the DPR question encoder and a FAISS passage index."""

    def __init__(self, ds):
        self.ds = ds

        print(f"[load-model] {DPR_QUESTION_ENCODER} ...")
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_QUESTION_ENCODER)
        self.model = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_ENCODER)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("[done] DPR question encoder loaded.")

    def retrieve(self, query: str, top_k: int = 1) -> list:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            query_embedding = self.model(**inputs).pooler_output.cpu().numpy()[0]

        scores, retrieved = self.ds.get_nearest_examples(
            "embeddings",
            query_embedding,
            k=top_k,
        )

        results = []
        for i in range(top_k):
            results.append(
                {
                    "text": retrieved["text"][i],
                    "title": retrieved["title"][i],
                    "score": float(scores[i]),
                    "passage_id": int(retrieved["id"][i]),
                }
            )
        return results


def main() -> None:
    os.makedirs(RETRIEVED_DIR, exist_ok=True)

    ds = load_wiki_dpr()
    retriever = DPRRetriever(ds)

    for dataset_name in DATASETS:
        query_path = os.path.join(QUERIES_DIR, f"{dataset_name}_queries.json")
        if not os.path.exists(query_path):
            print(f"[skip] Missing {query_path}. Run Step 1 first.")
            continue

        with open(query_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

        print(
            f"\n[retrieve] {dataset_name}: {len(queries)} queries, "
            f"top-{RETRIEVAL_TOP_K} ..."
        )
        results = []
        for item in tqdm(queries, desc=f"{dataset_name} retrieval"):
            retrieved = retriever.retrieve(item["question"], top_k=RETRIEVAL_TOP_K)
            item["retrieved_top1"] = retrieved[0]
            results.append(item)

        out_path = os.path.join(RETRIEVED_DIR, f"{dataset_name}_retrieved.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[done] Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
