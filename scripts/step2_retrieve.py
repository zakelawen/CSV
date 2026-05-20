"""
Step 2: 对每个 query 检索 top-1 文档

输入: data/queries/{nq,triviaqa}_queries.json
      data/dpr_download/wiki_dpr_nq/       (本地保存的 dataset)
      data/dpr_download/wiki_dpr_nq.faiss  (本地保存的 FAISS 索引)
输出: data/retrieved/{nq,triviaqa}_retrieved.json
"""
import json
import os
import sys

import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    QUERIES_DIR, RETRIEVED_DIR, DATASETS, DPR_DIR,
    DPR_QUESTION_ENCODER, RETRIEVAL_TOP_K,
)

WIKI_DPR_PATH = os.path.join(DPR_DIR, "wiki_dpr_nq")
FAISS_INDEX_PATH = os.path.join(DPR_DIR, "wiki_dpr_nq.faiss")


# ------------------------------------------------------------------
# 1. 加载本地数据集 + FAISS 索引
# ------------------------------------------------------------------
def load_wiki_dpr():
    """从本地加载 dataset 和 FAISS 索引"""
    print(f"[加载] dataset: {WIKI_DPR_PATH} ...")
    ds = load_from_disk(WIKI_DPR_PATH)

    print(f"[加载] FAISS 索引: {FAISS_INDEX_PATH} ...")
    ds.load_faiss_index("embeddings", FAISS_INDEX_PATH)

    print(f"[完成] {len(ds)} 个段落，FAISS 索引已加载")
    return ds


# ------------------------------------------------------------------
# 2. 检索器
# ------------------------------------------------------------------
class DPRRetriever:
    def __init__(self, ds):
        self.ds = ds

        print(f"[加载模型] {DPR_QUESTION_ENCODER} ...")
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(DPR_QUESTION_ENCODER)
        self.model = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_ENCODER)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("[完成] DPR question encoder 已加载")

    def retrieve(self, query: str, top_k: int = 1) -> list:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            query_embedding = self.model(**inputs).pooler_output.cpu().numpy()[0]

        scores, retrieved = self.ds.get_nearest_examples(
            "embeddings", query_embedding, k=top_k,
        )

        results = []
        for i in range(top_k):
            results.append({
                "text":       retrieved["text"][i],
                "title":      retrieved["title"][i],
                "score":      float(scores[i]),
                "passage_id": int(retrieved["id"][i]),
            })
        return results


# ------------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------------
def main():
    os.makedirs(RETRIEVED_DIR, exist_ok=True)

    ds = load_wiki_dpr()
    retriever = DPRRetriever(ds)

    for dataset_name in DATASETS:
        query_path = os.path.join(QUERIES_DIR, f"{dataset_name}_queries.json")
        if not os.path.exists(query_path):
            print(f"[跳过] 未找到 {query_path}，请先运行 step1")
            continue

        with open(query_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

        print(f"\n[检索] {dataset_name}: {len(queries)} 条 query, top-{RETRIEVAL_TOP_K} ...")
        results = []
        for item in tqdm(queries, desc=f"{dataset_name} 检索"):
            retrieved = retriever.retrieve(item["question"], top_k=RETRIEVAL_TOP_K)
            item["retrieved_top1"] = retrieved[0]
            results.append(item)

        out_path = os.path.join(RETRIEVED_DIR, f"{dataset_name}_retrieved.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[完成] 保存 {len(results)} 条到 {out_path}")


if __name__ == "__main__":
    main()