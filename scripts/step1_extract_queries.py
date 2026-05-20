"""
Step 1: 从 DPR 预处理数据中抽取 query 和 relevant document

输入: data/dpr_download/nq-train.json, trivia-train.json
输出: data/queries/nq_queries.json, triviaqa_queries.json

每条记录包含:
  - question: 问题文本
  - answers: 答案列表
  - relevant_doc: {text, title}  来自 DPR positive_ctxs[0]
"""
import json
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DPR_NQ_TRAIN, DPR_TRIVIA_TRAIN, QUERIES_DIR, QUERIES_PER_DATASET

DPR_FILES = {
    "nq":       DPR_NQ_TRAIN,
    "triviaqa": DPR_TRIVIA_TRAIN,
}


def extract_queries(dpr_path: str, n: int, seed: int = 42) -> list:
    """从 DPR 训练数据中抽取 n 条有 positive_ctxs 的 query"""
    with open(dpr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = [item for item in data if item.get("positive_ctxs")]

    random.seed(seed)
    if len(valid) > n:
        valid = random.sample(valid, n)

    results = []
    for item in valid:
        pos = item["positive_ctxs"][0]
        results.append({
            "question": item["question"],
            "answers":  item["answers"],
            "relevant_doc": {
                "text":  pos["text"],
                "title": pos.get("title", ""),
            },
        })
    return results


def main():
    os.makedirs(QUERIES_DIR, exist_ok=True)

    for dataset_name, dpr_path in DPR_FILES.items():
        if not os.path.exists(dpr_path):
            print(f"[跳过] 未找到 {dpr_path}，请先下载 DPR 数据")
            continue

        print(f"[处理] {dataset_name}: 从 {dpr_path} 抽取 {QUERIES_PER_DATASET} 条 query ...")
        queries = extract_queries(dpr_path, QUERIES_PER_DATASET)

        out_path = os.path.join(QUERIES_DIR, f"{dataset_name}_queries.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)

        print(f"[完成] 保存 {len(queries)} 条到 {out_path}")


if __name__ == "__main__":
    main()
