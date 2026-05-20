"""
Step 5: 为主实验准备测试集的检索结果

输入: data/dpr_download/nq-test.csv, trivia-test.csv + FAISS 索引
输出: data/final/test_retrieval_{nq,triviaqa}.json

前提: step2 已运行过（FAISS 索引和 wiki_dpr 数据已就绪）
"""
import json
import os
import sys
import csv

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DPR_NQ_TEST, DPR_TRIVIA_TEST, FINAL_DIR, RETRIEVAL_TOP_K,
)

TEST_QA_FILES = {
    "nq":       DPR_NQ_TEST,
    "triviaqa": DPR_TRIVIA_TEST,
}


def load_test_queries(path: str) -> list:
    """加载 DPR 格式的测试集 CSV: question \\t answers"""
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


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 复用 step2 的组件
    from step2_retrieve import load_wiki_dpr, DPRRetriever

    ds = load_wiki_dpr()
    retriever = DPRRetriever(ds)

    for dataset_name, qa_path in TEST_QA_FILES.items():
        if not os.path.exists(qa_path):
            print(f"[跳过] 未找到 {qa_path}，请下载 DPR 测试集")
            continue

        queries = load_test_queries(qa_path)
        print(f"[检索] {dataset_name} 测试集: {len(queries)} 条 query ...")

        results = []
        for item in tqdm(queries, desc=f"{dataset_name} 测试集检索"):
            retrieved = retriever.retrieve(item["question"], top_k=RETRIEVAL_TOP_K)
            results.append({
                "question":      item["question"],
                "answers":       item["answers"],
                "retrieved_doc": retrieved[0],
            })

        out_path = os.path.join(FINAL_DIR, f"test_retrieval_{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[完成] {len(results)} 条保存到 {out_path}")


if __name__ == "__main__":
    main()