"""
Step 4: 筛选并构建最终数据集

输入: data/annotated/{nq,triviaqa}_annotated.json
输出:
  - data/final/train.json     训练集（80%）
  - data/final/eval.json      评估集（20%）
  - data/final/stats.json     数据集统计信息

筛选条件:
  - annotation.label == "distracting"
  - relevant_doc 非空
"""
import json
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import ANNOTATED_DIR, FINAL_DIR, DATASETS, TRAIN_RATIO


def load_and_filter(dataset_name: str) -> list:
    """加载标注数据并按条件筛选"""
    path = os.path.join(ANNOTATED_DIR, f"{dataset_name}_annotated.json")
    if not os.path.exists(path):
        print(f"[跳过] 未找到 {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for item in data:
        ann = item.get("annotation", {})
        if (
            ann.get("label") == "distracting"
            and item.get("relevant_doc", {}).get("text", "").strip()
        ):
            filtered.append({
                "question":        item["question"],
                "answers":         item["answers"],
                "relevant_doc":    item["relevant_doc"],
                "distracting_doc": item["retrieved_top1"],
                "annotation":      ann,
                "source_dataset":  dataset_name,
            })

    return filtered


def main():
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
            "total_queries":      len(raw),
            "label_distribution": label_counts,
            "after_filter":       len(filtered),
        }

        print(f"[{dataset_name}] 标签分布: {label_counts}")
        print(f"[{dataset_name}] 筛选后: {len(filtered)} 条")

    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_set = all_samples[:split_idx]
    eval_set  = all_samples[split_idx:]

    stats["total_filtered"] = len(all_samples)
    stats["train_size"]     = len(train_set)
    stats["eval_size"]      = len(eval_set)

    for name, data in [("train", train_set), ("eval", eval_set), ("stats", stats)]:
        path = os.path.join(FINAL_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[保存] {path}")

    print(f"\n===== 最终统计 =====")
    print(f"筛选后总数: {len(all_samples)}")
    print(f"训练集: {len(train_set)}")
    print(f"评估集: {len(eval_set)}")


if __name__ == "__main__":
    main()