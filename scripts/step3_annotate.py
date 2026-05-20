"""
Step 3: 调用 GPT_MODEL 配置的标注模型标注检索到的 top-1 文档

输入: data/retrieved/{nq,triviaqa}_retrieved.json
输出: data/annotated/{nq,triviaqa}_annotated.json

每条记录新增:
  - annotation: {label, confidence, rationale, supporting_spans, inference_type}

支持断点续传: 已标注的 query 会跳过。
使用多线程并发调用 API，默认 10 个线程。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RETRIEVED_DIR, ANNOTATED_DIR, DATASETS,
    GPT_API_URL, GPT_API_KEY, GPT_MODEL,
)

# 并发数（根据 API 限速调整，一般 10-20 没问题）
MAX_WORKERS = 10
# 每批保存一次的间隔
SAVE_EVERY = 1

# ------------------------------------------------------------------
# 标注 Prompt（来自 Yeh & Li, 2026）
# ------------------------------------------------------------------
CLASSIFICATION_PROMPT = """You are an objective evidence classifier. Given a user question, a list of possible answers, and a single document, decide whether the document is relevant, distracting, or neutral with respect to answering the query.

• Do NOT produce a chain-of-thought. Provide only the required structured output (JSON, see schema below) and a concise 1-2 sentence rationale (no internal reasoning steps).
• Use external/world knowledge only to determine whether a document implicitly supports an answer via ordinary inference. Do not invent or hallucinate facts that are not in the document when justifying the label.
• Follow the definitions and heuristics below exactly.

## Required OUTPUT (JSON)
Return a single JSON object with these fields only:
{
  "label": "relevant" | "distracting" | "neutral",
  "confidence": 0.00-1.00,
  "rationale": "<one- or two-sentence justification>",
  "supporting_spans": ["<short excerpt(s) from the document that justify the judgment>"],
  "inference_type": "direct" | "indirect" | "multi-hop" | "contradiction"
}

## Label definitions & heuristics

### RELEVANT
• The correct answer (or parts of answer) directly appeared in the document → set inference_type = "direct".
• Or the document contains facts that clearly support the correct answer, either by single-step inference (set inference_type = "indirect") or by providing a necessary intermediate hop for a multi-hop inference (set inference_type = "multi-hop").
• If the doc contains intermediate facts that are required to get to the final answer (even though the final answer is not present), treat it as relevant (set inference_type = "multi-hop").
• Provide supporting_spans identifying the explicit sentence(s) or fact(s).

### DISTRACTING
• The document asserts claims that would lead a reader away from the correct answer (i.e., it contradicts the correct answer or makes claims that support an incorrect candidate). Use inference_type = "contradiction" if it explicitly contradicts.
• Or the document contains plausible but misleading facts that do not support the correct answer and could plausibly be mistaken for support.
• Or the document discusses other things that are related to some entities in the query, but does not provide hints for a reader to answer the question.

### NEUTRAL
• The document is unrelated to the query.

## Confidence scoring guidance
• >= 0.90: explicit textual statement of the answer or a clear contradiction/distraction.
• 0.75 - 0.89: strong indirect support or a strong but not explicit contradiction/distraction.
• 0.55 - 0.74: moderate evidence (document gives facts that imply the answer but not overwhelmingly).
• 0.30 - 0.54: weak or partial evidence, or small inconsistency; label should be conservative.
• <= 0.29: little or no evidence; use for neutral decisions.
Set a numeric value according to this guidance.

Question: {question}
Possible answers: {answers}
Document: {document}"""


def build_prompt(question: str, answers: list, document: str) -> str:
    """安全拼接 prompt，避免文档中的 {} 导致 format 报错"""
    return CLASSIFICATION_PROMPT.replace("{question}", question) \
                                .replace("{answers}", json.dumps(answers, ensure_ascii=False)) \
                                .replace("{document}", document)


# ------------------------------------------------------------------
# API 调用（单条）
# ------------------------------------------------------------------
def call_gpt_annotator(question: str, answers: list, document: str) -> dict | None:
    """调用 GPT_MODEL 配置的标注模型标注单个文档，返回解析后的 JSON 或 None"""
    prompt = build_prompt(question, answers, document)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT_API_KEY}",
    }
    payload = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000,
    }

    for attempt in range(3):
        try:
            resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            return json.loads(content)

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            time.sleep(2 ** attempt)

    return None


# ------------------------------------------------------------------
# 处理单条（供线程池调用）
# ------------------------------------------------------------------
def process_item(item: dict) -> dict:
    """标注一条数据，返回带 annotation 的 item"""
    result = call_gpt_annotator(
        question=item["question"],
        answers=item["answers"],
        document=item["retrieved_top1"]["text"],
    )

    if result is not None:
        item["annotation"] = result
    else:
        item["annotation"] = {"label": "error", "confidence": 0, "rationale": "API 调用失败"}

    return item


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    if not GPT_API_KEY:
        raise RuntimeError(
            "GPT_API_KEY is not set. Export it before running Step 3, e.g. "
            "`export GPT_API_KEY=<your_api_key>`."
        )

    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    for dataset_name in DATASETS:
        input_path = os.path.join(RETRIEVED_DIR, f"{dataset_name}_retrieved.json")
        output_path = os.path.join(ANNOTATED_DIR, f"{dataset_name}_annotated.json")

        if not os.path.exists(input_path):
            print(f"[跳过] 未找到 {input_path}，请先运行 step2")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 断点续传
        annotated = []
        done_questions = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                annotated = json.load(f)
            done_questions = {item["question"] for item in annotated}
            print(f"[续传] {dataset_name}: 已有 {len(annotated)} 条标注")

        todo = [item for item in data if item["question"] not in done_questions]
        print(f"[标注] {dataset_name}: 待标注 {len(todo)} 条，并发数 {MAX_WORKERS}")

        save_lock = Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item): item for item in todo}

            for future in tqdm(as_completed(futures), total=len(todo), desc=f"{dataset_name} 标注"):
                result = future.result()

                with save_lock:
                    annotated.append(result)
                    completed += 1

                    if completed % SAVE_EVERY == 0:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(annotated, f, indent=2, ensure_ascii=False)

        # 最终保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2, ensure_ascii=False)

        labels = {}
        for item in annotated:
            lbl = item.get("annotation", {}).get("label", "unknown")
            labels[lbl] = labels.get(lbl, 0) + 1
        print(f"[统计] {dataset_name}: {labels}")


if __name__ == "__main__":
    main()