"""
Step 3: Use the configured annotation model to label retrieved top-1 documents.

Inputs:
  data/retrieved/{nq,triviaqa}_retrieved.json

Outputs:
  data/annotated/{nq,triviaqa}_annotated.json

Each output record receives:
  - annotation: {label, confidence, rationale, supporting_spans, inference_type}

The script supports resume-by-output-file and uses a small thread pool for API
calls.
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
from config import ANNOTATED_DIR, DATASETS, GPT_API_KEY, GPT_API_URL, GPT_MODEL, RETRIEVED_DIR


MAX_WORKERS = 10
SAVE_EVERY = 1


CLASSIFICATION_PROMPT = """You are an objective evidence classifier. Given a user question, a list of possible answers, and a single document, decide whether the document is relevant, distracting, or neutral with respect to answering the query.

- Do NOT produce a chain-of-thought. Provide only the required structured output (JSON, see schema below) and a concise 1-2 sentence rationale (no internal reasoning steps).
- Use external/world knowledge only to determine whether a document implicitly supports an answer via ordinary inference. Do not invent or hallucinate facts that are not in the document when justifying the label.
- Follow the definitions and heuristics below exactly.

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
- The correct answer, or a part of the answer, directly appears in the document. Set inference_type = "direct".
- Or the document contains facts that clearly support the correct answer, either by single-step inference (set inference_type = "indirect") or by providing a necessary intermediate hop for a multi-hop inference (set inference_type = "multi-hop").
- If the document contains intermediate facts that are required to get to the final answer, even though the final answer is not present, treat it as relevant.
- Provide supporting_spans identifying the explicit sentence(s) or fact(s).

### DISTRACTING
- The document asserts claims that would lead a reader away from the correct answer. Use inference_type = "contradiction" if it explicitly contradicts.
- Or the document contains plausible but misleading facts that do not support the correct answer and could plausibly be mistaken for support.
- Or the document discusses entities related to the query but does not provide useful hints for answering it.

### NEUTRAL
- The document is unrelated to the query.

## Confidence scoring guidance
- >= 0.90: explicit textual statement of the answer or a clear contradiction/distraction.
- 0.75 - 0.89: strong indirect support or a strong but not explicit contradiction/distraction.
- 0.55 - 0.74: moderate evidence.
- 0.30 - 0.54: weak or partial evidence, or small inconsistency; label should be conservative.
- <= 0.29: little or no evidence; use for neutral decisions.
Set a numeric value according to this guidance.

Question: {question}
Possible answers: {answers}
Document: {document}"""


def build_prompt(question: str, answers: list, document: str) -> str:
    """Build the annotation prompt without relying on str.format for document text."""
    return (
        CLASSIFICATION_PROMPT.replace("{question}", question)
        .replace("{answers}", json.dumps(answers, ensure_ascii=False))
        .replace("{document}", document)
    )


def call_gpt_annotator(question: str, answers: list, document: str) -> dict | None:
    """Call the configured annotation model and parse its JSON response."""
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
            content = resp.json()["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            time.sleep(2**attempt)

    return None


def process_item(item: dict) -> dict:
    """Annotate one record and return it with an annotation field."""
    result = call_gpt_annotator(
        question=item["question"],
        answers=item["answers"],
        document=item["retrieved_top1"]["text"],
    )

    if result is not None:
        item["annotation"] = result
    else:
        item["annotation"] = {
            "label": "error",
            "confidence": 0,
            "rationale": "Annotation API call failed.",
        }

    return item


def main() -> None:
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
            print(f"[skip] Missing {input_path}. Run Step 2 first.")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotated = []
        done_questions = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                annotated = json.load(f)
            done_questions = {item["question"] for item in annotated}
            print(f"[resume] {dataset_name}: found {len(annotated)} existing annotations.")

        todo = [item for item in data if item["question"] not in done_questions]
        print(
            f"[annotate] {dataset_name}: {len(todo)} records pending, "
            f"{MAX_WORKERS} workers."
        )

        save_lock = Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_item, item): item for item in todo}

            for future in tqdm(as_completed(futures), total=len(todo), desc=f"{dataset_name} annotation"):
                result = future.result()

                with save_lock:
                    annotated.append(result)
                    completed += 1

                    if completed % SAVE_EVERY == 0:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(annotated, f, indent=2, ensure_ascii=False)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2, ensure_ascii=False)

        labels = {}
        for item in annotated:
            lbl = item.get("annotation", {}).get("label", "unknown")
            labels[lbl] = labels.get(lbl, 0) + 1
        print(f"[stats] {dataset_name}: {labels}")


if __name__ == "__main__":
    main()
