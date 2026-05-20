#!/usr/bin/env bash
set -euo pipefail

# Greedy Step11 external baselines + Step12 evaluation for RAG-Probe.
#
# Usage:
#   bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b nq 0
#   bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b triviaqa 0
#   bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b all 0
#
# Smoke test:
#   LIMIT=2 MAX_NEW_TOKENS=5 bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b nq 0
#
# Useful overrides:
#   METHODS="no_doc vanilla_rag cad_fixed" bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b nq 0
#   SKIP_EXISTING=1 bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b all 0
#   RUN_EVAL=0 bash scripts/run_step11_greedy_baselines_with_eval.sh gemma2b nq 0

MODEL=${1:-gemma2b}
DATASET=${2:-all}          # nq | triviaqa | all
GPU=${3:-0}


# ---------------------------------------------------------------------------
# Local model snapshot paths used by Step11.
# You can override any of these from the command line if a machine differs.
# ---------------------------------------------------------------------------
export GEMMA_MODEL_PATH=${GEMMA_MODEL_PATH:-google/gemma-2-2b}
export GEMMA9B_MODEL_PATH=${GEMMA9B_MODEL_PATH:-google/gemma-2-9b}
export QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}

# Optional local DoLA custom_generate path. If this file exists, Step11 will
# import it directly and will NOT query hf-mirror/HF.
LOCAL_DOLA_DIR=${LOCAL_DOLA_DIR:-$PWD/third_party/transformers-community-dola}
LOCAL_DOLA_GENERATE=${LOCAL_DOLA_GENERATE:-$LOCAL_DOLA_DIR/custom_generate/generate.py}
if [[ -f "$LOCAL_DOLA_GENERATE" ]]; then
  export DOLA_CUSTOM_GENERATE=${DOLA_CUSTOM_GENERATE:-$LOCAL_DOLA_GENERATE}
elif [[ -f "$LOCAL_DOLA_DIR/custom_generate/generate.py" ]]; then
  export DOLA_CUSTOM_GENERATE=${DOLA_CUSTOM_GENERATE:-$LOCAL_DOLA_DIR/custom_generate/generate.py}
else
  export DOLA_CUSTOM_GENERATE=${DOLA_CUSTOM_GENERATE:-transformers-community/dola}
fi

PYTHON_BIN=${PYTHON_BIN:-python}
STEP11_SCRIPT=${STEP11_SCRIPT:-scripts/step11_contrastive_decoding.py}
STEP12_SCRIPT=${STEP12_SCRIPT:-scripts/step12_evaluate_pipeline.py}

LIMIT=${LIMIT:-}           # e.g. LIMIT=2 for smoke test
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-1024}
CAD_ALPHA=${CAD_ALPHA:-0.5}
ACD_ALPHA=${ACD_ALPHA:-1.0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
RUN_EVAL=${RUN_EVAL:-1}

# Default baseline set. Override with env var, e.g. METHODS="dola".
METHODS_STR=${METHODS:-"no_doc vanilla_rag cad_fixed acd context_ucd dola"}
read -r -a METHODS_ARR <<< "$METHODS_STR"

if [[ "$DATASET" == "all" ]]; then
  DATASETS=(nq triviaqa)
elif [[ "$DATASET" == "nq" || "$DATASET" == "triviaqa" ]]; then
  DATASETS=("$DATASET")
else
  echo "[error] DATASET must be one of: nq | triviaqa | all; got: $DATASET" >&2
  exit 1
fi

LOG_DIR=${LOG_DIR:-logs/step11_baselines}
PIPELINE_DIR=${PIPELINE_DIR:-results/pipeline}
mkdir -p "$LOG_DIR" "$PIPELINE_DIR"

LIMIT_ARGS=()
LIMIT_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  LIMIT_SUFFIX="_limit${LIMIT}"
fi

# Step11's current filename convention with default alpha/tau/temp.
# Example: gemma2b_nq_acd_alpha1p0_tau0p5_temp0p0_limit2.json
format_float_for_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:g}".replace("-", "m").replace(".", "p")
print(s)
PY
}

alpha_for_method() {
  local method="$1"
  case "$method" in
    cad_fixed)
      echo "$CAD_ALPHA"
      ;;
    acd)
      echo "$ACD_ALPHA"
      ;;
    *)
      echo "1.0"
      ;;
  esac
}

result_path_for() {
  local model="$1"
  local dataset="$2"
  local method="$3"
  local alpha
  local alpha_name
  alpha=$(alpha_for_method "$method")
  alpha_name=$(format_float_for_name "$alpha")
  echo "${PIPELINE_DIR}/${model}_${dataset}_${method}_alpha${alpha_name}_tau0p5_temp0p0${LIMIT_SUFFIX}.json"
}

GENERATED_FILES=()

for DS in "${DATASETS[@]}"; do
  for METHOD in "${METHODS_ARR[@]}"; do
    OUT_FILE=$(result_path_for "$MODEL" "$DS" "$METHOD")
    GENERATED_FILES+=("$OUT_FILE")

    if [[ "$SKIP_EXISTING" == "1" && -s "$OUT_FILE" ]]; then
      echo
      echo "=============================="
      echo "[skip existing] $MODEL / $DS / $METHOD"
      echo "  $OUT_FILE"
      echo "=============================="
      continue
    fi

    echo
    echo "=============================="
    echo "Running $MODEL / $DS / $METHOD on GPU $GPU"
    echo "Output: $OUT_FILE"
    echo "=============================="

    COMMON_ARGS=(
      --model "$MODEL"
      --dataset "$DS"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --max_input_length "$MAX_INPUT_LENGTH"
      --temperature 0.0
      --top_p 1.0
      "${LIMIT_ARGS[@]}"
    )

    EXTRA_ARGS=()
    case "$METHOD" in
      no_doc|vanilla_rag)
        ;;
      cad_fixed)
        EXTRA_ARGS+=(--alpha "$CAD_ALPHA")
        ;;
      acd)
        EXTRA_ARGS+=(--alpha "$ACD_ALPHA")
        ;;
      context_ucd)
        EXTRA_ARGS+=(--context_ucd_beta 1.0 --context_ucd_relative_top 0.0)
        ;;
      dola)
        EXTRA_ARGS+=(--dola_layers high --dola_repetition_penalty 1.2)
        ;;
      *)
        echo "[error] Unknown method in METHODS: $METHOD" >&2
        exit 1
        ;;
    esac

    LOG_FILE="$LOG_DIR/${MODEL}_${DS}_${METHOD}${LIMIT_SUFFIX}.log"

    # Use tee so you can watch progress and keep a log file.
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$STEP11_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --method "$METHOD" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
  done
done

# Run Step12 once on exactly the files generated/expected in this run.
# This avoids mixing old limit/full outputs unless you intentionally set the same LIMIT.
if [[ "$RUN_EVAL" == "1" ]]; then
  echo
  echo "=============================="
  echo "Running Step12 evaluation"
  echo "=============================="

  EXISTING_FILES=()
  MISSING_FILES=()
  for f in "${GENERATED_FILES[@]}"; do
    if [[ -s "$f" ]]; then
      EXISTING_FILES+=("$f")
    else
      MISSING_FILES+=("$f")
    fi
  done

  if (( ${#MISSING_FILES[@]} > 0 )); then
    echo "[warning] Some expected Step11 outputs are missing and will not be evaluated:" >&2
    for f in "${MISSING_FILES[@]}"; do
      echo "  - $f" >&2
    done
  fi

  if (( ${#EXISTING_FILES[@]} == 0 )); then
    echo "[error] No Step11 output files found for evaluation." >&2
    exit 1
  fi

  "$PYTHON_BIN" "$STEP12_SCRIPT" --input "${EXISTING_FILES[@]}"

  echo
  echo "=============================="
  echo "Compact EM/F1 table"
  echo "=============================="
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
path = Path("results/pipeline/pipeline_eval_summary.csv")
if not path.exists():
    raise SystemExit(f"Missing {path}")
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
cols = ["model", "dataset", "method", "n", "em", "f1"]
rows.sort(key=lambda r: (r.get("model",""), r.get("dataset",""), r.get("method","")))
widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
print("  ".join(c.ljust(widths[c]) for c in cols))
print("  ".join("-" * widths[c] for c in cols))
for r in rows:
    out = []
    for c in cols:
        v = r.get(c, "")
        if c in {"em", "f1"} and v not in {"", "None", None}:
            try:
                v = f"{float(v):.4f}"
            except Exception:
                pass
        out.append(str(v).ljust(widths[c]))
    print("  ".join(out))
PY
fi

echo
echo "=============================="
echo "Done."
echo "Generation logs: $LOG_DIR"
echo "Generation outputs: $PIPELINE_DIR"
echo "Step12 summary: results/pipeline/pipeline_eval_summary.csv"
echo "=============================="
