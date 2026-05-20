#!/usr/bin/env bash
set -euo pipefail

# Run CAD-style Step11 alpha sweeps, then evaluate each generated output.
#
# Examples:
#   bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma9b all 0
#   bash scripts/run_step11_cad_alpha_sweep_with_eval.sh all all 0
#   ALPHAS="0.05 0.1 0.2 0.5" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b nq 0
#   LIMIT=200 ALPHAS="0.1 0.2" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh qwen3_4b triviaqa 0
#   METHOD=csv_gated_cad ALPHAS="0.1 0.2 0.5" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b all 0
#   METHOD=csv_gated_layer_delta_cad CONTRAST_LAYERS="23 21 20" ALPHAS="0.05 0.1 0.2" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b nq 0
#   TAU=0.2 METHOD=csv_gated_layer_delta_cad CONTRAST_LAYERS="23" ALPHAS="0.05 0.1" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b nq 0
#   TAUS="0.16 0.18 0.2" METHOD=csv_gated_layer_delta_cad CONTRAST_LAYERS="21 22 23" ALPHAS="0.0025 0.005 0.01" bash scripts/run_step11_cad_alpha_sweep_with_eval.sh gemma2b nq 0

MODEL_ARG=${1:-gemma9b}     # gemma2b | qwen3_4b | gemma9b | all
DATASET=${2:-all}          # nq | triviaqa | all
GPU=${3:-0}

export GEMMA_MODEL_PATH=${GEMMA_MODEL_PATH:-google/gemma-2-2b}
export GEMMA9B_MODEL_PATH=${GEMMA9B_MODEL_PATH:-google/gemma-2-9b}
export QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}

PYTHON_BIN=${PYTHON_BIN:-python}
STEP11_SCRIPT=${STEP11_SCRIPT:-scripts/step11_contrastive_decoding.py}
STEP12_SCRIPT=${STEP12_SCRIPT:-scripts/step12_evaluate_pipeline.py}
METHOD=${METHOD:-cad_fixed}
TAU=${TAU:-0.5}
TAUS_STR=${TAUS:-$TAU}
read -r -a TAUS_ARR <<< "$TAUS_STR"

ALPHAS_STR=${ALPHAS:-"0.1 0.2 0.5"}
read -r -a ALPHAS_ARR <<< "$ALPHAS_STR"

CONTRAST_LAYERS_STR=${CONTRAST_LAYERS:-${CONTRAST_LAYER:-23}}
read -r -a CONTRAST_LAYERS_ARR <<< "$CONTRAST_LAYERS_STR"

if [[ "$MODEL_ARG" == "all" ]]; then
  MODELS_STR=${MODELS:-"gemma2b qwen3_4b gemma9b"}
else
  MODELS_STR=${MODELS:-"$MODEL_ARG"}
fi
read -r -a MODELS_ARR <<< "$MODELS_STR"

LIMIT=${LIMIT:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-1024}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RUN_EVAL=${RUN_EVAL:-1}
LOG_DIR=${LOG_DIR:-logs/step11_cad_alpha_sweep}
PIPELINE_DIR=${PIPELINE_DIR:-results/pipeline}
mkdir -p "$LOG_DIR" "$PIPELINE_DIR"

if [[ "$DATASET" == "all" ]]; then
  DATASETS=(nq triviaqa)
elif [[ "$DATASET" == "nq" || "$DATASET" == "triviaqa" ]]; then
  DATASETS=("$DATASET")
else
  echo "[error] DATASET must be one of: nq | triviaqa | all; got: $DATASET" >&2
  exit 1
fi

LIMIT_ARGS=()
LIMIT_SUFFIX=""
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  LIMIT_SUFFIX="_limit${LIMIT}"
fi

format_float_for_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
s = f"{x:g}".replace("-", "m").replace(".", "p")
print(s)
PY
}

GENERATED_FILES=()

echo "=============================="
echo "CAD alpha sweep"
echo "Models:   ${MODELS_ARR[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Method:   $METHOD"
echo "Taus:     ${TAUS_ARR[*]}"
echo "Alphas:   ${ALPHAS_ARR[*]}"
if [[ "$METHOD" == "csv_gated_layer_delta_cad" ]]; then
  echo "Layers:   ${CONTRAST_LAYERS_ARR[*]}"
fi
echo "GPU:      $GPU"
echo "=============================="

for MODEL in "${MODELS_ARR[@]}"; do
  for DS in "${DATASETS[@]}"; do
    if [[ "$METHOD" == "csv_gated_layer_delta_cad" ]]; then
      LAYERS_TO_RUN=("${CONTRAST_LAYERS_ARR[@]}")
    else
      LAYERS_TO_RUN=("")
    fi
    for LAYER in "${LAYERS_TO_RUN[@]}"; do
      LAYER_SUFFIX=""
      LAYER_ARGS=()
      if [[ -n "$LAYER" ]]; then
        LAYER_SUFFIX="_layer${LAYER}"
        LAYER_ARGS=(--contrast_layer "$LAYER")
      fi
      for TAU_CUR in "${TAUS_ARR[@]}"; do
        for ALPHA in "${ALPHAS_ARR[@]}"; do
          ALPHA_NAME=$(format_float_for_name "$ALPHA")
          TAU_NAME=$(format_float_for_name "$TAU_CUR")
          OUT_FILE="${PIPELINE_DIR}/${MODEL}_${DS}_${METHOD}_alpha${ALPHA_NAME}_tau${TAU_NAME}_temp0p0${LAYER_SUFFIX}${LIMIT_SUFFIX}.json"
          LOG_FILE="${LOG_DIR}/${MODEL}_${DS}_${METHOD}_alpha${ALPHA_NAME}_tau${TAU_NAME}${LAYER_SUFFIX}${LIMIT_SUFFIX}.log"
          GENERATED_FILES+=("$OUT_FILE")

          if [[ "$SKIP_EXISTING" == "1" && -s "$OUT_FILE" ]]; then
            echo
            echo "=============================="
            echo "[skip existing] $MODEL / $DS / $METHOD / tau=$TAU_CUR / alpha=$ALPHA"
            echo "  $OUT_FILE"
            echo "=============================="
            continue
          fi

          echo
          echo "=============================="
          echo "Running $MODEL / $DS / $METHOD / tau=$TAU_CUR / alpha=$ALPHA on GPU $GPU"
          echo "Output: $OUT_FILE"
          echo "=============================="

          CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$STEP11_SCRIPT" \
            --model "$MODEL" \
            --dataset "$DS" \
            --method "$METHOD" \
            --alpha "$ALPHA" \
            --tau "$TAU_CUR" \
            "${LAYER_ARGS[@]}" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_input_length "$MAX_INPUT_LENGTH" \
            --temperature 0.0 \
            --top_p 1.0 \
            "${LIMIT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
        done
      done
    done
  done
done

if [[ "$RUN_EVAL" == "1" ]]; then
  EXISTING_FILES=()
  for f in "${GENERATED_FILES[@]}"; do
    if [[ -s "$f" ]]; then
      EXISTING_FILES+=("$f")
    else
      echo "[warning] missing expected output: $f" >&2
    fi
  done

  if (( ${#EXISTING_FILES[@]} > 0 )); then
    "$PYTHON_BIN" "$STEP12_SCRIPT" --input "${EXISTING_FILES[@]}"
    "$PYTHON_BIN" scripts/collect_finished_results.py
  else
    echo "[error] no CAD sweep outputs found for evaluation" >&2
    exit 1
  fi
fi

echo
echo "=============================="
echo "Done."
echo "CAD logs: $LOG_DIR"
echo "Generation outputs: $PIPELINE_DIR"
echo "Unified results: results/all_finished_experiments.csv"
echo "=============================="
