#!/usr/bin/env bash
set -euo pipefail

# Run Step9R hyperparameter studies for the retrieved-doc CSV classifier.
#
# Studies:
#   1. Classification layer c:
#      fixed best injection layer k, sweep cls_layer c.
#   2. Steering strength lambda:
#      fixed best (k, c), sweep --lam.
#
# Default model paths are set for the AUTODL server. Override them if needed:
#   export GEMMA_MODEL_PATH=/path/to/gemma-2-2b
#   export QWEN3_4B_MODEL_PATH=/path/to/Qwen3-4B-Base
#
# Example:
#   STUDY=c MODEL=gemma2b bash scripts/run_step9r_hparam_studies.sh
#   STUDY=lambda MODEL=qwen3_4b bash scripts/run_step9r_hparam_studies.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

MODEL=${MODEL:-all}       # gemma2b | qwen3_4b | all
STUDY=${STUDY:-all}       # c | lambda | all
DRY_RUN=${DRY_RUN:-0}

RESULTS_BASE=${RESULTS_BASE:-results/hparam_studies_retrieved}
LOG_DIR=${LOG_DIR:-logs/step9_retrieved_hparams}
mkdir -p "$RESULTS_BASE" "$LOG_DIR"

GEMMA_MODEL_PATH=${GEMMA_MODEL_PATH:-google/gemma-2-2b}
QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export GEMMA_MODEL_PATH QWEN3_4B_MODEL_PATH

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

DEFAULT_GPU=${CUDA_VISIBLE_DEVICES:-0}
GEMMA_GPU=${GEMMA_GPU:-$DEFAULT_GPU}
QWEN3_4B_GPU=${QWEN3_4B_GPU:-$DEFAULT_GPU}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}
COS_TEMP=${COS_TEMP:-0.1}

# Final/best settings used in the main retrieved-doc CSV_R experiments.
GEMMA_K=${GEMMA_K:-14}
GEMMA_C=${GEMMA_C:-18}
GEMMA_C_SWEEP=${GEMMA_C_SWEEP:-16,18,20,22,24,-1}
GEMMA_LR=${GEMMA_LR:-0.002}
GEMMA_EMA=${GEMMA_EMA:-0.99}
GEMMA_BATCH_SIZE=${GEMMA_BATCH_SIZE:-8}

QWEN_K=${QWEN_K:-21}
QWEN_C=${QWEN_C:-22}
QWEN_C_SWEEP=${QWEN_C_SWEEP:-22,23,24,27,30,-1}
QWEN_LR=${QWEN_LR:-0.001}
QWEN_EMA=${QWEN_EMA:-0.999}
QWEN_BATCH_SIZE=${QWEN_BATCH_SIZE:-4}

LAMBDA_GRID=${LAMBDA_GRID:-0.5 1 2 5 10}

die() {
  echo "[error] $*" >&2
  exit 1
}

float_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

run_step9r() {
  local model="$1"
  local gpu="$2"
  local results_dir="$3"
  local str_layer="$4"
  local cls_layers="$5"
  local lam="$6"
  local lr="$7"
  local ema="$8"
  local batch_size="$9"
  local tag="${10}"

  mkdir -p "$results_dir"
  local log_file="$LOG_DIR/${tag}.log"

  echo
  echo "===================================================================="
  echo "[run] $tag"
  echo "===================================================================="
  echo "model=$model gpu=$gpu results_dir=$results_dir"
  echo "str_layer=$str_layer cls_layers=$cls_layers lambda=$lam"
  echo "lr=$lr ema=$ema batch_size=$batch_size epochs=$NUM_EPOCHS patience=$PATIENCE"
  echo "log_file=$log_file"

  local cmd=(
    "$PYTHON_BIN" "$STEP9R_SCRIPT"
    --model "$model"
    --str_layer "$str_layer"
    --cls_layers "$cls_layers"
    --lam "$lam"
    --lr "$lr"
    --cos_temp "$COS_TEMP"
    --ema_decay "$ema"
    --batch_size "$batch_size"
    --num_epochs "$NUM_EPOCHS"
    --patience "$PATIENCE"
    --results_dir "$results_dir"
    --resume
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    echo
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" 2>&1 | tee "$log_file"
}

run_c_study_for_model() {
  local model="$1"
  case "$model" in
    gemma2b)
      run_step9r \
        "gemma2b" "$GEMMA_GPU" \
        "$RESULTS_BASE/gemma2b_c_sweep_fixed_k${GEMMA_K}" \
        "$GEMMA_K" "$GEMMA_C_SWEEP" "5.0" \
        "$GEMMA_LR" "$GEMMA_EMA" "$GEMMA_BATCH_SIZE" \
        "gemma2b_c_sweep_fixed_k${GEMMA_K}"
      ;;
    qwen3_4b)
      run_step9r \
        "qwen3_4b" "$QWEN3_4B_GPU" \
        "$RESULTS_BASE/qwen3_4b_c_sweep_fixed_k${QWEN_K}" \
        "$QWEN_K" "$QWEN_C_SWEEP" "5.0" \
        "$QWEN_LR" "$QWEN_EMA" "$QWEN_BATCH_SIZE" \
        "qwen3_4b_c_sweep_fixed_k${QWEN_K}"
      ;;
    *)
      die "Unsupported model for c study: $model"
      ;;
  esac
}

run_lambda_study_for_model() {
  local model="$1"
  local lam

  case "$model" in
    gemma2b)
      for lam in $LAMBDA_GRID; do
        local tag_lam
        tag_lam=$(float_tag "$lam")
        run_step9r \
          "gemma2b" "$GEMMA_GPU" \
          "$RESULTS_BASE/gemma2b_lambda_sweep_fixed_k${GEMMA_K}_c${GEMMA_C}/lam${tag_lam}" \
          "$GEMMA_K" "$GEMMA_C" "$lam" \
          "$GEMMA_LR" "$GEMMA_EMA" "$GEMMA_BATCH_SIZE" \
          "gemma2b_lambda_lam${tag_lam}_k${GEMMA_K}_c${GEMMA_C}"
      done
      ;;
    qwen3_4b)
      for lam in $LAMBDA_GRID; do
        local tag_lam
        tag_lam=$(float_tag "$lam")
        run_step9r \
          "qwen3_4b" "$QWEN3_4B_GPU" \
          "$RESULTS_BASE/qwen3_4b_lambda_sweep_fixed_k${QWEN_K}_c${QWEN_C}/lam${tag_lam}" \
          "$QWEN_K" "$QWEN_C" "$lam" \
          "$QWEN_LR" "$QWEN_EMA" "$QWEN_BATCH_SIZE" \
          "qwen3_4b_lambda_lam${tag_lam}_k${QWEN_K}_c${QWEN_C}"
      done
      ;;
    *)
      die "Unsupported model for lambda study: $model"
      ;;
  esac
}

case "$MODEL" in
  gemma2b|qwen3_4b)
    MODELS=("$MODEL")
    ;;
  all)
    MODELS=("gemma2b" "qwen3_4b")
    ;;
  *)
    die "MODEL must be gemma2b, qwen3_4b, or all; got $MODEL"
    ;;
esac

case "$STUDY" in
  c|lambda|all)
    ;;
  *)
    die "STUDY must be c, lambda, or all; got $STUDY"
    ;;
esac

echo "=== Step9R hyperparameter studies ==="
echo "MODEL=$MODEL STUDY=$STUDY DRY_RUN=$DRY_RUN"
echo "RESULTS_BASE=$RESULTS_BASE"
echo "LOG_DIR=$LOG_DIR"
echo "GEMMA_MODEL_PATH=$GEMMA_MODEL_PATH"
echo "QWEN3_4B_MODEL_PATH=$QWEN3_4B_MODEL_PATH"
echo "LAMBDA_GRID=$LAMBDA_GRID"

for model in "${MODELS[@]}"; do
  if [[ "$STUDY" == "c" || "$STUDY" == "all" ]]; then
    run_c_study_for_model "$model"
  fi
  if [[ "$STUDY" == "lambda" || "$STUDY" == "all" ]]; then
    run_lambda_study_for_model "$model"
  fi
done

echo
echo "[done] Hyperparameter jobs finished."
echo "Collect tables with:"
echo "  python scripts/collect_step9r_hparam_studies.py --base_dir $RESULTS_BASE"
