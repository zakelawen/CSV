#!/usr/bin/env bash
set -euo pipefail

# Run Step9 CSV search.
#
# Supports:
#   1. Run both models:
#      bash scripts/run_step9_dual_gpu.sh
#
#   2. Run only LLaMA:
#      TARGET_MODEL=llama8b LLAMA_GPU=0 LLAMA_BATCH_SIZE=8 bash scripts/run_step9_dual_gpu.sh
#
#   3. Run only Gemma:
#      TARGET_MODEL=gemma2b GEMMA_GPU=0 GEMMA_BATCH_SIZE=16 bash scripts/run_step9_dual_gpu.sh
#
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9 is called with --resume.

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9_SCRIPT=${STEP9_SCRIPT:-scripts/step9_train_csv.py}

# Which model to run:
#   both    -> run gemma2b and llama8b
#   gemma2b -> only run Gemma-2B
#   llama8b -> only run LLaMA-3.1-8B
TARGET_MODEL=${TARGET_MODEL:-both}

if [[ "$TARGET_MODEL" != "both" && "$TARGET_MODEL" != "gemma2b" && "$TARGET_MODEL" != "llama8b" ]]; then
  echo "[error] TARGET_MODEL must be one of: both, gemma2b, llama8b"
  echo "Example:"
  echo "  TARGET_MODEL=llama8b LLAMA_GPU=0 bash scripts/run_step9_dual_gpu.sh"
  exit 1
fi

# Local HuggingFace snapshot paths. Override these if you move the cache.
GEMMA_MODEL_PATH=${GEMMA_MODEL_PATH:-google/gemma-2-2b}
LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export GEMMA_MODEL_PATH
export LLAMA_MODEL_PATH

# GPU defaults:
# - both: gemma -> GPU0, llama -> GPU1
# - single model: default to GPU0
if [[ "$TARGET_MODEL" == "both" ]]; then
  GEMMA_GPU=${GEMMA_GPU:-0}
  LLAMA_GPU=${LLAMA_GPU:-1}
else
  GEMMA_GPU=${GEMMA_GPU:-0}
  LLAMA_GPU=${LLAMA_GPU:-0}
fi

# Recommended search space: early injection layers + even classification layers + final.
INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,4,5}
CLS_LAYERS=${CLS_LAYERS:-2,4,6,8,10,12,14,16,18,20,22,24,-1}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Adjust if OOM.
# For local 4090:
#   gemma2b: 8-16
#   llama8b: 1-2
# For A800/H800 80GB:
#   llama8b: try 8 first, then 16 if memory is enough.
GEMMA_BATCH_SIZE=${GEMMA_BATCH_SIZE:-8}
LLAMA_BATCH_SIZE=${LLAMA_BATCH_SIZE:-2}

LOG_DIR=${LOG_DIR:-logs/step9_dual}
mkdir -p "$LOG_DIR"
mkdir -p results/csv

echo "TARGET_MODEL=$TARGET_MODEL"
echo "Gemma local path: $GEMMA_MODEL_PATH"
echo "LLaMA local path:  $LLAMA_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo

start_job() {
  local model="$1"
  local gpu="$2"
  local batch_size="$3"
  local pid_file="$LOG_DIR/${model}.pid"
  local log_file="$LOG_DIR/${model}.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid=$(cat "$pid_file" || true)
    if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[skip] $model already running: PID=$old_pid, log=$log_file"
      return 0
    fi
  fi

  echo "[start] model=$model gpu=$gpu batch_size=$batch_size"
  echo "        log=$log_file"

  CUDA_VISIBLE_DEVICES="$gpu" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9_SCRIPT" \
    --model "$model" \
    --str_layers "$INJECT_LAYERS" \
    --cls_layers "$CLS_LAYERS" \
    --batch_size "$batch_size" \
    --num_epochs "$NUM_EPOCHS" \
    --patience "$PATIENCE" \
    --resume \
    > "$log_file" 2>&1 &

  local pid=$!
  echo "$pid" > "$pid_file"
  echo "[ok] $model PID=$pid"
}

case "$TARGET_MODEL" in
  both)
    start_job "gemma2b" "$GEMMA_GPU" "$GEMMA_BATCH_SIZE"
    start_job "llama8b" "$LLAMA_GPU" "$LLAMA_BATCH_SIZE"
    ;;
  gemma2b)
    start_job "gemma2b" "$GEMMA_GPU" "$GEMMA_BATCH_SIZE"
    ;;
  llama8b)
    start_job "llama8b" "$LLAMA_GPU" "$LLAMA_BATCH_SIZE"
    ;;
esac

echo
echo "Job submitted."
echo "Check status:"
if [[ "$TARGET_MODEL" == "both" || "$TARGET_MODEL" == "gemma2b" ]]; then
  echo "  ps -fp \$(cat $LOG_DIR/gemma2b.pid)"
fi
if [[ "$TARGET_MODEL" == "both" || "$TARGET_MODEL" == "llama8b" ]]; then
  echo "  ps -fp \$(cat $LOG_DIR/llama8b.pid)"
fi

echo
echo "Watch logs:"
if [[ "$TARGET_MODEL" == "both" || "$TARGET_MODEL" == "gemma2b" ]]; then
  echo "  tail -f $LOG_DIR/gemma2b.log"
fi
if [[ "$TARGET_MODEL" == "both" || "$TARGET_MODEL" == "llama8b" ]]; then
  echo "  tail -f $LOG_DIR/llama8b.log"
fi

echo
echo "If the server process is killed or training crashes, rerun this same script."
echo "Step9 --resume will skip completed configs and continue from the next unfinished config."