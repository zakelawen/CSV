#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B Step9R hparam grid with fixed cls=22 and inject=0..21.
#
# Important:
#   Each LR/EMA pair writes to a separate results_dir, because Step9R filenames
#   do not include lr / ema_decay. This avoids --resume skipping old configs
#   and avoids overwriting previous results.
#
#   Runs sequentially by default: one LR/EMA job at a time on one GPU. This avoids
#   launching multiple Qwen3-4B trainings onto the same 24G GPU.

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export QWEN3_4B_MODEL_PATH

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

QWEN3_4B_GPU=${QWEN3_4B_GPU:-1}
QWEN3_4B_BATCH_SIZE=${QWEN3_4B_BATCH_SIZE:-4}

INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}
CLS_LAYERS=${CLS_LAYERS:-22}

LAM=${LAM:-5.0}
COS_TEMP=${COS_TEMP:-0.1}
NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

LOG_DIR=${LOG_DIR:-logs/step9_retrieved/qwen3_4b_cls22_hparam_grid}
BASE_RESULTS_DIR=${BASE_RESULTS_DIR:-results/csv_retrieved_qwen3_4b_cls22_hparam_grid}
mkdir -p "$LOG_DIR" "$BASE_RESULTS_DIR"

# Grid. You can edit this list.
GRID=(
  "0.001 0.99"
  "0.002 0.99"
  "0.001 0.999"
  "0.0005 0.999"
)

echo "=== Qwen3-4B cls=22 hparam grid ==="
echo "GPU=$QWEN3_4B_GPU"
echo "BATCH_SIZE=$QWEN3_4B_BATCH_SIZE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "LAM=$LAM COS_TEMP=$COS_TEMP"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo "BASE_RESULTS_DIR=$BASE_RESULTS_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "Launch mode: sequential, one job at a time"
echo

for item in "${GRID[@]}"; do
  read -r LR EMA_DECAY <<< "$item"

  LR_TAG=$(echo "$LR" | sed 's/\./p/g')
  EMA_TAG=$(echo "$EMA_DECAY" | sed 's/\./p/g')

  RUN_TAG="lr${LR_TAG}_ema${EMA_TAG}"
  RESULTS_DIR="${BASE_RESULTS_DIR}/${RUN_TAG}"
  LOG_FILE="${LOG_DIR}/qwen3_4b_cls22_inj0-21_${RUN_TAG}.log"
  PID_FILE="${LOG_DIR}/qwen3_4b_cls22_inj0-21_${RUN_TAG}.pid"

  mkdir -p "$RESULTS_DIR"

  if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" || true)
    if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "[skip] already running: $RUN_TAG PID=$OLD_PID"
      continue
    fi
  fi

  echo "[start] $RUN_TAG"
  echo "  RESULTS_DIR=$RESULTS_DIR"
  echo "  LOG_FILE=$LOG_FILE"

  echo "$$" > "$PID_FILE"

  CUDA_VISIBLE_DEVICES="$QWEN3_4B_GPU" stdbuf -oL -eL "$PYTHON_BIN" "$STEP9R_SCRIPT" \
    --model qwen3_4b \
    --str_layers "$INJECT_LAYERS" \
    --cls_layers "$CLS_LAYERS" \
    --lam "$LAM" \
    --lr "$LR" \
    --cos_temp "$COS_TEMP" \
    --ema_decay "$EMA_DECAY" \
    --batch_size "$QWEN3_4B_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --patience "$PATIENCE" \
    --results_dir "$RESULTS_DIR" \
    --resume \
    2>&1 | tee "$LOG_FILE"

  rm -f "$PID_FILE"
  echo "[done] $RUN_TAG"
  echo
done

echo "All jobs finished."
echo
echo "Watch logs:"
echo "  tail -f ${LOG_DIR}/qwen3_4b_cls22_inj0-21_lr0p001_ema0p99.log"
