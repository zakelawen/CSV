#!/usr/bin/env bash
set -euo pipefail

# Run Step9R retrieved-doc CSV search for Gemma-2-9B only.
# This trains on data/final_retrieved/{train,eval}.json and saves to results/csv_retrieved/.
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9R is called with --resume.
#
# Usage:
#   bash scripts/run_step9r_gemma9b.sh
#
# Optional overrides:
#   GEMMA9B_GPU=0 bash scripts/run_step9r_gemma9b.sh
#   GEMMA9B_BATCH_SIZE=4 bash scripts/run_step9r_gemma9b.sh
#   INJECT_LAYERS=0,1,2 CLS_LAYERS=2,4,8,-1 bash scripts/run_step9r_gemma9b.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

# Local Gemma-2-9B snapshot path. Override if your server path differs.
GEMMA9B_MODEL_PATH=${GEMMA9B_MODEL_PATH:-google/gemma-2-9b}
export GEMMA9B_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

GEMMA9B_GPU=${GEMMA9B_GPU:-0}
GEMMA9B_BATCH_SIZE=${GEMMA9B_BATCH_SIZE:-12}

# Match your selected Gemma9B retrieved sweep.
INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,5}
CLS_LAYERS=${CLS_LAYERS:-2,4,8,12,16,20,24,30,36,-1}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

LAM=${LAM:-5.0}
LR=${LR:-0.005}
COS_TEMP=${COS_TEMP:-0.1}
EMA_DECAY=${EMA_DECAY:-0.99}

LOG_DIR=${LOG_DIR:-logs/step9_retrieved}
mkdir -p "$LOG_DIR"
mkdir -p results/csv_retrieved

MODEL_NAME="gemma9b_retrieved"
PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${MODEL_NAME}_inj0-1-2-3-5_cls2-4-8-12-16-20-24-30-36-final.log"

echo "Model: gemma9b retrieved-doc CSV"
echo "Gemma2-9B local path: $GEMMA9B_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU=$GEMMA9B_GPU"
echo "BATCH_SIZE=$GEMMA9B_BATCH_SIZE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo "LAM=$LAM LR=$LR COS_TEMP=$COS_TEMP EMA_DECAY=$EMA_DECAY"
echo "LOG_FILE=$LOG_FILE"
echo

if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] gemma9b retrieved already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

echo "[start] gemma9b retrieved on GPU $GEMMA9B_GPU"

CUDA_VISIBLE_DEVICES="$GEMMA9B_GPU" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9R_SCRIPT" \
  --model gemma9b \
  --str_layers "$INJECT_LAYERS" \
  --cls_layers "$CLS_LAYERS" \
  --lam "$LAM" \
  --lr "$LR" \
  --cos_temp "$COS_TEMP" \
  --ema_decay "$EMA_DECAY" \
  --batch_size "$GEMMA9B_BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --resume \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[ok] gemma9b retrieved PID=$PID"
echo
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
