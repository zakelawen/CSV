#!/usr/bin/env bash
set -euo pipefail

# Run Step9 CSV search for Gemma-2-9B only.
#
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9 is called with --resume.
#
# Usage:
#   bash scripts/run_step9_gemma9b.sh
#
# Optional overrides:
#   GEMMA9B_GPU=1 bash scripts/run_step9_gemma9b.sh
#   GEMMA9B_BATCH_SIZE=4 bash scripts/run_step9_gemma9b.sh
#   INJECT_LAYERS=0,1,2 CLS_LAYERS=2,4,6,8,-1 bash scripts/run_step9_gemma9b.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9_gemma9b.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9_SCRIPT=${STEP9_SCRIPT:-scripts/step9_train_csv_gemma9b.py}

# Local Gemma-2-9B snapshot path.
GEMMA9B_MODEL_PATH=${GEMMA9B_MODEL_PATH:-google/gemma-2-9b}
export GEMMA9B_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Choose GPU. If Gemma2-2B is already running on GPU0, set GEMMA9B_GPU=1.
GEMMA9B_GPU=${GEMMA9B_GPU:-1}

# Recommended small search first.
# Gemma2-9B is larger, so do not start with a huge sweep unless you have time.
INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,4,5}
CLS_LAYERS=${CLS_LAYERS:-2,4,6,8,10,12,14,16,18,20,22,24,-1}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Adjust if OOM. For 80GB GPUs, try 4 or 8 first.
GEMMA9B_BATCH_SIZE=${GEMMA9B_BATCH_SIZE:-1}

LOG_DIR=${LOG_DIR:-logs/step9_gemma9b}
mkdir -p "$LOG_DIR"
mkdir -p results/csv

MODEL_NAME="gemma9b"
PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

echo "Model: gemma9b"
echo "Gemma2-9B local path: $GEMMA9B_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU=$GEMMA9B_GPU"
echo "BATCH_SIZE=$GEMMA9B_BATCH_SIZE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo "LOG_FILE=$LOG_FILE"
echo

if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] gemma9b already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

echo "[start] gemma9b on GPU $GEMMA9B_GPU"

CUDA_VISIBLE_DEVICES="$GEMMA9B_GPU" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9_SCRIPT" \
  --model gemma9b \
  --str_layers "$INJECT_LAYERS" \
  --cls_layers "$CLS_LAYERS" \
  --batch_size "$GEMMA9B_BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --resume \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[ok] gemma9b PID=$PID"
echo
echo "Check status:"
echo "  ps -fp \$(cat $PID_FILE)"
echo
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
echo
echo "If the server process is killed or training crashes, rerun this same script."
echo "Step9 --resume will skip completed configs and continue from the next unfinished config."