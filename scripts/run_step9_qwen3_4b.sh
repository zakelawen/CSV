#!/usr/bin/env bash
set -euo pipefail

# Run Step9 CSV search for Qwen3-4B-Base only.
#
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9 is called with --resume.
#
# Usage:
#   bash scripts/run_step9_qwen3_4b.sh
#
# Optional overrides:
#   QWEN3_4B_GPU=0 bash scripts/run_step9_qwen3_4b.sh
#   QWEN3_4B_BATCH_SIZE=8 bash scripts/run_step9_qwen3_4b.sh
#   INJECT_LAYERS=0,1,2 CLS_LAYERS=2,4,6,8,-1 bash scripts/run_step9_qwen3_4b.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9_qwen3_4b.sh
#
#   # Override the model snapshot path (e.g., to a local HF cache):
#   QWEN3_4B_MODEL_PATH=/path/to/Qwen3-4B-Base bash scripts/run_step9_qwen3_4b.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9_SCRIPT=${STEP9_SCRIPT:-scripts/step9_train_csv_qwen3_4b.py}

# Default to the HuggingFace name. Override with a local snapshot path if you
# have one downloaded — the wrapper passes this through unchanged.
# Example after running snapshot_download():
#   /path/to/Qwen3-4B-Base
QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export QWEN3_4B_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
# Set HF_HUB_OFFLINE=0 the first time if Qwen3-4B-Base isn't in the cache yet.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Choose GPU. If Gemma2 is already running on GPU0, set QWEN3_4B_GPU=1.
QWEN3_4B_GPU=${QWEN3_4B_GPU:-0}

# Recommended small search first.
# Qwen3-4B has 36 transformer layers. Inject early, classify across the network.
LAM=${LAM:-5.0}
LR=${LR:-0.005}
COS_TEMP=${COS_TEMP:-0.1}
EMA_DECAY=${EMA_DECAY:-0.999686}


INJECT_LAYERS=${INJECT_LAYERS:-0,2,4,6}
CLS_LAYERS=${CLS_LAYERS:-8,16,23,24,30,-1}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Adjust if OOM. Qwen3-4B in fp16 is ~8GB. On a 24GB GPU try 4 first, then 8.
QWEN3_4B_BATCH_SIZE=${QWEN3_4B_BATCH_SIZE:-4}

LOG_DIR=${LOG_DIR:-logs/step9_qwen3_4b}
mkdir -p "$LOG_DIR"
mkdir -p results/csv

MODEL_NAME="qwen3_4b"
PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

echo "Model: qwen3_4b (Qwen3-4B-Base)"
echo "Qwen3-4B model path: $QWEN3_4B_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU=$QWEN3_4B_GPU"
echo "BATCH_SIZE=$QWEN3_4B_BATCH_SIZE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo "LAM=$LAM LR=$LR COS_TEMP=$COS_TEMP EMA_DECAY=$EMA_DECAY"
echo "LOG_FILE=$LOG_FILE"
echo

if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] qwen3_4b already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

echo "[start] qwen3_4b on GPU $QWEN3_4B_GPU"

CUDA_VISIBLE_DEVICES="$QWEN3_4B_GPU" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9_SCRIPT" \
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
  --resume \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[ok] qwen3_4b PID=$PID"
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
