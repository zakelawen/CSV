#!/usr/bin/env bash
set -euo pipefail

# Run Step9R retrieved-doc CSV sweep for Qwen3-4B with fixed cls_layer=22.
#
# Goal:
#   The expanded Qwen3-4B sweep found best config around:
#       inject=10, cls=22
#   So we fix cls=22 and sweep all valid injection layers below cls=22.
#
# This trains on:
#   data/final_retrieved/{train,eval}.json
# and saves to:
#   results/csv_retrieved/
#
# Default sweep:
#   cls_layers    = 22
#   inject_layers = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
#
# Notes:
#   - Mode B requires inject layer < cls layer.
#   - For cls=22, valid transformer-layer injection range is 0..21.
#   - Uses --resume, so completed configs such as inject_10_cls_22 will be skipped.
#   - Do NOT run this concurrently with another Qwen3-4B Step9R process writing
#     to the same results/csv_retrieved/qwen3_4b_retrieved_csv_sweep.json.
#
# Usage:
#   bash scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh
#
# Optional overrides:
#   QWEN3_4B_GPU=0 bash scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh
#   QWEN3_4B_BATCH_SIZE=8 bash scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh
#   INJECT_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11 bash scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9r_qwen3_4b_cls22_inject_sweep.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export QWEN3_4B_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ---------------------------------------------------------------------------
# GPU / batch / fixed-cls search space
# ---------------------------------------------------------------------------
QWEN3_4B_GPU=${QWEN3_4B_GPU:-1}
QWEN3_4B_BATCH_SIZE=${QWEN3_4B_BATCH_SIZE:-4}

# Fixed classification layer.
CLS_LAYER_FIXED=${CLS_LAYER_FIXED:-22}
CLS_LAYERS=${CLS_LAYERS:-$CLS_LAYER_FIXED}

# All valid injection layers for cls=22 are 0..21.
INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}
EXPECTED_CONFIGS=${EXPECTED_CONFIGS:-22}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Keep the current best Qwen3-4B retrieved-doc tuning hyperparams.
LAM=${LAM:-5.0}
LR=${LR:-0.002}
COS_TEMP=${COS_TEMP:-0.1}
EMA_DECAY=${EMA_DECAY:-0.99}

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------
RESULTS_DIR=${RESULTS_DIR:-results/csv_retrieved}
LOG_DIR=${LOG_DIR:-logs/step9_retrieved}
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

MODEL_NAME="qwen3_4b_retrieved"
THIS_RUN_NAME="qwen3_4b_retrieved_cls22_inject_sweep"
PID_FILE="$LOG_DIR/${THIS_RUN_NAME}.pid"
MAIN_PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${THIS_RUN_NAME}_inj0-21_cls22_lr0p002_ema0p99.log"

# By default, protect against concurrent writes to the same sweep JSON.
ALLOW_PARALLEL_STEP9R=${ALLOW_PARALLEL_STEP9R:-0}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "Model: qwen3_4b retrieved-doc CSV, fixed cls=22 inject sweep"
echo "Qwen3-4B local path: $QWEN3_4B_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU=$QWEN3_4B_GPU"
echo "BATCH_SIZE=$QWEN3_4B_BATCH_SIZE"
echo "INJECT_LAYERS=$INJECT_LAYERS"
echo "CLS_LAYERS=$CLS_LAYERS"
echo "EXPECTED_CONFIGS=$EXPECTED_CONFIGS"
echo "NUM_EPOCHS=$NUM_EPOCHS PATIENCE=$PATIENCE"
echo "LAM=$LAM LR=$LR COS_TEMP=$COS_TEMP EMA_DECAY=$EMA_DECAY"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "ALLOW_PARALLEL_STEP9R=$ALLOW_PARALLEL_STEP9R"
echo

# ---------------------------------------------------------------------------
# PID guards
# ---------------------------------------------------------------------------
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] fixed-cls22 Qwen sweep already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

if [[ "$ALLOW_PARALLEL_STEP9R" != "1" && -f "$MAIN_PID_FILE" ]]; then
  MAIN_PID=$(cat "$MAIN_PID_FILE" || true)
  if [[ -n "${MAIN_PID}" ]] && kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "[stop] another Qwen3-4B Step9R process appears to be running: PID=$MAIN_PID"
    echo "       It may be writing to the same results/csv_retrieved sweep JSON."
    echo "       Wait for it to finish, or set ALLOW_PARALLEL_STEP9R=1 if you are sure."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
echo "[start] qwen3_4b fixed cls=22 inject sweep on GPU $QWEN3_4B_GPU"

CUDA_VISIBLE_DEVICES="$QWEN3_4B_GPU" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9R_SCRIPT" \
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

echo "[ok] qwen3_4b fixed cls=22 sweep PID=$PID"
echo
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
echo
echo "Check cls=22 ranking after / during run:"
cat <<'PY'
python - <<'PYCODE'
import json
from pathlib import Path

p = Path("results/csv_retrieved/qwen3_4b_retrieved_csv_sweep.json")
if not p.exists():
    raise SystemExit(f"Missing sweep file: {p}")

sweep = json.load(open(p))
rows = []
for k, v in sweep.items():
    if "_cls_22" not in k:
        continue
    rows.append((k, v))

rows = sorted(rows, key=lambda kv: kv[1]["best_auroc"], reverse=True)
print("cls=22 completed configs:", len(rows))
print("expected configs:", 22)
print()
for k, v in rows:
    print(
        k,
        "AUROC=", round(v["best_auroc"], 6),
        "Acc=", round(v["best_accuracy"], 6),
        "epoch=", v["best_epoch"],
        "ckpt=", v["checkpoint"],
    )
PYCODE
PY
