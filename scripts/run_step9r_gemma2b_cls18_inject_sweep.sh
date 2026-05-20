#!/usr/bin/env bash
set -euo pipefail

# Run Step9R retrieved-doc CSV sweep for Gemma-2-2B with fixed cls_layer=18.
#
# Goal:
#   Fix classification layer at cls=18, then sweep all valid injection layers
#   below cls=18 to find the best inject layer for retrieved-doc CSV.
#
# This trains on:
#   data/final_retrieved/{train,eval}.json
# and saves to:
#   results/csv_retrieved/
#
# Default sweep:
#   cls_layers    = 18
#   inject_layers = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
#
# Notes:
#   - Mode B requires inject layer < cls layer, so for cls=18 the valid
#     transformer-layer injection range is 0..17.
#   - If you want to keep the previous convention of skipping inject=0, run:
#       INJECT_LAYERS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
#       bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
#   - Uses --resume, so completed configs like inject_1_cls_18 will be skipped.
#   - Do NOT run this concurrently with another Gemma2B Step9R process writing
#     to the same results/csv_retrieved/gemma2b_retrieved_csv_sweep.json unless
#     you explicitly know it is safe.
#
# Usage:
#   bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
#
# Optional overrides:
#   GEMMA_GPU=0 bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
#   GEMMA_BATCH_SIZE=16 bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
#   INJECT_LAYERS=0,1,2,3,4,5,6,10,12,16,17 bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9r_gemma2b_cls18_inject_sweep.sh

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
GEMMA_MODEL_PATH=${GEMMA_MODEL_PATH:-google/gemma-2-2b}
export GEMMA_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ---------------------------------------------------------------------------
# GPU / batch / fixed-cls search space
# ---------------------------------------------------------------------------
GEMMA_GPU=${GEMMA_GPU:-1}
GEMMA_BATCH_SIZE=${GEMMA_BATCH_SIZE:-8}

# Fixed classification layer.
CLS_LAYER_FIXED=${CLS_LAYER_FIXED:-18}
CLS_LAYERS=${CLS_LAYERS:-$CLS_LAYER_FIXED}

# All valid injection layers for cls=18 are 0..17.
# If you want a smaller focused sweep, override INJECT_LAYERS from the command line.
INJECT_LAYERS=${INJECT_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
EXPECTED_CONFIGS=${EXPECTED_CONFIGS:-18}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Keep the current best Gemma2B retrieved-doc tuning hyperparams.
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

MODEL_NAME="gemma2b_retrieved"
THIS_RUN_NAME="gemma2b_retrieved_cls18_inject_sweep"
PID_FILE="$LOG_DIR/${THIS_RUN_NAME}.pid"
MAIN_PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${THIS_RUN_NAME}_inj0-17_cls18_lr0p002_ema0p99.log"

# By default, protect against concurrent writes to the same sweep JSON.
ALLOW_PARALLEL_STEP9R=${ALLOW_PARALLEL_STEP9R:-0}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "Model: gemma2b retrieved-doc CSV, fixed cls=18 inject sweep"
echo "Gemma2-2B local path: $GEMMA_MODEL_PATH"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "GPU=$GEMMA_GPU"
echo "BATCH_SIZE=$GEMMA_BATCH_SIZE"
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
    echo "[skip] fixed-cls18 sweep already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

if [[ "$ALLOW_PARALLEL_STEP9R" != "1" && -f "$MAIN_PID_FILE" ]]; then
  MAIN_PID=$(cat "$MAIN_PID_FILE" || true)
  if [[ -n "${MAIN_PID}" ]] && kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "[stop] another Gemma2B Step9R process appears to be running: PID=$MAIN_PID"
    echo "       It may be writing to the same results/csv_retrieved sweep JSON."
    echo "       Wait for it to finish, or set ALLOW_PARALLEL_STEP9R=1 if you are sure."
    echo "       Main log likely: logs/step9_retrieved/gemma2b_retrieved_sweep_inj1-6-10-12-16_cls8-12-16-18-20-24-final_lr0p002_ema0p99.log"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
echo "[start] gemma2b fixed cls=18 inject sweep on GPU $GEMMA_GPU"

CUDA_VISIBLE_DEVICES="$GEMMA_GPU" nohup stdbuf -oL -eL "$PYTHON_BIN" "$STEP9R_SCRIPT" \
  --model gemma2b \
  --str_layers "$INJECT_LAYERS" \
  --cls_layers "$CLS_LAYERS" \
  --lam "$LAM" \
  --lr "$LR" \
  --cos_temp "$COS_TEMP" \
  --ema_decay "$EMA_DECAY" \
  --batch_size "$GEMMA_BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --resume \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[ok] gemma2b fixed cls=18 sweep PID=$PID"
echo
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
echo
echo "Check cls=18 ranking after / during run:"
cat <<'PY'
python - <<'PYCODE'
import json
from pathlib import Path

p = Path("results/csv_retrieved/gemma2b_retrieved_csv_sweep.json")
if not p.exists():
    raise SystemExit(f"Missing sweep file: {p}")

sweep = json.load(open(p))
rows = []
for k, v in sweep.items():
    if "_cls_18" not in k:
        continue
    rows.append((k, v))

rows = sorted(rows, key=lambda kv: kv[1]["best_auroc"], reverse=True)
print("cls=18 completed configs:", len(rows))
print("expected configs:", 18)
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
