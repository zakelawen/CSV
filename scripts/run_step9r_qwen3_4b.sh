#!/usr/bin/env bash
set -euo pipefail

# Run Step9R retrieved-doc CSV search for Qwen3-4B-Base only.
#
# This trains on:
#   data/final_retrieved/{train,eval}.json
# and saves to:
#   results/csv_retrieved/
#
# Current Qwen3-4B retrieved-doc tuning result:
#   fixed layer: inject=4, cls=30
#   best hyperparams: lr=0.002, ema_decay=0.99
#
# This expanded sweep uses:
#   inject_layers = 2,4,6,8,10,11
#   cls_layers    = 8,12,16,20,22,23,24,27,30,-1
#
# Notes:
#   - cls=23 is included because Qwen3-4B retrieved-doc probe baseline
#     is strongest around layer_23.
#   - cls=30 is included because the fixed-layer Qwen CSV tuning was done
#     on inject=4, cls=30.
#
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9R is called with --resume.
#
# Usage:
#   bash scripts/run_step9r_qwen3_4b.sh
#
# Optional overrides:
#   QWEN3_4B_GPU=1 bash scripts/run_step9r_qwen3_4b.sh
#   QWEN3_4B_BATCH_SIZE=8 bash scripts/run_step9r_qwen3_4b.sh
#   INJECT_LAYERS=4,6 CLS_LAYERS=22,23,24,27,30 bash scripts/run_step9r_qwen3_4b.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9r_qwen3_4b.sh
#
# Important:
#   If results/csv_retrieved/ already contains old qwen3_4b_retrieved_* files
#   from different hyperparams, move them away before running, otherwise --resume
#   may skip old configs.

PYTHON_BIN=${PYTHON_BIN:-python}
STEP9R_SCRIPT=${STEP9R_SCRIPT:-scripts/step9_train_csv_retrieved.py}

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
# Local Qwen3-4B snapshot path. Override if your cache path differs.
QWEN3_4B_MODEL_PATH=${QWEN3_4B_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export QWEN3_4B_MODEL_PATH

# Force local loading. If local snapshot is incomplete, fail instead of downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# ---------------------------------------------------------------------------
# GPU / batch / layer search space
# ---------------------------------------------------------------------------
QWEN3_4B_GPU=${QWEN3_4B_GPU:-1}
QWEN3_4B_BATCH_SIZE=${QWEN3_4B_BATCH_SIZE:-4}

# Retrieved-doc expanded sweep for Qwen3-4B.
# Based on fixed-layer tuning:
#   best fixed-layer config: inject=4, cls=30
#   best hyperparams: lr=0.002, ema_decay=0.99
#
# Also include cls=23 because retrieved-doc probe baseline peaks around layer_23.
INJECT_LAYERS=${INJECT_LAYERS:-2,4,6,8,10,11}
CLS_LAYERS=${CLS_LAYERS:-8,12,16,20,22,23,24,27,30,-1}

# With the default layer lists above:
# inject=2:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
# inject=4:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
# inject=6:  cls=8,12,16,20,22,23,24,27,30,-1 = 10
# inject=8:  cls=8 is skipped, remaining 9
# inject=10: cls=8 is skipped, remaining 9
# inject=11: cls=8 is skipped, remaining 9
# total = 57 valid configs
EXPECTED_CONFIGS=${EXPECTED_CONFIGS:-57}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

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
PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${MODEL_NAME}_expanded_inj2-4-6-8-10-11_cls8-12-16-20-22-23-24-27-30-final_lr0p002_ema0p99.log"

# Optional cleanup switch.
# If set to 1, move existing Qwen retrieved outputs away before starting.
RESET_QWEN_RESULTS=${RESET_QWEN_RESULTS:-0}
BACKUP_DIR=${BACKUP_DIR:-results/csv_retrieved_backups}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "Model: qwen3_4b retrieved-doc CSV"
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
echo "RESET_QWEN_RESULTS=$RESET_QWEN_RESULTS"
echo

# ---------------------------------------------------------------------------
# PID guard
# ---------------------------------------------------------------------------
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] qwen3_4b retrieved already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# Optional cleanup/backup of old Qwen retrieved outputs
# ---------------------------------------------------------------------------
if [[ "$RESET_QWEN_RESULTS" == "1" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  DEST="$BACKUP_DIR/qwen3_4b_retrieved_before_expanded_inj2-4-6-8-10-11_cls8-12-16-20-22-23-24-27-30-final_lr${LR}_ema${EMA_DECAY}_$TS"
  mkdir -p "$DEST"

  shopt -s nullglob
  QWEN_FILES=("$RESULTS_DIR"/qwen3_4b_retrieved_*)
  if (( ${#QWEN_FILES[@]} > 0 )); then
    echo "[backup] moving existing Qwen3-4B retrieved outputs to: $DEST"
    mv "${QWEN_FILES[@]}" "$DEST"/
  else
    echo "[backup] no existing Qwen3-4B retrieved outputs found."
  fi
  shopt -u nullglob
  echo
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
echo "[start] qwen3_4b retrieved on GPU $QWEN3_4B_GPU"

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

echo "[ok] qwen3_4b retrieved PID=$PID"
echo
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
echo
echo "Check best after finish:"
cat <<'PY'
python - <<'PYCODE'
import json
from pathlib import Path

p = Path("results/csv_retrieved/qwen3_4b_retrieved_csv_sweep.json")
if not p.exists():
    raise SystemExit(f"Missing sweep file: {p}")

sweep = json.load(open(p))
best_key = max(sweep, key=lambda k: sweep[k]["best_auroc"])
best = sweep[best_key]

print("configs:", len(sweep))
print("expected configs:", 57)
print("best_key:", best_key)
print("best_auroc:", best["best_auroc"])
print("best_accuracy:", best["best_accuracy"])
print("best_epoch:", best["best_epoch"])
print("checkpoint:", best["checkpoint"])

print("\nTop 10:")
for k, v in sorted(sweep.items(), key=lambda kv: kv[1]["best_auroc"], reverse=True)[:10]:
    print(
        k,
        "AUROC=", round(v["best_auroc"], 4),
        "Acc=", round(v["best_accuracy"], 4),
        "epoch=", v["best_epoch"],
    )
PYCODE
PY
