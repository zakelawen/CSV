#!/usr/bin/env bash
set -euo pipefail

# Run Step9R retrieved-doc CSV search for Gemma-2-2B only.
#
# This trains on:
#   data/final_retrieved/{train,eval}.json
# and saves to:
#   results/csv_retrieved/
#
# Current best fixed-layer tuning result for Gemma2B retrieved-doc CSV:
#   inject=1, cls=16, lr=0.002, ema_decay=0.99
#
# This expanded sweep uses:
#   inject_layers = 1,2,3,4,5,6,10,12,16
#   cls_layers    = 8,12,16,18,20,24,-1
#
# Safe for SSH/network disconnect: uses nohup.
# Safe for interrupted/restarted runs: Step9R is called with --resume.
#
# Usage:
#   bash scripts/run_step9r_gemma2b.sh
#
# Optional overrides:
#   GEMMA_GPU=1 bash scripts/run_step9r_gemma2b.sh
#   GEMMA_BATCH_SIZE=16 bash scripts/run_step9r_gemma2b.sh
#   INJECT_LAYERS=1,2,3 CLS_LAYERS=8,16,-1 bash scripts/run_step9r_gemma2b.sh
#   NUM_EPOCHS=10 PATIENCE=5 bash scripts/run_step9r_gemma2b.sh
#
# Important if you previously ran Gemma2B retrieved with different hyperparams/layers:
#   RESET_GEMMA_RESULTS=1 bash scripts/run_step9r_gemma2b.sh
# This moves existing gemma2b_retrieved files out of results/csv_retrieved/
# before starting, so --resume will not skip old configs.
#
# If you already ran the previous 48-config sweep and only want to add cls=18:
#   DO NOT use RESET_GEMMA_RESULTS=1
# Just run:
#   bash scripts/run_step9r_gemma2b.sh
# --resume will skip completed configs and run only the new cls=18 configs.

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
# GPU / batch / layer search space
# ---------------------------------------------------------------------------
GEMMA_GPU=${GEMMA_GPU:-1}
GEMMA_BATCH_SIZE=${GEMMA_BATCH_SIZE:-8}

# Retrieved-doc expanded sweep:
# - skip inject=0
# - include early injection layers 1-6
# - add mid-layer injection candidates 10,12,16
# - include cls=18 to refine the middle/late classification range
INJECT_LAYERS=${INJECT_LAYERS:-1,2,3,4,5,6,10,12,16}
CLS_LAYERS=${CLS_LAYERS:-8,12,16,18,20,24,-1}

# With the default layer lists above:
# inject=1,2,3,4,5,6 each has 7 valid cls choices = 42
# inject=10 has cls=12,16,18,20,24,-1 = 6
# inject=12 has cls=16,18,20,24,-1 = 5
# inject=16 has cls=18,20,24,-1 = 4
# total = 57 valid configs
EXPECTED_CONFIGS=${EXPECTED_CONFIGS:-57}

NUM_EPOCHS=${NUM_EPOCHS:-20}
PATIENCE=${PATIENCE:-5}

# Best Gemma2B retrieved-doc tuning result so far.
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
PID_FILE="$LOG_DIR/${MODEL_NAME}.pid"
LOG_FILE="$LOG_DIR/${MODEL_NAME}_sweep_inj1-6-10-12-16_cls8-12-16-18-20-24-final_lr0p002_ema0p99.log"

# If set to 1, move existing Gemma2B retrieved outputs away before starting.
# This avoids mixing old lr/ema/layer configs with the new best-param sweep.
RESET_GEMMA_RESULTS=${RESET_GEMMA_RESULTS:-0}
BACKUP_DIR=${BACKUP_DIR:-results/csv_retrieved_backups}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "Model: gemma2b retrieved-doc CSV"
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
echo "RESET_GEMMA_RESULTS=$RESET_GEMMA_RESULTS"
echo

# ---------------------------------------------------------------------------
# PID guard
# ---------------------------------------------------------------------------
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[skip] gemma2b retrieved already running: PID=$OLD_PID, log=$LOG_FILE"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# Optional cleanup/backup of old Gemma2B retrieved outputs
# ---------------------------------------------------------------------------
if [[ "$RESET_GEMMA_RESULTS" == "1" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  DEST="$BACKUP_DIR/gemma2b_retrieved_before_inj1-6-10-12-16_cls8-12-16-18-20-24-final_lr${LR}_ema${EMA_DECAY}_$TS"
  mkdir -p "$DEST"

  shopt -s nullglob
  GEMMA_FILES=("$RESULTS_DIR"/gemma2b_retrieved_*)
  if (( ${#GEMMA_FILES[@]} > 0 )); then
    echo "[backup] moving existing Gemma2B retrieved outputs to: $DEST"
    mv "${GEMMA_FILES[@]}" "$DEST"/
  else
    echo "[backup] no existing Gemma2B retrieved outputs found."
  fi
  shopt -u nullglob
  echo
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
echo "[start] gemma2b retrieved on GPU $GEMMA_GPU"

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

echo "[ok] gemma2b retrieved PID=$PID"
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

p = Path("results/csv_retrieved/gemma2b_retrieved_csv_sweep.json")
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