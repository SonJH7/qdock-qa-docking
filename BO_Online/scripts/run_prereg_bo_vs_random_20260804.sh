#!/usr/bin/env bash
# Prospective BO vs random search (PRE-REGISTERED 2026-08-04, QDock_Paper commit 93fd20c).
# embed_reuse = FALSE (no --embed-reuse flag) to match the exploratory 51-run pool -> poolable.
# Seeds are the registered ones. DO NOT change n, seeds, or rules mid-run.
# 2 parallel streams (3NQ9, 4JSZ); each runs 5 BO + 5 random sequentially.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_BIN="${QDOCK_QPU_PYTHON:-python3}"
CFG="$ROOT/sampler_config_qpu_reads1000_pilot20260804.json"   # pilot: Advantage2_system1 graph 01138bbada
DRV="$ROOT/BO_Online/scripts/run_online_bo.py"
PARENT="$ROOT/BO_Online_prereg_20260804"
MASTER="$PARENT/master.log"
mkdir -p "$PARENT"

run_one() {
  local obj="$1" pol="$2" seed="$3"
  local lig="${obj#quality_}"
  local out="$PARENT/${lig}_${pol}_s${seed}"
  mkdir -p "$out/logs"
  local tag="prereg_${lig}_${pol}_s${seed}"
  local polflag=""; [ "$pol" = "random" ] && polflag="--policy random"
  echo "[$(date '+%F %T')] START ${lig}_${pol}_s${seed}" | tee -a "$MASTER" >>"$out/logs/live.log"
  env -u PYTHONPATH -u PYTHONHOME "$CONDA_BIN" run --no-capture-output -n qdock-qpu \
    "$PYTHON_BIN" "$DRV" --mode qpu --objective "$obj" --budget 100 --n-init 20 --xi 0.1 \
    --num-reads 1000 $polflag --sampler-config "$CFG" \
    --out-dir "$out" --run-tag "$tag" --seed "$seed" --max-retries 3 \
    >>"$out/logs/live.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   ${lig}_${pol}_s${seed} rc=$rc" | tee -a "$MASTER" >>"$out/logs/live.log"
}

stream() {  # $1 = objective (quality_3nq9 | quality_4jsz)
  local obj="$1"
  for s in 20260804001 20260804002 20260804003 20260804004 20260804005; do run_one "$obj" bo "$s"; done
  for s in 20260804101 20260804102 20260804103 20260804104 20260804105; do run_one "$obj" random "$s"; done
}

echo "[$(date '+%F %T')] BATCH START (pre-reg 93fd20c, embed_reuse=False)" | tee -a "$MASTER"
stream quality_3nq9 &
stream quality_4jsz &
wait
echo "[$(date '+%F %T')] BATCH DONE" | tee -a "$MASTER"
