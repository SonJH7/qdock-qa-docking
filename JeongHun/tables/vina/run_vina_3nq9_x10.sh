#!/bin/bash
# AutoDock Vina × 10 runs on 3NQ9
# Box from FAM feature points; exhaustiveness=8, num_modes=20

set -e
cd "$(dirname "$0")"

VINA="${VINA_BIN:-vina}"
RECEPTOR=receptor.pdbqt
LIGAND=3nq9_ligand.pdbqt
CX=-4.357
CY=-6.847
CZ=-17.594
SX=17
SY=17
SZ=12

mkdir -p outputs logs

for r in 1 2 3 4 5 6 7 8 9 10; do
  out=outputs/vina_3nq9_r${r}.pdbqt
  log=logs/vina_3nq9_r${r}.log
  if [ -f "$out" ]; then
    echo "[skip] r${r} already done"
    continue
  fi
  echo "[run] r${r} -> $out"
  T0=$(python3 -c "import time; print(time.time())")
  $VINA \
    --receptor "$RECEPTOR" \
    --ligand   "$LIGAND" \
    --center_x $CX --center_y $CY --center_z $CZ \
    --size_x   $SX --size_y   $SY --size_z   $SZ \
    --exhaustiveness 8 \
    --num_modes      20 \
    --seed           $((2000 + r)) \
    --out "$out" \
    > "$log" 2>&1
  T1=$(python3 -c "import time; print(time.time())")
  ELAPSED=$(python3 -c "print(round($T1 - $T0, 3))")
  echo "    elapsed ${ELAPSED}s"
done
echo "DONE"
