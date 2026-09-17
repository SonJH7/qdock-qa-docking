#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_ROOT="$ROOT/annealing_time_sweep_qpu_fixedemb_v113_opt1_r1to10"
WORK_ROOT="$SWEEP_ROOT/work_cases"
CONFIG_DIR="$SWEEP_ROOT/configs"
LOG_DIR="$SWEEP_ROOT/logs"
SAMPLE_ROOT="$SWEEP_ROOT/samples"
RESULT_DIR="$SWEEP_ROOT/results"
FIXED_DIR="$SWEEP_ROOT/fixed_embeddings"

PY_SAMPLE="${QDOCK_QPU_PYTHON:-python3}"
PY_EVAL="${QDOCK_EVAL_PYTHON:-python3}"
SOLVER="Advantage2_system1.13;graph_id=01e1ea5685"
NUM_READS=1000

# Fixed penalty/embedding source (same as current chain-strength study)
SOURCE_CASE_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/cases/3nq9_Kdist7.500_Kmono19.000"
SOURCE_CASE_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/cases/4jsz_Kdist0.500_Kmono1.000"
CASE_3NQ9="$(basename "$SOURCE_CASE_3NQ9")"
CASE_4JSZ="$(basename "$SOURCE_CASE_4JSZ")"

EMB_SRC_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/samples/3nq9_Kdist7.500_Kmono19.000/online_bo_v113_b100_q3_n20_x0p1_r5_20260401_120636_s049/QUBOs/3nq9_ligand_embedding.json"
EMB_SRC_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/samples/4jsz_Kdist0.500_Kmono1.000/online_bo_v113_b100_quality_4jsz_n20_x0p1_r10_chain_20260405_025657_s011/QUBOs/4jsz_ligand_embedding.json"
EMB_3NQ9="$FIXED_DIR/3nq9_ligand_embedding.json"
EMB_4JSZ="$FIXED_DIR/4jsz_ligand_embedding.json"

# Per-ligand fixed chain_strength from lowcs r1~r10 best
CS_3NQ9="0.5"
CS_4JSZ="0.75"

# Annealing-time sweep grid (us)
ANNEAL_TIMES=(0.5 1 2 5 10 20 50 100 200 500 1000)
REPEATS=(1 2 3 4 5 6 7 8 9 10)

STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$WORK_ROOT" "$CONFIG_DIR" "$LOG_DIR" "$SAMPLE_ROOT" "$RESULT_DIR" "$FIXED_DIR"

# Snapshot exact BO cases into local work root.
rm -rf "$WORK_ROOT/$CASE_3NQ9" "$WORK_ROOT/$CASE_4JSZ"
cp -R "$SOURCE_CASE_3NQ9" "$WORK_ROOT/$CASE_3NQ9"
cp -R "$SOURCE_CASE_4JSZ" "$WORK_ROOT/$CASE_4JSZ"

cp -f "$EMB_SRC_3NQ9" "$EMB_3NQ9"
cp -f "$EMB_SRC_4JSZ" "$EMB_4JSZ"
SHA_3NQ9=$(shasum -a 256 "$EMB_3NQ9" | awk '{print $1}')
SHA_4JSZ=$(shasum -a 256 "$EMB_4JSZ" | awk '{print $1}')
QHASH_3NQ9=$(shasum -a 256 "$WORK_ROOT/$CASE_3NQ9/QUBOs/3nq9_ligand.npy" | awk '{print $1}')
QHASH_4JSZ=$(shasum -a 256 "$WORK_ROOT/$CASE_4JSZ/QUBOs/4jsz_ligand.npy" | awk '{print $1}')

MANIFEST="$RESULT_DIR/anneal_time_fixedemb_opt1_r1to10_manifest_${STAMP}.csv"
echo "annealing_time,chain_strength,repeat,ligand,case_name,run_tag,status,sample_log,eval_log,hash_ok,cached_ok,embedding_sha,qubo_sha" > "$MANIFEST"

echo "Fixed embedding SHA256:"
echo "  3nq9: $SHA_3NQ9"
echo "  4jsz: $SHA_4JSZ"
echo "Case QUBO SHA256:"
echo "  $CASE_3NQ9: $QHASH_3NQ9"
echo "  $CASE_4JSZ: $QHASH_4JSZ"
echo "Output root: $SWEEP_ROOT"

for REPEAT in "${REPEATS[@]}"; do
  for AT in "${ANNEAL_TIMES[@]}"; do
    AT_LABEL=${AT/./p}

    for TARGET in "3nq9" "4jsz"; do
      if [[ "$TARGET" == "3nq9" ]]; then
        CASE_NAME="$CASE_3NQ9"
        LIGAND="3nq9"
        CS="$CS_3NQ9"
        EMB_FIXED="$EMB_3NQ9"
        EMB_SHA="$SHA_3NQ9"
        QSHA="$QHASH_3NQ9"
      else
        CASE_NAME="$CASE_4JSZ"
        LIGAND="4jsz"
        CS="$CS_4JSZ"
        EMB_FIXED="$EMB_4JSZ"
        EMB_SHA="$SHA_4JSZ"
        QSHA="$QHASH_4JSZ"
      fi

      CS_LABEL=${CS/./p}
      RUN_TAG="atsweep_${LIGAND}_at${AT_LABEL}_cs${CS_LABEL}_r${REPEAT}_${STAMP}"
      CFG="$CONFIG_DIR/sampler_${LIGAND}_at${AT_LABEL}_r${REPEAT}.json"

      cat > "$CFG" << JSON
{
  "backend": "dwave_qpu",
  "backend_params": {"solver": "$SOLVER"},
  "sampler_kwargs": {
    "num_reads": $NUM_READS,
    "chain_strength": $CS,
    "annealing_time": $AT,
    "embedding_parameters": {"threads": 8},
    "return_embedding": true
  }
}
JSON

      mkdir -p "$SAMPLE_ROOT/$CASE_NAME/$RUN_TAG/QUBOs"
      cp -f "$EMB_FIXED" "$SAMPLE_ROOT/$CASE_NAME/$RUN_TAG/QUBOs/${LIGAND}_ligand_embedding.json"

      SAMPLE_LOG="$LOG_DIR/${RUN_TAG}_sample.log"
      EVAL_LOG="$LOG_DIR/${RUN_TAG}_eval.log"

      (
        export QDOCK_QPU_WORKDIR="$WORK_ROOT"
        export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
        export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
        export QDOCK_PDB_IDS="$CASE_NAME"
        export QDOCK_RUN_TAG="$RUN_TAG"
        export QDOCK_SAMPLER_CONFIG="$CFG"
        export QDOCK_EMBEDDING_REUSE=0
        "$PY_SAMPLE" "$ROOT/D_qpu_sample_fam.py"
      ) > "$SAMPLE_LOG" 2>&1 || {
        echo "$AT,$CS,$REPEAT,$LIGAND,$CASE_NAME,$RUN_TAG,sample_failed,$SAMPLE_LOG,,false,false,$EMB_SHA,$QSHA" >> "$MANIFEST"
        continue
      }

      if grep -q "ok_cases=0/1" "$SAMPLE_LOG"; then
        echo "$AT,$CS,$REPEAT,$LIGAND,$CASE_NAME,$RUN_TAG,sample_failed_no_ok,$SAMPLE_LOG,,false,false,$EMB_SHA,$QSHA" >> "$MANIFEST"
        continue
      fi

      EMB_RUN="$SAMPLE_ROOT/$CASE_NAME/$RUN_TAG/QUBOs/${LIGAND}_ligand_embedding.json"
      META="$SAMPLE_ROOT/$CASE_NAME/$RUN_TAG/${LIGAND}_ligand_sampler_meta.json"

      if [[ -f "$EMB_RUN" ]]; then
        RUN_SHA=$(shasum -a 256 "$EMB_RUN" | awk '{print $1}')
        HASH_OK=$([[ "$RUN_SHA" == "$EMB_SHA" ]] && echo true || echo false)
      else
        HASH_OK=false
      fi
      if [[ -f "$META" ]] && grep -q '"embedding_cached":[[:space:]]*true' "$META"; then
        CACHED_OK=true
      else
        CACHED_OK=false
      fi

      if [[ "$HASH_OK" != "true" || "$CACHED_OK" != "true" ]]; then
        echo "$AT,$CS,$REPEAT,$LIGAND,$CASE_NAME,$RUN_TAG,embedding_verify_failed,$SAMPLE_LOG,,$HASH_OK,$CACHED_OK,$EMB_SHA,$QSHA" >> "$MANIFEST"
        continue
      fi

      (
        export QDOCK_QPU_WORKDIR="$WORK_ROOT"
        export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
        export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
        export QDOCK_PDB_IDS="$CASE_NAME"
        export QDOCK_RUN_TAG="$RUN_TAG"
        export QDOCK_SAMPLER_CONFIG="$CFG"
        export QDOCK_REPORT=1
        "$PY_EVAL" "$ROOT/D_qpu_eval_fam.py"
      ) > "$EVAL_LOG" 2>&1 || {
        echo "$AT,$CS,$REPEAT,$LIGAND,$CASE_NAME,$RUN_TAG,eval_failed,$SAMPLE_LOG,$EVAL_LOG,$HASH_OK,$CACHED_OK,$EMB_SHA,$QSHA" >> "$MANIFEST"
        continue
      }

      echo "$AT,$CS,$REPEAT,$LIGAND,$CASE_NAME,$RUN_TAG,ok,$SAMPLE_LOG,$EVAL_LOG,$HASH_OK,$CACHED_OK,$EMB_SHA,$QSHA" >> "$MANIFEST"
    done
  done
done

echo "DONE manifest=$MANIFEST"
