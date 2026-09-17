#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_ROOT="$ROOT/chain_strength_sweep_qpu_fixedemb_v113_opt1_r1to10_fullcs_chainbreak_true"
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

# Fixed penalty cases from BO best runs.
SOURCE_CASE1="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/cases/3nq9_Kdist7.500_Kmono19.000"
SOURCE_CASE2="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/cases/4jsz_Kdist0.500_Kmono1.000"
CASE1="$(basename "$SOURCE_CASE1")"
CASE2="$(basename "$SOURCE_CASE2")"
CASES="$CASE1,$CASE2"

# Fixed embeddings from BO best steps.
EMB_SRC_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/samples/3nq9_Kdist7.500_Kmono19.000/online_bo_v113_b100_q3_n20_x0p1_r5_20260401_120636_s049/QUBOs/3nq9_ligand_embedding.json"
EMB_SRC_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/samples/4jsz_Kdist0.500_Kmono1.000/online_bo_v113_b100_quality_4jsz_n20_x0p1_r10_chain_20260405_025657_s011/QUBOs/4jsz_ligand_embedding.json"
EMB_3NQ9="$FIXED_DIR/3nq9_ligand_embedding.json"
EMB_4JSZ="$FIXED_DIR/4jsz_ligand_embedding.json"

# Same full chain-strength grid as existing opt1 fullcs.
STRENGTHS=(0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 7.0 8.0)
REPEATS=(1 2 3 4 5 6 7 8 9 10)
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$WORK_ROOT" "$CONFIG_DIR" "$LOG_DIR" "$SAMPLE_ROOT" "$RESULT_DIR" "$FIXED_DIR"

if [[ -z "${DWAVE_API_TOKEN:-}" ]]; then
  echo "ERROR: DWAVE_API_TOKEN is not set. Export token and re-run."
  exit 2
fi
export DWAVE_API_REGION="${DWAVE_API_REGION:-na-west-1}"

# Snapshot exact BO case directories into sweep-local work root.
rm -rf "$WORK_ROOT/$CASE1" "$WORK_ROOT/$CASE2"
cp -R "$SOURCE_CASE1" "$WORK_ROOT/$CASE1"
cp -R "$SOURCE_CASE2" "$WORK_ROOT/$CASE2"

cp -f "$EMB_SRC_3NQ9" "$EMB_3NQ9"
cp -f "$EMB_SRC_4JSZ" "$EMB_4JSZ"
SHA_3NQ9=$(shasum -a 256 "$EMB_3NQ9" | awk '{print $1}')
SHA_4JSZ=$(shasum -a 256 "$EMB_4JSZ" | awk '{print $1}')
QHASH_3NQ9=$(shasum -a 256 "$WORK_ROOT/$CASE1/QUBOs/3nq9_ligand.npy" | awk '{print $1}')
QHASH_4JSZ=$(shasum -a 256 "$WORK_ROOT/$CASE2/QUBOs/4jsz_ligand.npy" | awk '{print $1}')

MANIFEST="$RESULT_DIR/chain_strength_fixedemb_opt1_r1to10_fullcs_chainbreak_true_manifest_${STAMP}.csv"
echo "chain_strength,repeat,run_tag,status,sample_log,eval_log,hash_ok_3nq9,hash_ok_4jsz,cached_ok_3nq9,cached_ok_4jsz,chain_break_fraction_flag" > "$MANIFEST"

echo "Fixed embedding SHA256:"
echo "  3nq9: $SHA_3NQ9"
echo "  4jsz: $SHA_4JSZ"
echo "Case QUBO SHA256 (work/cases snapshot):"
echo "  $CASE1: $QHASH_3NQ9"
echo "  $CASE2: $QHASH_4JSZ"
echo "Output root: $SWEEP_ROOT"

for REPEAT in "${REPEATS[@]}"; do
  for CS in "${STRENGTHS[@]}"; do
    CS_LABEL=${CS/./p}
    RUN_TAG="csopt1full_cbftrue_cs${CS_LABEL}_r${REPEAT}_${STAMP}"
    CFG="$CONFIG_DIR/sampler_cs${CS_LABEL}_r${REPEAT}.json"

    cat > "$CFG" << JSON
{
  "backend": "dwave_qpu",
  "backend_params": {"solver": "$SOLVER"},
  "sampler_kwargs": {
    "num_reads": $NUM_READS,
    "chain_strength": $CS,
    "chain_break_fraction": true,
    "embedding_parameters": {"threads": 8},
    "return_embedding": true
  }
}
JSON

    mkdir -p "$SAMPLE_ROOT/$CASE1/$RUN_TAG/QUBOs" "$SAMPLE_ROOT/$CASE2/$RUN_TAG/QUBOs"
    cp -f "$EMB_3NQ9" "$SAMPLE_ROOT/$CASE1/$RUN_TAG/QUBOs/3nq9_ligand_embedding.json"
    cp -f "$EMB_4JSZ" "$SAMPLE_ROOT/$CASE2/$RUN_TAG/QUBOs/4jsz_ligand_embedding.json"

    SAMPLE_LOG="$LOG_DIR/${RUN_TAG}_sample.log"
    EVAL_LOG="$LOG_DIR/${RUN_TAG}_eval.log"

    (
      export QDOCK_QPU_WORKDIR="$WORK_ROOT"
      export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
      export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
      export QDOCK_PDB_IDS="$CASES"
      export QDOCK_RUN_TAG="$RUN_TAG"
      export QDOCK_SAMPLER_CONFIG="$CFG"
      export QDOCK_EMBEDDING_REUSE=0
      "$PY_SAMPLE" "$ROOT/D_qpu_sample_fam.py"
    ) > "$SAMPLE_LOG" 2>&1 || {
      echo "$CS,$REPEAT,$RUN_TAG,sample_failed,$SAMPLE_LOG,,,,,,true" >> "$MANIFEST"
      continue
    }

    if grep -q "ok_cases=0/2" "$SAMPLE_LOG"; then
      echo "$CS,$REPEAT,$RUN_TAG,sample_failed_no_ok,$SAMPLE_LOG,,,,,,true" >> "$MANIFEST"
      continue
    fi

    EMB_RUN_3NQ9="$SAMPLE_ROOT/$CASE1/$RUN_TAG/QUBOs/3nq9_ligand_embedding.json"
    EMB_RUN_4JSZ="$SAMPLE_ROOT/$CASE2/$RUN_TAG/QUBOs/4jsz_ligand_embedding.json"
    META_3NQ9="$SAMPLE_ROOT/$CASE1/$RUN_TAG/3nq9_ligand_sampler_meta.json"
    META_4JSZ="$SAMPLE_ROOT/$CASE2/$RUN_TAG/4jsz_ligand_sampler_meta.json"

    if [[ -f "$EMB_RUN_3NQ9" ]]; then
      SHA_RUN_3NQ9=$(shasum -a 256 "$EMB_RUN_3NQ9" | awk '{print $1}')
      HASH_OK_3NQ9=$([[ "$SHA_RUN_3NQ9" == "$SHA_3NQ9" ]] && echo true || echo false)
    else
      HASH_OK_3NQ9=false
    fi
    if [[ -f "$EMB_RUN_4JSZ" ]]; then
      SHA_RUN_4JSZ=$(shasum -a 256 "$EMB_RUN_4JSZ" | awk '{print $1}')
      HASH_OK_4JSZ=$([[ "$SHA_RUN_4JSZ" == "$SHA_4JSZ" ]] && echo true || echo false)
    else
      HASH_OK_4JSZ=false
    fi
    if [[ -f "$META_3NQ9" ]] && grep -q '"embedding_cached":[[:space:]]*true' "$META_3NQ9"; then
      CACHED_OK_3NQ9=true
    else
      CACHED_OK_3NQ9=false
    fi
    if [[ -f "$META_4JSZ" ]] && grep -q '"embedding_cached":[[:space:]]*true' "$META_4JSZ"; then
      CACHED_OK_4JSZ=true
    else
      CACHED_OK_4JSZ=false
    fi

    if [[ "$HASH_OK_3NQ9" != "true" || "$HASH_OK_4JSZ" != "true" || "$CACHED_OK_3NQ9" != "true" || "$CACHED_OK_4JSZ" != "true" ]]; then
      echo "$CS,$REPEAT,$RUN_TAG,embedding_verify_failed,$SAMPLE_LOG,,$HASH_OK_3NQ9,$HASH_OK_4JSZ,$CACHED_OK_3NQ9,$CACHED_OK_4JSZ,true" >> "$MANIFEST"
      continue
    fi

    (
      export QDOCK_QPU_WORKDIR="$WORK_ROOT"
      export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
      export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
      export QDOCK_PDB_IDS="$CASES"
      export QDOCK_RUN_TAG="$RUN_TAG"
      export QDOCK_SAMPLER_CONFIG="$CFG"
      export QDOCK_REPORT=1
      "$PY_EVAL" "$ROOT/D_qpu_eval_fam.py"
    ) > "$EVAL_LOG" 2>&1 || {
      echo "$CS,$REPEAT,$RUN_TAG,eval_failed,$SAMPLE_LOG,$EVAL_LOG,$HASH_OK_3NQ9,$HASH_OK_4JSZ,$CACHED_OK_3NQ9,$CACHED_OK_4JSZ,true" >> "$MANIFEST"
      continue
    }

    echo "$CS,$REPEAT,$RUN_TAG,ok,$SAMPLE_LOG,$EVAL_LOG,$HASH_OK_3NQ9,$HASH_OK_4JSZ,$CACHED_OK_3NQ9,$CACHED_OK_4JSZ,true" >> "$MANIFEST"
  done
done

echo "DONE manifest=$MANIFEST"
