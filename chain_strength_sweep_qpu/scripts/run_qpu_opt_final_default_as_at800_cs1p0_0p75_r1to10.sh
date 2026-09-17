#!/usr/bin/env bash
set -euo pipefail

# Final QPU run with default anneal schedule (implicit linear), r1..r10:
# - 3nq9: Kdist=7.5, Kmono=19.0, cs=1.0, at=800
# - 4jsz: Kdist=0.5, Kmono=1.0, cs=0.75, at=800
# - no explicit anneal_schedule / readout_thermalization / anneal_offsets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="$ROOT/qpu_opt_final_default_as_at800_cs1p0_0p75_r1to10"
WORK_ROOT="$RUN_ROOT/work_cases"
CONFIG_DIR="$RUN_ROOT/configs"
LOG_DIR="$RUN_ROOT/logs"
SAMPLE_ROOT="$RUN_ROOT/samples"
RESULT_DIR="$RUN_ROOT/results"
FIXED_DIR="$RUN_ROOT/fixed_embeddings"

PY_SAMPLE="${QDOCK_QPU_PYTHON:-python3}"
PY_EVAL="${QDOCK_EVAL_PYTHON:-python3}"
SOLVER_NAME="Advantage2_system1.13"
GRAPH_ID="01e1ea5685"
SOLVER="${SOLVER_NAME};graph_id=${GRAPH_ID}"

NUM_READS=1000
ANNEAL_TIME=800

SOURCE_CASE_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/cases/3nq9_Kdist7.500_Kmono19.000"
SOURCE_CASE_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/cases/4jsz_Kdist0.500_Kmono1.000"
CASE_3NQ9="$(basename "$SOURCE_CASE_3NQ9")"
CASE_4JSZ="$(basename "$SOURCE_CASE_4JSZ")"

EMB_SRC_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/samples/3nq9_Kdist7.500_Kmono19.000/online_bo_v113_b100_q3_n20_x0p1_r5_20260401_120636_s049/QUBOs/3nq9_ligand_embedding.json"
EMB_SRC_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/samples/4jsz_Kdist0.500_Kmono1.000/online_bo_v113_b100_quality_4jsz_n20_x0p1_r10_chain_20260405_025657_s011/QUBOs/4jsz_ligand_embedding.json"
EMB_3NQ9="$FIXED_DIR/3nq9_ligand_embedding.json"
EMB_4JSZ="$FIXED_DIR/4jsz_ligand_embedding.json"

CS_3NQ9="1.0"
CS_4JSZ="0.75"
REPEATS=(1 2 3 4 5 6 7 8 9 10)
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$WORK_ROOT" "$CONFIG_DIR" "$LOG_DIR" "$SAMPLE_ROOT" "$RESULT_DIR" "$FIXED_DIR"

if [[ -z "${DWAVE_API_TOKEN:-}" ]]; then
  echo "ERROR: DWAVE_API_TOKEN is not set. Export token and re-run."
  exit 2
fi
export DWAVE_API_REGION="${DWAVE_API_REGION:-na-west-1}"

rm -rf "$WORK_ROOT/$CASE_3NQ9" "$WORK_ROOT/$CASE_4JSZ"
cp -R "$SOURCE_CASE_3NQ9" "$WORK_ROOT/$CASE_3NQ9"
cp -R "$SOURCE_CASE_4JSZ" "$WORK_ROOT/$CASE_4JSZ"

cp -f "$EMB_SRC_3NQ9" "$EMB_3NQ9"
cp -f "$EMB_SRC_4JSZ" "$EMB_4JSZ"
SHA_3NQ9=$(shasum -a 256 "$EMB_3NQ9" | awk '{print $1}')
SHA_4JSZ=$(shasum -a 256 "$EMB_4JSZ" | awk '{print $1}')
QHASH_3NQ9=$(shasum -a 256 "$WORK_ROOT/$CASE_3NQ9/QUBOs/3nq9_ligand.npy" | awk '{print $1}')
QHASH_4JSZ=$(shasum -a 256 "$WORK_ROOT/$CASE_4JSZ/QUBOs/4jsz_ligand.npy" | awk '{print $1}')

MANIFEST="$RESULT_DIR/qpu_opt_final_default_as_at800_cs1p0_0p75_r1to10_manifest_${STAMP}.csv"
echo "ligand,case_name,repeat,chain_strength,annealing_time,setting,run_tag,status,sample_log,eval_log,hash_ok,cached_ok,embedding_sha,qubo_sha" > "$MANIFEST"

echo "Fixed embedding SHA256:"
echo "  3nq9: $SHA_3NQ9"
echo "  4jsz: $SHA_4JSZ"
echo "Case QUBO SHA256:"
echo "  $CASE_3NQ9: $QHASH_3NQ9"
echo "  $CASE_4JSZ: $QHASH_4JSZ"
echo "Output root: $RUN_ROOT"

build_cfg() {
  local ligand="$1"
  local cs="$2"
  local cfg="$3"
  local emb="$4"

  CS="$cs" CFG="$cfg" SOLVER="$SOLVER" NUM_READS="$NUM_READS" ANNEAL_TIME="$ANNEAL_TIME" \
  "$PY_SAMPLE" - <<'PY'
import json
import os
from pathlib import Path

cs = float(os.environ["CS"])
cfg = Path(os.environ["CFG"])
solver = os.environ["SOLVER"]
num_reads = int(os.environ["NUM_READS"])
anneal_time = float(os.environ["ANNEAL_TIME"])

sampler_kwargs = {
    "num_reads": num_reads,
    "chain_strength": cs,
    "annealing_time": anneal_time,
    "embedding_parameters": {"threads": 8},
    "return_embedding": True,
}

cfg_json = {
    "backend": "dwave_qpu",
    "backend_params": {"solver": solver},
    "sampler_kwargs": sampler_kwargs,
}
cfg.write_text(json.dumps(cfg_json, indent=2))
PY
}

run_one() {
  local ligand="$1"
  local repeat="$2"

  local case_name cs emb_fixed emb_sha qsha setting
  if [[ "$ligand" == "3nq9" ]]; then
    case_name="$CASE_3NQ9"
    cs="$CS_3NQ9"
    emb_fixed="$EMB_3NQ9"
    emb_sha="$SHA_3NQ9"
    qsha="$QHASH_3NQ9"
    setting="default_linear"
  else
    case_name="$CASE_4JSZ"
    cs="$CS_4JSZ"
    emb_fixed="$EMB_4JSZ"
    emb_sha="$SHA_4JSZ"
    qsha="$QHASH_4JSZ"
    setting="default_linear"
  fi

  local cs_label run_tag cfg sample_log eval_log
  cs_label="${cs/./p}"
  run_tag="finalopt_defaultas_${ligand}_cs${cs_label}_at800_r${repeat}_${STAMP}"
  cfg="$CONFIG_DIR/sampler_${ligand}_r${repeat}.json"
  sample_log="$LOG_DIR/${run_tag}_sample.log"
  eval_log="$LOG_DIR/${run_tag}_eval.log"

  build_cfg "$ligand" "$cs" "$cfg" "$emb_fixed"
  mkdir -p "$SAMPLE_ROOT/$case_name/$run_tag/QUBOs"
  cp -f "$emb_fixed" "$SAMPLE_ROOT/$case_name/$run_tag/QUBOs/${ligand}_ligand_embedding.json"

  (
    export QDOCK_QPU_WORKDIR="$WORK_ROOT"
    export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
    export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
    export QDOCK_PDB_IDS="$case_name"
    export QDOCK_RUN_TAG="$run_tag"
    export QDOCK_SAMPLER_CONFIG="$cfg"
    export QDOCK_EMBEDDING_REUSE=0
    "$PY_SAMPLE" "$ROOT/D_qpu_sample_fam.py"
  ) > "$sample_log" 2>&1 || {
    echo "$ligand,$case_name,$repeat,$cs,$ANNEAL_TIME,$setting,$run_tag,sample_failed,$sample_log,,false,false,$emb_sha,$qsha" >> "$MANIFEST"
    return
  }

  if grep -q "ok_cases=0/1" "$sample_log"; then
    echo "$ligand,$case_name,$repeat,$cs,$ANNEAL_TIME,$setting,$run_tag,sample_failed_no_ok,$sample_log,,false,false,$emb_sha,$qsha" >> "$MANIFEST"
    return
  fi

  local emb_run meta run_sha hash_ok cached_ok
  emb_run="$SAMPLE_ROOT/$case_name/$run_tag/QUBOs/${ligand}_ligand_embedding.json"
  meta="$SAMPLE_ROOT/$case_name/$run_tag/${ligand}_ligand_sampler_meta.json"

  if [[ -f "$emb_run" ]]; then
    run_sha=$(shasum -a 256 "$emb_run" | awk '{print $1}')
    hash_ok=$([[ "$run_sha" == "$emb_sha" ]] && echo true || echo false)
  else
    hash_ok=false
  fi
  if [[ -f "$meta" ]] && grep -q '"embedding_cached":[[:space:]]*true' "$meta"; then
    cached_ok=true
  else
    cached_ok=false
  fi

  if [[ "$hash_ok" != "true" || "$cached_ok" != "true" ]]; then
    echo "$ligand,$case_name,$repeat,$cs,$ANNEAL_TIME,$setting,$run_tag,embedding_verify_failed,$sample_log,,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
    return
  fi

  (
    export QDOCK_QPU_WORKDIR="$WORK_ROOT"
    export QDOCK_SAMPLE_DIR="$SAMPLE_ROOT"
    export QDOCK_EVAL_DIR="$SAMPLE_ROOT"
    export QDOCK_PDB_IDS="$case_name"
    export QDOCK_RUN_TAG="$run_tag"
    export QDOCK_SAMPLER_CONFIG="$cfg"
    export QDOCK_REPORT=1
    "$PY_EVAL" "$ROOT/D_qpu_eval_fam.py"
  ) > "$eval_log" 2>&1 || {
    echo "$ligand,$case_name,$repeat,$cs,$ANNEAL_TIME,$setting,$run_tag,eval_failed,$sample_log,$eval_log,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
    return
  }

  echo "$ligand,$case_name,$repeat,$cs,$ANNEAL_TIME,$setting,$run_tag,ok,$sample_log,$eval_log,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
}

for r in "${REPEATS[@]}"; do
  run_one "3nq9" "$r"
  run_one "4jsz" "$r"
done

echo "DONE manifest=$MANIFEST"
