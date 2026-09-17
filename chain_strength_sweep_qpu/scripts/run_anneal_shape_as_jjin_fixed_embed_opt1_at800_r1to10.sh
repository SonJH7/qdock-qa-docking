#!/usr/bin/env bash
set -euo pipefail

# Anneal-shape sweep (AS-JJIN) with everything else fixed:
# - fixed penalty cases
# - fixed embeddings
# - fixed chain strengths (3nq9=1.0, 4jsz=0.75)
# - fixed annealing_time = 800 us
# - repeat r1..r10
# - ONLY anneal_schedule variants (6 shapes)
#
# This script is intentionally "paper-grade reproducible":
# it snapshots work/cases, verifies embedding hash match + embedding_cached=true,
# and logs all statuses to a single manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_ROOT="$ROOT/as_jjin"
WORK_ROOT="$SWEEP_ROOT/work_cases"
CONFIG_DIR="$SWEEP_ROOT/configs"
LOG_DIR="$SWEEP_ROOT/logs"
SAMPLE_ROOT="$SWEEP_ROOT/samples"
RESULT_DIR="$SWEEP_ROOT/results"
FIXED_DIR="$SWEEP_ROOT/fixed_embeddings"

PY_SAMPLE="${QDOCK_QPU_PYTHON:-python3}"
PY_EVAL="${QDOCK_EVAL_PYTHON:-python3}"
SOLVER_NAME="Advantage2_system1.13"
GRAPH_ID="01e1ea5685"
SOLVER="${SOLVER_NAME};graph_id=${GRAPH_ID}"

NUM_READS=1000
ANNEAL_TIME=800

# Fixed penalty/embedding source
SOURCE_CASE_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/cases/3nq9_Kdist7.500_Kmono19.000"
SOURCE_CASE_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/cases/4jsz_Kdist0.500_Kmono1.000"
CASE_3NQ9="$(basename "$SOURCE_CASE_3NQ9")"
CASE_4JSZ="$(basename "$SOURCE_CASE_4JSZ")"

EMB_SRC_3NQ9="$ROOT/BO_Online_v113_b100_q3_n20_x0.1_r5/work/samples/3nq9_Kdist7.500_Kmono19.000/online_bo_v113_b100_q3_n20_x0p1_r5_20260401_120636_s049/QUBOs/3nq9_ligand_embedding.json"
EMB_SRC_4JSZ="$ROOT/BO_Online_v113_b100_q4_n20_x0.1_r10/work/samples/4jsz_Kdist0.500_Kmono1.000/online_bo_v113_b100_quality_4jsz_n20_x0p1_r10_chain_20260405_025657_s011/QUBOs/4jsz_ligand_embedding.json"
EMB_3NQ9="$FIXED_DIR/3nq9_ligand_embedding.json"
EMB_4JSZ="$FIXED_DIR/4jsz_ligand_embedding.json"

# Fixed chain strengths (from latest selected opt point)
CS_3NQ9="1.0"
CS_4JSZ="0.75"

REPEATS=(1 2 3 4 5 6 7 8 9 10)

# Anneal-shape variants requested:
# (a) linear (b) cubic smoothstep (c) fourier
# (d) low-pass (from linear) (e) bang-bang (monotone) (f) pause-quench
ANNEAL_SCHEDULE_VARIANTS=(
  linear
  cubic_smoothstep
  fourier
  lowpass_linear
  bangbang_monotone
  pause_quench
)

STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$WORK_ROOT" "$CONFIG_DIR" "$LOG_DIR" "$SAMPLE_ROOT" "$RESULT_DIR" "$FIXED_DIR"

if [[ -z "${DWAVE_API_TOKEN:-}" ]]; then
  echo "ERROR: DWAVE_API_TOKEN is not set. Export token and re-run."
  exit 2
fi
export DWAVE_API_REGION="${DWAVE_API_REGION:-na-west-1}"

# Snapshot exact BO cases into local work root
rm -rf "$WORK_ROOT/$CASE_3NQ9" "$WORK_ROOT/$CASE_4JSZ"
cp -R "$SOURCE_CASE_3NQ9" "$WORK_ROOT/$CASE_3NQ9"
cp -R "$SOURCE_CASE_4JSZ" "$WORK_ROOT/$CASE_4JSZ"

cp -f "$EMB_SRC_3NQ9" "$EMB_3NQ9"
cp -f "$EMB_SRC_4JSZ" "$EMB_4JSZ"
SHA_3NQ9=$(shasum -a 256 "$EMB_3NQ9" | awk '{print $1}')
SHA_4JSZ=$(shasum -a 256 "$EMB_4JSZ" | awk '{print $1}')
QHASH_3NQ9=$(shasum -a 256 "$WORK_ROOT/$CASE_3NQ9/QUBOs/3nq9_ligand.npy" | awk '{print $1}')
QHASH_4JSZ=$(shasum -a 256 "$WORK_ROOT/$CASE_4JSZ/QUBOs/4jsz_ligand.npy" | awk '{print $1}')

MANIFEST="$RESULT_DIR/anneal_shape_as_jjin_fixedemb_opt1_at800_r1to10_manifest_${STAMP}.csv"
echo "mode,variant,annealing_time,chain_strength,repeat,ligand,case_name,run_tag,status,sample_log,eval_log,hash_ok,cached_ok,embedding_sha,qubo_sha" > "$MANIFEST"

echo "Fixed embedding SHA256:"
echo "  3nq9: $SHA_3NQ9"
echo "  4jsz: $SHA_4JSZ"
echo "Case QUBO SHA256:"
echo "  $CASE_3NQ9: $QHASH_3NQ9"
echo "  $CASE_4JSZ: $QHASH_4JSZ"
echo "Output root: $SWEEP_ROOT"

build_cfg() {
  local mode="$1" variant="$2" cs="$3" cfg="$4" emb="$5"
  MODE="$mode" VARIANT="$variant" CS="$cs" CFG="$cfg" EMB="$emb" \
  SOLVER="$SOLVER" NUM_READS="$NUM_READS" ANNEAL_TIME="$ANNEAL_TIME" SOLVER_NAME="$SOLVER_NAME" \
  "$PY_SAMPLE" - <<'PY'
import json
import os
from pathlib import Path

mode = os.environ["MODE"]
variant = os.environ["VARIANT"]
cs = float(os.environ["CS"])
cfg = Path(os.environ["CFG"])
emb_path = Path(os.environ["EMB"])
solver = os.environ["SOLVER"]
num_reads = int(os.environ["NUM_READS"])
anneal_time = float(os.environ["ANNEAL_TIME"])
solver_name = os.environ["SOLVER_NAME"]

sampler_kwargs = {
    "num_reads": num_reads,
    "chain_strength": cs,
    "annealing_time": anneal_time,
    "embedding_parameters": {"threads": 8},
    "return_embedding": True,
}

def schedule_points(name, total_t):
    # Keep schedules monotonic in (t, s), with s in [0, 1].
    # Definitions follow the manuscript-style schedule family:
    # - linear: s(u)=u
    # - cubic smoothstep: s(u)=3u^2-2u^3
    # - fourier-modulated: s(u)=u+sum(theta_m sin(m*pi*u)), endpoint-preserving
    # - low-pass RC from linear reference: s_raw=u-(tau/T)(1-exp(-uT/tau)), then normalize
    # - bang-bang (piecewise-constant/on-off), approximated with tiny jump epsilon
    # - pause-quench custom
    import math

    def clamp01(x):
        return min(max(x, 0.0), 1.0)

    def monotoneize(points):
        # force nondecreasing s in case floating errors create tiny violations
        out = []
        cur = 0.0
        for t, s in points:
            s = clamp01(float(s))
            if s < cur:
                s = cur
            cur = s
            out.append([float(t), s])
        out[0][1] = 0.0
        out[-1][1] = 1.0
        return out

    if name == "linear":
        return [[0.0, 0.0], [total_t, 1.0]]
    if name == "cubic_smoothstep":
        xs = [i / 10.0 for i in range(11)]
        pts = []
        for x in xs:
            s = 3.0 * x * x - 2.0 * x * x * x
            pts.append([total_t * x, s])
        return monotoneize(pts)
    if name == "fourier":
        # s(u)=u + sum_{m=1}^M theta_m sin(m*pi*u), u=t/T.
        # sufficient monotonicity condition: sum m|theta_m| <= 1/pi.
        # choose M=2, theta={1:0.06, 2:-0.01} -> 1*0.06 + 2*0.01 = 0.08 <= 1/pi.
        xs = [i / 10.0 for i in range(11)]
        theta = {1: 0.06, 2: -0.01}
        pts = []
        for x in xs:
            s = x
            for m, th in theta.items():
                s += th * math.sin(m * math.pi * x)
            pts.append([total_t * x, s])
        return monotoneize(pts)
    if name == "lowpass_linear":
        # First-order RC smoothing of linear reference r(t)=u=t/T:
        # tau * s~'(t) + s~(t) = r(t), s~(0)=0
        # closed form: s~(t)=u-(tau/T)*(1-exp(-t/tau))
        # normalize: s(t)=s~(t)/s~(T)
        tau = 0.12 * total_t
        denom = 1.0 - (tau / total_t) * (1.0 - math.exp(-total_t / tau))
        xs = [i / 10.0 for i in range(11)]
        pts = []
        for x in xs:
            t = total_t * x
            s_raw = x - (tau / total_t) * (1.0 - math.exp(-t / tau))
            s = s_raw / denom
            pts.append([t, s])
        return monotoneize(pts)
    if name == "bangbang_monotone":
        # Piecewise-constant bang-bang profile (on/off levels), approximated
        # with tiny epsilon transitions because anneal_schedule interpolates linearly.
        t1 = 0.25 * total_t
        t2 = 0.70 * total_t
        eps = 0.001 * total_t
        b0 = 0.0
        b1 = 0.58
        b2 = 1.0
        pts = [
            [0.0, 0.0],
            [t1, b0],
            [t1 + eps, b1],
            [t2, b1],
            [t2 + eps, b2],
            [total_t, b2],
        ]
        return monotoneize(pts)
    if name == "pause_quench":
        pts = [
            [0.0, 0.0],
            [360.0, 0.45],
            [520.0, 0.45],
            [760.0, 0.72],
            [total_t, 1.0],
        ]
        return monotoneize(pts)
    raise ValueError(f"unknown anneal schedule variant: {name}")

if mode == "anneal_schedule":
    sampler_kwargs.pop("annealing_time", None)
    sampler_kwargs["anneal_schedule"] = schedule_points(variant, anneal_time)
else:
    raise ValueError(f"unsupported mode for AS-JJIN: {mode}")

cfg_json = {
    "backend": "dwave_qpu",
    "backend_params": {"solver": solver},
    "sampler_kwargs": sampler_kwargs,
}

cfg.write_text(json.dumps(cfg_json, indent=2))
PY
}

run_one() {
  local mode="$1" variant="$2" repeat="$3" target="$4"

  local case_name ligand cs emb_fixed emb_sha qsha
  if [[ "$target" == "3nq9" ]]; then
    case_name="$CASE_3NQ9"; ligand="3nq9"; cs="$CS_3NQ9"; emb_fixed="$EMB_3NQ9"; emb_sha="$SHA_3NQ9"; qsha="$QHASH_3NQ9"
  else
    case_name="$CASE_4JSZ"; ligand="4jsz"; cs="$CS_4JSZ"; emb_fixed="$EMB_4JSZ"; emb_sha="$SHA_4JSZ"; qsha="$QHASH_4JSZ"
  fi

  local mode_label variant_label cs_label run_tag cfg sample_log eval_log
  mode_label="${mode//[^a-zA-Z0-9]/_}"
  variant_label="${variant//[^a-zA-Z0-9._-]/_}"
  cs_label="${cs/./p}"
  run_tag="asuite_${mode_label}_${variant_label}_${ligand}_at800_cs${cs_label}_r${repeat}_${STAMP}"
  cfg="$CONFIG_DIR/sampler_${mode_label}_${variant_label}_${ligand}_r${repeat}.json"
  sample_log="$LOG_DIR/${run_tag}_sample.log"
  eval_log="$LOG_DIR/${run_tag}_eval.log"

  build_cfg "$mode" "$variant" "$cs" "$cfg" "$emb_fixed"

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
    echo "$mode,$variant,$ANNEAL_TIME,$cs,$repeat,$ligand,$case_name,$run_tag,sample_failed,$sample_log,,false,false,$emb_sha,$qsha" >> "$MANIFEST"
    return
  }

  if grep -q "ok_cases=0/1" "$sample_log"; then
    echo "$mode,$variant,$ANNEAL_TIME,$cs,$repeat,$ligand,$case_name,$run_tag,sample_failed_no_ok,$sample_log,,false,false,$emb_sha,$qsha" >> "$MANIFEST"
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
    echo "$mode,$variant,$ANNEAL_TIME,$cs,$repeat,$ligand,$case_name,$run_tag,embedding_verify_failed,$sample_log,,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
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
    echo "$mode,$variant,$ANNEAL_TIME,$cs,$repeat,$ligand,$case_name,$run_tag,eval_failed,$sample_log,$eval_log,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
    return
  }

  echo "$mode,$variant,$ANNEAL_TIME,$cs,$repeat,$ligand,$case_name,$run_tag,ok,$sample_log,$eval_log,$hash_ok,$cached_ok,$emb_sha,$qsha" >> "$MANIFEST"
}

# ---------- sweep loops (anneal_schedule only) ----------
for r in "${REPEATS[@]}"; do
  for v in "${ANNEAL_SCHEDULE_VARIANTS[@]}"; do
    run_one "anneal_schedule" "$v" "$r" "3nq9"
    run_one "anneal_schedule" "$v" "$r" "4jsz"
  done
done

echo "DONE manifest=$MANIFEST"
