#!/bin/bash
# ===========================================================================
# ariel_submit_parsing_only.sh (legacy filename)
#
# Submit the complete single-job pipeline. The state machine performs parsing,
# invariant-mass calculation, post-processing, the global-range scan,
# histogram creation, and final plot generation in one Python process.
#
# Usage:
#   bash submit_parsing_only.sh
# ===========================================================================

set -e

CONFIG="config_opendata.yaml"
CPUS_PER_JOB=4
MEM_PER_JOB="20gb"
WALLTIME="72:00:00"
QUEUE="N"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Set run name and base dir directly (avoids YAML anchor issues) ------
# Edit these two values to match your config.yaml
RUN_NAME="ariel_config_run"
BASE_DIR="/storage/agrp/arielama/data"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/"
cp "${PIPELINE_DIR}/ariel_submit_parsing_only.sh" "${LOG_DIR}/"

echo "=============================================="
echo "ATLAS Pipeline — Complete Single-Process Run"
echo "=============================================="
echo "Config   : ${CONFIG}"
echo "Run dir  : ${RUN_DIR}"
echo "Walltime : ${WALLTIME}"
echo "=============================================="

JOB_ID=$(qsub <<EOF
#!/bin/bash
#PBS -q ${QUEUE}
#PBS -N atlas_parsing
#PBS -o ${LOG_DIR}/
#PBS -e ${LOG_DIR}/
#PBS -l select=1:ncpus=${CPUS_PER_JOB}:mem=${MEM_PER_JOB}
#PBS -l io=5
#PBS -l walltime=${WALLTIME}

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"

source ${PIPELINE_DIR}/atlasenv/bin/activate

python -u main.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --tasks parsing,mass_calculating,post_processing,histogram_creation \
    > "${LOG_DIR}/parsing.out" \
    2> "${LOG_DIR}/parsing.err"
EOF
)

echo "Submitted: ${JOB_ID}"
echo "Run dir  : ${RUN_DIR}"
echo ""
echo "Monitor  : qstat -u \$USER"
echo "Log      : tail -f ${LOG_DIR}/parsing.out"
