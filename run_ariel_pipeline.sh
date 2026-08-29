#!/usr/bin/env bash
# Run the complete ATLAS Open Data pipeline with Ariel's configuration.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="/workspace/config_opendata.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "error: Ariel config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

# This script owns the task order and shared run directory. Use the environment
# variables above instead of passing workflow-control options through to main.
for argument in "$@"; do
    case "${argument}" in
        --config|--config=*|--tasks|--tasks=*|--run-dir|--run-dir=*|\
        --scan-only|--plots-only|--merge-only|--batch-job-index|\
        --batch-job-index=*|--total-batch-jobs|--total-batch-jobs=*)
            echo "error: ${argument} is managed by run_ariel_pipeline.sh" >&2
            exit 2
            ;;
    esac
done

mapfile -t RUN_SETTINGS < <(
    "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    metadata = yaml.safe_load(config_file).get("run_metadata", {})

print(metadata.get("base_output_dir", "./output"))
print(metadata.get("run_name", "pipeline_run"))
PY
)

if [[ ${#RUN_SETTINGS[@]} -ne 2 ]]; then
    echo "error: could not read run metadata from ${CONFIG_PATH}" >&2
    exit 1
fi

BASE_OUTPUT_DIR="${RUN_SETTINGS[0]}"
RUN_NAME="${RUN_SETTINGS[1]}"
RUN_DIR="${ARIEL_RUN_DIR:-${BASE_OUTPUT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

echo "Ariel pipeline run directory: ${RUN_DIR}"

# A dry run only validates the complete configuration. A range scan cannot run
# until the first pass has produced processed SQLite shards.
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        exec "${PYTHON_BIN}" -u main.py \
            --config "${CONFIG_PATH}" \
            "$@" \
            --run-dir "${RUN_DIR}" \
            --tasks parsing,mass_calculating,post_processing,histogram_creation
    fi
done

echo "[1/3] Fetching, parsing, calculating invariant masses, and post-processing"
"${PYTHON_BIN}" -u main.py \
    --config "${CONFIG_PATH}" \
    "$@" \
    --run-dir "${RUN_DIR}" \
    --tasks parsing,mass_calculating,post_processing

echo "[2/3] Scanning processed arrays for global histogram ranges"
"${PYTHON_BIN}" -u main.py \
    --config "${CONFIG_PATH}" \
    "$@" \
    --run-dir "${RUN_DIR}" \
    --scan-only

echo "[3/3] Creating histograms and plots"
"${PYTHON_BIN}" -u main.py \
    --config "${CONFIG_PATH}" \
    "$@" \
    --run-dir "${RUN_DIR}" \
    --tasks histogram_creation

echo "Ariel pipeline completed: ${RUN_DIR}"
