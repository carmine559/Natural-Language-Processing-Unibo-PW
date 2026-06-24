#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup_env.sh — run ONCE on giano.cs.unibo.it (the submit host).
#
# Creates the Python venv under /scratch.hpc/ (the home quota is only
# 400 MB — far too small for a torch venv). The model itself is downloaded
# at runtime by transformers on the first job, cached in HF_HOME on scratch.
#
# Usage:
#   ssh <name.surname>@giano.cs.unibo.it
#   cd <where-this-repo-lives>
#   bash setup_env.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRATCH="/scratch.hpc/${USER}/Natural-Language-Processing-Unibo-PW"
VENV="${SCRATCH}/venv"
export HF_HOME="/scratch.hpc/${USER}/hf_cache"

MODEL="${PROM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

echo "Scratch dir : ${SCRATCH}"
echo "Venv        : ${VENV}"
echo "HF_HOME     : ${HF_HOME}"
echo "Model       : ${MODEL}"

# Redirect pip / system temp to scratch — /tmp on giano is tiny and shared,
# which causes "No space left on device" when downloading large wheels (torch, triton).
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${SCRATCH}" "${HF_HOME}" "${TMPDIR}"

# 1. Virtual environment
if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip3 install --no-cache-dir --upgrade pip

# 2. torch for CUDA 11.8 (per the DISI cluster instructions)
pip3 install torch --no-cache-dir --index-url https://download.pytorch.org/whl/cu118

# 3. The rest of the dependencies
pip3 install --no-cache-dir -r requirements.txt

echo
echo "Setup complete. The model (${MODEL}) will be downloaded automatically"
echo "on the first job run and cached in ${HF_HOME} for subsequent runs."
echo ""
echo "Submit with:  sbatch job.sbatch"
