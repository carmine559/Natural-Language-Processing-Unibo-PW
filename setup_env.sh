#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup_env.sh — run ONCE on giano.cs.unibo.it (the submit host).
#
# Creates the Python venv under /scratch.hpc/ (the home quota is only
# 400 MB — far too small for a torch venv) and pre-downloads both models
# into HF_HOME on scratch (compute nodes have no internet access).
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

LIGHT_MODEL="${PROM_LIGHT_MODEL:-Qwen/Qwen3-14B}"
HEAVY_MODEL="${PROM_HEAVY_MODEL:-Qwen/Qwen3-32B}"

echo "Scratch dir  : ${SCRATCH}"
echo "Venv         : ${VENV}"
echo "HF_HOME      : ${HF_HOME}"
echo "Light model  : ${LIGHT_MODEL} (profiler/miner)"
echo "Heavy model  : ${HEAVY_MODEL} (solver/critic)"

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

# 4. Pre-download both models on giano — compute nodes have NO internet.
#    BF16 checkpoints: Qwen3-14B ~28 GB + Qwen3-32B ~62 GB. The cache may
#    still hold the old Qwen2.5 checkpoints (~90 GB, the rollback path):
#    check free space first, and if the quota is tight delete the Qwen2.5-14B
#    dir under ${HF_HOME}/hub (keep Qwen2.5-32B until Qwen3 is validated).
echo
echo "Scratch usage before model download:"
df -h "/scratch.hpc/${USER}" || true
du -sh "${HF_HOME}/hub" 2>/dev/null || true

hf download "${LIGHT_MODEL}"
hf download "${HEAVY_MODEL}"

echo
echo "Setup complete. Both models are cached in ${HF_HOME}."
echo "  Light: ${LIGHT_MODEL} (~9 GB 4-bit at runtime, thinking off)"
echo "  Heavy: ${HEAVY_MODEL} (~20 GB 4-bit at runtime, thinking mode)"
echo ""
echo "Submit with:  sbatch job.sbatch"
