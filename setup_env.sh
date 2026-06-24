#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup_env.sh — run ONCE on giano.cs.unibo.it (the submit host).
#
# Creates the Python venv and pre-downloads the model into /scratch.hpc/,
# because:
#   * the home quota is only 400 MB (a torch venv + model are several GB);
#   * compute nodes may have no outbound internet, so the model must be
#     cached on the shared filesystem before any job runs.
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

# 4. Enable Xet high-performance transfers (the new fast-download backend).
export HF_XET_HIGH_PERFORMANCE=1

# 5. HF_TOKEN — unauthenticated downloads are rate-limited and stall.
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "WARNING: HF_TOKEN is not set."
    echo "  Unauthenticated downloads are slow and often stall."
    echo "  Get a free token at https://huggingface.co/settings/tokens"
    echo "  then re-run with:  HF_TOKEN=hf_... bash setup_env.sh"
    echo ""
    echo "Continuing anyway (may be slow / unreliable) ..."
fi

# 6. Pre-download the model so compute nodes don't need internet.
#    `hf download` has built-in resume — if interrupted, just re-run.
echo "Downloading ${MODEL} into ${HF_HOME} ..."
hf download "${MODEL}" --cache-dir "${HF_HOME}"
echo "Model cached."

echo
echo "Setup complete."
echo "Now copy this repo + data under ${SCRATCH} (or run from a shared path)"
echo "and submit with:  sbatch job.sbatch"
