#!/bin/bash
# ─────────────────────────────────────────────────────────────
# job.sbatch — submit the Prometeia pipeline to the DISI cluster.
#
# Submit from giano:   sbatch job.sbatch
# (Run setup_env.sh once beforehand.)
#
# Edit the placeholders below (name.surname) before submitting.
# To target the RTX 2080 Ti instead, change --partition to rtx2080 and
# --partition l40 -> rtx2080 in the python command (smaller GPU; if the 14B
# OOMs, set PROM_MODEL=Qwen/Qwen2.5-7B-Instruct below).
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=nlp_pw
#SBATCH --mail-type=ALL
#SBATCH --mail-user=carmine.santella@studio.unibo.it
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/carmine.santella/Natural-Language-Processing-Unibo-PW/slurm-%j.out
#SBATCH --chdir=/scratch.hpc/carmine.santella/Natural-Language-Processing-Unibo-PW
#SBATCH --gres=gpu:1

set -euo pipefail

# Caches / outputs live on scratch (home quota is only 400 MB).
# The model will be downloaded on the first run and cached in HF_HOME;
# subsequent runs reuse the cache.
export HF_HOME="/scratch.hpc/${USER}/hf_cache"
export PROM_OUTPUT_DIR="/scratch.hpc/${USER}/Natural-Language-Processing-Unibo-PW/outputs"
mkdir -p "${HF_HOME}" "${PROM_OUTPUT_DIR}"

VENV="/scratch.hpc/${USER}/Natural-Language-Processing-Unibo-PW/venv"

# Full validation run (reports accuracy). For the test submission, change
# --data to data/test_set_unlabelled_IT.tsv and pick a different --output.
"${VENV}/bin/python3" run_pipeline.py \
    --partition l40 \
    --data data/validation_set_IT.tsv \
    --output "${PROM_OUTPUT_DIR}/predictions.json"
