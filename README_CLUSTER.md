# Prometeia pipeline — DISI GPU cluster

Multi-agent pipeline for the Prometeia / EVALITA Italian financial MCQ task
(BOOKS / FINANCIALS / PAPER, options A–E). Refactored from the original
Kaggle + 2×T4 + Groq setup to run **fully local on a single cluster GPU**.

## Pipeline stages
`Profiler → Context Miner → Specialist Solver → Adversarial Critic → Weighted Aggregator`

All LLM stages share **one model** loaded once on the single GPU
(`Qwen/Qwen2.5-14B-Instruct`, 4-bit). The solver + tiebreaker also use this
local model unless you pass `--use-groq`.

## One-time setup (on giano)
```bash
ssh <name.surname>@giano.cs.unibo.it
cd <this-repo>
bash setup_env.sh        # venv + deps + model cache, all under /scratch.hpc/<user>/
```
Everything goes under `/scratch.hpc/<user>/` because the home quota is only 400 MB.
`setup_env.sh` pre-downloads the model so compute nodes need no internet.

## Submitting a job
Edit `job.sbatch` (replace `name.surname`, choose `--partition`), then:
```bash
sbatch job.sbatch
```
- **L40 (48 GB)** is the default — comfortably runs the 14B 4-bit model.
- **RTX 2080 Ti (11 GB)**: switch `--partition` to `rtx2080`; if the 14B OOMs on
  long contexts, set `export PROM_MODEL=Qwen/Qwen2.5-7B-Instruct` in `job.sbatch`.

## Running directly (debugging / interactive)
```bash
# smoke test (5 samples, labelled validation set)
python run_pipeline.py --partition l40 --data data/validation_set_IT.tsv --limit 5

# full validation run (prints accuracy breakdown)
python run_pipeline.py --partition l40 --data data/validation_set_IT.tsv

# test-set submission (no gold labels)
python run_pipeline.py --partition l40 --data data/test_set_unlabelled_IT.tsv \
                       --output outputs/test_predictions.json
```

### Key flags
| flag | meaning |
|------|---------|
| `--data` | dataset TSV (default `data/validation_set_IT.tsv`) |
| `--output` | predictions JSON (default `$PROM_OUTPUT_DIR/predictions.json`) |
| `--limit N` | process only the first N samples |
| `--partition {l40,rtx2080}` | selects the model preset |
| `--model NAME` | override the HF model |
| `--use-groq` | route solver + tiebreaker through Groq (needs `GROQ_API_KEY`) |
| `--no-resume` | ignore the existing results file |

### Environment variables
| var | purpose |
|-----|---------|
| `HF_HOME` | model cache (set to `/scratch.hpc/<user>/hf_cache`) |
| `PROM_OUTPUT_DIR` | where predictions + checkpoints are written |
| `PROM_MODEL` | force a specific model (overrides `--partition` preset) |

## Outputs
- `predictions.json` — `[{"id", "answer"}, ...]` submission file.
- `pipeline_results.json` — per-sample checkpoint (enables `--resume`).

The Kaggle notebooks under `notebooks/` are superseded by `run_pipeline.py` and
kept only for reference.
