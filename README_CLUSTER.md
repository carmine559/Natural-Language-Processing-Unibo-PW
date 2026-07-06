# Prometeia pipeline — DISI GPU cluster

Multi-agent pipeline for the Prometeia / EVALITA Italian financial MCQ task
(BOOKS / FINANCIALS / PAPER, options A–E). Refactored from the original
Kaggle + 2×T4 + Groq setup to run **fully local on a single cluster GPU**.

## Pipeline stages
`Profiler → Context Miner → Specialist Solver → Adversarial Critic → Weighted Aggregator`

Two models load once on the single GPU (both NF4 4-bit):

- **light** `Qwen/Qwen3-14B` (~9 GB) — profiler + miner, thinking off (fast).
- **heavy** `Qwen/Qwen3-32B` (~20 GB) — solver, critic, statement evaluator and
  tiebreaker. Solver/evaluator/tiebreaker run with **thinking mode on**: the
  model reasons in a `<think>…</think>` block (stripped automatically) before
  answering, with Qwen3-recommended sampling (temp 0.6 / top_p 0.95 / top_k 20)
  and a larger generation budget (`ModelConfig.thinking_max_new_tokens`,
  default 3072; auto-doubled once if a think block truncates). The **critic
  runs thinking-off** (`ModelConfig.critic_enable_thinking`): the thinking
  critic over-flipped (27 flips / 220 samples, 33% correct) and dominated
  runtime. Critic `flip` verdicts are neutralized to `confirm` in the
  aggregator (`AggregatorConfig.flip_downgrade_to`).

The solver + tiebreaker use the local heavy model unless you pass `--use-groq`.

## One-time setup (on giano)
```bash
ssh <name.surname>@giano.cs.unibo.it
cd <this-repo>
bash setup_env.sh        # venv + deps + model cache, all under /scratch.hpc/<user>/
```
Everything goes under `/scratch.hpc/<user>/` because the home quota is only 400 MB.
`setup_env.sh` pre-downloads both models with `hf download`; this is optional
(compute nodes can reach Hugging Face and download at first run), but it keeps
GPU walltime from being spent on ~90 GB of checkpoint downloads.

## Submitting a job
Edit `job.sbatch` (replace `name.surname`, choose `--partition`), then:
```bash
sbatch job.sbatch
```
- **L40 (48 GB)** is the default — runs 14B + 32B together (~28 GB weights,
  ~19 GB headroom for KV caches).
- **RTX 2080 Ti (11 GB)**: switch `--partition` to `rtx2080` (single 14B for all
  stages, thinking off); if the 14B OOMs on long contexts, set
  `export PROM_LIGHT_MODEL=Qwen/Qwen3-8B PROM_HEAVY_MODEL=Qwen/Qwen3-8B`
  in `job.sbatch`.

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
| `PROM_LIGHT_MODEL` / `PROM_HEAVY_MODEL` | force specific models (override the `--partition` preset) |

## Outputs
- `predictions.json` — `[{"id", "answer"}, ...]` submission file.
- `pipeline_results.json` — per-sample checkpoint (enables `--resume`).

The Kaggle notebooks under `notebooks/` are superseded by `run_pipeline.py` and
kept only for reference.
