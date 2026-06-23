# ════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — Context Miner
# ════════════════════════════════════════════════════════════════
# %% [markdown]
# # Notebook 02 — Context Miner
#
# Extracts evidence from the question text for each option.
# - **BOOKS** questions: returns numbered statements as passages (no BM25 needed).
# - **FINANCIALS** questions: BM25 over the embedded financial table.
# - **PAPER** questions: BM25 over the question text.
#
# **GPU:** cuda:1  |  **Output:** `/kaggle/working/miner_outputs.json`

# %%
import json, sys, pathlib, time
sys.path.insert(0, "/kaggle/working")

from prometeia_pipeline.utils.data_loader import load_data_folder
from prometeia_pipeline.utils.model_loader import get_miner_model
from prometeia_pipeline.stages.miner import ContextMiner
from prometeia_pipeline.schemas import ProfilerOutput

DATA_FOLDER   = "/kaggle/input/prometeia-evalita-2026"
PROFILER_PATH = "/kaggle/working/profiler_outputs.json"
OUTPUT_PATH   = "/kaggle/working/miner_outputs.json"
RESUME        = True

# %%
samples = load_data_folder(DATA_FOLDER)
samples_map = {s.sample_id: s for s in samples}

with open(PROFILER_PATH) as f:
    profilers_map = {
        r["sample_id"]: ProfilerOutput(**{k: v for k, v in r.items() if k != "sample_id"})
        for r in json.load(f)
    }

# %%
done: dict[str, dict] = {}
if RESUME and pathlib.Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH) as f:
        done = {r["sample_id"]: r for r in json.load(f)}
    print(f"Resuming — {len(done)} done.")

pairs = [
    (samples_map[sid], profilers_map[sid])
    for sid in profilers_map
    if sid not in done and sid in samples_map
]
print(f"To mine: {len(pairs)}")

# %%
model  = get_miner_model()
miner  = ContextMiner(model)
results = dict(done)
start   = time.time()

for i, (sample, profiler) in enumerate(pairs):
    out = miner.run(sample, profiler)
    results[sample.sample_id] = {"sample_id": sample.sample_id, **out.model_dump()}

    if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(pairs)}]  {elapsed:.0f}s elapsed")

print(f"\n✓ {len(results)} miner outputs → {OUTPUT_PATH}")

# %%
from collections import Counter
cov = Counter(r["evidence_coverage"] for r in results.values())
print("Coverage:", dict(cov))
print("Tables found:", sum(1 for r in results.values() if r["tables_found"]))


# ════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Specialist Solver
# ════════════════════════════════════════════════════════════════
# %% [markdown]
# # Notebook 03 — Specialist Solver
#
# Calls Groq with a type-conditioned system prompt for each question.
# High-difficulty questions get self-consistency (3 samples).
#
# **API:** Groq  |  **Output:** `/kaggle/working/solver_outputs.json`

# %%
import json, os, sys, pathlib, time
sys.path.insert(0, "/kaggle/working")
from kaggle_secrets import UserSecretsClient
os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")

from prometeia_pipeline.utils.data_loader import load_data_folder
from prometeia_pipeline.utils.groq_client import GroqClient
from prometeia_pipeline.stages.solver import SpecialistSolver
from prometeia_pipeline.schemas import ProfilerOutput, MinerOutput, OptionEvidence, PassageScore

DATA_FOLDER   = "/kaggle/input/prometeia-evalita-2026"
PROFILER_PATH = "/kaggle/working/profiler_outputs.json"
MINER_PATH    = "/kaggle/working/miner_outputs.json"
OUTPUT_PATH   = "/kaggle/working/solver_outputs.json"
RESUME        = True

# %%
samples = load_data_folder(DATA_FOLDER)
samples_map = {s.sample_id: s for s in samples}

def _load_profilers(path):
    with open(path) as f:
        return {
            r["sample_id"]: ProfilerOutput(**{k: v for k, v in r.items() if k != "sample_id"})
            for r in json.load(f)
        }

def _load_miners(path):
    out = {}
    with open(path) as f:
        for r in json.load(f):
            sid = r["sample_id"]
            out[sid] = MinerOutput(
                relevant_passages=[PassageScore(**p) for p in r["relevant_passages"]],
                option_scores={k: OptionEvidence(**v) for k, v in r["option_scores"].items()},
                evidence_coverage=r["evidence_coverage"],
                tables_found=r["tables_found"],
                numerical_values_extracted=r["numerical_values_extracted"],
                missing_information=r.get("missing_information"),
            )
    return out

profilers_map = _load_profilers(PROFILER_PATH)
miners_map    = _load_miners(MINER_PATH)

# %%
done: dict[str, dict] = {}
if RESUME and pathlib.Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH) as f:
        done = {r["sample_id"]: r for r in json.load(f)}
    print(f"Resuming — {len(done)} done.")

triples = [
    (samples_map[sid], profilers_map[sid], miners_map[sid])
    for sid in profilers_map
    if sid not in done and sid in samples_map and sid in miners_map
]
print(f"To solve: {len(triples)}")

# %%
groq   = GroqClient()
solver = SpecialistSolver(groq)
results = dict(done)
start   = time.time()

for i, (sample, profiler, miner) in enumerate(triples):
    out = solver.run(sample, profiler, miner)
    results[sample.sample_id] = {
        "sample_id":         sample.sample_id,
        "proposed_answer":   out.proposed_answer,
        "confidence":        out.confidence,
        "reasoning_steps":   out.reasoning_steps,
        "miner_alignment":   out.miner_alignment,
        "consistency_score": out.consistency_score,
        "sc_votes":          out.sc_votes,
    }

    if (i + 1) % 20 == 0 or (i + 1) == len(triples):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(triples)}]  {elapsed:.0f}s elapsed")
        groq.print_stats()

print(f"\n✓ {len(results)} solver outputs → {OUTPUT_PATH}")

# %%
from collections import Counter
ans_dist = Counter(r["proposed_answer"] for r in results.values())
print("Answer distribution:", dict(sorted(ans_dist.items())))
avg_conf = sum(r["confidence"] for r in results.values()) / len(results)
print(f"Mean confidence: {avg_conf:.3f}")


# ════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Adversarial Critic
# ════════════════════════════════════════════════════════════════
# %% [markdown]
# # Notebook 04 — Adversarial Critic
#
# Challenges every solver answer with the local 7B on cuda:0.
#
# **GPU:** cuda:0  |  **Output:** `/kaggle/working/critic_outputs.json`

# %%
import json, sys, pathlib, time
sys.path.insert(0, "/kaggle/working")

from prometeia_pipeline.utils.data_loader import load_data_folder
from prometeia_pipeline.utils.model_loader import get_critic_model
from prometeia_pipeline.stages.critic import AdversarialCritic
from prometeia_pipeline.schemas import (
    ProfilerOutput, MinerOutput, SolverOutput,
    OptionEvidence, PassageScore,
)

DATA_FOLDER   = "/kaggle/input/prometeia-evalita-2026"
PROFILER_PATH = "/kaggle/working/profiler_outputs.json"
MINER_PATH    = "/kaggle/working/miner_outputs.json"
SOLVER_PATH   = "/kaggle/working/solver_outputs.json"
OUTPUT_PATH   = "/kaggle/working/critic_outputs.json"
RESUME        = True

# %%
samples = load_data_folder(DATA_FOLDER)
samples_map = {s.sample_id: s for s in samples}

# Reuse loader helpers from notebook 03
def _load_profilers(path):
    with open(path) as f:
        return {r["sample_id"]: ProfilerOutput(**{k:v for k,v in r.items() if k!="sample_id"})
                for r in json.load(f)}

def _load_miners(path):
    out = {}
    with open(path) as f:
        for r in json.load(f):
            out[r["sample_id"]] = MinerOutput(
                relevant_passages=[PassageScore(**p) for p in r["relevant_passages"]],
                option_scores={k: OptionEvidence(**v) for k, v in r["option_scores"].items()},
                evidence_coverage=r["evidence_coverage"],
                tables_found=r["tables_found"],
                numerical_values_extracted=r["numerical_values_extracted"],
                missing_information=r.get("missing_information"),
            )
    return out

def _load_solvers(path):
    with open(path) as f:
        return {
            r["sample_id"]: SolverOutput(
                proposed_answer=r["proposed_answer"], confidence=r["confidence"],
                reasoning_steps=r["reasoning_steps"],  miner_alignment=r["miner_alignment"],
                consistency_score=r["consistency_score"], sc_votes=r["sc_votes"],
            ) for r in json.load(f)
        }

profilers_map = _load_profilers(PROFILER_PATH)
miners_map    = _load_miners(MINER_PATH)
solvers_map   = _load_solvers(SOLVER_PATH)

# %%
done: dict[str, dict] = {}
if RESUME and pathlib.Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH) as f:
        done = {r["sample_id"]: r for r in json.load(f)}
    print(f"Resuming — {len(done)} done.")

quads = [
    (samples_map[sid], profilers_map[sid], miners_map[sid], solvers_map[sid])
    for sid in profilers_map
    if sid not in done
    and all(sid in m for m in [samples_map, miners_map, solvers_map])
]
print(f"To critique: {len(quads)}")

# %%
model  = get_critic_model()
critic = AdversarialCritic(model)
results = dict(done)
start   = time.time()

for i, (sample, profiler, miner, solver) in enumerate(quads):
    out = critic.run(sample, profiler, miner, solver)
    results[sample.sample_id] = {"sample_id": sample.sample_id, **out.model_dump()}

    if (i + 1) % 50 == 0 or (i + 1) == len(quads):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(quads)}]  {elapsed:.0f}s elapsed")

print(f"\n✓ {len(results)} critic outputs → {OUTPUT_PATH}")

# %%
from collections import Counter
verdicts = Counter(r["overall_verdict"] for r in results.values())
flips    = [r for r in results.values() if r["overall_verdict"] == "flip"]
print("Verdict distribution:", dict(verdicts))
print(f"Flips: {len(flips)} ({100*len(flips)/len(results):.1f}%)")


# ════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — Aggregator + Evaluation
# ════════════════════════════════════════════════════════════════
# %% [markdown]
# # Notebook 05 — Weighted Aggregator + Evaluation
#
# Deterministic CPU scoring + optional Groq tiebreaker.
# Produces final predictions and a full accuracy breakdown.

# %%
import json, os, sys, pathlib
sys.path.insert(0, "/kaggle/working")
from kaggle_secrets import UserSecretsClient
os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")

from prometeia_pipeline.utils.data_loader import load_data_folder, save_predictions
from prometeia_pipeline.utils.groq_client import GroqClient
from prometeia_pipeline.stages.aggregator import WeightedAggregator
from prometeia_pipeline.schemas import (
    ProfilerOutput, MinerOutput, SolverOutput, CriticOutput,
    OptionEvidence, PassageScore, PipelineResult, AggregatorOutput,
)

DATA_FOLDER  = "/kaggle/input/prometeia-evalita-2026"
PRED_PATH    = "/kaggle/working/predictions.json"

# %%
samples     = load_data_folder(DATA_FOLDER)
samples_map = {s.sample_id: s for s in samples}

def _j(p):
    with open(p) as f: return json.load(f)

def _load_profilers(path):
    return {r["sample_id"]: ProfilerOutput(**{k:v for k,v in r.items() if k!="sample_id"})
            for r in _j(path)}

def _load_miners(path):
    out = {}
    for r in _j(path):
        out[r["sample_id"]] = MinerOutput(
            relevant_passages=[PassageScore(**p) for p in r["relevant_passages"]],
            option_scores={k: OptionEvidence(**v) for k, v in r["option_scores"].items()},
            evidence_coverage=r["evidence_coverage"],
            tables_found=r["tables_found"],
            numerical_values_extracted=r["numerical_values_extracted"],
            missing_information=r.get("missing_information"),
        )
    return out

def _load_solvers(path):
    return {r["sample_id"]: SolverOutput(
        proposed_answer=r["proposed_answer"], confidence=r["confidence"],
        reasoning_steps=r["reasoning_steps"], miner_alignment=r["miner_alignment"],
        consistency_score=r["consistency_score"], sc_votes=r["sc_votes"],
    ) for r in _j(path)}

def _load_critics(path):
    return {r["sample_id"]: CriticOutput(**{k:v for k,v in r.items() if k!="sample_id"})
            for r in _j(path)}

profilers_map = _load_profilers("/kaggle/working/profiler_outputs.json")
miners_map    = _load_miners("/kaggle/working/miner_outputs.json")
solvers_map   = _load_solvers("/kaggle/working/solver_outputs.json")
critics_map   = _load_critics("/kaggle/working/critic_outputs.json")

common = sorted(set(profilers_map) & set(miners_map) & set(solvers_map) & set(critics_map))
print(f"Aggregating {len(common)} samples")

# %%
groq       = GroqClient()
aggregator = WeightedAggregator(groq)
results    = []

for sid in common:
    sample  = samples_map[sid]
    agg_out = aggregator.run(
        sample, profilers_map[sid], miners_map[sid],
        solvers_map[sid], critics_map[sid],
    )
    results.append(PipelineResult(
        sample_id=sid, final_answer=agg_out.final_answer,
        profiler=profilers_map[sid], miner=miners_map[sid],
        solver=solvers_map[sid], critic=critics_map[sid],
        aggregator=agg_out, gold=sample.gold,
    ))

save_predictions(results, PRED_PATH)

# %% — Evaluation (only meaningful when gold labels exist)
import pandas as pd

labelled = [r for r in results if r.gold is not None]
if not labelled:
    print("No gold labels — test set. Predictions saved.")
else:
    overall = sum(r.is_correct for r in labelled) / len(labelled)
    print(f"\nOverall accuracy: {overall:.4f}  ({sum(r.is_correct for r in labelled)}/{len(labelled)})")

    # Build a flat DataFrame for easy slicing
    rows = []
    for r in labelled:
        rows.append({
            "sample_id":    r.sample_id,
            "category":     r.profiler.question_type if r.profiler else "?",
            "difficulty":   r.profiler.difficulty    if r.profiler else "?",
            "verdict":      r.critic.overall_verdict if r.critic   else "?",
            "tier":         r.aggregator.confidence_tier if r.aggregator else "?",
            "correct":      r.is_correct,
        })
    df = pd.DataFrame(rows)

    print("\n── By question type ──")
    print(df.groupby("category")["correct"]
            .agg(["sum","count","mean"])
            .rename(columns={"sum":"correct","count":"total","mean":"accuracy"})
            .sort_values("accuracy", ascending=False)
            .to_string())

    print("\n── By difficulty ──")
    print(df.groupby("difficulty")["correct"]
            .agg(["sum","count","mean"])
            .rename(columns={"sum":"correct","count":"total","mean":"accuracy"})
            .to_string())

    print("\n── By critic verdict ──")
    print(df.groupby("verdict")["correct"]
            .agg(["sum","count","mean"])
            .rename(columns={"sum":"correct","count":"total","mean":"accuracy"})
            .to_string())

    print("\n── By confidence tier ──")
    print(df.groupby("tier")["correct"]
            .agg(["sum","count","mean"])
            .rename(columns={"sum":"correct","count":"total","mean":"accuracy"})
            .to_string())


# ════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — Full end-to-end pipeline
# ════════════════════════════════════════════════════════════════
# %% [markdown]
# # Notebook 06 — Full End-to-End Pipeline
#
# Runs all stages in one notebook via the `PrometeiaPipeline` orchestrator.
# Use for the test set submission or quick ablations on small slices.
# For large datasets prefer notebooks 01–05 (independent checkpointing).

# %%
import os, sys
sys.path.insert(0, "/kaggle/working")
from kaggle_secrets import UserSecretsClient
os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")

from prometeia_pipeline.orchestrator import PrometeiaPipeline
from prometeia_pipeline.utils.data_loader import load_data_folder, save_predictions

DATA_FOLDER = "/kaggle/input/prometeia-evalita-2026"
OUTPUT_PATH = "/kaggle/working/test_predictions.json"
SANITY_N    = 5    # set to None to run everything

# %%
samples = load_data_folder(DATA_FOLDER)
if SANITY_N:
    samples = samples[:SANITY_N]
    print(f"SANITY CHECK — running on first {SANITY_N} samples only.")

# %%
pipeline = PrometeiaPipeline()   # loads all models ~2 min

results = pipeline.run_batch(
    samples,
    output_path=OUTPUT_PATH,
    resume=True,
    save_every=10,
    verbose=True,
)

save_predictions(results, OUTPUT_PATH)

labelled = [r for r in results if r.gold is not None]
if labelled:
    acc = sum(r.is_correct for r in labelled) / len(labelled)
    print(f"\nAccuracy: {acc:.4f}")

pipeline.groq.print_stats()
