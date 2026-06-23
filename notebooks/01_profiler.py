# %% [markdown]
# # Notebook 01 — Question Profiler
#
# Classifies every question into a question type and extracts structured metadata.
#
# **Routing logic (no LLM wasted):**
# - BOOKS  → `fast_classify()` handles it directly (always statement_combination)
# - FINANCIALS / PAPER → local 7B on cuda:0
#
# **Output:** `/kaggle/working/profiler_outputs.json`
# **Runtime:** BOOKS are instant; expect ~8–12s per FINANCIALS/PAPER question on T4.

# %% — Imports
import json, os, sys, pathlib, time
sys.path.insert(0, "/kaggle/working")
from kaggle_secrets import UserSecretsClient
os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")

from prometeia_pipeline.utils.data_loader import load_data_folder
from prometeia_pipeline.utils.model_loader import get_profiler_model
from prometeia_pipeline.stages.profiler import QuestionProfiler, fast_classify

# %% — Config
DATA_FOLDER = "/kaggle/input/prometeia-evalita-2026"
OUTPUT_PATH = "/kaggle/working/profiler_outputs.json"
RESUME      = True

# %% — Load data
samples = load_data_folder(DATA_FOLDER)
print(f"\n{len(samples)} samples total")

from collections import Counter
cats = Counter(s.category for s in samples)
print(f"Category breakdown: {dict(cats)}")

# %% — Resume
done: dict[str, dict] = {}
if RESUME and pathlib.Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH) as f:
        done = {r["sample_id"]: r for r in json.load(f)}
    print(f"Resuming — {len(done)} already done.")

remaining = [s for s in samples if s.sample_id not in done]

# Split by whether LLM is needed
fast_samples = [s for s in remaining if s.category == "BOOKS"]
llm_samples  = [s for s in remaining if s.category != "BOOKS"]
print(f"To profile: {len(remaining)} total  "
      f"({len(fast_samples)} fast / {len(llm_samples)} LLM)")

# %% — Fast-classify BOOKS (no GPU needed)
results: dict[str, dict] = dict(done)

for s in fast_samples:
    out = fast_classify(s)   # always returns a result for BOOKS
    results[s.sample_id] = {"sample_id": s.sample_id, **out.model_dump()}

print(f"Fast-classified {len(fast_samples)} BOOKS samples instantly.")

# %% — LLM profiler for FINANCIALS + PAPER
if llm_samples:
    model    = get_profiler_model()   # loads Qwen2.5-7B on cuda:0 (~60s first time)
    profiler = QuestionProfiler(model)
    start    = time.time()
    SAVE_EVERY = 50

    for i, sample in enumerate(llm_samples):
        out = profiler.run(sample)
        results[sample.sample_id] = {"sample_id": sample.sample_id, **out.model_dump()}

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(llm_samples):
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
            elapsed = time.time() - start
            rate    = (i + 1) / elapsed
            eta     = (len(llm_samples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(llm_samples)}]  {elapsed:.0f}s elapsed  ~{eta:.0f}s ETA")
else:
    # Still save if only BOOKS were processed
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

print(f"\n✓ {len(results)} profiles → {OUTPUT_PATH}")

# %% — Distribution check
from collections import Counter

type_dist  = Counter(r["question_type"]     for r in results.values())
strat_dist = Counter(r["reasoning_strategy"] for r in results.values())
diff_dist  = Counter(r["difficulty"]         for r in results.values())

print("\nQuestion type distribution:")
for t, n in type_dist.most_common():
    print(f"  {t:25s} {n:4d}  {'█'*(n*30//len(results))}")

print(f"\nStrategy distribution: {dict(strat_dist)}")
print(f"Difficulty: {dict(diff_dist)}")
