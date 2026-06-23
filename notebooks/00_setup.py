# %% [markdown]
# # Notebook 00 — Setup & Sanity Checks
#
# Run once per Kaggle session. Installs packages, verifies GPU and API access,
# and previews the dataset structure.
#
# Dataset columns:
#   custom_id | category | question | choiceA | choiceB | choiceC | choiceD | choiceE
#   | correct_answer | difficulty_level | language

# %% — Installation
import subprocess, sys

PACKAGES = [
    "groq",
    "rank-bm25",
    "bitsandbytes>=0.41.0",
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    "pydantic>=2.0",
]
for pkg in PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Done. Restart kernel if this is the first run.")

# %% — Path setup
import os, sys, pathlib
sys.path.insert(0, "/kaggle/working")   # where prometeia_pipeline/ lives

# Groq API key from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")
print("Groq key loaded.")

# %% — GPU check
import torch
assert torch.cuda.is_available(), "Enable GPU in notebook settings."
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/1e9:.1f} GB")

# %% — Load data and inspect
import pandas as pd
from prometeia_pipeline.utils.data_loader import load_data_folder

DATA_FOLDER = "/kaggle/input/prometeia-evalita-2026"   # ← adjust to your dataset name

samples = load_data_folder(DATA_FOLDER)

# Quick look at a raw DataFrame for sanity
df = pd.read_csv(next(pathlib.Path(DATA_FOLDER).glob("*.tsv")), sep="\t")
print("\nColumns:", df.columns.tolist())
print(f"\nRows: {len(df)}")

print("\n── Category ──")
print(df["category"].value_counts().to_string())

print("\n── Difficulty ──")
print(df["difficulty_level"].value_counts().to_string())

print("\n── Correct answer distribution ──")
print(df["correct_answer"].value_counts().sort_index().to_string())

print("\n── Language ──")
print(df["language"].value_counts().to_string())

# %% — Preview first sample of each category
for cat in ["BOOKS", "FINANCIALS", "PAPER"]:
    s = next((x for x in samples if x.category == cat), None)
    if s:
        print(f"\n[{cat}] {s.sample_id}  diff={s.difficulty}  gold={s.gold}")
        print(f"  Q: {s.question[:100]}")
        print(f"  A: {s.options['A'][:60]}")
        print(f"  E: {s.options['E']}")
        print(f"  combo: {s.is_statement_combo}")

# %% — Groq smoke test
from prometeia_pipeline.utils.groq_client import GroqClient
from prometeia_pipeline.config import CONFIG

groq = GroqClient()
resp = groq.chat(
    messages=[{"role": "user", "content": "Rispondi solo: pronto"}],
    model=CONFIG.model.groq_light_model,
    max_tokens=5,
)
print(f"\nGroq: '{resp}'")

# %% — (Optional) Local model smoke test — loads ~4.5 GB, takes ~60s
# from prometeia_pipeline.utils.model_loader import get_profiler_model
# m = get_profiler_model()
# print(m.generate([{"role":"user","content":"ok"}], max_new_tokens=5))
