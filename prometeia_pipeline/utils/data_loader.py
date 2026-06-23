"""
utils/data_loader.py

Load the Prometeia / EVALITA Italian dataset from a single TSV file.

Column layouts seen in the wild:
  validation_set_IT.tsv : custom_id, category, question, choiceA..E,
                          correct_answer, difficulty_level, language   (labelled)
  test_set_unlabelled_IT.tsv : same but NO correct_answer, NO difficulty_level
  sample_data_ITA.tsv : different column order, no category column

So every column except custom_id / question / choiceA..E is treated as optional.
`category` is read from the column when present, else derived from the custom_id
prefix (BOOKS__ / FINANCIALS__ / PAPER__). The document context is embedded inside
the `question` field, so MCQSample.context stays empty (the miner chunks the
question text directly).
"""
import json
import pathlib
from typing import List, Optional

import pandas as pd

from ..schemas import MCQSample, OPTS


# ── public API ────────────────────────────────────────────────

def load_dataset(path: str) -> List[MCQSample]:
    """Load one TSV file and return a list of MCQSample."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    samples = [_row_to_sample(row) for _, row in df.iterrows()]

    print(f"Loaded {len(samples)} samples from {p.name}")
    cats = {}
    for s in samples:
        cats[s.category] = cats.get(s.category, 0) + 1
    print("  category:", cats)
    labelled = sum(1 for s in samples if s.gold)
    print(f"  labelled: {labelled} / {len(samples)}")
    return samples


def load_data_folder(folder: str = "data") -> List[MCQSample]:
    """Backwards-compatible helper: load every .tsv in a folder.

    Prefer load_dataset(path) for a single file — globbing a folder will mix
    the validation and test sets together.
    """
    out: List[MCQSample] = []
    for f in sorted(pathlib.Path(folder).glob("*.tsv")):
        out.extend(load_dataset(str(f)))
    return out


def save_predictions(results, path: str) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds = [{"id": r.sample_id, "answer": r.final_answer} for r in results]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(preds)} predictions → {out}")


# ── internals ────────────────────────────────────────────────

def _row_to_sample(row) -> MCQSample:
    custom_id = str(row["custom_id"]).strip()
    question  = str(row["question"])

    gold_raw = str(row.get("correct_answer", "")).strip().upper()
    gold     = gold_raw if gold_raw in OPTS else None

    category = str(row.get("category", "")).strip() or _category_from_id(custom_id)

    difficulty = str(row.get("difficulty_level", "")).strip() or None

    return MCQSample(
        sample_id           = custom_id,
        question            = question,
        context             = "",   # context is embedded in `question`
        options             = {k: str(row.get(f"choice{k}", "")) for k in OPTS},
        gold                = gold,
        category            = category,
        provided_difficulty = difficulty,
        is_statement_combo  = _is_statement_combo(question),
    )


def _category_from_id(custom_id: str) -> Optional[str]:
    """Derive BOOKS / FINANCIALS / PAPER from the custom_id prefix."""
    prefix = custom_id.split("__", 1)[0].upper()
    return prefix if prefix in ("BOOKS", "FINANCIALS", "PAPER") else None


def _is_statement_combo(q: str) -> bool:
    q = q.lower()
    return "affermazioni" in q and ("vere" in q or "false" in q)
