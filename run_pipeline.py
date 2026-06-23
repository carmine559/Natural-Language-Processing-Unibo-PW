#!/usr/bin/env python3
"""
run_pipeline.py — single launchable entry point for the Prometeia pipeline.

Replaces the Kaggle notebooks 01–06: loads one dataset TSV, runs the full
multi-agent pipeline on the single cluster GPU, writes predictions, and prints
an accuracy breakdown when gold labels are present.

Examples
--------
  # quick 5-sample smoke test on the labelled validation set (L40 preset)
  python run_pipeline.py --partition l40 --data data/validation_set_IT.tsv --limit 5

  # full validation run (reports accuracy)
  python run_pipeline.py --partition l40 --data data/validation_set_IT.tsv

  # generate test-set submission (no gold labels)
  python run_pipeline.py --partition l40 --data data/test_set_unlabelled_IT.tsv \
                         --output outputs/test_predictions.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Prometeia pipeline.")
    ap.add_argument("--data", default="data/validation_set_IT.tsv",
                    help="Path to a dataset TSV file.")
    ap.add_argument("--output", default=None,
                    help="Predictions JSON path (default: $PROM_OUTPUT_DIR/predictions.json).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N samples (smoke test).")
    ap.add_argument("--model", default=None,
                    help="Override the HF model name (else picked by --partition).")
    ap.add_argument("--partition", choices=["l40", "rtx2080"], default="l40",
                    help="Cluster partition; selects the default model preset.")
    ap.add_argument("--use-groq", action="store_true",
                    help="Route the solver + tiebreaker through Groq (needs GROQ_API_KEY).")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Ignore any existing results file and start fresh.")
    ap.add_argument("--save-every", type=int, default=20,
                    help="Checkpoint frequency (samples).")
    ap.set_defaults(resume=True)
    return ap.parse_args()


def configure(args) -> None:
    """Apply CLI args to the global CONFIG before the model is loaded."""
    from prometeia_pipeline.config import CONFIG, apply_partition_preset

    apply_partition_preset(args.partition)        # respects PROM_MODEL env
    if args.model:
        CONFIG.model.local_model_name = args.model
    CONFIG.model.use_groq = args.use_groq


def print_breakdown(results) -> None:
    labelled = [r for r in results if r.gold is not None]
    if not labelled:
        print("\nNo gold labels (test set) — predictions saved.")
        return

    import pandas as pd

    correct = sum(bool(r.is_correct) for r in labelled)
    print(f"\nOverall accuracy: {correct/len(labelled):.4f}  "
          f"({correct}/{len(labelled)})")

    rows = [{
        "category":   r.profiler.question_type if r.profiler else "?",
        "difficulty": r.profiler.difficulty    if r.profiler else "?",
        "verdict":    r.critic.overall_verdict if r.critic   else "?",
        "tier":       r.aggregator.confidence_tier if r.aggregator else "?",
        "correct":    bool(r.is_correct),
    } for r in labelled]
    df = pd.DataFrame(rows)

    for col, title in [("category", "question type"), ("difficulty", "difficulty"),
                       ("verdict", "critic verdict"), ("tier", "confidence tier")]:
        print(f"\n── By {title} ──")
        print(df.groupby(col)["correct"]
                .agg(["sum", "count", "mean"])
                .rename(columns={"sum": "correct", "count": "total", "mean": "accuracy"})
                .to_string())


def main() -> None:
    args = parse_args()
    configure(args)

    from prometeia_pipeline.config import CONFIG
    from prometeia_pipeline.orchestrator import PrometeiaPipeline
    from prometeia_pipeline.utils.data_loader import load_dataset, save_predictions

    out_dir = pathlib.Path(CONFIG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path    = args.output or str(out_dir / "predictions.json")
    results_path = str(out_dir / "pipeline_results.json")

    print(f"Model      : {CONFIG.model.local_model_name} (4bit={CONFIG.model.use_4bit})")
    print(f"Device     : {CONFIG.model.gpu_device}  |  use_groq={CONFIG.model.use_groq}")
    print(f"Data       : {args.data}")
    print(f"Output     : {pred_path}")

    samples = load_dataset(args.data)
    if args.limit:
        samples = samples[:args.limit]
        print(f"SMOKE TEST — first {args.limit} samples only.")

    t0 = time.time()
    pipeline = PrometeiaPipeline()
    results = pipeline.run_batch(
        samples,
        output_path=results_path,
        resume=args.resume,
        save_every=args.save_every,
        verbose=True,
    )

    # Build the submission file from the complete merged results on disk
    # (run_batch returns only this run's slice; on a resume the rest live in
    # results_path). Falls back to the in-memory slice if the file is missing.
    if pathlib.Path(results_path).exists():
        with open(results_path, encoding="utf-8") as f:
            merged = json.load(f)
        preds = [{"id": r["sample_id"], "answer": r["prediction"]} for r in merged]
        pathlib.Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(preds)} predictions → {pred_path}")
    else:
        save_predictions(results, pred_path)

    print_breakdown(results)
    pipeline.chat_client.print_stats()
    print(f"\nDone in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
