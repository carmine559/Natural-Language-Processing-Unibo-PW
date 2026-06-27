"""
orchestrator.py — end-to-end pipeline coordinator.

Stage execution order (all on a single GPU per cluster job):
  Stage 1A: Profiler              (local model)
  Stage 1B: Context Miner         (local model)
  Stage 2 : Specialist Solver     (local model, or Groq if --use-groq)
  Stage 3 : Adversarial Critic    (local model)
  Stage 4 : Weighted Aggregator   (CPU; tiebreaker via local model or Groq)

For large batches, intermediate results are saved to disk after each stage
so you can resume or inspect partial results.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

from .schemas import MCQSample, PipelineResult, ProfilerOutput, MinerOutput, SolverOutput, CriticOutput
from .stages.profiler   import QuestionProfiler, _fallback_profile
from .stages.miner      import ContextMiner, _fallback_miner
from .stages.statement_evaluator import StatementEvaluator
from .stages.solver     import SpecialistSolver
from .stages.critic     import AdversarialCritic, _fallback_critic
from .stages.aggregator import WeightedAggregator
from .utils.model_loader import get_light_model, get_heavy_model
from .utils.local_chat   import LocalChatClient
from .config import CONFIG


class PrometeiaPipeline:
    """
    Full pipeline: load once, call run_batch() for inference.

    On the L40 two models are loaded:
      light (14B) → profiler + miner (classification / evidence)
      heavy (32B) → solver + critic  (deeper reasoning)
    On the RTX 2080 Ti both point to the same 14B instance.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        print(f"Loading models on {CONFIG.model.gpu_device} ...")
        print(f"  Light (profiler/miner): {CONFIG.model.light_model_name}")
        print(f"  Heavy (solver/critic) : {CONFIG.model.heavy_model_name}")
        t0 = time.time()

        light_model = get_light_model()
        heavy_model = get_heavy_model()

        # Solver + aggregator client: Groq (optional) or the heavy local model.
        if CONFIG.model.use_groq:
            import os
            from .utils.groq_client import GroqClient
            key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
            self.chat_client = GroqClient(api_key=key)
            print("Solver/tiebreaker: Groq cloud API")
        else:
            self.chat_client = LocalChatClient(heavy_model)
            print("Solver/tiebreaker: heavy local model")

        self.profiler    = QuestionProfiler(light_model)
        self.miner       = ContextMiner(light_model)
        self.stmt_eval   = StatementEvaluator(heavy_model)
        self.solver      = SpecialistSolver(self.chat_client)
        self.critic      = AdversarialCritic(heavy_model)
        self.aggregator  = WeightedAggregator(self.chat_client)

        print(f"All models ready in {time.time()-t0:.0f}s.")

    # ── Single sample ─────────────────────────────────────────

    def run_one(self, sample: MCQSample, verbose: bool = False) -> PipelineResult:
        try:
            # Single GPU → stages run sequentially.
            prof_out = self.profiler.run(sample)
            if prof_out is None:
                prof_out = _fallback_profile(sample)

            # For statement_combination questions, the statement evaluator
            # pre-evaluates each statement and pre-scores options. Falls back
            # to the regular miner if extraction fails.
            if prof_out.question_type == "statement_combination":
                miner_out = self.stmt_eval.run(sample, prof_out)
                if miner_out is None:
                    miner_out = self.miner.run(sample, prof_out)
            else:
                miner_out = self.miner.run(sample, prof_out)
            solver_out = self.solver.run(sample, prof_out, miner_out)
            critic_out = self.critic.run(sample, prof_out, miner_out, solver_out)
            agg_out    = self.aggregator.run(sample, prof_out, miner_out,
                                             solver_out, critic_out)

            if verbose:
                print(
                    f"  [{sample.sample_id}] "
                    f"type={prof_out.question_type} "
                    f"solver={solver_out.proposed_answer}({solver_out.confidence:.2f}) "
                    f"critic={critic_out.overall_verdict} "
                    f"final={agg_out.final_answer} "
                    f"tier={agg_out.confidence_tier}"
                )

            return PipelineResult(
                sample_id=sample.sample_id,
                final_answer=agg_out.final_answer,
                profiler=prof_out,
                miner=miner_out,
                solver=solver_out,
                critic=critic_out,
                aggregator=agg_out,
                gold=sample.gold,
            )

        except Exception as e:
            print(f"  [Pipeline] ERROR on {sample.sample_id}: {e}")
            return PipelineResult(
                sample_id=sample.sample_id,
                final_answer=sample.options.get("A", "A")[0] if sample.options else "A",
                gold=sample.gold,
                error=str(e),
            )

    # ── Batch with checkpointing ──────────────────────────────

    def run_batch(
        self,
        samples: list[MCQSample],
        output_path: Optional[str] = None,
        resume: bool = True,
        save_every: int = 20,
        verbose: bool = True,
    ) -> list[PipelineResult]:
        output_path = output_path or str(
            pathlib.Path(CONFIG.output_dir) / "pipeline_results.json"
        )
        out_path = pathlib.Path(output_path)

        # Resume: load already-processed samples
        done: dict[str, dict] = {}
        if resume and out_path.exists():
            with open(out_path) as f:
                for r in json.load(f):
                    done[r["sample_id"]] = r
            print(f"Resuming — {len(done)} done, "
                  f"{len(samples)-len(done)} remaining.")

        remaining = [s for s in samples if s.sample_id not in done]
        results: dict[str, PipelineResult] = {}

        start = time.time()
        for i, sample in enumerate(remaining):
            result = self.run_one(sample, verbose=verbose)
            results[sample.sample_id] = result

            if (i + 1) % save_every == 0 or (i + 1) == len(remaining):
                all_results = list(done.values()) + [r.to_dict() for r in results.values()]
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                elapsed  = time.time() - start
                rate     = (i + 1) / elapsed
                eta      = (len(remaining) - i - 1) / rate if rate > 0 else 0
                acc_str  = ""
                gold_res = [r for r in results.values() if r.gold]
                if gold_res:
                    acc = sum(r.is_correct for r in gold_res) / len(gold_res)
                    acc_str = f" | acc={acc:.3f}"
                print(f"  [{i+1}/{len(remaining)}] {elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s ETA{acc_str}")

        return list(results.values())
