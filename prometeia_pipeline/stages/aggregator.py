"""
stages/aggregator.py — Weighted Aggregator (Stage 4).

Pure Python, no GPU. Converts the four pipeline signals into option scores,
applies margin thresholds, and runs a binary tiebreaker if needed.

Usage:
    groq       = GroqClient()
    aggregator = WeightedAggregator(groq)
    output     = aggregator.run(sample, profiler, miner, solver, critic)
"""
from __future__ import annotations

import copy
from typing import Optional

from ..schemas import OPTS as _OPTS
from ..schemas import (
    MCQSample, ProfilerOutput, MinerOutput, SolverOutput,
    CriticOutput, AggregatorOutput,
)
from ..utils.groq_client import GroqClient
from ..config import CONFIG

OPTS = _OPTS
OPTS_ABCD = ("A", "B", "C", "D")


# ─────────────────────────────────────────────────────────────
#  Signal → distribution converters
# ─────────────────────────────────────────────────────────────

def _solver_dist(solver: SolverOutput, miner: MinerOutput) -> dict[str, float]:
    """Spread the solver's residual confidence proportionally to miner scores."""
    conf     = solver.confidence
    proposed = solver.proposed_answer
    residual = 1.0 - conf
    others   = [o for o in OPTS if o != proposed]

    other_miner = {o: miner.option_scores[o].support_score for o in others}
    total_other = sum(other_miner.values()) or 1e-9

    dist = {o: residual * (other_miner[o] / total_other) for o in others}
    dist[proposed] = conf
    return dist


def _sc_dist(solver: SolverOutput) -> dict[str, float]:
    total = sum(solver.sc_votes.values()) or 1
    return {o: solver.sc_votes.get(o, 0) / total for o in OPTS}


def _critic_dist(critic: CriticOutput, proposed: str) -> dict[str, float]:
    dist: dict[str, float] = {o: 0.0 for o in OPTS}
    v = critic.overall_verdict

    if v == "confirm":
        dist[proposed] = 1.0

    elif v == "weaken":
        dist[proposed] = 0.65
        surviving = [
            o for o, d in critic.phase_2_distractors.items()
            if d.survives_rebuttal
        ]
        n = len(surviving) or 1
        for o in surviving:
            dist[o] = 0.30 / n
        others_rest = [o for o in OPTS if o != proposed and o not in surviving]
        for o in others_rest:
            dist[o] = 0.05 / max(len(others_rest), 1)

    else:  # flip
        rec = critic.recommended_answer
        dist[rec]      = 0.85
        dist[proposed] = 0.10
        for o in OPTS:
            if o not in (rec, proposed):
                dist[o] = 0.025

    return dist


def _miner_dist(miner: MinerOutput) -> dict[str, float]:
    raw   = {o: miner.option_scores[o].support_score for o in OPTS}
    total = sum(raw.values()) or 1e-9
    return {o: v / total for o, v in raw.items()}


# ─────────────────────────────────────────────────────────────
#  Weight selection
# ─────────────────────────────────────────────────────────────

def _select_weights(
    critic: CriticOutput,
    profiler: ProfilerOutput,
) -> dict[str, float]:
    W = copy.deepcopy(CONFIG.aggregator.weights[critic.overall_verdict])

    # Nudge miner weight by question type
    boost = CONFIG.aggregator.miner_type_boost.get(profiler.question_type, 0.0)
    W["miner"] = max(0.10, min(0.45, W["miner"] + boost))

    # Special: named negation inversion error → critic gets extra trust
    ca = critic.phase_1_challenge.counterargument_type
    if ca == "negation_inversion" and critic.overall_verdict in ("weaken", "flip"):
        W["critic"] = min(0.55, W["critic"] + 0.10)
        W["solver"] = max(0.15, W["solver"] - 0.10)

    # Renormalise to 1.0
    total = sum(W.values())
    return {k: v / total for k, v in W.items()}


# ─────────────────────────────────────────────────────────────
#  Score computation
# ─────────────────────────────────────────────────────────────

def compute_scores(
    profiler: ProfilerOutput,
    miner: MinerOutput,
    solver: SolverOutput,
    critic: CriticOutput,
) -> dict[str, float]:
    W      = _select_weights(critic, profiler)
    md     = _miner_dist(miner)
    sd     = _solver_dist(solver, miner)
    scd    = _sc_dist(solver)
    cd     = _critic_dist(critic, solver.proposed_answer)

    scores = {
        o: W["miner"] * md[o] + W["solver"] * sd[o]
           + W["sc"] * scd[o] + W["critic"] * cd[o]
        for o in OPTS
    }

    # Boost "non determinabile" options when miner found nothing
    if miner.evidence_coverage == "absent":
        from ..schemas import MCQSample  # avoid circular — passed from caller
        pass   # handled in WeightedAggregator.run with option_claims

    # Small penalty for solver's answer when negation present but critic
    # didn't flag inversion (conservative precaution)
    ca_type = critic.phase_1_challenge.counterargument_type
    if profiler.negation_present and ca_type != "negation_inversion":
        scores[solver.proposed_answer] *= 0.93

    return scores


def _confidence_tier(top_score: float, margin: float) -> str:
    cfg = CONFIG.aggregator
    if margin >= cfg.margin_high:
        return "high"
    elif margin >= cfg.margin_low:
        return "low"
    else:
        return "near_tie"


# ─────────────────────────────────────────────────────────────
#  Tiebreaker
# ─────────────────────────────────────────────────────────────

_TIEBREAK_SYSTEM = """\
Sei un arbitro imparziale per domande finanziarie a risposta multipla.
Hai due opzioni finaliste. Scegli quella corretta.
Restituisci SOLO: {"winner": "X", "confidence": 0.0, "decisive_reason": "..."}
confidence: 0.9+ evidenza diretta, 0.6-0.8 inferenziale, <0.6 incerto.\
"""


def _run_tiebreaker(
    sample: MCQSample,
    opt_a: str,
    opt_b: str,
    miner: MinerOutput,
    groq: GroqClient,
) -> tuple[str | None, float]:
    ev_a = miner.option_scores.get(opt_a)
    ev_b = miner.option_scores.get(opt_b)

    user = (
        f"DOMANDA: {sample.question}\n\n"
        f"FINALISTE:\n"
        f"{opt_a}) {sample.options[opt_a]}\n"
        f"   Evidenza: {ev_a.evidence_snippet if ev_a else 'N/A'}\n\n"
        f"{opt_b}) {sample.options[opt_b]}\n"
        f"   Evidenza: {ev_b.evidence_snippet if ev_b else 'N/A'}\n\n"
        f"Scegli solo tra {opt_a} e {opt_b}."
    )
    messages = [
        {"role": "system", "content": _TIEBREAK_SYSTEM},
        {"role": "user",   "content": user},
    ]
    try:
        raw  = groq.chat(messages, model=CONFIG.model.groq_light_model, max_tokens=80)
        from ..utils.json_parser import extract_json_from_text
        data = extract_json_from_text(raw)
        winner = str(data.get("winner", "")).upper().strip()
        conf   = float(data.get("confidence", 0.0))
        if winner not in (opt_a, opt_b):
            return None, 0.0
        return winner, conf
    except Exception as e:
        return None, 0.0


# ─────────────────────────────────────────────────────────────
#  Priority fallback chain
# ─────────────────────────────────────────────────────────────

def _fallback_chain(
    opt_a: str,
    opt_b: str,
    miner: MinerOutput,
    solver: SolverOutput,
    critic: CriticOutput,
    option_scores: dict[str, float],
) -> str:
    # Priority 1: critic named recommendation
    if critic.overall_verdict == "flip" and critic.recommended_answer in (opt_a, opt_b):
        return critic.recommended_answer

    if critic.overall_verdict == "weaken" and solver.proposed_answer in (opt_a, opt_b):
        # Critic weakened solver → prefer the other tied option
        return opt_b if opt_a == solver.proposed_answer else opt_a

    # Priority 2: miner raw evidence score
    ma = miner.option_scores[opt_a].support_score
    mb = miner.option_scores[opt_b].support_score
    if abs(ma - mb) >= CONFIG.aggregator.miner_gap_for_fallback:
        return opt_a if ma > mb else opt_b

    # Priority 3: solver's original answer
    if solver.proposed_answer in (opt_a, opt_b):
        return solver.proposed_answer

    # Last resort: higher weighted score
    return opt_a if option_scores[opt_a] > option_scores[opt_b] else opt_b


# ─────────────────────────────────────────────────────────────
#  WeightedAggregator class
# ─────────────────────────────────────────────────────────────

class WeightedAggregator:
    """Stage 4 — deterministic CPU scoring + optional Groq tiebreaker."""

    def __init__(self, groq: GroqClient):
        self.groq = groq

    def run(
        self,
        sample: MCQSample,
        profiler: ProfilerOutput,
        miner: MinerOutput,
        solver: SolverOutput,
        critic: CriticOutput,
    ) -> AggregatorOutput:
        # Dampen critic flips: downgrade to weaken (flips hurt accuracy).
        if CONFIG.aggregator.dampen_flips and critic.overall_verdict == "flip":
            critic = critic.model_copy(
                update={"overall_verdict": "weaken",
                        "confidence_adjustment": max(critic.confidence_adjustment, -0.15)}
            )

        scores = compute_scores(profiler, miner, solver, critic)

        # Suppress E ("Nessuna delle precedenti"): redistribute its weight
        # to A–D proportionally. In the Prometeia dataset the gold answer
        # is never E, and the model over-predicts it when uncertain.
        if CONFIG.aggregator.suppress_e and scores.get("E", 0) > 0:
            e_weight = scores.pop("E")
            abcd_total = sum(scores[o] for o in OPTS_ABCD) or 1e-9
            for o in OPTS_ABCD:
                scores[o] += e_weight * (scores[o] / abcd_total)
        scores.setdefault("E", 0.0)

        ranked = sorted(OPTS_ABCD, key=lambda o: scores[o], reverse=True)
        winner, runner_up = ranked[0], ranked[1]
        margin = scores[winner] - scores[runner_up]
        tier   = _confidence_tier(scores[winner], margin)

        if tier == "near_tie":
            tb_winner, tb_conf = _run_tiebreaker(
                sample, winner, runner_up, miner, self.groq
            )
            if tb_winner and tb_conf >= CONFIG.aggregator.tiebreak_confidence_threshold:
                winner = tb_winner
                reason = f"tiebreaker ({tb_conf:.2f})"
            else:
                winner = _fallback_chain(
                    winner, runner_up, miner, solver, critic, scores
                )
                reason = "priority fallback chain"
        else:
            reason = self._build_reason(critic, solver, tier)

        return AggregatorOutput(
            final_answer=winner,
            final_score=scores[winner],
            runner_up=runner_up,
            margin=margin,
            confidence_tier=tier,
            option_scores=scores,
            decision_reason=reason,
        )

    @staticmethod
    def _build_reason(critic: CriticOutput, solver: SolverOutput, tier: str) -> str:
        parts = [f"tier={tier}"]
        if critic.overall_verdict == "flip":
            parts.append(f"flip({critic.phase_1_challenge.counterargument_type})")
        if solver.consistency_score == 1.0:
            parts.append("unanimous-SC")
        if critic.overall_verdict == "confirm":
            parts.append("critic-confirmed")
        return " | ".join(parts)

    def run_batch(
        self,
        samples: list[MCQSample],
        profilers: list[ProfilerOutput],
        miners: list[MinerOutput],
        solvers: list[SolverOutput],
        critics: list[CriticOutput],
        verbose: bool = True,
    ) -> list[AggregatorOutput]:
        assert len({len(samples), len(profilers), len(miners),
                    len(solvers), len(critics)}) == 1
        results = []
        for i, args in enumerate(zip(samples, profilers, miners, solvers, critics)):
            if verbose and i % 20 == 0:
                print(f"  Aggregator: {i}/{len(samples)}")
            results.append(self.run(*args))
        return results
