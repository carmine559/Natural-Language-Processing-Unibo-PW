"""
schemas.py — shared data models.

Key changes vs. original:
  - MCQSample has 5 options (A-E), category, provided_difficulty, is_statement_combo
  - All Literal option types extended to include "E"
  - ProfilerOutput adds "statement_combination" question type + "statement_evaluation" strategy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator

OPTS = ("A", "B", "C", "D", "E")
OptionKey = Literal["A", "B", "C", "D", "E"]

# ─────────────────────────────────────────────────────────────
#  Input
# ─────────────────────────────────────────────────────────────

@dataclass
class MCQSample:
    sample_id:           str
    question:            str
    context:             str
    options:             Dict[str, str]       # A B C D E
    gold:                Optional[str] = None
    category:            Optional[str] = None  # BOOKS | FINANCIALS | PAPER
    provided_difficulty: Optional[str] = None  # easy | medium | hard (from dataset)
    is_statement_combo:  bool          = False  # BOOKS-style statement-evaluation questions


# ─────────────────────────────────────────────────────────────
#  Stage 1A — Profiler
# ─────────────────────────────────────────────────────────────

QuestionType = Literal[
    "factual", "numerical", "comparative", "temporal",
    "causal", "definitional", "negation", "inferential",
    "statement_combination",  # BOOKS__ / PAPER__ multi-statement true/false
]
ReasoningStrategy = Literal[
    "direct_lookup", "step_by_step_arithmetic", "comparison_table",
    "timeline_reconstruction", "negation_filter",
    "multi_hop", "causal_chain", "term_definition",
    "statement_evaluation",   # evaluate each statement, match combo to option
]

class ProfilerOutput(BaseModel):
    question_type:             QuestionType
    subtype:                   str
    difficulty:                Literal["low", "medium", "high"]
    negation_present:          bool
    superlative_present:       bool
    key_terms:                 List[str]
    key_numbers:               List[str]
    option_claims:             Dict[str, str]
    reasoning_strategy:        ReasoningStrategy
    context_required:          bool
    retrieval_focus:           str
    estimated_answer_location: str

    @field_validator("option_claims")
    @classmethod
    def five_options(cls, v):
        for k in OPTS:
            if k not in v:
                v[k] = ""
        return v


# ─────────────────────────────────────────────────────────────
#  Stage 1B — Context Miner
# ─────────────────────────────────────────────────────────────

class PassageScore(BaseModel):
    passage_id:      int
    relevance_score: float
    key_extraction:  str

class OptionEvidence(BaseModel):
    support_score:     float
    verdict:           Literal["strongly_supported","supported","neutral",
                               "contradicted","strongly_contradicted"]
    evidence_snippet:  str
    calculation_trace: Optional[str] = None
    confidence:        Literal["high", "medium", "low"]

class MinerOutput(BaseModel):
    relevant_passages:           List[PassageScore]
    option_scores:               Dict[str, OptionEvidence]
    evidence_coverage:           Literal["complete", "partial", "absent"]
    tables_found:                bool
    numerical_values_extracted:  List[str]
    missing_information:         Optional[str] = None

    @field_validator("option_scores")
    @classmethod
    def five_options(cls, v):
        for k in OPTS:
            if k not in v:
                v[k] = OptionEvidence(
                    support_score=0.2, verdict="neutral",
                    evidence_snippet="", confidence="low",
                )
        return v


# ─────────────────────────────────────────────────────────────
#  Stage 2 — Specialist Solver
# ─────────────────────────────────────────────────────────────

@dataclass
class SolverOutput:
    proposed_answer:  str
    confidence:       float
    reasoning_steps:  List[str]
    miner_alignment:  str
    consistency_score: float = 1.0
    sc_votes: Dict[str, int] = field(
        default_factory=lambda: {"A":0,"B":0,"C":0,"D":0,"E":0}
    )


# ─────────────────────────────────────────────────────────────
#  Stage 3 — Adversarial Critic
# ─────────────────────────────────────────────────────────────

class DistractorAnalysis(BaseModel):
    best_argument_for_option: str
    argument_strength:        Literal["weak", "moderate", "strong"]
    rebuttal:                 str
    survives_rebuttal:        bool

class Phase1Challenge(BaseModel):
    strongest_counterargument: str
    counterargument_type:      Literal[
        "arithmetic_error","qualifier_missed","wrong_period","wrong_base",
        "negation_inversion","knowledge_override","table_misread",
        "statement_misread","wrong_combination","other",
    ]
    counterargument_strength:  Literal["weak", "moderate", "strong"]
    rebuttal:                  str
    proposed_answer_survives:  bool

class CriticOutput(BaseModel):
    phase_1_challenge:    Phase1Challenge
    phase_2_distractors:  Dict[str, DistractorAnalysis]
    overall_verdict:      Literal["confirm", "weaken", "flip"]
    confidence_adjustment: float
    recommended_answer:   OptionKey
    flip_reason:          Optional[str] = None

    @field_validator("confidence_adjustment")
    @classmethod
    def clamp(cls, v): return max(-0.40, min(0.10, v))


# ─────────────────────────────────────────────────────────────
#  Stage 4 — Aggregator
# ─────────────────────────────────────────────────────────────

@dataclass
class AggregatorOutput:
    final_answer:    str
    final_score:     float
    runner_up:       str
    margin:          float
    confidence_tier: str
    option_scores:   Dict[str, float]
    decision_reason: str


# ─────────────────────────────────────────────────────────────
#  Full pipeline result
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    sample_id:    str
    final_answer: str
    profiler:     Optional[ProfilerOutput]   = None
    miner:        Optional[MinerOutput]      = None
    solver:       Optional[SolverOutput]     = None
    critic:       Optional[CriticOutput]     = None
    aggregator:   Optional[AggregatorOutput] = None
    gold:         Optional[str]              = None
    error:        Optional[str]              = None

    @property
    def is_correct(self) -> Optional[bool]:
        return None if self.gold is None else self.final_answer == self.gold

    def to_dict(self) -> dict:
        return {
            "sample_id":       self.sample_id,
            "prediction":      self.final_answer,
            "gold":            self.gold,
            "correct":         self.is_correct,
            "confidence_tier": self.aggregator.confidence_tier if self.aggregator else None,
            "question_type":   self.profiler.question_type if self.profiler else None,
            "category":        self.profiler.subtype if self.profiler else None,
            "critic_verdict":  self.critic.overall_verdict if self.critic else None,
            "error":           self.error,
        }
