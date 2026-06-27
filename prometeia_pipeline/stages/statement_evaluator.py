"""
stages/statement_evaluator.py — Statement Evaluator (Stage 1C).

Runs ONLY for statement_combination questions (BOOKS / PAPER with numbered
statements). Replaces the uniform miner output with directional evidence by:
  1. Extracting numbered statements from the question text
  2. Evaluating each statement as TRUE/FALSE via the heavy model (one call)
  3. Pre-scoring options by matching evaluated truth values against each
     option's statement combination

This separates statement evaluation from combination matching, which was the
main bottleneck when both were done in a single solver call.
"""
from __future__ import annotations

import re
from typing import Optional

from ..schemas import MCQSample, ProfilerOutput, MinerOutput, OptionEvidence, PassageScore, OPTS
from ..utils.model_loader import LocalModel
from ..config import CONFIG


# ─────────────────────────────────────────────────────────────
#  Statement extraction
# ─────────────────────────────────────────────────────────────

def extract_statements(question: str) -> list[dict]:
    """Extract numbered statements from question text.

    Handles both formats:
      - Multi-line: each statement on its own line ("1. ...\n2. ...")
      - Inline: all statements in a single paragraph ("1. ... 2. ... 3. ...")

    Returns [{"number": 1, "text": "..."}, ...] or empty list if no
    numbered statements found.
    """
    # Split on "N. " where N is a digit, keeping the number.
    # This handles both inline ("1. foo 2. bar") and multi-line formats.
    parts = re.split(r'(?:^|\s)(\d+)\.\s+', question)

    statements = []
    # parts[0] is text before first "N.", then alternating (number, text, number, text, ...)
    i = 1
    while i + 1 < len(parts):
        num = int(parts[i])
        text = parts[i + 1].strip()
        # Remove trailing question ("Quali di queste affermazioni...")
        text = re.split(r'\s*(?:Qual[ei]|Indicare|Seleziona)', text, maxsplit=1)[0].strip()
        text = text.rstrip('.')
        if text and num <= 20:
            statements.append({"number": num, "text": text})
        i += 2

    return statements


def _detect_question_polarity(question: str) -> str:
    """Detect whether the question asks for TRUE or FALSE statements."""
    q_lower = question.lower()
    if "false" in q_lower or "falsa" in q_lower or "errat" in q_lower:
        return "FALSE"
    return "TRUE"


# ─────────────────────────────────────────────────────────────
#  Option combination parsing
# ─────────────────────────────────────────────────────────────

def parse_option_combination(option_text: str, n_statements: int) -> Optional[list[int]]:
    """Parse which statement numbers an option references.

    "1, 2, 3" → [1, 2, 3]
    "Tutte le risposte sono corrette." → [1, 2, ..., n]
    "Nessuna delle precedenti" → None (can't resolve)
    """
    text = option_text.strip()

    if re.search(r'[Tt]utt[eiao]', text):
        return list(range(1, n_statements + 1))

    nums = [int(x) for x in re.findall(r'\d+', text)]
    if nums and all(1 <= n <= n_statements for n in nums):
        return sorted(set(nums))

    return None


# ─────────────────────────────────────────────────────────────
#  Evaluator prompt
# ─────────────────────────────────────────────────────────────

_EVALUATOR_SYSTEM = """\
Sei un esperto di economia, finanza e normativa italiana.
Il tuo compito è valutare OGNI affermazione come VERA o FALSA.

ISTRUZIONI:
- Valuta ogni affermazione INDIPENDENTEMENTE dalle altre.
- Usa le tue conoscenze di economia, finanza, management e normativa.
- Per ogni affermazione, fornisci un giudizio chiaro e una breve motivazione.
- NON considerare le opzioni di risposta — valuta solo le affermazioni.

Restituisci ESCLUSIVAMENTE un JSON valido con questo formato:
{
  "evaluations": [
    {"number": 1, "verdict": "TRUE"|"FALSE", "reason": "breve motivazione"},
    {"number": 2, "verdict": "TRUE"|"FALSE", "reason": "breve motivazione"},
    ...
  ]
}"""


def _build_evaluator_message(statements: list[dict]) -> list[dict]:
    stmts_block = "\n".join(
        f"{s['number']}. {s['text']}" for s in statements
    )
    return [
        {"role": "system", "content": _EVALUATOR_SYSTEM},
        {"role": "user", "content": f"AFFERMAZIONI DA VALUTARE:\n{stmts_block}"},
    ]


# ─────────────────────────────────────────────────────────────
#  Score options based on evaluated statements
# ─────────────────────────────────────────────────────────────

def score_options(
    verdicts: dict[int, bool],
    options: dict[str, str],
    polarity: str,
    n_statements: int,
) -> dict[str, OptionEvidence]:
    """Score each option by matching statement verdicts against combinations.

    If polarity == "FALSE", the question asks "which are FALSE?" so we look
    for statements evaluated as FALSE. If "TRUE", we look for TRUE ones.
    """
    target_value = (polarity == "TRUE")
    matching_stmts = sorted(
        num for num, is_true in verdicts.items() if is_true == target_value
    )

    scores: dict[str, OptionEvidence] = {}
    for opt in OPTS:
        opt_text = options.get(opt, "")
        if not opt_text or opt_text.strip().lower().startswith("nessuna"):
            scores[opt] = OptionEvidence(
                support_score=0.05,
                verdict="contradicted",
                evidence_snippet="Opzione E soppressa.",
                confidence="low",
            )
            continue

        combination = parse_option_combination(opt_text, n_statements)
        if combination is None:
            scores[opt] = OptionEvidence(
                support_score=0.15,
                verdict="neutral",
                evidence_snippet=f"Combinazione non riconosciuta: {opt_text[:60]}",
                confidence="low",
            )
            continue

        if set(combination) == set(matching_stmts):
            scores[opt] = OptionEvidence(
                support_score=0.95,
                verdict="strongly_supported",
                evidence_snippet=f"Combinazione corrisponde: affermazioni {polarity} = {matching_stmts}",
                confidence="high",
            )
        elif set(combination).issubset(set(matching_stmts)):
            scores[opt] = OptionEvidence(
                support_score=0.4,
                verdict="supported",
                evidence_snippet=f"Sottoinsieme: {combination} ⊂ {matching_stmts}",
                confidence="medium",
            )
        elif set(combination).intersection(set(matching_stmts)):
            scores[opt] = OptionEvidence(
                support_score=0.25,
                verdict="neutral",
                evidence_snippet=f"Sovrapposizione parziale: {combination} ∩ {matching_stmts}",
                confidence="low",
            )
        else:
            scores[opt] = OptionEvidence(
                support_score=0.05,
                verdict="contradicted",
                evidence_snippet=f"Nessuna sovrapposizione: {combination} vs {matching_stmts}",
                confidence="high",
            )

    return scores


# ─────────────────────────────────────────────────────────────
#  StatementEvaluator class
# ─────────────────────────────────────────────────────────────

class StatementEvaluator:
    """Evaluates numbered statements and pre-scores options for
    statement_combination questions."""

    def __init__(self, model: LocalModel):
        self.model = model

    def run(
        self,
        sample: MCQSample,
        profiler: ProfilerOutput,
    ) -> Optional[MinerOutput]:
        """Returns a directional MinerOutput, or None if extraction fails."""
        statements = extract_statements(sample.question)
        if len(statements) < 2:
            return None

        polarity = _detect_question_polarity(sample.question)
        messages = _build_evaluator_message(statements)

        from ..utils.json_parser import extract_json_from_text
        raw = self.model.generate(messages, max_new_tokens=CONFIG.local_max_new_tokens)

        try:
            data = extract_json_from_text(raw)
            evals = data.get("evaluations", [])
        except (ValueError, KeyError):
            return None

        verdicts: dict[int, bool] = {}
        for ev in evals:
            num = ev.get("number")
            verdict_str = str(ev.get("verdict", "")).upper().strip()
            if num is not None and verdict_str in ("TRUE", "FALSE"):
                verdicts[int(num)] = (verdict_str == "TRUE")

        if len(verdicts) < len(statements) * 0.5:
            return None

        option_scores = score_options(
            verdicts, sample.options, polarity, len(statements)
        )

        passages = [
            PassageScore(
                passage_id=s["number"],
                relevance_score=1.0,
                key_extraction=(
                    f"[{verdicts.get(s['number'], '?')}] {s['text'][:100]}"
                ),
            )
            for s in statements
        ]

        return MinerOutput(
            relevant_passages=passages,
            option_scores=option_scores,
            evidence_coverage="complete",
            tables_found=False,
            numerical_values_extracted=[],
            missing_information=None,
        )
