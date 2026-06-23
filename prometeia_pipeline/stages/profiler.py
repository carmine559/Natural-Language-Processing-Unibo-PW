"""
stages/profiler.py — Question Profiler (Stage 1A).

Key additions vs. original:
  - fast_classify(): directly assigns type for BOOKS__/PAPER__ statement questions
    and uses provided_difficulty from the dataset — skips LLM for obvious cases.
  - statement_combination type + statement_evaluation strategy.
  - Uses category (BOOKS/FINANCIALS/PAPER) as a strong routing prior.
"""
from __future__ import annotations

from typing import Optional

from ..schemas import MCQSample, ProfilerOutput, OPTS
from ..utils.model_loader import LocalModel


# ─────────────────────────────────────────────────────────────
#  Fast-classify (no LLM needed)
# ─────────────────────────────────────────────────────────────

_DIFFICULTY_MAP = {"easy": "low", "medium": "medium", "hard": "high"}


def fast_classify(sample: MCQSample) -> Optional[ProfilerOutput]:
    """
    Return a ProfilerOutput without calling the LLM when the category + question
    make the type unambiguous.  Returns None if LLM inference is still needed.
    """
    diff = _DIFFICULTY_MAP.get(sample.provided_difficulty or "", "medium")

    # ── BOOKS__: always multi-statement combo ───────────────────────────────
    if sample.category == "BOOKS" or sample.is_statement_combo:
        q_low = sample.question.lower()
        is_false_q = "false" in q_low or "falsa" in q_low
        return ProfilerOutput(
            question_type="statement_combination",
            subtype="books_false" if is_false_q else "books_true",
            difficulty=diff,
            negation_present=is_false_q,
            superlative_present=False,
            key_terms=[],
            key_numbers=[],
            option_claims={k: sample.options.get(k, "") for k in OPTS},
            reasoning_strategy="statement_evaluation",
            context_required=True,
            retrieval_focus="valuta ogni affermazione numerata come VERA o FALSA",
            estimated_answer_location="contesto — affermazioni numerate",
        )

    # ── PAPER__: single-row direct questions don't need a LLM nudge,
    #    but multi-statement ones do — handled by is_statement_combo above.
    #    Return None so the LLM profiler runs for single-row PAPER__ questions.
    return None


# ─────────────────────────────────────────────────────────────
#  LLM-based profiler prompts
# ─────────────────────────────────────────────────────────────

PROFILER_SYSTEM = """\
Sei un classificatore di domande per il benchmark finanziario Prometeia (EVALITA).
Analizza ogni domanda a scelta multipla e restituisci ESCLUSIVAMENTE un oggetto JSON valido.
Nessun testo prima o dopo. Nessun markdown.

CATEGORIE (dal prefisso del custom_id):
  BOOKS      → domande su testi di economia/management
  FINANCIALS → domande su tabelle finanziarie (bilanci, conti economici)
  PAPER      → domande su paper accademici

TIPI DI DOMANDA:
  "factual"              → risposta esplicita nel testo
  "numerical"            → calcoli, percentuali, differenze
  "comparative"          → confronto tra entità o periodi
  "temporal"             → date, sequenze, periodi
  "causal"               → cause, effetti, meccanismi
  "definitional"         → significato di un termine nel documento
  "negation"             → NON/non, inversione logica
  "inferential"          → ragionamento multi-step
  "statement_combination"→ valuta affermazioni numerate; le opzioni sono combinazioni

STRATEGIE (tipo → strategia):
  factual              → direct_lookup
  numerical            → step_by_step_arithmetic
  comparative          → comparison_table
  temporal             → timeline_reconstruction
  negation             → negation_filter
  inferential          → multi_hop
  causal               → causal_chain
  definitional         → term_definition
  statement_combination→ statement_evaluation

SCHEMA (tutti i campi obbligatori):
{
  "question_type": string,
  "subtype": string,
  "difficulty": "low"|"medium"|"high",
  "negation_present": boolean,
  "superlative_present": boolean,
  "key_terms": [string],
  "key_numbers": [string],
  "option_claims": {"A":string,"B":string,"C":string,"D":string,"E":string},
  "reasoning_strategy": string,
  "context_required": boolean,
  "retrieval_focus": string,
  "estimated_answer_location": string
}\
"""

_FEW_SHOT: list[dict] = [
    # FINANCIALS — numerical
    {
        "role": "user",
        "content": (
            "CATEGORIA: FINANCIALS | DIFFICOLTÀ FORNITA: hard\n"
            "DOMANDA: Qual è la percentuale di passività correnti rispetto al totale delle attività?\n"
            "OPZIONI: A=24.08% B=25.00% C=23.94% D=22.05% E=Nessuna delle precedenti\n"
            "CONTESTO: tabella stato patrimoniale con totale attivo 2.432.604 e passività correnti 585.889"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"question_type":"numerical","subtype":"percentage_of_total",'
            '"difficulty":"high","negation_present":false,"superlative_present":false,'
            '"key_terms":["passività correnti","totale attività"],'
            '"key_numbers":["585.889","2.432.604"],'
            '"option_claims":{"A":"24.08%","B":"25.00%","C":"23.94%","D":"22.05%","E":"Nessuna delle precedenti"},'
            '"reasoning_strategy":"step_by_step_arithmetic",'
            '"context_required":true,'
            '"retrieval_focus":"cerca totale attivo e totale passività correnti nella tabella",'
            '"estimated_answer_location":"tabella stato patrimoniale"}'
        ),
    },
    # PAPER — direct definitional
    {
        "role": "user",
        "content": (
            "CATEGORIA: PAPER | DIFFICOLTÀ FORNITA: medium\n"
            "DOMANDA: Cosa descrive meglio i costi di agenzia nel contesto dell'impresa?\n"
            "OPZIONI: A=Costi sostenuti dai mandanti... B=Costi associati all'emissione... "
            "C=Costi normativi... D=Costi indipendenti dalla struttura... E=Nessuna delle precedenti\n"
            "CONTESTO: (nessuno)"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"question_type":"definitional","subtype":"agency_cost_definition",'
            '"difficulty":"medium","negation_present":false,"superlative_present":false,'
            '"key_terms":["costi di agenzia","mandanti","agenti"],'
            '"key_numbers":[],'
            '"option_claims":{"A":"costi sostenuti dai mandanti per garantire che gli agenti agiscano nel loro interesse",'
            '"B":"costi associati esclusivamente all\'emissione di titoli",'
            '"C":"costi da conformità normativa",'
            '"D":"costi indipendenti dalla struttura proprietaria",'
            '"E":"Nessuna delle precedenti"},'
            '"reasoning_strategy":"term_definition",'
            '"context_required":false,'
            '"retrieval_focus":"definizione costi di agenzia teoria dell\'impresa",'
            '"estimated_answer_location":"conoscenza dominio economia aziendale"}'
        ),
    },
]


def _build_messages(sample: MCQSample) -> list[dict]:
    opts_str = " ".join(f"{k}={sample.options.get(k,'')[:40]}" for k in OPTS)
    ctx_prev = (sample.question[:500] + "…") if len(sample.question) > 500 else sample.question
    diff_hint = sample.provided_difficulty or "sconosciuta"
    cat_hint  = sample.category or "sconosciuta"

    user = (
        f"CATEGORIA: {cat_hint} | DIFFICOLTÀ FORNITA: {diff_hint}\n"
        f"DOMANDA: {sample.question}\n"
        f"OPZIONI: {opts_str}\n"
        f"CONTESTO: {ctx_prev or '(nessuno)'}"
    )
    return (
        [{"role": "system", "content": PROFILER_SYSTEM}]
        + _FEW_SHOT
        + [{"role": "user", "content": user}]
    )


# ─────────────────────────────────────────────────────────────
#  Profiler class
# ─────────────────────────────────────────────────────────────

class QuestionProfiler:
    def __init__(self, model: LocalModel):
        self.model = model
        self._fast_hits = 0
        self._llm_hits  = 0

    def run(self, sample: MCQSample) -> ProfilerOutput:
        # Try fast-classify first (saves GPU time for BOOKS__ questions)
        fast = fast_classify(sample)
        if fast is not None:
            self._fast_hits += 1
            # Override difficulty with the one from the dataset
            if sample.provided_difficulty:
                fast.difficulty = _DIFFICULTY_MAP.get(sample.provided_difficulty, fast.difficulty)
            return fast

        self._llm_hits += 1
        messages       = _build_messages(sample)
        parsed, error  = self.model.generate_json(messages, ProfilerOutput)

        if parsed is None:
            print(f"  [Profiler] WARN {sample.sample_id}: {error}")
            return _fallback_profile(sample)

        # Trust the dataset's difficulty over the model's inference
        if sample.provided_difficulty:
            parsed.difficulty = _DIFFICULTY_MAP.get(sample.provided_difficulty, parsed.difficulty)

        return parsed

    def run_batch(self, samples: list[MCQSample], verbose: bool = True) -> list[ProfilerOutput]:
        results = []
        for i, s in enumerate(samples):
            if verbose and i % 10 == 0:
                print(f"  Profiler: {i}/{len(samples)} "
                      f"(fast={self._fast_hits}, llm={self._llm_hits})")
            results.append(self.run(s))
        return results


# ─────────────────────────────────────────────────────────────
#  Fallback
# ─────────────────────────────────────────────────────────────

def _fallback_profile(sample: MCQSample) -> ProfilerOutput:
    diff = _DIFFICULTY_MAP.get(sample.provided_difficulty or "", "medium")
    return ProfilerOutput(
        question_type="factual",
        subtype="unknown",
        difficulty=diff,
        negation_present=("non " in sample.question.lower()),
        superlative_present=False,
        key_terms=[],
        key_numbers=[],
        option_claims={k: sample.options.get(k, "") for k in OPTS},
        reasoning_strategy="direct_lookup",
        context_required=True,
        retrieval_focus=sample.question[:100],
        estimated_answer_location="documento principale",
    )
