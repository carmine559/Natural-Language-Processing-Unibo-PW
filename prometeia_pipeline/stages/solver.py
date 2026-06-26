"""
stages/solver.py — Specialist Solver (Stage 2).

Key additions vs. original:
  - statement_evaluation system prompt for BOOKS__/PAPER__ combo questions.
  - All OPTS now include E ("Nessuna delle precedenti").
  - SC vote extraction handles A-E.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ..schemas import MCQSample, ProfilerOutput, MinerOutput, SolverOutput, OPTS
from ..utils.groq_client import GroqClient
from ..config import CONFIG

# ─────────────────────────────────────────────────────────────
#  System prompts
# ─────────────────────────────────────────────────────────────

_OUTPUT_SCHEMA = """\
{
  "reasoning_steps": [string],
  "proposed_answer": "A"|"B"|"C"|"D",
  "confidence": float,
  "miner_alignment": "aligned"|"partial_conflict"|"conflict"
}"""
_FOOTER = f"\nRestituisci ESCLUSIVAMENTE questo JSON:\n{_OUTPUT_SCHEMA}"

# All prompts force a commitment to A/B/C/D — never E.
# The model was over-predicting "Nessuna delle precedenti" when uncertain.
_COMMIT_RULE = (
    "\n\nREGOLA FONDAMENTALE: la risposta corretta è SEMPRE tra A, B, C, D. "
    "NON rispondere MAI E. Scegli l'opzione PIÙ PROBABILE anche se non sei sicuro."
)

SYSTEM_PROMPTS: dict[str, str] = {

"statement_evaluation": """\
Sei un esperto valutatore di affermazioni economico-finanziarie.
La domanda chiede quali affermazioni (numerate) sono VERE o FALSE.
Le opzioni NON sono risposte dirette — sono COMBINAZIONI di numeri di affermazione.

PROCESSO OBBLIGATORIO:
1. Leggi ogni affermazione numerata nel contesto.
2. Valuta ciascuna: VERA o FALSA, con una breve giustificazione.
3. Costruisci la lista corretta (es. "1, 3 sono FALSE").
4. Confronta con ogni opzione: quale combinazione corrisponde esattamente?
5. Se più opzioni sembrano plausibili, scegli quella con la corrispondenza più forte.

ATTENZIONE: non ragionare per esclusione — verifica ogni affermazione attivamente.
Se il testo non copre tutte le affermazioni, usa la tua conoscenza di economia e finanza.
""" + _COMMIT_RULE + _FOOTER,

"direct_lookup": """\
Sei un analista di documenti finanziari italiani. La risposta è nel testo.
REGOLA: Usa PRIMA il testo fornito, poi la tua conoscenza di dominio se il testo è insufficiente.
PROCESSO: parole chiave → trova il passaggio → cita → abbina all'opzione.
Se il testo non contiene la risposta esplicita, scegli l'opzione più coerente con le tue
conoscenze di economia, finanza e normativa italiana.
""" + _COMMIT_RULE + _FOOTER,

"step_by_step_arithmetic": """\
Sei un analista quantitativo. La domanda richiede calcoli.
REGOLA: Non saltare nessun passo aritmetico.
PROCESSO: 1) Estrai valori con unità → 2) Scrivi l'equazione → 3) Calcola passo per passo
→ 4) Verifica con le tracce dell'estrattore → 5) Abbina al risultato più vicino.
TRAPPOLE: base percentuale = valore INIZIALE; controlla unità (€M vs €K).
Se il risultato non è esatto, scegli l'opzione numericamente più vicina.
""" + _COMMIT_RULE + _FOOTER,

"comparison_table": """\
Sei un analista finanziario esperto di confronti tra dati.
PROCESSO:
1) Identifica ESATTAMENTE cosa si confronta (entità, periodi, metriche).
2) Estrai i valori dalle tabelle o dal testo — ATTENZIONE a riga/colonna corretta.
3) Se si chiede "maggiore/minore": confronta i valori ASSOLUTI.
   Se si chiede "variazione/crescita": calcola la differenza % = (nuovo-vecchio)/vecchio.
4) Verifica di non confondere variazione % con valore assoluto.
5) Scegli l'opzione che corrisponde al confronto corretto.

Se il testo non contiene tutti i dati, usa le tue conoscenze per integrare.
""" + _COMMIT_RULE + _FOOTER,

"timeline_reconstruction": """\
Sei un analista di documenti normativi italiani. La domanda riguarda date o sequenze.
PROCESSO: 1) Estrai [DATA → EVENTO] → 2) Ordina cronologicamente
→ 3) Distingui: pubblicazione / vigore / applicazione → 4) Rispondi.
""" + _COMMIT_RULE + _FOOTER,

"negation_filter": """\
Sei un analista. La domanda usa NON — cerchi l'opzione FALSA o ERRATA.
⚠️ ATTENZIONE: stai cercando ciò che NON è vero/corretto.
PROCESSO:
1) Per OGNI opzione, verifica se è VERA o FALSA nel contesto del documento.
2) La risposta corretta è l'opzione che risulta FALSA (NON vera).
3) Prima di rispondere, verifica: "L'opzione che ho scelto è davvero FALSA?"
4) Se più opzioni sono false, scegli quella PIÙ chiaramente falsa.

NON invertire la logica — la domanda chiede cosa NON è corretto.
""" + _COMMIT_RULE + _FOOTER,

"multi_hop": """\
Sei un analista senior. La risposta richiede combinare più evidenze.
PROCESSO: 1) Scomponi in sotto-domande → 2) Rispondi ognuna con evidenza → 3) Concatena.
Ogni passo deve essere giustificato dal testo o dalla tua conoscenza di dominio.
Se un passaggio manca nel testo, usa il tuo sapere di economia/finanza/normativa.
""" + _COMMIT_RULE + _FOOTER,

"causal_chain": """\
Sei un esperto di economia e finanza. La domanda chiede cause, effetti o meccanismi.
PROCESSO:
1) Identifica il FENOMENO di cui si chiede la causa o l'effetto.
2) Per ogni opzione, valuta: descrive una CAUSA DIRETTA, un effetto, o una correlazione?
3) Distingui: causalità (A causa B) vs correlazione (A e B coesistono).
4) Privilegia meccanismi causali riconosciuti nella teoria economico-finanziaria.
5) Se il testo non spiega il meccanismo, usa la tua conoscenza di dominio.

NON limitarti al testo se la domanda riguarda concetti teorici — usa anche le tue conoscenze.
""" + _COMMIT_RULE + _FOOTER,

"term_definition": """\
Sei un esperto di terminologia economico-finanziaria e normativa italiana.
PROCESSO:
1) Cerca prima la definizione nel documento — il documento ha priorità.
2) Se il documento non definisce il termine, usa la tua conoscenza di dominio.
3) Confronta OGNI opzione con la definizione corretta — attenzione a sfumature e qualificatori.
4) L'opzione corretta è quella SEMANTICAMENTE più vicina alla definizione.

NON rispondere "nessuna" — scegli sempre la migliore tra A, B, C, D.
""" + _COMMIT_RULE + _FOOTER,

"default": """\
Sei un esperto analista finanziario italiano con conoscenza approfondita di
economia, finanza, normativa e contabilità.
Ragiona passo per passo e seleziona l'opzione più corretta tra A, B, C, D.
Se il testo fornito non basta, integra con le tue conoscenze di dominio.
""" + _COMMIT_RULE + _FOOTER,
}


# ─────────────────────────────────────────────────────────────
#  User message builder
# ─────────────────────────────────────────────────────────────

def build_solver_user_message(
    sample: MCQSample,
    profiler: ProfilerOutput,
    miner: MinerOutput,
) -> str:
    opts_block = "\n".join(
        f"{k}) {sample.options.get(k, '')}" for k in OPTS
    )

    evidence_lines = []
    for opt in OPTS:
        ev    = miner.option_scores.get(opt)
        claim = profiler.option_claims.get(opt, sample.options.get(opt, ""))
        if ev:
            line = (
                f"Opzione {opt} — \"{claim[:80]}\"\n"
                f"  Evidenza: {ev.evidence_snippet or 'N/A'}\n"
                f"  Supporto: {ev.support_score:.2f} ({ev.verdict})"
            )
            if ev.calculation_trace:
                line += f"\n  Traccia: {ev.calculation_trace}"
            evidence_lines.append(line)

    coverage_note = {
        "complete": "",
        "partial":  "\n⚠️ Evidenza incompleta per alcune opzioni.",
        "absent":   "\n⚠️ Nessuna evidenza diretta nel documento.",
    }.get(miner.evidence_coverage, "")

    # For statement-combo questions, include the full context (the statements)
    ctx_block = ""
    if profiler.question_type == "statement_combination" and sample.question:
        ctx_block = f"\nAFFERMAZIONI DA VALUTARE:\n{sample.question}\n"

    return (
        f"DOMANDA: {sample.question}\n\n"
        f"OPZIONI:\n{opts_block}\n"
        + ctx_block
        + f"\nPRE-ANALISI EVIDENZE:{coverage_note}\n"
        + "\n\n".join(evidence_lines)
    )


# ─────────────────────────────────────────────────────────────
#  Response parser
# ─────────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[str, float, list[str], str]:
    from ..utils.json_parser import extract_json_from_text
    try:
        data    = extract_json_from_text(text)
        answer  = str(data.get("proposed_answer", "A")).upper().strip()
        conf    = float(data.get("confidence", 0.5))
        steps   = data.get("reasoning_steps", [text[:200]])
        align   = str(data.get("miner_alignment", "aligned"))
        if answer not in OPTS:
            answer = "A"
        return answer, max(0.0, min(1.0, conf)), steps, align
    except Exception:
        m      = re.search(r'"proposed_answer"\s*:\s*"([ABCDE])"', text)
        answer = m.group(1) if m else "A"
        return answer, 0.4, [text[:300]], "aligned"


# ─────────────────────────────────────────────────────────────
#  SpecialistSolver
# ─────────────────────────────────────────────────────────────

class SpecialistSolver:
    def __init__(self, groq: GroqClient):
        self.groq = groq

    def run(
        self,
        sample: MCQSample,
        profiler: ProfilerOutput,
        miner: MinerOutput,
        force_sc: bool = False,
    ) -> SolverOutput:
        system   = SYSTEM_PROMPTS.get(profiler.reasoning_strategy, SYSTEM_PROMPTS["default"])
        user     = build_solver_user_message(sample, profiler, miner)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        use_sc = force_sc or profiler.difficulty == "high"
        return self._run_with_sc(messages, profiler) if use_sc else self._run_single(messages)

    def _run_single(self, messages: list[dict]) -> SolverOutput:
        raw                          = self.groq.chat(messages, temperature=0.0)
        answer, conf, steps, align   = _parse_response(raw)
        return SolverOutput(
            proposed_answer=answer, confidence=conf,
            reasoning_steps=steps, miner_alignment=align,
            consistency_score=1.0,
            sc_votes={k: (1 if k == answer else 0) for k in OPTS},
        )

    def _run_with_sc(self, messages: list[dict], profiler: ProfilerOutput) -> SolverOutput:
        n           = CONFIG.model.groq_sc_samples
        responses, votes = self.groq.chat_with_sc(
            messages, n_samples=n,
            temperature=CONFIG.model.groq_temperature_high_difficulty,
        )
        parsed   = [_parse_response(r) for r in responses]
        answers  = [p[0] for p in parsed]
        confs    = [p[1] for p in parsed]
        majority, count = Counter(answers).most_common(1)[0]
        maj_confs = [c for a, c in zip(answers, confs) if a == majority]
        best_idx  = max(
            (i for i, a in enumerate(answers) if a == majority),
            key=lambda i: confs[i],
        )
        return SolverOutput(
            proposed_answer=majority,
            confidence=sum(maj_confs) / len(maj_confs) * (count / n),
            reasoning_steps=parsed[best_idx][2],
            miner_alignment=parsed[best_idx][3],
            consistency_score=count / n,
            sc_votes=dict(votes),
        )

    def run_batch(
        self,
        samples: list[MCQSample],
        profilers: list[ProfilerOutput],
        miners: list[MinerOutput],
        verbose: bool = True,
    ) -> list[SolverOutput]:
        assert len(samples) == len(profilers) == len(miners)
        results = []
        for i, (s, p, m) in enumerate(zip(samples, profilers, miners)):
            if verbose and i % 10 == 0:
                print(f"  Solver: {i}/{len(samples)}")
            results.append(self.run(s, p, m))
        return results
