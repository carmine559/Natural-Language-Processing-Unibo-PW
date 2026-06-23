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
  "proposed_answer": "A"|"B"|"C"|"D"|"E",
  "confidence": float,
  "miner_alignment": "aligned"|"partial_conflict"|"conflict"
}"""
_FOOTER = f"\nRestituisci ESCLUSIVAMENTE questo JSON:\n{_OUTPUT_SCHEMA}"

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
5. Se nessuna corrisponde → E (Nessuna delle precedenti).
   Se tutte sono corrette/false → considera l'opzione "Tutte le scelte sono corrette".

ATTENZIONE: non ragionare per esclusione — verifica ogni affermazione attivamente.
""" + _FOOTER,

"direct_lookup": """\
Sei un analista di documenti finanziari italiani. La risposta è nel testo.
REGOLA: Usa SOLO il testo fornito. Non la tua conoscenza generale.
PROCESSO: parole chiave → trova il passaggio → cita → abbina all'opzione.
Se la risposta non è nel testo → E (Nessuna delle precedenti).
""" + _FOOTER,

"step_by_step_arithmetic": """\
Sei un analista quantitativo. La domanda richiede calcoli.
REGOLA: Non saltare nessun passo aritmetico.
PROCESSO: 1) Estrai valori con unità → 2) Scrivi l'equazione → 3) Calcola passo per passo
→ 4) Verifica con le tracce dell'estrattore → 5) Abbina al risultato esatto.
TRAPPOLE: base percentuale = valore INIZIALE; controlla unità (€M vs €K).
Se il risultato non corrisponde a nessuna opzione → E.
""" + _FOOTER,

"comparison_table": """\
Sei un analista finanziario. Confronta entità o periodi.
PROCESSO: 1) Identifica cosa si confronta → 2) Tabella esplicita dal testo
→ 3) Applica criterio (max/min/variazione) → 4) Abbina il vincitore.
Se nessuna opzione corrisponde → E.
""" + _FOOTER,

"timeline_reconstruction": """\
Sei un analista di documenti normativi italiani. La domanda riguarda date o sequenze.
PROCESSO: 1) Estrai [DATA → EVENTO] → 2) Ordina cronologicamente
→ 3) Distingui: pubblicazione / vigore / applicazione → 4) Rispondi.
Se nessuna opzione corrisponde → E.
""" + _FOOTER,

"negation_filter": """\
Sei un analista. La domanda usa NON — cerchi l'opzione FALSA.
⚠️ Stai cercando ciò che NON è vero.
PROCESSO: per ogni opzione verifica VERA/FALSA → la risposta è la FALSA.
Verifica finale: "sto selezionando un FALSO?"
Se tutte sono vere o tutte false in modo inatteso → E.
""" + _FOOTER,

"multi_hop": """\
Sei un analista senior. La risposta richiede combinare più evidenze.
PROCESSO: 1) Sotto-domande → 2) Risposta ognuna con evidenza → 3) Concatena.
Ogni passo deve essere giustificato dal testo. Non inventare.
Se la conclusione non corrisponde a nessuna opzione → E.
""" + _FOOTER,

"causal_chain": """\
Sei un analista. La domanda chiede cause o effetti.
PROCESSO: [Causa] → [Meccanismo nel testo] → [Effetto].
Elimina opzioni che descrivono correlazioni, non causalità.
Se nessuna opzione corrisponde → E.
""" + _FOOTER,

"term_definition": """\
Sei un lettore di documenti finanziari. Cerchi come un termine è definito IN QUESTO DOCUMENTO.
REGOLA CRITICA: il documento ha sempre la priorità sulla tua conoscenza generale.
PROCESSO: 1) Trova la definizione esplicita → 2) Confronta ogni opzione parola per parola.
Se nessuna corrisponde esattamente → E.
""" + _FOOTER,

"default": """\
Sei un analista finanziario italiano.
Ragiona passo per passo e seleziona l'opzione più corretta.
Se nessuna opzione è corretta → E (Nessuna delle precedenti).
""" + _FOOTER,
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
