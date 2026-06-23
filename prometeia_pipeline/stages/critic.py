"""
stages/critic.py — Adversarial Critic (Stage 3).

Runs on the shared local model (single GPU per cluster job).
Two phases in one LLM call:
  Phase 1 — challenge the solver's proposed answer.
  Phase 2 — defend each distractor, then rebut the defense.

Usage:
    model  = get_shared_model()
    critic = AdversarialCritic(model)
    output = critic.run(sample, profiler_output, miner_output, solver_output)
"""
from __future__ import annotations

from typing import Optional

from ..schemas import (
    MCQSample, ProfilerOutput, MinerOutput, SolverOutput,
    CriticOutput, Phase1Challenge, DistractorAnalysis,
)
from ..utils.model_loader import LocalModel

# ─────────────────────────────────────────────────────────────
#  Attack vectors by question type
# ─────────────────────────────────────────────────────────────

ATTACK_VECTORS: dict[str, list[str]] = {
    "factual": [
        "Il solutore ha verificato TUTTI i passaggi, non solo il primo?",
        "Esiste un passaggio che contraddice quello usato?",
        "Ha usato conoscenza generale invece del documento?",
        "Un qualificatore ('esclusivamente', 'almeno', 'fino a') cambia la risposta?",
    ],
    "numerical": [
        "Il denominatore per la percentuale è il valore INIZIALE, non quello finale?",
        "Le unità sono coerenti (€M vs €K, % vs punti percentuali)?",
        "Ha letto la riga/colonna corretta nella tabella?",
        "C'è un errore aritmetico in un passaggio intermedio?",
        "Il periodo temporale è quello corretto?",
        "L'arrotondamento è stato applicato troppo presto?",
    ],
    "comparative": [
        "Il confronto è sulla metrica giusta?",
        "Il superlativo è rispettato (massimo assoluto vs massimo in un sottoinsieme)?",
        "Ha confrontato variazioni % invece di valori assoluti dove richiesto?",
    ],
    "temporal": [
        "Ha distinto tra data di pubblicazione, vigore, e applicazione?",
        "'A partire dal' vs 'entro il' — l'inclusività è corretta?",
        "Il periodo di riferimento del dato coincide con la domanda?",
    ],
    "negation": [
        "⚠️ PRIORITÀ MASSIMA: ha risposto alla domanda NEGATA, non a quella positiva?",
        "L'opzione selezionata è FALSA nel documento?",
        "Le opzioni non scelte sono tutte VERE nel documento?",
    ],
    "definitional": [
        "Ha usato la definizione del documento o quella generale di mercato?",
        "C'è una definizione esplicita ignorata?",
        "Un qualificatore nella definizione è stato rispettato?",
    ],
    "causal": [
        "Ha confuso causa con effetto?",
        "Il meccanismo causale è nel documento o inferito?",
        "Ha scambiato correlazione con causalità?",
    ],
    "statement_combination": [
        "Il solutore ha valutato TUTTE le affermazioni, non solo alcune?",
        "Ha confuso VERE con FALSE (inversione)?",
        "Ha letto correttamente quale combinazione corrisponde all'opzione scelta?",
        "Considera l'opzione E se nessuna combinazione corrisponde.",
    ],
    "inferential": [
        "Ogni passo inferenziale ha evidenza documentale?",
        "Ha usato conoscenza esterna per colmare una lacuna?",
        "La catena logica è valida senza passi impliciti?",
    ],
}


# ─────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """\
Sei un critico avversariale specializzato in domande finanziarie a risposta multipla italiane.
Il tuo compito NON è confermare — è trovare ogni possibile falla.

FASE 1: Costruisci il controargomento più forte contro la risposta proposta.
  1a. Usa il checklist specifico per il tipo di domanda.
  1b. Valuta la forza: "weak"|"moderate"|"strong"
  1c. Replica: perché il controargomento non regge (o perché sì)?

FASE 2: Per ogni distractor (opzione non scelta):
  2a. Costruisci il miglior caso possibile per cui potrebbe essere corretta.
  2b. Poi smontalo con un'evidenza specifica.
  2c. Il distractor "sopravvive" se non riesci a smontarlo convincentemente.

VERDETTO: "confirm"|"weaken"|"flip"
  confirm = controargomento debole E tutti i distractor eliminati
  weaken  = controargomento moderato O un distractor sopravvive
  flip    = controargomento forte E/O un distractor sopravvive

Restituisci ESCLUSIVAMENTE JSON valido. Nessun testo aggiuntivo.\
"""


def _build_critic_messages(
    sample: MCQSample,
    profiler: ProfilerOutput,
    miner: MinerOutput,
    solver: SolverOutput,
) -> list[dict]:
    attack_checklist = "\n".join(
        f"  □ {v}"
        for v in ATTACK_VECTORS.get(profiler.question_type, ATTACK_VECTORS["factual"])
    )

    distractors = "\n".join(
        f"  {opt}) \"{profiler.option_claims.get(opt, sample.options.get(opt, ''))}\""
        for opt in ("A", "B", "C", "D")
        if opt != solver.proposed_answer
    )

    passages_block = "\n\n".join(
        f"[{p.passage_id}] {p.key_extraction}"
        for p in (miner.relevant_passages or [])[:4]
    )

    calc_block = ""
    for opt in ("A", "B", "C", "D"):
        ev = miner.option_scores.get(opt)
        if ev and ev.calculation_trace:
            calc_block += f"\n  Opzione {opt}: {ev.calculation_trace}"

    neg_warning = (
        "\n⚠️ NEGAZIONE PRESENTE — controlla inversione logica come prima cosa."
        if profiler.negation_present else ""
    )

    reasoning_text = "\n".join(
        f"  {i+1}. {step}"
        for i, step in enumerate(solver.reasoning_steps[:6])
    )

    schema = """{
  "phase_1_challenge": {
    "strongest_counterargument": string,
    "counterargument_type": "arithmetic_error"|"qualifier_missed"|"wrong_period"|"wrong_base"|"negation_inversion"|"knowledge_override"|"table_misread"|"other",
    "counterargument_strength": "weak"|"moderate"|"strong",
    "rebuttal": string,
    "proposed_answer_survives": boolean
  },
  "phase_2_distractors": {
    "X": {"best_argument_for_option": string, "argument_strength": "weak"|"moderate"|"strong", "rebuttal": string, "survives_rebuttal": boolean}
  },
  "overall_verdict": "confirm"|"weaken"|"flip",
  "confidence_adjustment": float,
  "recommended_answer": "A"|"B"|"C"|"D",
  "flip_reason": string|null
}"""

    user_content = (
        f"DOMANDA: {sample.question}{neg_warning}\n\n"
        f"TIPO: {profiler.question_type}\n\n"
        f"OPZIONI:\n"
        + "\n".join(f"{k}) {sample.options[k]}" for k in "ABCD")
        + f"\n\nRISPOSTA PROPOSTA: {solver.proposed_answer} "
        f"(confidence={solver.confidence:.2f}, SC={solver.consistency_score:.2f})\n\n"
        f"RAGIONAMENTO DEL SOLUTORE:\n{reasoning_text}\n\n"
        f"CHECKLIST ATTACCHI ({profiler.question_type}):\n{attack_checklist}\n\n"
        f"EVIDENZE CHIAVE:\n{passages_block or 'Nessuna'}"
        + (f"\n\nTRACCE DI CALCOLO:{calc_block}" if calc_block else "")
        + f"\n\nDISTRATTORI DA DIFENDERE:\n{distractors}"
        + f"\n\nRestituisci questo JSON:\n{schema}"
    )

    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────
#  AdversarialCritic class
# ─────────────────────────────────────────────────────────────

class AdversarialCritic:
    """Stage 3 — runs on LocalModel (T4-0, shared with Profiler)."""

    def __init__(self, model: LocalModel):
        self.model = model

    def run(
        self,
        sample: MCQSample,
        profiler: ProfilerOutput,
        miner: MinerOutput,
        solver: SolverOutput,
    ) -> CriticOutput:
        messages    = _build_critic_messages(sample, profiler, miner, solver)
        parsed, err = self.model.generate_json(messages, CriticOutput, max_new_tokens=700)

        if parsed is None:
            print(f"  [Critic] WARN: {sample.sample_id}: {err}")
            return _fallback_critic(solver.proposed_answer)

        # Sanity: flip without a valid alternative → downgrade to weaken
        if (
            parsed.overall_verdict == "flip"
            and parsed.recommended_answer == solver.proposed_answer
        ):
            parsed.overall_verdict = "weaken"
            parsed.confidence_adjustment = max(parsed.confidence_adjustment, -0.15)

        return parsed

    def run_batch(
        self,
        samples: list[MCQSample],
        profilers: list[ProfilerOutput],
        miners: list[MinerOutput],
        solvers: list[SolverOutput],
        verbose: bool = True,
    ) -> list[CriticOutput]:
        assert len(samples) == len(profilers) == len(miners) == len(solvers)
        results = []
        for i, (s, p, m, sv) in enumerate(zip(samples, profilers, miners, solvers)):
            if verbose and i % 10 == 0:
                print(f"  Critic: {i}/{len(samples)}")
            results.append(self.run(s, p, m, sv))
        return results


def _fallback_critic(proposed: str) -> CriticOutput:
    """Safe default when the model fails: mildly confirm the solver."""
    from ..schemas import Phase1Challenge, DistractorAnalysis
    return CriticOutput(
        phase_1_challenge=Phase1Challenge(
            strongest_counterargument="Analisi non disponibile.",
            counterargument_type="other",
            counterargument_strength="weak",
            rebuttal="",
            proposed_answer_survives=True,
        ),
        phase_2_distractors={
            opt: DistractorAnalysis(
                best_argument_for_option="",
                argument_strength="weak",
                rebuttal="",
                survives_rebuttal=False,
            )
            for opt in ("A", "B", "C", "D")
            if opt != proposed
        },
        overall_verdict="confirm",
        confidence_adjustment=0.0,
        recommended_answer=proposed,  # type: ignore[arg-type]
        flip_reason=None,
    )
