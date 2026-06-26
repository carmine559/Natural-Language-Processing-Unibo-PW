"""
stages/miner.py — Context Miner (Stage 1B).

Runs on the shared local model (single GPU per cluster job).

Two internal sub-stages:
  1. Smart chunking + BM25 retrieval  (CPU, fast)
  2. LLM evidence scorer              (local model, GPU)

Usage:
    model = get_light_model()
    miner = ContextMiner(model)
    output = miner.run(sample, profiler_output)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

from ..schemas import MCQSample, ProfilerOutput, MinerOutput, OptionEvidence, PassageScore, OPTS
from ..utils.model_loader import LocalModel
from ..config import CONFIG


# ─────────────────────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: int
    text: str
    chunk_type: str          # "prose" | "table" | "list"
    source_hint: str         # section header or "unknown"


def _detect_tables(text: str) -> list[tuple[int, int, str]]:
    """
    Return (start, end, markdown_table) for each detected table.
    Handles pipe-formatted tables and whitespace-aligned columns.
    """
    tables = []
    # Pipe-formatted tables  |col1|col2|...
    for m in re.finditer(r'(\|[^\n]+\|\n(?:\|[-: ]+\|\n)?(?:\|[^\n]+\|\n?)+)', text):
        tables.append((m.start(), m.end(), m.group(0)))
    # Whitespace-tabular blocks (≥ 3 rows, each starting with spaces + word + 2+ spaces)
    for m in re.finditer(
        r'((?:[ \t]+\S+[ \t]{2,}.*\n){3,})',
        text,
    ):
        block = m.group(0)
        # Convert to markdown-style for the LLM
        rows = [re.split(r'[ \t]{2,}', ln.strip()) for ln in block.strip().splitlines()]
        md = "\n".join("| " + " | ".join(cells) + " |" for cells in rows if cells)
        tables.append((m.start(), m.end(), md))
    return tables


def smart_chunk(
    text: str,
    max_tokens: int = CONFIG.chunk_max_tokens,
    overlap: int = CONFIG.chunk_overlap_tokens,
) -> list[Chunk]:
    """
    Split a financial document into chunks that respect structure:
      - Tables are kept atomic (never split)
      - Section headers start new chunks
      - Paragraphs are grouped up to max_tokens
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    tables = _detect_tables(text)
    table_ranges = {(s, e) for s, e, _ in tables}

    # Mark table positions so we skip them in prose processing
    in_table_chars: set[int] = set()
    for start, end, md in tables:
        in_table_chars.update(range(start, end))
        chunks.append(Chunk(
            chunk_id=len(chunks),
            text=md,
            chunk_type="table",
            source_hint="tabella estratta",
        ))

    # Rebuild prose with tables removed
    prose = "".join(c if i not in in_table_chars else " " for i, c in enumerate(text))

    # Split by section headers (numbered: "1.", "2.1", or "##" markdown)
    section_pattern = re.compile(r'(?=\n(?:\d+\.)+\s|\n#{1,3}\s)')
    sections = section_pattern.split(prose)

    for section in sections:
        if not section.strip():
            continue
        # Extract section header
        header_match = re.match(r'((?:\d+\.)+\s+[^\n]+|#{1,3}\s+[^\n]+)', section)
        header = header_match.group(0).strip() if header_match else "sezione"

        # Split section into paragraphs and group them
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', section) if p.strip()]
        current_paras: list[str] = []
        current_toks = 0

        for para in paragraphs:
            para_toks = len(para.split())
            if current_toks + para_toks > max_tokens and current_paras:
                chunks.append(Chunk(
                    chunk_id=len(chunks),
                    text="\n\n".join(current_paras),
                    chunk_type="prose",
                    source_hint=header,
                ))
                # Overlap: keep last paragraph for context continuity
                current_paras = current_paras[-1:] if overlap > 0 else []
                current_toks = len(current_paras[0].split()) if current_paras else 0
            current_paras.append(para)
            current_toks += para_toks

        if current_paras:
            chunks.append(Chunk(
                chunk_id=len(chunks),
                text="\n\n".join(current_paras),
                chunk_type="prose",
                source_hint=header,
            ))

    # Re-assign sequential IDs
    for i, c in enumerate(chunks):
        c.chunk_id = i
    return chunks


# ─────────────────────────────────────────────────────────────
#  BM25 Retrieval
# ─────────────────────────────────────────────────────────────

def _simple_tokenize(text: str) -> list[str]:
    """
    Lowercase + split on non-alphanumeric. Tries spacy if available,
    falls back silently — spacy is optional.
    """
    try:
        import spacy
        nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
        doc = nlp(text.lower())
        return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.text.strip()]
    except Exception:
        return re.findall(r'\w+', text.lower())


def bm25_retrieve(
    chunks: list[Chunk],
    profiler: ProfilerOutput,
    top_k: int = CONFIG.bm25_top_k,
) -> list[Chunk]:
    """
    Retrieve the top_k most relevant chunks using BM25.
    Always includes tables (they contain answers disproportionately often).
    """
    if not chunks:
        return []

    query_parts = [
        profiler.retrieval_focus,
        " ".join(profiler.key_terms),
        " ".join(profiler.key_numbers),
        profiler.question_type,
    ]
    query = " ".join(query_parts)
    query_tokens = _simple_tokenize(query)

    corpus = [_simple_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top_ids = list(dict.fromkeys(ranked[:top_k]))  # deduplicate, preserve order

    # Always include tables if configured
    if CONFIG.bm25_always_include_tables:
        table_ids = [i for i, c in enumerate(chunks) if c.chunk_type == "table"]
        top_ids = list(dict.fromkeys(top_ids + table_ids))[:top_k + len(table_ids)]

    return [chunks[i] for i in top_ids]


# ─────────────────────────────────────────────────────────────
#  LLM Scorer prompts
# ─────────────────────────────────────────────────────────────

MINER_SYSTEM = """\
Sei un analista di evidenze per documenti finanziari italiani.
Per ogni opzione (A, B, C, D) valuta quanto è supportata dalle evidenze fornite.

Restituisci ESCLUSIVAMENTE un oggetto JSON valido. Nessun testo aggiuntivo.

support_score: 0.0=fortemente contraddetta ... 1.0=fortemente supportata
verdict: "strongly_supported"|"supported"|"neutral"|"contradicted"|"strongly_contradicted"
confidence: "high"|"medium"|"low"\
"""


def _build_miner_messages(
    sample: MCQSample,
    profiler: ProfilerOutput,
    top_chunks: list[Chunk],
) -> list[dict]:
    passages_block = "\n\n".join(
        f"[Passaggio {i+1} | {c.chunk_type} | {c.source_hint}]\n{c.text[:600]}"
        for i, c in enumerate(top_chunks)
    )

    calc_note = (
        '\n    "calculation_trace": "mostra ogni passo aritmetico",'
        if profiler.question_type == "numerical"
        else ""
    )

    schema = "{\n"
    schema += '  "relevant_passages": [{"passage_id": int, "relevance_score": float, "key_extraction": string}],\n'
    schema += '  "option_scores": {\n'
    for opt in ("A", "B", "C", "D"):
        claim = profiler.option_claims.get(opt, sample.options.get(opt, ""))
        schema += (
            f'    "{opt}": {{'
            f'"support_score": float, "verdict": string, '
            f'"evidence_snippet": string,{calc_note} '
            f'"confidence": string}},\n'
        )
    schema += "  },\n"
    schema += '  "evidence_coverage": "complete"|"partial"|"absent",\n'
    schema += '  "tables_found": boolean,\n'
    schema += '  "numerical_values_extracted": [string],\n'
    schema += '  "missing_information": string|null\n}'

    user_content = (
        f"DOMANDA: {sample.question}\n\n"
        f"AFFERMAZIONI DELLE OPZIONI:\n"
        + "\n".join(
            f"{k}: {profiler.option_claims.get(k, sample.options.get(k, ''))}"
            for k in ("A", "B", "C", "D")
        )
        + f"\n\nPASSAGGI ESTRATTI:\n{passages_block}"
        + f"\n\nRestituisci questo JSON:\n{schema}"
    )

    return [
        {"role": "system", "content": MINER_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────
#  ContextMiner class
# ─────────────────────────────────────────────────────────────

class ContextMiner:
    """Stage 1B — runs on LocalModel (T4-1)."""

    def __init__(self, model: LocalModel):
        self.model = model

    def run(
        self,
        sample: MCQSample,
        profiler: ProfilerOutput,
    ) -> MinerOutput:
        # For statement_combination questions the context IS the statements —
        # BM25 retrieval doesn't help. Return statements as uniform evidence.
        if profiler.question_type == "statement_combination":
            return _statement_combo_miner(sample)
        chunks      = smart_chunk(sample.question)
        top_chunks  = bm25_retrieve(chunks, profiler)
        messages    = _build_miner_messages(sample, profiler, top_chunks)
        parsed, err = self.model.generate_json(messages, MinerOutput)

        if parsed is None:
            print(f"  [Miner] WARN: {sample.sample_id}: {err}")
            return _fallback_miner(sample)
        return parsed

    def run_batch(
        self,
        samples: list[MCQSample],
        profilers: list[ProfilerOutput],
        verbose: bool = True,
    ) -> list[MinerOutput]:
        assert len(samples) == len(profilers)
        results = []
        for i, (s, p) in enumerate(zip(samples, profilers)):
            if verbose and i % 10 == 0:
                print(f"  Miner: {i}/{len(samples)}")
            results.append(self.run(s, p))
        return results



def _statement_combo_miner(sample: MCQSample) -> MinerOutput:
    """
    For BOOKS__/PAPER__ statement-combination questions:
    the context contains the numbered statements, not a searchable document.
    Return each statement as a passage; option evidence is uniform (solver evaluates).
    """
    lines = [l for l in sample.question.splitlines() if l.strip()]
    passages = [
        PassageScore(passage_id=i, relevance_score=1.0, key_extraction=l[:120])
        for i, l in enumerate(lines)
    ]
    uniform = OptionEvidence(
        support_score=0.2,
        verdict="neutral",
        evidence_snippet="Valutazione affidata al solver (combinazione di affermazioni).",
        confidence="low",
    )
    return MinerOutput(
        relevant_passages=passages,
        option_scores={k: uniform for k in OPTS},
        evidence_coverage="partial",
        tables_found=False,
        numerical_values_extracted=[],
        missing_information="Statement-combo: il solver valuta ogni affermazione.",
    )


def _fallback_miner(sample: MCQSample) -> MinerOutput:
    uniform = OptionEvidence(
        support_score=0.25,
        verdict="neutral",
        evidence_snippet="Evidenza non disponibile.",
        confidence="low",
    )
    return MinerOutput(
        relevant_passages=[],
        option_scores={k: uniform for k in ("A", "B", "C", "D")},
        evidence_coverage="absent",
        tables_found=False,
        numerical_values_extracted=[],
        missing_information="Estrazione fallita.",
    )
