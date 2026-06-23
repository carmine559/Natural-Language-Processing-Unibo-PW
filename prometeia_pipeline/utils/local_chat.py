"""
utils/local_chat.py — local drop-in replacement for GroqClient.

The Specialist Solver (Stage 2) and the Aggregator tiebreaker (Stage 4) were
written against the Groq SDK and call `.chat(...)` / `.chat_with_sc(...)`.
On the cluster we route those same calls through the shared local model so the
pipeline runs fully offline. The interface mirrors GroqClient exactly, so
solver.py / aggregator.py need no changes.
"""
from __future__ import annotations

import re

from ..config import CONFIG
from .model_loader import LocalModel


class LocalChatClient:
    """Mimics GroqClient.chat / chat_with_sc using a local LocalModel."""

    def __init__(self, model: LocalModel):
        self.model = model
        self.n_calls = 0

    # ── Core call ─────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,        # ignored — single local model
        temperature: float = CONFIG.model.groq_temperature_default,
        max_tokens: int = CONFIG.model.groq_max_tokens,
        max_retries: int = 1,            # local generation doesn't rate-limit
    ) -> str:
        self.n_calls += 1
        return self.model.generate(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )

    # ── Self-consistency ──────────────────────────────────────

    def chat_with_sc(
        self,
        messages: list[dict],
        n_samples: int = CONFIG.model.groq_sc_samples,
        model: str | None = None,
        temperature: float = CONFIG.model.groq_temperature_high_difficulty,
        max_tokens: int = CONFIG.model.groq_max_tokens,
    ) -> tuple[list[str], dict[str, int]]:
        """Run n_samples sampled generations; return (raw_responses, vote_counts)."""
        responses = []
        for i in range(n_samples):
            t = temperature + (i * 0.05)   # slight diversity, mirrors GroqClient
            responses.append(
                self.chat(messages, temperature=t, max_tokens=max_tokens)
            )

        votes: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        for r in responses:
            m = re.search(r'"proposed_answer"\s*:\s*"([ABCDE])"', r)
            if m:
                votes[m.group(1)] += 1
            else:
                letters = re.findall(r'\b([ABCDE])\b', r)
                if letters:
                    votes[letters[-1]] += 1

        return responses, votes

    def print_stats(self) -> None:
        print(f"LocalChatClient: {self.n_calls} generation calls")
