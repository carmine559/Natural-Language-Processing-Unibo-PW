"""
utils/groq_client.py — thin wrapper around the Groq SDK.

Handles:
  - Exponential backoff on RateLimitError and ServiceUnavailable
  - Automatic retry with jitter
  - Self-consistency sampling (multiple calls at higher temperature)
  - Usage logging so you can track token consumption during a run
"""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# `groq` is an optional dependency — only required when running with
# --use-groq. Import lazily so the fully-local cluster path works without it.
try:
    from groq import Groq, RateLimitError, APIStatusError
except ImportError:  # pragma: no cover
    Groq = None
    class RateLimitError(Exception): ...
    class APIStatusError(Exception):
        status_code = None

from ..config import CONFIG, GROQ_API_KEY


@dataclass
class UsageStats:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    rate_limit_hits: int = 0

    def log(self, usage) -> None:
        self.total_calls += 1
        if usage:
            self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self.total_completion_tokens += getattr(usage, "completion_tokens", 0)

    def summary(self) -> str:
        return (
            f"Groq: {self.total_calls} calls | "
            f"{self.total_prompt_tokens:,} prompt + "
            f"{self.total_completion_tokens:,} completion tokens | "
            f"{self.rate_limit_hits} rate-limit retries"
        )


class GroqClient:
    """
    Wraps groq.Groq with retry logic and SC helpers.
    Instantiate once at the top of your notebook.
    """

    def __init__(self, api_key: str = GROQ_API_KEY):
        if Groq is None:
            raise ImportError(
                "The 'groq' package is not installed but --use-groq was set. "
                "Install it (pip install groq) or run fully local."
            )
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is empty. Set os.environ['GROQ_API_KEY'] = '...' "
                "before running with --use-groq."
            )
        self.client = Groq(api_key=api_key)
        self.stats = UsageStats()

    # ── Core call ─────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = CONFIG.model.groq_temperature_default,
        max_tokens: int = CONFIG.model.groq_max_tokens,
        max_retries: int = CONFIG.groq_max_retries,
    ) -> str:
        """
        Call the Groq chat API with exponential backoff.
        Returns the assistant message content as a string.
        """
        model = model or CONFIG.model.groq_solver_model
        delay = CONFIG.groq_retry_base_delay

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.stats.log(getattr(resp, "usage", None))
                return resp.choices[0].message.content.strip()

            except RateLimitError:
                self.stats.rate_limit_hits += 1
                if attempt == max_retries - 1:
                    raise
                jitter = random.uniform(0, delay * 0.2)
                wait = min(delay + jitter, CONFIG.groq_retry_max_delay)
                print(f"  Rate limit hit — waiting {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                delay *= 2

            except APIStatusError as e:
                if e.status_code in (500, 502, 503, 529):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(min(delay, CONFIG.groq_retry_max_delay))
                    delay *= 2
                else:
                    raise

        raise RuntimeError(f"Groq call failed after {max_retries} attempts")

    # ── Self-consistency ──────────────────────────────────────

    def chat_with_sc(
        self,
        messages: list[dict],
        n_samples: int = CONFIG.model.groq_sc_samples,
        model: Optional[str] = None,
        temperature: float = CONFIG.model.groq_temperature_high_difficulty,
        max_tokens: int = CONFIG.model.groq_max_tokens,
    ) -> tuple[list[str], dict[str, int]]:
        """
        Run n_samples calls and return (raw_responses, vote_counts).
        vote_counts = {"A": 2, "B": 1, "C": 0, "D": 0} extracted from responses.
        """
        responses = []
        for i in range(n_samples):
            # Alternate temperature slightly for diversity
            t = temperature + (i * 0.05)
            responses.append(
                self.chat(messages, model=model, temperature=t, max_tokens=max_tokens)
            )

        # Extract A/B/C/D votes from each response
        import re
        votes: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        for r in responses:
            # Look for "proposed_answer": "X" in the JSON, or standalone "X"
            m = re.search(r'"proposed_answer"\s*:\s*"([ABCD])"', r)
            if m:
                votes[m.group(1)] += 1
            else:
                # Fallback: last standalone letter
                letters = re.findall(r'\b([ABCDE])\b', r)
                if letters:
                    votes[letters[-1]] += 1

        return responses, votes

    def print_stats(self) -> None:
        print(self.stats.summary())
