"""
utils/model_loader.py — load instruct models on the GPU.

The DISI cluster gives one GPU per job. On the L40 (48 GB) we load two models
simultaneously: a lighter one (Qwen3-14B) for the profiler + miner, and a
heavier one (Qwen3-32B, thinking mode) for the solver + critic. On the
RTX 2080 Ti (11 GB) both point to the same 14B model.

Qwen3 thinking mode: the heavy model reasons inside a <think>...</think>
block before answering. generate() strips the block and returns only the
final answer text; sampling and token budgets are overridden per the Qwen3
usage guidelines (greedy decoding degrades thinking mode).

Usage:
    from prometeia_pipeline.utils.model_loader import get_light_model, get_heavy_model
    light = get_light_model()    # profiler, miner
    heavy = get_heavy_model()    # solver, critic
"""
from __future__ import annotations

import gc
import time
from typing import Type, TypeVar

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import CONFIG
from .json_parser import parse_with_schema, build_retry_suffix

T = TypeVar("T", bound=BaseModel)


class LocalModel:
    """
    Wraps a quantized instruct model for text + structured JSON generation.
    One instance per model — instantiate once at startup and reuse.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        enable_thinking: bool = False,
    ):
        self.device = device or CONFIG.model.gpu_device
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        print(f"Loading {self.model_name} on {self.device} "
              f"(thinking={'on' if enable_thinking else 'off'}) ...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=CONFIG.model.use_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if CONFIG.model.use_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not CONFIG.model.use_4bit else None,
        )
        self.model.eval()

        # Qwen3 wraps its reasoning in <think>...</think>; resolve the token
        # ids once so generate() can split reasoning from the final answer.
        # On tokenizers without these tokens (e.g. Qwen2.5) stripping is a no-op.
        self._think_open_id = self._token_id_or_none("<think>")
        self._think_close_id = self._token_id_or_none("</think>")

        print(f"  {self.model_name} loaded. VRAM used: {self._vram_used_gb():.1f} GB")

    def _token_id_or_none(self, token: str) -> int | None:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk = self.tokenizer.unk_token_id
        if tid is None or (unk is not None and tid == unk):
            return None
        return tid

    # ── Raw generation ────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = CONFIG.local_max_new_tokens,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if self.enable_thinking:
            # Qwen3 thinking mode degrades under greedy decoding (endless
            # repetitions), and the <think> block alone can take 1-2k tokens —
            # override the caller's answer-sized budget and sampling settings
            # with the vendor-recommended ones (temp 0.6 / top_p 0.95 / top_k 20).
            gen_kwargs = dict(
                max_new_tokens=max(max_new_tokens,
                                   CONFIG.model.thinking_max_new_tokens),
                do_sample=True,
                temperature=max(0.6, temperature if do_sample else 0.0),
                top_p=0.95,
                top_k=20,
            )
        elif do_sample:
            # Non-thinking sampled calls (vendor-recommended top_p/top_k).
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.8,
                top_k=20,
            )
        else:
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = out[0][inputs.input_ids.shape[1]:].tolist()
        new_ids = self._strip_think_block(new_ids)
        if new_ids is None:
            # Think block opened but never closed: the budget ran out before
            # any answer was produced. Return "" so generate_json's retry
            # (or the caller's regex fallback) kicks in.
            return ""
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _strip_think_block(self, token_ids: list[int]) -> list[int] | None:
        """Drop the <think>...</think> reasoning block from generated ids.

        Returns None when a think block was opened but never closed
        (generation budget exhausted mid-reasoning → no usable answer).
        """
        if not self.enable_thinking or self._think_close_id is None:
            return token_ids
        try:
            idx = len(token_ids) - token_ids[::-1].index(self._think_close_id)
            return token_ids[idx:]
        except ValueError:
            if self._think_open_id is not None and self._think_open_id in token_ids:
                return None
            return token_ids

    # ── JSON generation with retry ────────────────────────────

    def generate_json(
        self,
        messages: list[dict],
        schema: Type[T],
        max_retries: int = CONFIG.local_json_max_retries,
        max_new_tokens: int = CONFIG.local_max_new_tokens,
    ) -> tuple[T | None, str | None]:
        current_messages = list(messages)

        for attempt in range(max_retries):
            raw = self.generate(current_messages, max_new_tokens=max_new_tokens)
            parsed, error = parse_with_schema(raw, schema)

            if parsed is not None:
                return parsed, None

            if attempt < max_retries - 1:
                current_messages.append({"role": "assistant", "content": raw})
                current_messages.append({
                    "role": "user",
                    "content": build_retry_suffix(error, schema),
                })

        return None, error

    # ── Utilities ─────────────────────────────────────────────

    def _vram_used_gb(self) -> float:
        idx = int(self.device.split(":")[-1])
        return torch.cuda.memory_allocated(idx) / 1e9

    def free(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Model freed from {self.device}")


# ── Singletons ────────────────────────────────────────────────
# On L40: light = Qwen3-14B (no thinking), heavy = Qwen3-32B (thinking mode),
# both loaded at once (~28 GB total).
# On RTX 2080 Ti: both point to the same 14B instance.

_light_model: LocalModel | None = None
_heavy_model: LocalModel | None = None


def get_light_model() -> LocalModel:
    """Return the light model (profiler + miner)."""
    global _light_model
    if _light_model is None:
        _light_model = LocalModel(
            model_name=CONFIG.model.light_model_name,
            device=CONFIG.model.gpu_device,
            enable_thinking=CONFIG.model.light_enable_thinking,
        )
    return _light_model


def get_heavy_model() -> LocalModel:
    """Return the heavy model (solver + critic).

    If the heavy name matches the light name, returns the same instance
    to avoid loading the same model twice (RTX 2080 Ti case). The shared
    instance keeps the LIGHT thinking setting — i.e. thinking stays off on
    single-model setups, where it would be prohibitively slow anyway.
    """
    global _heavy_model
    if _heavy_model is None:
        if CONFIG.model.heavy_model_name == CONFIG.model.light_model_name:
            _heavy_model = get_light_model()
        else:
            _heavy_model = LocalModel(
                model_name=CONFIG.model.heavy_model_name,
                device=CONFIG.model.gpu_device,
                enable_thinking=CONFIG.model.heavy_enable_thinking,
            )
    return _heavy_model
