"""
utils/model_loader.py — load the shared local instruct model on the GPU.

The DISI cluster gives one GPU per job, so a single model instance is loaded
once and shared by every stage (profiler, miner, critic, and — unless
--use-groq is set — the solver and tiebreaker too).

Usage:
    from prometeia_pipeline.utils.model_loader import get_shared_model
    model = get_shared_model()
    result, error = model.generate_json(messages, ProfilerOutput)
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
    One instance per GPU — instantiate once at startup and reuse.
    """

    def __init__(self, device: str | None = None):
        self.device = device or CONFIG.model.gpu_device
        self.model_name = CONFIG.model.local_model_name
        print(f"Loading {self.model_name} on {device} ...")

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
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not CONFIG.model.use_4bit else None,
        )
        self.model.eval()
        print(f"  Model loaded. VRAM used: {self._vram_used_gb():.1f} GB")

    # ── Raw generation ────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = CONFIG.local_max_new_tokens,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """Generate text from a chat-format message list."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── JSON generation with retry ────────────────────────────

    def generate_json(
        self,
        messages: list[dict],
        schema: Type[T],
        max_retries: int = CONFIG.local_json_max_retries,
        max_new_tokens: int = CONFIG.local_max_new_tokens,
    ) -> tuple[T | None, str | None]:
        """
        Generate a structured JSON response and parse it into `schema`.
        Returns (parsed_object, None) on success, (None, error_msg) on failure.

        On validation error the error message is appended to the conversation
        and the model gets one more chance — this simple re-prompt trick
        recovers most formatting mistakes without touching the system prompt.
        """
        current_messages = list(messages)

        for attempt in range(max_retries):
            raw = self.generate(current_messages, max_new_tokens=max_new_tokens)
            parsed, error = parse_with_schema(raw, schema)

            if parsed is not None:
                return parsed, None

            if attempt < max_retries - 1:
                # Feed error back so the model can self-correct
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
        """Release VRAM — call before loading another model on the same GPU."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Model freed from {self.device}")


# ── Shared singleton ──────────────────────────────────────────
# One GPU per job → one model instance shared by all stages.
# Call get_shared_model() once and pass it to every stage constructor.

_shared_model: LocalModel | None = None


def get_shared_model() -> LocalModel:
    global _shared_model
    if _shared_model is None:
        _shared_model = LocalModel(device=CONFIG.model.gpu_device)
    return _shared_model
