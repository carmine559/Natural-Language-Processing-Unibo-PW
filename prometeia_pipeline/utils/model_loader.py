"""
utils/model_loader.py — load instruct models on the GPU.

The DISI cluster gives one GPU per job. On the L40 (48 GB) we load two models
simultaneously: a lighter one (14B) for the profiler + miner, and a heavier
one (32B) for the solver + critic. On the RTX 2080 Ti (11 GB) both point to
the same 14B model.

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

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or CONFIG.model.gpu_device
        self.model_name = model_name
        print(f"Loading {self.model_name} on {self.device} ...")

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
        print(f"  {self.model_name} loaded. VRAM used: {self._vram_used_gb():.1f} GB")

    # ── Raw generation ────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = CONFIG.local_max_new_tokens,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
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
# On L40: light = 14B, heavy = 32B (both loaded at once, ~32 GB total).
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
        )
    return _light_model


def get_heavy_model() -> LocalModel:
    """Return the heavy model (solver + critic).

    If the heavy name matches the light name, returns the same instance
    to avoid loading the same model twice (RTX 2080 Ti case).
    """
    global _heavy_model
    if _heavy_model is None:
        if CONFIG.model.heavy_model_name == CONFIG.model.light_model_name:
            _heavy_model = get_light_model()
        else:
            _heavy_model = LocalModel(
                model_name=CONFIG.model.heavy_model_name,
                device=CONFIG.model.gpu_device,
            )
    return _heavy_model
