import os
from dataclasses import dataclass, field
from typing import Dict

# ─────────────────────────────────────────────────────────────
#  API keys — only used when CONFIG.model.use_groq is True.
#  On the cluster the pipeline runs fully local by default and
#  this can be left empty.
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")


# ─────────────────────────────────────────────────────────────
#  Model presets per cluster partition.
#  The DISI cluster gives ONE GPU per node:
#    l40       → Nvidia L40, 48 GB  → dual model: 14B + 32B (both 4-bit, ~28 GB)
#    rtx2080   → RTX 2080 Ti, 11 GB → single 14B-4bit only (~9 GB)
#  Qwen3 generation: the heavy model runs in thinking mode (reasoning traces
#  before the answer), which needs a larger generation budget — see
#  ModelConfig.thinking_max_new_tokens.
# ─────────────────────────────────────────────────────────────
MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "l40": {
        "light": "Qwen/Qwen3-14B",    # profiler + miner
        "heavy": "Qwen/Qwen3-32B",    # solver + critic (thinking mode)
    },
    "rtx2080": {
        "light": "Qwen/Qwen3-14B",    # all stages (11 GB limit; use Qwen3-8B if OOM)
        "heavy": "Qwen/Qwen3-14B",    # same — no room for 32B
    },
}


@dataclass
class ModelConfig:
    # Light model: profiler + miner (classification / evidence scoring).
    light_model_name: str = os.environ.get(
        "PROM_LIGHT_MODEL", "Qwen/Qwen3-14B"
    )
    # Heavy model: solver + critic (deeper reasoning).
    heavy_model_name: str = os.environ.get(
        "PROM_HEAVY_MODEL", "Qwen/Qwen3-32B"
    )
    use_4bit: bool = True            # ~9 GB for 14B, ~20 GB for 32B
    bnb_compute_dtype: str = "float16"

    # Qwen3 thinking mode. The light model answers directly (fast); the heavy
    # model reasons inside a <think>...</think> block before answering, which
    # measurably helps solver/critic/statement-evaluator quality.
    # Thinking calls ignore small per-call token budgets and generate up to
    # thinking_max_new_tokens (the think block alone can take 1-2k tokens).
    light_enable_thinking: bool = False
    heavy_enable_thinking: bool = True
    thinking_max_new_tokens: int = 3072

    # The critic runs on the heavy model but WITHOUT thinking: the Qwen3
    # thinking critic became flip-happy (27 flips on 220 samples vs 2 for
    # Qwen2.5, with only 33% of flip-verdict samples ending correct) and its
    # calls dominated runtime. Overrides heavy_enable_thinking for Stage 3.
    critic_enable_thinking: bool = False

    # The cluster exposes exactly one GPU per job (--gres=gpu:1).
    gpu_device: str = "cuda:0"

    # Optional Groq path (only used when use_groq=True). The local model
    # otherwise plays the role of the solver/tiebreaker too.
    use_groq: bool = False
    groq_solver_model: str = "llama-3.3-70b-versatile"
    groq_light_model: str = "llama-3.1-8b-instant"  # tiebreaker
    groq_max_tokens: int = 1024
    groq_temperature_default: float = 0.0
    groq_temperature_high_difficulty: float = 0.2   # used for SC samples
    groq_sc_samples: int = 3                         # for "high" difficulty


@dataclass
class AggregatorConfig:
    # Weights shift based on critic verdict — see design doc
    weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "confirm": {"miner": 0.25, "solver": 0.45, "sc": 0.15, "critic": 0.15},
        "weaken":  {"miner": 0.25, "solver": 0.35, "sc": 0.15, "critic": 0.25},
        "flip":    {"miner": 0.25, "solver": 0.25, "sc": 0.10, "critic": 0.40},
    })

    # Miner weight nudge by question type
    miner_type_boost: Dict[str, float] = field(default_factory=lambda: {
        "numerical":    +0.10,
        "factual":      +0.05,
        "comparative":  +0.02,
        "temporal":     +0.02,
        "definitional":  0.00,
        "causal":       -0.05,
        "negation":     -0.05,
        "inferential":  -0.10,
    })

    margin_high: float = 0.15   # commit directly
    margin_low: float = 0.05    # commit + flag
    # below margin_low → binary tiebreaker

    tiebreak_confidence_threshold: float = 0.65
    miner_gap_for_fallback: float = 0.08   # priority 2 in fallback chain

    # E suppression: in the Prometeia dataset the gold answer is never E.
    # When True, E's score is redistributed to A–D proportionally.
    suppress_e: bool = True

    # Critic flip dampening: flips hurt accuracy on Qwen2.5 (42% vs 66% for
    # weaken) and got worse with Qwen3 (33% of flip-verdict samples correct;
    # 19/27 flipped questions were previously answered right). Even the
    # "weaken" downgrade still moves weight off the solver, so flips are now
    # neutralized to "confirm" by default ("weaken" = legacy behaviour).
    dampen_flips: bool = True
    flip_downgrade_to: str = "confirm"


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)

    # Groq retry settings (optional path)
    groq_max_retries: int = 4
    groq_retry_base_delay: float = 2.0      # seconds, doubles each retry
    groq_retry_max_delay: float = 30.0

    # Local model generation
    local_max_new_tokens: int = 600
    local_json_max_retries: int = 3

    # BM25 retrieval
    bm25_top_k: int = 6
    bm25_always_include_tables: bool = True

    # Context chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # Paths — env-driven so the same code runs locally and on the cluster.
    # On the cluster, point these at /scratch.hpc/<user>/... via the launcher.
    data_dir: str = os.environ.get("PROM_DATA_DIR", "./data")
    output_dir: str = os.environ.get("PROM_OUTPUT_DIR", "./outputs")
    cache_dir: str = os.environ.get(
        "PROM_CACHE_DIR",
        os.path.join(os.environ.get("PROM_OUTPUT_DIR", "./outputs"), ".cache"),
    )


CONFIG = PipelineConfig()


def apply_partition_preset(partition: str) -> None:
    """Pick models for a given cluster partition (l40 / rtx2080),
    unless env vars already forced specific ones."""
    preset = MODEL_PRESETS.get(partition, {})
    if not os.environ.get("PROM_LIGHT_MODEL") and "light" in preset:
        CONFIG.model.light_model_name = preset["light"]
    if not os.environ.get("PROM_HEAVY_MODEL") and "heavy" in preset:
        CONFIG.model.heavy_model_name = preset["heavy"]
