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
#    l40       → Nvidia L40, 48 GB  → can host a 14B (or larger) in 4-bit
#    rtx2080   → RTX 2080 Ti, 11 GB → 14B-4bit fits but KV-cache is tight;
#                drop to the 7B model below if you hit OOM on long contexts.
# ─────────────────────────────────────────────────────────────
MODEL_PRESETS: Dict[str, str] = {
    "l40":     "Qwen/Qwen2.5-14B-Instruct",
    "rtx2080": "Qwen/Qwen2.5-14B-Instruct",   # OOM fallback: "Qwen/Qwen2.5-7B-Instruct"
}


@dataclass
class ModelConfig:
    # Single local model shared by every stage (one GPU per node).
    local_model_name: str = os.environ.get(
        "PROM_MODEL", "Qwen/Qwen2.5-14B-Instruct"
    )
    use_4bit: bool = True            # ~9 GB for 14B, ~5 GB for 7B
    bnb_compute_dtype: str = "float16"

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
    """Pick the model for a given cluster partition (l40 / rtx2080),
    unless PROM_MODEL / --model already forced a specific one."""
    if os.environ.get("PROM_MODEL"):
        return
    name = MODEL_PRESETS.get(partition)
    if name:
        CONFIG.model.local_model_name = name
