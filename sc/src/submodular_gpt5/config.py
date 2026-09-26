from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class AppConfig:
    dataset: str
    mode: str
    data_path: str
    output_path: str
    predictions_path: str | None = None

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
    embedding_device: str | None = None
    local_files_only: bool = False

    concurrency: int = 6
    answer_max_output_tokens: int = 2048
    distill_max_output_tokens: int = 1024
    reasoning_effort: str = "minimal"
    text_verbosity: str = "low"
    distill: bool = True
    seed: int = 42


@dataclass(slots=True)
class MemoryConfig:
    # Shared query-time GranularSelect.
    query_budget: int = 4096
    chunk_tokens: int = 128
    lambda_coverage: float = 0.10
    lambda_diversity: float = 0.10
    lambda_relevance: float = 0.80
    granular_epsilon: float = 0.10

    relevance_mode: str = "hybrid"
    dense_weight: float = 0.55
    lexical_weight: float = 0.30
    entity_weight: float = 0.10
    time_weight: float = 0.05
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Dynamic Stage-1. Ignored in static mode.
    stream_budget: int = 30000
    stream_lambda_coverage: float = 0.20
    stream_lambda_diversity: float = 0.80
    feature_coverage_tau: float = 1.0
    stream_epsilon: float = 0.30
    stream_beta: float = 1.0 / 8.0

    # Query-independent fact protection in dynamic mode.
    fact_protector: bool = True
    fact_buffer_budget: int = 16000
    fact_min_score: float = 0.17
    fact_cost_exponent: float = 0.25
    fact_replacement_budget: int = 16000
    fact_query_bonus: float = 0.15

    encode_batch_size: int = 32
    token_safety_margin_per_chunk: int = 0
    raise_on_oversized_turn: bool = False


def preset_config(dataset: str, mode: str) -> MemoryConfig:
    dataset = dataset.lower()
    mode = mode.lower()
    if dataset not in {"longmemeval", "locomo"}:
        raise ValueError("dataset must be 'longmemeval' or 'locomo'")
    if mode not in {"static", "dynamic"}:
        raise ValueError("mode must be 'static' or 'dynamic'")

    cfg = MemoryConfig()
    if mode == "static" and dataset == "longmemeval":
        cfg.chunk_tokens = 256
    else:
        cfg.chunk_tokens = 128

    if dataset == "longmemeval":
        cfg.stream_budget = 80000
        cfg.stream_beta = 1.0 / 32.0
    else:
        cfg.stream_budget = 30000
        cfg.stream_beta = 1.0 / 8.0
    return cfg
