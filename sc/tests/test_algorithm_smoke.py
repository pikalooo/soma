import hashlib

import numpy as np

from submodular_gpt5.config import MemoryConfig
from submodular_gpt5.memory.static import StaticSubmodularMemoryCompressor
from submodular_gpt5.memory.streaming import StreamingMemoryBank
from submodular_gpt5.schemas import SessionEvent, Turn


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).replace("\n", " \n ").split()


class DummySemanticModel:
    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        rows = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            v = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
            v = (v - 127.5) / 127.5
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm:
                    v = v / norm
            rows.append(v)
        return np.stack(rows, axis=0)


def make_compressor(monkeypatch):
    monkeypatch.setattr(
        StaticSubmodularMemoryCompressor,
        "_init_semantic_model",
        lambda self: DummySemanticModel(),
    )
    return StaticSubmodularMemoryCompressor(
        tokenizer=TinyTokenizer(),
        token_budget=20,
        lambda_coverage=0.1,
        lambda_diversity=0.1,
        lambda_relevance=0.8,
        semantic_model_name="dummy",
        local_files_only=True,
        encode_batch_size=4,
        relevance_mode="hybrid",
        granular_epsilon=0.1,
    )


def test_static_granular_select_runs(monkeypatch):
    compressor = make_compressor(monkeypatch)
    units = [
        "user: I moved to Seattle in 2022.",
        "assistant: You mentioned Seattle.",
        "user: My favorite cafe is near Pike Place Market.",
        "assistant: That sounds convenient.",
        "user: I later moved to Boston in 2025.",
        "assistant: I will remember the update.",
    ]
    costs = [max(len(TinyTokenizer().encode(x)), 1) for x in units]
    prepared = compressor.prepare_memory(
        units,
        token_costs_override=costs,
        token_budget_override=20,
    )
    memory, info = compressor.compress_precomputed(
        prepared,
        question="Where did I move later?",
        return_info=True,
    )
    assert memory
    assert info["selected_indices"]
    assert sum(costs[i] for i in info["selected_indices"]) <= 20


def test_dynamic_streaming_stage1_runs(monkeypatch):
    compressor = make_compressor(monkeypatch)
    events = [
        SessionEvent(
            session_idx=0,
            session_id="1",
            session_time="2024-01-01",
            turns=[
                Turn("user", "I started a pottery class.", "d1"),
                Turn("assistant", "Pottery can be relaxing.", "d2"),
                Turn("user", "The class is on Saturday mornings.", "d3"),
            ],
        ),
        SessionEvent(
            session_idx=1,
            session_id="2",
            session_time="2024-02-01",
            turns=[
                Turn("user", "I bought a blue bicycle.", "d4"),
                Turn("assistant", "Enjoy the new bicycle.", "d5"),
                Turn("user", "I ride it to the park.", "d6"),
            ],
        ),
    ]
    cfg = MemoryConfig(
        query_budget=20,
        chunk_tokens=14,
        stream_budget=45,
        fact_buffer_budget=20,
        fact_replacement_budget=15,
        encode_batch_size=3,
    )
    bank = StreamingMemoryBank(
        compressor=compressor,
        token_counter=lambda x: len(TinyTokenizer().encode(x)),
        cfg=cfg,
    )
    result = bank.build(events)
    assert result.records
    assert result.embeddings.shape[0] == len(result.records)
    assert result.info["stage1_candidate_token_cost"] <= cfg.stream_budget
