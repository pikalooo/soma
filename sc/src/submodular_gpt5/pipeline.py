from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import AppConfig, MemoryConfig
from .datasets.base import DatasetAdapter
from .llm import GPT5Client, LLMResult
from .memory import StaticSubmodularMemoryCompressor, StreamingMemoryBank
from .memory_utils import (
    build_chunks,
    chunk_record_dedupe_key,
    render_chunk_records,
    events_until_question,
)
from .schemas import MemoryGroup, QACase
from .tokenization import GPTTokenizer


class ExperimentRunner:
    def __init__(
        self,
        app: AppConfig,
        memory: MemoryConfig,
        adapter: DatasetAdapter,
    ):
        self.app = app
        self.memory_cfg = memory
        self.adapter = adapter

        random.seed(app.seed)
        np.random.seed(app.seed)

        self.tokenizer = GPTTokenizer(app.openai_model)
        self.compressor = StaticSubmodularMemoryCompressor(
            tokenizer=self.tokenizer,
            token_budget=memory.query_budget,
            lambda_coverage=memory.lambda_coverage,
            lambda_diversity=memory.lambda_diversity,
            lambda_relevance=memory.lambda_relevance,
            lambda_fact_query_bonus=(
                memory.fact_query_bonus if app.mode == "dynamic" else 0.0
            ),
            semantic_model_name=app.embedding_model,
            semantic_device=app.embedding_device,
            local_files_only=app.local_files_only,
            encode_batch_size=memory.encode_batch_size,
            unit_embedding_prefix="passage: ",
            query_embedding_prefix="query: ",
            relevance_mode=memory.relevance_mode,
            hybrid_dense_weight=memory.dense_weight,
            hybrid_lexical_weight=memory.lexical_weight,
            hybrid_entity_weight=memory.entity_weight,
            hybrid_time_weight=memory.time_weight,
            bm25_k1=memory.bm25_k1,
            bm25_b=memory.bm25_b,
            granular_epsilon=memory.granular_epsilon,
        )
        self.llm = GPT5Client(
            model=app.openai_model,
            concurrency=app.concurrency,
            reasoning_effort=app.reasoning_effort,
            text_verbosity=app.text_verbosity,
            answer_max_output_tokens=app.answer_max_output_tokens,
            distill_max_output_tokens=app.distill_max_output_tokens,
        )

    def _count(self, text: str) -> int:
        return self.tokenizer.count(text)

    def _static_bank(self, group: MemoryGroup) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        records = build_chunks(
            group.events,
            token_counter=self._count,
            chunk_tokens=self.memory_cfg.chunk_tokens,
            safety_margin=self.memory_cfg.token_safety_margin_per_chunk,
            raise_on_oversized_turn=self.memory_cfg.raise_on_oversized_turn,
        )
        embed_units = [r["embed_text"] for r in records]
        costs = [max(int(r["stream_token_cost"]), 1) for r in records]
        prepared = self.compressor.prepare_memory(
            embed_units,
            token_costs_override=costs,
            token_budget_override=self.memory_cfg.query_budget,
        )
        prepared["fact_scores"] = np.zeros(len(records), dtype=np.float32)
        return records, prepared

    def _prepare_reused_stage1(
        self,
        records: list[dict[str, Any]],
        embeddings: np.ndarray,
        facility_sim: np.ndarray | None,
        dpp_kernel: np.ndarray | None,
    ) -> dict[str, Any]:
        units = [
            str(r.get("embed_text") or r.get("rendered_text") or "")
            for r in records
        ]
        costs = [max(int(r.get("stream_token_cost", 1)), 1) for r in records]
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(records):
            raise ValueError("Stage-1 record/embedding mismatch.")

        if emb.size:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
                raise ValueError("Stage-1 embeddings contain invalid rows.")
            emb = (emb / norms).astype(np.float32, copy=False)

        origin_tokens = sum(costs)
        prepared = {
            "units": units,
            "token_costs": costs,
            "origin_tokens": origin_tokens,
            "text_origin_tokens": self._count("\n".join(units)) if units else 0,
            "total_units": len(units),
            "token_budget": int(self.memory_cfg.query_budget),
            "default_token_budget": int(self.compressor.token_budget),
            "costs_overridden": True,
            "embeddings": None,
            "facility_sim": None,
            "dpp_kernel": None,
            "semantic_model": self.compressor.semantic_model_name,
            "diversity_method": "dpp",
            "objective_relevance_mode": "raw_cost_weighted_modular",
            "relevance_mode": self.compressor.relevance_mode,
            "embedding_source": "reused_stage1_stream_embedding",
            "kernel_source": None,
            "fact_scores": np.asarray(
                [
                    float(np.clip(r.get("fact_protector_score", 0.0), 0.0, 1.0))
                    for r in records
                ],
                dtype=np.float32,
            ),
        }

        if not units or origin_tokens <= int(self.memory_cfg.query_budget):
            return prepared

        n = len(units)
        matrix_reuse_ok = (
            facility_sim is not None
            and dpp_kernel is not None
            and np.asarray(facility_sim).shape == (n, n)
            and np.asarray(dpp_kernel).shape == (n, n)
        )
        if matrix_reuse_ok:
            prepared["facility_sim"] = np.asarray(facility_sim, dtype=np.float32)
            prepared["dpp_kernel"] = np.asarray(dpp_kernel, dtype=np.float64)
            prepared["kernel_source"] = "reused_stage1_compacted_gram"
        else:
            emb64 = emb.astype(np.float64, copy=False)
            kernel = emb64 @ emb64.T
            kernel = 0.5 * (kernel + kernel.T)
            np.fill_diagonal(kernel, 1.0)
            facility = np.maximum(kernel, 0.0).astype(np.float32, copy=False)
            np.fill_diagonal(facility, 1.0)
            prepared["facility_sim"] = facility
            prepared["dpp_kernel"] = kernel
            prepared["kernel_source"] = "computed_in_stage2"
        prepared["embeddings"] = emb
        return prepared

    def _select_from_prepared(
        self,
        records: list[dict[str, Any]],
        prepared: dict[str, Any],
        qa: QACase,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        _, info = self.compressor.compress_precomputed(
            prepared_memory=prepared,
            question=qa.question,
            return_info=True,
        )
        if prepared["origin_tokens"] <= int(self.memory_cfg.query_budget):
            selected_indices = list(range(len(records)))
        else:
            selected_indices = sorted(
                {
                    int(i)
                    for i in info.get("selected_indices", [])
                    if 0 <= int(i) < len(records)
                }
            )

        selected_records = [records[i] for i in selected_indices]
        # Defensive dedupe while preserving optimizer-selected record order.
        seen, deduped = set(), []
        for record in selected_records:
            key = chunk_record_dedupe_key(record)
            if key not in seen:
                seen.add(key)
                deduped.append(record)
        selected_records = deduped

        memory = render_chunk_records(selected_records)
        fallback_pruning = False
        while selected_records and self._count(memory) > int(self.memory_cfg.query_budget):
            fallback_pruning = True
            selected_records.pop()
            memory = render_chunk_records(selected_records)

        info = dict(info)
        info.update({
            "selected_indices_final": selected_indices[:len(selected_records)],
            "rendered_memory_tokens": self._count(memory),
            "query_memory_budget": int(self.memory_cfg.query_budget),
            "budget_enforced_by_render_pruning": fallback_pruning,
        })
        return memory, selected_records, info

    async def _answer_case(
        self,
        group: MemoryGroup,
        qa: QACase,
        memory: str,
        selected_records: list[dict[str, Any]],
        compression_info: dict[str, Any],
        stage1_info: dict[str, Any] | None = None,
        future_event_count: int = 0,
    ) -> dict[str, Any]:
        stage_a = await self.llm.answer(
            memory=memory,
            question=qa.question,
            question_date=qa.question_date,
        )
        if self.app.distill:
            stage_b = await self.llm.distill(qa.question, stage_a.text)
            prediction = stage_b.text
            distill_applied = stage_b.status != "skipped" and stage_b.error is None
        else:
            stage_b = None
            prediction = stage_a.text
            distill_applied = False

        row: dict[str, Any] = {
            "dataset": self.adapter.name,
            "mode": self.app.mode,
            "group_id": group.group_id,
            "qa_id": qa.qa_id,
            "question": qa.question,
            "answer": qa.answer,
            "question_date": qa.question_date,
            "category": qa.category,
            "prediction": prediction,
            "semantic_prediction": stage_a.text,
            "selected_memory": memory,
            "selected_memory_tokens": self._count(memory),
            "selected_chunks": len(selected_records),
            "compression": compression_info,
            "stage1": stage1_info,
            "future_events_excluded": int(future_event_count),
            "openai_model": stage_a.model,
            "openai_input_tokens": stage_a.input_tokens,
            "openai_output_tokens": stage_a.output_tokens,
            "openai_total_tokens": stage_a.total_tokens,
            "openai_api_latency_seconds": stage_a.latency_seconds,
            "openai_error": stage_a.error,
            "answer_distill_applied": distill_applied,
            "method": (
                "static_granular_select_gpt5"
                if self.app.mode == "static"
                else "dynamic_feature_saturation_dpp_then_granular_select_gpt5"
            ),
        }
        if stage_b is not None:
            row.update({
                "distilled_prediction": stage_b.text,
                "distill_openai_model": stage_b.model,
                "distill_input_tokens": stage_b.input_tokens,
                "distill_output_tokens": stage_b.output_tokens,
                "distill_total_tokens": stage_b.total_tokens,
                "distill_api_latency_seconds": stage_b.latency_seconds,
                "distill_error": stage_b.error,
            })

        row.update(qa.extra)
        row.update(self.adapter.recall(selected_records, qa))
        return row

    async def _run_static_group(self, group: MemoryGroup) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        records, prepared = self._static_bank(group)
        bank_seconds = time.perf_counter() - t0

        prepared_cases = []
        for qa in group.qas:
            q0 = time.perf_counter()
            memory, selected, info = self._select_from_prepared(records, prepared, qa)
            info["memory_bank_build_seconds"] = bank_seconds
            info["query_compression_seconds"] = time.perf_counter() - q0
            prepared_cases.append((qa, memory, selected, info))

        tasks = [
            self._answer_case(group, qa, memory, selected, info)
            for qa, memory, selected, info in prepared_cases
        ]
        return list(await asyncio.gather(*tasks)) if tasks else []

    async def _run_dynamic_group(self, group: MemoryGroup) -> list[dict[str, Any]]:
        events = group.events
        future_count = 0
        # LongMemEval has one QA per group and an explicit question timestamp.
        if self.adapter.name == "longmemeval" and group.qas:
            events, future = events_until_question(events, group.qas[0].question_date)
            future_count = len(future)

        t0 = time.perf_counter()
        bank = StreamingMemoryBank(
            compressor=self.compressor,
            token_counter=self._count,
            cfg=self.memory_cfg,
        )
        stage1 = bank.build(events)
        stage1_seconds = time.perf_counter() - t0
        stage1.info["stage1_build_seconds"] = stage1_seconds
        stage1.info["future_events_excluded"] = future_count

        prepared = self._prepare_reused_stage1(
            stage1.records,
            stage1.embeddings,
            stage1.facility_sim,
            stage1.dpp_kernel,
        )

        prepared_cases = []
        for qa in group.qas:
            q0 = time.perf_counter()
            memory, selected, info = self._select_from_prepared(
                stage1.records,
                prepared,
                qa,
            )
            info.update({
                "query_compression_seconds": time.perf_counter() - q0,
                "stage2_reused_stage1_embeddings": True,
                "stage2_kernel_source": prepared.get("kernel_source"),
            })
            prepared_cases.append((qa, memory, selected, info))

        tasks = [
            self._answer_case(
                group,
                qa,
                memory,
                selected,
                info,
                stage1_info=stage1.info,
                future_event_count=future_count,
            )
            for qa, memory, selected, info in prepared_cases
        ]
        return list(await asyncio.gather(*tasks)) if tasks else []

    def _save(self, outputs: list[dict[str, Any]]) -> None:
        output_path = Path(self.app.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

        pred_path = (
            Path(self.app.predictions_path)
            if self.app.predictions_path
            else output_path.with_suffix(".jsonl")
        )
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in outputs:
                f.write(
                    json.dumps(
                        self.adapter.prediction_row(row),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    async def _run_group_safe(self, group: MemoryGroup) -> list[dict[str, Any]]:
        try:
            if self.app.mode == "static":
                return await self._run_static_group(group)
            if self.app.mode == "dynamic":
                return await self._run_dynamic_group(group)
            raise ValueError("mode must be 'static' or 'dynamic'")
        except Exception as exc:
            rows = []
            for qa in group.qas:
                row = {
                    "dataset": self.adapter.name,
                    "mode": self.app.mode,
                    "group_id": group.group_id,
                    "qa_id": qa.qa_id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "prediction": f"[ERROR] {exc!r}",
                    "error": repr(exc),
                }
                row.update(qa.extra)
                rows.append(row)
            return rows

    async def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        groups = self.adapter.load(self.app.data_path)
        outputs: list[dict[str, Any]] = []
        start = time.perf_counter()

        # Batch groups so LongMemEval (one QA per group) still gets concurrent
        # GPT requests, while the client semaphore remains the global API cap.
        batch_size = max(int(self.app.concurrency), 1)
        progress = tqdm(
            total=len(groups),
            desc=f"{self.adapter.name}/{self.app.mode}",
            dynamic_ncols=True,
        )
        try:
            for batch_start in range(0, len(groups), batch_size):
                batch = groups[batch_start:batch_start + batch_size]
                batch_rows = await asyncio.gather(
                    *(self._run_group_safe(group) for group in batch)
                )
                for rows in batch_rows:
                    outputs.extend(rows)
                self._save(outputs)
                progress.update(len(batch))
        finally:
            progress.close()

        elapsed = time.perf_counter() - start
        summary = {
            "dataset": self.adapter.name,
            "mode": self.app.mode,
            "groups": len(groups),
            "qa_count": len(outputs),
            "errors": sum(
                1
                for x in outputs
                if x.get("error") or x.get("openai_error")
            ),
            "elapsed_seconds": elapsed,
            "openai_model": self.app.openai_model,
            "embedding_model": self.app.embedding_model,
            "query_budget": self.memory_cfg.query_budget,
            "stream_budget": (
                self.memory_cfg.stream_budget
                if self.app.mode == "dynamic" else None
            ),
        }
        self._save(outputs)
        return outputs, summary
