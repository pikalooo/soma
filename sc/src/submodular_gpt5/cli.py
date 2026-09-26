from __future__ import annotations

import argparse
import asyncio
import json

from .config import AppConfig, preset_config
from .datasets import get_adapter
from .pipeline import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="submodular-gpt5",
        description=(
            "Unified LongMemEval/LoCoMo submodular-memory runner using OpenAI GPT-5."
        ),
    )
    p.add_argument("--dataset", choices=["longmemeval", "locomo"], required=True)
    p.add_argument("--mode", choices=["static", "dynamic"], required=True)
    p.add_argument("--data", required=True, dest="data_path")
    p.add_argument("--output", required=True, dest="output_path")
    p.add_argument("--predictions", dest="predictions_path")

    p.add_argument("--model", default="gpt-5", dest="openai_model")
    p.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    p.add_argument("--embedding-device")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--query-budget", type=int)
    p.add_argument("--stream-budget", type=int)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--no-distill", action="store_true")
    return p


async def _amain(args: argparse.Namespace) -> None:
    memory = preset_config(args.dataset, args.mode)
    if args.query_budget is not None:
        memory.query_budget = args.query_budget
    if args.stream_budget is not None:
        memory.stream_budget = args.stream_budget
    if args.chunk_tokens is not None:
        memory.chunk_tokens = args.chunk_tokens

    app = AppConfig(
        dataset=args.dataset,
        mode=args.mode,
        data_path=args.data_path,
        output_path=args.output_path,
        predictions_path=args.predictions_path,
        openai_model=args.openai_model,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        local_files_only=args.local_files_only,
        concurrency=args.concurrency,
        distill=not args.no_distill,
    )
    runner = ExperimentRunner(app, memory, get_adapter(args.dataset))
    _, summary = await runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
