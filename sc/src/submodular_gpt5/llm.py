from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import os
import re
import time
from typing import Any

from openai import AsyncOpenAI

from .prompts import (
    GENERATION_SYSTEM,
    DISTILL_SYSTEM,
    build_answer_input,
    build_distill_input,
)


def _clean_text(text: str | None) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    return text.strip()


@dataclass(slots=True)
class LLMResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model: str
    latency_seconds: float
    error: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPT5Client:
    """OpenAI Responses API client shared by answer and distillation stages."""

    def __init__(
        self,
        model: str = "gpt-5",
        concurrency: int = 6,
        reasoning_effort: str = "minimal",
        text_verbosity: str = "low",
        answer_max_output_tokens: int = 2048,
        distill_max_output_tokens: int = 1024,
        api_key: str | None = None,
    ):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Put it in the environment; never hard-code it."
            )
        self.model = str(model)
        self.client = AsyncOpenAI(api_key=key)
        self.semaphore = asyncio.Semaphore(max(int(concurrency), 1))
        self.reasoning_effort = str(reasoning_effort)
        self.text_verbosity = str(text_verbosity)
        self.answer_max_output_tokens = int(answer_max_output_tokens)
        self.distill_max_output_tokens = int(distill_max_output_tokens)

    async def _request(
        self,
        *,
        instructions: str,
        user_input: str,
        max_output_tokens: int,
        retry: int = 3,
    ) -> LLMResult:
        async with self.semaphore:
            for attempt in range(max(int(retry), 1)):
                t0 = time.perf_counter()
                try:
                    response = await self.client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=user_input,
                        reasoning={"effort": self.reasoning_effort},
                        text={"verbosity": self.text_verbosity},
                        max_output_tokens=int(max_output_tokens),
                        store=False,
                    )
                    latency = time.perf_counter() - t0
                    usage = getattr(response, "usage", None)
                    text = _clean_text(getattr(response, "output_text", ""))

                    status = getattr(response, "status", None)
                    if not text:
                        incomplete = getattr(response, "incomplete_details", None)
                        reason = getattr(incomplete, "reason", None) if incomplete else None
                        raise RuntimeError(
                            f"OpenAI returned empty output (status={status!r}, reason={reason!r})."
                        )

                    return LLMResult(
                        text=text,
                        input_tokens=(
                            int(getattr(usage, "input_tokens"))
                            if usage is not None
                            and getattr(usage, "input_tokens", None) is not None
                            else None
                        ),
                        output_tokens=(
                            int(getattr(usage, "output_tokens"))
                            if usage is not None
                            and getattr(usage, "output_tokens", None) is not None
                            else None
                        ),
                        total_tokens=(
                            int(getattr(usage, "total_tokens"))
                            if usage is not None
                            and getattr(usage, "total_tokens", None) is not None
                            else None
                        ),
                        model=str(getattr(response, "model", self.model)),
                        latency_seconds=latency,
                        error=None,
                        status=str(status) if status is not None else None,
                    )
                except Exception as exc:
                    if attempt + 1 >= max(int(retry), 1):
                        return LLMResult(
                            text=f"[ERROR] {exc!r}",
                            input_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            model=self.model,
                            latency_seconds=time.perf_counter() - t0,
                            error=repr(exc),
                            status="error",
                        )
                    await asyncio.sleep(2 ** (attempt + 1))

        raise RuntimeError("unreachable")

    async def answer(
        self,
        memory: str,
        question: str,
        question_date: str = "",
    ) -> LLMResult:
        return await self._request(
            instructions=GENERATION_SYSTEM,
            user_input=build_answer_input(memory, question, question_date),
            max_output_tokens=self.answer_max_output_tokens,
        )

    async def distill(
        self,
        question: str,
        semantic_answer: str,
    ) -> LLMResult:
        semantic_answer = _clean_text(semantic_answer)
        if semantic_answer.startswith("[ERROR]"):
            return LLMResult(
                text=semantic_answer,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self.model,
                latency_seconds=0.0,
                error="Stage-A failed; Stage-B skipped.",
                status="skipped",
            )
        if semantic_answer == "No information available.":
            return LLMResult(
                text=semantic_answer,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model=self.model,
                latency_seconds=0.0,
                error=None,
                status="skipped",
            )
        result = await self._request(
            instructions=DISTILL_SYSTEM,
            user_input=build_distill_input(question, semantic_answer),
            max_output_tokens=self.distill_max_output_tokens,
        )
        if result.error:
            # Preserve the semantically richer answer if distillation fails.
            return LLMResult(
                text=semantic_answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                model=result.model,
                latency_seconds=result.latency_seconds,
                error=f"Stage-B failed; used Stage-A fallback: {result.error}",
                status=result.status,
            )
        return result
