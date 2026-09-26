from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..schemas import MemoryGroup, QACase


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, path: str | Path) -> list[MemoryGroup]:
        raise NotImplementedError

    @abstractmethod
    def recall(self, selected_records: list[dict[str, Any]], qa: QACase) -> dict[str, Any]:
        raise NotImplementedError

    def prediction_row(self, output: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": output["qa_id"],
            "hypothesis": output["prediction"],
        }
