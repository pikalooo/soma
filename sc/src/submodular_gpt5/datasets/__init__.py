from .base import DatasetAdapter
from .longmemeval import LongMemEvalAdapter
from .locomo import LoCoMoAdapter

def get_adapter(name: str) -> DatasetAdapter:
    name = name.lower()
    if name == "longmemeval":
        return LongMemEvalAdapter()
    if name == "locomo":
        return LoCoMoAdapter()
    raise ValueError("dataset must be 'longmemeval' or 'locomo'")

__all__ = ["DatasetAdapter", "LongMemEvalAdapter", "LoCoMoAdapter", "get_adapter"]
