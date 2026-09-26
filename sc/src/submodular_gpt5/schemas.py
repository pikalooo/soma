from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Turn:
    role: str
    content: str
    turn_id: str = ""
    has_answer: bool = False

    @property
    def line(self) -> str:
        role = self.role.strip() or "unknown"
        return f"{role}: {self.content}".strip()


@dataclass(slots=True)
class SessionEvent:
    session_idx: int
    session_id: str
    session_time: str
    turns: list[Turn]


@dataclass(slots=True)
class QACase:
    qa_id: str
    question: str
    answer: str = ""
    question_date: str = ""
    category: str = ""
    answer_session_ids: list[str] = field(default_factory=list)
    evidence_turn_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryGroup:
    group_id: str
    events: list[SessionEvent]
    qas: list[QACase]
    extra: dict[str, Any] = field(default_factory=dict)
