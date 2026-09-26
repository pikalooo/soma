from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..schemas import MemoryGroup, QACase, SessionEvent, Turn


def _normalize_role(role: Any) -> str:
    role = str(role or "").strip()
    low = role.lower()
    if low in {"user", "human"}:
        return "user"
    if low in {"assistant", "ai", "bot"}:
        return "assistant"
    return role or "unknown"


def _session_turns(session: Any) -> list[Any]:
    if isinstance(session, list):
        return session
    if isinstance(session, dict):
        if isinstance(session.get("turns"), list):
            return session["turns"]
        if isinstance(session.get("messages"), list):
            return session["messages"]
        return [session]
    return [session]


def _load_json_or_jsonl(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "test", "validation", "train"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError(f"Unsupported LongMemEval format: {type(obj)}")


class LongMemEvalAdapter(DatasetAdapter):
    name = "longmemeval"

    def load(self, path: str | Path) -> list[MemoryGroup]:
        rows = _load_json_or_jsonl(Path(path))
        groups: list[MemoryGroup] = []

        for idx, raw in enumerate(rows):
            item = dict(raw) if isinstance(raw, dict) else {"question": str(raw)}
            qid = str(
                item.get("question_id")
                or item.get("id")
                or item.get("sample_id")
                or f"question_{idx}"
            )

            sessions = item.get("haystack_sessions") or []
            dates = item.get("haystack_dates") or []
            session_ids = item.get("haystack_session_ids") or []
            events: list[SessionEvent] = []

            for s_idx, session in enumerate(sessions):
                sid = str(session_ids[s_idx] if s_idx < len(session_ids) else s_idx + 1)
                stime = str(dates[s_idx] if s_idx < len(dates) else "Unknown date")
                turns: list[Turn] = []
                for turn in _session_turns(session):
                    if isinstance(turn, dict):
                        role = _normalize_role(turn.get("role", turn.get("speaker", "")))
                        content = turn.get("content", turn.get("text", ""))
                        content = "" if content is None else str(content)
                        if turn.get("blip_caption"):
                            content += f" [Image caption: {turn['blip_caption']}]"
                        tid = str(
                            turn.get("turn_id")
                            or turn.get("dia_id")
                            or turn.get("id")
                            or ""
                        )
                        has_answer = bool(turn.get("has_answer", False))
                    else:
                        role, content, tid, has_answer = "unknown", str(turn), "", False
                    turns.append(Turn(role=role, content=content, turn_id=tid, has_answer=has_answer))

                events.append(
                    SessionEvent(
                        session_idx=s_idx,
                        session_id=sid,
                        session_time=stime,
                        turns=turns,
                    )
                )

            answer_session_ids = [
                str(x) for x in (item.get("answer_session_ids") or [])
            ]
            qa = QACase(
                qa_id=qid,
                question=str(item.get("question", "") or ""),
                answer=str(item.get("answer", "") or ""),
                question_date=str(item.get("question_date", "") or ""),
                category=str(item.get("question_type", item.get("category", "")) or ""),
                answer_session_ids=answer_session_ids,
                extra={"source_index": idx},
            )
            groups.append(MemoryGroup(group_id=qid, events=events, qas=[qa]))

        return groups

    def recall(self, selected_records: list[dict[str, Any]], qa: QACase) -> dict[str, Any]:
        selected_session_ids = {
            str(record.get("session_id", ""))
            for record in selected_records
        }
        selected_answer_turn_count = sum(
            int(record.get("has_answer_turn_count", 0))
            for record in selected_records
        )
        total_answer_turn_count = max(
            (int(record.get("source_has_answer_turn_count", 0))
             for record in selected_records),
            default=0,
        )

        answer_ids = {str(x) for x in qa.answer_session_ids if str(x)}
        covered = sorted(answer_ids & selected_session_ids)
        session_recall = (
            len(covered) / len(answer_ids)
            if answer_ids else None
        )

        # total_answer_turn_count is filled by the pipeline from the full group.
        return {
            "selected_session_ids": sorted(selected_session_ids),
            "covered_answer_session_ids": covered,
            "answer_session_recall": session_recall,
            "selected_answer_turn_count": selected_answer_turn_count,
            "total_answer_turn_count": total_answer_turn_count or None,
        }
