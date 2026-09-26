from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from ..schemas import MemoryGroup, QACase, SessionEvent, Turn


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def _evidence_ids(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, (str, int, float)):
        s = str(value).strip()
        return [s] if s else []
    if isinstance(value, dict):
        for key in ("dia_id", "turn_id", "id", "evidence"):
            if key in value:
                out.extend(_evidence_ids(value[key]))
        return out
    if isinstance(value, (list, tuple, set)):
        for x in value:
            out.extend(_evidence_ids(x))
        return out
    return out


def _session_number(key: str) -> int | None:
    m = re.fullmatch(r"session_(\d+)", str(key))
    return int(m.group(1)) if m else None


class LoCoMoAdapter(DatasetAdapter):
    name = "locomo"

    def load(self, path: str | Path) -> list[MemoryGroup]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("LoCoMo must be a top-level JSON list.")

        groups: list[MemoryGroup] = []
        for s_idx, raw in enumerate(data):
            if not isinstance(raw, dict):
                raise ValueError(f"LoCoMo sample {s_idx} is not a dict.")
            sample_id = str(raw.get("sample_id") or raw.get("id") or f"sample_{s_idx}")
            conv = raw.get("conversation")
            if not isinstance(conv, dict):
                raise ValueError(f"LoCoMo sample {sample_id} has no conversation dict.")

            session_numbers = sorted(
                n for key in conv for n in [_session_number(key)] if n is not None
            )
            events: list[SessionEvent] = []
            for order, n in enumerate(session_numbers):
                turns_raw = conv.get(f"session_{n}") or []
                stime = str(conv.get(f"session_{n}_date_time", "Unknown date"))
                turns: list[Turn] = []
                for turn in turns_raw:
                    if isinstance(turn, dict):
                        role = str(turn.get("speaker") or turn.get("role") or "unknown").strip()
                        content = turn.get("text", turn.get("content", ""))
                        content = "" if content is None else str(content).strip()
                        cap = str(turn.get("blip_caption", "") or "").strip()
                        if cap and cap.lower() not in {"none", "nan"}:
                            content = (
                                f"{content} [Image caption: {cap}]"
                                if content else f"[Image caption: {cap}]"
                            )
                        tid = str(
                            turn.get("dia_id")
                            or turn.get("turn_id")
                            or turn.get("id")
                            or ""
                        ).strip()
                    else:
                        role, content, tid = "unknown", str(turn), ""
                    turns.append(Turn(role=role, content=content, turn_id=tid))
                events.append(
                    SessionEvent(
                        session_idx=order,
                        session_id=str(n),
                        session_time=stime,
                        turns=turns,
                    )
                )

            qas: list[QACase] = []
            qa_rows = raw.get("qa") or []
            if not isinstance(qa_rows, list):
                raise ValueError(f"LoCoMo sample {sample_id} qa is not a list.")
            for q_idx, qraw in enumerate(qa_rows):
                q = dict(qraw) if isinstance(qraw, dict) else {"question": str(qraw)}
                cat_raw = q.get("category", "")
                try:
                    cat_num = int(cat_raw)
                except Exception:
                    cat_num = None
                category = CATEGORY_NAMES.get(cat_num, str(cat_raw or ""))
                qas.append(
                    QACase(
                        qa_id=f"{sample_id}:{q_idx}",
                        question=str(q.get("question", "") or ""),
                        answer=str(q.get("answer", "") or ""),
                        category=category,
                        evidence_turn_ids=_evidence_ids(q.get("evidence")),
                        extra={
                            "sample_id": sample_id,
                            "qa_idx": q_idx,
                            "category_raw": cat_raw,
                            "evidence": q.get("evidence", []),
                            "adversarial_answer": q.get("adversarial_answer", ""),
                        },
                    )
                )
            groups.append(
                MemoryGroup(
                    group_id=sample_id,
                    events=events,
                    qas=qas,
                    extra={"source_index": s_idx},
                )
            )
        return groups

    def recall(self, selected_records: list[dict[str, Any]], qa: QACase) -> dict[str, Any]:
        selected_turn_ids: set[str] = set()
        for record in selected_records:
            for turn in record.get("turns", []) or []:
                tid = str(turn.get("turn_id", "") or "")
                if tid:
                    selected_turn_ids.add(tid)

        evidence = {str(x) for x in qa.evidence_turn_ids if str(x)}
        covered = sorted(evidence & selected_turn_ids)
        return {
            "selected_turn_ids": sorted(selected_turn_ids),
            "evidence_turn_ids": sorted(evidence),
            "covered_evidence_turn_ids": covered,
            "evidence_recall": len(covered) / len(evidence) if evidence else None,
        }

    def prediction_row(self, output: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": output.get("sample_id"),
            "qa_idx": output.get("qa_idx"),
            "question": output.get("question", ""),
            "answer": output.get("answer", ""),
            "adversarial_answer": output.get("adversarial_answer", ""),
            "category": output.get("category", ""),
            "evidence": output.get("evidence", []),
            "semantic_prediction": output.get("semantic_prediction", ""),
            "prediction": output.get("prediction", ""),
            "prediction_context": output.get("selected_turn_ids", []),
        }
