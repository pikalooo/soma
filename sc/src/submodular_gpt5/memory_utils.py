from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Iterable

from .schemas import SessionEvent


def normalize_id(value: Any) -> str:
    return str(value)


def session_header_text(session_id: Any, session_time: Any) -> str:
    return f"[Session {session_id} | Date: {session_time}]"


def parse_datetime_loose(value: Any) -> datetime | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s_clean = re.sub(r"\([A-Za-z]+\)", "", s).strip()
    s_clean = re.sub(r"\s+", " ", s_clean)

    for fmt in (
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%I:%M %p on %d %B, %Y",
        "%I:%M%p on %d %B, %Y",
        "%d %B, %Y",
    ):
        try:
            return datetime.strptime(s_clean, fmt)
        except ValueError:
            pass

    m = re.search(
        r"(\d{4})[/-](\d{1,2})[/-](\d{1,2}).*?(\d{1,2}):(\d{2})",
        s,
    )
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        try:
            return datetime(y, mo, d, h, mi)
        except ValueError:
            return None
    return None


def events_until_question(
    events: list[SessionEvent],
    question_date: str,
) -> tuple[list[SessionEvent], list[SessionEvent]]:
    """Chronologically sort events and remove sessions known to be after the question.

    If the question date is unavailable, all events are retained in chronological
    order. If the question date is known but a session date cannot be parsed, the
    unknown session is conservatively excluded from the observed online prefix.
    """
    qdt = parse_datetime_loose(question_date)
    decorated = [
        (parse_datetime_loose(e.session_time), e)
        for e in events
    ]
    decorated.sort(
        key=lambda x: (
            x[0] is None,
            x[0] or datetime.min,
            x[1].session_idx,
        )
    )
    if qdt is None:
        return [e for _, e in decorated], []

    consumed, future = [], []
    for dt, event in decorated:
        if dt is not None and dt <= qdt:
            consumed.append(event)
        else:
            future.append(event)
    return consumed, future


def event_to_dict(event: SessionEvent) -> dict[str, Any]:
    turns = []
    for turn in event.turns:
        turns.append({
            "role": turn.role,
            "content": turn.content,
            "turn_id": turn.turn_id,
            "has_answer": bool(turn.has_answer),
            "line": turn.line,
        })
    return {
        "session_idx": int(event.session_idx),
        "session_id": str(event.session_id),
        "session_id_norm": str(event.session_id),
        "session_time": str(event.session_time),
        "turns": turns,
    }


def render_turn_records(turn_records: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    last_session_key = None
    ordered = sorted(
        list(turn_records),
        key=lambda x: int(x.get("global_turn_order", 0)),
    )
    for turn in ordered:
        session_key = turn["session_key"]
        if session_key != last_session_key:
            parts.append(
                f"\n{session_header_text(turn['session_id'], turn['session_time'])}"
            )
            last_session_key = session_key
        parts.append(str(turn["line"]))
    return "\n".join(parts).strip()


def chunk_record_dedupe_key(chunk: dict[str, Any]) -> tuple:
    if not isinstance(chunk, dict):
        return ("raw", str(chunk))
    if chunk.get("chunk_order") is not None:
        return ("chunk_order", int(chunk["chunk_order"]))
    turn_keys = []
    for turn in chunk.get("turns", []) or []:
        turn_keys.append((
            turn.get("session_key"),
            normalize_id(turn.get("session_id", "")),
            str(turn.get("session_time", "")),
            turn.get("global_turn_order"),
            str(turn.get("turn_id", "")),
            str(turn.get("line", "")),
        ))
    if turn_keys:
        return ("turns", tuple(turn_keys))
    return (
        "text",
        normalize_id(chunk.get("session_id", "")),
        str(chunk.get("session_time", "")),
        str(chunk.get("rendered_text") or chunk.get("embed_text") or ""),
    )


def dedupe_chunk_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for record in records or []:
        key = chunk_record_dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def render_chunk_records(records: Iterable[dict[str, Any]]) -> str:
    turns = []
    for chunk in dedupe_chunk_records(records):
        turns.extend(chunk.get("turns", []) or [])
    return render_turn_records(turns)


def build_chunks(
    events: list[SessionEvent],
    token_counter,
    chunk_tokens: int,
    safety_margin: int = 0,
    raise_on_oversized_turn: bool = False,
) -> list[dict[str, Any]]:
    """Pack turns into session-local chunks while preserving render order."""
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")

    chunks: list[dict[str, Any]] = []
    next_turn_order = 0
    next_chunk_order = 0

    def make_chunk(event_dict: dict[str, Any], local_turns: list[dict[str, Any]]):
        nonlocal next_chunk_order
        session_id = event_dict["session_id"]
        session_time = event_dict["session_time"]
        session_key = (
            event_dict["session_idx"],
            normalize_id(session_id),
            session_time,
        )
        content_text = "\n".join(t["line"] for t in local_turns)
        embed_text = f"Session {session_id} | Date: {session_time}\n{content_text}"
        rendered_text = render_turn_records(local_turns)
        rendered_token_length = int(token_counter(rendered_text))
        record = {
            "chunk_order": next_chunk_order,
            "session_idx": event_dict["session_idx"],
            "session_id": session_id,
            "session_id_norm": normalize_id(session_id),
            "session_time": session_time,
            "session_key": session_key,
            "rendered_text": rendered_text,
            "embed_text": embed_text,
            "content_text": content_text,
            "content_token_length": int(token_counter(content_text)),
            "rendered_token_length": rendered_token_length,
            "stream_token_cost": max(rendered_token_length + int(safety_margin), 1),
            "turns": [dict(t) for t in local_turns],
            "turn_count": len(local_turns),
            "has_answer_turn_count": sum(
                1 for t in local_turns if bool(t.get("has_answer", False))
            ),
        }
        next_chunk_order += 1
        chunks.append(record)

    for event in events:
        e = event_to_dict(event)
        session_key = (
            e["session_idx"],
            normalize_id(e["session_id"]),
            e["session_time"],
        )
        enriched: list[dict[str, Any]] = []
        for turn in e["turns"]:
            t = dict(turn)
            t.update({
                "global_turn_order": next_turn_order,
                "session_idx": e["session_idx"],
                "session_id": e["session_id"],
                "session_id_norm": e["session_id_norm"],
                "session_time": e["session_time"],
                "session_key": session_key,
            })
            next_turn_order += 1
            enriched.append(t)

        current: list[dict[str, Any]] = []
        for turn in enriched:
            single_tokens = int(token_counter(render_turn_records([turn])))
            if single_tokens > chunk_tokens:
                if current:
                    make_chunk(e, current)
                    current = []
                if raise_on_oversized_turn:
                    raise ValueError(
                        f"One turn exceeds chunk_tokens={chunk_tokens}: {single_tokens}"
                    )
                make_chunk(e, [turn])
                continue

            if not current:
                current = [turn]
                continue

            candidate = current + [turn]
            if int(token_counter(render_turn_records(candidate))) <= chunk_tokens:
                current = candidate
            else:
                make_chunk(e, current)
                current = [turn]

        if current:
            make_chunk(e, current)

    # Provide full-source answer-turn count to selected-record recall diagnostics.
    total_answer_turns = sum(
        int(c.get("has_answer_turn_count", 0)) for c in chunks
    )
    for c in chunks:
        c["source_has_answer_turn_count"] = total_answer_turns
    return chunks
