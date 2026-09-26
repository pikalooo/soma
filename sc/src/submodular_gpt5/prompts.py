GENERATION_SYSTEM = (
    "You are a memory reasoning assistant. Use the provided long-term memory "
    "as the primary evidence for answering the question. Carefully combine "
    "relevant information across sessions and messages when needed. Reason "
    "about chronology, dates, durations, updates, and implicit connections. "
    "If multiple memories conflict, prefer the most recent applicable information "
    "unless the question asks about an earlier state. Resolve relative time "
    "expressions using the timestamp of the relevant memory. You may make logical "
    "inferences strongly supported by the memory, but do not invent user-specific "
    "facts. If the answer cannot be determined from the memory and safe inference, "
    "answer exactly: No information available. Return a concise but complete answer. "
    "Do not explain your reasoning."
)

DISTILL_SYSTEM = (
    "You are a reference-answer distiller for memory question answering. "
    "You receive a question and a semantically correct candidate answer. "
    "Transform it into the minimal sufficient short answer. Prefer deleting "
    "unnecessary words over rewriting. Preserve exact names, entities, locations, "
    "dates, numbers, activities, objects, and required relations. Do not add facts "
    "or use outside knowledge. For who/where/when/number questions, prefer only the "
    "required entity/location/date/number and necessary unit. For list questions, "
    "return all required items separated by commas. For why questions, return the "
    "shortest complete cause phrase. If the candidate is exactly "
    "'No information available.', return exactly that. Output only the final answer."
)


def build_answer_input(memory: str, question: str, question_date: str = "") -> str:
    date_block = (
        f"\n[Question date]\n{question_date}\n"
        if question_date else ""
    )
    return (
        f"[Long-term memory]\n{memory}\n"
        f"{date_block}"
        f"\n[Question]\n{question}\n\n[Answer]\n"
    )


def build_distill_input(question: str, candidate: str) -> str:
    return (
        f"[Question]\n{question}\n\n"
        f"[Candidate answer]\n{candidate}\n\n"
        f"[Final answer]\n"
    )
