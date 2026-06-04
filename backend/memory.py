"""
Simple session memory for follow-up questions.

Stores previous questions and answers in memory for the current server session.
No database, no advanced memory frameworks — just a Python list.
"""

# Each entry: {"question": "...", "answer": "..."}
chat_history = []


def add_exchange(question, answer):
    """Save a question-answer pair to memory."""
    chat_history.append({"question": question, "answer": answer})


def get_history_text():
    """
    Format conversation history as text for the LLM prompt.

    Example output:
        Q1: What is deadlock?
        A1: Deadlock is when...
        Q2: Explain it simply.
        A2: ...
    """
    if not chat_history:
        return "No previous conversation."

    lines = []
    for i, entry in enumerate(chat_history, start=1):
        lines.append(f"Q{i}: {entry['question']}")
        lines.append(f"A{i}: {entry['answer']}")
    return "\n".join(lines)


def build_retrieval_query(question):
    """
    Build a search query that works for follow-up questions.

    Follow-ups like "Explain it simply" are too vague on their own for
    semantic search. We prepend the last question so retrieval still finds
    relevant chunks (e.g. chunks about deadlock).
    """
    if not chat_history:
        return question

    last_question = chat_history[-1]["question"]
    return f"{last_question} {question}"


def clear_memory():
    """Clear all stored conversation history."""
    chat_history.clear()
