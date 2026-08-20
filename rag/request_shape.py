"""Coerce an OpenAI-shaped request body into what the pipeline expects.

Kept out of server.py on purpose: server.py cannot be imported without dspy,
vLLM and the embedding stack, so anything defined there is only ever exercised
against a *replica* app in the tests -- which is how a crash on
`"content": null` reached the branch that deploys. This module imports nothing
heavier than the standard library, so the real coercion is unit-tested.

The failure being locked down: OpenAI's wire format allows a null content
(tool-call turns, an unset system prompt). `m.get("content", "")` returns None
for an explicit null, and joining that raises TypeError -- which landed outside
the handler's guard and returned a plain-text 500 the caller cannot parse, on
the second message of any thread that keeps history, after the request had
already been counted against the caller's daily cap.
"""
from __future__ import annotations


def normalize_messages(messages) -> list[dict]:
    """Every returned message has a `content` that is a real string.

    Raises TypeError when `messages` is not a list, so the caller can answer
    400 rather than 500.
    """
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise TypeError("messages must be an array")
    clean = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            # OpenAI content parts: [{"type": "text", "text": "..."}, ...]
            content = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        clean.append({**message, "content": content})
    return clean


def last_question(messages) -> str:
    """The text of the final message, or "" for an empty conversation."""
    return messages[-1]["content"] if messages else ""


def history_and_context(messages, turns):
    """(history for the model, earlier text used to inherit a code name)."""
    history = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
    context = "\n".join(m["content"] for m in messages[-turns:-1])
    return history, context
