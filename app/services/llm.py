from collections.abc import Iterator

import ollama

from app.schemas import HistoryTurn
from app.services.prompt_builder import SYSTEM_PROMPT, build_user_message


def stream_answer(
    question: str,
    results: list,
    history: list[HistoryTurn],
    base_url: str,
    model: str,
) -> Iterator[dict]:
    client = ollama.Client(host=base_url)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": turn.role, "content": turn.content} for turn in history]
    messages.append({"role": "user", "content": build_user_message(question, results)})

    for chunk in client.chat(model=model, messages=messages, stream=True):
        message = chunk.message
        if message.thinking:
            yield {"event": "thinking", "text": message.thinking}
            continue
        if message.content:
            yield {"event": "token", "text": message.content}

    yield {"event": "done"}
