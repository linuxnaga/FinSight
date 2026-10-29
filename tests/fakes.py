"""A scripted stand-in for the course endpoint, so the agent loop can be
tested offline and deterministically."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Callable


class FakeClient:
    """`responder(system, user, call_index)` returns the model's text."""

    def __init__(self, responder: Callable[[str, str, int], str]):
        self.responder = responder
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        msgs = kw["messages"]
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        user = msgs[-1]["content"]
        self.calls.append(kw)
        text = self.responder(system, user, len(self.calls))
        if not isinstance(text, str):
            text = json.dumps(text)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def by_role(router=None, planner=None, answer=None, critic=None):
    """Build a responder that dispatches on which prompt is calling."""
    def respond(system: str, user: str, i: int):
        if system.startswith("Classify a question"):
            return router(user) if callable(router) else router
        if system.startswith("You plan tool calls"):
            return planner(user) if callable(planner) else planner
        if system.startswith("You are a financial-filings analyst"):
            return answer(user) if callable(answer) else answer
        if system.startswith("You check a draft answer"):
            return critic(user) if callable(critic) else critic
        raise AssertionError(f"unexpected prompt: {system[:60]}")
    return respond
