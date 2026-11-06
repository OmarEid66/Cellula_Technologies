"""Offline test doubles: a bag-of-words embedding model and a scripted chat model."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings, so semantic search works without an API."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _Structured:
    def __init__(self, llm: "ScriptedLLM", schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages, **_):
        name = self.schema.__name__
        handler = self.llm.structured.get(name)
        self.llm.calls.append(("structured", name))
        if handler is None:
            raise RuntimeError(f"no handler for {name}")
        out = handler(messages)
        return out if isinstance(out, self.schema) else self.schema(**out)


class _Tools:
    def __init__(self, llm: "ScriptedLLM"):
        self.llm = llm

    def invoke(self, messages, **_):
        self.llm.calls.append(("tools", ""))
        return self.llm.tool_step(messages)


class ScriptedLLM:
    """structured: {SchemaName: fn(messages)->dict|model}; tool_step: fn(messages)->AIMessage; text: fn(messages)->str."""

    def __init__(
        self,
        structured: dict[str, Callable[[list], Any]] | None = None,
        tool_step: Callable[[list], AIMessage] | None = None,
        text: Callable[[list], str] | None = None,
    ):
        self.structured = structured or {}
        self.tool_step = tool_step or (lambda m: AIMessage(content="done"))
        self.text = text or (lambda m: "text reply")
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema, **_):
        return _Structured(self, schema)

    def bind_tools(self, tools, **_):
        return _Tools(self)

    def invoke(self, messages, **_):
        self.calls.append(("text", ""))
        return AIMessage(content=self.text(messages))


def last_human(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return m.content
    return ""
