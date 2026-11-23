"""LLM and embedding factories (ChatOpenAI / OpenAIEmbeddings from LangChain)."""
from __future__ import annotations

from .config import Settings


def get_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.chat_model, temperature=0)


def get_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model)


def structured(llm, schema):
    """Structured output via function calling (tolerates optional fields with defaults)."""
    return llm.with_structured_output(schema, method="function_calling")
