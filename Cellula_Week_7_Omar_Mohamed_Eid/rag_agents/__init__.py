"""Multi-agent RAG: Retriever Agent + Analyst Agent + Answer Agent, orchestrated with LangGraph."""


def __getattr__(name):  # lazy import keeps `import rag_agents` cheap
    if name in {"RagSystem", "AskResult"}:
        from . import system

        return getattr(system, name)
    if name == "Settings":
        from .config import Settings

        return Settings
    raise AttributeError(name)


__all__ = ["RagSystem", "AskResult", "Settings"]
