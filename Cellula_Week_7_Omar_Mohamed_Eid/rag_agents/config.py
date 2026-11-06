"""Central configuration. Every value can be overridden with an environment variable (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class Settings:
    # --- models (ChatOpenAI / OpenAIEmbeddings from LangChain) ---
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- vector database (ChromaDB) ---
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "documents"))

    # --- document pipeline ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    max_table_chars: int = field(default_factory=lambda: _env_int("MAX_TABLE_CHARS", 3500))

    # --- retriever agent ---
    top_k_semantic: int = field(default_factory=lambda: _env_int("TOP_K_SEMANTIC", 20))
    top_k_keyword: int = field(default_factory=lambda: _env_int("TOP_K_KEYWORD", 20))
    rerank_candidates: int = field(default_factory=lambda: _env_int("RERANK_CANDIDATES", 20))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 12))
    context_chunks: int = field(default_factory=lambda: _env_int("CONTEXT_CHUNKS", 6))
    context_max_chars: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_CHARS", 12000))

    # --- analyst agent / feedback loop ---
    max_feedback_loops: int = field(default_factory=lambda: _env_int("MAX_FEEDBACK_LOOPS", 2))
    max_analyst_tool_steps: int = field(default_factory=lambda: _env_int("MAX_ANALYST_TOOL_STEPS", 6))
