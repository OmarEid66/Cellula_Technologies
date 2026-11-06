"""Retriever Agent tools: query rewriter, semantic search, keyword search, metadata filter, reranker, context selector."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..formatting import format_history
from ..llm import structured
from ..schemas import Evidence, MetadataFilter, RewrittenQuery
from ..vectorstore import KnowledgeBase, tokenize

# ---------------------------------------------------------------- 1. Query Rewriter

_REWRITE_SYSTEM = (
    "You rewrite a user's question into retrieval-friendly search queries for a collection of documents. "
    "Use the chat history to resolve pronouns and references (e.g. 'its disadvantages' -> name the subject). "
    "Expand abbreviations, clarify terminology, and keep every specific name, number, equation or ID the user "
    "mentioned. Return a standalone_query and up to 2 alternative_queries worded differently. "
    "Never answer the question."
)


def query_rewriter(llm, question: str, history: list[dict] | None = None) -> RewrittenQuery:
    try:
        result = structured(llm, RewrittenQuery).invoke(
            [
                SystemMessage(content=_REWRITE_SYSTEM),
                HumanMessage(content=f"Chat history:\n{format_history(history)}\n\nQuestion: {question}"),
            ]
        )
        if isinstance(result, dict):
            result = RewrittenQuery(**result)
        if not result.standalone_query.strip():
            raise ValueError("empty rewrite")
        result.alternative_queries = [q for q in result.alternative_queries if q.strip()][:2]
        return result
    except Exception:
        return RewrittenQuery(standalone_query=question, alternative_queries=[])


# ---------------------------------------------------------------- 4. Metadata Filter

_FILTER_SYSTEM = (
    "Decide whether the user's question restricts the search to part of the document collection. "
    "Set `source` only if the user names one of the available documents (use the exact name from the list). "
    "Set page_min/page_max only for an explicit page or page range ('page 5', 'pages 3-6'). "
    "Set chunk_type='table' only if the user explicitly asks about a table. "
    "If nothing is restricted, leave every field empty."
)
_HINT_WORDS = re.compile(r"\b(page|pages|pp\.?|table|tables|chapter|section)\b", re.IGNORECASE)


def _might_restrict(question: str, sources: list[str]) -> bool:
    q = question.lower()
    if _HINT_WORDS.search(q):
        return True
    for s in sources:
        stem = s.rsplit(".", 1)[0].lower()
        if s.lower() in q or (len(stem) >= 3 and stem in q):
            return True
    return False


def metadata_filter(llm, question: str, available_sources: list[str]) -> MetadataFilter:
    if not available_sources or not _might_restrict(question, available_sources):
        return MetadataFilter()  # skip the LLM call when the question clearly has no scope restriction
    try:
        result = structured(llm, MetadataFilter).invoke(
            [
                SystemMessage(content=_FILTER_SYSTEM),
                HumanMessage(content=f"Available documents: {available_sources}\n\nQuestion: {question}"),
            ]
        )
        if isinstance(result, dict):
            result = MetadataFilter(**result)
    except Exception:
        return MetadataFilter()
    if result.source:  # must be a real document; tolerate case / extension differences
        lowered = {s.lower(): s for s in available_sources}
        key = result.source.lower()
        match = lowered.get(key) or next(
            (orig for low, orig in lowered.items() if key in low or low.rsplit(".", 1)[0] == key), None
        )
        result.source = match
    if result.page_min is not None and result.page_max is not None and result.page_min > result.page_max:
        result.page_min, result.page_max = result.page_max, result.page_min
    return result


# ---------------------------------------------------------------- 2/3. Semantic + Keyword search


def semantic_search(kb: KnowledgeBase, query: str, k: int, flt: MetadataFilter | None = None) -> list[Evidence]:
    return kb.semantic_search(query, k, flt)


def keyword_search(kb: KnowledgeBase, query: str, k: int, flt: MetadataFilter | None = None) -> list[Evidence]:
    return kb.keyword_search(query, k, flt)


def reciprocal_rank_fusion(rank_lists: list[list[Evidence]], k: int = 60) -> list[Evidence]:
    """Combine several ranked lists (semantic + keyword, several queries) into one hybrid ranking."""
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}
    for ranking in rank_lists:
        for rank, ev in enumerate(ranking):
            scores[ev.chunk_id] = scores.get(ev.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if ev.chunk_id not in best:
                best[ev.chunk_id] = ev
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    top = scores[ordered[0]] if ordered else 1.0
    fused = []
    for cid in ordered:
        item = best[cid].model_copy()
        item.score = scores[cid] / top
        fused.append(item)
    return fused


# ---------------------------------------------------------------- 5. Reranker


class _RerankScore(BaseModel):
    index: int = Field(description="Index of the passage as shown in the prompt.")
    score: int = Field(description="Usefulness for answering the query, 0 (irrelevant) to 10 (directly answers).")


class _RerankResult(BaseModel):
    scores: list[_RerankScore]


_RERANK_SYSTEM = (
    "You are a relevance ranker. Given a query and numbered passages, give each passage an integer score from "
    "0 to 10 for how useful it is to answer the query. Judge the actual content, not keyword overlap. "
    "Score every passage."
)


def reranker(llm, query: str, candidates: list[Evidence], top_n: int) -> list[Evidence]:
    """Second stage of retrieval: re-score the fused candidates with the LLM and keep the best."""
    if len(candidates) <= 1:
        return candidates[:top_n]
    passages = "\n\n".join(f"[{i}] ({c.source}, {c.chunk_type}) {c.content[:700]}" for i, c in enumerate(candidates))
    try:
        result = structured(llm, _RerankResult).invoke(
            [SystemMessage(content=_RERANK_SYSTEM), HumanMessage(content=f"Query: {query}\n\nPassages:\n{passages}")]
        )
        if isinstance(result, dict):
            result = _RerankResult(**result)
        scored = {s.index: max(0, min(10, s.score)) for s in result.scores if 0 <= s.index < len(candidates)}
        if not scored:
            raise ValueError("no usable scores")
    except Exception:
        return candidates[:top_n]  # fall back to the hybrid-search order
    order = sorted(
        range(len(candidates)),
        key=lambda i: (scored.get(i, 0), -i),  # ties keep the fused order
        reverse=True,
    )
    reranked = []
    for i in order[:top_n]:
        item = candidates[i].model_copy()
        item.score = scored.get(i, 0) / 10.0
        reranked.append(item)
    return reranked


# ---------------------------------------------------------------- 6. Context Selector


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def context_selector(
    candidates: list[Evidence],
    max_chunks: int = 6,
    max_chars: int = 12000,
    lambda_relevance: float = 0.7,
    duplicate_threshold: float = 0.8,
) -> list[Evidence]:
    """Pick a compact, diverse set: relevance first, then penalise near-duplicates and favour unseen documents."""
    if not candidates:
        return []
    token_sets = [set(tokenize(c.content)) for c in candidates]
    remaining = list(range(len(candidates)))
    selected: list[int] = []
    seen_sources: set[str] = set()
    used_chars = 0
    while remaining and len(selected) < max_chunks:
        best_i, best_val = None, float("-inf")
        for i in remaining:
            redundancy = max((_jaccard(token_sets[i], token_sets[j]) for j in selected), default=0.0)
            if redundancy >= duplicate_threshold:
                continue
            coverage_bonus = 0.05 if candidates[i].source not in seen_sources else 0.0
            val = lambda_relevance * candidates[i].score - (1 - lambda_relevance) * redundancy + coverage_bonus
            if val > best_val:
                best_i, best_val = i, val
        if best_i is None:
            break
        size = len(candidates[best_i].content)
        if selected and used_chars + size > max_chars:
            remaining.remove(best_i)
            continue
        selected.append(best_i)
        remaining.remove(best_i)
        seen_sources.add(candidates[best_i].source)
        used_chars += size
    return [candidates[i] for i in selected]  # metadata (source, page) is preserved for citations
