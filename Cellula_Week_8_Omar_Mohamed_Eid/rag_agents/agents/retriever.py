"""Retriever Agent: finds relevant evidence in the vector DB. It never answers the question."""
from __future__ import annotations

from ..config import Settings
from ..schemas import MetadataFilter, RetrievalResult
from ..tools import retrieval as tools
from ..vectorstore import KnowledgeBase


class RetrieverAgent:
    def __init__(self, llm, kb: KnowledgeBase, settings: Settings):
        self.llm = llm
        self.kb = kb
        self.settings = settings

    def retrieve(self, question: str, history: list[dict] | None = None) -> RetrievalResult:
        s = self.settings
        trace: list[str] = []

        # 1. Query Rewriter
        rewritten = tools.query_rewriter(self.llm, question, history)
        queries = [rewritten.standalone_query, *rewritten.alternative_queries]
        trace.append(f"Query Rewriter -> {queries}")

        # 4. Metadata Filter
        sources = list(self.kb.list_sources().keys())
        flt = tools.metadata_filter(self.llm, question, sources)
        trace.append(f"Metadata Filter -> {flt.describe()}")

        candidates = self._search(queries, flt, trace)
        if not candidates and not flt.is_empty():
            trace.append("Filter returned nothing -> retrying without the filter")
            flt = MetadataFilter()
            candidates = self._search(queries, flt, trace)

        considered = len(candidates)
        candidates = candidates[: s.rerank_candidates]

        # 5. Reranker
        reranked = tools.reranker(self.llm, rewritten.standalone_query, candidates, s.rerank_top_n)
        trace.append(f"Reranker -> kept {len(reranked)} of {len(candidates)}")

        # 6. Context Selector
        selected = tools.context_selector(reranked, s.context_chunks, s.context_max_chars)
        trace.append(f"Context Selector -> {len(selected)} chunks")

        return RetrievalResult(
            standalone_query=rewritten.standalone_query,
            queries=queries,
            metadata_filter=flt,
            evidence=selected,
            candidates_considered=considered,
            trace=trace,
        )

    def _search(self, queries: list[str], flt: MetadataFilter, trace: list[str]):
        s = self.settings
        rank_lists = []
        for q in queries:
            sem = tools.semantic_search(self.kb, q, s.top_k_semantic, flt)  # 2. Semantic Search
            kw = tools.keyword_search(self.kb, q, s.top_k_keyword, flt)  # 3. Keyword Search
            rank_lists.extend([sem, kw])
        fused = tools.reciprocal_rank_fusion(rank_lists)
        trace.append(f"Semantic + Keyword search -> {len(fused)} unique candidates")
        return fused
