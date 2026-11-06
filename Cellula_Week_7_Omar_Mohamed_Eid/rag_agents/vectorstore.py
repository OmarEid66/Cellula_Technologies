"""VECTOR DB: ChromaDB wrapper offering semantic search, keyword (BM25) search and metadata filtering."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .schemas import Evidence, MetadataFilter

_TOKEN = re.compile(r"\w+(?:[-.]\w+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens; hyphenated / dotted terms (e.g. 'bert-base') are kept whole AND split."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.append(tok)
        if "-" in tok or "." in tok:
            tokens.extend(t for t in re.split(r"[-.]", tok) if t)
    return tokens


def to_chroma_where(flt: MetadataFilter | None) -> dict[str, Any] | None:
    if flt is None or flt.is_empty():
        return None
    clauses: list[dict[str, Any]] = []
    if flt.source:
        clauses.append({"source": flt.source})
    if flt.page_min is not None:
        clauses.append({"page": {"$gte": flt.page_min}})
    if flt.page_max is not None:
        clauses.append({"page": {"$lte": flt.page_max}})
    if flt.chunk_type:
        clauses.append({"chunk_type": flt.chunk_type})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _matches(meta: dict[str, Any], flt: MetadataFilter | None) -> bool:
    if flt is None or flt.is_empty():
        return True
    if flt.source and meta.get("source") != flt.source:
        return False
    if flt.chunk_type and meta.get("chunk_type") != flt.chunk_type:
        return False
    page = meta.get("page")
    if flt.page_min is not None and (page is None or page < flt.page_min):
        return False
    if flt.page_max is not None and (page is None or page > flt.page_max):
        return False
    return True


def _to_evidence(doc_id: str, text: str, meta: dict[str, Any], score: float) -> Evidence:
    return Evidence(
        chunk_id=meta.get("chunk_id", doc_id),
        source=meta.get("source", "unknown"),
        page=meta.get("page"),
        section=meta.get("section"),
        chunk_type=meta.get("chunk_type", "text"),
        content=text,
        score=float(score),
    )


class _BM25:
    """Okapi BM25 with the always-positive (Lucene-style) IDF so tiny collections still rank."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(texts)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q_terms = set(tokenize(query))
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class KnowledgeBase:
    def __init__(self, embeddings, persist_dir: str, collection_name: str = "documents"):
        self._embeddings = embeddings
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self.store = self._make_store()
        self._bm25: _BM25 | None = None
        self._bm25_items: list[tuple[str, str, dict[str, Any]]] = []

    def _make_store(self) -> Chroma:
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---- writing ----
    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        ids = [d.metadata["chunk_id"] for d in docs]
        self.store.add_documents(docs, ids=ids)
        self._bm25 = None

    def delete_source(self, source: str) -> None:
        self.store.delete(where={"source": source})
        self._bm25 = None

    def reset(self) -> None:
        self.store.delete_collection()
        self.store = self._make_store()
        self._bm25 = None

    # ---- reading ----
    def count(self) -> int:
        return self.store._collection.count()

    def get_all(self, where: dict[str, Any] | None = None) -> list[tuple[str, str, dict[str, Any]]]:
        res = self.store.get(where=where, include=["documents", "metadatas"])
        return list(zip(res["ids"], res["documents"], res["metadatas"]))

    def list_sources(self) -> dict[str, dict[str, int]]:
        sources: dict[str, dict[str, int]] = {}
        for _id, _text, meta in self.get_all():
            info = sources.setdefault(meta.get("source", "unknown"), {"chunks": 0, "tables": 0})
            info["chunks"] += 1
            if meta.get("chunk_type") == "table":
                info["tables"] += 1
        return sources

    # ---- searching ----
    def semantic_search(self, query: str, k: int, flt: MetadataFilter | None = None) -> list[Evidence]:
        if self.count() == 0:
            return []
        results = self.store.similarity_search_with_score(query, k=k, filter=to_chroma_where(flt))
        evidence = []
        for doc, distance in results:
            # cosine distance -> similarity in [0, 1]
            evidence.append(_to_evidence(doc.metadata.get("chunk_id", ""), doc.page_content, doc.metadata, 1.0 - float(distance)))
        return evidence

    def keyword_search(self, query: str, k: int, flt: MetadataFilter | None = None) -> list[Evidence]:
        if self._bm25 is None:
            self._bm25_items = self.get_all()
            self._bm25 = _BM25([text for _id, text, _meta in self._bm25_items])
        scores = self._bm25.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = scores[ranked[0]] if ranked and scores[ranked[0]] > 0 else 0.0
        out: list[Evidence] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            doc_id, text, meta = self._bm25_items[i]
            if not _matches(meta, flt):
                continue
            out.append(_to_evidence(doc_id, text, meta, scores[i] / top if top else 0.0))
            if len(out) >= k:
                break
        return out
