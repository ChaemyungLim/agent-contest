"""Hybrid 검색 (Dense + BM25) + RRF + Reranker.

- Dense: ChromaDB 임베디드 PersistentClient (collection 2개: official, history)
- Sparse (BM25): rank-bm25 in-memory 인덱스 (pickle 영속화)
- RRF로 통합 후 cross-encoder reranker로 최종 정렬
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi

from app.config import resolve_path, settings
from app.retrieval.embedder import get_embedder
from app.retrieval.reranker import get_reranker
from app.retrieval.text import parse_date, tokenize
from app.schemas import Hit, SourceType


__all__ = [
    "BM25Index",
    "Searcher",
    "bm25_index_path",
    "build_bm25_index",
    "get_chroma_client",
    "get_searcher",
    "reset_searcher",
]


def get_chroma_client() -> chromadb.api.ClientAPI:
    persist_dir = resolve_path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def bm25_index_path() -> Path:
    return resolve_path(settings.chroma_persist_dir) / "bm25_index.pkl"


@dataclass
class BM25Index:
    bm25: BM25Okapi
    records: list[Hit]  # 인덱스 i의 토큰 ↔ records[i]

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        n = len(scores)
        k = min(top_k, n)
        # argpartition로 top-k만 추려서 정렬 (전체 O(N log N) 회피)
        top_idx = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], dtype=int)
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class Searcher:
    def __init__(self) -> None:
        self.chroma = get_chroma_client()
        self.embedder = get_embedder()
        self.reranker = get_reranker()

        bm25_path = bm25_index_path()
        if not bm25_path.exists():
            raise RuntimeError(
                f"BM25 인덱스가 없습니다: {bm25_path}. "
                "indexer.build_index를 먼저 실행하세요."
            )
        self.bm25 = BM25Index.load(bm25_path)

        # SourceType → (collection_name, top_k) — 분기 1곳으로 통합
        self._source_meta: dict[SourceType, tuple[str, int]] = {
            "official": (settings.collection_official, settings.top_k_official),
            "history": (settings.collection_history, settings.top_k_history),
        }

        self.collections: dict[str, chromadb.Collection] = {}
        for collection_name, _ in self._source_meta.values():
            try:
                self.collections[collection_name] = self.chroma.get_collection(collection_name)
            except NotFoundError:
                # --skip-history 등으로 인덱싱되지 않은 collection은 건너뜀
                continue

    def _dense_one(self, query: str, source: SourceType) -> list[Hit]:
        collection_name, top_k = self._source_meta[source]
        collection = self.collections.get(collection_name)
        if collection is None:
            return []

        qvec = self.embedder.encode_query(query).tolist()
        results = collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        ids_list = results.get("ids", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        out: list[Hit] = []
        for chroma_id, metadata, distance in zip(ids_list, metadatas_list, distances_list):
            metadata = metadata or {}
            similarity = 1.0 - float(distance)  # Chroma cosine: [0,2] → [-1,1]
            out.append(
                Hit(
                    id=metadata.get("doc_id") or chroma_id,
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    source=metadata.get("source", source),
                    answered_at=parse_date(metadata.get("answered_at")),
                    answered_by=metadata.get("answered_by") or None,
                    score=similarity,
                )
            )
        return out

    def _sparse_one(self, query: str, source: SourceType, top_k: int) -> list[Hit]:
        # 4배 over-fetch: source 필터 후 top_k 보장. 부족 시 늘려도 BM25는 O(N)이라 영향 미미.
        raw = self.bm25.search(query, top_k=top_k * 4)
        hits: list[Hit] = []
        for idx, score in raw:
            if score <= 0:
                continue
            rec = self.bm25.records[idx]
            if rec.source != source:
                continue
            hits.append(rec.model_copy(update={"score": float(score)}))
            if len(hits) >= top_k:
                break
        return hits

    def _rrf_merge(self, dense: list[Hit], sparse: list[Hit]) -> list[Hit]:
        k = settings.rrf_k
        scores: dict[str, float] = defaultdict(float)
        hits: dict[str, Hit] = {}

        for ranked_list in (dense, sparse):
            for rank, hit in enumerate(ranked_list):
                scores[hit.id] += 1.0 / (k + rank + 1)
                hits.setdefault(hit.id, hit)

        merged = [hits[hid].model_copy(update={"score": s}) for hid, s in scores.items()]
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged

    def _rerank(self, query: str, candidates: list[Hit]) -> list[Hit]:
        if not candidates:
            return []
        head = candidates[: settings.rerank_top_k]
        scores = self.reranker.score(query, [h.question for h in head])
        reranked = [
            head[i].model_copy(update={"score": float(scores[i])})
            for i in range(len(head))
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[: settings.final_top_k]

    def search_by_source(self, query: str, source: SourceType) -> list[Hit]:
        """단일 소스 검색 (RRF + reranker, 임계치 미적용)."""
        _, top_k = self._source_meta[source]
        dense = self._dense_one(query, source)
        sparse = self._sparse_one(query, source, top_k=top_k)
        merged = self._rrf_merge(dense, sparse)
        return self._rerank(query, merged)

    def search(self, query: str) -> list[Hit]:
        """레거시: 두 소스 합쳐 reranker score 순 (eval 백워드 호환 전용).

        chat.py는 search_by_source 사용 (tiered 흐름).
        """
        merged = self.search_by_source(query, "official") + self.search_by_source(query, "history")
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[: settings.final_top_k]


@lru_cache(maxsize=1)
def get_searcher() -> Searcher:
    return Searcher()


def reset_searcher() -> None:
    get_searcher.cache_clear()


def build_bm25_index(records: Iterable[Hit]) -> BM25Index:
    records_list = list(records)
    tokenized = [tokenize(r.question) for r in records_list]
    return BM25Index(bm25=BM25Okapi(tokenized), records=records_list)
